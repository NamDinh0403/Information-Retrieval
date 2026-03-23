"""
Token Pruning Module
====================
Kỹ thuật cắt tỉa token cho Vision Transformer.

Implementations:
    - TokenPruner: V-Pruner dựa trên Fisher Information
    - AttentionBasedPruner: Cắt tỉa dựa trên attention scores

References:
    - V-Pruner: Fisher information + PPO cho quyết định cắt tỉa toàn cục
    - ATPViT: Attention-based token pruning, giảm 47% FLOPs
    - AdaptiVision: Phân cụm token động với soft k-means
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List
import numpy as np


class TokenPruner:
    """
    V-Pruner Implementation: Cắt tỉa token dựa trên Fisher Information.
    
    Fisher Information đo lường độ nhạy của loss đối với mask variables,
    giúp xác định token nào quan trọng nhất.
    
    Args:
        keep_ratio: Tỷ lệ token giữ lại (0.5 = giữ 50%)
        min_tokens: Số token tối thiểu phải giữ
        
    Example:
        >>> pruner = TokenPruner(keep_ratio=0.7)
        >>> scores = pruner.compute_fisher_scores(features, model, criterion)
        >>> pruned = pruner.prune_tokens(features, scores)
    """
    
    def __init__(
        self, 
        keep_ratio: float = 0.7,
        min_tokens: int = 16
    ):
        self.keep_ratio = keep_ratio
        self.min_tokens = min_tokens
    
    def compute_fisher_scores(
        self, 
        features: torch.Tensor,
        gradients: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Tính Fisher Information scores cho mỗi token.
        
        Fisher Score: F_i = E[(∂L/∂m_i)²] ≈ gradient²
        
        Args:
            features: Token features [B, N, D]
            gradients: Gradients của loss w.r.t features (optional)
            
        Returns:
            fisher_scores: Importance scores [B, N]
        """
        if gradients is None:
            # Fallback: sử dụng L2 norm của features như proxy
            fisher_scores = torch.norm(features, dim=-1)
        else:
            # Fisher = E[gradient²]
            fisher_scores = torch.pow(gradients, 2).mean(dim=-1)
        
        return fisher_scores
    
    def compute_attention_scores(
        self,
        attention_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Tính importance scores từ attention weights.
        
        Args:
            attention_weights: Attention từ CLS token [B, num_heads, N]
            
        Returns:
            attention_scores: [B, N]
        """
        # Trung bình qua các heads
        scores = attention_weights.mean(dim=1)
        return scores
    
    def prune_tokens(
        self, 
        features: torch.Tensor, 
        importance_scores: torch.Tensor,
        keep_cls: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cắt tỉa tokens dựa trên importance scores.
        
        Args:
            features: [B, N, D] - Token features
            importance_scores: [B, N] - Importance scores
            keep_cls: Giữ CLS token (index 0)
            
        Returns:
            pruned_features: [B, K, D] với K = keep_ratio * N
            kept_indices: [B, K] - Indices của tokens được giữ
        """
        B, N, D = features.shape
        num_keep = max(int(N * self.keep_ratio), self.min_tokens)
        
        if keep_cls:
            # Đảm bảo CLS token luôn được giữ
            # Tách CLS và patch tokens
            cls_token = features[:, :1, :]
            patch_tokens = features[:, 1:, :]
            patch_scores = importance_scores[:, 1:]
            
            # Lấy top-k patch tokens
            num_keep_patches = num_keep - 1
            _, topk_indices = torch.topk(patch_scores, num_keep_patches, dim=1)
            
            # Gather pruned patches
            topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, D)
            pruned_patches = torch.gather(patch_tokens, 1, topk_indices_expanded)
            
            # Concatenate CLS và pruned patches
            pruned_features = torch.cat([cls_token, pruned_patches], dim=1)
            kept_indices = torch.cat([
                torch.zeros(B, 1, dtype=torch.long, device=features.device),
                topk_indices + 1
            ], dim=1)
        else:
            _, topk_indices = torch.topk(importance_scores, num_keep, dim=1)
            topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, D)
            pruned_features = torch.gather(features, 1, topk_indices_expanded)
            kept_indices = topk_indices
        
        return pruned_features, kept_indices
    
    def apply_training_free_pruning(
        self,
        features: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Training-free pruning pipeline (Tuần 2).
        Không cần backward pass, sử dụng heuristics.
        
        Returns:
            pruned_features: Features sau cắt tỉa
            stats: Thống kê về quá trình cắt tỉa
        """
        B, N, D = features.shape
        
        # Tính scores bằng L2 norm (training-free heuristic)
        scores = self.compute_fisher_scores(features)
        
        # Cắt tỉa
        pruned, indices = self.prune_tokens(features, scores)
        
        stats = {
            'original_tokens': N,
            'kept_tokens': pruned.shape[1],
            'pruned_tokens': N - pruned.shape[1],
            'compression_ratio': pruned.shape[1] / N,
            'top_score': scores.max().item(),
            'min_kept_score': scores.gather(1, indices).min().item()
        }
        
        return pruned, stats


class AttentionBasedPruner(nn.Module):
    """
    ATPViT-style: Cắt tỉa tích hợp trong Attention layer.
    
    Học một prediction network để quyết định pruning mask
    trong quá trình training.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_tokens: int = 196,
        keep_ratio: float = 0.7
    ):
        super().__init__()
        self.keep_ratio = keep_ratio
        self.num_tokens = num_tokens
        
        # Prediction network cho pruning decision
        self.score_predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 1)
        )
    
    def forward(
        self, 
        x: torch.Tensor,
        return_mask: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward với learnable pruning.
        
        Args:
            x: [B, N, D] token features
            return_mask: Trả về pruning mask
            
        Returns:
            pruned_x: Pruned features
            mask: Pruning mask (optional)
        """
        B, N, D = x.shape
        num_keep = int(N * self.keep_ratio)
        
        # Predict scores
        scores = self.score_predictor(x).squeeze(-1)  # [B, N]
        
        # Straight-through estimator cho differentiable top-k
        if self.training:
            # Gumbel-softmax trick cho differentiable selection
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(scores) + 1e-8) + 1e-8)
            perturbed_scores = scores + gumbel_noise
        else:
            perturbed_scores = scores
        
        # Top-k selection
        _, indices = torch.topk(perturbed_scores, num_keep, dim=1)
        indices = indices.sort(dim=1).values  # Giữ thứ tự spatial
        
        # Gather selected tokens
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, D)
        pruned_x = torch.gather(x, 1, indices_expanded)
        
        if return_mask:
            mask = torch.zeros(B, N, device=x.device)
            mask.scatter_(1, indices, 1.0)
            return pruned_x, mask
        
        return pruned_x, None


class TokenMerger:
    """
    AdaptiVision-style: Phân cụm và ngưng tụ token.
    
    Sử dụng soft k-means để gộp các token tương tự
    thành "super-tokens", bảo toàn thông tin tốt hơn.
    """
    
    def __init__(
        self,
        num_clusters: int = 49,
        num_iterations: int = 3
    ):
        self.num_clusters = num_clusters
        self.num_iterations = num_iterations
    
    def soft_kmeans(
        self,
        features: torch.Tensor,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Soft K-means clustering cho token merging.
        
        Args:
            features: [B, N, D]
            temperature: Softmax temperature
            
        Returns:
            centroids: [B, K, D] cluster centers (super-tokens)
            assignments: [B, N, K] soft assignments
        """
        B, N, D = features.shape
        K = self.num_clusters
        
        # Initialize centroids (uniform sampling)
        indices = torch.linspace(0, N-1, K).long()
        centroids = features[:, indices, :]  # [B, K, D]
        
        for _ in range(self.num_iterations):
            # Compute distances
            # [B, N, D] @ [B, D, K] -> [B, N, K]
            distances = torch.cdist(features, centroids)
            
            # Soft assignments
            assignments = F.softmax(-distances / temperature, dim=-1)  # [B, N, K]
            
            # Update centroids
            # [B, K, N] @ [B, N, D] -> [B, K, D]
            weighted_sum = torch.bmm(assignments.transpose(1, 2), features)
            cluster_sizes = assignments.sum(dim=1, keepdim=True).transpose(1, 2)
            centroids = weighted_sum / (cluster_sizes + 1e-8)
        
        return centroids, assignments
    
    def merge_tokens(self, features: torch.Tensor) -> torch.Tensor:
        """Merge tokens thành super-tokens."""
        centroids, _ = self.soft_kmeans(features)
        return centroids


def analyze_pruning_effect(
    original_features: torch.Tensor,
    pruned_features: torch.Tensor
) -> Dict:
    """
    Phân tích ảnh hưởng của pruning lên feature distribution.
    """
    orig_mean = original_features.mean().item()
    orig_std = original_features.std().item()
    
    pruned_mean = pruned_features.mean().item()
    pruned_std = pruned_features.std().item()
    
    # Information preservation (cosine similarity)
    orig_flat = original_features.mean(dim=1)
    pruned_flat = pruned_features.mean(dim=1)
    
    cos_sim = F.cosine_similarity(orig_flat, pruned_flat, dim=-1).mean().item()
    
    return {
        'original_mean': orig_mean,
        'original_std': orig_std,
        'pruned_mean': pruned_mean,
        'pruned_std': pruned_std,
        'information_preservation': cos_sim,
        'compression_ratio': pruned_features.shape[1] / original_features.shape[1]
    }


if __name__ == "__main__":
    print("Testing Token Pruning Module...")
    
    # Test TokenPruner
    pruner = TokenPruner(keep_ratio=0.5)
    features = torch.randn(2, 197, 384)  # [B, N, D] with CLS
    
    pruned, stats = pruner.apply_training_free_pruning(features)
    print(f"\nTokenPruner Results:")
    print(f"  Original: {stats['original_tokens']} tokens")
    print(f"  Kept: {stats['kept_tokens']} tokens")
    print(f"  Compression: {stats['compression_ratio']:.2%}")
    
    # Test AttentionBasedPruner
    attn_pruner = AttentionBasedPruner(embed_dim=384, keep_ratio=0.7)
    attn_pruned, mask = attn_pruner(features, return_mask=True)
    print(f"\nAttentionBasedPruner Results:")
    print(f"  Output shape: {attn_pruned.shape}")
    
    # Test TokenMerger
    merger = TokenMerger(num_clusters=49)
    merged = merger.merge_tokens(features[:, 1:, :])  # Exclude CLS
    print(f"\nTokenMerger Results:")
    print(f"  Super-tokens shape: {merged.shape}")
    
    # Analyze effect
    analysis = analyze_pruning_effect(features, pruned)
    print(f"\nPruning Analysis:")
    print(f"  Information preservation: {analysis['information_preservation']:.4f}")
