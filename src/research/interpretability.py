"""
Interpretability Module
=======================
Phân tích giải thích nhân quả cho Vision Transformer.

Implementations:
    - ViTCXAnalyzer: Causal Explanation dựa trên patch masking
    - AttentionRollout: Tích lũy attention qua các layers

References:
    - ViT-CX: Tập trung vào patch embeddings thay vì attention weights
    - IA-ViT: Interpretable Architecture với explainer module
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np


class ViTCXAnalyzer:
    """
    ViT-CX (Causal Explanation) Analyzer.
    
    Phân tích tại sao model đưa ra quyết định bằng cách:
    1. Trích xuất features từ target layer
    2. Phân cụm patches tương đồng
    3. Mask từng cụm và đo impact
    
    Args:
        model: Vision Transformer model
        patch_size: Kích thước patch (default: 14 cho DINOv2)
        
    Example:
        >>> analyzer = ViTCXAnalyzer(model)
        >>> saliency = analyzer.compute_saliency(image, target_class=5)
        >>> analyzer.visualize(image, saliency)
    """
    
    def __init__(
        self,
        model: nn.Module,
        patch_size: int = 14,
        device: str = 'cpu'
    ):
        self.model = model
        self.patch_size = patch_size
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def _get_patch_grid_size(self, image: torch.Tensor) -> Tuple[int, int]:
        """Tính số patches theo H và W."""
        _, _, H, W = image.shape
        return H // self.patch_size, W // self.patch_size
    
    def compute_occlusion_sensitivity(
        self,
        image: torch.Tensor,
        target_idx: int = 0,
        mask_value: float = 0.0
    ) -> torch.Tensor:
        """
        Occlusion sensitivity: Che từng patch và đo output change.
        
        Args:
            image: [C, H, W] hoặc [1, C, H, W]
            target_idx: Index của output dimension cần phân tích
            mask_value: Giá trị để mask (0 = black, mean = gray)
            
        Returns:
            saliency_map: [H_patches, W_patches] importance map
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        image = image.to(self.device)
        H_p, W_p = self._get_patch_grid_size(image)
        
        # Original prediction
        with torch.no_grad():
            original_output, _ = self.model(image)
            original_score = original_output[0, target_idx]
        
        # Saliency map
        saliency = torch.zeros(H_p, W_p, device=self.device)
        
        for i in range(H_p):
            for j in range(W_p):
                # Create masked image
                masked = image.clone()
                y1, y2 = i * self.patch_size, (i + 1) * self.patch_size
                x1, x2 = j * self.patch_size, (j + 1) * self.patch_size
                masked[0, :, y1:y2, x1:x2] = mask_value
                
                # Get masked prediction
                with torch.no_grad():
                    masked_output, _ = self.model(masked)
                    masked_score = masked_output[0, target_idx]
                
                # Impact = drop in score when patch is masked
                impact = original_score - masked_score
                saliency[i, j] = impact.item()
        
        # Normalize
        saliency = saliency - saliency.min()
        if saliency.max() > 0:
            saliency = saliency / saliency.max()
        
        return saliency.cpu()
    
    def compute_gradient_saliency(
        self,
        image: torch.Tensor,
        target_idx: int = 0
    ) -> torch.Tensor:
        """
        Gradient-based saliency: Tính gradient của output w.r.t input.
        Nhanh hơn occlusion nhưng ít interpretable hơn.
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        image = image.to(self.device)
        image.requires_grad_(True)
        
        # Forward
        output, _ = self.model(image)
        score = output[0, target_idx]
        
        # Backward
        score.backward()
        
        # Get gradients
        gradients = image.grad[0]  # [C, H, W]
        
        # Compute saliency per pixel
        saliency = gradients.abs().mean(dim=0)  # [H, W]
        
        # Average to patch level
        H_p, W_p = self._get_patch_grid_size(image)
        patch_saliency = torch.zeros(H_p, W_p)
        
        for i in range(H_p):
            for j in range(W_p):
                y1, y2 = i * self.patch_size, (i + 1) * self.patch_size
                x1, x2 = j * self.patch_size, (j + 1) * self.patch_size
                patch_saliency[i, j] = saliency[y1:y2, x1:x2].mean()
        
        # Normalize
        patch_saliency = patch_saliency - patch_saliency.min()
        if patch_saliency.max() > 0:
            patch_saliency = patch_saliency / patch_saliency.max()
        
        return patch_saliency.cpu().detach()
    
    def compute_causal_saliency(
        self,
        image: torch.Tensor,
        target_idx: int = 0,
        num_samples: int = 100,
        mask_ratio: float = 0.5
    ) -> torch.Tensor:
        """
        Causal saliency với random masking (SHAP-like).
        
        Ước tính importance của mỗi patch bằng cách:
        - Random mask subset patches
        - Measure contribution của mỗi patch
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        image = image.to(self.device)
        H_p, W_p = self._get_patch_grid_size(image)
        total_patches = H_p * W_p
        
        # Track contributions
        contributions = torch.zeros(total_patches, device=self.device)
        counts = torch.zeros(total_patches, device=self.device)
        
        # Baseline (fully masked)
        with torch.no_grad():
            baseline_output, _ = self.model(torch.zeros_like(image))
            baseline_score = baseline_output[0, target_idx]
        
        # Sample random masks
        for _ in range(num_samples):
            # Random mask
            mask = torch.rand(total_patches) > mask_ratio
            
            # Create masked image
            masked = image.clone()
            for idx in range(total_patches):
                if not mask[idx]:
                    i, j = idx // W_p, idx % W_p
                    y1, y2 = i * self.patch_size, (i + 1) * self.patch_size
                    x1, x2 = j * self.patch_size, (j + 1) * self.patch_size
                    masked[0, :, y1:y2, x1:x2] = 0
            
            # Get score
            with torch.no_grad():
                output, _ = self.model(masked)
                score = output[0, target_idx]
            
            # Marginal contribution
            marginal = score - baseline_score
            
            # Attribute to each visible patch
            for idx in range(total_patches):
                if mask[idx]:
                    contributions[idx] += marginal
                    counts[idx] += 1
        
        # Average contributions
        contributions = contributions / (counts + 1e-8)
        
        # Reshape to grid
        saliency = contributions.view(H_p, W_p)
        
        # Normalize
        saliency = saliency - saliency.min()
        if saliency.max() > 0:
            saliency = saliency / saliency.max()
        
        return saliency.cpu()
    
    def analyze_pruning_impact(
        self,
        image: torch.Tensor,
        kept_indices: torch.Tensor,
        target_idx: int = 0
    ) -> Dict:
        """
        Phân tích xem pruning có bỏ qua vùng quan trọng không.
        
        Returns:
            Dict với metrics về overlap giữa pruned tokens và saliency
        """
        # Compute saliency
        saliency = self.compute_gradient_saliency(image, target_idx)
        H_p, W_p = saliency.shape
        
        # Convert kept_indices to mask
        mask = torch.zeros(H_p * W_p)
        mask[kept_indices.flatten()] = 1
        mask = mask.view(H_p, W_p)
        
        # Compute overlap
        high_saliency = saliency > 0.5  # Important regions
        kept_important = (mask * high_saliency).sum()
        total_important = high_saliency.sum()
        
        overlap = kept_important / (total_important + 1e-8)
        
        return {
            'saliency_coverage': overlap.item(),
            'important_kept': kept_important.item(),
            'total_important': total_important.item(),
            'saliency_map': saliency
        }


class AttentionRollout:
    """
    Attention Rollout: Tích lũy attention qua các layers.
    
    Visualize attention flow từ CLS token đến các patches
    để hiểu model focus vào đâu.
    """
    
    def __init__(self, model: nn.Module, head_fusion: str = 'mean'):
        """
        Args:
            model: ViT model
            head_fusion: 'mean', 'max', hoặc 'min' để combine heads
        """
        self.model = model
        self.head_fusion = head_fusion
        self.attentions = []
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Đăng ký hooks để capture attention weights."""
        def hook_fn(module, input, output):
            # output[1] là attention weights trong timm
            if isinstance(output, tuple) and len(output) > 1:
                self.attentions.append(output[1])
        
        for name, module in self.model.named_modules():
            if 'attn' in name.lower() and hasattr(module, 'qkv'):
                module.register_forward_hook(hook_fn)
    
    def rollout(
        self,
        image: torch.Tensor,
        discard_ratio: float = 0.0
    ) -> torch.Tensor:
        """
        Compute attention rollout.
        
        Args:
            image: Input image [B, C, H, W]
            discard_ratio: Tỷ lệ attention values nhỏ nhất bị bỏ qua
            
        Returns:
            rollout_attention: [num_patches] attention từ CLS
        """
        self.attentions = []
        
        # Forward pass
        with torch.no_grad():
            _ = self.model(image)
        
        if not self.attentions:
            print("[Warning] No attention captured. Model may not be compatible.")
            return None
        
        # Process attentions
        # Each attention: [B, num_heads, N, N]
        result = None
        
        for attention in self.attentions:
            # Fuse heads
            if self.head_fusion == 'mean':
                attention_heads_fused = attention.mean(dim=1)
            elif self.head_fusion == 'max':
                attention_heads_fused = attention.max(dim=1)[0]
            elif self.head_fusion == 'min':
                attention_heads_fused = attention.min(dim=1)[0]
            else:
                raise ValueError(f"Unknown head fusion: {self.head_fusion}")
            
            # Add residual connection
            I = torch.eye(attention_heads_fused.size(-1), device=attention.device)
            attention_heads_fused = (attention_heads_fused + I) / 2
            
            # Normalize
            attention_heads_fused = attention_heads_fused / attention_heads_fused.sum(dim=-1, keepdim=True)
            
            # Rollout
            if result is None:
                result = attention_heads_fused
            else:
                result = torch.bmm(attention_heads_fused, result)
        
        # Extract CLS attention
        if result is not None:
            mask = result[0, 0, 1:]  # Attention từ CLS token đến patches
            return mask.cpu()
        
        return None
    
    def visualize_attention(
        self,
        image: torch.Tensor,
        patch_size: int = 14
    ) -> np.ndarray:
        """
        Tạo attention map có cùng kích thước với ảnh gốc.
        """
        attention = self.rollout(image)
        
        if attention is None:
            return None
        
        # Reshape to grid
        num_patches = attention.shape[0]
        grid_size = int(num_patches ** 0.5)
        attention_map = attention.view(grid_size, grid_size)
        
        # Upsample to image size
        _, _, H, W = image.shape
        attention_map = F.interpolate(
            attention_map.unsqueeze(0).unsqueeze(0),
            size=(H, W),
            mode='bilinear',
            align_corners=False
        ).squeeze().numpy()
        
        return attention_map


def compare_interpretations(
    model: nn.Module,
    image: torch.Tensor,
    methods: List[str] = ['gradient', 'occlusion']
) -> Dict[str, torch.Tensor]:
    """
    So sánh các phương pháp interpretability khác nhau.
    """
    results = {}
    
    analyzer = ViTCXAnalyzer(model)
    
    if 'gradient' in methods:
        results['gradient'] = analyzer.compute_gradient_saliency(image)
    
    if 'occlusion' in methods:
        results['occlusion'] = analyzer.compute_occlusion_sensitivity(image)
    
    if 'causal' in methods:
        results['causal'] = analyzer.compute_causal_saliency(image, num_samples=50)
    
    return results


if __name__ == "__main__":
    import timm
    
    print("Testing Interpretability Module...")
    
    # Create model
    model = timm.create_model(
        'vit_small_patch14_dinov2.lvd142m',
        pretrained=False,
        num_classes=0,
        dynamic_img_size=True
    )
    
    # Wrap with simple hash head for testing
    class SimpleModel(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone
            self.head = nn.Linear(384, 64)
        
        def forward(self, x):
            features = self.backbone(x)
            return self.head(features), features
    
    model = SimpleModel(model)
    
    # Test analyzer
    analyzer = ViTCXAnalyzer(model, patch_size=14)
    
    image = torch.randn(3, 224, 224)
    
    print("\nComputing gradient saliency...")
    grad_saliency = analyzer.compute_gradient_saliency(image)
    print(f"  Saliency shape: {grad_saliency.shape}")
    print(f"  Max importance: {grad_saliency.max():.4f}")
    
    print("\nModule test complete!")
