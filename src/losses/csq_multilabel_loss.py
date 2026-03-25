"""
Multi-Label CSQ Loss for NUS-WIDE Dataset
==========================================

Modified Central Similarity Quantization (CSQ) Loss for multi-label setting.

Key differences from single-label CSQ:
1. Similarity is computed based on label overlap (Jaccard or cosine)
2. Pairwise loss instead of center-based loss
3. Hash codes should be similar if images share labels

Reference:
    - CSQ: Yuan et al. "Central Similarity Quantization for Efficient Image and Video Retrieval" (CVPR 2020)
    - HashNet: Cao et al. "HashNet: Deep Learning to Hash by Continuation" (ICCV 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MultiLabelCSQLoss(nn.Module):
    """
    Multi-Label CSQ Loss combining:
    1. Pairwise similarity loss (similar images -> similar hash codes)
    2. Quantization loss (push continuous values to ±1)
    
    For multi-label, two images are "similar" if they share at least one label.
    """
    
    def __init__(
        self,
        hash_bit: int,
        num_classes: int,
        lambda_q: float = 0.0001,
        margin: float = 0.5,
        similarity_type: str = 'cosine',  # 'cosine', 'jaccard', 'overlap'
    ):
        """
        Args:
            hash_bit: Number of hash bits
            num_classes: Number of label classes (e.g., 21 for NUS-WIDE)
            lambda_q: Weight for quantization loss
            margin: Margin for dissimilar pairs
            similarity_type: How to compute label similarity
        """
        super().__init__()
        self.hash_bit = hash_bit
        self.num_classes = num_classes
        self.lambda_q = lambda_q
        self.margin = margin
        self.similarity_type = similarity_type
        
        # Learnable class-wise hash centers (optional, for hybrid approach)
        self.register_buffer(
            'hash_centers',
            torch.sign(torch.randn(num_classes, hash_bit))
        )
    
    def _compute_label_similarity(
        self, 
        labels1: torch.Tensor, 
        labels2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute similarity between multi-label vectors.
        
        Args:
            labels1: [B1, C] multi-hot labels
            labels2: [B2, C] multi-hot labels
            
        Returns:
            similarity: [B1, B2] similarity matrix in [0, 1]
        """
        if self.similarity_type == 'cosine':
            # Cosine similarity
            labels1_norm = F.normalize(labels1.float(), dim=1)
            labels2_norm = F.normalize(labels2.float(), dim=1)
            sim = torch.mm(labels1_norm, labels2_norm.t())
            
        elif self.similarity_type == 'jaccard':
            # Jaccard similarity: |A ∩ B| / |A ∪ B|
            intersection = torch.mm(labels1.float(), labels2.float().t())
            sum1 = labels1.sum(dim=1, keepdim=True)  # [B1, 1]
            sum2 = labels2.sum(dim=1, keepdim=True)  # [B2, 1]
            union = sum1 + sum2.t() - intersection
            sim = intersection / (union + 1e-8)
            
        else:  # 'overlap' - binary: similar if share at least one label
            # |A ∩ B| > 0
            intersection = torch.mm(labels1.float(), labels2.float().t())
            sim = (intersection > 0).float()
        
        return sim
    
    def forward(
        self, 
        hash_outputs: torch.Tensor,
        labels: torch.Tensor,
        hash_outputs_2: Optional[torch.Tensor] = None,
        labels_2: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute multi-label CSQ loss.
        
        Args:
            hash_outputs: [B, hash_bit] continuous hash outputs in (-1, 1)
            labels: [B, num_classes] multi-hot labels
            hash_outputs_2: Optional second batch for asymmetric loss
            labels_2: Optional labels for second batch
            
        Returns:
            total_loss: Scalar loss value
        """
        if hash_outputs_2 is None:
            hash_outputs_2 = hash_outputs
            labels_2 = labels
        
        # Normalize hash outputs
        h1 = hash_outputs  # [B1, K]
        h2 = hash_outputs_2  # [B2, K]
        
        # Compute label similarity matrix
        S = self._compute_label_similarity(labels, labels_2)  # [B1, B2]
        
        # Compute hash similarity (inner product scaled to [-1, 1])
        hash_sim = torch.mm(h1, h2.t()) / self.hash_bit  # [B1, B2]
        
        # Pairwise loss: maximize similarity for similar pairs, minimize for dissimilar
        # Using log-cosh loss for smooth optimization
        # For similar pairs (S > 0): hash_sim should be high (close to S)
        # For dissimilar pairs (S = 0): hash_sim should be low
        
        # Weighted loss based on label similarity
        pos_mask = (S > 0).float()
        neg_mask = (S == 0).float()
        
        # Similar pairs: minimize distance
        pos_loss = pos_mask * (1 - hash_sim).pow(2)
        
        # Dissimilar pairs: push apart with margin
        neg_loss = neg_mask * F.relu(hash_sim + self.margin).pow(2)
        
        # Balance positive and negative
        num_pos = pos_mask.sum() + 1e-8
        num_neg = neg_mask.sum() + 1e-8
        
        similarity_loss = (pos_loss.sum() / num_pos + neg_loss.sum() / num_neg)
        
        # Quantization loss: push values to ±1
        q_loss_1 = torch.mean((torch.abs(h1) - 1.0).pow(2))
        q_loss_2 = torch.mean((torch.abs(h2) - 1.0).pow(2))
        q_loss = (q_loss_1 + q_loss_2) / 2
        
        # Total loss
        total_loss = similarity_loss + self.lambda_q * q_loss
        
        return total_loss


class MultiLabelDCHLoss(nn.Module):
    """
    Deep Cauchy Hashing (DCH) Loss for Multi-Label.
    
    Uses Cauchy distribution for better gradient flow.
    
    Reference:
        Cao et al. "Deep Cauchy Hashing for Hamming Space Retrieval" (CVPR 2018)
    """
    
    def __init__(
        self,
        hash_bit: int,
        num_classes: int,
        gamma: float = 20.0,
        lambda_q: float = 0.0001,
    ):
        super().__init__()
        self.hash_bit = hash_bit
        self.num_classes = num_classes
        self.gamma = gamma
        self.lambda_q = lambda_q
    
    def forward(
        self,
        hash_outputs: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hash_outputs: [B, hash_bit]
            labels: [B, num_classes] multi-hot
        """
        batch_size = hash_outputs.size(0)
        
        # Compute similarity from labels (any overlap = similar)
        S = torch.mm(labels.float(), labels.float().t()) > 0
        S = S.float() * 2 - 1  # Convert to {-1, 1}
        
        # Compute Hamming distance (approximated by inner product)
        dist = (self.hash_bit - torch.mm(hash_outputs, hash_outputs.t())) / 2
        
        # Cauchy distribution based likelihood
        cauchy = self.gamma / (dist + self.gamma)
        
        # Binary cross entropy with Cauchy likelihood
        # Similar pairs (S=1): maximize cauchy -> log(cauchy)
        # Dissimilar pairs (S=-1): minimize cauchy -> log(1-cauchy)
        pos_mask = (S > 0).float()
        neg_mask = (S < 0).float()
        
        loss = -pos_mask * torch.log(cauchy + 1e-8) - neg_mask * torch.log(1 - cauchy + 1e-8)
        
        # Remove diagonal (self-similarity)
        mask = 1 - torch.eye(batch_size, device=hash_outputs.device)
        loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        
        # Quantization loss
        q_loss = torch.mean((torch.abs(hash_outputs) - 1.0).pow(2))
        
        return loss + self.lambda_q * q_loss


class MultiLabelHashNetLoss(nn.Module):
    """
    HashNet Loss for Multi-Label.
    
    Uses continuation method with weighted sigmoid.
    
    Reference:
        Cao et al. "HashNet: Deep Learning to Hash by Continuation" (ICCV 2017)
    """
    
    def __init__(
        self,
        hash_bit: int,
        num_classes: int,
        alpha: float = 1.0,  # Weight balance (updated during training)
        lambda_q: float = 0.0001,
    ):
        super().__init__()
        self.hash_bit = hash_bit
        self.num_classes = num_classes
        self.alpha = alpha
        self.lambda_q = lambda_q
    
    def forward(
        self,
        hash_outputs: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hash_outputs: [B, hash_bit]
            labels: [B, num_classes] multi-hot
        """
        # Compute pairwise similarity from labels
        S = torch.mm(labels.float(), labels.float().t())
        S = (S > 0).float()  # Binary: similar if any label overlap
        
        # Hash similarity (inner product)
        theta = torch.mm(hash_outputs, hash_outputs.t()) / 2
        
        # Weighted sigmoid loss
        # Similar: S=1 -> maximize sigmoid(theta)
        # Dissimilar: S=0 -> minimize sigmoid(theta)
        
        # Class imbalance weight
        num_pos = S.sum()
        num_neg = S.numel() - num_pos
        weight = num_neg / (num_pos + 1e-8)
        
        # Binary cross entropy
        loss = S * weight * torch.log(1 + torch.exp(-theta)) + \
               (1 - S) * torch.log(1 + torch.exp(theta))
        
        # Remove diagonal
        batch_size = hash_outputs.size(0)
        mask = 1 - torch.eye(batch_size, device=hash_outputs.device)
        loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        
        # Quantization loss
        q_loss = torch.mean((torch.abs(hash_outputs) - 1.0).pow(2))
        
        return loss + self.lambda_q * q_loss


def get_multilabel_loss(
    loss_type: str,
    hash_bit: int,
    num_classes: int,
    **kwargs
) -> nn.Module:
    """
    Factory function to get multi-label loss.
    
    Args:
        loss_type: 'csq', 'dch', 'hashnet'
        hash_bit: Number of hash bits
        num_classes: Number of label classes
        
    Returns:
        Loss module
    """
    losses = {
        'csq': MultiLabelCSQLoss,
        'dch': MultiLabelDCHLoss,
        'hashnet': MultiLabelHashNetLoss,
    }
    
    if loss_type not in losses:
        raise ValueError(f"Unknown loss type: {loss_type}. Choose from {list(losses.keys())}")
    
    return losses[loss_type](hash_bit=hash_bit, num_classes=num_classes, **kwargs)


if __name__ == '__main__':
    # Test losses
    batch_size = 8
    hash_bit = 64
    num_classes = 21
    
    # Random inputs
    hash_outputs = torch.tanh(torch.randn(batch_size, hash_bit))
    labels = torch.randint(0, 2, (batch_size, num_classes)).float()
    
    print("Testing Multi-Label Losses")
    print("=" * 50)
    
    for loss_type in ['csq', 'dch', 'hashnet']:
        loss_fn = get_multilabel_loss(loss_type, hash_bit, num_classes)
        loss = loss_fn(hash_outputs, labels)
        print(f"{loss_type.upper()} Loss: {loss.item():.4f}")
