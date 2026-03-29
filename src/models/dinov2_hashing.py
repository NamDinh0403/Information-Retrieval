"""
DINOv3 Hashing Model
====================
Mô hình kết hợp DINOv2/v3 backbone với Hashing Head cho Content-Based Image Retrieval.

References:
    - DINOv3 (Meta AI, 08/2025): 7B parameter foundation model using Gram Anchoring
    - VTS (Vision Transformer for Hashing): Sử dụng ViT để capture global dependencies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Optional, Tuple, List


class HashingHead(nn.Module):
    """
    Hashing Head cho Vision Transformer.
    Chuyển đổi embedding thành mã hash nhị phân.
    
    Architecture:
        Input -> Dropout -> Dense(1024) -> ReLU -> Dense(hash_bit) -> Tanh
    """
    
    def __init__(
        self, 
        embed_dim: int, 
        hash_bit: int = 64,
        hidden_dim: int = 1024,
        dropout: float = 0.5
    ):
        super().__init__()
        self.hash_bit = hash_bit
        
        self.layers = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hash_bit),
            nn.Tanh()  # Output trong khoảng [-1, 1]
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Khởi tạo weights theo Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
    
    def get_binary_codes(self, x: torch.Tensor) -> torch.Tensor:
        """Chuyển đổi sang mã nhị phân {-1, 1} trong inference."""
        continuous = self.forward(x)
        return torch.sign(continuous)


class DINOv3Hashing(nn.Module):
    """
    DINOv3 Hashing Model cho Image Retrieval.
    
    Sử dụng DINOv2 làm proxy cho DINOv3 trong nghiên cứu.
    Hỗ trợ nhiều biến thể: ViT-S/14, ViT-B/14, ViT-L/14.
    
    Args:
        model_name: Tên model từ timm (default: vit_small_patch14_dinov2)
        pretrained: Có sử dụng pretrained weights không
        hash_bit: Số bit cho mã hash (16, 32, 64, 128)
        freeze_backbone: Đóng băng backbone trong fine-tuning
        use_gram_anchoring: Áp dụng Gram Anchoring (simulated)
    
    Example:
        >>> model = DINOv3Hashing(hash_bit=64)
        >>> hash_codes, features = model(images)
    """
    
    # Supported models với thông số tương ứng
    SUPPORTED_MODELS = {
        'vit_small_patch14_dinov2.lvd142m': {'embed_dim': 384, 'default_img_size': 518},
        'vit_base_patch14_dinov2.lvd142m': {'embed_dim': 768, 'default_img_size': 518},
        'vit_large_patch14_dinov2.lvd142m': {'embed_dim': 1024, 'default_img_size': 518},
        'vit_base_patch16_224': {'embed_dim': 768, 'default_img_size': 224},
    }
    
    def __init__(
        self,
        model_name: str = 'vit_small_patch14_dinov2.lvd142m',
        pretrained: bool = True,
        hash_bit: int = 64,
        freeze_backbone: bool = False,
        use_gram_anchoring: bool = False,
        hidden_dim: int = None,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.model_name = model_name
        self.hash_bit = hash_bit
        self.use_gram_anchoring = use_gram_anchoring
        
        # Load backbone từ timm
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            num_classes=0,  # Loại bỏ classification head
            dynamic_img_size=True  # Hỗ trợ nhiều kích thước ảnh
        )
        
        embed_dim = self.backbone.num_features
        
        # Auto-scale hidden_dim: avoid explosion for small embed_dim (e.g. 384)
        if hidden_dim is None:
            hidden_dim = min(embed_dim, 512)
        
        # LayerNorm to stabilize DINOv2 features BEFORE hashing head
        # DINOv2 features have large magnitude → overflow in float16 AMP
        self.feature_norm = nn.LayerNorm(embed_dim)
        
        # Hashing head
        self.hashing_head = HashingHead(
            embed_dim=embed_dim,
            hash_bit=hash_bit,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        # Gram Anchoring components (simulation)
        if use_gram_anchoring:
            self.gram_projection = nn.Linear(embed_dim, embed_dim)
        
        # Freeze backbone nếu cần
        if freeze_backbone:
            self._freeze_backbone()
    
    def _freeze_backbone(self):
        """Đóng băng tất cả parameters của backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def _compute_gram_matrix(self, features: torch.Tensor) -> torch.Tensor:
        """
        Tính toán ma trận Gram cho Gram Anchoring.
        G = F @ F.T
        """
        return torch.bmm(features, features.transpose(1, 2))
    
    def forward(
        self, 
        x: torch.Tensor, 
        return_patch_tokens: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input images [B, C, H, W]
            return_patch_tokens: Trả về tokens từng patch (cho token pruning)
            
        Returns:
            hash_codes: Mã hash continuous [-1, 1] shape [B, hash_bit]
            features: CLS token features [B, embed_dim]
        """
        # Extract features từ backbone
        features = self.backbone(x)  # [B, embed_dim]
        
        # Normalize features to prevent float16 overflow in AMP
        features = self.feature_norm(features)
        
        # Gram Anchoring (if enabled)
        if self.use_gram_anchoring and self.training:
            features = self.gram_projection(features)
        
        # Generate hash codes
        hash_codes = self.hashing_head(features)
        
        return hash_codes, features
    
    def get_binary_hash(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inference: Trả về mã hash nhị phân {-1, 1}.
        Dùng cho tính toán Hamming Distance.
        """
        self.eval()
        with torch.no_grad():
            hash_codes, _ = self.forward(x)
            return torch.sign(hash_codes)
    
    def get_patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        Lấy patch tokens (không bao gồm CLS token).
        Dùng cho Token Pruning trong Week 2.
        """
        # Forward qua patch embedding và transformer blocks
        x = self.backbone.patch_embed(x)
        x = self.backbone._pos_embed(x)
        x = self.backbone.patch_drop(x)
        x = self.backbone.norm_pre(x)
        x = self.backbone.blocks(x)
        
        # Trả về tất cả tokens (including CLS)
        return x
    
    def compute_flops(self, input_size: Tuple[int, int] = (224, 224)) -> int:
        """Ước tính số FLOPs cho một forward pass."""
        # Simplified FLOPs calculation
        H, W = input_size
        patch_size = 14
        num_patches = (H // patch_size) * (W // patch_size)
        embed_dim = self.backbone.num_features
        num_heads = 6  # ViT-S default
        num_layers = 12
        
        # Attention FLOPs per layer: 4 * N^2 * D
        attention_flops = 4 * (num_patches ** 2) * embed_dim * num_layers
        
        # MLP FLOPs per layer: 8 * N * D^2
        mlp_flops = 8 * num_patches * (embed_dim ** 2) * num_layers
        
        return attention_flops + mlp_flops


if __name__ == "__main__":
    print("Testing DINOv3Hashing Model...")
    
    # Test với pretrained=False để tránh download
    model = DINOv3Hashing(
        model_name='vit_small_patch14_dinov2.lvd142m',
        pretrained=False,
        hash_bit=64
    )
    
    # Test forward
    x = torch.randn(2, 3, 224, 224)
    hash_codes, features = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Hash codes shape: {hash_codes.shape}")
    print(f"Features shape: {features.shape}")
    print(f"Hash code range: [{hash_codes.min().item():.3f}, {hash_codes.max().item():.3f}]")
    
    # Test binary hash
    binary = model.get_binary_hash(x)
    print(f"Binary hash unique values: {torch.unique(binary).tolist()}")
    
    # Test FLOPs
    flops = model.compute_flops()
    print(f"Estimated FLOPs: {flops / 1e9:.2f}G")
