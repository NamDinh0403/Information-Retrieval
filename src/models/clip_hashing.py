"""
Cross-Modal Hashing Model (CLIP + Hashing)
==========================================

Sử dụng CLIP để encode cả text và image vào cùng embedding space,
sau đó thêm hash head để nén thành binary codes.

Hỗ trợ 3 loại query:
1. Text query: "an airport with airplanes"
2. Image query: [image tensor]
3. Multimodal query: text + image (fusion)

Architecture:
    ┌─────────────┐     ┌─────────────┐
    │ Text Input  │     │ Image Input │
    └──────┬──────┘     └──────┬──────┘
           │                   │
           ↓                   ↓
    ┌─────────────┐     ┌─────────────┐
    │ CLIP Text   │     │ CLIP Image  │
    │  Encoder    │     │  Encoder    │
    └──────┬──────┘     └──────┬──────┘
           │                   │
           ↓                   ↓
    ┌─────────────────────────────────┐
    │     Shared Embedding Space      │
    │          (512/768-dim)          │
    └─────────────┬───────────────────┘
                  │
                  ↓
    ┌─────────────────────────────────┐
    │         Hash Head               │
    │     (MLP → 64-bit hash)         │
    └─────────────────────────────────┘

Usage:
    model = CLIPHashing(hash_bit=64)
    
    # Text query
    text_hash = model.encode_text("an airport with planes")
    
    # Image query  
    image_hash = model.encode_image(image_tensor)
    
    # Cross-modal search
    results = search(text_hash, image_database)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List, Tuple
import numpy as np

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("[Warning] CLIP not installed. Run: pip install git+https://github.com/openai/CLIP.git")

try:
    from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer
    TRANSFORMERS_CLIP_AVAILABLE = True
except ImportError:
    TRANSFORMERS_CLIP_AVAILABLE = False


class HashingHead(nn.Module):
    """Hash head to convert embeddings to binary codes."""
    
    def __init__(self, embed_dim: int, hash_bit: int = 64):
        super().__init__()
        self.hash_bit = hash_bit
        
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, hash_bit),
            nn.Tanh()  # Output in [-1, 1]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class CLIPHashing(nn.Module):
    """
    CLIP-based Cross-Modal Hashing Model.
    
    Supports:
    - Text-to-Image retrieval
    - Image-to-Image retrieval
    - Multimodal queries
    """
    
    def __init__(
        self,
        hash_bit: int = 64,
        clip_model: str = "ViT-B/32",
        freeze_clip: bool = True,
        device: str = "cuda"
    ):
        super().__init__()
        
        self.hash_bit = hash_bit
        self.device = device
        self.freeze_clip = freeze_clip
        
        # Load CLIP
        if CLIP_AVAILABLE:
            self.clip_model, self.preprocess = clip.load(clip_model, device=device)
            self.embed_dim = self.clip_model.visual.output_dim
            self.use_openai_clip = True
        elif TRANSFORMERS_CLIP_AVAILABLE:
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.embed_dim = 512
            self.use_openai_clip = False
        else:
            raise ImportError("Please install CLIP: pip install git+https://github.com/openai/CLIP.git")
        
        # Freeze CLIP if specified
        if freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False
        
        # Shared hash head for both modalities
        self.hash_head = HashingHead(self.embed_dim, hash_bit)
        
        # Optional: Separate hash heads for text and image
        self.text_hash_head = HashingHead(self.embed_dim, hash_bit)
        self.image_hash_head = HashingHead(self.embed_dim, hash_bit)
        
        self.use_separate_heads = False  # Can be toggled
        
        print(f"[CLIPHashing] Initialized with {clip_model}")
        print(f"  Embedding dim: {self.embed_dim}")
        print(f"  Hash bits: {hash_bit}")
        print(f"  CLIP frozen: {freeze_clip}")
    
    def encode_image(
        self, 
        images: torch.Tensor,
        return_embedding: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Encode images to hash codes.
        
        Args:
            images: [B, C, H, W] preprocessed image tensor
            return_embedding: If True, also return CLIP embedding
            
        Returns:
            hash_codes: [B, hash_bit] continuous hash codes in [-1, 1]
            (optional) embeddings: [B, embed_dim] CLIP embeddings
        """
        # Get CLIP image features
        if self.use_openai_clip:
            with torch.no_grad() if self.freeze_clip else torch.enable_grad():
                image_features = self.clip_model.encode_image(images)
                image_features = image_features.float()
        else:
            outputs = self.clip_model.get_image_features(pixel_values=images)
            image_features = outputs
        
        # Normalize
        image_features = F.normalize(image_features, dim=-1)
        
        # Hash
        if self.use_separate_heads:
            hash_codes = self.image_hash_head(image_features)
        else:
            hash_codes = self.hash_head(image_features)
        
        if return_embedding:
            return hash_codes, image_features
        return hash_codes
    
    def encode_text(
        self,
        texts: Union[str, List[str]],
        return_embedding: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Encode text to hash codes.
        
        Args:
            texts: Single text or list of texts
            return_embedding: If True, also return CLIP embedding
            
        Returns:
            hash_codes: [B, hash_bit] continuous hash codes
            (optional) embeddings: [B, embed_dim] CLIP embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize and encode
        if self.use_openai_clip:
            text_tokens = clip.tokenize(texts).to(self.device)
            with torch.no_grad() if self.freeze_clip else torch.enable_grad():
                text_features = self.clip_model.encode_text(text_tokens)
                text_features = text_features.float()
        else:
            inputs = self.processor(text=texts, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            text_features = self.clip_model.get_text_features(**inputs)
        
        # Normalize
        text_features = F.normalize(text_features, dim=-1)
        
        # Hash
        if self.use_separate_heads:
            hash_codes = self.text_hash_head(text_features)
        else:
            hash_codes = self.hash_head(text_features)
        
        if return_embedding:
            return hash_codes, text_features
        return hash_codes
    
    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        texts: Optional[Union[str, List[str]]] = None,
    ) -> dict:
        """
        Forward pass for training.
        
        Returns dict with hash codes and embeddings for both modalities.
        """
        result = {}
        
        if images is not None:
            img_hash, img_embed = self.encode_image(images, return_embedding=True)
            result['image_hash'] = img_hash
            result['image_embed'] = img_embed
        
        if texts is not None:
            txt_hash, txt_embed = self.encode_text(texts, return_embedding=True)
            result['text_hash'] = txt_hash
            result['text_embed'] = txt_embed
        
        return result
    
    def get_binary_codes(self, hash_codes: torch.Tensor) -> torch.Tensor:
        """Convert continuous hash codes to binary {-1, 1}."""
        return torch.sign(hash_codes)


class CrossModalHashingLoss(nn.Module):
    """
    Loss for Cross-Modal Hashing training.
    
    Combines:
    1. Cross-modal similarity loss (text-image pairs should have similar hashes)
    2. Intra-modal similarity loss (similar images should have similar hashes)
    3. Quantization loss (push values to ±1)
    """
    
    def __init__(
        self,
        hash_bit: int,
        lambda_cross: float = 1.0,
        lambda_quant: float = 0.0001,
        margin: float = 0.5,
    ):
        super().__init__()
        self.hash_bit = hash_bit
        self.lambda_cross = lambda_cross
        self.lambda_quant = lambda_quant
        self.margin = margin
    
    def forward(
        self,
        image_hash: torch.Tensor,
        text_hash: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute loss.
        
        Args:
            image_hash: [B, hash_bit] image hash codes
            text_hash: [B, hash_bit] text hash codes
            labels: [B, num_classes] multi-hot labels (optional)
        """
        batch_size = image_hash.size(0)
        
        # 1. Cross-modal loss: matched pairs should be similar
        # Assuming i-th text describes i-th image (paired data)
        cross_sim = torch.sum(image_hash * text_hash, dim=1) / self.hash_bit
        cross_loss = torch.mean((1 - cross_sim) ** 2)  # Matched pairs: similarity → 1
        
        # 2. Intra-modal loss (if labels provided)
        intra_loss = 0.0
        if labels is not None:
            # Compute label similarity
            label_sim = torch.mm(labels, labels.t())
            label_sim = (label_sim > 0).float()  # Binary: share any label
            
            # Image-image similarity
            img_sim = torch.mm(image_hash, image_hash.t()) / self.hash_bit
            
            # Loss: similar images should have similar hashes
            pos_mask = label_sim
            neg_mask = 1 - label_sim
            
            pos_loss = pos_mask * (1 - img_sim) ** 2
            neg_loss = neg_mask * F.relu(img_sim + self.margin) ** 2
            
            # Remove diagonal
            mask = 1 - torch.eye(batch_size, device=image_hash.device)
            intra_loss = (pos_loss * mask).sum() / (mask.sum() + 1e-8)
            intra_loss += (neg_loss * mask).sum() / (mask.sum() + 1e-8)
        
        # 3. Quantization loss
        quant_loss = torch.mean((torch.abs(image_hash) - 1) ** 2)
        quant_loss += torch.mean((torch.abs(text_hash) - 1) ** 2)
        
        # Total loss
        total_loss = (
            self.lambda_cross * cross_loss +
            intra_loss +
            self.lambda_quant * quant_loss
        )
        
        return total_loss


def create_clip_hashing_model(
    hash_bit: int = 64,
    clip_model: str = "ViT-B/32",
    freeze_clip: bool = True,
    device: str = "cuda",
) -> CLIPHashing:
    """Factory function to create CLIP hashing model."""
    return CLIPHashing(
        hash_bit=hash_bit,
        clip_model=clip_model,
        freeze_clip=freeze_clip,
        device=device,
    )


if __name__ == '__main__':
    # Test
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("Testing CLIPHashing model...")
    
    model = CLIPHashing(hash_bit=64, device=device)
    model = model.to(device)
    
    # Test text encoding
    texts = ["an airport with airplanes", "a beach with palm trees"]
    text_hash = model.encode_text(texts)
    print(f"Text hash shape: {text_hash.shape}")  # [2, 64]
    
    # Test image encoding (dummy)
    dummy_images = torch.randn(2, 3, 224, 224).to(device)
    img_hash = model.encode_image(dummy_images)
    print(f"Image hash shape: {img_hash.shape}")  # [2, 64]
    
    # Test similarity
    text_binary = torch.sign(text_hash)
    img_binary = torch.sign(img_hash)
    
    # Hamming distance
    hamming = 0.5 * (64 - torch.mm(text_binary, img_binary.t()))
    print(f"Cross-modal Hamming distances:\n{hamming}")
    
    print("\n[✓] CLIPHashing model working!")
