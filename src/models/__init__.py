# Models Package
"""
Deep Hashing architectures for image retrieval.
"""

from src.models.vit_hashing import ViT_Hashing
from src.models.dinov2_hashing import DINOv3Hashing

# Cross-modal (CLIP-based) - requires CLIP installation
try:
    from src.models.clip_hashing import CLIPHashing, CrossModalHashingLoss
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    CLIPHashing = None
    CrossModalHashingLoss = None

__all__ = [
    'ViT_Hashing', 
    'DINOv3Hashing',
    'CLIPHashing',
    'CrossModalHashingLoss',
    'CLIP_AVAILABLE',
]
