# Models Package
"""
Deep Hashing architectures for image retrieval.
"""

from src.models.vit_hashing import ViT_Hashing
from src.models.dinov2_hashing import DINOv3Hashing

# Cross-modal (CLIP-based) - lazy import to avoid slow transformers load
CLIP_AVAILABLE = False
CLIPHashing = None
CrossModalHashingLoss = None


def get_clip_models():
    """Lazy-load CLIP models only when needed."""
    global CLIP_AVAILABLE, CLIPHashing, CrossModalHashingLoss
    try:
        from src.models.clip_hashing import CLIPHashing as _CLIPHashing
        from src.models.clip_hashing import CrossModalHashingLoss as _CrossModalLoss
        CLIPHashing = _CLIPHashing
        CrossModalHashingLoss = _CrossModalLoss
        CLIP_AVAILABLE = True
        return CLIPHashing, CrossModalHashingLoss
    except Exception:
        return None, None

__all__ = [
    'ViT_Hashing', 
    'DINOv3Hashing',
    'CLIPHashing',
    'CrossModalHashingLoss',
    'CLIP_AVAILABLE',
]
