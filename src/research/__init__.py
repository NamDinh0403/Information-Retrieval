"""
Research Package for Vision Transformer Optimization
=====================================================
Nghiên cứu tối ưu hóa và triển khai Vision Transformer trong điều kiện giới hạn thời gian.

Modules:
    - dinov3_hashing: DINOv3 backbone với hashing head cho CBIR
    - pruning: Kỹ thuật cắt tỉa token (V-Pruner, Fisher Information)
    - profiler: Công cụ đo lường độ trễ (latency profiling)
    - interpretability: Phân tích giải thích nhân quả (ViT-CX)
    - dataset_research: Data loaders cho NWPU-RESISC45, ChestXray8
"""

from .dinov3_hashing import DINOv3Hashing, HashingHead
from .pruning import TokenPruner, AttentionBasedPruner
from .profiler import LatencyProfiler, FLOPsCalculator
from .interpretability import ViTCXAnalyzer, AttentionRollout
from .dataset_research import get_nwpu_dataloader, get_chestxray_dataloader

__all__ = [
    'DINOv3Hashing',
    'HashingHead', 
    'TokenPruner',
    'AttentionBasedPruner',
    'LatencyProfiler',
    'FLOPsCalculator',
    'ViTCXAnalyzer',
    'AttentionRollout',
    'get_nwpu_dataloader',
    'get_chestxray_dataloader',
]

__version__ = '1.0.0'
