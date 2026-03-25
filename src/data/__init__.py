# Data Package
"""
Dataset loaders and utilities.
"""

from src.data.loaders import get_cifar10_dataloaders
from src.data.nuswide_loader import (
    get_nuswide_dataloaders,
    NUSWIDEDataset,
    NUSWIDE_21_LABELS,
    NUSWIDE_81_LABELS,
)
from src.data.retrieval_protocol import (
    get_nwpu_retrieval_loaders,
    create_nwpu_retrieval_protocol,
    RetrievalDataset,
)

__all__ = [
    # CIFAR-10
    'get_cifar10_dataloaders',
    # NUS-WIDE (multi-label)
    'get_nuswide_dataloaders',
    'NUSWIDEDataset',
    'NUSWIDE_21_LABELS',
    'NUSWIDE_81_LABELS',
    # NWPU Retrieval (single-label)
    'get_nwpu_retrieval_loaders',
    'create_nwpu_retrieval_protocol',
    'RetrievalDataset',
]
