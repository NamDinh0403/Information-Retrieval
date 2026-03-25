"""
Retrieval Protocol for Classification Datasets
===============================================

Chuyển đổi dataset classification (như NWPU-RESISC45) thành 
retrieval protocol chuẩn để đánh giá image hashing/retrieval.

Protocol:
    - TRAIN: Dùng để học hash function
    - QUERY: Các ảnh dùng để query (tìm kiếm)
    - DATABASE: Kho ảnh để tìm kiếm (retrieval gallery)

Relevance Definition:
    - Single-label: Relevant nếu cùng class
    - Multi-label: Relevant nếu share ít nhất 1 label
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from typing import Tuple, List, Dict, Optional
from pathlib import Path


class RetrievalDataset(Dataset):
    """
    Wrapper để lấy cả image và label cho retrieval evaluation.
    """
    def __init__(self, base_dataset: Dataset, indices: List[int]):
        self.base_dataset = base_dataset
        self.indices = indices
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        return self.base_dataset[real_idx]


def create_nwpu_retrieval_protocol(
    data_dir: str = './data/archive/Dataset/train/train',
    query_per_class: int = 50,
    db_per_class: int = 150,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Tạo retrieval protocol cho NWPU-RESISC45.
    
    NWPU-RESISC45: 45 classes × 700 images/class = 31,500 images
    
    Default split:
        - Query: 50/class × 45 = 2,250 images
        - Database: 150/class × 45 = 6,750 images  
        - Train: 500/class × 45 = 22,500 images
    
    Args:
        data_dir: Path to NWPU dataset
        query_per_class: Number of query images per class
        db_per_class: Number of database images per class
        seed: Random seed for reproducibility
        
    Returns:
        train_indices, query_indices, db_indices
    """
    np.random.seed(seed)
    
    # Load dataset to get class info
    temp_dataset = ImageFolder(data_dir)
    num_classes = len(temp_dataset.classes)
    images_per_class = len(temp_dataset) // num_classes
    
    print(f"[Protocol] Dataset: {len(temp_dataset)} images, {num_classes} classes")
    print(f"[Protocol] Images per class: {images_per_class}")
    
    train_indices = []
    query_indices = []
    db_indices = []
    
    for class_idx in range(num_classes):
        # Get indices for this class
        class_indices = [
            i for i, (_, label) in enumerate(temp_dataset.samples)
            if label == class_idx
        ]
        
        np.random.shuffle(class_indices)
        
        # Split
        query_indices.extend(class_indices[:query_per_class])
        db_indices.extend(class_indices[query_per_class:query_per_class + db_per_class])
        train_indices.extend(class_indices[query_per_class + db_per_class:])
    
    print(f"[Protocol] Train: {len(train_indices)} | Query: {len(query_indices)} | Database: {len(db_indices)}")
    
    return train_indices, query_indices, db_indices


def get_nwpu_retrieval_loaders(
    data_dir: str = './data/archive/Dataset/train/train',
    batch_size: int = 32,
    query_per_class: int = 50,
    db_per_class: int = 150,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Tạo DataLoaders cho NWPU-RESISC45 retrieval.
    
    Returns:
        train_loader, query_loader, db_loader, class_names
    """
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create protocol split
    train_indices, query_indices, db_indices = create_nwpu_retrieval_protocol(
        data_dir=data_dir,
        query_per_class=query_per_class,
        db_per_class=db_per_class,
        seed=seed,
    )
    
    # Base datasets
    train_base = ImageFolder(data_dir, transform=train_transform)
    eval_base = ImageFolder(data_dir, transform=eval_transform)
    
    # Create subsets
    train_dataset = RetrievalDataset(train_base, train_indices)
    query_dataset = RetrievalDataset(eval_base, query_indices)
    db_dataset = RetrievalDataset(eval_base, db_indices)
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    query_loader = DataLoader(
        query_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    db_loader = DataLoader(
        db_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, query_loader, db_loader, train_base.classes


def save_protocol_indices(
    train_indices: List[int],
    query_indices: List[int],
    db_indices: List[int],
    save_dir: str = './data/protocol'
):
    """Lưu indices để reproducibility."""
    os.makedirs(save_dir, exist_ok=True)
    
    np.save(os.path.join(save_dir, 'train_indices.npy'), train_indices)
    np.save(os.path.join(save_dir, 'query_indices.npy'), query_indices)
    np.save(os.path.join(save_dir, 'db_indices.npy'), db_indices)
    
    print(f"[Protocol] Saved to {save_dir}")


def load_protocol_indices(save_dir: str = './data/protocol'):
    """Load saved indices."""
    train_indices = np.load(os.path.join(save_dir, 'train_indices.npy')).tolist()
    query_indices = np.load(os.path.join(save_dir, 'query_indices.npy')).tolist()
    db_indices = np.load(os.path.join(save_dir, 'db_indices.npy')).tolist()
    
    return train_indices, query_indices, db_indices


# ============================================================================
# COMPARISON PROTOCOLS
# ============================================================================

RETRIEVAL_PROTOCOLS = {
    # NWPU-RESISC45 protocols (single-label)
    'nwpu_standard': {
        'description': 'NWPU standard: 50 query, 150 db per class',
        'query_per_class': 50,
        'db_per_class': 150,
        'relevance': 'same_class',  # Relevant if same class
    },
    'nwpu_large_db': {
        'description': 'NWPU large database: 50 query, 350 db per class',
        'query_per_class': 50,
        'db_per_class': 350,
        'relevance': 'same_class',
    },
    
    # NUS-WIDE protocols (multi-label)
    'nuswide_standard': {
        'description': 'NUS-WIDE: 100 query per class, rest as database',
        'query_per_class': 100,
        'num_classes': 21,
        'relevance': 'share_label',  # Relevant if share at least 1 label
    },
}


def print_protocol_comparison():
    """In bảng so sánh các protocol."""
    print("\n" + "=" * 70)
    print("RETRIEVAL PROTOCOL COMPARISON")
    print("=" * 70)
    print(f"{'Dataset':<20} {'Query':<12} {'Database':<12} {'Relevance':<20}")
    print("-" * 70)
    
    # NWPU
    print(f"{'NWPU-RESISC45':<20} {'2,250':<12} {'6,750':<12} {'Same class':<20}")
    print(f"{'NWPU (large DB)':<20} {'2,250':<12} {'15,750':<12} {'Same class':<20}")
    
    # NUS-WIDE  
    print(f"{'NUS-WIDE':<20} {'2,100':<12} {'~190,000':<12} {'Share ≥1 label':<20}")
    
    print("=" * 70)
    print("\nRelevance Definition:")
    print("  - NWPU (single-label): Image A relevant to B ⟺ class(A) == class(B)")
    print("  - NUS-WIDE (multi-label): Image A relevant to B ⟺ labels(A) ∩ labels(B) ≠ ∅")
    print()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, 
                        default='./data/archive/Dataset/train/train')
    parser.add_argument('--save-protocol', action='store_true',
                        help='Save protocol indices for reproducibility')
    parser.add_argument('--compare', action='store_true',
                        help='Print protocol comparison')
    args = parser.parse_args()
    
    if args.compare:
        print_protocol_comparison()
    else:
        # Test loading
        train_loader, query_loader, db_loader, classes = get_nwpu_retrieval_loaders(
            data_dir=args.data_dir,
            batch_size=8
        )
        
        print(f"\nClasses: {len(classes)}")
        print(f"Sample classes: {classes[:5]}...")
        
        # Test iteration
        for imgs, labels in query_loader:
            print(f"\nQuery batch: {imgs.shape}, labels: {labels[:5]}")
            break
        
        if args.save_protocol:
            train_idx, query_idx, db_idx = create_nwpu_retrieval_protocol(args.data_dir)
            save_protocol_indices(train_idx, query_idx, db_idx)
