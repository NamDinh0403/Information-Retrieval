"""
Build NUS-WIDE Vector Database
==============================

Xây dựng vector database cho NUS-WIDE từ các checkpoints đã train.
Hỗ trợ cả 64-bit và 128-bit models.

Usage:
    # Build database với 128-bit model
    python scripts/build_nuswide_db.py \
        --checkpoint checkpoints/best_model_nuswide_vit_128bit.pth \
        --output database/nuswide_128bit.npz
    
    # Build database với 64-bit model
    python scripts/build_nuswide_db.py \
        --checkpoint checkpoints/best_model_nuswide_vit_64bit.pth \
        --output database/nuswide_64bit.npz
    
    # Build cả 2 nếu có cả 2 checkpoints
    python scripts/build_nuswide_db.py --all
    
    # Sử dụng full database (193K images)
    python scripts/build_nuswide_db.py \
        --checkpoint checkpoints/best_model_nuswide_vit_128bit.pth \
        --output database/nuswide_128bit.npz \
        --use-full-db
"""

import os
import sys
import argparse
import glob
from datetime import datetime
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.nuswide_loader import NUSWIDEPreprocessedDataset, NUSWIDE_21_LABELS
from src.models.vit_hashing import ViT_Hashing
from src.models.dinov2_hashing import DINOv3Hashing


# ===========================================================================
# Model Loading
# ===========================================================================

def load_model(checkpoint_path: str, device: torch.device) -> Tuple[nn.Module, dict]:
    """Load model from checkpoint."""
    print(f"\n[*] Loading model: {checkpoint_path}")
    
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt['model_state_dict']
    
    # Auto-detect model type
    has_layers = any('hashing_head.layers.' in k for k in state)
    
    if has_layers:
        model_type = 'dinov2'
        hash_bit = state['hashing_head.layers.3.weight'].shape[0]
    else:
        model_type = 'vit'
        hash_bit = state['hashing_head.3.weight'].shape[0]
    
    num_classes = ckpt.get('num_classes', 21)
    
    print(f"    Type: {model_type}")
    print(f"    Hash bits: {hash_bit}")
    print(f"    Classes: {num_classes}")
    
    # Build model
    if model_type == 'vit':
        model = ViT_Hashing(hash_bit=hash_bit, num_classes=num_classes, pretrained=False)
    else:
        model = DINOv3Hashing(hash_bit=hash_bit, num_classes=num_classes)
    
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    
    return model, {
        'hash_bit': hash_bit,
        'model_type': model_type,
        'num_classes': num_classes,
        'epoch': ckpt.get('epoch', '?'),
        'mAP': ckpt.get('mAP', 'N/A'),
    }


# ===========================================================================
# Vector Database
# ===========================================================================

class NUSWIDEVectorDatabase:
    """Vector database for NUS-WIDE retrieval."""
    
    def __init__(self, hash_codes: np.ndarray, image_paths: List[str],
                 labels: np.ndarray, label_names: List[str],
                 hash_bit: int, model_type: str, created_at: str):
        self.hash_codes = hash_codes
        self.image_paths = image_paths
        self.labels = labels  # Multi-hot labels [N, 21]
        self.label_names = label_names
        self.hash_bit = hash_bit
        self.model_type = model_type
        self.created_at = created_at
    
    def save(self, path: str):
        """Save database to npz file."""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        np.savez_compressed(
            path,
            hash_codes=self.hash_codes,
            image_paths=np.array(self.image_paths, dtype=object),
            labels=self.labels,
            label_names=np.array(self.label_names, dtype=object),
            # For compatibility with app.py
            class_names=np.array(self.label_names, dtype=object),
            hash_bit=self.hash_bit,
            model_type=self.model_type,
            dataset='nuswide',
            created_at=self.created_at,
        )
        
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"\n[✓] Database saved: {path}")
        print(f"    Images: {len(self.image_paths):,}")
        print(f"    Hash bits: {self.hash_bit}")
        print(f"    Labels: {len(self.label_names)}")
        print(f"    Size: {size_mb:.2f} MB")
    
    @classmethod
    def load(cls, path: str) -> 'NUSWIDEVectorDatabase':
        """Load database from file."""
        data = np.load(path, allow_pickle=True)
        return cls(
            hash_codes=data['hash_codes'],
            image_paths=data['image_paths'].tolist(),
            labels=data['labels'],
            label_names=data.get('label_names', data.get('class_names', NUSWIDE_21_LABELS)).tolist(),
            hash_bit=int(data['hash_bit']),
            model_type=str(data['model_type']),
            created_at=str(data['created_at']),
        )


# ===========================================================================
# Extraction
# ===========================================================================

@torch.no_grad()
def extract_codes(model: nn.Module, loader: DataLoader, device: torch.device,
                  desc: str = "Extracting") -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extract hash codes from NUS-WIDE dataset.
    
    Returns:
        hash_codes: [N, hash_bit] binary codes
        labels: [N, 21] multi-hot labels  
        image_paths: [N] paths to images
    """
    model.eval()
    
    all_codes = []
    all_labels = []
    all_paths = []
    
    for batch in tqdm(loader, desc=desc):
        images = batch[0].to(device)
        labels = batch[1].numpy()
        
        # Get paths if available (custom loader)
        if hasattr(loader.dataset, 'image_paths'):
            start_idx = len(all_paths)
            batch_paths = loader.dataset.image_paths[start_idx:start_idx + len(images)]
        else:
            batch_paths = [f"image_{len(all_paths) + i}" for i in range(len(images))]
        
        # Forward
        hash_output, _ = model(images)
        binary_codes = torch.sign(hash_output).cpu().numpy()
        
        all_codes.append(binary_codes)
        all_labels.append(labels)
        all_paths.extend(batch_paths)
    
    return np.concatenate(all_codes), np.concatenate(all_labels), all_paths


def build_database(checkpoint_path: str, data_dir: str, output_path: str,
                   use_full_db: bool, batch_size: int, num_workers: int,
                   device: torch.device) -> NUSWIDEVectorDatabase:
    """Build vector database from checkpoint."""
    
    # Load model
    model, info = load_model(checkpoint_path, device)
    
    # Load dataset
    split = 'database' if use_full_db else 'train'
    print(f"\n[*] Loading {split} split from {data_dir}")
    
    dataset = NUSWIDEPreprocessedDataset(data_dir, split)
    print(f"    Found {len(dataset)} images")
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Extract codes
    hash_codes, labels, image_paths = extract_codes(model, loader, device,
                                                     f"Building {info['hash_bit']}-bit DB")
    
    # Create database
    db = NUSWIDEVectorDatabase(
        hash_codes=hash_codes,
        image_paths=image_paths,
        labels=labels,
        label_names=NUSWIDE_21_LABELS,
        hash_bit=info['hash_bit'],
        model_type=info['model_type'],
        created_at=datetime.now().isoformat(),
    )
    
    # Save
    db.save(output_path)
    
    return db


def find_nuswide_checkpoints(checkpoint_dir: str) -> dict:
    """Find NUS-WIDE checkpoints and group by hash bits."""
    patterns = [
        os.path.join(checkpoint_dir, "*nuswide*64*.pth"),
        os.path.join(checkpoint_dir, "*nuswide*128*.pth"),
    ]
    
    checkpoints = {}
    
    for pattern in patterns:
        found = glob.glob(pattern)
        for ckpt in found:
            name = os.path.basename(ckpt).lower()
            if '128' in name:
                checkpoints[128] = ckpt
            elif '64' in name:
                checkpoints[64] = ckpt
    
    return checkpoints


def main():
    parser = argparse.ArgumentParser(description='Build NUS-WIDE vector database')
    
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for database (.npz)')
    parser.add_argument('--all', action='store_true',
                        help='Build databases for all found checkpoints (64/128 bit)')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                        help='Directory containing checkpoints')
    parser.add_argument('--data-dir', type=str, default='./src/data/archive/NUS-WIDE',
                        help='NUS-WIDE data directory')
    parser.add_argument('--output-dir', type=str, default='./database',
                        help='Output directory for databases')
    parser.add_argument('--use-full-db', action='store_true',
                        help='Use full 193K database split instead of training set')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=4)
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.all:
        # Build all found checkpoints
        checkpoints = find_nuswide_checkpoints(args.checkpoint_dir)
        
        if not checkpoints:
            print(f"[!] No NUS-WIDE checkpoints found in {args.checkpoint_dir}")
            print("    Looking for patterns: *nuswide*64*.pth, *nuswide*128*.pth")
            return
        
        print(f"[*] Found {len(checkpoints)} checkpoints:")
        for bits, path in checkpoints.items():
            print(f"    {bits}-bit: {path}")
        
        for bits, ckpt_path in checkpoints.items():
            output_path = os.path.join(args.output_dir, f"nuswide_{bits}bit.npz")
            build_database(
                ckpt_path, args.data_dir, output_path,
                args.use_full_db, args.batch_size, args.num_workers, device
            )
    
    elif args.checkpoint:
        # Build single checkpoint
        if not os.path.exists(args.checkpoint):
            print(f"[!] Checkpoint not found: {args.checkpoint}")
            return
        
        # Auto-generate output path if not specified
        if args.output is None:
            # Infer hash bits from checkpoint name
            name = os.path.basename(args.checkpoint).lower()
            if '128' in name:
                bits = 128
            elif '64' in name:
                bits = 64
            elif '32' in name:
                bits = 32
            elif '16' in name:
                bits = 16
            else:
                bits = 'unknown'
            args.output = os.path.join(args.output_dir, f"nuswide_{bits}bit.npz")
        
        build_database(
            args.checkpoint, args.data_dir, args.output,
            args.use_full_db, args.batch_size, args.num_workers, device
        )
    
    else:
        parser.print_help()
        print("\nExamples:")
        print("  # Build 128-bit database:")
        print("  python scripts/build_nuswide_db.py \\")
        print("      --checkpoint checkpoints/best_model_nuswide_vit_128bit.pth")
        print()
        print("  # Build all available:")
        print("  python scripts/build_nuswide_db.py --all")


if __name__ == '__main__':
    main()
