"""
Training Script for Multi-Label Image Retrieval on NUS-WIDE
============================================================

NUS-WIDE is a multi-label dataset designed for image retrieval research.
This script trains ViT/DINOv2 + Deep Hashing for multi-label retrieval.

Key differences from single-label (NWPU):
    - Multi-label annotations (81 concepts, typically use 21 most frequent)
    - Pairwise similarity based on label overlap
    - Modified loss function for multi-label
    - Multi-label mAP evaluation

Usage:
    # Download dataset first
    python scripts/download_nuswide.py --method kaggle
    
    # Train with ViT
    python experiments/train_nuswide.py --model vit --epochs 30
    
    # Train with DINOv2
    python experiments/train_nuswide.py --model dinov2 --epochs 30
    
    # Quick test
    python experiments/train_nuswide.py --quick
"""

import os
import sys
import time
import argparse
import gc
from datetime import datetime
from typing import Dict, Tuple, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Data
from src.data.nuswide_loader import (
    get_nuswide_preprocessed_loaders,
    NUSWIDE_21_LABELS,
)

# Models
from src.models.vit_hashing import ViT_Hashing
from src.models.dinov2_hashing import DINOv3Hashing

# Loss
from src.losses.csq_multilabel_loss import (
    MultiLabelCSQLoss,
    MultiLabelDCHLoss,
    get_multilabel_loss
)

# Metrics
from src.utils.metrics_multilabel import (
    evaluate_multilabel,
    calculate_multilabel_map,
    calculate_precision_at_k
)


def check_gpu():
    """Check GPU availability and print info."""
    if not torch.cuda.is_available():
        print("[!] CUDA not available!")
        return False
    
    print(f"\n[GPU Info]")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"  CUDA Version: {torch.version.cuda}")
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    
    return True


def clear_memory():
    """Clear GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()


def build_model(
    model_type: str,
    num_classes: int,
    hash_bit: int = 64,
    device: torch.device = 'cuda'
) -> nn.Module:
    """
    Build hashing model.
    
    Args:
        model_type: 'vit' or 'dinov2'
        num_classes: Number of label classes (21 for NUS-WIDE standard)
        hash_bit: Number of hash bits
        device: torch device
    """
    print(f"\n[Model] Building {model_type.upper()} + Hashing Head")
    print(f"  Hash bits: {hash_bit}")
    print(f"  Classes: {num_classes}")
    
    if model_type == 'vit':
        model = ViT_Hashing(
            hash_bit=hash_bit,
            num_classes=num_classes,
            pretrained=True
        )
    elif model_type in ['dinov2', 'dinov3']:
        model = DINOv3Hashing(
            hash_bit=hash_bit,
            num_classes=num_classes,
            backbone_size='small'
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params / 1e6:.1f}M")
    print(f"  Trainable params: {trainable_params / 1e6:.1f}M")
    
    return model


def train_one_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    grad_accumulation: int = 4,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    optimizer.zero_grad()
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)  # Multi-hot labels
        
        # Forward
        hash_codes, _ = model(images)
        loss = criterion(hash_codes, labels)
        
        # Gradient accumulation
        loss = loss / grad_accumulation
        loss.backward()
        
        if (batch_idx + 1) % grad_accumulation == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * grad_accumulation
        num_batches += 1
        
        pbar.set_postfix({'loss': f'{total_loss / num_batches:.4f}'})
    
    # Handle remaining gradients
    if (batch_idx + 1) % grad_accumulation != 0:
        optimizer.step()
        optimizer.zero_grad()
    
    return total_loss / num_batches


def validate(
    model: nn.Module,
    query_loader: torch.utils.data.DataLoader,
    db_loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_samples: int = 500,  # Limit for faster validation
) -> dict:
    """Quick validation using a subset."""
    model.eval()
    
    # Extract query codes (limited)
    qB_list, qL_list = [], []
    count = 0
    
    with torch.no_grad():
        for images, labels in query_loader:
            if count >= max_samples:
                break
            images = images.to(device)
            hash_codes, _ = model(images)
            qB_list.append(torch.sign(hash_codes).cpu().numpy())
            qL_list.append(labels.numpy())
            count += images.size(0)
    
    qB = np.concatenate(qB_list)[:max_samples]
    qL = np.concatenate(qL_list)[:max_samples]
    
    # Extract database codes (limited)
    rB_list, rL_list = [], []
    count = 0
    
    with torch.no_grad():
        for images, labels in db_loader:
            if count >= max_samples * 2:
                break
            images = images.to(device)
            hash_codes, _ = model(images)
            rB_list.append(torch.sign(hash_codes).cpu().numpy())
            rL_list.append(labels.numpy())
            count += images.size(0)
    
    rB = np.concatenate(rB_list)[:max_samples * 2]
    rL = np.concatenate(rL_list)[:max_samples * 2]
    
    # Calculate mAP
    mAP = calculate_multilabel_map(qB, rB, qL, rL)
    p_at_k = calculate_precision_at_k(qB, rB, qL, rL, k_list=[10, 50])
    
    return {
        'mAP': mAP,
        'P@10': p_at_k[10],
        'P@50': p_at_k[50],
    }


def train(args):
    """Main training function."""
    print("=" * 70)
    print("NUS-WIDE Multi-Label Image Retrieval Training")
    print("=" * 70)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() and check_gpu() else 'cpu')
    print(f"\n[Device] Using: {device}")
    
    # Hyperparameters
    num_classes = 21  # Standard: 21 most frequent labels
    hash_bit = args.hash_bit
    batch_size = args.batch_size
    epochs = args.epochs
    lr_backbone = args.lr_backbone
    lr_head = args.lr_head
    
    print(f"\n[Hyperparameters]")
    print(f"  Model: {args.model}")
    print(f"  Hash bits: {hash_bit}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  LR backbone: {lr_backbone}")
    print(f"  LR head: {lr_head}")
    print(f"  Loss: {args.loss}")
    
    # Data
    print("\n[Loading Data]")
    train_loader, query_loader, db_loader = get_nuswide_preprocessed_loaders(
        data_dir=args.data_dir,
        batch_size=batch_size,
        num_workers=args.num_workers,
        train_per_class=50 if args.quick else 500,
    )
    label_names = NUSWIDE_21_LABELS
    
    # Model
    model = build_model(
        model_type=args.model,
        num_classes=num_classes,
        hash_bit=hash_bit,
        device=device
    )

    # Fine-tune from NWPU checkpoint (partial weight loading)
    if getattr(args, 'finetune_from', None):
        print(f"\n[Fine-tune] Loading weights from: {args.finetune_from}")
        ckpt = torch.load(args.finetune_from, weights_only=False, map_location='cpu')
        src_sd = ckpt['model_state_dict']

        # Always load backbone (ViT feature extractor transfers across domains)
        backbone_sd = {k: v for k, v in src_sd.items() if k.startswith('backbone')}
        missing, unexpected = model.load_state_dict(backbone_sd, strict=False)
        print(f"  Backbone: loaded {len(backbone_sd)} tensors  "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")

        # Optionally copy the hashing head (same 768→1024→64 arch, different hash centres in loss)
        if getattr(args, 'transfer_head', False):
            head_sd = {k: v for k, v in src_sd.items() if k.startswith('hashing_head')}
            model.load_state_dict(head_sd, strict=False)
            print(f"  Hashing head: loaded {len(head_sd)} tensors")

        # Already fine-tuned backbone needs a much smaller LR nudge
        lr_backbone = lr_backbone * 0.1
        print(f"  Backbone LR reduced to {lr_backbone:.2e} (0.1× of base)")

    # Loss
    criterion = get_multilabel_loss(
        loss_type=args.loss,
        hash_bit=hash_bit,
        num_classes=num_classes,
        lambda_q=args.lambda_q,
        lambda_c=args.lambda_c,
    )
    
    # Optimizer (different LR for backbone and head)
    backbone_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if 'backbone' in name or 'vit' in name:
            backbone_params.append(param)
        else:
            head_params.append(param)
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': lr_backbone},
        {'params': head_params, 'lr': lr_head}
    ], weight_decay=1e-4)
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training loop
    best_mAP = 0.0
    checkpoint_dir = args.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70)
    
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        
        # Train
        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            grad_accumulation=args.grad_accumulation
        )
        
        # Validate
        val_metrics = validate(
            model=model,
            query_loader=query_loader,
            db_loader=db_loader,
            device=device
        )
        
        # Update LR
        scheduler.step()
        
        # Log
        elapsed = time.time() - start_time
        print(f"\n[Epoch {epoch}/{epochs}] "
              f"Loss: {train_loss:.4f} | "
              f"mAP: {val_metrics['mAP']:.4f} | "
              f"P@10: {val_metrics['P@10']:.4f} | "
              f"Time: {elapsed:.1f}s")
        
        # Save best
        if val_metrics['mAP'] > best_mAP:
            best_mAP = val_metrics['mAP']
            checkpoint_path = os.path.join(
                checkpoint_dir,
                f"best_model_nuswide_{args.model}_{hash_bit}bit.pth"
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'mAP': best_mAP,
                'hash_bit': hash_bit,
                'model_type': args.model,
                'num_classes': num_classes,
            }, checkpoint_path)
            print(f"  [✓] New best! Saved to {checkpoint_path}")
        
        clear_memory()
    
    # Final evaluation
    print("\n" + "=" * 70)
    print("Final Evaluation on Full Test Set")
    print("=" * 70)
    
    # Reload best model
    best_checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    final_results = evaluate_multilabel(
        model=model,
        query_loader=query_loader,
        db_loader=db_loader,
        device=device
    )
    
    print("\n[Training Complete]")
    print(f"  Best mAP: {best_mAP:.4f}")
    print(f"  Model saved: {checkpoint_path}")
    
    return final_results


def main():
    parser = argparse.ArgumentParser(description='Train on NUS-WIDE')
    
    # Data
    parser.add_argument('--data-dir', type=str,
                        default='./src/data/archive/NUS-WIDE',
                        help='NUS-WIDE archive directory (contains database_img.txt etc.)')
    
    # Model
    parser.add_argument('--model', type=str, choices=['vit', 'dinov2', 'dinov3'],
                        default='vit', help='Model architecture')
    parser.add_argument('--hash-bit', type=int, default=64,
                        help='Number of hash bits')
    
    # Training
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr-backbone', type=float, default=1e-5)
    parser.add_argument('--lr-head', type=float, default=1e-4)
    parser.add_argument('--grad-accumulation', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=4)
    
    # Loss
    parser.add_argument('--loss', type=str, choices=['csq', 'dch', 'hashnet'],
                        default='csq', help='Loss function')
    parser.add_argument('--lambda-q', type=float, default=0.01,
                        help='Quantization loss weight')
    parser.add_argument('--lambda-c', type=float, default=1.0,
                        help='Center loss weight (CSQ only)')
    
    # Output
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints')
    
    # Quick test
    parser.add_argument('--quick', action='store_true',
                        help='Quick test with limited data')

    # Fine-tuning from NWPU checkpoint
    parser.add_argument('--finetune-from', type=str, default=None,
                        help='Path to NWPU checkpoint to initialise backbone weights from')
    parser.add_argument('--transfer-head', action='store_true',
                        help='Also copy hashing head weights (same 768→1024→hash_bit arch)')

    args = parser.parse_args()
    
    if args.quick:
        args.epochs = 3
        print("\n[Quick Test Mode] Running with limited data and 3 epochs")
    
    train(args)


if __name__ == '__main__':
    main()
