"""
GPU Training Script for ViT Hashing
===================================
Training tối ưu cho GPU có VRAM hạn chế (GTX 1650 4GB).

Tối ưu:
    - Batch size nhỏ + Gradient Accumulation
    - Mixed Precision (AMP) để giảm VRAM ~50%
    - Gradient checkpointing (optional)
    - Memory-efficient settings

Usage:
    python train_gpu.py                           # Default settings
    python train_gpu.py --batch-size 8            # Adjust batch size
    python train_gpu.py --accumulation-steps 4    # Gradient accumulation
    python train_gpu.py --quick                   # Quick test
"""

import os
import sys
import time
import argparse
import gc
from datetime import datetime
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model import ViT_Hashing
from src.loss import CSQLoss


def check_gpu():
    """Kiểm tra và in thông tin GPU."""
    if not torch.cuda.is_available():
        print("[!] CUDA không khả dụng! Sử dụng CPU.")
        return False
    
    print(f"[GPU Info]")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"  VRAM Free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**3:.1f} GB")
    print(f"  CUDA Version: {torch.version.cuda}")
    print(f"  cuDNN Version: {torch.backends.cudnn.version()}")
    
    # Optimize settings cho GPU
    torch.backends.cudnn.benchmark = True  # Auto-tune convolutions
    torch.backends.cuda.matmul.allow_tf32 = True  # TF32 cho Ampere+
    
    return True


def clear_memory():
    """Giải phóng bộ nhớ GPU."""
    gc.collect()
    torch.cuda.empty_cache()


def get_cifar10_loaders(
    data_dir: str = './data',
    batch_size: int = 8,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    """
    DataLoaders cho CIFAR-10, tối ưu cho GPU.
    """
    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomCrop(224, padding=28),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=False, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=False, transform=test_transform
    )
    
    # pin_memory=True để transfer nhanh hơn, num_workers>0 để load parallel
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=True if num_workers > 0 else False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    return train_loader, test_loader


class GPUTrainer:
    """
    Trainer tối ưu cho GPU với VRAM hạn chế.
    """
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        accumulation_steps: int = 4,
        use_amp: bool = True
    ):
        self.device = torch.device('cuda')
        self.accumulation_steps = accumulation_steps
        self.use_amp = use_amp
        
        # Move to GPU
        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)
        self.optimizer = optimizer
        
        # AMP setup
        self.scaler = torch.cuda.amp.GradScaler() if use_amp else None
        
        print(f"[Trainer]")
        print(f"  AMP: {'✓' if use_amp else '✗'}")
        print(f"  Gradient Accumulation: {accumulation_steps} steps")
        print(f"  Effective batch size: batch_size × {accumulation_steps}")
    
    def train_epoch(
        self, 
        train_loader: DataLoader, 
        epoch: int
    ) -> Dict[str, float]:
        """Train một epoch với gradient accumulation."""
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        start_time = time.time()
        
        self.optimizer.zero_grad()
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            # Forward với AMP
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    hash_codes, _ = self.model(images)
                    loss = self.criterion(hash_codes, labels)
                    loss = loss / self.accumulation_steps  # Scale loss
                
                self.scaler.scale(loss).backward()
            else:
                hash_codes, _ = self.model(images)
                loss = self.criterion(hash_codes, labels)
                loss = loss / self.accumulation_steps
                loss.backward()
            
            total_loss += loss.item() * self.accumulation_steps
            num_batches += 1
            
            # Gradient accumulation step
            if (batch_idx + 1) % self.accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()
            
            # Progress
            if (batch_idx + 1) % 100 == 0:
                avg_loss = total_loss / num_batches
                elapsed = time.time() - start_time
                samples_per_sec = (batch_idx + 1) * train_loader.batch_size / elapsed
                vram_used = torch.cuda.memory_allocated() / 1024**3
                print(f"  Batch [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {avg_loss:.4f} | "
                      f"Speed: {samples_per_sec:.1f} img/s | "
                      f"VRAM: {vram_used:.2f}GB")
        
        # Handle remaining gradients
        if len(train_loader) % self.accumulation_steps != 0:
            if self.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()
        
        epoch_time = time.time() - start_time
        avg_loss = total_loss / num_batches
        
        return {
            'loss': avg_loss,
            'time': epoch_time,
            'samples_per_sec': len(train_loader.dataset) / epoch_time
        }
    
    @torch.no_grad()
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Đánh giá model."""
        self.model.eval()
        
        all_hash_codes = []
        all_labels = []
        total_loss = 0.0
        num_batches = 0
        
        for images, labels in test_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    hash_codes, _ = self.model(images)
                    loss = self.criterion(hash_codes, labels)
            else:
                hash_codes, _ = self.model(images)
                loss = self.criterion(hash_codes, labels)
            
            all_hash_codes.append(hash_codes.cpu())
            all_labels.append(labels.cpu())
            total_loss += loss.item()
            num_batches += 1
        
        all_hash_codes = torch.cat(all_hash_codes, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        binary_codes = torch.sign(all_hash_codes)
        map_score = self._compute_map(binary_codes, all_labels)
        
        return {
            'loss': total_loss / num_batches,
            'mAP': map_score
        }
    
    def _compute_map(
        self, 
        hash_codes: torch.Tensor, 
        labels: torch.Tensor, 
    ) -> float:
        """Tính mAP."""
        n = hash_codes.size(0)
        similarity = hash_codes @ hash_codes.T
        gt_similarity = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        
        aps = []
        for i in range(min(n, 1000)):
            sim = similarity[i].clone()
            gt = gt_similarity[i]
            sim[i] = -float('inf')
            
            sorted_indices = torch.argsort(sim, descending=True)
            gt_sorted = gt[sorted_indices]
            
            num_relevant = gt_sorted.sum().item()
            if num_relevant == 0:
                continue
            
            cumsum = torch.cumsum(gt_sorted, dim=0)
            precision_at_k = cumsum / torch.arange(1, len(gt_sorted) + 1, dtype=torch.float)
            ap = (precision_at_k * gt_sorted).sum() / num_relevant
            aps.append(ap.item())
        
        return np.mean(aps) if aps else 0.0
    
    def save_checkpoint(self, path: str, epoch: int, metrics: Dict):
        """Lưu checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'metrics': metrics
        }, path)
        print(f"[✓] Saved: {path}")


def train(args):
    """Main training function."""
    
    print("=" * 60)
    print("GPU TRAINING - ViT HASHING")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check GPU
    if not check_gpu():
        print("[!] Không có GPU, dừng lại.")
        return
    
    device = torch.device('cuda')
    
    # Config
    effective_batch = args.batch_size * args.accumulation_steps
    print(f"\n[Config]")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Accumulation steps: {args.accumulation_steps}")
    print(f"  Effective batch: {effective_batch}")
    print(f"  Hash bit: {args.hash_bit}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    
    # Data
    print(f"\n[Data] Loading CIFAR-10...")
    train_loader, test_loader = get_cifar10_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    print(f"  Train: {len(train_loader.dataset):,} images")
    print(f"  Test: {len(test_loader.dataset):,} images")
    print(f"  Batches/epoch: {len(train_loader)}")
    
    # Model
    print(f"\n[Model] Creating ViT_Hashing...")
    
    weights_path = args.weights if args.weights else None
    model_name = 'vit_base_patch32_224' if args.patch_size == 32 else 'vit_base_patch16_224'
    
    model = ViT_Hashing(
        model_name=model_name,
        pretrained=weights_path is None,
        hash_bit=args.hash_bit,
        weights_path=weights_path
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {model_name}")
    print(f"  Parameters: {num_params:,} ({num_params/1e6:.1f}M)")
    
    # Loss & Optimizer
    criterion = CSQLoss(
        hash_bit=args.hash_bit,
        num_classes=10,
        lambda_q=0.0001
    )
    
    optimizer = optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': args.lr * 0.1},
        {'params': model.hashing_head.parameters(), 'lr': args.lr}
    ], weight_decay=0.01)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    
    # Trainer
    trainer = GPUTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        accumulation_steps=args.accumulation_steps,
        use_amp=True
    )
    
    # Clear memory trước khi train
    clear_memory()
    
    # Check VRAM usage
    print(f"\n[Memory]")
    print(f"  VRAM used: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"  VRAM reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    
    # Training loop
    print(f"\n{'=' * 60}")
    print("TRAINING")
    print("=" * 60)
    
    best_map = 0.0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n[Epoch {epoch}/{args.epochs}]")
        
        # Train
        train_metrics = trainer.train_epoch(train_loader, epoch)
        print(f"  Train Loss: {train_metrics['loss']:.4f} | "
              f"Time: {train_metrics['time']:.1f}s | "
              f"Speed: {train_metrics['samples_per_sec']:.1f} img/s")
        
        # Evaluate
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            clear_memory()  # Clear trước khi eval
            eval_metrics = trainer.evaluate(test_loader)
            print(f"  Eval Loss: {eval_metrics['loss']:.4f} | mAP: {eval_metrics['mAP']:.4f}")
            
            if eval_metrics['mAP'] > best_map:
                best_map = eval_metrics['mAP']
                trainer.save_checkpoint(
                    os.path.join(args.save_dir, 'best_model_gpu.pth'),
                    epoch, eval_metrics
                )
        
        scheduler.step()
        
        # Memory stats
        print(f"  VRAM: {torch.cuda.memory_allocated() / 1024**3:.2f}GB / "
              f"{torch.cuda.max_memory_allocated() / 1024**3:.2f}GB peak")
    
    # Final
    print(f"\n{'=' * 60}")
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best mAP: {best_map:.4f}")
    print(f"Model saved: {os.path.join(args.save_dir, 'best_model_gpu.pth')}")


def main():
    parser = argparse.ArgumentParser(
        description='GPU Training for ViT Hashing (GTX 1650 4GB optimized)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data
    parser.add_argument('--data-dir', default='./data', help='Data directory')
    parser.add_argument('--save-dir', default='./checkpoints', help='Save directory')
    
    # Model
    parser.add_argument('--hash-bit', type=int, default=64, help='Hash bit length')
    parser.add_argument('--weights', type=str, default=None, help='Path to .npz weights')
    parser.add_argument('--patch-size', type=int, default=32, choices=[16, 32])
    
    # Training - optimized for 4GB VRAM
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, 
                       help='Batch size (keep small for 4GB VRAM)')
    parser.add_argument('--accumulation-steps', type=int, default=4,
                       help='Gradient accumulation steps (effective_batch = batch_size × this)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num-workers', type=int, default=2, help='DataLoader workers')
    parser.add_argument('--eval-every', type=int, default=5, help='Evaluate every N epochs')
    
    # Quick test
    parser.add_argument('--quick', action='store_true', help='Quick test mode')
    
    args = parser.parse_args()
    
    if args.quick:
        args.epochs = 2
        args.batch_size = 4
        args.eval_every = 1
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    train(args)


if __name__ == "__main__":
    main()
