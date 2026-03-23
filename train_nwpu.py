"""
GPU Training for NWPU-RESISC45 Dataset
======================================
Fine-tune ViT Hashing trên bộ dữ liệu remote sensing.

Tối ưu cho GTX 1650 4GB VRAM.

Usage:
    python train_nwpu.py                    # Default
    python train_nwpu.py --batch-size 4     # Nếu OOM
    python train_nwpu.py --quick            # Quick test
"""

import os
import sys
import time
import argparse
import gc
from datetime import datetime
from typing import Dict, Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import numpy as np

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model import ViT_Hashing
from src.loss import CSQLoss


def check_gpu():
    """Kiểm tra GPU."""
    if not torch.cuda.is_available():
        print("[!] CUDA không khả dụng!")
        return False
    
    print(f"[GPU] {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    
    return True


def clear_memory():
    """Giải phóng VRAM."""
    gc.collect()
    torch.cuda.empty_cache()


def get_nwpu_loaders(
    data_dir: str = './data/archive/Dataset',
    batch_size: int = 8,
    num_workers: int = 2,
    val_split: float = 0.2
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Load NWPU-RESISC45 với train/val/test split.
    
    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),  # Ảnh vệ tinh có thể xoay
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Paths
    train_path = os.path.join(data_dir, 'train', 'train')
    test_path = os.path.join(data_dir, 'test', 'test')
    
    # Load full train set
    full_train = ImageFolder(train_path, transform=train_transform)
    class_names = full_train.classes
    num_classes = len(class_names)
    
    # Split train/val
    val_size = int(len(full_train) * val_split)
    train_size = len(full_train) - val_size
    train_dataset, val_dataset = random_split(
        full_train, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Test set
    test_dataset = ImageFolder(test_path, transform=test_transform)
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=True if num_workers > 0 else False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    return train_loader, val_loader, test_loader, class_names


class NWPUTrainer:
    """Trainer cho NWPU-RESISC45."""
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        accumulation_steps: int = 4
    ):
        self.device = torch.device('cuda')
        self.accumulation_steps = accumulation_steps
        
        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)
        self.optimizer = optimizer
        
        self.scaler = torch.cuda.amp.GradScaler()
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train một epoch."""
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        start_time = time.time()
        
        self.optimizer.zero_grad()
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            with torch.cuda.amp.autocast():
                hash_codes, _ = self.model(images)
                loss = self.criterion(hash_codes, labels)
                loss = loss / self.accumulation_steps
            
            self.scaler.scale(loss).backward()
            
            total_loss += loss.item() * self.accumulation_steps
            num_batches += 1
            
            if (batch_idx + 1) % self.accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            
            if (batch_idx + 1) % 50 == 0:
                avg_loss = total_loss / num_batches
                elapsed = time.time() - start_time
                speed = (batch_idx + 1) * train_loader.batch_size / elapsed
                vram = torch.cuda.memory_allocated() / 1024**3
                print(f"  [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {avg_loss:.4f} | {speed:.1f} img/s | VRAM: {vram:.2f}GB")
        
        # Remaining gradients
        if len(train_loader) % self.accumulation_steps != 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
        
        return {
            'loss': total_loss / num_batches,
            'time': time.time() - start_time,
            'samples_per_sec': len(train_loader.dataset) / (time.time() - start_time)
        }
    
    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, desc: str = "Eval") -> Dict[str, float]:
        """Đánh giá model."""
        self.model.eval()
        
        all_codes = []
        all_labels = []
        total_loss = 0.0
        num_batches = 0
        
        for images, labels in dataloader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            with torch.cuda.amp.autocast():
                hash_codes, _ = self.model(images)
                loss = self.criterion(hash_codes, labels)
            
            all_codes.append(hash_codes.cpu())
            all_labels.append(labels.cpu())
            total_loss += loss.item()
            num_batches += 1
        
        all_codes = torch.cat(all_codes, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        binary = torch.sign(all_codes)
        map_score = self._compute_map(binary, all_labels)
        
        return {
            'loss': total_loss / num_batches,
            'mAP': map_score
        }
    
    def _compute_map(self, codes: torch.Tensor, labels: torch.Tensor) -> float:
        """Tính mAP."""
        n = codes.size(0)
        sim = codes @ codes.T
        gt = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        
        aps = []
        for i in range(min(n, 500)):
            s = sim[i].clone()
            g = gt[i]
            s[i] = -float('inf')
            
            idx = torch.argsort(s, descending=True)
            g_sorted = g[idx]
            
            num_rel = g_sorted.sum().item()
            if num_rel == 0:
                continue
            
            cumsum = torch.cumsum(g_sorted, dim=0)
            prec = cumsum / torch.arange(1, len(g_sorted) + 1, dtype=torch.float)
            ap = (prec * g_sorted).sum() / num_rel
            aps.append(ap.item())
        
        return np.mean(aps) if aps else 0.0
    
    def save(self, path: str, epoch: int, metrics: Dict):
        """Lưu checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'metrics': metrics
        }, path)
        print(f"[✓] Saved: {path}")


def train(args):
    """Main training."""
    
    print("=" * 60)
    print("NWPU-RESISC45 TRAINING")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if not check_gpu():
        return
    
    # Data
    print(f"\n[Data] Loading NWPU-RESISC45...")
    train_loader, val_loader, test_loader, class_names = get_nwpu_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    num_classes = len(class_names)
    print(f"  Classes: {num_classes}")
    print(f"  Train: {len(train_loader.dataset)}")
    print(f"  Val: {len(val_loader.dataset)}")
    print(f"  Test: {len(test_loader.dataset)}")
    
    # Model
    print(f"\n[Model] Creating ViT_Hashing...")
    model_name = 'vit_base_patch32_224' if args.patch_size == 32 else 'vit_base_patch16_224'
    
    model = ViT_Hashing(
        model_name=model_name,
        pretrained=True,  # Load ImageNet pretrained
        hash_bit=args.hash_bit,
        weights_path=args.weights
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {model_name}")
    print(f"  Hash bit: {args.hash_bit}")
    print(f"  Parameters: {num_params/1e6:.1f}M")
    
    # Loss với 45 classes
    criterion = CSQLoss(
        hash_bit=args.hash_bit,
        num_classes=num_classes,  # 45 classes!
        lambda_q=0.0001
    )
    
    # Optimizer
    optimizer = optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': args.lr * 0.1},
        {'params': model.hashing_head.parameters(), 'lr': args.lr}
    ], weight_decay=0.01)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    
    # Trainer
    trainer = NWPUTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        accumulation_steps=args.accumulation_steps
    )
    
    clear_memory()
    
    # Config summary
    print(f"\n[Config]")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Accumulation: {args.accumulation_steps}")
    print(f"  Effective batch: {args.batch_size * args.accumulation_steps}")
    print(f"  Epochs: {args.epochs}")
    print(f"  LR: {args.lr}")
    
    # Training
    print(f"\n{'=' * 60}")
    print("TRAINING")
    print("=" * 60)
    
    best_map = 0.0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n[Epoch {epoch}/{args.epochs}]")
        
        # Train
        train_metrics = trainer.train_epoch(train_loader, epoch)
        print(f"  Train - Loss: {train_metrics['loss']:.4f} | "
              f"Time: {train_metrics['time']:.1f}s | "
              f"Speed: {train_metrics['samples_per_sec']:.1f} img/s")
        
        # Validate
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            clear_memory()
            val_metrics = trainer.evaluate(val_loader, "Val")
            print(f"  Val - Loss: {val_metrics['loss']:.4f} | mAP: {val_metrics['mAP']:.4f}")
            
            if val_metrics['mAP'] > best_map:
                best_map = val_metrics['mAP']
                trainer.save(
                    os.path.join(args.save_dir, 'best_model_nwpu.pth'),
                    epoch, val_metrics
                )
        
        scheduler.step()
    
    # Final test
    print(f"\n{'=' * 60}")
    print("FINAL TEST")
    print("=" * 60)
    
    clear_memory()
    test_metrics = trainer.evaluate(test_loader, "Test")
    print(f"  Test mAP: {test_metrics['mAP']:.4f}")
    
    # Summary
    print(f"\n{'=' * 60}")
    print("COMPLETE")
    print("=" * 60)
    print(f"  Best Val mAP: {best_map:.4f}")
    print(f"  Test mAP: {test_metrics['mAP']:.4f}")
    print(f"  Model: {os.path.join(args.save_dir, 'best_model_nwpu.pth')}")


def main():
    parser = argparse.ArgumentParser(
        description='Train ViT Hashing on NWPU-RESISC45',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data
    parser.add_argument('--data-dir', default='./data/archive/Dataset')
    parser.add_argument('--save-dir', default='./checkpoints')
    
    # Model
    parser.add_argument('--hash-bit', type=int, default=64)
    parser.add_argument('--patch-size', type=int, default=32, choices=[16, 32])
    parser.add_argument('--weights', type=str, default=None, help='Pretrained .npz weights')
    
    # Training
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--accumulation-steps', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--eval-every', type=int, default=5)
    
    # Quick
    parser.add_argument('--quick', action='store_true')
    
    args = parser.parse_args()
    
    if args.quick:
        args.epochs = 3
        args.batch_size = 4
        args.eval_every = 1
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    train(args)


if __name__ == "__main__":
    main()
