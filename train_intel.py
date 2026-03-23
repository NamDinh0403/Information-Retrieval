"""
Training Script for ViT Hashing (GPU Optimized)
================================================
Training Vision Transformer Hashing với PyTorch.
Tối ưu cho GPU với VRAM hạn chế (4GB+).

Usage:
    python train_intel.py                    # GPU với AMP
    python train_intel.py --batch-size 4     # Giảm batch nếu OOM
    python train_intel.py --device cpu       # Force CPU
    python train_intel.py --quick            # Quick test mode
"""

import os
import sys
import time
import argparse
from datetime import datetime
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import gc

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model import ViT_Hashing
from src.loss import CSQLoss

# ============================================================
# DEVICE SETUP
# ============================================================

def clear_memory():
    """Giải phóng VRAM."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def setup_gpu() -> Dict[str, Any]:
    """
    Setup GPU với các tối ưu hóa.
    """
    info = {
        'device': 'cuda',
        'name': torch.cuda.get_device_name(0),
        'vram_total': torch.cuda.get_device_properties(0).total_memory / 1024**3
    }
    
    # Optimize GPU settings
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    
    return info


def setup_cpu_optimizations() -> Dict[str, Any]:
    """
    Thiết lập các tối ưu hóa CPU cơ bản của PyTorch.
    """
    info = {
        'device': 'cpu',
        'device_name': 'CPU'
    }
    
    # Set optimal thread count
    num_threads = os.cpu_count() or 4
    torch.set_num_threads(num_threads)
    info['num_threads'] = num_threads
    
    # Enable oneDNN/MKL (built into PyTorch)
    try:
        torch.backends.mkldnn.enabled = True
        info['mkldnn'] = True
    except:
        info['mkldnn'] = False
    
    return info


def get_device(preferred: str = 'auto') -> Tuple[torch.device, Dict]:
    """
    Lấy device (ưu tiên GPU).
    """
    if preferred == 'auto' or preferred == 'cuda':
        if torch.cuda.is_available():
            info = setup_gpu()
            return torch.device('cuda'), info
        elif preferred == 'cuda':
            print("[!] CUDA không khả dụng, dùng CPU")
    
    info = setup_cpu_optimizations()
    return torch.device('cpu'), info


# ============================================================
# DATA LOADING
# ============================================================

def get_cifar10_loaders(
    data_dir: str = './data',
    batch_size: int = 16,
    num_workers: int = 2,
    use_gpu: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Tạo DataLoaders cho CIFAR-10.
    """
    # Transforms cho ViT (224x224)
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
    
    # Datasets (download=False vì đã có sẵn)
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=False, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=False, transform=test_transform
    )
    
    # DataLoaders - tối ưu cho GPU
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=use_gpu, drop_last=True,
        persistent_workers=True if num_workers > 0 else False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=use_gpu,
        persistent_workers=True if num_workers > 0 else False
    )
    
    return train_loader, test_loader


# ============================================================
# TRAINING FUNCTIONS
# ============================================================

class Trainer:
    """
    PyTorch Trainer với GPU + AMP support.
    """
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        accumulation_steps: int = 2
    ):
        self.device = device
        self.use_amp = device.type == 'cuda'
        self.accumulation_steps = accumulation_steps
        self.scaler = None
        
        # Move to device
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.optimizer = optimizer
        
        # Setup AMP (tự động cho GPU)
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
            print(f"[✓] AMP enabled")
            print(f"[✓] Gradient accumulation: {accumulation_steps} steps")
    
    def train_epoch(
        self, 
        train_loader: DataLoader, 
        epoch: int
    ) -> Dict[str, float]:
        """Train một epoch với AMP + gradient accumulation."""
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        start_time = time.time()
        
        self.optimizer.zero_grad()
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            # Forward pass với AMP nếu có
            if self.use_amp and self.scaler is not None:
                with torch.cuda.amp.autocast():
                    hash_codes, _ = self.model(images)
                    loss = self.criterion(hash_codes, labels)
                    loss = loss / self.accumulation_steps
                
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                # Standard FP32 training
                hash_codes, _ = self.model(images)
                loss = self.criterion(hash_codes, labels)
                loss = loss / self.accumulation_steps
                
                loss.backward()
                
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            total_loss += loss.item() * self.accumulation_steps
            num_batches += 1
            
            # Progress
            if (batch_idx + 1) % 50 == 0:
                avg_loss = total_loss / num_batches
                elapsed = time.time() - start_time
                samples_per_sec = (batch_idx + 1) * train_loader.batch_size / elapsed
                vram_info = ""
                if self.device.type == 'cuda':
                    vram = torch.cuda.memory_allocated() / 1024**3
                    vram_info = f" | VRAM: {vram:.2f}GB"
                print(f"  Batch [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {avg_loss:.4f} | "
                      f"Speed: {samples_per_sec:.1f} img/s{vram_info}")
        
        # Handle remaining gradients
        if len(train_loader) % self.accumulation_steps != 0:
            if self.use_amp and self.scaler:
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
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            hash_codes, _ = self.model(images)
            loss = self.criterion(hash_codes, labels)
            
            all_hash_codes.append(hash_codes.cpu())
            all_labels.append(labels.cpu())
            total_loss += loss.item()
            num_batches += 1
        
        all_hash_codes = torch.cat(all_hash_codes, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Convert to binary
        binary_codes = torch.sign(all_hash_codes)
        
        # Tính mAP (simplified)
        map_score = self._compute_map(binary_codes, all_labels)
        
        return {
            'loss': total_loss / num_batches,
            'mAP': map_score
        }
    
    def _compute_map(
        self, 
        hash_codes: torch.Tensor, 
        labels: torch.Tensor, 
        top_k: int = -1
    ) -> float:
        """Tính mean Average Precision."""
        n = hash_codes.size(0)
        
        # Similarity matrix
        similarity = hash_codes @ hash_codes.T
        
        # Ground truth similarity (same class = 1)
        gt_similarity = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        
        # Compute AP cho mỗi query
        aps = []
        for i in range(min(n, 1000)):  # Sample 1000 queries
            sim = similarity[i]
            gt = gt_similarity[i]
            
            # Exclude self
            sim[i] = -float('inf')
            
            # Sort
            sorted_indices = torch.argsort(sim, descending=True)
            gt_sorted = gt[sorted_indices]
            
            if top_k > 0:
                gt_sorted = gt_sorted[:top_k]
            
            # Compute AP
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
            'metrics': metrics
        }, path)
        print(f"[✓] Saved checkpoint: {path}")


# ============================================================
# MAIN TRAINING LOOP
# ============================================================

def train(args):
    """Main training function."""
    
    print("=" * 60)
    print("VIT HASHING TRAINING")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup device
    device, device_info = get_device(args.device)
    print(f"\n[Device]")
    print(f"  Type: {device.type}")
    if device.type == 'cuda':
        print(f"  Name: {device_info.get('name', 'N/A')}")
        print(f"  VRAM: {device_info.get('vram_total', 0):.1f} GB")
        print(f"  AMP: ✓ (auto)")
    else:
        print(f"  Threads: {device_info.get('num_threads', 'N/A')}")
        print(f"  MKL-DNN: {'✓' if device_info.get('mkldnn') else '✗'}")
    
    # Config
    effective_batch = args.batch_size * args.accumulation_steps
    config = {
        'hash_bit': args.hash_bit,
        'batch_size': args.batch_size,
        'accumulation': args.accumulation_steps,
        'effective_batch': effective_batch,
        'epochs': args.epochs,
        'lr': args.lr
    }
    print(f"\n[Config]")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    # Data
    print(f"\n[Data] Loading CIFAR-10...")
    train_loader, test_loader = get_cifar10_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_gpu=device.type == 'cuda'
    )
    print(f"  Train: {len(train_loader.dataset):,} images")
    print(f"  Test: {len(test_loader.dataset):,} images")
    
    # Model
    print(f"\n[Model] Creating ViT_Hashing...")
    
    # Xác định weights path
    weights_path = None
    if args.weights:
        weights_path = args.weights
        print(f"  Loading weights from: {weights_path}")
    
    # ViT-B/32 dùng patch size 32
    model_name = 'vit_base_patch32_224' if args.patch_size == 32 else 'vit_base_patch16_224'
    
    model = ViT_Hashing(
        model_name=model_name,
        pretrained=not args.no_pretrained and weights_path is None,
        hash_bit=args.hash_bit,
        weights_path=weights_path
    )
    
    num_params = sum(p.numel() for p in model.parameters())
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
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        accumulation_steps=args.accumulation_steps
    )
    
    # Clear memory
    clear_memory()
    
    # Training loop
    print(f"\n{'=' * 60}")
    print("TRAINING")
    print("=" * 60)
    
    best_map = 0.0
    history = []
    
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
            
            # Save best
            if eval_metrics['mAP'] > best_map:
                best_map = eval_metrics['mAP']
                trainer.save_checkpoint(
                    os.path.join(args.save_dir, 'best_model.pth'),
                    epoch, eval_metrics
                )
        
        # Scheduler step
        scheduler.step()
        
        # History
        history.append({
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'train_speed': train_metrics['samples_per_sec']
        })
    
    # Final summary
    print(f"\n{'=' * 60}")
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best mAP: {best_map:.4f}")
    print(f"Model saved: {os.path.join(args.save_dir, 'best_model.pth')}")
    
    return history


def main():
    parser = argparse.ArgumentParser(
        description='ViT Hashing Training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data
    parser.add_argument('--data-dir', default='./data', help='Data directory')
    parser.add_argument('--save-dir', default='./checkpoints', help='Save directory')
    
    # Model
    parser.add_argument('--hash-bit', type=int, default=64, help='Hash bit length')
    parser.add_argument('--no-pretrained', action='store_true', help='No pretrained weights')
    parser.add_argument('--weights', type=str, default=None, 
                       help='Path to .npz weights file (e.g., ViT-B_32.npz)')
    parser.add_argument('--patch-size', type=int, default=32, choices=[16, 32],
                       help='ViT patch size (16 or 32)')
    
    # Training
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size (giảm nếu OOM)')
    parser.add_argument('--accumulation-steps', type=int, default=2, 
                       help='Gradient accumulation steps')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num-workers', type=int, default=2, help='DataLoader workers')
    parser.add_argument('--eval-every', type=int, default=5, help='Evaluate every N epochs')
    
    # Device
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'],
                       help='Device to use (auto = prefer GPU)')
    
    # Quick test
    parser.add_argument('--quick', action='store_true', help='Quick test mode')
    
    args = parser.parse_args()
    
    # Quick mode
    if args.quick:
        args.epochs = 2
        args.batch_size = 8
        args.accumulation_steps = 1
        args.eval_every = 1
    
    # Create save dir
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Train
    train(args)


if __name__ == "__main__":
    main()
