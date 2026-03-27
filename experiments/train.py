"""
Training Script for Image Retrieval on NWPU-RESISC45
=====================================================
Hệ thống truy vấn ảnh viễn thám sử dụng ViT + Deep Hashing.

Phương pháp:
    - ViT (Vision Transformer): Extract image features
    - Deep Hashing (CSQ Loss): Chuyển features → binary hash codes
    - Hamming Distance: Tìm kiếm nhanh

Models:
    - vit: ViT-B/32 pretrained ImageNet (baseline, đơn giản)
    - dinov3: DINOv2 pretrained 142M images (optional, so sánh)

Tối ưu cho GTX 1650 4GB VRAM.

Usage:
    python train_nwpu.py                    # Train với ViT (default)
    python train_nwpu.py --model dinov3     # Train với DINOv2 (so sánh)
    python train_nwpu.py --quick            # Quick test (3 epochs)
    python train_nwpu.py --batch-size 4     # Nếu OOM
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
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Models
from src.models.vit_hashing import ViT_Hashing
from src.models.dinov2_hashing import DINOv3Hashing, HashingHead

# Loss
from src.losses.csq_loss import CSQLoss

# Utils (pruning - optional)
from src.utils.pruning import TokenPruner, AttentionBasedPruner, TokenMerger, analyze_pruning_effect


def check_gpu():
    """Kiểm tra GPU và in thông tin chi tiết."""
    if not torch.cuda.is_available():
        print("[!] CUDA không khả dụng!")
        return False
    
    print(f"\n[GPU Info]")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"  VRAM Free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**3:.1f} GB")
    print(f"  CUDA Version: {torch.version.cuda}")
    print(f"  cuDNN Enabled: {torch.backends.cudnn.enabled}")
    print(f"  PyTorch Version: {torch.__version__}")
    
    # Optimize settings cho GPU
    torch.backends.cudnn.benchmark = True  # Auto-tune convolutions
    torch.backends.cuda.matmul.allow_tf32 = True  # TF32 cho Ampere+
    torch.backends.cudnn.allow_tf32 = True  # TF32 cho cuDNN
    
    # Check torch.compile availability (PyTorch 2.0+)
    has_compile = hasattr(torch, 'compile')
    print(f"  torch.compile: {'Available' if has_compile else 'Not available (PyTorch < 2.0)'}")
    
    return True


def clear_memory():
    """Giải phóng VRAM."""
    gc.collect()
    torch.cuda.empty_cache()


def get_nwpu_loaders(
    data_dir: str = './data/archive/Dataset',
    batch_size: int = 8,
    num_workers: int = 2,
    val_split: float = 0.2,
    image_size: int = 224
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Load NWPU-RESISC45 với train/val/test split.
    
    Args:
        image_size: Input image size (default 224, use 182 for faster DINOv2)
    
    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    # Transforms
    resize_size = int(image_size * 256 / 224)  # Scale accordingly
    
    train_transform = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),  # Ảnh vệ tinh có thể xoay
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(image_size),
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
    """
    Trainer cho NWPU-RESISC45 với đầy đủ research features.
    
    Features:
        - Mixed Precision Training (AMP)
        - Gradient Accumulation
        - Token Pruning (V-Pruner / Attention-based)
        - DINOv3Hashing support
    """
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        accumulation_steps: int = 4,
        enable_pruning: bool = False,
        keep_ratio: float = 0.7,
        pruning_method: str = 'fisher'  # 'fisher' hoặc 'attention'
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.accumulation_steps = accumulation_steps
        self.enable_pruning = enable_pruning
        
        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)
        self.optimizer = optimizer
        
        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler()
        
        # Token Pruning setup
        if enable_pruning:
            self.pruner = TokenPruner(keep_ratio=keep_ratio, min_tokens=16)
            if pruning_method == 'attention':
                # Get embed_dim from model
                if hasattr(model, 'backbone'):
                    embed_dim = model.backbone.num_features
                else:
                    embed_dim = 768  # Default ViT-B
                self.attention_pruner = AttentionBasedPruner(
                    embed_dim=embed_dim, 
                    keep_ratio=keep_ratio
                ).to(self.device)
            else:
                self.attention_pruner = None
            print(f"  [Pruning] Enabled - Method: {pruning_method}, Keep ratio: {keep_ratio}")
        else:
            self.pruner = None
            self.attention_pruner = None
        
        # Pruning statistics
        self.pruning_stats = []
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train một epoch với Token Pruning support."""
        self.model.train()
        if self.attention_pruner:
            self.attention_pruner.train()
        
        total_loss = 0.0
        num_batches = 0
        start_time = time.time()
        epoch_pruning_stats = []
        
        self.optimizer.zero_grad()
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            with torch.cuda.amp.autocast():
                # Forward với Token Pruning (nếu enabled)
                if self.enable_pruning and hasattr(self.model, 'get_patch_tokens'):
                    # Lấy patch tokens trước khi hash
                    patch_tokens = self.model.get_patch_tokens(images)
                    
                    # Apply pruning
                    if self.attention_pruner:
                        pruned_tokens, _ = self.attention_pruner(patch_tokens)
                    else:
                        # Fisher-based pruning
                        scores = self.pruner.compute_fisher_scores(patch_tokens)
                        pruned_tokens, kept_indices = self.pruner.prune_tokens(
                            patch_tokens, scores, keep_cls=True
                        )
                    
                    # Lưu stats
                    epoch_pruning_stats.append({
                        'original': patch_tokens.shape[1],
                        'pruned': pruned_tokens.shape[1]
                    })
                    
                    # Forward qua phần còn lại của model
                    cls_token = pruned_tokens[:, 0]  # CLS token
                    hash_codes = self.model.hashing_head(cls_token)
                    features = cls_token
                else:
                    # Standard forward
                    hash_codes, features = self.model(images)
                
                loss = self.criterion(hash_codes, labels)
                loss = loss / self.accumulation_steps
            
            self.scaler.scale(loss).backward()
            
            total_loss += loss.item() * self.accumulation_steps
            num_batches += 1
            
            if (batch_idx + 1) % self.accumulation_steps == 0:
                # Gradient clipping để ổn định training
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            
            if (batch_idx + 1) % 50 == 0:
                avg_loss = total_loss / num_batches
                elapsed = time.time() - start_time
                speed = (batch_idx + 1) * train_loader.batch_size / elapsed
                vram = torch.cuda.memory_allocated() / 1024**3
                pruning_info = ""
                if epoch_pruning_stats:
                    avg_kept = np.mean([s['pruned']/s['original'] for s in epoch_pruning_stats])
                    pruning_info = f" | Kept: {avg_kept:.1%}"
                print(f"  [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {avg_loss:.4f} | {speed:.1f} img/s | VRAM: {vram:.2f}GB{pruning_info}")
        
        # Remaining gradients
        if len(train_loader) % self.accumulation_steps != 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
        
        # Store pruning stats
        if epoch_pruning_stats:
            self.pruning_stats.append({
                'epoch': epoch,
                'avg_kept_ratio': np.mean([s['pruned']/s['original'] for s in epoch_pruning_stats]),
                'num_samples': len(epoch_pruning_stats)
            })
        
        return {
            'loss': total_loss / num_batches,
            'time': time.time() - start_time,
            'samples_per_sec': len(train_loader.dataset) / (time.time() - start_time)
        }
    
    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, desc: str = "Eval") -> Dict[str, float]:
        """Đánh giá model với pruning support."""
        self.model.eval()
        if self.attention_pruner:
            self.attention_pruner.eval()
        
        all_codes = []
        all_labels = []
        total_loss = 0.0
        num_batches = 0
        
        for images, labels in dataloader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            with torch.cuda.amp.autocast():
                # Forward với Token Pruning (nếu enabled)
                if self.enable_pruning and hasattr(self.model, 'get_patch_tokens'):
                    patch_tokens = self.model.get_patch_tokens(images)
                    
                    if self.attention_pruner:
                        pruned_tokens, _ = self.attention_pruner(patch_tokens)
                    else:
                        scores = self.pruner.compute_fisher_scores(patch_tokens)
                        pruned_tokens, _ = self.pruner.prune_tokens(
                            patch_tokens, scores, keep_cls=True
                        )
                    
                    cls_token = pruned_tokens[:, 0]
                    hash_codes = self.model.hashing_head(cls_token)
                else:
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
        """Lưu checkpoint với pruning state."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'metrics': metrics,
            'pruning_enabled': self.enable_pruning,
            'pruning_stats': self.pruning_stats
        }
        
        # Save attention pruner nếu có
        if self.attention_pruner is not None:
            checkpoint['attention_pruner_state'] = self.attention_pruner.state_dict()
        
        torch.save(checkpoint, path)
        print(f"[✓] Saved: {path}")


def train(args):
    """Main training với full research pipeline."""
    
    print("=" * 70)
    print("NWPU-RESISC45 TRAINING - Full Research Pipeline")
    print("=" * 70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nPhương pháp: ViT + Deep Hashing + DINOv2/v3 backbone + Token Pruning")
    
    if not check_gpu():
        print("[!] Tiếp tục với CPU (chậm hơn nhiều)")
    
    # Determine optimal image size
    # DINOv2 patch14: 224x224 = 256 tokens, 182x182 = 169 tokens (34% faster)
    if args.model == 'dinov3' and not args.full_resolution:
        image_size = args.image_size if args.image_size else 182  # Faster for DINOv2
        print(f"\n[!] DINOv2 optimization: Using {image_size}x{image_size} input")
        print(f"    Tokens: {(image_size // 14) ** 2} (vs 256 at 224x224)")
    else:
        image_size = args.image_size if args.image_size else 224
    
    # Data
    print(f"\n[Data] Loading NWPU-RESISC45...")
    train_loader, val_loader, test_loader, class_names = get_nwpu_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=image_size
    )
    num_classes = len(class_names)
    print(f"  Classes: {num_classes}")
    print(f"  Train: {len(train_loader.dataset)}")
    print(f"  Val: {len(val_loader.dataset)}")
    print(f"  Test: {len(test_loader.dataset)}")
    
    # Model selection
    print(f"\n[Model] Creating model...")
    
    if args.model == 'dinov3':
        # DINOv3Hashing - Full research model
        model_name_dinov3 = args.dinov3_variant
        print(f"  Type: DINOv3Hashing")
        print(f"  Backbone: {model_name_dinov3}")
        
        model = DINOv3Hashing(
            model_name=model_name_dinov3,
            pretrained=args.pretrained,
            hash_bit=args.hash_bit,
            freeze_backbone=args.freeze_backbone,
            use_gram_anchoring=args.gram_anchoring,
            hidden_dim=1024,
            dropout=0.5
        )
        
        # Apply torch.compile for PyTorch 2.0+ speedup (10-30% faster)
        if args.use_compile and hasattr(torch, 'compile'):
            print(f"  [Compile] Using torch.compile() for speedup...")
            model = torch.compile(model, mode='reduce-overhead')
    else:
        # Basic ViT_Hashing
        model_name = 'vit_base_patch32_224' if args.patch_size == 32 else 'vit_base_patch16_224'
        print(f"  Type: ViT_Hashing (basic)")
        print(f"  Backbone: {model_name}")
        
        model = ViT_Hashing(
            model_name=model_name,
            pretrained=True,
            hash_bit=args.hash_bit,
            weights_path=args.weights
        )
    
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Hash bit: {args.hash_bit}")
    print(f"  Parameters: {num_params/1e6:.1f}M (Trainable: {trainable_params/1e6:.1f}M)")
    
    # Loss: CSQLoss với 45 classes
    print(f"\n[Loss] CSQLoss")
    criterion = CSQLoss(
        hash_bit=args.hash_bit,
        num_classes=num_classes,  # 45 classes!
        lambda_q=args.lambda_q
    )
    print(f"  lambda_q (Quantization): {args.lambda_q}")
    
    # Optimizer với differential learning rates
    print(f"\n[Optimizer] AdamW với differential LR")
    optimizer = optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': args.lr * 0.1, 'name': 'backbone'},
        {'params': model.hashing_head.parameters(), 'lr': args.lr, 'name': 'hashing_head'}
    ], weight_decay=args.weight_decay)
    print(f"  Backbone LR: {args.lr * 0.1}")
    print(f"  Head LR: {args.lr}")
    print(f"  Weight decay: {args.weight_decay}")
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    
    # Trainer với pruning support
    print(f"\n[Trainer] Creating trainer...")
    trainer = NWPUTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        accumulation_steps=args.accumulation_steps,
        enable_pruning=args.enable_pruning,
        keep_ratio=args.keep_ratio,
        pruning_method=args.pruning_method
    )
    
    clear_memory()
    
    # Config summary
    print(f"\n[Config Summary]")
    print(f"  Model: {args.model}")
    print(f"  Image size: {image_size}x{image_size}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Accumulation: {args.accumulation_steps}")
    print(f"  Effective batch: {args.batch_size * args.accumulation_steps}")
    print(f"  Epochs: {args.epochs}")
    if args.model == 'dinov3':
        print(f"  torch.compile: {args.use_compile and hasattr(torch, 'compile')}")
        print(f"  Freeze backbone: {args.freeze_backbone}")
    print(f"  Token Pruning: {'Enabled' if args.enable_pruning else 'Disabled'}")
    if args.enable_pruning:
        print(f"    - Method: {args.pruning_method}")
        print(f"    - Keep ratio: {args.keep_ratio}")
    
    # Training
    print(f"\n{'=' * 70}")
    print("TRAINING")
    print("=" * 70)
    
    best_map = 0.0
    training_history = []
    
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
            
            training_history.append({
                'epoch': epoch,
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics['loss'],
                'val_mAP': val_metrics['mAP']
            })
            
            if val_metrics['mAP'] > best_map:
                best_map = val_metrics['mAP']
                save_path = os.path.join(args.save_dir, f'best_model_nwpu_{args.model}.pth')
                trainer.save(save_path, epoch, val_metrics)
        
        scheduler.step()
    
    # Final test
    print(f"\n{'=' * 70}")
    print("FINAL TEST")
    print("=" * 70)
    
    clear_memory()
    test_metrics = trainer.evaluate(test_loader, "Test")
    print(f"  Test mAP: {test_metrics['mAP']:.4f}")
    
    # Pruning analysis
    if trainer.pruning_stats:
        print(f"\n[Pruning Analysis]")
        avg_kept = np.mean([s['avg_kept_ratio'] for s in trainer.pruning_stats])
        print(f"  Average tokens kept: {avg_kept:.1%}")
        print(f"  FLOPs reduction estimate: {(1 - avg_kept**2):.1%}")
    
    # Summary
    print(f"\n{'=' * 70}")
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Model: {args.model}")
    print(f"  Token Pruning: {'Enabled' if args.enable_pruning else 'Disabled'}")
    print(f"  Best Val mAP: {best_map:.4f}")
    print(f"  Test mAP: {test_metrics['mAP']:.4f}")
    print(f"  Checkpoint: {os.path.join(args.save_dir, f'best_model_nwpu_{args.model}.pth')}")
    
    # Save training history
    import json
    history_path = os.path.join(args.save_dir, f'training_history_{args.model}.json')
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    print(f"  History: {history_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Train ViT/DINOv3 Hashing on NWPU-RESISC45 with Token Pruning',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data
    parser.add_argument('--data-dir', default='./data/archive/Dataset',
                       help='Path to NWPU-RESISC45 dataset')
    parser.add_argument('--save-dir', default='./checkpoints',
                       help='Directory to save checkpoints')
    
    # Model selection
    parser.add_argument('--model', type=str, default='vit', choices=['vit', 'dinov3'],
                       help='Model type: vit (ViT_Hashing baseline) or dinov3 (DINOv2 backbone)')
    parser.add_argument('--dinov3-variant', type=str, default='vit_small_patch14_dinov2.lvd142m',
                       choices=[
                           'vit_small_patch14_dinov2.lvd142m',
                           'vit_base_patch14_dinov2.lvd142m',
                           'vit_large_patch14_dinov2.lvd142m',
                           'vit_base_patch16_224'
                       ],
                       help='DINOv2 backbone variant (only for --model dinov3)')
    parser.add_argument('--pretrained', action='store_true', default=False,
                       help='Load pretrained DINOv2 weights (requires internet)')
    parser.add_argument('--freeze-backbone', action='store_true', default=False,
                       help='Freeze backbone weights during training')
    parser.add_argument('--gram-anchoring', action='store_true', default=False,
                       help='Enable Gram Anchoring (DINOv3 simulation)')
    
    # Hash settings
    parser.add_argument('--hash-bit', type=int, default=64, choices=[16, 32, 64, 128],
                       help='Number of hash bits')
    parser.add_argument('--patch-size', type=int, default=32, choices=[14, 16, 32],
                       help='Patch size for basic ViT model')
    parser.add_argument('--weights', type=str, default=None, 
                       help='Pretrained .npz weights path (for basic ViT)')
    
    # Token Pruning
    parser.add_argument('--enable-pruning', action='store_true', default=False,
                       help='Enable Token Pruning for efficiency')
    parser.add_argument('--keep-ratio', type=float, default=0.7,
                       help='Token keep ratio for pruning (0.5 = 50%% tokens kept)')
    parser.add_argument('--pruning-method', type=str, default='fisher',
                       choices=['fisher', 'attention'],
                       help='Pruning method: fisher (V-Pruner) or attention (learnable)')
    
    # Speed optimizations (DINOv2)
    parser.add_argument('--image-size', type=int, default=None,
                       help='Input image size (default: 224 for ViT, 182 for DINOv2)')
    parser.add_argument('--full-resolution', action='store_true', default=False,
                       help='Use full 224x224 resolution for DINOv2 (slower but may be more accurate)')
    parser.add_argument('--use-compile', action='store_true', default=False,
                       help='Use torch.compile() for speedup (requires PyTorch 2.0+)')
    
    # Loss
    parser.add_argument('--lambda-q', type=float, default=0.0001,
                       help='Quantization loss weight in CSQLoss')
    
    # Training
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size per step')
    parser.add_argument('--accumulation-steps', type=int, default=4,
                       help='Gradient accumulation steps')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate for hashing head')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                       help='Weight decay for AdamW optimizer')
    parser.add_argument('--num-workers', type=int, default=2,
                       help='DataLoader workers')
    parser.add_argument('--eval-every', type=int, default=5,
                       help='Evaluate every N epochs')
    
    # Quick test
    parser.add_argument('--quick', action='store_true',
                       help='Quick test mode (3 epochs)')
    parser.add_argument('--fast', action='store_true',
                       help='Fast DINOv2 mode: smaller image + compile + freeze backbone')
    
    args = parser.parse_args()
    
    if args.quick:
        args.epochs = 3
        args.batch_size = 4
        args.eval_every = 1
        print("[Quick mode] epochs=3, batch_size=4")
    
    # Fast mode for DINOv2 - enable all optimizations
    if args.fast and args.model == 'dinov3':
        args.image_size = args.image_size or 182
        args.use_compile = True
        args.freeze_backbone = True
        print("[Fast DINOv2 mode] image_size=182, compile=True, freeze_backbone=True")
        print("  Expected speedup: ~3-5x faster training")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    train(args)


if __name__ == "__main__":
    main()
