"""
Cross-Dataset Comparison: NWPU-RESISC45 vs NUS-WIDE
====================================================

So sánh hiệu suất retrieval giữa:
1. NWPU-RESISC45: Single-label, Remote Sensing (khó)
2. NUS-WIDE: Multi-label, Web Images (benchmark chuẩn)

Research Questions:
- RQ1: ViT vs DINOv2 performance on each dataset
- RQ2: How does domain affect retrieval performance?
- RQ3: Single-label vs Multi-label retrieval characteristics

Usage:
    # Train on NWPU
    python experiments/compare_datasets.py --dataset nwpu --model vit
    
    # Train on NUS-WIDE  
    python experiments/compare_datasets.py --dataset nuswide --model vit
    
    # Full comparison
    python experiments/compare_datasets.py --full-comparison
"""

import os
import sys
import argparse
import json
from datetime import datetime
from typing import Dict, List

import torch
import numpy as np

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Data loaders
from src.data.retrieval_protocol import get_nwpu_retrieval_loaders
from src.data.nuswide_loader import get_nuswide_dataloaders

# Models
from src.models.vit_hashing import ViT_Hashing
from src.models.dinov2_hashing import DINOv3Hashing

# Metrics
from src.utils.metrics import calculate_map, evaluate
from src.utils.metrics_multilabel import evaluate_multilabel

# Losses
from src.losses.csq_loss import CSQLoss
from src.losses.csq_multilabel_loss import get_multilabel_loss


def get_dataset_info():
    """Thông tin so sánh 2 datasets."""
    info = """
╔════════════════════════════════════════════════════════════════════════╗
║                    DATASET COMPARISON                                   ║
╠════════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║  Dataset          │ NWPU-RESISC45          │ NUS-WIDE                  ║
║  ─────────────────┼────────────────────────┼─────────────────────────  ║
║  Domain           │ Remote Sensing         │ Web Images (Flickr)       ║
║  Total Images     │ 31,500                 │ ~270,000                  ║
║  Classes/Labels   │ 45 (single-label)      │ 21 (multi-label)          ║
║  Label Type       │ Mutually exclusive     │ Co-occurring              ║
║  ─────────────────┼────────────────────────┼─────────────────────────  ║
║  Train Set        │ ~22,500                │ ~10,500                   ║
║  Query Set        │ ~2,250                 │ ~2,100                    ║
║  Database         │ ~6,750                 │ ~190,000                  ║
║  ─────────────────┼────────────────────────┼─────────────────────────  ║
║  Relevance        │ Same class             │ Share ≥1 label            ║
║  Difficulty       │ High (domain-specific) │ Medium (general)          ║
║  Benchmark Use    │ RSIR papers            │ Hashing papers            ║
║                                                                        ║
╚════════════════════════════════════════════════════════════════════════╝
    """
    return info


def train_and_evaluate(
    dataset: str,
    model_type: str,
    hash_bit: int = 64,
    epochs: int = 30,
    batch_size: int = 32,
    quick: bool = False,
    device: str = 'cuda',
) -> Dict:
    """
    Train và evaluate trên một dataset.
    
    Args:
        dataset: 'nwpu' hoặc 'nuswide'
        model_type: 'vit' hoặc 'dinov2'
        
    Returns:
        Dictionary với kết quả
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*60}")
    print(f"Training: {model_type.upper()} on {dataset.upper()}")
    print(f"{'='*60}")
    
    # Load data
    if dataset == 'nwpu':
        num_classes = 45
        train_loader, query_loader, db_loader, class_names = get_nwpu_retrieval_loaders(
            batch_size=batch_size,
            num_workers=4
        )
        criterion = CSQLoss(hash_bit=hash_bit, num_classes=num_classes)
        is_multilabel = False
        
    elif dataset == 'nuswide':
        num_classes = 21
        max_samples = 1000 if quick else None
        train_loader, query_loader, db_loader, class_names = get_nuswide_dataloaders(
            batch_size=batch_size,
            use_21_labels=True,
            max_train=max_samples,
            max_query=max_samples,
            max_db=max_samples
        )
        criterion = get_multilabel_loss('csq', hash_bit=hash_bit, num_classes=num_classes)
        is_multilabel = True
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    # Build model
    if model_type == 'vit':
        model = ViT_Hashing(hash_bit=hash_bit, num_classes=num_classes)
    else:
        model = DINOv3Hashing(hash_bit=hash_bit, num_classes=num_classes)
    
    model = model.to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if 'backbone' in n or 'vit' in n], 
         'lr': 1e-5},
        {'params': [p for n, p in model.named_parameters() if 'backbone' not in n and 'vit' not in n], 
         'lr': 1e-4},
    ], weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training
    if quick:
        epochs = 3
    
    best_map = 0.0
    history = []
    
    from tqdm import tqdm
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for imgs, labels in pbar:
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            hash_codes, _ = model(imgs)
            
            if is_multilabel:
                loss = criterion(hash_codes, labels)
            else:
                loss = criterion(hash_codes, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        scheduler.step()
        
        # Evaluate
        if is_multilabel:
            results = evaluate_multilabel(model, query_loader, db_loader, device)
            current_map = results['mAP']
        else:
            current_map = evaluate(model, query_loader, db_loader, device)
        
        history.append({
            'epoch': epoch,
            'loss': total_loss / len(train_loader),
            'mAP': current_map
        })
        
        if current_map > best_map:
            best_map = current_map
            # Save checkpoint
            checkpoint_path = f'./checkpoints/best_{dataset}_{model_type}_{hash_bit}bit.pth'
            os.makedirs('./checkpoints', exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'mAP': best_map,
                'epoch': epoch,
            }, checkpoint_path)
        
        print(f"Epoch {epoch}: Loss={total_loss/len(train_loader):.4f}, mAP={current_map:.4f}")
    
    return {
        'dataset': dataset,
        'model': model_type,
        'hash_bit': hash_bit,
        'best_mAP': best_map,
        'final_mAP': history[-1]['mAP'],
        'history': history,
    }


def run_full_comparison(quick: bool = False) -> Dict:
    """
    Chạy so sánh đầy đủ trên cả 2 datasets với cả 2 models.
    
    Returns:
        Comparison results
    """
    print(get_dataset_info())
    
    results = {}
    
    configs = [
        ('nwpu', 'vit'),
        ('nwpu', 'dinov2'),
        ('nuswide', 'vit'),
        ('nuswide', 'dinov2'),
    ]
    
    for dataset, model_type in configs:
        key = f"{dataset}_{model_type}"
        try:
            results[key] = train_and_evaluate(
                dataset=dataset,
                model_type=model_type,
                quick=quick
            )
        except Exception as e:
            print(f"[Error] {key}: {e}")
            results[key] = {'error': str(e)}
    
    # Print comparison table
    print("\n" + "=" * 70)
    print("FINAL COMPARISON RESULTS")
    print("=" * 70)
    print(f"{'Config':<25} {'Dataset':<15} {'Model':<10} {'mAP':<10}")
    print("-" * 70)
    
    for key, res in results.items():
        if 'error' not in res:
            print(f"{key:<25} {res['dataset']:<15} {res['model']:<10} {res['best_mAP']:.4f}")
    
    print("=" * 70)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = f'./results/comparison_{timestamp}.json'
    os.makedirs('./results', exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {results_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Dataset Comparison')
    
    parser.add_argument('--dataset', type=str, choices=['nwpu', 'nuswide'],
                        help='Dataset to use')
    parser.add_argument('--model', type=str, choices=['vit', 'dinov2'], default='vit',
                        help='Model to use')
    parser.add_argument('--hash-bit', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--quick', action='store_true', help='Quick test mode')
    parser.add_argument('--full-comparison', action='store_true',
                        help='Run full comparison on both datasets')
    parser.add_argument('--info', action='store_true', help='Show dataset info')
    
    args = parser.parse_args()
    
    if args.info:
        print(get_dataset_info())
        return
    
    if args.full_comparison:
        run_full_comparison(quick=args.quick)
    elif args.dataset:
        train_and_evaluate(
            dataset=args.dataset,
            model_type=args.model,
            hash_bit=args.hash_bit,
            epochs=args.epochs,
            batch_size=args.batch_size,
            quick=args.quick
        )
    else:
        print("Please specify --dataset or --full-comparison")
        print("Use --info to see dataset comparison")


if __name__ == '__main__':
    main()
