"""
Evaluate NUS-WIDE Checkpoint - mAP Calculator
==============================================

Đánh giá mAP cho các checkpoints NUS-WIDE đã train với các hash bits khác nhau.

Usage:
    # Evaluate 64-bit model
    python scripts/evaluate_nuswide_checkpoint.py \
        --checkpoint checkpoints/best_model_nuswide_vit_64bit.pth
    
    # Evaluate 128-bit model
    python scripts/evaluate_nuswide_checkpoint.py \
        --checkpoint checkpoints/best_model_nuswide_vit_128bit.pth
    
    # Evaluate all models in checkpoints dir
    python scripts/evaluate_nuswide_checkpoint.py --all
    
    # Quick sanity check (1000 DB images)
    python scripts/evaluate_nuswide_checkpoint.py \
        --checkpoint checkpoints/best_model_nuswide_vit_64bit.pth --quick
"""

import os
import sys
import argparse
import glob
from datetime import datetime
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.nuswide_loader import NUSWIDEPreprocessedDataset
from src.models.vit_hashing import ViT_Hashing
from src.models.dinov2_hashing import DINOv3Hashing


# ===========================================================================
# Metrics
# ===========================================================================

def hamming_dist(qB: np.ndarray, rB: np.ndarray) -> np.ndarray:
    """Hamming distance matrix. Inputs: {-1,+1} codes."""
    return 0.5 * (qB.shape[1] - qB @ rB.T)


def calc_map(qB, rB, qL, rL, top_k=None):
    """
    Mean Average Precision for multi-label retrieval.
    Relevance: share >= 1 label.
    """
    dist = hamming_dist(qB, rB)  # [Q, DB]
    S = (qL @ rL.T) > 0  # [Q, DB] boolean

    aps = []
    for i in range(len(qB)):
        gnd = S[i].astype(np.float32)
        n_pos = gnd.sum()
        if n_pos == 0:
            continue
        rank = np.argsort(dist[i])
        if top_k is not None:
            rank = rank[:top_k]
        gnd_sorted = gnd[rank]
        cumsum = np.cumsum(gnd_sorted)
        prec = cumsum / (np.arange(len(gnd_sorted)) + 1.0)
        ap = (prec * gnd_sorted).sum() / n_pos
        aps.append(ap)

    return float(np.mean(aps)) if aps else 0.0


def precision_at_k(qB, rB, qL, rL, k=100):
    """Precision at top-k."""
    dist = hamming_dist(qB, rB)
    S = (qL @ rL.T) > 0
    precs = []
    for i in range(len(qB)):
        rank = np.argsort(dist[i])[:k]
        precs.append(S[i][rank].mean())
    return float(np.mean(precs))


# ===========================================================================
# Model Loading
# ===========================================================================

def load_model(checkpoint_path: str, device: torch.device):
    """Load checkpoint and infer model type."""
    print(f"\n[*] Loading: {checkpoint_path}")
    
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt['model_state_dict']
    
    # Auto-detect model type and hash bits
    has_layers = any('hashing_head.layers.' in k for k in state)
    
    if has_layers:
        model_type = 'dinov2'
        hash_bit = state['hashing_head.layers.3.weight'].shape[0]
    else:
        model_type = 'vit'
        hash_bit = state['hashing_head.3.weight'].shape[0]
    
    # Get num_classes if available
    num_classes = ckpt.get('num_classes', 21)
    
    print(f"    Model: {model_type}")
    print(f"    Hash bits: {hash_bit}")
    print(f"    Classes: {num_classes}")
    print(f"    Epoch: {ckpt.get('epoch', '?')}")
    print(f"    Train mAP: {ckpt.get('mAP', 'N/A')}")
    
    # Build model
    if model_type == 'vit':
        model = ViT_Hashing(hash_bit=hash_bit, num_classes=num_classes, pretrained=False)
    else:
        model = DINOv3Hashing(hash_bit=hash_bit, num_classes=num_classes)
    
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    
    return model, {'hash_bit': hash_bit, 'model_type': model_type, 'num_classes': num_classes}


# ===========================================================================
# Feature Extraction
# ===========================================================================

@torch.no_grad()
def extract_codes(model, loader, device, desc="Extracting"):
    """Extract binary hash codes and labels."""
    model.eval()
    codes, labels = [], []
    
    for imgs, lbl in tqdm(loader, desc=desc, leave=False):
        h, _ = model(imgs.to(device))
        codes.append(torch.sign(h).cpu().numpy())
        labels.append(lbl.numpy())
    
    return np.concatenate(codes), np.concatenate(labels)


# ===========================================================================
# Evaluation
# ===========================================================================

def evaluate_checkpoint(checkpoint_path: str, data_dir: str, batch_size: int,
                        num_workers: int, quick: bool, device: torch.device) -> dict:
    """Evaluate a single checkpoint."""
    
    # Load model
    model, info = load_model(checkpoint_path, device)
    
    # Load data
    query_ds = NUSWIDEPreprocessedDataset(data_dir, 'query')
    db_ds = NUSWIDEPreprocessedDataset(data_dir, 'database')
    
    if quick:
        db_ds = Subset(db_ds, list(range(min(1000, len(db_ds)))))
        print(f"    [Quick mode] Database limited to {len(db_ds)} images")
    
    query_loader = DataLoader(query_ds, batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    db_loader = DataLoader(db_ds, batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True)
    
    # Extract codes
    print("\n[*] Extracting query codes...")
    qB, qL = extract_codes(model, query_loader, device, "Query")
    print(f"    Shape: {qB.shape}")
    
    print("[*] Extracting database codes...")
    rB, rL = extract_codes(model, db_loader, device, "Database")
    print(f"    Shape: {rB.shape}")
    
    # Calculate metrics
    print("\n[*] Computing metrics...")
    
    results = {
        'checkpoint': checkpoint_path,
        'hash_bit': info['hash_bit'],
        'model_type': info['model_type'],
        'query_size': len(qB),
        'database_size': len(rB),
        'mAP@ALL': calc_map(qB, rB, qL, rL),
        'mAP@1000': calc_map(qB, rB, qL, rL, top_k=1000),
        'mAP@5000': calc_map(qB, rB, qL, rL, top_k=5000),
        'P@10': precision_at_k(qB, rB, qL, rL, k=10),
        'P@50': precision_at_k(qB, rB, qL, rL, k=50),
        'P@100': precision_at_k(qB, rB, qL, rL, k=100),
        'evaluated_at': datetime.now().isoformat(),
    }
    
    return results


def print_results(results: dict):
    """Pretty print evaluation results."""
    print("\n" + "=" * 60)
    print(f"EVALUATION RESULTS: {results['hash_bit']}-bit {results['model_type'].upper()}")
    print("=" * 60)
    print(f"  Checkpoint: {results['checkpoint']}")
    print(f"  Query: {results['query_size']} | Database: {results['database_size']}")
    print("-" * 60)
    print(f"  mAP@ALL  : {results['mAP@ALL']:.4f}")
    print(f"  mAP@1000 : {results['mAP@1000']:.4f}")
    print(f"  mAP@5000 : {results['mAP@5000']:.4f}")
    print("-" * 60)
    print(f"  P@10     : {results['P@10']:.4f}")
    print(f"  P@50     : {results['P@50']:.4f}")
    print(f"  P@100    : {results['P@100']:.4f}")
    print("=" * 60)
    
    # Reference comparison
    print("\n[Reference — CSQ paper on NUS-WIDE]")
    print("  CSQ (ResNet-50, 64-bit) : mAP@ALL = 0.748")
    print("  CSQ (ResNet-50, 128-bit): mAP@ALL = ~0.76")
    print("  HashNet (AlexNet, 64-bit): mAP@ALL = 0.618")


def find_nuswide_checkpoints(checkpoint_dir: str) -> list:
    """Find all NUS-WIDE checkpoints in directory."""
    patterns = [
        os.path.join(checkpoint_dir, "*nuswide*.pth"),
        os.path.join(checkpoint_dir, "*nus_wide*.pth"),
        os.path.join(checkpoint_dir, "*nus-wide*.pth"),
    ]
    
    checkpoints = []
    for pattern in patterns:
        checkpoints.extend(glob.glob(pattern))
    
    return sorted(set(checkpoints))


def main():
    parser = argparse.ArgumentParser(description='Evaluate NUS-WIDE checkpoints')
    
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to single checkpoint')
    parser.add_argument('--all', action='store_true',
                        help='Evaluate all NUS-WIDE checkpoints in checkpoint dir')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                        help='Directory containing checkpoints')
    parser.add_argument('--data-dir', type=str, default='./src/data/archive/NUS-WIDE',
                        help='NUS-WIDE data directory')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode with 1000 DB images')
    parser.add_argument('--save-results', type=str, default=None,
                        help='Save results to JSON file')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] {device}")
    
    # Collect checkpoints to evaluate
    checkpoints = []
    
    if args.checkpoint:
        checkpoints = [args.checkpoint]
    elif args.all:
        checkpoints = find_nuswide_checkpoints(args.checkpoint_dir)
        if not checkpoints:
            print(f"[!] No NUS-WIDE checkpoints found in {args.checkpoint_dir}")
            return
        print(f"[*] Found {len(checkpoints)} checkpoints:")
        for ckpt in checkpoints:
            print(f"    - {ckpt}")
    else:
        parser.print_help()
        return
    
    # Evaluate each checkpoint
    all_results = []
    
    for ckpt in checkpoints:
        if not os.path.exists(ckpt):
            print(f"[!] Checkpoint not found: {ckpt}")
            continue
        
        results = evaluate_checkpoint(
            ckpt, args.data_dir, args.batch_size,
            args.num_workers, args.quick, device
        )
        all_results.append(results)
        print_results(results)
    
    # Summary table if multiple checkpoints
    if len(all_results) > 1:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"{'Checkpoint':<45} {'Bits':>5} {'mAP@ALL':>10} {'P@100':>8}")
        print("-" * 70)
        for r in all_results:
            name = os.path.basename(r['checkpoint'])
            print(f"{name:<45} {r['hash_bit']:>5} {r['mAP@ALL']:>10.4f} {r['P@100']:>8.4f}")
        print("=" * 70)
    
    # Save results
    if args.save_results:
        with open(args.save_results, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n[✓] Results saved to {args.save_results}")


if __name__ == '__main__':
    main()
