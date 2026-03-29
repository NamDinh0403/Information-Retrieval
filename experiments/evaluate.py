"""
Evaluation Script for NWPU-RESISC45 Dataset
============================================
Đánh giá model ViT Hashing trên bộ dữ liệu NWPU-RESISC45.

Metrics:
    - mAP (mean Average Precision)
    - Precision@K
    - Recall@K
    - F1@K
    - Precision-Recall curve

Usage:
    python evaluate_nwpu.py --checkpoint checkpoints/best_model.pth
    python evaluate_nwpu.py --checkpoint checkpoints/best_model.pth --top-k 100
"""

import os
import sys
import argparse
import time
from typing import Dict, List, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import numpy as np

# Optional matplotlib
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.vit_hashing import ViT_Hashing
from src.models.dinov2_hashing import DINOv3Hashing


# ============================================================
# DATA LOADING
# ============================================================

def get_nwpu_loaders(
    data_dir: str = './data/archive/Dataset',
    batch_size: int = 32,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Load NWPU-RESISC45 dataset.
    
    Returns:
        train_loader, test_loader, class_names
    """
    # Transform cho ViT (224x224)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Dataset paths (nested structure)
    train_path = os.path.join(data_dir, 'train', 'train')
    test_path = os.path.join(data_dir, 'test', 'test')
    
    # Load datasets
    train_dataset = ImageFolder(train_path, transform=transform)
    test_dataset = ImageFolder(test_path, transform=transform)
    
    class_names = train_dataset.classes
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False
    )
    
    return train_loader, test_loader, class_names


# ============================================================
# HASH CODE EXTRACTION
# ============================================================

@torch.no_grad()
def extract_hash_codes(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract hash codes từ dataloader.
    
    Returns:
        hash_codes: [N, hash_bit] continuous values
        labels: [N]
    """
    model.eval()
    
    all_hash_codes = []
    all_labels = []
    
    print(f"  Extracting hash codes from {len(dataloader.dataset)} images...")
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        
        hash_codes, _ = model(images)
        
        all_hash_codes.append(hash_codes.cpu())
        all_labels.append(labels)
        
        if (batch_idx + 1) % 50 == 0:
            print(f"    Processed {(batch_idx + 1) * dataloader.batch_size}/{len(dataloader.dataset)}")
    
    all_hash_codes = torch.cat(all_hash_codes, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    return all_hash_codes, all_labels


# ============================================================
# METRICS COMPUTATION
# ============================================================

def compute_hamming_distance(query: torch.Tensor, database: torch.Tensor) -> torch.Tensor:
    """
    Compute Hamming distance giữa query và database.
    
    Args:
        query: [N_q, hash_bit] binary codes
        database: [N_db, hash_bit] binary codes
    
    Returns:
        distances: [N_q, N_db]
    """
    # Convert to binary: sign(x) -> {-1, 1} -> (1 - x) / 2 -> {0, 1}
    q_binary = (1 - query) / 2  # {-1, 1} -> {1, 0}
    db_binary = (1 - database) / 2
    
    # Hamming distance = number of different bits
    # dist = sum(q XOR db) = sum(q) + sum(db) - 2 * sum(q AND db)
    # For {0, 1}: q XOR db = q + db - 2*q*db
    distances = q_binary.sum(dim=1, keepdim=True) + db_binary.sum(dim=1) - 2 * (q_binary @ db_binary.T)
    
    return distances


def compute_map(
    query_codes: torch.Tensor,
    query_labels: torch.Tensor,
    database_codes: torch.Tensor,
    database_labels: torch.Tensor,
    top_k: int = -1,
    query_batch_size: int = 256
) -> float:
    """
    Compute mean Average Precision (mAP).
    Processes queries in batches to avoid OOM on large datasets.

    Args:
        query_codes: [N_q, hash_bit]
        query_labels: [N_q]
        database_codes: [N_db, hash_bit]
        database_labels: [N_db]
        top_k: -1 for all
        query_batch_size: number of queries processed at once

    Returns:
        mAP score
    """
    query_binary    = torch.sign(query_codes)
    database_binary = torch.sign(database_codes)

    aps = []
    n_queries = len(query_labels)

    for start in range(0, n_queries, query_batch_size):
        end = min(start + query_batch_size, n_queries)
        q_batch  = query_binary[start:end]    # [B, hash_bit]
        ql_batch = query_labels[start:end]    # [B]

        # Hamming distance for this batch only: [B, N_db]
        dist_batch = compute_hamming_distance(q_batch, database_binary)
        sorted_indices = torch.argsort(dist_batch, dim=1)

        for i in range(len(ql_batch)):
            sorted_labels = database_labels[sorted_indices[i]]
            relevant = (sorted_labels == ql_batch[i]).float()

            if top_k > 0:
                relevant = relevant[:top_k]

            num_relevant = relevant.sum().item()
            if num_relevant == 0:
                continue

            cumsum = torch.cumsum(relevant, dim=0)
            prec   = cumsum / torch.arange(1, len(relevant) + 1, dtype=torch.float)
            ap     = (prec * relevant).sum() / num_relevant
            aps.append(ap.item())

    return np.mean(aps) if aps else 0.0


def compute_precision_recall_at_k(
    query_codes: torch.Tensor,
    query_labels: torch.Tensor,
    database_codes: torch.Tensor,
    database_labels: torch.Tensor,
    k_values: List[int] = [1, 5, 10, 20, 50, 100],
    query_batch_size: int = 256
) -> Dict[str, Dict[int, float]]:
    """
    Compute Precision@K và Recall@K.
    Processes queries in batches to avoid OOM.
    """
    query_binary    = torch.sign(query_codes)
    database_binary = torch.sign(database_codes)
    max_k = max(k_values)

    precision_at_k = defaultdict(list)
    recall_at_k    = defaultdict(list)
    n_queries = len(query_labels)

    for start in range(0, n_queries, query_batch_size):
        end = min(start + query_batch_size, n_queries)
        q_batch  = query_binary[start:end]
        ql_batch = query_labels[start:end]

        dist_batch     = compute_hamming_distance(q_batch, database_binary)
        sorted_indices = torch.argsort(dist_batch, dim=1)

        for i in range(len(ql_batch)):
            sorted_labels  = database_labels[sorted_indices[i]]
            relevant       = (sorted_labels == ql_batch[i]).float()
            total_relevant = (database_labels == ql_batch[i]).sum().item()

            for k in k_values:
                if k > len(relevant):
                    continue
                relevant_in_k = relevant[:k].sum().item()
                precision_at_k[k].append(relevant_in_k / k)
                if total_relevant > 0:
                    recall_at_k[k].append(relevant_in_k / total_relevant)

    return {
        'precision': {k: np.mean(v) for k, v in precision_at_k.items()},
        'recall':    {k: np.mean(v) for k, v in recall_at_k.items()},
    }


def compute_precision_recall_curve(
    query_codes: torch.Tensor,
    query_labels: torch.Tensor,
    database_codes: torch.Tensor,
    database_labels: torch.Tensor,
    num_points: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Precision-Recall curve.
    """
    # Binary codes
    query_binary = torch.sign(query_codes)
    database_binary = torch.sign(database_codes)
    
    # Hamming distance
    hamming_dist = compute_hamming_distance(query_binary, database_binary)
    
    # Sort
    sorted_indices = torch.argsort(hamming_dist, dim=1)
    
    # Sample k values
    max_k = len(database_labels)
    k_values = np.linspace(1, max_k, num_points, dtype=int)
    
    precisions = []
    recalls = []
    
    for k in k_values:
        prec_list = []
        rec_list = []
        
        for i in range(min(len(query_labels), 500)):  # Sample 500 queries
            sorted_labels = database_labels[sorted_indices[i]]
            relevant = (sorted_labels == query_labels[i]).float()
            total_relevant = (database_labels == query_labels[i]).sum().item()
            
            relevant_in_k = relevant[:k].sum().item()
            
            prec_list.append(relevant_in_k / k)
            if total_relevant > 0:
                rec_list.append(relevant_in_k / total_relevant)
        
        precisions.append(np.mean(prec_list))
        recalls.append(np.mean(rec_list) if rec_list else 0)
    
    return np.array(recalls), np.array(precisions)


def compute_hash_quality_metrics(hash_codes: torch.Tensor) -> Dict[str, float]:
    """
    Compute các metrics về chất lượng hash codes.
    """
    binary_codes = torch.sign(hash_codes)
    
    # Bit balance: mỗi bit nên có ~50% là 1 và 50% là -1
    bit_mean = binary_codes.mean(dim=0)
    bit_balance = 1 - torch.abs(bit_mean).mean().item()  # 1 = perfect balance
    
    # Bit independence: các bit nên độc lập
    # Correlation matrix
    codes_centered = binary_codes - binary_codes.mean(dim=0)
    corr_matrix = (codes_centered.T @ codes_centered) / len(codes_centered)
    
    # Off-diagonal mean (should be close to 0)
    mask = ~torch.eye(corr_matrix.size(0), dtype=torch.bool)
    bit_independence = 1 - torch.abs(corr_matrix[mask]).mean().item()
    
    # Quantization quality: hash_codes nên gần ±1
    quant_quality = (torch.abs(hash_codes) > 0.5).float().mean().item()
    
    return {
        'bit_balance': bit_balance,
        'bit_independence': bit_independence,
        'quantization_quality': quant_quality
    }


# ============================================================
# VISUALIZATION
# ============================================================

def plot_results(
    pr_curve: Tuple[np.ndarray, np.ndarray],
    precision_at_k: Dict[int, float],
    recall_at_k: Dict[int, float],
    save_path: str = 'evaluation_results.png'
):
    """
    Plot evaluation results.
    """
    if not HAS_MATPLOTLIB:
        print("[!] matplotlib not available")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Precision-Recall curve
    recalls, precisions = pr_curve
    axes[0].plot(recalls, precisions, 'b-', linewidth=2)
    axes[0].set_xlabel('Recall')
    axes[0].set_ylabel('Precision')
    axes[0].set_title('Precision-Recall Curve')
    axes[0].grid(True)
    axes[0].set_xlim([0, 1])
    axes[0].set_ylim([0, 1])
    
    # Precision@K
    k_vals = sorted(precision_at_k.keys())
    prec_vals = [precision_at_k[k] for k in k_vals]
    axes[1].bar(range(len(k_vals)), prec_vals, color='steelblue')
    axes[1].set_xticks(range(len(k_vals)))
    axes[1].set_xticklabels([f'P@{k}' for k in k_vals])
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision@K')
    axes[1].set_ylim([0, 1])
    for i, v in enumerate(prec_vals):
        axes[1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=8)
    
    # Recall@K
    rec_vals = [recall_at_k[k] for k in k_vals]
    axes[2].bar(range(len(k_vals)), rec_vals, color='darkorange')
    axes[2].set_xticks(range(len(k_vals)))
    axes[2].set_xticklabels([f'R@{k}' for k in k_vals])
    axes[2].set_ylabel('Recall')
    axes[2].set_title('Recall@K')
    axes[2].set_ylim([0, 1])
    for i, v in enumerate(rec_vals):
        axes[2].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[✓] Saved plot: {save_path}")


# ============================================================
# MAIN EVALUATION
# ============================================================

def evaluate(args):
    """Main evaluation function."""
    
    print("=" * 60)
    print("NWPU-RESISC45 EVALUATION")
    print("=" * 60)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[Device] {device}")
    
    # Load model — auto-detect type and hash_bit from state_dict
    print(f"\n[Model] Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = checkpoint['model_state_dict']
    
    has_layers_prefix = any('hashing_head.layers.' in k for k in state)
    
    if has_layers_prefix:
        model_type = 'dinov3'
        hash_bit = state['hashing_head.layers.3.weight'].shape[0]
        embed_dim = state['hashing_head.layers.1.weight'].shape[1]
        hidden_dim = state['hashing_head.layers.1.weight'].shape[0]
        dinov2_map = {384: 'vit_small_patch14_dinov2.lvd142m',
                      768: 'vit_base_patch14_dinov2.lvd142m',
                      1024: 'vit_large_patch14_dinov2.lvd142m'}
        model_name = dinov2_map.get(embed_dim, 'vit_small_patch14_dinov2.lvd142m')
        model = DINOv3Hashing(model_name=model_name, pretrained=False,
                              hash_bit=hash_bit, hidden_dim=hidden_dim)
    else:
        model_type = 'vit'
        hash_bit = state['hashing_head.3.weight'].shape[0]
        pos_len = state['backbone.pos_embed'].shape[1]
        model_name = 'vit_base_patch32_224' if pos_len == 50 else 'vit_base_patch16_224'
        model = ViT_Hashing(model_name=model_name, pretrained=False, hash_bit=hash_bit)
    
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    
    print(f"  Model: {model_name} ({model_type})")
    print(f"  Hash bit: {hash_bit}")
    if 'metrics' in checkpoint:
        print(f"  Training mAP: {checkpoint['metrics'].get('mAP', 'N/A')}")
    
    # Load data
    print(f"\n[Data] Loading NWPU-RESISC45...")
    train_loader, test_loader, class_names = get_nwpu_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    print(f"  Train (database): {len(train_loader.dataset)} images")
    print(f"  Test (query): {len(test_loader.dataset)} images")
    print(f"  Classes: {len(class_names)}")
    
    # Extract hash codes
    print(f"\n[Extraction]")
    print("  Database (train):")
    start_time = time.time()
    db_codes, db_labels = extract_hash_codes(model, train_loader, device)
    db_time = time.time() - start_time
    print(f"    Time: {db_time:.1f}s | Shape: {db_codes.shape}")
    
    print("  Query (test):")
    start_time = time.time()
    query_codes, query_labels = extract_hash_codes(model, test_loader, device)
    query_time = time.time() - start_time
    print(f"    Time: {query_time:.1f}s | Shape: {query_codes.shape}")
    
    # Compute metrics
    print(f"\n{'=' * 60}")
    print("METRICS")
    print("=" * 60)
    
    # mAP
    print("\n[mAP]")
    start_time = time.time()
    map_all = compute_map(query_codes, query_labels, db_codes, db_labels, top_k=-1)
    map_100 = compute_map(query_codes, query_labels, db_codes, db_labels, top_k=100)
    map_500 = compute_map(query_codes, query_labels, db_codes, db_labels, top_k=500)
    print(f"  mAP@ALL: {map_all:.4f}")
    print(f"  mAP@100: {map_100:.4f}")
    print(f"  mAP@500: {map_500:.4f}")
    print(f"  Time: {time.time() - start_time:.1f}s")
    
    # Precision/Recall@K
    print("\n[Precision/Recall@K]")
    k_values = [1, 5, 10, 20, 50, 100]
    pr_at_k = compute_precision_recall_at_k(
        query_codes, query_labels, db_codes, db_labels, k_values
    )
    
    print("  K\tP@K\tR@K\tF1@K")
    print("  " + "-" * 35)
    for k in k_values:
        p = pr_at_k['precision'].get(k, 0)
        r = pr_at_k['recall'].get(k, 0)
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        print(f"  {k}\t{p:.4f}\t{r:.4f}\t{f1:.4f}")
    
    # Hash quality
    print("\n[Hash Quality]")
    quality_train = compute_hash_quality_metrics(db_codes)
    quality_test = compute_hash_quality_metrics(query_codes)
    print(f"  Bit Balance (train): {quality_train['bit_balance']:.4f}")
    print(f"  Bit Balance (test):  {quality_test['bit_balance']:.4f}")
    print(f"  Bit Independence:    {quality_train['bit_independence']:.4f}")
    print(f"  Quantization:        {quality_train['quantization_quality']:.4f}")
    
    # Plot
    if args.plot:
        if not HAS_MATPLOTLIB:
            print("\n[!] matplotlib not installed. Skipping plots.")
            print("    Install: pip install matplotlib")
        else:
            print("\n[Plotting]")
            pr_curve = compute_precision_recall_curve(
                query_codes, query_labels, db_codes, db_labels
            )
            plot_results(
                pr_curve, 
                pr_at_k['precision'], 
                pr_at_k['recall'],
                args.plot_path
            )
    
    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    print(f"  Dataset: NWPU-RESISC45 ({len(class_names)} classes)")
    print(f"  Hash bit: {hash_bit}")
    print(f"  mAP@ALL: {map_all:.4f}")
    print(f"  mAP@100: {map_100:.4f}")
    print(f"  P@10: {pr_at_k['precision'].get(10, 0):.4f}")
    print(f"  R@100: {pr_at_k['recall'].get(100, 0):.4f}")
    
    # Đánh giá
    print(f"\n[Đánh giá]")
    if map_all >= 0.7:
        print("  ★★★ Xuất sắc - Model hoạt động rất tốt trên remote sensing!")
    elif map_all >= 0.5:
        print("  ★★☆ Khá tốt - Có thể cải thiện bằng fine-tune thêm")
    elif map_all >= 0.3:
        print("  ★☆☆ Trung bình - Cần fine-tune trên NWPU để cải thiện")
    else:
        print("  ☆☆☆ Chưa tốt - Model chưa học được features của remote sensing")
    
    return {
        'mAP_all': map_all,
        'mAP_100': map_100,
        'mAP_500': map_500,
        'precision_at_k': pr_at_k['precision'],
        'recall_at_k': pr_at_k['recall'],
        'hash_quality': quality_train
    }


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate ViT Hashing on NWPU-RESISC45',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_model.pth',
                       help='Path to model checkpoint')
    
    # Data
    parser.add_argument('--data-dir', type=str, default='./data/archive/Dataset',
                       help='NWPU-RESISC45 data directory')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=0, help='DataLoader workers')
    
    # Output
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--plot-path', type=str, default='evaluation_results.png',
                       help='Plot output path')
    
    args = parser.parse_args()
    
    evaluate(args)


if __name__ == "__main__":
    main()
