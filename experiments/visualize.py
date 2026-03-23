"""
Visualization and Analysis Script
=================================
Phân tích kết quả và visualization cho đồ án Master's.

Features:
    - Attention map visualization
    - t-SNE của hash codes
    - Confusion matrix
    - Failure analysis
    - Retrieval examples

Usage:
    python visualize_analysis.py --checkpoint ./checkpoints/best_model_nwpu_vit.pth
    python visualize_analysis.py --checkpoint ./checkpoints/best_model_nwpu_vit.pth --analyze-failures
    python visualize_analysis.py --compare ./checkpoints/vit.pth ./checkpoints/dinov3.pth
"""

import os
import sys
import argparse
import json
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.vit_hashing import ViT_Hashing
from src.models.dinov2_hashing import DINOv3Hashing


def load_model(checkpoint_path: str, device: torch.device) -> Tuple[nn.Module, Dict]:
    """Load model từ checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Detect model type from checkpoint
    state_dict = checkpoint['model_state_dict']
    
    # Check embedding dimension to determine model type
    if 'hashing_head.layers.1.weight' in state_dict:
        # DINOv3Hashing
        embed_dim = state_dict['hashing_head.layers.1.weight'].shape[1]
        hash_bit = state_dict['hashing_head.layers.3.weight'].shape[0]
        model = DINOv3Hashing(pretrained=False, hash_bit=hash_bit)
        model_type = 'dinov3'
    else:
        # ViT_Hashing
        embed_dim = state_dict['hashing_head.1.weight'].shape[1]
        hash_bit = state_dict['hashing_head.4.weight'].shape[0]
        model = ViT_Hashing(pretrained=False, hash_bit=hash_bit)
        model_type = 'vit'
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    return model, {'type': model_type, 'hash_bit': hash_bit, 'metrics': checkpoint.get('metrics', {})}


def get_test_loader(data_dir: str, batch_size: int = 32) -> Tuple[DataLoader, List[str]]:
    """Load test data."""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_path = os.path.join(data_dir, 'test', 'test')
    dataset = ImageFolder(test_path, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return loader, dataset.classes


@torch.no_grad()
def extract_features(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract hash codes và features từ model."""
    all_hashes = []
    all_features = []
    all_labels = []
    
    for images, labels in loader:
        images = images.to(device)
        
        hash_codes, features = model(images)
        binary_codes = torch.sign(hash_codes)
        
        all_hashes.append(binary_codes.cpu().numpy())
        all_features.append(features.cpu().numpy())
        all_labels.append(labels.numpy())
    
    return (
        np.concatenate(all_hashes),
        np.concatenate(all_features),
        np.concatenate(all_labels)
    )


def compute_retrieval_results(
    hash_codes: np.ndarray, 
    labels: np.ndarray,
    top_k: int = 10
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute retrieval results cho tất cả queries.
    
    Returns:
        predictions: [N, K] - predicted labels cho top-K
        distances: [N, K] - Hamming distances
        mAP: Mean Average Precision
    """
    n = len(hash_codes)
    
    # Hamming distance matrix
    # d(a,b) = (K - a·b) / 2 for binary codes in {-1, +1}
    sim = hash_codes @ hash_codes.T
    K = hash_codes.shape[1]
    dist = (K - sim) / 2
    
    # Set self-distance to infinity
    np.fill_diagonal(dist, np.inf)
    
    # Get top-K indices
    top_k_indices = np.argsort(dist, axis=1)[:, :top_k]
    
    # Get predictions and distances
    predictions = labels[top_k_indices]
    distances = np.take_along_axis(dist, top_k_indices, axis=1)
    
    # Compute mAP
    aps = []
    for i in range(n):
        query_label = labels[i]
        sorted_labels = labels[np.argsort(dist[i])]
        
        relevant = (sorted_labels == query_label).astype(float)
        if relevant.sum() == 0:
            continue
            
        # Remove self
        relevant = relevant[1:]
        
        cumsum = np.cumsum(relevant)
        precision_at_k = cumsum / np.arange(1, len(relevant) + 1)
        ap = (precision_at_k * relevant).sum() / relevant.sum()
        aps.append(ap)
    
    mAP = np.mean(aps) if aps else 0.0
    
    return predictions, distances, mAP


def plot_confusion_matrix(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_names: List[str],
    save_path: str
):
    """Plot confusion matrix từ top-1 retrieval predictions."""
    # Top-1 predictions
    top1_preds = predictions[:, 0]
    
    cm = confusion_matrix(labels, top1_preds)
    
    # Normalize
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(20, 16))
    sns.heatmap(
        cm_normalized, 
        annot=False,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix (Top-1 Retrieval)', fontsize=14)
    plt.xlabel('Retrieved Class')
    plt.ylabel('Query Class')
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(fontsize=6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    print(f"[✓] Saved confusion matrix: {save_path}")


def find_confused_pairs(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_names: List[str],
    top_n: int = 10
) -> List[Tuple[str, str, int]]:
    """Find top confused class pairs."""
    top1_preds = predictions[:, 0]
    
    # Count confusions
    confusion_counts = {}
    for true_label, pred_label in zip(labels, top1_preds):
        if true_label != pred_label:
            pair = (min(true_label, pred_label), max(true_label, pred_label))
            confusion_counts[pair] = confusion_counts.get(pair, 0) + 1
    
    # Sort by count
    sorted_pairs = sorted(confusion_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Convert to class names
    result = []
    for (c1, c2), count in sorted_pairs[:top_n]:
        result.append((class_names[c1], class_names[c2], count))
    
    return result


def plot_tsne(
    features: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    save_path: str,
    n_samples: int = 2000
):
    """Plot t-SNE visualization of features/hash codes."""
    # Subsample if too many
    if len(features) > n_samples:
        indices = np.random.choice(len(features), n_samples, replace=False)
        features = features[indices]
        labels = labels[indices]
    
    print(f"  Computing t-SNE for {len(features)} samples...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features)
    
    plt.figure(figsize=(14, 10))
    
    # Plot each class with different color
    unique_labels = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(
            features_2d[mask, 0], 
            features_2d[mask, 1],
            c=[colors[i]],
            label=class_names[label] if i < 20 else None,  # Only show first 20 in legend
            alpha=0.6,
            s=10
        )
    
    plt.title('t-SNE Visualization of Hash Codes', fontsize=14)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    print(f"[✓] Saved t-SNE plot: {save_path}")


def plot_per_class_map(
    labels: np.ndarray,
    hash_codes: np.ndarray,
    class_names: List[str],
    save_path: str
):
    """Plot per-class mAP bar chart."""
    n_classes = len(class_names)
    class_aps = []
    
    # Compute similarity matrix
    sim = hash_codes @ hash_codes.T
    K = hash_codes.shape[1]
    dist = (K - sim) / 2
    np.fill_diagonal(dist, np.inf)
    
    for c in range(n_classes):
        query_mask = labels == c
        query_indices = np.where(query_mask)[0]
        
        aps = []
        for qi in query_indices:
            sorted_indices = np.argsort(dist[qi])
            sorted_labels = labels[sorted_indices]
            
            relevant = (sorted_labels == c).astype(float)
            relevant = relevant[1:]  # Remove self
            
            if relevant.sum() == 0:
                continue
            
            cumsum = np.cumsum(relevant)
            precision_at_k = cumsum / np.arange(1, len(relevant) + 1)
            ap = (precision_at_k * relevant).sum() / relevant.sum()
            aps.append(ap)
        
        class_aps.append(np.mean(aps) if aps else 0.0)
    
    # Sort by AP
    sorted_indices = np.argsort(class_aps)
    
    plt.figure(figsize=(14, 10))
    colors = ['red' if ap < 0.5 else 'orange' if ap < 0.7 else 'green' for ap in np.array(class_aps)[sorted_indices]]
    
    plt.barh(range(n_classes), np.array(class_aps)[sorted_indices], color=colors)
    plt.yticks(range(n_classes), [class_names[i] for i in sorted_indices], fontsize=7)
    plt.xlabel('Average Precision (AP)')
    plt.title('Per-class mAP (sorted)', fontsize=14)
    plt.axvline(x=np.mean(class_aps), color='blue', linestyle='--', label=f'Mean: {np.mean(class_aps):.3f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    print(f"[✓] Saved per-class mAP: {save_path}")
    
    # Return worst classes
    worst_5 = [(class_names[sorted_indices[i]], class_aps[sorted_indices[i]]) for i in range(5)]
    return worst_5


def analyze_failures(args):
    """Main failure analysis function."""
    print("\n" + "="*60)
    print("FAILURE ANALYSIS")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    print(f"\n[1] Loading model from {args.checkpoint}")
    model, info = load_model(args.checkpoint, device)
    print(f"    Model type: {info['type']}")
    print(f"    Hash bits: {info['hash_bit']}")
    
    # Load data
    print(f"\n[2] Loading test data...")
    loader, class_names = get_test_loader(args.data_dir)
    print(f"    Classes: {len(class_names)}")
    print(f"    Test samples: {len(loader.dataset)}")
    
    # Extract features
    print(f"\n[3] Extracting features...")
    hash_codes, features, labels = extract_features(model, loader, device)
    print(f"    Hash codes shape: {hash_codes.shape}")
    
    # Compute retrieval results
    print(f"\n[4] Computing retrieval results...")
    predictions, distances, mAP = compute_retrieval_results(hash_codes, labels, top_k=50)
    print(f"    mAP: {mAP:.4f}")
    
    # Create results directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Plot confusion matrix
    print(f"\n[5] Generating confusion matrix...")
    plot_confusion_matrix(
        labels, predictions, class_names,
        os.path.join(args.output_dir, 'confusion_matrix.png')
    )
    
    # Find confused pairs
    print(f"\n[6] Top confused class pairs:")
    confused_pairs = find_confused_pairs(labels, predictions, class_names, top_n=10)
    for i, (c1, c2, count) in enumerate(confused_pairs, 1):
        print(f"    {i}. {c1} ↔ {c2}: {count} confusions")
    
    # Save to file
    with open(os.path.join(args.output_dir, 'confused_pairs.json'), 'w') as f:
        json.dump(confused_pairs, f, indent=2)
    
    # t-SNE
    print(f"\n[7] Generating t-SNE...")
    plot_tsne(
        hash_codes, labels, class_names,
        os.path.join(args.output_dir, 'tsne_hash_codes.png')
    )
    
    # Per-class mAP
    print(f"\n[8] Computing per-class mAP...")
    worst_classes = plot_per_class_map(
        labels, hash_codes, class_names,
        os.path.join(args.output_dir, 'per_class_map.png')
    )
    print(f"    Worst 5 classes:")
    for name, ap in worst_classes:
        print(f"      - {name}: {ap:.3f}")
    
    # Summary
    print(f"\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Results saved to: {args.output_dir}/")
    print(f"  - confusion_matrix.png")
    print(f"  - confused_pairs.json")
    print(f"  - tsne_hash_codes.png")
    print(f"  - per_class_map.png")


def compare_models(args):
    """Compare two models."""
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    results = {}
    
    for ckpt_path in args.compare:
        print(f"\n[*] Evaluating {ckpt_path}")
        model, info = load_model(ckpt_path, device)
        loader, class_names = get_test_loader(args.data_dir)
        
        hash_codes, features, labels = extract_features(model, loader, device)
        _, _, mAP = compute_retrieval_results(hash_codes, labels)
        
        model_name = os.path.basename(ckpt_path)
        results[model_name] = {
            'type': info['type'],
            'hash_bit': info['hash_bit'],
            'mAP': mAP
        }
        print(f"    mAP: {mAP:.4f}")
    
    # Summary table
    print(f"\n" + "-"*40)
    print("COMPARISON SUMMARY")
    print("-"*40)
    print(f"{'Model':<40} {'Type':<10} {'mAP':<10}")
    print("-"*60)
    for name, data in results.items():
        print(f"{name:<40} {data['type']:<10} {data['mAP']:.4f}")
    
    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'comparison.json'), 'w') as f:
        json.dump(results, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Visualization and Analysis')
    
    parser.add_argument('--checkpoint', type=str, help='Model checkpoint path')
    parser.add_argument('--compare', type=str, nargs='+', help='Compare multiple checkpoints')
    parser.add_argument('--analyze-failures', action='store_true', help='Run failure analysis')
    parser.add_argument('--data-dir', type=str, default='./data/archive/Dataset')
    parser.add_argument('--output-dir', type=str, default='./results/analysis')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_models(args)
    elif args.checkpoint:
        if args.analyze_failures:
            analyze_failures(args)
        else:
            analyze_failures(args)  # Default to full analysis
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
