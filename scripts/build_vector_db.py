"""
Vector Database Builder for Image Retrieval
=============================================

Dùng model đã fine-tune để:
1. Trích xuất hash codes từ tất cả ảnh trong database
2. Lưu thành vector database
3. Hỗ trợ query nhanh với Hamming distance

Usage:
    # 1. Build database từ model đã train
    python scripts/build_vector_db.py --checkpoint ./checkpoints/best_model.pth \
                                      --data-dir ./data/archive/Dataset/train/train \
                                      --output ./database/nwpu_vectors.npz
    
    # 2. Query một ảnh
    python scripts/build_vector_db.py --query ./test_image.jpg \
                                      --database ./database/nwpu_vectors.npz \
                                      --checkpoint ./checkpoints/best_model.pth
    
    # 3. Interactive demo
    python scripts/build_vector_db.py --demo --database ./database/nwpu_vectors.npz
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
from tqdm import tqdm

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.vit_hashing import ViT_Hashing
from src.models.dinov2_hashing import DINOv3Hashing


@dataclass
class VectorDatabase:
    """Vector Database structure."""
    hash_codes: np.ndarray      # [N, hash_bit] binary codes {-1, 1}
    image_paths: List[str]      # [N] paths to images
    labels: np.ndarray          # [N] class labels
    class_names: List[str]      # Class name mapping
    hash_bit: int
    model_type: str
    created_at: str
    
    def save(self, path: str):
        """Save database to file."""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        np.savez_compressed(
            path,
            hash_codes=self.hash_codes,
            image_paths=np.array(self.image_paths, dtype=object),
            labels=self.labels,
            class_names=np.array(self.class_names, dtype=object),
            hash_bit=self.hash_bit,
            model_type=self.model_type,
            created_at=self.created_at,
        )
        print(f"[✓] Database saved to {path}")
        print(f"    - Images: {len(self.image_paths)}")
        print(f"    - Hash bits: {self.hash_bit}")
        print(f"    - Size: {os.path.getsize(path) / 1024 / 1024:.2f} MB")
    
    @classmethod
    def load(cls, path: str) -> 'VectorDatabase':
        """Load database from file."""
        data = np.load(path, allow_pickle=True)
        return cls(
            hash_codes=data['hash_codes'],
            image_paths=data['image_paths'].tolist(),
            labels=data['labels'],
            class_names=data['class_names'].tolist(),
            hash_bit=int(data['hash_bit']),
            model_type=str(data['model_type']),
            created_at=str(data['created_at']),
        )


def load_model(checkpoint_path: str, device: torch.device) -> Tuple[nn.Module, dict]:
    """Load finetuned model from checkpoint.
    
    Automatically infers model type and hash_bit from state_dict.
    """
    print(f"[*] Loading model from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint['model_state_dict']
    
    # Infer model config from state_dict keys
    # ViT_Hashing:    hashing_head.1.weight, hashing_head.3.weight  (nn.Sequential)
    # DINOv3Hashing:  hashing_head.layers.1.weight  (HashingHead wraps Sequential)
    has_layers_prefix = any('hashing_head.layers.' in k for k in state)
    
    if has_layers_prefix:
        model_type = 'dinov3'
        hash_bit = state['hashing_head.layers.3.weight'].shape[0]
        embed_dim = state['hashing_head.layers.1.weight'].shape[1]
        hidden_dim = state['hashing_head.layers.1.weight'].shape[0]
    else:
        model_type = 'vit'
        hash_bit = state['hashing_head.3.weight'].shape[0]
        embed_dim = state['hashing_head.1.weight'].shape[1]
        hidden_dim = state['hashing_head.1.weight'].shape[0]
    
    # Infer backbone variant from embed_dim
    if model_type == 'vit':
        pos_len = state['backbone.pos_embed'].shape[1]  # 1 + num_patches
        model_name = 'vit_base_patch32_224' if pos_len == 50 else 'vit_base_patch16_224'
    else:
        dinov2_map = {384: 'vit_small_patch14_dinov2.lvd142m',
                      768: 'vit_base_patch14_dinov2.lvd142m',
                      1024: 'vit_large_patch14_dinov2.lvd142m'}
        model_name = dinov2_map.get(embed_dim, 'vit_small_patch14_dinov2.lvd142m')
    
    print(f"    Type: {model_type} ({model_name})")
    print(f"    Hash bits: {hash_bit}, Embed dim: {embed_dim}")
    
    # Build model (pretrained=False — we load our own weights)
    if model_type == 'vit':
        model = ViT_Hashing(model_name=model_name, pretrained=False, hash_bit=hash_bit)
    else:
        model = DINOv3Hashing(model_name=model_name, pretrained=False,
                              hash_bit=hash_bit, hidden_dim=hidden_dim)
    
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    
    metrics = checkpoint.get('metrics', {})
    print(f"    Epoch: {checkpoint.get('epoch', '?')}, Val mAP: {metrics.get('mAP', 'N/A')}")
    print(f"[✓] Model loaded")
    
    return model, {
        'hash_bit': hash_bit,
        'model_type': model_type,
        'model_name': model_name,
    }


def get_transform():
    """Get image transform for inference."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])


@torch.no_grad()
def extract_hash_codes(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract hash codes from all images in dataloader.
    
    Returns:
        hash_codes: [N, hash_bit] binary codes
        labels: [N] class labels
    """
    model.eval()
    
    all_codes = []
    all_labels = []
    
    print("[*] Extracting hash codes...")
    
    for images, labels in tqdm(data_loader, desc="Extracting"):
        images = images.to(device)
        
        # Forward pass - get hash codes
        hash_output, _ = model(images)
        
        # Binarize: sign() converts to {-1, 1}
        binary_codes = torch.sign(hash_output).cpu().numpy()
        
        all_codes.append(binary_codes)
        all_labels.append(labels.numpy())
    
    hash_codes = np.concatenate(all_codes, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    print(f"[✓] Extracted {len(hash_codes)} hash codes")
    
    return hash_codes, labels


def build_database(
    checkpoint_path: str,
    data_dirs: List[str],
    output_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    device: str = 'cuda',
) -> VectorDatabase:
    """
    Build vector database from finetuned model.
    
    Supports multiple data directories (e.g. train + test) to build
    a full dataset database for production retrieval.
    
    Args:
        checkpoint_path: Path to model checkpoint
        data_dirs: List of image directories (ImageFolder format)
        output_path: Path to save database
        batch_size: Batch size for extraction
        num_workers: DataLoader workers
        device: 'cuda' or 'cpu'
    """
    from torch.utils.data import ConcatDataset
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"\n[Device] {device}")
    
    # Load model
    model, model_info = load_model(checkpoint_path, device)
    
    # Create dataset(s)
    transform = get_transform()
    datasets = []
    all_image_paths = []
    class_names = None
    
    for data_dir in data_dirs:
        print(f"\n[*] Loading images from {data_dir}")
        ds = ImageFolder(data_dir, transform=transform)
        print(f"    Found {len(ds)} images in {len(ds.classes)} classes")
        datasets.append(ds)
        all_image_paths.extend([s[0] for s in ds.samples])
        if class_names is None:
            class_names = ds.classes
    
    if len(datasets) == 1:
        combined = datasets[0]
    else:
        combined = ConcatDataset(datasets)
    
    total = len(combined)
    print(f"\n[*] Total: {total} images from {len(data_dirs)} source(s)")
    
    # Create dataloader
    data_loader = DataLoader(
        combined,
        batch_size=batch_size,
        shuffle=False,  # Important: keep order for path mapping
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Extract hash codes
    hash_codes, labels = extract_hash_codes(model, data_loader, device)
    
    # Create database
    from datetime import datetime
    
    db = VectorDatabase(
        hash_codes=hash_codes,
        image_paths=all_image_paths,
        labels=labels,
        class_names=class_names,
        hash_bit=model_info['hash_bit'],
        model_type=model_info['model_type'],
        created_at=datetime.now().isoformat(),
    )
    
    # Save
    db.save(output_path)
    
    return db


def hamming_distance(query: np.ndarray, database: np.ndarray) -> np.ndarray:
    """
    Compute Hamming distance between query and all database entries.
    
    Args:
        query: [hash_bit] single query hash code
        database: [N, hash_bit] database hash codes
        
    Returns:
        distances: [N] Hamming distances
    """
    # For {-1, 1} encoding: hamming = 0.5 * (K - dot_product)
    hash_bit = query.shape[0]
    dot_products = np.dot(database, query)
    distances = 0.5 * (hash_bit - dot_products)
    return distances.astype(np.int32)


@torch.no_grad()
def query_image(
    image_path: str,
    model: nn.Module,
    database: VectorDatabase,
    device: torch.device,
    top_k: int = 10,
) -> List[Dict]:
    """
    Query database with a single image.
    
    Returns:
        List of top-k results with paths, distances, and labels
    """
    # Load and transform image
    transform = get_transform()
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Extract hash code
    model.eval()
    hash_output, _ = model(image_tensor)
    query_hash = torch.sign(hash_output).cpu().numpy()[0]
    
    # Compute distances
    distances = hamming_distance(query_hash, database.hash_codes)
    
    # Get top-k
    top_indices = np.argsort(distances)[:top_k]
    
    results = []
    for idx in top_indices:
        results.append({
            'rank': len(results) + 1,
            'path': database.image_paths[idx],
            'distance': int(distances[idx]),
            'label': int(database.labels[idx]),
            'class_name': database.class_names[database.labels[idx]],
        })
    
    return results


def print_query_results(query_path: str, results: List[Dict], query_label: Optional[str] = None):
    """Pretty print query results."""
    print(f"\n{'='*60}")
    print(f"Query: {query_path}")
    if query_label:
        print(f"Query class: {query_label}")
    print(f"{'='*60}")
    print(f"{'Rank':<6} {'Distance':<10} {'Class':<20} {'Path':<30}")
    print(f"{'-'*60}")
    
    for r in results:
        relevance = "✓" if query_label and r['class_name'] == query_label else ""
        print(f"{r['rank']:<6} {r['distance']:<10} {r['class_name']:<20} {Path(r['path']).name:<30} {relevance}")
    
    print(f"{'='*60}")


def compute_retrieval_metrics(
    query_results: List[Dict],
    query_label: int,
    database: VectorDatabase,
) -> Dict:
    """Compute retrieval metrics for a single query."""
    relevant = [1 if r['label'] == query_label else 0 for r in query_results]
    
    # Precision at K
    p_at_5 = sum(relevant[:5]) / 5
    p_at_10 = sum(relevant[:10]) / 10 if len(relevant) >= 10 else sum(relevant) / len(relevant)
    
    # Average Precision
    ap = 0.0
    num_relevant = 0
    for i, rel in enumerate(relevant):
        if rel == 1:
            num_relevant += 1
            ap += num_relevant / (i + 1)
    
    ap = ap / num_relevant if num_relevant > 0 else 0.0
    
    return {
        'P@5': p_at_5,
        'P@10': p_at_10,
        'AP': ap,
        'num_relevant': num_relevant,
    }


def demo_retrieval(
    database_path: str,
    checkpoint_path: str,
    num_queries: int = 5,
):
    """Run interactive demo with random queries."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load database
    print(f"\n[*] Loading database from {database_path}")
    db = VectorDatabase.load(database_path)
    print(f"    Database: {len(db.image_paths)} images, {db.hash_bit} bits")
    
    # Load model
    model, _ = load_model(checkpoint_path, device)
    
    # Random queries
    np.random.seed(42)
    query_indices = np.random.choice(len(db.image_paths), num_queries, replace=False)
    
    all_metrics = []
    
    for idx in query_indices:
        query_path = db.image_paths[idx]
        query_label = db.labels[idx]
        query_class = db.class_names[query_label]
        
        # Query
        results = query_image(query_path, model, db, device, top_k=20)
        
        # Show results
        print_query_results(query_path, results[:10], query_class)
        
        # Compute metrics
        metrics = compute_retrieval_metrics(results, query_label, db)
        all_metrics.append(metrics)
        print(f"P@5: {metrics['P@5']:.2f} | P@10: {metrics['P@10']:.2f} | AP: {metrics['AP']:.4f}")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Mean P@5:  {np.mean([m['P@5'] for m in all_metrics]):.4f}")
    print(f"Mean P@10: {np.mean([m['P@10'] for m in all_metrics]):.4f}")
    print(f"Mean AP:   {np.mean([m['AP'] for m in all_metrics]):.4f}")


def evaluate_full_database(
    database_path: str,
    checkpoint_path: str,
    num_queries: int = 500,
):
    """Evaluate mAP on full database."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load database
    db = VectorDatabase.load(database_path)
    print(f"Database: {len(db.image_paths)} images")
    
    # Load model
    model, _ = load_model(checkpoint_path, device)
    
    # Sample queries (or use all)
    if num_queries < len(db.image_paths):
        np.random.seed(42)
        query_indices = np.random.choice(len(db.image_paths), num_queries, replace=False)
    else:
        query_indices = range(len(db.image_paths))
    
    all_ap = []
    
    print(f"\nEvaluating {len(query_indices)} queries...")
    
    for idx in tqdm(query_indices):
        query_path = db.image_paths[idx]
        query_label = db.labels[idx]
        
        results = query_image(query_path, model, db, device, top_k=100)
        metrics = compute_retrieval_metrics(results, query_label, db)
        all_ap.append(metrics['AP'])
    
    mAP = np.mean(all_ap)
    print(f"\n[Result] mAP@100: {mAP:.4f}")
    
    return mAP


def main():
    parser = argparse.ArgumentParser(description='Build and Query Vector Database')
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Build command
    build_parser = subparsers.add_parser('build', help='Build vector database')
    build_parser.add_argument('--checkpoint', type=str, required=True,
                              help='Path to model checkpoint')
    build_parser.add_argument('--data-dir', type=str, nargs='+',
                              help='Path(s) to image directory (can specify multiple)')
    build_parser.add_argument('--full', action='store_true',
                              help='Build from full NWPU dataset (train + test)')
    build_parser.add_argument('--dataset-root', type=str, default='./data/archive/Dataset',
                              help='Root of NWPU dataset (used with --full)')
    build_parser.add_argument('--output', type=str, default='./database/vectors.npz',
                              help='Output database path')
    build_parser.add_argument('--batch-size', type=int, default=32)
    build_parser.add_argument('--device', type=str, default='cuda')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query single image')
    query_parser.add_argument('--image', type=str, required=True,
                              help='Query image path')
    query_parser.add_argument('--database', type=str, required=True,
                              help='Database path')
    query_parser.add_argument('--checkpoint', type=str, required=True,
                              help='Model checkpoint')
    query_parser.add_argument('--top-k', type=int, default=10)
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Interactive demo')
    demo_parser.add_argument('--database', type=str, required=True)
    demo_parser.add_argument('--checkpoint', type=str, required=True)
    demo_parser.add_argument('--num-queries', type=int, default=5)
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate mAP')
    eval_parser.add_argument('--database', type=str, required=True)
    eval_parser.add_argument('--checkpoint', type=str, required=True)
    eval_parser.add_argument('--num-queries', type=int, default=500)
    
    args = parser.parse_args()
    
    if args.command == 'build':
        if args.full:
            data_dirs = [
                os.path.join(args.dataset_root, 'train', 'train'),
                os.path.join(args.dataset_root, 'test', 'test'),
            ]
            print(f"[*] Full dataset mode: train + test")
        elif args.data_dir:
            data_dirs = args.data_dir
        else:
            print("[Error] Specify --data-dir or --full")
            return
        
        build_database(
            checkpoint_path=args.checkpoint,
            data_dirs=data_dirs,
            output_path=args.output,
            batch_size=args.batch_size,
            device=args.device,
        )
        
    elif args.command == 'query':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        db = VectorDatabase.load(args.database)
        model, _ = load_model(args.checkpoint, device)
        
        results = query_image(args.image, model, db, device, top_k=args.top_k)
        print_query_results(args.image, results)
        
    elif args.command == 'demo':
        demo_retrieval(
            database_path=args.database,
            checkpoint_path=args.checkpoint,
            num_queries=args.num_queries,
        )
        
    elif args.command == 'evaluate':
        evaluate_full_database(
            database_path=args.database,
            checkpoint_path=args.checkpoint,
            num_queries=args.num_queries,
        )
        
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
