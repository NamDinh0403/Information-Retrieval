"""
Quick Image Retrieval Demo
==========================

Query database với một ảnh và hiển thị kết quả.

Usage:
    # Query với ảnh bất kỳ
    python scripts/query_image.py --image path/to/image.jpg
    
    # Query với ảnh từ dataset
    python scripts/query_image.py --random
    
    # Hiển thị top-20
    python scripts/query_image.py --image test.jpg --top-k 20
"""

import os
import sys
import argparse
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.vit_hashing import ViT_Hashing
from src.models.dinov2_hashing import DINOv3Hashing


# ============================================================================
# DEFAULT PATHS - Chỉnh sửa theo setup của bạn
# ============================================================================

DEFAULT_DATABASE = './database/nwpu_database.npz'
DEFAULT_CHECKPOINT = './checkpoints/best_model.pth'


def hamming_distance(query: np.ndarray, database: np.ndarray) -> np.ndarray:
    """Compute Hamming distances."""
    hash_bit = query.shape[0]
    return (0.5 * (hash_bit - np.dot(database, query))).astype(np.int32)


def load_database(path: str) -> dict:
    """Load database."""
    print(f"Loading database from {path}...")
    data = np.load(path, allow_pickle=True)
    return {
        'hash_codes': data['hash_codes'],
        'labels': data['labels'],
        'image_paths': data['image_paths'].tolist(),
        'class_names': data['class_names'].tolist(),
        'hash_bit': int(data['hash_bit']),
        'model_type': str(data['model_type']),
    }


def load_model(checkpoint_path: str, device: torch.device):
    """Load model."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    hash_bit = checkpoint.get('hash_bit', 64)
    model_type = checkpoint.get('model_type', 'vit')
    num_classes = checkpoint.get('num_classes', 45)
    
    if model_type == 'vit':
        model = ViT_Hashing(hash_bit=hash_bit, num_classes=num_classes)
    else:
        model = DINOv3Hashing(hash_bit=hash_bit, num_classes=num_classes)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model


@torch.no_grad()
def extract_query_hash(image_path: str, model, device) -> np.ndarray:
    """Extract hash code from query image."""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    hash_output, _ = model(image_tensor)
    return torch.sign(hash_output).cpu().numpy()[0]


def search(query_hash: np.ndarray, database: dict, top_k: int = 10):
    """Search database."""
    distances = hamming_distance(query_hash, database['hash_codes'])
    top_indices = np.argsort(distances)[:top_k]
    
    results = []
    for i, idx in enumerate(top_indices):
        results.append({
            'rank': i + 1,
            'distance': int(distances[idx]),
            'path': database['image_paths'][idx],
            'label': int(database['labels'][idx]),
            'class': database['class_names'][database['labels'][idx]],
        })
    
    return results


def print_results(query_path: str, results: list, query_class: str = None):
    """Pretty print results."""
    print("\n" + "=" * 70)
    print(f"QUERY: {Path(query_path).name}")
    if query_class:
        print(f"CLASS: {query_class}")
    print("=" * 70)
    
    print(f"\n{'Rank':<6}{'Dist':<8}{'Class':<25}{'File':<30}{'Match':<6}")
    print("-" * 70)
    
    correct = 0
    for r in results:
        is_match = "✓" if query_class and r['class'] == query_class else ""
        if is_match:
            correct += 1
        print(f"{r['rank']:<6}{r['distance']:<8}{r['class']:<25}{Path(r['path']).name:<30}{is_match:<6}")
    
    print("-" * 70)
    
    if query_class:
        precision = correct / len(results)
        print(f"Precision@{len(results)}: {precision:.2%} ({correct}/{len(results)} correct)")
    
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Query Image Retrieval')
    parser.add_argument('--image', type=str, help='Path to query image')
    parser.add_argument('--random', action='store_true', help='Use random image from database')
    parser.add_argument('--database', type=str, default=DEFAULT_DATABASE)
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    # Check files exist
    if not os.path.exists(args.database):
        print(f"[Error] Database not found: {args.database}")
        print("Run 'python scripts/extract_features.py' first to create the database.")
        return
    
    if not os.path.exists(args.checkpoint):
        print(f"[Error] Checkpoint not found: {args.checkpoint}")
        return
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load
    database = load_database(args.database)
    model = load_model(args.checkpoint, device)
    
    print(f"Database: {len(database['image_paths'])} images, {database['hash_bit']} bits")
    
    # Get query image
    if args.random:
        np.random.seed(None)  # Random seed
        idx = np.random.randint(0, len(database['image_paths']))
        query_path = database['image_paths'][idx]
        query_class = database['class_names'][database['labels'][idx]]
        print(f"\nRandom query: {query_path}")
    elif args.image:
        query_path = args.image
        query_class = None
        
        # Try to infer class from path
        for cls in database['class_names']:
            if cls in query_path:
                query_class = cls
                break
    else:
        print("Please specify --image or --random")
        return
    
    # Extract query hash
    print("Extracting query hash...")
    query_hash = extract_query_hash(query_path, model, device)
    
    # Search
    print(f"Searching top-{args.top_k}...")
    results = search(query_hash, database, top_k=args.top_k)
    
    # Print results
    print_results(query_path, results, query_class)


if __name__ == '__main__':
    main()
