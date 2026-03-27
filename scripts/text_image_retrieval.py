"""
Text-to-Image Retrieval System
==============================

Hệ thống truy vấn ảnh bằng câu mô tả ngôn ngữ tự nhiên.

Query Types:
    1. Text query: "an airport with many airplanes"
    2. Image query: [image file]
    3. Combined: text + image

Usage:
    # Build database (extract image hashes)
    python scripts/text_image_retrieval.py build \
        --data-dir ./data/archive/Dataset/train/train \
        --output ./database/clip_vectors.npz
    
    # Query with text
    python scripts/text_image_retrieval.py query \
        --text "an airport with many planes" \
        --database ./database/clip_vectors.npz
    
    # Query with image
    python scripts/text_image_retrieval.py query \
        --image ./test.jpg \
        --database ./database/clip_vectors.npz
    
    # Interactive demo
    python scripts/text_image_retrieval.py demo \
        --database ./database/clip_vectors.npz
"""

import os
import sys
import argparse
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from datetime import datetime
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("[!] Install CLIP: pip install git+https://github.com/openai/CLIP.git")


@dataclass
class CrossModalDatabase:
    """Database for cross-modal retrieval."""
    image_hashes: np.ndarray       # [N, hash_bit] or [N, embed_dim]
    image_paths: List[str]
    labels: np.ndarray
    class_names: List[str]
    hash_bit: int
    use_binary: bool               # True: binary hashes, False: continuous embeddings
    created_at: str
    
    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        np.savez_compressed(
            path,
            image_hashes=self.image_hashes,
            image_paths=np.array(self.image_paths, dtype=object),
            labels=self.labels,
            class_names=np.array(self.class_names, dtype=object),
            hash_bit=self.hash_bit,
            use_binary=self.use_binary,
            created_at=self.created_at,
        )
        print(f"[✓] Database saved: {path}")
    
    @classmethod
    def load(cls, path: str) -> 'CrossModalDatabase':
        data = np.load(path, allow_pickle=True)
        return cls(
            image_hashes=data['image_hashes'],
            image_paths=data['image_paths'].tolist(),
            labels=data['labels'],
            class_names=data['class_names'].tolist(),
            hash_bit=int(data['hash_bit']),
            use_binary=bool(data['use_binary']),
            created_at=str(data['created_at']),
        )


class TextImageRetrieval:
    """
    Text-to-Image and Image-to-Image Retrieval System.
    
    Uses CLIP for cross-modal embedding.
    """
    
    def __init__(
        self,
        model_name: str = "ViT-B/32",
        device: str = "cuda",
        use_binary_hash: bool = False,
        hash_bit: int = 64,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.use_binary_hash = use_binary_hash
        self.hash_bit = hash_bit
        
        print(f"[*] Loading CLIP model: {model_name}")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
        
        self.embed_dim = self.model.visual.output_dim
        print(f"    Embedding dim: {self.embed_dim}")
        print(f"    Device: {self.device}")
    
    @torch.no_grad()
    def encode_images(
        self,
        image_paths: List[str],
        batch_size: int = 32,
    ) -> np.ndarray:
        """Encode images to embeddings/hashes."""
        all_features = []
        
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Encoding images"):
            batch_paths = image_paths[i:i + batch_size]
            
            # Load and preprocess images
            images = []
            for path in batch_paths:
                try:
                    img = Image.open(path).convert('RGB')
                    img = self.preprocess(img)
                    images.append(img)
                except Exception as e:
                    print(f"[Warning] Cannot load {path}: {e}")
                    # Use blank image as placeholder
                    images.append(torch.zeros(3, 224, 224))
            
            images = torch.stack(images).to(self.device)
            
            # Encode
            features = self.model.encode_image(images)
            features = features / features.norm(dim=-1, keepdim=True)  # Normalize
            
            all_features.append(features.cpu().numpy())
        
        return np.concatenate(all_features, axis=0)
    
    @torch.no_grad()
    def encode_text(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Encode text to embeddings."""
        if isinstance(texts, str):
            texts = [texts]
        
        text_tokens = clip.tokenize(texts, truncate=True).to(self.device)
        features = self.model.encode_text(text_tokens)
        features = features / features.norm(dim=-1, keepdim=True)
        
        return features.cpu().numpy()
    
    @torch.no_grad()
    def encode_image(self, image_path: str) -> np.ndarray:
        """Encode single image."""
        image = Image.open(image_path).convert('RGB')
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        
        features = self.model.encode_image(image)
        features = features / features.norm(dim=-1, keepdim=True)
        
        return features.cpu().numpy()[0]
    
    def search(
        self,
        query_embedding: np.ndarray,
        database: CrossModalDatabase,
        top_k: int = 10,
    ) -> List[Dict]:
        """
        Search database with query embedding.
        
        Uses cosine similarity for continuous embeddings.
        """
        # Compute similarities
        if database.use_binary:
            # Hamming distance for binary
            distances = 0.5 * (database.hash_bit - np.dot(database.image_hashes, query_embedding))
            top_indices = np.argsort(distances)[:top_k]
            scores = -distances[top_indices]  # Negative distance as score
        else:
            # Cosine similarity for continuous
            similarities = np.dot(database.image_hashes, query_embedding)
            top_indices = np.argsort(similarities)[::-1][:top_k]
            scores = similarities[top_indices]
        
        results = []
        for i, idx in enumerate(top_indices):
            results.append({
                'rank': i + 1,
                'path': database.image_paths[idx],
                'score': float(scores[i]),
                'label': int(database.labels[idx]),
                'class': database.class_names[database.labels[idx]],
            })
        
        return results
    
    def text_search(
        self,
        text: str,
        database: CrossModalDatabase,
        top_k: int = 10,
    ) -> List[Dict]:
        """Search with text query."""
        text_embedding = self.encode_text(text)[0]
        return self.search(text_embedding, database, top_k)
    
    def image_search(
        self,
        image_path: str,
        database: CrossModalDatabase,
        top_k: int = 10,
    ) -> List[Dict]:
        """Search with image query."""
        image_embedding = self.encode_image(image_path)
        return self.search(image_embedding, database, top_k)


def build_database(
    data_dir: str,
    output_path: str,
    model_name: str = "ViT-B/32",
    batch_size: int = 32,
    device: str = "cuda",
):
    """Build image database for retrieval."""
    from torchvision.datasets import ImageFolder
    
    print("\n" + "=" * 60)
    print("BUILDING CROSS-MODAL DATABASE")
    print("=" * 60)
    
    # Initialize retrieval system
    retrieval = TextImageRetrieval(model_name=model_name, device=device)
    
    # Load dataset info
    print(f"\n[*] Loading dataset from {data_dir}")
    dataset = ImageFolder(data_dir)
    image_paths = [sample[0] for sample in dataset.samples]
    labels = np.array([sample[1] for sample in dataset.samples])
    class_names = dataset.classes
    
    print(f"    Images: {len(image_paths)}")
    print(f"    Classes: {len(class_names)}")
    
    # Encode all images
    print("\n[*] Encoding images...")
    image_hashes = retrieval.encode_images(image_paths, batch_size=batch_size)
    print(f"    Embeddings shape: {image_hashes.shape}")
    
    # Create database
    db = CrossModalDatabase(
        image_hashes=image_hashes,
        image_paths=image_paths,
        labels=labels,
        class_names=class_names,
        hash_bit=image_hashes.shape[1],
        use_binary=False,
        created_at=datetime.now().isoformat(),
    )
    
    # Save
    db.save(output_path)
    
    print(f"\n[✓] Database created: {output_path}")
    print(f"    Size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")


def query_database(
    database_path: str,
    text: Optional[str] = None,
    image: Optional[str] = None,
    top_k: int = 10,
    device: str = "cuda",
):
    """Query database with text or image."""
    # Load database
    print(f"\n[*] Loading database: {database_path}")
    db = CrossModalDatabase.load(database_path)
    print(f"    Images: {len(db.image_paths)}")
    
    # Initialize retrieval
    retrieval = TextImageRetrieval(device=device)
    
    # Query
    if text:
        print(f"\n[*] Text query: \"{text}\"")
        results = retrieval.text_search(text, db, top_k=top_k)
    elif image:
        print(f"\n[*] Image query: {image}")
        results = retrieval.image_search(image, db, top_k=top_k)
    else:
        print("[Error] Please provide --text or --image")
        return
    
    # Print results
    print("\n" + "=" * 70)
    print("RETRIEVAL RESULTS")
    print("=" * 70)
    print(f"{'Rank':<6}{'Score':<12}{'Class':<25}{'File':<30}")
    print("-" * 70)
    
    for r in results:
        print(f"{r['rank']:<6}{r['score']:.4f}      {r['class']:<25}{Path(r['path']).name:<30}")
    
    print("=" * 70)


def interactive_demo(database_path: str, device: str = "cuda"):
    """Interactive demo for text-to-image retrieval."""
    # Load database
    print(f"\n[*] Loading database: {database_path}")
    db = CrossModalDatabase.load(database_path)
    print(f"    Images: {len(db.image_paths)}")
    print(f"    Classes: {db.class_names[:10]}..." if len(db.class_names) > 10 else f"    Classes: {db.class_names}")
    
    # Initialize retrieval
    retrieval = TextImageRetrieval(device=device)
    
    print("\n" + "=" * 60)
    print("TEXT-TO-IMAGE RETRIEVAL DEMO")
    print("=" * 60)
    print("Enter a text description to search for images.")
    print("Type 'quit' to exit.\n")
    print("Example queries:")
    print("  - 'an airport with airplanes'")
    print("  - 'a beach with clear water'")
    print("  - 'dense residential area with many houses'")
    print("  - 'a bridge over water'")
    print("-" * 60)
    
    while True:
        try:
            query = input("\n🔍 Query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not query:
                continue
            
            # Search
            results = retrieval.text_search(query, db, top_k=10)
            
            # Print results
            print(f"\n{'Rank':<6}{'Score':<12}{'Class':<25}{'File':<35}")
            print("-" * 75)
            
            for r in results:
                print(f"{r['rank']:<6}{r['score']:.4f}      {r['class']:<25}{Path(r['path']).name:<35}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


def main():
    parser = argparse.ArgumentParser(description='Text-to-Image Retrieval')
    subparsers = parser.add_subparsers(dest='command')
    
    # Build command
    build_parser = subparsers.add_parser('build', help='Build database')
    build_parser.add_argument('--data-dir', type=str, required=True)
    build_parser.add_argument('--output', type=str, default='./database/clip_vectors.npz')
    build_parser.add_argument('--batch-size', type=int, default=32)
    build_parser.add_argument('--device', type=str, default='cuda')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query database')
    query_parser.add_argument('--database', type=str, required=True)
    query_parser.add_argument('--text', type=str, help='Text query')
    query_parser.add_argument('--image', type=str, help='Image query')
    query_parser.add_argument('--top-k', type=int, default=10)
    query_parser.add_argument('--device', type=str, default='cuda')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Interactive demo')
    demo_parser.add_argument('--database', type=str, required=True)
    demo_parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    if args.command == 'build':
        build_database(
            data_dir=args.data_dir,
            output_path=args.output,
            batch_size=args.batch_size,
            device=args.device,
        )
    elif args.command == 'query':
        query_database(
            database_path=args.database,
            text=args.text,
            image=args.image,
            top_k=args.top_k,
            device=args.device,
        )
    elif args.command == 'demo':
        interactive_demo(
            database_path=args.database,
            device=args.device,
        )
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
