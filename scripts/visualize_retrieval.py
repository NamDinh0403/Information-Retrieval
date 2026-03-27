"""
Visualization for Text-to-Image Retrieval Results
==================================================

Hiển thị kết quả truy vấn dưới dạng grid ảnh.

Usage:
    python scripts/visualize_retrieval.py \
        --database ./database/clip_vectors.npz \
        --text "an airport with planes" \
        --output ./results/retrieval_demo.png
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from typing import List, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.text_image_retrieval import TextImageRetrieval, CrossModalDatabase


def visualize_results(
    results: List[Dict],
    query_text: Optional[str] = None,
    query_image: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (15, 10),
):
    """
    Visualize retrieval results as image grid.
    """
    n_results = len(results)
    n_cols = min(5, n_results)
    n_rows = (n_results + n_cols - 1) // n_cols
    
    if query_image or query_text:
        n_rows += 1  # Extra row for query
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.array(axes).reshape(-1)
    
    # Hide all axes first
    for ax in axes:
        ax.axis('off')
    
    start_idx = 0
    
    # Show query if provided
    if query_image:
        ax = axes[0]
        img = Image.open(query_image)
        ax.imshow(img)
        ax.set_title("QUERY IMAGE", fontsize=12, fontweight='bold', color='blue')
        ax.axis('on')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color('blue')
            spine.set_linewidth(3)
        start_idx = n_cols
    elif query_text:
        ax = axes[n_cols // 2]
        ax.text(0.5, 0.5, f'Query:\n"{query_text}"', 
                ha='center', va='center', fontsize=14, 
                wrap=True, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        start_idx = n_cols
    
    # Show results
    for i, result in enumerate(results):
        ax = axes[start_idx + i]
        
        try:
            img = Image.open(result['path'])
            ax.imshow(img)
        except Exception as e:
            ax.text(0.5, 0.5, 'Image\nNot Found', ha='center', va='center')
        
        # Title with rank, class, and score
        title = f"#{result['rank']} {result['class']}\nScore: {result['score']:.3f}"
        ax.set_title(title, fontsize=9)
        
        ax.axis('on')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Color border based on score
        if result['score'] > 0.3:
            color = 'green'
        elif result['score'] > 0.2:
            color = 'orange'
        else:
            color = 'red'
        
        for spine in ax.spines.values():
            spine.set_color(color)
            spine.set_linewidth(2)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[✓] Saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_comparison_figure(
    queries: List[str],
    database_path: str,
    save_path: str,
    top_k: int = 5,
    device: str = 'cuda',
):
    """
    Create comparison figure with multiple queries.
    """
    # Load
    db = CrossModalDatabase.load(database_path)
    retrieval = TextImageRetrieval(device=device)
    
    n_queries = len(queries)
    n_cols = top_k + 1  # Query + results
    
    fig, axes = plt.subplots(n_queries, n_cols, figsize=(3 * n_cols, 3 * n_queries))
    
    for row, query in enumerate(queries):
        # Query text
        ax = axes[row, 0]
        ax.text(0.5, 0.5, f'"{query}"', 
                ha='center', va='center', fontsize=10, wrap=True,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        if row == 0:
            ax.set_title("Query", fontsize=12, fontweight='bold')
        
        # Results
        results = retrieval.text_search(query, db, top_k=top_k)
        
        for col, result in enumerate(results):
            ax = axes[row, col + 1]
            
            try:
                img = Image.open(result['path'])
                ax.imshow(img)
            except:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center')
            
            ax.set_title(f"{result['class']}\n{result['score']:.2f}", fontsize=8)
            ax.axis('off')
            
            if row == 0:
                ax.set_title(f"Result #{col+1}", fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[✓] Comparison figure saved to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize Retrieval Results')
    
    parser.add_argument('--database', type=str, required=True)
    parser.add_argument('--text', type=str, help='Text query')
    parser.add_argument('--image', type=str, help='Image query')
    parser.add_argument('--output', type=str, default='./results/retrieval_result.png')
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--comparison', action='store_true', 
                        help='Create comparison figure with multiple queries')
    
    args = parser.parse_args()
    
    if args.comparison:
        # Demo queries for NWPU classes
        queries = [
            "an airport with planes and runways",
            "a beach with clear water",
            "dense residential area with many buildings",
            "green forest with trees",
            "a bridge over a river",
        ]
        create_comparison_figure(
            queries=queries,
            database_path=args.database,
            save_path=args.output,
            device=args.device,
        )
    else:
        # Single query
        db = CrossModalDatabase.load(args.database)
        retrieval = TextImageRetrieval(device=args.device)
        
        if args.text:
            results = retrieval.text_search(args.text, db, top_k=args.top_k)
            visualize_results(results, query_text=args.text, save_path=args.output)
        elif args.image:
            results = retrieval.image_search(args.image, db, top_k=args.top_k)
            visualize_results(results, query_image=args.image, save_path=args.output)
        else:
            print("Please provide --text or --image query")


if __name__ == '__main__':
    main()
