"""
Multi-Label Metrics for NUS-WIDE Evaluation
============================================

Metrics for evaluating multi-label image retrieval:
- mAP: Mean Average Precision (for multi-label)
- NDCG: Normalized Discounted Cumulative Gain
- P@K: Precision at K

For multi-label, a retrieved image is "relevant" if it shares
at least one label with the query image.
"""

import torch
import numpy as np
from typing import Tuple, Optional, List
from tqdm import tqdm


def hamming_distance(qB: np.ndarray, rB: np.ndarray) -> np.ndarray:
    """
    Compute Hamming distance matrix.
    
    Args:
        qB: Query binary codes [num_query, hash_bit], values in {-1, 1}
        rB: Database binary codes [num_db, hash_bit], values in {-1, 1}
        
    Returns:
        dist: Distance matrix [num_query, num_db]
    """
    # Hamming distance = 0.5 * (K - qB @ rB.T) when values are in {-1, 1}
    hash_bit = qB.shape[1]
    return 0.5 * (hash_bit - np.dot(qB, rB.T))


def compute_multilabel_similarity(
    query_labels: np.ndarray,
    db_labels: np.ndarray,
) -> np.ndarray:
    """
    Compute ground truth similarity for multi-label.
    Two images are similar if they share at least one label.
    
    Args:
        query_labels: [num_query, num_classes] multi-hot
        db_labels: [num_db, num_classes] multi-hot
        
    Returns:
        similarity: [num_query, num_db] binary similarity matrix
    """
    # Dot product > 0 means at least one shared label
    sim = np.dot(query_labels, db_labels.T)
    return (sim > 0).astype(np.float32)


def calculate_multilabel_map(
    qB: np.ndarray,
    rB: np.ndarray,
    query_labels: np.ndarray,
    db_labels: np.ndarray,
    top_k: Optional[int] = None,
) -> float:
    """
    Calculate Mean Average Precision for multi-label retrieval.
    
    Args:
        qB: Query binary hash codes [num_query, hash_bit]
        rB: Database binary hash codes [num_db, hash_bit]
        query_labels: Query labels [num_query, num_classes]
        db_labels: Database labels [num_db, num_classes]
        top_k: If provided, only consider top-k retrieved results
        
    Returns:
        mAP: Mean Average Precision score
    """
    num_query = qB.shape[0]
    
    # Compute Hamming distance
    dist = hamming_distance(qB, rB)
    
    # Compute ground truth similarity
    gnd_sim = compute_multilabel_similarity(query_labels, db_labels)
    
    map_sum = 0.0
    valid_queries = 0
    
    for i in range(num_query):
        # Get ground truth for this query
        gnd = gnd_sim[i]
        
        # Skip if no relevant items
        if np.sum(gnd) == 0:
            continue
        
        valid_queries += 1
        
        # Sort by distance
        sorted_indices = np.argsort(dist[i])
        
        if top_k:
            sorted_indices = sorted_indices[:top_k]
        
        gnd_sorted = gnd[sorted_indices]
        
        # Compute AP
        num_relevant = np.sum(gnd_sorted)
        if num_relevant == 0:
            continue
            
        # Positions of relevant items (1-indexed)
        relevant_positions = np.where(gnd_sorted == 1)[0] + 1
        
        # Precision at each relevant position
        precisions = np.arange(1, len(relevant_positions) + 1) / relevant_positions
        
        ap = np.mean(precisions)
        map_sum += ap
    
    if valid_queries == 0:
        return 0.0
    
    return map_sum / valid_queries


def calculate_precision_at_k(
    qB: np.ndarray,
    rB: np.ndarray,
    query_labels: np.ndarray,
    db_labels: np.ndarray,
    k_list: List[int] = [1, 5, 10, 50, 100],
) -> dict:
    """
    Calculate Precision@K for multi-label retrieval.
    
    Args:
        qB: Query binary hash codes
        rB: Database binary hash codes
        query_labels: Query labels
        db_labels: Database labels
        k_list: List of K values
        
    Returns:
        dict: {k: precision} for each k
    """
    num_query = qB.shape[0]
    
    # Compute distances
    dist = hamming_distance(qB, rB)
    
    # Compute ground truth
    gnd_sim = compute_multilabel_similarity(query_labels, db_labels)
    
    results = {k: 0.0 for k in k_list}
    valid_queries = 0
    
    for i in range(num_query):
        gnd = gnd_sim[i]
        
        if np.sum(gnd) == 0:
            continue
            
        valid_queries += 1
        sorted_indices = np.argsort(dist[i])
        
        for k in k_list:
            top_k_indices = sorted_indices[:k]
            results[k] += np.mean(gnd[top_k_indices])
    
    if valid_queries > 0:
        for k in k_list:
            results[k] /= valid_queries
    
    return results


def calculate_ndcg_at_k(
    qB: np.ndarray,
    rB: np.ndarray,
    query_labels: np.ndarray,
    db_labels: np.ndarray,
    k: int = 100,
) -> float:
    """
    Calculate NDCG@K (Normalized Discounted Cumulative Gain).
    
    For multi-label, relevance can be graded based on label overlap.
    """
    num_query = qB.shape[0]
    
    # Compute distances
    dist = hamming_distance(qB, rB)
    
    # Compute relevance scores (number of shared labels)
    relevance = np.dot(query_labels, db_labels.T)  # Number of shared labels
    
    ndcg_sum = 0.0
    valid_queries = 0
    
    for i in range(num_query):
        rel = relevance[i]
        
        if np.sum(rel) == 0:
            continue
            
        valid_queries += 1
        
        # Sort by predicted distance
        sorted_indices = np.argsort(dist[i])[:k]
        sorted_rel = rel[sorted_indices]
        
        # DCG
        positions = np.arange(1, k + 1)
        dcg = np.sum(sorted_rel / np.log2(positions + 1))
        
        # IDCG (ideal: sort by relevance)
        ideal_rel = np.sort(rel)[::-1][:k]
        idcg = np.sum(ideal_rel / np.log2(positions + 1))
        
        if idcg > 0:
            ndcg_sum += dcg / idcg
    
    if valid_queries == 0:
        return 0.0
        
    return ndcg_sum / valid_queries


def evaluate_multilabel(
    model: torch.nn.Module,
    query_loader: torch.utils.data.DataLoader,
    db_loader: torch.utils.data.DataLoader,
    device: torch.device,
    top_k: Optional[int] = None,
) -> dict:
    """
    Full evaluation for multi-label retrieval.
    
    Args:
        model: Hash model
        query_loader: Query DataLoader
        db_loader: Database DataLoader
        device: torch device
        top_k: Optional top-k for mAP computation
        
    Returns:
        dict with mAP, P@K scores
    """
    model.eval()
    
    # Extract query codes and labels
    print("Extracting query hash codes...")
    qB_list = []
    qL_list = []
    
    with torch.no_grad():
        for imgs, labels in tqdm(query_loader):
            imgs = imgs.to(device)
            hash_codes, _ = model(imgs)
            binary_codes = torch.sign(hash_codes).cpu().numpy()
            qB_list.append(binary_codes)
            qL_list.append(labels.numpy())
    
    qB = np.concatenate(qB_list)
    qL = np.concatenate(qL_list)
    
    # Extract database codes and labels
    print("Extracting database hash codes...")
    rB_list = []
    rL_list = []
    
    with torch.no_grad():
        for imgs, labels in tqdm(db_loader):
            imgs = imgs.to(device)
            hash_codes, _ = model(imgs)
            binary_codes = torch.sign(hash_codes).cpu().numpy()
            rB_list.append(binary_codes)
            rL_list.append(labels.numpy())
    
    rB = np.concatenate(rB_list)
    rL = np.concatenate(rL_list)
    
    print(f"Query: {qB.shape}, Database: {rB.shape}")
    
    # Calculate metrics
    print("Calculating mAP...")
    mAP = calculate_multilabel_map(qB, rB, qL, rL, top_k=top_k)
    
    print("Calculating P@K...")
    p_at_k = calculate_precision_at_k(qB, rB, qL, rL, k_list=[1, 5, 10, 50, 100])
    
    print("Calculating NDCG@100...")
    ndcg = calculate_ndcg_at_k(qB, rB, qL, rL, k=100)
    
    results = {
        'mAP': mAP,
        'NDCG@100': ndcg,
        **{f'P@{k}': v for k, v in p_at_k.items()},
        'num_query': qB.shape[0],
        'num_database': rB.shape[0],
    }
    
    print("\n" + "=" * 50)
    print("Multi-Label Retrieval Results")
    print("=" * 50)
    print(f"mAP: {mAP:.4f}")
    print(f"NDCG@100: {ndcg:.4f}")
    for k, v in p_at_k.items():
        print(f"P@{k}: {v:.4f}")
    print("=" * 50)
    
    return results


def get_retrieval_results(
    model: torch.nn.Module,
    query_image: torch.Tensor,
    query_label: torch.Tensor,
    db_loader: torch.utils.data.DataLoader,
    device: torch.device,
    top_k: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get retrieval results for a single query.
    
    Returns:
        top_indices: Indices of top-k retrieved images
        top_labels: Labels of top-k retrieved images
        relevance: Whether each retrieved image is relevant
    """
    model.eval()
    
    with torch.no_grad():
        # Get query hash
        query_hash, _ = model(query_image.unsqueeze(0).to(device))
        query_hash = torch.sign(query_hash).cpu().numpy()
        
        # Get database hashes
        db_hashes = []
        db_labels = []
        
        for imgs, labels in db_loader:
            imgs = imgs.to(device)
            hashes, _ = model(imgs)
            hashes = torch.sign(hashes).cpu().numpy()
            db_hashes.append(hashes)
            db_labels.append(labels.numpy())
        
        db_hashes = np.concatenate(db_hashes)
        db_labels = np.concatenate(db_labels)
        
        # Compute distances
        dist = hamming_distance(query_hash, db_hashes)[0]
        
        # Get top-k
        top_indices = np.argsort(dist)[:top_k]
        top_labels = db_labels[top_indices]
        
        # Compute relevance
        query_label_np = query_label.numpy()
        relevance = (np.dot(query_label_np, top_labels.T) > 0).astype(np.float32)
        
        return top_indices, top_labels, relevance


if __name__ == '__main__':
    # Test metrics
    np.random.seed(42)
    
    num_query = 100
    num_db = 1000
    hash_bit = 64
    num_classes = 21
    
    # Random binary codes
    qB = np.sign(np.random.randn(num_query, hash_bit))
    rB = np.sign(np.random.randn(num_db, hash_bit))
    
    # Random multi-labels (at least one label per image)
    qL = np.random.randint(0, 2, (num_query, num_classes)).astype(np.float32)
    qL[:, 0] = 1  # Ensure at least one label
    
    rL = np.random.randint(0, 2, (num_db, num_classes)).astype(np.float32)
    rL[:, 0] = 1
    
    print("Testing Multi-Label Metrics")
    print("=" * 50)
    
    mAP = calculate_multilabel_map(qB, rB, qL, rL)
    print(f"mAP: {mAP:.4f}")
    
    p_at_k = calculate_precision_at_k(qB, rB, qL, rL)
    for k, v in p_at_k.items():
        print(f"P@{k}: {v:.4f}")
    
    ndcg = calculate_ndcg_at_k(qB, rB, qL, rL, k=100)
    print(f"NDCG@100: {ndcg:.4f}")
