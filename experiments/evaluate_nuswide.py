"""
Full Benchmark Evaluation on NUS-WIDE (standard hashing protocol)
==================================================================

Loads a trained checkpoint and evaluates mAP@ALL on the full
NUS-WIDE split identical to what CSQ, HashNet, GreedyHash papers use:

    Query    :   2,100 images  (test_img.txt)
    Database : 193,734 images  (database_img.txt)
    Relevance: share >= 1 label (multi-label, 21 concepts)

Usage:
    python experiments/evaluate_nuswide.py \\
        --checkpoint ./checkpoints/best_model_nuswide_vit_64bit.pth

    # Quick sanity check (1000 DB images):
    python experiments/evaluate_nuswide.py \\
        --checkpoint ./checkpoints/best_model_nuswide_vit_64bit.pth \\
        --quick
"""

import os, sys, argparse
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.nuswide_loader import NUSWIDEPreprocessedDataset
from src.models.vit_hashing import ViT_Hashing
from src.models.dinov2_hashing import DINOv3Hashing
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Hamming distance + mAP
# ---------------------------------------------------------------------------

def hamming_dist(qB: np.ndarray, rB: np.ndarray) -> np.ndarray:
    """Hamming distance matrix. Inputs: {-1,+1} codes."""
    return 0.5 * (qB.shape[1] - qB @ rB.T)


def calc_map(qB, rB, qL, rL, top_k=None):
    """
    Mean Average Precision for multi-label retrieval.

    Relevance: two images are relevant if they share >= 1 label.
    top_k: if set, only top-k retrieved items are scored (mAP@top_k).
    """
    dist   = hamming_dist(qB, rB)            # [Q, DB]
    S      = (qL @ rL.T) > 0                 # [Q, DB] boolean

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
        cumsum  = np.cumsum(gnd_sorted)
        prec    = cumsum / (np.arange(len(gnd_sorted)) + 1.0)
        ap      = (prec * gnd_sorted).sum() / n_pos
        aps.append(ap)

    return float(np.mean(aps)) if aps else 0.0


def precision_at_k(qB, rB, qL, rL, k=100):
    dist = hamming_dist(qB, rB)
    S    = (qL @ rL.T) > 0
    precs = []
    for i in range(len(qB)):
        rank = np.argsort(dist[i])[:k]
        precs.append(S[i][rank].mean())
    return float(np.mean(precs))


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_codes(model, loader, device):
    model.eval()
    codes, labels = [], []
    for imgs, lbl in tqdm(loader, desc="Extracting", leave=False):
        h, _ = model(imgs.to(device))
        codes.append(torch.sign(h).cpu().numpy())
        labels.append(lbl.numpy())
    return np.concatenate(codes), np.concatenate(labels)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True,
                        help='Path to .pth checkpoint')
    parser.add_argument('--data-dir', default='./src/data/archive/NUS-WIDE',
                        help='NUS-WIDE archive directory')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--quick', action='store_true',
                        help='Use 1000 DB images for fast sanity check')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Load checkpoint ──────────────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location=device)
    hash_bit   = ckpt.get('hash_bit', 64)
    model_type = ckpt.get('model_type', 'vit')
    print(f"Checkpoint: {args.checkpoint}")
    print(f"  model={model_type}, hash_bit={hash_bit}")

    if model_type == 'vit':
        model = ViT_Hashing(hash_bit=hash_bit)
    else:
        model = DINOv3Hashing(hash_bit=hash_bit)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)

    # ── Data ─────────────────────────────────────────────────────────────
    query_ds = NUSWIDEPreprocessedDataset(args.data_dir, 'query')
    db_ds    = NUSWIDEPreprocessedDataset(args.data_dir, 'database')

    if args.quick:
        from torch.utils.data import Subset
        db_ds = Subset(db_ds, list(range(1000)))
        print("[Quick] Database limited to 1000 images")

    query_loader = DataLoader(query_ds, args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    db_loader    = DataLoader(db_ds,    args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # ── Extract ──────────────────────────────────────────────────────────
    print("\nExtracting query codes ...")
    qB, qL = extract_codes(model, query_loader, device)
    print(f"  query  : {qB.shape}")

    print("Extracting database codes ...")
    rB, rL = extract_codes(model, db_loader, device)
    print(f"  database: {rB.shape}")

    # ── Evaluate ─────────────────────────────────────────────────────────
    print("\nComputing metrics ...")
    map_all  = calc_map(qB, rB, qL, rL)
    map_1000 = calc_map(qB, rB, qL, rL, top_k=1000)
    p_at_100 = precision_at_k(qB, rB, qL, rL, k=100)

    print("\n" + "=" * 50)
    print(f"  mAP@ALL   : {map_all:.4f}")
    print(f"  mAP@1000  : {map_1000:.4f}")
    print(f"  P@100     : {p_at_100:.4f}")
    print("=" * 50)
    print("\n[Reference — CSQ paper on NUS-WIDE, 64-bit]")
    print("  CSQ (ResNet-50) : mAP@ALL = 0.748")
    print("  HashNet (AlexNet): mAP@ALL = 0.618")


if __name__ == '__main__':
    main()
