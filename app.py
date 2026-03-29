"""
Streamlit Web App — Image Retrieval with Deep Hashing
=====================================================

Run:
    streamlit run app.py

Requires:
    - A trained checkpoint  (e.g. checkpoints/best_model_nwpu_vit.pth)
    - A vector database     (e.g. database/nwpu_vectors.npz)
      OR the raw image folders to build one on the fly.
"""

import os
import sys
import glob
import random
from pathlib import Path

import numpy as np
import torch
import streamlit as st
from PIL import Image
from torchvision import transforms

# ── project imports ──────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.models.vit_hashing import ViT_Hashing
from src.models.dinov2_hashing import DINOv3Hashing

# ── defaults ─────────────────────────────────────────────────
DEFAULT_CHECKPOINT = "./checkpoints/best_model_nwpu_vit.pth"
DEFAULT_DATABASE   = "./database/nwpu_vectors.npz"
DATASET_ROOT       = "./data/archive/Dataset"

# ═════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════

def get_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


@st.cache_resource(show_spinner="Loading model …")
def load_model(checkpoint_path: str):
    """Load model — auto-detect type from state_dict."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint["model_state_dict"]

    has_layers = any("hashing_head.layers." in k for k in state)

    if has_layers:
        model_type = "dinov3"
        hash_bit   = state["hashing_head.layers.3.weight"].shape[0]
        embed_dim  = state["hashing_head.layers.1.weight"].shape[1]
        hidden_dim = state["hashing_head.layers.1.weight"].shape[0]
        dmap = {384: "vit_small_patch14_dinov2.lvd142m",
                768: "vit_base_patch14_dinov2.lvd142m",
                1024: "vit_large_patch14_dinov2.lvd142m"}
        model_name = dmap.get(embed_dim, "vit_small_patch14_dinov2.lvd142m")
        model = DINOv3Hashing(model_name=model_name, pretrained=False,
                              hash_bit=hash_bit, hidden_dim=hidden_dim)
    else:
        model_type = "vit"
        hash_bit   = state["hashing_head.3.weight"].shape[0]
        pos_len    = state["backbone.pos_embed"].shape[1]
        model_name = "vit_base_patch32_224" if pos_len == 50 else "vit_base_patch16_224"
        model = ViT_Hashing(model_name=model_name, pretrained=False, hash_bit=hash_bit)

    model.load_state_dict(state)
    model.to(device).eval()

    epoch = checkpoint.get("epoch", "?")
    mAP   = checkpoint.get("metrics", {}).get("mAP", "N/A")
    info  = {"model_type": model_type, "model_name": model_name,
             "hash_bit": hash_bit, "epoch": epoch, "mAP": mAP, "device": device}
    return model, info


@st.cache_resource(show_spinner="Loading vector database …")
def load_database(db_path: str):
    """Load pre-built .npz database."""
    data = np.load(db_path, allow_pickle=True)
    return {
        "hash_codes":  data["hash_codes"],          # [N, hash_bit]  {-1, 1}
        "image_paths": data["image_paths"].tolist(),
        "labels":      data["labels"],
        "class_names": data["class_names"].tolist(),
        "hash_bit":    int(data["hash_bit"]),
        "model_type":  str(data["model_type"]),
    }


@st.cache_data(show_spinner="Scanning dataset folders …")
def scan_dataset_classes(dataset_root: str):
    """Return {class_name: [image_paths]} from train+test."""
    classes = {}
    for split in ["train/train", "test/test"]:
        root = os.path.join(dataset_root, split)
        if not os.path.isdir(root):
            continue
        for cls in sorted(os.listdir(root)):
            cls_dir = os.path.join(root, cls)
            if not os.path.isdir(cls_dir):
                continue
            imgs = sorted(glob.glob(os.path.join(cls_dir, "*.*")))
            classes.setdefault(cls, []).extend(imgs)
    return classes


def hamming_distance(query: np.ndarray, database: np.ndarray) -> np.ndarray:
    hash_bit = query.shape[0]
    return (0.5 * (hash_bit - database @ query)).astype(np.int32)


@torch.no_grad()
def extract_hash(image: Image.Image, model, device):
    """PIL Image → binary hash code np.array [hash_bit]."""
    tensor = get_transform()(image.convert("RGB")).unsqueeze(0).to(device)
    h, _ = model(tensor)
    return torch.sign(h).cpu().numpy()[0]


def search(query_hash, db, top_k):
    dists   = hamming_distance(query_hash, db["hash_codes"])
    indices = np.argsort(dists)[:top_k]
    results = []
    for rank, idx in enumerate(indices, 1):
        label = int(db["labels"][idx])
        results.append({
            "rank":     rank,
            "distance": int(dists[idx]),
            "path":     db["image_paths"][idx],
            "label":    label,
            "class":    db["class_names"][label],
        })
    return results


def render_hash_viz(hash_code: np.ndarray):
    """Render hash code as a colored bar (green=+1, red=−1)."""
    bits = ((hash_code + 1) / 2).astype(np.uint8)  # {0,1}
    h, w = 4, len(bits)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for i, b in enumerate(bits):
        img[:, i] = [46, 204, 113] if b else [231, 76, 60]
    return Image.fromarray(img).resize((w * 6, h * 6), Image.NEAREST)


# ═════════════════════════════════════════════════════════════
# STREAMLIT UI
# ═════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="Image Retrieval — Deep Hashing",
        page_icon="🔍",
        layout="wide",
    )

    st.title("🔍 Image Retrieval with Deep Hashing")
    st.caption("ViT / DINOv2 + CSQ Hashing  ·  Hamming Distance Search")

    # ── sidebar ──────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")

        ckpt_path = st.text_input("Checkpoint", value=DEFAULT_CHECKPOINT)
        db_path   = st.text_input("Database (.npz)", value=DEFAULT_DATABASE)
        top_k     = st.slider("Top-K results", 5, 100, 20, step=5)

        st.divider()
        st.header("📊 Query mode")
        mode = st.radio("Choose", ["Upload image", "Random from dataset",
                                    "Browse by class"], label_visibility="collapsed")

    # ── load model ───────────────────────────────────────────
    if not os.path.exists(ckpt_path):
        st.error(f"Checkpoint not found: `{ckpt_path}`")
        st.stop()

    model, model_info = load_model(ckpt_path)
    device = model_info["device"]

    # show model info
    with st.sidebar:
        st.divider()
        st.markdown("**Model info**")
        st.write(f"Type: `{model_info['model_type']}` ({model_info['model_name']})")
        st.write(f"Hash bits: `{model_info['hash_bit']}`")
        st.write(f"Epoch: `{model_info['epoch']}`  ·  Val mAP: `{model_info['mAP']}`")
        st.write(f"Device: `{device}`")

    # ── load database ────────────────────────────────────────
    if not os.path.exists(db_path):
        st.warning(f"Database not found: `{db_path}`")
        st.info("Build it first:\n```\npython scripts/build_vector_db.py build "
                "--checkpoint ./checkpoints/best_model_nwpu_vit.pth --full "
                "--output ./database/nwpu_vectors.npz\n```")

        # offer on-the-fly build
        st.divider()
        st.subheader("Or build now (may be slow on CPU)")
        if st.button("🔨 Build database from full dataset"):
            _build_database_ui(ckpt_path, db_path, model, model_info, device)
            st.rerun()
        st.stop()

    db = load_database(db_path)

    with st.sidebar:
        st.markdown(f"**Database**: {len(db['image_paths']):,} images  ·  "
                    f"{db['hash_bit']} bits")

    # ── get query image ──────────────────────────────────────
    query_image = None
    query_class = None

    if mode == "Upload image":
        uploaded = st.file_uploader("Upload a query image",
                                    type=["jpg", "jpeg", "png", "tif", "bmp"])
        if uploaded:
            query_image = Image.open(uploaded)

    elif mode == "Random from dataset":
        classes_map = scan_dataset_classes(DATASET_ROOT)
        if not classes_map:
            st.error("Dataset not found at " + DATASET_ROOT)
            st.stop()

        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🎲 New random"):
                st.session_state["rand_seed"] = random.randint(0, 1_000_000)

        seed = st.session_state.get("rand_seed", 42)
        rng  = random.Random(seed)
        all_imgs = [p for paths in classes_map.values() for p in paths]
        chosen   = rng.choice(all_imgs)
        query_class = Path(chosen).parent.name
        query_image = Image.open(chosen)

    elif mode == "Browse by class":
        classes_map = scan_dataset_classes(DATASET_ROOT)
        if not classes_map:
            st.error("Dataset not found")
            st.stop()

        cls_name = st.selectbox("Select class", sorted(classes_map.keys()))
        imgs     = classes_map[cls_name]

        idx = st.slider("Image index", 0, len(imgs) - 1, 0)
        chosen = imgs[idx]
        query_class = cls_name
        query_image = Image.open(chosen)

    if query_image is None:
        st.info("👆 Select or upload an image to start retrieval")
        st.stop()

    # ── run retrieval ────────────────────────────────────────
    query_hash = extract_hash(query_image, model, device)
    results    = search(query_hash, db, top_k)

    # ── display query ────────────────────────────────────────
    st.divider()
    col_q, col_h = st.columns([1, 3])
    with col_q:
        st.subheader("Query")
        st.image(query_image, use_container_width=True)
        if query_class:
            st.markdown(f"**Class:** `{query_class}`")
    with col_h:
        st.subheader("Hash code")
        st.image(render_hash_viz(query_hash), use_container_width=True)

        bits_str = "".join("1" if b > 0 else "0" for b in query_hash)
        st.code(bits_str, language=None)

    # ── precision metric ─────────────────────────────────────
    if query_class:
        correct = sum(1 for r in results if r["class"] == query_class)
        precision = correct / len(results) if results else 0
        st.metric(f"Precision@{top_k}", f"{precision:.1%}",
                  delta=f"{correct}/{len(results)} correct")

    # ── result grid ──────────────────────────────────────────
    st.divider()
    st.subheader(f"Top-{top_k} Results")

    cols_per_row = 5
    for row_start in range(0, len(results), cols_per_row):
        cols = st.columns(cols_per_row)
        for col, r in zip(cols, results[row_start:row_start + cols_per_row]):
            with col:
                is_match = query_class and r["class"] == query_class
                border_color = "#2ecc71" if is_match else "#e74c3c"
                if query_class is None:
                    border_color = "#3498db"

                try:
                    img = Image.open(r["path"])
                    st.image(img, use_container_width=True)
                except Exception:
                    st.warning(f"Cannot load\n{r['path']}")

                label = f"**#{r['rank']}**  d={r['distance']}"
                if is_match:
                    label += " ✅"
                elif query_class:
                    label += " ❌"
                st.markdown(label)
                st.caption(r["class"])

    # ── distance histogram ───────────────────────────────────
    st.divider()
    with st.expander("📊 Distance distribution", expanded=False):
        all_dists = hamming_distance(query_hash, db["hash_codes"])
        import pandas as pd
        df = pd.DataFrame({"Hamming distance": all_dists})
        st.bar_chart(df["Hamming distance"].value_counts().sort_index())

    # ── per-class breakdown ──────────────────────────────────
    if query_class:
        with st.expander("📋 Per-class breakdown in results", expanded=False):
            from collections import Counter
            cls_counts = Counter(r["class"] for r in results)
            import pandas as pd
            df = pd.DataFrame(cls_counts.most_common(),
                              columns=["Class", "Count"])
            st.dataframe(df, use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════
# ON-THE-FLY DATABASE BUILD
# ═════════════════════════════════════════════════════════════

def _build_database_ui(ckpt_path, db_path, model, model_info, device):
    """Build the vector database with a progress bar inside Streamlit."""
    from torch.utils.data import DataLoader, ConcatDataset
    from torchvision.datasets import ImageFolder
    from datetime import datetime

    transform = get_transform()
    data_dirs = [
        os.path.join(DATASET_ROOT, "train", "train"),
        os.path.join(DATASET_ROOT, "test", "test"),
    ]

    datasets = []
    all_paths = []
    class_names = None

    for d in data_dirs:
        if not os.path.isdir(d):
            continue
        ds = ImageFolder(d, transform=transform)
        datasets.append(ds)
        all_paths.extend([s[0] for s in ds.samples])
        if class_names is None:
            class_names = ds.classes

    combined = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    loader   = DataLoader(combined, batch_size=32, shuffle=False, num_workers=0)

    total_batches = len(loader)
    progress = st.progress(0, text="Extracting hash codes …")

    all_codes, all_labels = [], []
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(loader):
            imgs = imgs.to(device)
            h, _ = model(imgs)
            all_codes.append(torch.sign(h).cpu().numpy())
            all_labels.append(labels.numpy())
            progress.progress((i + 1) / total_batches,
                              text=f"Batch {i+1}/{total_batches}")

    hash_codes = np.concatenate(all_codes)
    labels_arr = np.concatenate(all_labels)

    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    np.savez_compressed(
        db_path,
        hash_codes=hash_codes,
        image_paths=np.array(all_paths, dtype=object),
        labels=labels_arr,
        class_names=np.array(class_names, dtype=object),
        hash_bit=model_info["hash_bit"],
        model_type=model_info["model_type"],
        created_at=datetime.now().isoformat(),
    )
    progress.empty()
    st.success(f"Database saved: {db_path}  ({len(all_paths):,} images)")


if __name__ == "__main__":
    main()
