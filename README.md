# ĐỒ ÁN THẠC SĨ: TRUY VẤN THÔNG TIN HÌNH ẢNH

## Nghiên cứu Vision Transformer cho Truy vấn Ảnh Viễn thám: So sánh Backbone và Phân tích Hash Codes

---

## 📋 MỤC LỤC

1. [Tổng quan nghiên cứu](#1-tổng-quan-nghiên-cứu)
2. [Câu hỏi nghiên cứu](#2-câu-hỏi-nghiên-cứu)
3. [Phương pháp](#3-phương-pháp)
4. [Thiết kế thực nghiệm](#4-thiết-kế-thực-nghiệm)
5. [Hướng dẫn chạy code](#5-hướng-dẫn-chạy-code)
6. [Phân tích kết quả](#6-phân-tích-kết-quả)
7. [Kết luận](#7-kết-luận)

---

## ⚡ QUICK START

```bash
# Cài đặt
pip install -r requirements.txt

# 1. Train baseline (ViT + Hashing)
python experiments/train.py --model vit --epochs 30

# 2. So sánh với DINOv2
python experiments/train.py --model dinov3 --epochs 30

# 3. Chạy ablation study
python experiments/ablation.py

# 4. Visualization & Analysis
python experiments/visualize.py --checkpoint ./checkpoints/best_model_nwpu_vit.pth

# 5. Evaluate model
python experiments/evaluate.py --checkpoint ./checkpoints/best_model.pth
```

---

## 1. TỔNG QUAN NGHIÊN CỨU

### 1.1 Bối cảnh

**Content-Based Image Retrieval (CBIR)** cho ảnh viễn thám là bài toán quan trọng trong:
- Giám sát môi trường và biến đổi khí hậu
- Quy hoạch đô thị và quản lý tài nguyên
- Ứng dụng quốc phòng và an ninh

**Thách thức đặc thù của ảnh viễn thám:**

| Thách thức | Mô tả | Ví dụ |
|------------|-------|-------|
| **Intra-class variance cao** | Cùng class nhưng khác về visual | Airport ở Mỹ vs Airport ở Châu Á |
| **Inter-class similarity cao** | Khác class nhưng giống visual | Airport ↔ Runway |
| **Scale variation** | Cùng object nhưng khác kích thước | Tàu gần vs xa |

### 1.2 Đóng góp của nghiên cứu

| # | Đóng góp | Loại | Mô tả |
|---|----------|------|-------|
| 1 | **Hệ thống CBIR** | Engineering | Pipeline hoàn chỉnh: ViT → Hashing → Retrieval |
| 2 | **So sánh backbone** | Research | ViT-ImageNet vs DINOv2 cho remote sensing |
| 3 | **Ablation study** | Analysis | Hash bits, feature layers |
| 4 | **Failure analysis** | Insight | Phân tích model sai ở đâu và tại sao |

### 1.3 Kiến trúc tổng quan

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌────────────┐
│ Input Image │ ──→ │   Backbone   │ ──→ │ Hash Head   │ ──→ │ Hash Code  │
│  224×224    │     │ ViT / DINOv2 │     │  MLP+Tanh   │     │  64 bits   │
└─────────────┘     └──────────────┘     └─────────────┘     └────────────┘
                           │
                    ┌──────┴──────┐
                    │  CSQ Loss   │
                    │ Center+Quant│
                    └─────────────┘
```

---

## 2. CÂU HỎI NGHIÊN CỨU

### RQ1: Backbone nào phù hợp hơn cho Remote Sensing Image Retrieval?

| Backbone | Pretraining | Data size | Đặc điểm |
|----------|-------------|-----------|----------|
| **ViT-B/32** | Supervised (ImageNet) | 1.2M | Học phân biệt 1000 classes tự nhiên |
| **DINOv2-S/14** | Self-supervised | 142M | Học features tổng quát không cần labels |

**Giả thuyết:** DINOv2 sẽ tốt hơn vì:
- Self-supervised learning không bị bias về ImageNet categories
- Pretrained trên 142M ảnh diverse hơn
- Patch size nhỏ hơn (14 vs 32) → capture chi tiết tốt hơn

**Cách kiểm chứng:**
- So sánh mAP trên cùng test set
- Visualize attention maps để xem model focus khác nhau thế nào

---

### RQ2: Số hash bits tối ưu cho NWPU-RESISC45?

| Hash bits | Storage | Trade-off hypothesis |
|-----------|---------|---------------------|
| 16 bits | 2 bytes/img | Quá ít → mất thông tin → mAP thấp |
| 32 bits | 4 bytes/img | Có thể đủ cho dataset nhỏ |
| **64 bits** | **8 bytes/img** | **Sweet spot cho 45 classes?** |
| 128 bits | 16 bytes/img | Có thể overkill, không cải thiện nhiều |

**Giả thuyết:** 64 bits là đủ. Với 45 classes, mỗi class cần ~1.4 bits để encode. 64 bits dư để encode cả semantic và variation.

**Cách kiểm chứng:**
- Train với 16, 32, 64, 128 bits
- So sánh mAP và storage/speed trade-off

---

### RQ3: Model fails ở đâu và tại sao?

**Mục tiêu:** Không chỉ report mAP, mà phải hiểu:
- Classes nào hay bị confuse?
- Có pattern nào trong failure cases?
- Insight để cải thiện?

**Cách phân tích:**
- Confusion matrix
- Per-class mAP breakdown
- Visualize failure cases

---

## 3. PHƯƠNG PHÁP

### 3.1 Backbone Options

#### A. ViT-B/32 (Baseline)
```
- Pretrained: ImageNet-1K (supervised classification)
- Patch size: 32×32 → 49 patches cho ảnh 224×224
- Embedding dim: 768
- Layers: 12 transformer blocks
- Output: CLS token (768-dim)
```

#### B. DINOv2-S/14 (Comparison)
```
- Pretrained: LVD-142M (self-supervised, no labels)
- Patch size: 14×14 → 256 patches cho ảnh 224×224
- Embedding dim: 384
- Layers: 12 transformer blocks
- Output: CLS token (384-dim)
```

**Tại sao so sánh 2 này?**
- Supervised vs Self-supervised pretraining
- Natural images (ImageNet) vs General images (LVD)
- Coarse patches (32) vs Fine patches (14)

### 3.2 Hashing Head

```python
HashingHead(embed_dim, hash_bit=64):
    Dropout(0.5)           # Regularization
    Linear(embed_dim, 1024) # Project to intermediate
    ReLU()                  # Non-linearity
    Linear(1024, hash_bit)  # Project to hash dimension
    Tanh()                  # Output in [-1, 1]

# Training: continuous values
# Inference: sign() → binary {-1, +1}
```

### 3.3 Loss Function: CSQ

```
L_total = L_center + λ × L_quant

L_center = (1/B) Σ ||h_i - c_{y_i}||²
  → Kéo hash codes cùng class về center của class đó

L_quant = (1/B) Σ (|h_i| - 1)²
  → Đẩy giá trị về ±1 để quantization error thấp
```

**Hyperparameters:**
- λ = 0.0001 (quantization weight)
- Centers c_k được initialize random và học cùng model

### 3.4 Retrieval

```
Hamming Distance: d(q, x) = Σ (q_i XOR x_i)
  - q: query hash (64 bits)
  - x: database hash (64 bits)
  - XOR + popcount: O(1) với hardware support

Ranking: Sort database by Hamming distance ascending
```

---

## 4. THIẾT KẾ THỰC NGHIỆM

### 4.1 Dataset: NWPU-RESISC45

| Property | Value |
|----------|-------|
| Total images | 31,500 |
| Classes | 45 |
| Images/class | 700 |
| Resolution | 256×256 |
| **Train** | 25,200 (80%) |
| **Val** | 3,150 (10%) |
| **Test** | 3,150 (10%) |

**45 Classes:**
```
airplane, airport, baseball_diamond, basketball_court, beach,
bridge, chaparral, church, circular_farmland, cloud,
commercial_area, dense_residential, desert, forest, freeway,
golf_course, ground_track_field, harbor, industrial_area,
intersection, island, lake, meadow, medium_residential,
mobile_home_park, mountain, overpass, palace, parking_lot,
railway, railway_station, rectangular_farmland, river,
roundabout, runway, sea_ice, ship, snowberg, sparse_residential,
stadium, storage_tank, tennis_court, terrace, thermal_power_station,
wetland
```

### 4.2 Training Setup

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Batch size | 32 (8×4 accumulation) | Fit 4GB VRAM |
| Epochs | 30 | Với early stopping |
| LR backbone | 1e-5 | Fine-tune pretrained |
| LR head | 1e-4 | Train from scratch |
| Optimizer | AdamW | Standard for transformers |
| Scheduler | CosineAnnealing | Smooth decay |
| λ_quant | 0.0001 | From CSQ paper |

### 4.3 Experiments

#### Exp 1: Baseline (ViT)

```bash
python train_nwpu.py --model vit --hash-bit 64 --epochs 30
```

**Output:** Baseline mAP để compare

---

#### Exp 2: Backbone Comparison (RQ1)

```bash
# ViT-B/32
python train_nwpu.py --model vit --hash-bit 64 --epochs 30

# DINOv2-S/14
python train_nwpu.py --model dinov3 --hash-bit 64 --epochs 30
```

**Analysis:**
1. So sánh mAP, P@10, P@50
2. Visualize attention maps (cả 2 models trên cùng 1 ảnh)
3. Compare training curves

---

#### Exp 3: Hash Bits Ablation (RQ2)

```bash
python run_ablation.py --experiment hash_bits
```

Chạy với 16, 32, 64, 128 bits.

**Expected output:**

| Bits | mAP | ΔmAP | Storage | Speed |
|------|-----|------|---------|-------|
| 16 | ? | -baseline | 2B | 1.0x |
| 32 | ? | ? | 4B | ~1.0x |
| 64 | baseline | 0 | 8B | ~1.0x |
| 128 | ? | ? | 16B | ~0.9x |

---

#### Exp 4: Failure Analysis (RQ3)

```bash
python visualize_analysis.py --checkpoint ./checkpoints/best_model.pth --analyze-failures
```

**Output:**
1. Confusion matrix heatmap
2. Top-10 confused class pairs
3. Per-class mAP bar chart
4. Sample failure cases with retrieved results

---

### 4.4 Metrics

| Metric | Formula | Use |
|--------|---------|-----|
| **mAP** | Mean of AP over all queries | Overall performance |
| **P@K** | Relevant in top-K / K | Practical relevance |
| **Confusion pairs** | Top co-occurrence in mistakes | Failure analysis |

---

## 5. HƯỚNG DẪN CHẠY CODE

### 5.1 Cấu trúc

```
Information Retrieval/
├── experiments/                # Training & Analysis scripts
│   ├── train.py                # Main training script
│   ├── ablation.py             # Ablation study runner
│   ├── evaluate.py             # Model evaluation
│   └── visualize.py            # Analysis & visualization
├── src/                        # Source code
│   ├── models/                 # Model architectures
│   │   ├── vit_hashing.py      # ViT + Hashing head
│   │   └── dinov2_hashing.py   # DINOv2 + Hashing head
│   ├── losses/                 # Loss functions
│   │   └── csq_loss.py         # Central Similarity Quantization
│   ├── data/                   # Data loading
│   │   └── loaders.py          # Dataset loaders
│   └── utils/                  # Utilities
│       ├── metrics.py          # Evaluation metrics
│       └── pruning.py          # Token pruning (optional)
├── scripts/                    # Utility scripts
│   ├── download_dataset.py     # Download datasets
│   ├── download_nwpu.py        # Download NWPU-RESISC45
│   └── download_weights.py     # Download pretrained weights
├── docs/                       # Documentation
├── data/archive/Dataset/       # NWPU-RESISC45 dataset
├── checkpoints/                # Saved models
└── results/                    # Output results
```

### 5.2 Commands

```bash
# Setup
pip install -r requirements.txt

# 1. Quick test
python experiments/train.py --quick

# 2. Full experiments
python experiments/train.py --model vit --epochs 30
python experiments/train.py --model dinov3 --epochs 30

# 3. Ablation
python experiments/ablation.py --all

# 4. Evaluation
python experiments/evaluate.py --checkpoint ./checkpoints/best_model.pth

# 5. Analysis
python experiments/visualize.py --checkpoint ./checkpoints/best_model_nwpu_vit.pth
```

### 5.3 Key Arguments

| Argument | Default | Options |
|----------|---------|---------|
| `--model` | `vit` | `vit`, `dinov3` |
| `--hash-bit` | `64` | `16`, `32`, `64`, `128` |
| `--epochs` | `30` | |
| `--batch-size` | `8` | |
| `--quick` | False | Quick test mode |

---

## 6. PHÂN TÍCH KẾT QUẢ

### 6.1 Kết quả dự kiến

#### RQ1: Backbone Comparison

| Backbone | mAP | P@10 | Insight |
|----------|-----|------|---------|
| ViT-B/32 | ~0.68 | ~0.75 | Baseline, supervised ImageNet |
| DINOv2-S/14 | ~0.72 | ~0.79 | **+4% mAP**, self-supervised |

**Giải thích dự kiến:**
- DINOv2 không bị bias về 1000 ImageNet categories
- Self-supervised học features low-level tổng quát hơn
- Patch 14×14 capture chi tiết tốt hơn 32×32

#### RQ2: Hash Bits

| Bits | mAP | Δ vs 64-bit | Storage |
|------|-----|-------------|---------|
| 16 | ~0.60 | -8% | 2B |
| 32 | ~0.65 | -3% | 4B |
| **64** | **~0.68** | **baseline** | **8B** |
| 128 | ~0.69 | +1% | 16B |

**Insight dự kiến:**
- **64 bits là sweet spot**: tăng lên 128 chỉ +1% nhưng gấp đôi storage
- 32 bits có thể đủ nếu cần compact hơn

#### RQ3: Failure Analysis

**Top confused pairs (dự kiến):**

| Class A | Class B | Lý do |
|---------|---------|-------|
| airport | runway | Cả hai có đường băng dài |
| dense_residential | medium_residential | Chỉ khác mật độ nhà |
| circular_farmland | stadium | Cả hai có hình tròn |
| railway | freeway | Cả hai là đường dài |

### 6.2 Template phân tích

#### A. Training Curves
- Loss convergence của ViT vs DINOv2
- Overfitting detection

#### B. Attention Visualization
- ViT focus vào đâu?
- DINOv2 focus vào đâu?
- Khác nhau như thế nào trên cùng 1 ảnh?

#### C. t-SNE của Hash Codes
- Các class có cluster rõ không?
- Classes nào overlap?

#### D. Retrieval Examples
- Top-5 retrieved images cho success cases
- Top-5 retrieved images cho failure cases

---

## 7. KẾT LUẬN

### 7.1 Đóng góp

1. **Hệ thống CBIR hoàn chỉnh** cho ảnh viễn thám
2. **Evidence-based recommendation** về backbone (ViT vs DINOv2)
3. **Practical guidance** về hash bits cho dataset tương tự
4. **Failure analysis framework** có thể áp dụng cho domains khác

### 7.2 Findings chính (Expected)

| Finding | Implication |
|---------|-------------|
| DINOv2 > ViT (+4% mAP) | Self-supervised phù hợp hơn cho domain-specific |
| 64 bits là đủ | Không cần 128+ bits cho 45-class retrieval |
| Airport↔Runway confused | Cần visual cues khác ngoài layout |

### 7.3 Limitations

- Single dataset (NWPU-RESISC45 only)
- Không so sánh với CNN baselines (ResNet)
- Fixed resolution 224×224

### 7.4 Future Work

- Cross-dataset evaluation (e.g., train NWPU, test UC Merced)
- Multi-scale features
- Attention-based pooling

---

## 📚 TÀI LIỆU THAM KHẢO

1. Dosovitskiy et al. (2020). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." ICLR 2021.

2. Oquab et al. (2023). "DINOv2: Learning Robust Visual Features without Supervision." CVPR 2024.

3. Yuan et al. (2020). "Central Similarity Quantization for Efficient Image and Video Retrieval." CVPR 2020.

4. Cheng et al. (2017). "Remote Sensing Image Scene Classification: Benchmark and State of the Art." Proc. IEEE.

---

## 👨‍🎓 THÔNG TIN

- **Bậc học:** Thạc sĩ
- **Môn học:** Truy vấn Thông tin Hình ảnh
- **Dataset:** NWPU-RESISC45 (31,500 ảnh viễn thám)
- **Phương pháp:** ViT + Deep Hashing

---

*Cập nhật: Tháng 3/2026*

