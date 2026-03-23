# ĐỒ ÁN MÔN HỌC: TRUY VẤN THÔNG TIN HÌNH ẢNH

## Hệ thống Truy vấn Ảnh Viễn thám sử dụng Vision Transformer và Deep Hashing

---

## 📋 MỤC LỤC

1. [Tổng quan](#1-tổng-quan)
2. [Bài toán Image Retrieval](#2-bài-toán-image-retrieval)
3. [Phương pháp](#3-phương-pháp)
4. [Thực nghiệm](#4-thực-nghiệm)
5. [Hướng dẫn chạy code](#5-hướng-dẫn-chạy-code)
6. [Kết quả](#6-kết-quả)

---

## ⚡ QUICK START

```bash
# Cài đặt
pip install -r requirements.txt

# Train với ViT (cơ bản)
python train_nwpu.py --model vit --epochs 30

# Train với DINOv2 (so sánh)
python train_nwpu.py --model dinov3 --epochs 30

# Quick test (3 epochs)
python train_nwpu.py --quick
```

---

## 1. TỔNG QUAN

### 1.1 Đề tài

**Xây dựng hệ thống Content-Based Image Retrieval (CBIR) cho ảnh viễn thám** sử dụng Vision Transformer kết hợp Deep Hashing.

### 1.2 Mục tiêu

| # | Mục tiêu | Đo lường |
|---|----------|----------|
| 1 | Xây dựng hệ thống CBIR **hoạt động** | Input ảnh → Output top-K ảnh tương tự |
| 2 | Đạt **mAP ≥ 0.65** trên NWPU-RESISC45 | Mean Average Precision |
| 3 | **(Optional)** So sánh ViT vs DINOv2 backbone | Bảng so sánh mAP |

### 1.3 Tại sao chọn đề tài này?

**Truy vấn ảnh viễn thám** có ứng dụng thực tế:
- Giám sát môi trường, biến đổi khí hậu
- Quy hoạch đô thị, nông nghiệp
- Phát hiện đối tượng (tàu, máy bay, công trình)

**Deep Hashing** là kỹ thuật hiệu quả cho retrieval:
- Chuyển ảnh → mã nhị phân (64 bits)
- Tìm kiếm bằng Hamming distance (rất nhanh)
- Tiết kiệm bộ nhớ lưu trữ

---

## 2. BÀI TOÁN IMAGE RETRIEVAL

### 2.1 Định nghĩa

```
INPUT:  Query image (ảnh truy vấn)
        Database (N ảnh trong cơ sở dữ liệu)
        
OUTPUT: Top-K ảnh trong database tương tự nhất với query
```

### 2.2 Ví dụ

```
Query: Ảnh sân bay (airport_001.jpg)
Database: 31,500 ảnh NWPU-RESISC45

Output (Top-5):
  1. airport_045.jpg  ✓ Relevant
  2. airport_123.jpg  ✓ Relevant  
  3. runway_089.jpg   ✗ (gần giống nhưng khác class)
  4. airport_234.jpg  ✓ Relevant
  5. airport_012.jpg  ✓ Relevant
  
→ Precision@5 = 4/5 = 0.80
```

### 2.3 Tại sao dùng Hashing?

| Phương pháp | Feature size | Phép tính | Tốc độ |
|-------------|--------------|-----------|--------|
| **Float features** | 768 × 4 = 3KB/ảnh | Cosine similarity | Chậm |
| **Binary hash** | 64 bits = 8 bytes/ảnh | XOR + count | **Nhanh** |

**Với database 1 triệu ảnh:**
- Float: 3GB storage, ~100ms/query
- Hash: 8MB storage, ~1ms/query

---

## 3. PHƯƠNG PHÁP

### 3.1 Kiến trúc tổng quan

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐      ┌────────────┐
│ Input Image │ ───→ │ ViT Backbone │ ───→ │ Hash Head   │ ───→ │ Hash Code  │
│ 224×224×3   │      │ (feature     │      │ (MLP+Tanh)  │      │ 64 bits    │
└─────────────┘      │  extraction) │      └─────────────┘      └────────────┘
                     └──────────────┘
```

### 3.2 Các thành phần

#### A. Backbone: Vision Transformer

**ViT-B/32 (cơ bản):**
- Pretrained trên ImageNet (1.2M ảnh)
- Chia ảnh thành patches 32×32
- Output: feature vector 768-dim

**DINOv2 (so sánh - optional):**
- Pretrained trên 142M ảnh (self-supervised)
- Chia ảnh thành patches 14×14
- Có thể tốt hơn cho domain-specific images

#### B. Hashing Head

```python
Hashing Head:
  Input (768-dim) 
  → Dropout(0.5) 
  → Linear(768, 1024) 
  → ReLU 
  → Linear(1024, 64) 
  → Tanh 
  → Output (64-dim, range [-1, 1])

Inference: sign(output) → binary {-1, +1}
```

#### C. Loss Function: CSQ (Central Similarity Quantization)

```
Total Loss = Center Loss + λ × Quantization Loss

Center Loss: Kéo hash codes cùng class về gần nhau
Quant Loss: Đẩy giá trị về ±1 để binary hóa tốt hơn
```

### 3.3 Retrieval Process

```
1. INDEXING (offline - chạy 1 lần):
   - Với mỗi ảnh trong database: image → model → hash code
   - Lưu tất cả hash codes
   
2. QUERY (online - mỗi lần tìm kiếm):
   - Query image → model → query hash
   - Tính Hamming distance với tất cả hash codes trong database
   - Trả về top-K ảnh có distance nhỏ nhất
```

---

## 4. THỰC NGHIỆM

### 4.1 Dataset: NWPU-RESISC45

| Thông tin | Giá trị |
|-----------|---------|
| **Tổng ảnh** | 31,500 |
| **Classes** | 45 (airport, beach, bridge, forest, ...) |
| **Resolution** | 256×256 pixels |
| **Train/Test** | 80% / 20% |

**45 Classes viễn thám:**
```
airplane, airport, baseball_diamond, basketball_court, beach,
bridge, chaparral, church, circular_farmland, cloud, ...
```

### 4.2 Thiết lập

| Parameter | Giá trị |
|-----------|---------|
| Batch size | 8 (accumulation 4 → effective 32) |
| Epochs | 30 |
| Learning rate | 1e-4 (head), 1e-5 (backbone) |
| Hash bits | 64 |
| Optimizer | AdamW |

### 4.3 Experiments

#### Experiment 1: Baseline với ViT

```bash
python train_nwpu.py --model vit --hash-bit 64 --epochs 30
```

**Mục tiêu:** Xây dựng hệ thống CBIR hoạt động, đạt mAP ≥ 0.65

#### Experiment 2: So sánh ViT vs DINOv2 (Optional)

```bash
# ViT-B/32 (ImageNet pretrained)
python train_nwpu.py --model vit --epochs 30

# DINOv2-S/14 (142M images pretrained)
python train_nwpu.py --model dinov3 --epochs 30
```

**Mục tiêu:** Xem backbone nào phù hợp hơn cho ảnh viễn thám

#### Experiment 3: Hash bits ablation (Optional)

```bash
python train_nwpu.py --model vit --hash-bit 16
python train_nwpu.py --model vit --hash-bit 32
python train_nwpu.py --model vit --hash-bit 64
python train_nwpu.py --model vit --hash-bit 128
```

**Mục tiêu:** Xác định số bits tối ưu (trade-off accuracy vs storage)

### 4.4 Metrics

| Metric | Công thức | Ý nghĩa |
|--------|-----------|---------|
| **mAP** | Mean Average Precision | Độ chính xác tổng thể |
| **P@K** | Precision at K | % relevant trong top-K |

### 4.5 Kết quả mong đợi

| Model | mAP (64-bit) | Ghi chú |
|-------|--------------|---------|
| ViT-B/32 | 0.65 - 0.70 | Baseline |
| DINOv2-S/14 | 0.68 - 0.73 | Pretrained tốt hơn? |

---

## 5. HƯỚNG DẪN CHẠY CODE

### 5.1 Cài đặt

```bash
# Clone và cài dependencies
cd "Information Retrieval"
pip install -r requirements.txt
```

### 5.2 Cấu trúc thư mục

```
Information Retrieval/
├── train_nwpu.py         # 🔥 Script training chính
├── src/
│   ├── model.py          # ViT_Hashing model
│   ├── loss.py           # CSQ Loss
│   └── research/
│       └── dinov3_hashing.py  # DINOv2 backbone (optional)
├── data/
│   └── archive/Dataset/  # NWPU-RESISC45
└── checkpoints/          # Saved models
```

### 5.3 Training

```bash
# Train cơ bản với ViT
python train_nwpu.py --model vit --epochs 30

# Quick test (3 epochs)
python train_nwpu.py --quick

# So sánh với DINOv2 (optional)
python train_nwpu.py --model dinov3 --epochs 30

# Nếu GPU yếu (4GB VRAM)
python train_nwpu.py --batch-size 4 --accumulation-steps 8
```

### 5.4 Arguments

| Argument | Default | Mô tả |
|----------|---------|-------|
| `--model` | `vit` | `vit` (ViT-B/32) hoặc `dinov3` (DINOv2) |
| `--hash-bit` | `64` | Số bits: 16, 32, 64, 128 |
| `--epochs` | `30` | Số epochs training |
| `--batch-size` | `8` | Batch size |
| `--lr` | `1e-4` | Learning rate |
| `--quick` | - | Quick test mode (3 epochs) |

---

## 6. KẾT QUẢ

### 6.1 Output sau training

```
./checkpoints/
├── best_model_nwpu_vit.pth      # Model checkpoint
└── training_history_vit.json    # Training logs
```

### 6.2 Kết quả mẫu

```
[Training Complete]
  Model: ViT-B/32 (64-bit hash)
  Dataset: NWPU-RESISC45
  
  Best Validation mAP: 0.6823
  Test mAP: 0.6756
  
  Checkpoint: ./checkpoints/best_model_nwpu_vit.pth
```

### 6.3 So sánh (nếu chạy cả 2 models)

| Backbone | mAP | P@10 | Training time |
|----------|-----|------|---------------|
| ViT-B/32 | 0.68 | 0.75 | ~2h |
| DINOv2-S/14 | 0.71 | 0.78 | ~2.5h |

**Nhận xét:** DINOv2 pretrained trên nhiều data hơn nên có thể extract features tốt hơn cho domain-specific images như viễn thám.

---

## 📚 TÀI LIỆU THAM KHẢO

1. Dosovitskiy et al. (2020). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale."
2. Oquab et al. (2023). "DINOv2: Learning Robust Visual Features without Supervision."
3. Yuan et al. (2020). "Central Similarity Quantization for Efficient Image and Video Retrieval."

---

## 👨‍🎓 THÔNG TIN

- **Môn học:** Truy vấn Thông tin Hình ảnh
- **Nội dung:** Xây dựng hệ thống CBIR cho ảnh viễn thám
- **Phương pháp:** ViT + Deep Hashing

---

*Cập nhật: Tháng 3/2026*
