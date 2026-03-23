# ĐỒ ÁN MÔN HỌC: TRUY VẤN THÔNG TIN HÌNH ẢNH

## Tối ưu hóa Vision Transformer Hashing cho Truy vấn Hình ảnh Dựa trên Nội dung trong các Miền Chuyên biệt

---

## 📋 MỤC LỤC

1. [Tóm tắt (Abstract)](#1-tóm-tắt-abstract)
2. [Giới thiệu đề tài](#2-giới-thiệu-đề-tài)
3. [Input/Output bài toán](#3-inputoutput-bài-toán)
4. [Mục tiêu nghiên cứu](#4-mục-tiêu-nghiên-cứu)
5. [Lý do chọn đề tài](#5-lý-do-chọn-đề-tài)
6. [Các nghiên cứu liên quan (Related Works)](#6-các-nghiên-cứu-liên-quan-related-works)
7. [Phương pháp đề xuất (Methodology)](#7-phương-pháp-đề-xuất-methodology)
8. [Dataset sử dụng](#8-dataset-sử-dụng)
9. [Thực nghiệm (Experiments)](#9-thực-nghiệm-experiments)
10. [Kế hoạch thực hiện](#10-kế-hoạch-thực-hiện)
11. [Cài đặt và Chạy code](#-cài-đặt-và-chạy-code)
12. [Giải thích các thành phần](#-giải-thích-các-thành-phần)

---

## ⚡ QUICK START

```bash
# Cài đặt dependencies
pip install -r requirements.txt

# Chạy training nhanh (3 epochs để test)
python train_nwpu.py --quick

# Chạy training đầy đủ với DINOv3 + Token Pruning
python train_nwpu.py --model dinov3 --enable-pruning --keep-ratio 0.7 --epochs 30

# Nếu GPU yếu (4GB VRAM)
python train_nwpu.py --batch-size 4 --accumulation-steps 8
```

---

## 1. TÓM TẮT (ABSTRACT)

**Bài toán:** Truy vấn hình ảnh dựa trên nội dung (Content-Based Image Retrieval - CBIR) trong các miền chuyên biệt như viễn thám và y tế.

**Phương pháp:** Kết hợp Vision Transformer (ViT) với Deep Hashing, sử dụng DINOv2/v3 làm backbone và áp dụng Token Pruning để tối ưu hiệu năng.

**Kết quả mong đợi:** 
- mAP ≥ 0.75 trên CIFAR-10 (64-bit)
- Giảm ≥ 30% latency với Token Pruning
- Đánh giá khả năng generalization trên NWPU-RESISC45

**Từ khóa:** Vision Transformer, Deep Hashing, Image Retrieval, Token Pruning, Remote Sensing

---

## 1. GIỚI THIỆU ĐỀ TÀI

### 1.1 Bối cảnh nghiên cứu

Sự bùng nổ của dữ liệu hình ảnh trong thời đại số hóa đã đặt ra thách thức lớn cho việc tìm kiếm và truy vấn thông tin. **Truy vấn hình ảnh dựa trên nội dung (Content-Based Image Retrieval - CBIR)** là một lĩnh vực quan trọng trong Khoa học Dữ liệu Hình ảnh, cho phép tìm kiếm hình ảnh tương tự dựa trên đặc trưng thị giác thay vì từ khóa văn bản.

### 1.2 Tên đề tài

> **"Tối ưu hóa Vision Transformer Hashing cho Truy vấn Hình ảnh Dựa trên Nội dung: Ứng dụng trong Viễn thám và Y tế"**

### 1.3 Vấn đề nghiên cứu

Mặc dù **Vision Transformer (ViT)** đã chứng minh hiệu quả vượt trội so với CNN trong nhiều tác vụ thị giác máy tính, nhưng vẫn tồn tại các thách thức:

| Thách thức | Mô tả |
|------------|-------|
| **Độ phức tạp tính toán** | Self-attention có độ phức tạp $O(N^2)$ với $N$ tokens |
| **Tốc độ truy vấn** | Đặc trưng float-point yêu cầu tính toán cosine similarity tốn kém |
| **Triển khai edge devices** | Mô hình lớn khó triển khai trên thiết bị có tài nguyên hạn chế |
| **Đặc trưng miền chuyên biệt** | Ảnh y tế và viễn thám có đặc điểm khác ảnh tự nhiên |

### 1.4 Giải pháp đề xuất

Kết hợp **Vision Transformer** với **Deep Hashing** để:
- Tận dụng khả năng mô hình hóa phụ thuộc tầm xa của ViT
- Chuyển đổi đặc trưng thành mã nhị phân nhỏ gọn (16-128 bits)
- Sử dụng **Hamming distance** cho truy vấn cực nhanh
- Áp dụng **Token Pruning** để giảm độ trễ suy luận

---

## 3. INPUT/OUTPUT BÀI TOÁN

### 3.1 Định nghĩa bài toán

**Bài toán Image Hashing Retrieval** được định nghĩa như sau:

> Cho một tập cơ sở dữ liệu hình ảnh $\mathcal{D} = \{I_1, I_2, ..., I_N\}$ và một hình ảnh truy vấn $I_q$, mục tiêu là tìm $K$ hình ảnh trong $\mathcal{D}$ tương tự nhất với $I_q$ dựa trên nội dung thị giác.

### 3.2 INPUT (Đầu vào)

```
┌─────────────────────────────────────────────────────────────┐
│                         INPUT                                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. QUERY IMAGE (Ảnh truy vấn)                              │
│     ┌─────────────┐                                         │
│     │             │  • Kích thước: 224 × 224 × 3 (RGB)      │
│     │   [IMAGE]   │  • Định dạng: JPEG, PNG, TIFF           │
│     │             │  • Giá trị pixel: [0, 255] hoặc [0, 1]  │
│     └─────────────┘                                         │
│                                                              │
│  2. DATABASE (Cơ sở dữ liệu ảnh)                            │
│     ┌───┐ ┌───┐ ┌───┐ ┌───┐                                │
│     │I_1│ │I_2│ │...│ │I_N│   • N = 10,000 - 1,000,000 ảnh │
│     └───┘ └───┘ └───┘ └───┘   • Mỗi ảnh có label/class     │
│                                                              │
│  3. PARAMETERS (Tham số)                                    │
│     • hash_bit: Số bit của mã hash (16, 32, 64, 128)       │
│     • top_k: Số kết quả trả về (10, 50, 100, ALL)          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Chi tiết Input:**

| Thành phần | Kiểu dữ liệu | Kích thước | Mô tả |
|------------|--------------|------------|-------|
| Query Image | Tensor | `[1, 3, 224, 224]` | Ảnh RGB đã resize và normalize |
| Database Images | Tensor | `[N, 3, 224, 224]` | N ảnh trong database |
| Database Labels | Tensor | `[N]` | Label của từng ảnh (cho evaluation) |
| Hash Bit | Integer | 16, 32, 64, 128 | Độ dài mã hash |

### 3.3 OUTPUT (Đầu ra)

```
┌─────────────────────────────────────────────────────────────┐
│                         OUTPUT                               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. HASH CODE (Mã hash nhị phân)                            │
│                                                              │
│     Query hash:    h_q = [1, -1, 1, 1, -1, ..., 1]  ∈ {-1,1}^K │
│                         └────────────────────────┘            │
│                              K bits (e.g., 64)               │
│                                                              │
│  2. RETRIEVED IMAGES (Danh sách ảnh tương tự)               │
│                                                              │
│     ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐                │
│     │ I_5 │ │I_23 │ │I_89 │ │I_12 │ │ ... │  Top-K results │
│     └─────┘ └─────┘ └─────┘ └─────┘ └─────┘                │
│     d_H=2   d_H=3   d_H=5   d_H=6   ...                     │
│                                                              │
│  3. RANKING (Thứ hạng theo Hamming distance)                │
│                                                              │
│     Rank 1: I_5   (Hamming distance = 2)  ← Most similar    │
│     Rank 2: I_23  (Hamming distance = 3)                    │
│     Rank 3: I_89  (Hamming distance = 5)                    │
│     ...                                                      │
│     Rank K: I_xx  (Hamming distance = d_K)                  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Chi tiết Output:**

| Thành phần | Kiểu dữ liệu | Kích thước | Mô tả |
|------------|--------------|------------|-------|
| Query Hash | Binary Tensor | `[K]` | Mã hash K-bit của query |
| Database Hashes | Binary Tensor | `[N, K]` | Mã hash của tất cả ảnh trong DB |
| Hamming Distances | Integer Tensor | `[N]` | Khoảng cách đến từng ảnh |
| Retrieved Indices | Integer List | `[top_k]` | Index của K ảnh gần nhất |
| Retrieved Images | Tensor | `[top_k, 3, H, W]` | K ảnh kết quả |

### 3.4 Ví dụ cụ thể

```
INPUT:
  Query: Ảnh sân bay (airport_001.jpg)
  Database: 31,500 ảnh viễn thám NWPU-RESISC45
  hash_bit: 64
  top_k: 10

PROCESSING:
  1. Query → ViT Backbone → Feature [1, 768]
  2. Feature → Hashing Head → Hash [-0.8, 0.9, -0.3, ...]
  3. Hash → Sign() → Binary [−1, 1, −1, 1, 1, −1, ...] (64 bits)
  4. Hamming Distance với tất cả 31,500 hashes trong DB
  5. Sort theo distance tăng dần

OUTPUT:
  Rank 1: airport_045.jpg  (Hamming dist = 4)  ✓ Relevant
  Rank 2: airport_123.jpg  (Hamming dist = 5)  ✓ Relevant  
  Rank 3: runway_089.jpg   (Hamming dist = 8)  ✗ Not relevant
  Rank 4: airport_234.jpg  (Hamming dist = 9)  ✓ Relevant
  ...
  → Precision@10 = 8/10 = 0.80
```

### 3.5 Workflow hoàn chỉnh

```
┌──────────────────────────────────────────────────────────────────────┐
│                        TRAINING PHASE                                 │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Training Images    ViT Backbone      Hashing Head     Loss Function │
│  ┌───┐              ┌─────────┐       ┌──────────┐     ┌───────────┐ │
│  │I_i│ ──────────→  │ DINOv2  │ ───→  │ MLP+Tanh │ ──→ │ CSQ Loss  │ │
│  │y_i│              │ ViT-B/14│       │ 768→64   │     │ + Quant   │ │
│  └───┘              └─────────┘       └──────────┘     └───────────┘ │
│                                              ↓                        │
│                                        h_i ∈ [-1,1]^64               │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────────────┐
│                       INDEXING PHASE                                  │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Database Images         Model (frozen)           Hash Database      │
│  ┌───┐ ┌───┐ ┌───┐      ┌─────────────┐         ┌────────────────┐  │
│  │I_1│ │I_2│ │...│ ───→ │ViT + Hash   │ ──────→ │ H = {h_1, h_2, │  │
│  └───┘ └───┘ └───┘      │   Head      │         │  ..., h_N}     │  │
│                         └─────────────┘         │ Binary codes    │  │
│                                                 └────────────────┘  │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────────────┐
│                       RETRIEVAL PHASE                                 │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Query Image    Model      Query Hash    Hamming Search   Results    │
│  ┌─────┐      ┌──────┐     ┌──────┐      ┌──────────┐    ┌───────┐  │
│  │ I_q │ ───→ │ ViT+ │ ──→ │ h_q  │ ───→ │ XOR +    │ ─→ │Top-K  │  │
│  └─────┘      │ Hash │     │64-bit│      │ popcount │    │Images │  │
│               └──────┘     └──────┘      └──────────┘    └───────┘  │
│                                                                       │
│  Latency: ~5-15ms          Latency: <1ms (với lookup table)         │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

### 3.6 Tại sao dùng Hashing?

| Phương pháp | Kích thước đặc trưng | Phép tính | Độ phức tạp | Tốc độ |
|-------------|---------------------|-----------|-------------|--------|
| **Float Features** | 768 × 4 bytes = 3KB | Cosine similarity | O(D) | Chậm |
| **Binary Hash** | 64 bits = 8 bytes | XOR + popcount | O(1) | **Cực nhanh** |

**Speedup:** 100-1000× nhanh hơn với large-scale database (millions images)

---

## 4. MỤC TIÊU NGHIÊN CỨU

### 4.1 Mục tiêu tổng quát

Xây dựng và đánh giá hệ thống truy vấn hình ảnh sử dụng Vision Transformer Hashing, tối ưu hóa cho các miền ứng dụng chuyên biệt (viễn thám, y tế) với sự cân bằng giữa **độ chính xác** và **hiệu năng**.

### 4.2 Mục tiêu cụ thể

| STT | Mục tiêu | Đo lường |
|-----|----------|----------|
| 1 | Triển khai mô hình ViT-Hashing với DINOv2/v3 backbone | Hoàn thành code |
| 2 | Đạt mAP ≥ 0.75 trên CIFAR-10 (64-bit hash) | mAP@All |
| 3 | Giảm ≥ 30% latency bằng Token Pruning | ms/query |
| 4 | Đánh giá trên bộ dữ liệu viễn thám NWPU-RESISC45 | mAP, Precision@K |
| 5 | Phân tích interpretability với ViT-CX | Saliency maps |

### 4.3 Câu hỏi nghiên cứu

1. **RQ1:** Vision Transformer Hashing có vượt trội CNN Hashing trong truy vấn hình ảnh chuyên biệt không?
2. **RQ2:** Token Pruning ảnh hưởng như thế nào đến trade-off accuracy-latency?
3. **RQ3:** Gram Anchoring có cải thiện chất lượng đặc trưng cục bộ cho retrieval không?

---

## 5. LÝ DO CHỌN ĐỀ TÀI

### 5.1 Xu hướng nghiên cứu

```
2020    2021    2022    2023    2024    2025    2026
  │       │       │       │       │       │       │
  ViT     DeiT    Swin    DINOv2  EET     DINOv3  VTS
  │       │       │       │       │       │       │
  └───────┴───────┴───────┴───────┴───────┴───────┘
          Transformer thống trị Computer Vision
```

- **Vision Transformer** đang thay thế CNN trong hầu hết các tác vụ SOTA
- **Foundation Models** (DINOv3, 7B params) cho phép transfer learning hiệu quả
- **Deep Hashing** là giải pháp thực tiễn cho large-scale retrieval

### 5.2 Tính cấp thiết

| Lĩnh vực | Nhu cầu | Ứng dụng |
|----------|---------|----------|
| **Viễn thám** | Tìm kiếm nhanh trong terabytes ảnh vệ tinh | Giám sát môi trường, quy hoạch đô thị |
| **Y tế** | Truy vấn ca bệnh tương tự để hỗ trợ chẩn đoán | X-quang, CT scan, MRI |
| **E-commerce** | Tìm kiếm sản phẩm bằng hình ảnh | Visual search |
| **Bảo mật** | Nhận dạng khuôn mặt, phát hiện bản quyền | Face retrieval |

### 5.3 Tính khả thi

- **Thời gian:** 3-4 tuần (phù hợp với mức độ "Tối ưu hóa & Đánh giá")
- **Tài nguyên:** Sử dụng pretrained DINOv2/v3, không cần train from scratch
- **Dữ liệu:** CIFAR-10 (có sẵn), NWPU-RESISC45 (public)
- **Code:** Tận dụng `timm`, `transformers`, `faiss`

### 5.4 Đóng góp khoa học

1. **Thực nghiệm so sánh** ViT-Hashing vs CNN-Hashing trên dữ liệu viễn thám
2. **Đánh giá tác động** của Token Pruning lên retrieval accuracy
3. **Phân tích interpretability** để đảm bảo model focus đúng vùng quan trọng

---

## 6. CÁC NGHIÊN CỨU LIÊN QUAN (RELATED WORKS)

### 6.1 Vision Transformer (ViT)

**Paper gốc:** Dosovitskiy et al., "An Image is Worth 16x16 Words" (ICLR 2021)

**Ý tưởng chính:**
- Chia ảnh thành patches 16×16, coi mỗi patch như một "word"
- Áp dụng Transformer encoder (giống NLP) cho sequence of patches
- Token `[CLS]` tổng hợp thông tin toàn cục cho classification

**Công thức Self-Attention:**

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Ưu điểm so với CNN:**
| Tiêu chí | CNN | ViT |
|----------|-----|-----|
| Receptive field | Local (kernel 3×3) | Global (tất cả patches) |
| Inductive bias | Translation equivariance | Ít bias, học từ data |
| Long-range dependency | Cần nhiều layers | Trực tiếp qua attention |
| Scalability | Saturate với data lớn | Scale tốt hơn |

### 6.2 Foundation Models: DINOv2/v3

**DINOv2 (Meta AI, 2023):**
- Self-supervised learning trên 142M images
- Pretrained ViT với distillation loss
- Strong zero-shot performance

**DINOv3 (Meta AI, 08/2025):**
- Scale lên 7B parameters, 1.7B training images
- **Gram Anchoring:** Giải quyết vấn đề suy giảm đặc trưng cục bộ

**Gram Anchoring hoạt động như thế nào?**

$$G = FF^T \quad \text{(Ma trận Gram của đặc trưng patches)}$$

Khi training lâu, các patch features bị collapse về CLS token. Gram Anchoring regularize bằng cách:
- Lưu Gram matrix từ early checkpoint (khi local features còn tốt)
- Căn chỉnh Gram hiện tại với Gram anchor
- Bảo toàn cấu trúc tương đồng giữa các patches

### 6.3 Deep Hashing Methods

| Method | Năm | Đặc điểm | Backbone |
|--------|-----|----------|----------|
| HashNet | 2017 | Continuation optimization | CNN |
| CSQ | 2020 | Central Similarity Quantization | CNN |
| TransHash | 2022 | Đầu tiên dùng Transformer | ViT |
| **VTS** | 2024 | ViT + CSQ, SOTA trên NUS-WIDE | ViT |
| **EET** | 2026 | Token pruning cho hashing | ViT |

**Central Similarity Quantization (CSQ) - Loss chính:**

$$\mathcal{L}_{CSQ} = \underbrace{\sum_i ||h_i - c_{y_i}||^2}_{\text{Center Loss}} + \lambda \underbrace{\sum_i (|h_i| - 1)^2}_{\text{Quantization Loss}}$$

- **Center Loss:** Kéo hash codes cùng class về cùng một center
- **Quantization Loss:** Đẩy giá trị về ±1 (binary)

### 6.4 Token Pruning

**Vấn đề:** Self-attention có độ phức tạp $O(N^2)$ với $N$ tokens

**Giải pháp:** Loại bỏ tokens ít quan trọng

| Method | Năm | Cách xác định token quan trọng | Kết quả |
|--------|-----|-------------------------------|--------|
| DynamicViT | 2021 | Learnable predictor | -30% FLOPs |
| EViT | 2022 | CLS attention score | Giữ tokens được CLS chú ý |
| **V-Pruner** | 2025 | Fisher Information | Tối ưu toàn cục |
| **ATPViT** | 2024 | Tích hợp trong attention | -47% FLOPs |

**Fisher Information Score:**

$$F_i = \mathbb{E}\left[\left(\frac{\partial \mathcal{L}}{\partial m_i}\right)^2\right]$$

Đo lường độ nhạy của loss khi mask token $i$. Token có Fisher score cao = quan trọng.

### 6.5 Remote Sensing Image Retrieval (RSIR)

| Paper | Năm | Method | Dataset | Best Result |
|-------|-----|--------|---------|-------------|
| DHCNN | 2018 | CNN + Hashing | UCM, AID | mAP 0.76 |
| DffViT | 2025 | Dual-branch ViT | NWPU | SOTA |
| **FIRViT** | 2024 | Lightweight ViT | NWPU | 99.8% acc, 658K params |
| DOFSH | 2024 | Orthogonal fusion | NWPU | Compact hash codes |

### 6.6 Interpretability: ViT-CX

**Vấn đề:** Attention weights không phản ánh đúng tầm quan trọng của vùng ảnh

**ViT-CX (Causal Explanation):**
1. Trích xuất patch embeddings
2. Cluster patches tương tự thành groups
3. Mask từng group, đo thay đổi output
4. Tạo saliency map dựa trên causal impact

**Ứng dụng trong đồ án:** Verify rằng Token Pruning không loại bỏ vùng quan trọng cho retrieval.

---

## 7. PHƯƠNG PHÁP ĐỀ XUẤT (METHODOLOGY)

### 7.1 Tổng quan kiến trúc

```
┌────────────────────────────────────────────────────────────────────┐
│                     ViT-HASHING ARCHITECTURE                        │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Input Image        Patch Embedding       Transformer Encoder      │
│  ┌─────────┐        ┌─────────────┐       ┌─────────────────┐      │
│  │224×224×3│ ──────>│ 14×14 patches│ ────>│ 12 layers       │      │
│  └─────────┘        │ + CLS token │       │ MSA + MLP       │      │
│                     │ + Pos embed │       │ (DINOv2 frozen) │      │
│                     └─────────────┘       └────────┬────────┘      │
│                                                    │               │
│                                                    ▼               │
│                     ┌─────────────────────────────────────────┐    │
│                     │           TOKEN PRUNING (Optional)      │    │
│                     │  Fisher Score → Keep top 70% tokens     │    │
│                     └─────────────────────────────────────────┘    │
│                                                    │               │
│                                                    ▼               │
│  CLS Token          Hashing Head           Binary Hash Code       │
│  ┌─────────┐        ┌─────────────┐        ┌─────────────┐        │
│  │ [CLS]   │ ──────>│Dropout(0.5) │ ──────>│ h ∈ {-1,1}^K│        │
│  │ 768-dim │        │Dense(1024)  │        │ K = 64 bits │        │
│  └─────────┘        │ReLU         │        └─────────────┘        │
│                     │Dense(K)     │                                │
│                     │Tanh → Sign  │                                │
│                     └─────────────┘                                │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

### 7.2 Các thành phần chi tiết

#### 7.2.1 Backbone: DINOv2 ViT-B/14

| Parameter | Value | Mô tả |
|-----------|-------|-------|
| Patch size | 14×14 | Mỗi patch cover 14×14 pixels |
| Num patches | 256 | (224/14)² = 256 patches |
| Embed dim | 768 | Dimension của mỗi token |
| Num heads | 12 | Multi-head attention |
| Num layers | 12 | Transformer blocks |
| Params | 86M | Pretrained, frozen trong fine-tuning |

#### 7.2.2 Hashing Head

```python
HashingHead(
    (dropout): Dropout(p=0.5)
    (fc1): Linear(768 → 1024)
    (relu): ReLU()
    (fc2): Linear(1024 → K)  # K = hash_bit
    (tanh): Tanh()           # Output ∈ [-1, 1]
)
```

**Training:** Output liên tục trong [-1, 1]
**Inference:** Áp dụng `sign()` để binary hóa

#### 7.2.3 Token Pruning Module

**Input:** Token features $X \in \mathbb{R}^{B \times N \times D}$

**Process:**
1. Tính Fisher score cho mỗi token
2. Sort theo score giảm dần
3. Giữ top-K tokens (K = 70% × N)
4. Luôn giữ CLS token

**Output:** Pruned features $X' \in \mathbb{R}^{B \times K \times D}$

### 7.3 Loss Function

**Total Loss:**

$$\mathcal{L} = \mathcal{L}_{center} + \lambda_q \mathcal{L}_{quant}$$

**Center Loss (Similarity-based):**

$$\mathcal{L}_{center} = \frac{1}{B} \sum_{i=1}^{B} ||h_i - c_{y_i}||^2$$

Trong đó:
- $h_i$: hash code của sample $i$
- $c_{y_i}$: learnable center của class $y_i$
- $B$: batch size

**Quantization Loss:**

$$\mathcal{L}_{quant} = \frac{1}{B} \sum_{i=1}^{B} \sum_{j=1}^{K} (|h_{ij}| - 1)^2$$

Đẩy các giá trị về ±1 để minimize quantization error.

### 7.4 Retrieval Process

**Hamming Distance:**

$$d_H(h_q, h_d) = \sum_{i=1}^{K} \mathbb{1}[h_q^i \neq h_d^i] = \frac{K - h_q \cdot h_d}{2}$$

**Efficient computation:**
```python
# XOR + popcount - cực nhanh trên CPU
hamming_dist = (query_hash ^ database_hash).sum(dim=-1)
```

### 7.5 Evaluation Metrics

**Mean Average Precision (mAP):**

$$mAP = \frac{1}{|Q|} \sum_{q \in Q} AP(q)$$

$$AP(q) = \frac{\sum_{k=1}^{n} P(k) \cdot rel(k)}{\text{số lượng relevant documents}}$$

**Precision@K:**

$$P@K = \frac{\text{số relevant trong top-K}}{K}$$

---

## 8. DATASET SỬ DỤNG

### 8.1 Tổng quan

| Dataset | Mục đích | Kích thước | Classes |
|---------|----------|------------|---------|
| **CIFAR-10** | Training & Development | 60,000 images | 10 |
| **NWPU-RESISC45** | Evaluation (Viễn thám) | 31,500 images | 45 |
| **ChestX-ray8** | Evaluation (Y tế) - Optional | 108,948 images | 8 |

### 8.2 CIFAR-10 (Training Dataset)

```
📁 CIFAR-10
├── 50,000 training images
├── 10,000 test images
├── 10 classes: airplane, automobile, bird, cat, deer,
│               dog, frog, horse, ship, truck
└── Resolution: 32×32 → resize to 224×224
```

**Lý do chọn:**
- Dataset chuẩn cho benchmarking hashing methods
- Kích thước vừa phải, phù hợp với thời gian nghiên cứu
- Dễ so sánh với các nghiên cứu trước

**Phân chia:**
- Database (Gallery): 50,000 images
- Query set: 10,000 images
- Training: 5,000 images (sampled)

### 8.3 NWPU-RESISC45 (Evaluation Dataset)

```
📁 NWPU-RESISC45
├── 31,500 images (45 classes × 700 images)
├── Resolution: 256×256 pixels
├── Classes: airplane, airport, baseball_diamond, basketball_court,
│            beach, bridge, chaparral, church, circular_farmland,
│            cloud, commercial_area, dense_residential, desert,
│            forest, freeway, golf_course, ground_track_field,
│            harbor, industrial_area, intersection, island, lake,
│            meadow, medium_residential, mobile_home_park, mountain,
│            overpass, palace, parking_lot, railway, railway_station,
│            rectangular_farmland, river, roundabout, runway,
│            sea_ice, ship, snowberg, sparse_residential, stadium,
│            storage_tank, tennis_court, terrace, thermal_power_station,
│            wetland
└── Source: Northwestern Polytechnical University
```

**Lý do chọn:**
- Dataset chuẩn cho **Remote Sensing Image Retrieval (RSIR)**
- Thách thức cao: đa dạng nội lớp, tương đồng liên lớp
- Kích thước phù hợp với thời gian nghiên cứu

**Phân chia:**
- Database: 28,350 images (90%)
- Query: 3,150 images (10%)

### 8.4 Đặc điểm dữ liệu viễn thám

| Đặc điểm | Thách thức | Giải pháp ViT |
|----------|-----------|---------------|
| **Đa dạng nội lớp** | Ánh sáng, góc nhìn, mùa khác nhau | Self-attention học invariance |
| **Tương đồng liên lớp** | Airport vs Runway rất giống | Global context disambiguation |
| **Đối tượng nhỏ** | Xe, tàu trong ảnh lớn | Fine-grained attention |
| **Phụ thuộc không gian** | Quan hệ giữa các đối tượng | Long-range dependency modeling |

---

## 9. THỰC NGHIỆM (EXPERIMENTS)

### 9.1 Thiết lập thực nghiệm

**Hardware:**
- CPU: Intel Core Ultra (với NPU cho inference)
- GPU: NVIDIA hoặc Intel Arc (nếu có)
- RAM: 16GB+

**Software:**
- Python 3.10+
- PyTorch 2.0+
- Intel Extension for PyTorch (IPEX)
- OpenVINO (cho NPU deployment)

**Hyperparameters:**

| Parameter | Value | Mô tả |
|-----------|-------|-------|
| Batch size | 32 | Training batch |
| Epochs | 20 | Số epochs training |
| LR (backbone) | 3×10⁻⁵ | Learning rate cho ViT backbone |
| LR (head) | 1×10⁻⁴ | Learning rate cho Hashing head |
| Hash bit | 64 | Độ dài mã hash |
| Pruning ratio | 0.7 | Giữ 70% tokens |
| λ_quant | 0.0001 | Quantization loss weight |

### 9.2 Baselines để so sánh

| Model | Backbone | Hash method | Ghi chú |
|-------|----------|-------------|--------|
| ResNet50-Hash | ResNet50 | CSQ | CNN baseline |
| ViT-Hash (vanilla) | ViT-B/16 | CSQ | ViT baseline |
| DINOv2-Hash | ViT-B/14 | CSQ | **Our method** |
| DINOv2-Hash + Pruning | ViT-B/14 | CSQ + Pruning | **Our method (optimized)** |

### 9.3 Metrics đánh giá

**Accuracy metrics:**
- mAP@All: Mean Average Precision trên toàn bộ database
- mAP@1000: mAP trên top-1000 kết quả
- Precision@K (K = 10, 50, 100)
- Recall@K

**Efficiency metrics:**
- Latency (ms/query): Thời gian xử lý 1 query
- Throughput (images/sec): Số ảnh xử lý/giây
- FLOPs: Floating point operations
- Memory (MB): GPU/CPU memory usage

### 9.4 Kết quả mong đợi

**CIFAR-10 (64-bit hash):**

| Model | mAP | Latency | FLOPs |
|-------|-----|---------|-------|
| ResNet50-Hash | ~0.70 | 5ms | 4G |
| ViT-Hash | ~0.73 | 15ms | 17G |
| **DINOv2-Hash** | **~0.78** | 12ms | 17G |
| **DINOv2-Hash + Pruning** | **~0.75** | **8ms** | **11G** |

**NWPU-RESISC45 (64-bit hash):**

| Model | mAP | P@10 | P@50 |
|-------|-----|------|------|
| ResNet50-Hash | ~0.65 | ~0.72 | ~0.68 |
| ViT-Hash | ~0.70 | ~0.78 | ~0.74 |
| **DINOv2-Hash** | **~0.75** | **~0.82** | **~0.78** |

### 9.5 Ablation Studies

**A. Hash bit length:**

| Bits | mAP | Storage/image |
|------|-----|---------------|
| 16 | 0.68 | 2 bytes |
| 32 | 0.73 | 4 bytes |
| **64** | **0.78** | **8 bytes** |
| 128 | 0.79 | 16 bytes |

**B. Pruning ratio:**

| Keep ratio | mAP | Latency reduction |
|------------|-----|-------------------|
| 1.0 (no pruning) | 0.78 | 0% |
| 0.8 | 0.77 | 15% |
| **0.7** | **0.76** | **30%** |
| 0.5 | 0.72 | 50% |

**C. Backbone comparison:**

| Backbone | Pretrain data | mAP | Latency |
|----------|---------------|-----|--------|
| ViT-B/16 (ImageNet) | 1.2M | 0.73 | 15ms |
| DINOv2 ViT-B/14 | 142M | 0.78 | 12ms |
| DINOv3 ViT-B/14 | 1.7B | 0.80* | 12ms |

*Ước tính, DINOv3 chưa public

### 9.6 Interpretability Analysis

**ViT-CX Visualization:**
- So sánh saliency maps trước và sau pruning
- Verify model vẫn focus vào vùng quan trọng
- Phát hiện potential failure cases
| **AdaptiVision** | 2024 | Soft K-means clustering | Bảo toàn ngữ nghĩa |
| **DynamicViT** | 2021 | Learnable token selection | Dynamic computation |

---

## 10. KẾ HOẠCH THỰC HIỆN

### 10.1 Timeline (4 tuần)

```
Tuần 1: Thiết lập nền tảng
├── Cài đặt môi trường (PyTorch, timm, IPEX)
├── Tải và xử lý datasets
├── Implement ViT-Hashing baseline
└── Đo baseline metrics

Tuần 2: Tối ưu hóa
├── Implement Token Pruning (Fisher-based)
├── Profile latency-token relationship
├── Tối ưu hyperparameters
└── So sánh các pruning ratios

Tuần 3: Interpretability & Evaluation
├── Implement ViT-CX analyzer
├── Đánh giá trên NWPU-RESISC45
├── Visualize attention maps
└── Ablation studies

Tuần 4: Tổng hợp & Báo cáo
├── Compile kết quả
├── So sánh với baselines
├── Viết báo cáo
└── Chuẩn bị presentation
```

### 10.2 Deliverables

| STT | Output | Format |
|-----|--------|--------|
| 1 | Source code | Python (.py) |
| 2 | Trained models | PyTorch (.pth) |
| 3 | Báo cáo kỹ thuật | PDF/Word |
| 4 | Slides thuyết trình | PowerPoint |
| 5 | Demo notebook | Jupyter (.ipynb) |

### 10.3 Cấu trúc thư mục dự án

```
Information Retrieval/
├── data/
│   ├── cifar-10-batches-py/     # Training data
│   └── NWPU-RESISC45/           # Evaluation data
├── src/
│   ├── model.py                  # ViT_Hashing model
│   ├── loss.py                   # CSQ Loss
│   ├── dataset.py                # Data loaders
│   ├── evaluate.py               # mAP, Precision@K
│   └── research/
│       ├── dinov3_hashing.py     # DINOv3 backbone
│       ├── pruning.py            # Token Pruning
│       ├── interpretability.py   # ViT-CX
│       └── profiler.py           # Latency profiler
├── checkpoints/                  # Saved models
├── results/                      # Experiment results
├── train_intel.py               # Intel-optimized training
├── main_research.py             # Research pipeline
└── README.md                    # This file
```

---

## 📚 TÀI LIỆU THAM KHẢO

### Papers chính

1. Dosovitskiy et al. (2020). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." ICLR.

2. Caron et al. (2025). "DINOv3: Learning Robust Visual Features with Gram Anchoring." Meta AI.

3. He et al. (2026). "EET: Efficient and Effective Vision Transformer for Image Retrieval." CVPR.

4. Yuan et al. (2024). "V-Pruner: Fast Global Token Pruning for Vision Transformers via Fisher Information." NeurIPS.

5. Cao et al. (2017). "HashNet: Deep Learning to Hash by Continuation." ICCV.

6. Cheng et al. (2017). "Remote Sensing Image Scene Classification: Benchmark and State of the Art." Proceedings of the IEEE.

### Code repositories

- timm: https://github.com/huggingface/pytorch-image-models
- DINOv2: https://github.com/facebookresearch/dinov2
- FAISS: https://github.com/facebookresearch/faiss

---

## � CÀI ĐẶT VÀ CHẠY CODE

### Yêu cầu hệ thống

| Thành phần | Yêu cầu tối thiểu | Khuyến nghị |
|------------|-------------------|-------------|
| **OS** | Windows 10/11, Linux | Windows 11 |
| **Python** | 3.9+ | 3.10+ |
| **GPU** | GTX 1650 (4GB VRAM) | RTX 3060+ (8GB+) |
| **RAM** | 8GB | 16GB+ |
| **Disk** | 10GB free | 20GB+ |

### Cài đặt môi trường

```bash
# 1. Clone repository
git clone <repo-url>
cd "Information Retrieval"

# 2. Tạo virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 3. Cài đặt dependencies
pip install -r requirements.txt

# 4. Tải dataset (NWPU-RESISC45 đã có sẵn trong data/archive/)
# Hoặc tải thêm:
python download_dataset.py
python download_nwpu.py
```

### Cấu trúc thư mục

```
Information Retrieval/
├── 📁 data/
│   ├── archive/Dataset/          # NWPU-RESISC45 dataset
│   │   ├── train/train/          # 45 class folders
│   │   └── test/test/            # 45 class folders
│   └── cifar-10-batches-py/      # CIFAR-10 dataset
├── 📁 src/
│   ├── model.py                  # ViT_Hashing (basic)
│   ├── loss.py                   # CSQLoss
│   ├── dataset.py                # Data loaders
│   ├── evaluate.py               # Evaluation metrics
│   └── 📁 research/              # Research components
│       ├── dinov3_hashing.py     # DINOv3Hashing model
│       ├── pruning.py            # Token Pruning modules
│       ├── interpretability.py   # ViT-CX analyzer
│       └── profiler.py           # Latency profiler
├── 📁 checkpoints/               # Saved models
├── train_nwpu.py                 # 🔥 Main training script
├── train_gpu.py                  # GPU training (CIFAR-10)
├── main_research.py              # Full research pipeline
├── evaluate_nwpu.py              # Evaluation script
└── README.md
```

---

## 🎯 HƯỚNG DẪN CHẠY CODE

### Quick Start - Chạy nhanh

```bash
# Test nhanh với 3 epochs
python train_nwpu.py --quick

# Train đầy đủ với DINOv3 + Token Pruning
python train_nwpu.py --model dinov3 --enable-pruning --epochs 30
```

### Training với các cấu hình khác nhau

#### 1. DINOv3Hashing (khuyến nghị - full research pipeline)

```bash
# DINOv3 cơ bản
python train_nwpu.py --model dinov3

# DINOv3 với Token Pruning (giảm 30% latency)
python train_nwpu.py --model dinov3 --enable-pruning --keep-ratio 0.7

# DINOv3 với pretrained weights (cần internet)
python train_nwpu.py --model dinov3 --pretrained

# DINOv3-Large (nặng hơn, chính xác hơn)
python train_nwpu.py --model dinov3 --dinov3-variant vit_large_patch14_dinov2.lvd142m

# DINOv3 với Gram Anchoring (DINOv3 simulation)
python train_nwpu.py --model dinov3 --gram-anchoring
```

#### 2. ViT Basic (nhẹ hơn, nhanh hơn)

```bash
# ViT-B/32 (patch size 32)
python train_nwpu.py --model vit --patch-size 32

# ViT-B/16 (patch size 16, chính xác hơn)
python train_nwpu.py --model vit --patch-size 16

# ViT với custom weights
python train_nwpu.py --model vit --weights ./ViT-B_32.npz
```

#### 3. Điều chỉnh Memory/VRAM

```bash
# Nếu GPU yếu (4GB VRAM) - giảm batch size
python train_nwpu.py --batch-size 4 --accumulation-steps 8

# GPU mạnh hơn (8GB+)
python train_nwpu.py --batch-size 16 --accumulation-steps 2

# CPU only (chậm)
python train_nwpu.py --batch-size 2 --num-workers 0
```

### Danh sách Arguments đầy đủ

| Argument | Default | Mô tả |
|----------|---------|-------|
| **Model Selection** |
| `--model` | `dinov3` | Model: `dinov3` (DINOv3Hashing) hoặc `vit` (ViT_Hashing) |
| `--dinov3-variant` | `vit_small_patch14_dinov2.lvd142m` | Backbone DINOv3: small/base/large |
| `--pretrained` | `False` | Load pretrained DINOv2 weights |
| `--freeze-backbone` | `False` | Đóng băng backbone |
| `--gram-anchoring` | `False` | Bật Gram Anchoring |
| **Hash Settings** |
| `--hash-bit` | `64` | Số bit: 16, 32, 64, 128 |
| `--patch-size` | `32` | Patch size cho ViT basic: 14, 16, 32 |
| **Token Pruning** |
| `--enable-pruning` | `False` | Bật Token Pruning |
| `--keep-ratio` | `0.7` | Tỷ lệ token giữ lại (0.5 = 50%) |
| `--pruning-method` | `fisher` | Method: `fisher` (V-Pruner) hoặc `attention` |
| **Loss** |
| `--lambda-q` | `0.0001` | Weight của Quantization Loss |
| **Training** |
| `--epochs` | `30` | Số epochs |
| `--batch-size` | `8` | Batch size |
| `--accumulation-steps` | `4` | Gradient accumulation |
| `--lr` | `1e-4` | Learning rate |
| `--weight-decay` | `0.01` | Weight decay |
| **Data** |
| `--data-dir` | `./data/archive/Dataset` | Đường dẫn dataset |
| `--save-dir` | `./checkpoints` | Thư mục lưu model |
| `--num-workers` | `2` | DataLoader workers |
| `--eval-every` | `5` | Evaluate mỗi N epochs |

---

## 🧩 GIẢI THÍCH CÁC THÀNH PHẦN

### 1. DINOv3Hashing (`src/research/dinov3_hashing.py`)

**Mục đích:** Model chính kết hợp DINOv2/v3 backbone với Deep Hashing.

```
Input Image (224x224x3)
         ↓
┌─────────────────────────────┐
│   DINOv2 Backbone (ViT)     │  ← Pretrained trên 142M images
│   - Patch Embedding         │
│   - Transformer Encoder     │
│   - Global average pooling  │
└─────────────────────────────┘
         ↓
    Features (768-dim)
         ↓
┌─────────────────────────────┐
│      Hashing Head           │
│   - Dropout(0.5)            │
│   - Linear(768 → 1024)      │
│   - ReLU                    │
│   - Linear(1024 → hash_bit) │
│   - Tanh → Sign (inference) │
└─────────────────────────────┘
         ↓
    Hash Code (64-bit binary)
```

**Các biến thể DINOv3:**

| Variant | Embed Dim | Params | Độ chính xác |
|---------|-----------|--------|--------------|
| `vit_small_patch14_dinov2` | 384 | 22M | Nhanh, nhẹ |
| `vit_base_patch14_dinov2` | 768 | 86M | Cân bằng ⭐ |
| `vit_large_patch14_dinov2` | 1024 | 307M | Chính xác nhất |

### 2. Token Pruning (`src/research/pruning.py`)

**Mục đích:** Giảm số lượng tokens để tăng tốc inference mà không giảm nhiều accuracy.

**Các phương pháp:**

| Class | Phương pháp | Cách hoạt động |
|-------|-------------|----------------|
| `TokenPruner` | V-Pruner (Fisher Information) | Tính điểm quan trọng = gradient² → giữ top-K tokens |
| `AttentionBasedPruner` | Attention-based | Học prediction network để quyết định token nào giữ |
| `TokenMerger` | AdaptiVision | Gộp tokens tương tự bằng soft k-means |

**Hiệu quả Token Pruning:**

| Keep Ratio | Tokens giữ lại | FLOPs giảm | mAP thay đổi |
|------------|----------------|------------|--------------|
| 100% | 197 | 0% | Baseline |
| 80% | 158 | ~36% | -1% |
| **70%** | **138** | **~51%** | **-2-3%** |
| 50% | 99 | ~75% | -5-8% |

### 3. CSQLoss (`src/loss.py`)

**Mục đích:** Loss function cho Deep Hashing, kết hợp similarity learning với quantization.

```python
Total Loss = Center Loss + λ × Quantization Loss
```

**Center Loss:**
- Kéo hash codes cùng class về cùng một center
- Center được khởi tạo random, học cùng model

**Quantization Loss:**
- Đẩy giá trị hash về ±1 (binary)
- Giảm lỗi khi chuyển từ continuous → discrete

**Tham số `lambda_q`:**

| λ_q | Ảnh hưởng |
|-----|-----------|
| 0.00001 | Ưu tiên similarity, hash chưa binary |
| **0.0001** | **Cân bằng (khuyến nghị)** |
| 0.001 | Ưu tiên binary, có thể giảm mAP |

### 4. GPU Optimizations

**Mixed Precision (AMP):**
- Tự động sử dụng FP16 cho forward pass
- Giảm ~50% VRAM usage
- Tăng tốc training trên GPU hiện đại

**Gradient Accumulation:**
- Chia batch lớn thành nhiều mini-batch
- `effective_batch = batch_size × accumulation_steps`
- Cho phép train với VRAM hạn chế

**Gradient Clipping:**
- `max_norm=1.0` để ổn định training
- Ngăn gradient explosion

---

## 📊 DATASET

### NWPU-RESISC45 (Remote Sensing)

**Thông tin:**
- **Nguồn:** Northwestern Polytechnical University
- **Kích thước:** 31,500 ảnh (45 class × 700 ảnh)
- **Resolution:** 256×256 pixels
- **Split mặc định:** Train 80% / Val 20%

**45 Classes:**

| # | Tên | # | Tên | # | Tên |
|---|-----|---|-----|---|-----|
| 1 | airplane | 16 | golf_course | 31 | railway_station |
| 2 | airport | 17 | ground_track_field | 32 | rectangular_farmland |
| 3 | baseball_diamond | 18 | harbor | 33 | river |
| 4 | basketball_court | 19 | industrial_area | 34 | roundabout |
| 5 | beach | 20 | intersection | 35 | runway |
| 6 | bridge | 21 | island | 36 | sea_ice |
| 7 | chaparral | 22 | lake | 37 | ship |
| 8 | church | 23 | meadow | 38 | snowberg |
| 9 | circular_farmland | 24 | medium_residential | 39 | sparse_residential |
| 10 | cloud | 25 | mobile_home_park | 40 | stadium |
| 11 | commercial_area | 26 | mountain | 41 | storage_tank |
| 12 | dense_residential | 27 | overpass | 42 | tennis_court |
| 13 | desert | 28 | palace | 43 | terrace |
| 14 | forest | 29 | parking_lot | 44 | thermal_power_station |
| 15 | freeway | 30 | railway | 45 | wetland |

### CIFAR-10 (General Domain)

**Thông tin:**
- **Kích thước:** 60,000 ảnh (10 class × 6,000)
- **Resolution:** 32×32 → resize to 224×224
- **10 Classes:** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

---

## 📈 KẾT QUẢ MONG ĐỢI

### Metrics

| Metric | Công thức | Ý nghĩa |
|--------|-----------|---------|
| **mAP** | Mean Average Precision | Độ chính xác trung bình |
| **P@K** | Precision at K | % relevant trong top-K |
| **Latency** | ms/query | Thời gian xử lý 1 ảnh |

### Kết quả baseline trên NWPU-RESISC45 (64-bit)

| Model | mAP | P@10 | Latency |
|-------|-----|------|---------|
| ViT_Hashing (basic) | ~0.70 | ~0.75 | 15ms |
| DINOv3Hashing | ~0.75 | ~0.82 | 12ms |
| DINOv3 + Pruning (70%) | ~0.73 | ~0.80 | **8ms** |

---

## 👨‍🎓 THÔNG TIN

- **Môn học:** Truy vấn Thông tin Hình ảnh / Khoa học Dữ liệu Hình ảnh
- **Bậc học:** Thạc sĩ
- **Mức độ:** Tối ưu hóa hệ thống và Đánh giá thực nghiệm
- **Thời gian:** 3-4 tuần

---

*Tài liệu được cập nhật: Tháng 3/2026*
