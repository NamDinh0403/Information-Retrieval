# DO AN THAC SI: TRUY VAN THONG TIN HINH ANH

## Vision Transformer + Deep Hashing cho Content-Based Image Retrieval

---

## MUC LUC

1. [Tong quan](#1-tong-quan)
2. [Cau truc project](#2-cau-truc-project)
3. [Cai dat](#3-cai-dat)
4. [Dataset](#4-dataset)
5. [Phuong phap](#5-phuong-phap)
6. [Huong dan chay](#6-huong-dan-chay)
7. [Ket qua](#7-ket-qua)
8. [Tai lieu tham khao](#8-tai-lieu-tham-khao)

---

## QUICK START

```bash
pip install -r requirements.txt

# Train tren NWPU-RESISC45 (vien tham, demo/app)
python experiments/train.py --model vit --epochs 30

# Train tren NUS-WIDE (benchmark chuan so sanh voi literature)
python experiments/train_nuswide.py --model vit --epochs 30

# Danh gia benchmark day du tren NUS-WIDE
python experiments/evaluate_nuswide.py \
    --checkpoint ./checkpoints/best_model_nuswide_vit_64bit.pth

# Chay demo app
streamlit run app.py
```

---

## 1. TONG QUAN

He thong CBIR (Content-Based Image Retrieval) ket hop **Vision Transformer** voi **Deep Hashing**:

- Backbone trich xuat feature ngu nghia tu anh
- Hashing head anh xa feature -> binary hash code (64 bits)
- Tim kiem gan dung bang Hamming distance trong khong gian nhi phan

### Hai bo du lieu muc tieu

| Dataset | Muc dich | Giao thuc |
|---------|----------|-----------|
| **NWPU-RESISC45** | Demo app, anh vien tham 45 class | Tu dinh nghia (khong phai benchmark chuan) |
| **NUS-WIDE** | Benchmark so sanh voi SOTA | Chuan CSQ/HashNet: query=2100, DB=193734 |

> **Luu y:** mAP tren NWPU khong so sanh duoc voi SOTA vi khac domain va giao thuc.
> Ket qua benchmark hop le phai chay tren NUS-WIDE.

### Kien truc tong quan

```
Input Image --> Backbone (ViT-B/32 / DINOv2-S) --> Hashing Head (FC->ReLU->FC->Tanh) --> Hash Code (64 bits)
                                                            |
                                               MultiLabel CSQ Loss
                                            (center + pair + quant)
```

---

## 2. CAU TRUC PROJECT

```
Information Retrieval/
|-- app.py                          # Streamlit demo app (dung NWPU checkpoint)
|-- requirements.txt
|-- ViT-B_32.npz                    # ViT-B/32 pretrained weights (JAX format)
|
|-- experiments/
|   |-- train.py                    # Train tren NWPU-RESISC45
|   |-- train_nuswide.py            # Train tren NUS-WIDE (benchmark chuan)
|   |-- evaluate.py                 # Evaluate tren NWPU
|   |-- evaluate_nuswide.py         # Evaluate day du tren NUS-WIDE
|   |-- ablation.py                 # Ablation study
|   `-- visualize.py
|
|-- src/
|   |-- models/
|   |   |-- vit_hashing.py          # ViT-B/32 + Hashing head
|   |   `-- dinov2_hashing.py       # DINOv2-S/14 + Hashing head
|   |-- losses/
|   |   |-- csq_loss.py             # CSQ (single-label, cho NWPU)
|   |   `-- csq_multilabel_loss.py  # MultiLabel CSQ (cho NUS-WIDE)
|   |-- data/
|   |   |-- loaders.py              # NWPU loader
|   |   |-- nuswide_loader.py       # NUS-WIDE loader (pre-split format)
|   |   `-- archive/NUS-WIDE/       # Dataset (269K anh + split files)
|   `-- utils/
|       |-- metrics.py              # Single-label metrics
|       |-- metrics_multilabel.py   # Multi-label metrics (mAP, P@K)
|       `-- pruning.py              # Token pruning (tuy chon)
|
|-- checkpoints/
|   |-- best_model_nwpu_vit.pth          # NWPU checkpoint (epoch 20, mAP=0.9112)
|   `-- best_model_nuswide_vit_64bit.pth # NUS-WIDE checkpoint (sau khi train)
|
|-- scripts/
|   |-- build_vector_db.py
|   |-- extract_features.py
|   |-- query_image.py
|   `-- download_nwpu.py / download_nuswide.py
|
`-- docs/
    `-- report.tex                  # Bao cao LaTeX
```

---

## 3. CAI DAT

```bash
pip install -r requirements.txt
```

Yeu cau chinh: `torch>=2.0`, `timm`, `streamlit`, `tqdm`, `numpy`, `Pillow`.

GPU: da test tren GTX 1650 (4GB VRAM). Neu thieu VRAM giam `--batch-size`.

---

## 4. DATASET

### 4.1 NWPU-RESISC45

| Thuoc tinh | Gia tri |
|-----------|---------|
| So anh | 31,500 |
| Classes | 45 (vien tham) |
| Anh/class | 700 |
| Train/Val/Test | 80/10/10% |
| Vi tri | `data/archive/Dataset/` |

```bash
# Tai ve (neu chua co)
python scripts/download_nwpu.py
```

### 4.2 NUS-WIDE (benchmark chuan)

| Thuoc tinh | Gia tri |
|-----------|---------|
| Tong so anh | 269,648 anh Flickr |
| Nhan | Multi-label, 21 nhan pho bien |
| Query set | 2,100 anh (test_img.txt) |
| Database | 193,734 anh (database_img.txt) |
| Train set | ~10,500 anh (500/class x 21, sample tu DB) |
| Vi tri | `src/data/archive/NUS-WIDE/` |

Cau truc thu muc NUS-WIDE (da co san, khong can tai them):

```
src/data/archive/NUS-WIDE/
|-- images/                      # 269,648 anh .jpg (flat, khong phan class)
|-- database_img.txt             # 193,734 duong dan
|-- database_label_onehot.txt    # 193,734 x 21 nhan (space-separated 0/1)
|-- test_img.txt                 # 2,100 duong dan  (query set)
`-- test_label_onehot.txt        # 2,100 x 21 nhan
```

> NUS-WIDE la **multi-label** -- 1 anh co the vua co nhan `water`, `sky`, `beach`.
> Khong to chuc theo thu muc class nhu NWPU.

---

## 5. PHUONG PHAP

### 5.1 Models

#### ViT-B/32  (`src/models/vit_hashing.py`)
- Pretrained: ImageNet-21k (supervised)
- Patch size: 32x32 -> 49 patches / anh 224x224
- Embedding: 768d -> Hashing head -> 64 bits
- Tham so: ~88.4M

#### DINOv2-S/14  (`src/models/dinov2_hashing.py`)
- Pretrained: LVD-142M (self-supervised)
- Patch size: 14x14 -> 256 patches / anh 224x224
- Embedding: 384d -> LayerNorm -> Hashing head -> 64 bits
- Tham so: ~22M
- Luu y: AMP bi tat (feature overflow float16)

### 5.2 Loss: MultiLabel CSQ  (`src/losses/csq_multilabel_loss.py`)

Danh cho NUS-WIDE. Ba thanh phan:

```
L = lambda_c * L_center + L_pair + lambda_q * L_quant
```

**Center loss** -- moi anh co target code tong hop tu cac centers cua nhan active:

```
t_i = sign(l_i @ H),   H in {-1,+1}^{C x K}  (fixed random, khong hoc)
L_center = (1/B) * sum_i ||h_i - t_i||^2
```

**Pairwise loss** -- hai anh chung nhan -> code gan nhau:

```
L_pair = sum_{pos} (1 - sim)^2 / N+  +  sum_{neg} relu(sim + margin)^2 / N-
```

**Quantization loss** -- ep gia tri ve +-1:

```
L_quant = E[(|h| - 1)^2]
```

Mac dinh: `lambda_c=1.0`, `lambda_q=0.01`, `margin=0.5`.

### 5.3 Loss don nhan  (`src/losses/csq_loss.py`)

Danh cho NWPU. Moi class co 1 center co dinh, loss keo code ve center cua class tuong ung.

### 5.4 Optimizer

```python
AdamW([
    {'params': backbone_params, 'lr': lr_backbone},  # 1e-5
    {'params': head_params,     'lr': lr_head},       # 1e-4
], weight_decay=1e-4)
# Scheduler: CosineAnnealingLR
# Khi fine-tune tu NWPU: lr_backbone tu dong * 0.1
```

---

## 6. HUONG DAN CHAY

### 6.1 Train tren NWPU-RESISC45

```bash
# Train co ban
python experiments/train.py --model vit --hash-bit 64 --epochs 30

# Voi DINOv2
python experiments/train.py --model dinov3 --hash-bit 64 --epochs 30

# Quick test (3 epoch, subset nho)
python experiments/train.py --quick
```

### 6.2 Train tren NUS-WIDE (benchmark chuan)

```bash
# Train co ban tu ImageNet pretrained
python experiments/train_nuswide.py \
    --model vit \
    --epochs 30 \
    --batch-size 32 \
    --hash-bit 64 \
    --loss csq \
    --lambda-c 1.0 \
    --lambda-q 0.01

# Quick test (3 epoch, 50 anh/class)
python experiments/train_nuswide.py --quick

# Fine-tune tu NWPU checkpoint (backbone transfer)
python experiments/train_nuswide.py \
    --model vit \
    --epochs 30 \
    --finetune-from ./checkpoints/best_model_nwpu_vit.pth

# Fine-tune ca backbone + hashing head
python experiments/train_nuswide.py \
    --model vit \
    --epochs 30 \
    --finetune-from ./checkpoints/best_model_nwpu_vit.pth \
    --transfer-head

# Dung pairwise loss thuan (DCH hoac HashNet)
python experiments/train_nuswide.py --loss dch
python experiments/train_nuswide.py --loss hashnet
```

**Tham so day du cua `train_nuswide.py`:**

| Tham so | Default | Mo ta |
|---------|---------|-------|
| `--data-dir` | `./src/data/archive/NUS-WIDE` | Thu muc chua split files |
| `--model` | `vit` | `vit` hoac `dinov2` |
| `--hash-bit` | `64` | So bits hash (16/32/64/128) |
| `--epochs` | `30` | So epoch |
| `--batch-size` | `32` | Batch size |
| `--lr-backbone` | `1e-5` | Learning rate backbone |
| `--lr-head` | `1e-4` | Learning rate hashing head |
| `--grad-accumulation` | `4` | Buoc tich luy gradient |
| `--loss` | `csq` | `csq`, `dch`, `hashnet` |
| `--lambda-c` | `1.0` | He so center loss (CSQ only) |
| `--lambda-q` | `0.01` | He so quantization loss |
| `--finetune-from` | `None` | Path checkpoint NWPU de transfer backbone |
| `--transfer-head` | `False` | Kem transfer hashing head |
| `--checkpoint-dir` | `./checkpoints` | Thu muc luu model |
| `--quick` | `False` | Chay nhanh 3 epoch, 50 anh/class |

### 6.3 Danh gia benchmark NUS-WIDE

```bash
python experiments/evaluate_nuswide.py \
    --checkpoint ./checkpoints/best_model_nuswide_vit_64bit.pth \
    --data-dir ./src/data/archive/NUS-WIDE

# Output: mAP@ALL, mAP@1000, P@100
# So sanh voi: CSQ (ResNet-50) = 0.748,  HashNet = 0.713
```

### 6.4 Danh gia NWPU

```bash
python experiments/evaluate.py \
    --checkpoint ./checkpoints/best_model_nwpu_vit.pth
```

### 6.5 Ablation study

```bash
python experiments/ablation.py --all
```

### 6.6 Visualization

```bash
python experiments/visualize.py \
    --checkpoint ./checkpoints/best_model_nwpu_vit.pth
```

### 6.7 Demo app

```bash
# Build vector database truoc
python scripts/build_vector_db.py \
    --checkpoint ./checkpoints/best_model_nwpu_vit.pth

# Chay app
streamlit run app.py
```

---

## 7. KET QUA

### 7.1 NWPU-RESISC45

| Model | mAP | Epoch |
|-------|-----|-------|
| ViT-B/32 + CSQ | **0.9112** | 20 |

> NWPU khong phai benchmark retrieval chuan -- so mAP nay khong so sanh duoc voi SOTA.

### 7.2 NUS-WIDE (tham chieu SOTA)

| Method | Backbone | mAP@64bit |
|--------|----------|-----------|
| HashNet | AlexNet | 0.713 |
| CSQ | ResNet-50 | 0.748 |
| **ViT + MultiLabel CSQ** | **ViT-B/32** | *cho thuc nghiem* |

Ky vong ViT backbone (pretrain ImageNet-21k multi-label) dat **0.80-0.85**.

### 7.3 Chi tiet checkpoint NWPU

```
checkpoints/best_model_nwpu_vit.pth
  epoch          : 20
  mAP            : 0.9112
  backbone keys  : 150 tensors  (ViT-B/32)
  hashing head   : hashing_head.{1,3}.{weight,bias}
  architecture   : 768 -> 1024 -> 64 bits
```

---

## 8. TAI LIEU THAM KHAO

1. Dosovitskiy et al. (2020). "An Image is Worth 16x16 Words." ICLR 2021.
2. Oquab et al. (2023). "DINOv2: Learning Robust Visual Features without Supervision." CVPR 2024.
3. Yuan et al. (2020). "Central Similarity Quantization for Efficient Image and Video Retrieval." CVPR 2020.
4. Cao et al. (2017). "HashNet: Deep Learning to Hash by Continuation." ICCV 2017.
5. Cao et al. (2018). "Deep Cauchy Hashing for Hamming Space Retrieval." CVPR 2018.
6. Cheng et al. (2017). "Remote Sensing Image Scene Classification." Proc. IEEE.

---

*Cap nhat: Thang 3/2026*
