# PROJECT SPECIFICATION: Vision Transformer (ViT) for Image Retrieval and Data Science

## 1. Context & Objectives (Master's Level)
- **Subject 1 (Visual Information Retrieval):** Focus on Content-Based Image Retrieval (CBIR) using Vision Transformer Hashing (VTS) and EET (2026).
- **Subject 2 (Image Data Science):** Focus on applying SOTA Foundation Models (DINOv3, Falcon, ChromaFormer) for scientific data like Medical and Remote Sensing.

## 2. Core Architecture: Vision Transformer Hashing (VTS)
VTS replaces CNN backbones with ViT to capture global dependencies for better hash code generation.

### Detailed Architectural Flow:
- **Image Patching:** Split input $I \in \mathbb{R}^{224 \times 224 \times 3}$ into $N$ non-overlapping patches of $16 \times 16$.
- **Linear Projection:** Flatten patches and project to embedding dimension $D$ (e.g., 768 for ViT-B).
- **CLS Token & Positional Encoding:** Prepend a learnable `[class]` token and add 1D learnable position embeddings to maintain spatial structures.
- **Transformer Encoder:** Pass through 12 blocks of Multi-Head Self-Attention (MHSA) and MLP layers.
- **Hashing Head (The Custom Part):**
  - **Input:** `[class]` token (cls) or global average of all tokens.
  - **Layer 1:** Dense (1024) + ReLU + Dropout (0.5).
  - **Layer 2:** Final Dense layer with $K$ neurons (where $K$ = 16, 32, or 64 bit hash length).
  - **Activation:** `tanh` or `sign` to produce pseudo-binary/binary codes $\{-1, 1\}^K$.

## 3. Latest SOTA Papers (2025-2026) for Comparative Analysis
Include these in the project's "Related Works" and "SOTA Benchmarking":

### Visual Information Retrieval:
- **EET (Efficient & Effective ViT, 2026):** Implements content-based token pruning, reducing inference latency by 42.7% and boosting 16-bit retrieval by 5.15%.
- **DINOv3 (Meta AI, 08/2025):** 7B parameter foundation model using Gram Anchoring to prevent dense feature degradation. Best for zero-shot retrieval.
- **SRGTNet (2025):** Uses significant region erasure to force the network to generate more compact and discriminative hash codes.

### Image Data Science:
- **Falcon (2025):** A Remote Sensing Vision-Language model (0.7B parameters) for 14 tasks like change detection and segmentation.
- **ChromaFormer (2025):** Specialized for multi-spectral land cover classification (Sentinel-2 data) with over 95% accuracy.
- **RCV-UNet (2026):** Best for medical segmentation (fractional-order dynamics in feature propagation).

## 4. Dataset Specifications for Fine-tuning
- **CIFAR-10 (Small-scale):** 60k images, 10 classes. Train on 5,000, Query on 1,000.
- **NUS-WIDE (Large-scale/Scientific):** 269k multi-label images. Query set: 2,100. Training set: 10,500.
- **MS-COCO:** Useful for complex scene retrieval benchmarks.
- **NWPU-RESISC45:** Ideal for Remote Sensing Data Science projects.
- **ChestX-ray8 / NIH Chest X-ray:** Standard for Medical Image Data Science.

## 5. Technical Requirements for Copilot Code Generation
- **Libraries:** `torch`, `torchvision`, `timm` (PyTorch Image Models), `transformers` (Hugging Face). `qdrant-client` or `faiss` for vector database indexing.

### Training Configuration (Fine-tuning):
- **Optimizer:** AdamW (learning rate $3 \times 10^{-5}$ for backbone, $1 \times 10^{-4}$ for hashing head).
- **Loss Function:**
  - **Central Similarity Quantization (CSQ):** Recommended as the best framework for VTS.
  - **Quantization Loss:** Minimizes error between continuous outputs and discrete binary codes.
- **Scheduler:** CosineAnnealingLR or OneCycleLR for smooth convergence.

## 6. Implementation Steps (Prompt for Copilot)
- **Define Model:** Create a `ViT_Hashing` class inheriting from `timm.models.vision_transformer`.
- **Modify Head:** Replace `model.head` with the VTS Hashing Head (Dropout-Dense-ReLU-Dense).
- **Data Loader:** Implement a multi-label data loader for NUS-WIDE or ChestX-ray8 using `torchxrayvision` or `datasets`.
- **Training Loop:** Implement CSQ loss (Binary Cross Entropy + Quantization Error).
- **Retrieval Engine:** Use cosine similarity or Hamming distance to retrieve Top-K images.
- **Evaluation:** Calculate mean Average Precision (mAP) and plot Precision-Recall Curves.