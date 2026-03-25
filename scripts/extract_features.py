"""
Simple Feature Extraction & Database Creation
==============================================

Script đơn giản để:
1. Load model đã fine-tune
2. Trích xuất hash codes từ toàn bộ dataset
3. Lưu thành database để query sau

Usage:
    python scripts/extract_features.py
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from datetime import datetime

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.vit_hashing import ViT_Hashing
from src.models.dinov2_hashing import DINOv3Hashing


# ============================================================================
# CONFIGURATION - Chỉnh sửa ở đây
# ============================================================================

CONFIG = {
    # Model checkpoint đã train
    'checkpoint_path': './checkpoints/best_model.pth',
    
    # Thư mục chứa ảnh NWPU
    'data_dir': './data/archive/Dataset/train/train',
    
    # Nơi lưu database
    'output_dir': './database',
    
    # Batch size (giảm nếu OOM)
    'batch_size': 32,
    
    # Sử dụng GPU
    'device': 'cuda',
}


# ============================================================================
# MAIN CODE
# ============================================================================

def main():
    print("=" * 60)
    print("FEATURE EXTRACTION & DATABASE CREATION")
    print("=" * 60)
    
    # Setup device
    device = torch.device(CONFIG['device'] if torch.cuda.is_available() else 'cpu')
    print(f"\n[Device] {device}")
    
    # -------------------------------------------------------------------------
    # 1. Load Model
    # -------------------------------------------------------------------------
    print(f"\n[1/4] Loading model from {CONFIG['checkpoint_path']}")
    
    checkpoint = torch.load(CONFIG['checkpoint_path'], map_location=device)
    
    # Get model info
    hash_bit = checkpoint.get('hash_bit', 64)
    model_type = checkpoint.get('model_type', 'vit')
    num_classes = checkpoint.get('num_classes', 45)
    
    print(f"      Model: {model_type}")
    print(f"      Hash bits: {hash_bit}")
    print(f"      Classes: {num_classes}")
    
    # Create model
    if model_type == 'vit':
        model = ViT_Hashing(hash_bit=hash_bit, num_classes=num_classes)
    else:
        model = DINOv3Hashing(hash_bit=hash_bit, num_classes=num_classes)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print("      [✓] Model loaded!")
    
    # -------------------------------------------------------------------------
    # 2. Load Dataset
    # -------------------------------------------------------------------------
    print(f"\n[2/4] Loading dataset from {CONFIG['data_dir']}")
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    dataset = ImageFolder(CONFIG['data_dir'], transform=transform)
    
    print(f"      Total images: {len(dataset)}")
    print(f"      Classes: {len(dataset.classes)}")
    
    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # -------------------------------------------------------------------------
    # 3. Extract Features
    # -------------------------------------------------------------------------
    print(f"\n[3/4] Extracting hash codes...")
    
    all_hash_codes = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting"):
            images = images.to(device)
            
            # Forward pass
            hash_output, _ = model(images)
            
            # Binarize
            binary_codes = torch.sign(hash_output).cpu().numpy()
            
            all_hash_codes.append(binary_codes)
            all_labels.append(labels.numpy())
    
    hash_codes = np.concatenate(all_hash_codes, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    print(f"      Hash codes shape: {hash_codes.shape}")
    print(f"      Labels shape: {labels.shape}")
    
    # -------------------------------------------------------------------------
    # 4. Save Database
    # -------------------------------------------------------------------------
    print(f"\n[4/4] Saving database...")
    
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # Get image paths
    image_paths = [sample[0] for sample in dataset.samples]
    
    # Save as npz
    output_path = os.path.join(CONFIG['output_dir'], 'nwpu_database.npz')
    
    np.savez_compressed(
        output_path,
        # Hash codes: [N, hash_bit] với giá trị {-1, 1}
        hash_codes=hash_codes,
        # Labels: [N] class indices
        labels=labels,
        # Image paths
        image_paths=np.array(image_paths, dtype=object),
        # Class names
        class_names=np.array(dataset.classes, dtype=object),
        # Metadata
        hash_bit=hash_bit,
        model_type=model_type,
        num_images=len(image_paths),
        created_at=datetime.now().isoformat(),
    )
    
    # Calculate size
    file_size = os.path.getsize(output_path) / 1024 / 1024
    
    print(f"      [✓] Saved to: {output_path}")
    print(f"      File size: {file_size:.2f} MB")
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("DATABASE CREATED SUCCESSFULLY!")
    print("=" * 60)
    print(f"""
Database Info:
  - Path: {output_path}
  - Images: {len(image_paths):,}
  - Hash bits: {hash_bit}
  - Classes: {len(dataset.classes)}
  - Size: {file_size:.2f} MB

Storage efficiency:
  - Original features: ~{len(image_paths) * 768 * 4 / 1024 / 1024:.1f} MB (768-dim float32)
  - Hash codes: ~{len(image_paths) * hash_bit / 8 / 1024 / 1024:.2f} MB ({hash_bit}-bit binary)
  - Compression ratio: {768 * 32 / hash_bit:.0f}x

Next steps:
  1. Query database:
     python scripts/build_vector_db.py query --image test.jpg --database {output_path} --checkpoint {CONFIG['checkpoint_path']}
  
  2. Run demo:
     python scripts/build_vector_db.py demo --database {output_path} --checkpoint {CONFIG['checkpoint_path']}
    """)


if __name__ == '__main__':
    main()
