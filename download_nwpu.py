"""
NWPU-RESISC45 Dataset Downloader
================================
Bộ dữ liệu viễn thám NWPU-RESISC45 cho đánh giá Image Retrieval.

Dataset info:
    - 31,500 images (45 classes × 700 images/class)
    - Resolution: 256×256 pixels
    - Size: ~400MB (compressed)

Usage:
    python download_nwpu.py
    python download_nwpu.py --method gdrive
    python download_nwpu.py --method kaggle
"""

import os
import sys
import zipfile
import shutil
import argparse
from pathlib import Path


def check_gdown():
    """Kiểm tra và cài đặt gdown nếu cần."""
    try:
        import gdown
        return True
    except ImportError:
        print("[!] Đang cài đặt gdown...")
        os.system(f"{sys.executable} -m pip install gdown -q")
        return True


def download_from_gdrive(data_dir: str = './data'):
    """
    Tải NWPU-RESISC45 từ Google Drive.
    
    Nguồn: Mirror từ các nghiên cứu công khai
    """
    check_gdown()
    import gdown
    
    # Tạo thư mục
    os.makedirs(data_dir, exist_ok=True)
    
    zip_path = os.path.join(data_dir, 'NWPU-RESISC45.zip')
    extract_dir = os.path.join(data_dir, 'NWPU-RESISC45')
    
    # Kiểm tra nếu đã tồn tại
    if os.path.exists(extract_dir) and len(os.listdir(extract_dir)) == 45:
        print(f"[✓] NWPU-RESISC45 đã tồn tại tại {extract_dir}")
        print(f"    Số classes: {len(os.listdir(extract_dir))}")
        return extract_dir
    
    print("=" * 60)
    print("NWPU-RESISC45 DATASET DOWNLOADER")
    print("=" * 60)
    print(f"[*] Thư mục đích: {os.path.abspath(extract_dir)}")
    print(f"[*] Kích thước ước tính: ~400MB")
    print()
    
    # Google Drive file ID (public mirror)
    # Nguồn: https://github.com/weecology/NWPU-RESISC45-Dataset hoặc mirrors khác
    gdrive_urls = [
        # Mirror 1: Common research mirror
        "https://drive.google.com/uc?id=1DnPSU5nVSN7xv95bpZ3XQ0JhKXZOKgIv",
        # Mirror 2: Alternative
        "https://drive.google.com/uc?id=1mKy5pLqJDm_hnGLjLTpXW1bSWNwWGJZF",
    ]
    
    downloaded = False
    
    for i, url in enumerate(gdrive_urls):
        try:
            print(f"[*] Thử mirror {i+1}/{len(gdrive_urls)}...")
            gdown.download(url, zip_path, quiet=False)
            
            if os.path.exists(zip_path) and os.path.getsize(zip_path) > 100_000_000:  # > 100MB
                downloaded = True
                break
        except Exception as e:
            print(f"[!] Mirror {i+1} thất bại: {e}")
            continue
    
    if not downloaded:
        print("\n[!] Không thể tải tự động từ Google Drive.")
        print_manual_instructions(data_dir)
        return None
    
    # Giải nén
    print(f"\n[*] Đang giải nén {zip_path}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print(f"[✓] Giải nén hoàn tất!")
        
        # Xóa file zip để tiết kiệm dung lượng
        os.remove(zip_path)
        print(f"[✓] Đã xóa file zip tạm")
        
    except zipfile.BadZipFile:
        print(f"[✗] File zip bị lỗi. Vui lòng tải thủ công.")
        print_manual_instructions(data_dir)
        return None
    
    # Kiểm tra kết quả
    if os.path.exists(extract_dir):
        classes = os.listdir(extract_dir)
        print(f"\n[✓] Tải thành công NWPU-RESISC45!")
        print(f"    Đường dẫn: {os.path.abspath(extract_dir)}")
        print(f"    Số classes: {len(classes)}")
        print(f"    Classes mẫu: {classes[:5]}...")
        return extract_dir
    
    return None


def download_from_kaggle(data_dir: str = './data'):
    """
    Tải NWPU-RESISC45 từ Kaggle.
    Yêu cầu: Kaggle API đã được cấu hình (~/.kaggle/kaggle.json)
    """
    print("=" * 60)
    print("NWPU-RESISC45 TỪ KAGGLE")
    print("=" * 60)
    
    try:
        import kaggle
    except ImportError:
        print("[!] Đang cài đặt kaggle...")
        os.system(f"{sys.executable} -m pip install kaggle -q")
    
    extract_dir = os.path.join(data_dir, 'NWPU-RESISC45')
    
    print("[*] Yêu cầu: File ~/.kaggle/kaggle.json với API token")
    print("[*] Lấy token tại: https://www.kaggle.com/settings -> API -> Create New Token")
    print()
    
    try:
        os.makedirs(data_dir, exist_ok=True)
        
        # Dataset trên Kaggle
        dataset_name = "tpsison/nwpu-resisc45"
        
        print(f"[*] Đang tải từ Kaggle: {dataset_name}")
        os.system(f"kaggle datasets download -d {dataset_name} -p {data_dir} --unzip")
        
        if os.path.exists(extract_dir):
            print(f"[✓] Tải thành công!")
            return extract_dir
            
    except Exception as e:
        print(f"[✗] Lỗi: {e}")
    
    print_manual_instructions(data_dir)
    return None


def download_from_onedrive(data_dir: str = './data'):
    """
    Hướng dẫn tải từ OneDrive chính thức.
    """
    print("=" * 60)
    print("NWPU-RESISC45 TỪ ONEDRIVE (Chính thức)")
    print("=" * 60)
    print()
    print("Nguồn chính thức từ Northwestern Polytechnical University:")
    print()
    print("1. Truy cập: https://gcheng.net/NWPU-RESISC45")
    print("2. Điền form yêu cầu (academic use)")
    print("3. Nhận link OneDrive qua email")
    print("4. Tải và giải nén vào thư mục:")
    print(f"   {os.path.abspath(os.path.join(data_dir, 'NWPU-RESISC45'))}")
    print()
    return None


def print_manual_instructions(data_dir: str):
    """In hướng dẫn tải thủ công."""
    extract_dir = os.path.join(data_dir, 'NWPU-RESISC45')
    
    print()
    print("=" * 60)
    print("HƯỚNG DẪN TẢI THỦ CÔNG")
    print("=" * 60)
    print()
    print("Cách 1: Google Drive (nhanh nhất)")
    print("  - Tìm kiếm 'NWPU-RESISC45 dataset google drive'")
    print("  - Hoặc: https://www.google.com/search?q=NWPU-RESISC45+dataset+download")
    print()
    print("Cách 2: Kaggle")
    print("  - https://www.kaggle.com/datasets/tpsison/nwpu-resisc45")
    print("  - Đăng nhập và Download")
    print()
    print("Cách 3: OneDrive chính thức")
    print("  - https://gcheng.net/NWPU-RESISC45")
    print("  - Điền form academic request")
    print()
    print("Sau khi tải, giải nén sao cho cấu trúc như sau:")
    print(f"  {extract_dir}/")
    print("    ├── airplane/")
    print("    ├── airport/")
    print("    ├── baseball_diamond/")
    print("    ├── ... (45 folders)")
    print("    └── wetland/")
    print()


def verify_dataset(data_dir: str = './data'):
    """Kiểm tra dataset đã tải đúng chưa."""
    extract_dir = os.path.join(data_dir, 'NWPU-RESISC45')
    
    print("\n" + "=" * 60)
    print("KIỂM TRA DATASET")
    print("=" * 60)
    
    if not os.path.exists(extract_dir):
        print(f"[✗] Không tìm thấy: {extract_dir}")
        return False
    
    classes = [d for d in os.listdir(extract_dir) 
               if os.path.isdir(os.path.join(extract_dir, d))]
    
    if len(classes) != 45:
        print(f"[!] Cảnh báo: Tìm thấy {len(classes)} classes (expected: 45)")
    
    total_images = 0
    class_counts = {}
    
    for cls in classes:
        cls_path = os.path.join(extract_dir, cls)
        images = [f for f in os.listdir(cls_path) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif'))]
        class_counts[cls] = len(images)
        total_images += len(images)
    
    print(f"[✓] Đường dẫn: {os.path.abspath(extract_dir)}")
    print(f"[✓] Số classes: {len(classes)}")
    print(f"[✓] Tổng số ảnh: {total_images:,}")
    print(f"[✓] Trung bình/class: {total_images // len(classes) if classes else 0}")
    
    # Hiển thị một số classes
    print(f"\nCác classes ({len(classes)}):")
    for i, cls in enumerate(sorted(classes)[:10]):
        print(f"  {i+1}. {cls}: {class_counts[cls]} images")
    if len(classes) > 10:
        print(f"  ... và {len(classes) - 10} classes khác")
    
    expected_images = 45 * 700  # 31,500
    if total_images >= expected_images * 0.95:
        print(f"\n[✓] Dataset hoàn chỉnh! Sẵn sàng cho evaluation.")
        return True
    else:
        print(f"\n[!] Dataset có thể chưa đầy đủ (expected ~{expected_images:,} images)")
        return False


def create_splits(data_dir: str = './data', query_ratio: float = 0.1, seed: int = 42):
    """
    Tạo train/query splits cho retrieval evaluation.
    
    Args:
        data_dir: Thư mục chứa NWPU-RESISC45
        query_ratio: Tỷ lệ query set (default: 10% = 70 images/class)
        seed: Random seed
    """
    import random
    random.seed(seed)
    
    extract_dir = os.path.join(data_dir, 'NWPU-RESISC45')
    splits_dir = os.path.join(data_dir, 'NWPU-RESISC45-splits')
    
    if not os.path.exists(extract_dir):
        print("[✗] Dataset chưa được tải!")
        return None
    
    print("\n" + "=" * 60)
    print("TẠO TRAIN/QUERY SPLITS")
    print("=" * 60)
    
    # Tạo thư mục splits
    train_dir = os.path.join(splits_dir, 'database')
    query_dir = os.path.join(splits_dir, 'query')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(query_dir, exist_ok=True)
    
    classes = sorted([d for d in os.listdir(extract_dir) 
                     if os.path.isdir(os.path.join(extract_dir, d))])
    
    train_count = 0
    query_count = 0
    
    for cls in classes:
        cls_path = os.path.join(extract_dir, cls)
        images = sorted([f for f in os.listdir(cls_path) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif'))])
        
        # Shuffle và split
        random.shuffle(images)
        n_query = int(len(images) * query_ratio)
        
        query_images = images[:n_query]
        train_images = images[n_query:]
        
        # Tạo thư mục class
        os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(query_dir, cls), exist_ok=True)
        
        # Copy/link files
        for img in train_images:
            src = os.path.join(cls_path, img)
            dst = os.path.join(train_dir, cls, img)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
            train_count += 1
        
        for img in query_images:
            src = os.path.join(cls_path, img)
            dst = os.path.join(query_dir, cls, img)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
            query_count += 1
    
    print(f"[✓] Database (gallery): {train_count:,} images")
    print(f"[✓] Query set: {query_count:,} images")
    print(f"[✓] Splits saved to: {os.path.abspath(splits_dir)}")
    
    # Lưu metadata
    metadata = {
        'database_size': train_count,
        'query_size': query_count,
        'num_classes': len(classes),
        'query_ratio': query_ratio,
        'seed': seed
    }
    
    import json
    with open(os.path.join(splits_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return splits_dir


def main():
    parser = argparse.ArgumentParser(
        description='Download NWPU-RESISC45 Dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_nwpu.py                    # Auto download từ Google Drive
  python download_nwpu.py --method kaggle    # Tải từ Kaggle
  python download_nwpu.py --method onedrive  # Hiển thị hướng dẫn OneDrive
  python download_nwpu.py --verify           # Kiểm tra dataset
  python download_nwpu.py --create-splits    # Tạo train/query splits
        """
    )
    
    parser.add_argument('--method', choices=['gdrive', 'kaggle', 'onedrive', 'manual'],
                       default='gdrive', help='Phương thức tải (default: gdrive)')
    parser.add_argument('--data-dir', default='./data', help='Thư mục lưu data')
    parser.add_argument('--verify', action='store_true', help='Chỉ kiểm tra dataset')
    parser.add_argument('--create-splits', action='store_true', 
                       help='Tạo train/query splits sau khi tải')
    parser.add_argument('--query-ratio', type=float, default=0.1,
                       help='Tỷ lệ query set (default: 0.1)')
    
    args = parser.parse_args()
    
    if args.verify:
        verify_dataset(args.data_dir)
        return
    
    # Download theo method
    result = None
    if args.method == 'gdrive':
        result = download_from_gdrive(args.data_dir)
    elif args.method == 'kaggle':
        result = download_from_kaggle(args.data_dir)
    elif args.method == 'onedrive':
        result = download_from_onedrive(args.data_dir)
    else:
        print_manual_instructions(args.data_dir)
    
    # Verify
    if result or os.path.exists(os.path.join(args.data_dir, 'NWPU-RESISC45')):
        verify_dataset(args.data_dir)
        
        # Create splits nếu được yêu cầu
        if args.create_splits:
            create_splits(args.data_dir, args.query_ratio)
    
    print("\n" + "=" * 60)
    print("HOÀN TẤT")
    print("=" * 60)
    print("Tiếp theo, chạy evaluation:")
    print("  python main_research.py --week 1")
    print()


if __name__ == "__main__":
    main()
