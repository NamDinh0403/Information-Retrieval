"""
NUS-WIDE Dataset Downloader
===========================

Downloads NUS-WIDE dataset from Kaggle for multi-label image retrieval.

Dataset info:
    - ~270,000 images from Flickr
    - 81 ground truth concepts (multi-label)
    - Standard benchmark for image hashing/retrieval
    - Source: https://www.kaggle.com/datasets/xinleili/nuswide

Usage:
    # Method 1: Using Kaggle API (recommended)
    python download_nuswide.py --method kaggle
    
    # Method 2: Manual download instructions
    python download_nuswide.py --method manual
    
Requirements:
    - kaggle package: pip install kaggle
    - Kaggle API token: ~/.kaggle/kaggle.json
"""

import os
import sys
import zipfile
import shutil
import argparse
from pathlib import Path


def check_kaggle_api():
    """Check if Kaggle API is available and configured."""
    try:
        import kaggle
        return True
    except ImportError:
        print("[!] Kaggle package not found. Installing...")
        os.system(f"{sys.executable} -m pip install kaggle -q")
        return True
    except OSError as e:
        if "Could not find kaggle.json" in str(e):
            return False
        raise


def setup_kaggle_credentials():
    """Guide user to setup Kaggle API credentials."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                    KAGGLE API SETUP REQUIRED                      ║
╠══════════════════════════════════════════════════════════════════╣
║ To download from Kaggle, you need to setup API credentials:       ║
║                                                                   ║
║ 1. Go to https://www.kaggle.com/account                          ║
║ 2. Click "Create New API Token"                                   ║
║ 3. Save kaggle.json to:                                          ║
║    - Windows: C:\\Users\\<username>\\.kaggle\\kaggle.json          ║
║    - Linux/Mac: ~/.kaggle/kaggle.json                            ║
║                                                                   ║
║ Then run this script again.                                       ║
╚══════════════════════════════════════════════════════════════════╝
    """)


def download_from_kaggle(data_dir: str = './data'):
    """
    Download NUS-WIDE from Kaggle.
    
    Dataset: https://www.kaggle.com/datasets/xinleili/nuswide
    """
    if not check_kaggle_api():
        setup_kaggle_credentials()
        return None
    
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
    except Exception as e:
        print(f"[Error] Kaggle authentication failed: {e}")
        setup_kaggle_credentials()
        return None
    
    os.makedirs(data_dir, exist_ok=True)
    
    extract_dir = os.path.join(data_dir, 'NUS-WIDE')
    
    # Check if already exists
    if os.path.exists(extract_dir) and os.path.exists(os.path.join(extract_dir, 'Flickr')):
        print(f"[✓] NUS-WIDE already exists at {extract_dir}")
        return extract_dir
    
    print("=" * 60)
    print("NUS-WIDE DATASET DOWNLOADER")
    print("=" * 60)
    print(f"[*] Target directory: {os.path.abspath(extract_dir)}")
    print("[*] This dataset is ~30GB, download may take a while...")
    print()
    
    # Download from Kaggle
    print("[1/3] Downloading from Kaggle...")
    try:
        api.dataset_download_files(
            'xinleili/nuswide',
            path=data_dir,
            unzip=False
        )
        print("[✓] Download completed!")
    except Exception as e:
        print(f"[Error] Download failed: {e}")
        return None
    
    # Find and extract zip file
    zip_path = os.path.join(data_dir, 'nuswide.zip')
    if not os.path.exists(zip_path):
        # Try to find any zip file
        for f in os.listdir(data_dir):
            if f.endswith('.zip') and 'nus' in f.lower():
                zip_path = os.path.join(data_dir, f)
                break
    
    if os.path.exists(zip_path):
        print(f"[2/3] Extracting {zip_path}...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            print("[✓] Extraction completed!")
            
            # Clean up zip file to save space
            print("[3/3] Cleaning up...")
            os.remove(zip_path)
            print("[✓] Cleanup completed!")
            
        except Exception as e:
            print(f"[Error] Extraction failed: {e}")
            return None
    
    print()
    print("=" * 60)
    print(f"[✓] NUS-WIDE dataset ready at: {os.path.abspath(extract_dir)}")
    print("=" * 60)
    
    return extract_dir


def print_manual_instructions():
    """Print instructions for manual download."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║               NUS-WIDE MANUAL DOWNLOAD INSTRUCTIONS               ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║ Option 1: Download from Kaggle (Recommended)                      ║
║ ─────────────────────────────────────────────                     ║
║ 1. Go to: https://www.kaggle.com/datasets/xinleili/nuswide       ║
║ 2. Click "Download" (requires Kaggle account)                     ║
║ 3. Extract to: ./data/NUS-WIDE/                                  ║
║                                                                   ║
║ Option 2: Download from Official Source                           ║
║ ───────────────────────────────────────                           ║
║ 1. Go to: https://lms.comp.nus.edu.sg/wp-content/uploads/        ║
║           2019/research/nuswide/NUS-WIDE.html                     ║
║ 2. Download:                                                      ║
║    - Flickr images (requires agreement form)                      ║
║    - Groundtruth labels                                           ║
║    - Image list                                                   ║
║ 3. Extract all to: ./data/NUS-WIDE/                              ║
║                                                                   ║
║ Expected directory structure:                                     ║
║ ─────────────────────────────                                     ║
║ ./data/NUS-WIDE/                                                  ║
║     Flickr/                  # Image files                        ║
║         *.jpg                                                     ║
║     Groundtruth/                                                  ║
║         AllLabels81/         # 81-class labels                    ║
║     ImageList/                                                    ║
║         Imagelist.txt        # All image list                     ║
║         TrainImagelist.txt   # Training images                    ║
║         TestImagelist.txt    # Test images                        ║
║                                                                   ║
╚══════════════════════════════════════════════════════════════════╝
    """)


def verify_dataset(data_dir: str = './data/NUS-WIDE') -> bool:
    """Verify that NUS-WIDE dataset is properly downloaded."""
    print("\n[*] Verifying NUS-WIDE dataset...")
    
    required_dirs = [
        'Flickr',
        'Groundtruth',
        'Groundtruth/AllLabels81',
        'ImageList',
    ]
    
    required_files = [
        'ImageList/Imagelist.txt',
        'ImageList/TrainImagelist.txt', 
        'ImageList/TestImagelist.txt',
    ]
    
    all_good = True
    
    for d in required_dirs:
        path = os.path.join(data_dir, d)
        if os.path.exists(path):
            print(f"  [✓] {d}/")
        else:
            print(f"  [✗] {d}/ - MISSING")
            all_good = False
    
    for f in required_files:
        path = os.path.join(data_dir, f)
        if os.path.exists(path):
            print(f"  [✓] {f}")
        else:
            print(f"  [✗] {f} - MISSING")
            all_good = False
    
    # Count images
    flickr_dir = os.path.join(data_dir, 'Flickr')
    if os.path.exists(flickr_dir):
        num_images = len([f for f in os.listdir(flickr_dir) if f.endswith('.jpg')])
        print(f"\n  Total images: {num_images:,}")
    
    if all_good:
        print("\n[✓] NUS-WIDE dataset verification PASSED!")
    else:
        print("\n[✗] NUS-WIDE dataset verification FAILED!")
        print("    Please check the missing components.")
    
    return all_good


def create_preprocessed_splits(data_dir: str = './data/NUS-WIDE'):
    """
    Create preprocessed train/query/database splits for faster loading.
    
    Standard protocol for hashing research:
    - Use 21 most frequent labels
    - Query: ~2,100 images (100 per class from test set)
    - Database: remaining test images
    - Train: all training images with at least one of 21 labels
    """
    print("\n[*] Creating preprocessed splits...")
    
    # Import the loader
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.data.nuswide_loader import preprocess_nuswide
    
    preprocess_nuswide(data_dir)


def main():
    parser = argparse.ArgumentParser(description='Download NUS-WIDE dataset')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Directory to save dataset')
    parser.add_argument('--method', type=str, choices=['kaggle', 'manual', 'verify'],
                        default='kaggle',
                        help='Download method: kaggle (auto), manual (instructions), verify (check existing)')
    parser.add_argument('--preprocess', action='store_true',
                        help='Create preprocessed splits after download')
    
    args = parser.parse_args()
    
    if args.method == 'manual':
        print_manual_instructions()
        return
    
    if args.method == 'verify':
        verify_dataset(os.path.join(args.data_dir, 'NUS-WIDE'))
        return
    
    # Download
    result = download_from_kaggle(args.data_dir)
    
    if result:
        # Verify
        if verify_dataset(result):
            # Preprocess if requested
            if args.preprocess:
                create_preprocessed_splits(result)
    else:
        print("\n[!] Automatic download failed. Showing manual instructions...")
        print_manual_instructions()


if __name__ == '__main__':
    main()
