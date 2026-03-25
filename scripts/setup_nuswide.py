"""
NUS-WIDE Complete Dataset Setup
================================

NUS-WIDE có cấu trúc phức tạp với labels lưu riêng biệt.
Script này giúp:
1. Kiểm tra dataset đã đủ chưa
2. Hướng dẫn download các phần còn thiếu
3. Tạo preprocessed files cho training

Official source:
    https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/NUS-WIDE.html

Kaggle (có thể thiếu labels):
    https://www.kaggle.com/datasets/xinleili/nuswide
"""

import os
import sys
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple


def check_nuswide_structure(data_dir: str) -> Dict[str, bool]:
    """
    Kiểm tra cấu trúc NUS-WIDE dataset.
    
    Returns:
        Dictionary với status của từng component
    """
    data_dir = Path(data_dir)
    
    checks = {
        'root_exists': data_dir.exists(),
        'flickr_images': (data_dir / 'Flickr').exists(),
        'groundtruth': (data_dir / 'Groundtruth').exists(),
        'labels_81': (data_dir / 'Groundtruth' / 'AllLabels81').exists(),
        'image_list': (data_dir / 'ImageList').exists(),
        'train_list': (data_dir / 'ImageList' / 'TrainImagelist.txt').exists(),
        'test_list': (data_dir / 'ImageList' / 'TestImagelist.txt').exists(),
    }
    
    # Count images if Flickr exists
    if checks['flickr_images']:
        flickr_dir = data_dir / 'Flickr'
        # Count all image files (handle nested structure)
        num_images = sum(1 for _ in flickr_dir.rglob('*.jpg'))
        checks['num_images'] = num_images
    else:
        checks['num_images'] = 0
    
    # Count label files if exists
    if checks['labels_81']:
        label_dir = data_dir / 'Groundtruth' / 'AllLabels81'
        label_files = list(label_dir.glob('Labels_*.txt'))
        checks['num_label_files'] = len(label_files)
    else:
        checks['num_label_files'] = 0
    
    return checks


def print_status(checks: Dict) -> bool:
    """Print status and return if dataset is complete."""
    print("\n" + "=" * 60)
    print("NUS-WIDE DATASET STATUS")
    print("=" * 60)
    
    status_icons = {True: "✓", False: "✗"}
    
    print(f"\n[Structure]")
    print(f"  {status_icons[checks['root_exists']]} Root directory exists")
    print(f"  {status_icons[checks['flickr_images']]} Flickr images folder")
    print(f"  {status_icons[checks['groundtruth']]} Groundtruth folder")
    print(f"  {status_icons[checks['labels_81']]} AllLabels81 folder")
    print(f"  {status_icons[checks['image_list']]} ImageList folder")
    print(f"  {status_icons[checks['train_list']]} Train image list")
    print(f"  {status_icons[checks['test_list']]} Test image list")
    
    print(f"\n[Counts]")
    print(f"  Images: {checks['num_images']:,}")
    print(f"  Label files: {checks['num_label_files']}/81")
    
    # Check if complete
    is_complete = (
        checks['flickr_images'] and 
        checks['labels_81'] and 
        checks['num_label_files'] >= 21 and  # At least 21 most frequent
        checks['train_list']
    )
    
    print(f"\n[Status] {'COMPLETE ✓' if is_complete else 'INCOMPLETE ✗'}")
    
    return is_complete


def print_download_instructions(checks: Dict):
    """Print instructions for missing components."""
    print("\n" + "=" * 60)
    print("DOWNLOAD INSTRUCTIONS")
    print("=" * 60)
    
    if not checks['flickr_images']:
        print("""
[MISSING: Images]
────────────────────
Option 1: Download from official source
  1. Go to: https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/NUS-WIDE.html
  2. Download "Flickr images" (requires form submission)
  3. Extract to: ./data/NUS-WIDE/Flickr/

Option 2: Download from Kaggle
  1. pip install kaggle
  2. kaggle datasets download -d xinleili/nuswide
  3. Extract to: ./data/NUS-WIDE/
        """)
    
    if not checks['labels_81'] or checks['num_label_files'] < 21:
        print("""
[MISSING: Labels (Groundtruth)]
────────────────────────────────
This is the most common issue when downloading from Kaggle!

Download from official source:
  1. Go to: https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/NUS-WIDE.html
  2. Download "Groundtruth" files
  3. Extract to: ./data/NUS-WIDE/Groundtruth/

Direct links (may change):
  - AllLabels81.zip: Contains 81 concept labels
  - TrainTestLabels.zip: Train/test split labels
        """)
    
    if not checks['train_list']:
        print("""
[MISSING: Image Lists]
──────────────────────
Download from official source:
  1. Download "ImageList" files
  2. Should contain:
     - Imagelist.txt (all images)
     - TrainImagelist.txt (training split)
     - TestImagelist.txt (test split)
        """)


# ============================================================================
# NUS-WIDE 21 CONCEPTS (most frequent, used in hashing papers)
# ============================================================================

NUSWIDE_21_CONCEPTS = [
    'clouds', 'person', 'water', 'animal', 'grass',
    'buildings', 'window', 'plants', 'lake', 'ocean',
    'road', 'tree', 'mountain', 'reflection', 'nighttime',
    'sky', 'street', 'beach', 'flowers', 'rocks', 'sunset'
]


def create_label_matrix(data_dir: str, concepts: List[str] = None) -> Tuple[np.ndarray, List[str]]:
    """
    Đọc labels từ các file txt và tạo label matrix.
    
    Args:
        data_dir: Path to NUS-WIDE root
        concepts: List of concepts to use (default: 21 most frequent)
    
    Returns:
        labels: [num_images, num_concepts] matrix
        image_list: List of image paths
    """
    data_dir = Path(data_dir)
    
    if concepts is None:
        concepts = NUSWIDE_21_CONCEPTS
    
    # Read image list
    with open(data_dir / 'ImageList' / 'Imagelist.txt', 'r') as f:
        image_list = [line.strip().replace('\\', '/') for line in f]
    
    num_images = len(image_list)
    print(f"Total images in list: {num_images}")
    
    # Read labels for each concept
    labels = np.zeros((num_images, len(concepts)), dtype=np.float32)
    
    label_dir = data_dir / 'Groundtruth' / 'AllLabels81'
    
    for i, concept in enumerate(concepts):
        label_file = label_dir / f'Labels_{concept}.txt'
        
        if label_file.exists():
            with open(label_file, 'r') as f:
                concept_labels = [int(line.strip()) for line in f]
            
            if len(concept_labels) == num_images:
                labels[:, i] = concept_labels
            else:
                print(f"[Warning] {concept}: {len(concept_labels)} labels != {num_images} images")
        else:
            print(f"[Warning] Label file not found: {label_file}")
    
    # Statistics
    print(f"\nLabel statistics:")
    print(f"  Total images: {num_images}")
    print(f"  Concepts: {len(concepts)}")
    print(f"  Images with at least 1 label: {(labels.sum(axis=1) > 0).sum()}")
    print(f"  Average labels per image: {labels.sum(axis=1).mean():.2f}")
    
    return labels, image_list


def create_train_test_split(
    data_dir: str,
    labels: np.ndarray,
    image_list: List[str]
) -> Dict:
    """
    Create train/test split based on official TrainImagelist/TestImagelist.
    """
    data_dir = Path(data_dir)
    
    # Read official splits
    with open(data_dir / 'ImageList' / 'TrainImagelist.txt', 'r') as f:
        train_images = set(line.strip().replace('\\', '/') for line in f)
    
    with open(data_dir / 'ImageList' / 'TestImagelist.txt', 'r') as f:
        test_images = set(line.strip().replace('\\', '/') for line in f)
    
    # Create indices
    train_indices = []
    test_indices = []
    
    for i, img_path in enumerate(image_list):
        if img_path in train_images:
            train_indices.append(i)
        elif img_path in test_images:
            test_indices.append(i)
    
    print(f"\nSplit statistics:")
    print(f"  Train: {len(train_indices)}")
    print(f"  Test: {len(test_indices)}")
    
    return {
        'train_indices': train_indices,
        'test_indices': test_indices,
        'train_images': [image_list[i] for i in train_indices],
        'test_images': [image_list[i] for i in test_indices],
        'train_labels': labels[train_indices],
        'test_labels': labels[test_indices],
    }


def preprocess_nuswide(
    data_dir: str,
    output_dir: str = None,
    use_21_concepts: bool = True
):
    """
    Preprocess NUS-WIDE dataset for faster loading during training.
    
    Creates:
        - train_images.txt: List of training image paths
        - train_labels.npy: Training labels [N, 21]
        - test_images.txt: List of test image paths  
        - test_labels.npy: Test labels [N, 21]
    """
    data_dir = Path(data_dir)
    
    if output_dir is None:
        output_dir = data_dir / 'preprocessed'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    concepts = NUSWIDE_21_CONCEPTS if use_21_concepts else None
    
    # Create label matrix
    print("\n[1/3] Reading labels...")
    labels, image_list = create_label_matrix(data_dir, concepts)
    
    # Create splits
    print("\n[2/3] Creating train/test split...")
    split = create_train_test_split(data_dir, labels, image_list)
    
    # Save
    print("\n[3/3] Saving preprocessed files...")
    
    suffix = '21' if use_21_concepts else '81'
    
    # Train
    with open(output_dir / f'train_images_{suffix}.txt', 'w') as f:
        for img in split['train_images']:
            f.write(img + '\n')
    np.save(output_dir / f'train_labels_{suffix}.npy', split['train_labels'])
    
    # Test (for query + database)
    with open(output_dir / f'test_images_{suffix}.txt', 'w') as f:
        for img in split['test_images']:
            f.write(img + '\n')
    np.save(output_dir / f'test_labels_{suffix}.npy', split['test_labels'])
    
    # Concepts list
    with open(output_dir / f'concepts_{suffix}.txt', 'w') as f:
        for c in (NUSWIDE_21_CONCEPTS if use_21_concepts else []):
            f.write(c + '\n')
    
    print(f"\n[✓] Preprocessed files saved to: {output_dir}")
    print(f"    - train_images_{suffix}.txt ({len(split['train_images'])} images)")
    print(f"    - train_labels_{suffix}.npy")
    print(f"    - test_images_{suffix}.txt ({len(split['test_images'])} images)")
    print(f"    - test_labels_{suffix}.npy")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='NUS-WIDE Dataset Setup')
    parser.add_argument('--data-dir', type=str, default='./data/NUS-WIDE',
                        help='Path to NUS-WIDE dataset')
    parser.add_argument('--check', action='store_true',
                        help='Check dataset structure')
    parser.add_argument('--preprocess', action='store_true',
                        help='Create preprocessed files')
    
    args = parser.parse_args()
    
    if args.check or not args.preprocess:
        checks = check_nuswide_structure(args.data_dir)
        is_complete = print_status(checks)
        
        if not is_complete:
            print_download_instructions(checks)
    
    if args.preprocess:
        checks = check_nuswide_structure(args.data_dir)
        if checks['labels_81'] and checks['num_label_files'] >= 21:
            preprocess_nuswide(args.data_dir)
        else:
            print("\n[Error] Cannot preprocess - labels are missing!")
            print("Please download Groundtruth files first.")


if __name__ == '__main__':
    main()
