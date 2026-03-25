"""
NUS-WIDE Dataset Loader for Multi-Label Image Retrieval
========================================================

NUS-WIDE Dataset Info:
    - ~270,000 images from Flickr
    - 81 ground truth concept labels (multi-label)
    - Standard splits for hashing research:
        * Database: ~190,000 images
        * Query: 2,100 images (selected from 21 most frequent tags)
        * Train: 10,500 images (for training hash functions)

Reference:
    Chua et al. "NUS-WIDE: A Real-World Web Image Database from 
    National University of Singapore" (CIVR 2009)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from typing import Optional, Tuple, List, Dict


# 21 most frequent tags used in hashing research
NUSWIDE_21_LABELS = [
    'clouds', 'person', 'water', 'animal', 'grass',
    'buildings', 'window', 'plants', 'lake', 'ocean',
    'road', 'tree', 'mountain', 'reflection', 'nighttime',
    'sky', 'street', 'beach', 'flowers', 'rocks', 'sunset'
]

# All 81 concept labels in NUS-WIDE
NUSWIDE_81_LABELS = [
    'airport', 'animal', 'beach', 'bear', 'birds', 'boats', 'book',
    'bridge', 'buildings', 'cars', 'castle', 'cat', 'cityscape',
    'clouds', 'computer', 'coral', 'cow', 'dancing', 'dog', 'earthquake',
    'elk', 'fire', 'fish', 'flags', 'flowers', 'food', 'fox', 'frost',
    'garden', 'glacier', 'grass', 'harbor', 'horses', 'house', 'lake',
    'leaf', 'map', 'military', 'moon', 'mountain', 'nighttime', 'ocean',
    'person', 'plane', 'plants', 'police', 'protest', 'railroad',
    'rainbow', 'reflection', 'road', 'rocks', 'running', 'sand', 'sign',
    'sky', 'snow', 'soccer', 'sports', 'statue', 'street', 'sun',
    'sunset', 'surf', 'swimmers', 'tattoo', 'temple', 'tiger', 'tower',
    'town', 'toy', 'train', 'tree', 'valley', 'vehicle', 'water',
    'waterfall', 'wedding', 'whales', 'window', 'zebra'
]


class NUSWIDEDataset(Dataset):
    """
    NUS-WIDE Dataset for Multi-Label Image Retrieval.
    
    Expected directory structure:
    data_dir/
        Flickr/                    # Image files
            actor0001.jpg
            actor0002.jpg
            ...
        Groundtruth/
            AllLabels81/          # 81-label annotations
                Labels_airport.txt
                Labels_animal.txt
                ...
            TrainTestLabels/      # Train/test splits
                Labels_airport_Train.txt
                Labels_airport_Test.txt
                ...
        ImageList/
            Imagelist.txt          # List of all images
            TrainImagelist.txt     # Training images
            TestImagelist.txt      # Test images
        NUS_WID_Tags/
            All_Tags.txt           # All tag annotations
        Concepts81.txt             # List of 81 concepts
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',  # 'train', 'query', 'database'
        use_21_labels: bool = True,  # Use 21 most frequent labels (standard for hashing)
        transform: Optional[transforms.Compose] = None,
        max_samples: Optional[int] = None,  # Limit samples for debugging
    ):
        """
        Args:
            data_dir: Root directory of NUS-WIDE dataset
            split: 'train', 'query', or 'database'
            use_21_labels: If True, use 21 most frequent labels (standard protocol)
            transform: Image transformations
            max_samples: Max number of samples (for debugging)
        """
        self.data_dir = data_dir
        self.split = split
        self.use_21_labels = use_21_labels
        self.labels_list = NUSWIDE_21_LABELS if use_21_labels else NUSWIDE_81_LABELS
        self.num_classes = len(self.labels_list)
        
        self.transform = transform or self._default_transform()
        
        # Load image list and labels
        self.image_paths, self.labels = self._load_data()
        
        if max_samples:
            self.image_paths = self.image_paths[:max_samples]
            self.labels = self.labels[:max_samples]
            
        print(f"[NUS-WIDE] Loaded {len(self)} images for '{split}' split")
        print(f"[NUS-WIDE] Using {self.num_classes} label concepts")
    
    def _default_transform(self) -> transforms.Compose:
        """Default transformation for ViT models."""
        if self.split == 'train':
            return transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def _load_data(self) -> Tuple[List[str], np.ndarray]:
        """Load image paths and multi-label annotations."""
        
        # Check for preprocessed files first (faster loading)
        preprocessed_dir = os.path.join(self.data_dir, 'preprocessed')
        split_suffix = '21' if self.use_21_labels else '81'
        
        img_list_file = os.path.join(preprocessed_dir, f'{self.split}_imgs_{split_suffix}.txt')
        label_file = os.path.join(preprocessed_dir, f'{self.split}_labels_{split_suffix}.npy')
        
        if os.path.exists(img_list_file) and os.path.exists(label_file):
            print(f"[NUS-WIDE] Loading preprocessed data from {preprocessed_dir}")
            with open(img_list_file, 'r') as f:
                image_paths = [line.strip() for line in f.readlines()]
            labels = np.load(label_file)
            return image_paths, labels
        
        # Otherwise, load from raw NUS-WIDE format
        print("[NUS-WIDE] Loading from raw format (first time may be slow)...")
        return self._load_raw_data()
    
    def _load_raw_data(self) -> Tuple[List[str], np.ndarray]:
        """Load data from raw NUS-WIDE directory structure."""
        
        image_list_dir = os.path.join(self.data_dir, 'ImageList')
        groundtruth_dir = os.path.join(self.data_dir, 'Groundtruth', 'AllLabels81')
        flickr_dir = os.path.join(self.data_dir, 'Flickr')
        
        # Determine which image list to use
        if self.split == 'train':
            list_file = os.path.join(image_list_dir, 'TrainImagelist.txt')
        else:  # query or database use test images
            list_file = os.path.join(image_list_dir, 'TestImagelist.txt')
        
        # Load image list
        if not os.path.exists(list_file):
            raise FileNotFoundError(
                f"Image list not found: {list_file}\n"
                f"Please download NUS-WIDE dataset and extract to {self.data_dir}"
            )
        
        with open(list_file, 'r') as f:
            all_images = [line.strip().replace('\\', '/') for line in f.readlines()]
        
        # Load labels for each concept
        all_labels = []
        for label_name in self.labels_list:
            label_file = os.path.join(groundtruth_dir, f'Labels_{label_name}.txt')
            if os.path.exists(label_file):
                with open(label_file, 'r') as f:
                    labels = [int(line.strip()) for line in f.readlines()]
                all_labels.append(labels)
            else:
                print(f"[Warning] Label file not found: {label_file}")
                all_labels.append([0] * len(all_images))
        
        # Convert to numpy array: [num_images, num_labels]
        labels_matrix = np.array(all_labels).T
        
        # Filter images that have at least one positive label
        valid_indices = np.where(labels_matrix.sum(axis=1) > 0)[0]
        
        # Create splits for query/database
        if self.split == 'query':
            # Standard: Use first 100 images per class for query (max ~2100 for 21 classes)
            query_indices = self._select_query_indices(labels_matrix, valid_indices)
            selected_indices = query_indices
        elif self.split == 'database':
            # Database: all valid test images except query
            query_indices = set(self._select_query_indices(labels_matrix, valid_indices))
            selected_indices = [i for i in valid_indices if i not in query_indices]
        else:  # train
            # Use all valid training images
            selected_indices = valid_indices.tolist()
        
        # Build final lists
        image_paths = []
        final_labels = []
        
        for idx in selected_indices:
            img_name = all_images[idx]
            img_path = os.path.join(flickr_dir, img_name)
            if os.path.exists(img_path):
                image_paths.append(img_path)
                final_labels.append(labels_matrix[idx])
        
        return image_paths, np.array(final_labels, dtype=np.float32)
    
    def _select_query_indices(self, labels_matrix: np.ndarray, valid_indices: np.ndarray, 
                              per_class: int = 100) -> List[int]:
        """Select query images: ~100 per class."""
        query_indices = set()
        
        for class_idx in range(self.num_classes):
            class_images = np.where(labels_matrix[valid_indices, class_idx] == 1)[0]
            selected = valid_indices[class_images[:per_class]].tolist()
            query_indices.update(selected)
        
        return list(query_indices)
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            image: Tensor [C, H, W]
            label: Tensor [num_classes] - multi-hot encoded
        """
        img_path = self.image_paths[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"[Error] Cannot load image: {img_path}")
            # Return a dummy black image
            image = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            image = self.transform(image)
        
        label = torch.from_numpy(self.labels[idx]).float()
        
        return image, label
    
    def get_label_names(self) -> List[str]:
        """Return list of label names."""
        return self.labels_list.copy()


def get_nuswide_dataloaders(
    data_dir: str = './data/NUS-WIDE',
    batch_size: int = 32,
    use_21_labels: bool = True,
    num_workers: int = 4,
    max_train: Optional[int] = None,
    max_query: Optional[int] = None,
    max_db: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Create DataLoaders for NUS-WIDE dataset.
    
    Args:
        data_dir: Root directory of NUS-WIDE dataset
        batch_size: Batch size
        use_21_labels: Use 21 most frequent labels (standard protocol)
        num_workers: Number of data loading workers
        max_train/query/db: Max samples for each split (for debugging)
    
    Returns:
        train_loader, query_loader, db_loader, label_names
    """
    print("=" * 60)
    print("Loading NUS-WIDE Dataset")
    print("=" * 60)
    
    train_dataset = NUSWIDEDataset(
        data_dir=data_dir,
        split='train',
        use_21_labels=use_21_labels,
        max_samples=max_train
    )
    
    query_dataset = NUSWIDEDataset(
        data_dir=data_dir,
        split='query',
        use_21_labels=use_21_labels,
        max_samples=max_query
    )
    
    db_dataset = NUSWIDEDataset(
        data_dir=data_dir,
        split='database',
        use_21_labels=use_21_labels,
        max_samples=max_db
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers, 
        pin_memory=True,
        drop_last=True
    )
    
    query_loader = DataLoader(
        query_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    db_loader = DataLoader(
        db_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print("=" * 60)
    print(f"Train: {len(train_dataset)} | Query: {len(query_dataset)} | Database: {len(db_dataset)}")
    print("=" * 60)
    
    return train_loader, query_loader, db_loader, train_dataset.get_label_names()


def preprocess_nuswide(data_dir: str, output_dir: Optional[str] = None):
    """
    Preprocess NUS-WIDE dataset for faster loading.
    Creates text files with image paths and numpy files with labels.
    
    Usage:
        python -c "from src.data.nuswide_loader import preprocess_nuswide; preprocess_nuswide('./data/NUS-WIDE')"
    """
    if output_dir is None:
        output_dir = os.path.join(data_dir, 'preprocessed')
    os.makedirs(output_dir, exist_ok=True)
    
    for use_21 in [True, False]:
        suffix = '21' if use_21 else '81'
        print(f"\n[Preprocessing] Using {suffix} labels...")
        
        for split in ['train', 'query', 'database']:
            print(f"  Processing {split}...")
            dataset = NUSWIDEDataset(
                data_dir=data_dir,
                split=split,
                use_21_labels=use_21
            )
            
            # Save image paths
            img_file = os.path.join(output_dir, f'{split}_imgs_{suffix}.txt')
            with open(img_file, 'w') as f:
                for path in dataset.image_paths:
                    f.write(path + '\n')
            
            # Save labels
            label_file = os.path.join(output_dir, f'{split}_labels_{suffix}.npy')
            np.save(label_file, dataset.labels)
            
            print(f"    Saved {len(dataset)} samples")
    
    print(f"\n[Done] Preprocessed data saved to {output_dir}")


if __name__ == '__main__':
    # Test loading
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='./data/NUS-WIDE')
    parser.add_argument('--preprocess', action='store_true', help='Preprocess dataset')
    args = parser.parse_args()
    
    if args.preprocess:
        preprocess_nuswide(args.data_dir)
    else:
        # Test loading
        train_loader, query_loader, db_loader, labels = get_nuswide_dataloaders(
            data_dir=args.data_dir,
            batch_size=4,
            max_train=100,
            max_query=50,
            max_db=100
        )
        
        # Test iteration
        for imgs, labels in train_loader:
            print(f"Images shape: {imgs.shape}")
            print(f"Labels shape: {labels.shape}")
            print(f"Sample label (multi-hot): {labels[0]}")
            print(f"Number of positive labels: {labels[0].sum().item()}")
            break
