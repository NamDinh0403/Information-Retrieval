"""
Dataset Module for Research
===========================
Data loaders cho các datasets chuyên biệt trong nghiên cứu.

Datasets:
    - NWPU-RESISC45: Remote Sensing (45 classes, 700 images/class)
    - ChestX-ray8: Medical Imaging (8 pathology labels)
    - CIFAR-10: Small-scale benchmark
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from PIL import Image
from typing import Tuple, List, Optional, Dict
import numpy as np


class NWPUDataset(Dataset):
    """
    NWPU-RESISC45 Dataset cho Remote Sensing Image Retrieval.
    
    Structure:
        NWPU-RESISC45/
            airplane/
                airplane_001.jpg
                ...
            airport/
                airport_001.jpg
                ...
            ...
    
    45 classes, mỗi class 700 images (256x256).
    """
    
    CLASSES = [
        'airplane', 'airport', 'baseball_diamond', 'basketball_court', 'beach',
        'bridge', 'chaparral', 'church', 'circular_farmland', 'cloud',
        'commercial_area', 'dense_residential', 'desert', 'forest', 'freeway',
        'golf_course', 'ground_track_field', 'harbor', 'industrial_area', 'intersection',
        'island', 'lake', 'meadow', 'medium_residential', 'mobile_home_park',
        'mountain', 'overpass', 'palace', 'parking_lot', 'railway',
        'railway_station', 'rectangular_farmland', 'river', 'roundabout', 'runway',
        'sea_ice', 'ship', 'snowberg', 'sparse_residential', 'stadium',
        'storage_tank', 'tennis_court', 'terrace', 'thermal_power_station', 'wetland'
    ]
    
    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform: Optional[transforms.Compose] = None,
        train_ratio: float = 0.7
    ):
        """
        Args:
            root: Đường dẫn đến thư mục NWPU-RESISC45
            split: 'train', 'query', hoặc 'database'
            transform: Augmentation transforms
            train_ratio: Tỷ lệ dữ liệu train
        """
        self.root = root
        self.split = split
        self.transform = transform or self._get_default_transform()
        
        # Load image paths và labels
        self.samples = []
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.CLASSES)}
        
        self._load_samples(train_ratio)
    
    def _get_default_transform(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _load_samples(self, train_ratio: float):
        """Load samples và split theo ratio."""
        all_samples = []
        
        for class_name in self.CLASSES:
            class_dir = os.path.join(self.root, class_name)
            if not os.path.exists(class_dir):
                continue
            
            images = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
            for img_name in images:
                img_path = os.path.join(class_dir, img_name)
                label = self.class_to_idx[class_name]
                all_samples.append((img_path, label))
        
        # Shuffle và split
        np.random.seed(42)
        np.random.shuffle(all_samples)
        
        n_train = int(len(all_samples) * train_ratio)
        n_query = int(len(all_samples) * 0.1)  # 10% cho query
        
        if self.split == 'train':
            self.samples = all_samples[:n_train]
        elif self.split == 'query':
            self.samples = all_samples[n_train:n_train + n_query]
        else:  # database
            self.samples = all_samples[n_train + n_query:]
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class ChestXrayDataset(Dataset):
    """
    ChestX-ray8 Dataset cho Medical Image Analysis.
    
    Multi-label classification với 8 pathologies:
    - Atelectasis, Cardiomegaly, Effusion, Infiltration,
    - Mass, Nodule, Pneumonia, Pneumothorax
    """
    
    PATHOLOGIES = [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
        'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax'
    ]
    
    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform: Optional[transforms.Compose] = None
    ):
        self.root = root
        self.split = split
        self.transform = transform or self._get_default_transform()
        
        self.samples = []
        self._load_samples()
    
    def _get_default_transform(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])  # Grayscale
        ])
    
    def _load_samples(self):
        """
        Load từ CSV metadata.
        Giả định có file `{split}_labels.csv` với format:
        image_name,Atelectasis,Cardiomegaly,...
        """
        csv_path = os.path.join(self.root, f'{self.split}_labels.csv')
        
        if not os.path.exists(csv_path):
            print(f"[Warning] {csv_path} not found. Using dummy data.")
            return
        
        import csv
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_path = os.path.join(self.root, 'images', row['image_name'])
                labels = [int(row[p]) for p in self.PATHOLOGIES]
                self.samples.append((img_path, labels))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, labels = self.samples[idx]
        
        if os.path.exists(img_path):
            image = Image.open(img_path).convert('RGB')
        else:
            # Dummy image nếu không tìm thấy
            image = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            image = self.transform(image)
        
        labels = torch.tensor(labels, dtype=torch.float32)
        return image, labels


def get_nwpu_dataloader(
    data_dir: str = './data/NWPU-RESISC45',
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Tạo data loaders cho NWPU-RESISC45.
    
    Returns:
        train_loader, query_loader, database_loader, class_names
    """
    # Check if directory exists
    if not os.path.exists(data_dir):
        print(f"[Warning] {data_dir} not found. Creating dummy data...")
        return _create_dummy_loaders(batch_size, num_classes=45)
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = NWPUDataset(data_dir, split='train', transform=train_transform)
    query_dataset = NWPUDataset(data_dir, split='query', transform=test_transform)
    db_dataset = NWPUDataset(data_dir, split='database', transform=test_transform)
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    query_loader = DataLoader(
        query_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    db_loader = DataLoader(
        db_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, query_loader, db_loader, NWPUDataset.CLASSES


def get_chestxray_dataloader(
    data_dir: str = './data/ChestXray8',
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Tạo data loaders cho ChestX-ray8.
    
    Returns:
        train_loader, query_loader, database_loader, pathology_names
    """
    if not os.path.exists(data_dir):
        print(f"[Warning] {data_dir} not found. Creating dummy data...")
        return _create_dummy_loaders(batch_size, num_classes=8)
    
    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = ChestXrayDataset(data_dir, split='train', transform=transform)
    query_dataset = ChestXrayDataset(data_dir, split='test', transform=transform)
    db_dataset = ChestXrayDataset(data_dir, split='database', transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    query_loader = DataLoader(query_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    db_loader = DataLoader(db_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, query_loader, db_loader, ChestXrayDataset.PATHOLOGIES


def _create_dummy_loaders(batch_size: int, num_classes: int):
    """Tạo dummy loaders cho testing khi không có data thực."""
    
    class DummyDataset(Dataset):
        def __init__(self, size: int = 100):
            self.size = size
            self.num_classes = num_classes
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            image = torch.randn(3, 224, 224)
            label = idx % self.num_classes
            return image, label
    
    train_loader = DataLoader(DummyDataset(500), batch_size=batch_size, shuffle=True)
    query_loader = DataLoader(DummyDataset(100), batch_size=batch_size, shuffle=False)
    db_loader = DataLoader(DummyDataset(400), batch_size=batch_size, shuffle=False)
    
    class_names = [f'class_{i}' for i in range(num_classes)]
    
    return train_loader, query_loader, db_loader, class_names


if __name__ == "__main__":
    print("Testing Dataset Module...")
    
    # Test NWPU loader (will use dummy if not available)
    train_loader, query_loader, db_loader, classes = get_nwpu_dataloader()
    
    print(f"\nNWPU-RESISC45 (or dummy):")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Query batches: {len(query_loader)}")
    print(f"  Database batches: {len(db_loader)}")
    print(f"  Number of classes: {len(classes)}")
    
    # Test one batch
    for images, labels in train_loader:
        print(f"  Batch shape: {images.shape}")
        print(f"  Labels shape: {labels.shape}")
        break
