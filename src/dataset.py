import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

def get_cifar10_dataloaders(data_dir='./data', batch_size=64, img_size=224, num_train=5000, num_query=1000):
    """
    Downloads and prepares CIFAR-10 dataset according to specification:
    - Train on 5,000
    - Query on 1,000
    """
    print("Downloading/Loading CIFAR-10 Dataset...")
    
    # ViT requires 224x224 input usually
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # CIFAR-10 Loading (download=False assumes you ran download_dataset.py)
    train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=False, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=False, transform=transform)
    
    # Sample subset as per spec
    np.random.seed(42)
    train_indices = np.random.choice(len(train_dataset), num_train, replace=False)
    query_indices = np.random.choice(len(test_dataset), num_query, replace=False)
    
    train_subset = Subset(train_dataset, train_indices)
    query_subset = Subset(test_dataset, query_indices)
    
    # Use remaining test data as database for retrieval
    db_indices = list(set(range(len(test_dataset))) - set(query_indices))
    db_subset = Subset(test_dataset, db_indices)
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    query_loader = DataLoader(query_subset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    db_loader = DataLoader(db_subset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    print(f"Data ready - Train: {len(train_subset)}, Query: {len(query_subset)}, Database: {len(db_subset)}")
    return train_loader, query_loader, db_loader, train_dataset.classes

if __name__ == "__main__":
    get_cifar10_dataloaders()