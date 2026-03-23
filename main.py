import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from src.model import ViT_Hashing
from src.dataset import get_cifar10_dataloaders
from src.loss import CSQLoss
from src.evaluate import evaluate

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Config
    EPOCHS = 10
    HASH_BIT = 32
    
    # 1. Data
    train_loader, query_loader, db_loader, classes = get_cifar10_dataloaders()
    
    # 2. Model (Downloads pre-trained weights automatically via timm)
    print(f"Initializing ViT_Hashing model with {HASH_BIT}-bit length...")
    model = ViT_Hashing(model_name='vit_base_patch16_224', pretrained=True, hash_bit=HASH_BIT)
    model.to(device)
    
    # 3. Loss & Optimizer (Spec: learning rate 3e-5 for backbone, 1e-4 for hashing head)
    criterion = CSQLoss(hash_bit=HASH_BIT, num_classes=len(classes))
    
    optimizer = optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': 3e-5},
        {'params': model.hashing_head.parameters(), 'lr': 1e-4}
    ])
    
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # 4. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            hash_codes, _ = model(imgs) # outputs continuous values via Tanh
            
            loss = criterion(hash_codes, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        scheduler.step()
        print(f"Epoch {epoch+1} - Average Loss: {total_loss / len(train_loader):.4f}")
        
        # Evaluate every few epochs
        if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
            evaluate(model, query_loader, db_loader, device)

    # Save model
    torch.save(model.state_dict(), f"vit_hashing_{HASH_BIT}bit.pth")
    print(f"Model saved to vit_hashing_{HASH_BIT}bit.pth")

if __name__ == "__main__":
    train()
