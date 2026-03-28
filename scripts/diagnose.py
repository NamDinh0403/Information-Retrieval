"""Quick diagnosis: check hash code quality of saved checkpoint."""
import torch, numpy as np, sys, importlib.util
sys.path.insert(0, '.')

spec = importlib.util.spec_from_file_location('vit_hashing', 'src/models/vit_hashing.py')
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
ViT_Hashing = mod.ViT_Hashing

from PIL import Image
import torchvision.transforms as T
from pathlib import Path

ck = torch.load('checkpoints/best_model.pth', map_location='cpu', weights_only=False)
model = ViT_Hashing(model_name='vit_base_patch32_224', pretrained=False, hash_bit=64)
model.load_state_dict(ck['model_state_dict'])
model.eval()

tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
test_dir = Path('./data/archive/Dataset/test/test')

classes = ['airplane', 'airport', 'beach', 'forest', 'mountain', 'runway']
class_codes = {}
for cls in classes:
    cls_dir = test_dir / cls
    if not cls_dir.exists():
        continue
    imgs = list(cls_dir.glob('*.jpg'))[:10]
    codes = []
    for p in imgs:
        t = tf(Image.open(p).convert('RGB')).unsqueeze(0)
        with torch.no_grad():
            h, _ = model(t)
            codes.append(torch.sign(h).squeeze(0).numpy())
    class_codes[cls] = np.array(codes)

all_codes = np.concatenate(list(class_codes.values()))
print(f'Bit balance (ideal=1.0): {1 - np.abs(all_codes.mean(axis=0)).mean():.4f}')
print(f'Quantization (ideal=1.0): {(np.abs(all_codes) == 1).mean():.4f}')
print()
print('=== Intra-class distance (LOW = good, ideal < 10) ===')
for cls, codes in class_codes.items():
    dists = [int((codes[i]!=codes[j]).sum())
             for i in range(len(codes)) for j in range(i+1, len(codes))]
    if dists:
        print(f'  {cls:20s}: mean={np.mean(dists):.1f}/64  min={min(dists)}  max={max(dists)}')

print()
print('=== Inter-class distance (HIGH = good, ideal > 30) ===')
cl = list(class_codes.keys())
for i in range(len(cl)):
    for j in range(i+1, len(cl)):
        dists = [int((a!=b).sum()) for a in class_codes[cl[i]] for b in class_codes[cl[j]]]
        overlap = "PROBLEM" if np.mean(dists) < 20 else ""
        print(f'  {cl[i]:12s} vs {cl[j]:12s}: mean={np.mean(dists):.1f}/64  {overlap}')

print()
print(f'Training mAP (val-to-val): {ck["metrics"]["mAP"]:.4f}')
print(f'Epoch: {ck["epoch"]}')
