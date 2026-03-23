import torch
import torch.nn as nn
import timm
import numpy as np

def load_weights_from_npz(model, npz_path: str, prefix: str = ''):
    """
    Load weights từ file .npz (Google ViT JAX format) sang PyTorch model.
    
    Args:
        model: timm ViT model
        npz_path: đường dẫn tới file .npz
        prefix: prefix trong npz keys (thường là '' hoặc 'Transformer/')
    """
    print(f"[*] Loading weights from {npz_path}...")
    weights = np.load(npz_path)
    
    # Debug: in ra các keys
    # print("NPZ keys:", list(weights.keys())[:10])
    
    with torch.no_grad():
        # Patch embedding
        if 'embedding/kernel' in weights:
            # Google format: (H, W, C_in, C_out) -> PyTorch: (C_out, C_in, H, W)
            kernel = weights['embedding/kernel']
            kernel = np.transpose(kernel, (3, 2, 0, 1))
            model.patch_embed.proj.weight.copy_(torch.from_numpy(kernel))
        if 'embedding/bias' in weights:
            model.patch_embed.proj.bias.copy_(torch.from_numpy(weights['embedding/bias']))
        
        # Class token
        if 'cls' in weights:
            cls_token = torch.from_numpy(weights['cls'])
            # Đảm bảo shape là [1, 1, embed_dim]
            if cls_token.dim() == 1:
                cls_token = cls_token.unsqueeze(0).unsqueeze(0)
            elif cls_token.dim() == 2:
                cls_token = cls_token.unsqueeze(0)
            model.cls_token.copy_(cls_token)
        
        # Position embedding
        if 'Transformer/posembed_input/pos_embedding' in weights:
            posemb = weights['Transformer/posembed_input/pos_embedding']
            # Resize if needed
            if posemb.shape[1] != model.pos_embed.shape[1]:
                print(f"  [!] Position embedding size mismatch: {posemb.shape[1]} vs {model.pos_embed.shape[1]}")
                # Simple: chỉ copy phần đầu hoặc pad
                min_len = min(posemb.shape[1], model.pos_embed.shape[1])
                model.pos_embed.data[:, :min_len] = torch.from_numpy(posemb[:, :min_len])
            else:
                model.pos_embed.copy_(torch.from_numpy(posemb))
        
        # Transformer blocks
        for i, block in enumerate(model.blocks):
            block_prefix = f'Transformer/encoderblock_{i}/'
            
            # LayerNorm 1
            if f'{block_prefix}LayerNorm_0/scale' in weights:
                block.norm1.weight.copy_(torch.from_numpy(weights[f'{block_prefix}LayerNorm_0/scale']))
                block.norm1.bias.copy_(torch.from_numpy(weights[f'{block_prefix}LayerNorm_0/bias']))
            
            # Attention
            if f'{block_prefix}MultiHeadDotProductAttention_1/query/kernel' in weights:
                q_w = weights[f'{block_prefix}MultiHeadDotProductAttention_1/query/kernel']
                k_w = weights[f'{block_prefix}MultiHeadDotProductAttention_1/key/kernel']
                v_w = weights[f'{block_prefix}MultiHeadDotProductAttention_1/value/kernel']
                
                # Reshape và concat QKV
                q_w = q_w.reshape(q_w.shape[0], -1)
                k_w = k_w.reshape(k_w.shape[0], -1)
                v_w = v_w.reshape(v_w.shape[0], -1)
                qkv_w = np.concatenate([q_w, k_w, v_w], axis=1)
                block.attn.qkv.weight.copy_(torch.from_numpy(qkv_w.T))
                
                q_b = weights[f'{block_prefix}MultiHeadDotProductAttention_1/query/bias'].flatten()
                k_b = weights[f'{block_prefix}MultiHeadDotProductAttention_1/key/bias'].flatten()
                v_b = weights[f'{block_prefix}MultiHeadDotProductAttention_1/value/bias'].flatten()
                qkv_b = np.concatenate([q_b, k_b, v_b])
                block.attn.qkv.bias.copy_(torch.from_numpy(qkv_b))
            
            # Attention output projection
            if f'{block_prefix}MultiHeadDotProductAttention_1/out/kernel' in weights:
                out_w = weights[f'{block_prefix}MultiHeadDotProductAttention_1/out/kernel']
                out_w = out_w.reshape(-1, out_w.shape[-1])
                block.attn.proj.weight.copy_(torch.from_numpy(out_w.T))
                block.attn.proj.bias.copy_(torch.from_numpy(
                    weights[f'{block_prefix}MultiHeadDotProductAttention_1/out/bias']))
            
            # LayerNorm 2
            if f'{block_prefix}LayerNorm_2/scale' in weights:
                block.norm2.weight.copy_(torch.from_numpy(weights[f'{block_prefix}LayerNorm_2/scale']))
                block.norm2.bias.copy_(torch.from_numpy(weights[f'{block_prefix}LayerNorm_2/bias']))
            
            # MLP
            if f'{block_prefix}MlpBlock_3/Dense_0/kernel' in weights:
                block.mlp.fc1.weight.copy_(torch.from_numpy(
                    weights[f'{block_prefix}MlpBlock_3/Dense_0/kernel'].T))
                block.mlp.fc1.bias.copy_(torch.from_numpy(
                    weights[f'{block_prefix}MlpBlock_3/Dense_0/bias']))
                block.mlp.fc2.weight.copy_(torch.from_numpy(
                    weights[f'{block_prefix}MlpBlock_3/Dense_1/kernel'].T))
                block.mlp.fc2.bias.copy_(torch.from_numpy(
                    weights[f'{block_prefix}MlpBlock_3/Dense_1/bias']))
        
        # Final LayerNorm
        if 'Transformer/encoder_norm/scale' in weights:
            model.norm.weight.copy_(torch.from_numpy(weights['Transformer/encoder_norm/scale']))
            model.norm.bias.copy_(torch.from_numpy(weights['Transformer/encoder_norm/bias']))
    
    print(f"[✓] Loaded weights from {npz_path}")
    return model


class ViT_Hashing(nn.Module):
    def __init__(self, model_name='vit_base_patch32_224', pretrained=True, hash_bit=32, 
                 weights_path: str = None):
        super(ViT_Hashing, self).__init__()
        # Load ViT backbone (không load pretrained nếu có weights_path)
        load_pretrained = pretrained and weights_path is None
        self.backbone = timm.create_model(model_name, pretrained=load_pretrained, num_classes=0)
        
        # Load weights từ file .npz nếu có
        if weights_path is not None:
            load_weights_from_npz(self.backbone, weights_path)
        
        embed_dim = self.backbone.num_features
        
        # VTS Hashing Head (Dropout-Dense-ReLU-Dense)
        self.hashing_head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(embed_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, hash_bit),
            nn.Tanh() # Use tanh for training to get continuous values in [-1, 1], sign during inference
        )

    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        # Generate pseudo-binary hash codes
        hash_codes = self.hashing_head(features)
        return hash_codes, features

if __name__ == "__main__":
    model = ViT_Hashing(hash_bit=32)
    sample_input = torch.randn(2, 3, 224, 224)
    hash_out, feat_out = model(sample_input)
    print("Hash Output Shape:", hash_out.shape)
    print("Feature Output Shape:", feat_out.shape)
