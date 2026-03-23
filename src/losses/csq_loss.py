import torch
import torch.nn as nn
import torch.nn.functional as F

class CSQLoss(nn.Module):
    """
    Central Similarity Quantization (CSQ) Loss
    As per spec: Central Similarity Quantization (CSQ) + Quantization Loss
    Computes BCE with quantization error minimization.
    """
    def __init__(self, hash_bit, num_classes, lambda_q=0.0001):
        super(CSQLoss, self).__init__()
        self.hash_bit = hash_bit
        self.num_classes = num_classes
        self.lambda_q = lambda_q
        
        # Initialize Hadamard matrix for class centers (pseudo random orthongonal centers)
        self.hash_targets = self._get_hash_targets(num_classes, hash_bit)

    def _get_hash_targets(self, num_classes, hash_bit):
        # A simple generation of target centers for CSQ
        # In a full implementation, Hadamard matrix is often used.
        # Here we use random binary targets for demonstration
        targets = torch.sign(torch.randn(num_classes, hash_bit))
        return targets

    def forward(self, hash_outputs, labels):
        device = hash_outputs.device
        self.hash_targets = self.hash_targets.to(device)
        
        # Get target hash codes for the current batch
        target_centers = self.hash_targets[labels]
        
        # Center Loss (similarity)
        # Using MSE between continuous output and discrete targets as a proxy for similarity BCE
        center_loss = F.mse_loss(hash_outputs, target_centers)
        
        # Quantization Loss: minimizes error between continuous outputs and discrete binary codes
        # sum | |h| - 1 |
        q_loss = torch.mean((torch.abs(hash_outputs) - 1.0) ** 2)
        
        total_loss = center_loss + self.lambda_q * q_loss
        return total_loss
