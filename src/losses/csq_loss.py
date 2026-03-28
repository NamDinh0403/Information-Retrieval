import torch
import torch.nn as nn
import torch.nn.functional as F


class CSQLoss(nn.Module):
    """
    Central Similarity Quantization (CSQ) Loss
    As per spec: Central Similarity Quantization (CSQ) + Quantization Loss

    Loss = Center Loss + lambda_q * Quantization Loss + lambda_b * Balance Loss

    Balance Loss forces each bit to be +1 and -1 with equal probability across
    the batch, fixing the "bit collapse" problem (bit_balance << 1.0).
    """

    def __init__(self, hash_bit, num_classes, lambda_q=0.001, lambda_b=0.1):
        super(CSQLoss, self).__init__()
        self.hash_bit = hash_bit
        self.num_classes = num_classes
        self.lambda_q = lambda_q
        self.lambda_b = lambda_b  # balance loss weight

        # Fixed-seed hash centers — consistent across all runs & checkpoints
        hash_targets = self._get_hash_targets(num_classes, hash_bit)
        self.register_buffer('hash_targets', hash_targets)

    def _get_hash_targets(self, num_classes, hash_bit):
        """
        Generate orthogonal-ish hash centers using fixed seed.
        Each class gets a unique binary code in {-1,+1}^hash_bit.
        """
        generator = torch.Generator()
        generator.manual_seed(42)
        targets = torch.sign(torch.randn(num_classes, hash_bit, generator=generator))
        targets[targets == 0] = 1.0
        return targets

    def forward(self, hash_outputs, labels):
        """
        Args:
            hash_outputs: [B, hash_bit] continuous values from Tanh, range (-1,1)
            labels:       [B] class indices
        """
        # 1. Center Loss — pull each sample toward its class hash center
        target_centers = self.hash_targets[labels]
        center_loss = F.mse_loss(hash_outputs, target_centers)

        # 2. Quantization Loss — push outputs toward binary {-1, +1}
        q_loss = torch.mean((torch.abs(hash_outputs) - 1.0) ** 2)

        # 3. Balance Loss — each bit should be +1 and -1 equally across batch
        #    bit_mean per bit should be ~0; penalize deviation
        bit_mean = hash_outputs.mean(dim=0)          # [hash_bit]
        balance_loss = torch.mean(bit_mean ** 2)

        total_loss = center_loss + self.lambda_q * q_loss + self.lambda_b * balance_loss
        return total_loss
