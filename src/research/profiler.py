"""
Latency Profiler Module
=======================
Công cụ đo lường độ trễ và hiệu năng cho Vision Transformer.

Implementations:
    - LatencyProfiler: Đo latency thực tế trên CPU/GPU
    - FLOPsCalculator: Tính toán FLOPs và memory footprint

References:
    - lm-Meter: Lightweight latency profiler cho mobile devices
    - nn-Meter: Kernel fusion detection cho accurate prediction
"""

import time
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Callable
import numpy as np
from contextlib import contextmanager


class LatencyProfiler:
    """
    Công cụ đo lường độ trễ thực tế.
    
    Đo latency qua nhiều runs để có độ chính xác cao,
    bao gồm warmup phase để ổn định GPU.
    
    Args:
        model: Model cần profile
        device: 'cpu' hoặc 'cuda'
        warmup_runs: Số runs warmup
        num_runs: Số runs để tính trung bình
        
    Example:
        >>> profiler = LatencyProfiler(model, device='cuda')
        >>> latency = profiler.measure_latency(input_tensor)
        >>> print(f"Latency: {latency:.2f} ms")
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        warmup_runs: int = 10,
        num_runs: int = 100
    ):
        self.model = model
        self.device = device
        self.warmup_runs = warmup_runs
        self.num_runs = num_runs
        
        self.model.to(device)
        self.model.eval()
    
    @contextmanager
    def _timer(self):
        """Context manager cho accurate timing."""
        if self.device == 'cuda':
            torch.cuda.synchronize()
            
        start = time.perf_counter()
        yield
        
        if self.device == 'cuda':
            torch.cuda.synchronize()
            
        self._elapsed = (time.perf_counter() - start) * 1000  # ms
    
    def measure_latency(
        self, 
        input_tensor: torch.Tensor,
        return_std: bool = False
    ) -> float:
        """
        Đo latency của một forward pass.
        
        Args:
            input_tensor: Input tensor
            return_std: Có trả về standard deviation không
            
        Returns:
            latency: Latency trung bình (ms)
        """
        input_tensor = input_tensor.to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(self.warmup_runs):
                _ = self.model(input_tensor)
        
        # Profiling
        latencies = []
        
        with torch.no_grad():
            for _ in range(self.num_runs):
                with self._timer():
                    _ = self.model(input_tensor)
                latencies.append(self._elapsed)
        
        avg_latency = np.mean(latencies)
        
        if return_std:
            return avg_latency, np.std(latencies)
        return avg_latency
    
    def measure_throughput(
        self,
        input_tensor: torch.Tensor,
        batch_sizes: List[int] = [1, 2, 4, 8, 16]
    ) -> Dict[int, float]:
        """
        Đo throughput (images/second) với các batch sizes khác nhau.
        """
        results = {}
        
        for bs in batch_sizes:
            try:
                # Tạo batch
                batch = input_tensor.repeat(bs, 1, 1, 1)
                
                # Đo latency
                latency = self.measure_latency(batch)
                
                # Throughput = batch_size / latency(s)
                throughput = (bs / latency) * 1000
                results[bs] = throughput
                
            except RuntimeError as e:
                # OOM hoặc lỗi khác
                print(f"[Profiler] Batch size {bs} failed: {e}")
                break
        
        return results
    
    def profile_token_latency_relationship(
        self,
        base_image_size: int = 224,
        patch_size: int = 14
    ) -> Dict:
        """
        Nghiên cứu Tuần 2: Profile mối quan hệ phi tuyến giữa số token và latency.
        
        Returns:
            Dict với token counts và latencies tương ứng
        """
        print("[Profiler] Đang phân tích Token-Latency Relationship...")
        
        # Các resolution khác nhau -> số tokens khác nhau
        resolutions = [112, 168, 224, 280, 336, 392]
        
        results = {
            'resolutions': [],
            'token_counts': [],
            'latencies': [],
            'latencies_std': []
        }
        
        for res in resolutions:
            # Số tokens = (H/patch_size) * (W/patch_size)
            num_tokens = (res // patch_size) ** 2
            
            try:
                input_tensor = torch.randn(1, 3, res, res)
                latency, std = self.measure_latency(input_tensor, return_std=True)
                
                results['resolutions'].append(res)
                results['token_counts'].append(num_tokens)
                results['latencies'].append(latency)
                results['latencies_std'].append(std)
                
                print(f"  Resolution {res}x{res} ({num_tokens} tokens): {latency:.2f} ± {std:.2f} ms")
                
            except Exception as e:
                print(f"  Resolution {res}x{res} failed: {e}")
        
        # Phân tích phi tuyến
        if len(results['token_counts']) >= 3:
            tokens = np.array(results['token_counts'])
            latencies = np.array(results['latencies'])
            
            # Fit linear
            linear_coef = np.polyfit(tokens, latencies, 1)
            linear_pred = np.polyval(linear_coef, tokens)
            linear_error = np.mean((latencies - linear_pred) ** 2)
            
            # Fit quadratic
            quad_coef = np.polyfit(tokens, latencies, 2)
            quad_pred = np.polyval(quad_coef, tokens)
            quad_error = np.mean((latencies - quad_pred) ** 2)
            
            results['analysis'] = {
                'linear_mse': linear_error,
                'quadratic_mse': quad_error,
                'is_nonlinear': quad_error < linear_error * 0.8,  # Quadratic fits better
                'linear_coefficient': linear_coef[0],
                'quadratic_coefficient': quad_coef[0]
            }
            
            print(f"\n[Analysis] Relationship type: {'Non-linear' if results['analysis']['is_nonlinear'] else 'Linear'}")
            print(f"           Linear MSE: {linear_error:.4f}, Quadratic MSE: {quad_error:.4f}")
        
        return results
    
    def profile_layer_latency(self) -> Dict[str, float]:
        """
        Profile latency từng layer (simplified).
        """
        # Đây là simplified version, full implementation cần hooks
        results = {}
        
        # Đếm layers
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.MultiheadAttention)):
                results[name] = 0.0  # Placeholder
        
        return results


class FLOPsCalculator:
    """
    Tính toán FLOPs và memory footprint cho ViT.
    """
    
    @staticmethod
    def calculate_vit_flops(
        image_size: int = 224,
        patch_size: int = 14,
        embed_dim: int = 384,
        num_heads: int = 6,
        num_layers: int = 12,
        mlp_ratio: float = 4.0
    ) -> Dict[str, int]:
        """
        Tính FLOPs cho Vision Transformer.
        
        Returns:
            Dict với breakdown của FLOPs theo component
        """
        # Số patches
        num_patches = (image_size // patch_size) ** 2
        N = num_patches + 1  # +1 cho CLS token
        D = embed_dim
        
        # Patch Embedding
        patch_embed_flops = (patch_size ** 2 * 3) * D * num_patches
        
        # Self-Attention per layer
        # Q, K, V projections: 3 * N * D * D
        qkv_flops = 3 * N * D * D
        # Attention scores: N * N * D
        attn_scores_flops = N * N * D
        # Attention @ V: N * N * D
        attn_output_flops = N * N * D
        # Output projection: N * D * D
        output_proj_flops = N * D * D
        
        attention_flops_per_layer = qkv_flops + attn_scores_flops + attn_output_flops + output_proj_flops
        
        # MLP per layer: 2 * N * D * (D * mlp_ratio)
        mlp_flops_per_layer = 2 * N * D * int(D * mlp_ratio)
        
        # Total
        total_attention = attention_flops_per_layer * num_layers
        total_mlp = mlp_flops_per_layer * num_layers
        total_flops = patch_embed_flops + total_attention + total_mlp
        
        return {
            'patch_embed': patch_embed_flops,
            'attention_per_layer': attention_flops_per_layer,
            'mlp_per_layer': mlp_flops_per_layer,
            'total_attention': total_attention,
            'total_mlp': total_mlp,
            'total': total_flops,
            'total_gflops': total_flops / 1e9
        }
    
    @staticmethod
    def calculate_memory_footprint(
        model: nn.Module,
        input_size: Tuple[int, int, int, int] = (1, 3, 224, 224)
    ) -> Dict[str, float]:
        """
        Ước tính memory footprint.
        """
        # Parameter memory
        param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
        
        # Gradient memory (same as params during training)
        grad_memory = param_memory
        
        # Activation memory (rough estimate)
        # Cần forward pass với hooks để tính chính xác
        
        return {
            'parameters_mb': param_memory / (1024 ** 2),
            'gradients_mb': grad_memory / (1024 ** 2),
            'total_training_mb': (param_memory + grad_memory) / (1024 ** 2)
        }
    
    @staticmethod
    def estimate_pruning_savings(
        original_tokens: int,
        kept_tokens: int,
        embed_dim: int = 384,
        num_layers: int = 12
    ) -> Dict[str, float]:
        """
        Ước tính savings khi pruning tokens.
        """
        ratio = kept_tokens / original_tokens
        
        # Attention scales O(N^2)
        attention_ratio = ratio ** 2
        
        # MLP scales O(N)
        mlp_ratio = ratio
        
        # Weighted average (attention dominant)
        overall_ratio = 0.6 * attention_ratio + 0.4 * mlp_ratio
        
        return {
            'token_reduction': 1 - ratio,
            'attention_savings': 1 - attention_ratio,
            'mlp_savings': 1 - mlp_ratio,
            'overall_flops_reduction': 1 - overall_ratio
        }


def profile_model_comprehensive(
    model: nn.Module,
    device: str = 'cpu',
    image_size: int = 224
) -> Dict:
    """
    Comprehensive profiling cho một model.
    """
    profiler = LatencyProfiler(model, device)
    
    input_tensor = torch.randn(1, 3, image_size, image_size)
    
    # Latency
    latency, std = profiler.measure_latency(input_tensor, return_std=True)
    
    # Throughput
    throughput = profiler.measure_throughput(input_tensor, batch_sizes=[1, 4, 8])
    
    # Memory
    memory = FLOPsCalculator.calculate_memory_footprint(model)
    
    return {
        'latency_ms': latency,
        'latency_std_ms': std,
        'throughput_imgs_per_sec': throughput,
        'memory': memory
    }


if __name__ == "__main__":
    import timm
    
    print("Testing Latency Profiler Module...")
    
    # Create simple model for testing
    model = timm.create_model(
        'vit_small_patch14_dinov2.lvd142m',
        pretrained=False,
        num_classes=0,
        dynamic_img_size=True
    )
    
    # Test profiler
    profiler = LatencyProfiler(model, device='cpu', warmup_runs=3, num_runs=10)
    
    input_tensor = torch.randn(1, 3, 224, 224)
    latency = profiler.measure_latency(input_tensor)
    print(f"\nLatency: {latency:.2f} ms")
    
    # Test FLOPs calculation
    flops = FLOPsCalculator.calculate_vit_flops()
    print(f"\nFLOPs Breakdown:")
    print(f"  Total: {flops['total_gflops']:.2f} GFLOPs")
    print(f"  Attention: {flops['total_attention'] / 1e9:.2f} GFLOPs")
    print(f"  MLP: {flops['total_mlp'] / 1e9:.2f} GFLOPs")
    
    # Test pruning savings
    savings = FLOPsCalculator.estimate_pruning_savings(196, 137)  # 70% keep ratio
    print(f"\nPruning Savings (70% keep ratio):")
    print(f"  FLOPs reduction: {savings['overall_flops_reduction']:.1%}")
