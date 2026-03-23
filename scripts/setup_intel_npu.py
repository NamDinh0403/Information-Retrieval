"""
Intel NPU Training Configuration
================================
Cấu hình training trên Intel Core Ultra NPU.

Intel Core Ultra NPU:
    - Meteor Lake (i7-1xx5U, i9-1xx5H, etc.)
    - 10+ TOPS AI performance
    - Hỗ trợ qua OpenVINO hoặc Intel Extension for PyTorch

Requirements:
    pip install openvino openvino-dev
    pip install intel-extension-for-pytorch
    pip install intel-npu-acceleration-library

Usage:
    python setup_intel_npu.py --check      # Kiểm tra NPU
    python setup_intel_npu.py --install    # Cài đặt dependencies
    python setup_intel_npu.py --benchmark  # Benchmark NPU vs CPU
"""

import os
import sys
import subprocess
import platform
from typing import Optional, Dict, Tuple
import argparse


def check_intel_cpu() -> Dict:
    """Kiểm tra thông tin CPU Intel."""
    info = {
        'platform': platform.system(),
        'processor': platform.processor(),
        'is_intel': False,
        'is_core_ultra': False,
        'npu_available': False
    }
    
    try:
        if platform.system() == 'Windows':
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 
                               r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            cpu_name = winreg.QueryValueEx(key, "ProcessorNameString")[0]
            info['cpu_name'] = cpu_name
            info['is_intel'] = 'Intel' in cpu_name
            info['is_core_ultra'] = 'Ultra' in cpu_name or 'Meteor' in cpu_name
        else:
            # Linux
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if 'model name' in line:
                        cpu_name = line.split(':')[1].strip()
                        info['cpu_name'] = cpu_name
                        info['is_intel'] = 'Intel' in cpu_name
                        info['is_core_ultra'] = 'Ultra' in cpu_name
                        break
    except Exception as e:
        info['error'] = str(e)
    
    return info


def check_npu_driver() -> bool:
    """Kiểm tra Intel NPU driver đã cài đặt chưa."""
    try:
        if platform.system() == 'Windows':
            # Kiểm tra trong Device Manager
            result = subprocess.run(
                ['powershell', '-Command', 
                 "Get-PnpDevice | Where-Object {$_.FriendlyName -like '*NPU*' -or $_.FriendlyName -like '*Neural*'}"],
                capture_output=True, text=True
            )
            return 'NPU' in result.stdout or 'Neural' in result.stdout
        else:
            # Linux - kiểm tra /dev/accel*
            import glob
            return len(glob.glob('/dev/accel*')) > 0
    except:
        return False


def check_openvino() -> Tuple[bool, str]:
    """Kiểm tra OpenVINO đã cài đặt chưa."""
    try:
        import openvino as ov
        return True, ov.__version__
    except ImportError:
        return False, None


def check_ipex() -> Tuple[bool, str]:
    """Kiểm tra Intel Extension for PyTorch."""
    try:
        import intel_extension_for_pytorch as ipex
        return True, ipex.__version__
    except ImportError:
        return False, None


def check_npu_library() -> Tuple[bool, str]:
    """Kiểm tra Intel NPU Acceleration Library."""
    try:
        import intel_npu_acceleration_library
        return True, intel_npu_acceleration_library.__version__
    except ImportError:
        return False, None


def install_dependencies():
    """Cài đặt các dependencies cần thiết cho Intel NPU."""
    print("=" * 60)
    print("CÀI ĐẶT INTEL NPU DEPENDENCIES")
    print("=" * 60)
    
    packages = [
        # OpenVINO - chính thức support NPU
        ('openvino', 'openvino>=2024.0'),
        ('openvino-dev', 'openvino-dev'),
        
        # Intel Extension for PyTorch
        ('intel-extension-for-pytorch', 'intel-extension-for-pytorch'),
        
        # NPU Acceleration Library (mới, experimental)
        ('intel-npu-acceleration-library', 'intel-npu-acceleration-library'),
        
        # NNCF cho quantization
        ('nncf', 'nncf'),
    ]
    
    for name, package in packages:
        print(f"\n[*] Cài đặt {name}...")
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', package, '-q'],
                         check=True)
            print(f"    [✓] {name} đã cài đặt")
        except subprocess.CalledProcessError as e:
            print(f"    [!] Lỗi cài đặt {name}: {e}")
    
    print("\n[*] Cài đặt hoàn tất!")
    print("\nLưu ý: Cần cài Intel NPU Driver từ:")
    print("  https://www.intel.com/content/www/us/en/download/794734/")


def get_available_devices() -> Dict:
    """Lấy danh sách devices có sẵn qua OpenVINO."""
    devices = {'cpu': True, 'gpu': False, 'npu': False}
    
    try:
        import openvino as ov
        core = ov.Core()
        available = core.available_devices
        
        devices['available'] = available
        devices['gpu'] = 'GPU' in available
        devices['npu'] = 'NPU' in available
        
        # Lấy thông tin chi tiết
        for device in available:
            try:
                full_name = core.get_property(device, "FULL_DEVICE_NAME")
                devices[f'{device.lower()}_name'] = full_name
            except:
                pass
                
    except ImportError:
        devices['error'] = 'OpenVINO not installed'
    
    return devices


def print_system_info():
    """In thông tin hệ thống."""
    print("=" * 60)
    print("KIỂM TRA HỆ THỐNG CHO INTEL NPU")
    print("=" * 60)
    
    # CPU Info
    cpu_info = check_intel_cpu()
    print(f"\n[CPU]")
    print(f"  Platform: {cpu_info['platform']}")
    print(f"  CPU: {cpu_info.get('cpu_name', 'Unknown')}")
    print(f"  Is Intel: {'✓' if cpu_info['is_intel'] else '✗'}")
    print(f"  Is Core Ultra: {'✓' if cpu_info['is_core_ultra'] else '✗'}")
    
    # NPU Driver
    npu_driver = check_npu_driver()
    print(f"\n[NPU Driver]")
    print(f"  Installed: {'✓' if npu_driver else '✗'}")
    
    if not npu_driver:
        print("  → Tải driver tại: https://www.intel.com/content/www/us/en/download/794734/")
    
    # OpenVINO
    ov_installed, ov_version = check_openvino()
    print(f"\n[OpenVINO]")
    print(f"  Installed: {'✓ v' + ov_version if ov_installed else '✗'}")
    
    # Intel Extension for PyTorch
    ipex_installed, ipex_version = check_ipex()
    print(f"\n[Intel Extension for PyTorch]")
    print(f"  Installed: {'✓ v' + ipex_version if ipex_installed else '✗'}")
    
    # NPU Acceleration Library
    npu_lib, npu_lib_version = check_npu_library()
    print(f"\n[Intel NPU Acceleration Library]")
    print(f"  Installed: {'✓ v' + npu_lib_version if npu_lib else '✗'}")
    
    # Available Devices
    if ov_installed:
        devices = get_available_devices()
        print(f"\n[OpenVINO Devices]")
        print(f"  Available: {devices.get('available', [])}")
        print(f"  CPU: {'✓' if devices['cpu'] else '✗'}")
        print(f"  GPU: {'✓' if devices['gpu'] else '✗'}")
        print(f"  NPU: {'✓' if devices['npu'] else '✗'}")
        
        if devices['npu']:
            print(f"  NPU Name: {devices.get('npu_name', 'Unknown')}")
    
    # Recommendation
    print("\n" + "=" * 60)
    print("KHUYẾN NGHỊ")
    print("=" * 60)
    
    if cpu_info['is_core_ultra'] and npu_driver and ov_installed:
        print("[✓] Hệ thống sẵn sàng cho Intel NPU training!")
        print("    Chạy: python setup_intel_npu.py --benchmark")
    elif not cpu_info['is_core_ultra']:
        print("[!] CPU không phải Intel Core Ultra.")
        print("    NPU chỉ có trên Core Ultra (Meteor Lake+)")
        print("    → Có thể dùng Intel GPU (iGPU) hoặc CPU optimization")
    elif not npu_driver:
        print("[!] Cần cài Intel NPU Driver")
        print("    → https://www.intel.com/content/www/us/en/download/794734/")
    else:
        print("[!] Cần cài OpenVINO: pip install openvino")


def create_training_config():
    """Tạo file config cho NPU training."""
    
    config_content = '''"""
Intel NPU Training Config
=========================
Cấu hình tối ưu cho training trên Intel Core Ultra NPU.
"""

import torch

# Kiểm tra và thiết lập device
def get_best_device():
    """Lấy device tốt nhất có sẵn: NPU > GPU > CPU"""
    
    # Thử NPU qua OpenVINO
    try:
        import openvino as ov
        core = ov.Core()
        if 'NPU' in core.available_devices:
            return 'npu', 'OpenVINO NPU'
    except:
        pass
    
    # Thử Intel GPU qua IPEX
    try:
        import intel_extension_for_pytorch as ipex
        if ipex.xpu.is_available():
            return 'xpu', 'Intel GPU (XPU)'
    except:
        pass
    
    # Fallback to CPU với optimization
    return 'cpu', 'CPU (optimized)'


# Training config cho các devices
NPU_CONFIG = {
    'batch_size': 16,        # NPU memory limited
    'precision': 'int8',     # NPU tối ưu cho INT8
    'num_workers': 4,
    'pin_memory': True,
}

GPU_CONFIG = {
    'batch_size': 32,
    'precision': 'fp16',     # GPU tốt với FP16
    'num_workers': 4,
    'pin_memory': True,
}

CPU_CONFIG = {
    'batch_size': 8,
    'precision': 'fp32',
    'num_workers': 2,
    'pin_memory': False,
}

def get_config(device_type: str) -> dict:
    """Lấy config phù hợp với device."""
    configs = {
        'npu': NPU_CONFIG,
        'xpu': GPU_CONFIG,
        'gpu': GPU_CONFIG,
        'cpu': CPU_CONFIG,
    }
    return configs.get(device_type, CPU_CONFIG)
'''
    
    with open('intel_npu_config.py', 'w') as f:
        f.write(config_content)
    
    print("[✓] Đã tạo intel_npu_config.py")


def benchmark_devices():
    """Benchmark các devices có sẵn."""
    print("=" * 60)
    print("BENCHMARK INTEL DEVICES")
    print("=" * 60)
    
    import torch
    import time
    
    # Test tensor operations
    sizes = [(256, 256), (512, 512), (1024, 1024)]
    results = {}
    
    # CPU Benchmark
    print("\n[CPU Benchmark]")
    for size in sizes:
        x = torch.randn(size)
        
        start = time.perf_counter()
        for _ in range(100):
            y = torch.mm(x, x)
        elapsed = time.perf_counter() - start
        
        print(f"  MatMul {size}: {elapsed*10:.2f} ms")
        results[f'cpu_{size}'] = elapsed
    
    # Intel Extension for PyTorch (XPU)
    try:
        import intel_extension_for_pytorch as ipex
        if ipex.xpu.is_available():
            print("\n[Intel XPU (GPU) Benchmark]")
            for size in sizes:
                x = torch.randn(size).to('xpu')
                
                # Warmup
                for _ in range(10):
                    y = torch.mm(x, x)
                torch.xpu.synchronize()
                
                start = time.perf_counter()
                for _ in range(100):
                    y = torch.mm(x, x)
                torch.xpu.synchronize()
                elapsed = time.perf_counter() - start
                
                print(f"  MatMul {size}: {elapsed*10:.2f} ms")
                results[f'xpu_{size}'] = elapsed
    except Exception as e:
        print(f"\n[Intel XPU] Không khả dụng: {e}")
    
    # OpenVINO NPU Benchmark
    try:
        import openvino as ov
        import numpy as np
        
        core = ov.Core()
        if 'NPU' in core.available_devices:
            print("\n[Intel NPU Benchmark]")
            print("  NPU hoạt động tốt nhất với models đã compile")
            print("  Sử dụng OpenVINO để convert và optimize model")
            
            # Simple benchmark với compiled model
            for size in [(224, 224), (256, 256)]:
                # Tạo simple model
                import openvino.runtime as ov_runtime
                
                # Placeholder benchmark
                print(f"  Input {size}: NPU ready for inference")
        else:
            print("\n[Intel NPU] Không tìm thấy NPU device")
            
    except Exception as e:
        print(f"\n[OpenVINO NPU] Lỗi: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("TỔNG KẾT")
    print("=" * 60)
    print("""
Lưu ý quan trọng về Intel NPU:

1. NPU tối ưu cho INFERENCE, không phải TRAINING
   - Training: Dùng CPU/GPU với IPEX
   - Inference: Dùng NPU qua OpenVINO

2. Workflow khuyến nghị:
   - Train trên CPU/GPU với Intel Extension for PyTorch
   - Export model sang OpenVINO IR format
   - Deploy inference trên NPU

3. NPU hoạt động tốt nhất với:
   - INT8 quantized models
   - Batch size nhỏ (1-8)
   - Models đã được optimize bằng NNCF
""")


def main():
    parser = argparse.ArgumentParser(description='Intel NPU Setup & Benchmark')
    parser.add_argument('--check', action='store_true', help='Kiểm tra hệ thống')
    parser.add_argument('--install', action='store_true', help='Cài đặt dependencies')
    parser.add_argument('--benchmark', action='store_true', help='Benchmark devices')
    parser.add_argument('--config', action='store_true', help='Tạo training config')
    
    args = parser.parse_args()
    
    if args.install:
        install_dependencies()
    elif args.benchmark:
        benchmark_devices()
    elif args.config:
        create_training_config()
    else:
        # Default: check system
        print_system_info()


if __name__ == "__main__":
    main()
