import torchvision
import os
import urllib.request
import tarfile
import zipfile

def setup_local_dataset_dir(data_dir='./data'):
    """
    Tạo thư mục local để lưu trữ dataset, tránh tải lại nhiều lần.
    """
    os.makedirs(data_dir, exist_ok=True)
    print(f"[*] Thư mục dữ liệu local: {os.path.abspath(data_dir)}")
    return data_dir

def download_cifar10(data_dir='./data'):
    data_dir = setup_local_dataset_dir(data_dir)
    print(f"Downloading CIFAR-10 to {data_dir}...")
    torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True)
    torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True)
    print("CIFAR-10 dataset download complete!")

def setup_nwpu_resisc45(data_dir='./data/NWPU-RESISC45'):
    """
    Chuẩn bị bộ dữ liệu NWPU-RESISC45 cho Viễn thám (RSIR).
    Người dùng cần tải thủ công nếu không có link direct public,
    hàm này tạo cấu trúc thư mục để người dùng rải data vào.
    """
    data_dir = setup_local_dataset_dir(data_dir)
    print(f"[*] Đang thiết lập NWPU-RESISC45 tại {data_dir}")
    print("!!! LƯU Ý: Vui lòng tải tập dữ liệu NWPU-RESISC45 (45 classes) trực tiếp")
    print(f"và giải nén các thư mục lớp vào: {data_dir}")
    
def setup_chestxray8(data_dir='./data/ChestXray8'):
    """
    Chuẩn bị bộ dữ liệu ChestX-ray8 cho Y tế.
    """
    data_dir = setup_local_dataset_dir(data_dir)
    print(f"[*] Đang thiết lập ChestX-ray8 tại {data_dir}")
    print("!!! LƯU Ý: Vui lòng sử dụng Kaggle API để tải dữ liệu: `kaggle datasets download -d nih-chest-xrays/data`")
    print(f"và giải nén vào: {data_dir}")

if __name__ == "__main__":
    print("=== DATASET DOWNLOADER & SETUP ===")
    setup_local_dataset_dir()
    # download_cifar10()
    setup_nwpu_resisc45()
    setup_chestxray8()
