import os
from torch.utils.data import ConcatDataset
from torchvision import datasets, transforms

# 动态获取项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'chest_xray')
PARTITIONS_DIR = os.path.join(BASE_DIR, 'data', 'partitions')

# ImageNet 标准归一化参数（适配预训练模型）
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transform():
    """包含数据增强的训练集变换"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def get_val_transform():
    """无数据增强的验证/测试集基础变换"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def load_raw_datasets(data_dir=RAW_DATA_DIR):
    """加载原始完整数据集"""
    transform = get_val_transform()
    train_set = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
    val_set = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform)
    test_set = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform)
    return train_set, val_set, test_set


def load_client_datasets(client_id, augment_train=True):
    """加载指定客户端的训练集和测试集 """
    client_dir = os.path.join(PARTITIONS_DIR, f'client_{client_id + 1}')
    if not os.path.exists(client_dir):
        raise FileNotFoundError(f"目录不存在: {client_dir}，请先运行 partition_data.py")

    train_transform = get_train_transform() if augment_train else get_val_transform()
    train_set = datasets.ImageFolder(os.path.join(client_dir, 'train'), transform=train_transform)
    test_set = datasets.ImageFolder(os.path.join(client_dir, 'test'), transform=get_val_transform())

    return train_set, test_set


def load_all_client_test_datasets(num_clients=3):
    """合并所有客户端的测试集，用于联邦学习全局模型评估"""
    test_datasets = [
        datasets.ImageFolder(
            root=os.path.join(PARTITIONS_DIR, f'client_{i + 1}', 'test'),
            transform=get_val_transform()
        ) for i in range(num_clients)
    ]
    return ConcatDataset(test_datasets)