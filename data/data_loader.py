import os
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'chest_xray')
PARTITIONS_DIR = os.path.join(BASE_DIR, 'data', 'partitions')

def get_train_transform():
    """训练集数据增强变换：随机水平翻转、随机旋转±10度"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def get_val_transform():
    """验证集/测试集基础变换（无增强）"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def load_raw_datasets(data_dir=RAW_DATA_DIR):
    """加载原始完整数据集（用于计算类别权重等）"""
    val_transform = get_val_transform()
    train_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, 'train'),
        transform=val_transform  # 注意：这里仅用于统计，实际训练时客户端会重新加载带增强的数据集
    )
    val_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, 'val'),
        transform=val_transform
    )
    test_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, 'test'),
        transform=val_transform
    )
    return train_dataset, val_dataset, test_dataset

def load_client_datasets(client_id, augment_train=True):
    """
    加载指定客户端（0-based）的本地训练集和测试集。
    客户端目录结构：partitions/client_{client_id+1}/{train,test}/NORMAL|PNEUMONIA
    参数:
        augment_train: 是否对训练集使用数据增强（默认True）
    返回：train_dataset, test_dataset
    """
    client_dir = os.path.join(PARTITIONS_DIR, f'client_{client_id+1}')
    if not os.path.exists(client_dir):
        raise FileNotFoundError(f"客户端目录不存在: {client_dir}，请先运行 partition_data.py 生成划分。")

    train_transform = get_train_transform() if augment_train else get_val_transform()
    val_transform = get_val_transform()

    train_dataset = datasets.ImageFolder(
        root=os.path.join(client_dir, 'train'),
        transform=train_transform
    )
    test_dataset = datasets.ImageFolder(
        root=os.path.join(client_dir, 'test'),
        transform=val_transform
    )
    return train_dataset, test_dataset

def load_all_client_test_datasets(num_clients=3):
    """
    合并所有客户端的测试集，用于全局评估。
    """
    test_datasets = []
    for i in range(num_clients):
        client_dir = os.path.join(PARTITIONS_DIR, f'client_{i+1}')
        test_dataset = datasets.ImageFolder(
            root=os.path.join(client_dir, 'test'),
            transform=get_val_transform()
        )
        test_datasets.append(test_dataset)
    return ConcatDataset(test_datasets)