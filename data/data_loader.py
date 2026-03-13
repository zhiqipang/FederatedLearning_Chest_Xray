import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 基础路径
PARTITIONS_DIR = 'data/partitions'

def get_client_dataloaders(client_id, batch_size=32, num_workers=0):
    """
    为指定客户端加载训练和测试 DataLoader
    client_id: 1, 2, 3
    batch_size: 批次大小
    num_workers: 数据加载进程数（Windows建议设为0）
    """
    # 训练集路径
    train_dir = os.path.join(PARTITIONS_DIR, f'client_{client_id}', 'train')
    test_dir = os.path.join(PARTITIONS_DIR, f'client_{client_id}', 'test')

    # 训练集数据增强与归一化
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),                # 统一尺寸
        transforms.RandomHorizontalFlip(p=0.5),       # 随机水平翻转
        transforms.RandomRotation(10),                 # 随机旋转±10度
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet 均值
                             std=[0.229, 0.224, 0.225])   # ImageNet 标准差
    ])

    # 测试集只做 resize 和归一化
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 使用 ImageFolder 加载数据
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              pin_memory=True)   # GPU 时可加速
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers,
                             pin_memory=True)

    return train_loader, test_loader, train_dataset.classes

# 简单测试（直接运行本文件时执行）
if __name__ == '__main__':
    for cid in [1, 2, 3]:
        train_loader, test_loader, classes = get_client_dataloaders(cid, batch_size=32)
        print(f"客户端 {cid}: 训练集样本数 = {len(train_loader.dataset)}, "
              f"测试集样本数 = {len(test_loader.dataset)}, 类别 = {classes}")