import os
import shutil
import random
from sklearn.model_selection import train_test_split

# 配置参数
RAW_DIR = 'data/raw/chest_xray'
PARTITIONS_DIR = 'data/partitions'
NUM_CLIENTS = 3
TEST_SIZE = 0.2          # 每个客户端划分20%作为本地测试集
RANDOM_SEED = 42         # 保证划分结果可复现

def clear_partitions():
    """清空之前的分区目录，避免旧文件干扰"""
    if os.path.exists(PARTITIONS_DIR):
        shutil.rmtree(PARTITIONS_DIR)
    os.makedirs(PARTITIONS_DIR)

def get_all_samples():
    """收集所有原始图片的路径和标签"""
    samples = []
    for split in ['train', 'val', 'test']:
        for label in ['NORMAL', 'PNEUMONIA']:
            folder = os.path.join(RAW_DIR, split, label)
            if not os.path.isdir(folder):
                continue
            for img in os.listdir(folder):
                samples.append((os.path.join(folder, img), label))
    random.seed(RANDOM_SEED)
    random.shuffle(samples)
    return samples

def partition_and_split():
    clear_partitions()
    samples = get_all_samples()
    # 简单轮询分配给3个客户端
    partitions = [[] for _ in range(NUM_CLIENTS)]
    for idx, s in enumerate(samples):
        partitions[idx % NUM_CLIENTS].append(s)

    # 为每个客户端划分本地训练集和测试集
    for i, client_samples in enumerate(partitions):
        # 按标签分层划分
        labels = [label for _, label in client_samples]
        train_samples, test_samples = train_test_split(
            client_samples,
            test_size=TEST_SIZE,
            random_state=RANDOM_SEED,
            stratify=labels
        )
        # 复制文件到对应客户端目录
        for split_name, split_samples in [('train', train_samples), ('test', test_samples)]:
            for img_path, label in split_samples:
                dest_dir = os.path.join(PARTITIONS_DIR, f'client_{i+1}', split_name, label)
                os.makedirs(dest_dir, exist_ok=True)
                shutil.copy2(img_path, dest_dir)
    print("数据集划分完成！结果保存在:", PARTITIONS_DIR)

if __name__ == '__main__':
    partition_and_split()