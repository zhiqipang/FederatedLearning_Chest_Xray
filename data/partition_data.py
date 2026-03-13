import os
import shutil
import random
from sklearn.model_selection import train_test_split

# 路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'chest_xray')
PARTITIONS_DIR = os.path.join(BASE_DIR, 'data', 'partitions')
NUM_CLIENTS = 3
TEST_SIZE = 0.2  # 每个客户端内划分 20% 作为测试集

def get_all_samples():
    """从原始 train/val/test 中收集所有样本（图片路径+标签）"""
    samples = []
    for split in ['train', 'val', 'test']:
        for label in ['NORMAL', 'PNEUMONIA']:
            src_dir = os.path.join(RAW_DIR, split, label)
            if not os.path.exists(src_dir):
                continue
            for img_name in os.listdir(src_dir):
                img_path = os.path.join(src_dir, img_name)
                samples.append((img_path, label))
    random.shuffle(samples)
    print(f"共收集到 {len(samples)} 张图片")
    return samples

def partition_and_split():
    # 清空 partitions 目录（如果存在）
    if os.path.exists(PARTITIONS_DIR):
        shutil.rmtree(PARTITIONS_DIR)
    os.makedirs(PARTITIONS_DIR, exist_ok=True)

    samples = get_all_samples()
    # 简单轮询分配给 NUM_CLIENTS 个客户端
    partitions = [[] for _ in range(NUM_CLIENTS)]
    for idx, s in enumerate(samples):
        partitions[idx % NUM_CLIENTS].append(s)

    for client_id, client_samples in enumerate(partitions, start=1):
        print(f"客户端 {client_id} 分配了 {len(client_samples)} 张图片")

        # 在客户端内部分割训练集和测试集（按标签分层）
        labels = [label for _, label in client_samples]
        train_samples, test_samples = train_test_split(
            client_samples,
            test_size=TEST_SIZE,
            random_state=42,
            stratify=labels
        )

        # 复制训练集图片
        for img_path, label in train_samples:
            dest_dir = os.path.join(PARTITIONS_DIR, f'client_{client_id}', 'train', label)
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy2(img_path, dest_dir)

        # 复制测试集图片
        for img_path, label in test_samples:
            dest_dir = os.path.join(PARTITIONS_DIR, f'client_{client_id}', 'test', label)
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy2(img_path, dest_dir)

    print("数据集划分与复制完成！")

if __name__ == '__main__':
    partition_and_split()