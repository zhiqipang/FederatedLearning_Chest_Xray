import os
import shutil
import random
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'chest_xray')
PARTITIONS_DIR = os.path.join(BASE_DIR, 'data', 'partitions')

NUM_CLIENTS = 3
TEST_SIZE = 0.2
CATEGORIES = ['NORMAL', 'PNEUMONIA']
RAW_SPLITS = ['train', 'val', 'test']


def get_all_samples():
    """从原始数据集收集所有图片路径及标签，并随机打乱"""
    samples = []
    for split in RAW_SPLITS:
        for label in CATEGORIES:
            src_dir = os.path.join(RAW_DIR, split, label)
            if not os.path.exists(src_dir):
                continue
            for img_name in os.listdir(src_dir):
                samples.append((os.path.join(src_dir, img_name), label))

    random.shuffle(samples)
    print(f"共收集到 {len(samples)} 张图片")
    return samples


def partition_and_split():
    """将数据轮询分配给各客户端，并在客户端内部进行分层训练/测试集划分"""
    if os.path.exists(PARTITIONS_DIR):
        shutil.rmtree(PARTITIONS_DIR)

    samples = get_all_samples()

    # 轮询分配，保证各客户端数据量均衡且类别混合
    partitions = [[] for _ in range(NUM_CLIENTS)]
    for idx, sample in enumerate(samples):
        partitions[idx % NUM_CLIENTS].append(sample)

    for client_id, client_samples in enumerate(partitions, start=1):
        print(f"客户端 {client_id} 分配了 {len(client_samples)} 张图片")

        labels = [label for _, label in client_samples]
        train_set, test_set = train_test_split(
            client_samples, test_size=TEST_SIZE, random_state=42, stratify=labels
        )

        # 统一处理训练集和测试集的文件复制
        for split_name, split_data in [('train', train_set), ('test', test_set)]:
            for img_path, label in split_data:
                dest_dir = os.path.join(PARTITIONS_DIR, f'client_{client_id}', split_name, label)
                os.makedirs(dest_dir, exist_ok=True)
                shutil.copy2(img_path, dest_dir)

    print("数据集划分与复制完成！")


if __name__ == '__main__':
    partition_and_split()