import os
import sys

# 将项目根目录加入路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def count_images_in_dir(dir_path):
    """统计指定目录下 NORMAL 和 PNEUMONIA 子文件夹中的图片数量"""
    counts = {'NORMAL': 0, 'PNEUMONIA': 0}
    if not os.path.exists(dir_path):
        return counts
    for label in ['NORMAL', 'PNEUMONIA']:
        label_dir = os.path.join(dir_path, label)
        if os.path.exists(label_dir):
            counts[label] = len([f for f in os.listdir(label_dir) if os.path.isfile(os.path.join(label_dir, f))])
    return counts

def print_stats(title, counts):
    print(f"{title}: 正常={counts['NORMAL']}, 肺炎={counts['PNEUMONIA']}, 总计={counts['NORMAL']+counts['PNEUMONIA']}")

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(base_dir, 'data', 'raw', 'chest_xray')
    partitions_dir = os.path.join(base_dir, 'data', 'partitions')

    print("========== 原始数据集统计 ==========")
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(raw_dir, split)
        counts = count_images_in_dir(split_dir)
        print_stats(f"原始 {split} 集", counts)

    print("\n========== 各客户端数据集统计 ==========")
    total_train_all = {'NORMAL': 0, 'PNEUMONIA': 0}
    total_test_all = {'NORMAL': 0, 'PNEUMONIA': 0}

    for client_id in range(1, 4):
        client_dir = os.path.join(partitions_dir, f'client_{client_id}')
        if not os.path.exists(client_dir):
            print(f"客户端 {client_id} 目录不存在，请先运行 partition_data.py")
            continue

        train_dir = os.path.join(client_dir, 'train')
        test_dir = os.path.join(client_dir, 'test')

        train_counts = count_images_in_dir(train_dir)
        test_counts = count_images_in_dir(test_dir)

        print(f"\n客户端 {client_id}:")
        print_stats("  训练集", train_counts)
        print_stats("  测试集", test_counts)

        # 累加用于总计
        for k in total_train_all:
            total_train_all[k] += train_counts[k]
        for k in total_test_all:
            total_test_all[k] += test_counts[k]

    print("\n========== 所有客户端汇总 ==========")
    print_stats("所有客户端训练集总和", total_train_all)
    print_stats("所有客户端测试集总和", total_test_all)
    print_stats("所有客户端总计 (训练+测试)", {'NORMAL': total_train_all['NORMAL']+total_test_all['NORMAL'],
                                              'PNEUMONIA': total_train_all['PNEUMONIA']+total_test_all['PNEUMONIA']})

if __name__ == "__main__":
    main()