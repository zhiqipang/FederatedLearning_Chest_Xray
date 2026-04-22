import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

# 路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
RESULTS_PLOTS_DIR = os.path.join(BASE_DIR, 'results', 'plots')
os.makedirs(RESULTS_PLOTS_DIR, exist_ok=True)

from models.cnn_model import PneumoniaCNN
from data.data_loader import load_all_client_test_datasets

# ================= 核心配置区（请修改为你实际的文件名） =================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_PATHS = {
    'A': 'final_model_10_dp.pth',  # 实验A：有DP，epoch=10
    'B': 'final_model_10.pth',  # 实验B：无DP，epoch=10
    'C': 'final_model_5.pth',  # 实验C：无DP，epoch=5
    'D': 'final_model_5_dp.pth'  # 实验D：有DP，epoch=5
}


# ========================================================================

def get_predictions(model_path):
    """加载模型并返回所有真实标签和预测标签"""
    model = PneumoniaCNN(num_classes=2)
    model.load_state_dict(torch.load(os.path.join(BASE_DIR, 'results', 'models', model_path),
                                     map_location=DEVICE, weights_only=False))
    model.to(DEVICE)
    model.eval()

    test_dataset = load_all_client_test_datasets(num_clients=3)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    all_labels, all_preds = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    return np.array(all_labels), np.array(all_preds)


def plot_cm(exp_id, labels, preds):
    """绘制并保存单个混淆矩阵"""
    cm = confusion_matrix(labels, preds)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(6, 5))

    # 使用 imshow 绘制热力图
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    # 设置坐标轴
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=['正常', '肺炎'],
           yticklabels=['正常', '肺炎'],
           title=f'实验{exp_id} - 混淆矩阵',
           ylabel='真实标签',
           xlabel='预测标签')

    # 旋转标签对齐
    plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")

    # 在方格内写入数字
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()

    save_path = os.path.join(RESULTS_PLOTS_DIR, f'confusion_matrix_exp_{exp_id}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"实验 {exp_id} 混淆矩阵已保存至: {save_path}")


if __name__ == "__main__":
    for exp_id, filename in MODEL_PATHS.items():
        print(f"正在处理实验 {exp_id} ({filename})...")
        labels, preds = get_predictions(filename)
        plot_cm(exp_id, labels, preds)
    print("全部混淆矩阵生成完毕！")