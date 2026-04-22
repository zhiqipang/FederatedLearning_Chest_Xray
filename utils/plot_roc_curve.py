import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
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

def get_probabilities(model_path):
    """加载模型并返回所有真实标签和正类预测概率"""
    model = PneumoniaCNN(num_classes=2)
    model.load_state_dict(torch.load(os.path.join(BASE_DIR, 'results', 'models', model_path),
                                     map_location=DEVICE, weights_only=False))
    model.to(DEVICE)
    model.eval()

    test_dataset = load_all_client_test_datasets(num_clients=3)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    all_labels, all_probs = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            # 提取类别 1 (肺炎) 的概率
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_labels), np.array(all_probs)


def plot_roc(exp_id, labels, probs):
    """绘制并保存单个ROC曲线"""
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 计算FPR, TPR
    fpr, tpr, thresholds = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))

    # 绘制ROC曲线
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.4f})')
    # 绘制随机猜测对角线
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率', fontsize=12)
    plt.ylabel('真阳性率', fontsize=12)
    plt.title(f'实验{exp_id} - ROC曲线', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    save_path = os.path.join(RESULTS_PLOTS_DIR, f'roc_curve_exp_{exp_id}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"实验 {exp_id} ROC曲线已保存至: {save_path} (AUC={roc_auc:.4f})")


if __name__ == "__main__":
    for exp_id, filename in MODEL_PATHS.items():
        print(f"正在处理实验 {exp_id} ({filename})...")
        labels, probs = get_probabilities(filename)
        plot_roc(exp_id, labels, probs)
    print("全部ROC曲线生成完毕！")