import sys
import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from torch.utils.data import DataLoader

# 动态获取项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from models.cnn_model import PneumoniaCNN
from data.data_loader import load_all_client_test_datasets


def evaluate_model(model, test_loader, device):
    """在测试集上评估模型，返回各项指标及原始预测结果"""
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)

            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # 提取正类(PNEUMONIA)的预测概率用于计算AUC

    return {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, pos_label=1, zero_division=0),
        'recall': recall_score(all_labels, all_preds, pos_label=1, zero_division=0),
        'f1': f1_score(all_labels, all_preds, pos_label=1, zero_division=0),
        'auc': roc_auc_score(all_labels, all_probs),
        'confusion_matrix': confusion_matrix(all_labels, all_preds),
        'labels': all_labels,
        'predictions': all_preds,
        'probabilities': all_probs
    }


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 拼接模型权重文件的绝对路径
    model_path = os.path.join(BASE_DIR, 'results', 'models', 'final_model_10_dp.pth')

    if not os.path.exists(model_path):
        print(f"错误：找不到模型文件 {model_path}")
        return

    # 加载合并的测试集
    test_dataset = load_all_client_test_datasets(num_clients=3)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 加载模型
    model = PneumoniaCNN(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.to(device)

    # 评估
    results = evaluate_model(model, test_loader, device)

    # 打印结果
    print("========== 全局模型在合并测试集上的性能 ==========")
    print(f"测试集样本数: {len(test_dataset)}")
    print(f"准确率:   {results['accuracy']:.4f}")
    print(f"精确率:   {results['precision']:.4f}")
    print(f"召回率:   {results['recall']:.4f}")
    print(f"F1分数 (F1): {results['f1']:.4f}")
    print(f"AUC:      {results['auc']:.4f}")
    print("混淆矩阵:")
    print(results['confusion_matrix'])
    print("================================================")


if __name__ == "__main__":
    main()