import sys
import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from torch.utils.data import DataLoader

# 将项目根目录加入路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn_model import PneumoniaCNN
from data.data_loader import load_all_client_test_datasets, get_transform

def evaluate_model(model, test_loader, device):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, pos_label=1, zero_division=0)
    recall = recall_score(all_labels, all_preds, pos_label=1, zero_division=0)
    f1 = f1_score(all_labels, all_preds, pos_label=1, zero_division=0)
    auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm,
        'labels': all_labels,
        'predictions': all_preds,
        'probabilities': all_probs
    }

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              'results', 'models', 'global_model_round_10.pth')

    # 加载合并的测试集
    test_dataset = load_all_client_test_datasets(num_clients=3)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 加载模型
    model = PneumoniaCNN(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.to(device)
    model.eval()

    # 评估
    results = evaluate_model(model, test_loader, device)

    # 打印结果
    print("========== 全局模型在合并测试集上的性能 ==========")
    print(f"测试集样本数: {len(test_dataset)}")
    print(f"准确率 (Accuracy): {results['accuracy']:.4f}")
    print(f"精确率 (Precision): {results['precision']:.4f}")
    print(f"召回率 (Recall): {results['recall']:.4f}")
    print(f"F1分数: {results['f1']:.4f}")
    print(f"AUC: {results['auc']:.4f}")
    print("混淆矩阵:")
    print(results['confusion_matrix'])
    print("================================================")

if __name__ == "__main__":
    main()