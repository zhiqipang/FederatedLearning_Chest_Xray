import sys
import os
# 将项目根目录添加到模块搜索路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from models.cnn_model import PneumoniaCNN
from data.data_loader import get_client_dataloaders
import copy

class FederatedClient:
    def __init__(self, client_id, device, batch_size=32, learning_rate=0.001):
        self.client_id = client_id
        self.device = device
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # 加载本地数据
        self.train_loader, self.test_loader, self.classes = get_client_dataloaders(
            client_id, batch_size=batch_size, num_workers=0
        )

        # 初始化模型（后续将接收全局参数）
        self.model = PneumoniaCNN().to(device)
        self.criterion = nn.CrossEntropyLoss()

    def set_parameters(self, parameters):
        """将全局模型参数加载到本地模型"""
        with torch.no_grad():
            for param, new_param in zip(self.model.parameters(), parameters):
                param.data = new_param.data.clone()

    def get_parameters(self):
        """返回当前本地模型参数（用于上传）"""
        return [param.data for param in self.model.parameters()]

    def train(self, epochs=1):
        """在本地数据上训练多个epochs，返回训练后的参数和样本数量"""
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)

        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            # 可选：打印每个epoch的loss（调试用）
            # print(f'Client {self.client_id}, Epoch {epoch+1}, Loss: {running_loss/len(self.train_loader):.4f}')

        # 返回更新后的参数和本地样本数量（用于联邦平均加权）
        return self.get_parameters(), len(self.train_loader.dataset)

    def evaluate(self):
        """在本地测试集上评估模型，返回准确率和loss"""
        self.model.eval()
        correct = 0
        total = 0
        test_loss = 0.0

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                test_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total if total > 0 else 0
        avg_loss = test_loss / len(self.test_loader) if len(self.test_loader) > 0 else 0
        return accuracy, avg_loss


# 简单测试（直接运行本文件时执行）
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 测试客户端1
    client = FederatedClient(client_id=1, device=device, batch_size=32, learning_rate=0.001)

    # 获取初始参数（随机初始化）
    initial_params = client.get_parameters()
    print(f"初始参数数量: {len(initial_params)}")

    # 训练一个epoch
    updated_params, num_samples = client.train(epochs=1)
    print(f"训练后参数数量: {len(updated_params)}，本地样本数: {num_samples}")

    # 评估
    acc, loss = client.evaluate()
    print(f"本地测试准确率: {acc:.4f}, 损失: {loss:.4f}")