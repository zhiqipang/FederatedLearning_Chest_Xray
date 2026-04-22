import sys
import os

# 将项目根目录加入环境变量
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional
import flwr as fl
from opacus import PrivacyEngine

from models.cnn_model import PneumoniaCNN
from data.data_loader import load_client_datasets, load_raw_datasets


class MedicalClient(fl.client.NumPyClient):
    def __init__(self, train_dataset, val_dataset, num_classes=2,
                 class_weights: Optional[torch.Tensor] = None, device=None,
                 local_epochs=10, lr=0.001, batch_size=32,
                 dp_config: Optional[dict] = None):

        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = PneumoniaCNN(num_classes=num_classes).to(self.device)

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        self.local_epochs = local_epochs
        self.lr = lr

        # 加权损失函数，用于缓解类别不均衡问题
        weights = class_weights.to(self.device) if class_weights is not None else None
        self.criterion = nn.CrossEntropyLoss(weight=weights)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # 初始化差分隐私引擎
        self.privacy_engine = None
        if dp_config:
            self.privacy_engine = PrivacyEngine(accountant='rdp')
            self.model, self.optimizer, self.train_loader = self.privacy_engine.make_private(
                module=self.model, optimizer=self.optimizer, data_loader=self.train_loader,
                noise_multiplier=dp_config['noise_multiplier'], max_grad_norm=dp_config['max_grad_norm']
            )
            print(f"差分隐私已启用 (noise={dp_config['noise_multiplier']})")

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v, device=self.device) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        total_loss = 0.0

        for _ in range(self.local_epochs):
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(images), labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

        avg_loss = total_loss / (len(self.train_loader) * self.local_epochs)
        print(f"  [Client] 本地训练损失: {avg_loss:.4f}")
        return self.get_parameters(config={}), len(self.train_loader.dataset), {"loss": avg_loss}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss += self.criterion(outputs, labels).item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = loss / len(self.val_loader)
        accuracy = correct / total if total > 0 else 0
        print(f"  [Client] 本地验证准确率: {accuracy:.4f}")
        return avg_loss, total, {"accuracy": accuracy}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--client-id", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--dp", action="store_true", help="启用差分隐私")
    args = parser.parse_args()

    train_dataset, test_dataset = load_client_datasets(args.client_id, augment_train=True)

    # 使用全局原始训练集统计类别权重，缓解本地数据分布偏移
    full_train, _, _ = load_raw_datasets()
    num_samples = np.bincount(full_train.targets)
    class_weights = 1.0 / torch.tensor(num_samples, dtype=torch.float)
    class_weights = class_weights / class_weights.sum() * 2  # 归一化并保持梯度量级

    dp_config = {'noise_multiplier': 0.3, 'max_grad_norm': 1.0} if args.dp else None

    client = MedicalClient(
        train_dataset=train_dataset, val_dataset=test_dataset, num_classes=2,
        class_weights=class_weights, local_epochs=10, lr=0.001, batch_size=32,
        dp_config=dp_config
    )

    fl.client.start_client(server_address="127.0.0.1:8080", client=client.to_client())