import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import flwr as fl
import numpy as np
from typing import Optional

from models.cnn_model import PneumoniaCNN
from data.data_loader import load_client_datasets, load_raw_datasets  # 注意导入

class MedicalClient(fl.client.NumPyClient):
    def __init__(
        self,
        train_dataset,
        val_dataset,          # 这里用测试集作为验证集
        num_classes=2,
        class_weights: Optional[torch.Tensor] = None,
        device=None,
        local_epochs=5,
        lr=0.001,
        batch_size=32,
        dp_config: Optional[dict] = None,
    ):
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model = PneumoniaCNN(num_classes=num_classes).to(self.device)
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        self.local_epochs = local_epochs
        self.lr = lr

        if class_weights is not None:
            class_weights = class_weights.to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.dp_config = dp_config
        self.privacy_engine = None
        if dp_config is not None:
            from opacus import PrivacyEngine
            self.privacy_engine = PrivacyEngine()
            self.model, self.optimizer, self.train_loader = self.privacy_engine.make_private(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=self.train_loader,
                noise_multiplier=dp_config['noise_multiplier'],
                max_grad_norm=dp_config['max_grad_norm'],
            )
            print(f"差分隐私已启用: noise_multiplier={dp_config['noise_multiplier']}, max_grad_norm={dp_config['max_grad_norm']}")

    def get_parameters(self, config):
        return [val.detach().cpu().numpy() for val in self.model.parameters()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.parameters(), parameters)
        for param, val in params_dict:
            param.data = torch.tensor(val, device=self.device)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        total_loss = 0.0
        for epoch in range(self.local_epochs):
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
        avg_loss = total_loss / len(self.train_loader) / self.local_epochs

        epsilon = None
        if self.privacy_engine is not None:
            epsilon = self.privacy_engine.get_epsilon(delta=1e-5)
            print(f"本轮训练隐私预算 ε = {epsilon:.2f} (δ=1e-5)")

        metrics = {"loss": avg_loss}
        if epsilon is not None:
            metrics["dp_epsilon"] = epsilon
        return self.get_parameters(config={}), len(self.train_loader.dataset), metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        correct = 0
        total = 0
        loss = 0.0
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss += self.criterion(outputs, labels).item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total if total > 0 else 0
        avg_loss = loss / len(self.val_loader)
        return avg_loss, total, {"accuracy": accuracy}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--client-id", type=int, default=0, choices=[0, 1, 2], help="客户端索引（0、1或2）")
    parser.add_argument("--dp", action="store_true", help="启用差分隐私")
    args = parser.parse_args()

    # 加载该客户端的本地训练集和测试集（测试集作为验证集使用）
    train_dataset, test_dataset = load_client_datasets(args.client_id)
    val_dataset = test_dataset  # 用测试集作为本地验证集

    # 计算类别权重（基于原始完整训练集，确保全局统一）
    full_train, _, _ = load_raw_datasets()
    labels = full_train.targets
    num_samples_per_class = np.bincount(labels)
    print(f"训练集类别分布: {num_samples_per_class}")
    class_weights = 1.0 / torch.tensor(num_samples_per_class, dtype=torch.float)
    class_weights = class_weights / class_weights.sum() * 2
    print(f"类别权重: {class_weights}")

    dp_config = None
    if args.dp:
        dp_config = {'noise_multiplier': 1.0, 'max_grad_norm': 1.0}

    client = MedicalClient(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_classes=2,
        class_weights=class_weights,
        local_epochs=10,
        lr=0.001,
        batch_size=32,
        dp_config=dp_config,
    )

    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=client.to_client()
    )