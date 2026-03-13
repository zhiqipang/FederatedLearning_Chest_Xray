import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import matplotlib.pyplot as plt
from federated.client import FederatedClient
from federated.fed_avg import fed_avg
from models.cnn_model import PneumoniaCNN
from data.data_loader import get_client_dataloaders


class FederatedServer:
    def __init__(self, num_clients=3, device='cuda', batch_size=32, learning_rate=0.001):
        self.num_clients = num_clients
        self.device = device
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.global_model = PneumoniaCNN().to(device)

        # 初始化所有客户端
        self.clients = []
        for i in range(1, num_clients + 1):
            client = FederatedClient(
                client_id=i,
                device=device,
                batch_size=batch_size,
                learning_rate=learning_rate
            )
            self.clients.append(client)

        # 记录训练历史
        self.history = {'round': [], 'server_eval_acc': [], 'server_eval_loss': []}

    def get_global_parameters(self):
        return [param.data for param in self.global_model.parameters()]

    def set_global_parameters(self, parameters):
        with torch.no_grad():
            for param, new_param in zip(self.global_model.parameters(), parameters):
                param.data = new_param.data.clone()

    def aggregate(self, client_params_list, client_weights_list):
        """聚合客户端参数，更新全局模型"""
        new_params = fed_avg(self.global_model, client_params_list, client_weights_list)
        self.set_global_parameters(new_params)

    def server_evaluate(self, test_loader):
        """在给定测试集上评估全局模型"""
        self.global_model.eval()
        correct = 0
        total = 0
        test_loss = 0.0
        criterion = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.global_model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total if total > 0 else 0
        avg_loss = test_loss / len(test_loader) if len(test_loader) > 0 else 0
        return accuracy, avg_loss

    def train_round(self, local_epochs=1):
        """执行一轮联邦训练"""
        # 下发当前全局参数
        global_params = self.get_global_parameters()
        for client in self.clients:
            client.set_parameters(global_params)

        # 各客户端本地训练
        client_params_list = []
        client_weights_list = []
        for client in self.clients:
            params, num_samples = client.train(epochs=local_epochs)
            client_params_list.append(params)
            client_weights_list.append(num_samples)

        # 服务器聚合
        self.aggregate(client_params_list, client_weights_list)
        return client_weights_list

    def fit(self, rounds=10, local_epochs=5, test_loader=None, save_path='results/models'):
        """执行多轮联邦训练"""
        if save_path:
            os.makedirs(save_path, exist_ok=True)

        for rnd in range(1, rounds + 1):
            print(f"\n=== Round {rnd} ===")
            weights = self.train_round(local_epochs=local_epochs)
            print(f"聚合完成，各客户端样本数: {weights}")

            if test_loader is not None:
                acc, loss = self.server_evaluate(test_loader)
                print(f"全局模型测试集准确率: {acc:.4f}, 损失: {loss:.4f}")
                self.history['round'].append(rnd)
                self.history['server_eval_acc'].append(acc)
                self.history['server_eval_loss'].append(loss)

            # 每5轮保存一次中间模型
            if save_path and rnd % 5 == 0:
                torch.save(self.global_model.state_dict(), f"{save_path}/global_model_round{rnd}.pth")

        if save_path:
            torch.save(self.global_model.state_dict(), f"{save_path}/global_model_final.pth")
            print(f"最终模型已保存至 {save_path}/global_model_final.pth")

        return self.history

    def plot_history(self):
        """绘制训练曲线并保存到 results/plots/"""
        if not self.history['round']:
            print("没有历史记录可绘制")
            return

        os.makedirs('results/plots', exist_ok=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(self.history['round'], self.history['server_eval_acc'], 'b-')
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Server Evaluation Accuracy')
        ax1.grid(True)

        ax2.plot(self.history['round'], self.history['server_eval_loss'], 'r-')
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Loss')
        ax2.set_title('Server Evaluation Loss')
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig('results/plots/training_curves.png')
        print("训练曲线已保存至 results/plots/training_curves.png")
        plt.show()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 创建服务器（3个客户端）
    server = FederatedServer(num_clients=3, device=device, batch_size=32, learning_rate=0.001)

    # 使用客户端1的测试集作为全局测试集（实际可用原始测试集，此处简化）
    _, test_loader, _ = get_client_dataloaders(1, batch_size=32)

    # 训练5轮（为快速演示，实际可调大）
    history = server.fit(rounds=5, local_epochs=2, test_loader=test_loader, save_path='results/models')

    # 绘制曲线
    server.plot_history()