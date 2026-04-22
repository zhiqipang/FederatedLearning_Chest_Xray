import os
import sys

# 将项目根目录加入环境变量
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import torch
import flwr as fl
import flwr.common as common

from models.cnn_model import PneumoniaCNN

RESULTS_MODELS_DIR = os.path.join(BASE_DIR, 'results', 'models')
os.makedirs(RESULTS_MODELS_DIR, exist_ok=True)


class SaveModelStrategy(fl.server.strategy.FedAvg):
    """FedAvg策略：在每轮聚合后自动保存全局模型"""

    def aggregate_fit(self, rnd, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(rnd, results, failures)

        if aggregated_parameters is not None:
            ndarrays = common.parameters_to_ndarrays(aggregated_parameters)
            # 将扁平化的参数列表还原为模型 state_dict
            state_dict = {
                k: torch.tensor(v)
                for k, v in zip(PneumoniaCNN(num_classes=2).state_dict().keys(), ndarrays)
            }

            model = PneumoniaCNN(num_classes=2)
            model.load_state_dict(state_dict)

            save_path = os.path.join(RESULTS_MODELS_DIR, f"global_model_round_{rnd}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"第 {rnd} 轮全局模型已保存至 {save_path}")

        return aggregated_parameters, aggregated_metrics