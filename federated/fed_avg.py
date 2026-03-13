import os
import torch
import flwr as fl
import flwr.common as common
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from models.cnn_model import PneumoniaCNN

RESULTS_MODELS_DIR = os.path.join(BASE_DIR, 'results', 'models')
os.makedirs(RESULTS_MODELS_DIR, exist_ok=True)

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(rnd, results, failures)

        if aggregated_parameters is not None:
            print(f"第 {rnd} 轮聚合完成，正在保存模型...")
            ndarrays = common.parameters_to_ndarrays(aggregated_parameters)
            model = PneumoniaCNN(num_classes=2)
            params_dict = zip(model.state_dict().keys(), ndarrays)
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
            model.load_state_dict(state_dict)

            save_path = os.path.join(RESULTS_MODELS_DIR, f"global_model_round_{rnd}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"模型已保存至 {save_path}")

        return aggregated_parameters, aggregated_metrics