import sys
import os
# 将项目根目录加入环境变量
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import flwr as fl
from federated.fed_avg import SaveModelStrategy

strategy = SaveModelStrategy(
    fraction_fit=1.0,       # 每轮要求 100% 的可用客户端参与训练
    fraction_evaluate=1.0,  # 每轮要求 100% 的可用客户端参与评估
    min_fit_clients=3,
    min_evaluate_clients=3,
    min_available_clients=3, # 等待至少 3 个客户端连接后才启动训练
)

fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=50),
    strategy=strategy,
)