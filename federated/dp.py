"""
差分隐私模块，封装Opacus相关功能，实现模块化调用。
"""

import torch
from typing import Optional, Tuple, Any

class DifferentialPrivacy:
    """
    差分隐私处理器，管理隐私引擎、噪声添加和隐私预算计算。
    """
    def __init__(
        self,
        noise_multiplier: float = 1.0,
        max_grad_norm: float = 1.0,
        delta: float = 1e-5,
    ):
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.delta = delta
        self.privacy_engine = None
        self.epsilon = None

    def make_private(
        self,
        module: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        data_loader: torch.utils.data.DataLoader,
    ) -> Tuple[torch.nn.Module, torch.optim.Optimizer, torch.utils.data.DataLoader]:
        # 延迟导入Opacus，避免启动时版本不兼容
        from opacus import PrivacyEngine
        self.privacy_engine = PrivacyEngine()
        module, optimizer, data_loader = self.privacy_engine.make_private(
            module=module,
            optimizer=optimizer,
            data_loader=data_loader,
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.max_grad_norm,
        )
        return module, optimizer, data_loader

    def get_epsilon(self) -> float:
        if self.privacy_engine is None:
            return 0.0
        self.epsilon = self.privacy_engine.get_epsilon(self.delta)
        return self.epsilon

    def get_dp_config(self) -> dict:
        return {
            'noise_multiplier': self.noise_multiplier,
            'max_grad_norm': self.max_grad_norm,
            'delta': self.delta,
        }