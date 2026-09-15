from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class Acc30BoundaryLoss(nn.Module):
    """MSE + 平滑惩罚越过严格 30% 相对误差边界。"""

    def __init__(self, weight: float = 0.3, temperature: float = 0.03, min_power_kw: float = 100.0,) -> None:
        super().__init__()
        self.weight = float(weight)
        self.temperature = float(temperature)
        self.min_power_kw = float(min_power_kw)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor, power_mean: torch.Tensor, power_std: torch.Tensor,) -> torch.Tensor:
        prediction = prediction.float()
        target = target.float()
        power_mean = power_mean.float()
        power_std = power_std.float()
        mse = torch.mean((prediction - target) ** 2)

        # 相对误差在 kW 空间算；只在真值功率超过阈值的点上惩罚
        target_kw = target * power_std + power_mean
        prediction_kw = prediction * power_std + power_mean
        valid = target_kw > self.min_power_kw
        relative_error = torch.abs(prediction_kw - target_kw) / target_kw
        boundary = F.softplus((relative_error - 0.30) / self.temperature) * self.temperature
        boundary_raw = boundary[valid].mean()
        total = mse + self.weight * boundary_raw
        return total
