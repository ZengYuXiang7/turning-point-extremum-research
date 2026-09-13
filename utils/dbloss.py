"""DBLoss for long-horizon forecasting.

Adapted from the official NeurIPS 2025 implementation:
https://github.com/decisionintelligence/DBLoss

保留官方 EMA 公式，用 GPU IIR 滤波实现长视界。
"""

from __future__ import annotations

import torch
from torch import nn
from torchaudio.functional import lfilter


class ExponentialMovingAverage(nn.Module):
    """因果 EMA，以第一个观测为初值。"""

    def __init__(self, alpha: float = 0.2) -> None:
        super().__init__()
        self.alpha = float(alpha)
        decay = 1.0 - self.alpha
        self.register_buffer("a_coefficients", torch.tensor([1.0, -decay]))
        self.register_buffer("b_coefficients", torch.tensor([self.alpha, 0.0]))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        # 输入约定 [batch, time]
        with torch.autocast(device_type=values.device.type, enabled=False):
            work = values.float()
            batch, steps = work.shape
            filtered = lfilter(
                work,
                self.a_coefficients.float(),
                self.b_coefficients.float(),
                clamp=False,
            )
            decay = 1.0 - self.alpha
            index = torch.arange(steps, device=work.device, dtype=work.dtype)
            initial_carry = torch.pow(decay, index + 1.0).unsqueeze(0) * work[:, :1]
            result = filtered + initial_carry
        return result


class DBLoss(nn.Module):
    """官方分解目标：残差 MSE + 趋势 MAE。"""

    def __init__(self, alpha: float = 0.2, beta: float = 0.5) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.ema = ExponentialMovingAverage(alpha=alpha)
        self.mse = nn.MSELoss(reduction="mean")
        self.mae = nn.L1Loss(reduction="mean")

    def components(self, prediction: torch.Tensor, target: torch.Tensor):
        prediction_trend = self.ema(prediction)
        target_trend = self.ema(target)
        prediction_residual = prediction.float() - prediction_trend
        target_residual = target.float() - target_trend
        residual_loss = self.mse(prediction_residual, target_residual)
        trend_loss_raw = self.mae(prediction_trend, target_trend)
        trend_loss_balanced = trend_loss_raw * (
            residual_loss / (trend_loss_raw + 1e-8)
        ).detach()
        total = self.beta * residual_loss + (1.0 - self.beta) * trend_loss_balanced
        return total, residual_loss, trend_loss_raw, trend_loss_balanced

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total, _, _, _ = self.components(prediction, target)
        return total
