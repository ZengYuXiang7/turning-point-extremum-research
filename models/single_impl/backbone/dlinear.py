from __future__ import annotations

import torch
from torch import nn


class MovingAverage(nn.Module):
    def __init__(self, kernel_size: int = 25) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        left = values[:, :1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        right = values[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        padded = torch.cat([left, values, right], dim=1)
        return self.pool(padded.transpose(1, 2)).transpose(1, 2)


class DLinear(nn.Module):
    """Official DLinear formulation with shared temporal projections."""

    def __init__(self, seq_len: int, pred_len: int, channels: int) -> None:
        super().__init__()
        self.moving_average = MovingAverage(25)
        self.seasonal = nn.Linear(seq_len, pred_len)
        self.trend = nn.Linear(seq_len, pred_len)
        nn.init.constant_(self.seasonal.weight, 1.0 / seq_len)
        nn.init.constant_(self.trend.weight, 1.0 / seq_len)
        self.channels = channels

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        trend = self.moving_average(values)
        seasonal = values - trend
        seasonal = self.seasonal(seasonal.transpose(1, 2)).transpose(1, 2)
        trend = self.trend(trend.transpose(1, 2)).transpose(1, 2)
        return seasonal + trend
