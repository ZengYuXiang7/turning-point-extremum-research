from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class AdaptiveRevINState:
    """保存逆归一化所需的历史窗口统计量。"""

    mean: torch.Tensor
    std: torch.Tensor
    drift: torch.Tensor


class AdaptiveRevIN(nn.Module):
    """FreqLite（arXiv:2606.01339v1）提出的自适应 RevIN。"""

    def __init__(self, horizon: int, eps: float = 1e-5) -> None:
        super().__init__()
        if horizon <= 0:
            raise ValueError(f"预测长度必须为正整数，当前值为 {horizon}")

        self.horizon = int(horizon)
        self.eps = float(eps)
        # 论文使用跨通道共享的标量仿射参数。
        self.affine_scale = nn.Parameter(torch.ones(()))
        self.affine_shift = nn.Parameter(torch.zeros(()))
        # 原始门控 r=0，则 sigmoid(r)=0.5，避免门控的梯度陷阱。
        self.raw_gate = nn.Parameter(torch.zeros(()))
        self.log_scale_correction = nn.Parameter(torch.zeros(self.horizon))
        self.shift_correction = nn.Parameter(torch.zeros(self.horizon))
        self.drift_coefficient = nn.Parameter(torch.zeros(self.horizon))

    def normalize(
        self, values: torch.Tensor
    ) -> tuple[torch.Tensor, AdaptiveRevINState]:
        if values.ndim != 3:
            raise ValueError(
                "输入形状应为 [批量,时间,特征]，"
                f"实际得到 {tuple(values.shape)}"
            )
        if values.shape[1] < 2:
            raise ValueError("A-RevIN 至少需要两个历史时间步")

        # 与论文公式（1）—（4）一致，统计量和漂移特征停止梯度。
        mean = values.mean(dim=1, keepdim=True).detach()
        variance = values.var(dim=1, keepdim=True, unbiased=False).detach()
        std = torch.sqrt(variance + self.eps)
        midpoint = values.shape[1] // 2
        early_mean = values[:, :midpoint].mean(dim=1, keepdim=True).detach()
        recent_mean = values[:, midpoint:].mean(dim=1, keepdim=True).detach()
        drift = (recent_mean - early_mean) / std

        normalized = (values - mean) / std
        normalized = self.affine_scale * normalized + self.affine_shift
        return normalized, AdaptiveRevINState(mean=mean, std=std, drift=drift)

    def denormalize_feature(
        self,
        prediction: torch.Tensor,
        state: AdaptiveRevINState,
        feature_index: int,
    ) -> torch.Tensor:
        """将单个特征的归一化预测恢复到该特征的输入尺度。"""
        if prediction.ndim != 2 or prediction.shape[1] != self.horizon:
            raise ValueError(
                f"预测形状应为 [批量,{self.horizon}]，"
                f"实际得到 {tuple(prediction.shape)}"
            )

        mean = state.mean[:, 0, feature_index : feature_index + 1]
        std = state.std[:, 0, feature_index : feature_index + 1]
        drift = state.drift[:, 0, feature_index : feature_index + 1]
        gate = torch.sigmoid(self.raw_gate)
        scale = torch.exp(gate * self.log_scale_correction).unsqueeze(0)
        shift = gate * (
            self.shift_correction.unsqueeze(0) * std
            + self.drift_coefficient.unsqueeze(0) * drift * std
        )
        restored = std * (prediction - self.affine_shift) / self.affine_scale
        return scale * restored + mean + shift

    @property
    def gate(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_gate)
