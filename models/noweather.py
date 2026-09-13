from __future__ import annotations

import torch
from torch import nn

from config.settings import (
    HISTORY_COLUMNS,
    HISTORY_STEPS,
    PATCH_LENGTHS,
    POWER_INDEX,
)
from layers.Embed import Emb
from models.backbone.dlinear import DLinear
from models.backbone.patch_config import PatchConfig
from models.backbone.patchmlp import Encoder as PatchEncoder
from models.backbone.patchmlp import series_decomp as PatchDecomposition


class RevIN(nn.Module):
    """沿历史时间维执行可逆实例归一化。"""

    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = float(eps)
        self.affine_weight = nn.Parameter(torch.ones(1, 1, channels))
        self.affine_bias = nn.Parameter(torch.zeros(1, 1, channels))

    def normalize(
        self,
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 每个样本使用自身历史窗统计量
        instance_mean = values.mean(dim=1, keepdim=True).detach()
        centered = values - instance_mean
        instance_std = torch.sqrt(
            centered.square().mean(dim=1, keepdim=True) + self.eps
        ).detach()
        normalized = centered / instance_std
        normalized = normalized * self.affine_weight + self.affine_bias
        return normalized, instance_mean, instance_std

    def denormalize(
        self,
        normalized: torch.Tensor,
        instance_mean: torch.Tensor,
        instance_std: torch.Tensor,
    ) -> torch.Tensor:
        # 使用同一样本的历史统计量恢复全部输出特征
        restored = (normalized - self.affine_bias) / (
            self.affine_weight + self.eps * self.eps
        )
        restored = restored * instance_std + instance_mean
        return restored


class NoFutureWeatherModel(nn.Module):
    """仅使用历史序列预测未来功率。"""

    def __init__(
        self,
        model_name: str,
        horizon: int,
        channels: int,
        power_index: int,
    ) -> None:
        super().__init__()
        self.model_name = model_name.lower()
        self.horizon = int(horizon)
        self.power_index = int(power_index)
        self.revin = RevIN(channels)

        if self.model_name in ("patchmlp", "patchmlpallfeatures"):
            config = PatchConfig(seq_len=HISTORY_STEPS, pred_len=self.horizon)
            self.patch_embedding = Emb(
                HISTORY_STEPS,
                config.d_model,
                patch_len=PATCH_LENGTHS,
            )
            self.patch_decomposition = PatchDecomposition(13)
            self.seasonal_encoders = nn.ModuleList()
            self.trend_encoders = nn.ModuleList()
            for _ in range(config.e_layers):
                self.seasonal_encoders.append(
                    PatchEncoder(config.d_model, channels)
                )
                self.trend_encoders.append(
                    PatchEncoder(config.d_model, channels)
                )
            self.output_projection = nn.Linear(
                config.d_model,
                self.horizon,
                bias=True,
            )
            self.turbine_embedding = nn.Embedding(16, config.d_model)
        elif self.model_name == "dlinear":
            self.backbone = DLinear(HISTORY_STEPS, self.horizon, channels)
            self.turbine_embedding = nn.Embedding(16, channels)
        elif self.model_name == "dlinearallfeatures":
            self.backbone = DLinear(HISTORY_STEPS, self.horizon, channels)
            self.turbine_embedding = nn.Embedding(16, channels)
            self.feature_projection = nn.Linear(channels, 1)
            nn.init.zeros_(self.feature_projection.weight)
            nn.init.zeros_(self.feature_projection.bias)
            with torch.no_grad():
                self.feature_projection.weight[0, self.power_index] = 1.0

        nn.init.normal_(self.turbine_embedding.weight, mean=0.0, std=0.02)

    def forward(
        self,
        past: torch.Tensor,
        turbine_id: torch.Tensor,
    ) -> torch.Tensor:
        # 在 RevIN 域内编码时序和风机身份
        normalized_past, instance_mean, instance_std = self.revin.normalize(past)

        if self.model_name in ("patchmlp", "patchmlpallfeatures"):
            embedded = self.patch_embedding(normalized_past.permute(0, 2, 1))
            turbine_embedding = self.turbine_embedding(turbine_id).unsqueeze(1)
            encoder_input = embedded + turbine_embedding
            seasonal, trend = self.patch_decomposition(encoder_input)

            for encoder in self.seasonal_encoders:
                seasonal = encoder(seasonal)
            for encoder in self.trend_encoders:
                trend = encoder(trend)

            encoded = seasonal + trend
            normalized_forecast = self.output_projection(encoded).permute(0, 2, 1)
        else:
            turbine_embedding = self.turbine_embedding(turbine_id).unsqueeze(1)
            encoder_input = normalized_past + turbine_embedding
            normalized_forecast = self.backbone(encoder_input)

        # 将预测恢复到历史输入的特征尺度
        forecast = self.revin.denormalize(
            normalized_forecast,
            instance_mean,
            instance_std,
        )
        if self.model_name == "dlinearallfeatures":
            power_forecast = self.feature_projection(forecast).squeeze(-1)
        else:
            power_forecast = forecast[:, :, self.power_index]
        return power_forecast
