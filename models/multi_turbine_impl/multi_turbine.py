from __future__ import annotations

import torch
from torch import nn

from models.multi_turbine_impl.satra_layers import (
    CrossTurbineTransformer,
    StateConditionedPatternExpert,
)


def relation_attention_mask(relation_weight: torch.Tensor) -> torch.Tensor:
    """将允许注意的风机关联矩阵转换为 PyTorch 屏蔽矩阵。"""
    allowed = relation_weight.to(dtype=torch.bool)
    turbine_count = allowed.shape[0]
    identity = torch.eye(turbine_count, device=allowed.device, dtype=torch.bool)
    allowed = allowed | identity
    mask = ~allowed
    return mask


class MultiTurbineRevIN(nn.Module):
    """沿历史时间维独立归一化每个样本、每台风机。"""

    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = float(eps)
        self.affine_weight = nn.Parameter(torch.ones(1, 1, 1, channels))
        self.affine_bias = nn.Parameter(torch.zeros(1, 1, 1, channels))

    def normalize(self, values: torch.Tensor,) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 每台风机只使用自身历史窗口的统计量。
        instance_mean = values.mean(dim=2, keepdim=True).detach()
        centered = values - instance_mean
        instance_std = torch.sqrt(centered.square().mean(dim=2, keepdim=True) + self.eps).detach()
        normalized = centered / instance_std
        normalized = normalized * self.affine_weight + self.affine_bias
        return normalized, instance_mean, instance_std

    def denormalize_power(self, normalized: torch.Tensor, instance_mean: torch.Tensor, instance_std: torch.Tensor, power_index: int,) -> torch.Tensor:
        # 功率预测使用对应风机历史功率通道的统计量恢复尺度。
        power_weight = self.affine_weight[..., power_index]
        power_bias = self.affine_bias[..., power_index]
        power_mean = instance_mean[..., power_index]
        power_std = instance_std[..., power_index]
        restored = (normalized - power_bias) / (power_weight + self.eps * self.eps)
        restored = restored * power_std + power_mean
        return restored


class MultiHorizonPowerHead(nn.Module):
    """先投影特征，再把历史时间轴投影到多步预测。"""

    def __init__(self, hidden_dim: int, history_steps: int, horizon: int) -> None:
        super().__init__()
        self.feature_to_power = nn.Linear(hidden_dim, 1)
        self.history_to_horizon = nn.Linear(history_steps, horizon)

    def forward(self, representation: torch.Tensor) -> torch.Tensor:
        power_history = self.feature_to_power(representation)
        forecast = self.history_to_horizon(power_history.transpose(-1, -2))
        forecast = forecast.squeeze(-2)
        return forecast


class MultiTurbineBackbone(nn.Module):
    """使用 SATRA PSTR-Net 设计的多风机联合面板预测模型。"""

    def __init__(self, channels: int, power_index: int, history_steps: int, horizon: int, use_revin: bool, use_spatial_relation: bool = True, relation_weight: torch.Tensor | None = None, use_dtw_prior: bool = False, hidden_dim: int = 64, depth: int = 10, kernel_size: int = 3, heads: int = 4, tower_layers: int = 1, dropout: float = 0.1, expert_top_k: int = 3,) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.channels = int(channels)
        self.power_index = int(power_index)
        self.history_steps = int(history_steps)
        self.use_revin = bool(use_revin)
        self.use_spatial_relation = bool(use_spatial_relation)
        self.use_dtw_prior = bool(use_dtw_prior)
        if self.use_revin:
            self.revin = MultiTurbineRevIN(self.channels)
        self.input_projection = nn.Linear(self.channels, self.hidden_dim)

        self.spe = StateConditionedPatternExpert(self.hidden_dim, self.history_steps, depth, kernel_size, expert_top_k, dropout,)
        self.semantic_towers = nn.ModuleList()
        if self.use_spatial_relation:
            for _ in range(tower_layers):
                tower = CrossTurbineTransformer(self.hidden_dim, heads, dropout)
                self.semantic_towers.append(tower)

        self.prior_towers = nn.ModuleList()
        if self.use_dtw_prior:
            for _ in range(tower_layers):
                tower = CrossTurbineTransformer(self.hidden_dim, heads, dropout)
                self.prior_towers.append(tower)
            self.register_buffer("prior_attention_mask", relation_attention_mask(relation_weight), persistent=False,)
            
        self.prediction_head = MultiHorizonPowerHead(self.hidden_dim, self.history_steps, horizon,)

    def encode_relation(self, representation: torch.Tensor) -> torch.Tensor:
        if not self.use_spatial_relation:
            return representation
        batch_size, turbine_count, history_steps, hidden_dim = representation.shape
        turbine_tokens = representation.permute(0, 2, 1, 3)
        turbine_tokens = turbine_tokens.reshape(batch_size * history_steps, turbine_count, hidden_dim,)

        # 全局语义塔学习每个时刻的全场关联。
        semantic_state = turbine_tokens
        for tower in self.semantic_towers:
            semantic_state = tower(semantic_state, attention_mask=None)

        fused = semantic_state
        if self.use_dtw_prior:
            # 先验塔仅在训练段 DTW 关系支持内学习注意力权重。
            prior_state = turbine_tokens
            for tower in self.prior_towers:
                prior_state = tower(prior_state, attention_mask=self.prior_attention_mask,)
            fused = semantic_state + prior_state

        output = fused.reshape(batch_size, history_steps, turbine_count, hidden_dim,)
        output = output.permute(0, 2, 1, 3)
        return output

    def encode_history(self, past: torch.Tensor) -> torch.Tensor:
        embedded = self.input_projection(past)
        specialized = self.spe(embedded)
        representation = self.encode_relation(specialized)
        return representation

    def encode_masked_history(self, past: torch.Tensor, token_mask: torch.Tensor, mask_token: torch.Tensor,) -> torch.Tensor:
        embedded = self.input_projection(past)
        masked = torch.where(token_mask.unsqueeze(-1), mask_token.to(embedded.dtype), embedded,)
        specialized = self.spe(masked)
        representation = self.encode_relation(specialized)
        return representation

    def forward(self, past: torch.Tensor, turbine_id: torch.Tensor,) -> torch.Tensor:
        model_input = past
        if self.use_revin:
            model_input, instance_mean, instance_std = self.revin.normalize(past)

        representation = self.encode_history(model_input)
        forecast = self.prediction_head(representation)
        
        if self.use_revin:
            forecast = self.revin.denormalize_power(forecast, instance_mean, instance_std, self.power_index,)
        return forecast
