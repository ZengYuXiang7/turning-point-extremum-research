from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from models.multi_turbine_impl.multi_turbine import (
    MultiHorizonPowerHead,
    MultiTurbineRevIN,
    relation_attention_mask,
)
from models.multi_turbine_impl.satra_layers import (
    CrossTurbineTransformer,
    StateConditionedExperts,
    TemporalConvEncoder,
)


class PatchCNNMoE(nn.Module):
    """以共享 CNN-MoE 独立处理每台风机的每个时间 patch。"""

    def __init__(self, hidden_dim: int, patch_length: int, depth: int, kernel_size: int, top_k: int, dropout: float,) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.patch_length = int(patch_length)
        self.encoder = TemporalConvEncoder(hidden_dim, depth, kernel_size)
        self.dropout = nn.Dropout(dropout)
        self.expert_dim = self.hidden_dim * 4
        self.expand = nn.Linear(self.hidden_dim, self.expert_dim)
        self.experts = StateConditionedExperts(self.patch_length, self.expert_dim, top_k, dropout,)
        self.reduce = nn.Linear(self.expert_dim, self.hidden_dim)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        batch_size, turbine_count, patch_count, patch_length, hidden_dim = patches.shape
        patch_batch = patches.reshape(batch_size * turbine_count * patch_count, patch_length, hidden_dim,)

        # 每个 patch 先提取局部时序模式，再按自身完整状态选择专家。
        encoded = self.encoder(patch_batch.transpose(1, 2)).transpose(1, 2)
        encoded = self.dropout(encoded)
        expanded = self.expand(encoded)
        routed = self.experts(expanded.reshape(patch_batch.shape[0], -1))
        reduced = self.reduce(routed.reshape(patch_batch.shape[0], patch_length, self.expert_dim,))

        output = reduced.reshape(batch_size, turbine_count, patch_count, patch_length, hidden_dim,)
        return output


class PatchHistoryEncoder(nn.Module):
    """切分重叠 patch，执行共享 CNN-MoE，并平均合并回历史时间轴。"""

    def __init__(self, hidden_dim: int, history_steps: int, patch_length: int, patch_stride: int, depth: int, kernel_size: int, top_k: int, dropout: float,) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.history_steps = int(history_steps)
        self.patch_length = int(patch_length)
        self.patch_stride = int(patch_stride)
        remaining_steps = self.history_steps - self.patch_length
        self.patch_count = (remaining_steps + self.patch_stride - 1) // self.patch_stride + 1
        self.padded_history_steps = (self.patch_count - 1) * self.patch_stride + self.patch_length
        self.padding_steps = self.padded_history_steps - self.history_steps
        self.patch_encoder = PatchCNNMoE(hidden_dim, patch_length, depth, kernel_size, top_k, dropout,)

        # overlap-add 后按每个时间点被覆盖的次数求平均。
        coverage = torch.zeros(self.padded_history_steps)
        for patch_index in range(self.patch_count):
            patch_start = patch_index * self.patch_stride
            patch_end = patch_start + self.patch_length
            coverage[patch_start:patch_end] += 1
        self.register_buffer("coverage", coverage.reshape(1, 1, -1, 1), persistent=False,)

    def split_patches(self, representation: torch.Tensor) -> torch.Tensor:
        # 右侧复制最后一个时间点，使末端也进入一个完整 patch。
        channel_first = representation.permute(0, 1, 3, 2)
        padded = F.pad(channel_first, (0, self.padding_steps, 0, 0), mode="replicate")
        patches = padded.unfold(3, self.patch_length, self.patch_stride)
        patches = patches.permute(0, 1, 3, 4, 2).contiguous()
        return patches

    def merge_patches(self, patches: torch.Tensor) -> torch.Tensor:
        batch_size, turbine_count, _, _, hidden_dim = patches.shape
        fold_input = patches.permute(0, 1, 4, 3, 2).contiguous()
        fold_input = fold_input.reshape(batch_size * turbine_count, hidden_dim * self.patch_length, self.patch_count,)

        # fold 在重叠位置求和，再用固定覆盖次数还原时间表征。
        merged = F.fold(fold_input, output_size=(1, self.padded_history_steps), kernel_size=(1, self.patch_length), stride=(1, self.patch_stride),)
        merged = merged.reshape(batch_size, turbine_count, hidden_dim, self.padded_history_steps,)
        merged = merged.permute(0, 1, 3, 2)
        merged = merged / self.coverage.to(merged.dtype)
        output = merged[:, :, : self.history_steps, :]
        return output

    def forward(self, representation: torch.Tensor) -> torch.Tensor:
        patches = self.split_patches(representation)
        encoded_patches = self.patch_encoder(patches)
        output = self.merge_patches(encoded_patches)
        return output


class Model2Backbone(nn.Module):
    """先做单风机 patch CNN-MoE，再融合风机空间关系的联合预测模型。"""

    def __init__(self, channels: int, power_index: int, history_steps: int, horizon: int, use_revin: bool, patch_length: int, patch_stride: int, use_spatial_relation: bool = True, relation_weight: torch.Tensor | None = None, use_dtw_prior: bool = False, hidden_dim: int = 64, depth: int = 10, kernel_size: int = 3, heads: int = 4, tower_layers: int = 1, dropout: float = 0.1, expert_top_k: int = 3,) -> None:
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
        self.patch_history_encoder = PatchHistoryEncoder(hidden_dim, history_steps, patch_length, patch_stride, depth, kernel_size, expert_top_k, dropout,)

        self.semantic_towers = nn.ModuleList()
        if self.use_spatial_relation:
            for _ in range(tower_layers):
                self.semantic_towers.append(CrossTurbineTransformer(hidden_dim, heads, dropout))

        self.prior_towers = nn.ModuleList()
        if self.use_dtw_prior:
            for _ in range(tower_layers):
                self.prior_towers.append(CrossTurbineTransformer(hidden_dim, heads, dropout))
            self.register_buffer("prior_attention_mask", relation_attention_mask(relation_weight), persistent=False,)

        self.prediction_head = MultiHorizonPowerHead(hidden_dim, history_steps, horizon)

    def encode_relation(self, representation: torch.Tensor) -> torch.Tensor:
        if not self.use_spatial_relation:
            return representation
        batch_size, turbine_count, history_steps, hidden_dim = representation.shape
        turbine_tokens = representation.permute(0, 2, 1, 3)
        turbine_tokens = turbine_tokens.reshape(batch_size * history_steps, turbine_count, hidden_dim,)

        # patch 合并后，在每个时间点学习全场风机关系。
        semantic_state = turbine_tokens
        for tower in self.semantic_towers:
            semantic_state = tower(semantic_state, attention_mask=None)

        fused = semantic_state
        if self.use_dtw_prior:
            prior_state = turbine_tokens
            for tower in self.prior_towers:
                prior_state = tower(prior_state, attention_mask=self.prior_attention_mask,)
            fused = semantic_state + prior_state

        output = fused.reshape(batch_size, history_steps, turbine_count, hidden_dim,)
        output = output.permute(0, 2, 1, 3)
        return output

    def encode_history(self, past: torch.Tensor) -> torch.Tensor:
        embedded = self.input_projection(past)
        patch_encoded = self.patch_history_encoder(embedded)
        representation = self.encode_relation(patch_encoded)
        return representation

    def encode_masked_history(self, past: torch.Tensor, token_mask: torch.Tensor, mask_token: torch.Tensor,) -> torch.Tensor:
        embedded = self.input_projection(past)
        masked = torch.where(token_mask.unsqueeze(-1), mask_token.to(embedded.dtype), embedded,)
        patch_encoded = self.patch_history_encoder(masked)
        representation = self.encode_relation(patch_encoded)
        return representation

    def forward(self, past: torch.Tensor, turbine_id: torch.Tensor) -> torch.Tensor:
        model_input = past
        if self.use_revin:
            model_input, instance_mean, instance_std = self.revin.normalize(past)

        representation = self.encode_history(model_input)
        forecast = self.prediction_head(representation)

        if self.use_revin:
            forecast = self.revin.denormalize_power(forecast, instance_mean, instance_std, self.power_index,)
        return forecast
