from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


EXPERT_BOTTLENECK_RATIOS = (1 / 16, 1 / 8, 1 / 4, 1 / 2, 3 / 4, 1)


class SamePadConv(nn.Module):
    """保持历史长度不变的一维时域卷积。"""

    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        padding = int(kernel_size) // 2
        self.conv = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.remove_last = int(kernel_size) % 2 == 0

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        convolved = self.conv(values)
        if self.remove_last:
            convolved = convolved[:, :, :-1]
        return convolved


class TemporalConvBlock(nn.Module):
    """两层残差时域卷积块。"""

    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        self.first = SamePadConv(channels, kernel_size)
        self.second = SamePadConv(channels, kernel_size)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        update = self.first(F.gelu(values))
        update = self.second(F.gelu(update))
        output = values + update
        return output


class TemporalConvEncoder(nn.Module):
    """沿单台风机历史轴提取局部时序模式。"""

    def __init__(self, hidden_dim: int, depth: int, kernel_size: int) -> None:
        super().__init__()
        blocks = []
        for _ in range(int(depth) + 1):
            blocks.append(TemporalConvBlock(hidden_dim, kernel_size))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        encoded = self.blocks(values)
        return encoded


class BottleneckExpert(nn.Module):
    """处理完整历史表征的一名异构瓶颈专家。"""

    def __init__(self, history_steps: int, model_dim: int, bottleneck_dim: int, dropout: float) -> None:
        super().__init__()
        self.history_steps = int(history_steps)
        self.model_dim = int(model_dim)
        self.net = nn.Sequential(nn.Linear(self.model_dim, bottleneck_dim), nn.GELU(), nn.Linear(bottleneck_dim, bottleneck_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(bottleneck_dim, self.model_dim),)

    def forward(self, flattened_history: torch.Tensor) -> torch.Tensor:
        history = flattened_history.reshape(-1, self.history_steps, self.model_dim)
        update = self.net(history)
        output = history + update
        return output.reshape(flattened_history.shape[0], self.history_steps * self.model_dim,)


class SparseExpertDispatcher:
    """按 Top-k 门控结果分发并聚合风机历史。"""

    def __init__(self, gates: torch.Tensor) -> None:
        self.gates = gates
        nonzero = torch.nonzero(gates, as_tuple=False)
        ordered = nonzero[torch.argsort(nonzero[:, 1])]
        self.sample_indices = ordered[:, 0]
        self.expert_indices = ordered[:, 1]
        self.part_sizes = (gates > 0).sum(dim=0).tolist()
        self.active_gates = gates[self.sample_indices, self.expert_indices]

    def dispatch(self, values: torch.Tensor):
        expert_inputs = torch.split(values[self.sample_indices], self.part_sizes, dim=0)
        return expert_inputs

    def combine(self, expert_outputs) -> torch.Tensor:
        joined = torch.cat(expert_outputs, dim=0)
        weighted = joined * self.active_gates.unsqueeze(-1).to(joined.dtype)
        output = joined.new_zeros((self.gates.shape[0], joined.shape[1]))
        output = output.index_add(0, self.sample_indices, weighted)
        return output


class StateConditionedExperts(nn.Module):
    """SATRA 原始 PSTR-Net 的六专家、状态条件 Top-3 路由。"""

    def __init__(self, history_steps: int, model_dim: int, top_k: int, dropout: float) -> None:
        super().__init__()
        bottleneck_dims = []
        for ratio in EXPERT_BOTTLENECK_RATIOS:
            bottleneck_dims.append(max(1, round(model_dim * ratio)))
        self.top_k = int(top_k)
        self.experts = nn.ModuleList([ BottleneckExpert(history_steps, model_dim, width, dropout) for width in bottleneck_dims ])
        router_width = int(history_steps) * int(model_dim)
        self.gate_weights = nn.Parameter(torch.zeros(router_width, len(bottleneck_dims)))
        self.noise_weights = nn.Parameter(torch.zeros(router_width, len(bottleneck_dims)))
        self.softplus = nn.Softplus()

    def forward(self, flattened_history: torch.Tensor) -> torch.Tensor:
        logits = flattened_history @ self.gate_weights
        if self.training:
            noise_scale = self.softplus(flattened_history @ self.noise_weights) + 1e-2
            logits = logits + torch.randn_like(logits) * noise_scale
        selected_logits, selected_indices = logits.topk(self.top_k, dim=1)
        selected_gates = torch.softmax(selected_logits.float(), dim=1).to(logits.dtype)
        gates = torch.zeros_like(logits).scatter(1, selected_indices, selected_gates)
        dispatcher = SparseExpertDispatcher(gates)
        expert_inputs = dispatcher.dispatch(flattened_history)
        expert_outputs = []
        for expert_index, expert in enumerate(self.experts):
            expert_outputs.append(expert(expert_inputs[expert_index]))
        output = dispatcher.combine(expert_outputs)
        return output


class StateConditionedPatternExpert(nn.Module):
    """SPE：时域卷积后按完整历史状态选择异构专家。"""

    def __init__(self, hidden_dim: int, history_steps: int, depth: int, kernel_size: int, top_k: int, dropout: float,) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.history_steps = int(history_steps)
        self.encoder = TemporalConvEncoder(hidden_dim, depth, kernel_size)
        self.dropout = nn.Dropout(dropout)
        self.expert_dim = self.hidden_dim * 4
        self.expand = nn.Linear(self.hidden_dim, self.expert_dim)
        self.experts = StateConditionedExperts(history_steps, self.expert_dim, top_k, dropout,)
        self.reduce = nn.Linear(self.expert_dim, self.hidden_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch_size, turbine_count, history_steps, hidden_dim = values.shape
        flattened = values.reshape(batch_size * turbine_count, history_steps, hidden_dim)
        
        encoded = self.encoder(flattened.transpose(1, 2)).transpose(1, 2)
        
        encoded = self.dropout(encoded)
        # 临时注释 experts 路由，先跑 encoder 直通效果：
        expanded = self.expand(encoded)
        routed = self.experts(expanded.reshape(batch_size * turbine_count, -1))
        output = self.reduce(routed.reshape(batch_size * turbine_count, history_steps, self.expert_dim))
        output = encoded.reshape(batch_size, turbine_count, history_steps, hidden_dim)
        return output


class CrossTurbineTransformer(nn.Module):
    """在一个时间点的风机截面上执行预归一化注意力。"""

    def __init__(self, hidden_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.first_norm = nn.RMSNorm(hidden_dim)
        self.second_norm = nn.RMSNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True,)
        self.dropout = nn.Dropout(dropout)
        self.feed_forward = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim * 2, hidden_dim), nn.Dropout(dropout),)

    def forward(self, values: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        normalized = self.first_norm(values)
        attended, _ = self.attention(normalized, normalized, normalized, attn_mask=attention_mask, need_weights=False,)
        updated = values + self.dropout(attended)
        output = updated + self.feed_forward(self.second_norm(updated))
        return output
