from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from config.settings import (
    HISTORY_COLUMNS,
    HISTORY_STEPS,
    POWER_INDEX,
    TEMPORAL_POOL_STEPS,
    WEATHER_COLUMNS,
)


def cross_sectional_z(values: torch.Tensor) -> torch.Tensor:
    # 同步风机截面标准化
    mean = values.mean(dim=0, keepdim=True)
    scale = values.std(dim=0, keepdim=True, unbiased=False)
    return ((values - mean) / scale).clamp(-6.0, 6.0)


class TemporalEncoder(nn.Module):
    """风电 StockEcho 使用的因果时序编码器。"""

    def __init__(self, channels: int, hidden_dim: int, dropout: float = 0.1, max_length: int = 64):
        super().__init__()
        self.input_projection = nn.Linear(channels, hidden_dim)
        # hidden=75 时 5 头整除
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=5,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        frequency = torch.exp(
            torch.arange(0, hidden_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / hidden_dim)
        )
        encoding = torch.zeros(max_length, hidden_dim, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(position * frequency)
        encoding[:, 1::2] = torch.cos(position * frequency[: encoding[:, 1::2].size(1)])
        self.register_buffer("position_encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        length = inputs.size(1)
        hidden = self.input_projection(inputs)
        hidden = hidden + self.position_encoding[:, :length].to(hidden.dtype)
        causal_mask = torch.triu(
            torch.ones(length, length, device=inputs.device, dtype=torch.bool),
            diagonal=1,
        )
        encoded = self.encoder(hidden, mask=causal_mask)
        tokens = self.output_norm(self.dropout(encoded) + hidden)
        return tokens[:, -1, :], tokens


class ReliableRelationEvidence(nn.Module):
    """多风机关系证据的投影与超参。"""

    def __init__(
        self,
        hidden_dim: int,
        match_window: int,
        max_lag: int,
        reliability_temperature: float,
        consensus_margin: float,
    ) -> None:
        super().__init__()
        self.match_window = int(match_window)
        self.max_lag = int(max_lag)
        self.reliability_temperature = float(reliability_temperature)
        self.consensus_margin = float(consensus_margin)
        relation_dim = 64
        if hidden_dim < 64:
            relation_dim = hidden_dim
        if relation_dim < 16:
            relation_dim = 16
        self.level_projection = nn.Sequential(
            nn.LayerNorm(self.match_window * 16),
            nn.Linear(self.match_window * 16, relation_dim, bias=False),
            nn.LayerNorm(relation_dim),
        )
        self.delta_projection = nn.Sequential(
            nn.LayerNorm((self.match_window - 1) * 16),
            nn.Linear((self.match_window - 1) * 16, relation_dim, bias=False),
            nn.LayerNorm(relation_dim),
        )


class DenseGraphConstructor(nn.Module):
    def __init__(self, temperature: float) -> None:
        super().__init__()
        self.log_temperature = nn.Parameter(torch.tensor(math.log(float(temperature))))


class ResponseTransport(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.response_norm = nn.LayerNorm(hidden_dim)
        self.response_value = nn.Linear(hidden_dim, hidden_dim, bias=False)


class AlignedResidualFusion(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        alignment_temperature: float,
        residual_gain: float,
        use_alignment: bool,
    ) -> None:
        super().__init__()
        self.use_alignment = bool(use_alignment)
        self.alignment_temperature = float(alignment_temperature)
        self.local_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.message_norm = nn.LayerNorm(hidden_dim)
        self.message_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.output_norm = nn.LayerNorm(hidden_dim)
        # residual_gain 映射到 logit，训练中可学习
        residual_gain = float(residual_gain)
        self.residual_gain_logit = nn.Parameter(
            torch.tensor(math.log(residual_gain / (1.0 - residual_gain)))
        )


class WindRelationFeatures(nn.Module):
    """把风电变量映射到 StockEcho 的 16 维关系空间。"""

    def __init__(self, channels: int = len(HISTORY_COLUMNS)) -> None:
        super().__init__()
        self.absolute_projection = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, 16),
            nn.GELU(),
            nn.LayerNorm(16),
        )
        self.mix_logit = nn.Parameter(torch.zeros(16))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        absolute = self.absolute_projection(inputs)
        if absolute.ndim == 4:
            mean = absolute.mean(dim=1, keepdim=True)
            scale = absolute.std(dim=1, keepdim=True, unbiased=False)
            relative = ((absolute - mean) / scale).clamp(-6.0, 6.0)
            mix = torch.sigmoid(self.mix_logit).view(1, 1, 1, -1)
        else:
            relative_steps = []
            for step in range(absolute.size(1)):
                relative_steps.append(cross_sectional_z(absolute[:, step, :]))
            relative = torch.stack(relative_steps, dim=1)
            mix = torch.sigmoid(self.mix_logit).view(1, 1, -1)
        return mix * absolute + (1.0 - mix) * relative


class StockEchoNoFutureWeather(nn.Module):
    """仅使用同步多风机历史关系预测未来功率。"""

    def __init__(self, horizon: int, hidden: int = 75) -> None:
        # horizon 为当前采样粒度下的步数；池化对齐约 4 分钟物理窗
        super().__init__()
        self.horizon = int(horizon)
        self.pool_factor = TEMPORAL_POOL_STEPS
        pooled_length = HISTORY_STEPS // self.pool_factor
        self.encoder = TemporalEncoder(len(HISTORY_COLUMNS), hidden, max_length=pooled_length)
        self.relation_features = WindRelationFeatures()
        self.relation = ReliableRelationEvidence(
            hidden_dim=hidden,
            match_window=5,
            max_lag=3,
            reliability_temperature=0.25,
            consensus_margin=0.05,
        )
        self.graph_constructor = DenseGraphConstructor(0.20)
        self.response_transport = ResponseTransport(hidden)
        self.fusion = AlignedResidualFusion(
            hidden_dim=hidden,
            alignment_temperature=0.20,
            residual_gain=0.20,
            use_alignment=True,
        )

        self.history_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(hidden, self.horizon),
        )

    @staticmethod
    def panel_center(values: torch.Tensor) -> torch.Tensor:
        return values - values.mean(dim=1, keepdim=True)

    def encode_relation_path(self, features: torch.Tensor, end_index: int, delta: bool) -> torch.Tensor:
        start = end_index - self.relation.match_window + 1
        path = features[:, :, start : end_index + 1, :]
        if delta:
            path = path[:, :, 1:, :] - path[:, :, :-1, :]
            projection = self.relation.delta_projection
        else:
            projection = self.relation.level_projection
        flat = path.reshape(path.size(0), path.size(1), -1)
        return F.normalize(projection(flat), dim=-1, eps=1e-6)

    def relation_fusion(self, pooled: torch.Tensor, hidden: torch.Tensor, tokens: torch.Tensor):
        # 可靠关系证据 → 稠密图 → 响应传递 → 对齐残差融合
        features = self.relation_features(pooled)
        batch, turbines, length, _ = features.shape
        lag_count = self.relation.max_lag
        panel_hidden = self.panel_center(hidden)
        valid = ~torch.eye(turbines, device=features.device, dtype=torch.bool)
        end_index = length - 1

        query_level = self.encode_relation_path(features, end_index, delta=False)
        query_delta = self.encode_relation_path(features, end_index, delta=True)
        scores = []
        for lag in range(1, lag_count + 1):
            candidate_level = self.encode_relation_path(features, end_index - lag, delta=False)
            candidate_delta = self.encode_relation_path(features, end_index - lag, delta=True)
            level_score = query_level @ candidate_level.transpose(-1, -2)
            delta_score = query_delta @ candidate_delta.transpose(-1, -2)
            reliability = torch.exp(
                -(level_score - delta_score).abs() / self.relation.reliability_temperature
            )
            score = 0.5 * (level_score + delta_score) * reliability
            scores.append(score.masked_fill(~valid.view(1, turbines, turbines), -1e4))

        score_stack = torch.stack(scores, dim=-1)
        valid_values = valid.to(score_stack.dtype).view(1, turbines, turbines, 1)
        consensus_score = (score_stack * valid_values).sum(dim=(1, 2))
        consensus_score = consensus_score / valid_values.sum(dim=(1, 2))
        consensus_lag = consensus_score.argmax(dim=-1)
        pair_score, pair_lag = score_stack.max(dim=-1)
        fallback_index = consensus_lag.view(batch, 1, 1, 1).expand(-1, turbines, turbines, 1)
        fallback_score = score_stack.gather(-1, fallback_index).squeeze(-1)
        use_pair_lag = (pair_score - fallback_score) > self.relation.consensus_margin
        lag_index = torch.where(
            use_pair_lag,
            pair_lag,
            consensus_lag.view(batch, 1, 1).expand(-1, turbines, turbines),
        )
        edge_score = score_stack.gather(-1, lag_index.unsqueeze(-1)).squeeze(-1)

        temperature = self.graph_constructor.log_temperature.exp()
        logits = (edge_score / temperature).masked_fill(
            ~valid.view(1, turbines, turbines), -1e4
        )
        graph = torch.softmax(logits, dim=-1) * valid.to(edge_score.dtype).view(
            1, turbines, turbines
        )
        graph = graph / graph.sum(dim=-1, keepdim=True)

        message = torch.zeros_like(panel_hidden)
        for lag_index_value in range(lag_count):
            lag = lag_index_value + 1
            response = tokens[:, :, -lag, :] - tokens[:, :, -lag - 1, :]
            response = self.panel_center(response)
            response = self.response_transport.response_value(
                self.response_transport.response_norm(response)
            )
            weights = graph * (lag_index == lag_index_value).to(graph.dtype)
            message = message + weights @ response
        message = self.panel_center(message)

        message_value = self.fusion.message_projection(self.fusion.message_norm(message))
        local_direction = F.normalize(self.fusion.local_projection(panel_hidden), dim=-1, eps=1e-6)
        message_direction = F.normalize(message_value, dim=-1, eps=1e-6)
        agreement = (local_direction * message_direction).sum(dim=-1, keepdim=True)
        confidence = torch.sigmoid(agreement / self.fusion.alignment_temperature)
        residual_gain = torch.sigmoid(self.fusion.residual_gain_logit)
        fused = self.fusion.output_norm(
            panel_hidden + residual_gain * confidence * message_value
        )
        fused = self.panel_center(fused)
        return fused

    def encode_history(self, past: torch.Tensor):
        # 编码历史时序与跨风机关系
        batch, turbines, steps, features = past.shape
        flat_past = past.reshape(batch * turbines, steps, features)
        pooled_flat = F.avg_pool1d(
            flat_past.transpose(1, 2),
            kernel_size=self.pool_factor,
            stride=self.pool_factor,
        ).transpose(1, 2)
        pooled = pooled_flat.reshape(batch, turbines, pooled_flat.size(1), features)
        hidden_flat, tokens_flat = self.encoder(pooled_flat)
        hidden = hidden_flat.reshape(batch, turbines, -1)
        tokens = tokens_flat.reshape(
            batch, turbines, tokens_flat.size(1), tokens_flat.size(2)
        )
        fused = self.relation_fusion(pooled, hidden, tokens)

        last_power = past[:, :, -1, POWER_INDEX : POWER_INDEX + 1]
        base_curve = last_power + self.history_head(fused)
        return base_curve, fused, tokens_flat, batch, turbines

    def forward(
        self,
        past: torch.Tensor,
        turbine_id: torch.Tensor,
    ) -> torch.Tensor:
        del turbine_id
        base_curve, _, _, _, _ = self.encode_history(past)
        return base_curve


class VariableSelectionNetwork(nn.Module):
    def __init__(self, variables: int, hidden: int) -> None:
        super().__init__()
        projections = []
        for _ in range(variables):
            projections.append(nn.Linear(1, hidden))
        self.projections = nn.ModuleList(projections)
        self.gate = nn.Sequential(
            nn.Linear(variables, hidden), nn.ELU(), nn.Linear(hidden, variables)
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.gate(values), dim=-1)
        projected_list = []
        for index, projection in enumerate(self.projections):
            projected_list.append(projection(values[..., index : index + 1]))
        projected = torch.stack(projected_list, dim=-2)
        selected = torch.sum(projected * weights.unsqueeze(-1), dim=-2)
        return selected


class StockEchoWindPower(StockEchoNoFutureWeather):
    """同步多风机历史关系和未来天气到未来功率。"""

    def __init__(self, horizon: int, hidden: int = 75) -> None:
        super().__init__(horizon, hidden)
        self.future_selection = VariableSelectionNetwork(len(WEATHER_COLUMNS), hidden)
        self.turbine_embedding = nn.Embedding(16, hidden)
        self.future_decoder = nn.GRU(hidden, hidden, batch_first=True)
        self.initial_state = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh())
        self.attention = nn.MultiheadAttention(hidden, 5, dropout=0.10, batch_first=True)
        self.base_projection = nn.Linear(1, hidden)
        self.gate = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.Sigmoid())
        self.weather_fusion = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.ELU(),
            nn.Dropout(0.10),
            nn.Linear(hidden, hidden),
        )
        self.output_norm = nn.LayerNorm(hidden)
        self.output_head = nn.Linear(hidden, 1)

    def forward(
        self,
        past: torch.Tensor,
        future_weather: torch.Tensor,
        turbine_id: torch.Tensor,
    ) -> torch.Tensor:
        base_curve, fused, tokens_flat, batch, turbines = self.encode_history(past)
        turbine_tokens = self.turbine_embedding(turbine_id).reshape(
            batch * turbines, 1, -1
        )
        weather_flat = future_weather.reshape(batch * turbines, self.horizon, -1)
        future_tokens = self.future_selection(weather_flat) + turbine_tokens
        fused_flat = fused.reshape(batch * turbines, -1)
        decoded, _ = self.future_decoder(
            future_tokens,
            self.initial_state(fused_flat).unsqueeze(0),
        )
        attended, _ = self.attention(decoded, tokens_flat, tokens_flat, need_weights=False)
        base_tokens = self.base_projection(
            base_curve.reshape(batch * turbines, self.horizon, 1)
        )
        joined = torch.cat([decoded, attended, base_tokens], dim=-1)
        representation = self.output_norm(
            decoded + self.gate(joined) * self.weather_fusion(joined)
        )
        weather_delta = self.output_head(representation).squeeze(-1)
        prediction = base_curve + weather_delta.reshape(batch, turbines, self.horizon)
        return prediction
