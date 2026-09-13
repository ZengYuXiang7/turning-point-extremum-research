from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from config.settings import (
    HISTORY_COLUMNS,
    HISTORY_STEPS,
    PATCH_LENGTHS,
    TEMPORAL_POOL_STEPS,
    WEATHER_COLUMNS,
)
from layers.Embed import Emb
from models.DLinear import DLinear
from models.PatchMLP import Model as OfficialPatchMLP


@dataclass
class PatchConfig:
    seq_len: int
    pred_len: int
    d_model: int = 320
    enc_in: int = len(HISTORY_COLUMNS)
    e_layers: int = 1
    output_attention: bool = False
    use_norm: bool = True


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
        return torch.sum(projected * weights.unsqueeze(-1), dim=-2)


class ForecastModel(nn.Module):
    """公共解码器；气象是否可见是唯一切景差异。"""

    def __init__(self, model_name: str, horizon: int, hidden: int = 192) -> None:
        # horizon 为当前采样粒度下的步数
        super().__init__()
        self.model_name = model_name.lower()
        self.horizon = int(horizon)

        if self.model_name == "patchmlp":
            config = PatchConfig(seq_len=HISTORY_STEPS, pred_len=self.horizon)
            self.backbone = OfficialPatchMLP(config)
            self.backbone.emb = Emb(HISTORY_STEPS, config.d_model, patch_len=PATCH_LENGTHS)
        elif self.model_name == "dlinear":
            self.backbone = DLinear(HISTORY_STEPS, self.horizon, len(HISTORY_COLUMNS))

        self.past_projection = nn.Linear(len(HISTORY_COLUMNS), hidden)
        self.future_selection = VariableSelectionNetwork(len(WEATHER_COLUMNS), hidden)
        self.turbine_embedding = nn.Embedding(16, hidden)
        self.future_decoder = nn.GRU(hidden, hidden, batch_first=True)
        self.initial_state = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh())
        self.attention = nn.MultiheadAttention(hidden, 4, dropout=0.1, batch_first=True)
        self.base_projection = nn.Linear(1, hidden)
        self.gate = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.Sigmoid())
        self.fusion = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
        )
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, 1)

    def base_curve(self, past: torch.Tensor) -> torch.Tensor:
        if self.model_name == "patchmlp":
            return self.backbone(past, None, None, None)[:, :, 0]
        return self.backbone(past)[:, :, 0]

    def forward(
        self, past: torch.Tensor, weather_or_zero: torch.Tensor, turbine_id: torch.Tensor
    ) -> torch.Tensor:
        base_curve = self.base_curve(past)

        turbine = self.turbine_embedding(turbine_id).unsqueeze(1)

        past_tokens = self.past_projection(past) + turbine

        past_tokens = F.avg_pool1d(
            past_tokens.transpose(1, 2),
            kernel_size=TEMPORAL_POOL_STEPS,
            stride=TEMPORAL_POOL_STEPS,
        ).transpose(1, 2)

        future_tokens = self.future_selection(weather_or_zero) + turbine


        decoded, _ = self.future_decoder(future_tokens, self.initial_state(past_tokens.mean(dim=1)).unsqueeze(0))


        attended, _ = self.attention(decoded, past_tokens, past_tokens, need_weights=False)

        base_tokens = self.base_projection(base_curve.unsqueeze(-1))
        joined = torch.cat([decoded, attended, base_tokens], dim=-1)

        representation = self.norm(decoded + self.gate(joined) * self.fusion(joined))
        return base_curve + self.head(representation).squeeze(-1)
