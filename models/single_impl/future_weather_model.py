from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

import config as project_config
from config import (
    HISTORY_COLUMNS,
    PATCH_LENGTHS,
    POWER_INDEX,
    TEMPORAL_POOL_STEPS,
    WEATHER_COLUMNS,
)
from models.single_impl.backbone.dlinear import DLinear
from models.single_impl.backbone.embed import Emb
from models.single_impl.backbone.patch_config import PatchConfig
from models.single_impl.backbone.patchmlp import Model as OfficialPatchMLP


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


class FutureWeatherModel(nn.Module):
    """历史序列和未来天气到未来功率。"""

    def __init__(self, model_name: str, horizon: int, hidden: int = 192) -> None:
        # horizon 为当前采样粒度下的步数
        super().__init__()
        self.model_name = model_name.lower()
        self.horizon = int(horizon)

        if self.model_name == "patchmlp":
            config = PatchConfig(
                seq_len=project_config.HISTORY_STEPS,
                pred_len=self.horizon,
            )
            self.backbone = OfficialPatchMLP(config)
            self.backbone.emb = Emb(
                project_config.HISTORY_STEPS,
                config.d_model,
                patch_len=PATCH_LENGTHS,
            )
        elif self.model_name == "dlinear":
            self.backbone = DLinear(
                project_config.HISTORY_STEPS,
                self.horizon,
                len(HISTORY_COLUMNS),
            )

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
            forecast = self.backbone(past, None, None, None)
        else:
            forecast = self.backbone(past)

        base_curve = forecast[:, :, POWER_INDEX]
        return base_curve

    def forward(
        self,
        past: torch.Tensor,
        future_weather: torch.Tensor,
        turbine_id: torch.Tensor,
    ) -> torch.Tensor:
        base_curve = self.base_curve(past)
        turbine = self.turbine_embedding(turbine_id).unsqueeze(1)
        past_tokens = self.past_projection(past) + turbine
        past_tokens = F.avg_pool1d(
            past_tokens.transpose(1, 2),
            kernel_size=TEMPORAL_POOL_STEPS,
            stride=TEMPORAL_POOL_STEPS,
        ).transpose(1, 2)

        future_tokens = self.future_selection(future_weather) + turbine
        initial_state = self.initial_state(past_tokens.mean(dim=1)).unsqueeze(0)
        decoded, _ = self.future_decoder(future_tokens, initial_state)
        attended, _ = self.attention(decoded, past_tokens, past_tokens, need_weights=False)

        base_tokens = self.base_projection(base_curve.unsqueeze(-1))
        joined = torch.cat([decoded, attended, base_tokens], dim=-1)
        representation = self.norm(decoded + self.gate(joined) * self.fusion(joined))
        forecast = base_curve + self.head(representation).squeeze(-1)
        return forecast
