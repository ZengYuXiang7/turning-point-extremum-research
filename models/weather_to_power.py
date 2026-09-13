from __future__ import annotations

import torch
from torch import nn

from config.settings import WEATHER_COLUMNS


class WeatherToPowerModel(nn.Module):
    """同时刻：天气相关特征 → 功率。"""

    def __init__(self, hidden: int = 128) -> None:
        super().__init__()
        weather_dim = len(WEATHER_COLUMNS)
        self.turbine_embedding = nn.Embedding(16, hidden)
        self.net = nn.Sequential(
            nn.Linear(weather_dim + hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        past_unused: torch.Tensor,
        weather: torch.Tensor,
        turbine_id: torch.Tensor,
    ) -> torch.Tensor:
        # weather: [B, 13]
        turbine = self.turbine_embedding(turbine_id)
        features = torch.cat([weather, turbine], dim=-1)
        return self.net(features)
