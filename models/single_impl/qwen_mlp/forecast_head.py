from __future__ import annotations

import torch
from torch import nn


class ResidualForecastHead(nn.Module):
    """Decode a pooled LLM representation as a correction to persistence."""

    def __init__(self, hidden_size: int, bottleneck: int, horizon: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, bottleneck), nn.GELU(), nn.Dropout(0.1), nn.Linear(bottleneck, horizon),)
        nn.init.normal_(self.layers[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, representation: torch.Tensor, last_power: torch.Tensor) -> torch.Tensor:
        correction = self.layers(representation)
        return last_power.unsqueeze(-1) + correction
