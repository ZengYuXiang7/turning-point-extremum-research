from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class MultivariatePatchAdapter(nn.Module):
    """Project a multivariate numeric patch into one Timer token."""

    def __init__(self, channels: int, patch_length: int, hidden_size: int) -> None:
        super().__init__()
        self.patch_length = int(patch_length)
        self.input_norm = nn.LayerNorm(int(channels))
        self.projection = nn.Conv1d(
            in_channels=int(channels),
            out_channels=int(hidden_size),
            kernel_size=self.patch_length,
            stride=self.patch_length,
        )
        self.activation = nn.GELU()
        self.output_norm = nn.LayerNorm(int(hidden_size))

        nn.init.normal_(self.projection.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.projection.bias)

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if values.ndim != 3:
            raise ValueError(f"expected [batch,time,channels], got {tuple(values.shape)}")

        length = int(values.shape[1])
        token_count = math.ceil(length / self.patch_length)
        padded_length = token_count * self.patch_length
        padding = padded_length - length

        values = self.input_norm(values)
        if padding:
            # Padding is applied after normalization, so zero represents a neutral value.
            values = F.pad(values, (0, 0, 0, padding), value=0.0)

        tokens = self.projection(values.transpose(1, 2)).transpose(1, 2)
        tokens = self.output_norm(self.activation(tokens))

        valid_fraction = values.new_ones((token_count, 1))
        if padding:
            valid_fraction[-1, 0] = (self.patch_length - padding) / self.patch_length
        return tokens, valid_fraction


class PatchForecastHead(nn.Module):
    """Decode each future-weather token into one power patch."""

    def __init__(self, hidden_size: int, bottleneck: int, patch_length: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(int(hidden_size)),
            nn.Linear(int(hidden_size), int(bottleneck)),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(int(bottleneck), int(patch_length)),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.network(hidden_states)
