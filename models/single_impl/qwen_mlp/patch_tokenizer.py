from __future__ import annotations

import torch
from torch import nn


class NumericPatchTokenizer(nn.Module):
    """Convert a multivariate history window into compact continuous tokens."""

    def __init__(self, input_channels: int, hidden_size: int, patch_length: int, patch_stride: int,) -> None:
        super().__init__()
        self.patch_length = int(patch_length)
        self.patch_stride = int(patch_stride)
        self.projection = nn.Conv1d(input_channels, hidden_size, kernel_size=self.patch_length, stride=self.patch_stride,)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        tokens = self.projection(history.transpose(1, 2)).transpose(1, 2)
        return self.norm(tokens)
