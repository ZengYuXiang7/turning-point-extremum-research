from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from transformers import AutoModel

from config import HISTORY_COLUMNS, POWER_INDEX, PROJECT_ROOT
from models.single_impl.qwen_mlp import (
    NumericPatchTokenizer,
    ResidualForecastHead,
)


class QwenMLP(nn.Module):
    """Frozen Qwen2.5 encoder plus a trainable numeric adapter and MLP head."""

    def __init__(self, horizon: int, pretrained_path: str = "", patch_length: int = 4, patch_stride: int = 2, bottleneck: int = 512, gradient_checkpointing: bool = True,) -> None:
        super().__init__()
        if pretrained_path:
            model_path = Path(pretrained_path)
        else:
            model_path = PROJECT_ROOT / "models" / "pretrained" / "Qwen2.5-1.5B"

        self.llm = AutoModel.from_pretrained(model_path, local_files_only=True, dtype=torch.bfloat16, low_cpu_mem_usage=True,)
        self.llm.config.use_cache = False
        for parameter in self.llm.parameters():
            parameter.requires_grad_(False)
        if gradient_checkpointing:
            self.llm.gradient_checkpointing_enable()

        hidden_size = int(self.llm.config.hidden_size)
        self.patch_tokenizer = NumericPatchTokenizer(input_channels=len(HISTORY_COLUMNS), hidden_size=hidden_size, patch_length=patch_length, patch_stride=patch_stride,)
        self.turbine_embedding = nn.Embedding(16, hidden_size)
        self.summary_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        nn.init.normal_(self.summary_token, mean=0.0, std=0.02)
        self.pool_gate = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size), nn.Sigmoid(),)
        self.forecast_head = ResidualForecastHead(hidden_size=hidden_size, bottleneck=bottleneck, horizon=int(horizon),)

    def train(self, mode: bool = True):
        super().train(mode)
        self.llm.eval()
        return self

    def forward(self, past: torch.Tensor, turbine_id: torch.Tensor,) -> torch.Tensor:
        tokens = self.patch_tokenizer(past)
        turbine = self.turbine_embedding(turbine_id).unsqueeze(1)
        tokens = tokens + turbine
        summary = self.summary_token.expand(tokens.shape[0], -1, -1) + turbine
        tokens = torch.cat([tokens, summary], dim=1)

        encoded = self.llm(inputs_embeds=tokens, use_cache=False, return_dict=True,).last_hidden_state
        last = encoded[:, -1]
        mean = encoded[:, :-1].mean(dim=1)
        gate = self.pool_gate(torch.cat([last, mean], dim=-1))
        representation = gate * last + (1.0 - gate) * mean
        last_power = past[:, -1, POWER_INDEX]
        return self.forecast_head(representation, last_power)
