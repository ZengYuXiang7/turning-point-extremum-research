from __future__ import annotations

import os
from pathlib import Path

import torch
from torch import nn

from config import HISTORY_COLUMNS, POWER_INDEX, PROJECT_ROOT, WEATHER_COLUMNS
from models.single_impl.timer_weather_adapter import (
    MultivariatePatchAdapter,
    PatchForecastHead,
)

# Keep Hugging Face dynamic modules and metadata inside the only writable project tree.
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "models" / "pretrained" / ".hf-cache"))
os.environ.setdefault("HF_MODULES_CACHE", str(PROJECT_ROOT / "models" / "pretrained" / ".hf-modules"),)

from transformers import AutoModelForCausalLM  # noqa: E402
from transformers.utils import logging as transformers_logging  # noqa: E402

transformers_logging.set_verbosity_error()


class TimerWeatherMLP(nn.Module):
    """Pre-trained Timer with concatenated history/weather tokens and an MLP head."""

    def __init__(self, horizon: int, pretrained_path: str = "", patch_length: int = 96, bottleneck: int = 512, unfreeze_layers: int = 2, gradient_checkpointing: bool = True, residual_forecast: bool = True,) -> None:
        super().__init__()
        model_path = (
            Path(pretrained_path)
            if pretrained_path
            else PROJECT_ROOT / "models" / "pretrained" / "timer-base-84m"
        )
        self.horizon = int(horizon)
        self.patch_length = int(patch_length)
        self.unfreeze_layers = int(unfreeze_layers)
        self.residual_forecast = bool(residual_forecast)

        self.timer = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, local_files_only=True, dtype=torch.float32, low_cpu_mem_usage=True,)
        self.timer.config.use_cache = False
        self.backbone = self.timer.get_decoder()
        self._rebuild_rotary_buffers()

        configured_patch = int(getattr(self.timer.config, "input_token_len", 0))
        if configured_patch != self.patch_length:
            raise ValueError(f"Timer checkpoint patch length is {configured_patch}, " f"but --timer-patch-length is {self.patch_length}")

        for parameter in self.timer.parameters():
            parameter.requires_grad_(False)

        layers = getattr(self.backbone, "layers", None)
        if layers is None:
            raise AttributeError("Timer backbone does not expose decoder layers")
        if not 0 <= self.unfreeze_layers <= len(layers):
            raise ValueError(f"unfreeze_layers must be in [0,{len(layers)}], got {self.unfreeze_layers}")
        if self.unfreeze_layers:
            for layer in layers[-self.unfreeze_layers :]:
                for parameter in layer.parameters():
                    parameter.requires_grad_(True)
            for parameter in self.backbone.norm.parameters():
                parameter.requires_grad_(True)

        if gradient_checkpointing and self.unfreeze_layers:
            try:
                self.timer.gradient_checkpointing_enable()
            except (AttributeError, ValueError):
                # Older Timer remote code exposes the switch on its decoder directly.
                self.backbone.gradient_checkpointing = False

        hidden_size = int(self.timer.config.hidden_size)
        self.history_adapter = MultivariatePatchAdapter(len(HISTORY_COLUMNS), self.patch_length, hidden_size)
        self.weather_adapter = MultivariatePatchAdapter(len(WEATHER_COLUMNS), self.patch_length, hidden_size)
        self.token_type_embedding = nn.Embedding(2, hidden_size)
        self.turbine_embedding = nn.Embedding(16, hidden_size)
        self.valid_fraction_embedding = nn.Linear(1, hidden_size, bias=False)
        self.input_norm = nn.LayerNorm(hidden_size)
        self.forecast_head = PatchForecastHead(hidden_size=hidden_size, bottleneck=int(bottleneck), patch_length=self.patch_length,)

    def _rebuild_rotary_buffers(self) -> None:
        """Repair non-persistent RoPE buffers after Transformers meta loading."""
        head_dim = int(self.timer.config.hidden_size) // int(self.timer.config.num_attention_heads)
        theta = float(getattr(self.timer.config, "rope_theta", 10000.0))
        max_positions = int(self.timer.config.max_position_embeddings)
        inv_freq = 1.0 / (
            theta
            ** (
                torch.arange(0, head_dim, 2, dtype=torch.float32)
                / float(head_dim)
            )
        )
        for layer in self.backbone.layers:
            rotary = layer.self_attn.rotary_emb
            rotary.inv_freq = inv_freq.clone()
            rotary._set_cos_sin_cache(seq_len=max_positions, device=rotary.inv_freq.device, dtype=torch.float32,)

    def forward(self, past: torch.Tensor, future_weather: torch.Tensor, turbine_id: torch.Tensor,) -> torch.Tensor:
        history_tokens, history_valid = self.history_adapter(past)
        weather_tokens, weather_valid = self.weather_adapter(future_weather)
        batch_size = past.shape[0]
        history_count = history_tokens.shape[1]
        future_count = weather_tokens.shape[1]

        tokens = torch.cat([history_tokens, weather_tokens], dim=1)
        token_types = torch.cat([ torch.zeros(history_count, dtype=torch.long, device=past.device), torch.ones(future_count, dtype=torch.long, device=past.device), ])
        valid_fraction = torch.cat([history_valid, weather_valid], dim=0)
        tokens = tokens + self.token_type_embedding(token_types).unsqueeze(0)
        tokens = tokens + self.valid_fraction_embedding(valid_fraction).unsqueeze(0)
        tokens = tokens + self.turbine_embedding(turbine_id).unsqueeze(1)
        tokens = self.input_norm(tokens)
        backbone_dtype = next(self.backbone.parameters()).dtype
        tokens = tokens.to(dtype=backbone_dtype)

        encoded = self.backbone(inputs_embeds=tokens, use_cache=False, return_dict=True,).last_hidden_state
        future_hidden = encoded[:, -future_count:, :]
        forecast = self.forecast_head(future_hidden).reshape(batch_size, -1)
        forecast = forecast[:, : self.horizon]

        if self.residual_forecast:
            forecast = forecast + past[:, -1, POWER_INDEX].unsqueeze(-1)
        return forecast

    def parameter_groups(self, adapter_lr: float, backbone_lr: float):
        adapter_parameters = []
        backbone_parameters = []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith("timer.") or name.startswith("backbone."):
                backbone_parameters.append(parameter)
            else:
                adapter_parameters.append(parameter)

        groups = [{"params": adapter_parameters, "lr": float(adapter_lr)}]
        if backbone_parameters:
            groups.append({"params": backbone_parameters, "lr": float(backbone_lr)})
        return groups
