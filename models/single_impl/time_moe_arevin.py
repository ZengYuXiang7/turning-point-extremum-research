from __future__ import annotations

import os
from pathlib import Path

import torch
from torch import nn

from config import PROJECT_ROOT
from models.single_impl.adaptive_revin import AdaptiveRevIN

# 模型元数据和动态模块缓存统一保存在当前项目中。
os.environ.setdefault(
    "HF_HOME", str(PROJECT_ROOT / "models" / "pretrained" / ".hf-cache")
)
os.environ.setdefault(
    "HF_MODULES_CACHE",
    str(PROJECT_ROOT / "models" / "pretrained" / ".hf-modules"),
)

from transformers import AutoModelForCausalLM  # noqa: E402
from transformers.utils import logging as transformers_logging  # noqa: E402

transformers_logging.set_verbosity_error()


class GatedNumericEmbedding(nn.Module):
    """按 TimeMoE 原始输入层的门控结构嵌入多变量时间步。"""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.value_projection = nn.Linear(input_size, hidden_size, bias=False)
        self.gate_projection = nn.Linear(input_size, hidden_size, bias=False)
        self.activation = nn.SiLU()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        value = self.value_projection(values)
        gate = self.activation(self.gate_projection(values))
        return gate * value


class TimeMoEAdaptiveRevIN(nn.Module):
    """结合 A-RevIN 与预训练 TimeMoE 的单风机功率预测模型。"""

    def __init__(
        self,
        horizon: int,
        channels: int,
        power_index: int,
        entity_count: int,
        pretrained_path: str = "",
        bottleneck: int = 512,
        unfreeze_layers: int = 0,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        model_path = (
            Path(pretrained_path)
            if pretrained_path
            else PROJECT_ROOT / "models" / "pretrained" / "TimeMoE-50M"
        )
        self.horizon = int(horizon)
        self.power_index = int(power_index)
        self.unfreeze_layers = int(unfreeze_layers)

        self.time_moe = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        self.time_moe.config.use_cache = False
        self.max_position_embeddings = int(
            getattr(self.time_moe.config, "max_position_embeddings", 0)
        )
        self.backbone = self.time_moe.get_decoder()
        for parameter in self.time_moe.parameters():
            parameter.requires_grad_(False)

        layers = getattr(self.backbone, "layers", None)
        
        if self.unfreeze_layers:
            for layer in layers[-self.unfreeze_layers :]:
                for parameter in layer.parameters():
                    parameter.requires_grad_(True)
            for parameter in self.backbone.norm.parameters():
                parameter.requires_grad_(True)

        if gradient_checkpointing and self.unfreeze_layers:
            self.time_moe.gradient_checkpointing_enable()

        hidden_size = int(self.time_moe.config.hidden_size)
        self.adaptive_revin = AdaptiveRevIN(self.horizon)
        self.input_embedding = GatedNumericEmbedding(channels, hidden_size)
        self.turbine_embedding = nn.Embedding(entity_count, hidden_size)
        self.input_norm = nn.LayerNorm(hidden_size)
        self.pool_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid(),
        )
        self.forecast_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, int(bottleneck)),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(int(bottleneck), self.horizon),
        )

    def forward(
        self, past: torch.Tensor, turbine_id: torch.Tensor
    ) -> torch.Tensor:
        
        normalized, revin_state = self.adaptive_revin.normalize(past)

        tokens = self.input_embedding(normalized)
        tokens = tokens + self.turbine_embedding(turbine_id).unsqueeze(1)

        tokens = self.input_norm(tokens)

        backbone_dtype = next(self.backbone.parameters()).dtype

        encoded = self.backbone(
            inputs_embeds=tokens.to(dtype=backbone_dtype),
            use_cache=False,
            return_dict=True,
        ).last_hidden_state


        last = encoded[:, -1]
        # mean = encoded.mean(dim=1)
        # gate = self.pool_gate(torch.cat([last, mean], dim=-1))
        # representation = gate * last + (1.0 - gate) * mean

        normalized_power = self.forecast_head(last)

        return self.adaptive_revin.denormalize_feature(
            normalized_power, revin_state, self.power_index
        )

    def parameter_groups(self, adapter_lr: float, backbone_lr: float):
        """分别为新增适配层和解冻的 TimeMoE 层设置学习率。"""
        adapter_parameters = []
        backbone_parameters = []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith("time_moe.") or name.startswith("backbone."):
                backbone_parameters.append(parameter)
            else:
                adapter_parameters.append(parameter)

        groups = [{"params": adapter_parameters, "lr": float(adapter_lr)}]
        if backbone_parameters:
            groups.append({"params": backbone_parameters, "lr": float(backbone_lr)})
        return groups
