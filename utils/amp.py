from __future__ import annotations

from contextlib import nullcontext

import torch


def autocast_context(device: torch.device):
    """CUDA 使用原有 bf16 autocast，CPU/MPS 保持普通精度。"""
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()
