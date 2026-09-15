from __future__ import annotations

from dataclasses import dataclass

from config import HISTORY_COLUMNS


@dataclass
class PatchConfig:
    seq_len: int
    pred_len: int
    d_model: int = 320
    enc_in: int = len(HISTORY_COLUMNS)
    e_layers: int = 1
    output_attention: bool = False
    use_norm: bool = True
