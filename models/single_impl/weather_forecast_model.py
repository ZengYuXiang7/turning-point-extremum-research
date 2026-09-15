from __future__ import annotations

import torch
from torch import nn

import config as project_config
from config import WEATHER_COLUMNS
from models.single_impl.dlinear import DLinear


class WeatherForecastModel(nn.Module):
    """程序1：历史气象 → 未来气象；输出与历史末点强制衔接。"""

    def __init__(self, model_name: str, horizon: int) -> None:
        super().__init__()
        self.model_name = model_name.lower()
        self.horizon = int(horizon)
        # 当前级联壳只用 DLinear；其它名字同样走 DLinear 通道数=天气维
        self.backbone = DLinear(project_config.HISTORY_STEPS, self.horizon, len(WEATHER_COLUMNS),)

    def forward(self, past_weather: torch.Tensor, weather_unused: torch.Tensor, turbine_id: torch.Tensor,) -> torch.Tensor:
        # backbone 先出原始多步预报
        forecast = self.backbone(past_weather)

        # 与历史最后一步衔接：整段平移，使 forecast[:,0,:]==past[:,-1,:]
        last = past_weather[:, -1:, :]
        first = forecast[:, :1, :]
        forecast = forecast + (last - first)
        return forecast
