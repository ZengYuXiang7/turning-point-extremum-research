from __future__ import annotations

import torch

from models.single_impl.future_weather_model import FutureWeatherModel
from models.single_impl.no_future_model import NoFutureWeatherModel, RevIN
from models.single_impl.weather_forecast_model import WeatherForecastModel


def build_model(args, repository, device: torch.device):
    # 顶层模型入口按 single 场景选择实际预测模型。
    weather_task = args.scenario == "ForecastWeather"
    predicted_weather = args.scenario == "PredictedFutureWeather"
    oracle = args.scenario == "OracleFutureWeather"
    qwen_mlp = args.model == "QwenMLP"
    timer_weather_mlp = args.model == "TimerWeatherMLP"

    if weather_task:
        model = WeatherForecastModel(args.model, args.horizon_steps).to(device)
        model_uses_future_weather = True
    elif predicted_weather:
        if timer_weather_mlp:
            from models.single_impl.backbone.timer import TimerWeatherMLP

            model = TimerWeatherMLP(
                horizon=args.horizon_steps,
                pretrained_path=args.timer_path,
                patch_length=args.timer_patch_length,
                bottleneck=args.timer_bottleneck,
                unfreeze_layers=args.timer_unfreeze_layers,
                gradient_checkpointing=bool(args.timer_gradient_checkpointing),
                residual_forecast=bool(args.timer_residual_forecast),
            ).to(device)
        else:
            model = FutureWeatherModel(args.model, args.horizon_steps).to(device)
        model_uses_future_weather = True
    elif qwen_mlp:
        from models.single_impl.backbone.qwen import QwenMLP

        model = QwenMLP(
            horizon=args.horizon_steps,
            pretrained_path=args.llm_path,
            patch_length=args.llm_patch_length,
            patch_stride=args.llm_patch_stride,
            bottleneck=args.llm_bottleneck,
            gradient_checkpointing=bool(args.llm_gradient_checkpointing),
        ).to(device)
        model_uses_future_weather = False
    elif timer_weather_mlp:
        if not oracle:
            raise ValueError(
                "TimerWeatherMLP requires OracleFutureWeather or "
                "PredictedFutureWeather"
            )
        from models.single_impl.backbone.timer import TimerWeatherMLP

        model = TimerWeatherMLP(
            horizon=args.horizon_steps,
            pretrained_path=args.timer_path,
            patch_length=args.timer_patch_length,
            bottleneck=args.timer_bottleneck,
            unfreeze_layers=args.timer_unfreeze_layers,
            gradient_checkpointing=bool(args.timer_gradient_checkpointing),
            residual_forecast=bool(args.timer_residual_forecast),
        ).to(device)
        model_uses_future_weather = True
    elif args.scenario == "NoFutureWeather":
        model = NoFutureWeatherModel(
            args.model,
            args.horizon_steps,
            len(repository.feature_names),
            repository.power_index,
        ).to(device)
        model_uses_future_weather = False
    else:
        model = FutureWeatherModel(args.model, args.horizon_steps).to(device)
        model_uses_future_weather = True

    return model, model_uses_future_weather
