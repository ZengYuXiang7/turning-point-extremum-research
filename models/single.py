from torch import nn

from models.single_impl.future_weather_model import FutureWeatherModel
from models.single_impl.no_future_model import NoFutureWeatherModel
from models.single_impl.weather_forecast_model import WeatherForecastModel


def select_backbone(args, repository):
    # Single 任务按场景和模型名显式选择骨干。
    weather_task = args.scenario == "ForecastWeather"
    predicted_weather = args.scenario == "PredictedFutureWeather"
    provided_weather = args.scenario == "ProvidedFutureWeather"
    oracle = args.scenario == "OracleFutureWeather"
    qwen_mlp = args.model == "QwenMLP"
    timer_weather_mlp = args.model == "TimerWeatherMLP"
    time_moe_arevin = args.model == "TimeMoEARevIN"

    if time_moe_arevin and args.scenario != "NoFutureWeather":
        raise ValueError("TimeMoEARevIN 目前仅支持 NoFutureWeather 场景")
    if time_moe_arevin and getattr(args, "pretrain", False):
        raise ValueError(
            "TimeMoEARevIN 不使用当前 PatchMLP 专用的掩码重建预训练，"
            "请移除 --pretrain"
        )

    if weather_task:
        backbone = WeatherForecastModel(args.model, args.horizon_steps)
    elif predicted_weather:
        if timer_weather_mlp:
            from models.single_impl.timer import TimerWeatherMLP

            backbone = TimerWeatherMLP(horizon=args.horizon_steps, pretrained_path=args.timer_path, patch_length=args.timer_patch_length, bottleneck=args.timer_bottleneck, unfreeze_layers=args.timer_unfreeze_layers, gradient_checkpointing=bool(args.timer_gradient_checkpointing), residual_forecast=bool(args.timer_residual_forecast),)
        else:
            backbone = FutureWeatherModel(args.model, args.horizon_steps, len(repository.feature_names), repository.power_index, len(repository.future_feature_names), len(repository.series),)
    elif provided_weather:
        backbone = FutureWeatherModel(args.model, args.horizon_steps, len(repository.feature_names), repository.power_index, len(repository.future_feature_names), len(repository.series),)
    elif qwen_mlp:
        from models.single_impl.qwen import QwenMLP

        backbone = QwenMLP(horizon=args.horizon_steps, pretrained_path=args.llm_path, patch_length=args.llm_patch_length, patch_stride=args.llm_patch_stride, bottleneck=args.llm_bottleneck, gradient_checkpointing=bool(args.llm_gradient_checkpointing),)
    elif timer_weather_mlp:
        if not oracle:
            raise ValueError("TimerWeatherMLP requires OracleFutureWeather or " "PredictedFutureWeather")
        from models.single_impl.timer import TimerWeatherMLP

        backbone = TimerWeatherMLP(horizon=args.horizon_steps, pretrained_path=args.timer_path, patch_length=args.timer_patch_length, bottleneck=args.timer_bottleneck, unfreeze_layers=args.timer_unfreeze_layers, gradient_checkpointing=bool(args.timer_gradient_checkpointing), residual_forecast=bool(args.timer_residual_forecast),)
    elif time_moe_arevin:
        from models.single_impl.time_moe_arevin import TimeMoEAdaptiveRevIN

        backbone = TimeMoEAdaptiveRevIN(
            horizon=args.horizon_steps,
            channels=len(repository.feature_names),
            power_index=repository.power_index,
            entity_count=len(repository.series),
            pretrained_path=args.time_moe_path,
            bottleneck=args.time_moe_bottleneck,
            unfreeze_layers=args.time_moe_unfreeze_layers,
            gradient_checkpointing=bool(args.time_moe_gradient_checkpointing),
        )
    elif args.scenario == "NoFutureWeather":
        backbone = NoFutureWeatherModel(args.model, args.horizon_steps, len(repository.feature_names), repository.power_index,)
    else:
        backbone = FutureWeatherModel(args.model, args.horizon_steps, len(repository.feature_names), repository.power_index, len(repository.future_feature_names), len(repository.series),)
    return backbone


class SingleModel(nn.Module):
    """Single 任务的顶层模型。"""

    def __init__(self, args, repository) -> None:
        super().__init__()
        self.backbone = select_backbone(args, repository)

    def forward(self, *inputs):
        return self.backbone(*inputs)
