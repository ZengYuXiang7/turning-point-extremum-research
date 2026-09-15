from __future__ import annotations

from argparse import Namespace

import torch

from models.single import build_model
from tasks.single.forecast_weather_dataset import make_forecast_weather_loaders
from tasks.single.no_future_dataset import make_no_future_weather_loaders
from tasks.single.oracle_future_dataset import make_oracle_future_weather_loaders
from tasks.single.predicted_future_weather_dataset import (
    make_predicted_future_weather_loaders,
)
from tasks.single.trainer import prepare_task_runtime, train_task


def build_task_components(args, device: torch.device, num_workers: int):
    weather_task = args.scenario == "ForecastWeather"
    predicted_weather = args.scenario == "PredictedFutureWeather"
    oracle = args.scenario == "OracleFutureWeather"
    all_features = args.model in (
        "DLinearAllFeatures",
        "PatchMLPAllFeatures",
    )
    correlated_features = args.model == "DLinearCorrelatedFeatures"

    # 天气预测任务读取自己的历史天气监督数据。
    if weather_task:
        repository, datasets, loaders = make_forecast_weather_loaders(
            horizon=args.horizon_steps,
            batch_size=args.batch_size,
            num_workers=num_workers,
        )
    elif predicted_weather:
        repository, datasets, loaders = make_predicted_future_weather_loaders(
            horizon=args.horizon_steps,
            batch_size=args.batch_size,
            num_workers=num_workers,
            weather_forecast_dir=args.weather_forecast_dir,
        )
    else:
        # 常规功率任务按未来天气口径读取单风机窗口。
        if oracle:
            repository, datasets, loaders = make_oracle_future_weather_loaders(
                horizon=args.horizon_steps,
                batch_size=args.batch_size,
                num_workers=num_workers,
            )
        else:
            repository, datasets, loaders = make_no_future_weather_loaders(
                horizon=args.horizon_steps,
                batch_size=args.batch_size,
                num_workers=num_workers,
                all_features=all_features,
                correlated_features=correlated_features,
            )

    model, model_uses_future_weather = build_model(args, repository, device)

    return repository, datasets, loaders, model, model_uses_future_weather


def build_checkpoint_components(config, split: str, device: torch.device):
    # checkpoint 配置直接复用同一个常规 Dataset 与 Model 构建链。
    args = Namespace(**config)
    repository, datasets, _, model, model_uses_future_weather = (
        build_task_components(args, device, num_workers=0)
    )
    dataset = datasets[split]
    return repository, dataset, model, model_uses_future_weather


def load_checkpoint_model(model, config, checkpoint):
    # 使用 single 训练阶段相同的编译包装载入最佳权重。
    if config["compile_mode"] == "reduce-overhead":
        model = torch.compile(model, mode="reduce-overhead")
    model.load_state_dict(
        checkpoint["model_state"],
        strict=config["model"] != "QwenMLP",
    )
    model.eval()
    return model, checkpoint["epoch"]


def train_and_test(args) -> dict:
    if args.scenario == "MultiTurbine":
        raise ValueError("single_task does not accept scenario MultiTurbine")
    if args.model in ("StockEcho", "MultiTurbine"):
        raise ValueError("StockEcho and MultiTurbine belong to the MultiTurbine task")

    device = prepare_task_runtime(args.seed)
    repository, datasets, loaders, model, model_uses_future_weather = (
        build_task_components(args, device, args.num_workers)
    )
    result = train_task(
        args,
        repository,
        datasets,
        loaders,
        model,
        device,
        model_uses_future_weather,
    )
    return result
