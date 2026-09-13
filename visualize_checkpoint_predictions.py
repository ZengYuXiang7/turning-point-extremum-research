from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

# 允许从项目根目录以脚本方式直接执行
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import (
    PROJECT_ROOT,
    apply_history_steps,
    apply_sample_seconds,
    apply_window_stride_steps,
)


os.environ.setdefault(
    "HF_HOME", str(PROJECT_ROOT / "models" / "pretrained" / ".hf-cache")
)
os.environ.setdefault(
    "HF_MODULES_CACHE",
    str(PROJECT_ROOT / "models" / "pretrained" / ".hf-modules"),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="从训练 checkpoint 重建固定样本，并导出 10 台风机的功率预测图。"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--selection-seed", type=int, default=20260913)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def build_dataset_and_model(config, split: str, device: torch.device):
    # 按训练记录重建数据集和模型
    from data_provider.forecast_weather_dataset import make_forecast_weather_loaders
    from data_provider.multi_turbine_relation_dataset import (
        make_multi_turbine_no_future_weather_loaders,
        make_multi_turbine_oracle_future_weather_loaders,
    )
    from data_provider.no_future_weather_dataset import make_no_future_weather_loaders
    from data_provider.oracle_future_weather_dataset import (
        make_oracle_future_weather_loaders,
    )
    from data_provider.predicted_future_weather_dataset import (
        make_predicted_future_weather_loaders,
    )
    from models.backbone.stockecho import StockEchoNoFutureWeather, StockEchoWindPower
    from models.futureweather import FutureWeatherModel
    from models.noweather import NoFutureWeatherModel
    from models.weatherforecast import WeatherForecastModel

    horizon_steps = config["horizon_steps"]
    no_future_weather = config["scenario"] == "NoFutureWeather"
    weather_task = config["scenario"] == "ForecastWeather"
    predicted_weather = config["scenario"] == "PredictedFutureWeather"
    oracle = config["scenario"] == "OracleFutureWeather"
    joint_panel = config["model"] == "StockEcho"
    qwen_mlp = config["model"] == "QwenMLP"
    timer_weather_mlp = config["model"] == "TimerWeatherMLP"
    all_features = config["model"] == "DLinearAllFeatures"

    if weather_task:
        repository, datasets, _ = make_forecast_weather_loaders(
            horizon=horizon_steps,
            batch_size=config["batch_size"],
            num_workers=0,
        )
        model = WeatherForecastModel(config["model"], horizon_steps).to(device)
    elif predicted_weather:
        repository, datasets, _ = make_predicted_future_weather_loaders(
            horizon=horizon_steps,
            batch_size=config["batch_size"],
            num_workers=0,
            weather_forecast_dir=config["weather_forecast_dir"],
        )
        if timer_weather_mlp:
            from models.backbone.timer import TimerWeatherMLP

            model = TimerWeatherMLP(
                horizon=horizon_steps,
                pretrained_path=config["timer_path"],
                patch_length=config["timer_patch_length"],
                bottleneck=config["timer_bottleneck"],
                unfreeze_layers=config["timer_unfreeze_layers"],
                gradient_checkpointing=bool(config["timer_gradient_checkpointing"]),
                residual_forecast=bool(config["timer_residual_forecast"]),
            ).to(device)
        else:
            model = FutureWeatherModel(config["model"], horizon_steps).to(device)
    elif joint_panel:
        if oracle:
            repository, datasets, _ = make_multi_turbine_oracle_future_weather_loaders(
                horizon=horizon_steps,
                batch_size=config["batch_size"],
                num_workers=0,
            )
            model = StockEchoWindPower(horizon_steps).to(device)
            model_uses_future_weather = True
        else:
            repository, datasets, _ = make_multi_turbine_no_future_weather_loaders(
                horizon=horizon_steps,
                batch_size=config["batch_size"],
                num_workers=0,
            )
            model = StockEchoNoFutureWeather(horizon_steps).to(device)
            model_uses_future_weather = False
    else:
        if oracle:
            repository, datasets, _ = make_oracle_future_weather_loaders(
                horizon=horizon_steps,
                batch_size=config["batch_size"],
                num_workers=0,
            )
        else:
            repository, datasets, _ = make_no_future_weather_loaders(
                horizon=horizon_steps,
                batch_size=config["batch_size"],
                num_workers=0,
                all_features=all_features,
            )
        if qwen_mlp:
            from models.backbone.qwen import QwenMLP

            model = QwenMLP(
                horizon=horizon_steps,
                pretrained_path=config["llm_path"],
                patch_length=config["llm_patch_length"],
                patch_stride=config["llm_patch_stride"],
                bottleneck=config["llm_bottleneck"],
                gradient_checkpointing=bool(config["llm_gradient_checkpointing"]),
            ).to(device)
            model_uses_future_weather = False
        elif no_future_weather:
            model = NoFutureWeatherModel(
                config["model"],
                horizon_steps,
                len(repository.feature_names),
                repository.power_index,
            ).to(device)
            model_uses_future_weather = False
        elif timer_weather_mlp:
            from models.backbone.timer import TimerWeatherMLP

            model = TimerWeatherMLP(
                horizon=horizon_steps,
                pretrained_path=config["timer_path"],
                patch_length=config["timer_patch_length"],
                bottleneck=config["timer_bottleneck"],
                unfreeze_layers=config["timer_unfreeze_layers"],
                gradient_checkpointing=bool(config["timer_gradient_checkpointing"]),
                residual_forecast=bool(config["timer_residual_forecast"]),
            ).to(device)
            model_uses_future_weather = True
        else:
            model = FutureWeatherModel(config["model"], horizon_steps).to(device)
            model_uses_future_weather = True

    if weather_task:
        model_uses_future_weather = True
    elif predicted_weather:
        model_uses_future_weather = True

    dataset = datasets[split]
    return repository, dataset, model, model_uses_future_weather


def load_checkpoint(model, config, checkpoint):
    # 保持与训练阶段相同的编译包装，再载入权重
    if config["compile_mode"] == "reduce-overhead":
        model = torch.compile(model, mode="reduce-overhead")

    model.load_state_dict(
        checkpoint["model_state"], strict=config["model"] != "QwenMLP"
    )
    model.eval()
    return model, checkpoint["epoch"]


def select_single_turbine_indices(dataset, selection_seed: int):
    # 按固定随机种子为每台风机选择一个窗口序号
    generator = np.random.default_rng(selection_seed)
    window_counts = np.zeros(10, dtype=np.int64)
    for window in dataset.windows:
        turbine_index = window[0]
        if turbine_index < 10:
            window_counts[turbine_index] += 1

    dataset_indices = []
    turbine_window_indices = []
    for turbine_index in range(10):
        turbine_window_index = int(generator.integers(window_counts[turbine_index]))
        observed_window_index = 0
        for dataset_index, window in enumerate(dataset.windows):
            candidate_turbine_index = window[0]
            if candidate_turbine_index == turbine_index:
                if observed_window_index == turbine_window_index:
                    dataset_indices.append(dataset_index)
                    turbine_window_indices.append(turbine_window_index)
                    break
                observed_window_index += 1
    return dataset_indices, turbine_window_indices


def collect_single_turbine_curves(
    repository,
    dataset,
    model,
    selection_seed: int,
    device: torch.device,
    no_future_weather: bool,
    model_uses_future_weather: bool,
):
    # 逐台执行 checkpoint 前向，并把标准化功率还原为 kW
    dataset_indices, turbine_window_indices = select_single_turbine_indices(
        dataset, selection_seed
    )
    past_inputs = []
    weather_inputs = []
    turbine_inputs = []
    histories_kw = []
    truths_kw = []
    predictions_kw = []
    turbine_ids = []
    target_starts = []

    with torch.no_grad():
        for dataset_index in dataset_indices:
            sample = dataset[dataset_index]
            if no_future_weather:
                past, target, turbine_id, target_start = sample
            else:
                past, weather, target, turbine_id, target_start = sample
                weather_inputs.append(weather.numpy())

            past_batch = past.unsqueeze(0).to(device)
            turbine_batch = turbine_id.reshape(1).to(device)
            if model_uses_future_weather:
                weather_batch = weather.unsqueeze(0).to(device)
                prediction_scaled = model(past_batch, weather_batch, turbine_batch)[0]
            else:
                prediction_scaled = model(past_batch, turbine_batch)[0]

            turbine_index = int(turbine_id.item())
            series = repository.series[turbine_index]
            history_scaled = past[:, repository.power_index].numpy()
            truth_scaled = target.numpy()
            prediction_scaled = prediction_scaled.float().cpu().numpy()
            history_kw = history_scaled * series.power_std + series.power_mean
            truth_kw = truth_scaled * series.power_std + series.power_mean
            prediction_kw = prediction_scaled * series.power_std + series.power_mean

            past_inputs.append(past.numpy())
            turbine_inputs.append(turbine_index)
            histories_kw.append(history_kw)
            truths_kw.append(truth_kw)
            predictions_kw.append(prediction_kw)
            turbine_ids.append(turbine_index + 1)
            target_starts.append(int(target_start.item()))

    input_past = np.stack(past_inputs).astype(np.float32)
    if no_future_weather:
        input_weather = np.empty((input_past.shape[0], 0, 0), dtype=np.float32)
    else:
        input_weather = np.stack(weather_inputs).astype(np.float32)

    input_turbine_id = np.asarray(turbine_inputs, dtype=np.int16)
    history_power_kw = np.stack(histories_kw).astype(np.float32)
    truth_power_kw = np.stack(truths_kw).astype(np.float32)
    prediction_power_kw = np.stack(predictions_kw).astype(np.float32)
    selected_turbine_id = np.asarray(turbine_ids, dtype=np.int16)
    target_start_ns = np.asarray(target_starts, dtype=np.int64)
    dataset_index = np.asarray(dataset_indices, dtype=np.int64)
    turbine_window_index = np.asarray(turbine_window_indices, dtype=np.int64)
    return (
        input_past,
        input_weather,
        input_turbine_id,
        history_power_kw,
        truth_power_kw,
        prediction_power_kw,
        selected_turbine_id,
        target_start_ns,
        dataset_index,
        turbine_window_index,
    )


def collect_joint_panel_curves(
    repository,
    dataset,
    model,
    selection_seed: int,
    device: torch.device,
    no_future_weather: bool,
    model_uses_future_weather: bool,
):
    # 多机模型以同一个面板样本完成一次联合前向
    generator = np.random.default_rng(selection_seed)
    panel_index = int(generator.integers(len(dataset)))
    sample = dataset[panel_index]
    if no_future_weather:
        past, target, turbine_id, target_start = sample
    else:
        past, weather, target, turbine_id, target_start = sample

    past_batch = past.unsqueeze(0).to(device)
    turbine_batch = turbine_id.unsqueeze(0).to(device)

    with torch.no_grad():
        if model_uses_future_weather:
            weather_batch = weather.unsqueeze(0).to(device)
            prediction_scaled = model(past_batch, weather_batch, turbine_batch)[0]
        else:
            prediction_scaled = model(past_batch, turbine_batch)[0]

    mean_values = []
    std_values = []
    for turbine_index in range(10):
        series = repository.series[turbine_index]
        mean_values.append(series.power_mean)
        std_values.append(series.power_std)
    power_means = np.asarray(mean_values, dtype=np.float32)[:, None]
    power_stds = np.asarray(std_values, dtype=np.float32)[:, None]

    history_scaled = past[:10, :, repository.power_index].numpy()
    truth_scaled = target[:10].numpy()
    prediction_scaled = prediction_scaled[:10].float().cpu().numpy()
    history_power_kw = history_scaled * power_stds + power_means
    truth_power_kw = truth_scaled * power_stds + power_means
    prediction_power_kw = prediction_scaled * power_stds + power_means
    selected_turbine_id = np.arange(1, 11, dtype=np.int16)
    target_start_ns = np.full(10, int(target_start.item()), dtype=np.int64)
    dataset_index = np.full(10, panel_index, dtype=np.int64)
    turbine_window_index = np.full(10, panel_index, dtype=np.int64)
    if no_future_weather:
        input_weather = np.empty((past.shape[0], 0, 0), dtype=np.float32)
    else:
        input_weather = weather.numpy().astype(np.float32)

    return (
        past.numpy().astype(np.float32),
        input_weather,
        turbine_id.numpy().astype(np.int16),
        history_power_kw.astype(np.float32),
        truth_power_kw.astype(np.float32),
        prediction_power_kw.astype(np.float32),
        selected_turbine_id,
        target_start_ns,
        dataset_index,
        turbine_window_index,
    )


def save_figure(
    output_path: Path,
    config,
    split: str,
    selection_seed: int,
    checkpoint_epoch: int,
    history_power_kw: np.ndarray,
    truth_power_kw: np.ndarray,
    prediction_power_kw: np.ndarray,
    selected_turbine_id: np.ndarray,
    turbine_window_index: np.ndarray,
):
    # 用一个横向面板展示十台风机的历史、真实未来和预测未来
    history_steps = history_power_kw.shape[1]
    horizon_steps = truth_power_kw.shape[1]
    sample_minutes = config["sample_seconds"] / 60.0
    history_axis = np.arange(-history_steps, 0) * sample_minutes
    future_axis = np.arange(horizon_steps) * sample_minutes
    figure, axes = plt.subplots(1, 10, figsize=(50, 5.6))

    for panel_index in range(10):
        axis = axes[panel_index]
        history_line = axis.plot(
            history_axis,
            history_power_kw[panel_index],
            color="#6b7280",
            linewidth=1.2,
            label="History",
        )
        truth_line = axis.plot(
            future_axis,
            truth_power_kw[panel_index],
            color="#111827",
            linewidth=1.6,
            marker="o",
            markersize=3,
            label="True future",
        )
        prediction_line = axis.plot(
            future_axis,
            prediction_power_kw[panel_index],
            color="#dc2626",
            linewidth=1.3,
            linestyle="--",
            marker="o",
            markersize=3,
            label="Predicted future",
        )
        axis.axvline(0.0, color="#374151", linewidth=0.8, linestyle=":")
        axis.set_title(
            f"Turbine {int(selected_turbine_id[panel_index]):02d}\n"
            f"window {int(turbine_window_index[panel_index])}",
            fontsize=12,
        )
        axis.set_xlabel("Minutes", fontsize=10)
        axis.grid(alpha=0.25, linewidth=0.5)
        axis.tick_params(labelsize=9)

        if panel_index == 0:
            axis.set_ylabel("Power (kW)", fontsize=10)

    figure.legend(
        [history_line[0], truth_line[0], prediction_line[0]],
        ["History", "True future", "Predicted future"],
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=12,
    )
    figure.suptitle(
        f"{config['model']} / {config['scenario']} / {split} / "
        f"fixed random seed {selection_seed} / checkpoint epoch {checkpoint_epoch} / "
        f"seq_len={history_steps}, pred_len={horizon_steps}",
        fontsize=15,
        y=0.99,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    figure.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(figure)


def main():
    # 从 checkpoint 读取 run.py 参数并恢复派生窗口常量
    args = parse_args()
    run_dir = args.run_dir
    checkpoint_path = run_dir / "best_checkpoint.pt"
    device = torch.device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    apply_sample_seconds(config["sample_seconds"])
    apply_history_steps(config["history_steps"])
    apply_window_stride_steps(config["window_stride_steps"])

    # 重建训练路径并加载最佳 checkpoint
    repository, dataset, model, model_uses_future_weather = build_dataset_and_model(
        config, args.split, device
    )
    model, checkpoint_epoch = load_checkpoint(model, config, checkpoint)
    no_future_weather = config["scenario"] == "NoFutureWeather"

    # 提取固定样本的真实输入和功率曲线
    if config["joint_multi_turbine_panel"]:
        curves = collect_joint_panel_curves(
            repository,
            dataset,
            model,
            args.selection_seed,
            device,
            no_future_weather,
            model_uses_future_weather,
        )
    else:
        curves = collect_single_turbine_curves(
            repository,
            dataset,
            model,
            args.selection_seed,
            device,
            no_future_weather,
            model_uses_future_weather,
        )
    (
        input_past,
        input_weather,
        input_turbine_id,
        history_power_kw,
        truth_power_kw,
        prediction_power_kw,
        selected_turbine_id,
        target_start_ns,
        dataset_index,
        turbine_window_index,
    ) = curves

    # 保存可复核输入和一页十图的矢量 PDF
    artifact_stem = (
        f"checkpoint_{args.split}_random_seed_{args.selection_seed}_turbines_01-10"
    )
    sample_path = run_dir / f"{artifact_stem}.npz"
    output_path = run_dir / f"{artifact_stem}.pdf"
    np.savez_compressed(
        sample_path,
        input_past=input_past,
        input_weather=input_weather,
        input_turbine_id=input_turbine_id,
        history_power_kw=history_power_kw,
        truth_power_kw=truth_power_kw,
        prediction_power_kw=prediction_power_kw,
        turbine_id=selected_turbine_id,
        target_start_ns=target_start_ns,
        dataset_index=dataset_index,
        turbine_window_index=turbine_window_index,
        selection_seed=np.asarray(args.selection_seed, dtype=np.int64),
        checkpoint_epoch=np.asarray(checkpoint_epoch, dtype=np.int64),
    )
    save_figure(
        output_path,
        config,
        args.split,
        args.selection_seed,
        checkpoint_epoch,
        history_power_kw,
        truth_power_kw,
        prediction_power_kw,
        selected_turbine_id,
        turbine_window_index,
    )
    print(
        json.dumps(
            {
                "event": "checkpoint_visualization_saved",
                "checkpoint": str(checkpoint_path),
                "sample_input": str(sample_path),
                "pdf": str(output_path),
                "split": args.split,
                "selection_seed": args.selection_seed,
                "turbines": selected_turbine_id.tolist(),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
