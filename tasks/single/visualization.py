from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch

import config as project_config
from config import PROJECT_ROOT
from tasks.single.task import build_checkpoint_components, load_checkpoint_model
from tasks.single.visualization_output import print_saved_artifacts
from tasks.single.visualization_plot import save_prediction_figure


os.environ["HF_HOME"] = str(PROJECT_ROOT / "models" / "pretrained" / ".hf-cache")
os.environ["HF_MODULES_CACHE"] = str(PROJECT_ROOT / "models" / "pretrained" / ".hf-modules")

FIGURE_OUTPUT_DIRECTORY = PROJECT_ROOT / "output" / "figs"
DISPLAY_TURBINE_COUNT = 10


def select_single_turbine_indices(dataset, selection_seed: int):
    # 用一次线性扫描定位每台风机的固定随机窗口
    generator = np.random.default_rng(selection_seed)
    window_counts = np.zeros(DISPLAY_TURBINE_COUNT, dtype=np.int64)
    for window in dataset.windows:
        turbine_index = window[0]
        if turbine_index < DISPLAY_TURBINE_COUNT:
            window_counts[turbine_index] += 1

    selected_window_indices = generator.integers(window_counts)
    selected_dataset_indices = np.empty(DISPLAY_TURBINE_COUNT, dtype=np.int64)
    observed_window_counts = np.zeros(DISPLAY_TURBINE_COUNT, dtype=np.int64)

    for candidate_dataset_index, window in enumerate(dataset.windows):
        turbine_index = window[0]
        if turbine_index < DISPLAY_TURBINE_COUNT:
            observed_window_index = observed_window_counts[turbine_index]
            if observed_window_index == selected_window_indices[turbine_index]:
                selected_dataset_indices[turbine_index] = candidate_dataset_index
            observed_window_counts[turbine_index] += 1

    return selected_dataset_indices, selected_window_indices


def collect_single_turbine_curves(repository, dataset, model, selection_seed: int, device: torch.device, no_future_weather: bool, model_uses_future_weather: bool,):
    # 逐台执行 checkpoint 前向，并把标准化功率还原为 kW
    selected_dataset_indices, selected_window_indices = select_single_turbine_indices(dataset, selection_seed)

    # 按首个样本的明确形状预分配绘图曲线。
    first_sample = dataset[int(selected_dataset_indices[0])]
    if no_future_weather:
        first_past, first_target, _, _ = first_sample
    else:
        first_past, _, first_target, _, _ = first_sample
    history_power_kw = np.empty((DISPLAY_TURBINE_COUNT, first_past.shape[0]), dtype=np.float32,)
    truth_power_kw = np.empty((DISPLAY_TURBINE_COUNT, *first_target.shape), dtype=np.float32,)
    prediction_power_kw = np.empty_like(truth_power_kw)
    selected_turbine_id = np.empty(DISPLAY_TURBINE_COUNT, dtype=np.int16)

    with torch.no_grad():
        for selection_position in range(DISPLAY_TURBINE_COUNT):
            selected_dataset_index = int(selected_dataset_indices[selection_position])
            sample = dataset[selected_dataset_index]
            if no_future_weather:
                past, target, turbine_id, target_start = sample
            else:
                past, weather, target, turbine_id, target_start = sample

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

            history_power_kw[selection_position] = history_kw
            truth_power_kw[selection_position] = truth_kw
            prediction_power_kw[selection_position] = prediction_kw
            selected_turbine_id[selection_position] = turbine_index + 1

    return history_power_kw, truth_power_kw, prediction_power_kw, selected_turbine_id, selected_window_indices


def visualize_checkpoint_predictions(run_dir: Path, split: str, selection_seed: int, device_name: str,):
    # 从 checkpoint 读取入口参数并恢复派生窗口常量
    checkpoint_path = run_dir / "best_checkpoint.pt"
    device = torch.device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    project_config.apply_point_interval_seconds(config["point_interval_seconds"])
    project_config.apply_history_steps(config["history_steps"])
    project_config.apply_window_stride_steps(config["window_stride_steps"])

    # 重建训练路径并加载最佳 checkpoint
    repository, dataset, model, model_uses_future_weather = build_checkpoint_components(config, split, device)
    model, checkpoint_epoch = load_checkpoint_model(model, config, checkpoint)
    no_future_weather = config["scenario"] == "NoFutureWeather"

    # 单风机任务分别选择十台风机各自的固定随机窗口。
    history_power_kw, truth_power_kw, prediction_power_kw, selected_turbine_id, selected_window_indices = collect_single_turbine_curves(repository, dataset, model, selection_seed, device, no_future_weather, model_uses_future_weather,)

    # 在统一图件目录只保存 PDF 图件。
    FIGURE_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    artifact_stem = (
        f"{config['dataset_name']}__{config['result_name']}__{split}"
        f"__selection_seed_{selection_seed}__epoch_{checkpoint_epoch}"
    )
    prediction_pdf_path = FIGURE_OUTPUT_DIRECTORY / f"{artifact_stem}__prediction_curves.pdf"
    save_prediction_figure(prediction_pdf_path, config, split, selection_seed, checkpoint_epoch, history_power_kw, truth_power_kw, prediction_power_kw, selected_turbine_id, selected_window_indices,)
    print_saved_artifacts(checkpoint_path, prediction_pdf_path, split, selection_seed, selected_turbine_id,)
