from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch

import config as project_config
from config import PROJECT_ROOT
from tasks.multi_turbine.task import build_checkpoint_components, load_checkpoint_model
from tasks.multi_turbine.visualization_plot import save_prediction_figure


os.environ["HF_HOME"] = str(PROJECT_ROOT / "models" / "pretrained" / ".hf-cache")
os.environ["HF_MODULES_CACHE"] = str(PROJECT_ROOT / "models" / "pretrained" / ".hf-modules")

DISPLAY_TURBINE_COUNT = 10


def collect_panel_curves(repository, dataset, model, selection_seed: int, device):
    # 固定随机源只选择一个联合窗口，并用完整16台风机执行一次前向。
    generator = np.random.default_rng(selection_seed)
    panel_dataset_index = int(generator.integers(len(dataset)))
    past, target, turbine_id, _ = dataset[panel_dataset_index]
    past_batch = past.unsqueeze(0).to(device)
    turbine_batch = turbine_id.unsqueeze(0).to(device)

    with torch.no_grad():
        prediction_scaled = model(past_batch, turbine_batch)[0]

    # 绘图只截取前10台，模型输入和联合关系计算仍使用全部16台。
    history_scaled = past[ :DISPLAY_TURBINE_COUNT, :, repository.power_index, ].numpy()
    truth_scaled = target[:DISPLAY_TURBINE_COUNT].numpy()
    prediction_scaled = (
        prediction_scaled[:DISPLAY_TURBINE_COUNT].float().cpu().numpy()
    )
    power_mean = repository.series[0].power_mean
    power_std = repository.series[0].power_std
    history_power_kw = history_scaled * power_std + power_mean
    truth_power_kw = truth_scaled * power_std + power_mean
    prediction_power_kw = prediction_scaled * power_std + power_mean
    display_turbine_id = turbine_id[:DISPLAY_TURBINE_COUNT].numpy() + 1

    return history_power_kw.astype(np.float32), truth_power_kw.astype(np.float32), prediction_power_kw.astype(np.float32), display_turbine_id.astype(np.int16), panel_dataset_index, int(turbine_id.numel())


def visualize_checkpoint_predictions(run_dir: Path, split: str, selection_seed: int, device_name: str, output_directory: Path, output_batch_name: str | None,) -> Path:
    # 按多风机任务配置重建联合面板 Dataset 和 Model。
    checkpoint_path = (run_dir / "best_checkpoint.pt").resolve()
    device = torch.device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    project_config.apply_point_interval_seconds(config["point_interval_seconds"])
    project_config.apply_history_steps(config["history_steps"])
    project_config.apply_window_stride_steps(config["window_stride_steps"])
    repository, dataset, model, _ = build_checkpoint_components(config, split, device,)
    model, checkpoint_epoch = load_checkpoint_model(model, config, checkpoint)

    # 使用完整16台输入推理，并保存前10台曲线的 PDF。
    history_power_kw, truth_power_kw, prediction_power_kw, display_turbine_id, panel_dataset_index, model_input_turbines = collect_panel_curves(repository, dataset, model, selection_seed, device)
    task_output_directory = output_directory / f"h{config['history_steps']}_p{config['horizon_steps']}"
    if output_batch_name is not None:
        task_output_directory = task_output_directory / output_batch_name
    task_output_directory.mkdir(parents=True, exist_ok=True)
    output_pdf_path = task_output_directory / f"{config['result_name']}_pdf1.pdf"
    save_prediction_figure(output_pdf_path, config, split, selection_seed, checkpoint_epoch, panel_dataset_index, history_power_kw, truth_power_kw, prediction_power_kw, display_turbine_id,)

    # 输出统一使用项目相对路径。
    display_checkpoint_path = checkpoint_path
    if display_checkpoint_path.is_relative_to(PROJECT_ROOT):
        display_checkpoint_path = display_checkpoint_path.relative_to(PROJECT_ROOT)
    display_output_pdf_path = output_pdf_path
    if display_output_pdf_path.is_relative_to(PROJECT_ROOT):
        display_output_pdf_path = display_output_pdf_path.relative_to(PROJECT_ROOT)
    print(json.dumps({ "event": "multi_turbine_visualization_saved", "checkpoint": display_checkpoint_path.as_posix(), "pdf": display_output_pdf_path.as_posix(), "split": split, "selection_seed": selection_seed, "panel_dataset_index": panel_dataset_index, "model_input_turbines": model_input_turbines, "display_turbines": display_turbine_id.tolist(), }, ensure_ascii=False,), flush=True,)
    return output_pdf_path
