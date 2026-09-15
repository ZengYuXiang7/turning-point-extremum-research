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
os.environ["HF_MODULES_CACHE"] = str(
    PROJECT_ROOT / "models" / "pretrained" / ".hf-modules"
)

FIGURE_OUTPUT_DIRECTORY = PROJECT_ROOT / "output" / "figs"
DISPLAY_TURBINE_COUNT = 10


def collect_panel_curves(repository, dataset, model, selection_seed: int, device):
    # 固定随机源只选择一个联合窗口，并用完整16台风机执行一次前向。
    generator = np.random.default_rng(selection_seed)
    panel_dataset_index = int(generator.integers(len(dataset)))
    past, target, turbine_id, target_start = dataset[panel_dataset_index]
    past_batch = past.unsqueeze(0).to(device)
    turbine_batch = turbine_id.unsqueeze(0).to(device)

    with torch.no_grad():
        prediction_scaled = model(past_batch, turbine_batch)[0]

    # 绘图只截取前10台，模型输入和联合关系计算仍使用全部16台。
    power_means = np.empty((DISPLAY_TURBINE_COUNT, 1), dtype=np.float32)
    power_stds = np.empty((DISPLAY_TURBINE_COUNT, 1), dtype=np.float32)
    for turbine_index in range(DISPLAY_TURBINE_COUNT):
        series = repository.series[turbine_index]
        power_means[turbine_index] = series.power_mean
        power_stds[turbine_index] = series.power_std

    history_scaled = past[
        :DISPLAY_TURBINE_COUNT,
        :,
        repository.power_index,
    ].numpy()
    truth_scaled = target[:DISPLAY_TURBINE_COUNT].numpy()
    prediction_scaled = (
        prediction_scaled[:DISPLAY_TURBINE_COUNT].float().cpu().numpy()
    )
    history_power_kw = history_scaled * power_stds + power_means
    truth_power_kw = truth_scaled * power_stds + power_means
    prediction_power_kw = prediction_scaled * power_stds + power_means
    display_turbine_id = turbine_id[:DISPLAY_TURBINE_COUNT].numpy() + 1

    return (
        past.numpy().astype(np.float32),
        turbine_id.numpy().astype(np.int16),
        history_power_kw.astype(np.float32),
        truth_power_kw.astype(np.float32),
        prediction_power_kw.astype(np.float32),
        display_turbine_id.astype(np.int16),
        int(target_start.item()),
        panel_dataset_index,
    )


def visualize_checkpoint_predictions(
    run_dir: Path,
    split: str,
    selection_seed: int,
    device_name: str,
) -> None:
    # 按多风机任务配置重建联合面板 Dataset 和 Model。
    checkpoint_path = (run_dir / "best_checkpoint.pt").resolve()
    device = torch.device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    project_config.apply_point_interval_seconds(config["point_interval_seconds"])
    project_config.apply_history_steps(config["history_steps"])
    project_config.apply_window_stride_steps(config["window_stride_steps"])
    repository, dataset, model, _ = build_checkpoint_components(
        config,
        split,
        device,
    )
    model, checkpoint_epoch = load_checkpoint_model(model, config, checkpoint)

    # 保存完整16台输入和前10台曲线，明确记录唯一联合窗口索引。
    (
        input_past,
        input_turbine_id,
        history_power_kw,
        truth_power_kw,
        prediction_power_kw,
        display_turbine_id,
        target_start_ns,
        panel_dataset_index,
    ) = collect_panel_curves(repository, dataset, model, selection_seed, device)
    artifact_stem = (
        f"{config['dataset_name']}__{config['result_name']}__{split}"
        f"__selection_seed_{selection_seed}__epoch_{checkpoint_epoch}"
    )
    FIGURE_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    sample_archive_path = FIGURE_OUTPUT_DIRECTORY / f"{artifact_stem}__panel_inputs.npz"
    output_pdf_path = FIGURE_OUTPUT_DIRECTORY / f"{artifact_stem}__prediction_curves.pdf"
    output_png_path = FIGURE_OUTPUT_DIRECTORY / f"{artifact_stem}__prediction_curves.png"
    np.savez_compressed(
        sample_archive_path,
        input_past=input_past,
        input_turbine_id=input_turbine_id,
        history_power_kw=history_power_kw,
        truth_power_kw=truth_power_kw,
        prediction_power_kw=prediction_power_kw,
        display_turbine_id=display_turbine_id,
        target_start_ns=np.asarray(target_start_ns, dtype=np.int64),
        panel_dataset_index=np.asarray(panel_dataset_index, dtype=np.int64),
        selection_seed=np.asarray(selection_seed, dtype=np.int64),
        checkpoint_epoch=np.asarray(checkpoint_epoch, dtype=np.int64),
    )
    save_prediction_figure(
        output_pdf_path,
        output_png_path,
        config,
        split,
        selection_seed,
        checkpoint_epoch,
        panel_dataset_index,
        history_power_kw,
        truth_power_kw,
        prediction_power_kw,
        display_turbine_id,
    )

    # 输出统一使用项目相对路径。
    display_checkpoint_path = checkpoint_path
    if display_checkpoint_path.is_relative_to(PROJECT_ROOT):
        display_checkpoint_path = display_checkpoint_path.relative_to(PROJECT_ROOT)
    print(
        json.dumps(
            {
                "event": "multi_turbine_visualization_saved",
                "checkpoint": display_checkpoint_path.as_posix(),
                "sample_input": sample_archive_path.relative_to(PROJECT_ROOT).as_posix(),
                "pdf": output_pdf_path.relative_to(PROJECT_ROOT).as_posix(),
                "png": output_png_path.relative_to(PROJECT_ROOT).as_posix(),
                "split": split,
                "selection_seed": selection_seed,
                "panel_dataset_index": panel_dataset_index,
                "model_input_turbines": int(input_past.shape[0]),
                "display_turbines": display_turbine_id.tolist(),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
