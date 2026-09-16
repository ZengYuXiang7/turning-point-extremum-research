from __future__ import annotations

import gc
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from models.multi_turbine import MultiTurbineModel
from tasks.multi_turbine.task import load_checkpoint_model
from utils.metrics import point_metrics


def select_issue_origins(times: np.ndarray, start: str, end: str, issue_interval_minutes: int) -> np.ndarray:
    # 按发布间隔选择时刻，历史窗口包含发布时刻的真实值。
    start_ns = pd.Timestamp(start).value
    end_ns = pd.Timestamp(end).value
    issue_interval_ns = pd.Timedelta(minutes=issue_interval_minutes).value
    in_period = (times >= start_ns) & (times < end_ns)
    on_issue_interval = times % issue_interval_ns == 0
    origin_indices = np.flatnonzero(in_period & on_issue_interval)
    return origin_indices


def build_history_batch(panel: np.ndarray, origin_indices: np.ndarray, history_steps: int, point_stride: int,) -> np.ndarray:
    # 每次发布时，窗口尾部使用刚获得的真实数据而非前次预测。
    history_offsets = np.arange(-history_steps + 1, 1) * point_stride
    history_indices = origin_indices[:, None] + history_offsets[None, :]
    past = panel[:, history_indices, :]
    past = np.transpose(past, (1, 0, 2, 3))
    past = np.ascontiguousarray(past)
    return past


def load_horizon_model(checkpoint_path: Path, repository, device: torch.device):
    # checkpoint 配置决定模型结构，权重载入沿用训练任务入口。
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = checkpoint["config"]
    model = MultiTurbineModel(
        Namespace(**model_config), repository, relation_weight=None
    ).to(device)
    model, checkpoint_epoch = load_checkpoint_model(model, model_config, checkpoint)
    return model, model_config, checkpoint_epoch


def predict_horizon(checkpoint_path: Path, repository, origin_indices: np.ndarray, batch_size: int, device: torch.device,):
    # 一个预测长度只保留末端值，对应发布时刻后的指定分钟数。
    model, model_config, checkpoint_epoch = load_horizon_model(
        checkpoint_path, repository, device
    )
    point_stride = (
        model_config["point_interval_seconds"] // repository.base_interval_seconds
    )
    turbine_count = len(repository.series)
    turbine_id = torch.arange(turbine_count, dtype=torch.long, device=device)
    prediction_batches = []

    batch_starts = range(0, len(origin_indices), batch_size)
    horizon_minutes = (
        model_config["horizon_steps"] * model_config["point_interval_seconds"] // 60
    )
    batches = tqdm(
        batch_starts,
        total=(len(origin_indices) + batch_size - 1) // batch_size,
        desc=f"+{horizon_minutes}min",
    )
    with torch.no_grad():
        for batch_start in batches:
            batch_indices = origin_indices[batch_start : batch_start + batch_size]
            past = build_history_batch(repository.panel, batch_indices, model_config["history_steps"], point_stride,)
            past_tensor = torch.from_numpy(past).to(device, non_blocking=True)
            turbine_batch = turbine_id.unsqueeze(0).expand(len(batch_indices), -1)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                prediction = model(past_tensor, turbine_batch)
            prediction_batches.append(prediction[..., -1].float().cpu().numpy())

    # 模型输出和真实值都用全部风机训练段的共享功率统计量还原为kW。
    prediction_scaled = np.concatenate(prediction_batches, axis=0)
    target_indices = origin_indices + model_config["horizon_steps"] * point_stride
    target_times = repository.times[target_indices]
    truth_scaled = repository.panel[
        :, target_indices, repository.power_index
    ].transpose(1, 0)
    power_mean = repository.series[0].power_mean
    power_std = repository.series[0].power_std
    prediction_kw = prediction_scaled * power_std + power_mean
    truth_kw = truth_scaled * power_std + power_mean

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return prediction_kw, truth_kw, target_times, model_config, checkpoint_epoch


def build_result_tables(args, repository, origin_indices: np.ndarray, horizon_minutes_values, device):
    # 指定长度的独立checkpoint在同一批发布时刻上评估。
    prediction_tables = []
    metric_rows = []
    origin_times = pd.to_datetime(repository.times[origin_indices])

    for requested_horizon_minutes in horizon_minutes_values:
        checkpoint_path = (
            args.checkpoint_root
            / f"h{args.checkpoint_history_steps}_p{requested_horizon_minutes}_s1_seed2026"
            / "best_checkpoint.pt"
        )
        prediction_kw, truth_kw, target_times, model_config, checkpoint_epoch = predict_horizon(checkpoint_path, repository, origin_indices, args.batch_size, device,)
        site_prediction_kw = prediction_kw.sum(axis=1)
        site_truth_kw = truth_kw.sum(axis=1)
        horizon_minutes = (
            model_config["horizon_steps"]
            * model_config["point_interval_seconds"]
            // 60
        )
        table = pd.DataFrame(
            {
                "issue_time": origin_times,
                "target_time": pd.to_datetime(target_times),
                "horizon_minutes": horizon_minutes,
                "predicted_power_mw": site_prediction_kw / 1000.0,
                "actual_power_mw": site_truth_kw / 1000.0,
            }
        )
        prediction_tables.append(table)

        site_metrics = point_metrics(site_prediction_kw, site_truth_kw)
        turbine_metrics = point_metrics(prediction_kw, truth_kw)
        metric_rows.append(
            {
                "history_steps": model_config["history_steps"],
                "horizon_steps": model_config["horizon_steps"],
                "horizon_minutes": horizon_minutes,
                "checkpoint_epoch": checkpoint_epoch,
                "samples": len(origin_indices),
                "site_acc30": site_metrics["strict_acc30"],
                "site_mape": site_metrics["mape"],
                "turbine_acc30": turbine_metrics["strict_acc30"],
                "turbine_mape": turbine_metrics["mape"],
                "result_name": model_config["result_name"],
            }
        )

    predictions = pd.concat(prediction_tables, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    return predictions, metrics
