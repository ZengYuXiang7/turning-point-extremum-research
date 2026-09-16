from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import torch

from config import PROJECT_ROOT
from tasks.multi_turbine.dataset import MultiTurbineRelationRepository
from tasks.multi_turbine.sampled_evaluation import (
    build_result_tables,
    select_issue_origins,
)
from tasks.multi_turbine.visualization import visualize_checkpoint_predictions


DEFAULT_CHECKPOINT_ROOT = (
    PROJECT_ROOT
    / ".runs"
    / "WeatherComparison"
    / "MultiTurbine"
    / "MultiTurbine"
    / "MSE"
    / "GuangningWindPower16Turbine1min"
    / "pretrain_e10_m0p3_no_spatial"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "multi_turbine_15min_forecast"
DEFAULT_HORIZON_MINUTES = (15, 30, 45, 60)
PLOT_DURATION_HOURS = (31 * 24, 7 * 24, 24, 12, 8, 4)


def parse_args():
    parser = argparse.ArgumentParser(
        description="默认生成滑动窗口图和每15分钟发布的多风机预测图。"
    )
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--checkpoint-history-steps", type=int, default=240)
    parser.add_argument("--horizon-minutes", type=int, nargs="+", default=DEFAULT_HORIZON_MINUTES)
    parser.add_argument("--issue-interval-minutes", type=int, default=15)
    parser.add_argument("--evaluation-start", type=str, default="2026-08-01 09:45:00")
    parser.add_argument("--evaluation-end", type=str, default="2026-09-01 09:45:00")
    parser.add_argument("--plot-start", type=str, default="2026-08-01 09:45:00")
    parser.add_argument("--sliding-split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--sliding-seed", type=int, default=20260913)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    return args


def save_sliding_window_figures(args, output_batch_name: str):
    # 每个checkpoint先调用原有联合滑动窗口绘图函数。
    output_paths = []
    for horizon_minutes in args.horizon_minutes:
        run_dir = args.checkpoint_root / f"h{args.checkpoint_history_steps}_p{horizon_minutes}_s1_seed2026"
        output_path = visualize_checkpoint_predictions(run_dir, args.sliding_split, args.sliding_seed, args.device, args.output_dir, output_batch_name)
        output_paths.append(output_path)
    return output_paths


def format_duration(duration_hours: int) -> str:
    if duration_hours % 24 == 0:
        duration_days = duration_hours // 24
        duration_label = f"{duration_days}d"
    else:
        duration_label = f"{duration_hours}h"
    return duration_label


def save_site_figure(predictions: pd.DataFrame, metric: pd.Series, output_pdf_path: Path, args, duration_hours: int) -> Path:
    # 预测点按目标时刻对齐，并在图内标注完整评估时段的场站指标。
    horizon_minutes = int(metric["horizon_minutes"])
    site_acc30 = float(metric["site_acc30"])
    site_mape = float(metric["site_mape"])
    evaluation_samples = int(metric["samples"])
    plot_start = pd.Timestamp(args.plot_start)
    plot_end = plot_start + pd.Timedelta(hours=duration_hours)
    selected = predictions[
        (predictions["issue_time"] >= plot_start)
        & (predictions["issue_time"] < plot_end)
    ]
    actual = selected[["target_time", "actual_power_mw"]]
    actual = actual.drop_duplicates("target_time").sort_values("target_time")

    figure, axis = plt.subplots(figsize=(18, 7))
    axis.plot(
        actual["target_time"],
        actual["actual_power_mw"],
        color="#111827",
        linewidth=1.6,
        label="Actual",
    )
    axis.plot(
        selected["target_time"],
        selected["predicted_power_mw"],
        color="#2563eb",
        linewidth=1.0,
        alpha=0.9,
        marker="o",
        markersize=2.2,
        markeredgewidth=0.0,
        label=f"Forecast target points (+{horizon_minutes} min)",
    )

    duration_label = format_duration(duration_hours)
    axis.set_title(f"Guangning wind farm: {args.issue_interval_minutes}-minute issue sampling ({duration_label})")
    axis.set_xlabel("Target time (forecast points plotted at valid time)")
    axis.set_ylabel("Power (MW)")
    axis.grid(alpha=0.25, linewidth=0.5)
    axis.legend(ncol=5, frameon=False)
    metric_text = (
        f"31-day {args.issue_interval_minutes}-min issue metrics ({evaluation_samples} issues)\n"
        f"Site Acc30: {site_acc30:.2f}%\n"
        f"Site MAPE: {site_mape:.2f}%"
    )
    axis.text(
        0.985,
        0.975,
        metric_text,
        transform=axis.transAxes,
        horizontalalignment="right",
        verticalalignment="top",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "#94a3b8", "alpha": 0.9},
    )
    locator = mdates.AutoDateLocator()
    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    figure.tight_layout()

    figure.savefig(output_pdf_path, bbox_inches="tight")
    plt.close(figure)
    return output_pdf_path


def save_sampled_task_outputs(predictions: pd.DataFrame, metrics: pd.DataFrame, args, output_batch_name: str):
    # 每个真实history/horizon配置独立保存预测表、指标表和第二张PDF。
    predictions_paths = []
    metrics_paths = []
    pdf2_paths = []
    for metric_index in range(len(metrics)):
        history_steps = int(metrics.iloc[metric_index]["history_steps"])
        horizon_steps = int(metrics.iloc[metric_index]["horizon_steps"])
        horizon_minutes = int(metrics.iloc[metric_index]["horizon_minutes"])
        result_name = str(metrics.iloc[metric_index]["result_name"])
        horizon_predictions = predictions[predictions["horizon_minutes"] == horizon_minutes]
        horizon_metrics = metrics.iloc[[metric_index]]
        task_output_directory = args.output_dir / f"h{history_steps}_p{horizon_steps}" / output_batch_name
        task_output_directory.mkdir(parents=True, exist_ok=True)

        predictions_path = task_output_directory / "site_predictions.csv"
        metrics_path = task_output_directory / "metrics.csv"
        horizon_predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")
        horizon_metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
        predictions_paths.append(predictions_path)
        metrics_paths.append(metrics_path)

        # 同一组31天评估结果同时生成六种观察时长的第二类图。
        for duration_hours in PLOT_DURATION_HOURS:
            duration_label = format_duration(duration_hours)
            output_pdf_path = task_output_directory / f"{result_name}_pdf2_{duration_label}.pdf"
            output_path = save_site_figure(horizon_predictions, metrics.iloc[metric_index], output_pdf_path, args, duration_hours)
            pdf2_paths.append(output_path)
    return predictions_paths, metrics_paths, pdf2_paths


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    output_batch_name = datetime.now().strftime("%m%d-%H%M")

    # 同一Python入口先画滑动窗口图，再画固定间隔连续发布图。
    pdf1_paths = save_sliding_window_figures(args, output_batch_name)

    # Repository从全部风机训练段重建共享StandardScaler，与checkpoint训练保持一致。
    repository = MultiTurbineRelationRepository(all_features=True)
    origin_indices = select_issue_origins(repository.times, args.evaluation_start, args.evaluation_end, args.issue_interval_minutes)
    predictions, metrics = build_result_tables(args, repository, origin_indices, args.horizon_minutes, device)

    # 按checkpoint真实h/p配置分别保存可复核的采样预测、指标和场站级曲线。
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_paths, metrics_paths, pdf2_paths = save_sampled_task_outputs(predictions, metrics, args, output_batch_name)

    print(metrics.to_string(index=False))
    for predictions_path in predictions_paths:
        print(f"predictions={predictions_path}")
    for metrics_path in metrics_paths:
        print(f"metrics={metrics_path}")
    for output_path in pdf1_paths:
        print(f"figure_pdf1={output_path}")
    for output_path in pdf2_paths:
        print(f"figure_pdf2={output_path}")


if __name__ == "__main__":
    main()
