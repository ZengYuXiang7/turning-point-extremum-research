from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DISPLAY_TURBINE_COUNT = 10


def save_prediction_figure(output_pdf_path: Path, config, split: str, selection_seed: int, checkpoint_epoch: int, panel_dataset_index: int, history_power_kw: np.ndarray, truth_power_kw: np.ndarray, prediction_power_kw: np.ndarray, display_turbine_id: np.ndarray,) -> None:
    # 两行五列展示同一个联合窗口内前10台风机的预测曲线。
    history_steps = history_power_kw.shape[1]
    horizon_steps = truth_power_kw.shape[1]
    point_minutes = config["point_interval_seconds"] / 60.0
    history_axis = np.arange(-history_steps + 1, 1) * point_minutes
    forecast_axis = np.arange(horizon_steps + 1) * point_minutes
    figure, axes = plt.subplots(2, 5, figsize=(25, 11.2))

    for panel_index in range(DISPLAY_TURBINE_COUNT):
        axis = axes.flat[panel_index]
        history_curve = history_power_kw[panel_index]
        truth_curve = np.concatenate((history_curve[-1:], truth_power_kw[panel_index]))
        prediction_curve = np.concatenate((history_curve[-1:], prediction_power_kw[panel_index]))
        history_artist, = axis.plot(
            history_axis,
            history_curve,
            color="#6b7280",
            linewidth=1.2,
            label="History",
        )
        truth_artist, = axis.plot(
            forecast_axis,
            truth_curve,
            color="#111827",
            linewidth=1.6,
            marker="o",
            markersize=3,
            label="True future",
        )
        prediction_artist, = axis.plot(
            forecast_axis,
            prediction_curve,
            color="#dc2626",
            linewidth=1.3,
            linestyle="--",
            marker="o",
            markersize=3,
            label="Predicted future",
        )
        axis.axvline(0.0, color="#374151", linewidth=0.8, linestyle=":")
        axis.set_title(f"Turbine {int(display_turbine_id[panel_index]):02d}", fontsize=12,)
        axis.set_xlabel("Minutes", fontsize=10)
        axis.grid(alpha=0.25, linewidth=0.5)
        axis.tick_params(labelsize=9)
        if panel_index == 0 or panel_index == 5:
            axis.set_ylabel("Power (kW)", fontsize=10)

    figure.legend([history_artist, truth_artist, prediction_artist], ["History", "True future", "Predicted future"], loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=3, frameon=False, fontsize=24, handlelength=3.0, handletextpad=0.8, columnspacing=2.0, markerscale=1.8,)
    figure.suptitle(f"{config['model']} / {split} / shared panel window {panel_dataset_index} / " f"seed {selection_seed} / epoch {checkpoint_epoch} / " f"seq_len={history_steps}, pred_len={horizon_steps}", fontsize=15, y=0.99,)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.89))
    figure.savefig(output_pdf_path, format="pdf", bbox_inches="tight")
    plt.close(figure)
