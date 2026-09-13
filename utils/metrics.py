from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from config.settings import STRICT_ACC_MIN_POWER_KW, STRICT_ACC_TOLERANCE

DTW_MAX_POINTS = 256


def downsample_curve(values: np.ndarray) -> np.ndarray:
    # 长视界按等宽区间平均，控制 DTW 的计算规模
    point_count = min(values.size, DTW_MAX_POINTS)
    if point_count == values.size:
        return values

    edges = np.linspace(0, values.size, point_count + 1, dtype=np.int64)
    sampled = np.empty(point_count, dtype=np.float64)
    for index in range(point_count):
        left = edges[index]
        right = edges[index + 1]
        sampled[index] = np.mean(values[left:right])
    return sampled


def dtw_alignment_path(
    truth: np.ndarray,
    prediction: np.ndarray,
) -> tuple[float, np.ndarray]:
    # 用滚动代价行和方向矩阵恢复最优 DTW 路径
    previous = np.full(prediction.size + 1, np.inf, dtype=np.float64)
    previous[0] = 0.0
    directions = np.empty((truth.size, prediction.size), dtype=np.int8)

    for truth_index in range(truth.size):
        current = np.full(prediction.size + 1, np.inf, dtype=np.float64)
        for prediction_index in range(1, prediction.size + 1):
            local_cost = abs(truth[truth_index] - prediction[prediction_index - 1])
            diagonal_cost = previous[prediction_index - 1]
            vertical_cost = previous[prediction_index]
            horizontal_cost = current[prediction_index - 1]

            if diagonal_cost <= vertical_cost:
                if diagonal_cost <= horizontal_cost:
                    best_previous = diagonal_cost
                    direction = 0
                else:
                    best_previous = horizontal_cost
                    direction = 2
            elif vertical_cost <= horizontal_cost:
                best_previous = vertical_cost
                direction = 1
            else:
                best_previous = horizontal_cost
                direction = 2

            current[prediction_index] = local_cost + best_previous
            directions[truth_index, prediction_index - 1] = direction
        previous = current

    # 方向 0/1/2 分别表示对角、真实轴、预测轴的前驱
    max_path_length = truth.size + prediction.size - 1
    reversed_path = np.empty((max_path_length, 2), dtype=np.int64)
    truth_index = truth.size - 1
    prediction_index = prediction.size - 1
    path_index = 0

    while True:
        reversed_path[path_index, 0] = truth_index
        reversed_path[path_index, 1] = prediction_index
        if truth_index == 0 and prediction_index == 0:
            break

        direction = directions[truth_index, prediction_index]
        if direction == 0:
            truth_index -= 1
            prediction_index -= 1
        elif direction == 1:
            truth_index -= 1
        else:
            prediction_index -= 1
        path_index += 1

    path = np.flip(reversed_path[: path_index + 1], axis=0).copy()
    distance = float(previous[-1])
    return distance, path


def normalized_dtw(prediction: np.ndarray, truth: np.ndarray) -> float:
    distance, _ = dtw_alignment_path(truth, prediction)
    normalized_distance = distance / prediction.size
    return float(normalized_distance)


def curve_dtw(prediction: np.ndarray, truth: np.ndarray) -> float:
    # 对所有窗口的平均预测轨迹和平均真实轨迹计算 DTW
    prediction_curve = np.mean(prediction, axis=0, dtype=np.float64)
    truth_curve = np.mean(truth, axis=0, dtype=np.float64)
    prediction_curve = downsample_curve(prediction_curve)
    truth_curve = downsample_curve(truth_curve)
    distance = normalized_dtw(prediction_curve, truth_curve)
    return distance


def sequence_metric_arrays(
    y_true: np.ndarray | torch.Tensor,
    y_pred: np.ndarray | torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    # 在评估边界统一转为 CPU NumPy 数组
    if isinstance(y_true, torch.Tensor):
        true_values = y_true.detach().to(device="cpu", dtype=torch.float64).numpy()
    else:
        true_values = np.asarray(y_true, dtype=np.float64)

    if isinstance(y_pred, torch.Tensor):
        predicted_values = y_pred.detach().to(device="cpu", dtype=torch.float64).numpy()
    else:
        predicted_values = np.asarray(y_pred, dtype=np.float64)

    if true_values.shape != predicted_values.shape:
        raise ValueError(
            "y_true and y_pred must have the same shape, got "
            f"{true_values.shape} and {predicted_values.shape}"
        )
    if true_values.ndim != 2:
        raise ValueError(
            f"y_true and y_pred must have shape [bs, seq_len], got {true_values.shape}"
        )
    if true_values.shape[1] <= 1:
        raise ValueError(
            f"seq_len must be greater than 1, got {true_values.shape[1]}"
        )
    if not np.all(np.isfinite(true_values)):
        raise ValueError("y_true contains NaN or Inf")
    if not np.all(np.isfinite(predicted_values)):
        raise ValueError("y_pred contains NaN or Inf")

    return true_values, predicted_values


def batched_dtw_metrics(
    y_true: np.ndarray | torch.Tensor,
    y_pred: np.ndarray | torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    true_values, predicted_values = sequence_metric_arrays(y_true, y_pred)

    # 每条序列独立恢复 DTW path，同时计算 DTW 和 TDI
    batch_size = true_values.shape[0]
    sequence_length = true_values.shape[1]
    dtw_values = np.empty(batch_size, dtype=np.float64)
    tdi_values = np.empty(batch_size, dtype=np.float64)
    for batch_index in range(batch_size):
        distance, path = dtw_alignment_path(
            true_values[batch_index],
            predicted_values[batch_index],
        )
        offsets = path[:, 0] - path[:, 1]
        squared_offsets = offsets.astype(np.float64) ** 2
        dtw_values[batch_index] = distance / sequence_length
        tdi_values[batch_index] = np.sum(squared_offsets) / sequence_length**2

    return dtw_values, tdi_values


def dtw(
    y_true: np.ndarray | torch.Tensor,
    y_pred: np.ndarray | torch.Tensor,
    reduction: str = "mean",
) -> float | np.ndarray:
    if reduction not in ("mean", "none"):
        raise ValueError(f"reduction must be 'mean' or 'none', got {reduction!r}")

    dtw_values, _ = batched_dtw_metrics(y_true, y_pred)

    if reduction == "none":
        return dtw_values

    mean_dtw = float(np.mean(dtw_values))
    return mean_dtw


def tdi(
    y_true: np.ndarray | torch.Tensor,
    y_pred: np.ndarray | torch.Tensor,
    reduction: str = "mean",
) -> float | np.ndarray:
    if reduction not in ("mean", "none"):
        raise ValueError(f"reduction must be 'mean' or 'none', got {reduction!r}")

    _, tdi_values = batched_dtw_metrics(y_true, y_pred)

    if reduction == "none":
        return tdi_values

    mean_tdi = float(np.mean(tdi_values))
    return mean_tdi


def point_metrics(prediction: np.ndarray, truth: np.ndarray) -> dict:
    prediction = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    error = prediction - truth
    # Acc30 / MAPE 共用有效功率阈值，低功率点不做相对误差
    mask = truth > STRICT_ACC_MIN_POWER_KW
    passed = np.abs(error[mask]) <= STRICT_ACC_TOLERANCE * truth[mask]
    mape = float(np.mean(np.abs(error[mask]) / truth[mask]) * 100.0)
    return {
        "strict_acc30": float(np.mean(passed) * 100.0),
        "strict_valid_points": int(mask.sum()),
        "total_points": int(truth.size),
        "strict_valid_fraction": float(mask.mean()),
        "mae_kw": float(np.mean(np.abs(error))),
        "mse_kw2": float(np.mean(error ** 2)),
        "rmse_kw": float(np.sqrt(np.mean(error ** 2))),
        "mape": mape,
    }


def metric_bundle(prediction: np.ndarray, truth: np.ndarray, turbine_ids: np.ndarray):
    # 汇总总体、分机和 macro16 点误差指标，不执行 DTW/TDI。
    rows = []

    curve_overall = point_metrics(prediction, truth)
    rows.append(
        {
            "kind": "curve_overall",
            "turbine_id": -1,
            "strict_acc30": curve_overall["strict_acc30"],
            "strict_valid_points": curve_overall["strict_valid_points"],
            "total_points": curve_overall["total_points"],
            "strict_valid_fraction": curve_overall["strict_valid_fraction"],
            "mae_kw": curve_overall["mae_kw"],
            "mse_kw2": curve_overall["mse_kw2"],
            "rmse_kw": curve_overall["rmse_kw"],
            "mape": curve_overall["mape"],
        }
    )

    endpoint_overall = point_metrics(prediction[:, -1], truth[:, -1])
    rows.append(
        {
            "kind": "endpoint_overall",
            "turbine_id": -1,
            "strict_acc30": endpoint_overall["strict_acc30"],
            "strict_valid_points": endpoint_overall["strict_valid_points"],
            "total_points": endpoint_overall["total_points"],
            "strict_valid_fraction": endpoint_overall["strict_valid_fraction"],
            "mae_kw": endpoint_overall["mae_kw"],
            "mse_kw2": endpoint_overall["mse_kw2"],
            "rmse_kw": endpoint_overall["rmse_kw"],
            "mape": endpoint_overall["mape"],
        }
    )

    curve_acc30_values = []
    curve_mae_values = []
    curve_mse_values = []
    curve_rmse_values = []
    curve_mape_values = []
    endpoint_acc30_values = []
    endpoint_mae_values = []
    endpoint_mse_values = []
    endpoint_rmse_values = []
    endpoint_mape_values = []

    for turbine_index in range(16):
        mask = turbine_ids == turbine_index
        curve = point_metrics(prediction[mask], truth[mask])
        endpoint = point_metrics(prediction[mask, -1], truth[mask, -1])
        rows.append(
            {
                "kind": "per_turbine_curve",
                "turbine_id": turbine_index + 1,
                "strict_acc30": curve["strict_acc30"],
                "strict_valid_points": curve["strict_valid_points"],
                "total_points": curve["total_points"],
                "strict_valid_fraction": curve["strict_valid_fraction"],
                "mae_kw": curve["mae_kw"],
                "mse_kw2": curve["mse_kw2"],
                "rmse_kw": curve["rmse_kw"],
                "mape": curve["mape"],
            }
        )
        rows.append(
            {
                "kind": "per_turbine_endpoint",
                "turbine_id": turbine_index + 1,
                "strict_acc30": endpoint["strict_acc30"],
                "strict_valid_points": endpoint["strict_valid_points"],
                "total_points": endpoint["total_points"],
                "strict_valid_fraction": endpoint["strict_valid_fraction"],
                "mae_kw": endpoint["mae_kw"],
                "mse_kw2": endpoint["mse_kw2"],
                "rmse_kw": endpoint["rmse_kw"],
                "mape": endpoint["mape"],
            }
        )
        curve_acc30_values.append(curve["strict_acc30"])
        curve_mae_values.append(curve["mae_kw"])
        curve_mse_values.append(curve["mse_kw2"])
        curve_rmse_values.append(curve["rmse_kw"])
        curve_mape_values.append(curve["mape"])
        endpoint_acc30_values.append(endpoint["strict_acc30"])
        endpoint_mae_values.append(endpoint["mae_kw"])
        endpoint_mse_values.append(endpoint["mse_kw2"])
        endpoint_rmse_values.append(endpoint["rmse_kw"])
        endpoint_mape_values.append(endpoint["mape"])

    rows.append(
        {
            "kind": "macro16_curve",
            "turbine_id": -1,
            "strict_acc30": float(np.nanmean(curve_acc30_values)),
            "strict_valid_points": -1,
            "total_points": -1,
            "strict_valid_fraction": float("nan"),
            "mae_kw": float(np.nanmean(curve_mae_values)),
            "mse_kw2": float(np.nanmean(curve_mse_values)),
            "rmse_kw": float(np.nanmean(curve_rmse_values)),
            "mape": float(np.nanmean(curve_mape_values)),
        }
    )
    rows.append(
        {
            "kind": "macro16_endpoint",
            "turbine_id": -1,
            "strict_acc30": float(np.nanmean(endpoint_acc30_values)),
            "strict_valid_points": -1,
            "total_points": -1,
            "strict_valid_fraction": float("nan"),
            "mae_kw": float(np.nanmean(endpoint_mae_values)),
            "mse_kw2": float(np.nanmean(endpoint_mse_values)),
            "rmse_kw": float(np.nanmean(endpoint_rmse_values)),
            "mape": float(np.nanmean(endpoint_mape_values)),
        }
    )
    return rows


def save_metrics(root: Path, metrics_rows: list) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "metrics.json").write_text(
        json.dumps(metrics_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (root / "per_turbine_metrics.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "turbine_id",
                "scope",
                "strict_acc30",
                "valid_points",
                "total_points",
                "mae_kw",
                "mse_kw2",
                "rmse_kw",
                "mape",
            ]
        )
        for row in metrics_rows:
            if row["kind"] == "per_turbine_curve":
                scope = "curve"
            elif row["kind"] == "per_turbine_endpoint":
                scope = "endpoint"
            else:
                continue
            writer.writerow(
                [
                    row["turbine_id"],
                    scope,
                    row["strict_acc30"],
                    row["strict_valid_points"],
                    row["total_points"],
                    row["mae_kw"],
                    row["mse_kw2"],
                    row["rmse_kw"],
                    row["mape"],
                ]
            )
