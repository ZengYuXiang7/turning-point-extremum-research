from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from config.settings import STRICT_ACC_MIN_POWER_KW, STRICT_ACC_TOLERANCE


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
    # 齐次扁平记录：总体、分机、macro16
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
