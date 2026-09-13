from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / ".runs" / "WeatherComparison"
RESULTS = ROOT / "results" / "WeatherComparison"


def main():
    rows = []

    # 旧目录布局：无 history 层，实验时固定 HISTORY_MINUTES=240
    for final_path in RUNS.glob("*/*/*/*/horizon_*m/seed2026/final.json"):
        payload = json.loads(final_path.read_text(encoding="utf-8"))
        if payload["status"] != "complete":
            continue

        # 从扁平 metrics 记录取出汇总字段
        metrics_rows = payload["metrics"]
        for row in metrics_rows:
            if row["kind"] == "curve_overall":
                curve_strict_acc30 = row["strict_acc30"]
                curve_valid_points = row["strict_valid_points"]
                curve_rmse_kw = row["rmse_kw"]
                curve_mae_kw = row["mae_kw"]
            if row["kind"] == "endpoint_overall":
                endpoint_strict_acc30 = row["strict_acc30"]
                endpoint_rmse_kw = row["rmse_kw"]
            if row["kind"] == "macro16_curve":
                macro16_curve_strict_acc30 = row["strict_acc30"]

        rows.append(
            {
                "grain": payload["grain"],
                "scenario": payload["scenario"],
                "model": payload["model"],
                "loss": payload["loss"],
                "history_minutes": 240,
                "horizon_minutes": payload["horizon_minutes"],
                "sample_seconds": payload["sample_seconds"],
                "best_epoch": payload["best_epoch"],
                "curve_strict_acc30": curve_strict_acc30,
                "curve_valid_points": curve_valid_points,
                "curve_rmse_kw": curve_rmse_kw,
                "curve_mae_kw": curve_mae_kw,
                "endpoint_strict_acc30": endpoint_strict_acc30,
                "endpoint_rmse_kw": endpoint_rmse_kw,
                "macro16_curve_strict_acc30": macro16_curve_strict_acc30,
            }
        )

    # 新目录布局：grain/scenario/model/loss/history_*/horizon_*/seed2026/final.json
    for final_path in RUNS.glob("*/*/*/*/history_*m/horizon_*m/seed2026/final.json"):
        payload = json.loads(final_path.read_text(encoding="utf-8"))
        if payload["status"] != "complete":
            continue

        metrics_rows = payload["metrics"]
        for row in metrics_rows:
            if row["kind"] == "curve_overall":
                curve_strict_acc30 = row["strict_acc30"]
                curve_valid_points = row["strict_valid_points"]
                curve_rmse_kw = row["rmse_kw"]
                curve_mae_kw = row["mae_kw"]
            if row["kind"] == "endpoint_overall":
                endpoint_strict_acc30 = row["strict_acc30"]
                endpoint_rmse_kw = row["rmse_kw"]
            if row["kind"] == "macro16_curve":
                macro16_curve_strict_acc30 = row["strict_acc30"]

        rows.append(
            {
                "grain": payload["grain"],
                "scenario": payload["scenario"],
                "model": payload["model"],
                "loss": payload["loss"],
                "history_minutes": payload["history_minutes"],
                "horizon_minutes": payload["horizon_minutes"],
                "sample_seconds": payload["sample_seconds"],
                "best_epoch": payload["best_epoch"],
                "curve_strict_acc30": curve_strict_acc30,
                "curve_valid_points": curve_valid_points,
                "curve_rmse_kw": curve_rmse_kw,
                "curve_mae_kw": curve_mae_kw,
                "endpoint_strict_acc30": endpoint_strict_acc30,
                "endpoint_rmse_kw": endpoint_rmse_kw,
                "macro16_curve_strict_acc30": macro16_curve_strict_acc30,
            }
        )

    rows.sort(
        key=lambda row: (
            row["grain"],
            row["history_minutes"],
            row["horizon_minutes"],
            row["scenario"],
            row["model"],
            row["loss"],
        )
    )
    RESULTS.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (RESULTS / "all_results.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (RESULTS / "all_results.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    indexed = {}
    for row in rows:
        key = (
            row["grain"],
            row["scenario"],
            row["model"],
            row["loss"],
            row["history_minutes"],
            row["horizon_minutes"],
        )
        indexed[key] = row

    uplift = []
    histories = sorted({row["history_minutes"] for row in rows})
    for grain in ("10s",):
        for model in ("PatchMLP", "DLinear", "StockEcho"):
            for loss in ("MSE", "DBLoss", "MSEAcc30"):
                for history in histories:
                    for horizon in (15, 720, 1440, 2880):
                        historical_key = (
                            grain,
                            "NoFutureWeather",
                            model,
                            loss,
                            history,
                            horizon,
                        )
                        oracle_key = (
                            grain,
                            "OracleFutureWeather",
                            model,
                            loss,
                            history,
                            horizon,
                        )
                        if historical_key not in indexed:
                            continue
                        if oracle_key not in indexed:
                            continue
                        historical = indexed[historical_key]
                        oracle = indexed[oracle_key]
                        uplift.append(
                            {
                                "grain": grain,
                                "model": model,
                                "loss": loss,
                                "history_minutes": history,
                                "horizon_minutes": horizon,
                                "no_future_acc30": historical["curve_strict_acc30"],
                                "oracle_acc30": oracle["curve_strict_acc30"],
                                "oracle_uplift_percentage_points": (
                                    oracle["curve_strict_acc30"]
                                    - historical["curve_strict_acc30"]
                                ),
                                "no_future_rmse_kw": historical["curve_rmse_kw"],
                                "oracle_rmse_kw": oracle["curve_rmse_kw"],
                            }
                        )

    fields_uplift = list(uplift[0])
    with (RESULTS / "oracle_weather_uplift.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields_uplift)
        writer.writeheader()
        writer.writerows(uplift)

    print(
        json.dumps(
            {
                "completed_runs": len(rows),
                "paired_uplifts": len(uplift),
                "output": str(RESULTS),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
