from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / ".runs" / "WeatherComparison"
OUTPUT = ROOT / "visualizations" / "weather_comparison"


def prediction_file(grain, scenario, model, loss, horizon, history):
    return (
        RUNS
        / grain
        / scenario
        / model
        / loss
        / f"history_{history}m"
        / f"horizon_{horizon}m"
        / "seed2026"
        / "test_predictions.npz"
    )


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)
    generated = 0
    for grain in ("10s",):
        for model in ("PatchMLP", "DLinear", "StockEcho"):
            for loss in ("MSE", "DBLoss"):
                for history in (240,):
                    for horizon in (15, 720, 1440, 2880):
                        no_path = prediction_file(
                            grain, "NoFutureWeather", model, loss, horizon, history
                        )
                        oracle_path = prediction_file(
                            grain, "OracleFutureWeather", model, loss, horizon, history
                        )
                        if not no_path.exists() or not oracle_path.exists():
                            continue
                        no = np.load(no_path)
                        oracle = np.load(oracle_path)
                        choices = []
                        for turbine in (1, 6, 11, 16):
                            candidates = np.flatnonzero(no["turbine_id"] == turbine)
                            choices.append(int(rng.choice(candidates)))
                        figure, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
                        for axis_index, index in enumerate(choices):
                            axis = axes.flat[axis_index]
                            truth = no["truth"][index]
                            steps = truth.shape[0]
                            x = np.arange(1, steps + 1)
                            axis.plot(x, truth, color="black", linewidth=1.5, label="Truth")
                            axis.plot(
                                x,
                                no["prediction"][index],
                                color="#377eb8",
                                linewidth=1.1,
                                label="NoFutureWeather",
                            )
                            axis.plot(
                                x,
                                oracle["prediction"][index],
                                color="#e41a1c",
                                linewidth=1.1,
                                label="OracleFutureWeather",
                            )
                            axis.fill_between(
                                x,
                                truth * 0.7,
                                truth * 1.3,
                                where=truth > 100,
                                color="gray",
                                alpha=0.13,
                                label="Strict ACC30 band",
                            )
                            start = np.datetime64(int(no["target_start_ns"][index]), "ns")
                            axis.set_title(f"Turbine {int(no['turbine_id'][index])}, start {start}")
                            axis.set_xlabel(f"Forecast step ({grain})")
                            axis.set_ylabel("Power (kW)")
                            axis.grid(alpha=0.2)
                        handles, labels = axes.flat[0].get_legend_handles_labels()
                        figure.legend(handles, labels, loc="upper center", ncol=4)
                        figure.suptitle(
                            f"{model} + {loss}, history={history}m, horizon={horizon} min ({grain} grain)"
                        )
                        figure.savefig(
                            OUTPUT
                            / f"{grain}_{model}_{loss}_history_{history}m_horizon_{horizon}m.png",
                            dpi=160,
                        )
                        plt.close(figure)
                        generated += 1
    print({"generated": generated, "output": str(OUTPUT)})


if __name__ == "__main__":
    main()
