from __future__ import annotations

import argparse
import os

from config.settings import (
    HISTORY_STEPS,
    PROJECT_ROOT,
    SAMPLE_SECONDS,
    SEED,
    WINDOW_STRIDE_STEPS,
    apply_history_steps,
    apply_sample_seconds,
    apply_window_stride_steps,
)

# All model metadata and dynamic modules must stay inside this project.
os.environ.setdefault(
    "HF_HOME", str(PROJECT_ROOT / "models" / "pretrained" / ".hf-cache")
)
os.environ.setdefault(
    "HF_MODULES_CACHE",
    str(PROJECT_ROOT / "models" / "pretrained" / ".hf-modules"),
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=(
            "PatchMLP",
            "PatchMLPAllFeatures",
            "DLinear",
            "DLinearAllFeatures",
            "StockEcho",
            "QwenMLP",
            "TimerWeatherMLP",
        ),
        required=True,
    )
    parser.add_argument(
        "--scenario",
        choices=(
            "NoFutureWeather",
            "OracleFutureWeather",
            "ForecastWeather",
            "PredictedFutureWeather",
        ),
        required=True,
    )
    parser.add_argument("--loss", choices=("MSE", "DBLoss", "MSEAcc30"), required=True)
    parser.add_argument("--dbloss-weight", type=float, default=0.5)
    parser.add_argument("--sample-seconds", type=int, default=SAMPLE_SECONDS)
    # 历史、预测和起点间隔统一使用采样点数
    parser.add_argument("--history-steps", type=int, default=HISTORY_STEPS)
    parser.add_argument("--horizon-steps", type=int, default=1)
    parser.add_argument("--window-stride-steps", type=int, default=WINDOW_STRIDE_STEPS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--show-progress", type=int, default=1, choices=(0, 1))
    parser.add_argument("--run-dir", type=str, default="")
    parser.add_argument("--dataset-name", type=str, default="GuangningWindPower15min")
    parser.add_argument("--result-name", type=str, default="")
    # 程序2读取程序1写出的 weather_forecast_{split}.npz 目录
    parser.add_argument("--weather-forecast-dir", type=str, default="")
    parser.add_argument("--llm-path", type=str, default="")
    parser.add_argument("--llm-patch-length", type=int, default=4)
    parser.add_argument("--llm-patch-stride", type=int, default=2)
    parser.add_argument("--llm-bottleneck", type=int, default=512)
    parser.add_argument("--llm-gradient-checkpointing", type=int, choices=(0, 1), default=1)
    parser.add_argument("--timer-path", type=str, default="")
    parser.add_argument("--timer-patch-length", type=int, default=96)
    parser.add_argument("--timer-bottleneck", type=int, default=512)
    parser.add_argument("--timer-unfreeze-layers", type=int, default=2)
    parser.add_argument("--timer-backbone-learning-rate", type=float, default=1e-5)
    parser.add_argument("--timer-gradient-checkpointing", type=int, choices=(0, 1), default=0)
    parser.add_argument("--timer-residual-forecast", type=int, choices=(0, 1), default=1)
    args = parser.parse_args()

    # 未指定目录时按场景/模型/损失及历史窗和视界落盘
    if args.run_dir == "":
        args.run_dir = str(
            PROJECT_ROOT
            / ".runs"
            / "WeatherComparison"
            / args.scenario
            / args.model
            / args.loss
            / f"history_{args.history_steps}steps"
            / f"horizon_{args.horizon_steps}steps"
            / f"stride_{args.window_stride_steps}steps"
            / f"seed{args.seed}"
        )
    return args


if __name__ == "__main__":
    arguments = parse_args()
    # 先定采样间隔、历史窗和起点间隔，再导入会绑定常量的训练链
    apply_sample_seconds(arguments.sample_seconds)
    apply_history_steps(arguments.history_steps)
    apply_window_stride_steps(arguments.window_stride_steps)
    from exp.trainer import train_and_test

    train_and_test(arguments)
