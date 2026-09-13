from __future__ import annotations

import argparse
import os

from config.settings import (
    HISTORY_MINUTES,
    PROJECT_ROOT,
    SEED,
    apply_history,
    apply_window_stride,
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
        choices=("PatchMLP", "DLinear", "StockEcho", "QwenMLP", "TimerWeatherMLP", "MLP"),
        required=True,
    )
    parser.add_argument(
        "--scenario",
        choices=(
            "NoFutureWeather",
            "OracleFutureWeather",
            "ForecastWeather",
            "PredictedFutureWeather",
            "WeatherToPower",
        ),
        required=True,
    )
    parser.add_argument("--loss", choices=("MSE", "DBLoss", "MSEAcc30"), required=True)
    # 视界分钟数不设硬编码上限；正式协议为15 / 720 / 1440 / 2880
    parser.add_argument("--horizon", type=int, default=15)
    # 历史窗分钟数；默认与 settings.HISTORY_MINUTES 一致
    parser.add_argument("--history", type=int, default=HISTORY_MINUTES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--show-progress", type=int, default=1, choices=(0, 1))
    parser.add_argument("--run-dir", type=str, default="")
    # 程序2读取程序1写出的 weather_forecast_{split}.npz 目录
    parser.add_argument("--weather-forecast-dir", type=str, default="")
    parser.add_argument("--llm-path", type=str, default="")
    parser.add_argument("--llm-patch-length", type=int, default=60)
    parser.add_argument("--llm-patch-stride", type=int, default=30)
    parser.add_argument("--llm-bottleneck", type=int, default=512)
    parser.add_argument("--llm-gradient-checkpointing", type=int, choices=(0, 1), default=1)
    parser.add_argument("--timer-path", type=str, default="")
    parser.add_argument("--timer-patch-length", type=int, default=96)
    parser.add_argument("--timer-bottleneck", type=int, default=512)
    parser.add_argument("--timer-unfreeze-layers", type=int, default=2)
    parser.add_argument("--timer-backbone-learning-rate", type=float, default=1e-5)
    parser.add_argument("--timer-gradient-checkpointing", type=int, choices=(0, 1), default=0)
    parser.add_argument("--timer-residual-forecast", type=int, choices=(0, 1), default=1)
    parser.add_argument("--window-stride-steps", type=int, default=1)
    args = parser.parse_args()

    # 未指定目录时按场景/模型/损失落盘；预报任务再叠 history/horizon
    if args.run_dir == "":
        if args.scenario == "WeatherToPower":
            args.run_dir = str(
                PROJECT_ROOT
                / ".runs"
                / "WeatherComparison"
                / "10s"
                / args.scenario
                / args.model
                / args.loss
                / f"seed{args.seed}"
            )
        else:
            args.run_dir = str(
                PROJECT_ROOT
                / ".runs"
                / "WeatherComparison"
                / "10s"
                / args.scenario
                / args.model
                / args.loss
                / f"history_{args.history}m"
                / f"horizon_{args.horizon}m"
                / f"seed{args.seed}"
            )
    return args


if __name__ == "__main__":
    arguments = parse_args()
    # 先定历史窗，再导入会绑定 HISTORY_STEPS 等常量的训练链
    apply_history(arguments.history)
    apply_window_stride(arguments.window_stride_steps)
    from exp.trainer import train_and_test

    train_and_test(arguments)
