import argparse

from config import (
    add_shared_arguments,
    apply_history_feature_path,
    finalize_arguments,
)


def add_task_arguments(parser: argparse.ArgumentParser) -> None:
    # Single 程序只声明单风机与天气模型。
    parser.add_argument("--model", choices=("PatchMLP", "PatchMLPAllFeatures", "DLinear", "DLinearAllFeatures", "DLinearCorrelatedFeatures", "QwenMLP", "TimerWeatherMLP",), required=True,)
    parser.add_argument("--scenario", choices=("NoFutureWeather", "OracleFutureWeather", "ForecastWeather", "PredictedFutureWeather", "ProvidedFutureWeather",), required=True,)
    parser.add_argument("--dataset-name", type=str, default="GuangningWindPower15min")
    parser.add_argument("--history-feature-path", type=str, default="")
    parser.add_argument("--weather-forecast-dir", type=str, default="")
    parser.add_argument("--provided-weather-dir", type=str, default="")
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


def parse_args():
    parser = argparse.ArgumentParser()
    add_shared_arguments(parser)
    add_task_arguments(parser)
    args = finalize_arguments(parser)
    apply_history_feature_path(args.history_feature_path)
    return args
