import argparse

from config import add_shared_arguments, apply_history_feature_path, finalize_arguments


def add_task_arguments(parser: argparse.ArgumentParser) -> None:
    # 多风机程序只声明联合面板模型与结构参数。
    parser.add_argument("--model", choices=("StockEcho", "MultiTurbine"), required=True,)
    parser.add_argument("--scenario", choices=("MultiTurbine",), required=True,)
    parser.add_argument("--dataset-name", type=str, default="GuangningWindPower16Turbine15min",)
    parser.add_argument("--satra-hidden-dim", type=int, default=64)
    parser.add_argument("--satra-depth", type=int, default=10)
    parser.add_argument("--satra-kernel-size", type=int, default=3)
    parser.add_argument("--satra-heads", type=int, default=4)
    parser.add_argument("--satra-tower-layers", type=int, default=1)
    parser.add_argument("--satra-dropout", type=float, default=0.1)
    parser.add_argument("--satra-expert-top-k", type=int, default=3)
    parser.add_argument("--revin", type=int, choices=(0, 1), default=1)
    parser.add_argument("--visualize-after-train", type=int, choices=(0, 1), default=1)
    parser.add_argument("--satra-use-spatial-relation", type=int, choices=(0, 1), default=1)
    parser.add_argument("--satra-use-dtw-prior", type=int, choices=(0, 1), default=0)
    parser.add_argument("--satra-prior-top-k", type=int, default=4)
    parser.add_argument("--satra-prior-candidate-k", type=int, default=8)
    parser.add_argument("--satra-prior-downsample", type=int, default=64)
    parser.add_argument("--satra-prior-band", type=int, default=6)


def parse_args():
    parser = argparse.ArgumentParser()
    add_shared_arguments(parser)
    add_task_arguments(parser)
    args = finalize_arguments(parser)
    apply_history_feature_path("")
    return args
