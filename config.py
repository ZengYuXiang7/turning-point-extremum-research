from __future__ import annotations

import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SEED = 2026
HISTORY_STEPS = 16

# 底层序列固定保留原始10秒点，模型点间跨度由运行参数设置。
BASE_INTERVAL_SECONDS = 10
POINT_INTERVAL_SECONDS = 15 * 60
POINT_STRIDE_STEPS = POINT_INTERVAL_SECONDS // BASE_INTERVAL_SECONDS
DATASET_ROOT = PROJECT_ROOT / "dataset" / "processed"
WINDOW_STRIDE_STEPS = 1
WINDOW_STRIDE_NS = WINDOW_STRIDE_STEPS * POINT_INTERVAL_SECONDS * 1_000_000_000
EXPECTED_DELTA_NS = BASE_INTERVAL_SECONDS * 1_000_000_000
HISTORY_FEATURE_PATH = ""
TEMPORAL_POOL_STEPS = 1
PATCH_LENGTHS = [4, 4, 2, 2]

# 按时间顺序将样本划分为训练、验证、测试集。
SPLIT_RATIOS = (7, 1, 2)

# 标准历史特征。
HISTORY_COLUMNS = [
    "风机-理论功率-计算",
    "风机-实时风速",
    "风机-实时风向_sin",
    "风机-实时风向_cos",
    "风机-环境温度",
    "机舱-舱内温度",
    "塔筒-塔底温度",
    "发电机-发电机转速",
    "传动链-主轴转速",
    "变桨轮毂-1#桨叶角度",
    "变桨轮毂-2#桨叶角度",
    "变桨轮毂-3#桨叶角度",
    "偏航系统-对风角度_sin",
    "偏航系统-对风角度_cos",
    "偏航系统-机舱位置_sin",
    "偏航系统-机舱位置_cos",
    "偏航系统-扭揽角度_sin",
    "偏航系统-扭揽角度_cos",
    "风机-P",
]

# 相关系数筛选的20个预测变量加历史功率。
CORRELATED_HISTORY_COLUMNS = [
    "变流器-网侧L1相电流",
    "变流器-网侧L2相电流",
    "变流器-网侧L3相电流",
    "风机-理论功率-计算",
    "风机-实时风速",
    "发电机-发电机定子U相线圈温度",
    "发电机-发电机定子V相线圈温度",
    "发电机-发电机定子W相线圈温度",
    "变桨轮毂-变桨电机1电流",
    "变桨轮毂-变桨电机2电流",
    "变桨轮毂-变桨电机3电流",
    "传动链-齿轮箱高速轴驱动端轴承温度",
    "传动链-齿轮箱高速轴非驱动端轴承温度",
    "发电机-发电机转速",
    "传动链-主轴转速",
    "传动链-齿轮箱中速轴非驱动端轴承温度",
    "传动链-齿轮箱入口油压",
    "机舱-舱内温度",
    "传动链-齿轮箱油路滤网前油压",
    "传动链-齿轮箱油温",
    "风机-P",
]

# 天气任务使用的未来气象与姿态特征。
WEATHER_COLUMNS = [
    "风机-实时风速",
    "风机-实时风向_sin",
    "风机-实时风向_cos",
    "风机-环境温度",
    "机舱-舱内温度",
    "塔筒-塔底温度",
    "偏航系统-对风角度_sin",
    "偏航系统-对风角度_cos",
    "偏航系统-机舱位置_sin",
    "偏航系统-机舱位置_cos",
    "偏航系统-扭揽角度_sin",
    "偏航系统-扭揽角度_cos",
]
WEATHER_INDICES = [HISTORY_COLUMNS.index(name) for name in WEATHER_COLUMNS]
POWER_INDEX = HISTORY_COLUMNS.index("风机-P")
STRICT_ACC_MIN_POWER_KW = 100.0
STRICT_ACC_TOLERANCE = 0.30


def apply_point_interval_seconds(seconds: int) -> None:
    # 设置相邻模型点跨度。
    global POINT_INTERVAL_SECONDS, POINT_STRIDE_STEPS, WINDOW_STRIDE_NS
    POINT_INTERVAL_SECONDS = seconds
    POINT_STRIDE_STEPS = seconds // BASE_INTERVAL_SECONDS
    WINDOW_STRIDE_NS = WINDOW_STRIDE_STEPS * seconds * 1_000_000_000


def apply_history_steps(steps: int) -> None:
    # 设置历史窗口点数。
    global HISTORY_STEPS
    HISTORY_STEPS = steps


def apply_window_stride_steps(steps: int) -> None:
    # 设置预测起点间隔。
    global WINDOW_STRIDE_STEPS, WINDOW_STRIDE_NS
    WINDOW_STRIDE_STEPS = steps
    WINDOW_STRIDE_NS = steps * POINT_INTERVAL_SECONDS * 1_000_000_000


def apply_history_feature_path(path: str) -> None:
    # 设置显式历史特征表。
    global HISTORY_FEATURE_PATH
    HISTORY_FEATURE_PATH = path


def add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    # 两类任务共享训练、窗口、产物与可视化参数。
    parser.add_argument("--mode", choices=("train", "test"), default="train")
    parser.add_argument("--loss", choices=("MSE", "DBLoss", "MSEAcc30"), required=True)
    parser.add_argument("--dbloss-weight", type=float, default=0.5)
    parser.add_argument("--point-interval-seconds", type=int, default=POINT_INTERVAL_SECONDS)
    parser.add_argument("--history-steps", type=int, default=HISTORY_STEPS)
    parser.add_argument("--horizon-steps", type=int, default=1)
    parser.add_argument("--window-stride-steps", type=int, default=WINDOW_STRIDE_STEPS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--pretrain", action="store_true", default=False)
    parser.add_argument("--pretrain-epochs", type=int, default=10)
    parser.add_argument("--pretrain-learning-rate", type=float, default=1e-3)
    parser.add_argument("--pretrain-lr-scheduler", choices=("step", "cosine", "cosine_hard_restarts", "linear", "none",), default="cosine_hard_restarts")
    parser.add_argument("--mae-mask-ratio", type=float, default=0.3)
    parser.add_argument("--pretrain-patience", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--tqdm", type=int, default=1, choices=(0, 1))  # 0=关闭；1=按 epoch 耗时自动
    parser.add_argument("--visualization-split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--visualization-seed", type=int, default=20260913)
    parser.add_argument("--visualization-device", type=str, default="cuda:0")
    parser.add_argument("--run-dir", type=str, default="")
    parser.add_argument("--result-name", type=str, default="")


def finalize_arguments(parser: argparse.ArgumentParser):
    # 解析后同步训练链共享的窗口设置。
    args = parser.parse_args()
    apply_point_interval_seconds(args.point_interval_seconds)
    apply_history_steps(args.history_steps)
    apply_window_stride_steps(args.window_stride_steps)

    # 未指定目录时按完整实验设置生成运行目录。
    if args.run_dir == "":
        args.run_dir = str(PROJECT_ROOT / ".runs" / "WeatherComparison" / args.scenario / args.model / args.loss / f"point_{args.point_interval_seconds}s" / f"history_{args.history_steps}steps" / f"horizon_{args.horizon_steps}steps" / f"stride_{args.window_stride_steps}steps" / f"seed{args.seed}")
        if args.pretrain:
            args.run_dir += (
                f"/pretrain_e{args.pretrain_epochs}_m{args.mae_mask_ratio:g}"
                f"_lr{args.pretrain_learning_rate:g}"
                f"_s{args.pretrain_lr_scheduler}_p{args.pretrain_patience}"
            )
    return args
