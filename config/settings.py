from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SEED = 2026
HISTORY_STEPS = 16
# 每个时间点为15分钟，对应15/30/45分钟和1/2/3/4/8/12/24小时。
HORIZON_STEPS = (1, 2, 3, 4, 8, 12, 16, 32, 48, 96)
# 业务 Acc 目标：超短期 75%，短期 60%；中期指标待定
ACC_TARGET_PERCENT = {
    1: 75.0,
    96: 60.0,
}

# 默认采样间隔；正式运行由 shell 传入 --sample-seconds 覆盖。
SAMPLE_SECONDS = 15 * 60
DATASET_ROOT = PROJECT_ROOT / "dataset" / "processed"
WINDOW_STRIDE_STEPS = 1
WINDOW_STRIDE_NS = WINDOW_STRIDE_STEPS * SAMPLE_SECONDS * 1_000_000_000
EXPECTED_DELTA_NS = SAMPLE_SECONDS * 1_000_000_000
# 15分钟粒度下不再做时间池化。
TEMPORAL_POOL_STEPS = 1
HORIZONS = {
    "ultra_15min": 1,
    "ultra_30min": 2,
    "ultra_45min": 3,
    "short_1h": 4,
    "short_2h": 8,
    "short_3h": 12,
    "short_4h": 16,
    "short_8h": 32,
    "short_12h": 48,
    "short_1d": 96,
}
PATCH_LENGTHS = [4, 4, 2, 2]

# 按时间顺序将样本划分为训练、验证、测试集
SPLIT_RATIOS = (7, 1, 2)

# 全历史特征；天气相关见 WEATHER_COLUMNS
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

# 天气→P / Oracle：原始气象 + 舱温塔底温 + 对风姿态与扭缆
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
THEORY_INDEX = HISTORY_COLUMNS.index("风机-理论功率-计算")
# 全部历史特征按每台风机的训练时段统计量进行 StandardScaler

STRICT_ACC_MIN_POWER_KW = 100.0
STRICT_ACC_TOLERANCE = 0.30


def apply_sample_seconds(seconds: int) -> None:
    # 切换采样间隔，并同步时间差与滑窗时间单位
    global SAMPLE_SECONDS
    global WINDOW_STRIDE_NS
    global EXPECTED_DELTA_NS

    SAMPLE_SECONDS = seconds
    WINDOW_STRIDE_NS = WINDOW_STRIDE_STEPS * SAMPLE_SECONDS * 1_000_000_000
    EXPECTED_DELTA_NS = SAMPLE_SECONDS * 1_000_000_000


def apply_history_steps(steps: int) -> None:
    # 切换历史窗口点数
    global HISTORY_STEPS

    HISTORY_STEPS = steps


def apply_window_stride_steps(steps: int) -> None:
    # 切换预测起点间隔点数
    global WINDOW_STRIDE_STEPS
    global WINDOW_STRIDE_NS

    WINDOW_STRIDE_STEPS = steps
    WINDOW_STRIDE_NS = steps * SAMPLE_SECONDS * 1_000_000_000
