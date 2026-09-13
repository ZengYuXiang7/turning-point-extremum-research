from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SEED = 2026
HISTORY_MINUTES = 240
# 统一预测视界：15分钟 / 12小时 / 1天 / 2天
HORIZON_MINUTES = (15, 720, 1440, 2880)
# 业务 Acc 目标：超短期 75%，短期 60%；中期指标待定
ACC_TARGET_PERCENT = {
    15: 75.0,
    1440: 60.0,
}

# 仅使用原生10秒数据
GRAIN = "10s"
SAMPLE_SECONDS = 10
DATASET_ROOT = PROJECT_ROOT / "dataset" / "GuangningWindPower10s"
STEPS_PER_MINUTE = 60 // SAMPLE_SECONDS
HISTORY_STEPS = HISTORY_MINUTES * STEPS_PER_MINUTE
# 每10秒移动一个采样点
WINDOW_STRIDE_STEPS = 1
EXPECTED_DELTA_NS = SAMPLE_SECONDS * 1_000_000_000
# 池化约4分钟一块，即24个10秒点
TEMPORAL_POOL_STEPS = 4 * STEPS_PER_MINUTE
HORIZONS = {
    "ultra_15min": 15 * STEPS_PER_MINUTE,
    "short_12h": 720 * STEPS_PER_MINUTE,
    "short_1d": 1440 * STEPS_PER_MINUTE,
    "mid_2d": 2880 * STEPS_PER_MINUTE,
}
PATCH_LENGTHS = [minutes * STEPS_PER_MINUTE for minutes in (30, 20, 12, 6)]

# 训练、验证、测试目标时间严格不重叠；验证和测试各3天
TRAIN_END = "2026-06-07T00:00:00"
VALID_START = "2026-06-08T00:00:00"
VALID_END = "2026-06-11T00:00:00"
TEST_START = "2026-06-11T00:00:00"
TEST_END = "2026-06-14T00:00:00"

# 全历史特征；天气相关见 WEATHER_COLUMNS
HISTORY_COLUMNS = [
    "风机-P",
    "风机-理论功率",
    "风机-实时风速",
    "风机-风速",
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
]

# 天气→P / Oracle：原始气象 + 舱温塔底温 + 对风姿态与扭缆
WEATHER_COLUMNS = [
    "风机-实时风速",
    "风机-风速",
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
THEORY_INDEX = HISTORY_COLUMNS.index("风机-理论功率")
CIRCULAR_INDICES = [4, 5, 14, 15, 16, 17, 18, 19]
# 连续特征用训练集 StandardScaler；sin/cos 角特征保持原样
CONTINUOUS_STANDARDIZE_INDICES = [0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13]

STRICT_ACC_MIN_POWER_KW = 100.0
STRICT_ACC_TOLERANCE = 0.30


def refresh_derived() -> None:
    # 由 SAMPLE_SECONDS 重算步数相关常量；窗步长固定为 1 个采样点
    global STEPS_PER_MINUTE
    global HISTORY_STEPS
    global WINDOW_STRIDE_STEPS
    global EXPECTED_DELTA_NS
    global TEMPORAL_POOL_STEPS
    global HORIZONS
    global PATCH_LENGTHS

    STEPS_PER_MINUTE = 60 // SAMPLE_SECONDS
    HISTORY_STEPS = HISTORY_MINUTES * STEPS_PER_MINUTE
    WINDOW_STRIDE_STEPS = 1
    EXPECTED_DELTA_NS = SAMPLE_SECONDS * 1_000_000_000
    TEMPORAL_POOL_STEPS = 4 * STEPS_PER_MINUTE
    HORIZONS = {
        "ultra_15min": 15 * STEPS_PER_MINUTE,
        "short_12h": 720 * STEPS_PER_MINUTE,
        "short_1d": 1440 * STEPS_PER_MINUTE,
        "mid_2d": 2880 * STEPS_PER_MINUTE,
    }
    PATCH_LENGTHS = [minutes * STEPS_PER_MINUTE for minutes in (30, 20, 12, 6)]


def apply_history(minutes: int) -> None:
    # 切换历史窗长度（分钟）；须在导入 data/model 之前调用
    global HISTORY_MINUTES

    HISTORY_MINUTES = minutes
    refresh_derived()


def apply_window_stride(steps: int) -> None:
    """Set the prediction-origin spacing after refresh_derived has run."""
    global WINDOW_STRIDE_STEPS

    WINDOW_STRIDE_STEPS = int(steps)
