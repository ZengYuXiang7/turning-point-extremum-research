#!/bin/bash
set -e
# 同时刻：10s 天气相关特征 → 功率（非历史预测未来）
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PYTHON="${PYTHON:-python}"
cd "$ROOT"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
# 系统 CUDA 12.1 在 LD_LIBRARY_PATH 中会抢在 pip cu124 库前面，导致 torch import 失败
unset LD_LIBRARY_PATH

# ---------------------------------------------------------------------------
# 输入 = WEATHER_COLUMNS（13 维，与目标同一时刻）：
#   风机-实时风速 / 风机-风速
#   风机-实时风向_sin/cos
#   风机-环境温度 / 机舱-舱内温度 / 塔筒-塔底温度
#   偏航-对风角度_sin/cos / 机舱位置_sin/cos / 扭揽角度_sin/cos
# 目标 = 风机-P
# 数据：16 机 train/val/test 点级样本，按时间先后，不做 shuffle
# 训练损失 MSE；早停/选模/LR 调度盯 val Acc30（trainer 内 WeatherToPower 分支）
# ---------------------------------------------------------------------------

# --- WeatherToPower / MLP / MSE ---
"$PYTHON" run.py --model MLP --scenario WeatherToPower --loss MSE --seed 2026 --epochs 200 --patience 50 --batch-size 4096
