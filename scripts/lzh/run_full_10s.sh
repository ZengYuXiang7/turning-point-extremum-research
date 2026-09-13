#!/bin/bash
set -e
# 原生 10s 细粒度；--horizon 传分钟，run.py 内换成步数
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${PYTHON:-python}"
cd "$ROOT"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
# 系统 CUDA 12.1 在 LD_LIBRARY_PATH 中会抢在 pip cu124 库前面，导致 torch import 失败
unset LD_LIBRARY_PATH

# 业务长度（分钟）：15分钟 / 12小时 / 1天 / 2天
HORIZONS="15 720 1440 2880"

# --- PatchMLP / NoFutureWeather / MSE ---
for horizon in $HORIZONS; do
  "$PYTHON" run.py --model PatchMLP --scenario NoFutureWeather --loss MSE --horizon "$horizon" --seed 2026 --epochs 100 --patience 10 --batch-size 4
done

# --- PatchMLP / NoFutureWeather / DBLoss ---
for horizon in $HORIZONS; do
  "$PYTHON" run.py --model PatchMLP --scenario NoFutureWeather --loss DBLoss --horizon "$horizon" --seed 2026 --epochs 100 --patience 10 --batch-size 4
done

# --- DLinear / NoFutureWeather / MSE ---
for horizon in $HORIZONS; do
  "$PYTHON" run.py --model DLinear --scenario NoFutureWeather --loss MSE --horizon "$horizon" --seed 2026 --epochs 100 --patience 10 --batch-size 256
done

# --- DLinear / NoFutureWeather / DBLoss ---
for horizon in $HORIZONS; do
  "$PYTHON" run.py --model DLinear --scenario NoFutureWeather --loss DBLoss --horizon "$horizon" --seed 2026 --epochs 100 --patience 10 --batch-size 256
done

# --- PatchMLP / OracleFutureWeather / MSE ---
for horizon in $HORIZONS; do
  "$PYTHON" run.py --model PatchMLP --scenario OracleFutureWeather --loss MSE --horizon "$horizon" --seed 2026 --epochs 100 --patience 10 --batch-size 4
done

# --- PatchMLP / OracleFutureWeather / DBLoss ---
for horizon in $HORIZONS; do
  "$PYTHON" run.py --model PatchMLP --scenario OracleFutureWeather --loss DBLoss --horizon "$horizon" --seed 2026 --epochs 100 --patience 10 --batch-size 4
done

# --- DLinear / OracleFutureWeather / MSE ---
for horizon in $HORIZONS; do
  "$PYTHON" run.py --model DLinear --scenario OracleFutureWeather --loss MSE --horizon "$horizon" --seed 2026 --epochs 100 --patience 10 --batch-size 256
done

# --- DLinear / OracleFutureWeather / DBLoss ---
for horizon in $HORIZONS; do
  "$PYTHON" run.py --model DLinear --scenario OracleFutureWeather --loss DBLoss --horizon "$horizon" --seed 2026 --epochs 100 --patience 10 --batch-size 256
done

# --- 汇总 ---
"$PYTHON" scripts/build_summary.py
"$PYTHON" scripts/visualize_comparison.py
echo "formal 10s campaign done"
