#!/bin/bash
set -e
# StockEcho Acc30 10s；--horizon 传分钟，run.py 内换成步数
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${PYTHON:-python}"
cd "$ROOT"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
# 系统 CUDA 12.1 在 LD_LIBRARY_PATH 中会抢在 pip cu124 库前面，导致 torch import 失败
unset LD_LIBRARY_PATH

# 业务长度（分钟）：15分钟 / 12小时 / 1天 / 2天
HORIZONS="15 720 1440 2880"

# --- StockEcho / NoFutureWeather / MSEAcc30 ---
for horizon in $HORIZONS; do
  "$PYTHON" run.py --model StockEcho --scenario NoFutureWeather --loss MSEAcc30 --horizon "$horizon" --seed 2026 --epochs 100 --patience 10 --batch-size 2
done

# --- StockEcho / OracleFutureWeather / MSEAcc30 ---
for horizon in $HORIZONS; do
  "$PYTHON" run.py --model StockEcho --scenario OracleFutureWeather --loss MSEAcc30 --horizon "$horizon" --seed 2026 --epochs 100 --patience 10 --batch-size 2
done

# --- 汇总 ---
"$PYTHON" scripts/build_summary.py
echo "stockecho acc30 10s campaign done"
