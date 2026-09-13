#!/bin/bash
set -e

# 10s Oracle：历史全特征 + 未来13维真实协变量，预测未来功率。
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PYTHON="${PYTHON:-python}"

cd "$ROOT"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
# 系统 CUDA 12.1 会抢在 pip cu124 库前面，导致 torch 导入失败。
unset LD_LIBRARY_PATH

# 历史60分钟，物理视界：15min=90步，1天=8640步，4天=34560步。
# 长视界的未来序列长，batch 相应下调。
# 注：当前 val/test 各3天（6/8–6/11、6/11–6/14），4天视界无窗口，5760 暂注释。

# --- DLinear / OracleFutureWeather / MSE ---
"$PYTHON" run.py --model DLinear --scenario OracleFutureWeather --loss MSE --horizon 10 --history 60 --seed 2026 --epochs 100 --patience 10 --batch-size 256 --show-progress 0
"$PYTHON" run.py --model DLinear --scenario OracleFutureWeather --loss MSE --horizon 15 --history 60 --seed 2026 --epochs 100 --patience 10 --batch-size 256 --show-progress 0
"$PYTHON" run.py --model DLinear --scenario OracleFutureWeather --loss MSE --horizon 30 --history 60 --seed 2026 --epochs 100 --patience 10 --batch-size 256 --show-progress 0
"$PYTHON" run.py --model DLinear --scenario OracleFutureWeather --loss MSE --horizon 60 --history 60 --seed 2026 --epochs 100 --patience 10 --batch-size 128 --show-progress 0
"$PYTHON" run.py --model DLinear --scenario OracleFutureWeather --loss MSE --horizon 120 --history 60 --seed 2026 --epochs 100 --patience 10 --batch-size 128 --show-progress 0
"$PYTHON" run.py --model DLinear --scenario OracleFutureWeather --loss MSE --horizon 240 --history 60 --seed 2026 --epochs 100 --patience 10 --batch-size 128 --show-progress 0
"$PYTHON" run.py --model DLinear --scenario OracleFutureWeather --loss MSE --horizon 1440 --history 60 --seed 2026 --epochs 100 --patience 10 --batch-size 64 --show-progress 0
# "$PYTHON" run.py --model DLinear --scenario OracleFutureWeather --loss MSE --horizon 5760 --history 60 --seed 2026 --epochs 100 --patience 10 --batch-size 32
