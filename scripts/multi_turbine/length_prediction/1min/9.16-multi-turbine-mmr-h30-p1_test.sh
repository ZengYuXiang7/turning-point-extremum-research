#!/bin/bash
set -e

# 评估连续30个1分钟真实点预测下一分钟的正式最优检查点。
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
unset LD_LIBRARY_PATH

POINT_INTERVAL_SECONDS=60
DATASET_NAME=GuangningWindPower16Turbine1min
RUN_ROOT=".runs/WeatherComparison/MultiTurbine/MultiTurbine/MSE/$DATASET_NAME/pretrain-e5-m0p3-spatial"
RUN_DIR="$RUN_ROOT/h30_p1_s1_seed2026"
RESULT_NAME=multi-turbine-mmr-e5-m0p3-spatial-h30-p1-s1-seed2026

uv run python run_multi.py --mode test --model MultiTurbine --scenario MultiTurbine --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 30 --horizon-steps 1 --seed 2026 --batch-size 32 --num-workers 4 --tqdm 0 --revin 1 --satra-use-spatial-relation 1 --satra-use-dtw-prior 0 --run-dir "$RUN_DIR" --dataset-name "$DATASET_NAME" --result-name "$RESULT_NAME"
