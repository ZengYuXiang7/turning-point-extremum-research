#!/bin/bash
set -e

# 评估Model2对应正式record指向的验证最优检查点。
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
unset LD_LIBRARY_PATH

POINT_INTERVAL_SECONDS=60
DATASET_NAME=GuangningWindPower16Turbine1min
RUN_ROOT=".runs/WeatherComparison/MultiTurbine/Model2/MSE/$DATASET_NAME/pretrain-e5-m0p3-spatial-patch-l6-s3"
RUN_DIR="$RUN_ROOT/h30_p1_s1_seed2026"
RESULT_NAME=model2-patch-l6-s3-mmr-e5-m0p3-spatial-h30-p1-s1-seed2026

uv run python run_multi.py --mode test --model Model2 --scenario MultiTurbine --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 30 --horizon-steps 1 --seed 2026 --batch-size 32 --num-workers 4 --tqdm 0 --revin 1 --satra-use-spatial-relation 1 --satra-use-dtw-prior 0 --model2-patch-length 6 --model2-patch-stride 3 --run-dir "$RUN_DIR" --dataset-name "$DATASET_NAME" --result-name "$RESULT_NAME"
