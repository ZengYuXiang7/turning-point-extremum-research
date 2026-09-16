#!/bin/bash
set -e

# 测试全部56个无缺失原始特征的DLinear；1分钟一个点。
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
unset LD_LIBRARY_PATH

POINT_INTERVAL_SECONDS=60
DATASET_NAME=GuangningWindPower1min
RUN_ROOT=".runs/WeatherComparison/NoFutureWeather/DLinearAllFeatures/MSE/$DATASET_NAME"

# 等长训练预测15/30分钟与1/2/4/8/12/24小时。
uv run python run_single.py --mode test --model DLinearAllFeatures --scenario NoFutureWeather --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 15 --horizon-steps 15 --seed 2026 --batch-size 1024 --num-workers 4 --tqdm 0 --run-dir "$RUN_ROOT/h15_p15_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name dlinear_all_features_h15_p15_s1_seed2026
uv run python run_single.py --mode test --model DLinearAllFeatures --scenario NoFutureWeather --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 30 --horizon-steps 30 --seed 2026 --batch-size 1024 --num-workers 4 --tqdm 0 --run-dir "$RUN_ROOT/h30_p30_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name dlinear_all_features_h30_p30_s1_seed2026
uv run python run_single.py --mode test --model DLinearAllFeatures --scenario NoFutureWeather --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 60 --horizon-steps 60 --seed 2026 --batch-size 1024 --num-workers 4 --tqdm 0 --run-dir "$RUN_ROOT/h60_p60_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name dlinear_all_features_h60_p60_s1_seed2026

# 等长训练预测2/4/8/12/24小时。
uv run python run_single.py --mode test --model DLinearAllFeatures --scenario NoFutureWeather --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 120 --horizon-steps 120 --seed 2026 --batch-size 1024 --num-workers 4 --tqdm 0 --run-dir "$RUN_ROOT/h120_p120_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name dlinear_all_features_h120_p120_s1_seed2026
uv run python run_single.py --mode test --model DLinearAllFeatures --scenario NoFutureWeather --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 240 --horizon-steps 240 --seed 2026 --batch-size 1024 --num-workers 4 --tqdm 0 --run-dir "$RUN_ROOT/h240_p240_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name dlinear_all_features_h240_p240_s1_seed2026
uv run python run_single.py --mode test --model DLinearAllFeatures --scenario NoFutureWeather --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 480 --horizon-steps 480 --seed 2026 --batch-size 1024 --num-workers 4 --tqdm 0 --run-dir "$RUN_ROOT/h480_p480_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name dlinear_all_features_h480_p480_s1_seed2026
uv run python run_single.py --mode test --model DLinearAllFeatures --scenario NoFutureWeather --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 720 --horizon-steps 720 --seed 2026 --batch-size 1024 --num-workers 4 --tqdm 0 --run-dir "$RUN_ROOT/h720_p720_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name dlinear_all_features_h720_p720_s1_seed2026
uv run python run_single.py --mode test --model DLinearAllFeatures --scenario NoFutureWeather --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 1440 --horizon-steps 1440 --seed 2026 --batch-size 1024 --num-workers 4 --tqdm 0 --run-dir "$RUN_ROOT/h1440_p1440_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name dlinear_all_features_h1440_p1440_s1_seed2026
