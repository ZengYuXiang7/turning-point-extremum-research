#!/bin/bash
set -e

# 训练全部56个无缺失原始特征的DLinear；10分钟一个点。
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
unset LD_LIBRARY_PATH

POINT_INTERVAL_SECONDS=600
DATASET_NAME=GuangningWindPower10min
RUN_ROOT=".runs/WeatherComparison/NoFutureWeather/DLinearAllFeatures/MSE/$DATASET_NAME"

# 相邻模型点从共享10秒底层序列中每隔60个基础点选取。

# 等长训练预测1个模型点、30分钟与1/2/4/8/12/24小时。
uv run python run_single.py --mode train --model DLinearAllFeatures --scenario NoFutureWeather --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 1 --horizon-steps 1 --seed 2026 --epochs 10 --patience 3 --batch-size 1024 --learning-rate 0.001 --num-workers 4 --tqdm 0 --run-dir "$RUN_ROOT/h1_p1_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name dlinear_all_features_h1_p1_s1_seed2026
uv run python run_single.py --mode train --model DLinearAllFeatures --scenario NoFutureWeather --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 3 --horizon-steps 3 --seed 2026 --epochs 10 --patience 3 --batch-size 1024 --learning-rate 0.001 --num-workers 4 --tqdm 0 --run-dir "$RUN_ROOT/h3_p3_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name dlinear_all_features_h3_p3_s1_seed2026
uv run python run_single.py --mode train --model DLinearAllFeatures --scenario NoFutureWeather --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 6 --horizon-steps 6 --seed 2026 --epochs 10 --patience 3 --batch-size 1024 --learning-rate 0.001 --num-workers 4 --tqdm 0 --run-dir "$RUN_ROOT/h6_p6_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name dlinear_all_features_h6_p6_s1_seed2026

# 等长训练预测2/4/8/12/24小时。
uv run python run_single.py --mode train --model DLinearAllFeatures --scenario NoFutureWeather --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 12 --horizon-steps 12 --seed 2026 --epochs 10 --patience 3 --batch-size 1024 --learning-rate 0.001 --num-workers 4 --tqdm 0 --run-dir "$RUN_ROOT/h12_p12_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name dlinear_all_features_h12_p12_s1_seed2026
uv run python run_single.py --mode train --model DLinearAllFeatures --scenario NoFutureWeather --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 24 --horizon-steps 24 --seed 2026 --epochs 10 --patience 3 --batch-size 1024 --learning-rate 0.001 --num-workers 4 --tqdm 0 --run-dir "$RUN_ROOT/h24_p24_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name dlinear_all_features_h24_p24_s1_seed2026
uv run python run_single.py --mode train --model DLinearAllFeatures --scenario NoFutureWeather --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 48 --horizon-steps 48 --seed 2026 --epochs 10 --patience 3 --batch-size 1024 --learning-rate 0.001 --num-workers 4 --tqdm 0 --run-dir "$RUN_ROOT/h48_p48_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name dlinear_all_features_h48_p48_s1_seed2026
uv run python run_single.py --mode train --model DLinearAllFeatures --scenario NoFutureWeather --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 72 --horizon-steps 72 --seed 2026 --epochs 10 --patience 3 --batch-size 1024 --learning-rate 0.001 --num-workers 4 --tqdm 0 --run-dir "$RUN_ROOT/h72_p72_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name dlinear_all_features_h72_p72_s1_seed2026
uv run python run_single.py --mode train --model DLinearAllFeatures --scenario NoFutureWeather --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 144 --horizon-steps 144 --seed 2026 --epochs 10 --patience 3 --batch-size 1024 --learning-rate 0.001 --num-workers 4 --tqdm 0 --run-dir "$RUN_ROOT/h144_p144_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name dlinear_all_features_h144_p144_s1_seed2026
