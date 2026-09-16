#!/bin/bash
set -e

# 训练绝对相关系数不低于0.6的20个预测变量加历史功率
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
unset LD_LIBRARY_PATH

POINT_INTERVAL_SECONDS=900
DATASET_NAME=GuangningWindPower10sPoint15min
CORRELATED_RUN_ROOT=".runs/WeatherComparison/NoFutureWeather/DLinearCorrelatedFeatures/MSE/$DATASET_NAME/corr21_absge0p6"

# 等长训练预测15/30分钟与1/2/4/8/12/24小时。
uv run python run_single.py --mode train --model DLinearCorrelatedFeatures --scenario NoFutureWeather --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 1 --horizon-steps 1 --seed 2026 --epochs 10 --patience 3 --batch-size 1024 --learning-rate 0.001 --num-workers 4 --tqdm 0 --run-dir "$CORRELATED_RUN_ROOT/h1_p1_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name dlinear_corr21_absge0p6_h1_p1_s1_seed2026
uv run python run_single.py --mode train --model DLinearCorrelatedFeatures --scenario NoFutureWeather --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 2 --horizon-steps 2 --seed 2026 --epochs 10 --patience 3 --batch-size 1024 --learning-rate 0.001 --num-workers 4 --tqdm 0 --run-dir "$CORRELATED_RUN_ROOT/h2_p2_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name dlinear_corr21_absge0p6_h2_p2_s1_seed2026
uv run python run_single.py --mode train --model DLinearCorrelatedFeatures --scenario NoFutureWeather --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 4 --horizon-steps 4 --seed 2026 --epochs 10 --patience 3 --batch-size 1024 --learning-rate 0.001 --num-workers 4 --tqdm 0 --run-dir "$CORRELATED_RUN_ROOT/h4_p4_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name dlinear_corr21_absge0p6_h4_p4_s1_seed2026
uv run python run_single.py --mode train --model DLinearCorrelatedFeatures --scenario NoFutureWeather --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 8 --horizon-steps 8 --seed 2026 --epochs 10 --patience 3 --batch-size 1024 --learning-rate 0.001 --num-workers 4 --tqdm 0 --run-dir "$CORRELATED_RUN_ROOT/h8_p8_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name dlinear_corr21_absge0p6_h8_p8_s1_seed2026
uv run python run_single.py --mode train --model DLinearCorrelatedFeatures --scenario NoFutureWeather --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 16 --horizon-steps 16 --seed 2026 --epochs 10 --patience 3 --batch-size 1024 --learning-rate 0.001 --num-workers 4 --tqdm 0 --run-dir "$CORRELATED_RUN_ROOT/h16_p16_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name dlinear_corr21_absge0p6_h16_p16_s1_seed2026
uv run python run_single.py --mode train --model DLinearCorrelatedFeatures --scenario NoFutureWeather --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 32 --horizon-steps 32 --seed 2026 --epochs 10 --patience 3 --batch-size 1024 --learning-rate 0.001 --num-workers 4 --tqdm 0 --run-dir "$CORRELATED_RUN_ROOT/h32_p32_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name dlinear_corr21_absge0p6_h32_p32_s1_seed2026
uv run python run_single.py --mode train --model DLinearCorrelatedFeatures --scenario NoFutureWeather --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 48 --horizon-steps 48 --seed 2026 --epochs 10 --patience 3 --batch-size 1024 --learning-rate 0.001 --num-workers 4 --tqdm 0 --run-dir "$CORRELATED_RUN_ROOT/h48_p48_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name dlinear_corr21_absge0p6_h48_p48_s1_seed2026
uv run python run_single.py --mode train --model DLinearCorrelatedFeatures --scenario NoFutureWeather --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 96 --horizon-steps 96 --seed 2026 --epochs 10 --patience 3 --batch-size 1024 --learning-rate 0.001 --num-workers 4 --tqdm 0 --run-dir "$CORRELATED_RUN_ROOT/h96_p96_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name dlinear_corr21_absge0p6_h96_p96_s1_seed2026
