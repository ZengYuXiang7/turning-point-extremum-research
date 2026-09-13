#!/bin/bash
set -e

# 全部可训练历史特征经过周期编码后为59维；5分钟一个点。
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
unset LD_LIBRARY_PATH

SAMPLE_SECONDS=300
DATASET_NAME=GuangningWindPower5min
RUN_ROOT=".runs/WeatherComparison/NoFutureWeather/DLinearAllFeatures/MSE/$DATASET_NAME"

# 重建5分钟数据；不同粒度脚本共用 dataset/processed，需依次运行。
uv run python generate_data.py --sample-seconds "$SAMPLE_SECONDS"

# 60分钟历史预测5/10/15/60分钟。
uv run python run.py --model DLinearAllFeatures --scenario NoFutureWeather --loss MSE --sample-seconds "$SAMPLE_SECONDS" --window-stride-steps 1 --history-steps 12 --horizon-steps 1 --seed 2026 --epochs 100 --patience 10 --batch-size 1024 --learning-rate 0.001 --num-workers 4 --show-progress 0 --run-dir "$RUN_ROOT/h12_p1_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name dlinear_all_features_h12_p1_s1_seed2026
uv run python run.py --model DLinearAllFeatures --scenario NoFutureWeather --loss MSE --sample-seconds "$SAMPLE_SECONDS" --window-stride-steps 1 --history-steps 12 --horizon-steps 2 --seed 2026 --epochs 100 --patience 10 --batch-size 1024 --learning-rate 0.001 --num-workers 4 --show-progress 0 --run-dir "$RUN_ROOT/h12_p2_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name dlinear_all_features_h12_p2_s1_seed2026
uv run python run.py --model DLinearAllFeatures --scenario NoFutureWeather --loss MSE --sample-seconds "$SAMPLE_SECONDS" --window-stride-steps 1 --history-steps 12 --horizon-steps 3 --seed 2026 --epochs 100 --patience 10 --batch-size 1024 --learning-rate 0.001 --num-workers 4 --show-progress 0 --run-dir "$RUN_ROOT/h12_p3_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name dlinear_all_features_h12_p3_s1_seed2026
uv run python run.py --model DLinearAllFeatures --scenario NoFutureWeather --loss MSE --sample-seconds "$SAMPLE_SECONDS" --window-stride-steps 1 --history-steps 12 --horizon-steps 12 --seed 2026 --epochs 100 --patience 10 --batch-size 1024 --learning-rate 0.001 --num-workers 4 --show-progress 0 --run-dir "$RUN_ROOT/h12_p12_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name dlinear_all_features_h12_p12_s1_seed2026

# 等长历史预测2/3/4/8/12/24小时。
uv run python run.py --model DLinearAllFeatures --scenario NoFutureWeather --loss MSE --sample-seconds "$SAMPLE_SECONDS" --window-stride-steps 1 --history-steps 24 --horizon-steps 24 --seed 2026 --epochs 100 --patience 10 --batch-size 1024 --learning-rate 0.001 --num-workers 4 --show-progress 0 --run-dir "$RUN_ROOT/h24_p24_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name dlinear_all_features_h24_p24_s1_seed2026
uv run python run.py --model DLinearAllFeatures --scenario NoFutureWeather --loss MSE --sample-seconds "$SAMPLE_SECONDS" --window-stride-steps 1 --history-steps 36 --horizon-steps 36 --seed 2026 --epochs 100 --patience 10 --batch-size 1024 --learning-rate 0.001 --num-workers 4 --show-progress 0 --run-dir "$RUN_ROOT/h36_p36_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name dlinear_all_features_h36_p36_s1_seed2026
uv run python run.py --model DLinearAllFeatures --scenario NoFutureWeather --loss MSE --sample-seconds "$SAMPLE_SECONDS" --window-stride-steps 1 --history-steps 48 --horizon-steps 48 --seed 2026 --epochs 100 --patience 10 --batch-size 1024 --learning-rate 0.001 --num-workers 4 --show-progress 0 --run-dir "$RUN_ROOT/h48_p48_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name dlinear_all_features_h48_p48_s1_seed2026
uv run python run.py --model DLinearAllFeatures --scenario NoFutureWeather --loss MSE --sample-seconds "$SAMPLE_SECONDS" --window-stride-steps 1 --history-steps 96 --horizon-steps 96 --seed 2026 --epochs 100 --patience 10 --batch-size 1024 --learning-rate 0.001 --num-workers 4 --show-progress 0 --run-dir "$RUN_ROOT/h96_p96_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name dlinear_all_features_h96_p96_s1_seed2026
uv run python run.py --model DLinearAllFeatures --scenario NoFutureWeather --loss MSE --sample-seconds "$SAMPLE_SECONDS" --window-stride-steps 1 --history-steps 144 --horizon-steps 144 --seed 2026 --epochs 100 --patience 10 --batch-size 1024 --learning-rate 0.001 --num-workers 4 --show-progress 0 --run-dir "$RUN_ROOT/h144_p144_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name dlinear_all_features_h144_p144_s1_seed2026
uv run python run.py --model DLinearAllFeatures --scenario NoFutureWeather --loss MSE --sample-seconds "$SAMPLE_SECONDS" --window-stride-steps 1 --history-steps 288 --horizon-steps 288 --seed 2026 --epochs 100 --patience 10 --batch-size 1024 --learning-rate 0.001 --num-workers 4 --show-progress 0 --run-dir "$RUN_ROOT/h288_p288_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name dlinear_all_features_h288_p288_s1_seed2026
