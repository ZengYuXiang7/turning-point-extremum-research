#!/bin/bash
set -e

# 全部可训练历史特征经过周期编码后为59维，跨特征线性投影输出未来功率。
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
unset LD_LIBRARY_PATH

SAMPLE_SECONDS=900
DATASET_NAME=GuangningWindPower15min

# 60分钟历史预测15/30/45/60分钟。
uv run python run.py --model DLinearAllFeatures --scenario NoFutureWeather --loss MSE --sample-seconds "$SAMPLE_SECONDS" --window-stride-steps 1 --history-steps 4 --horizon-steps 1 --seed 2026 --epochs 100 --patience 10 --batch-size 1024 --learning-rate 0.001 --num-workers 4 --show-progress 0 --dataset-name "$DATASET_NAME" --result-name dlinear_all_features_h4_p1_s1_seed2026
uv run python run.py --model DLinearAllFeatures --scenario NoFutureWeather --loss MSE --sample-seconds "$SAMPLE_SECONDS" --window-stride-steps 1 --history-steps 4 --horizon-steps 2 --seed 2026 --epochs 100 --patience 10 --batch-size 1024 --learning-rate 0.001 --num-workers 4 --show-progress 0 --dataset-name "$DATASET_NAME" --result-name dlinear_all_features_h4_p2_s1_seed2026
uv run python run.py --model DLinearAllFeatures --scenario NoFutureWeather --loss MSE --sample-seconds "$SAMPLE_SECONDS" --window-stride-steps 1 --history-steps 4 --horizon-steps 3 --seed 2026 --epochs 100 --patience 10 --batch-size 1024 --learning-rate 0.001 --num-workers 4 --show-progress 0 --dataset-name "$DATASET_NAME" --result-name dlinear_all_features_h4_p3_s1_seed2026
uv run python run.py --model DLinearAllFeatures --scenario NoFutureWeather --loss MSE --sample-seconds "$SAMPLE_SECONDS" --window-stride-steps 1 --history-steps 4 --horizon-steps 4 --seed 2026 --epochs 100 --patience 10 --batch-size 1024 --learning-rate 0.001 --num-workers 4 --show-progress 0 --dataset-name "$DATASET_NAME" --result-name dlinear_all_features_h4_p4_s1_seed2026

# 等长历史预测2/3/4/8/12/24小时。
uv run python run.py --model DLinearAllFeatures --scenario NoFutureWeather --loss MSE --sample-seconds "$SAMPLE_SECONDS" --window-stride-steps 1 --history-steps 8 --horizon-steps 8 --seed 2026 --epochs 100 --patience 10 --batch-size 1024 --learning-rate 0.001 --num-workers 4 --show-progress 0 --dataset-name "$DATASET_NAME" --result-name dlinear_all_features_h8_p8_s1_seed2026
uv run python run.py --model DLinearAllFeatures --scenario NoFutureWeather --loss MSE --sample-seconds "$SAMPLE_SECONDS" --window-stride-steps 1 --history-steps 12 --horizon-steps 12 --seed 2026 --epochs 100 --patience 10 --batch-size 1024 --learning-rate 0.001 --num-workers 4 --show-progress 0 --dataset-name "$DATASET_NAME" --result-name dlinear_all_features_h12_p12_s1_seed2026
uv run python run.py --model DLinearAllFeatures --scenario NoFutureWeather --loss MSE --sample-seconds "$SAMPLE_SECONDS" --window-stride-steps 1 --history-steps 16 --horizon-steps 16 --seed 2026 --epochs 100 --patience 10 --batch-size 1024 --learning-rate 0.001 --num-workers 4 --show-progress 0 --dataset-name "$DATASET_NAME" --result-name dlinear_all_features_h16_p16_s1_seed2026
uv run python run.py --model DLinearAllFeatures --scenario NoFutureWeather --loss MSE --sample-seconds "$SAMPLE_SECONDS" --window-stride-steps 1 --history-steps 32 --horizon-steps 32 --seed 2026 --epochs 100 --patience 10 --batch-size 1024 --learning-rate 0.001 --num-workers 4 --show-progress 0 --dataset-name "$DATASET_NAME" --result-name dlinear_all_features_h32_p32_s1_seed2026
uv run python run.py --model DLinearAllFeatures --scenario NoFutureWeather --loss MSE --sample-seconds "$SAMPLE_SECONDS" --window-stride-steps 1 --history-steps 48 --horizon-steps 48 --seed 2026 --epochs 100 --patience 10 --batch-size 1024 --learning-rate 0.001 --num-workers 4 --show-progress 0 --dataset-name "$DATASET_NAME" --result-name dlinear_all_features_h48_p48_s1_seed2026
uv run python run.py --model DLinearAllFeatures --scenario NoFutureWeather --loss MSE --sample-seconds "$SAMPLE_SECONDS" --window-stride-steps 1 --history-steps 96 --horizon-steps 96 --seed 2026 --epochs 100 --patience 10 --batch-size 1024 --learning-rate 0.001 --num-workers 4 --show-progress 0 --dataset-name "$DATASET_NAME" --result-name dlinear_all_features_h96_p96_s1_seed2026
