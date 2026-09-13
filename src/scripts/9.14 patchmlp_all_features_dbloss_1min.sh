#!/bin/bash
set -e

# 全部59维历史特征经PatchMLP交互，只对未来功率计算MSE与加权DBLoss；1分钟一个点。
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
unset LD_LIBRARY_PATH

SAMPLE_SECONDS=60
DBLOSS_WEIGHT=0.5
DATASET_NAME=GuangningWindPower1min
RUN_ROOT=".runs/WeatherComparison/NoFutureWeather/PatchMLPAllFeatures/DBLoss/$DATASET_NAME"

# 重建1分钟数据；不同粒度脚本共用 dataset/processed，需依次运行。
uv run python generate_data.py --sample-seconds "$SAMPLE_SECONDS"

# 60分钟历史预测1/2/3/60分钟。
uv run python run.py --model PatchMLPAllFeatures --scenario NoFutureWeather --loss DBLoss --dbloss-weight "$DBLOSS_WEIGHT" --sample-seconds "$SAMPLE_SECONDS" --window-stride-steps 1 --history-steps 60 --horizon-steps 1 --seed 2026 --epochs 100 --patience 10 --batch-size 4 --learning-rate 0.001 --num-workers 4 --show-progress 0 --run-dir "$RUN_ROOT/h60_p1_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name patchmlp_all_features_dbloss_w0p5_h60_p1_s1_seed2026
uv run python run.py --model PatchMLPAllFeatures --scenario NoFutureWeather --loss DBLoss --dbloss-weight "$DBLOSS_WEIGHT" --sample-seconds "$SAMPLE_SECONDS" --window-stride-steps 1 --history-steps 60 --horizon-steps 2 --seed 2026 --epochs 100 --patience 10 --batch-size 4 --learning-rate 0.001 --num-workers 4 --show-progress 0 --run-dir "$RUN_ROOT/h60_p2_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name patchmlp_all_features_dbloss_w0p5_h60_p2_s1_seed2026
uv run python run.py --model PatchMLPAllFeatures --scenario NoFutureWeather --loss DBLoss --dbloss-weight "$DBLOSS_WEIGHT" --sample-seconds "$SAMPLE_SECONDS" --window-stride-steps 1 --history-steps 60 --horizon-steps 3 --seed 2026 --epochs 100 --patience 10 --batch-size 4 --learning-rate 0.001 --num-workers 4 --show-progress 0 --run-dir "$RUN_ROOT/h60_p3_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name patchmlp_all_features_dbloss_w0p5_h60_p3_s1_seed2026
uv run python run.py --model PatchMLPAllFeatures --scenario NoFutureWeather --loss DBLoss --dbloss-weight "$DBLOSS_WEIGHT" --sample-seconds "$SAMPLE_SECONDS" --window-stride-steps 1 --history-steps 60 --horizon-steps 60 --seed 2026 --epochs 100 --patience 10 --batch-size 4 --learning-rate 0.001 --num-workers 4 --show-progress 0 --run-dir "$RUN_ROOT/h60_p60_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name patchmlp_all_features_dbloss_w0p5_h60_p60_s1_seed2026

# 等长历史预测2/3/4/8/12/24小时。
uv run python run.py --model PatchMLPAllFeatures --scenario NoFutureWeather --loss DBLoss --dbloss-weight "$DBLOSS_WEIGHT" --sample-seconds "$SAMPLE_SECONDS" --window-stride-steps 1 --history-steps 120 --horizon-steps 120 --seed 2026 --epochs 100 --patience 10 --batch-size 4 --learning-rate 0.001 --num-workers 4 --show-progress 0 --run-dir "$RUN_ROOT/h120_p120_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name patchmlp_all_features_dbloss_w0p5_h120_p120_s1_seed2026
uv run python run.py --model PatchMLPAllFeatures --scenario NoFutureWeather --loss DBLoss --dbloss-weight "$DBLOSS_WEIGHT" --sample-seconds "$SAMPLE_SECONDS" --window-stride-steps 1 --history-steps 180 --horizon-steps 180 --seed 2026 --epochs 100 --patience 10 --batch-size 4 --learning-rate 0.001 --num-workers 4 --show-progress 0 --run-dir "$RUN_ROOT/h180_p180_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name patchmlp_all_features_dbloss_w0p5_h180_p180_s1_seed2026
uv run python run.py --model PatchMLPAllFeatures --scenario NoFutureWeather --loss DBLoss --dbloss-weight "$DBLOSS_WEIGHT" --sample-seconds "$SAMPLE_SECONDS" --window-stride-steps 1 --history-steps 240 --horizon-steps 240 --seed 2026 --epochs 100 --patience 10 --batch-size 4 --learning-rate 0.001 --num-workers 4 --show-progress 0 --run-dir "$RUN_ROOT/h240_p240_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name patchmlp_all_features_dbloss_w0p5_h240_p240_s1_seed2026
uv run python run.py --model PatchMLPAllFeatures --scenario NoFutureWeather --loss DBLoss --dbloss-weight "$DBLOSS_WEIGHT" --sample-seconds "$SAMPLE_SECONDS" --window-stride-steps 1 --history-steps 480 --horizon-steps 480 --seed 2026 --epochs 100 --patience 10 --batch-size 4 --learning-rate 0.001 --num-workers 4 --show-progress 0 --run-dir "$RUN_ROOT/h480_p480_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name patchmlp_all_features_dbloss_w0p5_h480_p480_s1_seed2026
uv run python run.py --model PatchMLPAllFeatures --scenario NoFutureWeather --loss DBLoss --dbloss-weight "$DBLOSS_WEIGHT" --sample-seconds "$SAMPLE_SECONDS" --window-stride-steps 1 --history-steps 720 --horizon-steps 720 --seed 2026 --epochs 100 --patience 10 --batch-size 4 --learning-rate 0.001 --num-workers 4 --show-progress 0 --run-dir "$RUN_ROOT/h720_p720_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name patchmlp_all_features_dbloss_w0p5_h720_p720_s1_seed2026
uv run python run.py --model PatchMLPAllFeatures --scenario NoFutureWeather --loss DBLoss --dbloss-weight "$DBLOSS_WEIGHT" --sample-seconds "$SAMPLE_SECONDS" --window-stride-steps 1 --history-steps 1440 --horizon-steps 1440 --seed 2026 --epochs 100 --patience 10 --batch-size 4 --learning-rate 0.001 --num-workers 4 --show-progress 0 --run-dir "$RUN_ROOT/h1440_p1440_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name patchmlp_all_features_dbloss_w0p5_h1440_p1440_s1_seed2026
