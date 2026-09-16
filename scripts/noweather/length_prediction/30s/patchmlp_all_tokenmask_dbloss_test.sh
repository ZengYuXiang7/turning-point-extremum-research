#!/bin/bash
set -e

# NoFutureWeather 下 PatchMLPAllFeatures 的随机时间 token 热身与 DBLoss 长度预测；30s一个模型点。
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
unset LD_LIBRARY_PATH

POINT_INTERVAL_SECONDS=30
DATASET_NAME=GuangningWindPower30s
RUN_ROOT=".runs/WeatherComparison/NoFutureWeather/PatchMLPAllFeatures/DBLoss/$DATASET_NAME/pretrain_e10_m0p3"

# 仅评估对应正式 record 指向的验证最优检查点。
uv run python run_single.py --mode test --model PatchMLPAllFeatures --scenario NoFutureWeather --loss DBLoss --dbloss-weight 0.5 --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 80 --horizon-steps 30 --seed 2026 --batch-size 32 --num-workers 4 --tqdm 0 --run-dir "$RUN_ROOT/h80_p30_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name patchmlp_all_tokenmask_e10_m0p3_dbloss_30s_h80_p30_s1_seed2026
uv run python run_single.py --mode test --model PatchMLPAllFeatures --scenario NoFutureWeather --loss DBLoss --dbloss-weight 0.5 --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 80 --horizon-steps 60 --seed 2026 --batch-size 32 --num-workers 4 --tqdm 0 --run-dir "$RUN_ROOT/h80_p60_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name patchmlp_all_tokenmask_e10_m0p3_dbloss_30s_h80_p60_s1_seed2026
uv run python run_single.py --mode test --model PatchMLPAllFeatures --scenario NoFutureWeather --loss DBLoss --dbloss-weight 0.5 --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 80 --horizon-steps 120 --seed 2026 --batch-size 32 --num-workers 4 --tqdm 0 --run-dir "$RUN_ROOT/h80_p120_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name patchmlp_all_tokenmask_e10_m0p3_dbloss_30s_h80_p120_s1_seed2026
uv run python run_single.py --mode test --model PatchMLPAllFeatures --scenario NoFutureWeather --loss DBLoss --dbloss-weight 0.5 --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 80 --horizon-steps 240 --seed 2026 --batch-size 32 --num-workers 4 --tqdm 0 --run-dir "$RUN_ROOT/h80_p240_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name patchmlp_all_tokenmask_e10_m0p3_dbloss_30s_h80_p240_s1_seed2026
uv run python run_single.py --mode test --model PatchMLPAllFeatures --scenario NoFutureWeather --loss DBLoss --dbloss-weight 0.5 --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 80 --horizon-steps 480 --seed 2026 --batch-size 32 --num-workers 4 --tqdm 0 --run-dir "$RUN_ROOT/h80_p480_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name patchmlp_all_tokenmask_e10_m0p3_dbloss_30s_h80_p480_s1_seed2026
uv run python run_single.py --mode test --model PatchMLPAllFeatures --scenario NoFutureWeather --loss DBLoss --dbloss-weight 0.5 --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 80 --horizon-steps 960 --seed 2026 --batch-size 32 --num-workers 4 --tqdm 0 --run-dir "$RUN_ROOT/h80_p960_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name patchmlp_all_tokenmask_e10_m0p3_dbloss_30s_h80_p960_s1_seed2026
uv run python run_single.py --mode test --model PatchMLPAllFeatures --scenario NoFutureWeather --loss DBLoss --dbloss-weight 0.5 --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 80 --horizon-steps 1440 --seed 2026 --batch-size 32 --num-workers 4 --tqdm 0 --run-dir "$RUN_ROOT/h80_p1440_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name patchmlp_all_tokenmask_e10_m0p3_dbloss_30s_h80_p1440_s1_seed2026
uv run python run_single.py --mode test --model PatchMLPAllFeatures --scenario NoFutureWeather --loss DBLoss --dbloss-weight 0.5 --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 80 --horizon-steps 2880 --seed 2026 --batch-size 32 --num-workers 4 --tqdm 0 --run-dir "$RUN_ROOT/h80_p2880_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name patchmlp_all_tokenmask_e10_m0p3_dbloss_30s_h80_p2880_s1_seed2026
