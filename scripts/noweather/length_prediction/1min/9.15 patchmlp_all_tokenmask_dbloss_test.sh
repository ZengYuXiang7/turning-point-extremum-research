#!/bin/bash
set -e

# NoFutureWeather 下 PatchMLPAllFeatures 的随机时间 token 热身与 DBLoss 长度预测；1min一个模型点。
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
unset LD_LIBRARY_PATH

POINT_INTERVAL_SECONDS=60
DATASET_NAME=GuangningWindPower1min
RUN_ROOT=".runs/WeatherComparison/NoFutureWeather/PatchMLPAllFeatures/DBLoss/$DATASET_NAME/pretrain_e20_m0p3"

# 仅评估对应正式 record 指向的验证最优检查点。
uv run python run_single.py --mode test --model PatchMLPAllFeatures --scenario NoFutureWeather --loss DBLoss --dbloss-weight 0.5 --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 60 --horizon-steps 15 --seed 2026 --batch-size 32 --num-workers 4 --show-progress 0 --run-dir "$RUN_ROOT/h60_p15_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name patchmlp_all_tokenmask_e20_m0p3_dbloss_1min_h60_p15_s1_seed2026
uv run python run_single.py --mode test --model PatchMLPAllFeatures --scenario NoFutureWeather --loss DBLoss --dbloss-weight 0.5 --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 60 --horizon-steps 30 --seed 2026 --batch-size 32 --num-workers 4 --show-progress 0 --run-dir "$RUN_ROOT/h60_p30_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name patchmlp_all_tokenmask_e20_m0p3_dbloss_1min_h60_p30_s1_seed2026
uv run python run_single.py --mode test --model PatchMLPAllFeatures --scenario NoFutureWeather --loss DBLoss --dbloss-weight 0.5 --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 60 --horizon-steps 60 --seed 2026 --batch-size 32 --num-workers 4 --show-progress 0 --run-dir "$RUN_ROOT/h60_p60_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name patchmlp_all_tokenmask_e20_m0p3_dbloss_1min_h60_p60_s1_seed2026
uv run python run_single.py --mode test --model PatchMLPAllFeatures --scenario NoFutureWeather --loss DBLoss --dbloss-weight 0.5 --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 120 --horizon-steps 120 --seed 2026 --batch-size 32 --num-workers 4 --show-progress 0 --run-dir "$RUN_ROOT/h120_p120_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name patchmlp_all_tokenmask_e20_m0p3_dbloss_1min_h120_p120_s1_seed2026
uv run python run_single.py --mode test --model PatchMLPAllFeatures --scenario NoFutureWeather --loss DBLoss --dbloss-weight 0.5 --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 240 --horizon-steps 240 --seed 2026 --batch-size 32 --num-workers 4 --show-progress 0 --run-dir "$RUN_ROOT/h240_p240_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name patchmlp_all_tokenmask_e20_m0p3_dbloss_1min_h240_p240_s1_seed2026
uv run python run_single.py --mode test --model PatchMLPAllFeatures --scenario NoFutureWeather --loss DBLoss --dbloss-weight 0.5 --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 480 --horizon-steps 480 --seed 2026 --batch-size 32 --num-workers 4 --show-progress 0 --run-dir "$RUN_ROOT/h480_p480_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name patchmlp_all_tokenmask_e20_m0p3_dbloss_1min_h480_p480_s1_seed2026
uv run python run_single.py --mode test --model PatchMLPAllFeatures --scenario NoFutureWeather --loss DBLoss --dbloss-weight 0.5 --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 720 --horizon-steps 720 --seed 2026 --batch-size 32 --num-workers 4 --show-progress 0 --run-dir "$RUN_ROOT/h720_p720_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name patchmlp_all_tokenmask_e20_m0p3_dbloss_1min_h720_p720_s1_seed2026
uv run python run_single.py --mode test --model PatchMLPAllFeatures --scenario NoFutureWeather --loss DBLoss --dbloss-weight 0.5 --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 1440 --horizon-steps 1440 --seed 2026 --batch-size 32 --num-workers 4 --show-progress 0 --run-dir "$RUN_ROOT/h1440_p1440_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name patchmlp_all_tokenmask_e20_m0p3_dbloss_1min_h1440_p1440_s1_seed2026
