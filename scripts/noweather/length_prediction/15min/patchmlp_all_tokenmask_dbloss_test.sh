#!/bin/bash
set -e

# NoFutureWeather 下 PatchMLPAllFeatures 的随机时间 token 热身与 DBLoss 长度预测；15min一个模型点。
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
unset LD_LIBRARY_PATH

POINT_INTERVAL_SECONDS=900
DATASET_NAME=GuangningWindPower15min
RUN_ROOT=".runs/WeatherComparison/NoFutureWeather/PatchMLPAllFeatures/DBLoss/$DATASET_NAME/pretrain_e10_m0p3"

# 仅评估对应正式 record 指向的验证最优检查点。
uv run python run_single.py --mode test --model PatchMLPAllFeatures --scenario NoFutureWeather --loss DBLoss --dbloss-weight 0.5 --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 4 --horizon-steps 1 --seed 2026 --batch-size 32 --num-workers 4 --tqdm 0 --run-dir "$RUN_ROOT/h4_p1_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name patchmlp_all_tokenmask_e10_m0p3_dbloss_15min_h4_p1_s1_seed2026
uv run python run_single.py --mode test --model PatchMLPAllFeatures --scenario NoFutureWeather --loss DBLoss --dbloss-weight 0.5 --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 4 --horizon-steps 2 --seed 2026 --batch-size 32 --num-workers 4 --tqdm 0 --run-dir "$RUN_ROOT/h4_p2_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name patchmlp_all_tokenmask_e10_m0p3_dbloss_15min_h4_p2_s1_seed2026
uv run python run_single.py --mode test --model PatchMLPAllFeatures --scenario NoFutureWeather --loss DBLoss --dbloss-weight 0.5 --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 4 --horizon-steps 3 --seed 2026 --batch-size 32 --num-workers 4 --tqdm 0 --run-dir "$RUN_ROOT/h4_p3_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name patchmlp_all_tokenmask_e10_m0p3_dbloss_15min_h4_p3_s1_seed2026
uv run python run_single.py --mode test --model PatchMLPAllFeatures --scenario NoFutureWeather --loss DBLoss --dbloss-weight 0.5 --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 4 --horizon-steps 4 --seed 2026 --batch-size 32 --num-workers 4 --tqdm 0 --run-dir "$RUN_ROOT/h4_p4_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name patchmlp_all_tokenmask_e10_m0p3_dbloss_15min_h4_p4_s1_seed2026
uv run python run_single.py --mode test --model PatchMLPAllFeatures --scenario NoFutureWeather --loss DBLoss --dbloss-weight 0.5 --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 8 --horizon-steps 8 --seed 2026 --batch-size 32 --num-workers 4 --tqdm 0 --run-dir "$RUN_ROOT/h8_p8_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name patchmlp_all_tokenmask_e10_m0p3_dbloss_15min_h8_p8_s1_seed2026
uv run python run_single.py --mode test --model PatchMLPAllFeatures --scenario NoFutureWeather --loss DBLoss --dbloss-weight 0.5 --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 12 --horizon-steps 12 --seed 2026 --batch-size 32 --num-workers 4 --tqdm 0 --run-dir "$RUN_ROOT/h12_p12_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name patchmlp_all_tokenmask_e10_m0p3_dbloss_15min_h12_p12_s1_seed2026
uv run python run_single.py --mode test --model PatchMLPAllFeatures --scenario NoFutureWeather --loss DBLoss --dbloss-weight 0.5 --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 16 --horizon-steps 16 --seed 2026 --batch-size 32 --num-workers 4 --tqdm 0 --run-dir "$RUN_ROOT/h16_p16_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name patchmlp_all_tokenmask_e10_m0p3_dbloss_15min_h16_p16_s1_seed2026
uv run python run_single.py --mode test --model PatchMLPAllFeatures --scenario NoFutureWeather --loss DBLoss --dbloss-weight 0.5 --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 32 --horizon-steps 32 --seed 2026 --batch-size 32 --num-workers 4 --tqdm 0 --run-dir "$RUN_ROOT/h32_p32_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name patchmlp_all_tokenmask_e10_m0p3_dbloss_15min_h32_p32_s1_seed2026
uv run python run_single.py --mode test --model PatchMLPAllFeatures --scenario NoFutureWeather --loss DBLoss --dbloss-weight 0.5 --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 48 --horizon-steps 48 --seed 2026 --batch-size 32 --num-workers 4 --tqdm 0 --run-dir "$RUN_ROOT/h48_p48_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name patchmlp_all_tokenmask_e10_m0p3_dbloss_15min_h48_p48_s1_seed2026
uv run python run_single.py --mode test --model PatchMLPAllFeatures --scenario NoFutureWeather --loss DBLoss --dbloss-weight 0.5 --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 96 --horizon-steps 96 --seed 2026 --batch-size 32 --num-workers 4 --tqdm 0 --run-dir "$RUN_ROOT/h96_p96_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name patchmlp_all_tokenmask_e10_m0p3_dbloss_15min_h96_p96_s1_seed2026
