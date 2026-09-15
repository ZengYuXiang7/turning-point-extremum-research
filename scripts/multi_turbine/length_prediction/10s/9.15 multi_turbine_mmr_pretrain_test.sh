#!/bin/bash
set -e

# 16台风机联合面板 MultiTurbine 的遮蔽市场重建热身后长度预测；10s一个模型点，DTW先验关闭。
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
unset LD_LIBRARY_PATH

POINT_INTERVAL_SECONDS=10
DATASET_NAME=GuangningWindPower16Turbine10s
RUN_ROOT=".runs/WeatherComparison/MultiTurbine/MultiTurbine/MSE/$DATASET_NAME/pretrain_e20_m0p3"

# 仅评估对应正式 record 指向的验证最优检查点。
uv run python run_multi.py --mode test --model MultiTurbine --scenario MultiTurbine --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 1440 --horizon-steps 90 --seed 2026 --batch-size 1 --num-workers 4 --show-progress 0 --satra-use-dtw-prior 0 --run-dir "$RUN_ROOT/h1440_p90_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name multi_turbine_mmr_e20_m0p3_10s_h1440_p90_s1_seed2026
uv run python run_multi.py --mode test --model MultiTurbine --scenario MultiTurbine --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 1440 --horizon-steps 180 --seed 2026 --batch-size 1 --num-workers 4 --show-progress 0 --satra-use-dtw-prior 0 --run-dir "$RUN_ROOT/h1440_p180_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name multi_turbine_mmr_e20_m0p3_10s_h1440_p180_s1_seed2026
uv run python run_multi.py --mode test --model MultiTurbine --scenario MultiTurbine --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 1440 --horizon-steps 270 --seed 2026 --batch-size 1 --num-workers 4 --show-progress 0 --satra-use-dtw-prior 0 --run-dir "$RUN_ROOT/h1440_p270_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name multi_turbine_mmr_e20_m0p3_10s_h1440_p270_s1_seed2026
uv run python run_multi.py --mode test --model MultiTurbine --scenario MultiTurbine --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 1440 --horizon-steps 360 --seed 2026 --batch-size 1 --num-workers 4 --show-progress 0 --satra-use-dtw-prior 0 --run-dir "$RUN_ROOT/h1440_p360_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name multi_turbine_mmr_e20_m0p3_10s_h1440_p360_s1_seed2026
uv run python run_multi.py --mode test --model MultiTurbine --scenario MultiTurbine --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 1440 --horizon-steps 720 --seed 2026 --batch-size 1 --num-workers 4 --show-progress 0 --satra-use-dtw-prior 0 --run-dir "$RUN_ROOT/h1440_p720_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name multi_turbine_mmr_e20_m0p3_10s_h1440_p720_s1_seed2026
uv run python run_multi.py --mode test --model MultiTurbine --scenario MultiTurbine --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 1440 --horizon-steps 1080 --seed 2026 --batch-size 1 --num-workers 4 --show-progress 0 --satra-use-dtw-prior 0 --run-dir "$RUN_ROOT/h1440_p1080_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name multi_turbine_mmr_e20_m0p3_10s_h1440_p1080_s1_seed2026
uv run python run_multi.py --mode test --model MultiTurbine --scenario MultiTurbine --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 1440 --horizon-steps 1440 --seed 2026 --batch-size 1 --num-workers 4 --show-progress 0 --satra-use-dtw-prior 0 --run-dir "$RUN_ROOT/h1440_p1440_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name multi_turbine_mmr_e20_m0p3_10s_h1440_p1440_s1_seed2026
uv run python run_multi.py --mode test --model MultiTurbine --scenario MultiTurbine --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 1440 --horizon-steps 2880 --seed 2026 --batch-size 1 --num-workers 4 --show-progress 0 --satra-use-dtw-prior 0 --run-dir "$RUN_ROOT/h1440_p2880_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name multi_turbine_mmr_e20_m0p3_10s_h1440_p2880_s1_seed2026
uv run python run_multi.py --mode test --model MultiTurbine --scenario MultiTurbine --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 1440 --horizon-steps 4320 --seed 2026 --batch-size 1 --num-workers 4 --show-progress 0 --satra-use-dtw-prior 0 --run-dir "$RUN_ROOT/h1440_p4320_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name multi_turbine_mmr_e20_m0p3_10s_h1440_p4320_s1_seed2026
uv run python run_multi.py --mode test --model MultiTurbine --scenario MultiTurbine --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 1440 --horizon-steps 8640 --seed 2026 --batch-size 1 --num-workers 4 --show-progress 0 --satra-use-dtw-prior 0 --run-dir "$RUN_ROOT/h1440_p8640_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name multi_turbine_mmr_e20_m0p3_10s_h1440_p8640_s1_seed2026
