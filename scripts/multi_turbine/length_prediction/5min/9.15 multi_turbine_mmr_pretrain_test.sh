#!/bin/bash
set -e

# 16台风机联合面板 MultiTurbine 的遮蔽市场重建热身后长度预测；5min一个模型点，DTW先验关闭。
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
unset LD_LIBRARY_PATH

POINT_INTERVAL_SECONDS=300
DATASET_NAME=GuangningWindPower16Turbine5min
RUN_ROOT=".runs/WeatherComparison/MultiTurbine/MultiTurbine/MSE/$DATASET_NAME/pretrain_e20_m0p3"

# 仅评估对应正式 record 指向的验证最优检查点。
uv run python run_multi.py --mode test --model MultiTurbine --scenario MultiTurbine --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 48 --horizon-steps 3 --seed 2026 --batch-size 32 --num-workers 4 --show-progress 0 --satra-use-dtw-prior 0 --run-dir "$RUN_ROOT/h48_p3_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name multi_turbine_mmr_e20_m0p3_h48_p3_s1_seed2026
uv run python run_multi.py --mode test --model MultiTurbine --scenario MultiTurbine --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 48 --horizon-steps 6 --seed 2026 --batch-size 32 --num-workers 4 --show-progress 0 --satra-use-dtw-prior 0 --run-dir "$RUN_ROOT/h48_p6_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name multi_turbine_mmr_e20_m0p3_h48_p6_s1_seed2026
uv run python run_multi.py --mode test --model MultiTurbine --scenario MultiTurbine --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 48 --horizon-steps 9 --seed 2026 --batch-size 32 --num-workers 4 --show-progress 0 --satra-use-dtw-prior 0 --run-dir "$RUN_ROOT/h48_p9_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name multi_turbine_mmr_e20_m0p3_h48_p9_s1_seed2026
uv run python run_multi.py --mode test --model MultiTurbine --scenario MultiTurbine --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 48 --horizon-steps 12 --seed 2026 --batch-size 32 --num-workers 4 --show-progress 0 --satra-use-dtw-prior 0 --run-dir "$RUN_ROOT/h48_p12_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name multi_turbine_mmr_e20_m0p3_h48_p12_s1_seed2026
uv run python run_multi.py --mode test --model MultiTurbine --scenario MultiTurbine --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 48 --horizon-steps 24 --seed 2026 --batch-size 32 --num-workers 4 --show-progress 0 --satra-use-dtw-prior 0 --run-dir "$RUN_ROOT/h48_p24_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name multi_turbine_mmr_e20_m0p3_h48_p24_s1_seed2026
uv run python run_multi.py --mode test --model MultiTurbine --scenario MultiTurbine --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 48 --horizon-steps 36 --seed 2026 --batch-size 32 --num-workers 4 --show-progress 0 --satra-use-dtw-prior 0 --run-dir "$RUN_ROOT/h48_p36_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name multi_turbine_mmr_e20_m0p3_h48_p36_s1_seed2026
uv run python run_multi.py --mode test --model MultiTurbine --scenario MultiTurbine --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 48 --horizon-steps 48 --seed 2026 --batch-size 32 --num-workers 4 --show-progress 0 --satra-use-dtw-prior 0 --run-dir "$RUN_ROOT/h48_p48_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name multi_turbine_mmr_e20_m0p3_h48_p48_s1_seed2026
uv run python run_multi.py --mode test --model MultiTurbine --scenario MultiTurbine --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 48 --horizon-steps 96 --seed 2026 --batch-size 32 --num-workers 4 --show-progress 0 --satra-use-dtw-prior 0 --run-dir "$RUN_ROOT/h48_p96_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name multi_turbine_mmr_e20_m0p3_h48_p96_s1_seed2026
uv run python run_multi.py --mode test --model MultiTurbine --scenario MultiTurbine --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 48 --horizon-steps 144 --seed 2026 --batch-size 32 --num-workers 4 --show-progress 0 --satra-use-dtw-prior 0 --run-dir "$RUN_ROOT/h48_p144_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name multi_turbine_mmr_e20_m0p3_h48_p144_s1_seed2026
uv run python run_multi.py --mode test --model MultiTurbine --scenario MultiTurbine --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 48 --horizon-steps 288 --seed 2026 --batch-size 32 --num-workers 4 --show-progress 0 --satra-use-dtw-prior 0 --run-dir "$RUN_ROOT/h48_p288_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name multi_turbine_mmr_e20_m0p3_h48_p288_s1_seed2026
