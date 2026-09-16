#!/bin/bash
set -e

# 16台风机联合面板 MultiTurbine 的遮蔽市场重建热身后长度预测；15min一个模型点，DTW先验关闭。
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
unset LD_LIBRARY_PATH

POINT_INTERVAL_SECONDS=900
DATASET_NAME=GuangningWindPower16Turbine15min
RUN_ROOT=".runs/WeatherComparison/MultiTurbine/MultiTurbine/MSE/$DATASET_NAME/pretrain_e10_m0p3"

# 仅评估对应正式 record 指向的验证最优检查点。
uv run python run_multi.py --mode test --model MultiTurbine --scenario MultiTurbine --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 16 --horizon-steps 1 --seed 2026 --batch-size 32 --num-workers 4 --tqdm 0 --satra-use-dtw-prior 0 --run-dir "$RUN_ROOT/h16_p1_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name multi_turbine_mmr_e10_m0p3_h16_p1_s1_seed2026
uv run python run_multi.py --mode test --model MultiTurbine --scenario MultiTurbine --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 16 --horizon-steps 2 --seed 2026 --batch-size 32 --num-workers 4 --tqdm 0 --satra-use-dtw-prior 0 --run-dir "$RUN_ROOT/h16_p2_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name multi_turbine_mmr_e10_m0p3_h16_p2_s1_seed2026
uv run python run_multi.py --mode test --model MultiTurbine --scenario MultiTurbine --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 16 --horizon-steps 3 --seed 2026 --batch-size 32 --num-workers 4 --tqdm 0 --satra-use-dtw-prior 0 --run-dir "$RUN_ROOT/h16_p3_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name multi_turbine_mmr_e10_m0p3_h16_p3_s1_seed2026
uv run python run_multi.py --mode test --model MultiTurbine --scenario MultiTurbine --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 16 --horizon-steps 4 --seed 2026 --batch-size 32 --num-workers 4 --tqdm 0 --satra-use-dtw-prior 0 --run-dir "$RUN_ROOT/h16_p4_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name multi_turbine_mmr_e10_m0p3_h16_p4_s1_seed2026
uv run python run_multi.py --mode test --model MultiTurbine --scenario MultiTurbine --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 16 --horizon-steps 8 --seed 2026 --batch-size 32 --num-workers 4 --tqdm 0 --satra-use-dtw-prior 0 --run-dir "$RUN_ROOT/h16_p8_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name multi_turbine_mmr_e10_m0p3_h16_p8_s1_seed2026
uv run python run_multi.py --mode test --model MultiTurbine --scenario MultiTurbine --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 16 --horizon-steps 12 --seed 2026 --batch-size 32 --num-workers 4 --tqdm 0 --satra-use-dtw-prior 0 --run-dir "$RUN_ROOT/h16_p12_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name multi_turbine_mmr_e10_m0p3_h16_p12_s1_seed2026
uv run python run_multi.py --mode test --model MultiTurbine --scenario MultiTurbine --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 16 --horizon-steps 16 --seed 2026 --batch-size 32 --num-workers 4 --tqdm 0 --satra-use-dtw-prior 0 --run-dir "$RUN_ROOT/h16_p16_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name multi_turbine_mmr_e10_m0p3_h16_p16_s1_seed2026
uv run python run_multi.py --mode test --model MultiTurbine --scenario MultiTurbine --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 16 --horizon-steps 32 --seed 2026 --batch-size 32 --num-workers 4 --tqdm 0 --satra-use-dtw-prior 0 --run-dir "$RUN_ROOT/h16_p32_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name multi_turbine_mmr_e10_m0p3_h16_p32_s1_seed2026
uv run python run_multi.py --mode test --model MultiTurbine --scenario MultiTurbine --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 16 --horizon-steps 48 --seed 2026 --batch-size 32 --num-workers 4 --tqdm 0 --satra-use-dtw-prior 0 --run-dir "$RUN_ROOT/h16_p48_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name multi_turbine_mmr_e10_m0p3_h16_p48_s1_seed2026
uv run python run_multi.py --mode test --model MultiTurbine --scenario MultiTurbine --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 16 --horizon-steps 96 --seed 2026 --batch-size 32 --num-workers 4 --tqdm 0 --satra-use-dtw-prior 0 --run-dir "$RUN_ROOT/h16_p96_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name multi_turbine_mmr_e10_m0p3_h16_p96_s1_seed2026
