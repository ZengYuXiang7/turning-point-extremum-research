#!/bin/bash
set -e

# 16台风机联合面板 MultiTurbine 的遮蔽市场重建热身后长度预测；30s一个模型点，DTW先验关闭。
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
unset LD_LIBRARY_PATH

POINT_INTERVAL_SECONDS=30
DATASET_NAME=GuangningWindPower16Turbine30s
RUN_ROOT=".runs/WeatherComparison/MultiTurbine/MultiTurbine/MSE/$DATASET_NAME/pretrain_e10_m0p3"

# 仅评估对应正式 record 指向的验证最优检查点。
uv run python run_multi.py --mode test --model MultiTurbine --scenario MultiTurbine --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 480 --horizon-steps 30 --seed 2026 --batch-size 2 --num-workers 4 --tqdm 0 --satra-use-dtw-prior 0 --run-dir "$RUN_ROOT/h480_p30_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name multi_turbine_mmr_e10_m0p3_30s_h480_p30_s1_seed2026
uv run python run_multi.py --mode test --model MultiTurbine --scenario MultiTurbine --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 480 --horizon-steps 60 --seed 2026 --batch-size 2 --num-workers 4 --tqdm 0 --satra-use-dtw-prior 0 --run-dir "$RUN_ROOT/h480_p60_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name multi_turbine_mmr_e10_m0p3_30s_h480_p60_s1_seed2026
uv run python run_multi.py --mode test --model MultiTurbine --scenario MultiTurbine --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 480 --horizon-steps 90 --seed 2026 --batch-size 2 --num-workers 4 --tqdm 0 --satra-use-dtw-prior 0 --run-dir "$RUN_ROOT/h480_p90_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name multi_turbine_mmr_e10_m0p3_30s_h480_p90_s1_seed2026
uv run python run_multi.py --mode test --model MultiTurbine --scenario MultiTurbine --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 480 --horizon-steps 120 --seed 2026 --batch-size 2 --num-workers 4 --tqdm 0 --satra-use-dtw-prior 0 --run-dir "$RUN_ROOT/h480_p120_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name multi_turbine_mmr_e10_m0p3_30s_h480_p120_s1_seed2026
uv run python run_multi.py --mode test --model MultiTurbine --scenario MultiTurbine --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 480 --horizon-steps 240 --seed 2026 --batch-size 2 --num-workers 4 --tqdm 0 --satra-use-dtw-prior 0 --run-dir "$RUN_ROOT/h480_p240_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name multi_turbine_mmr_e10_m0p3_30s_h480_p240_s1_seed2026
uv run python run_multi.py --mode test --model MultiTurbine --scenario MultiTurbine --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 480 --horizon-steps 360 --seed 2026 --batch-size 2 --num-workers 4 --tqdm 0 --satra-use-dtw-prior 0 --run-dir "$RUN_ROOT/h480_p360_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name multi_turbine_mmr_e10_m0p3_30s_h480_p360_s1_seed2026
uv run python run_multi.py --mode test --model MultiTurbine --scenario MultiTurbine --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 480 --horizon-steps 480 --seed 2026 --batch-size 2 --num-workers 4 --tqdm 0 --satra-use-dtw-prior 0 --run-dir "$RUN_ROOT/h480_p480_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name multi_turbine_mmr_e10_m0p3_30s_h480_p480_s1_seed2026
uv run python run_multi.py --mode test --model MultiTurbine --scenario MultiTurbine --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 480 --horizon-steps 960 --seed 2026 --batch-size 2 --num-workers 4 --tqdm 0 --satra-use-dtw-prior 0 --run-dir "$RUN_ROOT/h480_p960_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name multi_turbine_mmr_e10_m0p3_30s_h480_p960_s1_seed2026
uv run python run_multi.py --mode test --model MultiTurbine --scenario MultiTurbine --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 480 --horizon-steps 1440 --seed 2026 --batch-size 2 --num-workers 4 --tqdm 0 --satra-use-dtw-prior 0 --run-dir "$RUN_ROOT/h480_p1440_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name multi_turbine_mmr_e10_m0p3_30s_h480_p1440_s1_seed2026
uv run python run_multi.py --mode test --model MultiTurbine --scenario MultiTurbine --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 480 --horizon-steps 2880 --seed 2026 --batch-size 2 --num-workers 4 --tqdm 0 --satra-use-dtw-prior 0 --run-dir "$RUN_ROOT/h480_p2880_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name multi_turbine_mmr_e10_m0p3_30s_h480_p2880_s1_seed2026
