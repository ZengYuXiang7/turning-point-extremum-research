#!/bin/bash
set -e

# 16台风机联合面板 MultiTurbine 的遮蔽市场重建热身后长度预测；1min一个模型点，DTW先验关闭。
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
unset LD_LIBRARY_PATH

POINT_INTERVAL_SECONDS=60
DATASET_NAME=GuangningWindPower16Turbine1min
RUN_ROOT=".runs/WeatherComparison/MultiTurbine/MultiTurbine/MSE/$DATASET_NAME/pretrain_e20_m0p3"

# 仅评估对应正式 record 指向的验证最优检查点。
uv run python run_multi.py --mode test --model MultiTurbine --scenario MultiTurbine --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 240 --horizon-steps 15 --seed 2026 --batch-size 32 --num-workers 4 --show-progress 0 --satra-use-dtw-prior 0 --run-dir "$RUN_ROOT/h240_p15_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name multi_turbine_mmr_e20_m0p3_h240_p15_s1_seed2026
uv run python run_multi.py --mode test --model MultiTurbine --scenario MultiTurbine --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 240 --horizon-steps 30 --seed 2026 --batch-size 32 --num-workers 4 --show-progress 0 --satra-use-dtw-prior 0 --run-dir "$RUN_ROOT/h240_p30_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name multi_turbine_mmr_e20_m0p3_h240_p30_s1_seed2026
uv run python run_multi.py --mode test --model MultiTurbine --scenario MultiTurbine --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 240 --horizon-steps 45 --seed 2026 --batch-size 32 --num-workers 4 --show-progress 0 --satra-use-dtw-prior 0 --run-dir "$RUN_ROOT/h240_p45_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name multi_turbine_mmr_e20_m0p3_h240_p45_s1_seed2026
uv run python run_multi.py --mode test --model MultiTurbine --scenario MultiTurbine --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 240 --horizon-steps 60 --seed 2026 --batch-size 32 --num-workers 4 --show-progress 0 --satra-use-dtw-prior 0 --run-dir "$RUN_ROOT/h240_p60_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name multi_turbine_mmr_e20_m0p3_h240_p60_s1_seed2026
uv run python run_multi.py --mode test --model MultiTurbine --scenario MultiTurbine --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 240 --horizon-steps 120 --seed 2026 --batch-size 32 --num-workers 4 --show-progress 0 --satra-use-dtw-prior 0 --run-dir "$RUN_ROOT/h240_p120_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name multi_turbine_mmr_e20_m0p3_h240_p120_s1_seed2026
uv run python run_multi.py --mode test --model MultiTurbine --scenario MultiTurbine --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 240 --horizon-steps 180 --seed 2026 --batch-size 32 --num-workers 4 --show-progress 0 --satra-use-dtw-prior 0 --run-dir "$RUN_ROOT/h240_p180_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name multi_turbine_mmr_e20_m0p3_h240_p180_s1_seed2026
uv run python run_multi.py --mode test --model MultiTurbine --scenario MultiTurbine --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 240 --horizon-steps 240 --seed 2026 --batch-size 32 --num-workers 4 --show-progress 0 --satra-use-dtw-prior 0 --run-dir "$RUN_ROOT/h240_p240_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name multi_turbine_mmr_e20_m0p3_h240_p240_s1_seed2026
uv run python run_multi.py --mode test --model MultiTurbine --scenario MultiTurbine --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 240 --horizon-steps 480 --seed 2026 --batch-size 32 --num-workers 4 --show-progress 0 --satra-use-dtw-prior 0 --run-dir "$RUN_ROOT/h240_p480_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name multi_turbine_mmr_e20_m0p3_h240_p480_s1_seed2026
uv run python run_multi.py --mode test --model MultiTurbine --scenario MultiTurbine --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 240 --horizon-steps 720 --seed 2026 --batch-size 32 --num-workers 4 --show-progress 0 --satra-use-dtw-prior 0 --run-dir "$RUN_ROOT/h240_p720_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name multi_turbine_mmr_e20_m0p3_h240_p720_s1_seed2026
uv run python run_multi.py --mode test --model MultiTurbine --scenario MultiTurbine --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 240 --horizon-steps 1440 --seed 2026 --batch-size 32 --num-workers 4 --show-progress 0 --satra-use-dtw-prior 0 --run-dir "$RUN_ROOT/h240_p1440_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name multi_turbine_mmr_e20_m0p3_h240_p1440_s1_seed2026
