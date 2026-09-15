#!/bin/bash
set -e

# 16台风机联合面板 MultiTurbine 的长度预测；1min一个模型点，空间关系与DTW先验关闭。
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
unset LD_LIBRARY_PATH

POINT_INTERVAL_SECONDS=60
DATASET_NAME=GuangningWindPower16Turbine1min
RUN_ROOT=".runs/WeatherComparison/MultiTurbine/MultiTurbine/MSE/$DATASET_NAME/no_spatial"

COMMON_ARGS="--mode train --model MultiTurbine --scenario MultiTurbine --loss MSE --point-interval-seconds $POINT_INTERVAL_SECONDS --window-stride-steps 1 --history-steps 240 --seed 2026 --epochs 100 --patience 10 --learning-rate 0.001 --batch-size 128 --num-workers 4 --show-progress 0 --satra-use-spatial-relation 0 --satra-use-dtw-prior 0 --dataset-name $DATASET_NAME"

# 固定4小时历史，扫描当前粒度下的多个预测长度。
uv run python run_multi.py $COMMON_ARGS --horizon-steps 15 --run-dir "$RUN_ROOT/h240_p15_s1_seed2026" --result-name multi_turbine_no_spatial_h240_p15_s1_seed2026
uv run python run_multi.py $COMMON_ARGS --horizon-steps 30 --run-dir "$RUN_ROOT/h240_p30_s1_seed2026" --result-name multi_turbine_no_spatial_h240_p30_s1_seed2026
uv run python run_multi.py $COMMON_ARGS --horizon-steps 45 --run-dir "$RUN_ROOT/h240_p45_s1_seed2026" --result-name multi_turbine_no_spatial_h240_p45_s1_seed2026
uv run python run_multi.py $COMMON_ARGS --horizon-steps 60 --run-dir "$RUN_ROOT/h240_p60_s1_seed2026" --result-name multi_turbine_no_spatial_h240_p60_s1_seed2026
uv run python run_multi.py $COMMON_ARGS --horizon-steps 120 --run-dir "$RUN_ROOT/h240_p120_s1_seed2026" --result-name multi_turbine_no_spatial_h240_p120_s1_seed2026
uv run python run_multi.py $COMMON_ARGS --horizon-steps 180 --run-dir "$RUN_ROOT/h240_p180_s1_seed2026" --result-name multi_turbine_no_spatial_h240_p180_s1_seed2026
uv run python run_multi.py $COMMON_ARGS --horizon-steps 240 --run-dir "$RUN_ROOT/h240_p240_s1_seed2026" --result-name multi_turbine_no_spatial_h240_p240_s1_seed2026
uv run python run_multi.py $COMMON_ARGS --horizon-steps 480 --run-dir "$RUN_ROOT/h240_p480_s1_seed2026" --result-name multi_turbine_no_spatial_h240_p480_s1_seed2026
uv run python run_multi.py $COMMON_ARGS --horizon-steps 720 --run-dir "$RUN_ROOT/h240_p720_s1_seed2026" --result-name multi_turbine_no_spatial_h240_p720_s1_seed2026
uv run python run_multi.py $COMMON_ARGS --horizon-steps 1440 --run-dir "$RUN_ROOT/h240_p1440_s1_seed2026" --result-name multi_turbine_no_spatial_h240_p1440_s1_seed2026
