#!/bin/bash
set -e

# 使用16台风机联合面板训练StockEcho预测长度扫描
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
unset LD_LIBRARY_PATH

POINT_INTERVAL_SECONDS=900
DATASET_NAME=GuangningWindPower16Turbine15min
RUN_ROOT=".runs/WeatherComparison/MultiTurbine/StockEcho/MSE/$DATASET_NAME/length_prediction"

# 固定4小时历史，预测15/30/45分钟及1/2/3/4/8/12/24小时
uv run python run_multi.py --mode train --model StockEcho --scenario MultiTurbine --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 16 --horizon-steps 1 --seed 2026 --epochs 100 --patience 10 --batch-size 32 --learning-rate 0.001 --num-workers 4 --show-progress 0 --run-dir "$RUN_ROOT/h16_p1_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name stockecho_h16_p1_s1_seed2026
uv run python run_multi.py --mode train --model StockEcho --scenario MultiTurbine --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 16 --horizon-steps 2 --seed 2026 --epochs 100 --patience 10 --batch-size 32 --learning-rate 0.001 --num-workers 4 --show-progress 0 --run-dir "$RUN_ROOT/h16_p2_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name stockecho_h16_p2_s1_seed2026
uv run python run_multi.py --mode train --model StockEcho --scenario MultiTurbine --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 16 --horizon-steps 3 --seed 2026 --epochs 100 --patience 10 --batch-size 32 --learning-rate 0.001 --num-workers 4 --show-progress 0 --run-dir "$RUN_ROOT/h16_p3_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name stockecho_h16_p3_s1_seed2026
uv run python run_multi.py --mode train --model StockEcho --scenario MultiTurbine --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 16 --horizon-steps 4 --seed 2026 --epochs 100 --patience 10 --batch-size 32 --learning-rate 0.001 --num-workers 4 --show-progress 0 --run-dir "$RUN_ROOT/h16_p4_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name stockecho_h16_p4_s1_seed2026
uv run python run_multi.py --mode train --model StockEcho --scenario MultiTurbine --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 16 --horizon-steps 8 --seed 2026 --epochs 100 --patience 10 --batch-size 32 --learning-rate 0.001 --num-workers 4 --show-progress 0 --run-dir "$RUN_ROOT/h16_p8_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name stockecho_h16_p8_s1_seed2026
uv run python run_multi.py --mode train --model StockEcho --scenario MultiTurbine --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 16 --horizon-steps 12 --seed 2026 --epochs 100 --patience 10 --batch-size 32 --learning-rate 0.001 --num-workers 4 --show-progress 0 --run-dir "$RUN_ROOT/h16_p12_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name stockecho_h16_p12_s1_seed2026
uv run python run_multi.py --mode train --model StockEcho --scenario MultiTurbine --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 16 --horizon-steps 16 --seed 2026 --epochs 100 --patience 10 --batch-size 32 --learning-rate 0.001 --num-workers 4 --show-progress 0 --run-dir "$RUN_ROOT/h16_p16_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name stockecho_h16_p16_s1_seed2026
uv run python run_multi.py --mode train --model StockEcho --scenario MultiTurbine --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 16 --horizon-steps 32 --seed 2026 --epochs 100 --patience 10 --batch-size 32 --learning-rate 0.001 --num-workers 4 --show-progress 0 --run-dir "$RUN_ROOT/h16_p32_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name stockecho_h16_p32_s1_seed2026
uv run python run_multi.py --mode train --model StockEcho --scenario MultiTurbine --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 16 --horizon-steps 48 --seed 2026 --epochs 100 --patience 10 --batch-size 32 --learning-rate 0.001 --num-workers 4 --show-progress 0 --run-dir "$RUN_ROOT/h16_p48_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name stockecho_h16_p48_s1_seed2026
uv run python run_multi.py --mode train --model StockEcho --scenario MultiTurbine --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 16 --horizon-steps 96 --seed 2026 --epochs 100 --patience 10 --batch-size 32 --learning-rate 0.001 --num-workers 4 --show-progress 0 --run-dir "$RUN_ROOT/h16_p96_s1_seed2026" --dataset-name "$DATASET_NAME" --result-name stockecho_h16_p96_s1_seed2026
