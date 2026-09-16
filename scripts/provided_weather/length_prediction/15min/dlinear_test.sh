#!/bin/bash
set -e

# 测试甲方未来预测风速场景下已训练的DLinear checkpoint。
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
unset LD_LIBRARY_PATH

POINT_INTERVAL_SECONDS=900
DATASET_NAME=GuangningSiteProvidedWeather15min
PROVIDED_WEATHER_DIR="甲方提供新的内容/9月11日"
RUN_ROOT=".runs/WeatherComparison/ProvidedFutureWeather/DLinear/MSE/$DATASET_NAME"

uv run python run_single.py --mode test --model DLinear --scenario ProvidedFutureWeather --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 4 --horizon-steps 1 --seed 2026 --batch-size 64 --num-workers 4 --tqdm 0 --run-dir "$RUN_ROOT/h4_p1_s1_seed2026" --dataset-name "$DATASET_NAME" --provided-weather-dir "$PROVIDED_WEATHER_DIR" --result-name provided_weather_dlinear_h4_p1_s1_seed2026
uv run python run_single.py --mode test --model DLinear --scenario ProvidedFutureWeather --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 4 --horizon-steps 2 --seed 2026 --batch-size 64 --num-workers 4 --tqdm 0 --run-dir "$RUN_ROOT/h4_p2_s1_seed2026" --dataset-name "$DATASET_NAME" --provided-weather-dir "$PROVIDED_WEATHER_DIR" --result-name provided_weather_dlinear_h4_p2_s1_seed2026
uv run python run_single.py --mode test --model DLinear --scenario ProvidedFutureWeather --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 4 --horizon-steps 3 --seed 2026 --batch-size 64 --num-workers 4 --tqdm 0 --run-dir "$RUN_ROOT/h4_p3_s1_seed2026" --dataset-name "$DATASET_NAME" --provided-weather-dir "$PROVIDED_WEATHER_DIR" --result-name provided_weather_dlinear_h4_p3_s1_seed2026
uv run python run_single.py --mode test --model DLinear --scenario ProvidedFutureWeather --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 4 --horizon-steps 4 --seed 2026 --batch-size 64 --num-workers 4 --tqdm 0 --run-dir "$RUN_ROOT/h4_p4_s1_seed2026" --dataset-name "$DATASET_NAME" --provided-weather-dir "$PROVIDED_WEATHER_DIR" --result-name provided_weather_dlinear_h4_p4_s1_seed2026
uv run python run_single.py --mode test --model DLinear --scenario ProvidedFutureWeather --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 8 --horizon-steps 8 --seed 2026 --batch-size 64 --num-workers 4 --tqdm 0 --run-dir "$RUN_ROOT/h8_p8_s1_seed2026" --dataset-name "$DATASET_NAME" --provided-weather-dir "$PROVIDED_WEATHER_DIR" --result-name provided_weather_dlinear_h8_p8_s1_seed2026
uv run python run_single.py --mode test --model DLinear --scenario ProvidedFutureWeather --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 12 --horizon-steps 12 --seed 2026 --batch-size 64 --num-workers 4 --tqdm 0 --run-dir "$RUN_ROOT/h12_p12_s1_seed2026" --dataset-name "$DATASET_NAME" --provided-weather-dir "$PROVIDED_WEATHER_DIR" --result-name provided_weather_dlinear_h12_p12_s1_seed2026
uv run python run_single.py --mode test --model DLinear --scenario ProvidedFutureWeather --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 16 --horizon-steps 16 --seed 2026 --batch-size 64 --num-workers 4 --tqdm 0 --run-dir "$RUN_ROOT/h16_p16_s1_seed2026" --dataset-name "$DATASET_NAME" --provided-weather-dir "$PROVIDED_WEATHER_DIR" --result-name provided_weather_dlinear_h16_p16_s1_seed2026
uv run python run_single.py --mode test --model DLinear --scenario ProvidedFutureWeather --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 32 --horizon-steps 32 --seed 2026 --batch-size 64 --num-workers 4 --tqdm 0 --run-dir "$RUN_ROOT/h32_p32_s1_seed2026" --dataset-name "$DATASET_NAME" --provided-weather-dir "$PROVIDED_WEATHER_DIR" --result-name provided_weather_dlinear_h32_p32_s1_seed2026
uv run python run_single.py --mode test --model DLinear --scenario ProvidedFutureWeather --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 48 --horizon-steps 48 --seed 2026 --batch-size 64 --num-workers 4 --tqdm 0 --run-dir "$RUN_ROOT/h48_p48_s1_seed2026" --dataset-name "$DATASET_NAME" --provided-weather-dir "$PROVIDED_WEATHER_DIR" --result-name provided_weather_dlinear_h48_p48_s1_seed2026
uv run python run_single.py --mode test --model DLinear --scenario ProvidedFutureWeather --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 96 --horizon-steps 96 --seed 2026 --batch-size 64 --num-workers 4 --tqdm 0 --run-dir "$RUN_ROOT/h96_p96_s1_seed2026" --dataset-name "$DATASET_NAME" --provided-weather-dir "$PROVIDED_WEATHER_DIR" --result-name provided_weather_dlinear_h96_p96_s1_seed2026
