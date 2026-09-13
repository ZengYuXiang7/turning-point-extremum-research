#!/bin/bash
set -e

# NoFutureWeather：仅使用预测起点之前的19维历史特征预测未来功率。
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

# 清除系统 CUDA 12.1，使用项目环境中的 PyTorch cu124。
unset LD_LIBRARY_PATH

# 数据采样与训练共用该变量；900 秒即每点15分钟。
SAMPLE_SECONDS=900

# 修改采样间隔后，先取消下一行注释以重建 dataset/processed。
# uv run python generate_data.py --sample-seconds "$SAMPLE_SECONDS"

# 每个点为15分钟；输入60分钟（4点），预测15分钟（1点）。
uv run python run.py --model DLinear --scenario NoFutureWeather --loss MSE --sample-seconds "$SAMPLE_SECONDS" --window-stride-steps 1 --history-steps 4 --horizon-steps 1 --batch-size 1024

# 输入60分钟（4点），预测30分钟（2点）。
uv run python run.py --model DLinear --scenario NoFutureWeather --loss MSE --sample-seconds "$SAMPLE_SECONDS" --window-stride-steps 1 --history-steps 4 --horizon-steps 2 --batch-size 1024

# 输入60分钟（4点），预测45分钟（3点）。
uv run python run.py --model DLinear --scenario NoFutureWeather --loss MSE --sample-seconds "$SAMPLE_SECONDS" --window-stride-steps 1 --history-steps 4 --horizon-steps 3 --batch-size 1024

# 输入60分钟（4点），预测1小时（4点）。
uv run python run.py --model DLinear --scenario NoFutureWeather --loss MSE --sample-seconds "$SAMPLE_SECONDS" --window-stride-steps 1 --history-steps 4 --horizon-steps 4 --batch-size 1024

# 输入2小时（8点），预测2小时（8点）。
uv run python run.py --model DLinear --scenario NoFutureWeather --loss MSE --sample-seconds "$SAMPLE_SECONDS" --window-stride-steps 1 --history-steps 8 --horizon-steps 8 --batch-size 1024

# 输入3小时（12点），预测3小时（12点）。
uv run python run.py --model DLinear --scenario NoFutureWeather --loss MSE --sample-seconds "$SAMPLE_SECONDS" --window-stride-steps 1 --history-steps 12 --horizon-steps 12 --batch-size 1024

# 输入4小时（16点），预测4小时（16点）。
uv run python run.py --model DLinear --scenario NoFutureWeather --loss MSE --sample-seconds "$SAMPLE_SECONDS" --window-stride-steps 1 --history-steps 16 --horizon-steps 16 --batch-size 1024

# 输入8小时（32点），预测8小时（32点）。
uv run python run.py --model DLinear --scenario NoFutureWeather --loss MSE --sample-seconds "$SAMPLE_SECONDS" --window-stride-steps 1 --history-steps 32 --horizon-steps 32 --batch-size 1024

# 输入12小时（48点），预测12小时（48点）。
uv run python run.py --model DLinear --scenario NoFutureWeather --loss MSE --sample-seconds "$SAMPLE_SECONDS" --window-stride-steps 1 --history-steps 48 --horizon-steps 48 --batch-size 1024

# 输入24小时（96点），预测24小时（96点）。
uv run python run.py --model DLinear --scenario NoFutureWeather --loss MSE --sample-seconds "$SAMPLE_SECONDS" --window-stride-steps 1 --history-steps 96 --horizon-steps 96 --batch-size 1024
