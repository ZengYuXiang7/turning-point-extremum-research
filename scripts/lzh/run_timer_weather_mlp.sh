#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${PYTHON:-python}"
GPU="${GPU:-0}"
TIMER_PATH="${TIMER_PATH:-$ROOT/models/pretrained/timer-base-84m}"
BATCH_SIZE="${BATCH_SIZE:-64}"
WINDOW_STRIDE_STEPS="${WINDOW_STRIDE_STEPS:-360}"

cd "$ROOT"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="$GPU"
unset LD_LIBRARY_PATH

"$PYTHON" run.py \
  --model TimerWeatherMLP \
  --scenario OracleFutureWeather \
  --loss MSE \
  --horizon 15 \
  --history 240 \
  --seed 2026 \
  --epochs 100 \
  --patience 10 \
  --batch-size "$BATCH_SIZE" \
  --window-stride-steps "$WINDOW_STRIDE_STEPS" \
  --learning-rate 0.0001 \
  --timer-backbone-learning-rate 0.00001 \
  --timer-path "$TIMER_PATH" \
  --timer-patch-length 96 \
  --timer-bottleneck 512 \
  --timer-unfreeze-layers 2 \
  --timer-gradient-checkpointing 0 \
  --timer-residual-forecast 1
