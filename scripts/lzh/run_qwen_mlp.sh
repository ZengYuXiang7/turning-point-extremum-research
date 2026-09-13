#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${PYTHON:-python}"
cd "$ROOT"
export PYTHONUNBUFFERED=1
unset LD_LIBRARY_PATH

HORIZONS="${HORIZONS:-15 720 1440 2880}"
LOSSES="${LOSSES:-MSE DBLoss}"
BATCH_SIZE="${BATCH_SIZE:-4}"
WINDOW_STRIDE_STEPS="${WINDOW_STRIDE_STEPS:-360}"

for loss in $LOSSES; do
  for horizon in $HORIZONS; do
    "$PYTHON" run.py \
      --model QwenMLP \
      --scenario NoFutureWeather \
      --loss "$loss" \
      --horizon "$horizon" \
      --history 240 \
      --seed 2026 \
      --epochs 100 \
      --patience 10 \
      --batch-size "$BATCH_SIZE" \
      --window-stride-steps "$WINDOW_STRIDE_STEPS" \
      --learning-rate 0.0001
  done
done
