#!/bin/bash
set -e

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

export PYTHONUNBUFFERED=1
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"

PYTHON="${PYTHON:-python}"
MODE="${MODE:-train}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-0}"
EPOCHS="${EPOCHS:-3}"
PATIENCE="${PATIENCE:-3}"
RESULT_NAME="${RESULT_NAME:-lmt_TimeMoEARevIN_Guangning_Mac}"
RUN_DIR="${RUN_DIR:-.runs/WeatherComparison/NoFutureWeather/TimeMoEARevIN/MSE/point_900s/history_16steps/horizon_1steps/stride_1steps/seed2026_mac}"

"$PYTHON" run_single.py \
  --mode "$MODE" \
  --model TimeMoEARevIN \
  --scenario NoFutureWeather \
  --loss MSE \
  --point-interval-seconds 900 \
  --history-steps 16 \
  --horizon-steps 1 \
  --window-stride-steps 1 \
  --seed 2026 \
  --epochs "$EPOCHS" \
  --patience "$PATIENCE" \
  --batch-size "$BATCH_SIZE" \
  --learning-rate 0.001 \
  --num-workers "$NUM_WORKERS" \
  --tqdm 1 \
  --time-moe-path models/pretrained/TimeMoE-50M \
  --time-moe-bottleneck 512 \
  --time-moe-unfreeze-layers 0 \
  --time-moe-backbone-learning-rate 0.00001 \
  --time-moe-gradient-checkpointing 0 \
  --visualize-after-train 0 \
  --dataset-name GuangningWindPower15min \
  --run-dir "$RUN_DIR" \
  --result-name "$RESULT_NAME"
