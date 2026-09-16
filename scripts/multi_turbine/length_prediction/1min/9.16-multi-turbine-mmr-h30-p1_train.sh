#!/bin/bash
set -e

# 16台风机使用连续30个1分钟真实点预测下一分钟；空间关系开启，DTW先验关闭。
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
unset LD_LIBRARY_PATH

POINT_INTERVAL_SECONDS=60
DATASET_NAME=GuangningWindPower16Turbine1min
RUN_ROOT=".runs/WeatherComparison/MultiTurbine/MultiTurbine/MSE/$DATASET_NAME/pretrain-e5-m0p3-spatial"
RUN_DIR="$RUN_ROOT/h30_p1_s1_seed2026"
RESULT_NAME=multi-turbine-mmr-e5-m0p3-spatial-h30-p1-s1-seed2026
PLOT_OUTPUT="output/multi-turbine-1min-forecast"
EVALUATION_START="2026-08-01 09:45:00"
EVALUATION_END="2026-09-01 09:45:00"
PLOT_START="2026-08-01 09:45:00"

# 训练完成后按每分钟发布一次的场景评估31天，并生成滑动窗口图和六种时长的连续预测图。
uv run python run_multi.py --mode train --model MultiTurbine --scenario MultiTurbine --loss MSE --point-interval-seconds "$POINT_INTERVAL_SECONDS" --window-stride-steps 1 --history-steps 30 --horizon-steps 1 --seed 2026 --epochs 2 --patience 2 --learning-rate 0.001 --batch-size 128 --num-workers 4 --tqdm 1 --revin 1 --visualize-after-train 0 --satra-use-spatial-relation 1 --satra-use-dtw-prior 0 --dataset-name "$DATASET_NAME" --pretrain --pretrain-epochs 5 --pretrain-learning-rate 0.0001 --pretrain-lr-scheduler cosine_hard_restarts --mae-mask-ratio 0.3 --pretrain-patience 3 --run-dir "$RUN_DIR" --result-name "$RESULT_NAME"
uv run python plot_multi_turbine_15min_forecast.py --checkpoint-root "$RUN_ROOT" --checkpoint-history-steps 30 --horizon-minutes 1 --issue-interval-minutes 1 --evaluation-start "$EVALUATION_START" --evaluation-end "$EVALUATION_END" --plot-start "$PLOT_START" --batch-size 128 --device cuda:0 --output-dir "$PLOT_OUTPUT"
