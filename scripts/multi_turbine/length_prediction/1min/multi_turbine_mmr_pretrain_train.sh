#!/bin/bash
set -e

# 16台风机联合面板 MultiTurbine 的遮蔽市场重建热身后长度预测；1min一个模型点，空间关系与DTW先验关闭。
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
unset LD_LIBRARY_PATH

POINT_INTERVAL_SECONDS=60
DATASET_NAME=GuangningWindPower16Turbine1min
RUN_ROOT=".runs/WeatherComparison/MultiTurbine/MultiTurbine/MSE/$DATASET_NAME/pretrain_e10_m0p3_no_spatial"
EVALUATION_START="2026-08-01 09:45:00"
EVALUATION_END="2026-09-01 09:45:00"
PLOT_START="2026-08-01 09:45:00"

# 
COMMON_ARGS="--mode train --model MultiTurbine --scenario MultiTurbine --loss MSE --point-interval-seconds $POINT_INTERVAL_SECONDS --window-stride-steps 1 --seed 2026  --epochs 2 --patience 2 --learning-rate 0.001 --batch-size 128 --num-workers 4 --tqdm 1 --revin 1 --visualize-after-train 0 --satra-use-spatial-relation 1 --satra-use-dtw-prior 0 --dataset-name $DATASET_NAME --pretrain --pretrain-epochs 5 --pretrain-learning-rate 0.0001 --pretrain-lr-scheduler cosine_hard_restarts --mae-mask-ratio 0.3 --pretrain-patience 3"

# 各预测长度完成训练后，立即按15分钟发布节奏评估并绘图。
uv run python run_multi.py $COMMON_ARGS --history-steps 90 --horizon-steps 15 --run-dir "$RUN_ROOT/h240_p15_s1_seed2026" --result-name multi_turbine_mmr_e10_m0p3_no_spatial_h240_p15_s1_seed2026
uv run python plot_multi_turbine_15min_forecast.py --checkpoint-root "$RUN_ROOT" --horizon-minutes 15 --evaluation-start "$EVALUATION_START" --evaluation-end "$EVALUATION_END" --plot-start "$PLOT_START" --device cuda:0
uv run python run_multi.py $COMMON_ARGS --history-steps 90 --horizon-steps 30 --run-dir "$RUN_ROOT/h240_p30_s1_seed2026" --result-name multi_turbine_mmr_e10_m0p3_no_spatial_h240_p30_s1_seed2026
uv run python plot_multi_turbine_15min_forecast.py --checkpoint-root "$RUN_ROOT" --horizon-minutes 30 --evaluation-start "$EVALUATION_START" --evaluation-end "$EVALUATION_END" --plot-start "$PLOT_START" --device cuda:0
uv run python run_multi.py $COMMON_ARGS --history-steps 135 --horizon-steps 45 --run-dir "$RUN_ROOT/h240_p45_s1_seed2026" --result-name multi_turbine_mmr_e10_m0p3_no_spatial_h240_p45_s1_seed2026
uv run python plot_multi_turbine_15min_forecast.py --checkpoint-root "$RUN_ROOT" --horizon-minutes 45 --evaluation-start "$EVALUATION_START" --evaluation-end "$EVALUATION_END" --plot-start "$PLOT_START" --device cuda:0
uv run python run_multi.py $COMMON_ARGS --history-steps 180 --horizon-steps 60 --run-dir "$RUN_ROOT/h240_p60_s1_seed2026" --result-name multi_turbine_mmr_e10_m0p3_no_spatial_h240_p60_s1_seed2026
uv run python plot_multi_turbine_15min_forecast.py --checkpoint-root "$RUN_ROOT" --horizon-minutes 60 --evaluation-start "$EVALUATION_START" --evaluation-end "$EVALUATION_END" --plot-start "$PLOT_START" --device cuda:0
uv run python run_multi.py $COMMON_ARGS --history-steps 360 --horizon-steps 120 --run-dir "$RUN_ROOT/h240_p120_s1_seed2026" --result-name multi_turbine_mmr_e10_m0p3_no_spatial_h240_p120_s1_seed2026
uv run python plot_multi_turbine_15min_forecast.py --checkpoint-root "$RUN_ROOT" --horizon-minutes 120 --evaluation-start "$EVALUATION_START" --evaluation-end "$EVALUATION_END" --plot-start "$PLOT_START" --device cuda:0
uv run python run_multi.py $COMMON_ARGS --history-steps 540 --horizon-steps 180 --run-dir "$RUN_ROOT/h240_p180_s1_seed2026" --result-name multi_turbine_mmr_e10_m0p3_no_spatial_h240_p180_s1_seed2026
uv run python plot_multi_turbine_15min_forecast.py --checkpoint-root "$RUN_ROOT" --horizon-minutes 180 --evaluation-start "$EVALUATION_START" --evaluation-end "$EVALUATION_END" --plot-start "$PLOT_START" --device cuda:0
uv run python run_multi.py $COMMON_ARGS --history-steps 720 --horizon-steps 240 --run-dir "$RUN_ROOT/h240_p240_s1_seed2026" --result-name multi_turbine_mmr_e10_m0p3_no_spatial_h240_p240_s1_seed2026
uv run python plot_multi_turbine_15min_forecast.py --checkpoint-root "$RUN_ROOT" --horizon-minutes 240 --evaluation-start "$EVALUATION_START" --evaluation-end "$EVALUATION_END" --plot-start "$PLOT_START" --device cuda:0
uv run python run_multi.py $COMMON_ARGS --history-steps 720 --horizon-steps 480 --run-dir "$RUN_ROOT/h240_p480_s1_seed2026" --result-name multi_turbine_mmr_e10_m0p3_no_spatial_h240_p480_s1_seed2026
uv run python plot_multi_turbine_15min_forecast.py --checkpoint-root "$RUN_ROOT" --horizon-minutes 480 --evaluation-start "$EVALUATION_START" --evaluation-end "$EVALUATION_END" --plot-start "$PLOT_START" --device cuda:0
uv run python run_multi.py $COMMON_ARGS --history-steps 720 --horizon-steps 720 --run-dir "$RUN_ROOT/h240_p720_s1_seed2026" --result-name multi_turbine_mmr_e10_m0p3_no_spatial_h240_p720_s1_seed2026
uv run python plot_multi_turbine_15min_forecast.py --checkpoint-root "$RUN_ROOT" --horizon-minutes 720 --evaluation-start "$EVALUATION_START" --evaluation-end "$EVALUATION_END" --plot-start "$PLOT_START" --device cuda:0
uv run python run_multi.py $COMMON_ARGS --history-steps 1440 --horizon-steps 1440 --run-dir "$RUN_ROOT/h240_p1440_s1_seed2026" --result-name multi_turbine_mmr_e10_m0p3_no_spatial_h240_p1440_s1_seed2026
uv run python plot_multi_turbine_15min_forecast.py --checkpoint-root "$RUN_ROOT" --horizon-minutes 1440 --evaluation-start "$EVALUATION_START" --evaluation-end "$EVALUATION_END" --plot-start "$PLOT_START" --device cuda:0
