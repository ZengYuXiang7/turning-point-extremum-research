from __future__ import annotations

import argparse
import json
import time

from utils.lightgbm_feature_data import *
from utils.lightgbm_feature_importance import create_importance_tables
from utils.lightgbm_feature_reporting import write_feature_artifacts
from utils.lightgbm_feature_training import evaluate_top_k, fit_lightgbm


def parse_args():
    # 声明可复现的筛选参数
    parser = argparse.ArgumentParser(
        description="用LightGBM从16台风机筛选20个预测字段",
    )
    parser.add_argument("--point-interval-seconds", type=int, default=900)
    parser.add_argument("--history-steps", type=int, default=4)
    parser.add_argument("--forecast-step", type=int, default=1)
    parser.add_argument("--window-stride-steps", type=int, default=1)
    parser.add_argument(
        "--top-k-values", type=int, nargs="+",
        default=[1, 5, 10, 15, 20, 30, 40, 55],
    )
    parser.add_argument("--importance-samples-per-turbine", type=int, default=320)
    parser.add_argument("--permutation-repeats", type=int, default=3)
    parser.add_argument("--num-boost-round", type=int, default=1000)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--min-child-samples", type=int, default=50)
    parser.add_argument("--reg-alpha", type=float, default=0.0)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main():
    # 构造16台风机的训练与验证窗口
    started = time.perf_counter()
    args = parse_args()
    columns = json.loads(COLUMNS_PATH.read_text(encoding="utf-8"))
    result_name = (
        f"p{args.point_interval_seconds}_h{args.history_steps}_"
        f"f{args.forecast_step}_ws{args.window_stride_steps}_seed{args.seed}_"
        "top20_plus_power"
    )
    output_root = OUTPUT_BASE / result_name
    output_root.mkdir(parents=True, exist_ok=True)
    samples = load_all_turbine_samples(columns, args)
    train_values, train_targets, train_times = samples[:3]
    valid_values, valid_targets, valid_ids, valid_times = samples[3:]
    lag_names, lag_points, raw_names = build_lag_feature_names(
        columns, args.history_steps, args.forecast_step, args.point_interval_seconds,
    )
    print(
        f"风机数={TURBINE_COUNT}，训练样本={len(train_targets):,}，"
        f"验证样本={len(valid_targets):,}，输入维数={train_values.shape[1]}",
        flush=True,
    )

    # 训练全字段模型并计算预测字段排名
    full_model = fit_lightgbm(
        train_values, train_targets, valid_values, valid_targets, args,
    )
    importance_sample = select_balanced_importance_sample(
        valid_values, valid_targets, valid_ids, args,
    )
    importance_values, importance_targets, importance_ids = importance_sample
    importance_tables = create_importance_tables(
        full_model, importance_values, importance_targets, importance_ids,
        columns, lag_names, lag_points, raw_names, args,
    )
    lag_frame, all_feature_frame, predictor_frame, turbine_frame = importance_tables

    # 固定保留历史功率并比较预测字段Top-K
    top_k_result = evaluate_top_k(
        train_values, train_targets, valid_values, valid_targets,
        valid_ids, columns, predictor_frame, args,
    )
    top_k_frame, turbine_top_k_frame, best_top_k = top_k_result
    elapsed_seconds = time.perf_counter() - started
    top20_path = write_feature_artifacts(
        lag_frame, all_feature_frame, predictor_frame, turbine_frame,
        top_k_frame, turbine_top_k_frame, best_top_k, train_times, valid_times,
        len(importance_targets), elapsed_seconds, args, output_root,
    )
    print(f"Macro16验证RMSE最优预测字段数: Top-{best_top_k}")
    print(f"DLinear固定Top20特征表: {top20_path.relative_to(PROJECT_ROOT)}")
    print(f"输出目录: {output_root.relative_to(PROJECT_ROOT)}")
    print(f"总耗时: {elapsed_seconds:.1f}秒")


if __name__ == "__main__":
    main()
