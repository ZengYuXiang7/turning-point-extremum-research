from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.lightgbm_feature_data import POWER_COLUMN, TURBINE_COUNT


def grouped_permutation_importance(model, values, targets, turbine_ids,
                                   history_steps, feature_count, repeats, seed):
    # 按风机分别置换一个字段的全部历史位置
    baseline_prediction = model.predict(values, num_iteration=model.best_iteration_)
    baseline_error = baseline_prediction - targets
    baseline_rmse = np.sqrt(np.mean(baseline_error**2))
    repeated_importance = np.empty((feature_count, repeats), dtype=np.float64)
    turbine_repeated = np.empty(
        (feature_count, repeats, TURBINE_COUNT), dtype=np.float64,
    )
    turbine_baseline_rmse = np.empty(TURBINE_COUNT, dtype=np.float64)
    for turbine_index in range(TURBINE_COUNT):
        turbine_id = turbine_index + 1
        turbine_error = baseline_error[turbine_ids == turbine_id]
        turbine_baseline_rmse[turbine_index] = np.sqrt(np.mean(turbine_error**2))
    random_generator = np.random.default_rng(seed)

    # 每次置换保持风机内分布和全部历史位置的联合关系
    for feature_index in tqdm(range(feature_count), desc="分组置换重要性", unit="字段"):
        group_columns = np.arange(
            feature_index, history_steps * feature_count, feature_count, dtype=np.int64,
        )
        for repeat_index in range(repeats):
            permuted_values = values.copy()
            for turbine_id in range(1, TURBINE_COUNT + 1):
                turbine_indices = np.flatnonzero(turbine_ids == turbine_id)
                row_order = random_generator.permutation(turbine_indices)
                destination = np.ix_(turbine_indices, group_columns)
                source = np.ix_(row_order, group_columns)
                permuted_values[destination] = values[source]
            prediction = model.predict(
                permuted_values, num_iteration=model.best_iteration_,
            )
            error = prediction - targets
            permuted_rmse = np.sqrt(np.mean(error**2))
            repeated_importance[feature_index, repeat_index] = permuted_rmse - baseline_rmse
            for turbine_index in range(TURBINE_COUNT):
                turbine_id = turbine_index + 1
                turbine_error = error[turbine_ids == turbine_id]
                turbine_rmse = np.sqrt(np.mean(turbine_error**2))
                turbine_repeated[feature_index, repeat_index, turbine_index] = (
                    turbine_rmse - turbine_baseline_rmse[turbine_index]
                )

    mean_importance = np.mean(repeated_importance, axis=1)
    std_importance = np.std(repeated_importance, axis=1)
    turbine_mean = np.mean(turbine_repeated, axis=1)
    turbine_std = np.std(turbine_repeated, axis=1)
    return mean_importance, std_importance, turbine_mean, turbine_std


def create_importance_tables(model, values, targets, turbine_ids, columns,
                             lag_names, lag_points, raw_names, args):
    # 提取每个历史字段的增益和原生SHAP重要性
    gain = model.booster_.feature_importance(
        importance_type="gain", iteration=model.best_iteration_,
    ).astype(np.float64)
    shap_contributions = model.booster_.predict(
        values, num_iteration=model.best_iteration_, pred_contrib=True,
    )
    mean_absolute_shap = np.mean(np.abs(shap_contributions[:, :-1]), axis=0)
    gain_share = gain / np.sum(gain)
    shap_share = mean_absolute_shap / np.sum(mean_absolute_shap)
    lag_frame = pd.DataFrame({
        "滞后特征": lag_names,
        "原始字段": raw_names,
        "距目标点数": lag_points,
        "Gain": gain,
        "Gain占比": gain_share,
        "平均绝对SHAP_kW": mean_absolute_shap,
        "SHAP占比": shap_share,
    })
    lag_frame = lag_frame.sort_values("Gain", ascending=False).reset_index(drop=True)

    # 聚合历史位置并计算字段级分组置换重要性
    feature_count = len(columns)
    grouped_gain = np.sum(gain.reshape(args.history_steps, feature_count), axis=0)
    grouped_shap = np.sum(
        mean_absolute_shap.reshape(args.history_steps, feature_count), axis=0,
    )
    permutation = grouped_permutation_importance(
        model, values, targets, turbine_ids, args.history_steps, feature_count,
        args.permutation_repeats, args.seed,
    )
    permutation_mean, permutation_std = permutation[:2]
    turbine_permutation_mean, turbine_permutation_std = permutation[2:]
    turbine_mean = np.mean(turbine_permutation_mean, axis=1)
    turbine_std = np.std(turbine_permutation_mean, axis=1)
    positive_turbine_count = np.sum(turbine_permutation_mean > 0.0, axis=1)
    all_feature_frame = pd.DataFrame({
        "字段": columns,
        "Gain": grouped_gain,
        "Gain占比": grouped_gain / np.sum(grouped_gain),
        "平均绝对SHAP_kW": grouped_shap,
        "SHAP占比": grouped_shap / np.sum(grouped_shap),
        "总体置换RMSE增量_kW": permutation_mean,
        "总体置换RMSE增量标准差_kW": permutation_std,
        "跨风机平均置换RMSE增量_kW": turbine_mean,
        "风机间置换RMSE增量标准差_kW": turbine_std,
        "置换后RMSE上升风机数": positive_turbine_count,
    })

    # 历史功率固定保留，其余55个字段参加筛选排名
    predictor_frame = all_feature_frame[all_feature_frame["字段"] != POWER_COLUMN].copy()
    predictor_frame["Gain名次"] = predictor_frame["Gain"].rank(
        method="min", ascending=False,
    )
    predictor_frame["SHAP名次"] = predictor_frame["平均绝对SHAP_kW"].rank(
        method="min", ascending=False,
    )
    predictor_frame["置换名次"] = predictor_frame[
        "跨风机平均置换RMSE增量_kW"
    ].rank(method="min", ascending=False)
    rank_columns = ["Gain名次", "SHAP名次", "置换名次"]
    predictor_frame["综合平均名次"] = predictor_frame[rank_columns].mean(axis=1)
    predictor_frame = predictor_frame.sort_values(
        ["综合平均名次", "跨风机平均置换RMSE增量_kW", "Gain"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    predictor_frame.insert(0, "综合排名", np.arange(1, len(predictor_frame) + 1))

    # 展开候选字段在每台风机上的置换稳定性
    turbine_rows = []
    for predictor_index in range(len(predictor_frame)):
        feature_name = predictor_frame.loc[predictor_index, "字段"]
        feature_index = columns.index(feature_name)
        for turbine_index in range(TURBINE_COUNT):
            turbine_rows.append({
                "综合排名": predictor_index + 1,
                "字段": feature_name,
                "风机编号": turbine_index + 1,
                "置换RMSE增量均值_kW": turbine_permutation_mean[
                    feature_index, turbine_index,
                ],
                "置换RMSE增量标准差_kW": turbine_permutation_std[
                    feature_index, turbine_index,
                ],
            })
    turbine_frame = pd.DataFrame(turbine_rows)
    return lag_frame, all_feature_frame, predictor_frame, turbine_frame
