from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from tqdm import tqdm

from utils.lightgbm_feature_data import POWER_COLUMN, TURBINE_COUNT


STRICT_ACC_MIN_POWER_KW = 100.0
STRICT_ACC_TOLERANCE = 0.30


def point_metrics(prediction, truth):
    # 计算与项目正式评估一致的点预测指标
    error = prediction.astype(np.float64) - truth.astype(np.float64)
    valid_mask = truth > STRICT_ACC_MIN_POWER_KW
    valid_error = np.abs(error[valid_mask])
    valid_truth = truth[valid_mask]
    passed = valid_error <= STRICT_ACC_TOLERANCE * valid_truth
    relative_error = valid_error / valid_truth
    metrics = {
        "strict_acc30": float(np.mean(passed) * 100.0),
        "mae_kw": float(np.mean(np.abs(error))),
        "rmse_kw": float(np.sqrt(np.mean(error**2))),
        "mape": float(np.mean(relative_error) * 100.0),
    }
    return metrics


def fit_lightgbm(train_values, train_targets, valid_values, valid_targets, args):
    # 用时间验证段早停训练回归模型
    model = LGBMRegressor(objective="regression_l2", n_estimators=args.num_boost_round, learning_rate=args.learning_rate, num_leaves=args.num_leaves, min_child_samples=args.min_child_samples, subsample=1.0, colsample_bytree=1.0, reg_alpha=args.reg_alpha, reg_lambda=args.reg_lambda, random_state=args.seed, n_jobs=args.num_workers, importance_type="gain", verbosity=-1,)
    stopping = early_stopping(stopping_rounds=args.early_stopping_rounds, first_metric_only=True, verbose=False,)
    logging = log_evaluation(period=0)
    callbacks = [stopping, logging]
    model.fit(train_values, train_targets, eval_set=[(valid_values, valid_targets)], eval_metric="rmse", callbacks=callbacks,)
    return model


def expand_group_columns(selected_indices, history_steps, feature_count):
    # 展开所选字段在全部历史位置对应的矩阵列
    selected_count = len(selected_indices)
    group_columns = np.empty(history_steps * selected_count, dtype=np.int64)
    output_index = 0

    for history_index in range(history_steps):
        right = output_index + selected_count
        group_columns[output_index:right] = history_index * feature_count + selected_indices
        output_index = right

    return group_columns


def evaluate_top_k(train_values, train_targets, valid_values, valid_targets, valid_ids, columns, predictor_frame, args):
    # 固定保留历史功率并重训各个Top-K预测字段子集
    ranked_names = predictor_frame["字段"].to_numpy()
    feature_count = len(columns)
    ranked_indices = np.empty(len(ranked_names), dtype=np.int64)
    for rank_index in range(len(ranked_names)):
        ranked_indices[rank_index] = columns.index(ranked_names[rank_index])
    power_index = columns.index(POWER_COLUMN)

    metric_rows = []
    turbine_metric_rows = []
    for top_k in tqdm(args.top_k_values, desc="Top-K验证", unit="配置"):
        selected_indices = np.empty(top_k + 1, dtype=np.int64)
        selected_indices[:top_k] = ranked_indices[:top_k]
        selected_indices[-1] = power_index
        group_columns = expand_group_columns(selected_indices, args.history_steps, feature_count,)
        selected_train = train_values[:, group_columns]
        selected_valid = valid_values[:, group_columns]
        model = fit_lightgbm(selected_train, train_targets, selected_valid, valid_targets, args,)
        prediction = model.predict(selected_valid, num_iteration=model.best_iteration_)
        overall_metrics = point_metrics(prediction, valid_targets)

        # 计算16台风机等权指标
        turbine_rmse = np.empty(TURBINE_COUNT, dtype=np.float64)
        turbine_mae = np.empty(TURBINE_COUNT, dtype=np.float64)
        turbine_acc30 = np.empty(TURBINE_COUNT, dtype=np.float64)
        turbine_mape = np.empty(TURBINE_COUNT, dtype=np.float64)
        for turbine_index in range(TURBINE_COUNT):
            turbine_id = turbine_index + 1
            turbine_mask = valid_ids == turbine_id
            turbine_metrics = point_metrics(prediction[turbine_mask], valid_targets[turbine_mask],)
            turbine_rmse[turbine_index] = turbine_metrics["rmse_kw"]
            turbine_mae[turbine_index] = turbine_metrics["mae_kw"]
            turbine_acc30[turbine_index] = turbine_metrics["strict_acc30"]
            turbine_mape[turbine_index] = turbine_metrics["mape"]
            turbine_metric_rows.append({ "Top-K预测字段数": top_k, "风机编号": turbine_id, "验证RMSE_kW": turbine_metrics["rmse_kw"], "验证MAE_kW": turbine_metrics["mae_kw"], "验证Acc30_percent": turbine_metrics["strict_acc30"], "验证MAPE_percent": turbine_metrics["mape"], })

        metric_rows.append({ "Top-K预测字段数": top_k, "实际输入字段数": top_k + 1, "展开输入维数": len(group_columns), "最佳迭代轮数": model.best_iteration_, "总体验证RMSE_kW": overall_metrics["rmse_kw"], "总体验证MAE_kW": overall_metrics["mae_kw"], "总体验证Acc30_percent": overall_metrics["strict_acc30"], "总体验证MAPE_percent": overall_metrics["mape"], "Macro16验证RMSE_kW": float(np.mean(turbine_rmse)), "Macro16验证MAE_kW": float(np.mean(turbine_mae)), "Macro16验证Acc30_percent": float(np.mean(turbine_acc30)), "Macro16验证MAPE_percent": float(np.mean(turbine_mape)), "最大单机验证RMSE_kW": float(np.max(turbine_rmse)), })

    top_k_frame = pd.DataFrame(metric_rows)
    turbine_top_k_frame = pd.DataFrame(turbine_metric_rows)
    best_row_index = top_k_frame["Macro16验证RMSE_kW"].idxmin()
    best_top_k = int(top_k_frame.loc[best_row_index, "Top-K预测字段数"])
    return top_k_frame, turbine_top_k_frame, best_top_k
