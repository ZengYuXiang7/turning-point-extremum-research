from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

from utils.lightgbm_feature_data import POWER_COLUMN, PROJECT_ROOT, TURBINE_COUNT


FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
TOP_PREDICTOR_COUNT = 20


def plot_feature_importance(predictor_frame, output_root):
    # 绘制固定Top20预测字段的三类重要性
    display_frame = predictor_frame.head(TOP_PREDICTOR_COUNT).sort_values("综合排名", ascending=False,)
    labels = display_frame["字段"].to_numpy()
    label_font = font_manager.FontProperties(fname=FONT_PATH, size=10)
    title_font = font_manager.FontProperties(fname=FONT_PATH, size=13)
    axis_font = font_manager.FontProperties(fname=FONT_PATH, size=11)
    figure, axes = plt.subplots(1, 3, figsize=(24, 11), sharey=True)

    axes[0].barh(labels, display_frame["Gain占比"] * 100, color="#4C78A8")
    axes[0].set_title("LightGBM Gain", fontproperties=title_font)
    axes[0].set_xlabel("全字段Gain占比 (%)", fontproperties=axis_font)
    axes[1].barh(labels, display_frame["SHAP占比"] * 100, color="#F58518")
    axes[1].set_title("原生SHAP", fontproperties=title_font)
    axes[1].set_xlabel("全字段绝对SHAP占比 (%)", fontproperties=axis_font)
    axes[2].barh(
        labels,
        display_frame["跨风机平均置换RMSE增量_kW"],
        xerr=display_frame["风机间置换RMSE增量标准差_kW"],
        color="#54A24B",
    )
    axes[2].axvline(0.0, color="#666666", linewidth=1)
    axes[2].set_title("16台风机分组置换", fontproperties=title_font)
    axes[2].set_xlabel("跨风机平均RMSE增量 (kW)", fontproperties=axis_font)

    for axis in axes:
        axis.tick_params(axis="x", labelsize=10)
        for tick_label in axis.get_yticklabels():
            tick_label.set_fontproperties(label_font)
    figure.suptitle("16台风机 LightGBM Top20预测字段", fontproperties=font_manager.FontProperties(fname=FONT_PATH, size=18),)
    figure.tight_layout()
    pdf_path = output_root / "feature_importance_top20.pdf"
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_top_k_curve(top_k_frame, output_root):
    # 绘制预测字段数量变化下的Macro16验证指标
    title_font = font_manager.FontProperties(fname=FONT_PATH, size=17)
    axis_font = font_manager.FontProperties(fname=FONT_PATH, size=12)
    legend_font = font_manager.FontProperties(fname=FONT_PATH, size=11)
    figure, error_axis = plt.subplots(figsize=(11, 7))
    accuracy_axis = error_axis.twinx()
    top_k_values = top_k_frame["Top-K预测字段数"].to_numpy()
    error_axis.plot(
        top_k_values, top_k_frame["Macro16验证RMSE_kW"], marker="o",
        linewidth=2, label="RMSE", color="#4C78A8",
    )
    error_axis.plot(
        top_k_values, top_k_frame["Macro16验证MAE_kW"], marker="s",
        linewidth=2, label="MAE", color="#F58518",
    )
    accuracy_axis.plot(
        top_k_values, top_k_frame["Macro16验证Acc30_percent"], marker="^",
        linewidth=2, label="Acc30", color="#54A24B",
    )
    error_axis.set_xlabel("保留预测字段数 Top-K", fontproperties=axis_font)
    error_axis.set_ylabel("Macro16误差 (kW)", fontproperties=axis_font)
    accuracy_axis.set_ylabel("Macro16 Acc30 (%)", fontproperties=axis_font)
    error_axis.set_xticks(top_k_values)
    error_axis.grid(alpha=0.25)
    error_axis.set_title("LightGBM Top-K预测字段验证曲线", fontproperties=title_font)
    error_handles, error_labels = error_axis.get_legend_handles_labels()
    accuracy_handles, accuracy_labels = accuracy_axis.get_legend_handles_labels()
    handles = error_handles + accuracy_handles
    labels = error_labels + accuracy_labels
    error_axis.legend(handles, labels, prop=legend_font, loc="best")
    figure.tight_layout()
    figure.savefig(output_root / "top_k_validation_curve.pdf", bbox_inches="tight", facecolor="white",)
    plt.close(figure)


def write_summary(predictor_frame, top_k_frame, best_top_k, train_times, valid_times, sample_count, elapsed_seconds, args, output_root):
    # 写出筛选协议、Top-K结果和固定Top20
    train_start = pd.to_datetime(np.min(train_times), unit="s")
    train_end = pd.to_datetime(np.max(train_times), unit="s")
    valid_start = pd.to_datetime(np.min(valid_times), unit="s")
    valid_end = pd.to_datetime(np.max(valid_times), unit="s")
    forecast_minutes = args.forecast_step * args.point_interval_seconds / 60
    history_minutes = args.history_steps * args.point_interval_seconds / 60
    lines = [
        "# 16台风机 LightGBM特征筛选", "", "## 协议", "",
        f"- 数据：`dataset/processed/turbine_01.npy` 至 `turbine_{TURBINE_COUNT:02d}.npy`",
        f"- 历史：{args.history_steps}点（{history_minutes:g}分钟）",
        f"- 目标：最后历史点后第{args.forecast_step}点（{forecast_minutes:g}分钟）功率",
        f"- 训练：{train_start} 至 {train_end}，共{len(train_times):,}个样本",
        f"- 验证：{valid_start} 至 {valid_end}，共{len(valid_times):,}个样本",
        f"- 重要性样本：每台{args.importance_samples_per_turbine}个，共{sample_count:,}个",
        "- 历史功率固定保留，不参加Top-K预测字段排名",
        "- Top20产物包含20个预测字段加历史功率，共21个输入字段",
        "- 20%测试段未参与训练、重要性计算或Top-K选择", "",
        "## Top-K验证", "",
        "| 预测字段数 | 实际字段数 | 输入维数 | 最佳轮数 | 总体RMSE | Macro16 RMSE | Macro16 Acc30 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row_index in range(len(top_k_frame)):
        row = top_k_frame.iloc[row_index]
        lines.append(f"| {int(row['Top-K预测字段数'])} | {int(row['实际输入字段数'])} | " f"{int(row['展开输入维数'])} | {int(row['最佳迭代轮数'])} | " f"{row['总体验证RMSE_kW']:.4f} | {row['Macro16验证RMSE_kW']:.4f} | " f"{row['Macro16验证Acc30_percent']:.4f} |")

    # 固定交付排名前20个预测字段
    lines.extend(["", f"Macro16验证RMSE最优为Top-{best_top_k}。", "", "## 固定Top20", "",
                  "| 排名 | 字段 | Gain占比 | SHAP占比 | 正增量风机数 |",
                  "|---:|---|---:|---:|---:|"])
    top20_frame = predictor_frame.head(TOP_PREDICTOR_COUNT)
    for row_index in range(len(top20_frame)):
        row = top20_frame.iloc[row_index]
        lines.append(f"| {int(row['综合排名'])} | {row['字段']} | {row['Gain占比']:.6f} | " f"{row['SHAP占比']:.6f} | {int(row['置换后RMSE上升风机数'])} |")
    lines.extend([
        "", "## 解释边界", "",
        "- 同一Top20将交给15分钟DLinear长度实验；该列表由1小时历史预测下一15分钟任务选出。",
        "- 高相关字段会分摊树模型重要性，DLinear结果仍需与raw56及corr21分别比较。",
        f"- 本次运行耗时：{elapsed_seconds:.1f}秒。", "",
    ])
    summary_path = output_root / "summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def write_feature_artifacts(lag_frame, all_feature_frame, predictor_frame, turbine_frame, top_k_frame, turbine_top_k_frame, best_top_k, train_times, valid_times, sample_count, elapsed_seconds, args, output_root):
    # 保存分析表、固定Top20列表、图件与总结
    lag_frame.to_csv(output_root / "lag_feature_importance.csv", index=False, encoding="utf-8-sig")
    all_feature_frame.to_csv(output_root / "all_feature_importance.csv", index=False, encoding="utf-8-sig")
    predictor_frame.to_csv(output_root / "predictor_feature_ranking.csv", index=False, encoding="utf-8-sig")
    turbine_frame.to_csv(output_root / "per_turbine_permutation_importance.csv", index=False, encoding="utf-8-sig")
    top_k_frame.to_csv(output_root / "top_k_validation_metrics.csv", index=False, encoding="utf-8-sig")
    turbine_top_k_frame.to_csv(output_root / "top_k_validation_metrics_by_turbine.csv", index=False, encoding="utf-8-sig")
    predictor_frame.head(best_top_k).to_csv(output_root / "validation_best_predictors.csv", index=False, encoding="utf-8-sig",)
    top20_frame = predictor_frame.head(TOP_PREDICTOR_COUNT).copy()
    top20_frame.to_csv(output_root / "top20_predictors.csv", index=False, encoding="utf-8-sig")
    top20_features = top20_frame["字段"].tolist()
    top20_features.append(POWER_COLUMN)
    top20_path = output_root / "top20_features.json"
    top20_path.write_text(json.dumps(top20_features, ensure_ascii=False, indent=2), encoding="utf-8",)
    plot_feature_importance(predictor_frame, output_root)
    plot_top_k_curve(top_k_frame, output_root)
    write_summary(predictor_frame, top_k_frame, best_top_k, train_times, valid_times, sample_count, elapsed_seconds, args, output_root,)
    return top20_path
