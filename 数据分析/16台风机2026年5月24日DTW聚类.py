from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


# 固定5月24日DTW距离、原始10秒功率与聚类产物路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_ROOT = Path(__file__).resolve().parent
PROCESSED_ROOT = PROJECT_ROOT / "dataset" / "processed"
COLUMNS_PATH = PROCESSED_ROOT / "columns.json"
DISTANCE_PATH = ANALYSIS_ROOT / "16台风机2026年5月24日DTW距离矩阵.csv"
CLUSTER_PATH = ANALYSIS_ROOT / "16台风机2026年5月24日DTW聚类结果.csv"
EVALUATION_PATH = ANALYSIS_ROOT / "16台风机2026年5月24日DTW候选簇数评估.csv"
HEATMAP_PATH = ANALYSIS_ROOT / "16台风机2026年5月24日DTW聚类热力图.png"
HEATMAP_PDF_PATH = ANALYSIS_ROOT / "16台风机2026年5月24日DTW聚类热力图.pdf"
CURVES_PATH = ANALYSIS_ROOT / "16台风机2026年5月24日DTW聚类功率曲线.png"
CURVES_PDF_PATH = ANALYSIS_ROOT / "16台风机2026年5月24日DTW聚类功率曲线.pdf"
FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
TURBINE_COUNT = 16
SAMPLE_SECONDS = 10
DAY_SAMPLE_COUNT = 24 * 60 * 60 // SAMPLE_SECONDS
HIGH_SIMILARITY_DISTANCE = 0.30
DAY_START = np.datetime64("2026-05-24T00:00:00", "s").astype(np.int64)
DAY_END = np.datetime64("2026-05-25T00:00:00", "s").astype(np.int64)


def main():
    # 读取已计算的DTW距离矩阵
    distance_frame = pd.read_csv(DISTANCE_PATH, index_col=0)
    distance_matrix = distance_frame.to_numpy(dtype=np.float64)
    turbine_labels = distance_frame.index.to_numpy(dtype=object)

    # 评估2至8个粗粒度聚类的轮廓系数
    candidate_cluster_counts = np.arange(2, 9)
    silhouette_scores = np.empty(len(candidate_cluster_counts), dtype=np.float64)
    for candidate_index in range(len(candidate_cluster_counts)):
        candidate_cluster_count = candidate_cluster_counts[candidate_index]
        candidate_labels = AgglomerativeClustering(
            n_clusters=candidate_cluster_count,
            metric="precomputed",
            linkage="average",
        ).fit_predict(distance_matrix)
        silhouette_scores[candidate_index] = silhouette_score(
            distance_matrix,
            candidate_labels,
            metric="precomputed",
        )
    best_candidate_index = np.argmax(silhouette_scores)
    coarse_cluster_count = candidate_cluster_counts[best_candidate_index]
    evaluation_values = np.column_stack(
        (candidate_cluster_counts, silhouette_scores)
    )
    evaluation_frame = pd.DataFrame(
        evaluation_values,
        columns=["候选簇数", "轮廓系数"],
    )
    evaluation_frame["候选簇数"] = evaluation_frame["候选簇数"].astype(int)
    evaluation_frame.to_csv(EVALUATION_PATH, index=False, encoding="utf-8-sig")

    # 以完全链接保证同簇内任意两台均满足高相似阈值
    condensed_distance = squareform(distance_matrix, checks=True)
    linkage_matrix = linkage(condensed_distance, method="complete")
    raw_cluster_labels = fcluster(
        linkage_matrix,
        t=HIGH_SIMILARITY_DISTANCE,
        criterion="distance",
    )
    raw_cluster_ids = np.unique(raw_cluster_labels)
    raw_cluster_sizes = np.empty(len(raw_cluster_ids), dtype=np.int64)
    raw_cluster_first_turbines = np.empty(len(raw_cluster_ids), dtype=np.int64)
    for raw_cluster_index in range(len(raw_cluster_ids)):
        raw_cluster_id = raw_cluster_ids[raw_cluster_index]
        member_indices = np.flatnonzero(raw_cluster_labels == raw_cluster_id)
        raw_cluster_sizes[raw_cluster_index] = len(member_indices)
        raw_cluster_first_turbines[raw_cluster_index] = member_indices[0]

    # 多风机簇优先排列，簇内按风机编号排序
    singleton_flags = raw_cluster_sizes == 1
    raw_cluster_order = np.lexsort(
        (raw_cluster_first_turbines, singleton_flags)
    )
    ordered_raw_cluster_ids = raw_cluster_ids[raw_cluster_order]
    cluster_labels = np.empty(TURBINE_COUNT, dtype=np.int64)
    for cluster_index in range(len(ordered_raw_cluster_ids)):
        raw_cluster_id = ordered_raw_cluster_ids[cluster_index]
        cluster_labels[raw_cluster_labels == raw_cluster_id] = cluster_index + 1
    cluster_count = len(ordered_raw_cluster_ids)
    display_order = np.argsort(cluster_labels, kind="stable")

    # 保存每台风机的簇归属和簇内数量
    turbine_cluster_sizes = np.empty(TURBINE_COUNT, dtype=np.int64)
    for cluster_id in range(1, cluster_count + 1):
        cluster_members = cluster_labels == cluster_id
        turbine_cluster_sizes[cluster_members] = np.sum(cluster_members)
    cluster_frame = pd.DataFrame(
        {
            "风机": turbine_labels,
            "簇": cluster_labels,
            "簇内风机数": turbine_cluster_sizes,
        }
    )
    cluster_frame = cluster_frame.sort_values(["簇", "风机"]).reset_index(drop=True)
    cluster_frame.to_csv(CLUSTER_PATH, index=False, encoding="utf-8-sig")

    # 设置聚类图的中文字体与配色
    title_font = font_manager.FontProperties(fname=FONT_PATH, size=21)
    axis_font = font_manager.FontProperties(fname=FONT_PATH, size=15)
    panel_font = font_manager.FontProperties(fname=FONT_PATH, size=13)
    tick_font = font_manager.FontProperties(fname=FONT_PATH, size=11)
    legend_font = font_manager.FontProperties(fname=FONT_PATH, size=10)
    cluster_colors = plt.cm.tab10(np.linspace(0, 1, cluster_count))
    turbine_colors = plt.cm.turbo(np.linspace(0.04, 0.96, TURBINE_COUNT))

    # 按簇重排并标注DTW距离热力图
    ordered_distances = distance_matrix[np.ix_(display_order, display_order)]
    ordered_turbines = turbine_labels[display_order]
    ordered_clusters = cluster_labels[display_order]
    grouped_labels = np.empty(TURBINE_COUNT, dtype=object)
    for ordered_index in range(TURBINE_COUNT):
        grouped_labels[ordered_index] = (
            f"簇{ordered_clusters[ordered_index]} | {ordered_turbines[ordered_index]}"
        )
    heatmap_figure, heatmap_axis = plt.subplots(figsize=(14, 12))
    heatmap_image = heatmap_axis.imshow(
        ordered_distances,
        cmap="YlOrRd",
        vmin=0,
        vmax=np.max(distance_matrix),
        interpolation="nearest",
        aspect="equal",
    )
    heatmap_axis.set_xticks(np.arange(TURBINE_COUNT))
    heatmap_axis.set_yticks(np.arange(TURBINE_COUNT))
    heatmap_axis.set_xticklabels(
        grouped_labels,
        rotation=50,
        ha="right",
        rotation_mode="anchor",
        fontproperties=tick_font,
    )
    heatmap_axis.set_yticklabels(grouped_labels, fontproperties=tick_font)
    heatmap_axis.set_xlabel("按聚类结果重排的风机", fontproperties=axis_font, labelpad=10)
    heatmap_axis.set_ylabel("按聚类结果重排的风机", fontproperties=axis_font, labelpad=10)
    heatmap_axis.set_title(
        f"2026年5月24日16台风机DTW聚类热力图（{cluster_count}簇）\n"
        f"完全链接层次聚类｜高相似距离阈值 < {HIGH_SIMILARITY_DISTANCE:.2f}｜"
        "每台8640个原始10秒点",
        fontproperties=title_font,
        pad=18,
    )

    # 标注距离并用分隔线划分各簇
    annotation_boundary = np.max(distance_matrix) * 0.58
    for row_index in range(TURBINE_COUNT):
        for column_index in range(TURBINE_COUNT):
            distance = ordered_distances[row_index, column_index]
            text_color = "white"
            if distance < annotation_boundary:
                text_color = "#202020"
            heatmap_axis.text(
                column_index,
                row_index,
                f"{distance:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=9.2,
            )
    cluster_boundaries = np.flatnonzero(np.diff(ordered_clusters)) + 0.5
    for boundary in cluster_boundaries:
        heatmap_axis.axhline(boundary, color="#202020", linewidth=2.0)
        heatmap_axis.axvline(boundary, color="#202020", linewidth=2.0)

    # 用簇颜色区分坐标标签并导出图片
    x_tick_labels = heatmap_axis.get_xticklabels()
    y_tick_labels = heatmap_axis.get_yticklabels()
    for ordered_index in range(TURBINE_COUNT):
        cluster_color = cluster_colors[ordered_clusters[ordered_index] - 1]
        x_tick_labels[ordered_index].set_color(cluster_color)
        y_tick_labels[ordered_index].set_color(cluster_color)
    color_bar = heatmap_figure.colorbar(
        heatmap_image,
        ax=heatmap_axis,
        fraction=0.046,
        pad=0.04,
    )
    color_bar.set_label(
        "长度归一化DTW距离（越小越相似）",
        fontproperties=axis_font,
        labelpad=12,
    )
    heatmap_figure.tight_layout()
    heatmap_figure.savefig(
        HEATMAP_PATH,
        dpi=360,
        bbox_inches="tight",
        facecolor="white",
    )
    heatmap_figure.savefig(
        HEATMAP_PDF_PATH,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(heatmap_figure)

    # 读取与DTW一致的全天标准化功率曲线
    columns = json.loads(COLUMNS_PATH.read_text(encoding="utf-8"))
    power_index = columns.index("风机-P") + 1
    standardized_power = np.empty(
        (TURBINE_COUNT, DAY_SAMPLE_COUNT),
        dtype=np.float64,
    )
    timestamps = np.empty(DAY_SAMPLE_COUNT, dtype="datetime64[s]")
    for turbine_index in range(TURBINE_COUNT):
        turbine_id = turbine_index + 1
        data_path = PROCESSED_ROOT / f"turbine_{turbine_id:02d}.npy"
        data = np.load(data_path, mmap_mode="r")
        start_index = np.searchsorted(data[:, 0], DAY_START)
        end_index = np.searchsorted(data[:, 0], DAY_END)
        power_values = np.asarray(data[start_index:end_index, power_index])
        power_mean = np.mean(power_values)
        power_standard_deviation = np.std(power_values)
        standardized_power[turbine_index] = (
            power_values - power_mean
        ) / power_standard_deviation
        timestamps[:] = data[start_index:end_index, 0].astype("datetime64[s]")

    # 每个簇单独展示全量10秒功率曲线与簇均值
    plt.rcParams["path.simplify"] = False
    plt.rcParams["agg.path.chunksize"] = 10000
    curve_figure, curve_axes = plt.subplots(
        4,
        2,
        figsize=(24, 20),
        sharex=True,
        sharey=True,
    )
    curve_axes = curve_axes.ravel()
    for cluster_index in range(cluster_count):
        cluster_id = cluster_index + 1
        axis = curve_axes[cluster_index]
        member_indices = np.flatnonzero(cluster_labels == cluster_id)
        for member_position in range(len(member_indices)):
            turbine_index = member_indices[member_position]
            axis.plot(
                timestamps,
                standardized_power[turbine_index],
                color=turbine_colors[turbine_index],
                linewidth=0.55,
                alpha=0.78,
                rasterized=True,
                label=turbine_labels[turbine_index],
            )
        if len(member_indices) > 1:
            cluster_mean = np.mean(standardized_power[member_indices], axis=0)
            axis.plot(
                timestamps,
                cluster_mean,
                color="#111111",
                linewidth=1.2,
                alpha=0.9,
                rasterized=True,
                label="簇均值",
            )
        member_text = "、".join(turbine_labels[member_indices].tolist())
        axis.set_title(
            f"簇 {cluster_id}（{len(member_indices)}台）：{member_text}",
            fontproperties=panel_font,
            color=cluster_colors[cluster_index],
            pad=8,
        )
        axis.xaxis.set_major_locator(mdates.HourLocator(interval=4))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        axis.grid(color="#B8B8B8", linewidth=0.5, alpha=0.55)
        axis.tick_params(axis="both", labelsize=10)
        axis.legend(prop=legend_font, loc="upper left", ncol=2)
        if cluster_index % 2 == 0:
            axis.set_ylabel("z-score标准化功率", fontproperties=axis_font)
        if cluster_index >= 5:
            axis.set_xlabel("时刻", fontproperties=axis_font)
    curve_axes[-1].axis("off")

    # 标注聚类口径并导出分簇曲线图
    curve_figure.suptitle(
        f"2026年5月24日16台风机DTW聚类功率曲线（{cluster_count}簇）\n"
        f"同簇内任意两台DTW距离 < {HIGH_SIMILARITY_DISTANCE:.2f}｜"
        "每台8640个原始10秒点，无降采样",
        fontproperties=title_font,
        y=0.995,
    )
    curve_figure.tight_layout(rect=(0, 0, 1, 0.955))
    curve_figure.savefig(
        CURVES_PATH,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    curve_figure.savefig(
        CURVES_PDF_PATH,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(curve_figure)

    # 输出簇数选择与产物路径
    print(f"轮廓系数最佳粗粒度簇数: {coarse_cluster_count}")
    print(
        f"高相似阈值分组: {cluster_count}簇 "
        f"(完全链接, DTW < {HIGH_SIMILARITY_DISTANCE:.2f})"
    )
    for cluster_id in range(1, cluster_count + 1):
        member_indices = np.flatnonzero(cluster_labels == cluster_id)
        member_text = "、".join(turbine_labels[member_indices].tolist())
        print(f"簇{cluster_id}: {member_text}")
    print(f"聚类结果: {CLUSTER_PATH.relative_to(PROJECT_ROOT)}")
    print(f"候选簇数评估: {EVALUATION_PATH.relative_to(PROJECT_ROOT)}")
    print(f"聚类热力图: {HEATMAP_PATH.relative_to(PROJECT_ROOT)}")
    print(f"聚类热力图PDF: {HEATMAP_PDF_PATH.relative_to(PROJECT_ROOT)}")
    print(f"分簇功率曲线: {CURVES_PATH.relative_to(PROJECT_ROOT)}")
    print(f"分簇功率曲线PDF: {CURVES_PDF_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
