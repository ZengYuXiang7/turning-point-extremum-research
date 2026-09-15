from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dtaidistance import dtw
from matplotlib import font_manager


# 固定5月24日16台风机原始10秒功率与DTW产物路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_ROOT = Path(__file__).resolve().parent
PROCESSED_ROOT = PROJECT_ROOT / "dataset" / "processed"
COLUMNS_PATH = PROCESSED_ROOT / "columns.json"
FIGURE_PATH = ANALYSIS_ROOT / "16台风机2026年5月24日DTW距离热力图.png"
PDF_PATH = ANALYSIS_ROOT / "16台风机2026年5月24日DTW距离热力图.pdf"
DISTANCE_PATH = ANALYSIS_ROOT / "16台风机2026年5月24日DTW距离矩阵.csv"
PAIR_PATH = ANALYSIS_ROOT / "16台风机2026年5月24日DTW风机对排名.csv"
FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
TURBINE_COUNT = 16
SAMPLE_SECONDS = 10
DAY_SAMPLE_COUNT = 24 * 60 * 60 // SAMPLE_SECONDS
MAX_WARP_MINUTES = 30
MAX_WARP_SAMPLES = MAX_WARP_MINUTES * 60 // SAMPLE_SECONDS
DAY_START = np.datetime64("2026-05-24T00:00:00", "s").astype(np.int64)
DAY_END = np.datetime64("2026-05-25T00:00:00", "s").astype(np.int64)


def main():
    # 加载16台风机全天8640个原始功率点
    columns = json.loads(COLUMNS_PATH.read_text(encoding="utf-8"))
    power_index = columns.index("风机-P") + 1
    power_series = np.empty((TURBINE_COUNT, DAY_SAMPLE_COUNT), dtype=np.float64)
    turbine_labels = np.empty(TURBINE_COUNT, dtype=object)

    for turbine_index in range(TURBINE_COUNT):
        turbine_id = turbine_index + 1
        data_path = PROCESSED_ROOT / f"turbine_{turbine_id:02d}.npy"
        data = np.load(data_path, mmap_mode="r")
        start_index = np.searchsorted(data[:, 0], DAY_START)
        end_index = np.searchsorted(data[:, 0], DAY_END)
        power_values = np.asarray(data[start_index:end_index, power_index])
        power_mean = np.mean(power_values)
        power_standard_deviation = np.std(power_values)
        power_series[turbine_index] = (
            power_values - power_mean
        ) / power_standard_deviation
        turbine_labels[turbine_index] = f"风机{turbine_id:02d}"

    # 在30分钟Sakoe-Chiba窗口内计算全量标准化功率DTW距离
    distance_matrix = dtw.distance_matrix_fast(power_series, compact=False, parallel=True, use_pruning=True, only_triu=False, window=MAX_WARP_SAMPLES + 1,)
    normalized_distance_matrix = distance_matrix / np.sqrt(DAY_SAMPLE_COUNT)
    distance_frame = pd.DataFrame(normalized_distance_matrix, index=turbine_labels, columns=turbine_labels,)
    distance_frame.to_csv(DISTANCE_PATH, encoding="utf-8-sig", float_format="%.6f")

    # 按距离由小到大保存120个不重复风机对
    upper_rows, upper_columns = np.triu_indices(TURBINE_COUNT, k=1)
    pair_frame = pd.DataFrame({ "风机A": turbine_labels[upper_rows], "风机B": turbine_labels[upper_columns], "长度归一化DTW距离": normalized_distance_matrix[ upper_rows, upper_columns ], })
    pair_frame = pair_frame.sort_values("长度归一化DTW距离").reset_index(drop=True)
    pair_frame.to_csv(PAIR_PATH, index=False, encoding="utf-8-sig", float_format="%.6f")

    # 绘制紧凑的全对称距离热力图
    title_font = font_manager.FontProperties(fname=FONT_PATH, size=21)
    axis_font = font_manager.FontProperties(fname=FONT_PATH, size=15)
    tick_font = font_manager.FontProperties(fname=FONT_PATH, size=12)
    note_font = font_manager.FontProperties(fname=FONT_PATH, size=11)
    figure, axis = plt.subplots(figsize=(13.5, 11.5))
    image = axis.imshow(normalized_distance_matrix, cmap="YlOrRd", vmin=0, vmax=np.max(normalized_distance_matrix), interpolation="nearest", aspect="equal",)

    axis.set_xticks(np.arange(TURBINE_COUNT))
    axis.set_yticks(np.arange(TURBINE_COUNT))
    axis.set_xticklabels(turbine_labels, rotation=45, ha="right", rotation_mode="anchor", fontproperties=tick_font,)
    axis.set_yticklabels(turbine_labels, fontproperties=tick_font)
    axis.set_xlabel("风机编号", fontproperties=axis_font, labelpad=10)
    axis.set_ylabel("风机编号", fontproperties=axis_font, labelpad=10)
    axis.set_title("16台风机2026年5月24日功率DTW距离\n" "全天原始10秒粒度（每台8640点，无降采样）｜" "单台z-score标准化｜最大时间错位30分钟", fontproperties=title_font, pad=18,)

    # 标注每个风机对的长度归一化距离
    annotation_boundary = np.max(normalized_distance_matrix) * 0.58
    for row_index in range(TURBINE_COUNT):
        for column_index in range(TURBINE_COUNT):
            distance = normalized_distance_matrix[row_index, column_index]
            text_color = "white"
            if distance < annotation_boundary:
                text_color = "#202020"
            axis.text(column_index, row_index, f"{distance:.2f}", ha="center", va="center", color=text_color, fontsize=9.5,)

    # 添加距离标尺并导出高清产物
    color_bar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    color_bar.set_label("长度归一化DTW距离（越小越相似）", fontproperties=axis_font, labelpad=12,)
    for tick_label in color_bar.ax.get_yticklabels():
        tick_label.set_fontproperties(note_font)

    figure.tight_layout()
    figure.savefig(FIGURE_PATH, dpi=360, bbox_inches="tight", facecolor="white")
    figure.savefig(PDF_PATH, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    # 输出数据口径、最相似风机对与产物路径
    print("时间范围: 2026-05-24 00:00:00 至 2026-05-24 23:59:50")
    print(f"每台样本数: {DAY_SAMPLE_COUNT:,}（原始{SAMPLE_SECONDS}秒粒度）")
    print(f"DTW最大时间错位: {MAX_WARP_MINUTES}分钟")
    print("最相似的5个风机对:")
    print(pair_frame.head(5).to_string(index=False))
    print(f"距离矩阵: {DISTANCE_PATH.relative_to(PROJECT_ROOT)}")
    print(f"风机对排名: {PAIR_PATH.relative_to(PROJECT_ROOT)}")
    print(f"热力图: {FIGURE_PATH.relative_to(PROJECT_ROOT)}")
    print(f"PDF: {PDF_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
