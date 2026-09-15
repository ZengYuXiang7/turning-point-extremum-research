from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager


# 固定风机1数据与分析产物路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "dataset" / "processed" / "turbine_01.npy"
COLUMNS_PATH = PROJECT_ROOT / "dataset" / "processed" / "columns.json"
HEATMAP_PATH = ANALYSIS_ROOT / "风机1特征相关性热力图.png"
PDF_PATH = ANALYSIS_ROOT / "风机1特征相关性热力图.pdf"
CORRELATION_PATH = ANALYSIS_ROOT / "风机1特征相关性矩阵.csv"
FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")


def main():
    # 加载时间戳后的56个原始特征
    data = np.load(DATA_PATH, mmap_mode="r")
    columns = json.loads(COLUMNS_PATH.read_text(encoding="utf-8"))
    feature_values = data[:, 1:]
    feature_frame = pd.DataFrame(feature_values, columns=columns)

    # 计算并保存全样本Pearson相关矩阵
    correlation_frame = feature_frame.corr(method="pearson")
    correlation_frame.to_csv(CORRELATION_PATH, encoding="utf-8-sig")
    correlation_values = correlation_frame.to_numpy(dtype=np.float64, copy=True)
    masked_correlations = np.ma.masked_invalid(correlation_values)
    constant_columns = correlation_frame.columns[ correlation_frame.isna().all(axis=0) ].tolist()

    # 使用服务器现有中文字体绘制全部特征
    label_font = font_manager.FontProperties(fname=FONT_PATH, size=10)
    title_font = font_manager.FontProperties(fname=FONT_PATH, size=24)
    axis_font = font_manager.FontProperties(fname=FONT_PATH, size=16)
    note_font = font_manager.FontProperties(fname=FONT_PATH, size=14)
    color_map = plt.get_cmap("RdBu_r").with_extremes(bad="#d9d9d9")

    # 灰色单元格保留零方差特征的未定义相关系数
    figure, axis = plt.subplots(figsize=(36, 32))
    image = axis.pcolormesh(masked_correlations, cmap=color_map, vmin=-1.0, vmax=1.0, shading="nearest",)
    axis.set_aspect("equal")
    tick_positions = np.arange(len(columns))
    axis.set_xticks(tick_positions)
    axis.set_yticks(tick_positions)
    axis.set_xticklabels(columns, rotation=55, ha="right", rotation_mode="anchor", fontproperties=label_font,)
    axis.set_yticklabels(columns, fontproperties=label_font)
    axis.tick_params(axis="both", which="both", length=0)
    axis.set_xlabel("特征", fontproperties=axis_font, labelpad=18)
    axis.set_ylabel("特征", fontproperties=axis_font, labelpad=18)
    constant_text = "灰色表示相关系数未定义；零方差特征：" + "、".join(constant_columns)
    axis.set_title(f"风机 1 特征 Pearson 相关性热力图（全部 {len(feature_frame):,} 个样本）\n" f"{constant_text}", fontproperties=title_font, pad=22,)

    # 添加相关系数色标
    color_bar = figure.colorbar(image, ax=axis, fraction=0.035, pad=0.025)
    color_bar.set_label("Pearson 相关系数", fontproperties=axis_font, labelpad=12)
    for tick_label in color_bar.ax.get_yticklabels():
        tick_label.set_fontproperties(note_font)
    figure.subplots_adjust(left=0.37, bottom=0.31, right=0.92, top=0.94)
    figure.savefig(HEATMAP_PATH, dpi=220, bbox_inches="tight", facecolor="white")
    figure.savefig(PDF_PATH, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    # 输出本次分析规模和产物位置
    print(f"样本数: {len(feature_frame):,}")
    print(f"特征数: {len(columns)}")
    print(f"零方差特征: {'、'.join(constant_columns)}")
    print(f"热力图: {HEATMAP_PATH.relative_to(PROJECT_ROOT)}")
    print(f"PDF: {PDF_PATH.relative_to(PROJECT_ROOT)}")
    print(f"相关矩阵: {CORRELATION_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
