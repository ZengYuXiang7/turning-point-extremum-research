from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager


# 固定风机1数据与功率分布产物路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "dataset" / "processed" / "turbine_01.npy"
COLUMNS_PATH = PROJECT_ROOT / "dataset" / "processed" / "columns.json"
PDF_PATH = ANALYSIS_ROOT / "风机1功率分布.pdf"
STATISTICS_PATH = ANALYSIS_ROOT / "风机1功率分布统计.csv"
FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")


def main():
    # 加载全时段风机功率序列
    data = np.load(DATA_PATH, mmap_mode="r")
    columns = json.loads(COLUMNS_PATH.read_text(encoding="utf-8"))
    power_index = columns.index("风机-P")
    power_values = data[:, power_index + 1]
    timestamps = pd.to_datetime(data[:, 0].astype(np.int64), unit="s")

    # 计算功率分位数与运行状态统计
    percentiles = np.percentile(power_values, [1, 5, 25, 50, 75, 95, 99])
    zero_count = int(np.sum(power_values == 0.0))
    negative_count = int(np.sum(power_values < 0.0))
    statistic_names = [
        "样本数",
        "最小值 (kW)",
        "1% 分位数 (kW)",
        "5% 分位数 (kW)",
        "25% 分位数 (kW)",
        "中位数 (kW)",
        "均值 (kW)",
        "75% 分位数 (kW)",
        "95% 分位数 (kW)",
        "99% 分位数 (kW)",
        "最大值 (kW)",
        "标准差 (kW)",
        "零功率样本数",
        "零功率比例 (%)",
        "负功率样本数",
        "负功率比例 (%)",
    ]
    statistic_values = [
        len(power_values),
        np.min(power_values),
        percentiles[0],
        percentiles[1],
        percentiles[2],
        percentiles[3],
        np.mean(power_values),
        percentiles[4],
        percentiles[5],
        percentiles[6],
        np.max(power_values),
        np.std(power_values),
        zero_count,
        zero_count * 100 / len(power_values),
        negative_count,
        negative_count * 100 / len(power_values),
    ]
    statistics_frame = pd.DataFrame({"统计量": statistic_names, "数值": statistic_values})
    statistics_frame.to_csv(STATISTICS_PATH, index=False, encoding="utf-8-sig")

    # 设置中文字体与分布绘图参数
    title_font = font_manager.FontProperties(fname=FONT_PATH, size=22)
    axis_font = font_manager.FontProperties(fname=FONT_PATH, size=16)
    legend_font = font_manager.FontProperties(fname=FONT_PATH, size=13)
    bin_edges = np.histogram_bin_edges(power_values, bins="fd")
    sorted_power_values = np.sort(power_values)
    cumulative_percent = np.arange(1, len(power_values) + 1) * 100 / len(power_values)

    # 绘制频数直方图与经验累积分布
    figure, (histogram_axis, cumulative_axis) = plt.subplots(1, 2, figsize=(22, 8))
    histogram_axis.hist(
        power_values,
        bins=bin_edges,
        color="#4C78A8",
        edgecolor="white",
        linewidth=0.7,
    )
    histogram_axis.axvline(
        percentiles[3], color="#F58518", linewidth=2.2, label="中位数"
    )
    histogram_axis.axvline(
        np.mean(power_values), color="#E45756", linewidth=2.2, label="均值"
    )
    histogram_axis.set_title("功率频数分布", fontproperties=axis_font, pad=14)
    histogram_axis.set_xlabel("风机功率 (kW)", fontproperties=axis_font, labelpad=10)
    histogram_axis.set_ylabel("样本数", fontproperties=axis_font, labelpad=10)
    histogram_axis.tick_params(axis="both", labelsize=13)
    histogram_axis.legend(prop=legend_font)

    cumulative_axis.step(
        sorted_power_values,
        cumulative_percent,
        where="post",
        color="#54A24B",
        linewidth=2.2,
    )
    cumulative_axis.axhline(50, color="#F58518", linewidth=1.8, linestyle="--")
    cumulative_axis.axvline(
        percentiles[3], color="#F58518", linewidth=1.8, linestyle="--"
    )
    cumulative_axis.set_title("功率经验累积分布", fontproperties=axis_font, pad=14)
    cumulative_axis.set_xlabel("风机功率 (kW)", fontproperties=axis_font, labelpad=10)
    cumulative_axis.set_ylabel("累计样本比例 (%)", fontproperties=axis_font, labelpad=10)
    cumulative_axis.tick_params(axis="both", labelsize=13)
    cumulative_axis.set_ylim(0, 100)

    # 标注全样本时间范围并导出图件
    start_time = timestamps[0].strftime("%Y-%m-%d %H:%M:%S")
    end_time = timestamps[-1].strftime("%Y-%m-%d %H:%M:%S")
    figure.suptitle(f"风机 1 功率数据分布（全量 {len(power_values):,} 个样本）\n" f"{start_time} 至 {end_time}", fontproperties=title_font, y=1.02,)
    figure.tight_layout()
    figure.savefig(PDF_PATH, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    # 输出功率分布产物与核心统计量
    print(f"样本数: {len(power_values):,}")
    print(f"均值: {np.mean(power_values):.2f} kW")
    print(f"中位数: {percentiles[3]:.2f} kW")
    print(f"零功率比例: {zero_count * 100 / len(power_values):.2f}%")
    print(f"PDF: {PDF_PATH.relative_to(PROJECT_ROOT)}")
    print(f"统计表: {STATISTICS_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
