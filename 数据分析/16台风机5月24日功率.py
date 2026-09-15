from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


# 固定16台风机5月24日原始10秒序列与单页PDF路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_ROOT = Path(__file__).resolve().parent
PROCESSED_ROOT = PROJECT_ROOT / "dataset" / "processed"
COLUMNS_PATH = PROCESSED_ROOT / "columns.json"
PDF_PATH = ANALYSIS_ROOT / "16台风机2026年5月24日功率.pdf"
FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
TURBINE_COUNT = 16
DAY_START = np.datetime64("2026-05-24T00:00:00", "s").astype(np.int64)
DAY_END = np.datetime64("2026-05-25T00:00:00", "s").astype(np.int64)


def load_day_bounds(power_index):
    # 扫描所选日原始功率范围，确保16个面板共用纵轴。
    minimum_values = np.empty(TURBINE_COUNT, dtype=np.float64)
    maximum_values = np.empty(TURBINE_COUNT, dtype=np.float64)

    for turbine_index in range(TURBINE_COUNT):
        turbine_id = turbine_index + 1
        data_path = PROCESSED_ROOT / f"turbine_{turbine_id:02d}.npy"
        data = np.load(data_path, mmap_mode="r")
        start_index = np.searchsorted(data[:, 0], DAY_START)
        end_index = np.searchsorted(data[:, 0], DAY_END)
        power_values = data[start_index:end_index, power_index]
        minimum_values[turbine_index] = np.min(power_values)
        maximum_values[turbine_index] = np.max(power_values)

    return minimum_values, maximum_values


def main():
    # 读取所选日原始10秒功率列及其全局范围。
    columns = json.loads(COLUMNS_PATH.read_text(encoding="utf-8"))
    power_index = columns.index("风机-P") + 1
    minimum_values, maximum_values = load_day_bounds(power_index)
    lower_bound = np.floor(np.min(minimum_values) / 100) * 100
    upper_bound = np.ceil(np.max(maximum_values) / 100) * 100

    # 保留每个原始10秒样本，关闭Matplotlib路径简化。
    plt.rcParams["path.simplify"] = False
    plt.rcParams["agg.path.chunksize"] = 10000
    title_font = font_manager.FontProperties(fname=FONT_PATH, size=24)
    panel_font = font_manager.FontProperties(fname=FONT_PATH, size=14)
    label_font = font_manager.FontProperties(fname=FONT_PATH, size=12)
    figure, axes = plt.subplots(4, 4, figsize=(32, 18), sharex=True, sharey=True)
    axes = axes.ravel()

    # 逐台绘制5月24日全天的原始细粒度功率曲线。
    for turbine_index in range(TURBINE_COUNT):
        turbine_id = turbine_index + 1
        data_path = PROCESSED_ROOT / f"turbine_{turbine_id:02d}.npy"
        data = np.load(data_path, mmap_mode="r")
        start_index = np.searchsorted(data[:, 0], DAY_START)
        end_index = np.searchsorted(data[:, 0], DAY_END)
        timestamps = data[start_index:end_index, 0].astype("datetime64[s]")
        power_values = data[start_index:end_index, power_index]
        axis = axes[turbine_index]

        axis.plot(
            timestamps,
            power_values,
            color="#1F77B4",
            linewidth=0.28,
            rasterized=True,
            solid_capstyle="butt",
        )
        axis.set_title(f"风机 {turbine_id:02d}", fontproperties=panel_font, pad=7)
        axis.set_xlim(timestamps[0], timestamps[-1])
        axis.set_ylim(lower_bound, upper_bound)
        axis.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        axis.grid(axis="y", color="#B8B8B8", linewidth=0.5, alpha=0.6)
        axis.tick_params(axis="both", labelsize=10)

        if turbine_index % 4 == 0:
            axis.set_ylabel("功率 (kW)", fontproperties=label_font)

        if turbine_index >= 12:
            axis.set_xlabel("时刻", fontproperties=label_font)

    # 标注所选日数据口径并导出单页PDF。
    figure.suptitle(f"16台风机2026年5月24日功率（原始10秒粒度，无降采样/聚合）\n" f"每台 {len(power_values):,} 个样本；统一纵轴 {lower_bound:.0f}–{upper_bound:.0f} kW", fontproperties=title_font, y=0.995,)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(PDF_PATH, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    # 输出PDF路径和数据口径。
    print(f"PDF: {PDF_PATH.relative_to(PROJECT_ROOT)}")
    print("时间范围: 2026-05-24 00:00:00 至 2026-05-24 23:59:50")
    print(f"每台样本数: {len(power_values):,}")
    print(f"共享纵轴: {lower_bound:.0f} 至 {upper_bound:.0f} kW")


if __name__ == "__main__":
    main()
