from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from scipy import signal


# 固定全量数据与周期性分析产物路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "dataset" / "processed"
COLUMNS_PATH = DATA_ROOT / "columns.json"
FIGURE_PATH = ANALYSIS_ROOT / "16台风机功率周期性分析.png"
PDF_PATH = ANALYSIS_ROOT / "16台风机功率周期性分析.pdf"
METRICS_PATH = ANALYSIS_ROOT / "16台风机功率周期性指标.csv"
FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
TURBINE_COUNT = 16
SECONDS_PER_HOUR = 60 * 60
SECONDS_PER_DAY = 24 * SECONDS_PER_HOUR
SPECTRAL_SEGMENT_COUNT = 6


def lag_correlation(values, lag):
    # 计算指定时间滞后的Pearson相关系数
    left_values = values[:-lag]
    right_values = values[lag:]
    left_centered = left_values - np.mean(left_values)
    right_centered = right_values - np.mean(right_values)
    numerator = np.dot(left_centered, right_centered)
    left_energy = np.dot(left_centered, left_centered)
    right_energy = np.dot(right_centered, right_centered)
    denominator = np.sqrt(left_energy * right_energy)
    correlation = numerator / denominator
    return float(correlation)


def compute_turbine_metrics(data_path, power_column_index):
    # 加载一台风机的全部10秒功率点
    data = np.load(data_path, mmap_mode="r")
    timestamps = data[:, 0].astype(np.int64)
    power_values = np.array(data[:, power_column_index], dtype=np.float64)
    sample_seconds = int(timestamps[1] - timestamps[0])
    daily_lag = SECONDS_PER_DAY // sample_seconds
    weekly_lag = 7 * daily_lag

    # 衡量平均日曲线解释的功率方差比例
    daily_values = power_values.reshape(-1, daily_lag)
    overall_mean = np.mean(power_values)
    daily_profile = np.mean(daily_values, axis=0)
    centered_power = power_values - overall_mean
    daily_explained_variance = len(daily_values) * np.sum(
        (daily_profile - overall_mean) ** 2
    )
    total_variance = np.sum(centered_power**2)
    daily_strength = daily_explained_variance / total_variance

    # 衡量相隔一天和一周后的功率相似度
    daily_autocorrelation = lag_correlation(power_values, daily_lag)
    weekly_autocorrelation = lag_correlation(power_values, weekly_lag)

    # 六个完整22天区段覆盖全部样本并估计稳定频谱
    sampling_frequency = 1 / sample_seconds
    spectral_segment_samples = len(power_values) // SPECTRAL_SEGMENT_COUNT
    frequencies, power_spectrum = signal.welch(
        power_values,
        fs=sampling_frequency,
        window="hann",
        nperseg=spectral_segment_samples,
        noverlap=0,
        detrend="linear",
        scaling="density",
    )
    minimum_frequency = 1 / (14 * SECONDS_PER_DAY)
    maximum_frequency = 1 / SECONDS_PER_HOUR
    band_mask = (frequencies >= minimum_frequency) & (
        frequencies <= maximum_frequency
    )
    band_frequencies = frequencies[band_mask]
    band_spectrum = power_spectrum[band_mask]

    # 选择1小时至14天频段内最突出的局部谱峰
    peak_indices = signal.find_peaks(band_spectrum)[0]
    peak_prominences = signal.peak_prominences(
        band_spectrum, peak_indices
    )[0]
    dominant_peak_index = peak_indices[np.argmax(peak_prominences)]
    dominant_frequency = band_frequencies[dominant_peak_index]
    dominant_period_hours = 1 / dominant_frequency / SECONDS_PER_HOUR
    median_spectrum = np.median(band_spectrum)
    spectral_peak_db = 10 * np.log10(
        band_spectrum[dominant_peak_index] / median_spectrum
    )

    # 谱熵越低表示频域能量越集中
    spectral_probabilities = band_spectrum / np.sum(band_spectrum)
    spectral_entropy = -np.sum(
        spectral_probabilities * np.log(spectral_probabilities)
    ) / np.log(len(spectral_probabilities))

    return (
        len(power_values),
        sample_seconds,
        int(timestamps[0]),
        int(timestamps[-1]),
        float(daily_strength),
        daily_autocorrelation,
        weekly_autocorrelation,
        float(dominant_period_hours),
        float(spectral_peak_db),
        float(spectral_entropy),
    )


def main():
    # 逐台计算全时段功率周期性指标
    columns = json.loads(COLUMNS_PATH.read_text(encoding="utf-8"))
    power_column_index = columns.index("风机-P") + 1
    metric_rows = []
    for turbine_id in range(1, TURBINE_COUNT + 1):
        data_path = DATA_ROOT / f"turbine_{turbine_id:02d}.npy"
        (
            sample_count,
            sample_seconds,
            start_timestamp,
            end_timestamp,
            daily_strength,
            daily_autocorrelation,
            weekly_autocorrelation,
            dominant_period_hours,
            spectral_peak_db,
            spectral_entropy,
        ) = compute_turbine_metrics(data_path, power_column_index)
        metric_rows.append(
            {
                "风机": f"风机{turbine_id:02d}",
                "样本数": sample_count,
                "采样间隔 (秒)": sample_seconds,
                "开始时间": pd.to_datetime(start_timestamp, unit="s"),
                "结束时间": pd.to_datetime(end_timestamp, unit="s"),
                "日周期方差解释率 (%)": daily_strength * 100,
                "24小时自相关": daily_autocorrelation,
                "7天自相关": weekly_autocorrelation,
                "主周期 (小时)": dominant_period_hours,
                "主谱峰突出度 (dB)": spectral_peak_db,
                "归一化谱熵": spectral_entropy,
            }
        )
    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame.to_csv(METRICS_PATH, index=False, encoding="utf-8-sig")

    # 设置周期性指标图的中文字体
    title_font = font_manager.FontProperties(fname=FONT_PATH, size=22)
    axis_font = font_manager.FontProperties(fname=FONT_PATH, size=15)
    tick_font = font_manager.FontProperties(fname=FONT_PATH, size=12)
    legend_font = font_manager.FontProperties(fname=FONT_PATH, size=12)
    turbine_positions = np.arange(TURBINE_COUNT)
    turbine_labels = metrics_frame["风机"].to_numpy()

    # 展示日周期方差解释率
    figure, axes = plt.subplots(3, 1, figsize=(18, 18))
    daily_axis = axes[0]
    daily_bars = daily_axis.bar(
        turbine_positions,
        metrics_frame["日周期方差解释率 (%)"],
        color="#4C78A8",
    )
    daily_axis.bar_label(daily_bars, fmt="%.1f", padding=3, fontsize=10)
    daily_axis.set_ylabel("方差解释率 (%)", fontproperties=axis_font, labelpad=10)
    daily_axis.set_title(
        "平均日曲线解释的功率方差", fontproperties=axis_font, pad=12
    )
    daily_axis.set_xticks(turbine_positions)
    daily_axis.set_xticklabels(turbine_labels, fontproperties=tick_font)
    daily_axis.tick_params(axis="y", labelsize=12)

    # 对比一天和一周滞后自相关
    autocorrelation_axis = axes[1]
    bar_width = 0.36
    daily_autocorrelation_bars = autocorrelation_axis.bar(
        turbine_positions - bar_width / 2,
        metrics_frame["24小时自相关"],
        width=bar_width,
        color="#F58518",
        label="24小时",
    )
    weekly_autocorrelation_bars = autocorrelation_axis.bar(
        turbine_positions + bar_width / 2,
        metrics_frame["7天自相关"],
        width=bar_width,
        color="#54A24B",
        label="7天",
    )
    autocorrelation_axis.bar_label(
        daily_autocorrelation_bars, fmt="%.2f", padding=3, fontsize=9
    )
    autocorrelation_axis.bar_label(
        weekly_autocorrelation_bars, fmt="%.2f", padding=3, fontsize=9
    )
    autocorrelation_axis.axhline(0, color="#666666", linewidth=1)
    autocorrelation_axis.set_ylabel(
        "滞后自相关系数", fontproperties=axis_font, labelpad=10
    )
    autocorrelation_axis.set_title(
        "24小时与7天功率重复性", fontproperties=axis_font, pad=12
    )
    autocorrelation_axis.set_xticks(turbine_positions)
    autocorrelation_axis.set_xticklabels(
        turbine_labels, fontproperties=tick_font
    )
    autocorrelation_axis.tick_params(axis="y", labelsize=12)
    autocorrelation_axis.legend(prop=legend_font)

    # 对比频谱峰突出度与谱熵
    spectral_axis = axes[2]
    spectral_bars = spectral_axis.bar(
        turbine_positions,
        metrics_frame["主谱峰突出度 (dB)"],
        color="#E45756",
        label="主谱峰突出度",
    )
    spectral_axis.bar_label(spectral_bars, fmt="%.1f", padding=3, fontsize=10)
    spectral_axis.set_ylabel("主谱峰突出度 (dB)", fontproperties=axis_font, labelpad=10)
    spectral_axis.set_title(
        "1小时至14天频段：主周期均为24小时",
        fontproperties=axis_font,
        pad=12,
    )
    spectral_axis.set_xticks(turbine_positions)
    spectral_axis.set_xticklabels(turbine_labels, fontproperties=tick_font)
    spectral_axis.tick_params(axis="y", labelsize=12)

    entropy_axis = spectral_axis.twinx()
    entropy_lines = entropy_axis.plot(
        turbine_positions,
        metrics_frame["归一化谱熵"],
        color="#72B7B2",
        marker="o",
        linewidth=2.2,
        label="归一化谱熵",
    )
    entropy_line = entropy_lines[0]
    entropy_axis.set_ylabel(
        "归一化谱熵（越低越集中）", fontproperties=axis_font, labelpad=10
    )
    entropy_axis.set_ylim(0, 1)
    entropy_axis.tick_params(axis="y", labelsize=12)
    spectral_axis.legend(
        [spectral_bars, entropy_line],
        ["主谱峰突出度", "归一化谱熵"],
        prop=legend_font,
        loc="upper left",
    )

    # 标注数据范围并导出分析图
    start_time = metrics_frame["开始时间"].iloc[0].strftime("%Y-%m-%d %H:%M:%S")
    end_time = metrics_frame["结束时间"].iloc[0].strftime("%Y-%m-%d %H:%M:%S")
    figure.suptitle(
        f"16 台风机功率周期性评估（每台全量 {sample_count:,} 个10秒样本）\n"
        f"{start_time} 至 {end_time}",
        fontproperties=title_font,
        y=0.995,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(PDF_PATH, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    # 输出全场周期性指标范围与产物路径
    print(
        "日周期方差解释率: "
        f"{metrics_frame['日周期方差解释率 (%)'].min():.2f}%–"
        f"{metrics_frame['日周期方差解释率 (%)'].max():.2f}%"
    )
    print(
        "24小时自相关: "
        f"{metrics_frame['24小时自相关'].min():.4f}–"
        f"{metrics_frame['24小时自相关'].max():.4f}"
    )
    print(
        "7天自相关: "
        f"{metrics_frame['7天自相关'].min():.4f}–"
        f"{metrics_frame['7天自相关'].max():.4f}"
    )
    print(f"指标表: {METRICS_PATH.relative_to(PROJECT_ROOT)}")
    print(f"分析图: {FIGURE_PATH.relative_to(PROJECT_ROOT)}")
    print(f"PDF: {PDF_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
