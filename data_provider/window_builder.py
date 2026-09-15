from __future__ import annotations

import numpy as np

import config as project_config
from config import EXPECTED_DELTA_NS


def belongs(repository, split: str, start: int, end: int) -> bool:
    # 目标窗口归属由目标时间决定，各区间均为左闭右开
    if split == "train":
        return end < repository.train_end
    if split == "val":
        return start >= repository.train_end and end < repository.valid_end
    return start >= repository.valid_end


def build_single_turbine_windows(repository, split: str, horizon: int):
    # 按连续时间段构造单风机历史与预测窗口
    windows = []
    raw_span_steps = (
        (project_config.HISTORY_STEPS + horizon - 1)
        * project_config.POINT_STRIDE_STEPS
        + 1
    )
    window_stride_base_steps = (
        project_config.WINDOW_STRIDE_STEPS * project_config.POINT_STRIDE_STEPS
    )

    for turbine_index, series in enumerate(repository.series):
        breaks = np.flatnonzero(np.diff(series.times) != EXPECTED_DELTA_NS) + 1
        bounds = np.concatenate([[0], breaks, [len(series.times)]])
        segment_count = len(bounds) - 1

        for segment_index in range(segment_count):
            left = int(bounds[segment_index])
            right = int(bounds[segment_index + 1])

            if right - left < raw_span_steps:
                continue

            # 将首个预测起点对齐到模型点时间网格
            first_target_index = (
                left
                + project_config.HISTORY_STEPS
                * project_config.POINT_STRIDE_STEPS
            )
            first_target_ns = int(series.times[first_target_index])
            alignment_ns = (-first_target_ns) % project_config.WINDOW_STRIDE_NS
            alignment_steps = alignment_ns // EXPECTED_DELTA_NS
            first_history_start = left + alignment_steps

            for history_start in range(
                first_history_start,
                right - raw_span_steps + 1,
                window_stride_base_steps,
            ):
                target_start_index = (
                    history_start
                    + project_config.HISTORY_STEPS
                    * project_config.POINT_STRIDE_STEPS
                )
                target_end_index = (
                    target_start_index
                    + (horizon - 1) * project_config.POINT_STRIDE_STEPS
                )
                target_start_ns = int(series.times[target_start_index])
                target_end_ns = int(series.times[target_end_index])

                if belongs(repository, split, target_start_ns, target_end_ns):
                    windows.append((turbine_index, history_start))

    return windows


def build_multi_turbine_windows(repository, split: str, horizon: int):
    # 按参考时间轴构造16台风机联合窗口
    windows = []
    reference_times = repository.series[0].times
    raw_span_steps = (
        (project_config.HISTORY_STEPS + horizon - 1)
        * project_config.POINT_STRIDE_STEPS
        + 1
    )
    window_stride_base_steps = (
        project_config.WINDOW_STRIDE_STEPS * project_config.POINT_STRIDE_STEPS
    )
    breaks = np.flatnonzero(np.diff(reference_times) != EXPECTED_DELTA_NS) + 1
    bounds = np.concatenate([[0], breaks, [len(reference_times)]])
    segment_count = len(bounds) - 1

    for segment_index in range(segment_count):
        left = int(bounds[segment_index])
        right = int(bounds[segment_index + 1])

        if right - left < raw_span_steps:
            continue

        # 将首个预测起点对齐到模型点时间网格
        first_target_index = (
            left
            + project_config.HISTORY_STEPS
            * project_config.POINT_STRIDE_STEPS
        )
        first_target_ns = int(reference_times[first_target_index])
        alignment_ns = (-first_target_ns) % project_config.WINDOW_STRIDE_NS
        alignment_steps = alignment_ns // EXPECTED_DELTA_NS
        first_history_start = left + alignment_steps

        for history_start in range(
            first_history_start,
            right - raw_span_steps + 1,
            window_stride_base_steps,
        ):
            target_start_index = (
                history_start
                + project_config.HISTORY_STEPS
                * project_config.POINT_STRIDE_STEPS
            )
            target_end_index = (
                target_start_index
                + (horizon - 1) * project_config.POINT_STRIDE_STEPS
            )
            target_start_ns = int(reference_times[target_start_index])
            target_end_ns = int(reference_times[target_end_index])

            if belongs(repository, split, target_start_ns, target_end_ns):
                windows.append(history_start)

    return windows
