from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "dataset" / "processed"
COLUMNS_PATH = DATA_ROOT / "columns.json"
OUTPUT_BASE = PROJECT_ROOT / "output" / "lightgbm_feature_selection_all_turbines"
POWER_COLUMN = "风机-P"
TURBINE_COUNT = 16
BASE_INTERVAL_SECONDS = 10
TRAIN_RATIO = 7
VALID_RATIO = 1
TEST_RATIO = 2


def build_lag_feature_names(columns, history_steps, forecast_step, point_seconds):
    # 为每个字段的各历史位置建立可追溯名称
    feature_count = len(columns)
    output_count = history_steps * feature_count
    lag_feature_names = np.empty(output_count, dtype=object)
    lag_points = np.empty(output_count, dtype=np.int64)
    raw_feature_names = np.empty(output_count, dtype=object)
    output_index = 0

    for history_index in range(history_steps):
        lag_point = forecast_step + history_steps - 1 - history_index
        lag_minutes = lag_point * point_seconds / 60
        for feature_name in columns:
            lag_feature_names[output_index] = f"{feature_name}__距目标{lag_minutes:g}分钟"
            lag_points[output_index] = lag_point
            raw_feature_names[output_index] = feature_name
            output_index += 1

    return lag_feature_names, lag_points, raw_feature_names


def build_supervised_samples(data, columns, args):
    # 从10秒底层序列构造历史窗口和单个未来目标
    timestamps = data[:, 0].astype(np.int64)
    feature_values = data[:, 1:]
    power_index = columns.index(POWER_COLUMN)
    point_stride = args.point_interval_seconds // BASE_INTERVAL_SECONDS
    target_offset = (args.history_steps - 1 + args.forecast_step) * point_stride
    first_target_timestamp = timestamps[target_offset]
    window_stride_seconds = args.window_stride_steps * args.point_interval_seconds
    alignment_seconds = (-first_target_timestamp) % window_stride_seconds
    first_history_start = alignment_seconds // BASE_INTERVAL_SECONDS
    history_starts = np.arange(
        first_history_start,
        len(data) - target_offset,
        args.window_stride_steps * point_stride,
        dtype=np.int64,
    )
    history_offsets = np.arange(args.history_steps, dtype=np.int64) * point_stride
    history_indices = history_starts[:, None] + history_offsets[None, :]
    target_indices = history_starts + target_offset
    target_timestamps = timestamps[target_indices]

    # 按目标时刻复用项目的70/10/20时间边界
    ratio_total = TRAIN_RATIO + VALID_RATIO + TEST_RATIO
    train_count = len(timestamps) * TRAIN_RATIO // ratio_total
    valid_count = len(timestamps) * (TRAIN_RATIO + VALID_RATIO) // ratio_total
    train_end = timestamps[train_count]
    valid_end = timestamps[valid_count]
    train_mask = target_timestamps < train_end
    valid_mask = (target_timestamps >= train_end) & (target_timestamps < valid_end)

    # LightGBM直接使用原始量纲并展平历史点与字段
    window_values = feature_values[history_indices].astype(np.float32)
    flattened_values = window_values.reshape(len(window_values), -1)
    targets = feature_values[target_indices, power_index].astype(np.float32)
    train_values = flattened_values[train_mask]
    train_targets = targets[train_mask]
    valid_values = flattened_values[valid_mask]
    valid_targets = targets[valid_mask]
    train_timestamps = target_timestamps[train_mask]
    valid_timestamps = target_timestamps[valid_mask]
    return train_values, train_targets, valid_values, valid_targets, train_timestamps, valid_timestamps


def load_all_turbine_samples(columns, args):
    # 用首台风机确定联合数组规模
    first_turbine_id = 1
    first_path = DATA_ROOT / f"turbine_{first_turbine_id:02d}.npy"
    first_data = np.load(first_path, mmap_mode="r")
    first_samples = build_supervised_samples(first_data, columns, args)
    first_train_values, first_train_targets = first_samples[:2]
    first_valid_values, first_valid_targets = first_samples[2:4]
    first_train_timestamps, first_valid_timestamps = first_samples[4:]
    train_per_turbine = len(first_train_targets)
    valid_per_turbine = len(first_valid_targets)
    input_dimension = first_train_values.shape[1]
    train_values = np.empty((TURBINE_COUNT * train_per_turbine, input_dimension), dtype=np.float32)
    train_targets = np.empty(TURBINE_COUNT * train_per_turbine, dtype=np.float32)
    train_timestamps = np.empty(TURBINE_COUNT * train_per_turbine, dtype=np.int64)
    valid_values = np.empty((TURBINE_COUNT * valid_per_turbine, input_dimension), dtype=np.float32)
    valid_targets = np.empty(TURBINE_COUNT * valid_per_turbine, dtype=np.float32)
    valid_turbine_ids = np.empty(TURBINE_COUNT * valid_per_turbine, dtype=np.int64)
    valid_timestamps = np.empty(TURBINE_COUNT * valid_per_turbine, dtype=np.int64)

    # 写入首台风机样本
    train_values[:train_per_turbine] = first_train_values
    train_targets[:train_per_turbine] = first_train_targets
    train_timestamps[:train_per_turbine] = first_train_timestamps
    valid_values[:valid_per_turbine] = first_valid_values
    valid_targets[:valid_per_turbine] = first_valid_targets
    valid_turbine_ids[:valid_per_turbine] = first_turbine_id
    valid_timestamps[:valid_per_turbine] = first_valid_timestamps

    # 逐台加载剩余风机
    progress = tqdm(total=TURBINE_COUNT, initial=1, desc="构造全部风机窗口", unit="台")
    for turbine_id in range(2, TURBINE_COUNT + 1):
        data_path = DATA_ROOT / f"turbine_{turbine_id:02d}.npy"
        data = np.load(data_path, mmap_mode="r")
        samples = build_supervised_samples(data, columns, args)
        turbine_train_values, turbine_train_targets = samples[:2]
        turbine_valid_values, turbine_valid_targets = samples[2:4]
        turbine_train_timestamps, turbine_valid_timestamps = samples[4:]
        train_left = (turbine_id - 1) * train_per_turbine
        train_right = turbine_id * train_per_turbine
        valid_left = (turbine_id - 1) * valid_per_turbine
        valid_right = turbine_id * valid_per_turbine
        train_values[train_left:train_right] = turbine_train_values
        train_targets[train_left:train_right] = turbine_train_targets
        train_timestamps[train_left:train_right] = turbine_train_timestamps
        valid_values[valid_left:valid_right] = turbine_valid_values
        valid_targets[valid_left:valid_right] = turbine_valid_targets
        valid_turbine_ids[valid_left:valid_right] = turbine_id
        valid_timestamps[valid_left:valid_right] = turbine_valid_timestamps
        progress.update(1)
    progress.close()
    return train_values, train_targets, train_timestamps, valid_values, valid_targets, valid_turbine_ids, valid_timestamps


def select_balanced_importance_sample(valid_values, valid_targets, valid_ids, args):
    # 每台风机抽取等量验证样本
    samples_per_turbine = args.importance_samples_per_turbine
    selected_indices = np.empty(TURBINE_COUNT * samples_per_turbine, dtype=np.int64)
    random_generator = np.random.default_rng(args.seed)

    for turbine_id in range(1, TURBINE_COUNT + 1):
        turbine_indices = np.flatnonzero(valid_ids == turbine_id)
        turbine_selected = random_generator.choice(
            turbine_indices, size=samples_per_turbine, replace=False,
        )
        left = (turbine_id - 1) * samples_per_turbine
        right = turbine_id * samples_per_turbine
        selected_indices[left:right] = turbine_selected

    sample_values = valid_values[selected_indices]
    sample_targets = valid_targets[selected_indices]
    sample_ids = valid_ids[selected_indices]
    return sample_values, sample_targets, sample_ids
