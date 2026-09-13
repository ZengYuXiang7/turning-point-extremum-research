from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from config.settings import *


CIRCULAR_SOURCE_COLUMNS = (
    "风机-实时风向",
    "偏航系统-对风角度",
    "偏航系统-机舱位置",
    "偏航系统-扭揽角度",
)
CONSTANT_SOURCE_COLUMN = "传动链-液压站压力"


@dataclass
class TurbineSeries:
    times: np.ndarray
    raw: np.ndarray
    scaled: np.ndarray
    power_mean: float
    power_std: float


class Repository:
    def __init__(self, root: Path = DATASET_ROOT, all_features: bool = False) -> None:
        self.root = Path(root)
        reference_data = np.load(self.root / "turbine_01.npy", mmap_mode="r")
        reference_times = reference_data[:, 0].astype(np.int64) * 1_000_000_000
        split_ratio_total = sum(SPLIT_RATIOS)
        train_count = len(reference_times) * SPLIT_RATIOS[0] // split_ratio_total
        valid_count = len(reference_times) * sum(SPLIT_RATIOS[:2]) // split_ratio_total
        self.train_end = int(reference_times[train_count])
        self.valid_end = int(reference_times[valid_count])
        raw_series = []
        columns_path = self.root / "columns.json"
        source_columns = json.loads(columns_path.read_text(encoding="utf-8"))

        # 全特征口径保留全部有效字段，周期角度以 sin/cos 替换原始角度。
        all_feature_names = []
        for source_column in source_columns:
            if source_column == CONSTANT_SOURCE_COLUMN:
                continue
            if source_column in CIRCULAR_SOURCE_COLUMNS:
                sin_name = f"{source_column}_sin"
                cos_name = f"{source_column}_cos"
                all_feature_names.append(sin_name)
                all_feature_names.append(cos_name)
            else:
                all_feature_names.append(source_column)

        if all_features:
            feature_names = all_feature_names
        else:
            feature_names = HISTORY_COLUMNS

        self.feature_names = feature_names
        self.power_index = feature_names.index("风机-P")
        theory_index = source_columns.index("风机-理论功率-计算")
        real_wind_speed_index = source_columns.index("风机-实时风速")
        real_wind_direction_index = source_columns.index("风机-实时风向")
        environment_temperature_index = source_columns.index("风机-环境温度")
        cabin_temperature_index = source_columns.index("机舱-舱内温度")
        tower_temperature_index = source_columns.index("塔筒-塔底温度")
        generator_speed_index = source_columns.index("发电机-发电机转速")
        shaft_speed_index = source_columns.index("传动链-主轴转速")
        blade_1_angle_index = source_columns.index("变桨轮毂-1#桨叶角度")
        blade_2_angle_index = source_columns.index("变桨轮毂-2#桨叶角度")
        blade_3_angle_index = source_columns.index("变桨轮毂-3#桨叶角度")
        yaw_angle_index = source_columns.index("偏航系统-对风角度")
        nacelle_position_index = source_columns.index("偏航系统-机舱位置")
        twist_angle_index = source_columns.index("偏航系统-扭揽角度")

        for turbine in range(1, 17):
            data = np.load(
                self.root / f"turbine_{turbine:02d}.npy", mmap_mode="r"
            )
            times = data[:, 0].astype(np.int64) * 1_000_000_000
            source = data[:, 1:]

            if all_features:
                feature_values = []
                for source_index in range(len(source_columns)):
                    source_column = source_columns[source_index]
                    if source_column == CONSTANT_SOURCE_COLUMN:
                        continue
                    values = source[:, source_index]
                    if source_column in CIRCULAR_SOURCE_COLUMNS:
                        radians = np.deg2rad(values)
                        sin_values = np.sin(radians)
                        cos_values = np.cos(radians)
                        feature_values.append(sin_values)
                        feature_values.append(cos_values)
                    else:
                        feature_values.append(values)
                raw = np.column_stack(feature_values).astype(np.float32)
            else:
                raw = np.empty((len(source), len(HISTORY_COLUMNS)), dtype=np.float32)
                raw[:, 0] = source[:, theory_index]
                raw[:, 1] = source[:, real_wind_speed_index]
                real_wind_direction = np.deg2rad(source[:, real_wind_direction_index])
                raw[:, 2] = np.sin(real_wind_direction)
                raw[:, 3] = np.cos(real_wind_direction)
                raw[:, 4] = source[:, environment_temperature_index]
                raw[:, 5] = source[:, cabin_temperature_index]
                raw[:, 6] = source[:, tower_temperature_index]
                raw[:, 7] = source[:, generator_speed_index]
                raw[:, 8] = source[:, shaft_speed_index]
                raw[:, 9] = source[:, blade_1_angle_index]
                raw[:, 10] = source[:, blade_2_angle_index]
                raw[:, 11] = source[:, blade_3_angle_index]
                yaw_angle = np.deg2rad(source[:, yaw_angle_index])
                raw[:, 12] = np.sin(yaw_angle)
                raw[:, 13] = np.cos(yaw_angle)
                nacelle_position = np.deg2rad(source[:, nacelle_position_index])
                raw[:, 14] = np.sin(nacelle_position)
                raw[:, 15] = np.cos(nacelle_position)
                twist_angle = np.deg2rad(source[:, twist_angle_index])
                raw[:, 16] = np.sin(twist_angle)
                raw[:, 17] = np.cos(twist_angle)
                raw[:, -1] = source[:, -1]

            train_mask = times < self.train_end
            train_features = raw[train_mask]
            feature_means = np.mean(train_features, axis=0, dtype=np.float64)
            feature_stds = np.std(train_features, axis=0, dtype=np.float64)
            scaled = ((raw - feature_means) / feature_stds).astype(np.float32)
            power_mean = float(feature_means[self.power_index])
            power_std = float(feature_stds[self.power_index])

            series = TurbineSeries(
                times=times,
                raw=raw,
                scaled=scaled,
                power_mean=power_mean,
                power_std=power_std,
            )

            raw_series.append(series)

        self.series = raw_series



def belongs(repository: Repository, split: str, start: int, end: int) -> bool:
    # 目标窗口归属由目标时间决定，各区间均为左闭右开
    if split == "train":
        return end < repository.train_end
    if split == "val":
        return start >= repository.train_end and end < repository.valid_end
    return start >= repository.valid_end



def build_single_turbine_windows(repository: Repository, split: str, horizon: int):
    windows = []
    total = HISTORY_STEPS + horizon

    for turbine_index, series in enumerate(repository.series):
        breaks = np.flatnonzero(np.diff(series.times) != EXPECTED_DELTA_NS) + 1
        bounds = np.concatenate([[0], breaks, [len(series.times)]])
        segment_count = len(bounds) - 1

        for segment_index in range(segment_count):
            left = int(bounds[segment_index])
            right = int(bounds[segment_index + 1])

            if right - left < total:
                continue

            # 将首个预测起点对齐到采样时间网格
            first_target_index = left + HISTORY_STEPS
            first_target_ns = int(series.times[first_target_index])
            alignment_ns = (-first_target_ns) % WINDOW_STRIDE_NS
            alignment_steps = alignment_ns // EXPECTED_DELTA_NS
            first_history_start = left + alignment_steps

            for history_start in range(
                first_history_start,
                right - total + 1,
                WINDOW_STRIDE_STEPS,
            ):
                target_start_index = history_start + HISTORY_STEPS
                target_end_index = target_start_index + horizon - 1
                target_start_ns = int(series.times[target_start_index])
                target_end_ns = int(series.times[target_end_index])

                if belongs(repository, split, target_start_ns, target_end_ns):
                    windows.append((turbine_index, history_start))

    return windows


def build_multi_turbine_windows(repository: Repository, split: str, horizon: int):
    windows = []
    reference_times = repository.series[0].times
    total = HISTORY_STEPS + horizon
    breaks = np.flatnonzero(np.diff(reference_times) != EXPECTED_DELTA_NS) + 1
    bounds = np.concatenate([[0], breaks, [len(reference_times)]])
    segment_count = len(bounds) - 1

    for segment_index in range(segment_count):
        left = int(bounds[segment_index])
        right = int(bounds[segment_index + 1])

        if right - left < total:
            continue

        # 将首个预测起点对齐到采样时间网格
        first_target_index = left + HISTORY_STEPS
        first_target_ns = int(reference_times[first_target_index])
        alignment_ns = (-first_target_ns) % WINDOW_STRIDE_NS
        alignment_steps = alignment_ns // EXPECTED_DELTA_NS
        first_history_start = left + alignment_steps

        for history_start in range(
            first_history_start,
            right - total + 1,
            WINDOW_STRIDE_STEPS,
        ):
            target_start_index = history_start + HISTORY_STEPS
            target_end_index = target_start_index + horizon - 1
            target_start_ns = int(reference_times[target_start_index])
            target_end_ns = int(reference_times[target_end_index])

            if belongs(repository, split, target_start_ns, target_end_ns):
                windows.append(history_start)

    return windows


def load_weather_forecast_table(path: Path):
    payload = np.load(path)
    weather = payload["weather"]
    turbine_id = payload["turbine_id"]
    target_start_ns = payload["target_start_ns"]
    table = {}

    for index in range(len(turbine_id)):
        key = f"{int(turbine_id[index])}:{int(target_start_ns[index])}"
        table[key] = np.ascontiguousarray(weather[index])

    return table


def make_loaders(
    train_dataset,
    val_dataset,
    test_dataset,
    batch_size: int,
    num_workers: int,
    shuffle_train: bool,
):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    loaders = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }
    return loaders
