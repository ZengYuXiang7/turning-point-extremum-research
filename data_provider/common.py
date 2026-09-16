from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

import config as project_config
from config import (
    BASE_INTERVAL_SECONDS,
    CORRELATED_HISTORY_COLUMNS,
    DATASET_ROOT,
    HISTORY_COLUMNS,
    SPLIT_RATIOS,
    WEATHER_COLUMNS,
)
from data_provider.window_builder import (
    build_multi_turbine_windows,
    build_single_turbine_windows,
)


@dataclass
class TurbineSeries:
    times: np.ndarray
    raw: np.ndarray
    scaled: np.ndarray
    power_mean: float
    power_std: float


class Repository:
    def __init__(self, root: Path = DATASET_ROOT, all_features: bool = False, correlated_features: bool = False, shared_feature_scaling: bool = False,) -> None:
        self.root = Path(root)
        reference_data = np.load(self.root / "turbine_01.npy", mmap_mode="r")
        reference_times = reference_data[:, 0].astype(np.int64) * 1_000_000_000
        split_ratio_total = sum(SPLIT_RATIOS)
        train_count = len(reference_times) * SPLIT_RATIOS[0] // split_ratio_total
        valid_count = len(reference_times) * sum(SPLIT_RATIOS[:2]) // split_ratio_total
        self.train_end = int(reference_times[train_count])
        self.valid_end = int(reference_times[valid_count])
        raw_values = []
        series_times = []
        columns_path = self.root / "columns.json"
        source_columns = json.loads(columns_path.read_text(encoding="utf-8"))

        if correlated_features:
            feature_names = CORRELATED_HISTORY_COLUMNS
            if project_config.HISTORY_FEATURE_PATH:
                feature_path = project_config.PROJECT_ROOT / project_config.HISTORY_FEATURE_PATH
                feature_names = json.loads(feature_path.read_text(encoding="utf-8"))
            correlated_feature_indices = []
            for feature_name in feature_names:
                feature_index = source_columns.index(feature_name)
                correlated_feature_indices.append(feature_index)
        elif all_features:
            feature_names = source_columns
        else:
            feature_names = HISTORY_COLUMNS

        self.feature_names = feature_names
        self.future_feature_names = WEATHER_COLUMNS
        self.power_index = feature_names.index("风机-P")
        self.base_interval_seconds = BASE_INTERVAL_SECONDS
        self.source_description = "dataset/processed/turbine_01.npy ... turbine_16.npy"
        self.split_description = "chronological_70_10_20"
        self.normalization_description = "per_turbine_train_standard_scaler_then_window_revin"
        self.target_name = "风机-P"
        self.target_transform_description = "per_turbine_train_standard_scaler"
        self.entity_name = "turbine"
        if shared_feature_scaling:
            self.normalization_description = "all_turbines_train_standard_scaler"
            self.target_transform_description = "all_turbines_train_standard_scaler"
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
            data = np.load(self.root / f"turbine_{turbine:02d}.npy", mmap_mode="r")
            times = data[:, 0].astype(np.int64) * 1_000_000_000
            source = data[:, 1:]

            if correlated_features:
                raw = source[:, correlated_feature_indices].astype(np.float32)
            elif all_features:
                raw = source.astype(np.float32)
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

            raw_values.append(raw)
            series_times.append(times)

        # 联合面板仅以全部风机的训练段数据拟合每个特征的一套统计量。
        if shared_feature_scaling:
            joint_train_values = []
            for turbine_index in range(len(raw_values)):
                times = series_times[turbine_index]
                train_mask = times < self.train_end
                joint_train_values.append(raw_values[turbine_index][train_mask])
            joint_train_features = np.concatenate(joint_train_values, axis=0)
            scaler = StandardScaler()
            scaler.fit(joint_train_features)
            shared_feature_means = scaler.mean_
            shared_feature_stds = scaler.scale_

        raw_series = []
        for turbine_index in range(len(raw_values)):
            times = series_times[turbine_index]
            raw = raw_values[turbine_index]
            if shared_feature_scaling:
                scaled = scaler.transform(raw).astype(np.float32)
                feature_means = shared_feature_means
                feature_stds = shared_feature_stds
            else:
                train_mask = times < self.train_end
                train_features = raw[train_mask]
                if all_features or correlated_features:
                    scaler = StandardScaler()
                    scaler.fit(train_features)
                    scaled = scaler.transform(raw).astype(np.float32)
                    feature_means = scaler.mean_
                    feature_stds = scaler.scale_
                else:
                    feature_means = np.mean(train_features, axis=0, dtype=np.float64)
                    feature_stds = np.std(train_features, axis=0, dtype=np.float64)
                    scaled = ((raw - feature_means) / feature_stds).astype(np.float32)
            power_mean = float(feature_means[self.power_index])
            power_std = float(feature_stds[self.power_index])
            series = TurbineSeries(times=times, raw=raw, scaled=scaled, power_mean=power_mean, power_std=power_std,)
            raw_series.append(series)

        self.series = raw_series


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


def make_loaders(train_dataset, val_dataset, test_dataset, batch_size: int, num_workers: int, shuffle_train: bool,):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers, pin_memory=True, drop_last=True,)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=True,)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False,)
    loaders = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }
    return loaders
