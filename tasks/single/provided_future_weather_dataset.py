from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

import config as project_config
from config import SPLIT_RATIOS
from data_provider.common import TurbineSeries, make_loaders


SOURCE_HISTORY_COLUMNS = [
    "超短期预测功率(MW)",
    "实际功率(MW)",
    "风电理论功率(MW)",
    "实际风速(m/s)",
    "风电可用功率(MW)",
    "风电电网调令(MW)",
]
HISTORY_COLUMNS = [
    "超短期预测功率(kW)",
    "实际功率(kW)",
    "风电理论功率(kW)",
    "实际风速(m/s)",
    "风电可用功率(kW)",
    "风电电网调令(kW)",
]
FUTURE_COLUMNS = ["预测风速(m/s)"]
POWER_COLUMN = "实际功率(kW)"
POWER_SOURCE_INDICES = [0, 1, 2, 4, 5]


def belongs_to_split(repository, split: str, target_start: int, target_end: int) -> bool:
    # 预测目标必须完整落在同一个时间切分中。
    if split == "train":
        return target_end < repository.train_end
    if split == "val":
        return target_start >= repository.train_end and target_end < repository.valid_end
    return target_start >= repository.valid_end


def build_provided_weather_windows(repository, split: str, horizon: int):
    # 每行就是一个15分钟模型点，只保留未来预测风速完整的窗口。
    history_steps = project_config.HISTORY_STEPS
    window_size = history_steps + horizon
    windows = []
    excluded_windows = 0
    series = repository.series[0]

    for history_start in range(0, len(series.times) - window_size + 1, project_config.WINDOW_STRIDE_STEPS,):
        target_start_index = history_start + history_steps
        target_end_index = target_start_index + horizon - 1
        target_start = int(series.times[target_start_index])
        target_end = int(series.times[target_end_index])
        if not belongs_to_split(repository, split, target_start, target_end):
            continue

        future_weather = repository.future_scaled[target_start_index : target_end_index + 1]
        if np.all(np.isfinite(future_weather)):
            windows.append((0, history_start))
        else:
            excluded_windows += 1

    return windows, excluded_windows


class ProvidedFutureWeatherRepository:
    """甲方场站级15分钟功率与已提供的未来预测风速。"""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        frames = []
        source_paths = sorted(self.root.glob("yc*.xlsx"))
        for source_path in source_paths:
            frame = pd.read_excel(source_path, sheet_name="预测数据")
            frames.append(frame)
        table = pd.concat(frames, ignore_index=True)
        table["时间"] = pd.to_datetime(table["时间"])
        table = table.sort_values("时间").reset_index(drop=True)

        # 功率统一转成项目评估协议使用的kW。
        times = table["时间"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
        raw = table[SOURCE_HISTORY_COLUMNS].to_numpy(dtype=np.float32)
        raw[:, POWER_SOURCE_INDICES] *= 1_000.0
        future_raw = table[FUTURE_COLUMNS].to_numpy(dtype=np.float32)

        split_ratio_total = sum(SPLIT_RATIOS)
        train_count = len(times) * SPLIT_RATIOS[0] // split_ratio_total
        valid_count = len(times) * sum(SPLIT_RATIOS[:2]) // split_ratio_total
        self.train_end = int(times[train_count])
        self.valid_end = int(times[valid_count])

        # 历史特征和甲方预测风速分别用训练段统计量标准化。
        history_scaler = StandardScaler()
        history_scaler.fit(raw[:train_count])
        scaled = history_scaler.transform(raw).astype(np.float32)
        forecast_train = future_raw[:train_count]
        forecast_train = forecast_train[np.all(np.isfinite(forecast_train), axis=1)]
        future_scaler = StandardScaler()
        future_scaler.fit(forecast_train)
        self.future_scaled = future_scaler.transform(future_raw).astype(np.float32)

        self.feature_names = HISTORY_COLUMNS
        self.future_feature_names = FUTURE_COLUMNS
        self.power_index = HISTORY_COLUMNS.index(POWER_COLUMN)
        self.provided_power_forecast_kw = np.ascontiguousarray(raw[:, 0])
        self.missing_forecast_points = int(np.count_nonzero(~np.isfinite(future_raw)))
        self.base_interval_seconds = 15 * 60
        self.source_description = f"{self.root.as_posix()}/yc*.xlsx"
        self.split_description = "chronological_70_10_20"
        self.normalization_description = "site_train_standard_scaler"
        self.target_name = POWER_COLUMN
        self.target_transform_description = "site_train_standard_scaler"
        self.entity_name = "site"

        power_mean = float(history_scaler.mean_[self.power_index])
        power_std = float(history_scaler.scale_[self.power_index])
        series = TurbineSeries(times=times, raw=raw, scaled=scaled, power_mean=power_mean, power_std=power_std,)
        self.series = [series]


class ProvidedFutureWeatherDataset(Dataset):
    """历史场站数据和甲方预测风速到未来实际功率。"""

    def __init__(self, repository: ProvidedFutureWeatherRepository, split: str, horizon: int,) -> None:
        self.repository = repository
        self.split = split
        self.horizon = int(horizon)
        self.windows, self.excluded_windows = build_provided_weather_windows(repository, split, self.horizon,)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int):
        _, history_start = self.windows[index]
        series = self.repository.series[0]
        target_start = history_start + project_config.HISTORY_STEPS
        target_end = target_start + self.horizon
        start_ns = int(series.times[target_start])
        past = np.ascontiguousarray(series.scaled[history_start:target_start])
        weather = np.ascontiguousarray(self.repository.future_scaled[target_start:target_end])
        target = np.ascontiguousarray(series.scaled[target_start:target_end, self.repository.power_index])

        return (
            torch.from_numpy(past),
            torch.from_numpy(weather),
            torch.from_numpy(target),
            torch.tensor(0, dtype=torch.long),
            torch.tensor(start_ns, dtype=torch.long),
        )


def make_provided_future_weather_loaders(horizon: int, batch_size: int, num_workers: int, provided_weather_dir: str,):
    repository = ProvidedFutureWeatherRepository(Path(provided_weather_dir))
    train_dataset = ProvidedFutureWeatherDataset(repository, "train", horizon)
    val_dataset = ProvidedFutureWeatherDataset(repository, "val", horizon)
    test_dataset = ProvidedFutureWeatherDataset(repository, "test", horizon)
    datasets = {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
    }
    loaders = make_loaders(train_dataset, val_dataset, test_dataset, batch_size, num_workers, shuffle_train=True,)
    return repository, datasets, loaders
