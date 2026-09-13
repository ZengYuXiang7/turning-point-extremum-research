from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from config.settings import *


def ns(value: str) -> int:
    return int(np.datetime64(value, "ns").astype(np.int64))


@dataclass
class TurbineSeries:
    times: np.ndarray
    raw: np.ndarray
    scaled: np.ndarray
    power_mean: float
    power_std: float


class Repository:
    def __init__(self, root: Path = DATASET_ROOT) -> None:
        self.root = Path(root)
        train_end = ns(TRAIN_END)
        raw_series = []

        for turbine in range(1, 17):
            
            times = np.load(self.root / "arrays" / f"turbine_{turbine:02d}_times.npy")
            raw = np.load(
                self.root / "arrays" / f"turbine_{turbine:02d}_values.npy"
            ).astype(np.float32)
            
            train_mask = times < train_end
            power_mean = float(np.mean(raw[train_mask, POWER_INDEX]))
            power_std = float(np.std(raw[train_mask, POWER_INDEX]))
            scaled = raw.copy()

            for index in CONTINUOUS_STANDARDIZE_INDICES:
                mean = float(np.mean(raw[train_mask, index]))
                std = float(np.std(raw[train_mask, index]))
                scaled[:, index] = (raw[:, index] - mean) / std

            series = TurbineSeries(
                times=times,
                raw=raw,
                scaled=scaled,
                power_mean=power_mean,
                power_std=power_std,
            )
            
            raw_series.append(series)

        self.series = raw_series


def belongs(split: str, start: int, end: int) -> bool:
    # 目标窗口归属由目标时间决定，各区间均为左闭右开
    if split == "train":
        return end < ns(TRAIN_END)
    if split == "val":
        return start >= ns(VALID_START) and end < ns(VALID_END)
    return start >= ns(TEST_START) and end < ns(TEST_END)


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

            for history_start in range(left, right - total + 1, WINDOW_STRIDE_STEPS):
                target_start_index = history_start + HISTORY_STEPS
                target_end_index = target_start_index + horizon - 1
                target_start_ns = int(series.times[target_start_index])
                target_end_ns = int(series.times[target_end_index])

                if belongs(split, target_start_ns, target_end_ns):
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

        for history_start in range(left, right - total + 1, WINDOW_STRIDE_STEPS):
            target_start_index = history_start + HISTORY_STEPS
            target_end_index = target_start_index + horizon - 1
            target_start_ns = int(reference_times[target_start_index])
            target_end_ns = int(reference_times[target_end_index])

            if belongs(split, target_start_ns, target_end_ns):
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
