from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from config.settings import *
from data_provider.common import Repository, build_single_turbine_windows, make_loaders


class ForecastWeatherDataset(Dataset):
    """历史天气相关特征到未来天气相关特征。"""

    def __init__(self, repository: Repository, split: str, horizon: int) -> None:
        self.repository = repository
        self.split = split
        self.horizon = int(horizon)
        self.windows = build_single_turbine_windows(repository, split, self.horizon)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int):
        turbine_index, history_start = self.windows[index]
        series = self.repository.series[turbine_index]
        target_start = history_start + HISTORY_STEPS
        target_end = target_start + self.horizon
        start_ns = int(series.times[target_start])
        past = np.ascontiguousarray(
            series.scaled[history_start:target_start, WEATHER_INDICES]
        )
        weather = np.zeros(
            (self.horizon, len(WEATHER_INDICES)), dtype=np.float32
        )
        target = np.ascontiguousarray(
            series.scaled[target_start:target_end, WEATHER_INDICES]
        )

        return (
            torch.from_numpy(past),
            torch.from_numpy(weather),
            torch.from_numpy(target),
            torch.tensor(turbine_index, dtype=torch.long),
            torch.tensor(start_ns, dtype=torch.long),
        )


def make_forecast_weather_loaders(
    horizon: int, batch_size: int, num_workers: int
):
    repository = Repository()
    train_dataset = ForecastWeatherDataset(repository, "train", horizon)
    val_dataset = ForecastWeatherDataset(repository, "val", horizon)
    test_dataset = ForecastWeatherDataset(repository, "test", horizon)
    datasets = {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
    }
    loaders = make_loaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size,
        num_workers,
        shuffle_train=True,
    )
    return repository, datasets, loaders
