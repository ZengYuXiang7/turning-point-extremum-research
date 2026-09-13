from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from config.settings import *
from data_provider.common import (
    Repository,
    build_single_turbine_windows,
    load_weather_forecast_table,
    make_loaders,
)


class PredictedFutureWeatherDataset(Dataset):
    """历史全特征和程序一预测的未来天气到未来功率。"""

    def __init__(
        self, repository: Repository, split: str, horizon: int, forecast_table
    ) -> None:
        self.repository = repository
        self.split = split
        self.horizon = int(horizon)
        self.forecast_table = forecast_table
        self.windows = build_single_turbine_windows(repository, split, self.horizon)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int):
        turbine_index, history_start = self.windows[index]
        series = self.repository.series[turbine_index]
        target_start = history_start + HISTORY_STEPS
        target_end = target_start + self.horizon
        start_ns = int(series.times[target_start])
        key = f"{turbine_index}:{start_ns}"
        past = np.ascontiguousarray(series.scaled[history_start:target_start])
        weather = self.forecast_table[key]
        target = np.ascontiguousarray(
            series.scaled[target_start:target_end, POWER_INDEX]
        )

        return (
            torch.from_numpy(past),
            torch.from_numpy(weather),
            torch.from_numpy(target),
            torch.tensor(turbine_index, dtype=torch.long),
            torch.tensor(start_ns, dtype=torch.long),
        )


def make_predicted_future_weather_loaders(
    horizon: int,
    batch_size: int,
    num_workers: int,
    weather_forecast_dir: str,
):
    repository = Repository()
    forecast_root = Path(weather_forecast_dir)
    train_forecast = load_weather_forecast_table(
        forecast_root / "weather_forecast_train.npz"
    )
    val_forecast = load_weather_forecast_table(
        forecast_root / "weather_forecast_val.npz"
    )
    test_forecast = load_weather_forecast_table(
        forecast_root / "weather_forecast_test.npz"
    )
    train_dataset = PredictedFutureWeatherDataset(
        repository, "train", horizon, train_forecast
    )
    val_dataset = PredictedFutureWeatherDataset(
        repository, "val", horizon, val_forecast
    )
    test_dataset = PredictedFutureWeatherDataset(
        repository, "test", horizon, test_forecast
    )
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
