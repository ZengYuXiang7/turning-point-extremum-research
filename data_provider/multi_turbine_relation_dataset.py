from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from config.settings import *
from data_provider.common import Repository, build_multi_turbine_windows, make_loaders


class MultiTurbineNoFutureWeatherDataset(Dataset):
    """同步十六台风机的历史特征到未来功率，不提供未来天气。"""

    def __init__(self, repository: Repository, split: str, horizon: int) -> None:
        self.repository = repository
        self.split = split
        self.horizon = int(horizon)
        self.windows = build_multi_turbine_windows(repository, split, self.horizon)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int):
        history_start = self.windows[index]
        target_start = history_start + HISTORY_STEPS
        target_end = target_start + self.horizon
        past_list = []
        target_list = []

        for series in self.repository.series:
            past_list.append(series.scaled[history_start:target_start])
            target_list.append(series.scaled[target_start:target_end, POWER_INDEX])

        past = np.stack(past_list, axis=0).astype(np.float32, copy=False)
        weather = np.zeros(
            (len(self.repository.series), self.horizon, len(WEATHER_INDICES)),
            dtype=np.float32,
        )
        target = np.stack(target_list, axis=0).astype(np.float32, copy=False)
        turbine_id = torch.arange(len(self.repository.series), dtype=torch.long)
        target_start_ns = int(self.repository.series[0].times[target_start])

        return (
            torch.from_numpy(np.ascontiguousarray(past)),
            torch.from_numpy(weather),
            torch.from_numpy(np.ascontiguousarray(target)),
            turbine_id,
            torch.tensor(target_start_ns, dtype=torch.long),
        )


class MultiTurbineOracleFutureWeatherDataset(Dataset):
    """同步十六台风机的历史特征和未来真实天气到未来功率。"""

    def __init__(self, repository: Repository, split: str, horizon: int) -> None:
        self.repository = repository
        self.split = split
        self.horizon = int(horizon)
        self.windows = build_multi_turbine_windows(repository, split, self.horizon)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int):
        history_start = self.windows[index]
        target_start = history_start + HISTORY_STEPS
        target_end = target_start + self.horizon
        past_list = []
        weather_list = []
        target_list = []

        for series in self.repository.series:
            past_list.append(series.scaled[history_start:target_start])
            weather_list.append(
                series.scaled[target_start:target_end, WEATHER_INDICES]
            )
            target_list.append(series.scaled[target_start:target_end, POWER_INDEX])

        past = np.stack(past_list, axis=0).astype(np.float32, copy=False)
        weather = np.stack(weather_list, axis=0).astype(np.float32, copy=False)
        target = np.stack(target_list, axis=0).astype(np.float32, copy=False)
        turbine_id = torch.arange(len(self.repository.series), dtype=torch.long)
        target_start_ns = int(self.repository.series[0].times[target_start])

        return (
            torch.from_numpy(np.ascontiguousarray(past)),
            torch.from_numpy(np.ascontiguousarray(weather)),
            torch.from_numpy(np.ascontiguousarray(target)),
            turbine_id,
            torch.tensor(target_start_ns, dtype=torch.long),
        )


def make_multi_turbine_no_future_weather_loaders(
    horizon: int, batch_size: int, num_workers: int
):
    repository = Repository()
    train_dataset = MultiTurbineNoFutureWeatherDataset(repository, "train", horizon)
    val_dataset = MultiTurbineNoFutureWeatherDataset(repository, "val", horizon)
    test_dataset = MultiTurbineNoFutureWeatherDataset(repository, "test", horizon)
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


def make_multi_turbine_oracle_future_weather_loaders(
    horizon: int, batch_size: int, num_workers: int
):
    repository = Repository()
    train_dataset = MultiTurbineOracleFutureWeatherDataset(
        repository, "train", horizon
    )
    val_dataset = MultiTurbineOracleFutureWeatherDataset(repository, "val", horizon)
    test_dataset = MultiTurbineOracleFutureWeatherDataset(
        repository, "test", horizon
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
