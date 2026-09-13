from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from config.settings import *
from data_provider.common import Repository, belongs, make_loaders


class WeatherToPowerDataset(Dataset):
    """同一时刻的天气相关特征到功率。"""

    def __init__(self, repository: Repository, split: str) -> None:
        self.repository = repository
        self.split = split
        self.points = []

        for turbine_index, series in enumerate(repository.series):
            for point_index in range(len(series.times)):
                time_ns = int(series.times[point_index])

                if belongs(split, time_ns, time_ns):
                    self.points.append((turbine_index, point_index, time_ns))

        self.points.sort(key=lambda point: (point[2], point[0]))

    def __len__(self) -> int:
        return len(self.points)

    def __getitem__(self, index: int):
        turbine_index, point_index, time_ns = self.points[index]
        series = self.repository.series[turbine_index]
        past = np.zeros((1, 1), dtype=np.float32)
        weather = np.ascontiguousarray(
            series.scaled[point_index, WEATHER_INDICES]
        )
        target = np.asarray(
            [series.scaled[point_index, POWER_INDEX]], dtype=np.float32
        )

        return (
            torch.from_numpy(past),
            torch.from_numpy(weather),
            torch.from_numpy(target),
            torch.tensor(turbine_index, dtype=torch.long),
            torch.tensor(time_ns, dtype=torch.long),
        )


def make_weather_to_power_loaders(batch_size: int, num_workers: int):
    repository = Repository()
    train_dataset = WeatherToPowerDataset(repository, "train")
    val_dataset = WeatherToPowerDataset(repository, "val")
    test_dataset = WeatherToPowerDataset(repository, "test")
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
        shuffle_train=False,
    )
    return repository, datasets, loaders
