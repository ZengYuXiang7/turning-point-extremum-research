from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

import config as project_config
from data_provider.common import Repository, build_single_turbine_windows, make_loaders


class NoFutureWeatherDataset(Dataset):
    """预测起点之前的全部历史特征到未来功率。"""

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
        target_start = (
            history_start
            + project_config.HISTORY_STEPS * project_config.POINT_STRIDE_STEPS
        )
        target_end = target_start + self.horizon * project_config.POINT_STRIDE_STEPS
        start_ns = int(series.times[target_start])
        past = np.ascontiguousarray(series.scaled[ history_start:target_start:project_config.POINT_STRIDE_STEPS ])
        target = np.ascontiguousarray(series.scaled[ target_start:target_end:project_config.POINT_STRIDE_STEPS, self.repository.power_index, ])

        return (
            torch.from_numpy(past),
            torch.from_numpy(target),
            torch.tensor(turbine_index, dtype=torch.long),
            torch.tensor(start_ns, dtype=torch.long),
        )


def make_no_future_weather_loaders(horizon: int, batch_size: int, num_workers: int, all_features: bool = False, correlated_features: bool = False,):
    repository = Repository(all_features=all_features, correlated_features=correlated_features,)
    train_dataset = NoFutureWeatherDataset(repository, "train", horizon)
    val_dataset = NoFutureWeatherDataset(repository, "val", horizon)
    test_dataset = NoFutureWeatherDataset(repository, "test", horizon)
    datasets = {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
    }
    loaders = make_loaders(train_dataset, val_dataset, test_dataset, batch_size, num_workers, shuffle_train=True,)
    return repository, datasets, loaders
