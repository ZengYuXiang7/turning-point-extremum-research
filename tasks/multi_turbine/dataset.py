from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

import config as project_config
from config import WEATHER_INDICES
from data_provider.common import Repository, build_multi_turbine_windows, make_loaders
from tasks.multi_turbine.relation_prior import build_wind_dtw_relation


def stack_multi_turbine_panel(repository: Repository) -> np.ndarray:
    # 将同步风机序列合成统一的[风机, 时间, 特征]面板
    scaled_series = []
    for series in repository.series:
        scaled_series.append(series.scaled)
    panel = np.stack(scaled_series, axis=0).astype(np.float32, copy=False)

    # 各风机序列改为面板视图，避免长期保留第二份标准化数据
    for turbine_index in range(len(repository.series)):
        repository.series[turbine_index].scaled = panel[turbine_index]

    return panel


class MultiTurbineRelationRepository(Repository):
    """同步十六台风机的联合时间面板。"""

    def __init__(self, all_features: bool = False, pstr_relation_top_k: int = 0, pstr_relation_candidate_k: int = 0, pstr_relation_downsample: int = 0, pstr_relation_band: int = 0,) -> None:
        super().__init__(all_features=all_features, shared_feature_scaling=True)
        self.panel = stack_multi_turbine_panel(self)
        self.times = self.series[0].times

        # 三个切分均沿统一时间轴取视图
        self.train_end_index = int(np.searchsorted(self.times, self.train_end))
        self.valid_end_index = int(np.searchsorted(self.times, self.valid_end))
        self.train_panel = self.panel[:, : self.train_end_index]
        self.val_panel = self.panel[:, self.train_end_index : self.valid_end_index]
        self.test_panel = self.panel[:, self.valid_end_index :]
        if pstr_relation_top_k:
            self.pstr_relation_weight = build_wind_dtw_relation(self.panel, self.train_end_index, self.power_index, pstr_relation_top_k, pstr_relation_candidate_k, pstr_relation_downsample, pstr_relation_band,)


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
        target_start = (
            history_start
            + project_config.HISTORY_STEPS * project_config.POINT_STRIDE_STEPS
        )
        target_end = target_start + self.horizon * project_config.POINT_STRIDE_STEPS
        past = np.ascontiguousarray(self.repository.panel[ :, history_start:target_start:project_config.POINT_STRIDE_STEPS, : ])
        target = np.ascontiguousarray(self.repository.panel[ :, target_start:target_end:project_config.POINT_STRIDE_STEPS, self.repository.power_index, ])
        turbine_id = torch.arange(len(self.repository.series), dtype=torch.long)
        target_start_ns = int(self.repository.times[target_start])

        return (
            torch.from_numpy(past),
            torch.from_numpy(target),
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
        target_start = (
            history_start
            + project_config.HISTORY_STEPS * project_config.POINT_STRIDE_STEPS
        )
        target_end = target_start + self.horizon * project_config.POINT_STRIDE_STEPS
        past = np.ascontiguousarray(self.repository.panel[ :, history_start:target_start:project_config.POINT_STRIDE_STEPS, : ])
        weather = np.ascontiguousarray(self.repository.panel[ :, target_start:target_end:project_config.POINT_STRIDE_STEPS, WEATHER_INDICES, ])
        target = np.ascontiguousarray(self.repository.panel[ :, target_start:target_end:project_config.POINT_STRIDE_STEPS, self.repository.power_index, ])
        turbine_id = torch.arange(len(self.repository.series), dtype=torch.long)
        target_start_ns = int(self.repository.times[target_start])

        return (
            torch.from_numpy(past),
            torch.from_numpy(weather),
            torch.from_numpy(target),
            turbine_id,
            torch.tensor(target_start_ns, dtype=torch.long),
        )


def make_multi_turbine_no_future_weather_loaders(horizon: int, batch_size: int, num_workers: int, all_features: bool = False, pstr_relation_top_k: int = 0, pstr_relation_candidate_k: int = 0, pstr_relation_downsample: int = 0, pstr_relation_band: int = 0,):
    repository = MultiTurbineRelationRepository(all_features, pstr_relation_top_k, pstr_relation_candidate_k, pstr_relation_downsample, pstr_relation_band,)
    train_dataset = MultiTurbineNoFutureWeatherDataset(repository, "train", horizon)
    val_dataset = MultiTurbineNoFutureWeatherDataset(repository, "val", horizon)
    test_dataset = MultiTurbineNoFutureWeatherDataset(repository, "test", horizon)
    datasets = {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
    }
    loaders = make_loaders(train_dataset, val_dataset, test_dataset, batch_size, num_workers, shuffle_train=True,)
    return repository, datasets, loaders


def make_multi_turbine_oracle_future_weather_loaders(horizon: int, batch_size: int, num_workers: int):
    repository = MultiTurbineRelationRepository()
    train_dataset = MultiTurbineOracleFutureWeatherDataset(repository, "train", horizon)
    val_dataset = MultiTurbineOracleFutureWeatherDataset(repository, "val", horizon)
    test_dataset = MultiTurbineOracleFutureWeatherDataset(repository, "test", horizon)
    datasets = {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
    }
    loaders = make_loaders(train_dataset, val_dataset, test_dataset, batch_size, num_workers, shuffle_train=True,)
    return repository, datasets, loaders
