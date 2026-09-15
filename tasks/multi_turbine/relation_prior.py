from __future__ import annotations

import numpy as np
import torch


def standardize_trajectories(values: np.ndarray) -> np.ndarray:
    means = values.mean(axis=1, keepdims=True)
    scales = values.std(axis=1, keepdims=True)
    standardized = (values - means) / np.maximum(scales, 1e-6)
    return standardized


def downsample_trajectories(values: np.ndarray, length: int) -> np.ndarray:
    turbine_count, time_steps = values.shape
    if time_steps <= length:
        return values.astype(np.float32, copy=False)
    edges = np.linspace(0, time_steps, length + 1).round().astype(np.int64)
    downsampled = np.empty((turbine_count, length), dtype=np.float32)
    for index in range(length):
        start = int(edges[index])
        end = int(edges[index + 1])
        downsampled[:, index] = values[:, start:end].mean(axis=1)
    return downsampled


def dtw_distance(first: np.ndarray, second: np.ndarray, band: int) -> float:
    length = first.shape[0]
    infinity = np.float32(1e20)
    previous = np.full(length + 1, infinity, dtype=np.float32)
    current = np.full(length + 1, infinity, dtype=np.float32)
    previous[0] = 0.0
    for first_index in range(1, length + 1):
        current.fill(infinity)
        lower = max(1, first_index - band)
        upper = min(length, first_index + band)
        for second_index in range(lower, upper + 1):
            cost = (first[first_index - 1] - second[second_index - 1]) ** 2
            current[second_index] = cost + min(
                previous[second_index],
                current[second_index - 1],
                previous[second_index - 1],
            )
        previous, current = current, previous
    distance = float(np.sqrt(previous[length] / length))
    return distance


def build_wind_dtw_relation(
    panel: np.ndarray,
    train_end_index: int,
    power_index: int,
    top_k: int,
    candidate_k: int,
    downsample: int,
    band: int,
) -> torch.Tensor:
    """只用训练段功率轨迹构建风机 DTW 关系先验。"""
    train_power = panel[:, :train_end_index, power_index]
    standardized = standardize_trajectories(train_power)
    trajectories = downsample_trajectories(standardized, downsample)
    turbine_count = trajectories.shape[0]
    candidate_count = min(candidate_k, turbine_count)
    relation_count = min(top_k, turbine_count - 1)

    # 先用欧氏距离缩小 DTW 的候选集，与原 SATRA 管线一致。
    squared_distances = (trajectories[:, None, :] - trajectories[None, :, :]) ** 2
    euclidean_distances = squared_distances.sum(axis=-1)
    candidate_indices = np.argsort(euclidean_distances, axis=1)[:, :candidate_count]
    relation = np.zeros((turbine_count, turbine_count), dtype=np.uint8)

    # 对每台风机选择训练期历史 DTW 最相近的邻机。
    for turbine_index in range(turbine_count):
        scored = []
        for candidate_index in candidate_indices[turbine_index]:
            if candidate_index == turbine_index:
                continue
            distance = dtw_distance(
                trajectories[turbine_index],
                trajectories[candidate_index],
                band,
            )
            scored.append((distance, candidate_index))
        scored.sort(key=lambda item: item[0])
        for _, candidate_index in scored[:relation_count]:
            relation[turbine_index, candidate_index] = 1

    # 保持原始 PSTR-Net 的无向历史相似邻域定义。
    relation = np.maximum(relation, relation.T)
    matrix = torch.from_numpy(relation)
    return matrix
