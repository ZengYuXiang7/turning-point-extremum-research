import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from config import HISTORY_COLUMNS, WEATHER_INDICES
from tasks.multi_turbine.dataset import (
    MultiTurbineNoFutureWeatherDataset,
    MultiTurbineOracleFutureWeatherDataset,
    stack_multi_turbine_panel,
)


class TestMultiTurbineRelationDataset(unittest.TestCase):
    def test_stack_multi_turbine_panel(self):
        sequence_length = 12
        feature_count = 3
        series = []
        for turbine_index in range(16):
            scaled = np.full((sequence_length, feature_count), turbine_index, dtype=np.float32,)
            series.append(SimpleNamespace(scaled=scaled))
        repository = SimpleNamespace(series=series)

        panel = stack_multi_turbine_panel(repository)

        self.assertEqual(tuple(panel.shape), (16, sequence_length, feature_count))
        for turbine_index in range(16):
            np.testing.assert_array_equal(panel[turbine_index], turbine_index)
            self.assertTrue(np.shares_memory(repository.series[turbine_index].scaled, panel))

    def test_history_dataset_slices_the_joint_panel(self):
        history_steps = 3
        horizon_steps = 2
        point_stride_steps = 2
        sequence_length = (history_steps + horizon_steps) * point_stride_steps
        feature_count = len(HISTORY_COLUMNS)
        panel = np.arange(16 * sequence_length * feature_count, dtype=np.float32,).reshape(16, sequence_length, feature_count)
        times = np.arange(sequence_length, dtype=np.int64) * 10_000_000_000
        series = []
        for turbine_index in range(16):
            turbine_series = SimpleNamespace(times=times, scaled=panel[turbine_index],)
            series.append(turbine_series)
        repository = SimpleNamespace(panel=panel, times=times, series=series, power_index=feature_count - 1,)

        with (
            patch("config.HISTORY_STEPS", history_steps,),
            patch("config.POINT_STRIDE_STEPS", point_stride_steps,),
            patch("tasks.multi_turbine.dataset.build_multi_turbine_windows", return_value=[0],),
        ):
            dataset = MultiTurbineNoFutureWeatherDataset(repository, "train", horizon_steps,)
            past, target, turbine_id, target_start_ns = dataset[0]

        target_start = history_steps * point_stride_steps
        target_end = target_start + horizon_steps * point_stride_steps
        expected_past = panel[:, :target_start:point_stride_steps]
        expected_target = panel[
            :, target_start:target_end:point_stride_steps, feature_count - 1
        ]
        np.testing.assert_array_equal(past.numpy(), expected_past)
        np.testing.assert_array_equal(target.numpy(), expected_target)
        np.testing.assert_array_equal(turbine_id.numpy(), np.arange(16))
        self.assertEqual(tuple(past.shape), (16, history_steps, feature_count))
        self.assertEqual(tuple(target.shape), (16, horizon_steps))
        self.assertEqual(target_start_ns.item(), times[target_start])

    def test_oracle_dataset_slices_weather_from_the_joint_panel(self):
        history_steps = 3
        horizon_steps = 2
        point_stride_steps = 2
        sequence_length = (history_steps + horizon_steps) * point_stride_steps
        feature_count = len(HISTORY_COLUMNS)
        panel = np.arange(16 * sequence_length * feature_count, dtype=np.float32,).reshape(16, sequence_length, feature_count)
        times = np.arange(sequence_length, dtype=np.int64) * 10_000_000_000
        series = []
        for turbine_index in range(16):
            turbine_series = SimpleNamespace(times=times, scaled=panel[turbine_index],)
            series.append(turbine_series)
        repository = SimpleNamespace(panel=panel, times=times, series=series, power_index=feature_count - 1,)

        with (
            patch("config.HISTORY_STEPS", history_steps,),
            patch("config.POINT_STRIDE_STEPS", point_stride_steps,),
            patch("tasks.multi_turbine.dataset.build_multi_turbine_windows", return_value=[0],),
        ):
            dataset = MultiTurbineOracleFutureWeatherDataset(repository, "train", horizon_steps,)
            past, weather, target, turbine_id, target_start_ns = dataset[0]

        target_start = history_steps * point_stride_steps
        target_end = target_start + horizon_steps * point_stride_steps
        expected_weather = panel[
            :, target_start:target_end:point_stride_steps
        ][:, :, WEATHER_INDICES]
        np.testing.assert_array_equal(weather.numpy(), expected_weather)
        self.assertEqual(tuple(past.shape), (16, history_steps, feature_count))
        self.assertEqual(tuple(weather.shape), (16, horizon_steps, len(WEATHER_INDICES)),)
        self.assertEqual(tuple(target.shape), (16, horizon_steps))
        np.testing.assert_array_equal(turbine_id.numpy(), np.arange(16))
        self.assertEqual(target_start_ns.item(), times[target_start])


if __name__ == "__main__":
    unittest.main()
