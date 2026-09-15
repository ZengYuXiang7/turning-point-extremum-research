import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from data_provider.common import build_multi_turbine_windows, build_single_turbine_windows


class TestWindowStride(unittest.TestCase):
    def setUp(self):
        start = np.datetime64("2026-06-08T00:00:00", "ns").astype(np.int64)
        step_ns = 10 * 1_000_000_000
        times = start + np.arange(1_200, dtype=np.int64) * step_ns
        series = SimpleNamespace(times=times)
        self.repository = SimpleNamespace(series=[series], train_end=int(times[0]), valid_end=int(times[-1] + step_ns),)

    def test_single_turbine_windows_align_to_quarter_hour(self):
        with (
            patch("config.HISTORY_STEPS", 6),
            patch("config.POINT_STRIDE_STEPS", 90),
            patch("data_provider.window_builder.EXPECTED_DELTA_NS", 10 * 1_000_000_000),
            patch("config.WINDOW_STRIDE_STEPS", 1),
            patch("config.WINDOW_STRIDE_NS", 15 * 60 * 1_000_000_000),
        ):
            windows = build_single_turbine_windows(self.repository, "val", horizon=6)

        target_starts = []
        for _, history_start in windows:
            target_index = history_start + 6 * 90
            target_starts.append(self.repository.series[0].times[target_index])

        target_starts = np.asarray(target_starts)
        quarter_hour_ns = 15 * 60 * 1_000_000_000
        np.testing.assert_array_equal(target_starts % quarter_hour_ns, 0)
        np.testing.assert_array_equal(np.diff(target_starts), quarter_hour_ns)

    def test_multi_turbine_windows_align_to_quarter_hour(self):
        with (
            patch("config.HISTORY_STEPS", 6),
            patch("config.POINT_STRIDE_STEPS", 90),
            patch("data_provider.window_builder.EXPECTED_DELTA_NS", 10 * 1_000_000_000),
            patch("config.WINDOW_STRIDE_STEPS", 1),
            patch("config.WINDOW_STRIDE_NS", 15 * 60 * 1_000_000_000),
        ):
            windows = build_multi_turbine_windows(self.repository, "val", horizon=6)

        target_starts = []
        for history_start in windows:
            target_index = history_start + 6 * 90
            target_starts.append(self.repository.series[0].times[target_index])

        target_starts = np.asarray(target_starts)
        quarter_hour_ns = 15 * 60 * 1_000_000_000
        np.testing.assert_array_equal(target_starts % quarter_hour_ns, 0)
        np.testing.assert_array_equal(np.diff(target_starts), quarter_hour_ns)


if __name__ == "__main__":
    unittest.main()
