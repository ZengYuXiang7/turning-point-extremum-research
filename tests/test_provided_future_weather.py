import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from models.single import SingleModel
from tasks.single.provided_future_weather_dataset import (
    ProvidedFutureWeatherDataset,
    ProvidedFutureWeatherRepository,
)


class TestProvidedFutureWeather(unittest.TestCase):
    def test_dataset_and_model_use_supplied_forecast_wind(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            point_count = 100
            values = np.linspace(1.0, 10.0, point_count)
            forecast_wind = values.copy()
            forecast_wind[10] = np.nan
            table = pd.DataFrame(
                {
                    "时间": pd.date_range("2026-05-01", periods=point_count, freq="15min"),
                    "场站名": ["广宁风电场"] * point_count,
                    "超短期预测功率(MW)": values,
                    "实际功率(MW)": values + 1.0,
                    "风电理论功率(MW)": values + 2.0,
                    "预测风速(m/s)": forecast_wind,
                    "实际风速(m/s)": values / 2.0,
                    "风电可用功率(MW)": values + 3.0,
                    "风电电网调令(MW)": values + 4.0,
                }
            )
            table.to_excel(root / "yc_fixture.xlsx", sheet_name="预测数据", index=False)

            with (
                patch("config.HISTORY_STEPS", 4),
                patch("config.WINDOW_STRIDE_STEPS", 1),
            ):
                repository = ProvidedFutureWeatherRepository(root)
                dataset = ProvidedFutureWeatherDataset(repository, "train", horizon=2)
                past, future_weather, target, site_id, _ = dataset[0]
                args = SimpleNamespace(scenario="ProvidedFutureWeather", model="DLinear", horizon_steps=2,)
                model = SingleModel(args, repository).eval()
                with torch.no_grad():
                    prediction = model(past.unsqueeze(0), future_weather.unsqueeze(0), site_id.reshape(1),)
                    changed_future_weather = future_weather + 1.0
                    changed_prediction = model(past.unsqueeze(0), changed_future_weather.unsqueeze(0), site_id.reshape(1),)

        self.assertEqual(repository.missing_forecast_points, 1)
        self.assertEqual(dataset.excluded_windows, 2)
        self.assertEqual(tuple(past.shape), (4, 6))
        self.assertEqual(tuple(future_weather.shape), (2, 1))
        self.assertEqual(tuple(target.shape), (2,))
        self.assertEqual(tuple(prediction.shape), (1, 2))
        self.assertTrue(torch.isfinite(prediction).all())
        self.assertFalse(torch.allclose(prediction, changed_prediction))
        self.assertAlmostEqual(repository.series[0].raw[0, 0], 1_000.0)


if __name__ == "__main__":
    unittest.main()
