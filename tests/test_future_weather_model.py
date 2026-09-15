import unittest

import torch

from config import HISTORY_COLUMNS, HISTORY_STEPS, WEATHER_COLUMNS
from models.single_impl.future_weather_model import FutureWeatherModel


class TestFutureWeatherModel(unittest.TestCase):
    def test_dlinear_future_weather_forward(self):
        horizon = 3
        model = FutureWeatherModel("DLinear", horizon=horizon)
        past = torch.randn(2, HISTORY_STEPS, len(HISTORY_COLUMNS))
        future_weather = torch.randn(2, horizon, len(WEATHER_COLUMNS))
        turbine_id = torch.tensor([0, 1], dtype=torch.long)

        prediction = model(past, future_weather, turbine_id)

        self.assertEqual(tuple(prediction.shape), (2, horizon))


if __name__ == "__main__":
    unittest.main()
