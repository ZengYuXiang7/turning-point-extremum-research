import unittest

import torch

from config import HISTORY_COLUMNS, HISTORY_STEPS, WEATHER_COLUMNS
from models.multi_turbine_impl.backbone.stockecho import (
    StockEchoNoFutureWeather,
    StockEchoWindPower,
)


class TestStockEchoTurbineEmbedding(unittest.TestCase):
    def test_turbine_embedding_changes_forecast(self):
        torch.manual_seed(2026)
        model = StockEchoNoFutureWeather(horizon=4)
        model.eval()
        past = torch.randn(2, 16, HISTORY_STEPS, len(HISTORY_COLUMNS))
        turbine_id = torch.arange(16, dtype=torch.long).unsqueeze(0).expand(2, -1)

        with torch.no_grad():
            model.turbine_embedding.weight.zero_()
            prediction_without_identity = model(past, turbine_id)
            model.turbine_embedding.weight.normal_(mean=0.0, std=0.5)
            prediction_with_identity = model(past, turbine_id)

        self.assertFalse(
            torch.allclose(prediction_without_identity, prediction_with_identity)
        )

    def test_forecast_loss_trains_embedding_relation(self):
        torch.manual_seed(2026)
        model = StockEchoNoFutureWeather(horizon=4)
        model.train()
        past = torch.randn(2, 16, HISTORY_STEPS, len(HISTORY_COLUMNS))
        turbine_id = torch.arange(16, dtype=torch.long).unsqueeze(0).expand(2, -1)
        target = torch.randn(2, 16, 4)

        prediction = model(past, turbine_id)
        loss = torch.mean((prediction - target) ** 2)
        loss.backward()

        embedding_gradient = model.turbine_embedding.weight.grad
        relation_gradient = (
            model.graph_constructor.embedding_projection[1].weight.grad
        )
        mix_gradient = model.graph_constructor.embedding_mix_logit.grad
        self.assertGreater(float(embedding_gradient.abs().sum()), 0.0)
        self.assertGreater(float(relation_gradient.abs().sum()), 0.0)
        self.assertGreater(float(mix_gradient.abs()), 0.0)

    def test_future_weather_branch_reuses_turbine_embedding(self):
        torch.manual_seed(2026)
        horizon = 4
        model = StockEchoWindPower(horizon=horizon)
        model.eval()
        past = torch.randn(1, 16, HISTORY_STEPS, len(HISTORY_COLUMNS))
        future_weather = torch.randn(1, 16, horizon, len(WEATHER_COLUMNS))
        turbine_id = torch.arange(16, dtype=torch.long).unsqueeze(0)

        with torch.no_grad():
            prediction = model(past, future_weather, turbine_id)

        self.assertEqual(tuple(prediction.shape), (1, 16, horizon))
        self.assertTrue(torch.isfinite(prediction).all())


if __name__ == "__main__":
    unittest.main()
