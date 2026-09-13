import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from config.settings import (
    HISTORY_COLUMNS,
    HISTORY_STEPS,
    POWER_INDEX,
    SAMPLE_SECONDS,
)
from data_provider.no_future_weather_dataset import NoFutureWeatherDataset
from exp.trainer import print_data_summary
from models.backbone.stockecho import StockEchoNoFutureWeather
from models.noweather import NoFutureWeatherModel, RevIN as NoWeatherRevIN


class SummaryDataset(Dataset):
    def __init__(self, start_ns: int) -> None:
        self.start_ns = start_ns
        self.repository = SimpleNamespace(feature_names=HISTORY_COLUMNS)

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int):
        past = torch.zeros(HISTORY_STEPS, len(HISTORY_COLUMNS))
        target = torch.zeros(1)
        turbine_id = torch.tensor(0, dtype=torch.long)
        target_start_ns = torch.tensor(
            self.start_ns + index * SAMPLE_SECONDS * 1_000_000_000,
            dtype=torch.long,
        )
        return past, target, turbine_id, target_start_ns


class TestNoFutureWeather(unittest.TestCase):
    def test_dataset_returns_no_future_weather(self):
        steps = HISTORY_STEPS + 1
        times = np.arange(steps, dtype=np.int64) * SAMPLE_SECONDS * 1_000_000_000
        scaled = np.zeros((steps, len(HISTORY_COLUMNS)), dtype=np.float32)
        series = SimpleNamespace(times=times, scaled=scaled)
        repository = SimpleNamespace(series=[series], power_index=POWER_INDEX)

        with patch(
            "data_provider.no_future_weather_dataset.build_single_turbine_windows",
            return_value=[(0, 0)],
        ):
            dataset = NoFutureWeatherDataset(repository, "train", 1)

        past, target, turbine_id, target_start_ns = dataset[0]
        self.assertEqual(tuple(past.shape), (HISTORY_STEPS, len(HISTORY_COLUMNS)))
        self.assertEqual(tuple(target.shape), (1,))
        self.assertEqual(turbine_id.item(), 0)
        self.assertEqual(target_start_ns.item(), times[HISTORY_STEPS])

    def test_summary_has_no_future_weather_shape(self):
        train_start = np.datetime64("2026-06-01T00:00:00", "ns").astype(np.int64)
        val_start = np.datetime64("2026-06-08T00:00:00", "ns").astype(np.int64)
        test_start = np.datetime64("2026-06-11T00:00:00", "ns").astype(np.int64)
        datasets = {
            "train": SummaryDataset(train_start),
            "val": SummaryDataset(val_start),
            "test": SummaryDataset(test_start),
        }
        loaders = {}
        for split in datasets:
            loaders[split] = DataLoader(datasets[split], batch_size=2)

        output = io.StringIO()
        with redirect_stdout(output):
            print_data_summary(
                datasets,
                loaders,
                horizon_steps=1,
                window_stride_steps=1,
                no_future_weather=True,
                weather_task=False,
                oracle=False,
                predicted_weather=False,
            )

        text = output.getvalue()
        self.assertIn(
            f"输入  past shape=[2, {HISTORY_STEPS}, 19]  turbine_id shape=[2]",
            text,
        )
        self.assertNotIn("future_weather shape", text)
        self.assertIn(
            f"历史长度={HISTORY_STEPS * SAMPLE_SECONDS / 60:g}分钟 "
            "预测长度=15分钟 滑窗步长=15分钟",
            text,
        )
        self.assertIn(
            f"时间维=1点（每点{SAMPLE_SECONDS / 60:g}分钟）",
            text,
        )

    def test_dlinear_history_only_forward(self):
        model = NoFutureWeatherModel(
            "DLinear",
            horizon=1,
            channels=len(HISTORY_COLUMNS),
            power_index=POWER_INDEX,
        )
        past = torch.randn(2, HISTORY_STEPS, len(HISTORY_COLUMNS))
        turbine_id = torch.tensor([0, 1], dtype=torch.long)

        prediction = model(past, turbine_id)

        self.assertEqual(tuple(prediction.shape), (2, 1))

    def test_patchmlp_history_only_forward(self):
        model = NoFutureWeatherModel(
            "PatchMLP",
            horizon=1,
            channels=len(HISTORY_COLUMNS),
            power_index=POWER_INDEX,
        )
        past = torch.randn(2, HISTORY_STEPS, len(HISTORY_COLUMNS))
        turbine_id = torch.tensor([0, 1], dtype=torch.long)

        prediction = model(past, turbine_id)

        self.assertEqual(tuple(prediction.shape), (2, 1))

    def test_noweather_revin_restores_features(self):
        revin = NoWeatherRevIN(len(HISTORY_COLUMNS))
        past = torch.randn(2, HISTORY_STEPS, len(HISTORY_COLUMNS))

        normalized, instance_mean, instance_std = revin.normalize(past)
        restored = revin.denormalize(normalized, instance_mean, instance_std)

        torch.testing.assert_close(restored, past)

    def test_dlinear_uses_turbine_embedding(self):
        model = NoFutureWeatherModel(
            "DLinear",
            horizon=1,
            channels=len(HISTORY_COLUMNS),
            power_index=POWER_INDEX,
        )
        history = torch.randn(1, HISTORY_STEPS, len(HISTORY_COLUMNS))
        past = history.expand(2, -1, -1).clone()
        turbine_id = torch.tensor([0, 1], dtype=torch.long)

        with torch.no_grad():
            model.turbine_embedding.weight.zero_()
            model.turbine_embedding.weight[1, POWER_INDEX] = 1.0
            prediction = model(past, turbine_id)

        self.assertFalse(torch.allclose(prediction[0], prediction[1]))

    def test_patchmlp_uses_turbine_embedding(self):
        model = NoFutureWeatherModel(
            "PatchMLP",
            horizon=1,
            channels=len(HISTORY_COLUMNS),
            power_index=POWER_INDEX,
        )
        model.eval()
        history = torch.randn(1, HISTORY_STEPS, len(HISTORY_COLUMNS))
        past = history.expand(2, -1, -1).clone()
        turbine_id = torch.tensor([0, 1], dtype=torch.long)

        with torch.no_grad():
            model.turbine_embedding.weight.zero_()
            embedding_width = model.turbine_embedding.embedding_dim
            model.turbine_embedding.weight[1] = torch.linspace(
                -1.0,
                1.0,
                embedding_width,
            )
            prediction = model(past, turbine_id)

        self.assertFalse(torch.allclose(prediction[0], prediction[1]))

    def test_dlinear_all_features_mixes_non_power_channel(self):
        model = NoFutureWeatherModel(
            "DLinearAllFeatures",
            horizon=1,
            channels=len(HISTORY_COLUMNS),
            power_index=POWER_INDEX,
        )
        model.eval()
        history = torch.randn(1, HISTORY_STEPS, len(HISTORY_COLUMNS))
        past = history.expand(2, -1, -1).clone()
        past[1, :, 0] = torch.linspace(-2.0, 3.0, HISTORY_STEPS)
        turbine_id = torch.zeros(2, dtype=torch.long)

        with torch.no_grad():
            model.turbine_embedding.weight.zero_()
            model.feature_projection.weight.zero_()
            model.feature_projection.bias.zero_()
            model.feature_projection.weight[0, 0] = 1.0
            prediction = model(past, turbine_id)

        self.assertFalse(torch.allclose(prediction[0], prediction[1]))

    def test_stockecho_history_only_forward(self):
        model = StockEchoNoFutureWeather(horizon=1)
        model.eval()
        past = torch.randn(1, 16, HISTORY_STEPS, len(HISTORY_COLUMNS))
        turbine_id = torch.arange(16, dtype=torch.long).unsqueeze(0)

        with torch.no_grad():
            prediction = model(past, turbine_id)

        self.assertEqual(tuple(prediction.shape), (1, 16, 1))

if __name__ == "__main__":
    unittest.main()
