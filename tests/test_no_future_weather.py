import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from config import (
    BASE_INTERVAL_SECONDS,
    CORRELATED_HISTORY_COLUMNS,
    HISTORY_COLUMNS,
    HISTORY_STEPS,
    POINT_INTERVAL_SECONDS,
    POINT_STRIDE_STEPS,
    POWER_INDEX,
)
from tasks.single.no_future_dataset import NoFutureWeatherDataset
from models.multi_turbine_impl.stockecho import StockEchoNoFutureWeather
from models.single_impl.no_future_model import (
    NoFutureWeatherModel,
    RevIN as NoWeatherRevIN,
)
from tasks.single.summary import print_data_summary


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
            self.start_ns + index * POINT_INTERVAL_SECONDS * 1_000_000_000,
            dtype=torch.long,
        )
        return past, target, turbine_id, target_start_ns


class TestNoFutureWeather(unittest.TestCase):
    def check_point_stride(self, point_stride_steps: int):
        history_steps = 6
        horizon_steps = 3
        raw_steps = (history_steps + horizon_steps) * point_stride_steps
        times = (
            np.arange(raw_steps, dtype=np.int64)
            * BASE_INTERVAL_SECONDS
            * 1_000_000_000
        )
        scaled = np.arange(raw_steps, dtype=np.float32).reshape(-1, 1)
        series = SimpleNamespace(times=times, scaled=scaled)
        repository = SimpleNamespace(series=[series], power_index=0)

        with (
            patch("config.HISTORY_STEPS", history_steps),
            patch(
                "config.POINT_STRIDE_STEPS",
                point_stride_steps,
            ),
            patch(
                "tasks.single.no_future_dataset.build_single_turbine_windows",
                return_value=[(0, 0)],
            ),
        ):
            dataset = NoFutureWeatherDataset(repository, "train", horizon_steps)
            past, target, _, target_start_ns = dataset[0]

        expected_past = np.arange(history_steps) * point_stride_steps
        expected_target = (
            np.arange(history_steps, history_steps + horizon_steps)
            * point_stride_steps
        )
        np.testing.assert_array_equal(past[:, 0].numpy(), expected_past)
        np.testing.assert_array_equal(target.numpy(), expected_target)
        self.assertEqual(tuple(past.shape), (history_steps, 1))
        self.assertEqual(tuple(target.shape), (horizon_steps,))
        self.assertEqual(target_start_ns.item(), times[history_steps * point_stride_steps])

    def test_one_minute_points_use_six_base_steps(self):
        self.check_point_stride(6)

    def test_ten_minute_points_use_sixty_base_steps(self):
        self.check_point_stride(60)

    def test_dataset_returns_no_future_weather(self):
        steps = HISTORY_STEPS * POINT_STRIDE_STEPS + 1
        times = (
            np.arange(steps, dtype=np.int64)
            * BASE_INTERVAL_SECONDS
            * 1_000_000_000
        )
        scaled = np.zeros((steps, len(HISTORY_COLUMNS)), dtype=np.float32)
        scaled[:, 0] = np.arange(steps)
        scaled[:, POWER_INDEX] = np.arange(steps)
        series = SimpleNamespace(times=times, scaled=scaled)
        repository = SimpleNamespace(series=[series], power_index=POWER_INDEX)

        with patch(
            "tasks.single.no_future_dataset.build_single_turbine_windows",
            return_value=[(0, 0)],
        ):
            dataset = NoFutureWeatherDataset(repository, "train", 1)

        past, target, turbine_id, target_start_ns = dataset[0]
        self.assertEqual(tuple(past.shape), (HISTORY_STEPS, len(HISTORY_COLUMNS)))
        self.assertEqual(tuple(target.shape), (1,))
        self.assertEqual(turbine_id.item(), 0)
        selected_indices = np.arange(HISTORY_STEPS) * POINT_STRIDE_STEPS
        np.testing.assert_array_equal(past[:, 0].numpy(), selected_indices)
        self.assertEqual(target.item(), HISTORY_STEPS * POINT_STRIDE_STEPS)
        self.assertEqual(
            target_start_ns.item(), times[HISTORY_STEPS * POINT_STRIDE_STEPS]
        )

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
            f"历史长度={HISTORY_STEPS * POINT_INTERVAL_SECONDS / 60:g}分钟 "
            "预测长度=15分钟 滑窗步长=15分钟",
            text,
        )
        self.assertIn(
            f"时间维=1点（每点{POINT_INTERVAL_SECONDS / 60:g}分钟）",
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

    def test_dlinear_correlated_features_forward(self):
        power_index = CORRELATED_HISTORY_COLUMNS.index("风机-P")
        model = NoFutureWeatherModel(
            "DLinearCorrelatedFeatures",
            horizon=1,
            channels=len(CORRELATED_HISTORY_COLUMNS),
            power_index=power_index,
        )
        past = torch.randn(2, HISTORY_STEPS, len(CORRELATED_HISTORY_COLUMNS))
        turbine_id = torch.tensor([0, 1], dtype=torch.long)

        prediction = model(past, turbine_id)

        self.assertEqual(len(CORRELATED_HISTORY_COLUMNS), 21)
        self.assertIn("机舱-舱内温度", CORRELATED_HISTORY_COLUMNS)
        self.assertNotIn("变流器-功率因数", CORRELATED_HISTORY_COLUMNS)
        self.assertEqual(tuple(prediction.shape), (2, 1))

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
