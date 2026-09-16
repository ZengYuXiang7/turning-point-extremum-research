import unittest
from unittest.mock import patch

import numpy as np
import torch

from models.multi_turbine_impl.multi_turbine import MultiTurbineBackbone, MultiTurbineRevIN
from tasks.multi_turbine.configuration import parse_args
from tasks.multi_turbine.pretrain import MaskedMarketReconstruction
from tasks.multi_turbine.relation_prior import build_wind_dtw_relation


def make_relation(turbine_count: int) -> torch.Tensor:
    relation = torch.eye(turbine_count, dtype=torch.uint8)
    turbine_indices = torch.arange(turbine_count)
    neighbor_indices = (turbine_indices + 1) % turbine_count
    relation[turbine_indices, neighbor_indices] = 1
    return relation


class TestMultiTurbine(unittest.TestCase):
    def make_model(self, channels: int = 5, horizon: int = 3, use_revin: bool = True, use_dtw_prior: bool = False,) -> MultiTurbineBackbone:
        model = MultiTurbineBackbone(channels=channels, power_index=channels - 1, history_steps=8, horizon=horizon, use_revin=use_revin, relation_weight=make_relation(4), use_dtw_prior=use_dtw_prior, hidden_dim=16, depth=1, heads=4, dropout=0.0,)
        return model

    def test_revin_restores_each_turbine_power_independently(self):
        revin = MultiTurbineRevIN(channels=3)
        past = torch.tensor([[[[1.0, 10.0, 100.0], [3.0, 20.0, 200.0]], [[5.0, 30.0, 1000.0], [9.0, 60.0, 2000.0]]]])

        normalized, instance_mean, instance_std = revin.normalize(past)
        restored_power = revin.denormalize_power(normalized[..., 2], instance_mean, instance_std, power_index=2,)

        torch.testing.assert_close(restored_power, past[..., 2])

    def test_revin_argument_controls_multi_turbine_backbone(self):
        command = ["run_multi.py", "--model", "MultiTurbine", "--scenario", "MultiTurbine", "--loss", "MSE"]
        with patch("sys.argv", command):
            enabled_args = parse_args()
        with patch("sys.argv", command + ["--revin", "0"]):
            disabled_args = parse_args()

        self.assertEqual(enabled_args.revin, 1)
        self.assertEqual(disabled_args.revin, 0)
        self.assertTrue(self.make_model(use_revin=True).use_revin)
        self.assertFalse(self.make_model(use_revin=False).use_revin)

    def test_multi_turbine_forecasts_the_joint_panel(self):
        model = self.make_model()
        past = torch.randn(2, 4, 8, 5)
        turbine_id = torch.arange(4).unsqueeze(0).expand(2, -1)

        prediction = model(past, turbine_id)
        loss = prediction.square().mean()
        loss.backward()

        self.assertEqual(tuple(prediction.shape), (2, 4, 3))
        self.assertTrue(torch.isfinite(prediction).all())
        self.assertGreater(float(model.input_projection.weight.grad.abs().sum()), 0.0)

    def test_dtw_prior_is_disabled_by_default(self):
        model = self.make_model()

        self.assertFalse(model.use_dtw_prior)
        self.assertEqual(len(model.prior_towers), 0)
        self.assertFalse(hasattr(model, "prior_attention_mask"))

    def test_dtw_prior_can_be_enabled(self):
        model = self.make_model(use_dtw_prior=True)
        past = torch.randn(2, 4, 8, 5)
        turbine_id = torch.arange(4).unsqueeze(0).expand(2, -1)

        prediction = model(past, turbine_id)

        self.assertTrue(model.use_dtw_prior)
        self.assertEqual(len(model.prior_towers), 1)
        self.assertTrue(hasattr(model, "prior_attention_mask"))
        self.assertEqual(tuple(prediction.shape), (2, 4, 3))

    def test_masked_market_reconstruction_hides_masked_values(self):
        model = self.make_model()
        pretrainer = MaskedMarketReconstruction(model)
        pretrainer.eval()
        past = torch.randn(1, 4, 8, 5)
        token_mask = torch.zeros(1, 4, 8, dtype=torch.bool)
        token_mask[:, :, 2] = True
        altered = past.clone()
        altered[token_mask] += 1000.0

        with torch.no_grad():
            reconstruction = pretrainer(past, token_mask)
            altered_reconstruction = pretrainer(altered, token_mask)

        torch.testing.assert_close(reconstruction, altered_reconstruction)

    def test_dtw_prior_uses_training_prefix_only(self):
        generator = np.random.default_rng(2026)
        panel = generator.normal(size=(4, 20, 5)).astype(np.float32)
        changed_panel = panel.copy()
        changed_panel[:, 12:, -1] += 1000.0

        first_relation = build_wind_dtw_relation(panel, train_end_index=12, power_index=4, top_k=2, candidate_k=3, downsample=8, band=2,)
        second_relation = build_wind_dtw_relation(changed_panel, train_end_index=12, power_index=4, top_k=2, candidate_k=3, downsample=8, band=2,)

        torch.testing.assert_close(first_relation, second_relation)


if __name__ == "__main__":
    unittest.main()
