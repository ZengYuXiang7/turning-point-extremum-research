import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from models.single import select_backbone
from models.single_impl.adaptive_revin import AdaptiveRevIN
from models.single_impl.time_moe_arevin import TimeMoEAdaptiveRevIN
from tasks.single.result import select_model_metadata


class FakeDecoder(nn.Module):
    def __init__(self, hidden_size: int = 24) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(2)]
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, inputs_embeds, **kwargs):
        hidden = inputs_embeds
        for layer in self.layers:
            hidden = hidden + torch.tanh(layer(hidden))
        return SimpleNamespace(last_hidden_state=self.norm(hidden))


class FakeTimeMoE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=24,
            max_position_embeddings=32,
            use_cache=True,
        )
        self.decoder = FakeDecoder()

    def get_decoder(self):
        return self.decoder

    def gradient_checkpointing_enable(self):
        return None


class TestAdaptiveRevIN(unittest.TestCase):
    def test_identity_parameters_recover_plain_revin(self):
        torch.manual_seed(2026)
        layer = AdaptiveRevIN(horizon=4)
        history = torch.randn(2, 8, 3)
        prediction = torch.randn(2, 4)
        _, state = layer.normalize(history)

        actual = layer.denormalize_feature(prediction, state, feature_index=1)
        expected = state.std[:, 0, 1:2] * prediction + state.mean[:, 0, 1:2]

        torch.testing.assert_close(actual, expected)
        self.assertAlmostEqual(float(layer.gate.detach()), 0.5, places=6)


class TestTimeMoEAdaptiveRevIN(unittest.TestCase):
    def build_model(self):
        with patch(
            "models.single_impl.time_moe_arevin.AutoModelForCausalLM.from_pretrained",
            return_value=FakeTimeMoE(),
        ):
            model = TimeMoEAdaptiveRevIN(
                horizon=3,
                channels=19,
                power_index=18,
                entity_count=16,
                bottleneck=12,
                unfreeze_layers=1,
            )
        return model

    def test_target_project_forward_contract_and_gradients(self):
        torch.manual_seed(2026)
        model = self.build_model()
        prediction = model(
            torch.randn(2, 16, 19),
            torch.tensor([0, 15]),
        )
        self.assertEqual(tuple(prediction.shape), (2, 3))
        self.assertTrue(torch.isfinite(prediction).all())

        prediction.square().mean().backward()
        self.assertIsNotNone(model.input_embedding.value_projection.weight.grad)
        self.assertIsNotNone(model.adaptive_revin.shift_correction.grad)

    def test_result_metadata_identifies_time_moe(self):
        feature_fusion, kernel, temporal_projection, purpose = (
            select_model_metadata("TimeMoEARevIN")
        )
        self.assertEqual(feature_fusion, "gated_numeric_embedding")
        self.assertIsNone(kernel)
        self.assertEqual(temporal_projection, "time_moe_last_token")
        self.assertIn("TimeMoE", purpose)

    def test_single_selector_rejects_unsupported_modes(self):
        repository = SimpleNamespace(
            feature_names=list(range(19)),
            power_index=18,
            series=[None] * 16,
        )
        common = {
            "model": "TimeMoEARevIN",
            "horizon_steps": 1,
            "pretrain": False,
        }
        with self.assertRaisesRegex(ValueError, "NoFutureWeather"):
            select_backbone(
                SimpleNamespace(**common, scenario="OracleFutureWeather"),
                repository,
            )
        with self.assertRaisesRegex(ValueError, "--pretrain"):
            select_backbone(
                SimpleNamespace(
                    **{**common, "pretrain": True},
                    scenario="NoFutureWeather",
                ),
                repository,
            )


if __name__ == "__main__":
    unittest.main()
