import unittest

import torch

from models.multi_turbine_impl.model2 import Model2Backbone, PatchHistoryEncoder


class TestModel2(unittest.TestCase):
    def test_patch_encoder_restores_history_shape_and_routes_moe(self):
        encoder = PatchHistoryEncoder(hidden_dim=8, history_steps=11, patch_length=4, patch_stride=3, depth=1, kernel_size=3, top_k=3, dropout=0.0,)
        representation = torch.randn(2, 4, 11, 8)

        output = encoder(representation)
        output.square().mean().backward()

        self.assertEqual(tuple(output.shape), (2, 4, 11, 8))
        gate_gradient = encoder.patch_encoder.experts.gate_weights.grad
        self.assertIsNotNone(gate_gradient)
        self.assertTrue(torch.isfinite(output).all())

    def test_model2_forecasts_joint_panel(self):
        model = Model2Backbone(channels=5, power_index=4, history_steps=10, horizon=3, use_revin=True, patch_length=4, patch_stride=3, use_spatial_relation=True, hidden_dim=8, depth=1, heads=2, dropout=0.0,)
        past = torch.randn(2, 4, 10, 5)
        turbine_id = torch.arange(4).unsqueeze(0).expand(2, -1)

        prediction = model(past, turbine_id)

        self.assertEqual(tuple(prediction.shape), (2, 4, 3))
        self.assertTrue(torch.isfinite(prediction).all())


if __name__ == "__main__":
    unittest.main()
