import unittest
from unittest.mock import patch

import torch
from torch import nn

from models.multi_turbine import MultiTurbineModel
from models.single import SingleModel


class EchoBackbone(nn.Module):
    def forward(self, *inputs):
        return inputs


class TestModelEntrypoints(unittest.TestCase):
    def test_single_model_delegates_to_its_backbone(self):
        backbone = EchoBackbone()
        with patch("models.single.select_backbone", return_value=backbone):
            model = SingleModel(object(), object())
        first = torch.tensor([1.0])
        second = torch.tensor([2.0])

        output = model(first, second)

        self.assertIs(model.backbone, backbone)
        self.assertEqual(output, (first, second))

    def test_multi_turbine_model_delegates_to_its_backbone(self):
        backbone = EchoBackbone()
        with patch("models.multi_turbine.select_backbone", return_value=backbone):
            model = MultiTurbineModel(object(), object(), None)
        panel = torch.tensor([1.0])
        turbine_id = torch.tensor([0])

        output = model(panel, turbine_id)

        self.assertIs(model.backbone, backbone)
        self.assertEqual(output, (panel, turbine_id))


if __name__ == "__main__":
    unittest.main()
