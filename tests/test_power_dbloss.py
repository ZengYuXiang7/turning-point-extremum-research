import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from config import HISTORY_STEPS
from models.single_impl.no_future_model import NoFutureWeatherModel
from tasks.single.epoch import run_epoch
from utils.dbloss import DBLoss


class PowerSequenceDataset(Dataset):
    def __init__(self, target: torch.Tensor) -> None:
        self.target = target

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int):
        past = torch.zeros(2, 3)
        turbine_id = torch.tensor(0, dtype=torch.long)
        target_start_ns = torch.tensor(0, dtype=torch.long)
        return past, self.target, turbine_id, target_start_ns


class FixedPowerModel(nn.Module):
    def __init__(self, prediction: torch.Tensor) -> None:
        super().__init__()
        self.prediction = nn.Parameter(prediction)

    def forward(self, past: torch.Tensor, turbine_id: torch.Tensor,) -> torch.Tensor:
        prediction = self.prediction.unsqueeze(0).expand(past.shape[0], -1)
        return prediction


class TestPowerDBLoss(unittest.TestCase):
    def test_power_dbloss_weight_adds_to_power_mse(self):
        prediction = torch.tensor([0.0, 1.0, 2.0, 3.0])
        target = torch.tensor([1.0, 0.0, 3.0, 2.0])
        model = FixedPowerModel(prediction)
        loader = DataLoader(PowerSequenceDataset(target), batch_size=1)
        objective = DBLoss(alpha=0.2, beta=0.5)
        weight = 0.5

        expected_mse = torch.mean((prediction - target) ** 2)
        expected_loss = expected_mse + weight * objective(prediction.unsqueeze(0), target.unsqueeze(0),)
        loss, mse, _ = run_epoch(model, loader, torch.device("cpu"), None, objective, weight, False, torch.tensor([200.0]), torch.tensor([1.0]), 0, 1, True, False, False,)

        self.assertAlmostEqual(loss, expected_loss.item(), places=6)
        self.assertAlmostEqual(mse, expected_mse.item(), places=6)

    def test_single_step_power_dbloss_keeps_mse_gradient(self):
        model = FixedPowerModel(torch.tensor([0.0]))
        loader = DataLoader(PowerSequenceDataset(torch.tensor([1.0])), batch_size=1)
        objective = DBLoss(alpha=0.2, beta=0.5)

        loss, mse, _ = run_epoch(model, loader, torch.device("cpu"), torch.optim.SGD(model.parameters(), lr=0.1), objective, 0.5, True, torch.tensor([200.0]), torch.tensor([1.0]), 0, 1, True, False, False,)

        self.assertAlmostEqual(loss, 1.0, places=6)
        self.assertAlmostEqual(mse, 1.0, places=6)
        self.assertNotEqual(model.prediction.item(), 0.0)

    def test_patchmlp_all_features_forward(self):
        channels = 59
        power_index = 58
        model = NoFutureWeatherModel("PatchMLPAllFeatures", horizon=3, channels=channels, power_index=power_index,)
        past = torch.randn(2, HISTORY_STEPS, channels)
        turbine_id = torch.tensor([0, 1], dtype=torch.long)

        prediction = model(past, turbine_id)

        self.assertEqual(tuple(prediction.shape), (2, 3))


if __name__ == "__main__":
    unittest.main()
