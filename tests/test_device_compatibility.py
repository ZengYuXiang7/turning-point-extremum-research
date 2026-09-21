import unittest
from unittest.mock import patch

import torch

from tasks.single.training_setup import prepare_task_runtime
from utils.amp import autocast_context


class TestDeviceCompatibility(unittest.TestCase):
    def test_mps_uses_plain_precision_contexts(self):
        device = torch.device("mps")
        with autocast_context(device):
            value = torch.tensor([1.0]) + 1.0
        self.assertEqual(value.item(), 2.0)

    def test_only_time_moe_opt_in_enables_mps_selection(self):
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=True),
        ):
            self.assertEqual(prepare_task_runtime(2026, allow_mps=False).type, "cpu")
            self.assertEqual(prepare_task_runtime(2026, allow_mps=True).type, "mps")


if __name__ == "__main__":
    unittest.main()
