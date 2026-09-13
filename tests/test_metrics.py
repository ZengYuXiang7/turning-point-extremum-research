import unittest

import numpy as np
import torch

from utils.metrics import dtw, tdi


class TestDTWMetrics(unittest.TestCase):
    def test_identical_sequences_have_zero_tdi(self):
        y_true = np.array([[1, 2, 3, 2, 1]], dtype=np.float64)
        y_pred = np.array([[1, 2, 3, 2, 1]], dtype=np.float64)

        value = tdi(y_true, y_pred)

        self.assertAlmostEqual(value, 0.0)

    def test_delayed_prediction_has_positive_tdi(self):
        y_true = np.array([[0, 1, 3, 1, 0, 0]], dtype=np.float64)
        y_pred = np.array([[0, 0, 1, 3, 1, 0]], dtype=np.float64)

        value = tdi(y_true, y_pred)
        distance = dtw(y_true, y_pred)

        self.assertAlmostEqual(value, 5.0 / 36.0)
        self.assertGreater(value, 0.0)
        self.assertAlmostEqual(distance, 0.0)

    def test_dtw_batch_reductions_support_torch_tensors(self):
        y_true = torch.zeros((2, 4), dtype=torch.float32)
        y_pred = torch.tensor(
            [
                [0, 0, 0, 0],
                [1, 1, 1, 1],
            ],
            dtype=torch.float32,
        )

        per_sequence = dtw(y_true, y_pred, reduction="none")
        mean_value = dtw(y_true, y_pred, reduction="mean")

        np.testing.assert_allclose(per_sequence, np.array([0.0, 1.0]))
        self.assertIsInstance(mean_value, float)
        self.assertAlmostEqual(mean_value, 0.5)

    def test_batch_reductions_support_torch_tensors(self):
        y_true = torch.tensor(
            [
                [1, 2, 3, 2, 1],
                [0, 1, 3, 1, 0],
            ],
            dtype=torch.float32,
        )
        y_pred = torch.tensor(
            [
                [1, 2, 3, 2, 1],
                [0, 0, 1, 3, 1],
            ],
            dtype=torch.float32,
        )

        per_sequence = tdi(y_true, y_pred, reduction="none")
        mean_value = tdi(y_true, y_pred, reduction="mean")

        self.assertEqual(per_sequence.shape, (2,))
        self.assertIsInstance(mean_value, float)
        self.assertAlmostEqual(mean_value, float(np.mean(per_sequence)))

    def test_invalid_inputs_raise_clear_errors(self):
        valid = np.ones((2, 3), dtype=np.float64)

        with self.assertRaisesRegex(ValueError, "same shape"):
            tdi(valid, np.ones((2, 4), dtype=np.float64))
        with self.assertRaisesRegex(ValueError, r"\[bs, seq_len\]"):
            tdi(np.ones(3), np.ones(3))
        with self.assertRaisesRegex(ValueError, "greater than 1"):
            tdi(np.ones((2, 1)), np.ones((2, 1)))
        with self.assertRaisesRegex(ValueError, "y_true contains NaN or Inf"):
            tdi(np.array([[0.0, np.nan]]), np.ones((1, 2)))
        with self.assertRaisesRegex(ValueError, "y_pred contains NaN or Inf"):
            tdi(np.ones((1, 2)), np.array([[0.0, np.inf]]))
        with self.assertRaisesRegex(ValueError, "reduction"):
            tdi(valid, valid, reduction="sum")
        with self.assertRaisesRegex(ValueError, "reduction"):
            dtw(valid, valid, reduction="sum")


if __name__ == "__main__":
    unittest.main()
