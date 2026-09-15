import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset

from config import HISTORY_STEPS
from models.single import SingleModel
from tasks.single.pretrain import pretrain_power_encoder
from tasks.single.pretrain_epoch import (
    MaskedTokenPretraining,
    run_masked_epoch,
    sample_token_mask,
)
from tasks.single.configuration import parse_args


class TestMaskedPretraining(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2026)
        model_args = SimpleNamespace(model="PatchMLPAllFeatures", scenario="NoFutureWeather", horizon_steps=3,)
        repository = SimpleNamespace(feature_names=list(range(56)), power_index=55,)
        self.model = SingleModel(model_args, repository)
        self.past = torch.randn(4, HISTORY_STEPS, 56)
        self.turbine_id = torch.tensor([0, 1, 2, 3])
        self.pretrainer = MaskedTokenPretraining(self.model.backbone, HISTORY_STEPS,)

    def test_pretrain_flag_controls_activation_independently_from_epochs(self):
        # 轮数是训练参数，只有pretrain开关决定是否执行第一阶段
        command = [
            "run_single.py", "--model", "PatchMLPAllFeatures",
            "--scenario", "NoFutureWeather", "--loss", "MSE",
        ]
        with patch("sys.argv", command):
            direct_args = parse_args()
        with patch("sys.argv", command + ["--pretrain"]):
            pretrain_args = parse_args()

        self.assertFalse(direct_args.pretrain)
        self.assertTrue(pretrain_args.pretrain)
        self.assertEqual(direct_args.pretrain_epochs, 20)
        self.assertEqual(pretrain_args.pretrain_epochs, 20)
        self.assertEqual(pretrain_args.mae_mask_ratio, 0.3)
        self.assertEqual(pretrain_args.pretrain_lr_scheduler, "cosine_hard_restarts",)

    def test_random_tokens_share_all_features_and_support_short_history(self):
        # 最短历史也同时保留可见与缺失 token
        generator = torch.Generator().manual_seed(2026)
        short_past = self.past[:, :4]
        mask = sample_token_mask(short_past, 0.4, generator)
        self.assertEqual(tuple(mask.shape), (4, 4, 56))
        torch.testing.assert_close(mask[:, :, 0].sum(dim=1), torch.full((4,), 2))
        torch.testing.assert_close(mask, mask[:, :, :1].expand_as(mask))

    def test_hidden_values_cannot_leak_through_normalization_or_embedding(self):
        # 修改缺失位置不能改变任何重建输出
        self.pretrainer.eval()
        generator = torch.Generator().manual_seed(7)
        mask = sample_token_mask(self.past, 0.4, generator)
        altered = self.past.clone()
        altered[mask] += 1000.0
        with torch.no_grad():
            original = self.pretrainer(self.past, self.turbine_id, mask)
            changed = self.pretrainer(altered, self.turbine_id, mask)
        torch.testing.assert_close(original, changed, rtol=0, atol=0)

    def test_loss_uses_only_masked_history_and_validation_masks_are_fixed(self):
        # 未来标签设为 NaN，重建仍只使用历史和缺失位置
        targets = torch.full((4, 3), float("nan"))
        starts = torch.arange(4)
        dataset = TensorDataset(self.past, targets, self.turbine_id, starts)
        loader = DataLoader(dataset, batch_size=4)
        self.pretrainer.eval()
        mask = sample_token_mask(self.past, 0.4, torch.Generator().manual_seed(7))
        with torch.no_grad():
            reconstruction = self.pretrainer(self.past, self.turbine_id, mask)
            expected = (reconstruction - self.past).square()[mask].mean().item()
        first = run_masked_epoch(self.pretrainer, loader, None, torch.device("cpu"), 0.4, 7, False, 0, 1, None, False,)
        second = run_masked_epoch(self.pretrainer, loader, None, torch.device("cpu"), 0.4, 7, False, 0, 2, None, False,)
        self.assertAlmostEqual(first, expected, places=6)
        self.assertEqual(first, second)

    def test_best_encoder_transfers_and_power_head_trains_in_stage_two(self):
        # 合成样本验证热身选模、随机流恢复与第二阶段梯度
        targets = torch.full((4, 3), float("nan"))
        starts = torch.arange(4)
        train_dataset = TensorDataset(self.past, targets, self.turbine_id, starts)
        validation_dataset = TensorDataset(self.past + 0.2, targets, self.turbine_id, starts)
        validation_dataset.windows = [(0, 0), (0, 1), (0, 2), (0, 3)]
        repository = SimpleNamespace(train_end=10, series=[SimpleNamespace(times=[0, 10, 20, 30])],)
        datasets = {"train": train_dataset, "val": validation_dataset}
        loaders = {"train": DataLoader(train_dataset, batch_size=2, shuffle=True)}
        args = SimpleNamespace(model="PatchMLPAllFeatures", seed=2026, history_steps=HISTORY_STEPS, batch_size=2, num_workers=0, result_name="", dataset_name="Synthetic", pretrain_epochs=2, pretrain_learning_rate=0.001, pretrain_lr_scheduler="cosine_hard_restarts", mae_mask_ratio=0.3, pretrain_patience=2, print_freq=1, show_progress=0,)
        backbone = self.model.backbone
        initial_embedding = (
            backbone.patch_embedding.EmbLayer_1.ff[0].weight.detach().clone()
        )
        initial_power_head = backbone.output_projection.weight.detach().clone()
        rng_state = torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as directory:
            summary = pretrain_power_encoder(args, self.model, repository, datasets, loaders, torch.device("cpu"), Path(directory),)
            torch.testing.assert_close(torch.get_rng_state(), rng_state)
            self.assertEqual(summary["validation_windows"], 3)
            history = json.loads(Path(summary["history"]).read_text())
            best_mse = min(row["validation_mse_scaled"] for row in history)
            self.assertEqual(summary["best_validation_mse_scaled"], best_mse)
            self.assertEqual(summary["lr_scheduler"], "cosine_hard_restarts")
            self.assertNotEqual(history[0]["learning_rate"], 0.001)
            checkpoint = torch.load(summary["checkpoint"], weights_only=True)
            state = checkpoint["model_state"]
            for name, parameter in self.model.state_dict().items():
                torch.testing.assert_close(parameter, state[name])

        # 热身更新编码器，预测头保持初始化且不携带重建头
        trained_embedding = backbone.patch_embedding.EmbLayer_1.ff[0].weight
        self.assertFalse(torch.equal(initial_embedding, trained_embedding))
        torch.testing.assert_close(initial_power_head, backbone.output_projection.weight)
        self.assertEqual(backbone.output_projection.weight.grad, None)
        self.assertFalse(any("reconstruction" in name for name in self.model.state_dict()))

        # 第二阶段完整历史输入联合训练预测头与编码器
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = self.model(self.past, self.turbine_id)
        prediction.square().mean().backward()
        optimizer.step()
        self.assertEqual(tuple(prediction.shape), (4, 3))
        self.assertTrue(torch.isfinite(prediction).all())
        self.assertFalse(torch.equal(initial_power_head, backbone.output_projection.weight))


if __name__ == "__main__":
    unittest.main()
