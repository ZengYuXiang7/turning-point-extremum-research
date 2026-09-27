from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from data_provider.common import Repository  # noqa: E402
from models.single_impl.time_moe_arevin import TimeMoEAdaptiveRevIN  # noqa: E402
from tasks.single.no_future_dataset import NoFutureWeatherDataset  # noqa: E402


def select_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="用真实广宁数据执行一次 TimeMoE+A-RevIN 训练步"
    )
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT / "dataset" / "processed")
    parser.add_argument("--time-moe-path", type=Path, default=PROJECT_ROOT / "models" / "pretrained" / "TimeMoE-50M")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    config.apply_point_interval_seconds(900)
    config.apply_history_steps(16)
    config.apply_window_stride_steps(1)
    repository = Repository(root=args.dataset_root.expanduser().resolve())
    dataset = NoFutureWeatherDataset(repository, "train", horizon=1)
    past, target, turbine_id, target_start = dataset[0]

    device = select_device(args.device)
    model = TimeMoEAdaptiveRevIN(
        horizon=1,
        channels=len(repository.feature_names),
        power_index=repository.power_index,
        entity_count=len(repository.series),
        pretrained_path=str(args.time_moe_path.expanduser().resolve()),
        bottleneck=512,
        unfreeze_layers=0,
        gradient_checkpointing=False,
    ).to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameter_groups(adapter_lr=1e-3, backbone_lr=1e-5),
        weight_decay=1e-4,
    )
    optimizer.zero_grad(set_to_none=True)
    prediction = model(past.unsqueeze(0).to(device), turbine_id.unsqueeze(0).to(device))
    loss = torch.nn.functional.mse_loss(prediction, target.unsqueeze(0).to(device))
    loss.backward()
    trainable_gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    finite_gradients = bool(trainable_gradients) and all(
        torch.isfinite(gradient).all() for gradient in trainable_gradients
    )
    optimizer.step()

    print(f"device={device}")
    print(f"train_windows={len(dataset)}")
    print(f"input_shape={tuple(past.shape)}")
    print(f"target_shape={tuple(target.shape)}")
    print(f"output_shape={tuple(prediction.shape)}")
    print(f"target_start_ns={int(target_start)}")
    print(f"finite_output={bool(torch.isfinite(prediction).all().item())}")
    print(f"finite_gradients={finite_gradients}")
    print(f"smoke_loss={float(loss.item()):.6f}")


if __name__ == "__main__":
    main()
