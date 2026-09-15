import json
import time

import torch
from torch.utils.data import DataLoader, Subset

from config import PROJECT_ROOT
from observability import CONTRACT_VERSION, result_checkpoint_path
from tasks.multi_turbine.pretrain_epoch import (
    MaskedMarketReconstruction,
    run_satra_masked_epoch,
)
from tasks.multi_turbine.pretrain_scheduler import build_pretrain_scheduler


def pretrain_satra_encoder(args, model, repository, datasets, loaders, device, run_dir):
    """训练 MMR 重建头，随后将验证最优的 SATRA 骨干保留给功率预测。"""
    validation_indices = []
    for index, history_start in enumerate(datasets["val"].windows):
        if repository.times[history_start] >= repository.train_end:
            validation_indices.append(index)
    validation_dataset = Subset(datasets["val"], validation_indices)
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    # 第一阶段保存独立的重建检查点和逐轮历史。
    checkpoint_path = run_dir / "satra_pretrain_checkpoint.pt"
    if args.result_name:
        round_path = result_checkpoint_path(args.dataset_name, args.result_name, 1)
        checkpoint_path = round_path.with_name("round_1_satra_pretrain.pt")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history_path = run_dir / "satra_pretrain_history.json"
    history = []
    best_mse = float("inf")
    best_epoch = 0
    wait = 0
    started = time.time()
    cuda_devices = []
    if device.type == "cuda":
        cuda_devices.append(device.index)
    # 第一阶段独立随机流，避免改变第二阶段的数据采样顺序。
    with torch.random.fork_rng(devices=cuda_devices):
        pretrainer = MaskedMarketReconstruction(model.backbone).to(device)
        optimizer = torch.optim.AdamW(
            pretrainer.parameters(),
            lr=args.pretrain_learning_rate,
        )
        scheduler, scheduler_steps_per_batch = build_pretrain_scheduler(
            args,
            optimizer,
            len(loaders["train"]),
        )
        for epoch in range(1, args.pretrain_epochs + 1):
            epoch_started = time.time()
            train_mse = run_satra_masked_epoch(
                pretrainer,
                loaders["train"],
                optimizer,
                device,
                args.mae_mask_ratio,
                args.seed + epoch,
                True,
                args.show_progress,
                epoch,
                scheduler,
                scheduler_steps_per_batch,
            )
            val_mse = run_satra_masked_epoch(
                pretrainer,
                validation_loader,
                optimizer,
                device,
                args.mae_mask_ratio,
                args.seed,
                False,
                args.show_progress,
                epoch,
                scheduler,
                scheduler_steps_per_batch,
            )
            if not scheduler_steps_per_batch:
                scheduler.step()

            improved = val_mse < best_mse
            if improved:
                best_mse = val_mse
                best_epoch = epoch
                wait = 0
                torch.save(
                    {
                        "contract_version": CONTRACT_VERSION,
                        "dataset": args.dataset_name,
                        "result_name": args.result_name,
                        "model_name": args.model,
                        "stage": "masked_market_reconstruction",
                        "seed": args.seed,
                        "epoch": epoch,
                        "validation_mse_scaled": val_mse,
                        "model_state": pretrainer.state_dict(),
                        "config": vars(args).copy(),
                    },
                    checkpoint_path,
                )
            else:
                wait += 1

            seconds = time.time() - epoch_started
            history.append(
                {
                    "epoch": epoch,
                    "train_mse_scaled": train_mse,
                    "validation_mse_scaled": val_mse,
                    "seconds": seconds,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "saved": improved,
                }
            )
            history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
            if wait >= args.pretrain_patience:
                break
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        pretrainer.load_state_dict(checkpoint["model_state"])

    summary = {
        "stage": "masked_market_reconstruction",
        "best_epoch": best_epoch,
        "best_validation_mse_scaled": best_mse,
        "completed_epochs": len(history),
        "mask_mode": "random_turbine_time_tokens_all_features",
        "mask_ratio": args.mae_mask_ratio,
        "learning_rate": args.pretrain_learning_rate,
        "lr_scheduler": args.pretrain_lr_scheduler,
        "normalization": "per_turbine_train_standard_scaler",
        "validation_mask_seed": args.seed,
        "train_windows": len(datasets["train"]),
        "validation_windows": len(validation_dataset),
        "checkpoint": checkpoint_path.as_posix(),
        "history": history_path.as_posix(),
        "seconds": time.time() - started,
    }
    summary_path = run_dir / "satra_pretrain_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    display_path = checkpoint_path.resolve()
    if display_path.is_relative_to(PROJECT_ROOT):
        display_path = display_path.relative_to(PROJECT_ROOT)
    print(
        f"[SATRA Pretrain] transferred_epoch={best_epoch} "
        f"val_MSE={best_mse:.6f} checkpoint={display_path}",
        flush=True,
    )
    return summary
