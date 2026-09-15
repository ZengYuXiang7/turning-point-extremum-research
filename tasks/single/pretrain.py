import json
import math
import time

import torch
from torch.utils.data import DataLoader, Subset

from config import PROJECT_ROOT
from observability import CONTRACT_VERSION, result_checkpoint_path
from tasks.single.pretrain_epoch import MaskedTokenPretraining, run_masked_epoch
from tasks.single.pretrain_scheduler import build_pretrain_scheduler

def pretrain_power_encoder(args, model, repository, datasets, loaders, device, run_dir):
    # 验证重建窗口完全位于验证段，排除边界处的训练历史
    validation_dataset = datasets["val"]
    validation_indices = []
    for index, window in enumerate(validation_dataset.windows):
        turbine_index, history_start = window
        history_time = repository.series[turbine_index].times[history_start]
        if history_time >= repository.train_end:
            validation_indices.append(index)
    validation_subset = Subset(validation_dataset, validation_indices)
    validation_loader = DataLoader(validation_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda", drop_last=False,)

    # 第一阶段保存独立检查点与完整重建历史
    checkpoint_path = run_dir / "pretrain_checkpoint.pt"
    if args.result_name:
        round_path = result_checkpoint_path(args.dataset_name, args.result_name, 1)
        checkpoint_path = round_path.with_name("round_1_pretrain.pt")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history_path = run_dir / "pretrain_history.json"
    history = []
    best_mse = float("inf")
    best_epoch = 0
    wait = 0
    started = time.time()

    # fork_rng 在热身后恢复随机状态，保持第二阶段采样随机流
    cuda_devices = []
    if device.type == "cuda":
        cuda_devices.append(device.index)
    with torch.random.fork_rng(devices=cuda_devices):
        pretrainer = MaskedTokenPretraining(model.backbone, args.history_steps,).to(device)
        
        optimizer = torch.optim.AdamW(pretrainer.parameters(), lr=args.pretrain_learning_rate,)
        scheduler, scheduler_steps_per_batch = build_pretrain_scheduler(args, optimizer, len(loaders["train"]),)
        
        for epoch in range(1, args.pretrain_epochs + 1):
            epoch_started = time.time()
            train_mse = run_masked_epoch(pretrainer, loaders["train"], optimizer, device, args.mae_mask_ratio, args.seed + epoch, True, args.show_progress, epoch, scheduler, scheduler_steps_per_batch,)
            val_mse = run_masked_epoch(pretrainer, validation_loader, optimizer, device, args.mae_mask_ratio, args.seed, False, args.show_progress, epoch, scheduler, scheduler_steps_per_batch,)

            # 硬重启按batch更新，其余调度器按epoch更新
            if scheduler_steps_per_batch is False:
                scheduler.step()

            # 按固定掩码的验证重建误差选模
            improved = val_mse < best_mse
            if improved:
                best_mse = val_mse
                best_epoch = epoch
                wait = 0
                torch.save({ "contract_version": CONTRACT_VERSION, "dataset": args.dataset_name, "result_name": args.result_name, "model_name": args.model, "stage": "masked_history_reconstruction", "seed": args.seed, "epoch": epoch, "validation_mse_scaled": val_mse, "model_state": pretrainer.state_dict(), "config": vars(args).copy(), "target_transform": "per_turbine_train_standard_scaler", }, checkpoint_path,)
            else:
                wait += 1

            # 按显式频率输出常规日志，选模事件立即打印
            seconds = time.time() - epoch_started
            early_stop = wait >= args.pretrain_patience
            log_epoch = epoch == 1 or epoch % args.print_freq == 0
            if epoch == args.pretrain_epochs or improved:
                log_epoch = True
            if log_epoch or early_stop:
                print(f"[Pretrain] epoch={epoch}/{args.pretrain_epochs} " f"train_MSE={train_mse:.6f} val_MSE={val_mse:.6f} " f"best={best_mse:.6f}@{best_epoch} saved={improved} " f"early_stop={early_stop} seconds={seconds:.2f}", flush=True,)
            history.append({ "epoch": epoch, "train_mse_scaled": train_mse, "validation_mse_scaled": val_mse, "seconds": seconds, "learning_rate": optimizer.param_groups[0]["lr"], "saved": improved, })
            history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
            if early_stop:
                break

        # 共享编码器恢复最佳状态，重建头随第一阶段退出
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        pretrainer.load_state_dict(checkpoint["model_state"])

    # 第二阶段重新创建优化器，仅训练预测模型
    summary = {
        "stage": "masked_history_reconstruction",
        "best_epoch": best_epoch,
        "best_validation_mse_scaled": best_mse,
        "completed_epochs": len(history),
        "mask_mode": "random_time_tokens_all_features",
        "mask_ratio": args.mae_mask_ratio,
        "masked_tokens_per_window": math.ceil(args.history_steps * args.mae_mask_ratio),
        "learning_rate": args.pretrain_learning_rate,
        "lr_scheduler": args.pretrain_lr_scheduler,
        "scheduler_warmup_ratio": 0.1,
        "scheduler_cycles": 1,
        "scheduler_minimum_ratio": 0.01,
        "normalization": "visible_only_window_revin",
        "validation_mask_seed": args.seed,
        "train_windows": len(datasets["train"]),
        "validation_windows": len(validation_subset),
        "checkpoint": checkpoint_path.as_posix(),
        "history": history_path.as_posix(),
        "seconds": time.time() - started,
    }
    summary_path = run_dir / "pretrain_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    display_checkpoint_path = checkpoint_path.resolve()
    if display_checkpoint_path.is_relative_to(PROJECT_ROOT):
        display_checkpoint_path = display_checkpoint_path.relative_to(PROJECT_ROOT)
    print(f"[Pretrain] transferred_epoch={best_epoch} " f"val_MSE={best_mse:.6f} checkpoint={display_checkpoint_path}", flush=True,)
    return summary
