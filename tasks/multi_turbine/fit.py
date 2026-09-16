import json
import time

import torch

from observability import CONTRACT_VERSION, result_checkpoint_path
from observability.exp_progress import EpochProgressPolicy, dynamic_tqdm_enabled
from tasks.multi_turbine.epoch import run_epoch


def validation_improved(val_mse, val_acc30, best_mse, best_acc30):
    # 联合功率预测固定以 Acc30 为主指标，同 Acc30 时比较 MSE。
    if val_acc30 > best_acc30 + 1e-10:
        improved = True
    elif abs(val_acc30 - best_acc30) <= 1e-10:
        improved = val_mse < best_mse - 1e-10
    else:
        improved = False
    return improved


def fit_model(args, model, loaders, device, objective, optimizer, scheduler, power_means, power_stds, config, run_dir,):
    # 多风机程序独立执行联合面板训练、选模与早停。
    best_mse = float("inf")
    best_acc30 = float("-inf")
    best_epoch = 0
    wait = 0
    history = []
    checkpoint = run_dir / "best_checkpoint.pt"
    contract_checkpoint = checkpoint
    if args.result_name:
        contract_checkpoint = result_checkpoint_path(args.dataset_name, args.result_name, 1)
        contract_checkpoint.parent.mkdir(parents=True, exist_ok=True)

    progress = EpochProgressPolicy(dynamic_tqdm_enabled(args.tqdm), args.epochs)
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        _, train_mse, train_acc30 = run_epoch(model, loaders["train"], device, optimizer, objective, args.dbloss_weight, True, power_means, power_stds, progress.show_progress(), epoch,)
        _, val_mse, val_acc30 = run_epoch(model, loaders["val"], device, optimizer, objective, args.dbloss_weight, False, power_means, power_stds, progress.show_progress(), epoch,)
        scheduler.step(val_acc30)

        seconds = time.time() - started
        progress.observe(seconds)
        record = {
            "task": "MultiTurbinePower",
            "epoch": epoch,
            "train_mse": round(train_mse, 6),
            "train_acc30": round(train_acc30, 2),
            "val_mse": round(val_mse, 6),
            "val_acc30": round(val_acc30, 2),
            "learning_rate": round(optimizer.param_groups[0]["lr"], 6),
            "seconds": round(seconds, 2),
        }
        history.append(record)
        if progress.should_log(epoch):
            print(f"[MultiTurbine] epoch {epoch}/{args.epochs} " f"train_mse={train_mse:.6f} train_acc30={train_acc30:.2f}% " f"val_mse={val_mse:.6f} val_acc30={val_acc30:.2f}% " f"lr={optimizer.param_groups[0]['lr']:.1e} {seconds:.1f}s", flush=True,)

        improved = validation_improved(val_mse, val_acc30, best_mse, best_acc30)
        if improved:
            best_mse = val_mse
            best_acc30 = val_acc30
            best_epoch = epoch
            wait = 0
            payload = {
                "model_state": model.state_dict(),
                "epoch": epoch,
                "config": config,
                "contract_version": CONTRACT_VERSION,
                "dataset": args.dataset_name,
                "result_name": args.result_name,
                "model_name": args.model,
                "seed": args.seed,
                "target_transform": config["target_transform"],
            }
            torch.save(payload, checkpoint)
            if args.result_name:
                torch.save(payload, contract_checkpoint)
        else:
            wait += 1
            if wait >= args.patience:
                print(f"[MultiTurbine] early_stop epoch={epoch} best_epoch={best_epoch}", flush=True,)
                break

    history_path = run_dir / "history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return (
        checkpoint,
        contract_checkpoint,
        best_epoch,
        best_mse,
        best_acc30,
        history,
    )
