import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config as project_config
from config import BASE_INTERVAL_SECONDS, PROJECT_ROOT


def dump_weather_forecasts(
    model,
    datasets,
    batch_size,
    num_workers,
    device,
    run_dir,
    show_progress,
):
    # 按窗口写出三份未来天气，供预测天气场景读取。
    model.eval()
    for split in ("train", "val", "test"):
        loader = DataLoader(
            datasets[split],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )
        weather_chunks = []
        turbine_chunks = []
        start_chunks = []
        batches = tqdm(
            loader,
            total=len(loader),
            desc=f"dump weather {split}",
            disable=show_progress == 0,
            leave=False,
        )
        with torch.no_grad():
            for past, weather, target, turbine_id, target_start in batches:
                past = past.to(device, non_blocking=True)
                weather = weather.to(device, non_blocking=True)
                turbine_gpu = turbine_id.to(device, non_blocking=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    prediction = model(past, weather, turbine_gpu)
                weather_chunks.append(prediction.float().cpu().numpy())
                turbine_chunks.append(turbine_id.numpy())
                start_chunks.append(target_start.numpy())

        weather_pred = np.concatenate(weather_chunks).astype(np.float32)
        turbine_ids = np.concatenate(turbine_chunks).astype(np.int16)
        starts = np.concatenate(start_chunks).astype(np.int64)
        out_path = run_dir / f"weather_forecast_{split}.npz"
        np.savez_compressed(
            out_path,
            weather=weather_pred,
            turbine_id=turbine_ids,
            target_start_ns=starts,
        )
        display_path = out_path.resolve()
        if display_path.is_relative_to(PROJECT_ROOT):
            display_path = display_path.relative_to(PROJECT_ROOT)
        print(
            json.dumps(
                {
                    "event": "weather_forecast_saved",
                    "split": split,
                    "path": display_path.as_posix(),
                    "windows": int(weather_pred.shape[0]),
                    "shape": list(weather_pred.shape),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )


def evaluate_weather(args, model, datasets, device, checkpoint, best_epoch, best_mse):
    # 天气程序生成预测文件并汇总测试尺度误差。
    run_dir = Path(args.run_dir)
    dump_weather_forecasts(
        model,
        datasets,
        args.batch_size,
        args.num_workers,
        device,
        run_dir,
        args.show_progress,
    )
    test_loader = DataLoader(
        datasets["test"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_mse_total = 0.0
    test_points = 0
    with torch.no_grad():
        for past, weather, target, turbine_id, _ in test_loader:
            past = past.to(device, non_blocking=True)
            weather = weather.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            turbine_id = turbine_id.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                prediction = model(past, weather, turbine_id)
                mse = torch.mean((prediction - target) ** 2)
            count = int(target.numel())
            test_mse_total += float(mse.detach().cpu()) * count
            test_points += count
    test_mse_scaled = test_mse_total / test_points

    status = "complete"
    final_path = run_dir / "final.json"
    if args.mode == "test":
        status = "tested"
        final_path = run_dir / "test_final.json"
    final = {
        "status": status,
        "mode": args.mode,
        "model": args.model,
        "scenario": args.scenario,
        "loss": args.loss,
        "history_steps": project_config.HISTORY_STEPS,
        "horizon_steps": args.horizon_steps,
        "base_interval_seconds": BASE_INTERVAL_SECONDS,
        "point_interval_seconds": project_config.POINT_INTERVAL_SECONDS,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_val_mse_scaled": best_mse,
        "test_mse_scaled": test_mse_scaled,
        "checkpoint": str(checkpoint),
        "weather_forecast_dir": str(run_dir),
        "weather_forecast_files": [
            "weather_forecast_train.npz",
            "weather_forecast_val.npz",
            "weather_forecast_test.npz",
        ],
    }
    final_path.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    history_minutes = (
        project_config.HISTORY_STEPS * project_config.POINT_INTERVAL_SECONDS / 60
    )
    horizon_minutes = args.horizon_steps * project_config.POINT_INTERVAL_SECONDS / 60
    stride_minutes = args.window_stride_steps * project_config.POINT_INTERVAL_SECONDS / 60
    log_line = (
        f"history={history_minutes:g}min horizon={horizon_minutes:g}min "
        f"window_stride={stride_minutes:g}min "
        f"val_mse_scaled={best_mse:.6f} test_mse_scaled={test_mse_scaled:.6f} "
        f"model={args.model} scenario={args.scenario} best_epoch={best_epoch}\n"
    )
    if args.mode == "test":
        log_line = "mode=test " + log_line
    with (PROJECT_ROOT / "run.log").open("a", encoding="utf-8") as handle:
        handle.write(log_line)
    display_checkpoint = checkpoint.resolve()
    if display_checkpoint.is_relative_to(PROJECT_ROOT):
        display_checkpoint = display_checkpoint.relative_to(PROJECT_ROOT)
    print(
        f"[ForecastWeather] {status} best_epoch={best_epoch} "
        f"test_mse_scaled={test_mse_scaled:.6f} "
        f"checkpoint={display_checkpoint.as_posix()}",
        flush=True,
    )
    return final
