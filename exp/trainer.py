from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.settings import (
    GRAIN,
    HISTORY_MINUTES,
    HISTORY_STEPS,
    PROJECT_ROOT,
    SAMPLE_SECONDS,
    STEPS_PER_MINUTE,
)
from data_provider.forecast_weather_dataset import make_forecast_weather_loaders
from data_provider.multi_turbine_relation_dataset import (
    make_multi_turbine_no_future_weather_loaders,
    make_multi_turbine_oracle_future_weather_loaders,
)
from data_provider.no_future_weather_dataset import make_no_future_weather_loaders
from data_provider.oracle_future_weather_dataset import make_oracle_future_weather_loaders
from data_provider.predicted_future_weather_dataset import (
    make_predicted_future_weather_loaders,
)
from data_provider.weather_to_power_dataset import make_weather_to_power_loaders
from models.forecast_model import ForecastModel
from models.StockEcho_windpower import StockEchoWindPower
from models.weather_forecast import WeatherForecastModel
from models.weather_to_power import WeatherToPowerModel
from utils.acc30_loss import Acc30BoundaryLoss
from utils.dbloss import DBLoss
from utils.metrics import metric_bundle, save_metrics
from utils.reproducibility import seed_everything


def run_epoch(
    model,
    loader,
    device,
    optimizer,
    objective,
    train,
    power_means,
    power_stds,
    show_progress,
    epoch,
    weather_task,
):
    # 单个 epoch 的前向与可选反传；bf16 AMP，clip_norm=1.0
    model.train(train)
    total_loss = 0.0
    total_mse = 0.0
    points = 0
    strict_passed = 0
    strict_valid = 0

    if train:
        context = torch.enable_grad()
        stage = "train"
    else:
        context = torch.no_grad()
        stage = "val"

    batches = tqdm(
        loader,
        total=len(loader),
        desc=f"epoch {epoch} {stage}",
        disable=show_progress == 0,
        leave=False,
    )

    with context:
        for past, weather, target, turbine_id, _ in batches:
            past = past.to(device, non_blocking=True)
            weather = weather.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            turbine_id = turbine_id.to(device, non_blocking=True)
            batch_mean = power_means[turbine_id].unsqueeze(-1)
            batch_std = power_stds[turbine_id].unsqueeze(-1)

            if train:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                prediction = model(past, weather, turbine_id)
                mse = torch.mean((prediction - target) ** 2)

                if objective is None:
                    loss = mse
                elif isinstance(objective, Acc30BoundaryLoss):
                    loss = objective(prediction, target, batch_mean, batch_std)
                else:
                    loss = objective(
                        prediction.reshape(-1, prediction.shape[-1]),
                        target.reshape(-1, target.shape[-1]),
                    )

            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            count = int(target.numel())
            total_loss += float(loss.detach().cpu()) * count
            total_mse += float(mse.detach().cpu()) * count
            points += count

            # 功率任务才算 Acc30；天气任务跳过
            if not weather_task:
                with torch.no_grad():
                    truth_kw = target * batch_std + batch_mean
                    prediction_kw = prediction * batch_std + batch_mean
                    valid = truth_kw > 100.0
                    passed = valid & (
                        torch.abs(prediction_kw - truth_kw) <= 0.30 * truth_kw
                    )
                    strict_valid += int(valid.sum().item())
                    strict_passed += int(passed.sum().item())

            batches.set_postfix(
                loss=total_loss / points,
                mse=total_mse / points,
                refresh=False,
            )

    if weather_task:
        strict_acc30 = 0.0
    else:
        strict_acc30 = 100.0 * strict_passed / strict_valid
    return total_loss / points, total_mse / points, strict_acc30


def dump_weather_forecasts(model, datasets, batch_size, device, run_dir, show_progress):
    # 程序1：按窗口写出 train/val/test 未来气象，供程序2读取
    model.eval()
    for split in ("train", "val", "test"):
        loader = DataLoader(
            datasets[split],
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
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
        print(
            json.dumps(
                {
                    "event": "weather_forecast_saved",
                    "split": split,
                    "path": str(out_path),
                    "windows": int(weather_pred.shape[0]),
                    "shape": list(weather_pred.shape),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )


def train_and_test(args) -> dict:
    seed_everything(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weather_task = args.scenario == "ForecastWeather"
    weather_to_power = args.scenario == "WeatherToPower"
    predicted_weather = args.scenario == "PredictedFutureWeather"
    oracle = args.scenario == "OracleFutureWeather"
    joint_panel = args.model == "StockEcho"
    qwen_mlp = args.model == "QwenMLP"
    timer_weather_mlp = args.model == "TimerWeatherMLP"

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    horizon_steps = args.horizon * STEPS_PER_MINUTE

    if weather_to_power:
        # 同时刻天气→功率；无历史窗/视界
        horizon_steps = 1
        repository, datasets, loaders = make_weather_to_power_loaders(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        model = WeatherToPowerModel().to(device)
    elif weather_task:
        repository, datasets, loaders = make_forecast_weather_loaders(
            horizon=horizon_steps,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        model = WeatherForecastModel(args.model, horizon_steps).to(device)
    elif predicted_weather:
        # 程序2：天气来自程序1的 npy
        repository, datasets, loaders = make_predicted_future_weather_loaders(
            horizon=horizon_steps,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            weather_forecast_dir=args.weather_forecast_dir,
        )
        if timer_weather_mlp:
            from models.TimerWeatherMLP import TimerWeatherMLP

            model = TimerWeatherMLP(
                horizon=horizon_steps,
                pretrained_path=args.timer_path,
                patch_length=args.timer_patch_length,
                bottleneck=args.timer_bottleneck,
                unfreeze_layers=args.timer_unfreeze_layers,
                gradient_checkpointing=bool(args.timer_gradient_checkpointing),
                residual_forecast=bool(args.timer_residual_forecast),
            ).to(device)
        else:
            model = ForecastModel(args.model, horizon_steps).to(device)
    elif joint_panel:
        if oracle:
            repository, datasets, loaders = (
                make_multi_turbine_oracle_future_weather_loaders(
                    horizon=horizon_steps,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                )
            )
        else:
            repository, datasets, loaders = (
                make_multi_turbine_no_future_weather_loaders(
                    horizon=horizon_steps,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                )
            )
        model = StockEchoWindPower(horizon_steps).to(device)
    else:
        if oracle:
            repository, datasets, loaders = make_oracle_future_weather_loaders(
                horizon=horizon_steps,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
        else:
            repository, datasets, loaders = make_no_future_weather_loaders(
                horizon=horizon_steps,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
        if qwen_mlp:
            from models.QwenMLP import QwenMLP

            model = QwenMLP(
                horizon=horizon_steps,
                pretrained_path=args.llm_path,
                patch_length=args.llm_patch_length,
                patch_stride=args.llm_patch_stride,
                bottleneck=args.llm_bottleneck,
                gradient_checkpointing=bool(args.llm_gradient_checkpointing),
            ).to(device)
        elif timer_weather_mlp:
            if not (oracle or predicted_weather):
                raise ValueError(
                    "TimerWeatherMLP requires OracleFutureWeather or PredictedFutureWeather"
                )
            from models.TimerWeatherMLP import TimerWeatherMLP

            model = TimerWeatherMLP(
                horizon=horizon_steps,
                pretrained_path=args.timer_path,
                patch_length=args.timer_patch_length,
                bottleneck=args.timer_bottleneck,
                unfreeze_layers=args.timer_unfreeze_layers,
                gradient_checkpointing=bool(args.timer_gradient_checkpointing),
                residual_forecast=bool(args.timer_residual_forecast),
            ).to(device)
        else:
            model = ForecastModel(args.model, horizon_steps).to(device)

    for split in ("train", "val", "test"):
        past, weather, target, turbine_id, target_start_ns = datasets[split][0]
        print(
            json.dumps(
                {
                    "event": "data_summary",
                    "split": split,
                    "windows": len(datasets[split]),
                    "batches": len(loaders[split]),
                    "past_shape": list(past.shape),
                    "weather_shape": list(weather.shape),
                    "target_shape": list(target.shape),
                    "turbine_id_shape": list(turbine_id.shape),
                    "batch_past_shape": [args.batch_size] + list(past.shape),
                    "batch_weather_shape": [args.batch_size] + list(weather.shape),
                    "batch_target_shape": [args.batch_size] + list(target.shape),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    logging.getLogger("torch.fx.experimental.symbolic_shapes").setLevel(logging.ERROR)
    if not (qwen_mlp or timer_weather_mlp):
        model = torch.compile(model, mode="reduce-overhead")

    if args.loss == "MSE":
        objective = None
    elif args.loss == "DBLoss":
        objective = DBLoss(alpha=0.2, beta=0.5).to(device)
    else:
        objective = Acc30BoundaryLoss(weight=0.3, temperature=0.03).to(device)

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if timer_weather_mlp:
        optimizer = torch.optim.AdamW(
            model.parameter_groups(
                adapter_lr=args.learning_rate,
                backbone_lr=args.timer_backbone_learning_rate,
            ),
            weight_decay=1e-4,
        )
    else:
        optimizer = torch.optim.Adam(trainable_parameters, lr=args.learning_rate)
    # WeatherToPower 侧重 Acc30：训练仍可用 MSE，早停/选模/调 LR 盯 val Acc30
    select_acc30 = args.loss == "MSEAcc30" or weather_to_power
    if select_acc30:
        scheduler_mode = "max"
        selection_metric = "validation_strict_acc30"
    else:
        scheduler_mode = "min"
        selection_metric = "validation_mse_scaled"
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=scheduler_mode, factor=0.5, patience=3
    )

    power_mean_values = []
    power_std_values = []
    for series in repository.series:
        power_mean_values.append(series.power_mean)
        power_std_values.append(series.power_std)
    power_means = torch.tensor(power_mean_values, dtype=torch.float32, device=device)
    power_stds = torch.tensor(power_std_values, dtype=torch.float32, device=device)

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config["grain"] = GRAIN
    config["sample_seconds"] = SAMPLE_SECONDS
    config["history_minutes"] = HISTORY_MINUTES
    config["history_steps"] = HISTORY_STEPS
    config["horizon_minutes"] = args.horizon
    config["horizon_steps"] = horizon_steps
    config["oracle_future_weather"] = oracle
    config["weather_task"] = weather_task
    config["weather_to_power"] = weather_to_power
    config["predicted_future_weather"] = predicted_weather
    config["selection_metric"] = selection_metric
    config["acc30_loss_weight"] = 0.3 if args.loss == "MSEAcc30" else None
    config["acc30_loss_temperature"] = 0.03 if args.loss == "MSEAcc30" else None
    sample_past, sample_weather, sample_target, _, _ = datasets["train"][0]
    config["train_windows"] = len(datasets["train"])
    config["val_windows"] = len(datasets["val"])
    config["test_windows"] = len(datasets["test"])
    config["past_shape"] = list(sample_past.shape)
    config["weather_shape"] = list(sample_weather.shape)
    config["target_shape"] = list(sample_target.shape)
    config["parameter_count"] = int(sum(p.numel() for p in model.parameters()))
    config["trainable_parameter_count"] = int(
        sum(p.numel() for p in model.parameters() if p.requires_grad)
    )
    config["joint_multi_turbine_panel"] = joint_panel
    config["amp_dtype"] = "bf16"
    config["tf32"] = True
    if qwen_mlp:
        config["compile_mode"] = "disabled_for_qwen_mlp"
    elif timer_weather_mlp:
        config["compile_mode"] = "disabled_for_timer_weather_mlp"
    else:
        config["compile_mode"] = "reduce-overhead"
    config["max_clip_norm"] = 1.0
    (run_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 天气任务：启动时打持续法基线（抽样若干 val batch，避免全量扫描过慢）
    if weather_task:
        persist_total = 0.0
        persist_points = 0
        baseline_batches = 0
        with torch.no_grad():
            for past, weather, target, turbine_id, _ in loaders["val"]:
                past = past.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                persist = past[:, -1:, :].expand_as(target)
                mse = torch.mean((persist - target) ** 2)
                count = int(target.numel())
                persist_total += float(mse.detach().cpu()) * count
                persist_points += count
                baseline_batches += 1
                if baseline_batches >= 64:
                    break
        persist_val_mse = persist_total / persist_points
        print(
            f"[ForecastWeather] baseline persistence val_mse≈{persist_val_mse:.6f}  "
            f"(历史末步填满未来；模型 val_mse 应低于此)",
            flush=True,
        )
        config["persist_val_mse_scaled"] = persist_val_mse
        (run_dir / "config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # --- 训练与早停 ---
    best_mse = float("inf")
    best_acc30 = float("-inf")
    best_epoch = 0
    wait = 0
    history = []
    checkpoint = run_dir / "best_checkpoint.pt"

    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train_loss, train_mse, train_acc30 = run_epoch(
            model,
            loaders["train"],
            device,
            optimizer,
            objective,
            True,
            power_means,
            power_stds,
            args.show_progress,
            epoch,
            weather_task,
        )
        _, val_mse, val_acc30 = run_epoch(
            model,
            loaders["val"],
            device,
            optimizer,
            objective,
            False,
            power_means,
            power_stds,
            args.show_progress,
            epoch,
            weather_task,
        )

        if select_acc30:
            scheduler.step(val_acc30)
        else:
            scheduler.step(val_mse)

        if weather_task:
            record = {
                "task": "ForecastWeather",
                "epoch": epoch,
                "train_mse": round(train_mse, 6),
                "val_mse": round(val_mse, 6),
                "learning_rate": round(optimizer.param_groups[0]["lr"], 6),
                "seconds": round(time.time() - started, 2),
            }
            history.append(record)
            print(
                f"[ForecastWeather] epoch {epoch}/{args.epochs}  "
                f"train_mse={train_mse:.6f}  val_mse={val_mse:.6f}  "
                f"lr={optimizer.param_groups[0]['lr']:.1e}  "
                f"{record['seconds']:.1f}s",
                flush=True,
            )
        else:
            record = {
                "task": "Power",
                "epoch": epoch,
                "train_mse": round(train_mse, 6),
                "train_acc30": round(train_acc30, 2),
                "val_mse": round(val_mse, 6),
                "val_acc30": round(val_acc30, 2),
                "learning_rate": round(optimizer.param_groups[0]["lr"], 6),
                "seconds": round(time.time() - started, 2),
            }
            history.append(record)
            print(
                f"[Power] epoch {epoch}/{args.epochs}  "
                f"train_mse={train_mse:.6f}  train_acc30={train_acc30:.2f}%  "
                f"val_mse={val_mse:.6f}  val_acc30={val_acc30:.2f}%  "
                f"lr={optimizer.param_groups[0]['lr']:.1e}  "
                f"{record['seconds']:.1f}s",
                flush=True,
            )

        if select_acc30:
            if val_acc30 > best_acc30 + 1e-10:
                improved = True
            elif abs(val_acc30 - best_acc30) <= 1e-10:
                improved = val_mse < best_mse - 1e-10
            else:
                improved = False
        else:
            improved = val_mse < best_mse - 1e-10

        if improved:
            best_mse = val_mse
            best_acc30 = val_acc30
            best_epoch = epoch
            wait = 0
            if qwen_mlp:
                model_state = {
                    name: value.detach().cpu()
                    for name, value in model.state_dict().items()
                    if not name.startswith("llm.")
                }
            else:
                model_state = model.state_dict()
            torch.save(
                {"model_state": model_state, "epoch": epoch, "config": config},
                checkpoint,
            )
        else:
            wait += 1
            if wait >= args.patience:
                if weather_task:
                    tag = "ForecastWeather"
                else:
                    tag = "Power"
                print(
                    f"[{tag}] early_stop  epoch={epoch}  best_epoch={best_epoch}",
                    flush=True,
                )
                break

    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    # --- 测试 ---
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"], strict=not qwen_mlp)
    model.eval()

    # 程序1：写出三份天气 npy，并在 test 上记天气 MSE
    if weather_task:
        dump_weather_forecasts(
            model,
            datasets,
            args.batch_size,
            device,
            run_dir,
            args.show_progress,
        )

        # test 天气 MSE（scaled），供 history 搜索写 run.log
        test_mse_total = 0.0
        test_points = 0
        test_loader = DataLoader(
            datasets["test"],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )
        with torch.no_grad():
            for past, weather, target, turbine_id, _ in test_loader:
                past = past.to(device, non_blocking=True)
                weather = weather.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                turbine_gpu = turbine_id.to(device, non_blocking=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    prediction = model(past, weather, turbine_gpu)
                    mse = torch.mean((prediction - target) ** 2)
                count = int(target.numel())
                test_mse_total += float(mse.detach().cpu()) * count
                test_points += count
        test_mse_scaled = test_mse_total / test_points

        final = {
            "status": "complete",
            "model": args.model,
            "scenario": args.scenario,
            "loss": args.loss,
            "grain": GRAIN,
            "history_minutes": HISTORY_MINUTES,
            "history_steps": HISTORY_STEPS,
            "horizon_minutes": args.horizon,
            "horizon_steps": horizon_steps,
            "sample_seconds": SAMPLE_SECONDS,
            "seed": args.seed,
            "best_epoch": best_epoch,
            "best_val_mse_scaled": best_mse,
            "test_mse_scaled": test_mse_scaled,
            "weather_forecast_dir": str(run_dir),
            "weather_forecast_files": [
                "weather_forecast_train.npz",
                "weather_forecast_val.npz",
                "weather_forecast_test.npz",
            ],
        }
        (run_dir / "final.json").write_text(
            json.dumps(final, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log_line = (
            f"grain={GRAIN} history={HISTORY_MINUTES} horizon={args.horizon} "
            f"model={args.model} scenario={args.scenario} best_epoch={best_epoch} "
            f"val_mse_scaled={best_mse:.6f} test_mse_scaled={test_mse_scaled:.6f}\n"
        )
        with (PROJECT_ROOT / "run.log").open("a", encoding="utf-8") as handle:
            handle.write(log_line)
        print(
            json.dumps(
                {
                    "event": "complete",
                    "run_dir": str(run_dir),
                    "best_epoch": best_epoch,
                    "best_val_mse_scaled": best_mse,
                    "test_mse_scaled": test_mse_scaled,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return final

    predictions = []
    truths = []
    turbines = []
    target_starts = []
    test_batches = tqdm(
        loaders["test"],
        total=len(loaders["test"]),
        desc="test",
        disable=args.show_progress == 0,
        leave=False,
    )
    with torch.no_grad():
        for past, weather, target, turbine_id, target_start in test_batches:
            past = past.to(device, non_blocking=True)
            weather = weather.to(device, non_blocking=True)
            turbine_gpu = turbine_id.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                prediction = model(past, weather, turbine_gpu)
            predictions.append(prediction.float().cpu().numpy())
            truths.append(target.numpy())
            turbines.append(turbine_id.numpy())
            target_starts.append(target_start.numpy())

    prediction_scaled = np.concatenate(predictions)
    truth_scaled = np.concatenate(truths)
    turbine_ids = np.concatenate(turbines).astype(np.int16)
    starts = np.concatenate(target_starts).astype(np.int64)

    if joint_panel:
        prediction_scaled = prediction_scaled.reshape(-1, prediction_scaled.shape[-1])
        truth_scaled = truth_scaled.reshape(-1, truth_scaled.shape[-1])
        turbine_ids = turbine_ids.reshape(-1)
        starts = np.repeat(starts, 16)

    mean_list = []
    std_list = []
    for turbine_index in turbine_ids:
        series = repository.series[int(turbine_index)]
        mean_list.append(series.power_mean)
        std_list.append(series.power_std)
    means = np.asarray(mean_list, dtype=np.float32)[:, None]
    stds = np.asarray(std_list, dtype=np.float32)[:, None]
    prediction = prediction_scaled * stds + means
    truth = truth_scaled * stds + means

    np.savez_compressed(
        run_dir / "test_predictions.npz",
        prediction=prediction.astype(np.float32),
        truth=truth.astype(np.float32),
        turbine_id=(turbine_ids + 1).astype(np.int16),
        target_start_ns=starts,
    )

    metrics = metric_bundle(prediction, truth, turbine_ids)
    save_metrics(run_dir, metrics)
    curve_overall = metrics[0]

    if select_acc30:
        best_val_strict_acc30 = best_acc30
    else:
        best_val_strict_acc30 = None

    final = {
        "status": "complete",
        "model": args.model,
        "scenario": args.scenario,
        "loss": args.loss,
        "grain": GRAIN,
        "history_minutes": HISTORY_MINUTES,
        "history_steps": HISTORY_STEPS,
        "horizon_minutes": args.horizon,
        "horizon_steps": horizon_steps,
        "sample_seconds": SAMPLE_SECONDS,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_val_mse_scaled": best_mse,
        "best_val_strict_acc30": best_val_strict_acc30,
        "weather_forecast_dir": args.weather_forecast_dir if predicted_weather else "",
        "metrics": metrics,
    }
    (run_dir / "final.json").write_text(
        json.dumps(final, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log_line = (
        f"grain={GRAIN} history={HISTORY_MINUTES} horizon={args.horizon} model={args.model} "
        f"scenario={args.scenario} best_epoch={best_epoch} "
        f"strict_acc30={curve_overall['strict_acc30']:.6f} "
        f"mse_kw2={curve_overall['mse_kw2']:.6f} "
        f"mape={curve_overall['mape']:.6f}\n"
    )
    with (PROJECT_ROOT / "run.log").open("a", encoding="utf-8") as handle:
        handle.write(log_line)
    print(
        f"[Power] test  best_epoch={best_epoch}  "
        f"val_mse={best_mse:.6f}  val_acc30={best_acc30:.2f}%  "
        f"test_acc30={curve_overall['strict_acc30']:.2f}%  "
        f"test_mse_kw2={curve_overall['mse_kw2']:.2f}  "
        f"test_mape={curve_overall['mape']:.2f}%",
        flush=True,
    )
    return final
