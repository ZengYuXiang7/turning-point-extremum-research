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
    HISTORY_COLUMNS,
    HISTORY_STEPS,
    PROJECT_ROOT,
    SAMPLE_SECONDS,
    WEATHER_COLUMNS,
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
from models.backbone.stockecho import StockEchoNoFutureWeather, StockEchoWindPower
from models.futureweather import FutureWeatherModel
from models.noweather import NoFutureWeatherModel
from models.weatherforecast import WeatherForecastModel
from observability import (
    CONTRACT_VERSION,
    aggregate_test_metrics,
    build_dataset_significance,
    result_checkpoint_path,
    result_record_path,
    result_report_path,
    write_result_record,
    write_result_report,
)
from utils.acc30_loss import Acc30BoundaryLoss
from utils.dbloss import DBLoss
from utils.metrics import metric_bundle, save_metrics
from utils.reproducibility import seed_everything


def format_ns_date(timestamp_ns: int) -> str:
    date = np.datetime_as_string(np.datetime64(timestamp_ns, "ns"), unit="D")
    return f"{date[:4]}年{date[5:7]}月{date[8:10]}日"


def print_data_summary(
    datasets,
    loaders,
    horizon_steps,
    window_stride_steps,
    no_future_weather,
    weather_task,
    oracle,
    predicted_weather,
):
    # 确定当前场景的输入与监督特征
    past_features = datasets["train"].repository.feature_names
    target_features = ["风机-P"]

    if weather_task:
        past_features = WEATHER_COLUMNS
        target_features = WEATHER_COLUMNS
        weather_source = "模型不使用，仅保留天气预测任务的统一接口"
    elif oracle:
        weather_source = "真实未来天气"
    elif predicted_weather:
        weather_source = "程序1预测的未来天气"

    print("\n========== 数据集与 Batch 输入 ==========")
    print(f"past 特征（{len(past_features)} 个）：")
    for feature_index, feature_name in enumerate(past_features, start=1):
        print(f"  {feature_index:02d}. {feature_name}")

    if no_future_weather:
        print("NoFutureWeather：不构造未来天气输入。")
    else:
        print(f"future_weather 特征（{len(WEATHER_COLUMNS)} 个，{weather_source}）：")
        for feature_index, feature_name in enumerate(WEATHER_COLUMNS, start=1):
            print(f"  {feature_index:02d}. {feature_name}")

    print(f"target 特征（{len(target_features)} 个）：{', '.join(target_features)}")
    print("turbine_id：从 0 开始的风机索引；单机样本为标量，多机面板样本为 16 个索引")
    history_minutes = HISTORY_STEPS * SAMPLE_SECONDS / 60
    horizon_minutes = horizon_steps * SAMPLE_SECONDS / 60
    window_stride_minutes = window_stride_steps * SAMPLE_SECONDS / 60
    sample_minutes = SAMPLE_SECONDS / 60
    print(
        f"历史长度={history_minutes:g}分钟 预测长度={horizon_minutes:g}分钟 "
        f"滑窗步长={window_stride_minutes:g}分钟\n"
    )
    print("各切分的预测目标覆盖范围、样本占比和 Batch 张量形状：")

    total_windows = sum(len(datasets[split]) for split in ("train", "val", "test"))
    target_span_ns = (horizon_steps - 1) * SAMPLE_SECONDS * 1_000_000_000
    total_coverage_ns = 0

    for split in ("train", "val", "test"):
        dataset = datasets[split]
        first_target_start_ns = int(dataset[0][-1].item())
        last_target_end_ns = int(dataset[-1][-1].item()) + target_span_ns
        total_coverage_ns += last_target_end_ns - first_target_start_ns

    for split in ("train", "val", "test"):
        dataset = datasets[split]
        loader = loaders[split]
        first_sample = dataset[0]
        last_sample = dataset[-1]
        target_start_ns = int(first_sample[-1].item())
        target_end_ns = int(last_sample[-1].item()) + target_span_ns
        coverage_days = (target_end_ns - target_start_ns) / 86_400_000_000_000
        time_ratio = 100.0 * (target_end_ns - target_start_ns) / total_coverage_ns
        window_ratio = 100.0 * len(dataset) / total_windows
        batch_items = []

        for sample_index in range(loader.batch_size):
            batch_items.append(dataset[sample_index])

        batch = loader.collate_fn(batch_items)
        print(
            f"  {split:<5}  目标时间={format_ns_date(target_start_ns)} 至 "
            f"{format_ns_date(target_end_ns)}（约 {coverage_days:.1f} 天）  "
            f"时间占比={time_ratio:.2f}%  窗口={len(dataset):,}（{window_ratio:.2f}%）  "
            f"batches={len(loader):,}"
        )

        if no_future_weather:
            past, target, turbine_id, target_start = batch
            print(
                f"    输入  past shape={list(past.shape)}  "
                f"turbine_id shape={list(turbine_id.shape)}"
            )
        else:
            past, future_weather, target, turbine_id, target_start = batch
            print(
                f"    输入  past shape={list(past.shape)}  "
                f"future_weather shape={list(future_weather.shape)}  "
                f"turbine_id shape={list(turbine_id.shape)}"
            )

        print(
            f"    标签  target shape={list(target.shape)}  "
            f"时间维={horizon_steps}点（每点{sample_minutes:g}分钟）  "
            f"target_start_ns shape={list(target_start.shape)}",
            flush=True,
        )
        print('')


def run_epoch(
    model,
    loader,
    device,
    optimizer,
    objective,
    dbloss_weight,
    train,
    power_means,
    power_stds,
    show_progress,
    epoch,
    no_future_weather,
    model_uses_future_weather,
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
        for batch in batches:
            if no_future_weather:
                past, target, turbine_id, _ = batch
            else:
                past, future_weather, target, turbine_id, _ = batch

            past = past.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            turbine_id = turbine_id.to(device, non_blocking=True)

            if model_uses_future_weather:
                future_weather = future_weather.to(device, non_blocking=True)

            batch_mean = power_means[turbine_id].unsqueeze(-1)
            batch_std = power_stds[turbine_id].unsqueeze(-1)

            if train:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                if model_uses_future_weather:
                    prediction = model(past, future_weather, turbine_id)
                else:
                    prediction = model(past, turbine_id)

                mse = torch.mean((prediction - target) ** 2)

                if objective is None:
                    loss = mse
                elif isinstance(objective, Acc30BoundaryLoss):
                    loss = objective(prediction, target, batch_mean, batch_std)
                else:
                    power_dbloss = objective(prediction, target)
                    loss = mse + dbloss_weight * power_dbloss

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

            # 验证仅保留点误差指标。
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


def dump_weather_forecasts(
    model,
    datasets,
    batch_size,
    num_workers,
    device,
    run_dir,
    show_progress,
):
    # 程序1：按窗口写出 train/val/test 未来气象，供程序2读取
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


def write_power_result_contract(
    args,
    repository,
    datasets,
    checkpoint,
    best_epoch,
    best_mse,
    history,
    curve_overall,
):
    # 将单 seed 最优模型与测试指标写入正式实验契约。
    test_metrics = {
        "Acc30": curve_overall["strict_acc30"],
        "MAE": curve_overall["mae_kw"],
        "MSE": curve_overall["mse_kw2"],
        "RMSE": curve_overall["rmse_kw"],
        "MAPE": curve_overall["mape"],
    }
    round_record = {
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_valid": {"MSE_scaled": best_mse},
        "test_metrics": test_metrics,
        "checkpoint": checkpoint.as_posix(),
        "early_stop": {
            "patience": args.patience,
            "completed_epochs": len(history),
        },
    }
    rounds = [round_record]
    mean_std = aggregate_test_metrics(rounds)

    available_samples = 0
    for dataset in datasets.values():
        available_samples += len(dataset)

    condition = {
        "domain": "wind_power_forecasting",
        "scenario": args.scenario,
        "sample_seconds": args.sample_seconds,
        "history_steps": args.history_steps,
        "horizon_steps": args.horizon_steps,
        "window_stride_steps": args.window_stride_steps,
        "loss": args.loss,
        "dbloss_weight": args.dbloss_weight,
    }

    if args.model == "PatchMLPAllFeatures":
        feature_fusion = "cross_variable_mlp"
        moving_average_kernel = 13
        temporal_projection = "multi_scale_patch_mlp"
        purpose = "全部有效历史特征的PatchMLP功率预测"
    else:
        feature_fusion = "learned_linear_projection"
        moving_average_kernel = 25
        temporal_projection = "shared_across_channels"
        purpose = "全部有效历史特征的DLinear功率预测"

    model_config = {
        "input_features": len(repository.feature_names),
        "feature_fusion": feature_fusion,
        "moving_average_kernel": moving_average_kernel,
        "temporal_projection": temporal_projection,
        "turbine_embedding": True,
        "revin": True,
    }
    train_config = {
        "seed": args.seed,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "num_workers": args.num_workers,
        "loss": args.loss,
        "dbloss_weight": args.dbloss_weight,
        "selection_metric": "validation_mse_scaled",
    }
    data_config = {
        "source": "dataset/processed/turbine_01.npy ... turbine_16.npy",
        "available_samples": available_samples,
        "split": "chronological_70_10_20",
        "split_seeds": "none",
        "train_samples": len(datasets["train"]),
        "valid_samples": len(datasets["val"]),
        "test_samples": len(datasets["test"]),
        "features": ", ".join(repository.feature_names),
        "feature_normalization": "per_turbine_train_standard_scaler_then_window_revin",
        "target": "风机-P",
        "target_transform": "per_turbine_train_standard_scaler",
    }
    result = {
        "contract_version": CONTRACT_VERSION,
        "dataset": args.dataset_name,
        "result_name": args.result_name,
        "model_name": args.model,
        "purpose": purpose,
        "condition": condition,
        "model_config": model_config,
        "train_config": train_config,
        "data_config": data_config,
        "mean_std": mean_std,
        "significance": [],
        "rounds": rounds,
    }
    result["significance"] = build_dataset_significance(
        args.dataset_name,
        args.result_name,
        result,
    )

    report_path = result_report_path(args.dataset_name, args.result_name)
    record_path = result_record_path(args.dataset_name, args.result_name)
    write_result_report(report_path, result)
    write_result_record(record_path, result)
    return report_path, record_path


def train_and_test(args) -> dict:
    seed_everything(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    no_future_weather = args.scenario == "NoFutureWeather"
    weather_task = args.scenario == "ForecastWeather"
    predicted_weather = args.scenario == "PredictedFutureWeather"
    oracle = args.scenario == "OracleFutureWeather"
    joint_panel = args.model == "StockEcho"
    qwen_mlp = args.model == "QwenMLP"
    timer_weather_mlp = args.model == "TimerWeatherMLP"
    all_features = args.model in ("DLinearAllFeatures", "PatchMLPAllFeatures")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    horizon_steps = args.horizon_steps

    if weather_task:
        repository, datasets, loaders = make_forecast_weather_loaders(
            horizon=horizon_steps,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        print_data_summary(
            datasets,
            loaders,
            horizon_steps,
            args.window_stride_steps,
            no_future_weather,
            weather_task,
            oracle,
            predicted_weather,
        )
        model = WeatherForecastModel(args.model, horizon_steps).to(device)
        model_uses_future_weather = True
    elif predicted_weather:
        # 程序2：天气来自程序1的 npy
        repository, datasets, loaders = make_predicted_future_weather_loaders(
            horizon=horizon_steps,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            weather_forecast_dir=args.weather_forecast_dir,
        )
        print_data_summary(
            datasets,
            loaders,
            horizon_steps,
            args.window_stride_steps,
            no_future_weather,
            weather_task,
            oracle,
            predicted_weather,
        )
        if timer_weather_mlp:
            from models.backbone.timer import TimerWeatherMLP

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
            model = FutureWeatherModel(args.model, horizon_steps).to(device)
        model_uses_future_weather = True
    elif joint_panel:
        if oracle:
            repository, datasets, loaders = (
                make_multi_turbine_oracle_future_weather_loaders(
                    horizon=horizon_steps,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                )
            )
            model = StockEchoWindPower(horizon_steps).to(device)
            model_uses_future_weather = True
        else:
            repository, datasets, loaders = (
                make_multi_turbine_no_future_weather_loaders(
                    horizon=horizon_steps,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                )
            )
            model = StockEchoNoFutureWeather(horizon_steps).to(device)
            model_uses_future_weather = False
        print_data_summary(
            datasets,
            loaders,
            horizon_steps,
            args.window_stride_steps,
            no_future_weather,
            weather_task,
            oracle,
            predicted_weather,
        )
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
                all_features=all_features,
            )
        print_data_summary(
            datasets,
            loaders,
            horizon_steps,
            args.window_stride_steps,
            no_future_weather,
            weather_task,
            oracle,
            predicted_weather,
        )
        if qwen_mlp:
            from models.backbone.qwen import QwenMLP

            model = QwenMLP(
                horizon=horizon_steps,
                pretrained_path=args.llm_path,
                patch_length=args.llm_patch_length,
                patch_stride=args.llm_patch_stride,
                bottleneck=args.llm_bottleneck,
                gradient_checkpointing=bool(args.llm_gradient_checkpointing),
            ).to(device)
            model_uses_future_weather = False
        elif timer_weather_mlp:
            if not (oracle or predicted_weather):
                raise ValueError(
                    "TimerWeatherMLP requires OracleFutureWeather or PredictedFutureWeather"
                )
            from models.backbone.timer import TimerWeatherMLP

            model = TimerWeatherMLP(
                horizon=horizon_steps,
                pretrained_path=args.timer_path,
                patch_length=args.timer_patch_length,
                bottleneck=args.timer_bottleneck,
                unfreeze_layers=args.timer_unfreeze_layers,
                gradient_checkpointing=bool(args.timer_gradient_checkpointing),
                residual_forecast=bool(args.timer_residual_forecast),
            ).to(device)
            model_uses_future_weather = True
        elif no_future_weather:
            model = NoFutureWeatherModel(
                args.model,
                horizon_steps,
                len(repository.feature_names),
                repository.power_index,
            ).to(device)
            model_uses_future_weather = False
        else:
            model = FutureWeatherModel(args.model, horizon_steps).to(device)
            model_uses_future_weather = True

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
    select_acc30 = args.loss == "MSEAcc30"
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
    config["sample_seconds"] = SAMPLE_SECONDS
    config["history_steps"] = HISTORY_STEPS
    config["horizon_steps"] = horizon_steps
    config["oracle_future_weather"] = oracle
    config["weather_task"] = weather_task
    config["predicted_future_weather"] = predicted_weather
    config["selection_metric"] = selection_metric
    config["acc30_loss_weight"] = 0.3 if args.loss == "MSEAcc30" else None
    config["acc30_loss_temperature"] = 0.03 if args.loss == "MSEAcc30" else None
    train_sample = datasets["train"][0]
    if no_future_weather:
        sample_past, sample_target, _, _ = train_sample
    else:
        sample_past, sample_weather, sample_target, _, _ = train_sample
        config["weather_shape"] = list(sample_weather.shape)

    config["train_windows"] = len(datasets["train"])
    config["val_windows"] = len(datasets["val"])
    config["test_windows"] = len(datasets["test"])
    config["feature_names"] = repository.feature_names
    config["input_feature_count"] = len(repository.feature_names)
    config["past_shape"] = list(sample_past.shape)
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
    contract_checkpoint = checkpoint
    if args.result_name:
        contract_checkpoint = result_checkpoint_path(args.dataset_name, args.result_name, 1)
        contract_checkpoint.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train_loss, train_mse, train_acc30 = run_epoch(
            model,
            loaders["train"],
            device,
            optimizer,
            objective,
            args.dbloss_weight,
            True,
            power_means,
            power_stds,
            args.show_progress,
            epoch,
            no_future_weather,
            model_uses_future_weather,
            weather_task,
        )
        _, val_mse, val_acc30 = run_epoch(
            model,
            loaders["val"],
            device,
            optimizer,
            objective,
            args.dbloss_weight,
            False,
            power_means,
            power_stds,
            args.show_progress,
            epoch,
            no_future_weather,
            model_uses_future_weather,
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
            log_epoch = epoch == 1 or epoch % 10 == 0
            if epoch == args.epochs:
                log_epoch = True
            if log_epoch:
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
            log_epoch = epoch == 1 or epoch % 10 == 0
            if epoch == args.epochs:
                log_epoch = True
            if log_epoch:
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
            checkpoint_payload = {
                "model_state": model_state,
                "epoch": epoch,
                "config": config,
                "contract_version": CONTRACT_VERSION,
                "dataset": args.dataset_name,
                "result_name": args.result_name,
                "model_name": args.model,
                "seed": args.seed,
                "target_transform": "per_turbine_train_standard_scaler",
            }
            torch.save(checkpoint_payload, checkpoint)
            if args.result_name:
                torch.save(checkpoint_payload, contract_checkpoint)
            print(
                f"[Power] checkpoint  epoch={epoch}  val_mse={val_mse:.6f}  "
                f"path={contract_checkpoint}",
                flush=True,
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
    history_minutes = HISTORY_STEPS * SAMPLE_SECONDS / 60
    horizon_minutes = horizon_steps * SAMPLE_SECONDS / 60
    window_stride_minutes = args.window_stride_steps * SAMPLE_SECONDS / 60

    # 程序1：写出三份天气 npy，并在 test 上记天气 MSE
    if weather_task:
        dump_weather_forecasts(
            model,
            datasets,
            args.batch_size,
            args.num_workers,
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
            num_workers=args.num_workers,
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
            "history_steps": HISTORY_STEPS,
            "horizon_steps": horizon_steps,
            "sample_seconds": SAMPLE_SECONDS,
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
        (run_dir / "final.json").write_text(
            json.dumps(final, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log_line = (
            f"history={history_minutes:g}min horizon={horizon_minutes:g}min "
            f"window_stride={window_stride_minutes:g}min "
            f"val_mse_scaled={best_mse:.6f} test_mse_scaled={test_mse_scaled:.6f} "
            f"model={args.model} scenario={args.scenario} best_epoch={best_epoch}\n"
        )
        with (PROJECT_ROOT / "run.log").open("a", encoding="utf-8") as handle:
            handle.write(log_line)
        print(
            json.dumps(
                {
                    "event": "complete",
                    "run_dir": str(run_dir),
                    "checkpoint": str(checkpoint),
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
        for batch in test_batches:
            if no_future_weather:
                past, target, turbine_id, target_start = batch
            else:
                past, future_weather, target, turbine_id, target_start = batch

            past = past.to(device, non_blocking=True)
            turbine_gpu = turbine_id.to(device, non_blocking=True)

            if model_uses_future_weather:
                future_weather = future_weather.to(device, non_blocking=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                if model_uses_future_weather:
                    prediction = model(past, future_weather, turbine_gpu)
                else:
                    prediction = model(past, turbine_gpu)

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

    report_path = ""
    record_path = ""
    if args.result_name:
        report_path, record_path = write_power_result_contract(
            args,
            repository,
            datasets,
            contract_checkpoint,
            best_epoch,
            best_mse,
            history,
            curve_overall,
        )

    if select_acc30:
        best_val_strict_acc30 = best_acc30
    else:
        best_val_strict_acc30 = None

    final = {
        "status": "complete",
        "model": args.model,
        "scenario": args.scenario,
        "loss": args.loss,
        "history_steps": HISTORY_STEPS,
        "horizon_steps": horizon_steps,
        "sample_seconds": SAMPLE_SECONDS,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_val_mse_scaled": best_mse,
        "best_val_strict_acc30": best_val_strict_acc30,
        "checkpoint": str(checkpoint),
        "contract_checkpoint": str(contract_checkpoint),
        "report": str(report_path),
        "record": str(record_path),
        "weather_forecast_dir": args.weather_forecast_dir if predicted_weather else "",
        "metrics": metrics,
    }
    (run_dir / "final.json").write_text(
        json.dumps(final, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log_line = (
        f"history={history_minutes:g}min horizon={horizon_minutes:g}min "
        f"window_stride={window_stride_minutes:g}min "
        f"test_acc30={curve_overall['strict_acc30']:.2f}%  "
        f"test_mape={curve_overall['mape']:.2f}%  "
        f"test_mae={curve_overall['mae_kw']:.2f}  "
        f"test_rmse={curve_overall['rmse_kw']:.2f}  "
        f"model={args.model} scenario={args.scenario} best_epoch={best_epoch}\n"
    )
    with (PROJECT_ROOT / "run.log").open("a", encoding="utf-8") as handle:
        handle.write(log_line)
    print(
        f"[Power] test  best_epoch={best_epoch}  "
        f"val_mse={best_mse:.6f}  val_acc30={best_acc30:.2f}%  "
        f"test_acc30={curve_overall['strict_acc30']:.2f}%  "
        f"test_mae={curve_overall['mae_kw']:.2f}  "
        f"test_mse_kw2={curve_overall['mse_kw2']:.2f}  "
        f"test_rmse={curve_overall['rmse_kw']:.2f}  "
        f"test_mape={curve_overall['mape']:.2f}%  "
        f"checkpoint={checkpoint}",
        flush=True,
    )
    return final
