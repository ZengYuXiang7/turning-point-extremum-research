import json
import logging

import torch

import config as project_config
from config import BASE_INTERVAL_SECONDS
from utils.acc30_loss import Acc30BoundaryLoss
from utils.dbloss import DBLoss
from utils.reproducibility import seed_everything


def prepare_task_runtime(seed: int) -> torch.device:
    # Single 程序独立设置随机状态与计算精度。
    seed_everything(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return device


def compile_model(args, model):
    # Qwen 与 Timer 保持原先不编译的执行策略。
    logging.getLogger("torch.fx.experimental.symbolic_shapes").setLevel(logging.ERROR)
    if args.model == "QwenMLP":
        compile_mode = "disabled_for_qwen_mlp"
    elif args.model == "TimerWeatherMLP":
        compile_mode = "disabled_for_timer_weather_mlp"
    else:
        model = torch.compile(model, mode="reduce-overhead")
        compile_mode = "reduce-overhead"
    return model, compile_mode


def build_optimization(args, model, device):
    # Single 程序独立建立损失、优化器和验证调度器。
    if args.loss == "MSE":
        objective = None
    elif args.loss == "DBLoss":
        objective = DBLoss(alpha=0.2, beta=0.5).to(device)
    else:
        objective = Acc30BoundaryLoss(weight=0.3, temperature=0.03).to(device)

    trainable_parameters = []
    for parameter in model.parameters():
        if parameter.requires_grad:
            trainable_parameters.append(parameter)
    if args.model == "TimerWeatherMLP":
        optimizer = torch.optim.AdamW(
            model.backbone.parameter_groups(
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
        optimizer,
        mode=scheduler_mode,
        factor=0.5,
        patience=3,
    )
    return objective, optimizer, scheduler, select_acc30, selection_metric


def build_power_scales(repository, device):
    # Single batch 按风机编号索引训练段标准化统计量。
    mean_values = []
    std_values = []
    for series in repository.series:
        mean_values.append(series.power_mean)
        std_values.append(series.power_std)
    power_means = torch.tensor(mean_values, dtype=torch.float32, device=device)
    power_stds = torch.tensor(std_values, dtype=torch.float32, device=device)
    return power_means, power_stds


def build_training_config(
    args,
    model,
    repository,
    datasets,
    model_uses_future_weather,
    pretraining,
    compile_mode,
    selection_metric,
):
    # Single checkpoint 只记录自身 Dataset、Model 与场景协议。
    oracle = args.scenario == "OracleFutureWeather"
    weather_task = args.scenario == "ForecastWeather"
    predicted_weather = args.scenario == "PredictedFutureWeather"
    no_future_weather = args.scenario == "NoFutureWeather"
    config = vars(args).copy()
    if args.pretrain:
        config["pretraining"] = pretraining
    config["base_interval_seconds"] = BASE_INTERVAL_SECONDS
    config["point_interval_seconds"] = project_config.POINT_INTERVAL_SECONDS
    config["history_steps"] = project_config.HISTORY_STEPS
    config["horizon_steps"] = args.horizon_steps
    config["oracle_future_weather"] = oracle
    config["weather_task"] = weather_task
    config["predicted_future_weather"] = predicted_weather
    config["selection_metric"] = selection_metric
    config["acc30_loss_weight"] = None
    config["acc30_loss_temperature"] = None
    if args.loss == "MSEAcc30":
        config["acc30_loss_weight"] = 0.3
        config["acc30_loss_temperature"] = 0.03

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
    parameter_count = 0
    trainable_parameter_count = 0
    for parameter in model.parameters():
        parameter_count += parameter.numel()
        if parameter.requires_grad:
            trainable_parameter_count += parameter.numel()
    config["parameter_count"] = int(parameter_count)
    config["trainable_parameter_count"] = int(trainable_parameter_count)
    config["joint_multi_turbine_panel"] = False
    config["model_uses_future_weather"] = model_uses_future_weather
    config["amp_dtype"] = "bf16"
    config["tf32"] = True
    config["compile_mode"] = compile_mode
    config["max_clip_norm"] = 1.0
    return config


def write_training_config(run_dir, config) -> None:
    # 配置与 checkpoint 使用同一个 single 运行目录。
    config_path = run_dir / "config.json"
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def print_weather_baseline(args, datasets, loaders, device, config, run_dir) -> None:
    # 天气任务以历史末点持续法作为启动基线。
    if args.scenario == "ForecastWeather":
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
        write_training_config(run_dir, config)
