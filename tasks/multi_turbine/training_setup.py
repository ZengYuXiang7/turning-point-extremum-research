import json
import logging

import torch

import config as project_config
from config import BASE_INTERVAL_SECONDS
from utils.acc30_loss import Acc30BoundaryLoss
from utils.dbloss import DBLoss
from utils.reproducibility import seed_everything


def prepare_task_runtime(seed: int) -> torch.device:
    # 多风机程序独立设置随机状态与计算精度。
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
    # 稀疏 Top-3 MoE 保持 eager，StockEcho 继续编译。
    logging.getLogger("torch.fx.experimental.symbolic_shapes").setLevel(logging.ERROR)
    if args.model in ("MultiTurbine", "Model2"):
        compile_mode = "disabled_for_multi_turbine_sparse_top3_moe"
    else:
        model = torch.compile(model, mode="reduce-overhead")
        compile_mode = "reduce-overhead"
    return model, compile_mode


def build_optimization(args, model, device):
    # 多风机程序独立建立监督优化流程。
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
    optimizer = torch.optim.Adam(trainable_parameters, lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3,)
    selection_metric = "validation_strict_acc30"
    return objective, optimizer, scheduler, selection_metric


def build_power_scales(repository, device):
    # 联合面板的功率标签共用全部风机训练段拟合的统计量。
    turbine_count = len(repository.series)
    power_mean = repository.series[0].power_mean
    power_std = repository.series[0].power_std
    power_means = torch.full((turbine_count,), power_mean, dtype=torch.float32, device=device)
    power_stds = torch.full((turbine_count,), power_std, dtype=torch.float32, device=device)
    return power_means, power_stds


def build_training_config(args, model, repository, datasets, pretraining, compile_mode, selection_metric,):
    # 多风机 checkpoint 只记录联合面板协议。
    config = vars(args).copy()
    if args.pretrain:
        config["pretraining"] = pretraining
    config["base_interval_seconds"] = BASE_INTERVAL_SECONDS
    config["point_interval_seconds"] = project_config.POINT_INTERVAL_SECONDS
    config["history_steps"] = project_config.HISTORY_STEPS
    config["horizon_steps"] = args.horizon_steps
    config["oracle_future_weather"] = False
    config["weather_task"] = False
    config["predicted_future_weather"] = False
    config["selection_metric"] = selection_metric
    config["acc30_loss_weight"] = None
    config["acc30_loss_temperature"] = None
    if args.loss == "MSEAcc30":
        config["acc30_loss_weight"] = 0.3
        config["acc30_loss_temperature"] = 0.03
    sample_past, sample_target, _, _ = datasets["train"][0]
    config["train_windows"] = len(datasets["train"])
    config["val_windows"] = len(datasets["val"])
    config["test_windows"] = len(datasets["test"])
    config["feature_names"] = repository.feature_names
    config["input_feature_count"] = len(repository.feature_names)
    feature_normalization = repository.normalization_description
    if args.model in ("MultiTurbine", "Model2") and args.revin:
        feature_normalization = f"{feature_normalization}_then_window_revin"
    config["feature_normalization"] = feature_normalization
    config["target_transform"] = repository.target_transform_description
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
    config["joint_multi_turbine_panel"] = True
    config["amp_dtype"] = "bf16"
    config["tf32"] = True
    config["compile_mode"] = compile_mode
    config["max_clip_norm"] = 1.0
    return config


def write_training_config(run_dir, config) -> None:
    # 配置与 checkpoint 使用同一个多风机运行目录。
    config_path = run_dir / "config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8",)
