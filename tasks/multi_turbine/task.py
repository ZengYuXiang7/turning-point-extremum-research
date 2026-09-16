from __future__ import annotations

from argparse import Namespace

import torch

from models.multi_turbine import MultiTurbineModel
from tasks.multi_turbine.dataset import (
    make_multi_turbine_no_future_weather_loaders,
)
from tasks.multi_turbine.trainer import prepare_task_runtime, train_task


def build_task_components(args, device: torch.device, num_workers: int):
    # MultiTurbine 模型按开关决定是否读取训练段 DTW 关系。
    if args.model == "MultiTurbine":
        use_dtw_prior = bool(args.satra_use_spatial_relation)
        if use_dtw_prior:
            use_dtw_prior = bool(args.satra_use_dtw_prior)
        if use_dtw_prior:
            repository, datasets, loaders = make_multi_turbine_no_future_weather_loaders(horizon=args.horizon_steps, batch_size=args.batch_size, num_workers=num_workers, all_features=True, pstr_relation_top_k=args.satra_prior_top_k, pstr_relation_candidate_k=args.satra_prior_candidate_k, pstr_relation_downsample=args.satra_prior_downsample, pstr_relation_band=args.satra_prior_band,)
            relation_weight = repository.pstr_relation_weight
        else:
            repository, datasets, loaders = make_multi_turbine_no_future_weather_loaders(horizon=args.horizon_steps, batch_size=args.batch_size, num_workers=num_workers, all_features=True,)
            relation_weight = None
    elif args.model == "StockEcho":
        repository, datasets, loaders = make_multi_turbine_no_future_weather_loaders(horizon=args.horizon_steps, batch_size=args.batch_size, num_workers=num_workers,)
        relation_weight = None

    model = MultiTurbineModel(args, repository, relation_weight).to(device)
    return repository, datasets, loaders, model


def build_checkpoint_components(config, split: str, device: torch.device):
    # checkpoint 配置直接复用同一个多风机 Dataset 与 Model 构建链。
    args = Namespace(**config)
    repository, datasets, _, model = (
        build_task_components(args, device, num_workers=0)
    )
    dataset = datasets[split]
    return repository, dataset, model, False


def load_checkpoint_model(model, config, checkpoint):
    # 使用多风机训练阶段相同的编译包装载入最佳权重。
    if config["compile_mode"] == "reduce-overhead":
        model = torch.compile(model, mode="reduce-overhead")
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint["epoch"]


def train_and_test(args) -> dict:
    if args.scenario != "MultiTurbine":
        raise ValueError("multi_turbine_task requires scenario MultiTurbine")

    if args.model not in ("StockEcho", "MultiTurbine"):
        raise ValueError("MultiTurbine task supports StockEcho and MultiTurbine only")

    device = prepare_task_runtime(args.seed)

    repository, datasets, loaders, model = (
        build_task_components(args, device, args.num_workers)
    )
    
    result = train_task(args, repository, datasets, loaders, model, device)
    return result
