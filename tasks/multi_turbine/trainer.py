import json
from pathlib import Path

from config import PROJECT_ROOT
from observability import result_record_path
from tasks.multi_turbine.evaluation import test_checkpoint
from tasks.multi_turbine.fit import fit_model
from tasks.multi_turbine.pretrain import pretrain_satra_encoder
from tasks.multi_turbine.summary import print_data_summary
from tasks.multi_turbine.training_setup import (
    build_optimization,
    build_power_scales,
    build_training_config,
    compile_model,
    prepare_task_runtime,
    write_training_config,
)


def test_existing_result(args, model, repository, datasets, loaders, device):
    # 多风机测试模式只读取联合面板的正式 record。
    record_path = result_record_path(args.dataset_name, args.result_name)
    result = json.loads(record_path.read_text(encoding="utf-8"))
    round_record = result["rounds"][0]
    checkpoint = PROJECT_ROOT / round_record["checkpoint"]
    best_epoch = round_record["best_epoch"]
    best_mse = round_record["best_valid"]["MSE_scaled"]
    tested = test_checkpoint(args, model, repository, datasets, loaders, device, checkpoint, checkpoint, best_epoch, best_mse, None, [], None,)
    return tested


def train_task(args, repository, datasets, loaders, model, device) -> dict:
    # 多风机训练器只处理联合面板场景。
    print_data_summary(datasets, loaders, args.horizon_steps, args.window_stride_steps,)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    pretraining = None
    if args.mode == "train":
        if args.pretrain:
            pretraining = pretrain_satra_encoder(args, model, repository, datasets, loaders, device, run_dir)
    model, compile_mode = compile_model(args, model)
    if args.mode == "test":
        result = test_existing_result(args, model, repository, datasets, loaders, device)
        return result

    objective, optimizer, scheduler, select_acc30, selection_metric = (
        build_optimization(args, model, device)
    )
    power_means, power_stds = build_power_scales(repository, device)
    config = build_training_config(args, model, repository, datasets, pretraining, compile_mode, selection_metric,)
    write_training_config(run_dir, config)
    (checkpoint, contract_checkpoint, best_epoch, best_mse, best_acc30, history,) = fit_model(args, model, loaders, device, objective, optimizer, scheduler, select_acc30, power_means, power_stds, config, run_dir,)
    result = test_checkpoint(args, model, repository, datasets, loaders, device, checkpoint, contract_checkpoint, best_epoch, best_mse, best_acc30, history, pretraining,)
    return result
