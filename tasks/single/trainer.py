import json
from pathlib import Path

from config import PROJECT_ROOT
from observability import result_record_path
from tasks.single.evaluation import test_checkpoint
from tasks.single.fit import fit_model
from tasks.single.pretrain import pretrain_power_encoder
from tasks.single.summary import print_data_summary
from tasks.single.training_setup import (
    build_optimization,
    build_power_scales,
    build_training_config,
    compile_model,
    prepare_task_runtime,
    print_weather_baseline,
    write_training_config,
)


def test_existing_result(args, model, repository, datasets, loaders, device, model_uses_future_weather,):
    # Single 测试模式只读取自己的正式 record。
    record_path = result_record_path(args.dataset_name, args.result_name)
    if not record_path.exists():
        raise FileNotFoundError(
            f"未找到结果记录 {record_path}："
            "请先用相同的 --result-name 与 --run-dir 跑完 train，生成 record 后再执行 test。"
        )
    result = json.loads(record_path.read_text(encoding="utf-8"))
    round_record = result["rounds"][0]
    checkpoint = PROJECT_ROOT / round_record["checkpoint"]
    best_epoch = round_record["best_epoch"]
    best_mse = round_record["best_valid"]["MSE_scaled"]
    tested = test_checkpoint(args, model, repository, datasets, loaders, device, checkpoint, checkpoint, best_epoch, best_mse, None, [], None, model_uses_future_weather,)
    return tested


def train_task(args, repository, datasets, loaders, model, device, model_uses_future_weather,) -> dict:
    # Single 训练器只处理单风机与天气场景。
    no_future_weather = args.scenario == "NoFutureWeather"
    weather_task = args.scenario == "ForecastWeather"
    oracle = args.scenario == "OracleFutureWeather"
    predicted_weather = args.scenario == "PredictedFutureWeather"
    provided_weather = args.scenario == "ProvidedFutureWeather"
    print_data_summary(datasets, loaders, args.horizon_steps, args.window_stride_steps, no_future_weather, weather_task, oracle, predicted_weather, provided_weather,)

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    pretraining = None
    if args.mode == "train":
        if args.pretrain:
            pretraining = pretrain_power_encoder(args, model, repository, datasets, loaders, device, run_dir)
    model, compile_mode = compile_model(args, model)
    if args.mode == "test":
        result = test_existing_result(args, model, repository, datasets, loaders, device, model_uses_future_weather,)
        return result

    objective, optimizer, scheduler, selection_metric = build_optimization(args, model, device)
    power_means, power_stds = build_power_scales(repository, device)
    config = build_training_config(args, model, repository, datasets, model_uses_future_weather, pretraining, compile_mode, selection_metric,)
    write_training_config(run_dir, config)
    print_weather_baseline(args, datasets, loaders, device, config, run_dir)
    (checkpoint, contract_checkpoint, best_epoch, best_mse, best_acc30, history,) = fit_model(args, model, datasets, loaders, device, model_uses_future_weather, objective, optimizer, scheduler, power_means, power_stds, config, run_dir,)
    result = test_checkpoint(args, model, repository, datasets, loaders, device, checkpoint, contract_checkpoint, best_epoch, best_mse, best_acc30, history, pretraining, model_uses_future_weather,)
    return result
