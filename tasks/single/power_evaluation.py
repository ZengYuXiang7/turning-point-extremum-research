import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import config as project_config
from config import BASE_INTERVAL_SECONDS, PROJECT_ROOT
from observability import result_record_path, result_report_path
from tasks.single.result import write_power_result_contract
from utils.metrics import metric_bundle, save_metrics


def collect_predictions(args, model, loaders, device, model_uses_future_weather):
    # Single 功率任务逐窗口收集预测、标签和风机编号。
    predictions = []
    truths = []
    turbines = []
    target_starts = []
    batches = tqdm(loaders["test"], total=len(loaders["test"]), desc="test", disable=args.show_progress == 0, leave=False,)
    no_future_weather = args.scenario == "NoFutureWeather"
    with torch.no_grad():
        for batch in batches:
            if no_future_weather:
                past, target, turbine_id, target_start = batch
            else:
                past, future_weather, target, turbine_id, target_start = batch
            past = past.to(device, non_blocking=True)
            turbine_gpu = turbine_id.to(device, non_blocking=True)
            if model_uses_future_weather:
                future_weather = future_weather.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda",):
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
    return prediction_scaled, truth_scaled, turbine_ids, starts


def restore_power(repository, prediction_scaled, truth_scaled, turbine_ids):
    # 按各风机训练统计量还原功率单位。
    mean_values = []
    std_values = []
    for turbine_index in turbine_ids:
        series = repository.series[int(turbine_index)]
        mean_values.append(series.power_mean)
        std_values.append(series.power_std)
    means = np.asarray(mean_values, dtype=np.float32)[:, None]
    stds = np.asarray(std_values, dtype=np.float32)[:, None]
    prediction = prediction_scaled * stds + means
    truth = truth_scaled * stds + means
    return prediction, truth


def evaluate_power(args, model, repository, datasets, loaders, device, checkpoint, contract_checkpoint, best_epoch, best_mse, best_acc30, history, pretraining, model_uses_future_weather,):
    # Single 功率程序保存预测、指标和正式结果契约。
    run_dir = Path(args.run_dir)
    prediction_scaled, truth_scaled, turbine_ids, starts = collect_predictions(args, model, loaders, device, model_uses_future_weather)
    prediction, truth = restore_power(repository, prediction_scaled, truth_scaled, turbine_ids)
    np.savez_compressed(run_dir / "test_predictions.npz", prediction=prediction.astype(np.float32), truth=truth.astype(np.float32), turbine_id=(turbine_ids + 1).astype(np.int16), target_start_ns=starts,)
    metrics = metric_bundle(prediction, truth, turbine_ids)
    save_metrics(run_dir, metrics)
    curve_overall = metrics[0]

    report_path = ""
    record_path = ""
    if args.result_name:
        if args.mode == "train":
            report_path, record_path = write_power_result_contract(args, repository, datasets, contract_checkpoint, best_epoch, best_mse, history, curve_overall, pretraining,)
        else:
            report_path = result_report_path(args.dataset_name, args.result_name)
            record_path = result_record_path(args.dataset_name, args.result_name)

    best_val_acc30 = None
    if args.mode == "train":
        if args.loss == "MSEAcc30":
            best_val_acc30 = best_acc30
    status = "complete"
    final_path = run_dir / "final.json"
    if args.mode == "test":
        status = "tested"
        final_path = run_dir / "test_final.json"
    weather_forecast_dir = ""
    if args.scenario == "PredictedFutureWeather":
        weather_forecast_dir = args.weather_forecast_dir
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
        "best_val_strict_acc30": best_val_acc30,
        "checkpoint": str(checkpoint),
        "contract_checkpoint": str(contract_checkpoint),
        "report": str(report_path),
        "record": str(record_path),
        "weather_forecast_dir": weather_forecast_dir,
        "metrics": metrics,
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
        f"test_acc30={curve_overall['strict_acc30']:.2f}%  "
        f"test_mape={curve_overall['mape']:.2f}%  "
        f"test_mae={curve_overall['mae_kw']:.2f}  "
        f"test_rmse={curve_overall['rmse_kw']:.2f}  "
        f"model={args.model} scenario={args.scenario} best_epoch={best_epoch}\n"
    )
    if args.mode == "test":
        log_line = "mode=test " + log_line
    with (PROJECT_ROOT / "run.log").open("a", encoding="utf-8") as handle:
        handle.write(log_line)
    metric_text = (
        f"test_acc30={curve_overall['strict_acc30']:.2f}%  "
        f"test_mape={curve_overall['mape']:.2f}%"
    )
    if args.mode == "train":
        metric_text = f"val_acc30={best_acc30:.2f}%  {metric_text}"
    display_checkpoint = checkpoint.resolve()
    if display_checkpoint.is_relative_to(PROJECT_ROOT):
        display_checkpoint = display_checkpoint.relative_to(PROJECT_ROOT)
    print(f"[Power] {args.mode} best_epoch={best_epoch} {metric_text} " f"checkpoint={display_checkpoint.as_posix()}", flush=True,)
    return final
