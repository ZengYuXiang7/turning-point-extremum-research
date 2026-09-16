from observability import (
    CONTRACT_VERSION,
    aggregate_test_metrics,
    build_dataset_significance,
    result_record_path,
    result_report_path,
    write_result_record,
    write_result_report,
)


def select_model_metadata(model_name: str):
    # Single 模型元数据保持原实验契约含义。
    if model_name == "PatchMLPAllFeatures":
        feature_fusion = "cross_variable_mlp"
        moving_average_kernel = 13
        temporal_projection = "multi_scale_patch_mlp"
        purpose = "全部有效历史特征的PatchMLP功率预测"
    elif model_name == "DLinearCorrelatedFeatures":
        feature_fusion = "learned_linear_projection"
        moving_average_kernel = 25
        temporal_projection = "shared_across_channels"
        purpose = "20个绝对相关系数不低于0.6的预测变量加历史功率的DLinear预测"
    else:
        feature_fusion = "learned_linear_projection"
        moving_average_kernel = 25
        temporal_projection = "shared_across_channels"
        purpose = "全部有效历史特征的DLinear功率预测"
    return feature_fusion, moving_average_kernel, temporal_projection, purpose


def write_power_result_contract(args, repository, datasets, checkpoint, best_epoch, best_mse, best_acc30, history, curve_overall, pretraining,):
    # 写入 single 功率任务的正式 report 与 record。
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
        "best_valid": {"Acc30": best_acc30, "MSE_scaled": best_mse},
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
        "base_interval_seconds": repository.base_interval_seconds,
        "point_interval_seconds": args.point_interval_seconds,
        "history_steps": args.history_steps,
        "horizon_steps": args.horizon_steps,
        "window_stride_steps": args.window_stride_steps,
        "loss": args.loss,
        "dbloss_weight": args.dbloss_weight,
    }
    feature_fusion, moving_average_kernel, temporal_projection, purpose = (
        select_model_metadata(args.model)
    )
    if args.scenario == "ProvidedFutureWeather":
        purpose = "历史场站数据与甲方未来预测风速的功率预测"
    model_config = {
        "input_features": len(repository.feature_names),
        "feature_fusion": feature_fusion,
        "moving_average_kernel": moving_average_kernel,
        "temporal_projection": temporal_projection,
        "turbine_embedding": len(repository.series) > 1,
        "turbine_embedding_relation": "none",
        "embedding_relation_mix_initial": None,
        "revin": args.scenario == "NoFutureWeather",
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
        "selection_metric": "validation_strict_acc30",
    }
    if args.pretrain:
        purpose = "随机历史时间token重建热身后的PatchMLP功率预测"
        round_record["pretraining"] = pretraining
        train_config["pretrain_epochs"] = args.pretrain_epochs
        train_config["pretrain_learning_rate"] = args.pretrain_learning_rate
        train_config["pretrain_lr_scheduler"] = args.pretrain_lr_scheduler
        train_config["mae_mask_ratio"] = args.mae_mask_ratio
        train_config["pretrain_patience"] = args.pretrain_patience
        train_config["pretrain_selection_metric"] = (
            "fixed_mask_validation_reconstruction_mse"
        )
        train_config["pretrain_checkpoint"] = pretraining["checkpoint"]
        train_config["pretrain_best_epoch"] = pretraining["best_epoch"]
        train_config["pretrain_normalization"] = pretraining["normalization"]

    data_config = {
        "source": repository.source_description,
        "available_samples": available_samples,
        "split": repository.split_description,
        "split_seeds": "none",
        "train_samples": len(datasets["train"]),
        "valid_samples": len(datasets["val"]),
        "test_samples": len(datasets["test"]),
        "features": ", ".join(repository.feature_names),
        "feature_normalization": repository.normalization_description,
        "target": repository.target_name,
        "target_transform": repository.target_transform_description,
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
    result["significance"] = build_dataset_significance(args.dataset_name, args.result_name, result,)
    report_path = result_report_path(args.dataset_name, args.result_name)
    record_path = result_record_path(args.dataset_name, args.result_name)
    write_result_report(report_path, result)
    write_result_record(record_path, result)
    return report_path, record_path
