from config import BASE_INTERVAL_SECONDS
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
    # 多风机模型元数据仅包含联合面板实现。
    if model_name == "StockEcho":
        feature_fusion = "dynamic_relation_graph"
        temporal_projection = "causal_transformer"
        turbine_embedding = True
        relation = "projected_pairwise_cosine_learned_mix"
        relation_mix = 0.20
        purpose = "16台风机联合面板的身份嵌入与动态关系功率预测"
    else:
        feature_fusion = "state_conditioned_pattern_experts"
        temporal_projection = "residual_temporal_cnn_plus_top3_moe"
        turbine_embedding = False
        relation = "none"
        relation_mix = None
        purpose = "16台风机联合面板的SATRA PSTR-Net功率预测"
    return (
        feature_fusion,
        temporal_projection,
        turbine_embedding,
        relation,
        relation_mix,
        purpose,
    )


def add_satra_config(args, model_config) -> None:
    # SATRA 结构开关只在 MultiTurbine 模型记录中出现。
    if args.model == "MultiTurbine":
        model_config["satra_hidden_dim"] = args.satra_hidden_dim
        model_config["satra_depth"] = args.satra_depth
        model_config["satra_kernel_size"] = args.satra_kernel_size
        model_config["satra_heads"] = args.satra_heads
        model_config["satra_tower_layers"] = args.satra_tower_layers
        model_config["satra_expert_top_k"] = args.satra_expert_top_k
        model_config["satra_use_spatial_relation"] = bool(args.satra_use_spatial_relation)
        if args.satra_use_spatial_relation:
            model_config["spatial_relation"] = "semantic_cross_turbine_transformer"
        else:
            model_config["spatial_relation"] = "disabled"
        use_dtw_prior = bool(args.satra_use_dtw_prior)
        if not args.satra_use_spatial_relation:
            use_dtw_prior = False
        model_config["satra_use_dtw_prior"] = use_dtw_prior
        if use_dtw_prior:
            model_config["turbine_embedding_relation"] = (
                "training_split_dtw_prior_tower"
            )
            model_config["satra_prior"] = "train_split_dtw"
            model_config["satra_prior_top_k"] = args.satra_prior_top_k
            model_config["satra_prior_candidate_k"] = args.satra_prior_candidate_k
            model_config["satra_prior_downsample"] = args.satra_prior_downsample
            model_config["satra_prior_band"] = args.satra_prior_band
        else:
            model_config["satra_prior"] = "disabled"


def write_power_result_contract(args, repository, datasets, checkpoint, best_epoch, best_mse, history, curve_overall, pretraining,):
    # 写入多风机联合面板的正式 report 与 record。
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
        "base_interval_seconds": BASE_INTERVAL_SECONDS,
        "point_interval_seconds": args.point_interval_seconds,
        "history_steps": args.history_steps,
        "horizon_steps": args.horizon_steps,
        "window_stride_steps": args.window_stride_steps,
        "loss": args.loss,
        "dbloss_weight": args.dbloss_weight,
    }
    metadata = select_model_metadata(args.model)
    feature_fusion = metadata[0]
    temporal_projection = metadata[1]
    turbine_embedding = metadata[2]
    relation = metadata[3]
    relation_mix = metadata[4]
    purpose = metadata[5]
    model_config = {
        "input_features": len(repository.feature_names),
        "feature_fusion": feature_fusion,
        "moving_average_kernel": None,
        "temporal_projection": temporal_projection,
        "turbine_embedding": turbine_embedding,
        "turbine_embedding_relation": relation,
        "embedding_relation_mix_initial": relation_mix,
        "revin": False,
    }
    add_satra_config(args, model_config)
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
    if args.pretrain:
        purpose = "遮蔽风机时间token重建热身后的SATRA PSTR-Net功率预测"
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
        "source": "dataset/processed/turbine_01.npy ... turbine_16.npy",
        "available_samples": available_samples,
        "split": "chronological_70_10_20",
        "split_seeds": "none",
        "train_samples": len(datasets["train"]),
        "valid_samples": len(datasets["val"]),
        "test_samples": len(datasets["test"]),
        "features": ", ".join(repository.feature_names),
        "feature_normalization": "per_turbine_train_standard_scaler",
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
    result["significance"] = build_dataset_significance(args.dataset_name, args.result_name, result,)
    report_path = result_report_path(args.dataset_name, args.result_name)
    record_path = result_record_path(args.dataset_name, args.result_name)
    write_result_report(report_path, result)
    write_result_record(record_path, result)
    return report_path, record_path
