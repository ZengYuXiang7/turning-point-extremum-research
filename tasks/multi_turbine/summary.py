import numpy as np

import config as project_config


def format_ns_date(timestamp_ns: int) -> str:
    date = np.datetime_as_string(np.datetime64(timestamp_ns, "ns"), unit="D")
    return f"{date[:4]}年{date[5:7]}月{date[8:10]}日"


def print_data_summary(datasets, loaders, horizon_steps, window_stride_steps):
    # 多风机摘要只展示联合面板输入协议。
    feature_names = datasets["train"].repository.feature_names
    print("\n========== 多风机联合面板 Dataset 与 Batch ==========")
    print(f"past 特征（{len(feature_names)} 个）：")
    for feature_index, feature_name in enumerate(feature_names, start=1):
        print(f"  {feature_index:02d}. {feature_name}")
    print("future_weather：不构造。")
    print("target 特征（1 个）：风机-P")
    print("turbine_id：每个样本固定包含16个风机索引。")

    history_minutes = (
        project_config.HISTORY_STEPS * project_config.POINT_INTERVAL_SECONDS / 60
    )
    horizon_minutes = horizon_steps * project_config.POINT_INTERVAL_SECONDS / 60
    stride_minutes = window_stride_steps * project_config.POINT_INTERVAL_SECONDS / 60
    point_minutes = project_config.POINT_INTERVAL_SECONDS / 60
    print(
        f"历史长度={history_minutes:g}分钟 预测长度={horizon_minutes:g}分钟 "
        f"滑窗步长={stride_minutes:g}分钟\n"
    )

    total_windows = 0
    for split in ("train", "val", "test"):
        total_windows += len(datasets[split])
    target_span_ns = (
        (horizon_steps - 1)
        * project_config.POINT_INTERVAL_SECONDS
        * 1_000_000_000
    )
    total_coverage_ns = 0
    for split in ("train", "val", "test"):
        dataset = datasets[split]
        first_start = int(dataset[0][-1].item())
        last_end = int(dataset[-1][-1].item()) + target_span_ns
        total_coverage_ns += last_end - first_start

    for split in ("train", "val", "test"):
        dataset = datasets[split]
        loader = loaders[split]
        target_start_ns = int(dataset[0][-1].item())
        target_end_ns = int(dataset[-1][-1].item()) + target_span_ns
        time_ratio = 100.0 * (target_end_ns - target_start_ns) / total_coverage_ns
        window_ratio = 100.0 * len(dataset) / total_windows
        batch_items = []
        for sample_index in range(loader.batch_size):
            batch_items.append(dataset[sample_index])
        past, target, turbine_id, _ = loader.collate_fn(batch_items)
        print(
            f"  {split:<5}  目标时间={format_ns_date(target_start_ns)} 至 "
            f"{format_ns_date(target_end_ns)}  时间占比={time_ratio:.2f}%  "
            f"窗口={len(dataset):,}（{window_ratio:.2f}%）  batches={len(loader):,}"
        )
        print(
            f"    输入 past shape={list(past.shape)}  "
            f"turbine_id shape={list(turbine_id.shape)}"
        )
        print(
            f"    标签 target shape={list(target.shape)}  "
            f"时间维={horizon_steps}点（每点{point_minutes:g}分钟）",
            flush=True,
        )
