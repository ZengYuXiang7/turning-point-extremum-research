import numpy as np

import config as project_config
from config import WEATHER_COLUMNS

def format_ns_date(timestamp_ns: int) -> str:
    date = np.datetime_as_string(np.datetime64(timestamp_ns, "ns"), unit="D")
    return f"{date[:4]}年{date[5:7]}月{date[8:10]}日"


def print_data_summary(datasets, loaders, horizon_steps, window_stride_steps, no_future_weather, weather_task, oracle, predicted_weather, provided_weather,):
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
    elif provided_weather:
        weather_source = "甲方提供的未来预测天气"
        target_features = [datasets["train"].repository.target_name]

    future_features = datasets["train"].repository.future_feature_names

    print("\n========== 数据集与 Batch 输入 ==========")
    print(f"past 特征（{len(past_features)} 个）：")
    for feature_index, feature_name in enumerate(past_features, start=1):
        print(f"  {feature_index:02d}. {feature_name}")

    if no_future_weather:
        print("NoFutureWeather：不构造未来天气输入。")
    else:
        print(f"future_weather 特征（{len(future_features)} 个，{weather_source}）：")
        for feature_index, feature_name in enumerate(future_features, start=1):
            print(f"  {feature_index:02d}. {feature_name}")

    print(f"target 特征（{len(target_features)} 个）：{', '.join(target_features)}")
    if provided_weather:
        print("site_id：场站标量索引，当前广宁风电场固定为0。\n")
    else:
        print("turbine_id：从 0 开始的单个风机标量索引。\n")

    if provided_weather:
        repository = datasets["train"].repository
        excluded_windows = 0
        for dataset in datasets.values():
            excluded_windows += dataset.excluded_windows
        print(f"甲方预测风速缺失点={repository.missing_forecast_points}，" f"因此排除窗口={excluded_windows}。\n")

    history_minutes = (
        project_config.HISTORY_STEPS * project_config.POINT_INTERVAL_SECONDS / 60
    )
    horizon_minutes = horizon_steps * project_config.POINT_INTERVAL_SECONDS / 60
    window_stride_minutes = (
        window_stride_steps * project_config.POINT_INTERVAL_SECONDS / 60
    )
    point_minutes = project_config.POINT_INTERVAL_SECONDS / 60
    print(f"历史长度={history_minutes:g}分钟 预测长度={horizon_minutes:g}分钟 " f"滑窗步长={window_stride_minutes:g}分钟\n")
    print("各切分的预测目标覆盖范围、样本占比和 Batch 张量形状：")

    total_windows = sum(len(datasets[split]) for split in ("train", "val", "test"))
    target_span_ns = (
        (horizon_steps - 1)
        * project_config.POINT_INTERVAL_SECONDS
        * 1_000_000_000
    )
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
        print(f"  {split:<5}  目标时间={format_ns_date(target_start_ns)} 至 " f"{format_ns_date(target_end_ns)}（约 {coverage_days:.1f} 天）  " f"时间占比={time_ratio:.2f}%  窗口={len(dataset):,}（{window_ratio:.2f}%）  " f"batches={len(loader):,}")

        if no_future_weather:
            past, target, turbine_id, target_start = batch
            print(f"    输入  past shape={list(past.shape)}  " f"turbine_id shape={list(turbine_id.shape)}")
        else:
            past, future_weather, target, turbine_id, target_start = batch
            entity_id_name = "turbine_id"
            if provided_weather:
                entity_id_name = "site_id"
            print(f"    输入  past shape={list(past.shape)}  " f"future_weather shape={list(future_weather.shape)}  " f"{entity_id_name} shape={list(turbine_id.shape)}")

        print(f"    标签  target shape={list(target.shape)}  " f"时间维={horizon_steps}点（每点{point_minutes:g}分钟）", flush=True,)
        print('')
