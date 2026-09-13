from __future__ import annotations

# 正式数据入口：原生 10s 采样，已废弃分钟平均
import json
import re
import time
from pathlib import Path

import numpy as np
import xlrd

ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = ROOT / "汕头电力数据"
OUTPUT = ROOT / "dataset" / "GuangningWindPower10s"

# 数值列：功率 / 风速气温 / 舱温塔底温 / 机组状态
RAW_NUMERIC = [
    "风机-P",
    "风机-理论功率",
    "风机-实时风速",
    "风机-风速",
    "风机-环境温度",
    "机舱-舱内温度",
    "塔筒-塔底温度",
    "发电机-发电机转速",
    "传动链-主轴转速",
    "变桨轮毂-1#桨叶角度",
    "变桨轮毂-2#桨叶角度",
    "变桨轮毂-3#桨叶角度",
]
# 角度转 sin/cos；扭缆角纳入风向相关输入
RAW_ANGLES = [
    "风机-实时风向",
    "偏航系统-对风角度",
    "偏航系统-机舱位置",
    "偏航系统-扭揽角度",
]
DEVICE_PATTERN = re.compile(r"广宁风电场(\d{2})号风机")


def parse_time_ns(text, cache):
    # 同一时间字符串只解析一次
    if text in cache:
        return cache[text]
    value = int(np.datetime64(text, "ns").astype(np.int64))
    cache[text] = value
    return value


def feature_row(row, numeric_idx, angle_idx):
    # 数值按 RAW_NUMERIC；角度按 RAW_ANGLES 转 sin/cos
    numeric_values = []
    for index in numeric_idx:
        numeric_values.append(float(row[index]))
    numeric = np.asarray(numeric_values, dtype=np.float64)

    angle_values = []
    for index in angle_idx:
        angle_values.append(float(row[index]))
    radians = np.deg2rad(np.asarray(angle_values, dtype=np.float64))
    sine = np.sin(radians)
    cosine = np.cos(radians)

    # 顺序对齐 settings.HISTORY_COLUMNS
    vector = np.asarray(
        [
            numeric[0],
            numeric[1],
            numeric[2],
            numeric[3],
            sine[0],
            cosine[0],
            numeric[4],
            numeric[5],
            numeric[6],
            numeric[7],
            numeric[8],
            numeric[9],
            numeric[10],
            numeric[11],
            sine[1],
            cosine[1],
            sine[2],
            cosine[2],
            sine[3],
            cosine[3],
        ],
        dtype=np.float32,
    )
    return vector


def pack_turbine(turbine_id, all_turbine_ids, all_times, all_values):
    # 按时间排序；同秒重复时保留后写入样本
    mask = all_turbine_ids == turbine_id
    times = all_times[mask]
    values = all_values[mask]
    order = np.argsort(times, kind="stable")
    times = times[order]
    values = values[order]
    unique_times, first, counts = np.unique(times, return_index=True, return_counts=True)
    selected = np.empty((len(unique_times), values.shape[1]), dtype=np.float32)
    for index in range(len(unique_times)):
        begin = first[index]
        count = counts[index]
        selected[index] = values[begin + count - 1]
    return unique_times.astype(np.int64), selected


def read_raw_series():
    # 逐文件读取原始 10s 行，不做分钟聚合
    workbook_paths = sorted(RAW_ROOT.rglob("*.xls"))
    turbine_ids = []
    time_values = []
    feature_rows = []
    timestamp_cache = {}
    started = time.time()

    for file_index, workbook_path in enumerate(workbook_paths, start=1):
        print(
            f"[{file_index}/{len(workbook_paths)}] {workbook_path.relative_to(ROOT)}",
            flush=True,
        )
        book = xlrd.open_workbook(str(workbook_path), on_demand=True)
        sheet = book.sheet_by_index(0)

        header = []
        for value in sheet.row_values(1):
            header.append(str(value).strip())

        device_idx = header.index("设备名称")
        time_idx = header.index("时间")
        numeric_idx = []
        for name in RAW_NUMERIC:
            numeric_idx.append(header.index(name))
        angle_idx = []
        for name in RAW_ANGLES:
            angle_idx.append(header.index(name))

        for row in sheet._cell_values[2:]:
            device = str(row[device_idx])
            turbine = int(DEVICE_PATTERN.fullmatch(device).group(1))
            time_ns = parse_time_ns(str(row[time_idx]), timestamp_cache)
            turbine_ids.append(turbine)
            time_values.append(time_ns)
            feature_rows.append(feature_row(row, numeric_idx, angle_idx))

        book.unload_sheet(0)
        book.release_resources()

    all_turbine_ids = np.asarray(turbine_ids, dtype=np.int32)
    all_times = np.asarray(time_values, dtype=np.int64)
    all_values = np.asarray(feature_rows, dtype=np.float32)

    # 按风机打包连续 10s 序列
    series = []
    for turbine_id in range(1, 17):
        times, values = pack_turbine(
            turbine_id, all_turbine_ids, all_times, all_values
        )
        series.append((times, values))

    elapsed = time.time() - started
    return series, elapsed, len(workbook_paths)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "arrays").mkdir(exist_ok=True)

    # 构建并落盘
    series, elapsed, file_count = read_raw_series()
    for turbine, (times, values) in enumerate(series, start=1):
        np.save(OUTPUT / "arrays" / f"turbine_{turbine:02d}_times.npy", times)
        np.save(OUTPUT / "arrays" / f"turbine_{turbine:02d}_values.npy", values)

    print(
        json.dumps(
            {
                "event": "dataset_complete",
                "root": str(OUTPUT),
                "files": file_count,
                "feature_dim": int(series[0][1].shape[1]),
                "elapsed_seconds": round(elapsed, 1),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
