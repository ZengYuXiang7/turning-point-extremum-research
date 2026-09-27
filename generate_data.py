from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import re
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import xlrd
from tqdm import tqdm

from config import BASE_INTERVAL_SECONDS


ROOT = Path(__file__).resolve().parent
DATASET_ROOT = ROOT / "dataset"
RAW_ROOT = DATASET_ROOT / "raw"
OUTPUT_ROOT = DATASET_ROOT / "processed"
TURBINE_COUNT = 16
WORKER_COUNT = 8
DEVICE_PATTERN = re.compile(r"广宁风电场(\d{2})号风机")
IDENTITY_COLUMNS = ("场站", "设备名称", "时间")
LABEL_COLUMN = "风机-P"
REQUIRED_SOURCE_COLUMNS = (
    "风机-理论功率-计算",
    "风机-实时风速",
    "风机-实时风向",
    "风机-环境温度",
    "机舱-舱内温度",
    "塔筒-塔底温度",
    "发电机-发电机转速",
    "传动链-主轴转速",
    "变桨轮毂-1#桨叶角度",
    "变桨轮毂-2#桨叶角度",
    "变桨轮毂-3#桨叶角度",
    "偏航系统-对风角度",
    "偏航系统-机舱位置",
    "偏航系统-扭揽角度",
    LABEL_COLUMN,
)


def collect_member_paths(raw_root):
    # 按风机和文件名排序，给 raw 中每个日文件分配固定槽位。
    if not raw_root.is_dir():
        raise FileNotFoundError(f"原始数据目录不存在: {raw_root}")
    member_paths = sorted(raw_root.rglob("*.xls"))
    if not member_paths:
        raise FileNotFoundError(f"原始数据目录中没有 .xls 文件: {raw_root}")
    return member_paths


def collect_headers(member_path):
    # 从首个日文件读取统一的原始列顺序。
    workbook = xlrd.open_workbook(member_path, on_demand=True)
    sheet = workbook.sheet_by_index(0)
    headers = []
    for value in sheet.row_values(0):
        headers.append(str(value).strip())
    workbook.release_resources()
    return headers


def select_feature_columns(headers):
    # 保持输入列相对顺序，标签列固定放在数值矩阵末尾。
    source_indices = []
    feature_names = []
    label_index = headers.index(LABEL_COLUMN)

    for column_index in range(len(headers)):
        header = headers[column_index]
        if header in IDENTITY_COLUMNS:
            continue
        if column_index == label_index:
            continue
        source_indices.append(column_index)
        feature_names.append(header)

    source_indices.append(label_index)
    feature_names.append(LABEL_COLUMN)

    return source_indices, feature_names


def convert_numeric_column(column_values):
    # 原始空单元格映射为 NaN，扫描阶段据此删除整列。
    values = np.asarray(column_values, dtype=object)
    empty_mask = values == ""
    values[empty_mask] = np.nan
    numeric_values = values.astype(np.float64)
    return numeric_values


def scan_member(member_index, member_path, source_indices, expected_headers):
    # 并行读取单个日文件的风机编号、原始行数和缺失列。
    workbook = xlrd.open_workbook(member_path, on_demand=True)
    sheet = workbook.sheet_by_index(0)
    headers = tuple(str(value).strip() for value in sheet.row_values(0))
    if headers != expected_headers:
        workbook.release_resources()
        raise ValueError(f"{member_path} 的列名或列顺序与首个文件不一致")
    turbine_match = DEVICE_PATTERN.search(member_path.name)
    if turbine_match is None:
        workbook.release_resources()
        raise ValueError(f"无法从文件名识别风机编号: {member_path.name}")
    turbine_index = int(turbine_match.group(1)) - 1
    if not 0 <= turbine_index < TURBINE_COUNT:
        workbook.release_resources()
        raise ValueError(f"风机编号必须为 01-{TURBINE_COUNT:02d}: {member_path.name}")
    raw_row_count = sheet.nrows - 1
    column_count = sheet.ncols
    missing_columns = np.zeros(len(source_indices), dtype=bool)

    for feature_index in range(len(source_indices)):
        source_index = source_indices[feature_index]
        column_values = sheet.col_values(source_index, start_rowx=1)
        numeric_values = convert_numeric_column(column_values)
        missing_columns[feature_index] = np.isnan(numeric_values).any()

    workbook.release_resources()
    return (
        member_index,
        turbine_index,
        raw_row_count,
        column_count,
        missing_columns,
    )


def build_slots(member_paths, source_indices, expected_headers, progress, worker_count):
    # 并行统计每个日文件的原始行数，再分配不重叠槽位。
    member_count = len(member_paths)
    turbine_indices = np.empty(member_count, dtype=np.int64)
    row_counts = np.empty(member_count, dtype=np.int64)
    missing_columns = np.zeros(len(source_indices), dtype=bool)

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = []
        for member_index in range(member_count):
            member_path = member_paths[member_index]
            future = executor.submit(
                scan_member, member_index, member_path, source_indices, expected_headers
            )
            futures.append(future)

        for future in as_completed(futures):
            (member_index, turbine_index, raw_row_count, column_count, member_missing,) = future.result()
            turbine_indices[member_index] = turbine_index
            row_counts[member_index] = raw_row_count
            missing_columns |= member_missing
            member_path = member_paths[member_index]
            display_member_path = member_path
            print(f"[{member_index + 1}/{member_count}] {display_member_path}: " f"raw_shape=({raw_row_count}, {column_count}), " f"raw_points={raw_row_count}", flush=True,)
            progress.update(1)

    slot_starts = np.empty(member_count, dtype=np.int64)
    sequence_lengths = np.zeros(TURBINE_COUNT, dtype=np.int64)
    for member_index in range(member_count):
        turbine_index = turbine_indices[member_index]
        slot_starts[member_index] = sequence_lengths[turbine_index]
        sequence_lengths[turbine_index] += row_counts[member_index]

    missing_turbines = np.flatnonzero(sequence_lengths == 0) + 1
    if len(missing_turbines):
        missing_text = ", ".join(f"{value:02d}" for value in missing_turbines)
        raise ValueError(f"缺少风机数据: {missing_text}")

    return turbine_indices, slot_starts, sequence_lengths, missing_columns


def create_output_arrays(building_root, sequence_lengths, feature_dim):
    # 每台风机只生成一个 [seq_len, 1 + d] 的 NPY 文件。
    data_paths = []
    output_dim = feature_dim + 1

    for turbine_index in range(TURBINE_COUNT):
        turbine_id = turbine_index + 1
        data_path = building_root / f"turbine_{turbine_id:02d}.npy"
        sequence_length = int(sequence_lengths[turbine_index])
        data_array = np.lib.format.open_memmap(data_path, mode="w+", dtype=np.float64, shape=(sequence_length, output_dim),)
        data_array.flush()
        data_paths.append(data_path)

    return data_paths


def write_member(member_index, member_path, source_indices, time_index, slot_start, data_path,):
    # 并行解码一个日文件，并写入原始10秒时刻的目标槽位。
    workbook = xlrd.open_workbook(member_path, on_demand=True)
    sheet = workbook.sheet_by_index(0)
    raw_row_count = sheet.nrows - 1
    times = np.asarray(sheet.col_values(time_index, start_rowx=1), dtype="datetime64[s]").astype(np.int64)
    values = np.empty((raw_row_count, len(source_indices) + 1), dtype=np.float64)
    values[:, 0] = times

    for output_index in range(len(source_indices)):
        source_index = source_indices[output_index]
        column_values = sheet.col_values(source_index, start_rowx=1)
        values[:, output_index + 1] = convert_numeric_column(column_values)

    order = np.argsort(values[:, 0], kind="stable")
    values = values[order]
    begin = int(slot_start)
    end = begin + len(values)
    data_array = np.load(data_path, mmap_mode="r+")
    data_array[begin:end] = values
    data_array.flush()
    workbook.release_resources()
    return member_index


def write_members(member_paths, source_indices, time_index, turbine_indices, slot_starts, data_paths, progress, worker_count,):
    # 多进程写入不同槽位，任务完成先后不影响数组中的时间顺序。
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = []
        for member_index in range(len(member_paths)):
            member_path = member_paths[member_index]
            turbine_index = turbine_indices[member_index]
            data_path = data_paths[turbine_index]
            future = executor.submit(write_member, member_index, member_path, source_indices, time_index, slot_starts[member_index], data_path,)
            futures.append(future)

        for future in as_completed(futures):
            future.result()
            progress.update(1)


def sort_turbine_arrays(data_paths):
    # 汇总全部日文件后，按首列时间戳稳定排序每台风机序列。
    for turbine_index in range(TURBINE_COUNT):
        data_path = data_paths[turbine_index]
        data_array = np.load(data_path, mmap_mode="r+")
        order = np.argsort(data_array[:, 0], kind="stable")
        sorted_data = data_array[order].copy()
        data_array[:] = sorted_data
        data_array.flush()


def create_building_root(output_root):
    # 为本次构建创建独占的临时输出目录，避免并行运行互相删除文件。
    output_root.parent.mkdir(parents=True, exist_ok=True)
    building_root = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}_build_", dir=output_root.parent)
    )
    return building_root


def publish_output(building_root, output_root):
    # 全部 NPY 完成后再整体替换旧 processed 目录。
    backup_root = output_root.with_name(f".{output_root.name}_previous")
    if backup_root.exists():
        shutil.rmtree(backup_root)
    if output_root.exists():
        output_root.rename(backup_root)
    try:
        building_root.rename(output_root)
    except Exception:
        if backup_root.exists() and not output_root.exists():
            backup_root.rename(output_root)
        raise
    if backup_root.exists():
        shutil.rmtree(backup_root)


def save_feature_names(building_root, feature_names):
    # 列名与二维数组同序，最后一项固定为目标。
    columns_path = building_root / "columns.json"
    columns_path.write_text(json.dumps(feature_names, ensure_ascii=False, indent=2), encoding="utf-8")


def validate_required_columns(feature_names):
    missing = [name for name in REQUIRED_SOURCE_COLUMNS if name not in feature_names]
    if missing:
        raise ValueError(f"模型所需列缺失或含空值: {missing}")


def validate_output(data_paths):
    # 联合面板需要16台风机拥有完全相同的时间轴。
    reference = np.load(data_paths[0], mmap_mode="r")[:, 0]
    if len(reference) == 0 or np.any(np.diff(reference) <= 0):
        raise ValueError("风机01时间轴为空、重复或非递增")
    for turbine_index in range(1, TURBINE_COUNT):
        candidate = np.load(data_paths[turbine_index], mmap_mode="r")[:, 0]
        if not np.array_equal(candidate, reference):
            raise ValueError(
                f"风机{turbine_index + 1:02d}与风机01的时间轴不一致"
            )


def build_dataset(raw_root, output_root, worker_count, building_root):
    member_paths = collect_member_paths(raw_root)
    headers = collect_headers(member_paths[0])
    source_indices, feature_names = select_feature_columns(headers)
    time_index = headers.index("时间")

    progress = tqdm(total=2 * len(member_paths), desc=f"构建{BASE_INTERVAL_SECONDS}秒 NPY", unit="file",)
    turbine_indices, slot_starts, sequence_lengths, missing_columns = (
        build_slots(member_paths, source_indices, tuple(headers), progress, worker_count)
    )
    source_indices = np.asarray(source_indices, dtype=np.int64)
    feature_names = np.asarray(feature_names)
    source_indices = source_indices[~missing_columns].tolist()
    feature_names = feature_names[~missing_columns].tolist()
    validate_required_columns(feature_names)

    print("[0] Unix 秒级时间戳")
    for feature_index in range(len(feature_names)):
        print(f"[{feature_index + 1}] {feature_names[feature_index]}")

    data_paths = create_output_arrays(building_root, sequence_lengths, len(feature_names))
    save_feature_names(building_root, feature_names)
    write_members(member_paths, source_indices, time_index, turbine_indices, slot_starts, data_paths, progress, worker_count,)
    progress.close()

    sort_turbine_arrays(data_paths)
    validate_output(data_paths)
    publish_output(building_root, output_root)


def main(raw_root=RAW_ROOT, output_root=OUTPUT_ROOT, worker_count=WORKER_COUNT):
    # 扫描原始记录，并行写入每台风机的10秒底层序列。
    started = time.time()
    raw_root = Path(raw_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if worker_count < 1:
        raise ValueError("worker_count 必须大于等于1")
    building_root = create_building_root(output_root)
    try:
        build_dataset(raw_root, output_root, worker_count, building_root)
    finally:
        if building_root.exists():
            shutil.rmtree(building_root)

    elapsed_seconds = time.time() - started
    columns = json.loads((output_root / "columns.json").read_text(encoding="utf-8"))
    print(f"完成: {output_root}，风机数={TURBINE_COUNT}，" f"底层间隔={BASE_INTERVAL_SECONDS}秒，矩阵维度={len(columns) + 1}，" f"耗时 {elapsed_seconds:.1f} 秒", flush=True,)


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(
        description="从原始 XLS 构建固定10秒间隔的 processed NPY"
    )
    argument_parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    argument_parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    argument_parser.add_argument("--workers", type=int, default=WORKER_COUNT)
    arguments = argument_parser.parse_args()

    main(arguments.raw_root, arguments.output_root, arguments.workers)
