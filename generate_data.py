from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import re
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import xlrd
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent
DATASET_ROOT = ROOT / "dataset"
RAW_ROOT = DATASET_ROOT / "raw"
OUTPUT_ROOT = DATASET_ROOT / "processed"
TURBINE_COUNT = 16
WORKER_COUNT = 8
DEVICE_PATTERN = re.compile(r"广宁风电场(\d{2})号风机")
IDENTITY_COLUMNS = ("场站", "设备名称", "时间")


def collect_member_paths():
    # 按风机和文件名排序，给 raw 中每个日文件分配固定槽位。
    member_paths = []
    for member_path in RAW_ROOT.rglob("*.xls"):
        member_paths.append(member_path)
    member_paths.sort()
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
    # 保留全部原始数值列，身份列不进入数值矩阵。
    source_indices = []
    feature_names = []

    for column_index in range(len(headers)):
        header = headers[column_index]
        if header in IDENTITY_COLUMNS:
            continue
        source_indices.append(column_index)
        feature_names.append(header)

    return source_indices, feature_names


def scan_member(member_index, member_path):
    # 并行读取单个日文件的风机编号和行数。
    workbook = xlrd.open_workbook(member_path, on_demand=True)
    sheet = workbook.sheet_by_index(0)
    turbine_match = DEVICE_PATTERN.search(member_path.name)
    turbine_index = int(turbine_match.group(1)) - 1
    row_count = sheet.nrows - 1
    column_count = sheet.ncols
    workbook.release_resources()
    return member_index, turbine_index, row_count, column_count


def build_slots(member_paths, progress):
    # 并行统计每个日文件长度，再在各风机数组中分配不重叠槽位。
    member_count = len(member_paths)
    turbine_indices = np.empty(member_count, dtype=np.int64)
    row_counts = np.empty(member_count, dtype=np.int64)

    with ProcessPoolExecutor(max_workers=WORKER_COUNT) as executor:
        futures = []
        for member_index in range(member_count):
            member_path = member_paths[member_index]
            future = executor.submit(scan_member, member_index, member_path)
            futures.append(future)

        for future in as_completed(futures):
            member_index, turbine_index, row_count, column_count = future.result()
            turbine_indices[member_index] = turbine_index
            row_counts[member_index] = row_count
            member_path = member_paths[member_index]
            print(
                f"[{member_index + 1}/{member_count}] {member_path}: "
                f"shape=({row_count}, {column_count})",
                flush=True,
            )
            progress.update(1)

    slot_starts = np.empty(member_count, dtype=np.int64)
    sequence_lengths = np.zeros(TURBINE_COUNT, dtype=np.int64)
    for member_index in range(member_count):
        turbine_index = turbine_indices[member_index]
        slot_starts[member_index] = sequence_lengths[turbine_index]
        sequence_lengths[turbine_index] += row_counts[member_index]

    return turbine_indices, row_counts, slot_starts, sequence_lengths


def create_output_arrays(building_root, sequence_lengths, feature_dim):
    # 每台风机只生成一个 [seq_len, 1 + d] 的 NPY 文件。
    data_paths = []
    output_dim = feature_dim + 1

    for turbine_index in range(TURBINE_COUNT):
        turbine_id = turbine_index + 1
        data_path = building_root / f"turbine_{turbine_id:02d}.npy"
        sequence_length = int(sequence_lengths[turbine_index])
        data_array = np.lib.format.open_memmap(
            data_path,
            mode="w+",
            dtype=np.float64,
            shape=(sequence_length, output_dim),
        )
        data_array.flush()
        data_paths.append(data_path)

    return data_paths


def convert_numeric_column(column_values):
    # 原始空单元格映射为 NaN，保留该特征列及其缺失位置。
    values = np.asarray(column_values, dtype=object)
    empty_mask = values == ""
    values[empty_mask] = np.nan
    numeric_values = values.astype(np.float64)
    return numeric_values


def write_member(
    member_index,
    member_path,
    source_indices,
    time_index,
    slot_start,
    data_path,
):
    # 并行解码一个日文件，并仅写入该文件预分配的目标槽位。
    workbook = xlrd.open_workbook(member_path, on_demand=True)
    sheet = workbook.sheet_by_index(0)
    row_count = sheet.nrows - 1
    times = np.asarray(
        sheet.col_values(time_index, start_rowx=1), dtype="datetime64[s]"
    ).astype(np.int64)
    values = np.empty((row_count, len(source_indices) + 1), dtype=np.float64)
    values[:, 0] = times

    for output_index in range(len(source_indices)):
        source_index = source_indices[output_index]
        column_values = sheet.col_values(source_index, start_rowx=1)
        values[:, output_index + 1] = convert_numeric_column(column_values)

    order = np.argsort(values[:, 0], kind="stable")
    begin = int(slot_start)
    end = begin + row_count
    data_array = np.load(data_path, mmap_mode="r+")
    data_array[begin:end] = values[order]
    data_array.flush()
    workbook.release_resources()
    return member_index


def write_members(
    member_paths,
    source_indices,
    time_index,
    turbine_indices,
    slot_starts,
    data_paths,
    progress,
):
    # 多进程写入不同槽位，任务完成先后不影响数组中的时间顺序。
    with ProcessPoolExecutor(max_workers=WORKER_COUNT) as executor:
        futures = []
        for member_index in range(len(member_paths)):
            member_path = member_paths[member_index]
            turbine_index = turbine_indices[member_index]
            data_path = data_paths[turbine_index]
            future = executor.submit(
                write_member,
                member_index,
                member_path,
                source_indices,
                time_index,
                slot_starts[member_index],
                data_path,
            )
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


def create_building_root():
    # 为本次构建创建独占的临时输出目录，避免并行运行互相删除文件。
    building_root = Path(tempfile.mkdtemp(prefix=".processed_build_", dir=DATASET_ROOT))
    return building_root


def publish_output(building_root):
    # 全部 NPY 完成后再整体替换旧 processed 目录。
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    building_root.rename(OUTPUT_ROOT)


def main():
    # 扫描、并行写入并发布每台风机的完整原始数值序列。
    started = time.time()
    building_root = create_building_root()
    member_paths = collect_member_paths()
    headers = collect_headers(member_paths[0])
    source_indices, feature_names = select_feature_columns(headers)
    time_index = headers.index("时间")

    print("[0] Unix 秒级时间戳")
    for feature_index in range(len(feature_names)):
        print(f"[{feature_index + 1}] {feature_names[feature_index]}")

    progress = tqdm(
        total=2 * len(member_paths), desc="构建原始 NPY", unit="file"
    )
    turbine_indices, row_counts, slot_starts, sequence_lengths = build_slots(
        member_paths, progress
    )
    data_paths = create_output_arrays(
        building_root, sequence_lengths, len(feature_names)
    )
    write_members(
        member_paths,
        source_indices,
        time_index,
        turbine_indices,
        slot_starts,
        data_paths,
        progress,
    )
    progress.close()

    sort_turbine_arrays(data_paths)
    publish_output(building_root)
    elapsed_seconds = time.time() - started
    print(
        f"完成: {OUTPUT_ROOT}，风机数={TURBINE_COUNT}，"
        f"矩阵维度={len(feature_names) + 1}，耗时 {elapsed_seconds:.1f} 秒",
        flush=True,
    )


if __name__ == "__main__":
    main()
