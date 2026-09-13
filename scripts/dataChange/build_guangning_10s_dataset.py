from __future__ import annotations

import re
import shutil
import time
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import xlrd
from tqdm import tqdm


# 处理 9 月 12 日交付的广宁风电场原始压缩包。
ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_PATH = ROOT / "甲方提供新的内容" / "9月12日" / "guangningshuju.zip"
DATASET_ROOT = ROOT / "dataset"
BUILDING_ROOT = DATASET_ROOT / ".GuangningWindPower10s_building"
OUTPUT_ROOT = DATASET_ROOT / "GuangningWindPower10s"

# 广宁包包含 16 台风机，每个日文件是 8640 个原生 10 秒采样点。
TURBINE_COUNT = 16
SAMPLE_SECONDS = 10
SAMPLES_PER_DAY = 24 * 60 * 60 // SAMPLE_SECONDS
DEVICE_PATTERN = re.compile(r"广宁风电场(\d{2})号风机")

# 新交付中两个计算列对应原协议的理论功率和风速位置。
RAW_NUMERIC = [
    "风机-P",
    "风机-理论功率-计算",
    "风机-实时风速",
    "风机-风速-计算",
    "风机-环境温度",
    "机舱-舱内温度",
    "塔筒-塔底温度",
    "发电机-发电机转速",
    "传动链-主轴转速",
    "变桨轮毂-1#桨叶角度",
    "变桨轮毂-2#桨叶角度",
    "变桨轮毂-3#桨叶角度",
]
RAW_ANGLES = [
    "风机-实时风向",
    "偏航系统-对风角度",
    "偏航系统-机舱位置",
    "偏航系统-扭揽角度",
]


def collect_members(archive: ZipFile):
    # 按风机和日期排列日文件，使落盘数组天然按时间递增。
    members_by_turbine = []
    for turbine_index in range(TURBINE_COUNT):
        members_by_turbine.append([])

    for member_name in archive.namelist():
        if member_name.endswith(".xls"):
            turbine_match = DEVICE_PATTERN.search(member_name)
            turbine_index = int(turbine_match.group(1)) - 1
            members_by_turbine[turbine_index].append(member_name)

    for turbine_index in range(TURBINE_COUNT):
        members_by_turbine[turbine_index].sort()

    return members_by_turbine


def read_day(archive: ZipFile, member_name: str):
    # 读取首行表头和其后的 10 秒数据，直接构造 20 维训练特征。
    payload = archive.read(member_name)
    workbook = xlrd.open_workbook(file_contents=payload, on_demand=True)
    sheet = workbook.sheet_by_index(0)

    header = []
    for value in sheet.row_values(0):
        header.append(str(value).strip())

    device_index = header.index("设备名称")
    time_index = header.index("时间")
    numeric_indices = []
    for name in RAW_NUMERIC:
        numeric_indices.append(header.index(name))

    angle_indices = []
    for name in RAW_ANGLES:
        angle_indices.append(header.index(name))

    row_count = sheet.nrows - 1
    time_text = sheet.col_values(time_index, start_rowx=1)
    times = np.asarray(time_text, dtype="datetime64[ns]").astype(np.int64)
    numeric = np.empty((row_count, len(numeric_indices)), dtype=np.float64)
    angles = np.empty((row_count, len(angle_indices)), dtype=np.float64)

    for feature_index in range(len(numeric_indices)):
        source_index = numeric_indices[feature_index]
        numeric[:, feature_index] = np.asarray(
            sheet.col_values(source_index, start_rowx=1), dtype=np.float64
        )

    for feature_index in range(len(angle_indices)):
        source_index = angle_indices[feature_index]
        angles[:, feature_index] = np.asarray(
            sheet.col_values(source_index, start_rowx=1), dtype=np.float64
        )

    # 角度统一转为 sin/cos，位置与 config/settings.py 的特征协议一致。
    radians = np.deg2rad(angles)
    values = np.empty((row_count, 20), dtype=np.float32)
    values[:, 0] = numeric[:, 0]
    values[:, 1] = numeric[:, 1]
    values[:, 2] = numeric[:, 2]
    values[:, 3] = numeric[:, 3]
    values[:, 4] = np.sin(radians[:, 0])
    values[:, 5] = np.cos(radians[:, 0])
    values[:, 6] = numeric[:, 4]
    values[:, 7] = numeric[:, 5]
    values[:, 8] = numeric[:, 6]
    values[:, 9] = numeric[:, 7]
    values[:, 10] = numeric[:, 8]
    values[:, 11] = numeric[:, 9]
    values[:, 12] = numeric[:, 10]
    values[:, 13] = numeric[:, 11]
    values[:, 14] = np.sin(radians[:, 1])
    values[:, 15] = np.cos(radians[:, 1])
    values[:, 16] = np.sin(radians[:, 2])
    values[:, 17] = np.cos(radians[:, 2])
    values[:, 18] = np.sin(radians[:, 3])
    values[:, 19] = np.cos(radians[:, 3])

    device = str(sheet.cell_value(1, device_index))
    workbook.release_resources()
    return device, times, values


def create_arrays(members_by_turbine):
    # 用 NPY memmap 逐日写入，避免把 132 天的数据堆积在内存。
    array_root = BUILDING_ROOT / "arrays"
    array_root.mkdir(parents=True)
    time_arrays = []
    value_arrays = []

    for turbine_index in range(TURBINE_COUNT):
        day_count = len(members_by_turbine[turbine_index])
        row_count = day_count * SAMPLES_PER_DAY
        turbine_id = turbine_index + 1
        time_path = array_root / f"turbine_{turbine_id:02d}_times.npy"
        value_path = array_root / f"turbine_{turbine_id:02d}_values.npy"
        time_array = np.lib.format.open_memmap(
            time_path, mode="w+", dtype=np.int64, shape=(row_count,)
        )
        value_array = np.lib.format.open_memmap(
            value_path, mode="w+", dtype=np.float32, shape=(row_count, 20)
        )
        time_arrays.append(time_array)
        value_arrays.append(value_array)

    return time_arrays, value_arrays


def write_arrays(archive: ZipFile, members_by_turbine, time_arrays, value_arrays):
    # 逐文件解析并写入所属风机的预分配数组。
    file_count = 0
    for turbine_index in range(TURBINE_COUNT):
        file_count += len(members_by_turbine[turbine_index])

    offsets = np.zeros(TURBINE_COUNT, dtype=np.int64)
    progress = tqdm(total=file_count, desc="处理广宁 10 秒数据", unit="file")
    for expected_turbine_index in range(TURBINE_COUNT):
        for member_name in members_by_turbine[expected_turbine_index]:
            device, times, values = read_day(archive, member_name)
            turbine_match = DEVICE_PATTERN.fullmatch(device)
            turbine_index = int(turbine_match.group(1)) - 1
            begin = int(offsets[turbine_index])
            end = begin + len(times)
            time_arrays[turbine_index][begin:end] = times
            value_arrays[turbine_index][begin:end] = values
            offsets[turbine_index] = end
            progress.update(1)

    progress.close()
    for turbine_index in range(TURBINE_COUNT):
        time_arrays[turbine_index].flush()
        value_arrays[turbine_index].flush()


def inspect_arrays():
    # 重新打开已写入数组，报告每台风机的时间范围和断点数。
    expected_delta = SAMPLE_SECONDS * 1_000_000_000
    for turbine_index in range(TURBINE_COUNT):
        turbine_id = turbine_index + 1
        time_path = BUILDING_ROOT / "arrays" / f"turbine_{turbine_id:02d}_times.npy"
        value_path = BUILDING_ROOT / "arrays" / f"turbine_{turbine_id:02d}_values.npy"
        times = np.load(time_path, mmap_mode="r")
        values = np.load(value_path, mmap_mode="r")
        break_count = int(np.count_nonzero(np.diff(times) != expected_delta))
        start_time = np.datetime64(int(times[0]), "ns")
        end_time = np.datetime64(int(times[-1]), "ns")
        print(
            f"风机{turbine_id:02d}: rows={len(times)}, shape={values.shape}, "
            f"start={start_time}, end={end_time}, breaks={break_count}",
            flush=True,
        )


def replace_processed_dataset():
    # 完整临时数据通过重新读取后，替换旧的 10 秒处理结果。
    shutil.rmtree(OUTPUT_ROOT)
    BUILDING_ROOT.rename(OUTPUT_ROOT)


def main():
    # 建立临时输出，完整处理后才删除旧数据。
    started = time.time()
    BUILDING_ROOT.mkdir()

    # 读取全部日文件并落盘为 16 组原生 10 秒序列。
    with ZipFile(ARCHIVE_PATH) as archive:
        members_by_turbine = collect_members(archive)
        time_arrays, value_arrays = create_arrays(members_by_turbine)
        write_arrays(archive, members_by_turbine, time_arrays, value_arrays)

    # 重开数组确认输出可读取，再将临时目录提升为正式数据集。
    inspect_arrays()
    replace_processed_dataset()
    elapsed_seconds = time.time() - started
    print(f"完成: {OUTPUT_ROOT}，耗时 {elapsed_seconds:.1f} 秒", flush=True)


if __name__ == "__main__":
    main()
