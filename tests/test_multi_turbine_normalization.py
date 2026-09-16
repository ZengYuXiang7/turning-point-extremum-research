import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_provider.common import Repository


SOURCE_COLUMNS = (
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
    "风机-P",
)


def write_multi_turbine_fixture(root: Path) -> None:
    columns_path = root / "columns.json"
    columns_path.write_text(json.dumps(SOURCE_COLUMNS, ensure_ascii=False), encoding="utf-8")

    for turbine_id in range(1, 17):
        timestamps = np.arange(10, dtype=np.float32)
        source = np.empty((10, len(SOURCE_COLUMNS)), dtype=np.float32)
        for feature_index in range(len(SOURCE_COLUMNS)):
            source[:, feature_index] = turbine_id * 100.0 + feature_index * 10.0 + timestamps
        data = np.column_stack((timestamps, source))
        np.save(root / f"turbine_{turbine_id:02d}.npy", data)


class TestMultiTurbineNormalization(unittest.TestCase):
    def test_shared_scaler_uses_all_turbine_training_rows_only(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            write_multi_turbine_fixture(root)
            repository = Repository(root=root, all_features=True, shared_feature_scaling=True)

            training_values = []
            for series in repository.series:
                training_values.append(series.raw[:7])
            joint_training_values = np.concatenate(training_values, axis=0)
            feature_means = np.mean(joint_training_values, axis=0)
            feature_stds = np.std(joint_training_values, axis=0)
            expected_scaled = (repository.series[0].raw - feature_means) / feature_stds

            np.testing.assert_allclose(repository.series[0].scaled, expected_scaled, atol=1e-6)
            np.testing.assert_allclose(repository.series[0].power_mean, feature_means[repository.power_index])
            np.testing.assert_allclose(repository.series[15].power_mean, feature_means[repository.power_index])
            self.assertNotAlmostEqual(float(np.mean(repository.series[0].scaled[:7, repository.power_index])), 0.0)


if __name__ == "__main__":
    unittest.main()
