import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import config as project_config
from config import CORRELATED_HISTORY_COLUMNS, DATASET_ROOT
from data_provider.common import Repository


class TestFeatureSelectionHandoff(unittest.TestCase):
    def setUp(self):
        # 构造包含项目全部原始字段的小型连续序列
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        columns_path = DATASET_ROOT / "columns.json"
        self.source_columns = json.loads(columns_path.read_text(encoding="utf-8"))
        (self.root / "columns.json").write_text(json.dumps(self.source_columns, ensure_ascii=False), encoding="utf-8",)
        row_count = 10
        timestamps = np.arange(row_count, dtype=np.float64)[:, None]
        row_values = np.arange(row_count, dtype=np.float64)[:, None]
        column_values = np.arange(len(self.source_columns), dtype=np.float64)[None, :]
        features = row_values + column_values
        self.data = np.concatenate([timestamps, features], axis=1)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def build_repository(self, feature_path: str):
        # 让16台风机共享同一小型数组以验证字段装配
        with (
            patch.object(project_config, "HISTORY_FEATURE_PATH", feature_path),
            patch("data_provider.common.np.load", return_value=self.data),
        ):
            repository = Repository(root=self.root, correlated_features=True)
        return repository

    def test_json_order_drives_twenty_predictors_plus_power(self):
        # 模拟LightGBM按重要性输出Top20并固定追加历史功率
        predictor_names = self.source_columns[:20]
        feature_names = predictor_names.copy()
        feature_names.append("风机-P")
        feature_path = self.root / "top20_features.json"
        feature_path.write_text(json.dumps(feature_names, ensure_ascii=False), encoding="utf-8",)

        repository = self.build_repository(str(feature_path))

        self.assertEqual(repository.feature_names, feature_names)
        self.assertEqual(repository.power_index, 20)
        self.assertEqual(repository.series[0].raw.shape[1], 21)

    def test_empty_path_keeps_original_corr21_branch(self):
        # 空路径继续使用原相关系数消融字段
        repository = self.build_repository("")

        self.assertEqual(repository.feature_names, CORRELATED_HISTORY_COLUMNS)
        self.assertEqual(repository.power_index, 20)
        self.assertEqual(repository.series[0].raw.shape[1], 21)


if __name__ == "__main__":
    unittest.main()
