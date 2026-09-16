import unittest
from unittest.mock import patch

from observability.exp_progress import EpochProgressPolicy, dynamic_tqdm_enabled


class TestEpochProgressPolicy(unittest.TestCase):
    def test_tqdm_disabled_never_shows_a_bar(self):
        # 显式关闭优先于首轮与耗时策略。
        progress = EpochProgressPolicy(False, 20)
        self.assertFalse(progress.show_progress())
        progress.observe(6.0)
        self.assertFalse(progress.show_progress())
        self.assertTrue(progress.should_log(1))
        self.assertTrue(progress.should_log(2))

    def test_fast_first_epoch_disables_later_bars_and_logs_each_tenth_epoch(self):
        # 首轮默认显示，耗时不超过5秒后改为稀疏文本日志。
        progress = EpochProgressPolicy(True, 20)
        self.assertTrue(progress.show_progress())
        progress.observe(5.0)
        self.assertFalse(progress.show_progress())
        self.assertTrue(progress.should_log(1))
        self.assertFalse(progress.should_log(2))
        self.assertTrue(progress.should_log(10))
        self.assertTrue(progress.should_log(20))

    def test_slow_first_epoch_keeps_bars_and_logs_every_epoch(self):
        # 超过5秒后，进度条与文本日志都保持逐轮显示。
        progress = EpochProgressPolicy(True, 20)
        progress.observe(5.01)
        self.assertTrue(progress.show_progress())
        self.assertTrue(progress.should_log(2))

    def test_long_fast_run_logs_each_hundredth_epoch(self):
        # 长任务的快速 epoch 使用更稀疏的固定节奏。
        progress = EpochProgressPolicy(True, 1_000)
        progress.observe(5.0)
        self.assertFalse(progress.should_log(99))
        self.assertTrue(progress.should_log(100))
        self.assertTrue(progress.should_log(1_000))

    def test_background_process_disables_dynamic_tqdm(self):
        # 后台进程组不同于终端前台进程组，禁止动态刷新。
        with (
            patch("observability.exp_progress.os.isatty", return_value=True),
            patch("observability.exp_progress.os.tcgetpgrp", return_value=10),
            patch("observability.exp_progress.os.getpgrp", return_value=11),
        ):
            self.assertFalse(dynamic_tqdm_enabled(1))


if __name__ == "__main__":
    unittest.main()
