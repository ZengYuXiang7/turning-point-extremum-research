"""Compact, transient terminal status lines."""

from __future__ import annotations

import math
import os
import sys
from typing import Any


def _value(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and math.isfinite(value):
        if value != 0.0 and (abs(value) < 1e-4 or abs(value) >= 1e4):
            return f"{value:.3e}"
        return f"{value:.4f}"
    return str(value).replace("\n", " ").replace("|", "/")


class LineStatusFormatter:
    """Build plain STAT, MON and ROUND lines without file logging."""

    def format_epoch(self, *, run: int, total_runs: int, epoch: int, total_epochs: int, train_loss: Any, monitor: str, mode: str, current: Any, best: Any, best_epoch: Any, patience: str | None = None) -> str:
        stat = f"STAT run={run}/{total_runs} ep={epoch}/{total_epochs} train_loss={_value(train_loss)}"
        mon = f"MON valid/{monitor} {mode} cur={_value(current)} best={_value(best)}@{_value(best_epoch)}"
        return stat + (f"\n{mon} patience={patience}" if patience else f"\n{mon}")

    def format_round(self, *, round_index: int, seed: int, best_epoch: Any, best_valid: Any, test_metrics: dict[str, Any]) -> str:
        metrics = " ".join(f"{name}={_value(value)}" for name, value in test_metrics.items())
        return (f"ROUND round={round_index} seed={seed} best_ep={_value(best_epoch)} " f"best_valid={_value(best_valid)} {metrics}").rstrip()


def dynamic_tqdm_enabled(tqdm):
    # 仅前台终端允许动态刷新，后台进程使用普通日志。
    if tqdm == 0:
        return False
    stream_fd = sys.stderr.fileno()
    if not os.isatty(stream_fd):
        return False
    return os.tcgetpgrp(stream_fd) == os.getpgrp()


class EpochProgressPolicy:
    """首个正常 epoch 用代表性耗时确定进度条与文本日志节奏。"""

    SLOW_SECONDS = 5.0

    def __init__(self, tqdm_enabled, total_epochs):
        self.tqdm_enabled = tqdm_enabled
        self.total_epochs = total_epochs
        self.show_bar = tqdm_enabled
        self.interval = None

    def show_progress(self):
        # 前台首轮默认显示，首轮结束后按耗时固定后续状态。
        return self.show_bar

    def observe(self, seconds):
        # 首个正常 epoch 后固定进度条与日志间隔，后续不再变化
        if self.interval is not None:
            return
        self.show_bar = self.tqdm_enabled and seconds > self.SLOW_SECONDS
        if seconds > self.SLOW_SECONDS:
            self.interval = 1
        elif self.total_epochs < 1000:
            self.interval = 10
        else:
            self.interval = 100

    def should_log(self, epoch, extra=False):
        if epoch == 1 or epoch == self.total_epochs or extra:
            return True
        return epoch % self.interval == 0
