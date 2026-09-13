"""Compact, transient terminal status lines."""

from __future__ import annotations

import math
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

    def format_epoch(self, *, run: int, total_runs: int, epoch: int, total_epochs: int, train_loss: Any, monitor: str,
                     mode: str, current: Any, best: Any, best_epoch: Any, patience: str | None = None) -> str:
        stat = f"STAT run={run}/{total_runs} ep={epoch}/{total_epochs} train_loss={_value(train_loss)}"
        mon = f"MON valid/{monitor} {mode} cur={_value(current)} best={_value(best)}@{_value(best_epoch)}"
        return stat + (f"\n{mon} patience={patience}" if patience else f"\n{mon}")

    def format_round(self, *, round_index: int, seed: int, best_epoch: Any, best_valid: Any, test_metrics: dict[str, Any]) -> str:
        metrics = " ".join(f"{name}={_value(value)}" for name, value in test_metrics.items())
        return (
            f"ROUND round={round_index} seed={seed} best_ep={_value(best_epoch)} "
            f"best_valid={_value(best_valid)} {metrics}"
        ).rstrip()
