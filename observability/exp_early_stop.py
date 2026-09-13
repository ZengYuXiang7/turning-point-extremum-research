"""Optional minimal early-stopping state machine.

Use only when the project already has early stopping or the user explicitly
asks to add it. Result formatting alone never requires it.
"""

from __future__ import annotations

from typing import Any


class EarlyStopping:
    def __init__(self, mode: str, patience: int, min_delta: float = 0.0):
        if mode not in {"max", "min"}:
            raise ValueError(f"unsupported mode: {mode}")
        if patience < 1:
            raise ValueError("patience must be positive")
        self.mode = mode
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best: float | None = None
        self.best_epoch: int | None = None
        self.counter = 0
        self.stopped_epoch: int | None = None

    def step(self, current: float, epoch: int) -> bool:
        current = float(current)
        improved = self.best is None or (
            current > self.best + self.min_delta if self.mode == "max" else current < self.best - self.min_delta
        )
        if improved:
            self.best = current
            self.best_epoch = int(epoch)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stopped_epoch = int(epoch)
        return improved

    @property
    def should_stop(self) -> bool:
        return self.stopped_epoch is not None


def build_early_stop_summary(stopper: EarlyStopping | None) -> dict[str, Any] | None:
    if stopper is None:
        return None
    return {
        "patience_effective": stopper.patience,
        "min_delta": stopper.min_delta,
        "best": stopper.best,
        "best_epoch": stopper.best_epoch,
        "counter": stopper.counter,
        "stopped_epoch": stopper.stopped_epoch,
    }
