"""Task-agnostic metric helpers."""

from __future__ import annotations

from typing import Any


MINIMIZE_HINTS = ("loss", "error", "err", "mae", "mse", "rmse", "mape", "wape", "wer", "per")


def is_improved(current: float, best: float | None, direction: str, min_delta: float = 0.0) -> bool:
    if direction not in {"max", "min"}:
        raise ValueError(f"unsupported direction: {direction}")
    if best is None:
        return True
    if direction == "max":
        return float(current) > float(best) + float(min_delta)
    return float(current) < float(best) - float(min_delta)


def validate_monitor_direction(metric_spec: dict[str, Any], early_stop: bool = True) -> bool:
    direction = metric_spec.get("direction")
    if direction in {"max", "min"}:
        return True
    if early_stop:
        raise ValueError("early_stop=True requires metric_spec['direction'] in {'max', 'min'}")
    return False


def build_metric_spec(metrics: list[str] | None = None, primary_metric: str | None = None, direction: str | None = None):
    names = list(metrics or ["loss"])
    primary = primary_metric or names[0]
    if direction is None and primary.lower() in MINIMIZE_HINTS:
        direction = "min"
    return {
        "task_type": "unknown",
        "metrics": names,
        "display_metrics": names,
        "primary_metric": primary,
        "direction": direction or "NEEDS_USER_CONFIRMATION",
        "source": "valid",
        "metric_shapes": {},
        "metric_aliases": {},
    }
