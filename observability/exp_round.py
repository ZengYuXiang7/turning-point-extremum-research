"""One compact terminal line per completed round."""

from __future__ import annotations

from typing import Any


def _metric(name: str, value: Any) -> str:
    try:
        return f"{name}={float(value):.4f}"
    except (TypeError, ValueError):
        return f"{name}=N/A"


def format_round_summary(
    run_id: int,
    seed: int,
    best_epoch: int | None,
    monitor_name: str,
    best_value: Any,
    test_metrics: dict[str, Any],
) -> str:
    values = " ".join(_metric(name, value) for name, value in test_metrics.items())
    return (
        f"ROUND round={run_id + 1} seed={seed} best_ep={best_epoch if best_epoch is not None else 'N/A'} "
        f"{_metric(f'Valid{monitor_name}', best_value)} {values}"
    ).rstrip()
