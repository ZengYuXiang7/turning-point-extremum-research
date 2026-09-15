"""Write a line-oriented human log and a structured JSON record per experiment."""

from __future__ import annotations

import json
import math
import os
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

from scipy.stats import ttest_rel


METRIC_LABELS = {
    "kendall_tau": "KT",
    "mape_percent": "MAPE",
    "Acc_10": "Acc@10",
    "MAE": "MAE",
    "MSE": "MSE",
    "RMSE": "RMSE",
    "NMAE": "NMAE",
    "NRMSE": "NRMSE",
}

CONTRACT_VERSION = "experiment-contract-v2"
REQUIRED_RESULT_KEYS = {
    "contract_version",
    "dataset",
    "result_name",
    "model_name",
    "purpose",
    "condition",
    "model_config",
    "train_config",
    "data_config",
    "mean_std",
    "significance",
    "rounds",
}


def _validate_result_key(dataset: str, result_name: str) -> None:
    for field, value in (("dataset", dataset), ("result_name", result_name)):
        if not isinstance(value, str) or not value or Path(value).name != value or value in {".", ".."}:
            raise ValueError(f"{field} must be one safe path component, got {value!r}")


def _validate_result_record(result: dict[str, Any], *, require_significance: bool = True,) -> None:
    required_keys = REQUIRED_RESULT_KEYS if require_significance else REQUIRED_RESULT_KEYS - {"significance"}
    missing = required_keys - result.keys()
    if missing:
        raise ValueError(f"result record is missing keys: {sorted(missing)}")
    if result["contract_version"] != CONTRACT_VERSION:
        raise ValueError(f"result record requires contract_version={CONTRACT_VERSION!r}, " f"got {result['contract_version']!r}")
    _validate_result_key(result["dataset"], result["result_name"])
    if not isinstance(result["model_name"], str) or not result["model_name"].strip():
        raise ValueError("model_name must be a non-empty string")


def _result_path(dataset: str, result_name: str, artifact_dir: str, suffix: str) -> Path:
    _validate_result_key(dataset, result_name)
    return Path("results") / artifact_dir / dataset / f"{result_name}{suffix}"


def result_report_path(dataset: str, result_name: str) -> Path:
    return _result_path(dataset, result_name, "reports", ".log")


def result_record_path(dataset: str, result_name: str) -> Path:
    return _result_path(dataset, result_name, "records", ".json")


def result_checkpoint_path(dataset: str, result_name: str, round_number: int) -> Path:
    _validate_result_key(dataset, result_name)
    if not isinstance(round_number, int) or isinstance(round_number, bool) or round_number < 1:
        raise ValueError(f"round_number must be a positive integer, got {round_number!r}")
    return Path("checkpoints") / dataset / result_name / f"round_{round_number}.pt"


def aggregate_test_metrics(rounds: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    if not rounds:
        raise ValueError("cannot aggregate zero rounds")
    names = set(rounds[0]["test_metrics"])
    for record in rounds[1:]:
        names.intersection_update(record["test_metrics"])
    return {
        name: {
            "mean": statistics.fmean(float(record["test_metrics"][name]) for record in rounds),
            "std": statistics.pstdev(float(record["test_metrics"][name]) for record in rounds),
        }
        for name in sorted(names)
    }


def _display_value(value: Any) -> str:
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def _metric_label(metric_name: str) -> str:
    return METRIC_LABELS.get(metric_name, metric_name)


def _ordered_metric_names(metrics: dict[str, Any]) -> list[str]:
    preferred = ("kendall_tau", "mape_percent", "Acc_10", "MAE", "MSE", "RMSE", "NMAE", "NRMSE")
    ordered = [name for name in preferred if name in metrics]
    ordered.extend(name for name in metrics if name not in ordered)
    return ordered


def _aligned_fields(items: list[tuple[str, Any]]) -> list[str]:
    width = max(len(name) for name, _ in items)
    return [f"{name:<{width}} : {_display_value(value)}" for name, value in items]


def _display_pvalue(value: float) -> str:
    return "<0.0001" if value < 0.0001 else f"{value:.4f}"


def _display_statistic(value: float | str) -> str:
    return value if isinstance(value, str) else f"{value:.4f}"


def _metric_values_by_seed(result: dict[str, Any]) -> dict[int, dict[str, float]]:
    values_by_seed: dict[int, dict[str, float]] = {}
    for round_result in result["rounds"]:
        seed = int(round_result["seed"])
        if seed in values_by_seed:
            raise ValueError(f"duplicate seed in result record: {seed}")
        values_by_seed[seed] = {name: float(value) for name, value in round_result["test_metrics"].items()}
    return values_by_seed


def _holm_adjust(p_values: list[float]) -> list[float]:
    adjusted = [0.0] * len(p_values)
    previous = 0.0
    for rank, index in enumerate(sorted(range(len(p_values)), key=p_values.__getitem__)):
        current = min(1.0, p_values[index] * (len(p_values) - rank))
        previous = max(previous, current)
        adjusted[index] = previous
    return adjusted


def build_dataset_significance(dataset: str, result_name: str, result: dict[str, Any], records_root: Path = Path("results") / "records",) -> list[dict[str, Any]]:
    _validate_result_record(result, require_significance=False)
    if result["dataset"] != dataset or result["result_name"] != result_name:
        raise ValueError("dataset/result_name arguments must match the result record")
    record_dir = records_root / dataset
    if not record_dir.exists():
        return []

    own_by_seed = _metric_values_by_seed(result)
    comparisons: list[dict[str, Any]] = []
    all_metric_tests: list[dict[str, Any]] = []
    for peer_path in sorted(record_dir.glob("*.json")):
        if peer_path.stem == result_name:
            continue
        peer_result = json.loads(peer_path.read_text(encoding="utf-8"))
        _validate_result_record(peer_result)
        if peer_path.stem != peer_result["result_name"]:
            raise ValueError(f"record filename does not match result_name: {peer_path}")
        if peer_result["condition"] != result["condition"]:
            continue
        peer_by_seed = _metric_values_by_seed(peer_result)
        metrics: dict[str, dict[str, Any]] = {}
        for metric_name in result["mean_std"]:
            seeds = sorted(seed for seed in own_by_seed.keys() & peer_by_seed.keys() if metric_name in own_by_seed[seed] and metric_name in peer_by_seed[seed])
            if len(seeds) < 2:
                continue
            own_values = [own_by_seed[seed][metric_name] for seed in seeds]
            peer_values = [peer_by_seed[seed][metric_name] for seed in seeds]
            if not all(math.isfinite(value) for value in (*own_values, *peer_values)):
                raise ValueError(f"non-finite value for significance test: {metric_name}")
            differences = [own_value - peer_value for own_value, peer_value in zip(own_values, peer_values, strict=True)]
            difference = statistics.fmean(differences)
            if all(value == 0.0 for value in differences):
                statistic, p_value = 0.0, 1.0
            elif all(value == differences[0] for value in differences):
                statistic, p_value = ("inf" if differences[0] > 0 else "-inf"), 0.0
            else:
                test = ttest_rel(own_values, peer_values, alternative="two-sided")
                statistic, p_value = float(test.statistic), float(test.pvalue)
            metric_test = {
                "paired_seeds": seeds,
                "n": len(seeds),
                "mean_difference": difference,
                "t_statistic": statistic,
                "p_value": p_value,
            }
            metrics[metric_name] = metric_test
            all_metric_tests.append(metric_test)
        if metrics:
            comparisons.append({ "against": peer_result["result_name"], "test": "paired_two_sided_t_test", "multiple_testing": "holm", "metrics": metrics, })
    for metric_test, adjusted_p_value in zip(all_metric_tests, _holm_adjust([float(metric_test["p_value"]) for metric_test in all_metric_tests]), strict=True,):
        metric_test["p_value_holm"] = adjusted_p_value
    return comparisons


def write_result_report(path: Path, result: dict[str, Any]) -> None:
    _validate_result_record(result)
    metrics = result["mean_std"]
    metric_names = _ordered_metric_names(metrics)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    def line(message: str) -> str:
        return message

    lines = [
        f"{'=' * 20} {timestamp} {'=' * 20}",
        line(f"Dataset : {result['dataset']}"),
        line(f"Model   : {result['model_name']}"),
        line(f"Result  : {result['result_name']}"),
        line(f"Purpose : {_display_value(result['purpose'])}"),
    ]
    lines.append(line("******************** Summary ********************"))
    summary_width = max(len(_metric_label(name)) for name in metric_names)
    lines.extend(line(f"{_metric_label(name):<{summary_width}} : " f"{metrics[name]['mean']:>9.4f} ± {metrics[name]['std']:>8.4f}") for name in metric_names)
    lines.append(line("******************** Significance ********************"))
    significance = result.get("significance", [])
    if not significance:
        lines.append(line("No compatible opponent record."))
    else:
        lines.append(line("Paired two-sided t-test; Holm-adjusted p-values."))
        for comparison in significance:
            for metric_name, metric_test in comparison["metrics"].items():
                lines.append(line(f"vs={comparison['against']} {_metric_label(metric_name)}: " f"delta={metric_test['mean_difference']:.4f} " f"t={_display_statistic(metric_test['t_statistic'])} " f"p={_display_pvalue(metric_test['p_value'])} " f"p_holm={_display_pvalue(metric_test['p_value_holm'])} " f"n={metric_test['n']}"))
    lines.append(line("******************** Rounds ********************"))
    for index, record in enumerate(result["rounds"], start=1):
        best = ", ".join(f"V-{_metric_label(name)}={float(value):.4f}" for name, value in record["best_valid"].items()) or "N/A"
        values = " ".join(f"{_metric_label(name)}={float(record['test_metrics'][name]):.4f}" for name in metric_names)
        checkpoint_name = Path(record["checkpoint"]).name
        lines.append(line(f"R={index} S={record['seed']} BE={record['best_epoch']} " f"{best} | {values} | CKPT={checkpoint_name}"))
    lines.append(line("******************** Model Configuration ********************"))
    lines.extend(line(text) for text in _aligned_fields(list(result["model_config"].items())))
    lines.append(line("******************** Training Configuration ********************"))
    lines.extend(line(text) for text in _aligned_fields(list(result["train_config"].items())))
    data_config = result["data_config"]
    data_items = [
        ("source", data_config["source"]),
        ("domain", result["condition"]["domain"]),
        ("available_samples", data_config["available_samples"]),
        ("split", data_config["split"]),
        ("split_seeds", data_config["split_seeds"]),
        ("train/valid/test", f"{data_config['train_samples']}/{data_config['valid_samples']}/{data_config['test_samples']}"),
        ("features", data_config["features"]),
        ("feature_normalization", data_config["feature_normalization"]),
        ("target", data_config["target"]),
        ("target_transform", data_config["target_transform"]),
    ]
    lines.append(line("******************** Data Processing ********************"))
    lines.extend(line(text) for text in _aligned_fields(data_items))
    report = "\n".join(lines) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(report)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def write_result_record(path: Path, result: dict[str, Any]) -> None:
    _validate_result_record(result)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)
