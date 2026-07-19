"""Multi-seed optimization statistics compatible with the current optimizer API."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from isrs_scl.link import LinkModel
from isrs_scl.optimization.adaptive_isrs import AdaptiveLaunchOptimizer, OptimizationResult


@dataclass(frozen=True)
class MultiSeedOptimizationResult:
    best_result: OptimizationResult
    run_summary: pd.DataFrame
    history: pd.DataFrame
    confidence: dict[str, float | str]


def _interval(values: np.ndarray) -> tuple[float, float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.nan, np.nan, np.nan
    return tuple(map(float, np.percentile(finite, [2.5, 50, 97.5])))


def _metric(result: OptimizationResult, key: str, *, optimized: bool) -> float:
    source = result.optimized_metrics if optimized else result.initial_metrics
    return float(source.get(key, np.nan))


def run_multiseed_optimization(link: LinkModel, cfg: dict, initial_profile_dbm: np.ndarray) -> MultiSeedOptimizationResult:
    count = max(int(cfg["optimization"].get("multi_seed_runs", 1)), 1)
    base_seed = int(cfg["optimization"].get("seed", 0))
    results: list[OptimizationResult] = []
    rows: list[dict[str, Any]] = []
    histories: list[pd.DataFrame] = []
    for run in range(count):
        run_cfg = deepcopy(cfg)
        seed = base_seed + run
        run_cfg["optimization"]["seed"] = seed
        result = AdaptiveLaunchOptimizer(link, run_cfg).optimize(initial_profile_dbm)
        results.append(result)
        initial_capacity = _metric(result, "target_thresholded_net_capacity_tbps", optimized=False)
        optimized_capacity = _metric(result, "target_thresholded_net_capacity_tbps", optimized=True)
        row: dict[str, Any] = {
            "run": run, "seed": seed, "improved": result.improved,
            "selected_restart": result.selected_restart, "acceptance_reason": result.acceptance_reason,
            "initial_objective": result.initial_objective, "optimized_objective": result.optimized_objective,
            "initial_air_tbps": _metric(result, "target_air_tbps", optimized=False),
            "optimized_air_tbps": _metric(result, "target_air_tbps", optimized=True),
            "initial_thresholded_net_capacity_tbps": initial_capacity,
            "optimized_thresholded_net_capacity_tbps": optimized_capacity,
            "capacity_gain_tbps": optimized_capacity - initial_capacity,
            "initial_working_fraction": _metric(result, "target_working_fraction", optimized=False),
            "optimized_working_fraction": _metric(result, "target_working_fraction", optimized=True),
            "initial_minimum_ngmi": _metric(result, "target_minimum_ngmi", optimized=False),
            "optimized_minimum_ngmi": _metric(result, "target_minimum_ngmi", optimized=True),
            "robust_gain_ci95_low_tbps": float(result.robust_summary.get("ci95_low", np.nan)),
            "robust_probability_positive": float(result.robust_summary.get("probability_positive", np.nan)),
            "robust_training_hash": result.robust_training_hash or "",
        }
        for band in ("S", "C", "L"):
            row[f"optimized_working_fraction_{band}"] = _metric(result, f"target_working_fraction_{band}", optimized=True)
            row[f"band_feasible_{band}"] = bool(row[f"optimized_working_fraction_{band}"] >= float(cfg["optimization"]["minimum_band_working_fraction"][band]))
        rows.append(row)
        if not result.history.empty:
            history = result.history.copy()
            history["run"] = run
            histories.append(history)
    summary = pd.DataFrame(rows)
    history = pd.concat(histories, ignore_index=True) if histories else pd.DataFrame()
    best_index = max(range(len(results)), key=lambda index: (
        bool(results[index].improved),
        _metric(results[index], "target_thresholded_net_capacity_tbps", optimized=True),
        float(results[index].robust_summary.get("ci95_low", -np.inf)),
        _metric(results[index], "target_air_tbps", optimized=True),
        -results[index].optimized_objective,
    ))
    low, median, high = _interval(summary["capacity_gain_tbps"].to_numpy())
    robust_low, robust_median, robust_high = _interval(summary["robust_gain_ci95_low_tbps"].to_numpy())
    confidence: dict[str, float | str] = {
        "runs": float(count), "method": "percentile_interval" if count >= 5 else "descriptive_only",
        "improved_fraction": float(summary["improved"].mean()),
        "capacity_gain_ci95_low_tbps": low, "capacity_gain_median_tbps": median, "capacity_gain_ci95_high_tbps": high,
        "robust_ci_low_distribution_low": robust_low, "robust_ci_low_distribution_median": robust_median, "robust_ci_low_distribution_high": robust_high,
    }
    return MultiSeedOptimizationResult(results[best_index], summary, history, confidence)
