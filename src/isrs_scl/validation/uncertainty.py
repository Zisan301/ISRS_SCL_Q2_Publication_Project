"""Correlated uncertainty propagation with independent holdout evaluation."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from isrs_scl.fiber.amplification import dbm_to_w
from isrs_scl.link import LinkModel
from isrs_scl.optimization.robust import (
    RobustScenarioBatch,
    cvar_lower,
    generate_training_batch,
    paired_gain_summary,
)
from isrs_scl.system.capacity import summarize_capacity
from isrs_scl.system.grid import build_grid


@dataclass(frozen=True)
class UncertaintyResult:
    samples: pd.DataFrame
    summary: pd.DataFrame
    sensitivity: pd.DataFrame
    convergence: pd.DataFrame
    paired_gains: pd.DataFrame
    successful_fraction: float
    batch_hash: str


def _correlation(cfg: Mapping[str, Any], names: tuple[str, ...]) -> np.ndarray | None:
    value = cfg.get("correlation")
    if value is None:
        return None
    if isinstance(value, Mapping):
        matrix = np.eye(len(names))
        for pair, coefficient in value.items():
            left, right = str(pair).split("|", 1)
            i, j = names.index(left), names.index(right)
            matrix[i, j] = matrix[j, i] = float(coefficient)
        value = matrix
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (len(names), len(names)):
        raise ValueError("uncertainty.correlation shape does not match distributions")
    if not np.allclose(matrix, matrix.T, atol=1e-12) or not np.allclose(np.diag(matrix), 1.0):
        raise ValueError("uncertainty.correlation must be symmetric with a unit diagonal")
    if np.linalg.eigvalsh(matrix).min() < -1e-10:
        raise ValueError("uncertainty.correlation is not positive semidefinite")
    return matrix


def build_holdout_batch(cfg: Mapping[str, Any], *, samples: int | None = None, seed: int | None = None) -> RobustScenarioBatch:
    uncertainty = cfg["uncertainty"]
    distributions = uncertainty["distributions"]
    names = tuple(distributions)
    return generate_training_batch(
        distributions,
        samples=int(samples or uncertainty["holdout_samples"]),
        seed=int(seed if seed is not None else uncertainty["holdout_seed"]),
        correlation=_correlation(uncertainty, names),
        role="publication_holdout",
    )


def _perturb_config(cfg: Mapping[str, Any], values: Mapping[str, float]) -> dict[str, Any]:
    scenario = deepcopy(dict(cfg))
    attenuation_scale = float(values.get("attenuation_scale", 1.0))
    scenario["fiber"]["attenuation_anchors"]["db_per_km"] = [float(x) * attenuation_scale for x in scenario["fiber"]["attenuation_anchors"]["db_per_km"]]
    scenario["fiber"]["gamma_per_w_km_at_1550"] *= float(values.get("gamma_scale", 1.0))
    dispersion_scale = float(values.get("dispersion_scale", 1.0))
    scenario["fiber"]["dispersion_ps_nm_km_at_1550"] *= dispersion_scale
    scenario["fiber"]["dispersion_slope_ps_nm2_km"] *= dispersion_scale
    nf_offset = float(values.get("noise_figure_offset_db", 0.0))
    for band in scenario["amplification"]["bands"].values():
        band["noise_figure_db"] = [float(x) + nf_offset for x in band["noise_figure_db"]]
    scenario["raman"]["gain_peak_m_per_w"] *= float(values.get("raman_gain_scale", 1.0))
    pump_scale = float(values.get("pump_power_scale", 1.0))
    for pump in scenario["raman"].get("pumps", []):
        pump["power_w"] = float(pump["power_w"]) * pump_scale
    scenario["nli"]["transceiver_snr_db"] += float(values.get("transceiver_snr_offset_db", 0.0))
    return scenario


def _band_fractions(result: Any, bands: np.ndarray, threshold: float) -> dict[str, float]:
    return {str(band): float(np.mean(result.ngmi[bands == band] >= threshold)) for band in ("S", "C", "L") if np.any(bands == band)}


def run_uncertainty_analysis(
    cfg: dict[str, Any],
    profiles_dbm: Mapping[str, np.ndarray],
    target_spans: int,
    *,
    samples: int | None = None,
    seed: int | None = None,
    batch: RobustScenarioBatch | None = None,
) -> UncertaintyResult:
    holdout = batch or build_holdout_batch(cfg, samples=samples, seed=seed)
    rows: list[dict[str, Any]] = []
    for sample_index, values in enumerate(holdout.transformed):
        scenario = _perturb_config(cfg, values)
        try:
            grid = build_grid(scenario["grid"])
            link = LinkModel(grid, scenario)
            for strategy, profile in profiles_dbm.items():
                launch = np.asarray(profile, dtype=float)
                if launch.shape != grid.frequencies_hz.shape:
                    raise ValueError(f"Profile {strategy!r} shape does not match the grid")
                result = link.evaluate(dbm_to_w(launch), int(target_spans))
                capacity = summarize_capacity(
                    result.gmi,
                    result.ngmi,
                    float(scenario["fec"]["ngmi_target"]),
                    float(scenario["modulation"]["symbol_rate_gbaud"]) * 1e9,
                    int(scenario["modulation"]["bits_per_symbol_per_pol"]),
                    float(scenario["fec"]["overhead_fraction"]),
                )
                band_fractions = _band_fractions(result, grid.bands, float(scenario["fec"]["ngmi_target"]))
                rows.append({
                    "sample": sample_index,
                    "strategy": strategy,
                    "success": 1,
                    "error": "",
                    "scenario_hash": holdout.batch_hash,
                    **{key: float(value) for key, value in values.items()},
                    "minimum_gsnr_db": float(np.min(result.gsnr_db)),
                    "mean_gsnr_db": float(np.mean(result.gsnr_db)),
                    "gsnr_std_db": float(np.std(result.gsnr_db)),
                    "minimum_ngmi": float(np.min(result.ngmi)),
                    "mean_ngmi": float(np.mean(result.ngmi)),
                    "working_fraction": float(capacity.working_fraction),
                    "air_tbps": float(capacity.air_bps / 1e12),
                    "thresholded_net_capacity_tbps": float(capacity.thresholded_net_line_bps / 1e12),
                    **{f"working_fraction_{band}": value for band, value in band_fractions.items()},
                })
        except Exception as exc:
            for strategy in profiles_dbm:
                rows.append({
                    "sample": sample_index,
                    "strategy": strategy,
                    "success": 0,
                    "error": f"{type(exc).__name__}: {exc}",
                    "scenario_hash": holdout.batch_hash,
                    **{key: float(value) for key, value in values.items()},
                })
    frame = pd.DataFrame(rows)
    successful_samples = frame.groupby("sample")["success"].min()
    successful_fraction = float(successful_samples.mean()) if not successful_samples.empty else 0.0
    successful = frame[frame["success"] == 1].copy()

    metrics = [
        "minimum_gsnr_db", "minimum_ngmi", "working_fraction", "air_tbps",
        "thresholded_net_capacity_tbps", "working_fraction_S", "working_fraction_C", "working_fraction_L",
    ]
    summary_rows: list[dict[str, Any]] = []
    alpha = float(cfg["uncertainty"].get("cvar_alpha", 0.10))
    for strategy, group in successful.groupby("strategy", sort=False):
        for metric in metrics:
            if metric not in group:
                continue
            values = group[metric].dropna().to_numpy(float)
            if values.size == 0:
                continue
            low, median, high = np.percentile(values, [2.5, 50, 97.5])
            summary_rows.append({
                "strategy": strategy, "metric": metric, "samples": values.size,
                "mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                "ci95_low": float(low), "median": float(median), "ci95_high": float(high),
                "cvar_lower": cvar_lower(values, alpha), "worst_case": float(np.min(values)),
            })
    summary = pd.DataFrame(summary_rows)

    def _safe_spearman(pair: pd.DataFrame, parameter: str, metric: str) -> tuple[float, str]:
        if len(pair) < 4:
            return np.nan, "insufficient_samples"
        if pair[parameter].nunique(dropna=True) <= 1:
            return np.nan, "constant_parameter"
        if pair[metric].nunique(dropna=True) <= 1:
            return np.nan, "constant_metric"
        rho = pair[parameter].corr(pair[metric], method="spearman")
        if pd.isna(rho):
            return np.nan, "undefined"
        return float(rho), "ok"

    parameter_names = [name for name in holdout.names if name in successful]
    sensitivity_rows: list[dict[str, Any]] = []
    for strategy, group in successful.groupby("strategy", sort=False):
        for parameter in parameter_names:
            for metric in metrics:
                if metric not in group:
                    continue
                pair = group[[parameter, metric]].dropna()
                rho, status = _safe_spearman(pair, parameter, metric)
                sensitivity_rows.append({
                    "strategy": strategy,
                    "parameter": parameter,
                    "metric": metric,
                    "spearman_rho": rho,
                    "abs_spearman_rho": abs(rho) if np.isfinite(rho) else np.nan,
                    "sensitivity_status": status,
                })
    sensitivity = pd.DataFrame(sensitivity_rows)

    convergence_rows: list[dict[str, Any]] = []
    counts = sorted(set([min(len(holdout.transformed), x) for x in (16, 32, 64, 128, 256, 512, len(holdout.transformed)) if x >= 4]))
    for count in counts:
        subset = successful[successful["sample"] < count]
        for strategy, group in subset.groupby("strategy", sort=False):
            values = group["thresholded_net_capacity_tbps"].dropna().to_numpy(float)
            if values.size:
                low, median, high = np.percentile(values, [2.5, 50, 97.5])
                convergence_rows.append({"samples": count, "strategy": strategy, "ci95_low": low, "median": median, "ci95_high": high})
    convergence = pd.DataFrame(convergence_rows)

    paired_rows: list[dict[str, Any]] = []
    pivot = successful.pivot_table(index="sample", columns="strategy", values="thresholded_net_capacity_tbps", aggfunc="first")
    for baseline in ("flat", "fixed"):
        if {"adaptive", baseline}.issubset(pivot.columns) and len(pivot.dropna(subset=["adaptive", baseline])) >= 4:
            pair = pivot.dropna(subset=["adaptive", baseline])
            stats = paired_gain_summary(pair["adaptive"], pair[baseline], bootstrap_samples=int(cfg["uncertainty"].get("bootstrap_samples", 2000)), seed=int(cfg["uncertainty"]["holdout_seed"]), cvar_alpha=alpha)
            paired_rows.append({"adaptive": "adaptive", "baseline": baseline, **stats.__dict__})
    paired = pd.DataFrame(paired_rows)
    return UncertaintyResult(frame, summary, sensitivity, convergence, paired, successful_fraction, holdout.batch_hash)
