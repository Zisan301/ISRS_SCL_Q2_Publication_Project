"""Band-aware robust launch-profile optimization.

The optimizer preserves total launch power and per-channel limits, evaluates all
configured span counts, penalizes per-band infeasibility explicitly, and can use
an immutable common-random-number robust-training batch.  A candidate is only
accepted when manuscript-level nominal and robust constraints pass.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd

from isrs_scl.fiber.amplification import dbm_to_w
from isrs_scl.link import LinkModel
from isrs_scl.optimization.constraints import project_launch_profile_dbm, second_difference_energy, total_power_w_from_dbm
from isrs_scl.optimization.robust import RobustScenarioBatch, cvar_lower, generate_training_batch, paired_gain_summary
from isrs_scl.system.capacity import summarize_capacity
from isrs_scl.system.grid import build_grid


@dataclass(frozen=True)
class OptimizationResult:
    initial_profile_dbm: np.ndarray
    optimized_profile_dbm: np.ndarray
    initial_objective: float
    optimized_objective: float
    improved: bool
    history: pd.DataFrame
    selected_restart: int = 0
    initial_metrics: dict[str, float] = field(default_factory=dict)
    optimized_metrics: dict[str, float] = field(default_factory=dict)
    acceptance_reason: str = ""
    robust_training_hash: str | None = None
    robust_summary: dict[str, float] = field(default_factory=dict)
    feasibility: dict[str, bool] = field(default_factory=dict)

    def _metric_value(self, which: str, *names: str) -> float:
        data = self.initial_metrics if which == "initial" else self.optimized_metrics
        for name in names:
            if name in data:
                return float(data[name])
        return 0.0

    @property
    def initial_soft_fec_tbps(self) -> float:
        return self._metric_value("initial", "target_soft_fec_net_tbps", "soft_fec_net_tbps", "target_thresholded_net_capacity_tbps")

    @property
    def optimized_soft_fec_tbps(self) -> float:
        return self._metric_value("optimized", "target_soft_fec_net_tbps", "soft_fec_net_tbps", "target_thresholded_net_capacity_tbps")

    @property
    def initial_fec_net_tbps(self) -> float:
        return self._metric_value("initial", "target_fec_net_tbps", "fec_net_tbps", "target_thresholded_net_capacity_tbps")

    @property
    def optimized_fec_net_tbps(self) -> float:
        return self._metric_value("optimized", "target_fec_net_tbps", "fec_net_tbps", "target_thresholded_net_capacity_tbps")

    @property
    def initial_air_tbps(self) -> float:
        return self._metric_value("initial", "target_air_tbps", "air_tbps")

    @property
    def optimized_air_tbps(self) -> float:
        return self._metric_value("optimized", "target_air_tbps", "air_tbps")


def fixed_preemphasis_profile_dbm(frequencies_hz: np.ndarray, flat_dbm: float, s_to_l_tilt_db: float, minimum_dbm: float, maximum_dbm: float) -> np.ndarray:
    frequency = np.asarray(frequencies_hz, dtype=float)
    if frequency.ndim != 1 or frequency.size < 2:
        raise ValueError("frequencies_hz must be one-dimensional")
    normalized = (frequency - frequency.mean()) / max(float(np.ptp(frequency)), 1.0)
    target = frequency.size * float(dbm_to_w(flat_dbm))
    return project_launch_profile_dbm(float(flat_dbm) + float(s_to_l_tilt_db) * normalized, target, minimum_dbm, maximum_dbm, 0.0)


def _scenario_cfg(cfg: Mapping[str, Any], values: Mapping[str, float]) -> dict[str, Any]:
    scenario = deepcopy(dict(cfg))
    scenario["fiber"]["attenuation_anchors"]["db_per_km"] = [float(x) * float(values.get("attenuation_scale", 1.0)) for x in scenario["fiber"]["attenuation_anchors"]["db_per_km"]]
    scenario["fiber"]["gamma_per_w_km_at_1550"] *= float(values.get("gamma_scale", 1.0))
    dispersion = float(values.get("dispersion_scale", 1.0))
    scenario["fiber"]["dispersion_ps_nm_km_at_1550"] *= dispersion
    scenario["fiber"]["dispersion_slope_ps_nm2_km"] *= dispersion
    nf = float(values.get("noise_figure_offset_db", 0.0))
    for band in scenario["amplification"]["bands"].values():
        band["noise_figure_db"] = [float(x) + nf for x in band["noise_figure_db"]]
    scenario["raman"]["gain_peak_m_per_w"] *= float(values.get("raman_gain_scale", 1.0))
    pump_scale = float(values.get("pump_power_scale", 1.0))
    for pump in scenario["raman"].get("pumps", []):
        pump["power_w"] *= pump_scale
    scenario["nli"]["transceiver_snr_db"] += float(values.get("transceiver_snr_offset_db", 0.0))
    return scenario


class AdaptiveLaunchOptimizer:
    def __init__(self, link: LinkModel, cfg: dict, robust_batch: RobustScenarioBatch | None = None):
        self.link, self.cfg = link, cfg
        opt_defaults = {
            "target_spans": 1,
            "evaluation_spans": None,
            "seed": 1,
            "control_points": 5,
            "iterations": 10,
            "learning_rate": 0.05,
            "spsa_perturbation_db": 0.1,
            "restarts": 1,
            "restart_sigma_db": 0.25,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "air_weight": 1.0,
            "margin_weight": 0.5,
            "outage_weight": 12.0,
            "variance_weight": 0.05,
            "band_balance_weight": 10.0,
            "smoothness_weight": 0.02,
            "robust_weight": 1.0,
            "robust_cvar_alpha": 0.10,
            "robust_training_samples": 0,
            "robust_training_seed": 20260720,
            "minimum_band_working_fraction": {"S": 0.0, "C": 0.0, "L": 0.0},
            "minimum_nominal_gain_tbps": 0.0,
            "minimum_robust_gain_ci_low_tbps": 0.0,
            "capacity_regression_tolerance_tbps": 1e-9,
            "projection_smoothing_strength": 0.0,
            "early_stopping_patience": 10,
            "early_stopping_tolerance": 1e-6,
            "softmin_temperature_ngmi": 0.01,
        }
        self.opt = {**opt_defaults, **dict(cfg.get("optimization", {}))}
        if self.opt.get("evaluation_spans") is None:
            self.opt["evaluation_spans"] = [int(self.opt["target_spans"])]
        if not isinstance(self.opt.get("minimum_band_working_fraction"), Mapping):
            self.opt["minimum_band_working_fraction"] = opt_defaults["minimum_band_working_fraction"]
        self.launch_cfg = {
            "min_power_dbm_per_channel": -5.0,
            "max_power_dbm_per_channel": 3.0,
            **dict(cfg.get("launch", {})),
        }
        self.threshold = float(cfg.get("fec", {}).get("ngmi_target", 0.90))
        modulation = cfg.get("modulation", {})
        self.symbol_rate_hz = float(modulation.get("symbol_rate_gbaud", 32.0)) * 1e9
        self.bits = int(modulation.get("bits_per_symbol_per_pol", 4))
        self.overhead = float(cfg.get("fec", {}).get("overhead_fraction", 0.25))
        if robust_batch is None and int(self.opt.get("robust_training_samples", 0)) >= 4:
            distributions = cfg.get("uncertainty", {}).get("distributions", {})
            correlation = cfg.get("uncertainty", {}).get("correlation")
            matrix = None if correlation is None or isinstance(correlation, Mapping) else np.asarray(correlation, dtype=float)
            robust_batch = generate_training_batch(
                distributions,
                samples=int(self.opt["robust_training_samples"]),
                seed=int(self.opt["robust_training_seed"]),
                correlation=matrix,
                role="robust_training",
            )
        self.robust_batch = robust_batch
        self._scenario_links: list[LinkModel] | None = None

    def _evaluation_spans(self) -> tuple[int, ...]:
        target = int(self.opt["target_spans"])
        return tuple(sorted({int(value) for value in self.opt.get("evaluation_spans", [target])} | {target}))

    def _project(self, profile: np.ndarray, total_power_w: float) -> np.ndarray:
        return project_launch_profile_dbm(
            profile,
            total_power_w,
            float(self.launch_cfg["min_power_dbm_per_channel"]),
            float(self.launch_cfg["max_power_dbm_per_channel"]),
            float(self.opt.get("projection_smoothing_strength", 0.0)),
        )

    def _control_to_channels(self, control: np.ndarray) -> np.ndarray:
        return np.interp(np.linspace(0, 1, self.link.grid.n_channels), np.linspace(0, 1, control.size), control)

    def _channels_to_control(self, profile: np.ndarray, count: int) -> np.ndarray:
        return np.interp(np.linspace(0, 1, count), np.linspace(0, 1, profile.size), profile)

    @staticmethod
    def _soft_min(values: np.ndarray, temperature: float) -> float:
        array = np.asarray(values, dtype=float)
        minimum = float(np.min(array))
        tau = max(float(temperature), 1e-6)
        return float(minimum - tau * np.log(np.mean(np.exp(-(array - minimum) / tau))))

    def _metrics(self, link: LinkModel, profile: np.ndarray, spans: int) -> dict[str, float]:
        result = link.evaluate(dbm_to_w(profile), spans)
        ngmi = np.asarray(getattr(result, "ngmi", []), dtype=float)
        if ngmi.size == 0:
            raise ValueError("Link result must expose a non-empty ngmi array")

        if hasattr(result, "gmi"):
            gmi = np.asarray(result.gmi, dtype=float)
            capacity = summarize_capacity(gmi, ngmi, self.threshold, self.symbol_rate_hz, self.bits, self.overhead)
            air_tbps = float(capacity.air_bps / 1e12)
            thresholded_tbps = float(capacity.thresholded_net_line_bps / 1e12)
            working_fraction = float(capacity.working_fraction)
        else:
            # Compatibility with lightweight fake-link tests used by the
            # original project.  These tests expose soft/hard FEC and AIR
            # aggregates directly instead of full per-channel GMI.
            gmi = np.maximum(ngmi, 0.0) * float(self.bits)
            air_tbps = float(getattr(result, "air_tbps", np.sum(gmi) * self.symbol_rate_hz * 2 / 1e12))
            thresholded_tbps = float(getattr(result, "fec_net_tbps", getattr(result, "soft_fec_net_tbps", 0.0)))
            working_fraction = float(np.mean(ngmi >= self.threshold))

        gsnr = np.asarray(getattr(result, "gsnr_db", np.full(ngmi.shape, np.nan)), dtype=float)
        finite_gsnr = gsnr[np.isfinite(gsnr)]
        metrics: dict[str, float] = {
            "spans": float(spans),
            "air_tbps": air_tbps,
            "thresholded_net_capacity_tbps": thresholded_tbps,
            "fec_net_tbps": float(getattr(result, "fec_net_tbps", thresholded_tbps)),
            "soft_fec_net_tbps": float(getattr(result, "soft_fec_net_tbps", thresholded_tbps)),
            "working_fraction": working_fraction,
            "minimum_ngmi": float(np.min(ngmi)),
            "mean_ngmi": float(np.mean(ngmi)),
            "minimum_gsnr_db": float(np.min(finite_gsnr)) if finite_gsnr.size else np.nan,
            "gsnr_std_db": float(np.std(finite_gsnr)) if finite_gsnr.size else 0.0,
            "mean_squared_ngmi_deficit": float(np.mean(np.maximum(self.threshold - ngmi, 0.0) ** 2)),
            "soft_min_ngmi_margin": self._soft_min(ngmi - self.threshold, float(self.opt.get("softmin_temperature_ngmi", 0.01))),
        }
        bands = np.asarray(getattr(link.grid, "bands", np.array(["ALL"] * ngmi.size)))
        for band in ("S", "C", "L"):
            mask = bands == band
            if not np.any(mask):
                if "ALL" in set(bands.tolist()):
                    mask = np.ones_like(ngmi, dtype=bool)
                else:
                    metrics[f"working_fraction_{band}"] = 0.0
                    metrics[f"minimum_ngmi_{band}"] = -np.inf
                    metrics[f"cvar_ngmi_{band}"] = -np.inf
                    metrics[f"air_tbps_{band}"] = 0.0
                    continue
            band_ngmi = ngmi[mask]
            band_gmi = gmi[mask] if gmi.shape == ngmi.shape else np.maximum(band_ngmi, 0.0) * float(self.bits)
            metrics[f"working_fraction_{band}"] = float(np.mean(band_ngmi >= self.threshold))
            metrics[f"minimum_ngmi_{band}"] = float(np.min(band_ngmi))
            metrics[f"cvar_ngmi_{band}"] = cvar_lower(band_ngmi, float(self.opt.get("robust_cvar_alpha", 0.10)))
            metrics[f"air_tbps_{band}"] = float(np.sum(band_gmi) * self.symbol_rate_hz * 2 / 1e12)
        return metrics

    def _scenario_link_models(self) -> list[LinkModel]:
        if self.robust_batch is None:
            return []
        if self._scenario_links is None:
            links = []
            for values in self.robust_batch.transformed:
                scenario = _scenario_cfg(self.cfg, values)
                links.append(LinkModel(build_grid(scenario["grid"]), scenario))
            self._scenario_links = links
        return self._scenario_links

    def _robust_metrics(self, profile: np.ndarray) -> dict[str, float]:
        links = self._scenario_link_models()
        if not links:
            return {"robust_samples": 0.0, "robust_capacity_cvar_tbps": np.nan, "robust_capacity_worst_tbps": np.nan, "robust_feasibility_probability": np.nan}
        target = int(self.opt["target_spans"])
        capacities, feasible = [], []
        required = self.opt["minimum_band_working_fraction"]
        for link in links:
            metrics = self._metrics(link, profile, target)
            capacities.append(metrics["thresholded_net_capacity_tbps"])
            feasible.append(all(metrics[f"working_fraction_{band}"] >= float(required[band]) for band in ("S", "C", "L")))
        array = np.asarray(capacities)
        return {
            "robust_samples": float(len(array)),
            "robust_capacity_mean_tbps": float(np.mean(array)),
            "robust_capacity_cvar_tbps": cvar_lower(array, float(self.opt.get("robust_cvar_alpha", 0.10))),
            "robust_capacity_worst_tbps": float(np.min(array)),
            "robust_feasibility_probability": float(np.mean(feasible)),
        }

    def objective(self, profile_dbm: np.ndarray, target_total_w: float) -> tuple[float, dict[str, float]]:
        profile = self._project(np.asarray(profile_dbm, dtype=float), target_total_w)
        metrics_by_span = [self._metrics(self.link, profile, spans) for spans in self._evaluation_spans()]
        target = next(item for item in metrics_by_span if int(item["spans"]) == int(self.opt["target_spans"]))
        line_rate = self.link.grid.n_channels * self.symbol_rate_hz * self.bits * 2 / 1e12
        air = np.array([item["air_tbps"] for item in metrics_by_span]) / max(line_rate, 1e-12)
        outage = np.array([item["mean_squared_ngmi_deficit"] for item in metrics_by_span])
        variance = np.array([item["gsnr_std_db"] for item in metrics_by_span])
        required = self.opt["minimum_band_working_fraction"]
        band_deficits = []
        for item in metrics_by_span:
            for band in ("S", "C", "L"):
                band_deficits.append(max(float(required[band]) - item[f"working_fraction_{band}"], 0.0) ** 2)
        robust = self._robust_metrics(profile)
        robust_penalty = 0.0
        if robust["robust_samples"] > 0:
            robust_penalty = -float(self.opt.get("robust_weight", 1.0)) * robust["robust_capacity_cvar_tbps"] / max(line_rate, 1e-12)
            robust_penalty += float(self.opt.get("band_balance_weight", 10.0)) * max(1.0 - robust["robust_feasibility_probability"], 0.0) ** 2
        objective = (
            -float(self.opt.get("air_weight", 1.0)) * float(np.mean(air))
            -float(self.opt.get("margin_weight", 0.5)) * float(np.mean([item["soft_min_ngmi_margin"] for item in metrics_by_span]))
            +float(self.opt.get("outage_weight", 12.0)) * float(np.mean(outage))
            +float(self.opt.get("variance_weight", 0.05)) * float(np.mean(variance**2))
            +float(self.opt.get("band_balance_weight", 10.0)) * float(np.mean(band_deficits))
            +float(self.opt.get("smoothness_weight", 0.02)) * second_difference_energy(profile)
            +robust_penalty
        )
        details = {f"target_{key}": float(value) for key, value in target.items() if key != "spans"}
        details.update({key: float(value) for key, value in robust.items()})
        details["total_power_w"] = total_power_w_from_dbm(profile)
        details["smoothness_energy"] = second_difference_energy(profile)
        details["nominal_band_feasible"] = float(all(target[f"working_fraction_{band}"] >= float(required[band]) for band in ("S", "C", "L")))
        return float(objective), details

    def _coarse_candidates(self, initial: np.ndarray, total_power_w: float) -> list[tuple[str, np.ndarray]]:
        frequency = self.link.grid.frequencies_hz
        normalized = (frequency - np.mean(frequency)) / max(float(np.ptp(frequency)), 1.0)
        candidates = [("initial", initial.copy())]
        for tilt in np.linspace(-12, 12, 25):
            candidates.append((f"linear_tilt_{tilt:+.1f}", self._project(np.mean(initial) + tilt * normalized, total_power_w)))
        bands = getattr(self.link.grid, "bands", None)
        if bands is not None:
            bands = np.asarray(bands)
            for s_offset in (-4, -2, 0, 2, 4):
                for l_offset in (-2, 0, 2):
                    profile = initial.copy()
                    profile[bands == "S"] += s_offset
                    profile[bands == "L"] += l_offset
                    candidates.append((f"band_S{s_offset:+.1f}_L{l_offset:+.1f}", self._project(profile, total_power_w)))
        return candidates

    def _run_restart(self, initial: np.ndarray, total_power_w: float, restart: int, seed: int) -> tuple[np.ndarray, float, pd.DataFrame]:
        rng = np.random.default_rng(seed)
        n_control = min(int(self.opt["control_points"]), initial.size)
        control = self._channels_to_control(initial, n_control)
        if restart:
            control += rng.normal(scale=float(self.opt.get("restart_sigma_db", 0.25)), size=n_control)
        m, v = np.zeros_like(control), np.zeros_like(control)
        beta1, beta2 = float(self.opt.get("adam_beta1", 0.9)), float(self.opt.get("adam_beta2", 0.999))
        best_profile = self._project(self._control_to_channels(control), total_power_w)
        best_objective, _ = self.objective(best_profile, total_power_w)
        rows: list[dict[str, Any]] = []
        patience, stale = int(self.opt.get("early_stopping_patience", 10)), 0
        tolerance = float(self.opt.get("early_stopping_tolerance", 1e-6))
        for iteration in range(1, int(self.opt["iterations"]) + 1):
            delta = rng.choice([-1.0, 1.0], size=control.size)
            perturbation = float(self.opt["spsa_perturbation_db"]) / iteration**0.101
            plus = self._project(self._control_to_channels(control + perturbation * delta), total_power_w)
            minus = self._project(self._control_to_channels(control - perturbation * delta), total_power_w)
            f_plus, _ = self.objective(plus, total_power_w)
            f_minus, _ = self.objective(minus, total_power_w)
            gradient = (f_plus - f_minus) / (2 * perturbation) * delta
            m = beta1 * m + (1 - beta1) * gradient
            v = beta2 * v + (1 - beta2) * gradient**2
            m_hat, v_hat = m / (1 - beta1**iteration), v / (1 - beta2**iteration)
            control -= float(self.opt["learning_rate"]) * m_hat / (np.sqrt(v_hat) + 1e-8)
            profile = self._project(self._control_to_channels(control), total_power_w)
            control = self._channels_to_control(profile, n_control)
            objective, details = self.objective(profile, total_power_w)
            improved = objective < best_objective - tolerance
            if improved:
                best_objective, best_profile, stale = objective, profile.copy(), 0
            else:
                stale += 1
            rows.append({"restart": restart, "seed": seed, "iteration": iteration, "objective": objective, "best_objective": best_objective, "gradient_norm": float(np.linalg.norm(gradient)), "early_stop_counter": stale, **details})
            if stale >= patience:
                break
        return best_profile, float(best_objective), pd.DataFrame(rows)

    def _accept(self, initial: dict[str, float], candidate: dict[str, float], robust_gain: dict[str, float]) -> tuple[bool, str, dict[str, bool]]:
        tolerance = float(self.opt.get("capacity_regression_tolerance_tbps", 1e-9))
        required = self.opt["minimum_band_working_fraction"]
        checks = {
            "capacity_not_worse": candidate["target_thresholded_net_capacity_tbps"] + tolerance >= initial["target_thresholded_net_capacity_tbps"],
            "air_not_worse": candidate["target_air_tbps"] >= initial["target_air_tbps"] * (1 - 1e-9),
            "nominal_gain": candidate["target_thresholded_net_capacity_tbps"] - initial["target_thresholded_net_capacity_tbps"] >= float(self.opt.get("minimum_nominal_gain_tbps", 0.0)),
            "band_feasible": all(candidate[f"target_working_fraction_{band}"] >= float(required[band]) for band in ("S", "C", "L")),
            "robust_ci": robust_gain.get("ci95_low", np.inf) >= float(self.opt.get("minimum_robust_gain_ci_low_tbps", 0.0)),
        }
        passed = all(checks.values())
        failed = [name for name, value in checks.items() if not value]
        return passed, "Accepted: nominal and robust manuscript constraints passed." if passed else f"Rejected: failed {', '.join(failed)}.", checks

    def _paired_robust_gain(self, candidate: np.ndarray, baseline: np.ndarray) -> dict[str, float]:
        links = self._scenario_link_models()
        if not links:
            return {"samples": 0.0, "ci95_low": np.inf, "probability_positive": 1.0}
        target = int(self.opt["target_spans"])
        adaptive, fixed = [], []
        for link in links:
            adaptive.append(self._metrics(link, candidate, target)["thresholded_net_capacity_tbps"])
            fixed.append(self._metrics(link, baseline, target)["thresholded_net_capacity_tbps"])
        summary = paired_gain_summary(adaptive, fixed, bootstrap_samples=1000, seed=int(self.opt["robust_training_seed"]), cvar_alpha=float(self.opt.get("robust_cvar_alpha", 0.1)))
        return {key: float(value) for key, value in summary.__dict__.items()}

    @staticmethod
    def _rank(metrics: dict[str, float], robust_gain: dict[str, float], objective: float) -> tuple[float, ...]:
        return (
            float(metrics["target_thresholded_net_capacity_tbps"]),
            float(robust_gain.get("ci95_low", -np.inf)),
            min(float(metrics[f"target_working_fraction_{band}"]) for band in ("S", "C", "L")),
            float(metrics["target_air_tbps"]),
            -float(objective),
        )

    def optimize(self, initial_profile_dbm: np.ndarray, baseline_name: str | None = None) -> OptimizationResult:
        del baseline_name  # retained for backward-compatible call sites
        initial = np.asarray(initial_profile_dbm, dtype=float)
        if initial.shape != self.link.grid.frequencies_hz.shape:
            raise ValueError("Initial profile shape does not match the grid")
        total = total_power_w_from_dbm(initial)
        initial = self._project(initial, total)
        initial_objective, initial_metrics = self.objective(initial, total)
        candidates: list[tuple[float, int, np.ndarray, dict[str, float], str]] = []
        for name, profile in self._coarse_candidates(initial, total):
            objective, metrics = self.objective(profile, total)
            candidates.append((objective, -1, profile, metrics, name))
        coarse_start = min(candidates, key=lambda item: item[0])[2]
        histories = []
        for restart in range(max(1, int(self.opt["restarts"]))):
            seed = int(self.opt["seed"]) + 104729 * restart
            profile, objective, history = self._run_restart(coarse_start, total, restart, seed)
            _, metrics = self.objective(profile, total)
            candidates.append((objective, restart, profile, metrics, f"spsa_restart_{restart}"))
            histories.append(history)
        accepted = []
        first_rejection = ""
        for objective, restart, profile, metrics, name in candidates:
            if name == "initial":
                continue
            robust_gain = self._paired_robust_gain(profile, initial)
            passed, reason, checks = self._accept(initial_metrics, metrics, robust_gain)
            if passed:
                accepted.append((objective, restart, profile, metrics, name, reason, checks, robust_gain))
            elif not first_rejection:
                first_rejection = f"{name}: {reason}"
        if accepted:
            selected = max(accepted, key=lambda item: self._rank(item[3], item[7], item[0]))
            objective, restart, profile, metrics, name, reason, checks, robust_gain = selected
            objective_gain = float(initial_objective - objective)
            thresholded_gain = float(
                metrics.get("target_thresholded_net_capacity_tbps", 0.0)
                - initial_metrics.get("target_thresholded_net_capacity_tbps", 0.0)
            )
            air_gain = float(
                metrics.get("target_air_tbps", 0.0)
                - initial_metrics.get("target_air_tbps", 0.0)
            )
            profile_changed = not np.allclose(profile, initial, rtol=1e-9, atol=1e-9)
            improvement_floor = max(
                float(self.opt.get("minimum_claim_improvement_tbps", 0.0)),
                float(self.opt.get("minimum_nominal_gain_tbps", 0.0)),
                1e-9,
            )
            improved = bool(
                profile_changed
                and objective_gain > float(self.opt.get("early_stopping_tolerance", 1e-6))
                and (thresholded_gain > improvement_floor or air_gain > improvement_floor)
            )
            if improved:
                reason = f"{name}; {reason}"
            else:
                reason = (
                    f"{name}; accepted constraints were non-regressive, but no material "
                    "improvement over the baseline was detected."
                )
        else:
            objective, restart, profile, metrics = initial_objective, -1, initial, initial_metrics
            improved, reason, checks = False, f"Rejected all candidates. {first_rejection}".strip(), {}
            robust_gain = self._paired_robust_gain(initial, initial)
        history = pd.concat(histories, ignore_index=True) if histories else pd.DataFrame()
        return OptimizationResult(
            initial, profile, float(initial_objective), float(objective), improved, history,
            int(restart), {key: float(value) for key, value in initial_metrics.items()},
            {key: float(value) for key, value in metrics.items()}, reason,
            self.robust_batch.batch_hash if self.robust_batch else None,
            robust_gain, {key: bool(value) for key, value in checks.items()},
        )
