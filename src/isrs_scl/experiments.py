"""Deterministic, stage-oriented S+C+L publication study pipeline."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping
import json
import sys
import time

import numpy as np
import pandas as pd

from isrs_scl.dsp.receiver import coherent_receiver, propagate_representative_channel
from isrs_scl.dsp.transmitter import generate_dp16qam
from isrs_scl.fiber.amplification import dbm_to_w
from isrs_scl.link import LinkModel
from isrs_scl.optimization.adaptive_isrs import fixed_preemphasis_profile_dbm
from isrs_scl.optimization.statistics import run_multiseed_optimization
from isrs_scl.system.capacity import line_rate_capacity, summarize_capacity
from isrs_scl.system.grid import build_grid
from isrs_scl.system.parameters import load_config, validate_config
from isrs_scl.validation.analytical_raman import run_undepleted_pump_validation
from isrs_scl.validation.calibration import configuration_can_be_calibrated, load_calibration_files
from isrs_scl.validation.convergence import run_step_convergence
from isrs_scl.validation.external_validation import compare_external_validation, load_external_validation
from isrs_scl.validation.publication_gate import evaluate_publication_gate, write_publication_gate
from isrs_scl.validation.reproducibility import (
    RunPaths, atomic_write_json, build_run_manifest, enforce_git_policy, finalize_manifest,
    mark_stage, prepare_run_directory,
)
from isrs_scl.validation.uncertainty import build_holdout_batch, run_uncertainty_analysis
from isrs_scl.visualization.publication_plots import (
    plot_capacity_reach, plot_constellation, plot_external_validation,
    plot_gsnr_distance_heatmap, plot_gsnr_profiles, plot_launch_profiles,
    plot_optimizer_history, plot_paired_robust_gain, plot_raman_validation,
    plot_sensitivity, plot_spectral_tilt, plot_uncertainty_intervals,
    plot_waveform_metrics_comparison, plot_waveform_power_consistency,
    reset_figure_registry,
)


@dataclass(frozen=True)
class MonotoneReceiverCalibration:
    input_snr_db: np.ndarray
    measured_snr_db: np.ndarray
    measured_ngmi: np.ndarray

    @property
    def minimum_input_snr_db(self) -> float: return float(self.input_snr_db.min())
    @property
    def maximum_input_snr_db(self) -> float: return float(self.input_snr_db.max())
    def _check(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if np.any(array < self.minimum_input_snr_db) or np.any(array > self.maximum_input_snr_db):
            raise ValueError("Receiver calibration extrapolation is forbidden")
        return array
    def predict_snr_db(self, input_snr_db: np.ndarray) -> np.ndarray:
        values = self._check(input_snr_db)
        return np.interp(values, self.input_snr_db, self.measured_snr_db)
    def predict_ngmi(self, input_snr_db: np.ndarray) -> np.ndarray:
        values = self._check(input_snr_db)
        return np.interp(values, self.input_snr_db, self.measured_ngmi)


@dataclass
class StudyContext:
    cfg: dict[str, Any]
    paths: RunPaths
    manifest: dict[str, Any]
    grid: Any | None = None
    link: LinkModel | None = None
    calibration: MonotoneReceiverCalibration | None = None


def _prepare_cfg(cfg: dict[str, Any], grid_mode: str | None, smoke: bool, no_uncertainty: bool) -> dict[str, Any]:
    output = deepcopy(cfg)
    if grid_mode: output["grid"]["mode"] = grid_mode
    if no_uncertainty:
        if output["run"]["mode"] == "publication":
            raise ValueError("Publication mode cannot disable uncertainty analysis")
        output["uncertainty"]["enabled"] = False
        output["validation"]["require_uncertainty_analysis"] = False
    if smoke:
        output["run"]["mode"] = "smoke"
        output["grid"]["mode"] = "paper_240_subset"
        output["grid"]["subset_channels"] = min(int(output["grid"]["subset_channels"]), 41)
        output["raman"]["integration_step_m"] = 1600.0
        output["raman"]["save_step_m"] = 1600.0
        output["fiber"]["max_spans"] = min(int(output["fiber"]["max_spans"]), 2)
        output["optimization"]["target_spans"] = min(int(output["optimization"]["target_spans"]), 2)
        output["optimization"]["evaluation_spans"] = [int(output["optimization"]["target_spans"])]
        output["optimization"]["iterations"] = 1
        output["optimization"]["restarts"] = 1
        output["optimization"]["multi_seed_runs"] = 2
        output["optimization"]["robust_training_samples"] = 4
        output["modulation"]["waveform_symbols"] = 4096
        output["waveform"]["pilot_symbols"] = 512
        output["waveform"]["payload_symbols"] = 2048
        output["waveform"]["repeats_per_band"] = 1
        output["waveform"]["metric_bootstrap_blocks"] = 10
        output["validation"]["minimum_waveform_repeats"] = 1
        output["validation"]["minimum_optimizer_seeds"] = 2
        output["validation"]["minimum_uncertainty_holdout_samples"] = 4
        output["uncertainty"]["holdout_samples"] = 4
        output["fec"]["b2b_snr_sweep_db"] = [8.0, 20.0, 4.0]
        output["output"]["png_dpi"] = min(int(output["output"]["png_dpi"]), 180)
    validate_config(output)
    return output


def _stage(ctx: StudyContext, name: str, function: Callable[[], Any]) -> Any:
    started = time.perf_counter()
    print(f"[ISRS-SCL] START {name}", flush=True)
    try:
        value = function()
        elapsed = time.perf_counter() - started
        mark_stage(ctx.manifest, name, passed=True, details={"elapsed_seconds": round(elapsed, 3)})
        print(f"[ISRS-SCL] DONE  {name} in {elapsed:.1f}s", flush=True)
        return value
    except Exception as exc:
        elapsed = time.perf_counter() - started
        mark_stage(ctx.manifest, name, passed=False, details={"error": f"{type(exc).__name__}: {exc}", "elapsed_seconds": round(elapsed, 3)})
        atomic_write_json(ctx.paths.metadata / "RUN_MANIFEST.failed.json", {key: value for key, value in ctx.manifest.items() if not key.startswith("_runtime_")})
        print(f"[ISRS-SCL] FAILED {name} after {elapsed:.1f}s: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise


def _profile_frame(grid: Any, profiles: Mapping[str, np.ndarray]) -> pd.DataFrame:
    return pd.concat([
        pd.DataFrame({
            "strategy": strategy, "channel": np.arange(grid.n_channels), "band": grid.bands,
            "frequency_thz": grid.frequencies_hz / 1e12, "wavelength_nm": grid.wavelengths_nm,
            "launch_power_dbm": np.asarray(profile, dtype=float),
        }) for strategy, profile in profiles.items()
    ], ignore_index=True)


def _strategy_sweeps(link: LinkModel, profiles: Mapping[str, np.ndarray], cfg: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    channel_tables, summaries, band_tables = [], [], []
    symbol_rate = float(cfg["modulation"]["symbol_rate_gbaud"]) * 1e9
    bits, overhead, threshold = int(cfg["modulation"]["bits_per_symbol_per_pol"]), float(cfg["fec"]["overhead_fraction"]), float(cfg["fec"]["ngmi_target"])
    for strategy, profile in profiles.items():
        for result in link.sweep_spans(dbm_to_w(profile)):
            frame = result.to_frame(link.grid); frame.insert(0, "strategy", strategy); frame.insert(1, "spans", result.n_spans); channel_tables.append(frame)
            capacity = summarize_capacity(result.gmi, result.ngmi, threshold, symbol_rate, bits, overhead)
            summaries.append({
                "strategy": strategy, "spans": result.n_spans, "distance_km": result.distance_km,
                "minimum_gsnr_db": float(np.min(result.gsnr_db)), "mean_gsnr_db": float(np.mean(result.gsnr_db)),
                "gsnr_std_db": float(np.std(result.gsnr_db)), "metric_basis": result.metric_basis, **capacity.as_dict(),
            })
            band = result.band_summary(link.grid, threshold); band.insert(0, "strategy", strategy); band.insert(1, "spans", result.n_spans); band.insert(2, "distance_km", result.distance_km); band_tables.append(band)
    return pd.concat(channel_tables, ignore_index=True), pd.DataFrame(summaries), pd.concat(band_tables, ignore_index=True)


def _b2b_points(cfg: Mapping[str, Any], predicted_range: tuple[float, float] | None) -> np.ndarray:
    start, stop, step = map(float, cfg["fec"]["b2b_snr_sweep_db"])
    if predicted_range is not None:
        # The calibrated LinkModel deliberately rejects extrapolation.  Therefore
        # every mode, including smoke, must cover the physical GSNR range observed
        # in the pre-calibration nominal sweep.  Smoke still uses a sparse set of
        # points, but it may not shrink the range to 8..20 dB when the link model
        # produces lower or higher GSNR values.
        low, high = map(float, predicted_range)
        if np.isfinite(low) and np.isfinite(high):
            start = min(start, low - 2.0)
            stop = max(stop, high + 2.0)
    if stop <= start:
        raise ValueError(f"Invalid B2B SNR sweep range: start={start:g}, stop={stop:g}")
    if cfg.get("run", {}).get("mode") == "smoke":
        # Smoke runs are for plumbing and API validation, not final receiver
        # calibration.  Use few points but always include the dynamic endpoints so
        # nominal_sweeps cannot request receiver-calibration extrapolation.
        candidates = [start, stop]
        for value in (8.0, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0):
            if start < value < stop:
                candidates.append(value)
        return np.unique(np.round(np.array(sorted(candidates), dtype=float), 6))
    coarse = np.arange(start, stop + step / 2, step)
    # Dense region around likely 16-QAM FEC transition.
    dense = np.arange(max(start, 6.0), min(stop, 18.0) + 0.25, 0.25)
    return np.unique(np.round(np.concatenate([coarse, dense]), 6))


def run_b2b_calibration(cfg: Mapping[str, Any], output_dir: Path, predicted_range: tuple[float, float] | None = None) -> tuple[pd.DataFrame, MonotoneReceiverCalibration]:
    points = _b2b_points(cfg, predicted_range)
    configured_repeats = int(cfg["waveform"].get("repeats_per_band", 3))
    repeats = max(configured_repeats, 1 if cfg.get("run", {}).get("mode") == "smoke" else 3)
    rows = []
    n_symbols = int(cfg["waveform"]["pilot_symbols"] + cfg["waveform"]["payload_symbols"])
    symbol_rate = float(cfg["modulation"]["symbol_rate_gbaud"]) * 1e9
    for point_index, input_snr_db in enumerate(points):
        for repeat in range(repeats):
            seed = int(cfg["metadata"]["random_seed"]) + 10_000 + point_index * 100 + repeat
            channel_power = 1e-3
            tx = generate_dp16qam(n_symbols, int(cfg["modulation"]["oversampling"]), float(cfg["modulation"]["roll_off"]), int(cfg["modulation"]["rrc_span_symbols"]), seed, channel_power, pilot_symbols=int(cfg["waveform"]["pilot_symbols"]), pilot_spacing=int(cfg["waveform"]["pilot_spacing"]), symbol_rate_hz=symbol_rate)
            noise = channel_power / 10.0 ** (float(input_snr_db) / 10.0)
            channel = propagate_representative_channel(tx, symbol_rate, 1550.0, 0.0, 0.0, 1, noise, 0.0, 0.0, 0.0, False, False, seed + 100_000)
            rx = coherent_receiver(channel, tx, 0.0, int(cfg["waveform"]["cma_taps"]), float(cfg["waveform"]["cma_step_size"]), int(cfg["waveform"]["cma_training_symbols"]), int(cfg["waveform"]["bps_trial_phases"]), int(cfg["waveform"]["bps_block_symbols"]), equalizer_mode=str(cfg["waveform"]["equalizer_mode"]), equalizer_ridge=float(cfg["waveform"]["equalizer_ridge"]), bootstrap_samples=int(cfg["waveform"]["metric_bootstrap_blocks"]), bootstrap_block_symbols=int(cfg["waveform"]["bootstrap_block_symbols"]), bootstrap_seed=seed + 200_000, carrier_recovery_mode=str(cfg["waveform"]["carrier_recovery_mode"]), bps_overlap=float(cfg["waveform"]["bps_overlap"]), bps_smoothing_blocks=int(cfg["waveform"]["bps_smoothing_blocks"]), minimum_reliability=float(cfg["waveform"]["minimum_reliability"]))
            valid_metrics = [item for item in rx.metrics_per_pol if "snr_db" in item]
            rows.append({
                "input_snr_db": input_snr_db, "repeat": repeat, "seed": seed,
                "acquisition_success": rx.acquisition_success,
                "measured_snr_db": float(np.mean([item["snr_db"] for item in valid_metrics])) if valid_metrics else np.nan,
                "ngmi": float(np.mean([item["ngmi"] for item in valid_metrics])) if valid_metrics else np.nan,
                "gmi": float(np.mean([item["gmi_bits_per_2d_symbol_per_pol"] for item in valid_metrics])) if valid_metrics else np.nan,
                "ber": float(np.mean([item["ber"] for item in valid_metrics])) if valid_metrics else np.nan,
                "failure_reason": rx.failure_reason,
            })
    frame = pd.DataFrame(rows); frame.to_csv(output_dir / "b2b_calibration_repeats.csv", index=False)
    successful = frame[frame["acquisition_success"].astype(bool)].dropna(subset=["measured_snr_db", "ngmi"])
    grouped = successful.groupby("input_snr_db", sort=True).agg(
        repeats=("repeat", "count"), measured_snr_db=("measured_snr_db", "mean"), measured_snr_std_db=("measured_snr_db", "std"),
        ngmi=("ngmi", "mean"), ngmi_std=("ngmi", "std"), ber=("ber", "mean"), acquisition_fraction=("acquisition_success", "mean"),
    ).reset_index()
    if len(grouped) < 4:
        raise RuntimeError("B2B calibration has fewer than four successful SNR points")
    grouped["measured_snr_db"] = np.maximum.accumulate(grouped["measured_snr_db"])
    grouped["ngmi"] = np.maximum.accumulate(grouped["ngmi"])
    grouped.to_csv(output_dir / "b2b_calibration.csv", index=False)
    calibration = MonotoneReceiverCalibration(grouped["input_snr_db"].to_numpy(float), grouped["measured_snr_db"].to_numpy(float), grouped["ngmi"].to_numpy(float))
    return frame, calibration


def _waveform_operating_points(link: LinkModel, result: Any, cfg: Mapping[str, Any]) -> list[tuple[str, int]]:
    """Choose representative waveform channels without assuming all S/C/L bands exist.

    Smoke runs intentionally use a reduced wavelength subset for speed.  That
    subset can contain only C-band channels, so blindly calling ``argmax`` on an
    empty S- or L-band slice aborts the run.  Publication/full runs still produce
    S/C/L operating points when the grid contains them; reduced runs skip absent
    bands and record whatever bands are actually present.
    """
    points: list[tuple[str, int]] = []
    acquisition_threshold = float(cfg["waveform"]["acquisition_snr_threshold_db"])
    bands = np.asarray(link.grid.bands, dtype=str)
    gsnr_db = np.asarray(result.gsnr_db, dtype=float)

    if bands.shape[0] != gsnr_db.shape[0]:
        raise ValueError(
            f"Grid/result length mismatch in waveform selection: "
            f"{bands.shape[0]} bands for {gsnr_db.shape[0]} GSNR values"
        )

    missing_bands: list[str] = []
    for band in ("S", "C", "L"):
        indices = np.flatnonzero(np.char.upper(bands.astype(str)) == band)
        if indices.size:
            finite = np.isfinite(gsnr_db[indices])
            indices = indices[finite]
        if indices.size == 0:
            missing_bands.append(band)
            continue

        values = gsnr_db[indices]
        high = int(indices[np.argmax(values)])
        near = int(indices[np.argmin(np.abs(values - max(acquisition_threshold, 10.0)))])
        low = int(indices[np.argmin(values)])
        points.extend([("high_margin", high), ("near_threshold", near), ("below_threshold", low)])

    if not points:
        finite_indices = np.flatnonzero(np.isfinite(gsnr_db))
        if finite_indices.size == 0:
            raise ValueError("No finite GSNR values are available for waveform validation")
        values = gsnr_db[finite_indices]
        high = int(finite_indices[np.argmax(values)])
        near = int(finite_indices[np.argmin(np.abs(values - max(acquisition_threshold, 10.0)))])
        low = int(finite_indices[np.argmin(values)])
        points.extend([("high_margin", high), ("near_threshold", near), ("below_threshold", low)])

    if missing_bands and str(cfg.get("run", {}).get("mode", "debug")) != "publication":
        print(
            "[ISRS-SCL] waveform_validation: reduced grid does not contain "
            f"{','.join(missing_bands)} band channel(s); validating present band(s) only.",
            flush=True,
        )
    return points


def run_waveform_validation(link: LinkModel, profile_dbm: np.ndarray, cfg: Mapping[str, Any], output_dir: Path, figure_dir: Path, dpi: int, calibration: MonotoneReceiverCalibration) -> pd.DataFrame:
    target_spans = int(cfg["optimization"]["target_spans"])
    power = link.evaluate(dbm_to_w(profile_dbm), target_spans)
    symbol_rate = float(cfg["modulation"]["symbol_rate_gbaud"]) * 1e9
    rows = []
    for point_index, (operating_point, channel_index) in enumerate(_waveform_operating_points(link, power, cfg)):
        band = str(link.grid.bands[channel_index])
        for repeat in range(int(cfg["waveform"]["repeats_per_band"])):
            seed = int(cfg["metadata"]["random_seed"]) + 20_000 + point_index * 1000 + repeat
            n_symbols = int(cfg["waveform"]["pilot_symbols"] + cfg["waveform"]["payload_symbols"])
            tx = generate_dp16qam(n_symbols, int(cfg["modulation"]["oversampling"]), float(cfg["modulation"]["roll_off"]), int(cfg["modulation"]["rrc_span_symbols"]), seed, float(power.launch_power_w[channel_index]), pilot_symbols=int(cfg["waveform"]["pilot_symbols"]), pilot_spacing=int(cfg["waveform"]["pilot_spacing"]), symbol_rate_hz=symbol_rate)
            channel = propagate_representative_channel(tx, symbol_rate, float(link.grid.wavelengths_nm[channel_index]), float(link.span_model.beta2[channel_index]), float(cfg["fiber"]["span_length_km"]) * 1000, target_spans, float(power.noise_budget.ase_receiver_w[channel_index] / target_spans), float(power.span.nli.nli_power_w_per_span[channel_index]), float(cfg["fiber"]["pmd_ps_sqrt_km"]), float(cfg["waveform"]["laser_linewidth_hz"]), bool(cfg["waveform"]["apply_pmd"]), bool(cfg["waveform"]["apply_phase_noise"]), seed + 500_000, nli_coherence_epsilon=float(cfg["nli"]["coherence_epsilon"]), carrier_frequency_offset_hz=float(cfg["waveform"]["carrier_frequency_offset_hz"]))
            rx = coherent_receiver(channel, tx, float(link.span_model.beta2[channel_index]), int(cfg["waveform"]["cma_taps"]), float(cfg["waveform"]["cma_step_size"]), int(cfg["waveform"]["cma_training_symbols"]), int(cfg["waveform"]["bps_trial_phases"]), int(cfg["waveform"]["bps_block_symbols"]), equalizer_mode=str(cfg["waveform"]["equalizer_mode"]), equalizer_ridge=float(cfg["waveform"]["equalizer_ridge"]), bootstrap_samples=int(cfg["waveform"]["metric_bootstrap_blocks"]), bootstrap_block_symbols=int(cfg["waveform"]["bootstrap_block_symbols"]), bootstrap_seed=seed + 700_000, carrier_recovery_mode=str(cfg["waveform"]["carrier_recovery_mode"]), bps_overlap=float(cfg["waveform"]["bps_overlap"]), bps_smoothing_blocks=int(cfg["waveform"]["bps_smoothing_blocks"]), minimum_reliability=float(cfg["waveform"]["minimum_reliability"]))
            metrics = [item for item in rx.metrics_per_pol if "snr_db" in item]
            analytical = float(power.gsnr_db[channel_index])
            expected = float(calibration.predict_snr_db(np.array([analytical]))[0]) if calibration.minimum_input_snr_db <= analytical <= calibration.maximum_input_snr_db else np.nan
            row = {
                "band": band, "operating_point": operating_point, "repeat": repeat, "seed": seed,
                "channel": channel_index, "wavelength_nm": float(link.grid.wavelengths_nm[channel_index]),
                "spans": target_spans, "distance_km": float(power.distance_km), "analytical_gsnr_db": analytical,
                "b2b_expected_receiver_snr_db": expected, "acquisition_success": rx.acquisition_success,
                "cycle_slips": rx.cycle_slips, "training_symbols": rx.training_symbols, "payload_symbols": rx.payload_symbols,
                "failure_reason": rx.failure_reason,
                "sample_snr_db": float(np.mean([item["snr_db"] for item in metrics])) if metrics else np.nan,
                "snr_ci95_low_db": float(np.mean([item["snr_ci95_low_db"] for item in metrics])) if metrics else np.nan,
                "snr_ci95_high_db": float(np.mean([item["snr_ci95_high_db"] for item in metrics])) if metrics else np.nan,
                "sample_ber": float(np.mean([item["ber"] for item in metrics])) if metrics else np.nan,
                "sample_gmi": float(np.mean([item["gmi_bits_per_2d_symbol_per_pol"] for item in metrics])) if metrics else np.nan,
                "sample_ngmi": float(np.mean([item["ngmi"] for item in metrics])) if metrics else np.nan,
            }
            row["snr_consistency_error_db"] = row["sample_snr_db"] - expected if np.isfinite(row["sample_snr_db"]) and np.isfinite(expected) else np.nan
            rows.append(row)
            if repeat == 0 and rx.acquisition_success and metrics:
                plot_constellation(rx.aligned_tx_symbols[0], rx.recovered_symbols[0], figure_dir / f"waveform_{band}_{operating_point}", dpi, title=f"{band} band — {operating_point}", annotation=f"GSNR={analytical:.2f} dB\nSNR={row['sample_snr_db']:.2f} dB")
    frame = pd.DataFrame(rows); frame.to_csv(output_dir / "waveform_metrics.csv", index=False)
    return frame


def run_publication_study(config_path: str | Path, grid_mode: str | None = None, smoke: bool = False, *, strict: bool = False, no_uncertainty: bool = False) -> dict[str, Any]:
    loaded = load_config(config_path)
    cfg = _prepare_cfg(loaded, grid_mode, smoke, no_uncertainty)
    run_id = str(cfg["run"].get("run_id") or f"{cfg['run']['mode']}-{cfg['metadata']['random_seed']}")
    cfg["run"]["run_id"] = run_id
    print(f"[ISRS-SCL] Run mode={cfg['run']['mode']} run_id={run_id}", flush=True)
    paths = prepare_run_directory(cfg["run"]["output_root"], run_id, overwrite=bool(cfg["run"].get("overwrite", False)))
    cfg["output"]["directory"], cfg["output"]["figure_directory"] = str(paths.results), str(paths.figures)
    validate_config(cfg, base_dir=Path(config_path).parent)
    Path(paths.metadata / "resolved_config.yaml").write_text(__import__("yaml").safe_dump(cfg, sort_keys=False), encoding="utf-8")
    calibration_files = [item.get("file") for item in cfg["metadata"].get("calibration_sources", []) if isinstance(item, Mapping) and item.get("file")]
    manifest = build_run_manifest(cfg, run_root=paths.root, repository_root=Path.cwd(), input_files=[cfg["validation"]["external_reference_csv"]], calibration_files=calibration_files)
    enforce_git_policy(manifest["git"], publication=cfg["run"]["mode"] == "publication", allow_untracked=bool(cfg["run"].get("allow_untracked_provenance", False)))
    atomic_write_json(paths.metadata / "RUN_MANIFEST.initial.json", {key: value for key, value in manifest.items() if not key.startswith("_runtime_")})
    ctx = StudyContext(cfg, paths, manifest)
    reset_figure_registry()

    def calibration_stage() -> Any:
        if not calibration_files: return None
        bundle = load_calibration_files(calibration_files)
        bundle.provenance.to_csv(paths.metadata / "calibration_provenance.csv", index=False)
        passed, reasons = configuration_can_be_calibrated(cfg, bundle)
        if cfg["run"]["mode"] == "publication" and not passed: raise RuntimeError("Calibration evidence failed: " + " | ".join(reasons))
        return bundle
    _stage(ctx, "calibration", calibration_stage)

    def raman_stage() -> tuple[pd.DataFrame, bool, pd.DataFrame, bool]:
        frame, max_error = run_undepleted_pump_validation(step_m=float(cfg["raman"]["integration_step_m"])); frame.to_csv(paths.results / "raman_analytical_validation.csv", index=False)
        plot_raman_validation(frame, paths.figures / "01_raman_validation", int(cfg["output"]["png_dpi"]), metadata={"run_id": run_id})
        base = float(cfg["raman"]["integration_step_m"]); convergence = run_step_convergence(build_grid(cfg["grid"]), cfg, steps_m=(base, base / 2, base / 4)); convergence.to_csv(paths.results / "raman_step_convergence.csv", index=False)
        return frame, max_error <= float(cfg["validation"]["raman_max_relative_error_limit"]), convergence, bool(convergence.loc[convergence["step_m"] == base, "max_abs_power_error_db"].iloc[0] <= float(cfg["validation"]["convergence_relative_tolerance"]))
    _, raman_passed, _, convergence_passed = _stage(ctx, "raman_validation", raman_stage)

    ctx.grid = build_grid(cfg["grid"]); ctx.grid.to_frame().to_csv(paths.results / "channel_grid.csv", index=False)
    uncalibrated_link = LinkModel(ctx.grid, cfg)
    flat = np.full(ctx.grid.n_channels, float(cfg["launch"]["flat_power_dbm_per_channel"]))
    fixed = fixed_preemphasis_profile_dbm(ctx.grid.frequencies_hz, float(cfg["launch"]["flat_power_dbm_per_channel"]), float(cfg["launch"]["fixed_preemphasis_s_to_l_db"]), float(cfg["launch"]["min_power_dbm_per_channel"]), float(cfg["launch"]["max_power_dbm_per_channel"]))

    def optimization_stage() -> Any:
        result = run_multiseed_optimization(uncalibrated_link, cfg, fixed)
        result.run_summary.to_csv(paths.results / "optimizer_multiseed.csv", index=False); result.history.to_csv(paths.results / "optimizer_history.csv", index=False)
        atomic_write_json(paths.results / "optimizer_confidence.json", result.confidence)
        atomic_write_json(paths.results / "optimizer_result.json", {"accepted": result.best_result.improved, "acceptance_reason": result.best_result.acceptance_reason, "initial_objective": result.best_result.initial_objective, "optimized_objective": result.best_result.optimized_objective, "robust_training_hash": result.best_result.robust_training_hash, "robust_summary": result.best_result.robust_summary, "feasibility": result.best_result.feasibility})
        return result
    optimization = _stage(ctx, "robust_optimization", optimization_stage)
    adaptive = optimization.best_result.optimized_profile_dbm
    profiles = {"flat": flat, "fixed": fixed, "adaptive": adaptive}
    profile_table = _profile_frame(ctx.grid, profiles); profile_table.to_csv(paths.results / "launch_profiles.csv", index=False)
    plot_launch_profiles(profile_table, paths.figures / "03_launch_profiles", int(cfg["output"]["png_dpi"]), metadata={"run_id": run_id})
    if not optimization.history.empty: plot_optimizer_history(optimization.history, paths.figures / "09_optimizer_convergence", int(cfg["output"]["png_dpi"]), multiseed=optimization.run_summary, metadata={"run_id": run_id})

    raw_channel, raw_summary, raw_band = _strategy_sweeps(uncalibrated_link, profiles, cfg)
    predicted_range = (float(raw_channel["gsnr_db"].min()), float(raw_channel["gsnr_db"].max()))
    _, receiver_calibration = _stage(ctx, "b2b_receiver_calibration", lambda: run_b2b_calibration(cfg, paths.results, predicted_range))
    ctx.calibration = receiver_calibration; ctx.link = LinkModel(ctx.grid, cfg, receiver_calibration=receiver_calibration)
    channel_sweep, strategy_summary, band_summary = _stage(ctx, "nominal_sweeps", lambda: _strategy_sweeps(ctx.link, profiles, cfg))
    channel_sweep.to_csv(paths.results / "channel_performance.csv", index=False); strategy_summary.to_csv(paths.results / "strategy_summary.csv", index=False); band_summary.to_csv(paths.results / "band_summary.csv", index=False)
    plot_capacity_reach(strategy_summary, paths.figures / "08_capacity_reach", int(cfg["output"]["png_dpi"]), metadata={"run_id": run_id})
    plot_gsnr_distance_heatmap(channel_sweep, paths.figures / "05_gsnr_distance_heatmap", int(cfg["output"]["png_dpi"]), metadata={"run_id": run_id})
    target = channel_sweep[channel_sweep["spans"] == int(cfg["optimization"]["target_spans"])]
    plot_gsnr_profiles(target, paths.figures / "04_gsnr_vs_wavelength", int(cfg["output"]["png_dpi"]), metadata={"run_id": run_id})

    def tilt_stage() -> pd.DataFrame:
        span = ctx.link.span_model.evaluate(dbm_to_w(adaptive)); passive = span.passive_result.output_powers_w
        frame = pd.DataFrame({"channel": np.arange(ctx.grid.n_channels), "band": ctx.grid.bands, "wavelength_nm": ctx.grid.wavelengths_nm, "launch_power_dbm": adaptive, "span_output_power_dbm": 10 * np.log10(span.raman_result.output_powers_w / 1e-3), "no_isrs_output_power_dbm": 10 * np.log10(passive / 1e-3), "differential_isrs_db": 10 * np.log10(span.raman_result.output_powers_w / np.maximum(passive, 1e-30)), "amplifier_residual_db": span.amplifier.residual_db})
        frame.to_csv(paths.results / "span_spectral_tilt.csv", index=False); plot_spectral_tilt(frame, paths.figures / "02_isrs_spectral_tilt", int(cfg["output"]["png_dpi"]), metadata={"run_id": run_id}); return frame
    _stage(ctx, "spectral_tilt", tilt_stage)

    waveform = _stage(ctx, "waveform_validation", lambda: run_waveform_validation(ctx.link, adaptive, cfg, paths.results, paths.figures, int(cfg["output"]["png_dpi"]), receiver_calibration))
    plot_waveform_metrics_comparison(waveform, paths.figures / "06_waveform_metrics", int(cfg["output"]["png_dpi"]), metadata={"run_id": run_id}); plot_waveform_power_consistency(waveform, paths.figures / "07_waveform_power_consistency", int(cfg["output"]["png_dpi"]), metadata={"run_id": run_id})

    external_requirements = None

    def _external_thresholds() -> dict[str, Any]:
        return {
            "minimum_external_coverage": cfg["validation"]["external_minimum_coverage_fraction"],
            "minimum_sources": cfg["validation"]["external_minimum_sources"],
            "minimum_source_types": cfg["validation"]["external_minimum_source_types"],
            "minimum_wavelengths_per_band": cfg["validation"]["external_minimum_wavelengths_per_band"],
            "minimum_span_counts": cfg["validation"]["external_minimum_span_counts"],
            "maximum_gsnr_db_rmse": cfg["validation"]["external_max_gsnr_rmse_db"],
            "maximum_gsnr_db_absolute_bias": cfg["validation"]["external_max_gsnr_bias_db"],
            "maximum_nli_relative_rmse": cfg["validation"]["external_max_nli_relative_rmse"],
            "required_external_metrics": ("gsnr_db",),
        }

    def _write_external_skip(reason: str) -> None:
        nonlocal external_requirements
        external_requirements = pd.DataFrame([
            {
                "requirement": "external_validation_data",
                "passed": False,
                "value": 0.0,
                "threshold": 1.0,
                "reason": reason,
                "run_mode": cfg["run"]["mode"],
            }
        ])
        pd.DataFrame().to_csv(paths.results / "external_validation_comparisons.csv", index=False)
        pd.DataFrame().to_csv(paths.results / "external_validation_summary.csv", index=False)
        external_requirements.to_csv(paths.results / "external_validation_requirements.csv", index=False)
        mark_stage(ctx.manifest, "external_validation", passed=False, details={"skipped": True, "reason": reason})
        print(
            f"[ISRS-SCL] SKIP external_validation: {reason}. "
            "Smoke/debug runs may continue; publication mode requires real independent validation data.",
            flush=True,
        )

    def external_stage(external: pd.DataFrame | None = None) -> Any:
        nonlocal external_requirements
        reference = external if external is not None else load_external_validation(cfg["validation"]["external_reference_csv"])
        result = compare_external_validation(
            channel_sweep,
            reference,
            wavelength_tolerance_nm=float(cfg["validation"]["external_max_wavelength_error_nm"]),
            thresholds=_external_thresholds(),
            allow_interpolation=bool(cfg["validation"]["external_allow_interpolation"]),
        )
        result.comparisons.to_csv(paths.results / "external_validation_comparisons.csv", index=False)
        result.summary.to_csv(paths.results / "external_validation_summary.csv", index=False)
        result.requirements.to_csv(paths.results / "external_validation_requirements.csv", index=False)
        external_requirements = result.requirements
        plot_external_validation(result.comparisons, paths.figures / "12_external_validation", int(cfg["output"]["png_dpi"]), metadata={"run_id": run_id})
        return result

    external_result = None
    if cfg["run"]["mode"] == "publication":
        external_result = _stage(ctx, "external_validation", external_stage)
    else:
        try:
            external_table = load_external_validation(cfg["validation"]["external_reference_csv"])
        except (FileNotFoundError, ValueError) as exc:
            _write_external_skip(str(exc))
        else:
            external_result = _stage(ctx, "external_validation", lambda: external_stage(external_table))

    uncertainty = None
    if bool(cfg["uncertainty"]["enabled"]):
        def uncertainty_stage() -> Any:
            holdout = build_holdout_batch(cfg)
            result = run_uncertainty_analysis(cfg, profiles, int(cfg["optimization"]["target_spans"]), batch=holdout)
            result.samples.to_csv(paths.results / "uncertainty_holdout_samples.csv", index=False); result.summary.to_csv(paths.results / "uncertainty_summary.csv", index=False); result.sensitivity.to_csv(paths.results / "uncertainty_sensitivity.csv", index=False); result.convergence.to_csv(paths.results / "uncertainty_convergence.csv", index=False); result.paired_gains.to_csv(paths.results / "paired_robust_gains.csv", index=False)
            plot_uncertainty_intervals(result.summary, paths.figures / "10_uncertainty_intervals", int(cfg["output"]["png_dpi"]), metadata={"run_id": run_id}); plot_sensitivity(result.sensitivity[result.sensitivity["metric"] == "thresholded_net_capacity_tbps"], paths.figures / "11_global_sensitivity", int(cfg["output"]["png_dpi"]), metadata={"run_id": run_id})
            if not result.paired_gains.empty: plot_paired_robust_gain(result.paired_gains, paths.figures / "13_paired_robust_gain", int(cfg["output"]["png_dpi"]), metadata={"run_id": run_id})
            return result
        uncertainty = _stage(ctx, "uncertainty_holdout", uncertainty_stage)

    required_outputs = [paths.metadata / "resolved_config.yaml", paths.results / "channel_grid.csv", paths.results / "raman_analytical_validation.csv", paths.results / "raman_step_convergence.csv", paths.results / "launch_profiles.csv", paths.results / "optimizer_multiseed.csv", paths.results / "channel_performance.csv", paths.results / "strategy_summary.csv", paths.results / "waveform_metrics.csv", paths.results / "b2b_calibration.csv"]
    if external_result is not None: required_outputs += [paths.results / "external_validation_comparisons.csv", paths.results / "external_validation_requirements.csv"]
    if uncertainty is not None: required_outputs += [paths.results / "uncertainty_holdout_samples.csv", paths.results / "paired_robust_gains.csv"]
    manifest_path = paths.metadata / "RUN_MANIFEST.json"; finalize_manifest(ctx.manifest, paths.root, manifest_path)
    gate = evaluate_publication_gate(cfg, grid_bandwidth_thz=ctx.grid.bandwidth_hz / 1e12, convergence_passed=convergence_passed, raman_validation_passed=raman_passed, strategy_summary=strategy_summary, channel_sweep=channel_sweep, waveform_metrics=waveform, external_validation_summary=external_requirements, uncertainty_summary=None if uncertainty is None else uncertainty.summary, uncertainty_samples=None if uncertainty is None else uncertainty.samples, uncertainty_success_fraction=None if uncertainty is None else uncertainty.successful_fraction, optimizer_accepted=optimization.best_result.improved, output_files=required_outputs, manifest_path=manifest_path, run_root=paths.root, optimizer_multiseed=optimization.run_summary, paired_gains=None if uncertainty is None else uncertainty.paired_gains, robust_training_hash=optimization.best_result.robust_training_hash, holdout_hash=None if uncertainty is None else uncertainty.batch_hash)
    write_publication_gate(paths.metadata / "VALIDATION_STATUS.json", gate); finalize_manifest(ctx.manifest, paths.root, manifest_path)
    line_rate = line_rate_capacity(ctx.grid.n_channels, float(cfg["modulation"]["symbol_rate_gbaud"]) * 1e9, int(cfg["modulation"]["bits_per_symbol_per_pol"]), float(cfg["fec"]["overhead_fraction"]))
    publication_gaps = [check.name for check in gate.failures]
    summary = {
        "study_title": cfg["metadata"]["study_title"],
        "run_id": run_id,
        "run_directory": str(paths.root),
        "run_mode": cfg["run"]["mode"],
        "calibration_status": cfg["metadata"]["calibration_status"],
        "channels": ctx.grid.n_channels,
        "bandwidth_thz": ctx.grid.bandwidth_hz / 1e12,
        "gross_line_rate_tbps": line_rate.gross_tbps,
        "net_line_rate_tbps": line_rate.net_tbps,
        "optimizer_candidate_accepted": optimization.best_result.improved,
        "publication_ready_numerical_claims": gate.passed if cfg["run"]["mode"] == "publication" else False,
        "publication_gate_evaluated": True,
        "publication_gaps_if_submitted": publication_gaps,
        "failed_publication_checks": publication_gaps if cfg["run"]["mode"] == "publication" else [],
        "smoke_run_completed": cfg["run"]["mode"] == "smoke",
        "debug_run_completed": cfg["run"]["mode"] == "debug",
        "manifest_path": str(manifest_path),
        "validation_status_path": str(paths.metadata / "VALIDATION_STATUS.json"),
    }
    atomic_write_json(paths.metadata / "summary.json", summary); finalize_manifest(ctx.manifest, paths.root, manifest_path)
    if strict and not gate.passed: raise RuntimeError("Strict publication gate failed: " + ", ".join(publication_gaps))
    return summary
