"""Strict machine-checkable publication-readiness gate."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping
import json

import numpy as np
import pandas as pd

from isrs_scl.validation.reproducibility import atomic_write_json, sha256_file, verify_manifest


@dataclass(frozen=True)
class GateCheck:
    name: str
    passed: bool
    severity: str
    evidence: dict[str, Any]
    message: str


@dataclass(frozen=True)
class PublicationGateResult:
    passed: bool
    checks: tuple[GateCheck, ...]

    @property
    def failures(self) -> tuple[GateCheck, ...]:
        return tuple(check for check in self.checks if not check.passed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "publication_ready_numerical_claims": self.passed,
            "failed_required_checks": [check.name for check in self.failures if check.severity == "required"],
            "checks": [check.__dict__ for check in self.checks],
        }


def _check(name: str, passed: bool, message: str, evidence: Mapping[str, Any] | None = None, severity: str = "required") -> GateCheck:
    return GateCheck(name, bool(passed), severity, dict(evidence or {}), message)


def _manifest_check(manifest_path: str | Path | None, run_root: str | Path | None, required_outputs: Iterable[str | Path]) -> GateCheck:
    if manifest_path is None or run_root is None:
        paths = [Path(path) for path in required_outputs]
        missing = [str(path) for path in paths if not path.exists() or path.stat().st_size == 0]
        return _check("current_run_artifact_integrity", not missing, "All required outputs must exist and be non-empty.", {"missing": missing, "legacy_mode": True})
    manifest = Path(manifest_path)
    if not manifest.exists():
        return _check("current_run_artifact_integrity", False, "Current-run manifest is missing.", {"manifest": str(manifest)})
    verified, errors = verify_manifest(manifest)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    root = Path(run_root).resolve()
    listed = {str(item["path"]) for item in payload.get("artifacts", [])}
    missing_listed: list[str] = []
    wrong_root: list[str] = []
    for output in required_outputs:
        path = Path(output).resolve()
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:
            wrong_root.append(str(path))
            continue
        if relative not in listed:
            missing_listed.append(relative)
    passed = verified and not errors and not missing_listed and not wrong_root
    return _check("current_run_artifact_integrity", passed, "Artifacts must belong to and verify against the current run manifest.", {"verification_errors": errors, "unlisted": missing_listed, "outside_run_root": wrong_root, "artifact_count": len(listed)})


def _band_usability(channel_sweep: pd.DataFrame, cfg: Mapping[str, Any]) -> GateCheck:
    target = int(cfg["optimization"]["target_spans"])
    threshold = float(cfg["fec"]["ngmi_target"])
    data = channel_sweep[(channel_sweep["strategy"].astype(str).str.lower() == "adaptive") & (channel_sweep["spans"] == target)].copy()
    fractions: dict[str, float] = {}
    for band in ("S", "C", "L"):
        group = data[data["band"] == band]
        fractions[band] = float(np.mean(group["ngmi"] >= threshold)) if not group.empty else 0.0
    required = cfg["optimization"].get("minimum_band_working_fraction")
    if required is None:
        fallback = float(cfg.get("validation", {}).get("minimum_target_band_working_fraction", 0.80))
        required = {band: fallback for band in ("S", "C", "L")}
    passed = all(fractions[band] >= float(required[band]) for band in ("S", "C", "L"))
    return _check("per_band_target_usability", passed, "Adaptive launch must satisfy the configured working fraction in S, C, and L.", {"target_spans": target, "fractions": fractions, "required": required})


def _waveform_check(waveform: pd.DataFrame, cfg: Mapping[str, Any]) -> GateCheck:
    required_columns = {"band", "operating_point", "acquisition_success", "training_symbols", "payload_symbols", "sample_snr_db", "snr_consistency_error_db", "snr_ci95_low_db", "snr_ci95_high_db", "sample_gmi"}
    missing = sorted(required_columns.difference(waveform.columns))
    if missing or waveform.empty:
        return _check("waveform_validation", False, "Waveform evidence must contain held-out evaluation and confidence intervals.", {"missing_columns": missing})
    intended = waveform[waveform["operating_point"].isin(["high_margin", "near_threshold"])]
    diagnostic = waveform[waveform["operating_point"] == "below_threshold"]
    minimum_repeats = int(cfg["validation"]["minimum_waveform_repeats"])
    repeat_ok = not intended.empty and int(intended.groupby(["band", "operating_point"]).size().min()) >= minimum_repeats
    acquired = bool(intended["acquisition_success"].astype(bool).all())
    disjoint = bool((intended["training_symbols"] > 0).all() and (intended["payload_symbols"] > 0).all())
    finite = bool(np.isfinite(intended[["sample_snr_db", "sample_gmi", "snr_ci95_low_db", "snr_ci95_high_db"]].to_numpy(float)).all())
    tolerance = float(cfg["waveform"]["consistency_tolerance_db"])
    consistent = bool((intended["snr_consistency_error_db"].abs() <= tolerance).all())
    passed = repeat_ok and acquired and disjoint and finite and consistent
    return _check("waveform_validation", passed, "Intended waveform points must acquire and agree with calibrated predictions on held-out payload data.", {
        "minimum_repeats_required": minimum_repeats, "repeat_ok": repeat_ok, "all_intended_acquired": acquired,
        "heldout_split_present": disjoint, "finite": finite, "maximum_error_db": float(intended["snr_consistency_error_db"].abs().max()) if not intended.empty else np.nan,
        "tolerance_db": tolerance, "below_threshold_failures": int((~diagnostic.get("acquisition_success", pd.Series(dtype=bool)).astype(bool)).sum()),
    })


def _external_check(summary: pd.DataFrame | None, cfg: Mapping[str, Any]) -> GateCheck:
    required = bool(cfg["validation"]["require_external_validation"])
    if not required:
        return _check("independent_external_validation", True, "External validation is not required by configuration.", {"required": False})
    if summary is None or summary.empty:
        return _check("independent_external_validation", False, "Independent external validation is missing.", {"required": True})
    if {"requirement", "passed"}.issubset(summary.columns):
        passed = bool(summary["passed"].astype(bool).all())
        failed = summary.loc[~summary["passed"].astype(bool), "requirement"].astype(str).tolist()
    else:
        passed = "passed" in summary and bool(summary["passed"].astype(bool).all())
        failed = [] if passed else ["legacy_external_summary"]
    return _check("independent_external_validation", passed, "External evidence must pass source diversity, coverage, provenance, and metric-specific thresholds.", {"failed_requirements": failed, "rows": len(summary)})


def _robust_check(uncertainty_samples: pd.DataFrame | None, paired_gains: pd.DataFrame | None, cfg: Mapping[str, Any]) -> GateCheck:
    minimum_samples = int(cfg["validation"]["minimum_uncertainty_holdout_samples"])
    minimum_probability = float(cfg["validation"]["minimum_probability_of_improvement"])
    minimum_ci = float(cfg["validation"]["minimum_robust_capacity_gain_tbps"])
    if uncertainty_samples is None or uncertainty_samples.empty or paired_gains is None or paired_gains.empty:
        return _check("robust_adaptive_gain", False, "Independent holdout paired-gain evidence is missing.")
    successful = uncertainty_samples[uncertainty_samples["success"] == 1]
    sample_count = int(successful["sample"].nunique())
    fixed = paired_gains[paired_gains["baseline"].astype(str).str.lower() == "fixed"]
    if fixed.empty:
        return _check("robust_adaptive_gain", False, "Adaptive-versus-fixed paired evidence is missing.", {"holdout_samples": sample_count})
    row = fixed.iloc[0]
    passed = sample_count >= minimum_samples and float(row["ci95_low"]) > minimum_ci and float(row["probability_positive"]) >= minimum_probability
    return _check("robust_adaptive_gain", passed, "Adaptive-minus-fixed gain must be positive with sufficient independent holdout evidence.", {
        "holdout_samples": sample_count, "minimum_samples": minimum_samples,
        "ci95_low_tbps": float(row["ci95_low"]), "minimum_ci_low_tbps": minimum_ci,
        "probability_positive": float(row["probability_positive"]), "minimum_probability": minimum_probability,
    })


def evaluate_publication_gate(
    cfg: Mapping[str, Any],
    *,
    grid_bandwidth_thz: float,
    convergence_passed: bool,
    raman_validation_passed: bool,
    strategy_summary: pd.DataFrame,
    channel_sweep: pd.DataFrame,
    waveform_metrics: pd.DataFrame,
    external_validation_summary: pd.DataFrame | None,
    uncertainty_summary: pd.DataFrame | None,
    uncertainty_samples: pd.DataFrame | None,
    uncertainty_success_fraction: float | None,
    optimizer_accepted: bool,
    output_files: Iterable[str | Path],
    manifest_path: str | Path | None = None,
    run_root: str | Path | None = None,
    optimizer_multiseed: pd.DataFrame | None = None,
    paired_gains: pd.DataFrame | None = None,
    robust_training_hash: str | None = None,
    holdout_hash: str | None = None,
) -> PublicationGateResult:
    checks: list[GateCheck] = []
    metadata = cfg["metadata"]
    sources = metadata.get("calibration_sources", [])
    calibrated = str(metadata.get("calibration_status", "")).upper() == "CALIBRATED" and isinstance(sources, list) and len(sources) > 0
    checks.append(_check("calibrated_parameters", calibrated, "Calibration must be traceable and evidence-driven.", {"status": metadata.get("calibration_status"), "sources": len(sources)}))
    checks.append(_check("raman_analytical_validation", raman_validation_passed, "Raman analytical validation must pass."))
    checks.append(_check("raman_step_convergence", convergence_passed, "Raman step convergence must pass."))
    model = str(cfg["nli"]["primary_model"]).lower()
    limit = float(cfg["validation"]["nominal_closed_form_bandwidth_limit_thz"])
    applicability = not ("semrau" in model and float(grid_bandwidth_thz) > limit)
    checks.append(_check("nli_model_applicability", applicability, "The selected NLI model must be used within validated bandwidth.", {"model": model, "bandwidth_thz": grid_bandwidth_thz, "limit_thz": limit}))
    checks.append(_band_usability(channel_sweep, cfg))
    checks.append(_check("optimizer_candidate_accepted", optimizer_accepted, "The optimizer must accept a candidate based on paper-level and robust constraints."))
    minimum_seeds = int(cfg["validation"]["minimum_optimizer_seeds"])
    seed_count = int(optimizer_multiseed["seed"].nunique()) if optimizer_multiseed is not None and not optimizer_multiseed.empty and "seed" in optimizer_multiseed else 0
    checks.append(_check("optimizer_multiseed_stability", seed_count >= minimum_seeds, "Optimizer conclusions require sufficient independent seeds.", {"observed": seed_count, "required": minimum_seeds}))
    checks.append(_waveform_check(waveform_metrics, cfg))
    checks.append(_external_check(external_validation_summary, cfg))
    uncertainty_required = bool(cfg["validation"]["require_uncertainty_analysis"])
    success = float(uncertainty_success_fraction or 0.0)
    checks.append(_check("uncertainty_numerical_success", not uncertainty_required or (uncertainty_summary is not None and not uncertainty_summary.empty and success >= float(cfg["validation"]["minimum_uncertainty_success_fraction"])), "Uncertainty analysis must complete with an acceptable numerical success rate.", {"success_fraction": success}))
    leakage_free = robust_training_hash is not None and holdout_hash is not None and robust_training_hash != holdout_hash
    checks.append(_check("no_robust_holdout_leakage", leakage_free, "Robust-training and publication holdout batches must be distinct.", {"training_hash": robust_training_hash, "holdout_hash": holdout_hash}))
    checks.append(_robust_check(uncertainty_samples, paired_gains, cfg))
    checks.append(_manifest_check(manifest_path, run_root, output_files))
    failures = [check for check in checks if check.severity == "required" and not check.passed]
    return PublicationGateResult(not failures, tuple(checks))


def write_publication_gate(path: str | Path, result: PublicationGateResult) -> None:
    atomic_write_json(path, result.to_dict())
