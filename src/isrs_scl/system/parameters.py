"""Strict, versioned configuration loading for publication and debug studies.

The module intentionally uses only the standard library and PyYAML.  It accepts
legacy project configurations through :func:`migrate_config`, but validation is
strict after migration: unknown keys, ambiguous units, unsafe publication
settings, placeholder validation data, and training/holdout seed reuse are
reported with dotted-path errors.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping
import json

import yaml

SCHEMA_VERSION = 2


class ConfigError(ValueError):
    """Raised when the study configuration is incomplete or inconsistent."""


_DEFAULTS: dict[str, Any] = {
    "schema_version": SCHEMA_VERSION,
    "run": {
        "mode": "debug",
        "run_id": None,
        "output_root": "runs",
        "overwrite": False,
        "allow_untracked_provenance": False,
    },
    "metadata": {
        "calibration_status": "UNVALIDATED_DEFAULTS",
        "calibration_sources": [],
        "random_seed": 20260718,
        "notes": "",
    },
    "optimization": {
        "evaluation_spans": None,
        "restarts": 5,
        "multi_seed_runs": 5,
        "air_weight": 1.0,
        "margin_weight": 0.5,
        "outage_weight": 12.0,
        "variance_weight": 0.05,
        "smoothness_weight": 0.02,
        "band_balance_weight": 10.0,
        "cvar_weight": 2.0,
        "robust_weight": 1.0,
        "robust_training_samples": 64,
        "robust_training_seed": 20260720,
        "robust_cvar_alpha": 0.10,
        "minimum_air_gain_fraction": 0.001,
        "minimum_ngmi_gain": 0.0001,
        "minimum_nominal_gain_tbps": 0.0,
        "minimum_robust_gain_ci_low_tbps": 0.0,
        "capacity_regression_tolerance_tbps": 1e-9,
        "minimum_band_working_fraction": {"S": 0.80, "C": 0.80, "L": 0.80},
        "projection_smoothing_strength": 0.0,
        "early_stopping_patience": 10,
        "early_stopping_tolerance": 1e-6,
    },
    "waveform": {
        "equalizer_mode": "pilot_aided",
        "carrier_recovery_mode": "pilot_aided",
        "equalizer_ridge": 1e-5,
        "timing_search": True,
        "metric_bootstrap_blocks": 100,
        "bootstrap_block_symbols": 256,
        "repeats_per_band": 5,
        "pilot_symbols": 4096,
        "payload_symbols": 32768,
        "pilot_spacing": 64,
        "consistency_tolerance_db": 1.5,
        "acquisition_snr_threshold_db": 5.0,
        "bps_overlap": 0.5,
        "bps_smoothing_blocks": 3,
        "minimum_reliability": 0.02,
    },
    "validation": {
        "require_external_validation": True,
        "external_reference_csv": "validation_data/external_reference.csv",
        "external_max_wavelength_error_nm": 0.5,
        "external_minimum_coverage_fraction": 0.80,
        "external_minimum_sources": 2,
        "external_minimum_source_types": 2,
        "external_minimum_wavelengths_per_band": 3,
        "external_minimum_span_counts": 2,
        "external_max_gsnr_rmse_db": 1.5,
        "external_max_gsnr_bias_db": 0.75,
        "external_max_nli_relative_rmse": 0.25,
        "external_allow_interpolation": False,
        "require_uncertainty_analysis": True,
        "minimum_uncertainty_success_fraction": 0.95,
        "minimum_uncertainty_holdout_samples": 200,
        "minimum_waveform_repeats": 3,
        "minimum_optimizer_seeds": 5,
        "minimum_probability_of_improvement": 0.95,
        "minimum_adaptive_capacity_gain_fraction": 0.0,
        "minimum_adaptive_ngmi_gain": 0.0001,
        "require_nonzero_uniform_reach": True,
        "minimum_target_band_working_fraction": 0.80,
        "minimum_robust_capacity_gain_tbps": 0.0,
        "nominal_closed_form_bandwidth_limit_thz": 15.0,
        "raman_relative_rmse_limit": 0.01,
        "raman_max_relative_error_limit": 0.03,
        "convergence_relative_tolerance": 0.01,
        "required_figures": [],
    },
    "uncertainty": {
        "enabled": True,
        "holdout_samples": 256,
        "holdout_seed": 20260721,
        "bootstrap_samples": 2000,
        "cvar_alpha": 0.10,
        "distributions": {
            "attenuation_scale": {"distribution": "normal", "mean": 1.0, "std": 0.02, "minimum": 0.85, "maximum": 1.15},
            "gamma_scale": {"distribution": "lognormal", "mean": 1.0, "std": 0.05, "minimum": 0.75, "maximum": 1.30},
            "dispersion_scale": {"distribution": "normal", "mean": 1.0, "std": 0.02, "minimum": 0.85, "maximum": 1.15},
            "noise_figure_offset_db": {"distribution": "normal", "mean": 0.0, "std": 0.30, "minimum": -1.0, "maximum": 1.0},
            "raman_gain_scale": {"distribution": "lognormal", "mean": 1.0, "std": 0.08, "minimum": 0.60, "maximum": 1.50},
            "pump_power_scale": {"distribution": "normal", "mean": 1.0, "std": 0.03, "minimum": 0.80, "maximum": 1.20},
            "transceiver_snr_offset_db": {"distribution": "normal", "mean": 0.0, "std": 0.50, "minimum": -2.0, "maximum": 2.0},
        },
        "correlation": None,
    },
    "reproducibility": {
        "strict_git_clean": False,
        "write_manifest": True,
        "repository_url": None,
        "lock_files": ["pyproject.toml"],
    },
}

_ALLOWED: dict[str, Any] = {
    "schema_version": None,
    "run": {"mode": None, "run_id": None, "output_root": None, "overwrite": None, "allow_untracked_provenance": None},
    "metadata": {"study_title": None, "random_seed": None, "calibration_status": None, "calibration_sources": None, "notes": None},
    "grid": {"mode": None, "lambda_min_nm": None, "lambda_max_nm": None, "spacing_ghz": None, "subset_channels": None, "subset_center_nm": None},
    "modulation": {"symbol_rate_gbaud": None, "roll_off": None, "format": None, "bits_per_symbol_per_pol": None, "oversampling": None, "rrc_span_symbols": None, "waveform_symbols": None},
    "fec": {"overhead_fraction": None, "pre_fec_ber_target": None, "ngmi_target": None, "threshold_metric": None, "b2b_snr_sweep_db": None},
    "fiber": {"span_length_km": None, "max_spans": None, "effective_area_um2": None, "gamma_per_w_km_at_1550": None, "dispersion_ps_nm_km_at_1550": None, "dispersion_slope_ps_nm2_km": None, "pmd_ps_sqrt_km": None, "attenuation_anchors": {"wavelength_nm": None, "db_per_km": None}},
    "raman": {"integration_step_m": None, "save_step_m": None, "gain_peak_m_per_w": None, "gain_csv": None, "semrau_linear_slope_per_w_km_thz": None, "pumps": None, "equivalent_noise_figure_db": None},
    "amplification": {"noise_bandwidth_multiplier": None, "receiver_equivalent_noise_bandwidth_hz": None, "bands": None, "gain_flatness_tolerance_db": None},
    "nli": {"primary_model": None, "modulation_correction": None, "coherence_epsilon": None, "transceiver_snr_db": None, "semrau_valid_bandwidth_thz": None},
    "launch": {"flat_power_dbm_per_channel": None, "min_power_dbm_per_channel": None, "max_power_dbm_per_channel": None, "fixed_preemphasis_s_to_l_db": None},
    "optimization": {key: None for key in _DEFAULTS["optimization"]} | {"target_spans": None, "method": None, "iterations": None, "control_points": None, "learning_rate": None, "spsa_perturbation_db": None, "seed": None, "restart_sigma_db": None, "adam_beta1": None, "adam_beta2": None, "softmin_temperature_ngmi": None},
    "waveform": {key: None for key in _DEFAULTS["waveform"]} | {"selected_wavelengths_nm": None, "cma_taps": None, "cma_step_size": None, "cma_training_symbols": None, "bps_trial_phases": None, "bps_block_symbols": None, "laser_linewidth_hz": None, "carrier_frequency_offset_hz": None, "apply_pmd": None, "apply_phase_noise": None, "use_gn_nli_noise": None},
    "output": {"directory": None, "figure_directory": None, "png_dpi": None, "save_profiles": None},
    "validation": {key: None for key in _DEFAULTS["validation"]},
    "uncertainty": {"enabled": None, "holdout_samples": None, "holdout_seed": None, "bootstrap_samples": None, "cvar_alpha": None, "distributions": None, "correlation": None, "samples": None, "seed": None, "sigma": None},
    "reproducibility": {key: None for key in _DEFAULTS["reproducibility"]},
}


def _deep_merge(base: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(base))
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def migrate_config(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Migrate legacy schema-1 configurations to the canonical schema-2 form."""
    cfg = deepcopy(dict(raw))
    version = int(cfg.get("schema_version", 1))
    if version > SCHEMA_VERSION:
        raise ConfigError(f"schema_version {version} is newer than supported {SCHEMA_VERSION}")
    cfg["schema_version"] = SCHEMA_VERSION
    cfg.setdefault("run", {})
    opt = cfg.setdefault("optimization", {})
    aliases = {
        "minimum_capacity_gain_fraction": "minimum_air_gain_fraction",
        "maximum_capacity_regression_fraction": "capacity_regression_tolerance_tbps",
        "minimum_ngmi_weight": "margin_weight",
    }
    for old, new in aliases.items():
        if old in opt and new not in opt:
            opt[new] = opt.pop(old)
        elif old in opt:
            opt.pop(old)

    # Legacy schema-1 key retained in old config_q2_final.yaml files.
    # The schema-2 optimizer uses softmin_temperature_ngmi for NGMI-domain
    # margins; the old dB-domain value has no safe one-to-one conversion.
    # Drop it after migration so strict unknown-key validation still catches
    # genuinely unsupported keys.
    opt.pop("softmin_temperature_db", None)
    unc = cfg.setdefault("uncertainty", {})
    if "samples" in unc and "holdout_samples" not in unc:
        unc["holdout_samples"] = unc.pop("samples")
    if "seed" in unc and "holdout_seed" not in unc:
        unc["holdout_seed"] = unc.pop("seed")
    if "sigma" in unc and "distributions" not in unc:
        sigma = unc.pop("sigma") or {}
        unc["distributions"] = deepcopy(_DEFAULTS["uncertainty"]["distributions"])
        mapping = {
            "attenuation_fraction": "attenuation_scale",
            "gamma_fraction": "gamma_scale",
            "dispersion_fraction": "dispersion_scale",
            "noise_figure_db": "noise_figure_offset_db",
            "raman_gain_fraction": "raman_gain_scale",
            "pump_power_fraction": "pump_power_scale",
            "transceiver_snr_db": "transceiver_snr_offset_db",
        }
        for old, name in mapping.items():
            if old in sigma:
                unc["distributions"][name]["std"] = float(sigma[old])
    return cfg


def apply_defaults(cfg: Mapping[str, Any]) -> dict[str, Any]:
    migrated = migrate_config(cfg)
    result = _deep_merge(_DEFAULTS, migrated)
    if result["optimization"].get("evaluation_spans") is None:
        result["optimization"]["evaluation_spans"] = [int(result["optimization"].get("target_spans", 1))]
    return result


def _reject_unknown(value: Mapping[str, Any], allowed: Mapping[str, Any], path: str = "") -> None:
    for key, item in value.items():
        dotted = f"{path}.{key}" if path else key
        if key not in allowed:
            raise ConfigError(f"Unknown configuration key: {dotted}")
        child = allowed[key]
        if isinstance(item, Mapping) and isinstance(child, Mapping):
            _reject_unknown(item, child, dotted)


def _number(value: Any, path: str, *, minimum: float | None = None, maximum: float | None = None, strict_minimum: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{path} must be numeric")
    number = float(value)
    if minimum is not None and (number <= minimum if strict_minimum else number < minimum):
        op = ">" if strict_minimum else ">="
        raise ConfigError(f"{path} must be {op} {minimum}")
    if maximum is not None and number > maximum:
        raise ConfigError(f"{path} must be <= {maximum}")
    return number


def _require_sections(cfg: Mapping[str, Any]) -> None:
    required = {"metadata", "grid", "modulation", "fec", "fiber", "raman", "amplification", "nli", "launch", "optimization", "waveform", "output", "validation", "uncertainty", "reproducibility", "run"}
    missing = sorted(required.difference(cfg))
    if missing:
        raise ConfigError(f"Missing top-level sections: {missing}")


def _validate_calibration_sources(sources: Any, publication: bool) -> None:
    if not isinstance(sources, list):
        raise ConfigError("metadata.calibration_sources must be a list")
    seen: set[str] = set()
    for index, source in enumerate(sources):
        path = f"metadata.calibration_sources[{index}]"
        if not isinstance(source, Mapping):
            raise ConfigError(f"{path} must be a mapping")
        required = {"source_id", "source_type", "reference", "parameter_group"}
        missing = required.difference(source)
        if missing:
            raise ConfigError(f"{path} missing {sorted(missing)}")
        source_id = str(source["source_id"]).strip()
        if not source_id or source_id in seen:
            raise ConfigError(f"{path}.source_id must be non-empty and unique")
        seen.add(source_id)
        if publication and str(source["reference"]).strip().lower() in {"", "todo", "placeholder"}:
            raise ConfigError(f"{path}.reference is not traceable")


def _validate_distributions(cfg: Mapping[str, Any]) -> None:
    allowed = {"normal", "lognormal", "uniform", "triangular", "empirical"}
    distributions = cfg["uncertainty"].get("distributions")
    if not isinstance(distributions, Mapping) or not distributions:
        raise ConfigError("uncertainty.distributions must be a non-empty mapping")
    for name, spec in distributions.items():
        path = f"uncertainty.distributions.{name}"
        if not isinstance(spec, Mapping):
            raise ConfigError(f"{path} must be a mapping")
        kind = str(spec.get("distribution", "")).lower()
        if kind not in allowed:
            raise ConfigError(f"{path}.distribution must be one of {sorted(allowed)}")
        if kind in {"normal", "lognormal"}:
            _number(spec.get("std"), f"{path}.std", minimum=0.0, strict_minimum=True)
        elif kind == "uniform":
            if _number(spec.get("minimum"), f"{path}.minimum") >= _number(spec.get("maximum"), f"{path}.maximum"):
                raise ConfigError(f"{path}.minimum must be less than maximum")
        elif kind == "triangular":
            low, mode, high = (float(spec.get(k)) for k in ("minimum", "mode", "maximum"))
            if not low <= mode <= high or low == high:
                raise ConfigError(f"{path} triangular bounds are invalid")
        elif kind == "empirical":
            values = spec.get("values")
            if not isinstance(values, list) or len(values) < 4:
                raise ConfigError(f"{path}.values must contain at least four values")


def validate_config(cfg: Mapping[str, Any], *, base_dir: str | Path | None = None) -> None:
    _require_sections(cfg)
    _reject_unknown(cfg, _ALLOWED)
    if int(cfg.get("schema_version", 0)) != SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {SCHEMA_VERSION}")
    mode = str(cfg["run"].get("mode", "debug")).lower()
    if mode not in {"publication", "smoke", "debug"}:
        raise ConfigError("run.mode must be publication, smoke, or debug")
    publication = mode == "publication"

    grid = cfg["grid"]
    if grid["mode"] not in {"full_scl", "paper_240_subset"}:
        raise ConfigError("grid.mode must be full_scl or paper_240_subset")
    low = _number(grid["lambda_min_nm"], "grid.lambda_min_nm", minimum=1000)
    high = _number(grid["lambda_max_nm"], "grid.lambda_max_nm", minimum=1000)
    if low >= high:
        raise ConfigError("grid.lambda_min_nm must be less than lambda_max_nm")
    _number(grid["spacing_ghz"], "grid.spacing_ghz", minimum=0, strict_minimum=True)
    if int(grid["subset_channels"]) < 2:
        raise ConfigError("grid.subset_channels must be at least 2")

    mod = cfg["modulation"]
    if str(mod["format"]).upper() != "DP-16QAM" or int(mod["bits_per_symbol_per_pol"]) != 4:
        raise ConfigError("Only DP-16QAM with four bits/polarization is supported")
    _number(mod["symbol_rate_gbaud"], "modulation.symbol_rate_gbaud", minimum=0, strict_minimum=True)
    _number(mod["roll_off"], "modulation.roll_off", minimum=0, maximum=1)
    for key in ("oversampling", "rrc_span_symbols", "waveform_symbols"):
        if int(mod[key]) < 1:
            raise ConfigError(f"modulation.{key} must be positive")

    fec = cfg["fec"]
    if fec["threshold_metric"] not in {"ngmi", "ber"}:
        raise ConfigError("fec.threshold_metric must be ngmi or ber")
    _number(fec["ngmi_target"], "fec.ngmi_target", minimum=0, maximum=1, strict_minimum=True)
    sweep = list(fec["b2b_snr_sweep_db"])
    if len(sweep) != 3 or float(sweep[0]) >= float(sweep[1]) or float(sweep[2]) <= 0:
        raise ConfigError("fec.b2b_snr_sweep_db must be [start, stop, positive_step]")
    if publication and (float(sweep[0]) > -4.0 or float(sweep[1]) < 30.0 or float(sweep[2]) > 1.0):
        raise ConfigError("Publication B2B sweep must cover at least -4..30 dB with step <= 1 dB")

    fiber = cfg["fiber"]
    for key in ("span_length_km", "max_spans", "effective_area_um2", "gamma_per_w_km_at_1550"):
        _number(fiber[key], f"fiber.{key}", minimum=0, strict_minimum=True)
    anchors = fiber["attenuation_anchors"]
    wl, loss = list(anchors["wavelength_nm"]), list(anchors["db_per_km"])
    if len(wl) != len(loss) or len(wl) < 2 or any(float(x) <= 0 for x in loss):
        raise ConfigError("fiber.attenuation_anchors arrays must have equal length >=2 and positive loss")
    if any(float(wl[i]) >= float(wl[i + 1]) for i in range(len(wl) - 1)):
        raise ConfigError("fiber attenuation wavelengths must be strictly increasing")

    raman = cfg["raman"]
    for key in ("integration_step_m", "save_step_m", "gain_peak_m_per_w"):
        _number(raman[key], f"raman.{key}", minimum=0, strict_minimum=True)
    if float(raman["integration_step_m"]) > float(raman["save_step_m"]):
        raise ConfigError("raman.integration_step_m cannot exceed save_step_m")
    for index, pump in enumerate(raman.get("pumps", [])):
        if pump.get("direction") not in {"forward", "backward"}:
            raise ConfigError(f"raman.pumps[{index}].direction must be forward/backward")
        _number(pump.get("power_w"), f"raman.pumps[{index}].power_w", minimum=0, strict_minimum=True)

    launch = cfg["launch"]
    minimum, maximum, flat = map(float, (launch["min_power_dbm_per_channel"], launch["max_power_dbm_per_channel"], launch["flat_power_dbm_per_channel"]))
    if minimum >= maximum or not minimum <= flat <= maximum:
        raise ConfigError("Invalid launch-power bounds or flat launch value")

    opt = cfg["optimization"]
    for key in ("target_spans", "iterations", "control_points", "restarts", "multi_seed_runs", "learning_rate", "spsa_perturbation_db", "robust_training_samples"):
        _number(opt[key], f"optimization.{key}", minimum=0, strict_minimum=True)
    spans = sorted({int(x) for x in opt["evaluation_spans"]})
    if int(opt["target_spans"]) not in spans or min(spans) < 1 or max(spans) > int(fiber["max_spans"]):
        raise ConfigError("optimization.evaluation_spans must include target_spans and lie within fiber.max_spans")
    fractions = opt["minimum_band_working_fraction"]
    if set(fractions) != {"S", "C", "L"} or any(not 0 <= float(v) <= 1 for v in fractions.values()):
        raise ConfigError("optimization.minimum_band_working_fraction must define S/C/L values in [0,1]")

    waveform = cfg["waveform"]
    if len(waveform["selected_wavelengths_nm"]) < 3:
        raise ConfigError("waveform.selected_wavelengths_nm must include S/C/L representatives")
    if int(waveform["pilot_symbols"]) < 64 or int(waveform["payload_symbols"]) < 256:
        raise ConfigError("waveform pilot/payload symbol counts are too small")
    if waveform["equalizer_mode"] not in {"pilot_aided", "cma"}:
        raise ConfigError("waveform.equalizer_mode must be pilot_aided or cma")
    if waveform["carrier_recovery_mode"] not in {"pilot_aided", "bps"}:
        raise ConfigError("waveform.carrier_recovery_mode must be pilot_aided or bps")

    validation = cfg["validation"]
    for key in ("external_minimum_coverage_fraction", "minimum_uncertainty_success_fraction", "minimum_probability_of_improvement", "minimum_target_band_working_fraction"):
        _number(validation[key], f"validation.{key}", minimum=0, maximum=1)
    _validate_calibration_sources(cfg["metadata"].get("calibration_sources", []), publication)
    _validate_distributions(cfg)
    holdout_seed = int(cfg["uncertainty"]["holdout_seed"])
    training_seed = int(opt["robust_training_seed"])
    if holdout_seed == training_seed:
        raise ConfigError("optimization.robust_training_seed and uncertainty.holdout_seed must differ")
    if publication:
        if str(cfg["metadata"].get("calibration_status", "")).upper() != "CALIBRATED":
            raise ConfigError("Publication mode requires metadata.calibration_status=CALIBRATED")
        if not cfg["metadata"].get("calibration_sources"):
            raise ConfigError("Publication mode requires traceable calibration sources")
        if not bool(cfg["uncertainty"]["enabled"]):
            raise ConfigError("Publication mode cannot disable uncertainty analysis")
        if int(cfg["uncertainty"]["holdout_samples"]) < int(validation["minimum_uncertainty_holdout_samples"]):
            raise ConfigError("uncertainty.holdout_samples is below the publication minimum")
        if not bool(cfg["reproducibility"]["strict_git_clean"]):
            raise ConfigError("Publication mode requires reproducibility.strict_git_clean=true")
        base = Path(base_dir or ".")
        reference = Path(validation["external_reference_csv"])
        reference = reference if reference.is_absolute() else base / reference
        if bool(validation["require_external_validation"]) and not reference.exists():
            raise ConfigError(f"External validation file does not exist: {reference}")
        if not cfg["run"].get("run_id"):
            raise ConfigError("Publication mode requires a run-specific run.run_id")


def load_config(path: str | Path, *, write_resolved_to: str | Path | None = None) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(target)
    raw = yaml.safe_load(target.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ConfigError("The YAML root must be a mapping")
    cfg = apply_defaults(raw)
    validate_config(cfg, base_dir=target.parent)
    if write_resolved_to is not None:
        destination = Path(write_resolved_to)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return cfg


def with_overrides(cfg: Mapping[str, Any], **overrides: Any) -> dict[str, Any]:
    output = deepcopy(dict(cfg))
    for dotted_key, value in overrides.items():
        node = output
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                raise ConfigError(f"Cannot override unknown path {dotted_key}")
            node = node[part]
        node[parts[-1]] = value
    output = apply_defaults(output)
    validate_config(output)
    return output


def resolved_config_json(cfg: Mapping[str, Any]) -> str:
    """Return canonical machine-readable configuration JSON."""
    return json.dumps(cfg, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
