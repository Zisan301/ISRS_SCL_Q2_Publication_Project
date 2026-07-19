"""Independent-tool and experimental validation with explicit units/provenance."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

_SUPPORTED_METRICS = {
    "gsnr_db": "gsnr_db",
    "osnr_01nm_db": "osnr_01nm_db",
    "nli_w": "nli_w",
    "ase_channel_w": "ase_channel_w",
    "launch_power_dbm": "launch_power_dbm",
}
_METRIC_UNITS = {"gsnr_db": "dB", "osnr_01nm_db": "dB", "nli_w": "W", "ase_channel_w": "W", "launch_power_dbm": "dBm"}
_STRATEGY_ALIASES = {"flat": "flat", "uniform": "flat", "fixed": "fixed", "preemphasis": "fixed", "fixed_preemphasis": "fixed", "adaptive": "adaptive", "optimized": "adaptive", "optimised": "adaptive"}
_SOURCE_ALIASES = {"gnpy": "gnpy", "ssfm": "ssfm", "experiment": "experiment", "measurement": "experiment", "vendor": "vendor"}


@dataclass(frozen=True)
class ExternalValidationResult:
    comparisons: pd.DataFrame
    summary: pd.DataFrame
    requirements: pd.DataFrame
    passed: bool
    reasons: tuple[str, ...]


def external_validation_template() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "source_id", "source_type", "tool_version", "configuration_hash", "date",
        "provenance_reference", "independent", "strategy", "spans", "band",
        "wavelength_nm", "metric", "metric_unit", "reference_value",
        "reference_uncertainty", "notes",
    ])


def _canonical(value: Any, aliases: Mapping[str, str], label: str) -> str:
    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if key not in aliases:
        raise ValueError(f"Unsupported {label}: {value!r}")
    return aliases[key]


def _infer_band(wavelength_nm: float) -> str:
    if wavelength_nm < 1530:
        return "S"
    if wavelength_nm < 1565:
        return "C"
    return "L"


def _normalize_external(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if {"metric", "reference_value"}.issubset(result.columns):
        pass
    else:
        metric_columns = [metric for metric in _SUPPORTED_METRICS if metric in result.columns]
        if not metric_columns:
            raise ValueError("External validation requires metric/reference_value columns or supported wide metrics")
        ids = [column for column in result.columns if column not in metric_columns]
        result = result.melt(id_vars=ids, value_vars=metric_columns, var_name="metric", value_name="reference_value")
    required = {
        "source_id", "source_type", "tool_version", "configuration_hash", "date",
        "provenance_reference", "independent", "strategy", "spans", "wavelength_nm",
        "metric", "metric_unit", "reference_value", "reference_uncertainty",
    }
    missing = required.difference(result.columns)
    if missing:
        raise ValueError(f"External validation missing columns: {sorted(missing)}")
    result["source_id"] = result["source_id"].astype(str).str.strip()
    result["source_type"] = result["source_type"].map(lambda x: _canonical(x, _SOURCE_ALIASES, "source_type"))
    result["strategy"] = result["strategy"].map(lambda x: _canonical(x, _STRATEGY_ALIASES, "strategy"))
    result["metric"] = result["metric"].astype(str).str.strip().str.lower()
    unsupported = sorted(set(result["metric"]) - set(_SUPPORTED_METRICS))
    if unsupported:
        raise ValueError(f"Unsupported external metrics: {unsupported}")
    result["metric_unit"] = result["metric_unit"].astype(str).str.strip()
    for metric, group in result.groupby("metric"):
        expected = _METRIC_UNITS[metric].lower()
        if any(str(value).lower() != expected for value in group["metric_unit"]):
            raise ValueError(f"Metric {metric} must use unit {_METRIC_UNITS[metric]}")
    for column in ("spans", "wavelength_nm", "reference_value", "reference_uncertainty"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["spans"] = result["spans"].astype("Int64")
    result["independent"] = result["independent"].map(lambda value: str(value).strip().lower() in {"1", "true", "yes", "y"})
    text_columns = ("tool_version", "configuration_hash", "date", "provenance_reference")
    invalid_text = np.zeros(len(result), dtype=bool)
    for column in text_columns:
        result[column] = result[column].astype(str).str.strip()
        invalid_text |= result[column].str.lower().isin({"", "nan", "todo", "placeholder", "real_value"}).to_numpy()
    invalid = result["source_id"].eq("") | result["reference_value"].isna() | result["reference_uncertainty"].isna() | (result["reference_uncertainty"] < 0) | result["spans"].isna() | result["wavelength_nm"].isna() | ~result["independent"] | invalid_text
    if invalid.any():
        raise ValueError(f"External validation contains {int(invalid.sum())} placeholder, dependent, blank, or invalid rows")
    result["spans"] = result["spans"].astype(int)
    if "band" not in result:
        result["band"] = result["wavelength_nm"].map(_infer_band)
    else:
        result["band"] = result["band"].astype(str).str.strip().str.upper()
    identity = ["source_id", "strategy", "spans", "wavelength_nm", "metric"]
    if result.duplicated(identity).any():
        raise ValueError("External validation contains duplicate reference identities")
    return result.reset_index(drop=True)


def load_external_validation(path: str | Path) -> pd.DataFrame:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(target)
    return _normalize_external(pd.read_csv(target))


def _model_subset(model: pd.DataFrame, item: pd.Series) -> pd.DataFrame:
    strategy = _canonical(item["strategy"], _STRATEGY_ALIASES, "strategy")
    model_strategy = model["strategy"].map(lambda x: _canonical(x, _STRATEGY_ALIASES, "strategy"))
    return model[(model_strategy == strategy) & (pd.to_numeric(model["spans"], errors="coerce") == int(item["spans"]))].sort_values("wavelength_nm")


def _match_value(subset: pd.DataFrame, item: pd.Series, model_column: str, tolerance: float, allow_interpolation: bool) -> dict[str, Any]:
    if subset.empty or model_column not in subset:
        return {"model_value": np.nan, "matched_wavelength_nm": np.nan, "wavelength_error_nm": np.nan, "interpolated": 0, "matched": 0}
    target = float(item["wavelength_nm"])
    x = subset["wavelength_nm"].to_numpy(float)
    y = subset[model_column].to_numpy(float)
    differences = np.abs(x - target)
    nearest = int(np.argmin(differences))
    if differences[nearest] <= tolerance:
        return {"model_value": float(y[nearest]), "matched_wavelength_nm": float(x[nearest]), "wavelength_error_nm": float(differences[nearest]), "interpolated": 0, "matched": 1}
    if allow_interpolation and x.min() <= target <= x.max():
        return {"model_value": float(np.interp(target, x, y)), "matched_wavelength_nm": target, "wavelength_error_nm": 0.0, "interpolated": 1, "matched": 1}
    return {"model_value": np.nan, "matched_wavelength_nm": float(x[nearest]), "wavelength_error_nm": float(differences[nearest]), "interpolated": 0, "matched": 0}


def compare_external_validation(
    model_channel_sweep: pd.DataFrame,
    external_frame: pd.DataFrame,
    *,
    wavelength_tolerance_nm: float = 0.25,
    thresholds: Mapping[str, Any] | None = None,
    allow_interpolation: bool = False,
) -> ExternalValidationResult:
    required_model = {"strategy", "spans", "wavelength_nm"}
    missing = required_model.difference(model_channel_sweep.columns)
    if missing:
        raise ValueError(f"Model frame missing columns: {sorted(missing)}")
    external = _normalize_external(external_frame)
    model = model_channel_sweep.copy()
    rows: list[dict[str, Any]] = []
    for external_index, item in external.iterrows():
        metric = str(item["metric"])
        match = _match_value(_model_subset(model, item), item, _SUPPORTED_METRICS[metric], float(wavelength_tolerance_nm), bool(allow_interpolation))
        reference = float(item["reference_value"])
        model_value = float(match["model_value"]) if match["matched"] else np.nan
        residual = model_value - reference if match["matched"] else np.nan
        uncertainty = float(item["reference_uncertainty"])
        rows.append({
            **item.to_dict(), **match, "external_row": external_index,
            "residual": residual,
            "relative_residual": residual / max(abs(reference), 1e-30) if match["matched"] else np.nan,
            "standardized_residual": residual / uncertainty if match["matched"] and uncertainty > 0 else np.nan,
        })
    comparisons = pd.DataFrame(rows)

    summary_rows: list[dict[str, Any]] = []
    groupings = {
        "overall": [], "metric": ["metric"], "source": ["source_id"], "source_type": ["source_type"],
        "band": ["band"], "span": ["spans"], "strategy": ["strategy"],
    }
    for level, columns in groupings.items():
        groups = [((), comparisons)] if not columns else comparisons.groupby(columns[0] if len(columns) == 1 else columns, sort=False)
        for key, group in groups:
            matched = group[group["matched"] == 1].dropna(subset=["residual"])
            residual = matched["residual"].to_numpy(float)
            relative = matched["relative_residual"].to_numpy(float)
            standardized = matched["standardized_residual"].dropna().to_numpy(float)
            key_tuple = key if isinstance(key, tuple) else (key,)
            row = {"level": level, "requested_rows": len(group), "matched_rows": len(matched), "coverage": len(matched) / max(len(group), 1)}
            row.update({column: value for column, value in zip(columns, key_tuple)})
            row.update({
                "rmse": float(np.sqrt(np.mean(residual**2))) if residual.size else np.nan,
                "mae": float(np.mean(np.abs(residual))) if residual.size else np.nan,
                "bias": float(np.mean(residual)) if residual.size else np.nan,
                "max_abs_error": float(np.max(np.abs(residual))) if residual.size else np.nan,
                "relative_rmse": float(np.sqrt(np.mean(relative**2))) if relative.size else np.nan,
                "reduced_chi_square": float(np.mean(standardized**2)) if standardized.size else np.nan,
            })
            summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    limits = dict(thresholds or {})
    requirements: list[dict[str, Any]] = []
    reasons: list[str] = []
    def requirement(name: str, passed: bool, evidence: Any) -> None:
        requirements.append({"requirement": name, "passed": bool(passed), "evidence": evidence})
        if not passed:
            reasons.append(f"{name}: {evidence}")

    overall = summary[summary["level"] == "overall"].iloc[0]
    requirement("minimum_overall_coverage", float(overall["coverage"]) >= float(limits.get("minimum_external_coverage", 0.90)), float(overall["coverage"]))
    requirement("minimum_sources", external["source_id"].nunique() >= int(limits.get("minimum_sources", 1)), int(external["source_id"].nunique()))
    requirement("minimum_source_types", external["source_type"].nunique() >= int(limits.get("minimum_source_types", 1)), int(external["source_type"].nunique()))
    requirement("minimum_span_counts", external["spans"].nunique() >= int(limits.get("minimum_span_counts", 1)), int(external["spans"].nunique()))
    for band in ("S", "C", "L"):
        band_rows = external[external["band"] == band]
        requirement(f"minimum_wavelengths_{band}", band_rows["wavelength_nm"].nunique() >= int(limits.get("minimum_wavelengths_per_band", 1)), int(band_rows["wavelength_nm"].nunique()))
    metric_summary = summary[summary["level"] == "metric"]
    required_metrics = tuple(limits.get("required_external_metrics", ("gsnr_db",)))
    for metric in required_metrics:
        item = metric_summary[metric_summary["metric"] == metric]
        requirement(f"metric_present_{metric}", not item.empty, int(len(item)))
        if item.empty:
            continue
        row = item.iloc[0]
        if metric == "gsnr_db":
            requirement("gsnr_rmse", np.isfinite(row["rmse"]) and float(row["rmse"]) <= float(limits.get("maximum_gsnr_db_rmse", np.inf)), float(row["rmse"]))
            requirement("gsnr_bias", np.isfinite(row["bias"]) and abs(float(row["bias"])) <= float(limits.get("maximum_gsnr_db_absolute_bias", np.inf)), float(row["bias"]))
        if metric == "nli_w":
            requirement("nli_relative_rmse", np.isfinite(row["relative_rmse"]) and float(row["relative_rmse"]) <= float(limits.get("maximum_nli_relative_rmse", np.inf)), float(row["relative_rmse"]))
    requirements_frame = pd.DataFrame(requirements)
    return ExternalValidationResult(comparisons, summary, requirements_frame, not reasons, tuple(reasons))
