"""Traceable calibration-data ingestion and interpolation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping
import json

import numpy as np
import pandas as pd

from isrs_scl.validation.reproducibility import sha256_file

_REQUIRED_COLUMNS = {
    "source_id", "source_type", "parameter_group", "parameter", "value", "unit",
    "uncertainty", "reference", "independent",
}


class CalibrationError(ValueError):
    """Raised for malformed, untraceable, or unsupported calibration evidence."""


@dataclass(frozen=True)
class CalibrationBundle:
    parameters: dict[str, Any]
    provenance: pd.DataFrame
    source_hashes: dict[str, str]
    calibrated: bool
    reasons: tuple[str, ...]


def _load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".json", ".jsonl"}:
        if suffix == ".jsonl":
            return pd.read_json(path, lines=True)
        payload = json.loads(path.read_text(encoding="utf-8"))
        return pd.DataFrame(payload if isinstance(payload, list) else payload.get("records", []))
    raise CalibrationError(f"Unsupported calibration format: {path.suffix}")


def validate_calibration_frame(frame: pd.DataFrame, *, source_path: str | Path | None = None) -> pd.DataFrame:
    missing = _REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise CalibrationError(f"Calibration data missing columns: {sorted(missing)}")
    result = frame.copy()
    result["source_id"] = result["source_id"].astype(str).str.strip()
    result["source_type"] = result["source_type"].astype(str).str.strip().str.lower()
    result["parameter_group"] = result["parameter_group"].astype(str).str.strip().str.lower()
    result["parameter"] = result["parameter"].astype(str).str.strip()
    result["unit"] = result["unit"].astype(str).str.strip()
    result["reference"] = result["reference"].astype(str).str.strip()
    result["value"] = pd.to_numeric(result["value"], errors="coerce")
    result["uncertainty"] = pd.to_numeric(result["uncertainty"], errors="coerce")
    result["independent"] = result["independent"].map(lambda value: str(value).strip().lower() in {"1", "true", "yes", "y"})
    if "wavelength_nm" in result:
        result["wavelength_nm"] = pd.to_numeric(result["wavelength_nm"], errors="coerce")
    if "frequency_thz" in result:
        result["frequency_thz"] = pd.to_numeric(result["frequency_thz"], errors="coerce")
    invalid = result[
        result["source_id"].eq("")
        | result["reference"].str.lower().isin({"", "todo", "placeholder", "real_value"})
        | result["value"].isna()
        | result["uncertainty"].isna()
        | (result["uncertainty"] < 0)
        | result["unit"].eq("")
    ]
    if not invalid.empty:
        raise CalibrationError(f"Calibration contains {len(invalid)} blank/placeholder/invalid rows")
    identity = [column for column in ("source_id", "parameter_group", "parameter", "wavelength_nm", "frequency_thz") if column in result]
    if result.duplicated(identity).any():
        raise CalibrationError("Calibration contains duplicate parameter identities")
    if source_path is not None:
        result["source_file"] = Path(source_path).name
        result["source_sha256"] = sha256_file(source_path)
    return result.reset_index(drop=True)


def load_calibration_files(paths: Iterable[str | Path]) -> CalibrationBundle:
    frames: list[pd.DataFrame] = []
    hashes: dict[str, str] = {}
    reasons: list[str] = []
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            reasons.append(f"missing calibration file: {path}")
            continue
        hashes[path.name] = sha256_file(path)
        frames.append(validate_calibration_frame(_load_table(path), source_path=path))
    if not frames:
        return CalibrationBundle({}, pd.DataFrame(), hashes, False, tuple(reasons or ["no calibration files supplied"]))
    provenance = pd.concat(frames, ignore_index=True)
    parameters: dict[str, Any] = {}
    for (group, parameter), rows in provenance.groupby(["parameter_group", "parameter"], sort=False):
        key = f"{group}.{parameter}"
        records = rows.to_dict(orient="records")
        parameters[key] = records[0]["value"] if len(records) == 1 else records
    source_types = set(provenance["source_type"])
    independent = bool(provenance["independent"].all())
    calibrated = independent and provenance["source_id"].nunique() >= 1 and len(source_types) >= 1
    if not independent:
        reasons.append("one or more calibration sources are not independent")
    return CalibrationBundle(parameters, provenance, hashes, calibrated, tuple(reasons))


def interpolate_parameter(
    frame: pd.DataFrame,
    parameter: str,
    wavelengths_nm: np.ndarray,
    *,
    allow_extrapolation: bool = False,
    combine_rule: str = "inverse_variance",
) -> tuple[np.ndarray, np.ndarray]:
    subset = frame[frame["parameter"] == parameter].dropna(subset=["wavelength_nm", "value", "uncertainty"]).copy()
    if subset.empty:
        raise CalibrationError(f"No wavelength-resolved rows for {parameter!r}")
    target = np.asarray(wavelengths_nm, dtype=float)
    low, high = float(subset["wavelength_nm"].min()), float(subset["wavelength_nm"].max())
    if not allow_extrapolation and (target.min() < low or target.max() > high):
        raise CalibrationError(f"{parameter} requested outside calibrated range {low:g}..{high:g} nm")
    per_source: list[tuple[np.ndarray, np.ndarray]] = []
    for _, group in subset.groupby("source_id", sort=False):
        group = group.sort_values("wavelength_nm")
        x = group["wavelength_nm"].to_numpy(float)
        y = group["value"].to_numpy(float)
        u = group["uncertainty"].to_numpy(float)
        per_source.append((np.interp(target, x, y), np.interp(target, x, u)))
    values = np.vstack([item[0] for item in per_source])
    uncertainties = np.vstack([item[1] for item in per_source])
    if len(per_source) == 1:
        return values[0], uncertainties[0]
    if combine_rule != "inverse_variance":
        raise CalibrationError("Multiple sources require combine_rule='inverse_variance'")
    weights = 1.0 / np.maximum(uncertainties, 1e-15) ** 2
    combined = np.sum(weights * values, axis=0) / np.sum(weights, axis=0)
    combined_u = np.sqrt(1.0 / np.sum(weights, axis=0))
    return combined, combined_u


def configuration_can_be_calibrated(cfg: Mapping[str, Any], bundle: CalibrationBundle) -> tuple[bool, tuple[str, ...]]:
    reasons = list(bundle.reasons)
    declared = cfg.get("metadata", {}).get("calibration_sources", [])
    declared_ids = {str(item.get("source_id")) for item in declared if isinstance(item, Mapping)}
    actual_ids = set(bundle.provenance.get("source_id", pd.Series(dtype=str)).astype(str))
    if not bundle.calibrated:
        reasons.append("calibration bundle did not satisfy independence/provenance rules")
    if declared_ids and not declared_ids.issubset(actual_ids):
        reasons.append(f"declared sources absent from files: {sorted(declared_ids - actual_ids)}")
    required_groups = {"fiber", "raman", "amplification", "transceiver"}
    present_groups = set(bundle.provenance.get("parameter_group", pd.Series(dtype=str)).astype(str))
    missing = required_groups - present_groups
    if missing:
        reasons.append(f"missing calibrated parameter groups: {sorted(missing)}")
    return not reasons, tuple(dict.fromkeys(reasons))
