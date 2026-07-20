"""Held-out external validation diagnostics for calibrated GSNR evidence."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from isrs_scl.validation.external_validation import compare_external_validation


@dataclass(frozen=True)
class WavelengthLinearCorrection:
    center_wavelength_nm: float
    intercept_db: float
    slope_db_per_nm: float

    def predict(self, wavelength_nm: pd.Series | np.ndarray | list[float]) -> np.ndarray:
        wavelength = np.asarray(wavelength_nm, dtype=float)
        return self.intercept_db + self.slope_db_per_nm * (wavelength - self.center_wavelength_nm)

    def as_dict(self) -> dict[str, float | str]:
        return {
            "model": "wavelength_linear",
            "residual_definition": "model_value - reference_value",
            "center_wavelength_nm": self.center_wavelength_nm,
            "intercept_db": self.intercept_db,
            "slope_db_per_nm": self.slope_db_per_nm,
        }


@dataclass(frozen=True)
class HeldoutExternalValidationResult:
    calibration_comparisons: pd.DataFrame
    holdout_comparisons: pd.DataFrame
    summary: pd.DataFrame
    requirements: pd.DataFrame
    correction: WavelengthLinearCorrection
    passed: bool
    reasons: tuple[str, ...]


def _comparison_thresholds() -> dict[str, Any]:
    return {
        "minimum_external_coverage": 0.0,
        "minimum_sources": 1,
        "minimum_source_types": 1,
        "minimum_wavelengths_per_band": 0,
        "minimum_span_counts": 1,
        "maximum_gsnr_db_rmse": np.inf,
        "maximum_gsnr_db_absolute_bias": np.inf,
        "required_external_metrics": ("gsnr_db",),
    }


def _matched_gsnr(comparisons: pd.DataFrame) -> pd.DataFrame:
    if comparisons.empty:
        return comparisons.copy()
    metric = comparisons["metric"].astype(str).str.lower()
    matched = comparisons["matched"].astype(int) == 1
    data = comparisons[metric.eq("gsnr_db") & matched].copy()
    needed = ["wavelength_nm", "model_value", "reference_value", "residual"]
    return data.dropna(subset=needed)


def _fit_wavelength_linear(data: pd.DataFrame, center_wavelength_nm: float) -> WavelengthLinearCorrection:
    train = _matched_gsnr(data)
    if len(train) < 3:
        raise ValueError("At least three matched GNPy calibration rows are required")
    if train["wavelength_nm"].nunique() < 2:
        raise ValueError("Calibration rows must include at least two distinct wavelengths")
    x = train["wavelength_nm"].to_numpy(float) - float(center_wavelength_nm)
    y = train["residual"].to_numpy(float)
    slope, intercept = np.polyfit(x, y, deg=1)
    return WavelengthLinearCorrection(float(center_wavelength_nm), float(intercept), float(slope))


def _apply_correction(frame: pd.DataFrame, correction: WavelengthLinearCorrection) -> pd.DataFrame:
    output = frame.copy()
    if output.empty:
        return output
    matched = output["matched"].astype(int) == 1
    predicted = np.full(len(output), np.nan, dtype=float)
    predicted[matched.to_numpy()] = correction.predict(output.loc[matched, "wavelength_nm"])
    output["linear_predicted_residual_db"] = predicted
    output["linear_corrected_model_value"] = output["model_value"] - predicted
    output["linear_corrected_residual"] = output["linear_corrected_model_value"] - output[
        "reference_value"
    ]
    return output


def _leave_one_out_corrected(calibration: pd.DataFrame, center_wavelength_nm: float) -> pd.DataFrame:
    output = calibration.copy()
    output["loo_linear_predicted_residual_db"] = np.nan
    output["loo_linear_corrected_residual"] = np.nan
    matched = _matched_gsnr(output)
    for row_index in matched.index:
        train = matched.drop(index=row_index)
        correction = _fit_wavelength_linear(train, center_wavelength_nm)
        wavelength = [float(matched.loc[row_index, "wavelength_nm"])]
        predicted = float(correction.predict(wavelength)[0])
        output.loc[row_index, "loo_linear_predicted_residual_db"] = predicted
        output.loc[row_index, "loo_linear_corrected_residual"] = (
            float(matched.loc[row_index, "residual"]) - predicted
        )
    return output


def _metrics(
    split: str,
    correction: str,
    comparisons: pd.DataFrame,
    residual_column: str,
) -> dict[str, Any]:
    requested = len(comparisons)
    data = _matched_gsnr(comparisons)
    if residual_column not in data:
        residual = np.asarray([], dtype=float)
    else:
        residual = data[residual_column].dropna().to_numpy(float)
    return {
        "split": split,
        "correction": correction,
        "requested_rows": requested,
        "matched_rows": int(len(data)),
        "coverage": float(len(data) / max(requested, 1)),
        "n": int(residual.size),
        "rmse_db": float(np.sqrt(np.mean(residual**2))) if residual.size else np.nan,
        "mae_db": float(np.mean(np.abs(residual))) if residual.size else np.nan,
        "bias_db": float(np.mean(residual)) if residual.size else np.nan,
        "max_abs_error_db": float(np.max(np.abs(residual))) if residual.size else np.nan,
    }


def _identity_set(frame: pd.DataFrame) -> set[tuple[Any, ...]]:
    data = frame.copy()
    if data.empty:
        return set()
    return set(
        zip(
            data["source_id"].astype(str),
            data["strategy"].astype(str),
            data["spans"].astype(int),
            data["wavelength_nm"].round(6),
            data["metric"].astype(str),
        )
    )


def _requirement(
    rows: list[dict[str, Any]],
    reasons: list[str],
    name: str,
    passed: bool,
    observed: Any,
    threshold: Any,
) -> None:
    rows.append(
        {
            "requirement": name,
            "passed": bool(passed),
            "observed": observed,
            "threshold": threshold,
        }
    )
    if not passed:
        reasons.append(f"{name}: observed={observed!r}, threshold={threshold!r}")


def run_heldout_external_validation(
    model_channel_sweep: pd.DataFrame,
    calibration_external: pd.DataFrame,
    holdout_external: pd.DataFrame,
    *,
    wavelength_tolerance_nm: float = 0.5,
    allow_interpolation: bool = False,
    thresholds: Mapping[str, Any] | None = None,
    center_wavelength_nm: float = 1550.0,
) -> HeldoutExternalValidationResult:
    limits = dict(thresholds or {})
    calibration = compare_external_validation(
        model_channel_sweep,
        calibration_external,
        wavelength_tolerance_nm=wavelength_tolerance_nm,
        thresholds=_comparison_thresholds(),
        allow_interpolation=allow_interpolation,
    ).comparisons
    holdout = compare_external_validation(
        model_channel_sweep,
        holdout_external,
        wavelength_tolerance_nm=wavelength_tolerance_nm,
        thresholds=_comparison_thresholds(),
        allow_interpolation=allow_interpolation,
    ).comparisons

    correction = _fit_wavelength_linear(calibration, center_wavelength_nm)
    calibration_corrected = _apply_correction(calibration, correction)
    calibration_corrected = _leave_one_out_corrected(calibration_corrected, center_wavelength_nm)
    holdout_corrected = _apply_correction(holdout, correction)

    summary = pd.DataFrame(
        [
            _metrics("calibration", "raw", calibration_corrected, "residual"),
            _metrics(
                "calibration",
                "wavelength_linear_in_sample",
                calibration_corrected,
                "linear_corrected_residual",
            ),
            _metrics(
                "calibration",
                "wavelength_linear_loo",
                calibration_corrected,
                "loo_linear_corrected_residual",
            ),
            _metrics("holdout", "raw", holdout_corrected, "residual"),
            _metrics(
                "holdout",
                "wavelength_linear_trained_on_calibration",
                holdout_corrected,
                "linear_corrected_residual",
            ),
        ]
    )

    rows: list[dict[str, Any]] = []
    reasons: list[str] = []
    holdout_row = summary[
        (summary["split"] == "holdout")
        & (summary["correction"] == "wavelength_linear_trained_on_calibration")
    ].iloc[0]
    loo_row = summary[
        (summary["split"] == "calibration")
        & (summary["correction"] == "wavelength_linear_loo")
    ].iloc[0]
    overlap = _identity_set(calibration_external).intersection(_identity_set(holdout_external))

    _requirement(
        rows,
        reasons,
        "no_holdout_identity_overlap",
        len(overlap) == 0,
        len(overlap),
        0,
    )
    _requirement(
        rows,
        reasons,
        "holdout_minimum_coverage",
        float(holdout_row["coverage"]) >= float(limits.get("minimum_holdout_coverage", 1.0)),
        float(holdout_row["coverage"]),
        float(limits.get("minimum_holdout_coverage", 1.0)),
    )
    _requirement(
        rows,
        reasons,
        "holdout_gsnr_rmse",
        float(holdout_row["rmse_db"]) <= float(limits.get("maximum_holdout_gsnr_rmse_db", 1.0)),
        float(holdout_row["rmse_db"]),
        float(limits.get("maximum_holdout_gsnr_rmse_db", 1.0)),
    )
    _requirement(
        rows,
        reasons,
        "holdout_gsnr_absolute_bias",
        abs(float(holdout_row["bias_db"]))
        <= float(limits.get("maximum_holdout_gsnr_absolute_bias_db", 0.75)),
        abs(float(holdout_row["bias_db"])),
        float(limits.get("maximum_holdout_gsnr_absolute_bias_db", 0.75)),
    )
    _requirement(
        rows,
        reasons,
        "calibration_loo_gsnr_rmse",
        float(loo_row["rmse_db"]) <= float(limits.get("maximum_calibration_loo_gsnr_rmse_db", 1.0)),
        float(loo_row["rmse_db"]),
        float(limits.get("maximum_calibration_loo_gsnr_rmse_db", 1.0)),
    )

    requirements = pd.DataFrame(rows)
    return HeldoutExternalValidationResult(
        calibration_corrected,
        holdout_corrected,
        summary,
        requirements,
        correction,
        not reasons,
        tuple(reasons),
    )
