"""Cross-model and independent external-validation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from isrs_scl.link import LinkModel


def compare_nli_models(link: "LinkModel", power_dbm: float, n_spans: int) -> pd.DataFrame:
    from isrs_scl.fiber.amplification import dbm_to_w

    launch = np.full(link.grid.n_channels, float(dbm_to_w(power_dbm)))
    profile = link.evaluate(launch, n_spans, "power_profile_gn")
    semrau = link.evaluate(launch, n_spans, "semrau_closed_form")
    return pd.DataFrame(
        {
            "channel": np.arange(link.grid.n_channels),
            "band": link.grid.bands,
            "wavelength_nm": link.grid.wavelengths_nm,
            "profile_gsnr_db": profile.gsnr_db,
            "semrau_gsnr_db": semrau.gsnr_db,
            "difference_db": profile.gsnr_db - semrau.gsnr_db,
        }
    )


def _normalise_reference_columns(external: pd.DataFrame, metric_column: str) -> pd.DataFrame:
    frame = external.copy()
    canonical = f"reference_{metric_column}"
    if canonical not in frame.columns:
        candidates = [
            f"gnpy_{metric_column}",
            f"ssfm_{metric_column}",
            f"experimental_{metric_column}",
            f"measured_{metric_column}",
        ]
        found = [name for name in candidates if name in frame.columns]
        if len(found) != 1:
            raise ValueError(
                f"External CSV requires {canonical!r}, or exactly one of {candidates}"
            )
        frame = frame.rename(columns={found[0]: canonical})
        if "source" not in frame.columns:
            frame["source"] = found[0].removesuffix(f"_{metric_column}")
    if "source" not in frame.columns:
        frame["source"] = "external"
    return frame


def compare_external_validation(
    prediction: pd.DataFrame,
    external_csv: str | Path,
    metric_column: str = "gsnr_db",
    strategy: str | None = None,
    spans: int | None = None,
    wavelength_tolerance_nm: float = 0.05,
) -> pd.DataFrame:
    """Join predictions with independently generated GNPy/SSFM/measurement data.

    Minimum CSV columns are ``wavelength_nm`` and
    ``reference_<metric_column>``. Optional columns ``source``, ``strategy`` and
    ``spans`` permit multiple independent datasets in one file.
    """

    external_path = Path(external_csv)
    if not external_path.exists():
        raise FileNotFoundError(external_path)

    predicted = prediction.copy()
    if strategy is not None and "strategy" in predicted.columns:
        predicted = predicted[predicted["strategy"] == strategy]
    if spans is not None and "spans" in predicted.columns:
        predicted = predicted[predicted["spans"] == int(spans)]
    if predicted.empty:
        raise ValueError("No prediction rows remain after strategy/span filtering")
    if {"wavelength_nm", metric_column}.difference(predicted.columns):
        raise ValueError(
            f"Prediction table requires wavelength_nm and {metric_column}"
        )

    external = _normalise_reference_columns(pd.read_csv(external_path), metric_column)
    if strategy is not None and "strategy" in external.columns:
        external = external[external["strategy"] == strategy]
    if spans is not None and "spans" in external.columns:
        external = external[external["spans"] == int(spans)]
    if external.empty:
        raise ValueError("No external rows remain after strategy/span filtering")

    reference_column = f"reference_{metric_column}"
    required = {"wavelength_nm", reference_column, "source"}
    missing = required.difference(external.columns)
    if missing:
        raise ValueError(f"External validation CSV requires {sorted(missing)}")

    groups: list[pd.DataFrame] = []
    for source, reference in external.groupby("source", dropna=False):
        merged = pd.merge_asof(
            reference.sort_values("wavelength_nm"),
            predicted.sort_values("wavelength_nm"),
            on="wavelength_nm",
            direction="nearest",
            tolerance=float(wavelength_tolerance_nm),
            suffixes=("_reference", "_prediction"),
        )
        merged["source"] = str(source)
        merged["matched"] = merged[metric_column].notna()
        merged["validation_error_db"] = (
            merged[metric_column] - merged[reference_column]
        )
        merged["absolute_validation_error_db"] = merged[
            "validation_error_db"
        ].abs()
        groups.append(merged)

    return pd.concat(groups, ignore_index=True)


def summarize_external_validation(
    merged: pd.DataFrame,
    metric_column: str = "gsnr_db",
) -> pd.DataFrame:
    """Calculate coverage, bias, MAE, RMSE and worst error by source and overall."""

    reference_column = f"reference_{metric_column}"
    required = {
        "source",
        "matched",
        "validation_error_db",
        "absolute_validation_error_db",
        reference_column,
    }
    missing = required.difference(merged.columns)
    if missing:
        raise ValueError(f"Merged validation table lacks {sorted(missing)}")

    rows: list[dict] = []
    for source, frame in list(merged.groupby("source", dropna=False)) + [
        ("ALL", merged)
    ]:
        matched = frame[frame["matched"]].copy()
        errors = matched["validation_error_db"].to_numpy(dtype=float)
        rows.append(
            {
                "source": str(source),
                "reference_points": int(len(frame)),
                "matched_points": int(len(matched)),
                "coverage_fraction": float(len(matched) / max(len(frame), 1)),
                "bias_db": float(np.mean(errors)) if errors.size else float("nan"),
                "mae_db": float(np.mean(np.abs(errors)))
                if errors.size
                else float("nan"),
                "rmse_db": float(np.sqrt(np.mean(errors**2)))
                if errors.size
                else float("nan"),
                "maximum_abs_error_db": float(np.max(np.abs(errors)))
                if errors.size
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def external_validation_passes(summary: pd.DataFrame, cfg: dict) -> tuple[bool, list[str]]:
    """Evaluate external-validation thresholds from the configuration."""

    validation = cfg["validation"]
    overall = summary[summary["source"] == "ALL"]
    if overall.empty:
        return False, ["External validation summary has no ALL row"]
    row = overall.iloc[0]
    failures: list[str] = []
    if float(row["coverage_fraction"]) < float(
        validation["minimum_coverage_fraction"]
    ):
        failures.append("External validation wavelength coverage is insufficient")
    if not np.isfinite(float(row["rmse_db"])) or float(row["rmse_db"]) > float(
        validation["maximum_gsnr_rmse_db"]
    ):
        failures.append("External validation GSNR RMSE exceeds the configured limit")
    if not np.isfinite(float(row["maximum_abs_error_db"])) or float(
        row["maximum_abs_error_db"]
    ) > float(validation["maximum_gsnr_abs_error_db"]):
        failures.append("External validation maximum error exceeds the configured limit")
    if not np.isfinite(float(row["bias_db"])) or abs(float(row["bias_db"])) > float(
        validation["maximum_bias_db"]
    ):
        failures.append("External validation bias exceeds the configured limit")
    return not failures, failures