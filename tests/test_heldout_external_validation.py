import pandas as pd
import pytest

from isrs_scl.validation.heldout_external_validation import run_heldout_external_validation


def _residual_db(wavelength_nm: float) -> float:
    return -1.5 - 0.05 * (wavelength_nm - 1550.0)


def _reference_gsnr_db(spans: int, wavelength_nm: float) -> float:
    return 30.0 - 0.8 * spans + 0.01 * (wavelength_nm - 1550.0)


def _model_sweep() -> pd.DataFrame:
    rows = []
    for spans in (1, 4, 8):
        for wavelength_nm in (1535.0, 1540.0, 1545.0, 1550.0, 1555.0, 1560.0):
            rows.append(
                {
                    "strategy": "flat",
                    "spans": spans,
                    "wavelength_nm": wavelength_nm,
                    "gsnr_db": _reference_gsnr_db(spans, wavelength_nm)
                    + _residual_db(wavelength_nm),
                }
            )
    return pd.DataFrame(rows)


def _external_rows(wavelengths: tuple[float, ...], source_prefix: str) -> pd.DataFrame:
    rows = []
    for spans in (1, 4, 8):
        for wavelength_nm in wavelengths:
            rows.append(
                {
                    "source_id": f"{source_prefix}_{spans}span_{wavelength_nm:.0f}nm",
                    "source_type": "GNPy",
                    "tool_version": "test-gnpy",
                    "configuration_hash": f"hash-{source_prefix}-{spans}-{wavelength_nm:.0f}",
                    "date": "2026-07-20",
                    "provenance_reference": f"test://{source_prefix}/{spans}/{wavelength_nm:.0f}",
                    "independent": True,
                    "strategy": "flat",
                    "spans": spans,
                    "band": "C",
                    "wavelength_nm": wavelength_nm,
                    "metric": "gsnr_db",
                    "metric_unit": "dB",
                    "reference_value": _reference_gsnr_db(spans, wavelength_nm),
                    "reference_uncertainty": 0.3,
                }
            )
    return pd.DataFrame(rows)


def test_wavelength_linear_correction_generalizes_to_heldout_rows():
    calibration = _external_rows((1535.0, 1550.0, 1560.0), "calibration")
    holdout = _external_rows((1540.0, 1545.0, 1555.0), "holdout")

    result = run_heldout_external_validation(
        _model_sweep(),
        calibration,
        holdout,
        thresholds={
            "maximum_holdout_gsnr_rmse_db": 0.1,
            "maximum_holdout_gsnr_absolute_bias_db": 0.1,
            "maximum_calibration_loo_gsnr_rmse_db": 0.1,
        },
    )

    assert result.passed
    assert result.reasons == ()
    assert result.correction.intercept_db == pytest.approx(-1.5, abs=1e-12)
    assert result.correction.slope_db_per_nm == pytest.approx(-0.05, abs=1e-12)
    holdout_row = result.summary[
        (result.summary["split"] == "holdout")
        & (result.summary["correction"] == "wavelength_linear_trained_on_calibration")
    ].iloc[0]
    assert holdout_row["coverage"] == pytest.approx(1.0)
    assert holdout_row["rmse_db"] < 1e-12


def test_holdout_overlap_is_rejected():
    calibration = _external_rows((1535.0, 1550.0, 1560.0), "same")

    result = run_heldout_external_validation(_model_sweep(), calibration, calibration)

    assert not result.passed
    failed = result.requirements[~result.requirements["passed"]]
    assert "no_holdout_identity_overlap" in set(failed["requirement"])
