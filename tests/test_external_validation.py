import pandas as pd
import pytest

from isrs_scl.validation.external_validation import compare_external_validation


def model():
    return pd.DataFrame({"strategy": ["flat", "fixed", "adaptive"] * 3, "spans": [1, 1, 1, 2, 2, 2, 8, 8, 8], "wavelength_nm": [1495, 1550, 1595] * 3, "gsnr_db": [10, 15, 13, 9, 14, 12, 7, 12, 10], "nli_w": [1e-6] * 9})


def external(strategy="Flat", value=10.2):
    return pd.DataFrame([{ "source_id": "g1", "source_type": "GNPy", "tool_version": "2.10", "configuration_hash": "abc", "date": "2026-01-01", "provenance_reference": "doi:example", "independent": True, "strategy": strategy, "spans": 1, "band": "S", "wavelength_nm": 1495, "metric": "gsnr_db", "metric_unit": "dB", "reference_value": value, "reference_uncertainty": 0.2 }])


def test_case_insensitive_strategy_matching():
    result = compare_external_validation(model(), external(), thresholds={"minimum_sources": 1, "minimum_source_types": 1, "minimum_wavelengths_per_band": 0, "minimum_span_counts": 1, "maximum_gsnr_db_rmse": 1, "maximum_gsnr_db_absolute_bias": 1})
    assert result.comparisons.iloc[0]["matched"] == 1


def test_placeholder_rows_are_rejected():
    frame = external(); frame["reference_value"] = None
    with pytest.raises(ValueError): compare_external_validation(model(), frame)


def test_relative_nli_threshold_is_dimensionless():
    frame = external(strategy="flat"); frame["metric"] = "nli_w"; frame["metric_unit"] = "W"; frame["reference_value"] = 1.1e-6; frame["reference_uncertainty"] = 1e-7
    result = compare_external_validation(model(), frame, thresholds={"minimum_wavelengths_per_band": 0, "maximum_nli_relative_rmse": 0.2, "required_external_metrics": ("nli_w",)})
    metric = result.summary[(result.summary["level"] == "metric") & (result.summary["metric"] == "nli_w")].iloc[0]
    assert 0 <= metric["relative_rmse"] < 0.2
