#!/usr/bin/env python
r"""Prepare a same-day C-band preliminary paper/evidence package.

This script collects the clean C-band validation files and creates:
- paper claim summary
- evidence README
- key result tables
- a compact markdown section you can paste into the paper

Usage:
python tools\prepare_today_cband_paper_package.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd


PACKAGE_DIR = Path("paper_package_cband_preliminary_20260720")
EXTERNAL_ONLY = Path("runs/q3-cband-external-only-2db-001")
BIAS = Path("runs/q3-cband-bias-diagnostic-001")
WAVELENGTH_BIAS = Path("runs/q3-cband-wavelength-bias-001")


def copy_if_exists(src: Path, dst_dir: Path) -> None:
    if src.exists():
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_dir / src.name)
        print(f"Copied {src}")
    else:
        print(f"Missing, skipped: {src}")


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
    (PACKAGE_DIR / "evidence").mkdir(exist_ok=True)
    (PACKAGE_DIR / "tables").mkdir(exist_ok=True)

    evidence_files = [
        Path("validation_data/external_reference.csv"),
        EXTERNAL_ONLY / "external_validation_comparisons.csv",
        EXTERNAL_ONLY / "external_validation_summary.csv",
        EXTERNAL_ONLY / "external_validation_requirements.csv",
        EXTERNAL_ONLY / "external_only_model_channel_sweep.csv",
        BIAS / "cband_bias_diagnostic_report.json",
        BIAS / "cband_bias_row_diagnostics.csv",
        BIAS / "cband_bias_by_span.csv",
        BIAS / "cband_bias_by_wavelength.csv",
        WAVELENGTH_BIAS / "cband_wavelength_bias_report.json",
        WAVELENGTH_BIAS / "cband_wavelength_bias_rows.csv",
        WAVELENGTH_BIAS / "cband_wavelength_bias_by_span.csv",
        WAVELENGTH_BIAS / "cband_wavelength_bias_by_wavelength.csv",
    ]

    for src in evidence_files:
        copy_if_exists(src, PACKAGE_DIR / "evidence")

    # Build a compact result table.
    wb_report = load_json(WAVELENGTH_BIAS / "cband_wavelength_bias_report.json")
    raw = wb_report.get("raw", {})
    global_corr = wb_report.get("global_constant_bias_correction", {}).get("metrics", {})
    linear = wb_report.get("linear_wavelength_bias_model", {})
    linear_in = linear.get("in_sample_metrics", {})
    linear_loo = linear.get("leave_one_out_metrics", {})

    result_rows = [
        {
            "analysis": "Raw model vs GNPy",
            "rmse_db": raw.get("rmse_db"),
            "bias_db": raw.get("bias_db"),
            "max_abs_error_db": raw.get("max_abs_error_db"),
            "note": "Uncalibrated comparison",
        },
        {
            "analysis": "Global offset diagnostic",
            "rmse_db": global_corr.get("rmse_db"),
            "bias_db": global_corr.get("bias_db"),
            "max_abs_error_db": global_corr.get("max_abs_error_db"),
            "note": "Single constant residual offset",
        },
        {
            "analysis": "Linear wavelength diagnostic, in-sample",
            "rmse_db": linear_in.get("rmse_db"),
            "bias_db": linear_in.get("bias_db"),
            "max_abs_error_db": linear_in.get("max_abs_error_db"),
            "note": "Residual fitted versus wavelength",
        },
        {
            "analysis": "Linear wavelength diagnostic, leave-one-out",
            "rmse_db": linear_loo.get("rmse_db"),
            "bias_db": linear_loo.get("bias_db"),
            "max_abs_error_db": linear_loo.get("max_abs_error_db"),
            "note": "Cross-validated diagnostic",
        },
    ]
    results_table = pd.DataFrame(result_rows)
    results_table.to_csv(PACKAGE_DIR / "tables" / "key_cband_validation_results.csv", index=False)

    formula = linear.get("formula", "predicted_residual_db = intercept + slope*(wavelength_nm - center)")
    intercept = linear.get("intercept_db")
    slope = linear.get("slope_db_per_nm")
    center = wb_report.get("center_wavelength_nm", 1550.0)

    paper_text = f"""# Preliminary C-band GNPy Validation Package

## Recommended title

A Preliminary GNPy-Assisted External Validation Workflow for ISRS-Aware GSNR Modeling in C-Band Coherent Optical Links

## Scope statement

This package supports a limited C-band preliminary validation claim only. It does not support a full S+C+L publication claim yet.

## Validation setup

The clean external validation set contains 9 C-band GNPy rows:

- Wavelength targets: 1535, 1550, and 1560 nm
- Span counts: 1, 4, and 8
- Strategy: flat launch
- Metric: GSNR in dB
- Source type: GNPy

## Main result

The external-only C-band diagnostic matched all 9 rows, giving 100% row coverage. The raw GSNR comparison had:

- RMSE: {raw.get("rmse_db", float("nan")):.3f} dB
- Bias: {raw.get("bias_db", float("nan")):.3f} dB
- Maximum absolute error: {raw.get("max_abs_error_db", float("nan")):.3f} dB

A global residual-offset diagnostic reduced RMSE to:

- RMSE: {global_corr.get("rmse_db", float("nan")):.3f} dB
- Bias: {global_corr.get("bias_db", float("nan")):.3f} dB

A wavelength-dependent residual diagnostic showed stronger agreement:

- In-sample RMSE: {linear_in.get("rmse_db", float("nan")):.3f} dB
- Leave-one-out RMSE: {linear_loo.get("rmse_db", float("nan")):.3f} dB
- Leave-one-out bias: {linear_loo.get("bias_db", float("nan")):.3f} dB

The fitted residual trend was:

```text
predicted_residual_db = {intercept:+.3f} + {slope:+.4f} * (wavelength_nm - {center:.1f})
```

where residual is defined as model GSNR minus GNPy reference GSNR.

## Honest interpretation

The raw model is consistently lower than GNPy, and the error increases with wavelength. This suggests the remaining mismatch is dominated by wavelength-dependent calibration or modeling assumptions, such as attenuation slope, dispersion, Raman/ISRS tilt, or amplifier wavelength dependence.

The wavelength-dependent correction is diagnostic, not final proof. It should be described as calibration analysis unless it is later validated on an independent held-out external set.

## Safe claim for same-day project

The project now provides a reproducible C-band GNPy-assisted validation workflow with complete 9-point row coverage. Raw external agreement is within approximately 2 dB, and a leave-one-out wavelength-bias diagnostic reduces the residual error to approximately {linear_loo.get("rmse_db", float("nan")):.3f} dB RMSE, indicating that the dominant remaining discrepancy is a systematic wavelength-dependent calibration trend.

## Unsafe claims to avoid

Do not claim:

- Full S+C+L validation
- Final Q2/Q3 journal readiness
- Robust adaptive gain improvement
- Fully calibrated physical parameter evidence
- Independent source diversity beyond GNPy

## Next research step

To strengthen this into a real Q3 submission, add one of the following:

1. Held-out GNPy rows at new C-band wavelengths, such as 1540, 1545, and 1555 nm.
2. A second independent source type, preferably SSFM.
3. Physical tuning of wavelength-dependent assumptions, then evaluate on held-out rows.
"""

    (PACKAGE_DIR / "PAPER_CLAIMS_AND_RESULTS.md").write_text(paper_text, encoding="utf-8")

    readme = """# Same-day C-band preliminary package

This folder contains the cleaned evidence for a limited C-band preliminary validation project.

Important:
This is not a full S+C+L Q2/Q3 evidence package.

Recommended next action:
Commit this package, then write the paper using the reduced C-band scope.
"""
    (PACKAGE_DIR / "README.md").write_text(readme, encoding="utf-8")

    print(f"Prepared package: {PACKAGE_DIR}")
    print(f"Main text: {PACKAGE_DIR / 'PAPER_CLAIMS_AND_RESULTS.md'}")
    print(f"Key table: {PACKAGE_DIR / 'tables' / 'key_cband_validation_results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
