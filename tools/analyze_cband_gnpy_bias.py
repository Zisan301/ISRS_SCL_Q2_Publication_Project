#!/usr/bin/env python
"""Diagnose clean C-band GNPy-vs-model GSNR bias.

Reads external_validation_comparisons.csv and reports:
- raw RMSE/MAE/bias/max error
- bias-corrected preview using one global offset
- residual by span
- residual by wavelength
- recommendation for next physical tuning step

This does NOT modify your model. It is diagnostic only.

Usage:
python tools\analyze_cband_gnpy_bias.py ^
  --comparisons runs\q3-cband-external-only-2db-001\external_validation_comparisons.csv ^
  --out-dir runs\q3-cband-bias-diagnostic-001
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise SystemExit(f"Could not find any column from {candidates}. Available columns: {list(df.columns)}")


def _metrics(residual: pd.Series) -> dict[str, float]:
    r = pd.to_numeric(residual, errors="coerce").dropna().to_numpy(float)
    if r.size == 0:
        return {
            "n": 0,
            "rmse_db": float("nan"),
            "mae_db": float("nan"),
            "bias_db": float("nan"),
            "max_abs_error_db": float("nan"),
        }
    return {
        "n": int(r.size),
        "rmse_db": float(np.sqrt(np.mean(r**2))),
        "mae_db": float(np.mean(np.abs(r))),
        "bias_db": float(np.mean(r)),
        "max_abs_error_db": float(np.max(np.abs(r))),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comparisons", required=True)
    ap.add_argument("--out-dir", default="runs/q3-cband-bias-diagnostic-001")
    ap.add_argument("--target-rmse-db", type=float, default=1.0)
    ap.add_argument("--target-bias-db", type=float, default=1.0)
    args = ap.parse_args()

    comparison_path = Path(args.comparisons)
    if not comparison_path.exists():
        raise SystemExit(f"Not found: {comparison_path}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(comparison_path)
    matched_col = "matched" if "matched" in df.columns else None
    if matched_col:
        df = df[pd.to_numeric(df[matched_col], errors="coerce").fillna(0).astype(int) == 1].copy()

    if df.empty:
        raise SystemExit("No matched rows found in comparisons file")

    residual_col = _find_col(df, ["residual", "residual_db", "gsnr_residual_db"])
    ref_col = _find_col(df, ["reference_value", "reference_gsnr_db"])
    model_col = _find_col(df, ["model_value", "model_gsnr_db"])

    df["residual"] = pd.to_numeric(df[residual_col], errors="coerce")
    df["reference_value"] = pd.to_numeric(df[ref_col], errors="coerce")
    df["model_value"] = pd.to_numeric(df[model_col], errors="coerce")
    df["abs_residual"] = df["residual"].abs()

    raw = _metrics(df["residual"])
    global_offset_to_add_to_model_db = -raw["bias_db"]
    df["bias_corrected_model_value"] = df["model_value"] + global_offset_to_add_to_model_db
    df["bias_corrected_residual"] = df["bias_corrected_model_value"] - df["reference_value"]
    corrected = _metrics(df["bias_corrected_residual"])

    span_summary = (
        df.groupby("spans", dropna=False)
        .agg(
            n=("residual", "count"),
            rmse_db=("residual", lambda x: float(np.sqrt(np.mean(np.asarray(x, float) ** 2)))),
            bias_db=("residual", "mean"),
            mae_db=("abs_residual", "mean"),
            max_abs_error_db=("abs_residual", "max"),
        )
        .reset_index()
        .sort_values("spans")
    )

    wavelength_summary = (
        df.groupby("wavelength_nm", dropna=False)
        .agg(
            n=("residual", "count"),
            rmse_db=("residual", lambda x: float(np.sqrt(np.mean(np.asarray(x, float) ** 2)))),
            bias_db=("residual", "mean"),
            mae_db=("abs_residual", "mean"),
            max_abs_error_db=("abs_residual", "max"),
        )
        .reset_index()
        .sort_values("wavelength_nm")
    )

    # Simple heuristic recommendation.
    span_bias_range = float(span_summary["bias_db"].max() - span_summary["bias_db"].min()) if len(span_summary) else float("nan")
    wavelength_bias_range = float(wavelength_summary["bias_db"].max() - wavelength_summary["bias_db"].min()) if len(wavelength_summary) else float("nan")

    if abs(raw["bias_db"]) > args.target_bias_db and span_bias_range < 0.75 and wavelength_bias_range < 0.75:
        recommendation = (
            "The error is mostly a global GSNR offset. First investigate amplifier/noise calibration "
            "or a documented global calibration offset. Do not retune many physical parameters yet."
        )
    elif span_bias_range >= wavelength_bias_range:
        recommendation = (
            "The error changes mostly with span count. Investigate span loss, amplifier noise figure, "
            "ASE accumulation, and NLI accumulation assumptions."
        )
    else:
        recommendation = (
            "The error changes mostly with wavelength. Investigate fiber attenuation slope, dispersion, "
            "Raman/ISRS tilt, and wavelength-dependent amplifier assumptions."
        )

    row_cols = [
        "source_id",
        "spans",
        "band",
        "wavelength_nm",
        "reference_value",
        "model_value",
        "residual",
        "bias_corrected_model_value",
        "bias_corrected_residual",
        "abs_residual",
    ]
    row_cols = [c for c in row_cols if c in df.columns]

    df[row_cols].to_csv(out / "cband_bias_row_diagnostics.csv", index=False)
    span_summary.to_csv(out / "cband_bias_by_span.csv", index=False)
    wavelength_summary.to_csv(out / "cband_bias_by_wavelength.csv", index=False)

    report = {
        "input_comparisons": str(comparison_path),
        "matched_rows_used": int(len(df)),
        "raw": raw,
        "global_offset_to_add_to_model_db": global_offset_to_add_to_model_db,
        "bias_corrected_preview": corrected,
        "span_bias_range_db": span_bias_range,
        "wavelength_bias_range_db": wavelength_bias_range,
        "target_rmse_db": args.target_rmse_db,
        "target_bias_db": args.target_bias_db,
        "recommendation": recommendation,
        "important_note": (
            "Bias-corrected preview is diagnostic. If you use a fitted offset in the paper, "
            "you must call it calibration and validate on held-out external rows."
        ),
    }
    (out / "cband_bias_diagnostic_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("C-band GNPy bias diagnostic")
    print(f"Input:  {comparison_path}")
    print(f"Output: {out}")
    print()
    print("Raw comparison:")
    print(f"  matched rows:  {raw['n']}")
    print(f"  RMSE:          {raw['rmse_db']:.3f} dB")
    print(f"  MAE:           {raw['mae_db']:.3f} dB")
    print(f"  bias:          {raw['bias_db']:.3f} dB")
    print(f"  max abs error: {raw['max_abs_error_db']:.3f} dB")
    print()
    print("Global-offset diagnostic preview:")
    print(f"  offset to add to model: {global_offset_to_add_to_model_db:+.3f} dB")
    print(f"  corrected RMSE:         {corrected['rmse_db']:.3f} dB")
    print(f"  corrected bias:         {corrected['bias_db']:.3f} dB")
    print(f"  corrected max abs err:  {corrected['max_abs_error_db']:.3f} dB")
    print()
    print("By span:")
    print(span_summary.to_string(index=False))
    print()
    print("By wavelength:")
    print(wavelength_summary.to_string(index=False))
    print()
    print("Recommendation:")
    print(" ", recommendation)
    print()
    print("Files written:")
    print(f"  {out / 'cband_bias_row_diagnostics.csv'}")
    print(f"  {out / 'cband_bias_by_span.csv'}")
    print(f"  {out / 'cband_bias_by_wavelength.csv'}")
    print(f"  {out / 'cband_bias_diagnostic_report.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
