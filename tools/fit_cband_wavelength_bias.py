#!/usr/bin/env python
"""Fit and validate a wavelength-dependent C-band GSNR bias model.

This is diagnostic only. It does not modify the ISRS-SCL model.

It compares three views:
1. Raw model-vs-GNPy residuals
2. Global constant bias correction
3. Linear wavelength-dependent residual correction
4. Leave-one-out linear correction, to avoid over-trusting an in-sample fit

Residual definition:
    residual = model_value - reference_value

Correction:
    corrected_model = model_value - predicted_residual

Usage:
python tools\fit_cband_wavelength_bias.py ^
  --comparisons runs\q3-cband-external-only-2db-001\external_validation_comparisons.csv ^
  --out-dir runs\q3-cband-wavelength-bias-001
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def metrics(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "n": 0,
            "rmse_db": float("nan"),
            "mae_db": float("nan"),
            "bias_db": float("nan"),
            "max_abs_error_db": float("nan"),
        }
    return {
        "n": int(values.size),
        "rmse_db": float(np.sqrt(np.mean(values ** 2))),
        "mae_db": float(np.mean(np.abs(values))),
        "bias_db": float(np.mean(values)),
        "max_abs_error_db": float(np.max(np.abs(values))),
    }


def fit_linear(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    # y = a + b*x
    b, a = np.polyfit(x, y, deg=1)
    return float(a), float(b)


def predict_linear(x: np.ndarray, a: float, b: float) -> np.ndarray:
    return a + b * x


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comparisons", required=True)
    ap.add_argument("--out-dir", default="runs/q3-cband-wavelength-bias-001")
    ap.add_argument("--center-wavelength-nm", type=float, default=1550.0)
    args = ap.parse_args()

    path = Path(args.comparisons)
    if not path.exists():
        raise SystemExit(f"Not found: {path}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(path)
    if "matched" in df.columns:
        df = df[pd.to_numeric(df["matched"], errors="coerce").fillna(0).astype(int) == 1].copy()
    df = df[pd.to_numeric(df.get("wavelength_nm"), errors="coerce").notna()].copy()
    df = df[pd.to_numeric(df.get("residual"), errors="coerce").notna()].copy()
    if df.empty:
        raise SystemExit("No matched residual rows found")

    df["wavelength_nm"] = pd.to_numeric(df["wavelength_nm"], errors="coerce")
    df["reference_value"] = pd.to_numeric(df["reference_value"], errors="coerce")
    df["model_value"] = pd.to_numeric(df["model_value"], errors="coerce")
    df["residual"] = pd.to_numeric(df["residual"], errors="coerce")
    df["x_nm_from_center"] = df["wavelength_nm"] - float(args.center_wavelength_nm)

    x = df["x_nm_from_center"].to_numpy(float)
    y = df["residual"].to_numpy(float)

    raw_metrics = metrics(y)

    # Constant/global correction
    global_pred = np.full_like(y, raw_metrics["bias_db"], dtype=float)
    df["global_predicted_residual"] = global_pred
    df["global_corrected_model_value"] = df["model_value"] - df["global_predicted_residual"]
    df["global_corrected_residual"] = df["global_corrected_model_value"] - df["reference_value"]
    global_metrics = metrics(df["global_corrected_residual"].to_numpy(float))

    # In-sample linear wavelength correction
    a, b = fit_linear(x, y)
    df["linear_predicted_residual"] = predict_linear(x, a, b)
    df["linear_corrected_model_value"] = df["model_value"] - df["linear_predicted_residual"]
    df["linear_corrected_residual"] = df["linear_corrected_model_value"] - df["reference_value"]
    linear_metrics = metrics(df["linear_corrected_residual"].to_numpy(float))

    # Leave-one-out linear correction
    loo_pred = np.full(len(df), np.nan, dtype=float)
    if len(df) >= 4:
        for i in range(len(df)):
            mask = np.ones(len(df), dtype=bool)
            mask[i] = False
            ai, bi = fit_linear(x[mask], y[mask])
            loo_pred[i] = predict_linear(np.array([x[i]]), ai, bi)[0]
    df["loo_linear_predicted_residual"] = loo_pred
    df["loo_linear_corrected_model_value"] = df["model_value"] - df["loo_linear_predicted_residual"]
    df["loo_linear_corrected_residual"] = df["loo_linear_corrected_model_value"] - df["reference_value"]
    loo_metrics = metrics(df["loo_linear_corrected_residual"].to_numpy(float))

    by_wavelength = (
        df.groupby("wavelength_nm")
        .agg(
            n=("residual", "count"),
            raw_bias_db=("residual", "mean"),
            raw_rmse_db=("residual", lambda s: float(np.sqrt(np.mean(np.asarray(s, float) ** 2)))),
            linear_corrected_bias_db=("linear_corrected_residual", "mean"),
            linear_corrected_rmse_db=("linear_corrected_residual", lambda s: float(np.sqrt(np.mean(np.asarray(s, float) ** 2)))),
            loo_linear_corrected_bias_db=("loo_linear_corrected_residual", "mean"),
            loo_linear_corrected_rmse_db=("loo_linear_corrected_residual", lambda s: float(np.sqrt(np.mean(np.asarray(s.dropna(), float) ** 2))) if s.dropna().size else float("nan")),
        )
        .reset_index()
    )

    by_span = (
        df.groupby("spans")
        .agg(
            n=("residual", "count"),
            raw_bias_db=("residual", "mean"),
            raw_rmse_db=("residual", lambda s: float(np.sqrt(np.mean(np.asarray(s, float) ** 2)))),
            linear_corrected_bias_db=("linear_corrected_residual", "mean"),
            linear_corrected_rmse_db=("linear_corrected_residual", lambda s: float(np.sqrt(np.mean(np.asarray(s, float) ** 2)))),
            loo_linear_corrected_bias_db=("loo_linear_corrected_residual", "mean"),
            loo_linear_corrected_rmse_db=("loo_linear_corrected_residual", lambda s: float(np.sqrt(np.mean(np.asarray(s.dropna(), float) ** 2))) if s.dropna().size else float("nan")),
        )
        .reset_index()
    )

    cols = [
        "source_id", "spans", "band", "wavelength_nm", "reference_value", "model_value",
        "residual", "global_corrected_residual", "linear_predicted_residual",
        "linear_corrected_residual", "loo_linear_predicted_residual",
        "loo_linear_corrected_residual",
    ]
    cols = [c for c in cols if c in df.columns]
    df[cols].to_csv(out / "cband_wavelength_bias_rows.csv", index=False)
    by_wavelength.to_csv(out / "cband_wavelength_bias_by_wavelength.csv", index=False)
    by_span.to_csv(out / "cband_wavelength_bias_by_span.csv", index=False)

    report = {
        "input_comparisons": str(path),
        "matched_rows": int(len(df)),
        "center_wavelength_nm": float(args.center_wavelength_nm),
        "residual_definition": "model_value - reference_value",
        "raw": raw_metrics,
        "global_constant_bias_correction": {
            "residual_offset_db": raw_metrics["bias_db"],
            "correction_rule": "corrected_model = model_value - residual_offset_db",
            "metrics": global_metrics,
        },
        "linear_wavelength_bias_model": {
            "formula": "predicted_residual_db = intercept_db + slope_db_per_nm * (wavelength_nm - center_wavelength_nm)",
            "intercept_db": a,
            "slope_db_per_nm": b,
            "correction_rule": "corrected_model = model_value - predicted_residual_db",
            "in_sample_metrics": linear_metrics,
            "leave_one_out_metrics": loo_metrics,
        },
        "interpretation": (
            "If leave-one-out linear corrected RMSE is much lower than raw RMSE, the mismatch is dominated "
            "by a wavelength-dependent modeling/calibration trend. Treat this as calibration evidence only "
            "after validating on held-out data."
        ),
    }
    (out / "cband_wavelength_bias_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("C-band wavelength-bias diagnostic")
    print(f"Input:  {path}")
    print(f"Output: {out}")
    print()
    print("Raw:")
    print(f"  RMSE: {raw_metrics['rmse_db']:.3f} dB")
    print(f"  bias: {raw_metrics['bias_db']:.3f} dB")
    print(f"  max abs error: {raw_metrics['max_abs_error_db']:.3f} dB")
    print()
    print("Global correction:")
    print(f"  residual offset: {raw_metrics['bias_db']:+.3f} dB")
    print(f"  corrected RMSE:  {global_metrics['rmse_db']:.3f} dB")
    print(f"  corrected bias:  {global_metrics['bias_db']:.3f} dB")
    print()
    print("Linear wavelength correction:")
    print(f"  predicted_residual = {a:+.3f} + {b:+.4f}*(wavelength_nm - {args.center_wavelength_nm:.1f})")
    print(f"  in-sample RMSE:     {linear_metrics['rmse_db']:.3f} dB")
    print(f"  in-sample bias:     {linear_metrics['bias_db']:.3f} dB")
    print(f"  LOO RMSE:           {loo_metrics['rmse_db']:.3f} dB")
    print(f"  LOO bias:           {loo_metrics['bias_db']:.3f} dB")
    print()
    print("By wavelength:")
    print(by_wavelength.to_string(index=False))
    print()
    print("By span:")
    print(by_span.to_string(index=False))
    print()
    print("Files written:")
    print(f"  {out / 'cband_wavelength_bias_report.json'}")
    print(f"  {out / 'cband_wavelength_bias_rows.csv'}")
    print(f"  {out / 'cband_wavelength_bias_by_wavelength.csv'}")
    print(f"  {out / 'cband_wavelength_bias_by_span.csv'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
