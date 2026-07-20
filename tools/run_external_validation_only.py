#!/usr/bin/env python
"""Run external validation only, without heavy optimization/waveform/uncertainty stages.

Purpose:
Your smoke run is fast but smoke mode reduces the grid/span range, so it may only
match 2 of your 9 C-band rows. This script builds the configured grid/link and
computes flat GSNR sweeps only, then compares validation_data/external_reference.csv.

It is a diagnostic for a same-day C-band preliminary paper/package, not a full
publication gate.

Usage:
python tools\run_external_validation_only.py --config config_q2_final.yaml --out-dir runs\q3-cband-external-only-001
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from isrs_scl.fiber.amplification import dbm_to_w
from isrs_scl.link import LinkModel
from isrs_scl.system.grid import build_grid
from isrs_scl.system.parameters import load_config, validate_config
from isrs_scl.validation.external_validation import compare_external_validation, load_external_validation


def flat_channel_sweep(cfg: dict) -> pd.DataFrame:
    grid = build_grid(cfg["grid"])
    link = LinkModel(grid, cfg)
    flat_dbm = np.full(grid.n_channels, float(cfg["launch"]["flat_power_dbm_per_channel"]))
    rows = []
    for result in link.sweep_spans(dbm_to_w(flat_dbm)):
        frame = result.to_frame(grid)
        frame.insert(0, "strategy", "flat")
        frame.insert(1, "spans", int(result.n_spans))
        rows.append(frame)
    if not rows:
        raise RuntimeError("No model sweep rows were produced")
    return pd.concat(rows, ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_q2_final.yaml")
    ap.add_argument("--out-dir", default="runs/q3-cband-external-only-001")
    ap.add_argument("--external-csv", default=None)
    ap.add_argument("--wavelength-tolerance-nm", type=float, default=0.25)
    ap.add_argument("--allow-interpolation", action="store_true")
    ap.add_argument("--max-rmse-db", type=float, default=1.5)
    ap.add_argument("--max-bias-db", type=float, default=1.5)
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)
    validate_config(cfg, base_dir=cfg_path.parent)

    external_csv = Path(args.external_csv or cfg["validation"]["external_reference_csv"])
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    external = load_external_validation(external_csv)
    c_only = external[external["band"].astype(str).str.upper() == "C"].copy()
    if c_only.empty:
        raise RuntimeError("No C-band external validation rows found")

    channel_sweep = flat_channel_sweep(cfg)
    channel_sweep.to_csv(out / "external_only_model_channel_sweep.csv", index=False)

    thresholds = {
        "minimum_external_coverage": 0.90,
        "minimum_sources": 1,
        "minimum_source_types": 1,
        "minimum_wavelengths_per_band": 0,
        "minimum_span_counts": 1,
        "maximum_gsnr_db_rmse": args.max_rmse_db,
        "maximum_gsnr_db_absolute_bias": args.max_bias_db,
        "required_external_metrics": ("gsnr_db",),
    }

    result = compare_external_validation(
        channel_sweep,
        c_only,
        wavelength_tolerance_nm=args.wavelength_tolerance_nm,
        thresholds=thresholds,
        allow_interpolation=args.allow_interpolation,
    )

    result.comparisons.to_csv(out / "external_validation_comparisons.csv", index=False)
    result.summary.to_csv(out / "external_validation_summary.csv", index=False)
    result.requirements.to_csv(out / "external_validation_requirements.csv", index=False)

    overall = result.summary[result.summary["level"] == "overall"].iloc[0]
    metric = result.summary[
        (result.summary["level"] == "metric") & (result.summary["metric"] == "gsnr_db")
    ].iloc[0]

    print("External-only C-band diagnostic")
    print(f"External CSV: {external_csv}")
    print(f"Output dir:   {out}")
    print()
    print(f"Requested rows: {int(overall['requested_rows'])}")
    print(f"Matched rows:   {int(overall['matched_rows'])}")
    print(f"Coverage:       {float(overall['coverage']):.3f}")
    print(f"GSNR RMSE:      {float(metric['rmse']):.3f} dB")
    print(f"GSNR MAE:       {float(metric['mae']):.3f} dB")
    print(f"GSNR bias:      {float(metric['bias']):.3f} dB")
    print(f"Max abs error:  {float(metric['max_abs_error']):.3f} dB")
    print()
    print("Requirements:")
    print(result.requirements.to_string(index=False))
    print()

    bad = result.comparisons[result.comparisons["matched"].astype(int) == 0]
    if not bad.empty:
        print("Unmatched rows:")
        cols = ["source_id", "spans", "band", "wavelength_nm", "matched_wavelength_nm", "wavelength_error_nm"]
        cols = [c for c in cols if c in bad.columns]
        print(bad[cols].to_string(index=False))
        print()

    if result.passed:
        print("RESULT: PASS for a limited C-band external-validation diagnostic.")
    else:
        print("RESULT: NOT PASS yet.")
        print("Reasons:")
        for reason in result.reasons:
            print(f"  - {reason}")

    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
