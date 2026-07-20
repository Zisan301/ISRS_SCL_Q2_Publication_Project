#!/usr/bin/env python
"""Summarize external validation comparison results after a smoke/debug run."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("comparison_csv", help="Path to runs/.../results/external_validation_comparisons.csv")
    args = ap.parse_args()

    path = Path(args.comparison_csv)
    if not path.exists():
        raise SystemExit(f"Not found: {path}")

    df = pd.read_csv(path)
    print(f"Rows: {len(df)}")
    if "matched" in df:
        print(f"Matched rows: {int(df['matched'].fillna(0).astype(int).sum())}")
    if "residual" in df:
        matched = df[df.get("matched", 0).fillna(0).astype(int) == 1].copy()
        if not matched.empty:
            residual_abs = matched["residual"].abs()
            print(f"Matched residual abs mean: {residual_abs.mean():.3f} dB")
            print(f"Matched residual abs max:  {residual_abs.max():.3f} dB")
            print()
            cols = [
                "source_id", "spans", "band", "wavelength_nm", "reference_value",
                "model_value", "residual", "wavelength_error_nm", "notes",
            ]
            cols = [c for c in cols if c in matched.columns]
            print(matched[cols].to_string(index=False))
        else:
            print("No matched rows.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
