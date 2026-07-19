#!/usr/bin/env python
from __future__ import annotations

import csv
import sys
from pathlib import Path

REQUIRED = [
    "source_id", "source_type", "tool_version", "configuration_hash", "date",
    "provenance_reference", "independent", "strategy", "spans", "wavelength_nm",
    "metric", "metric_unit", "reference_value", "reference_uncertainty",
]

def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "validation_data/external_reference.csv")
    if not path.exists():
        print(f"ERROR: not found: {path}")
        return 2
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in REQUIRED if c not in (reader.fieldnames or [])]
        if missing:
            print(f"ERROR: missing required columns for ISRS_SCL validator: {missing}")
            return 2
        rows = list(reader)
    if not rows:
        print("ERROR: no rows")
        return 2

    errors = []
    seen_source_ids = set()
    for i, row in enumerate(rows, start=2):
        sid = row.get("source_id", "")
        if sid in seen_source_ids:
            errors.append(f"line {i}: duplicate source_id {sid}")
        seen_source_ids.add(sid)
        for col in REQUIRED:
            if str(row.get(col, "")).strip() == "":
                errors.append(f"line {i}: blank {col}")
        if str(row.get("independent", "")).strip().lower() not in {"true", "1", "yes", "y"}:
            errors.append(f"line {i}: independent must be true")
        for col in ["spans", "wavelength_nm", "reference_value", "reference_uncertainty"]:
            try:
                float(row.get(col, ""))
            except ValueError:
                errors.append(f"line {i}: {col} must be numeric")
        if row.get("metric") == "gsnr_db" and row.get("metric_unit") != "dB":
            errors.append(f"line {i}: gsnr_db must use dB")

    if errors:
        print("ERROR: external_reference.csv is not ready:")
        for e in errors[:100]:
            print("  -", e)
        if len(errors) > 100:
            print(f"  ... and {len(errors) - 100} more")
        return 1

    spans = sorted({str(r.get("spans")) for r in rows})
    strategies = sorted({str(r.get("strategy")) for r in rows})
    bands = sorted({str(r.get("band")) for r in rows if "band" in r})
    print(f"OK: {path} has {len(rows)} complete ISRS_SCL-compatible external-validation rows.")
    print(f"spans={spans}; strategies={strategies}; bands={bands}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
