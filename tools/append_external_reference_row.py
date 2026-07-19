#!/usr/bin/env python
"""Append one independent external validation row."""
from __future__ import annotations

import argparse
import csv
import hashlib
from datetime import date
from pathlib import Path

COLUMNS = [
    "source_id",
    "source_type",
    "tool_version",
    "configuration_hash",
    "date",
    "independent",
    "provenance_reference",
    "span_count",
    "wavelength_nm",
    "band",
    "strategy",
    "metric",
    "metric_value",
    "metric_unit",
    "uncertainty",
    "notes",
]

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", default="validation_data/external_reference.csv")
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--source-type", required=True)
    parser.add_argument("--tool-version", required=True)
    parser.add_argument("--raw-file", required=True)
    parser.add_argument("--span-count", required=True, type=int)
    parser.add_argument("--wavelength-nm", required=True, type=float)
    parser.add_argument("--band", required=True, choices=["S", "C", "L"])
    parser.add_argument("--strategy", required=True, choices=["flat", "fixed", "adaptive"])
    parser.add_argument("--metric", required=True)
    parser.add_argument("--metric-value", required=True, type=float)
    parser.add_argument("--metric-unit", required=True)
    parser.add_argument("--uncertainty", required=True, type=float)
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    raw = Path(args.raw_file)
    if not raw.exists():
        raise SystemExit(f"raw file not found: {raw}")

    row = {
        "source_id": args.source_id,
        "source_type": args.source_type,
        "tool_version": args.tool_version,
        "configuration_hash": sha256(raw),
        "date": str(date.today()),
        "independent": "true",
        "provenance_reference": str(raw).replace("\\", "/"),
        "span_count": args.span_count,
        "wavelength_nm": args.wavelength_nm,
        "band": args.band,
        "strategy": args.strategy,
        "metric": args.metric,
        "metric_value": args.metric_value,
        "metric_unit": args.metric_unit,
        "uncertainty": args.uncertainty,
        "notes": args.notes,
    }

    needs_header = not csv_path.exists() or csv_path.stat().st_size == 0
    if csv_path.exists() and csv_path.stat().st_size > 0:
        with csv_path.open(newline="", encoding="utf-8") as f:
            existing_header = csv.DictReader(f).fieldnames or []
        if existing_header != COLUMNS:
            print("Replacing old/placeholder CSV header with the required publication schema.")
            backup = csv_path.with_suffix(".csv.bak")
            backup.write_text(csv_path.read_text(encoding="utf-8"), encoding="utf-8")
            csv_path.write_text("", encoding="utf-8")
            needs_header = True

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"Added row to {csv_path}: {args.source_id}")
    print(f"configuration_hash={row['configuration_hash']}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
