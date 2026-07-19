#!/usr/bin/env python
"""Repair external_reference.csv after a failed 27-row GNPy parse.

What it does:
- backs up validation_data/external_reference.csv
- removes exact duplicate source_id rows
- keeps only unique real GNPy rows
- prints coverage summary by spans/strategy/band

This does NOT fake missing S/C/L rows.
"""
from __future__ import annotations

import csv
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

CSV_PATH = Path("validation_data/external_reference.csv")

def main() -> int:
    if not CSV_PATH.exists():
        print(f"ERROR: missing {CSV_PATH}")
        return 2

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = CSV_PATH.with_name(f"external_reference_before_dedupe_{stamp}.csv")
    shutil.copy2(CSV_PATH, backup)

    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if not fieldnames:
        print("ERROR: CSV has no header")
        return 2

    kept = []
    seen_source = set()
    dropped = []
    for row in rows:
        sid = row.get("source_id", "").strip()
        if sid in seen_source:
            dropped.append(row)
            continue
        seen_source.add(sid)
        kept.append(row)

    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)

    print(f"Backup saved: {backup}")
    print(f"Kept {len(kept)} unique row(s); dropped {len(dropped)} duplicate row(s).")

    by_span = Counter(row.get("spans", "") for row in kept)
    by_strategy = Counter(row.get("strategy", "") for row in kept)
    by_band = Counter(row.get("band", "") for row in kept)

    print(f"Coverage by spans: {dict(by_span)}")
    print(f"Coverage by strategy: {dict(by_strategy)}")
    print(f"Coverage by band: {dict(by_band)}")

    if len(kept) < 27:
        print()
        print("NOTE: This is valid partial external evidence, not the full 27-row S+C+L set.")
        print("Reason: the GNPy output did not contain enough unique target wavelengths.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
