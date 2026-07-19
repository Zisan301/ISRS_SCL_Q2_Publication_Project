#!/usr/bin/env python
"""Remove duplicate rows from validation_data/external_reference.csv.

Duplicates are detected by the ISRS_SCL validation identity:
source_id + strategy + spans + wavelength_nm + metric

Usage:
python tools\dedupe_external_reference.py validation_data\external_reference.csv
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from datetime import datetime

def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "validation_data/external_reference.csv")
    if not path.exists():
        print(f"ERROR: not found: {path}")
        return 2

    backup = path.with_name(path.stem + "_before_dedupe_" + datetime.now().strftime("%Y%m%d_%H%M%S") + path.suffix)
    backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if not fieldnames:
        print("ERROR: CSV has no header")
        return 2

    seen = set()
    kept = []
    removed = []

    for row in rows:
        key = (
            str(row.get("source_id", "")).strip(),
            str(row.get("strategy", "")).strip().lower(),
            str(row.get("spans", "")).strip(),
            str(row.get("wavelength_nm", "")).strip(),
            str(row.get("metric", "")).strip().lower(),
        )
        if key in seen:
            removed.append(row)
        else:
            seen.add(key)
            kept.append(row)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)

    print(f"Backup saved: {backup}")
    print(f"Rows before: {len(rows)}")
    print(f"Rows after:  {len(kept)}")
    print(f"Removed duplicates: {len(removed)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
