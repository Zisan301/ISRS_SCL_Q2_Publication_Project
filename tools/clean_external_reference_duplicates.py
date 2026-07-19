#!/usr/bin/env python
"""Remove duplicate source_id rows from validation_data/external_reference.csv.

Keeps the first occurrence and writes a backup before modifying the file.
"""
from __future__ import annotations

import csv
import shutil
from datetime import datetime
from pathlib import Path

def main() -> int:
    path = Path("validation_data/external_reference.csv")
    if not path.exists():
        raise SystemExit(f"not found: {path}")

    backup = path.with_name(f"external_reference_before_dedup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    shutil.copy2(path, backup)

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if "source_id" not in fieldnames:
        raise SystemExit("source_id column not found")

    seen = set()
    kept = []
    removed = []
    for row in rows:
        sid = row.get("source_id", "")
        if sid in seen:
            removed.append(sid)
            continue
        seen.add(sid)
        kept.append(row)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)

    print(f"Backup saved: {backup}")
    print(f"Rows before: {len(rows)}")
    print(f"Rows after:  {len(kept)}")
    print(f"Duplicates removed: {len(removed)}")
    if removed:
        print("Removed duplicate source_id values:")
        for sid in sorted(set(removed)):
            print(f"  - {sid}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
