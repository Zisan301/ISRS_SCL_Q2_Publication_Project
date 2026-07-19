#!/usr/bin/env python
"""Backup and reset validation_data/external_reference.csv to the ISRS_SCL schema header."""
from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

COLUMNS = [
    "source_id", "source_type", "tool_version", "configuration_hash", "date",
    "provenance_reference", "independent", "strategy", "spans", "band",
    "wavelength_nm", "metric", "metric_unit", "reference_value",
    "reference_uncertainty", "notes",
]

def main() -> int:
    path = Path("validation_data/external_reference.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = path.with_name(f"external_reference_backup_{stamp}.csv")
        shutil.copy2(path, backup)
        print(f"Backup saved: {backup}")
    path.write_text(",".join(COLUMNS) + "\n", encoding="utf-8")
    print(f"Reset: {path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
