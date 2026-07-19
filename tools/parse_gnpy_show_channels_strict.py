#!/usr/bin/env python
"""Strict GNPy parser that refuses to duplicate one GNPy channel for multiple target wavelengths.

Use this instead of the earlier loose parser when building publication evidence.

Example:
python tools\parse_gnpy_show_channels_strict.py ^
  --raw-file external_validation\gnpy\raw_outputs\gnpy_scl_flat_1span.txt ^
  --tool-version 2.12.0 ^
  --spans 1 ^
  --strategy flat ^
  --targets 1495,1515,1525,1535,1550,1560,1570,1595,1610 ^
  --max-target-error-nm 1.0
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import re
from datetime import date
from pathlib import Path

COLUMNS = [
    "source_id", "source_type", "tool_version", "configuration_hash", "date",
    "provenance_reference", "independent", "strategy", "spans", "band",
    "wavelength_nm", "metric", "metric_unit", "reference_value",
    "reference_uncertainty", "notes",
]

def infer_band(wl: float) -> str:
    if wl < 1530:
        return "S"
    if wl < 1565:
        return "C"
    return "L"

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def parse_rows(path: Path):
    clean = re.sub(r"\x1b\[[0-9;]*m", "", path.read_text(errors="replace"))
    rows = []
    for line in clean.splitlines():
        m = re.match(r"\s*(\d+)\s+([0-9.]+)\s+(-?[0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*$", line)
        if not m:
            continue
        freq_thz = float(m.group(2))
        wl = 299792458 / (freq_thz * 1e12) * 1e9
        rows.append({
            "channel": int(m.group(1)),
            "frequency_thz": freq_thz,
            "wavelength_nm": wl,
            "power_dbm": float(m.group(3)),
            "osnr_signal_bw_db": float(m.group(4)),
            "snr_nli_signal_bw_db": float(m.group(5)),
            "gsnr_signal_bw_db": float(m.group(6)),
        })
    if not rows:
        raise SystemExit(f"No GNPy channel rows found in {path}")
    return rows

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-file", required=True)
    ap.add_argument("--csv-path", default="validation_data/external_reference.csv")
    ap.add_argument("--tool-version", required=True)
    ap.add_argument("--spans", required=True, type=int)
    ap.add_argument("--strategy", required=True, choices=["flat", "fixed", "adaptive"])
    ap.add_argument("--targets", required=True)
    ap.add_argument("--uncertainty", default=0.30, type=float)
    ap.add_argument("--max-target-error-nm", default=1.0, type=float)
    args = ap.parse_args()

    raw = Path(args.raw_file)
    parsed = parse_rows(raw)
    targets = [float(x.strip()) for x in args.targets.split(",") if x.strip()]
    used_channels: set[int] = set()
    digest = sha256(raw)

    out_rows = []
    skipped = []
    for target in targets:
        item = min(parsed, key=lambda row: abs(row["wavelength_nm"] - target))
        error = abs(item["wavelength_nm"] - target)

        if error > args.max_target_error_nm:
            skipped.append((target, item["wavelength_nm"], error, "too far"))
            continue
        if item["channel"] in used_channels:
            skipped.append((target, item["wavelength_nm"], error, "same GNPy channel already used"))
            continue

        used_channels.add(item["channel"])
        wl = item["wavelength_nm"]
        sid_wl = f"{target:.1f}".replace(".", "p")
        out_rows.append({
            "source_id": f"gnpy_{args.spans}span_{args.strategy}_requested_{sid_wl}nm_gsnr",
            "source_type": "GNPy",
            "tool_version": args.tool_version,
            "configuration_hash": digest,
            "date": str(date.today()),
            "provenance_reference": str(raw).replace("\\", "/"),
            "independent": "true",
            "strategy": args.strategy,
            "spans": args.spans,
            "band": infer_band(wl),
            "wavelength_nm": f"{wl:.6f}",
            "metric": "gsnr_db",
            "metric_unit": "dB",
            "reference_value": f"{item['gsnr_signal_bw_db']:.2f}",
            "reference_uncertainty": f"{args.uncertainty:.2f}",
            "notes": (
                f"Requested {target:.3f} nm; nearest GNPy channel {wl:.6f} nm; "
                f"channel={item['channel']}, frequency_thz={item['frequency_thz']:.5f}, "
                f"channel_power_dbm={item['power_dbm']:.2f}, osnr_signal_bw_db={item['osnr_signal_bw_db']:.2f}, "
                f"snr_nli_signal_bw_db={item['snr_nli_signal_bw_db']:.2f}."
            ),
        })

    csv_path = Path(args.csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not csv_path.exists() or csv_path.stat().st_size == 0
    if csv_path.exists() and csv_path.stat().st_size > 0:
        with csv_path.open(newline="", encoding="utf-8") as f:
            fieldnames = csv.DictReader(f).fieldnames or []
        if fieldnames != COLUMNS:
            backup = csv_path.with_suffix(".csv.bak")
            backup.write_text(csv_path.read_text(encoding="utf-8"), encoding="utf-8")
            csv_path.write_text("", encoding="utf-8")
            needs_header = True
            print(f"Replaced old header; backup saved: {backup}")

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if needs_header:
            writer.writeheader()
        writer.writerows(out_rows)

    print(f"Added {len(out_rows)} strict row(s) from {raw}")
    if skipped:
        print("Skipped targets:")
        for target, nearest, error, reason in skipped:
            print(f"  requested {target:.1f} nm -> nearest {nearest:.3f} nm, error {error:.3f} nm: {reason}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
