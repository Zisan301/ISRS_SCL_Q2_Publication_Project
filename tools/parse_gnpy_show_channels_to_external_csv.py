#!/usr/bin/env python
"""Strict parser for GNPy --show-channels output.

Unlike the earlier parser, this one SKIPS a requested wavelength if GNPy did not
actually produce a nearby channel. This prevents false S/L evidence.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import re
from datetime import date
from pathlib import Path

COLUMNS = [
    "source_id",
    "source_type",
    "tool_version",
    "configuration_hash",
    "date",
    "provenance_reference",
    "independent",
    "strategy",
    "spans",
    "band",
    "wavelength_nm",
    "metric",
    "metric_unit",
    "reference_value",
    "reference_uncertainty",
    "notes",
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
    text = path.read_text(errors="replace")
    clean = re.sub(r"\x1b\[[0-9;]*m", "", text)
    rows = []
    for line in clean.splitlines():
        m = re.match(r"\s*(\d+)\s+([0-9.]+)\s+(-?[0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*$", line)
        if not m:
            continue
        ch = int(m.group(1))
        freq_thz = float(m.group(2))
        power_dbm = float(m.group(3))
        osnr_db = float(m.group(4))
        snr_nli_db = float(m.group(5))
        gsnr_db = float(m.group(6))
        wl = 299792458 / (freq_thz * 1e12) * 1e9
        rows.append({
            "channel": ch,
            "frequency_thz": freq_thz,
            "wavelength_nm": wl,
            "power_dbm": power_dbm,
            "osnr_signal_bw_db": osnr_db,
            "snr_nli_signal_bw_db": snr_nli_db,
            "gsnr_signal_bw_db": gsnr_db,
        })
    if not rows:
        raise SystemExit(f"No GNPy channel rows found in {path}")
    return rows

def ensure_header(path: Path) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        return True
    with path.open(newline="", encoding="utf-8") as f:
        fieldnames = csv.DictReader(f).fieldnames or []
    if fieldnames != COLUMNS:
        backup = path.with_suffix(".csv.bak")
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        path.write_text("", encoding="utf-8")
        print(f"Replaced old header; backup saved to {backup}")
        return True
    return False

def existing_source_ids(path: Path) -> set[str]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with path.open(newline="", encoding="utf-8") as f:
        return {row.get("source_id", "") for row in csv.DictReader(f)}

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
    if not raw.exists():
        raise SystemExit(f"raw file not found: {raw}")

    parsed = parse_rows(raw)
    targets = [float(x.strip()) for x in args.targets.split(",") if x.strip()]
    digest = sha256(raw)
    csv_path = Path(args.csv_path)
    needs_header = ensure_header(csv_path)
    used_ids = existing_source_ids(csv_path)

    out_rows = []
    skipped = []
    for target in targets:
        item = min(parsed, key=lambda row: abs(row["wavelength_nm"] - target))
        error = abs(item["wavelength_nm"] - target)
        if error > args.max_target_error_nm:
            skipped.append((target, item["wavelength_nm"], error))
            continue

        wl = item["wavelength_nm"]
        target_tag = f"{target:.1f}".replace(".", "p")
        actual_tag = f"{wl:.3f}".replace(".", "p")
        source_id = f"gnpy_{args.spans}span_{args.strategy}_requested_{target_tag}nm_actual_{actual_tag}nm_gsnr"
        if source_id in used_ids:
            print(f"SKIP duplicate existing source_id: {source_id}")
            continue
        used_ids.add(source_id)

        out_rows.append({
            "source_id": source_id,
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
                f"Requested {target:.3f} nm; actual nearest GNPy channel {wl:.6f} nm; "
                f"channel={item['channel']}, frequency_thz={item['frequency_thz']:.5f}, "
                f"channel_power_dbm={item['power_dbm']:.2f}, osnr_signal_bw_db={item['osnr_signal_bw_db']:.2f}, "
                f"snr_nli_signal_bw_db={item['snr_nli_signal_bw_db']:.2f}."
            ),
        })

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if needs_header:
            writer.writeheader()
        writer.writerows(out_rows)

    print(f"Added {len(out_rows)} valid row(s) to {csv_path}")
    if skipped:
        print("Skipped requested wavelengths because GNPy did not output a nearby channel:")
        for target, nearest, error in skipped:
            print(f"  - requested {target:.3f} nm, nearest {nearest:.3f} nm, error {error:.3f} nm")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
