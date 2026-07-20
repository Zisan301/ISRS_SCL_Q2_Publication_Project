#!/usr/bin/env python
"""Rebuild validation_data/external_reference.csv as a clean C-band-only GNPy set.

This intentionally removes:
- fake S-band rows mapped to nearest C-band channels,
- multiband rows using channel_power_dbm around -20 dBm,
- any row with "WARNING nearest differs" in notes.

It rebuilds 9 clean C-band rows from the flat GNPy raw outputs:
- wavelengths: 1535, 1550, 1560 nm
- spans: 1, 4, 8
- strategy: flat
- source_type: GNPy

Usage:
python tools\rebuild_clean_cband_external_reference.py --tool-version %GNPY_VERSION%
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import re
import shutil
from datetime import date, datetime
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

TARGETS_NM = [1535.0, 1550.0, 1560.0]
SPAN_FILES = {
    1: Path("external_validation/gnpy/raw_outputs/gnpy_scl_flat_1span.txt"),
    4: Path("external_validation/gnpy/raw_outputs/gnpy_scl_flat_4span.txt"),
    8: Path("external_validation/gnpy/raw_outputs/gnpy_scl_flat_8span.txt"),
}

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def parse_gnpy_channels(path: Path) -> list[dict[str, float]]:
    text = path.read_text(errors="replace")
    clean = re.sub(r"\x1b\[[0-9;]*m", "", text)
    rows = []
    for line in clean.splitlines():
        # Ch. #, frequency THz, power dBm, OSNR, SNR_NLI, GSNR
        m = re.match(r"\s*(\d+)\s+([0-9.]+)\s+(-?[0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*$", line)
        if not m:
            continue
        freq_thz = float(m.group(2))
        wavelength_nm = 299792458.0 / (freq_thz * 1e12) * 1e9
        rows.append({
            "channel": int(m.group(1)),
            "frequency_thz": freq_thz,
            "wavelength_nm": wavelength_nm,
            "channel_power_dbm": float(m.group(3)),
            "osnr_signal_bw_db": float(m.group(4)),
            "snr_nli_signal_bw_db": float(m.group(5)),
            "gsnr_signal_bw_db": float(m.group(6)),
        })
    if not rows:
        raise SystemExit(f"No GNPy channel table rows found in {path}")
    return rows

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-path", default="validation_data/external_reference.csv")
    ap.add_argument("--tool-version", required=True)
    ap.add_argument("--uncertainty", type=float, default=0.30)
    ap.add_argument("--max-target-error-nm", type=float, default=1.0)
    ap.add_argument("--expected-power-dbm", type=float, default=-2.0)
    ap.add_argument("--max-power-error-db", type=float, default=0.75)
    args = ap.parse_args()

    csv_path = Path(args.csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if csv_path.exists():
        backup = csv_path.with_name(
            f"external_reference_before_clean_cband_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        shutil.copy2(csv_path, backup)
        print(f"Backup saved: {backup}")

    out_rows = []
    for spans, raw_file in SPAN_FILES.items():
        if not raw_file.exists():
            raise SystemExit(
                f"Missing {raw_file}. Re-run your flat GNPy cases first or check the raw output path."
            )

        parsed = parse_gnpy_channels(raw_file)
        digest = sha256(raw_file)

        for target_nm in TARGETS_NM:
            item = min(parsed, key=lambda r: abs(r["wavelength_nm"] - target_nm))
            wavelength_error_nm = abs(item["wavelength_nm"] - target_nm)
            power_error_db = abs(item["channel_power_dbm"] - args.expected_power_dbm)

            if wavelength_error_nm > args.max_target_error_nm:
                raise SystemExit(
                    f"{raw_file}: nearest channel to {target_nm:.3f} nm is "
                    f"{item['wavelength_nm']:.3f} nm, error={wavelength_error_nm:.3f} nm. "
                    "Not clean enough for C-band rebuild."
                )

            if power_error_db > args.max_power_error_db:
                raise SystemExit(
                    f"{raw_file}: target {target_nm:.3f} nm has channel_power_dbm="
                    f"{item['channel_power_dbm']:.2f}, expected about {args.expected_power_dbm:.2f}. "
                    "This looks like the bad multiband -20 dBm mismatch; not using it."
                )

            target_tag = f"{target_nm:.0f}"
            actual_tag = f"{item['wavelength_nm']:.3f}".replace(".", "p")
            source_id = f"gnpy_clean_cband_{spans}span_flat_{target_tag}nm_actual_{actual_tag}nm_gsnr"

            out_rows.append({
                "source_id": source_id,
                "source_type": "GNPy",
                "tool_version": args.tool_version,
                "configuration_hash": digest,
                "date": str(date.today()),
                "provenance_reference": str(raw_file).replace("\\", "/"),
                "independent": "true",
                "strategy": "flat",
                "spans": str(spans),
                "band": "C",
                "wavelength_nm": f"{item['wavelength_nm']:.6f}",
                "metric": "gsnr_db",
                "metric_unit": "dB",
                "reference_value": f"{item['gsnr_signal_bw_db']:.2f}",
                "reference_uncertainty": f"{args.uncertainty:.2f}",
                "notes": (
                    f"Clean C-band rebuild. Requested {target_nm:.3f} nm; "
                    f"actual nearest GNPy channel {item['wavelength_nm']:.6f} nm; "
                    f"channel={int(item['channel'])}, frequency_thz={item['frequency_thz']:.5f}, "
                    f"channel_power_dbm={item['channel_power_dbm']:.2f}, "
                    f"osnr_signal_bw_db={item['osnr_signal_bw_db']:.2f}, "
                    f"snr_nli_signal_bw_db={item['snr_nli_signal_bw_db']:.2f}."
                ),
            })

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Wrote clean C-band external reference: {csv_path}")
    print(f"Rows written: {len(out_rows)}")
    print("Rows:")
    for row in out_rows:
        print(
            f"  {row['source_id']}: spans={row['spans']}, "
            f"wl={row['wavelength_nm']} nm, GSNR={row['reference_value']} dB"
        )

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
