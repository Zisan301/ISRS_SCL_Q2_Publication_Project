from pathlib import Path
import csv
import hashlib
from datetime import date

C = 299792458.0

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

RAW_FILES = [
    (1, Path("external_validation/gnpy/raw_outputs/gnpy_holdout_flat_1span.txt")),
    (4, Path("external_validation/gnpy/raw_outputs/gnpy_holdout_flat_4span.txt")),
    (8, Path("external_validation/gnpy/raw_outputs/gnpy_holdout_flat_8span.txt")),
]

TARGETS = [1540.0, 1545.0, 1555.0]
OUT = Path("validation_data/gnpy_holdout_1540_1545_1555.csv")
TOOL_VERSION = "2.14.1"

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def parse_raw(path):
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        parts = line.strip().split()
        if len(parts) < 6:
            continue
        if not parts[0].isdigit():
            continue
        try:
            ch = int(parts[0])
            freq_thz = float(parts[1])
            power_dbm = float(parts[2])
            osnr_db = float(parts[3])
            snr_nli_db = float(parts[4])
            gsnr_db = float(parts[5])
        except ValueError:
            continue

        wl_nm = C / (freq_thz * 1e12) * 1e9
        rows.append({
            "channel": ch,
            "frequency_thz": freq_thz,
            "wavelength_nm": wl_nm,
            "power_dbm": power_dbm,
            "osnr_db": osnr_db,
            "snr_nli_db": snr_nli_db,
            "gsnr_db": gsnr_db,
        })
    return rows

OUT.parent.mkdir(parents=True, exist_ok=True)
all_out = []

for spans, raw in RAW_FILES:
    print(f"\nRAW FILE: {raw}")
    if not raw.exists():
        raise SystemExit(f"Missing raw file: {raw}")

    parsed = parse_raw(raw)
    print("parsed rows:", len(parsed))

    if not parsed:
        print("ERROR: no rows parsed. Open this file and check the GNPy table.")
        raise SystemExit(1)

    digest = sha256(raw)

    for target in TARGETS:
        nearest = min(parsed, key=lambda row: abs(row["wavelength_nm"] - target))
        error = abs(nearest["wavelength_nm"] - target)
        print(
            f"target {target:.1f} nm -> nearest {nearest['wavelength_nm']:.6f} nm, "
            f"error {error:.3f} nm, GSNR {nearest['gsnr_db']:.2f} dB"
        )

        if error > 1.0:
            raise SystemExit(
                f"Nearest channel too far for target {target} nm in {raw}: "
                f"nearest={nearest['wavelength_nm']:.6f}, error={error:.3f} nm"
            )

        target_tag = str(target).replace(".", "p")
        actual_tag = f"{nearest['wavelength_nm']:.3f}".replace(".", "p")

        all_out.append({
            "source_id": f"gnpy_holdout_{spans}span_flat_requested_{target_tag}nm_actual_{actual_tag}nm_gsnr",
            "source_type": "GNPy",
            "tool_version": TOOL_VERSION,
            "configuration_hash": digest,
            "date": str(date.today()),
            "provenance_reference": str(raw).replace("\\", "/"),
            "independent": "true",
            "strategy": "flat",
            "spans": str(spans),
            "band": "C",
            "wavelength_nm": f"{nearest['wavelength_nm']:.6f}",
            "metric": "gsnr_db",
            "metric_unit": "dB",
            "reference_value": f"{nearest['gsnr_db']:.2f}",
            "reference_uncertainty": "0.30",
            "notes": (
                f"Held-out GNPy row. Requested {target:.3f} nm; "
                f"actual nearest channel {nearest['wavelength_nm']:.6f} nm; "
                f"channel={nearest['channel']}, frequency_thz={nearest['frequency_thz']:.5f}, "
                f"channel_power_dbm={nearest['power_dbm']:.2f}, "
                f"osnr_signal_bw_db={nearest['osnr_db']:.2f}, "
                f"snr_nli_signal_bw_db={nearest['snr_nli_db']:.2f}."
            ),
        })

with OUT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=COLUMNS)
    writer.writeheader()
    writer.writerows(all_out)

print(f"\nWROTE: {OUT}")
print(f"ROWS: {len(all_out)}")
