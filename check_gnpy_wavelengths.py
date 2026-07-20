from pathlib import Path
import re

files = [
    "external_validation/gnpy/raw_outputs/gnpy_scl_flat_1span.txt",
    "external_validation/gnpy/raw_outputs/gnpy_scl_flat_4span.txt",
    "external_validation/gnpy/raw_outputs/gnpy_scl_flat_8span.txt",
]

for file in files:
    path = Path(file)
    print("\nFILE:", file)

    if not path.exists():
        print("MISSING")
        continue

    text = path.read_text(errors="replace")
    clean = re.sub(r"\x1b\[[0-9;]*m", "", text)

    rows = []
    for line in clean.splitlines():
        m = re.match(r"\s*(\d+)\s+([0-9.]+)\s+(-?[0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*$", line)
        if m:
            ch = int(m.group(1))
            freq_thz = float(m.group(2))
            gsnr_db = float(m.group(6))
            wl = 299792458 / (freq_thz * 1e12) * 1e9
            rows.append((ch, wl, gsnr_db))

    print("parsed rows:", len(rows))

    if not rows:
        continue

    print("min wavelength:", min(x[1] for x in rows))
    print("max wavelength:", max(x[1] for x in rows))

    for target in [1540, 1545, 1555]:
        nearest = min(rows, key=lambda x: abs(x[1] - target))
        print(
            f"target {target} nm -> nearest {nearest[1]:.6f} nm, "
            f"error {abs(nearest[1]-target):.3f} nm, GSNR {nearest[2]:.2f} dB"
        )
