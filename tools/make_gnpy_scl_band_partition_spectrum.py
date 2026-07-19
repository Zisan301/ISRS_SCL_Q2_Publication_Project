#!/usr/bin/env python
"""Create a multiband-compatible S+C+L spectrum for GNPy."""
from __future__ import annotations

import json
from pathlib import Path

C = 299792458.0

def freq_hz(wavelength_nm: float) -> float:
    return C / (wavelength_nm * 1e-9)

def band_partition(label: str, wl_min_nm: float, wl_max_nm: float) -> dict:
    f_min = freq_hz(wl_max_nm)
    f_max = freq_hz(wl_min_nm)
    return {
        "f_min": f_min,
        "f_max": f_max,
        "baud_rate": 32e9,
        "slot_width": 50e9,
        "roll_off": 0.15,
        "tx_osnr": 40,
        "delta_pdb": 0,
        "label": label,
    }

def main() -> int:
    out = Path("external_validation/gnpy/cases/scl_band_partition_flat_spectrum.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    spectrum = {
        "spectrum": [
            band_partition("S-band-flat-32G", 1495.0, 1525.0),
            band_partition("C-band-flat-32G", 1535.0, 1560.0),
            band_partition("L-band-flat-32G", 1570.0, 1610.0),
        ]
    }
    out.write_text(json.dumps(spectrum, indent=2), encoding="utf-8")

    print(f"Wrote {out}")
    for item in spectrum["spectrum"]:
        print(
            f"{item['label']}: f_min={item['f_min']/1e12:.5f} THz, "
            f"f_max={item['f_max']/1e12:.5f} THz, spacing={item['slot_width']/1e9:.1f} GHz"
        )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
