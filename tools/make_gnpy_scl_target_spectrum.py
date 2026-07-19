#!/usr/bin/env python
"""Create a GNPy spectrum JSON with the requested S+C+L target wavelengths.

The spectrum uses one channel per target wavelength:
S: 1495, 1515, 1525 nm
C: 1535, 1550, 1560 nm
L: 1570, 1595, 1610 nm

Each channel uses:
- baud_rate: 32e9
- slot_width: 50e9
- roll_off: 0.15
- tx_osnr: 40 dB
- delta_pdb: 0 for flat launch profile

Run:
python tools\make_gnpy_scl_target_spectrum.py
"""
from __future__ import annotations

import json
from pathlib import Path

C = 299792458.0

TARGETS_NM = [1495.0, 1515.0, 1525.0, 1535.0, 1550.0, 1560.0, 1570.0, 1595.0, 1610.0]

def main() -> int:
    out = Path("external_validation/gnpy/cases/scl_9_targets_flat_spectrum.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    spectrum = []
    for wl_nm in TARGETS_NM:
        f_hz = C / (wl_nm * 1e-9)
        spectrum.append({
            "f_min": f_hz,
            "f_max": f_hz,
            "baud_rate": 32e9,
            "slot_width": 50e9,
            "roll_off": 0.15,
            "tx_osnr": 40,
            "delta_pdb": 0
        })

    out.write_text(json.dumps({"spectrum": spectrum}, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    print("Targets:")
    for wl_nm in TARGETS_NM:
        f_thz = C / (wl_nm * 1e-9) / 1e12
        print(f"  {wl_nm:8.3f} nm -> {f_thz:10.5f} THz")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
