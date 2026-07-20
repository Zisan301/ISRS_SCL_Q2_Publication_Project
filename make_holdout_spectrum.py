import json
from pathlib import Path

C = 299792458.0
targets_nm = [1540.0, 1545.0, 1555.0]
slot_width_hz = 50_000_000_000.0

spectrum = []
for wl_nm in targets_nm:
    center_hz = C / (wl_nm * 1e-9)
    spectrum.append({
        "f_min": center_hz - slot_width_hz / 2,
        "f_max": center_hz + slot_width_hz / 2,
        "baud_rate": 32000000000.0,
        "slot_width": slot_width_hz,
        "roll_off": 0.15,
        "tx_osnr": 40,
        "delta_pdb": 0
    })

path = Path("external_validation/gnpy/cases/scl_holdout_1540_1545_1555_flat_spectrum.json")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps({"spectrum": spectrum}, indent=2), encoding="utf-8")

print("Wrote", path)
for wl_nm, item in zip(targets_nm, spectrum):
    center = C / (wl_nm * 1e-9)
    print(f"{wl_nm:.1f} nm center={center:.5f} Hz f_min={item['f_min']:.5f} f_max={item['f_max']:.5f}")
