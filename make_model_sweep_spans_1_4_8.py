from pathlib import Path
import numpy as np
import pandas as pd

from isrs_scl.system.parameters import load_config, validate_config
from isrs_scl.system.grid import build_grid
from isrs_scl.link import LinkModel
from isrs_scl.fiber.amplification import dbm_to_w

cfg = load_config("config_fast_preview.yaml")

# Important: NOT smoke. Smoke mode clamps spans to 1/2.
cfg["run"]["mode"] = "debug"
cfg["grid"]["mode"] = "paper_240_subset"
cfg["grid"]["subset_channels"] = 80

# Important: validation needs spans 1, 4, 8.
cfg["fiber"]["max_spans"] = 8
cfg["optimization"]["target_spans"] = 8
cfg["optimization"]["evaluation_spans"] = [1, 4, 8]

# Keep this fast; we only need model GSNR channel rows.
cfg["validation"]["require_uncertainty_analysis"] = False
if "uncertainty" in cfg:
    cfg["uncertainty"]["enabled"] = False

validate_config(cfg)

grid = build_grid(cfg["grid"])
link = LinkModel(grid, cfg)

flat_dbm = float(cfg["launch"]["flat_power_dbm_per_channel"])
flat_power_w = np.full(grid.n_channels, dbm_to_w(flat_dbm))

frames = []
for result in link.sweep_spans(flat_power_w, max_spans=8):
    if result.n_spans not in {1, 4, 8}:
        continue

    frame = result.to_frame(grid)
    frame.insert(0, "strategy", "flat")
    frame.insert(1, "spans", result.n_spans)
    frames.append(frame)

out = pd.concat(frames, ignore_index=True)

out_dir = Path("runs/q3-heldout-model-001/results")
out_dir.mkdir(parents=True, exist_ok=True)

out_path = out_dir / "channel_performance_holdout_spans_1_4_8.csv"
out.to_csv(out_path, index=False)

print("WROTE:", out_path)
print("ROWS:", len(out))
print("SPANS:", sorted(out["spans"].unique()))
print("WAVELENGTH MIN:", out["wavelength_nm"].min())
print("WAVELENGTH MAX:", out["wavelength_nm"].max())

for spans in [1, 4, 8]:
    sub = out[(out["strategy"] == "flat") & (out["spans"] == spans)]
    print(f"\nSPAN {spans}")
    for target in [1535, 1540, 1545, 1550, 1555, 1560]:
        nearest = sub.iloc[(sub["wavelength_nm"] - target).abs().argsort().iloc[0]]
        print(
            f"target {target} -> nearest {nearest['wavelength_nm']:.6f}, "
            f"gsnr {nearest['gsnr_db']:.3f}"
        )
