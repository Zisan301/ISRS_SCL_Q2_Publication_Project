"""Generate a validated, total-power-conserving fixed pre-emphasis baseline."""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from isrs_scl.fiber.amplification import dbm_to_w
from isrs_scl.link import LinkModel
from isrs_scl.optimization.adaptive_isrs import fixed_preemphasis_profile_dbm
from isrs_scl.optimization.constraints import total_power_w_from_dbm
from isrs_scl.system.grid import build_grid
from isrs_scl.system.parameters import load_config
from isrs_scl.validation.reproducibility import build_run_manifest, finalize_manifest, prepare_run_directory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", default="config_q2_final.yaml"); parser.add_argument("--run-id", required=True); parser.add_argument("--output-root", default="runs/partial"); parser.add_argument("--overwrite", action="store_true"); args = parser.parse_args()
    cfg = load_config(args.config); paths = prepare_run_directory(args.output_root, args.run_id, overwrite=args.overwrite); grid = build_grid(cfg["grid"]); link = LinkModel(grid, cfg)
    profile = fixed_preemphasis_profile_dbm(grid.frequencies_hz, float(cfg["launch"]["flat_power_dbm_per_channel"]), float(cfg["launch"]["fixed_preemphasis_s_to_l_db"]), float(cfg["launch"]["min_power_dbm_per_channel"]), float(cfg["launch"]["max_power_dbm_per_channel"]))
    target = grid.n_channels * float(dbm_to_w(cfg["launch"]["flat_power_dbm_per_channel"]))
    if not np.isclose(total_power_w_from_dbm(profile), target, rtol=1e-10): raise RuntimeError("Fixed profile does not conserve total power")
    pd.DataFrame({"channel": np.arange(grid.n_channels), "wavelength_nm": grid.wavelengths_nm, "launch_power_dbm": profile}).to_csv(paths.results / "fixed_profile.csv", index=False)
    tables=[]
    for result in link.sweep_spans(dbm_to_w(profile)):
        frame=result.to_frame(grid); frame.insert(0,"spans",result.n_spans); frame.insert(1,"strategy","fixed"); tables.append(frame)
    pd.concat(tables,ignore_index=True).to_csv(paths.results/"fixed_channel_performance.csv",index=False); finalize_manifest(build_run_manifest(cfg,run_root=paths.root),paths.root,paths.metadata/"RUN_MANIFEST.json")
    print(f"{paths.root}\nPartial evidence only until the full publication gate passes."); return 0


if __name__ == "__main__": raise SystemExit(main())
