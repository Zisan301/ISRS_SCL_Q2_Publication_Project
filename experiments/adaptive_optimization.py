"""Run the robust multiseed optimization stage as partial evidence."""
from __future__ import annotations

import argparse
from pathlib import Path
import json

import numpy as np

from isrs_scl.fiber.amplification import dbm_to_w
from isrs_scl.link import LinkModel
from isrs_scl.optimization.adaptive_isrs import fixed_preemphasis_profile_dbm
from isrs_scl.optimization.statistics import run_multiseed_optimization
from isrs_scl.system.grid import build_grid
from isrs_scl.system.parameters import load_config
from isrs_scl.validation.reproducibility import atomic_write_json, finalize_manifest, build_run_manifest, prepare_run_directory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config_q2_final.yaml")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", default="runs/partial")
    parser.add_argument("--mode", choices=["publication", "smoke", "debug"], default="debug")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args(); cfg = load_config(args.config)
    cfg["run"].update({"mode": args.mode, "run_id": args.run_id, "output_root": args.output_root, "overwrite": args.overwrite})
    paths = prepare_run_directory(args.output_root, args.run_id, overwrite=args.overwrite)
    grid = build_grid(cfg["grid"]); link = LinkModel(grid, cfg)
    fixed = fixed_preemphasis_profile_dbm(grid.frequencies_hz, float(cfg["launch"]["flat_power_dbm_per_channel"]), float(cfg["launch"]["fixed_preemphasis_s_to_l_db"]), float(cfg["launch"]["min_power_dbm_per_channel"]), float(cfg["launch"]["max_power_dbm_per_channel"]))
    result = run_multiseed_optimization(link, cfg, fixed)
    result.run_summary.to_csv(paths.results / "optimizer_multiseed.csv", index=False); result.history.to_csv(paths.results / "optimizer_history.csv", index=False)
    atomic_write_json(paths.results / "optimizer_confidence.json", result.confidence)
    np.savetxt(paths.results / "adaptive_profile_dbm.csv", result.best_result.optimized_profile_dbm, delimiter=",", header="launch_power_dbm", comments="")
    manifest = build_run_manifest(cfg, run_root=paths.root); finalize_manifest(manifest, paths.root, paths.metadata / "RUN_MANIFEST.json")
    print(json.dumps({"run_directory": str(paths.root), "accepted": result.best_result.improved, "notice": "Partial evidence only until full publication gate passes."}, indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
