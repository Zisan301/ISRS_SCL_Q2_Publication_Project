"""Run local sensitivity, global uncertainty, or independent robust holdout."""
from __future__ import annotations

import argparse
import json

import numpy as np

from isrs_scl.link import LinkModel
from isrs_scl.optimization.adaptive_isrs import fixed_preemphasis_profile_dbm
from isrs_scl.system.grid import build_grid
from isrs_scl.system.parameters import load_config
from isrs_scl.validation.reproducibility import build_run_manifest, finalize_manifest, prepare_run_directory
from isrs_scl.validation.uncertainty import run_uncertainty_analysis


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config_q2_final.yaml")
    parser.add_argument("--run-id", required=True); parser.add_argument("--output-root", default="runs/partial"); parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--analysis", choices=["local", "global", "holdout"], default="holdout")
    parser.add_argument("--samples", type=int); parser.add_argument("--seed", type=int)
    args = parser.parse_args(); cfg = load_config(args.config)
    if cfg["run"]["mode"] == "publication" and args.samples is not None and args.samples < int(cfg["validation"]["minimum_uncertainty_holdout_samples"]): raise ValueError("Requested samples are below the publication minimum")
    paths = prepare_run_directory(args.output_root, args.run_id, overwrite=args.overwrite); grid = build_grid(cfg["grid"]); link = LinkModel(grid, cfg)
    flat = np.full(grid.n_channels, float(cfg["launch"]["flat_power_dbm_per_channel"])); fixed = fixed_preemphasis_profile_dbm(grid.frequencies_hz, float(cfg["launch"]["flat_power_dbm_per_channel"]), float(cfg["launch"]["fixed_preemphasis_s_to_l_db"]), float(cfg["launch"]["min_power_dbm_per_channel"]), float(cfg["launch"]["max_power_dbm_per_channel"]))
    profiles = {"flat": flat, "fixed": fixed, "adaptive": fixed.copy()}
    result = run_uncertainty_analysis(cfg, profiles, int(cfg["optimization"]["target_spans"]), samples=args.samples, seed=args.seed)
    result.samples.to_csv(paths.results / f"{args.analysis}_samples.csv", index=False); result.summary.to_csv(paths.results / f"{args.analysis}_summary.csv", index=False); result.sensitivity.to_csv(paths.results / f"{args.analysis}_sensitivity.csv", index=False); result.convergence.to_csv(paths.results / f"{args.analysis}_convergence.csv", index=False); result.paired_gains.to_csv(paths.results / f"{args.analysis}_paired_gains.csv", index=False)
    finalize_manifest(build_run_manifest(cfg, run_root=paths.root), paths.root, paths.metadata / "RUN_MANIFEST.json")
    print(json.dumps({"run_directory": str(paths.root), "successful_fraction": result.successful_fraction, "batch_hash": result.batch_hash}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
