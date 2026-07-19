"""Generate a run-isolated flat-launch baseline."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from isrs_scl.link import LinkModel
from isrs_scl.system.grid import build_grid
from isrs_scl.system.parameters import load_config
from isrs_scl.validation.reproducibility import build_run_manifest, finalize_manifest, prepare_run_directory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", default="config_q2_final.yaml"); parser.add_argument("--run-id", required=True); parser.add_argument("--output-root", default="runs/partial"); parser.add_argument("--overwrite", action="store_true"); args = parser.parse_args()
    cfg = load_config(args.config); paths = prepare_run_directory(args.output_root, args.run_id, overwrite=args.overwrite)
    grid = build_grid(cfg["grid"]); link = LinkModel(grid, cfg); launch = link.flat_launch_w()
    tables = []
    for result in link.sweep_spans(launch):
        frame = result.to_frame(grid); frame.insert(0, "spans", result.n_spans); frame.insert(1, "strategy", "flat"); tables.append(frame)
    pd.concat(tables, ignore_index=True).to_csv(paths.results / "flat_channel_performance.csv", index=False)
    finalize_manifest(build_run_manifest(cfg, run_root=paths.root), paths.root, paths.metadata / "RUN_MANIFEST.json")
    print(f"{paths.root}\nPartial evidence only until the full publication gate passes."); return 0


if __name__ == "__main__": raise SystemExit(main())
