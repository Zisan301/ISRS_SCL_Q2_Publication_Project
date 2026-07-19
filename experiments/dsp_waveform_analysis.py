"""Run waveform validation against a profile from the same hashed analytical run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from isrs_scl.experiments import MonotoneReceiverCalibration, run_waveform_validation
from isrs_scl.link import LinkModel
from isrs_scl.system.grid import build_grid
from isrs_scl.system.parameters import load_config
from isrs_scl.validation.reproducibility import build_run_manifest, finalize_manifest, prepare_run_directory, sha256_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config_q2_final.yaml")
    parser.add_argument("--profile", required=True, help="CSV containing launch_power_dbm")
    parser.add_argument("--b2b-calibration", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", default="runs/partial")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    source_manifest = json.loads(Path(args.source_manifest).read_text(encoding="utf-8"))
    if source_manifest.get("configuration_sha256") is None:
        raise ValueError("Source manifest does not contain a configuration hash")
    profile_frame = __import__("pandas").read_csv(args.profile)
    if "launch_power_dbm" not in profile_frame:
        raise ValueError("Profile CSV requires launch_power_dbm")
    calibration_frame = __import__("pandas").read_csv(args.b2b_calibration)
    calibration = MonotoneReceiverCalibration(
        calibration_frame["input_snr_db"].to_numpy(float),
        calibration_frame["measured_snr_db"].to_numpy(float),
        calibration_frame["ngmi"].to_numpy(float),
    )
    paths = prepare_run_directory(args.output_root, args.run_id, overwrite=args.overwrite)
    grid = build_grid(cfg["grid"]); profile = profile_frame["launch_power_dbm"].to_numpy(float)
    if profile.size != grid.n_channels: raise ValueError("Profile and grid channel counts differ")
    link = LinkModel(grid, cfg, receiver_calibration=calibration)
    frame = run_waveform_validation(link, profile, cfg, paths.results, paths.figures, int(cfg["output"]["png_dpi"]), calibration)
    manifest = build_run_manifest(cfg, run_root=paths.root, input_files=[args.profile, args.b2b_calibration, args.source_manifest]); finalize_manifest(manifest, paths.root, paths.metadata / "RUN_MANIFEST.json")
    print(json.dumps({"run_directory": str(paths.root), "rows": len(frame), "profile_sha256": sha256_file(args.profile), "notice": "Representative-channel validation only; not full-grid SSFM."}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
