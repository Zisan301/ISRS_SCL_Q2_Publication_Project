"""Run calibration-to-held-out GNPy validation from exported channel performance."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from isrs_scl.validation.heldout_external_validation import run_heldout_external_validation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-channel-sweep",
        required=True,
        help="Path to channel_performance.csv exported by the main study run.",
    )
    parser.add_argument(
        "--calibration-csv",
        default="validation_data/gnpy_calibration_train_1535_1550_1560.csv",
        help="Calibration/train GNPy CSV. Defaults to the frozen original 9-row split.",
    )
    parser.add_argument(
        "--holdout-csv",
        required=True,
        help="Unseen held-out GNPy CSV, for example 1540/1545/1555 nm rows.",
    )
    parser.add_argument("--output-dir", help="Directory for held-out validation outputs.")
    parser.add_argument("--wavelength-tolerance-nm", type=float, default=0.5)
    parser.add_argument("--allow-interpolation", action="store_true")
    parser.add_argument("--center-wavelength-nm", type=float, default=1550.0)
    parser.add_argument("--max-holdout-rmse-db", type=float, default=1.0)
    parser.add_argument("--max-holdout-bias-db", type=float, default=0.75)
    parser.add_argument("--max-calibration-loo-rmse-db", type=float, default=1.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    model_path = Path(args.model_channel_sweep)
    calibration_path = Path(args.calibration_csv)
    holdout_path = Path(args.holdout_csv)
    output_dir = Path(args.output_dir) if args.output_dir else model_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    result = run_heldout_external_validation(
        pd.read_csv(model_path),
        pd.read_csv(calibration_path),
        pd.read_csv(holdout_path),
        wavelength_tolerance_nm=args.wavelength_tolerance_nm,
        allow_interpolation=args.allow_interpolation,
        center_wavelength_nm=args.center_wavelength_nm,
        thresholds={
            "maximum_holdout_gsnr_rmse_db": args.max_holdout_rmse_db,
            "maximum_holdout_gsnr_absolute_bias_db": args.max_holdout_bias_db,
            "maximum_calibration_loo_gsnr_rmse_db": args.max_calibration_loo_rmse_db,
            "minimum_holdout_coverage": 1.0,
        },
    )

    result.calibration_comparisons.to_csv(
        output_dir / "heldout_external_calibration_comparisons.csv", index=False
    )
    result.holdout_comparisons.to_csv(
        output_dir / "heldout_external_holdout_comparisons.csv", index=False
    )
    result.summary.to_csv(output_dir / "heldout_external_validation_summary.csv", index=False)
    result.requirements.to_csv(
        output_dir / "heldout_external_validation_requirements.csv", index=False
    )
    payload = {
        "passed": result.passed,
        "reasons": list(result.reasons),
        "correction": result.correction.as_dict(),
    }
    (output_dir / "heldout_external_validation_status.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if result.passed else 4


if __name__ == "__main__":
    raise SystemExit(main())
