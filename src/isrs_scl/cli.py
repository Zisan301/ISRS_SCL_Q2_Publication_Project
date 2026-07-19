"""Safe command-line interface for publication, smoke, and debug studies."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence
import json
import sys
import tempfile

import yaml

from isrs_scl.experiments import run_publication_study
from isrs_scl.system.parameters import ConfigError, apply_defaults, validate_config
from isrs_scl.validation.external_validation import load_external_validation
from isrs_scl.validation.reproducibility import ProvenanceError

EXIT_OK = 0
EXIT_CONFIGURATION = 2
EXIT_NUMERICAL = 3
EXIT_VALIDATION = 4
EXIT_PROVENANCE = 5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config_q2_final.yaml", help="Study YAML; defaults to the final Q2 configuration")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--publication", action="store_true", help="Strict final evidence run; every gate is mandatory")
    modes.add_argument("--smoke", action="store_true", help="Reduced verification run; output is not journal evidence")
    modes.add_argument("--debug", action="store_true", help="Diagnostic run; output is not journal evidence")
    parser.add_argument("--grid-mode", choices=["full_scl", "paper_240_subset"])
    parser.add_argument("--run-id", help="Required unique identifier for publication output")
    parser.add_argument("--output-root", default="runs", help="Parent directory for isolated run directories")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing run directory; forbidden by default")
    parser.add_argument("--no-uncertainty", action="store_true", help="Debug only; publication mode rejects this option")
    parser.add_argument("--allow-untracked-provenance", action="store_true", help="Debug escape hatch when Git provenance is unavailable/dirty")
    return parser


def resolve_mode(args: argparse.Namespace) -> str:
    if args.publication: return "publication"
    if args.smoke: return "smoke"
    return "debug"


def _load_raw(path: Path) -> dict:
    if not path.exists(): raise FileNotFoundError(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict): raise ConfigError("The YAML root must be a mapping")
    return apply_defaults(raw)


def prepare_cli_config(args: argparse.Namespace) -> dict:
    path = Path(args.config)
    cfg = _load_raw(path)
    mode = resolve_mode(args)
    if mode == "publication" and args.no_uncertainty:
        raise ConfigError("Publication mode cannot use --no-uncertainty")
    cfg["run"].update({
        "mode": mode,
        "run_id": args.run_id or cfg["run"].get("run_id") or (None if mode == "publication" else f"{mode}-{cfg['metadata']['random_seed']}"),
        "output_root": args.output_root,
        "overwrite": bool(args.overwrite),
        "allow_untracked_provenance": bool(args.allow_untracked_provenance),
    })
    if mode == "publication":
        cfg["reproducibility"]["strict_git_clean"] = True
        cfg["uncertainty"]["enabled"] = True
        cfg["validation"]["require_uncertainty_analysis"] = True
        if str(cfg["metadata"].get("calibration_status", "")).upper() != "CALIBRATED":
            raise ConfigError("Publication mode rejects UNVALIDATED_DEFAULTS; provide real calibration evidence first")
        if bool(cfg["validation"]["require_external_validation"]):
            reference = Path(cfg["validation"]["external_reference_csv"])
            reference = reference if reference.is_absolute() else path.parent / reference
            load_external_validation(reference)  # rejects placeholders/blank rows
    if args.grid_mode: cfg["grid"]["mode"] = args.grid_mode
    validate_config(cfg, base_dir=path.parent)
    return cfg


@contextmanager
def materialized_config(original_path: str | Path, cfg: dict) -> Iterator[Path]:
    original = Path(original_path).resolve()
    descriptor, temporary = tempfile.mkstemp(prefix=".isrs_resolved_", suffix=".yaml", dir=original.parent)
    path = Path(temporary)
    try:
        with open(descriptor, "w", encoding="utf-8", closefd=True) as handle:
            yaml.safe_dump(cfg, handle, sort_keys=False)
        yield path
    finally:
        path.unlink(missing_ok=True)


def run_from_args(args: argparse.Namespace) -> int:
    try:
        cfg = prepare_cli_config(args)
        with materialized_config(args.config, cfg) as resolved:
            summary = run_publication_study(
                resolved,
                grid_mode=args.grid_mode,
                smoke=resolve_mode(args) == "smoke",
                strict=resolve_mode(args) == "publication",
                no_uncertainty=bool(args.no_uncertainty),
            )
        print(json.dumps(summary, indent=2, sort_keys=True))
        print(f"Run directory: {summary['run_directory']}")
        print(f"Manifest: {summary['manifest_path']}")
        print(f"Validation status: {summary['validation_status_path']}")
        if resolve_mode(args) == "publication" and summary["failed_publication_checks"]:
            print("Failed publication checks: " + ", ".join(summary["failed_publication_checks"]))
        elif resolve_mode(args) != "publication":
            gaps = summary.get("publication_gaps_if_submitted", [])
            if gaps:
                print("Publication gaps if submitted now: " + ", ".join(gaps))
            print("SMOKE/DEBUG COMPLETED: pipeline plumbing passed, but this output is not journal evidence.")
        return EXIT_OK if summary["publication_ready_numerical_claims"] or resolve_mode(args) != "publication" else EXIT_VALIDATION
    except (ConfigError, FileNotFoundError, yaml.YAMLError, ValueError) as exc:
        print(f"Configuration failure: {exc}", file=sys.stderr)
        return EXIT_CONFIGURATION
    except ProvenanceError as exc:
        print(f"Provenance failure: {exc}", file=sys.stderr)
        return EXIT_PROVENANCE
    except RuntimeError as exc:
        message = str(exc)
        if "publication gate failed" in message.lower():
            print(message, file=sys.stderr)
            return EXIT_VALIDATION
        print(f"Numerical/pipeline failure: {message}", file=sys.stderr)
        return EXIT_NUMERICAL
    except Exception as exc:
        print(f"Unexpected numerical failure: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_NUMERICAL


def main(argv: Sequence[str] | None = None) -> int:
    return run_from_args(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
