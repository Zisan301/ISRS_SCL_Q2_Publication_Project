"""Compatibility alias for users running all_run_experiment.py.

The canonical entry point is run_all_experiments.py.
"""
from __future__ import annotations

from run_all_experiments import main


if __name__ == "__main__":
    raise SystemExit(main())
