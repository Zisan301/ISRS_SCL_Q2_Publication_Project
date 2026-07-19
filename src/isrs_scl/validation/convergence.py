"""Raman RK4 spatial-step convergence checks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from isrs_scl.fiber.amplification import dbm_to_w
from isrs_scl.fiber.span_model import SpanModel
from isrs_scl.system.grid import OpticalGrid


def run_step_convergence(
    grid: OpticalGrid,
    cfg: dict,
    steps_m: tuple[float, ...] = (80.0, 40.0, 20.0),
    maximum_channels: int = 81,
) -> pd.DataFrame:
    # Keep the convergence test tractable while retaining S/C/L representatives.
    if grid.n_channels > maximum_channels:
        # Use a contiguous central slice so the exact 50-GHz spacing is preserved.
        start = (grid.n_channels - maximum_channels) // 2
        stop = start + maximum_channels
        subgrid = OpticalGrid(
            grid.frequencies_hz[start:stop],
            grid.wavelengths_nm[start:stop],
            grid.bands[start:stop],
            grid.spacing_hz,
            grid.mode,
        )
    else:
        subgrid = grid
    launch = np.full(subgrid.n_channels, float(dbm_to_w(cfg["launch"]["flat_power_dbm_per_channel"])))
    outputs = {}
    for step in steps_m:
        model = SpanModel(subgrid, cfg, integration_step_m=step)
        outputs[step] = model.evaluate(launch).raman_result.output_powers_w
    reference = outputs[min(steps_m)]
    rows = []
    for step in steps_m:
        error_db = 10.0 * np.log10(outputs[step] / reference)
        rows.append(
            {
                "step_m": step,
                "max_abs_power_error_db": float(np.max(np.abs(error_db))),
                "rms_power_error_db": float(np.sqrt(np.mean(error_db**2))),
                "channels": subgrid.n_channels,
            }
        )
    return pd.DataFrame(rows).sort_values("step_m", ascending=False)
