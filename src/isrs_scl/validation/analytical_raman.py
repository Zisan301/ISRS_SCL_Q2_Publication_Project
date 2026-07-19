"""Analytical undepleted-pump validation experiment."""

from __future__ import annotations

import numpy as np
import pandas as pd

from isrs_scl.constants import C_M_PER_S
from isrs_scl.fiber.raman_solver import (
    RamanGainSpectrum,
    RamanPump,
    RamanSolver,
    analytical_undepleted_pump_signal,
)


def run_undepleted_pump_validation(
    length_km: float = 80.0,
    step_m: float = 80.0,
    signal_wavelength_nm: float = 1550.0,
    pump_wavelength_nm: float = 1450.0,
    signal_power_w: float = 1e-6,
    pump_power_w: float = 0.25,
    alpha_signal_np_per_m: float = 4.6e-5,
    alpha_pump_db_per_km: float = 0.23,
    effective_area_m2: float = 80e-12,
    gain_peak_m_per_w: float = 8e-14,
) -> tuple[pd.DataFrame, float]:
    f_signal = C_M_PER_S / (signal_wavelength_nm * 1e-9)
    gain = RamanGainSpectrum(gain_peak_m_per_w)
    pump = RamanPump(pump_wavelength_nm, pump_power_w, alpha_pump_db_per_km, "backward")
    solver = RamanSolver(
        np.array([f_signal]),
        np.array([alpha_signal_np_per_m]),
        effective_area_m2,
        gain,
        [pump],
    )
    result = solver.solve(
        np.array([signal_power_w]), length_km * 1000.0, step_m, save_step_m=1000.0
    )
    # Evaluate the analytical expression at every saved z with the same backward
    # pump boundary convention (pump power specified at z=L).
    g_value = float(gain(pump.frequency_hz - f_signal))
    analytical = []
    for z in result.z_m:
        # Over 0..z, the backward pump is P(L) exp[-alpha_p(L-u)].
        alpha_p = pump.alpha_np_per_m
        if alpha_p > 0:
            pump_integral = (
                pump.power_w
                * np.exp(-alpha_p * (length_km * 1000.0 - z))
                * (1.0 - np.exp(-alpha_p * z))
                / alpha_p
            )
        else:
            pump_integral = pump.power_w * z
        analytical.append(
            signal_power_w
            * np.exp(-alpha_signal_np_per_m * z + g_value * pump_integral / effective_area_m2)
        )
    analytical = np.asarray(analytical)
    relative_error = np.abs(result.powers_w[:, 0] - analytical) / np.maximum(analytical, 1e-30)
    frame = pd.DataFrame(
        {
            "z_km": result.z_m / 1000.0,
            "numerical_power_w": result.powers_w[:, 0],
            "analytical_power_w": analytical,
            "relative_error": relative_error,
        }
    )
    return frame, float(relative_error.max())
