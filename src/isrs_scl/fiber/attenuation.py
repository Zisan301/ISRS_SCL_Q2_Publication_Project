"""Frequency-dependent loss and dispersion helpers."""

from __future__ import annotations

import numpy as np

from isrs_scl.constants import C_M_PER_S


def db_per_km_to_np_per_m(alpha_db_per_km: np.ndarray | float) -> np.ndarray:
    """Convert a *power* attenuation coefficient from dB/km to Np/m.

    For power, ``P(z)=P(0) exp(-alpha_np z)`` and
    ``alpha_np = ln(10)/10 * alpha_dB``.
    """

    return np.asarray(alpha_db_per_km, dtype=float) * np.log(10.0) / 10.0 / 1000.0


def np_per_m_to_db_per_km(alpha_np_per_m: np.ndarray | float) -> np.ndarray:
    return np.asarray(alpha_np_per_m, dtype=float) * 10.0 / np.log(10.0) * 1000.0


def attenuation_db_per_km(wavelength_nm: np.ndarray, fiber_cfg: dict) -> np.ndarray:
    anchors = fiber_cfg["attenuation_anchors"]
    x = np.asarray(anchors["wavelength_nm"], dtype=float)
    y = np.asarray(anchors["db_per_km"], dtype=float)
    if np.any(np.diff(x) <= 0):
        raise ValueError("Attenuation wavelengths must be strictly increasing")
    return np.interp(np.asarray(wavelength_nm, dtype=float), x, y, left=y[0], right=y[-1])


def beta2_beta3_si(wavelength_nm: np.ndarray, fiber_cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return beta2 [s^2/m] and beta3 [s^3/m] from D and dispersion slope.

    D(lambda) is approximated linearly around 1550 nm. beta3 is evaluated by a
    stable finite difference of beta2 versus angular frequency, avoiding a hidden
    sign/unit error in hand-coded conversions.
    """

    wl_nm = np.asarray(wavelength_nm, dtype=float)
    d0 = float(fiber_cfg["dispersion_ps_nm_km_at_1550"])
    slope = float(fiber_cfg["dispersion_slope_ps_nm2_km"])

    def beta2_at(wl_nm_local: np.ndarray) -> np.ndarray:
        d_ps_nm_km = d0 + slope * (wl_nm_local - 1550.0)
        d_si = d_ps_nm_km * 1e-6  # ps/(nm km) -> s/m^2
        wl_m = wl_nm_local * 1e-9
        return -(wl_m**2 / (2.0 * np.pi * C_M_PER_S)) * d_si

    beta2 = beta2_at(wl_nm)
    delta_nm = 0.01
    bp = beta2_at(wl_nm + delta_nm)
    bm = beta2_at(wl_nm - delta_nm)
    dbeta2_dlambda = (bp - bm) / (2.0 * delta_nm * 1e-9)
    wl_m = wl_nm * 1e-9
    domega_dlambda = -2.0 * np.pi * C_M_PER_S / wl_m**2
    beta3 = dbeta2_dlambda / domega_dlambda
    return beta2, beta3


def gamma_per_w_m(wavelength_nm: np.ndarray, fiber_cfg: dict) -> np.ndarray:
    """Weak wavelength scaling of nonlinear coefficient through effective frequency."""

    wl = np.asarray(wavelength_nm, dtype=float)
    gamma_1550_per_w_m = float(fiber_cfg["gamma_per_w_km_at_1550"]) / 1000.0
    return gamma_1550_per_w_m * 1550.0 / wl
