"""Launch-profile constraints and projections."""

from __future__ import annotations

import numpy as np

from isrs_scl.fiber.amplification import dbm_to_w


def total_power_w_from_dbm(profile_dbm: np.ndarray) -> float:
    return float(np.sum(dbm_to_w(profile_dbm)))


def project_total_power_and_bounds_dbm(
    profile_dbm: np.ndarray,
    target_total_power_w: float,
    minimum_dbm: float,
    maximum_dbm: float,
) -> np.ndarray:
    """Project by a common dB shift plus clipping.

    The bisection solves the monotone equation
    ``sum(10^((clip(x+delta)-30)/10)) = target_total_power``.
    """

    x = np.asarray(profile_dbm, dtype=float)
    n = x.size
    min_total = n * float(dbm_to_w(minimum_dbm))
    max_total = n * float(dbm_to_w(maximum_dbm))
    if not min_total - 1e-15 <= target_total_power_w <= max_total + 1e-15:
        raise ValueError("Target total power is infeasible under channel bounds")

    low, high = -100.0, 100.0
    for _ in range(100):
        mid = 0.5 * (low + high)
        candidate = np.clip(x + mid, minimum_dbm, maximum_dbm)
        total = total_power_w_from_dbm(candidate)
        if total < target_total_power_w:
            low = mid
        else:
            high = mid
    projected = np.clip(x + 0.5 * (low + high), minimum_dbm, maximum_dbm)
    return projected


def smooth_profile_dbm(profile_dbm: np.ndarray, strength: float) -> np.ndarray:
    """Tikhonov smoothing with a second-difference penalty."""

    x = np.asarray(profile_dbm, dtype=float)
    if x.size < 3 or strength <= 0:
        return x.copy()
    n = x.size
    d2 = np.zeros((n - 2, n))
    rows = np.arange(n - 2)
    d2[rows, rows] = 1.0
    d2[rows, rows + 1] = -2.0
    d2[rows, rows + 2] = 1.0
    matrix = np.eye(n) + float(strength) * (d2.T @ d2)
    return np.linalg.solve(matrix, x)


def project_launch_profile_dbm(
    profile_dbm: np.ndarray,
    target_total_power_w: float,
    minimum_dbm: float,
    maximum_dbm: float,
    smoothing_strength: float,
) -> np.ndarray:
    smoothed = smooth_profile_dbm(profile_dbm, smoothing_strength)
    return project_total_power_and_bounds_dbm(
        smoothed, target_total_power_w, minimum_dbm, maximum_dbm
    )


def second_difference_energy(profile_dbm: np.ndarray) -> float:
    x = np.asarray(profile_dbm, dtype=float)
    return float(np.mean(np.diff(x, n=2) ** 2)) if x.size >= 3 else 0.0
