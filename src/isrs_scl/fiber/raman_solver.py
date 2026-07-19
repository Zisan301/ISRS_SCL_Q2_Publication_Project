"""Coupled signal-to-signal ISRS and undepleted backward-pump RK4 solver."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.signal import fftconvolve

from isrs_scl.constants import C_M_PER_S
from isrs_scl.fiber.attenuation import db_per_km_to_np_per_m


@dataclass(frozen=True)
class RamanPump:
    wavelength_nm: float
    power_w: float
    attenuation_db_per_km: float
    direction: str = "backward"

    @property
    def frequency_hz(self) -> float:
        return C_M_PER_S / (self.wavelength_nm * 1e-9)

    @property
    def alpha_np_per_m(self) -> float:
        return float(db_per_km_to_np_per_m(self.attenuation_db_per_km))


@dataclass(frozen=True)
class RamanResult:
    z_m: np.ndarray
    powers_w: np.ndarray  # shape [n_saved_z, n_channels]

    @property
    def output_powers_w(self) -> np.ndarray:
        return self.powers_w[-1]


class RamanGainSpectrum:
    """Silica Raman gain spectrum.

    A measured CSV can be supplied with columns ``shift_thz`` and
    ``gain_m_per_w``. Otherwise a smooth multi-Gaussian approximation is used
    and normalized to the configured peak. The approximation is deliberately
    explicit so that journal work can replace it with measured fiber data.
    """

    def __init__(self, peak_m_per_w: float, csv_path: str | Path | None = None):
        self.peak_m_per_w = float(peak_m_per_w)
        self._shift_hz: np.ndarray | None = None
        self._gain: np.ndarray | None = None
        if csv_path:
            data = np.genfromtxt(csv_path, delimiter=",", names=True)
            if "shift_thz" not in data.dtype.names or "gain_m_per_w" not in data.dtype.names:
                raise ValueError("Raman CSV needs shift_thz,gain_m_per_w columns")
            order = np.argsort(data["shift_thz"])
            self._shift_hz = np.asarray(data["shift_thz"][order], dtype=float) * 1e12
            self._gain = np.asarray(data["gain_m_per_w"][order], dtype=float)

    def __call__(self, delta_frequency_hz: np.ndarray | float) -> np.ndarray:
        df = np.asarray(delta_frequency_hz, dtype=float)
        positive = np.clip(df, 0.0, None)
        if self._shift_hz is not None and self._gain is not None:
            out = np.interp(positive, self._shift_hz, self._gain, left=0.0, right=0.0)
        else:
            x = positive / 1e12
            # Broad silica response with a principal peak near 13.2 THz.
            shape = (
                0.62 * np.exp(-0.5 * ((x - 13.2) / 2.25) ** 2)
                + 0.25 * np.exp(-0.5 * ((x - 14.8) / 4.2) ** 2)
                + 0.10 * np.exp(-0.5 * ((x - 6.5) / 2.8) ** 2)
                + 0.03 * np.exp(-0.5 * ((x - 20.0) / 4.5) ** 2)
            )
            # Normalize using a dense fixed grid rather than the current query.
            xr = np.linspace(0.0, 30.0, 6001)
            ref = (
                0.62 * np.exp(-0.5 * ((xr - 13.2) / 2.25) ** 2)
                + 0.25 * np.exp(-0.5 * ((xr - 14.8) / 4.2) ** 2)
                + 0.10 * np.exp(-0.5 * ((xr - 6.5) / 2.8) ** 2)
                + 0.03 * np.exp(-0.5 * ((xr - 20.0) / 4.5) ** 2)
            ).max()
            out = self.peak_m_per_w * shape / ref
        return np.where(df > 0.0, out, 0.0)


class RamanSolver:
    """RK4 solver for an exactly spaced WDM frequency grid.

    Frequencies must be strictly increasing. Pairwise signal interactions use
    a convolution form. For each high/low-frequency pair, the high-frequency
    depletion includes the photon-frequency ratio nu_high/nu_low, ensuring the
    correct energy-transfer relation.
    """

    def __init__(
        self,
        frequencies_hz: np.ndarray,
        alpha_np_per_m: np.ndarray,
        effective_area_m2: float,
        gain_spectrum: RamanGainSpectrum,
        pumps: Iterable[RamanPump] = (),
    ):
        self.f = np.asarray(frequencies_hz, dtype=float)
        self.alpha = np.asarray(alpha_np_per_m, dtype=float)
        if self.f.ndim != 1 or self.alpha.shape != self.f.shape:
            raise ValueError("Frequency and attenuation arrays must be 1-D and equal length")
        if self.f.size > 1 and np.any(np.diff(self.f) <= 0):
            raise ValueError("Frequencies must be strictly increasing")
        self.aeff = float(effective_area_m2)
        if self.aeff <= 0:
            raise ValueError("Effective area must be positive")
        self.gain = gain_spectrum
        self.pumps = tuple(pumps)

        if self.f.size > 1:
            spacing = np.diff(self.f)
            if not np.allclose(spacing, spacing[0], rtol=1e-10, atol=1.0):
                raise ValueError("Fast Raman solver requires a uniform frequency grid")
            shifts = np.arange(self.f.size, dtype=float) * spacing[0]
        else:
            shifts = np.zeros(1)
        self.kernel = self.gain(shifts)
        self.kernel[0] = 0.0

    def _pump_power(self, pump: RamanPump, z_m: float, length_m: float) -> float:
        if pump.direction.lower() == "backward":
            return pump.power_w * np.exp(-pump.alpha_np_per_m * (length_m - z_m))
        if pump.direction.lower() == "forward":
            return pump.power_w * np.exp(-pump.alpha_np_per_m * z_m)
        raise ValueError(f"Unsupported pump direction {pump.direction!r}")

    def derivative(self, z_m: float, power_w: np.ndarray, length_m: float) -> np.ndarray:
        p = np.maximum(np.asarray(power_w, dtype=float), 0.0)
        # Gain received from higher-frequency channels.
        gain_from_high = fftconvolve(p[::-1], self.kernel, mode="full")[: p.size][::-1]
        # High-frequency depletion to lower-frequency channels; photon ratio is exact.
        weighted_low = fftconvolve(p / self.f, self.kernel, mode="full")[: p.size]
        loss_to_low = self.f * weighted_low
        interaction = p * (gain_from_high - loss_to_low) / self.aeff

        pump_gain_per_m = np.zeros_like(p)
        for pump in self.pumps:
            df = pump.frequency_hz - self.f
            local_pump = self._pump_power(pump, z_m, length_m)
            pump_gain_per_m += self.gain(df) * local_pump / self.aeff

        return -self.alpha * p + interaction + pump_gain_per_m * p

    def solve(
        self,
        launch_powers_w: np.ndarray,
        length_m: float,
        step_m: float,
        save_step_m: float | None = None,
    ) -> RamanResult:
        p = np.asarray(launch_powers_w, dtype=float).copy()
        if p.shape != self.f.shape or np.any(p <= 0):
            raise ValueError("Launch powers must be positive and match the grid")
        if length_m <= 0 or step_m <= 0:
            raise ValueError("Length and step must be positive")

        n_steps = int(np.ceil(length_m / step_m))
        h = length_m / n_steps
        save_stride = max(1, int(round((save_step_m or h) / h)))
        saved_z = [0.0]
        saved_p = [p.copy()]
        z = 0.0
        for step in range(n_steps):
            k1 = self.derivative(z, p, length_m)
            k2 = self.derivative(z + 0.5 * h, np.maximum(p + 0.5 * h * k1, 1e-30), length_m)
            k3 = self.derivative(z + 0.5 * h, np.maximum(p + 0.5 * h * k2, 1e-30), length_m)
            k4 = self.derivative(z + h, np.maximum(p + h * k3, 1e-30), length_m)
            p = np.maximum(p + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0, 1e-30)
            z = (step + 1) * h
            if (step + 1) % save_stride == 0 or step + 1 == n_steps:
                saved_z.append(z)
                saved_p.append(p.copy())
        return RamanResult(np.asarray(saved_z), np.vstack(saved_p))


def pumps_from_config(raman_cfg: dict) -> tuple[RamanPump, ...]:
    return tuple(RamanPump(**entry) for entry in raman_cfg.get("pumps", []))


def analytical_undepleted_pump_signal(
    signal_power_w: float,
    signal_alpha_np_per_m: float,
    pump_power_at_launch_w: float,
    pump_alpha_np_per_m: float,
    raman_gain_m_per_w: float,
    effective_area_m2: float,
    length_m: float,
) -> float:
    if pump_alpha_np_per_m > 0:
        pump_integral_w_m = pump_power_at_launch_w * (
            1.0 - np.exp(-pump_alpha_np_per_m * length_m)
        ) / pump_alpha_np_per_m
    else:
        pump_integral_w_m = pump_power_at_launch_w * length_m
    exponent = (
        -signal_alpha_np_per_m * length_m
        + raman_gain_m_per_w * pump_integral_w_m / effective_area_m2
    )
    return float(signal_power_w * np.exp(exponent))
