"""Band-aware amplification and explicit ASE/noise-bandwidth accounting."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from isrs_scl.constants import C_M_PER_S, H_J_S
from isrs_scl.dsp.noise import psd_to_dual_pol_power, receiver_equivalent_noise_bandwidth


@dataclass(frozen=True)
class AmplifierResult:
    gain_linear: np.ndarray
    gain_db: np.ndarray
    output_signal_w: np.ndarray
    ase_psd_w_per_hz: np.ndarray
    ase_channel_w: np.ndarray
    residual_db: np.ndarray
    optical_noise_bandwidth_hz: float = 0.0
    receiver_equivalent_noise_bandwidth_hz: float = 0.0
    ase_receiver_w: np.ndarray | None = None
    ase_01nm_w: np.ndarray | None = None


def dbm_to_w(dbm: np.ndarray | float) -> np.ndarray:
    return 1e-3 * 10.0 ** (np.asarray(dbm, dtype=float) / 10.0)


def w_to_dbm(power_w: np.ndarray | float) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(np.asarray(power_w, dtype=float), 1e-30) / 1e-3)


def _band_parameters(wavelength_nm: np.ndarray, amp_cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    wl = np.asarray(wavelength_nm, dtype=float)
    nf_db = np.full(wl.shape, np.nan)
    max_gain_db = np.zeros(wl.shape)
    for band in amp_cfg["bands"].values():
        low, high = map(float, band["wavelength_nm"])
        mask = (wl >= low) & (wl <= high)
        if not bool(band["enabled"]):
            nf_db[mask], max_gain_db[mask] = np.inf, 0.0
            continue
        values = np.asarray(band["noise_figure_db"], dtype=float)
        if values.size == 1:
            nf_db[mask] = values[0]
        elif values.size == 2:
            nf_db[mask] = np.interp(wl[mask], [low, high], values)
        else:
            raise ValueError("noise_figure_db must contain one value or two edge values")
        max_gain_db[mask] = float(band["max_gain_db"])
    if np.any(np.isnan(nf_db)):
        raise ValueError("Amplifier band definitions do not cover all channels")
    return nf_db, max_gain_db


def dual_pol_ase_psd_w_per_hz(frequency_hz: np.ndarray, gain_linear: np.ndarray, noise_figure_db: np.ndarray) -> np.ndarray:
    """Return dual-polarization ASE PSD at the amplifier output in W/Hz."""
    frequency = np.asarray(frequency_hz, dtype=float)
    gain = np.asarray(gain_linear, dtype=float)
    noise_figure = 10.0 ** (np.asarray(noise_figure_db, dtype=float) / 10.0)
    output = np.zeros_like(frequency)
    active = gain > 1.0 + 1e-12
    nsp = np.zeros_like(frequency)
    nsp[active] = noise_figure[active] / (2.0 * (1.0 - 1.0 / gain[active]))
    output[active] = 2.0 * nsp[active] * H_J_S * frequency[active] * (gain[active] - 1.0)
    return output


def reference_bandwidth_01nm_hz(wavelength_nm: np.ndarray) -> np.ndarray:
    wl_m = np.asarray(wavelength_nm, dtype=float) * 1e-9
    return C_M_PER_S / wl_m**2 * 0.1e-9


def ase_psd_to_channel_power(ase_psd_w_per_hz: np.ndarray, optical_bandwidth_hz: float) -> np.ndarray:
    return psd_to_dual_pol_power(ase_psd_w_per_hz, optical_bandwidth_hz)


def ase_psd_to_decision_variance_per_pol(ase_psd_w_per_hz: np.ndarray, receiver_bandwidth_hz: float) -> np.ndarray:
    return psd_to_dual_pol_power(ase_psd_w_per_hz, receiver_bandwidth_hz) / 2.0


class LumpedAmplifier:
    def __init__(self, amp_cfg: dict, symbol_rate_hz: float, roll_off: float):
        self.cfg = amp_cfg
        self.symbol_rate_hz = float(symbol_rate_hz)
        self.roll_off = float(roll_off)
        multiplier = float(amp_cfg.get("noise_bandwidth_multiplier", 1.0))
        self.optical_noise_bandwidth_hz = self.symbol_rate_hz * (1.0 + self.roll_off) * multiplier
        configured_receiver = amp_cfg.get("receiver_equivalent_noise_bandwidth_hz")
        self.receiver_equivalent_noise_bandwidth_hz = (
            float(configured_receiver)
            if configured_receiver is not None
            else receiver_equivalent_noise_bandwidth(self.symbol_rate_hz, self.roll_off)
        )
        # Backward-compatible name used elsewhere in the project.
        self.noise_bandwidth_hz = self.optical_noise_bandwidth_hz

    def equalize(self, span_output_w: np.ndarray, target_launch_w: np.ndarray, frequencies_hz: np.ndarray, wavelengths_nm: np.ndarray) -> AmplifierResult:
        span_output = np.maximum(np.asarray(span_output_w, dtype=float), 1e-30)
        target = np.maximum(np.asarray(target_launch_w, dtype=float), 1e-30)
        required_gain = target / span_output
        nf_db, max_gain_db = _band_parameters(wavelengths_nm, self.cfg)
        max_gain = 10.0 ** (max_gain_db / 10.0)
        gain = np.where(required_gain < 1.0, required_gain, np.minimum(required_gain, max_gain))
        output = span_output * gain
        active_gain = np.maximum(gain, 1.0)
        ase_psd = dual_pol_ase_psd_w_per_hz(frequencies_hz, active_gain, nf_db)
        ase_channel = ase_psd_to_channel_power(ase_psd, self.optical_noise_bandwidth_hz)
        ase_receiver = ase_psd_to_channel_power(ase_psd, self.receiver_equivalent_noise_bandwidth_hz)
        ase_01nm = ase_psd * reference_bandwidth_01nm_hz(wavelengths_nm)
        return AmplifierResult(
            gain_linear=gain,
            gain_db=10.0 * np.log10(np.maximum(gain, 1e-30)),
            output_signal_w=output,
            ase_psd_w_per_hz=ase_psd,
            ase_channel_w=ase_channel,
            residual_db=w_to_dbm(output) - w_to_dbm(target),
            optical_noise_bandwidth_hz=self.optical_noise_bandwidth_hz,
            receiver_equivalent_noise_bandwidth_hz=self.receiver_equivalent_noise_bandwidth_hz,
            ase_receiver_w=ase_receiver,
            ase_01nm_w=ase_01nm,
        )


def equivalent_distributed_raman_ase_psd(frequencies_hz: np.ndarray, pump_gain_linear: np.ndarray, equivalent_noise_figure_db: float) -> np.ndarray:
    frequency = np.asarray(frequencies_hz, dtype=float)
    gain = np.maximum(np.asarray(pump_gain_linear, dtype=float), 1.0)
    nf = np.full_like(frequency, float(equivalent_noise_figure_db))
    return dual_pol_ase_psd_w_per_hz(frequency, gain, nf)
