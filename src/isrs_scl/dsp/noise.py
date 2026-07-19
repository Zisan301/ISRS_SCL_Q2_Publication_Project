"""Explicit optical-PSD, sample-domain, and decision-domain noise conversions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class NoiseBandwidths:
    optical_hz: float
    receiver_equivalent_hz: float
    sample_rate_hz: float
    symbol_rate_hz: float
    roll_off: float


@dataclass(frozen=True)
class NoiseBudget:
    dual_pol_psd_w_per_hz: float
    dual_pol_optical_power_w: float
    dual_pol_receiver_power_w: float
    sample_variance_per_pol: float
    decision_variance_per_pol: float
    bandwidths: NoiseBandwidths


def receiver_equivalent_noise_bandwidth(symbol_rate_hz: float, roll_off: float, *, multiplier: float = 1.0) -> float:
    """Equivalent two-sided noise bandwidth used by the analytical receiver model.

    The project models a root-raised-cosine transmitter/matched-filter pair.  The
    default equivalent bandwidth is ``Rs * (1 + roll_off)``; callers can supply a
    documented calibrated multiplier without changing the remaining conversions.
    """
    rs = float(symbol_rate_hz)
    beta = float(roll_off)
    if rs <= 0 or not 0 <= beta <= 1 or multiplier <= 0:
        raise ValueError("Invalid symbol rate, roll-off, or bandwidth multiplier")
    return rs * (1.0 + beta) * float(multiplier)


def psd_to_dual_pol_power(psd_w_per_hz: np.ndarray | float, bandwidth_hz: np.ndarray | float) -> np.ndarray:
    psd = np.asarray(psd_w_per_hz, dtype=float)
    bandwidth = np.asarray(bandwidth_hz, dtype=float)
    if np.any(psd < 0) or np.any(bandwidth <= 0):
        raise ValueError("PSD must be non-negative and bandwidth positive")
    return psd * bandwidth


def dual_pol_power_to_psd(power_w: np.ndarray | float, bandwidth_hz: np.ndarray | float) -> np.ndarray:
    power = np.asarray(power_w, dtype=float)
    bandwidth = np.asarray(bandwidth_hz, dtype=float)
    if np.any(power < 0) or np.any(bandwidth <= 0):
        raise ValueError("Power must be non-negative and bandwidth positive")
    return power / bandwidth


def matched_filter_noise_gain(rrc_taps: np.ndarray) -> float:
    taps = np.asarray(rrc_taps, dtype=float).reshape(-1)
    if taps.size < 2 or not np.isfinite(taps).all():
        raise ValueError("RRC taps must be finite and non-empty")
    return float(np.sum(np.abs(taps) ** 2))


def decision_variance_to_sample_variance(decision_variance_per_pol: float, rrc_taps: np.ndarray) -> float:
    gain = matched_filter_noise_gain(rrc_taps)
    return float(decision_variance_per_pol) / max(gain, 1e-30)


def sample_variance_to_decision_variance(sample_variance_per_pol: float, rrc_taps: np.ndarray) -> float:
    return float(sample_variance_per_pol) * matched_filter_noise_gain(rrc_taps)


def optical_psd_to_waveform_budget(
    dual_pol_psd_w_per_hz: float,
    *,
    optical_bandwidth_hz: float,
    symbol_rate_hz: float,
    roll_off: float,
    sample_rate_hz: float,
    rrc_taps: np.ndarray,
    receiver_bandwidth_multiplier: float = 1.0,
) -> NoiseBudget:
    optical = float(psd_to_dual_pol_power(dual_pol_psd_w_per_hz, optical_bandwidth_hz))
    receiver_bw = receiver_equivalent_noise_bandwidth(symbol_rate_hz, roll_off, multiplier=receiver_bandwidth_multiplier)
    receiver = float(psd_to_dual_pol_power(dual_pol_psd_w_per_hz, receiver_bw))
    decision_per_pol = receiver / 2.0
    sample_per_pol = decision_variance_to_sample_variance(decision_per_pol, rrc_taps)
    return NoiseBudget(
        dual_pol_psd_w_per_hz=float(dual_pol_psd_w_per_hz),
        dual_pol_optical_power_w=optical,
        dual_pol_receiver_power_w=receiver,
        sample_variance_per_pol=sample_per_pol,
        decision_variance_per_pol=decision_per_pol,
        bandwidths=NoiseBandwidths(float(optical_bandwidth_hz), receiver_bw, float(sample_rate_hz), float(symbol_rate_hz), float(roll_off)),
    )


def integrated_receiver_power_to_sample_variance(total_dual_pol_receiver_noise_w: float, rrc_taps: np.ndarray) -> float:
    if total_dual_pol_receiver_noise_w < 0:
        raise ValueError("Noise power must be non-negative")
    return decision_variance_to_sample_variance(float(total_dual_pol_receiver_noise_w) / 2.0, rrc_taps)


def add_complex_awgn_from_sample_variance(waveform: np.ndarray, sample_variance_per_pol: float, rng: np.random.Generator) -> np.ndarray:
    values = np.asarray(waveform, dtype=complex)
    if values.ndim != 2 or values.shape[0] != 2:
        raise ValueError("waveform must have shape [2, N]")
    variance = float(sample_variance_per_pol)
    if variance < 0:
        raise ValueError("sample variance must be non-negative")
    sigma = np.sqrt(variance / 2.0)
    noise = sigma * (rng.standard_normal(values.shape) + 1j * rng.standard_normal(values.shape))
    return values + noise


def monte_carlo_decision_variance(
    rrc_taps: np.ndarray,
    *,
    sample_variance_per_pol: float,
    samples_per_symbol: int,
    symbols: int = 20000,
    seed: int = 0,
) -> float:
    """Numerically verify matched-filter decision variance for one polarization."""
    if symbols < 100 or samples_per_symbol < 1:
        raise ValueError("Insufficient symbols or invalid samples_per_symbol")
    rng = np.random.default_rng(seed)
    noise = np.sqrt(sample_variance_per_pol / 2.0) * (
        rng.standard_normal(symbols * samples_per_symbol) + 1j * rng.standard_normal(symbols * samples_per_symbol)
    )
    filtered = np.convolve(noise, np.asarray(rrc_taps), mode="full")
    start = len(rrc_taps) - 1
    decisions = filtered[start::samples_per_symbol][:symbols]
    return float(np.var(decisions))
