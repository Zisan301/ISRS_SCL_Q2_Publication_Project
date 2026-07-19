"""Reproducible DP-16QAM transmitter with explicit pilot/payload framing."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from isrs_scl.dsp.metrics import bits_to_16qam


@dataclass(frozen=True)
class TransmitterResult:
    bits: np.ndarray
    symbols: np.ndarray
    waveform: np.ndarray
    rrc_taps: np.ndarray
    samples_per_symbol: int
    pilot_mask: np.ndarray
    payload_mask: np.ndarray
    pilot_symbols: np.ndarray
    waveform_average_power_w: float
    symbol_decision_power_w: float
    pulse_energy: float
    sample_rate_hz: float | None
    occupied_bandwidth_hz: float | None
    requested_channel_power_w: float
    normalization_factor: float


def root_raised_cosine_taps(roll_off: float, span_symbols: int, samples_per_symbol: int) -> np.ndarray:
    beta, span, sps = float(roll_off), int(span_symbols), int(samples_per_symbol)
    if not 0 <= beta <= 1 or span < 2 or sps < 2:
        raise ValueError("Invalid RRC parameters")
    n = np.arange(-span * sps / 2, span * sps / 2 + 1, dtype=float)
    t = n / sps
    h = np.empty_like(t)
    for index, value in enumerate(t):
        if abs(value) < 1e-12:
            h[index] = 1.0 + beta * (4.0 / np.pi - 1.0)
        elif beta > 0 and abs(abs(value) - 1.0 / (4.0 * beta)) < 1e-10:
            h[index] = beta / np.sqrt(2.0) * (
                (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta))
                + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta))
            )
        else:
            numerator = np.sin(np.pi * value * (1.0 - beta)) + 4.0 * beta * value * np.cos(np.pi * value * (1.0 + beta))
            denominator = np.pi * value * (1.0 - (4.0 * beta * value) ** 2)
            h[index] = numerator / denominator
    h /= np.sqrt(np.sum(h**2))
    return h


def _pilot_mask(n_symbols: int, pilot_symbols: int, pilot_spacing: int | None) -> np.ndarray:
    mask = np.zeros(n_symbols, dtype=bool)
    preamble = min(max(int(pilot_symbols), 0), n_symbols)
    mask[:preamble] = True
    if pilot_spacing is not None and int(pilot_spacing) > 0:
        mask[preamble:: int(pilot_spacing)] = True
    return mask


def generate_dp16qam(
    n_symbols: int,
    samples_per_symbol: int,
    roll_off: float,
    rrc_span_symbols: int,
    seed: int,
    channel_power_w: float = 1.0,
    *,
    pilot_symbols: int = 0,
    pilot_spacing: int | None = None,
    symbol_rate_hz: float | None = None,
) -> TransmitterResult:
    if n_symbols < 64 or channel_power_w <= 0:
        raise ValueError("n_symbols must be >=64 and channel power positive")
    rng = np.random.default_rng(int(seed))
    bits = rng.integers(0, 2, size=(2, int(n_symbols), 4), dtype=np.uint8)
    symbols = np.vstack([bits_to_16qam(bits[pol]) for pol in range(2)])
    pilot_mask = _pilot_mask(int(n_symbols), int(pilot_symbols), pilot_spacing)
    payload_mask = ~pilot_mask
    taps = root_raised_cosine_taps(roll_off, rrc_span_symbols, samples_per_symbol)
    pulse_energy = float(np.sum(np.abs(taps) ** 2))
    # Symbols are unit-energy per polarization.  Scale the two polarizations so
    # their combined symbol-decision power equals channel_power_w.
    unscaled_decision_power = float(np.mean(np.sum(np.abs(symbols) ** 2, axis=0)))
    normalization = np.sqrt(float(channel_power_w) / max(unscaled_decision_power, 1e-30))
    waveforms = []
    for pol in range(2):
        up = np.zeros(int(n_symbols) * int(samples_per_symbol), dtype=complex)
        up[:: int(samples_per_symbol)] = symbols[pol] * normalization
        waveforms.append(np.convolve(up, taps, mode="full"))
    waveform = np.vstack(waveforms)
    decision_power = float(np.mean(np.sum(np.abs(symbols * normalization) ** 2, axis=0)))
    if not np.isclose(decision_power, channel_power_w, rtol=5e-3, atol=1e-15):
        raise FloatingPointError("Transmitter symbol-decision power normalization failed")
    waveform_power = float(np.mean(np.sum(np.abs(waveform) ** 2, axis=0)))
    sample_rate = float(symbol_rate_hz) * int(samples_per_symbol) if symbol_rate_hz else None
    occupied = float(symbol_rate_hz) * (1.0 + float(roll_off)) if symbol_rate_hz else None
    return TransmitterResult(
        bits=bits,
        symbols=symbols,
        waveform=waveform,
        rrc_taps=taps,
        samples_per_symbol=int(samples_per_symbol),
        pilot_mask=pilot_mask,
        payload_mask=payload_mask,
        pilot_symbols=symbols[:, pilot_mask],
        waveform_average_power_w=waveform_power,
        symbol_decision_power_w=decision_power,
        pulse_energy=pulse_energy,
        sample_rate_hz=sample_rate,
        occupied_bandwidth_hz=occupied,
        requested_channel_power_w=float(channel_power_w),
        normalization_factor=float(normalization),
    )
