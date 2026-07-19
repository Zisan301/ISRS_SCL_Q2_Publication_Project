"""Representative-channel propagation and held-out coherent DSP validation.

This module does not claim full-grid waveform SSFM.  It injects the power-domain
ASE/NLI budget into one representative channel using the explicit conversions in
:mod:`isrs_scl.dsp.noise`, performs acquisition on pilots/training symbols, and
computes final metrics only on a disjoint payload set.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from isrs_scl.dsp.carrier_recovery import (
    CarrierRecoveryResult,
    blind_phase_search,
    pilot_aided_phase_recovery,
)
from isrs_scl.dsp.equalizer import EqualizerResult, cma_equalize_2x2, pilot_aided_equalize_2x2
from isrs_scl.dsp.metrics import estimate_complex_gain, sample_metrics_16qam
from isrs_scl.dsp.noise import (
    add_complex_awgn_from_sample_variance,
    integrated_receiver_power_to_sample_variance,
    sample_variance_to_decision_variance,
)
from isrs_scl.dsp.transmitter import TransmitterResult


@dataclass(frozen=True)
class WaveformChannelResult:
    received_waveform: np.ndarray
    accumulated_cd_length_m: float
    sample_rate_hz: float
    injected_ase_w: float = 0.0
    injected_nli_w: float = 0.0
    injected_sample_variance_per_pol: float = 0.0
    expected_decision_variance_per_pol: float = 0.0
    phase_noise_applied: bool = False


@dataclass(frozen=True)
class ReceiverResult:
    recovered_symbols: np.ndarray
    aligned_tx_symbols: np.ndarray
    metrics_per_pol: tuple[dict[str, float], dict[str, float]]
    equalizer: EqualizerResult
    phase_trace: np.ndarray
    equalizer_mode: str
    timing_phase: int = 0
    acquisition_success: bool = True
    cycle_slips: int = 0
    training_symbols: int = 0
    payload_symbols: int = 0
    failure_reason: str = ""
    diagnostics: dict[str, float | str] = field(default_factory=dict)


def _frequency_axis(n: int, sample_rate_hz: float) -> np.ndarray:
    return np.fft.fftfreq(int(n), d=1.0 / float(sample_rate_hz))


def apply_chromatic_dispersion(waveform: np.ndarray, beta2_s2_per_m: float, length_m: float, sample_rate_hz: float) -> np.ndarray:
    values = np.asarray(waveform, dtype=complex)
    if values.ndim != 2 or values.shape[0] != 2:
        raise ValueError("waveform must have shape [2, N]")
    omega = 2.0 * np.pi * _frequency_axis(values.shape[1], sample_rate_hz)
    transfer = np.exp(-0.5j * float(beta2_s2_per_m) * omega**2 * float(length_m))
    return np.fft.ifft(np.fft.fft(values, axis=1) * transfer[None, :], axis=1)


def _random_unitary(rng: np.random.Generator) -> np.ndarray:
    matrix = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    q, r = np.linalg.qr(matrix)
    phases = np.diag(r)
    q *= np.exp(-1j * np.angle(phases))[None, :]
    return q


def apply_pmd_and_rotation(waveform: np.ndarray, sample_rate_hz: float, dgd_s: float, rng: np.random.Generator) -> np.ndarray:
    values = np.asarray(waveform, dtype=complex)
    rotated = _random_unitary(rng) @ values
    frequency = _frequency_axis(values.shape[1], sample_rate_hz)
    spectrum = np.fft.fft(rotated, axis=1)
    spectrum[0] *= np.exp(-1j * np.pi * frequency * float(dgd_s))
    spectrum[1] *= np.exp(+1j * np.pi * frequency * float(dgd_s))
    return np.fft.ifft(spectrum, axis=1)


def apply_laser_phase_noise(waveform: np.ndarray, linewidth_hz: float, sample_rate_hz: float, rng: np.random.Generator) -> np.ndarray:
    values = np.asarray(waveform, dtype=complex)
    if linewidth_hz <= 0:
        return values.copy()
    increment_std = np.sqrt(2.0 * np.pi * float(linewidth_hz) / float(sample_rate_hz))
    phase = np.cumsum(rng.normal(scale=increment_std, size=values.shape[1]))
    return values * np.exp(1j * phase)[None, :]


def apply_frequency_offset(waveform: np.ndarray, frequency_offset_hz: float, sample_rate_hz: float) -> np.ndarray:
    values = np.asarray(waveform, dtype=complex)
    if frequency_offset_hz == 0:
        return values.copy()
    time = np.arange(values.shape[1], dtype=float) / float(sample_rate_hz)
    return values * np.exp(1j * 2.0 * np.pi * float(frequency_offset_hz) * time)[None, :]


def add_complex_awgn_dp(waveform: np.ndarray, total_dual_pol_noise_w: float, rng: np.random.Generator, *, rrc_taps: np.ndarray | None = None) -> np.ndarray:
    """Backward-compatible AWGN helper.

    When ``rrc_taps`` is provided, ``total_dual_pol_noise_w`` is interpreted as
    receiver decision-domain integrated noise power.  Without taps it retains the
    legacy interpretation as direct dual-pol sample variance.
    """
    if rrc_taps is None:
        sample_variance = max(float(total_dual_pol_noise_w), 0.0) / 2.0
    else:
        sample_variance = integrated_receiver_power_to_sample_variance(float(total_dual_pol_noise_w), rrc_taps)
    return add_complex_awgn_from_sample_variance(waveform, sample_variance, rng)


def propagate_representative_channel(
    tx: TransmitterResult,
    symbol_rate_hz: float,
    center_wavelength_nm: float,
    beta2_s2_per_m: float,
    span_length_m: float,
    n_spans: int,
    ase_w_per_span: float,
    nli_w_per_span: float,
    pmd_ps_sqrt_km: float,
    linewidth_hz: float,
    apply_pmd: bool,
    apply_phase_noise: bool,
    seed: int,
    nli_coherence_epsilon: float = 0.0,
    carrier_frequency_offset_hz: float = 0.0,
    *,
    noise_values_are_sample_variance: bool = False,
) -> WaveformChannelResult:
    del center_wavelength_nm
    if n_spans < 1 or nli_coherence_epsilon < 0:
        raise ValueError("Invalid span count or NLI coherence exponent")
    if ase_w_per_span < 0 or nli_w_per_span < 0:
        raise ValueError("ASE/NLI powers must be non-negative")
    rng = np.random.default_rng(int(seed))
    sample_rate = float(symbol_rate_hz) * tx.samples_per_symbol
    waveform = np.asarray(tx.waveform, dtype=complex).copy()
    exponent = 1.0 + float(nli_coherence_epsilon)
    total_ase, total_nli, total_sample_variance = 0.0, 0.0, 0.0
    for span_index in range(int(n_spans)):
        waveform = apply_chromatic_dispersion(waveform, beta2_s2_per_m, span_length_m, sample_rate)
        if apply_pmd:
            dgd_ps = float(pmd_ps_sqrt_km) * np.sqrt(float(span_length_m) / 1000.0)
            waveform = apply_pmd_and_rotation(waveform, sample_rate, dgd_ps * 1e-12, rng)
        previous, current = float(span_index) ** exponent, float(span_index + 1) ** exponent
        nli_increment = (current - previous) * float(nli_w_per_span)
        integrated_increment = float(ase_w_per_span) + nli_increment
        if noise_values_are_sample_variance:
            sample_variance = integrated_increment / 2.0
        else:
            sample_variance = integrated_receiver_power_to_sample_variance(integrated_increment, tx.rrc_taps)
        waveform = add_complex_awgn_from_sample_variance(waveform, sample_variance, rng)
        total_ase += float(ase_w_per_span)
        total_nli += nli_increment
        total_sample_variance += sample_variance
    if apply_phase_noise:
        waveform = apply_laser_phase_noise(waveform, linewidth_hz, sample_rate, rng)
    waveform = apply_frequency_offset(waveform, carrier_frequency_offset_hz, sample_rate)
    expected_decision = sample_variance_to_decision_variance(total_sample_variance, tx.rrc_taps)
    return WaveformChannelResult(
        waveform,
        float(n_spans) * float(span_length_m),
        sample_rate,
        total_ase,
        total_nli,
        total_sample_variance,
        expected_decision,
        bool(apply_phase_noise and linewidth_hz > 0),
    )


def _matched_filter(waveform: np.ndarray, taps: np.ndarray) -> np.ndarray:
    return np.vstack([np.convolve(np.asarray(waveform)[pol], taps, mode="full") for pol in range(2)])


def matched_filter_and_sample(
    waveform: np.ndarray,
    rrc_taps: np.ndarray,
    samples_per_symbol: int,
    n_symbols: int,
    timing_phase: int | None = None,
    *,
    reference_symbols: np.ndarray | None = None,
    pilot_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, int] | np.ndarray:
    filtered = _matched_filter(waveform, rrc_taps)
    nominal_start = len(rrc_taps) - 1
    sps = int(samples_per_symbol)
    if sps < 1:
        raise ValueError("samples_per_symbol must be positive")
    if timing_phase is not None:
        phase = int(timing_phase)
        if not 0 <= phase < sps:
            raise ValueError("timing_phase is outside the samples-per-symbol range")
        return filtered[:, nominal_start + phase :: sps][:, :n_symbols]
    candidates: list[tuple[float, int, np.ndarray]] = []
    reference = None if reference_symbols is None else np.asarray(reference_symbols, dtype=complex)
    mask = None if pilot_mask is None else np.asarray(pilot_mask, dtype=bool)
    for phase in range(sps):
        sampled = filtered[:, nominal_start + phase :: sps][:, :n_symbols]
        if sampled.shape[1] < 64:
            continue
        power = np.mean(np.abs(sampled) ** 2, axis=1, keepdims=True)
        normalized = sampled / np.sqrt(np.maximum(power, 1e-30))
        if reference is not None and mask is not None:
            n = min(sampled.shape[1], reference.shape[1], mask.size)
            valid = mask[:n]
            if valid.sum() < 16:
                continue
            # Timing uses only pilot symbols and a per-polarization scalar.
            score = 0.0
            for pol in range(2):
                gain = estimate_complex_gain(reference[pol, :n][valid], normalized[pol, :n][valid])
                score += float(np.mean(np.abs(gain * normalized[pol, :n][valid] - reference[pol, :n][valid]) ** 2))
        else:
            fourth = float(np.mean(np.abs(normalized) ** 4))
            score = abs(fourth - 1.32)
        candidates.append((score, phase, sampled))
    if not candidates:
        raise ValueError("No valid symbol timing phase was found")
    _, selected_phase, selected = min(candidates, key=lambda item: item[0])
    return selected, selected_phase


def _training_alignment(
    reference: np.ndarray,
    recovered: np.ndarray,
    training_mask: np.ndarray,
    max_delay: int = 96,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int], tuple[int, int], tuple[complex, complex], int]:
    n = min(reference.shape[1], recovered.shape[1], training_mask.size)
    reference, recovered, mask = reference[:, :n], recovered[:, :n], training_mask[:n]
    best: tuple[float, Any] | None = None
    for permutation in ((0, 1), (1, 0)):
        for rotation0 in range(4):
            for rotation1 in range(4):
                rotated = np.vstack([recovered[permutation[0]] * (1j**rotation0), recovered[permutation[1]] * (1j**rotation1)])
                for lag in range(-max_delay, max_delay + 1):
                    if lag >= 0:
                        count = min(reference.shape[1], rotated.shape[1] - lag)
                        ref = reference[:, :count]
                        rec = rotated[:, lag : lag + count]
                        valid = mask[:count]
                    else:
                        count = min(reference.shape[1] + lag, rotated.shape[1])
                        ref = reference[:, -lag : -lag + count]
                        rec = rotated[:, :count]
                        valid = mask[-lag : -lag + count]
                    if valid.sum() < 32:
                        continue
                    gains = tuple(estimate_complex_gain(ref[pol, valid], rec[pol, valid]) for pol in range(2))
                    aligned = np.vstack([gains[pol] * rec[pol] for pol in range(2)])
                    score = float(np.mean(np.abs(aligned[:, valid] - ref[:, valid]) ** 2))
                    if best is None or score < best[0]:
                        best = (score, ref.copy(), aligned.copy(), valid.copy(), permutation, (rotation0, rotation1), gains, lag)
    if best is None:
        raise ValueError("Unable to acquire training alignment")
    _, ref, aligned, valid, permutation, rotations, gains, lag = best
    return ref, aligned, valid, permutation, rotations, gains, lag


def coherent_receiver(
    channel: WaveformChannelResult,
    tx: TransmitterResult,
    beta2_s2_per_m: float,
    cma_taps: int,
    cma_step_size: float,
    cma_training_symbols: int,
    bps_trial_phases: int,
    bps_block_symbols: int,
    equalizer_mode: str = "pilot_aided",
    equalizer_ridge: float = 1e-5,
    bootstrap_samples: int = 0,
    bootstrap_block_symbols: int = 256,
    bootstrap_seed: int = 0,
    *,
    carrier_recovery_mode: str = "pilot_aided",
    bps_overlap: float = 0.5,
    bps_smoothing_blocks: int = 3,
    minimum_reliability: float = 0.02,
) -> ReceiverResult:
    compensated = apply_chromatic_dispersion(channel.received_waveform, beta2_s2_per_m, -channel.accumulated_cd_length_m, channel.sample_rate_hz)
    sampled, timing_phase = matched_filter_and_sample(
        compensated,
        tx.rrc_taps,
        tx.samples_per_symbol,
        tx.symbols.shape[1],
        reference_symbols=tx.symbols,
        pilot_mask=tx.pilot_mask,
    )
    power_scale = np.sqrt(np.maximum(np.mean(np.abs(sampled) ** 2, axis=1, keepdims=True), 1e-30))
    normalized = sampled / power_scale
    mode = str(equalizer_mode).lower()
    if mode in {"pilot", "pilot_aided", "training_aided", "data_aided"}:
        equalized = pilot_aided_equalize_2x2(normalized, tx.symbols, taps=cma_taps, training_symbols=cma_training_symbols, ridge=equalizer_ridge, training_mask=tx.pilot_mask)
        mode = "pilot_aided"
    elif mode == "cma":
        equalized = cma_equalize_2x2(normalized, taps=cma_taps, step_size=cma_step_size, training_symbols=cma_training_symbols)
    else:
        raise ValueError("equalizer_mode must be pilot_aided or cma")
    center = int(cma_taps) // 2
    usable = equalized.symbols.shape[1]
    reference = tx.symbols[:, center : center + usable]
    pilot_mask = tx.pilot_mask[center : center + usable]
    if not equalized.success:
        empty_metrics = ({"acquisition_success": 0.0}, {"acquisition_success": 0.0})
        return ReceiverResult(equalized.symbols, reference, empty_metrics, equalized, np.zeros(usable), mode, timing_phase, False, 0, int(pilot_mask.sum()), int((~pilot_mask).sum()), equalized.failure_reason, {"normalization_x": float(power_scale[0, 0]), "normalization_y": float(power_scale[1, 0])})

    carrier_mode = str(carrier_recovery_mode).lower()
    if carrier_mode in {"pilot", "pilot_aided"}:
        carrier = pilot_aided_phase_recovery(equalized.symbols, reference, pilot_mask, minimum_reliability=minimum_reliability)
    elif carrier_mode in {"bps", "blind_phase_search"}:
        carrier = blind_phase_search(equalized.symbols, bps_trial_phases, bps_block_symbols, overlap=bps_overlap, smoothing_blocks=bps_smoothing_blocks, minimum_reliability=minimum_reliability)
    else:
        raise ValueError("carrier_recovery_mode must be pilot_aided or bps")

    try:
        aligned_tx, aligned_rx, aligned_training_mask, permutation, rotations, gains, lag = _training_alignment(reference, carrier.corrected_symbols, pilot_mask)
        payload_mask = ~aligned_training_mask
        if payload_mask.sum() < 64:
            raise ValueError("Too few held-out payload symbols after alignment")
        metrics = tuple(
            sample_metrics_16qam(
                aligned_tx[pol, payload_mask],
                aligned_rx[pol, payload_mask],
                calibration_gain=1.0 + 0j,
                allow_payload_fit=False,
                acquisition_success=carrier.success,
                bootstrap_samples=bootstrap_samples,
                bootstrap_block_symbols=bootstrap_block_symbols,
                bootstrap_seed=int(bootstrap_seed) + pol,
            )
            for pol in range(2)
        )
        measured_decision_variance = float(np.mean([item["noise_variance"] for item in metrics]))
        expected = float(channel.expected_decision_variance_per_pol)
        variance_error_db = 10.0 * np.log10(max(measured_decision_variance, 1e-30) / max(expected, 1e-30)) if expected > 0 else np.nan
        success = bool(carrier.success and np.isfinite(measured_decision_variance))
        reason = carrier.failure_reason if not success else ""
        diagnostics: dict[str, float | str] = {
            "equalizer_mode": mode,
            "carrier_recovery_mode": carrier.mode,
            "carrier_reliability": carrier.reliability,
            "cycle_slips": float(carrier.cycle_slips),
            "timing_phase": float(timing_phase),
            "delay_symbols": float(lag),
            "polarization_permutation": str(permutation),
            "quadrant_rotations": str(rotations),
            "training_symbols": float(aligned_training_mask.sum()),
            "payload_symbols": float(payload_mask.sum()),
            "normalization_x": float(power_scale[0, 0]),
            "normalization_y": float(power_scale[1, 0]),
            "gain_x_real": float(np.real(gains[0])), "gain_x_imag": float(np.imag(gains[0])),
            "gain_y_real": float(np.real(gains[1])), "gain_y_imag": float(np.imag(gains[1])),
            "injected_ase_w": float(channel.injected_ase_w), "injected_nli_w": float(channel.injected_nli_w),
            "expected_decision_variance_per_pol": expected,
            "measured_payload_error_variance": measured_decision_variance,
            "decision_variance_consistency_error_db": float(variance_error_db),
        }
        return ReceiverResult(aligned_rx[:, payload_mask], aligned_tx[:, payload_mask], metrics, equalized, carrier.phase_trace, mode, timing_phase, success, carrier.cycle_slips, int(aligned_training_mask.sum()), int(payload_mask.sum()), reason, diagnostics)
    except Exception as exc:
        metrics = ({"acquisition_success": 0.0}, {"acquisition_success": 0.0})
        return ReceiverResult(carrier.corrected_symbols, reference, metrics, equalized, carrier.phase_trace, mode, timing_phase, False, carrier.cycle_slips, int(pilot_mask.sum()), int((~pilot_mask).sum()), f"{type(exc).__name__}: {exc}", {"carrier_reliability": carrier.reliability})
