"""Blind and pilot-aided carrier-phase recovery with reliability diagnostics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from isrs_scl.dsp.metrics import constellation_16qam


@dataclass(frozen=True)
class CarrierRecoveryResult:
    corrected_symbols: np.ndarray
    phase_trace: np.ndarray
    cycle_slips: int
    reliability: float
    success: bool
    mode: str
    failure_reason: str = ""

    def __iter__(self) -> Iterator[np.ndarray]:
        # Backward compatibility with ``corrected, phase = blind_phase_search(...)``.
        yield self.corrected_symbols
        yield self.phase_trace


def _smooth(values: np.ndarray, width: int) -> np.ndarray:
    width = max(int(width), 1)
    if width == 1 or values.size < width:
        return values
    kernel = np.ones(width) / width
    padded = np.pad(values, (width // 2, width - 1 - width // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def blind_phase_search(
    symbols: np.ndarray,
    trial_phases: int = 64,
    block_symbols: int = 64,
    *,
    overlap: float = 0.5,
    smoothing_blocks: int = 3,
    minimum_reliability: float = 0.02,
) -> CarrierRecoveryResult:
    values = np.asarray(symbols, dtype=complex)
    if values.ndim != 2 or values.shape[0] != 2:
        raise ValueError("Input must have shape [2, N]")
    block = max(int(block_symbols), 8)
    step = max(1, int(round(block * (1.0 - float(overlap)))))
    const, _ = constellation_16qam()
    phases = np.linspace(-np.pi / 4.0, np.pi / 4.0, int(trial_phases), endpoint=False)
    centers: list[int] = []
    raw_phases: list[float] = []
    reliabilities: list[float] = []
    previous = 0.0
    slips = 0
    for start in range(0, values.shape[1], step):
        stop = min(values.shape[1], start + block)
        if stop - start < 8:
            break
        sample = values[:, start:stop]
        costs = np.empty(phases.size)
        for index, phase in enumerate(phases):
            rotated = sample * np.exp(-1j * phase)
            distances = np.min(np.abs(rotated[:, :, None] - const[None, None, :]) ** 2, axis=2)
            costs[index] = float(np.mean(distances))
        order = np.argsort(costs)
        raw = float(phases[order[0]])
        candidates = raw + np.arange(-4, 5) * np.pi / 2.0
        phase = float(candidates[np.argmin(np.abs(candidates - previous))])
        if abs(phase - previous) > np.pi / 4:
            slips += 1
        previous = phase
        gap = float((costs[order[1]] - costs[order[0]]) / max(abs(costs[order[1]]), 1e-15))
        centers.append((start + stop - 1) // 2)
        raw_phases.append(phase)
        reliabilities.append(gap)
    if not centers:
        return CarrierRecoveryResult(values.copy(), np.zeros(values.shape[1]), 0, 0.0, False, "bps", "no valid BPS blocks")
    smoothed = _smooth(np.unwrap(np.asarray(raw_phases) * 4.0) / 4.0, smoothing_blocks)
    trace = np.interp(np.arange(values.shape[1]), np.asarray(centers), smoothed, left=smoothed[0], right=smoothed[-1])
    corrected = values * np.exp(-1j * trace)[None, :]
    reliability = float(np.median(reliabilities))
    success = reliability >= float(minimum_reliability) and slips <= max(2, len(centers) // 20)
    reason = "" if success else f"low BPS reliability or cycle slips (reliability={reliability:.4g}, slips={slips})"
    return CarrierRecoveryResult(corrected, trace, slips, reliability, success, "bps", reason)


def pilot_aided_phase_recovery(
    symbols: np.ndarray,
    reference_symbols: np.ndarray,
    pilot_mask: np.ndarray,
    *,
    minimum_reliability: float = 0.02,
) -> CarrierRecoveryResult:
    values = np.asarray(symbols, dtype=complex)
    reference = np.asarray(reference_symbols, dtype=complex)
    mask = np.asarray(pilot_mask, dtype=bool).reshape(-1)
    n = min(values.shape[1], reference.shape[1], mask.size)
    values, reference, mask = values[:, :n], reference[:, :n], mask[:n]
    pilot_indices = np.flatnonzero(mask)
    if values.ndim != 2 or values.shape[0] != 2 or pilot_indices.size < 8:
        return CarrierRecoveryResult(values.copy(), np.zeros(n), 0, 0.0, False, "pilot_aided", "insufficient pilots")
    phasors = np.sum(values[:, pilot_indices] * np.conj(reference[:, pilot_indices]), axis=0)
    magnitude = np.abs(phasors)
    valid = magnitude > 1e-15
    if valid.sum() < 8:
        return CarrierRecoveryResult(values.copy(), np.zeros(n), 0, 0.0, False, "pilot_aided", "pilot phasors are degenerate")
    phase_samples = np.unwrap(np.angle(phasors[valid]))
    indices = pilot_indices[valid]
    trace = np.interp(np.arange(n), indices, phase_samples, left=phase_samples[0], right=phase_samples[-1])
    jumps = np.diff(phase_samples)
    slips = int(np.sum(np.abs(jumps) > np.pi / 2))
    coherence = float(np.abs(np.mean(np.exp(1j * (np.angle(phasors[valid]) - phase_samples)))))
    residual = values[:, indices] * np.exp(-1j * trace[indices])[None, :] - reference[:, indices]
    reliability = float(1.0 / (1.0 + np.mean(np.abs(residual) ** 2))) * coherence
    success = reliability >= minimum_reliability and slips == 0
    reason = "" if success else f"pilot phase recovery failed (reliability={reliability:.4g}, slips={slips})"
    return CarrierRecoveryResult(values * np.exp(-1j * trace)[None, :], trace, slips, reliability, success, "pilot_aided", reason)
