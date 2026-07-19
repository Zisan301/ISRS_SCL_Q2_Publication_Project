"""2x2 polarization equalizers with explicit training/payload diagnostics."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


@dataclass(frozen=True)
class EqualizerResult:
    symbols: np.ndarray
    taps: np.ndarray
    error_trace: np.ndarray
    success: bool = True
    mode: str = ""
    training_symbols: int = 0
    payload_symbols: int = 0
    condition_number: float = np.nan
    training_mse: float = np.nan
    payload_residual_power: float = np.nan
    tap_energy: float = np.nan
    failure_reason: str = ""
    diagnostics: dict[str, float | str] = field(default_factory=dict)


def _validate(input_symbols: np.ndarray, taps: int) -> tuple[np.ndarray, int]:
    values = np.asarray(input_symbols, dtype=complex)
    taps = int(taps)
    if values.ndim != 2 or values.shape[0] != 2:
        raise ValueError("Input must have shape [2, N]")
    if taps < 1 or taps % 2 == 0 or values.shape[1] <= taps:
        raise ValueError("Equalizer taps must be odd and shorter than the signal")
    return values, taps


def cma_equalize_2x2(
    input_symbols: np.ndarray,
    taps: int = 11,
    step_size: float = 2e-4,
    training_symbols: int = 6000,
    modulus: float = 1.32,
    *,
    normalized_step: bool = True,
    divergence_limit: float = 100.0,
) -> EqualizerResult:
    values, taps = _validate(input_symbols, taps)
    if step_size <= 0:
        raise ValueError("CMA step size must be positive")
    weights = np.zeros((2, 2, taps), dtype=complex)
    center = taps // 2
    weights[0, 0, center] = weights[1, 1, center] = 1.0
    n_out = values.shape[1] - taps + 1
    output = np.zeros((2, n_out), dtype=complex)
    train_count = min(max(int(training_symbols), 32), n_out)
    trace: list[float] = []
    success, reason = True, ""
    for index in range(n_out):
        block = values[:, index : index + taps][:, ::-1]
        output[:, index] = np.einsum("oit,it->o", np.conj(weights), block)
        if index >= train_count:
            continue
        error = (float(modulus) - np.abs(output[:, index]) ** 2) * np.conj(output[:, index])
        normalization = max(float(np.sum(np.abs(block) ** 2)), 1e-12) if normalized_step else 1.0
        for out_pol in range(2):
            for in_pol in range(2):
                weights[out_pol, in_pol] += float(step_size) * error[out_pol] * block[in_pol] / normalization
        metric = float(np.mean(np.abs(float(modulus) - np.abs(output[:, index]) ** 2)))
        trace.append(metric)
        energy = float(np.sum(np.abs(weights) ** 2))
        if not np.isfinite(energy) or energy > divergence_limit:
            success, reason = False, "CMA diverged"
            break
        if index and index % 256 == 0:
            first, second = weights[0].reshape(-1), weights[1].reshape(-1)
            second -= np.vdot(first, second) / max(float(np.vdot(first, first).real), 1e-15) * first
            first /= np.sqrt(max(float(np.vdot(first, first).real), 1e-15))
            second /= np.sqrt(max(float(np.vdot(second, second).real), 1e-15))
            weights[0], weights[1] = first.reshape(2, taps), second.reshape(2, taps)
    tap_energy = float(np.sum(np.abs(weights) ** 2))
    return EqualizerResult(output, weights, np.asarray(trace), success, "cma", train_count, max(0, n_out - train_count), np.nan, float(np.mean(trace[-min(100, len(trace)):])) if trace else np.nan, np.nan, tap_energy, reason)


def pilot_aided_equalize_2x2(
    input_symbols: np.ndarray,
    reference_symbols: np.ndarray,
    taps: int = 11,
    training_symbols: int = 6000,
    ridge: float = 1e-5,
    *,
    training_mask: np.ndarray | None = None,
) -> EqualizerResult:
    values, taps = _validate(input_symbols, taps)
    reference = np.asarray(reference_symbols, dtype=complex)
    if reference.ndim != 2 or reference.shape[0] != 2:
        raise ValueError("Reference symbols must have shape [2, N]")
    n_out = values.shape[1] - taps + 1
    center = taps // 2
    usable = min(n_out, reference.shape[1] - center)
    if usable < 64:
        raise ValueError("Insufficient overlapping symbols")
    x_windows = sliding_window_view(values[0], taps)[:usable]
    y_windows = sliding_window_view(values[1], taps)[:usable]
    design = np.concatenate([x_windows, y_windows], axis=1)
    target = reference[:, center : center + usable].T
    if training_mask is None:
        mask = np.zeros(usable, dtype=bool)
        mask[: min(max(int(training_symbols), 32), usable)] = True
    else:
        original = np.asarray(training_mask, dtype=bool)
        mask = original[center : center + usable]
        if mask.sum() < 32:
            raise ValueError("At least 32 training/pilot symbols are required")
    train_design, train_target = design[mask], target[mask]
    gram = train_design.conj().T @ train_design
    condition = float(np.linalg.cond(gram))
    regularizer = float(ridge) * np.eye(gram.shape[0], dtype=complex)
    try:
        full_coefficients = np.linalg.solve(gram + regularizer, train_design.conj().T @ train_target)
        # A short training set can make a long FIR overfit AWGN.  Compare it with
        # a memoryless 2x2 pilot solution and retain the lower training-error
        # model.  This is model selection on pilots, never on payload labels.
        center_design = np.column_stack([x_windows[:, center], y_windows[:, center]])
        train_center = center_design[mask]
        center_gram = train_center.conj().T @ train_center
        center_coefficients = np.linalg.solve(
            center_gram + float(ridge) * np.eye(2, dtype=complex),
            train_center.conj().T @ train_target,
        )
        memory_coefficients = np.zeros_like(full_coefficients)
        memory_coefficients[center] = center_coefficients[0]
        memory_coefficients[taps + center] = center_coefficients[1]
        full_mse = float(np.mean(np.abs(train_design @ full_coefficients - train_target) ** 2))
        memory_mse = float(np.mean(np.abs(train_design @ memory_coefficients - train_target) ** 2))
        coefficients = memory_coefficients if memory_mse <= full_mse * 1.02 else full_coefficients
    except np.linalg.LinAlgError as exc:
        return EqualizerResult(np.empty((2, 0), complex), np.zeros((2, 2, taps), complex), np.array([]), False, "pilot_aided", int(mask.sum()), int((~mask).sum()), condition, np.nan, np.nan, np.nan, f"ill-conditioned solve: {exc}")
    output = (design @ coefficients).T
    estimated_taps = np.zeros((2, 2, taps), dtype=complex)
    for pol in range(2):
        estimated_taps[pol, 0] = coefficients[:taps, pol]
        estimated_taps[pol, 1] = coefficients[taps:, pol]
    prediction = train_design @ coefficients
    training_mse = float(np.mean(np.abs(prediction - train_target) ** 2))
    payload = ~mask
    payload_residual = float(np.mean(np.abs(output[:, payload].T - target[payload]) ** 2)) if payload.any() else np.nan
    trace = np.array([float(np.mean(np.abs(prediction[start : start + 256] - train_target[start : start + 256]) ** 2)) for start in range(0, len(prediction), 256)])
    tap_energy = float(np.sum(np.abs(estimated_taps) ** 2))
    success = np.isfinite(condition) and condition < 1e12 and np.isfinite(training_mse)
    return EqualizerResult(output, estimated_taps, trace, success, "pilot_aided", int(mask.sum()), int(payload.sum()), condition, training_mse, payload_residual, tap_energy, "" if success else "ill-conditioned or non-finite equalizer")
