"""Publication-grade analytical and held-out DP-16QAM metrics."""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable, Mapping

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize_scalar
from scipy.special import erfc, erfcinv, logsumexp

_LEVELS = np.array([-3.0, -1.0, 1.0, 3.0])
_LEVEL_BITS = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.uint8)
_CONSTELLATION = np.array([i + 1j * q for i in _LEVELS for q in _LEVELS], dtype=complex)
_CONSTELLATION /= np.sqrt(np.mean(np.abs(_CONSTELLATION) ** 2))
_CONST_BITS = np.array([np.concatenate((_LEVEL_BITS[ii], _LEVEL_BITS[qq])) for ii in range(4) for qq in range(4)], dtype=np.uint8)


def constellation_16qam() -> tuple[np.ndarray, np.ndarray]:
    return _CONSTELLATION.copy(), _CONST_BITS.copy()


def analytical_ber_16qam(snr_linear: np.ndarray | float) -> np.ndarray:
    snr = np.maximum(np.asarray(snr_linear, dtype=float), 1e-15)
    return 3.0 / 8.0 * erfc(np.sqrt(snr / 10.0))


def normalized_evm_from_snr(snr_linear: np.ndarray | float) -> np.ndarray:
    return 1.0 / np.sqrt(np.maximum(np.asarray(snr_linear, dtype=float), 1e-15))


def q_factor_db_from_ber(ber: np.ndarray | float) -> np.ndarray:
    probability = np.clip(np.asarray(ber, dtype=float), 1e-300, 0.499999999)
    q_linear = np.sqrt(2.0) * erfcinv(2.0 * probability)
    return 20.0 * np.log10(np.maximum(q_linear, 1e-15))


def bits_to_16qam(bits: np.ndarray) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.uint8)
    if bits.shape[-1] != 4:
        raise ValueError("Last dimension must contain four bits")
    flat = bits.reshape(-1, 4)
    table = np.empty(16, dtype=complex)
    for index, label in enumerate(_CONST_BITS):
        code = int(label[0] * 8 + label[1] * 4 + label[2] * 2 + label[3])
        table[code] = _CONSTELLATION[index]
    packed = flat[:, 0] * 8 + flat[:, 1] * 4 + flat[:, 2] * 2 + flat[:, 3]
    return table[packed]


def nearest_16qam(symbols: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(symbols, dtype=complex).reshape(-1)
    distances = np.abs(values[:, None] - _CONSTELLATION[None, :]) ** 2
    index = np.argmin(distances, axis=1)
    return _CONSTELLATION[index], _CONST_BITS[index], index


def exact_bit_llrs_16qam(symbols: np.ndarray, noise_variance: float) -> np.ndarray:
    values = np.asarray(symbols, dtype=complex).reshape(-1)
    n0 = max(float(noise_variance), 1e-15)
    metric = -np.abs(values[:, None] - _CONSTELLATION[None, :]) ** 2 / n0
    llr = np.empty((values.size, 4), dtype=float)
    for bit in range(4):
        llr[:, bit] = logsumexp(metric[:, _CONST_BITS[:, bit] == 0], axis=1) - logsumexp(metric[:, _CONST_BITS[:, bit] == 1], axis=1)
    return llr


def _gmi_for_scale(bits: np.ndarray, llrs: np.ndarray, scale: float) -> float:
    bit_array = np.asarray(bits, dtype=np.uint8).reshape(-1, 4)
    llr_array = np.asarray(llrs, dtype=float).reshape(-1, 4)
    if bit_array.shape != llr_array.shape:
        raise ValueError("Bits and LLR arrays must match")
    sign = 1.0 - 2.0 * bit_array
    penalty = np.logaddexp(0.0, -sign * float(scale) * llr_array) / np.log(2.0)
    return float(4.0 - np.mean(np.sum(penalty, axis=1)))


def gmi_from_llrs(bits: np.ndarray, llrs: np.ndarray, *, optimize_scale: bool = True) -> float | tuple[float, float]:
    if optimize_scale:
        result = minimize_scalar(lambda x: -_gmi_for_scale(bits, llrs, x), bounds=(0.0, 8.0), method="bounded")
        value, scale = -float(result.fun), float(result.x)
        if not result.success:
            scale, value = 1.0, _gmi_for_scale(bits, llrs, 1.0)
        return float(np.clip(value, 0.0, 4.0)), scale
    return float(_gmi_for_scale(bits, llrs, 1.0))


def _wilson_interval(errors: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    probability = errors / total
    denominator = 1.0 + z**2 / total
    center = (probability + z**2 / (2.0 * total)) / denominator
    half = z * np.sqrt(probability * (1.0 - probability) / total + z**2 / (4.0 * total**2)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def _block_bootstrap_interval(tx: np.ndarray, rx: np.ndarray, statistic: Callable[[np.ndarray, np.ndarray], float], seed: int, samples: int, block_size: int) -> tuple[float, float]:
    if samples <= 0:
        return float("nan"), float("nan")
    n = min(tx.size, rx.size)
    block = min(max(int(block_size), 16), n)
    blocks = int(np.ceil(n / block))
    starts = np.arange(max(n - block + 1, 1))
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(int(samples)):
        selected = rng.choice(starts, size=blocks, replace=True)
        indices = np.concatenate([np.arange(start, min(start + block, n)) for start in selected])[:n]
        values.append(float(statistic(tx[indices], rx[indices])))
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size < max(10, samples // 4):
        return float("nan"), float("nan")
    return tuple(map(float, np.percentile(finite, [2.5, 97.5])))


def _snr_statistic(tx: np.ndarray, rx: np.ndarray) -> float:
    error = rx - tx
    return 10.0 * np.log10(max(float(np.mean(np.abs(tx) ** 2)), 1e-30) / max(float(np.mean(np.abs(error) ** 2)), 1e-30))


def estimate_complex_gain(training_tx: np.ndarray, training_rx: np.ndarray) -> complex:
    tx = np.asarray(training_tx, dtype=complex).reshape(-1)
    rx = np.asarray(training_rx, dtype=complex).reshape(-1)
    n = min(tx.size, rx.size)
    if n < 16:
        raise ValueError("At least 16 training samples are required")
    return complex(np.vdot(rx[:n], tx[:n]) / max(float(np.vdot(rx[:n], rx[:n]).real), 1e-30))


def sample_metrics_16qam(
    tx: np.ndarray,
    rx: np.ndarray,
    *,
    calibration_gain: complex | None = None,
    allow_payload_fit: bool = True,
    acquisition_success: bool = True,
    bootstrap_samples: int = 0,
    bootstrap_block_symbols: int = 256,
    bootstrap_seed: int = 0,
    minimum_error_count: int = 100,
) -> dict[str, float]:
    transmitted = np.asarray(tx, dtype=complex).reshape(-1)
    received = np.asarray(rx, dtype=complex).reshape(-1)
    n = min(transmitted.size, received.size)
    if n < 64:
        raise ValueError("At least 64 aligned payload symbols are required")
    transmitted, received = transmitted[:n], received[:n]
    finite = np.isfinite(transmitted.real) & np.isfinite(transmitted.imag) & np.isfinite(received.real) & np.isfinite(received.imag)
    transmitted, received = transmitted[finite], received[finite]
    if transmitted.size < 64:
        raise ValueError("Too few finite payload symbols remain")
    if calibration_gain is None:
        if not allow_payload_fit:
            raise ValueError("A training-derived calibration_gain is required when payload fitting is forbidden")
        calibration_gain = estimate_complex_gain(transmitted, received)
    aligned = complex(calibration_gain) * received
    error = aligned - transmitted
    signal_power = float(np.mean(np.abs(transmitted) ** 2))
    noise_variance = float(np.mean(np.abs(error) ** 2))
    snr = signal_power / max(noise_variance, 1e-30)
    evm = np.sqrt(noise_variance / max(signal_power, 1e-30))
    decisions, tx_bits, tx_indices = nearest_16qam(transmitted)
    _, rx_bits, rx_indices = nearest_16qam(aligned)
    bit_errors = int(np.count_nonzero(tx_bits != rx_bits))
    symbol_errors = int(np.count_nonzero(tx_indices != rx_indices))
    n_bits = int(tx_bits.size)
    ber, ser = bit_errors / n_bits, symbol_errors / transmitted.size
    ber_low, ber_high = _wilson_interval(bit_errors, n_bits)
    ser_low, ser_high = _wilson_interval(symbol_errors, transmitted.size)
    llrs = exact_bit_llrs_16qam(aligned, noise_variance)
    raw_gmi = float(gmi_from_llrs(tx_bits, llrs, optimize_scale=False))
    gmi, llr_scale = gmi_from_llrs(tx_bits, llrs, optimize_scale=True)
    snr_low, snr_high = _block_bootstrap_interval(transmitted, aligned, _snr_statistic, bootstrap_seed, bootstrap_samples, bootstrap_block_symbols)
    zero_error_upper = float(3.0 / n_bits) if bit_errors == 0 else np.nan
    guidance = float(bit_errors >= int(minimum_error_count))
    return {
        "acquisition_success": float(bool(acquisition_success)),
        "snr_linear": float(snr), "snr_db": float(10 * np.log10(max(snr, 1e-30))),
        "snr_ci95_low_db": snr_low, "snr_ci95_high_db": snr_high,
        "evm": float(evm), "evm_percent": float(100 * evm),
        "ber": float(ber), "ber_wilson_low": ber_low, "ber_wilson_high": ber_high,
        "ber_zero_error_upper_95": zero_error_upper, "ber_error_count_sufficient": guidance,
        "ser": float(ser), "ser_wilson_low": ser_low, "ser_wilson_high": ser_high,
        "bit_errors": float(bit_errors), "symbol_errors": float(symbol_errors),
        "n_bits": float(n_bits), "n_symbols": float(transmitted.size),
        "q_factor_db": float(q_factor_db_from_ber(max(ber, 0.5 / max(n_bits, 1)))),
        "gmi_bits_per_2d_symbol_per_pol": float(gmi), "gmi_raw_bits_per_2d_symbol_per_pol": raw_gmi,
        "gmi_llr_scale": float(llr_scale), "gmi_was_clipped": 0.0, "ngmi": float(gmi / 4.0),
        "noise_variance": noise_variance,
        "complex_gain_real": float(np.real(calibration_gain)), "complex_gain_imag": float(np.imag(calibration_gain)),
        "payload_gain_was_fitted": float(calibration_gain is not None and allow_payload_fit),
    }


def paired_metric_comparison(analytical: np.ndarray, measured: np.ndarray) -> dict[str, float]:
    x, y = np.asarray(analytical, dtype=float), np.asarray(measured, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() == 0:
        return {"samples": 0.0, "rmse": np.nan, "bias": np.nan, "max_abs_error": np.nan}
    residual = y[valid] - x[valid]
    return {"samples": float(valid.sum()), "rmse": float(np.sqrt(np.mean(residual**2))), "bias": float(np.mean(residual)), "max_abs_error": float(np.max(np.abs(residual)))}


@lru_cache(maxsize=512)
def _gmi_quadrature_scalar(snr_db_rounded: float, order: int = 8) -> float:
    snr = 10.0 ** (snr_db_rounded / 10.0)
    n0 = 1.0 / snr
    sigma = np.sqrt(n0 / 2.0)
    nodes, weights = hermgauss(order)
    weights /= np.sqrt(np.pi)
    real_nodes, imag_nodes = np.meshgrid(nodes, nodes, indexing="ij")
    real_weights, imag_weights = np.meshgrid(weights, weights, indexing="ij")
    noise = np.sqrt(2.0) * sigma * (real_nodes.reshape(-1) + 1j * imag_nodes.reshape(-1))
    noise_weights = (real_weights * imag_weights).reshape(-1)
    received = (_CONSTELLATION[:, None] + noise[None, :]).reshape(-1)
    tx_bits = np.repeat(_CONST_BITS, noise.size, axis=0)
    sample_weights = np.tile(noise_weights, 16) / 16.0
    metric = -np.abs(received[:, None] - _CONSTELLATION[None, :]) ** 2 / n0
    llr = np.empty((received.size, 4))
    for bit in range(4):
        llr[:, bit] = logsumexp(metric[:, _CONST_BITS[:, bit] == 0], axis=1) - logsumexp(metric[:, _CONST_BITS[:, bit] == 1], axis=1)
    sign = 1.0 - 2.0 * tx_bits
    penalty = np.sum(np.logaddexp(0.0, -sign * llr) / np.log(2.0), axis=1)
    return float(np.clip(4.0 - np.sum(sample_weights * penalty), 0.0, 4.0))


@lru_cache(maxsize=8)
def _gmi_lookup(order: int = 8) -> tuple[np.ndarray, np.ndarray]:
    grid = np.arange(-8.0, 35.0001, 0.25)
    values = np.array([_gmi_quadrature_scalar(round(float(value), 4), order) for value in grid])
    return grid, np.maximum.accumulate(values)


def gmi_16qam_awgn_from_snr_db(snr_db: np.ndarray | float, order: int = 8) -> np.ndarray:
    values = np.asarray(snr_db, dtype=float)
    grid, lookup = _gmi_lookup(int(order))
    return np.interp(values, grid, lookup, left=lookup[0], right=lookup[-1])
