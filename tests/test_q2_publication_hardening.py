from __future__ import annotations

import numpy as np

from isrs_scl.dsp.metrics import sample_metrics_16qam
from isrs_scl.dsp.receiver import apply_laser_phase_noise
from isrs_scl.dsp.transmitter import generate_dp16qam
from isrs_scl.system.capacity import achievable_information_rate_bps, line_rate_capacity, summarize_capacity
from isrs_scl.validation.reproducibility import stable_hash


def test_air_is_not_divided_by_fec_overhead() -> None:
    gmi = np.array([4.0, 3.0])
    air = achievable_information_rate_bps(gmi, 32e9, polarizations=2, maximum_bits_per_symbol_per_pol=4)
    assert np.isclose(air, 448e9)
    metrics = summarize_capacity(gmi, gmi / 4.0, 0.90, 32e9, 4, 0.25)
    assert np.isclose(metrics.air_bps, air)
    assert metrics.working_channels == 1


def test_line_rate_overhead_convention() -> None:
    result = line_rate_capacity(1, 32e9, 4, 0.25)
    assert np.isclose(result.gross_bps, 256e9)
    assert np.isclose(result.net_bps, 204.8e9)


def test_gmi_is_finite_nonnegative_and_unclipped() -> None:
    tx = generate_dp16qam(4096, 2, 0.1, 8, 7).symbols[0]
    rng = np.random.default_rng(8)
    rx = tx + 0.2 * (rng.standard_normal(tx.size) + 1j * rng.standard_normal(tx.size))
    metrics = sample_metrics_16qam(tx, rx)
    assert np.isfinite(metrics["gmi_bits_per_2d_symbol_per_pol"])
    assert 0.0 <= metrics["gmi_bits_per_2d_symbol_per_pol"] <= 4.0
    assert metrics["gmi_was_clipped"] == 0.0


def test_phase_noise_is_reproducible_for_one_end_to_end_process() -> None:
    waveform = np.ones((2, 2048), dtype=complex)
    first = apply_laser_phase_noise(waveform, 100e3, 128e9, np.random.default_rng(9))
    second = apply_laser_phase_noise(waveform, 100e3, 128e9, np.random.default_rng(9))
    assert np.allclose(first, second)
    assert not np.allclose(first, waveform)


def test_configuration_hash_is_order_independent() -> None:
    assert stable_hash({"b": 2, "a": 1}) == stable_hash({"a": 1, "b": 2})