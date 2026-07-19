from __future__ import annotations

import numpy as np

from isrs_scl.dsp.equalizer import pilot_aided_equalize_2x2
from isrs_scl.dsp.metrics import bits_to_16qam, sample_metrics_16qam
from isrs_scl.dsp.receiver import coherent_receiver, propagate_representative_channel
from isrs_scl.dsp.transmitter import generate_dp16qam


def test_pilot_equalizer_recovers_static_polarization_mixing() -> None:
    rng = np.random.default_rng(20260718)
    bits = rng.integers(0, 2, size=(2, 8000, 4), dtype=np.uint8)
    reference = np.vstack([bits_to_16qam(bits[pol]) for pol in range(2)])
    mixing = np.array(
        [[0.8 + 0.1j, 0.35 - 0.2j], [-0.25 - 0.3j, 0.85 - 0.05j]],
        dtype=complex,
    )
    received = mixing @ reference
    received += 0.01 * (
        rng.standard_normal(received.shape) + 1j * rng.standard_normal(received.shape)
    )

    equalized = pilot_aided_equalize_2x2(
        received,
        reference,
        taps=5,
        training_symbols=4000,
        ridge=1e-6,
    )
    center = 2
    target = reference[:, center : center + equalized.symbols.shape[1]]
    mse = np.mean(np.abs(equalized.symbols - target) ** 2)
    assert mse < 2e-3


def test_reported_gmi_and_ngmi_are_bounded() -> None:
    rng = np.random.default_rng(7)
    bits = rng.integers(0, 2, size=(3000, 4), dtype=np.uint8)
    transmitted = bits_to_16qam(bits)
    received = 0.05 * (
        rng.standard_normal(transmitted.shape)
        + 1j * rng.standard_normal(transmitted.shape)
    )
    metrics = sample_metrics_16qam(transmitted, received)
    assert 0.0 <= metrics["gmi_bits_per_2d_symbol_per_pol"] <= 4.0
    assert 0.0 <= metrics["ngmi"] <= 1.0
    assert metrics["ber_wilson_low"] <= metrics["ber"] <= metrics["ber_wilson_high"]


def test_back_to_back_waveform_snr_matches_injected_snr() -> None:
    channel_power_w = 1e-3
    injected_snr_db = 15.0
    tx = generate_dp16qam(8192, 4, 0.10, 12, 11, channel_power_w)
    noise_w = channel_power_w / 10.0 ** (injected_snr_db / 10.0)
    channel = propagate_representative_channel(
        tx,
        32e9,
        1550.0,
        0.0,
        0.0,
        1,
        noise_w,
        0.0,
        0.0,
        0.0,
        False,
        False,
        12,
    )
    receiver = coherent_receiver(
        channel,
        tx,
        0.0,
        11,
        2e-4,
        4000,
        64,
        64,
        equalizer_mode="pilot_aided",
    )
    measured = np.mean([metric["snr_db"] for metric in receiver.metrics_per_pol])
    assert abs(measured - injected_snr_db) < 0.75