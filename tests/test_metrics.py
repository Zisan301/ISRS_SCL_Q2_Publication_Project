import numpy as np

from isrs_scl.dsp.metrics import (
    analytical_ber_16qam,
    bits_to_16qam,
    gmi_16qam_awgn_from_snr_db,
    nearest_16qam,
    sample_metrics_16qam,
)


def test_gray_mapping_round_trip():
    rng = np.random.default_rng(7)
    bits = rng.integers(0, 2, size=(1000, 4), dtype=np.uint8)
    symbols = bits_to_16qam(bits)
    _, recovered_bits, _ = nearest_16qam(symbols)
    assert np.array_equal(bits, recovered_bits)


def test_ber_and_gmi_are_monotonic():
    snr_db = np.array([8.0, 12.0, 16.0, 20.0])
    ber = analytical_ber_16qam(10 ** (snr_db / 10))
    gmi = gmi_16qam_awgn_from_snr_db(snr_db, order=8)
    assert np.all(np.diff(ber) < 0)
    assert np.all(np.diff(gmi) > 0)
    assert np.all((gmi >= 0) & (gmi <= 4))


def test_sample_metrics_noiseless():
    rng = np.random.default_rng(8)
    bits = rng.integers(0, 2, size=(2000, 4), dtype=np.uint8)
    tx = bits_to_16qam(bits)
    metrics = sample_metrics_16qam(tx, tx * np.exp(1j * 0.37) * 1.4)
    assert metrics["ber"] == 0.0
    assert metrics["evm"] < 1e-12
    assert metrics["ngmi"] > 0.999
