import numpy as np

from isrs_scl.dsp.metrics import (
    bits_to_16qam,
    exact_bit_llrs_16qam,
    optimized_gmi_from_llrs,
    sample_metrics_16qam,
)


def test_optimized_gmi_uses_nonnegative_mismatched_decoder_scale():
    rng = np.random.default_rng(7)
    bits = rng.integers(0, 2, size=(4096, 4), dtype=np.uint8)
    tx = bits_to_16qam(bits)
    rx = rng.normal(size=tx.size) + 1j * rng.normal(size=tx.size)
    llrs = exact_bit_llrs_16qam(rx, noise_variance=2.0)
    raw, optimized, scale = optimized_gmi_from_llrs(bits, llrs)
    assert np.isfinite(raw)
    assert 0.0 <= optimized <= 4.0
    assert scale >= 0.0


def test_sample_metrics_reports_raw_and_optimized_gmi_without_clip_flag():
    rng = np.random.default_rng(11)
    bits = rng.integers(0, 2, size=(8192, 4), dtype=np.uint8)
    tx = bits_to_16qam(bits)
    noise = 0.12 * (rng.normal(size=tx.size) + 1j * rng.normal(size=tx.size))
    metrics = sample_metrics_16qam(tx, tx + noise)
    assert "gmi_raw_bits_per_2d_symbol_per_pol" in metrics
    assert "gmi_scale" in metrics
    assert metrics["gmi_was_clipped"] == 0.0
    assert 0.0 <= metrics["ngmi"] <= 1.0