import numpy as np

from isrs_scl.dsp.metrics import analytical_ber_16qam, gmi_16qam_awgn_from_snr_db
from isrs_scl.dsp.noise import (
    decision_variance_to_sample_variance, monte_carlo_decision_variance,
)
from isrs_scl.dsp.transmitter import root_raised_cosine_taps


def test_analytical_metrics_are_monotone_over_calibration_range():
    snr_db = np.linspace(-4, 30, 100)
    snr = 10 ** (snr_db / 10)
    ber = analytical_ber_16qam(snr)
    gmi = gmi_16qam_awgn_from_snr_db(snr_db)
    assert np.all(np.diff(ber) <= 0)
    assert np.all(np.diff(gmi) >= -1e-12)


def test_noise_conversion_matches_monte_carlo_decision_variance():
    taps = root_raised_cosine_taps(0.1, 8, 4)
    requested = 2e-5
    sample = decision_variance_to_sample_variance(requested, taps)
    measured = monte_carlo_decision_variance(taps, sample_variance_per_pol=sample, samples_per_symbol=4, symbols=30000, seed=9)
    assert np.isclose(measured, requested, rtol=0.08)


def test_transmitter_power_convention_across_rrc_settings():
    from isrs_scl.dsp.transmitter import generate_dp16qam
    for sps in (2, 4, 8):
        for roll_off in (0.0, 0.1, 0.25):
            tx = generate_dp16qam(2048, sps, roll_off, 20, seed=100 + sps, channel_power_w=2e-3, pilot_symbols=256, pilot_spacing=32, symbol_rate_hz=32e9)
            assert np.isclose(tx.symbol_decision_power_w, 2e-3, rtol=1e-12)
            assert tx.payload_mask.sum() > 0 and tx.pilot_mask.sum() > 0
            assert np.isclose(tx.pulse_energy, 1.0, rtol=1e-12)
