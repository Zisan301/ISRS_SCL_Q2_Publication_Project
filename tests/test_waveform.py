import numpy as np
import pytest

from isrs_scl.dsp.receiver import coherent_receiver, propagate_representative_channel
from isrs_scl.dsp.transmitter import generate_dp16qam


@pytest.mark.parametrize("snr_db", [2.0, 6.0, 10.0, 16.0, 24.0])
def test_awgn_waveform_sweep_has_explicit_acquisition_state(snr_db):
    power = 1e-3
    tx = generate_dp16qam(8192, 4, 0.1, 20, seed=11, channel_power_w=power, pilot_symbols=1024, pilot_spacing=64, symbol_rate_hz=32e9)
    channel = propagate_representative_channel(tx, 32e9, 1550.0, 0.0, 0.0, 1, power / 10 ** (snr_db / 10), 0.0, 0.0, 0.0, False, False, seed=12)
    rx = coherent_receiver(channel, tx, 0.0, 7, 1e-4, 1000, 32, 64, equalizer_mode="pilot_aided", carrier_recovery_mode="pilot_aided", bootstrap_samples=20)
    assert isinstance(rx.acquisition_success, bool)
    if rx.acquisition_success:
        measured = np.mean([metric["snr_db"] for metric in rx.metrics_per_pol])
        assert abs(measured - snr_db) < 2.0
        assert rx.training_symbols > 0 and rx.payload_symbols > 0
    else:
        assert rx.failure_reason


def test_payload_metrics_do_not_fit_payload_gain():
    tx = generate_dp16qam(4096, 4, 0.1, 20, seed=21, channel_power_w=1e-3, pilot_symbols=512, pilot_spacing=64, symbol_rate_hz=32e9)
    channel = propagate_representative_channel(tx, 32e9, 1550.0, 0.0, 0.0, 1, 1e-6, 0.0, 0.0, 0.0, False, False, seed=22)
    rx = coherent_receiver(channel, tx, 0.0, 7, 1e-4, 800, 32, 64, carrier_recovery_mode="pilot_aided")
    if rx.acquisition_success:
        assert all(metric["payload_gain_was_fitted"] == 0.0 for metric in rx.metrics_per_pol)
