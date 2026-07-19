import numpy as np
import pytest

from isrs_scl.system.capacity import (
    achievable_information_rate_bps,
    thresholded_net_capacity_bps,
    throughput_from_gmi,
)


def test_air_uses_per_channel_gmi_without_binary_threshold():
    gmi = np.array([4.0, 3.6, 2.0])
    rate = achievable_information_rate_bps(gmi, 32e9, polarizations=2, maximum_bits_per_symbol_per_pol=4)
    assert rate == pytest.approx(np.sum(gmi) * 32e9 * 2)


def test_hard_fec_and_soft_utility_are_reported_separately():
    gmi = np.array([4.0, 3.6, 3.56, 0.0])  # NGMI: 1.0, 0.9, 0.89, 0.0
    result = throughput_from_gmi(
        gmi,
        symbol_rate_baud=32e9,
        bits_per_symbol_per_pol=4,
        fec_overhead_fraction=0.25,
        ngmi_threshold=0.90,
        soft_transition=0.01,
    )
    nominal_channel_rate = 4 * 32e9 * 2 / 1.25
    assert result.working_channels == 2
    assert result.fec_net_bps == pytest.approx(2 * nominal_channel_rate)
    assert 0.0 < result.soft_fec_net_bps < 4 * nominal_channel_rate
    assert result.air_bps == pytest.approx(np.sum(gmi) * 32e9 * 2)


def test_legacy_thresholded_capacity_remains_backward_compatible():
    ngmi = np.array([0.91, 0.90, 0.89])
    capacity = thresholded_net_capacity_bps(ngmi, 0.90, 32e9, 4, 0.25)
    assert capacity == pytest.approx(2 * 4 * 32e9 * 2 / 1.25)