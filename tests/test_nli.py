import numpy as np

from isrs_scl.fiber.nonlinear_gn import PowerProfileGN


def test_cross_channel_nli_is_used_and_positive():
    n = 5
    f = 193e12 + np.arange(n) * 50e9
    alpha = np.full(n, 4.6e-5)
    beta2 = np.full(n, -21.7e-27)
    gamma = np.full(n, 1.3e-3)
    model = PowerProfileGN(f, 32e9, alpha, beta2, gamma)
    z = np.linspace(0, 80e3, 81)
    launch = np.full(n, 1e-3)
    profiles = launch[None, :] * np.exp(-z[:, None] * alpha[None, :])
    result = model.evaluate(launch, z, profiles)
    assert np.all(result.nli_power_w_per_span > 0)
    assert np.all(result.eta_xci_per_w2 > 0)
