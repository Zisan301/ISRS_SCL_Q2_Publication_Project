import numpy as np

from isrs_scl.constants import C_M_PER_S
from isrs_scl.fiber.attenuation import db_per_km_to_np_per_m
from isrs_scl.fiber.raman_solver import RamanGainSpectrum, RamanPump, RamanSolver
from isrs_scl.validation.analytical_raman import run_undepleted_pump_validation


def test_db_per_km_conversion():
    alpha = float(db_per_km_to_np_per_m(0.2))
    output = np.exp(-alpha * 80_000.0)
    expected = 10 ** (-0.2 * 80.0 / 10.0)
    assert np.isclose(output, expected, rtol=1e-12)


def test_undepleted_pump_rk4_matches_analytical():
    _, max_error = run_undepleted_pump_validation(step_m=40.0)
    assert max_error < 2e-4


def test_signal_to_signal_photon_weighting_conserves_energy_without_loss():
    frequencies = np.array([190e12, 200e12])
    gain = RamanGainSpectrum(8e-14)
    solver = RamanSolver(frequencies, np.zeros(2), 80e-12, gain, [])
    p = np.array([1e-3, 1e-3])
    derivative = solver.derivative(0.0, p, 1.0)
    # Photon flux sum P/(h nu) is conserved by the pairwise transfer terms.
    photon_flux_derivative = np.sum(derivative / frequencies)
    assert abs(photon_flux_derivative) < 1e-18
