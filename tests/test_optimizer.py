from types import SimpleNamespace

import numpy as np

from isrs_scl.optimization.adaptive_isrs import AdaptiveLaunchOptimizer
from isrs_scl.optimization.constraints import total_power_w_from_dbm


class FakeLink:
    def __init__(self):
        self.grid = SimpleNamespace(
            n_channels=12,
            frequencies_hz=np.linspace(185e12, 205e12, 12),
            bands=np.array(["L"] * 4 + ["C"] * 4 + ["S"] * 4),
        )
    def evaluate(self, launch_w, spans):
        dbm = 10 * np.log10(np.asarray(launch_w) / 1e-3)
        band_bonus = np.r_[np.zeros(8), dbm[8:] * 0.03]
        ngmi = np.clip(0.86 + band_bonus - 0.005 * (spans - 1), 0, 1)
        gmi = 4 * ngmi
        return SimpleNamespace(ngmi=ngmi, gmi=gmi, gsnr_db=8 + 12 * ngmi, distance_km=80 * spans)


def config():
    return {
        "optimization": {"target_spans": 1, "evaluation_spans": [1], "restarts": 1, "iterations": 2, "control_points": 5, "learning_rate": 0.05, "spsa_perturbation_db": 0.1, "seed": 3, "robust_training_samples": 0, "minimum_band_working_fraction": {"S": 0.75, "C": 0.75, "L": 0.75}, "minimum_nominal_gain_tbps": 0.0, "minimum_robust_gain_ci_low_tbps": 0.0},
        "launch": {"min_power_dbm_per_channel": -5.0, "max_power_dbm_per_channel": 3.0},
        "fec": {"ngmi_target": 0.85, "overhead_fraction": 0.25},
        "modulation": {"symbol_rate_gbaud": 32.0, "bits_per_symbol_per_pol": 4},
        "uncertainty": {"distributions": {}},
    }


def test_optimizer_preserves_total_power_and_band_constraints():
    initial = np.zeros(12); optimizer = AdaptiveLaunchOptimizer(FakeLink(), config()); result = optimizer.optimize(initial)
    assert np.isclose(total_power_w_from_dbm(result.optimized_profile_dbm), total_power_w_from_dbm(initial), rtol=1e-10)
    assert np.all(result.optimized_profile_dbm >= -5) and np.all(result.optimized_profile_dbm <= 3)
    if result.improved:
        assert all(result.optimized_metrics[f"target_working_fraction_{band}"] >= 0.75 for band in "SCL")
