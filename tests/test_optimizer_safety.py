from types import SimpleNamespace

import numpy as np

from isrs_scl.optimization.adaptive_isrs import AdaptiveLaunchOptimizer


class _FakeLink:
    def __init__(self, channels: int = 9):
        self.grid = SimpleNamespace(
            n_channels=channels,
            frequencies_hz=193e12 + np.arange(channels) * 50e9,
        )

    def evaluate(self, launch_power_w: np.ndarray, n_spans: int):
        del n_spans
        profile_dbm = 10.0 * np.log10(np.maximum(launch_power_w, 1e-30) / 1e-3)
        penalty = float(np.mean((profile_dbm - np.mean(profile_dbm)) ** 2))
        quality = max(1.0 - 0.1 * penalty, 0.0)
        return SimpleNamespace(
            ngmi=np.full(profile_dbm.size, 0.95 * quality),
            soft_fec_net_tbps=quality,
            fec_net_tbps=1.0 if quality >= 0.95 else 0.0,
            air_tbps=1.1 * quality,
        )


def _config() -> dict:
    return {
        "fec": {"ngmi_target": 0.90},
        "launch": {
            "min_power_dbm_per_channel": -5.0,
            "max_power_dbm_per_channel": 3.0,
        },
        "optimization": {
            "target_spans": 2,
            "seed": 12,
            "control_points": 5,
            "iterations": 6,
            "learning_rate": 0.1,
            "spsa_perturbation_db": 0.2,
            "smoothness_weight": 0.0,
            "soft_fec_weight": 1.0,
            "air_weight": 0.2,
            "worst_ngmi_weight": 0.15,
            "ngmi_std_weight": 0.05,
            "softmin_temperature_ngmi": 0.01,
            "gradient_clip": 10.0,
            "early_stopping_patience": 6,
            "minimum_claim_improvement_tbps": 0.01,
            "minimum_relative_claim_improvement": 0.001,
        },
    }


def test_optimizer_never_returns_a_profile_worse_than_baseline():
    optimizer = AdaptiveLaunchOptimizer(_FakeLink(), _config())
    baseline = np.zeros(optimizer.link.grid.n_channels)
    result = optimizer.optimize(baseline, baseline_name="Flat")

    assert result.optimized_soft_fec_tbps >= result.initial_soft_fec_tbps
    assert result.optimized_fec_net_tbps >= result.initial_fec_net_tbps
    assert result.optimized_air_tbps >= result.initial_air_tbps
    assert np.allclose(result.optimized_profile_dbm, baseline)
    assert not result.improved