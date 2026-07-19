import numpy as np
import pytest

from isrs_scl.optimization.robust import (
    cvar_lower, ensure_independent_batches, generate_training_batch, paired_gain_summary,
)


def distributions():
    return {
        "attenuation_scale": {"distribution": "normal", "mean": 1.0, "std": 0.02, "minimum": 0.9, "maximum": 1.1},
        "noise_figure_offset_db": {"distribution": "uniform", "minimum": -0.3, "maximum": 0.3},
    }


def test_scenario_hashes_are_deterministic_and_separated():
    first = generate_training_batch(distributions(), samples=16, seed=1, role="robust_training")
    second = generate_training_batch(distributions(), samples=16, seed=1, role="robust_training")
    holdout = generate_training_batch(distributions(), samples=16, seed=2, role="publication_holdout")
    assert first.batch_hash == second.batch_hash
    ensure_independent_batches(first, holdout)
    with pytest.raises(ValueError): ensure_independent_batches(first, second)


def test_cvar_and_paired_gain_reject_negative_lower_bound():
    assert cvar_lower([1, 2, 3, 4], 0.25) == 1
    summary = paired_gain_summary([2, 3, 4, 5], [1, 2, 3, 4], bootstrap_samples=200, seed=3)
    assert summary.ci95_low > 0 and summary.probability_positive == 1
    negative = paired_gain_summary([0, 1, 2, 3], [1, 2, 3, 4], bootstrap_samples=200, seed=3)
    assert negative.ci95_high < 0
