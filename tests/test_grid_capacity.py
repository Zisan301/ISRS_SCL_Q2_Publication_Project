import numpy as np

from isrs_scl.system.capacity import line_rate_capacity
from isrs_scl.system.grid import build_grid
from isrs_scl.system.parameters import load_config


def test_full_grid_has_417_exactly_spaced_channels():
    cfg = load_config("config.yaml")
    cfg["grid"]["mode"] = "full_scl"
    grid = build_grid(cfg["grid"])
    assert grid.n_channels == 417
    assert np.allclose(np.diff(grid.frequencies_hz), 50e9, rtol=0, atol=1.0)
    assert set(grid.bands) == {"S", "C", "L"}


def test_paper_subset_has_240_channels():
    cfg = load_config("config.yaml")
    cfg["grid"]["mode"] = "paper_240_subset"
    grid = build_grid(cfg["grid"])
    assert grid.n_channels == 240
    assert np.allclose(np.diff(grid.frequencies_hz), 50e9, rtol=0, atol=1.0)


def test_capacity_labels_are_correct():
    result = line_rate_capacity(240, 32e9, 4, 0.25)
    assert result.gross_tbps == 61.44
    assert result.net_tbps == 49.152
