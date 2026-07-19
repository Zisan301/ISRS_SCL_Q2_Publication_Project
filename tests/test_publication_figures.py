from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from isrs_scl.visualization.publication_plots import (  # noqa: E402
    plot_gsnr_distance_heatmap,
    plot_launch_profiles,
    plot_waveform_metrics_comparison,
    plot_waveform_power_consistency,
)


def _assert_pair(base: Path) -> None:
    assert base.with_suffix(".pdf").exists()
    assert base.with_suffix(".png").exists()
    assert base.with_suffix(".pdf").stat().st_size > 0
    assert base.with_suffix(".png").stat().st_size > 0


def test_missing_numbered_figures_are_emitted(tmp_path: Path) -> None:
    wavelengths = np.array([1490.0, 1535.0, 1570.0, 1610.0])
    channel_rows = []
    for spans in range(1, 4):
        for wavelength in wavelengths:
            channel_rows.append(
                {
                    "strategy": "Adaptive",
                    "spans": spans,
                    "distance_km": 80.0 * spans,
                    "wavelength_nm": wavelength,
                    "gsnr_db": 24.0 - 1.8 * spans - 0.003 * abs(wavelength - 1550.0),
                }
            )
    channel_sweep = pd.DataFrame(channel_rows)
    heatmap_base = tmp_path / "03_gsnr_distance_heatmap"
    plot_gsnr_distance_heatmap(
        channel_sweep,
        "Adaptive",
        heatmap_base,
        150,
        threshold_db=17.0,
    )
    _assert_pair(heatmap_base)

    waveform = pd.DataFrame(
        {
            "band": ["S", "S", "C", "C", "L", "L"],
            "evm_percent": [18.0, 18.5, 15.0, 15.2, 16.0, 16.4],
            "ber": [2e-3, 2.3e-3, 7e-4, 8e-4, 1.1e-3, 1.3e-3],
            "ngmi": [0.91, 0.905, 0.94, 0.938, 0.925, 0.92],
            "power_domain_gsnr_db": [15.0, 15.0, 17.0, 17.0, 16.0, 16.0],
            "snr_db": [14.4, 14.6, 16.4, 16.6, 15.5, 15.3],
        }
    )
    metrics_base = tmp_path / "07_evm_ber_ngmi_comparison"
    plot_waveform_metrics_comparison(
        waveform,
        metrics_base,
        150,
        ber_target=0.004,
        ngmi_target=0.90,
    )
    _assert_pair(metrics_base)

    consistency_base = tmp_path / "13_power_waveform_consistency"
    plot_waveform_power_consistency(
        waveform,
        consistency_base,
        150,
        tolerance_db=2.0,
    )
    _assert_pair(consistency_base)

    launch = pd.DataFrame(
        {
            "wavelength_nm": wavelengths,
            "flat_dbm": np.zeros(wavelengths.size),
            "fixed_dbm": np.linspace(-1.0, 1.0, wavelengths.size),
            "adaptive_dbm": np.linspace(-0.8, 0.8, wavelengths.size),
        }
    )
    launch_base = tmp_path / "12_launch_profiles"
    plot_launch_profiles(launch, launch_base, 150)
    _assert_pair(launch_base)