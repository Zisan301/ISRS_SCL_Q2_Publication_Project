from pathlib import Path

import pytest
import yaml

from isrs_scl.system.parameters import ConfigError, apply_defaults, validate_config


def base_config(tmp_path: Path):
    external = tmp_path / "external.csv"; external.write_text("x\n", encoding="utf-8")
    return apply_defaults({
        "metadata": {"study_title": "test", "random_seed": 1, "calibration_status": "UNVALIDATED_DEFAULTS", "calibration_sources": []},
        "grid": {"mode": "paper_240_subset", "lambda_min_nm": 1460, "lambda_max_nm": 1625, "spacing_ghz": 50, "subset_channels": 12, "subset_center_nm": 1542.5},
        "modulation": {"symbol_rate_gbaud": 32, "roll_off": 0.1, "format": "DP-16QAM", "bits_per_symbol_per_pol": 4, "oversampling": 4, "rrc_span_symbols": 8, "waveform_symbols": 4096},
        "fec": {"overhead_fraction": 0.25, "pre_fec_ber_target": 0.004, "ngmi_target": 0.9, "threshold_metric": "ngmi", "b2b_snr_sweep_db": [-4, 30, 1]},
        "fiber": {"span_length_km": 80, "max_spans": 2, "effective_area_um2": 80, "gamma_per_w_km_at_1550": 1.3, "dispersion_ps_nm_km_at_1550": 17, "dispersion_slope_ps_nm2_km": 0.058, "pmd_ps_sqrt_km": 0.05, "attenuation_anchors": {"wavelength_nm": [1460, 1625], "db_per_km": [0.22, 0.23]}},
        "raman": {"integration_step_m": 100, "save_step_m": 100, "gain_peak_m_per_w": 8e-14, "gain_csv": None, "semrau_linear_slope_per_w_km_thz": 0.028, "pumps": [], "equivalent_noise_figure_db": -0.5},
        "amplification": {"noise_bandwidth_multiplier": 1.1, "bands": {"all": {"enabled": True, "wavelength_nm": [1460, 1625], "noise_figure_db": [5, 6], "max_gain_db": 40}}, "gain_flatness_tolerance_db": 0.05},
        "nli": {"primary_model": "power_profile_gn", "modulation_correction": 0.86, "coherence_epsilon": 0.08, "transceiver_snr_db": 35, "semrau_valid_bandwidth_thz": 15},
        "launch": {"flat_power_dbm_per_channel": 0, "min_power_dbm_per_channel": -5, "max_power_dbm_per_channel": 3, "fixed_preemphasis_s_to_l_db": 6},
        "optimization": {"target_spans": 1, "evaluation_spans": [1], "method": "spsa_adam", "iterations": 1, "control_points": 5, "learning_rate": 0.1, "spsa_perturbation_db": 0.1, "seed": 4},
        "waveform": {"selected_wavelengths_nm": [1495, 1550, 1595], "cma_taps": 7, "cma_step_size": 1e-4, "cma_training_symbols": 512, "bps_trial_phases": 32, "bps_block_symbols": 64, "laser_linewidth_hz": 0, "carrier_frequency_offset_hz": 0, "apply_pmd": False, "apply_phase_noise": False, "use_gn_nli_noise": True},
        "output": {"directory": "results", "figure_directory": "figures", "png_dpi": 180, "save_profiles": True},
        "validation": {"external_reference_csv": str(external)},
    })


def test_unknown_keys_are_rejected(tmp_path):
    cfg = base_config(tmp_path); cfg["grid"]["typo"] = 1
    with pytest.raises(ConfigError, match="Unknown configuration key"):
        validate_config(cfg, base_dir=tmp_path)


def test_training_and_holdout_seeds_must_differ(tmp_path):
    cfg = base_config(tmp_path); cfg["optimization"]["robust_training_seed"] = cfg["uncertainty"]["holdout_seed"]
    with pytest.raises(ConfigError, match="must differ"):
        validate_config(cfg, base_dir=tmp_path)
