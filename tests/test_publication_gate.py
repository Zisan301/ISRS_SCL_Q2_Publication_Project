from pathlib import Path

import pandas as pd

from isrs_scl.validation.publication_gate import evaluate_publication_gate


def cfg():
    return {
        "metadata": {"calibration_status": "CALIBRATED", "calibration_sources": [{"source_id": "x"}]},
        "nli": {"primary_model": "power_profile_gn"},
        "optimization": {"target_spans": 8, "minimum_band_working_fraction": {"S": 0.8, "C": 0.8, "L": 0.8}},
        "fec": {"ngmi_target": 0.9},
        "waveform": {"consistency_tolerance_db": 1.5},
        "validation": {
            "nominal_closed_form_bandwidth_limit_thz": 15.0, "minimum_waveform_repeats": 1,
            "require_external_validation": True, "require_uncertainty_analysis": True,
            "minimum_uncertainty_success_fraction": 0.95, "minimum_uncertainty_holdout_samples": 4,
            "minimum_probability_of_improvement": 0.75, "minimum_robust_capacity_gain_tbps": 0.0,
            "minimum_optimizer_seeds": 2,
        },
    }


def evidence(tmp_path: Path):
    sweep = pd.DataFrame([
        {"strategy": "adaptive", "spans": 8, "band": band, "ngmi": 0.95}
        for band in "SCL" for _ in range(5)
    ])
    waveform = pd.DataFrame([
        {"band": band, "operating_point": point, "acquisition_success": True, "training_symbols": 128,
         "payload_symbols": 512, "sample_snr_db": 14.0, "snr_consistency_error_db": 0.2,
         "snr_ci95_low_db": 13.8, "snr_ci95_high_db": 14.2, "sample_gmi": 3.8}
        for band in "SCL" for point in ("high_margin", "near_threshold")
    ])
    external = pd.DataFrame([{"requirement": "all", "passed": True}])
    uncertainty = pd.DataFrame([{"sample": sample, "strategy": strategy, "success": 1} for sample in range(4) for strategy in ("flat", "fixed", "adaptive")])
    paired = pd.DataFrame([{"baseline": "fixed", "ci95_low": 0.1, "probability_positive": 1.0}])
    optimizer = pd.DataFrame([{"seed": 1}, {"seed": 2}])
    output = tmp_path / "output.csv"; output.write_text("x\n1\n", encoding="utf-8")
    return sweep, waveform, external, uncertainty, paired, optimizer, output


def test_each_major_gate_can_pass_with_valid_synthetic_evidence(tmp_path):
    sweep, waveform, external, uncertainty, paired, optimizer, output = evidence(tmp_path)
    result = evaluate_publication_gate(
        cfg(), grid_bandwidth_thz=20.8, convergence_passed=True, raman_validation_passed=True,
        strategy_summary=pd.DataFrame(), channel_sweep=sweep, waveform_metrics=waveform,
        external_validation_summary=external, uncertainty_summary=pd.DataFrame([{"x": 1}]),
        uncertainty_samples=uncertainty, uncertainty_success_fraction=1.0, optimizer_accepted=True,
        output_files=[output], optimizer_multiseed=optimizer, paired_gains=paired,
        robust_training_hash="training", holdout_hash="holdout",
    )
    assert result.passed


def test_s_band_sacrifice_fails(tmp_path):
    sweep, waveform, external, uncertainty, paired, optimizer, output = evidence(tmp_path)
    sweep.loc[sweep["band"] == "S", "ngmi"] = 0.1
    result = evaluate_publication_gate(cfg(), grid_bandwidth_thz=20.8, convergence_passed=True, raman_validation_passed=True, strategy_summary=pd.DataFrame(), channel_sweep=sweep, waveform_metrics=waveform, external_validation_summary=external, uncertainty_summary=pd.DataFrame([{"x": 1}]), uncertainty_samples=uncertainty, uncertainty_success_fraction=1.0, optimizer_accepted=True, output_files=[output], optimizer_multiseed=optimizer, paired_gains=paired, robust_training_hash="training", holdout_hash="holdout")
    assert "per_band_target_usability" in [item.name for item in result.failures]
