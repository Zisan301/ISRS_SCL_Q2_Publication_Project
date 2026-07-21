from pathlib import Path
import json
import pandas as pd

base = Path("runs/q3-heldout-model-001/results")

status_path = base / "heldout_external_validation_status.json"
summary_path = base / "heldout_external_validation_summary.csv"
requirements_path = base / "heldout_external_validation_requirements.csv"
cal_cmp_path = base / "heldout_external_calibration_comparisons.csv"
hold_cmp_path = base / "heldout_external_holdout_comparisons.csv"

status = json.loads(status_path.read_text(encoding="utf-8"))
summary = pd.read_csv(summary_path)
requirements = pd.read_csv(requirements_path)
cal = pd.read_csv(cal_cmp_path)
hold = pd.read_csv(hold_cmp_path)

print("\n================ STATUS JSON ================")
print(json.dumps(status, indent=2))

print("\n================ SUMMARY TABLE ================")
print(summary.to_string(index=False))

print("\n================ REQUIREMENTS TABLE ================")
print(requirements.to_string(index=False))

print("\n================ HOLDOUT COMPARISON KEY ROWS ================")
cols = [
    "spans",
    "wavelength_nm",
    "reference_value",
    "model_value",
    "linear_corrected_model_value",
    "residual",
    "linear_corrected_residual",
    "matched",
    "interpolated",
]
available = [c for c in cols if c in hold.columns]
print(hold[available].to_string(index=False))

print("\n================ PAPER NUMBERS ================")
holdout = summary[
    (summary["split"] == "holdout")
    & (summary["correction"] == "wavelength_linear_trained_on_calibration")
].iloc[0]

loo = summary[
    (summary["split"] == "calibration")
    & (summary["correction"] == "wavelength_linear_loo")
].iloc[0]

print(f"Holdout coverage: {holdout['coverage']}")
print(f"Holdout corrected RMSE dB: {holdout['rmse_db']}")
print(f"Holdout corrected MAE dB: {holdout['mae_db']}")
print(f"Holdout corrected bias dB: {holdout['bias_db']}")
print(f"Holdout corrected max abs error dB: {holdout['max_abs_error_db']}")
print(f"Calibration LOO RMSE dB: {loo['rmse_db']}")
print(f"Calibration LOO bias dB: {loo['bias_db']}")
