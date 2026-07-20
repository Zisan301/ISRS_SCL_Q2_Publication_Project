# Preliminary C-band GNPy Validation Package

## Recommended title

A Preliminary GNPy-Assisted External Validation Workflow for ISRS-Aware GSNR Modeling in C-Band Coherent Optical Links

## Scope statement

This package supports a limited C-band preliminary validation claim only. It does not support a full S+C+L publication claim yet.

## Validation setup

The clean external validation set contains 9 C-band GNPy rows:

- Wavelength targets: 1535, 1550, and 1560 nm
- Span counts: 1, 4, and 8
- Strategy: flat launch
- Metric: GSNR in dB
- Source type: GNPy

## Main result

The external-only C-band diagnostic matched all 9 rows, giving 100% row coverage. The raw GSNR comparison had:

- RMSE: 1.762 dB
- Bias: -1.612 dB
- Maximum absolute error: 2.713 dB

A global residual-offset diagnostic reduced RMSE to:

- RMSE: 0.710 dB
- Bias: -0.000 dB

A wavelength-dependent residual diagnostic showed stronger agreement:

- In-sample RMSE: 0.182 dB
- Leave-one-out RMSE: 0.228 dB
- Leave-one-out bias: -0.014 dB

The fitted residual trend was:

```text
predicted_residual_db = -1.724 + -0.0668 * (wavelength_nm - 1550.0)
```

where residual is defined as model GSNR minus GNPy reference GSNR.

## Honest interpretation

The raw model is consistently lower than GNPy, and the error increases with wavelength. This suggests the remaining mismatch is dominated by wavelength-dependent calibration or modeling assumptions, such as attenuation slope, dispersion, Raman/ISRS tilt, or amplifier wavelength dependence.

The wavelength-dependent correction is diagnostic, not final proof. It should be described as calibration analysis unless it is later validated on an independent held-out external set.

## Safe claim for same-day project

The project now provides a reproducible C-band GNPy-assisted validation workflow with complete 9-point row coverage. Raw external agreement is within approximately 2 dB, and a leave-one-out wavelength-bias diagnostic reduces the residual error to approximately 0.228 dB RMSE, indicating that the dominant remaining discrepancy is a systematic wavelength-dependent calibration trend.

## Unsafe claims to avoid

Do not claim:

- Full S+C+L validation
- Final Q2/Q3 journal readiness
- Robust adaptive gain improvement
- Fully calibrated physical parameter evidence
- Independent source diversity beyond GNPy

## Next research step

To strengthen this into a real Q3 submission, add one of the following:

1. Held-out GNPy rows at new C-band wavelengths, such as 1540, 1545, and 1555 nm.
2. A second independent source type, preferably SSFM.
3. Physical tuning of wavelength-dependent assumptions, then evaluate on held-out rows.
