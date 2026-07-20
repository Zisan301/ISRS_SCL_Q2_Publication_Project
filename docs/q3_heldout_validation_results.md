git add docs\q3_heldout_validation_results.md# Q3 Held-Out GNPy Validation Results

## Validation status

- Passed: **True**
- Reasons: `[]`

## Fitted calibration correction

- center_wavelength_nm: `1550.0`
- intercept_db: `2.543253976730291`
- model: `wavelength_linear`
- residual_definition: `model_value - reference_value`
- slope_db_per_nm: `-0.008742608622570414`

## Summary metrics

| split       | correction                               |   requested_rows |   matched_rows |   coverage |   n |   rmse_db |   mae_db |      bias_db |   max_abs_error_db |
|:------------|:-----------------------------------------|-----------------:|---------------:|-----------:|----:|----------:|---------:|-------------:|-------------------:|
| calibration | raw                                      |                9 |              9 |          1 |   9 |  2.58395  | 2.55782  |  2.55782     |           2.97695  |
| calibration | wavelength_linear_in_sample              |                9 |              9 |          1 |   9 |  0.355314 | 0.333152 | -3.94746e-16 |           0.527547 |
| calibration | wavelength_linear_loo                    |                9 |              9 |          1 |   9 |  0.465965 | 0.435075 |  0.00369284  |           0.724812 |
| holdout     | raw                                      |                9 |              9 |          1 |   9 |  2.80386  | 2.78394  |  2.78394     |           3.10951  |
| holdout     | wavelength_linear_trained_on_calibration |                9 |              9 |          1 |   9 |  0.388795 | 0.377702 |  0.210965    |           0.477103 |

## Requirement checks

| requirement                 | passed   |   observed |   threshold |
|:----------------------------|:---------|-----------:|------------:|
| no_holdout_identity_overlap | True     |   0        |        0    |
| holdout_minimum_coverage    | True     |   1        |        1    |
| holdout_gsnr_rmse           | True     |   0.388795 |        1    |
| holdout_gsnr_absolute_bias  | True     |   0.210965 |        0.75 |
| calibration_loo_gsnr_rmse   | True     |   0.465965 |        1    |

## Paper-ready interpretation

The original GNPy reference points at 1535, 1550, and 1560 nm over 1, 4, and 8 spans were used only as calibration evidence. A separate held-out GNPy validation set was generated at unseen wavelengths near 1540, 1545, and 1555 nm over the same span counts. A wavelength-linear residual correction was fitted on the calibration split only and then applied to the held-out split without refitting. The held-out validation gate passed all configured requirements, with no failed validation reasons reported.

## Limitation

This validation is independent simulation-to-simulation validation against GNPy. It should not be described as laboratory or field experimental validation unless physical measurement data are added.