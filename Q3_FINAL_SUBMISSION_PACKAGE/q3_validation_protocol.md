# Q3 held-out external-validation protocol

This protocol upgrades the current C-band GNPy diagnostic into a safer Q3-level evidence path.

## Current calibration/train evidence

The original clean C-band GNPy rows are frozen as the calibration/train split:

```text
validation_data/gnpy_calibration_train_1535_1550_1560.csv
```

These rows cover:

- 1535, 1550 and 1560 nm;
- 1, 4 and 8 spans;
- flat launch;
- `gsnr_db` from GNPy.

They may be used to estimate a simple systematic residual trend. They must not be reported as an
independent held-out test after that correction is fitted.

## Required held-out GNPy rows

Generate a new file only after real GNPy runs are completed:

```text
validation_data/gnpy_holdout_1540_1545_1555.csv
```

Required rows:

| wavelength_nm | spans | strategy | metric |
|---:|---:|---|---|
| 1540 | 1 | flat | gsnr_db |
| 1540 | 4 | flat | gsnr_db |
| 1540 | 8 | flat | gsnr_db |
| 1545 | 1 | flat | gsnr_db |
| 1545 | 4 | flat | gsnr_db |
| 1545 | 8 | flat | gsnr_db |
| 1555 | 1 | flat | gsnr_db |
| 1555 | 4 | flat | gsnr_db |
| 1555 | 8 | flat | gsnr_db |

The file must use the same schema as `validation_data/external_reference.csv`:

```text
source_id,source_type,tool_version,configuration_hash,date,provenance_reference,independent,strategy,spans,band,wavelength_nm,metric,metric_unit,reference_value,reference_uncertainty,notes
```

Do not add placeholder, guessed or interpolated GNPy values. The loader intentionally rejects blank,
placeholder and dependent rows.

## Validation rule

The project now includes:

```text
src/isrs_scl/validation/heldout_external_validation.py
```

The diagnostic performs:

1. model-vs-GNPy comparison on the calibration split;
2. wavelength-linear residual fitting on calibration rows only;
3. leave-one-out calibration diagnostics;
4. model-vs-GNPy comparison on unseen 1540/1545/1555 nm rows;
5. correction application to held-out rows without refitting.

Correction model:

```text
residual_db = intercept_db + slope_db_per_nm * (wavelength_nm - 1550.0)
corrected_model_value = model_value - residual_db
```

Residual definition:

```text
residual = model_value - reference_value
```

## Q3 minimum target

A defensible Q3-level C-band claim should satisfy:

- calibration LOO RMSE below 0.5 to 1.0 dB;
- held-out corrected RMSE below 0.5 to 1.0 dB;
- held-out corrected absolute bias below 0.75 dB;
- 100% coverage on the nine held-out rows;
- no source identity overlap between calibration and held-out rows.

## Safe manuscript wording

Use wording like:

> The calibration trend was estimated only from the original 1535/1550/1560 nm C-band GNPy rows and
> was then evaluated on unseen 1540/1545/1555 nm GNPy rows. This split separates calibration from
> held-out validation and reduces the risk of over-claiming diagnostic agreement.

Avoid claiming full S+C+L journal-level validation until independent SSFM or experimental evidence is
also added.
