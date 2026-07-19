# ISRS-SCL fast preview configs

These configs are for quickly generating draft CSV outputs and figures. They are **not journal evidence**.

## Fast preview, usually a few minutes

```bat
python run_all_experiments.py --debug --config config_fast_preview.yaml --run-id q2-fast-preview-001 --output-root runs --overwrite
```

## Turbo preview, fastest draft check

```bat
python run_all_experiments.py --debug --config config_turbo_preview.yaml --run-id q2-turbo-preview-001 --output-root runs --overwrite --no-uncertainty
```

Use the outputs only to inspect plots, tables, and pipeline behavior. For Q2 publication claims, go back to `config_q2_final.yaml` and publication mode after adding real calibration and independent external validation data.
