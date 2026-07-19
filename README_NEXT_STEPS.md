# Correct GNPy external-reference next step

Your uploaded GNPy output was parsed successfully.

Best extracted rows from that file:

| requested nm | actual GNPy wavelength nm | channel | GSNR signal-bw dB |
|---:|---:|---:|---:|
| 1550.0 | 1550.116122 | 42 | 26.99 |
| 1560.0 | 1560.200146 | 17 | 27.11 |
| 1565.0 | 1565.087225 | 5 | 27.33 |


Important:
This uses the ISRS_SCL validator's real schema:
- `spans`
- `reference_value`
- `reference_uncertainty`

Use the parser like this:

python tools\parse_gnpy_show_channels_to_external_csv.py ^
  --raw-file external_validation\gnpy\raw_outputs\gnpy_test_flat_1span.txt ^
  --tool-version YOUR_GNPY_VERSION ^
  --spans 1 ^
  --strategy flat ^
  --targets 1550,1560,1565

Then check:

python tools\check_external_reference_schema.py validation_data\external_reference.csv

Then run:

python run_all_experiments.py --debug --config config_q2_final.yaml --run-id q2-debug-external-test-001 --output-root runs --overwrite

This file is only 1-span flat. It is not enough for full Q2 validation.
