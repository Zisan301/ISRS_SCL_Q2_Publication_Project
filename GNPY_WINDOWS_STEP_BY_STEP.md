# GNPy External Validation — Windows CMD Step-by-Step

Use this after you already installed GNPy with Option B and got the GNPy version.

## Folder assumed

Project:
E:\VS Code\ISRS_SCL_Q2_Publication_Project

## 1. Enter your project folder

cd /d "E:\VS Code\ISRS_SCL_Q2_Publication_Project"

## 2. Activate the GNPy environment

Change the path if your environment is somewhere else.

E:\VS Code\gnpy_env\Scripts\activate

Check:

python -c "import importlib.metadata as m; print(m.version('gnpy'))"

Copy the printed version. You will use it as `tool_version`.

## 3. Create raw-output folders

mkdir external_validation\gnpy\cases
mkdir external_validation\gnpy\raw_outputs

## 4. Find the GNPy example-data path

Run:

python -c "from gnpy.tools.cli_examples import show_example_data_dir; print(show_example_data_dir())"

Copy the printed folder path.

## 5. Run ONE first GNPy test

Use the folder path printed above. Example:

gnpy-transmission-example "PASTE_PRINTED_PATH_HERE\edfa_example_network.json" --show-channels -po 0 > external_validation\gnpy\raw_outputs\gnpy_test_flat_1span.txt

## 6. Open the output

notepad external_validation\gnpy\raw_outputs\gnpy_test_flat_1span.txt

Find a row/field named SNR or GSNR. Copy the numerical dB value.

## 7. Append the first real row

Replace YOUR_GNPY_VERSION and REAL_SNR_VALUE_FROM_OUTPUT:

python tools\append_external_reference_row.py ^
  --source-id gnpy_test_flat_1span_1550nm_gsnr ^
  --source-type GNPy ^
  --tool-version YOUR_GNPY_VERSION ^
  --raw-file external_validation\gnpy\raw_outputs\gnpy_test_flat_1span.txt ^
  --span-count 1 ^
  --wavelength-nm 1550.0 ^
  --band C ^
  --strategy flat ^
  --metric gsnr_db ^
  --metric-value REAL_SNR_VALUE_FROM_OUTPUT ^
  --metric-unit dB ^
  --uncertainty 0.30 ^
  --notes "First independent GNPy smoke reference row"

## 8. Check the CSV

python tools\check_external_reference_schema.py validation_data\external_reference.csv

Expected:

OK: validation_data\external_reference.csv has 1 complete external-validation rows.

## 9. Run your model with this CSV

python run_all_experiments.py --debug --config config_q2_final.yaml --run-id q2-debug-external-test-001 --output-root runs --overwrite

## 10. After one row works, repeat for real coverage

Minimum flat-only debug coverage:
S band: 1495, 1515, 1525 nm
C band: 1535, 1550, 1560 nm
L band: 1570, 1595, 1610 nm
Spans: 1, 4, 8
Strategy: flat
Metric: gsnr_db

That is 27 rows.

Important:
The built-in GNPy EDFA example is enough to learn the workflow, but for paper evidence you must build GNPy cases matching your project assumptions. Do not copy ISRS_SCL outputs into external_reference.csv.
