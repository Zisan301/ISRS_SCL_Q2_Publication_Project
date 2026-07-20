# Today C-band cleanup workflow

Goal: make a clean same-day preliminary C-band evidence package.

This is not a full S+C+L Q2/Q3 evidence package. It removes bad rows and rebuilds only clean C-band GNPy rows from the flat GNPy raw outputs.

## It removes/replaces

- fake S-band rows where the parser mapped requested 1495 nm to nearest 1535 nm
- multiband rows with channel_power_dbm around -20.03 dBm
- rows containing WARNING nearest differs
- C/L mixed rows not comparable to the current model

## It writes

validation_data/external_reference.csv

with 9 rows:

- 1535, 1550, 1560 nm
- 1, 4, 8 spans
- strategy: flat
- source_type: GNPy
- expected GNPy channel power around -2 dBm

## Commands

From your project root:

cd /d "E:\VS Code\ISRS_SCL_Q2_Publication_Project"
E:\VS Code\gnpy_env\Scripts\activate
set GNPY_VERSION=2.14.1

Run all:

run_today_clean_cband_q3_smoke.bat

Or manual:

python tools\rebuild_clean_cband_external_reference.py --tool-version %GNPY_VERSION%
python tools\check_external_reference_schema.py validation_data\external_reference.csv
python run_all_experiments.py --smoke --no-uncertainty --config config_q2_final.yaml --run-id q3-cband-clean-smoke-001 --output-root runs --overwrite
python tools\summarize_external_validation_comparisons.py runs\q3-cband-clean-smoke-001\results\external_validation_comparisons.csv

## After run

Open:

type runs\q3-cband-clean-smoke-001\metadata\VALIDATION_STATUS.json
type runs\q3-cband-clean-smoke-001\results\external_validation_requirements.csv
type runs\q3-cband-clean-smoke-001\results\external_validation_summary.csv
