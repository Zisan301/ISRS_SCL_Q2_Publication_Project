# Easy GNPy flat external validation

This pack makes the next step easy:

1. Build 1/4/8-span flat GNPy network JSON files.
2. Run GNPy for those 3 cases.
3. Parse nearest 1550/1560/1565 nm GSNR rows into validation_data/external_reference.csv.
4. Run the schema checker.

## Commands

Open CMD:

cd /d "E:\VS Code\ISRS_SCL_Q2_Publication_Project"

Activate your GNPy environment:

E:\VS Code\gnpy_env\Scripts\activate

Get your GNPy example-data path:

python -c "from gnpy.tools.cli_examples import show_example_data_dir; print(show_example_data_dir())"

Use the printed path:

python tools\make_gnpy_flat_span_cases.py --example-data "PASTE_PRINTED_PATH_HERE"

Set your GNPy version:

set GNPY_VERSION=YOUR_VERSION_HERE

Run all flat cases:

run_gnpy_flat_cases.bat

Check CSV:

python tools\check_external_reference_schema.py validation_data\external_reference.csv

Then run model debug:

python run_all_experiments.py --debug --config config_q2_final.yaml --run-id q2-debug-external-flat-001 --output-root runs --overwrite

## Important

This easy pack gives you C/L-edge flat references using GNPy's built-in EDFA example spectrum. It is useful for external-validation workflow and partial independent evidence. It still does not fully solve S-band because the built-in example output you showed starts around 191.35 THz, which corresponds to around 1566.6 nm and does not include true S band.
