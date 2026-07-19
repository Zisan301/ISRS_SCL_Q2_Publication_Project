# GNPy S+C+L 27 flat rows

This pack tries to produce the 27 flat rows:

9 target wavelengths:
S: 1495, 1515, 1525 nm
C: 1535, 1550, 1560 nm
L: 1570, 1595, 1610 nm

3 span counts:
1, 4, 8

1 strategy:
flat

Total:
9 × 3 × 1 = 27 rows

## Commands

Open CMD:

cd /d "E:\VS Code\ISRS_SCL_Q2_Publication_Project"

Activate your GNPy env:

E:\VS Code\gnpy_env\Scripts\activate

Find GNPy example data:

python -c "from gnpy.tools.cli_examples import show_example_data_dir; print(show_example_data_dir())"

Build span network cases:

python tools\make_gnpy_flat_span_cases.py --example-data "PASTE_PRINTED_PATH_HERE"

Build the S+C+L custom spectrum:

python tools\make_gnpy_scl_target_spectrum.py

Set GNPy version:

set GNPY_VERSION=YOUR_VERSION_HERE

Run the 27-row flat workflow:

run_gnpy_scl_27_flat.bat

Then run your model:

python run_all_experiments.py --debug --config config_q2_final.yaml --run-id q2-debug-external-27flat-001 --output-root runs --overwrite

## If GNPy fails

If it fails during the S+C+L run, open:

external_validation\gnpy\raw_outputs\gnpy_scl_flat_1span.txt

The most likely reason is that the built-in EDFA example equipment does not support the full S-band and full L-band frequency range. That is not your Python pipeline bug; it means the GNPy external reference setup must use proper S+C+L/multiband equipment definitions.
