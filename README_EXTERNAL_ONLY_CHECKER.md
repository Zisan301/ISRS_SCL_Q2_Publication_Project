# External-validation-only checker

Your smoke run matched only 2 of 9 rows because smoke mode reduces the grid/span range. This checker avoids the heavy pipeline and runs only:

1. build configured grid
2. build LinkModel
3. compute flat GSNR sweeps
4. compare validation_data/external_reference.csv

## Run

From project root:

cd /d "E:\VS Code\ISRS_SCL_Q2_Publication_Project"
E:\VS Code\gnpy_env\Scripts\activate
python -m pip install -e .

Then:

run_today_external_only_cband_check.bat

or manual:

python tools\run_external_validation_only.py --config config_q2_final.yaml --out-dir runs\q3-cband-external-only-001 --allow-interpolation

## What you want

Requested rows: 9
Matched rows: 9
Coverage: near 1.000
GSNR RMSE: below 1.5 dB
GSNR bias: below 1.5 dB

If that passes, you have a defendable limited C-band diagnostic result for a preliminary paper/package.
