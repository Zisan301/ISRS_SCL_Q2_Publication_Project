@echo off
setlocal
cd /d "%~dp0"

echo Running external-validation-only diagnostic...
python tools\run_external_validation_only.py --config config_q2_final.yaml --out-dir runs\q3-cband-external-only-001 --allow-interpolation
echo.
echo Files:
echo runs\q3-cband-external-only-001\external_validation_requirements.csv
echo runs\q3-cband-external-only-001\external_validation_summary.csv
echo runs\q3-cband-external-only-001\external_validation_comparisons.csv
