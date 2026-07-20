@echo off
setlocal
cd /d "%~dp0"

echo Running C-band GSNR bias diagnostic...
python tools\analyze_cband_gnpy_bias.py ^
  --comparisons runs\q3-cband-external-only-2db-001\external_validation_comparisons.csv ^
  --out-dir runs\q3-cband-bias-diagnostic-001

echo.
echo Open these files:
echo runs\q3-cband-bias-diagnostic-001\cband_bias_diagnostic_report.json
echo runs\q3-cband-bias-diagnostic-001\cband_bias_by_span.csv
echo runs\q3-cband-bias-diagnostic-001\cband_bias_by_wavelength.csv
