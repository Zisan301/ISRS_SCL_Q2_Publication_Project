@echo off
setlocal
cd /d "%~dp0"

echo Running wavelength-dependent C-band bias diagnostic...
python tools\fit_cband_wavelength_bias.py ^
  --comparisons runs\q3-cband-external-only-2db-001\external_validation_comparisons.csv ^
  --out-dir runs\q3-cband-wavelength-bias-001

echo.
echo Open:
echo runs\q3-cband-wavelength-bias-001\cband_wavelength_bias_report.json
echo runs\q3-cband-wavelength-bias-001\cband_wavelength_bias_rows.csv
