@echo off
setlocal
cd /d "%~dp0"

echo Preparing same-day C-band preliminary paper/evidence package...
python tools\prepare_today_cband_paper_package.py

echo.
echo Package created:
echo paper_package_cband_preliminary_20260720
echo.
echo Open:
echo paper_package_cband_preliminary_20260720\PAPER_CLAIMS_AND_RESULTS.md
echo paper_package_cband_preliminary_20260720\tables\key_cband_validation_results.csv
