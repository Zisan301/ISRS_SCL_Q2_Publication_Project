@echo off
setlocal
cd /d "%~dp0"

if "%GNPY_VERSION%"=="" (
  echo ERROR: set GNPY_VERSION first, for example:
  echo set GNPY_VERSION=2.14.1
  exit /b 1
)

echo Rebuilding clean C-band-only external_reference.csv...
python tools\rebuild_clean_cband_external_reference.py --tool-version %GNPY_VERSION%
if errorlevel 1 exit /b 1

echo Checking external_reference.csv schema...
python tools\check_external_reference_schema.py validation_data\external_reference.csv
if errorlevel 1 exit /b 1

echo Running fast smoke without uncertainty...
python run_all_experiments.py --smoke --no-uncertainty --config config_q2_final.yaml --run-id q3-cband-clean-smoke-001 --output-root runs --overwrite
if errorlevel 1 exit /b 1

echo Summarizing external validation comparisons...
python tools\summarize_external_validation_comparisons.py runs\q3-cband-clean-smoke-001\results\external_validation_comparisons.csv

echo.
echo Open validation status:
echo type runs\q3-cband-clean-smoke-001\metadata\VALIDATION_STATUS.json
