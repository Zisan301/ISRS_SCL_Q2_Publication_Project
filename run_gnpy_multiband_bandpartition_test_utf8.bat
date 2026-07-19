@echo off
setlocal
cd /d "%~dp0"

REM Fix Windows console UnicodeEncodeError from GNPy output, e.g. arrow symbol U+2192.
chcp 65001 > nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

if "%GNPY_VERSION%"=="" (
  echo ERROR: Set GNPY_VERSION first, e.g. set GNPY_VERSION=2.12.0
  exit /b 1
)

if "%GNPY_DATA%"=="" (
  echo ERROR: Set GNPY_DATA first, e.g.
  echo set GNPY_DATA=E:\VS Code\gnpy_env\Lib\site-packages\gnpy\example-data
  exit /b 1
)

set NETWORK=%GNPY_DATA%\multiband_example_network.json
set EQUIP=%GNPY_DATA%\eqpt_config_multiband.json
set SPECTRUM=external_validation\gnpy\cases\scl_band_partition_flat_spectrum.json
set TARGETS=1495,1515,1525,1535,1550,1560,1570,1595,1610

echo Building multiband-compatible S+C+L spectrum...
python tools\make_gnpy_scl_band_partition_spectrum.py
if errorlevel 1 exit /b 1

echo Running built-in multiband test first...
gnpy-transmission-example "%NETWORK%" --equipment "%EQUIP%" --spectrum "%GNPY_DATA%\multiband_spectrum.json" --show-channels -po 0 > external_validation\gnpy\raw_outputs\gnpy_multiband_builtin_test.txt 2>&1
if errorlevel 1 (
  echo ERROR: Built-in multiband test failed. See external_validation\gnpy\raw_outputs\gnpy_multiband_builtin_test.txt
  exit /b 1
)

echo Running S+C+L band-partition multiband case...
gnpy-transmission-example "%NETWORK%" --equipment "%EQUIP%" --spectrum "%SPECTRUM%" --show-channels -po 0 > external_validation\gnpy\raw_outputs\gnpy_scl_multiband_bandpartition_1span.txt 2>&1
if errorlevel 1 (
  echo ERROR: S+C+L band-partition GNPy run failed.
  echo See external_validation\gnpy\raw_outputs\gnpy_scl_multiband_bandpartition_1span.txt
  exit /b 1
)

echo Parsing target rows if output is valid...
python tools\parse_gnpy_show_channels_to_external_csv.py --raw-file external_validation\gnpy\raw_outputs\gnpy_scl_multiband_bandpartition_1span.txt --tool-version %GNPY_VERSION% --spans 1 --strategy flat --targets %TARGETS%

python tools\check_external_reference_schema.py validation_data\external_reference.csv
echo Done.
