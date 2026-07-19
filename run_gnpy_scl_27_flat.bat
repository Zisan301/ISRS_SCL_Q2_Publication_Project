@echo off
setlocal
cd /d "%~dp0"

if "%GNPY_VERSION%"=="" (
  echo ERROR: Set GNPY_VERSION first, e.g. set GNPY_VERSION=2.12.0
  exit /b 1
)

set TARGETS=1495,1515,1525,1535,1550,1560,1570,1595,1610
set SPECTRUM=external_validation\gnpy\cases\scl_9_targets_flat_spectrum.json

echo Resetting validation_data\external_reference.csv...
python tools\reset_external_reference_csv.py
if errorlevel 1 exit /b 1

echo Running 1-span S+C+L target spectrum...
gnpy-transmission-example external_validation\gnpy\cases\gnpy_flat_1span_network.json --spectrum %SPECTRUM% --show-channels -po 0 > external_validation\gnpy\raw_outputs\gnpy_scl_flat_1span.txt
if errorlevel 1 (
  echo.
  echo ERROR: GNPy failed for S+C+L spectrum.
  echo Most likely the built-in EDFA equipment does not support the S/C/L frequency range.
  echo Open external_validation\gnpy\raw_outputs\gnpy_scl_flat_1span.txt for details.
  exit /b 1
)

echo Running 4-span S+C+L target spectrum...
gnpy-transmission-example external_validation\gnpy\cases\gnpy_flat_4span_network.json --spectrum %SPECTRUM% --show-channels -po 0 > external_validation\gnpy\raw_outputs\gnpy_scl_flat_4span.txt
if errorlevel 1 exit /b 1

echo Running 8-span S+C+L target spectrum...
gnpy-transmission-example external_validation\gnpy\cases\gnpy_flat_8span_network.json --spectrum %SPECTRUM% --show-channels -po 0 > external_validation\gnpy\raw_outputs\gnpy_scl_flat_8span.txt
if errorlevel 1 exit /b 1

echo Parsing 27 rows into validation_data\external_reference.csv...
python tools\parse_gnpy_show_channels_to_external_csv.py --raw-file external_validation\gnpy\raw_outputs\gnpy_scl_flat_1span.txt --tool-version %GNPY_VERSION% --spans 1 --strategy flat --targets %TARGETS%
python tools\parse_gnpy_show_channels_to_external_csv.py --raw-file external_validation\gnpy\raw_outputs\gnpy_scl_flat_4span.txt --tool-version %GNPY_VERSION% --spans 4 --strategy flat --targets %TARGETS%
python tools\parse_gnpy_show_channels_to_external_csv.py --raw-file external_validation\gnpy\raw_outputs\gnpy_scl_flat_8span.txt --tool-version %GNPY_VERSION% --spans 8 --strategy flat --targets %TARGETS%

python tools\check_external_reference_schema.py validation_data\external_reference.csv
echo Done.
