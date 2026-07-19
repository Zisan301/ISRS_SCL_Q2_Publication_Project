@echo off
setlocal
cd /d "%~dp0"

if "%GNPY_VERSION%"=="" (
  echo ERROR: Set GNPY_VERSION first, e.g. set GNPY_VERSION=2.12.0
  exit /b 1
)

echo Running 1-span flat GNPy case...
gnpy-transmission-example external_validation\gnpy\cases\gnpy_flat_1span_network.json --show-channels -po 0 > external_validation\gnpy\raw_outputs\gnpy_flat_1span.txt
if errorlevel 1 exit /b 1

echo Running 4-span flat GNPy case...
gnpy-transmission-example external_validation\gnpy\cases\gnpy_flat_4span_network.json --show-channels -po 0 > external_validation\gnpy\raw_outputs\gnpy_flat_4span.txt
if errorlevel 1 exit /b 1

echo Running 8-span flat GNPy case...
gnpy-transmission-example external_validation\gnpy\cases\gnpy_flat_8span_network.json --show-channels -po 0 > external_validation\gnpy\raw_outputs\gnpy_flat_8span.txt
if errorlevel 1 exit /b 1

echo Parsing rows into validation_data\external_reference.csv...
python tools\parse_gnpy_show_channels_to_external_csv.py --raw-file external_validation\gnpy\raw_outputs\gnpy_flat_1span.txt --tool-version %GNPY_VERSION% --spans 1 --strategy flat --targets 1550,1560,1565
python tools\parse_gnpy_show_channels_to_external_csv.py --raw-file external_validation\gnpy\raw_outputs\gnpy_flat_4span.txt --tool-version %GNPY_VERSION% --spans 4 --strategy flat --targets 1550,1560,1565
python tools\parse_gnpy_show_channels_to_external_csv.py --raw-file external_validation\gnpy\raw_outputs\gnpy_flat_8span.txt --tool-version %GNPY_VERSION% --spans 8 --strategy flat --targets 1550,1560,1565

python tools\check_external_reference_schema.py validation_data\external_reference.csv
echo Done.
