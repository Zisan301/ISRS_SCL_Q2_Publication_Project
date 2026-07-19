# GNPy UTF-8 console fix

Your error was:

UnicodeEncodeError: 'charmap' codec can't encode character '\u2192'

That is a Windows console encoding problem, not a GNPy optical simulation problem.

This fixed batch file sets:
- chcp 65001
- PYTHONUTF8=1
- PYTHONIOENCODING=utf-8

It also redirects stderr into the output files with `2>&1`.

## Run

From CMD or PowerShell:

cd /d "E:\VS Code\ISRS_SCL_Q2_Publication_Project"
E:\VS Code\gnpy_env\Scripts\activate
set GNPY_DATA=E:\VS Code\gnpy_env\Lib\site-packages\gnpy\example-data
set GNPY_VERSION=YOUR_VERSION_HERE
run_gnpy_multiband_bandpartition_test_utf8.bat
