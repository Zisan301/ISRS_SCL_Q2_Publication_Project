# GNPy multiband spectrum fix

Your error:

IndexError: index 1 is out of bounds for axis 0 with size 1

happens because the earlier S+C+L spectrum used isolated one-channel spectrum slots. The multiband amplifier interpolation needs more than one channel in the active band.

This pack creates a band-partition spectrum with many 50 GHz channels in each band:
- S: 1495-1525 nm
- C: 1535-1560 nm
- L: 1570-1610 nm

## Commands

From CMD:

cd /d "E:\VS Code\ISRS_SCL_Q2_Publication_Project"
E:\VS Code\gnpy_env\Scripts\activate

Make sure these are set:

set GNPY_DATA=E:\VS Code\gnpy_env\Lib\site-packages\gnpy\example-data
set GNPY_VERSION=YOUR_VERSION_HERE

Run:

run_gnpy_multiband_bandpartition_test.bat

If successful, it writes:
external_validation\gnpy\raw_outputs\gnpy_scl_multiband_bandpartition_1span.txt

and appends target rows to:
validation_data\external_reference.csv

Important:
This produces only the 1-span multiband test first. After that works, use the same idea for 4-span and 8-span.
