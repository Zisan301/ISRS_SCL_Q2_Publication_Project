# C-band GSNR bias diagnostic

You already passed the limited 2 dB external-only diagnostic.

Next goal:
Understand why model GSNR is around 1.61 dB lower than GNPy.

## Run

From project root:

cd /d "E:\VS Code\ISRS_SCL_Q2_Publication_Project"
E:\VS Code\gnpy_env\Scripts\activate
python -m pip install -e .

Then:

run_today_cband_bias_diagnostic.bat

## What to look for

If the report says:

- offset to add to model is around +1.61 dB
- corrected RMSE becomes below 1 dB
- bias by span and wavelength is not changing much

Then the mismatch is mostly a global calibration/noise offset.

That means the next research improvement is:
- tune amplifier noise figure / ASE calibration,
- confirm span loss and launch power alignment,
- or document a calibrated offset and validate on held-out rows.

Do not claim final Q3/Q2 readiness from this alone. It strengthens a preliminary C-band validation package.
