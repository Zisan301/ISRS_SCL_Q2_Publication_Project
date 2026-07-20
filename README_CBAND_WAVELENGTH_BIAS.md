# C-band wavelength bias diagnostic

Your previous diagnostic showed:
- raw RMSE around 1.762 dB
- global offset correction RMSE around 0.710 dB
- residual grows with wavelength from 1535 to 1560 nm

This pack tests a linear wavelength-dependent residual correction:
predicted_residual_db = a + b * (wavelength_nm - 1550)

It also performs leave-one-out validation so we do not trust only an in-sample fit.

## Run

cd /d "E:\VS Code\ISRS_SCL_Q2_Publication_Project"
E:\VS Code\gnpy_env\Scripts\activate
python -m pip install -e .
run_today_cband_wavelength_bias_check.bat

## What to send back

Send the console output, especially:
- in-sample RMSE
- leave-one-out RMSE
- slope_db_per_nm
