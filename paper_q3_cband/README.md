# Q3-oriented C-band manuscript

This folder contains a LaTeX manuscript prepared for the reduced, defensible paper scope:

**A Reproducible GNPy-Assisted External Validation Workflow for ISRS-Aware GSNR Modeling in C-Band Coherent Optical Links**

## Important scope note

This is **not** a final S+C+L validation paper. The manuscript is written as a preliminary C-band validation workflow because the current clean external evidence covers:

- C-band only
- GNPy source type only
- flat launch strategy only
- 9 reference rows: 1535, 1550, and 1560 nm over 1, 4, and 8 spans

The manuscript avoids unsupported claims of full Q2/Q3 readiness, full S+C+L validation, or optimizer validation.

## Compile

From this folder:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

or, if available:

```bash
latexmk -pdf main.tex
```

## Evidence files expected before submission

The manuscript refers to the generated evidence package. Before journal submission, make sure the repository contains or archives:

```text
paper_package_cband_preliminary_20260720/
validation_data/external_reference.csv
tools/run_external_validation_only.py
tools/analyze_cband_gnpy_bias.py
tools/fit_cband_wavelength_bias.py
tools/prepare_today_cband_paper_package.py
```

The most important generated evidence files are:

```text
external_validation_comparisons.csv
external_validation_summary.csv
external_validation_requirements.csv
external_only_model_channel_sweep.csv
cband_bias_diagnostic_report.json
cband_wavelength_bias_report.json
key_cband_validation_results.csv
```

## Current results used in the manuscript

- Requested rows: 9
- Matched rows: 9
- Coverage: 1.000
- Raw GSNR RMSE: 1.762 dB
- Raw GSNR bias: -1.612 dB
- Global offset corrected RMSE: 0.710 dB
- Wavelength-bias in-sample RMSE: 0.182 dB
- Wavelength-bias leave-one-out RMSE: 0.228 dB
- Wavelength-bias leave-one-out bias: -0.014 dB

## What must be improved before a stronger Q3 submission

1. Add held-out C-band GNPy rows that were not used in the wavelength-bias fit, for example 1540, 1545, and 1555 nm.
2. Add a second independent source type, preferably SSFM.
3. Extend the clean validation to S and L bands only after power levels and wavelength matching are physically comparable.
4. Avoid claiming optimizer performance until the external validation gate and optimizer-acceptance gate both pass.
5. Freeze a commit hash and archive the exact evidence package before submission.
