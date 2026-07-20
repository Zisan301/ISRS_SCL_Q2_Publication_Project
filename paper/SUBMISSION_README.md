# Q3 Paper Submission Notes

This folder contains the LaTeX manuscript draft for the ISRS-SCL Q3 submission package.

## Files

- `main.tex` — IEEE-style manuscript draft.
- `references.bib` — bibliography used by `main.tex`.

## Compile

From this folder, run one of the following.

```bash
latexmk -pdf main.tex
```

or:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Before submission

Replace the following placeholders before submitting:

1. author names;
2. affiliation;
3. author email;
4. acknowledgment/funding text;
5. venue-specific template, if the target conference or journal requires a different class file.

## Evidence used in the paper

The paper uses the held-out validation result from:

- `docs/q3_heldout_validation_results.md`
- `validation_data/gnpy_calibration_train_1535_1550_1560.csv`
- `validation_data/gnpy_holdout_1540_1545_1555.csv`
- `runs/q3-heldout-model-001/results/heldout_external_validation_summary.csv`
- `runs/q3-heldout-model-001/results/heldout_external_validation_requirements.csv`
- `runs/q3-heldout-model-001/results/heldout_external_validation_status.json`

The strongest validated claim is C-band, flat-launch, simulation-to-simulation agreement against GNPy on a calibration-separated held-out split. Do not describe this as laboratory or field experimental validation unless measured data are added.

## Recommended safe claim

> The proposed ISRS-aware GSNR workflow was externally checked against GNPy using a calibration-separated C-band split. A wavelength-linear residual correction fitted on 1535/1550/1560 nm calibration rows generalized to unseen 1540/1545/1555 nm held-out rows over 1, 4, and 8 spans, achieving 0.389 dB corrected GSNR RMSE and 0.211 dB corrected bias with 100% held-out coverage.
