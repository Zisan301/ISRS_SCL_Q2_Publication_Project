# ISRS_SCL_Q2_Publication_Project — Q2 Remediation Prompt Pack

Repository: `Zisan301/ISRS_SCL_Q2_Publication_Project`

## Current verdict

The repository is a strong simulation foundation, but it is not ready for a Q2 journal submission. The current publication gate fails five required checks:

1. Physical parameters are still uncalibrated.
2. S-band usability at the eight-span target is far below the configured threshold.
3. Representative waveform results are inconsistent with the analytical model in S band.
4. Independent GNPy/SSFM/experimental validation data are missing.
5. Adaptive launch optimization is not robust under parameter uncertainty.

The prompts below are written so they can be given individually to a coding model. Each prompt requires complete code, no placeholders, backward-compatible APIs where reasonable, tests, and documented outputs.

---

# A. Existing Python files that must be changed

## 1. `run_all_experiments.py`

### Problem
The main runner defaults to the weaker/older `config.yaml`, permits a non-strict final run, writes into shared output folders, and can mix fresh and stale results.

### Copy-ready prompt
> Modify `run_all_experiments.py` to make the publication workflow fail-safe. Set the default configuration to `config_q2_final.yaml`. Add mutually exclusive modes `--publication`, `--smoke`, and `--debug`; publication mode must automatically enable strict gate enforcement, uncertainty analysis, clean Git verification, and run-isolated output directories. Add `--run-id`, `--output-root`, and `--overwrite` options. Before a publication run, reject placeholder external-validation rows, reject `UNVALIDATED_DEFAULTS`, reject a dirty or unavailable Git revision unless an explicit `--allow-untracked-provenance` debugging flag is supplied, and refuse to reuse a non-empty run directory. Return distinct non-zero exit codes for configuration failure, numerical failure, validation-gate failure, and provenance failure. Print the final run directory, manifest path, validation status, and failed checks. Keep smoke/debug behavior available but label their outputs as non-publication evidence. Add type hints and unit-testable helper functions; do not hide exceptions without preserving the original cause.

### Expected result
A final paper run cannot accidentally use old configuration, stale artifacts, disabled uncertainty, or failed validation.

---

## 2. `src/isrs_scl/cli.py`

### Problem
The installed CLI is less capable than `run_all_experiments.py` and still defaults to `config.yaml`.

### Copy-ready prompt
> Refactor `src/isrs_scl/cli.py` so the installed `isrs-scl-study` command exposes the same safe interface and exit behavior as `run_all_experiments.py`. Reuse a shared parser/runner function rather than duplicating logic. Default to `config_q2_final.yaml`; provide publication, smoke, and debug modes; support run ID, isolated output root, strict gate, uncertainty control, and provenance options. Publication mode must not allow `--no-uncertainty`. Ensure `main()` returns an integer and the module exits with `SystemExit(main())`. Add clear help text explaining that smoke/debug outputs are not journal evidence.

### Expected result
The package entry point and repository runner behave identically and safely.

---

## 3. `src/isrs_scl/system/parameters.py`

### Problem
Configuration schema has drifted. Several old keys remain, several new keys are weakly validated, unknown keys are silently accepted, and publication/debug requirements are not separated.

### Copy-ready prompt
> Replace the loose configuration validation in `src/isrs_scl/system/parameters.py` with a strict, versioned schema implemented without adding a heavy dependency. Define explicit allowed keys, types, units, ranges, and cross-field constraints for every section. Reject unknown keys with a useful dotted-path error. Add `schema_version`, `run.mode`, calibration provenance, external-validation requirements, robust-optimization settings, waveform pilot/evaluation settings, uncertainty distributions, correlation settings, and output isolation settings. Remove or migrate obsolete optimization keys so one canonical name exists for each concept. Validate that the B2B SNR sweep covers the full predicted GSNR range with adequate resolution; validation data paths exist in publication mode; calibration sources have identifiers and traceable references; final runs have at least the configured minimum independent sources, bands, span counts, and wavelengths; robust-training and holdout uncertainty seeds differ; and publication output directories are run-specific. Provide a migration function from the old config schema and emit a machine-readable resolved configuration. Add precise `ConfigError` messages and tests for valid, invalid, unknown, and migrated configurations.

### Expected result
Configuration mistakes cannot silently alter paper claims, and only one canonical final configuration remains.

---

## 4. `src/isrs_scl/experiments.py`

### Problem
The orchestration mixes responsibilities, writes to shared folders, uses coarse/single-seed B2B calibration, clamps extrapolation, validates only fixed representative wavelengths, and does not isolate robust-optimization training from uncertainty holdout evaluation.

### Copy-ready prompt
> Refactor `src/isrs_scl/experiments.py` into a deterministic, stage-oriented publication pipeline. Create explicit stages for configuration/provenance validation, calibration loading, Raman convergence, nominal baselines, robust adaptive optimization, B2B receiver calibration, waveform validation, external validation, independent uncertainty holdout, plotting, gate evaluation, and manifest finalization. Each stage must write to a run-specific directory and return typed results. Do not reuse stale files.
>
> For B2B calibration, run multiple seeds per SNR point, cover below the minimum predicted GSNR through above the maximum predicted GSNR, use fine spacing near the FEC threshold, export confidence intervals, fit a monotone calibration curve, and reject extrapolation rather than clamping to endpoint values.
>
> For waveform validation, evaluate several operating points per band: high-margin, near-threshold, and below-threshold diagnostic points. Use disjoint pilot/training and payload-evaluation symbols. Export acquisition success, cycle slips, SNR/GMI confidence intervals, and reasons for failed receiver acquisition. Do not mark an unusable below-threshold channel as a model-consistency failure unless the receiver is expected to acquire there; report it separately.
>
> Run robust optimization on a training scenario set and evaluate the selected profile on an independently seeded holdout uncertainty set. Add paired comparisons against flat and fixed baselines at all configured span counts. Require external validation before final plotting/gate success. Include every final CSV, JSON, PDF, PNG, resolved config, source-data hash, and calibration file in the current-run required-artifact list.

### Expected result
One command produces a clean, traceable evidence package with statistically meaningful calibration and no training/evaluation leakage.

---

## 5. `src/isrs_scl/optimization/adaptive_isrs.py`

### Problem
The optimizer is nominal rather than robust, uses global average penalties that can sacrifice S band, and accepts gains that do not guarantee per-band usability or positive uncertainty-adjusted improvement.

### Copy-ready prompt
> Redesign `src/isrs_scl/optimization/adaptive_isrs.py` as a band-aware, robust constrained optimizer while preserving total launch power and per-channel bounds. Add per-band metrics for S, C, and L: minimum NGMI, lower-tail/CVaR NGMI, working fraction, AIR, and thresholded capacity. Replace the global mean outage term with configurable band-balanced penalties so the large/strong bands cannot hide S-band failure. Add hard or augmented-Lagrangian constraints for minimum working fraction in every band at every required span count. Support robust scenario evaluation supplied by a separate robust-scenario module, optimizing a weighted combination of nominal performance, worst-case/CVaR performance, and paired adaptive-minus-fixed gain. Use separate scenario batches or common random numbers for stable SPSA gradients. Add early stopping, gradient/objective diagnostics, feasibility diagnostics, multistart statistics, and deterministic candidate ranking. Candidate acceptance must require: no capacity/AIR regression, configured per-band feasibility, positive nominal gain, and a configured lower confidence bound for robust gain. Never relax a physical constraint merely to make the publication gate pass.

### Expected result
The selected adaptive profile either provides defensible S/C/L and robust gains or is correctly rejected.

---

## 6. `src/isrs_scl/optimization/statistics.py`

### Problem
This file uses an obsolete optimizer API and result attributes, so it is currently runtime-broken and untested.

### Copy-ready prompt
> Rewrite `src/isrs_scl/optimization/statistics.py` against the current `AdaptiveLaunchOptimizer.optimize(initial_profile_dbm)` API and the actual `OptimizationResult` fields. Remove unsupported arguments and nonexistent attributes. Implement multi-seed optimization summaries using `initial_metrics` and `optimized_metrics`, including objective, AIR, thresholded net capacity, working fraction, minimum NGMI, per-band feasibility, selected restart, and acceptance reason. Compute paired bootstrap or percentile confidence intervals only when the sample size is sufficient, and report sample size and method. Integrate optional robust holdout results. Add a compatibility test that instantiates the current optimizer and runs at least two seeds on a small grid. If this functionality is not used by the publication pipeline, delete the module instead of leaving dead incompatible code.

### Expected result
Multi-seed statistics execute successfully and are integrated, or the dead module is removed.

---

## 7. `src/isrs_scl/validation/uncertainty.py`

### Problem
The uncertainty model assumes independent Gaussian perturbations, clips tails ad hoc, uses only 64 samples, and evaluates nominally optimized profiles without a separate robust-training/holdout design.

### Copy-ready prompt
> Upgrade `src/isrs_scl/validation/uncertainty.py` to support traceable parameter distributions, bounds, and correlations derived from calibration sources. Permit normal, lognormal, uniform, triangular, and empirical distributions. Use physically meaningful transformations rather than clipping all scale parameters at 0.25. Support an input correlation matrix and validate positive semidefiniteness. Separate robust-optimization training samples from independent publication holdout samples with different seeds and hashes. Use paired scenario evaluation for flat, fixed, and adaptive profiles. Report failure reasons by parameter/scenario, effective sample size, convergence of quantiles versus sample count, bootstrap confidence intervals for paired gains, probability of adaptive improvement, probability of per-band feasibility, and worst-case/CVaR metrics. Make the minimum final holdout sample count configurable and substantially larger than four; publication defaults should be at least hundreds when computationally feasible. Preserve raw draws and transformed parameter values in exported tables.

### Expected result
Uncertainty conclusions are reproducible, physically interpretable, and independent of optimizer training.

---

## 8. `src/isrs_scl/validation/external_validation.py`

### Problem
Strategy matching is case-sensitive; the original template used `Flat` while model outputs use `flat`. The configured relative NLI threshold is passed as an absolute watt RMSE threshold. Provenance, uncertainty weighting, source diversity, and coverage structure are too weak.

### Copy-ready prompt
> Correct and extend `src/isrs_scl/validation/external_validation.py`. Normalize strategy, source type, metric, and band labels case-insensitively and map aliases to canonical values. Validate source IDs, tool/version, configuration hash, date, provenance reference, span count, wavelength, strategy, metric units, uncertainty, and independence from the model under test. Replace the incorrect use of a relative NLI threshold as absolute-watt RMSE: implement separate absolute RMSE, normalized RMSE, relative RMSE, bias, maximum error, and uncertainty-normalized residual limits with explicit units. Support interpolation only when allowed and record interpolation distance; otherwise use nearest-channel matching. Compute metrics by source, source type, band, span count, strategy, and overall. Require configured minimum coverage across all S/C/L bands and multiple distances, not merely overall row coverage. Add uncertainty-weighted chi-square or standardized residual diagnostics when reference uncertainty is available. Reject blank placeholder rows and duplicate reference identities. Export a clear pass/fail reason for each requirement.

### Expected result
Valid independent data match reliably, and incorrect/placeholder validation cannot pass.

---

## 9. `src/isrs_scl/validation/publication_gate.py`

### Problem
The gate checks file existence rather than current-run provenance, does not require figures/manifests/external comparisons in its artifact list, and some checks are too easy to satisfy or have insufficient statistical requirements.

### Copy-ready prompt
> Strengthen `src/isrs_scl/validation/publication_gate.py`. Require a calibrated configuration with structured traceable sources and matching data hashes. Verify that every required artifact is listed in the current run manifest, exists under the current run directory, has the expected configuration/run ID, and matches its SHA-256 hash. Require resolved config, Git revision, clean-state policy, external-comparison tables, uncertainty holdout tables, optimizer multiseed results, waveform confidence intervals, and all final figures.
>
> Replace single-point adaptive checks with multi-span, per-band, and statistical checks. Require minimum S/C/L working fractions, positive paired robust-gain lower confidence bound, minimum probability of improvement, sufficient uncertainty holdout size, sufficient optimizer seeds, and no data leakage between robust training and holdout. For waveform evidence, require acquisition success at intended operating points, disjoint training/evaluation symbols, confidence intervals, and calibrated prediction intervals; treat intentionally below-threshold diagnostics separately. For external validation, require multiple independent sources or explicitly configured source diversity, minimum bands/distances/strategies, and all metric-specific thresholds. Ensure every failed required check appears in the final summary with numerical evidence.

### Expected result
A passed gate means the exact current evidence package meets strong, machine-verifiable publication criteria.

---

## 10. `src/isrs_scl/validation/reproducibility.py`

### Problem
Manifests store absolute Windows paths, `strict_git_clean` is not enforced, stale files can be collected, and verification is not portable.

### Copy-ready prompt
> Make `src/isrs_scl/validation/reproducibility.py` portable and run-isolated. Store only POSIX-style paths relative to the run root, never absolute local paths or the Python executable path unless placed in a separate environment diagnostic section. Add repository URL, commit SHA, branch/tag, dirty state, resolved configuration hash, input-data hashes, calibration-data hashes, code-package lock hash, command line, run mode, and stage completion status. Enforce `strict_git_clean` in publication mode and fail when Git provenance is unavailable unless explicitly allowed for debugging. Collect artifacts only from the current run directory and reject symlinks or paths escaping it. Include artifact role, MIME/extension, size, and hash. Make `verify_manifest` resolve paths relative to the manifest location and verify config/run IDs embedded in JSON/CSV metadata where applicable. Add schema versioning and backward-compatible manifest migration.

### Expected result
Another researcher can verify the package on another computer without the original drive paths.

---

## 11. `src/isrs_scl/dsp/receiver.py`

### Problem
Low-SNR S-band acquisition collapses while C/L agree. The receiver uses blind phase search after a pilot-aided equalizer, normalizes power without recording the scaling, and aligns using the complete transmitted sequence, which mixes acquisition and evaluation.

### Copy-ready prompt
> Refactor `src/isrs_scl/dsp/receiver.py` into explicit acquisition, training, and held-out evaluation stages. Preserve and report all power-normalization factors. Use only configured pilot/training symbols to estimate timing, polarization mapping, delay, complex gain, and phase; compute final BER/EVM/SNR/GMI on a disjoint payload set. Add a pilot-aided carrier-phase option with cycle-slip detection and keep blind phase search as a separately reported receiver mode. Return acquisition success/failure, cycle-slip count, training and payload symbol counts, estimated delay, polarization permutation, timing phase, gain, and noise diagnostics. Do not use the entire payload to optimize alignment. Accept sample-domain noise variance or PSD through a clearly defined interface from a dedicated noise-conversion module. Add consistency checks that injected ASE+NLI variance at matched-filter decisions agrees with the requested analytical variance within tolerance. Keep the representative-channel limitation explicit.

### Expected result
S-band failures can be distinguished between physical low GSNR, receiver acquisition failure, and noise-scaling error, with unbiased payload metrics.

---

## 12. `src/isrs_scl/dsp/carrier_recovery.py`

### Problem
Blind phase search can fail or cycle-slip at low SNR, but the current function provides no confidence, failure flag, or pilot-aided alternative.

### Copy-ready prompt
> Extend `src/isrs_scl/dsp/carrier_recovery.py` with two well-tested modes: blind phase search and pilot-aided phase recovery. For BPS, implement phase unwrapping/cycle-slip detection, configurable smoothing, block overlap, and a reliability metric based on the cost separation between the best and next-best trial phases. For pilot-aided mode, estimate phase from known pilots, interpolate across payload symbols, and avoid using payload decisions for final metric estimation. Return a typed result containing corrected symbols, phase trace, cycle slips, reliability, and success status. Add deterministic tests across a sweep of SNR, linewidth, block size, and phase offsets, including expected graceful failure below acquisition threshold.

### Expected result
Carrier recovery no longer silently converts an acquisition failure into misleading waveform metrics.

---

## 13. `src/isrs_scl/dsp/equalizer.py`

### Problem
The pilot-aided equalizer and CMA are not evaluated with a strict training/payload split, and convergence/acquisition diagnostics are limited.

### Copy-ready prompt
> Update `src/isrs_scl/dsp/equalizer.py` so both CMA and pilot-aided equalization explicitly separate training from held-out payload evaluation. Return convergence status, condition number, regularization diagnostics, training MSE, payload residual statistics, and tap-energy sanity checks. For pilot-aided least squares, prevent accidental use of payload targets. For CMA, add divergence detection, step-size normalization, and optional CMA-to-decision-directed transition that is disabled for unbiased validation unless explicitly configured. Add tests for polarization swaps, DGD, noise, insufficient training, ill-conditioning, and low-SNR acquisition failure.

### Expected result
Equalizer success is measurable and cannot be confused with model accuracy.

---

## 14. `src/isrs_scl/dsp/metrics.py`

### Problem
Metric estimation uses a fitted complex scalar on the same payload being evaluated, confidence intervals are incomplete, and there is no explicit acquisition-failure state.

### Copy-ready prompt
> Improve `src/isrs_scl/dsp/metrics.py` for publication-grade held-out evaluation. Accept calibration/alignment parameters estimated on a training set and apply them unchanged to payload data. Add an option that forbids fitting gain/phase on the evaluated payload. Report effective SNR, EVM, BER, SER, GMI, NGMI, Q, sample counts, error counts, Wilson intervals, block-bootstrap intervals, and acquisition status. Add minimum-error-count guidance and mark zero-error BER as an upper bound rather than zero certainty. Validate GMI/LLR calculations under mismatched noise and low SNR. Add paired metric comparison helpers for analytical versus waveform predictions and tests across a dense AWGN SNR sweep.

### Expected result
Reported waveform metrics have defensible uncertainty and no training-on-test bias.

---

## 15. `src/isrs_scl/dsp/transmitter.py`

### Problem
The power convention and pilot structure are implicit.

### Copy-ready prompt
> Extend `src/isrs_scl/dsp/transmitter.py` with an explicit dual-polarization power convention and deterministic pilot/payload framing. Add configurable pilot positions or preamble, return pilot and payload masks, and verify that the matched-filter symbol-decision power equals the requested total channel power within numerical tolerance. Record waveform average power, symbol-decision power, pulse energy, sample rate, occupied bandwidth, and normalization factors. Add tests for multiple oversampling factors, roll-off values, symbol counts, and channel powers.

### Expected result
Noise and launch-power calculations can be traced consistently from the power-domain model to waveform samples.

---

## 16. `src/isrs_scl/fiber/amplification.py`

### Problem
ASE is exported as integrated channel power using a configured noise-bandwidth multiplier, but waveform injection needs an explicit conversion between optical PSD, receiver-equivalent noise bandwidth, sample rate, and matched-filter decision variance.

### Copy-ready prompt
> Clarify and harden noise-bandwidth accounting in `src/isrs_scl/fiber/amplification.py`. Define and document separate quantities for dual-polarization ASE PSD, optical integration bandwidth, reference 0.1-nm bandwidth, receiver equivalent noise bandwidth, and expected matched-filter decision variance. Avoid hiding these distinctions inside one `noise_bandwidth_multiplier`. Return all relevant bandwidths and powers in the amplifier result. Add functions with explicit units to convert PSD to integrated power and to decision-domain variance. Validate formulas near unity gain and at S/C/L band edges. Add analytical unit tests and cross-check that OSNR/GSNR calculations remain unchanged under equivalent representations.

### Expected result
ASE values cannot be misused as arbitrary sample-domain variance.

---

## 17. `src/isrs_scl/link.py`

### Problem
The link result lacks explicit noise-bandwidth metadata and uses analytical GSNR directly for AWGN GMI without incorporating the measured receiver calibration curve into threshold/reach calculations.

### Copy-ready prompt
> Extend `src/isrs_scl/link.py` with an explicit noise-budget object containing ASE PSD, optical integrated ASE, receiver-equivalent ASE, NLI, transceiver noise, bandwidth definitions, and units. Keep raw physical GSNR, but add calibrated receiver-predicted SNR/NGMI through a separately supplied monotone B2B calibration model; do not silently mix calibrated and uncalibrated metrics. Reach/capacity summaries used for publication must state which metric is used. Add per-band summaries and validation that no unsupported extrapolation occurs. Preserve backward-compatible raw fields where possible and add unit tests showing consistency among PSD, channel power, GSNR, OSNR, and waveform decision variance.

### Expected result
Analytical and waveform layers use the same explicitly defined noise and receiver conventions.

---

## 18. `src/isrs_scl/visualization/publication_plots.py`

### Problem
Several plots show means or nominal curves without confidence intervals, source counts, failed acquisitions, or model-validity regions. Old duplicate figures can remain in the shared folder.

### Copy-ready prompt
> Upgrade `src/isrs_scl/visualization/publication_plots.py` so every publication plot is generated from the current run tables and includes appropriate uncertainty. Add confidence bands/error bars for optimizer seeds, waveform repeats, external validation, and uncertainty holdout. Mark failed receiver acquisitions separately rather than averaging them away. Show FEC thresholds, per-band working fractions, model-validity warnings, and source/tool labels where relevant. Add paired adaptive-minus-fixed gain plots with confidence intervals and probability of improvement. Use stable unique figure IDs/names from a central registry and fail on duplicate output names. Include figure metadata in PDF/PNG where feasible and export a figure-data CSV reference. Never read prior files implicitly from shared output directories.

### Expected result
Figures communicate statistical strength, limitations, and provenance rather than only nominal performance.

---

## 19. `experiments/adaptive_optimization.py`

### Problem
The script hardcodes `config.yaml`, writes into shared folders, and runs nominal optimization without publication checks.

### Copy-ready prompt
> Refactor `experiments/adaptive_optimization.py` into a thin tested CLI wrapper around the shared publication pipeline stage. Accept `--config`, `--run-id`, `--output-root`, `--mode`, and robust-training options. Default to `config_q2_final.yaml`. Never write into an existing run directory without `--overwrite`. Run multiseed/robust optimization, export nominal and robust metrics, and write a stage manifest. Clearly label standalone output as partial evidence until the full publication gate passes.

### Expected result
Standalone optimization is reproducible and cannot overwrite or masquerade as the complete study.

---

## 20. `experiments/baseline_flat_launch.py`

### Copy-ready prompt
> Refactor `experiments/baseline_flat_launch.py` to use the shared validated configuration loader and run-directory manager. Accept CLI paths and mode, default to `config_q2_final.yaml`, export the resolved config and stage manifest, include calibrated/raw metrics and per-band summaries, and refuse stale output reuse. Remove all hardcoded `config.yaml` and shared result paths.

### Expected result
The flat baseline is generated under the same assumptions as the final study.

---

## 21. `experiments/fixed_preemphasis.py`

### Copy-ready prompt
> Refactor `experiments/fixed_preemphasis.py` to use the shared stage runner, final configuration, run isolation, provenance, per-band metrics, calibrated/raw outputs, and manifest. Validate total launch-power conservation and bounds before simulation. Clearly identify the fixed profile definition and parameters in exported metadata.

### Expected result
The fixed baseline is directly reproducible and comparable with adaptive results.

---

## 22. `experiments/dsp_waveform_analysis.py`

### Copy-ready prompt
> Rewrite `experiments/dsp_waveform_analysis.py` as a safe wrapper for the new waveform-validation stage. It must load a profile and matching power-domain result from the same run/config hash, reject mismatched or stale inputs, run multiple operating points and seeds per band, use pilot/payload separation, export acquisition diagnostics and confidence intervals, and write a stage manifest. Default to `config_q2_final.yaml`; do not claim full-grid SSFM.

### Expected result
Waveform results are linked to the exact analytical run and include defensible diagnostics.

---

## 23. `experiments/sensitivity_analysis.py`

### Copy-ready prompt
> Refactor `experiments/sensitivity_analysis.py` to distinguish local one-factor sensitivity, global uncertainty sensitivity, and robust holdout analysis. Use the shared configuration/run manager, default to `config_q2_final.yaml`, export raw samples and convergence diagnostics, and avoid reusing the nominal optimizer result without hash verification. Add options for sample count and seed, but enforce publication minimums in publication mode.

### Expected result
Sensitivity results are statistically meaningful and tied to the correct run.

---

## 24. `tests/test_waveform.py`

### Problem
Only one easy high-SNR back-to-back case is tested.

### Copy-ready prompt
> Expand `tests/test_waveform.py` into a parameterized suite covering SNR from below acquisition threshold to high SNR, all receiver modes, pilot/payload separation, CD, DGD/polarization rotation, phase noise, carrier offset, ASE/NLI variance conversion, and deterministic seeds. Assert analytical-versus-measured SNR agreement with confidence-aware tolerances when acquisition succeeds, and assert explicit failure status when acquisition should fail. Test that evaluation metrics do not change when held-out payload labels are unavailable to the training stage.

### Expected result
The exact failure seen in S-band waveform consistency becomes reproducible and preventable.

---

## 25. `tests/test_optimizer.py`

### Problem
It tests only power projection, not the optimizer.

### Copy-ready prompt
> Expand `tests/test_optimizer.py` to run the actual optimizer on a small deterministic link. Test total-power conservation, channel bounds, reproducibility, accepted/rejected candidates, per-band constraints, multi-span constraints, multiseed statistics, robust training versus independent holdout, and positive/negative gain cases. Add a regression test ensuring the optimizer cannot improve overall mean performance by sacrificing S band below the configured working-fraction constraint.

### Expected result
Optimizer claims are protected by behavioral tests, not only projection math.

---

# B. New Python files to add

## 26. `src/isrs_scl/validation/calibration.py`

### Copy-ready prompt
> Create `src/isrs_scl/validation/calibration.py`. Implement typed loading, validation, interpolation, and provenance tracking for measured/vendor/literature fiber attenuation, Raman gain spectra, pump parameters, amplifier gain/NF, transceiver SNR, and B2B receiver calibration. Require units and uncertainty for every parameter. Hash every source file, reject unsupported extrapolation by default, and return a resolved calibrated parameter bundle plus a detailed provenance table. Support combining multiple sources only with an explicit rule. Include functions that determine whether a configuration may truthfully be marked `CALIBRATED`.

### Expected result
Calibration status is evidence-driven rather than a manually edited string.

---

## 27. `src/isrs_scl/dsp/noise.py`

### Copy-ready prompt
> Create `src/isrs_scl/dsp/noise.py`. Implement explicit, unit-tested conversions among dual-polarization PSD in W/Hz, optical integrated noise power, receiver equivalent noise bandwidth, sample-domain complex AWGN variance, and matched-filter decision variance for an RRC waveform. Use typed dataclasses to carry units/metadata. Include Monte Carlo verification that generated noise reaches the requested decision-domain SNR across oversampling and roll-off settings.

### Expected result
Power-domain and waveform-domain noise use one validated convention.

---

## 28. `src/isrs_scl/optimization/robust.py`

### Copy-ready prompt
> Create `src/isrs_scl/optimization/robust.py`. Provide deterministic generation of robust-training scenarios from calibrated uncertainty distributions, common-random-number batches for objective evaluation, CVaR/worst-case aggregators, paired baseline comparisons, scenario hashes, and strict separation from holdout uncertainty samples. Keep the module independent of plotting. Add small-grid tests and expose a clean interface consumed by `AdaptiveLaunchOptimizer`.

### Expected result
Robust optimization is modular, reproducible, and free of holdout leakage.

---

## 29. `tests/test_external_validation.py`

### Copy-ready prompt
> Create `tests/test_external_validation.py` covering case-insensitive strategy matching, alias normalization, blank placeholders, duplicate identities, wavelength tolerance, interpolation policy, missing bands, missing distances, source diversity, absolute versus relative NLI thresholds, uncertainty-weighted residuals, and complete pass/fail reasons.

---

## 30. `tests/test_publication_gate.py`

### Copy-ready prompt
> Create `tests/test_publication_gate.py` with minimal synthetic tables for every gate. Verify that each required failure is independently detected, stale/wrong-run artifacts fail, missing hashes fail, uncalibrated metadata fails, insufficient source diversity fails, S-band sacrifice fails, waveform acquisition failure is handled correctly, robust lower-bound failure fails, and a fully valid synthetic evidence package passes.

---

## 31. `tests/test_reproducibility.py`

### Copy-ready prompt
> Create `tests/test_reproducibility.py` to verify portable relative paths, artifact hashing, run-root confinement, dirty/unavailable Git policy, manifest migration, tamper detection, source-data hashes, and successful verification after moving a run directory to a different absolute location.

---

## 32. `tests/test_waveform_snr_sweep.py`

### Copy-ready prompt
> Create `tests/test_waveform_snr_sweep.py` to run deterministic AWGN and representative-channel sweeps across the full B2B calibration range. Validate monotonic SNR/BER/GMI behavior, receiver acquisition threshold, phase-recovery reliability, confidence intervals, and analytical-versus-waveform agreement after noise conversion.

---

## 33. `tests/test_robust_optimizer.py`

### Copy-ready prompt
> Create `tests/test_robust_optimizer.py` with a lightweight mock or small optical grid. Verify scenario separation, deterministic hashes, CVaR calculations, common-random-number behavior, per-band constraints, positive paired gain acceptance, negative lower-bound rejection, and no holdout data used during optimization.

---

## 34. `tests/test_config_schema.py`

### Copy-ready prompt
> Create `tests/test_config_schema.py` to test the strict configuration schema, unknown keys, units/ranges, publication/debug differences, calibration provenance, B2B sweep coverage, robust/holdout seed separation, path requirements, old-config migration, and resolved-config hashing.

---

## 35. `tests/test_publication_pipeline_smoke.py`

### Copy-ready prompt
> Create `tests/test_publication_pipeline_smoke.py` that runs a reduced study in a temporary directory with synthetic calibrated/external data. Verify isolated outputs, no stale-file reuse, manifest/hash integrity, all expected tables/figures, deterministic reruns, strict failure when one required input is removed, and successful completion only when every synthetic gate condition is met.

---

## 36. `tools/clean_artifacts.py`

### Copy-ready prompt
> Create `tools/clean_artifacts.py` to remove generated results, figures, bytecode, caches, and package metadata safely. It must operate only inside the repository root, support `--dry-run`, preserve tracked source/input files, and refuse suspicious paths. Add an option to retain one explicitly named release evidence directory.

### Expected result
Developers can clean stale artifacts without risking source or calibration data.

---

# C. Non-Python files that must be changed or added

## Must change

- `config_q2_final.yaml`
  - Make this the only final publication configuration.
  - Keep `calibration_status: UNVALIDATED_DEFAULTS` until real sources are loaded.
  - Add structured calibration source records, robust-training/holdout settings, fine B2B calibration sweep, run isolation, and stricter source coverage requirements.

- `config.yaml`
  - Rename to `config_example.yaml` or `config_smoke.yaml`.
  - Do not leave it as the default final-run configuration.

- `README.md`
  - Update all commands to use `config_q2_final.yaml`.
  - Document publication/debug distinctions, required external data schema, calibration provenance, run directories, and exact limitations.
  - Remove output names that are no longer generated or make names match the actual pipeline.

- `pyproject.toml`
  - Add test coverage threshold, stricter Ruff settings, typing/static checking if adopted, and package-data rules.
  - Add reproducibility dependencies only when justified.
  - Do not commit generated `.egg-info`.

- Add `.gitignore`
  - Include `__pycache__/`, `*.py[cod]`, `.pytest_cache/`, `.ruff_cache/`, `.mypy_cache/`, `.coverage*`, `htmlcov/`, `.venv/`, `build/`, `dist/`, `*.egg-info/`, temporary run outputs, and local large data.

- Add `.github/workflows/ci.yml`
  - Run lint, tests, coverage, config validation, and a reduced pipeline on supported Python versions.
  - Do not run expensive final simulations in ordinary CI.

- `validation_data/external_reference.csv`
  - Replace blank values with real independently generated/measured values.
  - Add complete source/tool/version/config/provenance/uncertainty fields.
  - Include multiple wavelengths in S/C/L, multiple span counts, and relevant strategies.

## Delete or remove from normal Git tracking

- Every `__pycache__/` directory and every `*.pyc`.
- `src/isrs_scl_q2.egg-info/`.
- Stale generated contents under `results/publication/` and `figures/`.
  - Regenerate them into run-ID directories.
  - Publish one frozen evidence archive through a release/Zenodo-style archive rather than mixing successive runs in the source tree.
- Duplicate placeholder file `data/raw/external_validation_template.csv` after consolidating the schema under `validation_data/`.
- Any duplicate/obsolete figures with conflicting numeric prefixes.
- Do **not** delete core physics source files merely to force the gate to pass.
- Rewrite `src/isrs_scl/optimization/statistics.py`; delete it only if multiseed statistics are intentionally removed from the project.

---

# D. Recommended implementation order

1. Repository cleanup, `.gitignore`, run isolation, and manifest portability.
2. Strict configuration schema and calibration-data module.
3. External-validation normalization and real data ingestion.
4. Waveform noise convention, pilot/payload split, and low-SNR receiver diagnostics.
5. Band-aware robust optimizer and independent uncertainty holdout.
6. Strong publication gate.
7. Plot/statistical upgrades.
8. Expanded tests and CI.
9. Regenerate all results from a clean tagged commit.
10. Write the manuscript only from the final run tables and archived evidence package.

---

# E. Final acceptance criteria before writing/submitting the paper

- `VALIDATION_STATUS.json` passes every required check.
- Calibration sources are real, traceable, hashed, and uncertainty-annotated.
- Independent GNPy/SSFM/experimental comparisons meet preregistered thresholds across S/C/L and multiple distances.
- S-band performance is either genuinely improved to the advertised target or the paper’s target distance/scope is revised honestly.
- Waveform and power-domain predictions agree within confidence-aware tolerance at intended operating points.
- Adaptive-minus-fixed robust gain has a positive lower confidence bound on an independent holdout set.
- Multiple optimizer seeds show stable conclusions.
- All figures and tables are generated from one clean commit, one resolved configuration, and one run ID.
- The manifest verifies successfully after the evidence directory is moved to another computer.
- No placeholders, bytecode, stale duplicate outputs, or generated package metadata remain in source control.
