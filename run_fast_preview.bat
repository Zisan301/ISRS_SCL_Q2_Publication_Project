@echo off
REM Fast draft run: should finish in a few minutes on most laptops.
python run_all_experiments.py --debug --config config_fast_preview.yaml --run-id q2-fast-preview-001 --output-root runs --overwrite
pause
