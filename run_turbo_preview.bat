@echo off
REM Very fast draft run. Uncertainty disabled and coarser grid. Not journal evidence.
python run_all_experiments.py --debug --config config_turbo_preview.yaml --run-id q2-turbo-preview-001 --output-root runs --overwrite --no-uncertainty
pause
