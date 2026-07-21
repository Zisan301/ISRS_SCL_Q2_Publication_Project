# cleanup_journal_repo_safe.ps1
# Run from the ROOT of your local repo on your journal branch.
# This removes obsolete preliminary files only.

$ErrorActionPreference = "Stop"

$RemovePaths = @(
    "paper_package_cband_preliminary_20260720",
    "run_prepare_today_cband_paper_package.bat",
    "run_today_cband_bias_diagnostic.bat",
    "run_today_cband_wavelength_bias_check.bat",
    "tools\analyze_cband_gnpy_bias.py",
    "tools\fit_cband_wavelength_bias.py",
    "tools\prepare_today_cband_paper_package.py"
)

foreach ($p in $RemovePaths) {
    if (Test-Path $p) {
        Remove-Item $p -Recurse -Force
        Write-Host "REMOVED: $p"
    } else {
        Write-Host "SKIP missing: $p"
    }
}

Write-Host ""
Write-Host "Now run:"
Write-Host "git diff --name-status"
Write-Host ""
Write-Host "Only obsolete preliminary files should appear as deleted."
