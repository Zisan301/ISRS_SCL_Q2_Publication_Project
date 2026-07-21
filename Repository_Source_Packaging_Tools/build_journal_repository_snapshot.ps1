# build_journal_repository_snapshot.ps1
# Run this script from the ROOT of your cleaned local GitHub repository:
# Example:
#   cd "E:\VS Code\ISRS_S-C-L"
#   powershell -ExecutionPolicy Bypass -File ..\build_journal_repository_snapshot.ps1
#
# It creates a clean source-code ZIP for journal supplementary/reproducibility review.
# It copies only files that are useful for reproducing the final system/paper evidence.

$ErrorActionPreference = "Stop"

$Root = Get-Location
$OutRoot = Join-Path $Root "journal_repository_snapshot"
$ZipPath = Join-Path $Root "ISRS_S-C-L_journal_repository_snapshot.zip"

if (Test-Path $OutRoot) { Remove-Item $OutRoot -Recurse -Force }
if (Test-Path $ZipPath) { Remove-Item $ZipPath -Force }
New-Item -ItemType Directory -Path $OutRoot | Out-Null

$IncludePaths = @(
    "README.md",
    "LICENSE",
    "pyproject.toml",
    "requirements.txt",
    "config.yaml",
    "config_q2_final.yaml",
    "config_fast_preview.yaml",
    "config_turbo_preview.yaml",
    "src",
    "scripts",
    "tests",
    "docs",
    "validation_data",
    "external_validation"
)

$ExcludeNames = @(
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "env",
    "gnpy_env",
    "runs",
    "results",
    "paper_package_cband_preliminary_20260720"
)

$ExcludeFiles = @(
    "run_prepare_today_cband_paper_package.bat",
    "run_today_cband_bias_diagnostic.bat",
    "run_today_cband_wavelength_bias_check.bat",
    "cleanup_journal_repo.py"
)

function Should-ExcludePath($Path) {
    foreach ($part in $Path.Split([IO.Path]::DirectorySeparatorChar)) {
        if ($ExcludeNames -contains $part) { return $true }
    }
    foreach ($file in $ExcludeFiles) {
        if ([IO.Path]::GetFileName($Path) -eq $file) { return $true }
    }
    return $false
}

foreach ($rel in $IncludePaths) {
    $src = Join-Path $Root $rel
    $dst = Join-Path $OutRoot $rel

    if (!(Test-Path $src)) {
        Write-Host "SKIP missing: $rel"
        continue
    }

    if ((Get-Item $src).PSIsContainer) {
        Get-ChildItem $src -Recurse -Force | ForEach-Object {
            $full = $_.FullName
            $relChild = $full.Substring($Root.Path.Length + 1)
            if (Should-ExcludePath $relChild) { return }

            $target = Join-Path $OutRoot $relChild
            if ($_.PSIsContainer) {
                if (!(Test-Path $target)) { New-Item -ItemType Directory -Path $target | Out-Null }
            } else {
                $targetDir = Split-Path $target -Parent
                if (!(Test-Path $targetDir)) { New-Item -ItemType Directory -Path $targetDir | Out-Null }
                Copy-Item $full $target -Force
            }
        }
    } else {
        if (Should-ExcludePath $rel) { continue }
        $dstDir = Split-Path $dst -Parent
        if (!(Test-Path $dstDir)) { New-Item -ItemType Directory -Path $dstDir | Out-Null }
        Copy-Item $src $dst -Force
    }
}

# Add reproducibility README inside the snapshot.
@'
# ISRS_S-C-L journal repository snapshot

This ZIP is a clean source-code snapshot intended for journal reproducibility review.

Recommended run/check sequence:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest -q
python -m compileall src scripts
```

Final held-out validation evidence should include:
- validation_data/gnpy_calibration_train_1535_1550_1560.csv
- validation_data/gnpy_holdout_1540_1545_1555.csv
- scripts/run_heldout_external_validation.py

The journal manuscript describes simulation-to-simulation validation against GNPy, not laboratory or field experimental validation.
'@ | Set-Content (Join-Path $OutRoot "README_REPRODUCIBILITY_SNAPSHOT.md")

Compress-Archive -Path (Join-Path $OutRoot "*") -DestinationPath $ZipPath -Force

Write-Host ""
Write-Host "Created clean repository snapshot:"
Write-Host $ZipPath
Write-Host ""
Write-Host "Now review it before giving it to journal/supplementary materials."
