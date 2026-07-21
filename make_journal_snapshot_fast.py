from pathlib import Path
import zipfile

ROOT = Path.cwd()
ZIP_PATH = ROOT / "ISRS_S-C-L_journal_repository_snapshot.zip"

FILES = [
    "README.md",
    "LICENSE",
    "pyproject.toml",
    "config_q2_final.yaml",
    "config_fast_preview.yaml",
    "config_turbo_preview.yaml",
    "validation_data/gnpy_calibration_train_1535_1550_1560.csv",
    "validation_data/gnpy_holdout_1540_1545_1555.csv",
]

DIRS = [
    "src",
    "tests",
]

count = 0

if ZIP_PATH.exists():
    ZIP_PATH.unlink()

with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
    for f in FILES:
        p = ROOT / f
        if p.exists():
            zf.write(p, f.replace("\\", "/"))
            count += 1
        else:
            print(f"SKIP missing: {f}")

    for d in DIRS:
        p = ROOT / d
        if p.exists():
            for child in p.rglob("*"):
                if child.is_file() and "__pycache__" not in child.parts and child.suffix != ".pyc":
                    zf.write(child, child.relative_to(ROOT).as_posix())
                    count += 1
        else:
            print(f"SKIP missing: {d}")

    zf.writestr("README_REPRODUCIBILITY_SNAPSHOT.md", "Clean journal repository snapshot for ISRS-S-C-L held-out GNPy validation paper.\n")
    count += 1

print("DONE")
print(f"Files added: {count}")
print(f"Created: {ZIP_PATH}")
