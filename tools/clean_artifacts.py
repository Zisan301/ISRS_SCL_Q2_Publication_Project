"""Safely remove generated artifacts and Python caches from this repository."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess

_PATTERNS = ("__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", "htmlcov", "build", "dist")
_SUFFIXES = (".pyc", ".pyo", ".coverage")


def repository_root(start: str | Path = ".") -> Path:
    current = Path(start).resolve()
    try:
        value = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=current, text=True, stderr=subprocess.DEVNULL).strip()
        return Path(value).resolve()
    except Exception:
        if (current / "pyproject.toml").exists(): return current
        raise RuntimeError("Cannot identify repository root")


def candidates(root: Path, retain_release: str | None = None) -> list[Path]:
    result: set[Path] = set()
    for path in root.rglob("*"):
        if path.name in _PATTERNS or path.name.endswith(".egg-info") or (path.is_file() and path.suffix in _SUFFIXES): result.add(path)
    for directory in (root / "results", root / "figures", root / "runs"):
        if not directory.exists(): continue
        for child in directory.iterdir():
            if retain_release and child.name == retain_release: continue
            result.add(child)
    return sorted(result, key=lambda path: (len(path.parts), str(path)), reverse=True)


def safe_remove(path: Path, root: Path, *, dry_run: bool) -> None:
    resolved, root_resolved = path.resolve(), root.resolve()
    if resolved == root_resolved or root_resolved not in resolved.parents:
        raise RuntimeError(f"Refusing suspicious path: {path}")
    if dry_run:
        print(path.relative_to(root)); return
    if path.is_symlink() or path.is_file(): path.unlink(missing_ok=True)
    elif path.is_dir(): shutil.rmtree(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--root", default="."); parser.add_argument("--dry-run", action="store_true"); parser.add_argument("--retain-release"); args = parser.parse_args()
    root = repository_root(args.root)
    for path in candidates(root, args.retain_release): safe_remove(path, root, dry_run=args.dry_run)
    return 0


if __name__ == "__main__": raise SystemExit(main())
