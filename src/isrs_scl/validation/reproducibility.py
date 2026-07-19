"""Portable provenance, run isolation, and artifact-integrity utilities."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import hashlib
import importlib.metadata
import json
import mimetypes
import os
import platform
import shlex
import subprocess
import sys
import tempfile

import numpy as np

MANIFEST_SCHEMA_VERSION = 2
_PACKAGE_NAMES = ("numpy", "scipy", "pandas", "matplotlib", "PyYAML")


class ProvenanceError(RuntimeError):
    """Raised when publication provenance cannot be established safely."""


@dataclass(frozen=True)
class RunPaths:
    root: Path
    results: Path
    figures: Path
    metadata: Path


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, default=_json_default, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def stable_hash(value: Any, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    digest.update(canonical_json_bytes(value))
    return digest.hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    target = Path(path)
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, default=_json_default, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def prepare_run_directory(output_root: str | Path, run_id: str, *, overwrite: bool = False) -> RunPaths:
    if not run_id or any(part in run_id for part in ("..", "/", "\\")):
        raise ProvenanceError("run_id must be a simple non-empty identifier")
    root = (Path(output_root) / run_id).resolve()
    if root.exists() and any(root.iterdir()) and not overwrite:
        raise ProvenanceError(f"Run directory is not empty: {root}")
    if root.exists() and overwrite:
        for path in sorted(root.rglob("*"), reverse=True):
            if path.is_symlink():
                path.unlink()
            elif path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
    results, figures, metadata = root / "results", root / "figures", root / "metadata"
    for directory in (results, figures, metadata):
        directory.mkdir(parents=True, exist_ok=True)
    return RunPaths(root, results, figures, metadata)


def _run_git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, stderr=subprocess.DEVNULL, text=True, timeout=10).strip()


def git_revision(repository_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(repository_root or Path.cwd()).resolve()
    try:
        commit = _run_git(root, "rev-parse", "HEAD")
        branch = _run_git(root, "rev-parse", "--abbrev-ref", "HEAD")
        dirty = bool(_run_git(root, "status", "--porcelain"))
        try:
            remote = _run_git(root, "config", "--get", "remote.origin.url")
        except subprocess.SubprocessError:
            remote = None
        try:
            tag = _run_git(root, "describe", "--tags", "--exact-match")
        except subprocess.SubprocessError:
            tag = None
        return {"available": True, "commit": commit, "branch": branch, "tag": tag, "dirty": dirty, "repository_url": remote}
    except (OSError, subprocess.SubprocessError):
        return {"available": False, "commit": None, "branch": None, "tag": None, "dirty": None, "repository_url": None}


def enforce_git_policy(git: Mapping[str, Any], *, publication: bool, allow_untracked: bool = False) -> None:
    if not publication:
        return
    if not bool(git.get("available")) and not allow_untracked:
        raise ProvenanceError("Git provenance is unavailable in publication mode")
    if bool(git.get("dirty")) and not allow_untracked:
        raise ProvenanceError("The Git working tree is dirty in publication mode")


def environment_snapshot() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in _PACKAGE_NAMES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "utc_timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": packages,
    }


def seed_ledger(cfg: Mapping[str, Any]) -> dict[str, int]:
    base = int(cfg.get("metadata", {}).get("random_seed", 0))
    return {
        "base": base,
        "back_to_back": base + 10_000,
        "waveform": base + 20_000,
        "optimizer": int(cfg.get("optimization", {}).get("seed", base + 30_000)),
        "robust_training": int(cfg.get("optimization", {}).get("robust_training_seed", base + 40_000)),
        "uncertainty_holdout": int(cfg.get("uncertainty", {}).get("holdout_seed", base + 50_000)),
    }


def _hash_existing_files(paths: Iterable[str | Path], base: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in paths:
        path = Path(raw)
        path = path if path.is_absolute() else base / path
        if path.exists() and path.is_file():
            rows.append({"path": path.resolve().relative_to(base.resolve()).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return rows


def build_run_manifest(
    cfg: Mapping[str, Any],
    *,
    run_root: str | Path,
    repository_root: str | Path | None = None,
    command_line: Sequence[str] | None = None,
    input_files: Iterable[str | Path] = (),
    calibration_files: Iterable[str | Path] = (),
) -> dict[str, Any]:
    repo_root = Path(repository_root or Path.cwd()).resolve()
    run_root_path = Path(run_root).resolve()
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": str(cfg.get("run", {}).get("run_id") or stable_hash(cfg)[:16]),
        "run_mode": str(cfg.get("run", {}).get("mode", "debug")),
        "configuration_sha256": stable_hash(cfg),
        "configuration": dict(cfg),
        "git": git_revision(repo_root),
        "environment": environment_snapshot(),
        "command_line": shlex.join(command_line or sys.argv),
        "seeds": seed_ledger(cfg),
        "inputs": _hash_existing_files(input_files, repo_root),
        "calibration_inputs": _hash_existing_files(calibration_files, repo_root),
        "run_root": ".",
        "stage_status": {},
        "artifacts": [],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "_runtime_run_root": str(run_root_path),
    }


def mark_stage(manifest: dict[str, Any], stage: str, *, passed: bool, details: Mapping[str, Any] | None = None) -> None:
    manifest.setdefault("stage_status", {})[stage] = {
        "passed": bool(passed),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "details": dict(details or {}),
    }


def _safe_relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    root_resolved = root.resolve()
    try:
        relative = resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ProvenanceError(f"Artifact escapes run root: {path}") from exc
    if path.is_symlink():
        raise ProvenanceError(f"Symlink artifacts are forbidden: {path}")
    return relative.as_posix()


def collect_artifacts(run_root: str | Path, *, exclude_names: Iterable[str] = ("RUN_MANIFEST.json",)) -> list[dict[str, Any]]:
    root = Path(run_root).resolve()
    excluded = set(exclude_names)
    records: list[dict[str, Any]] = []
    if not root.exists():
        return records
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name in excluded:
            continue
        relative = _safe_relative(path, root)
        mime, _ = mimetypes.guess_type(path.name)
        role = relative.split("/", 1)[0] if "/" in relative else "metadata"
        records.append({"path": relative, "role": role, "extension": path.suffix.lower(), "mime_type": mime or "application/octet-stream", "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return records


def finalize_manifest(manifest: dict[str, Any], run_root: str | Path, output_path: str | Path) -> dict[str, Any]:
    finalized = {key: value for key, value in manifest.items() if not key.startswith("_runtime_")}
    finalized["artifacts"] = collect_artifacts(run_root)
    finalized["artifact_count"] = len(finalized["artifacts"])
    finalized["completed_utc"] = datetime.now(timezone.utc).isoformat()
    finalized["manifest_content_sha256"] = stable_hash({key: value for key, value in finalized.items() if key != "manifest_content_sha256"})
    atomic_write_json(output_path, finalized)
    return finalized


def migrate_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    version = int(result.get("schema_version", 1))
    if version == MANIFEST_SCHEMA_VERSION:
        return result
    if version != 1:
        raise ProvenanceError(f"Unsupported manifest schema {version}")
    for item in result.get("artifacts", []):
        path = Path(str(item.get("relative_to_root") or item.get("path", "")))
        item["path"] = path.name if path.is_absolute() else path.as_posix()
        item.pop("relative_to_root", None)
    result["schema_version"] = MANIFEST_SCHEMA_VERSION
    result.setdefault("stage_status", {})
    result.setdefault("run_root", ".")
    return result


def verify_manifest(path: str | Path) -> tuple[bool, list[str]]:
    target = Path(path).resolve()
    payload = migrate_manifest(json.loads(target.read_text(encoding="utf-8")))
    root = target.parent.parent if target.parent.name == "metadata" else target.parent
    errors: list[str] = []
    expected_content = payload.get("manifest_content_sha256")
    if expected_content:
        actual_content = stable_hash({key: value for key, value in payload.items() if key != "manifest_content_sha256"})
        if actual_content != expected_content:
            errors.append("manifest content hash mismatch")
    for item in payload.get("artifacts", []):
        artifact = (root / item["path"]).resolve()
        try:
            artifact.relative_to(root.resolve())
        except ValueError:
            errors.append(f"path escapes run root: {item['path']}")
            continue
        if not artifact.exists():
            errors.append(f"missing: {item['path']}")
            continue
        if sha256_file(artifact) != item.get("sha256"):
            errors.append(f"checksum mismatch: {item['path']}")
    return not errors, errors
