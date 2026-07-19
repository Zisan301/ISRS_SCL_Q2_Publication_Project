import json
import shutil
from pathlib import Path

import pytest

from isrs_scl.validation.reproducibility import (
    ProvenanceError, build_run_manifest, collect_artifacts, finalize_manifest,
    prepare_run_directory, verify_manifest,
)


def test_manifest_is_portable_and_detects_tampering(tmp_path):
    paths = prepare_run_directory(tmp_path, "run-1")
    artifact = paths.results / "data.csv"; artifact.write_text("x\n1\n", encoding="utf-8")
    manifest = build_run_manifest({"run": {"run_id": "run-1", "mode": "debug"}, "metadata": {}, "optimization": {}, "uncertainty": {}}, run_root=paths.root)
    target = paths.metadata / "RUN_MANIFEST.json"; finalize_manifest(manifest, paths.root, target)
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert all(not Path(item["path"]).is_absolute() for item in payload["artifacts"])
    moved = tmp_path / "moved"; shutil.copytree(paths.root, moved)
    assert verify_manifest(moved / "metadata" / "RUN_MANIFEST.json")[0]
    (moved / "results" / "data.csv").write_text("tampered", encoding="utf-8")
    ok, errors = verify_manifest(moved / "metadata" / "RUN_MANIFEST.json")
    assert not ok and any("checksum" in error for error in errors)


def test_run_directory_refuses_stale_reuse(tmp_path):
    prepare_run_directory(tmp_path, "run-1")
    with pytest.raises(ProvenanceError):
        prepare_run_directory(tmp_path, "run-1")
