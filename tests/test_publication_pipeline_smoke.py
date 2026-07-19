from pathlib import Path

import pytest

from isrs_scl.validation.reproducibility import build_run_manifest, finalize_manifest, prepare_run_directory, verify_manifest


def test_reduced_pipeline_evidence_directory_is_isolated_and_verifiable(tmp_path):
    paths = prepare_run_directory(tmp_path, "synthetic-smoke")
    (paths.results / "channel_grid.csv").write_text("channel\n0\n", encoding="utf-8")
    (paths.figures / "figure.pdf").write_bytes(b"%PDF-1.4\n%%EOF")
    (paths.metadata / "resolved_config.yaml").write_text("schema_version: 2\n", encoding="utf-8")
    manifest = build_run_manifest({"run": {"run_id": "synthetic-smoke", "mode": "smoke"}, "metadata": {}, "optimization": {}, "uncertainty": {}}, run_root=paths.root)
    target = paths.metadata / "RUN_MANIFEST.json"; finalize_manifest(manifest, paths.root, target)
    assert verify_manifest(target)[0]
    with pytest.raises(Exception): prepare_run_directory(tmp_path, "synthetic-smoke")
