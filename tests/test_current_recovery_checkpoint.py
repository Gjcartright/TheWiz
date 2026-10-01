"""Checks for the optional restore-tested recovery packager."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
from pathlib import Path

import pytest


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts/build_current_recovery_checkpoint.py"
    spec = importlib.util.spec_from_file_location("current_recovery_checkpoint", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _baseline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, clean: bool):
    module = _module()
    active = tmp_path / "reports/active"
    active.mkdir(parents=True)
    manifest = active / "corrective_baseline_manifest.json"
    files = active / "corrective_baseline_files.csv"
    secret_scan = active / "corrective_secret_scan.csv"
    source_manifest = active / "corrective_source_commit_manifest.csv"
    files.write_text("path,sha256\n", encoding="utf-8")
    secret_scan.write_text("path,severity\n", encoding="utf-8")
    source_manifest.write_text("path,git_status\n", encoding="utf-8")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    manifest.write_text(
        json.dumps(
            {
                "secret_blockers": 0,
                "clean_checkout_reproducible": clean,
                "commit_candidate_files": 0,
                "worktree_entries": 0,
                "git_head": "a" * 40,
                "git_branch": "diagnostic",
                "manifest_sha256": digest(source_manifest),
                "baseline_files_sha256": digest(files),
                "secret_scan_sha256": digest(secret_scan),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "BASELINE_MANIFEST", manifest)
    monkeypatch.setattr(module, "BASELINE_FILES", files)
    monkeypatch.setattr(module, "SECRET_SCAN", secret_scan)
    monkeypatch.setattr(module, "SOURCE_COMMIT_MANIFEST", source_manifest)
    monkeypatch.setattr(module, "_current_git_state", lambda: ("a" * 40, "diagnostic", []))
    return module, files, source_manifest, manifest


def test_clean_checkout_can_use_git_bundle_without_changed_source_rows(tmp_path, monkeypatch):
    module, _files, _source_manifest, _manifest = _baseline(tmp_path, monkeypatch, clean=True)
    assert module._load_source_rows() == []


def test_empty_source_rows_require_verified_clean_checkout(tmp_path, monkeypatch):
    module, _files, _source_manifest, _manifest = _baseline(tmp_path, monkeypatch, clean=False)
    with pytest.raises(ValueError, match="verified clean checkout"):
        module._load_source_rows()


def test_source_manifest_rejects_parent_traversal(tmp_path, monkeypatch):
    module, files, source_manifest, manifest = _baseline(tmp_path, monkeypatch, clean=False)
    files.write_text("path,sha256\n../outside,unused\n", encoding="utf-8")
    source_manifest.write_text("path,git_status\n../outside,??\n", encoding="utf-8")
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    metadata["commit_candidate_files"] = 1
    metadata["worktree_entries"] = 1
    metadata["baseline_files_sha256"] = hashlib.sha256(files.read_bytes()).hexdigest()
    metadata["manifest_sha256"] = hashlib.sha256(source_manifest.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(metadata), encoding="utf-8")
    monkeypatch.setattr(module, "_current_git_state", lambda: ("a" * 40, "diagnostic", [("??", "../outside")]))
    with pytest.raises(ValueError, match="forbidden recovery source path"):
        module._load_source_rows()


def test_clean_baseline_rejects_new_worktree_path(tmp_path, monkeypatch):
    module, _files, _source_manifest, _manifest = _baseline(tmp_path, monkeypatch, clean=True)
    monkeypatch.setattr(module, "_current_git_state", lambda: ("a" * 40, "diagnostic", [("??", "late.py")]))
    with pytest.raises(ValueError, match="worktree paths changed"):
        module._load_source_rows()


def test_restore_runtime_preserves_venv_entry_path():
    assert _module()._python_runtime() == Path(sys.executable)
