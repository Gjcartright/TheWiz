from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

from quant_platform.orchestration.corrective_artifact_retention import (
    run_corrective_artifact_retention,
)


def _policy(root: Path, *, minimum: int = 1, threshold: int = 100) -> None:
    path = root / "config" / "corrective_artifact_retention.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "thewiz.corrective_artifact_retention_policy.v1",
                "operational_receipt_directories": ["data/research/operational"],
                "receipt_retention_days": 14,
                "minimum_newest_per_directory": minimum,
                "log_paths": ["reports/active/schedule_logs/*.log"],
                "log_rotation_threshold_bytes": threshold,
                "log_rotations_to_keep": 2,
                "never_archive_directories": ["data/research/scientific"],
            }
        ),
        encoding="utf-8",
    )


def _old(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(path.name, encoding="utf-8")
    old = datetime(2026, 7, 1, tzinfo=UTC).timestamp()
    os.utime(path, (old, old))


def test_retention_dry_run_moves_nothing_and_protects_active_reference(tmp_path) -> None:
    _policy(tmp_path)
    referenced = tmp_path / "data/research/operational/referenced.json"
    candidate = tmp_path / "data/research/operational/candidate.json"
    newest = tmp_path / "data/research/operational/newest.json"
    _old(referenced)
    _old(candidate)
    newest.write_text("new", encoding="utf-8")
    active = tmp_path / "reports" / "active" / "pointer.json"
    active.parent.mkdir(parents=True)
    active.write_text(json.dumps({"path": "data/research/operational/referenced.json"}))

    result = run_corrective_artifact_retention(root=tmp_path, now=datetime(2026, 8, 14, tzinfo=UTC))

    assert result.summary["status"] == "PASS_RETENTION_DRY_RUN"
    assert result.summary["dry_run_moved_nothing"] is True
    assert referenced.exists() and candidate.exists() and newest.exists()
    rows = {row["source_path"]: row for row in result.summary["receipt_rows"]}
    assert rows["data/research/operational/referenced.json"]["eligible"] is False
    assert rows["data/research/operational/candidate.json"]["action"] == "would_archive"


def test_retention_apply_archives_allowlisted_candidate_and_rotates_large_log(tmp_path) -> None:
    _policy(tmp_path, threshold=10)
    candidate = tmp_path / "data/research/operational/candidate.json"
    newest = tmp_path / "data/research/operational/newest.json"
    _old(candidate)
    newest.parent.mkdir(parents=True, exist_ok=True)
    newest.write_text("new", encoding="utf-8")
    log = tmp_path / "reports/active/schedule_logs/runner.log"
    log.parent.mkdir(parents=True)
    log.write_text("x" * 20, encoding="utf-8")

    result = run_corrective_artifact_retention(
        root=tmp_path,
        now=datetime(2026, 8, 14, tzinfo=UTC),
        apply=True,
    )

    assert not candidate.exists()
    assert result.summary["files_archived"] == 1
    assert result.paths["archive_manifest"].is_file()
    assert log.read_text(encoding="utf-8") == ""
    assert log.with_name("runner.log.1").is_file()
    assert result.summary["scientific_evidence_directories_touched"] == []
