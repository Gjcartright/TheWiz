from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from quant_platform.orchestration import (
    current_wizard_hyperliquid_archive_release as release_module,
)
from quant_platform.orchestration import current_wizard_hyperliquid_storage as storage
from quant_platform.orchestration.current_wizard_hyperliquid_archive import (
    stage_current_wizard_hyperliquid_archive_copy,
)
from quant_platform.orchestration.current_wizard_hyperliquid_archive_release import (
    build_current_wizard_hyperliquid_archive_release_dry_run,
)


def _build_verified_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, Path, object]:
    root = tmp_path / "project"
    destination = tmp_path / "external"
    active = root / "reports" / "active"
    source = (
        root
        / "reports"
        / "snapshots"
        / "current_wizard_hyperliquid"
        / "cwconcentration_old"
    )
    active.mkdir(parents=True)
    source.mkdir(parents=True)
    destination.mkdir()
    (source / "manifest.json").write_text(
        json.dumps({"snapshot_id": "cwconcentration_old"}), encoding="utf-8"
    )
    (source / "evidence.csv").write_text("pair,value\nBTC-ETH,1\n", encoding="utf-8")
    size_bytes, file_count, tree_sha256 = storage._tree_inventory(source)
    manifest_sha256 = storage._file_hash(source / "manifest.json")
    reclamation_id = "cwreclaim_test"
    pd.DataFrame(
        [
            {
                "reclamation_id": reclamation_id,
                "snapshot_path": str(source.relative_to(root)),
                "size_bytes": size_bytes,
                "file_count": file_count,
                "manifest_sha256": manifest_sha256,
                "tree_sha256": tree_sha256,
                "hash_verified": True,
                "safe_to_archive_later": True,
            }
        ]
    ).to_csv(
        active / "current_wizard_hyperliquid_storage_reclamation_plan.csv",
        index=False,
    )
    (active / "current_wizard_hyperliquid_storage_reclamation_manifest.json").write_text(
        json.dumps({"reclamation_id": reclamation_id}), encoding="utf-8"
    )

    root_device = 1
    destination_device = 2

    def fake_device_id(path: Path) -> int:
        resolved = path.resolve()
        if resolved == destination.resolve():
            return destination_device
        return root_device

    monkeypatch.setattr(storage, "_device_id", fake_device_id)
    monkeypatch.setattr(release_module, "_device_id", fake_device_id)
    monkeypatch.setattr(
        storage.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=10 * 1024**3),
    )

    result = stage_current_wizard_hyperliquid_archive_copy(
        root=root,
        archive_destination=destination,
        approval_id="test-copy-approval",
        now=datetime(2026, 8, 8, 20, 0, tzinfo=timezone.utc),
    )
    return root, destination, source, result


def test_archive_copy_is_hash_verified_idempotent_and_never_releases_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, destination, source, result = _build_verified_copy(tmp_path, monkeypatch)
    reclamation_id = str(result.summary["reclamation_id"])
    receipt = pd.read_csv(result.paths["receipt"], keep_default_na=False)
    validation = pd.read_csv(result.paths["validation"], keep_default_na=False)
    archived = (
        destination
        / reclamation_id
        / source.relative_to(root)
    )

    assert receipt["copy_status"].tolist() == ["COPIED_VERIFIED"]
    assert receipt["tree_hash_match"].astype(bool).all()
    assert validation["status"].eq("PASS").all()
    assert archived.is_dir()
    assert storage._tree_inventory(archived) == storage._tree_inventory(source)
    assert source.is_dir()
    assert result.summary["archive_copy_completed"] is True
    assert result.summary["archive_release_evidence_complete"] is True
    assert result.summary["archive_release_preflight_ready"] is False
    assert result.summary["source_release_authorized"] is False
    assert result.summary["source_move_performed"] is False
    assert result.summary["source_delete_performed"] is False
    assert result.summary["live_trading_authorized"] is False
    first_destination_receipt = Path(result.summary["destination_receipt"])
    assert first_destination_receipt.is_file()

    retry = stage_current_wizard_hyperliquid_archive_copy(
        root=root,
        archive_destination=destination,
        approval_id="test-copy-approval",
        now=datetime(2026, 8, 8, 20, 1, tzinfo=timezone.utc),
    )
    retry_receipt = pd.read_csv(retry.paths["receipt"], keep_default_na=False)
    assert retry_receipt["copy_status"].tolist() == ["ALREADY_VERIFIED"]
    assert retry.summary["bytes_copied_this_run"] == 0
    assert retry.summary["bytes_requiring_copy_at_preflight"] == 0
    second_destination_receipt = Path(retry.summary["destination_receipt"])
    assert second_destination_receipt.is_file()
    assert second_destination_receipt != first_destination_receipt
    assert len(list(second_destination_receipt.parent.glob("*.json"))) == 2
    assert source.is_dir()


def test_archive_release_dry_run_reverifies_copy_but_never_deletes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _destination, source, copy_result = _build_verified_copy(
        tmp_path, monkeypatch
    )
    result = build_current_wizard_hyperliquid_archive_release_dry_run(
        root=root,
        now=datetime(2026, 8, 8, 21, 0, tzinfo=timezone.utc),
        available_disk_bytes=3 * 1024**3,
    )
    plan = pd.read_csv(result.paths["plan"], keep_default_na=False)
    validation = pd.read_csv(result.paths["validation"], keep_default_na=False)

    assert result.summary["copy_id"] == copy_result.summary["copy_id"]
    assert result.summary["release_dry_run_ready"] is True
    assert result.summary["release_status"] == "READY_FOR_SEPARATE_APPROVAL"
    assert result.summary["release_candidates"] == 1
    assert result.summary["recoverable_bytes"] > 0
    assert plan["release_candidate"].astype(bool).all()
    assert plan["tree_hash_match"].astype(bool).all()
    assert validation["status"].eq("PASS").all()
    assert result.summary["source_release_authorized"] is False
    assert result.summary["source_move_performed"] is False
    assert result.summary["source_delete_performed"] is False
    assert result.summary["order_submission_performed"] is False
    assert result.summary["live_trading_authorized"] is False
    assert source.is_dir()

    active = root / "reports" / "active"
    (active / "new_research_manifest.json").write_text(
        json.dumps({"snapshot": str(source.relative_to(root))}), encoding="utf-8"
    )
    blocked = build_current_wizard_hyperliquid_archive_release_dry_run(
        root=root,
        now=datetime(2026, 8, 8, 21, 1, tzinfo=timezone.utc),
        available_disk_bytes=3 * 1024**3,
    )
    blocked_plan = pd.read_csv(blocked.paths["plan"], keep_default_na=False)
    assert blocked.summary["release_dry_run_ready"] is False
    assert blocked.summary["release_status"] == "BLOCKED"
    assert "source_became_current_lineage" in blocked.summary["release_blocker"]
    assert not blocked_plan["release_candidate"].astype(bool).any()
    assert not blocked_plan["source_delete_performed"].astype(bool).any()
    assert source.is_dir()


def test_archive_release_dry_run_blocks_without_copy_receipt(tmp_path: Path) -> None:
    result = build_current_wizard_hyperliquid_archive_release_dry_run(
        root=tmp_path,
        now=datetime(2026, 8, 8, 21, 0, tzinfo=timezone.utc),
        available_disk_bytes=100,
    )
    plan = pd.read_csv(result.paths["plan"], keep_default_na=False)

    assert result.summary["release_dry_run_ready"] is False
    assert result.summary["release_blocker"] == "verified_off_volume_archive_copy_missing"
    assert not plan["release_candidate"].astype(bool).any()
    assert result.summary["source_release_authorized"] is False
    assert result.summary["source_delete_performed"] is False
    assert result.summary["live_trading_authorized"] is False


def test_archive_copy_requires_explicit_approval(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="archive_copy_approval_id_required"):
        stage_current_wizard_hyperliquid_archive_copy(
            root=tmp_path,
            archive_destination=tmp_path / "external",
            approval_id="",
        )
