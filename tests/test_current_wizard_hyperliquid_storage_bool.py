"""Persisted gate booleans must be parsed before reclamation validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_platform.orchestration.current_wizard_hyperliquid_storage import _validation


def _plan(*, safe: str, hash_verified: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "snapshot_path": "reports/snapshots/current_wizard_hyperliquid/cwconcentration_old",
                "safe_to_archive_later": safe,
                "manifest_present": "True",
                "hash_verified": hash_verified,
                "manifest_sha256": "a" * 64,
                "tree_sha256": "b" * 64,
                "protected_current_paths": 0,
                "current_lineage": "False",
                "move_performed": "False",
                "delete_performed": "False",
                "live_trading_authorized": "False",
            }
        ]
    )


def _destination() -> dict[str, object]:
    return {
        "archive_copy_preflight_ready": "False",
        "archive_destination_configured": "False",
        "archive_destination_exists": "False",
        "archive_destination_writable": "False",
        "archive_destination_same_device": "False",
        "archive_destination_free_bytes": 0,
        "archive_destination_required_bytes": 1,
        "archive_destination_status": "BLOCKED",
        "archive_release_preflight_ready": "False",
    }


def test_storage_validation_rejects_string_false_hash_without_false_alarms(
    tmp_path: Path,
) -> None:
    validation = _validation(
        plan=_plan(safe="True", hash_verified="False"),
        protected_paths={tmp_path / "protected"},
        root=tmp_path,
        destination=_destination(),
    ).set_index("check")

    assert validation.loc["safe_candidates_have_verified_tree_hashes", "status"] == "FAIL"
    assert validation.drop("safe_candidates_have_verified_tree_hashes")["status"].eq("PASS").all()


def test_storage_validation_excludes_string_false_safe_candidate(tmp_path: Path) -> None:
    validation = _validation(
        plan=_plan(safe="False", hash_verified="False"),
        protected_paths={tmp_path / "protected"},
        root=tmp_path,
        destination=_destination(),
    ).set_index("check")

    assert validation["status"].eq("PASS").all()
