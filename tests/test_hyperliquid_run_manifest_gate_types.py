"""Persisted Hyperliquid readiness values must not gain authority by string truthiness."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from quant_platform.orchestration.hyperliquid_run_manifest import (
    build_hyperliquid_authority_state,
)

_GATE_FIELDS = (
    "lineage_ready",
    "research_evidence_ready",
    "cost_evidence_ready",
    "wizard_parity_ready",
    "candidate_blockers_clear",
)


def _write_gate_fixture(root: Path, *, false_field: str | None) -> None:
    active = root / "reports" / "active"
    council = root / "reports" / "orchestration" / "teacher_council"
    paper = root / "reports" / "paper"
    for directory in (active, council, paper):
        directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": "run_1",
        "candidate_set_id": "set_1",
        **{field: "False" if field == false_field else "True" for field in _GATE_FIELDS},
    }
    (active / "hyperliquid_run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    pd.DataFrame([{"run_id": "run_1", "candidate_set_id": "set_1", "status": "SHADOW_TEST", "action": "observe"}]).to_csv(
        council / "council_decisions.csv", index=False
    )
    pd.DataFrame([{"verdict": "pass"}]).to_csv(council / "portfolio_critic.csv", index=False)
    pd.DataFrame([{"scope": "supervised_student", "status": "READY"}]).to_csv(
        council / "student_training_readiness.csv", index=False
    )
    pd.DataFrame([{"ready_for_no_order_preflight": True, "submit_orders_enabled": True}]).to_csv(
        active / "hyperliquid_testnet_preflight.csv", index=False
    )
    pd.DataFrame([{"status": "PASS"}]).to_csv(
        active / "hyperliquid_testnet_lifecycle_gate.csv", index=False
    )
    pd.DataFrame([{"accepted": True}]).to_csv(
        paper / "hyperliquid_realized_validation.csv", index=False
    )


@pytest.mark.parametrize("false_field", _GATE_FIELDS)
def test_persisted_false_string_cannot_grant_live_authority(tmp_path: Path, false_field: str) -> None:
    _write_gate_fixture(tmp_path, false_field=false_field)

    result = build_hyperliquid_authority_state(root=tmp_path)

    assert result["research_ready"] is False
    assert result["paper_ready"] is False
    assert result["live_ready"] is False
    assert result["execution_allowed"] is False
    assert result["status"] == "BLOCKED"


def test_persisted_true_strings_preserve_research_readiness_without_submission(tmp_path: Path) -> None:
    _write_gate_fixture(tmp_path, false_field=None)
    preflight = tmp_path / "reports" / "active" / "hyperliquid_testnet_preflight.csv"
    frame = pd.read_csv(preflight)
    frame["submit_orders_enabled"] = False
    frame.to_csv(preflight, index=False)

    result = build_hyperliquid_authority_state(root=tmp_path)

    assert result["research_ready"] is True
    assert result["live_ready"] is False
    assert result["execution_allowed"] is False
