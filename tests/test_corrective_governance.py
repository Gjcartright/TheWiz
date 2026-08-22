from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from quant_platform.orchestration.corrective_governance import (
    GovernanceError,
    build_corrective_governance,
    content_identity,
    semantic_hypothesis_id,
)

ROOT = Path(__file__).resolve().parents[1]


def _root(tmp_path: Path) -> Path:
    for relative in (
        "config/acceptance_policy_manifest.json",
        "config/research_holdout_policy.json",
        "config/research.yaml",
        "config/wizard_discovery_policy.json",
        "config/hyperliquid_perp_cost_profile.json",
    ):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "failure_attribution_id": "run_1",
                "experiment_id": "display_name_a",
                "pair": "ETH-USD-WIF-USD",
                "asset_x": "ETH",
                "asset_y": "WIF",
                "wizard_timeframe": "Daily",
                "exact_mode": "Copula",
                "orientation": "reverse",
                "observed_trades": 20,
                "walkforward_trades": 12,
                "consolidated_status": "BLOCKED_CONCENTRATION",
                "first_blocker": "concentration_gate_failed",
                "evidence_path": "reports/active/evidence.csv",
            }
        ]
    ).to_csv(active / "current_wizard_hyperliquid_failure_attribution.csv", index=False)
    return tmp_path


def test_governance_materializes_policy_tamper_holdout_and_ledger(tmp_path):
    root = _root(tmp_path)
    result = build_corrective_governance(
        root=root, now=datetime(2026, 8, 9, tzinfo=UTC)
    )

    assert result.summary["status"] == "PASS"
    assert result.summary["semantic_hypothesis_families"] == 1
    assert result.summary["ledger_records"] == 1
    assert result.summary["live_trading_authorized"] is False
    tamper = pd.read_csv(result.paths["gate_policy_tamper_results"])
    assert len(tamper) > 30
    assert tamper["identity_changed"].all()
    assert not tamper["promotion_authority_inherited"].any()


def test_semantic_identity_ignores_display_and_run_renames():
    base = {
        "experiment_id": "old-name",
        "failure_attribution_id": "run-old",
        "asset_x": "ETH",
        "asset_y": "WIF",
        "timeframe": "Daily",
        "exact_mode": "Copula",
        "orientation": "reverse",
    }
    renamed = {**base, "experiment_id": "new-name", "failure_attribution_id": "run-new", "pair": "WIF/ETH"}
    assert semantic_hypothesis_id(base, policy_id="p", holdout_policy_id="h") == semantic_hypothesis_id(
        renamed, policy_id="p", holdout_policy_id="h"
    )


def test_policy_mutation_changes_identity_and_blocks_source_contract(tmp_path):
    root = _root(tmp_path)
    policy_path = root / "config" / "acceptance_policy_manifest.json"
    policy = json.loads(policy_path.read_text())
    before = content_identity(policy)
    policy["research_gates"]["minimum_profit_factor"] = 9.0
    assert content_identity(policy) != before

    research = root / "config" / "research.yaml"
    research.write_text(research.read_text() + "\n# unauthorized drift\n")
    result = build_corrective_governance
    with pytest.raises(GovernanceError, match="acceptance policy receipt is blocked"):
        result(root=root)


def test_ledger_tampering_fails_closed(tmp_path):
    root = _root(tmp_path)
    result = build_corrective_governance(root=root)
    ledger = result.paths["hypothesis_ledger"]
    row = json.loads(ledger.read_text().splitlines()[0])
    row["attempt_number"] = 999
    ledger.write_text(json.dumps(row) + "\n")
    with pytest.raises(GovernanceError, match="ledger chain invalid"):
        build_corrective_governance(root=root)
