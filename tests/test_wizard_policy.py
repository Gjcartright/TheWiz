from __future__ import annotations

import json

import pytest

from quant_platform.wizard_policy import DEFAULT_WIZARD_DISCOVERY_POLICY, load_wizard_discovery_policy


def test_missing_policy_uses_canonical_defaults(tmp_path):
    policy = load_wizard_discovery_policy(tmp_path)

    assert policy == DEFAULT_WIZARD_DISCOVERY_POLICY
    assert policy.min_sharpe == 1.75
    assert policy.min_returns_total_pct == 10.0
    assert policy.min_closed_trades_for_proof == 5
    assert len(policy.policy_hash) == 64


def test_policy_file_loads_and_hashes_canonical_content(tmp_path):
    path = tmp_path / "config" / "wizard_discovery_policy.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "wizard_discovery_policy.v1",
                "raw_leaderboard": {"min_sharpe": 2.0, "min_returns_total_pct": 15.0},
                "proof_spend": {"min_closed_trades": 7},
            }
        ),
        encoding="utf-8",
    )

    first = load_wizard_discovery_policy(tmp_path)
    second = load_wizard_discovery_policy(tmp_path)

    assert first.min_sharpe == 2.0
    assert first.min_returns_total_pct == 15.0
    assert first.min_closed_trades_for_proof == 7
    assert first.policy_hash == second.policy_hash


def test_malformed_existing_policy_fails_closed(tmp_path):
    path = tmp_path / "config" / "wizard_discovery_policy.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not-json", encoding="utf-8")

    with pytest.raises((ValueError, json.JSONDecodeError)):
        load_wizard_discovery_policy(tmp_path)
