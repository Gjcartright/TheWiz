from datetime import UTC, datetime

import pandas as pd

from quant_platform.crypto_wizards_catalog import BASE_URL
from quant_platform.crypto_wizards_sweep import (
    run_authorized_wizard_discovery_sweep,
    run_wizard_discovery_sweep,
)
from quant_platform.orchestration.corrective_external_effects import (
    external_effect_issuer_session,
)
from quant_platform.orchestration.effect_authority import (
    PHASE00_WIZARD_RESEARCH_PROFILE,
    EffectAuthority,
)
from quant_platform.orchestration.nodes import run_stage, stages_for_group
from quant_platform.orchestration.state import OrchestratorState, StageStatus
from quant_platform.wizard_control_plane import build_wizard_control_plane
from quant_platform.wizard_evidence import build_wizard_pair_settings_capture_template

NOW = datetime(2026, 8, 7, 12, 0, tzinfo=UTC)
TEST_HASH = "a" * 64


def _credit_usage_fetcher(*used_values: int):
    readings = iter(used_values)

    def fetcher(**_):
        return {"credits_used": next(readings), "credit_limit": 1000}

    return fetcher


def _complete_sweep(
    tmp_path,
    *,
    source_timestamp: str = "2026-08-07T11:30:00Z",
    returns_total=0.25,
    closed: int = 5,
):
    def fake_prescanned(**kwargs):
        return [
            {
                "pair_id": "btc-eth",
                "symbol_1": "BTC-USD",
                "symbol_2": "ETH-USD",
                "sharpe": 2.4,
                "returns_total": returns_total,
                "closed": closed,
                "backtest_ts": source_timestamp,
                "spread_type": "Static",
                "period": 365,
                "x_weighting": 0.5,
                "y_weighting": 0.5,
            }
        ]

    authority = EffectAuthority(
        root=tmp_path,
        secret=b"wizard-control-plane-test-authority",
        issuer_id="wizard-control-plane-test",
        profile=PHASE00_WIZARD_RESEARCH_PROFILE,
    )
    with external_effect_issuer_session(
        authority=authority,
        run_id="wizard-control-plane-run",
        intended_slot_id="wizard-control-plane-slot",
        source_fingerprint_sha256=TEST_HASH,
        runtime_fingerprint_sha256=TEST_HASH,
        configuration_fingerprint_sha256=TEST_HASH,
        provider_id="crypto_wizards",
        account_scope_id="crypto_wizards:research:test",
        allowed_targets=frozenset(
            {
                f"{BASE_URL}/v1beta/credits-used",
                f"{BASE_URL}/v1beta/prescanned",
            }
        ),
        allowed_credential_keys=frozenset({"CRYPTO_WIZARDS_API_KEY"}),
        max_total_requests=3,
        max_total_credits=10,
    ):
        return run_authorized_wizard_discovery_sweep(
            root=tmp_path,
            api_key="secret",
            exchanges=("Dydx",),
            intervals=("Daily",),
            strategies=("Spread",),
            credits_fetcher=_credit_usage_fetcher(0, 10),
            prescanned_fetcher=fake_prescanned,
            now=NOW,
        )


def test_preflight_sweep_is_visible_but_cannot_authorize_ranking(tmp_path):
    run_wizard_discovery_sweep(
        root=tmp_path,
        exchanges=("Dydx",),
        intervals=("Daily",),
        strategies=("Spread",),
        now=NOW,
    )

    result = build_wizard_control_plane(root=tmp_path, now=NOW)
    health = pd.read_csv(result.paths["wizard_control_plane_health"])
    queue = pd.read_csv(result.paths["wizard_sweep_settings_capture_queue"])

    assert result.summary["ready"] is False
    assert "wizard_sweep_not_complete" in result.summary["blockers"]
    assert "wizard_candidates_missing" in result.summary["blockers"]
    assert result.summary["api_contract_ready"] is True
    assert health.loc[health["check"] == "ranking_authority", "status"].item() == "block"
    assert queue.empty


def test_complete_fresh_sweep_builds_actionable_settings_queue_with_lineage(tmp_path):
    _complete_sweep(tmp_path)

    result = build_wizard_control_plane(root=tmp_path, now=NOW)
    queue = pd.read_csv(result.paths["wizard_sweep_settings_capture_queue"])
    lineage = pd.read_csv(result.paths["wizard_config_lineage_audit"])
    contract = pd.read_csv(result.paths["wizard_api_contract_report"])
    template_result = build_wizard_pair_settings_capture_template(root=tmp_path)
    template = pd.read_csv(template_result.paths["wizard_pair_settings_capture_template"])

    assert result.summary["ready"] is True
    assert result.summary["actionable_settings_rows"] == 1
    assert result.summary["research_spend_gate_rows"] == 1
    assert queue.loc[0, "pair"] == "BTC-USD / ETH-USD"
    assert queue.loc[0, "exact_mode"] == "static_spread"
    assert bool(queue.loc[0, "actionable"])
    assert len(queue.loc[0, "discovery_config_hash"]) == 64
    assert len(queue.loc[0, "candidate_config_hash"]) == 64
    assert len(queue.loc[0, "discovery_policy_hash"]) == 64
    assert queue.loc[0, "min_closed_trades_for_proof"] == 5
    assert bool(queue.loc[0, "passes_research_spend_gate"])
    assert "commission_rate" in queue.loc[0, "missing_settings"]
    assert template.loc[0, "candidate_config_hash"] == queue.loc[0, "candidate_config_hash"]
    assert template.loc[0, "exact_mode"] == "static_spread"
    assert lineage[lineage["required_for_ranking"].astype(bool)]["status"].eq("pass").all()
    assert not ((contract["blocking"].astype(bool)) & contract["status"].eq("fail")).any()


def test_raw_discovery_stays_actionable_but_one_trade_cannot_spend_proof_credits(tmp_path):
    _complete_sweep(tmp_path, closed=1)

    result = build_wizard_control_plane(root=tmp_path, now=NOW)
    queue = pd.read_csv(result.paths["wizard_sweep_settings_capture_queue"])

    assert result.summary["discovery_gate_rows"] == 1
    assert result.summary["actionable_settings_rows"] == 1
    assert result.summary["research_spend_gate_rows"] == 0
    assert bool(queue.loc[0, "passes_discovery_gate"])
    assert not bool(queue.loc[0, "passes_research_spend_gate"])


def test_stale_source_timestamp_blocks_candidate_action(tmp_path):
    _complete_sweep(tmp_path, source_timestamp="2026-08-01T12:00:00Z")

    result = build_wizard_control_plane(root=tmp_path, now=NOW, max_age_hours=24)
    freshness = pd.read_csv(result.paths["wizard_freshness_blockers"])
    queue = pd.read_csv(result.paths["wizard_sweep_settings_capture_queue"])

    assert result.summary["ready"] is False
    assert "no_fresh_wizard_candidates" in result.summary["blockers"]
    assert "source_data_stale" in freshness.loc[0, "freshness_blocker"]
    assert not bool(queue.loc[0, "actionable"])
    assert queue.loc[0, "capture_status"] == "BLOCKED_CONTROL_PLANE"


def test_validated_settings_capture_completes_matching_queue_row(tmp_path):
    _complete_sweep(tmp_path)
    first = build_wizard_control_plane(root=tmp_path, now=NOW)
    queue = pd.read_csv(first.paths["wizard_sweep_settings_capture_queue"])
    candidate_hash = queue.loc[0, "candidate_config_hash"]
    pd.DataFrame(
        [
            {
                "candidate_config_hash": candidate_hash,
                "settings_config_hash": "a" * 64,
                "capture_confirmed": True,
                "backtest_settings_complete": True,
                "capture_timestamp_utc": NOW.isoformat(),
                "capture_evidence_path": "reports/active/live_capture.md",
            }
        ]
    ).to_csv(
        tmp_path / "reports" / "active" / "crypto_wizards_pair_page_capture_settings.csv",
        index=False,
    )

    second = build_wizard_control_plane(root=tmp_path, now=NOW)
    refreshed = pd.read_csv(second.paths["wizard_sweep_settings_capture_queue"])

    assert refreshed.loc[0, "capture_status"] == "SETTINGS_COMPLETE"
    assert pd.isna(refreshed.loc[0, "missing_settings"])
    assert refreshed.loc[0, "settings_config_hash"] == "a" * 64
    assert refreshed.loc[0, "settings_capture_evidence_path"] == "reports/active/live_capture.md"


def test_malformed_api_discovery_value_fails_contract(tmp_path):
    _complete_sweep(tmp_path, returns_total="not-a-number")

    result = build_wizard_control_plane(root=tmp_path, now=NOW)
    contract = pd.read_csv(result.paths["wizard_api_contract_report"])
    numeric = contract[contract["check"] == "numeric_discovery_values"].iloc[0]

    assert result.summary["ready"] is False
    assert "wizard_api_contract_failed" in result.summary["blockers"]
    assert numeric["status"] == "fail"
    assert numeric["failure_count"] == 1


def test_orchestrator_checks_control_plane_before_wizard_discovery(tmp_path):
    stages = stages_for_group("discovery")
    assert stages.index("wizard_control_plane") < stages.index("discover_wizard_candidates")

    state = OrchestratorState(root=tmp_path)
    control = run_stage("wizard_control_plane", state, root=tmp_path)
    discovery = run_stage("discover_wizard_candidates", state, root=tmp_path)

    assert control.status == StageStatus.BLOCKED
    assert discovery.status == StageStatus.BLOCKED
    assert discovery.reason == "wizard_discovery_not_rebuilt_from_partial_or_stale_sweep"
    assert not (tmp_path / "data" / "processed" / "wizard_evidence.csv").exists()
