import json
from datetime import UTC, datetime

import pandas as pd

from quant_platform.api_extraction import CryptoWizardsFetchError
from quant_platform.crypto_wizards_catalog import BASE_URL
from quant_platform.crypto_wizards_sweep import (
    build_wizard_sweep_cells,
    parse_wizard_credit_usage,
    restore_complete_wizard_sweep_from_raw,
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
from quant_platform.wizard_run_config import WizardRunConfiguration

TEST_RUN_ID = "wizard-sweep-authority-test"
TEST_SLOT_ID = "wizard-sweep-authority-slot"
TEST_HASH = "a" * 64


def _run_authorized_sweep(root, **kwargs):
    authority = EffectAuthority(
        root=root,
        secret=b"wizard-sweep-test-authority-secret",
        issuer_id="wizard-sweep-test-supervisor",
        profile=PHASE00_WIZARD_RESEARCH_PROFILE,
    )
    with external_effect_issuer_session(
        authority=authority,
        run_id=TEST_RUN_ID,
        intended_slot_id=TEST_SLOT_ID,
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
        max_total_requests=100,
        max_total_credits=1000,
    ):
        result = run_authorized_wizard_discovery_sweep(root=root, **kwargs)
    return result, authority


def _credit_usage_fetcher(*used_values: int):
    readings = iter(used_values)

    def fetcher(**_):
        return {"credits_used": next(readings), "credit_limit": 1000}

    return fetcher


def test_wizard_configuration_hash_is_stable_and_cost_sensitive():
    left = WizardRunConfiguration(
        source=" Backtest API ",
        symbol_1="btc-usd",
        symbol_2="eth-usd",
        wizard_exchange="DYDX",
        interval="Daily",
        period=365,
        strategy="ZScoreRoll",
        spread_type="OU",
        roll_window=42,
        commission_rate=0.001,
        slippage_rate=0.0005,
    )
    same = WizardRunConfiguration(
        source="backtest_api",
        symbol_1="BTC-USD",
        symbol_2="ETH-USD",
        wizard_exchange="Dydx",
        interval="1d",
        period=365,
        strategy="zscore_roll",
        spread_type="ou",
        roll_window=42,
        commission_rate=0.001,
        slippage_rate=0.0005,
    )
    changed_cost = WizardRunConfiguration(
        source="backtest_api",
        symbol_1="BTC-USD",
        symbol_2="ETH-USD",
        wizard_exchange="Dydx",
        interval="Daily",
        period=365,
        strategy="ZScoreRoll",
        spread_type="OU",
        roll_window=42,
        commission_rate=0.002,
        slippage_rate=0.0005,
    )

    assert left.config_hash == same.config_hash
    assert left.config_hash != changed_cost.config_hash
    assert left.canonical_dict()["exact_mode"] == "ou_zscore_r"


def test_default_wizard_sweep_plans_complete_crypto_matrix():
    cells = build_wizard_sweep_cells(sweep_id="test")

    assert len(cells) == 30
    assert sum(cell.credit_cost for cell in cells) == 300
    assert len({cell.request_id for cell in cells}) == 30
    assert len({cell.config_hash for cell in cells}) == 30
    assert {cell.exchange for cell in cells} == {
        "Binance",
        "BinanceUs",
        "ByBit",
        "Coinbase",
        "Dydx",
    }
    assert {cell.interval for cell in cells} == {"Daily", "Hourly"}
    assert {cell.strategy for cell in cells} == {"Spread", "ZScoreRoll", "Copula"}


def test_credit_usage_parser_handles_nested_used_and_limit():
    usage = parse_wizard_credit_usage(
        {"data": {"credits_used": "125", "daily_allowance": "1000"}},
        configured_limit=900,
    )

    assert usage.known is True
    assert usage.used == 125
    assert usage.limit == 1000
    assert usage.remaining == 875
    assert usage.source_fields == "data.credits_used"


def test_credit_usage_parser_handles_official_scalar_used_response():
    usage = parse_wizard_credit_usage(300, configured_limit=1000)

    assert usage.known is True
    assert usage.used == 300
    assert usage.limit == 1000
    assert usage.remaining == 700
    assert usage.source_fields == "scalar_response"


def test_sweep_is_dry_by_default_and_cannot_claim_complete_discovery(tmp_path):
    result = run_wizard_discovery_sweep(
        root=tmp_path,
        exchanges=("Dydx",),
        intervals=("Daily",),
        strategies=("Spread",),
        now=datetime(2026, 8, 7, tzinfo=UTC),
    )

    manifest = pd.read_csv(result.paths["manifest"])
    candidates = pd.read_csv(result.paths["candidates"])
    assert result.summary["execute"] is False
    assert result.summary["discovery_authority"] == "preflight_only"
    assert result.summary["sweep_complete"] is False
    assert manifest.loc[0, "status"] == "planned"
    assert manifest.loc[0, "sweep_blocker"] == "execution_not_requested"
    assert candidates.empty
    assert {"sweep_config_hash", "discovery_authority", "sweep_blocker"}.issubset(
        candidates.columns
    )
    assert not result.paths["raw_snapshot_dir"].exists()


def test_credit_preflight_blocks_entire_sweep_before_any_paid_call(tmp_path):
    calls = []

    def fake_prescanned(**kwargs):
        calls.append(kwargs)
        return []

    result, _authority = _run_authorized_sweep(
        root=tmp_path,
        api_key="secret",
        exchanges=("Dydx",),
        intervals=("Daily", "Hourly"),
        strategies=("Spread", "Copula"),
        daily_credit_limit=100,
        reserved_credits=20,
        credits_fetcher=lambda **kwargs: {"credits_used": 50},
        prescanned_fetcher=fake_prescanned,
    )

    manifest = pd.read_csv(result.paths["manifest"])
    assert calls == []
    assert result.summary["planned_credits"] == 40
    assert result.summary["attempted_credits"] == 0
    assert result.summary["blocker"] == "insufficient_credits_for_complete_sweep"
    assert manifest["status"].eq("blocked_insufficient_credits_for_complete_sweep").all()


def test_direct_paid_sweep_without_issuer_cannot_reach_provider(tmp_path):
    calls = []

    result = run_wizard_discovery_sweep(
        root=tmp_path,
        execute=True,
        api_key="secret",
        exchanges=("Dydx",),
        intervals=("Daily",),
        strategies=("Spread",),
        credits_fetcher=lambda **kwargs: calls.append(kwargs),
        prescanned_fetcher=lambda **kwargs: calls.append(kwargs),
    )

    assert calls == []
    assert result.summary["attempted_credits"] == 0
    assert result.summary["blocker"] == (
        "reused_credit_reservation_does_not_authorize_external_replay"
    )


def test_blocked_sweep_preserves_prior_complete_active_snapshot(tmp_path):
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    prior_candidates = pd.DataFrame([{"pair_id": 42, "symbol_1": "BTC", "symbol_2": "ETH"}])
    prior_manifest = pd.DataFrame([{"status": "completed", "sweep_complete": True}])
    prior_candidates.to_csv(active / "wizard_sweep_candidates.csv", index=False)
    prior_manifest.to_csv(active / "wizard_sweep_manifest.csv", index=False)

    result, _authority = _run_authorized_sweep(
        root=tmp_path,
        api_key="secret",
        exchanges=("Dydx",),
        intervals=("Daily",),
        strategies=("Spread",),
        daily_credit_limit=100,
        reserved_credits=20,
        credits_fetcher=lambda **kwargs: {"credits_used": 80},
    )

    active_candidates = pd.read_csv(active / "wizard_sweep_candidates.csv")
    active_manifest = pd.read_csv(active / "wizard_sweep_manifest.csv")
    attempt_manifest = pd.read_csv(result.paths["manifest"])
    assert result.summary["active_snapshot_preserved"] is True
    assert result.summary["active_candidate_rows"] == 1
    assert active_candidates["pair_id"].tolist() == [42]
    assert active_manifest["status"].tolist() == ["completed"]
    assert attempt_manifest["status"].str.startswith("blocked_").all()


def test_restore_complete_sweep_from_raw_rebuilds_canonical_active_files(tmp_path):
    cells = build_wizard_sweep_cells(sweep_id="complete-raw")
    raw_dir = tmp_path / "data" / "raw" / "crypto_wizards" / "prescanned" / "2026-08-07"
    raw_dir.mkdir(parents=True)
    for cell in cells:
        response = [
            {
                "pair_id": int(cell.request_id[:8], 16),
                "symbol_1": "BTC-USD",
                "symbol_2": "ETH-USD",
                "backtest_ts": 1_786_000_000,
            }
        ]
        envelope = {
            "capture_metadata": {
                "schema_version": "wizard_discovery_sweep.v1",
                "sweep_id": "complete-raw",
                "request_id": cell.request_id,
                "config_hash": cell.config_hash,
                "endpoint": cell.endpoint,
                "credit_cost": cell.credit_cost,
                "captured_at": "2026-08-07T12:00:00+00:00",
                "response_hash": "hash",
            },
            "request": cell.params(),
            "response": response,
        }
        (raw_dir / f"complete-raw_{cell.request_id}.json").write_text(
            json.dumps(envelope), encoding="utf-8"
        )

    result = restore_complete_wizard_sweep_from_raw(root=tmp_path)

    restored = pd.read_csv(result.paths["candidates"])
    manifest = pd.read_csv(result.paths["manifest"])
    assert result.summary["restored_from_raw"] is True
    assert result.summary["candidate_rows"] == 30
    assert len(restored) == 30
    assert manifest["status"].eq("completed").all()
    assert manifest["sweep_complete"].astype(bool).all()


def test_successful_sweep_writes_raw_evidence_candidates_and_complete_authority(tmp_path):
    def fake_prescanned(**kwargs):
        return [
            {
                "pair_id": f"{kwargs['exchange']}-{kwargs['strategy']}",
                "symbol_1": "BTC-USD",
                "symbol_2": "ETH-USD",
                "sharpe": 2.4,
                "returns_total": 0.25,
            }
        ]

    result, authority = _run_authorized_sweep(
        root=tmp_path,
        api_key="secret",
        exchanges=("Dydx", "Coinbase"),
        intervals=("Daily",),
        strategies=("Spread", "Copula"),
        credits_fetcher=_credit_usage_fetcher(100, 140),
        prescanned_fetcher=fake_prescanned,
        now=datetime(2026, 8, 7, 12, 30, tzinfo=UTC),
    )

    manifest = pd.read_csv(result.paths["manifest"])
    candidates = pd.read_csv(result.paths["candidates"])
    raw_paths = sorted(result.paths["raw_snapshot_dir"].glob("*.json"))

    assert result.summary["planned_cells"] == 4
    assert result.summary["planned_credits"] == 40
    assert result.summary["attempted_credits"] == 40
    assert result.summary["credits_used_before"] == 100
    assert result.summary["credits_used_after"] == 140
    assert result.summary["observed_credit_delta"] == 40
    assert result.summary["credit_reconciliation_status"] == "PASS_RECONCILED"
    assert result.summary["sweep_complete"] is True
    assert result.summary["discovery_authority"] == "complete_discovery"
    assert manifest["status"].eq("completed").all()
    assert manifest["config_hash"].str.len().eq(64).all()
    assert candidates.shape[0] == 4
    assert candidates["discovery_authority"].eq("complete_discovery").all()
    assert (
        candidates["sweep_evidence_path"]
        .str.startswith("data/raw/crypto_wizards/prescanned/")
        .all()
    )
    assert len(raw_paths) == 4
    for key in (
        "credit_usage_before_evidence",
        "credit_usage_after_evidence",
    ):
        evidence_path = result.paths[key]
        assert evidence_path.is_file()
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        assert evidence["capture_metadata"]["capture_type"] == "credit_usage"
        assert "api_key" not in json.dumps(evidence).lower()
    assert len(result.summary["credit_usage_before_evidence_sha256"]) == 64
    assert len(result.summary["credit_usage_after_evidence_sha256"]) == 64
    accounting = authority.run_accounting(
        run_id=TEST_RUN_ID,
        intended_slot_id=TEST_SLOT_ID,
    )
    assert accounting["external_calls"] == 6
    assert accounting["external_credits_consumed"] == 40
    assert (
        accounting["matching_permits"]
        - accounting["consumed_provider_effect_permits"]
        == 1
    )
    envelope = json.loads(raw_paths[0].read_text(encoding="utf-8"))
    assert envelope["capture_metadata"]["response_hash"]
    assert envelope["request"]["interval"] == "Daily"
    assert "api_key" not in json.dumps(envelope).lower()


def test_successful_attempt_only_sweep_does_not_replace_active_snapshot(tmp_path):
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    pd.DataFrame([{"pair_id": 7, "symbol_1": "OLD", "symbol_2": "PAIR"}]).to_csv(
        active / "wizard_sweep_candidates.csv", index=False
    )
    pd.DataFrame([{"status": "completed", "sweep_complete": True}]).to_csv(
        active / "wizard_sweep_manifest.csv", index=False
    )

    result, _authority = _run_authorized_sweep(
        root=tmp_path,
        api_key="secret",
        exchanges=("Dydx",),
        intervals=("Daily",),
        strategies=("Spread",),
        credits_fetcher=_credit_usage_fetcher(0, 10),
        prescanned_fetcher=lambda **kwargs: [
            {"pair_id": 8, "symbol_1": "NEW", "symbol_2": "PAIR"}
        ],
        publish_active=False,
    )

    active_candidates = pd.read_csv(active / "wizard_sweep_candidates.csv")
    attempt_candidates = pd.read_csv(result.paths["candidates"])
    assert result.summary["sweep_complete"] is True
    assert result.summary["observed_credit_delta"] == 10
    assert result.summary["credit_reconciliation_status"] == "PASS_RECONCILED"
    assert result.summary["published_active_snapshot"] is False
    assert result.summary["active_snapshot_preserved"] is True
    assert active_candidates["pair_id"].tolist() == [7]
    assert attempt_candidates["pair_id"].tolist() == [8]


def test_one_failed_cell_blocks_complete_discovery_but_preserves_other_evidence(tmp_path):
    def fake_prescanned(**kwargs):
        if kwargs["strategy"] == "Copula":
            raise CryptoWizardsFetchError("copula temporarily unavailable")
        return [{"symbol_1": "BTC-USD", "symbol_2": "ETH-USD"}]

    result, _authority = _run_authorized_sweep(
        root=tmp_path,
        api_key="secret",
        exchanges=("Dydx",),
        intervals=("Daily",),
        strategies=("Spread", "Copula"),
        credits_fetcher=_credit_usage_fetcher(0, 20),
        prescanned_fetcher=fake_prescanned,
    )

    manifest = pd.read_csv(result.paths["manifest"])
    assert result.summary["completed_cells"] == 1
    assert result.summary["failed_cells"] == 1
    assert result.summary["attempted_credits"] == 20
    assert result.summary["observed_credit_delta"] == 20
    assert result.summary["credit_reconciliation_status"] == "PASS_RECONCILED"
    assert result.summary["sweep_complete"] is False
    assert result.summary["discovery_authority"] == "blocked_partial_discovery"
    assert result.summary["blocker"] == "failed_requests:1"
    assert set(manifest["status"]) == {"completed", "failed"}
