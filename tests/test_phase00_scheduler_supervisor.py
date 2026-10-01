from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path

import pytest

from quant_platform import cli as quant_cli
from quant_platform.crypto_wizards_sweep import (
    run_authorized_wizard_discovery_sweep,
)
from quant_platform.execution import (
    refresh_hyperliquid_testnet_market_inventory,
)
from quant_platform.orchestration.corrective_external_effects import (
    current_external_effect_issuer,
)
from quant_platform.orchestration.corrective_runtime import (
    launch_agent_runtime_environment,
    scheduler_contract,
    scheduler_run_identity,
)
from quant_platform.orchestration.corrective_scheduler_supervisor import (
    intended_scheduler_slot,
    recover_abandoned_scheduler_runs,
    supervise_scheduler_run,
)
from quant_platform.orchestration.corrective_scheduler_terminal import (
    build_scheduler_run_intent,
    new_scheduler_run_id,
    publish_scheduler_run_intent,
)
from quant_platform.orchestration.current_wizard_hyperliquid_cadence import (
    MINIMUM_FREE_BYTES,
)
from quant_platform.orchestration.current_wizard_hyperliquid_daily_runner import (
    run_current_wizard_hyperliquid_daily_pipeline,
)
from quant_platform.orchestration.effect_authority import (
    PHASE00_WIZARD_RESEARCH_PROFILE,
    EffectAuthority,
    EffectKind,
    EffectRequest,
    publication_authority_session,
)

NOW = datetime(2026, 8, 22, 12, 7, 42, tzinfo=UTC)


def _prepare_root(root: Path) -> None:
    (root / "src" / "quant_platform").mkdir(parents=True, exist_ok=True)
    (root / "src" / "quant_platform" / "fixture.py").write_text(
        "VALUE = 1\n",
        encoding="utf-8",
    )
    (root / "pyproject.toml").write_text(
        "[project]\nname='fixture'\n",
        encoding="utf-8",
    )
    (root / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    policy_source = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "phase00_external_effect_policy.json"
    )
    policy_target = root / "config" / "phase00_external_effect_policy.json"
    policy_target.parent.mkdir(parents=True, exist_ok=True)
    policy_target.write_bytes(policy_source.read_bytes())


def _launchd_environment(root: Path, monkeypatch: pytest.MonkeyPatch, key: str) -> None:
    contract = scheduler_contract(key)
    environment = launch_agent_runtime_environment(root, contract=contract)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)


def _runtime_identity(root: Path, key: str = "hyperliquid_l2") -> dict:
    contract = scheduler_contract(key)
    return scheduler_run_identity(
        root,
        contract=contract,
        environment=launch_agent_runtime_environment(root, contract=contract),
        require_launchd=True,
    )


def _recovery_authority(root: Path, *, at: datetime) -> EffectAuthority:
    return EffectAuthority(
        root=root,
        secret=b"recovery-test-authority-secret-32-bytes",
        issuer_id="scheduler-recovery-test",
        profile=PHASE00_WIZARD_RESEARCH_PROFILE,
        clock=lambda: at,
    )


def test_interval_and_calendar_slots_are_deterministic() -> None:
    interval = intended_scheduler_slot(
        scheduler_contract("hyperliquid_l2"),
        now=NOW,
    )
    daily = intended_scheduler_slot(
        scheduler_contract("daily_research"),
        now=NOW,
    )
    assert interval == "2026-08-22T12:05:00+00:00/PT300S"
    assert daily == "2026-08-22T10:15:00+00:00/P1D"


def test_missing_launchd_provenance_blocks_before_callback(tmp_path: Path) -> None:
    _prepare_root(tmp_path)
    calls: list[str] = []
    result = supervise_scheduler_run(
        root=tmp_path,
        contract_key="hyperliquid_l2",
        publication_scope="public_l2",
        callback=lambda: calls.append("called"),
        now=NOW,
        require_launchd_provenance=True,
    )
    assert calls == []
    assert result.exit_code == 2
    assert result.terminal_receipt["terminal_status"] == "BLOCKED"
    assert "scheduler_launchd_provenance_missing" in result.terminal_receipt["blockers"]
    assert result.terminal_paths["terminal_receipt"].is_file()


def test_valid_provenance_passes_and_publishes_terminal_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_root(tmp_path)
    _launchd_environment(tmp_path, monkeypatch, "hyperliquid_l2")

    def callback() -> dict:
        return {
            "summary": {
                "status": "PASS",
                "external_calls": 0,
                "promotion_authority": False,
                "live_trading_authorized": False,
            },
            "paths": {},
        }

    result = supervise_scheduler_run(
        root=tmp_path,
        contract_key="hyperliquid_l2",
        publication_scope="public_l2",
        callback=callback,
        now=NOW,
    )
    assert result.exit_code == 0
    assert result.terminal_receipt["terminal_status"] == "PASS"
    assert result.terminal_receipt["intended_slot_credit"] is True
    assert result.terminal_receipt["order_submissions"] == 0


def test_same_scheduler_slot_executes_once_and_second_run_gets_no_credit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_root(tmp_path)
    _launchd_environment(tmp_path, monkeypatch, "hyperliquid_l2")
    calls: list[str] = []

    def callback() -> dict:
        calls.append("called")
        return {
            "status": "PASS",
            "promotion_authority": False,
            "live_trading_authorized": False,
        }

    first = supervise_scheduler_run(
        root=tmp_path,
        contract_key="hyperliquid_l2",
        publication_scope="public_l2",
        callback=callback,
        now=NOW,
    )
    second = supervise_scheduler_run(
        root=tmp_path,
        contract_key="hyperliquid_l2",
        publication_scope="public_l2",
        callback=callback,
        now=NOW,
    )

    assert calls == ["called"]
    assert first.terminal_receipt["intended_slot_credit"] is True
    assert second.terminal_receipt["terminal_status"] == "DEFERRED"
    assert second.terminal_receipt["intended_slot_credit"] is False
    assert second.terminal_receipt["blockers"] == ["scheduler_intended_slot_already_claimed"]
    receipts = list(
        (tmp_path / "data" / "research" / "scheduler_terminal_receipts").rglob("*.json")
    )
    assert len(receipts) == 2


def test_maintenance_defers_work_but_terminal_lane_survives(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_root(tmp_path)
    _launchd_environment(tmp_path, monkeypatch, "hyperliquid_l2")
    marker = tmp_path / ".runtime_control" / "phase00_maintenance.json"
    marker.parent.mkdir(exist_ok=True)
    marker.write_text(
        json.dumps({"maintenance_id": "maintenance_test", "status": "ACTIVE"}),
        encoding="utf-8",
    )
    calls: list[str] = []
    result = supervise_scheduler_run(
        root=tmp_path,
        contract_key="hyperliquid_l2",
        publication_scope="public_l2",
        callback=lambda: calls.append("called"),
        now=NOW,
    )
    assert calls == []
    assert result.terminal_receipt["terminal_status"] == "DEFERRED"
    assert result.terminal_receipt["intended_slot_credit"] is False
    assert result.terminal_paths["terminal_receipt"].is_file()


def test_callback_crash_is_receipted_then_reraised(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_root(tmp_path)
    _launchd_environment(tmp_path, monkeypatch, "hyperliquid_l2")

    def crash() -> None:
        raise RuntimeError("private detail must not enter receipt")

    with pytest.raises(RuntimeError, match="private detail"):
        supervise_scheduler_run(
            root=tmp_path,
            contract_key="hyperliquid_l2",
            publication_scope="public_l2",
            callback=crash,
            now=NOW,
        )
    receipts = list(
        (tmp_path / "data" / "research" / "scheduler_terminal_receipts").rglob("*.json")
    )
    assert len(receipts) == 1
    receipt = json.loads(receipts[0].read_text(encoding="utf-8"))
    assert receipt["terminal_status"] == "FAILED"
    assert receipt["blockers"] == ["scheduler_callback_crash:RuntimeError"]


def test_scheduler_result_cannot_claim_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_root(tmp_path)
    _launchd_environment(tmp_path, monkeypatch, "hyperliquid_l2")
    result = supervise_scheduler_run(
        root=tmp_path,
        contract_key="hyperliquid_l2",
        publication_scope="public_l2",
        callback=lambda: {
            "status": "PASS",
            "testnet_order_authority": True,
        },
        now=NOW,
    )
    assert result.exit_code == 2
    assert result.terminal_receipt["terminal_status"] == "FAILED"
    assert "scheduler_result_claimed_prohibited_authority" in (result.terminal_receipt["blockers"])


def test_wizard_supervisor_installs_no_order_external_issuer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_root(tmp_path)
    _launchd_environment(tmp_path, monkeypatch, "wizard_proof")
    observed: dict[str, object] = {}

    def callback() -> dict:
        issuer = current_external_effect_issuer()
        assert issuer is not None
        observed["provider_id"] = issuer.provider_id
        observed["profile"] = issuer.authority.profile.name
        observed["order_allowed"] = EffectKind.ORDER_SUBMISSION in (
            issuer.authority.profile.allowed_effects
        )
        observed["account_scope_id"] = issuer.account_scope_id
        observed["allowed_targets"] = issuer.allowed_targets
        observed["allowed_credential_keys"] = issuer.allowed_credential_keys
        observed["max_total_requests"] = issuer.max_total_requests
        observed["max_total_credits"] = issuer.max_total_credits
        observed["policy_version"] = issuer.policy_version
        return {
            "status": "PASS",
            "external_calls": 0,
            "external_credits_reserved": 0,
            "external_credits_consumed": 0,
            "promotion_authority": False,
            "live_trading_authorized": False,
        }

    result = supervise_scheduler_run(
        root=tmp_path,
        contract_key="wizard_proof",
        publication_scope="wizard_external_research",
        callback=callback,
        now=NOW,
    )

    assert result.terminal_receipt["terminal_status"] == "PASS"
    assert observed == {
        "provider_id": "crypto_wizards",
        "profile": "PHASE00_WIZARD_RESEARCH_NO_ORDER",
        "order_allowed": False,
        "account_scope_id": "crypto_wizards:research",
        "allowed_targets": frozenset(
            {
                "https://api.cryptowizards.net/v1beta/backtest",
                "https://api.cryptowizards.net/v1beta/cointegration",
                "https://api.cryptowizards.net/v1beta/copula",
                "https://api.cryptowizards.net/v1beta/correlations",
                "https://api.cryptowizards.net/v1beta/credits-used",
                "https://api.cryptowizards.net/v1beta/prescanned",
                "https://api.cryptowizards.net/v1beta/spread",
                "https://api.cryptowizards.net/v1beta/zscores",
            }
        ),
        "allowed_credential_keys": frozenset({"CRYPTO_WIZARDS_API_KEY"}),
        "max_total_requests": 2048,
        "max_total_credits": 1000,
        "policy_version": (
            "thewiz.phase00_external_effect_policy.v1:sha256:"
            + result.result_summary["external_effect_policy_sha256"]
        ),
    }
    assert result.result_summary["external_effect_policy_status"] == "PASS"


def test_daily_supervisor_installs_exact_wizard_and_hyperliquid_provider_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_root(tmp_path)
    _launchd_environment(tmp_path, monkeypatch, "daily_research")
    observed: dict[str, object] = {}

    def callback() -> dict[str, object]:
        assert current_external_effect_issuer() is None
        wizard = current_external_effect_issuer("crypto_wizards")
        hyperliquid = current_external_effect_issuer("hyperliquid_public")
        assert wizard is not None
        assert hyperliquid is not None
        observed["wizard_contracts"] = {
            (
                row.operation,
                row.method,
                row.target,
                row.credit_units_per_request,
            )
            for row in wizard.allowed_call_contracts
        }
        observed["hyperliquid_contracts"] = {
            (
                row.operation,
                row.method,
                row.target,
                row.credit_units_per_request,
            )
            for row in hyperliquid.allowed_call_contracts
        }
        observed["hyperliquid_credentials"] = (
            hyperliquid.allowed_credential_keys
        )
        return {
            "status": "PASS",
            "external_calls": 0,
            "external_credits_reserved": 0,
            "external_credits_consumed": 0,
            "promotion_authority": False,
            "live_trading_authorized": False,
        }

    result = supervise_scheduler_run(
        root=tmp_path,
        contract_key="daily_research",
        publication_scope="wizard_external_research",
        callback=callback,
        now=NOW,
    )

    assert result.exit_code == 0
    assert result.terminal_receipt["terminal_status"] == "PASS"
    assert result.result_summary["external_effect_provider_ids"] == [
        "crypto_wizards",
        "hyperliquid_public",
    ]
    assert observed["hyperliquid_credentials"] == frozenset()
    assert {
        (
            "HYPERLIQUID_TESTNET_META",
            "POST",
            "https://api.hyperliquid-testnet.xyz/info",
            0,
        ),
        (
            "HYPERLIQUID_TESTNET_ALL_MIDS",
            "POST",
            "https://api.hyperliquid-testnet.xyz/info",
            0,
        ),
    } <= observed["hyperliquid_contracts"]
    assert len(observed["hyperliquid_contracts"]) == 11
    assert all(
        contract[3] == 0
        for contract in observed["hyperliquid_contracts"]
    )
    assert len(observed["wizard_contracts"]) == 14


def test_supervised_daily_run_reconciles_both_provider_lanes_without_orders(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CRYPTO_WIZARDS_API_KEY", raising=False)
    _prepare_root(tmp_path)
    _launchd_environment(tmp_path, monkeypatch, "daily_research")
    monkeypatch.setattr(quant_cli, "ROOT", tmp_path)
    credential = tmp_path / ".env.local"
    credential.write_text(
        "CRYPTO_WIZARDS_API_KEY=fixture-secret\n",
        encoding="utf-8",
    )
    credential.chmod(0o600)
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True, exist_ok=True)
    (active / "current_wizard_hyperliquid_pair_history_queue.csv").write_text(
        "pair_group_key,history_request_status\n"
        "binance|daily|BTC|ETH,READY_TO_FETCH\n",
        encoding="utf-8",
    )
    credit_readings = iter((0, 300))
    local_stage_calls: list[str] = []
    hyperliquid_transport_calls: list[str] = []

    def wizard_stage(**kwargs):
        return run_authorized_wizard_discovery_sweep(
            **kwargs,
            credits_fetcher=lambda **_: {
                "credits_used": next(credit_readings),
                "credit_limit": 1000,
            },
            prescanned_fetcher=lambda **_: {"pairs": []},
        )

    def hyperliquid_fetch(payload, **_):
        request_type = str(payload["type"])
        hyperliquid_transport_calls.append(request_type)
        if request_type == "meta":
            return {
                "universe": [
                    {
                        "name": "BTC",
                        "szDecimals": 5,
                        "maxLeverage": 40,
                        "marginTableId": 0,
                        "isDelisted": False,
                    }
                ],
                "marginTables": [],
            }
        return {"BTC": "60000"}

    def fake_cli_main(argv=None, *, load_environment=True):
        command = str((argv or [""])[0])
        local_stage_calls.append(command)
        assert load_environment is False
        if command == "hyperliquid-testnet-market-inventory":
            frame = refresh_hyperliquid_testnet_market_inventory(
                root=tmp_path,
                now=NOW,
                post_json_fetcher=hyperliquid_fetch,
            )
            print(
                json.dumps(
                    {
                        "summary": {
                            "rows": len(frame),
                            "research_only": True,
                        },
                        "paths": {
                            "inventory": str(
                                active / "hyperliquid_testnet_market_inventory.csv"
                            ),
                            "inventory_evidence": str(
                                active
                                / "hyperliquid_testnet_market_inventory_evidence.json"
                            ),
                        },
                    }
                )
            )
            return
        print(json.dumps({"summary": {"research_only": True}, "paths": {}}))

    monkeypatch.setattr(quant_cli, "main", fake_cli_main)

    def callback():
        daily = run_current_wizard_hyperliquid_daily_pipeline(
            root=tmp_path,
            execute=True,
            now=NOW,
            available_disk_bytes=MINIMUM_FREE_BYTES + 1,
            wizard_stage_runner=wizard_stage,
            semantic_evidence_builder=lambda **_: {
                "status": "PASS",
                "blocker": "",
                "evidence_path": "reports/runs/test/semantic.json",
                "evidence_sha256": "a" * 64,
            },
            stage_timeout_seconds=2,
        )
        return {
            "summary": {
                **daily.summary,
                "status": "PASS"
                if daily.summary["run_status"] == "PASS"
                else "FAILED",
                "blockers": [],
            },
            "paths": daily.paths,
        }

    result = supervise_scheduler_run(
        root=tmp_path,
        contract_key="daily_research",
        publication_scope="wizard_external_research",
        callback=callback,
        now=NOW,
    )

    terminal = result.terminal_receipt
    hyperliquid_evidence = json.loads(
        (
            active / "hyperliquid_testnet_market_inventory_evidence.json"
        ).read_text(encoding="utf-8")
    )
    assert result.exit_code == 0
    assert terminal["terminal_status"] == "PASS"
    assert terminal["external_calls"] == 34
    assert terminal["external_credits_reserved"] == 300
    assert terminal["external_credits_consumed"] == 300
    assert terminal["external_credits_reconciled"] == 300
    assert terminal["order_attempts"] == 0
    assert terminal["order_submissions"] == 0
    assert result.result_summary["external_effect_provider_ids"] == [
        "crypto_wizards",
        "hyperliquid_public",
    ]
    assert result.result_summary["wizard_api_credential_source"] == ".env.local"
    assert hyperliquid_transport_calls == ["meta", "allMids"]
    assert hyperliquid_evidence["external_requests"] == 2
    assert hyperliquid_evidence["external_credits"] == 0
    assert hyperliquid_evidence["order_submission_included"] is False
    assert hyperliquid_evidence["live_trading_authorized"] is False
    assert len(local_stage_calls) == 18


def test_wizard_supervisor_missing_effect_policy_blocks_before_slot_and_callback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_root(tmp_path)
    (tmp_path / "config" / "phase00_external_effect_policy.json").unlink()
    _launchd_environment(tmp_path, monkeypatch, "wizard_proof")
    calls: list[str] = []

    result = supervise_scheduler_run(
        root=tmp_path,
        contract_key="wizard_proof",
        publication_scope="wizard_external_research",
        callback=lambda: calls.append("called"),
        now=NOW,
    )

    assert calls == []
    assert result.terminal_receipt["terminal_status"] == "BLOCKED"
    assert result.terminal_receipt["intended_slot_credit"] is False
    assert result.terminal_receipt["blockers"] == [
        "phase00_external_effect_policy_missing"
    ]
    assert result.result_summary["external_effect_policy_status"] == "BLOCKED"
    slot_claims = (
        tmp_path / "data" / "research" / "scheduler_slot_claims" / "wizard_proof"
    )
    assert not slot_claims.exists()


def test_abandoned_intent_without_slot_or_effects_requires_manual_reauthorization(
    tmp_path: Path,
) -> None:
    _prepare_root(tmp_path)
    started = NOW - timedelta(minutes=5)
    identity = _runtime_identity(tmp_path)
    run_id = new_scheduler_run_id(scheduler_key="hyperliquid_l2", now=started)
    intended_slot = intended_scheduler_slot(
        scheduler_contract("hyperliquid_l2"),
        now=started,
    )
    authority = _recovery_authority(tmp_path, at=started + timedelta(seconds=30))
    with publication_authority_session(
        authority=authority,
        run_id=run_id,
        intended_slot_id=intended_slot,
        policy_version="test.recovery.v1",
        source_fingerprint_sha256=identity["source_fingerprint_sha256"],
        runtime_fingerprint_sha256=identity["runtime_contract_sha256"],
        configuration_fingerprint_sha256=identity[
            "configuration_fingerprint_sha256"
        ],
        allowed_scopes=frozenset({"scheduler_terminal"}),
        allowed_target_prefixes=(tmp_path.resolve(),),
        max_total_bytes=1024**2,
    ):
        publish_scheduler_run_intent(
            tmp_path,
            build_scheduler_run_intent(
                run_id=run_id,
                intended_slot=intended_slot,
                runtime_identity=identity,
                started_at=started,
            ),
        )

    result = recover_abandoned_scheduler_runs(
        root=tmp_path,
        scheduler_key="hyperliquid_l2",
        authority=authority,
        recovered_at=NOW,
        minimum_age_seconds=0,
        owner_matcher=lambda _intent: False,
    )

    assert result.blockers == ()
    assert len(result.recovered_receipts) == 1
    receipt = result.recovered_receipts[0]
    assert receipt["terminal_status"] == "CRASH_RECOVERED"
    assert receipt["retryable"] is False
    assert "scheduler_crash_retry_requires_manual_reauthorization" in receipt["blockers"]
    assert receipt["intended_slot_credit"] is False
    assert receipt["order_submissions"] == 0


@pytest.mark.parametrize("absence_proof", [None, False, "true"])
def test_abandoned_reservation_without_absence_proof_remains_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    absence_proof: object,
) -> None:
    _prepare_root(tmp_path)
    started = NOW - timedelta(minutes=5)
    identity = _runtime_identity(tmp_path)
    run_id = new_scheduler_run_id(scheduler_key="hyperliquid_l2", now=started)
    intended_slot = intended_scheduler_slot(
        scheduler_contract("hyperliquid_l2"),
        now=started,
    )
    authority = _recovery_authority(tmp_path, at=started + timedelta(seconds=30))
    with publication_authority_session(
        authority=authority,
        run_id=run_id,
        intended_slot_id=intended_slot,
        policy_version="test.recovery.v1",
        source_fingerprint_sha256=identity["source_fingerprint_sha256"],
        runtime_fingerprint_sha256=identity["runtime_contract_sha256"],
        configuration_fingerprint_sha256=identity[
            "configuration_fingerprint_sha256"
        ],
        allowed_scopes=frozenset({"scheduler_terminal"}),
        allowed_target_prefixes=(tmp_path.resolve(),),
        max_total_bytes=1024**2,
    ):
        publish_scheduler_run_intent(
            tmp_path,
            build_scheduler_run_intent(
                run_id=run_id,
                intended_slot=intended_slot,
                runtime_identity=identity,
                started_at=started,
            ),
        )
    reservation_sha256 = sha256(b"reservation-only-crash").hexdigest()
    authority.register_external_reservation(
        run_id=run_id,
        intended_slot_id=intended_slot,
        provider_id="crypto_wizards",
        account_scope_id="wizard-research",
        reservation_id="reservation-only-crash",
        reservation_sha256=reservation_sha256,
        max_total_requests=3,
        max_total_credits=10,
    )
    original_accounting = authority.run_accounting
    if absence_proof is not None:
        def accounting_with_unproven_absence(**kwargs: object) -> dict[str, object]:
            return {
                **original_accounting(**kwargs),
                "provider_effect_absence_proven": absence_proof,
            }

        monkeypatch.setattr(authority, "run_accounting", accounting_with_unproven_absence)

    result = recover_abandoned_scheduler_runs(
        root=tmp_path,
        scheduler_key="hyperliquid_l2",
        authority=authority,
        recovered_at=NOW,
        minimum_age_seconds=0,
        owner_matcher=lambda _intent: False,
    )

    receipt = result.recovered_receipts[0]
    accounting = original_accounting(
        run_id=run_id,
        intended_slot_id=intended_slot,
    )
    assert receipt["retryable"] is False
    assert "scheduler_crash_retry_requires_manual_reauthorization" in receipt["blockers"]
    assert accounting["open_reservations"] == 1
    assert accounting["failed_reservations"] == 0
    assert "external_reservation_open" in receipt["blockers"]
    assert "external_reservation_abandoned_before_provider_effect" not in receipt["blockers"]
    assert authority.external_reservation_retry_safe(
        reservation_id="reservation-only-crash",
        reservation_sha256=reservation_sha256,
    ) is False


def test_abandoned_effects_are_recovered_with_exact_nonretryable_accounting(
    tmp_path: Path,
) -> None:
    _prepare_root(tmp_path)
    started = NOW - timedelta(minutes=5)
    identity = _runtime_identity(tmp_path)
    run_id = new_scheduler_run_id(scheduler_key="hyperliquid_l2", now=started)
    intended_slot = intended_scheduler_slot(
        scheduler_contract("hyperliquid_l2"),
        now=started,
    )
    authority = _recovery_authority(tmp_path, at=started + timedelta(seconds=30))
    with publication_authority_session(
        authority=authority,
        run_id=run_id,
        intended_slot_id=intended_slot,
        policy_version="test.recovery.v1",
        source_fingerprint_sha256=identity["source_fingerprint_sha256"],
        runtime_fingerprint_sha256=identity["runtime_contract_sha256"],
        configuration_fingerprint_sha256=identity[
            "configuration_fingerprint_sha256"
        ],
        allowed_scopes=frozenset({"scheduler_terminal"}),
        allowed_target_prefixes=(tmp_path.resolve(),),
        max_total_bytes=1024**2,
    ):
        publish_scheduler_run_intent(
            tmp_path,
            build_scheduler_run_intent(
                run_id=run_id,
                intended_slot=intended_slot,
                runtime_identity=identity,
                started_at=started,
            ),
        )
    authority.register_external_reservation(
        run_id=run_id,
        intended_slot_id=intended_slot,
        provider_id="crypto_wizards",
        account_scope_id="wizard-research",
        reservation_id="abandoned-reservation",
        reservation_sha256=sha256(b"abandoned-reservation").hexdigest(),
        max_total_requests=3,
        max_total_credits=10,
    )
    common = {
        "run_id": run_id,
        "intended_slot_id": intended_slot,
        "target": "https://api.cryptowizards.net/v1beta/backtest",
        "operation": "post:backtest_post",
        "effect_scope": "credit:abandoned-reservation",
        "payload_sha256": "d" * 64,
        "policy_version": "test.v1",
        "source_fingerprint_sha256": identity["source_fingerprint_sha256"],
        "runtime_fingerprint_sha256": identity["runtime_contract_sha256"],
        "configuration_fingerprint_sha256": identity[
            "configuration_fingerprint_sha256"
        ],
        "account_scope_id": "wizard-research",
    }
    network = authority.issue(
        **common,
        effect_kind=EffectKind.AUTHENTICATED_NETWORK,
        max_units=1,
    )
    credit = authority.issue(
        **common,
        effect_kind=EffectKind.CREDIT_SPEND,
        max_units=2,
    )
    authority.consume(network, EffectRequest.from_permit(network, actual_units=1))
    authority.consume(credit, EffectRequest.from_permit(credit, actual_units=2))
    accounting = authority.run_accounting(
        run_id=run_id,
        intended_slot_id=intended_slot,
    )

    result = recover_abandoned_scheduler_runs(
        root=tmp_path,
        scheduler_key="hyperliquid_l2",
        authority=authority,
        recovered_at=NOW,
        minimum_age_seconds=0,
        owner_matcher=lambda _intent: False,
    )

    receipt = result.recovered_receipts[0]
    assert receipt["retryable"] is False
    assert receipt["external_calls"] == 1
    assert receipt["external_credits_reserved"] == 10
    assert receipt["external_credits_consumed"] == 2
    assert accounting["open_reservations"] == 1
    assert "external_reservation_open" in accounting["blockers"]
    assert "effect_outcome_unknown_or_in_flight" in accounting["blockers"]
    assert "external_reservation_open" in receipt["blockers"]
    assert "effect_outcome_unknown_or_in_flight" in receipt["blockers"]
    assert "external_credit_reconciliation_required" in receipt["blockers"]
