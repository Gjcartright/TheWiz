from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from hashlib import sha256

import pandas as pd

from quant_platform.orchestration import corrective_release_gates as release_gates
from quant_platform.orchestration.corrective_live_parity_capture import (
    ALLOWED_INFO_REQUEST_TYPES,
    HYPERLIQUID_LIVE_INFO_URL,
    HYPERLIQUID_TESTNET_INFO_URL,
    capture_live_input_parity_evidence,
)
from quant_platform.orchestration.corrective_release_gates import (
    CANDIDATE_SCHEMA_VERSION,
    build_live_release_gates,
)
from tests.candidate_queue_support import seal_candidate_with_valid_queue
from tests.test_corrective_live_canary import _install_mock_stage6_release

NOW = datetime(2026, 8, 10, 12, tzinfo=UTC)
ADDRESS = "0x" + "1" * 40


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class _ReadOnlySession:
    def __init__(self, *, price_offset=0.0, malformed_l2=False):
        self.price_offset = price_offset
        self.malformed_l2 = malformed_l2
        self.calls = []

    def post(self, url, *, json, timeout):
        self.calls.append({"url": url, "json": dict(json), "timeout": timeout})
        request_type = json["type"]
        if request_type == "metaAndAssetCtxs":
            return _Response(
                [
                    {
                        "universe": [
                            {"name": "BTC", "szDecimals": 5, "maxLeverage": 40},
                            {"name": "ETH", "szDecimals": 4, "maxLeverage": 25},
                        ]
                    },
                    [
                        {"markPx": str(60_000 + self.price_offset), "funding": "0.00001"},
                        {"markPx": str(3_000 + self.price_offset), "funding": "-0.00002"},
                    ],
                ]
            )
        if request_type == "allMids":
            return _Response(
                {
                    "BTC": str(60_000 + self.price_offset),
                    "ETH": str(3_000 + self.price_offset),
                }
            )
        if request_type == "l2Book":
            if self.malformed_l2:
                return _Response({"levels": []})
            center = 60_000.0 if json["coin"] == "BTC" else 3_000.0
            center += self.price_offset
            return _Response(
                {
                    "coin": json["coin"],
                    "levels": [
                        [
                            {"px": str(center - 1.0), "sz": "1.0", "n": 1},
                            {"px": str(center - 2.0), "sz": "1.0", "n": 1},
                        ],
                        [
                            {"px": str(center + 1.0), "sz": "1.0", "n": 1},
                            {"px": str(center + 2.0), "sz": "1.0", "n": 1},
                        ],
                    ],
                }
            )
        if request_type == "clearinghouseState":
            return _Response(
                {
                    "marginSummary": {
                        "accountValue": "1000",
                        "totalMarginUsed": "0",
                    },
                    "withdrawable": "1000",
                    "crossMaintenanceMarginUsed": "0",
                    "assetPositions": [],
                }
            )
        raise AssertionError(f"unexpected request type: {request_type}")


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _policy():
    return {
        "schema_version": "thewiz.live_canary_policy.v2",
        "policy_version": "capture-test-v1",
        "effective_at_utc": NOW.isoformat(),
        "prerequisites": {
            "testnet_sample_sufficiency_pass": True,
            "supreme_team_testnet_checkpoint_pass": True,
            "shadow_testnet_live_input_parity_pass": True,
            "zero_unresolved_execution_incidents": True,
            "current_data_and_cost_evidence": True,
            "explicit_user_authorization_for_exact_order": True,
        },
        "canary": {
            "maximum_simultaneous_pairs": 1,
            "maximum_leverage": 1.0,
            "maximum_total_notional_usd": 25.0,
            "maximum_input_age_seconds": 60,
            "maximum_snapshot_skew_seconds": 5,
            "notional_policy": "smallest_venue_valid_notional",
            "authorization_reuse_allowed": False,
            "automatic_repeat_allowed": False,
            "automatic_scaling_allowed": False,
            "automatic_leverage_allowed": False,
        },
        "post_canary": {
            "reconciliation_required": True,
            "realized_cost_comparison_required": True,
            "supreme_team_review_required": True,
            "new_explicit_authorization_required_for_any_next_order": True,
        },
        "authorized_signer_address": "",
        "testnet_order_authority": False,
        "current_live_trading_authorized": False,
    }


def _candidate(root):
    receipt = {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "generated_at_utc": NOW.isoformat(),
        "candidate_status": "READY_FOR_NO_ORDER_PREFLIGHT",
        "candidate_experiment_id": "experiment-1",
        "candidate_leverage": 1.0,
        "pair_group_key": "BTC__ETH",
        "pair": "BTC-USD-ETH-USD",
        "asset_x": "BTC",
        "asset_y": "ETH",
        "timeframe": "daily",
        "exact_mode": "OU Optimal",
        "orientation": "original",
        "cost_model_id": "cost-1",
        "model_training_dataset_id": "dataset-1",
        "model_artifact_sha256": "a" * 64,
        "registered_learning_id": "learning-1",
        "registered_learning_receipt_sha256": "b" * 64,
        "registered_execution_id": "execution-1",
        "survivor_receipt_id": "survivor-1",
        "cadence_pass": True,
        "model_oos_authority_ready": True,
        "strict_cost_model_current": True,
        "hyperliquid_markets_current": True,
        "blockers": [],
        "order_submission_performed": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
        "source_artifact_hashes": {},
        "evidence_path": "",
    }
    sealed = seal_candidate_with_valid_queue(root=root, receipt=receipt)
    _write_json(root / "reports" / "active" / "testnet_candidate_receipt.json", sealed)
    return sealed


def _upstream(root):
    _candidate(root)
    sample = root / "reports" / "active" / "realized_testnet_sample_sufficiency.csv"
    sample.parent.mkdir(parents=True, exist_ok=True)
    sample.write_text("check,status\nall,PASS\n", encoding="utf-8")
    _write_json(
        root / "reports" / "supreme_team" / "testnet_evidence_checkpoint.json",
        {"checkpoint_status": "PASS"},
    )
    _write_json(root / "config" / "live_canary_policy.json", _policy())
    calculation = root / "src" / "parity_calculation.py"
    calculation.parent.mkdir(parents=True, exist_ok=True)
    calculation.write_text("# deterministic parity calculation\n", encoding="utf-8")
    return calculation


def test_upstream_block_makes_zero_hyperliquid_requests(tmp_path):
    testnet = _ReadOnlySession()
    live = _ReadOnlySession()

    result = capture_live_input_parity_evidence(
        root=tmp_path,
        now=NOW,
        account_address=ADDRESS,
        testnet_session=testnet,
        live_session=live,
        refresh_upstream=False,
        upstream_prevalidated=True,
    )

    assert result["status"] == "BLOCKED_UPSTREAM"
    assert testnet.calls == []
    assert live.calls == []
    assert not (tmp_path / "reports" / "active" / "testnet_live_input_parity_evidence.json").exists()


def test_unvalidated_upstream_bypass_is_rejected_before_network(tmp_path):
    calculation = _upstream(tmp_path)
    testnet = _ReadOnlySession()
    live = _ReadOnlySession()

    result = capture_live_input_parity_evidence(
        root=tmp_path,
        now=NOW,
        account_address=ADDRESS,
        testnet_session=testnet,
        live_session=live,
        calculation_artifact_path=calculation,
        refresh_upstream=False,
        upstream_prevalidated=False,
    )

    assert result["status"] == "BLOCKED_UPSTREAM"
    assert "stage6_upstream_evidence_not_revalidated" in result["blockers"]
    assert testnet.calls == []
    assert live.calls == []


def test_standalone_capture_rebuilds_stage6_before_network(tmp_path, monkeypatch):
    calculation = tmp_path / "src" / "parity_calculation.py"
    calls = []

    def rebuild(*, root, now):
        calls.append((root, now))
        _upstream(root)
        return {"checkpoint": {"checkpoint_status": "PASS"}}

    monkeypatch.setattr(
        release_gates,
        "build_testnet_supreme_team_checkpoint",
        rebuild,
    )
    testnet = _ReadOnlySession()
    live = _ReadOnlySession()

    result = capture_live_input_parity_evidence(
        root=tmp_path,
        now=NOW,
        account_address=ADDRESS,
        testnet_session=testnet,
        live_session=live,
        calculation_artifact_path=calculation,
    )

    assert calls == [(tmp_path, NOW)]
    assert result["status"] == "PASS"
    evidence = json.loads(result["evidence_path"].read_text())
    assert evidence["stage6_upstream_revalidated"] is True
    assert len(testnet.calls) == len(live.calls) == 5


def test_capture_uses_only_readonly_info_requests_and_proves_all_six_inputs(tmp_path):
    calculation = _upstream(tmp_path)
    testnet = _ReadOnlySession()
    live = _ReadOnlySession(price_offset=0.25)

    result = capture_live_input_parity_evidence(
        root=tmp_path,
        now=NOW,
        account_address=ADDRESS,
        testnet_session=testnet,
        live_session=live,
        calculation_artifact_path=calculation,
        refresh_upstream=False,
        upstream_prevalidated=True,
    )

    assert result["status"] == "PASS"
    assert result["order_submission_performed"] is False
    assert result["live_trading_authorized"] is False
    assert len(testnet.calls) == len(live.calls) == 5
    for session, expected_url in (
        (testnet, HYPERLIQUID_TESTNET_INFO_URL),
        (live, HYPERLIQUID_LIVE_INFO_URL),
    ):
        assert {call["url"] for call in session.calls} == {expected_url}
        assert {call["json"]["type"] for call in session.calls}.issubset(
            ALLOWED_INFO_REQUEST_TYPES
        )
        assert all("privateKey" not in call["json"] for call in session.calls)
    parity = pd.read_csv(result["parity_path"])
    assert len(parity) == 6
    assert parity["status"].eq("PASS").all()
    assert parity["source_artifacts_hash_bound"].all()
    evidence = json.loads(result["evidence_path"].read_text())
    assert len(evidence["inputs"]) == 6
    assert evidence["private_key_accessed"] is False
    assert evidence["order_submission_performed"] is False


def test_malformed_capture_fails_closed_without_replacing_prior_pass(tmp_path):
    calculation = _upstream(tmp_path)
    first = capture_live_input_parity_evidence(
        root=tmp_path,
        now=NOW,
        account_address=ADDRESS,
        testnet_session=_ReadOnlySession(),
        live_session=_ReadOnlySession(),
        calculation_artifact_path=calculation,
        refresh_upstream=False,
        upstream_prevalidated=True,
    )
    prior_hash = sha256(first["evidence_path"].read_bytes()).hexdigest()

    failed = capture_live_input_parity_evidence(
        root=tmp_path,
        now=NOW + timedelta(seconds=10),
        account_address=ADDRESS,
        testnet_session=_ReadOnlySession(malformed_l2=True),
        live_session=_ReadOnlySession(),
        calculation_artifact_path=calculation,
        refresh_upstream=False,
        upstream_prevalidated=True,
    )

    assert failed["status"] == "BLOCKED_CAPTURE_FAILED"
    assert sha256(first["evidence_path"].read_bytes()).hexdigest() == prior_hash
    status = json.loads(failed["status_path"].read_text())
    assert status["private_key_accessed"] is False
    assert status["order_submission_performed"] is False


def test_release_gate_automatically_captures_parity_after_stage6(tmp_path, monkeypatch):
    calculation = _upstream(tmp_path)
    candidate = json.loads(
        (tmp_path / "reports" / "active" / "testnet_candidate_receipt.json").read_text()
    )
    release, release_path = _install_mock_stage6_release(
        monkeypatch, tmp_path, candidate
    )
    release_pointer = tmp_path / "reports" / "active" / "stage6_release_status.json"
    _write_json(release_pointer, {"status": "PASS"})
    monkeypatch.setattr(
        release_gates,
        "build_stage6_release_receipt",
        lambda **_: {
            "status": "PASS",
            "pointer_path": release_pointer,
            "receipt": release,
            "receipt_path": release_path,
        },
    )
    sample_path = tmp_path / "reports" / "active" / "realized_testnet_sample_sufficiency.csv"
    supreme_path = tmp_path / "reports" / "supreme_team" / "testnet_evidence_checkpoint.json"
    sample_frame = pd.read_csv(sample_path)
    monkeypatch.setattr(
        release_gates,
        "build_testnet_sample_sufficiency",
        lambda **_: {"path": sample_path, "frame": sample_frame, "status": "PASS"},
    )
    monkeypatch.setattr(
        release_gates,
        "build_testnet_supreme_team_checkpoint",
        lambda **_: {
            "path": supreme_path,
            "checkpoint": {"checkpoint_status": "PASS"},
        },
    )
    testnet = _ReadOnlySession()
    live = _ReadOnlySession(price_offset=0.25)

    result = build_live_release_gates(
        root=tmp_path,
        now=NOW,
        parity_account_address=ADDRESS,
        parity_testnet_session=testnet,
        parity_live_session=live,
        parity_calculation_artifact_path=calculation,
    )

    capture_status = json.loads(result["capture_status"].read_text())
    assert capture_status["status"] == "PASS"
    assert capture_status["stage6_upstream_revalidated"] is True
    assert pd.read_csv(result["parity"])["status"].eq("PASS").all()
    assert len(testnet.calls) == len(live.calls) == 5
    authorization = json.loads(result["authorization"].read_text())
    assert authorization["live_trading_authorized"] is False
    assert authorization["order_submission_performed"] is False

    evidence_path = tmp_path / "reports" / "active" / "testnet_live_input_parity_evidence.json"
    evidence_hash = sha256(evidence_path.read_bytes()).hexdigest()
    reused_testnet = _ReadOnlySession()
    reused_live = _ReadOnlySession()
    reused = build_live_release_gates(
        root=tmp_path,
        now=NOW + timedelta(seconds=10),
        parity_account_address=ADDRESS,
        parity_testnet_session=reused_testnet,
        parity_live_session=reused_live,
        parity_calculation_artifact_path=calculation,
    )
    reused_status = json.loads(reused["capture_status"].read_text())
    assert reused_status["status"] == "PASS_REUSED_CURRENT"
    assert reused_testnet.calls == []
    assert reused_live.calls == []
    assert sha256(evidence_path.read_bytes()).hexdigest() == evidence_hash

    approval_path = result["approval"]
    committed_approval = json.loads(approval_path.read_text())
    committed_approval.update(
        {
            "approved": True,
            "approval_id": "committed-approval-must-not-be-rewritten",
            "wallet_signature": "committed-signature-must-not-be-rewritten",
        }
    )
    _write_json(approval_path, committed_approval)
    stale_testnet = _ReadOnlySession()
    stale_live = _ReadOnlySession()
    preserved = build_live_release_gates(
        root=tmp_path,
        now=NOW + timedelta(seconds=120),
        parity_account_address=ADDRESS,
        parity_testnet_session=stale_testnet,
        parity_live_session=stale_live,
        parity_calculation_artifact_path=calculation,
    )
    preserved_status = json.loads(preserved["capture_status"].read_text())
    assert preserved_status["status"] == "BLOCKED_COMMITTED_APPROVAL_PARITY_STALE"
    assert preserved_status["parity_receipt_preserved"] is True
    assert stale_testnet.calls == []
    assert stale_live.calls == []
    assert sha256(evidence_path.read_bytes()).hexdigest() == evidence_hash
    assert json.loads(approval_path.read_text()) == committed_approval


def test_release_gate_blocked_stage6_makes_zero_parity_requests(tmp_path, monkeypatch):
    sample_path = tmp_path / "reports" / "active" / "realized_testnet_sample_sufficiency.csv"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample_path.write_text("check,status\nall,BLOCKED\n", encoding="utf-8")
    sample_frame = pd.read_csv(sample_path)
    supreme_path = tmp_path / "reports" / "supreme_team" / "testnet_evidence_checkpoint.json"
    _write_json(supreme_path, {"checkpoint_status": "BLOCKED"})
    _write_json(tmp_path / "config" / "live_canary_policy.json", _policy())
    monkeypatch.setattr(
        release_gates,
        "build_testnet_sample_sufficiency",
        lambda **_: {"path": sample_path, "frame": sample_frame, "status": "BLOCKED"},
    )
    monkeypatch.setattr(
        release_gates,
        "build_testnet_supreme_team_checkpoint",
        lambda **_: {
            "path": supreme_path,
            "checkpoint": {"checkpoint_status": "BLOCKED"},
        },
    )
    testnet = _ReadOnlySession()
    live = _ReadOnlySession()

    result = build_live_release_gates(
        root=tmp_path,
        now=NOW,
        parity_account_address=ADDRESS,
        parity_testnet_session=testnet,
        parity_live_session=live,
    )

    capture_status = json.loads(result["capture_status"].read_text())
    assert capture_status["status"] == "BLOCKED_UPSTREAM"
    assert testnet.calls == []
    assert live.calls == []
