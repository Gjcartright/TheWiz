from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from hashlib import sha256

import pandas as pd
from eth_account import Account
from eth_account.messages import encode_defunct

from quant_platform.orchestration import corrective_live_canary_executor as executor_preflight
from quant_platform.orchestration import corrective_release_gates as release_gates
from quant_platform.orchestration.corrective_live_canary import (
    PARITY_ARTIFACT_SCHEMA_VERSION,
    PARITY_INPUT_CONTRACTS,
    PARITY_SCHEMA_VERSION,
    REQUIRED_PARITY_INPUTS,
    _canonical_json,
    _payload_hash,
    _policy_id,
    build_live_canary_control_plane,
)
from quant_platform.orchestration.corrective_live_canary_execution import (
    live_canary_executor_contract,
)
from quant_platform.orchestration.corrective_live_canary_outcome import (
    EXECUTION_SCHEMA_VERSION,
    REVIEW_SCHEMA_VERSION,
    _identity,
    evaluate_live_canary_outcome,
)
from quant_platform.orchestration.corrective_release_gates import (
    CANDIDATE_SCHEMA_VERSION,
)
from tests.candidate_queue_support import seal_candidate_with_valid_queue

NOW = datetime(2026, 8, 10, 12, tzinfo=UTC)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _policy(signer=""):
    return {
        "schema_version": "thewiz.live_canary_policy.v2",
        "policy_version": "test-v1",
        "effective_at_utc": "2026-08-10T00:00:00Z",
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
        "authorized_signer_address": signer,
        "testnet_order_authority": False,
        "current_live_trading_authorized": False,
    }


def _candidate():
    return {
        "candidate_status": "READY_FOR_NO_ORDER_PREFLIGHT",
        "candidate_receipt_id": "testnetcandidate_accepted",
        "receipt_sha256": "a" * 64,
        "pair": "BTC-USD-ETH-USD",
        "asset_x": "BTC",
        "asset_y": "ETH",
    }


def _valid_candidate(root, *, generated_at=NOW):
    receipt = {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "generated_at_utc": generated_at.isoformat(),
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
    candidate = seal_candidate_with_valid_queue(root=root, receipt=receipt)
    _write_json(
        root / "reports" / "active" / "testnet_candidate_receipt.json",
        candidate,
    )
    return candidate


def _write_upstream(root):
    sample = root / "reports" / "active" / "realized_testnet_sample_sufficiency.csv"
    supreme = root / "reports" / "supreme_team" / "testnet_evidence_checkpoint.json"
    sample.parent.mkdir(parents=True, exist_ok=True)
    sample.write_text("check,status\nall,PASS\n", encoding="utf-8")
    _write_json(supreme, {"checkpoint_status": "PASS"})
    return sample, supreme


def _install_mock_stage6_release(monkeypatch, root, candidate):
    release = {
        "schema_version": release_gates.STAGE6_RELEASE_RECEIPT_SCHEMA_VERSION,
        "stage6_release_receipt_id": "stage6release_test_fixture",
        "candidate_receipt_id": candidate["candidate_receipt_id"],
        "candidate_receipt_sha256": candidate["receipt_sha256"],
        "stage6_release_authority": True,
        "order_submission_performed": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    release["receipt_sha256"] = _payload_hash(release)
    path = (
        root
        / "data"
        / "testnet"
        / "stage6_releases"
        / "stage6release_test_fixture.json"
    )
    _write_json(path, release)
    monkeypatch.setattr(
        release_gates,
        "validate_stage6_release_evidence",
        lambda **_: (True, release, path, []),
    )
    return release, path


def _write_parity(root, policy, candidate):
    inputs = []
    for input_index, name in enumerate(REQUIRED_PARITY_INPUTS, start=1):
        calculation = root / "data" / "parity" / f"calculation-{name}.txt"
        calculation.parent.mkdir(parents=True, exist_ok=True)
        calculation.write_text(f"normalized parity calculation for {name}\n", encoding="utf-8")
        calculation_hash = sha256(calculation.read_bytes()).hexdigest()
        row = {"input": name}
        for environment, age_seconds, value_offset in (
            ("testnet", 10, 0.0),
            ("live", 9, 0.01),
        ):
            captured_at = (NOW - timedelta(seconds=age_seconds)).isoformat()
            artifact = root / "data" / "parity" / f"{name}-{environment}.json"
            snapshot = {
                "schema_version": PARITY_ARTIFACT_SCHEMA_VERSION,
                "input": name,
                "environment": environment,
                "candidate_receipt_id": candidate["candidate_receipt_id"],
                "captured_at_utc": captured_at,
                "values": {
                    field: float(input_index + field_index / 10 + value_offset)
                    for field_index, field in enumerate(PARITY_INPUT_CONTRACTS[name], start=1)
                },
                "units": PARITY_INPUT_CONTRACTS[name],
                "calculation_artifact_path": str(calculation.relative_to(root)),
                "calculation_artifact_sha256": calculation_hash,
                "source_artifacts": _write_raw_parity_sources(
                    root,
                    input_name=name,
                    environment=environment,
                    captured_at=captured_at,
                ),
                "capture_complete": True,
                "private_key_accessed": False,
                "order_submission_performed": False,
                "live_trading_authorized": False,
            }
            snapshot["receipt_sha256"] = _payload_hash(snapshot)
            _write_json(artifact, snapshot)
            row[f"{environment}_timestamp_utc"] = captured_at
            row[f"{environment}_artifact_path"] = str(artifact.relative_to(root))
            row[f"{environment}_artifact_sha256"] = sha256(artifact.read_bytes()).hexdigest()
        inputs.append(row)
    core = {
        "schema_version": PARITY_SCHEMA_VERSION,
        "captured_at_utc": NOW.isoformat(),
        "candidate_receipt_id": candidate["candidate_receipt_id"],
        "live_canary_policy_id": _policy_id(policy),
        "inputs": inputs,
        "order_submission_performed": False,
        "live_trading_authorized": False,
    }
    evidence = {
        **core,
        "parity_receipt_id": "liveinputparity_" + _payload_hash(core)[:20],
    }
    evidence["receipt_sha256"] = _payload_hash(evidence)
    _write_json(
        root / "reports" / "active" / "testnet_live_input_parity_evidence.json",
        evidence,
    )
    return evidence


def _write_raw_parity_sources(root, *, input_name, environment, captured_at):
    request_types = {
        "market_metadata": ("metaAndAssetCtxs",),
        "mark_and_mid_prices": ("metaAndAssetCtxs", "allMids"),
        "funding_rate_and_timestamp": ("metaAndAssetCtxs",),
        "l2_depth_and_slippage": ("l2Book", "l2Book"),
        "size_precision_and_minimum_notional": ("metaAndAssetCtxs",),
        "margin_and_liquidation_inputs": ("clearinghouseState",),
    }[input_name]
    info_url = (
        "https://api.hyperliquid-testnet.xyz/info"
        if environment == "testnet"
        else "https://api.hyperliquid.xyz/info"
    )
    sources = {}
    for index, request_type in enumerate(request_types):
        name = f"source_{index}_{request_type}"
        request = {"type": request_type}
        if request_type == "l2Book":
            request["coin"] = ("BTC", "ETH")[index]
        elif request_type == "clearinghouseState":
            request["user"] = "0x" + "1" * 40
        raw = {
            "schema_version": "thewiz.hyperliquid_readonly_parity_raw.v1",
            "environment": environment,
            "info_url": info_url,
            "request": request,
            "captured_at_utc": captured_at,
            "response": {},
            "read_only_info_request": True,
            "private_key_accessed": False,
            "order_submission_performed": False,
        }
        raw["receipt_sha256"] = _payload_hash(raw)
        path = root / "data" / "parity" / "raw" / f"{input_name}-{environment}-{index}.json"
        _write_json(path, raw)
        sources[name] = {
            "path": str(path.relative_to(root)),
            "sha256": sha256(path.read_bytes()).hexdigest(),
            "request_type": request_type,
        }
    return sources


def _reseal_parity(evidence):
    core = {
        key: value
        for key, value in evidence.items()
        if key not in {"parity_receipt_id", "receipt_sha256"}
    }
    evidence["parity_receipt_id"] = "liveinputparity_" + _payload_hash(core)[:20]
    evidence["receipt_sha256"] = _payload_hash(evidence)
    return evidence


def _build(
    root,
    *,
    candidate_valid,
    sample_status,
    supreme_status,
    candidate=None,
):
    sample, supreme = _write_upstream(root)
    return build_live_canary_control_plane(
        root=root,
        now=NOW,
        candidate=candidate or _candidate(),
        candidate_valid=candidate_valid,
        sample_status=sample_status,
        supreme_status=supreme_status,
        sample_evidence_path=sample,
        supreme_evidence_path=supreme,
    )


def test_upstream_block_keeps_policy_unregistered_and_live_disabled(tmp_path):
    _write_json(tmp_path / "config" / "live_canary_policy.json", _policy())

    result = _build(
        tmp_path,
        candidate_valid=False,
        sample_status="BLOCKED",
        supreme_status="BLOCKED",
    )
    authorization = json.loads(result["authorization"].read_text())
    parity = pd.read_csv(result["parity"])

    assert authorization["authorization_status"] == "NOT_AUTHORIZED"
    assert authorization["live_trading_authorized"] is False
    assert authorization["canary_execution_authority"] is False
    assert parity["status"].eq("BLOCKED").all()
    assert not (tmp_path / "data" / "live" / "active_canary_policy.json").exists()
    outcome = json.loads(result["outcome"].read_text())
    assert outcome["canary_status"] == "NOT_EXECUTED"
    assert outcome["repeat_authorized"] is False
    assert outcome["live_trading_authorized"] is False


def test_upstream_block_revokes_stale_preflight_without_keychain_or_network(tmp_path, monkeypatch):
    policy = _policy()
    _write_json(tmp_path / "config" / "live_canary_policy.json", policy)
    preflight_path = (
        tmp_path / "reports" / "active" / "hyperliquid_live_canary_executor_preflight.json"
    )
    _write_json(
        preflight_path,
        {
            "schema_version": "stale-schema",
            "preflight_status": "PASS_READ_ONLY",
            "executor_contract_id": "stale-contract",
            "live_trading_authorized": False,
        },
    )
    account = Account.create()
    monkeypatch.setenv("HYPERLIQUID_LIVE_NETWORK", "mainnet")
    monkeypatch.setenv("HYPERLIQUID_LIVE_BASE_URL", executor_preflight.HYPERLIQUID_MAINNET_URL)
    monkeypatch.setenv("HYPERLIQUID_LIVE_MASTER_ADDRESS", "0x" + "1" * 40)
    monkeypatch.setenv("HYPERLIQUID_LIVE_AGENT_ADDRESS", account.address)
    monkeypatch.setenv("HYPERLIQUID_LIVE_AGENT_KEYCHAIN_SERVICE", "thewiz-live-canary-test")
    for name in executor_preflight.FORBIDDEN_RAW_KEY_ENVS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(executor_preflight.importlib.util, "find_spec", lambda name: object())

    def forbidden_keychain(*args, **kwargs):
        raise AssertionError("upstream-blocked refresh must not access the keychain")

    def forbidden_network(*args, **kwargs):
        raise AssertionError("upstream-blocked refresh must not access Hyperliquid")

    monkeypatch.setattr(
        executor_preflight,
        "read_live_agent_key_from_keychain",
        forbidden_keychain,
    )
    monkeypatch.setattr(executor_preflight, "_default_info_client", forbidden_network)

    result = _build(
        tmp_path,
        candidate_valid=False,
        sample_status="BLOCKED",
        supreme_status="BLOCKED",
    )

    receipt = json.loads(preflight_path.read_text())
    authorization = json.loads(result["authorization"].read_text())
    current_contract = live_canary_executor_contract()
    assert receipt["preflight_status"] == "BLOCKED_UPSTREAM"
    assert receipt["credential_and_network_checks_permitted"] is False
    assert "live_canary_executor_upstream_not_ready" in receipt["blockers"]
    assert receipt["executor_contract_id"] == current_contract["executor_contract_id"]
    assert receipt["agent_key_present"] is False
    assert receipt["agent_role_checked"] is False
    assert receipt["order_submission_performed"] is False
    assert receipt["canary_execution_authority"] is False
    assert receipt["live_trading_authorized"] is False
    assert authorization["authorization_status"] == "NOT_AUTHORIZED"
    assert authorization["canary_execution_authority"] is False
    assert (
        "live_canary_executor_preflight_contract_binding_mismatch" not in authorization["blockers"]
    )
    assert "live_canary_executor_preflight_stale_or_future" not in authorization["blockers"]


def test_claimed_candidate_validity_cannot_bypass_canonical_validator(tmp_path):
    policy = _policy()
    candidate = _candidate()
    _write_json(tmp_path / "config" / "live_canary_policy.json", policy)
    _write_parity(tmp_path, policy, candidate)

    result = _build(
        tmp_path,
        candidate_valid=True,
        sample_status="PASS",
        supreme_status="PASS",
        candidate=candidate,
    )
    authorization = json.loads(result["authorization"].read_text())

    assert authorization["authorization_status"] == "NOT_AUTHORIZED"
    assert "validated_live_candidate_identity_missing" in authorization["blockers"]
    assert "testnet_candidate_active_receipt_invalid" in authorization["blockers"]
    assert not (tmp_path / "data" / "live" / "active_canary_policy.json").exists()


def test_claimed_pass_statuses_cannot_bypass_blocked_evidence_files(tmp_path):
    policy = _policy()
    candidate = _valid_candidate(tmp_path)
    _write_json(tmp_path / "config" / "live_canary_policy.json", policy)
    _write_parity(tmp_path, policy, candidate)
    sample = tmp_path / "reports" / "active" / "realized_testnet_sample_sufficiency.csv"
    sample.parent.mkdir(parents=True, exist_ok=True)
    sample.write_text("check,status\nall,BLOCKED\n", encoding="utf-8")
    supreme = tmp_path / "reports" / "supreme_team" / "testnet_evidence_checkpoint.json"
    _write_json(supreme, {"checkpoint_status": "BLOCKED", "blockers": ["fixture"]})

    result = build_live_canary_control_plane(
        root=tmp_path,
        now=NOW,
        candidate=candidate,
        candidate_valid=True,
        sample_status="PASS",
        supreme_status="PASS",
        sample_evidence_path=sample,
        supreme_evidence_path=supreme,
    )
    authorization = json.loads(result["authorization"].read_text())

    assert authorization["authorization_status"] == "NOT_AUTHORIZED"
    assert "realized_testnet_sample_sufficiency_not_passed" in authorization["blockers"]
    assert "testnet_supreme_team_checkpoint_not_passed" in authorization["blockers"]
    assert not (tmp_path / "data" / "live" / "active_canary_policy.json").exists()


def test_fresh_authorization_cannot_refresh_an_old_candidate_receipt(tmp_path):
    policy = _policy()
    candidate = _valid_candidate(
        tmp_path,
        generated_at=NOW - timedelta(hours=1),
    )
    _write_json(tmp_path / "config" / "live_canary_policy.json", policy)
    _write_parity(tmp_path, policy, candidate)

    result = _build(
        tmp_path,
        candidate_valid=True,
        sample_status="PASS",
        supreme_status="PASS",
        candidate=candidate,
    )
    authorization = json.loads(result["authorization"].read_text())

    assert authorization["authorization_status"] == "NOT_AUTHORIZED"
    assert "validated_live_candidate_identity_stale_or_future" in authorization["blockers"]
    assert not (tmp_path / "data" / "live" / "active_canary_policy.json").exists()


def test_only_blank_legacy_approval_template_rotates_to_current_contract(tmp_path):
    _write_json(tmp_path / "config" / "live_canary_policy.json", _policy())
    approval_path = tmp_path / "reports" / "active" / "live_canary_user_approval.json"
    legacy_blank = {
        "schema_version": "thewiz.live_canary_user_approval.v1",
        "approved": False,
        "approval_id": "",
        "wallet_signature": "",
        "order_submission_performed": False,
        "live_trading_authorized": False,
    }
    _write_json(approval_path, legacy_blank)

    _build(
        tmp_path,
        candidate_valid=False,
        sample_status="BLOCKED",
        supreme_status="BLOCKED",
    )
    rotated = json.loads(approval_path.read_text())
    contract = live_canary_executor_contract()

    assert rotated["schema_version"] == "thewiz.live_canary_user_approval.v3"
    assert rotated["executor_contract_id"] == contract["executor_contract_id"]
    assert (
        rotated["executor_implementation_bundle_sha256"]
        == contract["executor_implementation_bundle_sha256"]
    )
    assert rotated["approved"] is False
    assert rotated["live_trading_authorized"] is False

    legacy_signed = {
        **legacy_blank,
        "approved": True,
        "approval_id": "livecanaryapproval_legacy",
        "wallet_signature": "signed-legacy-evidence",
    }
    _write_json(approval_path, legacy_signed)
    _build(
        tmp_path,
        candidate_valid=False,
        sample_status="BLOCKED",
        supreme_status="BLOCKED",
    )

    assert json.loads(approval_path.read_text()) == legacy_signed


def test_policy_freezes_only_after_stage6_and_cannot_rotate(tmp_path, monkeypatch):
    policy_path = tmp_path / "config" / "live_canary_policy.json"
    policy = _policy()
    candidate = _valid_candidate(tmp_path)
    _install_mock_stage6_release(monkeypatch, tmp_path, candidate)
    _write_json(policy_path, policy)
    _write_parity(tmp_path, policy, candidate)

    first = _build(
        tmp_path,
        candidate_valid=True,
        sample_status="PASS",
        supreme_status="PASS",
        candidate=candidate,
    )
    assert (tmp_path / "data" / "live" / "active_canary_policy.json").is_file()
    assert pd.read_csv(first["parity"])["status"].eq("PASS").all()

    policy["canary"]["maximum_total_notional_usd"] = 10.0
    _write_json(policy_path, policy)
    second = _build(
        tmp_path,
        candidate_valid=True,
        sample_status="PASS",
        supreme_status="PASS",
        candidate=candidate,
    )
    authorization = json.loads(second["authorization"].read_text())

    assert authorization["authorization_status"] == "NOT_AUTHORIZED"
    assert "live_canary_policy_rotation_after_stage6_forbidden" in authorization["blockers"]


def test_self_attested_arbitrary_parity_artifact_fails_closed(tmp_path):
    policy = _policy()
    _write_json(tmp_path / "config" / "live_canary_policy.json", policy)
    evidence = _write_parity(tmp_path, policy, _candidate())
    first = evidence["inputs"][0]
    artifact = tmp_path / first["testnet_artifact_path"]
    _write_json(artifact, {"source": "testnet", "schema_match": True})
    first["testnet_artifact_sha256"] = sha256(artifact.read_bytes()).hexdigest()
    first.update(
        {
            "schema_match": True,
            "units_match": True,
            "calculation_match": True,
            "capture_complete": True,
        }
    )
    _write_json(
        tmp_path / "reports" / "active" / "testnet_live_input_parity_evidence.json",
        _reseal_parity(evidence),
    )

    result = _build(
        tmp_path,
        candidate_valid=True,
        sample_status="PASS",
        supreme_status="PASS",
    )
    parity = pd.read_csv(result["parity"]).set_index("input")

    assert parity.loc[REQUIRED_PARITY_INPUTS[0], "status"] == "BLOCKED"
    assert bool(parity.loc[REQUIRED_PARITY_INPUTS[0], "schema_match"]) is False


def test_reused_parity_snapshot_paths_invalidate_entire_receipt(tmp_path):
    policy = _policy()
    _write_json(tmp_path / "config" / "live_canary_policy.json", policy)
    evidence = _write_parity(tmp_path, policy, _candidate())
    for environment in ("testnet", "live"):
        evidence["inputs"][1][f"{environment}_artifact_path"] = evidence["inputs"][0][
            f"{environment}_artifact_path"
        ]
        evidence["inputs"][1][f"{environment}_artifact_sha256"] = evidence["inputs"][0][
            f"{environment}_artifact_sha256"
        ]
    _write_json(
        tmp_path / "reports" / "active" / "testnet_live_input_parity_evidence.json",
        _reseal_parity(evidence),
    )

    result = _build(
        tmp_path,
        candidate_valid=True,
        sample_status="PASS",
        supreme_status="PASS",
    )

    assert pd.read_csv(result["parity"])["status"].eq("BLOCKED").all()


def test_parity_calculation_artifact_tamper_is_detected(tmp_path):
    policy = _policy()
    _write_json(tmp_path / "config" / "live_canary_policy.json", policy)
    evidence = _write_parity(tmp_path, policy, _candidate())
    testnet_snapshot = json.loads(
        (tmp_path / evidence["inputs"][0]["testnet_artifact_path"]).read_text()
    )
    calculation = tmp_path / testnet_snapshot["calculation_artifact_path"]
    calculation.write_text("tampered calculation\n", encoding="utf-8")

    result = _build(
        tmp_path,
        candidate_valid=True,
        sample_status="PASS",
        supreme_status="PASS",
    )
    parity = pd.read_csv(result["parity"]).set_index("input")

    assert parity.loc[REQUIRED_PARITY_INPUTS[0], "status"] == "BLOCKED"
    assert bool(parity.loc[REQUIRED_PARITY_INPUTS[0], "calculation_match"]) is False


def test_parity_snapshot_unit_drift_is_independently_detected(tmp_path):
    policy = _policy()
    _write_json(tmp_path / "config" / "live_canary_policy.json", policy)
    evidence = _write_parity(tmp_path, policy, _candidate())
    first = evidence["inputs"][0]
    artifact = tmp_path / first["live_artifact_path"]
    snapshot = json.loads(artifact.read_text())
    unit_field = next(iter(snapshot["units"]))
    snapshot["units"][unit_field] = "wrong_unit"
    snapshot["receipt_sha256"] = _payload_hash(snapshot)
    _write_json(artifact, snapshot)
    first["live_artifact_sha256"] = sha256(artifact.read_bytes()).hexdigest()
    _write_json(
        tmp_path / "reports" / "active" / "testnet_live_input_parity_evidence.json",
        _reseal_parity(evidence),
    )

    result = _build(
        tmp_path,
        candidate_valid=True,
        sample_status="PASS",
        supreme_status="PASS",
    )
    parity = pd.read_csv(result["parity"]).set_index("input")

    assert parity.loc[REQUIRED_PARITY_INPUTS[0], "status"] == "BLOCKED"
    assert bool(parity.loc[REQUIRED_PARITY_INPUTS[0], "units_match"]) is False


def test_parity_raw_source_tamper_is_independently_detected(tmp_path):
    policy = _policy()
    _write_json(tmp_path / "config" / "live_canary_policy.json", policy)
    evidence = _write_parity(tmp_path, policy, _candidate())
    first = evidence["inputs"][0]
    snapshot = json.loads((tmp_path / first["testnet_artifact_path"]).read_text())
    source = next(iter(snapshot["source_artifacts"].values()))
    raw = tmp_path / source["path"]
    raw.write_text(raw.read_text() + "\n", encoding="utf-8")

    result = _build(
        tmp_path,
        candidate_valid=True,
        sample_status="PASS",
        supreme_status="PASS",
    )
    parity = pd.read_csv(result["parity"]).set_index("input")

    assert parity.loc[REQUIRED_PARITY_INPUTS[0], "status"] == "BLOCKED"
    assert bool(parity.loc[REQUIRED_PARITY_INPUTS[0], "source_artifacts_hash_bound"]) is False


def test_self_asserted_stage6_pass_files_cannot_authorize_live_canary(tmp_path):
    policy = _policy()
    candidate = _valid_candidate(tmp_path)
    _write_json(tmp_path / "config" / "live_canary_policy.json", policy)
    _write_parity(tmp_path, policy, candidate)

    result = _build(
        tmp_path,
        candidate_valid=True,
        sample_status="PASS",
        supreme_status="PASS",
        candidate=candidate,
    )
    authorization = json.loads(result["authorization"].read_text())

    assert authorization["authorization_status"] == "NOT_AUTHORIZED"
    assert authorization["canary_execution_authority"] is False
    assert authorization["prerequisite_stage6_release_pass"] is False
    assert "stage6_release_pointer_invalid_or_blocked" in authorization["blockers"]


def test_valid_wallet_approval_and_preflight_authorize_exactly_one_canary(
    tmp_path, monkeypatch
):
    account = Account.create()
    policy = _policy(account.address)
    candidate = _valid_candidate(tmp_path)
    _install_mock_stage6_release(monkeypatch, tmp_path, candidate)
    _write_json(tmp_path / "config" / "live_canary_policy.json", policy)
    parity = _write_parity(tmp_path, policy, candidate)
    first = _build(
        tmp_path,
        candidate_valid=True,
        sample_status="PASS",
        supreme_status="PASS",
        candidate=candidate,
    )
    approval_path = first["approval"]
    approval = json.loads(approval_path.read_text())
    approval.update(
        {
            "approved": True,
            "issued_at_utc": NOW.isoformat(),
            "expires_at_utc": (NOW + timedelta(minutes=10)).isoformat(),
            "authorized_signer_address": account.address,
            "parity_receipt_id": parity["parity_receipt_id"],
            "legs": [
                {
                    "market": "BTC",
                    "side": "BUY",
                    "size": 0.0002,
                    "limit_price": 60_000.0,
                },
                {
                    "market": "ETH",
                    "side": "SELL",
                    "size": 0.004,
                    "limit_price": 3_000.0,
                },
            ],
            "maximum_total_notional_usd": 25.0,
            "maximum_slippage_bps": 10.0,
            "nonce": "single-use-test-nonce",
        }
    )
    core = {
        key: value
        for key, value in approval.items()
        if key not in {"approval_id", "wallet_signature"}
    }
    approval["approval_id"] = "livecanaryapproval_" + _payload_hash(core)[:20]
    signed = {key: value for key, value in approval.items() if key != "wallet_signature"}
    approval["wallet_signature"] = Account.sign_message(
        encode_defunct(text=_canonical_json(signed)), account.key
    ).signature.hex()
    _write_json(approval_path, approval)
    executor_config = executor_preflight.HyperliquidLiveCanaryConfig(
        network="mainnet",
        base_url=executor_preflight.HYPERLIQUID_MAINNET_URL,
        master_address="0x" + "1" * 40,
        agent_address=account.address,
        keychain_service="thewiz-live-canary-test",
    )

    def info(request):
        if request["type"] == "userRole":
            return {"role": "agent", "data": {"user": executor_config.master_address}}
        if request["type"] == "meta":
            return {"universe": [{"name": "BTC"}, {"name": "ETH"}]}
        if request["type"] == "clearinghouseState":
            return {
                "marginSummary": {"accountValue": "100"},
                "withdrawable": "75",
                "assetPositions": [],
            }
        if request["type"] == "openOrders":
            return []
        raise AssertionError(request)

    executor_preflight.build_live_canary_executor_preflight(
        root=tmp_path,
        now=NOW,
        candidate=candidate,
        policy_id=_policy_id(policy),
        config=executor_config,
        info_client=info,
        keychain_reader=lambda service, address: account.key.hex(),
    )

    ready = _build(
        tmp_path,
        candidate_valid=True,
        sample_status="PASS",
        supreme_status="PASS",
        candidate=candidate,
    )
    authorization = json.loads(ready["authorization"].read_text())
    assert authorization["authorization_status"] == "AUTHORIZED_FOR_ONE_LIVE_CANARY"
    assert authorization["user_authorization_present"] is True
    assert authorization["canary_execution_authority"] is True
    assert authorization["manual_executor_available"] is True
    assert authorization["live_trading_authorized"] is False
    assert authorization["blockers"] == []

    ledger = tmp_path / "data" / "live" / "canary_authorization_use_ledger.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text(json.dumps({"approval_id": approval["approval_id"]}) + "\n")
    replay = _build(
        tmp_path,
        candidate_valid=True,
        sample_status="PASS",
        supreme_status="PASS",
        candidate=candidate,
    )
    replay_authorization = json.loads(replay["authorization"].read_text())
    assert replay_authorization["authorization_status"] == "NOT_AUTHORIZED"
    assert "live_canary_user_approval_already_used" in replay_authorization["blockers"]


def test_execution_evidence_without_authority_is_an_incident(tmp_path):
    _write_json(tmp_path / "config" / "live_canary_policy.json", _policy())
    active = tmp_path / "reports" / "active"
    _write_json(
        active / "live_canary_execution_receipt.json",
        {"schema_version": EXECUTION_SCHEMA_VERSION, "execution_id": "forged"},
    )

    result = _build(
        tmp_path,
        candidate_valid=False,
        sample_status="BLOCKED",
        supreme_status="BLOCKED",
    )
    outcome = json.loads(result["outcome"].read_text())
    assert outcome["canary_status"] == "INCIDENT_UNAUTHORIZED_EXECUTION_EVIDENCE"
    assert outcome["orders_submitted"] == 0
    assert outcome["live_trading_authorized"] is False


def test_valid_canary_reconciliation_records_completion_without_new_authority(tmp_path):
    active = tmp_path / "reports" / "active"
    authorization_path = active / "live_canary_authorization.json"
    approval = {
        "approval_id": "livecanaryapproval_test",
        "legs": [
            {"market": "BTC", "side": "BUY", "size": 0.0002},
            {"market": "ETH", "side": "SELL", "size": 0.004},
        ],
    }
    policy = _policy()
    candidate = {
        **_candidate(),
        "candidate_receipt_id": "testnetcandidate_accepted",
    }
    executor_contract = live_canary_executor_contract()
    authorization = {
        "authorization_status": "AUTHORIZED_FOR_ONE_LIVE_CANARY",
        "canary_execution_authority": True,
        "manual_executor_available": True,
        "authorization_reusable": False,
        "live_canary_policy_id": _policy_id(policy),
        "executor_contract_id": executor_contract["executor_contract_id"],
        "executor_source_sha256": executor_contract["executor_source_sha256"],
        "executor_implementation_bundle_sha256": executor_contract[
            "executor_implementation_bundle_sha256"
        ],
    }
    _write_json(authorization_path, authorization)
    entry_fills = [
        {
            "fill_id": "entry-btc",
            "exchange_order_id": "order-entry-btc",
            "market": "BTC",
            "side": "BUY",
            "size": 0.0002,
            "price": 60_000.0,
            "fee_usd": 0.01,
            "funding_pnl_usd": 0.0,
            "timestamp_utc": (NOW - timedelta(minutes=4)).isoformat(),
        },
        {
            "fill_id": "entry-eth",
            "exchange_order_id": "order-entry-eth",
            "market": "ETH",
            "side": "SELL",
            "size": 0.004,
            "price": 3_000.0,
            "fee_usd": 0.01,
            "funding_pnl_usd": 0.0,
            "timestamp_utc": (NOW - timedelta(minutes=4)).isoformat(),
        },
    ]
    exit_fills = [
        {
            "fill_id": "exit-btc",
            "exchange_order_id": "order-exit-btc",
            "market": "BTC",
            "side": "SELL",
            "size": 0.0002,
            "price": 61_000.0,
            "fee_usd": 0.01,
            "funding_pnl_usd": 0.0,
            "timestamp_utc": (NOW - timedelta(minutes=2)).isoformat(),
        },
        {
            "fill_id": "exit-eth",
            "exchange_order_id": "order-exit-eth",
            "market": "ETH",
            "side": "BUY",
            "size": 0.004,
            "price": 2_900.0,
            "fee_usd": 0.01,
            "funding_pnl_usd": 0.0,
            "timestamp_utc": (NOW - timedelta(minutes=2)).isoformat(),
        },
    ]
    execution = {
        "schema_version": EXECUTION_SCHEMA_VERSION,
        "authorization_receipt_sha256": sha256(authorization_path.read_bytes()).hexdigest(),
        "approval_id": approval["approval_id"],
        "candidate_receipt_id": candidate["candidate_receipt_id"],
        "live_canary_policy_id": _policy_id(policy),
        "executor_contract_id": executor_contract["executor_contract_id"],
        "executor_source_sha256": executor_contract["executor_source_sha256"],
        "executor_implementation_bundle_sha256": executor_contract[
            "executor_implementation_bundle_sha256"
        ],
        "pair": candidate["pair"],
        "started_at_utc": (NOW - timedelta(minutes=5)).isoformat(),
        "completed_at_utc": (NOW - timedelta(minutes=1)).isoformat(),
        "entry_fills": entry_fills,
        "exit_fills": exit_fills,
        "gross_pnl_usd": 0.6,
        "fees_usd": 0.04,
        "funding_pnl_usd": 0.0,
        "net_pnl_usd": 0.56,
        "final_positions": {"BTC": 0.0, "ETH": 0.0},
        "open_order_ids": [],
        "unresolved_incidents": [],
        "reconciled_flat": True,
        "entry_retry_attempted": False,
        "repeat_authorized": False,
        "live_trading_authorized": False,
    }
    execution_identity = _identity(execution, "execution")
    execution_id = "livecanaryexec_" + execution_identity[:20]
    immutable_execution = tmp_path / "data" / "live" / "canary_executions" / f"{execution_id}.json"
    execution.update(
        {
            "execution_id": execution_id,
            "receipt_identity_sha256": execution_identity,
            "immutable_execution_path": str(immutable_execution.relative_to(tmp_path)),
        }
    )
    execution_path = active / "live_canary_execution_receipt.json"
    _write_json(execution_path, execution)
    _write_json(immutable_execution, execution)
    ledger = tmp_path / "data" / "live" / "canary_authorization_use_ledger.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text(
        json.dumps(
            {
                "approval_id": approval["approval_id"],
                "execution_id": execution_id,
                "authorization_receipt_sha256": sha256(authorization_path.read_bytes()).hexdigest(),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    review = {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "review_status": "PASS",
        "execution_id": execution_id,
        "execution_receipt_sha256": sha256(execution_path.read_bytes()).hexdigest(),
        "reconciliation_pass": True,
        "realized_cost_comparison_pass": True,
        "zero_unresolved_incidents": True,
        "new_explicit_authorization_required_for_any_next_order": True,
        "repeat_authorized": False,
        "scaling_authorized": False,
        "leverage_authorized": False,
        "live_trading_authorized": False,
    }
    review_identity = _identity(review, "review")
    review_id = "livecanaryreview_" + review_identity[:20]
    immutable_review = tmp_path / "data" / "live" / "canary_reviews" / f"{review_id}.json"
    review.update(
        {
            "review_id": review_id,
            "receipt_identity_sha256": review_identity,
            "immutable_review_path": str(immutable_review.relative_to(tmp_path)),
        }
    )
    review_path = tmp_path / "reports" / "supreme_team" / "live_canary_post_canary_review.json"
    _write_json(review_path, review)
    _write_json(immutable_review, review)

    outcome_path = evaluate_live_canary_outcome(
        root=tmp_path,
        now=NOW,
        authorization=authorization,
        authorization_path=authorization_path,
        approval=approval,
        policy=policy,
        candidate=candidate,
    )
    outcome = json.loads(outcome_path.read_text())
    assert outcome["canary_status"] == "PASS_ONE_CANARY_COMPLETE_NO_FURTHER_AUTHORITY"
    assert outcome["orders_submitted"] == 4
    assert outcome["fills_observed"] == 4
    assert outcome["net_pnl_usd"] == 0.56
    assert outcome["reconciled_flat"] is True
    assert outcome["repeat_authorized"] is False
    assert outcome["live_trading_authorized"] is False
