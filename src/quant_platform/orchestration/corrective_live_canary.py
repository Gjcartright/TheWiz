"""Non-executing Hyperliquid live-canary control plane."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd

from quant_platform.orchestration.corrective_live_canary_execution import (
    APPROVAL_SCHEMA_VERSION,
    AUTHORIZATION_SCHEMA_VERSION,
    live_canary_executor_contract,
)
from quant_platform.orchestration.corrective_live_canary_executor import (
    build_live_canary_executor_preflight,
    validate_live_canary_executor_preflight,
)
from quant_platform.orchestration.corrective_live_canary_outcome import (
    evaluate_live_canary_outcome,
)
from quant_platform.orchestration.corrective_runtime import promote_staged_file

ROOT = Path(__file__).resolve().parents[3]
POLICY_SCHEMA_VERSION = "thewiz.live_canary_policy.v2"
PARITY_SCHEMA_VERSION = "thewiz.testnet_live_input_parity_evidence.v2"
PARITY_ARTIFACT_SCHEMA_VERSION = "thewiz.live_input_parity_snapshot.v1"
PARITY_RAW_ARTIFACT_SCHEMA_VERSION = "thewiz.hyperliquid_readonly_parity_raw.v1"
REQUIRED_PARITY_INPUTS = (
    "market_metadata",
    "mark_and_mid_prices",
    "funding_rate_and_timestamp",
    "l2_depth_and_slippage",
    "size_precision_and_minimum_notional",
    "margin_and_liquidation_inputs",
)
PARITY_INPUT_CONTRACTS = {
    "market_metadata": {
        "asset_x_size_decimals": "decimal_places",
        "asset_y_size_decimals": "decimal_places",
        "asset_x_max_leverage": "leverage_ratio",
        "asset_y_max_leverage": "leverage_ratio",
        "asset_x_minimum_notional": "usd",
        "asset_y_minimum_notional": "usd",
    },
    "mark_and_mid_prices": {
        "asset_x_mark": "usd_per_asset",
        "asset_y_mark": "usd_per_asset",
        "asset_x_mid": "usd_per_asset",
        "asset_y_mid": "usd_per_asset",
    },
    "funding_rate_and_timestamp": {
        "asset_x_funding_rate": "rate_fraction",
        "asset_y_funding_rate": "rate_fraction",
        "asset_x_funding_timestamp": "unix_milliseconds",
        "asset_y_funding_timestamp": "unix_milliseconds",
    },
    "l2_depth_and_slippage": {
        "asset_x_bid_depth": "usd",
        "asset_x_ask_depth": "usd",
        "asset_y_bid_depth": "usd",
        "asset_y_ask_depth": "usd",
        "asset_x_slippage": "basis_points",
        "asset_y_slippage": "basis_points",
    },
    "size_precision_and_minimum_notional": {
        "asset_x_size_decimals": "decimal_places",
        "asset_y_size_decimals": "decimal_places",
        "asset_x_minimum_notional": "usd",
        "asset_y_minimum_notional": "usd",
    },
    "margin_and_liquidation_inputs": {
        "account_value": "usd",
        "withdrawable": "usd",
        "initial_margin_requirement": "usd",
        "maintenance_margin_requirement": "usd",
        "asset_x_liquidation_price": "usd_per_asset",
        "asset_y_liquidation_price": "usd_per_asset",
    },
}
PARITY_SOURCE_REQUEST_TYPES = {
    "market_metadata": ("metaAndAssetCtxs",),
    "mark_and_mid_prices": ("metaAndAssetCtxs", "allMids"),
    "funding_rate_and_timestamp": ("metaAndAssetCtxs",),
    "l2_depth_and_slippage": ("l2Book", "l2Book"),
    "size_precision_and_minimum_notional": ("metaAndAssetCtxs",),
    "margin_and_liquidation_inputs": ("clearinghouseState",),
}
PARITY_INFO_URLS = {
    "testnet": "https://api.hyperliquid-testnet.xyz/info",
    "live": "https://api.hyperliquid.xyz/info",
}


def build_live_canary_control_plane(
    *,
    root: Path = ROOT,
    now: datetime | None = None,
    candidate: dict[str, Any],
    candidate_valid: bool,
    sample_status: str,
    supreme_status: str,
    sample_evidence_path: Path,
    supreme_evidence_path: Path,
) -> dict[str, Any]:
    """Build live prerequisites and exact-order authorization without submitting."""

    as_of = _as_utc(now)
    active = root / "reports" / "active"
    active.mkdir(parents=True, exist_ok=True)
    policy_config_path = root / "config" / "live_canary_policy.json"
    policy_config = _read_json(policy_config_path)
    policy_valid, policy_blockers = _validate_policy_core(policy_config)
    policy_id = _policy_id(policy_config) if policy_valid else ""
    maximum_input_age = (
        _number((policy_config.get("canary") or {}).get("maximum_input_age_seconds")) or 0.0
    )
    executor_contract = live_canary_executor_contract()
    sample_evidence_sha256 = (
        _file_hash(sample_evidence_path) if sample_evidence_path.is_file() else ""
    )
    supreme_evidence_sha256 = (
        _file_hash(supreme_evidence_path) if supreme_evidence_path.is_file() else ""
    )
    from quant_platform.orchestration.corrective_release_gates import (
        _validated_testnet_candidate_receipt,
        validate_stage6_release_evidence,
    )

    canonical_candidate_valid, canonical_candidate_blockers = _validated_testnet_candidate_receipt(
        root=root, candidate=candidate
    )
    candidate_time = pd.to_datetime(candidate.get("generated_at_utc"), utc=True, errors="coerce")
    candidate_current = bool(
        pd.notna(candidate_time)
        and candidate_time <= pd.Timestamp(as_of)
        and (pd.Timestamp(as_of) - candidate_time).total_seconds() <= maximum_input_age
    )
    sample_evidence_pass = _sample_evidence_file_passes(sample_evidence_path)
    supreme_evidence_pass = _supreme_evidence_file_passes(supreme_evidence_path)
    (
        stage6_release_valid,
        stage6_release,
        stage6_release_path,
        stage6_release_blockers,
    ) = validate_stage6_release_evidence(
        root=root,
        candidate=candidate,
        sample_evidence_path=sample_evidence_path,
        supreme_evidence_path=supreme_evidence_path,
    )
    upstream_ready = bool(
        candidate_valid
        and canonical_candidate_valid
        and candidate_current
        and sample_status == "PASS"
        and sample_evidence_pass
        and supreme_status == "PASS"
        and supreme_evidence_pass
        and sample_evidence_sha256
        and supreme_evidence_sha256
        and stage6_release_valid
    )
    _policy_receipt, policy_path, registration_blockers = _register_policy(
        root=root,
        policy=policy_config,
        permit_registration=upstream_ready,
    )
    parity_path = active / "testnet_live_input_parity.csv"
    parity_evidence_path = active / "testnet_live_input_parity_evidence.json"
    parity_evidence = _read_json(parity_evidence_path)
    parity = _parity_rows(
        root=root,
        evidence=parity_evidence,
        candidate=candidate,
        policy=policy_config,
        policy_id=policy_id,
        as_of=as_of,
        upstream_ready=upstream_ready,
    )
    _atomic_csv(parity, parity_path)
    parity_pass = bool(not parity.empty and parity["status"].eq("PASS").all())
    parity_receipt_id = str(parity_evidence.get("parity_receipt_id", "")) if parity_pass else ""

    approval_path = active / "live_canary_user_approval.json"
    expected_approval = _approval_template(
        root=root,
        candidate=candidate,
        policy_id=policy_id,
        parity_receipt_id=parity_receipt_id,
        sample_evidence_sha256=sample_evidence_sha256,
        supreme_evidence_sha256=supreme_evidence_sha256,
        stage6_release=stage6_release,
        stage6_release_path=stage6_release_path,
        as_of=as_of,
        executor_contract=executor_contract,
    )
    existing_approval = _read_json(approval_path)
    if not approval_path.is_file() or (
        _approval_template_refreshable(existing_approval)
        and _approval_bindings_changed(existing_approval, expected_approval)
    ):
        _atomic_json(expected_approval, approval_path)
    approval = _read_json(approval_path)
    approval_ready, approval_blockers = _validate_approval(
        root=root,
        approval=approval,
        policy=policy_config,
        policy_id=policy_id,
        policy_path=policy_path,
        candidate=candidate,
        parity_evidence=parity_evidence,
        parity_pass=parity_pass,
        sample_evidence_sha256=sample_evidence_sha256,
        supreme_evidence_sha256=supreme_evidence_sha256,
        stage6_release=stage6_release,
        stage6_release_path=stage6_release_path,
        as_of=as_of,
    )
    executor_preflight_path = active / "hyperliquid_live_canary_executor_preflight.json"
    if not upstream_ready:
        build_live_canary_executor_preflight(
            root=root,
            now=as_of,
            candidate=candidate,
            policy_id=policy_id,
            permit_credential_and_network_checks=False,
        )
    executor_preflight = _read_json(executor_preflight_path)
    if executor_preflight_path.is_file():
        executor_preflight_ready, executor_preflight_blockers = (
            validate_live_canary_executor_preflight(
                receipt=executor_preflight,
                candidate=candidate,
                policy_id=policy_id,
                as_of=as_of,
                maximum_age_seconds=maximum_input_age,
            )
        )
    else:
        executor_preflight_ready = False
        executor_preflight_blockers = ["live_canary_executor_preflight_missing"]
    blockers = list(
        dict.fromkeys(
            [
                *policy_blockers,
                *registration_blockers,
                *(
                    []
                    if candidate_valid and canonical_candidate_valid
                    else ["validated_live_candidate_identity_missing"]
                ),
                *canonical_candidate_blockers,
                *stage6_release_blockers,
                *(
                    []
                    if candidate_current
                    else ["validated_live_candidate_identity_stale_or_future"]
                ),
                *(
                    []
                    if sample_status == "PASS" and sample_evidence_pass
                    else ["realized_testnet_sample_sufficiency_not_passed"]
                ),
                *(
                    []
                    if supreme_status == "PASS" and supreme_evidence_pass
                    else ["testnet_supreme_team_checkpoint_not_passed"]
                ),
                *([] if parity_pass else ["testnet_live_input_parity_not_passed"]),
                *approval_blockers,
                *executor_preflight_blockers,
            ]
        )
    )
    authorization_ready = bool(
        upstream_ready
        and policy_valid
        and policy_path is not None
        and parity_pass
        and approval_ready
        and executor_preflight_ready
        and executor_contract.get("submission_capable") is True
        and executor_contract.get("automatic_execution") is False
        and executor_contract.get("one_run_only") is True
        and executor_contract.get("persistent_live_authority") is False
    )
    authorization = {
        "schema_version": AUTHORIZATION_SCHEMA_VERSION,
        "generated_at_utc": as_of.isoformat(),
        "authorization_status": (
            "AUTHORIZED_FOR_ONE_LIVE_CANARY" if authorization_ready else "NOT_AUTHORIZED"
        ),
        "candidate_receipt_id": str(candidate.get("candidate_receipt_id", "")),
        "candidate_receipt_sha256": str(candidate.get("receipt_sha256", "")),
        "live_canary_policy_id": policy_id,
        "live_canary_policy_receipt_sha256": (
            _file_hash(policy_path) if policy_path and policy_path.is_file() else ""
        ),
        "parity_receipt_id": parity_receipt_id,
        "testnet_sample_evidence_sha256": sample_evidence_sha256,
        "testnet_supreme_team_evidence_sha256": supreme_evidence_sha256,
        "stage6_release_receipt_id": str(
            stage6_release.get("stage6_release_receipt_id", "")
        ),
        "stage6_release_receipt_path": (
            _relative(stage6_release_path, root)
            if stage6_release_path is not None
            else ""
        ),
        "stage6_release_receipt_sha256": (
            _file_hash(stage6_release_path)
            if stage6_release_path is not None and stage6_release_path.is_file()
            else ""
        ),
        "user_approval_id": str(approval.get("approval_id", "")) if approval_ready else "",
        "exact_pair": str(candidate.get("pair", "")) if approval_ready else "",
        "exact_side_and_sizes": approval.get("legs", []) if approval_ready else [],
        "executor_contract_id": executor_contract["executor_contract_id"],
        "executor_source_sha256": executor_contract["executor_source_sha256"],
        "executor_implementation_bundle_sha256": executor_contract[
            "executor_implementation_bundle_sha256"
        ],
        "maximum_leverage": 1.0,
        "authorization_reusable": False,
        "user_authorization_present": approval_ready,
        "prerequisite_sample_pass": sample_status == "PASS",
        "prerequisite_supreme_team_pass": supreme_status == "PASS",
        "prerequisite_stage6_release_pass": stage6_release_valid,
        "prerequisite_input_parity_pass": parity_pass,
        "prerequisite_executor_preflight_pass": executor_preflight_ready,
        "executor_preflight_path": (
            _relative(executor_preflight_path, root) if executor_preflight_path.is_file() else ""
        ),
        "executor_preflight_sha256": (
            _file_hash(executor_preflight_path) if executor_preflight_path.is_file() else ""
        ),
        "authorization_id": "",
        "immutable_authorization_path": "",
        "manual_executor_available": authorization_ready,
        "canary_execution_authority": authorization_ready,
        "order_submission_performed": False,
        "live_trading_authorized": False,
        "blockers": blockers,
    }
    if authorization_ready:
        authorization_id = "livecanaryauth_" + _payload_hash(authorization)[:20]
        immutable_authorization_path = (
            root / "data" / "live" / "canary_authorizations" / f"{authorization_id}.json"
        )
        authorization["authorization_id"] = authorization_id
        authorization["immutable_authorization_path"] = _relative(
            immutable_authorization_path, root
        )
    authorization["receipt_sha256"] = _payload_hash(authorization)
    if authorization_ready:
        _write_immutable_json(authorization, immutable_authorization_path)
    auth_path = active / "live_canary_authorization.json"
    _atomic_json(authorization, auth_path)
    outcome_path = evaluate_live_canary_outcome(
        root=root,
        now=as_of,
        authorization=authorization,
        authorization_path=auth_path,
        approval=approval,
        policy=policy_config,
        candidate=candidate,
    )
    return {
        "parity": parity_path,
        "authorization": auth_path,
        "outcome": outcome_path,
        "policy": policy_path,
        "approval": approval_path,
        "executor_preflight": executor_preflight_path,
        "executor_preflight_ready": executor_preflight_ready,
        "authorization_ready": authorization_ready,
        "live_trading_authorized": False,
    }


def _sample_evidence_file_passes(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        rows = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return False
    if rows.empty or "status" not in rows.columns:
        return False
    authority_columns = {
        "testnet_order_authority",
        "live_trading_authorized",
    } & set(rows.columns)
    authority_false = all(
        not rows[column].astype(str).str.strip().str.lower().isin({"true", "1", "yes"}).any()
        for column in authority_columns
    )
    return bool(
        rows["status"].astype(str).str.strip().str.upper().eq("PASS").all() and authority_false
    )


def _supreme_evidence_file_passes(path: Path) -> bool:
    evidence = _read_json(path)
    return bool(
        path.is_file()
        and evidence.get("checkpoint_status") == "PASS"
        and not evidence.get("blockers")
        and evidence.get("testnet_order_authority") is not True
        and evidence.get("live_trading_authorized") is not True
    )


def _validate_policy_core(policy: dict[str, Any]) -> tuple[bool, list[str]]:
    blockers: list[str] = []
    prerequisites = policy.get("prerequisites")
    canary = policy.get("canary")
    post_canary = policy.get("post_canary")
    if policy.get("schema_version") != POLICY_SCHEMA_VERSION:
        blockers.append("live_canary_policy_schema_invalid")
    if not str(policy.get("policy_version", "")).strip():
        blockers.append("live_canary_policy_version_missing")
    if pd.isna(pd.to_datetime(policy.get("effective_at_utc"), utc=True, errors="coerce")):
        blockers.append("live_canary_policy_effective_time_invalid")
    if (
        not isinstance(prerequisites, dict)
        or not prerequisites
        or not all(value is True for value in prerequisites.values())
    ):
        blockers.append("live_canary_policy_prerequisites_invalid")
    if not isinstance(canary, dict) or not (
        canary.get("maximum_simultaneous_pairs") == 1
        and _number(canary.get("maximum_leverage")) == 1.0
        and 0.0 < (_number(canary.get("maximum_total_notional_usd")) or 0.0) <= 25.0
        and (_number(canary.get("maximum_input_age_seconds")) or 0.0) > 0.0
        and (_number(canary.get("maximum_snapshot_skew_seconds")) or 0.0) > 0.0
        and canary.get("authorization_reuse_allowed") is False
        and canary.get("automatic_repeat_allowed") is False
        and canary.get("automatic_scaling_allowed") is False
        and canary.get("automatic_leverage_allowed") is False
    ):
        blockers.append("live_canary_policy_risk_limits_invalid")
    if (
        not isinstance(post_canary, dict)
        or not post_canary
        or not all(value is True for value in post_canary.values())
    ):
        blockers.append("live_canary_policy_post_canary_controls_invalid")
    signer = str(policy.get("authorized_signer_address", "")).strip()
    if signer and not _valid_address(signer):
        blockers.append("live_canary_policy_authorized_signer_invalid")
    if policy.get("current_live_trading_authorized") is not False:
        blockers.append("live_canary_policy_must_start_unauthorized")
    if policy.get("testnet_order_authority") is not False:
        blockers.append("live_canary_policy_testnet_authority_invalid")
    return not blockers, blockers


def _policy_id(policy: dict[str, Any]) -> str:
    return "livecanarypolicy_" + _payload_hash(policy)[:20]


def _register_policy(
    *, root: Path, policy: dict[str, Any], permit_registration: bool
) -> tuple[dict[str, Any], Path | None, list[str]]:
    valid, blockers = _validate_policy_core(policy)
    if not valid:
        return {}, None, blockers
    policy_id = _policy_id(policy)
    receipt = {
        **policy,
        "live_canary_policy_id": policy_id,
    }
    receipt["receipt_sha256"] = _payload_hash(receipt)
    if not permit_registration:
        return receipt, None, ["live_canary_policy_registration_waits_for_stage6"]
    directory = root / "data" / "live" / "canary_policies"
    path = directory / f"{policy_id}.json"
    pointer_path = root / "data" / "live" / "active_canary_policy.json"
    existing_pointer = _read_json(pointer_path)
    if existing_pointer and existing_pointer.get("live_canary_policy_id") != policy_id:
        return {}, None, ["live_canary_policy_rotation_after_stage6_forbidden"]
    _write_immutable_json(receipt, path)
    pointer = {
        "schema_version": "thewiz.live_canary_policy_pointer.v1",
        "live_canary_policy_id": policy_id,
        "policy_path": _relative(path, root),
        "policy_sha256": _file_hash(path),
        "live_trading_authorized": False,
    }
    pointer["receipt_sha256"] = _payload_hash(pointer)
    _write_immutable_json(pointer, pointer_path)
    return receipt, path, []


def _parity_rows(
    *,
    root: Path,
    evidence: dict[str, Any],
    candidate: dict[str, Any],
    policy: dict[str, Any],
    policy_id: str,
    as_of: datetime,
    upstream_ready: bool,
) -> pd.DataFrame:
    evidence_valid = _parity_receipt_valid(evidence)
    inputs = evidence.get("inputs") if isinstance(evidence.get("inputs"), list) else []
    by_name = {str(row.get("input", "")): row for row in inputs if isinstance(row, dict)}
    max_age = _number((policy.get("canary") or {}).get("maximum_input_age_seconds")) or 0.0
    max_skew = _number((policy.get("canary") or {}).get("maximum_snapshot_skew_seconds")) or 0.0
    rows: list[dict[str, Any]] = []
    for name in REQUIRED_PARITY_INPUTS:
        row = by_name.get(name, {})
        testnet_time = pd.to_datetime(row.get("testnet_timestamp_utc"), utc=True, errors="coerce")
        live_time = pd.to_datetime(row.get("live_timestamp_utc"), utc=True, errors="coerce")
        testnet_artifact = _safe_artifact(root, str(row.get("testnet_artifact_path", "")))
        live_artifact = _safe_artifact(root, str(row.get("live_artifact_path", "")))
        testnet_snapshot = _read_json(testnet_artifact) if testnet_artifact else {}
        live_snapshot = _read_json(live_artifact) if live_artifact else {}
        artifacts_valid = bool(
            testnet_artifact
            and live_artifact
            and testnet_artifact.is_file()
            and live_artifact.is_file()
            and _file_hash(testnet_artifact) == str(row.get("testnet_artifact_sha256", ""))
            and _file_hash(live_artifact) == str(row.get("live_artifact_sha256", ""))
        )
        schema_match = bool(
            artifacts_valid
            and _parity_snapshot_schema_valid(
                testnet_snapshot,
                input_name=name,
                environment="testnet",
                candidate_receipt_id=str(candidate.get("candidate_receipt_id", "")),
            )
            and _parity_snapshot_schema_valid(
                live_snapshot,
                input_name=name,
                environment="live",
                candidate_receipt_id=str(candidate.get("candidate_receipt_id", "")),
            )
        )
        units_match = bool(
            schema_match
            and testnet_snapshot.get("units") == PARITY_INPUT_CONTRACTS[name]
            and live_snapshot.get("units") == PARITY_INPUT_CONTRACTS[name]
        )
        calculation_match = bool(
            schema_match
            and _parity_calculation_artifacts_match(
                root=root,
                testnet_snapshot=testnet_snapshot,
                live_snapshot=live_snapshot,
            )
        )
        source_lineage_match = bool(
            schema_match
            and _parity_source_artifacts_match(
                root=root,
                snapshot=testnet_snapshot,
                input_name=name,
                environment="testnet",
                candidate=candidate,
            )
            and _parity_source_artifacts_match(
                root=root,
                snapshot=live_snapshot,
                input_name=name,
                environment="live",
                candidate=candidate,
            )
        )
        capture_complete = bool(
            schema_match
            and testnet_snapshot.get("capture_complete") is True
            and live_snapshot.get("capture_complete") is True
        )
        embedded_testnet_time = pd.to_datetime(
            testnet_snapshot.get("captured_at_utc"), utc=True, errors="coerce"
        )
        embedded_live_time = pd.to_datetime(
            live_snapshot.get("captured_at_utc"), utc=True, errors="coerce"
        )
        timestamps_valid = bool(
            pd.notna(testnet_time)
            and pd.notna(live_time)
            and testnet_time <= pd.Timestamp(as_of)
            and live_time <= pd.Timestamp(as_of)
            and (pd.Timestamp(as_of) - testnet_time).total_seconds() <= max_age
            and (pd.Timestamp(as_of) - live_time).total_seconds() <= max_age
            and abs((testnet_time - live_time).total_seconds()) <= max_skew
            and pd.notna(embedded_testnet_time)
            and pd.notna(embedded_live_time)
            and embedded_testnet_time == testnet_time
            and embedded_live_time == live_time
        )
        passed = bool(
            upstream_ready
            and evidence_valid
            and evidence.get("candidate_receipt_id") == candidate.get("candidate_receipt_id")
            and evidence.get("live_canary_policy_id") == policy_id
            and schema_match
            and units_match
            and calculation_match
            and source_lineage_match
            and capture_complete
            and artifacts_valid
            and timestamps_valid
        )
        rows.append(
            {
                "input": name,
                "status": "PASS" if passed else "BLOCKED",
                "blocker": "" if passed else f"live_input_parity_unproven:{name}",
                "testnet_timestamp_utc": row.get("testnet_timestamp_utc", ""),
                "live_timestamp_utc": row.get("live_timestamp_utc", ""),
                "schema_match": schema_match,
                "units_match": units_match,
                "calculation_match": calculation_match,
                "source_artifacts_hash_bound": source_lineage_match,
                "capture_complete": capture_complete,
                "artifacts_hash_bound": artifacts_valid,
                "timestamps_current_and_aligned": timestamps_valid,
                "shadow_only": True,
                "order_submission_performed": False,
                "live_trading_authorized": False,
            }
        )
    return pd.DataFrame(rows)


def _parity_receipt_valid(evidence: dict[str, Any]) -> bool:
    if evidence.get("schema_version") != PARITY_SCHEMA_VERSION:
        return False
    supplied_id = str(evidence.get("parity_receipt_id", ""))
    core = {
        key: value
        for key, value in evidence.items()
        if key not in {"parity_receipt_id", "receipt_sha256"}
    }
    expected_id = "liveinputparity_" + _payload_hash(core)[:20]
    inputs = evidence.get("inputs") if isinstance(evidence.get("inputs"), list) else []
    names = [str(row.get("input", "")) for row in inputs if isinstance(row, dict)]
    artifact_paths = [
        str(row.get(field, "")).strip()
        for row in inputs
        if isinstance(row, dict)
        for field in ("testnet_artifact_path", "live_artifact_path")
    ]
    return bool(
        supplied_id == expected_id
        and str(evidence.get("receipt_sha256", "")) == _payload_hash(evidence)
        and len(inputs) == len(REQUIRED_PARITY_INPUTS)
        and len(names) == len(REQUIRED_PARITY_INPUTS)
        and set(names) == set(REQUIRED_PARITY_INPUTS)
        and len(set(names)) == len(REQUIRED_PARITY_INPUTS)
        and len(artifact_paths) == 2 * len(REQUIRED_PARITY_INPUTS)
        and all(artifact_paths)
        and len(set(artifact_paths)) == len(artifact_paths)
        and evidence.get("order_submission_performed") is False
        and evidence.get("live_trading_authorized") is False
    )


def _parity_snapshot_schema_valid(
    snapshot: dict[str, Any],
    *,
    input_name: str,
    environment: str,
    candidate_receipt_id: str,
) -> bool:
    values = snapshot.get("values") if isinstance(snapshot.get("values"), dict) else {}
    expected_fields = set(PARITY_INPUT_CONTRACTS[input_name])
    return bool(
        snapshot.get("schema_version") == PARITY_ARTIFACT_SCHEMA_VERSION
        and snapshot.get("input") == input_name
        and snapshot.get("environment") == environment
        and snapshot.get("candidate_receipt_id") == candidate_receipt_id
        and set(values) == expected_fields
        and all(
            not isinstance(value, bool) and _number(value) is not None for value in values.values()
        )
        and isinstance(snapshot.get("units"), dict)
        and pd.notna(pd.to_datetime(snapshot.get("captured_at_utc"), utc=True, errors="coerce"))
        and snapshot.get("capture_complete") is True
        and isinstance(snapshot.get("source_artifacts"), dict)
        and snapshot.get("private_key_accessed") is False
        and snapshot.get("order_submission_performed") is False
        and snapshot.get("live_trading_authorized") is False
        and str(snapshot.get("receipt_sha256", "")) == _payload_hash(snapshot)
    )


def _parity_calculation_artifacts_match(
    *,
    root: Path,
    testnet_snapshot: dict[str, Any],
    live_snapshot: dict[str, Any],
) -> bool:
    testnet_path = _safe_artifact(root, str(testnet_snapshot.get("calculation_artifact_path", "")))
    live_path = _safe_artifact(root, str(live_snapshot.get("calculation_artifact_path", "")))
    testnet_sha = str(testnet_snapshot.get("calculation_artifact_sha256", ""))
    live_sha = str(live_snapshot.get("calculation_artifact_sha256", ""))
    return bool(
        testnet_path
        and live_path
        and testnet_path.resolve() == live_path.resolve()
        and testnet_path.is_file()
        and live_path.is_file()
        and testnet_sha
        and testnet_sha == live_sha
        and _file_hash(testnet_path) == testnet_sha
        and _file_hash(live_path) == live_sha
    )


def _parity_source_artifacts_match(
    *,
    root: Path,
    snapshot: dict[str, Any],
    input_name: str,
    environment: str,
    candidate: dict[str, Any],
) -> bool:
    sources = snapshot.get("source_artifacts")
    expected_types = PARITY_SOURCE_REQUEST_TYPES[input_name]
    if not isinstance(sources, dict) or len(sources) != len(expected_types):
        return False
    descriptors = list(sources.values())
    observed_types = sorted(
        str(item.get("request_type", "")) for item in descriptors if isinstance(item, dict)
    )
    paths = [str(item.get("path", "")).strip() for item in descriptors if isinstance(item, dict)]
    if (
        len(descriptors) != len(observed_types)
        or observed_types != sorted(expected_types)
        or len(paths) != len(expected_types)
        or not all(paths)
        or len(set(paths)) != len(paths)
    ):
        return False
    expected_assets = {
        str(candidate.get("asset_x", "")).strip().upper(),
        str(candidate.get("asset_y", "")).strip().upper(),
    } - {""}
    observed_l2_assets: set[str] = set()
    for descriptor in descriptors:
        if not isinstance(descriptor, dict):
            return False
        artifact = _safe_artifact(root, str(descriptor.get("path", "")))
        if (
            artifact is None
            or not artifact.is_file()
            or _file_hash(artifact) != str(descriptor.get("sha256", ""))
        ):
            return False
        raw = _read_json(artifact)
        request = raw.get("request") if isinstance(raw.get("request"), dict) else {}
        request_type = str(descriptor.get("request_type", ""))
        if (
            raw.get("schema_version") != PARITY_RAW_ARTIFACT_SCHEMA_VERSION
            or raw.get("environment") != environment
            or raw.get("info_url") != PARITY_INFO_URLS[environment]
            or request.get("type") != request_type
            or raw.get("captured_at_utc") != snapshot.get("captured_at_utc")
            or "response" not in raw
            or raw.get("read_only_info_request") is not True
            or raw.get("private_key_accessed") is not False
            or raw.get("order_submission_performed") is not False
            or str(raw.get("receipt_sha256", "")) != _payload_hash(raw)
        ):
            return False
        if request_type == "l2Book":
            observed_l2_assets.add(str(request.get("coin", "")).strip().upper())
    return bool(input_name != "l2_depth_and_slippage" or observed_l2_assets == expected_assets)


def _approval_template(
    *,
    root: Path,
    candidate: dict[str, Any],
    policy_id: str,
    parity_receipt_id: str,
    sample_evidence_sha256: str,
    supreme_evidence_sha256: str,
    as_of: datetime,
    executor_contract: dict[str, Any],
    stage6_release: dict[str, Any],
    stage6_release_path: Path | None,
) -> dict[str, Any]:
    return {
        "schema_version": APPROVAL_SCHEMA_VERSION,
        "approval_version": "hyperliquid-live-canary-v1",
        "approval_id": "",
        "approved": False,
        "one_run_only": True,
        "issued_at_utc": as_of.isoformat(),
        "expires_at_utc": "",
        "authorized_signer_address": "",
        "candidate_receipt_id": str(candidate.get("candidate_receipt_id", "")),
        "candidate_receipt_sha256": str(candidate.get("receipt_sha256", "")),
        "live_canary_policy_id": policy_id,
        "parity_receipt_id": parity_receipt_id,
        "testnet_sample_evidence_sha256": sample_evidence_sha256,
        "testnet_supreme_team_evidence_sha256": supreme_evidence_sha256,
        "stage6_release_receipt_id": str(
            stage6_release.get("stage6_release_receipt_id", "")
        ),
        "stage6_release_receipt_path": (
            _relative(stage6_release_path, root)
            if stage6_release_path is not None
            else ""
        ),
        "stage6_release_receipt_sha256": (
            _file_hash(stage6_release_path)
            if stage6_release_path is not None and stage6_release_path.is_file()
            else ""
        ),
        "executor_contract_id": str(executor_contract.get("executor_contract_id", "")),
        "executor_source_sha256": str(executor_contract.get("executor_source_sha256", "")),
        "executor_implementation_bundle_sha256": str(
            executor_contract.get("executor_implementation_bundle_sha256", "")
        ),
        "pair": str(candidate.get("pair", "")),
        "requested_leverage": 1.0,
        "legs": [],
        "maximum_total_notional_usd": 0.0,
        "maximum_slippage_bps": 0.0,
        "nonce": "",
        "purpose": "one_exact_minimal_hyperliquid_live_canary",
        "wallet_signature": "",
        "order_submission_performed": False,
        "live_trading_authorized": False,
    }


def _approval_template_refreshable(approval: dict[str, Any]) -> bool:
    return bool(
        approval.get("approved") is False
        and not str(approval.get("approval_id", "")).strip()
        and not str(approval.get("wallet_signature", "")).strip()
        and approval.get("order_submission_performed") is False
        and approval.get("live_trading_authorized") is False
    )


def _approval_bindings_changed(approval: dict[str, Any], expected: dict[str, Any]) -> bool:
    fields = (
        "candidate_receipt_id",
        "candidate_receipt_sha256",
        "live_canary_policy_id",
        "parity_receipt_id",
        "testnet_sample_evidence_sha256",
        "testnet_supreme_team_evidence_sha256",
        "stage6_release_receipt_id",
        "stage6_release_receipt_path",
        "stage6_release_receipt_sha256",
        "executor_contract_id",
        "executor_source_sha256",
        "executor_implementation_bundle_sha256",
        "pair",
    )
    return any(approval.get(field) != expected.get(field) for field in fields)


def _validate_approval(
    *,
    root: Path,
    approval: dict[str, Any],
    policy: dict[str, Any],
    policy_id: str,
    policy_path: Path | None,
    candidate: dict[str, Any],
    parity_evidence: dict[str, Any],
    parity_pass: bool,
    sample_evidence_sha256: str,
    supreme_evidence_sha256: str,
    stage6_release: dict[str, Any],
    stage6_release_path: Path | None,
    as_of: datetime,
) -> tuple[bool, list[str]]:
    blockers: list[str] = []
    if approval.get("schema_version") != APPROVAL_SCHEMA_VERSION:
        blockers.append("live_canary_user_approval_schema_invalid")
    if approval.get("approved") is not True or approval.get("one_run_only") is not True:
        blockers.append("live_canary_explicit_one_run_user_approval_missing")
    issued = pd.to_datetime(approval.get("issued_at_utc"), utc=True, errors="coerce")
    expires = pd.to_datetime(approval.get("expires_at_utc"), utc=True, errors="coerce")
    if not (
        pd.notna(issued)
        and pd.notna(expires)
        and issued <= pd.Timestamp(as_of) < expires
        and expires - issued <= pd.Timedelta(minutes=15)
    ):
        blockers.append("live_canary_user_approval_expired_or_invalid")
    expected_bindings = {
        "candidate_receipt_id": candidate.get("candidate_receipt_id", ""),
        "candidate_receipt_sha256": candidate.get("receipt_sha256", ""),
        "live_canary_policy_id": policy_id,
        "parity_receipt_id": (parity_evidence.get("parity_receipt_id", "") if parity_pass else ""),
        "testnet_sample_evidence_sha256": sample_evidence_sha256,
        "testnet_supreme_team_evidence_sha256": supreme_evidence_sha256,
        "stage6_release_receipt_id": stage6_release.get(
            "stage6_release_receipt_id", ""
        ),
        "stage6_release_receipt_path": (
            _relative(stage6_release_path, root)
            if stage6_release_path is not None
            else ""
        ),
        "stage6_release_receipt_sha256": (
            _file_hash(stage6_release_path)
            if stage6_release_path is not None and stage6_release_path.is_file()
            else ""
        ),
        "executor_contract_id": live_canary_executor_contract()["executor_contract_id"],
        "executor_source_sha256": live_canary_executor_contract()["executor_source_sha256"],
        "executor_implementation_bundle_sha256": live_canary_executor_contract()[
            "executor_implementation_bundle_sha256"
        ],
        "pair": candidate.get("pair", ""),
    }
    if policy_path is None or any(
        approval.get(field) != expected for field, expected in expected_bindings.items()
    ):
        blockers.append("live_canary_user_approval_evidence_binding_mismatch")
    legs = approval.get("legs") if isinstance(approval.get("legs"), list) else []
    assets = {
        str(candidate.get("asset_x", "")).strip().upper(),
        str(candidate.get("asset_y", "")).strip().upper(),
    } - {""}
    markets: set[str] = set()
    sides: set[str] = set()
    total_notional = 0.0
    legs_valid = len(legs) == 2
    for leg in legs:
        if not isinstance(leg, dict):
            legs_valid = False
            continue
        market = str(leg.get("market", "")).strip().upper().removesuffix("-USD")
        side = str(leg.get("side", "")).strip().upper()
        size = _number(leg.get("size"))
        limit_price = _number(leg.get("limit_price"))
        if not market or side not in {"BUY", "SELL"} or not size or not limit_price:
            legs_valid = False
            continue
        markets.add(market)
        sides.add(side)
        total_notional += size * limit_price
    cap = _number((policy.get("canary") or {}).get("maximum_total_notional_usd")) or 0.0
    legs_valid = bool(
        legs_valid
        and markets == assets
        and sides == {"BUY", "SELL"}
        and 0.0 < total_notional <= cap
        and _number(approval.get("maximum_total_notional_usd")) == cap
        and _number(approval.get("requested_leverage")) == 1.0
        and (_number(approval.get("maximum_slippage_bps")) or 0.0) > 0.0
    )
    if not legs_valid:
        blockers.append("live_canary_exact_pair_sizing_or_risk_limits_invalid")
    core = {
        key: value
        for key, value in approval.items()
        if key not in {"approval_id", "wallet_signature"}
    }
    expected_id = "livecanaryapproval_" + _payload_hash(core)[:20]
    if approval.get("approval_id") != expected_id:
        blockers.append("live_canary_user_approval_id_invalid")
    if not _wallet_signature_valid(approval=approval, policy=policy):
        blockers.append("live_canary_wallet_signature_invalid")
    used = _approval_already_used(
        root / "data" / "live" / "canary_authorization_use_ledger.jsonl",
        str(approval.get("approval_id", "")),
    )
    reservation_path = (
        root
        / "data"
        / "live"
        / "canary_authorization_reservations"
        / f"{approval.get('approval_id', '')!s}.json"
    )
    used = used or reservation_path.is_file()
    if used:
        blockers.append("live_canary_user_approval_already_used")
    if approval.get("order_submission_performed") is not False:
        blockers.append("live_canary_approval_claims_order_submission")
    if approval.get("live_trading_authorized") is not False:
        blockers.append("live_canary_approval_cannot_self_authorize")
    return not blockers, blockers


def _wallet_signature_valid(*, approval: dict[str, Any], policy: dict[str, Any]) -> bool:
    configured = str(policy.get("authorized_signer_address", "")).strip()
    supplied = str(approval.get("authorized_signer_address", "")).strip()
    signature = str(approval.get("wallet_signature", "")).strip()
    if not configured or configured.lower() != supplied.lower() or not signature:
        return False
    signed = {key: value for key, value in approval.items() if key != "wallet_signature"}
    try:
        from eth_account import Account
        from eth_account.messages import encode_defunct

        recovered = Account.recover_message(
            encode_defunct(text=_canonical_json(signed)), signature=signature
        )
    except (ImportError, TypeError, ValueError):
        return False
    return recovered.lower() == configured.lower()


def _approval_already_used(path: Path, approval_id: str) -> bool:
    if not approval_id or not path.is_file():
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            return True
        if isinstance(row, dict) and row.get("approval_id") == approval_id:
            return True
    return False


def _valid_address(value: str) -> bool:
    text = value.strip().lower()
    return (
        len(text) == 42
        and text.startswith("0x")
        and all(char in "0123456789abcdef" for char in text[2:])
    )


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _safe_artifact(root: Path, raw: str) -> Path | None:
    if not raw.strip():
        return None
    candidate = Path(raw)
    path = candidate if candidate.is_absolute() else root / candidate
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return None
    return path


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _payload_hash(payload: dict[str, Any]) -> str:
    core = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    return sha256(_canonical_json(core).encode("utf-8")).hexdigest()


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_immutable_json(payload: dict[str, Any], path: Path) -> None:
    existing = _read_json(path)
    if existing and existing != payload:
        raise ValueError(f"immutable live-canary artifact conflict: {path}")
    _atomic_json(payload, path)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    promote_staged_file(temporary, path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    promote_staged_file(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _as_utc(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(UTC)
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
