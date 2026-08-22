"""Council-aligned learning rows, portfolio veto, and Testnet lifecycle gates."""

from __future__ import annotations

from quant_platform.orchestration.corrective_runtime import promote_staged_file

from quant_platform.orchestration.corrective_runtime import atomic_write_text

import fcntl
import hmac
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from quant_platform.hyperliquid_testnet import (
    HyperliquidTestnetConfig,
    read_hyperliquid_agent_key_from_keychain,
)
from quant_platform.orchestration.student_readiness import (
    STUDENT_TRAINING_COLUMNS,
    write_student_training_readiness,
)

ROOT = Path(__file__).resolve().parents[3]
TESTNET_APPROVAL_VERSION = "hyperliquid-testnet-smoke-v10"
TESTNET_CANDIDATE_MAX_AGE = pd.Timedelta(hours=2)
TESTNET_ENTRY_CONTEXT_MAX_AGE = pd.Timedelta(minutes=5)
TESTNET_MAXIMUM_EXIT_SLIPPAGE_BPS = 50.0
TESTNET_STAGE5_COHORT_BINDING_FIELDS = (
    "model_training_dataset_id",
    "model_artifact_sha256",
    "registered_learning_id",
    "registered_learning_receipt_path",
    "registered_learning_receipt_sha256",
    "registered_stage5_protocol_id",
    "registered_stage5_protocol_sha256",
    "registered_execution_id",
)
TESTNET_CANDIDATE_BINDING_FIELDS = (
    "candidate_receipt_id",
    "candidate_experiment_id",
    "pair_group_key",
    "pair",
    "timeframe",
    "exact_mode",
    "orientation",
    "cost_model_id",
    *TESTNET_STAGE5_COHORT_BINDING_FIELDS,
)


@dataclass(frozen=True)
class PortfolioPolicy:
    max_pairs: int = 5
    max_shared_asset_share: float = 0.40
    max_gross_allocation: float = 1.0
    minimum_margin_buffer: float = 0.50


def materialize_council_learning_dataset(*, root: Path = ROOT) -> dict[str, object]:
    """Prefer OOS trade-entry rows, with summary rows as an explicit fallback."""

    source_path = (
        root / "reports" / "orchestration" / "teacher_council" / "teacher_evidence_coverage.csv"
    )
    trade_source_path = (
        root / "reports" / "orchestration" / "teacher_council" / "walkforward_trade_events.csv"
    )
    auxiliary_trade_path = (
        root
        / "reports"
        / "orchestration"
        / "teacher_council"
        / "auxiliary_4h_walkforward_trade_events.csv"
    )
    source = _read_csv(source_path)
    primary_trades = _read_csv(trade_source_path)
    auxiliary_trades = _read_csv(auxiliary_trade_path)
    if (
        not primary_trades.empty
        and not auxiliary_trades.empty
        and "run_id" in primary_trades.columns
        and "run_id" in auxiliary_trades.columns
    ):
        current_runs = set(primary_trades["run_id"].dropna().astype(str))
        auxiliary_trades = auxiliary_trades.loc[
            auxiliary_trades["run_id"].astype(str).isin(current_runs)
        ].copy()
    trade_source = pd.concat([primary_trades, auxiliary_trades], ignore_index=True)
    if not trade_source.empty and "training_event_id" in trade_source.columns:
        trade_source = trade_source.drop_duplicates(subset=["training_event_id"], keep="last")
    dataset_path = root / "data" / "ml" / "student_teacher_training_dataset.csv"
    mode_manifest_path = (
        root
        / "reports"
        / "orchestration"
        / "teacher_council"
        / "student_mode_training_manifest.csv"
    )
    summary_path = (
        root
        / "reports"
        / "orchestration"
        / "teacher_council"
        / "student_dataset_materialization.md"
    )
    if not trade_source.empty:
        research_rows = len(trade_source)
        if "training_eligible" in trade_source.columns:
            eligible = trade_source["training_eligible"].map(_truthy)
            dataset = trade_source.loc[eligible].copy()
        else:
            dataset = trade_source.copy()
        excluded_rows = research_rows - len(dataset)
        if not dataset.empty:
            mode_counts = dataset["exact_mode"].astype(str).value_counts()
            inverse = (
                dataset["exact_mode"].astype(str).map(lambda value: 1.0 / float(mode_counts[value]))
            )
            dataset["sample_weight"] = inverse / float(inverse.mean())
            dataset["training_topology"] = "per_mode_specialists_with_weighted_pooled_baseline"
        for column in STUDENT_TRAINING_COLUMNS:
            if column not in dataset.columns:
                dataset[column] = np.nan
        ordered = [
            *STUDENT_TRAINING_COLUMNS,
            *[column for column in dataset.columns if column not in STUDENT_TRAINING_COLUMNS],
        ]
        dataset = dataset.loc[:, ordered]
        _atomic_csv(dataset, dataset_path)
        mode_manifest = _mode_training_manifest(dataset)
        _atomic_csv(mode_manifest, mode_manifest_path)
        readiness = write_student_training_readiness(root=root, dataset_path=dataset_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(summary_path, "\n".join(
                [
                    "# Council Learning Dataset",
                    "",
                    f"- Rows materialized: `{len(dataset)}`",
                    f"- Raw OOS trade rows: `{research_rows}`",
                    f"- Explicitly ineligible rows excluded: `{excluded_rows}`",
                    "- Granularity: **out-of-sample trade entry**",
                    "- Topology: **one specialist per exact mode plus an inverse-frequency weighted pooled baseline**",
                    f"- Supervised status: **{readiness['supervised_status']}**",
                    f"- Contextual-bandit status: **{readiness['bandit_status']}**",
                    "- Deterministic replay propensity is logged as 1.0 but does not establish exploratory action support.",
                    "- Crypto Wizards is never a label authority.",
                    "",
                    f"Sources: `{trade_source_path}`; `{auxiliary_trade_path}`",
                    f"Readiness: `{readiness['markdown']}`",
                    "",
                ]
            ), encoding="utf-8")
        return {
            "status": "MATERIALIZED",
            "rows": len(dataset),
            "dataset": dataset_path,
            "readiness": readiness["audit"],
            "summary": summary_path,
            "mode_manifest": mode_manifest_path,
            "supervised_status": readiness["supervised_status"],
            "bandit_status": readiness["bandit_status"],
        }

    rows: list[dict[str, object]] = []
    for _, row in source.iterrows():
        feature_timestamp = str(row.get("train_end", ""))
        label_timestamp = str(row.get("test_end", ""))
        event_material = "|".join(
            [
                str(row.get("run_id", "")),
                str(row.get("setup_id", "")),
                str(row.get("exact_mode", "")),
                feature_timestamp,
                label_timestamp,
            ]
        )
        lower = _number(row.get("expectancy_lower_95"))
        selection_pass = str(row.get("walkforward_selection_status", "")) == "PASS"
        rows.append(
            {
                "training_event_id": "training_"
                + sha256(event_material.encode("utf-8")).hexdigest()[:20],
                "context_id": str(row.get("source_snapshot_id", "")),
                "candidate_id": str(row.get("setup_id", "")),
                "pair": str(row.get("pair", "")),
                "timeframe": str(row.get("timeframe", "")),
                "exact_mode": str(row.get("exact_mode", "")),
                "proposed_action": str(row.get("proposed_action", "")),
                "feature_timestamp": feature_timestamp,
                "label_timestamp": label_timestamp,
                "point_in_time_status": str(row.get("point_in_time_status", "")),
                "math_version": str(row.get("math_version", "")),
                "source_system": "hyperliquid_teacher_council",
                "label_source": "walkforward_backtest_summary",
                "uses_wizard_as_label": False,
                "uses_dashboard_hindsight": False,
                "profit_after_cost": _number(row.get("total_return")),
                "good_trade": int(lower is not None and lower > 0.0 and selection_pass),
                "max_adverse_excursion": _number(row.get("max_drawdown")),
                "max_favorable_excursion": np.nan,
                "hold_bars": _number(row.get("expected_holding_bars")),
                "exit_reason": "mode_window_summary_not_trade_level",
                "action_propensity": np.nan,
                "evidence_path": str(row.get("evidence_path", source_path)),
                "run_id": str(row.get("run_id", "")),
                "candidate_set_id": str(row.get("candidate_set_id", "")),
                "setup_identity": str(row.get("setup_identity", "")),
                "record_granularity": "mode_window_summary",
                "training_eligible": False,
                "propensity_source": "not_logged_behavior_policy",
                "training_blocker": "trade_level_rows_and_logged_propensities_required",
            }
        )
    extras = [
        "run_id",
        "candidate_set_id",
        "setup_identity",
        "record_granularity",
        "training_eligible",
        "propensity_source",
        "training_blocker",
    ]
    dataset = pd.DataFrame(rows, columns=[*STUDENT_TRAINING_COLUMNS, *extras])
    _atomic_csv(dataset, dataset_path)
    readiness = write_student_training_readiness(root=root, dataset_path=dataset_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(summary_path, "\n".join(
            [
                "# Council Learning Dataset",
                "",
                f"- Rows materialized: `{len(dataset)}`",
                "- Granularity: **mode-window summary**",
                "- Supervised training eligibility: **blocked until trade-level events exist**",
                "- Contextual-bandit eligibility: **blocked until behavior propensities are logged**",
                "- Crypto Wizards is never a label authority.",
                "",
                f"Readiness: `{readiness['markdown']}`",
                "",
            ]
        ), encoding="utf-8")
    return {
        "status": "MATERIALIZED" if not dataset.empty else "BLOCKED",
        "rows": len(dataset),
        "dataset": dataset_path,
        "readiness": readiness["audit"],
        "summary": summary_path,
        "supervised_status": readiness["supervised_status"],
        "bandit_status": readiness["bandit_status"],
    }


def build_portfolio_critic(
    *,
    root: Path = ROOT,
    policy: PortfolioPolicy | None = None,
) -> dict[str, object]:
    """Apply a council-wide veto after pair-level critics and before execution."""

    policy = policy or PortfolioPolicy()
    directory = root / "reports" / "orchestration" / "teacher_council"
    decisions = _read_csv(directory / "council_decisions.csv")
    candidates = _read_csv(root / "reports" / "active" / "hyperliquid_run_candidates.csv")
    margin = _read_csv(root / "reports" / "active" / "hyperliquid_testnet_margin_snapshot.csv")
    manifest_ids = _single_ids(candidates)
    if decisions.empty:
        eligible = pd.DataFrame()
    else:
        status = decisions.get("status", pd.Series("", index=decisions.index)).astype(str)
        action = decisions.get("action", pd.Series("", index=decisions.index)).astype(str)
        run_match = (
            decisions.get("run_id", pd.Series("", index=decisions.index))
            .astype(str)
            .eq(manifest_ids[0])
        )
        set_match = (
            decisions.get("candidate_set_id", pd.Series("", index=decisions.index))
            .astype(str)
            .eq(manifest_ids[1])
        )
        eligible = decisions.loc[
            status.eq("SHADOW_TEST") & action.ne("abstain") & run_match & set_match
        ].copy()
    blockers: list[str] = []
    if decisions.empty:
        blockers.append("council_decisions_missing")
    if eligible.empty:
        blockers.append("no_execution_eligible_council_decisions")
    if len(eligible) > policy.max_pairs:
        blockers.append("portfolio_pair_count_above_limit")

    asset_shares: dict[str, float] = {}
    if not eligible.empty:
        pair_allocation = 1.0 / len(eligible)
        for pair in eligible["pair"].astype(str):
            candidate = _pair_row(candidates, pair)
            for asset in (str(candidate.get("asset_x", "")), str(candidate.get("asset_y", ""))):
                if asset:
                    asset_shares[asset] = asset_shares.get(asset, 0.0) + pair_allocation / 2.0
    max_asset_share = max(asset_shares.values(), default=0.0)
    if max_asset_share > policy.max_shared_asset_share:
        blockers.append("shared_asset_concentration_above_limit")
    gross_allocation = 1.0 if not eligible.empty else 0.0
    if gross_allocation > policy.max_gross_allocation:
        blockers.append("gross_allocation_above_limit")

    margin_buffer = _number(margin.iloc[0].get("margin_buffer")) if not margin.empty else None
    if margin.empty:
        blockers.append("account_margin_snapshot_missing")
    elif margin_buffer is None:
        margin_blockers = [
            value for value in str(margin.iloc[0].get("blockers", "")).split(";") if value
        ]
        blockers.extend(margin_blockers or ["account_margin_buffer_unavailable"])
    elif margin_buffer < policy.minimum_margin_buffer:
        blockers.append("margin_buffer_below_limit")
    verdict = "pass" if not blockers else "veto"
    row = {
        "run_id": manifest_ids[0],
        "candidate_set_id": manifest_ids[1],
        "critic_type": "portfolio",
        "verdict": verdict,
        "eligible_pairs": len(eligible),
        "max_pairs": policy.max_pairs,
        "gross_allocation": gross_allocation,
        "max_gross_allocation": policy.max_gross_allocation,
        "max_shared_asset_share": max_asset_share,
        "shared_asset_limit": policy.max_shared_asset_share,
        "margin_buffer": margin_buffer,
        "minimum_margin_buffer": policy.minimum_margin_buffer,
        "blocker_codes": ";".join(blockers),
        "reason": "portfolio constraints satisfied" if verdict == "pass" else ";".join(blockers),
        "execution_allowed": False,
        "evidence_path": (
            "reports/orchestration/teacher_council/council_decisions.csv;"
            "reports/active/hyperliquid_run_candidates.csv;"
            "reports/active/hyperliquid_testnet_margin_snapshot.csv"
        ),
    }
    frame = pd.DataFrame([row])
    path = directory / "portfolio_critic.csv"
    markdown = directory / "portfolio_critic.md"
    _atomic_csv(frame, path)
    atomic_write_text(markdown, "# Portfolio Critic\n\n" + frame.to_markdown(index=False) + "\n", encoding="utf-8")
    return {"status": verdict.upper(), "critic": path, "summary": markdown, **row}


def build_testnet_lifecycle_gate(
    *,
    root: Path = ROOT,
    approval_secret: str | None = None,
    config: HyperliquidTestnetConfig | None = None,
) -> dict[str, object]:
    """Document the bounded two-leg smoke-test requirements without submitting."""

    active = root / "reports" / "active"
    preflight = _read_csv(active / "hyperliquid_testnet_preflight.csv")
    approval_path = active / "hyperliquid_testnet_smoke_approval.json"
    receipt_path = active / "hyperliquid_testnet_smoke_receipt.json"
    protocol_path = active / "current_wizard_hyperliquid_testnet_protocol_manifest.json"
    approval = _read_json(approval_path)
    receipt = _read_json(receipt_path)
    protocol = _read_json(protocol_path)
    protocol_ready = bool(
        protocol.get("protocol_status") == "PASS"
        and protocol.get("simulation_only") is True
        and protocol.get("simulation_is_testnet_proof") is False
        and protocol.get("order_submission_performed") is False
        and protocol.get("live_trading_authorized") is False
    )
    receipt_checks = _testnet_receipt_checks(
        receipt,
        approval=approval,
        protocol=protocol,
        root=root,
    )
    checks = [
        (
            "no_order_preflight",
            bool(
                not preflight.empty
                and preflight.get("ready_for_no_order_preflight", pd.Series([False]))
                .map(_truthy)
                .iloc[0]
            ),
            "hyperliquid_no_order_preflight_not_ready",
        ),
        (
            "deterministic_protocol_simulation",
            protocol_ready,
            "deterministic_testnet_protocol_validation_missing_or_failed",
        ),
        *_testnet_approval_checks(
            approval, root=root, approval_secret=approval_secret, config=config
        ),
        *receipt_checks,
    ]
    frame = pd.DataFrame(
        [
            {
                "check": name,
                "status": "PASS" if passed else "BLOCKED",
                "blocker": "" if passed else blocker,
                "execution_allowed": False,
            }
            for name, passed, blocker in checks
        ]
    )
    path = active / "hyperliquid_testnet_lifecycle_gate.csv"
    markdown = active / "hyperliquid_testnet_lifecycle_gate.md"
    _atomic_csv(frame, path)
    atomic_write_text(markdown, "# Hyperliquid Testnet Lifecycle Gate\n\nNo order is submitted by this check.\n\n"
        + frame.to_markdown(index=False)
        + "\n", encoding="utf-8")
    status = "PASS" if frame["status"].eq("PASS").all() else "BLOCKED"
    return {"status": status, "checks": len(frame), "gate": path, "summary": markdown}


def write_testnet_smoke_approval_template(
    *,
    root: Path = ROOT,
    config: HyperliquidTestnetConfig | None = None,
) -> dict[str, object]:
    """Create a non-approved one-run template without enabling submission."""

    path = root / "reports" / "active" / "hyperliquid_testnet_smoke_approval.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    created = False
    updated = False
    resolved = config or HyperliquidTestnetConfig.paper_testnet_from_env()
    manifest = _read_json(root / "reports" / "active" / "hyperliquid_run_manifest.json")
    candidate_binding = _testnet_candidate_binding(root)
    if not path.exists():
        payload = {
            "approval_version": TESTNET_APPROVAL_VERSION,
            "approved": False,
            "approval_id": "",
            "run_id": str(manifest.get("run_id", "")),
            "candidate_set_id": str(manifest.get("candidate_set_id", "")),
            "master_address": str(resolved.master_address or ""),
            "agent_address": str(resolved.agent_address or ""),
            "network": "testnet",
            "one_run_only": True,
            "issued_at_utc": "",
            "expires_at_utc": "",
            "max_total_notional_usd": 25.0,
            "maximum_exit_slippage_bps": TESTNET_MAXIMUM_EXIT_SLIPPAGE_BPS,
            "requested_leverage": int(resolved.requested_leverage),
            "margin_mode": str(resolved.margin_mode),
            "one_x_testnet_proof_id": str(resolved.one_x_testnet_proof_id or ""),
            "leverage_scenario_id": str(resolved.leverage_scenario_id or ""),
            "purpose": "bounded two-leg lifecycle proof only",
            "collateral_transfer_policy": {
                "approved": False,
                "direction": "spot_to_perp",
                "amount_usd": 25.0,
                "one_use": True,
                "maximum_transfer_attempts": 1,
            },
            "legs": [
                {"market": "", "side": "BUY", "size": 0.0, "limit_price": 0.0},
                {"market": "", "side": "SELL", "size": 0.0, "limit_price": 0.0},
            ],
            "required_outcomes": [
                "two_leg_entry",
                "two_leg_exit",
                "reconciled",
                "idempotency",
            ],
            "exit_policy": {
                "reduce_only": True,
                "opposite_side": True,
                "market_scope": "approved_entry_markets",
                "maximum_size": "approved_entry_size_per_leg",
                "entry_notional_cap_not_reapplied_to_exit": True,
            },
            "payload_hash": "",
            "approval_signature": "",
            "signature_scheme": "hmac-sha256-keychain-v1",
            "execution_authority": "none_until_user_sets_approved_true_for_this_exact_payload",
            "risk_override_applied": False,
            "entry_context": {
                "feature_timestamp_utc": "",
                "regime": "",
                "trade_quality_score": None,
                "strategy_signal_id": "",
            },
            **candidate_binding,
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        promote_staged_file(temporary, path)
        created = True
    else:
        payload = _read_json(path)
        if payload.get("approved") is not True:
            expected = {
                "approval_version": TESTNET_APPROVAL_VERSION,
                "run_id": str(manifest.get("run_id", "")),
                "candidate_set_id": str(manifest.get("candidate_set_id", "")),
                "master_address": str(resolved.master_address or ""),
                "agent_address": str(resolved.agent_address or ""),
                "max_total_notional_usd": 25.0,
                "maximum_exit_slippage_bps": TESTNET_MAXIMUM_EXIT_SLIPPAGE_BPS,
                "requested_leverage": int(resolved.requested_leverage),
                "margin_mode": str(resolved.margin_mode),
                "one_x_testnet_proof_id": str(resolved.one_x_testnet_proof_id or ""),
                "leverage_scenario_id": str(resolved.leverage_scenario_id or ""),
                "collateral_transfer_policy": {
                    "approved": False,
                    "direction": "spot_to_perp",
                    "amount_usd": 25.0,
                    "one_use": True,
                    "maximum_transfer_attempts": 1,
                },
                "issued_at_utc": "",
                "required_outcomes": [
                    "two_leg_entry",
                    "two_leg_exit",
                    "reconciled",
                    "idempotency",
                ],
                "exit_policy": {
                    "reduce_only": True,
                    "opposite_side": True,
                    "market_scope": "approved_entry_markets",
                    "maximum_size": "approved_entry_size_per_leg",
                    "entry_notional_cap_not_reapplied_to_exit": True,
                },
                "payload_hash": "",
                "approval_signature": "",
                "signature_scheme": "hmac-sha256-keychain-v1",
                "risk_override_applied": False,
                "entry_context": {
                    "feature_timestamp_utc": "",
                    "regime": "",
                    "trade_quality_score": None,
                    "strategy_signal_id": "",
                },
                **candidate_binding,
            }
            changed = False
            for key, value in expected.items():
                if payload.get(key) != value:
                    payload[key] = value
                    changed = True
            if changed:
                temporary = path.with_suffix(path.suffix + ".tmp")
                temporary.write_text(
                    json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
                )
                promote_staged_file(temporary, path)
                updated = True
    status = "TEMPLATE_CREATED" if created else "TEMPLATE_UPDATED" if updated else "EXISTS"
    return {"status": status, "approval": path, "approved": False}


def sign_testnet_smoke_approval(
    *,
    root: Path = ROOT,
    approval_secret: str | None = None,
    config: HyperliquidTestnetConfig | None = None,
) -> dict[str, object]:
    """Sign an explicitly approved Testnet payload without submitting an order."""

    path = root / "reports" / "active" / "hyperliquid_testnet_smoke_approval.json"
    approval = _read_json(path)
    resolved = config or HyperliquidTestnetConfig.paper_testnet_from_env()
    secret = approval_secret or _approval_secret_from_keychain(resolved)
    if not secret:
        raise ValueError("testnet approval signing secret unavailable")
    checks = _testnet_approval_checks(
        approval,
        root=root,
        approval_secret=secret,
        config=resolved,
        require_signature=False,
    )
    blockers = [blocker for _, passed, blocker in checks if not passed]
    if blockers:
        raise ValueError("testnet approval cannot be signed: " + ";".join(blockers))
    payload_hash = _approval_payload_hash(approval)
    approval["payload_hash"] = payload_hash
    approval["approval_signature"] = hmac.new(
        secret.encode("utf-8"), payload_hash.encode("ascii"), digestmod="sha256"
    ).hexdigest()
    approval["signature_scheme"] = "hmac-sha256-keychain-v1"
    identity_path = _testnet_smoke_approval_identity_path(root, str(approval["approval_id"]))
    lock_path = identity_path.parent / ".approval-sign.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            if identity_path.exists() and _read_json(identity_path) != approval:
                raise ValueError("testnet approval id already bound to a different payload")
            if not identity_path.exists():
                _atomic_json(approval, identity_path)
            _atomic_json(approval, path)
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
    return {
        "status": "SIGNED_TESTNET_ONLY",
        "approval": path,
        "approval_identity": identity_path,
        "approval_identity_sha256": _sha256_file(identity_path),
        "payload_hash": payload_hash,
    }


def validate_testnet_smoke_approval(
    *,
    approval_id: str,
    root: Path = ROOT,
    approval_secret: str | None = None,
    config: HyperliquidTestnetConfig | None = None,
    validation_scope: str = "entry",
    now: datetime | None = None,
) -> dict[str, object]:
    """Verify signed entry or journal-bound risk-reducing exit authority."""

    path = root / "reports" / "active" / "hyperliquid_testnet_smoke_approval.json"
    approval = _read_json(path)
    checks = _testnet_approval_checks(
        approval,
        root=root,
        approval_secret=approval_secret,
        config=config,
        validation_scope=validation_scope,
        now=now,
    )
    runtime_id_matches = bool(str(approval_id).strip()) and hmac.compare_digest(
        str(approval_id).strip(),
        str(approval.get("approval_id", "")).strip(),
    )
    checks.append(
        (
            "runtime_approval_id_bound",
            runtime_id_matches,
            "runtime_hyperliquid_order_approval_id_mismatch",
        )
    )
    scope_valid = validation_scope in {"entry", "exit"}
    checks.append(
        (
            "runtime_validation_scope",
            scope_valid,
            "runtime_hyperliquid_approval_validation_scope_invalid",
        )
    )
    blockers = [blocker for _, passed, blocker in checks if not passed]
    authority_granted = not blockers
    return {
        "status": "PASS" if authority_granted else "BLOCKED",
        "approval": path,
        "checks": len(checks),
        "blockers": blockers,
        "execution_allowed": authority_granted,
        "testnet_order_authority": authority_granted,
        "authority_scope": validation_scope if scope_valid else "invalid",
        "live_trading_authorized": False,
    }


def _testnet_approval_checks(
    approval: dict[str, object],
    *,
    root: Path,
    approval_secret: str | None = None,
    config: HyperliquidTestnetConfig | None = None,
    require_signature: bool = True,
    validation_scope: str = "entry",
    now: datetime | None = None,
) -> list[tuple[str, bool, str]]:
    legs = approval.get("legs") if isinstance(approval.get("legs"), list) else []
    approval_id = str(approval.get("approval_id", "")).strip()
    issued = pd.to_datetime(approval.get("issued_at_utc"), utc=True, errors="coerce")
    expires = pd.to_datetime(approval.get("expires_at_utc"), utc=True, errors="coerce")
    evaluated_at = pd.Timestamp(now or datetime.now(UTC))
    max_notional = _number(approval.get("max_total_notional_usd"))
    leg_notional = 0.0
    leg_shape_valid = len(legs) == 2
    markets: list[str] = []
    sides: list[str] = []
    if leg_shape_valid:
        for leg in legs:
            if not isinstance(leg, dict):
                leg_shape_valid = False
                continue
            market = str(leg.get("market", "")).strip().upper()
            side = str(leg.get("side", "")).strip().upper()
            size = _number(leg.get("size"))
            price = _number(leg.get("limit_price"))
            markets.append(market)
            sides.append(side)
            if (
                not market
                or side not in {"BUY", "SELL"}
                or size is None
                or size <= 0.0
                or price is None
                or price <= 0.0
            ):
                leg_shape_valid = False
            else:
                leg_notional += size * price
    if len(markets) == 2 and markets[0] == markets[1]:
        leg_shape_valid = False
    if len(sides) == 2 and sides[0] == sides[1]:
        leg_shape_valid = False
    per_leg_notional_valid = True
    if leg_shape_valid:
        per_leg_notional_valid = all(
            (_number(leg.get("size")) or 0.0) * (_number(leg.get("limit_price")) or 0.0) >= 10.0
            for leg in legs
            if isinstance(leg, dict)
        )
    bounded = (
        max_notional is not None
        and 20.0 <= max_notional <= 25.0
        and leg_notional <= max_notional
        and per_leg_notional_valid
    )
    resolved = config or HyperliquidTestnetConfig.paper_testnet_from_env()
    requested_leverage = _number(approval.get("requested_leverage"))
    leverage_matches = bool(
        requested_leverage is not None
        and requested_leverage.is_integer()
        and int(requested_leverage) == resolved.requested_leverage
    )
    margin_mode_matches = str(approval.get("margin_mode", "")) == resolved.margin_mode
    leverage_evidence_matches = bool(
        resolved.requested_leverage == 1
        or (
            str(resolved.one_x_testnet_proof_id or "").strip()
            and str(resolved.leverage_scenario_id or "").strip()
            and approval.get("one_x_testnet_proof_id") == resolved.one_x_testnet_proof_id
            and approval.get("leverage_scenario_id") == resolved.leverage_scenario_id
        )
    )
    is_exit = validation_scope == "exit"
    manifest = _read_json(root / "reports" / "active" / "hyperliquid_run_manifest.json")
    if is_exit:
        candidate = _immutable_testnet_candidate_for_approval(root, approval)
        candidate_binding = {
            field: str(candidate.get(field, "")).strip()
            for field in TESTNET_CANDIDATE_BINDING_FIELDS
        }
        candidate_assets = _candidate_assets(candidate)
    else:
        candidate_binding = _testnet_candidate_binding(root)
        candidate_assets = _testnet_candidate_assets(root)
        candidate = _read_json(root / "reports" / "active" / "testnet_candidate_receipt.json")
    candidate_time = pd.to_datetime(
        candidate.get("generated_at_utc"), utc=True, errors="coerce"
    )
    candidate_bound = bool(
        all(candidate_binding.get(field) for field in TESTNET_CANDIDATE_BINDING_FIELDS)
        and all(
            approval.get(field) == candidate_binding.get(field)
            for field in TESTNET_CANDIDATE_BINDING_FIELDS
        )
    )
    approved_markets = {
        str(market).removesuffix("-USD").removesuffix("/USD")
        for market in markets
        if str(market).strip()
    }
    candidate_markets_bound = bool(
        len(candidate_assets) == 2 and approved_markets == candidate_assets
    )
    entry_context = (
        approval.get("entry_context") if isinstance(approval.get("entry_context"), dict) else {}
    )
    entry_context_time = pd.to_datetime(
        entry_context.get("feature_timestamp_utc"), utc=True, errors="coerce"
    )
    entry_score = _number(entry_context.get("trade_quality_score"))
    entry_context_valid = bool(
        pd.notna(entry_context_time)
        and str(entry_context.get("regime", "")) in {"bull", "bear", "range", "crisis"}
        and entry_score is not None
        and 0.0 <= entry_score <= 1.0
        and str(entry_context.get("strategy_signal_id", "")).strip()
    )
    candidate_current = bool(
        (
            is_exit
            and candidate
            and pd.notna(issued)
            and issued <= evaluated_at
        )
        or (
            not is_exit
            and pd.notna(candidate_time)
            and pd.notna(issued)
            and candidate_time <= issued <= evaluated_at
            and evaluated_at - candidate_time <= TESTNET_CANDIDATE_MAX_AGE
        )
    )
    entry_context_current = bool(
        pd.notna(entry_context_time)
        and pd.notna(issued)
        and entry_context_time <= issued <= evaluated_at
        and (
            is_exit
            or evaluated_at - entry_context_time <= TESTNET_ENTRY_CONTEXT_MAX_AGE
        )
    )
    approval_window_valid = bool(
        pd.notna(issued)
        and pd.notna(expires)
        and issued <= evaluated_at
        and issued < expires
        and expires - issued <= pd.Timedelta(minutes=15)
        and (is_exit or evaluated_at < expires)
    )
    maximum_exit_slippage_bps = _number(
        approval.get("maximum_exit_slippage_bps")
    )
    exit_slippage_policy_valid = bool(
        maximum_exit_slippage_bps is not None
        and 0.0 < maximum_exit_slippage_bps <= TESTNET_MAXIMUM_EXIT_SLIPPAGE_BPS
    )
    expected_hash = _approval_payload_hash(approval)
    supplied_hash = str(approval.get("payload_hash", "")).strip().lower()
    secret = approval_secret or _approval_secret_from_keychain(resolved)
    expected_signature = (
        hmac.new(
            secret.encode("utf-8"), expected_hash.encode("ascii"), digestmod="sha256"
        ).hexdigest()
        if secret
        else ""
    )
    supplied_signature = str(approval.get("approval_signature", "")).strip().lower()
    checks = [
        (
            "explicit_smoke_approval",
            approval.get("approved") is True,
            "explicit_testnet_smoke_approval_missing",
        ),
        ("approval_id", bool(approval_id), "testnet_smoke_approval_id_missing"),
        (
            "testnet_only_scope",
            str(approval.get("network", "")).lower() == "testnet",
            "testnet_smoke_scope_not_testnet",
        ),
        (
            "one_run_only_scope",
            approval.get("one_run_only") is True,
            "testnet_smoke_not_one_run_only",
        ),
        (
            "approval_not_expired",
            approval_window_valid,
            "testnet_smoke_approval_missing_or_expired",
        ),
        ("two_leg_payload_valid", leg_shape_valid, "testnet_smoke_two_leg_payload_invalid"),
        (
            "notional_bounded",
            bounded,
            "testnet_smoke_notional_not_10_usd_per_leg_or_above_25_usd_total",
        ),
        (
            "approval_version",
            approval.get("approval_version") == TESTNET_APPROVAL_VERSION,
            "testnet_smoke_approval_version_invalid",
        ),
        (
            "accepted_candidate_identity_bound",
            candidate_bound,
            "testnet_smoke_candidate_identity_missing_or_mismatch",
        ),
        (
            "accepted_candidate_markets_bound",
            candidate_markets_bound,
            "testnet_smoke_candidate_markets_missing_or_mismatch",
        ),
        (
            "accepted_candidate_current",
            candidate_current,
            "testnet_smoke_candidate_receipt_stale_or_future",
        ),
        (
            "entry_context_bound",
            entry_context_valid,
            "testnet_smoke_entry_context_missing_or_invalid",
        ),
        (
            "entry_context_current",
            entry_context_current,
            "testnet_smoke_entry_context_stale_or_future",
        ),
        (
            "risk_policy_not_overridden",
            approval.get("risk_override_applied") is False,
            "testnet_smoke_risk_policy_override_detected",
        ),
        (
            "exit_policy_bound",
            approval.get("exit_policy")
            == {
                "reduce_only": True,
                "opposite_side": True,
                "market_scope": "approved_entry_markets",
                "maximum_size": "approved_entry_size_per_leg",
                "entry_notional_cap_not_reapplied_to_exit": True,
            },
            "testnet_smoke_exit_policy_invalid",
        ),
        (
            "exit_slippage_policy_bound",
            exit_slippage_policy_valid,
            "testnet_smoke_exit_slippage_policy_invalid",
        ),
        ("requested_leverage_bound", leverage_matches, "testnet_smoke_requested_leverage_mismatch"),
        ("margin_mode_bound", margin_mode_matches, "testnet_smoke_margin_mode_mismatch"),
        (
            "leverage_evidence_bound",
            leverage_evidence_matches,
            "testnet_smoke_leverage_evidence_missing_or_mismatch",
        ),
        (
            "run_id_bound",
            bool(approval.get("run_id"))
            and (is_exit or approval.get("run_id") == manifest.get("run_id")),
            "testnet_smoke_run_id_mismatch",
        ),
        (
            "candidate_set_bound",
            bool(approval.get("candidate_set_id"))
            and (
                is_exit
                or approval.get("candidate_set_id") == manifest.get("candidate_set_id")
            ),
            "testnet_smoke_candidate_set_id_mismatch",
        ),
        (
            "master_address_bound",
            bool(resolved.master_address)
            and str(approval.get("master_address", "")).lower()
            == str(resolved.master_address).lower(),
            "testnet_smoke_master_address_mismatch",
        ),
        (
            "agent_address_bound",
            bool(resolved.agent_address)
            and str(approval.get("agent_address", "")).lower()
            == str(resolved.agent_address).lower(),
            "testnet_smoke_agent_address_mismatch",
        ),
    ]
    if require_signature:
        identity_path = _testnet_smoke_approval_identity_path(root, approval_id)
        immutable_approval = _read_json(identity_path)
        checks.extend(
            [
                (
                    "immutable_approval_identity",
                    identity_path.is_file() and immutable_approval == approval,
                    "testnet_smoke_approval_identity_artifact_mismatch",
                ),
                (
                    "payload_hash_bound",
                    bool(supplied_hash) and hmac.compare_digest(supplied_hash, expected_hash),
                    "testnet_smoke_payload_hash_mismatch",
                ),
                (
                    "approval_signature",
                    bool(expected_signature)
                    and hmac.compare_digest(supplied_signature, expected_signature),
                    "testnet_smoke_approval_signature_missing_or_invalid",
                ),
            ]
        )
    return checks


def _single_ids(frame: pd.DataFrame) -> tuple[str, str]:
    values = []
    for column in ("run_id", "candidate_set_id"):
        unique = frame.get(column, pd.Series(dtype=str)).dropna().astype(str).unique()
        values.append(unique[0] if len(unique) == 1 else "")
    return values[0], values[1]


def _approval_payload_hash(approval: dict[str, object]) -> str:
    signed_fields = {
        key: approval.get(key)
        for key in (
            "approval_version",
            "approved",
            "approval_id",
            "run_id",
            "candidate_set_id",
            "master_address",
            "agent_address",
            "network",
            "one_run_only",
            "issued_at_utc",
            "expires_at_utc",
            "max_total_notional_usd",
            "maximum_exit_slippage_bps",
            "requested_leverage",
            "margin_mode",
            "one_x_testnet_proof_id",
            "leverage_scenario_id",
            "purpose",
            "collateral_transfer_policy",
            "legs",
            "required_outcomes",
            "exit_policy",
            "risk_override_applied",
            "entry_context",
            *TESTNET_CANDIDATE_BINDING_FIELDS,
        )
    }
    canonical = json.dumps(signed_fields, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256(canonical.encode("utf-8")).hexdigest()


def _testnet_smoke_approval_identity_path(root: Path, approval_id: str) -> Path:
    digest = sha256(str(approval_id).strip().encode("utf-8")).hexdigest()
    return root / "data" / "testnet" / "smoke_approvals" / f"testnetapproval_{digest}.json"


def _testnet_receipt_checks(
    receipt: dict[str, object],
    *,
    approval: dict[str, object],
    protocol: dict[str, object],
    root: Path,
    manifest: dict[str, object] | None = None,
    candidate_binding: dict[str, str] | None = None,
) -> list[tuple[str, bool, str]]:
    """Validate structured exchange evidence; summary booleans are not proof."""

    manifest = manifest or _read_json(
        root / "reports" / "active" / "hyperliquid_run_manifest.json"
    )
    events = receipt.get("events") if isinstance(receipt.get("events"), list) else []
    valid_events = [event for event in events if isinstance(event, dict)]
    event_types = {str(event.get("event_type", "")) for event in valid_events}
    event_ids = [str(event.get("event_id", "")).strip() for event in valid_events]
    event_times = pd.to_datetime(
        [event.get("timestamp_utc") for event in valid_events],
        utc=True,
        errors="coerce",
    )
    required_outcomes = {
        "two_leg_entry",
        "two_leg_exit",
        "reconciled",
        "idempotency",
    }
    references_valid = bool(valid_events) and all(
        isinstance(event.get("exchange_reference_ids"), list)
        and bool(event.get("exchange_reference_ids"))
        and all(str(value).strip() for value in event["exchange_reference_ids"])
        for event in valid_events
    )
    two_leg_entry = next(
        (event for event in valid_events if event.get("event_type") == "two_leg_entry"),
        {},
    )
    two_leg_exit = next(
        (event for event in valid_events if event.get("event_type") == "two_leg_exit"),
        {},
    )
    entry_evidence = _two_leg_event_evidence(two_leg_entry)
    exit_evidence = _two_leg_event_evidence(two_leg_exit)
    anomaly_types = {
        str(event.get("anomaly_type", ""))
        for event in valid_events
        if event.get("event_type") == "execution_anomaly"
    }
    anomaly_detected = bool(anomaly_types)
    partial_anomaly = "partial_fill" in anomaly_types
    orphan_anomaly = "orphan_position" in anomaly_types
    recovery_event_types = {
        str(event.get("event_type", ""))
        for event in valid_events
        if event.get("event_type")
        in {"partial_fill_recovery", "cancel_and_unwind", "emergency_flatten"}
    }
    recovery_reduce_only = all(
        event.get("reduce_only") is True
        for event in valid_events
        if event.get("event_type") in recovery_event_types
    )
    idempotency = next(
        (event for event in valid_events if event.get("event_type") == "idempotency"),
        {},
    )
    final_state = receipt.get("final_state") if isinstance(receipt.get("final_state"), dict) else {}
    position_x = _number(final_state.get("position_x"))
    position_y = _number(final_state.get("position_y"))
    open_order_count = _number(final_state.get("open_order_count"))
    final_timestamp = pd.to_datetime(
        final_state.get("account_state_timestamp_utc"), utc=True, errors="coerce"
    )
    started = pd.to_datetime(receipt.get("started_at_utc"), utc=True, errors="coerce")
    completed = pd.to_datetime(receipt.get("completed_at_utc"), utc=True, errors="coerce")
    terminal_fill_times = pd.to_datetime(
        [
            fill.get("fill_time_ms")
            for event in (two_leg_entry, two_leg_exit)
            for fill in (
                event.get("fill_evidence")
                if isinstance(event.get("fill_evidence"), list)
                else []
            )
            if isinstance(fill, dict)
        ],
        unit="ms",
        utc=True,
        errors="coerce",
    )
    event_timestamps_valid = bool(
        len(event_times) == len(valid_events)
        and not event_times.isna().any()
        and event_times.is_monotonic_increasing
    )
    temporal_order_valid = bool(
        pd.notna(started)
        and pd.notna(completed)
        and pd.notna(final_timestamp)
        and started <= completed
        and final_timestamp <= completed
        and (
            len(terminal_fill_times) == 0
            or (
                not terminal_fill_times.isna().any()
                and final_timestamp >= terminal_fill_times.max()
            )
        )
        and (len(event_times) == 0 or (event_times >= started).all())
        and (len(event_times) == 0 or (event_times <= completed).all())
    )
    flat_reconciled = bool(
        final_state.get("reconciled") is True
        and position_x == 0.0
        and position_y == 0.0
        and open_order_count == 0.0
    )
    supplied_hash = str(receipt.get("receipt_hash", "")).strip().lower()
    expected_hash = _testnet_receipt_payload_hash(receipt)
    economics_valid = _testnet_receipt_economics_valid(receipt)
    fills_match_approved_intents = _testnet_receipt_fills_match_approval(
        receipt=receipt,
        approval=approval,
    )
    candidate_binding = candidate_binding or _testnet_candidate_binding(root)
    receipt_candidate_bound = bool(
        all(candidate_binding.get(field) for field in TESTNET_CANDIDATE_BINDING_FIELDS)
        and all(
            receipt.get(field) == candidate_binding.get(field)
            and receipt.get(field) == approval.get(field)
            for field in TESTNET_CANDIDATE_BINDING_FIELDS
        )
    )
    receipt_entry_context_bound = bool(
        isinstance(receipt.get("entry_context"), dict)
        and receipt.get("entry_context") == approval.get("entry_context")
    )
    receipt_risk_policy_bound = bool(
        receipt.get("risk_override_applied") is False
        and receipt.get("risk_override_applied") == approval.get("risk_override_applied")
    )
    return [
        (
            "receipt_schema",
            receipt.get("receipt_version") == "hyperliquid-testnet-lifecycle-v3",
            "testnet_lifecycle_receipt_schema_invalid",
        ),
        (
            "receipt_testnet_scope",
            receipt.get("actual_testnet") is True
            and str(receipt.get("network", "")).lower() == "testnet"
            and receipt.get("receipt_source") == "hyperliquid_testnet_lifecycle_evidence_capture",
            "testnet_lifecycle_receipt_scope_invalid",
        ),
        (
            "receipt_approval_bound",
            bool(approval.get("approval_id"))
            and receipt.get("approval_id") == approval.get("approval_id"),
            "testnet_lifecycle_receipt_approval_mismatch",
        ),
        (
            "receipt_candidate_identity_bound",
            receipt_candidate_bound,
            "testnet_lifecycle_candidate_identity_mismatch",
        ),
        (
            "receipt_entry_context_bound",
            receipt_entry_context_bound,
            "testnet_lifecycle_entry_context_mismatch",
        ),
        (
            "receipt_risk_policy_bound",
            receipt_risk_policy_bound,
            "testnet_lifecycle_risk_policy_override_or_mismatch",
        ),
        (
            "receipt_run_bound",
            bool(manifest.get("run_id")) and receipt.get("run_id") == manifest.get("run_id"),
            "testnet_lifecycle_receipt_run_mismatch",
        ),
        (
            "receipt_candidate_set_bound",
            bool(manifest.get("candidate_set_id"))
            and receipt.get("candidate_set_id") == manifest.get("candidate_set_id"),
            "testnet_lifecycle_receipt_candidate_set_mismatch",
        ),
        (
            "receipt_protocol_bound",
            bool(protocol.get("protocol_id"))
            and receipt.get("protocol_id") == protocol.get("protocol_id"),
            "testnet_lifecycle_receipt_protocol_mismatch",
        ),
        (
            "receipt_events_complete",
            required_outcomes.issubset(event_types),
            "testnet_lifecycle_required_events_missing",
        ),
        (
            "receipt_event_identity",
            bool(event_ids)
            and all(event_ids)
            and len(event_ids) == len(set(event_ids))
            and references_valid,
            "testnet_lifecycle_event_identity_or_exchange_reference_invalid",
        ),
        (
            "receipt_event_timestamps",
            event_timestamps_valid and temporal_order_valid,
            "testnet_lifecycle_event_timestamps_invalid",
        ),
        (
            "two_leg_entry_receipt",
            entry_evidence,
            "two_leg_entry_not_proven",
        ),
        (
            "two_leg_exit_receipt",
            exit_evidence,
            "two_leg_exit_not_proven",
        ),
        (
            "receipt_fills_match_approved_intents",
            fills_match_approved_intents,
            "testnet_lifecycle_fills_do_not_match_approved_two_leg_intents",
        ),
        (
            "partial_fill_recovery",
            not partial_anomaly
            or ("partial_fill_recovery" in event_types and recovery_reduce_only),
            "partial_fill_recovery_not_proven",
        ),
        (
            "cancel_and_unwind",
            not anomaly_detected or ("cancel_and_unwind" in event_types and recovery_reduce_only),
            "cancel_and_unwind_not_proven",
        ),
        (
            "reconciliation",
            "reconciled" in event_types and flat_reconciled,
            "testnet_reconciliation_not_proven",
        ),
        (
            "idempotency",
            "idempotency" in event_types and idempotency.get("duplicate_submit_blocked") is True,
            "testnet_idempotency_not_proven",
        ),
        (
            "emergency_flatten",
            not orphan_anomaly or ("emergency_flatten" in event_types and recovery_reduce_only),
            "emergency_flatten_not_proven",
        ),
        (
            "receipt_after_cost_economics",
            economics_valid,
            "testnet_lifecycle_after_cost_economics_missing_or_invalid",
        ),
        (
            "receipt_hash_bound",
            bool(supplied_hash) and hmac.compare_digest(supplied_hash, expected_hash),
            "testnet_lifecycle_receipt_hash_missing_or_invalid",
        ),
    ]


def _testnet_receipt_payload_hash(receipt: dict[str, object]) -> str:
    payload = {key: value for key, value in receipt.items() if key != "receipt_hash"}
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return sha256(canonical.encode("utf-8")).hexdigest()


def _testnet_receipt_economics_valid(receipt: dict[str, object]) -> bool:
    events = receipt.get("events") if isinstance(receipt.get("events"), list) else []
    terminal_events = [
        event
        for event in events
        if isinstance(event, dict) and event.get("event_type") in {"two_leg_entry", "two_leg_exit"}
    ]
    fills = [
        fill
        for event in terminal_events
        for fill in (
            event.get("fill_evidence") if isinstance(event.get("fill_evidence"), list) else []
        )
        if isinstance(fill, dict)
    ]
    funding_events = (
        receipt.get("funding_events") if isinstance(receipt.get("funding_events"), list) else []
    )
    economics = receipt.get("economics") if isinstance(receipt.get("economics"), dict) else {}
    started = pd.to_datetime(receipt.get("started_at_utc"), utc=True, errors="coerce")
    completed = pd.to_datetime(receipt.get("completed_at_utc"), utc=True, errors="coerce")
    fill_ids = [str(fill.get("trade_id", "")).strip() for fill in fills]
    transaction_ids = [
        f"{str(fill.get('transaction_hash', '')).strip().lower()}|{trade_id}"
        for fill, trade_id in zip(fills, fill_ids, strict=True)
    ]
    fill_times = pd.to_datetime(
        [fill.get("fill_time_ms") for fill in fills],
        unit="ms",
        utc=True,
        errors="coerce",
    )
    event_fill_bindings_valid = all(
        {
            str(fill.get("order_id", "")).strip()
            for fill in event.get("fill_evidence", [])
            if isinstance(fill, dict)
        }.issubset(
            {
                str(reference).strip()
                for reference in event.get("exchange_reference_ids", [])
                if str(reference).strip()
            }
        )
        for event in terminal_events
    )
    entry_event = next(
        (event for event in terminal_events if event.get("event_type") == "two_leg_entry"),
        {},
    )
    exit_event = next(
        (event for event in terminal_events if event.get("event_type") == "two_leg_exit"),
        {},
    )
    entry_fills = entry_event.get("fill_evidence", [])
    exit_fills = exit_event.get("fill_evidence", [])
    entry_times = pd.to_numeric(
        [fill.get("fill_time_ms") for fill in entry_fills if isinstance(fill, dict)],
        errors="coerce",
    )
    exit_times = pd.to_numeric(
        [fill.get("fill_time_ms") for fill in exit_fills if isinstance(fill, dict)],
        errors="coerce",
    )
    if not (
        {"two_leg_entry", "two_leg_exit"}
        == {str(event.get("event_type", "")) for event in terminal_events}
        and len(fills) >= 4
        and all(fill.get("economics_complete") is True for fill in fills)
        and all(fill_ids)
        and len(fill_ids) == len(set(fill_ids))
        and all(value.split("|", 1)[0] for value in transaction_ids)
        and len(transaction_ids) == len(set(transaction_ids))
        and all(str(fill.get("order_id", "")).strip() for fill in fills)
        and event_fill_bindings_valid
        and pd.notna(started)
        and pd.notna(completed)
        and started <= completed
        and len(fill_times) == len(fills)
        and not fill_times.isna().any()
        and (fill_times >= started).all()
        and (fill_times <= completed).all()
        and len(entry_times) >= 2
        and len(exit_times) >= 2
        and not pd.isna(entry_times).any()
        and not pd.isna(exit_times).any()
        and max(entry_times) <= min(exit_times)
        and all(
            _number(fill.get("price")) is not None
            and (_number(fill.get("price")) or 0.0) > 0
            and _number(fill.get("size")) is not None
            and (_number(fill.get("size")) or 0.0) > 0
            and _number(fill.get("fee_usd")) is not None
            and (_number(fill.get("fee_usd")) or 0.0) >= 0
            and str(fill.get("fee_token", "USDC")).upper() == "USDC"
            and _number(fill.get("closed_pnl_usd")) is not None
            and _number(fill.get("implementation_shortfall_vs_limit_usd")) is not None
            for fill in fills
        )
        and receipt.get("funding_query_complete") is True
        and all(
            isinstance(event, dict)
            and event.get("economics_complete") is True
            and _number(event.get("funding_pnl_usd")) is not None
            for event in funding_events
        )
        and economics.get("economics_complete") is True
    ):
        return False
    gross = sum(float(fill["closed_pnl_usd"]) for fill in fills)
    fees = sum(float(fill["fee_usd"]) for fill in fills)
    shortfall = sum(float(fill["implementation_shortfall_vs_limit_usd"]) for fill in fills)
    funding_pnl = sum(float(event["funding_pnl_usd"]) for event in funding_events)
    expected = {
        "gross_realized_pnl_usd": gross,
        "fees_usd": fees,
        "funding_pnl_usd": funding_pnl,
        "implementation_shortfall_vs_limit_usd": shortfall,
        "net_realized_pnl_after_cost_usd": gross - fees + funding_pnl,
    }
    return bool(
        int(_number(economics.get("terminal_fill_count")) or -1) == len(fills)
        and all(
            _number(economics.get(key)) is not None and abs(float(economics[key]) - value) <= 1e-9
            for key, value in expected.items()
        )
        and economics.get("slippage_treatment")
        == "diagnostic_only_already_reflected_in_realized_pnl_not_double_subtracted"
    )


def _testnet_candidate_binding(root: Path) -> dict[str, str]:
    candidate = _read_json(root / "reports" / "active" / "testnet_candidate_receipt.json")
    from quant_platform.orchestration.corrective_release_gates import (
        _validated_testnet_candidate_receipt,
    )

    ready, _ = _validated_testnet_candidate_receipt(root=root, candidate=candidate)
    return {
        field: str(candidate.get(field, "")).strip() if ready else ""
        for field in TESTNET_CANDIDATE_BINDING_FIELDS
    }


def _testnet_candidate_assets(root: Path) -> set[str]:
    candidate = _read_json(root / "reports" / "active" / "testnet_candidate_receipt.json")
    from quant_platform.orchestration.corrective_release_gates import (
        _validated_testnet_candidate_receipt,
    )

    ready, _ = _validated_testnet_candidate_receipt(root=root, candidate=candidate)
    if not ready:
        return set()
    assets = {
        str(candidate.get("asset_x", "")).strip().upper(),
        str(candidate.get("asset_y", "")).strip().upper(),
    } - {""}
    if len(assets) == 2:
        return assets
    parts = [
        value.strip().upper()
        for value in str(candidate.get("pair_group_key", "")).split("|")
        if value.strip()
    ]
    return set(parts[-2:]) if len(parts) >= 2 else set()


def _immutable_testnet_candidate_for_approval(
    root: Path,
    approval: dict[str, object],
) -> dict[str, object]:
    """Resolve the content-addressed candidate that authorized an open lifecycle."""

    receipt_id = str(approval.get("candidate_receipt_id", "")).strip()
    if not receipt_id.startswith("testnetcandidate_"):
        return {}
    path = root / "data" / "testnet" / "candidate_receipts" / f"{receipt_id}.json"
    candidate = _read_json(path)
    if not candidate:
        return {}
    from quant_platform.orchestration.corrective_release_gates import (
        _candidate_identity_core,
        _payload_hash,
    )

    core = _candidate_identity_core(candidate)
    expected_id = "testnetcandidate_" + _payload_hash(core)[:20]
    expected = {**core, "candidate_receipt_id": expected_id}
    expected["receipt_sha256"] = _payload_hash(expected)
    if candidate != expected or expected_id != receipt_id:
        return {}
    return candidate


def _candidate_assets(candidate: dict[str, object]) -> set[str]:
    assets = {
        str(candidate.get("asset_x", "")).strip().upper(),
        str(candidate.get("asset_y", "")).strip().upper(),
    } - {""}
    if len(assets) == 2:
        return assets
    parts = [
        value.strip().upper()
        for value in str(candidate.get("pair_group_key", "")).split("|")
        if value.strip()
    ]
    return set(parts[-2:]) if len(parts) >= 2 else set()


def _two_leg_event_evidence(event: dict[str, object]) -> bool:
    references = event.get("exchange_reference_ids", [])
    terminal_fill_statuses = {"filled", "fully_filled", "testnet_filled"}
    leg_x_status = str(event.get("leg_x_status", "")).strip().lower()
    leg_y_status = str(event.get("leg_y_status", "")).strip().lower()
    return bool(
        isinstance(references, list)
        and len({str(value) for value in references if str(value).strip()}) >= 2
        and leg_x_status in terminal_fill_statuses
        and leg_y_status in terminal_fill_statuses
    )


def _testnet_receipt_fills_match_approval(
    *, receipt: dict[str, object], approval: dict[str, object]
) -> bool:
    approved_legs = (
        approval.get("legs") if isinstance(approval.get("legs"), list) else []
    )
    if len(approved_legs) != 2:
        return False
    approved: dict[str, tuple[str, float]] = {}
    for leg in approved_legs:
        if not isinstance(leg, dict):
            return False
        market = _testnet_market(str(leg.get("market", "")))
        side = str(leg.get("side", "")).strip().upper()
        size = _number(leg.get("size"))
        if not market or side not in {"BUY", "SELL"} or size is None or size <= 0:
            return False
        approved[market] = (side, size)
    if len(approved) != 2 or {side for side, _ in approved.values()} != {"BUY", "SELL"}:
        return False

    events = receipt.get("events") if isinstance(receipt.get("events"), list) else []
    terminal = [
        event
        for event in events
        if isinstance(event, dict)
        and event.get("event_type") in {"two_leg_entry", "two_leg_exit"}
    ]
    if len(terminal) != 2 or {
        str(event.get("event_type", "")) for event in terminal
    } != {"two_leg_entry", "two_leg_exit"}:
        return False

    for event in terminal:
        references = {
            str(reference).strip()
            for reference in event.get("exchange_reference_ids", [])
            if str(reference).strip()
        }
        fills = (
            event.get("fill_evidence")
            if isinstance(event.get("fill_evidence"), list)
            else []
        )
        if len(references) != 2 or not fills or event.get("fill_economics_complete") is not True:
            return False
        expected_exit = event.get("event_type") == "two_leg_exit"
        filled_sizes = {market: 0.0 for market in approved}
        order_ids_by_market = {market: set() for market in approved}
        observed_order_ids: set[str] = set()
        for fill in fills:
            if not isinstance(fill, dict):
                return False
            market = _testnet_market(str(fill.get("coin", "")))
            order_id = str(fill.get("order_id", "")).strip()
            size = _number(fill.get("size"))
            side = str(fill.get("intent_side", "")).strip().upper()
            if market not in approved or order_id not in references or size is None or size <= 0:
                return False
            approved_side, _ = approved[market]
            expected_side = (
                "SELL" if approved_side == "BUY" else "BUY"
            ) if expected_exit else approved_side
            if side != expected_side:
                return False
            filled_sizes[market] += size
            order_ids_by_market[market].add(order_id)
            observed_order_ids.add(order_id)
        if observed_order_ids != references or any(
            len(order_ids) != 1 for order_ids in order_ids_by_market.values()
        ):
            return False
        for market, (_, expected_size) in approved.items():
            tolerance = max(1e-12, expected_size * 1e-8)
            if abs(filled_sizes[market] - expected_size) > tolerance:
                return False
    return True


def _testnet_market(value: str) -> str:
    return value.strip().upper().removesuffix("-USD").removesuffix("/USD")


def _approval_secret_from_keychain(config: HyperliquidTestnetConfig) -> str | None:
    service = os.getenv("HYPERLIQUID_TESTNET_APPROVAL_KEYCHAIN_SERVICE", "").strip()
    account = os.getenv("HYPERLIQUID_TESTNET_APPROVAL_KEYCHAIN_ACCOUNT", "").strip()
    if not service or not account:
        return None
    try:
        secret = read_hyperliquid_agent_key_from_keychain(service, account).strip()
    except Exception:  # noqa: BLE001 - keychain adapters expose platform-specific errors.
        return None
    return secret or None


def _pair_row(frame: pd.DataFrame, pair: str) -> pd.Series:
    if frame.empty or "pair" not in frame.columns:
        return pd.Series(dtype=object)
    rows = frame.loc[frame["pair"].astype(str).eq(pair)]
    return rows.iloc[0] if not rows.empty else pd.Series(dtype=object)


def _number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _mode_training_manifest(dataset: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "exact_mode",
        "rows",
        "pairs",
        "timeframes",
        "label_classes",
        "positive_label_rate",
        "sample_weight_total",
        "training_role",
        "status",
        "blocker",
    ]
    if dataset.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    for mode, group in dataset.groupby("exact_mode", sort=True):
        pairs = int(group["pair"].astype(str).nunique())
        timeframes = int(group["timeframe"].astype(str).nunique())
        labels = pd.to_numeric(group["good_trade"], errors="coerce").dropna()
        blockers: list[str] = []
        if len(group) < 25:
            blockers.append("mode_training_rows_below_25")
        if pairs < 3:
            blockers.append("mode_pair_coverage_below_3")
        if labels.nunique() < 2:
            blockers.append("mode_label_diversity_missing")
        rows.append(
            {
                "exact_mode": mode,
                "rows": len(group),
                "pairs": pairs,
                "timeframes": timeframes,
                "label_classes": int(labels.nunique()),
                "positive_label_rate": float(labels.mean()) if not labels.empty else np.nan,
                "sample_weight_total": float(
                    pd.to_numeric(group["sample_weight"], errors="coerce").sum()
                ),
                "training_role": "dedicated_mode_specialist",
                "status": "READY_FOR_SHADOW_TRAINING" if not blockers else "BLOCKED",
                "blocker": ";".join(blockers),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    promote_staged_file(temporary, path)


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    promote_staged_file(temporary, path)


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest() if path.is_file() else ""


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame()


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}
