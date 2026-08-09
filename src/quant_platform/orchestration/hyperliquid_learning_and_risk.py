"""Council-aligned learning rows, portfolio veto, and Testnet lifecycle gates."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import hmac
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from quant_platform.hyperliquid_testnet import HyperliquidTestnetConfig, read_hyperliquid_agent_key_from_keychain
from quant_platform.orchestration.student_readiness import STUDENT_TRAINING_COLUMNS, write_student_training_readiness


ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class PortfolioPolicy:
    max_pairs: int = 5
    max_shared_asset_share: float = 0.40
    max_gross_allocation: float = 1.0
    minimum_margin_buffer: float = 0.50


def materialize_council_learning_dataset(*, root: Path = ROOT) -> dict[str, object]:
    """Prefer OOS trade-entry rows, with summary rows as an explicit fallback."""

    source_path = root / "reports" / "orchestration" / "teacher_council" / "teacher_evidence_coverage.csv"
    trade_source_path = root / "reports" / "orchestration" / "teacher_council" / "walkforward_trade_events.csv"
    auxiliary_trade_path = root / "reports" / "orchestration" / "teacher_council" / "auxiliary_4h_walkforward_trade_events.csv"
    source = _read_csv(source_path)
    primary_trades = _read_csv(trade_source_path)
    auxiliary_trades = _read_csv(auxiliary_trade_path)
    if not primary_trades.empty and not auxiliary_trades.empty and "run_id" in primary_trades.columns and "run_id" in auxiliary_trades.columns:
        current_runs = set(primary_trades["run_id"].dropna().astype(str))
        auxiliary_trades = auxiliary_trades.loc[auxiliary_trades["run_id"].astype(str).isin(current_runs)].copy()
    trade_source = pd.concat([primary_trades, auxiliary_trades], ignore_index=True)
    if not trade_source.empty and "training_event_id" in trade_source.columns:
        trade_source = trade_source.drop_duplicates(subset=["training_event_id"], keep="last")
    dataset_path = root / "data" / "ml" / "student_teacher_training_dataset.csv"
    mode_manifest_path = root / "reports" / "orchestration" / "teacher_council" / "student_mode_training_manifest.csv"
    summary_path = root / "reports" / "orchestration" / "teacher_council" / "student_dataset_materialization.md"
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
            inverse = dataset["exact_mode"].astype(str).map(lambda value: 1.0 / float(mode_counts[value]))
            dataset["sample_weight"] = inverse / float(inverse.mean())
            dataset["training_topology"] = "per_mode_specialists_with_weighted_pooled_baseline"
        for column in STUDENT_TRAINING_COLUMNS:
            if column not in dataset.columns:
                dataset[column] = np.nan
        ordered = [*STUDENT_TRAINING_COLUMNS, *[column for column in dataset.columns if column not in STUDENT_TRAINING_COLUMNS]]
        dataset = dataset.loc[:, ordered]
        _atomic_csv(dataset, dataset_path)
        mode_manifest = _mode_training_manifest(dataset)
        _atomic_csv(mode_manifest, mode_manifest_path)
        readiness = write_student_training_readiness(root=root, dataset_path=dataset_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            "\n".join(
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
            ),
            encoding="utf-8",
        )
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
                "training_event_id": "training_" + sha256(event_material.encode("utf-8")).hexdigest()[:20],
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
    summary_path.write_text(
        "\n".join(
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
        ),
        encoding="utf-8",
    )
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
        run_match = decisions.get("run_id", pd.Series("", index=decisions.index)).astype(str).eq(manifest_ids[0])
        set_match = decisions.get("candidate_set_id", pd.Series("", index=decisions.index)).astype(str).eq(manifest_ids[1])
        eligible = decisions.loc[status.eq("SHADOW_TEST") & action.ne("abstain") & run_match & set_match].copy()
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
        margin_blockers = [value for value in str(margin.iloc[0].get("blockers", "")).split(";") if value]
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
        "evidence_path": ";".join(
            [
                "reports/orchestration/teacher_council/council_decisions.csv",
                "reports/active/hyperliquid_run_candidates.csv",
                "reports/active/hyperliquid_testnet_margin_snapshot.csv",
            ]
        ),
    }
    frame = pd.DataFrame([row])
    path = directory / "portfolio_critic.csv"
    markdown = directory / "portfolio_critic.md"
    _atomic_csv(frame, path)
    markdown.write_text("# Portfolio Critic\n\n" + frame.to_markdown(index=False) + "\n", encoding="utf-8")
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
    protocol_path = (
        active / "current_wizard_hyperliquid_testnet_protocol_manifest.json"
    )
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
        ("no_order_preflight", bool(not preflight.empty and preflight.get("ready_for_no_order_preflight", pd.Series([False])).map(_truthy).iloc[0]), "hyperliquid_no_order_preflight_not_ready"),
        (
            "deterministic_protocol_simulation",
            protocol_ready,
            "deterministic_testnet_protocol_validation_missing_or_failed",
        ),
        *_testnet_approval_checks(approval, root=root, approval_secret=approval_secret, config=config),
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
    markdown.write_text(
        "# Hyperliquid Testnet Lifecycle Gate\n\nNo order is submitted by this check.\n\n" + frame.to_markdown(index=False) + "\n",
        encoding="utf-8",
    )
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
    if not path.exists():
        payload = {
            "approval_version": "hyperliquid-testnet-smoke-v4",
            "approved": False,
            "approval_id": "",
            "run_id": str(manifest.get("run_id", "")),
            "candidate_set_id": str(manifest.get("candidate_set_id", "")),
            "master_address": str(resolved.master_address or ""),
            "agent_address": str(resolved.agent_address or ""),
            "network": "testnet",
            "one_run_only": True,
            "expires_at_utc": "",
            "max_total_notional_usd": 25.0,
            "requested_leverage": int(resolved.requested_leverage),
            "margin_mode": str(resolved.margin_mode),
            "one_x_testnet_proof_id": str(
                resolved.one_x_testnet_proof_id or ""
            ),
            "leverage_scenario_id": str(resolved.leverage_scenario_id or ""),
            "purpose": "bounded two-leg lifecycle proof only",
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
                "maximum_total_notional_usd": 25.0,
            },
            "payload_hash": "",
            "approval_signature": "",
            "signature_scheme": "hmac-sha256-keychain-v1",
            "execution_authority": "none_until_user_sets_approved_true_for_this_exact_payload",
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(path)
        created = True
    else:
        payload = _read_json(path)
        if payload.get("approved") is not True:
            expected = {
                "approval_version": "hyperliquid-testnet-smoke-v4",
                "run_id": str(manifest.get("run_id", "")),
                "candidate_set_id": str(manifest.get("candidate_set_id", "")),
                "master_address": str(resolved.master_address or ""),
                "agent_address": str(resolved.agent_address or ""),
                "max_total_notional_usd": 25.0,
                "requested_leverage": int(resolved.requested_leverage),
                "margin_mode": str(resolved.margin_mode),
                "one_x_testnet_proof_id": str(
                    resolved.one_x_testnet_proof_id or ""
                ),
                "leverage_scenario_id": str(
                    resolved.leverage_scenario_id or ""
                ),
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
                    "maximum_total_notional_usd": 25.0,
                },
                "payload_hash": "",
                "approval_signature": "",
                "signature_scheme": "hmac-sha256-keychain-v1",
            }
            changed = False
            for key, value in expected.items():
                if payload.get(key) != value:
                    payload[key] = value
                    changed = True
            if changed:
                temporary = path.with_suffix(path.suffix + ".tmp")
                temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
                temporary.replace(path)
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
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(approval, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)
    return {"status": "SIGNED_TESTNET_ONLY", "approval": path, "payload_hash": payload_hash}


def validate_testnet_smoke_approval(
    *,
    approval_id: str,
    root: Path = ROOT,
    approval_secret: str | None = None,
    config: HyperliquidTestnetConfig | None = None,
) -> dict[str, object]:
    """Verify the signed one-run artifact at the final executor boundary."""

    path = root / "reports" / "active" / "hyperliquid_testnet_smoke_approval.json"
    approval = _read_json(path)
    checks = _testnet_approval_checks(
        approval,
        root=root,
        approval_secret=approval_secret,
        config=config,
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
    blockers = [blocker for _, passed, blocker in checks if not passed]
    return {
        "status": "PASS" if not blockers else "BLOCKED",
        "approval": path,
        "checks": len(checks),
        "blockers": blockers,
        "execution_allowed": False,
        "live_trading_authorized": False,
    }


def _testnet_approval_checks(
    approval: dict[str, object],
    *,
    root: Path,
    approval_secret: str | None = None,
    config: HyperliquidTestnetConfig | None = None,
    require_signature: bool = True,
) -> list[tuple[str, bool, str]]:
    legs = approval.get("legs") if isinstance(approval.get("legs"), list) else []
    approval_id = str(approval.get("approval_id", "")).strip()
    expires = pd.to_datetime(approval.get("expires_at_utc"), utc=True, errors="coerce")
    now = pd.Timestamp(datetime.now(timezone.utc))
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
            if not market or side not in {"BUY", "SELL"} or size is None or size <= 0.0 or price is None or price <= 0.0:
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
            (_number(leg.get("size")) or 0.0)
            * (_number(leg.get("limit_price")) or 0.0)
            >= 10.0
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
            and approval.get("one_x_testnet_proof_id")
            == resolved.one_x_testnet_proof_id
            and approval.get("leverage_scenario_id")
            == resolved.leverage_scenario_id
        )
    )
    manifest = _read_json(root / "reports" / "active" / "hyperliquid_run_manifest.json")
    expected_hash = _approval_payload_hash(approval)
    supplied_hash = str(approval.get("payload_hash", "")).strip().lower()
    secret = approval_secret or _approval_secret_from_keychain(resolved)
    expected_signature = (
        hmac.new(secret.encode("utf-8"), expected_hash.encode("ascii"), digestmod="sha256").hexdigest()
        if secret
        else ""
    )
    supplied_signature = str(approval.get("approval_signature", "")).strip().lower()
    checks = [
        ("explicit_smoke_approval", approval.get("approved") is True, "explicit_testnet_smoke_approval_missing"),
        ("approval_id", bool(approval_id), "testnet_smoke_approval_id_missing"),
        ("testnet_only_scope", str(approval.get("network", "")).lower() == "testnet", "testnet_smoke_scope_not_testnet"),
        ("one_run_only_scope", approval.get("one_run_only") is True, "testnet_smoke_not_one_run_only"),
        ("approval_not_expired", bool(pd.notna(expires) and expires > now), "testnet_smoke_approval_missing_or_expired"),
        ("two_leg_payload_valid", leg_shape_valid, "testnet_smoke_two_leg_payload_invalid"),
        (
            "notional_bounded",
            bounded,
            "testnet_smoke_notional_not_10_usd_per_leg_or_above_25_usd_total",
        ),
        ("approval_version", approval.get("approval_version") == "hyperliquid-testnet-smoke-v4", "testnet_smoke_approval_version_invalid"),
        (
            "exit_policy_bound",
            approval.get("exit_policy")
            == {
                "reduce_only": True,
                "opposite_side": True,
                "market_scope": "approved_entry_markets",
                "maximum_size": "approved_entry_size_per_leg",
                "maximum_total_notional_usd": 25.0,
            },
            "testnet_smoke_exit_policy_invalid",
        ),
        ("requested_leverage_bound", leverage_matches, "testnet_smoke_requested_leverage_mismatch"),
        ("margin_mode_bound", margin_mode_matches, "testnet_smoke_margin_mode_mismatch"),
        ("leverage_evidence_bound", leverage_evidence_matches, "testnet_smoke_leverage_evidence_missing_or_mismatch"),
        ("run_id_bound", bool(manifest.get("run_id")) and approval.get("run_id") == manifest.get("run_id"), "testnet_smoke_run_id_mismatch"),
        ("candidate_set_bound", bool(manifest.get("candidate_set_id")) and approval.get("candidate_set_id") == manifest.get("candidate_set_id"), "testnet_smoke_candidate_set_id_mismatch"),
        ("master_address_bound", bool(resolved.master_address) and str(approval.get("master_address", "")).lower() == str(resolved.master_address).lower(), "testnet_smoke_master_address_mismatch"),
        ("agent_address_bound", bool(resolved.agent_address) and str(approval.get("agent_address", "")).lower() == str(resolved.agent_address).lower(), "testnet_smoke_agent_address_mismatch"),
    ]
    if require_signature:
        checks.extend(
            [
                ("payload_hash_bound", bool(supplied_hash) and hmac.compare_digest(supplied_hash, expected_hash), "testnet_smoke_payload_hash_mismatch"),
                ("approval_signature", bool(expected_signature) and hmac.compare_digest(supplied_signature, expected_signature), "testnet_smoke_approval_signature_missing_or_invalid"),
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
            "expires_at_utc",
            "max_total_notional_usd",
            "requested_leverage",
            "margin_mode",
            "one_x_testnet_proof_id",
            "leverage_scenario_id",
            "purpose",
            "legs",
            "required_outcomes",
            "exit_policy",
        )
    }
    canonical = json.dumps(signed_fields, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256(canonical.encode("utf-8")).hexdigest()


def _testnet_receipt_checks(
    receipt: dict[str, object],
    *,
    approval: dict[str, object],
    protocol: dict[str, object],
    root: Path,
) -> list[tuple[str, bool, str]]:
    """Validate structured exchange evidence; summary booleans are not proof."""

    manifest = _read_json(root / "reports" / "active" / "hyperliquid_run_manifest.json")
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
    final_state = (
        receipt.get("final_state")
        if isinstance(receipt.get("final_state"), dict)
        else {}
    )
    position_x = _number(final_state.get("position_x"))
    position_y = _number(final_state.get("position_y"))
    open_order_count = _number(final_state.get("open_order_count"))
    final_timestamp = pd.to_datetime(
        final_state.get("account_state_timestamp_utc"), utc=True, errors="coerce"
    )
    started = pd.to_datetime(receipt.get("started_at_utc"), utc=True, errors="coerce")
    completed = pd.to_datetime(receipt.get("completed_at_utc"), utc=True, errors="coerce")
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
    return [
        (
            "receipt_schema",
            receipt.get("receipt_version") == "hyperliquid-testnet-lifecycle-v2",
            "testnet_lifecycle_receipt_schema_invalid",
        ),
        (
            "receipt_testnet_scope",
            receipt.get("actual_testnet") is True
            and str(receipt.get("network", "")).lower() == "testnet"
            and receipt.get("receipt_source")
            == "hyperliquid_testnet_lifecycle_evidence_capture",
            "testnet_lifecycle_receipt_scope_invalid",
        ),
        (
            "receipt_approval_bound",
            bool(approval.get("approval_id"))
            and receipt.get("approval_id") == approval.get("approval_id"),
            "testnet_lifecycle_receipt_approval_mismatch",
        ),
        (
            "receipt_run_bound",
            bool(manifest.get("run_id"))
            and receipt.get("run_id") == manifest.get("run_id"),
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
            "partial_fill_recovery",
            not partial_anomaly
            or (
                "partial_fill_recovery" in event_types and recovery_reduce_only
            ),
            "partial_fill_recovery_not_proven",
        ),
        (
            "cancel_and_unwind",
            not anomaly_detected
            or ("cancel_and_unwind" in event_types and recovery_reduce_only),
            "cancel_and_unwind_not_proven",
        ),
        (
            "reconciliation",
            "reconciled" in event_types and flat_reconciled,
            "testnet_reconciliation_not_proven",
        ),
        (
            "idempotency",
            "idempotency" in event_types
            and idempotency.get("duplicate_submit_blocked") is True,
            "testnet_idempotency_not_proven",
        ),
        (
            "emergency_flatten",
            not orphan_anomaly
            or ("emergency_flatten" in event_types and recovery_reduce_only),
            "emergency_flatten_not_proven",
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


def _approval_secret_from_keychain(config: HyperliquidTestnetConfig) -> str | None:
    service = os.getenv("HYPERLIQUID_TESTNET_APPROVAL_KEYCHAIN_SERVICE", "").strip()
    account = os.getenv("HYPERLIQUID_TESTNET_APPROVAL_KEYCHAIN_ACCOUNT", "").strip()
    if not service or not account:
        return None
    try:
        secret = read_hyperliquid_agent_key_from_keychain(service, account).strip()
    except Exception:
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
                "sample_weight_total": float(pd.to_numeric(group["sample_weight"], errors="coerce").sum()),
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
    temporary.replace(path)


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
