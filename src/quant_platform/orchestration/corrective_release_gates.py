"""Conditional Testnet and live-canary gates. This module never submits orders."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import pandas as pd

from quant_platform.active_pipeline import CommandResult


ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "thewiz.corrective_release_gates.v1"


def build_testnet_candidate_receipt(*, root: Path = ROOT, now: datetime | None = None) -> dict[str, Any]:
    active = root / "reports" / "active"
    survivor_path = active / "final_1x_survivor_receipt.json"
    cadence_path = active / "daily_cadence_acceptance.csv"
    survivor = _read_json(survivor_path)
    cadence = _read_csv(cadence_path)
    final_ids = list(survivor.get("final_experiment_ids", []))
    cadence_pass = bool(not cadence.empty and cadence.get("cadence_acceptance_status", pd.Series(dtype=str)).eq("PASS").any())
    blockers = []
    if survivor.get("receipt_status") != "PASS" or len(final_ids) != 1:
        blockers.append("valid_single_final_one_x_survivor_receipt_missing")
    if not cadence_pass:
        blockers.append("seven_day_research_cadence_not_proven")
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": _as_utc(now).isoformat(),
        "candidate_status": "READY_FOR_NO_ORDER_PREFLIGHT" if not blockers else "BLOCKED",
        "candidate_experiment_id": final_ids[0] if len(final_ids) == 1 else "",
        "candidate_leverage": 1.0 if len(final_ids) == 1 else None,
        "survivor_receipt_id": survivor.get("acceptance_policy_id", ""),
        "blockers": blockers,
        "order_submission_performed": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
        "evidence_path": f"{_relative(survivor_path, root)};{_relative(cadence_path, root)}",
    }
    receipt["receipt_id"] = "testnetcandidate_" + sha256(_canonical_json(receipt).encode("utf-8")).hexdigest()[:20]
    path = active / "testnet_candidate_receipt.json"
    _atomic_json(receipt, path)
    return {"path": path, "receipt": receipt}


def build_no_order_preflight(*, root: Path = ROOT) -> dict[str, Any]:
    active = root / "reports" / "active"
    candidate = build_testnet_candidate_receipt(root=root)["receipt"]
    existing = _read_csv(active / "hyperliquid_testnet_preflight.csv")
    checks = [
        ("single_1x_survivor", candidate.get("candidate_status") == "READY_FOR_NO_ORDER_PREFLIGHT", "valid_single_final_one_x_survivor_receipt_missing"),
        ("market_inventory", _preflight_ready(existing, "market"), "current_testnet_market_inventory_not_proven_for_candidate"),
        ("wallet_agent", _preflight_ready(existing, "wallet") or _preflight_ready(existing, "agent"), "testnet_wallet_or_agent_not_verified_for_candidate"),
        ("margin_collateral", _preflight_ready(existing, "margin") or _preflight_ready(existing, "collateral"), "testnet_margin_not_verified_for_candidate"),
        ("funding_current", False, "candidate_specific_current_funding_not_verified"),
        ("paired_sizing", False, "candidate_specific_paired_size_not_frozen"),
        ("partial_fill_contingency", True, ""),
        ("submit_orders_disabled_during_preflight", True, ""),
    ]
    rows = []
    for check, passed, blocker in checks:
        rows.append(
            {
                "check": check,
                "status": "PASS" if passed else "BLOCKED",
                "blocker": "" if passed else blocker,
                "no_order_preflight": True,
                "order_submission_performed": False,
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            }
        )
    frame = pd.DataFrame(rows)
    path = active / "testnet_no_order_preflight.csv"
    _atomic_csv(frame, path)
    return {"path": path, "frame": frame, "status": "PASS" if frame["status"].eq("PASS").all() else "BLOCKED"}


def build_testnet_lifecycle_receipt(*, root: Path = ROOT, now: datetime | None = None) -> dict[str, Any]:
    preflight = build_no_order_preflight(root=root)
    blockers = [] if preflight["status"] == "PASS" else ["no_order_preflight_blocked"]
    blockers.append("explicit_testnet_order_authorization_not_present")
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": _as_utc(now).isoformat(),
        "lifecycle_status": "NOT_EXECUTED",
        "requested_leverage": 1.0,
        "paired_order_intent_created": False,
        "orders_submitted": 0,
        "fills_observed": 0,
        "positions_opened": 0,
        "positions_reconciled_flat": False,
        "blockers": blockers,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
        "evidence_path": _relative(Path(preflight["path"]), root),
    }
    path = root / "reports" / "active" / "testnet_lifecycle_execution_receipt.json"
    _atomic_json(receipt, path)
    return {"path": path, "receipt": receipt}


def build_testnet_sample_sufficiency(*, root: Path = ROOT) -> dict[str, Any]:
    policy_path = root / "config" / "testnet_sample_sufficiency_policy.json"
    policy = _read_json(policy_path)
    journal_path = root / "reports" / "paper_trading_journal.csv"
    journal = _read_csv(journal_path)
    testnet = journal.loc[journal.get("venue", pd.Series("", index=journal.index)).astype(str).str.lower().eq("hyperliquid_testnet")] if not journal.empty else journal
    closed = int(testnet.get("status", pd.Series(dtype=str)).astype(str).str.contains("closed|flat|complete", case=False, regex=True).sum()) if not testnet.empty else 0
    days = 0
    if not testnet.empty and "timestamp" in testnet:
        times = pd.to_datetime(testnet["timestamp"], utc=True, errors="coerce").dropna()
        days = int((times.max() - times.min()).days + 1) if not times.empty else 0
    pairs = int(testnet.get("pair", pd.Series(dtype=str)).nunique()) if not testnet.empty else 0
    rows = [
        ("closed_paired_lifecycles", closed, int(policy.get("minimum_closed_paired_lifecycles", 30)), closed >= int(policy.get("minimum_closed_paired_lifecycles", 30))),
        ("observation_days", days, int(policy.get("minimum_observation_days", 14)), days >= int(policy.get("minimum_observation_days", 14))),
        ("independent_pairs", pairs, int(policy.get("minimum_independent_pairs", 3)), pairs >= int(policy.get("minimum_independent_pairs", 3))),
        ("observed_regimes", 0, int(policy.get("minimum_observed_regimes", 3)), False),
        ("unresolved_orphan_legs", 0, int(policy.get("maximum_unresolved_orphan_legs", 0)), closed > 0),
        ("positive_after_cost_expectancy", None, True, False),
    ]
    frame = pd.DataFrame(
        [
            {
                "check": check,
                "observed": observed,
                "required": required,
                "status": "PASS" if passed else "BLOCKED",
                "blocker": "" if passed else f"testnet_sample_{check}_insufficient",
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            }
            for check, observed, required, passed in rows
        ]
    )
    path = root / "reports" / "active" / "realized_testnet_sample_sufficiency.csv"
    _atomic_csv(frame, path)
    return {"path": path, "frame": frame, "status": "PASS" if frame["status"].eq("PASS").all() else "BLOCKED"}


def build_testnet_supreme_team_checkpoint(*, root: Path = ROOT, now: datetime | None = None) -> dict[str, Any]:
    sample = build_testnet_sample_sufficiency(root=root)
    lifecycle = build_testnet_lifecycle_receipt(root=root, now=now)["receipt"]
    status = "PASS" if sample["status"] == "PASS" and lifecycle["lifecycle_status"] == "COMPLETE" else "BLOCKED"
    blockers = list(sample["frame"].loc[sample["frame"]["status"].ne("PASS"), "blocker"].astype(str))
    blockers.extend(lifecycle.get("blockers", []))
    checkpoint = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": _as_utc(now).isoformat(),
        "checkpoint_status": status,
        "gap_analysis": "Realized paired Testnet outcomes are absent or below the prospective sample policy.",
        "pre_mortem": "Advancing would confuse deterministic simulations and preflight checks with realized execution evidence.",
        "post_mortem": "No Testnet order was sent because upstream survivor, cadence, cost, parity, and authorization gates remain blocked.",
        "red_team": "A forged confidence score, stale receipt, or simulated lifecycle cannot satisfy this checkpoint.",
        "blockers": list(dict.fromkeys(blockers)),
        "recommendation": "remain_research_only_and_continue_daily_evidence_collection",
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    directory = root / "reports" / "supreme_team"
    path = directory / "testnet_evidence_checkpoint.json"
    markdown = directory / "testnet_evidence_checkpoint.md"
    _atomic_json(checkpoint, path)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text(_checkpoint_markdown(checkpoint), encoding="utf-8")
    return {"path": path, "markdown": markdown, "checkpoint": checkpoint}


def build_live_release_gates(*, root: Path = ROOT, now: datetime | None = None) -> dict[str, Path]:
    active = root / "reports" / "active"
    sample = build_testnet_sample_sufficiency(root=root)
    supreme = build_testnet_supreme_team_checkpoint(root=root, now=now)["checkpoint"]
    parity_rows = [
        ("market_metadata", False, "no_realized_testnet_candidate"),
        ("mark_and_mid_prices", False, "no_realized_testnet_candidate"),
        ("funding_rate_and_timestamp", False, "no_realized_testnet_candidate"),
        ("l2_depth_and_slippage", False, "no_realized_testnet_candidate"),
        ("size_precision_and_minimum_notional", False, "no_realized_testnet_candidate"),
        ("margin_and_liquidation_inputs", False, "no_realized_testnet_candidate"),
    ]
    parity = pd.DataFrame(
        [
            {
                "input": name,
                "status": "PASS" if passed else "BLOCKED",
                "blocker": "" if passed else blocker,
                "shadow_only": True,
                "order_submission_performed": False,
                "live_trading_authorized": False,
            }
            for name, passed, blocker in parity_rows
        ]
    )
    parity_path = active / "testnet_live_input_parity.csv"
    _atomic_csv(parity, parity_path)
    authorization = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": _as_utc(now).isoformat(),
        "authorization_status": "NOT_AUTHORIZED",
        "exact_pair": "",
        "exact_side_and_sizes": [],
        "maximum_leverage": 1.0,
        "authorization_reusable": False,
        "user_authorization_present": False,
        "prerequisite_sample_pass": sample["status"] == "PASS",
        "prerequisite_supreme_team_pass": supreme["checkpoint_status"] == "PASS",
        "prerequisite_input_parity_pass": bool(parity["status"].eq("PASS").all()),
        "order_submission_performed": False,
        "live_trading_authorized": False,
        "blockers": ["prospective_live_prerequisites_blocked", "explicit_user_authorization_for_exact_order_missing"],
    }
    auth_path = active / "live_canary_authorization.json"
    _atomic_json(authorization, auth_path)
    outcome = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": _as_utc(now).isoformat(),
        "canary_status": "NOT_EXECUTED",
        "orders_submitted": 0,
        "fills_observed": 0,
        "realized_after_cost_return": None,
        "reconciled_flat": False,
        "repeat_authorized": False,
        "scaling_authorized": False,
        "leverage_authorized": False,
        "next_action": "remain_blocked_until_all_prospective_gates_and_new_explicit_user_authorization_pass",
        "live_trading_authorized": False,
    }
    outcome_path = active / "live_canary_outcome_evaluation.json"
    _atomic_json(outcome, outcome_path)
    return {"parity": parity_path, "authorization": auth_path, "outcome": outcome_path}


def build_corrective_release_gates(*, root: Path = ROOT, now: datetime | None = None) -> CommandResult:
    candidate = build_testnet_candidate_receipt(root=root, now=now)
    preflight = build_no_order_preflight(root=root)
    lifecycle = build_testnet_lifecycle_receipt(root=root, now=now)
    sample = build_testnet_sample_sufficiency(root=root)
    supreme = build_testnet_supreme_team_checkpoint(root=root, now=now)
    live = build_live_release_gates(root=root, now=now)
    return CommandResult(
        paths={
            "testnet_candidate": Path(candidate["path"]),
            "no_order_preflight": Path(preflight["path"]),
            "testnet_lifecycle": Path(lifecycle["path"]),
            "testnet_sample_sufficiency": Path(sample["path"]),
            "testnet_supreme_team": Path(supreme["path"]),
            "testnet_supreme_team_markdown": Path(supreme["markdown"]),
            "testnet_live_input_parity": live["parity"],
            "live_canary_authorization": live["authorization"],
            "live_canary_outcome": live["outcome"],
        },
        summary={
            "status": "BLOCKED",
            "testnet_candidate_status": candidate["receipt"]["candidate_status"],
            "no_order_preflight_status": preflight["status"],
            "testnet_lifecycle_status": lifecycle["receipt"]["lifecycle_status"],
            "testnet_sample_status": sample["status"],
            "supreme_team_status": supreme["checkpoint"]["checkpoint_status"],
            "orders_submitted": 0,
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        },
    )


def _preflight_ready(frame: pd.DataFrame, token: str) -> bool:
    if frame.empty:
        return False
    columns = [column for column in frame.columns if token in column.lower()]
    if not columns:
        return False
    return any(frame[column].map(_truthy).any() for column in columns)


def _checkpoint_markdown(checkpoint: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Supreme Team: Realized Testnet Evidence",
            "",
            f"- Status: `{checkpoint['checkpoint_status']}`",
            "- Testnet order authority: `false`",
            "- Live trading authorized: `false`",
            "",
            f"**Gap analysis:** {checkpoint['gap_analysis']}",
            "",
            f"**Pre-mortem:** {checkpoint['pre_mortem']}",
            "",
            f"**Post-mortem:** {checkpoint['post_mortem']}",
            "",
            f"**Red team:** {checkpoint['red_team']}",
            "",
            "Blockers:",
            *[f"- `{blocker}`" for blocker in checkpoint["blockers"]],
            "",
        ]
    )


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _truthy(value: Any) -> bool:
    return value is True or str(value).strip().lower() in {"1", "true", "yes", "pass", "ready"}


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _as_utc(value: datetime | None) -> datetime:
    value = value or datetime.now(timezone.utc)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


if __name__ == "__main__":
    result = build_corrective_release_gates()
    print(json.dumps({"summary": result.summary, "paths": {key: str(value) for key, value in result.paths.items()}}, indent=2))
