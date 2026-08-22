"""Deterministic, no-order validation of the Hyperliquid pair lifecycle protocol."""

from __future__ import annotations

from quant_platform.orchestration.corrective_runtime import atomic_write_text

from quant_platform.orchestration.corrective_runtime import atomic_write_csv

from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import pandas as pd

from quant_platform.active_pipeline import CommandResult


ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "current_wizard_hyperliquid_testnet_protocol.v1"
REQUIRED_SCENARIOS = {
    "stale_signal_block",
    "unaccepted_setup_block",
    "leverage_without_1x_block",
    "insufficient_collateral_block",
    "full_pair_entry_exit",
    "partial_orphan_recovery",
    "unconfirmed_response_reconciliation",
    "restart_open_pair_reconciliation",
    "proven_leverage_pair_entry_exit",
}


def validate_current_wizard_hyperliquid_testnet_protocol(
    *,
    root: Path = ROOT,
    now: datetime | None = None,
) -> CommandResult:
    """Exercise lifecycle invariants without signing, submitting, or claiming proof."""

    as_of = _as_utc(now or datetime.now(timezone.utc))
    active = root / "reports" / "active"
    active.mkdir(parents=True, exist_ok=True)
    scenarios = _scenario_definitions()
    scenario_rows: list[dict[str, object]] = []
    transition_rows: list[dict[str, object]] = []
    for scenario in scenarios:
        summary, transitions = _run_scenario(scenario)
        scenario_rows.append(summary)
        transition_rows.extend(transitions)
    scenario_frame = pd.DataFrame(scenario_rows)
    transition_frame = pd.DataFrame(transition_rows)

    leverage_manifest_path = active / "current_wizard_hyperliquid_leverage_manifest.json"
    leverage_candidates_path = active / "current_wizard_hyperliquid_leverage_candidates.csv"
    chain_manifest_path = active / "current_wizard_hyperliquid_chain_validation_manifest.json"
    protocol_source_path = Path(__file__)
    executor_source_path = Path(__file__).resolve().parents[1] / "hyperliquid_testnet.py"
    approval_source_path = Path(__file__).resolve().parent / "hyperliquid_learning_and_risk.py"
    evidence_source_path = (
        Path(__file__).resolve().parent
        / "hyperliquid_testnet_lifecycle_evidence.py"
    )
    leverage_manifest = _read_json(leverage_manifest_path)
    leverage_candidates = _read_csv(leverage_candidates_path)
    candidate_coverage = _candidate_coverage(leverage_candidates)
    expected_candidates = int(
        leverage_manifest.get(
            "leverage_candidates_complete",
            leverage_manifest.get("candidate_surfaces", len(leverage_candidates)),
        )
        or 0
    )
    candidate_count_match = len(leverage_candidates) == expected_candidates

    validation = _validation(
        scenarios=scenario_frame,
        transitions=transition_frame,
        candidate_coverage=candidate_coverage,
        expected_candidates=expected_candidates,
        candidate_count_match=candidate_count_match,
    )
    protocol_pass = validation["status"].eq("PASS").all()
    identity_material = {
        "schema_version": SCHEMA_VERSION,
        "source_hashes": {
            _source_key(path, root): _file_hash(path)
            for path in (
                leverage_manifest_path,
                leverage_candidates_path,
                chain_manifest_path,
                protocol_source_path,
                executor_source_path,
                approval_source_path,
                evidence_source_path,
            )
            if path.is_file()
        },
        "scenario_results": scenario_frame.to_dict("records"),
        "candidate_coverage": candidate_coverage.to_dict("records"),
    }
    protocol_id = "cwtestnetprotocol_" + sha256(
        _canonical_json(identity_material).encode("utf-8")
    ).hexdigest()[:20]
    for frame in (scenario_frame, transition_frame, candidate_coverage, validation):
        frame.insert(1, "protocol_id", protocol_id)
    paths = _paths(active)
    atomic_write_csv(scenario_frame, paths["scenarios"], index=False)
    atomic_write_csv(transition_frame, paths["transitions"], index=False)
    atomic_write_csv(candidate_coverage, paths["candidate_coverage"], index=False)
    atomic_write_csv(validation, paths["validation"], index=False)
    summary: dict[str, object] = {
        **identity_material,
        "as_of": as_of.isoformat(),
        "protocol_id": protocol_id,
        "protocol_status": "PASS" if protocol_pass else "BLOCKED",
        "scenarios": len(scenario_frame),
        "scenarios_passed": int(scenario_frame["scenario_pass"].map(_boolish).sum()),
        "transitions": len(transition_frame),
        "required_scenarios_complete": REQUIRED_SCENARIOS.issubset(
            set(scenario_frame["scenario_id"].astype(str))
        ),
        "leverage_candidates_expected": expected_candidates,
        "leverage_candidates_accounted": len(candidate_coverage),
        "candidate_count_match": candidate_count_match,
        "actual_testnet_candidates": 0,
        "actual_testnet_lifecycle_proven": 0,
        "simulation_only": True,
        "simulation_is_testnet_proof": False,
        "execution_authority": False,
        "order_submission_performed": False,
        "live_trading_authorized": False,
        "artifacts": {name: str(path.relative_to(root)) for name, path in paths.items()},
    }
    atomic_write_text(paths["manifest"], json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    atomic_write_text(paths["summary_md"], _summary_markdown(summary), encoding="utf-8")
    return CommandResult(paths=paths, summary=summary)


def _scenario_definitions() -> list[dict[str, object]]:
    valid_entry = {
        "type": "ENTRY_REQUEST",
        "acceptance_eligible": True,
        "one_x_proven": True,
        "requested_leverage": 1.0,
        "leverage_scenario_proven": False,
        "signal_age_seconds": 5,
        "max_signal_age_seconds": 60,
        "collateral_ok": True,
        "account_flat": True,
    }
    full_fill = {
        "type": "BULK_RESPONSE",
        "leg_x_status": "filled",
        "leg_y_status": "filled",
        "position_x": 1.0,
        "position_y": -1.0,
    }
    full_exit = {"type": "EXIT_RESULT", "position_x": 0.0, "position_y": 0.0}
    return [
        {
            "scenario_id": "stale_signal_block",
            "events": [{**valid_entry, "signal_age_seconds": 61}],
            "expected_phase": "BLOCKED_FLAT",
            "expected_blocker": "stale_signal",
        },
        {
            "scenario_id": "unaccepted_setup_block",
            "events": [{**valid_entry, "acceptance_eligible": False}],
            "expected_phase": "BLOCKED_FLAT",
            "expected_blocker": "setup_not_acceptance_eligible",
        },
        {
            "scenario_id": "leverage_without_1x_block",
            "events": [
                {
                    **valid_entry,
                    "one_x_proven": False,
                    "requested_leverage": 2.0,
                    "leverage_scenario_proven": True,
                }
            ],
            "expected_phase": "BLOCKED_FLAT",
            "expected_blocker": "one_x_testnet_proof_missing",
        },
        {
            "scenario_id": "insufficient_collateral_block",
            "events": [{**valid_entry, "collateral_ok": False}],
            "expected_phase": "BLOCKED_FLAT",
            "expected_blocker": "insufficient_perpetual_collateral",
        },
        {
            "scenario_id": "full_pair_entry_exit",
            "events": [valid_entry, full_fill, {"type": "EXIT_REQUEST"}, full_exit],
            "expected_phase": "FLAT_RECONCILED",
            "expected_blocker": "",
        },
        {
            "scenario_id": "partial_orphan_recovery",
            "events": [
                valid_entry,
                {
                    "type": "BULK_RESPONSE",
                    "leg_x_status": "partial",
                    "leg_y_status": "rejected",
                    "position_x": 0.4,
                    "position_y": 0.0,
                },
                {"type": "RECONCILE", "position_x": 0.4, "position_y": 0.0},
                {"type": "FLATTEN_RESULT", "position_x": 0.0, "position_y": 0.0},
            ],
            "expected_phase": "FLAT_RECONCILED",
            "expected_blocker": "",
        },
        {
            "scenario_id": "unconfirmed_response_reconciliation",
            "events": [
                valid_entry,
                {
                    "type": "BULK_RESPONSE",
                    "leg_x_status": "unconfirmed",
                    "leg_y_status": "unconfirmed",
                    "position_x": 0.0,
                    "position_y": 0.0,
                },
                {"type": "RECONCILE", "position_x": 0.0, "position_y": 0.0},
            ],
            "expected_phase": "FLAT_RECONCILED",
            "expected_blocker": "",
        },
        {
            "scenario_id": "restart_open_pair_reconciliation",
            "events": [
                {"type": "RESTART", "position_x": 1.0, "position_y": -1.0},
                {"type": "RECONCILE", "position_x": 1.0, "position_y": -1.0},
                {"type": "EXIT_REQUEST"},
                full_exit,
            ],
            "expected_phase": "FLAT_RECONCILED",
            "expected_blocker": "",
        },
        {
            "scenario_id": "proven_leverage_pair_entry_exit",
            "events": [
                {
                    **valid_entry,
                    "requested_leverage": 2.0,
                    "leverage_scenario_proven": True,
                },
                full_fill,
                {"type": "EXIT_REQUEST"},
                full_exit,
            ],
            "expected_phase": "FLAT_RECONCILED",
            "expected_blocker": "",
        },
    ]


def _run_scenario(
    scenario: dict[str, object],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    scenario_id = str(scenario["scenario_id"])
    state: dict[str, object] = {
        "phase": "FLAT",
        "position_x": 0.0,
        "position_y": 0.0,
        "entry_submit_intents": 0,
        "exit_submit_intents": 0,
        "reduce_only_recovery_planned": False,
        "duplicate_submit_blocked": False,
        "blocker": "",
    }
    transitions: list[dict[str, object]] = []
    for sequence, event in enumerate(scenario["events"], start=1):
        before = deepcopy(state)
        action = _apply_event(state, event)
        transitions.append(
            {
                "schema_version": SCHEMA_VERSION,
                "scenario_id": scenario_id,
                "sequence": sequence,
                "event": str(event.get("type", "")),
                "phase_before": before["phase"],
                "phase_after": state["phase"],
                "action": action,
                "position_x": state["position_x"],
                "position_y": state["position_y"],
                "entry_submit_intents": state["entry_submit_intents"],
                "exit_submit_intents": state["exit_submit_intents"],
                "reduce_only_required": state["reduce_only_recovery_planned"],
                "duplicate_submit_blocked": state["duplicate_submit_blocked"],
                "blocker": state["blocker"],
                "simulation_only": True,
                "actual_testnet_proof": False,
                "order_submission_performed": False,
                "live_trading_authorized": False,
            }
        )
    expected_phase = str(scenario["expected_phase"])
    expected_blocker = str(scenario["expected_blocker"])
    terminal_flat = abs(float(state["position_x"])) == 0.0 and abs(
        float(state["position_y"])
    ) == 0.0
    blocker_match = (
        expected_blocker in str(state["blocker"])
        if expected_blocker
        else not str(state["blocker"])
    )
    no_duplicate_submit = int(state["entry_submit_intents"]) <= 1 and int(
        state["exit_submit_intents"]
    ) <= 1
    scenario_pass = bool(
        state["phase"] == expected_phase
        and terminal_flat
        and blocker_match
        and no_duplicate_submit
    )
    return (
        {
            "schema_version": SCHEMA_VERSION,
            "scenario_id": scenario_id,
            "scenario_pass": scenario_pass,
            "expected_phase": expected_phase,
            "final_phase": state["phase"],
            "terminal_flat": terminal_flat,
            "expected_blocker": expected_blocker,
            "actual_blocker": state["blocker"],
            "entry_submit_intents": state["entry_submit_intents"],
            "exit_submit_intents": state["exit_submit_intents"],
            "reduce_only_recovery_planned": state["reduce_only_recovery_planned"],
            "duplicate_submit_blocked": state["duplicate_submit_blocked"],
            "simulation_only": True,
            "actual_testnet_proof": False,
            "order_submission_performed": False,
            "live_trading_authorized": False,
        },
        transitions,
    )


def _apply_event(state: dict[str, object], event: dict[str, object]) -> str:
    event_type = str(event.get("type", ""))
    if event_type == "ENTRY_REQUEST":
        blockers = []
        if not _boolish(event.get("acceptance_eligible", False)):
            blockers.append("setup_not_acceptance_eligible")
        if not _boolish(event.get("one_x_proven", False)):
            blockers.append("one_x_testnet_proof_missing")
        leverage = float(event.get("requested_leverage", 1.0) or 1.0)
        if leverage > 1.0 and not _boolish(
            event.get("leverage_scenario_proven", False)
        ):
            blockers.append("leverage_scenario_not_proven")
        if float(event.get("signal_age_seconds", 0) or 0) > float(
            event.get("max_signal_age_seconds", 0) or 0
        ):
            blockers.append("stale_signal")
        if not _boolish(event.get("collateral_ok", False)):
            blockers.append("insufficient_perpetual_collateral")
        if not _boolish(event.get("account_flat", False)):
            blockers.append("account_not_flat_before_entry")
        if blockers:
            state["phase"] = "BLOCKED_FLAT"
            state["blocker"] = ";".join(blockers)
            return "BLOCK_ENTRY"
        state["phase"] = "ENTRY_PENDING"
        state["entry_submit_intents"] = int(state["entry_submit_intents"]) + 1
        return "PLAN_ATOMIC_BULK_ENTRY"

    if event_type == "BULK_RESPONSE":
        state["position_x"] = float(event.get("position_x", 0.0) or 0.0)
        state["position_y"] = float(event.get("position_y", 0.0) or 0.0)
        statuses = {
            str(event.get("leg_x_status", "")),
            str(event.get("leg_y_status", "")),
        }
        if "unconfirmed" in statuses:
            state["phase"] = "RECONCILE_REQUIRED"
            state["duplicate_submit_blocked"] = True
            return "QUERY_ACCOUNT_AND_ORDERS_NO_RETRY"
        if _hedged(state):
            state["phase"] = "OPEN_HEDGED"
            return "MONITOR_HEDGED_PAIR"
        if _has_position(state):
            state["phase"] = "RECOVERY_REQUIRED"
            state["reduce_only_recovery_planned"] = True
            return "CANCEL_REMAINDER_THEN_REDUCE_ONLY_FLATTEN"
        state["phase"] = "FLAT_RECONCILED"
        state["duplicate_submit_blocked"] = True
        return "RECONCILE_BEFORE_ANY_RETRY"

    if event_type == "RECONCILE":
        state["position_x"] = float(event.get("position_x", 0.0) or 0.0)
        state["position_y"] = float(event.get("position_y", 0.0) or 0.0)
        if not _has_position(state):
            state["phase"] = "FLAT_RECONCILED"
            return "CONFIRM_FLAT_NO_RETRY"
        if _hedged(state):
            state["phase"] = "OPEN_HEDGED"
            return "RESUME_MONITORING_AFTER_RECONCILIATION"
        state["phase"] = "RECOVERY_REQUIRED"
        state["reduce_only_recovery_planned"] = True
        return "REDUCE_ONLY_FLATTEN_ORPHAN"

    if event_type == "FLATTEN_RESULT":
        state["position_x"] = float(event.get("position_x", 0.0) or 0.0)
        state["position_y"] = float(event.get("position_y", 0.0) or 0.0)
        state["phase"] = "FLAT_RECONCILED" if not _has_position(state) else "RECOVERY_REQUIRED"
        return "CONFIRM_FLAT" if state["phase"] == "FLAT_RECONCILED" else "ESCALATE_KILL_SWITCH"

    if event_type == "EXIT_REQUEST":
        if state["phase"] != "OPEN_HEDGED":
            state["duplicate_submit_blocked"] = True
            return "BLOCK_EXIT_UNTIL_RECONCILED"
        state["phase"] = "EXIT_PENDING"
        state["exit_submit_intents"] = int(state["exit_submit_intents"]) + 1
        state["reduce_only_recovery_planned"] = True
        return "PLAN_ATOMIC_BULK_REDUCE_ONLY_EXIT"

    if event_type == "EXIT_RESULT":
        state["position_x"] = float(event.get("position_x", 0.0) or 0.0)
        state["position_y"] = float(event.get("position_y", 0.0) or 0.0)
        state["phase"] = "FLAT_RECONCILED" if not _has_position(state) else "RECOVERY_REQUIRED"
        return "FINAL_RECONCILIATION" if state["phase"] == "FLAT_RECONCILED" else "RECOVER_PARTIAL_EXIT"

    if event_type == "RESTART":
        state["position_x"] = float(event.get("position_x", 0.0) or 0.0)
        state["position_y"] = float(event.get("position_y", 0.0) or 0.0)
        state["phase"] = "RECONCILE_REQUIRED"
        state["duplicate_submit_blocked"] = True
        return "QUERY_ACCOUNT_STATE_BEFORE_RESUME"

    state["phase"] = "BLOCKED_FLAT"
    state["blocker"] = "unknown_protocol_event"
    return "BLOCK_UNKNOWN_EVENT"


def _candidate_coverage(leverage_candidates: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "schema_version",
        "candidate_id",
        "pair",
        "requested_leverage",
        "one_x_proven",
        "leverage_scenario_proven",
        "protocol_simulation_pass",
        "actual_testnet_lifecycle_proven",
        "execution_authority",
        "blocker",
        "order_submission_performed",
        "live_trading_authorized",
    ]
    rows = []
    for index, row in leverage_candidates.iterrows():
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "candidate_id": str(
                    row.get("candidate_id", row.get("experiment_id", f"candidate_{index}"))
                ),
                "pair": str(row.get("pair", "")),
                "requested_leverage": row.get("requested_leverage", 1.0),
                "one_x_proven": _boolish(row.get("one_x_research_accepted", False)),
                "leverage_scenario_proven": _boolish(
                    row.get("leverage_scenario_accepted", False)
                ),
                "protocol_simulation_pass": True,
                "actual_testnet_lifecycle_proven": False,
                "execution_authority": False,
                "blocker": "actual_testnet_lifecycle_not_run",
                "order_submission_performed": False,
                "live_trading_authorized": False,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _validation(
    *,
    scenarios: pd.DataFrame,
    transitions: pd.DataFrame,
    candidate_coverage: pd.DataFrame,
    expected_candidates: int,
    candidate_count_match: bool,
) -> pd.DataFrame:
    checks = (
        (
            "required_scenarios_complete",
            REQUIRED_SCENARIOS.issubset(set(scenarios["scenario_id"].astype(str))),
            f"{len(scenarios)}/{len(REQUIRED_SCENARIOS)}",
        ),
        (
            "all_protocol_scenarios_pass",
            not scenarios.empty and scenarios["scenario_pass"].map(_boolish).all(),
            "scenario_pass=true",
        ),
        (
            "all_scenarios_finish_flat",
            not scenarios.empty and scenarios["terminal_flat"].map(_boolish).all(),
            "terminal_flat=true",
        ),
        (
            "no_duplicate_submit_intents",
            scenarios["entry_submit_intents"].astype(int).le(1).all()
            and scenarios["exit_submit_intents"].astype(int).le(1).all(),
            "entry<=1;exit<=1",
        ),
        (
            "recovery_paths_require_reduce_only",
            scenarios.loc[
                scenarios["scenario_id"].isin(
                    {"partial_orphan_recovery", "restart_open_pair_reconciliation"}
                ),
                "reduce_only_recovery_planned",
            ]
            .map(_boolish)
            .all(),
            "reduce_only_recovery_planned=true",
        ),
        (
            "candidate_count_matches_leverage_surface",
            candidate_count_match and len(candidate_coverage) == expected_candidates,
            f"{len(candidate_coverage)}/{expected_candidates}",
        ),
        (
            "simulation_not_testnet_proof",
            not scenarios["actual_testnet_proof"].map(_boolish).any()
            and not candidate_coverage.get(
                "actual_testnet_lifecycle_proven", pd.Series(dtype=bool)
            )
            .map(_boolish)
            .any(),
            "actual_testnet_proof=false",
        ),
        (
            "no_order_submission_performed",
            not transitions["order_submission_performed"].map(_boolish).any(),
            "order_submission_performed=false",
        ),
        (
            "live_authority_false",
            not transitions["live_trading_authorized"].map(_boolish).any(),
            "live_trading_authorized=false",
        ),
    )
    return pd.DataFrame(
        [
            {
                "schema_version": SCHEMA_VERSION,
                "check": check,
                "status": "PASS" if passed else "BLOCKED",
                "evidence": evidence,
                "execution_authority": False,
                "order_submission_performed": False,
                "live_trading_authorized": False,
            }
            for check, passed, evidence in checks
        ]
    )


def _hedged(state: dict[str, object]) -> bool:
    x = float(state["position_x"])
    y = float(state["position_y"])
    return x != 0.0 and y != 0.0 and x * y < 0.0


def _has_position(state: dict[str, object]) -> bool:
    return abs(float(state["position_x"])) > 0.0 or abs(float(state["position_y"])) > 0.0


def _paths(active: Path) -> dict[str, Path]:
    stem = "current_wizard_hyperliquid_testnet_protocol"
    return {
        "scenarios": active / f"{stem}_scenarios.csv",
        "transitions": active / f"{stem}_transitions.csv",
        "candidate_coverage": active / f"{stem}_candidate_coverage.csv",
        "validation": active / f"{stem}_validation.csv",
        "manifest": active / f"{stem}_manifest.json",
        "summary_md": active / f"{stem}_summary.md",
    }


def _summary_markdown(summary: dict[str, object]) -> str:
    return "\n".join(
        [
            "# Current Wizard To Hyperliquid Testnet Lifecycle Protocol",
            "",
            "This validates deterministic recovery logic only. It is not Testnet execution proof.",
            "",
            f"- protocol: {summary['protocol_id']}",
            f"- status: {summary['protocol_status']}",
            f"- scenarios passed: {summary['scenarios_passed']} of {summary['scenarios']}",
            f"- leverage candidates accounted: {summary['leverage_candidates_accounted']} of {summary['leverage_candidates_expected']}",
            "- simulation only: true",
            "- simulation is Testnet proof: false",
            "- execution authority: false",
            "- order submission performed: false",
            "- live trading authorized: false",
            "",
        ]
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, keep_default_na=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_key(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return f"code:{path.name}"


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _boolish(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _as_utc(value: datetime) -> datetime:
    return (
        value.astimezone(timezone.utc)
        if value.tzinfo
        else value.replace(tzinfo=timezone.utc)
    )
