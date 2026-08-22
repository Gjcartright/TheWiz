from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from quant_platform.orchestration.corrective_runtime import (
    atomic_append_text,
    atomic_write_csv,
    atomic_write_text,
)

BRAIN_SCHEMA_VERSION = "brain_v1"
BRAIN_ARTIFACT_VERSION = "brain_v1"
BRAIN_MEMORY_AGENT = "brain_entity"

PHASE1_SPINE_SCHEMA_VERSION = "phase1_spine_v1"
THREE_BRAIN_CONTRACT_VERSION = PHASE1_SPINE_SCHEMA_VERSION
CANDIDATE_SETUP_PACKET_VERSION = PHASE1_SPINE_SCHEMA_VERSION
PAPER_OUTCOME_PACKET_VERSION = PHASE1_SPINE_SCHEMA_VERSION
FORWARD_WALK_PACKET_VERSION = PHASE1_SPINE_SCHEMA_VERSION
READINESS_PACKET_VERSION = PHASE1_SPINE_SCHEMA_VERSION

WIZARD_LANE = "wizard"
NATIVE_LANE = "native"
OVERALL_LANE = "overall"
CANDIDATE_LANES = (WIZARD_LANE, NATIVE_LANE)
BRAIN_LANES = (WIZARD_LANE, NATIVE_LANE, OVERALL_LANE)
READINESS_AREAS = ("wizard_readiness", "native_readiness", "overall_readiness", "paper_status")

REQUIRED_BRAIN_SUGGESTION_COLUMNS = [
    "cycle_id",
    "pair",
    "strategy_name",
    "variant",
    "entry_logic",
    "exit_logic",
    "entry_threshold",
    "hold_bars_min",
    "hold_bars_max",
    "confidence",
    "expected_return_delta",
    "expected_drawdown_delta",
    "evidence_path",
    "blocker",
    "status",
    "created_at",
    "schema_version",
]

OPTIONAL_PROVIDER_COLUMNS = [
    "provider",
    "provider_source",
    "provider_rows",
    "provider_quality_score",
]

REQUIRED_BRAIN_MEMORY_KEYS = {
    "timestamp",
    "agent",
    "task_id",
    "task_type",
    "cycle_id",
    "pair",
    "outcome_known",
    "outcome_label",
    "blocker",
    "next_step",
}

THREE_BRAIN_CONTRACT = {
    "schema_version": THREE_BRAIN_CONTRACT_VERSION,
    "brains": [
        {
            "lane": WIZARD_LANE,
            "role": "Crypto Wizards and dYdX discovery/validation lane only",
            "allowed_actions": [
                "discover_candidates",
                "normalize_candidates",
                "validate_exact_mode",
                "paper_route_wizard_origin_candidates",
                "run_local_dydx_verification",
            ],
            "forbidden_actions": [
                "promote_from_wizard_only_evidence",
                "override_forward_walk_blockers",
                "override_native_or_overall_truth",
            ],
        },
        {
            "lane": NATIVE_LANE,
            "role": "Local discovery and local math lane only",
            "allowed_actions": [
                "discover_candidates",
                "normalize_candidates",
                "validate_local_math",
                "paper_route_native_origin_candidates",
            ],
            "forbidden_actions": [
                "treat_wizard_discovery_as_native_proof",
                "override_forward_walk_blockers",
                "override_overall_truth",
            ],
        },
        {
            "lane": OVERALL_LANE,
            "role": "Comparison and arbitration lane only in Phase 1",
            "allowed_actions": [
                "compare_lane_outputs",
                "summarize_readiness",
                "report_blockers",
            ],
            "forbidden_actions": [
                "invent_independent_trades",
                "override_blocked_lane_truth",
                "mark_paper_ready_without_forward_walk",
            ],
        },
    ],
}

CANDIDATE_SETUP_COLUMNS = [
    "candidate_id",
    "lane",
    "source_type",
    "source_path",
    "pair",
    "venue",
    "detection_timestamp",
    "timeframe",
    "setup_identity",
    "setup_rank",
    "setup_role",
    "strategy_family",
    "strategy_mode",
    "normalized_feature_bundle_ref",
    "regime_snapshot",
    "confidence",
    "blocker_state",
    "backtest_summary_ref",
    "forward_walk_summary_ref",
    "paper_outcome_ref",
    "provenance",
    "schema_version",
]

PAPER_OUTCOME_COLUMNS = [
    "candidate_id",
    "lane",
    "paper_venue",
    "submission_timestamp",
    "entry",
    "exit",
    "hold_duration",
    "realized_return",
    "drawdown",
    "slippage_cost_assumptions",
    "result_status",
    "verification_status",
    "outcome_evidence_path",
    "provenance",
    "schema_version",
]

FORWARD_WALK_COLUMNS = [
    "candidate_id",
    "lane",
    "rolling_split_definition",
    "oos_sharpe",
    "oos_profit_factor",
    "oos_max_drawdown",
    "oos_trade_count",
    "stability_metrics",
    "forward_walk_status",
    "blocker_reason",
    "dataset_provenance",
    "run_manifest_ref",
    "provenance",
    "schema_version",
]

READINESS_COLUMNS = [
    "area",
    "lane",
    "status",
    "ready",
    "blocker",
    "summary",
    "pair",
    "candidate_id",
    "setup_identity",
    "setup_role",
    "setup_status",
    "setup_blocker",
    "evidence_path",
    "authoritative_source",
    "schema_version",
    "written_at",
]


_TEXT_NULL_VALUES = {"", "nan", "none", "null", "na", "n/a"}


def _clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in _TEXT_NULL_VALUES:
        return ""
    return text


def _coerce_blockers(blockers: list[str] | tuple[str, ...]) -> str:
    compact: list[str] = []
    for value in blockers:
        text = _clean_text(value)
        for token in text.split(";"):
            cleaned = _clean_text(token)
            if cleaned and cleaned not in compact:
                compact.append(cleaned)
    return ";".join(compact)


def brain_output_paths(root: Path, cycle_id: str) -> dict[str, Path]:
    reports = root / "reports" / "brain"
    return {
        "candidate_csv": reports / f"brain_candidates_{cycle_id}.csv",
        "candidate_rollup": reports / "brain_candidate_rollup.csv",
        "cycle_summary": reports / f"brain_cycle_summary_{cycle_id}.json",
        "memory_jsonl": root / "data" / "agent_memory" / f"{BRAIN_MEMORY_AGENT}.jsonl",
        "three_brain_contract": reports / "three_brain_contract.json",
        "candidate_setup_csv": reports / "candidate_setup_packets.csv",
        "candidate_setup_jsonl": reports / "candidate_setup_packets.jsonl",
        "paper_outcome_csv": reports / "paper_outcomes.csv",
        "paper_outcome_jsonl": reports / "paper_outcomes.jsonl",
        "forward_walk_csv": reports / "forward_walk_results.csv",
        "forward_walk_jsonl": reports / "forward_walk_results.jsonl",
        "wizard_readiness": reports / "wizard_readiness.csv",
        "native_readiness": reports / "native_readiness.csv",
        "overall_readiness": reports / "overall_readiness.csv",
        "paper_status": reports / "paper_status.csv",
    }


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_pair(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip().replace("_", "-").replace("/", "-").upper()


def suggest_columns() -> list[str]:
    return REQUIRED_BRAIN_SUGGESTION_COLUMNS.copy()


def valid_brain_suggestion_row(row: dict[str, object]) -> tuple[bool, str]:
    missing = [name for name in REQUIRED_BRAIN_SUGGESTION_COLUMNS if name not in row]
    if missing:
        return False, f"missing_required_columns:{';'.join(missing)}"
    try:
        confidence = float(row.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        return False, "confidence_invalid"
    if confidence < 0 or confidence > 1:
        return False, "confidence_out_of_range"
    if str(row.get("schema_version", "")) != BRAIN_SCHEMA_VERSION:
        return False, "schema_version_mismatch"
    return True, "valid"


def validate_suggestion_frame(frame: pd.DataFrame) -> tuple[bool, str]:
    if frame.empty:
        return False, "empty_frame"
    missing = [name for name in REQUIRED_BRAIN_SUGGESTION_COLUMNS if name not in frame.columns]
    if missing:
        return False, f"missing_columns:{';'.join(missing)}"
    for _, row in frame.iterrows():
        _, reason = valid_brain_suggestion_row(row.to_dict())
        if reason != "valid":
            return False, reason
    return True, "valid"


def write_suggestion_frame(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = frame.copy()
    keep_columns = list(REQUIRED_BRAIN_SUGGESTION_COLUMNS)
    for column in OPTIONAL_PROVIDER_COLUMNS:
        if column in normalized.columns:
            keep_columns.append(column)
    for column in REQUIRED_BRAIN_SUGGESTION_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = ""
    if "created_at" in normalized.columns:
        normalized["created_at"] = normalized["created_at"].fillna(now_utc_iso())
    else:
        normalized["created_at"] = now_utc_iso()
    normalized["schema_version"] = normalized["schema_version"].fillna(BRAIN_SCHEMA_VERSION).replace("", BRAIN_SCHEMA_VERSION)
    normalized["pair"] = normalized["pair"].map(normalize_pair)
    normalized = normalized[keep_columns]
    atomic_write_csv(normalized, path, index=False)
    return path


def write_cycle_summary(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": BRAIN_SCHEMA_VERSION,
        "artifact_version": BRAIN_ARTIFACT_VERSION,
        "written_at": now_utc_iso(),
        **payload,
    }
    atomic_write_text(path, json.dumps(summary, sort_keys=True, default=str), encoding="utf-8")
    return path


def append_brain_memory(path: Path, event: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if "timestamp" not in event:
        event["timestamp"] = now_utc_iso()
    row = dict(event)
    row.setdefault("agent", BRAIN_MEMORY_AGENT)
    atomic_append_text(
        path,
        json.dumps(row, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def valid_memory_event(event: dict[str, object]) -> tuple[bool, str]:
    missing = [key for key in REQUIRED_BRAIN_MEMORY_KEYS if key not in event]
    if missing:
        return False, f"missing_memory_keys:{';'.join(missing)}"
    if str(event.get("agent", "")) != BRAIN_MEMORY_AGENT:
        return False, "agent_mismatch"
    if not bool(event.get("outcome_known")) and bool(event.get("outcome_label")):
        return False, "outcome_label_without_outcome_known"
    return True, "valid"


def valid_candidate_setup_packet(packet: dict[str, object]) -> tuple[bool, str]:
    missing = [name for name in CANDIDATE_SETUP_COLUMNS if name not in packet]
    if missing:
        return False, f"missing_candidate_keys:{';'.join(missing)}"
    if str(packet.get("schema_version", "")) != CANDIDATE_SETUP_PACKET_VERSION:
        return False, "candidate_schema_version_mismatch"
    lane = str(packet.get("lane", "")).strip().lower()
    if lane not in CANDIDATE_LANES:
        return False, "candidate_lane_invalid"
    if not str(packet.get("candidate_id", "")).strip():
        return False, "candidate_id_missing"
    if not str(packet.get("pair", "")).strip():
        return False, "candidate_pair_missing"
    confidence_ok, reason = _validate_unit_interval(packet.get("confidence", 0.0), "candidate_confidence")
    if not confidence_ok:
        return False, reason
    return True, "valid"


def validate_candidate_setup_frame(frame: pd.DataFrame) -> tuple[bool, str]:
    return _validate_frame(frame, CANDIDATE_SETUP_COLUMNS, valid_candidate_setup_packet)


def write_candidate_setup_frame(frame: pd.DataFrame, path: Path) -> Path:
    normalized = frame.copy()
    for column in CANDIDATE_SETUP_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = ""
    normalized["pair"] = normalized["pair"].map(normalize_pair)
    normalized["lane"] = normalized["lane"].astype(str).str.strip().str.lower()
    normalized["schema_version"] = (
        normalized["schema_version"].fillna(CANDIDATE_SETUP_PACKET_VERSION).replace("", CANDIDATE_SETUP_PACKET_VERSION)
    )
    normalized = normalized[CANDIDATE_SETUP_COLUMNS]
    return _write_frame(normalized, path)


def append_candidate_setup_packets(path: Path, packets: list[dict[str, object]]) -> Path:
    return _append_jsonl_rows(path, packets, valid_candidate_setup_packet)


def valid_paper_outcome_packet(packet: dict[str, object]) -> tuple[bool, str]:
    missing = [name for name in PAPER_OUTCOME_COLUMNS if name not in packet]
    if missing:
        return False, f"missing_paper_outcome_keys:{';'.join(missing)}"
    if str(packet.get("schema_version", "")) != PAPER_OUTCOME_PACKET_VERSION:
        return False, "paper_outcome_schema_version_mismatch"
    if not str(packet.get("candidate_id", "")).strip():
        return False, "paper_outcome_candidate_id_missing"
    lane = str(packet.get("lane", "")).strip().lower()
    if lane not in CANDIDATE_LANES:
        return False, "paper_outcome_lane_invalid"
    return True, "valid"


def append_paper_outcome_packets(path: Path, packets: list[dict[str, object]]) -> Path:
    return _append_jsonl_rows(path, packets, valid_paper_outcome_packet)


def write_paper_outcome_frame(frame: pd.DataFrame, path: Path) -> Path:
    normalized = frame.copy()
    for column in PAPER_OUTCOME_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = ""
    normalized["lane"] = normalized["lane"].astype(str).str.strip().str.lower()
    normalized["schema_version"] = (
        normalized["schema_version"].fillna(PAPER_OUTCOME_PACKET_VERSION).replace("", PAPER_OUTCOME_PACKET_VERSION)
    )
    normalized = normalized[PAPER_OUTCOME_COLUMNS]
    return _write_frame(normalized, path)


def valid_forward_walk_packet(packet: dict[str, object]) -> tuple[bool, str]:
    missing = [name for name in FORWARD_WALK_COLUMNS if name not in packet]
    if missing:
        return False, f"missing_forward_walk_keys:{';'.join(missing)}"
    if str(packet.get("schema_version", "")) != FORWARD_WALK_PACKET_VERSION:
        return False, "forward_walk_schema_version_mismatch"
    if not str(packet.get("candidate_id", "")).strip():
        return False, "forward_walk_candidate_id_missing"
    lane = str(packet.get("lane", "")).strip().lower()
    if lane not in CANDIDATE_LANES:
        return False, "forward_walk_lane_invalid"
    if str(packet.get("forward_walk_status", "")).strip().lower() not in {"pass", "blocked", "missing"}:
        return False, "forward_walk_status_invalid"
    return True, "valid"


def append_forward_walk_packets(path: Path, packets: list[dict[str, object]]) -> Path:
    return _append_jsonl_rows(path, packets, valid_forward_walk_packet)


def write_forward_walk_frame(frame: pd.DataFrame, path: Path) -> Path:
    normalized = frame.copy()
    for column in FORWARD_WALK_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = ""
    normalized["lane"] = normalized["lane"].astype(str).str.strip().str.lower()
    normalized["schema_version"] = (
        normalized["schema_version"].fillna(FORWARD_WALK_PACKET_VERSION).replace("", FORWARD_WALK_PACKET_VERSION)
    )
    normalized = normalized[FORWARD_WALK_COLUMNS]
    return _write_frame(normalized, path)


def candidate_paper_credibility_status(
    candidate_packet: dict[str, object],
    *,
    forward_walk_packet: dict[str, object] | None = None,
    local_verification_passed: bool = False,
) -> tuple[bool, str]:
    ok, reason = valid_candidate_setup_packet(candidate_packet)
    if not ok:
        return False, reason
    lane = str(candidate_packet.get("lane", "")).strip().lower()
    if not str(candidate_packet.get("forward_walk_summary_ref", "")).strip():
        return False, "forward_walk_link_missing"
    if forward_walk_packet is None:
        return False, "forward_walk_packet_missing"
    ok, reason = valid_forward_walk_packet(forward_walk_packet)
    if not ok:
        return False, reason
    if str(forward_walk_packet.get("candidate_id", "")).strip() != str(candidate_packet.get("candidate_id", "")).strip():
        return False, "forward_walk_candidate_mismatch"
    if str(forward_walk_packet.get("forward_walk_status", "")).strip().lower() != "pass":
        return False, str(forward_walk_packet.get("blocker_reason", "")).strip() or "forward_walk_blocked"
    blocker_value = candidate_packet.get("blocker_state", "")
    blocker_state = "" if pd.isna(blocker_value) else str(blocker_value).strip()
    if blocker_state:
        return False, blocker_state
    if lane == WIZARD_LANE and not local_verification_passed:
        return False, "wizard_local_verification_missing"
    return True, "paper_credible"


def build_overall_candidate_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["lane", "candidate_count", "blocked_count", "mean_confidence"])
    working = frame.copy()
    if "confidence" in working.columns:
        working["confidence"] = pd.to_numeric(working["confidence"], errors="coerce").fillna(0.0)
    else:
        working["confidence"] = 0.0
    if "blocker_state" not in working.columns:
        working["blocker_state"] = ""
    grouped = (
        working.groupby("lane", dropna=False)
        .agg(
            candidate_count=("candidate_id", "count"),
            blocked_count=("blocker_state", lambda s: int(s.astype(str).str.strip().ne("").sum())),
            mean_confidence=("confidence", "mean"),
        )
        .reset_index()
    )
    return grouped


def write_three_brain_contract(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(THREE_BRAIN_CONTRACT, sort_keys=True, indent=2), encoding="utf-8")
    return path


def build_phase1_readiness_surfaces(root: Path) -> dict[str, Path]:
    paths = brain_output_paths(root, "current")
    write_three_brain_contract(paths["three_brain_contract"])

    wizard_source = root / "reports" / "brain" / "brain_readiness_report.csv"
    native_source = paths["candidate_setup_csv"]
    paper_source = root / "reports" / "rl" / "base_rl_paper_handoff_status.csv"

    wizard_row = _wizard_readiness_row(root, wizard_source)
    native_row = _native_readiness_row(root, native_source)
    paper_row = _paper_status_row(root, paper_source)
    overall_row = _overall_readiness_row(root, wizard_row, native_row, paper_row)

    _write_frame(pd.DataFrame([wizard_row], columns=READINESS_COLUMNS), paths["wizard_readiness"])
    _write_frame(pd.DataFrame([native_row], columns=READINESS_COLUMNS), paths["native_readiness"])
    _write_frame(pd.DataFrame([overall_row], columns=READINESS_COLUMNS), paths["overall_readiness"])
    _write_frame(pd.DataFrame([paper_row], columns=READINESS_COLUMNS), paths["paper_status"])
    return {
        "three_brain_contract": paths["three_brain_contract"],
        "wizard_readiness": paths["wizard_readiness"],
        "native_readiness": paths["native_readiness"],
        "overall_readiness": paths["overall_readiness"],
        "paper_status": paths["paper_status"],
    }


def _validate_frame(
    frame: pd.DataFrame,
    required_columns: list[str],
    validator: Any,
) -> tuple[bool, str]:
    if frame.empty:
        return False, "empty_frame"
    missing = [name for name in required_columns if name not in frame.columns]
    if missing:
        return False, f"missing_columns:{';'.join(missing)}"
    for _, row in frame.iterrows():
        ok, reason = validator(row.to_dict())
        if not ok:
            return False, reason
    return True, "valid"


def _validate_unit_interval(value: object, label: str) -> tuple[bool, str]:
    try:
        numeric = float(value or 0.0)
    except (TypeError, ValueError):
        return False, f"{label}_invalid"
    if numeric < 0.0 or numeric > 1.0:
        return False, f"{label}_out_of_range"
    return True, "valid"


def _write_frame(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(frame, path, index=False)
    return path


def _append_jsonl_rows(path: Path, packets: list[dict[str, object]], validator: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded: list[str] = []
    for packet in packets:
        ok, reason = validator(packet)
        if not ok:
            raise ValueError(reason)
        encoded.append(json.dumps(packet, sort_keys=True, default=str) + "\n")
    if encoded:
        atomic_append_text(path, "".join(encoded), encoding="utf-8")
    return path


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _readiness_row(
    *,
    area: str,
    lane: str,
    status: str,
    blocker: str,
    summary: str,
    evidence_path: Path,
    authoritative_source: Path,
    pair: str = "",
    candidate_id: str = "",
    setup_identity: str = "",
    setup_role: str = "",
    setup_status: str = "",
    setup_blocker: str = "",
) -> dict[str, object]:
    ready = str(status).strip().lower() in {"ready", "pass", "paper_ready", "authorized", "paper_authorized"}
    return {
        "area": area,
        "lane": lane,
        "status": status,
        "ready": ready,
        "blocker": blocker,
        "summary": summary,
        "pair": pair,
        "candidate_id": candidate_id,
        "setup_identity": setup_identity,
        "setup_role": setup_role,
        "setup_status": setup_status,
        "setup_blocker": setup_blocker,
        "evidence_path": str(evidence_path),
        "authoritative_source": str(authoritative_source),
        "schema_version": READINESS_PACKET_VERSION,
        "written_at": now_utc_iso(),
    }


def _wizard_readiness_row(root: Path, source: Path) -> dict[str, object]:
    frame = _read_csv(source)
    fallback_status, fallback_blocker, fallback_summary, fallback_evidence = _wizard_capture_truth_status(root)
    if frame.empty:
        status = fallback_status
        blocker = fallback_blocker
        summary = fallback_summary
        authoritative_source = fallback_evidence
    else:
        row = frame.iloc[0]
        gate = str(row.get("score_gate", row.get("readiness_gate", "hold"))).strip().lower()
        status = "ready" if gate == "pass" else "blocked"
        report_blocker = "" if status == "ready" else str(row.get("readiness_warnings", "")).strip() or "wizard_readiness_not_ready"
        blocker = report_blocker
        summary = f"wizard_score={float(row.get('readiness_score', 0.0) or 0.0):.4f};gate={gate}"
        authoritative_source = source
        if fallback_blocker:
            blocker = fallback_blocker if not blocker else f"{blocker};{fallback_blocker}"
            status = "blocked"
            summary = summary + f";{fallback_summary}"
    detail_setup = _preferred_wizard_detail_setup_row(root)
    setup = detail_setup if detail_setup is not None else _preferred_setup_row(root, WIZARD_LANE)
    setup_blocker = _packet_text(setup, "blocker_state")
    setup_status = setup_blocker or ("ready" if status == "ready" else "blocked")

    investigation = _read_csv(root / "reports" / "brain" / "wizard_dashboard_investigation.csv")
    if not investigation.empty:
        active_blockers = investigation[investigation.get("status", pd.Series(dtype=object)).astype(str).isin(["blocked", "signin_redirect"])].copy()
        if not active_blockers.empty:
            areas = active_blockers.get("investigation_area", pd.Series(dtype=object)).astype(str).tolist()
            investigation_blocker = "wizard_investigation:" + ";".join(areas)
            blocker = investigation_blocker if not blocker else f"{blocker};{investigation_blocker}"
            status = "blocked"
            summary = summary + f";investigation_blockers={len(active_blockers)}"
    if setup is not None and setup_blocker:
        blocker = setup_blocker if not blocker else f"{blocker};{setup_blocker}"
        status = "blocked"
        summary = summary + ";setup_blocked=1"
    return _readiness_row(
        area="wizard_readiness",
        lane=WIZARD_LANE,
        status=status,
        blocker=blocker,
        summary=summary,
        pair=_packet_text(setup, "pair"),
        candidate_id=_packet_text(setup, "candidate_id"),
        setup_identity=_packet_text(setup, "setup_identity"),
        setup_role=_packet_text(setup, "setup_role"),
        setup_status=setup_status,
        setup_blocker=setup_blocker or blocker,
        evidence_path=authoritative_source,
        authoritative_source=authoritative_source,
    )


def _wizard_capture_truth_status(root: Path) -> tuple[str, str, str, Path]:
    detail_path = root / "reports" / "active" / "wizard_research_pair_detail_capture.csv"
    live_path = root / "reports" / "active" / "crypto_wizards_live_scanner_capture.csv"
    detail = _read_csv(detail_path)
    if detail.empty:
        return (
            "blocked",
            "wizard_pair_detail_capture_missing",
            "wizard_pair_detail_capture_missing=1",
            detail_path,
        )

    live = _read_csv(live_path)
    current_pairs: list[str] = []
    if not live.empty and "pair" in live.columns:
        current_pairs = [normalize_pair(value) for value in live.get("pair", pd.Series(dtype=object)).astype(str).tolist() if str(value).strip()]
    if not current_pairs and "pair" in detail.columns:
        current_pairs = [
            normalize_pair(value)
            for value in detail.get("pair", pd.Series(dtype=object)).astype(str).tolist()
            if str(value).strip()
        ]
    current_pairs = sorted({pair for pair in current_pairs if pair})
    if not current_pairs:
        return (
            "blocked",
            "wizard_current_pairs_missing",
            "wizard_current_pairs_missing=1",
            detail_path,
        )

    working = detail.copy()
    working["pair"] = working.get("pair", pd.Series(dtype=object)).map(normalize_pair)
    working["timeframe"] = working.get("timeframe", pd.Series(dtype=object)).astype(str).str.strip()
    working["capture_status"] = working.get("capture_status", pd.Series(dtype=object)).astype(str).str.strip()
    working = working[
        working["pair"].isin(current_pairs)
        & working["capture_status"].eq("captured")
        & working["timeframe"].isin(["Daily", "4 Hour", "1 Hour", "5 Min"])
    ].copy()

    missing_pairs: list[str] = []
    missing_timeframes_total = 0
    for pair in current_pairs:
        pair_timeframes = set(working.loc[working["pair"] == pair, "timeframe"].astype(str).tolist())
        missing = [label for label in ("Daily", "4 Hour", "1 Hour", "5 Min") if label not in pair_timeframes]
        if missing:
            missing_pairs.append(f"{pair}:{','.join(missing)}")
            missing_timeframes_total += len(missing)

    summary = (
        f"current_pairs={len(current_pairs)};"
        f"captured_rows={len(working)};"
        f"missing_pairs={len(missing_pairs)};"
        f"missing_timeframes={missing_timeframes_total}"
    )
    if missing_pairs:
        blocker = "wizard_timeframe_capture_incomplete"
        if len(missing_pairs) <= 3:
            blocker = blocker + ":" + ";".join(missing_pairs)
        return ("blocked", blocker, summary, detail_path)
    return ("ready", "", summary, detail_path)


def _native_readiness_row(root: Path, source: Path) -> dict[str, object]:
    frame = _read_csv(source)
    if frame.empty:
        return _readiness_row(
            area="native_readiness",
            lane=NATIVE_LANE,
            status="contract_only",
            blocker="native_candidate_packets_missing",
            summary="Native lane is aligned to the Phase 1 contract but has no current candidate packet rollup yet.",
            evidence_path=source,
            authoritative_source=source,
        )
    native_only = frame[frame.get("lane", pd.Series(dtype=object)).astype(str) == NATIVE_LANE].copy()
    packet_ok, reason = validate_candidate_setup_frame(native_only if not native_only.empty else frame)
    status = "ready" if packet_ok else "contract_only"
    blocker = "" if packet_ok else reason
    preferred = _preferred_packet_row(native_only if not native_only.empty else frame)
    proof = _packet_text(preferred, "provenance")
    if "wizard" in proof.lower() and proof:
        status = "contract_only"
        blocker = blocker or "native_proof_must_not_depend_on_wizard_origin"
    native_quality = _read_csv(root / "reports" / "brain" / "native_quality_report.csv")
    native_promotion = _read_csv(root / "reports" / "brain" / "promotion_ladder.csv")
    quality_rows = native_quality[native_quality.get("candidate_id", pd.Series(dtype=object)).astype(str).isin(
        native_only.get("candidate_id", pd.Series(dtype=object)).astype(str)
    )] if not native_quality.empty else pd.DataFrame()
    promotion_rows = native_promotion[native_promotion.get("lane", pd.Series(dtype=object)).astype(str) == NATIVE_LANE] if not native_promotion.empty else pd.DataFrame()
    if packet_ok:
        supported = quality_rows.get("quality_status", pd.Series(dtype=object)).astype(str).eq("supported").any() if not quality_rows.empty else False
        paper_credible = promotion_rows.get("paper_credible", pd.Series(dtype=bool)).astype(bool).any() if not promotion_rows.empty else False
        if not supported:
            status = "candidate_quality_blocked"
            blocker = blocker or "native_candidate_quality_not_supported"
        elif not paper_credible:
            status = "candidate_quality_blocked"
            blocker = blocker or "native_forward_walk_or_promotion_blocked"
    summary = f"candidate_packets={len(native_only if not native_only.empty else frame)};schema={CANDIDATE_SETUP_PACKET_VERSION}"
    return _readiness_row(
        area="native_readiness",
        lane=NATIVE_LANE,
        status=status,
        blocker=blocker,
        summary=summary,
        pair=_packet_text(preferred, "pair"),
        candidate_id=_packet_text(preferred, "candidate_id"),
        setup_identity=_packet_text(preferred, "setup_identity"),
        setup_role=_packet_text(preferred, "setup_role"),
        setup_status=_packet_text(preferred, "blocker_state") or ("ready" if status == "ready" else "contract_only"),
        setup_blocker=_packet_text(preferred, "blocker_state") or blocker,
        evidence_path=source,
        authoritative_source=source,
    )


def _paper_status_row(root: Path, source: Path) -> dict[str, object]:
    frame = _read_csv(source)
    if frame.empty:
        return _readiness_row(
            area="paper_status",
            lane=OVERALL_LANE,
            status="blocked",
            blocker="paper_handoff_missing",
            summary="Paper status alias blocked because the current paper handoff file is missing.",
            evidence_path=source,
            authoritative_source=source,
        )
    row = frame.iloc[0]
    status = _clean_text(row.get("status")) or "blocked"
    blocker = _clean_text(row.get("blocker", ""))
    if status == "research_only" and blocker in {"", "strategy_acceptance_not_ready"}:
        blocker = _clean_text(_paper_route_blocker(root)) or blocker
    summary = (
        f"paper_authorized={bool(row.get('paper_authorized', False))};"
        f"status={status};"
        f"execution_truth_mode={str(row.get('execution_truth_mode', 'paper_only'))};"
        f"injective_mirrorable_pairs={int(row.get('injective_mirrorable_pairs', 0) or 0)}"
    )
    promotion = _read_csv(root / "reports" / "brain" / "promotion_ladder.csv")
    preferred = _preferred_promotion_row(promotion)
    return _readiness_row(
        area="paper_status",
        lane=OVERALL_LANE,
        status=status,
        blocker=blocker,
        summary=summary,
        pair=_packet_text(preferred, "pair"),
        candidate_id=_packet_text(preferred, "candidate_id"),
        setup_identity=_packet_text(preferred, "setup_identity"),
        setup_role=_packet_text(preferred, "setup_role"),
        setup_status=_packet_text(preferred, "promotion_stage") or status,
        setup_blocker=_packet_text(preferred, "blocker") or blocker,
        evidence_path=source,
        authoritative_source=source,
    )


def _overall_readiness_row(
    root: Path,
    wizard_row: dict[str, object],
    native_row: dict[str, object],
    paper_row: dict[str, object],
) -> dict[str, object]:
    blockers = [
        _clean_text(wizard_row.get("blocker", "")),
        _clean_text(native_row.get("blocker", "")),
        _clean_text(paper_row.get("blocker", "")),
    ]
    clean_blockers = _coerce_blockers(blockers)
    ready = bool(wizard_row.get("ready")) and bool(native_row.get("ready")) and bool(paper_row.get("ready"))
    status = "ready" if ready else "blocked"
    summary = (
        f"wizard={wizard_row.get('status', '')};"
        f"native={native_row.get('status', '')};"
        f"paper={paper_row.get('status', '')}"
    )
    overall = _read_csv(root / "reports" / "brain" / "overall_brain_summary.csv")
    preferred = _preferred_overall_row(overall)
    pair = _packet_text(paper_row, "pair") or _packet_text(native_row, "pair") or _packet_text(wizard_row, "pair") or _packet_text(preferred, "pair")
    candidate_id = (
        _packet_text(paper_row, "candidate_id")
        or _packet_text(native_row, "candidate_id")
        or _packet_text(wizard_row, "candidate_id")
        or _packet_text(preferred, "wizard_candidate_id")
        or _packet_text(preferred, "native_candidate_id")
    )
    setup_identity = (
        _packet_text(paper_row, "setup_identity")
        or _packet_text(native_row, "setup_identity")
        or _packet_text(wizard_row, "setup_identity")
        or _packet_text(preferred, "wizard_setup_identity")
        or _packet_text(preferred, "native_setup_identity")
    )
    setup_role = _packet_text(paper_row, "setup_role") or _packet_text(native_row, "setup_role") or _packet_text(wizard_row, "setup_role")
    setup_status = _packet_text(paper_row, "setup_status") or _packet_text(preferred, "agreement_status") or status
    return _readiness_row(
        area="overall_readiness",
        lane=OVERALL_LANE,
        status=status,
        blocker=clean_blockers,
        summary=summary,
        pair=pair,
        candidate_id=candidate_id,
        setup_identity=setup_identity,
        setup_role=setup_role,
        setup_status=setup_status,
        setup_blocker=clean_blockers,
        evidence_path=brain_output_paths(root, "current")["overall_readiness"],
        authoritative_source=brain_output_paths(root, "current")["overall_readiness"],
    )


def _preferred_setup_row(root: Path, lane: str) -> pd.Series | None:
    packets = _read_csv(root / "reports" / "brain" / f"{lane}_candidate_packets.csv")
    if packets.empty:
        return None
    if lane == WIZARD_LANE:
        return _preferred_wizard_packet_row(packets)
    return _preferred_packet_row(packets)


def _preferred_wizard_detail_setup_row(root: Path) -> pd.Series | None:
    detail_path = root / "reports" / "active" / "wizard_research_pair_detail_capture.csv"
    detail = _read_csv(detail_path)
    if detail.empty:
        return None

    live = _read_csv(root / "reports" / "active" / "crypto_wizards_live_scanner_capture.csv")
    current_pairs: set[str] = set()
    if not live.empty and "pair" in live.columns:
        current_pairs = {
            normalize_pair(value)
            for value in live.get("pair", pd.Series(dtype=object)).astype(str).tolist()
            if str(value).strip()
        }

    working = detail.copy()
    working["_pair"] = working.get("pair", pd.Series(dtype=object)).map(normalize_pair)
    if current_pairs:
        working = working[working["_pair"].isin(current_pairs)].copy()
    working["_timeframe"] = working.get("timeframe", pd.Series(dtype=object)).astype(str).str.strip()
    working["_capture_status"] = working.get("capture_status", pd.Series(dtype=object)).astype(str).str.strip().str.lower()
    working = working[
        working["_capture_status"].eq("captured")
        & working["_timeframe"].isin(["Daily", "4 Hour", "1 Hour", "5 Min"])
    ].copy()
    if working.empty:
        return None

    sharpe_source = working.get("sharpe_top", working.get("sharpe_detail", pd.Series(dtype=object)))
    working["_sharpe"] = pd.to_numeric(sharpe_source, errors="coerce")
    drawdown_source = working.get("max_drawdown_top", working.get("max_drawdown_detail", pd.Series(dtype=object)))
    working["_drawdown"] = pd.to_numeric(drawdown_source, errors="coerce")
    working["_quality_pass"] = working["_sharpe"].ge(1.75) & (working["_drawdown"].isna() | working["_drawdown"].le(25.0))
    working = working[working["_quality_pass"]].copy()
    if working.empty:
        return None

    timeframe_rank = {"Daily": 0, "4 Hour": 1, "1 Hour": 2, "5 Min": 3}
    working["_timeframe_rank"] = working["_timeframe"].map(timeframe_rank).fillna(9)
    working = working.sort_values(
        ["_timeframe_rank", "_sharpe", "_drawdown", "_pair"],
        ascending=[True, False, True, True],
    )
    row = working.iloc[0]
    pair = _packet_text(row, "_pair")
    timeframe = _packet_text(row, "_timeframe")
    strategy = _packet_text(row, "strategy_label") or _packet_text(row, "strategy_variant_from_row")
    strategy_key = strategy.lower().replace(" ", "_").replace("(", "").replace(")", "")
    timeframe_key = timeframe.lower().replace(" ", "")
    asset_x = _packet_text(row, "asset_x")
    asset_y = _packet_text(row, "asset_y")
    if not asset_x or not asset_y:
        parts = pair.split("-USD-")
        if len(parts) == 2:
            asset_x = f"{parts[0]}-USD"
            asset_y = f"{parts[1]}-USD"
    return pd.Series(
        {
            "candidate_id": f"wizard:{pair}:{strategy_key}:{timeframe_key}",
            "lane": WIZARD_LANE,
            "source_type": "wizard_pair_detail_capture",
            "source_path": str(detail_path),
            "pair": pair,
            "venue": _packet_text(row, "venue") or "dydx",
            "detection_timestamp": _packet_text(row, "detail_capture_timestamp_utc")
            or _packet_text(row, "capture_timestamp_utc"),
            "timeframe": timeframe,
            "setup_identity": f"{asset_x}|{asset_y}|{timeframe_key}|{_packet_text(row, 'periods_input')}|{strategy_key}",
            "setup_rank": 1,
            "setup_role": "primary",
            "strategy_family": _wizard_strategy_family_from_label(strategy),
            "strategy_mode": strategy,
            "confidence": 1.0,
            "blocker_state": "",
            "backtest_summary_ref": str(detail_path),
            "forward_walk_summary_ref": "",
            "paper_outcome_ref": _packet_text(row, "paper_trade_id"),
            "provenance": "wizard_pair_detail_capture",
            "schema_version": CANDIDATE_SETUP_PACKET_VERSION,
        }
    )


def _wizard_strategy_family_from_label(label: str) -> str:
    lowered = str(label or "").strip().lower()
    if "copula" in lowered:
        return "copula"
    if "dyn" in lowered or "dynamic" in lowered:
        return "adaptive"
    if "ou" in lowered:
        return "mean_reversion"
    return "zscore"


def _preferred_packet_row(frame: pd.DataFrame) -> pd.Series | None:
    if frame.empty:
        return None
    working = frame.copy()
    if "setup_role" in working.columns:
        working["_primary"] = working["setup_role"].astype(str).eq("primary")
    else:
        working["_primary"] = False
    if "setup_rank" in working.columns:
        working["_rank"] = pd.to_numeric(working["setup_rank"], errors="coerce").fillna(999999)
    else:
        working["_rank"] = 999999
    if "confidence" in working.columns:
        working["_confidence"] = pd.to_numeric(working["confidence"], errors="coerce").fillna(0.0)
    else:
        working["_confidence"] = 0.0
    working = working.sort_values(["_primary", "_rank", "_confidence"], ascending=[False, True, False])
    return working.iloc[0]


def _preferred_wizard_packet_row(frame: pd.DataFrame) -> pd.Series | None:
    if frame.empty:
        return None
    working = frame.copy()
    if "setup_role" in working.columns:
        working["_primary"] = working["setup_role"].astype(str).eq("primary")
    else:
        working["_primary"] = False
    if "setup_rank" in working.columns:
        working["_rank"] = pd.to_numeric(working["setup_rank"], errors="coerce").fillna(999999)
    else:
        working["_rank"] = 999999
    if "confidence" in working.columns:
        working["_confidence"] = pd.to_numeric(working["confidence"], errors="coerce").fillna(0.0)
    else:
        working["_confidence"] = 0.0
    working["_exact_mode_rank"] = working.apply(_wizard_exact_mode_rank, axis=1)
    working["_blocker_rank"] = working.apply(_wizard_blocker_rank, axis=1)
    working = working.sort_values(
        ["_primary", "_exact_mode_rank", "_blocker_rank", "_rank", "_confidence"],
        ascending=[False, True, True, True, False],
    )
    return working.iloc[0]


def _wizard_exact_mode_rank(row: pd.Series) -> int:
    blocker = _packet_text(row, "blocker_state")
    strategy_mode = _packet_text(row, "strategy_mode")
    strategy_family = _packet_text(row, "strategy_family")
    setup_identity = _packet_text(row, "setup_identity")
    if "missing_exact_mode" in blocker:
        return 2
    if strategy_mode or strategy_family:
        return 0
    if "||" in setup_identity or "|nan" in setup_identity.lower():
        return 1
    return 0


def _wizard_blocker_rank(row: pd.Series) -> int:
    blocker = _packet_text(row, "blocker_state")
    if not blocker:
        return 0
    if "missing_exact_mode" in blocker:
        return 4
    if "missing_correlation" in blocker or "missing_ecm" in blocker or "missing_copula" in blocker:
        return 3
    if "wizard_local_verification_missing" in blocker or "local_verification_not_run" in blocker:
        return 2
    if "parity_" in blocker:
        return 1
    return 0


def _preferred_promotion_row(frame: pd.DataFrame) -> pd.Series | None:
    if frame.empty:
        return None
    working = frame.copy()
    working["_credible"] = working.get("paper_credible", pd.Series(dtype=object)).astype(bool)
    working["_stage_rank"] = working.get("promotion_stage", pd.Series(dtype=object)).astype(str).map(
        {
            "paper_eligible": 0,
            "orchestrator_review": 1,
            "candidate_validated": 2,
            "rl_interpretation": 3,
            "forward_walk_blocked": 4,
        }
    ).fillna(9)
    if "setup_role" in working.columns:
        working["_primary"] = working["setup_role"].astype(str).eq("primary")
    else:
        working["_primary"] = False
    working["_route_rank"] = pd.to_numeric(working.get("route_priority", pd.Series(dtype=object)), errors="coerce").fillna(9)
    working["_support_rank"] = working.get("model_support_status", pd.Series(dtype=object)).astype(str).map(
        {
            "strong_model_support": 0,
            "weak_model_support": 1,
            "pair_missing_from_model_predictions": 2,
            "no_model_support": 3,
            "model_predictions_missing": 4,
        }
    ).fillna(9)
    working["_support_trades"] = pd.to_numeric(working.get("model_taken_trades", pd.Series(dtype=object)), errors="coerce").fillna(0)
    working = working.sort_values(
        ["_credible", "_stage_rank", "_route_rank", "_support_rank", "_support_trades", "_primary"],
        ascending=[False, True, True, True, False, False],
    )
    return working.iloc[0]


def _preferred_overall_row(frame: pd.DataFrame) -> pd.Series | None:
    if frame.empty:
        return None
    working = frame.copy()
    working["_decision_rank"] = working.get("arbitration_decision", pd.Series(dtype=object)).astype(str).map(
        {"agreement_strongest": 0, "prefer_wizard": 1, "prefer_native": 2, "wizard_reject_only": 3, "native_reject_only": 4, "ignore_both": 5}
    ).fillna(9)
    working = working.sort_values(["_decision_rank"])
    return working.iloc[0]


def _paper_route_blocker(root: Path) -> str:
    readiness = _read_csv(root / "reports" / "priority_readiness.csv")
    if not readiness.empty and "gate" in readiness.columns:
        for gate in ("strategy_acceptance", "paper_execution_gate", "dydx_testnet_readiness"):
            matches = readiness[readiness["gate"].astype(str) == gate]
            if matches.empty:
                continue
            candidate = matches.iloc[0]
            blocker = _packet_text(candidate, "blocker")
            if blocker:
                return blocker
            if not _packet_bool(candidate, "ready"):
                return f"{gate}_not_ready"

    handoff = _read_csv(root / "reports" / "rl" / "base_rl_paper_handoff_status.csv")
    if not handoff.empty:
        row = handoff.iloc[0]
        blocker = _packet_text(row, "blocker")
        if blocker:
            return blocker

        blockers: list[str] = []
        paper_authorized = _packet_bool(row, "paper_authorized")
        if not paper_authorized:
            if "strategy_acceptance_ready" in row.index and not _packet_bool(row, "strategy_acceptance_ready"):
                blockers.append("strategy_acceptance_not_ready")
            if "paper_execution_ready" in row.index and not _packet_bool(row, "paper_execution_ready"):
                blockers.append("paper_execution_not_ready")
            if "model_gate_accepted" in row.index and not _packet_bool(row, "model_gate_accepted"):
                model_blocker = _packet_text(row, "route_model_gate_blocker")
                blockers.append(model_blocker or "model_gate_not_accepted")
        if blockers:
            return _coerce_blockers(blockers)
    return "strategy_acceptance_not_ready"


def _packet_bool(row: pd.Series | None, key: str) -> bool:
    if row is None:
        return False
    value = row.get(key, False)
    if pd.isna(value):
        return False
    value_text = str(value).strip().lower()
    return value_text in {"1", "true", "yes", "on"}


def _packet_text(row: pd.Series | None, key: str) -> str:
    if row is None:
        return ""
    value = row.get(key, "")
    return _clean_text(value)
