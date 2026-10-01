from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from numbers import Integral
from pathlib import Path
from typing import Any

import pandas as pd

from quant_platform.orchestration.corrective_runtime import (
    atomic_append_text,
    atomic_write_csv,
)


@dataclass(frozen=True)
class TradeRecord:
    trade_id: str
    timestamp: datetime
    pair: str
    strategy: str
    regime: str
    features: dict[str, float]
    signal: dict[str, Any]
    execution: dict[str, Any]
    outcome: dict[str, float]


class JsonlTradeStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: TradeRecord) -> None:
        payload = asdict(record)
        payload["timestamp"] = record.timestamp.isoformat()
        atomic_append_text(self.path, json.dumps(payload, sort_keys=True) + "\n")

    def trade_ids(self) -> set[str]:
        return {str(record.get("trade_id", "")) for record in self.read_all() if record.get("trade_id")}

    def append_if_new(self, record: TradeRecord) -> bool:
        if record.trade_id in self.trade_ids():
            return False
        self.append(record)
        return True

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]


LEARNING_EVENT_SUMMARY_COLUMNS = [
    "source",
    "events",
    "ready_for_modeling",
    "blocked_events",
    "research_rejected_events",
    "dydx_config_blocked_events",
    "paper_ready_events",
    "blocked_fill_events",
    "submitted_fill_events",
    "outcome_events",
    "profitable_outcomes",
    "avg_realized_return",
    "audit_only_events",
    "modeling_event_threshold",
    "outcome_events_remaining",
    "notes",
]


def learning_event_summary(
    paper_journal_path: str | Path,
    trade_store_path: str | Path,
    min_modeling_events: int = 100,
) -> list[dict[str, object]]:
    if isinstance(min_modeling_events, bool) or not isinstance(min_modeling_events, Integral) or min_modeling_events <= 0:
        raise ValueError("min_modeling_events must be a positive integer")
    min_modeling_events = int(min_modeling_events)
    paper_rows = _read_paper_journal(paper_journal_path)
    trade_rows, malformed_trade_rows = _read_trade_store(trade_store_path)
    paper_claims = _paper_reported_claims(paper_rows)
    trade_claims = _trade_reported_claims(trade_rows)
    rows = [
        _paper_journal_summary(paper_rows, min_modeling_events, paper_claims),
        _trade_store_summary(trade_rows, malformed_trade_rows, min_modeling_events, trade_claims),
    ]
    total_events = int(sum(int(row["events"]) for row in rows))
    # No provenance-bound verifier consumes either source. A reported return is
    # diagnostic evidence, never a verified modeling outcome or gate authority.
    reported_count = _unique_reported_claim_count(paper_claims + trade_claims)
    rows.append(
        {
            "source": "combined",
            "events": total_events,
            "ready_for_modeling": False,
            "blocked_events": int(sum(int(row["blocked_events"]) for row in rows)),
            "research_rejected_events": int(sum(int(row["research_rejected_events"]) for row in rows)),
            "dydx_config_blocked_events": int(sum(int(row["dydx_config_blocked_events"]) for row in rows)),
            "paper_ready_events": int(sum(int(row["paper_ready_events"]) for row in rows)),
            "blocked_fill_events": int(sum(int(row["blocked_fill_events"]) for row in rows)),
            "submitted_fill_events": int(sum(int(row["submitted_fill_events"]) for row in rows)),
            "outcome_events": 0,
            "profitable_outcomes": 0,
            "avg_realized_return": "",
            "audit_only_events": total_events,
            "modeling_event_threshold": min_modeling_events,
            "outcome_events_remaining": min_modeling_events,
            "notes": _unverified_notes(reported_count),
        }
    )
    return rows


def write_learning_event_summary_report(
    paper_journal_path: str | Path,
    trade_store_path: str | Path,
    output_path: str | Path,
    min_modeling_events: int = 100,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(pd.DataFrame(
        learning_event_summary(paper_journal_path, trade_store_path, min_modeling_events),
        columns=LEARNING_EVENT_SUMMARY_COLUMNS,
    ), output, index=False)
    return output


def _read_paper_journal(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(file_path)
    except (pd.errors.EmptyDataError, OSError, UnicodeDecodeError):
        return pd.DataFrame()


def _read_trade_store(path: str | Path) -> tuple[list[dict[str, Any]], int]:
    file_path = Path(path)
    if not file_path.exists():
        return [], 0
    records: list[dict[str, Any]] = []
    malformed = 0
    try:
        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError:
                    malformed += 1
                    continue
                if isinstance(parsed, dict):
                    records.append(parsed)
                else:
                    malformed += 1
    except OSError:
        return [], 0
    return records, malformed


def _paper_journal_summary(
    frame: pd.DataFrame,
    min_modeling_events: int,
    reported_claims: list[tuple[str, float, str, str]],
) -> dict[str, object]:
    if frame.empty:
        return _empty_summary("paper_journal", "missing_or_empty", min_modeling_events)
    statuses = frame.get("plan_status", pd.Series(dtype=str)).fillna("").astype(str)
    reasons = frame.get("plan_reason", pd.Series([""] * len(frame))).fillna("").astype(str)
    fills = frame.get("fills_json", pd.Series([""] * len(frame))).fillna("").astype(str)
    fill_statuses = _fill_statuses(fills)
    events = int(len(frame))
    return {
        "source": "paper_journal",
        "events": events,
        "ready_for_modeling": False,
        "blocked_events": int((statuses == "blocked").sum()),
        "research_rejected_events": int(reasons.str.startswith("research_rejected", na=False).sum()),
        "dydx_config_blocked_events": int(reasons.str.startswith("dydx_not_ready", na=False).sum()),
        "paper_ready_events": int((statuses == "paper_ready").sum()),
        "blocked_fill_events": sum(1 for status in fill_statuses if str(status).startswith("paper_blocked")),
        "submitted_fill_events": sum(1 for status in fill_statuses if status in {"paper_submitted", "confirmed_on_exchange"}),
        "outcome_events": 0,
        "profitable_outcomes": 0,
        "avg_realized_return": "",
        "audit_only_events": events,
        "modeling_event_threshold": min_modeling_events,
        "outcome_events_remaining": min_modeling_events,
        "notes": _unverified_notes(_unique_reported_claim_count(reported_claims)),
    }


def _paper_journal_row_has_verified_outcome(row: pd.Series) -> bool:
    """This source has no bound outcome verifier; a journal row cannot prove one."""
    return False


def _json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    text = str(value or "").strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _snapshot_has_exit_price(snapshot: dict[str, Any]) -> bool:
    return all(
        (price := _finite_number(snapshot.get(f"venue_exit_price_{leg}"))) is not None and price > 0.0
        for leg in ("x", "y")
    )


def _trade_store_summary(
    records: list[dict[str, Any]],
    malformed_rows: int,
    min_modeling_events: int,
    reported_claims: list[tuple[str, float, str, str]],
) -> dict[str, object]:
    if not records:
        note = "missing_or_empty" if malformed_rows == 0 else f"malformed_rows={malformed_rows}"
        return _empty_summary("trade_store", note, min_modeling_events)
    events = len(records)
    notes = _unverified_notes(_unique_reported_claim_count(reported_claims))
    if malformed_rows:
        notes = f"{notes};malformed_rows={malformed_rows}"
    return {
        "source": "trade_store",
        "events": events,
        "ready_for_modeling": False,
        "blocked_events": 0,
        "research_rejected_events": 0,
        "dydx_config_blocked_events": 0,
        "paper_ready_events": 0,
        "blocked_fill_events": 0,
        "submitted_fill_events": 0,
        "outcome_events": 0,
        "profitable_outcomes": 0,
        "avg_realized_return": "",
        "audit_only_events": events,
        "modeling_event_threshold": min_modeling_events,
        "outcome_events_remaining": min_modeling_events,
        "notes": notes,
    }


def _empty_summary(source: str, notes: str, min_modeling_events: int) -> dict[str, object]:
    return {
        "source": source,
        "events": 0,
        "ready_for_modeling": False,
        "blocked_events": 0,
        "research_rejected_events": 0,
        "dydx_config_blocked_events": 0,
        "paper_ready_events": 0,
        "blocked_fill_events": 0,
        "submitted_fill_events": 0,
        "outcome_events": 0,
        "profitable_outcomes": 0,
        "avg_realized_return": "",
        "audit_only_events": 0,
        "modeling_event_threshold": min_modeling_events,
        "outcome_events_remaining": min_modeling_events,
        "notes": f"{notes};verified_outcome_consumer_not_bound",
    }


def _realized_return(outcome: Any) -> float | None:
    if not isinstance(outcome, dict):
        return None
    values = [_finite_number(outcome[key]) for key in ("realized_return", "return") if key in outcome]
    if not values or any(value is None for value in values):
        return None
    return values[0] if all(value == values[0] for value in values) else None


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or type(value).__name__ == "bool_":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _identity_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    text = value.strip()
    return "" if text.lower() in {"", "nan", "none", "null", "<na>"} else text


def _paper_reported_claims(frame: pd.DataFrame) -> list[tuple[str, float, str, str]]:
    claims = []
    for _, row in frame.iterrows():
        if str(row.get("plan_status", "")).strip().lower() not in {"paper_completed", "completed", "closed"}:
            continue
        if str(row.get("lifecycle_status", "")).strip().lower() != "closed":
            continue
        trade_id = _identity_text(row.get("trade_id"))
        value = _realized_return({"realized_return": row.get("realized_return")})
        if not trade_id or value is None or not _snapshot_has_exit_price(_json_dict(row.get("exit_snapshot_json"))):
            continue
        claims.append((trade_id, value, str(row.get("pair", "")).strip().upper(), str(row.get("venue", "")).strip().lower()))
    return claims


def _trade_reported_claims(records: list[dict[str, Any]]) -> list[tuple[str, float, str, str]]:
    claims = []
    for record in records:
        trade_id = _identity_text(record.get("trade_id"))
        value = _realized_return(record.get("outcome"))
        if not trade_id or value is None:
            continue
        execution = record.get("execution")
        venue = execution.get("venue") if isinstance(execution, dict) else ""
        claims.append((trade_id, value, str(record.get("pair", "")).strip().upper(), str(venue or "").strip().lower()))
    return claims


def _unique_reported_claim_count(claims: list[tuple[str, float, str, str]]) -> int:
    by_trade_id: dict[str, set[tuple[float, str, str]]] = {}
    for trade_id, value, pair, venue in claims:
        by_trade_id.setdefault(trade_id, set()).add((value, pair, venue))
    return sum(len(values) == 1 for values in by_trade_id.values())


def _unverified_notes(reported_count: int) -> str:
    return f"reported_normalized_outcomes={reported_count};verified_outcome_consumer_not_bound"


def _fill_statuses(values: pd.Series) -> list[str]:
    statuses: list[str] = []
    for value in values:
        try:
            parsed = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(parsed, list):
            continue
        for item in parsed:
            if isinstance(item, dict) and item.get("status") is not None:
                statuses.append(str(item["status"]))
    return statuses


def _weighted_average_return(rows: list[dict[str, object]]) -> float | str:
    numerator = 0.0
    denominator = 0
    for row in rows:
        avg = row["avg_realized_return"]
        outcome_events = int(row["outcome_events"])
        if avg == "" or outcome_events == 0:
            continue
        numerator += float(avg) * outcome_events
        denominator += outcome_events
    return numerator / denominator if denominator else ""
