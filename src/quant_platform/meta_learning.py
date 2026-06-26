from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import json
from typing import Any

import pandas as pd


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
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

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
    paper_rows = _read_paper_journal(paper_journal_path)
    trade_rows, malformed_trade_rows = _read_trade_store(trade_store_path)
    rows = [
        _paper_journal_summary(paper_rows, min_modeling_events),
        _trade_store_summary(trade_rows, malformed_trade_rows, min_modeling_events),
    ]
    total_events = int(sum(int(row["events"]) for row in rows))
    outcome_events = int(sum(int(row["outcome_events"]) for row in rows))
    ready = outcome_events >= min_modeling_events
    rows.append(
        {
            "source": "combined",
            "events": total_events,
            "ready_for_modeling": ready,
            "blocked_events": int(sum(int(row["blocked_events"]) for row in rows)),
            "research_rejected_events": int(sum(int(row["research_rejected_events"]) for row in rows)),
            "dydx_config_blocked_events": int(sum(int(row["dydx_config_blocked_events"]) for row in rows)),
            "paper_ready_events": int(sum(int(row["paper_ready_events"]) for row in rows)),
            "blocked_fill_events": int(sum(int(row["blocked_fill_events"]) for row in rows)),
            "submitted_fill_events": int(sum(int(row["submitted_fill_events"]) for row in rows)),
            "outcome_events": outcome_events,
            "profitable_outcomes": int(sum(int(row["profitable_outcomes"]) for row in rows)),
            "avg_realized_return": _weighted_average_return(rows),
            "audit_only_events": max(total_events - outcome_events, 0),
            "modeling_event_threshold": min_modeling_events,
            "outcome_events_remaining": max(min_modeling_events - outcome_events, 0),
            "notes": "modeling_ready" if ready else "needs_more_realized_outcomes",
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
    pd.DataFrame(
        learning_event_summary(paper_journal_path, trade_store_path, min_modeling_events),
        columns=LEARNING_EVENT_SUMMARY_COLUMNS,
    ).to_csv(output, index=False)
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


def _paper_journal_summary(frame: pd.DataFrame, min_modeling_events: int) -> dict[str, object]:
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
        "submitted_fill_events": sum(1 for status in fill_statuses if status == "paper_submitted"),
        "outcome_events": 0,
        "profitable_outcomes": 0,
        "avg_realized_return": "",
        "audit_only_events": events,
        "modeling_event_threshold": min_modeling_events,
        "outcome_events_remaining": min_modeling_events,
        "notes": "handoff_audit_only" if events < min_modeling_events else "needs_realized_outcomes",
    }


def _trade_store_summary(records: list[dict[str, Any]], malformed_rows: int, min_modeling_events: int) -> dict[str, object]:
    if not records:
        note = "missing_or_empty" if malformed_rows == 0 else f"malformed_rows={malformed_rows}"
        return _empty_summary("trade_store", note, min_modeling_events)
    realized_returns = [_realized_return(record.get("outcome", {})) for record in records]
    realized_returns = [value for value in realized_returns if value is not None]
    events = len(records)
    outcome_events = len(realized_returns)
    ready = outcome_events >= min_modeling_events
    notes = "modeling_ready" if ready else "needs_more_realized_outcomes"
    if malformed_rows:
        notes = f"{notes};malformed_rows={malformed_rows}"
    return {
        "source": "trade_store",
        "events": events,
        "ready_for_modeling": ready,
        "blocked_events": 0,
        "research_rejected_events": 0,
        "dydx_config_blocked_events": 0,
        "paper_ready_events": 0,
        "blocked_fill_events": 0,
        "submitted_fill_events": 0,
        "outcome_events": outcome_events,
        "profitable_outcomes": sum(1 for value in realized_returns if value > 0),
        "avg_realized_return": sum(realized_returns) / outcome_events if outcome_events else "",
        "audit_only_events": max(events - outcome_events, 0),
        "modeling_event_threshold": min_modeling_events,
        "outcome_events_remaining": max(min_modeling_events - outcome_events, 0),
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
        "notes": notes,
    }


def _realized_return(outcome: Any) -> float | None:
    if not isinstance(outcome, dict):
        return None
    for key in ("realized_return", "return", "pnl_pct", "pnl"):
        value = outcome.get(key)
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


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
