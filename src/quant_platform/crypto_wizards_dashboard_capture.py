"""Ingest exhaustive live Crypto Wizards scanner captures."""

from __future__ import annotations

from quant_platform.orchestration.corrective_runtime import atomic_write_text

from quant_platform.orchestration.corrective_runtime import atomic_write_csv

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd

from quant_platform.crypto_wizards_scanner import scanner_rows_from_payload
from quant_platform.runtime_types import CommandResult
from quant_platform.wizard_symbols import normalize_wizard_exchange

ROOT = Path(__file__).resolve().parents[2]
CAPTURE_SCHEMA_VERSION = "exhaustive_wizard_dashboard_capture.v1"
EXPECTED_CRYPTO_VENUES = ("binance", "binanceus", "bybit", "coinbase", "dydx")
EXPECTED_TIMEFRAMES = ("daily", "hourly")
ALL_CASE_FILTERS = {
    "cointegration": "null",
    "correlation": "null",
    "hurst": "null",
    "half_life": "null",
    "copula": "false",
    "strategy": "null",
    "symbol": "",
}


def ingest_exhaustive_wizard_dashboard_captures(
    *,
    root: Path = ROOT,
    input_dir: Path | None = None,
) -> CommandResult:
    """Normalize all live scanner captures without filtering or deduplicating rows."""

    input_dir = input_dir or _latest_capture_directory(root)
    active = root / "reports" / "active"
    active.mkdir(parents=True, exist_ok=True)
    paths = {
        "rows": active / "exhaustive_wizard_dashboard_rows.csv",
        "capture_manifest": active / "exhaustive_wizard_dashboard_capture_manifest.csv",
        "coverage_validation": active / "exhaustive_wizard_dashboard_capture_validation.csv",
        "summary": active / "exhaustive_wizard_dashboard_capture_summary.json",
        "summary_md": active / "exhaustive_wizard_dashboard_capture_summary.md",
    }

    row_records: list[dict[str, object]] = []
    manifest_records: list[dict[str, object]] = []
    capture_files = sorted(input_dir.glob("*.json")) if input_dir and input_dir.exists() else []
    for capture_path in capture_files:
        payload = _read_json(capture_path)
        metadata = payload.get("capture_metadata", {}) if isinstance(payload, dict) else {}
        filters = payload.get("scanner_filters", {}) if isinstance(payload, dict) else {}
        raw_rows = payload.get("scanner_rows", []) if isinstance(payload, dict) else []
        if not isinstance(metadata, dict):
            metadata = {}
        if not isinstance(filters, dict):
            filters = {}
        if not isinstance(raw_rows, list):
            raw_rows = []
        parsed_rows = scanner_rows_from_payload(payload, source_path=_relative(capture_path, root))
        file_hash = _file_hash(capture_path)
        capture_id = "wcapture_" + file_hash[:20]
        exchange = normalize_wizard_exchange(filters.get("exchange")) or ""
        timeframe = _token(filters.get("interval"))
        expected_cell = exchange in EXPECTED_CRYPTO_VENUES and timeframe in EXPECTED_TIMEFRAMES
        filters_all_cases, filter_blocker = _all_case_filter_status(filters)
        pagination_complete = _truthy(metadata.get("pagination_complete"))
        declared_count = _integer(metadata.get("loaded_row_count"))
        count_matches = declared_count == len(raw_rows) == len(parsed_rows)
        blockers: list[str] = []
        if not expected_cell:
            blockers.append("unexpected_or_missing_venue_timeframe")
        if not filters_all_cases:
            blockers.append(filter_blocker)
        if not pagination_complete:
            blockers.append("infinite_scroll_completion_not_proven")
        if not count_matches:
            blockers.append("capture_row_count_mismatch")
        if not parsed_rows:
            blockers.append("capture_contains_no_rows")
        capture_status = "COMPLETE" if not blockers else "BLOCKED"
        manifest_records.append(
            {
                "schema_version": CAPTURE_SCHEMA_VERSION,
                "capture_id": capture_id,
                "capture_path": _relative(capture_path, root),
                "capture_sha256": file_hash,
                "captured_at": _text(metadata.get("captured_at")),
                "source_url": _text(metadata.get("source_url")),
                "wizard_exchange": exchange,
                "timeframe": timeframe,
                "cell_id": f"{exchange}|{timeframe}" if exchange and timeframe else "",
                "priority_order": _text(filters.get("priority")),
                "discovery_prefilter_applied": not filters_all_cases,
                "pagination_mechanism": _text(metadata.get("pagination_mechanism")),
                "pagination_completion_signal": _text(metadata.get("pagination_completion_signal")),
                "pagination_complete": pagination_complete,
                "declared_row_count": declared_count,
                "raw_row_count": len(raw_rows),
                "parsed_row_count": len(parsed_rows),
                "row_count_matches": count_matches,
                "capture_status": capture_status,
                "blocker": ";".join(blockers),
            }
        )
        for index, parsed in enumerate(parsed_rows):
            normalized = parsed.to_row()
            original = raw_rows[index] if index < len(raw_rows) and isinstance(raw_rows[index], dict) else {}
            record: dict[str, object] = {
                "capture_contract_schema_version": CAPTURE_SCHEMA_VERSION,
                "capture_id": capture_id,
                "capture_path": _relative(capture_path, root),
                "capture_sha256": file_hash,
                "capture_status": capture_status,
                "capture_blocker": ";".join(blockers),
                "capture_row_index": index,
                "capture_row_count": len(parsed_rows),
                "scanner_exchange": exchange,
                "scanner_interval": timeframe,
                "pagination_complete": pagination_complete,
                "discovery_prefilter_applied": not filters_all_cases,
            }
            record.update(normalized)
            for key, value in original.items():
                record[f"dashboard_raw_{key}"] = _json_cell(value)
            row_records.append(record)

    rows = pd.DataFrame(row_records)
    manifest = pd.DataFrame(manifest_records, columns=_manifest_columns())
    expected_cells = {
        f"{venue}|{timeframe}"
        for venue in EXPECTED_CRYPTO_VENUES
        for timeframe in EXPECTED_TIMEFRAMES
    }
    completed_cells = set(
        manifest.loc[manifest["capture_status"].eq("COMPLETE"), "cell_id"].astype(str)
    ) if not manifest.empty else set()
    duplicate_cells = sorted(
        cell for cell, count in manifest.get("cell_id", pd.Series(dtype=object)).value_counts().items() if cell and count > 1
    )
    missing_cells = sorted(expected_cells - completed_cells)
    unexpected_cells = sorted(set(manifest.get("cell_id", pd.Series(dtype=object)).astype(str)) - expected_cells - {""})
    validation = pd.DataFrame(
        [
            _validation_row("capture_directory_available", bool(input_dir and input_dir.exists()), "capture directory missing"),
            _validation_row("capture_files_available", bool(capture_files), "no dashboard capture files found"),
            _validation_row("all_expected_crypto_cells_complete", not missing_cells, ";".join(missing_cells)),
            _validation_row("no_duplicate_capture_cells", not duplicate_cells, ";".join(duplicate_cells)),
            _validation_row("no_unexpected_capture_cells", not unexpected_cells, ";".join(unexpected_cells)),
            _validation_row(
                "all_source_rows_preserved",
                int(manifest["parsed_row_count"].sum()) == len(rows) if not manifest.empty else False,
                f"manifest_rows={int(manifest['parsed_row_count'].sum()) if not manifest.empty else 0};ledger_rows={len(rows)}",
            ),
            _validation_row(
                "zero_discovery_prefilters",
                bool(not rows.empty and not rows["discovery_prefilter_applied"].astype(bool).any()),
                "one or more capture cells used a discovery filter",
            ),
            _validation_row(
                "all_infinite_scroll_captures_complete",
                bool(not manifest.empty and manifest["pagination_complete"].astype(bool).all()),
                "one or more cells lacks a terminal infinite-scroll signal",
            ),
        ],
        columns=["check", "status", "reason"],
    )
    all_complete = bool(not validation.empty and validation["status"].eq("PASS").all())
    summary: dict[str, object] = {
        "schema_version": CAPTURE_SCHEMA_VERSION,
        "input_dir": _relative(input_dir, root) if input_dir else "",
        "capture_files": len(capture_files),
        "expected_cells": len(expected_cells),
        "completed_cells": len(completed_cells & expected_cells),
        "missing_cells": missing_cells,
        "duplicate_cells": duplicate_cells,
        "unexpected_cells": unexpected_cells,
        "source_rows": len(rows),
        "all_expected_cells_complete": all_complete,
        "discovery_policy": "all_cases_no_threshold_prefilter",
        "pagination_policy": "scroll_until_five_consecutive_bottom_fetches_without_growth",
        "live_trading_authorized": False,
    }
    atomic_write_csv(rows, paths["rows"], index=False)
    atomic_write_csv(manifest, paths["capture_manifest"], index=False)
    atomic_write_csv(validation, paths["coverage_validation"], index=False)
    atomic_write_text(paths["summary"], json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    atomic_write_text(paths["summary_md"], _summary_markdown(summary), encoding="utf-8")
    return CommandResult(paths=paths, summary=summary)


def _latest_capture_directory(root: Path) -> Path:
    base = root / "data" / "raw" / "crypto_wizards" / "dashboard_scanner"
    candidates = sorted(path for path in base.glob("*") if path.is_dir()) if base.exists() else []
    return candidates[-1] if candidates else base


def _all_case_filter_status(filters: dict[str, Any]) -> tuple[bool, str]:
    blockers: list[str] = []
    for field, expected in ALL_CASE_FILTERS.items():
        actual = _token(filters.get(field))
        if actual != expected:
            blockers.append(f"{field}={actual or 'missing'}")
    return not blockers, "non_all_case_filters:" + ",".join(blockers) if blockers else ""


def _manifest_columns() -> list[str]:
    return [
        "schema_version",
        "capture_id",
        "capture_path",
        "capture_sha256",
        "captured_at",
        "source_url",
        "wizard_exchange",
        "timeframe",
        "cell_id",
        "priority_order",
        "discovery_prefilter_applied",
        "pagination_mechanism",
        "pagination_completion_signal",
        "pagination_complete",
        "declared_row_count",
        "raw_row_count",
        "parsed_row_count",
        "row_count_matches",
        "capture_status",
        "blocker",
    ]


def _validation_row(check: str, passed: bool, reason: str) -> dict[str, str]:
    return {"check": check, "status": "PASS" if passed else "FAIL", "reason": "" if passed else reason}


def _summary_markdown(summary: dict[str, object]) -> str:
    return "\n".join(
        [
            "# Exhaustive Crypto Wizards Dashboard Capture",
            "",
            f"- Capture files: {summary['capture_files']}",
            f"- Complete venue/timeframe cells: {summary['completed_cells']} / {summary['expected_cells']}",
            f"- Source rows retained: {summary['source_rows']}",
            f"- Complete: `{str(summary['all_expected_cells_complete']).lower()}`",
            f"- Missing cells: `{';'.join(summary['missing_cells']) or 'none'}`",
            f"- Duplicate cells: `{';'.join(summary['duplicate_cells']) or 'none'}`",
            "- Discovery thresholds applied: `false`",
            "- Live trading authorized: `false`",
            "",
            "Completeness means every configured crypto venue/timeframe reached a terminal infinite-scroll state under all-case filters. It does not make dashboard statistics point-in-time acceptance evidence.",
            "",
        ]
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def _integer(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _truthy(value: object) -> bool:
    return value is True or _text(value).lower() in {"1", "true", "yes", "pass", "complete"}


def _token(value: object) -> str:
    return "".join(character for character in _text(value).strip().lower() if character.isalnum())


def _text(value: object) -> str:
    return "" if value is None else str(value).strip()


def _json_cell(value: object) -> object:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)
