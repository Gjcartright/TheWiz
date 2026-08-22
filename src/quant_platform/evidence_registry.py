"""Generated registry for immutable first-class research and market captures."""

from __future__ import annotations

from quant_platform.orchestration.corrective_runtime import atomic_write_text

from quant_platform.orchestration.corrective_runtime import atomic_write_csv

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Iterable

import pandas as pd

SCHEMA_VERSION = "thewiz.evidence_registry.v1"
TIMESTAMP_KEYS = (
    "source_timestamp",
    "captured_at",
    "fetched_at",
    "created_at_utc",
    "generated_at_utc",
    "checked_at_utc",
)


@dataclass(frozen=True)
class SourceRule:
    source_type: str
    source_system: str
    globs: tuple[str, ...]
    freshness_hours: float | None
    point_in_time_if_timestamped: bool
    acceptance_input_eligible: bool
    notes: str


SOURCE_RULES = (
    SourceRule(
        "wizard_capture",
        "crypto_wizards",
        (
            "data/raw/crypto_wizards/**/*.json",
            "data/raw/crypto_wizards_scanner/**/*.json",
            "data/raw/crypto_wizards_custom_series_proofs/**/*.json",
            "data/raw/crypto_wizards_copula_behavioral_proofs/**/*.json",
        ),
        26.0,
        True,
        False,
        "discovery_and_formula_diagnostics_only",
    ),
    SourceRule(
        "hyperliquid_l2",
        "hyperliquid",
        ("data/raw/hyperliquid_l2_books/**/*.json",),
        2.0,
        True,
        True,
        "point_in_time_depth_input_not_historical_fill_proof",
    ),
    SourceRule(
        "hyperliquid_funding",
        "hyperliquid",
        ("data/raw/hyperliquid_funding/**/*.json",),
        26.0,
        True,
        True,
        "local_cost_input_requires_history_alignment",
    ),
    SourceRule(
        "hyperliquid_candles",
        "hyperliquid",
        ("data/raw/hyperliquid_candles/**/*.json",),
        26.0,
        True,
        True,
        "local_replay_input_subject_to_interval_freshness",
    ),
    SourceRule(
        "apify_capture",
        "apify",
        ("data/raw/enrichment/**/*",),
        24.0,
        True,
        False,
        "supplemental_research_only",
    ),
    SourceRule(
        "youtube_capture",
        "youtube",
        ("data/external/youtube/**/*.json", "data/external/youtube/**/*.vtt"),
        None,
        False,
        False,
        "untrusted_educational_research_only",
    ),
    SourceRule(
        "udemy_transcript_metadata",
        "udemy",
        ("data/external/udemy/restricted_transcripts/*.metadata.json",),
        None,
        False,
        False,
        "restricted_transcript_hash_metadata_research_only",
    ),
)


def build_evidence_registry(
    *,
    root: Path,
    now: datetime | None = None,
    rules: Iterable[SourceRule] = SOURCE_RULES,
) -> dict[str, object]:
    as_of = _as_utc(now or datetime.now(timezone.utc))
    rows: list[dict[str, object]] = []
    seen: set[Path] = set()
    for rule in rules:
        for pattern in rule.globs:
            for path in sorted(root.glob(pattern)):
                if (
                    path in seen
                    or not path.is_file()
                    or path.is_symlink()
                    or path.name.startswith(".")
                ):
                    continue
                seen.add(path)
                rows.append(_capture_row(root=root, path=path, rule=rule, as_of=as_of))
    columns = _columns()
    frame = pd.DataFrame(rows, columns=columns)
    if not frame.empty:
        frame = frame.sort_values(
            ["source_system", "source_type", "capture_timestamp", "path"],
            na_position="last",
        ).reset_index(drop=True)
    active = root / "reports" / "active"
    active.mkdir(parents=True, exist_ok=True)
    output = active / "evidence_registry.csv"
    markdown = active / "evidence_registry.md"
    health = active / "evidence_source_health.csv"
    atomic_write_csv(frame, output, index=False, quoting=csv.QUOTE_MINIMAL)
    health_frame = _source_health(frame)
    atomic_write_csv(health_frame, health, index=False)
    atomic_write_text(markdown, _markdown(frame, health_frame, as_of=as_of), encoding="utf-8")
    return {
        "registry": output,
        "markdown": markdown,
        "source_health": health,
        "captures": len(frame),
        "source_systems": int(frame["source_system"].nunique()) if not frame.empty else 0,
        "fresh_captures": int(frame["freshness_status"].eq("fresh").sum())
        if not frame.empty
        else 0,
        "stale_captures": int(frame["freshness_status"].eq("stale").sum())
        if not frame.empty
        else 0,
        "unknown_point_in_time": int(frame["point_in_time_status"].eq("unknown").sum())
        if not frame.empty
        else 0,
        "live_authority_rows": 0,
    }


def _capture_row(
    *,
    root: Path,
    path: Path,
    rule: SourceRule,
    as_of: datetime,
) -> dict[str, object]:
    digest, size = _hash(path)
    timestamp, timestamp_source = _capture_timestamp(path)
    age_hours = (
        max((as_of - timestamp).total_seconds() / 3600.0, 0.0)
        if timestamp is not None
        else float("nan")
    )
    if rule.freshness_hours is None:
        freshness = "not_applicable_research_reference"
        stale_reason = ""
    elif timestamp is None:
        freshness = "unknown"
        stale_reason = "capture_timestamp_missing_or_unparseable"
    elif age_hours <= rule.freshness_hours:
        freshness = "fresh"
        stale_reason = ""
    else:
        freshness = "stale"
        stale_reason = f"age_hours>{rule.freshness_hours:g}"
    point_in_time = (
        "confirmed_capture_timestamp"
        if timestamp is not None and rule.point_in_time_if_timestamped
        else "research_reference_not_market_point_in_time"
        if not rule.point_in_time_if_timestamped
        else "unknown"
    )
    acceptance_input = bool(
        rule.acceptance_input_eligible
        and point_in_time == "confirmed_capture_timestamp"
        and freshness == "fresh"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "capture_id": f"evidence_{digest[:24]}",
        "source_type": rule.source_type,
        "source_system": rule.source_system,
        "path": str(path.relative_to(root)),
        "content_sha256": digest,
        "size_bytes": size,
        "capture_timestamp": timestamp.isoformat() if timestamp is not None else "",
        "timestamp_source": timestamp_source,
        "filesystem_modified_at": datetime.fromtimestamp(
            path.stat().st_mtime, timezone.utc
        ).isoformat(),
        "age_hours": age_hours,
        "freshness_policy_hours": rule.freshness_hours if rule.freshness_hours is not None else "",
        "freshness_status": freshness,
        "stale_reason": stale_reason,
        "point_in_time_status": point_in_time,
        "research_eligible": True,
        "acceptance_input_eligible": acceptance_input,
        "testnet_authority": False,
        "live_authority": False,
        "untrusted_text": rule.source_system in {"crypto_wizards", "apify", "youtube", "udemy"},
        "notes": rule.notes,
    }


def _capture_timestamp(path: Path) -> tuple[datetime | None, str]:
    if path.suffix.lower() == ".json" and path.stat().st_size <= 20_000_000:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            payload = None
        timestamp = _find_timestamp(payload)
        if timestamp is not None:
            return timestamp, "payload"
    return None, "missing"


def _find_timestamp(value: object, *, depth: int = 0) -> datetime | None:
    if depth > 3:
        return None
    if isinstance(value, dict):
        for key in TIMESTAMP_KEYS:
            if key in value:
                parsed = pd.to_datetime(value[key], utc=True, errors="coerce")
                if pd.notna(parsed):
                    return parsed.to_pydatetime()
        for key in ("metadata", "request", "response", "payload"):
            if key in value:
                nested = _find_timestamp(value[key], depth=depth + 1)
                if nested is not None:
                    return nested
    return None


def _hash(path: Path) -> tuple[str, int]:
    digest = sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return digest.hexdigest(), size


def _source_health(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "source_system",
        "source_type",
        "captures",
        "latest_capture_timestamp",
        "fresh_captures",
        "stale_captures",
        "unknown_freshness",
        "acceptance_input_eligible",
        "source_status",
        "blocker",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    for (system, source_type), group in frame.groupby(["source_system", "source_type"], sort=True):
        latest = pd.to_datetime(group["capture_timestamp"], utc=True, errors="coerce").max()
        fresh = int(group["freshness_status"].eq("fresh").sum())
        stale = int(group["freshness_status"].eq("stale").sum())
        unknown = int(group["freshness_status"].eq("unknown").sum())
        eligible = int(group["acceptance_input_eligible"].astype(bool).sum())
        reference_only = group["freshness_status"].eq("not_applicable_research_reference").all()
        status = "reference_only" if reference_only else "fresh" if fresh else "blocked"
        blocker = "" if status != "blocked" else "no_fresh_timestamped_capture"
        rows.append(
            {
                "source_system": system,
                "source_type": source_type,
                "captures": len(group),
                "latest_capture_timestamp": latest.isoformat() if pd.notna(latest) else "",
                "fresh_captures": fresh,
                "stale_captures": stale,
                "unknown_freshness": unknown,
                "acceptance_input_eligible": eligible,
                "source_status": status,
                "blocker": blocker,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _markdown(frame: pd.DataFrame, health: pd.DataFrame, *, as_of: datetime) -> str:
    fresh = int(frame["freshness_status"].eq("fresh").sum()) if not frame.empty else 0
    stale = int(frame["freshness_status"].eq("stale").sum()) if not frame.empty else 0
    return "\n".join(
        [
            "# Evidence Registry",
            "",
            f"Generated at `{as_of.isoformat()}` from immutable first-class captures.",
            "Derived dashboards do not override raw capture timestamps or authority.",
            "No registry row grants Testnet or live order authority.",
            "",
            "## Summary",
            "",
            f"- Captures: `{len(frame)}`",
            f"- Content hashes: `{frame['content_sha256'].nunique() if not frame.empty else 0}`",
            f"- Fresh market captures: `{fresh}`",
            f"- Stale market captures: `{stale}`",
            "",
            "## Source Health",
            "",
            health.to_markdown(index=False) if not health.empty else "No captures found.",
            "",
        ]
    )


def _columns() -> list[str]:
    return [
        "schema_version",
        "capture_id",
        "source_type",
        "source_system",
        "path",
        "content_sha256",
        "size_bytes",
        "capture_timestamp",
        "timestamp_source",
        "filesystem_modified_at",
        "age_hours",
        "freshness_policy_hours",
        "freshness_status",
        "stale_reason",
        "point_in_time_status",
        "research_eligible",
        "acceptance_input_eligible",
        "testnet_authority",
        "live_authority",
        "untrusted_text",
        "notes",
    ]


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
