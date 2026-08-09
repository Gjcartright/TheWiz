from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import pandas as pd

from quant_platform.active_pipeline import CommandResult, ROOT


RESEARCH_SOURCE_COLUMNS = [
    "source_id",
    "source_type",
    "title",
    "author",
    "channel_or_publisher",
    "date_added",
    "source_path_or_url",
    "topic_tags",
    "status",
    "review_status",
    "quarantine_status",
    "confidence",
    "processed_at",
    "notes",
    "manifest_path",
]

ALLOWED_SOURCE_TYPES = {"youtube", "udemy", "book", "pdf", "article", "note", "manual_rule", "ccxt"}
ALLOWED_SOURCE_STATES = {"active", "provisional", "quarantined", "rejected"}


@dataclass(frozen=True)
class ResearchSourceRecord:
    source_id: str
    source_type: str
    title: str
    author: str
    channel_or_publisher: str
    date_added: str
    source_path_or_url: str
    topic_tags: str
    status: str
    review_status: str
    quarantine_status: str
    confidence: float
    processed_at: str
    notes: str
    manifest_path: str


def research_source_base(root: Path = ROOT) -> Path:
    return root / "data" / "external" / "research_sources"


def research_manifest_dir(root: Path = ROOT) -> Path:
    return research_source_base(root) / "manifests"


def research_registry_path(root: Path = ROOT) -> Path:
    return research_source_base(root) / "research_source_registry.csv"


def _audit_paths(root: Path = ROOT) -> tuple[Path, Path]:
    report_dir = root / "reports" / "research"
    return report_dir / "source_ingestion_audit.csv", report_dir / "source_ingestion_audit.md"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_source_id(value: str) -> str:
    token = "".join(char.lower() if char.isalnum() else "_" for char in value.strip())
    token = "_".join(part for part in token.split("_") if part)
    return token or "research_source"


def _normalize_status(value: str, *, default: str = "active") -> str:
    token = str(value or "").strip().lower() or default
    if token not in ALLOWED_SOURCE_STATES:
        raise ValueError(f"unsupported source status: {token}")
    return token


def _normalize_source_type(value: str) -> str:
    token = str(value or "").strip().lower()
    if token not in ALLOWED_SOURCE_TYPES:
        raise ValueError(f"unsupported research source type: {token}")
    return token


def _normalize_confidence(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.5
    return max(0.0, min(1.0, numeric))


def _stringify_tags(value: Any) -> str:
    if isinstance(value, (list, tuple, set)):
        return ";".join(str(item).strip() for item in value if str(item).strip())
    return str(value or "").strip()


def _read_registry(root: Path = ROOT) -> pd.DataFrame:
    path = research_registry_path(root)
    if not path.exists():
        return pd.DataFrame(columns=RESEARCH_SOURCE_COLUMNS)
    try:
        return pd.read_csv(path).fillna("")
    except (pd.errors.EmptyDataError, OSError, UnicodeDecodeError):
        return pd.DataFrame(columns=RESEARCH_SOURCE_COLUMNS)


def _write_registry(frame: pd.DataFrame, root: Path = ROOT) -> Path:
    path = research_registry_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = frame.reindex(columns=RESEARCH_SOURCE_COLUMNS, fill_value="")
    normalized.to_csv(path, index=False)
    return path


def build_research_source_registry(root: Path = ROOT) -> CommandResult:
    manifest_dir = research_manifest_dir(root)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for path in sorted(manifest_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            rows.append(
                {
                    "source_id": path.stem,
                    "source_type": "",
                    "title": "",
                    "author": "",
                    "channel_or_publisher": "",
                    "date_added": "",
                    "source_path_or_url": "",
                    "topic_tags": "",
                    "status": "quarantined",
                    "review_status": "manifest_invalid",
                    "quarantine_status": "quarantined",
                    "confidence": 0.0,
                    "processed_at": "",
                    "notes": "invalid_manifest_json",
                    "manifest_path": str(path),
                }
            )
            continue
        rows.append(_record_from_payload(payload, manifest_path=path))
    frame = pd.DataFrame(rows, columns=RESEARCH_SOURCE_COLUMNS)
    output = _write_registry(frame, root=root)
    return CommandResult(paths={"research_source_registry": output}, summary={"sources": int(len(frame))})


def ingest_research_source(
    *,
    source_type: str,
    title: str,
    source_path_or_url: str,
    root: Path = ROOT,
    author: str = "",
    channel_or_publisher: str = "",
    topic_tags: str = "",
    status: str = "active",
    review_status: str = "unreviewed",
    quarantine_status: str = "active",
    confidence: float = 0.5,
    notes: str = "",
    source_id: str = "",
) -> CommandResult:
    manifest_dir = research_manifest_dir(root)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    normalized_source_type = _normalize_source_type(source_type)
    normalized_status = _normalize_status(status, default="active")
    normalized_quarantine = _normalize_status(quarantine_status, default=normalized_status)
    date_added = _now_iso()
    record_source_id = source_id or _safe_source_id(f"{normalized_source_type}_{title}")
    manifest_path = manifest_dir / f"{record_source_id}.json"
    payload = {
        "source_id": record_source_id,
        "source_type": normalized_source_type,
        "title": str(title).strip(),
        "author": str(author).strip(),
        "channel_or_publisher": str(channel_or_publisher).strip(),
        "date_added": date_added,
        "source_path_or_url": str(source_path_or_url).strip(),
        "topic_tags": _stringify_tags(topic_tags),
        "status": normalized_status,
        "review_status": str(review_status or "unreviewed").strip(),
        "quarantine_status": normalized_quarantine,
        "confidence": _normalize_confidence(confidence),
        "processed_at": "",
        "notes": str(notes or "").strip(),
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    result = build_research_source_registry(root=root)
    return CommandResult(
        paths={**result.paths, "research_source_manifest": manifest_path},
        summary={"source_id": record_source_id, "sources": result.summary.get("sources", 0)},
    )


def research_source_audit(root: Path = ROOT) -> CommandResult:
    registry = _read_registry(root)
    audit_rows: list[dict[str, object]] = []
    if registry.empty:
        audit_rows.append(
            {
                "metric": "registry_status",
                "value": "empty",
                "detail": "no_research_sources_registered",
            }
        )
    else:
        audit_rows.extend(
            [
                {"metric": "sources_total", "value": int(len(registry)), "detail": ""},
                {
                    "metric": "active_sources",
                    "value": int(registry["status"].astype(str).eq("active").sum()),
                    "detail": "",
                },
                {
                    "metric": "quarantined_sources",
                    "value": int(registry["quarantine_status"].astype(str).eq("quarantined").sum()),
                    "detail": "",
                },
                {
                    "metric": "mean_confidence",
                    "value": round(pd.to_numeric(registry["confidence"], errors="coerce").fillna(0.0).mean(), 4),
                    "detail": "",
                },
                {
                    "metric": "structured_registry_coverage",
                    "value": int(
                        registry[["source_id", "source_type", "title", "source_path_or_url"]]
                        .astype(str)
                        .replace("", pd.NA)
                        .notna()
                        .all(axis=1)
                        .sum()
                    ),
                    "detail": "rows_with_required_registry_fields",
                },
            ]
        )
    audit = pd.DataFrame(audit_rows, columns=["metric", "value", "detail"])
    csv_path, md_path = _audit_paths(root)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(csv_path, index=False)
    md_path.write_text(_audit_markdown(audit), encoding="utf-8")
    return CommandResult(paths={"research_source_audit": csv_path, "research_source_audit_md": md_path}, summary={"rows": int(len(audit))})


def active_research_sources(root: Path = ROOT) -> pd.DataFrame:
    registry = _read_registry(root)
    if registry.empty:
        return registry
    active_mask = registry["status"].astype(str).isin({"active", "provisional"})
    quarantine_mask = ~registry["quarantine_status"].astype(str).isin({"quarantined", "rejected"})
    return registry[active_mask & quarantine_mask].copy().reset_index(drop=True)


def mark_research_sources_processed(source_type: str, *, root: Path = ROOT, processed_at: str | None = None) -> Path:
    """Update freshness lineage without rebuilding or filtering the registry."""

    registry = _read_registry(root)
    if registry.empty:
        return research_registry_path(root)
    normalized_type = _normalize_source_type(source_type)
    mask = registry["source_type"].astype(str).str.lower().eq(normalized_type)
    if mask.any():
        registry.loc[mask, "processed_at"] = processed_at or _now_iso()
        return _write_registry(registry, root=root)
    return research_registry_path(root)


def _record_from_payload(payload: dict[str, Any], *, manifest_path: Path) -> dict[str, object]:
    source_type = _normalize_source_type(str(payload.get("source_type", "") or ""))
    status = _normalize_status(payload.get("status", "active"), default="active")
    quarantine_status = _normalize_status(payload.get("quarantine_status", status), default=status)
    return {
        "source_id": str(payload.get("source_id", manifest_path.stem)).strip() or manifest_path.stem,
        "source_type": source_type,
        "title": str(payload.get("title", "")).strip(),
        "author": str(payload.get("author", "")).strip(),
        "channel_or_publisher": str(payload.get("channel_or_publisher", "")).strip(),
        "date_added": str(payload.get("date_added", "")).strip(),
        "source_path_or_url": str(payload.get("source_path_or_url", "")).strip(),
        "topic_tags": _stringify_tags(payload.get("topic_tags", "")),
        "status": status,
        "review_status": str(payload.get("review_status", "unreviewed")).strip(),
        "quarantine_status": quarantine_status,
        "confidence": _normalize_confidence(payload.get("confidence", 0.5)),
        "processed_at": str(payload.get("processed_at", "")).strip(),
        "notes": str(payload.get("notes", "")).strip(),
        "manifest_path": str(manifest_path),
    }


def _audit_markdown(frame: pd.DataFrame) -> str:
    lines = ["# Research Source Audit", ""]
    if frame.empty:
        lines.append("- no audit rows")
    else:
        for _, row in frame.iterrows():
            detail = f" ({row['detail']})" if str(row.get("detail", "")).strip() else ""
            lines.append(f"- {row['metric']}: {row['value']}{detail}")
    return "\n".join(lines) + "\n"
