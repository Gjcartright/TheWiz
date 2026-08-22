from __future__ import annotations

from quant_platform.orchestration.corrective_runtime import atomic_write_text

from quant_platform.orchestration.corrective_runtime import atomic_write_csv

import json
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import pandas as pd

from quant_platform.active_pipeline import ROOT, CommandResult

SCHEMA_VERSION = "udemy_transcript_vault.v1"
PROMOTION_AUTHORITY = "none_research_only"
MANIFEST_COLUMNS = (
    "schema_version",
    "lecture_id",
    "course",
    "section",
    "video_title",
    "lecture_url",
    "expected_evidence_source",
    "vault_status",
    "transcript_path",
    "transcript_sha256",
    "transcript_bytes",
    "transcript_word_count",
    "transcript_cue_count",
    "captured_at_utc",
    "independent_rereview_ready",
    "transcript_text_ingested",
    "point_in_time_status",
    "live_signal_eligible",
    "promotion_authority",
    "blocker",
)


def udemy_transcript_vault_paths(root: Path = ROOT) -> dict[str, Path]:
    vault = root / "data" / "external" / "udemy" / "restricted_transcripts"
    reports = root / "reports" / "research"
    return {
        "vault": vault,
        "capture_queue": vault / "capture_queue.json",
        "manifest": reports / "udemy_transcript_vault_manifest.csv",
        "audit": reports / "udemy_transcript_vault_audit.json",
        "audit_md": reports / "udemy_transcript_vault_audit.md",
    }


def build_udemy_transcript_vault_inventory(
    *,
    root: Path = ROOT,
    lecture_index_path: Path | None = None,
    recheck_unavailable: bool = False,
) -> CommandResult:
    paths = udemy_transcript_vault_paths(root)
    source = lecture_index_path or (
        root / "data" / "processed" / "research_knowledge" / "udemy_lecture_index.csv"
    )
    lectures = _read_csv(source)
    required = {"lecture_id", "course", "section", "video_title", "lecture_url"}
    missing = sorted(required - set(lectures.columns))
    if missing:
        raise ValueError(f"udemy_lecture_index_missing_columns:{';'.join(missing)}")

    paths["vault"].mkdir(parents=True, exist_ok=True)
    paths["manifest"].parent.mkdir(parents=True, exist_ok=True)
    rows = [
        _manifest_row(row, root=root, vault=paths["vault"]) for row in lectures.to_dict("records")
    ]
    manifest = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    manifest = _block_duplicate_transcript_bodies(manifest)
    atomic_write_csv(manifest, paths["manifest"], index=False)

    queue = [
        {
            "lecture_id": row["lecture_id"],
            "course": row["course"],
            "section": row["section"],
            "video_title": row["video_title"],
            "lecture_url": row["lecture_url"],
            "vault_status": row["vault_status"],
        }
        for row in manifest.to_dict("records")
        if row["vault_status"] != "captured"
        and (recheck_unavailable or row["vault_status"] != "transcript_unavailable")
    ]
    atomic_write_text(paths["capture_queue"], json.dumps(queue, indent=2, sort_keys=True), encoding="utf-8")

    summary = _summary(
        manifest,
        queue=queue,
        source=source,
        recheck_unavailable=recheck_unavailable,
    )
    atomic_write_text(paths["audit"], json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    atomic_write_text(paths["audit_md"], _audit_markdown(summary), encoding="utf-8")
    return CommandResult(
        paths={key: value for key, value in paths.items() if key != "vault"},
        summary=summary,
    )


def _manifest_row(row: dict[str, object], *, root: Path, vault: Path) -> dict[str, object]:
    lecture_id = _text(row.get("lecture_id"))
    transcript = vault / f"{lecture_id}.txt"
    metadata_path = vault / f"{lecture_id}.metadata.json"
    metadata = _read_json(metadata_path)
    status = _text(metadata.get("status"))
    blocker = _text(metadata.get("blocker"))
    transcript_text = ""
    if transcript.is_file():
        transcript_text = transcript.read_text(encoding="utf-8").strip()
        status = "captured" if transcript_text else "empty_transcript"
        if not transcript_text:
            blocker = "transcript_file_empty"
    elif status == "transcript_unavailable":
        blocker = blocker or "udemy_transcript_not_available"
    else:
        status = status or "missing"
        blocker = blocker or "transcript_capture_pending"

    source_url = _text(metadata.get("lecture_url"))
    expected_url = _text(row.get("lecture_url"))
    if status == "captured" and source_url and source_url != expected_url:
        status = "blocked_source_url_mismatch"
        blocker = "captured_lecture_url_does_not_match_index"
    ready = status == "captured" and bool(transcript_text)
    return {
        "schema_version": SCHEMA_VERSION,
        "lecture_id": lecture_id,
        "course": _text(row.get("course")),
        "section": _text(row.get("section")),
        "video_title": _text(row.get("video_title")),
        "lecture_url": expected_url,
        "expected_evidence_source": _text(row.get("evidence_source")),
        "vault_status": status,
        "transcript_path": _relative(transcript, root) if ready else "",
        "transcript_sha256": _hash_file(transcript) if ready else "",
        "transcript_bytes": transcript.stat().st_size if ready else 0,
        "transcript_word_count": len(transcript_text.split()) if ready else 0,
        "transcript_cue_count": int(metadata.get("cue_count") or 0) if ready else 0,
        "captured_at_utc": _text(metadata.get("captured_at_utc")),
        "independent_rereview_ready": ready,
        "transcript_text_ingested": False,
        "point_in_time_status": "reference_only_not_market_data",
        "live_signal_eligible": False,
        "promotion_authority": PROMOTION_AUTHORITY,
        "blocker": "" if ready else blocker,
    }


def _summary(
    manifest: pd.DataFrame,
    *,
    queue: list[dict[str, object]],
    source: Path,
    recheck_unavailable: bool,
) -> dict[str, object]:
    captured = (
        manifest["vault_status"].eq("captured") if not manifest.empty else pd.Series(dtype=bool)
    )
    unavailable = (
        manifest["vault_status"].eq("transcript_unavailable")
        if not manifest.empty
        else pd.Series(dtype=bool)
    )
    direct = (
        manifest["expected_evidence_source"].eq("udemy_transcript")
        if not manifest.empty
        else pd.Series(dtype=bool)
    )
    direct_complete = bool((captured | ~direct).all()) if not manifest.empty else False
    terminal = captured | unavailable
    resolution_complete = bool(terminal.all()) if not manifest.empty else False
    rereview_coverage = float(captured.mean() * 100.0) if not manifest.empty else 0.0
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "lecture_index_path": str(source),
        "lectures_total": int(len(manifest)),
        "transcripts_captured": int(captured.sum()),
        "transcripts_unavailable": int(unavailable.sum()),
        "terminal_lecture_records": int(terminal.sum()),
        "captures_pending": int(len(queue)),
        "recheck_unavailable": bool(recheck_unavailable),
        "capture_resolution_complete": resolution_complete,
        "all_available_transcripts_rereview_ready": bool(resolution_complete and len(queue) == 0),
        "independent_rereview_coverage_pct": round(rereview_coverage, 4),
        "direct_transcript_expected": int(direct.sum()),
        "direct_transcript_complete": direct_complete,
        "independent_rereview_ready": bool(direct_complete and len(queue) == 0),
        "transcript_text_ingested": False,
        "live_signal_eligible": False,
        "promotion_authority": PROMOTION_AUTHORITY,
    }


def _block_duplicate_transcript_bodies(manifest: pd.DataFrame) -> pd.DataFrame:
    if manifest.empty:
        return manifest
    captured = manifest["vault_status"].eq("captured")
    hashes = manifest["transcript_sha256"].astype(str)
    duplicated = captured & hashes.ne("") & hashes.duplicated(keep=False)
    if not duplicated.any():
        return manifest
    result = manifest.copy()
    result.loc[duplicated, "vault_status"] = "blocked_duplicate_transcript_body"
    result.loc[duplicated, "independent_rereview_ready"] = False
    result.loc[duplicated, "blocker"] = "duplicate_transcript_body_across_lecture_ids"
    return result


def _audit_markdown(summary: dict[str, object]) -> str:
    return "\n".join(
        [
            "# Udemy Restricted Transcript Vault Audit",
            "",
            f"- lectures total: {summary['lectures_total']}",
            f"- transcripts captured: {summary['transcripts_captured']}",
            f"- transcripts unavailable: {summary['transcripts_unavailable']}",
            f"- terminal lecture records: {summary['terminal_lecture_records']}",
            f"- captures pending: {summary['captures_pending']}",
            "- capture resolution complete: "
            f"`{str(summary['capture_resolution_complete']).lower()}`",
            "- available transcripts re-review ready: "
            f"`{str(summary['all_available_transcripts_rereview_ready']).lower()}`",
            f"- independent re-review coverage: {summary['independent_rereview_coverage_pct']}%",
            f"- direct transcript expected: {summary['direct_transcript_expected']}",
            f"- direct transcript complete: `{str(summary['direct_transcript_complete']).lower()}`",
            "- independent re-review ready: "
            f"`{str(summary['independent_rereview_ready']).lower()}`",
            "- transcript bodies are stored only in the ignored restricted local vault",
            "- transcript bodies are never ingested as model features or live signals",
            "- promotion authority: `none_research_only`",
            "",
        ]
    )


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path).fillna("") if path.is_file() else pd.DataFrame()


def _read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {"status": "invalid_metadata", "blocker": "metadata_json_invalid"}
    return value if isinstance(value, dict) else {}


def _hash_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest() if path.is_file() else ""


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _text(value: object) -> str:
    return str(value or "").strip()
