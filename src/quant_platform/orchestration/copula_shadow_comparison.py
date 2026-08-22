"""Append-only comparison of sequential and dynamic Copula shadow decisions."""

from __future__ import annotations

from quant_platform.orchestration.corrective_runtime import atomic_write_text

from quant_platform.orchestration.corrective_runtime import atomic_write_csv

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd

from quant_platform.orchestration.contracts import CandidateIdentity, ComparisonEvent, EvidencePacket
from quant_platform.orchestration.copula_shadow_cell import run_copula_shadow_cell
from quant_platform.orchestration.dynamic_ledger import DynamicAgentLedger
from quant_platform.orchestration.venue_gate import VenuePolicy


ROOT = Path(__file__).resolve().parents[3]
EVENTS_FILENAME = "copula_comparison_events.jsonl"
MISSING_DASHBOARD_TIMESTAMP = datetime(1970, 1, 1, tzinfo=timezone.utc)


def build_copula_shadow_comparison(
    *,
    candidate: CandidateIdentity,
    root: Path = ROOT,
    venue_policy: VenuePolicy | None = None,
    now: datetime | None = None,
) -> dict[str, Path | str | bool]:
    """Append one immutable, shadow-only comparison event and refresh its view."""

    now = now or datetime.now(timezone.utc)
    dynamic = run_copula_shadow_cell(candidate=candidate, root=root, venue_policy=venue_policy, now=now)
    sequential_bucket, sequential_reason, sequential_path = _sequential_status(candidate, root)
    dynamic_decision = dynamic["decision"]
    packets = tuple(dynamic["evidence_packets"])
    source_timestamp, source_snapshot_id = _source_snapshot(candidate, packets)
    classification = _classification(sequential_bucket, dynamic_decision.decision.value)
    configuration_id = candidate.candidate_id
    event_id = _event_id(
        configuration_id=configuration_id,
        source_snapshot_id=source_snapshot_id,
        sequential_decision=sequential_bucket,
        dynamic_decision=dynamic_decision.decision.value,
        evidence_packet_ids=tuple(packet.packet_id for packet in packets),
        veto_ids=dynamic_decision.veto_ids,
    )
    event = ComparisonEvent(
        event_id=event_id,
        candidate_id=candidate.candidate_id,
        configuration_id=configuration_id,
        source_snapshot_id=source_snapshot_id,
        pair=candidate.pair,
        venue=candidate.venue,
        timeframe=candidate.timeframe,
        source_timestamp=source_timestamp,
        sequential_decision=sequential_bucket,
        sequential_reason=sequential_reason,
        sequential_evidence_path=sequential_path,
        dynamic_decision=dynamic_decision.decision,
        dynamic_reason=dynamic_decision.reason,
        dynamic_veto_count=len(dynamic_decision.veto_ids),
        dynamic_evidence_count=len(dynamic_decision.evidence_packet_ids),
        evidence_packet_ids=dynamic_decision.evidence_packet_ids,
        comparison=classification,
        created_at=source_timestamp,
    )
    ledger = DynamicAgentLedger(root)
    appended = ledger.append_comparison_event(event)
    output, markdown = _write_comparison_view(root)
    return {
        "csv": output,
        "markdown": markdown,
        "comparison": classification,
        "event_id": event.event_id,
        "appended": appended,
    }


def load_copula_comparison_events(root: Path = ROOT) -> pd.DataFrame:
    """Load immutable events; corrupted rows fail closed instead of being ignored."""

    path = root / "reports" / "orchestration" / "dynamic_agents" / EVENTS_FILENAME
    if not path.exists():
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(ComparisonEvent.model_validate_json(line).model_dump(mode="json"))
    return pd.DataFrame(rows)


def _write_comparison_view(root: Path) -> tuple[Path, Path]:
    output = root / "reports" / "orchestration" / "dynamic_agents" / "copula_shadow_comparison.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = load_copula_comparison_events(root)
    if not frame.empty:
        frame = frame.sort_values(["source_timestamp", "event_id"]).reset_index(drop=True)
    atomic_write_csv(frame, output, index=False)
    markdown = output.with_suffix(".md")
    body = frame.to_markdown(index=False) if not frame.empty else "No immutable Copula comparison events have been recorded."
    atomic_write_text(markdown, "# Copula Shadow Comparison Events\n\n" + body + "\n", encoding="utf-8")
    return output, markdown


def _source_snapshot(candidate: CandidateIdentity, packets: tuple[EvidencePacket, ...]) -> tuple[datetime, str]:
    dashboard = next((packet for packet in packets if packet.evidence_type == "dashboard_snapshot"), None)
    if dashboard is not None:
        raw = f"{candidate.candidate_id}|{dashboard.source_timestamp.isoformat()}|{dashboard.evidence_content_hash}"
        return dashboard.source_timestamp, f"snapshot_{sha256(raw.encode()).hexdigest()[:20]}"
    raw = f"{candidate.candidate_id}|missing_dashboard"
    return MISSING_DASHBOARD_TIMESTAMP, f"snapshot_{sha256(raw.encode()).hexdigest()[:20]}"


def _event_id(
    *,
    configuration_id: str,
    source_snapshot_id: str,
    sequential_decision: str,
    dynamic_decision: str,
    evidence_packet_ids: tuple[str, ...],
    veto_ids: tuple[str, ...],
) -> str:
    raw = "|".join(
        [configuration_id, source_snapshot_id, sequential_decision, dynamic_decision, *sorted(evidence_packet_ids), *sorted(veto_ids)]
    )
    return f"comparison_{sha256(raw.encode()).hexdigest()[:20]}"


def _sequential_status(candidate: CandidateIdentity, root: Path) -> tuple[str, str, str]:
    slug = candidate.pair.lower().replace("/", "").replace("-", "")
    matches = [path for path in (root / "reports" / "active").glob("*copula*after_cost.csv") if slug in path.stem.replace("_", "").lower()]
    if not matches:
        return "FETCH_MORE_DATA", "missing_pair_specific_local_replay", ""
    path = matches[-1]
    frame = pd.read_csv(path)
    if frame.empty:
        return "FETCH_MORE_DATA", "empty_pair_specific_local_replay", str(path.relative_to(root))
    row = frame.iloc[-1]
    acceptance = str(row.get("acceptance", "")).upper()
    if acceptance == "ACCEPT":
        return "TEST", str(row.get("acceptance_reason", "accepted")), str(path.relative_to(root))
    return "REJECT", str(row.get("acceptance_reason", "local_replay_rejected")), str(path.relative_to(root))


def _classification(sequential: str, dynamic: str) -> str:
    if sequential == dynamic:
        return "exact_agreement"
    if sequential in {"REJECT", "FETCH_MORE_DATA"} and dynamic in {"REJECT", "FETCH_MORE_DATA"}:
        return "both_non_promoting"
    return "requires_review"
