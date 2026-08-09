"""Artifact-backed, non-executing Copula strategy cell."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path

import pandas as pd

from quant_platform.orchestration.contracts import CandidateIdentity, EvidencePacket
from quant_platform.orchestration.dynamic_arbiter import arbitrate_copula_shadow
from quant_platform.orchestration.dynamic_ledger import DynamicAgentLedger
from quant_platform.orchestration.dynamic_router import RoutingRequest, route_shadow_request
from quant_platform.orchestration.evidence_integrity import evidence_hash
from quant_platform.orchestration.venue_gate import VenuePolicy, evaluate_venue_gate


ROOT = Path(__file__).resolve().parents[3]
CELL_EVIDENCE_VERSION = "copula-shadow-v3-integrity"


def run_copula_shadow_cell(*, candidate: CandidateIdentity, root: Path = ROOT, now: datetime | None = None, venue_policy: VenuePolicy | None = None) -> dict[str, object]:
    """Collect existing Copula artifacts, route shadow tasks, and arbitrate evidence only."""

    now = now or datetime.now(timezone.utc)
    ledger = DynamicAgentLedger(root)
    dashboard_path = root / "reports" / "active" / "wizard_two_hour_copula_report.csv"
    dashboard = _matching_row(dashboard_path, candidate.pair)
    source_timestamp = _timestamp(dashboard, "detail_capture_timestamp_utc", now) if dashboard is not None else now - timedelta(days=365)
    source_paths = (str(dashboard_path.relative_to(root)),)
    route = route_shadow_request(
        RoutingRequest(candidate=candidate, event_type="copula_dislocation", source_timestamp=source_timestamp, requested_at=now, evidence_paths=source_paths),
        ledger=ledger,
    )
    packets = _packets(candidate, root=root, now=now, dashboard=dashboard, dashboard_path=dashboard_path)
    venue_result = evaluate_venue_gate(candidate, policy=venue_policy or VenuePolicy(venue=candidate.venue), root=root)
    vetoes = ()
    if isinstance(venue_result, EvidencePacket):
        packets.append(venue_result)
    else:
        vetoes = (venue_result,)
        ledger.append_veto(venue_result)
    for packet in packets:
        ledger.append_evidence(packet)
    decision = arbitrate_copula_shadow(candidate_id=candidate.candidate_id, evidence=tuple(packets), vetoes=vetoes, ledger=ledger, root=root, now=now)
    return {"route": route, "decision": decision, "evidence_packets": tuple(packets), "shadow_only": True}


def _packets(candidate: CandidateIdentity, *, root: Path, now: datetime, dashboard: pd.Series | None, dashboard_path: Path) -> list[EvidencePacket]:
    rows: list[tuple[str, str, str, pd.Series | None]] = [
        ("dashboard_snapshot", "copula_research_agent", str(dashboard_path.relative_to(root)), dashboard),
        ("reference_check", "copula_reference_agent", "docs/formula_dictionary.md", pd.Series(dtype=object) if (root / "docs" / "formula_dictionary.md").exists() else None),
        ("local_replay", "copula_test_agent", _find_pair_artifact(root, candidate.pair, "*copula*after_cost.csv"), None),
        ("cost_risk", "cost_risk_agent", _find_pair_artifact(root, candidate.pair, "*copula*cost_comparison.csv"), None),
    ]
    packets = []
    for evidence_type, agent, relative_path, row in rows:
        path = root / relative_path
        present = row is not None if evidence_type in {"dashboard_snapshot", "venue_check"} else bool(relative_path) and path.is_file()
        if not present:
            continue
        source_time = _timestamp(row, "detail_capture_timestamp_utc", _artifact_timestamp(path, now)) if row is not None else _artifact_timestamp(path, now)
        token = sha256(f"{CELL_EVIDENCE_VERSION}|{candidate.candidate_id}|{evidence_type}|{relative_path}|{source_time.isoformat()}".encode()).hexdigest()[:16]
        packets.append(EvidencePacket(packet_id=f"packet_{token}", candidate_id=candidate.candidate_id, producing_agent=agent, evidence_type=evidence_type, event_timestamp=source_time, source_timestamp=source_time, point_in_time_status="confirmed" if source_time <= now else "unknown", formula_version=candidate.formula_version, test_configuration_hash=token, evidence_content_hash=evidence_hash(root, relative_path), finding=f"{evidence_type}_artifact_present", evidence_paths=(relative_path,)))
    return packets


def _matching_row(path: Path, pair: str) -> pd.Series | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    if frame.empty or "pair" not in frame:
        return None
    rows = frame[frame["pair"].astype(str).str.upper().eq(pair.upper())]
    return rows.iloc[-1] if not rows.empty else None


def _find_pair_artifact(root: Path, pair: str, pattern: str) -> str:
    slug = pair.lower().replace("/", "").replace("-", "")
    matches = [path for path in (root / "reports" / "active").glob(pattern) if slug in path.stem.replace("_", "").lower()]
    return str(matches[-1].relative_to(root)) if matches else ""


def _timestamp(row: pd.Series | None, column: str, fallback: datetime) -> datetime:
    if row is None or column not in row or not str(row.get(column, "") or ""):
        return fallback
    value = pd.to_datetime(row[column], utc=True, errors="coerce")
    return value.to_pydatetime() if not pd.isna(value) else fallback


def _artifact_timestamp(path: Path, fallback: datetime) -> datetime:
    if not path.is_file():
        return fallback
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
