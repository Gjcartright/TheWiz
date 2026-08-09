"""Read-only venue and account capability gate for dynamic-agent research."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import pandas as pd

from quant_platform.orchestration.contracts import CandidateIdentity, EvidencePacket, VetoRecord
from quant_platform.orchestration.evidence_integrity import evidence_hash


ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class VenuePolicy:
    venue: str
    account_eligible: bool = False
    product_type: str = "unknown"
    supports_short_leg: bool = False
    capability_status: str = "missing"
    capability_evidence_path: str = ""


def evaluate_venue_gate(candidate: CandidateIdentity, *, policy: VenuePolicy, root: Path = ROOT) -> EvidencePacket | VetoRecord:
    """Return confirmation evidence or a veto. This function never submits orders."""

    now = datetime.now(timezone.utc)
    venue = policy.venue.lower()
    if policy.capability_status != "verified":
        return _veto(candidate, f"venue_capability_{policy.capability_status}", "venue capability evidence is missing, invalid, or stale", policy.capability_evidence_path, now)
    if not policy.account_eligible:
        return _veto(candidate, "account_eligibility_unconfirmed", "account eligibility is not verified", "", now)
    if policy.product_type == "spot" and not policy.supports_short_leg:
        return _veto(candidate, "spot_short_leg_unavailable", "spot venue cannot support the required short leg", "", now)
    path, confirmed = _compatibility(candidate, venue, root)
    if not path:
        return _veto(candidate, "venue_compatibility_missing", "no compatibility source for venue", "", now)
    if not confirmed:
        return _veto(candidate, "both_legs_not_confirmed", "both legs are not confirmed for the requested route", path, now)
    source_timestamp = datetime.fromtimestamp((root / path).stat().st_mtime, tz=timezone.utc)
    content_hash = evidence_hash(root, path)
    token = sha256(f"{candidate.candidate_id}|{venue}|{path}|{content_hash}|{source_timestamp.isoformat()}".encode()).hexdigest()[:16]
    return EvidencePacket(packet_id=f"packet_{token}", candidate_id=candidate.candidate_id, producing_agent="venue_evidence_agent", evidence_type="venue_check", event_timestamp=source_timestamp, source_timestamp=source_timestamp, point_in_time_status="confirmed", formula_version=candidate.formula_version, test_configuration_hash=token, evidence_content_hash=content_hash, finding="both_legs_confirmed_for_research_route", confidence_band="high", evidence_paths=(path,))


def _compatibility(candidate: CandidateIdentity, venue: str, root: Path) -> tuple[str, bool]:
    active = root / "reports" / "active"
    if venue == "dydx":
        path = active / "dydx_execution_market_compatibility.csv"
        if not path.exists(): return "", False
        frame = pd.read_csv(path)
        markets = {str(value).upper() for value in frame.loc[frame.get("compatible_for_paper_submit", False).astype(bool), "market"]}
        return str(path.relative_to(root)), set(candidate.pair.split("/")) <= markets
    if venue == "hyperliquid":
        path = active / "hyperliquid_execution_market_compatibility.csv"
        if not path.exists(): return "", False
        frame = pd.read_csv(path)
        pair = candidate.pair.replace("/", "-")
        rows = frame[frame.get("pair", pd.Series(dtype=str)).astype(str).str.upper().eq(pair)]
        return str(path.relative_to(root)), bool(not rows.empty and rows.iloc[-1].get("mirrorable_for_paper", False))
    return "", False


def _veto(candidate: CandidateIdentity, blocker: str, reason: str, path: str, now: datetime) -> VetoRecord:
    token = sha256(f"{candidate.candidate_id}|{blocker}|{path}|{now.isoformat()}".encode()).hexdigest()[:16]
    return VetoRecord(veto_id=f"veto_{token}", candidate_id=candidate.candidate_id, vetoing_agent="venue_evidence_agent", blocker_code=blocker, reason=reason, evidence_paths=(path or "venue_policy",), created_at=now)
