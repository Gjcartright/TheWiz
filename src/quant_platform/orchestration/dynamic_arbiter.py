"""Deterministic evidence arbitration for shadow-mode Copula candidates."""

from __future__ import annotations

from hashlib import sha256
from datetime import datetime, timezone
from pathlib import Path

from quant_platform.orchestration.contracts import DecisionBucket, DecisionRecord, EvidencePacket, VetoRecord
from quant_platform.orchestration.dynamic_ledger import DynamicAgentLedger
from quant_platform.orchestration.evidence_integrity import EvidenceIntegrityError, evidence_hash


COPULA_REQUIRED_EVIDENCE = frozenset({"dashboard_snapshot", "reference_check", "local_replay", "venue_check", "cost_risk"})
EVIDENCE_MAX_AGE_SECONDS = {
    "dashboard_snapshot": 2 * 60 * 60 + 30 * 60,
    "venue_check": 2 * 60 * 60 + 30 * 60,
    "local_replay": 24 * 60 * 60,
    "cost_risk": 24 * 60 * 60,
    "reference_check": 30 * 24 * 60 * 60,
}
ROOT = Path(__file__).resolve().parents[3]


def arbitrate_copula_shadow(
    *,
    candidate_id: str,
    evidence: tuple[EvidencePacket, ...],
    vetoes: tuple[VetoRecord, ...],
    ledger: DynamicAgentLedger,
    root: Path = ROOT,
    now: datetime | None = None,
) -> DecisionRecord:
    """Return TEST only for a complete fresh Copula packet; no paper authorization here."""

    now = now or datetime.now(timezone.utc)
    packet_ids = tuple(packet.packet_id for packet in evidence)
    veto_ids = tuple(veto.veto_id for veto in vetoes)
    observed_types = {packet.evidence_type for packet in evidence}
    validation_errors = validate_copula_evidence(candidate_id, evidence, vetoes, root=root, now=now)
    has_hindsight = any(packet.point_in_time_status != "confirmed" for packet in evidence)
    missing = sorted(COPULA_REQUIRED_EVIDENCE - observed_types)
    if validation_errors:
        decision, reason = DecisionBucket.FETCH_MORE_DATA, f"invalid_evidence:{','.join(sorted(validation_errors))}"
    elif vetoes:
        decision, reason = DecisionBucket.FETCH_MORE_DATA, "safety_veto_present"
    elif has_hindsight:
        decision, reason = DecisionBucket.FETCH_MORE_DATA, "point_in_time_evidence_not_confirmed"
    elif missing:
        decision, reason = DecisionBucket.FETCH_MORE_DATA, f"missing_required_evidence:{','.join(missing)}"
    else:
        decision, reason = DecisionBucket.TEST, "complete_copula_shadow_packet_ready_for_local_acceptance"
    token = sha256(f"{candidate_id}|{decision}|{reason}|{'|'.join(packet_ids)}|{'|'.join(veto_ids)}".encode("utf-8")).hexdigest()[:16]
    record = DecisionRecord(
        decision_id=f"decision_{token}",
        candidate_id=candidate_id,
        decision=decision,
        reason=reason,
        evidence_packet_ids=packet_ids or ("missing_evidence",),
        veto_ids=veto_ids,
        created_at=max((item for item in [*(packet.event_timestamp for packet in evidence), *(veto.created_at for veto in vetoes)]), default=now),
    )
    ledger.append_decision(record)
    return record


def validate_copula_evidence(
    candidate_id: str,
    evidence: tuple[EvidencePacket, ...],
    vetoes: tuple[VetoRecord, ...],
    *,
    root: Path,
    now: datetime,
) -> set[str]:
    errors: set[str] = set()
    if len(set(packet.packet_id for packet in evidence)) != len(evidence):
        errors.add("duplicate_packet_id")
    if len(set(packet.evidence_type for packet in evidence)) != len(evidence):
        errors.add("duplicate_evidence_type")
    if any(packet.candidate_id != candidate_id for packet in evidence):
        errors.add("packet_candidate_mismatch")
    if any(veto.candidate_id != candidate_id for veto in vetoes):
        errors.add("veto_candidate_mismatch")
    for packet in evidence:
        maximum_age = EVIDENCE_MAX_AGE_SECONDS.get(packet.evidence_type, 0)
        if maximum_age <= 0 or (now - packet.source_timestamp).total_seconds() > maximum_age:
            errors.add(f"stale_{packet.evidence_type}")
        if packet.source_timestamp > now:
            errors.add(f"future_{packet.evidence_type}")
        if len(packet.evidence_paths) != 1:
            errors.add(f"invalid_path_count_{packet.evidence_type}")
            continue
        try:
            if evidence_hash(root, packet.evidence_paths[0]) != packet.evidence_content_hash:
                errors.add(f"hash_mismatch_{packet.evidence_type}")
        except EvidenceIntegrityError:
            errors.add(f"invalid_path_{packet.evidence_type}")
    return errors
