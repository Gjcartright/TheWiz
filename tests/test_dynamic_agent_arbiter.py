from __future__ import annotations

from datetime import datetime, timedelta, timezone

from quant_platform.orchestration.contracts import CandidateIdentity, EvidencePacket, VetoRecord
from quant_platform.orchestration.dynamic_arbiter import arbitrate_copula_shadow
from quant_platform.orchestration.dynamic_ledger import DynamicAgentLedger
from quant_platform.orchestration.evidence_integrity import evidence_hash


def _candidate() -> CandidateIdentity:
    return CandidateIdentity(pair="ETC-USD/SYRUP-USD", venue="crypto_wizards", strategy_family="Copula", timeframe="2h", lookback=320, formula_version="copula-v1")


def _packet(root, kind: str, *, candidate_id: str | None = None, timestamp: datetime | None = None, mutate_hash: bool = False) -> EvidencePacket:
    now = timestamp or datetime.now(timezone.utc)
    path = root / "reports" / f"{kind}.txt"; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(kind, encoding="utf-8")
    return EvidencePacket(packet_id=f"packet-{kind}", candidate_id=candidate_id or _candidate().candidate_id, producing_agent="copula_agent", evidence_type=kind, event_timestamp=now, source_timestamp=now, point_in_time_status="confirmed", formula_version="copula-v1", test_configuration_hash="hash", evidence_content_hash="0" * 64 if mutate_hash else evidence_hash(root, str(path.relative_to(root))), finding="verified", evidence_paths=(str(path.relative_to(root)),))


def test_arbiter_requires_complete_confirmed_copula_packet(tmp_path):
    evidence = tuple(_packet(tmp_path, kind) for kind in ("dashboard_snapshot", "reference_check", "local_replay", "venue_check", "cost_risk"))
    result = arbitrate_copula_shadow(candidate_id=evidence[0].candidate_id, evidence=evidence, vetoes=(), ledger=DynamicAgentLedger(tmp_path), root=tmp_path)
    assert result.decision == "TEST"


def test_arbiter_rejects_mismatched_candidate_stale_duplicate_and_changed_evidence(tmp_path):
    candidate = _candidate()
    fresh = _packet(tmp_path, "dashboard_snapshot")
    mismatched = _packet(tmp_path, "reference_check", candidate_id="candidate-wrong")
    stale = _packet(tmp_path, "local_replay", timestamp=datetime.now(timezone.utc) - timedelta(days=2))
    bad_hash = _packet(tmp_path, "venue_check", mutate_hash=True)
    duplicate = _packet(tmp_path, "dashboard_snapshot")
    result = arbitrate_copula_shadow(candidate_id=candidate.candidate_id, evidence=(fresh, mismatched, stale, bad_hash, duplicate), vetoes=(), ledger=DynamicAgentLedger(tmp_path), root=tmp_path)
    assert result.decision == "FETCH_MORE_DATA"
    assert "packet_candidate_mismatch" in result.reason
    assert "stale_local_replay" in result.reason
    assert "hash_mismatch_venue_check" in result.reason
    assert "duplicate_packet_id" in result.reason


def test_veto_overrides_complete_packet(tmp_path):
    evidence = (_packet(tmp_path, "dashboard_snapshot"),)
    veto = VetoRecord(veto_id="veto-1", candidate_id=evidence[0].candidate_id, vetoing_agent="red_team_agent", blocker_code="stale_cost", reason="cost stale", evidence_paths=("reports/cost.csv",))
    result = arbitrate_copula_shadow(candidate_id=evidence[0].candidate_id, evidence=evidence, vetoes=(veto,), ledger=DynamicAgentLedger(tmp_path), root=tmp_path)
    assert result.decision == "FETCH_MORE_DATA"
    assert result.reason == "safety_veto_present"
