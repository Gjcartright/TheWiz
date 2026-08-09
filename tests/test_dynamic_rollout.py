from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from quant_platform.orchestration.contracts import ComparisonEvent, EvidencePacket
from quant_platform.orchestration.dynamic_ledger import DynamicAgentLedger
from quant_platform.orchestration.dynamic_rollout import build_dynamic_rollout_gate
from quant_platform.orchestration.evidence_integrity import evidence_hash


NOW = datetime(2026, 7, 24, 12, tzinfo=timezone.utc)


def _event(index: int, *, requires_review: bool = False, stale: bool = False, vetoes: int = 0, configuration_id: str | None = None) -> ComparisonEvent:
    timestamp = NOW - timedelta(hours=3) if stale else NOW - timedelta(minutes=index)
    return ComparisonEvent(
        event_id=f"comparison-{index}-{requires_review}-{stale}-{vetoes}",
        candidate_id=f"candidate-{index}",
        configuration_id=configuration_id or f"candidate-{index}",
        source_snapshot_id=f"snapshot-{index}",
        pair=f"ASSET{index}-USD/OTHER{index}-USD",
        venue="dydx",
        timeframe="1d",
        source_timestamp=timestamp,
        sequential_decision="TEST",
        sequential_reason="accepted",
        dynamic_decision="TEST",
        dynamic_reason="complete",
        dynamic_veto_count=vetoes,
        dynamic_evidence_count=5,
        evidence_packet_ids=tuple(f"packet-{index}-{kind}" for kind in range(5)),
        comparison="requires_review" if requires_review else "exact_agreement",
        created_at=timestamp,
    )


def _write_events(tmp_path, events: list[ComparisonEvent]) -> None:
    ledger = DynamicAgentLedger(tmp_path)
    for event in events:
        for packet_id, evidence_type in zip(event.evidence_packet_ids, ("dashboard_snapshot", "reference_check", "local_replay", "venue_check", "cost_risk"), strict=True):
            path = tmp_path / "reports" / "evidence" / f"{packet_id}.txt"; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(packet_id, encoding="utf-8")
            relative_path = str(path.relative_to(tmp_path))
            ledger.append_evidence(EvidencePacket(packet_id=packet_id, candidate_id=event.candidate_id, producing_agent="test_agent", evidence_type=evidence_type, event_timestamp=event.source_timestamp, source_timestamp=event.source_timestamp, point_in_time_status="confirmed", formula_version="copula-v1", test_configuration_hash=f"test-{packet_id}", evidence_content_hash=evidence_hash(tmp_path, relative_path), finding="test evidence", evidence_paths=(relative_path,)))
        assert ledger.append_comparison_event(event)


def test_other_cells_stay_blocked_until_five_complete_copula_agreements(tmp_path):
    _write_events(tmp_path, [_event(index) for index in range(4)])
    result = build_dynamic_rollout_gate(root=tmp_path, now=NOW)
    frame = pd.read_csv(result["csv"])
    assert result["copula_rollout_status"] == "blocked"
    assert set(frame.loc[frame["strategy_family"] != "Copula", "rollout_status"]) == {"BLOCKED_PENDING_COPULA_SHADOW_ACCEPTANCE"}
    assert not frame["promotion_allowed"].any()
    assert not frame["paper_execution_allowed"].any()


def test_only_unique_fresh_clean_events_open_other_cells_for_shadow_mode(tmp_path):
    _write_events(tmp_path, [_event(index) for index in range(5)])
    result = build_dynamic_rollout_gate(root=tmp_path, now=NOW)
    frame = pd.read_csv(result["csv"])
    assert result["copula_rollout_status"] == "ready"
    assert frame.loc[frame["strategy_family"] == "Copula", "rollout_status"].item() == "ELIGIBLE_FOR_NEXT_SHADOW_CELL"
    assert set(frame.loc[frame["strategy_family"] != "Copula", "rollout_status"]) == {"ELIGIBLE_FOR_SHADOW_ONLY"}
    assert not frame["promotion_allowed"].any()
    assert not frame["paper_execution_allowed"].any()


def test_duplicate_configuration_stale_or_reviewed_events_cannot_unlock_rollout(tmp_path):
    events = [_event(index) for index in range(3)]
    events.extend([_event(3, configuration_id="candidate-0"), _event(4, stale=True), _event(5, requires_review=True), _event(6, vetoes=1)])
    _write_events(tmp_path, events)
    result = build_dynamic_rollout_gate(root=tmp_path, now=NOW)
    frame = pd.read_csv(result["csv"])
    assert result["copula_rollout_status"] == "blocked"
    assert "copula_shadow_requires_review" in frame.loc[frame["strategy_family"] == "Copula", "reason"].item()


def test_changed_evidence_file_revokes_an_otherwise_qualifying_event(tmp_path):
    events = [_event(index) for index in range(5)]
    _write_events(tmp_path, events)
    path = tmp_path / "reports" / "evidence" / f"{events[0].evidence_packet_ids[0]}.txt"
    path.write_text("tampered", encoding="utf-8")
    result = build_dynamic_rollout_gate(root=tmp_path, now=NOW)
    assert result["copula_rollout_status"] == "blocked"
    assert result["qualifying_agreements"] == 4
