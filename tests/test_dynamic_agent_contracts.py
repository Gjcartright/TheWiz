from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from quant_platform.orchestration.contracts import (
    AgentCapability,
    CandidateIdentity,
    DecisionBucket,
    DecisionRecord,
    EvidencePacket,
    normalize_pair,
)


def _candidate() -> CandidateIdentity:
    return CandidateIdentity(
        pair=" sol-usd / wld-usd ",
        venue="crypto_wizards",
        strategy_family="OU ZScoreR",
        timeframe="1d",
        lookback=320,
        formula_version="wizard-ou-v1",
    )


def test_candidate_identity_normalizes_pair_and_is_stable():
    first = _candidate()
    second = _candidate()

    assert first.pair == "SOL-USD/WLD-USD"
    assert first.candidate_id == second.candidate_id
    assert normalize_pair("BTC-USD/ETH-USD") == "BTC-USD/ETH-USD"


def test_candidate_identity_rejects_wrong_or_ambiguous_pair_identity():
    with pytest.raises(ValidationError):
        CandidateIdentity(
            pair="BTC-USD-ETH-USD",
            venue="crypto_wizards",
            strategy_family="Copula",
            timeframe="1d",
            lookback=320,
            formula_version="copula-v1",
        )


def test_evidence_packet_requires_point_in_time_lineage():
    now = datetime.now(timezone.utc)
    packet = EvidencePacket(
        packet_id="evidence-1",
        candidate_id=_candidate().candidate_id,
        producing_agent="copula_test_agent",
        evidence_type="local_replay",
        event_timestamp=now,
        source_timestamp=now,
        point_in_time_status="confirmed",
        formula_version="copula-v1",
        test_configuration_hash="abc123",
        evidence_content_hash="a" * 64,
        finding="after-cost replay completed",
        evidence_paths=("reports/active/copula_replay.csv",),
    )

    assert packet.point_in_time_status == "confirmed"
    with pytest.raises(ValidationError):
        EvidencePacket(
            packet_id="evidence-2",
            candidate_id=packet.candidate_id,
            producing_agent="copula_test_agent",
            evidence_type="local_replay",
            event_timestamp=datetime.now(),
            source_timestamp=now,
            point_in_time_status="confirmed",
            formula_version="copula-v1",
            test_configuration_hash="abc123",
            evidence_content_hash="a" * 64,
            finding="bad timestamp",
            evidence_paths=("reports/active/copula_replay.csv",),
        )


def test_vetoes_and_agent_capabilities_cannot_authorize_execution():
    with pytest.raises(ValidationError):
        DecisionRecord(
            decision_id="decision-1",
            candidate_id=_candidate().candidate_id,
            decision=DecisionBucket.PAPER_AUTHORIZED,
            reason="not allowed with a veto",
            evidence_packet_ids=("evidence-1",),
            veto_ids=("veto-1",),
        )
    with pytest.raises(ValidationError, match="cannot authorize paper"):
        DecisionRecord(
            decision_id="decision-2",
            candidate_id=_candidate().candidate_id,
            decision=DecisionBucket.PAPER_AUTHORIZED,
            reason="not permitted for dynamic agents",
            evidence_packet_ids=("evidence-1",),
        )
    with pytest.raises(ValidationError):
        AgentCapability(agent="venue_agent", can_submit_orders=True)
