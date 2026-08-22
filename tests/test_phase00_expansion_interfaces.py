from __future__ import annotations

import ast
import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from quant_platform.orchestration.corrective_expansion_interfaces import (
    DEFAULT_INSTRUMENT_SOURCE_PRECEDENCE,
    EXPANSION_CONTRACT_VERSION,
    EXPANSION_INTERFACE_SCHEMA_VERSION,
    IDENTITY_BINDINGS,
    INSTRUMENT_CONSENSUS_FIELDS,
    LEGAL_ONBOARDING_TRANSITIONS,
    LEGAL_TWO_LEG_TRANSITIONS,
    TERMINAL_TWO_LEG_STATES,
    CompleteHeldOutValidationReceipt,
    EvidenceLane,
    ExecutionLegContract,
    HeldOutDimension,
    HeldOutDimensionReceipt,
    HeldOutMetrics,
    InstrumentReconciliationPolicy,
    InstrumentReconciliationRequest,
    InstrumentSourceObservation,
    OnboardingState,
    OnboardingTransitionRequest,
    OutcomeLabel,
    OutcomeLabelPartition,
    ProposalCapabilityContract,
    ProposalCapabilityRegistry,
    ProposalCapabilityRole,
    ProposalOnlyReceipt,
    ProposalOutputType,
    RecoveryAuthorityReference,
    TwoLegExecutionJournal,
    TwoLegExecutionPlan,
    TwoLegJournalEvent,
    TwoLegState,
    UnknownExchangeOnboardingRecord,
    VenueCostEvidence,
    VenueSelectorCandidate,
    VenueSelectorInput,
    evaluate_onboarding_transition,
    reconcile_instrument,
    select_venue,
    validate_two_leg_journal,
)
from quant_platform.orchestration.identity_ontology import (
    IDENTITY_ONTOLOGY_VERSION,
    AccountScopeIdentity,
    VenueIdentity,
    VenueInstrumentIdentity,
    VenueProductLaneIdentity,
)
from quant_platform.orchestration.venue_policy_registry import (
    VENUE_POLICY_SCHEMA_VERSION,
    VenueLane,
)

HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
NOW = datetime(2026, 8, 21, 12, tzinfo=UTC)


def _instrument(
    *,
    venue_code: str = "hyperliquid",
    lane: str = "hyperliquid_perp",
    market: str = "BTC",
    symbol: str = "BTC/USDC:USDC",
    base: str = "asset_btc",
    quote: str = "asset_usdc",
) -> VenueInstrumentIdentity:
    venue_id = VenueIdentity(venue_code=venue_code).venue_id
    settle_asset_id = quote
    return VenueInstrumentIdentity(
        venue_id=venue_id,
        product_lane_id=VenueProductLaneIdentity(
            venue_id=venue_id,
            product_lane=lane,
            settlement_asset_id=settle_asset_id,
        ).product_lane_id,
        venue_market_id=market,
        unified_symbol=symbol,
        base_asset_id=base,
        quote_asset_id=quote,
        settle_asset_id=settle_asset_id,
    )


def _account(
    *, venue_code: str = "hyperliquid", environment: str = "testnet"
) -> AccountScopeIdentity:
    return AccountScopeIdentity(
        venue_id=VenueIdentity(venue_code=venue_code).venue_id,
        environment=environment,
        account_id_hash=HASH_A,
    )


def _costs(*, total: str = "9", observed_at: datetime = NOW) -> VenueCostEvidence:
    return VenueCostEvidence(
        observed_at=observed_at,
        freshness_deadline=observed_at + timedelta(minutes=5),
        source_snapshot_hash=HASH_B,
        fee_bps=Decimal(2),
        slippage_bps=Decimal(3),
        funding_bps=Decimal(1),
        borrow_short_bps=Decimal(1),
        network_bps=Decimal(1),
        execution_risk_bps=Decimal(1),
        all_in_cost_bps=Decimal(total),
    )


def _selector_candidate(
    *,
    lane: VenueLane = VenueLane.HYPERLIQUID_PERP,
    instrument: VenueInstrumentIdentity | None = None,
    account: AccountScopeIdentity | None = None,
    total: str = "9",
    observed_at: datetime = NOW,
) -> VenueSelectorCandidate:
    return VenueSelectorCandidate(
        lane=lane,
        instrument=instrument or _instrument(),
        account_scope=account or _account(),
        observed_at=observed_at,
        freshness_deadline=observed_at + timedelta(minutes=5),
        market_snapshot_hash=HASH_A,
        costs=_costs(total=total, observed_at=observed_at),
    )


def _held_out_dimension(dimension: HeldOutDimension) -> HeldOutDimensionReceipt:
    temporal = dimension == HeldOutDimension.TIME
    return HeldOutDimensionReceipt(
        dimension=dimension,
        source_dataset_id="dataset_v2",
        split_snapshot_hash=HASH_C,
        training_members=(f"train_{dimension.value}",),
        held_out_members=(f"holdout_{dimension.value}",),
        training_sample_count=100,
        held_out_sample_count=25,
        training_end=NOW if temporal else None,
        held_out_start=NOW + timedelta(days=1) if temporal else None,
        evaluated_at=NOW + timedelta(days=2),
        metrics=HeldOutMetrics(
            trade_count=25,
            profit_factor="1.2",
            sharpe="0.8",
            max_drawdown="0.1",
            expectancy="0.02",
        ),
    )


def _plan() -> TwoLegExecutionPlan:
    leg_a = ExecutionLegContract(
        leg_name="leg_a",
        instrument=_instrument(),
        account_scope=_account(),
        side="buy",
        quantity="1",
        client_order_id="leg-a-client",
        idempotency_key="leg-a-key",
    )
    leg_b = ExecutionLegContract(
        leg_name="leg_b",
        instrument=_instrument(
            market="ETH",
            symbol="ETH/USDC:USDC",
            base="asset_eth",
        ),
        account_scope=_account(),
        side="sell",
        quantity="2",
        client_order_id="leg-b-client",
        idempotency_key="leg-b-key",
    )
    return TwoLegExecutionPlan(
        proposal_id="proposal-1",
        evidence_lane=EvidenceLane.TESTNET,
        created_at=NOW,
        leg_a=leg_a,
        leg_b=leg_b,
    )


def _event(
    sequence: int,
    from_state: TwoLegState,
    to_state: TwoLegState,
    *,
    authority: RecoveryAuthorityReference | None = None,
) -> TwoLegJournalEvent:
    return TwoLegJournalEvent(
        sequence=sequence,
        event_id=f"event-{sequence}",
        idempotency_key=f"event-key-{sequence}",
        occurred_at=NOW + timedelta(seconds=sequence),
        from_state=from_state,
        to_state=to_state,
        recovery_authority=authority,
    )


def test_config_matches_immutable_contract_constants() -> None:
    config_path = Path(__file__).parents[1] / "config" / "phase00_expansion_interfaces.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == EXPANSION_INTERFACE_SCHEMA_VERSION
    assert payload["contract_version"] == EXPANSION_CONTRACT_VERSION
    assert payload["identity_bindings"] == IDENTITY_BINDINGS
    assert tuple(payload["instrument_registry"]["source_precedence"]) == (
        DEFAULT_INSTRUMENT_SOURCE_PRECEDENCE
    )
    assert tuple(payload["instrument_registry"]["consensus_fields"]) == (
        INSTRUMENT_CONSENSUS_FIELDS
    )
    assert payload["identity_bindings"]["identity_ontology_version"] == (IDENTITY_ONTOLOGY_VERSION)
    assert payload["identity_bindings"]["venue_policy_schema_version"] == (
        VENUE_POLICY_SCHEMA_VERSION
    )
    assert {
        state: sorted(destinations)
        for state, destinations in payload["unknown_exchange_onboarding"][
            "legal_transitions"
        ].items()
    } == {
        state.value: sorted(destination.value for destination in destinations)
        for state, destinations in LEGAL_ONBOARDING_TRANSITIONS.items()
    }
    assert {
        state: sorted(destinations)
        for state, destinations in payload["two_leg_journal"]["legal_transitions"].items()
    } == {
        state.value: sorted(destination.value for destination in destinations)
        for state, destinations in LEGAL_TWO_LEG_TRANSITIONS.items()
    }
    assert payload["phase00_effect_policy"]["pure_validators_only"] is True
    assert all(
        value is False
        for key, value in payload["phase00_effect_policy"].items()
        if key != "pure_validators_only"
    )


def test_contract_module_has_no_effectful_imports_or_authority_surface() -> None:
    source_path = (
        Path(__file__).parents[1]
        / "src"
        / "quant_platform"
        / "orchestration"
        / "corrective_expansion_interfaces.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", maxsplit=1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", maxsplit=1)[0])
    function_names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    assert imported_roots.isdisjoint(
        {"ccxt", "httpx", "os", "requests", "socket", "subprocess", "urllib"}
    )
    assert function_names.isdisjoint(
        {"create_order", "mint_permit", "publish", "run_scheduler", "submit_order"}
    )


def test_reconciliation_uses_precedence_only_after_full_identity_consensus() -> None:
    instrument = _instrument()
    observations = tuple(
        InstrumentSourceObservation(
            source_system=source,
            observed_at=NOW,
            freshness_deadline=NOW + timedelta(minutes=5),
            source_snapshot_hash=hash_value,
            instrument=instrument,
        )
        for source, hash_value in (
            ("crypto_wizards", HASH_A),
            ("venue_public_api", HASH_B),
        )
    )
    receipt = reconcile_instrument(
        InstrumentReconciliationRequest(
            policy=InstrumentReconciliationPolicy(
                policy_id="canonical-instrument-v1",
                source_precedence=DEFAULT_INSTRUMENT_SOURCE_PRECEDENCE,
            ),
            as_of=NOW + timedelta(minutes=1),
            observations=observations,
        )
    )

    assert receipt.decision == "RECONCILED"
    assert receipt.authoritative_source == "venue_public_api"
    assert receipt.canonical_instrument == instrument
    assert receipt.blockers == ()


def test_reconciliation_abstains_on_conflicting_identity() -> None:
    observations = (
        InstrumentSourceObservation(
            source_system="venue_public_api",
            observed_at=NOW,
            freshness_deadline=NOW + timedelta(minutes=5),
            source_snapshot_hash=HASH_A,
            instrument=_instrument(),
        ),
        InstrumentSourceObservation(
            source_system="ccxt_public",
            observed_at=NOW,
            freshness_deadline=NOW + timedelta(minutes=5),
            source_snapshot_hash=HASH_B,
            instrument=_instrument(market="BTC-PERP"),
        ),
    )
    receipt = reconcile_instrument(
        InstrumentReconciliationRequest(
            policy=InstrumentReconciliationPolicy(
                policy_id="canonical-instrument-v1",
                source_precedence=DEFAULT_INSTRUMENT_SOURCE_PRECEDENCE,
            ),
            as_of=NOW,
            observations=observations,
        )
    )

    assert receipt.decision == "ABSTAIN"
    assert any(blocker.startswith("identity_conflict:") for blocker in receipt.blockers)


def test_malformed_identity_is_rejected_before_selection() -> None:
    wrong_venue_instrument = _instrument(venue_code="dydx", lane="dydx_perp")

    with pytest.raises(ValidationError, match="instrument venue identity does not match lane"):
        _selector_candidate(instrument=wrong_venue_instrument)


def test_selector_abstains_when_any_evidence_is_stale() -> None:
    stale = _selector_candidate(observed_at=NOW - timedelta(minutes=10))
    receipt = select_venue(
        VenueSelectorInput(
            selection_id="selection-stale",
            evidence_lane=EvidenceLane.TESTNET,
            as_of=NOW,
            candidates=(stale,),
        )
    )

    assert receipt.decision == "ABSTAIN"
    assert any(blocker.startswith("stale_") for blocker in receipt.blockers)


def test_selector_abstains_on_duplicate_route_ambiguity() -> None:
    first = _selector_candidate(total="9")
    second = _selector_candidate(total="9")
    receipt = select_venue(
        VenueSelectorInput(
            selection_id="selection-ambiguous",
            evidence_lane=EvidenceLane.TESTNET,
            as_of=NOW,
            candidates=(first, second),
        )
    )

    assert receipt.decision == "ABSTAIN"
    assert receipt.blockers == (f"ambiguous_duplicate_route:{first.instrument.instrument_id}",)


def test_selector_tie_break_is_stable_under_input_reordering() -> None:
    hyperliquid = _selector_candidate(
        account=_account(environment="research"),
        total="9",
    )
    dydx = _selector_candidate(
        lane=VenueLane.DYDX_PERP,
        instrument=_instrument(
            venue_code="dydx",
            lane="dydx_perp",
            market="BTC-USD",
        ),
        account=_account(venue_code="dydx", environment="research"),
        total="9",
    )
    forward = select_venue(
        VenueSelectorInput(
            selection_id="stable-tie",
            evidence_lane=EvidenceLane.SHADOW,
            as_of=NOW,
            candidates=(hyperliquid, dydx),
        )
    )
    reverse = select_venue(
        VenueSelectorInput(
            selection_id="stable-tie",
            evidence_lane=EvidenceLane.SHADOW,
            as_of=NOW,
            candidates=(dydx, hyperliquid),
        )
    )

    assert forward.decision == reverse.decision == "SELECT"
    assert forward.selected_candidate_id == reverse.selected_candidate_id
    assert forward.ranked_candidate_ids == reverse.ranked_candidate_ids


def test_selector_rejects_missing_or_incorrect_cost_components() -> None:
    payload = _costs().model_dump()
    payload.pop("slippage_bps")
    with pytest.raises(ValidationError, match="slippage_bps"):
        VenueCostEvidence.model_validate(payload)


def test_selector_respects_closed_testnet_and_live_venue_policy() -> None:
    dydx = _selector_candidate(
        lane=VenueLane.DYDX_PERP,
        instrument=_instrument(
            venue_code="dydx",
            lane="dydx_perp",
            market="BTC-USD",
        ),
        account=_account(venue_code="dydx"),
    )
    testnet_receipt = select_venue(
        VenueSelectorInput(
            selection_id="closed-testnet-lane",
            evidence_lane=EvidenceLane.TESTNET,
            as_of=NOW,
            candidates=(dydx,),
        )
    )
    assert testnet_receipt.decision == "ABSTAIN"
    assert testnet_receipt.blockers == ("lane_not_testnet_eligible:dydx_perp",)

    live = _selector_candidate(account=_account(environment="live"))
    live_receipt = select_venue(
        VenueSelectorInput(
            selection_id="closed-live-lane",
            evidence_lane=EvidenceLane.LIVE,
            as_of=NOW,
            candidates=(live,),
        )
    )
    assert live_receipt.decision == "ABSTAIN"
    assert live_receipt.blockers == ("phase00_effect_lane_forbidden:live",)

    payload = _costs().model_dump()
    payload["all_in_cost_bps"] = Decimal(0)
    with pytest.raises(ValidationError, match="complete cost formula"):
        VenueCostEvidence.model_validate(payload)


def test_unknown_exchange_onboarding_rejects_illegal_skip() -> None:
    record = UnknownExchangeOnboardingRecord(
        venue=VenueIdentity(venue_code="newdex"),
        discovered_at=NOW,
        source_snapshot_hash=HASH_A,
    )
    receipt = evaluate_onboarding_transition(
        OnboardingTransitionRequest(
            record=record,
            onboarding_id=record.onboarding_id,
            from_state=OnboardingState.DISCOVERED,
            to_state=OnboardingState.TESTNET_ELIGIBLE,
            requested_at=NOW,
            evidence_hash=HASH_B,
        )
    )

    assert not receipt.allowed
    assert receipt.resulting_state == OnboardingState.DISCOVERED
    assert receipt.blockers == ("illegal_onboarding_transition",)


def test_unknown_onboarding_refuses_known_venue_and_terminal_transitions() -> None:
    with pytest.raises(ValidationError, match="known Phase 00 venue"):
        UnknownExchangeOnboardingRecord(
            venue=VenueIdentity(venue_code="hyperliquid"),
            discovered_at=NOW,
            source_snapshot_hash=HASH_A,
        )

    assert LEGAL_ONBOARDING_TRANSITIONS[OnboardingState.REJECTED] == frozenset()
    assert LEGAL_ONBOARDING_TRANSITIONS[OnboardingState.TESTNET_ELIGIBLE] == frozenset()


def test_onboarding_transition_cannot_claim_a_false_current_state() -> None:
    record = UnknownExchangeOnboardingRecord(
        venue=VenueIdentity(venue_code="newdex"),
        discovered_at=NOW,
        source_snapshot_hash=HASH_A,
    )
    with pytest.raises(ValidationError, match="from_state does not match"):
        OnboardingTransitionRequest(
            record=record,
            onboarding_id=record.onboarding_id,
            from_state=OnboardingState.EXECUTION_CONTRACT_VERIFIED,
            to_state=OnboardingState.TESTNET_ELIGIBLE,
            requested_at=NOW,
            evidence_hash=HASH_B,
        )


def test_partial_leg_recovery_without_authority_is_invalid() -> None:
    plan = _plan()
    journal = TwoLegExecutionJournal(
        plan=plan,
        events=(
            _event(1, TwoLegState.PROPOSED, TwoLegState.AUTHORITY_CONFIRMED),
            _event(2, TwoLegState.AUTHORITY_CONFIRMED, TwoLegState.FIRST_LEG_PENDING),
            _event(3, TwoLegState.FIRST_LEG_PENDING, TwoLegState.FIRST_LEG_CONFIRMED),
            _event(4, TwoLegState.FIRST_LEG_CONFIRMED, TwoLegState.RECOVERY_REQUIRED),
            _event(5, TwoLegState.RECOVERY_REQUIRED, TwoLegState.RECOVERY_AUTHORIZED),
        ),
    )
    receipt = validate_two_leg_journal(journal, as_of=NOW + timedelta(minutes=1))

    assert not receipt.valid
    assert "recovery_authority_missing:event-5" in receipt.blockers


def test_two_leg_plan_rejects_label_and_account_environment_mixing() -> None:
    payload = _plan().model_dump()
    payload["evidence_lane"] = EvidenceLane.SHADOW
    payload["plan_id"] = ""

    with pytest.raises(ValidationError, match="environments do not match evidence lane"):
        TwoLegExecutionPlan.model_validate(payload)


def test_trusted_recovery_path_is_valid_and_terminal() -> None:
    plan = _plan()
    authority = RecoveryAuthorityReference(
        authority_receipt_id="trusted-recovery-receipt",
        authority_receipt_hash=HASH_A,
        permit_id="external-permit-reference",
        plan_id=plan.plan_id,
        allowed_action="flatten_filled_leg",
        scope_hash=HASH_C,
        issued_at=NOW,
        expires_at=NOW + timedelta(minutes=5),
    )
    journal = TwoLegExecutionJournal(
        plan=plan,
        events=(
            _event(1, TwoLegState.PROPOSED, TwoLegState.AUTHORITY_CONFIRMED),
            _event(2, TwoLegState.AUTHORITY_CONFIRMED, TwoLegState.FIRST_LEG_PENDING),
            _event(3, TwoLegState.FIRST_LEG_PENDING, TwoLegState.RECOVERY_REQUIRED),
            _event(
                4,
                TwoLegState.RECOVERY_REQUIRED,
                TwoLegState.RECOVERY_AUTHORIZED,
                authority=authority,
            ),
            _event(
                5,
                TwoLegState.RECOVERY_AUTHORIZED,
                TwoLegState.RECOVERY_PENDING,
                authority=authority,
            ),
            _event(
                6,
                TwoLegState.RECOVERY_PENDING,
                TwoLegState.RECOVERED,
                authority=authority,
            ),
        ),
    )
    receipt = validate_two_leg_journal(
        journal,
        as_of=NOW + timedelta(minutes=1),
        trusted_recovery_authority_receipt_ids=frozenset({"trusted-recovery-receipt"}),
        trusted_recovery_authority_receipt_hashes=frozenset({HASH_A}),
    )

    assert receipt.valid
    assert receipt.terminal
    assert receipt.final_state == TwoLegState.RECOVERED


def test_two_leg_journal_rejects_idempotency_reuse_and_terminal_transition() -> None:
    events = (
        _event(1, TwoLegState.PROPOSED, TwoLegState.ABORTED),
        TwoLegJournalEvent(
            sequence=2,
            event_id="event-2",
            idempotency_key="event-key-1",
            occurred_at=NOW + timedelta(seconds=2),
            from_state=TwoLegState.ABORTED,
            to_state=TwoLegState.COMPLETED,
        ),
    )
    receipt = validate_two_leg_journal(
        TwoLegExecutionJournal(plan=_plan(), events=events),
        as_of=NOW + timedelta(minutes=1),
    )

    assert not receipt.valid
    assert "duplicate_idempotency_key:event-key-1" in receipt.blockers
    assert "transition_after_terminal:aborted" in receipt.blockers
    assert TwoLegState.ABORTED in TERMINAL_TWO_LEG_STATES


@pytest.mark.parametrize("role", list(ProposalCapabilityRole))
def test_all_learning_and_specialist_roles_are_proposal_only(
    role: ProposalCapabilityRole,
) -> None:
    contract = ProposalCapabilityContract(
        role=role,
        capability_name=f"{role.value}-v1",
        specialist_domain="ou_spread" if role == ProposalCapabilityRole.SPECIALIST else None,
        allowed_outputs=(ProposalOutputType.PROPOSAL, ProposalOutputType.EVALUATION),
    )

    assert contract.execution_authority is False
    assert contract.can_mint_permits is False
    assert contract.can_submit_orders is False
    assert contract.can_publish_live_evidence is False


def test_proposal_contract_rejects_authority_escalation_and_permit_injection() -> None:
    with pytest.raises(ValidationError, match="execution_authority"):
        ProposalCapabilityContract(
            role="rl",
            capability_name="rl-v1",
            allowed_outputs=("proposal",),
            execution_authority=True,
        )

    contract = ProposalCapabilityContract(
        role="teacher",
        capability_name="teacher-v1",
        allowed_outputs=("hypothesis",),
    )
    with pytest.raises(ValidationError, match="permit_id"):
        ProposalOnlyReceipt(
            capability=contract,
            output_type="hypothesis",
            evidence_lane="shadow",
            input_snapshot_hash=HASH_A,
            proposal_payload_hash=HASH_B,
            produced_at=NOW,
            permit_id="forged-permit",
        )

    with pytest.raises(ValidationError, match="not allowed by capability"):
        ProposalOnlyReceipt(
            capability=contract,
            output_type="score",
            evidence_lane="shadow",
            input_snapshot_hash=HASH_A,
            proposal_payload_hash=HASH_B,
            produced_at=NOW,
        )


def test_capability_registry_requires_every_proposal_role() -> None:
    contracts = tuple(
        ProposalCapabilityContract(
            role=role,
            capability_name=f"{role.value}-v1",
            specialist_domain="copula" if role == ProposalCapabilityRole.SPECIALIST else None,
            allowed_outputs=(ProposalOutputType.PROPOSAL,),
        )
        for role in ProposalCapabilityRole
    )
    assert len(ProposalCapabilityRegistry(contracts=contracts).contracts) == 6

    with pytest.raises(ValidationError, match="at least 6|missing roles"):
        ProposalCapabilityRegistry(contracts=contracts[:-1])


def test_outcome_label_partitions_reject_cross_lane_mixing() -> None:
    backtest = OutcomeLabel(
        outcome_id="outcome-backtest",
        evidence_lane=EvidenceLane.BACKTEST,
        source_run_id="run-backtest",
        observed_at=NOW,
        outcome_hash=HASH_A,
    )
    live = OutcomeLabel(
        outcome_id="outcome-live",
        evidence_lane=EvidenceLane.LIVE,
        source_run_id="run-live",
        observed_at=NOW,
        outcome_hash=HASH_B,
    )

    with pytest.raises(ValidationError, match="cannot mix evidence lanes"):
        OutcomeLabelPartition(
            evidence_lane=EvidenceLane.BACKTEST,
            labels=(backtest, live),
        )


def test_outcome_lane_enum_keeps_all_five_labels_distinct() -> None:
    assert {lane.value for lane in EvidenceLane} == {
        "backtest",
        "shadow",
        "testnet",
        "canary",
        "live",
    }


def test_complete_validation_receipt_requires_all_held_out_dimensions() -> None:
    dimensions = tuple(_held_out_dimension(item) for item in HeldOutDimension)
    receipt = CompleteHeldOutValidationReceipt(
        validation_run_id="validation-1",
        model_or_strategy_id="ou-v2",
        source_dataset_id="dataset_v2",
        source_label_lane=EvidenceLane.BACKTEST,
        dimensions=dimensions,
        created_at=NOW + timedelta(days=3),
    )

    assert {item.dimension for item in receipt.dimensions} == set(HeldOutDimension)

    with pytest.raises(ValidationError, match="dimensions"):
        CompleteHeldOutValidationReceipt(
            validation_run_id="validation-missing",
            model_or_strategy_id="ou-v2",
            source_dataset_id="dataset_v2",
            source_label_lane=EvidenceLane.BACKTEST,
            dimensions=dimensions[:-1],
            created_at=NOW + timedelta(days=3),
        )


def test_held_out_receipts_reject_overlap_and_invalid_time_boundary() -> None:
    payload = _held_out_dimension(HeldOutDimension.PAIR).model_dump()
    payload["held_out_members"] = payload["training_members"]
    payload["dimension_receipt_id"] = ""
    with pytest.raises(ValidationError, match="must be disjoint"):
        HeldOutDimensionReceipt.model_validate(payload)

    payload = _held_out_dimension(HeldOutDimension.TIME).model_dump()
    payload["held_out_start"] = payload["training_end"]
    payload["dimension_receipt_id"] = ""
    with pytest.raises(ValidationError, match="must follow training window"):
        HeldOutDimensionReceipt.model_validate(payload)
