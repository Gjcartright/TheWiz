from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from hashlib import sha256

import pandas as pd
import pytest

from quant_platform.orchestration.corrective_registered_rerun import (
    CONCLUSION_SCHEMA_VERSION,
    _accepted_mode_evidence_mask,
    _candidate_gate_row,
    _collapse_pending_family_preflight_rows,
    _contract_conclusively_accounted,
    _load_or_create_contract,
    _next_vendor_parity_action,
    _pending_family_preflight_row,
    _proof_scheduler_mixed_evidence_complete,
    _resolve_bound_cost_models,
    _resolve_registered_outcomes,
    resolve_registered_source_family,
    validate_registered_rerun_contract_identity,
)
from quant_platform.wizard_credit_ledger import (
    PROOF_LANE,
    reconcile_wizard_credit_lane,
    reserve_wizard_credit_lane,
)
from tests.capture_reconciliation_support import (
    write_valid_capture_reconciliation_evidence,
)
from tests.pair_cost_bundle_support import publish_valid_pair_cost_bundle

NOW = datetime(2026, 8, 9, 19, 30, tzinfo=UTC)
REGISTERED = NOW - timedelta(hours=6)


def test_production_registered_contract_id_rejects_rehashed_identity_edit():
    material = {
        "schema_version": "thewiz.corrective_registered_rerun.v1",
        "registered_candidates": [
            {
                "semantic_hypothesis_id": "hypothesis-1",
                "source_experiment_id": "experiment-1",
                "pair_group_key": "hyperliquid|1h|ETH|BTC",
                "pair": "ETH-USD-BTC-USD",
                "exact_mode": "Static (Spread)",
                "orientation": "original",
            }
        ],
        "acceptance_policy_id": "acceptance-1",
        "holdout_policy_id": "holdout-1",
        "discovery_policy_sha256": "a" * 64,
        "source_family_sha256": "b" * 64,
        "source_family_rows": 1,
        "candidate_selection_path": "reports/active/candidates.csv",
        "candidate_selection_sha256": "c" * 64,
        "candidate_selection_status": "PASS",
        "generation": 1,
        "prior_contract_id": "",
    }
    contract = {
        **material,
        "contract_id": "registeredrerun_"
        + sha256(
            json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:20],
    }
    validate_registered_rerun_contract_identity(contract)

    contract["registered_candidates"][0]["orientation"] = "reverse"
    with pytest.raises(ValueError, match="immutable contract identity mismatch"):
        validate_registered_rerun_contract_identity(contract)


def test_legacy_production_registered_contract_id_remains_reproducible():
    material = {
        "schema_version": "thewiz.corrective_registered_rerun.v1",
        "registered_candidates": [
            {
                "semantic_hypothesis_id": "hypothesis-legacy",
                "source_experiment_id": "experiment-legacy",
                "pair_group_key": "hyperliquid|1h|ETH|BTC",
                "pair": "ETH-USD-BTC-USD",
                "exact_mode": "Copula",
                "orientation": "reverse",
            }
        ],
        "acceptance_policy_id": "acceptance-1",
        "holdout_policy_id": "holdout-1",
        "discovery_policy_sha256": "a" * 64,
        "source_family_sha256": "b" * 64,
        "source_family_rows": 1,
    }
    contract = {
        **material,
        "contract_id": "registeredrerun_"
        + sha256(
            json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:20],
    }

    validate_registered_rerun_contract_identity(contract)


def test_generation_contract_before_selection_binding_remains_reproducible():
    material = {
        "schema_version": "thewiz.corrective_registered_rerun.v1",
        "registered_candidates": [
            {
                "semantic_hypothesis_id": "hypothesis-generation",
                "source_experiment_id": "experiment-generation",
            }
        ],
        "acceptance_policy_id": "acceptance-1",
        "holdout_policy_id": "holdout-1",
        "discovery_policy_sha256": "a" * 64,
        "source_family_sha256": "b" * 64,
        "source_family_rows": 1,
        "generation": 2,
        "prior_contract_id": "registeredrerun_prior",
    }
    contract = {
        **material,
        "contract_id": "registeredrerun_"
        + sha256(
            json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:20],
    }

    validate_registered_rerun_contract_identity(contract)


def test_pending_family_preflight_proves_non_vendor_rerun_inputs(tmp_path):
    history_path = tmp_path / "data" / "history" / "pair.json"
    history_path.parent.mkdir(parents=True)
    history_path.write_text("{}\n", encoding="utf-8")
    candidate = {
        "semantic_hypothesis_id": "hypothesis-1",
        "experiment_id": "experiment-1",
        "equivalence_cluster_id": "cluster-1",
        "pair": "ETH-USD-WLD-USD",
        "exact_mode": "Dyn (ZScoreR)",
        "orientation": "reverse",
        "registration_cohort_role": "walkforward_near_miss",
        "hypothesis_registered_before_next_test": True,
        "next_test_executed": False,
        "execution_authority": False,
        "live_trading_authorized": False,
    }
    queue = pd.DataFrame(
        [
            {
                "semantic_hypothesis_id": "hypothesis-1",
                "experiment_id": "experiment-1",
                "pair_group_key": "binance|daily|ETH|WLD",
            },
            {
                "semantic_hypothesis_id": "hypothesis-1",
                "experiment_id": "experiment-2",
                "pair_group_key": "coinbase|daily|ETH|WLD",
            },
        ]
    )
    matrix = pd.DataFrame(
        [
            {
                "experiment_id": "experiment-1",
                "pair_group_key": "binance|daily|ETH|WLD",
                "pair": "ETH-USD-WLD-USD",
                "exact_mode": "Dyn (ZScoreR)",
                "orientation": "reverse",
            }
        ]
    )
    ledger_audit = pd.DataFrame(
        [
            {
                "semantic_hypothesis_id": "hypothesis-1",
                "ledger_chain_valid": True,
                "acceptance_policy_id": "acceptance-1",
                "holdout_policy_id": "holdout-1",
                "promotion_authority": False,
                "live_trading_authorized": False,
            }
        ]
    )
    costs = pd.DataFrame(
        [
            {
                "pair_group_key": "binance|daily|ETH|WLD",
                "cost_model_id": "cost-1",
                "model_as_of_utc": (NOW - timedelta(minutes=10)).isoformat(),
                "strict_observed_cost_ready": True,
                "cost_acceptance_ready": True,
                "cost_model_status": "STRICT_OBSERVED",
                "candidate_set_sha256": "a" * 64,
                "cost_status_sha256": "b" * 64,
                "l2_samples_sha256": "c" * 64,
                "fee_profile_sha256": "d" * 64,
            }
        ]
    )
    history = pd.DataFrame(
        [
            {
                "pair_group_key": "binance|daily|ETH|WLD",
                "history_run_id": "history-1",
                "history_status": "READY_FOR_CANONICAL_1X_REPLAY",
                "history_rows": 1500,
                "history_path": "data/history/pair.json",
            }
        ]
    )
    policies = (
        {"status": "PASS", "policy_id": "acceptance-1"},
        {"status": "PASS", "policy_id": "holdout-1"},
    )

    ready = _pending_family_preflight_row(
        candidate=candidate,
        as_of=NOW,
        queue=queue,
        matrix=matrix,
        ledger_audit=ledger_audit,
        costs=costs,
        history_results=history,
        acceptance=policies[0],
        holdout=policies[1],
        history_manifest_bound=True,
        cost_bundle_valid=True,
        source_family_valid=True,
        root=tmp_path,
    )
    stale = _pending_family_preflight_row(
        candidate=candidate,
        as_of=NOW + timedelta(hours=3),
        queue=queue,
        matrix=matrix,
        ledger_audit=ledger_audit,
        costs=costs,
        history_results=history,
        acceptance=policies[0],
        holdout=policies[1],
        history_manifest_bound=True,
        cost_bundle_valid=True,
        source_family_valid=True,
        root=tmp_path,
    )

    assert ready["non_vendor_preflight_ready"] is True
    assert ready["registration_ready"] is True
    assert ready["identity_ready"] is True
    assert ready["strict_cost_ready"] is True
    assert ready["history_ready"] is True
    assert ready["promotion_authority"] is False
    assert stale["non_vendor_preflight_ready"] is False
    assert stale["blocker"] == "strict_cost_identity_missing_stale_or_unbound"

    unknown_experiment = _pending_family_preflight_row(
        candidate={**candidate, "experiment_id": "experiment-unknown"},
        as_of=NOW,
        queue=queue,
        matrix=matrix,
        ledger_audit=ledger_audit,
        costs=costs,
        history_results=history,
        acceptance=policies[0],
        holdout=policies[1],
        history_manifest_bound=True,
        cost_bundle_valid=True,
        source_family_valid=True,
        root=tmp_path,
    )
    assert unknown_experiment["registration_ready"] is False
    assert unknown_experiment["identity_ready"] is False


def test_family_preflight_selects_one_ready_variant_per_semantic_hypothesis():
    rows = pd.DataFrame(
        [
            {
                "semantic_hypothesis_id": "hypothesis-1",
                "experiment_id": "experiment-blocked",
                "equivalence_cluster_id": "cluster-1",
                "pair": "ETH-USD-WLD-USD",
                "wizard_timeframe": "daily",
                "exact_mode": "Dyn (ZScoreR)",
                "orientation": "reverse",
                "registration_ready": True,
                "identity_ready": True,
                "strict_cost_ready": False,
                "history_ready": True,
                "non_vendor_preflight_ready": False,
                "blocker": "strict_cost_identity_missing_stale_or_unbound",
            },
            {
                "semantic_hypothesis_id": "hypothesis-1",
                "experiment_id": "experiment-ready",
                "equivalence_cluster_id": "cluster-1",
                "pair": "ETH-USD-WLD-USD",
                "wizard_timeframe": "daily",
                "exact_mode": "Dyn (ZScoreR)",
                "orientation": "reverse",
                "registration_ready": True,
                "identity_ready": True,
                "strict_cost_ready": True,
                "history_ready": True,
                "non_vendor_preflight_ready": True,
                "blocker": "",
            },
        ]
    )

    selected = _collapse_pending_family_preflight_rows(rows)

    assert len(selected) == 1
    assert selected.iloc[0]["experiment_id"] == "experiment-ready"
    assert bool(selected.iloc[0]["non_vendor_preflight_ready"]) is True
    assert int(selected.iloc[0]["source_experiment_count"]) == 2
    assert int(selected.iloc[0]["ready_experiment_count"]) == 1
    assert selected.iloc[0]["representative_selection_reason"] == (
        "EVIDENCE_READY_VARIANT_SELECTED"
    )

    conflicting = rows.copy()
    conflicting.loc[1, "orientation"] = "original"
    blocked = _collapse_pending_family_preflight_rows(conflicting)
    assert bool(blocked.iloc[0]["non_vendor_preflight_ready"]) is False
    assert "semantic_identity_conflict_across_experiment_variants" in blocked.iloc[0][
        "blocker"
    ]


def test_family_preflight_uses_immutable_cost_bundle_when_active_layer_rotates(tmp_path):
    artifacts = publish_valid_pair_cost_bundle(
        root=tmp_path,
        now=NOW,
        specs=[
            {
                "pair_group_key": "dydx|daily|ETH|PYTH",
                "pair": "ETH-USD-PYTH-USD",
                "asset_x": "ETH",
                "asset_y": "PYTH",
            }
        ],
    )
    active = artifacts["active"]
    frozen = artifacts["models"]
    receipt = artifacts["bundle"]
    artifacts["pointer"].unlink()
    expected_cost_id = pd.read_csv(frozen, keep_default_na=False).iloc[0][
        "cost_model_id"
    ]
    pd.DataFrame([{"cost_model_id": "active-rotated"}]).to_csv(active, index=False)
    l2_status = {
        "pair_cost_bundle_manifest_path": str(receipt.relative_to(tmp_path)),
        "pair_cost_bundle_manifest_sha256": sha256(receipt.read_bytes()).hexdigest(),
    }

    costs, source, valid, active_matches = _resolve_bound_cost_models(
        root=tmp_path,
        active_costs_path=active,
        l2_status=l2_status,
    )

    assert source == frozen
    assert valid is True
    assert active_matches is False
    assert costs.iloc[0]["cost_model_id"] == expected_cost_id


def test_family_preflight_rejects_partial_legacy_cost_bundle_downgrade(tmp_path):
    active = tmp_path / "data" / "processed" / "hyperliquid_pair_cost_models.csv"
    frozen = tmp_path / "data" / "research" / "cost-bundle" / "pair_cost_models.csv"
    receipt = tmp_path / "data" / "research" / "cost-bundle" / "receipt.json"
    active.parent.mkdir(parents=True)
    frozen.parent.mkdir(parents=True)
    pd.DataFrame([{"cost_model_id": "partial-model"}]).to_csv(active, index=False)
    frozen.write_bytes(active.read_bytes())
    receipt.write_text(
        json.dumps(
            {
                "pair_cost_models_path": str(frozen.relative_to(tmp_path)),
                "pair_cost_models_sha256": sha256(frozen.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )

    costs, source, valid, active_matches = _resolve_bound_cost_models(
        root=tmp_path,
        active_costs_path=active,
        l2_status={
            "pair_cost_bundle_manifest_path": str(receipt.relative_to(tmp_path)),
            "pair_cost_bundle_manifest_sha256": sha256(receipt.read_bytes()).hexdigest(),
        },
    )

    assert costs.empty
    assert source is None
    assert valid is False
    assert active_matches is False


def test_active_cost_bundle_pointer_is_bound_and_fails_closed_on_drift(tmp_path):
    artifacts = publish_valid_pair_cost_bundle(
        root=tmp_path,
        now=NOW,
        specs=[
            {
                "pair_group_key": "dydx|daily|ETH|PYTH",
                "pair": "ETH-USD-PYTH-USD",
                "asset_x": "ETH",
                "asset_y": "PYTH",
            }
        ],
    )
    active = artifacts["active"]
    frozen = artifacts["models"]
    manifest_path = artifacts["bundle"]
    model_rows = pd.read_csv(frozen, keep_default_na=False).to_dict("records")
    expected_cost_id = model_rows[0]["cost_model_id"]
    pointer = json.loads(artifacts["pointer"].read_text(encoding="utf-8"))
    pointer_material = {
        key: value
        for key, value in pointer.items()
        if key not in {"pointer_id", "receipt_sha256"}
    }

    def write_pointer(material):
        pointer = {
            **material,
            "pointer_id": "l2costpointer_"
            + sha256(
                json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()[:20],
        }
        pointer["receipt_sha256"] = sha256(
            json.dumps(pointer, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        pointer_path = (
            tmp_path / "reports" / "active" / "hyperliquid_pair_cost_bundle_pointer.json"
        )
        pointer_path.parent.mkdir(parents=True, exist_ok=True)
        pointer_path.write_text(json.dumps(pointer, indent=2, sort_keys=True) + "\n")

    write_pointer(pointer_material)
    costs, source, valid, active_matches = _resolve_bound_cost_models(
        root=tmp_path,
        active_costs_path=active,
        l2_status={},
    )
    assert valid is True
    assert active_matches is True
    assert source == frozen
    assert costs.iloc[0]["cost_model_id"] == expected_cost_id

    pd.DataFrame([{**model_rows[0], "cost_model_id": "rotated"}]).to_csv(
        active, index=False
    )
    _, source, valid, active_matches = _resolve_bound_cost_models(
        root=tmp_path,
        active_costs_path=active,
        l2_status={
            "pair_cost_bundle_manifest_path": str(manifest_path.relative_to(tmp_path)),
            "pair_cost_bundle_manifest_sha256": sha256(manifest_path.read_bytes()).hexdigest(),
        },
    )
    assert source is None
    assert valid is False
    assert active_matches is False

    active.write_bytes(frozen.read_bytes())
    write_pointer({**pointer_material, "live_trading_authorized": True})
    _, source, valid, active_matches = _resolve_bound_cost_models(
        root=tmp_path,
        active_costs_path=active,
        l2_status={},
    )
    assert source is None
    assert valid is False
    assert active_matches is False


def _conclusion_receipt(payload: dict) -> dict:
    core = {"schema_version": CONCLUSION_SCHEMA_VERSION, **payload}
    return {
        **core,
        "conclusion_id": "registeredconclusion_"
        + sha256(
            json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:20],
    }


def test_conclusion_receipt_requires_exact_identity_coverage_and_counts():
    contract = {
        "contract_id": "registeredrerun-test",
        "registered_candidates": [
            {"semantic_hypothesis_id": "hypothesis-one"}
        ],
    }
    core = {
        "contract_id": "registeredrerun-test",
        "registered_hypotheses": 1,
        "accepted_registered_hypotheses": 0,
        "rejected_registered_hypotheses": 1,
        "outcomes": {"hypothesis-one": "REJECTED_BY_FROZEN_GATES"},
        "conclusion_status": "CONCLUSIVE_ZERO_REGISTERED_SURVIVORS",
        "promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    valid = _conclusion_receipt(core)
    assert _contract_conclusively_accounted(contract, valid)

    duplicated_contract = {
        **contract,
        "registered_candidates": contract["registered_candidates"] * 2,
    }
    assert not _contract_conclusively_accounted(duplicated_contract, valid)

    identity_tampered = {**valid, "rejected_registered_hypotheses": 0}
    assert not _contract_conclusively_accounted(contract, identity_tampered)

    count_forged = _conclusion_receipt(
        {**core, "accepted_registered_hypotheses": 1}
    )
    assert not _contract_conclusively_accounted(contract, count_forged)

    incomplete = _conclusion_receipt(
        {**core, "outcomes": {"different-hypothesis": "REJECTED_BY_FROZEN_GATES"}}
    )
    assert not _contract_conclusively_accounted(contract, incomplete)


def test_frozen_source_family_survives_daily_active_matrix_rotation(tmp_path):
    snapshot = (
        tmp_path
        / "reports"
        / "snapshots"
        / "current_wizard_hyperliquid"
        / "refresh-one"
        / "handoff-one"
        / "experiment_matrix.csv"
    )
    snapshot.parent.mkdir(parents=True)
    family = pd.DataFrame(
        [
            {"experiment_id": "exp-1", "pair": "ETH-USD-PYTH-USD"},
            {"experiment_id": "exp-2", "pair": "BTC-USD-ETH-USD"},
        ]
    )
    family.to_csv(snapshot, index=False)
    expected_hash = sha256(snapshot.read_bytes()).hexdigest()
    current = tmp_path / "reports" / "active" / "current_matrix.csv"
    current.parent.mkdir(parents=True)
    pd.DataFrame([{"experiment_id": "new-exp"}]).to_csv(current, index=False)
    contract = {
        "contract_id": "registeredrerun-frozen",
        "source_family_sha256": expected_hash,
        "source_family_rows": 2,
    }

    frozen, receipt = resolve_registered_source_family(
        root=tmp_path,
        contract=contract,
        current_matrix_path=current,
    )
    pd.DataFrame([{"experiment_id": "newer-exp"}]).to_csv(current, index=False)
    repeated, repeated_receipt = resolve_registered_source_family(
        root=tmp_path,
        contract=contract,
        current_matrix_path=current,
    )

    assert frozen == repeated
    assert receipt == repeated_receipt
    assert sha256(frozen.read_bytes()).hexdigest() == expected_hash
    assert json.loads(receipt.read_text())["daily_active_layer_independent"] is True

    frozen.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source family receipt mismatch"):
        resolve_registered_source_family(
            root=tmp_path,
            contract=contract,
            current_matrix_path=current,
        )


def _candidate(**overrides):
    return {
        "semantic_hypothesis_id": "hypothesis-1",
        "source_experiment_id": "experiment-1",
        "pair_group_key": "pair-group-1",
        "pair": "ETH-USD-PYTH-USD",
        "exact_mode": "Copula",
        "orientation": "reverse",
        "registered_at_utc": REGISTERED.isoformat(),
        "hypothesis_registered_before_next_test": True,
        "next_test_executed_at_contract": False,
        "execution_authority_at_contract": False,
        "testnet_order_authority_at_contract": False,
        "live_trading_authorized_at_contract": False,
        **overrides,
    }


def _inputs(*, cost_at=NOW, matrix_pair="ETH-USD-PYTH-USD", parity_value=True):
    queue = pd.DataFrame(
        [
            {
                "semantic_hypothesis_id": "hypothesis-1",
                "pair": "ETH-USD-PYTH-USD",
                "execution_authority": False,
                "promotion_authority": False,
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            }
        ]
    )
    ledger = [
        {
            "semantic_hypothesis_id": "hypothesis-1",
            "registered_at_utc": REGISTERED.isoformat(),
            "promotion_authority": False,
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        }
    ]
    audit = pd.DataFrame(
        [
            {
                "semantic_hypothesis_id": "hypothesis-1",
                "ledger_chain_valid": True,
                "promotion_authority": False,
                "live_trading_authorized": False,
            }
        ]
    )
    matrix = pd.DataFrame(
        [
            {
                "experiment_id": "experiment-1",
                "pair_group_key": "pair-group-1",
                "pair": matrix_pair,
                "exact_mode": "Copula",
                "orientation": "reverse",
            }
        ]
    )
    costs = pd.DataFrame(
        [
            {
                "pair_group_key": "pair-group-1",
                "cost_model_id": "cost-1",
                "model_as_of_utc": cost_at.isoformat(),
                "strict_observed_cost_ready": True,
                "cost_acceptance_ready": True,
                "cost_model_status": "STRICT_OBSERVED",
                "candidate_set_sha256": "a" * 64,
                "cost_status_sha256": "b" * 64,
                "l2_samples_sha256": "c" * 64,
                "fee_profile_sha256": "d" * 64,
            }
        ]
    )
    parity = pd.DataFrame(
        [
            {
                "exact_mode": "Copula",
                "orientation": "reverse",
                "vendor_exact_mode_parity_proven": parity_value,
            }
        ]
    )
    return queue, ledger, audit, matrix, costs, parity


def _run(candidate=None, **input_overrides):
    queue, ledger, audit, matrix, costs, parity = _inputs(**input_overrides)
    return _candidate_gate_row(
        candidate=candidate or _candidate(),
        as_of=NOW,
        queue=queue,
        ledger=ledger,
        ledger_audit=audit,
        matrix=matrix,
        costs=costs,
        parity=parity,
        parity_complete=True,
        proof_queue_complete=True,
        stage_two_complete=True,
        policy_ready=True,
        chain_integrity=True,
    )


def test_registered_rerun_gate_accepts_only_full_family_research_rerun():
    row = _run()

    assert row["pre_rerun_gate_ready"] is True
    assert row["rerun_scope"] == "full_policy_defined_family"
    assert row["promotion_evaluation_scope"] == (
        "registered_semantic_hypothesis_only"
    )
    assert row["rerun_execution_included"] is False
    assert row["testnet_order_authority"] is False
    assert row["live_trading_authorized"] is False


def test_mode_evidence_requires_formula_except_for_behavioral_copula():
    parity = pd.DataFrame(
        [
            {
                "exact_mode": "Static (Spread)",
                "vendor_exact_mode_parity_proven": True,
                "vendor_behavioral_parity_proven": False,
                "vendor_mode_evidence_gate_passed": True,
            },
            {
                "exact_mode": "Dyn (Spread)",
                "vendor_exact_mode_parity_proven": False,
                "vendor_behavioral_parity_proven": True,
                "vendor_mode_evidence_gate_passed": True,
            },
            {
                "exact_mode": "Copula",
                "vendor_exact_mode_parity_proven": False,
                "vendor_behavioral_parity_proven": True,
                "vendor_mode_evidence_gate_passed": True,
            },
        ]
    )

    assert _accepted_mode_evidence_mask(parity).tolist() == [True, False, True]


def test_registered_rerun_gate_blocks_identity_stale_cost_and_partial_parity():
    forged = _run(matrix_pair="ETH-USD-BTC-USD")
    stale = _run(cost_at=REGISTERED - timedelta(minutes=1))
    partial = _run(parity_value=False)

    assert forged["pre_rerun_gate_ready"] is False
    assert "registered_identity_missing" in forged["blocker"]
    assert stale["pre_rerun_gate_ready"] is False
    assert "strict_cost_evidence_missing" in stale["blocker"]
    assert partial["pre_rerun_gate_ready"] is False
    assert "candidate_mode_orientation_parity_not_proven" in partial["blocker"]


def test_registered_rerun_gate_blocks_forged_execution_authority():
    row = _run(candidate=_candidate(execution_authority_at_contract=True))

    assert row["authority_boundary_safe"] is False
    assert row["registration_valid"] is False
    assert row["pre_rerun_gate_ready"] is False
    assert "registration_or_ledger_invalid" in row["blocker"]


def test_registered_outcome_accounting_accepts_survivor_or_explicit_rejection():
    candidates = [
        _candidate(),
        _candidate(
            semantic_hypothesis_id="hypothesis-2",
            source_experiment_id="experiment-2",
        ),
        _candidate(
            semantic_hypothesis_id="hypothesis-3",
            source_experiment_id="experiment-3",
        ),
    ]
    queue = pd.DataFrame([{"semantic_hypothesis_id": "hypothesis-2"}])
    outcomes = _resolve_registered_outcomes(
        candidates=candidates,
        queue=queue,
        final_survivor={"final_experiment_ids": ["experiment-1"]},
    )

    assert outcomes == {
        "hypothesis-1": "ACCEPTED_SURVIVOR",
        "hypothesis-2": "REJECTED_BY_FROZEN_GATES",
        "hypothesis-3": "UNACCOUNTED",
    }


def test_registered_outcome_accounting_rejects_early_failure_from_full_ledger():
    outcomes = _resolve_registered_outcomes(
        candidates=[_candidate()],
        queue=pd.DataFrame(),
        final_survivor={"final_experiment_ids": []},
        failure_attribution=pd.DataFrame(
            [
                {
                    "experiment_id": "experiment-1",
                    "first_blocking_stage": "canonical_replay",
                }
            ]
        ),
    )

    assert outcomes == {"hypothesis-1": "REJECTED_BY_FROZEN_GATES"}


def test_vendor_parity_action_switches_after_response_capture_finishes():
    assert (
        _next_vendor_parity_action(
            {
                "queue_eligible": 28,
                "completed_after": 4,
                "responses_captured_after": 28,
            }
        )
        == "reconcile_frozen_wizard_capture_manifest"
    )
    assert _next_vendor_parity_action(
        {
            "queue_eligible": 28,
            "completed_after": 4,
            "responses_captured_after": 28,
            "capture_reconciliation_complete": True,
        }
    ).startswith("implement_mode_specific_comparators")
    assert _next_vendor_parity_action(
        {
            "queue_eligible": 28,
            "completed_after": 0,
            "responses_captured_after": 3,
            "capture_reconciliation_complete": True,
        }
    ).startswith("proof_scheduler_completes")


def test_mixed_formula_and_immutable_copula_evidence_completes_queue(tmp_path):
    cohort_id = "copulacohort-mixed"
    cohort_path = (
        tmp_path
        / "data"
        / "research"
        / "wizard_copula_behavioral_cohorts"
        / f"{cohort_id}.json"
    )
    cohort_path.parent.mkdir(parents=True)
    cohort_path.write_text("{}\n", encoding="utf-8")
    reservation = reserve_wizard_credit_lane(
        root=tmp_path,
        lane=PROOF_LANE,
        planned_credits=68,
        now=NOW,
    )
    reconciliation = reconcile_wizard_credit_lane(
        root=tmp_path,
        lane=PROOF_LANE,
        reservation_id=reservation.summary["reservation_id"],
        reconciliation_key="mixed-proof-cycle",
        attempted_credits=56,
        completed_credits=56,
        external_requests=32,
        observed_used_before=0,
        observed_used_after=56,
        activity_rows=[
            {
                "lane": "exact_mode",
                "external_requests": 24,
                "credit_cost": 2,
                "attempted_credits": 48,
                "completed_credits": 48,
            },
            {
                "lane": "copula_behavioral",
                "external_requests": 8,
                "credit_cost": 1,
                "attempted_credits": 8,
                "completed_credits": 8,
            },
        ],
        now=NOW,
    )
    cycle = {
        **write_valid_capture_reconciliation_evidence(tmp_path),
        "attempt_date_utc": NOW.date().isoformat(),
        "status": "COMPLETE_ACCEPTED_MODE_EVIDENCE",
        "queue_eligible": 28,
        "completed_after": 24,
        "responses_captured_after": 28,
        "formula_proofs_expected": 24,
        "accepted_mode_evidence_cells": 28,
        "copula_behavioral_expected_cells": 4,
        "copula_behavioral_cells_passed": 4,
        "copula_behavioral_provenance_cells": 4,
        "copula_behavioral_endpoint_calls": 8,
        "copula_behavioral_responses_captured_this_cycle": 8,
        "copula_behavioral_response_accounting_valid": True,
        "copula_behavioral_parity_proven": True,
        "copula_formula_parity_proven": False,
        "copula_cohort_receipt_id": cohort_id,
        "copula_cohort_receipt_path": str(cohort_path.relative_to(tmp_path)),
        "copula_cohort_receipt_sha256": sha256(
            cohort_path.read_bytes()
        ).hexdigest(),
        "copula_cohort_receipt_valid": True,
        "parity_refresh_status": "PASS",
        "exact_mode_requests_attempted": 24,
        "exact_mode_responses_captured_this_cycle": 24,
        "proof_lane_attempted_credits": 56,
        "proof_lane_completed_credits": 56,
        "proof_lane_uncompleted_attempted_credits": 0,
        "credit_reconciliation_status": "PASS_RECONCILED",
        "credit_reservation_id": reservation.summary["reservation_id"],
        "credit_reservation_path": str(
            reservation.paths["reservation"].relative_to(tmp_path)
        ),
        "credit_reconciliation_id": reconciliation.summary["reconciliation_id"],
        "credit_reconciliation_path": str(
            reconciliation.paths["reconciliation"].relative_to(tmp_path)
        ),
        "candidate_promotion_authority": False,
        "order_submission_included": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    cycle["receipt_id"] = "wizardproof_" + sha256(
        json.dumps(cycle, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:20]
    cycle_path = (
        tmp_path
        / "reports"
        / "active"
        / "wizard_proof_scheduler_receipts"
        / "mixed.json"
    )
    cycle_path.parent.mkdir(parents=True)
    cycle_path.write_text(json.dumps(cycle), encoding="utf-8")
    immutable_cycle_path = (
        tmp_path
        / "data"
        / "research"
        / "wizard_proof_scheduler_receipts"
        / f"{cycle['receipt_id']}.json"
    )
    immutable_cycle_path.parent.mkdir(parents=True)
    immutable_cycle_path.write_bytes(cycle_path.read_bytes())
    latest = {
        **cycle,
        "receipt_path": str(cycle_path.relative_to(tmp_path)),
    }

    assert _proof_scheduler_mixed_evidence_complete(
        root=tmp_path, proof_scheduler=latest
    )

    continuity_false = {**cycle, "capture_manifest_continuity_valid": False}
    continuity_false.pop("receipt_id")
    continuity_false["receipt_id"] = "wizardproof_" + sha256(
        json.dumps(
            continuity_false, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()[:20]
    cycle_path.write_text(json.dumps(continuity_false), encoding="utf-8")
    assert not _proof_scheduler_mixed_evidence_complete(
        root=tmp_path,
        proof_scheduler={
            **continuity_false,
            "receipt_path": str(cycle_path.relative_to(tmp_path)),
        },
    )
    cycle_path.write_text(json.dumps(cycle), encoding="utf-8")

    cycle_false = {**cycle, "copula_behavioral_parity_proven": False}
    cycle_false.pop("receipt_id")
    cycle_false["receipt_id"] = "wizardproof_" + sha256(
        json.dumps(
            cycle_false, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()[:20]
    cycle_path.write_text(json.dumps(cycle_false), encoding="utf-8")
    false_cycle_true_latest = {
        **cycle_false,
        "copula_behavioral_parity_proven": True,
        "receipt_path": str(cycle_path.relative_to(tmp_path)),
    }
    assert not _proof_scheduler_mixed_evidence_complete(
        root=tmp_path, proof_scheduler=false_cycle_true_latest
    )
    cycle_path.write_text(json.dumps(cycle), encoding="utf-8")

    incomplete = {**latest, "accepted_mode_evidence_cells": 27}
    assert not _proof_scheduler_mixed_evidence_complete(
        root=tmp_path, proof_scheduler=incomplete
    )

    cohort_path.write_text("tampered\n", encoding="utf-8")
    assert not _proof_scheduler_mixed_evidence_complete(
        root=tmp_path, proof_scheduler=latest
    )
    cohort_path.write_text("{}\n", encoding="utf-8")

    v5_contract = tmp_path / "config" / "wizard_ou_comparator_v5_holdout.json"
    v5_receipt = tmp_path / "reports" / "active" / "wizard_ou_v5_holdout_receipt.json"
    v5_contract.parent.mkdir(parents=True, exist_ok=True)
    v5_receipt.parent.mkdir(parents=True, exist_ok=True)
    v5_contract.write_text("{}\n", encoding="utf-8")
    v5_receipt.write_text("{}\n", encoding="utf-8")
    assert not _proof_scheduler_mixed_evidence_complete(
        root=tmp_path, proof_scheduler=latest
    )
    v5_contract.unlink()
    v5_receipt.unlink()
    assert _proof_scheduler_mixed_evidence_complete(
        root=tmp_path, proof_scheduler=latest
    )

    source_receipt = json.loads(
        (tmp_path / cycle["capture_manifest_source_receipt_path"]).read_text()
    )
    source_snapshot = tmp_path / source_receipt["source_artifacts"][0]["snapshot_path"]
    source_snapshot.chmod(0o600)
    source_snapshot.write_text("tampered\n", encoding="utf-8")
    assert not _proof_scheduler_mixed_evidence_complete(
        root=tmp_path, proof_scheduler=latest
    )


def test_mixed_evidence_rejects_unresolved_or_tampered_credit_lineage(tmp_path):
    reservation = reserve_wizard_credit_lane(
        root=tmp_path,
        lane=PROOF_LANE,
        planned_credits=68,
        now=NOW,
    )
    reconciliation = reconcile_wizard_credit_lane(
        root=tmp_path,
        lane=PROOF_LANE,
        reservation_id=reservation.summary["reservation_id"],
        reconciliation_key="hostile-proof-cycle",
        attempted_credits=2,
        completed_credits=2,
        external_requests=1,
        observed_used_before=0,
        observed_used_after=2,
        activity_rows=[
            {
                "lane": "copula_behavioral",
                "external_requests": 1,
                "credit_cost": 2,
                "attempted_credits": 2,
                "completed_credits": 2,
            }
        ],
        now=NOW,
    )
    cohort_id = "copulacohort-hostile"
    cohort_path = (
        tmp_path
        / "data"
        / "research"
        / "wizard_copula_behavioral_cohorts"
        / f"{cohort_id}.json"
    )
    cohort_path.parent.mkdir(parents=True)
    cohort_path.write_text("{}\n", encoding="utf-8")
    cycle = {
        "attempt_date_utc": NOW.date().isoformat(),
        "status": "COMPLETE_ACCEPTED_MODE_EVIDENCE",
        "queue_eligible": 1,
        "completed_after": 0,
        "responses_captured_after": 1,
        "formula_proofs_expected": 0,
        "accepted_mode_evidence_cells": 1,
        "copula_behavioral_expected_cells": 1,
        "copula_behavioral_cells_passed": 1,
        "copula_behavioral_provenance_cells": 1,
        "copula_behavioral_endpoint_calls": 1,
        "copula_behavioral_responses_captured_this_cycle": 1,
        "copula_behavioral_response_accounting_valid": True,
        "copula_behavioral_parity_proven": True,
        "copula_formula_parity_proven": False,
        "copula_cohort_receipt_id": cohort_id,
        "copula_cohort_receipt_path": str(cohort_path.relative_to(tmp_path)),
        "copula_cohort_receipt_sha256": sha256(cohort_path.read_bytes()).hexdigest(),
        "copula_cohort_receipt_valid": True,
        "parity_refresh_status": "PASS",
        "exact_mode_requests_attempted": 0,
        "exact_mode_responses_captured_this_cycle": 0,
        "proof_lane_attempted_credits": 2,
        "proof_lane_completed_credits": 1,
        "proof_lane_uncompleted_attempted_credits": 1,
        "credit_reconciliation_status": "PASS_RECONCILED",
        "credit_reservation_id": reservation.summary["reservation_id"],
        "credit_reservation_path": str(
            reservation.paths["reservation"].relative_to(tmp_path)
        ),
        "credit_reconciliation_id": reconciliation.summary["reconciliation_id"],
        "credit_reconciliation_path": str(
            reconciliation.paths["reconciliation"].relative_to(tmp_path)
        ),
        "candidate_promotion_authority": False,
        "order_submission_included": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    cycle["receipt_id"] = "wizardproof_" + sha256(
        json.dumps(cycle, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:20]
    cycle_path = (
        tmp_path
        / "reports"
        / "active"
        / "wizard_proof_scheduler_receipts"
        / "hostile.json"
    )
    cycle_path.parent.mkdir(parents=True)
    cycle_path.write_text(json.dumps(cycle), encoding="utf-8")
    latest = {**cycle, "receipt_path": str(cycle_path.relative_to(tmp_path))}

    assert not _proof_scheduler_mixed_evidence_complete(
        root=tmp_path, proof_scheduler=latest
    )

    cycle["proof_lane_completed_credits"] = 2
    cycle["proof_lane_uncompleted_attempted_credits"] = 0
    cycle.pop("receipt_id")
    cycle["receipt_id"] = "wizardproof_" + sha256(
        json.dumps(cycle, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:20]
    cycle_path.write_text(json.dumps(cycle), encoding="utf-8")
    latest = {**cycle, "receipt_path": str(cycle_path.relative_to(tmp_path))}
    assert not _proof_scheduler_mixed_evidence_complete(
        root=tmp_path, proof_scheduler=latest
    )


def test_registered_contract_rotates_only_after_conclusive_accounting(tmp_path):
    active = tmp_path / "reports" / "active" / "registered_contract.json"
    discovery = tmp_path / "config" / "discovery.json"
    matrix_path = tmp_path / "reports" / "matrix.csv"
    discovery.parent.mkdir(parents=True)
    matrix_path.parent.mkdir(parents=True)
    discovery.write_text('{"policy": "frozen"}', encoding="utf-8")

    def inputs(suffix: str):
        semantic_id = f"hypothesis-{suffix}"
        experiment_id = f"experiment-{suffix}"
        pair = f"ETH-USD-{suffix.upper()}-USD"
        batch = pd.DataFrame(
            [
                {
                    "semantic_hypothesis_id": semantic_id,
                    "experiment_id": experiment_id,
                    "pair": pair,
                    "exact_mode": "Copula",
                    "orientation": "original",
                    "hypothesis_registered_before_next_test": True,
                    "next_test_executed": False,
                    "execution_authority": False,
                    "testnet_order_authority": False,
                    "live_trading_authorized": False,
                }
            ]
        )
        queue = pd.DataFrame(
            [
                {
                    "semantic_hypothesis_id": semantic_id,
                    "pair_group_key": f"hyperliquid|1h|ETH|{suffix.upper()}",
                }
            ]
        )
        ledger = [
            {
                "semantic_hypothesis_id": semantic_id,
                "registered_at_utc": NOW.isoformat(),
                "record_hash": suffix * 64,
            }
        ]
        matrix = pd.concat(
            [
                batch,
                pd.DataFrame(
                    [
                        {
                            "experiment_id": f"family-{suffix}-2",
                            "pair": "BTC-USD-ETH-USD",
                        },
                        {
                            "experiment_id": f"family-{suffix}-3",
                            "pair": "SOL-USD-ETH-USD",
                        },
                    ]
                ),
            ],
            ignore_index=True,
        )
        matrix.to_csv(matrix_path, index=False)
        chain = {
            "chain_status": "PASS",
            "experiment_authority_count": len(matrix),
        }
        return batch, queue, ledger, chain

    def build(suffix: str, *, generation_rollover_ready: bool = True):
        batch, queue, ledger, chain = inputs(suffix)
        return _load_or_create_contract(
            root=tmp_path,
            as_of=NOW,
            active_contract_path=active,
            batch=batch,
            queue=queue,
            ledger=ledger,
            acceptance={"policy_id": "acceptance-v1"},
            holdout={"policy_id": "holdout-v1"},
            discovery_policy_path=discovery,
            matrix_path=matrix_path,
            chain=chain,
            generation_rollover_ready=generation_rollover_ready,
        )

    first = build("one")
    old_immutable = tmp_path / first["immutable_contract_path"]
    old_payload = json.loads(old_immutable.read_text())

    pinned = build("two")
    assert pinned["contract_id"] == first["contract_id"]

    conclusion = (
        tmp_path
        / "data"
        / "research"
        / "registered_rerun_conclusions"
        / f"{first['contract_id']}.json"
    )
    conclusion.parent.mkdir(parents=True)
    conclusion.write_text(
        json.dumps(
            _conclusion_receipt({
                "contract_id": first["contract_id"],
                "conclusion_status": "CONCLUSIVE_ZERO_REGISTERED_SURVIVORS",
                "registered_hypotheses": 1,
                "accepted_registered_hypotheses": 0,
                "rejected_registered_hypotheses": 1,
                "outcomes": {"hypothesis-one": "REJECTED_BY_FROZEN_GATES"},
                "promotion_authority": False,
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            })
        ),
        encoding="utf-8",
    )

    blocked_rollover = build("two", generation_rollover_ready=False)
    assert blocked_rollover["contract_id"] == first["contract_id"]

    second = build("two")
    assert second["contract_id"] != first["contract_id"]
    assert second["generation"] == 2
    assert second["prior_contract_id"] == first["contract_id"]
    assert json.loads(old_immutable.read_text()) == old_payload
    assert json.loads(active.read_text()) == second


def test_registered_contract_binds_preflight_representative_and_rejects_duplicates(
    tmp_path,
):
    active = tmp_path / "reports" / "active" / "registered_contract.json"
    discovery = tmp_path / "config" / "discovery.json"
    matrix_path = tmp_path / "reports" / "matrix.csv"
    selection_path = tmp_path / "reports" / "selected.csv"
    discovery.parent.mkdir(parents=True)
    matrix_path.parent.mkdir(parents=True)
    discovery.write_text('{"policy": "frozen"}', encoding="utf-8")
    selected = pd.DataFrame(
        [
            {
                "semantic_hypothesis_id": "hypothesis-one",
                "experiment_id": "experiment-ready",
                "pair_group_key": "binance|daily|ETH|WLD",
                "pair": "ETH-USD-WLD-USD",
                "exact_mode": "Copula",
                "orientation": "reverse",
                "representative_selection_reason": "EVIDENCE_READY_VARIANT_SELECTED",
                "alternative_experiment_ids": "experiment-blocked",
                "cost_model_id": "cost-one",
                "cost_model_as_of_utc": NOW.isoformat(),
                "history_run_id": "history-one",
                "history_path": "data/history/one.json",
                "history_sha256": "a" * 64,
                "hypothesis_registered_before_next_test": True,
                "next_test_executed": False,
                "execution_authority": False,
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            }
        ]
    )
    selected.to_csv(selection_path, index=False)
    selected.to_csv(matrix_path, index=False)
    queue = pd.DataFrame(
        [
            {
                "semantic_hypothesis_id": "hypothesis-one",
                "experiment_id": "experiment-ready",
                "pair_group_key": "wrong|daily|ETH|WLD",
            }
        ]
    )
    ledger = [
        {
            "semantic_hypothesis_id": "hypothesis-one",
            "registered_at_utc": NOW.isoformat(),
            "record_hash": "b" * 64,
        }
    ]
    kwargs = {
        "root": tmp_path,
        "as_of": NOW,
        "queue": queue,
        "ledger": ledger,
        "acceptance": {"policy_id": "acceptance-v1"},
        "holdout": {"policy_id": "holdout-v1"},
        "discovery_policy_path": discovery,
        "matrix_path": matrix_path,
        "chain": {"chain_status": "PASS", "experiment_authority_count": 1},
        "candidate_selection_path": selection_path,
        "candidate_selection_status": "PASS",
    }

    contract = _load_or_create_contract(
        active_contract_path=active,
        batch=selected,
        **kwargs,
    )

    assert contract["candidate_selection_path"] == "reports/selected.csv"
    assert contract["candidate_selection_sha256"] == sha256(
        selection_path.read_bytes()
    ).hexdigest()
    assert contract["candidate_selection_status"] == "PASS"
    assert len(contract["registered_candidates"]) == 1
    candidate = contract["registered_candidates"][0]
    assert candidate["source_experiment_id"] == "experiment-ready"
    assert candidate["pair_group_key"] == "binance|daily|ETH|WLD"
    assert candidate["cost_model_id"] == "cost-one"
    assert candidate["history_run_id"] == "history-one"

    duplicated = pd.concat(
        [selected, selected.assign(experiment_id="experiment-duplicate")],
        ignore_index=True,
    )
    with pytest.raises(
        ValueError,
        match="registered rerun contract semantic candidate identities invalid",
    ):
        _load_or_create_contract(
            active_contract_path=tmp_path / "reports" / "duplicate.json",
            batch=duplicated,
            **kwargs,
        )


def test_registered_contract_preserves_active_generation_when_selection_is_empty(
    tmp_path,
):
    active = tmp_path / "reports" / "active" / "registered_contract.json"
    discovery = tmp_path / "config" / "discovery.json"
    matrix_path = tmp_path / "reports" / "matrix.csv"
    discovery.parent.mkdir(parents=True)
    matrix_path.parent.mkdir(parents=True)
    discovery.write_text('{"policy": "frozen"}', encoding="utf-8")
    selected = pd.DataFrame(
        [
            {
                "semantic_hypothesis_id": "hypothesis-one",
                "experiment_id": "experiment-one",
                "pair": "ETH-USD-WLD-USD",
                "exact_mode": "Copula",
                "orientation": "original",
                "hypothesis_registered_before_next_test": True,
                "next_test_executed": False,
                "execution_authority": False,
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            }
        ]
    )
    selected.to_csv(matrix_path, index=False)
    kwargs = {
        "root": tmp_path,
        "as_of": NOW,
        "queue": pd.DataFrame(),
        "ledger": [
            {
                "semantic_hypothesis_id": "hypothesis-one",
                "registered_at_utc": NOW.isoformat(),
                "record_hash": "c" * 64,
            }
        ],
        "acceptance": {"policy_id": "acceptance-v1"},
        "holdout": {"policy_id": "holdout-v1"},
        "discovery_policy_path": discovery,
        "matrix_path": matrix_path,
        "chain": {"chain_status": "PASS", "experiment_authority_count": 1},
    }

    first = _load_or_create_contract(
        active_contract_path=active,
        batch=selected,
        **kwargs,
    )
    retained = _load_or_create_contract(
        active_contract_path=active,
        batch=pd.DataFrame(),
        **kwargs,
    )

    assert retained == first
    assert json.loads(active.read_text()) == first
    with pytest.raises(ValueError, match="no candidate selection"):
        _load_or_create_contract(
            active_contract_path=tmp_path / "reports" / "new-empty.json",
            batch=pd.DataFrame(),
            **kwargs,
        )
