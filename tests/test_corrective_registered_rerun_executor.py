from __future__ import annotations

import json
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import pandas as pd
import pytest

from quant_platform.active_pipeline import CommandResult
from quant_platform.orchestration.corrective_registered_rerun import (
    CONCLUSION_SCHEMA_VERSION,
)
from quant_platform.orchestration.corrective_registered_rerun_executor import (
    DEFAULT_STAGE_RUNNERS,
    _prepare_registered_workspace,
    _registered_pair_keys,
    run_registered_research_rerun,
)
from quant_platform.orchestration.corrective_statistical_remediation import (
    _registered_execution_survivor_bridge,
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

NOW = datetime(2026, 8, 10, 0, 10, tzinfo=UTC)
INDEPENDENT_PAIR_KEYS = (
    "dydx|daily|BTC|WLD",
    "dydx|daily|ETH|PYTH",
    "dydx|daily|SOL|TURBO",
)


def test_registered_stage4_chain_keeps_every_required_acceptance_stage():
    assert tuple(DEFAULT_STAGE_RUNNERS) == (
        "canonical_replay",
        "cost_evidence",
        "observed_cost_replay",
        "walkforward",
        "ou_optimal_overlay",
        "regime_attribution",
        "robustness",
        "concentration",
        "failure_attribution",
        "leverage_surface",
        "learning_ledger",
        "chain_validation",
    )


def test_registered_executor_rejects_duplicate_semantic_contract_candidates():
    contract = {
        "registered_candidates": [
            {"semantic_hypothesis_id": "hypothesis-one"},
            {"semantic_hypothesis_id": "hypothesis-one"},
        ]
    }
    gate = pd.DataFrame(
        [
            {
                "semantic_hypothesis_id": "hypothesis-one",
                "pair_group_key": "dydx|daily|ETH|WLD",
                "pre_rerun_gate_ready": True,
            }
        ]
    )

    with pytest.raises(
        ValueError,
        match="registered rerun contract semantic identities are invalid",
    ):
        _registered_pair_keys(gate, contract)


def test_registered_executor_rejects_duplicate_gate_row_replacing_missing_hypothesis():
    candidates = [
        {
            "semantic_hypothesis_id": "hypothesis-one",
            "source_experiment_id": "experiment-one",
            "pair_group_key": "dydx|daily|ETH|PYTH",
            "pair": "ETH-USD-PYTH-USD",
            "exact_mode": "Copula",
            "orientation": "reverse",
        },
        {
            "semantic_hypothesis_id": "hypothesis-two",
            "source_experiment_id": "experiment-two",
            "pair_group_key": "dydx|daily|BTC|WLD",
            "pair": "BTC-USD-WLD-USD",
            "exact_mode": "Static (Spread)",
            "orientation": "original",
        },
    ]
    duplicate = {**candidates[0], "pre_rerun_gate_ready": True}
    gate = pd.DataFrame([duplicate, duplicate])

    with pytest.raises(
        ValueError,
        match="registered rerun candidate gate identity coverage mismatch",
    ):
        _registered_pair_keys(gate, {"registered_candidates": candidates})


def test_registered_executor_rejects_gate_identity_binding_mismatch():
    candidate = {
        "semantic_hypothesis_id": "hypothesis-one",
        "source_experiment_id": "experiment-one",
        "pair_group_key": "dydx|daily|ETH|PYTH",
        "pair": "ETH-USD-PYTH-USD",
        "exact_mode": "Copula",
        "orientation": "reverse",
    }
    gate = pd.DataFrame(
        [{**candidate, "pair_group_key": "dydx|daily|ETH|WLD", "pre_rerun_gate_ready": True}]
    )

    with pytest.raises(
        ValueError,
        match="registered rerun candidate gate binding mismatch:.*pair_group_key",
    ):
        _registered_pair_keys(gate, {"registered_candidates": [candidate]})


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _conclusion_receipt(payload: dict) -> dict:
    core = {"schema_version": CONCLUSION_SCHEMA_VERSION, **payload}
    return {
        **core,
        "conclusion_id": "registeredconclusion_"
        + sha256(json.dumps(core, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:20],
    }


def _setup(
    root: Path,
    *,
    gate_status: str = "READY_FOR_NEXT_SCHEDULED_RESEARCH_RERUN",
    independent_candidates: int = 1,
):
    active = root / "reports" / "active"
    active.mkdir(parents=True)
    matrix_path = active / "current_wizard_hyperliquid_experiment_matrix.csv"
    matrix_rows = [
        {
            "experiment_id": "experiment-1",
            "pair_group_key": "dydx|daily|ETH|PYTH",
            "pair": "ETH-USD-PYTH-USD",
            "exact_mode": "Copula",
            "orientation": "reverse",
        },
        {
            "experiment_id": "experiment-2",
            "pair_group_key": (
                "dydx|daily|BTC|WLD" if independent_candidates >= 3 else "dydx|daily|ETH|PYTH"
            ),
            "pair": ("BTC-USD-WLD-USD" if independent_candidates >= 3 else "ETH-USD-PYTH-USD"),
            "exact_mode": "Static (Spread)",
            "orientation": "original",
        },
    ]
    if independent_candidates >= 3:
        matrix_rows.append(
            {
                "experiment_id": "experiment-3",
                "pair_group_key": "dydx|daily|SOL|TURBO",
                "pair": "SOL-USD-TURBO-USD",
                "exact_mode": "OU (Spread)",
                "orientation": "original",
            }
        )
    pd.DataFrame(matrix_rows).to_csv(matrix_path, index=False)
    snapshot_matrix = root / "reports" / "snapshots" / "handoff" / "experiment_matrix.csv"
    snapshot_matrix.parent.mkdir(parents=True)
    snapshot_matrix.write_bytes(matrix_path.read_bytes())
    _write_json(
        active / "current_wizard_hyperliquid_handoff_manifest.json",
        {"artifacts": {"snapshot_experiments": str(snapshot_matrix.relative_to(root))}},
    )
    discovery = root / "config" / "wizard_discovery_policy.json"
    _write_json(discovery, {"frozen": True})
    _write_json(
        root / "config" / "acceptance_policy_manifest.json",
        {
            "schema_version": "thewiz.acceptance_policy.v1",
            "research_gates": {"minimum_independent_supporting_clusters": 3},
        },
    )
    _write_json(
        active / "acceptance_policy_receipt.json",
        {"policy_id": "acceptance-1", "status": "PASS"},
    )
    _write_json(
        active / "holdout_policy_receipt.json",
        {"policy_id": "holdout-1", "status": "PASS"},
    )
    contract_id = "registeredrerun-test"
    immutable = root / "data" / "research" / "registered_rerun_contracts" / f"{contract_id}.json"
    contract = {
        "schema_version": "thewiz.corrective_registered_rerun.v1",
        "contract_id": contract_id,
        "immutable_contract_path": str(immutable.relative_to(root)),
        "source_family_sha256": _hash(matrix_path),
        "source_family_rows": len(matrix_rows),
        "discovery_policy_sha256": _hash(discovery),
        "acceptance_policy_id": "acceptance-1",
        "holdout_policy_id": "holdout-1",
        "full_family_multiplicity_required": True,
        "promotion_evaluation_registered_only": True,
        "threshold_changes_after_contract_permitted": False,
        "promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
        "registered_candidates": [
            {
                "semantic_hypothesis_id": "hypothesis-1",
                "source_experiment_id": "experiment-1",
                "pair_group_key": "dydx|daily|ETH|PYTH",
                "pair": "ETH-USD-PYTH-USD",
                "exact_mode": "Copula",
                "orientation": "reverse",
            },
            *(
                [
                    {
                        "semantic_hypothesis_id": "hypothesis-2",
                        "source_experiment_id": "experiment-2",
                        "pair_group_key": "dydx|daily|BTC|WLD",
                        "pair": "BTC-USD-WLD-USD",
                        "exact_mode": "Static (Spread)",
                        "orientation": "original",
                    },
                    {
                        "semantic_hypothesis_id": "hypothesis-3",
                        "source_experiment_id": "experiment-3",
                        "pair_group_key": "dydx|daily|SOL|TURBO",
                        "pair": "SOL-USD-TURBO-USD",
                        "exact_mode": "OU (Spread)",
                        "orientation": "original",
                    },
                ]
                if independent_candidates >= 3
                else []
            ),
        ],
    }
    _write_json(immutable, contract)
    contract_path = active / "registered_research_rerun_contract.json"
    _write_json(contract_path, contract)
    parity_path = active / "wizard_mode_parity.csv"
    pd.DataFrame(
        [
            {
                "exact_mode": candidate["exact_mode"],
                "orientation": candidate["orientation"],
                "vendor_exact_mode_parity_proven": True,
            }
            for candidate in contract["registered_candidates"]
        ]
    ).to_csv(parity_path, index=False)
    cohort_id = "copulacohort-ready"
    cohort_path = (
        root / "data" / "research" / "wizard_copula_behavioral_cohorts" / f"{cohort_id}.json"
    )
    _write_json(cohort_path, {"cohort_id": cohort_id})
    scheduler_cycle = active / "wizard_proof_scheduler_receipts" / "ready-cycle.json"
    scheduler_cycle_relative = str(scheduler_cycle.relative_to(root))
    reservation = reserve_wizard_credit_lane(
        root=root,
        lane=PROOF_LANE,
        planned_credits=68,
        now=NOW,
    )
    reconciliation = reconcile_wizard_credit_lane(
        root=root,
        lane=PROOF_LANE,
        reservation_id=reservation.summary["reservation_id"],
        reconciliation_key=f"ready-cycle-{independent_candidates}",
        attempted_credits=4,
        completed_credits=4,
        external_requests=3,
        now=NOW,
    )
    capture_evidence = write_valid_capture_reconciliation_evidence(root)
    scheduler_payload = {
        **capture_evidence,
        "attempt_date_utc": NOW.date().isoformat(),
        "status": "COMPLETE_ACCEPTED_MODE_EVIDENCE",
        "queue_eligible": 2,
        "completed_after": 1,
        "responses_captured_after": 2,
        "formula_proofs_expected": 1,
        "accepted_mode_evidence_cells": 2,
        "copula_behavioral_expected_cells": 1,
        "copula_behavioral_cells_passed": 1,
        "copula_behavioral_provenance_cells": 1,
        "copula_behavioral_endpoint_calls": 2,
        "copula_behavioral_responses_captured_this_cycle": 2,
        "copula_behavioral_response_accounting_valid": True,
        "copula_behavioral_parity_proven": True,
        "copula_formula_parity_proven": False,
        "copula_cohort_receipt_id": cohort_id,
        "copula_cohort_receipt_path": str(cohort_path.relative_to(root)),
        "copula_cohort_receipt_sha256": _hash(cohort_path),
        "copula_cohort_receipt_valid": True,
        "parity_refresh_status": "PASS",
        "exact_mode_requests_attempted": 1,
        "exact_mode_responses_captured_this_cycle": 1,
        "proof_lane_attempted_credits": 4,
        "proof_lane_completed_credits": 4,
        "proof_lane_uncompleted_attempted_credits": 0,
        "credit_reconciliation_status": "PASS_RECONCILED",
        "credit_reservation_id": reservation.summary["reservation_id"],
        "credit_reservation_path": str(reservation.paths["reservation"].relative_to(root)),
        "credit_reconciliation_id": reconciliation.summary["reconciliation_id"],
        "credit_reconciliation_path": str(reconciliation.paths["reconciliation"].relative_to(root)),
        "candidate_promotion_authority": False,
        "order_submission_included": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
        "final_immutable_receipt_required": True,
    }
    scheduler_payload["receipt_id"] = (
        "wizardproof_"
        + sha256(
            json.dumps(scheduler_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:20]
    )
    _write_json(scheduler_cycle, scheduler_payload)
    immutable_scheduler_cycle = (
        root
        / "data"
        / "research"
        / "wizard_proof_scheduler_receipts"
        / f"{scheduler_payload['receipt_id']}.json"
    )
    immutable_scheduler_cycle.parent.mkdir(parents=True, exist_ok=True)
    immutable_scheduler_cycle.write_bytes(scheduler_cycle.read_bytes())
    ready_path = root / "data" / "research" / "registered_rerun_ready" / f"{contract_id}.json"
    ready_core = {
        "schema_version": "thewiz.corrective_registered_rerun_ready.v2",
        "contract_id": contract_id,
        "evidence_ready_at_utc": NOW.isoformat(),
        "parity_sha256": _hash(parity_path),
        "proof_scheduler_receipt_id": scheduler_payload["receipt_id"],
        "proof_scheduler_cycle_receipt_path": scheduler_cycle_relative,
        "proof_scheduler_cycle_receipt_sha256": _hash(scheduler_cycle),
        "proof_cells_completed": 2,
        "formula_proof_cells_completed": 1,
        "formula_proof_cells_expected": 1,
        "copula_behavioral_cells_completed": 1,
        "copula_behavioral_cells_expected": 1,
        "accepted_mode_evidence_cells": 2,
        "copula_cohort_receipt_id": cohort_id,
        "copula_cohort_receipt_path": str(cohort_path.relative_to(root)),
        "copula_cohort_receipt_sha256": _hash(cohort_path),
        "capture_reconciliation_id": capture_evidence["capture_reconciliation_id"],
        "capture_reconciliation_immutable_path": capture_evidence[
            "capture_reconciliation_immutable_path"
        ],
        "capture_reconciliation_immutable_sha256": capture_evidence[
            "capture_reconciliation_immutable_sha256"
        ],
        "capture_reconciliation_manifest_id": capture_evidence[
            "capture_reconciliation_manifest_id"
        ],
        "capture_reconciliation_manifest_path": capture_evidence[
            "capture_reconciliation_manifest_path"
        ],
        "capture_reconciliation_manifest_sha256": capture_evidence[
            "capture_reconciliation_manifest_sha256"
        ],
        "capture_reconciliation_required_calls": capture_evidence[
            "capture_reconciliation_required_calls"
        ],
        "capture_reconciliation_completed_calls": capture_evidence[
            "capture_reconciliation_completed_calls"
        ],
        "scheduled_research_rerun_authorized": True,
        "promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    ready = {
        **ready_core,
        "ready_receipt_id": "registeredready_"
        + sha256(
            json.dumps(ready_core, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:20],
    }
    _write_json(ready_path, ready)
    gate_path = active / "registered_research_rerun_gate.csv"
    pd.DataFrame(
        [
            {
                "semantic_hypothesis_id": candidate["semantic_hypothesis_id"],
                "source_experiment_id": candidate["source_experiment_id"],
                "pair_group_key": candidate["pair_group_key"],
                "pair": candidate["pair"],
                "exact_mode": candidate["exact_mode"],
                "orientation": candidate["orientation"],
                "pre_rerun_gate_ready": gate_status.startswith("READY"),
            }
            for candidate in contract["registered_candidates"]
        ]
    ).to_csv(gate_path, index=False)

    def gate_builder(**_):
        return CommandResult(
            paths={"contract": contract_path, "gate": gate_path, "ready_receipt": ready_path},
            summary={
                "status": gate_status,
                "registered_rerun_results_accounted": False,
            },
        )

    return contract, matrix_path, gate_builder


def _stage_runners(
    root: Path,
    calls: list[str],
    *,
    experiments: int = 2,
    unsafe: str = "",
    expected_pair_group_keys: tuple[str, ...] = ("dydx|daily|ETH|PYTH",),
    expected_fetch_funding: bool = False,
):
    runners = {}
    for stage_name in DEFAULT_STAGE_RUNNERS:

        def runner(*, root: Path, now: datetime, _stage=stage_name, **kwargs):
            calls.append(_stage)
            manifest = root / "snapshots" / _stage / "manifest.json"
            summary = {
                f"{_stage}_id": f"{_stage}-1",
                "experiments_accounted": experiments,
                "experiment_status_accounted": True,
                "promotion_authority": False,
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            }
            if _stage == "ou_optimal_overlay":
                summary.update({"source_rows": 10, "source_rows_accounted": 10})
            if _stage == "chain_validation":
                summary.update(
                    {
                        "experiment_authority_count": experiments,
                        "chain_status": "PASS",
                        "checks": 12,
                        "checks_passed": 12,
                        "stages_complete": 14,
                        "stages_expected": 14,
                    }
                )
            if unsafe == _stage:
                summary["testnet_order_authority"] = True
            _write_json(manifest, summary)
            paths = {"snapshot_manifest": manifest}
            if _stage == "cost_evidence":
                assert kwargs["pair_group_keys"] == expected_pair_group_keys
                assert kwargs["fetch_funding"] is expected_fetch_funding
                summary.update(
                    {
                        "funding_assets_fetched_from_network": (2 if expected_fetch_funding else 0),
                        "funding_assets_reused_from_cache": (0 if expected_fetch_funding else 2),
                    }
                )
                pairs = root / "snapshots" / _stage / "pairs.csv"
                pd.DataFrame(
                    [
                        {
                            "pair_group_key": pair_group_key,
                            "cost_acceptance_ready": True,
                        }
                        for pair_group_key in expected_pair_group_keys
                    ]
                ).to_csv(pairs, index=False)
                paths["pairs"] = pairs
            return CommandResult(paths=paths, summary=summary)

        runners[stage_name] = runner
    return runners


def _reconciler(
    root: Path,
    *,
    accounted: bool = True,
    accepted: bool = False,
    include_unregistered_survivor: bool = False,
):
    def reconcile(**_):
        active = root / "reports" / "active"
        contract = json.loads(
            (active / "registered_research_rerun_contract.json").read_text(encoding="utf-8")
        )
        registered_candidates = contract["registered_candidates"]
        chain = active / "current_wizard_hyperliquid_chain_validation_manifest.json"
        survivor = active / "final_1x_survivor_receipt.json"
        failure = active / "current_wizard_hyperliquid_failure_attribution.csv"
        failure_manifest = active / "current_wizard_hyperliquid_failure_attribution_manifest.json"
        _write_json(chain, {"chain_status": "PASS"})
        final_ids = (
            [candidate["source_experiment_id"] for candidate in registered_candidates]
            if accepted
            else []
        )
        if accepted and include_unregistered_survivor:
            final_ids.append("experiment-unregistered")
        final_pairs = (
            sorted(
                {
                    "-".join(sorted(candidate["pair_group_key"].split("|")[-2:]))
                    for candidate in registered_candidates
                }
            )
            if accepted
            else []
        )
        _write_json(
            survivor,
            {
                "schema_version": "thewiz.final_one_x_survivor_receipt.v1",
                "receipt_status": "PASS" if accepted else "ZERO_SURVIVORS",
                "acceptance_policy_id": "acceptance-1",
                "holdout_policy_id": "holdout-1",
                "independent_supporting_clusters": len(final_pairs),
                "independent_full_survivor_clusters": len(final_pairs),
                "independent_supporting_pairs": len(final_pairs),
                "independent_full_survivor_pairs": len(final_pairs),
                "final_one_x_survivors": len(final_ids),
                "final_experiment_ids": final_ids,
                "final_canonical_pairs": final_pairs,
                "blockers": [] if accepted else ["zero_final_survivors"],
                "thresholds_changed_after_results": False,
                "testnet_candidate_authority": accepted,
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            },
        )
        pd.DataFrame(
            [
                {
                    "experiment_id": candidate["source_experiment_id"],
                    "first_blocking_stage": "" if accepted else "walkforward",
                }
                for candidate in registered_candidates
            ]
        ).to_csv(failure, index=False)
        _write_json(
            failure_manifest,
            {"experiments_accounted": len(registered_candidates)},
        )
        conclusion = (
            root
            / "data"
            / "research"
            / "registered_rerun_conclusions"
            / "registeredrerun-test.json"
        )
        _write_json(
            conclusion,
            _conclusion_receipt(
                {
                    "contract_id": "registeredrerun-test",
                    "conclusion_status": (
                        "ACCEPTED_REGISTERED_SURVIVORS"
                        if accepted
                        else "CONCLUSIVE_ZERO_REGISTERED_SURVIVORS"
                    ),
                    "accepted_registered_hypotheses": (
                        len(registered_candidates) if accepted else 0
                    ),
                    "rejected_registered_hypotheses": (
                        0 if accepted else len(registered_candidates)
                    ),
                    "registered_hypotheses": len(registered_candidates),
                    "outcomes": {
                        candidate["semantic_hypothesis_id"]: (
                            "ACCEPTED_SURVIVOR" if accepted else "REJECTED_BY_FROZEN_GATES"
                        )
                        for candidate in registered_candidates
                    },
                    "result_chain_sha256": _hash(chain),
                    "final_survivor_receipt_sha256": _hash(survivor),
                    "failure_attribution_sha256": _hash(failure),
                    "failure_attribution_manifest_sha256": _hash(failure_manifest),
                    "promotion_authority": False,
                    "testnet_order_authority": False,
                    "live_trading_authorized": False,
                }
            ),
        )
        return CommandResult(
            paths={"registered_rerun_conclusion_receipt": conclusion},
            summary={
                "registered_rerun_results_accounted": accounted,
                "registered_rerun_gate_status": (
                    "PASS_REGISTERED_RERUN_ACCOUNTED" if accounted else "BLOCKED"
                ),
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            },
        )

    return reconcile


def _checkpoint(*, root: Path, **_):
    path = root / "checkpoint.csv"
    pd.DataFrame(
        [
            {
                "stage": 4,
                "status": "PASS",
                "evidence_progress": (
                    "active_contract_id=registeredrerun-test;"
                    "final_receipt_conclusion_bound=True;"
                    "stage4_terminal_outcome="
                    "ACCEPTED_VALID_INDEPENDENT_SURVIVORS"
                ),
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            }
        ]
    ).to_csv(path, index=False)
    return CommandResult(
        paths={"seven_stage_checkpoint": path},
        summary={
            "operational_acceptance_status": "BLOCKED",
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        },
    )


def test_registered_rerun_plan_does_not_invoke_the_chain(tmp_path):
    _, _, gate_builder = _setup(tmp_path)
    calls: list[str] = []
    result = run_registered_research_rerun(
        root=tmp_path,
        now=NOW,
        execute=False,
        gate_builder=gate_builder,
        stage_runners=_stage_runners(tmp_path, calls),
        reconciler=_reconciler(tmp_path),
        checkpoint_refresher=_checkpoint,
    )

    assert result.summary["status"] == "PLANNED"
    assert result.summary["testnet_order_authority"] is False
    assert calls == []


def test_registered_rerun_executes_full_frozen_chain_once(tmp_path):
    _, _, gate_builder = _setup(tmp_path)
    calls: list[str] = []
    kwargs = {
        "root": tmp_path,
        "now": NOW,
        "execute": True,
        "gate_builder": gate_builder,
        "stage_runners": _stage_runners(tmp_path, calls),
        "reconciler": _reconciler(tmp_path),
        "checkpoint_refresher": _checkpoint,
    }
    first = run_registered_research_rerun(**kwargs)
    second = run_registered_research_rerun(**kwargs)

    assert first.summary["status"] == "PASS_REGISTERED_RERUN_ACCOUNTED"
    assert second.summary["status"] == "ALREADY_COMPLETE"
    assert calls == list(DEFAULT_STAGE_RUNNERS)
    receipt = json.loads(first.paths["execution_receipt"].read_text())
    assert len(receipt["stages"]) == len(DEFAULT_STAGE_RUNNERS)
    assert receipt["frozen_family_replayed"] is True
    bridge = _registered_execution_survivor_bridge(root=tmp_path)
    assert bridge["execution_id"] == receipt["execution_id"]
    assert bridge["receipt"]["final_one_x_survivors"] == 0
    assert Path(bridge["path"]).is_file()
    assert receipt["order_submission_performed"] is False
    assert receipt["testnet_order_authority"] is False
    assert receipt["funding_refresh_requested"] is False
    assert receipt["funding_cost_evidence_stage_run"] is True
    assert receipt["funding_refresh_dispatched_to_cost_stage"] is False
    assert receipt["funding_assets_fetched_from_network"] == 0
    assert receipt["funding_assets_reused_from_cache"] == 2
    assert receipt["funding_refresh_outcome"] == "CACHE_REUSED"
    assert not any(
        (tmp_path / "reports" / "active" / name).exists()
        for name in (
            ".corrective_daily.lock",
            ".corrective_l2_capture.lock",
            ".corrective_registered_rerun.lock",
        )
    )


def test_registered_workspace_survives_board_rotation_and_rejects_tamper(
    tmp_path,
):
    contract, matrix, _ = _setup(tmp_path)
    active = tmp_path / "reports" / "active"
    handoff_dir = tmp_path / "reports" / "snapshots" / "handoff"
    handoff_manifest = {
        "handoff_id": "handoff-1",
        "artifacts": {
            "snapshot_experiments": str(
                (handoff_dir / "experiment_matrix.csv").relative_to(tmp_path)
            ),
        },
    }
    _write_json(handoff_dir / "manifest.json", handoff_manifest)
    source_bundle = (
        tmp_path
        / "data"
        / "research"
        / "registered_rerun_source_families"
        / contract["contract_id"]
    )
    source_bundle.mkdir(parents=True)
    source_matrix = source_bundle / "experiment_matrix.csv"
    source_matrix.write_bytes((handoff_dir / "experiment_matrix.csv").read_bytes())
    _write_json(
        source_bundle / "receipt.json",
        {
            "schema_version": "thewiz.registered_rerun_source_family.v1",
            "contract_id": contract["contract_id"],
            "source_family_sha256": contract["source_family_sha256"],
            "source_family_rows": contract["source_family_rows"],
            "recovered_from_path": str(
                (handoff_dir / "experiment_matrix.csv").relative_to(tmp_path)
            ),
            "promotion_authority": False,
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        },
    )
    for name, content in (
        ("asset_fetch_queue.csv", "asset\nETH\nPYTH\n"),
        ("pair_history_queue.csv", "pair_group_key\ndydx|daily|ETH|PYTH\n"),
        ("pair_detail_status.csv", "pair_group_key\ndydx|daily|ETH|PYTH\n"),
        ("validation.csv", "check,status\nsource,PASS\n"),
        ("summary.md", "# frozen\n"),
    ):
        (handoff_dir / name).write_text(content, encoding="utf-8")
    refresh_snapshot = tmp_path / "reports" / "snapshots" / "refresh" / "api_candidates.csv"
    refresh_snapshot.parent.mkdir(parents=True)
    refresh_snapshot.write_text("api_source_row_id\nrow-1\n", encoding="utf-8")
    inputs = handoff_dir / "inputs"
    inputs.mkdir()
    (inputs / "exhaustive_wizard_api_refresh_source_accounting.csv").write_text(
        "api_source_row_id\nrow-1\n", encoding="utf-8"
    )
    _write_json(
        inputs / "exhaustive_wizard_api_refresh_manifest.json",
        {
            "refresh_id": "refresh-1",
            "artifacts": {
                "snapshot_api_candidates": str(refresh_snapshot.relative_to(tmp_path)),
                "snapshot_source_accounting": str(
                    (inputs / "exhaustive_wizard_api_refresh_source_accounting.csv").relative_to(
                        tmp_path
                    )
                ),
            },
        },
    )
    history_dir = handoff_dir / "history_runs" / "history-1"
    history_dir.mkdir(parents=True)
    pair_results = history_dir / "pair_history_results.csv"
    pd.DataFrame(
        [
            {
                "pair_group_key": "dydx|daily|ETH|PYTH",
                "history_status": "READY_FOR_CANONICAL_1X_REPLAY",
            }
        ]
    ).to_csv(pair_results, index=False)
    asset_results = history_dir / "asset_history_results.csv"
    pd.DataFrame([{"asset": "ETH"}, {"asset": "PYTH"}]).to_csv(asset_results, index=False)
    for name in ("validation.csv", "summary.md"):
        (history_dir / name).write_text("ok\n", encoding="utf-8")
    for directory in ("assets", "pairs"):
        source = history_dir / directory / "source.csv"
        source.parent.mkdir()
        source.write_text("timestamp,value\n2026-08-01,1\n", encoding="utf-8")
    _write_json(
        history_dir / "manifest.json",
        {
            "history_run_id": "history-1",
            "artifacts": {
                "snapshot_asset_results": str(asset_results.relative_to(tmp_path)),
                "snapshot_pair_results": str(pair_results.relative_to(tmp_path)),
                "snapshot_validation": str((history_dir / "validation.csv").relative_to(tmp_path)),
                "snapshot_summary_md": str((history_dir / "summary.md").relative_to(tmp_path)),
                "snapshot_manifest": str((history_dir / "manifest.json").relative_to(tmp_path)),
            },
        },
    )
    funding_dir = history_dir / "cost_evidence" / "cost-1"
    funding_dir.mkdir(parents=True)
    funding_file = funding_dir / "funding.json"
    _write_json(funding_file, {"rows": []})
    funding_assets = funding_dir / "funding_asset_results.csv"
    pd.DataFrame(
        [
            {
                "asset": "ETH",
                "funding_path": str(funding_file.relative_to(tmp_path)),
            },
            {
                "asset": "PYTH",
                "funding_path": str(funding_file.relative_to(tmp_path)),
            },
        ]
    ).to_csv(funding_assets, index=False)
    pair_costs = funding_dir / "pair_cost_evidence.csv"
    pd.DataFrame(
        [
            {
                "pair_group_key": "dydx|daily|ETH|PYTH",
                "funding_status": "COMPLETE",
            }
        ]
    ).to_csv(pair_costs, index=False)
    _write_json(
        funding_dir / "manifest.json",
        {
            "as_of": NOW.isoformat(),
            "artifacts": {
                "snapshot_assets": str(funding_assets.relative_to(tmp_path)),
                "snapshot_pairs": str(pair_costs.relative_to(tmp_path)),
            },
        },
    )
    cost_artifacts = publish_valid_pair_cost_bundle(
        root=tmp_path,
        now=NOW,
        specs=[
            {
                "pair_group_key": "dydx|daily|ETH|PYTH",
                "pair": "ETH-USD-PYTH-USD",
                "asset_x": "ETH",
                "asset_y": "PYTH",
                "registered_contract_id": contract["contract_id"],
                "registered_contract_candidate": True,
            }
        ],
    )
    bundle_receipt = cost_artifacts["bundle"]
    _write_json(
        active / "corrective_l2_capture_status.json",
        {
            "pair_cost_bundle_manifest_path": str(bundle_receipt.relative_to(tmp_path)),
            "pair_cost_bundle_manifest_sha256": _hash(bundle_receipt),
        },
    )
    workspace, receipt = _prepare_registered_workspace(
        root=tmp_path,
        contract=contract,
        execution_id="execution-1",
        ready_path=(
            tmp_path / "data" / "research" / "registered_rerun_ready" / "registeredrerun-test.json"
        ),
        pair_keys=("dydx|daily|ETH|PYTH",),
    )
    frozen = workspace / "reports" / "active" / matrix.name
    original_hash = _hash(frozen)
    matrix.write_text("experiment_id\nrotated-board\n", encoding="utf-8")
    workspace_again, _ = _prepare_registered_workspace(
        root=tmp_path,
        contract=contract,
        execution_id="execution-1",
        ready_path=(
            tmp_path / "data" / "research" / "registered_rerun_ready" / "registeredrerun-test.json"
        ),
        pair_keys=("dydx|daily|ETH|PYTH",),
    )
    assert workspace_again == workspace
    assert _hash(frozen) == original_hash
    assert receipt.is_file()
    frozen.write_text("experiment_id\ntampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="workspace input changed"):
        _prepare_registered_workspace(
            root=tmp_path,
            contract=contract,
            execution_id="execution-1",
            ready_path=(
                tmp_path
                / "data"
                / "research"
                / "registered_rerun_ready"
                / "registeredrerun-test.json"
            ),
            pair_keys=("dydx|daily|ETH|PYTH",),
        )


def test_registered_rerun_blocked_gate_invokes_no_stage(tmp_path):
    _, _, gate_builder = _setup(tmp_path, gate_status="BLOCKED_VENDOR_PARITY")
    calls: list[str] = []
    result = run_registered_research_rerun(
        root=tmp_path,
        now=NOW,
        execute=True,
        gate_builder=gate_builder,
        stage_runners=_stage_runners(tmp_path, calls),
        checkpoint_refresher=_checkpoint,
    )

    assert result.summary["status"] == "BLOCKED_GATE"
    assert "BLOCKED_VENDOR_PARITY" in result.summary["blocker"]
    assert calls == []
    learning_status = json.loads(
        Path(result.paths["registered_learning_status"]).read_text(encoding="utf-8")
    )
    assert learning_status["status"] == "BLOCKED_STAGE4"
    assert learning_status["stage5_research_gate_pass"] is False
    assert learning_status["order_submission_performed"] is False
    assert learning_status["testnet_order_authority"] is False
    assert learning_status["live_trading_authorized"] is False


def test_registered_rerun_blocked_gate_records_undispatched_funding_request(
    tmp_path,
):
    _, _, gate_builder = _setup(tmp_path, gate_status="BLOCKED_VENDOR_PARITY")
    calls: list[str] = []
    result = run_registered_research_rerun(
        root=tmp_path,
        now=NOW,
        execute=True,
        fetch_funding=True,
        gate_builder=gate_builder,
        stage_runners=_stage_runners(tmp_path, calls),
        checkpoint_refresher=_checkpoint,
    )

    assert result.summary["status"] == "BLOCKED_GATE"
    assert result.summary["funding_refresh_requested_for_invocation"] is True
    assert result.summary["funding_refresh_dispatched_to_cost_stage_for_invocation"] is False
    assert result.summary["funding_cost_evidence_stage_run_for_invocation"] is False
    assert result.summary["funding_assets_fetched_from_network_for_invocation"] == 0
    assert result.summary["funding_refresh_outcome_for_invocation"] == "NOT_REACHED"
    assert calls == []


def test_registered_rerun_binds_funding_refresh_evidence_to_receipt(tmp_path):
    _, _, gate_builder = _setup(tmp_path)
    calls: list[str] = []
    result = run_registered_research_rerun(
        root=tmp_path,
        now=NOW,
        execute=True,
        fetch_funding=True,
        gate_builder=gate_builder,
        stage_runners=_stage_runners(
            tmp_path,
            calls,
            expected_fetch_funding=True,
        ),
        reconciler=_reconciler(tmp_path),
        checkpoint_refresher=_checkpoint,
    )

    assert result.summary["status"] == "PASS_REGISTERED_RERUN_ACCOUNTED"
    assert result.summary["funding_refresh_requested_for_invocation"] is True
    assert result.summary["funding_refresh_dispatched_to_cost_stage_for_invocation"] is True
    assert result.summary["funding_assets_fetched_from_network_for_invocation"] == 2
    assert result.summary["funding_refresh_outcome_for_invocation"] == "NETWORK_FETCHED"
    receipt = json.loads(result.paths["execution_receipt"].read_text())
    assert receipt["funding_refresh_requested"] is True
    assert receipt["funding_refresh_dispatched_to_cost_stage"] is True
    assert receipt["funding_assets_fetched_from_network"] == 2
    assert receipt["funding_assets_reused_from_cache"] == 0
    assert receipt["funding_refresh_outcome"] == "NETWORK_FETCHED"
    cost_stage = next(row for row in receipt["stages"] if row["stage"] == "cost_evidence")
    assert cost_stage["funding_assets_fetched_from_network"] == 2
    assert cost_stage["funding_refresh_outcome"] == "NETWORK_FETCHED"


def test_registered_rerun_rejects_family_drift_between_stages(tmp_path):
    _, matrix, gate_builder = _setup(tmp_path)
    calls: list[str] = []
    runners = _stage_runners(tmp_path, calls)
    original = runners["canonical_replay"]

    def drift(**kwargs):
        result = original(**kwargs)
        matrix.write_text(matrix.read_text() + "\n", encoding="utf-8")
        return result

    runners["canonical_replay"] = drift
    with pytest.raises(ValueError, match="source family hash drifted"):
        run_registered_research_rerun(
            root=tmp_path,
            now=NOW,
            execute=True,
            gate_builder=gate_builder,
            stage_runners=runners,
            reconciler=_reconciler(tmp_path),
            checkpoint_refresher=_checkpoint,
        )
    assert calls == ["canonical_replay"]


def test_registered_rerun_rejects_swapped_handoff_family(tmp_path):
    _, _, gate_builder = _setup(tmp_path)
    handoff_path = (
        tmp_path / "reports" / "active" / "current_wizard_hyperliquid_handoff_manifest.json"
    )
    handoff = json.loads(handoff_path.read_text())
    snapshot = tmp_path / handoff["artifacts"]["snapshot_experiments"]
    snapshot.write_text("experiment_id\nforged\n", encoding="utf-8")

    with pytest.raises(ValueError, match="immutable handoff family mismatch"):
        run_registered_research_rerun(
            root=tmp_path,
            now=NOW,
            execute=True,
            gate_builder=gate_builder,
            stage_runners=_stage_runners(tmp_path, []),
            reconciler=_reconciler(tmp_path),
            checkpoint_refresher=_checkpoint,
        )


def test_registered_rerun_rejects_partial_accounting_and_authority(tmp_path):
    _, _, gate_builder = _setup(tmp_path)
    with pytest.raises(ValueError, match="accounted 1 of 2"):
        run_registered_research_rerun(
            root=tmp_path,
            now=NOW,
            execute=True,
            gate_builder=gate_builder,
            stage_runners=_stage_runners(tmp_path, [], experiments=1),
            reconciler=_reconciler(tmp_path),
            checkpoint_refresher=_checkpoint,
        )

    other = tmp_path / "other"
    _, _, other_gate = _setup(other)
    with pytest.raises(ValueError, match="unexpectedly acquired authority"):
        run_registered_research_rerun(
            root=other,
            now=NOW,
            execute=True,
            gate_builder=other_gate,
            stage_runners=_stage_runners(other, [], unsafe="walkforward"),
            reconciler=_reconciler(other),
            checkpoint_refresher=_checkpoint,
        )


def test_registered_rerun_rejects_forged_conclusion_accounting(tmp_path):
    _, _, gate_builder = _setup(tmp_path)
    reconcile = _reconciler(tmp_path, accepted=True)

    def forged_reconciler(**kwargs):
        result = reconcile(**kwargs)
        path = Path(result.paths["registered_rerun_conclusion_receipt"])
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["accepted_registered_hypotheses"] = 0
        payload["rejected_registered_hypotheses"] = 1
        payload.pop("conclusion_id")
        _write_json(path, _conclusion_receipt(payload))
        return result

    with pytest.raises(ValueError, match="accepted count mismatch"):
        run_registered_research_rerun(
            root=tmp_path,
            now=NOW,
            execute=True,
            gate_builder=gate_builder,
            stage_runners=_stage_runners(tmp_path, []),
            reconciler=forged_reconciler,
            checkpoint_refresher=_checkpoint,
        )


def test_registered_rerun_rejects_conclusion_source_hash_mismatch(tmp_path):
    _, _, gate_builder = _setup(tmp_path)
    reconcile = _reconciler(tmp_path, accepted=True)

    def source_tampering_reconciler(**kwargs):
        result = reconcile(**kwargs)
        chain = (
            tmp_path
            / "reports"
            / "active"
            / "current_wizard_hyperliquid_chain_validation_manifest.json"
        )
        chain.write_bytes(chain.read_bytes() + b"\n")
        return result

    with pytest.raises(ValueError, match="source binding mismatch"):
        run_registered_research_rerun(
            root=tmp_path,
            now=NOW,
            execute=True,
            gate_builder=gate_builder,
            stage_runners=_stage_runners(tmp_path, []),
            reconciler=source_tampering_reconciler,
            checkpoint_refresher=_checkpoint,
        )


def test_registered_rerun_detects_tampered_immutable_receipt(tmp_path):
    _, _, gate_builder = _setup(tmp_path)
    calls: list[str] = []
    kwargs = {
        "root": tmp_path,
        "now": NOW,
        "execute": True,
        "gate_builder": gate_builder,
        "stage_runners": _stage_runners(tmp_path, calls),
        "reconciler": _reconciler(tmp_path),
        "checkpoint_refresher": _checkpoint,
    }
    result = run_registered_research_rerun(**kwargs)
    receipt = json.loads(result.paths["execution_receipt"].read_text())
    receipt["source_family_rows"] = 999
    _write_json(result.paths["execution_receipt"], receipt)

    with pytest.raises(ValueError, match="execution receipt hash mismatch"):
        run_registered_research_rerun(**kwargs)


def test_registered_rerun_rejects_tampered_ready_receipt_before_stages(
    tmp_path,
):
    contract, _, gate_builder = _setup(tmp_path)
    ready_path = (
        tmp_path
        / "data"
        / "research"
        / "registered_rerun_ready"
        / f"{contract['contract_id']}.json"
    )
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    ready["evidence_ready_at_utc"] = "2026-01-01T00:00:00+00:00"
    _write_json(ready_path, ready)
    calls: list[str] = []

    with pytest.raises(ValueError, match="content identity mismatch"):
        run_registered_research_rerun(
            root=tmp_path,
            now=NOW,
            execute=True,
            gate_builder=gate_builder,
            stage_runners=_stage_runners(tmp_path, calls),
            reconciler=_reconciler(tmp_path),
            checkpoint_refresher=_checkpoint,
        )

    assert calls == []


def test_registered_rerun_rejects_tampered_proof_cycle_before_stages(
    tmp_path,
):
    contract, _, gate_builder = _setup(tmp_path)
    ready_path = (
        tmp_path
        / "data"
        / "research"
        / "registered_rerun_ready"
        / f"{contract['contract_id']}.json"
    )
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    cycle_path = tmp_path / ready["proof_scheduler_cycle_receipt_path"]
    cycle_path.write_bytes(cycle_path.read_bytes() + b"\n")
    calls: list[str] = []

    with pytest.raises(ValueError, match="scheduler binding mismatch"):
        run_registered_research_rerun(
            root=tmp_path,
            now=NOW,
            execute=True,
            gate_builder=gate_builder,
            stage_runners=_stage_runners(tmp_path, calls),
            reconciler=_reconciler(tmp_path),
            checkpoint_refresher=_checkpoint,
        )

    assert calls == []


def test_registered_rerun_rejects_mutable_only_proof_cycle_before_stages(
    tmp_path,
):
    contract, _, gate_builder = _setup(tmp_path)
    ready_path = (
        tmp_path
        / "data"
        / "research"
        / "registered_rerun_ready"
        / f"{contract['contract_id']}.json"
    )
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    immutable_cycle = (
        tmp_path
        / "data"
        / "research"
        / "wizard_proof_scheduler_receipts"
        / f"{ready['proof_scheduler_receipt_id']}.json"
    )
    immutable_cycle.unlink()
    calls: list[str] = []

    with pytest.raises(
        ValueError,
        match="scheduler evidence is incomplete",
    ):
        run_registered_research_rerun(
            root=tmp_path,
            now=NOW,
            execute=True,
            gate_builder=gate_builder,
            stage_runners=_stage_runners(tmp_path, calls),
            reconciler=_reconciler(tmp_path),
            checkpoint_refresher=_checkpoint,
        )

    assert calls == []


def test_registered_rerun_detects_tampered_stage_snapshot_artifact(tmp_path):
    _, _, gate_builder = _setup(tmp_path)
    calls: list[str] = []
    kwargs = {
        "root": tmp_path,
        "now": NOW,
        "execute": True,
        "gate_builder": gate_builder,
        "stage_runners": _stage_runners(tmp_path, calls),
        "reconciler": _reconciler(tmp_path),
        "checkpoint_refresher": _checkpoint,
    }
    result = run_registered_research_rerun(**kwargs)
    receipt = json.loads(result.paths["execution_receipt"].read_text())
    first_artifact = next(iter(receipt["stages"][0]["immutable_artifact_hashes"]))
    artifact_path = tmp_path / first_artifact
    artifact_path.write_text(artifact_path.read_text() + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="stage (manifest|artifact) hash mismatch"):
        run_registered_research_rerun(**kwargs)


def test_registered_execution_rejects_tampered_bound_stage4_survivor_receipt(
    tmp_path,
):
    _, _, gate_builder = _setup(tmp_path, independent_candidates=3)
    kwargs = {
        "root": tmp_path,
        "now": NOW,
        "execute": True,
        "gate_builder": gate_builder,
        "stage_runners": _stage_runners(
            tmp_path,
            [],
            experiments=3,
            expected_pair_group_keys=INDEPENDENT_PAIR_KEYS,
        ),
        "reconciler": _reconciler(tmp_path, accepted=True),
        "checkpoint_refresher": _checkpoint,
    }
    first = run_registered_research_rerun(**kwargs)
    receipt = json.loads(first.paths["execution_receipt"].read_text())
    survivor_path = tmp_path / receipt["stage4_survivor_receipt_path"]
    survivor_path.write_text(
        survivor_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Stage 4 survivor receipt hash mismatch"):
        run_registered_research_rerun(**kwargs)


def test_registered_execution_rejects_unregistered_final_survivor(tmp_path):
    _, _, gate_builder = _setup(tmp_path, independent_candidates=3)

    with pytest.raises(
        ValueError,
        match="not the exact accepted registered cohort",
    ):
        run_registered_research_rerun(
            root=tmp_path,
            now=NOW,
            execute=True,
            gate_builder=gate_builder,
            stage_runners=_stage_runners(
                tmp_path,
                [],
                experiments=3,
                expected_pair_group_keys=INDEPENDENT_PAIR_KEYS,
            ),
            reconciler=_reconciler(
                tmp_path,
                accepted=True,
                include_unregistered_survivor=True,
            ),
            checkpoint_refresher=_checkpoint,
        )


def test_registered_execution_rejects_one_pair_claiming_independent_acceptance(
    tmp_path,
):
    _, _, gate_builder = _setup(tmp_path)

    with pytest.raises(
        ValueError,
        match="Stage 4 survivor receipt acceptance is invalid",
    ):
        run_registered_research_rerun(
            root=tmp_path,
            now=NOW,
            execute=True,
            gate_builder=gate_builder,
            stage_runners=_stage_runners(tmp_path, []),
            reconciler=_reconciler(tmp_path, accepted=True),
            checkpoint_refresher=_checkpoint,
        )


def test_registered_rerun_hands_receipt_to_learning_only_after_unlock(tmp_path):
    _, _, gate_builder = _setup(tmp_path, independent_candidates=3)
    learning_calls: list[str] = []

    def learning_runner(*, execution_receipt_path: Path, **_):
        learning_calls.append(str(execution_receipt_path))
        assert execution_receipt_path.is_file()
        assert not any(
            (tmp_path / "reports" / "active" / name).exists()
            for name in (
                ".corrective_daily.lock",
                ".corrective_l2_capture.lock",
                ".corrective_registered_rerun.lock",
            )
        )
        status = tmp_path / "reports" / "active" / "registered_learning_status.json"
        _write_json(status, {"status": "REJECTED_RESEARCH_LEARNING_GATES"})
        return CommandResult(
            paths={"status": status},
            summary={
                "status": "REJECTED_RESEARCH_LEARNING_GATES",
                "stage5_research_gate_pass": False,
                "blocker": "model_or_rl_oos_gate_failed",
                "promotion_authority": False,
                "testnet_candidate_authority": False,
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            },
        )

    result = run_registered_research_rerun(
        root=tmp_path,
        now=NOW,
        execute=True,
        gate_builder=gate_builder,
        stage_runners=_stage_runners(
            tmp_path,
            [],
            experiments=3,
            expected_pair_group_keys=INDEPENDENT_PAIR_KEYS,
        ),
        reconciler=_reconciler(tmp_path, accepted=True),
        checkpoint_refresher=_checkpoint,
        learning_runner=learning_runner,
    )

    assert len(learning_calls) == 1
    assert result.summary["learning_handoff_status"] == ("REJECTED_RESEARCH_LEARNING_GATES")
    assert result.summary["stage5_research_gate_pass"] is False
    assert result.summary["testnet_order_authority"] is False
    assert "registered_learning_status" in result.paths


def test_partial_accepted_generation_cannot_launch_learning_before_whole_cohort(
    tmp_path,
):
    _, _, gate_builder = _setup(tmp_path, independent_candidates=3)
    learning_calls = []

    def incomplete_checkpoint(*, root: Path, **_):
        path = root / "checkpoint-incomplete.csv"
        pd.DataFrame(
            [
                {
                    "stage": 4,
                    "status": "IN_PROGRESS",
                    "evidence_progress": ("stage4_terminal_outcome=INCOMPLETE_CURRENT_FAMILY"),
                    "next_action": (
                        "advance_registered_rerun_to_unaccounted_current_family_cohort"
                    ),
                    "testnet_order_authority": False,
                    "live_trading_authorized": False,
                }
            ]
        ).to_csv(path, index=False)
        return CommandResult(
            paths={"seven_stage_checkpoint": path},
            summary={
                "operational_acceptance_status": "BLOCKED",
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            },
        )

    result = run_registered_research_rerun(
        root=tmp_path,
        now=NOW,
        execute=True,
        gate_builder=gate_builder,
        stage_runners=_stage_runners(
            tmp_path,
            [],
            experiments=3,
            expected_pair_group_keys=INDEPENDENT_PAIR_KEYS,
        ),
        reconciler=_reconciler(tmp_path, accepted=True),
        checkpoint_refresher=incomplete_checkpoint,
        learning_runner=lambda **kwargs: learning_calls.append(kwargs),
    )

    assert learning_calls == []
    assert result.summary["learning_handoff_status"] == ("BLOCKED_STAGE4_CURRENT_FAMILY_INCOMPLETE")
    assert result.summary["learning_handoff_blocker"] == (
        "registered_stage4_whole_cohort_or_independent_breadth_incomplete"
    )
    assert result.summary["stage5_research_gate_pass"] is False
    assert result.summary["testnet_order_authority"] is False


def test_zero_survivor_conclusion_does_not_launch_learning(tmp_path):
    _, _, gate_builder = _setup(tmp_path)
    learning_calls = []

    result = run_registered_research_rerun(
        root=tmp_path,
        now=NOW,
        execute=True,
        gate_builder=gate_builder,
        stage_runners=_stage_runners(tmp_path, []),
        reconciler=_reconciler(tmp_path, accepted=False),
        checkpoint_refresher=_checkpoint,
        learning_runner=lambda **kwargs: learning_calls.append(kwargs),
    )

    assert learning_calls == []
    assert result.summary["learning_handoff_status"] == ("BLOCKED_STAGE4_NO_ACCEPTED_SURVIVOR")
    assert result.summary["stage5_research_gate_pass"] is False
    learning_status = json.loads(
        Path(result.paths["registered_learning_status"]).read_text(encoding="utf-8")
    )
    assert learning_status["status"] == ("BLOCKED_STAGE4_NO_ACCEPTED_SURVIVOR")
    assert learning_status["accepted_registered_hypotheses"] == 0
    assert result.summary["testnet_order_authority"] is False


def test_registered_rerun_learning_handoff_fails_closed(tmp_path):
    _, _, gate_builder = _setup(tmp_path, independent_candidates=3)

    def unsafe_learning_runner(**_):
        return CommandResult(
            paths={},
            summary={
                "status": "UNSAFE",
                "testnet_candidate_authority": True,
            },
        )

    result = run_registered_research_rerun(
        root=tmp_path,
        now=NOW,
        execute=True,
        gate_builder=gate_builder,
        stage_runners=_stage_runners(
            tmp_path,
            [],
            experiments=3,
            expected_pair_group_keys=INDEPENDENT_PAIR_KEYS,
        ),
        reconciler=_reconciler(tmp_path, accepted=True),
        checkpoint_refresher=_checkpoint,
        learning_runner=unsafe_learning_runner,
    )

    assert result.summary["learning_handoff_status"] == "HANDOFF_FAILED"
    assert "testnet_candidate_authority" in result.summary["learning_handoff_blocker"]
    assert result.summary["stage5_research_gate_pass"] is False
    assert result.summary["testnet_order_authority"] is False
