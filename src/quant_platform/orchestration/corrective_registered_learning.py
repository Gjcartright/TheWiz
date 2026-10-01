"""Build and evaluate the registered exact-mode learning cohort."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_platform.active_pipeline import (
    CommandResult,
    _filter_predictions_to_model,
    _filter_predictions_to_untouched_evaluation,
    _global_label_purge_proven,
    _leakage_audit,
    _model_gain_concentration,
    _model_gated_acceptance,
    _model_gated_comparison,
    _model_pair_concentration,
    _model_selection_isolation_proven,
    _score_bucket_report,
    _score_buckets_monotonic,
    _stage_trade_dataset_candidate,
    promote_trade_dataset,
    run_model_gated_backtest,
    train_trade_gate,
)
from quant_platform.ml_filter import (
    GLOBAL_PURGED_SPLIT_SCHEME,
    MINIMUM_TRAINING_TAKE_RATE,
    MODEL_SELECTION_ISOLATION_SCHEME,
    THRESHOLD_CALIBRATION_SCHEME,
    model_selection_leaderboard,
    walkforward_prediction_membership_matches,
)
from quant_platform.orchestration.corrective_agent_governance import (
    build_corrective_agent_governance,
)
from quant_platform.orchestration.corrective_daily_scheduler import _acquire_lock
from quant_platform.orchestration.corrective_redaction import (
    safe_exception_code,
    safe_validation_exception_code,
)
from quant_platform.orchestration.corrective_registered_learning_protocol import (
    validate_registered_stage5_protocol,
)
from quant_platform.orchestration.corrective_registered_rerun_executor import (
    _validate_execution_receipt,
)
from quant_platform.orchestration.corrective_runtime import promote_staged_file
from quant_platform.rl.rl_acceptance import return_summary, rl_acceptance_report
from quant_platform.rl.rl_backtest import run_rl_research
from quant_platform.rl.rl_learning_agent import _chronological_rl_partitions
from quant_platform.wizard_mode_replay import CANONICAL_WIZARD_MODES

ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "thewiz.corrective_registered_learning.v1"
SURVIVOR_SUPPORT_POLICY_SCHEMA = "thewiz.registered_survivor_oos_support_policy.v4"
LOCK_TIMEOUT_SECONDS = 4 * 60 * 60
REQUIRED_ORIENTATIONS = {"original", "reverse"}
REQUIRED_LEARNING_EVIDENCE_ROLES = {
    "stage4_execution_receipt",
    "dataset_receipt",
    "active_dataset_pointer",
    "model_metrics",
    "model_lineage",
    "model_predictions",
    "model_selection_leaderboard",
    "model_backtest",
    "model_acceptance",
    "model_score_buckets",
    "model_pair_concentration",
    "model_gain_concentration",
    "model_failure_attribution",
    "model_pair_support",
    "registered_survivor_support_policy",
    "registered_stage5_protocol",
    "registered_survivor_attribution",
    "rl_acceptance",
    "rl_evaluation",
    "rl_execution_backtest",
    "registered_survivor_rl_attribution",
    "rl_split_audit",
    "rl_leakage_audit",
    "rl_feature_schema",
    "rl_lineage",
    "model_authority",
}
REQUIRED_CAUSAL_FEATURES = {
    "feature_timestamp",
    "mode_metric",
    "spread",
    "spread_slope",
    "realized_volatility_percentile",
    "correlation",
    "hedge_ratio",
    "hedge_ratio_stability",
    "funding_bps_per_day",
    "liquidity_score",
    "feature_known_at_or_before_entry",
    "feature_uses_future_data",
    "uses_dashboard_hindsight",
}

ResearchRunner = Callable[..., CommandResult]


def build_registered_exact_mode_trade_dataset(
    *,
    root: Path = ROOT,
    execution_receipt_path: Path | None = None,
) -> CommandResult:
    """Stage a strict-cost, causal cohort from one immutable Stage 4 execution."""

    receipt_path, receipt, contract = _registered_execution(
        root=root, execution_receipt_path=execution_receipt_path
    )
    if not _stage4_has_accepted_survivor(receipt):
        raise ValueError("registered learning requires an accepted Stage 4 survivor")
    conclusion_path, hypothesis_lineage = _registered_hypothesis_lineage(
        root=root,
        receipt=receipt,
        contract=contract,
    )
    stages = {
        str(row.get("stage", "")): row for row in receipt.get("stages", []) if isinstance(row, dict)
    }
    required_stages = {"walkforward", "cost_evidence", "regime_attribution"}
    missing_stages = sorted(required_stages - set(stages))
    if missing_stages:
        raise ValueError("registered learning stages missing: " + ";".join(missing_stages))
    walk_manifest = _stage_manifest(root, stages["walkforward"])
    cost_manifest = _stage_manifest(root, stages["cost_evidence"])
    regime_manifest = _stage_manifest(root, stages["regime_attribution"])
    expected_experiments = int(contract.get("source_family_rows", 0) or 0)
    for name in required_stages:
        if int(stages[name].get("experiments_accounted", 0) or 0) != expected_experiments:
            raise ValueError(f"registered learning {name} family accounting mismatch")

    trades_path = _bound_manifest_artifact(
        root, stages["walkforward"], walk_manifest, "snapshot_trades"
    )
    bars_path = _bound_manifest_artifact(
        root, stages["walkforward"], walk_manifest, "snapshot_bars"
    )
    costs_path = _bound_manifest_artifact(
        root, stages["cost_evidence"], cost_manifest, "snapshot_pairs"
    )
    regime_trades_path = _bound_manifest_artifact(
        root, stages["regime_attribution"], regime_manifest, "snapshot_trades"
    )
    trades = _read_csv(trades_path)
    bars = _read_csv(bars_path)
    costs = _read_csv(costs_path)
    regime_trades = _read_csv(regime_trades_path)
    _validate_walkforward_ledgers(trades=trades, bars=bars)

    strict_cost = costs.get("cost_acceptance_ready", pd.Series(False, index=costs.index)).map(
        _truthy
    )
    strict_costs = costs.loc[strict_cost].copy()
    strict_pair_keys = set(strict_costs.get("pair_group_key", pd.Series(dtype=str)).astype(str)) - {
        ""
    }
    cohort = trades.loc[trades["pair_group_key"].astype(str).isin(strict_pair_keys)].copy()
    if cohort.empty:
        raise ValueError("registered learning has no strict-cost walk-forward trades")
    cohort = cohort.merge(
        hypothesis_lineage[
            [
                "source_experiment_id",
                "registered_semantic_hypothesis_id",
                "registered_hypothesis_outcome",
                "registered_candidate",
                "accepted_stage4_survivor",
                "registered_pair_group_key",
                "registered_pair",
                "registered_exact_mode",
                "registered_orientation",
            ]
        ],
        left_on="experiment_id",
        right_on="source_experiment_id",
        how="left",
        validate="many_to_one",
    ).drop(columns=["source_experiment_id"])
    cohort["registered_semantic_hypothesis_id"] = (
        cohort["registered_semantic_hypothesis_id"].fillna("").astype(str)
    )
    cohort["registered_hypothesis_outcome"] = (
        cohort["registered_hypothesis_outcome"].fillna("UNREGISTERED_FULL_FAMILY").astype(str)
    )
    cohort["registered_candidate"] = cohort["registered_candidate"].fillna(False).map(_truthy)
    cohort["accepted_stage4_survivor"] = (
        cohort["accepted_stage4_survivor"].fillna(False).map(_truthy)
    )
    registered_identity_fields = {
        "pair_group_key": "registered_pair_group_key",
        "pair": "registered_pair",
        "exact_mode": "registered_exact_mode",
        "orientation": "registered_orientation",
    }
    registered_rows = cohort["registered_candidate"]
    identity_mismatch = pd.Series(False, index=cohort.index)
    for observed_field, registered_field in registered_identity_fields.items():
        observed = cohort[observed_field].fillna("").astype(str).str.strip()
        expected = cohort[registered_field].fillna("").astype(str).str.strip()
        identity_mismatch |= registered_rows & observed.ne(expected)
    if identity_mismatch.any():
        raise ValueError("registered learning trade identity does not match Stage 4 contract")
    cohort = cohort.drop(columns=list(registered_identity_fields.values()))
    expected_accepted_ids = set(
        hypothesis_lineage.loc[
            hypothesis_lineage["accepted_stage4_survivor"].map(_truthy),
            "registered_semantic_hypothesis_id",
        ].astype(str)
    )
    represented_accepted_ids = set(
        cohort.loc[
            cohort["accepted_stage4_survivor"],
            "registered_semantic_hypothesis_id",
        ].astype(str)
    )
    if not expected_accepted_ids or represented_accepted_ids != expected_accepted_ids:
        raise ValueError(
            "registered learning accepted Stage 4 survivor trade support is incomplete"
        )
    modes = set(cohort["exact_mode"].astype(str))
    orientations = set(cohort["orientation"].astype(str))
    if modes != set(CANONICAL_WIZARD_MODES):
        raise ValueError("registered learning exact-mode family is incomplete")
    if orientations != REQUIRED_ORIENTATIONS:
        raise ValueError("registered learning orientation family is incomplete")
    expected_mode_orientation_cells = {
        (mode, orientation)
        for mode in CANONICAL_WIZARD_MODES
        for orientation in REQUIRED_ORIENTATIONS
    }
    observed_mode_orientation_cells = set(
        cohort[["exact_mode", "orientation"]].astype(str).itertuples(index=False, name=None)
    )
    if observed_mode_orientation_cells != expected_mode_orientation_cells:
        raise ValueError("registered learning mode-orientation family is incomplete")
    if not cohort["feature_known_at_or_before_entry"].map(_truthy).all():
        raise ValueError("registered learning entry features are not proven causal")
    if cohort["feature_uses_future_data"].map(_truthy).any():
        raise ValueError("registered learning entry features use future data")
    if cohort["uses_dashboard_hindsight"].map(_truthy).any():
        raise ValueError("registered learning entry features use dashboard hindsight")

    cohort = _join_causal_regimes(cohort, regime_trades)
    cohort = _materialize_learning_rows(
        cohort=cohort,
        bars=bars,
        receipt=receipt,
        evidence_paths=(
            trades_path,
            bars_path,
            costs_path,
            regime_trades_path,
            conclusion_path,
        ),
        root=root,
    )
    leakage = _leakage_audit(cohort)
    causal_blocked = (
        ~cohort["feature_known_at_or_before_entry"].map(_truthy)
        | cohort["feature_uses_future_data"].map(_truthy)
        | cohort["uses_dashboard_hindsight"].map(_truthy)
    )
    leakage.loc[causal_blocked, "leakage_blocker"] = "registered_causal_feature_contract_failed"
    if leakage["leakage_blocker"].fillna("").astype(str).str.strip().ne("").any():
        raise ValueError("registered learning leakage audit failed")

    selected_pair_keys = sorted(set(cohort["pair_group_key"].astype(str)))
    selected_costs = strict_costs.loc[
        strict_costs["pair_group_key"].astype(str).isin(selected_pair_keys)
    ].copy()
    if len(selected_costs) != len(selected_pair_keys):
        raise ValueError("registered learning strict-cost pair coverage mismatch")
    source_selection = pd.DataFrame(
        [
            {
                "pair_group_key": row.get("pair_group_key", ""),
                "pair": row.get("pair", ""),
                "source_venue": "hyperliquid",
                "timeframe": row.get("hyperliquid_interval", ""),
                "source_path": _relative(costs_path, root),
                "selection_status": "SELECTED_CANONICAL",
                "strict_observed_cost_ready": True,
                "strict_cost_model_path": _relative(costs_path, root),
                "cost_model_id": row.get("cost_model_id", ""),
                "promotion_authority": False,
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            }
            for row in selected_costs.to_dict("records")
        ]
    )
    registry_audit = pd.DataFrame(
        [
            {
                "registry_path": _relative(path, root),
                "registry_sha256": _file_hash(path),
                "registry_status": "HASH_BOUND_REGISTERED_STAGE4_SNAPSHOT",
                "eligible_for_candidate_dataset": True,
                "blocker": "",
                "promotion_authority": False,
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            }
            for path in (trades_path, bars_path, costs_path, regime_trades_path)
        ]
    )
    ready_path = root / str(receipt.get("ready_receipt_path", ""))
    ready = _read_json(ready_path)
    context = {
        "dataset_scope": "registered_full_family_strict_cost_walkforward_trades",
        "registered_contract_id": str(receipt.get("contract_id", "")),
        "registered_execution_id": str(receipt.get("execution_id", "")),
        "registered_execution_receipt_path": _relative(receipt_path, root),
        "registered_execution_receipt_sha256": _file_hash(receipt_path),
        "registered_conclusion_status": str(receipt.get("conclusion_status", "")),
        "registered_conclusion_path": _relative(conclusion_path, root),
        "registered_conclusion_sha256": _file_hash(conclusion_path),
        "accepted_registered_hypotheses": int(
            receipt.get("accepted_registered_hypotheses", 0) or 0
        ),
        "accepted_survivor_dataset_rows": int(cohort["accepted_stage4_survivor"].sum()),
        "registered_candidate_dataset_rows": int(cohort["registered_candidate"].sum()),
        "source_family_sha256": str(receipt.get("source_family_sha256", "")),
        "source_family_rows": expected_experiments,
        "exact_mode_parity_sha256": str(ready.get("parity_sha256", "")),
        "full_family_accounted": True,
        "causal_entry_features_proven": True,
        "strict_cost_coverage_complete": True,
        "mode_orientation_cells": len(observed_mode_orientation_cells),
        "testnet_candidate_authority": False,
    }
    blockers = []
    if len(context["exact_mode_parity_sha256"]) != 64:
        blockers.append("registered_exact_mode_parity_receipt_missing")
    staged = _stage_trade_dataset_candidate(
        root=root,
        dataset=cohort,
        leakage_audit=leakage,
        source_selection=source_selection,
        registry_audit=registry_audit,
        receipt_context=context,
        additional_acceptance_blockers=blockers,
    )
    return CommandResult(
        paths=staged["paths"],
        summary={
            "status": "VALIDATED_CANDIDATE",
            "dataset_id": staged["dataset_id"],
            "rows": len(cohort),
            "pairs": cohort["pair"].nunique(),
            "exact_modes": len(modes),
            "mode_orientation_cells": context["mode_orientation_cells"],
            "accepted_survivor_dataset_rows": context["accepted_survivor_dataset_rows"],
            "registered_candidate_dataset_rows": context["registered_candidate_dataset_rows"],
            "registered_execution_id": context["registered_execution_id"],
            "research_acceptance_blockers": blockers,
            "promotion_authority": False,
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        },
    )


def run_registered_learning_research(
    *,
    root: Path = ROOT,
    now: datetime | None = None,
    execute: bool = False,
    execution_receipt_path: Path | None = None,
    dataset_builder: ResearchRunner = build_registered_exact_mode_trade_dataset,
    promoter: ResearchRunner = promote_trade_dataset,
    trainer: ResearchRunner = train_trade_gate,
    model_backtester: ResearchRunner = run_model_gated_backtest,
    rl_runner: ResearchRunner = run_rl_research,
    governance_builder: ResearchRunner = build_corrective_agent_governance,
) -> CommandResult:
    """Run supervised and RL research once per immutable Stage 4 execution."""

    as_of = _as_utc(now)
    active = root / "reports" / "active"
    active.mkdir(parents=True, exist_ok=True)
    status_path = active / "registered_learning_research_status.json"
    try:
        receipt_path, stage4, _ = _registered_execution(
            root=root, execution_receipt_path=execution_receipt_path
        )
    except (FileNotFoundError, TypeError, ValueError) as exc:
        return _publish_status(
            status_path=status_path,
            status="BLOCKED_STAGE4",
            blocker=safe_validation_exception_code(exc),
            execute=execute,
        )
    if not _stage4_has_accepted_survivor(stage4):
        accepted_contract_rows = int(stage4.get("accepted_registered_hypotheses", 0) or 0)
        return _publish_status(
            status_path=status_path,
            status="BLOCKED_STAGE4_NO_ACCEPTED_SURVIVOR",
            blocker=(
                "registered_stage4_whole_cohort_or_independent_breadth_incomplete"
                if accepted_contract_rows > 0
                else "registered_stage4_conclusively_rejected_all_hypotheses"
            ),
            execute=execute,
            extra={
                "registered_execution_id": stage4.get("execution_id", ""),
                "registered_conclusion_status": stage4.get("conclusion_status", ""),
                "accepted_registered_hypotheses": int(
                    stage4.get("accepted_registered_hypotheses", 0) or 0
                ),
            },
        )
    if not _stage4_checkpoint_allows_registered_learning(
        root=root,
        receipt=stage4,
    ):
        return _publish_status(
            status_path=status_path,
            status="BLOCKED_STAGE4_CURRENT_FAMILY_INCOMPLETE",
            blocker=(
                "registered_stage4_checkpoint_does_not_prove_whole_cohort_"
                "accepted_independent_survivors"
            ),
            execute=execute,
            extra={
                "registered_execution_id": stage4.get("execution_id", ""),
                "registered_contract_id": stage4.get("contract_id", ""),
                "registered_conclusion_status": stage4.get("conclusion_status", ""),
                "accepted_registered_hypotheses": int(
                    stage4.get("accepted_registered_hypotheses", 0) or 0
                ),
            },
        )
    survivor_support_policy_path = root / "config" / "registered_survivor_oos_support_policy.json"
    try:
        survivor_support_policy = _load_survivor_support_policy(survivor_support_policy_path)
    except (FileNotFoundError, ValueError) as exc:
        return _publish_status(
            status_path=status_path,
            status="BLOCKED_STAGE5_SUPPORT_POLICY",
            blocker=safe_validation_exception_code(exc),
            execute=execute,
        )
    try:
        stage5_protocol_path, stage5_protocol = validate_registered_stage5_protocol(
            root=root,
            stage4_completed_at_utc=str(stage4.get("completed_at_utc", "")),
        )
        supervised_protocol = _registered_supervised_protocol(stage5_protocol)
        rl_protocol = _registered_rl_protocol(stage5_protocol)
    except (FileNotFoundError, TypeError, ValueError) as exc:
        return _publish_status(
            status_path=status_path,
            status="BLOCKED_STAGE5_PROTOCOL",
            blocker=safe_validation_exception_code(exc),
            execute=execute,
        )
    learning_id = (
        "registeredlearning_"
        + sha256(
            (
                f"{SCHEMA_VERSION}|{stage4['execution_id']}|"
                f"{_file_hash(receipt_path)}|"
                f"{_file_hash(survivor_support_policy_path)}|"
                f"{stage5_protocol['protocol_id']}|"
                f"{_file_hash(stage5_protocol_path)}"
            ).encode()
        ).hexdigest()[:20]
    )
    immutable = root / "data" / "research" / "registered_learning" / f"{learning_id}.json"
    if immutable.is_file():
        learning = _validate_learning_receipt(immutable, root=root, stage4=stage4)
        return _publish_status(
            status_path=status_path,
            status="ALREADY_COMPLETE",
            blocker="",
            execute=execute,
            learning=learning,
            receipt_path=immutable,
        )
    if not execute:
        return _publish_status(
            status_path=status_path,
            status="PLANNED",
            blocker="execution_not_requested",
            execute=False,
            extra={
                "learning_id": learning_id,
                "registered_execution_id": stage4["execution_id"],
                "registered_stage5_protocol_id": stage5_protocol["protocol_id"],
            },
        )

    lock_path = active / ".corrective_registered_learning.lock"
    _acquire_lock(lock_path, now=as_of, timeout_seconds=LOCK_TIMEOUT_SECONDS)
    try:
        if immutable.is_file():
            learning = _validate_learning_receipt(immutable, root=root, stage4=stage4)
            return _publish_status(
                status_path=status_path,
                status="ALREADY_COMPLETE",
                blocker="",
                execute=True,
                learning=learning,
                receipt_path=immutable,
            )
        candidate = dataset_builder(root=root, execution_receipt_path=receipt_path)
        _assert_no_authority("dataset_builder", candidate.summary)
        candidate_receipt = _read_json(Path(candidate.paths["receipt"]))
        blockers = candidate_receipt.get("research_acceptance_blockers", [])
        if blockers:
            return _publish_status(
                status_path=status_path,
                status="BLOCKED_DATASET",
                blocker=";".join(str(value) for value in blockers),
                execute=True,
                extra={"learning_id": learning_id},
            )
        promoted = promoter(
            root=root, candidate_pointer_path=Path(candidate.paths["candidate_pointer"])
        )
        _assert_no_authority("dataset_promotion", promoted.summary)
        trained = trainer(
            root=root,
            walkforward_splits=supervised_protocol["walkforward_splits"],
            min_train_rows=supervised_protocol["minimum_train_rows"],
            embargo_periods=supervised_protocol["embargo_periods"],
        )
        _assert_no_authority("trade_gate_training", trained.summary)
        model_gate = model_backtester(root=root)
        _assert_no_authority("model_gated_backtest", model_gate.summary)
        survivor_attribution_path = _write_registered_survivor_attribution(
            root=root,
            promoted=promoted,
            trained=trained,
            model_gate=model_gate,
            survivor_support_policy=survivor_support_policy,
        )
        rl = rl_runner(
            root=root,
            train_fraction=rl_protocol["train_fraction"],
            validation_fraction=rl_protocol["validation_fraction"],
            minimum_rows=tuple(rl_protocol["minimum_rows"]),
            entry_threshold_quantile=rl_protocol["entry_threshold_quantile"],
        )
        _assert_no_authority("rl_research", rl.summary)
        rl_survivor_attribution_path = _write_registered_survivor_rl_attribution(
            root=root,
            promoted=promoted,
            rl=rl,
            survivor_support_policy=survivor_support_policy,
        )
        governance = governance_builder(root=root, now=as_of)
        _assert_no_authority("agent_governance", governance.summary)

        acceptance = _reconcile_registered_learning_acceptance(
            root=root,
            promoted=promoted,
            trained=trained,
            model_gate=model_gate,
            rl=rl,
            governance=governance,
            survivor_attribution_path=survivor_attribution_path,
            survivor_support_policy_path=survivor_support_policy_path,
            rl_survivor_attribution_path=rl_survivor_attribution_path,
            stage5_protocol=stage5_protocol,
        )
        training_accepted = acceptance["training_accepted_oos"]
        model_accepted = acceptance["model_accepted_oos"]
        rl_accepted = acceptance["rl_accepted_oos"]
        governance_pass = acceptance["agent_governance_pass"]
        stage5_pass = acceptance["stage5_research_gate_pass"]
        artifacts = {
            "stage4_execution_receipt": receipt_path,
            "dataset_receipt": Path(candidate.paths["receipt"]),
            "active_dataset_pointer": Path(promoted.paths["active_pointer"]),
            "model_metrics": Path(trained.paths["metrics"]),
            "model_lineage": Path(trained.paths["lineage"]),
            "model_predictions": Path(model_gate.paths["predictions"]),
            "model_selection_leaderboard": Path(model_gate.paths["selection_leaderboard"]),
            "model_backtest": Path(model_gate.paths["backtest"]),
            "model_acceptance": Path(model_gate.paths["acceptance"]),
            "model_score_buckets": Path(model_gate.paths["score_buckets"]),
            "model_pair_concentration": Path(model_gate.paths["pair_concentration"]),
            "model_gain_concentration": Path(model_gate.paths["gain_concentration"]),
            "model_failure_attribution": Path(model_gate.paths["failures"]),
            "model_pair_support": Path(model_gate.paths["model_gate_pair_support_report"]),
            "registered_survivor_support_policy": (survivor_support_policy_path),
            "registered_stage5_protocol": stage5_protocol_path,
            "registered_survivor_attribution": survivor_attribution_path,
            "rl_acceptance": Path(rl.paths["acceptance_report"]),
            "rl_evaluation": Path(rl.paths["evaluation_report"]),
            "rl_execution_backtest": Path(rl.paths["execution_backtest"]),
            "registered_survivor_rl_attribution": (rl_survivor_attribution_path),
            "rl_split_audit": Path(rl.paths["split_audit"]),
            "rl_leakage_audit": Path(rl.paths["leakage_audit"]),
            "rl_feature_schema": Path(rl.paths["feature_schema"]),
            "rl_lineage": Path(rl.paths["lineage_report_json"]),
            "model_authority": Path(governance.paths["model_authority"]),
        }
        artifact_hashes = _required_artifact_hashes(artifacts, root=root)
        learning = {
            "schema_version": SCHEMA_VERSION,
            "learning_id": learning_id,
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "status": (
                "PASS_RESEARCH_LEARNING_GATES"
                if stage5_pass
                else "REJECTED_RESEARCH_LEARNING_GATES"
            ),
            "registered_contract_id": stage4["contract_id"],
            "registered_execution_id": stage4["execution_id"],
            "registered_conclusion_status": stage4["conclusion_status"],
            "registered_stage5_protocol_id": stage5_protocol["protocol_id"],
            "registered_stage5_protocol_sha256": _file_hash(stage5_protocol_path),
            "accepted_registered_hypotheses": int(
                stage4.get("accepted_registered_hypotheses", 0) or 0
            ),
            "training_dataset_id": promoted.summary.get("dataset_id", ""),
            "training_accepted_oos": training_accepted,
            "model_accepted_oos": model_accepted,
            "rl_accepted_oos": rl_accepted,
            "agent_governance_pass": governance_pass,
            "stage5_research_gate_pass": stage5_pass,
            "acceptance_checks": acceptance["acceptance_checks"],
            "acceptance_blockers": acceptance["acceptance_blockers"],
            "artifact_roles": {role: _relative(path, root) for role, path in artifacts.items()},
            "artifact_hashes": artifact_hashes,
            "order_submission_performed": False,
            "promotion_authority": False,
            "testnet_candidate_authority": False,
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        }
        learning["receipt_sha256"] = _payload_hash(learning)
        _write_immutable_json(learning, immutable)
        return _publish_status(
            status_path=status_path,
            status=learning["status"],
            blocker=("" if stage5_pass else ";".join(acceptance["acceptance_blockers"])),
            execute=True,
            learning=learning,
            receipt_path=immutable,
        )
    except Exception as exc:
        _publish_status(
            status_path=status_path,
            status="FAILED",
            blocker=f"{safe_exception_code(exc)}",
            execute=True,
        )
        raise
    finally:
        lock_path.unlink(missing_ok=True)


def _registered_execution(
    *, root: Path, execution_receipt_path: Path | None
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    if execution_receipt_path is None:
        status = _read_json(
            root / "reports" / "active" / "registered_research_rerun_execution.json"
        )
        raw_path = str(status.get("execution_receipt_path", ""))
        if not raw_path:
            raise FileNotFoundError("registered Stage 4 execution receipt is unavailable")
        execution_receipt_path = root / raw_path
    if not execution_receipt_path.is_file():
        raise FileNotFoundError(str(execution_receipt_path))
    raw = _read_json(execution_receipt_path)
    contract_id = str(raw.get("contract_id", ""))
    contract_path = (
        root / "data" / "research" / "registered_rerun_contracts" / f"{contract_id}.json"
    )
    contract = _read_json(contract_path)
    if not contract or contract.get("contract_id") != contract_id:
        raise ValueError("registered Stage 4 immutable contract is missing")
    receipt = _validate_execution_receipt(
        receipt_path=execution_receipt_path,
        contract=contract,
        root=root,
    )
    if receipt.get("status") != "PASS_REGISTERED_RERUN_ACCOUNTED":
        raise ValueError("registered Stage 4 execution is not fully accounted")
    return execution_receipt_path, receipt, contract


def _registered_supervised_protocol(protocol: dict[str, Any]) -> dict[str, Any]:
    contract = _registered_protocol_contract(protocol)
    supervised = contract.get("supervised")
    if not isinstance(supervised, dict):
        raise TypeError("registered Stage 5 supervised protocol is missing")
    required_values = {
        "candidate_families": [
            "logistic_regression",
            "regularized_random_forest",
            "boosted_tree_with_deterministic_fallback",
        ],
        "selection_scheme": MODEL_SELECTION_ISOLATION_SCHEME,
        "global_split_scheme": GLOBAL_PURGED_SPLIT_SCHEME,
        "prediction_membership": "exact_registered_row_fold_phase_v1",
        "threshold_calibration": THRESHOLD_CALIBRATION_SCHEME,
        "threshold_grid": {
            "start": 0.5,
            "stop_inclusive": 0.8,
            "step": 0.05,
        },
        "minimum_training_take_rate": MINIMUM_TRAINING_TAKE_RATE,
        "minimum_oos_take_rate": MINIMUM_TRAINING_TAKE_RATE,
        "model_selection_uses_untouched_evaluation": False,
        "score_bucket_monotonicity_required": True,
        "pair_regime_timeframe_concentration_required": True,
    }
    for field, expected in required_values.items():
        if supervised.get(field) != expected:
            raise ValueError(f"registered Stage 5 supervised protocol mismatch: {field}")
    selection_dimensions = supervised.get("selection_concentration_dimensions")
    if selection_dimensions != ["pair", "timeframe", "regime"]:
        raise ValueError(
            "registered Stage 5 supervised protocol mismatch: selection_concentration_dimensions"
        )
    try:
        values = {
            "walkforward_splits": int(supervised["walkforward_splits"]),
            "minimum_train_rows": int(supervised["minimum_train_rows"]),
            "embargo_periods": int(supervised["embargo_periods"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("registered Stage 5 supervised protocol parameters are invalid") from exc
    if (
        values["walkforward_splits"] < 3
        or values["minimum_train_rows"] < 2
        or values["embargo_periods"] < 1
    ):
        raise ValueError("registered Stage 5 supervised protocol parameters are unsafe")
    values["selection_concentration_dimensions"] = list(selection_dimensions)
    return values


def _registered_rl_protocol(protocol: dict[str, Any]) -> dict[str, Any]:
    contract = _registered_protocol_contract(protocol)
    rl = contract.get("reinforcement_learning")
    if not isinstance(rl, dict):
        raise TypeError("registered Stage 5 RL protocol is missing")
    required_values = {
        "policy": "simulated_quantile_hold_policy",
        "calibration_split": "globally_purged_train",
        "validation_split": "policy_selection",
        "test_split": "untouched_evaluation",
        "minimum_take_rate": 0.1,
        "positive_validation_after_cost_return_required": True,
        "positive_test_after_cost_return_required": True,
        "pair_regime_timeframe_concentration_required": True,
    }
    for field, expected in required_values.items():
        if rl.get(field) != expected:
            raise ValueError(f"registered Stage 5 RL protocol mismatch: {field}")
    try:
        train_fraction = float(rl["train_fraction"])
        validation_fraction = float(rl["validation_fraction"])
        minimum_rows = [int(value) for value in rl["minimum_rows"]]
        entry_quantile = float(rl["entry_threshold_quantile"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("registered Stage 5 RL protocol parameters are invalid") from exc
    if (
        not 0.0 < train_fraction < 1.0
        or not 0.0 < validation_fraction < 1.0
        or train_fraction + validation_fraction >= 1.0
        or len(minimum_rows) != 3
        or any(value <= 0 for value in minimum_rows)
        or not 0.0 <= entry_quantile <= 1.0
    ):
        raise ValueError("registered Stage 5 RL protocol parameters are unsafe")
    return {
        "train_fraction": train_fraction,
        "validation_fraction": validation_fraction,
        "minimum_rows": minimum_rows,
        "entry_threshold_quantile": entry_quantile,
    }


def _registered_protocol_contract(protocol: dict[str, Any]) -> dict[str, Any]:
    contract = protocol.get("protocol_contract")
    if not isinstance(contract, dict):
        raise TypeError("registered Stage 5 protocol contract is missing")
    required_values = {
        "cohort": "immutable_accepted_registered_stage4_full_family_strict_cost_trades",
        "features": "entry_time_causal_no_dashboard_hindsight",
        "selection_rule": "both_supervised_and_rl_oos_gates_must_pass_without_trade_count_collapse",
        "failed_run_action": "research_only_no_quantization_no_testnet_candidate",
    }
    for field, expected in required_values.items():
        if contract.get(field) != expected:
            raise ValueError(f"registered Stage 5 protocol mismatch: {field}")
    return contract


def _stage4_has_accepted_survivor(receipt: dict[str, Any]) -> bool:
    final_ids = receipt.get("stage4_final_experiment_ids", [])
    final_pairs = receipt.get("stage4_final_canonical_pairs", [])
    return bool(
        receipt.get("conclusion_status") == "ACCEPTED_REGISTERED_SURVIVORS"
        and int(receipt.get("accepted_registered_hypotheses", 0) or 0) > 0
        and int(receipt.get("stage4_final_one_x_survivors", 0) or 0) >= 3
        and isinstance(final_ids, list)
        and len(final_ids) == int(receipt.get("stage4_final_one_x_survivors", 0) or 0)
        and int(receipt.get("stage4_independent_supporting_clusters", 0) or 0) >= 3
        and int(receipt.get("stage4_independent_full_survivor_clusters", 0) or 0) >= 3
        and int(receipt.get("stage4_independent_supporting_pairs", 0) or 0) >= 3
        and int(receipt.get("stage4_independent_full_survivor_pairs", 0) or 0) >= 3
        and isinstance(final_pairs, list)
        and len(final_pairs) == int(receipt.get("stage4_independent_full_survivor_pairs", 0) or 0)
    )


def _stage4_checkpoint_allows_registered_learning(*, root: Path, receipt: dict[str, Any]) -> bool:
    active_contract = _read_json(
        root / "reports" / "active" / "registered_research_rerun_contract.json"
    )
    checkpoint = _read_csv(root / "reports" / "active" / "seven_stage_goal_checkpoint.csv")
    if checkpoint.empty or "stage" not in checkpoint:
        return False
    selected = checkpoint.loc[checkpoint["stage"].astype(str).eq("4")]
    if len(selected) != 1:
        return False
    row = selected.iloc[0]
    evidence = str(row.get("evidence_progress", ""))
    contract_id = str(receipt.get("contract_id", "")).strip()
    return bool(
        contract_id
        and active_contract.get("contract_id") == contract_id
        and str(row.get("status", "")) == "PASS"
        and f"active_contract_id={contract_id}" in evidence
        and "stage4_terminal_outcome=ACCEPTED_VALID_INDEPENDENT_SURVIVORS" in evidence
        and "final_receipt_conclusion_bound=True" in evidence
        and not _truthy(row.get("testnet_order_authority"))
        and not _truthy(row.get("live_trading_authorized"))
    )


def _registered_hypothesis_lineage(
    *,
    root: Path,
    receipt: dict[str, Any],
    contract: dict[str, Any],
) -> tuple[Path, pd.DataFrame]:
    relative = str(receipt.get("conclusion_path", "")).strip()
    conclusion_path = root / relative
    expected_hash = str(receipt.get("conclusion_sha256", "")).strip()
    if (
        not relative
        or not conclusion_path.is_file()
        or len(expected_hash) != 64
        or _file_hash(conclusion_path) != expected_hash
    ):
        raise ValueError("registered Stage 4 conclusion is not hash-bound")
    conclusion = _read_json(conclusion_path)
    outcomes = conclusion.get("outcomes", {})
    candidates = contract.get("registered_candidates", [])
    if not isinstance(outcomes, dict) or not isinstance(candidates, list):
        raise TypeError("registered Stage 4 hypothesis lineage is malformed")
    expected_ids = {
        str(candidate.get("semantic_hypothesis_id", "")).strip()
        for candidate in candidates
        if isinstance(candidate, dict)
    } - {""}
    if not expected_ids or set(outcomes) != expected_ids:
        raise ValueError("registered Stage 4 conclusion outcome coverage mismatch")
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        semantic_id = str(candidate.get("semantic_hypothesis_id", "")).strip()
        experiment_id = str(candidate.get("source_experiment_id", "")).strip()
        pair_group_key = str(candidate.get("pair_group_key", "")).strip()
        pair = str(candidate.get("pair", "")).strip()
        exact_mode = str(candidate.get("exact_mode", "")).strip()
        orientation = str(candidate.get("orientation", "")).strip()
        outcome = str(outcomes.get(semantic_id, "")).strip()
        if (
            not semantic_id
            or not experiment_id
            or not pair_group_key
            or not pair
            or exact_mode not in CANONICAL_WIZARD_MODES
            or orientation not in REQUIRED_ORIENTATIONS
            or outcome
            not in {
                "ACCEPTED_SURVIVOR",
                "REJECTED_BY_FROZEN_GATES",
            }
        ):
            raise ValueError("registered Stage 4 hypothesis outcome is invalid")
        rows.append(
            {
                "source_experiment_id": experiment_id,
                "registered_semantic_hypothesis_id": semantic_id,
                "registered_hypothesis_outcome": outcome,
                "registered_candidate": True,
                "accepted_stage4_survivor": outcome == "ACCEPTED_SURVIVOR",
                "registered_pair_group_key": pair_group_key,
                "registered_pair": pair,
                "registered_exact_mode": exact_mode,
                "registered_orientation": orientation,
            }
        )
    lineage = pd.DataFrame(rows)
    if lineage.empty or lineage["source_experiment_id"].duplicated().any():
        raise ValueError("registered Stage 4 experiment lineage is not one-to-one")
    accepted = int(lineage["accepted_stage4_survivor"].sum())
    if accepted != int(receipt.get("accepted_registered_hypotheses", -1)):
        raise ValueError("registered Stage 4 accepted survivor count mismatch")
    accepted_experiment_ids = set(
        lineage.loc[
            lineage["accepted_stage4_survivor"].map(_truthy),
            "source_experiment_id",
        ].astype(str)
    ) - {""}
    final_experiment_ids = [
        str(value).strip() for value in receipt.get("stage4_final_experiment_ids", [])
    ]
    if (
        len(final_experiment_ids) != len(set(final_experiment_ids))
        or set(final_experiment_ids) != accepted_experiment_ids
        or int(receipt.get("stage4_final_one_x_survivors", -1)) != len(accepted_experiment_ids)
    ):
        raise ValueError(
            "registered learning survivor identity does not match the accepted Stage 4 cohort"
        )
    return conclusion_path, lineage


def _write_registered_survivor_attribution(
    *,
    root: Path,
    promoted: CommandResult,
    trained: CommandResult,
    model_gate: CommandResult,
    survivor_support_policy: dict[str, Any],
) -> Path:
    active_pointer = _read_json(_result_path(promoted, "active_pointer"))
    dataset = _read_csv(root / str(active_pointer.get("active_dataset_path", "")))
    metrics = _read_json(_result_path(trained, "metrics"))
    predictions = _filter_predictions_to_untouched_evaluation(
        _filter_predictions_to_model(
            _read_csv(_result_path(model_gate, "predictions")),
            str(metrics.get("best_model", "")),
        )
    )
    frame = _registered_survivor_attribution_frame(
        dataset=dataset,
        predictions=predictions,
        survivor_support_policy=survivor_support_policy,
    )
    path = root / "reports" / "ml" / "registered_survivor_model_attribution.csv"
    _atomic_csv(frame, path)
    return path


def _registered_survivor_attribution_frame(
    *,
    dataset: pd.DataFrame,
    predictions: pd.DataFrame,
    survivor_support_policy: dict[str, Any],
) -> pd.DataFrame:
    columns = [
        "support_policy_id",
        "support_policy_version",
        "registered_semantic_hypothesis_id",
        "experiment_id",
        "pair",
        "exact_mode",
        "orientation",
        "dataset_rows",
        "oos_prediction_rows",
        "oos_folds",
        "oos_regimes",
        "oos_taken_rows",
        "oos_take_rate",
        "oos_taken_return_rows",
        "oos_taken_return_sum",
        "oos_taken_mean_return",
        "minimum_dataset_rows",
        "minimum_model_selection_folds",
        "minimum_oos_prediction_rows",
        "minimum_oos_folds",
        "minimum_oos_regimes",
        "minimum_oos_taken_rows",
        "minimum_oos_take_rate",
        "minimum_oos_taken_return_sum_exclusive",
        "return_basis",
        "support_status",
        "promotion_authority",
        "testnet_candidate_authority",
        "testnet_order_authority",
        "live_trading_authorized",
    ]
    required = {
        "registered_semantic_hypothesis_id",
        "accepted_stage4_survivor",
        "experiment_id",
    }
    if dataset.empty or not required.issubset(dataset.columns):
        return pd.DataFrame(columns=columns)
    accepted = dataset.loc[dataset["accepted_stage4_survivor"].map(_truthy)].copy()
    if accepted.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for semantic_id, cohort in accepted.groupby(
        "registered_semantic_hypothesis_id", sort=True, dropna=False
    ):
        semantic_id = str(semantic_id).strip()
        identity = {
            field: set(cohort.get(field, pd.Series(dtype=str)).astype(str)) - {""}
            for field in ("experiment_id", "pair", "exact_mode", "orientation")
        }
        if any(len(values) != 1 for values in identity.values()):
            raise ValueError("registered survivor dataset identity is not one-to-one")
        expected = {field: next(iter(values)) for field, values in identity.items()}
        prediction_cohort = predictions.loc[
            predictions.get(
                "registered_semantic_hypothesis_id",
                pd.Series("", index=predictions.index),
            )
            .astype(str)
            .eq(semantic_id)
            & predictions.get(
                "accepted_stage4_survivor",
                pd.Series(False, index=predictions.index),
            ).map(_truthy)
        ].copy()
        for field, value in expected.items():
            prediction_cohort = prediction_cohort.loc[
                prediction_cohort.get(field, pd.Series("", index=prediction_cohort.index))
                .astype(str)
                .eq(value)
            ]
        taken = prediction_cohort.get(
            "shadow_take", pd.Series(False, index=prediction_cohort.index)
        ).map(_truthy)
        raw_returns = pd.to_numeric(
            prediction_cohort.loc[taken].get("realized_return", pd.Series(dtype=float)),
            errors="coerce",
        )
        finite_returns = raw_returns.loc[np.isfinite(raw_returns)]
        dataset_rows = len(cohort)
        oos_rows = len(prediction_cohort)
        oos_folds = int(
            prediction_cohort.get("fold", pd.Series(dtype=object))
            .dropna()
            .astype(str)
            .str.strip()
            .replace("", np.nan)
            .dropna()
            .nunique()
        )
        oos_regimes = int(
            prediction_cohort.get("regime", pd.Series(dtype=object))
            .dropna()
            .astype(str)
            .str.strip()
            .replace("", np.nan)
            .dropna()
            .nunique()
        )
        taken_rows = int(taken.sum())
        taken_return_rows = len(finite_returns)
        take_rate = taken_rows / oos_rows if oos_rows else 0.0
        return_sum = float(finite_returns.sum()) if not finite_returns.empty else 0.0
        minimum_dataset_rows = int(survivor_support_policy["minimum_dataset_rows"])
        minimum_oos_rows = int(survivor_support_policy["minimum_oos_prediction_rows"])
        minimum_oos_folds = int(survivor_support_policy["minimum_oos_folds"])
        minimum_oos_regimes = int(survivor_support_policy["minimum_oos_regimes"])
        minimum_taken_rows = int(survivor_support_policy["minimum_oos_taken_rows"])
        minimum_take_rate = float(survivor_support_policy["minimum_oos_take_rate"])
        minimum_return_sum = float(
            survivor_support_policy["minimum_oos_taken_return_sum_exclusive"]
        )
        support_status = "READY"
        if dataset_rows < minimum_dataset_rows:
            support_status = "BLOCKED_DATASET_SUPPORT"
        elif oos_rows < minimum_oos_rows:
            support_status = "BLOCKED_OOS_SAMPLE_DEPTH"
        elif oos_folds < minimum_oos_folds:
            support_status = "BLOCKED_OOS_FOLD_DEPTH"
        elif oos_regimes < minimum_oos_regimes:
            support_status = "BLOCKED_OOS_REGIME_DEPTH"
        elif taken_rows < minimum_taken_rows:
            support_status = "BLOCKED_MODEL_SKIPPED_SURVIVOR"
        elif take_rate < minimum_take_rate:
            support_status = "BLOCKED_MODEL_TAKE_RATE"
        elif taken_return_rows != taken_rows:
            support_status = "BLOCKED_OOS_RETURN_COVERAGE"
        elif return_sum <= minimum_return_sum:
            support_status = "BLOCKED_OOS_AFTER_COST_RETURN"
        rows.append(
            {
                "support_policy_id": survivor_support_policy["support_policy_id"],
                "support_policy_version": survivor_support_policy["policy_version"],
                "registered_semantic_hypothesis_id": semantic_id,
                "experiment_id": expected["experiment_id"],
                "pair": expected["pair"],
                "exact_mode": expected["exact_mode"],
                "orientation": expected["orientation"],
                "dataset_rows": dataset_rows,
                "oos_prediction_rows": oos_rows,
                "oos_folds": oos_folds,
                "oos_regimes": oos_regimes,
                "oos_taken_rows": taken_rows,
                "oos_take_rate": take_rate,
                "oos_taken_return_rows": taken_return_rows,
                "oos_taken_return_sum": return_sum,
                "oos_taken_mean_return": (
                    float(finite_returns.mean()) if not finite_returns.empty else 0.0
                ),
                "minimum_dataset_rows": minimum_dataset_rows,
                "minimum_oos_prediction_rows": minimum_oos_rows,
                "minimum_oos_folds": minimum_oos_folds,
                "minimum_oos_regimes": minimum_oos_regimes,
                "minimum_oos_taken_rows": minimum_taken_rows,
                "minimum_oos_take_rate": minimum_take_rate,
                "minimum_oos_taken_return_sum_exclusive": (minimum_return_sum),
                "return_basis": survivor_support_policy["return_basis"],
                "support_status": support_status,
                "promotion_authority": False,
                "testnet_candidate_authority": False,
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _write_registered_survivor_rl_attribution(
    *,
    root: Path,
    promoted: CommandResult,
    rl: CommandResult,
    survivor_support_policy: dict[str, Any],
) -> Path:
    active_pointer = _read_json(_result_path(promoted, "active_pointer"))
    dataset = _read_csv(root / str(active_pointer.get("active_dataset_path", "")))
    execution = _read_csv(_result_path(rl, "execution_backtest"))
    frame = _registered_survivor_rl_attribution_frame(
        dataset=dataset,
        execution=execution,
        survivor_support_policy=survivor_support_policy,
    )
    path = root / "reports" / "rl" / "registered_survivor_rl_attribution.csv"
    _atomic_csv(frame, path)
    return path


def _registered_survivor_rl_attribution_frame(
    *,
    dataset: pd.DataFrame,
    execution: pd.DataFrame,
    survivor_support_policy: dict[str, Any],
) -> pd.DataFrame:
    columns = [
        "support_policy_id",
        "support_policy_version",
        "registered_semantic_hypothesis_id",
        "experiment_id",
        "pair",
        "exact_mode",
        "orientation",
        "dataset_rows",
        "validation_rows",
        "validation_entered_rows",
        "validation_take_rate",
        "validation_return_rows",
        "validation_return_sum",
        "held_out_test_rows",
        "held_out_test_entered_rows",
        "held_out_test_take_rate",
        "held_out_test_return_rows",
        "held_out_test_return_sum",
        "minimum_rl_rows_per_split",
        "minimum_rl_entered_rows_per_split",
        "minimum_rl_take_rate_per_split",
        "minimum_rl_validation_return_sum_exclusive",
        "minimum_rl_test_return_sum_exclusive",
        "return_basis",
        "support_status",
        "promotion_authority",
        "testnet_candidate_authority",
        "testnet_order_authority",
        "live_trading_authorized",
    ]
    required = {
        "registered_semantic_hypothesis_id",
        "accepted_stage4_survivor",
        "experiment_id",
    }
    if dataset.empty or not required.issubset(dataset.columns):
        return pd.DataFrame(columns=columns)
    accepted = dataset.loc[dataset["accepted_stage4_survivor"].map(_truthy)].copy()
    if accepted.empty:
        return pd.DataFrame(columns=columns)
    minimum_rows = int(survivor_support_policy["minimum_rl_rows_per_split"])
    minimum_entered = int(survivor_support_policy["minimum_rl_entered_rows_per_split"])
    minimum_take_rate = float(survivor_support_policy["minimum_rl_take_rate_per_split"])
    minimum_validation_return = float(
        survivor_support_policy["minimum_rl_validation_return_sum_exclusive"]
    )
    minimum_test_return = float(survivor_support_policy["minimum_rl_test_return_sum_exclusive"])
    expected_return_basis = str(survivor_support_policy["rl_return_basis"])
    rows: list[dict[str, Any]] = []
    for semantic_id, cohort in accepted.groupby(
        "registered_semantic_hypothesis_id", sort=True, dropna=False
    ):
        semantic_id = str(semantic_id).strip()
        identity = {
            field: set(cohort.get(field, pd.Series(dtype=str)).astype(str)) - {""}
            for field in ("experiment_id", "pair", "exact_mode", "orientation")
        }
        if any(len(values) != 1 for values in identity.values()):
            raise ValueError("registered survivor RL dataset identity is not one-to-one")
        expected = {field: next(iter(values)) for field, values in identity.items()}
        execution_cohort = execution.loc[
            execution.get(
                "registered_semantic_hypothesis_id",
                pd.Series("", index=execution.index),
            )
            .astype(str)
            .eq(semantic_id)
            & execution.get(
                "accepted_stage4_survivor",
                pd.Series(False, index=execution.index),
            ).map(_truthy)
        ].copy()
        for field, value in expected.items():
            execution_cohort = execution_cohort.loc[
                execution_cohort.get(field, pd.Series("", index=execution_cohort.index))
                .astype(str)
                .eq(value)
            ]
        split_metrics: dict[str, dict[str, float | int]] = {}
        return_basis_ready = True
        for split in ("validation", "held_out_test"):
            split_frame = execution_cohort.loc[
                execution_cohort.get(
                    "evaluation_split",
                    pd.Series("", index=execution_cohort.index),
                )
                .astype(str)
                .eq(split)
            ]
            entered = (
                split_frame.get("simulation_reason", pd.Series("", index=split_frame.index))
                .astype(str)
                .eq("entered")
            )
            raw_returns = pd.to_numeric(
                split_frame.loc[entered].get("simulated_return", pd.Series(dtype=float)),
                errors="coerce",
            )
            finite_returns = raw_returns.loc[np.isfinite(raw_returns)]
            split_rows = len(split_frame)
            entered_rows = int(entered.sum())
            split_metrics[split] = {
                "rows": split_rows,
                "entered_rows": entered_rows,
                "take_rate": (entered_rows / split_rows if split_rows else 0.0),
                "return_rows": len(finite_returns),
                "return_sum": (float(finite_returns.sum()) if not finite_returns.empty else 0.0),
            }
            basis = split_frame.get("return_basis", pd.Series("", index=split_frame.index)).astype(
                str
            )
            return_basis_ready = bool(
                return_basis_ready and not basis.empty and basis.eq(expected_return_basis).all()
            )
        validation = split_metrics["validation"]
        held_out = split_metrics["held_out_test"]
        support_status = "READY"
        if int(validation["rows"]) < minimum_rows:
            support_status = "BLOCKED_RL_VALIDATION_SAMPLE_DEPTH"
        elif int(held_out["rows"]) < minimum_rows:
            support_status = "BLOCKED_RL_TEST_SAMPLE_DEPTH"
        elif int(validation["entered_rows"]) < minimum_entered:
            support_status = "BLOCKED_RL_VALIDATION_PARTICIPATION"
        elif int(held_out["entered_rows"]) < minimum_entered:
            support_status = "BLOCKED_RL_TEST_PARTICIPATION"
        elif float(validation["take_rate"]) < minimum_take_rate:
            support_status = "BLOCKED_RL_VALIDATION_TAKE_RATE"
        elif float(held_out["take_rate"]) < minimum_take_rate:
            support_status = "BLOCKED_RL_TEST_TAKE_RATE"
        elif int(validation["return_rows"]) != int(validation["entered_rows"]) or int(
            held_out["return_rows"]
        ) != int(held_out["entered_rows"]):
            support_status = "BLOCKED_RL_RETURN_COVERAGE"
        elif not return_basis_ready:
            support_status = "BLOCKED_RL_RETURN_BASIS"
        elif float(validation["return_sum"]) <= minimum_validation_return:
            support_status = "BLOCKED_RL_VALIDATION_AFTER_COST_RETURN"
        elif float(held_out["return_sum"]) <= minimum_test_return:
            support_status = "BLOCKED_RL_TEST_AFTER_COST_RETURN"
        rows.append(
            {
                "support_policy_id": survivor_support_policy["support_policy_id"],
                "support_policy_version": survivor_support_policy["policy_version"],
                "registered_semantic_hypothesis_id": semantic_id,
                "experiment_id": expected["experiment_id"],
                "pair": expected["pair"],
                "exact_mode": expected["exact_mode"],
                "orientation": expected["orientation"],
                "dataset_rows": len(cohort),
                "validation_rows": validation["rows"],
                "validation_entered_rows": validation["entered_rows"],
                "validation_take_rate": validation["take_rate"],
                "validation_return_rows": validation["return_rows"],
                "validation_return_sum": validation["return_sum"],
                "held_out_test_rows": held_out["rows"],
                "held_out_test_entered_rows": held_out["entered_rows"],
                "held_out_test_take_rate": held_out["take_rate"],
                "held_out_test_return_rows": held_out["return_rows"],
                "held_out_test_return_sum": held_out["return_sum"],
                "minimum_rl_rows_per_split": minimum_rows,
                "minimum_rl_entered_rows_per_split": minimum_entered,
                "minimum_rl_take_rate_per_split": minimum_take_rate,
                "minimum_rl_validation_return_sum_exclusive": minimum_validation_return,
                "minimum_rl_test_return_sum_exclusive": minimum_test_return,
                "return_basis": expected_return_basis,
                "support_status": support_status,
                "promotion_authority": False,
                "testnet_candidate_authority": False,
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _rl_evaluation_from_execution(execution: pd.DataFrame) -> pd.DataFrame:
    required = {
        "evaluation_split",
        "simulation_reason",
        "base_return",
        "simulated_return",
        "pair",
        "timeframe",
        "regime",
    }
    if execution.empty or not required.issubset(execution.columns):
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for split in ("validation", "held_out_test"):
        source = execution.loc[execution["evaluation_split"].astype(str).eq(split)].copy()
        if source.empty:
            continue
        baseline_returns = pd.to_numeric(source["base_return"], errors="coerce")
        entered = source["simulation_reason"].astype(str).eq("entered")
        selected = source.loc[entered].copy()
        simulated_returns = pd.to_numeric(selected["simulated_return"], errors="coerce")
        if not np.isfinite(baseline_returns).all() or not np.isfinite(simulated_returns).all():
            continue
        baseline = return_summary(
            "non_rl_baseline",
            source,
            baseline_returns,
            len(source),
            source_frame=source,
        )
        policy = return_summary(
            "safe_rl_policy",
            selected,
            simulated_returns,
            len(source),
            source_frame=source,
        )
        baseline["evaluation_split"] = split
        policy["evaluation_split"] = split
        rows.extend((baseline, policy))
    return pd.DataFrame(rows)


def _rl_evaluation_matches_execution(
    artifact: pd.DataFrame,
    recomputed: pd.DataFrame,
) -> bool:
    if artifact.empty or recomputed.empty:
        return False
    identity = ("evaluation_split", "variant")
    if not set(identity).issubset(artifact.columns):
        return False
    if artifact.duplicated(list(identity)).any():
        return False
    artifact_keys = set(
        zip(
            artifact["evaluation_split"].astype(str),
            artifact["variant"].astype(str),
            strict=False,
        )
    )
    recomputed_keys = set(
        zip(
            recomputed["evaluation_split"].astype(str),
            recomputed["variant"].astype(str),
            strict=False,
        )
    )
    if artifact_keys != recomputed_keys:
        return False
    for row in recomputed.to_dict(orient="records"):
        match = artifact.loc[
            artifact["evaluation_split"].astype(str).eq(str(row["evaluation_split"]))
            & artifact["variant"].astype(str).eq(str(row["variant"]))
        ]
        if len(match) != 1 or not _acceptance_rows_match(
            artifact=match.iloc[0].to_dict(),
            recomputed=row,
        ):
            return False
    return True


def _model_predictions_match_active_dataset(
    predictions: pd.DataFrame,
    dataset: pd.DataFrame,
) -> bool:
    required_text_fields = (
        "pair",
        "timeframe",
        "source_venue",
        "strategy_id",
        "strategy_name",
        "family",
        "backtest_mode",
        "regime",
        "exact_mode",
        "orientation",
        "experiment_id",
        "registered_contract_id",
        "registered_execution_id",
        "registered_semantic_hypothesis_id",
        "registered_hypothesis_outcome",
    )
    required_boolean_fields = (
        "registered_candidate",
        "accepted_stage4_survivor",
    )
    required_timestamp_fields = (
        "feature_timestamp",
        "entry_timestamp",
        "exit_timestamp",
        "label_timestamp",
    )
    required_numeric_fields = (
        "realized_return",
        "label_profitable",
    )
    required = {
        "trade_id",
        *required_text_fields,
        *required_boolean_fields,
        *required_timestamp_fields,
        *required_numeric_fields,
    }
    if (
        predictions.empty
        or dataset.empty
        or not required.issubset(predictions.columns)
        or not required.issubset(dataset.columns)
    ):
        return False

    observed = predictions[list(required)].copy()
    expected = dataset[list(required)].copy()
    observed["trade_id"] = observed["trade_id"].fillna("").astype(str)
    expected["trade_id"] = expected["trade_id"].fillna("").astype(str)
    if (
        observed["trade_id"].str.strip().eq("").any()
        or expected["trade_id"].str.strip().eq("").any()
        or observed["trade_id"].duplicated().any()
        or expected["trade_id"].duplicated().any()
        or not set(observed["trade_id"]).issubset(set(expected["trade_id"]))
    ):
        return False

    merged = observed.merge(
        expected,
        on="trade_id",
        how="inner",
        suffixes=("_observed", "_expected"),
        validate="one_to_one",
    )
    if len(merged) != len(observed):
        return False
    for field in required_text_fields:
        if (
            not merged[f"{field}_observed"]
            .fillna("")
            .astype(str)
            .eq(merged[f"{field}_expected"].fillna("").astype(str))
            .all()
        ):
            return False
    for field in required_boolean_fields:
        if (
            not merged[f"{field}_observed"]
            .map(_truthy)
            .eq(merged[f"{field}_expected"].map(_truthy))
            .all()
        ):
            return False
    for field in required_timestamp_fields:
        observed_time = pd.to_datetime(
            merged[f"{field}_observed"],
            utc=True,
            errors="coerce",
            format="mixed",
        )
        expected_time = pd.to_datetime(
            merged[f"{field}_expected"],
            utc=True,
            errors="coerce",
            format="mixed",
        )
        if observed_time.isna().any() or not observed_time.eq(expected_time).all():
            return False
    for field in required_numeric_fields:
        observed_value = pd.to_numeric(merged[f"{field}_observed"], errors="coerce")
        expected_value = pd.to_numeric(merged[f"{field}_expected"], errors="coerce")
        if (
            not np.isfinite(observed_value).all()
            or not np.isfinite(expected_value).all()
            or not np.allclose(
                observed_value,
                expected_value,
                rtol=1e-12,
                atol=1e-12,
            )
        ):
            return False
    return True


def _rl_split_audit_matches_dataset(
    artifact: pd.DataFrame,
    recomputed: pd.DataFrame,
) -> bool:
    if artifact.empty or recomputed.empty or "split" not in artifact.columns:
        return False
    if artifact["split"].astype(str).duplicated().any():
        return False
    if set(artifact["split"].astype(str)) != set(recomputed["split"].astype(str)):
        return False
    for row in recomputed.to_dict(orient="records"):
        match = artifact.loc[artifact["split"].astype(str).eq(str(row["split"]))]
        if len(match) != 1 or not _acceptance_rows_match(
            artifact=match.iloc[0].to_dict(),
            recomputed=row,
        ):
            return False
    return True


def _rl_split_chronology_proven(audit: pd.DataFrame) -> bool:
    if audit.empty or set(audit.get("split", pd.Series(dtype=str)).astype(str)) != {
        "train",
        "validation",
        "test",
    }:
        return False
    indexed = audit.set_index(audit["split"].astype(str))
    if (
        not audit.get("status", pd.Series(dtype=str)).astype(str).eq("ready").all()
        or not audit.get("global_label_purge", pd.Series(False, index=audit.index))
        .map(_truthy)
        .all()
    ):
        return False
    train_label_end = pd.to_datetime(
        indexed.loc["train", "label_end_max"], utc=True, errors="coerce"
    )
    validation_start = pd.to_datetime(
        indexed.loc["train", "validation_start"], utc=True, errors="coerce"
    )
    validation_label_end = pd.to_datetime(
        indexed.loc["validation", "label_end_max"],
        utc=True,
        errors="coerce",
    )
    test_start = pd.to_datetime(indexed.loc["validation", "test_start"], utc=True, errors="coerce")
    return bool(
        pd.notna(train_label_end)
        and pd.notna(validation_start)
        and pd.notna(validation_label_end)
        and pd.notna(test_start)
        and train_label_end < validation_start
        and validation_label_end < test_start
    )


def _rl_execution_matches_partitions(
    execution: pd.DataFrame,
    partitions: dict[str, pd.DataFrame],
) -> bool:
    required = {
        "trade_id",
        "evaluation_split",
        "base_return",
        "return_basis",
    }
    if execution.empty or not required.issubset(execution.columns):
        return False
    if execution["trade_id"].astype(str).duplicated().any():
        return False
    expected_return_basis = "net_after_cost_strategy_return"
    if not execution["return_basis"].astype(str).eq(expected_return_basis).all():
        return False
    split_map = {"validation": "validation", "held_out_test": "test"}
    if set(execution["evaluation_split"].astype(str)) != set(split_map):
        return False
    text_fields = (
        "pair",
        "timeframe",
        "regime",
        "exact_mode",
        "orientation",
        "experiment_id",
        "registered_semantic_hypothesis_id",
    )
    timestamp_fields = (
        "feature_timestamp",
        "entry_timestamp",
        "exit_timestamp",
    )
    for execution_split, partition_name in split_map.items():
        observed = execution.loc[
            execution["evaluation_split"].astype(str).eq(execution_split)
        ].copy()
        expected = partitions.get(partition_name, pd.DataFrame()).copy()
        if expected.empty or "trade_id" not in expected.columns:
            return False
        observed["trade_id"] = observed["trade_id"].astype(str)
        expected["trade_id"] = expected["trade_id"].astype(str)
        if (
            observed["trade_id"].duplicated().any()
            or expected["trade_id"].duplicated().any()
            or set(observed["trade_id"]) != set(expected["trade_id"])
        ):
            return False
        merged = observed.merge(
            expected,
            on="trade_id",
            how="inner",
            suffixes=("_observed", "_expected"),
            validate="one_to_one",
        )
        if len(merged) != len(expected):
            return False
        for field in text_fields:
            observed_field = f"{field}_observed"
            expected_field = f"{field}_expected"
            if (
                observed_field in merged
                and expected_field in merged
                and not merged[observed_field]
                .fillna("")
                .astype(str)
                .eq(merged[expected_field].fillna("").astype(str))
                .all()
            ):
                return False
        observed_survivor = "accepted_stage4_survivor_observed"
        expected_survivor = "accepted_stage4_survivor_expected"
        if (
            observed_survivor in merged
            and expected_survivor in merged
            and not merged[observed_survivor]
            .map(_truthy)
            .eq(merged[expected_survivor].map(_truthy))
            .all()
        ):
            return False
        for field in timestamp_fields:
            observed_field = f"{field}_observed"
            expected_field = f"{field}_expected"
            if observed_field in merged and expected_field in merged:
                observed_time = pd.to_datetime(merged[observed_field], utc=True, errors="coerce")
                expected_time = pd.to_datetime(merged[expected_field], utc=True, errors="coerce")
                if observed_time.isna().any() or not observed_time.eq(expected_time).all():
                    return False
        return_column = next(
            (
                field
                for field in (
                    "profit_after_cost",
                    "realized_return",
                    "trade_return",
                    "return",
                    "returns",
                )
                if field in expected.columns
            ),
            "",
        )
        if not return_column:
            return False
        expected_returns = pd.to_numeric(merged[return_column], errors="coerce")
        observed_returns = pd.to_numeric(merged["base_return"], errors="coerce")
        if (
            not np.isfinite(expected_returns).all()
            or not np.isfinite(observed_returns).all()
            or not np.allclose(
                observed_returns,
                expected_returns,
                rtol=1e-12,
                atol=1e-12,
            )
        ):
            return False
    return True


def _reconcile_registered_learning_acceptance(
    *,
    root: Path,
    promoted: CommandResult,
    trained: CommandResult,
    model_gate: CommandResult,
    rl: CommandResult,
    governance: CommandResult,
    survivor_attribution_path: Path,
    survivor_support_policy_path: Path,
    rl_survivor_attribution_path: Path,
    stage5_protocol: dict[str, Any],
) -> dict[str, Any]:
    supervised_protocol = _registered_supervised_protocol(stage5_protocol)
    rl_protocol = _registered_rl_protocol(stage5_protocol)
    survivor_support_policy = _load_survivor_support_policy(survivor_support_policy_path)
    active_pointer_path = _result_path(promoted, "active_pointer")
    model_metrics_path = _result_path(trained, "metrics")
    model_lineage_path = _result_path(trained, "lineage")
    model_predictions_path = _result_path(model_gate, "predictions")
    model_selection_leaderboard_path = _result_path(model_gate, "selection_leaderboard")
    model_backtest_path = _result_path(model_gate, "backtest")
    model_acceptance_path = _result_path(model_gate, "acceptance")
    model_score_buckets_path = _result_path(model_gate, "score_buckets")
    model_pair_concentration_path = _result_path(model_gate, "pair_concentration")
    model_gain_concentration_path = _result_path(model_gate, "gain_concentration")
    model_failure_attribution_path = _result_path(model_gate, "failures")
    model_pair_support_path = _result_path(model_gate, "model_gate_pair_support_report")
    if not survivor_attribution_path.is_file():
        raise ValueError("registered survivor attribution evidence missing")
    rl_acceptance_path = _result_path(rl, "acceptance_report")
    rl_evaluation_path = _result_path(rl, "evaluation_report")
    rl_execution_path = _result_path(rl, "execution_backtest")
    rl_split_path = _result_path(rl, "split_audit")
    rl_leakage_path = _result_path(rl, "leakage_audit")
    rl_feature_schema_path = _result_path(rl, "feature_schema")
    rl_lineage_path = _result_path(rl, "lineage_report_json")
    model_authority_path = _result_path(governance, "model_authority")

    active_pointer = _read_json(active_pointer_path)
    metrics = _read_json(model_metrics_path)
    model_lineage = _read_json(model_lineage_path)
    predictions = _read_csv(model_predictions_path)
    selection_leaderboard = _read_csv(model_selection_leaderboard_path)
    model_backtest = _read_csv(model_backtest_path)
    model_acceptance = _read_csv(model_acceptance_path)
    score_buckets = _read_csv(model_score_buckets_path)
    pair_concentration = _read_csv(model_pair_concentration_path)
    gain_concentration = _read_csv(model_gain_concentration_path)
    failure_attribution = _read_csv(model_failure_attribution_path)
    pair_support = _read_csv(model_pair_support_path)
    survivor_attribution = _read_csv(survivor_attribution_path)
    rl_acceptance = _read_csv(rl_acceptance_path)
    rl_evaluation = _read_csv(rl_evaluation_path)
    rl_execution = _read_csv(rl_execution_path)
    rl_survivor_attribution = _read_csv(rl_survivor_attribution_path)
    rl_split = _read_csv(rl_split_path)
    rl_leakage = _read_csv(rl_leakage_path)
    rl_feature_schema = _read_json(rl_feature_schema_path)
    rl_lineage = _read_json(rl_lineage_path)
    model_authority = _read_json(model_authority_path)

    dataset_id = str(promoted.summary.get("dataset_id", ""))
    dataset_path = root / str(active_pointer.get("active_dataset_path", ""))
    active_dataset = _read_csv(dataset_path)
    pointer_hash = str(active_pointer.get("active_dataset_sha256", ""))
    best_model = str(metrics.get("best_model", ""))
    computed_selection_leaderboard = model_selection_leaderboard(predictions)
    replayed_best_model = (
        str(computed_selection_leaderboard.iloc[0]["model_name"])
        if not computed_selection_leaderboard.empty
        else ""
    )
    selected_model_predictions = _filter_predictions_to_model(predictions, best_model)
    selected_predictions = _filter_predictions_to_untouched_evaluation(selected_model_predictions)
    computed_model_backtest = pd.DataFrame()
    computed_model_acceptance = pd.DataFrame()
    if not selected_predictions.empty:
        computed_model_backtest = _model_gated_comparison(selected_predictions)
        computed_model_acceptance = _model_gated_acceptance(computed_model_backtest)
    computed_score_buckets = _score_bucket_report(selected_predictions)
    computed_gain = _model_gain_concentration(selected_predictions)
    computed_pair = _model_pair_concentration(selected_predictions)
    computed_survivor_attribution = _registered_survivor_attribution_frame(
        dataset=active_dataset,
        predictions=selected_predictions,
        survivor_support_policy=survivor_support_policy,
    )
    survivor_dataset_rows = pd.to_numeric(
        computed_survivor_attribution.get("dataset_rows", pd.Series(dtype=float)),
        errors="coerce",
    ).fillna(0)
    survivor_oos_rows = pd.to_numeric(
        computed_survivor_attribution.get("oos_prediction_rows", pd.Series(dtype=float)),
        errors="coerce",
    ).fillna(0)
    survivor_oos_folds = pd.to_numeric(
        computed_survivor_attribution.get("oos_folds", pd.Series(dtype=float)),
        errors="coerce",
    ).fillna(0)
    survivor_oos_regimes = pd.to_numeric(
        computed_survivor_attribution.get("oos_regimes", pd.Series(dtype=float)),
        errors="coerce",
    ).fillna(0)
    survivor_taken_rows = pd.to_numeric(
        computed_survivor_attribution.get("oos_taken_rows", pd.Series(dtype=float)),
        errors="coerce",
    ).fillna(0)
    survivor_take_rates = pd.to_numeric(
        computed_survivor_attribution.get("oos_take_rate", pd.Series(dtype=float)),
        errors="coerce",
    ).fillna(0.0)
    survivor_taken_return_rows = pd.to_numeric(
        computed_survivor_attribution.get("oos_taken_return_rows", pd.Series(dtype=float)),
        errors="coerce",
    ).fillna(0)
    survivor_taken_returns = pd.to_numeric(
        computed_survivor_attribution.get("oos_taken_return_sum", pd.Series(dtype=float)),
        errors="coerce",
    )
    survivor_support_status = computed_survivor_attribution.get(
        "support_status", pd.Series(dtype=str)
    ).astype(str)
    survivor_policy_ids = computed_survivor_attribution.get(
        "support_policy_id", pd.Series(dtype=str)
    ).astype(str)
    minimum_survivor_dataset_rows = int(survivor_support_policy["minimum_dataset_rows"])
    minimum_model_selection_folds = int(survivor_support_policy["minimum_model_selection_folds"])
    minimum_survivor_oos_rows = int(survivor_support_policy["minimum_oos_prediction_rows"])
    minimum_survivor_oos_folds = int(survivor_support_policy["minimum_oos_folds"])
    minimum_survivor_oos_regimes = int(survivor_support_policy["minimum_oos_regimes"])
    minimum_survivor_taken_rows = int(survivor_support_policy["minimum_oos_taken_rows"])
    minimum_survivor_take_rate = float(survivor_support_policy["minimum_oos_take_rate"])
    minimum_survivor_return_sum = float(
        survivor_support_policy["minimum_oos_taken_return_sum_exclusive"]
    )
    minimum_rl_take_rate_per_split = float(
        survivor_support_policy["minimum_rl_take_rate_per_split"]
    )
    minimum_model_filtered_trades = int(survivor_support_policy["minimum_model_filtered_trades"])
    minimum_model_gated_trades = int(survivor_support_policy["minimum_model_gated_trades"])
    minimum_nonempty_score_buckets = int(survivor_support_policy["minimum_nonempty_score_buckets"])
    minimum_rl_trades = int(survivor_support_policy["minimum_rl_trades"])
    maximum_model_gain_concentration = float(
        survivor_support_policy["maximum_model_gain_concentration"]
    )
    maximum_model_selection_concentration = float(
        survivor_support_policy["maximum_model_selection_concentration"]
    )
    maximum_rl_concentration = float(survivor_support_policy["maximum_rl_concentration"])

    required_gain_dimensions = {"pair", "timeframe", "regime", "strategy"}
    observed_gain_dimensions = set(computed_gain.get("dimension", pd.Series(dtype=str)).astype(str))
    prediction_scheme = selected_predictions.get("split_scheme", pd.Series(dtype=str)).astype(str)
    nonempty_buckets = (
        pd.to_numeric(score_buckets.get("rows", pd.Series(dtype=float)), errors="coerce")
        .fillna(0)
        .gt(0)
    )
    pair_share = pd.to_numeric(
        computed_pair.get("share_of_taken", pd.Series(dtype=float)),
        errors="coerce",
    )
    gain_share = pd.to_numeric(
        computed_gain.get("share_of_positive_returns", pd.Series(dtype=float)),
        errors="coerce",
    )
    model_acceptance_row = _single_row(model_acceptance)
    computed_model_row = _single_row(computed_model_acceptance)
    training_checks = {
        "runner_summary_accepted": _truthy(trained.summary.get("accepted")),
        "metrics_accepted": _truthy(metrics.get("accepted")),
        "active_dataset_lineage": bool(
            dataset_id
            and active_pointer.get("status") == "ACTIVE_RESEARCH_DATASET"
            and active_pointer.get("dataset_id") == dataset_id
            and dataset_path.is_file()
            and len(pointer_hash) == 64
            and _file_hash(dataset_path) == pointer_hash
        ),
        "model_dataset_lineage": bool(
            metrics.get("training_dataset_id") == dataset_id
            and model_lineage.get("training_dataset_id") == dataset_id
        ),
        "model_selection_isolation": bool(
            metrics.get("selection_isolation_scheme") == MODEL_SELECTION_ISOLATION_SCHEME
            and _truthy(metrics.get("model_selection_isolation_proven"))
            and int(metrics.get("selection_folds", 0) or 0) >= minimum_model_selection_folds
            and int(metrics.get("untouched_evaluation_folds", 0) or 0) >= minimum_survivor_oos_folds
            and _model_selection_isolation_proven(
                selected_model_predictions,
                minimum_selection_folds=minimum_model_selection_folds,
                minimum_evaluation_folds=minimum_survivor_oos_folds,
            )
        ),
        "model_selection_leaderboard_matches": _frames_match(
            selection_leaderboard,
            computed_selection_leaderboard,
        ),
        "best_model_replayed": bool(
            replayed_best_model
            and replayed_best_model == best_model
            and _truthy(metrics.get("model_winner_replayed"))
        ),
        "globally_purged_pair_aware_oos": bool(
            metrics.get("evaluation_scheme") == GLOBAL_PURGED_SPLIT_SCHEME
            and not prediction_scheme.empty
            and prediction_scheme.eq(GLOBAL_PURGED_SPLIT_SCHEME).all()
            and _global_label_purge_proven(selected_predictions)
            and selected_predictions.get("fold", pd.Series(dtype=float)).nunique() >= 2
        ),
        "minimum_take_rate": (_number(metrics, "median_take_rate") >= minimum_survivor_take_rate),
        "minimum_filtered_trades": (
            _number(metrics, "total_filtered_trades") >= minimum_model_filtered_trades
        ),
        "score_buckets_monotonic": bool(
            _truthy(metrics.get("score_buckets_monotonic"))
            and _score_buckets_monotonic(selected_predictions)
            and int(nonempty_buckets.sum()) >= minimum_nonempty_score_buckets
            and score_buckets.get("monotonic", pd.Series(dtype=object)).map(_truthy).all()
        ),
        "gain_dimensions_complete": observed_gain_dimensions == required_gain_dimensions,
        "gain_concentration_within_limit": bool(
            not gain_share.empty and gain_share.max() <= maximum_model_gain_concentration + 1e-12
        ),
        "pair_selection_concentration_within_limit": bool(
            not pair_share.empty and pair_share.max() <= maximum_model_selection_concentration
        ),
        "pair_timeframe_regime_selection_concentration_within_limit": (
            _selection_concentration_within_limit(
                computed_gain,
                dimensions=supervised_protocol["selection_concentration_dimensions"],
                maximum=maximum_model_selection_concentration,
            )
        ),
        "metrics_report_no_failures": not str(metrics.get("failing_checks", "")).strip(),
        "model_lineage_research_only": bool(
            _authority_safe_json(model_lineage) and _authority_safe_frame(selection_leaderboard)
        ),
        "registered_survivor_dataset_support": bool(
            not survivor_dataset_rows.empty
            and survivor_dataset_rows.ge(minimum_survivor_dataset_rows).all()
        ),
        "registered_survivor_support_policy_bound": bool(
            not survivor_policy_ids.empty
            and survivor_policy_ids.eq(survivor_support_policy["support_policy_id"]).all()
        ),
    }
    model_checks = {
        "runner_summary_accepted": _truthy(model_gate.summary.get("accepted")),
        "acceptance_artifact_accepted": _truthy(model_acceptance_row.get("accepted")),
        "acceptance_recomputed": _truthy(computed_model_row.get("accepted")),
        "acceptance_artifact_matches_recomputed": _acceptance_rows_match(
            artifact=model_acceptance_row,
            recomputed=computed_model_row,
        ),
        "backtest_artifact_matches_recomputed": _frames_match(
            model_backtest,
            computed_model_backtest,
        ),
        "score_bucket_artifact_matches_recomputed": _frames_match(
            score_buckets,
            computed_score_buckets,
        ),
        "acceptance_artifact_take_rate": (
            _number(model_acceptance_row, "gated_take_rate") >= minimum_survivor_take_rate
        ),
        "acceptance_artifact_trades": (
            _number(model_acceptance_row, "gated_trades") >= minimum_model_gated_trades
        ),
        "predictions_present": not selected_predictions.empty,
        "predictions_match_active_dataset": (
            _model_predictions_match_active_dataset(
                selected_model_predictions,
                active_dataset,
            )
        ),
        "prediction_membership_matches_registered_protocol": (
            walkforward_prediction_membership_matches(
                selected_model_predictions,
                active_dataset,
                n_splits=supervised_protocol["walkforward_splits"],
                min_train_rows=supervised_protocol["minimum_train_rows"],
                embargo_periods=supervised_protocol["embargo_periods"],
            )
        ),
        "gated_take_rate": (
            _number(computed_model_row, "gated_take_rate") >= minimum_survivor_take_rate
        ),
        "gated_trades": (_number(computed_model_row, "gated_trades") >= minimum_model_gated_trades),
        "gain_artifact_complete": set(
            gain_concentration.get("dimension", pd.Series(dtype=str)).astype(str)
        )
        == required_gain_dimensions,
        "gain_artifact_matches_recomputed": _frames_match(
            gain_concentration,
            computed_gain,
        ),
        "gain_artifact_within_limit": bool(
            not gain_concentration.empty
            and pd.to_numeric(
                gain_concentration["share_of_positive_returns"],
                errors="coerce",
            ).max()
            <= maximum_model_gain_concentration + 1e-12
        ),
        "pair_artifact_within_limit": bool(
            not pair_concentration.empty
            and pd.to_numeric(pair_concentration["share_of_taken"], errors="coerce").max()
            <= maximum_model_selection_concentration + 1e-12
        ),
        "pair_timeframe_regime_selection_artifact_within_limit": (
            _selection_concentration_within_limit(
                gain_concentration,
                dimensions=supervised_protocol["selection_concentration_dimensions"],
                maximum=maximum_model_selection_concentration,
            )
        ),
        "pair_artifact_matches_recomputed": _frames_match(
            pair_concentration,
            computed_pair,
        ),
        "registered_survivor_attribution_matches": _frames_match(
            survivor_attribution,
            computed_survivor_attribution,
        ),
        "registered_survivor_oos_support": bool(
            not survivor_oos_rows.empty and survivor_oos_rows.ge(minimum_survivor_oos_rows).all()
        ),
        "registered_survivor_oos_fold_depth": bool(
            not survivor_oos_folds.empty and survivor_oos_folds.ge(minimum_survivor_oos_folds).all()
        ),
        "registered_survivor_oos_regime_depth": bool(
            not survivor_oos_regimes.empty
            and survivor_oos_regimes.ge(minimum_survivor_oos_regimes).all()
        ),
        "registered_survivor_taken_support": bool(
            not survivor_taken_rows.empty
            and survivor_taken_rows.ge(minimum_survivor_taken_rows).all()
        ),
        "registered_survivor_take_rate": bool(
            not survivor_take_rates.empty
            and survivor_take_rates.ge(minimum_survivor_take_rate).all()
        ),
        "registered_survivor_return_coverage": bool(
            not survivor_taken_return_rows.empty
            and survivor_taken_return_rows.eq(survivor_taken_rows).all()
        ),
        "registered_survivor_positive_after_cost_return": bool(
            not survivor_taken_returns.empty
            and np.isfinite(survivor_taken_returns).all()
            and survivor_taken_returns.gt(minimum_survivor_return_sum).all()
        ),
        "registered_survivor_support_status_ready": bool(
            not survivor_support_status.empty and survivor_support_status.eq("READY").all()
        ),
        "pair_support_covers_predictions": bool(
            not selected_predictions.empty
            and set(selected_predictions["pair"].astype(str)).issubset(
                set(pair_support.get("pair", pd.Series(dtype=str)).astype(str))
            )
        ),
        "failure_and_support_artifacts_research_only": bool(
            _authority_safe_frame(failure_attribution)
            and _authority_safe_frame(pair_support)
            and _authority_safe_frame(survivor_attribution)
        ),
        "artifacts_research_only": all(
            _authority_safe_frame(frame)
            for frame in (
                model_acceptance,
                model_backtest,
                score_buckets,
                pair_concentration,
                gain_concentration,
            )
        ),
    }

    computed_rl_evaluation = _rl_evaluation_from_execution(rl_execution)
    computed_rl = rl_acceptance_report(computed_rl_evaluation)
    rl_acceptance_row = _single_row(rl_acceptance)
    computed_rl_row = _single_row(computed_rl)
    computed_rl_survivor_attribution = _registered_survivor_rl_attribution_frame(
        dataset=active_dataset,
        execution=rl_execution,
        survivor_support_policy=survivor_support_policy,
    )
    rl_survivor_status = computed_rl_survivor_attribution.get(
        "support_status", pd.Series(dtype=str)
    ).astype(str)
    rl_survivor_policy_ids = computed_rl_survivor_attribution.get(
        "support_policy_id", pd.Series(dtype=str)
    ).astype(str)
    _, recomputed_rl_partitions, recomputed_rl_split = _chronological_rl_partitions(
        active_dataset,
        train_fraction=rl_protocol["train_fraction"],
        validation_fraction=rl_protocol["validation_fraction"],
        minimum_rows=tuple(rl_protocol["minimum_rows"]),
    )
    split_names = set(rl_split.get("split", pd.Series(dtype=str)).astype(str))
    forbidden_policy_inputs = {
        "hold_bars",
        "trade_bars",
        "max_adverse_excursion",
        "max_favorable_excursion",
        "exit_timestamp",
    }
    concentration_fields = (
        "pair_concentration",
        "pair_pnl_concentration",
        "timeframe_concentration",
        "timeframe_pnl_concentration",
        "regime_concentration",
        "regime_pnl_concentration",
    )
    rl_checks = {
        "runner_summary_accepted": bool(
            _truthy(rl.summary.get("accepted"))
            and _truthy(computed_rl_row.get("accepted"))
        ),
        "acceptance_artifact_accepted": _truthy(rl_acceptance_row.get("accepted")),
        "acceptance_recomputed": _truthy(computed_rl_row.get("accepted")),
        "acceptance_artifact_matches_recomputed": _acceptance_rows_match(
            artifact=rl_acceptance_row,
            recomputed=computed_rl_row,
        ),
        "evaluation_artifact_matches_raw_execution": (
            _rl_evaluation_matches_execution(
                rl_evaluation,
                computed_rl_evaluation,
            )
        ),
        "validation_passed": _truthy(computed_rl_row.get("validation_passed")),
        "held_out_test_passed": _truthy(computed_rl_row.get("held_out_test_passed")),
        "out_of_sample_evidence": _truthy(computed_rl_row.get("out_of_sample_evidence")),
        "split_audit_ready": bool(
            split_names == {"train", "validation", "test"}
            and rl_split.get("status", pd.Series(dtype=str)).astype(str).eq("ready").all()
            and rl_split.get("global_label_purge", pd.Series(dtype=object)).map(_truthy).all()
            and set(rl_split.get("selection_use", pd.Series(dtype=str)).astype(str))
            == {"diagnostic_only", "policy_selection", "untouched_evaluation"}
        ),
        "split_audit_matches_active_dataset": (
            _rl_split_audit_matches_dataset(
                rl_split,
                recomputed_rl_split,
            )
        ),
        "global_label_purge_recomputed": _rl_split_chronology_proven(recomputed_rl_split),
        "execution_membership_matches_active_dataset": (
            _rl_execution_matches_partitions(
                rl_execution,
                recomputed_rl_partitions,
            )
        ),
        "leakage_audit_safe": bool(
            not rl_leakage.empty
            and not rl_leakage.get("uses_future_data", pd.Series(True, index=rl_leakage.index))
            .map(_truthy)
            .any()
            and rl_leakage.get("global_label_purge", pd.Series(False, index=rl_leakage.index))
            .map(_truthy)
            .all()
            and rl_leakage.get("split_status", pd.Series("", index=rl_leakage.index))
            .astype(str)
            .eq("ready")
            .all()
            and rl_leakage.get("leakage_blocker", pd.Series("", index=rl_leakage.index))
            .fillna("")
            .astype(str)
            .str.strip()
            .eq("")
            .all()
        ),
        "minimum_take_rate": (
            _number(computed_rl_row, "rl_take_rate") >= minimum_rl_take_rate_per_split
        ),
        "minimum_trades": (_number(computed_rl_row, "rl_trades") >= minimum_rl_trades),
        "concentration_within_limit": all(
            _number(computed_rl_row, field, default=1.0) <= maximum_rl_concentration
            for field in concentration_fields
        ),
        "dataset_lineage": bool(
            rl_lineage.get("training_dataset_id") == dataset_id
            and rl_lineage.get("training_dataset_sha256") == pointer_hash
            and _truthy(rl_lineage.get("accepted"))
        ),
        "registered_protocol_lineage": bool(
            rl_lineage.get("registered_policy_parameters") == rl_protocol
        ),
        "execution_policy_matches_registered_protocol": bool(
            not rl_execution.empty
            and rl_execution.get("policy_name", pd.Series(dtype=str))
            .astype(str)
            .eq("simulated_quantile_hold_policy")
            .all()
            and pd.to_numeric(
                rl_execution.get(
                    "entry_threshold_calibration_quantile",
                    pd.Series(dtype=float),
                ),
                errors="coerce",
            )
            .eq(rl_protocol["entry_threshold_quantile"])
            .all()
        ),
        "registered_survivor_attribution_matches": _frames_match(
            rl_survivor_attribution,
            computed_rl_survivor_attribution,
        ),
        "registered_survivor_support_policy_bound": bool(
            not rl_survivor_policy_ids.empty
            and rl_survivor_policy_ids.eq(survivor_support_policy["support_policy_id"]).all()
        ),
        "registered_survivor_validation_and_test_ready": bool(
            not rl_survivor_status.empty and rl_survivor_status.eq("READY").all()
        ),
        "causal_feature_schema": bool(
            forbidden_policy_inputs.issubset(
                set(rl_feature_schema.get("forbidden_policy_action_inputs", []))
            )
            and not _truthy(rl_feature_schema.get("testnet_order_authority"))
            and not _truthy(rl_feature_schema.get("live_trading_authorized"))
        ),
        "artifacts_research_only": bool(
            _authority_safe_json(rl_lineage)
            and _authority_safe_frame(rl_acceptance)
            and _authority_safe_frame(rl_evaluation)
            and _authority_safe_frame(rl_execution)
            and _authority_safe_frame(rl_survivor_attribution)
            and _authority_safe_frame(rl_split)
            and _authority_safe_frame(rl_leakage)
        ),
    }
    authority_blockers = {
        str(value).strip() for value in model_authority.get("blockers", []) if str(value).strip()
    }
    governance_checks = {
        "runner_status_pass": governance.summary.get("status") == "PASS",
        "runner_summary_research_only": _authority_safe_json(governance.summary),
        "authority_schema": model_authority.get("schema_version")
        == "thewiz.agent_learning_governance.v1",
        "authority_dataset_lineage": bool(
            model_authority.get("active_dataset_id") == dataset_id
            and _truthy(model_authority.get("training_dataset_lineage_ready"))
            and _truthy(model_authority.get("model_active_dataset_lineage_matches"))
            and _truthy(model_authority.get("model_artifact_hash_matches"))
            and _truthy(model_authority.get("rl_active_dataset_lineage_matches"))
        ),
        "authority_model_evidence": bool(
            _truthy(model_authority.get("out_of_sample_incremental_edge_accepted"))
            and _truthy(model_authority.get("exact_mode_trade_provenance_ready"))
            and _truthy(model_authority.get("strict_cost_training_evidence_ready"))
            and not model_authority.get("training_dataset_lineage_blockers", [])
        ),
        "authority_rl_evidence": bool(
            _truthy(model_authority.get("rl_global_label_purge_ready"))
            and _truthy(model_authority.get("rl_validation_passed"))
            and _truthy(model_authority.get("rl_held_out_test_passed"))
            and _truthy(model_authority.get("rl_out_of_sample_accepted"))
        ),
        "authority_research_only_state": bool(
            model_authority.get("model_authority") == "RESEARCH_ONLY"
            and authority_blockers.issubset({"realized_testnet_sample_not_available"})
            and not _truthy(model_authority.get("quantization_authorized"))
        ),
        "model_authority_research_only": _authority_safe_json(model_authority),
    }
    groups = {
        "training": training_checks,
        "model": model_checks,
        "rl": rl_checks,
        "governance": governance_checks,
    }
    flattened = {
        f"{group}.{check}": bool(passed)
        for group, checks in groups.items()
        for check, passed in checks.items()
    }
    blockers = [name for name, passed in flattened.items() if not passed]
    training_accepted = all(training_checks.values())
    model_accepted = all(model_checks.values())
    rl_accepted = all(rl_checks.values())
    governance_pass = all(governance_checks.values())
    return {
        "training_accepted_oos": training_accepted,
        "model_accepted_oos": model_accepted,
        "rl_accepted_oos": rl_accepted,
        "agent_governance_pass": governance_pass,
        "stage5_research_gate_pass": bool(
            training_accepted and model_accepted and rl_accepted and governance_pass
        ),
        "acceptance_checks": flattened,
        "acceptance_blockers": blockers,
    }


def _selection_concentration_within_limit(
    frame: pd.DataFrame,
    *,
    dimensions: list[str],
    maximum: float,
) -> bool:
    """Require complete, normalized taken-trade shares for each frozen dimension."""

    required_columns = {"dimension", "share_of_taken"}
    if (
        frame.empty
        or not required_columns.issubset(frame.columns)
        or not dimensions
        or len(dimensions) != len(set(dimensions))
        or not 0.0 < maximum <= 1.0
    ):
        return False
    observed = frame["dimension"].astype(str)
    for dimension in dimensions:
        shares = pd.to_numeric(
            frame.loc[observed.eq(dimension), "share_of_taken"],
            errors="coerce",
        )
        if (
            shares.empty
            or not np.isfinite(shares).all()
            or shares.lt(0.0).any()
            or shares.gt(1.0).any()
            or not np.isclose(shares.sum(), 1.0, rtol=1e-12, atol=1e-12)
            or shares.max() > maximum + 1e-12
        ):
            return False
    return True


def _result_path(result: CommandResult, key: str) -> Path:
    path = Path(result.paths.get(key) or "")
    if not path.is_file():
        raise ValueError(f"registered learning evidence missing: {key}")
    return path


def _single_row(frame: pd.DataFrame) -> dict[str, Any]:
    if len(frame) != 1:
        return {}
    return frame.iloc[0].to_dict()


def _acceptance_rows_match(*, artifact: dict[str, Any], recomputed: dict[str, Any]) -> bool:
    if not artifact or not recomputed:
        return False
    for key, expected in recomputed.items():
        if key not in artifact:
            return False
        observed = artifact[key]
        if pd.isna(observed) and expected in {None, ""}:
            continue
        if isinstance(expected, (bool, np.bool_)):
            if _truthy(observed) is not bool(expected):
                return False
        elif isinstance(expected, (int, float, np.number)):
            try:
                expected_number = float(expected)
                observed_number = float(observed)
            except (TypeError, ValueError):
                return False
            if not np.isclose(
                observed_number,
                expected_number,
                rtol=1e-12,
                atol=1e-12,
                equal_nan=True,
            ):
                return False
        elif str(observed) != str(expected):
            return False
    return True


def _frames_match(artifact: pd.DataFrame, recomputed: pd.DataFrame) -> bool:
    if set(artifact.columns) != set(recomputed.columns) or len(artifact) != len(recomputed):
        return False
    columns = sorted(artifact.columns)
    left = artifact[columns].sort_values(columns).reset_index(drop=True)
    right = recomputed[columns].sort_values(columns).reset_index(drop=True)
    for column in columns:
        if not (
            pd.api.types.is_numeric_dtype(left[column])
            and pd.api.types.is_numeric_dtype(right[column])
        ):
            left[column] = left[column].fillna("").astype(str)
            right[column] = right[column].fillna("").astype(str)
    try:
        pd.testing.assert_frame_equal(
            left,
            right,
            check_dtype=False,
            check_categorical=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )
    except AssertionError:
        return False
    return True


def _number(payload: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = float(payload.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) or value == float("inf") else default


def _authority_safe_json(payload: dict[str, Any]) -> bool:
    return bool(payload) and all(
        not _truthy(payload.get(field))
        for field in (
            "order_submission_performed",
            "promotion_authority",
            "testnet_candidate_authority",
            "testnet_order_authority",
            "live_trading_authorized",
        )
    )


def _authority_safe_frame(frame: pd.DataFrame) -> bool:
    return all(
        not frame.get(field, pd.Series(False, index=frame.index)).map(_truthy).any()
        for field in (
            "order_submission_performed",
            "promotion_authority",
            "testnet_candidate_authority",
            "testnet_order_authority",
            "live_trading_authorized",
        )
    )


def _stage_manifest(root: Path, stage: dict[str, Any]) -> dict[str, Any]:
    path = root / str(stage.get("manifest_path", ""))
    manifest = _read_json(path)
    if not manifest:
        raise ValueError("registered learning stage manifest is missing")
    return manifest


def _bound_manifest_artifact(
    root: Path,
    stage: dict[str, Any],
    manifest: dict[str, Any],
    key: str,
) -> Path:
    relative = str(manifest.get("artifacts", {}).get(key, ""))
    path = root / relative
    expected = stage.get("immutable_artifact_hashes", {}).get(relative)
    if not relative or not expected or _file_hash(path) != str(expected):
        raise ValueError(f"registered learning artifact is not hash-bound: {key}")
    return path


def _validate_walkforward_ledgers(*, trades: pd.DataFrame, bars: pd.DataFrame) -> None:
    required_identity = {
        "experiment_id",
        "pair_group_key",
        "pair",
        "exact_mode",
        "orientation",
        "fold_number",
        "trade_id",
        "entry_timestamp",
        "exit_timestamp",
        "profit_after_cost",
    }
    missing = sorted((required_identity | REQUIRED_CAUSAL_FEATURES) - set(trades))
    if trades.empty or missing:
        raise ValueError(
            "registered learning walk-forward trades missing causal fields: " + ";".join(missing)
        )
    if bars.empty or not {
        "experiment_id",
        "fold_number",
        "timestamp",
        "net_return",
    }.issubset(bars):
        raise ValueError("registered learning walk-forward bar ledger is incomplete")


def _join_causal_regimes(cohort: pd.DataFrame, regime_trades: pd.DataFrame) -> pd.DataFrame:
    joined = cohort.copy()
    keys = ["experiment_id", "fold_number", "trade_id", "entry_timestamp"]
    regime_columns = [
        column
        for column in (
            "regime",
            "regime_uses_future_data",
            "pair_volatility_20",
            "rolling_correlation_60",
            "regime_attribution_id",
        )
        if column in regime_trades.columns
    ]
    if not regime_trades.empty and set(keys).issubset(regime_trades):
        lookup = regime_trades[keys + regime_columns].drop_duplicates(keys)
        joined = joined.merge(lookup, on=keys, how="left", validate="one_to_one")
    future = (
        joined.get("regime_uses_future_data", pd.Series(False, index=joined.index))
        .fillna(False)
        .map(_truthy)
    )
    if future.any():
        raise ValueError("registered learning regime attribution uses future data")
    fallback = np.select(
        [
            pd.to_numeric(joined["realized_volatility_percentile"], errors="coerce").ge(0.75),
            pd.to_numeric(joined["correlation"], errors="coerce").le(0.25),
        ],
        ["high_vol", "correlation_break"],
        default="calm",
    )
    joined["regime"] = (
        joined.get("regime", pd.Series("", index=joined.index)).fillna("").astype(str)
    )
    joined.loc[joined["regime"].eq(""), "regime"] = fallback[joined["regime"].eq("")]
    joined["regime_uses_future_data"] = False
    return joined


def _materialize_learning_rows(
    *,
    cohort: pd.DataFrame,
    bars: pd.DataFrame,
    receipt: dict[str, Any],
    evidence_paths: tuple[Path, ...],
    root: Path,
) -> pd.DataFrame:
    frame = cohort.copy()
    frame["source_trade_id"] = frame["trade_id"]
    identities = frame.apply(
        lambda row: "|".join(
            (
                str(receipt.get("execution_id", "")),
                str(row.get("experiment_id", "")),
                str(row.get("fold_number", "")),
                str(row.get("source_trade_id", "")),
                str(row.get("entry_timestamp", "")),
            )
        ),
        axis=1,
    )
    frame["trade_id"] = identities.map(
        lambda value: "registeredtrade_" + sha256(value.encode()).hexdigest()[:24]
    )
    if frame["trade_id"].duplicated().any():
        raise ValueError("registered learning trade identity collision")
    frame["source_venue"] = "hyperliquid"
    frame["timeframe"] = frame.get("hyperliquid_interval", pd.Series("", index=frame.index))
    frame["signal_side"] = frame.get("side", pd.Series(0, index=frame.index)).map(
        lambda value: "long_x_short_y" if float(value) < 0 else "short_x_long_y"
    )
    returns = pd.to_numeric(frame["profit_after_cost"], errors="coerce")
    frame["good_trade"] = returns.gt(0.0).astype(int)
    frame["label_profitable"] = frame["good_trade"]
    frame["realized_return"] = returns
    frame["hold_bars"] = pd.to_numeric(frame.get("bars"), errors="coerce").fillna(0)
    frame["return_aggregation"] = "compounded_bar_returns_zero_floor"
    frame["return_unit"] = "fraction_of_equity"
    frame["backtest_label"] = True
    frame["paper_label"] = ""
    frame["live_label"] = ""
    frame["feature_completeness_score"] = (
        frame[sorted(REQUIRED_CAUSAL_FEATURES)].notna().mean(axis=1).round(4)
    )
    frame["evidence_path"] = ";".join(_relative(path, root) for path in evidence_paths)
    frame = _attach_excursions(frame, bars)
    feature_time = pd.to_datetime(
        frame["feature_timestamp"], utc=True, errors="coerce", format="mixed"
    )
    entry_time = pd.to_datetime(frame["entry_timestamp"], utc=True, errors="coerce", format="mixed")
    label_time = pd.to_datetime(frame["label_timestamp"], utc=True, errors="coerce", format="mixed")
    if (
        feature_time.isna().any()
        or entry_time.isna().any()
        or label_time.isna().any()
        or feature_time.gt(entry_time).any()
        or entry_time.ge(label_time).any()
    ):
        raise ValueError("registered learning feature-label timestamp contract failed")
    frame["registered_contract_id"] = receipt["contract_id"]
    frame["registered_execution_id"] = receipt["execution_id"]
    frame["promotion_authority"] = False
    frame["testnet_order_authority"] = False
    frame["live_trading_authorized"] = False
    return frame


def _attach_excursions(trades: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    result = trades.copy()
    working = bars.copy()
    working["_time"] = pd.to_datetime(
        working["timestamp"], utc=True, errors="coerce", format="mixed"
    )
    working["_net"] = pd.to_numeric(working["net_return"], errors="coerce").fillna(0.0)
    groups = {
        (str(experiment), str(fold)): group.sort_values("_time")
        for (experiment, fold), group in working.groupby(
            ["experiment_id", "fold_number"], dropna=False
        )
    }
    adverse: list[float] = []
    favorable: list[float] = []
    for row in result.itertuples():
        group = groups.get((str(row.experiment_id), str(row.fold_number)))
        entry = pd.to_datetime(row.entry_timestamp, utc=True, errors="coerce")
        exit_at = pd.to_datetime(row.exit_timestamp, utc=True, errors="coerce")
        if group is None or pd.isna(entry) or pd.isna(exit_at):
            adverse.append(float("nan"))
            favorable.append(float("nan"))
            continue
        segment = group.loc[group["_time"].between(entry, exit_at), "_net"]
        path = (1.0 + segment.clip(lower=-1.0)).cumprod().sub(1.0)
        adverse.append(float(min(0.0, path.min())) if not path.empty else 0.0)
        favorable.append(float(max(0.0, path.max())) if not path.empty else 0.0)
    result["max_adverse_excursion"] = adverse
    result["max_favorable_excursion"] = favorable
    return result


def _validate_learning_receipt(path: Path, *, root: Path, stage4: dict[str, Any]) -> dict[str, Any]:
    receipt = _read_json(path)
    expected = str(receipt.pop("receipt_sha256", ""))
    if not expected or _payload_hash(receipt) != expected:
        raise ValueError("registered learning immutable receipt hash mismatch")
    receipt["receipt_sha256"] = expected
    if receipt.get("registered_execution_id") != stage4.get("execution_id"):
        raise ValueError("registered learning Stage 4 lineage mismatch")
    if not _stage4_has_accepted_survivor(stage4):
        raise ValueError("registered learning Stage 4 survivor is not accepted")
    roles = receipt.get("artifact_roles", {})
    if not isinstance(roles, dict) or set(roles) != REQUIRED_LEARNING_EVIDENCE_ROLES:
        raise ValueError("registered learning evidence roles are incomplete")
    for relative, digest in receipt.get("artifact_hashes", {}).items():
        if _file_hash(root / str(relative)) != str(digest):
            raise ValueError("registered learning artifact hash mismatch")
    role_paths = {
        role: _receipt_artifact_path(root, str(relative)) for role, relative in roles.items()
    }
    if set(receipt.get("artifact_hashes", {})) != {str(relative) for relative in roles.values()}:
        raise ValueError("registered learning artifact role/hash coverage mismatch")
    protocol_path, protocol = validate_registered_stage5_protocol(
        root=root,
        stage4_completed_at_utc=str(stage4.get("completed_at_utc", "")),
    )
    if (
        role_paths["registered_stage5_protocol"].resolve() != protocol_path.resolve()
        or receipt.get("registered_stage5_protocol_id") != protocol.get("protocol_id")
        or receipt.get("registered_stage5_protocol_sha256") != _file_hash(protocol_path)
    ):
        raise ValueError("registered learning Stage 5 protocol lineage mismatch")
    recomputed = _reconcile_registered_learning_acceptance(
        root=root,
        promoted=CommandResult(
            paths={"active_pointer": role_paths["active_dataset_pointer"]},
            summary={"dataset_id": receipt.get("training_dataset_id", "")},
        ),
        trained=CommandResult(
            paths={
                "metrics": role_paths["model_metrics"],
                "lineage": role_paths["model_lineage"],
            },
            summary={"accepted": receipt.get("training_accepted_oos", False)},
        ),
        model_gate=CommandResult(
            paths={
                "predictions": role_paths["model_predictions"],
                "selection_leaderboard": role_paths["model_selection_leaderboard"],
                "backtest": role_paths["model_backtest"],
                "acceptance": role_paths["model_acceptance"],
                "score_buckets": role_paths["model_score_buckets"],
                "pair_concentration": role_paths["model_pair_concentration"],
                "gain_concentration": role_paths["model_gain_concentration"],
                "failures": role_paths["model_failure_attribution"],
                "model_gate_pair_support_report": role_paths["model_pair_support"],
            },
            summary={"accepted": receipt.get("model_accepted_oos", False)},
        ),
        rl=CommandResult(
            paths={
                "acceptance_report": role_paths["rl_acceptance"],
                "evaluation_report": role_paths["rl_evaluation"],
                "execution_backtest": role_paths["rl_execution_backtest"],
                "split_audit": role_paths["rl_split_audit"],
                "leakage_audit": role_paths["rl_leakage_audit"],
                "feature_schema": role_paths["rl_feature_schema"],
                "lineage_report_json": role_paths["rl_lineage"],
            },
            summary={"accepted": receipt.get("rl_accepted_oos", False)},
        ),
        governance=CommandResult(
            paths={"model_authority": role_paths["model_authority"]},
            summary={"status": ("PASS" if receipt.get("agent_governance_pass") else "BLOCKED")},
        ),
        survivor_attribution_path=role_paths["registered_survivor_attribution"],
        survivor_support_policy_path=role_paths["registered_survivor_support_policy"],
        rl_survivor_attribution_path=role_paths["registered_survivor_rl_attribution"],
        stage5_protocol=protocol,
    )
    for field in (
        "training_accepted_oos",
        "model_accepted_oos",
        "rl_accepted_oos",
        "agent_governance_pass",
        "stage5_research_gate_pass",
        "acceptance_checks",
        "acceptance_blockers",
    ):
        if receipt.get(field) != recomputed.get(field):
            raise ValueError(f"registered learning acceptance reconciliation mismatch: {field}")
    expected_status = (
        "PASS_RESEARCH_LEARNING_GATES"
        if receipt.get("stage5_research_gate_pass")
        else "REJECTED_RESEARCH_LEARNING_GATES"
    )
    if receipt.get("status") != expected_status:
        raise ValueError("registered learning receipt status mismatch")
    _assert_no_authority("registered_learning_receipt", receipt)
    return receipt


def latest_verified_registered_learning(*, root: Path = ROOT) -> dict[str, Any]:
    """Recompute the active immutable Stage 5 result without trusting its pointer."""

    active_path = root / "reports" / "active" / "registered_learning_research_status.json"
    active = _read_json(active_path)
    base = {
        "active_status": str(active.get("status", "MISSING")),
        "registered_execution_id": str(active.get("registered_execution_id", "")),
        "learning_receipt_path": "",
        "learning_receipt_sha256": "",
        "evidence_valid": False,
        "stage5_research_gate_pass": False,
        "acceptance_blockers": [],
        "research_only": True,
        "order_submission_performed": False,
        "promotion_authority": False,
        "testnet_candidate_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }

    try:
        _assert_no_authority("registered_learning_active_status", active)
        active_status = str(active.get("status", ""))
        if active_status not in {
            "PASS_RESEARCH_LEARNING_GATES",
            "REJECTED_RESEARCH_LEARNING_GATES",
            "ALREADY_COMPLETE",
        }:
            raise ValueError("registered learning active status is not terminal")
        raw_path = str(active.get("learning_receipt_path", "")).strip()
        if not raw_path:
            raise ValueError("registered learning receipt path is missing")
        candidate = Path(raw_path)
        receipt_path = candidate if candidate.is_absolute() else root / candidate
        expected_parent = (root / "data" / "research" / "registered_learning").resolve()
        try:
            receipt_path.resolve().relative_to(expected_parent)
        except (OSError, ValueError) as exc:
            raise ValueError("registered learning receipt path is unsafe") from exc
        supplied_file_hash = str(active.get("learning_receipt_sha256", ""))
        if (
            not receipt_path.is_file()
            or len(supplied_file_hash) != 64
            or _file_hash(receipt_path) != supplied_file_hash
        ):
            raise ValueError("registered learning active receipt hash mismatch")
        untrusted_learning = _read_json(receipt_path)
        untrusted_roles = untrusted_learning.get("artifact_roles", {})
        if not isinstance(untrusted_roles, dict):
            raise TypeError("registered learning artifact roles are missing")
        stage4_receipt_path = _receipt_artifact_path(
            root,
            str(untrusted_roles.get("stage4_execution_receipt", "")),
        )
        _, stage4, _ = _registered_execution(
            root=root,
            execution_receipt_path=stage4_receipt_path,
        )
        learning = _validate_learning_receipt(
            receipt_path,
            root=root,
            stage4=stage4,
        )
        terminal_status = str(learning.get("status", ""))
        if active_status != "ALREADY_COMPLETE" and active_status != terminal_status:
            raise ValueError("registered learning active and immutable status mismatch")
        if active.get("registered_execution_id") != learning.get("registered_execution_id"):
            raise ValueError("registered learning active execution lineage mismatch")
        gate_pass = bool(learning.get("stage5_research_gate_pass", False))
        if bool(active.get("stage5_research_gate_pass", False)) != gate_pass:
            raise ValueError("registered learning active gate result mismatch")
    except (FileNotFoundError, ValueError, TypeError, KeyError) as exc:
        return {
            **base,
            "status": "BLOCKED_REGISTERED_LEARNING_EVIDENCE",
            "blockers": [safe_validation_exception_code(exc)],
        }

    relative = _relative(receipt_path, root)
    acceptance_blockers = [
        str(value) for value in learning.get("acceptance_blockers", []) if str(value)
    ]
    return {
        **base,
        "status": (
            "PASS_VERIFIED_REGISTERED_LEARNING_ACCEPTANCE"
            if gate_pass
            else "PASS_VERIFIED_REGISTERED_LEARNING_REJECTION"
        ),
        "active_status": active_status,
        "registered_execution_id": str(learning.get("registered_execution_id", "")),
        "learning_id": str(learning.get("learning_id", "")),
        "learning_receipt_path": relative,
        "learning_receipt_sha256": supplied_file_hash,
        "evidence_valid": True,
        "stage5_research_gate_pass": gate_pass,
        "acceptance_blockers": acceptance_blockers,
        "blockers": acceptance_blockers,
    }


def _receipt_artifact_path(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("registered learning artifact path is unsafe")
    path = root / candidate
    if not path.is_file():
        raise ValueError("registered learning artifact path is missing")
    return path


def _required_artifact_hashes(artifacts: dict[str, Path], *, root: Path) -> dict[str, str]:
    missing = sorted(name for name, path in artifacts.items() if not path.is_file())
    if missing:
        raise ValueError("registered learning required artifacts missing: " + ";".join(missing))
    return {_relative(path, root): _file_hash(path) for path in artifacts.values()}


def _publish_status(
    *,
    status_path: Path,
    status: str,
    blocker: str,
    execute: bool,
    learning: dict[str, Any] | None = None,
    receipt_path: Path | None = None,
    extra: dict[str, Any] | None = None,
) -> CommandResult:
    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": status,
        "execution_requested": execute,
        "blocker": blocker,
        "registered_execution_id": str((learning or {}).get("registered_execution_id", "")),
        "stage5_research_gate_pass": bool((learning or {}).get("stage5_research_gate_pass", False)),
        "order_submission_performed": False,
        "promotion_authority": False,
        "testnet_candidate_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
        **(extra or {}),
    }
    paths = {"status": status_path}
    if receipt_path is not None:
        summary["learning_receipt_path"] = str(receipt_path)
        summary["learning_receipt_sha256"] = _file_hash(receipt_path)
        paths["learning_receipt"] = receipt_path
    _atomic_json(summary, status_path)
    return CommandResult(paths=paths, summary=summary)


def _assert_no_authority(name: str, payload: dict[str, Any]) -> None:
    for field in (
        "order_submission_performed",
        "promotion_authority",
        "testnet_candidate_authority",
        "testnet_order_authority",
        "live_trading_authorized",
    ):
        if _truthy(payload.get(field)):
            raise ValueError(f"{name} unexpectedly acquired {field}")


def _load_survivor_support_policy(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = _read_json(path)
    if payload.get("schema_version") != SURVIVOR_SUPPORT_POLICY_SCHEMA:
        raise ValueError("unsupported registered survivor support policy schema")
    required_positive_integers = (
        "minimum_dataset_rows",
        "minimum_model_selection_folds",
        "minimum_oos_prediction_rows",
        "minimum_oos_folds",
        "minimum_oos_regimes",
        "minimum_oos_taken_rows",
        "minimum_rl_rows_per_split",
        "minimum_rl_entered_rows_per_split",
        "minimum_model_filtered_trades",
        "minimum_model_gated_trades",
        "minimum_nonempty_score_buckets",
        "minimum_rl_trades",
    )
    for field in required_positive_integers:
        value = payload.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"invalid registered survivor support policy: {field}")
    for field in (
        "minimum_oos_take_rate",
        "minimum_rl_take_rate_per_split",
        "maximum_model_gain_concentration",
        "maximum_model_selection_concentration",
        "maximum_rl_concentration",
    ):
        value = payload.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or not 0.0 < float(value) <= 1.0
        ):
            raise ValueError(f"invalid registered survivor support policy: {field}")
    for field in (
        "minimum_oos_taken_return_sum_exclusive",
        "minimum_rl_validation_return_sum_exclusive",
        "minimum_rl_test_return_sum_exclusive",
    ):
        value = payload.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or float(value) < 0.0
        ):
            raise ValueError(f"invalid registered survivor support policy: {field}")
    if (
        not str(payload.get("policy_version", "")).strip()
        or payload.get("return_basis") != "realized_return_after_cost"
        or payload.get("rl_return_basis") != "net_after_cost_strategy_return"
        or payload.get("thresholds_changed_after_results") is not False
    ):
        raise ValueError("registered survivor support policy provenance is invalid")
    effective_at = pd.to_datetime(payload.get("effective_at_utc"), utc=True, errors="coerce")
    if pd.isna(effective_at):
        raise ValueError("invalid registered survivor support policy: effective_at_utc")
    for field in (
        "promotion_authority",
        "testnet_candidate_authority",
        "testnet_order_authority",
        "live_trading_authorized",
    ):
        if _truthy(payload.get(field)):
            raise ValueError(f"registered survivor support policy unexpectedly grants {field}")
    return {
        **payload,
        "support_policy_id": "survivorsupport_" + _payload_hash(payload),
    }


def _as_utc(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(UTC)
    return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "pass", "ready"}


def _payload_hash(payload: dict[str, Any]) -> str:
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest() if path.is_file() else ""


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    promote_staged_file(temporary, path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    promote_staged_file(temporary, path)


def _write_immutable_json(payload: dict[str, Any], path: Path) -> None:
    existing = _read_json(path)
    if existing and existing != payload:
        raise ValueError("registered learning immutable receipt mismatch")
    _atomic_json(payload, path)
