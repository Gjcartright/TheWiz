"""Conditional Testnet and live-canary gates. This module never submits orders."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd
from scipy.stats import t as student_t

from quant_platform.active_pipeline import CommandResult
from quant_platform.orchestration.corrective_daily_scheduler import (
    build_daily_cadence_acceptance,
)
from quant_platform.orchestration.corrective_data_evidence import (
    PAIR_COST_BUNDLE_POINTER_NAME,
    PAIR_COST_BUNDLE_POINTER_SCHEMA_VERSION,
    validate_pair_cost_bundle_artifacts,
)
from quant_platform.orchestration.corrective_registered_rerun import (
    resolve_registered_source_family,
)
from quant_platform.orchestration.corrective_testnet_cohort_readiness import (
    build_testnet_prospective_cohort_readiness,
)
from quant_platform.orchestration.current_wizard_hyperliquid_testnet_protocol import (
    REQUIRED_SCENARIOS as REQUIRED_TESTNET_PROTOCOL_SCENARIOS,
)
from quant_platform.orchestration.current_wizard_hyperliquid_testnet_protocol import (
    validate_current_wizard_hyperliquid_testnet_protocol,
)

ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "thewiz.corrective_release_gates.v1"
CANDIDATE_SCHEMA_VERSION = "thewiz.testnet_candidate_receipt.v5"
CANDIDATE_IDENTITY_SCHEMA_VERSION = "thewiz.testnet_candidate_identity.v4"
SAMPLE_POLICY_SCHEMA_VERSION = "thewiz.testnet_sample_sufficiency.v5"
SAMPLE_MEAN_LCB_METHOD = "student_t_two_sided_95_lower_bound"
QUEUE_ROW_SCHEMA_VERSION = "thewiz.testnet_candidate_queue.v3"
QUEUE_RECEIPT_SCHEMA_VERSION = "thewiz.testnet_candidate_queue_receipt.v5"
QUEUE_POINTER_SCHEMA_VERSION = "thewiz.testnet_candidate_queue_pointer.v5"
STAGE6_RELEASE_RECEIPT_SCHEMA_VERSION = "thewiz.stage6_release_receipt.v1"
STAGE6_RELEASE_POINTER_SCHEMA_VERSION = "thewiz.stage6_release_pointer.v1"
QUEUE_COLUMNS = (
    "schema_version",
    "candidate_experiment_id",
    "registered_semantic_hypothesis_id",
    "registered_stage4_contract_id",
    "pair_group_key",
    "pair",
    "asset_x",
    "asset_y",
    "timeframe",
    "exact_mode",
    "orientation",
    "cost_model_id",
    "strict_cost_model_current",
    "hyperliquid_markets_current",
    "immutable_stage4_candidate_identity_ready",
    "validated_current_model_candidate_samples",
    "validated_current_model_pair_samples",
    "local_blockers",
    "queue_status",
    "selection_rank",
    "selected_active",
    "order_submission_performed",
    "testnet_order_authority",
    "live_trading_authorized",
)


def build_testnet_candidate_receipt(
    *, root: Path = ROOT, now: datetime | None = None
) -> dict[str, Any]:
    active = root / "reports" / "active"
    as_of = _as_utc(now)
    queue = build_testnet_candidate_queue(root=root, now=as_of)
    selected = queue["selected"]
    blockers = list(queue["blockers"])
    if not selected:
        blockers.append("testnet_candidate_queue_has_no_selected_candidate")
    receipt = {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "generated_at_utc": as_of.isoformat(),
        "candidate_status": "READY_FOR_NO_ORDER_PREFLIGHT" if not blockers else "BLOCKED",
        "candidate_experiment_id": str(selected.get("candidate_experiment_id", "")),
        "registered_semantic_hypothesis_id": str(
            selected.get("registered_semantic_hypothesis_id", "")
        ),
        "registered_stage4_contract_id": str(queue["registered_stage4_contract_id"]),
        "registered_stage4_conclusion_sha256": str(queue["registered_stage4_conclusion_sha256"]),
        "immutable_stage4_candidate_identity_ready": bool(
            selected.get("immutable_stage4_candidate_identity_ready", False)
        ),
        "candidate_leverage": 1.0 if selected else None,
        "pair_group_key": str(selected.get("pair_group_key", "")),
        "pair": str(selected.get("pair", "")),
        "asset_x": str(selected.get("asset_x", "")),
        "asset_y": str(selected.get("asset_y", "")),
        "timeframe": str(selected.get("timeframe", "")),
        "exact_mode": str(selected.get("exact_mode", "")),
        "orientation": str(selected.get("orientation", "")),
        "cost_model_id": str(selected.get("cost_model_id", "")),
        "model_training_dataset_id": str(queue["model_training_dataset_id"]),
        "model_artifact_sha256": str(queue["model_artifact_sha256"]),
        "registered_learning_id": str(queue["registered_learning_id"]),
        "registered_learning_receipt_path": str(queue["registered_learning_receipt_path"]),
        "registered_learning_receipt_sha256": str(queue["registered_learning_receipt_sha256"]),
        "registered_stage5_protocol_id": str(queue["registered_stage5_protocol_id"]),
        "registered_stage5_protocol_sha256": str(queue["registered_stage5_protocol_sha256"]),
        "registered_execution_id": str(queue["registered_execution_id"]),
        "survivor_receipt_id": str(queue["survivor_receipt_id"]),
        "candidate_queue_id": str(queue["candidate_queue_id"]),
        "candidate_queue_path": str(queue["candidate_queue_path"]),
        "candidate_queue_sha256": str(queue["candidate_queue_sha256"]),
        "candidate_queue_selection_rank": selected.get("selection_rank"),
        "validated_current_model_candidate_samples": selected.get(
            "validated_current_model_candidate_samples", 0
        ),
        "validated_current_model_pair_samples": selected.get(
            "validated_current_model_pair_samples", 0
        ),
        "ready_candidate_pairs": int(queue["ready_candidate_pairs"]),
        "required_candidate_pairs": int(queue["required_candidate_pairs"]),
        "cadence_pass": bool(queue["cadence_pass"]),
        "model_oos_authority_ready": bool(queue["model_ready"]),
        "strict_cost_model_current": bool(selected.get("strict_cost_model_current", False)),
        "hyperliquid_markets_current": bool(selected.get("hyperliquid_markets_current", False)),
        "blockers": list(dict.fromkeys(blockers)),
        "order_submission_performed": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
        "source_artifact_hashes": dict(queue["source_artifact_hashes"]),
    }
    queue_path = _safe_repo_artifact(root, str(queue["candidate_queue_path"]))
    if queue_path is not None and queue_path.is_file():
        receipt["source_artifact_hashes"][_relative(queue_path, root)] = _sha256_file(queue_path)
    receipt["evidence_path"] = ";".join(receipt["source_artifact_hashes"])
    if not blockers:
        receipt = _seal_testnet_candidate_receipt(receipt=receipt, root=root)
    else:
        receipt_id = "blockedtestnetcandidate_" + _payload_hash(receipt)[:20]
        receipt["candidate_receipt_id"] = receipt_id
        receipt["receipt_id"] = receipt_id
        receipt["candidate_identity_path"] = ""
        receipt["candidate_identity_sha256"] = ""
        receipt["receipt_sha256"] = _payload_hash(receipt)
    path = active / "testnet_candidate_receipt.json"
    _atomic_json(receipt, path)
    return {"path": path, "receipt": receipt}


def build_testnet_candidate_queue(
    *, root: Path = ROOT, now: datetime | None = None
) -> dict[str, Any]:
    """Build a deterministic, non-executing queue of all accepted Testnet pairs."""

    active = root / "reports" / "active"
    as_of = _as_utc(now)
    survivor_path = active / "final_1x_survivor_receipt.json"
    cadence, cadence_path, cadence_validation_blocker = _rebuild_validated_daily_cadence(
        root=root, now=as_of
    )
    current_matrix_path = active / "current_wizard_hyperliquid_experiment_matrix.csv"
    (
        cost_path,
        cost_bundle_manifest_path,
        cost_bundle_blockers,
    ) = _validated_pair_cost_bundle(root=root)
    market_path = root / "data" / "processed" / "hyperliquid_market_manifest.csv"
    policy_path = root / "config" / "testnet_sample_sufficiency_policy.json"
    lifecycle_index_path = root / "data" / "testnet" / "lifecycle_index.csv"
    survivor = _read_json(survivor_path)
    costs = _read_csv(cost_path)
    markets = _read_csv(market_path)
    sample_policy = _read_json(policy_path)
    learning, model, stage4, learning_path, learning_blocker = _validated_registered_learning(
        root=root
    )
    final_ids = sorted(
        {
            str(value).strip()
            for value in survivor.get("final_experiment_ids", [])
            if str(value).strip()
        }
    )
    survivor_ready = bool(
        survivor.get("receipt_status") == "PASS"
        and survivor.get("testnet_candidate_authority") is True
        and survivor.get("thresholds_changed_after_results") is False
        and len(final_ids) >= 1
        and int(survivor.get("final_one_x_survivors", 0) or 0) == len(final_ids)
    )
    stage4_lineage_claimed = bool(
        survivor_ready
        or final_ids
        or stage4
        or survivor.get("receipt_status") == "PASS"
        or survivor.get("testnet_candidate_authority") is True
    )
    matrix = pd.DataFrame()
    registered_identities: dict[str, dict[str, Any]] = {}
    source_family_path: Path | None = None
    source_family_receipt_path: Path | None = None
    stage4_contract_path: Path | None = None
    stage4_identity_blocker = ""
    if stage4_lineage_claimed:
        try:
            (
                matrix,
                registered_identities,
                source_family_path,
                source_family_receipt_path,
                stage4_contract_path,
            ) = _validated_stage4_survivor_source_family(
                root=root,
                stage4=stage4,
                survivor=survivor,
                current_matrix_path=current_matrix_path,
            )
        except (OSError, TypeError, ValueError) as exc:
            stage4_identity_blocker = (
                f"stage4_survivor_identity_lineage_invalid:{type(exc).__name__}:{exc}"
            )
    cadence_pass = bool(not cadence_validation_blocker and _daily_cadence_pass(cadence))
    model_take_rate = _finite(model.get("median_take_rate"))
    minimum_take_rate = _finite(model.get("minimum_take_rate"), default=0.10)
    model_ready = bool(
        learning.get("status") == "PASS_RESEARCH_LEARNING_GATES"
        and learning.get("stage5_research_gate_pass") is True
        and learning.get("training_accepted_oos") is True
        and learning.get("model_accepted_oos") is True
        and learning.get("rl_accepted_oos") is True
        and learning.get("agent_governance_pass") is True
        and model.get("out_of_sample_incremental_edge_accepted") is True
        and model.get("rl_out_of_sample_accepted") is True
        and model.get("exact_mode_trade_provenance_ready") is True
        and model.get("strict_cost_training_evidence_ready") is True
        and model.get("training_dataset_lineage_ready") is True
        and model.get("model_active_dataset_lineage_matches") is True
        and model.get("model_artifact_hash_matches") is True
        and math.isfinite(model_take_rate)
        and math.isfinite(minimum_take_rate)
        and model_take_rate >= minimum_take_rate
    )
    conclusion_path = _safe_repo_artifact(root, str(stage4.get("conclusion_path", "")))
    conclusion = _read_json(conclusion_path) if conclusion_path else {}
    survivor_hash_bound = bool(
        stage4_lineage_claimed
        and survivor_path.is_file()
        and conclusion
        and conclusion_path is not None
        and str(stage4.get("conclusion_sha256", "")) == _sha256_file(conclusion_path)
        and str(conclusion.get("final_survivor_receipt_sha256", "")) == _sha256_file(survivor_path)
    )
    policy_id = _testnet_sample_policy_id(sample_policy)
    required_pairs = int(sample_policy.get("minimum_independent_pairs", 3) or 3)
    model_training_dataset_id = str(learning.get("training_dataset_id", "")).strip()
    model_artifact_sha256 = str(model.get("model_artifact_sha256", "")).strip()
    registered_learning_id = str(learning.get("learning_id", "")).strip()
    registered_learning_receipt_path = (
        _relative(learning_path, root)
        if learning_path is not None and learning_path.is_file()
        else ""
    )
    registered_learning_receipt_sha256 = (
        _sha256_file(learning_path) if learning_path is not None and learning_path.is_file() else ""
    )
    registered_stage5_protocol_id = str(learning.get("registered_stage5_protocol_id", "")).strip()
    registered_stage5_protocol_sha256 = str(
        learning.get("registered_stage5_protocol_sha256", "")
    ).strip()
    registered_execution_id = str(learning.get("registered_execution_id", "")).strip()
    registered_learning_lineage_ready = bool(
        registered_learning_id
        and registered_learning_receipt_path
        and len(registered_learning_receipt_sha256) == 64
        and registered_stage5_protocol_id
        and len(registered_stage5_protocol_sha256) == 64
        and registered_execution_id
    )
    model_ready = bool(model_ready and registered_learning_lineage_ready)
    lifecycles = _validated_lifecycle_index(root=root, index_path=lifecycle_index_path)
    lifecycle_stage5_lineage = {
        "model_training_dataset_id": model_training_dataset_id,
        "model_artifact_sha256": model_artifact_sha256,
        "registered_learning_id": registered_learning_id,
        "registered_learning_receipt_path": registered_learning_receipt_path,
        "registered_learning_receipt_sha256": registered_learning_receipt_sha256,
        "registered_stage5_protocol_id": registered_stage5_protocol_id,
        "registered_stage5_protocol_sha256": registered_stage5_protocol_sha256,
        "registered_execution_id": registered_execution_id,
    }
    lifecycles = _filter_testnet_lifecycles_by_stage5_lineage(
        lifecycles,
        lifecycle_stage5_lineage,
    )
    global_blockers: list[str] = []
    if not survivor_ready:
        global_blockers.append("valid_multi_pair_final_one_x_survivor_receipt_missing")
    if not cadence_pass:
        global_blockers.append("seven_day_research_cadence_not_proven")
    if cadence_validation_blocker:
        global_blockers.append(cadence_validation_blocker)
    if not model_ready:
        global_blockers.append("accepted_ml_rl_oos_authority_missing")
    if learning and not registered_learning_lineage_ready:
        global_blockers.append("registered_stage5_lineage_missing_or_invalid")
    learning_status = _read_json(active / "registered_learning_research_status.json").get("status")
    if learning_blocker and learning_status in {
        "PASS_RESEARCH_LEARNING_GATES",
        "ALREADY_COMPLETE",
    }:
        global_blockers.append(learning_blocker)
    if stage4_identity_blocker:
        global_blockers.append(stage4_identity_blocker)
    if stage4_lineage_claimed and not survivor_hash_bound:
        global_blockers.append("stage4_stage5_survivor_lineage_mismatch")
    if not policy_id:
        global_blockers.append("testnet_sample_policy_invalid_for_candidate_queue")
    global_blockers.extend(cost_bundle_blockers)
    rows: list[dict[str, Any]] = []
    for experiment_id in final_ids:
        registered_identity = registered_identities.get(experiment_id, {})
        experiment_rows = matrix.loc[
            matrix.get("experiment_id", pd.Series(dtype=str)).astype(str).eq(experiment_id)
        ].copy()
        experiment = experiment_rows.iloc[0].to_dict() if len(experiment_rows) == 1 else {}
        pair_group_key = str(experiment.get("pair_group_key", "")).strip()
        pair = str(experiment.get("pair", "")).strip()
        assets = (
            str(experiment.get("asset_a", "")).strip().upper(),
            str(experiment.get("asset_b", "")).strip().upper(),
        )
        cost_rows = (
            costs.loc[
                costs.get("pair_group_key", pd.Series(dtype=str)).astype(str).eq(pair_group_key)
                & costs.get(
                    "strict_observed_cost_ready",
                    pd.Series(False, index=costs.index),
                ).map(_truthy)
            ].copy()
            if pair_group_key and not costs.empty
            else pd.DataFrame()
        )
        cost = (
            cost_rows.sort_values("model_as_of_utc").iloc[-1].to_dict()
            if not cost_rows.empty
            else {}
        )
        cost_time = pd.to_datetime(cost.get("model_as_of_utc"), utc=True, errors="coerce")
        cost_current = bool(
            cost
            and pd.notna(cost_time)
            and cost_time <= pd.Timestamp(as_of)
            and pd.Timestamp(as_of) - cost_time <= pd.Timedelta(hours=2)
        )
        market_rows = (
            markets.loc[
                markets.get("asset", pd.Series(dtype=str)).astype(str).str.upper().isin(assets)
            ].copy()
            if all(assets) and not markets.empty
            else pd.DataFrame()
        )
        market_times = pd.to_datetime(
            market_rows.get("source_timestamp", pd.Series(dtype=str)),
            utc=True,
            errors="coerce",
        )
        markets_ready = bool(
            len(market_rows) == 2
            and market_rows.get("tradable_perp", pd.Series(False, index=market_rows.index))
            .map(_truthy)
            .all()
            and market_times.notna().all()
            and (market_times <= pd.Timestamp(as_of)).all()
            and pd.Timestamp(as_of) - market_times.max() <= pd.Timedelta(hours=24)
        )
        local_blockers: list[str] = []
        if len(experiment_rows) != 1:
            local_blockers.append("accepted_experiment_identity_missing_or_ambiguous")
        immutable_identity_ready = bool(
            registered_identity
            and len(experiment_rows) == 1
            and all(
                str(experiment.get(field, "")).strip()
                == str(registered_identity.get(field, "")).strip()
                for field in (
                    "experiment_id",
                    "pair_group_key",
                    "pair",
                    "exact_mode",
                    "orientation",
                )
            )
        )
        if not immutable_identity_ready:
            local_blockers.append("candidate_identity_not_bound_to_immutable_stage4_source_family")
        if not cost_current:
            local_blockers.append("candidate_strict_pair_cost_model_missing_or_stale")
        if not markets_ready:
            local_blockers.append("candidate_hyperliquid_markets_missing_or_stale")
        pair_samples = int(lifecycles.get("pair", pd.Series(dtype=str)).astype(str).eq(pair).sum())
        candidate_samples = int(
            lifecycles.get("candidate_experiment_id", pd.Series(dtype=str))
            .astype(str)
            .eq(experiment_id)
            .sum()
        )
        rows.append(
            {
                "schema_version": QUEUE_ROW_SCHEMA_VERSION,
                "candidate_experiment_id": experiment_id,
                "registered_semantic_hypothesis_id": str(
                    registered_identity.get("semantic_hypothesis_id", "")
                ),
                "registered_stage4_contract_id": str(stage4.get("contract_id", "")),
                "pair_group_key": pair_group_key,
                "pair": pair,
                "asset_x": assets[0],
                "asset_y": assets[1],
                "timeframe": str(experiment.get("timeframe", "")).strip(),
                "exact_mode": str(experiment.get("exact_mode", "")).strip(),
                "orientation": str(experiment.get("orientation", "")).strip(),
                "cost_model_id": str(cost.get("cost_model_id", "")).strip(),
                "strict_cost_model_current": cost_current,
                "hyperliquid_markets_current": markets_ready,
                "immutable_stage4_candidate_identity_ready": (immutable_identity_ready),
                "validated_current_model_candidate_samples": candidate_samples,
                "validated_current_model_pair_samples": pair_samples,
                "local_blockers": ";".join(local_blockers),
                "queue_status": (
                    "READY" if not global_blockers and not local_blockers else "BLOCKED"
                ),
                "selection_rank": None,
                "selected_active": False,
                "order_submission_performed": False,
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            }
        )
    ready = [row for row in rows if row["queue_status"] == "READY"]
    ready.sort(
        key=lambda row: (
            int(row["validated_current_model_candidate_samples"]),
            int(row["validated_current_model_pair_samples"]),
            str(row["candidate_experiment_id"]),
        )
    )
    for rank, row in enumerate(ready, start=1):
        row["selection_rank"] = rank
    ready_pair_count = len({str(row["pair"]) for row in ready if str(row["pair"])})
    if ready_pair_count < required_pairs:
        global_blockers.append("testnet_candidate_queue_independent_pair_coverage_below_policy")
    selected = ready[0] if ready and ready_pair_count >= required_pairs else {}
    if selected:
        selected["selected_active"] = True
    rows.sort(key=lambda row: str(row["candidate_experiment_id"]))
    source_hashes = _existing_artifact_hashes(
        root=root,
        paths=(
            survivor_path,
            cadence_path,
            source_family_path,
            source_family_receipt_path,
            stage4_contract_path,
            cost_path,
            cost_bundle_manifest_path,
            market_path,
            policy_path,
            lifecycle_index_path,
            learning_path,
            conclusion_path,
        ),
    )
    queue_core = {
        "schema_version": QUEUE_RECEIPT_SCHEMA_VERSION,
        "registered_stage4_contract_id": str(stage4.get("contract_id", "")),
        "registered_stage4_conclusion_sha256": str(stage4.get("conclusion_sha256", "")),
        "immutable_stage4_candidate_identity_ready": bool(
            stage4_lineage_claimed
            and final_ids
            and not stage4_identity_blocker
            and len(registered_identities) == len(final_ids)
        ),
        "survivor_receipt_id": str(survivor.get("acceptance_policy_id", "")),
        "model_training_dataset_id": model_training_dataset_id,
        "model_artifact_sha256": model_artifact_sha256,
        "registered_learning_id": registered_learning_id,
        "registered_learning_receipt_path": registered_learning_receipt_path,
        "registered_learning_receipt_sha256": registered_learning_receipt_sha256,
        "registered_stage5_protocol_id": registered_stage5_protocol_id,
        "registered_stage5_protocol_sha256": registered_stage5_protocol_sha256,
        "registered_execution_id": registered_execution_id,
        "cadence_evidence_validated": not cadence_validation_blocker,
        "cadence_acceptance_status": "PASS" if cadence_pass else "BLOCKED",
        "cadence_validation_blocker": cadence_validation_blocker,
        "testnet_sample_policy_id": policy_id,
        "required_candidate_pairs": required_pairs,
        "ready_candidate_pairs": ready_pair_count,
        "rows": rows,
        "global_blockers": list(dict.fromkeys(global_blockers)),
        "candidate_selection_ready": bool(selected and not global_blockers),
        "source_artifact_hashes": source_hashes,
        "order_submission_performed": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    queue_id = "testnetcandidatequeue_" + _payload_hash(queue_core)[:20]
    queue_receipt = {**queue_core, "candidate_queue_id": queue_id}
    queue_receipt["receipt_sha256"] = _payload_hash(queue_receipt)
    queue_path = root / "data" / "testnet" / "candidate_queues" / f"{queue_id}.json"
    _write_immutable_json(queue_receipt, queue_path)
    active_queue_path = active / "testnet_candidate_queue.csv"
    _atomic_csv(pd.DataFrame(rows, columns=QUEUE_COLUMNS), active_queue_path)
    active_queue_receipt = active / "testnet_candidate_queue_receipt.json"
    queue_pointer = {
        "schema_version": QUEUE_POINTER_SCHEMA_VERSION,
        "generated_at_utc": as_of.isoformat(),
        "registered_stage4_contract_id": str(stage4.get("contract_id", "")),
        "registered_stage4_conclusion_sha256": str(stage4.get("conclusion_sha256", "")),
        "immutable_stage4_candidate_identity_ready": bool(
            queue_core["immutable_stage4_candidate_identity_ready"]
        ),
        "cadence_evidence_validated": not cadence_validation_blocker,
        "cadence_acceptance_status": "PASS" if cadence_pass else "BLOCKED",
        "cadence_validation_blocker": cadence_validation_blocker,
        "candidate_queue_id": queue_id,
        "candidate_queue_path": _relative(queue_path, root),
        "candidate_queue_sha256": _sha256_file(queue_path),
        "immutable_queue_receipt_sha256": str(queue_receipt["receipt_sha256"]),
        "registered_learning_id": registered_learning_id,
        "registered_learning_receipt_path": registered_learning_receipt_path,
        "registered_learning_receipt_sha256": registered_learning_receipt_sha256,
        "registered_stage5_protocol_id": registered_stage5_protocol_id,
        "registered_stage5_protocol_sha256": registered_stage5_protocol_sha256,
        "registered_execution_id": registered_execution_id,
        "active_queue_path": _relative(active_queue_path, root),
        "active_queue_sha256": _sha256_file(active_queue_path),
        "candidate_selection_ready": bool(selected and not global_blockers),
        "ready_candidate_pairs": ready_pair_count,
        "required_candidate_pairs": required_pairs,
        "global_blockers": list(dict.fromkeys(global_blockers)),
        "order_submission_performed": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    queue_pointer["receipt_sha256"] = _payload_hash(queue_pointer)
    _atomic_json(queue_pointer, active_queue_receipt)
    return {
        "selected": selected,
        "blockers": list(dict.fromkeys(global_blockers)),
        "registered_stage4_contract_id": str(stage4.get("contract_id", "")),
        "registered_stage4_conclusion_sha256": str(stage4.get("conclusion_sha256", "")),
        "candidate_queue_id": queue_id,
        "candidate_queue_path": _relative(queue_path, root),
        "candidate_queue_sha256": _sha256_file(queue_path),
        "ready_candidate_pairs": ready_pair_count,
        "required_candidate_pairs": required_pairs,
        "model_training_dataset_id": model_training_dataset_id,
        "model_artifact_sha256": model_artifact_sha256,
        "cadence_evidence_validated": not cadence_validation_blocker,
        "cadence_validation_blocker": cadence_validation_blocker,
        "registered_learning_id": registered_learning_id,
        "registered_learning_receipt_path": registered_learning_receipt_path,
        "registered_learning_receipt_sha256": registered_learning_receipt_sha256,
        "registered_stage5_protocol_id": registered_stage5_protocol_id,
        "registered_stage5_protocol_sha256": registered_stage5_protocol_sha256,
        "registered_execution_id": registered_execution_id,
        "survivor_receipt_id": str(survivor.get("acceptance_policy_id", "")),
        "cadence_pass": cadence_pass,
        "model_ready": model_ready,
        "source_artifact_hashes": source_hashes,
        "queue": active_queue_path,
        "queue_receipt": active_queue_receipt,
    }


def build_no_order_preflight(*, root: Path = ROOT, now: datetime | None = None) -> dict[str, Any]:
    active = root / "reports" / "active"
    as_of = _as_utc(now)
    candidate = build_testnet_candidate_receipt(root=root, now=as_of)["receipt"]
    existing = _read_csv(active / "hyperliquid_testnet_preflight.csv")
    margin = _read_csv(active / "hyperliquid_testnet_margin_snapshot.csv")
    approval = _read_json(active / "hyperliquid_testnet_smoke_approval.json")
    protocol_validation_blocker = ""
    try:
        protocol_result = validate_current_wizard_hyperliquid_testnet_protocol(
            root=root,
            now=as_of,
        )
        protocol = dict(protocol_result.summary)
    except Exception as exc:  # noqa: BLE001 - preflight must fail closed as data
        protocol = {}
        protocol_validation_blocker = (
            f"testnet_protocol_revalidation_failed:{type(exc).__name__}:{exc}"
        )
    preflight_row = existing.iloc[-1].to_dict() if not existing.empty else {}
    preflight_time = pd.to_datetime(preflight_row.get("checked_at_utc"), utc=True, errors="coerce")
    wallet_agent_ready = bool(
        preflight_row
        and preflight_row.get("network") == "testnet"
        and _truthy(preflight_row.get("ready_for_no_order_preflight"))
        and not _truthy(preflight_row.get("submit_orders_enabled"))
        and _truthy(preflight_row.get("agent_key_matches_address"))
        and _truthy(preflight_row.get("agent_authorized_for_master"))
        and pd.notna(preflight_time)
        and preflight_time <= pd.Timestamp(as_of)
        and pd.Timestamp(as_of) - preflight_time <= pd.Timedelta(hours=24)
    )
    margin_row = margin.iloc[-1].to_dict() if not margin.empty else {}
    margin_time = pd.to_datetime(margin_row.get("checked_at_utc"), utc=True, errors="coerce")
    margin_ready = bool(
        margin_row
        and margin_row.get("network") == "testnet"
        and str(margin_row.get("status", "")).upper() in {"PASS", "READY"}
        and str(margin_row.get("point_in_time_status", "")).lower() == "confirmed"
        and (pd.isna(margin_row.get("blockers")) or not str(margin_row.get("blockers", "")).strip())
        and _finite(margin_row.get("withdrawable_usd"), default=0.0) >= 25.0
        and pd.notna(margin_time)
        and margin_time <= pd.Timestamp(as_of)
        and pd.Timestamp(as_of) - margin_time <= pd.Timedelta(hours=2)
    )
    approval_binding_ready, sizing_ready = _approval_candidate_sizing_ready(approval, candidate)
    scenario_results = (
        protocol.get("scenario_results")
        if isinstance(protocol.get("scenario_results"), list)
        else []
    )
    required_recovery_scenarios = {
        "partial_orphan_recovery",
        "unconfirmed_response_reconciliation",
        "restart_open_pair_reconciliation",
    }
    passed_recovery_scenarios = {
        str(row.get("scenario_id", ""))
        for row in scenario_results
        if isinstance(row, dict)
        and row.get("scenario_pass") is True
        and row.get("simulation_only") is True
        and row.get("order_submission_performed") is False
    }
    protocol_identity_material = {
        "schema_version": protocol.get("schema_version"),
        "source_hashes": protocol.get("source_hashes"),
        "scenario_results": scenario_results,
        "candidate_coverage": protocol.get("candidate_coverage"),
    }
    expected_protocol_id = (
        "cwtestnetprotocol_"
        + sha256(_canonical_json(protocol_identity_material).encode("utf-8")).hexdigest()[:20]
    )
    protocol_current_binding_ready = bool(
        not protocol_validation_blocker
        and protocol.get("schema_version") == "current_wizard_hyperliquid_testnet_protocol.v1"
        and protocol.get("protocol_id") == expected_protocol_id
        and isinstance(protocol.get("source_hashes"), dict)
        and len(protocol.get("source_hashes", {})) >= 4
        and protocol.get("required_scenarios_complete") is True
        and REQUIRED_TESTNET_PROTOCOL_SCENARIOS.issubset(
            {str(row.get("scenario_id", "")) for row in scenario_results if isinstance(row, dict)}
        )
        and protocol.get("execution_authority") is False
        and protocol.get("order_submission_performed") is False
        and protocol.get("live_trading_authorized") is False
    )
    recovery_ready = bool(
        protocol_current_binding_ready
        and protocol.get("protocol_status") == "PASS"
        and protocol.get("simulation_is_testnet_proof") is False
        and protocol.get("order_submission_performed") is False
        and required_recovery_scenarios.issubset(passed_recovery_scenarios)
    )
    checks = [
        (
            "testnet_protocol_current_binding",
            protocol_current_binding_ready,
            protocol_validation_blocker or "testnet_recovery_protocol_current_binding_not_proven",
        ),
        (
            "single_1x_survivor",
            candidate.get("candidate_status") == "READY_FOR_NO_ORDER_PREFLIGHT",
            "valid_single_final_one_x_survivor_receipt_missing",
        ),
        (
            "market_inventory",
            candidate.get("hyperliquid_markets_current") is True,
            "current_testnet_market_inventory_not_proven_for_candidate",
        ),
        ("wallet_agent", wallet_agent_ready, "testnet_wallet_or_agent_not_verified_for_candidate"),
        ("margin_collateral", margin_ready, "testnet_margin_not_verified_for_candidate"),
        (
            "funding_current",
            candidate.get("strict_cost_model_current") is True,
            "candidate_specific_current_funding_not_verified",
        ),
        (
            "candidate_approval_binding",
            approval_binding_ready,
            "candidate_specific_approval_binding_not_frozen",
        ),
        ("paired_sizing", sizing_ready, "candidate_specific_paired_size_not_frozen"),
        (
            "partial_fill_contingency",
            recovery_ready,
            "testnet_partial_fill_recovery_protocol_not_proven",
        ),
        (
            "submit_orders_disabled_during_preflight",
            not _truthy(preflight_row.get("submit_orders_enabled")),
            "testnet_submission_enabled_during_no_order_preflight",
        ),
    ]
    rows = []
    for check, passed, blocker in checks:
        rows.append(
            {
                "check": check,
                "status": "PASS" if passed else "BLOCKED",
                "blocker": "" if passed else blocker,
                "no_order_preflight": True,
                "candidate_receipt_id": candidate.get("candidate_receipt_id", ""),
                "order_submission_performed": False,
                "testnet_order_authority": False,
                "live_trading_authorized": False,
                "evidence_path": (
                    "reports/active/testnet_candidate_receipt.json;"
                    "reports/active/hyperliquid_testnet_preflight.csv;"
                    "reports/active/hyperliquid_testnet_margin_snapshot.csv;"
                    "reports/active/hyperliquid_testnet_smoke_approval.json;"
                    "reports/active/current_wizard_hyperliquid_testnet_protocol_manifest.json"
                ),
            }
        )
    frame = pd.DataFrame(rows)
    path = active / "testnet_no_order_preflight.csv"
    _atomic_csv(frame, path)
    return {
        "path": path,
        "frame": frame,
        "status": "PASS" if frame["status"].eq("PASS").all() else "BLOCKED",
    }


def build_testnet_lifecycle_receipt(
    *, root: Path = ROOT, now: datetime | None = None
) -> dict[str, Any]:
    preflight = build_no_order_preflight(root=root, now=now)
    active = root / "reports" / "active"
    candidate = _read_json(active / "testnet_candidate_receipt.json")
    from quant_platform.orchestration.hyperliquid_learning_and_risk import (
        TESTNET_CANDIDATE_BINDING_FIELDS,
        build_testnet_lifecycle_gate,
    )

    mature_gate = build_testnet_lifecycle_gate(root=root)
    gate = _read_csv(Path(mature_gate["gate"]))
    smoke_receipt_path = active / "hyperliquid_testnet_smoke_receipt.json"
    smoke_receipt = _read_json(smoke_receipt_path)
    final_state = (
        smoke_receipt.get("final_state")
        if isinstance(smoke_receipt.get("final_state"), dict)
        else {}
    )
    events = smoke_receipt.get("events") if isinstance(smoke_receipt.get("events"), list) else []
    event_types = {str(event.get("event_type", "")) for event in events if isinstance(event, dict)}
    candidate_bound = bool(
        candidate.get("candidate_status") == "READY_FOR_NO_ORDER_PREFLIGHT"
        and all(str(candidate.get(field, "")).strip() for field in TESTNET_CANDIDATE_BINDING_FIELDS)
        and all(
            smoke_receipt.get(field) == candidate.get(field)
            for field in TESTNET_CANDIDATE_BINDING_FIELDS
        )
    )
    reconciled_flat = bool(
        final_state.get("reconciled") is True
        and _finite(final_state.get("position_x"), default=math.nan) == 0.0
        and _finite(final_state.get("position_y"), default=math.nan) == 0.0
        and _finite(final_state.get("open_order_count"), default=math.nan) == 0.0
    )
    complete = bool(
        preflight["status"] == "PASS"
        and mature_gate.get("status") == "PASS"
        and candidate_bound
        and reconciled_flat
        and {"two_leg_entry", "two_leg_exit", "reconciled", "idempotency"}.issubset(event_types)
    )
    blockers: list[str] = []
    if preflight["status"] != "PASS":
        blockers.append("no_order_preflight_blocked")
    if mature_gate.get("status") != "PASS":
        blockers.extend(
            str(value)
            for value in gate.loc[
                gate.get("status", pd.Series(dtype=str)).ne("PASS"), "blocker"
            ].dropna()
            if str(value).strip()
        )
    if not candidate_bound:
        blockers.append("testnet_lifecycle_candidate_identity_not_proven")
    if smoke_receipt and not reconciled_flat:
        blockers.append("testnet_lifecycle_not_reconciled_flat")
    if not smoke_receipt:
        blockers.append("actual_testnet_lifecycle_receipt_missing")
    terminal_events = [
        event
        for event in events
        if isinstance(event, dict) and event.get("event_type") in {"two_leg_entry", "two_leg_exit"}
    ]
    exchange_references = [
        str(reference)
        for event in terminal_events
        for reference in event.get("exchange_reference_ids", [])
        if str(reference).strip()
    ]
    terminal_fills = [
        fill
        for event in terminal_events
        for fill in (
            event.get("fill_evidence") if isinstance(event.get("fill_evidence"), list) else []
        )
        if isinstance(fill, dict)
    ]
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": _as_utc(now).isoformat(),
        "lifecycle_status": "COMPLETE_RECONCILED" if complete else "NOT_EXECUTED",
        "requested_leverage": 1.0,
        "candidate_receipt_id": candidate.get("candidate_receipt_id", ""),
        "candidate_experiment_id": candidate.get("candidate_experiment_id", ""),
        "pair": candidate.get("pair", ""),
        **{field: candidate.get(field, "") for field in TESTNET_CANDIDATE_BINDING_FIELDS},
        "paired_order_intent_created": bool(smoke_receipt),
        "orders_submitted": len(set(exchange_references)) if complete else 0,
        "fills_observed": len(terminal_fills) if complete else 0,
        "positions_opened": 2 if complete else 0,
        "positions_reconciled_flat": reconciled_flat if complete else False,
        "mature_lifecycle_gate_status": mature_gate.get("status", "BLOCKED"),
        "actual_testnet_receipt_hash": smoke_receipt.get("receipt_hash", ""),
        "blockers": list(dict.fromkeys(blockers)),
        "testnet_order_authority": False,
        "live_trading_authorized": False,
        "evidence_path": ";".join(
            (
                _relative(Path(preflight["path"]), root),
                _relative(Path(mature_gate["gate"]), root),
                _relative(smoke_receipt_path, root),
                "reports/active/testnet_candidate_receipt.json",
            )
        ),
    }
    path = root / "reports" / "active" / "testnet_lifecycle_execution_receipt.json"
    _atomic_json(receipt, path)
    return {"path": path, "receipt": receipt}


def archive_validated_testnet_lifecycle(
    *,
    root: Path = ROOT,
    now: datetime | None = None,
    approval_secret: str | None = None,
    config: Any | None = None,
) -> dict[str, Any]:
    """Archive one validated exchange-backed lifecycle without granting authority."""

    active = root / "reports" / "active"
    sample_policy, sample_policy_path, sample_policy_blockers = _seal_testnet_sample_policy(
        root=root
    )
    source_paths = {
        "receipt": active / "hyperliquid_testnet_smoke_receipt.json",
        "approval": active / "hyperliquid_testnet_smoke_approval.json",
        "run_manifest": active / "hyperliquid_run_manifest.json",
        "protocol": active / "current_wizard_hyperliquid_testnet_protocol_manifest.json",
        "candidate": active / "testnet_candidate_receipt.json",
        "sample_policy": sample_policy_path or (active / ".missing_testnet_sample_policy.json"),
    }
    active_candidate = _read_json(source_paths["candidate"])
    candidate_identity_path = _safe_repo_artifact(
        root, str(active_candidate.get("candidate_identity_path", ""))
    )
    source_paths["candidate_identity"] = candidate_identity_path or (
        active / ".missing_testnet_candidate_identity.json"
    )
    candidate_queue_path = _safe_repo_artifact(
        root, str(active_candidate.get("candidate_queue_path", ""))
    )
    source_paths["candidate_queue"] = candidate_queue_path or (
        active / ".missing_testnet_candidate_queue.json"
    )
    candidate_queue = _read_json(candidate_queue_path) if candidate_queue_path else {}
    candidate_source_hashes = candidate_queue.get("source_artifact_hashes")
    if isinstance(candidate_source_hashes, dict):
        for index, relative in enumerate(sorted(candidate_source_hashes)):
            source_path = _safe_repo_artifact(root, str(relative))
            source_paths[f"candidate_source_{index:04d}"] = source_path or (
                active / f".missing_testnet_candidate_source_{index:04d}"
            )
    index_path = root / "data" / "testnet" / "lifecycle_index.csv"
    from quant_platform.orchestration.hyperliquid_learning_and_risk import (
        _testnet_receipt_payload_hash,
        build_testnet_lifecycle_gate,
    )

    mature_gate = build_testnet_lifecycle_gate(
        root=root,
        approval_secret=approval_secret,
        config=config,
    )
    receipt = _read_json(source_paths["receipt"])
    supplied_hash = str(receipt.get("receipt_hash", "")).strip().lower()
    expected_hash = _testnet_receipt_payload_hash(receipt) if receipt else ""
    blockers: list[str] = []
    blockers.extend(sample_policy_blockers)
    if mature_gate.get("status") != "PASS":
        blockers.append("validated_testnet_lifecycle_gate_not_passed")
    if not supplied_hash or supplied_hash != expected_hash:
        blockers.append("validated_testnet_lifecycle_receipt_hash_invalid")
    if not all(path.is_file() for path in source_paths.values()):
        blockers.append("validated_testnet_lifecycle_source_artifacts_missing")
    candidate_valid, candidate_blockers = _validated_testnet_candidate_receipt(
        root=root,
        candidate=active_candidate,
        identity_path=candidate_identity_path,
    )
    if not candidate_valid:
        blockers.extend(candidate_blockers)
    existing_index = _read_csv(index_path)
    existing_policy_ids = set(
        existing_index.get("sample_policy_id", pd.Series(dtype=str)).dropna().astype(str)
    )
    if existing_policy_ids and existing_policy_ids != {
        str(sample_policy.get("sample_policy_id", ""))
    }:
        blockers.append("testnet_sample_policy_changed_after_samples_started")
    if blockers:
        return {
            "status": "BLOCKED",
            "archived": False,
            "already_archived": False,
            "blockers": blockers,
            "index": index_path,
            "order_submission_performed": False,
            "live_trading_authorized": False,
        }

    archive_root = root / "data" / "testnet" / "lifecycle_receipts"
    archive_path = archive_root / supplied_hash
    archive_manifest_path = archive_path / "archive_manifest.json"
    already_archived = archive_path.exists()
    if already_archived:
        archive_valid, archive_blockers = _validate_lifecycle_archive(
            archive_path=archive_path,
            receipt_hash=supplied_hash,
        )
        if not archive_valid:
            return {
                "status": "BLOCKED",
                "archived": False,
                "already_archived": True,
                "blockers": archive_blockers,
                "index": index_path,
                "order_submission_performed": False,
                "live_trading_authorized": False,
            }
    else:
        archive_root.mkdir(parents=True, exist_ok=True)
        temporary = archive_root / f".{supplied_hash}.tmp"
        if temporary.exists():
            return {
                "status": "BLOCKED",
                "archived": False,
                "already_archived": False,
                "blockers": ["validated_testnet_lifecycle_archive_temporary_exists"],
                "index": index_path,
                "order_submission_performed": False,
                "live_trading_authorized": False,
            }
        temporary.mkdir(parents=False)
        archived_artifacts: list[dict[str, str]] = []
        for name, source in source_paths.items():
            destination = temporary / f"{name}{source.suffix}"
            destination.write_bytes(source.read_bytes())
            archived_artifacts.append(
                {
                    "name": name,
                    "source_path": _relative(source, root),
                    "archive_file": destination.name,
                    "sha256": _sha256_file(destination),
                }
            )
        gate_path = Path(mature_gate["gate"])
        gate_destination = temporary / "lifecycle_gate.csv"
        gate_destination.write_bytes(gate_path.read_bytes())
        archived_artifacts.append(
            {
                "name": "lifecycle_gate",
                "source_path": _relative(gate_path, root),
                "archive_file": gate_destination.name,
                "sha256": _sha256_file(gate_destination),
            }
        )
        manifest = {
            "schema_version": "thewiz.testnet_lifecycle_archive.v1",
            "archive_id": f"testnetlifecycle_{supplied_hash[:20]}",
            "archived_at_utc": _as_utc(now).isoformat(),
            "receipt_hash": supplied_hash,
            "actual_testnet": True,
            "exchange_backed": True,
            "mature_lifecycle_gate_status": "PASS",
            "artifacts": archived_artifacts,
            "order_submission_performed_by_archiver": False,
            "live_trading_authorized": False,
        }
        _atomic_json(manifest, temporary / "archive_manifest.json")
        temporary.rename(archive_path)

    archived_receipt = _read_json(archive_path / "receipt.json")
    archived_approval = _read_json(archive_path / "approval.json")
    archive_manifest = _read_json(archive_manifest_path)
    row = _lifecycle_index_row(
        receipt=archived_receipt,
        approval=archived_approval,
        manifest=archive_manifest,
        archive_path=archive_path,
        root=root,
    )
    existing = _read_csv(index_path)
    if not existing.empty and supplied_hash in set(
        existing.get("receipt_hash", pd.Series(dtype=str)).astype(str)
    ):
        matching = existing.loc[
            existing.get("receipt_hash", pd.Series(dtype=str)).astype(str).eq(supplied_hash)
        ]
        if len(matching) != 1 or not _index_row_matches_receipt(matching.iloc[0].to_dict(), row):
            return {
                "status": "BLOCKED",
                "archived": True,
                "already_archived": True,
                "blockers": ["validated_testnet_lifecycle_index_hash_conflict"],
                "index": index_path,
                "archive": archive_path,
                "order_submission_performed": False,
                "live_trading_authorized": False,
            }
    else:
        updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
        updated = updated.sort_values(
            ["started_at_utc", "receipt_hash"], kind="stable"
        ).reset_index(drop=True)
        _atomic_csv(updated, index_path)
    return {
        "status": "PASS",
        "archived": True,
        "already_archived": already_archived,
        "blockers": [],
        "index": index_path,
        "archive": archive_path,
        "receipt_hash": supplied_hash,
        "order_submission_performed": False,
        "live_trading_authorized": False,
    }


def build_testnet_sample_sufficiency(*, root: Path = ROOT) -> dict[str, Any]:
    policy_path = root / "config" / "testnet_sample_sufficiency_policy.json"
    active_policy = _read_json(policy_path)
    policy = active_policy
    index_path = root / "data" / "testnet" / "lifecycle_index.csv"
    testnet = _validated_lifecycle_index(root=root, index_path=index_path)
    candidate_path = root / "reports" / "active" / "testnet_candidate_receipt.json"
    active_candidate = _read_json(candidate_path)
    candidate_valid, _candidate_blockers = _validated_testnet_candidate_receipt(
        root=root,
        candidate=active_candidate,
    )
    candidate_queue_path = _safe_repo_artifact(
        root, str(active_candidate.get("candidate_queue_path", ""))
    )
    candidate_queue = _read_json(candidate_queue_path) if candidate_queue_path else {}
    candidate_queue_rows = (
        candidate_queue.get("rows") if isinstance(candidate_queue.get("rows"), list) else []
    )
    accepted_candidate_ids = {
        str(row.get("candidate_experiment_id", "")).strip()
        for row in candidate_queue_rows
        if isinstance(row, dict)
        and row.get("schema_version") == QUEUE_ROW_SCHEMA_VERSION
        and row.get("queue_status") == "READY"
        and str(row.get("candidate_experiment_id", "")).strip()
    }
    from quant_platform.orchestration.hyperliquid_learning_and_risk import (
        TESTNET_STAGE5_COHORT_BINDING_FIELDS,
    )

    active_stage5_lineage = {
        field: str(active_candidate.get(field, "")).strip()
        for field in TESTNET_STAGE5_COHORT_BINDING_FIELDS
    }
    active_training_dataset_id = active_stage5_lineage["model_training_dataset_id"]
    active_model_sha256 = active_stage5_lineage["model_artifact_sha256"]
    model_cohort_ready = bool(
        candidate_valid and accepted_candidate_ids and all(active_stage5_lineage.values())
    )
    active_policy_id = _testnet_sample_policy_id(active_policy)
    policy_stable = bool(active_policy_id)
    frozen_policy_path: Path | None = None
    if not testnet.empty:
        policy_ids = set(testnet["sample_policy_id"].dropna().astype(str))
        first_archive = _safe_repo_artifact(root, str(testnet.iloc[0].get("archive_path", "")))
        frozen_policy_path = (
            first_archive / "sample_policy.json" if first_archive is not None else None
        )
        frozen_policy = _read_json(frozen_policy_path) if frozen_policy_path else {}
        frozen_valid, _ = _validate_testnet_sample_policy(frozen_policy)
        frozen_policy_id = str(frozen_policy.get("sample_policy_id", ""))
        policy_stable = bool(
            frozen_valid
            and policy_ids == {frozen_policy_id}
            and active_policy_id == frozen_policy_id
        )
        policy = frozen_policy if frozen_valid else {}
    if model_cohort_ready and not testnet.empty:
        cohort = _filter_testnet_lifecycles_by_stage5_lineage(
            testnet,
            active_stage5_lineage,
        )
        cohort = cohort.loc[
            cohort.get("candidate_experiment_id", pd.Series(dtype=str))
            .astype(str)
            .isin(accepted_candidate_ids)
        ].copy()
    else:
        cohort = testnet.iloc[0:0].copy()
    attempts, attempt_report_path = _validated_testnet_execution_attempts(root=root)
    valid_attempts = (
        attempts.loc[
            attempts.get("evidence_valid", pd.Series(False, index=attempts.index)).map(_truthy)
        ].copy()
        if not attempts.empty
        else attempts
    )
    if model_cohort_ready and not valid_attempts.empty:
        attempt_cohort = _filter_testnet_lifecycles_by_stage5_lineage(
            valid_attempts,
            active_stage5_lineage,
        )
        attempt_cohort = attempt_cohort.loc[
            attempt_cohort.get("candidate_experiment_id", pd.Series(dtype=str))
            .astype(str)
            .isin(accepted_candidate_ids)
        ].copy()
    else:
        attempt_cohort = valid_attempts.iloc[0:0].copy()
    excluded_obsolete_model_samples = len(testnet) - len(cohort)
    duplicate_exchange_evidence = (
        int(
            cohort.get("duplicate_exchange_evidence", pd.Series(False, index=cohort.index))
            .map(_truthy)
            .sum()
        )
        if not cohort.empty
        else 0
    )
    duplicate_strategy_signals = (
        int(
            cohort.get("duplicate_strategy_signal", pd.Series(False, index=cohort.index))
            .map(_truthy)
            .sum()
        )
        if not cohort.empty
        else 0
    )
    eligible = (
        cohort.loc[
            ~cohort.get("duplicate_exchange_evidence", pd.Series(False, index=cohort.index)).map(
                _truthy
            )
            & ~cohort.get("duplicate_strategy_signal", pd.Series(False, index=cohort.index)).map(
                _truthy
            )
        ].copy()
        if not cohort.empty
        else cohort
    )
    closed = len(eligible)
    times = pd.to_datetime(
        eligible.get("started_at_utc", pd.Series(dtype=str)),
        utc=True,
        errors="coerce",
    ).dropna()
    days = int(times.dt.date.nunique()) if not times.empty else 0
    pairs = int(eligible.get("pair", pd.Series(dtype=str)).nunique()) if not eligible.empty else 0
    regimes = (
        int(eligible.get("regime", pd.Series(dtype=str)).nunique()) if not eligible.empty else 0
    )
    candidate_counts = (
        eligible["candidate_experiment_id"].astype(str).value_counts()
        if not eligible.empty and "candidate_experiment_id" in eligible
        else pd.Series(dtype=int)
    )
    pair_counts = (
        eligible["pair"].astype(str).value_counts()
        if not eligible.empty and "pair" in eligible
        else pd.Series(dtype=int)
    )
    regime_counts = (
        eligible["regime"].astype(str).value_counts()
        if not eligible.empty and "regime" in eligible
        else pd.Series(dtype=int)
    )
    accepted_candidate_coverage = bool(
        accepted_candidate_ids and set(candidate_counts.index.astype(str)) == accepted_candidate_ids
    )
    minimum_candidate_samples = (
        int(candidate_counts.reindex(sorted(accepted_candidate_ids), fill_value=0).min())
        if accepted_candidate_ids
        else 0
    )
    minimum_pair_samples = int(pair_counts.min()) if not pair_counts.empty else 0
    minimum_regime_samples = int(regime_counts.min()) if not regime_counts.empty else 0
    orphan_legs = (
        int(eligible.get("unresolved_orphan_legs", pd.Series(dtype=float)).fillna(0).sum())
        if not eligible.empty
        else 0
    )
    unreconciled_orders = (
        int(eligible.get("unreconciled_orders", pd.Series(dtype=float)).fillna(0).sum())
        if not eligible.empty
        else 0
    )
    successful_approval_ids = set(
        eligible.get("approval_id", pd.Series(dtype=str)).dropna().astype(str)
    )
    failed_approval_ids = (
        set(
            attempt_cohort.loc[
                attempt_cohort.get("attempt_outcome", pd.Series(dtype=str)).eq("FAILURE"),
                "approval_id",
            ]
            .dropna()
            .astype(str)
        )
        if not attempt_cohort.empty
        else set()
    )
    failed_approval_ids -= successful_approval_ids
    pending_approval_ids = (
        set(
            attempt_cohort.loc[
                attempt_cohort.get("attempt_outcome", pd.Series(dtype=str)).eq("PENDING"),
                "approval_id",
            ]
            .dropna()
            .astype(str)
        )
        if not attempt_cohort.empty
        else set()
    )
    pending_approval_ids -= successful_approval_ids | failed_approval_ids
    terminal_attempt_count = len(successful_approval_ids | failed_approval_ids)
    failure_rate = (
        len(failed_approval_ids) / terminal_attempt_count if terminal_attempt_count else math.nan
    )
    invalid_attempt_evidence = (
        int(
            (
                ~attempts.get("evidence_valid", pd.Series(False, index=attempts.index)).map(_truthy)
            ).sum()
        )
        if not attempts.empty
        else 0
    )
    mean_net = (
        float(eligible.get("net_realized_pnl_after_cost_usd", pd.Series(dtype=float)).mean())
        if not eligible.empty
        else math.nan
    )
    net_results = pd.to_numeric(
        eligible.get("net_realized_pnl_after_cost_usd", pd.Series(dtype=float)),
        errors="coerce",
    ).dropna()
    positive_outcome_share = float(net_results.gt(0.0).mean()) if not net_results.empty else 0.0
    mean_lcb_95 = _mean_lcb_95(net_results)
    pair_sample_share = (
        float(eligible["pair"].astype(str).value_counts(normalize=True).max())
        if not eligible.empty and "pair" in eligible
        else 1.0
    )
    regime_sample_share = (
        float(eligible["regime"].astype(str).value_counts(normalize=True).max())
        if not eligible.empty and "regime" in eligible
        else 1.0
    )
    all_prospective = bool(
        not eligible.empty
        and eligible.get("prospective_evidence", pd.Series(False, index=eligible.index))
        .map(_truthy)
        .all()
    )
    no_risk_override = bool(
        not eligible.empty
        and ~eligible.get("risk_override_applied", pd.Series(True, index=eligible.index))
        .map(_truthy)
        .any()
    )
    rows = [
        (
            "active_model_lineage_bound",
            model_cohort_ready,
            True,
            model_cohort_ready,
        ),
        (
            "frozen_sample_policy",
            policy_stable,
            True,
            policy_stable,
        ),
        (
            "closed_paired_lifecycles",
            closed,
            int(policy.get("minimum_closed_paired_lifecycles", 30)),
            closed >= int(policy.get("minimum_closed_paired_lifecycles", 30)),
        ),
        (
            "observation_days",
            days,
            int(policy.get("minimum_observation_days", 14)),
            days >= int(policy.get("minimum_observation_days", 14)),
        ),
        (
            "independent_pairs",
            pairs,
            int(policy.get("minimum_independent_pairs", 3)),
            pairs >= int(policy.get("minimum_independent_pairs", 3)),
        ),
        (
            "observed_regimes",
            regimes,
            int(policy.get("minimum_observed_regimes", 3)),
            regimes >= int(policy.get("minimum_observed_regimes", 3)),
        ),
        (
            "accepted_candidate_coverage",
            len(candidate_counts),
            len(accepted_candidate_ids),
            accepted_candidate_coverage,
        ),
        (
            "lifecycles_per_accepted_candidate",
            minimum_candidate_samples,
            int(policy.get("minimum_lifecycles_per_candidate", 5)),
            accepted_candidate_coverage
            and minimum_candidate_samples >= int(policy.get("minimum_lifecycles_per_candidate", 5)),
        ),
        (
            "lifecycles_per_pair",
            minimum_pair_samples,
            int(policy.get("minimum_lifecycles_per_pair", 5)),
            pairs >= int(policy.get("minimum_independent_pairs", 3))
            and minimum_pair_samples >= int(policy.get("minimum_lifecycles_per_pair", 5)),
        ),
        (
            "lifecycles_per_regime",
            minimum_regime_samples,
            int(policy.get("minimum_lifecycles_per_regime", 5)),
            regimes >= int(policy.get("minimum_observed_regimes", 3))
            and minimum_regime_samples >= int(policy.get("minimum_lifecycles_per_regime", 5)),
        ),
        (
            "unresolved_orphan_legs",
            orphan_legs,
            int(policy.get("maximum_unresolved_orphan_legs", 0)),
            closed > 0 and orphan_legs <= int(policy.get("maximum_unresolved_orphan_legs", 0)),
        ),
        (
            "unreconciled_orders",
            unreconciled_orders,
            int(policy.get("maximum_unreconciled_orders", 0)),
            closed > 0 and unreconciled_orders <= int(policy.get("maximum_unreconciled_orders", 0)),
        ),
        (
            "execution_attempt_evidence_valid",
            invalid_attempt_evidence,
            0,
            invalid_attempt_evidence == 0,
        ),
        (
            "submitted_attempts_accounted",
            len(pending_approval_ids),
            0,
            len(pending_approval_ids) == 0,
        ),
        (
            "execution_failure_rate",
            failure_rate if math.isfinite(failure_rate) else None,
            float(policy.get("maximum_execution_failure_rate", 0.02)),
            closed > 0
            and math.isfinite(failure_rate)
            and failure_rate <= float(policy.get("maximum_execution_failure_rate", 0.02)),
        ),
        (
            "duplicate_exchange_evidence",
            duplicate_exchange_evidence,
            0,
            duplicate_exchange_evidence == 0,
        ),
        (
            "duplicate_strategy_signals",
            duplicate_strategy_signals,
            0,
            duplicate_strategy_signals == 0,
        ),
        (
            "positive_after_cost_expectancy",
            mean_net if math.isfinite(mean_net) else None,
            True,
            closed > 0 and math.isfinite(mean_net) and mean_net > 0,
        ),
        (
            "positive_lifecycle_share",
            positive_outcome_share,
            float(policy.get("minimum_positive_lifecycle_share", 0.55)),
            closed > 0
            and positive_outcome_share
            >= float(policy.get("minimum_positive_lifecycle_share", 0.55)),
        ),
        (
            "after_cost_mean_lcb_95_usd",
            mean_lcb_95 if math.isfinite(mean_lcb_95) else None,
            float(policy.get("minimum_after_cost_mean_lcb_95_usd", 0.0)),
            closed > 0
            and math.isfinite(mean_lcb_95)
            and mean_lcb_95 > float(policy.get("minimum_after_cost_mean_lcb_95_usd", 0.0)),
        ),
        (
            "pair_sample_concentration",
            pair_sample_share,
            float(policy.get("maximum_pair_sample_share", 0.50)),
            closed > 0
            and pair_sample_share <= float(policy.get("maximum_pair_sample_share", 0.50)),
        ),
        (
            "regime_sample_concentration",
            regime_sample_share,
            float(policy.get("maximum_regime_sample_share", 0.50)),
            closed > 0
            and regime_sample_share <= float(policy.get("maximum_regime_sample_share", 0.50)),
        ),
        ("prospective_feature_and_outcome_logging", all_prospective, True, all_prospective),
        ("no_risk_policy_override", no_risk_override, True, no_risk_override),
    ]
    frame = pd.DataFrame(
        [
            {
                "check": check,
                "observed": observed,
                "required": required,
                "status": "PASS" if passed else "BLOCKED",
                "blocker": "" if passed else f"testnet_sample_{check}_insufficient",
                "testnet_order_authority": False,
                "live_trading_authorized": False,
                "evidence_path": ";".join(
                    path
                    for path in (
                        _relative(index_path, root),
                        _relative(policy_path, root),
                        _relative(candidate_path, root),
                        _relative(attempt_report_path, root),
                        (
                            _relative(candidate_queue_path, root)
                            if candidate_queue_path is not None
                            else ""
                        ),
                        (
                            _relative(frozen_policy_path, root)
                            if frozen_policy_path is not None
                            else ""
                        ),
                    )
                    if path
                ),
            }
            for check, observed, required, passed in rows
        ]
    )
    frame["calculation_method"] = ""
    frame.loc[
        frame["check"].eq("after_cost_mean_lcb_95_usd"),
        "calculation_method",
    ] = SAMPLE_MEAN_LCB_METHOD
    path = root / "reports" / "active" / "realized_testnet_sample_sufficiency.csv"
    _atomic_csv(frame, path)
    return {
        "path": path,
        "frame": frame,
        "status": "PASS" if frame["status"].eq("PASS").all() else "BLOCKED",
        "active_model_training_dataset_id": active_training_dataset_id,
        "active_model_artifact_sha256": active_model_sha256,
        "active_registered_learning_id": active_stage5_lineage.get("registered_learning_id", ""),
        "active_registered_stage5_protocol_id": active_stage5_lineage.get(
            "registered_stage5_protocol_id", ""
        ),
        "active_registered_execution_id": active_stage5_lineage.get("registered_execution_id", ""),
        "validated_lifecycle_samples": len(testnet),
        "current_model_cohort_samples": len(cohort),
        "excluded_obsolete_model_samples": excluded_obsolete_model_samples,
        "excluded_cross_lineage_samples": excluded_obsolete_model_samples,
        "current_model_terminal_execution_attempts": terminal_attempt_count,
        "current_model_failed_execution_attempts": len(failed_approval_ids),
        "current_model_pending_execution_attempts": len(pending_approval_ids),
        "invalid_execution_attempt_evidence": invalid_attempt_evidence,
    }


def build_testnet_supreme_team_checkpoint(
    *, root: Path = ROOT, now: datetime | None = None
) -> dict[str, Any]:
    sample = build_testnet_sample_sufficiency(root=root)
    lifecycle = build_testnet_lifecycle_receipt(root=root, now=now)["receipt"]
    status = (
        "PASS"
        if sample["status"] == "PASS" and lifecycle["lifecycle_status"] == "COMPLETE_RECONCILED"
        else "BLOCKED"
    )
    blockers = list(
        sample["frame"].loc[sample["frame"]["status"].ne("PASS"), "blocker"].astype(str)
    )
    blockers.extend(lifecycle.get("blockers", []))
    checkpoint = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": _as_utc(now).isoformat(),
        "checkpoint_status": status,
        "gap_analysis": "Realized paired Testnet outcomes are absent or below the prospective sample policy.",
        "pre_mortem": "Advancing would confuse deterministic simulations and preflight checks with realized execution evidence.",
        "post_mortem": "No Testnet order was sent because upstream survivor, cadence, cost, parity, and authorization gates remain blocked.",
        "red_team": "A forged confidence score, stale receipt, or simulated lifecycle cannot satisfy this checkpoint.",
        "blockers": list(dict.fromkeys(blockers)),
        "recommendation": "remain_research_only_and_continue_daily_evidence_collection",
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    directory = root / "reports" / "supreme_team"
    path = directory / "testnet_evidence_checkpoint.json"
    markdown = directory / "testnet_evidence_checkpoint.md"
    _atomic_json(checkpoint, path)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text(_checkpoint_markdown(checkpoint), encoding="utf-8")
    return {"path": path, "markdown": markdown, "checkpoint": checkpoint}


def build_stage6_release_receipt(
    *, root: Path = ROOT, now: datetime | None = None
) -> dict[str, Any]:
    """Seal Stage 6 evidence only after rebuilding every authoritative gate."""

    as_of = _as_utc(now)
    active = root / "reports" / "active"
    sample = build_testnet_sample_sufficiency(root=root)
    supreme = build_testnet_supreme_team_checkpoint(root=root, now=as_of)
    candidate_path = active / "testnet_candidate_receipt.json"
    candidate = _read_json(candidate_path)
    candidate_valid, candidate_blockers = _validated_testnet_candidate_receipt(
        root=root,
        candidate=candidate,
    )
    sample_path = Path(sample["path"])
    supreme_path = Path(supreme["path"])
    index_path = root / "data" / "testnet" / "lifecycle_index.csv"
    policy, policy_path, policy_blockers = _seal_testnet_sample_policy(root=root)
    lifecycles = _validated_lifecycle_index(root=root, index_path=index_path)
    archive_hashes = sorted(
        lifecycles.get("receipt_hash", pd.Series(dtype=str)).dropna().astype(str)
    )
    blockers = list(candidate_blockers)
    if sample["status"] != "PASS":
        blockers.append("stage6_release_sample_sufficiency_not_passed")
    if supreme["checkpoint"].get("checkpoint_status") != "PASS":
        blockers.append("stage6_release_supreme_team_not_passed")
    if not candidate_valid:
        blockers.append("stage6_release_candidate_invalid")
    if lifecycles.empty:
        blockers.append("stage6_release_validated_lifecycle_archive_empty")
    blockers.extend(policy_blockers)
    if not all(
        path.is_file()
        for path in (
            candidate_path,
            sample_path,
            supreme_path,
            index_path,
            policy_path or active / ".missing_stage6_sample_policy.json",
        )
    ):
        blockers.append("stage6_release_source_artifact_missing")

    receipt_path: Path | None = None
    receipt: dict[str, Any] = {}
    if not blockers:
        from quant_platform.orchestration.hyperliquid_learning_and_risk import (
            TESTNET_STAGE5_COHORT_BINDING_FIELDS,
        )

        core = {
            "schema_version": STAGE6_RELEASE_RECEIPT_SCHEMA_VERSION,
            "generated_at_utc": as_of.isoformat(),
            "candidate_receipt_id": str(candidate.get("candidate_receipt_id", "")),
            "candidate_receipt_sha256": str(candidate.get("receipt_sha256", "")),
            "candidate_queue_id": str(candidate.get("candidate_queue_id", "")),
            "candidate_queue_sha256": str(candidate.get("candidate_queue_sha256", "")),
            **{
                field: str(candidate.get(field, ""))
                for field in TESTNET_STAGE5_COHORT_BINDING_FIELDS
            },
            "sample_policy_id": str(policy.get("sample_policy_id", "")),
            "sample_policy_sha256": _sha256_file(policy_path),
            "validated_lifecycle_count": len(lifecycles),
            "validated_lifecycle_receipt_hashes": archive_hashes,
            "lifecycle_index_sha256": _sha256_file(index_path),
            "testnet_sample_evidence_sha256": _sha256_file(sample_path),
            "testnet_supreme_team_evidence_sha256": _sha256_file(supreme_path),
            "source_artifact_hashes": {
                _relative(candidate_path, root): _sha256_file(candidate_path),
                _relative(index_path, root): _sha256_file(index_path),
                _relative(policy_path, root): _sha256_file(policy_path),
                _relative(sample_path, root): _sha256_file(sample_path),
                _relative(supreme_path, root): _sha256_file(supreme_path),
            },
            "stage6_release_authority": True,
            "order_submission_performed": False,
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        }
        release_id = "stage6release_" + _payload_hash(core)[:20]
        receipt = {**core, "stage6_release_receipt_id": release_id}
        receipt["receipt_sha256"] = _payload_hash(receipt)
        receipt_path = root / "data" / "testnet" / "stage6_releases" / f"{release_id}.json"
        _write_immutable_json(receipt, receipt_path)

    pointer = {
        "schema_version": STAGE6_RELEASE_POINTER_SCHEMA_VERSION,
        "generated_at_utc": as_of.isoformat(),
        "status": "PASS" if receipt_path is not None else "BLOCKED",
        "stage6_release_receipt_id": str(receipt.get("stage6_release_receipt_id", "")),
        "stage6_release_receipt_path": (
            _relative(receipt_path, root) if receipt_path is not None else ""
        ),
        "stage6_release_receipt_sha256": (
            _sha256_file(receipt_path) if receipt_path is not None else ""
        ),
        "blockers": list(dict.fromkeys(blockers)),
        "order_submission_performed": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    pointer["receipt_sha256"] = _payload_hash(pointer)
    pointer_path = active / "stage6_release_status.json"
    _atomic_json(pointer, pointer_path)
    return {
        "status": pointer["status"],
        "pointer": pointer,
        "pointer_path": pointer_path,
        "receipt": receipt,
        "receipt_path": receipt_path,
        "blockers": pointer["blockers"],
    }


def validate_stage6_release_evidence(
    *,
    root: Path,
    candidate: dict[str, Any],
    sample_evidence_path: Path,
    supreme_evidence_path: Path,
) -> tuple[bool, dict[str, Any], Path | None, list[str]]:
    """Revalidate immutable Stage 6 closure without trusting PASS labels."""

    pointer_path = root / "reports" / "active" / "stage6_release_status.json"
    pointer = _read_json(pointer_path)
    blockers: list[str] = []
    if not (
        pointer.get("schema_version") == STAGE6_RELEASE_POINTER_SCHEMA_VERSION
        and pointer.get("status") == "PASS"
        and pointer.get("receipt_sha256") == _payload_hash(pointer)
        and pointer.get("order_submission_performed") is False
        and pointer.get("testnet_order_authority") is False
        and pointer.get("live_trading_authorized") is False
        and not pointer.get("blockers")
    ):
        blockers.append("stage6_release_pointer_invalid_or_blocked")
    receipt_path = _safe_repo_artifact(
        root,
        str(pointer.get("stage6_release_receipt_path", "")),
    )
    receipt = _read_json(receipt_path) if receipt_path is not None else {}
    if not (
        receipt_path is not None
        and receipt_path.is_file()
        and receipt_path.parent.resolve()
        == (root / "data" / "testnet" / "stage6_releases").resolve()
        and pointer.get("stage6_release_receipt_sha256") == _sha256_file(receipt_path)
        and pointer.get("stage6_release_receipt_id") == receipt.get("stage6_release_receipt_id")
        and receipt.get("schema_version") == STAGE6_RELEASE_RECEIPT_SCHEMA_VERSION
        and receipt.get("receipt_sha256") == _payload_hash(receipt)
        and receipt.get("stage6_release_authority") is True
        and receipt.get("order_submission_performed") is False
        and receipt.get("testnet_order_authority") is False
        and receipt.get("live_trading_authorized") is False
    ):
        blockers.append("stage6_release_immutable_receipt_invalid")
    candidate_valid, candidate_blockers = _validated_testnet_candidate_receipt(
        root=root,
        candidate=candidate,
    )
    if not candidate_valid:
        blockers.append("stage6_release_current_candidate_invalid")
        blockers.extend(candidate_blockers)
    if receipt:
        from quant_platform.orchestration.hyperliquid_learning_and_risk import (
            TESTNET_STAGE5_COHORT_BINDING_FIELDS,
        )

        candidate_bindings = {
            "candidate_receipt_id": candidate.get("candidate_receipt_id", ""),
            "candidate_receipt_sha256": candidate.get("receipt_sha256", ""),
            "candidate_queue_id": candidate.get("candidate_queue_id", ""),
            "candidate_queue_sha256": candidate.get("candidate_queue_sha256", ""),
            **{field: candidate.get(field, "") for field in TESTNET_STAGE5_COHORT_BINDING_FIELDS},
        }
        if any(receipt.get(field) != value for field, value in candidate_bindings.items()):
            blockers.append("stage6_release_candidate_lineage_mismatch")
    source_hashes = receipt.get("source_artifact_hashes")
    if not isinstance(source_hashes, dict) or not source_hashes:
        blockers.append("stage6_release_source_closure_missing")
    else:
        for relative, expected_hash in source_hashes.items():
            source = _safe_repo_artifact(root, str(relative))
            if source is None or not source.is_file() or _sha256_file(source) != str(expected_hash):
                blockers.append(f"stage6_release_source_changed:{relative}")
    if not sample_evidence_path.is_file() or receipt.get(
        "testnet_sample_evidence_sha256"
    ) != _sha256_file(sample_evidence_path):
        blockers.append("stage6_release_sample_evidence_mismatch")
    if not supreme_evidence_path.is_file() or receipt.get(
        "testnet_supreme_team_evidence_sha256"
    ) != _sha256_file(supreme_evidence_path):
        blockers.append("stage6_release_supreme_evidence_mismatch")
    index_path = root / "data" / "testnet" / "lifecycle_index.csv"
    lifecycles = _validated_lifecycle_index(root=root, index_path=index_path)
    observed_hashes = sorted(
        lifecycles.get("receipt_hash", pd.Series(dtype=str)).dropna().astype(str)
    )
    if (
        lifecycles.empty
        or receipt.get("validated_lifecycle_count") != len(lifecycles)
        or receipt.get("validated_lifecycle_receipt_hashes") != observed_hashes
        or not index_path.is_file()
        or receipt.get("lifecycle_index_sha256") != _sha256_file(index_path)
    ):
        blockers.append("stage6_release_lifecycle_archive_closure_invalid")
    rebuilt_sample = build_testnet_sample_sufficiency(root=root)
    if (
        rebuilt_sample["status"] != "PASS"
        or Path(rebuilt_sample["path"]).resolve() != sample_evidence_path.resolve()
        or receipt.get("testnet_sample_evidence_sha256") != _sha256_file(sample_evidence_path)
    ):
        blockers.append("stage6_release_authoritative_sample_rebuild_failed")
    supreme = _read_json(supreme_evidence_path)
    if not (
        supreme.get("checkpoint_status") == "PASS"
        and not supreme.get("blockers")
        and supreme.get("testnet_order_authority") is False
        and supreme.get("live_trading_authorized") is False
    ):
        blockers.append("stage6_release_supreme_checkpoint_invalid")
    policy_id = str(receipt.get("sample_policy_id", ""))
    policy_path = root / "data" / "testnet" / "sample_policies" / f"{policy_id}.json"
    policy = _read_json(policy_path)
    policy_valid, policy_blockers = _validate_testnet_sample_policy(policy)
    blockers.extend(policy_blockers)
    if not (
        policy_valid
        and receipt.get("sample_policy_id") == policy.get("sample_policy_id")
        and policy_path.is_file()
        and receipt.get("sample_policy_sha256") == _sha256_file(policy_path)
    ):
        blockers.append("stage6_release_sample_policy_mismatch")
    return not blockers, receipt, receipt_path, list(dict.fromkeys(blockers))


def build_live_release_gates(
    *,
    root: Path = ROOT,
    now: datetime | None = None,
    parity_account_address: str | None = None,
    parity_testnet_session: Any | None = None,
    parity_live_session: Any | None = None,
    parity_calculation_artifact_path: Path | None = None,
) -> dict[str, Path]:
    sample = build_testnet_sample_sufficiency(root=root)
    supreme = build_testnet_supreme_team_checkpoint(root=root, now=now)["checkpoint"]
    stage6_release = build_stage6_release_receipt(root=root, now=now)
    candidate = _read_json(root / "reports" / "active" / "testnet_candidate_receipt.json")
    candidate_valid, _ = _validated_testnet_candidate_receipt(root=root, candidate=candidate)
    from quant_platform.orchestration.corrective_live_canary import (
        _parity_rows,
        _policy_id,
        _validate_policy_core,
        build_live_canary_control_plane,
    )
    from quant_platform.orchestration.corrective_live_parity_capture import (
        capture_live_input_parity_evidence,
    )

    as_of = _as_utc(now)
    active = root / "reports" / "active"
    policy = _read_json(root / "config" / "live_canary_policy.json")
    policy_valid, _policy_blockers = _validate_policy_core(policy)
    existing_evidence = _read_json(active / "testnet_live_input_parity_evidence.json")
    existing_parity = _parity_rows(
        root=root,
        evidence=existing_evidence,
        candidate=candidate,
        policy=policy,
        policy_id=_policy_id(policy),
        as_of=as_of,
        upstream_ready=bool(
            candidate_valid
            and sample["status"] == "PASS"
            and supreme.get("checkpoint_status") == "PASS"
            and policy_valid
        ),
    )
    parity_current = bool(not existing_parity.empty and existing_parity["status"].eq("PASS").all())
    approval = _read_json(active / "live_canary_user_approval.json")
    approval_committed = bool(
        approval.get("approved") is True
        or str(approval.get("approval_id", "")).strip()
        or str(approval.get("wallet_signature", "")).strip()
    )
    capture_status_path = active / "live_input_parity_capture_status.json"
    if parity_current or approval_committed:
        capture_status = {
            "schema_version": "thewiz.live_input_parity_capture_status.v1",
            "generated_at_utc": as_of.isoformat(),
            "status": (
                "PASS_REUSED_CURRENT"
                if parity_current
                else "BLOCKED_COMMITTED_APPROVAL_PARITY_STALE"
            ),
            "blockers": (
                []
                if parity_current
                else ["committed_live_approval_requires_explicit_reset_after_parity_expiry"]
            ),
            "evidence_path": str(active / "testnet_live_input_parity_evidence.json"),
            "parity_path": str(active / "testnet_live_input_parity.csv"),
            "stage6_upstream_revalidated": True,
            "parity_receipt_preserved": True,
            "approval_committed": approval_committed,
            "read_only_info_requests": True,
            "private_key_accessed": False,
            "order_submission_performed": False,
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        }
        capture_status["receipt_sha256"] = _payload_hash(capture_status)
        _atomic_json(capture_status, capture_status_path)
        capture = {"status_path": capture_status_path}
    else:
        capture = capture_live_input_parity_evidence(
            root=root,
            now=as_of,
            account_address=parity_account_address,
            testnet_session=parity_testnet_session,
            live_session=parity_live_session,
            calculation_artifact_path=parity_calculation_artifact_path,
            refresh_upstream=False,
            upstream_prevalidated=True,
        )

    control = build_live_canary_control_plane(
        root=root,
        now=as_of,
        candidate=candidate,
        candidate_valid=candidate_valid,
        sample_status=sample["status"],
        supreme_status=str(supreme.get("checkpoint_status", "BLOCKED")),
        sample_evidence_path=Path(sample["path"]),
        supreme_evidence_path=(
            root / "reports" / "supreme_team" / "testnet_evidence_checkpoint.json"
        ),
    )
    return {
        "capture_status": Path(capture["status_path"]),
        "parity": Path(control["parity"]),
        "authorization": Path(control["authorization"]),
        "outcome": Path(control["outcome"]),
        "approval": Path(control["approval"]),
        "stage6_release": Path(stage6_release["pointer_path"]),
        "executor_preflight": Path(control["executor_preflight"]),
    }


def build_corrective_release_gates(
    *, root: Path = ROOT, now: datetime | None = None
) -> CommandResult:
    as_of = _as_utc(now)
    prospective = build_testnet_prospective_cohort_readiness(root=root, now=as_of)
    candidate = build_testnet_candidate_receipt(root=root, now=as_of)
    preflight = build_no_order_preflight(root=root, now=as_of)
    lifecycle = build_testnet_lifecycle_receipt(root=root, now=as_of)
    archive = archive_validated_testnet_lifecycle(root=root, now=as_of)
    sample = build_testnet_sample_sufficiency(root=root)
    supreme = build_testnet_supreme_team_checkpoint(root=root, now=as_of)
    live = build_live_release_gates(root=root, now=as_of)
    return CommandResult(
        paths={
            **prospective.paths,
            "testnet_candidate": Path(candidate["path"]),
            "testnet_candidate_queue": root / "reports" / "active" / "testnet_candidate_queue.csv",
            "testnet_candidate_queue_receipt": root
            / "reports"
            / "active"
            / "testnet_candidate_queue_receipt.json",
            "no_order_preflight": Path(preflight["path"]),
            "testnet_lifecycle": Path(lifecycle["path"]),
            "testnet_lifecycle_index": Path(archive["index"]),
            "testnet_sample_sufficiency": Path(sample["path"]),
            "testnet_supreme_team": Path(supreme["path"]),
            "testnet_supreme_team_markdown": Path(supreme["markdown"]),
            "stage6_release_status": live["stage6_release"],
            "testnet_live_input_parity": live["parity"],
            "live_input_parity_capture_status": live["capture_status"],
            "live_canary_user_approval": live["approval"],
            "live_canary_executor_preflight": live["executor_preflight"],
            "live_canary_authorization": live["authorization"],
            "live_canary_outcome": live["outcome"],
        },
        summary={
            "status": "BLOCKED",
            "prospective_cohort_readiness_status": prospective.summary["status"],
            "prospective_cohort_pairs": prospective.summary["cohort_pairs"],
            "prospectively_ready_pairs": prospective.summary["prospectively_ready_pairs"],
            "testnet_candidate_status": candidate["receipt"]["candidate_status"],
            "no_order_preflight_status": preflight["status"],
            "testnet_lifecycle_status": lifecycle["receipt"]["lifecycle_status"],
            "testnet_lifecycle_archive_status": archive["status"],
            "testnet_sample_status": sample["status"],
            "stage6_release_status": _read_json(live["stage6_release"]).get("status", "BLOCKED"),
            "supreme_team_status": supreme["checkpoint"]["checkpoint_status"],
            "live_input_parity_capture_status": _read_json(live["capture_status"]).get(
                "status", "BLOCKED"
            ),
            "orders_submitted": 0,
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        },
    )


def _preflight_ready(frame: pd.DataFrame, token: str) -> bool:
    if frame.empty:
        return False
    columns = [column for column in frame.columns if token in column.lower()]
    if not columns:
        return False
    return any(frame[column].map(_truthy).any() for column in columns)


def _approval_candidate_sizing_ready(
    approval: dict[str, Any], candidate: dict[str, Any]
) -> tuple[bool, bool]:
    from quant_platform.orchestration.hyperliquid_learning_and_risk import (
        TESTNET_CANDIDATE_BINDING_FIELDS,
    )

    binding_fields = TESTNET_CANDIDATE_BINDING_FIELDS
    binding_ready = bool(
        candidate.get("candidate_status") == "READY_FOR_NO_ORDER_PREFLIGHT"
        and all(str(candidate.get(field, "")).strip() for field in binding_fields)
        and all(
            str(approval.get(field, "")).strip() == str(candidate.get(field, "")).strip()
            for field in binding_fields
        )
    )
    legs = approval.get("legs") if isinstance(approval.get("legs"), list) else []
    assets = {
        str(candidate.get("asset_x", "")).strip().upper(),
        str(candidate.get("asset_y", "")).strip().upper(),
    } - {""}
    markets: set[str] = set()
    sides: set[str] = set()
    notionals: list[float] = []
    for leg in legs:
        if not isinstance(leg, dict):
            continue
        market = str(leg.get("market", "")).strip().upper()
        market = market.removesuffix("-USD").removesuffix("/USD")
        side = str(leg.get("side", "")).strip().upper()
        size = _finite(leg.get("size"))
        price = _finite(leg.get("limit_price"))
        if market and side in {"BUY", "SELL"} and size > 0 and price > 0:
            markets.add(market)
            sides.add(side)
            notionals.append(size * price)
    sizing_ready = bool(
        binding_ready
        and len(legs) == 2
        and markets == assets
        and sides == {"BUY", "SELL"}
        and len(notionals) == 2
        and all(10.0 <= notional <= 12.5 for notional in notionals)
        and sum(notionals) <= 25.0
        and _finite(approval.get("max_total_notional_usd")) == 25.0
        and approval.get("one_run_only") is True
        and str(approval.get("network", "")).lower() == "testnet"
    )
    return binding_ready, sizing_ready


def _validate_lifecycle_archive(*, archive_path: Path, receipt_hash: str) -> tuple[bool, list[str]]:
    from quant_platform.orchestration.hyperliquid_learning_and_risk import (
        TESTNET_CANDIDATE_BINDING_FIELDS,
        _testnet_receipt_checks,
        _testnet_receipt_economics_valid,
        _testnet_receipt_payload_hash,
    )

    blockers: list[str] = []
    manifest_path = archive_path / "archive_manifest.json"
    receipt_path = archive_path / "receipt.json"
    approval_path = archive_path / "approval.json"
    candidate_path = archive_path / "candidate.json"
    candidate_identity_path = archive_path / "candidate_identity.json"
    candidate_queue_path = archive_path / "candidate_queue.json"
    run_manifest_path = archive_path / "run_manifest.json"
    protocol_path = archive_path / "protocol.json"
    sample_policy_path = archive_path / "sample_policy.json"
    manifest = _read_json(manifest_path)
    receipt = _read_json(receipt_path)
    approval = _read_json(approval_path)
    candidate = _read_json(candidate_path)
    run_manifest = _read_json(run_manifest_path)
    protocol = _read_json(protocol_path)
    sample_policy = _read_json(sample_policy_path)
    if archive_path.name != receipt_hash:
        blockers.append("testnet_lifecycle_archive_directory_hash_mismatch")
    if not manifest or manifest.get("receipt_hash") != receipt_hash:
        blockers.append("testnet_lifecycle_archive_manifest_invalid")
    if (
        not receipt
        or receipt.get("receipt_hash") != receipt_hash
        or _testnet_receipt_payload_hash(receipt) != receipt_hash
    ):
        blockers.append("testnet_lifecycle_archive_receipt_hash_invalid")
    if not approval or approval.get("risk_override_applied") is not False:
        blockers.append("testnet_lifecycle_archive_approval_invalid")
    if not _testnet_receipt_economics_valid(receipt):
        blockers.append("testnet_lifecycle_archive_economics_invalid")
    candidate_valid, candidate_blockers = _validated_testnet_candidate_receipt(
        root=archive_path,
        candidate=candidate,
        identity_path=candidate_identity_path,
        queue_path=candidate_queue_path,
        verify_current_source_artifacts=False,
    )
    if not candidate_valid:
        blockers.extend(candidate_blockers)
    sample_policy_valid, sample_policy_blockers = _validate_testnet_sample_policy(sample_policy)
    if not sample_policy_valid:
        blockers.extend(sample_policy_blockers)
    archived_candidate_binding = {
        field: str(candidate.get(field, "")).strip() for field in TESTNET_CANDIDATE_BINDING_FIELDS
    }
    archived_receipt_checks = _testnet_receipt_checks(
        receipt,
        approval=approval,
        protocol=protocol,
        root=archive_path,
        manifest=run_manifest,
        candidate_binding=(archived_candidate_binding if candidate_valid else {}),
    )
    blockers.extend(
        f"testnet_lifecycle_archive_reconciliation_failed:{name}:{blocker}"
        for name, passed, blocker in archived_receipt_checks
        if not passed
    )
    final_state = receipt.get("final_state") if isinstance(receipt.get("final_state"), dict) else {}
    sample_policy_path = archive_path / "sample_policy.json"
    sample_policy = _read_json(sample_policy_path)
    if not (
        receipt.get("actual_testnet") is True
        and receipt.get("risk_override_applied") is False
        and final_state.get("reconciled") is True
        and _finite(final_state.get("position_x"), default=math.nan) == 0.0
        and _finite(final_state.get("position_y"), default=math.nan) == 0.0
        and _finite(final_state.get("open_order_count"), default=math.nan) == 0.0
    ):
        blockers.append("testnet_lifecycle_archive_not_actual_reconciled_flat")
    artifacts = manifest.get("artifacts") if isinstance(manifest.get("artifacts"), list) else []
    required = {
        "receipt",
        "approval",
        "run_manifest",
        "protocol",
        "candidate",
        "candidate_identity",
        "candidate_queue",
        "sample_policy",
        "lifecycle_gate",
    }
    observed: set[str] = set()
    archived_candidate_sources: list[tuple[str, str]] = []
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        name = str(artifact.get("name", ""))
        source_path = str(artifact.get("source_path", ""))
        file_name = str(artifact.get("archive_file", ""))
        expected_sha = str(artifact.get("sha256", "")).lower()
        path = archive_path / file_name
        if name:
            observed.add(name)
        if name.startswith("candidate_source_"):
            archived_candidate_sources.append((source_path, expected_sha))
        if (
            not file_name
            or path.parent != archive_path
            or not path.is_file()
            or not expected_sha
            or _sha256_file(path) != expected_sha
        ):
            blockers.append(f"testnet_lifecycle_archive_artifact_invalid:{name or 'unknown'}")
    if not required.issubset(observed):
        blockers.append("testnet_lifecycle_archive_artifact_set_incomplete")
    queue = _read_json(candidate_queue_path)
    queue_source_hashes = queue.get("source_artifact_hashes")
    expected_candidate_sources = (
        sorted((str(path), str(digest).lower()) for path, digest in queue_source_hashes.items())
        if isinstance(queue_source_hashes, dict)
        else []
    )
    if (
        not expected_candidate_sources
        or sorted(archived_candidate_sources) != expected_candidate_sources
    ):
        blockers.append("testnet_lifecycle_archive_candidate_source_closure_invalid")
    return not blockers, list(dict.fromkeys(blockers))


def _lifecycle_index_row(
    *,
    receipt: dict[str, Any],
    approval: dict[str, Any],
    manifest: dict[str, Any],
    archive_path: Path,
    root: Path,
) -> dict[str, Any]:
    from quant_platform.orchestration.hyperliquid_learning_and_risk import (
        TESTNET_CANDIDATE_BINDING_FIELDS,
    )

    events = receipt.get("events") if isinstance(receipt.get("events"), list) else []
    valid_events = [event for event in events if isinstance(event, dict)]
    anomaly_count = sum(event.get("event_type") == "execution_anomaly" for event in valid_events)
    recovery_count = sum(
        event.get("event_type")
        in {"partial_fill_recovery", "cancel_and_unwind", "emergency_flatten"}
        for event in valid_events
    )
    context = receipt.get("entry_context") if isinstance(receipt.get("entry_context"), dict) else {}
    economics = receipt.get("economics") if isinstance(receipt.get("economics"), dict) else {}
    final_state = receipt.get("final_state") if isinstance(receipt.get("final_state"), dict) else {}
    sample_policy_path = archive_path / "sample_policy.json"
    sample_policy = _read_json(sample_policy_path)
    terminal_fills = _terminal_fills(receipt)
    network = str(receipt.get("network", "testnet")).strip().lower()
    fill_trade_keys = sorted(
        f"{network}|{str(fill.get('trade_id', '')).strip()}"
        for fill in terminal_fills
        if str(fill.get("trade_id", "")).strip()
    )
    exchange_order_keys = sorted(
        {
            f"{network}|{str(fill.get('order_id', '')).strip()}"
            for fill in terminal_fills
            if str(fill.get("order_id", "")).strip()
        }
    )
    feature_time = pd.to_datetime(context.get("feature_timestamp_utc"), utc=True, errors="coerce")
    started = pd.to_datetime(receipt.get("started_at_utc"), utc=True, errors="coerce")
    prospective = bool(
        pd.notna(feature_time)
        and pd.notna(started)
        and feature_time <= started
        and str(context.get("regime", "")).strip()
        and str(context.get("strategy_signal_id", "")).strip()
        and _finite(context.get("trade_quality_score"), default=math.nan) >= 0.0
        and _finite(context.get("trade_quality_score"), default=math.nan) <= 1.0
    )
    return {
        "schema_version": "thewiz.testnet_lifecycle_index.v3",
        "lifecycle_sample_id": f"testnetlifecycle_{str(receipt.get('receipt_hash', ''))[:20]}",
        "receipt_hash": str(receipt.get("receipt_hash", "")),
        "approval_id": str(receipt.get("approval_id", "")),
        "archive_manifest_sha256": _sha256_file(archive_path / "archive_manifest.json"),
        "sample_policy_id": str(sample_policy.get("sample_policy_id", "")),
        "sample_policy_sha256": _sha256_file(sample_policy_path),
        "archive_path": _relative(archive_path, root),
        "archived_at_utc": str(manifest.get("archived_at_utc", "")),
        **{field: str(receipt.get(field, "")) for field in TESTNET_CANDIDATE_BINDING_FIELDS},
        "feature_timestamp_utc": str(context.get("feature_timestamp_utc", "")),
        "regime": str(context.get("regime", "")),
        "trade_quality_score": _finite(context.get("trade_quality_score")),
        "strategy_signal_id": str(context.get("strategy_signal_id", "")),
        "fill_trade_keys_json": _canonical_json(fill_trade_keys),
        "exchange_order_keys_json": _canonical_json(exchange_order_keys),
        "fill_identity_sha256": sha256(
            _canonical_json(fill_trade_keys).encode("utf-8")
        ).hexdigest(),
        "order_identity_sha256": sha256(
            _canonical_json(exchange_order_keys).encode("utf-8")
        ).hexdigest(),
        "started_at_utc": str(receipt.get("started_at_utc", "")),
        "completed_at_utc": str(receipt.get("completed_at_utc", "")),
        "terminal_fill_count": int(_finite(economics.get("terminal_fill_count"), default=0.0)),
        "gross_realized_pnl_usd": _finite(economics.get("gross_realized_pnl_usd")),
        "fees_usd": _finite(economics.get("fees_usd")),
        "funding_pnl_usd": _finite(economics.get("funding_pnl_usd")),
        "implementation_shortfall_vs_limit_usd": _finite(
            economics.get("implementation_shortfall_vs_limit_usd")
        ),
        "net_realized_pnl_after_cost_usd": _finite(
            economics.get("net_realized_pnl_after_cost_usd")
        ),
        "anomaly_count": anomaly_count,
        "recovery_count": recovery_count,
        "execution_failure": int(anomaly_count > 0),
        "unresolved_orphan_legs": 0 if final_state.get("reconciled") is True else 1,
        "unreconciled_orders": int(_finite(final_state.get("open_order_count"), default=0.0)),
        "reconciled_flat": bool(final_state.get("reconciled") is True),
        "prospective_evidence": prospective,
        "risk_override_applied": approval.get("risk_override_applied") is True,
        "actual_testnet": receipt.get("actual_testnet") is True,
        "exchange_backed": receipt.get("receipt_source")
        == "hyperliquid_testnet_lifecycle_evidence_capture",
    }


def _index_row_matches_receipt(existing: dict[str, Any], expected: dict[str, Any]) -> bool:
    for key, expected_value in expected.items():
        actual = existing.get(key)
        if isinstance(expected_value, bool):
            if _truthy(actual) != expected_value:
                return False
        elif isinstance(expected_value, (int, float)):
            actual_number = _finite(actual)
            expected_number = _finite(expected_value)
            if not (
                math.isfinite(actual_number)
                and math.isfinite(expected_number)
                and abs(actual_number - expected_number) <= 1e-9
            ):
                return False
        elif str(actual) != str(expected_value):
            return False
    return True


def _filter_testnet_lifecycles_by_stage5_lineage(
    frame: pd.DataFrame,
    lineage: dict[str, Any],
) -> pd.DataFrame:
    from quant_platform.orchestration.hyperliquid_learning_and_risk import (
        TESTNET_STAGE5_COHORT_BINDING_FIELDS,
    )

    expected = {
        field: str(lineage.get(field, "")).strip() for field in TESTNET_STAGE5_COHORT_BINDING_FIELDS
    }
    if (
        frame.empty
        or not all(expected.values())
        or not set(TESTNET_STAGE5_COHORT_BINDING_FIELDS).issubset(frame.columns)
    ):
        return frame.iloc[0:0].copy()
    matches = pd.Series(True, index=frame.index, dtype=bool)
    for field, value in expected.items():
        matches &= frame[field].astype(str).eq(value)
    return frame.loc[matches].copy()


def _validated_lifecycle_index(*, root: Path, index_path: Path) -> pd.DataFrame:
    index = _read_csv(index_path)
    if index.empty or "receipt_hash" not in index:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _, indexed in index.iterrows():
        receipt_hash = str(indexed.get("receipt_hash", "")).strip().lower()
        if not receipt_hash or receipt_hash in seen:
            continue
        archive_path = root / "data" / "testnet" / "lifecycle_receipts" / receipt_hash
        valid, _ = _validate_lifecycle_archive(archive_path=archive_path, receipt_hash=receipt_hash)
        if not valid:
            continue
        receipt = _read_json(archive_path / "receipt.json")
        approval = _read_json(archive_path / "approval.json")
        manifest = _read_json(archive_path / "archive_manifest.json")
        expected = _lifecycle_index_row(
            receipt=receipt,
            approval=approval,
            manifest=manifest,
            archive_path=archive_path,
            root=root,
        )
        if not _index_row_matches_receipt(indexed.to_dict(), expected):
            continue
        rows.append(expected)
        seen.add(receipt_hash)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    duplicate_exchange = [False] * len(frame)
    duplicate_signal = frame["strategy_signal_id"].astype(str).duplicated(keep=False).tolist()
    key_members: dict[str, list[int]] = {}
    for row_index, row in frame.iterrows():
        for field in ("fill_trade_keys_json", "exchange_order_keys_json"):
            for key in _json_string_list(row.get(field)):
                key_members.setdefault(f"{field}|{key}", []).append(int(row_index))
    for members in key_members.values():
        if len(set(members)) > 1:
            for member in members:
                duplicate_exchange[member] = True
    frame["duplicate_exchange_evidence"] = duplicate_exchange
    frame["duplicate_strategy_signal"] = duplicate_signal
    frame["independent_sample_eligible"] = ~(
        frame["duplicate_exchange_evidence"] | frame["duplicate_strategy_signal"]
    )
    return frame


def _validated_testnet_execution_attempts(*, root: Path) -> tuple[pd.DataFrame, Path]:
    """Reconcile every one-use reservation with its immutable executor result."""

    from quant_platform.orchestration.hyperliquid_learning_and_risk import (
        TESTNET_CANDIDATE_BINDING_FIELDS,
    )

    execution_root = root / "data" / "testnet" / "pair_executions"
    reservation_root = root / "data" / "testnet" / "pair_reservations"
    preflight_root = root / "data" / "testnet" / "pair_preflights"
    report_path = root / "reports" / "active" / "testnet_execution_attempts.csv"
    reservations: dict[str, tuple[dict[str, Any], Path]] = {}
    for path in sorted(reservation_root.glob("*.json")):
        reservation = _read_json(path)
        preflight_id = str(reservation.get("preflight_id", "")).strip()
        if preflight_id:
            reservations[preflight_id] = (reservation, path)

    rows: list[dict[str, Any]] = []
    execution_preflights: set[str] = set()
    for path in sorted(execution_root.glob("*.json")):
        execution = _read_json(path)
        if execution.get("execution_invoked") is not True:
            continue
        blockers: list[str] = []
        execution_id = str(execution.get("execution_id", "")).strip()
        execution_core = {key: value for key, value in execution.items() if key != "execution_id"}
        expected_id = (
            "testnetpairexecution_"
            + sha256(_canonical_json(execution_core).encode("utf-8")).hexdigest()[:20]
        )
        if execution.get("schema_version") != "thewiz.hyperliquid_testnet_pair_execution.v1":
            blockers.append("testnet_attempt_execution_schema_invalid")
        if execution_id != expected_id:
            blockers.append("testnet_attempt_execution_identity_invalid")
        preflight_id = str(execution.get("preflight_id", "")).strip()
        preflight_path = _safe_repo_artifact(
            root, str(execution.get("immutable_preflight_path", ""))
        )
        preflight = _read_json(preflight_path) if preflight_path else {}
        if not (
            preflight_id
            and preflight_path is not None
            and preflight_path == preflight_root / f"{preflight_id}.json"
            and preflight_path.is_file()
            and _sha256_file(preflight_path) == str(execution.get("immutable_preflight_sha256", ""))
            and preflight.get("preflight_id") == preflight_id
            and preflight.get("status") == "READY_REQUIRES_EXPLICIT_EXECUTION"
            and preflight.get("blockers") == []
        ):
            blockers.append("testnet_attempt_immutable_preflight_invalid")
        result = execution.get("executor_result")
        result = result if isinstance(result, dict) else {}
        if not result or str(result.get("status", "")) != str(execution.get("status", "")):
            blockers.append("testnet_attempt_executor_result_invalid")
        if bool(result.get("order_submission_performed", False)) != bool(
            execution.get("order_submission_performed", False)
        ):
            blockers.append("testnet_attempt_submission_status_mismatch")
        if (
            execution.get("testnet_order_authority") is not False
            or execution.get("live_trading_authorized") is not False
        ):
            blockers.append("testnet_attempt_authority_boundary_invalid")
        if any(
            not str(execution.get(field, "")).strip()
            or str(execution.get(field, "")).strip() != str(preflight.get(field, "")).strip()
            for field in TESTNET_CANDIDATE_BINDING_FIELDS
        ):
            blockers.append("testnet_attempt_candidate_or_stage5_lineage_invalid")
        reservation_entry = reservations.get(preflight_id)
        if reservation_entry is None:
            blockers.append("testnet_attempt_one_use_reservation_missing")
            reservation_path = None
        else:
            reservation, reservation_path = reservation_entry
            if not (
                reservation.get("schema_version")
                == "thewiz.hyperliquid_testnet_pair_reservation.v1"
                and reservation.get("one_use") is True
                and reservation.get("status") == "CONSUMED"
                and reservation.get("approval_id") == execution.get("approval_id")
                and reservation.get("action") == execution.get("action")
                and reservation.get("executor_status") == execution.get("status")
            ):
                blockers.append("testnet_attempt_one_use_reservation_invalid")
        execution_preflights.add(preflight_id)
        status = str(execution.get("status", ""))
        rows.append(
            {
                "schema_version": "thewiz.testnet_execution_attempt_audit.v1",
                "execution_id": execution_id,
                "preflight_id": preflight_id,
                "approval_id": str(execution.get("approval_id", "")),
                "action": str(execution.get("action", "")),
                **{
                    field: str(execution.get(field, "")).strip()
                    for field in TESTNET_CANDIDATE_BINDING_FIELDS
                },
                "execution_status": status,
                "attempt_outcome": "PENDING" if status == "pair_submitted" else "FAILURE",
                "order_submission_performed": bool(
                    execution.get("order_submission_performed", False)
                ),
                "reconciled": bool(result.get("reconciled", False)),
                "evidence_valid": not blockers,
                "blockers": ";".join(dict.fromkeys(blockers)),
                "execution_path": _relative(path, root),
                "reservation_path": (_relative(reservation_path, root) if reservation_path else ""),
                "preflight_path": (_relative(preflight_path, root) if preflight_path else ""),
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            }
        )

    for preflight_id, (reservation, reservation_path) in reservations.items():
        if preflight_id in execution_preflights:
            continue
        preflight_path = preflight_root / f"{preflight_id}.json"
        preflight = _read_json(preflight_path)
        blockers = ["testnet_attempt_execution_receipt_missing"]
        if not (
            reservation.get("schema_version") == "thewiz.hyperliquid_testnet_pair_reservation.v1"
            and reservation.get("one_use") is True
            and preflight.get("preflight_id") == preflight_id
        ):
            blockers.append("testnet_attempt_orphan_reservation_invalid")
        rows.append(
            {
                "schema_version": "thewiz.testnet_execution_attempt_audit.v1",
                "execution_id": "",
                "preflight_id": preflight_id,
                "approval_id": str(reservation.get("approval_id", "")),
                "action": str(reservation.get("action", "")),
                **{
                    field: str(preflight.get(field, "")).strip()
                    for field in TESTNET_CANDIDATE_BINDING_FIELDS
                },
                "execution_status": str(reservation.get("executor_status", "")),
                "attempt_outcome": "PENDING",
                "order_submission_performed": bool(
                    reservation.get("order_submission_performed", False)
                ),
                "reconciled": bool(reservation.get("reconciled", False)),
                "evidence_valid": False,
                "blockers": ";".join(blockers),
                "execution_path": "",
                "reservation_path": _relative(reservation_path, root),
                "preflight_path": (
                    _relative(preflight_path, root) if preflight_path.is_file() else ""
                ),
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            }
        )
    columns = [
        "schema_version",
        "execution_id",
        "preflight_id",
        "approval_id",
        "action",
        *TESTNET_CANDIDATE_BINDING_FIELDS,
        "execution_status",
        "attempt_outcome",
        "order_submission_performed",
        "reconciled",
        "evidence_valid",
        "blockers",
        "execution_path",
        "reservation_path",
        "preflight_path",
        "testnet_order_authority",
        "live_trading_authorized",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    _atomic_csv(frame, report_path)
    return frame, report_path


def _terminal_fills(receipt: dict[str, Any]) -> list[dict[str, Any]]:
    events = receipt.get("events") if isinstance(receipt.get("events"), list) else []
    return [
        fill
        for event in events
        if isinstance(event, dict) and event.get("event_type") in {"two_leg_entry", "two_leg_exit"}
        for fill in (
            event.get("fill_evidence") if isinstance(event.get("fill_evidence"), list) else []
        )
        if isinstance(fill, dict)
    ]


def _json_string_list(value: Any) -> list[str]:
    try:
        payload = json.loads(str(value))
    except (TypeError, ValueError):
        return []
    if not isinstance(payload, list):
        return []
    return [str(item) for item in payload if str(item).strip()]


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _checkpoint_markdown(checkpoint: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Supreme Team: Realized Testnet Evidence",
            "",
            f"- Status: `{checkpoint['checkpoint_status']}`",
            "- Testnet order authority: `false`",
            "- Live trading authorized: `false`",
            "",
            f"**Gap analysis:** {checkpoint['gap_analysis']}",
            "",
            f"**Pre-mortem:** {checkpoint['pre_mortem']}",
            "",
            f"**Post-mortem:** {checkpoint['post_mortem']}",
            "",
            f"**Red team:** {checkpoint['red_team']}",
            "",
            "Blockers:",
            *[f"- `{blocker}`" for blocker in checkpoint["blockers"]],
            "",
        ]
    )


def _testnet_sample_policy_core(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key not in {"sample_policy_id", "receipt_sha256"}
    }


def _testnet_sample_policy_core_valid(payload: dict[str, Any]) -> bool:
    integer_floors = {
        "minimum_closed_paired_lifecycles": 30,
        "minimum_observation_days": 14,
        "minimum_independent_pairs": 3,
        "minimum_observed_regimes": 3,
        "minimum_lifecycles_per_candidate": 5,
        "minimum_lifecycles_per_pair": 5,
        "minimum_lifecycles_per_regime": 5,
    }
    integer_thresholds_valid = all(
        math.isfinite(_finite(payload.get(field), default=math.nan))
        and _finite(payload.get(field), default=math.nan).is_integer()
        and _finite(payload.get(field), default=math.nan) >= minimum
        for field, minimum in integer_floors.items()
    )
    maximum_failure_rate = _finite(payload.get("maximum_execution_failure_rate"), default=math.nan)
    minimum_positive_share = _finite(
        payload.get("minimum_positive_lifecycle_share"), default=math.nan
    )
    minimum_mean_lcb = _finite(payload.get("minimum_after_cost_mean_lcb_95_usd"), default=math.nan)
    maximum_pair_share = _finite(payload.get("maximum_pair_sample_share"), default=math.nan)
    maximum_regime_share = _finite(payload.get("maximum_regime_sample_share"), default=math.nan)
    return bool(
        payload.get("schema_version") == SAMPLE_POLICY_SCHEMA_VERSION
        and str(payload.get("policy_version", "")).strip()
        and pd.notna(pd.to_datetime(payload.get("effective_at_utc"), utc=True, errors="coerce"))
        and integer_thresholds_valid
        and _finite(payload.get("maximum_unresolved_orphan_legs"), default=math.nan) == 0.0
        and _finite(payload.get("maximum_unreconciled_orders"), default=math.nan) == 0.0
        and math.isfinite(maximum_failure_rate)
        and 0.0 <= maximum_failure_rate <= 0.02
        and math.isfinite(minimum_positive_share)
        and 0.55 <= minimum_positive_share <= 1.0
        and math.isfinite(minimum_mean_lcb)
        and minimum_mean_lcb >= 0.0
        and payload.get("after_cost_mean_lcb_95_method") == SAMPLE_MEAN_LCB_METHOD
        and math.isfinite(maximum_pair_share)
        and 0.0 < maximum_pair_share <= 0.50
        and math.isfinite(maximum_regime_share)
        and 0.0 < maximum_regime_share <= 0.50
        and payload.get("requires_positive_after_cost_expectancy") is True
        and payload.get("requires_no_risk_policy_override") is True
        and payload.get("requires_prospective_feature_and_outcome_logging") is True
        and payload.get("testnet_order_authority") is False
        and payload.get("live_trading_authorized") is False
    )


def _mean_lcb_95(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return math.nan
    if len(numeric) == 1:
        return float(numeric.iloc[0])
    standard_error = float(numeric.std(ddof=1)) / math.sqrt(len(numeric))
    critical_value = float(student_t.ppf(0.975, df=len(numeric) - 1))
    if not math.isfinite(standard_error) or not math.isfinite(critical_value):
        return math.nan
    return float(numeric.mean()) - critical_value * standard_error


def _testnet_sample_policy_id(payload: dict[str, Any]) -> str:
    core = _testnet_sample_policy_core(payload)
    if not _testnet_sample_policy_core_valid(core):
        return ""
    return "testnetsamplepolicy_" + _payload_hash(core)[:20]


def _validate_testnet_sample_policy(receipt: dict[str, Any]) -> tuple[bool, list[str]]:
    blockers: list[str] = []
    expected_id = _testnet_sample_policy_id(receipt)
    if not expected_id:
        blockers.append("testnet_sample_policy_schema_or_thresholds_invalid")
    if str(receipt.get("sample_policy_id", "")) != expected_id:
        blockers.append("testnet_sample_policy_id_invalid")
    if str(receipt.get("receipt_sha256", "")) != _payload_hash(receipt):
        blockers.append("testnet_sample_policy_receipt_hash_invalid")
    return not blockers, blockers


def _seal_testnet_sample_policy(*, root: Path) -> tuple[dict[str, Any], Path | None, list[str]]:
    config = _read_json(root / "config" / "testnet_sample_sufficiency_policy.json")
    policy_id = _testnet_sample_policy_id(config)
    if not policy_id:
        return {}, None, ["testnet_sample_policy_schema_or_thresholds_invalid"]
    receipt = {
        **_testnet_sample_policy_core(config),
        "sample_policy_id": policy_id,
    }
    receipt["receipt_sha256"] = _payload_hash(receipt)
    path = root / "data" / "testnet" / "sample_policies" / f"{policy_id}.json"
    try:
        _write_immutable_json(receipt, path)
    except ValueError as exc:
        return {}, None, [f"testnet_sample_policy_immutable_conflict:{exc}"]
    return receipt, path, []


def _validated_stage4_survivor_source_family(
    *,
    root: Path,
    stage4: dict[str, Any],
    survivor: dict[str, Any],
    current_matrix_path: Path,
) -> tuple[
    pd.DataFrame,
    dict[str, dict[str, Any]],
    Path,
    Path,
    Path,
]:
    """Resolve accepted Testnet identities from immutable Stage 4 evidence."""

    contract_id = str(stage4.get("contract_id", "")).strip()
    if not contract_id:
        raise ValueError("registered Stage 4 contract id is missing")
    contract_path = (
        root / "data" / "research" / "registered_rerun_contracts" / f"{contract_id}.json"
    )
    contract = _read_json(contract_path)
    if (
        contract.get("schema_version") != "thewiz.corrective_registered_rerun.v1"
        or contract.get("contract_id") != contract_id
        or contract.get("immutable_contract_path") != _relative(contract_path, root)
        or contract.get("source_family_sha256") != stage4.get("source_family_sha256")
        or any(
            _truthy(contract.get(field))
            for field in (
                "promotion_authority",
                "testnet_order_authority",
                "live_trading_authorized",
            )
        )
    ):
        raise ValueError("registered Stage 4 immutable contract is invalid")

    conclusion_path = _safe_repo_artifact(root, str(stage4.get("conclusion_path", "")))
    if (
        conclusion_path is None
        or not conclusion_path.is_file()
        or _sha256_file(conclusion_path) != str(stage4.get("conclusion_sha256", ""))
    ):
        raise ValueError("registered Stage 4 conclusion binding is invalid")
    conclusion = _read_json(conclusion_path)
    outcomes = conclusion.get("outcomes")
    candidates = contract.get("registered_candidates")
    if (
        conclusion.get("contract_id") != contract_id
        or conclusion.get("conclusion_status") != "ACCEPTED_REGISTERED_SURVIVORS"
        or not isinstance(outcomes, dict)
        or not isinstance(candidates, list)
    ):
        raise ValueError("registered Stage 4 accepted conclusion is invalid")

    candidate_by_semantic: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise TypeError("registered Stage 4 candidate identity is malformed")
        semantic_id = str(candidate.get("semantic_hypothesis_id", "")).strip()
        experiment_id = str(candidate.get("source_experiment_id", "")).strip()
        if not semantic_id or not experiment_id or semantic_id in candidate_by_semantic:
            raise ValueError("registered Stage 4 candidate identity is not unique")
        candidate_by_semantic[semantic_id] = candidate
    if set(outcomes) != set(candidate_by_semantic):
        raise ValueError("registered Stage 4 conclusion candidate coverage mismatch")

    accepted_identities: dict[str, dict[str, Any]] = {}
    for semantic_id, outcome in outcomes.items():
        if outcome != "ACCEPTED_SURVIVOR":
            continue
        candidate = candidate_by_semantic[semantic_id]
        experiment_id = str(candidate.get("source_experiment_id", "")).strip()
        if experiment_id in accepted_identities:
            raise ValueError("registered Stage 4 accepted experiment identity is duplicated")
        accepted_identities[experiment_id] = {
            "semantic_hypothesis_id": semantic_id,
            "experiment_id": experiment_id,
            "pair_group_key": str(candidate.get("pair_group_key", "")).strip(),
            "pair": str(candidate.get("pair", "")).strip(),
            "exact_mode": str(candidate.get("exact_mode", "")).strip(),
            "orientation": str(candidate.get("orientation", "")).strip(),
        }
    final_ids = [str(value).strip() for value in survivor.get("final_experiment_ids", [])]
    stage4_final_ids = [
        str(value).strip() for value in stage4.get("stage4_final_experiment_ids", [])
    ]
    accepted_ids = set(accepted_identities)
    if (
        not accepted_ids
        or len(final_ids) != len(set(final_ids))
        or set(final_ids) != accepted_ids
        or stage4_final_ids != final_ids
        or int(survivor.get("final_one_x_survivors", -1)) != len(accepted_ids)
        or int(stage4.get("stage4_final_one_x_survivors", -1)) != len(accepted_ids)
        or int(stage4.get("accepted_registered_hypotheses", -1)) != len(accepted_ids)
        or int(conclusion.get("accepted_registered_hypotheses", -1)) != len(accepted_ids)
    ):
        raise ValueError("Stage 4 survivor IDs are not the exact accepted registered cohort")

    source_path, source_receipt_path = resolve_registered_source_family(
        root=root,
        contract=contract,
        current_matrix_path=current_matrix_path,
    )
    source = _read_csv(source_path)
    if (
        source.empty
        or "experiment_id" not in source
        or source["experiment_id"].astype(str).duplicated().any()
    ):
        raise ValueError("registered Stage 4 source-family identity is invalid")
    for experiment_id, identity in accepted_identities.items():
        selected = source.loc[source["experiment_id"].astype(str).eq(experiment_id)]
        if len(selected) != 1:
            raise ValueError(
                "accepted Stage 4 experiment is absent from its immutable source family"
            )
        row = selected.iloc[0].to_dict()
        for field in (
            "experiment_id",
            "pair_group_key",
            "pair",
            "exact_mode",
            "orientation",
        ):
            if str(row.get(field, "")).strip() != str(identity.get(field, "")).strip():
                raise ValueError(
                    "accepted Stage 4 candidate identity differs from its immutable "
                    f"source-family row: {field}"
                )
    return (
        source,
        accepted_identities,
        source_path,
        source_receipt_path,
        contract_path,
    )


def _validated_registered_learning(
    *, root: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path | None, str]:
    """Revalidate immutable Stage 5 evidence instead of trusting its active summary."""

    status_path = root / "reports" / "active" / "registered_learning_research_status.json"
    status = _read_json(status_path)
    try:
        if (
            status.get("status")
            not in {
                "PASS_RESEARCH_LEARNING_GATES",
                "ALREADY_COMPLETE",
            }
            or status.get("stage5_research_gate_pass") is not True
        ):
            raise ValueError("registered_learning_stage5_pass_missing")
        receipt_path = _safe_repo_artifact(root, str(status.get("learning_receipt_path", "")))
        expected_parent = (root / "data" / "research" / "registered_learning").resolve()
        if receipt_path is None or not receipt_path.is_file():
            raise ValueError("registered_learning_immutable_receipt_missing")
        receipt_path.resolve().relative_to(expected_parent)
        if _sha256_file(receipt_path) != str(status.get("learning_receipt_sha256", "")):
            raise ValueError("registered_learning_active_pointer_hash_mismatch")
        receipt = _read_json(receipt_path)
        roles = receipt.get("artifact_roles")
        if not isinstance(roles, dict):
            raise TypeError("registered_learning_artifact_roles_missing")
        stage4_path = _safe_repo_artifact(root, str(roles.get("stage4_execution_receipt", "")))
        if stage4_path is None or not stage4_path.is_file():
            raise ValueError("registered_learning_stage4_receipt_missing")
        from quant_platform.orchestration.corrective_registered_learning import (
            _registered_execution,
            _validate_learning_receipt,
        )

        _, stage4, _ = _registered_execution(
            root=root,
            execution_receipt_path=stage4_path,
        )
        validated = _validate_learning_receipt(receipt_path, root=root, stage4=stage4)
        if validated.get("learning_id") and status.get("registered_execution_id") != validated.get(
            "registered_execution_id"
        ):
            raise ValueError("registered_learning_active_pointer_lineage_mismatch")
        model_path = _safe_repo_artifact(
            root, str(validated["artifact_roles"].get("model_authority", ""))
        )
        if model_path is None or not model_path.is_file():
            raise ValueError("registered_learning_model_authority_missing")
        return validated, _read_json(model_path), stage4, receipt_path, ""
    except (KeyError, OSError, TypeError, ValueError) as exc:
        return {}, {}, {}, None, f"registered_learning_validation_failed:{exc}"


def _candidate_identity_core(receipt: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "candidate_experiment_id",
        "registered_semantic_hypothesis_id",
        "registered_stage4_contract_id",
        "registered_stage4_conclusion_sha256",
        "immutable_stage4_candidate_identity_ready",
        "candidate_leverage",
        "pair_group_key",
        "pair",
        "asset_x",
        "asset_y",
        "timeframe",
        "exact_mode",
        "orientation",
        "cost_model_id",
        "model_training_dataset_id",
        "model_artifact_sha256",
        "registered_learning_id",
        "registered_learning_receipt_path",
        "registered_learning_receipt_sha256",
        "registered_stage5_protocol_id",
        "registered_stage5_protocol_sha256",
        "registered_execution_id",
        "survivor_receipt_id",
        "candidate_queue_id",
        "candidate_queue_path",
        "candidate_queue_sha256",
        "candidate_queue_selection_rank",
        "validated_current_model_candidate_samples",
        "validated_current_model_pair_samples",
        "ready_candidate_pairs",
        "required_candidate_pairs",
    )
    hashes = receipt.get("source_artifact_hashes")
    return {
        "schema_version": CANDIDATE_IDENTITY_SCHEMA_VERSION,
        **{field: receipt.get(field) for field in fields},
        "source_artifact_hashes": dict(sorted(hashes.items())) if isinstance(hashes, dict) else {},
        "order_submission_performed": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }


def _seal_testnet_candidate_receipt(*, receipt: dict[str, Any], root: Path) -> dict[str, Any]:
    sealed = dict(receipt)
    identity = _candidate_identity_core(sealed)
    receipt_id = "testnetcandidate_" + _payload_hash(identity)[:20]
    identity["candidate_receipt_id"] = receipt_id
    identity["receipt_sha256"] = _payload_hash(identity)
    identity_path = root / "data" / "testnet" / "candidate_receipts" / f"{receipt_id}.json"
    _write_immutable_json(identity, identity_path)
    existing_path = root / "reports" / "active" / "testnet_candidate_receipt.json"
    existing = _read_json(existing_path)
    if (
        existing.get("candidate_status") == "READY_FOR_NO_ORDER_PREFLIGHT"
        and existing.get("candidate_receipt_id") == receipt_id
        and _candidate_identity_core(existing) == _candidate_identity_core(sealed)
    ):
        existing_valid, _existing_blockers = _validated_testnet_candidate_receipt(
            root=root,
            candidate=existing,
            identity_path=identity_path,
        )
        if existing_valid:
            return existing
    sealed.update(
        {
            "candidate_receipt_id": receipt_id,
            "receipt_id": receipt_id,
            "candidate_identity_path": _relative(identity_path, root),
            "candidate_identity_sha256": _sha256_file(identity_path),
        }
    )
    sealed["receipt_sha256"] = _payload_hash(sealed)
    return sealed


def _validated_testnet_candidate_receipt(
    *,
    root: Path,
    candidate: dict[str, Any] | None = None,
    identity_path: Path | None = None,
    queue_path: Path | None = None,
    verify_current_source_artifacts: bool = True,
) -> tuple[bool, list[str]]:
    candidate = candidate or _read_json(
        root / "reports" / "active" / "testnet_candidate_receipt.json"
    )
    blockers: list[str] = []
    receipt_id = str(candidate.get("candidate_receipt_id", "")).strip()
    supplied_receipt_hash = str(candidate.get("receipt_sha256", "")).strip()
    if (
        candidate.get("schema_version") != CANDIDATE_SCHEMA_VERSION
        or candidate.get("candidate_status") != "READY_FOR_NO_ORDER_PREFLIGHT"
        or not receipt_id.startswith("testnetcandidate_")
        or supplied_receipt_hash != _payload_hash(candidate)
    ):
        blockers.append("testnet_candidate_active_receipt_invalid")
    expected_relative = f"data/testnet/candidate_receipts/{receipt_id}.json"
    if identity_path is None:
        if candidate.get("candidate_identity_path") != expected_relative:
            blockers.append("testnet_candidate_identity_path_invalid")
        identity_path = _safe_repo_artifact(root, str(candidate.get("candidate_identity_path", "")))
    identity = _read_json(identity_path) if identity_path else {}
    if (
        identity_path is None
        or not identity_path.is_file()
        or _sha256_file(identity_path) != str(candidate.get("candidate_identity_sha256", ""))
    ):
        blockers.append("testnet_candidate_identity_artifact_invalid")
    identity_hash = str(identity.get("receipt_sha256", ""))
    identity_core = _candidate_identity_core(candidate)
    expected_id = "testnetcandidate_" + _payload_hash(identity_core)[:20]
    expected_identity = {
        **identity_core,
        "candidate_receipt_id": expected_id,
    }
    expected_identity["receipt_sha256"] = _payload_hash(expected_identity)
    if (
        identity != expected_identity
        or receipt_id != expected_id
        or identity_hash != _payload_hash(identity)
    ):
        blockers.append("testnet_candidate_identity_content_invalid")
    blockers.extend(
        _validate_testnet_candidate_queue_binding(
            root=root,
            candidate=candidate,
            queue_path=queue_path,
            verify_current_source_artifacts=verify_current_source_artifacts,
        )
    )
    return not blockers, list(dict.fromkeys(blockers))


def _validate_testnet_candidate_queue_binding(
    *,
    root: Path,
    candidate: dict[str, Any],
    queue_path: Path | None = None,
    verify_current_source_artifacts: bool = True,
) -> list[str]:
    blockers: list[str] = []
    queue_id = str(candidate.get("candidate_queue_id", "")).strip()
    expected_relative = f"data/testnet/candidate_queues/{queue_id}.json"
    if (
        not queue_id.startswith("testnetcandidatequeue_")
        or candidate.get("candidate_queue_path") != expected_relative
    ):
        blockers.append("testnet_candidate_queue_path_invalid")
    if queue_path is None:
        queue_path = _safe_repo_artifact(root, str(candidate.get("candidate_queue_path", "")))
    queue = _read_json(queue_path) if queue_path else {}
    queue_sha256 = str(candidate.get("candidate_queue_sha256", "")).strip()
    if (
        queue_path is None
        or not queue_path.is_file()
        or not queue_sha256
        or _sha256_file(queue_path) != queue_sha256
    ):
        blockers.append("testnet_candidate_queue_artifact_invalid")

    queue_core = {
        key: value
        for key, value in queue.items()
        if key not in {"candidate_queue_id", "receipt_sha256"}
    }
    expected_queue_id = (
        "testnetcandidatequeue_" + _payload_hash(queue_core)[:20] if queue_core else ""
    )
    if (
        queue.get("schema_version") != QUEUE_RECEIPT_SCHEMA_VERSION
        or str(queue.get("receipt_sha256", "")) != _payload_hash(queue)
        or queue_id != expected_queue_id
        or str(queue.get("candidate_queue_id", "")) != expected_queue_id
    ):
        blockers.append("testnet_candidate_queue_receipt_invalid")

    rows = queue.get("rows") if isinstance(queue.get("rows"), list) else []
    ready_rows = [
        row
        for row in rows
        if isinstance(row, dict)
        and row.get("schema_version") == QUEUE_ROW_SCHEMA_VERSION
        and row.get("queue_status") == "READY"
    ]
    selected_rows = [row for row in ready_rows if row.get("selected_active") is True]
    required_pairs = int(_finite(queue.get("required_candidate_pairs"), default=0))
    ready_pairs = {str(row.get("pair", "")).strip() for row in ready_rows} - {""}
    supplied_ready_pairs = int(_finite(queue.get("ready_candidate_pairs"), default=-1))
    stage4_contract_id = str(queue.get("registered_stage4_contract_id", "")).strip()
    stage4_conclusion_sha256 = str(queue.get("registered_stage4_conclusion_sha256", "")).strip()
    immutable_rows_ready = bool(
        ready_rows
        and stage4_contract_id
        and len(stage4_conclusion_sha256) == 64
        and queue.get("immutable_stage4_candidate_identity_ready") is True
        and all(
            row.get("immutable_stage4_candidate_identity_ready") is True
            and str(row.get("registered_stage4_contract_id", "")).strip() == stage4_contract_id
            and bool(str(row.get("registered_semantic_hypothesis_id", "")).strip())
            for row in ready_rows
        )
    )
    policy_ready = bool(
        required_pairs >= 1
        and supplied_ready_pairs == len(ready_pairs)
        and len(ready_pairs) >= required_pairs
        and queue.get("candidate_selection_ready") is True
        and not queue.get("global_blockers")
        and len(selected_rows) == 1
        and immutable_rows_ready
        and bool(str(queue.get("registered_learning_id", "")).strip())
        and bool(str(queue.get("registered_learning_receipt_path", "")).strip())
        and len(str(queue.get("registered_learning_receipt_sha256", "")).strip()) == 64
        and bool(str(queue.get("registered_stage5_protocol_id", "")).strip())
        and len(str(queue.get("registered_stage5_protocol_sha256", "")).strip()) == 64
        and bool(str(queue.get("registered_execution_id", "")).strip())
    )
    if not policy_ready:
        blockers.append("testnet_candidate_queue_policy_or_selection_invalid")

    selected = selected_rows[0] if len(selected_rows) == 1 else {}
    expected_selected = min(
        ready_rows,
        key=lambda row: (
            _finite(
                row.get("validated_current_model_candidate_samples"),
                default=math.inf,
            ),
            _finite(row.get("validated_current_model_pair_samples"), default=math.inf),
            str(row.get("candidate_experiment_id", "")),
        ),
        default={},
    )
    if selected != expected_selected:
        blockers.append("testnet_candidate_queue_deficit_ranking_invalid")

    row_bindings = {
        "candidate_experiment_id": "candidate_experiment_id",
        "registered_semantic_hypothesis_id": ("registered_semantic_hypothesis_id"),
        "registered_stage4_contract_id": "registered_stage4_contract_id",
        "immutable_stage4_candidate_identity_ready": ("immutable_stage4_candidate_identity_ready"),
        "pair_group_key": "pair_group_key",
        "pair": "pair",
        "asset_x": "asset_x",
        "asset_y": "asset_y",
        "timeframe": "timeframe",
        "exact_mode": "exact_mode",
        "orientation": "orientation",
        "cost_model_id": "cost_model_id",
        "candidate_queue_selection_rank": "selection_rank",
        "validated_current_model_candidate_samples": ("validated_current_model_candidate_samples"),
        "validated_current_model_pair_samples": "validated_current_model_pair_samples",
    }
    if any(
        candidate.get(field) != selected.get(row_field) for field, row_field in row_bindings.items()
    ):
        blockers.append("testnet_candidate_queue_selected_row_mismatch")
    if (
        candidate.get("ready_candidate_pairs") != supplied_ready_pairs
        or candidate.get("required_candidate_pairs") != required_pairs
        or candidate.get("model_training_dataset_id") != queue.get("model_training_dataset_id")
        or candidate.get("model_artifact_sha256") != queue.get("model_artifact_sha256")
        or candidate.get("survivor_receipt_id") != queue.get("survivor_receipt_id")
        or candidate.get("registered_learning_id") != queue.get("registered_learning_id")
        or candidate.get("registered_learning_receipt_path")
        != queue.get("registered_learning_receipt_path")
        or candidate.get("registered_learning_receipt_sha256")
        != queue.get("registered_learning_receipt_sha256")
        or candidate.get("registered_stage5_protocol_id")
        != queue.get("registered_stage5_protocol_id")
        or candidate.get("registered_stage5_protocol_sha256")
        != queue.get("registered_stage5_protocol_sha256")
        or candidate.get("registered_execution_id") != queue.get("registered_execution_id")
        or candidate.get("registered_stage4_contract_id")
        != queue.get("registered_stage4_contract_id")
        or candidate.get("registered_stage4_conclusion_sha256")
        != queue.get("registered_stage4_conclusion_sha256")
        or candidate.get("immutable_stage4_candidate_identity_ready") is not True
        or queue.get("immutable_stage4_candidate_identity_ready") is not True
    ):
        blockers.append("testnet_candidate_queue_lineage_mismatch")
    queue_hashes = queue.get("source_artifact_hashes")
    candidate_hashes = candidate.get("source_artifact_hashes")
    if not isinstance(queue_hashes, dict) or not isinstance(candidate_hashes, dict):
        blockers.append("testnet_candidate_queue_source_hashes_invalid")
    else:
        if not queue_hashes:
            blockers.append("testnet_candidate_queue_source_hashes_empty")
        learning_relative = str(queue.get("registered_learning_receipt_path", "")).strip()
        learning_sha256 = str(queue.get("registered_learning_receipt_sha256", "")).strip()
        if (
            not learning_relative
            or len(learning_sha256) != 64
            or queue_hashes.get(learning_relative) != learning_sha256
        ):
            blockers.append("testnet_candidate_queue_learning_receipt_binding_invalid")
        for relative, digest in queue_hashes.items():
            source_path = _safe_repo_artifact(root, str(relative))
            if (
                source_path is None
                or len(str(digest)) != 64
                or (
                    verify_current_source_artifacts
                    and (not source_path.is_file() or _sha256_file(source_path) != str(digest))
                )
            ):
                blockers.append(f"testnet_candidate_queue_source_artifact_changed:{relative}")
        queue_relative = str(candidate.get("candidate_queue_path", ""))
        if (
            any(candidate_hashes.get(path) != digest for path, digest in queue_hashes.items())
            or candidate_hashes.get(queue_relative) != queue_sha256
        ):
            blockers.append("testnet_candidate_queue_source_hashes_mismatch")
    if (
        queue.get("order_submission_performed") is not False
        or queue.get("testnet_order_authority") is not False
        or queue.get("live_trading_authorized") is not False
        or selected.get("order_submission_performed") is not False
        or selected.get("testnet_order_authority") is not False
        or selected.get("live_trading_authorized") is not False
    ):
        blockers.append("testnet_candidate_queue_authority_invalid")
    return list(dict.fromkeys(blockers))


def _existing_artifact_hashes(*, root: Path, paths: tuple[Path | None, ...]) -> dict[str, str]:
    return {
        _relative(path, root): _sha256_file(path)
        for path in paths
        if path is not None and path.is_file()
    }


def _validated_pair_cost_bundle(*, root: Path) -> tuple[Path, Path | None, list[str]]:
    """Resolve the latest cost model to immutable bundle artifacts."""

    pointer_path = root / "reports" / "active" / PAIR_COST_BUNDLE_POINTER_NAME
    pointer = _read_json(pointer_path)
    blockers: list[str] = []
    if pointer.get("schema_version") != PAIR_COST_BUNDLE_POINTER_SCHEMA_VERSION:
        blockers.append("pair_cost_bundle_pointer_schema_invalid")
    if pointer.get("receipt_sha256") != _payload_hash(pointer):
        blockers.append("pair_cost_bundle_pointer_receipt_invalid")
    bundle_id = str(pointer.get("bundle_id", "")).strip()
    expected_root = f"data/research/l2_cost_model_receipts/{bundle_id}"
    expected_manifest = f"{expected_root}/receipt.json"
    expected_models = f"{expected_root}/pair_cost_models.csv"
    if not bundle_id.startswith("l2costbundle_"):
        blockers.append("pair_cost_bundle_id_invalid")
    if pointer.get("bundle_manifest_path") != expected_manifest:
        blockers.append("pair_cost_bundle_manifest_path_invalid")
    if pointer.get("pair_cost_models_path") != expected_models:
        blockers.append("pair_cost_bundle_model_path_invalid")

    manifest_path = _safe_repo_artifact(root, str(pointer.get("bundle_manifest_path", "")))
    model_path = _safe_repo_artifact(root, str(pointer.get("pair_cost_models_path", "")))
    if manifest_path is None or not manifest_path.is_file():
        blockers.append("pair_cost_bundle_manifest_missing")
    elif _sha256_file(manifest_path) != str(pointer.get("bundle_manifest_sha256", "")):
        blockers.append("pair_cost_bundle_manifest_hash_mismatch")
    if model_path is None or not model_path.is_file():
        blockers.append("pair_cost_bundle_model_missing")
    elif _sha256_file(model_path) != str(pointer.get("pair_cost_models_sha256", "")):
        blockers.append("pair_cost_bundle_model_hash_mismatch")

    manifest = _read_json(manifest_path) if manifest_path is not None else {}
    if (
        manifest.get("schema_version") != "thewiz.l2_cost_model_receipt.v1"
        or manifest.get("bundle_id") != bundle_id
        or manifest.get("pair_cost_models_path") != expected_models
        or manifest.get("pair_cost_models_sha256") != pointer.get("pair_cost_models_sha256")
    ):
        blockers.append("pair_cost_bundle_manifest_content_invalid")
    if manifest_path is not None and manifest_path.is_file():
        blockers.extend(
            validate_pair_cost_bundle_artifacts(
                root=root,
                bundle_manifest_path=manifest_path,
                pair_cost_models_path=model_path,
            )
        )
    if any(
        payload.get(field) is not False
        for payload in (pointer, manifest)
        for field in (
            "promotion_authority",
            "testnet_order_authority",
            "live_trading_authorized",
        )
    ):
        blockers.append("pair_cost_bundle_authority_invalid")

    missing = root / "reports" / "active" / ".missing_pair_cost_bundle.csv"
    return model_path or missing, manifest_path, list(dict.fromkeys(blockers))


def _rebuild_validated_daily_cadence(
    *, root: Path, now: datetime
) -> tuple[pd.DataFrame, Path, str]:
    """Derive release cadence from immutable receipts and fail closed on drift."""

    expected_path = root / "reports" / "active" / "daily_cadence_acceptance.csv"
    try:
        rebuilt_path = build_daily_cadence_acceptance(root=root, now=now)
        if rebuilt_path.resolve() != expected_path.resolve():
            return (
                pd.DataFrame(),
                expected_path,
                "daily_cadence_acceptance_path_invalid",
            )
        frame = _read_csv(expected_path)
    except (OSError, TypeError, ValueError, pd.errors.ParserError) as exc:
        return (
            pd.DataFrame(),
            expected_path,
            f"daily_cadence_acceptance_rebuild_failed:{type(exc).__name__}",
        )

    required_columns = {
        "run_date",
        "receipt_valid",
        "semantic_contract_status",
        "qualifying_cycle",
        "consecutive_complete_cycles",
        "required_cycles",
        "cadence_acceptance_status",
        "zero_execution_authority",
        "live_trading_authorized",
    }
    if not required_columns.issubset(frame.columns):
        return frame, expected_path, "daily_cadence_acceptance_contract_invalid"
    return frame, expected_path, ""


def _daily_cadence_pass(frame: pd.DataFrame) -> bool:
    if frame.empty:
        return False
    consecutive = pd.to_numeric(frame["consecutive_complete_cycles"], errors="coerce")
    required = pd.to_numeric(frame["required_cycles"], errors="coerce")
    return bool(
        (
            frame["cadence_acceptance_status"].eq("PASS")
            & frame["receipt_valid"].map(_truthy)
            & frame["semantic_contract_status"].eq("PASS")
            & frame["qualifying_cycle"].map(_truthy)
            & frame["zero_execution_authority"].map(_truthy)
            & ~frame["live_trading_authorized"].map(_truthy)
            & required.eq(7)
            & consecutive.ge(required)
        ).any()
    )


def _safe_repo_artifact(root: Path, raw: str) -> Path | None:
    if not raw.strip():
        return None
    candidate = Path(raw)
    path = candidate if candidate.is_absolute() else root / candidate
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return None
    return path


def _payload_hash(payload: dict[str, Any]) -> str:
    core = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    return sha256(_canonical_json(core).encode("utf-8")).hexdigest()


def _write_immutable_json(payload: dict[str, Any], path: Path) -> None:
    existing = _read_json(path)
    if existing and existing != payload:
        raise ValueError(f"immutable artifact conflict: {path}")
    _atomic_json(payload, path)


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _truthy(value: Any) -> bool:
    return value is True or str(value).strip().lower() in {"1", "true", "yes", "pass", "ready"}


def _finite(value: Any, *, default: float = math.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _as_utc(value: datetime | None) -> datetime:
    value = value or datetime.now(UTC)
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


if __name__ == "__main__":
    result = build_corrective_release_gates()
    print(
        json.dumps(
            {
                "summary": result.summary,
                "paths": {key: str(value) for key, value in result.paths.items()},
            },
            indent=2,
        )
    )
