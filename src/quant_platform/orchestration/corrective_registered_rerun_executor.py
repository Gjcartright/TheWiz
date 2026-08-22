"""Execute one frozen, registered research rerun after all evidence gates pass."""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd

from quant_platform.active_pipeline import CommandResult
from quant_platform.orchestration.corrective_daily_scheduler import _acquire_lock
from quant_platform.orchestration.corrective_data_evidence import (
    validate_pair_cost_bundle_artifacts,
)
from quant_platform.orchestration.corrective_redaction import safe_exception_code
from quant_platform.orchestration.corrective_registered_rerun import (
    CONCLUSION_SCHEMA_VERSION,
    _validate_ready_receipt_lineage,
    build_registered_rerun_gate,
    resolve_registered_source_family,
    validate_registered_rerun_contract_identity,
)
from quant_platform.orchestration.corrective_runtime import (
    atomic_write_bytes,
    promote_staged_directory,
    promote_staged_file,
)
from quant_platform.orchestration.corrective_statistical_remediation import (
    build_corrective_statistical_remediation,
)
from quant_platform.orchestration.current_wizard_hyperliquid_concentration import (
    build_current_wizard_hyperliquid_concentration,
)
from quant_platform.orchestration.current_wizard_hyperliquid_costs import (
    materialize_current_wizard_hyperliquid_cost_evidence,
)
from quant_platform.orchestration.current_wizard_hyperliquid_failure_attribution import (
    build_current_wizard_hyperliquid_failure_attribution,
)
from quant_platform.orchestration.current_wizard_hyperliquid_learning import (
    build_current_wizard_hyperliquid_learning_ledger,
)
from quant_platform.orchestration.current_wizard_hyperliquid_leverage import (
    build_current_wizard_hyperliquid_leverage_surface,
)
from quant_platform.orchestration.current_wizard_hyperliquid_observed_replay import (
    run_current_wizard_hyperliquid_observed_cost_replay,
)
from quant_platform.orchestration.current_wizard_hyperliquid_regimes import (
    build_current_wizard_hyperliquid_regime_attribution,
)
from quant_platform.orchestration.current_wizard_hyperliquid_replay import (
    run_current_wizard_hyperliquid_canonical_replay,
)
from quant_platform.orchestration.current_wizard_hyperliquid_robustness import (
    run_current_wizard_hyperliquid_robustness,
)
from quant_platform.orchestration.current_wizard_hyperliquid_validation import (
    validate_current_wizard_hyperliquid_chain,
)
from quant_platform.orchestration.current_wizard_hyperliquid_walkforward import (
    run_current_wizard_hyperliquid_walkforward,
)
from quant_platform.orchestration.current_wizard_ou_optimal_overlay import (
    build_current_wizard_ou_optimal_overlay,
)

ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "thewiz.corrective_registered_rerun_execution.v1"
LEARNING_STATUS_SCHEMA_VERSION = "thewiz.corrective_registered_learning.v1"
LOCK_TIMEOUT_SECONDS = 4 * 60 * 60

StageRunner = Callable[..., CommandResult]
GateBuilder = Callable[..., CommandResult]
LearningRunner = Callable[..., CommandResult]

DEFAULT_STAGE_RUNNERS: dict[str, StageRunner] = {
    "canonical_replay": run_current_wizard_hyperliquid_canonical_replay,
    "cost_evidence": materialize_current_wizard_hyperliquid_cost_evidence,
    "observed_cost_replay": run_current_wizard_hyperliquid_observed_cost_replay,
    "walkforward": run_current_wizard_hyperliquid_walkforward,
    "ou_optimal_overlay": build_current_wizard_ou_optimal_overlay,
    "regime_attribution": build_current_wizard_hyperliquid_regime_attribution,
    "robustness": run_current_wizard_hyperliquid_robustness,
    "concentration": build_current_wizard_hyperliquid_concentration,
    "failure_attribution": build_current_wizard_hyperliquid_failure_attribution,
    "leverage_surface": build_current_wizard_hyperliquid_leverage_surface,
    "learning_ledger": build_current_wizard_hyperliquid_learning_ledger,
    "chain_validation": validate_current_wizard_hyperliquid_chain,
}

FAMILY_ACCOUNTING_STAGES = {
    "canonical_replay",
    "cost_evidence",
    "observed_cost_replay",
    "walkforward",
    "regime_attribution",
    "robustness",
    "concentration",
    "failure_attribution",
    "leverage_surface",
    "learning_ledger",
    "chain_validation",
}


def run_registered_research_rerun(
    *,
    root: Path = ROOT,
    now: datetime | None = None,
    execute: bool = False,
    fetch_funding: bool = False,
    gate_builder: GateBuilder = build_registered_rerun_gate,
    stage_runners: Mapping[str, StageRunner] | None = None,
    reconciler: GateBuilder = build_corrective_statistical_remediation,
    checkpoint_refresher: GateBuilder | None = None,
    learning_runner: LearningRunner | None = None,
) -> CommandResult:
    """Run the frozen downstream family once; never capture Wizard data or place orders."""

    requested_at = _as_utc(now)
    active = root / "reports" / "active"
    active.mkdir(parents=True, exist_ok=True)
    status_path = active / "registered_research_rerun_execution.json"
    lock_paths = (
        active / ".corrective_daily.lock",
        active / ".corrective_l2_capture.lock",
        active / ".corrective_registered_rerun.lock",
    )
    acquired: list[Path] = []
    result: CommandResult | None = None
    refresh_after_unlock = False
    learning_receipt_after_unlock: Path | None = None
    funding_audit: dict[str, Any] = {
        "funding_refresh_requested": bool(fetch_funding),
        "funding_cost_evidence_stage_run": False,
        "funding_refresh_dispatched_to_cost_stage": False,
        "funding_assets_fetched_from_network": 0,
        "funding_assets_reused_from_cache": 0,
        "funding_refresh_outcome": "NOT_REACHED",
    }
    try:
        for lock_path in lock_paths:
            _acquire_lock(
                lock_path,
                now=requested_at,
                timeout_seconds=LOCK_TIMEOUT_SECONDS,
            )
            acquired.append(lock_path)

        gate = gate_builder(root=root, now=requested_at)
        contract_path = Path(
            gate.paths.get("contract", active / "registered_research_rerun_contract.json")
        )
        contract = _validate_contract(root=root, contract_path=contract_path)
        contract_id = _required_text(contract.get("contract_id"), "contract_id")
        receipt_path = (
            root / "data" / "research" / "registered_rerun_executions" / f"{contract_id}.json"
        )
        if receipt_path.is_file():
            receipt = _validate_execution_receipt(
                receipt_path=receipt_path,
                contract=contract,
                root=root,
            )
            result = _publish_result(
                root=root,
                status_path=status_path,
                status="ALREADY_COMPLETE",
                contract=contract,
                execute=execute,
                receipt_path=receipt_path,
                receipt=receipt,
                blocker="",
                funding_audit=funding_audit,
            )
            refresh_after_unlock = True
            if execute and (stage_runners is None or learning_runner is not None):
                learning_receipt_after_unlock = receipt_path
        else:
            gate_status = str(gate.summary.get("status", "BLOCKED"))
            if gate_status != "READY_FOR_NEXT_SCHEDULED_RESEARCH_RERUN":
                result = _publish_result(
                    root=root,
                    status_path=status_path,
                    status=(
                        "ALREADY_ACCOUNTED_EXTERNAL_CHAIN"
                        if bool(gate.summary.get("registered_rerun_results_accounted"))
                        else "BLOCKED_GATE"
                    ),
                    contract=contract,
                    execute=execute,
                    blocker=f"registered_rerun_gate_status:{gate_status}",
                    funding_audit=funding_audit,
                )
                _publish_stage5_blocked_status(
                    root=root,
                    result=result,
                    requested_at=requested_at,
                    execute=execute,
                    status="BLOCKED_STAGE4",
                    blocker=f"registered_rerun_gate_status:{gate_status}",
                    contract=contract,
                )
            else:
                ready_path = Path(
                    gate.paths.get(
                        "ready_receipt",
                        root
                        / "data"
                        / "research"
                        / "registered_rerun_ready"
                        / f"{contract_id}.json",
                    )
                )
                ready = _validate_ready_receipt(
                    ready_path=ready_path,
                    contract=contract,
                    root=root,
                )
                gate_frame = _read_csv(
                    Path(gate.paths.get("gate", active / "registered_research_rerun_gate.csv"))
                )
                pair_keys = _registered_pair_keys(gate_frame, contract)
                execution_id = _execution_id(
                    root=root,
                    contract=contract,
                    ready_path=ready_path,
                    gate_path=Path(gate.paths["gate"]),
                    pair_keys=pair_keys,
                )
                if not execute:
                    result = _publish_result(
                        root=root,
                        status_path=status_path,
                        status="PLANNED",
                        contract=contract,
                        execute=False,
                        blocker="execution_not_requested",
                        extra={
                            "execution_id": execution_id,
                            "registered_pair_group_keys": list(pair_keys),
                        },
                        funding_audit=funding_audit,
                    )
                else:
                    ready_at = pd.to_datetime(
                        ready.get("evidence_ready_at_utc"), utc=True, errors="coerce"
                    )
                    if pd.isna(ready_at):
                        raise ValueError("registered rerun ready timestamp is invalid")
                    run_at = max(
                        requested_at,
                        ready_at.to_pydatetime() + timedelta(microseconds=1),
                    )
                    runners = dict(DEFAULT_STAGE_RUNNERS)
                    if stage_runners is not None:
                        runners.update(stage_runners)
                    missing_runners = set(DEFAULT_STAGE_RUNNERS) - set(runners)
                    if missing_runners:
                        raise ValueError(
                            "registered rerun stage runners missing: "
                            + ",".join(sorted(missing_runners))
                        )
                    execution_root = root
                    workspace_receipt_path: Path | None = None
                    if stage_runners is None:
                        execution_root, workspace_receipt_path = _prepare_registered_workspace(
                            root=root,
                            contract=contract,
                            execution_id=execution_id,
                            ready_path=ready_path,
                            pair_keys=pair_keys,
                        )
                    stage_rows = _run_frozen_chain(
                        root=execution_root,
                        artifact_root=root,
                        run_at=run_at,
                        contract=contract,
                        pair_keys=pair_keys,
                        fetch_funding=fetch_funding,
                        runners=runners,
                        funding_audit=funding_audit,
                    )
                    reconciliation = reconciler(root=execution_root, now=run_at)
                    _assert_no_order_authority("statistical_reconciliation", reconciliation.summary)
                    _validate_source_family(
                        root=execution_root,
                        contract=contract,
                    )
                    conclusion_path = Path(
                        reconciliation.paths.get(
                            "registered_rerun_conclusion_receipt",
                            execution_root
                            / "data"
                            / "research"
                            / "registered_rerun_conclusions"
                            / f"{contract_id}.json",
                        )
                    )
                    conclusion = _validate_conclusion(
                        path=conclusion_path,
                        contract=contract,
                        summary=reconciliation.summary,
                        root=execution_root,
                    )
                    stage4_survivor_path = (
                        execution_root / "reports" / "active" / "final_1x_survivor_receipt.json"
                    )
                    stage4_survivor = _validate_stage4_survivor_receipt(
                        path=stage4_survivor_path,
                        root=execution_root,
                        contract=contract,
                        conclusion=conclusion,
                    )
                    if execution_root != root:
                        canonical_conclusion_path = (
                            root
                            / "data"
                            / "research"
                            / "registered_rerun_conclusions"
                            / f"{contract_id}.json"
                        )
                        _write_immutable_json(
                            conclusion,
                            canonical_conclusion_path,
                        )
                        conclusion_path = canonical_conclusion_path
                    receipt = {
                        "schema_version": SCHEMA_VERSION,
                        "execution_id": execution_id,
                        "contract_id": contract_id,
                        "started_at_utc": run_at.isoformat(),
                        "completed_at_utc": datetime.now(UTC).isoformat(),
                        "status": "PASS_REGISTERED_RERUN_ACCOUNTED",
                        "source_family_sha256": contract["source_family_sha256"],
                        "source_family_rows": int(contract["source_family_rows"]),
                        "acceptance_policy_id": contract["acceptance_policy_id"],
                        "holdout_policy_id": contract["holdout_policy_id"],
                        "discovery_policy_sha256": contract["discovery_policy_sha256"],
                        "ready_receipt_path": _relative(ready_path, root),
                        "ready_receipt_sha256": _file_hash(ready_path),
                        "registered_pair_group_keys": list(pair_keys),
                        "isolated_workspace_used": execution_root != root,
                        "isolated_workspace_path": (
                            _relative(execution_root, root) if execution_root != root else ""
                        ),
                        "workspace_receipt_path": (
                            _relative(workspace_receipt_path, root)
                            if workspace_receipt_path is not None
                            else ""
                        ),
                        "workspace_receipt_sha256": (
                            _file_hash(workspace_receipt_path)
                            if workspace_receipt_path is not None
                            else ""
                        ),
                        "stages": stage_rows,
                        "conclusion_path": _relative(conclusion_path, root),
                        "conclusion_sha256": _file_hash(conclusion_path),
                        "stage4_survivor_receipt_path": _relative(stage4_survivor_path, root),
                        "stage4_survivor_receipt_sha256": _file_hash(stage4_survivor_path),
                        "stage4_independent_supporting_clusters": int(
                            stage4_survivor.get("independent_supporting_clusters", 0) or 0
                        ),
                        "stage4_independent_full_survivor_clusters": int(
                            stage4_survivor.get("independent_full_survivor_clusters", 0) or 0
                        ),
                        "stage4_independent_supporting_pairs": int(
                            stage4_survivor.get("independent_supporting_pairs", 0) or 0
                        ),
                        "stage4_independent_full_survivor_pairs": int(
                            stage4_survivor.get("independent_full_survivor_pairs", 0) or 0
                        ),
                        "stage4_final_one_x_survivors": int(
                            stage4_survivor.get("final_one_x_survivors", 0) or 0
                        ),
                        "stage4_final_experiment_ids": list(
                            stage4_survivor.get("final_experiment_ids", [])
                        ),
                        "stage4_final_canonical_pairs": list(
                            stage4_survivor.get("final_canonical_pairs", [])
                        ),
                        "conclusion_status": conclusion["conclusion_status"],
                        "accepted_registered_hypotheses": int(
                            conclusion.get("accepted_registered_hypotheses", 0)
                        ),
                        "rejected_registered_hypotheses": int(
                            conclusion.get("rejected_registered_hypotheses", 0)
                        ),
                        "frozen_family_replayed": True,
                        "thresholds_changed_after_contract": False,
                        **funding_audit,
                        "authenticated_wizard_capture_performed": False,
                        "order_submission_performed": False,
                        "promotion_authority": False,
                        "testnet_order_authority": False,
                        "live_trading_authorized": False,
                    }
                    receipt["receipt_sha256"] = _payload_hash(receipt)
                    _write_immutable_json(receipt, receipt_path)
                    result = _publish_result(
                        root=root,
                        status_path=status_path,
                        status="PASS_REGISTERED_RERUN_ACCOUNTED",
                        contract=contract,
                        execute=True,
                        receipt_path=receipt_path,
                        receipt=receipt,
                        blocker="",
                        funding_audit=funding_audit,
                    )
                    refresh_after_unlock = True
                    if stage_runners is None or learning_runner is not None:
                        learning_receipt_after_unlock = receipt_path
    except FileExistsError as exc:
        result = _publish_result(
            root=root,
            status_path=status_path,
            status="BLOCKED_LOCK",
            contract={},
            execute=execute,
            blocker=safe_exception_code(exc),
            funding_audit=funding_audit,
        )
    except Exception as exc:
        _publish_result(
            root=root,
            status_path=status_path,
            status="FAILED",
            contract={},
            execute=execute,
            blocker=f"{safe_exception_code(exc)}",
            funding_audit=funding_audit,
        )
        raise
    finally:
        for lock_path in reversed(acquired):
            lock_path.unlink(missing_ok=True)

    if result is None:
        raise RuntimeError("registered rerun did not produce a result")
    refresher = checkpoint_refresher
    checkpoint_stage4: dict[str, str] = {}
    if refresh_after_unlock:
        if refresher is None:
            from quant_platform.orchestration.corrective_program import (
                complete_corrective_plan,
            )

            refresher = complete_corrective_plan
        try:
            checkpoint = refresher(root=root, now=requested_at)
            _assert_no_order_authority("corrective_checkpoint", checkpoint.summary)
            result.summary["checkpoint_refresh_status"] = str(
                checkpoint.summary.get("operational_acceptance_status", "BLOCKED")
            )
            result.paths["seven_stage_checkpoint"] = Path(
                checkpoint.paths.get(
                    "seven_stage_checkpoint",
                    root / "reports" / "active" / "seven_stage_goal_checkpoint.csv",
                )
            )
            checkpoint_stage4 = _stage4_checkpoint_row(
                root=root,
                path=result.paths["seven_stage_checkpoint"],
            )
        except Exception as exc:  # noqa: BLE001 - immutable replay remains recoverable
            result.summary["checkpoint_refresh_status"] = "REFRESH_FAILED"
            result.summary["checkpoint_refresh_blocker"] = f"{safe_exception_code(exc)}"
    if learning_receipt_after_unlock is not None:
        stage4_receipt = _validate_execution_receipt(
            receipt_path=learning_receipt_after_unlock,
            contract=contract,
            root=root,
        )
        checkpoint_allows_learning = _stage4_checkpoint_allows_learning(
            checkpoint_stage4,
            contract_id=str(stage4_receipt.get("contract_id", "")),
        )
        if _stage4_learning_eligible(stage4_receipt) and checkpoint_allows_learning:
            runner = learning_runner
            if runner is None:
                from quant_platform.orchestration.corrective_registered_learning import (
                    run_registered_learning_research,
                )

                runner = run_registered_learning_research
            try:
                learning = runner(
                    root=root,
                    now=requested_at,
                    execute=True,
                    execution_receipt_path=learning_receipt_after_unlock,
                )
                _assert_no_order_authority("registered_learning_handoff", learning.summary)
                result.summary["learning_handoff_status"] = str(
                    learning.summary.get("status", "NOT_EVALUATED")
                )
                result.summary["stage5_research_gate_pass"] = bool(
                    learning.summary.get("stage5_research_gate_pass", False)
                )
                result.summary["learning_handoff_blocker"] = str(
                    learning.summary.get("blocker", "")
                )
                for name, path in learning.paths.items():
                    result.paths[f"registered_learning_{name}"] = Path(path)
            except Exception as exc:  # noqa: BLE001 - Stage 4 receipt remains recoverable
                result.summary["learning_handoff_status"] = "HANDOFF_FAILED"
                result.summary["stage5_research_gate_pass"] = False
                result.summary["learning_handoff_blocker"] = f"{safe_exception_code(exc)}"
            if refresher is not None:
                try:
                    checkpoint = refresher(root=root, now=requested_at)
                    _assert_no_order_authority(
                        "post_learning_corrective_checkpoint", checkpoint.summary
                    )
                    result.summary["checkpoint_refresh_status"] = str(
                        checkpoint.summary.get("operational_acceptance_status", "BLOCKED")
                    )
                except Exception as exc:  # noqa: BLE001 - learning remains research-only
                    result.summary["checkpoint_refresh_status"] = "REFRESH_FAILED"
                    result.summary["checkpoint_refresh_blocker"] = (
                        f"post_learning:{safe_exception_code(exc)}"
                    )
        else:
            conclusion_zero = bool(
                stage4_receipt.get("conclusion_status") == "CONCLUSIVE_ZERO_REGISTERED_SURVIVORS"
            )
            terminal_rejection = bool(
                "stage4_terminal_outcome=CONCLUSIVE_REJECTION_CURRENT_FAMILY"
                in str(checkpoint_stage4.get("evidence_progress", ""))
            )
            status = (
                "BLOCKED_STAGE4_NO_ACCEPTED_SURVIVOR"
                if conclusion_zero or terminal_rejection
                else "BLOCKED_STAGE4_CURRENT_FAMILY_INCOMPLETE"
            )
            blocker = (
                "registered_stage4_current_family_conclusively_rejected"
                if terminal_rejection
                else "registered_stage4_current_contract_rejected_awaiting_family_rollover"
                if conclusion_zero
                else "registered_stage4_whole_cohort_or_independent_breadth_incomplete"
            )
            _publish_stage5_blocked_status(
                root=root,
                result=result,
                requested_at=requested_at,
                execute=execute,
                status=status,
                blocker=blocker,
                contract=contract,
                receipt=stage4_receipt,
            )
    if refresh_after_unlock or learning_receipt_after_unlock is not None:
        _atomic_json(result.summary, status_path)
    return result


def _run_frozen_chain(
    *,
    root: Path,
    artifact_root: Path | None = None,
    run_at: datetime,
    contract: dict[str, Any],
    pair_keys: tuple[str, ...],
    fetch_funding: bool,
    runners: Mapping[str, StageRunner],
    funding_audit: dict[str, Any],
) -> list[dict[str, Any]]:
    artifact_root = artifact_root or root
    _validate_execution_source_family(root=root, contract=contract)
    rows: list[dict[str, Any]] = []
    for name in DEFAULT_STAGE_RUNNERS:
        runner = runners[name]
        if name == "cost_evidence":
            funding_audit["funding_cost_evidence_stage_run"] = True
            funding_audit["funding_refresh_dispatched_to_cost_stage"] = bool(fetch_funding)
            funding_audit["funding_refresh_outcome"] = (
                "DISPATCHED_AWAITING_RESULT" if fetch_funding else "CACHE_ONLY_AWAITING_RESULT"
            )
            stage = runner(
                root=root,
                now=run_at,
                pair_group_keys=pair_keys,
                fetch_funding=fetch_funding,
            )
        else:
            stage = runner(root=root, now=run_at)
        _assert_no_order_authority(name, stage.summary)
        _validate_source_family(root=root, contract=contract)
        _validate_execution_source_family(root=root, contract=contract)
        row = _validate_stage_result(
            name=name,
            result=stage,
            root=root,
            artifact_root=artifact_root,
            expected_experiments=int(contract["source_family_rows"]),
        )
        if name == "cost_evidence":
            _validate_registered_cost_rows(
                result=stage,
                pair_keys=pair_keys,
            )
            fetched = int(stage.summary.get("funding_assets_fetched_from_network", 0) or 0)
            reused = int(stage.summary.get("funding_assets_reused_from_cache", 0) or 0)
            funding_audit.update(
                {
                    "funding_assets_fetched_from_network": fetched,
                    "funding_assets_reused_from_cache": reused,
                    "funding_refresh_outcome": (
                        "NETWORK_FETCHED"
                        if fetched > 0
                        else "CACHE_REUSED"
                        if reused > 0
                        else "NO_NETWORK_FETCH_NEEDED"
                        if fetch_funding
                        else "FUNDING_REFRESH_DISABLED"
                    ),
                }
            )
            row.update(funding_audit)
        rows.append(row)
    return rows


def _prepare_registered_workspace(
    *,
    root: Path,
    contract: dict[str, Any],
    execution_id: str,
    ready_path: Path,
    pair_keys: tuple[str, ...],
) -> tuple[Path, Path]:
    """Create a private replay root from immutable, contract-bound inputs."""

    frozen_matrix, source_receipt_path = resolve_registered_source_family(
        root=root,
        contract=contract,
        current_matrix_path=(
            root / "reports" / "active" / "current_wizard_hyperliquid_experiment_matrix.csv"
        ),
    )
    source_receipt = _read_json(source_receipt_path)
    recovered_matrix = root / _required_text(
        source_receipt.get("recovered_from_path"),
        "registered source recovered_from_path",
    )
    handoff_dir = recovered_matrix.parent
    handoff_manifest_path = handoff_dir / "manifest.json"
    handoff_manifest = _read_json(handoff_manifest_path)
    if not handoff_manifest or _file_hash(recovered_matrix) != contract.get("source_family_sha256"):
        raise ValueError("registered workspace frozen handoff is unavailable")
    history_manifest_path = _select_registered_history_manifest(
        root=root,
        handoff_dir=handoff_dir,
        pair_keys=pair_keys,
    )
    history_manifest = _read_json(history_manifest_path)
    l2_status_path = root / "reports" / "active" / "corrective_l2_capture_status.json"
    l2_status = _read_json(l2_status_path)
    cost_bundle_receipt_path = root / _required_text(
        l2_status.get("pair_cost_bundle_manifest_path"),
        "pair_cost_bundle_manifest_path",
    )
    cost_bundle_receipt = _read_json(cost_bundle_receipt_path)
    if not cost_bundle_receipt or _file_hash(cost_bundle_receipt_path) != _text(
        l2_status.get("pair_cost_bundle_manifest_sha256")
    ):
        raise ValueError("registered workspace cost bundle is missing or changed")
    model_source = root / _required_text(
        cost_bundle_receipt.get("pair_cost_models_path"),
        "pair_cost_models_path",
    )
    l2_source = root / _required_text(
        cost_bundle_receipt.get("input_paths", {}).get("strict_l2_window"),
        "strict_l2_window",
    )
    fee_source = root / _required_text(
        cost_bundle_receipt.get("input_paths", {}).get("fee_profile"),
        "fee_profile",
    )
    if (
        _file_hash(model_source) != _text(cost_bundle_receipt.get("pair_cost_models_sha256"))
        or _file_hash(l2_source)
        != _text(cost_bundle_receipt.get("input_hashes", {}).get("l2_samples_sha256"))
        or _file_hash(fee_source)
        != _text(cost_bundle_receipt.get("input_hashes", {}).get("fee_profile_sha256"))
    ):
        raise ValueError("registered workspace cost bundle artifact mismatch")
    bundle_blockers = validate_pair_cost_bundle_artifacts(
        root=root,
        bundle_manifest_path=cost_bundle_receipt_path,
        pair_cost_models_path=model_source,
    )
    if bundle_blockers:
        raise ValueError(
            "registered workspace cost bundle lineage invalid: "
            + ";".join(bundle_blockers)
        )

    workspace = root / "data" / "research" / "registered_rerun_workspaces" / execution_id
    receipt_path = workspace / "workspace_receipt.json"
    if receipt_path.is_file():
        _validate_workspace_receipt(
            root=root,
            workspace=workspace,
            receipt_path=receipt_path,
            contract=contract,
            ready_path=ready_path,
            cost_bundle_receipt_path=cost_bundle_receipt_path,
        )
        return workspace, receipt_path

    temporary = workspace.with_name(workspace.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    (temporary / "reports" / "active").mkdir(parents=True, exist_ok=True)
    (temporary / "reports" / "snapshots").mkdir(parents=True, exist_ok=True)
    (temporary / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (temporary / "data" / "research").mkdir(parents=True, exist_ok=True)
    shutil.copytree(root / "config", temporary / "config")

    seeded: dict[str, str] = {}
    mutable_active_inputs: dict[str, str] = {}
    for config_file in (temporary / "config").rglob("*"):
        if config_file.is_file():
            seeded[_relative(config_file, temporary)] = _file_hash(config_file)

    def seed(source: Path, relative: Path | str) -> Path:
        if not source.is_file():
            raise FileNotFoundError(f"registered workspace input missing: {source}")
        target = temporary / Path(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(target, source.read_bytes())
        seeded[_relative(target, temporary)] = _file_hash(target)
        return target

    def seed_at_same_path(source: Path) -> Path:
        return seed(source, source.relative_to(root))

    def seed_mutable_active(filename: str, source: Path) -> None:
        immutable = seed(
            source,
            Path("data/research/registered_rerun_workspace_inputs") / filename,
        )
        target = temporary / workspace_active / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(target, immutable.read_bytes())
        mutable_active_inputs[filename] = _relative(immutable, temporary)

    for source in _frozen_handoff_input_files(
        handoff_dir=handoff_dir,
        history_manifest_path=history_manifest_path,
        history_manifest=history_manifest,
    ):
        seed_at_same_path(source)
    refresh_manifest_path = handoff_dir / "inputs" / "exhaustive_wizard_api_refresh_manifest.json"
    refresh_manifest = _read_json(refresh_manifest_path)
    for value in refresh_manifest.get("artifacts", {}).values():
        candidate = root / _text(value)
        if candidate.is_file() and "reports/snapshots/" in _relative(candidate, root):
            seed_at_same_path(candidate)

    active = root / "reports" / "active"
    workspace_active = Path("reports/active")
    active_sources = {
        "current_wizard_hyperliquid_handoff_manifest.json": handoff_manifest_path,
        "current_wizard_hyperliquid_experiment_matrix.csv": frozen_matrix,
        "current_wizard_hyperliquid_asset_fetch_queue.csv": handoff_dir / "asset_fetch_queue.csv",
        "current_wizard_hyperliquid_pair_history_queue.csv": handoff_dir / "pair_history_queue.csv",
        "current_wizard_pair_detail_status.csv": handoff_dir / "pair_detail_status.csv",
        "current_wizard_hyperliquid_handoff_validation.csv": handoff_dir / "validation.csv",
        "current_wizard_hyperliquid_handoff_summary.md": handoff_dir / "summary.md",
        "current_wizard_hyperliquid_history_manifest.json": history_manifest_path,
        "current_wizard_hyperliquid_asset_history_results.csv": root
        / _text(history_manifest.get("artifacts", {}).get("snapshot_asset_results")),
        "current_wizard_hyperliquid_pair_history_results.csv": root
        / _text(history_manifest.get("artifacts", {}).get("snapshot_pair_results")),
        "current_wizard_hyperliquid_history_validation.csv": root
        / _text(history_manifest.get("artifacts", {}).get("snapshot_validation")),
        "current_wizard_hyperliquid_history_summary.md": root
        / _text(history_manifest.get("artifacts", {}).get("snapshot_summary_md")),
        "exhaustive_wizard_api_refresh_manifest.json": refresh_manifest_path,
        "exhaustive_wizard_api_refresh_source_accounting.csv": handoff_dir
        / "inputs"
        / "exhaustive_wizard_api_refresh_source_accounting.csv",
        "wizard_sweep_candidates.csv": root
        / _text(refresh_manifest.get("artifacts", {}).get("snapshot_api_candidates")),
    }
    passthrough_active = (
        "hypothesis_ledger_audit.csv",
        "wizard_mode_parity.csv",
        "corrective_wizard_proof_scheduler_status.json",
        "corrective_l2_capture_status.json",
        "registered_research_rerun_contract.json",
        "hyperliquid_testnet_market_inventory.csv",
        "hyperliquid_testnet_margin_tiers.csv",
    )
    for filename in passthrough_active:
        source = active / filename
        if source.is_file():
            active_sources[filename] = source
    funding_ledger = _select_frozen_funding_ledger(
        root=root,
        history_manifest_path=history_manifest_path,
        pair_keys=pair_keys,
    )
    if funding_ledger is not None:
        seed_at_same_path(funding_ledger)
        seed_mutable_active(
            "current_wizard_hyperliquid_funding_asset_results.csv",
            funding_ledger,
        )
        for funding_file in _funding_files_from_ledger(root, funding_ledger):
            seed_at_same_path(funding_file)
    for filename, source in active_sources.items():
        seed(source, workspace_active / filename)
    for filename in (
        "acceptance_policy_receipt.json",
        "holdout_policy_receipt.json",
    ):
        source = active / filename
        if source.is_file():
            seed_mutable_active(filename, source)

    seed(model_source, "data/processed/hyperliquid_pair_cost_models.csv")
    seed(l2_source, "data/processed/hyperliquid_l2_slippage_samples.csv")
    seed(fee_source, "config/hyperliquid_perp_cost_profile.json")
    for input_name in ("candidate_set", "cost_status"):
        seed_at_same_path(
            root
            / _required_text(
                cost_bundle_receipt.get("input_paths", {}).get(input_name),
                input_name,
            )
        )
    ledger = root / "data" / "research" / "hypothesis_ledger.jsonl"
    if ledger.is_file():
        seed(ledger, "data/research/hypothesis_ledger.jsonl")
    immutable_contract = root / _required_text(
        contract.get("immutable_contract_path"), "immutable_contract_path"
    )
    seed_at_same_path(immutable_contract)
    seed_at_same_path(source_receipt_path)
    seed_at_same_path(frozen_matrix)
    seed_at_same_path(ready_path)
    for evidence in _receipt_evidence_paths(root=root, receipt=_read_json(ready_path)):
        seed_at_same_path(evidence)
    proof_status = _read_json(active / "corrective_wizard_proof_scheduler_status.json")
    for evidence in _receipt_evidence_paths(root=root, receipt=proof_status):
        seed_at_same_path(evidence)

    receipt_core = {
        "schema_version": "thewiz.registered_rerun_workspace.v1",
        "execution_id": execution_id,
        "contract_id": contract["contract_id"],
        "source_family_sha256": contract["source_family_sha256"],
        "source_family_rows": int(contract["source_family_rows"]),
        "history_manifest_path": _relative(history_manifest_path, root),
        "history_manifest_sha256": _file_hash(history_manifest_path),
        "ready_receipt_path": _relative(ready_path, root),
        "ready_receipt_sha256": _file_hash(ready_path),
        "cost_bundle_receipt_path": _relative(cost_bundle_receipt_path, root),
        "cost_bundle_receipt_sha256": _file_hash(cost_bundle_receipt_path),
        "seeded_input_hashes": dict(sorted(seeded.items())),
        "mutable_active_input_sources": dict(sorted(mutable_active_inputs.items())),
        "daily_active_layer_modified": False,
        "authenticated_wizard_capture_performed": False,
        "order_submission_performed": False,
        "promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    receipt = {
        **receipt_core,
        "receipt_id": "registeredworkspace_" + _payload_hash(receipt_core)[:20],
    }
    _atomic_json(receipt, temporary / "workspace_receipt.json")
    workspace.parent.mkdir(parents=True, exist_ok=True)
    promote_staged_directory(temporary, workspace)
    _validate_workspace_receipt(
        root=root,
        workspace=workspace,
        receipt_path=receipt_path,
        contract=contract,
        ready_path=ready_path,
        cost_bundle_receipt_path=cost_bundle_receipt_path,
    )
    return workspace, receipt_path


def _select_registered_history_manifest(
    *, root: Path, handoff_dir: Path, pair_keys: tuple[str, ...]
) -> Path:
    matches: list[Path] = []
    expected = set(pair_keys)
    for manifest_path in sorted(handoff_dir.glob("history_runs/*/manifest.json")):
        manifest = _read_json(manifest_path)
        pair_path = root / _text(manifest.get("artifacts", {}).get("snapshot_pair_results"))
        pairs = _read_csv(pair_path)
        present = set(pairs.get("pair_group_key", pd.Series(dtype=str)).astype(str))
        selected = pairs.loc[
            pairs.get("pair_group_key", pd.Series(dtype=str)).astype(str).isin(expected)
        ]
        statuses = selected.get("history_status", pd.Series(dtype=str)).astype(str)
        if (
            expected.issubset(present)
            and len(selected) == len(expected)
            and statuses.eq("READY_FOR_CANONICAL_1X_REPLAY").all()
        ):
            matches.append(manifest_path)
    if not matches:
        raise ValueError("registered workspace has no complete frozen history run")
    return matches[-1]


def _frozen_handoff_input_files(
    *,
    handoff_dir: Path,
    history_manifest_path: Path,
    history_manifest: dict[str, Any],
) -> list[Path]:
    files = [path for path in handoff_dir.iterdir() if path.is_file()]
    inputs = handoff_dir / "inputs"
    if inputs.is_dir():
        files.extend(path for path in inputs.rglob("*") if path.is_file())
    files.extend(
        root_path for root_path in history_manifest_path.parent.iterdir() if root_path.is_file()
    )
    for directory in ("assets", "pairs"):
        source_dir = history_manifest_path.parent / directory
        if source_dir.is_dir():
            files.extend(path for path in source_dir.rglob("*") if path.is_file())
    return list(dict.fromkeys(files))


def _select_frozen_funding_ledger(
    *, root: Path, history_manifest_path: Path, pair_keys: tuple[str, ...]
) -> Path | None:
    expected = set(pair_keys)
    candidates: list[tuple[str, Path]] = []
    for manifest_path in sorted(history_manifest_path.parent.glob("cost_evidence/*/manifest.json")):
        manifest = _read_json(manifest_path)
        pair_path = root / _text(manifest.get("artifacts", {}).get("snapshot_pairs"))
        asset_path = root / _text(manifest.get("artifacts", {}).get("snapshot_assets"))
        pairs = _read_csv(pair_path)
        selected = pairs.loc[
            pairs.get("pair_group_key", pd.Series(dtype=str)).astype(str).isin(expected)
        ]
        if (
            len(selected) == len(expected)
            and selected.get("funding_status", pd.Series(dtype=str))
            .astype(str)
            .eq("COMPLETE")
            .all()
            and asset_path.is_file()
        ):
            candidates.append((_text(manifest.get("as_of")), asset_path))
    return max(candidates, default=("", None), key=lambda item: item[0])[1]


def _funding_files_from_ledger(root: Path, ledger_path: Path) -> list[Path]:
    frame = _read_csv(ledger_path)
    paths: list[Path] = []
    for field in ("funding_path", "funding_source_path"):
        for value in frame.get(field, pd.Series(dtype=str)).astype(str):
            path = root / value
            if value and path.is_file():
                paths.append(path)
    return list(dict.fromkeys(paths))


def _receipt_evidence_paths(*, root: Path, receipt: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []

    def collect(value: Any) -> None:
        if isinstance(value, dict):
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)
        elif isinstance(value, str) and value and not value.startswith("/"):
            candidate = root / value
            if candidate.is_file():
                paths.append(candidate)

    collect(receipt)
    return list(dict.fromkeys(paths))


def _validate_workspace_receipt(
    *,
    root: Path,
    workspace: Path,
    receipt_path: Path,
    contract: dict[str, Any],
    ready_path: Path,
    cost_bundle_receipt_path: Path,
) -> None:
    receipt = _read_json(receipt_path)
    if (
        receipt.get("schema_version") != "thewiz.registered_rerun_workspace.v1"
        or receipt.get("contract_id") != contract.get("contract_id")
        or receipt.get("source_family_sha256") != contract.get("source_family_sha256")
        or receipt.get("ready_receipt_sha256") != _file_hash(ready_path)
        or receipt.get("cost_bundle_receipt_sha256") != _file_hash(cost_bundle_receipt_path)
        or _truthy(receipt.get("daily_active_layer_modified"))
        or _truthy(receipt.get("promotion_authority"))
        or _truthy(receipt.get("testnet_order_authority"))
        or _truthy(receipt.get("live_trading_authorized"))
    ):
        raise ValueError("registered rerun workspace receipt mismatch")
    hashes = receipt.get("seeded_input_hashes", {})
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("registered rerun workspace has no bound inputs")
    for relative, expected in hashes.items():
        if _file_hash(workspace / str(relative)) != str(expected):
            raise ValueError(f"registered rerun workspace input changed: {relative}")
    _validate_execution_source_family(root=workspace, contract=contract)


def _validate_contract(*, root: Path, contract_path: Path) -> dict[str, Any]:
    contract = _read_json(contract_path)
    if not contract:
        raise ValueError("registered rerun contract is missing")
    validate_registered_rerun_contract_identity(contract)
    immutable_path = root / _required_text(
        contract.get("immutable_contract_path"), "immutable_contract_path"
    )
    if _read_json(immutable_path) != contract:
        raise ValueError("registered rerun immutable contract mismatch")
    if not bool(contract.get("full_family_multiplicity_required")):
        raise ValueError("registered rerun does not require full-family multiplicity")
    if not bool(contract.get("promotion_evaluation_registered_only")):
        raise ValueError("registered rerun promotion scope is not frozen")
    if bool(contract.get("threshold_changes_after_contract_permitted")):
        raise ValueError("registered rerun permits post-contract threshold changes")
    _assert_no_order_authority("registered_contract", contract)
    _validate_source_family(root=root, contract=contract)
    discovery = root / "config" / "wizard_discovery_policy.json"
    if _file_hash(discovery) != contract.get("discovery_policy_sha256"):
        raise ValueError("registered rerun discovery policy hash drifted")
    for filename, field in (
        ("acceptance_policy_receipt.json", "acceptance_policy_id"),
        ("holdout_policy_receipt.json", "holdout_policy_id"),
    ):
        payload = _read_json(root / "reports" / "active" / filename)
        if payload.get("policy_id") != contract.get(field):
            raise ValueError(f"registered rerun {field} drifted")
    return contract


def _validate_source_family(*, root: Path, contract: dict[str, Any]) -> None:
    current = root / "reports" / "active" / "current_wizard_hyperliquid_experiment_matrix.csv"
    path, _ = resolve_registered_source_family(
        root=root,
        contract=contract,
        current_matrix_path=current,
    )
    if _file_hash(path) != contract.get("source_family_sha256"):
        raise ValueError("registered rerun source family hash drifted")
    frame = _read_csv(path)
    if len(frame) != int(contract.get("source_family_rows", -1)):
        raise ValueError("registered rerun source family row count drifted")
    if "experiment_id" not in frame or frame["experiment_id"].duplicated().any():
        raise ValueError("registered rerun source family identity is invalid")


def _validate_execution_source_family(*, root: Path, contract: dict[str, Any]) -> None:
    """Require the stage runner's active layer to be the frozen family exactly."""

    active = root / "reports" / "active"
    matrix = active / "current_wizard_hyperliquid_experiment_matrix.csv"
    if _file_hash(matrix) != contract.get("source_family_sha256"):
        raise ValueError("registered rerun source family hash drifted")
    handoff = _read_json(active / "current_wizard_hyperliquid_handoff_manifest.json")
    snapshot = root / str(handoff.get("artifacts", {}).get("snapshot_experiments", ""))
    if _file_hash(snapshot) != contract.get("source_family_sha256"):
        raise ValueError("registered rerun immutable handoff family mismatch")


def _validate_ready_receipt(
    *, ready_path: Path, contract: dict[str, Any], root: Path
) -> dict[str, Any]:
    ready = _read_json(ready_path)
    if not ready or ready.get("contract_id") != contract.get("contract_id"):
        raise ValueError("registered rerun ready receipt is missing or mismatched")
    if not bool(ready.get("scheduled_research_rerun_authorized")):
        raise ValueError("registered rerun ready receipt lacks research authority")
    _validate_ready_receipt_lineage(
        root=root,
        ready_receipt=ready,
        contract=contract,
        parity_path=root / "reports" / "active" / "wizard_mode_parity.csv",
    )
    _assert_no_order_authority("registered_ready_receipt", ready)
    return ready


def _registered_semantic_ids(contract: dict[str, Any]) -> set[str]:
    candidates = contract.get("registered_candidates", [])
    if not isinstance(candidates, list):
        raise TypeError("registered rerun contract candidates are invalid")
    candidate_ids = [
        str(row.get("semantic_hypothesis_id", "")).strip()
        for row in candidates
        if isinstance(row, dict)
    ]
    if (
        not candidate_ids
        or "" in candidate_ids
        or len(candidate_ids) != len(candidates)
        or len(candidate_ids) != len(set(candidate_ids))
    ):
        raise ValueError("registered rerun contract semantic identities are invalid")
    return set(candidate_ids)


def _registered_pair_keys(gate: pd.DataFrame, contract: dict[str, Any]) -> tuple[str, ...]:
    expected_ids = _registered_semantic_ids(contract)
    if gate.empty or "semantic_hypothesis_id" not in gate:
        raise ValueError("registered rerun candidate gate is empty")
    selected = gate.loc[gate["semantic_hypothesis_id"].astype(str).isin(expected_ids)].copy()
    selected_ids = selected["semantic_hypothesis_id"].map(_text)
    if (
        len(selected) != len(expected_ids)
        or selected_ids.duplicated().any()
        or set(selected_ids) != expected_ids
    ):
        raise ValueError("registered rerun candidate gate identity coverage mismatch")
    if (
        not selected.get("pre_rerun_gate_ready", pd.Series(False, index=selected.index))
        .map(_truthy)
        .all()
    ):
        raise ValueError("registered rerun candidate gates are not all ready")

    candidates = contract.get("registered_candidates", [])
    expected_by_id = {
        _text(candidate.get("semantic_hypothesis_id")): candidate
        for candidate in candidates
        if isinstance(candidate, Mapping)
    }
    field_bindings = (
        ("source_experiment_id", "source_experiment_id"),
        ("pair_group_key", "pair_group_key"),
        ("pair", "pair"),
        ("exact_mode", "exact_mode"),
        ("orientation", "orientation"),
    )
    for row in selected.to_dict("records"):
        semantic_id = _text(row.get("semantic_hypothesis_id"))
        candidate = expected_by_id.get(semantic_id, {})
        for gate_field, contract_field in field_bindings:
            if _text(row.get(gate_field)) != _text(candidate.get(contract_field)):
                raise ValueError(
                    f"registered rerun candidate gate binding mismatch:{semantic_id}:{gate_field}"
                )

    pair_keys = tuple(
        sorted(
            {_text(candidate.get("pair_group_key")) for candidate in expected_by_id.values()} - {""}
        )
    )
    if not pair_keys:
        raise ValueError("registered rerun has no registered pair keys")
    return pair_keys


def _validate_stage_result(
    *,
    name: str,
    result: CommandResult,
    root: Path,
    artifact_root: Path | None = None,
    expected_experiments: int,
) -> dict[str, Any]:
    artifact_root = artifact_root or root
    summary = result.summary
    manifest = Path(result.paths.get("snapshot_manifest") or result.paths.get("manifest") or "")
    if not manifest.is_file():
        raise ValueError(f"registered rerun {name} immutable manifest is missing")
    accounted = _experiment_count(summary)
    if name in FAMILY_ACCOUNTING_STAGES and accounted != expected_experiments:
        raise ValueError(
            f"registered rerun {name} accounted {accounted} of {expected_experiments} experiments"
        )
    if name == "chain_validation" and (
        summary.get("chain_status") != "PASS"
        or int(summary.get("checks", 0)) != int(summary.get("checks_passed", -1))
        or int(summary.get("stages_complete", 0)) != int(summary.get("stages_expected", -1))
    ):
        raise ValueError("registered rerun chain validation did not pass completely")
    if name == "ou_optimal_overlay" and int(summary.get("source_rows_accounted", -1)) != int(
        summary.get("source_rows", -2)
    ):
        raise ValueError("registered rerun OU Optimal overlay is not fully accounted")
    immutable_artifacts = {
        _relative(Path(path), artifact_root): _file_hash(Path(path))
        for key, path in result.paths.items()
        if Path(path).is_file()
        and (
            str(key).startswith("snapshot")
            or "reports/snapshots/" in _relative(Path(path), artifact_root)
        )
    }
    manifest_relative = _relative(manifest, artifact_root)
    if immutable_artifacts.get(manifest_relative) != _file_hash(manifest):
        raise ValueError(f"registered rerun {name} immutable manifest is not snapshot-bound")
    return {
        "stage": name,
        "stage_identity": _stage_identity(summary),
        "experiments_accounted": accounted,
        "manifest_path": manifest_relative,
        "manifest_sha256": _file_hash(manifest),
        "immutable_artifact_hashes": immutable_artifacts,
        "order_submission_performed": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }


def _validate_registered_cost_rows(*, result: CommandResult, pair_keys: tuple[str, ...]) -> None:
    path = Path(result.paths.get("pairs") or "")
    frame = _read_csv(path)
    selected = frame.loc[
        frame.get("pair_group_key", pd.Series(dtype=str)).astype(str).isin(pair_keys)
    ]
    if len(selected) != len(pair_keys):
        raise ValueError("registered rerun cost rows do not cover every registered pair")
    if (
        not selected.get("cost_acceptance_ready", pd.Series(False, index=selected.index))
        .map(_truthy)
        .all()
    ):
        raise ValueError("registered rerun pair costs are not strict acceptance-ready")


def _validate_conclusion(
    *,
    path: Path,
    contract: dict[str, Any],
    summary: dict[str, Any],
    root: Path,
) -> dict[str, Any]:
    conclusion = _read_json(path)
    if not path.is_file() or conclusion.get("contract_id") != contract.get("contract_id"):
        raise ValueError("registered rerun conclusion receipt is missing or mismatched")
    if not bool(summary.get("registered_rerun_results_accounted")):
        raise ValueError("registered rerun reconciliation did not account for results")
    if summary.get("registered_rerun_gate_status") != "PASS_REGISTERED_RERUN_ACCOUNTED":
        raise ValueError("registered rerun reconciliation gate did not pass")
    status = str(conclusion.get("conclusion_status", "INCOMPLETE"))
    if status not in {
        "ACCEPTED_REGISTERED_SURVIVORS",
        "CONCLUSIVE_ZERO_REGISTERED_SURVIVORS",
    }:
        raise ValueError("registered rerun conclusion is incomplete")
    _validate_conclusion_accounting(conclusion=conclusion, contract=contract)
    source_bindings = (
        (
            "result_chain_sha256",
            root
            / "reports"
            / "active"
            / "current_wizard_hyperliquid_chain_validation_manifest.json",
        ),
        (
            "final_survivor_receipt_sha256",
            root / "reports" / "active" / "final_1x_survivor_receipt.json",
        ),
        (
            "failure_attribution_sha256",
            root / "reports" / "active" / "current_wizard_hyperliquid_failure_attribution.csv",
        ),
        (
            "failure_attribution_manifest_sha256",
            root
            / "reports"
            / "active"
            / "current_wizard_hyperliquid_failure_attribution_manifest.json",
        ),
    )
    for field, source_path in source_bindings:
        if _file_hash(source_path) != str(conclusion.get(field, "")):
            raise ValueError(f"registered rerun conclusion source binding mismatch: {field}")
    _assert_no_order_authority("registered_conclusion", conclusion)
    return conclusion


def _validate_conclusion_accounting(
    *, conclusion: dict[str, Any], contract: dict[str, Any]
) -> None:
    if conclusion.get("schema_version") != CONCLUSION_SCHEMA_VERSION:
        raise ValueError("registered rerun conclusion schema mismatch")
    conclusion_core = {key: value for key, value in conclusion.items() if key != "conclusion_id"}
    expected_conclusion_id = "registeredconclusion_" + _payload_hash(conclusion_core)[:20]
    if conclusion.get("conclusion_id") != expected_conclusion_id:
        raise ValueError("registered rerun conclusion content identity mismatch")
    expected_ids = _registered_semantic_ids(contract)
    outcomes = conclusion.get("outcomes")
    if not isinstance(outcomes, dict) or set(outcomes) != expected_ids:
        raise ValueError("registered rerun conclusion does not account for every candidate")
    allowed = {"ACCEPTED_SURVIVOR", "REJECTED_BY_FROZEN_GATES"}
    if any(value not in allowed for value in outcomes.values()):
        raise ValueError("registered rerun conclusion has an invalid outcome")
    accepted = sum(value == "ACCEPTED_SURVIVOR" for value in outcomes.values())
    rejected = sum(value == "REJECTED_BY_FROZEN_GATES" for value in outcomes.values())
    if int(conclusion.get("registered_hypotheses", -1)) != len(expected_ids):
        raise ValueError("registered rerun conclusion registered count mismatch")
    if accepted != int(conclusion.get("accepted_registered_hypotheses", -1)):
        raise ValueError("registered rerun conclusion accepted count mismatch")
    if rejected != int(conclusion.get("rejected_registered_hypotheses", -1)):
        raise ValueError("registered rerun conclusion rejected count mismatch")
    status = str(conclusion.get("conclusion_status", ""))
    if (status == "ACCEPTED_REGISTERED_SURVIVORS" and accepted <= 0) or (
        status == "CONCLUSIVE_ZERO_REGISTERED_SURVIVORS" and accepted != 0
    ):
        raise ValueError("registered rerun conclusion status/count mismatch")


def _stage4_learning_eligible(receipt: dict[str, Any]) -> bool:
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


def _stage4_checkpoint_row(*, root: Path, path: Path) -> dict[str, Any]:
    resolved = path if path.is_absolute() else root / path
    frame = _read_csv(resolved)
    if frame.empty or "stage" not in frame:
        return {}
    selected = frame.loc[frame["stage"].astype(str).eq("4")]
    return selected.iloc[0].to_dict() if len(selected) == 1 else {}


def _stage4_checkpoint_allows_learning(row: dict[str, Any], *, contract_id: str) -> bool:
    evidence = str(row.get("evidence_progress", ""))
    return bool(
        row
        and contract_id
        and str(row.get("status", "")) == "PASS"
        and f"active_contract_id={contract_id}" in evidence
        and "stage4_terminal_outcome=ACCEPTED_VALID_INDEPENDENT_SURVIVORS" in evidence
        and "final_receipt_conclusion_bound=True" in evidence
        and not _truthy(row.get("testnet_order_authority"))
        and not _truthy(row.get("live_trading_authorized"))
    )


def _validate_stage4_survivor_receipt(
    *,
    path: Path,
    root: Path,
    contract: dict[str, Any],
    conclusion: dict[str, Any],
) -> dict[str, Any]:
    receipt = _read_json(path)
    if _file_hash(path) != str(conclusion.get("final_survivor_receipt_sha256", "")):
        raise ValueError("registered Stage 4 survivor receipt hash mismatch")
    policy = _read_json(root / "config" / "acceptance_policy_manifest.json")
    try:
        minimum_clusters = int(
            policy.get("research_gates", {}).get("minimum_independent_supporting_clusters", 0)
        )
        final_survivors = int(receipt.get("final_one_x_survivors", -1))
        independent_clusters = int(receipt.get("independent_supporting_clusters", -1))
        full_survivor_clusters = int(receipt.get("independent_full_survivor_clusters", -1))
        independent_pairs = int(receipt.get("independent_supporting_pairs", -1))
        full_survivor_pairs = int(receipt.get("independent_full_survivor_pairs", -1))
    except (TypeError, ValueError):
        raise ValueError("registered Stage 4 survivor receipt counts are invalid") from None
    final_ids = receipt.get("final_experiment_ids", [])
    final_pairs = receipt.get("final_canonical_pairs", [])
    blockers = receipt.get("blockers", [])
    if (
        policy.get("schema_version") != "thewiz.acceptance_policy.v1"
        or minimum_clusters <= 0
        or receipt.get("schema_version") != "thewiz.final_one_x_survivor_receipt.v1"
        or receipt.get("acceptance_policy_id") != contract.get("acceptance_policy_id")
        or receipt.get("holdout_policy_id") != contract.get("holdout_policy_id")
        or not isinstance(final_ids, list)
        or len(final_ids) != final_survivors
        or not isinstance(final_pairs, list)
        or len(final_pairs) != len({str(value) for value in final_pairs})
        or not isinstance(blockers, list)
        or receipt.get("thresholds_changed_after_results") is not False
        or _truthy(receipt.get("testnet_order_authority"))
        or _truthy(receipt.get("live_trading_authorized"))
    ):
        raise ValueError("registered Stage 4 survivor receipt lineage is invalid")
    if final_survivors > 0:
        if (
            receipt.get("receipt_status") != "PASS"
            or blockers
            or independent_clusters < minimum_clusters
            or full_survivor_clusters < minimum_clusters
            or independent_pairs < minimum_clusters
            or full_survivor_pairs < minimum_clusters
            or final_survivors < minimum_clusters
            or receipt.get("testnet_candidate_authority") is not True
        ):
            raise ValueError("registered Stage 4 survivor receipt acceptance is invalid")
    elif (
        final_survivors != 0
        or receipt.get("receipt_status") != "ZERO_SURVIVORS"
        or _truthy(receipt.get("testnet_candidate_authority"))
    ):
        raise ValueError("registered Stage 4 zero-survivor receipt is invalid")
    candidates = contract.get("registered_candidates", [])
    outcomes = conclusion.get("outcomes", {})
    if not isinstance(candidates, list) or not isinstance(outcomes, dict):
        raise TypeError("registered Stage 4 accepted identity lineage is malformed")
    accepted_experiment_ids = {
        str(candidate.get("source_experiment_id", "")).strip()
        for candidate in candidates
        if isinstance(candidate, dict)
        and outcomes.get(str(candidate.get("semantic_hypothesis_id", "")).strip())
        == "ACCEPTED_SURVIVOR"
    } - {""}
    supplied_final_ids = [str(value).strip() for value in final_ids]
    candidate_by_experiment = {
        str(candidate.get("source_experiment_id", "")).strip(): candidate
        for candidate in candidates
        if isinstance(candidate, dict) and str(candidate.get("source_experiment_id", "")).strip()
    }
    accepted_pairs = {
        _registered_candidate_pair(candidate_by_experiment[experiment_id])
        for experiment_id in accepted_experiment_ids
        if experiment_id in candidate_by_experiment
    } - {""}
    supplied_final_pairs = {str(value).strip() for value in final_pairs} - {""}
    if (
        len(supplied_final_ids) != len(set(supplied_final_ids))
        or set(supplied_final_ids) != accepted_experiment_ids
        or final_survivors != len(accepted_experiment_ids)
        or int(conclusion.get("accepted_registered_hypotheses", -1)) != len(accepted_experiment_ids)
        or (
            final_survivors > 0
            and (
                supplied_final_pairs != accepted_pairs
                or len(accepted_pairs) != full_survivor_pairs
                or len(accepted_pairs) < minimum_clusters
            )
        )
    ):
        raise ValueError(
            "registered Stage 4 survivor receipt is not the exact accepted registered cohort"
        )
    return receipt


def _registered_candidate_pair(candidate: dict[str, Any]) -> str:
    pair_group_key = str(candidate.get("pair_group_key", "")).strip()
    parts = [part.strip().upper() for part in pair_group_key.split("|") if part.strip()]
    if len(parts) >= 2:
        assets = [part for part in parts[-2:] if part not in {"DAILY", "HOURLY"}]
        if len(assets) == 2:
            return "-".join(sorted(assets))
    pair = str(candidate.get("pair", "")).upper().replace("/", "-")
    tokens = [token for token in pair.split("-") if token and token != "USD"]
    return "-".join(sorted(tokens[:2])) if len(tokens) >= 2 else ""


def _validate_execution_receipt(
    *, receipt_path: Path, contract: dict[str, Any], root: Path
) -> dict[str, Any]:
    contract_id = _required_text(contract.get("contract_id"), "contract_id")
    canonical_path = (
        root / "data" / "research" / "registered_rerun_executions" / f"{contract_id}.json"
    )
    if receipt_path.resolve() != canonical_path.resolve():
        raise ValueError("registered rerun execution receipt path is not canonical")
    receipt = _read_json(receipt_path)
    expected_hash = str(receipt.pop("receipt_sha256", ""))
    if not expected_hash or _payload_hash(receipt) != expected_hash:
        raise ValueError("registered rerun immutable execution receipt hash mismatch")
    receipt["receipt_sha256"] = expected_hash
    if receipt.get("contract_id") != contract_id:
        raise ValueError("registered rerun execution receipt contract mismatch")
    if receipt.get("source_family_sha256") != contract.get("source_family_sha256"):
        raise ValueError("registered rerun execution receipt family mismatch")
    ready_path = root / str(receipt.get("ready_receipt_path", ""))
    if _file_hash(ready_path) != receipt.get("ready_receipt_sha256"):
        raise ValueError("registered rerun immutable ready receipt hash mismatch")
    evidence_root = root
    if bool(receipt.get("isolated_workspace_used")):
        workspace = root / _required_text(
            receipt.get("isolated_workspace_path"), "isolated_workspace_path"
        )
        evidence_root = workspace
        workspace_receipt = root / _required_text(
            receipt.get("workspace_receipt_path"), "workspace_receipt_path"
        )
        if _file_hash(workspace_receipt) != receipt.get("workspace_receipt_sha256"):
            raise ValueError("registered rerun workspace receipt hash mismatch")
        workspace_payload = _read_json(workspace_receipt)
        cost_bundle_receipt = root / _required_text(
            workspace_payload.get("cost_bundle_receipt_path"),
            "cost_bundle_receipt_path",
        )
        _validate_workspace_receipt(
            root=root,
            workspace=workspace,
            receipt_path=workspace_receipt,
            contract=contract,
            ready_path=ready_path,
            cost_bundle_receipt_path=cost_bundle_receipt,
        )
    for stage in receipt.get("stages", []):
        path = root / str(stage.get("manifest_path", ""))
        if _file_hash(path) != stage.get("manifest_sha256"):
            raise ValueError("registered rerun immutable stage manifest hash mismatch")
        artifacts = stage.get("immutable_artifact_hashes", {})
        if not isinstance(artifacts, dict) or not artifacts:
            raise ValueError("registered rerun immutable stage artifacts are unbound")
        for relative, expected_hash in artifacts.items():
            if _file_hash(root / str(relative)) != str(expected_hash):
                raise ValueError("registered rerun immutable stage artifact hash mismatch")
    conclusion_path = root / str(receipt.get("conclusion_path", ""))
    if _file_hash(conclusion_path) != receipt.get("conclusion_sha256"):
        raise ValueError("registered rerun immutable conclusion hash mismatch")
    conclusion = _read_json(conclusion_path)
    _validate_conclusion_accounting(conclusion=conclusion, contract=contract)
    for field in (
        "conclusion_status",
        "accepted_registered_hypotheses",
        "rejected_registered_hypotheses",
    ):
        if receipt.get(field) != conclusion.get(field):
            raise ValueError(f"registered rerun execution/conclusion mismatch: {field}")
    stage4_survivor_path = root / _required_text(
        receipt.get("stage4_survivor_receipt_path"),
        "stage4_survivor_receipt_path",
    )
    if _file_hash(stage4_survivor_path) != receipt.get("stage4_survivor_receipt_sha256"):
        raise ValueError("registered rerun Stage 4 survivor receipt hash mismatch")
    survivor = _validate_stage4_survivor_receipt(
        path=stage4_survivor_path,
        root=evidence_root,
        contract=contract,
        conclusion=conclusion,
    )
    expected_stage4_fields = {
        "stage4_independent_supporting_clusters": int(
            survivor.get("independent_supporting_clusters", 0) or 0
        ),
        "stage4_independent_full_survivor_clusters": int(
            survivor.get("independent_full_survivor_clusters", 0) or 0
        ),
        "stage4_independent_supporting_pairs": int(
            survivor.get("independent_supporting_pairs", 0) or 0
        ),
        "stage4_independent_full_survivor_pairs": int(
            survivor.get("independent_full_survivor_pairs", 0) or 0
        ),
        "stage4_final_one_x_survivors": int(survivor.get("final_one_x_survivors", 0) or 0),
        "stage4_final_experiment_ids": list(survivor.get("final_experiment_ids", [])),
        "stage4_final_canonical_pairs": list(survivor.get("final_canonical_pairs", [])),
    }
    if any(receipt.get(key) != value for key, value in expected_stage4_fields.items()):
        raise ValueError("registered rerun Stage 4 survivor summary mismatch")
    _assert_no_order_authority("registered_execution_receipt", receipt)
    return receipt


def _execution_id(
    *,
    root: Path,
    contract: dict[str, Any],
    ready_path: Path,
    gate_path: Path,
    pair_keys: tuple[str, ...],
) -> str:
    material = {
        "contract_id": contract["contract_id"],
        "source_family_sha256": contract["source_family_sha256"],
        "ready_receipt_sha256": _file_hash(ready_path),
        "candidate_gate_sha256": _file_hash(gate_path),
        "strict_pair_cost_models_sha256": _file_hash(
            root / "data" / "processed" / "hyperliquid_pair_cost_models.csv"
        ),
        "registered_pair_group_keys": pair_keys,
    }
    return "registeredrun_" + sha256(_canonical_json(material).encode("utf-8")).hexdigest()[:20]


def _publish_result(
    *,
    root: Path,
    status_path: Path,
    status: str,
    contract: dict[str, Any],
    execute: bool,
    blocker: str,
    receipt_path: Path | None = None,
    receipt: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
    funding_audit: dict[str, Any] | None = None,
) -> CommandResult:
    audit = funding_audit or {
        "funding_refresh_requested": False,
        "funding_cost_evidence_stage_run": False,
        "funding_refresh_dispatched_to_cost_stage": False,
        "funding_assets_fetched_from_network": 0,
        "funding_assets_reused_from_cache": 0,
        "funding_refresh_outcome": "NOT_REACHED",
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": status,
        "execution_requested": execute,
        "contract_id": str(contract.get("contract_id", "")),
        "source_family_rows": int(contract.get("source_family_rows", 0) or 0),
        "blocker": blocker,
        "frozen_family_replayed": bool(receipt and receipt.get("frozen_family_replayed")),
        "conclusion_status": str((receipt or {}).get("conclusion_status", "INCOMPLETE")),
        "funding_refresh_requested_for_invocation": bool(audit.get("funding_refresh_requested")),
        "funding_cost_evidence_stage_run_for_invocation": bool(
            audit.get("funding_cost_evidence_stage_run")
        ),
        "funding_refresh_dispatched_to_cost_stage_for_invocation": bool(
            audit.get("funding_refresh_dispatched_to_cost_stage")
        ),
        "funding_assets_fetched_from_network_for_invocation": int(
            audit.get("funding_assets_fetched_from_network", 0) or 0
        ),
        "funding_assets_reused_from_cache_for_invocation": int(
            audit.get("funding_assets_reused_from_cache", 0) or 0
        ),
        "funding_refresh_outcome_for_invocation": str(
            audit.get("funding_refresh_outcome", "NOT_REACHED")
        ),
        "authenticated_wizard_capture_performed": False,
        "order_submission_performed": False,
        "promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
        **(extra or {}),
    }
    paths = {"status": status_path}
    if receipt_path is not None:
        summary["execution_receipt_path"] = _relative(receipt_path, root)
        summary["execution_receipt_sha256"] = _file_hash(receipt_path)
        paths["execution_receipt"] = receipt_path
    _atomic_json(summary, status_path)
    return CommandResult(paths=paths, summary=summary)


def _publish_stage5_blocked_status(
    *,
    root: Path,
    result: CommandResult,
    requested_at: datetime,
    execute: bool,
    status: str,
    blocker: str,
    contract: dict[str, Any],
    receipt: dict[str, Any] | None = None,
) -> None:
    status_path = root / "reports" / "active" / "registered_learning_research_status.json"
    payload = {
        "schema_version": LEARNING_STATUS_SCHEMA_VERSION,
        "generated_at_utc": requested_at.isoformat(),
        "status": status,
        "execution_requested": execute,
        "blocker": blocker,
        "registered_contract_id": str(contract.get("contract_id", "")),
        "registered_execution_id": str((receipt or {}).get("execution_id", "")),
        "registered_conclusion_status": str((receipt or {}).get("conclusion_status", "INCOMPLETE")),
        "accepted_registered_hypotheses": int(
            (receipt or {}).get("accepted_registered_hypotheses", 0) or 0
        ),
        "stage5_research_gate_pass": False,
        "order_submission_performed": False,
        "promotion_authority": False,
        "testnet_candidate_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    _atomic_json(payload, status_path)
    result.paths["registered_learning_status"] = status_path
    result.summary["learning_handoff_status"] = status
    result.summary["stage5_research_gate_pass"] = False
    result.summary["learning_handoff_blocker"] = blocker


def _experiment_count(summary: dict[str, Any]) -> int:
    for key in (
        "experiments_accounted",
        "experiment_authority_count",
        "unique_experiment_ids",
        "experiments",
    ):
        if key in summary:
            return int(summary.get(key, 0) or 0)
    return 0


def _stage_identity(summary: dict[str, Any]) -> str:
    for key, value in summary.items():
        if key.endswith("_id") and key not in {"schema_version"} and str(value).strip():
            return str(value).strip()
    return ""


class ResearchAuthorityViolation(ValueError):
    """A project-owned research result attempted to claim execution authority."""

    def __init__(self, unsafe_fields: list[str]):
        reason_code = "unexpected_research_authority:" + ",".join(sorted(unsafe_fields))
        super().__init__(reason_code)
        self.evidence_reason_code = reason_code


def _assert_no_order_authority(name: str, payload: dict[str, Any]) -> None:
    forbidden = (
        "execution_authority",
        "order_submission_authority",
        "order_submission_performed",
        "promotion_authority",
        "testnet_candidate_authority",
        "testnet_order_authority",
        "live_trading_authorized",
    )
    unsafe = [key for key in forbidden if _truthy(payload.get(key))]
    if unsafe:
        raise ResearchAuthorityViolation(unsafe)


def _write_immutable_json(payload: dict[str, Any], path: Path) -> None:
    existing = _read_json(path)
    if existing and existing != payload:
        raise ValueError("registered rerun immutable execution receipt mismatch")
    _atomic_json(payload, path)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    promote_staged_file(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, keep_default_na=False)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame()


def _file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest() if path.is_file() else ""


def _payload_hash(payload: dict[str, Any]) -> str:
    material = dict(payload)
    material.pop("receipt_sha256", None)
    return sha256(_canonical_json(material).encode("utf-8")).hexdigest()


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _truthy(value: Any) -> bool:
    return value is True or str(value).strip().lower() in {"1", "true", "yes", "y"}


def _text(value: Any) -> str:
    return str(value or "").strip()


def _required_text(value: Any, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"registered rerun {name} is missing")
    return text


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _as_utc(value: datetime | None) -> datetime:
    value = value or datetime.now(UTC)
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--fetch-funding", action="store_true")
    args = parser.parse_args()
    result = run_registered_research_rerun(
        execute=args.execute,
        fetch_funding=args.fetch_funding,
    )
    print(json.dumps(result.summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
