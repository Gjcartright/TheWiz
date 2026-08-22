"""Equivalence, near-miss, breadth, and final-survivor corrective controls."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd

from quant_platform.active_pipeline import CommandResult
from quant_platform.orchestration.corrective_governance import (
    build_acceptance_policy_receipt,
    build_holdout_policy_receipt,
    load_acceptance_policy,
    semantic_hypothesis_id,
)
from quant_platform.orchestration.corrective_registered_rerun import (
    build_registered_rerun_gate,
)
from quant_platform.orchestration.corrective_runtime import atomic_write_text, promote_staged_file
from quant_platform.orchestration.corrective_stage4_handoff_readiness import (
    build_corrective_stage4_handoff_readiness,
)

ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "thewiz.corrective_statistical_remediation.v1"


def build_strategy_equivalence_clusters(*, root: Path = ROOT) -> dict[str, Any]:
    active = root / "reports" / "active"
    candidates = _read_csv(active / "current_wizard_hyperliquid_walkforward_candidates.csv")
    candidates = candidates.loc[
        candidates.get("walkforward_status", pd.Series(dtype=str)).eq("PASS_RESEARCH_WALK_FORWARD")
    ].copy()
    trades = _read_csv(active / "current_wizard_hyperliquid_walkforward_trades.csv")
    ids = list(candidates.get("experiment_id", pd.Series(dtype=str)).astype(str))
    parent = {value: value for value in ids}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        lroot, rroot = find(left), find(right)
        if lroot != rroot:
            parent[max(lroot, rroot)] = min(lroot, rroot)

    trade_sets = {
        experiment_id: {
            (_text(row.entry_timestamp), _text(row.exit_timestamp), _text(row.side))
            for row in trades.loc[
                trades.get("experiment_id", pd.Series(dtype=str)).astype(str).eq(experiment_id)
            ].itertuples()
        }
        for experiment_id in ids
    }
    maximum_overlap: dict[str, float] = {value: 0.0 for value in ids}
    for left_index, left in enumerate(ids):
        left_row = candidates.loc[candidates["experiment_id"].astype(str).eq(left)].iloc[0]
        for right in ids[left_index + 1 :]:
            right_row = candidates.loc[candidates["experiment_id"].astype(str).eq(right)].iloc[0]
            comparable = (
                _canonical_pair(left_row) == _canonical_pair(right_row)
                and _text(left_row.get("wizard_timeframe"))
                == _text(right_row.get("wizard_timeframe"))
                and _text(left_row.get("orientation")) == _text(right_row.get("orientation"))
            )
            if not comparable:
                continue
            overlap = _jaccard(trade_sets[left], trade_sets[right])
            maximum_overlap[left] = max(maximum_overlap[left], overlap)
            maximum_overlap[right] = max(maximum_overlap[right], overlap)
            if overlap >= 0.95:
                union(left, right)
    clusters: dict[str, list[str]] = {}
    for experiment_id in ids:
        clusters.setdefault(find(experiment_id), []).append(experiment_id)
    cluster_ids = {
        member: "eq_" + sha256("|".join(sorted(members)).encode("utf-8")).hexdigest()[:20]
        for members in clusters.values()
        for member in members
    }
    rows = []
    for row in candidates.to_dict("records"):
        experiment_id = _text(row.get("experiment_id"))
        cluster_members = clusters[find(experiment_id)]
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "equivalence_cluster_id": cluster_ids[experiment_id],
                "experiment_id": experiment_id,
                "pair": _text(row.get("pair")),
                "canonical_pair": _canonical_pair(row),
                "wizard_timeframe": _text(row.get("wizard_timeframe")),
                "exact_mode": _text(row.get("exact_mode")),
                "orientation": _text(row.get("orientation")),
                "formula_lineage": _formula_lineage(row.get("exact_mode")),
                "maximum_trade_overlap": maximum_overlap[experiment_id],
                "cluster_member_count": len(cluster_members),
                "independent_evidence_weight": 1.0 / len(cluster_members),
                "raw_row_counts_as_independent": len(cluster_members) == 1,
                "signal_correlation_status": "not_materialized_trade_overlap_used",
                "evidence_path": "reports/active/current_wizard_hyperliquid_walkforward_trades.csv",
                "promotion_authority": False,
                "live_trading_authorized": False,
            }
        )
    frame = pd.DataFrame(rows)
    path = active / "strategy_equivalence_clusters.csv"
    _atomic_csv(frame, path)
    attacks = _duplicate_mode_attacks(frame, root=root)
    return {
        "path": path,
        "frame": frame,
        "experiments": len(frame),
        "clusters": frame.get("equivalence_cluster_id", pd.Series(dtype=str)).nunique(),
        "duplicate_rows": int(frame.get("cluster_member_count", pd.Series(dtype=int)).gt(1).sum()),
        "attacks": attacks,
        "status": "PASS" if not frame.empty and attacks["status"].eq("PASS").all() else "BLOCKED",
    }


def build_near_miss_queue(*, root: Path = ROOT, now: datetime | None = None) -> dict[str, Any]:
    active = root / "reports" / "active"
    clusters = build_strategy_equivalence_clusters(root=root)["frame"]
    failure = _read_csv(active / "current_wizard_hyperliquid_failure_attribution.csv")
    required_clusters = _minimum_independent_clusters(root)
    selected = _build_registered_candidate_cohort(
        failure,
        clusters,
        required_pairs=required_clusters,
    )
    selected = _overlay_registered_pair_cost_evidence(
        selected,
        _read_csv(root / "data" / "processed" / "hyperliquid_pair_cost_models.csv"),
        now=now,
    )
    acceptance = build_acceptance_policy_receipt(root=root, now=now)
    holdout = build_holdout_policy_receipt(root=root, now=now)
    selected["semantic_hypothesis_id"] = selected.apply(
        lambda row: semantic_hypothesis_id(
            row.to_dict(),
            policy_id=str(acceptance["policy_id"]),
            holdout_policy_id=str(holdout["policy_id"]),
        ),
        axis=1,
    )
    selected["missing_proof_diagnosis"] = selected.apply(_missing_proof, axis=1)
    selected["all_missing_proofs"] = selected.apply(
        lambda row: ";".join(_missing_proofs(row)), axis=1
    )
    selected["missing_proof_count"] = selected.apply(lambda row: len(_missing_proofs(row)), axis=1)
    selected["prospective_next_test"] = selected["missing_proof_diagnosis"].map(_next_test)
    selected["hypothesis_registered_before_next_test"] = True
    selected["next_test_executed"] = False
    selected["execution_authority"] = False
    selected["live_trading_authorized"] = False
    selected = selected.sort_values(["overall_research_rank", "experiment_id"])
    path = active / "walkforward_near_miss_queue.csv"
    _atomic_csv(selected, path)
    markdown = active / "walkforward_near_miss_queue.md"
    atomic_write_text(markdown, _near_miss_markdown(selected, acceptance, holdout), encoding="utf-8")
    batch_columns = [
        "semantic_hypothesis_id",
        "experiment_id",
        "equivalence_cluster_id",
        "pair",
        "wizard_timeframe",
        "exact_mode",
        "orientation",
        "missing_proof_diagnosis",
        "all_missing_proofs",
        "missing_proof_count",
        "prospective_next_test",
        "registration_cohort_role",
        "acceptance_evidence_weight",
        "hypothesis_registered_before_next_test",
        "next_test_executed",
        "execution_authority",
        "live_trading_authorized",
    ]
    batch = selected.reindex(columns=batch_columns).copy()
    batch["confirmation_role"] = batch.apply(
        lambda row: (
            "independent_eth_wif_copula_confirmation"
            if _is_eth_wif_copula(row)
            else (
                "prospective_current_family_breadth"
                if _text(row.get("registration_cohort_role"))
                == "prospective_current_family_breadth"
                else "near_miss_remediation"
            )
        ),
        axis=1,
    )
    batch["batch_status"] = "REGISTERED_BLOCKED_PENDING_NEW_EVIDENCE"
    batch_path = active / "current_hypothesis_batch.csv"
    _atomic_csv(batch, batch_path)
    return {
        "queue": path,
        "markdown": markdown,
        "batch": batch_path,
        "frame": selected,
        "candidates": len(selected),
        "required_independent_pairs": required_clusters,
        "independent_candidate_pairs": int(selected.apply(_canonical_pair, axis=1).nunique()),
        "status": (
            "PASS"
            if not selected.empty
            and selected["hypothesis_registered_before_next_test"].all()
            and selected.apply(_canonical_pair, axis=1).nunique() >= required_clusters
            else "BLOCKED"
        ),
    }


def _build_registered_candidate_cohort(
    failure: pd.DataFrame,
    clusters: pd.DataFrame,
    *,
    required_pairs: int,
) -> pd.DataFrame:
    """Add current-family candidates without treating them as accepted evidence."""

    if failure.empty:
        return failure.copy()
    cluster_columns = [
        "experiment_id",
        "equivalence_cluster_id",
        "cluster_member_count",
        "independent_evidence_weight",
    ]
    available_clusters = clusters.reindex(columns=cluster_columns).copy()
    cluster_ids = set(available_clusters.get("experiment_id", pd.Series(dtype=str)).astype(str))
    selected = failure.loc[
        failure.get("experiment_id", pd.Series(dtype=str)).astype(str).isin(cluster_ids)
    ].copy()
    if not selected.empty:
        selected = selected.merge(
            available_clusters,
            on="experiment_id",
            how="left",
            validate="one_to_one",
        )
        selected["registration_cohort_role"] = "walkforward_near_miss"
        selected["acceptance_evidence_weight"] = selected["independent_evidence_weight"]

    selected_pairs = {
        _canonical_pair(row) for row in selected.to_dict("records") if _canonical_pair(row) != "-"
    }
    eligible = failure.loc[
        failure.get("matrix_status", pd.Series(dtype=str)).eq("READY_FOR_POINT_IN_TIME_HISTORY")
        & failure.get("canonical_replay_status", pd.Series(dtype=str)).eq(
            "RESEARCH_REPLAY_COMPLETE"
        )
    ].copy()
    eligible["_rank"] = pd.to_numeric(
        eligible.get("overall_research_rank", pd.Series(index=eligible.index)),
        errors="coerce",
    )
    eligible["_canonical_pair"] = eligible.apply(_canonical_pair, axis=1)
    eligible = eligible.loc[
        eligible["_canonical_pair"].ne("-")
        & eligible.get("pair_group_key", pd.Series("", index=eligible.index))
        .astype(str)
        .str.len()
        .gt(0)
    ].sort_values(["_rank", "experiment_id"], na_position="last")

    prospective_rows: list[dict[str, Any]] = []
    for row in eligible.to_dict("records"):
        canonical_pair = _text(row.get("_canonical_pair"))
        if canonical_pair in selected_pairs:
            continue
        pair_group_key = _text(row.get("pair_group_key"))
        record = {key: value for key, value in row.items() if not key.startswith("_")}
        record.update(
            {
                "equivalence_cluster_id": "prospective_pair_"
                + sha256(pair_group_key.encode("utf-8")).hexdigest()[:20],
                "cluster_member_count": 1,
                "independent_evidence_weight": 0.0,
                "registration_cohort_role": "prospective_current_family_breadth",
                "acceptance_evidence_weight": 0.0,
            }
        )
        prospective_rows.append(record)
        selected_pairs.add(canonical_pair)
        if len(selected_pairs) >= required_pairs:
            break

    if prospective_rows:
        selected = pd.concat(
            [selected, pd.DataFrame(prospective_rows)],
            ignore_index=True,
            sort=False,
        )
    return selected.drop_duplicates("experiment_id", keep="first")


def build_corrective_research_funnel(
    *, root: Path = ROOT, now: datetime | None = None
) -> dict[str, Any]:
    near_miss = build_near_miss_queue(root=root, now=now)
    frame = near_miss["frame"].copy()
    frame["corrective_run_status"] = "NOT_RERUN_UNCHANGED_OR_BLOCKED_EVIDENCE"
    frame["identity_match"] = True
    frame["new_history_ready"] = ~frame["all_independent_blockers"].fillna("").astype(
        str
    ).str.contains("history", case=False)
    frame["strict_observed_cost_ready"] = frame.get(
        "strict_cost_calibration_ready", pd.Series(False, index=frame.index)
    ).map(_truthy)
    frame["vendor_parity_ready"] = frame.get(
        "vendor_exact_mode_parity_proven", pd.Series(False, index=frame.index)
    ).map(_truthy)
    frame["all_current_research_gates_pass"] = frame.get(
        "one_x_research_survivor", pd.Series(False, index=frame.index)
    ).map(_truthy)
    frame["rerun_blocker"] = frame.apply(_rerun_blocker, axis=1)
    frame["live_trading_authorized"] = False
    path = root / "reports" / "active" / "corrective_research_funnel.csv"
    _atomic_csv(frame, path)
    md = root / "reports" / "active" / "corrective_research_funnel.md"
    atomic_write_text(md, _funnel_markdown(frame), encoding="utf-8")
    return {"path": path, "markdown": md, "frame": frame}


def build_independent_strategy_breadth(
    *, root: Path = ROOT, now: datetime | None = None
) -> dict[str, Any]:
    funnel = build_corrective_research_funnel(root=root, now=now)["frame"]
    required_clusters = _minimum_independent_clusters(root)
    passed = funnel.loc[
        funnel.get("all_current_research_gates_pass", pd.Series(False, index=funnel.index)).map(
            _truthy
        )
    ]
    selected = funnel.loc[
        funnel.get("statistical_selection_status", pd.Series(dtype=str)).eq("PASS")
        & funnel.get("research_robustness_status", pd.Series(dtype=str)).eq(
            "PASS_RESEARCH_ROBUSTNESS"
        )
        & funnel.get("regime_stability_status", pd.Series(dtype=str)).eq(
            "PASS_RESEARCH_REGIME_STABILITY"
        )
    ]
    supporting_clusters = selected.get("equivalence_cluster_id", pd.Series(dtype=str)).nunique()
    full_survivor_clusters = passed.get("equivalence_cluster_id", pd.Series(dtype=str)).nunique()
    supporting_pairs = _distinct_canonical_pairs(selected)
    full_survivor_pairs = _distinct_canonical_pairs(passed)
    rows = [
        {"metric": "required_independent_clusters", "value": required_clusters, "status": "POLICY"},
        {
            "metric": "statistical_regime_robust_clusters",
            "value": supporting_clusters,
            "status": "PASS" if supporting_clusters >= required_clusters else "BLOCKED",
        },
        {
            "metric": "full_one_x_survivor_clusters",
            "value": full_survivor_clusters,
            "status": "PASS" if full_survivor_clusters >= required_clusters else "BLOCKED",
        },
        {
            "metric": "statistical_regime_robust_distinct_pairs",
            "value": supporting_pairs,
            "status": "PASS" if supporting_pairs >= required_clusters else "BLOCKED",
        },
        {
            "metric": "full_one_x_survivor_distinct_pairs",
            "value": full_survivor_pairs,
            "status": "PASS" if full_survivor_pairs >= required_clusters else "BLOCKED",
        },
        {"metric": "raw_candidate_rows", "value": len(funnel), "status": "DIAGNOSTIC_ONLY"},
        {
            "metric": "renamed_or_duplicate_rows",
            "value": int(funnel.get("cluster_member_count", pd.Series(dtype=int)).gt(1).sum()),
            "status": "CONTROLLED",
        },
        {"metric": "live_trading_authorized", "value": False, "status": "BLOCKED"},
    ]
    frame = pd.DataFrame(rows)
    path = root / "reports" / "active" / "independent_strategy_breadth.csv"
    _atomic_csv(frame, path)
    return {
        "path": path,
        "supporting_clusters": supporting_clusters,
        "full_survivor_clusters": int(full_survivor_clusters),
        "supporting_pairs": supporting_pairs,
        "full_survivor_pairs": full_survivor_pairs,
        "status": (
            "PASS"
            if supporting_clusters >= required_clusters
            and full_survivor_clusters >= required_clusters
            and supporting_pairs >= required_clusters
            and full_survivor_pairs >= required_clusters
            else "BLOCKED"
        ),
    }


def issue_final_one_x_survivor_receipt(
    *, root: Path = ROOT, now: datetime | None = None
) -> dict[str, Any]:
    funnel_result = build_corrective_research_funnel(root=root, now=now)
    funnel = funnel_result["frame"]
    breadth = build_independent_strategy_breadth(root=root, now=now)
    survivors = funnel.loc[
        funnel.get("one_x_research_survivor", pd.Series(False, index=funnel.index)).map(_truthy)
    ]
    strict = survivors.loc[
        survivors.get("strict_cost_calibration_ready", pd.Series(False, index=survivors.index)).map(
            _truthy
        )
    ]
    parity = strict.loc[
        strict.get("vendor_exact_mode_parity_proven", pd.Series(False, index=strict.index)).map(
            _truthy
        )
    ]
    final = parity if breadth["status"] == "PASS" else parity.iloc[0:0]
    blockers = []
    if survivors.empty:
        blockers.append("zero_current_one_x_research_survivors")
    if strict.empty:
        blockers.append("zero_strict_observed_cost_survivors")
    if parity.empty:
        blockers.append("zero_vendor_parity_proven_survivors")
    if breadth["status"] != "PASS":
        blockers.append("independent_strategy_breadth_below_three_clusters_or_pairs")
    acceptance = build_acceptance_policy_receipt(root=root, now=now)
    holdout = build_holdout_policy_receipt(root=root, now=now)
    receipt = {
        "schema_version": "thewiz.final_one_x_survivor_receipt.v1",
        "generated_at_utc": _as_utc(now).isoformat(),
        "receipt_status": "PASS" if not final.empty and not blockers else "ZERO_SURVIVORS",
        "acceptance_policy_id": acceptance["policy_id"],
        "holdout_policy_id": holdout["policy_id"],
        "near_miss_candidates": len(funnel),
        "current_one_x_research_survivors": len(survivors),
        "strict_cost_survivors": len(strict),
        "vendor_parity_survivors": len(parity),
        "independent_supporting_clusters": breadth["supporting_clusters"],
        "independent_full_survivor_clusters": breadth["full_survivor_clusters"],
        "independent_supporting_pairs": breadth["supporting_pairs"],
        "independent_full_survivor_pairs": breadth["full_survivor_pairs"],
        "final_one_x_survivors": len(final),
        "final_experiment_ids": list(final.get("experiment_id", pd.Series(dtype=str)).astype(str)),
        "final_canonical_pairs": sorted(
            {
                pair
                for pair in final.apply(_canonical_pair, axis=1).astype(str)
                if pair not in {"", "-"}
            }
        )
        if not final.empty
        else [],
        "blockers": blockers,
        "thresholds_changed_after_results": False,
        "testnet_candidate_authority": bool(not final.empty and not blockers),
        "testnet_order_authority": False,
        "live_trading_authorized": False,
        "evidence_path": "reports/active/corrective_research_funnel.csv;reports/active/independent_strategy_breadth.csv",
    }
    path = root / "reports" / "active" / "final_1x_survivor_receipt.json"
    _atomic_json(receipt, path)
    registered_bridge = _registered_execution_survivor_bridge(root=root)
    if registered_bridge:
        source = Path(registered_bridge["path"])
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(source.read_bytes())
        promote_staged_file(temporary, path)
        receipt = dict(registered_bridge["receipt"])
    return {"path": path, "receipt": receipt}


def build_corrective_statistical_remediation(
    *, root: Path = ROOT, now: datetime | None = None
) -> CommandResult:
    clusters = build_strategy_equivalence_clusters(root=root)
    queue = build_near_miss_queue(root=root, now=now)
    funnel = build_corrective_research_funnel(root=root, now=now)
    breadth = build_independent_strategy_breadth(root=root, now=now)
    receipt = issue_final_one_x_survivor_receipt(root=root, now=now)
    rerun_gate = build_registered_rerun_gate(root=root, now=now)
    stage4_readiness = build_corrective_stage4_handoff_readiness(root=root, now=now)
    paths = {
        "equivalence_clusters": Path(clusters["path"]),
        "duplicate_mode_attacks": root
        / "reports"
        / "red_team"
        / "duplicate_mode_attack_results.csv",
        "near_miss_queue": Path(queue["queue"]),
        "near_miss_markdown": Path(queue["markdown"]),
        "hypothesis_batch": Path(queue["batch"]),
        "research_funnel": Path(funnel["path"]),
        "research_funnel_markdown": Path(funnel["markdown"]),
        "independent_breadth": Path(breadth["path"]),
        "final_survivor_receipt": Path(receipt["path"]),
        "registered_rerun_contract": Path(rerun_gate.paths["contract"]),
        "registered_rerun_gate": Path(rerun_gate.paths["gate"]),
        "registered_rerun_gate_summary": Path(rerun_gate.paths["summary"]),
        "registered_rerun_family_preflight": Path(rerun_gate.paths["family_preflight"]),
        "registered_rerun_family_preflight_summary": Path(
            rerun_gate.paths["family_preflight_summary"]
        ),
        "stage4_handoff_readiness_checks": Path(stage4_readiness.paths["checks"]),
        "stage4_handoff_readiness": Path(stage4_readiness.paths["status"]),
        "stage4_handoff_readiness_summary": Path(stage4_readiness.paths["summary"]),
        "stage4_handoff_readiness_receipt": Path(
            stage4_readiness.paths["immutable_receipt"]
        ),
    }
    if "ready_receipt" in rerun_gate.paths:
        paths["registered_rerun_ready_receipt"] = Path(rerun_gate.paths["ready_receipt"])
    if "conclusion_receipt" in rerun_gate.paths:
        paths["registered_rerun_conclusion_receipt"] = Path(rerun_gate.paths["conclusion_receipt"])
    return CommandResult(
        paths=paths,
        summary={
            "status": "PASS" if receipt["receipt"]["receipt_status"] == "PASS" else "BLOCKED",
            "near_miss_candidates": queue["candidates"],
            "equivalence_clusters": clusters["clusters"],
            "duplicate_candidate_rows": clusters["duplicate_rows"],
            "independent_supporting_clusters": int(
                receipt["receipt"].get(
                    "independent_supporting_clusters", breadth["supporting_clusters"]
                )
                or 0
            ),
            "independent_full_survivor_clusters": int(
                receipt["receipt"].get(
                    "independent_full_survivor_clusters",
                    breadth["full_survivor_clusters"],
                )
                or 0
            ),
            "independent_supporting_pairs": int(
                receipt["receipt"].get(
                    "independent_supporting_pairs", breadth["supporting_pairs"]
                )
                or 0
            ),
            "independent_full_survivor_pairs": int(
                receipt["receipt"].get(
                    "independent_full_survivor_pairs",
                    breadth["full_survivor_pairs"],
                )
                or 0
            ),
            "final_one_x_survivors": receipt["receipt"]["final_one_x_survivors"],
            "registered_rerun_gate_status": rerun_gate.summary["status"],
            "registered_rerun_candidates": rerun_gate.summary["registered_candidates"],
            "registered_rerun_candidate_gates_ready": rerun_gate.summary["candidate_gates_ready"],
            "pending_family_preflight_status": rerun_gate.summary[
                "pending_family_preflight_status"
            ],
            "pending_family_hypotheses": rerun_gate.summary[
                "pending_family_hypotheses"
            ],
            "pending_family_ready": rerun_gate.summary["pending_family_ready"],
            "pending_family_pairs": rerun_gate.summary["pending_family_pairs"],
            "pending_family_independent_clusters": rerun_gate.summary[
                "pending_family_independent_clusters"
            ],
            "pending_family_rollover_required": rerun_gate.summary[
                "pending_family_rollover_required"
            ],
            "registered_rerun_results_accounted": rerun_gate.summary[
                "registered_rerun_results_accounted"
            ],
            "registered_rerun_conclusion_status": rerun_gate.summary[
                "registered_rerun_conclusion_status"
            ],
            "accepted_registered_hypotheses": rerun_gate.summary["accepted_registered_hypotheses"],
            "rejected_registered_hypotheses": rerun_gate.summary["rejected_registered_hypotheses"],
            "registered_rerun_next_action": rerun_gate.summary["next_action"],
            "stage4_handoff_readiness_status": stage4_readiness.summary["status"],
            "stage4_handoff_state": stage4_readiness.summary["handoff_state"],
            "stage4_handoff_receipt_id": stage4_readiness.summary["receipt_id"],
            "stage4_handoff_checked_at_utc": stage4_readiness.summary["checked_at_utc"],
            "stage4_handoff_checks_passed": stage4_readiness.summary["checks_passed"],
            "stage4_handoff_checks_total": stage4_readiness.summary["checks_total"],
            "stage4_handoff_blockers": stage4_readiness.summary["blockers"],
            "blockers": receipt["receipt"]["blockers"],
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        },
    )


def _registered_execution_survivor_bridge(*, root: Path) -> dict[str, Any]:
    """Return the hash-bound survivor receipt from the active registered execution."""

    active = root / "reports" / "active"
    status = _read_json_fail_closed(active / "registered_research_rerun_execution.json")
    receipt_path = _safe_root_artifact(
        root, str(status.get("execution_receipt_path", ""))
    )
    expected_receipt_file_hash = str(status.get("execution_receipt_sha256", ""))
    if (
        receipt_path is None
        or not receipt_path.is_file()
        or len(expected_receipt_file_hash) != 64
        or _file_hash(receipt_path) != expected_receipt_file_hash
    ):
        return {}
    execution = _read_json_fail_closed(receipt_path)
    embedded_hash = str(execution.pop("receipt_sha256", ""))
    if embedded_hash != sha256(
        json.dumps(
            execution,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest():
        return {}
    execution["receipt_sha256"] = embedded_hash
    active_contract = _read_json_fail_closed(
        active / "registered_research_rerun_contract.json"
    )
    if (
        execution.get("status") != "PASS_REGISTERED_RERUN_ACCOUNTED"
        or execution.get("contract_id") != active_contract.get("contract_id")
        or status.get("contract_id") != active_contract.get("contract_id")
        or _truthy(execution.get("order_submission_performed"))
        or _truthy(execution.get("testnet_order_authority"))
        or _truthy(execution.get("live_trading_authorized"))
    ):
        return {}
    conclusion_path = _safe_root_artifact(
        root, str(execution.get("conclusion_path", ""))
    )
    survivor_path = _safe_root_artifact(
        root, str(execution.get("stage4_survivor_receipt_path", ""))
    )
    if (
        conclusion_path is None
        or survivor_path is None
        or _file_hash(conclusion_path) != str(execution.get("conclusion_sha256", ""))
        or _file_hash(survivor_path)
        != str(execution.get("stage4_survivor_receipt_sha256", ""))
    ):
        return {}
    conclusion = _read_json_fail_closed(conclusion_path)
    survivor = _read_json_fail_closed(survivor_path)
    final_ids = survivor.get("final_experiment_ids", [])
    try:
        final_survivors = int(survivor.get("final_one_x_survivors", -1))
        independent_clusters = int(
            survivor.get("independent_supporting_clusters", -1)
        )
    except (TypeError, ValueError):
        return {}
    if (
        conclusion.get("contract_id") != execution.get("contract_id")
        or conclusion.get("final_survivor_receipt_sha256") != _file_hash(survivor_path)
        or survivor.get("schema_version") != "thewiz.final_one_x_survivor_receipt.v1"
        or survivor.get("acceptance_policy_id")
        != execution.get("acceptance_policy_id")
        or survivor.get("holdout_policy_id") != execution.get("holdout_policy_id")
        or not isinstance(final_ids, list)
        or len(final_ids) != final_survivors
        or final_survivors
        != int(execution.get("stage4_final_one_x_survivors", -1))
        or independent_clusters
        != int(execution.get("stage4_independent_supporting_clusters", -1))
        or list(execution.get("stage4_final_experiment_ids", [])) != final_ids
        or _truthy(survivor.get("testnet_order_authority"))
        or _truthy(survivor.get("live_trading_authorized"))
    ):
        return {}
    return {
        "path": survivor_path,
        "receipt": survivor,
        "execution_receipt_path": receipt_path,
        "execution_id": str(execution.get("execution_id", "")),
    }


def _duplicate_mode_attacks(frame: pd.DataFrame, *, root: Path) -> pd.DataFrame:
    duplicate_rows = frame.loc[frame.get("cluster_member_count", pd.Series(dtype=int)).gt(1)]
    rows = [
        {
            "case": "renamed_duplicate_mode",
            "duplicate_rows_detected": len(duplicate_rows),
            "raw_rows_counted_as_independent": bool(
                not duplicate_rows.empty and duplicate_rows["raw_row_counts_as_independent"].any()
            ),
            "effective_cluster_count": frame.get(
                "equivalence_cluster_id", pd.Series(dtype=str)
            ).nunique(),
            "status": "PASS"
            if duplicate_rows.empty or not duplicate_rows["raw_row_counts_as_independent"].any()
            else "FAIL",
            "live_trading_authorized": False,
        }
    ]
    result = pd.DataFrame(rows)
    path = root / "reports" / "red_team" / "duplicate_mode_attack_results.csv"
    _atomic_csv(result, path)
    return result


def _missing_proof(row: pd.Series) -> str:
    proofs = _missing_proofs(row)
    return proofs[0] if proofs else "independent_future_holdout_confirmation"


def _overlay_registered_pair_cost_evidence(
    frame: pd.DataFrame,
    models: pd.DataFrame,
    *,
    now: datetime | None = None,
) -> pd.DataFrame:
    """Overlay current strict-cost evidence without rewriting source results."""

    if frame.empty:
        return frame.copy()
    as_of = _as_utc(now)
    if (
        not models.empty
        and not models.get("pair_group_key", pd.Series(dtype=str)).astype(str).duplicated().any()
    ):
        model_by_pair = {
            _text(row.get("pair_group_key")): row
            for row in models.to_dict("records")
            if _text(row.get("pair_group_key"))
        }
    elif models.empty:
        model_by_pair = {}
    else:
        raise ValueError("registered pair cost models contain duplicate pair identities")
    rows: list[dict[str, Any]] = []
    for source in frame.to_dict("records"):
        row = dict(source)
        row["source_strict_cost_calibration_ready"] = _truthy(
            source.get("strict_cost_calibration_ready")
        )
        row["source_all_independent_blockers"] = _text(source.get("all_independent_blockers"))
        pair_key = _text(source.get("pair_group_key"))
        model = model_by_pair.get(pair_key)
        blockers: list[str] = []
        if model is None:
            blockers.append("registered_pair_cost_model_missing")
            model = {}
        model_pair = "-".join(sorted((_text(model.get("asset_x")), _text(model.get("asset_y")))))
        source_pair = _canonical_pair(source)
        identity_match = bool(model and model_pair == source_pair and source_pair != "-")
        if model and not identity_match:
            blockers.append("registered_pair_cost_identity_mismatch")
        model_at = pd.to_datetime(model.get("model_as_of_utc"), utc=True, errors="coerce")
        fresh = bool(
            pd.notna(model_at)
            and model_at <= pd.Timestamp(as_of)
            and pd.Timestamp(as_of) - model_at <= pd.Timedelta(hours=2)
        )
        if model and not fresh:
            blockers.append("registered_pair_cost_model_stale_or_future")
        hash_fields = (
            "candidate_set_sha256",
            "cost_status_sha256",
            "l2_samples_sha256",
            "fee_profile_sha256",
        )
        hashes_valid = bool(
            model and all(len(_text(model.get(field))) == 64 for field in hash_fields)
        )
        if model and not hashes_valid:
            blockers.append("registered_pair_cost_input_hash_missing")
        cost = pd.to_numeric(
            pd.Series([model.get("estimated_pair_round_trip_cost_bps")]),
            errors="coerce",
        ).iloc[0]
        cost_finite = bool(pd.notna(cost) and float(cost) >= 0)
        if model and not cost_finite:
            blockers.append("registered_pair_round_trip_cost_invalid")
        model_ready = bool(
            _truthy(model.get("strict_observed_cost_ready"))
            and _truthy(model.get("cost_acceptance_ready"))
            and _text(model.get("cost_model_status")) == "STRICT_OBSERVED"
        )
        if model and not model_ready:
            blockers.append(
                _text(model.get("cost_model_blocker")) or "registered_pair_cost_model_not_strict"
            )
        ready = bool(
            model
            and identity_match
            and fresh
            and hashes_valid
            and cost_finite
            and model_ready
            and not blockers
        )
        independent_blockers = [
            token.strip()
            for token in row["source_all_independent_blockers"].split(";")
            if token.strip() and not token.strip().startswith("execution_cost:")
        ]
        if not ready:
            independent_blockers.append(
                "execution_cost:" + (blockers[0] if blockers else "strict_cost_missing")
            )
        model_evidence = _text(model.get("evidence_path"))
        evidence_paths = [
            _text(source.get("evidence_path")),
            "data/processed/hyperliquid_pair_cost_models.csv",
            model_evidence,
        ]
        row.update(
            {
                "strict_cost_calibration_ready": ready,
                "strict_cost_overlay_status": "PASS" if ready else "BLOCKED",
                "strict_cost_overlay_blocker": ";".join(dict.fromkeys(blockers)),
                "strict_cost_identity_match": identity_match,
                "strict_cost_model_fresh": fresh,
                "strict_cost_input_hashes_valid": hashes_valid,
                "registered_pair_cost_model_id": _text(model.get("cost_model_id")),
                "registered_pair_cost_model_as_of_utc": _text(model.get("model_as_of_utc")),
                "observed_pair_round_trip_cost_bps": (float(cost) if cost_finite else math.nan),
                "all_independent_blockers": ";".join(dict.fromkeys(independent_blockers)),
                "evidence_path": ";".join(dict.fromkeys(path for path in evidence_paths if path)),
                "testnet_order_authority": False,
                "promotion_authority": False,
                "live_trading_authorized": False,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _missing_proofs(row: pd.Series) -> list[str]:
    proofs: list[str] = []
    for field, success, blocker in (
        ("strict_cost_calibration_ready", True, "strict_observed_cost_calibration"),
        ("vendor_exact_mode_parity_proven", True, "vendor_formula_parity"),
        ("statistical_selection_status", "PASS", "multiplicity_adjusted_selection"),
        ("regime_stability_status", "PASS_RESEARCH_REGIME_STABILITY", "regime_stability"),
        ("research_robustness_status", "PASS_RESEARCH_ROBUSTNESS", "parameter_and_cost_robustness"),
        ("concentration_gate_pass", True, "cross_cell_concentration"),
    ):
        value = row.get(field)
        passed = _truthy(value) if success is True else _text(value) == success
        if not passed:
            proofs.append(blocker)
    return proofs


def _next_test(diagnosis: str) -> str:
    return {
        "strict_observed_cost_calibration": "collect_12_plus_fresh_l2_samples_both_legs_and_rerun_cost_stress",
        "vendor_formula_parity": "capture_comparable_vendor_output_series_and_run_parity",
        "multiplicity_adjusted_selection": "retain_in_registered_family_wait_for_new_data_no_parameter_search",
        "regime_stability": "evaluate_registered_future_regime_window",
        "parameter_and_cost_robustness": "run_registered_bounded_sensitivity_on_new_evidence",
        "cross_cell_concentration": "seek_independent_pair_mode_regime_confirmation",
        "independent_future_holdout_confirmation": "unlock_once_registered_holdout_conditions_are_met",
    }.get(diagnosis, "diagnose_before_testing")


def _rerun_blocker(row: pd.Series) -> str:
    blockers = []
    if not _truthy(row.get("strict_cost_calibration_ready")):
        blockers.append("strict_observed_cost_evidence_missing")
    if not _truthy(row.get("vendor_exact_mode_parity_proven")):
        blockers.append("vendor_formula_parity_missing")
    blockers.append("prospective_new_evidence_not_yet_available")
    return ";".join(blockers)


def _canonical_pair(row: Any) -> str:
    if isinstance(row, pd.Series):
        x, y = _text(row.get("asset_x")), _text(row.get("asset_y"))
    else:
        x, y = _text(row.get("asset_x")), _text(row.get("asset_y"))
    return "-".join(sorted((x, y)))


def _distinct_canonical_pairs(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    pairs = frame.apply(_canonical_pair, axis=1)
    pairs = pairs.loc[~pairs.isin({"", "-"})]
    return int(pairs.nunique())


def _minimum_independent_clusters(root: Path) -> int:
    policy = load_acceptance_policy(root)
    value = policy.get("research_gates", {}).get("minimum_independent_supporting_clusters", 3)
    try:
        required = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("minimum independent cluster policy must be an integer") from error
    if required < 1:
        raise ValueError("minimum independent cluster policy must be positive")
    return required


def _formula_lineage(mode: Any) -> str:
    token = _text(mode).lower()
    if "copula" in token:
        return "copula_conditional_tail"
    if "dyn" in token:
        return "dynamic_hedge_ratio"
    if "ou" in token:
        return "ornstein_uhlenbeck"
    if "static" in token:
        return "static_hedge_ratio"
    return "unknown"


def _jaccard(left: set[Any], right: set[Any]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _is_eth_wif_copula(row: pd.Series) -> bool:
    pair = _text(row.get("pair"))
    return "ETH" in pair and "WIF" in pair and _text(row.get("exact_mode")) == "Copula"


def _near_miss_markdown(
    frame: pd.DataFrame, acceptance: dict[str, Any], holdout: dict[str, Any]
) -> str:
    return "\n".join(
        [
            "# Registered Walk-Forward Near Misses",
            "",
            f"- Candidates: `{len(frame)}`",
            f"- Acceptance policy: `{acceptance['policy_id']}`",
            f"- Holdout policy: `{holdout['policy_id']}`",
            "- Next tests executed: `false`",
            "- Testnet order authority: `false`",
            "- Live trading authorized: `false`",
            "",
            "Each candidate has one first missing-proof diagnosis. No parameter search or holdout reuse is permitted while the required new evidence is absent.",
            "",
        ]
    )


def _funnel_markdown(frame: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# Corrective Research Funnel",
            "",
            f"- Registered near misses: `{len(frame)}`",
            f"- Current 1x research survivors: `{int(frame.get('one_x_research_survivor', pd.Series(False, index=frame.index)).map(_truthy).sum())}`",
            "- Corrective rerun: `blocked until prospectively registered new history, strict cost, and parity evidence exists`",
            "- Threshold changes after results: `false`",
            "- Live trading authorized: `false`",
            "",
        ]
    )


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _read_json_fail_closed(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_root_artifact(root: Path, relative: str) -> Path | None:
    if not relative or Path(relative).is_absolute():
        return None
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return candidate


def _file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest() if path.is_file() else ""


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    promote_staged_file(temporary, path)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    promote_staged_file(temporary, path)


def _truthy(value: Any) -> bool:
    return value is True or str(value).strip().lower() in {"1", "true", "yes", "pass", "ready"}


def _text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def _as_utc(value: datetime | None) -> datetime:
    value = value or datetime.now(UTC)
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


if __name__ == "__main__":
    result = build_corrective_statistical_remediation()
    print(
        json.dumps(
            {
                "summary": result.summary,
                "paths": {key: str(value) for key, value in result.paths.items()},
            },
            indent=2,
        )
    )
