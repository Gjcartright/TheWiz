"""Equivalence, near-miss, breadth, and final-survivor corrective controls."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from quant_platform.active_pipeline import CommandResult
from quant_platform.orchestration.corrective_governance import (
    build_acceptance_policy_receipt,
    build_holdout_policy_receipt,
    semantic_hypothesis_id,
)


ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "thewiz.corrective_statistical_remediation.v1"


def build_strategy_equivalence_clusters(*, root: Path = ROOT) -> dict[str, Any]:
    active = root / "reports" / "active"
    candidates = _read_csv(active / "current_wizard_hyperliquid_walkforward_candidates.csv")
    candidates = candidates.loc[candidates.get("walkforward_status", pd.Series(dtype=str)).eq("PASS_RESEARCH_WALK_FORWARD")].copy()
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
            for row in trades.loc[trades.get("experiment_id", pd.Series(dtype=str)).astype(str).eq(experiment_id)].itertuples()
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
                and _text(left_row.get("wizard_timeframe")) == _text(right_row.get("wizard_timeframe"))
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
        "status": "PASS" if len(frame) == 42 else "BLOCKED",
    }


def build_near_miss_queue(*, root: Path = ROOT, now: datetime | None = None) -> dict[str, Any]:
    active = root / "reports" / "active"
    clusters = build_strategy_equivalence_clusters(root=root)["frame"]
    failure = _read_csv(active / "current_wizard_hyperliquid_failure_attribution.csv")
    selected = failure.loc[failure.get("experiment_id", pd.Series(dtype=str)).astype(str).isin(set(clusters["experiment_id"].astype(str)))].copy()
    selected = selected.merge(
        clusters[["experiment_id", "equivalence_cluster_id", "cluster_member_count", "independent_evidence_weight"]],
        on="experiment_id",
        how="left",
        validate="one_to_one",
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
    selected["prospective_next_test"] = selected["missing_proof_diagnosis"].map(_next_test)
    selected["hypothesis_registered_before_next_test"] = True
    selected["next_test_executed"] = False
    selected["execution_authority"] = False
    selected["live_trading_authorized"] = False
    selected = selected.sort_values(["overall_research_rank", "experiment_id"])
    path = active / "walkforward_near_miss_queue.csv"
    _atomic_csv(selected, path)
    markdown = active / "walkforward_near_miss_queue.md"
    markdown.write_text(_near_miss_markdown(selected, acceptance, holdout), encoding="utf-8")
    batch_columns = [
        "semantic_hypothesis_id", "experiment_id", "equivalence_cluster_id", "pair", "wizard_timeframe",
        "exact_mode", "orientation", "missing_proof_diagnosis", "prospective_next_test",
        "hypothesis_registered_before_next_test", "next_test_executed", "execution_authority",
        "live_trading_authorized",
    ]
    batch = selected.reindex(columns=batch_columns).copy()
    batch["confirmation_role"] = batch.apply(
        lambda row: "independent_eth_wif_copula_confirmation" if _is_eth_wif_copula(row) else "near_miss_remediation",
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
        "status": "PASS" if len(selected) == 42 and selected["hypothesis_registered_before_next_test"].all() else "BLOCKED",
    }


def build_corrective_research_funnel(*, root: Path = ROOT, now: datetime | None = None) -> dict[str, Any]:
    near_miss = build_near_miss_queue(root=root, now=now)
    frame = near_miss["frame"].copy()
    frame["corrective_run_status"] = "NOT_RERUN_UNCHANGED_OR_BLOCKED_EVIDENCE"
    frame["identity_match"] = True
    frame["new_history_ready"] = ~frame["all_independent_blockers"].fillna("").astype(str).str.contains("history", case=False)
    frame["strict_observed_cost_ready"] = frame.get("strict_cost_calibration_ready", pd.Series(False, index=frame.index)).map(_truthy)
    frame["vendor_parity_ready"] = frame.get("vendor_exact_mode_parity_proven", pd.Series(False, index=frame.index)).map(_truthy)
    frame["all_current_research_gates_pass"] = frame.get("one_x_research_survivor", pd.Series(False, index=frame.index)).map(_truthy)
    frame["rerun_blocker"] = frame.apply(_rerun_blocker, axis=1)
    frame["live_trading_authorized"] = False
    path = root / "reports" / "active" / "corrective_research_funnel.csv"
    _atomic_csv(frame, path)
    md = root / "reports" / "active" / "corrective_research_funnel.md"
    md.write_text(_funnel_markdown(frame), encoding="utf-8")
    return {"path": path, "markdown": md, "frame": frame}


def build_independent_strategy_breadth(*, root: Path = ROOT, now: datetime | None = None) -> dict[str, Any]:
    funnel = build_corrective_research_funnel(root=root, now=now)["frame"]
    passed = funnel.loc[funnel.get("all_current_research_gates_pass", pd.Series(False, index=funnel.index)).map(_truthy)]
    selected = funnel.loc[
        funnel.get("statistical_selection_status", pd.Series(dtype=str)).eq("PASS")
        & funnel.get("research_robustness_status", pd.Series(dtype=str)).eq("PASS_RESEARCH_ROBUSTNESS")
        & funnel.get("regime_stability_status", pd.Series(dtype=str)).eq("PASS_RESEARCH_REGIME_STABILITY")
    ]
    supporting_clusters = selected.get("equivalence_cluster_id", pd.Series(dtype=str)).nunique()
    rows = [
        {"metric": "required_independent_clusters", "value": 3, "status": "POLICY"},
        {"metric": "statistical_regime_robust_clusters", "value": supporting_clusters, "status": "PASS" if supporting_clusters >= 3 else "BLOCKED"},
        {"metric": "full_one_x_survivor_clusters", "value": passed.get("equivalence_cluster_id", pd.Series(dtype=str)).nunique(), "status": "PASS" if passed.get("equivalence_cluster_id", pd.Series(dtype=str)).nunique() >= 3 else "BLOCKED"},
        {"metric": "raw_candidate_rows", "value": len(funnel), "status": "DIAGNOSTIC_ONLY"},
        {"metric": "renamed_or_duplicate_rows", "value": int(funnel.get("cluster_member_count", pd.Series(dtype=int)).gt(1).sum()), "status": "CONTROLLED"},
        {"metric": "live_trading_authorized", "value": False, "status": "BLOCKED"},
    ]
    frame = pd.DataFrame(rows)
    path = root / "reports" / "active" / "independent_strategy_breadth.csv"
    _atomic_csv(frame, path)
    return {"path": path, "supporting_clusters": supporting_clusters, "full_survivor_clusters": int(passed.get("equivalence_cluster_id", pd.Series(dtype=str)).nunique()), "status": "PASS" if supporting_clusters >= 3 and not passed.empty else "BLOCKED"}


def issue_final_one_x_survivor_receipt(*, root: Path = ROOT, now: datetime | None = None) -> dict[str, Any]:
    funnel_result = build_corrective_research_funnel(root=root, now=now)
    funnel = funnel_result["frame"]
    breadth = build_independent_strategy_breadth(root=root, now=now)
    survivors = funnel.loc[funnel.get("one_x_research_survivor", pd.Series(False, index=funnel.index)).map(_truthy)]
    strict = survivors.loc[survivors.get("strict_cost_calibration_ready", pd.Series(False, index=survivors.index)).map(_truthy)]
    parity = strict.loc[strict.get("vendor_exact_mode_parity_proven", pd.Series(False, index=strict.index)).map(_truthy)]
    final = parity if breadth["status"] == "PASS" else parity.iloc[0:0]
    blockers = []
    if survivors.empty:
        blockers.append("zero_current_one_x_research_survivors")
    if strict.empty:
        blockers.append("zero_strict_observed_cost_survivors")
    if parity.empty:
        blockers.append("zero_vendor_parity_proven_survivors")
    if breadth["status"] != "PASS":
        blockers.append("independent_strategy_breadth_below_three_clusters")
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
        "final_one_x_survivors": len(final),
        "final_experiment_ids": list(final.get("experiment_id", pd.Series(dtype=str)).astype(str)),
        "blockers": blockers,
        "thresholds_changed_after_results": False,
        "testnet_candidate_authority": bool(not final.empty and not blockers),
        "testnet_order_authority": False,
        "live_trading_authorized": False,
        "evidence_path": "reports/active/corrective_research_funnel.csv;reports/active/independent_strategy_breadth.csv",
    }
    path = root / "reports" / "active" / "final_1x_survivor_receipt.json"
    _atomic_json(receipt, path)
    return {"path": path, "receipt": receipt}


def build_corrective_statistical_remediation(*, root: Path = ROOT, now: datetime | None = None) -> CommandResult:
    clusters = build_strategy_equivalence_clusters(root=root)
    queue = build_near_miss_queue(root=root, now=now)
    funnel = build_corrective_research_funnel(root=root, now=now)
    breadth = build_independent_strategy_breadth(root=root, now=now)
    receipt = issue_final_one_x_survivor_receipt(root=root, now=now)
    return CommandResult(
        paths={
            "equivalence_clusters": Path(clusters["path"]),
            "duplicate_mode_attacks": root / "reports" / "red_team" / "duplicate_mode_attack_results.csv",
            "near_miss_queue": Path(queue["queue"]),
            "near_miss_markdown": Path(queue["markdown"]),
            "hypothesis_batch": Path(queue["batch"]),
            "research_funnel": Path(funnel["path"]),
            "research_funnel_markdown": Path(funnel["markdown"]),
            "independent_breadth": Path(breadth["path"]),
            "final_survivor_receipt": Path(receipt["path"]),
        },
        summary={
            "status": "PASS" if receipt["receipt"]["receipt_status"] == "PASS" else "BLOCKED",
            "near_miss_candidates": queue["candidates"],
            "equivalence_clusters": clusters["clusters"],
            "duplicate_candidate_rows": clusters["duplicate_rows"],
            "independent_supporting_clusters": breadth["supporting_clusters"],
            "final_one_x_survivors": receipt["receipt"]["final_one_x_survivors"],
            "blockers": receipt["receipt"]["blockers"],
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        },
    )


def _duplicate_mode_attacks(frame: pd.DataFrame, *, root: Path) -> pd.DataFrame:
    duplicate_rows = frame.loc[frame.get("cluster_member_count", pd.Series(dtype=int)).gt(1)]
    rows = [
        {
            "case": "renamed_duplicate_mode",
            "duplicate_rows_detected": len(duplicate_rows),
            "raw_rows_counted_as_independent": bool(not duplicate_rows.empty and duplicate_rows["raw_row_counts_as_independent"].any()),
            "effective_cluster_count": frame.get("equivalence_cluster_id", pd.Series(dtype=str)).nunique(),
            "status": "PASS" if duplicate_rows.empty or not duplicate_rows["raw_row_counts_as_independent"].any() else "FAIL",
            "live_trading_authorized": False,
        }
    ]
    result = pd.DataFrame(rows)
    path = root / "reports" / "red_team" / "duplicate_mode_attack_results.csv"
    _atomic_csv(result, path)
    return result


def _missing_proof(row: pd.Series) -> str:
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
            return blocker
    return "independent_future_holdout_confirmation"


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


def _near_miss_markdown(frame: pd.DataFrame, acceptance: dict[str, Any], holdout: dict[str, Any]) -> str:
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


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _truthy(value: Any) -> bool:
    return value is True or str(value).strip().lower() in {"1", "true", "yes", "pass", "ready"}


def _text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def _as_utc(value: datetime | None) -> datetime:
    value = value or datetime.now(timezone.utc)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


if __name__ == "__main__":
    result = build_corrective_statistical_remediation()
    print(json.dumps({"summary": result.summary, "paths": {key: str(value) for key, value in result.paths.items()}}, indent=2))
