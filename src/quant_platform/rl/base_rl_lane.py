from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd

from quant_platform.active_pipeline import CommandResult, ROOT, build_trade_dataset, paper_candidate_shortlist_rows
from quant_platform.pair_market_utils import pair_markets_from_pair
from quant_platform.execution import (
    DydxNetworkConfig,
    effective_dydx_account_state_snapshot,
    dydx_execution_compatibility_snapshot,
    injective_execution_compatibility_snapshot,
)
from quant_platform.rl.rl_backtest import run_rl_research
from quant_platform.rl.rl_learning_agent import run_rl_learning_cycle


BASE_RL_STATUS_COLUMNS = [
    "status",
    "blocker",
    "paper_authorized",
    "journal_monitoring_authorized",
    "journal_learning_ready",
    "strategy_acceptance_ready",
    "paper_execution_ready",
    "model_gate_accepted",
    "global_model_gate_accepted",
    "route_model_gate_accepted",
    "route_model_gate_blocker",
    "shortlist_pairs",
    "pairs_with_rl_support",
    "pair_support_coverage",
    "execution_truth_mode",
    "injective_checked",
    "injective_blocker",
    "injective_mirrorable_pairs",
    "injective_spot_first_pairs",
    "injective_spot_first_research_rejected",
    "injective_pair_support_coverage",
    "compatibility_audit_pairs",
    "compatibility_audit_execution_compatible_pairs",
    "compatibility_audit_blocked_pairs",
    "next_action",
    "evidence_path",
    "candidate_set_hash",
    "candidate_set_path",
    "generated_at",
]

BASE_RL_COMPARISON_COLUMNS = [
    "pair",
    "base_coverage",
    "augmented_coverage",
    "base_policy",
    "augmented_policy",
    "agreement",
]

BASE_RL_PROMOTION_COLUMNS = [
    "pair",
    "in_route",
    "route_status",
    "coverage_status",
    "base_handoff_ready",
    "paper_eligible",
    "reason",
]

BASE_RL_PAIR_BLOCKER_COLUMNS = [
    "pair",
    "in_route",
    "route_status",
    "dydx_recommended_strategy",
    "dydx_recommendation_source",
    "rl_policy_name",
    "coverage_status",
    "global_gates_ready",
    "pair_blockers",
    "paper_status",
    "paper_eligible",
    "reason",
    "next_action",
]

BLOCKER_DELTA_COLUMNS = ["metric", "previous_value", "current_value", "delta", "notes"]


def base_rl_paths(root: Path = ROOT) -> dict[str, Path]:
    reports = root / "reports" / "rl"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = reports / "base_rl_runs" / run_id
    return {
        "training_report": reports / "base_rl_training_report.csv",
        "evaluation_report": reports / "base_rl_evaluation_report.csv",
        "pair_coverage": reports / "base_rl_pair_coverage.csv",
        "policy_summary": reports / "base_rl_policy_summary.json",
        "blocked_actions": reports / "base_rl_blocked_actions.csv",
        "paper_handoff_status": reports / "base_rl_paper_handoff_status.csv",
        "feedback_summary": reports / "base_rl_feedback_summary.csv",
        "outcome_training_dataset": reports / "base_rl_outcome_training_dataset.csv",
        "route_candidate_snapshot": reports / "base_rl_route_candidates.csv",
        "run_manifest": reports / "base_rl_run_manifest.json",
        "run_manifest_snapshot": run_dir / "manifest.json",
        "candidate_set_snapshot": run_dir / "route_candidates.csv",
        "run_evaluation_snapshot": run_dir / "pair_coverage.csv",
        "comparison_report": reports / "base_vs_augmented_pair_comparison.csv",
        "promotion_readiness_report": reports / "base_rl_promotion_readiness.csv",
        "promotion_decision_report": reports / "base_rl_promotion_decision.csv",
        "blocker_delta_report": reports / "base_rl_blocker_delta_report.csv",
        "pair_blocker_report": reports / "base_rl_pair_blocker_report.csv",
        "augmented_policy": reports / "augmented_rl_policy.json",
        "augmented_evidence": reports / "augmented_rl_evidence.json",
    }


def run_base_rl(root: Path = ROOT, pair_id: str = "") -> CommandResult:
    dataset_path = root / "data" / "ml" / "trade_training_dataset.csv"
    if not dataset_path.exists():
        build_trade_dataset(root=root)
    research = run_rl_research(root=root, pair_id=pair_id)
    previous_manifest = _load_manifest(base_rl_paths(root)["run_manifest"])
    frozen_candidates = _resolve_frozen_candidates(root=root, pair_id=pair_id)
    learning = run_rl_learning_cycle(root=root, pair_id=pair_id, policy_candidates=12)

    reports = base_rl_paths(root)
    training = _read_csv(research.paths.get("training_report", Path()))
    evaluation = _read_csv(research.paths.get("evaluation_report", Path()))
    blocked = _read_csv(research.paths.get("blocked_actions", Path()))
    learning_summary = _read_csv(learning.paths.get("summary", Path()))

    coverage = _base_rl_pair_coverage(root, learning_summary, frozen_candidates)
    candidate_hash = _candidate_set_hash(frozen_candidates)
    handoff = base_rl_paper_handoff_report(root=root)

    policy_payload = _load_json(learning.paths.get("best_policy", Path()))
    manifest = _build_base_rl_manifest(root=root, pair_id=pair_id, candidate_set=frozen_candidates, manifest_path=reports["run_manifest"], candidate_hash=candidate_hash, status=handoff.iloc[0].get("status", "research_only") if not handoff.empty else "research_only")

    _write_json(reports["run_manifest"], manifest)
    _write_json(reports["run_manifest_snapshot"], manifest)
    frozen_candidates.to_csv(reports["route_candidate_snapshot"], index=False)
    frozen_candidates.to_csv(reports["candidate_set_snapshot"], index=False)
    _write_paper_readiness_files(root=root, coverage=coverage, pair_id=pair_id)
    _build_blocker_delta_report(root=root, current_manifest=manifest, previous_manifest=previous_manifest)

    reports["training_report"].parent.mkdir(parents=True, exist_ok=True)
    training.to_csv(reports["training_report"], index=False)
    evaluation.to_csv(reports["evaluation_report"], index=False)
    coverage.to_csv(reports["pair_coverage"], index=False)
    blocked.to_csv(reports["blocked_actions"], index=False)
    reports["policy_summary"].write_text(json.dumps(policy_payload, indent=2, sort_keys=True), encoding="utf-8")

    # Refresh current-state aliases after the authoritative handoff artifact exists.
    from quant_platform.rl.brain_contract import build_phase1_readiness_surfaces

    build_phase1_readiness_surfaces(root)

    summary = {
        "status": str(handoff.get("status", pd.Series(["research_only"])).iloc[0]) if not handoff.empty else "research_only",
        "pair_rows": int(len(coverage)),
        "paper_authorized": bool(handoff.get("paper_authorized", pd.Series([False])).astype(bool).iloc[0]) if not handoff.empty else False,
        "candidate_rows": int(len(frozen_candidates)),
        "candidate_hash": candidate_hash,
    }
    return CommandResult(
        paths={
            **reports,
            "research_training_report": research.paths.get("training_report", Path()),
            "research_evaluation_report": research.paths.get("evaluation_report", Path()),
        },
        summary=summary,
    )


def evaluate_base_rl(root: Path = ROOT) -> CommandResult:
    reports = base_rl_paths(root)
    manifest = _load_manifest(reports["run_manifest"])
    candidates = _candidate_set_from_manifest(root, manifest)
    learning_summary = _read_csv(root / "reports" / "rl" / "rl_learning_cycle_summary.csv")
    coverage = _base_rl_pair_coverage(root, learning_summary, candidates)
    handoff = base_rl_paper_handoff_report(root=root)
    comparison, promotion = _build_comparison_and_promotion(root=root, coverage=coverage, handoff=handoff)
    blocker_report = _build_pair_blocker_report(root=root, coverage=coverage, handoff=handoff)
    _write_csv(comparison, reports["comparison_report"])
    _write_csv(promotion, reports["promotion_readiness_report"])
    _write_csv(blocker_report, reports["pair_blocker_report"])

    coverage.to_csv(reports["pair_coverage"], index=False)
    _write_promotion_decision_report(root=root, coverage=coverage, handoff=handoff)
    return CommandResult(
        paths={
            "pair_coverage": reports["pair_coverage"],
            "paper_handoff_status": reports["paper_handoff_status"],
            "comparison_report": reports["comparison_report"],
            "promotion_readiness_report": reports["promotion_readiness_report"],
            "promotion_decision_report": reports["promotion_decision_report"],
            "pair_blocker_report": reports["pair_blocker_report"],
        },
        summary={"pair_rows": int(len(coverage)), "paper_authorized": bool(handoff.get("paper_authorized", pd.Series([False])).astype(bool).iloc[0]) if not handoff.empty else False},
    )


def base_rl_paper_handoff_report(root: Path = ROOT) -> pd.DataFrame:
    reports = base_rl_paths(root)
    reports["paper_handoff_status"].parent.mkdir(parents=True, exist_ok=True)
    readiness = _read_csv(root / "reports" / "priority_readiness.csv")
    paper_preflight = _read_csv(root / "reports" / "paper_execution_preflight.csv")
    model = _read_csv(root / "reports" / "ml" / "model_gated_acceptance.csv")

    manifest = _load_manifest(reports["run_manifest"])
    frozen_candidates = _candidate_set_from_manifest(root, manifest)
    learning_summary = _read_csv(root / "reports" / "rl" / "rl_learning_cycle_summary.csv")
    coverage = _base_rl_pair_coverage(root, learning_summary, frozen_candidates)
    route_pairs = sorted({str(value) for value in coverage.get("pair", pd.Series(dtype=object)).astype(str).tolist() if str(value).strip()}) if not coverage.empty else []
    route_markets = _route_markets_from_candidates(coverage if not coverage.empty else frozen_candidates)
    compatibility = dydx_execution_compatibility_snapshot(route_markets, root=root)
    injective = injective_execution_compatibility_snapshot(route_pairs, root=root)
    injective_spot_first = _read_csv(root / "reports" / "active" / "injective_spot_first_candidate_shortlist.csv")
    account_state = effective_dydx_account_state_snapshot(DydxNetworkConfig.paper_testnet_from_env(), root=root)

    strategy_ready = _gate_ready(readiness, "strategy_acceptance")
    failed_preflight_blocker = _first_failed_preflight_blocker(paper_preflight)
    paper_ready = _gate_ready(readiness, "paper_execution_gate") and bool(
        not paper_preflight.empty and paper_preflight.get("ready", pd.Series(dtype=bool)).fillna(False).astype(bool).all()
    )
    global_model_ready = bool(not model.empty and model.get("accepted", pd.Series([False])).astype(bool).iloc[0])
    route_model_ready, route_model_blocker = _route_model_gate_status(root=root, candidate_set=frozen_candidates)
    model_ready = bool(global_model_ready or route_model_ready)

    shortlist_pairs = int(len(coverage))
    compatibility_audit = _route_candidate_compatibility_audit(root=root) if shortlist_pairs == 0 else pd.DataFrame()
    compatibility_audit_pairs = int(len(compatibility_audit))
    compatibility_audit_execution_compatible_pairs = int(
        compatibility_audit.get("execution_compatible", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()
    ) if not compatibility_audit.empty else 0
    compatibility_audit_blocked_pairs = max(0, compatibility_audit_pairs - compatibility_audit_execution_compatible_pairs)
    empty_route_execution_blocker = ""
    if shortlist_pairs == 0 and compatibility_audit_pairs > 0 and compatibility_audit_execution_compatible_pairs == 0:
        empty_route_execution_blocker = "route_market_unconfirmed_on_exchange"

    if coverage.empty:
        pairs_with_support = 0
    else:
        rl_supported = coverage.get("rl_supported", pd.Series(dtype=bool)).fillna(False).astype(bool)
        execution_compatible = coverage.get("execution_compatible", pd.Series(dtype=bool)).fillna(False).astype(bool)
        pairs_with_support = int((rl_supported & execution_compatible).sum())
    pair_support_coverage = float(pairs_with_support / shortlist_pairs) if shortlist_pairs else 0.0

    compatibility_ready = bool(compatibility.get("checked", False)) and not compatibility.get("blocker")
    account_state_ready = bool(account_state.get("checked", False)) and not account_state.get("blocker")
    injective_checked = bool(injective.get("checked", False))
    injective_mirrorable_pairs = int(len(injective.get("mirrorable_pairs", [])))
    injective_spot_pairs = int(len(injective.get("spot_pairs", [])))
    injective_spot_first_pairs = int(len(injective_spot_first))
    injective_spot_first_research_rejected = int(
        injective_spot_first.get("injective_lane_status", pd.Series(dtype=object))
        .astype(str)
        .eq("injective_supported_but_research_rejected")
        .sum()
    ) if not injective_spot_first.empty else 0
    injective_blocker = str(injective.get("blocker", ""))
    if not injective_blocker and injective_spot_first_pairs > 0 and injective_mirrorable_pairs == 0:
        injective_blocker = "injective_supported_pairs_need_research_repair"
    elif injective_blocker == "injective_no_spot_pairs" and injective_spot_first_pairs > 0:
        injective_blocker = "injective_supported_pairs_need_research_repair"
    injective_pair_support_coverage = float(injective_mirrorable_pairs / shortlist_pairs) if shortlist_pairs else 0.0
    paper_authorized = bool(
        strategy_ready
        and paper_ready
        and compatibility_ready
        and account_state_ready
        and model_ready
        and shortlist_pairs > 0
        and pairs_with_support == shortlist_pairs
    )
    journal_monitoring_authorized = bool(strategy_ready and paper_ready)
    journal_learning_ready = bool(journal_monitoring_authorized and account_state_ready)
    execution_truth_mode = "paper_only"
    if empty_route_execution_blocker:
        execution_truth_mode = "journal_plus_injective_spot_research" if injective_spot_first_pairs > 0 else "journal_only_execution_blocked"
    elif compatibility_ready and account_state_ready:
        execution_truth_mode = "dydx_confirmed"
    elif injective_spot_pairs > 0:
        execution_truth_mode = "paper_plus_injective_spot"
    blocker = ""
    if account_state.get("blocker"):
        blocker = str(account_state.get("blocker", "account_state_blocked"))
    elif not strategy_ready:
        blocker = "strategy_acceptance_not_ready"
    elif not paper_ready:
        blocker = failed_preflight_blocker or "paper_execution_not_ready"
    elif not model_ready:
        blocker = route_model_blocker or "model_gate_not_accepted"
    elif shortlist_pairs == 0:
        blocker = empty_route_execution_blocker or route_model_blocker or "route_shortlist_empty"
    elif pairs_with_support < shortlist_pairs:
        blocker = "pair_specific_rl_support_missing"
    elif compatibility.get("blocker"):
        blocker = str(compatibility.get("blocker", "execution_compatibility_blocked"))
    status = "paper_authorized" if paper_authorized else "research_only"
    if paper_authorized:
        next_action = (
            "paper handoff is authorized and Injective spot routing is available for supported pairs"
            if execution_truth_mode == "paper_plus_injective_spot"
            else "paper handoff is authorized for supported shortlist pairs"
        )
    elif blocker == "strategy_acceptance_not_ready":
        next_action = "improve strategy acceptance before base RL paper handoff"
    elif blocker == "paper_execution_not_ready":
        next_action = "restore paper execution readiness before base RL paper handoff"
    elif blocker == "route_market_compatibility_missing":
        next_action = "collect execution compatibility evidence for every route market before paper handoff"
    elif blocker == "route_market_unconfirmed_on_exchange":
        next_action = (
            "route paper only through markets that confirm on exchange; keep journal/live monitoring active while repairing execution compatibility"
            if injective_spot_pairs > 0 or injective_spot_first_pairs > 0
            else "route paper only through markets that confirm on exchange and re-run handoff"
        )
    elif blocker == "orphan_leg_open":
        next_action = "flatten lingering account positions before any new paper handoff"
    elif blocker == "account_state_unverified":
        next_action = "restore live account-state visibility before trusting paper handoff"
    elif blocker in {
        "model_gate_not_accepted",
        "route_model_gate_not_accepted",
        "route_model_support_missing",
        "route_shortlist_empty",
    }:
        next_action = "improve model gate acceptance before base RL paper handoff"
    elif blocker == "pair_specific_rl_support_missing":
        next_action = "close pair-specific RL support gaps before base RL paper handoff"
    else:
        next_action = "improve blockers before base RL paper handoff"

    frame = pd.DataFrame(
        [
            {
                "status": status,
                "blocker": blocker,
                "paper_authorized": paper_authorized,
                "journal_monitoring_authorized": journal_monitoring_authorized,
                "journal_learning_ready": journal_learning_ready,
                "strategy_acceptance_ready": strategy_ready,
                "paper_execution_ready": paper_ready,
                "model_gate_accepted": model_ready,
                "global_model_gate_accepted": global_model_ready,
                "route_model_gate_accepted": route_model_ready,
                "route_model_gate_blocker": route_model_blocker,
                "shortlist_pairs": shortlist_pairs,
        "pairs_with_rl_support": pairs_with_support,
        "pair_support_coverage": pair_support_coverage,
                "execution_truth_mode": execution_truth_mode,
                "injective_checked": injective_checked,
                "injective_blocker": injective_blocker,
                "injective_mirrorable_pairs": injective_mirrorable_pairs,
                "injective_spot_first_pairs": injective_spot_first_pairs,
                "injective_spot_first_research_rejected": injective_spot_first_research_rejected,
                "injective_pair_support_coverage": injective_pair_support_coverage,
                "compatibility_audit_pairs": compatibility_audit_pairs,
                "compatibility_audit_execution_compatible_pairs": compatibility_audit_execution_compatible_pairs,
                "compatibility_audit_blocked_pairs": compatibility_audit_blocked_pairs,
                "next_action": next_action,
                "candidate_set_path": str(reports["route_candidate_snapshot"]),
                "candidate_set_hash": _candidate_set_hash(frozen_candidates),
                "evidence_path": ";".join(
                    [
                        str(root / "reports" / "priority_readiness.csv"),
                        str(root / "reports" / "paper_execution_preflight.csv"),
                        str(root / "reports" / "active" / "dydx_execution_market_compatibility.csv"),
                        str(root / "reports" / "active" / "injective_execution_market_compatibility.csv"),
                        str(root / "reports" / "active" / "injective_mirror_candidate_queue.csv"),
                        str(root / "reports" / "active" / "injective_spot_first_candidate_shortlist.csv"),
                        str(root / "reports" / "ml" / "model_gated_acceptance.csv"),
                        str(reports["pair_coverage"]),
                    ]
                ),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
        ],
        columns=BASE_RL_STATUS_COLUMNS,
    )
    frame.to_csv(reports["paper_handoff_status"], index=False)
    from quant_platform.active_pipeline import current_state
    from quant_platform.rl.brain_contract import build_phase1_readiness_surfaces

    build_phase1_readiness_surfaces(root)
    current_state(root)
    return frame


def _route_markets_from_candidates(frame: pd.DataFrame) -> list[str]:
    if frame.empty or "pair" not in frame.columns:
        return []
    markets: list[str] = []
    for pair in frame.get("pair", pd.Series(dtype=object)).astype(str).tolist():
        markets.extend(_pair_to_markets(pair))
    return sorted({market for market in markets if market})


def _load_dydx_execution_compatibility_map(root: Path) -> dict[str, bool]:
    table = _read_csv(root / "reports" / "active" / "dydx_execution_market_compatibility.csv")
    compatibility_map: dict[str, bool] = {}
    if table.empty or "market" not in table.columns:
        return compatibility_map
    working = table.copy()
    working["market"] = working["market"].astype(str).str.upper().str.strip()
    for _, row in working.iterrows():
        market = str(row.get("market", "")).upper().strip()
        if not market:
            continue
        compatibility_map[market] = _coerce_bool(row.get("compatible_for_paper_submit", False))
    return compatibility_map


def _pair_execution_compatibility_context(pair: str, compatibility_map: dict[str, bool]) -> dict[str, object]:
    markets = _pair_to_markets(pair)
    confirmed: list[str] = []
    missing: list[str] = []
    unsupported: list[str] = []
    for market in markets:
        status = compatibility_map.get(market)
        if status is None:
            missing.append(market)
        elif bool(status):
            confirmed.append(market)
        else:
            unsupported.append(market)

    if not markets:
        blocker = "pair_markets_unresolved"
        compatible = False
    elif not compatibility_map:
        blocker = "execution_compatibility_table_missing"
        compatible = False
    elif missing:
        blocker = "route_market_compatibility_missing"
        compatible = False
    elif unsupported:
        blocker = "route_market_unconfirmed_on_exchange"
        compatible = False
    else:
        blocker = ""
        compatible = True

    return {
        "execution_markets": ";".join(markets),
        "confirmed_execution_markets": ";".join(confirmed),
        "missing_execution_markets": ";".join(unsupported + missing),
        "execution_compatible": compatible,
        "execution_blocker": blocker,
    }


def _pair_to_markets(pair: str) -> list[str]:
    return pair_markets_from_pair(pair)


def _route_model_gate_status(root: Path, candidate_set: pd.DataFrame) -> tuple[bool, str]:
    if candidate_set.empty or "pair" not in candidate_set.columns:
        return False, "route_shortlist_empty"
    pair_support = _read_csv(root / "reports" / "ml" / "model_gate_pair_support_report.csv")
    if pair_support.empty or "pair" not in pair_support.columns:
        return False, "route_model_support_missing"

    route_pairs = {
        _normalize_pair_text_for_rl(pair)
        for pair in candidate_set.get("pair", pd.Series(dtype=object)).astype(str).tolist()
        if _normalize_pair_text_for_rl(pair)
    }

    if not route_pairs:
        return False, "route_model_support_missing"

    support_pairs: set[str] = set()
    strong_pairs: set[str] = set()
    for _, row in pair_support.iterrows():
        raw_pair = str(row.get("pair", ""))
        normalized_pair = _normalize_pair_text_for_rl(raw_pair)
        if not normalized_pair:
            continue
        support_pairs.add(normalized_pair)
        status = str(row.get("support_status", "")).strip()
        if status == "strong_model_support":
            strong_pairs.add(normalized_pair)

    missing = [pair for pair in sorted(route_pairs) if pair not in support_pairs]
    if missing:
        return False, "route_model_support_missing"

    if len(strong_pairs) < len(route_pairs):
        return False, "route_model_gate_not_accepted"
    return True, ""


def _route_candidate_compatibility_audit(root: Path = ROOT) -> pd.DataFrame:
    try:
        audit = paper_candidate_shortlist_rows(root=root, max_pairs=10, require_execution_compatible=False)
    except Exception:
        return pd.DataFrame()
    if audit.empty or "pair" not in audit.columns:
        return pd.DataFrame()
    return audit.copy()


def run_augmented_rl(root: Path = ROOT, pair_id: str = "") -> CommandResult:
    # Augmented lane uses base route and base learning artifacts as input.
    run_base_rl(root=root, pair_id=pair_id)
    reports = base_rl_paths(root)
    research_signals = _read_csv(root / "data" / "processed" / "research_knowledge" / "research_strategy_hints.csv")
    risk_priors = _read_csv(root / "data" / "processed" / "research_knowledge" / "research_risk_priors.csv")
    features = _read_csv(root / "data" / "processed" / "research_knowledge" / "research_features.csv")
    research_frames = [frame for frame in (research_signals, risk_priors, features) if not frame.empty]
    research_rows = pd.concat(research_frames, ignore_index=True) if research_frames else pd.DataFrame()
    source_counts = (
        research_rows.get("source_type", pd.Series(dtype=str)).astype(str).replace("", "unknown").value_counts().to_dict()
        if not research_rows.empty
        else {}
    )
    research_source_types = ";".join(sorted(str(value) for value in source_counts))
    udemy_research_rows = int(source_counts.get("udemy", 0))

    summary = _read_csv(root / "reports" / "rl" / "base_rl_pair_coverage.csv")
    base_policy = str(summary.get("rl_policy_name", pd.Series([""])).iloc[0]) if not summary.empty else ""

    augmented = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S"),
        "pair_id": pair_id,
        "base_policy": base_policy,
        "research_signals": int(len(research_signals)),
        "risk_priors": int(len(risk_priors)),
        "features": int(len(features)),
        "research_source_types": research_source_types,
        "udemy_research_rows": udemy_research_rows,
        "educational_evidence_authority": "research_only_no_trade_authority",
        "status": "comparison_only",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    pair_signals = _load_augmented_pair_signals(summary, research_signals, risk_priors, features)
    comparison_rows: list[dict[str, object]] = []
    for _, row in summary.iterrows():
        pair = str(row.get("pair", ""))
        base_supported = bool(row.get("rl_supported", False))
        augmented_supported = bool(base_supported and pair_signals.get(pair, {}).get("allow", True))
        augmented_policy = _augmented_policy_name(base_policy, pair_signals.get(pair, {}))
        comparison_rows.append(
            {
                "pair": pair,
                "base_coverage": base_supported,
                "augmented_coverage": augmented_supported,
                "base_policy": base_policy,
                "augmented_policy": augmented_policy,
                "agreement": base_supported == augmented_supported,
            }
        )

    _write_json(reports["augmented_policy"], augmented)
    _write_json(
        reports["augmented_evidence"],
        {
            "pair_id": pair_id,
            "strategy_hints": int(len(research_signals)),
            "risk_priors": int(len(risk_priors)),
            "features": int(len(features)),
            "source_type_counts": source_counts,
            "udemy_research_rows": udemy_research_rows,
            "educational_evidence_authority": "research_only_no_trade_authority",
            "pairs_considered": int(len(summary)),
            "pairs_disagreed": int(
                (pd.DataFrame(comparison_rows)["agreement"].astype(bool) == False).sum()
                if comparison_rows
                else 0
            ),
            "source": "research_knowledge",
        },
    )
    _write_csv(pd.DataFrame(comparison_rows, columns=BASE_RL_COMPARISON_COLUMNS), reports["comparison_report"])
    _write_csv(pd.DataFrame([augmented]), root / "reports" / "rl" / "base_rl_promotion_readiness.csv")
    return CommandResult(
        paths={
            "augmented_policy": reports["augmented_policy"],
            "augmented_evidence": reports["augmented_evidence"],
            "comparison_report": reports["comparison_report"],
            "run_manifest": reports["run_manifest"],
        },
        summary={
            "status": "comparison_ready",
            "rows": int(len(summary)),
            "research_signal_rows": int(len(research_signals) + len(risk_priors) + len(features)),
            "research_source_types": research_source_types,
            "udemy_research_rows": udemy_research_rows,
        },
    )


def refresh_base_rl_feedback(root: Path = ROOT) -> CommandResult:
    reports = base_rl_paths(root)
    outcome = _build_outcome_training_dataset(root)
    feedback = _feedback_summary(outcome)
    reports["feedback_summary"].parent.mkdir(parents=True, exist_ok=True)
    outcome.to_csv(reports["outcome_training_dataset"], index=False)
    feedback.to_csv(reports["feedback_summary"], index=False)
    return CommandResult(paths={"feedback_summary": reports["feedback_summary"], "outcome_training_dataset": reports["outcome_training_dataset"]}, summary={"rows": int(len(outcome)), "verified_outcomes": int(feedback.get("verified_outcomes", pd.Series([0])).iloc[0]) if not feedback.empty else 0})


def _resolve_frozen_candidates(root: Path, pair_id: str = "") -> pd.DataFrame:
    candidates = paper_candidate_shortlist_rows(root=root, require_execution_compatible=True)
    if "execution_compatible" in candidates.columns:
        candidates = candidates[candidates["execution_compatible"].map(_coerce_bool)].copy()
    if pair_id:
        candidates = candidates[candidates.get("pair", pd.Series(dtype=str)).astype(str).str.contains(pair_id.replace("/", "-"), case=False, regex=False)].copy()
    if candidates.empty:
        return candidates.reset_index(drop=True)
    if "route_context" not in candidates.columns:
        candidates["route_context"] = "paper_route_default"
        candidates["route_blocker"] = candidates.get("best_execution_venue", pd.Series(dtype=str)).astype(str).map(lambda value: "" if str(value).lower() == "dydx" else "non_default_route")
    return candidates.reset_index(drop=True)


def _load_augmented_pair_signals(
    base_coverage: pd.DataFrame,
    strategy_hints: pd.DataFrame,
    risk_priors: pd.DataFrame,
    features: pd.DataFrame,
) -> dict[str, dict[str, object]]:
    pair_signals: dict[str, dict[str, object]] = {}
    for _, row in base_coverage.iterrows():
        pair = str(row.get("pair", ""))
        pair_signals[pair] = {"allow": True, "disables": []}
        if pair and _research_rows_apply_to_pair(strategy_hints, pair):
            pair_signals[pair]["allow"] = False
            pair_signals[pair]["disables"].append("strategy_hint_disallowed")
        if pair and _research_rows_apply_to_pair(risk_priors, pair):
            prior = _first_matching_research_row(risk_priors, pair)
            if prior is not None and _risk_prior_disallows(prior):
                pair_signals[pair]["allow"] = False
                pair_signals[pair]["disables"].append("risk_prior_disallowed")
        if pair and _research_rows_apply_to_pair(features, pair):
            pair_signals[pair]["allow"] = True
    return pair_signals


def _research_rows_apply_to_pair(rows: pd.DataFrame, pair: str) -> bool:
    if rows.empty:
        return False
    pair_key = _normalize_pair_text_for_rl(pair)
    for _, row in rows.iterrows():
        raw_filter = row.get("pair_filter", "")
        candidates = [part.strip() for part in str(raw_filter).split(";") if part.strip()]
        if not candidates:
            continue
        if any(_normalize_pair_text_for_rl(value) == pair_key for value in candidates):
            return True
    return False


def _first_matching_research_row(rows: pd.DataFrame, pair: str) -> pd.Series | None:
    pair_key = _normalize_pair_text_for_rl(pair)
    for _, row in rows.iterrows():
        raw_filter = row.get("pair_filter", "")
        candidates = [part.strip() for part in str(raw_filter).split(";") if part.strip()]
        if not candidates:
            continue
        if any(_normalize_pair_text_for_rl(value) == pair_key for value in candidates):
            return row
    return None


def _risk_prior_disallows(row: pd.Series) -> bool:
    rule = str(row.get("risk_rule", "")).lower()
    if any(flag in rule for flag in ("avoid", "avoid_pair", "exclude", "exclude_pair", "ban", "forbid", "not_trade")):
        return True
    confidence = pd.to_numeric(row.get("confidence", 0.0), errors="coerce")
    return float(confidence if pd.notna(confidence) else 0.0) >= 0.6


def _augmented_policy_name(base_policy: str, signal: dict[str, object]) -> str:
    policy = base_policy or "base_policy_unknown"
    if not signal:
        return policy
    if signal.get("disables"):
        return f"{policy}_augmented_blocked_by_research"
    if signal.get("allow") is False:
        return f"{policy}_augmented_research_block"
    return f"{policy}_augmented"


def _normalize_pair_text_for_rl(value: object) -> str:
    normalized = str(value or "").replace("_", "-").replace("/", "-").replace(" ", "")
    parts = [part.upper() for part in normalized.split("-") if part and part.lower() != "usd"]
    if not parts:
        return ""
    parts = sorted(set(parts))
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return parts[0]


def _candidate_set_from_manifest(root: Path, manifest: dict[str, object]) -> pd.DataFrame:
    snapshot = _read_csv(root / "reports" / "rl" / "base_rl_route_candidates.csv")
    if manifest and isinstance(manifest, dict):
        snapshot_path = manifest.get("route_candidate_snapshot")
        if snapshot_path:
            candidate_path = root / str(snapshot_path)
            if candidate_path.exists():
                snapshot = _read_csv(candidate_path)
    return snapshot


def _build_base_rl_manifest(
    root: Path,
    *,
    pair_id: str,
    candidate_set: pd.DataFrame,
    manifest_path: Path,
    candidate_hash: str,
    status: str,
) -> dict[str, object]:
    return {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pair_id": pair_id,
        "status": status,
        "route": "dydx_default",
        "candidate_rows": int(len(candidate_set)),
        "candidate_set_hash": candidate_hash,
        "route_candidate_snapshot": str(manifest_path).replace("base_rl_run_manifest.json", "base_rl_route_candidates.csv"),
        "manifest_artifact": str(manifest_path),
        "paper_handoff_path": str(root / "reports" / "rl" / "base_rl_paper_handoff_status.csv"),
        "coverage_path": str(root / "reports" / "rl" / "base_rl_pair_coverage.csv"),
    }


def _base_rl_pair_coverage(root: Path, learning_summary: pd.DataFrame, candidate_set: pd.DataFrame | None = None) -> pd.DataFrame:
    shortlist = candidate_set.copy() if candidate_set is not None else _read_csv(root / "reports" / "dashboard" / "paper_candidate_shortlist.csv")
    if candidate_set is None and shortlist.empty:
        shortlist = _read_csv(root / "data" / "processed" / "pair_universe.csv")
        if not shortlist.empty and "decision_bucket" in shortlist.columns:
            shortlist = shortlist[shortlist["decision_bucket"].astype(str) == "PROMOTE"].copy()
    compatibility_map = _load_dydx_execution_compatibility_map(root)
    model_support = _read_csv(root / "reports" / "ml" / "model_gate_pair_support_report.csv")
    model_support_by_pair = {
        _normalize_pair_text_for_rl(row.get("pair", "")): str(
            row.get("support_status", row.get("pair_model_support_status", "")) or ""
        ).strip()
        for _, row in model_support.iterrows()
        if str(row.get("pair", "") or "").strip()
    }
    if learning_summary.empty:
        winner_rows = pd.DataFrame()
    else:
        winner_series = learning_summary.get("winner", pd.Series(dtype=int, index=learning_summary.index))
        winner_mask = winner_series.fillna(0).astype(int) == 1
        winner_rows = learning_summary.loc[winner_mask]
    winner = winner_rows.iloc[0] if not winner_rows.empty else (learning_summary.sort_values("profit_factor", ascending=False).iloc[0] if not learning_summary.empty else pd.Series(dtype=object))
    pair_focus = _parse_top_pairs_entered(str(winner.get("top_pairs_entered", ""))) if not winner.empty else {}
    rows = []
    for idx, row in shortlist.iterrows():
        pair = str(row.get("pair", ""))
        frozen_focus_count = _safe_int(row.get("rl_pair_focus_count", 0))
        focus_count = max(int(pair_focus.get(pair, 0)), frozen_focus_count)
        pair_model_support_status = str(row.get("pair_model_support_status", "")).strip()
        if not pair_model_support_status:
            pair_model_support_status = model_support_by_pair.get(_normalize_pair_text_for_rl(pair), "")
        pair_model_taken_trades = _safe_int(row.get("pair_model_taken_trades", 0))
        execution_context = _pair_execution_compatibility_context(pair, compatibility_map)
        winner_flag = _coerce_bool(row.get("rl_policy_winner", False))
        candidate_support = bool(
            focus_count > 0
            or pair_model_support_status == "strong_model_support"
            or (
                winner_flag
                and pair_model_support_status == "strong_model_support"
                and pair_model_taken_trades > 0
            )
        )
        rows.append(
            {
                "shortlist_rank": int(idx + 1),
                "pair": pair,
                "best_execution_venue": str(row.get("best_execution_venue", "dydx")),
                "rl_policy_name": str(winner.get("policy_name", "")) if not winner.empty else "",
                "rl_focus_count": focus_count,
                "rl_supported": candidate_support,
                **execution_context,
                "evidence_path": str(row.get("evidence_path", "")),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "shortlist_rank",
            "pair",
            "best_execution_venue",
            "rl_policy_name",
            "rl_focus_count",
            "rl_supported",
            "execution_markets",
            "confirmed_execution_markets",
            "missing_execution_markets",
            "execution_compatible",
            "execution_blocker",
            "evidence_path",
        ],
    )


def _build_comparison_and_promotion(root: Path, coverage: pd.DataFrame, handoff: pd.DataFrame):
    comparison = pd.DataFrame(columns=BASE_RL_COMPARISON_COLUMNS)
    promotion = pd.DataFrame(columns=BASE_RL_PROMOTION_COLUMNS)
    if coverage.empty:
        return comparison, promotion

    base_policy = str(coverage.get("rl_policy_name", pd.Series([""])).iloc[0]) if not coverage.empty else ""
    handoff_ready = bool(handoff.get("status", pd.Series(["research_only"])).astype(str).iloc[0] == "paper_authorized")
    base_rows = []
    promo_rows = []
    for _, row in coverage.iterrows():
        pair = str(row.get("pair", ""))
        execution_compatible = bool(row.get("execution_compatible", False))
        in_route = "dydx" in str(row.get("best_execution_venue", "")).lower()
        supported = bool(row.get("rl_supported", False))
        coverage_supported = bool(supported and execution_compatible)
        base_rows.append(
            {
                "pair": pair,
                "base_coverage": bool(supported),
                "augmented_coverage": bool(coverage_supported),
                "base_policy": base_policy,
                "augmented_policy": base_policy,
                "agreement": True,
            }
        )
        reason = "route and RL support alignment"
        if not execution_compatible:
            reason = str(row.get("execution_blocker", "route_market_unconfirmed_on_exchange")) or "route_market_unconfirmed_on_exchange"
        elif not in_route:
            reason = "out_of_route"
        elif not supported:
            reason = "pair_specific_rl_support_missing"
        elif not bool(handoff.get("paper_authorized", pd.Series([False])).astype(bool).iloc[0]):
            reason = str(handoff.get("blocker", pd.Series(["base_gate_not_ready"])) .astype(str).iloc[0])
        promo_rows.append(
            {
                "pair": pair,
                "in_route": in_route,
                "route_status": "accepted" if "dydx" in str(row.get("best_execution_venue", "")).lower() else "out_of_route",
                "coverage_status": "supported" if coverage_supported else "unsupported",
                "base_handoff_ready": handoff_ready,
                "paper_eligible": bool(supported and in_route and coverage_supported and handoff_ready),
                "reason": reason,
            }
        )

    return pd.DataFrame(base_rows, columns=BASE_RL_COMPARISON_COLUMNS), pd.DataFrame(promo_rows, columns=BASE_RL_PROMOTION_COLUMNS)


def _write_paper_readiness_files(root: Path, coverage: pd.DataFrame, pair_id: str) -> None:
    handoff = base_rl_paper_handoff_report(root=root)
    comparison, promotion = _build_comparison_and_promotion(root=root, coverage=coverage, handoff=handoff)
    paths = base_rl_paths(root)
    _write_csv(comparison, paths["comparison_report"])
    _write_csv(promotion, paths["promotion_readiness_report"])
    _write_promotion_decision_report(root=root, coverage=coverage, handoff=handoff)
    blocker_report = _build_pair_blocker_report(root=root, coverage=coverage, handoff=handoff)
    _write_csv(blocker_report, paths["pair_blocker_report"])


def _build_pair_blocker_report(root: Path, coverage: pd.DataFrame, handoff: pd.DataFrame) -> pd.DataFrame:
    if coverage.empty:
        return pd.DataFrame(columns=BASE_RL_PAIR_BLOCKER_COLUMNS)

    strategy_ready = bool(handoff.get("strategy_acceptance_ready", pd.Series([False])).astype(bool).iloc[0]) if not handoff.empty else False
    paper_ready = bool(handoff.get("paper_execution_ready", pd.Series([False])).astype(bool).iloc[0]) if not handoff.empty else False
    model_ready = bool(handoff.get("model_gate_accepted", pd.Series([False])).astype(bool).iloc[0]) if not handoff.empty else False
    global_ready = bool(strategy_ready and paper_ready and model_ready)

    def _coerce_bool(value: object, default: bool = False) -> bool:
        if pd.isna(value):
            return default
        text = str(value).strip().lower()
        if text in {"nan", "none", "null", ""}:
            return default
        if text in {"true", "1", "yes", "y"}:
            return True
        if text in {"false", "0", "no", "n"}:
            return False
        try:
            return bool(int(float(text)))
        except Exception:
            if isinstance(value, bool):
                return value
        return bool(value) if not isinstance(value, float) else default

    rows: list[dict[str, object]] = []
    for _, row in coverage.iterrows():
        pair = str(row.get("pair", ""))
        venue = str(row.get("best_execution_venue", "")).strip().lower()
        in_route = venue == "dydx"
        execution_compatible = _coerce_bool(row.get("execution_compatible", False), default=False)
        rl_supported = _coerce_bool(row.get("rl_supported", False), default=False)
        coverage_ok = bool(rl_supported and execution_compatible)
        blockers: list[str] = []
        if not execution_compatible:
            execution_blocker = str(row.get("execution_blocker", "route_market_unconfirmed_on_exchange")).strip()
            if execution_blocker:
                blockers.append(execution_blocker)
        if not in_route:
            blockers.append("out_of_route")
        if not rl_supported:
            blockers.append("no_rl_support")
        if not global_ready:
            blockers.append("global_gate_block")

        route_status = "accepted" if in_route else "out_of_route"
        coverage_status = "supported" if coverage_ok else "unsupported"

        if not in_route:
            paper_status = "out_of_route"
        elif not execution_compatible:
            paper_status = "execution_incompatible"
        elif not rl_supported:
            paper_status = "no_rl_support"
        elif not global_ready:
            paper_status = "global_gate_blocked"
        else:
            paper_status = "paper_candidate"

        paper_eligible = in_route and coverage_ok and global_ready
        reason = ""
        if blockers:
            reason = ";".join(blockers)
        else:
            reason = "paired_and_supported"

        rows.append(
            {
                "pair": pair,
                "in_route": in_route,
                "route_status": route_status,
                "dydx_recommended_strategy": str(
                    row.get("dydx_recommended_strategy", row.get("rl_policy_name", row.get("rl_policy_winner", row.get("shortlist_reason", ""))))
                ),
                "dydx_recommendation_source": str(row.get("dydx_recommendation_source", "")),
                "rl_policy_name": str(row.get("rl_policy_name", "")),
                "coverage_status": coverage_status,
                "global_gates_ready": global_ready,
                "pair_blockers": reason,
                "paper_status": paper_status,
                "paper_eligible": paper_eligible,
                "reason": reason,
                "next_action": "hold_for_gate_unblock"
                if not global_ready
                else (
                    "wait_for_pair_execution_compatibility"
                    if not execution_compatible
                    else ("wait_for_pair_support" if not bool(row.get("rl_supported", False)) else ("skip_non_dydx_route" if not in_route else "eligible_for_paper"))
                ),
            }
        )
    return pd.DataFrame(rows, columns=BASE_RL_PAIR_BLOCKER_COLUMNS)


def _write_promotion_decision_report(root: Path, coverage: pd.DataFrame, handoff: pd.DataFrame) -> None:
    paths = base_rl_paths(root)
    if coverage.empty:
        _write_csv(pd.DataFrame(columns=BASE_RL_PROMOTION_COLUMNS), paths["promotion_decision_report"])
        return

    rows = []
    base_ready = bool(handoff.get("status", pd.Series(["research_only"])).astype(str).iloc[0] == "paper_authorized")
    handoff_blocker = str(handoff.get("blocker", pd.Series([""])).iloc[0]).strip() if not handoff.empty else ""
    for _, row in coverage.iterrows():
        in_route = "dydx" in str(row.get("best_execution_venue", "")).lower()
        supported = bool(row.get("rl_supported", False))
        execution_compatible = bool(row.get("execution_compatible", False))
        coverage_supported = bool(supported and execution_compatible)
        paper_eligible = bool(base_ready and coverage_supported and in_route)
        reason = ""
        if paper_eligible:
            reason = "eligible_for_paper"
        elif not base_ready:
            reason = handoff_blocker or "base_gate_not_ready"
        elif not execution_compatible:
            reason = str(row.get("execution_blocker", "route_market_unconfirmed_on_exchange"))
        elif not in_route:
            reason = "out_of_route"
        elif not supported:
            reason = "pair_specific_rl_support_missing"
        else:
            reason = handoff_blocker or "base_gate_not_ready"
        if paper_eligible:
            route_status = "accepted"
        else:
            route_status = "in_route" if in_route else "out_of_route"
        rows.append(
            {
                "pair": str(row.get("pair", "")),
                "in_route": in_route,
                "route_status": "accepted" if in_route else "out_of_route",
                "coverage_status": "supported" if coverage_supported else "unsupported",
                "base_handoff_ready": base_ready,
                "paper_eligible": paper_eligible,
                "reason": reason,
            }
        )
    _write_csv(pd.DataFrame(rows, columns=BASE_RL_PROMOTION_COLUMNS), paths["promotion_decision_report"])


def _build_blocker_delta_report(root: Path, current_manifest: dict[str, object], previous_manifest: dict[str, object]) -> None:
    reports = base_rl_paths(root)
    prev_status = str(previous_manifest.get("status", "")) if isinstance(previous_manifest, dict) else ""
    curr_status = str(current_manifest.get("status", "")) if isinstance(current_manifest, dict) else ""
    prev_candidates = str(previous_manifest.get("candidate_rows", "")) if isinstance(previous_manifest, dict) else ""
    curr_candidates = str(current_manifest.get("candidate_rows", "")) if isinstance(current_manifest, dict) else ""
    previous_handoff = _read_csv(root / "reports" / "rl" / "base_rl_paper_handoff_status.csv")
    current_handoff = _read_csv(root / "reports" / "rl" / "base_rl_paper_handoff_status.csv")
    prev_blocker = str(previous_handoff.get("blocker", pd.Series([""])).iloc[0]) if not previous_handoff.empty else ""
    curr_blocker = str(current_handoff.get("blocker", pd.Series([""])).iloc[0]) if not current_handoff.empty else ""

    rows = [
        {"metric": "status", "previous_value": prev_status, "current_value": curr_status, "delta": "changed" if prev_status != curr_status else "unchanged", "notes": "manifest_status"},
        {"metric": "candidate_rows", "previous_value": prev_candidates, "current_value": curr_candidates, "delta": "changed" if prev_candidates != curr_candidates else "unchanged", "notes": "frozen_candidate_snapshot"},
        {"metric": "paper_blocker", "previous_value": prev_blocker, "current_value": curr_blocker, "delta": "changed" if prev_blocker != curr_blocker else "unchanged", "notes": "paper handoff blocker"},
    ]
    _write_csv(pd.DataFrame(rows, columns=BLOCKER_DELTA_COLUMNS), reports["blocker_delta_report"])


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _parse_top_pairs_entered(value: str) -> dict[str, int]:
    mapping: dict[str, int] = {}
    text = str(value or "")
    for part in text.split(";"):
        token = part.strip()
        if not token or ":" not in token:
            continue
        pair, count = token.rsplit(":", 1)
        try:
            mapping[pair.strip()] = int(float(count))
        except (TypeError, ValueError):
            continue
    return mapping


def _candidate_set_hash(candidate_set: pd.DataFrame) -> str:
    if candidate_set.empty:
        return ""
    ordered = candidate_set.copy().reset_index(drop=True)
    hash_columns = [col for col in ["pair", "route_context", "route_blocker", "best_execution_venue", "decision_bucket", "decision_reason"] if col in ordered.columns]
    if "pair" in hash_columns:
        key_columns = hash_columns
    else:
        key_columns = list(ordered.columns)
    ordered = ordered.loc[:, key_columns].fillna("")
    if key_columns:
        if "pair" in ordered.columns:
            ordered["pair"] = ordered["pair"].astype(str).map(_normalize_pair_text_for_rl)
    sort_columns = ["pair"] if "pair" in ordered.columns else None
    if "route_context" in ordered.columns and "route_context" not in sort_columns:
        sort_columns.append("route_context")
    if "route_blocker" in ordered.columns and "route_blocker" not in sort_columns:
        sort_columns.append("route_blocker")
    if sort_columns and all(col in ordered.columns for col in sort_columns):
        ordered = ordered.sort_values(sort_columns)
    values = ["|".join(str(v) for v in row) for row in ordered.itertuples(index=False, name=None)]
    payload = "|".join([str(len(ordered)), ";".join(values), ";".join(key_columns)])
    digest = hashlib.sha256(payload.encode("utf-8"))
    return digest.hexdigest()


def _gate_ready(frame: pd.DataFrame, gate: str) -> bool:
    if frame.empty:
        return False
    rows = frame[frame.get("gate", pd.Series(dtype=str)).astype(str) == gate]
    return bool(not rows.empty and rows.get("ready", pd.Series([False])).fillna(False).astype(bool).any())


def _first_failed_preflight_blocker(frame: pd.DataFrame) -> str:
    """Prefer a concrete failed execution check over a generic gate label."""
    if frame.empty:
        return ""
    for _, row in frame.iterrows():
        ready = str(row.get("ready", "")).strip().lower() in {"true", "1", "yes"}
        blocker = str(row.get("blocker", "")).strip()
        if not ready and blocker and blocker.lower() not in {"nan", "none", "null"}:
            return blocker
    return ""


def _build_outcome_training_dataset(root: Path) -> pd.DataFrame:
    journal = _read_csv(root / "reports" / "paper_trading_journal.csv")
    shared = _read_csv(root / "reports" / "brain" / "shared_outcome_memory.csv")
    trade_store_path = root / "data" / "meta_learning" / "trades.jsonl"
    trade_rows = _read_jsonl(trade_store_path)
    records: list[dict[str, object]] = []
    for item in trade_rows:
        outcome = item.get("outcome", {}) if isinstance(item, dict) else {}
        execution = item.get("execution", {}) if isinstance(item, dict) else {}
        records.append(
            {
                "trade_id": str(item.get("trade_id", "")),
                "pair": str(item.get("pair", "")),
                "regime": str(item.get("regime", "unknown")),
                "strategy": str(item.get("strategy", "")),
                "action": str(item.get("signal", {}).get("action", "")) if isinstance(item.get("signal"), dict) else "",
                "risk_configuration": json.dumps(item.get("signal", {}), sort_keys=True) if isinstance(item.get("signal"), dict) else "{}",
                "submission_type": str(execution.get("submission_type", "outcome-verified")),
                "execution_quality_tier": str(execution.get("quality_tier", "outcome-verified")),
                "realized_return": _safe_float(outcome.get("realized_return", "")),
                "outcome_state": "outcome_verified",
                }
            )
    if not shared.empty:
        for _, row in shared.iterrows():
            if str(row.get("result_status", "")).strip().lower() == "broadcast_accepted_unconfirmed":
                continue
            records.append(
                {
                    "trade_id": str(row.get("candidate_id", "")) or str(row.get("submission_timestamp", "")),
                    "pair": str(row.get("pair", "")),
                    "regime": str(row.get("regime_bucket", "unknown")),
                    "strategy": str(row.get("strategy_family", "")),
                    "action": str(row.get("strategy_mode", "")),
                    "risk_configuration": json.dumps(
                        {
                            "source_lane": str(row.get("source_lane", "")),
                            "paper_venue": str(row.get("paper_venue", "")),
                            "promotion_authority": str(row.get("promotion_authority", "")),
                            "learning_weight": _safe_float(row.get("learning_weight", "")),
                        },
                        sort_keys=True,
                    ),
                    "submission_type": str(row.get("result_status", "")) or "paper_completed",
                    "execution_quality_tier": str(row.get("verification_status", "")) or "unverified",
                    "realized_return": _safe_float(row.get("realized_return", "")),
                    "outcome_state": str(row.get("outcome_state", "")) or "unknown",
                }
            )
    if not journal.empty:
        for _, row in journal.iterrows():
            plan_status = str(row.get("plan_status", ""))
            if plan_status == "broadcast_accepted_unconfirmed":
                continue
            verified_journal_outcome = _journal_row_has_verified_outcome(row)
            outcome_state = "audit_only"
            if plan_status in {"paper_submitted", "confirmed_on_exchange"}:
                outcome_state = "paper_submitted"
            elif plan_status in {"paper_completed", "completed", "closed"}:
                outcome_state = "paper_completed" if verified_journal_outcome else "audit_only"
            records.append(
                {
                    "trade_id": str(row.get("trade_id", "")) or str(row.get("timestamp_utc", "")),
                    "pair": str(row.get("pair", "")),
                    "regime": "unknown",
                    "strategy": str(row.get("strategy_id", "")),
                    "action": "",
                    "risk_configuration": str(row.get("intents_json", "")),
                    "submission_type": outcome_state,
                    "execution_quality_tier": outcome_state,
                    "realized_return": _safe_float(row.get("realized_return", "")) if verified_journal_outcome else "",
                    "outcome_state": outcome_state,
                }
            )
    frame = pd.DataFrame(
        records,
        columns=[
            "trade_id",
            "pair",
            "regime",
            "strategy",
            "action",
            "risk_configuration",
            "submission_type",
            "execution_quality_tier",
            "realized_return",
            "outcome_state",
        ],
    )
    return frame.drop_duplicates(subset=["trade_id", "pair", "submission_type"], keep="last")


def _feedback_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame([{"rows": 0, "verified_outcomes": 0, "paper_submitted": 0, "audit_only": 0}])
    return pd.DataFrame(
        [
            {
                "rows": int(len(frame)),
                "verified_outcomes": int(frame["outcome_state"].astype(str).eq("outcome_verified").sum()),
                "paper_submitted": int(frame["outcome_state"].astype(str).eq("paper_submitted").sum()),
                "audit_only": int(frame["outcome_state"].astype(str).eq("audit_only").sum()),
            }
        ]
    )


def _read_csv(path: Path) -> pd.DataFrame:
    if not path or not Path(path).exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path).fillna("")
    except (pd.errors.EmptyDataError, OSError, UnicodeDecodeError):
        return pd.DataFrame()


def _load_json(path: Path) -> dict[str, object]:
    if not path or not Path(path).exists():
        return {}
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _load_manifest(path: Path) -> dict[str, object]:
    return _load_json(path)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                rows.append(parsed)
    return rows


def _safe_float(value: object) -> object:
    try:
        return float(value)
    except (TypeError, ValueError):
        return ""


def _journal_row_has_verified_outcome(row: pd.Series) -> bool:
    status = str(row.get("plan_status", "") or "").strip().lower()
    if status not in {"paper_completed", "completed", "closed"}:
        return False
    try:
        float(row.get("realized_return", ""))
    except (TypeError, ValueError):
        return False
    snapshot = _json_dict(row.get("exit_snapshot_json", ""))
    if not snapshot:
        return False
    for key, value in snapshot.items():
        key_text = str(key).lower()
        if "exit" in key_text and "price" in key_text and str(value or "").strip():
            return True
    return False


def _json_dict(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return value
    text = str(value or "").strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_int(value: object) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _coerce_bool(value: object) -> bool:
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n", ""}:
        return False
    return bool(value)
