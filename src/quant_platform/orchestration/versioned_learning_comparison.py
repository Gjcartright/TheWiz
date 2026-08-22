"""Paired V1/V1.1 research comparisons with a fail-closed authority boundary."""

from __future__ import annotations

from quant_platform.orchestration.corrective_runtime import atomic_write_text

from quant_platform.orchestration.corrective_runtime import atomic_write_csv

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from quant_platform.runtime_types import ROOT, CommandResult

VERSION_LABELS = ("V1", "V1.1")
CONTRACT_PATH = Path("config/research_version_contracts.yaml")
REPORT_DIR = Path("reports/version_comparison")
MEMORY_PATH = Path("data/agent_memory/versioned_learning_memory.jsonl")

BASE_FEATURES = (
    ("math_v2_features", "mathematics", "reports/active/math_v2_acceptance.json"),
    (
        "exact_mode_features",
        "strategy",
        "reports/orchestration/teacher_council/teacher_proposals.csv",
    ),
    (
        "after_cost_outcomes",
        "costs",
        "data/ml/student_teacher_training_dataset.csv",
    ),
    (
        "funding_features",
        "costs",
        "reports/active/current_wizard_hyperliquid_funding_asset_results.csv",
    ),
    ("regime_features", "regime", "reports/regime_dataset_report.csv"),
    (
        "wizard_discovery_features",
        "discovery",
        "reports/crypto_wizards_scanner_rows.csv",
    ),
    (
        "local_outcome_features",
        "outcomes",
        "data/ml/student_teacher_training_dataset.csv",
    ),
)

PAPER_FEATURES = (
    (
        "panel_residual_probability",
        "paper_model",
        "reports/research/papers/reproductions/panel_residual_walkforward_predictions.csv",
    ),
    (
        "gph_fractional_d_proxy",
        "paper_stability",
        "reports/research/papers/reproductions/one_sided_stability_diagnostics.csv",
    ),
    (
        "cusum_break_alarm",
        "paper_stability",
        "reports/research/papers/reproductions/one_sided_stability_diagnostics.csv",
    ),
    (
        "characteristic_cluster",
        "paper_discovery",
        "reports/research/papers/reproductions/crypto_characteristic_candidates.csv",
    ),
    (
        "characteristic_discovery_score",
        "paper_discovery",
        "reports/research/papers/reproductions/crypto_characteristic_candidates.csv",
    ),
    (
        "dynamic_cost_vol_band",
        "paper_costs",
        "reports/research/papers/reproductions/dynamic_no_trade_band_results.csv",
    ),
)


def build_versioned_learning_comparison(
    *,
    root: Path = ROOT,
    now: datetime | None = None,
) -> CommandResult:
    """Build a matched champion/challenger evidence layer without changing execution."""

    generated_at = (now or datetime.now(UTC)).astimezone(UTC).isoformat()
    config = _load_contract(root)
    versions = config["versions"]
    code_ref_verified = _git_ref_exists(root, str(versions["V1"]["historical_code_ref"]))
    runtime_fingerprint = _runtime_fingerprint(root)

    contracts = _contract_frame(
        config,
        code_ref_verified=code_ref_verified,
        runtime_fingerprint=runtime_fingerprint,
    )
    code_delta = _code_delta_frame(root, config, code_ref_verified=code_ref_verified)
    readiness_path = root / "reports/orchestration/teacher_council/student_training_readiness.csv"
    readiness = _read_csv(readiness_path)
    supervised_ready = _scope_ready(readiness, "supervised_student")
    bandit_ready = _scope_ready(readiness, "contextual_bandit")
    features = _feature_registry(
        root,
        config,
        supervised_ready=supervised_ready,
        bandit_ready=bandit_ready,
    )

    teacher_path = root / "reports/dashboard/teacher_council_decisions.csv"
    teacher = _teacher_comparison(
        root,
        config,
        decisions=_read_csv(teacher_path),
        source_path=teacher_path,
        supervised_ready=supervised_ready,
        bandit_ready=bandit_ready,
    )
    rl = _rl_comparison(root, config, bandit_ready=bandit_ready)
    shadow, disagreements = _shadow_comparison(
        rl,
        minimum_outcomes=int(config["minimum_prospective_shadow_outcomes"]),
    )
    promotion = _promotion_gate(
        root,
        config,
        code_ref_verified=code_ref_verified,
        bandit_ready=bandit_ready,
        shadow=shadow,
    )
    red_team = _red_team(
        config,
        contracts=contracts,
        teacher=teacher,
        rl=rl,
        shadow=shadow,
        promotion=promotion,
        code_ref_verified=code_ref_verified,
        bandit_ready=bandit_ready,
    )

    output_dir = root / REPORT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "contracts_csv": output_dir / "version_contracts.csv",
        "contracts_json": output_dir / "version_contracts.json",
        "code_delta": output_dir / "version_code_delta.csv",
        "features": output_dir / "version_feature_registry.csv",
        "teacher": output_dir / "student_teacher_version_comparison.csv",
        "rl": output_dir / "rl_version_candidates.csv",
        "shadow": output_dir / "shadow_version_decisions.csv",
        "disagreements": output_dir / "shadow_disagreements.csv",
        "promotion": output_dir / "promotion_gate.csv",
        "red_team": output_dir / "version_red_team.csv",
        "red_team_md": output_dir / "version_red_team.md",
        "summary": output_dir / "v1_vs_v1_1_summary.md",
        "memory": root / MEMORY_PATH,
    }
    frames = {
        "contracts_csv": contracts,
        "code_delta": code_delta,
        "features": features,
        "teacher": teacher,
        "rl": rl,
        "shadow": shadow,
        "disagreements": disagreements,
        "promotion": promotion,
        "red_team": red_team,
    }
    for name, frame in frames.items():
        atomic_write_csv(frame, paths[name], index=False)

    contract_payload = {
        "schema_version": config["schema_version"],
        "comparison_runtime": config["comparison_runtime"],
        "generated_at": generated_at,
        "runtime_fingerprint": runtime_fingerprint,
        "versions": contracts.to_dict(orient="records"),
    }
    atomic_write_text(paths["contracts_json"], json.dumps(contract_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["memory"].parent.mkdir(parents=True, exist_ok=True)
    memory_records = _memory_records(shadow, generated_at=generated_at)
    atomic_write_text(paths["memory"], "".join(json.dumps(row, sort_keys=True) + "\n" for row in memory_records), encoding="utf-8")
    atomic_write_text(paths["red_team_md"], _red_team_markdown(red_team), encoding="utf-8")
    atomic_write_text(paths["summary"], _summary_markdown(
            config,
            contracts=contracts,
            teacher=teacher,
            rl=rl,
            shadow=shadow,
            disagreements=disagreements,
            promotion=promotion,
            generated_at=generated_at,
        ), encoding="utf-8")

    return CommandResult(
        paths=paths,
        summary={
            "research_versions": list(VERSION_LABELS),
            "teacher_comparison_rows": len(teacher),
            "rl_comparison_rows": len(rl),
            "shadow_comparison_rows": len(shadow),
            "comparison_keys": int(shadow["comparison_key"].nunique()),
            "disagreements": int(disagreements["disagreement_present"].sum()),
            "prospective_outcomes": int(shadow["outcome_known"].sum()),
            "v1_1_promoted": bool(
                promotion.loc[promotion["research_version"].eq("V1.1"), "promotion_authority"].any()
            ),
            "execution_authority_rows": int(shadow["execution_authority"].sum()),
            "red_team_failures": int(red_team["status"].eq("FAIL").sum()),
            "red_team_blockers": int(red_team["status"].eq("BLOCKED").sum()),
        },
    )


def _load_contract(root: Path) -> dict[str, Any]:
    path = root / CONTRACT_PATH
    if not path.exists():
        raise ValueError(f"version contract missing: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    versions = payload.get("versions", {})
    if tuple(versions) != VERSION_LABELS:
        raise ValueError("version contract must define V1 followed by V1.1")
    if str(versions["V1.1"].get("parent_version", "")) != "V1":
        raise ValueError("V1.1 parent_version must be V1")
    if bool(versions["V1.1"].get("execution_authority", True)):
        raise ValueError("V1.1 execution_authority must remain false")
    return payload


def _contract_frame(
    config: dict[str, Any],
    *,
    code_ref_verified: bool,
    runtime_fingerprint: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for version in VERSION_LABELS:
        contract = config["versions"][version]
        rows.append(
            {
                "research_version": version,
                "parent_version": str(contract.get("parent_version", "")),
                "version_role": str(contract["version_role"]),
                "historical_code_ref": str(contract["historical_code_ref"]),
                "historical_code_ref_verified": code_ref_verified if version == "V1" else False,
                "runtime_materialized": True,
                "historical_boundary": str(contract["historical_boundary"]),
                "comparison_runtime": str(config["comparison_runtime"]),
                "runtime_fingerprint": runtime_fingerprint,
                "historical_fidelity": (
                    "verified_code_ref_reconstructed_artifacts"
                    if version == "V1" and code_ref_verified
                    else "unverified_code_ref_reconstructed_artifacts"
                    if version == "V1"
                    else "current_worktree_paper_feature_gate"
                ),
                "paper_features_allowed": bool(contract["paper_features_allowed"]),
                "paper_teacher_weight": float(contract["paper_teacher_weight"]),
                "shadow_only": bool(contract["shadow_only"]),
                "promotion_authority": bool(contract["promotion_authority"]),
                "execution_authority": bool(contract["execution_authority"]),
                "notes": str(contract["notes"]),
            }
        )
    return pd.DataFrame(rows)


def _code_delta_frame(
    root: Path,
    config: dict[str, Any],
    *,
    code_ref_verified: bool,
) -> pd.DataFrame:
    v1_ref = str(config["versions"]["V1"]["historical_code_ref"])
    rows: list[dict[str, Any]] = [
        {
            "research_version": "V1",
            "parent_version": "",
            "version_role": "champion",
            "path": "",
            "change_status": "HISTORICAL_ANCHOR" if code_ref_verified else "UNVERIFIED_ANCHOR",
            "historical_code_ref": v1_ref,
            "comparison_runtime": config["comparison_runtime"],
            "execution_authority": False,
        }
    ]
    status = _git(root, ["status", "--porcelain=v1", "--untracked-files=all"])
    if status is None:
        rows.append(
            {
                "research_version": "V1.1",
                "parent_version": "V1",
                "version_role": "challenger",
                "path": "",
                "change_status": "WORKTREE_UNAVAILABLE",
                "historical_code_ref": "WORKTREE",
                "comparison_runtime": config["comparison_runtime"],
                "execution_authority": False,
            }
        )
    else:
        for line in status.splitlines() or ["   "]:
            change_status = line[:2].strip() or "CLEAN"
            path = line[3:].strip() if len(line) > 3 else ""
            rows.append(
                {
                    "research_version": "V1.1",
                    "parent_version": "V1",
                    "version_role": "challenger",
                    "path": path,
                    "change_status": change_status,
                    "historical_code_ref": "WORKTREE",
                    "comparison_runtime": config["comparison_runtime"],
                    "execution_authority": False,
                }
            )
    return pd.DataFrame(rows)


def _feature_registry(
    root: Path,
    config: dict[str, Any],
    *,
    supervised_ready: bool,
    bandit_ready: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for version in VERSION_LABELS:
        contract = config["versions"][version]
        for feature_name, group, source in BASE_FEATURES:
            source_exists = (root / source).exists()
            rows.append(
                _feature_row(
                    version,
                    contract,
                    feature_name=feature_name,
                    feature_group=group,
                    source=source,
                    source_exists=source_exists,
                    is_paper_feature=False,
                    available=source_exists,
                    blocker="" if source_exists else "base_feature_source_missing",
                    supervised_ready=supervised_ready,
                    bandit_ready=bandit_ready,
                )
            )
        for feature_name, group, source in PAPER_FEATURES:
            source_exists = (root / source).exists()
            allowed = bool(contract["paper_features_allowed"])
            rows.append(
                _feature_row(
                    version,
                    contract,
                    feature_name=feature_name,
                    feature_group=group,
                    source=source,
                    source_exists=source_exists,
                    is_paper_feature=True,
                    available=allowed and source_exists,
                    blocker=(
                        "paper_features_forbidden_in_v1"
                        if not allowed
                        else "paper_feature_source_missing"
                        if not source_exists
                        else "research_only_no_point_in_time_or_promotion_authority"
                    ),
                    supervised_ready=supervised_ready,
                    bandit_ready=bandit_ready,
                )
            )
    return pd.DataFrame(rows)


def _feature_row(
    version: str,
    contract: dict[str, Any],
    *,
    feature_name: str,
    feature_group: str,
    source: str,
    source_exists: bool,
    is_paper_feature: bool,
    available: bool,
    blocker: str,
    supervised_ready: bool,
    bandit_ready: bool,
) -> dict[str, Any]:
    paper_blocked = is_paper_feature
    return {
        "research_version": version,
        "parent_version": str(contract.get("parent_version", "")),
        "version_role": str(contract["version_role"]),
        "feature_name": feature_name,
        "feature_group": feature_group,
        "is_paper_feature": is_paper_feature,
        "source_exists": source_exists,
        "feature_available": available,
        "student_training_eligible": available and not paper_blocked and supervised_ready,
        "rl_observation_eligible": available and not paper_blocked and bandit_ready,
        "reward_eligible": (
            available and not paper_blocked and bandit_ready and feature_group == "outcomes"
        ),
        "teacher_vote_eligible": available and not paper_blocked,
        "promotion_authority": False,
        "execution_authority": False,
        "blocker": blocker,
        "evidence_path": source,
    }


def _teacher_comparison(
    root: Path,
    config: dict[str, Any],
    *,
    decisions: pd.DataFrame,
    source_path: Path,
    supervised_ready: bool,
    bandit_ready: bool,
) -> pd.DataFrame:
    source_rows = decisions.to_dict(orient="records")
    if not source_rows:
        source_rows = [
            {
                "context_id": "teacher_source_missing",
                "pair": "SOURCE_MISSING",
                "timeframe": "",
                "selected_mode": "",
                "action": "abstain",
                "status": "BLOCKED",
                "blocker_codes": "teacher_council_decisions_missing",
                "evidence_paths": _relative(source_path, root),
            }
        ]
    paper_gate = "reports/research/papers/reproductions/paper_reproduction_gate.csv"
    rows: list[dict[str, Any]] = []
    for source in source_rows:
        context_id = _text(source.get("context_id")) or _stable_id(
            "teacher_context", source.get("pair"), source.get("timeframe")
        )
        comparison_key = f"teacher|{context_id}"
        base_action = _text(source.get("action")) or "abstain"
        base_blocker = _text(source.get("blocker_codes")) or _text(source.get("reason"))
        for version in VERSION_LABELS:
            contract = config["versions"][version]
            paper_allowed = bool(contract["paper_features_allowed"])
            overlay_status = "not_available_in_v1"
            blocker = base_blocker
            if paper_allowed:
                overlay_status = "advisory_only_zero_teacher_weight"
                blocker = _join_codes(
                    blocker,
                    "paper_teacher_weight_zero",
                    "paper_point_in_time_vintage_unconfirmed",
                    "paper_signal_gate_failed",
                )
            rows.append(
                {
                    "research_version": version,
                    "parent_version": str(contract.get("parent_version", "")),
                    "version_role": str(contract["version_role"]),
                    "comparison_key": comparison_key,
                    "context_id": context_id,
                    "pair": _text(source.get("pair")),
                    "timeframe": _text(source.get("timeframe")),
                    "selected_mode": _text(source.get("selected_mode")),
                    "base_action": base_action,
                    "version_action": base_action,
                    "source_status": _text(source.get("status")) or "BLOCKED",
                    "paper_overlay_status": overlay_status,
                    "paper_teacher_weight": float(contract["paper_teacher_weight"]),
                    "supervised_student_ready": supervised_ready,
                    "contextual_bandit_ready": bandit_ready,
                    "decision_changed_from_parent": False,
                    "shadow_only": True,
                    "promotion_authority": False,
                    "execution_authority": False,
                    "blocker": blocker,
                    "evidence_path": _join_codes(
                        _text(source.get("evidence_paths")),
                        _relative(source_path, root),
                        paper_gate if paper_allowed else "",
                    ),
                }
            )
    return pd.DataFrame(rows)


def _rl_comparison(
    root: Path,
    config: dict[str, Any],
    *,
    bandit_ready: bool,
) -> pd.DataFrame:
    idea_path = root / "reports/agents/rl_ideas.csv"
    ideas = _read_csv(idea_path)
    if ideas.empty:
        idea_path = root / "reports/brain/shadow_rl_candidate_set.csv"
        ideas = _read_csv(idea_path)
    paper_path = root / "reports/research/papers/reproductions/crypto_characteristic_candidates.csv"
    papers = _read_csv(paper_path)
    paper_pairs = {
        _pair_key(value) for value in papers.get("pair", pd.Series(dtype=str)).astype(str)
    }
    candidates: list[dict[str, Any]] = []
    for index, source in ideas.iterrows():
        pair = _text(source.get("pair"))
        timeframe = _text(source.get("timeframe")) or "unknown"
        strategy = _text(source.get("strategy")) or _text(source.get("strategy_name")) or "rl_idea"
        candidate_id = _text(source.get("idea_id")) or _text(source.get("candidate_id"))
        candidate_id = candidate_id or _stable_id("base_rl", pair, timeframe, strategy, index)
        candidates.append(
            {
                "comparison_key": f"base_rl|{candidate_id}",
                "candidate_id": candidate_id,
                "pair": pair,
                "timeframe": timeframe,
                "strategy": strategy,
                "source_family": "base_rl",
                "source_status": _text(source.get("status")) or "candidate",
                "score": _number(source.get("confidence_score"), source.get("confidence")),
                "paper_overlay_match": _pair_key(pair) in paper_pairs,
                "paper_only": False,
                "point_in_time_status": "unverified",
                "evidence_path": _join_codes(
                    _text(source.get("evidence_path")), _relative(idea_path, root)
                ),
            }
        )
    for index, source in papers.iterrows():
        pair = _text(source.get("pair"))
        candidate_id = _stable_id("paper_characteristic", pair, index)
        candidates.append(
            {
                "comparison_key": f"paper_characteristic|{pair}|1d|{index}",
                "candidate_id": candidate_id,
                "pair": pair,
                "timeframe": "1d",
                "strategy": "characteristic_similarity_discovery",
                "source_family": "research_paper",
                "source_status": _text(source.get("decision_bucket")) or "RESEARCH_ONLY",
                "score": _number(source.get("discovery_score")),
                "paper_overlay_match": True,
                "paper_only": True,
                "point_in_time_status": (
                    "confirmed" if _truthy(source.get("point_in_time_vintage")) else "unconfirmed"
                ),
                "evidence_path": _join_codes(
                    _text(source.get("evidence_path")), _relative(paper_path, root)
                ),
            }
        )
    if not candidates:
        candidates.append(
            {
                "comparison_key": "rl|source_missing",
                "candidate_id": "rl_source_missing",
                "pair": "SOURCE_MISSING",
                "timeframe": "",
                "strategy": "",
                "source_family": "missing",
                "source_status": "BLOCKED",
                "score": None,
                "paper_overlay_match": False,
                "paper_only": False,
                "point_in_time_status": "unconfirmed",
                "evidence_path": _join_codes(
                    _relative(idea_path, root), _relative(paper_path, root)
                ),
            }
        )

    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        for version in VERSION_LABELS:
            contract = config["versions"][version]
            paper_allowed = bool(contract["paper_features_allowed"])
            available = not candidate["paper_only"] or paper_allowed
            paper_overlay = paper_allowed and bool(candidate["paper_overlay_match"])
            blockers = []
            if not available:
                blockers.append("paper_candidate_not_available_in_v1")
            if not bandit_ready:
                blockers.append("contextual_bandit_propensity_or_counterfactual_gate_failed")
            if candidate["point_in_time_status"] != "confirmed":
                blockers.append("point_in_time_status_unconfirmed")
            if paper_overlay:
                blockers.extend(
                    [
                        "paper_overlay_research_only",
                        "paper_reproduction_signal_gate_failed",
                    ]
                )
            rows.append(
                {
                    "research_version": version,
                    "parent_version": str(contract.get("parent_version", "")),
                    "version_role": str(contract["version_role"]),
                    "comparison_key": candidate["comparison_key"],
                    "candidate_id": candidate["candidate_id"],
                    "pair": candidate["pair"],
                    "timeframe": candidate["timeframe"],
                    "strategy": candidate["strategy"],
                    "source_family": candidate["source_family"],
                    "candidate_available": available,
                    "paper_overlay_available": paper_overlay,
                    "paper_only_candidate": bool(candidate["paper_only"]),
                    "candidate_status": candidate["source_status"]
                    if available
                    else "NOT_AVAILABLE",
                    "candidate_score": candidate["score"] if available else None,
                    "point_in_time_status": candidate["point_in_time_status"],
                    "contextual_bandit_ready": bandit_ready,
                    "rl_policy_training_eligible": bool(
                        available
                        and bandit_ready
                        and not paper_overlay
                        and candidate["point_in_time_status"] == "confirmed"
                    ),
                    "reward_eligible": False,
                    "shadow_only": True,
                    "promotion_authority": False,
                    "execution_authority": False,
                    "blocker": ";".join(dict.fromkeys(blockers)),
                    "evidence_path": candidate["evidence_path"],
                }
            )
    return pd.DataFrame(rows)


def _shadow_comparison(
    rl: pd.DataFrame,
    *,
    minimum_outcomes: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    shadow = rl.copy()
    shadow["shadow_research_action"] = shadow["candidate_available"].map(
        {True: "observe_candidate", False: "absent_from_version"}
    )
    shadow["order_action"] = "abstain"
    shadow["outcome_known"] = False
    shadow["hypothetical_profit_after_cost"] = pd.NA
    shadow["winner_eligible"] = False
    shadow["cross_version_memory_pooling"] = False
    shadow["execution_authority"] = False

    rows: list[dict[str, Any]] = []
    for key, group in shadow.groupby("comparison_key", sort=True):
        by_version = group.set_index("research_version")
        v1 = by_version.loc["V1"]
        v1_1 = by_version.loc["V1.1"]
        availability_changed = bool(v1["candidate_available"]) != bool(v1_1["candidate_available"])
        overlay_changed = bool(v1_1["paper_overlay_available"])
        disagreement_type = (
            "candidate_availability"
            if availability_changed
            else "paper_overlay"
            if overlay_changed
            else "none"
        )
        rows.append(
            {
                "research_version": "V1.1",
                "parent_version": "V1",
                "version_role": "challenger",
                "comparison_key": key,
                "pair": v1_1["pair"],
                "timeframe": v1_1["timeframe"],
                "strategy": v1_1["strategy"],
                "v1_candidate_available": bool(v1["candidate_available"]),
                "v1_1_candidate_available": bool(v1_1["candidate_available"]),
                "disagreement_present": availability_changed or overlay_changed,
                "disagreement_type": disagreement_type,
                "outcome_known": False,
                "winner": "UNRESOLVED",
                "prospective_outcomes_observed": 0,
                "prospective_outcomes_required": minimum_outcomes,
                "promotion_authority": False,
                "execution_authority": False,
                "blocker": "matched_prospective_outcome_missing",
                "evidence_path": _join_codes(
                    _text(v1["evidence_path"]), _text(v1_1["evidence_path"])
                ),
            }
        )
    return shadow, pd.DataFrame(rows)


def _promotion_gate(
    root: Path,
    config: dict[str, Any],
    *,
    code_ref_verified: bool,
    bandit_ready: bool,
    shadow: pd.DataFrame,
) -> pd.DataFrame:
    gate_path = root / "reports/research/papers/reproductions/paper_reproduction_gate.csv"
    audit_path = root / "reports/research/papers/reproductions/paper_reproduction_data_audit.csv"
    metrics_path = (
        root / "reports/research/papers/reproductions/panel_residual_walkforward_metrics.csv"
    )
    gates = _read_csv(gate_path)
    audit = _read_csv(audit_path)
    metrics = _read_csv(metrics_path)
    paper_signal_ready = bool(
        not gates.empty
        and gates["signal_eligible"].map(_truthy).all()
        and gates["exact_paper_reproduction"].map(_truthy).all()
    )
    point_in_time_ready = bool(
        not audit.empty and audit["point_in_time_vintage"].map(_truthy).all()
    )
    outcomes = int(shadow["outcome_known"].map(_truthy).sum())
    minimum_outcomes = int(config["minimum_prospective_shadow_outcomes"])
    incremental_accuracy = _incremental_balanced_accuracy(metrics)
    minimum_accuracy = float(config["minimum_incremental_balanced_accuracy"])

    rows = [
        {
            "research_version": "V1",
            "parent_version": "",
            "version_role": "champion",
            "status": "REFERENCE_ONLY",
            "historical_code_ref_verified": code_ref_verified,
            "paper_signal_gate_passed": False,
            "point_in_time_vintage_passed": False,
            "contextual_bandit_ready": bandit_ready,
            "prospective_outcomes_observed": outcomes,
            "prospective_outcomes_required": minimum_outcomes,
            "incremental_balanced_accuracy": None,
            "minimum_incremental_balanced_accuracy": minimum_accuracy,
            "promotion_authority": False,
            "execution_authority": False,
            "blocker": _join_codes(
                "comparison_does_not_change_existing_execution_authority",
                "historical_artifacts_reconstructed_with_shared_runtime",
            ),
            "evidence_path": str(CONTRACT_PATH),
        },
        {
            "research_version": "V1.1",
            "parent_version": "V1",
            "version_role": "challenger",
            "status": "BLOCKED_RESEARCH_CHALLENGER",
            "historical_code_ref_verified": code_ref_verified,
            "paper_signal_gate_passed": paper_signal_ready,
            "point_in_time_vintage_passed": point_in_time_ready,
            "contextual_bandit_ready": bandit_ready,
            "prospective_outcomes_observed": outcomes,
            "prospective_outcomes_required": minimum_outcomes,
            "incremental_balanced_accuracy": incremental_accuracy,
            "minimum_incremental_balanced_accuracy": minimum_accuracy,
            "promotion_authority": False,
            "execution_authority": False,
            "blocker": _join_codes(
                "paper_signal_gate_failed" if not paper_signal_ready else "",
                "paper_point_in_time_vintage_failed" if not point_in_time_ready else "",
                "contextual_bandit_gate_failed" if not bandit_ready else "",
                "minimum_prospective_shadow_outcomes_not_met"
                if outcomes < minimum_outcomes
                else "",
                "minimum_incremental_accuracy_not_met"
                if incremental_accuracy is None or incremental_accuracy < minimum_accuracy
                else "",
            ),
            "evidence_path": _join_codes(
                _relative(gate_path, root),
                _relative(audit_path, root),
                _relative(metrics_path, root),
            ),
        },
    ]
    return pd.DataFrame(rows)


def _red_team(
    config: dict[str, Any],
    *,
    contracts: pd.DataFrame,
    teacher: pd.DataFrame,
    rl: pd.DataFrame,
    shadow: pd.DataFrame,
    promotion: pd.DataFrame,
    code_ref_verified: bool,
    bandit_ready: bool,
) -> pd.DataFrame:
    labels_complete = all(
        set(frame["research_version"].dropna().astype(str)).issubset(set(VERSION_LABELS))
        for frame in (contracts, teacher, rl, shadow, promotion)
    )
    pair_counts = shadow.groupby("comparison_key")["research_version"].agg(
        lambda values: set(values.astype(str))
    )
    pairing_complete = bool((pair_counts == set(VERSION_LABELS)).all())
    execution_rows = int(
        sum(frame["execution_authority"].map(_truthy).sum() for frame in (teacher, rl, shadow))
    )
    paper_features_in_v1 = int(
        rl.loc[rl["research_version"].eq("V1"), "paper_overlay_available"].map(_truthy).sum()
    )
    v1_1_promotion = bool(
        promotion.loc[promotion["research_version"].eq("V1.1"), "promotion_authority"]
        .map(_truthy)
        .any()
    )
    v1_1_gate = promotion.loc[promotion["research_version"].eq("V1.1")].iloc[0]
    paper_signal_ready = _truthy(v1_1_gate["paper_signal_gate_passed"])
    paper_vintage_ready = _truthy(v1_1_gate["point_in_time_vintage_passed"])
    incremental_accuracy = _number(v1_1_gate["incremental_balanced_accuracy"])
    minimum_accuracy = float(v1_1_gate["minimum_incremental_balanced_accuracy"])
    checks = {
        "V1": [
            (
                "historical_code_ref",
                "PASS" if code_ref_verified else "FAIL",
                str(code_ref_verified),
                "V1 commit object is available",
                "verify or restore the immutable V1 commit reference",
            ),
            (
                "historical_artifact_fidelity",
                "BLOCKED",
                "reconstructed with shared current runtime",
                "exact immutable V1 artifacts",
                "preserve future run manifests and artifact snapshots by research_version",
            ),
            (
                "paper_feature_exclusion",
                "PASS" if paper_features_in_v1 == 0 else "FAIL",
                str(paper_features_in_v1),
                "0 V1 paper overlays",
                "remove all paper-derived V1 features",
            ),
            (
                "paired_comparison_keys",
                "PASS" if pairing_complete else "FAIL",
                str(pairing_complete),
                "every key has exactly V1 and V1.1",
                "repair unmatched shadow rows",
            ),
        ],
        "V1.1": [
            (
                "version_labels_complete",
                "PASS" if labels_complete else "FAIL",
                str(labels_complete),
                "all rows explicitly labeled V1 or V1.1",
                "reject unlabeled learning artifacts",
            ),
            (
                "paper_teacher_weight",
                "PASS"
                if float(config["versions"]["V1.1"]["paper_teacher_weight"]) == 0.0
                else "FAIL",
                str(config["versions"]["V1.1"]["paper_teacher_weight"]),
                "0.0 until prospective acceptance",
                "reset paper teacher weight to zero",
            ),
            (
                "paper_reproduction_signal_gate",
                "PASS" if paper_signal_ready else "FAIL",
                str(paper_signal_ready),
                "all paper experiments exact and signal eligible",
                "retain the paper layer as research-only and reproduce failed experiments",
            ),
            (
                "paper_point_in_time_vintages",
                "PASS" if paper_vintage_ready else "FAIL",
                str(paper_vintage_ready),
                "all historical features captured as-traded",
                "collect immutable prospective snapshots before using paper features",
            ),
            (
                "paper_incremental_accuracy",
                "PASS"
                if incremental_accuracy is not None and incremental_accuracy >= minimum_accuracy
                else "FAIL",
                str(incremental_accuracy),
                f">= {minimum_accuracy}",
                "beat simple baselines out of sample before model or teacher escalation",
            ),
            (
                "contextual_bandit_support",
                "PASS" if bandit_ready else "BLOCKED",
                str(bandit_ready),
                "real propensities and counterfactual support",
                "collect behavior-policy propensities and supported alternatives",
            ),
            (
                "prospective_matched_outcomes",
                "PASS" if shadow["outcome_known"].map(_truthy).any() else "BLOCKED",
                str(int(shadow["outcome_known"].map(_truthy).sum())),
                str(config["minimum_prospective_shadow_outcomes"]),
                "collect matched V1/V1.1 outcomes without changing orders",
            ),
            (
                "challenger_promotion_firewall",
                "PASS" if not v1_1_promotion else "FAIL",
                str(v1_1_promotion),
                "no promotion while gates fail",
                "force V1.1 back to research-only",
            ),
        ],
    }
    rows: list[dict[str, Any]] = []
    for version in VERSION_LABELS:
        parent = str(config["versions"][version].get("parent_version", ""))
        for check, status, observed, required, corrective_action in checks[version]:
            rows.append(
                {
                    "research_version": version,
                    "parent_version": parent,
                    "version_role": config["versions"][version]["version_role"],
                    "check": check,
                    "status": status,
                    "observed": observed,
                    "required": required,
                    "risk": "comparison_or_authority_misstatement" if status != "PASS" else "",
                    "corrective_action": corrective_action,
                    "promotion_authority": False,
                    "execution_authority": False,
                }
            )
    for version in VERSION_LABELS:
        rows.append(
            {
                "research_version": version,
                "parent_version": str(config["versions"][version].get("parent_version", "")),
                "version_role": config["versions"][version]["version_role"],
                "check": "execution_authority_firewall",
                "status": "PASS" if execution_rows == 0 else "FAIL",
                "observed": str(execution_rows),
                "required": "0 rows with execution authority",
                "risk": "research_challenger_could_submit_orders" if execution_rows else "",
                "corrective_action": "block all version-comparison order actions",
                "promotion_authority": False,
                "execution_authority": False,
            }
        )
    return pd.DataFrame(rows)


def _memory_records(shadow: pd.DataFrame, *, generated_at: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in shadow.to_dict(orient="records"):
        version = _text(source["research_version"])
        comparison_key = _text(source["comparison_key"])
        rows.append(
            {
                "event_id": _stable_id("version_memory", version, comparison_key),
                "event_type": "versioned_shadow_candidate",
                "research_version": version,
                "parent_version": _text(source["parent_version"]),
                "version_role": _text(source["version_role"]),
                "memory_partition": version,
                "comparison_key": comparison_key,
                "candidate_id": _text(source["candidate_id"]),
                "pair": _text(source["pair"]),
                "timeframe": _text(source["timeframe"]),
                "strategy": _text(source["strategy"]),
                "candidate_available": bool(source["candidate_available"]),
                "paper_overlay_available": bool(source["paper_overlay_available"]),
                "outcome_known": False,
                "outcome_label": None,
                "cross_version_memory_pooling": False,
                "promotion_authority": False,
                "execution_authority": False,
                "evidence_path": _text(source["evidence_path"]),
                "generated_at": generated_at,
            }
        )
    return rows


def _summary_markdown(
    config: dict[str, Any],
    *,
    contracts: pd.DataFrame,
    teacher: pd.DataFrame,
    rl: pd.DataFrame,
    shadow: pd.DataFrame,
    disagreements: pd.DataFrame,
    promotion: pd.DataFrame,
    generated_at: str,
) -> str:
    v1_1 = promotion.loc[promotion["research_version"].eq("V1.1")].iloc[0]
    paper_candidates = int(
        rl.loc[rl["research_version"].eq("V1.1") & rl["paper_only_candidate"].map(_truthy)].shape[0]
    )
    return "\n".join(
        [
            "# V1 vs V1.1 Research Comparison",
            "",
            f"Generated: `{generated_at}`",
            "",
            "## Version Contract",
            "",
            (
                "- **V1** is the pre-paper champion reference anchored to commit "
                f"`{contracts.iloc[0]['historical_code_ref']}`."
            ),
            (
                "- **V1.1** is the paper-augmented challenger. It remains shadow-only with zero "
                "promotion or execution authority."
            ),
            (
                f"- Both use `{config['comparison_runtime']}` so paper features are the controlled "
                "difference, not a different engine."
            ),
            (
                "- V1 historical artifacts are reconstructed; this limitation is explicit and "
                "blocks claims of an exact historical replay."
            ),
            "",
            "## Current Evidence",
            "",
            f"- Teacher rows: **{len(teacher)}** ({len(teacher) // 2} paired contexts).",
            f"- RL candidate rows: **{len(rl)}** ({len(rl) // 2} paired candidates).",
            f"- V1.1 paper-only candidates: **{paper_candidates}**.",
            (
                f"- Shadow rows: **{len(shadow)}**; known prospective outcomes: "
                f"**{int(shadow['outcome_known'].map(_truthy).sum())}**."
            ),
            (
                "- Candidate or overlay disagreements: "
                f"**{int(disagreements['disagreement_present'].map(_truthy).sum())}**."
            ),
            "",
            "## Decision",
            "",
            f"V1.1 status: **{v1_1['status']}**.",
            "",
            f"Blockers: `{v1_1['blocker']}`.",
            "",
            (
                "The papers can propose student features, RL observations, and shadow candidates, "
                "but they cannot vote, provide rewards, promote strategies, resize orders, or "
                "submit trades until point-in-time, prospective, and incremental-performance "
                "gates pass."
            ),
            "",
        ]
    )


def _red_team_markdown(red_team: pd.DataFrame) -> str:
    lines = ["# V1/V1.1 Red Team", ""]
    for row in red_team.to_dict(orient="records"):
        lines.extend(
            [
                f"## {row['research_version']}: {row['check']}",
                "",
                f"- Status: **{row['status']}**",
                f"- Observed: `{row['observed']}`",
                f"- Required: `{row['required']}`",
                f"- Corrective action: {row['corrective_action']}",
                "",
            ]
        )
    return "\n".join(lines)


def _incremental_balanced_accuracy(metrics: pd.DataFrame) -> float | None:
    required = {"model", "population", "balanced_accuracy"}
    if metrics.empty or not required.issubset(metrics.columns):
        return None
    whole = metrics.loc[metrics["population"].astype(str).eq("whole_asset_holdout")].copy()
    whole["balanced_accuracy"] = pd.to_numeric(whole["balanced_accuracy"], errors="coerce")
    means = whole.groupby("model")["balanced_accuracy"].mean().dropna()
    if "pooled_logistic" not in means:
        return None
    baselines = means.drop(labels=["pooled_logistic"], errors="ignore")
    if baselines.empty:
        return None
    return float(means["pooled_logistic"] - baselines.max())


def _scope_ready(readiness: pd.DataFrame, scope: str) -> bool:
    required = {"scope", "status"}
    if readiness.empty or not required.issubset(readiness.columns):
        return False
    selected = readiness.loc[readiness["scope"].astype(str).eq(scope)]
    return bool(not selected.empty and selected["status"].astype(str).eq("PASS").all())


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError):
        return pd.DataFrame()


def _runtime_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    paths = [
        CONTRACT_PATH,
        Path("config/research_paper_assessments.yaml"),
        Path("src/quant_platform/research_paper_reproduction.py"),
        Path("src/quant_platform/research_paper_math_audit.py"),
        Path("src/quant_platform/orchestration/teacher_council.py"),
        Path("src/quant_platform/orchestration/student_readiness.py"),
    ]
    for relative in paths:
        path = root / relative
        digest.update(str(relative).encode())
        if path.exists():
            digest.update(path.read_bytes())
        else:
            digest.update(b"MISSING")
    return digest.hexdigest()


def _git_ref_exists(root: Path, ref: str) -> bool:
    return _git(root, ["cat-file", "-e", f"{ref}^{{commit}}"], allow_empty=True) is not None


def _git(root: Path, args: list[str], *, allow_empty: bool = False) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    output = result.stdout.rstrip("\n")
    return output if output or allow_empty else ""


def _stable_id(prefix: str, *parts: Any) -> str:
    payload = "|".join(_text(part) for part in parts)
    return f"{prefix}_{hashlib.sha256(payload.encode()).hexdigest()[:20]}"


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _pair_key(value: Any) -> str:
    return _text(value).upper().replace("/", "-").replace("_", "-").replace(" ", "")


def _text(value: Any) -> str:
    if value is None or value is pd.NA:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "<na>"} else text


def _number(*values: Any) -> float | None:
    for value in values:
        number = pd.to_numeric(value, errors="coerce")
        if pd.notna(number):
            return float(number)
    return None


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or value is pd.NA:
        return False
    if isinstance(value, (int, float)):
        return bool(value) and not pd.isna(value)
    return str(value).strip().lower() in {"1", "true", "yes", "pass", "ready"}


def _join_codes(*values: Any) -> str:
    parts: list[str] = []
    for value in values:
        for part in _text(value).split(";"):
            part = part.strip()
            if part and part not in parts:
                parts.append(part)
    return ";".join(parts)
