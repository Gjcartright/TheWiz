"""Lightweight dynamic-agent stage runner with no ML pipeline dependency."""

from __future__ import annotations

from pathlib import Path

from quant_platform.math_v2_acceptance import build_math_v2_acceptance
from quant_platform.orchestration.contracts import CandidateIdentity
from quant_platform.orchestration.copula_shadow_comparison import build_copula_shadow_comparison
from quant_platform.orchestration.dynamic_rollout import build_dynamic_rollout_gate
from quant_platform.orchestration.dynamic_supreme_team import build_dynamic_supreme_team_checkpoint
from quant_platform.orchestration.hyperliquid_research_cycle import run_hyperliquid_research_cycle
from quant_platform.orchestration.state import OrchestratorState, StageResult, StageStatus
from quant_platform.orchestration.teacher_adapters import build_teacher_evidence_adapters
from quant_platform.orchestration.teacher_control_plane import build_teacher_council_control_plane
from quant_platform.orchestration.teacher_evidence_materializer import materialize_teacher_evidence
from quant_platform.orchestration.venue_capabilities import load_venue_policy


ROOT = Path(__file__).resolve().parents[3]
DYNAMIC_STAGE_GROUPS: dict[str, list[str]] = {
    "copula_shadow": ["copula_shadow_comparison", "dynamic_rollout_gate"],
    "dynamic_rollout": ["dynamic_rollout_gate"],
    "dynamic_supreme_team": ["dynamic_supreme_team_checkpoint"],
    "math_v2": ["math_v2_acceptance"],
    "teacher_council": [
        "math_v2_acceptance",
        "teacher_evidence_materializer",
        "teacher_evidence_adapters",
        "teacher_council_control_plane",
    ],
    "hyperliquid_research": ["hyperliquid_research_cycle"],
}
DYNAMIC_STAGE_NAMES = frozenset(stage for stages in DYNAMIC_STAGE_GROUPS.values() for stage in stages)


def dynamic_stages_for_group(group: str) -> list[str]:
    return DYNAMIC_STAGE_GROUPS.get(group, [group])


def run_dynamic_stage(stage: str, state: OrchestratorState, root: Path = ROOT) -> StageResult:
    if state.dry_run:
        return StageResult(stage=stage, status=StageStatus.DRY_RUN, reason="stage_would_run", next_step="remove --dry-run to execute")
    if stage == "math_v2_acceptance":
        result = build_math_v2_acceptance(root=root)
        passed = result["status"] == "passed"
        return StageResult(
            stage=stage,
            status=StageStatus.PASSED if passed else StageStatus.BLOCKED,
            blocker="" if passed else "core_math_v2_checks_failed",
            reason=f"math_v2_checks={result['passed_checks']}/{result['total_checks']}",
            evidence_path=f"{result['marker']};{result['reconciliation']};{result['statistical_validity']}",
            next_step="build point-in-time exact-mode inputs" if passed else "repair failed Math V2 checks",
            rows=int(result["total_checks"]),
        )
    if stage == "hyperliquid_research_cycle":
        result = run_hyperliquid_research_cycle(root=root, collect_l2=False)
        ready = bool(result.get("execution_allowed", False))
        return StageResult(
            stage=stage,
            status=StageStatus.PASSED if ready else StageStatus.BLOCKED,
            blocker="" if ready else str(result.get("blockers", "hyperliquid_research_cycle_blocked")),
            reason=(
                f"run_id={result.get('run_id', '')};candidate_set_id={result.get('candidate_set_id', '')};"
                f"blocked_steps={result.get('blocked_steps', 0)}"
            ),
            evidence_path=f"{result['receipt']};{result['receipt_json']};{result['summary']}",
            next_step=(
                "authorize a separately approved bounded Testnet smoke test"
                if ready
                else "resolve the ranked empirical blockers in the research-cycle receipt"
            ),
            rows=int(result.get("step_count", 0)),
        )
    if stage == "teacher_evidence_adapters":
        result = build_teacher_evidence_adapters(root=root)
        ready = result["status"] == "READY_FOR_COUNCIL"
        return StageResult(
            stage=stage,
            status=StageStatus.PASSED if ready else StageStatus.BLOCKED,
            blocker="" if ready else "teacher_evidence_inputs_not_ready",
            reason=f"proposals={result['proposal_count']};assessments={result['assessment_count']}",
            evidence_path=f"{result['readiness']};{result['markdown']}",
            next_step=(
                "run deterministic council arbitration"
                if ready
                else "produce seven point-in-time mode replays and six independent critic inputs"
            ),
            rows=int(result["proposal_count"]) + int(result["assessment_count"]),
        )
    if stage == "teacher_evidence_materializer":
        result = materialize_teacher_evidence(root=root)
        materialized = result["status"] == "MATERIALIZED"
        return StageResult(
            stage=stage,
            status=StageStatus.PASSED if materialized else StageStatus.BLOCKED,
            blocker="" if materialized else "teacher_evidence_materialization_blocked",
            reason=(
                f"contexts={result['contexts']};teachers={result['teacher_rows']};"
                f"critics={result['critic_rows']};critic_vetoes={result['critic_vetoes']}"
            ),
            evidence_path=(
                f"{result['teacher_coverage']};{result['critic_coverage']};"
                f"{result['next_actions']};{result['summary']}"
            ),
            next_step=(
                "validate adapter contracts and arbitrate in shadow mode"
                if materialized
                else "refresh Hyperliquid histories and Math V2 evidence"
            ),
            rows=int(result["teacher_rows"]) + int(result["critic_rows"]),
        )
    if stage == "copula_shadow_comparison":
        return _copula_shadow_result(stage, state=state, root=root)
    if stage == "dynamic_rollout_gate":
        result = build_dynamic_rollout_gate(root=root)
        ready = result["copula_rollout_status"] == "ready"
        return StageResult(
            stage=stage,
            status=StageStatus.PASSED if ready else StageStatus.BLOCKED,
            blocker="" if ready else "copula_shadow_gate_not_yet_satisfied",
            reason=f"copula_shadow_qualifying_agreements={result['qualifying_agreements']}",
            evidence_path=f"{result['csv']};{result['markdown']}",
            next_step="start one additional strategy cell in shadow mode" if ready else "collect complete, fresh, veto-free Copula comparison events",
            rows=7,
        )
    if stage == "dynamic_supreme_team_checkpoint":
        result = build_dynamic_supreme_team_checkpoint(root=root)
        return StageResult(stage=stage, status=StageStatus.PASSED, reason=f"dynamic_supreme_team_qualifying_agreements={result['qualifying_agreements']}", evidence_path=f"{result['csv']};{result['markdown']}", next_step="apply checkpoint repairs before expansion", rows=4)
    if stage == "teacher_council_control_plane":
        result = build_teacher_council_control_plane(root=root)
        ready = result["status"] == "READY_FOR_SHADOW_ONLY"
        return StageResult(
            stage=stage,
            status=StageStatus.PASSED if ready else StageStatus.BLOCKED,
            blocker="" if ready else "teacher_council_readiness_blocked",
            reason=(
                f"teacher_council_status={result['status']};"
                f"proposals={result['proposal_count']};assessments={result['assessment_count']}"
            ),
            evidence_path=f"{result['readiness']};{result['markdown']}",
            next_step=(
                "run the complete council in shadow mode"
                if ready
                else (
                    "resolve teacher blockers and critic vetoes, then build the leakage-safe student dataset"
                    if int(result["proposal_count"]) > 0 and int(result["assessment_count"]) > 0
                    else "emit seven point-in-time teacher proposals and six independent critic assessments"
                )
            ),
            rows=int(result["teacher_count"]) + int(result["critic_count"]),
        )
    return StageResult(stage=stage, status=StageStatus.FAILED, blocker="unknown_dynamic_stage", next_step="check dynamic stage name")


def _copula_shadow_result(stage: str, *, state: OrchestratorState, root: Path) -> StageResult:
    pair = str(state.pair_id or "")
    if "/" not in pair:
        return StageResult(stage=stage, status=StageStatus.SKIPPED, blocker="copula_shadow_requires_pair_id", next_step="rerun with --pair-id ASSET-USD/ASSET-USD")
    candidate = CandidateIdentity(pair=pair, venue="dydx", strategy_family="Copula", timeframe="1d", lookback=320, formula_version="copula-v1")
    policy = load_venue_policy(venue=candidate.venue, root=root)
    result = build_copula_shadow_comparison(candidate=candidate, root=root, venue_policy=policy)
    return StageResult(stage=stage, status=StageStatus.PASSED, reason=f"copula_shadow_{result['comparison']}", evidence_path=f"{result['csv']};{result['markdown']}", next_step="review immutable event; shadow output has no promotion authority", rows=1)
