"""Authority inventory, hostile agent packets, and learning-label controls."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from quant_platform.active_pipeline import CommandResult


ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "thewiz.agent_learning_governance.v1"


AUTHORITY_EDGES = (
    ("orchestrator", "research_stage", "schedule_and_route_research", False, False),
    ("mini_agent_research", "orchestrator", "propose_research_task", False, False),
    ("memory_specialist", "research_memory", "append_evidence_memory", False, False),
    ("test_specialist", "research_reports", "run_declared_tests", False, False),
    ("reference_specialist", "research_reports", "cite_reference_evidence", False, False),
    ("static_spread_teacher", "teacher_council", "shadow_proposal", False, False),
    ("static_zscorer_teacher", "teacher_council", "shadow_proposal", False, False),
    ("dynamic_spread_teacher", "teacher_council", "shadow_proposal", False, False),
    ("dynamic_zscorer_teacher", "teacher_council", "shadow_proposal", False, False),
    ("ou_spread_teacher", "teacher_council", "shadow_proposal", False, False),
    ("ou_zscorer_teacher", "teacher_council", "shadow_proposal", False, False),
    ("copula_teacher", "teacher_council", "shadow_proposal", False, False),
    ("dependency_critic", "teacher_council", "veto_or_warn", False, False),
    ("regime_critic", "teacher_council", "veto_or_warn", False, False),
    ("risk_critic", "teacher_council", "veto_or_warn", False, False),
    ("cost_critic", "teacher_council", "veto_or_warn", False, False),
    ("execution_critic", "teacher_council", "veto_or_warn", False, False),
    ("outcome_critic", "teacher_council", "veto_or_warn", False, False),
    ("teacher_council", "shadow_queue", "shadow_test_or_abstain", False, False),
    ("student_router", "teacher_council", "advisory_mode_probabilities", False, False),
    ("student_outcome_model", "teacher_council", "advisory_after_cost_forecast", False, False),
    ("trade_gate_model", "research_replay", "shadow_gate_or_size", False, False),
    ("rl_idea_scout", "research_hypothesis_queue", "propose_hypothesis", False, False),
    ("contextual_bandit", "research_hypothesis_queue", "rank_shadow_action", False, False),
    ("ppo_policy", "research_simulator", "shadow_policy_action", False, False),
    ("model_quantizer", "model_artifacts", "export_only_after_acceptance", False, False),
    ("deterministic_risk_gate", "testnet_preflight", "veto_only", False, False),
    ("hyperliquid_testnet_executor", "hyperliquid_testnet", "submit_only_with_external_explicit_approval", True, False),
    ("live_canary_executor", "hyperliquid_live", "disabled_until_explicit_live_policy_and_user_authorization", False, False),
)


def build_agent_authority_inventory(*, root: Path = ROOT) -> pd.DataFrame:
    rows = []
    for source, target, action, testnet_submit_capability, live_submit_capability in AUTHORITY_EDGES:
        research_only = not testnet_submit_capability and not live_submit_capability
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "source_component": source,
                "target_component": target,
                "allowed_action": action,
                "research_only": research_only,
                "can_change_acceptance_policy": False,
                "can_override_veto": False,
                "can_read_secrets": source in {"hyperliquid_testnet_executor", "live_canary_executor"},
                "can_submit_testnet_orders": testnet_submit_capability,
                "can_submit_live_orders": live_submit_capability,
                "requires_external_explicit_approval": source in {"hyperliquid_testnet_executor", "live_canary_executor"},
                "current_authority_status": "DISABLED" if source in {"hyperliquid_testnet_executor", "live_canary_executor"} else "RESEARCH_ONLY",
                "live_trading_authorized": False,
            }
        )
    frame = pd.DataFrame(rows)
    path = root / "reports" / "active" / "agent_authority_inventory.csv"
    _atomic_csv(frame, path)
    return frame


def validate_agent_packet(packet: dict[str, Any]) -> list[str]:
    blockers = []
    required = {"candidate_id", "model_version", "feature_schema_version", "confidence", "label_source", "feature_timestamp", "label_timestamp", "evidence_hash", "requested_action"}
    missing = sorted(required - set(packet))
    if missing:
        return ["missing_fields:" + ",".join(missing)]
    confidence = _finite(packet.get("confidence"))
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        blockers.append("confidence_invalid")
    if not str(packet.get("candidate_id", "")).startswith(("candidate_", "hypothesis_", "cwexp_")):
        blockers.append("candidate_identity_invalid")
    if not str(packet.get("model_version", "")).strip():
        blockers.append("model_version_missing")
    if not str(packet.get("feature_schema_version", "")).strip():
        blockers.append("feature_schema_version_missing")
    if str(packet.get("label_source")) not in {"backtest_label", "paper_label", "live_label"}:
        blockers.append("label_source_invalid")
    feature_time = pd.to_datetime(packet.get("feature_timestamp"), utc=True, errors="coerce")
    label_time = pd.to_datetime(packet.get("label_timestamp"), utc=True, errors="coerce")
    if pd.isna(feature_time) or pd.isna(label_time) or feature_time >= label_time:
        blockers.append("feature_label_time_contract_failed")
    if len(str(packet.get("evidence_hash", ""))) != 64:
        blockers.append("evidence_hash_invalid")
    if str(packet.get("requested_action")) not in {"research", "shadow", "abstain"}:
        blockers.append("agent_execution_action_forbidden")
    return blockers


def run_agent_authority_fuzz_tests(*, root: Path = ROOT) -> pd.DataFrame:
    base = {
        "candidate_id": "candidate_1234567890abcdef1234",
        "model_version": "model-v1",
        "feature_schema_version": "features-v1",
        "confidence": 0.6,
        "label_source": "backtest_label",
        "feature_timestamp": "2026-08-01T00:00:00+00:00",
        "label_timestamp": "2026-08-02T00:00:00+00:00",
        "evidence_hash": "a" * 64,
        "requested_action": "research",
    }
    cases = {
        "valid_research_packet": base,
        "forged_confidence": {**base, "confidence": 99},
        "forged_candidate_id": {**base, "candidate_id": "trust-me"},
        "missing_model_version": {**base, "model_version": ""},
        "forged_label": {**base, "label_source": "wizard_dashboard_label"},
        "future_feature": {**base, "feature_timestamp": "2026-08-03T00:00:00+00:00"},
        "forged_evidence_hash": {**base, "evidence_hash": "abc"},
        "forged_council_vote": {**base, "requested_action": "submit_live_order"},
    }
    rows = []
    for case, packet in cases.items():
        blockers = validate_agent_packet(packet)
        expected_pass = case == "valid_research_packet"
        actual_pass = not blockers
        rows.append(
            {
                "case": case,
                "expected_status": "PASS" if expected_pass else "BLOCKED",
                "actual_status": "PASS" if actual_pass else "BLOCKED",
                "blocker": ";".join(blockers),
                "promotion_authority": False,
                "testnet_order_authority": False,
                "live_trading_authorized": False,
                "status": "PASS" if expected_pass == actual_pass else "FAIL",
            }
        )
    frame = pd.DataFrame(rows)
    path = root / "reports" / "red_team" / "agent_authority_fuzz_results.csv"
    _atomic_csv(frame, path)
    if not frame["status"].eq("PASS").all():
        raise ValueError("an adversarial agent packet escaped authority validation")
    return frame


def build_learning_label_contract(*, root: Path = ROOT) -> dict[str, Any]:
    rows = [
        {
            "label_source": "backtest_label",
            "feature_requirement": "feature_timestamp_strictly_before_label_timestamp",
            "eligible_training_use": "backtest_trained_research_model",
            "realized_execution_evidence": False,
            "may_claim_live_validation": False,
            "may_authorize_orders": False,
        },
        {
            "label_source": "paper_label",
            "feature_requirement": "prospective_testnet_feature_snapshot_before_order_intent",
            "eligible_training_use": "testnet_realized_research_after_sample_sufficiency",
            "realized_execution_evidence": True,
            "may_claim_live_validation": False,
            "may_authorize_orders": False,
        },
        {
            "label_source": "live_label",
            "feature_requirement": "prospective_live_feature_snapshot_before_authorized_canary",
            "eligible_training_use": "live_validation_only_after_governance_review",
            "realized_execution_evidence": True,
            "may_claim_live_validation": True,
            "may_authorize_orders": False,
        },
    ]
    frame = pd.DataFrame(rows)
    frame["schema_version"] = SCHEMA_VERSION
    frame["dashboard_hindsight_allowed"] = False
    frame["live_trading_authorized"] = False
    path = root / "reports" / "active" / "learning_label_contract.csv"
    _atomic_csv(frame, path)
    leakage_path = root / "reports" / "ml" / "leakage_audit.csv"
    leakage = _read_csv(leakage_path)
    future = int(leakage.get("uses_future_data", pd.Series(False, index=leakage.index)).map(_truthy).sum())
    hindsight = int(leakage.get("uses_dashboard_hindsight", pd.Series(False, index=leakage.index)).map(_truthy).sum())
    invalid_time = 0
    if not leakage.empty and {"feature_timestamp", "label_timestamp"}.issubset(leakage.columns):
        feature = pd.to_datetime(leakage["feature_timestamp"], utc=True, errors="coerce")
        label = pd.to_datetime(leakage["label_timestamp"], utc=True, errors="coerce")
        invalid_time = int((feature.isna() | label.isna() | (feature >= label)).sum())
    audit = {
        "schema_version": SCHEMA_VERSION,
        "rows_audited": len(leakage),
        "future_feature_rows": future,
        "dashboard_hindsight_rows": hindsight,
        "invalid_feature_label_time_rows": invalid_time,
        "status": "PASS" if not any((future, hindsight, invalid_time)) and not leakage.empty else "BLOCKED",
        "training_authority": "backtest_research_only",
        "testnet_order_authority": False,
        "live_trading_authorized": False,
        "evidence_path": _relative(leakage_path, root),
    }
    audit_path = root / "reports" / "active" / "learning_label_audit.json"
    _atomic_json(audit, audit_path)
    return {"contract": path, "audit": audit_path, "summary": audit}


def build_model_authority_status(*, root: Path = ROOT, now: datetime | None = None) -> dict[str, Any]:
    metrics_path = root / "models" / "trade_gate" / "metrics.json"
    acceptance_path = root / "reports" / "ml" / "model_gated_acceptance.csv"
    metrics = _read_json(metrics_path)
    acceptance = _read_csv(acceptance_path)
    accepted = bool(metrics.get("accepted", False)) and bool(not acceptance.empty and acceptance.get("accepted", pd.Series(False)).map(_truthy).all())
    take_rate = _finite(metrics.get("median_take_rate"))
    monotonic = bool(metrics.get("score_buckets_monotonic", False))
    blockers = []
    if not accepted:
        blockers.append("model_incremental_edge_not_accepted")
    if not math.isfinite(take_rate) or take_rate < 0.10:
        blockers.append("model_take_rate_below_minimum")
    if not monotonic:
        blockers.append("score_buckets_not_monotonic")
    blockers.append("realized_testnet_sample_not_available")
    status = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": _as_utc(now).isoformat(),
        "model_version": str(metrics.get("best_model", "")),
        "label_source": str(metrics.get("label_source", "backtest_trained")),
        "out_of_sample_incremental_edge_accepted": accepted,
        "median_take_rate": take_rate if math.isfinite(take_rate) else None,
        "minimum_take_rate": 0.10,
        "score_buckets_monotonic": monotonic,
        "realized_testnet_sample_sufficient": False,
        "model_authority": "RESEARCH_ONLY",
        "quantization_authorized": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
        "blockers": blockers,
        "evidence_path": f"{_relative(metrics_path, root)};{_relative(acceptance_path, root)}",
    }
    path = root / "reports" / "active" / "model_authority_status.json"
    _atomic_json(status, path)
    return {"path": path, "summary": status}


def build_corrective_agent_governance(*, root: Path = ROOT, now: datetime | None = None) -> CommandResult:
    inventory = build_agent_authority_inventory(root=root)
    fuzz = run_agent_authority_fuzz_tests(root=root)
    labels = build_learning_label_contract(root=root)
    model = build_model_authority_status(root=root, now=now)
    return CommandResult(
        paths={
            "authority_inventory": root / "reports" / "active" / "agent_authority_inventory.csv",
            "authority_fuzz": root / "reports" / "red_team" / "agent_authority_fuzz_results.csv",
            "label_contract": Path(labels["contract"]),
            "label_audit": Path(labels["audit"]),
            "model_authority": Path(model["path"]),
        },
        summary={
            "status": "PASS" if labels["summary"]["status"] == "PASS" and fuzz["status"].eq("PASS").all() else "BLOCKED",
            "authority_edges": len(inventory),
            "fuzz_cases": len(fuzz),
            "label_rows_audited": labels["summary"]["rows_audited"],
            "model_authority": model["summary"]["model_authority"],
            "model_blockers": model["summary"]["blockers"],
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        },
    )


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


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _finite(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def _truthy(value: Any) -> bool:
    return value is True or str(value).strip().lower() in {"1", "true", "yes", "pass", "ready"}


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _as_utc(value: datetime | None) -> datetime:
    value = value or datetime.now(timezone.utc)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


if __name__ == "__main__":
    result = build_corrective_agent_governance()
    print(json.dumps({"summary": result.summary, "paths": {key: str(value) for key, value in result.paths.items()}}, indent=2))
