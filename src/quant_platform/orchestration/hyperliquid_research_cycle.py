"""Deterministic Crypto Wizards to Hyperliquid research remediation cycle."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import pandas as pd

from quant_platform.hyperliquid import (
    DEFAULT_SLIPPAGE_CALIBRATION_CADENCE_MINUTES,
    DEFAULT_SLIPPAGE_CALIBRATION_MIN_SAMPLES,
    DEFAULT_SLIPPAGE_CALIBRATION_WINDOW_HOURS,
    build_hyperliquid_evidence_cadence,
    build_hyperliquid_pair_cost_model,
    refresh_hyperliquid_execution_cost_snapshot,
)
from quant_platform.hyperliquid_testnet import (
    write_hyperliquid_testnet_margin_snapshot,
    write_hyperliquid_testnet_preflight_report,
)
from quant_platform.math_v2_acceptance import build_math_v2_acceptance
from quant_platform.orchestration.hyperliquid_learning_and_risk import (
    build_portfolio_critic,
    build_testnet_lifecycle_gate,
    materialize_council_learning_dataset,
    write_testnet_smoke_approval_template,
)
from quant_platform.orchestration.hyperliquid_research_validation import (
    build_hyperliquid_auxiliary_timeframe_validation,
    build_hyperliquid_research_family_controls,
    build_hyperliquid_walkforward_validation,
)
from quant_platform.orchestration.hyperliquid_run_manifest import (
    build_hyperliquid_authority_state,
    build_hyperliquid_run_manifest,
)
from quant_platform.orchestration.teacher_adapters import build_teacher_evidence_adapters
from quant_platform.orchestration.teacher_control_plane import build_teacher_council_control_plane
from quant_platform.orchestration.teacher_evidence_materializer import materialize_teacher_evidence


ROOT = Path(__file__).resolve().parents[3]
CYCLE_VERSION = "hyperliquid-research-cycle-v1"


def run_hyperliquid_research_cycle(
    *,
    root: Path = ROOT,
    collect_l2: bool = False,
    now: datetime | None = None,
) -> dict[str, object]:
    """Run one identity-bound cycle. This never submits an order."""

    now = _as_utc(now or datetime.now(timezone.utc))
    steps: list[dict[str, object]] = []

    math_result = build_math_v2_acceptance(root=root)
    _record(steps, "math_v2", math_result["status"], math_result.get("marker"), "")

    preliminary = build_hyperliquid_run_manifest(root=root, now=now)
    candidates_path = Path(preliminary["candidates"])
    _record(steps, "freeze_candidate_set", preliminary["status"], preliminary["manifest"], "" if preliminary["lineage_ready"] else "manifest_lineage_blocked")

    if collect_l2 and int(preliminary.get("candidate_count", 0)) > 0:
        l2 = refresh_hyperliquid_execution_cost_snapshot(
            root=root,
            max_pairs=int(preliminary["candidate_count"]),
            captured_at=now,
            candidate_path=candidates_path,
            min_samples=DEFAULT_SLIPPAGE_CALIBRATION_MIN_SAMPLES,
            window_hours=DEFAULT_SLIPPAGE_CALIBRATION_WINDOW_HOURS,
        )
        _record(steps, "l2_snapshot", "MATERIALIZED", l2.paths.get("hyperliquid_l2_slippage_samples"), "")
    cost_result = build_hyperliquid_pair_cost_model(
        root=root,
        max_pairs=max(int(preliminary.get("candidate_count", 0)), 1),
        min_samples=DEFAULT_SLIPPAGE_CALIBRATION_MIN_SAMPLES,
        window_hours=DEFAULT_SLIPPAGE_CALIBRATION_WINDOW_HOURS,
        as_of=now,
        candidate_path=candidates_path,
    )
    _synchronize_cost_evidence(root=root, run_id=str(preliminary["run_id"]), candidate_set_id=str(preliminary["candidate_set_id"]))
    _record(
        steps,
        "canonical_pair_cost_model",
        "READY" if int(cost_result.summary.get("slippage_models_ready", 0)) == int(preliminary.get("candidate_count", 0)) else "BLOCKED",
        cost_result.paths.get("hyperliquid_pair_cost_model"),
        "" if int(cost_result.summary.get("slippage_models_ready", 0)) == int(preliminary.get("candidate_count", 0)) else "slippage_calibration_incomplete",
    )
    cadence = build_hyperliquid_evidence_cadence(
        root=root,
        target_samples=DEFAULT_SLIPPAGE_CALIBRATION_MIN_SAMPLES,
        cadence_minutes=DEFAULT_SLIPPAGE_CALIBRATION_CADENCE_MINUTES,
        window_hours=DEFAULT_SLIPPAGE_CALIBRATION_WINDOW_HOURS,
        as_of=now,
    )
    _record(steps, "l2_evidence_cadence", "MATERIALIZED", cadence.paths.get("hyperliquid_evidence_cadence"), "")

    manifest = build_hyperliquid_run_manifest(root=root, now=now)
    _record(steps, "validated_run_manifest", manifest["status"], manifest["validation"], "" if manifest["lineage_ready"] else "manifest_lineage_blocked")

    walkforward = build_hyperliquid_walkforward_validation(root=root)
    _record(
        steps,
        "walkforward_and_selection_controls",
        walkforward["status"],
        walkforward["markdown"],
        "" if int(walkforward["selection_passes"]) > 0 else "no_mode_passes_selection_controls",
    )
    auxiliary = build_hyperliquid_auxiliary_timeframe_validation(root=root, refresh=False)
    _record(
        steps,
        "auxiliary_4h_walkforward",
        auxiliary.get("status", "BLOCKED"),
        auxiliary.get("markdown", auxiliary.get("bundle", "")),
        "" if int(auxiliary.get("trade_rows", 0)) > 0 else str(auxiliary.get("blocker", "auxiliary_timeframe_trade_evidence_missing")),
    )
    family_controls = build_hyperliquid_research_family_controls(root=root)
    _record(
        steps,
        "research_family_selection_controls",
        family_controls.get("status", "BLOCKED"),
        family_controls.get("markdown", family_controls.get("selection", "")),
        ""
        if bool(family_controls.get("family_complete", False))
        else "daily_and_auxiliary_research_family_incomplete",
    )

    materialized = materialize_teacher_evidence(root=root, now=now)
    _record(steps, "teacher_evidence", materialized["status"], materialized["summary"], "" if int(materialized["eligible_teachers"]) > 0 else "no_blocker_free_teacher_rows")
    adapters = build_teacher_evidence_adapters(root=root)
    _record(steps, "teacher_adapters", adapters["status"], adapters["readiness"], "" if adapters["status"] == "READY_FOR_COUNCIL" else "teacher_adapter_blocked")
    council = build_teacher_council_control_plane(root=root)
    _record(steps, "teacher_council", council["status"], council["decisions"], "" if council["status"] == "READY_FOR_SHADOW_ONLY" else "teacher_council_blocked")

    learning = materialize_council_learning_dataset(root=root)
    learning_blocker = "" if learning["supervised_status"] != "BLOCKED" else "student_supervised_readiness_blocked"
    _record(steps, "council_learning_dataset", learning["status"], learning["dataset"], learning_blocker)
    # Refresh council readiness now that the explicit research-only dataset exists.
    council = build_teacher_council_control_plane(root=root)
    margin = write_hyperliquid_testnet_margin_snapshot(root=root, now=now)
    margin_ready = bool(not margin.empty and margin.get("status", pd.Series(["BLOCKED"])).astype(str).eq("READY").iloc[0])
    _record(
        steps,
        "hyperliquid_testnet_margin_snapshot",
        "PASS" if margin_ready else "BLOCKED",
        root / "reports" / "active" / "hyperliquid_testnet_margin_snapshot.csv",
        "" if margin_ready else str(margin.iloc[0].get("blockers", "testnet_margin_snapshot_blocked")),
    )
    portfolio = build_portfolio_critic(root=root)
    _record(steps, "portfolio_critic", portfolio["status"], portfolio["critic"], "" if portfolio["status"] == "PASS" else str(portfolio["blocker_codes"]))

    preflight = write_hyperliquid_testnet_preflight_report(root=root)
    preflight_ready = bool(not preflight.empty and preflight.get("ready_for_no_order_preflight", pd.Series([False])).astype(bool).iloc[0])
    _record(steps, "hyperliquid_no_order_preflight", "PASS" if preflight_ready else "BLOCKED", root / "reports" / "active" / "hyperliquid_testnet_preflight.csv", "" if preflight_ready else "testnet_preflight_blocked")
    approval = write_testnet_smoke_approval_template(root=root)
    _record(steps, "testnet_smoke_approval_template", approval["status"], approval["approval"], "explicit_one_run_user_approval_required")
    lifecycle = build_testnet_lifecycle_gate(root=root)
    _record(steps, "testnet_lifecycle_gate", lifecycle["status"], lifecycle["gate"], "" if lifecycle["status"] == "PASS" else "approved_two_leg_smoke_receipt_missing")

    authority = build_hyperliquid_authority_state(root=root, now=now)
    _record(steps, "layered_authority", authority["status"], authority["authority"], str(authority["blocker"]))
    return _write_cycle_receipt(root=root, now=now, steps=steps, manifest=manifest, authority=authority)


def _synchronize_cost_evidence(*, root: Path, run_id: str, candidate_set_id: str) -> None:
    active = root / "reports" / "active"
    source = active / "hyperliquid_pair_cost_model.csv"
    frame = _read_csv(source)
    if not frame.empty:
        frame.insert(0, "candidate_set_id", candidate_set_id)
        frame.insert(0, "run_id", run_id)
    _atomic_csv(frame, source)
    _atomic_csv(frame, active / "workflow_hyperliquid_pair_cost_model.csv")


def _write_cycle_receipt(
    *,
    root: Path,
    now: datetime,
    steps: list[dict[str, object]],
    manifest: dict[str, object],
    authority: dict[str, object],
) -> dict[str, object]:
    output = root / "reports" / "orchestration" / "hyperliquid_run"
    output.mkdir(parents=True, exist_ok=True)
    step_frame = pd.DataFrame(steps, columns=["step", "status", "blocker", "evidence_path", "evidence_hash"])
    receipt_path = output / "research_cycle_receipt.csv"
    markdown_path = output / "research_cycle_receipt.md"
    json_path = output / "research_cycle_receipt.json"
    _atomic_csv(step_frame, receipt_path)
    payload: dict[str, Any] = {
        "cycle_version": CYCLE_VERSION,
        "generated_at": now.isoformat(),
        "run_id": manifest.get("run_id", ""),
        "candidate_set_id": manifest.get("candidate_set_id", ""),
        "execution_truth_mode": "hyperliquid_testnet",
        "wizard_authority": "discovery_only",
        "status": authority.get("status", "BLOCKED"),
        "execution_allowed": bool(authority.get("execution_allowed", False)),
        "blockers": authority.get("blocker", ""),
        "step_count": len(step_frame),
        "blocked_steps": int(step_frame["blocker"].astype(str).str.strip().ne("").sum()),
        "receipt_hash": sha256(step_frame.to_csv(index=False).encode("utf-8")).hexdigest(),
    }
    _atomic_json(payload, json_path)
    markdown_path.write_text(
        "\n".join(
            [
                "# Hyperliquid Research Cycle",
                "",
                f"- Run: `{payload['run_id']}`",
                f"- Candidate set: `{payload['candidate_set_id']}`",
                f"- Status: **{payload['status']}**",
                f"- Execution allowed: **{payload['execution_allowed']}**",
                "- Orders submitted by this cycle: **none**",
                "",
                step_frame.to_markdown(index=False),
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {**payload, "receipt": receipt_path, "receipt_json": json_path, "summary": markdown_path}


def _record(steps: list[dict[str, object]], step: str, status: object, evidence: object, blocker: str) -> None:
    path = Path(str(evidence)) if evidence not in (None, "") else None
    steps.append(
        {
            "step": step,
            "status": str(status),
            "blocker": blocker,
            "evidence_path": str(path) if path is not None else "",
            "evidence_hash": _file_hash(path),
        }
    )


def _file_hash(path: Path | None) -> str:
    if path is None or not path.exists() or not path.is_file():
        return ""
    return sha256(path.read_bytes()).hexdigest()


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame()


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)
