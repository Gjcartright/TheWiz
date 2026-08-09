"""Deterministic downstream validation chain for frozen Wizard evidence."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

from quant_platform.active_pipeline import CommandResult
from quant_platform.orchestration.exhaustive_wizard_hyperliquid_concentration import (
    build_exhaustive_wizard_hyperliquid_concentration,
)
from quant_platform.orchestration.exhaustive_wizard_hyperliquid_cost_bridge import (
    build_exhaustive_wizard_hyperliquid_cost_evidence,
)
from quant_platform.orchestration.exhaustive_wizard_hyperliquid_leverage import (
    build_exhaustive_wizard_hyperliquid_leverage_surface,
)
from quant_platform.orchestration.exhaustive_wizard_hyperliquid_learning import (
    build_exhaustive_wizard_hyperliquid_learning_ledger,
)
from quant_platform.orchestration.exhaustive_wizard_hyperliquid_observed_cost_replay import (
    run_exhaustive_wizard_hyperliquid_observed_cost_replay,
)
from quant_platform.orchestration.exhaustive_wizard_hyperliquid_regimes import (
    build_exhaustive_wizard_hyperliquid_regime_attribution,
)
from quant_platform.orchestration.exhaustive_wizard_hyperliquid_robustness import (
    run_exhaustive_wizard_hyperliquid_robustness,
)
from quant_platform.orchestration.exhaustive_wizard_hyperliquid_walkforward import (
    run_exhaustive_wizard_hyperliquid_walkforward,
)


ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "exhaustive_wizard_hyperliquid_validation.v1"


def run_exhaustive_wizard_hyperliquid_validation(
    *,
    root: Path = ROOT,
    now: datetime | None = None,
) -> CommandResult:
    """Rebuild the ordered research chain without capture or order submission."""

    as_of = _as_utc(now or datetime.now(timezone.utc))
    stages = [
        ("cost_evidence", build_exhaustive_wizard_hyperliquid_cost_evidence),
        ("observed_cost_replay", run_exhaustive_wizard_hyperliquid_observed_cost_replay),
        ("walkforward", run_exhaustive_wizard_hyperliquid_walkforward),
        ("regime_attribution", build_exhaustive_wizard_hyperliquid_regime_attribution),
        ("robustness", run_exhaustive_wizard_hyperliquid_robustness),
        ("concentration", build_exhaustive_wizard_hyperliquid_concentration),
        ("leverage_surface", build_exhaustive_wizard_hyperliquid_leverage_surface),
    ]
    stage_rows: list[dict[str, object]] = []
    run_id = ""
    experiment_count: int | None = None
    for stage_name, stage in stages:
        result = stage(root=root, now=as_of)
        summary = result.summary
        current_run_id = str(summary.get("run_id") or "").strip()
        if not current_run_id:
            raise ValueError(f"{stage_name} did not report run_id")
        if run_id and current_run_id != run_id:
            raise ValueError(f"{stage_name} run identity does not match prior stages")
        run_id = current_run_id
        current_experiments = int(summary.get("experiments") or 0)
        if experiment_count is None:
            experiment_count = current_experiments
        if current_experiments != experiment_count:
            raise ValueError(f"{stage_name} experiment accounting changed")
        if not bool(summary.get("experiment_status_accounted", False)):
            raise ValueError(f"{stage_name} did not prove complete experiment accounting")
        if bool(summary.get("live_trading_authorized", False)):
            raise ValueError(f"{stage_name} improperly authorized live trading")
        stage_rows.append(
            {
                "stage": stage_name,
                "run_id": current_run_id,
                "experiments": current_experiments,
                "experiment_status_accounted": True,
                "stage_identity": _stage_identity(summary),
                "acceptance_eligible_replays": int(
                    summary.get("acceptance_eligible_replays") or 0
                ),
                "live_trading_authorized": False,
                "manifest_path": _manifest_path(result, root),
            }
        )

    material = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "as_of": as_of.isoformat(),
        "stages": stage_rows,
        "frozen_evidence_only": True,
        "authenticated_wizard_capture_automated": False,
        "order_submission_performed": False,
        "live_trading_authorized": False,
    }
    digest = sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    validation_id = f"hlvalidation_{as_of.strftime('%Y%m%dT%H%M%S%fZ')}_{digest[:8]}"
    learning_result = build_exhaustive_wizard_hyperliquid_learning_ledger(
        root=root,
        now=as_of,
        validation_id=validation_id,
        validation_stages=stage_rows,
    )
    learning = learning_result.summary
    if str(learning.get("run_id") or "") != run_id:
        raise ValueError("Learning ledger run identity does not match validation")
    if int(learning.get("records") or 0) != (experiment_count or 0):
        raise ValueError("Learning ledger experiment accounting changed")
    if not bool(learning.get("experiment_status_accounted", False)):
        raise ValueError("Learning ledger did not prove complete experiment accounting")
    if int(learning.get("training_eligible_records") or 0) != 0:
        raise ValueError("Experiment summaries must not become training-eligible rows")
    if int(learning.get("paper_label_records") or 0) != 0:
        raise ValueError("Research validation improperly created paper labels")
    if int(learning.get("live_label_records") or 0) != 0:
        raise ValueError("Research validation improperly created live labels")
    if bool(learning.get("live_trading_authorized", False)):
        raise ValueError("Learning ledger improperly authorized live trading")
    active = root / "reports" / "active"
    snapshot = (
        root
        / "reports"
        / "snapshots"
        / "exhaustive_wizard_hyperliquid"
        / run_id
        / "validations"
        / validation_id
    )
    active.mkdir(parents=True, exist_ok=True)
    snapshot.mkdir(parents=True, exist_ok=True)
    paths = {
        "manifest": active / "exhaustive_wizard_hyperliquid_validation_manifest.json",
        "summary_md": active / "exhaustive_wizard_hyperliquid_validation_summary.md",
        "snapshot_manifest": snapshot / "manifest.json",
        "snapshot_summary_md": snapshot / "summary.md",
        **{
            f"learning_{name}": path
            for name, path in learning_result.paths.items()
        },
    }
    summary = {
        **material,
        "validation_id": validation_id,
        "experiments": experiment_count or 0,
        "stages_complete": len(stage_rows),
        "expected_stages": len(stages),
        "stage_accounting_complete": len(stage_rows) == len(stages),
        "learning_ledger": {
            "learning_ledger_id": learning["learning_ledger_id"],
            "records": learning["records"],
            "unique_experiment_ids": learning["unique_experiment_ids"],
            "experiment_status_accounted": learning["experiment_status_accounted"],
            "outcome_type": learning["outcome_type"],
            "record_granularity": learning["record_granularity"],
            "training_eligible_records": learning["training_eligible_records"],
            "paper_label_records": learning["paper_label_records"],
            "live_label_records": learning["live_label_records"],
            "snapshot_manifest": _manifest_path(learning_result, root),
            "live_trading_authorized": False,
        },
        "artifacts": {name: _relative(path, root) for name, path in paths.items()},
    }
    manifest = json.dumps(summary, indent=2, sort_keys=True)
    markdown = _summary_markdown(summary)
    for path in (paths["manifest"], paths["snapshot_manifest"]):
        path.write_text(manifest, encoding="utf-8")
    for path in (paths["summary_md"], paths["snapshot_summary_md"]):
        path.write_text(markdown, encoding="utf-8")
    return CommandResult(paths=paths, summary=summary)


def _stage_identity(summary: dict[str, object]) -> str:
    for key in (
        "leverage_surface_id",
        "concentration_id",
        "robustness_id",
        "regime_attribution_id",
        "walkforward_id",
        "observed_cost_replay_id",
        "cost_evidence_id",
    ):
        value = str(summary.get(key) or "").strip()
        if value:
            return value
    return ""


def _manifest_path(result: CommandResult, root: Path) -> str:
    path = result.paths.get("snapshot_manifest") or result.paths.get("manifest")
    return _relative(path, root) if path is not None else ""


def _summary_markdown(summary: dict[str, object]) -> str:
    lines = [
        "# Exhaustive Wizard To Hyperliquid Validation",
        "",
        "This command rebuilds the deterministic downstream research chain from frozen evidence. It does not capture the authenticated Wizard dashboard and cannot submit an order.",
        "",
        f"- validation: {summary['validation_id']}",
        f"- exhaustive run: {summary['run_id']}",
        f"- experiments accounted: {summary['experiments']}",
        f"- stages complete: {summary['stages_complete']} of {summary['expected_stages']}",
        "- order submission performed: false",
        "- live trading authorized: false",
        "",
        "## Stages",
        "",
    ]
    for row in summary["stages"]:
        lines.append(
            f"- {row['stage']}: {row['stage_identity']} "
            f"({row['experiments']} experiments accounted)"
        )
    learning = summary["learning_ledger"]
    lines.extend(
        [
            "",
            "## Learning Ledger",
            "",
            f"- ledger: {learning['learning_ledger_id']}",
            f"- records: {learning['records']}",
            f"- training-eligible records: {learning['training_eligible_records']}",
            f"- paper labels: {learning['paper_label_records']}",
            f"- live labels: {learning['live_label_records']}",
            "- live trading authorized: false",
        ]
    )
    return "\n".join(lines) + "\n"


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
