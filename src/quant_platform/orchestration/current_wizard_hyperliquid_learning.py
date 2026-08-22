"""Dated, research-only learning ledger for the current Wizard board."""

from __future__ import annotations

from quant_platform.orchestration.corrective_runtime import atomic_write_text

from quant_platform.orchestration.corrective_runtime import atomic_write_csv

from quant_platform.orchestration.corrective_runtime import atomic_write_bytes

from datetime import datetime, timezone
import gzip
from hashlib import sha256
import json
import math
from pathlib import Path

import pandas as pd

from quant_platform.active_pipeline import CommandResult
from quant_platform.orchestration.snapshot_lineage import (
    unique_file_bytes,
    verified_snapshot_reference,
)


ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "current_wizard_hyperliquid_learning.v1"

INPUT_FILENAMES = {
    "matrix": "current_wizard_hyperliquid_experiment_matrix.csv",
    "failure_attribution": "current_wizard_hyperliquid_failure_attribution.csv",
    "leverage_status": "current_wizard_hyperliquid_leverage_status.csv",
    "concentration_manifest": "current_wizard_hyperliquid_concentration_manifest.json",
    "failure_manifest": "current_wizard_hyperliquid_failure_attribution_manifest.json",
    "leverage_manifest": "current_wizard_hyperliquid_leverage_manifest.json",
    "handoff_manifest": "current_wizard_hyperliquid_handoff_manifest.json",
}

BASE_COLUMNS = [
    "experiment_id",
    "pair_group_key",
    "pair",
    "wizard_exchange",
    "timeframe",
    "asset_a",
    "asset_b",
    "exact_mode",
    "orientation",
    "mode_observed_in_current_scanner",
    "local_mode_implemented",
    "experiment_status",
    "experiment_blocker",
    "canonical_replay_leverage",
    "vendor_pair_detail_status",
    "vendor_parity_claimed",
    "discovery_prefilter_applied",
]

FAILURE_COLUMNS = [
    "canonical_replay_status",
    "cost_evidence_status",
    "observed_cost_replay_status",
    "walkforward_status",
    "regime_status",
    "regime_stability_status",
    "robustness_status",
    "research_robustness_status",
    "statistical_selection_status",
    "concentration_status",
    "concentration_gate_pass",
    "strict_cost_calibration_ready",
    "mode_fidelity_status",
    "vendor_exact_mode_parity_proven",
    "one_x_research_survivor",
    "leverage_research_eligible",
    "execution_acceptance_ready",
    "testnet_preflight_eligible",
    "consolidated_status",
    "first_blocking_stage",
    "first_blocker",
    "all_independent_blockers",
    "next_action",
    "research_progress_rank",
    "observed_trades",
    "observed_profit_factor",
    "observed_sharpe",
    "observed_max_drawdown",
    "observed_total_return",
    "walkforward_trades",
    "walkforward_profit_factor",
    "walkforward_sharpe",
    "walkforward_max_drawdown",
    "walkforward_total_return",
    "walkforward_bh_qvalue",
    "hedge_ratio_cv",
    "regime_profit_concentration",
    "worst_regime_expectancy",
    "robustness_parameter_pass_ratio",
    "worst_robustness_drawdown",
]

LEVERAGE_COLUMNS = [
    "prior_one_x_research_status",
    "leverage_surface_status",
    "leverage_surface_blocker",
    "maximum_research_leverage",
    "ready_for_testnet_1x_lifecycle",
]


def build_current_wizard_hyperliquid_learning_ledger(
    *,
    root: Path = ROOT,
    now: datetime | None = None,
) -> CommandResult:
    """Write one non-training research outcome for every experiment cell."""

    as_of = _as_utc(now or datetime.now(timezone.utc))
    active = root / "reports" / "active"
    input_paths = {name: active / filename for name, filename in INPUT_FILENAMES.items()}
    missing = [str(path) for path in input_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Current learning-ledger inputs missing: " + ",".join(missing)
        )

    matrix = _read_csv(input_paths["matrix"])
    failure = _read_csv(input_paths["failure_attribution"])
    leverage = _read_csv(input_paths["leverage_status"])
    if matrix.empty:
        raise ValueError("Current learning ledger requires a non-empty experiment matrix")
    _require_columns(matrix, ["handoff_id", "refresh_id", *BASE_COLUMNS], "matrix")
    _require_columns(
        failure,
        ["failure_attribution_id", "experiment_id", *FAILURE_COLUMNS],
        "failure_attribution",
    )
    _require_columns(
        leverage,
        ["leverage_surface_id", "experiment_id", *LEVERAGE_COLUMNS],
        "leverage_status",
    )
    _require_all_cell_accounting(matrix, failure, leverage)

    concentration_manifest = _read_json(input_paths["concentration_manifest"])
    failure_manifest = _read_json(input_paths["failure_manifest"])
    leverage_manifest = _read_json(input_paths["leverage_manifest"])
    handoff_manifest = _read_json(input_paths["handoff_manifest"])
    handoff_id = _single_value(matrix, "handoff_id", "matrix")
    refresh_id = _single_value(matrix, "refresh_id", "matrix")
    failure_id = _single_value(
        failure, "failure_attribution_id", "failure_attribution"
    )
    leverage_id = _single_value(leverage, "leverage_surface_id", "leverage_status")
    concentration_id = _text(concentration_manifest.get("concentration_id"))
    if handoff_id != _text(handoff_manifest.get("handoff_id")):
        raise ValueError("Current learning handoff identity mismatch")
    if failure_id != _text(failure_manifest.get("failure_attribution_id")):
        raise ValueError("Current learning failure-attribution identity mismatch")
    if leverage_id != _text(leverage_manifest.get("leverage_surface_id")):
        raise ValueError("Current learning leverage identity mismatch")
    if failure_id != _text(leverage_manifest.get("failure_attribution_id")):
        raise ValueError("Current learning leverage does not descend from failure attribution")
    failure_inputs = failure_manifest.get("input_hashes", {})
    expected_concentration_hash = _text(failure_inputs.get("concentration_manifest"))
    if not concentration_id or expected_concentration_hash != _file_hash(
        input_paths["concentration_manifest"]
    ):
        raise ValueError("Current learning concentration lineage mismatch")

    material = {
        "schema_version": SCHEMA_VERSION,
        "as_of": as_of.isoformat(),
        "refresh_id": refresh_id,
        "handoff_id": handoff_id,
        "concentration_id": concentration_id,
        "failure_attribution_id": failure_id,
        "leverage_surface_id": leverage_id,
        "input_hashes": {name: _file_hash(path) for name, path in input_paths.items()},
    }
    learning_id = "cwlearning_" + sha256(
        _canonical_json(material).encode("utf-8")
    ).hexdigest()[:20]
    leverage_snapshot = root / _text(
        leverage_manifest.get("artifacts", {}).get("snapshot_manifest")
    )
    if not leverage_snapshot.exists():
        raise FileNotFoundError("Current learning leverage snapshot manifest is missing")
    snapshot_dir = leverage_snapshot.parent / "learning" / learning_id
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_specs = {
        "matrix": (handoff_manifest, "experiments"),
        "failure_attribution": (failure_manifest, "attribution"),
        "leverage_status": (leverage_manifest, "status"),
        "concentration_manifest": (concentration_manifest, "manifest"),
        "failure_manifest": (failure_manifest, "manifest"),
        "leverage_manifest": (leverage_manifest, "manifest"),
        "handoff_manifest": (handoff_manifest, "manifest"),
    }
    snapshot_references = {
        name: verified_snapshot_reference(
            root=root,
            active_path=source,
            upstream_manifest=snapshot_specs[name][0],
            artifact_key=snapshot_specs[name][1],
        )
        for name, source in input_paths.items()
    }

    ledger = matrix.loc[:, BASE_COLUMNS].copy().rename(
        columns={
            "timeframe": "wizard_timeframe",
            "asset_a": "asset_x",
            "asset_b": "asset_y",
        }
    )
    ledger = ledger.merge(
        failure.loc[:, ["experiment_id", *FAILURE_COLUMNS]],
        on="experiment_id",
        how="left",
        validate="one_to_one",
    )
    ledger = ledger.merge(
        leverage.loc[:, ["experiment_id", *LEVERAGE_COLUMNS]],
        on="experiment_id",
        how="left",
        validate="one_to_one",
    )
    ledger.insert(0, "schema_version", SCHEMA_VERSION)
    ledger.insert(1, "learning_ledger_id", learning_id)
    ledger.insert(2, "refresh_id", refresh_id)
    ledger.insert(3, "handoff_id", handoff_id)
    ledger.insert(4, "concentration_id", concentration_id)
    ledger.insert(5, "failure_attribution_id", failure_id)
    ledger.insert(6, "leverage_surface_id", leverage_id)
    ledger.insert(
        7,
        "learning_record_id",
        ledger["experiment_id"].map(
            lambda value: "cwlearnrec_"
            + sha256(f"{learning_id}|{value}".encode("utf-8")).hexdigest()[:20]
        ),
    )
    ledger.insert(8, "outcome_known_at", as_of.isoformat())
    ledger["outcome_type"] = "backtest_research_outcome"
    ledger["record_granularity"] = "experiment_summary"
    ledger["research_outcome_label"] = ledger.apply(_outcome_label, axis=1)
    ledger["backtest_label"] = ledger["research_outcome_label"]
    ledger["paper_label"] = ""
    ledger["live_label"] = ""
    ledger["entry_feature_row_available"] = False
    ledger["realized_outcome_available"] = False
    ledger["point_in_time_source_lineage"] = True
    ledger["uses_dashboard_hindsight_as_training_feature"] = False
    ledger["wizard_is_label_authority"] = False
    ledger["training_eligible"] = False
    ledger["training_blocker"] = (
        "aggregate_experiment_summary_not_trade_entry_row;"
        "post_evaluation_fields_cannot_be_entry_features;"
        "paper_and_live_outcomes_not_available"
    )
    ledger["research_only"] = True
    ledger["order_submission_performed"] = False
    ledger["testnet_order_authority"] = False
    ledger["live_trading_authorized"] = False
    ledger["stage_manifest_evidence_path"] = (
        "reports/active/current_wizard_hyperliquid_concentration_manifest.json;"
        "reports/active/current_wizard_hyperliquid_failure_attribution_manifest.json;"
        "reports/active/current_wizard_hyperliquid_leverage_manifest.json"
    )
    ledger["learning_snapshot_evidence_path"] = _relative(snapshot_dir, root)

    paths = _paths(root, active, snapshot_dir)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    csv_bytes = gzip.compress(ledger.to_csv(index=False).encode("utf-8"), mtime=0)
    jsonl_bytes = gzip.compress(
        ledger.to_json(orient="records", lines=True, date_format="iso").encode(
            "utf-8"
        ),
        mtime=0,
    )
    for key in ("ledger", "dataset", "snapshot_ledger"):
        atomic_write_bytes(paths[key], csv_bytes)
    for key in ("ledger_jsonl", "dataset_jsonl", "snapshot_ledger_jsonl"):
        atomic_write_bytes(paths[key], jsonl_bytes)

    outcome_counts = {
        _text(key): int(value)
        for key, value in ledger["research_outcome_label"].value_counts().items()
    }
    validation = _validation(matrix, ledger)
    if not validation["status"].eq("PASS").all():
        failed = validation.loc[validation["status"].ne("PASS"), "check"].tolist()
        raise ValueError("Current learning validation failed: " + ",".join(failed))
    atomic_write_csv(validation, paths["validation"], index=False)
    atomic_write_csv(validation, paths["snapshot_validation"], index=False)

    summary: dict[str, object] = {
        **material,
        "learning_ledger_id": learning_id,
        "records": int(len(ledger)),
        "unique_experiment_ids": int(ledger["experiment_id"].nunique()),
        "experiment_status_accounted": bool(
            len(ledger) == len(matrix) == ledger["experiment_id"].nunique()
        ),
        "outcome_type": "backtest_research_outcome",
        "record_granularity": "experiment_summary",
        "outcome_counts": outcome_counts,
        "training_eligible_records": 0,
        "paper_label_records": 0,
        "live_label_records": 0,
        "realized_outcome_records": 0,
        "wizard_is_label_authority": False,
        "order_submission_performed": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
        "artifacts": {name: _relative(path, root) for name, path in paths.items()},
        "input_snapshots": {
            name: _relative(path, root)
            for name, path in snapshot_references.items()
        },
        "input_snapshot_modes": {
            name: "verified_upstream_reference" for name in snapshot_references
        },
        "referenced_upstream_bytes": unique_file_bytes(
            list(snapshot_references.values())
        ),
        "locally_copied_input_bytes": 0,
    }
    manifest_text = json.dumps(summary, indent=2, sort_keys=True)
    summary_text = _summary_markdown(summary)
    for key in ("manifest", "snapshot_manifest"):
        atomic_write_text(paths[key], manifest_text, encoding="utf-8")
    for key in ("summary_md", "snapshot_summary_md"):
        atomic_write_text(paths[key], summary_text, encoding="utf-8")
    return CommandResult(paths=paths, summary=summary)


def _outcome_label(row: pd.Series) -> str:
    if _truthy(row.get("ready_for_testnet_1x_lifecycle")):
        return "READY_FOR_TESTNET_1X_RESEARCH"
    if _truthy(row.get("leverage_research_eligible")):
        return "READY_FOR_LEVERAGE_RESEARCH"
    if _truthy(row.get("concentration_gate_pass")):
        return "CONCENTRATION_PASS_NOT_PROMOTED"
    if _text(row.get("statistical_selection_status")) == "PASS":
        return "STATISTICALLY_SELECTED_NOT_PROMOTED"
    if _text(row.get("research_robustness_status")) == "PASS_RESEARCH_ROBUSTNESS":
        return "RESEARCH_ROBUSTNESS_PASS_NOT_PROMOTED"
    if _text(row.get("walkforward_status")) == "PASS_RESEARCH_WALK_FORWARD":
        return "PRACTICAL_WALKFORWARD_PASS_NOT_PROMOTED"
    if _text(row.get("observed_cost_replay_status")) == (
        "OBSERVED_COST_RESEARCH_REPLAY_COMPLETE"
    ):
        return "RESEARCH_REPLAY_EVALUATED_NOT_PROMOTED"
    return "BLOCKED_BEFORE_RESEARCH_EVALUATION"


def _validation(matrix: pd.DataFrame, ledger: pd.DataFrame) -> pd.DataFrame:
    authority_ids = set(matrix["experiment_id"].astype(str))
    checks = {
        "experiment_count_preserved": len(ledger) == len(matrix),
        "experiment_ids_unique": ledger["experiment_id"].nunique() == len(matrix),
        "experiment_id_set_preserved": set(ledger["experiment_id"].astype(str))
        == authority_ids,
        "learning_record_ids_unique": ledger["learning_record_id"].nunique()
        == len(matrix),
        "outcome_known_at_present": ledger["outcome_known_at"].astype(str).ne("").all(),
        "outcome_labels_present": ledger["research_outcome_label"]
        .astype(str)
        .ne("")
        .all(),
        "aggregate_rows_not_training_eligible": not ledger["training_eligible"]
        .map(_truthy)
        .any(),
        "paper_labels_absent": ledger["paper_label"].astype(str).eq("").all(),
        "live_labels_absent": ledger["live_label"].astype(str).eq("").all(),
        "realized_outcomes_absent": not ledger["realized_outcome_available"]
        .map(_truthy)
        .any(),
        "wizard_not_label_authority": not ledger["wizard_is_label_authority"]
        .map(_truthy)
        .any(),
        "order_submission_absent": not ledger["order_submission_performed"]
        .map(_truthy)
        .any(),
        "testnet_order_authority_disabled": not ledger["testnet_order_authority"]
        .map(_truthy)
        .any(),
        "live_trading_disabled": not ledger["live_trading_authorized"]
        .map(_truthy)
        .any(),
    }
    return pd.DataFrame(
        [
            {"check": check, "status": "PASS" if passed else "FAIL"}
            for check, passed in checks.items()
        ]
    )


def _paths(root: Path, active: Path, snapshot: Path) -> dict[str, Path]:
    stem = "current_wizard_hyperliquid_learning"
    return {
        "ledger": active / f"{stem}_ledger.csv.gz",
        "ledger_jsonl": active / f"{stem}_ledger.jsonl.gz",
        "dataset": root
        / "data"
        / "meta_learning"
        / "current_wizard_hyperliquid_research_outcomes.csv.gz",
        "dataset_jsonl": root
        / "data"
        / "meta_learning"
        / "current_wizard_hyperliquid_research_outcomes.jsonl.gz",
        "validation": active / f"{stem}_validation.csv",
        "manifest": active / f"{stem}_manifest.json",
        "summary_md": active / f"{stem}_summary.md",
        "snapshot_ledger": snapshot / "learning_ledger.csv.gz",
        "snapshot_ledger_jsonl": snapshot / "learning_ledger.jsonl.gz",
        "snapshot_validation": snapshot / "validation.csv",
        "snapshot_manifest": snapshot / "manifest.json",
        "snapshot_summary_md": snapshot / "summary.md",
    }


def _summary_markdown(summary: dict[str, object]) -> str:
    lines = [
        "# Current Wizard To Hyperliquid Learning Ledger",
        "",
        "This is a dated research-outcome ledger, not a trade-entry training dataset and not realized paper or live evidence.",
        "",
        f"- learning ledger: {summary['learning_ledger_id']}",
        f"- current-board refresh: {summary['refresh_id']}",
        f"- experiment records: {summary['records']}",
        f"- unique experiment IDs: {summary['unique_experiment_ids']}",
        f"- complete accounting: {str(summary['experiment_status_accounted']).lower()}",
        f"- training-eligible records: {summary['training_eligible_records']}",
        f"- paper labels: {summary['paper_label_records']}",
        f"- live labels: {summary['live_label_records']}",
        "- Crypto Wizards label authority: false",
        "- order submission performed: false",
        "- live trading authorized: false",
        "",
        "## Outcome Counts",
        "",
    ]
    for label, count in summary["outcome_counts"].items():
        lines.append(f"- {label}: {count}")
    return "\n".join(lines) + "\n"


def _require_all_cell_accounting(
    matrix: pd.DataFrame,
    failure: pd.DataFrame,
    leverage: pd.DataFrame,
) -> None:
    authority_ids = set(matrix["experiment_id"].astype(str))
    if matrix["experiment_id"].astype(str).duplicated().any():
        raise ValueError("Current learning matrix has duplicate experiment_id")
    for name, frame in (("failure", failure), ("leverage", leverage)):
        if frame["experiment_id"].astype(str).duplicated().any():
            raise ValueError(f"Current learning {name} has duplicate experiment_id")
        ids = set(frame["experiment_id"].astype(str))
        if ids != authority_ids:
            raise ValueError(
                f"Current learning {name} accounting mismatch: "
                f"missing={len(authority_ids - ids)},extra={len(ids - authority_ids)}"
            )


def _require_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"Current learning {name} is missing columns: {missing}")


def _single_value(frame: pd.DataFrame, column: str, name: str) -> str:
    values = {_text(value) for value in frame[column] if _text(value)}
    if len(values) != 1:
        raise ValueError(
            f"Current learning {name} requires one non-empty {column}; found {values}"
        )
    return next(iter(values))


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False, low_memory=False)


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _as_utc(value: datetime) -> datetime:
    return (
        value.astimezone(timezone.utc)
        if value.tzinfo
        else value.replace(tzinfo=timezone.utc)
    )


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return _text(value).lower() in {"1", "true", "yes", "y", "pass", "ready"}


def _text(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()
