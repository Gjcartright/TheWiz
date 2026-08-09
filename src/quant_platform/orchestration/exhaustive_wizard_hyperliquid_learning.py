"""Dated, research-only learning ledger for exhaustive validation outcomes."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil

import pandas as pd

from quant_platform.active_pipeline import CommandResult


ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "exhaustive_wizard_hyperliquid_learning.v1"

SOURCE_FILES = {
    "experiment_matrix": "exhaustive_wizard_experiment_matrix.csv",
    "cost": "exhaustive_wizard_hyperliquid_experiment_cost_readiness.csv",
    "observed": "exhaustive_wizard_hyperliquid_observed_cost_replay.csv",
    "walkforward": "exhaustive_wizard_hyperliquid_walkforward_status.csv",
    "regime": "exhaustive_wizard_hyperliquid_regime_status.csv",
    "robustness": "exhaustive_wizard_hyperliquid_robustness_status.csv",
    "concentration": "exhaustive_wizard_hyperliquid_concentration_status.csv",
    "leverage": "exhaustive_wizard_hyperliquid_leverage_status.csv",
}

STAGE_ID_FIELDS = {
    "cost_evidence": ("cost", "cost_evidence_id"),
    "observed_cost_replay": ("observed", "observed_cost_replay_id"),
    "walkforward": ("walkforward", "walkforward_id"),
    "regime_attribution": ("regime", "regime_attribution_id"),
    "robustness": ("robustness", "robustness_id"),
    "concentration": ("concentration", "concentration_id"),
    "leverage_surface": ("leverage", "leverage_surface_id"),
}

BASE_COLUMNS = [
    "experiment_id",
    "pair_group_id",
    "pair",
    "wizard_exchange",
    "timeframe",
    "exact_mode",
    "orientation",
    "asset_x",
    "asset_y",
    "hyperliquid_pair_ready",
    "hyperliquid_pair_max_leverage",
    "hyperliquid_mapping_blocker",
    "source_row_ids",
    "evidence_path",
]

STAGE_COLUMNS = {
    "cost": {
        "preflight_status": "preflight_status",
        "cost_evidence_status": "cost_evidence_status",
        "provisional_cost_research_ready": "provisional_cost_research_ready",
        "cost_acceptance_ready": "cost_acceptance_ready",
        "cost_replay_status": "cost_replay_status",
        "cost_replay_blocker": "cost_replay_blocker",
    },
    "observed": {
        "scanner_cutoff_at": "scanner_cutoff_at",
        "hyperliquid_interval": "hyperliquid_interval",
        "canonical_replay_leverage": "canonical_replay_leverage",
        "observed_funding_rows": "observed_funding_rows",
        "funding_both_coverage": "funding_both_coverage",
        "taker_fee_bps": "taker_fee_bps",
        "slippage_x_p95_bps": "slippage_x_p95_bps",
        "slippage_y_p95_bps": "slippage_y_p95_bps",
        "execution_risk_bps": "execution_risk_bps",
        "replay_status": "observed_replay_status",
        "replay_blocker": "observed_replay_blocker",
        "mode_fidelity_status": "mode_fidelity_status",
        "mode_fidelity_reason": "mode_fidelity_reason",
        "math_version": "math_version",
        "trades": "observed_trades",
        "profit_factor": "observed_profit_factor",
        "expectancy": "observed_expectancy",
        "sharpe": "observed_sharpe",
        "max_drawdown": "observed_max_drawdown",
        "total_return": "observed_total_return",
        "research_rank_eligible": "observed_research_rank_eligible",
        "research_rank_blocker": "observed_research_rank_blocker",
    },
    "walkforward": {
        "walkforward_status": "walkforward_status",
        "walkforward_blocker": "walkforward_blocker",
        "folds_complete": "walkforward_folds_complete",
        "positive_folds": "walkforward_positive_folds",
        "aggregate_trades": "walkforward_trades",
        "aggregate_profit_factor": "walkforward_profit_factor",
        "aggregate_expectancy": "walkforward_expectancy",
        "aggregate_sharpe": "walkforward_sharpe",
        "aggregate_max_drawdown": "walkforward_max_drawdown",
        "aggregate_total_return": "walkforward_total_return",
        "fold_return_raw_pvalue": "walkforward_raw_pvalue",
        "bh_qvalue": "walkforward_bh_qvalue",
        "parameter_stability_status": "parameter_stability_status",
        "deflated_sharpe_status": "deflated_sharpe_status",
        "statistical_selection_status": "statistical_selection_status",
        "statistical_selection_blocker": "statistical_selection_blocker",
    },
    "regime": {
        "regime_status": "regime_status",
        "regime_blocker": "regime_blocker",
        "regimes_observed": "regimes_observed",
        "regimes_with_minimum_trades": "regimes_with_minimum_trades",
        "positive_expectancy_regimes": "positive_expectancy_regimes",
        "regime_profit_concentration": "regime_profit_concentration",
        "worst_regime_expectancy": "worst_regime_expectancy",
        "crisis_trades": "crisis_trades",
        "crisis_expectancy": "crisis_expectancy",
        "regime_stability_status": "regime_stability_status",
        "regime_stability_blocker": "regime_stability_blocker",
    },
    "robustness": {
        "robustness_status": "robustness_status",
        "robustness_blocker": "robustness_blocker",
        "scenarios_complete": "robustness_scenarios_complete",
        "parameter_positive_ratio": "parameter_positive_ratio",
        "parameter_pass_ratio": "parameter_pass_ratio",
        "worst_parameter_drawdown": "worst_parameter_drawdown",
        "research_robustness_status": "research_robustness_status",
        "research_robustness_blocker": "research_robustness_blocker",
        "promotion_readiness": "robustness_promotion_readiness",
        "promotion_blocker": "robustness_promotion_blocker",
    },
    "concentration": {
        "concentration_status": "concentration_status",
        "concentration_blocker": "concentration_blocker",
        "concentration_gate_pass": "concentration_gate_pass",
        "ready_for_leverage_gate": "ready_for_leverage_gate",
    },
    "leverage": {
        "leverage_surface_status": "leverage_surface_status",
        "leverage_surface_blocker": "leverage_surface_blocker",
        "ready_for_testnet_1x_lifecycle": "ready_for_testnet_1x_lifecycle",
        "acceptance_status": "final_acceptance_status",
        "acceptance_reason": "final_acceptance_reason",
        "acceptance_eligible": "final_acceptance_eligible",
    },
}


def build_exhaustive_wizard_hyperliquid_learning_ledger(
    *,
    root: Path = ROOT,
    now: datetime | None = None,
    validation_id: str | None = None,
    validation_stages: list[dict[str, object]] | None = None,
) -> CommandResult:
    """Materialize one non-training research outcome for every experiment."""

    as_of = _as_utc(now or datetime.now(timezone.utc))
    active = root / "reports" / "active"
    if validation_id is None or validation_stages is None:
        validation = _read_json(
            active / "exhaustive_wizard_hyperliquid_validation_manifest.json"
        )
        validation_id = validation_id or _text(validation.get("validation_id"))
        validation_stages = validation_stages or list(validation.get("stages") or [])
    if not validation_id:
        raise ValueError("A frozen validation_id is required for the learning ledger")

    input_paths = {name: active / filename for name, filename in SOURCE_FILES.items()}
    missing = [str(path) for path in input_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Exhaustive learning inputs missing: {missing}")
    frames = {name: _read_csv(path) for name, path in input_paths.items()}
    matrix = frames["experiment_matrix"]
    _require_columns(matrix, ["exhaustive_run_id", *BASE_COLUMNS], "experiment_matrix")
    _require_unique(matrix, "experiment_id", "experiment_matrix")
    run_id = _single_value(matrix, "exhaustive_run_id", "experiment_matrix")
    experiment_ids = set(matrix["experiment_id"].astype(str))

    stage_context = {
        _text(row.get("stage")): row for row in validation_stages if _text(row.get("stage"))
    }
    missing_stages = sorted(set(STAGE_ID_FIELDS).difference(stage_context))
    if missing_stages:
        raise ValueError(f"Validation context is missing stages: {missing_stages}")
    stage_manifest_paths: dict[str, Path] = {}
    for stage_name in STAGE_ID_FIELDS:
        manifest_value = _text(stage_context[stage_name].get("manifest_path"))
        manifest_path = Path(manifest_value)
        if not manifest_path.is_absolute():
            manifest_path = root / manifest_path
        if not manifest_value or not manifest_path.exists():
            raise FileNotFoundError(
                f"Validation stage manifest is missing for {stage_name}: {manifest_path}"
            )
        stage_manifest_paths[stage_name] = manifest_path
    for stage_name, (frame_name, identity_column) in STAGE_ID_FIELDS.items():
        frame = frames[frame_name]
        _require_unique(frame, "experiment_id", frame_name)
        current_ids = set(frame["experiment_id"].astype(str))
        if current_ids != experiment_ids:
            missing_ids = len(experiment_ids.difference(current_ids))
            extra_ids = len(current_ids.difference(experiment_ids))
            raise ValueError(
                f"{frame_name} experiment accounting mismatch: "
                f"missing={missing_ids}, extra={extra_ids}"
            )
        expected_identity = _text(stage_context[stage_name].get("stage_identity"))
        actual_identity = _single_value(frame, identity_column, frame_name)
        if actual_identity != expected_identity:
            raise ValueError(
                f"{frame_name} identity {actual_identity} does not match "
                f"validation stage {expected_identity}"
            )
        if "exhaustive_run_id" in frame.columns:
            frame_run_id = _single_value(frame, "exhaustive_run_id", frame_name)
            if frame_run_id != run_id:
                raise ValueError(f"{frame_name} run identity does not match experiment matrix")

    input_hashes = {name: _file_hash(path) for name, path in input_paths.items()}
    material = {
        "schema_version": SCHEMA_VERSION,
        "validation_id": validation_id,
        "run_id": run_id,
        "as_of": as_of.isoformat(),
        "stage_identities": {
            stage: _text(stage_context[stage].get("stage_identity"))
            for stage in STAGE_ID_FIELDS
        },
        "input_hashes": input_hashes,
        "stage_manifest_hashes": {
            stage: _file_hash(path) for stage, path in stage_manifest_paths.items()
        },
    }
    digest = sha256(_canonical_json(material).encode("utf-8")).hexdigest()
    learning_id = f"hllearning_{as_of.strftime('%Y%m%dT%H%M%S%fZ')}_{digest[:8]}"
    snapshot_dir = (
        root
        / "reports"
        / "snapshots"
        / "exhaustive_wizard_hyperliquid"
        / run_id
        / "validations"
        / validation_id
        / "learning"
    )
    snapshot_inputs = snapshot_dir / "inputs"
    snapshot_inputs.mkdir(parents=True, exist_ok=True)
    for source in input_paths.values():
        shutil.copy2(source, snapshot_inputs / source.name)
    for stage, source in stage_manifest_paths.items():
        shutil.copy2(source, snapshot_inputs / f"{stage}_manifest.json")

    ledger = matrix.loc[:, BASE_COLUMNS].copy()
    ledger = ledger.rename(
        columns={
            "timeframe": "wizard_timeframe",
            "evidence_path": "source_evidence_path",
        }
    )
    for frame_name, column_map in STAGE_COLUMNS.items():
        ledger = _join_stage(ledger, frames[frame_name], column_map, frame_name)

    stage_evidence = ";".join(
        _text(stage_context[name].get("manifest_path")) for name in STAGE_ID_FIELDS
    )
    ledger.insert(0, "schema_version", SCHEMA_VERSION)
    ledger.insert(1, "learning_ledger_id", learning_id)
    ledger.insert(2, "validation_id", validation_id)
    ledger.insert(3, "exhaustive_run_id", run_id)
    ledger.insert(4, "learning_record_id", ledger["experiment_id"].map(
        lambda value: "hllearnrec_"
        + sha256(f"{validation_id}|{value}".encode("utf-8")).hexdigest()[:20]
    ))
    ledger.insert(5, "outcome_known_at", as_of.isoformat())
    ledger["outcome_type"] = "backtest_research_outcome"
    ledger["record_granularity"] = "experiment_summary"
    ledger["research_outcome_label"] = ledger.apply(_research_outcome_label, axis=1)
    ledger["backtest_label"] = ledger["research_outcome_label"]
    ledger["paper_label"] = ""
    ledger["live_label"] = ""
    ledger["entry_feature_row_available"] = False
    ledger["realized_outcome_available"] = False
    ledger["point_in_time_source_lineage"] = True
    ledger["uses_dashboard_hindsight_as_feature"] = False
    ledger["wizard_is_label_authority"] = False
    ledger["training_eligible"] = False
    ledger["training_blocker"] = (
        "aggregate_experiment_summary_not_trade_entry_row;"
        "post_evaluation_fields_cannot_be_entry_features;"
        "paper_and_live_outcomes_not_available"
    )
    ledger["research_only"] = True
    ledger["order_submission_performed"] = False
    ledger["live_trading_authorized"] = False
    ledger["stage_manifest_evidence_path"] = stage_evidence
    ledger["learning_snapshot_evidence_path"] = _relative(snapshot_dir, root)

    paths = {
        "ledger": active / "exhaustive_wizard_hyperliquid_learning_ledger.csv",
        "ledger_jsonl": active / "exhaustive_wizard_hyperliquid_learning_ledger.jsonl",
        "dataset": root
        / "data"
        / "meta_learning"
        / "exhaustive_wizard_hyperliquid_research_outcomes.csv",
        "dataset_jsonl": root
        / "data"
        / "meta_learning"
        / "exhaustive_wizard_hyperliquid_research_outcomes.jsonl",
        "manifest": active / "exhaustive_wizard_hyperliquid_learning_manifest.json",
        "summary_md": active / "exhaustive_wizard_hyperliquid_learning_summary.md",
        "snapshot_ledger": snapshot_dir / "learning_ledger.csv",
        "snapshot_ledger_jsonl": snapshot_dir / "learning_ledger.jsonl",
        "snapshot_manifest": snapshot_dir / "manifest.json",
        "snapshot_summary_md": snapshot_dir / "summary.md",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    csv_text = ledger.to_csv(index=False)
    jsonl_text = ledger.to_json(orient="records", lines=True, date_format="iso")
    for key in ("ledger", "dataset", "snapshot_ledger"):
        paths[key].write_text(csv_text, encoding="utf-8")
    for key in ("ledger_jsonl", "dataset_jsonl", "snapshot_ledger_jsonl"):
        paths[key].write_text(jsonl_text, encoding="utf-8")

    outcome_counts = {
        str(key): int(value)
        for key, value in ledger["research_outcome_label"].value_counts().items()
    }
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
        "training_eligible_records": int(ledger["training_eligible"].map(_truthy).sum()),
        "paper_label_records": int(ledger["paper_label"].astype(str).str.strip().ne("").sum()),
        "live_label_records": int(ledger["live_label"].astype(str).str.strip().ne("").sum()),
        "realized_outcome_records": int(
            ledger["realized_outcome_available"].map(_truthy).sum()
        ),
        "wizard_is_label_authority": False,
        "order_submission_performed": False,
        "live_trading_authorized": False,
        "artifacts": {name: _relative(path, root) for name, path in paths.items()},
        "input_snapshots": {
            **{
                name: _relative(snapshot_inputs / path.name, root)
                for name, path in input_paths.items()
            },
            **{
                f"stage_manifest_{stage}": _relative(
                    snapshot_inputs / f"{stage}_manifest.json", root
                )
                for stage in stage_manifest_paths
            },
        },
    }
    manifest = json.dumps(summary, indent=2, sort_keys=True)
    markdown = _summary_markdown(summary)
    for key in ("manifest", "snapshot_manifest"):
        paths[key].write_text(manifest, encoding="utf-8")
    for key in ("summary_md", "snapshot_summary_md"):
        paths[key].write_text(markdown, encoding="utf-8")
    return CommandResult(paths=paths, summary=summary)


def _join_stage(
    ledger: pd.DataFrame,
    source: pd.DataFrame,
    column_map: dict[str, str],
    source_name: str,
) -> pd.DataFrame:
    _require_columns(source, ["experiment_id", *column_map], source_name)
    selected = source.loc[:, ["experiment_id", *column_map]].rename(columns=column_map)
    return ledger.merge(selected, on="experiment_id", how="left", validate="one_to_one")


def _research_outcome_label(row: pd.Series) -> str:
    if _truthy(row.get("ready_for_testnet_1x_lifecycle")):
        return "READY_FOR_TESTNET_1X_RESEARCH"
    if _truthy(row.get("ready_for_leverage_gate")):
        return "READY_FOR_LEVERAGE_RESEARCH"
    if _text(row.get("statistical_selection_status")) == "PASS":
        return "STATISTICALLY_SELECTED_NOT_PROMOTED"
    if _text(row.get("walkforward_status")) == "PASS_RESEARCH_WALK_FORWARD":
        return "PRACTICAL_WALKFORWARD_PASS_NOT_PROMOTED"
    if _text(row.get("observed_replay_status")) == "OBSERVED_COST_RESEARCH_REPLAY_COMPLETE":
        return "RESEARCH_REPLAY_EVALUATED_NOT_PROMOTED"
    return "BLOCKED_BEFORE_RESEARCH_EVALUATION"


def _summary_markdown(summary: dict[str, object]) -> str:
    lines = [
        "# Exhaustive Wizard To Hyperliquid Learning Ledger",
        "",
        "This is a dated research-outcome ledger, not a trade-entry training dataset and not realized paper or live evidence.",
        "",
        f"- learning ledger: {summary['learning_ledger_id']}",
        f"- validation: {summary['validation_id']}",
        f"- exhaustive run: {summary['run_id']}",
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


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _require_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _require_unique(frame: pd.DataFrame, column: str, name: str) -> None:
    _require_columns(frame, [column], name)
    if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
        raise ValueError(f"{name} has blank {column} values")
    if frame[column].astype(str).duplicated().any():
        raise ValueError(f"{name} has duplicate {column} values")


def _single_value(frame: pd.DataFrame, column: str, name: str) -> str:
    _require_columns(frame, [column], name)
    values = sorted({_text(value) for value in frame[column] if _text(value)})
    if len(values) != 1:
        raise ValueError(f"{name} must contain one non-empty {column}; found {values}")
    return values[0]


def _file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return _text(value).lower() in {"1", "true", "yes", "pass", "ready"}


def _text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
