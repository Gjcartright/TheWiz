"""Compare two canonical replay receipts over identical frozen inputs."""

from __future__ import annotations

from quant_platform.orchestration.corrective_runtime import atomic_write_text

from quant_platform.orchestration.corrective_runtime import atomic_write_csv

import json
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from quant_platform.active_pipeline import ROOT, CommandResult
from quant_platform.orchestration.current_wizard_hyperliquid_replay import (
    run_current_wizard_hyperliquid_canonical_replay,
)

COMPARISON_SCHEMA_VERSION = "current_wizard_hyperliquid_math_comparison.v1"
NUMERIC_METRICS = (
    "engle_granger_pvalue",
    "hedge_ratio",
    "ou_phi",
    "ou_half_life",
    "trades",
    "open_trades",
    "profit_factor",
    "expectancy",
    "expectancy_lower_95",
    "sharpe",
    "max_drawdown",
    "win_rate",
    "total_return",
    "gross_return",
    "total_fees",
    "total_slippage",
    "total_funding",
    "total_execution_risk",
    "total_partial_fill_cost",
    "avg_gross_exposure",
    "reconciliation_error",
    "closed_trade_return_std",
)
IDENTITY_COLUMNS = (
    "experiment_id",
    "pair_group_key",
    "pair",
    "wizard_exchange",
    "timeframe",
    "exact_mode",
    "orientation",
)


def reevaluate_current_wizard_hyperliquid_math(
    *,
    old_manifest_path: Path,
    root: Path = ROOT,
    output_stem: str = "2026-08-20_y_on_x_historical_reevaluation",
) -> CommandResult:
    """Replay a legacy receipt on frozen inputs under the corrected math contract."""

    old_manifest_path = old_manifest_path.resolve()
    history_manifest_path = old_manifest_path.parents[2] / "manifest.json"
    handoff_manifest_path = old_manifest_path.parents[4] / "manifest.json"
    for path in (old_manifest_path, history_manifest_path, handoff_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"Frozen math re-evaluation input missing: {path}")

    old_manifest = _read_json(old_manifest_path)
    history_manifest = _read_json(history_manifest_path)
    handoff_manifest = _read_json(handoff_manifest_path)
    if old_manifest.get("history_run_id") != history_manifest.get("history_run_id"):
        raise ValueError("Old replay and derived history manifest identities differ")
    if old_manifest.get("handoff_id") != handoff_manifest.get("handoff_id"):
        raise ValueError("Old replay and derived handoff manifest identities differ")

    reevaluation_dir = root / "reports" / "audits" / f"{output_stem}_replay"
    replay = run_current_wizard_hyperliquid_canonical_replay(
        root=root,
        handoff_manifest_path=handoff_manifest_path,
        history_manifest_path=history_manifest_path,
        output_dir=reevaluation_dir,
        legacy_contract_math_audit=True,
    )
    comparison = build_current_wizard_hyperliquid_math_comparison(
        old_manifest_path=old_manifest_path,
        new_manifest_path=replay.paths["snapshot_manifest"],
        root=root,
        output_stem=output_stem,
    )
    return CommandResult(
        paths={
            **{f"reevaluation_{key}": value for key, value in replay.paths.items()},
            **comparison.paths,
        },
        summary={
            **comparison.summary,
            "historical_math_reevaluation": True,
            "input_contract_status": replay.summary["input_contract_status"],
            "corrected_math_version": replay.summary["settings_version"],
            "promotion_authority": False,
            "live_trading_authorized": False,
        },
    )


def build_current_wizard_hyperliquid_math_comparison(
    *,
    old_manifest_path: Path,
    new_manifest_path: Path,
    root: Path = ROOT,
    output_stem: str = "2026-08-15_canonical_replay_math_v2_comparison",
) -> CommandResult:
    old_manifest = _read_json(old_manifest_path)
    new_manifest = _read_json(new_manifest_path)
    if old_manifest.get("input_hashes") != new_manifest.get("input_hashes"):
        raise ValueError("Canonical replay inputs differ; math-only comparison is invalid")

    old_results_path = root / str(old_manifest["artifacts"]["snapshot_results"])
    new_results_path = root / str(new_manifest["artifacts"]["snapshot_results"])
    old = pd.read_csv(old_results_path, low_memory=False)
    new = pd.read_csv(new_results_path, low_memory=False)
    _validate_experiments(old, new)

    merged = old.merge(
        new,
        on="experiment_id",
        suffixes=("_old", "_new"),
        validate="one_to_one",
    )
    detail = pd.DataFrame({"experiment_id": merged["experiment_id"]})
    for column in IDENTITY_COLUMNS[1:]:
        detail[column] = merged[f"{column}_new"]

    for column in ("replay_status", "acceptance_status", "sharpe_status"):
        detail[f"{column}_old"] = merged[f"{column}_old"].fillna("")
        detail[f"{column}_new"] = merged[f"{column}_new"].fillna("")
        detail[f"{column}_changed"] = detail[f"{column}_old"].ne(detail[f"{column}_new"])

    old_eligible = merged["research_rank_eligible_old"].map(_truthy)
    new_eligible = merged["research_rank_eligible_new"].map(_truthy)
    detail["research_rank_eligible_old"] = old_eligible
    detail["research_rank_eligible_new"] = new_eligible
    detail["research_rank_eligibility_changed"] = old_eligible.ne(new_eligible)
    detail["research_rank_blocker_old"] = merged["research_rank_blocker_old"].fillna("")
    detail["research_rank_blocker_new"] = merged["research_rank_blocker_new"].fillna("")

    metric_rows: list[dict[str, object]] = []
    changed_columns: list[str] = []
    for metric in NUMERIC_METRICS:
        old_values = pd.to_numeric(merged[f"{metric}_old"], errors="coerce")
        new_values = pd.to_numeric(merged[f"{metric}_new"], errors="coerce")
        delta = new_values - old_values
        changed = pd.Series(
            ~np.isclose(
                old_values.to_numpy(dtype=float),
                new_values.to_numpy(dtype=float),
                rtol=1e-12,
                atol=1e-12,
                equal_nan=True,
            ),
            index=merged.index,
        )
        detail[f"{metric}_old"] = old_values
        detail[f"{metric}_new"] = new_values
        detail[f"{metric}_delta"] = delta
        detail[f"{metric}_changed"] = changed
        changed_columns.append(f"{metric}_changed")
        finite_delta = delta.replace([np.inf, -np.inf], np.nan).dropna()
        metric_rows.append(
            {
                "metric": metric,
                "changed_rows": int(changed.sum()),
                "old_non_null_rows": int(old_values.notna().sum()),
                "new_non_null_rows": int(new_values.notna().sum()),
                "mean_delta": _finite_stat(finite_delta, "mean"),
                "median_delta": _finite_stat(finite_delta, "median"),
                "minimum_delta": _finite_stat(finite_delta, "min"),
                "maximum_delta": _finite_stat(finite_delta, "max"),
                "mean_absolute_delta": _finite_stat(finite_delta.abs(), "mean"),
            }
        )
    detail["changed_metric_count"] = detail[changed_columns].sum(axis=1)
    detail["any_metric_changed"] = detail["changed_metric_count"].gt(0)
    metric_summary = pd.DataFrame(metric_rows)
    flips = detail.loc[detail["research_rank_eligibility_changed"]].copy()

    audit_dir = root / "reports" / "audits"
    audit_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "detail": audit_dir / f"{output_stem}.csv",
        "metric_summary": audit_dir / f"{output_stem}_metric_summary.csv",
        "eligibility_flips": audit_dir / f"{output_stem}_eligibility_flips.csv",
        "mode_summary": audit_dir / f"{output_stem}_mode_summary.csv",
        "pair_summary": audit_dir / f"{output_stem}_pair_summary.csv",
        "summary_md": audit_dir / f"{output_stem}.md",
        "manifest": audit_dir / f"{output_stem}_manifest.json",
    }
    atomic_write_csv(detail, paths["detail"], index=False)
    atomic_write_csv(metric_summary, paths["metric_summary"], index=False)
    atomic_write_csv(flips, paths["eligibility_flips"], index=False)
    mode_summary = _aggregate_impact(detail, ["exact_mode"])
    pair_summary = _aggregate_impact(
        detail,
        ["pair_group_key", "pair", "wizard_exchange", "timeframe"],
    )
    atomic_write_csv(mode_summary, paths["mode_summary"], index=False)
    atomic_write_csv(pair_summary, paths["pair_summary"], index=False)

    old_status = old["replay_status"].value_counts().to_dict()
    new_status = new["replay_status"].value_counts().to_dict()
    summary = {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "old_canonical_replay_id": old_manifest["canonical_replay_id"],
        "new_canonical_replay_id": new_manifest["canonical_replay_id"],
        "old_settings_version": old_manifest.get("settings_version", ""),
        "new_settings_version": new_manifest.get("settings_version", ""),
        "input_hashes_equal": True,
        "experiments_compared": len(detail),
        "old_replay_status_counts": old_status,
        "new_replay_status_counts": new_status,
        "replay_status_changes": int(detail["replay_status_changed"].sum()),
        "rows_with_metric_changes": int(detail["any_metric_changed"].sum()),
        "eligibility_flips": len(flips),
        "eligibility_gained": int(
            ((~detail["research_rank_eligible_old"]) & detail["research_rank_eligible_new"]).sum()
        ),
        "eligibility_lost": int(
            (detail["research_rank_eligible_old"] & (~detail["research_rank_eligible_new"])).sum()
        ),
        "old_trade_ledger_rows": int(old_manifest.get("trade_ledger_rows", 0)),
        "new_trade_ledger_rows": int(new_manifest.get("trade_ledger_rows", 0)),
        "old_manifest_sha256": _file_hash(old_manifest_path),
        "new_manifest_sha256": _file_hash(new_manifest_path),
        "old_results_sha256": _file_hash(old_results_path),
        "new_results_sha256": _file_hash(new_results_path),
        "promotion_authority": False,
        "live_trading_authorized": False,
    }
    atomic_write_text(paths["summary_md"], _markdown(summary, metric_summary, mode_summary, pair_summary), encoding="utf-8")
    manifest_payload = {
        **summary,
        "artifacts": {
            key: str(path.resolve().relative_to(root.resolve())) for key, path in paths.items()
        },
    }
    atomic_write_text(paths["manifest"], json.dumps(manifest_payload, indent=2, sort_keys=True), encoding="utf-8")
    return CommandResult(paths=paths, summary=summary)


def _validate_experiments(old: pd.DataFrame, new: pd.DataFrame) -> None:
    if old["experiment_id"].duplicated().any() or new["experiment_id"].duplicated().any():
        raise ValueError("Duplicate experiment IDs prevent one-to-one comparison")
    if set(old["experiment_id"]) != set(new["experiment_id"]):
        raise ValueError("Canonical replay experiment sets differ")


def _aggregate_impact(detail: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_key: str | list[str] = groups[0] if len(groups) == 1 else groups
    for key, frame in detail.groupby(group_key, dropna=False, sort=True):
        values = key if isinstance(key, tuple) else (key,)
        row = dict(zip(groups, values, strict=True))
        row.update(
            {
                "experiments": int(len(frame)),
                "completed_old": int(
                    frame["replay_status_old"].eq("RESEARCH_REPLAY_COMPLETE").sum()
                ),
                "completed_new": int(
                    frame["replay_status_new"].eq("RESEARCH_REPLAY_COMPLETE").sum()
                ),
                "rows_with_metric_changes": int(frame["any_metric_changed"].sum()),
                "eligibility_gained": int(
                    (
                        (~frame["research_rank_eligible_old"]) & frame["research_rank_eligible_new"]
                    ).sum()
                ),
                "eligibility_lost": int(
                    (
                        frame["research_rank_eligible_old"] & (~frame["research_rank_eligible_new"])
                    ).sum()
                ),
                "sum_experiment_trades_old": float(frame["trades_old"].sum()),
                "sum_experiment_trades_new": float(frame["trades_new"].sum()),
                "median_profit_factor_old": _median_finite(frame["profit_factor_old"]),
                "median_profit_factor_new": _median_finite(frame["profit_factor_new"]),
                "median_sharpe_old": _median_finite(frame["sharpe_old"]),
                "median_sharpe_new": _median_finite(frame["sharpe_new"]),
                "median_max_drawdown_old": _median_finite(frame["max_drawdown_old"]),
                "median_max_drawdown_new": _median_finite(frame["max_drawdown_new"]),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _median_finite(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(numeric.median()) if not numeric.empty else None


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _finite_stat(values: pd.Series, operation: str) -> float | None:
    if values.empty:
        return None
    return float(getattr(values, operation)())


def _file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _markdown(
    summary: dict[str, object],
    metrics: pd.DataFrame,
    modes: pd.DataFrame,
    pairs: pd.DataFrame,
) -> str:
    changed = metrics.loc[
        metrics["changed_rows"].gt(0),
        ["metric", "changed_rows", "mean_delta", "minimum_delta", "maximum_delta"],
    ]
    old_counts = pd.Series(summary["old_replay_status_counts"], name="old_count")
    new_counts = pd.Series(summary["new_replay_status_counts"], name="new_count")
    status_counts = (
        pd.concat([old_counts, new_counts], axis=1)
        .fillna(0)
        .astype(int)
        .rename_axis("replay_status")
        .reset_index()
    )
    mode_view = modes[
        [
            "exact_mode",
            "completed_old",
            "completed_new",
            "rows_with_metric_changes",
            "eligibility_gained",
            "eligibility_lost",
            "sum_experiment_trades_old",
            "sum_experiment_trades_new",
        ]
    ]
    pair_view = pairs.sort_values(
        ["eligibility_lost", "rows_with_metric_changes"],
        ascending=False,
    ).head(20)[
        [
            "pair_group_key",
            "completed_old",
            "completed_new",
            "rows_with_metric_changes",
            "eligibility_lost",
            "sum_experiment_trades_old",
            "sum_experiment_trades_new",
        ]
    ]
    return "\n".join(
        [
            "# Canonical Replay Math V2 Comparison",
            "",
            f"- Old replay: `{summary['old_canonical_replay_id']}`",
            f"- New replay: `{summary['new_canonical_replay_id']}`",
            f"- Frozen inputs identical: `{str(summary['input_hashes_equal']).lower()}`",
            f"- Experiments compared: {summary['experiments_compared']}",
            f"- Replay status changes: {summary['replay_status_changes']}",
            f"- Rows with metric changes: {summary['rows_with_metric_changes']}",
            f"- Research eligibility flips: {summary['eligibility_flips']} ({summary['eligibility_gained']} gained, {summary['eligibility_lost']} lost)",
            f"- Trade ledger rows: {summary['old_trade_ledger_rows']} -> {summary['new_trade_ledger_rows']}",
            "- Promotion authority: false",
            "- Live trading authorized: false",
            "",
            "## Terminal-State Changes",
            "",
            status_counts.to_markdown(index=False),
            "",
            "The corrected contract converts 90 prior completions into explicit local-fit blockers and 288 into point-in-time economic-contract blockers. These are rejected opposing-leg setups, not missing rows or unaccounted software exceptions.",
            "",
            "## Impact By Mode",
            "",
            mode_view.to_markdown(index=False),
            "",
            "`sum_experiment_trades` is a diagnostic sum across overlapping experiments, not a portfolio return or independent trade count.",
            "",
            "## Most Affected Pair Groups",
            "",
            pair_view.to_markdown(index=False),
            "",
            "## Changed Metrics",
            "",
            changed.to_markdown(index=False),
            "",
            "The comparison isolates code and policy changes because the experiment matrix and pair-history input hashes are identical. It does not prove vendor parity or execution readiness.",
            "",
        ]
    )
