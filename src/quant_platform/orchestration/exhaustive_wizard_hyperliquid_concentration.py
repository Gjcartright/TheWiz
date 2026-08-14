"""Cross-cell concentration controls for exhaustive Hyperliquid research."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import shutil

import pandas as pd

from quant_platform.active_pipeline import CommandResult


ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "exhaustive_wizard_hyperliquid_concentration.v1"
COHORTS = (
    "practical_walkforward_pass",
    "research_robustness_pass",
    "statistically_selected_robustness_pass",
)
DIMENSION_POLICIES: dict[str, dict[str, float | int]] = {
    "canonical_pair": {
        "minimum_values": 3,
        "maximum_positive_profit_share": 0.35,
        "maximum_trade_share": 0.40,
        "minimum_effective_profit_breadth": 3.0,
    },
    "wizard_timeframe": {
        "minimum_values": 2,
        "maximum_positive_profit_share": 0.80,
        "maximum_trade_share": 0.85,
        "minimum_effective_profit_breadth": 1.5,
    },
    "wizard_exchange": {
        "minimum_values": 2,
        "maximum_positive_profit_share": 0.60,
        "maximum_trade_share": 0.70,
        "minimum_effective_profit_breadth": 2.0,
    },
    "exact_mode": {
        "minimum_values": 2,
        "maximum_positive_profit_share": 0.60,
        "maximum_trade_share": 0.70,
        "minimum_effective_profit_breadth": 2.0,
    },
    "orientation": {
        "minimum_values": 2,
        "maximum_positive_profit_share": 0.75,
        "maximum_trade_share": 0.80,
        "minimum_effective_profit_breadth": 1.5,
    },
    "regime": {
        "minimum_values": 2,
        "maximum_positive_profit_share": 0.70,
        "maximum_trade_share": 0.75,
        "minimum_effective_profit_breadth": 1.5,
    },
}
RESEARCH_ONLY_REASON = (
    "concentration_is_research_only;"
    "current_l2_depth_is_point_in_time_not_historical_execution_evidence;"
    "mode_fidelity_parity_not_proven;testnet_lifecycle_not_proven"
)


def build_exhaustive_wizard_hyperliquid_concentration(
    *,
    root: Path = ROOT,
    now: datetime | None = None,
) -> CommandResult:
    """Measure whether research gains collapse into a narrow source cell."""

    as_of = _as_utc(now or datetime.now(timezone.utc))
    active = root / "reports" / "active"
    robustness_manifest_path = (
        active / "exhaustive_wizard_hyperliquid_robustness_manifest.json"
    )
    walkforward_manifest_path = (
        active / "exhaustive_wizard_hyperliquid_walkforward_manifest.json"
    )
    regime_manifest_path = active / "exhaustive_wizard_hyperliquid_regime_manifest.json"
    for path in (
        robustness_manifest_path,
        walkforward_manifest_path,
        regime_manifest_path,
    ):
        if not path.exists():
            raise FileNotFoundError(f"Concentration manifest missing: {path}")

    robustness_manifest = _read_json(robustness_manifest_path)
    walkforward_manifest = _read_json(walkforward_manifest_path)
    regime_manifest = _read_json(regime_manifest_path)
    run_id = _text(robustness_manifest.get("run_id"))
    walkforward_id = _text(robustness_manifest.get("walkforward_id"))
    robustness_id = _text(robustness_manifest.get("robustness_id"))
    regime_attribution_id = _text(regime_manifest.get("regime_attribution_id"))
    if walkforward_id != _text(walkforward_manifest.get("walkforward_id")):
        raise ValueError("Robustness and walk-forward identities do not match")
    if walkforward_id != _text(regime_manifest.get("walkforward_id")):
        raise ValueError("Regime and walk-forward identities do not match")

    robustness_artifacts = robustness_manifest.get("artifacts", {})
    walkforward_artifacts = walkforward_manifest.get("artifacts", {})
    regime_artifacts = regime_manifest.get("artifacts", {})
    input_paths = {
        "robustness_status": root / _text(robustness_artifacts.get("snapshot_status")),
        "robustness_candidates": root
        / _text(robustness_artifacts.get("snapshot_candidates")),
        "walkforward_candidates": root
        / _text(walkforward_artifacts.get("snapshot_candidates")),
        "walkforward_trades": root / _text(walkforward_artifacts.get("snapshot_trades")),
        "regime_detail": root / _text(regime_artifacts.get("snapshot_detail")),
        "robustness_manifest": robustness_manifest_path,
        "walkforward_manifest": walkforward_manifest_path,
        "regime_manifest": regime_manifest_path,
    }
    missing = [str(path) for path in input_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Concentration inputs missing: {missing}")

    statuses = _read_csv(input_paths["robustness_status"])
    robustness = _read_csv(input_paths["robustness_candidates"])
    candidates = _read_csv(input_paths["walkforward_candidates"])
    trades = _read_csv(input_paths["walkforward_trades"])
    regimes = _read_csv(input_paths["regime_detail"])
    _require_unique(statuses, "experiment_id")
    _require_unique(robustness, "experiment_id")
    _require_unique(candidates, "experiment_id")
    if statuses.empty:
        raise ValueError("Concentration requires non-empty robustness status evidence")

    cohort_ids = {
        "practical_walkforward_pass": _id_set(
            candidates,
            candidates.get("walkforward_status", pd.Series("", index=candidates.index)).eq(
                "PASS_RESEARCH_WALK_FORWARD"
            ),
        ),
        "research_robustness_pass": _id_set(
            robustness,
            robustness.get(
                "research_robustness_status", pd.Series("", index=robustness.index)
            ).eq("PASS_RESEARCH_ROBUSTNESS"),
        ),
        "statistically_selected_robustness_pass": _id_set(
            robustness,
            robustness.get(
                "research_robustness_status", pd.Series("", index=robustness.index)
            ).eq("PASS_RESEARCH_ROBUSTNESS")
            & robustness.get(
                "statistical_selection_status", pd.Series("", index=robustness.index)
            ).eq("PASS"),
        ),
    }
    if not cohort_ids["research_robustness_pass"].issubset(
        cohort_ids["practical_walkforward_pass"]
    ):
        raise ValueError("Robustness-pass cohort is not a subset of walk-forward passes")
    if not cohort_ids["statistically_selected_robustness_pass"].issubset(
        cohort_ids["research_robustness_pass"]
    ):
        raise ValueError("Statistically selected cohort is not a subset of robustness passes")

    policy = {
        "cohorts": list(COHORTS),
        "dimensions": DIMENSION_POLICIES,
        "profit_contribution": "positive_group_net_profit_after_cost",
        "loss_contribution": "absolute_negative_group_net_profit_after_cost",
        "trade_contribution": "closed_trade_count",
        "promotion_cohort": "statistically_selected_robustness_pass",
        "promotion_requires_all_dimensions": True,
        "acceptance_authority": False,
    }
    material = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "walkforward_id": walkforward_id,
        "regime_attribution_id": regime_attribution_id,
        "robustness_id": robustness_id,
        "as_of": as_of.isoformat(),
        "policy": policy,
        "input_hashes": {name: _file_hash(path) for name, path in input_paths.items()},
    }
    digest = sha256(_canonical_json(material).encode("utf-8")).hexdigest()
    concentration_id = f"hlconcentration_{as_of.strftime('%Y%m%dT%H%M%S%fZ')}_{digest[:8]}"
    robustness_snapshot_manifest = root / _text(
        robustness_artifacts.get("snapshot_manifest")
    )
    snapshot_dir = robustness_snapshot_manifest.parent / "concentration" / concentration_id
    snapshot_inputs_dir = snapshot_dir / "inputs"
    snapshot_inputs_dir.mkdir(parents=True, exist_ok=True)
    snapshot_inputs: dict[str, Path] = {}
    for name, source in input_paths.items():
        target = snapshot_inputs_dir / f"{name}{source.suffix or '.dat'}"
        shutil.copy2(source, target)
        snapshot_inputs[name] = target

    dimension_rows: list[dict[str, object]] = []
    contributor_rows: list[dict[str, object]] = []
    cohort_rows: list[dict[str, object]] = []
    for cohort in COHORTS:
        cohort_dimensions, cohort_contributors = _evaluate_cohort(
            cohort=cohort,
            experiment_ids=cohort_ids[cohort],
            candidates=candidates,
            trades=trades,
            regimes=regimes,
            concentration_id=concentration_id,
        )
        dimension_rows.extend(cohort_dimensions)
        contributor_rows.extend(cohort_contributors)
        passed = bool(
            cohort_ids[cohort]
            and len(cohort_dimensions) == len(DIMENSION_POLICIES)
            and all(_truthy(row["dimension_gate_pass"]) for row in cohort_dimensions)
        )
        blockers = _deduplicate(
            [
                _text(row.get("dimension_blocker"))
                for row in cohort_dimensions
                if _text(row.get("dimension_blocker"))
            ]
        )
        if not cohort_ids[cohort]:
            blockers = ["empty_cohort"]
        cohort_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "concentration_id": concentration_id,
                "cohort": cohort,
                "candidate_count": len(cohort_ids[cohort]),
                "dimensions_required": len(DIMENSION_POLICIES),
                "dimensions_complete": len(cohort_dimensions),
                "dimensions_passed": sum(
                    _truthy(row["dimension_gate_pass"]) for row in cohort_dimensions
                ),
                "cohort_concentration_status": (
                    "PASS_RESEARCH_CONCENTRATION"
                    if passed
                    else (
                        "BLOCKED_EMPTY_COHORT"
                        if not cohort_ids[cohort]
                        else "FAIL_RESEARCH_CONCENTRATION"
                    )
                ),
                "cohort_concentration_blocker": ";".join(blockers),
                "promotion_cohort": cohort
                == "statistically_selected_robustness_pass",
                "ready_for_next_research_gate": bool(
                    passed and cohort == "statistically_selected_robustness_pass"
                ),
                "acceptance_status": "BLOCKED",
                "acceptance_reason": RESEARCH_ONLY_REASON,
                "live_trading_authorized": False,
            }
        )

    cohort_frame = pd.DataFrame(cohort_rows)
    dimension_frame = pd.DataFrame(dimension_rows)
    contributor_frame = pd.DataFrame(contributor_rows)
    promotion_row = cohort_frame.loc[
        cohort_frame["cohort"].eq("statistically_selected_robustness_pass")
    ].iloc[0]
    promotion_pass = _truthy(promotion_row["ready_for_next_research_gate"])
    status_frame = _status_frame(
        statuses,
        robust_ids=cohort_ids["research_robustness_pass"],
        selected_ids=cohort_ids["statistically_selected_robustness_pass"],
        concentration_id=concentration_id,
        promotion_pass=promotion_pass,
        promotion_blocker=_text(promotion_row["cohort_concentration_blocker"]),
        evidence_paths=snapshot_inputs.values(),
        root=root,
    )
    if len(status_frame) != len(statuses) or status_frame["experiment_id"].nunique() != len(
        statuses
    ):
        raise ValueError("Concentration failed complete experiment accounting")
    concentration_status_defaults = {
        "concentration_gate_pass": False,
        "ready_for_leverage_gate": False,
    }
    for column, default in concentration_status_defaults.items():
        status_frame[column] = status_frame.get(
            column,
            pd.Series(index=status_frame.index, dtype=object),
        ).fillna(default)

    paths = {
        "status": active / "exhaustive_wizard_hyperliquid_concentration_status.csv",
        "cohorts": active / "exhaustive_wizard_hyperliquid_concentration_cohorts.csv",
        "dimensions": active
        / "exhaustive_wizard_hyperliquid_concentration_dimensions.csv",
        "contributors": active
        / "exhaustive_wizard_hyperliquid_concentration_contributors.csv",
        "manifest": active / "exhaustive_wizard_hyperliquid_concentration_manifest.json",
        "summary_md": active / "exhaustive_wizard_hyperliquid_concentration_summary.md",
        "snapshot_status": snapshot_dir / "concentration_status.csv",
        "snapshot_cohorts": snapshot_dir / "concentration_cohorts.csv",
        "snapshot_dimensions": snapshot_dir / "concentration_dimensions.csv",
        "snapshot_contributors": snapshot_dir / "concentration_contributors.csv",
        "snapshot_manifest": snapshot_dir / "manifest.json",
        "snapshot_summary_md": snapshot_dir / "summary.md",
    }
    for frame, active_path, snapshot_path in (
        (status_frame, paths["status"], paths["snapshot_status"]),
        (cohort_frame, paths["cohorts"], paths["snapshot_cohorts"]),
        (dimension_frame, paths["dimensions"], paths["snapshot_dimensions"]),
        (contributor_frame, paths["contributors"], paths["snapshot_contributors"]),
    ):
        frame.to_csv(active_path, index=False)
        frame.to_csv(snapshot_path, index=False)

    status_counts = _status_counts(status_frame, "concentration_status")
    summary: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "walkforward_id": walkforward_id,
        "regime_attribution_id": regime_attribution_id,
        "robustness_id": robustness_id,
        "concentration_id": concentration_id,
        "created_at": as_of.isoformat(),
        "experiments": int(len(status_frame)),
        "unique_experiment_ids": int(status_frame["experiment_id"].nunique()),
        "status_counts": status_counts,
        "experiment_status_accounted": bool(sum(status_counts.values()) == len(status_frame)),
        "practical_walkforward_candidates": len(
            cohort_ids["practical_walkforward_pass"]
        ),
        "research_robustness_candidates": len(
            cohort_ids["research_robustness_pass"]
        ),
        "statistically_selected_robustness_candidates": len(
            cohort_ids["statistically_selected_robustness_pass"]
        ),
        "cohort_rows": int(len(cohort_frame)),
        "dimension_rows": int(len(dimension_frame)),
        "contributor_rows": int(len(contributor_frame)),
        "promotion_concentration_pass": promotion_pass,
        "ready_for_leverage_gate": promotion_pass,
        "acceptance_eligible_replays": 0,
        "live_trading_authorized": False,
        "policy": policy,
        "input_hashes": material["input_hashes"],
        "artifacts": {name: _relative(path, root) for name, path in paths.items()},
        "input_snapshots": {
            name: _relative(path, root) for name, path in snapshot_inputs.items()
        },
    }
    manifest_text = json.dumps(summary, indent=2, sort_keys=True)
    summary_text = _summary_markdown(summary, cohort_frame, dimension_frame)
    paths["manifest"].write_text(manifest_text, encoding="utf-8")
    paths["snapshot_manifest"].write_text(manifest_text, encoding="utf-8")
    paths["summary_md"].write_text(summary_text, encoding="utf-8")
    paths["snapshot_summary_md"].write_text(summary_text, encoding="utf-8")
    return CommandResult(paths=paths, summary=summary)


def _evaluate_cohort(
    *,
    cohort: str,
    experiment_ids: set[str],
    candidates: pd.DataFrame,
    trades: pd.DataFrame,
    regimes: pd.DataFrame,
    concentration_id: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    candidate_subset = candidates.loc[
        candidates.get("experiment_id", pd.Series("", index=candidates.index))
        .astype(str)
        .isin(experiment_ids)
    ].copy()
    trade_subset = trades.loc[
        trades.get("experiment_id", pd.Series("", index=trades.index))
        .astype(str)
        .isin(experiment_ids)
    ].copy()
    if not trade_subset.empty and "is_closed" in trade_subset.columns:
        trade_subset = trade_subset.loc[trade_subset["is_closed"].map(_truthy)].copy()
    if not trade_subset.empty:
        trade_subset["canonical_pair"] = trade_subset.apply(_canonical_pair, axis=1)
        trade_subset["_net_profit"] = pd.to_numeric(
            trade_subset.get("profit_after_cost"), errors="coerce"
        ).fillna(0.0)
        trade_subset["_trade_count"] = 1

    regime_subset = regimes.loc[
        regimes.get("experiment_id", pd.Series("", index=regimes.index))
        .astype(str)
        .isin(experiment_ids)
    ].copy()
    if not regime_subset.empty:
        regime_subset["_net_profit"] = pd.to_numeric(
            regime_subset.get("total_trade_profit_after_cost"), errors="coerce"
        ).fillna(0.0)
        regime_subset["_trade_count"] = pd.to_numeric(
            regime_subset.get("closed_trades"), errors="coerce"
        ).fillna(0).astype(int)

    dimension_rows: list[dict[str, object]] = []
    contributor_rows: list[dict[str, object]] = []
    for dimension, thresholds in DIMENSION_POLICIES.items():
        source = regime_subset if dimension == "regime" else trade_subset
        grouped = _group_contributions(source, dimension=dimension)
        summary, contributors = _concentration_metrics(
            grouped,
            cohort=cohort,
            dimension=dimension,
            candidate_count=len(candidate_subset),
            concentration_id=concentration_id,
            thresholds=thresholds,
        )
        dimension_rows.append(summary)
        contributor_rows.extend(contributors)
    return dimension_rows, contributor_rows


def _group_contributions(frame: pd.DataFrame, *, dimension: str) -> pd.DataFrame:
    columns = ["dimension_value", "candidate_count", "trade_count", "net_profit"]
    if frame.empty or dimension not in frame.columns:
        return pd.DataFrame(columns=columns)
    usable = frame.loc[frame[dimension].fillna("").astype(str).ne("")].copy()
    if usable.empty:
        return pd.DataFrame(columns=columns)
    grouped = (
        usable.groupby(dimension, dropna=False)
        .agg(
            candidate_count=("experiment_id", "nunique"),
            trade_count=("_trade_count", "sum"),
            net_profit=("_net_profit", "sum"),
        )
        .reset_index()
        .rename(columns={dimension: "dimension_value"})
    )
    return grouped


def _concentration_metrics(
    grouped: pd.DataFrame,
    *,
    cohort: str,
    dimension: str,
    candidate_count: int,
    concentration_id: str,
    thresholds: dict[str, float | int],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    values = len(grouped)
    total_trades = int(grouped["trade_count"].sum()) if not grouped.empty else 0
    total_net_profit = float(grouped["net_profit"].sum()) if not grouped.empty else 0.0
    positive = grouped["net_profit"].clip(lower=0.0) if not grouped.empty else pd.Series(dtype=float)
    losses = -grouped["net_profit"].clip(upper=0.0) if not grouped.empty else pd.Series(dtype=float)
    total_positive = float(positive.sum())
    total_losses = float(losses.sum())
    profit_shares = positive / total_positive if total_positive > 0.0 else positive * 0.0
    trade_shares = (
        grouped["trade_count"] / total_trades
        if total_trades > 0 and not grouped.empty
        else pd.Series(0.0, index=grouped.index)
    )
    profit_hhi = float((profit_shares**2).sum()) if total_positive > 0.0 else None
    trade_hhi = float((trade_shares**2).sum()) if total_trades > 0 else None
    effective_profit = 1.0 / profit_hhi if profit_hhi and profit_hhi > 0.0 else 0.0
    effective_trades = 1.0 / trade_hhi if trade_hhi and trade_hhi > 0.0 else 0.0
    max_profit_share = float(profit_shares.max()) if total_positive > 0.0 else 0.0
    max_trade_share = float(trade_shares.max()) if total_trades > 0 else 0.0
    top_profit_value = (
        _text(grouped.loc[profit_shares.idxmax(), "dimension_value"])
        if total_positive > 0.0
        else ""
    )
    top_trade_value = (
        _text(grouped.loc[trade_shares.idxmax(), "dimension_value"])
        if total_trades > 0
        else ""
    )
    blockers: list[str] = []
    minimum_values = int(thresholds["minimum_values"])
    maximum_profit_share = float(thresholds["maximum_positive_profit_share"])
    maximum_trade_share = float(thresholds["maximum_trade_share"])
    minimum_effective_profit = float(thresholds["minimum_effective_profit_breadth"])
    if candidate_count == 0:
        blockers.append("empty_cohort")
    if values < minimum_values:
        blockers.append(f"values_observed<{minimum_values}")
    if total_trades <= 0:
        blockers.append("closed_trade_evidence_missing")
    if total_positive <= 0.0:
        blockers.append("positive_group_net_profit_missing")
    if max_profit_share > maximum_profit_share:
        blockers.append(f"max_positive_profit_share>{maximum_profit_share:g}")
    if max_trade_share > maximum_trade_share:
        blockers.append(f"max_trade_share>{maximum_trade_share:g}")
    if effective_profit < minimum_effective_profit:
        blockers.append(f"effective_profit_breadth<{minimum_effective_profit:g}")
    gate_pass = not blockers
    summary = {
        "schema_version": SCHEMA_VERSION,
        "concentration_id": concentration_id,
        "cohort": cohort,
        "dimension": dimension,
        "candidate_count": candidate_count,
        "values_observed": values,
        "profitable_values": int((positive > 0.0).sum()),
        "total_closed_trades": total_trades,
        "total_net_profit_after_cost": total_net_profit,
        "total_positive_group_profit": total_positive,
        "total_loss_magnitude": total_losses,
        "maximum_positive_profit_share": max_profit_share,
        "profit_hhi": profit_hhi,
        "effective_profit_breadth": effective_profit,
        "maximum_trade_share": max_trade_share,
        "trade_hhi": trade_hhi,
        "effective_trade_breadth": effective_trades,
        "top_profit_contributor": top_profit_value,
        "top_trade_contributor": top_trade_value,
        "policy_minimum_values": minimum_values,
        "policy_maximum_positive_profit_share": maximum_profit_share,
        "policy_maximum_trade_share": maximum_trade_share,
        "policy_minimum_effective_profit_breadth": minimum_effective_profit,
        "dimension_gate_pass": gate_pass,
        "dimension_status": (
            "PASS_RESEARCH_CONCENTRATION_DIMENSION"
            if gate_pass
            else "FAIL_RESEARCH_CONCENTRATION_DIMENSION"
        ),
        "dimension_blocker": ";".join(blockers),
        "acceptance_status": "BLOCKED",
        "acceptance_reason": RESEARCH_ONLY_REASON,
        "live_trading_authorized": False,
    }
    contributors: list[dict[str, object]] = []
    for index, row in grouped.iterrows():
        contributors.append(
            {
                "schema_version": SCHEMA_VERSION,
                "concentration_id": concentration_id,
                "cohort": cohort,
                "dimension": dimension,
                "dimension_value": _text(row["dimension_value"]),
                "candidate_count": int(row["candidate_count"]),
                "trade_count": int(row["trade_count"]),
                "net_profit_after_cost": float(row["net_profit"]),
                "positive_group_profit": float(positive.loc[index]),
                "loss_magnitude": float(losses.loc[index]),
                "positive_profit_share": float(profit_shares.loc[index]),
                "trade_share": float(trade_shares.loc[index]),
                "acceptance_status": "BLOCKED",
                "live_trading_authorized": False,
            }
        )
    return summary, contributors


def _status_frame(
    statuses: pd.DataFrame,
    *,
    robust_ids: set[str],
    selected_ids: set[str],
    concentration_id: str,
    promotion_pass: bool,
    promotion_blocker: str,
    evidence_paths: object,
    root: Path,
) -> pd.DataFrame:
    evidence = ";".join(_relative(Path(path), root) for path in evidence_paths)
    rows: list[dict[str, object]] = []
    for record in statuses.to_dict("records"):
        experiment_id = _text(record.get("experiment_id"))
        if experiment_id in selected_ids:
            status = (
                "PASS_CROSS_CELL_CONCENTRATION"
                if promotion_pass
                else "FAIL_CROSS_CELL_CONCENTRATION"
            )
            blocker = "" if promotion_pass else promotion_blocker
        elif experiment_id in robust_ids:
            status = "DIAGNOSTIC_ONLY_STATISTICAL_SELECTION_BLOCKED"
            blocker = "statistical_selection_not_passed"
        else:
            status = "NOT_SELECTED_PRIOR_RESEARCH_GATES"
            blocker = (
                _text(record.get("robustness_blocker"))
                or _text(record.get("robustness_status"))
            )
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "concentration_id": concentration_id,
                "experiment_id": experiment_id,
                "pair_group_id": _text(record.get("pair_group_id")),
                "pair": _text(record.get("pair")),
                "wizard_exchange": _text(record.get("wizard_exchange")),
                "wizard_timeframe": _text(record.get("wizard_timeframe")),
                "hyperliquid_interval": _text(record.get("hyperliquid_interval")),
                "exact_mode": _text(record.get("exact_mode")),
                "orientation": _text(record.get("orientation")),
                "prior_robustness_status": _text(record.get("robustness_status")),
                "prior_statistical_selection_status": _text(
                    record.get("statistical_selection_status")
                ),
                "concentration_status": status,
                "concentration_blocker": blocker,
                "concentration_gate_pass": bool(
                    experiment_id in selected_ids and promotion_pass
                ),
                "ready_for_leverage_gate": bool(
                    experiment_id in selected_ids and promotion_pass
                ),
                "acceptance_status": "BLOCKED",
                "acceptance_reason": RESEARCH_ONLY_REASON,
                "acceptance_eligible": False,
                "evidence_path": evidence,
                "live_trading_authorized": False,
            }
        )
    return pd.DataFrame(rows)


def _canonical_pair(row: pd.Series) -> str:
    assets = sorted(
        {
            _text(row.get("asset_x")).upper(),
            _text(row.get("asset_y")).upper(),
        }
        - {""}
    )
    return "/".join(assets) if len(assets) == 2 else _text(row.get("pair"))


def _id_set(frame: pd.DataFrame, mask: pd.Series) -> set[str]:
    if frame.empty or "experiment_id" not in frame.columns:
        return set()
    return set(frame.loc[mask.fillna(False), "experiment_id"].astype(str))


def _summary_markdown(
    summary: dict[str, object],
    cohorts: pd.DataFrame,
    dimensions: pd.DataFrame,
) -> str:
    return "\n".join(
        [
            "# Exhaustive Wizard To Hyperliquid Cross-Cell Concentration",
            "",
            "This gate tests whether apparent after-cost gains and trades are dominated by a narrow pair, timeframe, Wizard venue, exact mode, orientation, or causal regime.",
            "Practical and robustness cohorts are diagnostics. Only a non-empty statistically selected robustness cohort can pass the promotion concentration gate.",
            "This artifact never authorizes acceptance, Testnet submission, leverage, or live trading.",
            "",
            "## Summary",
            "",
            pd.DataFrame(
                [
                    {"metric": "concentration_id", "value": summary["concentration_id"]},
                    {"metric": "experiments", "value": summary["experiments"]},
                    {
                        "metric": "practical_walkforward_candidates",
                        "value": summary["practical_walkforward_candidates"],
                    },
                    {
                        "metric": "research_robustness_candidates",
                        "value": summary["research_robustness_candidates"],
                    },
                    {
                        "metric": "statistically_selected_robustness_candidates",
                        "value": summary[
                            "statistically_selected_robustness_candidates"
                        ],
                    },
                    {
                        "metric": "promotion_concentration_pass",
                        "value": summary["promotion_concentration_pass"],
                    },
                    {"metric": "live_trading_authorized", "value": False},
                ]
            ).to_markdown(index=False),
            "",
            "## Cohorts",
            "",
            cohorts.to_markdown(index=False),
            "",
            "## Dimensions",
            "",
            dimensions.to_markdown(index=False),
            "",
        ]
    )


def _require_unique(frame: pd.DataFrame, column: str) -> None:
    if frame.empty:
        return
    if column not in frame.columns or frame[column].astype(str).duplicated().any():
        raise ValueError(f"Concentration input requires unique {column}")


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, keep_default_na=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _status_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    return {
        _text(key): int(value)
        for key, value in frame[column].value_counts(dropna=False).sort_index().items()
    }


def _file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _deduplicate(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return _text(value).lower() in {"1", "true", "yes", "y"}


def _text(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
