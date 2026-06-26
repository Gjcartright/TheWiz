from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import isfinite
from pathlib import Path
from typing import Iterable

import pandas as pd

from quant_platform.experiments import AcceptanceGate, ExperimentConfig, ExperimentHarness, PairDataset, strategy_acceptance_report
from quant_platform.feature_engine import FeatureEngine
from quant_platform.strategies import STRATEGIES, StrategySpec


@dataclass(frozen=True)
class FamilyRunArtifact:
    family: str
    results: pd.DataFrame
    acceptance: pd.DataFrame
    best_strategy_id: int | None
    best_strategy_name: str
    output_dir: Path


def strategy_family_registry(strategies: Iterable[StrategySpec] = STRATEGIES) -> pd.DataFrame:
    strategy_list = tuple(strategies)
    frame = pd.DataFrame(
        {
            "id": str(s.id),
            "name": s.name,
            "family": s.family,
            "hypothesis": s.hypothesis,
            "primary_fields": ";".join(s.primary_fields),
            "required_tests": ";".join(s.required_tests),
            "executable_signal": str(s.signal_function is not None),
        }
        for s in strategy_list
    )
    if frame.empty:
        return frame
    frame["id"] = pd.to_numeric(frame["id"], errors="coerce").fillna(0).astype(int)
    return frame.sort_values(["family", "id"]).reset_index(drop=True)


def run_family_matrix(
    datasets: Iterable[PairDataset],
    *,
    output_dir: str | Path,
    strategies: Iterable[StrategySpec] = STRATEGIES,
    gate: AcceptanceGate | None = None,
    feature_engine: FeatureEngine | None = None,
    max_combo_size: int = 4,
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    gate = gate or AcceptanceGate()
    feature_engine = feature_engine or FeatureEngine()
    strategy_list = tuple(strategies)

    registry = strategy_family_registry(strategy_list)
    registry_path = output / "family_registry.csv"
    registry.to_csv(registry_path, index=False)

    by_family: dict[str, list[StrategySpec]] = {}
    for strategy in strategy_list:
        by_family.setdefault(strategy.family, []).append(strategy)

    family_artifacts: list[FamilyRunArtifact] = []
    separate_rows: list[dict[str, object]] = []
    families_root = output / "families"
    families_root.mkdir(parents=True, exist_ok=True)

    for family in sorted(by_family):
        family_output = families_root / family
        family_output.mkdir(parents=True, exist_ok=True)
        harness = ExperimentHarness(
            strategies=by_family[family],
            config=ExperimentConfig(gate=gate),
            feature_engine=feature_engine,
        )
        results = harness.run(datasets)
        harness.write_reports(results, family_output)
        acceptance = strategy_acceptance_report(results, gate)
        acceptance_path = family_output / "family_acceptance_report.csv"
        acceptance.to_csv(acceptance_path, index=False)
        best_row = _best_acceptance_row(acceptance)
        artifact = FamilyRunArtifact(
            family=family,
            results=results,
            acceptance=acceptance,
            best_strategy_id=int(best_row["strategy_id"]) if best_row is not None else None,
            best_strategy_name=str(best_row["strategy_name"]) if best_row is not None else "",
            output_dir=family_output,
        )
        family_artifacts.append(artifact)
        separate_rows.append(
            {
                "family": family,
                "strategies_in_family": int(len(by_family[family])),
                "evaluated_rows": int(len(results)),
                "best_strategy_id": artifact.best_strategy_id if artifact.best_strategy_id is not None else "",
                "best_strategy_name": artifact.best_strategy_name,
                "production_eligible": bool(best_row["production_eligible"]) if best_row is not None else False,
                "preferred_eligible": bool(best_row["preferred_eligible"]) if best_row is not None else False,
                "acceptance_reason": str(best_row["acceptance_reason"]) if best_row is not None else "no_results",
                "preferred_reason": str(best_row["preferred_reason"]) if best_row is not None else "no_results",
                "passing_pairs": int(best_row["passing_pairs"]) if best_row is not None else 0,
                "total_trades": int(best_row["total_trades"]) if best_row is not None else 0,
                "median_profit_factor": _safe_metric_value(best_row["median_profit_factor"]) if best_row is not None else 0.0,
                "median_sharpe": _safe_metric_value(best_row["median_sharpe"]) if best_row is not None else 0.0,
                "worst_drawdown": _safe_metric_value(best_row["worst_drawdown"]) if best_row is not None else 0.0,
                "family_output_dir": str(family_output),
            }
        )

    separate_summary = pd.DataFrame(separate_rows).sort_values(
        [
            "production_eligible",
            "preferred_eligible",
            "passing_pairs",
            "median_sharpe",
            "median_profit_factor",
            "total_trades",
            "worst_drawdown",
        ],
        ascending=[False, False, False, False, False, False, True],
    )
    separate_summary_path = output / "family_separate_summary.csv"
    separate_summary.to_csv(separate_summary_path, index=False)

    best_frame = separate_summary[
        [
            "family",
            "best_strategy_id",
            "best_strategy_name",
            "production_eligible",
            "preferred_eligible",
            "acceptance_reason",
            "preferred_reason",
            "passing_pairs",
            "total_trades",
            "median_profit_factor",
            "median_sharpe",
            "worst_drawdown",
        ]
    ].copy()
    best_frame_path = output / "family_best_strategies.csv"
    best_frame.to_csv(best_frame_path, index=False)

    combo_summary, combo_pair_summary, combo_detail = _build_combo_reports(
        family_artifacts,
        gate,
        max_combo_size=max_combo_size,
    )
    combo_summary_path = output / "family_combo_summary.csv"
    combo_pair_summary_path = output / "family_combo_pair_summary.csv"
    combo_detail_path = output / "family_combo_detail.csv"
    combo_summary.to_csv(combo_summary_path, index=False)
    combo_pair_summary.to_csv(combo_pair_summary_path, index=False)
    combo_detail.to_csv(combo_detail_path, index=False)

    runbook_path = output / "family_matrix_runbook.md"
    runbook_path.write_text(
        _family_matrix_runbook(separate_summary, combo_summary, max_combo_size=max_combo_size),
        encoding="utf-8",
    )

    return {
        "registry": registry_path,
        "separate_summary": separate_summary_path,
        "best_strategies": best_frame_path,
        "combo_summary": combo_summary_path,
        "combo_pair_summary": combo_pair_summary_path,
        "combo_detail": combo_detail_path,
        "runbook": runbook_path,
    }


def _best_acceptance_row(frame: pd.DataFrame) -> pd.Series | None:
    if frame.empty:
        return None
    ranked = frame.sort_values(
        [
            "production_eligible",
            "preferred_eligible",
            "passing_pairs",
            "median_sharpe",
            "median_profit_factor",
            "total_trades",
            "worst_drawdown",
        ],
        ascending=[False, False, False, False, False, False, True],
    ).reset_index(drop=True)
    return ranked.iloc[0]


def _safe_metric_value(value: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return numeric if isfinite(numeric) else 0.0


def _build_combo_reports(
    family_artifacts: list[FamilyRunArtifact],
    gate: AcceptanceGate,
    *,
    max_combo_size: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    valid = [artifact for artifact in family_artifacts if artifact.best_strategy_id is not None]
    summary_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    detail_frames: list[pd.DataFrame] = []
    combo_counter = 1

    upper = min(len(valid), max(2, max_combo_size))
    for combo_size in range(2, upper + 1):
        for members in combinations(valid, combo_size):
            summary, pair_summary, detail = _evaluate_combo_members(
                members,
                gate,
                combo_counter,
                label_prefix=f"combo_{combo_size}way",
            )
            summary_rows.append(summary)
            pair_rows.extend(pair_summary)
            detail_frames.append(detail)
            combo_counter += 1

    if len(valid) >= 2 and len(valid) > upper:
        summary, pair_summary, detail = _evaluate_combo_members(tuple(valid), gate, combo_counter, label_prefix="full_stack")
        summary_rows.append(summary)
        pair_rows.extend(pair_summary)
        detail_frames.append(detail)

    summary_frame = pd.DataFrame(summary_rows).sort_values(
        [
            "production_eligible",
            "preferred_eligible",
            "passing_pairs",
            "median_sharpe",
            "median_profit_factor",
            "total_trades",
            "worst_drawdown",
        ],
        ascending=[False, False, False, False, False, False, True],
    ) if summary_rows else pd.DataFrame()
    pair_frame = pd.DataFrame(pair_rows).sort_values(["combo_name", "pair"]) if pair_rows else pd.DataFrame()
    detail_frame = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    return summary_frame, pair_frame, detail_frame


def _evaluate_combo_members(
    members: tuple[FamilyRunArtifact, ...],
    gate: AcceptanceGate,
    combo_counter: int,
    label_prefix: str = "pair_combo",
) -> tuple[dict[str, object], list[dict[str, object]], pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    families: list[str] = []
    strategies: list[str] = []
    strategy_ids: list[int] = []
    for artifact in members:
        family_rows = artifact.results[artifact.results["strategy_id"] == artifact.best_strategy_id].copy()
        if family_rows.empty:
            continue
        family_rows["component_family"] = artifact.family
        family_rows["component_strategy_name"] = artifact.best_strategy_name
        frames.append(family_rows)
        families.append(artifact.family)
        strategies.append(artifact.best_strategy_name)
        strategy_ids.append(int(artifact.best_strategy_id))
    if not frames:
        empty = {
            "combo_name": f"{label_prefix}_{combo_counter:02d}",
            "combo_size": 0,
            "families": "",
            "strategies": "",
            "production_eligible": False,
            "preferred_eligible": False,
            "acceptance_reason": "no_component_rows",
            "preferred_reason": "no_component_rows",
            "passing_pairs": 0,
            "total_trades": 0,
            "median_profit_factor": 0.0,
            "median_sharpe": 0.0,
            "worst_drawdown": 0.0,
        }
        return empty, [], pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combo_name = f"{label_prefix}_{combo_counter:02d}"
    combined["combo_name"] = combo_name
    combined["strategy_id"] = 800000 + combo_counter
    combined["strategy_name"] = f"{combo_name}:{' + '.join(families)}"
    combined["family"] = "family_combo"

    decision = gate.evaluate_strategy(combined)
    pair_rows: list[dict[str, object]] = []
    evaluated = combined[(combined["status"] == "evaluated")].copy()
    for pair, group in evaluated.groupby("pair"):
        pair_rows.append(
            {
                "combo_name": combo_name,
                "pair": pair,
                "families": ";".join(families),
                "strategies": ";".join(strategies),
                "component_count": int(len(members)),
                "total_trades": int(group["trades"].sum()),
                "median_profit_factor": _safe_metric_value(group["profit_factor"].median()),
                "median_sharpe": _safe_metric_value(group["sharpe"].median()),
                "worst_drawdown": _safe_metric_value(group["max_drawdown"].max()),
                "eligible_rows": int(group["eligible"].sum()),
            }
        )

    summary = {
        "combo_name": combo_name,
        "combo_size": int(len(members)),
        "families": ";".join(families),
        "strategies": ";".join(strategies),
        "strategy_ids": ";".join(str(item) for item in strategy_ids),
        **decision,
    }
    for field in ("median_profit_factor", "median_sharpe", "worst_drawdown"):
        summary[field] = _safe_metric_value(summary.get(field))
    return summary, pair_rows, combined


def _family_matrix_runbook(
    separate_summary: pd.DataFrame,
    combo_summary: pd.DataFrame,
    *,
    max_combo_size: int = 4,
) -> str:
    lines = [
        "# Family Matrix Runbook",
        "",
        "This workflow runs every family separately, then builds combo tests from the best strategy in each family.",
        "",
        "## What you get",
        "",
        "- a separate report folder for each family",
        "- a best-strategy row for each family",
        f"- family-combo tests from size 2 through size {max_combo_size}",
        "- one full-stack combo using all best families when the family count is larger than the max combo size",
        "",
        "## Separate Family Ranking",
        "",
    ]
    for _, row in separate_summary.iterrows():
        lines.append(
            f"- `{row['family']}` -> `{row['best_strategy_name']}` "
            f"(passing_pairs={int(row['passing_pairs'])}, sharpe={float(row['median_sharpe']):.3f}, "
            f"pf={float(row['median_profit_factor']):.3f}, trades={int(row['total_trades'])})"
        )
    lines.extend(["", "## Top Combo Rows", ""])
    if combo_summary.empty:
        lines.append("- no combos were produced")
    else:
        for _, row in combo_summary.head(10).iterrows():
            lines.append(
                f"- `{row['combo_name']}` [{row['families']}] "
                f"(passing_pairs={int(row['passing_pairs'])}, sharpe={float(row['median_sharpe']):.3f}, "
                f"pf={float(row['median_profit_factor']):.3f}, trades={int(row['total_trades'])})"
            )
        lines.extend(["", "## Combo Sizes", ""])
        size_counts = combo_summary.groupby("combo_size").size()
        for size, count in size_counts.items():
            lines.append(f"- `{int(size)}`-way combos: {int(count)}")
    lines.append("")
    return "\n".join(lines)
