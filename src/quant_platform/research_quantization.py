from __future__ import annotations

from pathlib import Path

import pandas as pd


def quantize_family_matrix(
    family_matrix_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    top_n: int = 10,
) -> dict[str, Path]:
    base = Path(family_matrix_dir)
    output = Path(output_dir) if output_dir is not None else base / "quantized"
    output.mkdir(parents=True, exist_ok=True)

    separate = _read_csv_or_empty(base / "family_separate_summary.csv")
    combos = _read_csv_or_empty(base / "family_combo_summary.csv")

    candidates = _candidate_frame(separate, combos)
    ranked = _rank_candidates(candidates)
    top = ranked.head(max(1, top_n)).copy()

    ranked_path = output / "research_quantization_ranked.csv"
    top_path = output / "research_quantization_top.csv"
    summary_path = output / "research_quantization_summary.csv"
    notes_path = output / "research_quantization_runbook.md"

    ranked.to_csv(ranked_path, index=False)
    top.to_csv(top_path, index=False)
    _summary_frame(ranked, top_n=top_n).to_csv(summary_path, index=False)
    notes_path.write_text(_runbook_text(base, ranked, top), encoding="utf-8")

    return {
        "ranked": ranked_path,
        "top": top_path,
        "summary": summary_path,
        "runbook": notes_path,
    }


def _read_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError, UnicodeDecodeError):
        return pd.DataFrame()


def _candidate_frame(separate: pd.DataFrame, combos: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if not separate.empty:
        for _, row in separate.iterrows():
            rows.append(
                {
                    "candidate_type": "family",
                    "candidate_name": str(row.get("family", "")),
                    "families": str(row.get("family", "")),
                    "strategy_name": str(row.get("best_strategy_name", "")),
                    "strategy_ids": str(row.get("best_strategy_id", "")),
                    "combo_size": 1,
                    "production_eligible": _bool_value(row.get("production_eligible", False)),
                    "preferred_eligible": _bool_value(row.get("preferred_eligible", False)),
                    "acceptance_reason": str(row.get("acceptance_reason", "")),
                    "preferred_reason": str(row.get("preferred_reason", "")),
                    "passing_pairs": _numeric_int(row.get("passing_pairs")),
                    "total_trades": _numeric_int(row.get("total_trades")),
                    "median_profit_factor": _numeric_float(row.get("median_profit_factor")),
                    "median_sharpe": _numeric_float(row.get("median_sharpe")),
                    "worst_drawdown": _numeric_float(row.get("worst_drawdown")),
                }
            )
    if not combos.empty:
        for _, row in combos.iterrows():
            rows.append(
                {
                    "candidate_type": "combo",
                    "candidate_name": str(row.get("combo_name", "")),
                    "families": str(row.get("families", "")),
                    "strategy_name": str(row.get("strategies", "")),
                    "strategy_ids": str(row.get("strategy_ids", "")),
                    "combo_size": _numeric_int(row.get("combo_size"), default=2),
                    "production_eligible": _bool_value(row.get("production_eligible", False)),
                    "preferred_eligible": _bool_value(row.get("preferred_eligible", False)),
                    "acceptance_reason": str(row.get("acceptance_reason", "")),
                    "preferred_reason": str(row.get("preferred_reason", "")),
                    "passing_pairs": _numeric_int(row.get("passing_pairs")),
                    "total_trades": _numeric_int(row.get("total_trades")),
                    "median_profit_factor": _numeric_float(row.get("median_profit_factor")),
                    "median_sharpe": _numeric_float(row.get("median_sharpe")),
                    "worst_drawdown": _numeric_float(row.get("worst_drawdown")),
                }
            )
    return pd.DataFrame(rows)


def _numeric_float(value: object, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if pd.isna(numeric):
        return default
    return numeric


def _numeric_int(value: object, default: int = 0) -> int:
    return int(round(_numeric_float(value, default=float(default))))


def _bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def _rank_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    ranked = frame.copy()
    ranked["gate_score"] = ranked.apply(_gate_score, axis=1)
    ranked["pair_score"] = ranked["passing_pairs"].clip(lower=0, upper=2) / 2.0 * 20.0
    ranked["sharpe_score"] = ranked["median_sharpe"].clip(lower=0.0, upper=1.5) / 1.5 * 20.0
    ranked["profit_factor_score"] = (ranked["median_profit_factor"] - 1.0).clip(lower=0.0, upper=1.0) * 15.0
    ranked["drawdown_score"] = (1.0 - ranked["worst_drawdown"].clip(lower=0.0, upper=0.30) / 0.30) * 10.0
    ranked["trade_count_score"] = ranked["total_trades"].clip(lower=0, upper=250) / 250.0 * 10.0
    ranked["quant_score"] = (
        ranked["gate_score"]
        + ranked["pair_score"]
        + ranked["sharpe_score"]
        + ranked["profit_factor_score"]
        + ranked["drawdown_score"]
        + ranked["trade_count_score"]
    ).round(3)
    ranked["decision_bucket"] = ranked.apply(_decision_bucket, axis=1)
    ranked["decision_reason"] = ranked.apply(_decision_reason, axis=1)
    ranked = ranked.sort_values(
        [
            "production_eligible",
            "preferred_eligible",
            "quant_score",
            "passing_pairs",
            "median_sharpe",
            "median_profit_factor",
            "total_trades",
            "worst_drawdown",
        ],
        ascending=[False, False, False, False, False, False, False, True],
    ).reset_index(drop=True)
    ranked.insert(0, "quant_rank", range(1, len(ranked) + 1))
    return ranked


def _gate_score(row: pd.Series) -> float:
    if _bool_value(row.get("production_eligible", False)):
        return 25.0
    if _bool_value(row.get("preferred_eligible", False)):
        return 15.0
    passing_pairs = _numeric_int(row.get("passing_pairs"))
    if passing_pairs > 0:
        return 8.0
    return 0.0


def _decision_bucket(row: pd.Series) -> str:
    score = _numeric_float(row.get("quant_score"))
    pf = _numeric_float(row.get("median_profit_factor"))
    sharpe = _numeric_float(row.get("median_sharpe"))
    trades = _numeric_int(row.get("total_trades"))
    passing_pairs = _numeric_int(row.get("passing_pairs"))
    if _bool_value(row.get("production_eligible", False)):
        return "promote_now"
    if score >= 50 and passing_pairs >= 1 and sharpe >= 0.5:
        return "shadow_ready"
    if score >= 25 or (pf >= 1.2 and sharpe >= 0.2 and trades >= 100):
        return "watchlist"
    return "reject"


def _decision_reason(row: pd.Series) -> str:
    bucket = str(row.get("decision_bucket", ""))
    if bucket == "promote_now":
        return "production_eligible_under_current_gate"
    if bucket == "shadow_ready":
        return "score_supports_shadow_mode_but_gate_not_fully_cleared"
    if bucket == "watchlist":
        reason = str(row.get("acceptance_reason", "")).strip()
        return reason or "mixed_metrics_need_more_pair_depth"
    reason = str(row.get("acceptance_reason", "")).strip()
    return reason or "insufficient_risk_adjusted_evidence"


def _summary_frame(ranked: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
    if ranked.empty:
        return pd.DataFrame(
            [
                {
                    "candidates": 0,
                    "top_n": top_n,
                    "promote_now": 0,
                    "shadow_ready": 0,
                    "watchlist": 0,
                    "reject": 0,
                    "top_quant_score": 0.0,
                }
            ]
        )
    counts = ranked["decision_bucket"].value_counts()
    return pd.DataFrame(
        [
            {
                "candidates": int(len(ranked)),
                "top_n": int(top_n),
                "promote_now": int(counts.get("promote_now", 0)),
                "shadow_ready": int(counts.get("shadow_ready", 0)),
                "watchlist": int(counts.get("watchlist", 0)),
                "reject": int(counts.get("reject", 0)),
                "top_quant_score": float(ranked["quant_score"].max()),
            }
        ]
    )


def _runbook_text(base: Path, ranked: pd.DataFrame, top: pd.DataFrame) -> str:
    lines = [
        "# Research Quantization Runbook",
        "",
        f"- source_dir: `{base}`",
        f"- candidates_scored: {len(ranked)}",
        "",
        "## Decision Buckets",
        "",
        "- `promote_now`: already clears the current production gate.",
        "- `shadow_ready`: strong enough to track closely, but not enough to change behavior.",
        "- `watchlist`: worth keeping in the narrowed search.",
        "- `reject`: currently too weak under the present evidence.",
        "",
        "## Top Candidates",
        "",
    ]
    if top.empty:
        lines.append("- no candidates were available to score")
    else:
        for _, row in top.iterrows():
            lines.append(
                f"- rank {int(row['quant_rank'])}: `{row['candidate_name']}` "
                f"[{row['candidate_type']}] score={float(row['quant_score']):.2f} "
                f"bucket=`{row['decision_bucket']}` "
                f"(pf={float(row['median_profit_factor']):.3f}, sharpe={float(row['median_sharpe']):.3f}, "
                f"dd={float(row['worst_drawdown']):.3f}, trades={int(row['total_trades'])}, "
                f"passing_pairs={int(row['passing_pairs'])})"
            )
    return "\n".join(lines) + "\n"
