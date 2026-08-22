"""Research-only side-by-side replay for captured Crypto Wizards modes."""

from __future__ import annotations

from quant_platform.orchestration.corrective_runtime import atomic_write_text

from quant_platform.orchestration.corrective_runtime import atomic_write_csv

from dataclasses import asdict
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from quant_platform.active_pipeline import CommandResult
from quant_platform.backtest import BacktestResult, CostModel, backtest_two_leg_spread
from quant_platform.wizard_candidate_set import wizard_hyperliquid_candidate_set_id
from quant_platform.wizard_mode_replay import build_local_mode_signal, normalize_exact_mode


ROOT = Path(__file__).resolve().parents[2]

COMPARISON_COLUMNS = [
    "comparison_group_id",
    "candidate_set_id",
    "pair",
    "asset_x",
    "asset_y",
    "exchange",
    "interval",
    "period",
    "exact_mode",
    "captured_modes_in_group",
    "history_path",
    "history_exchange",
    "history_match_status",
    "source_history_rows",
    "history_rows",
    "history_last_timestamp",
    "comparison_window",
    "sample_parity_status",
    "mode_replay_status",
    "mode_fidelity_status",
    "mode_fidelity_reason",
    "mode_metric_name",
    "mode_missing_inputs",
    "mode_computation_notes",
    "cost_case",
    "cost_assumption_authority",
    "pnl_weighting_source",
    "intended_weight_x",
    "intended_weight_y",
    "realized_weight_x",
    "realized_weight_y",
    "funding_coverage_pct",
    "trades",
    "closed_trades",
    "open_trades",
    "profit_factor",
    "sharpe",
    "sharpe_status",
    "max_drawdown",
    "expectancy",
    "win_rate",
    "total_return",
    "gross_return",
    "total_fees",
    "total_slippage",
    "total_funding",
    "total_execution_risk",
    "total_partial_fill_cost",
    "open_trade_policy",
    "forced_close_total_return",
    "forced_close_max_drawdown",
    "conservative_total_return",
    "conservative_max_drawdown",
    "acceptance_eligible",
    "paper_or_execution_eligible",
    "promotion_allowed",
    "comparison_validity",
    "validity_blocker",
    "training_eligible",
    "next_step",
    "settings_capture_evidence_path",
    "evidence_path",
]


def build_wizard_mode_comparison(root: Path = ROOT) -> CommandResult:
    """Compare all valid captured modes on common local history.

    The output is deliberately research-only. It uses provisional cost cases to
    make mode differences inspectable while separate venue economics and vendor
    custom-series proof remain requirements for acceptance.
    """

    active = root / "reports" / "active"
    captures_path = active / "crypto_wizards_pair_page_capture_settings.csv"
    captures = _read_csv(captures_path)
    captures = _valid_captures(captures)
    captures = _bind_candidate_set_ids(
        captures,
        _read_csv(active / "hyperliquid_wizard_hypothesis_queue.csv"),
    )
    rows: list[dict[str, object]] = []
    for _, group in _capture_groups(captures):
        rows.extend(_comparison_rows_for_group(group, root=root, captures_path=captures_path))
    frame = pd.DataFrame(rows, columns=COMPARISON_COLUMNS)
    if not frame.empty:
        cost_rank = frame["cost_case"].map({"provisional_base": 1, "zero_cost_upper_bound": 2, "provisional_stress": 3}).fillna(9)
        frame = frame.assign(_cost_rank=cost_rank).sort_values(
            ["comparison_group_id", "exact_mode", "_cost_rank"],
            ascending=[True, True, True],
        ).drop(columns=["_cost_rank"])
    output = active / "wizard_mode_comparison.csv"
    summary = active / "wizard_mode_comparison.md"
    _write_csv(frame, output)
    _write_text(summary, _comparison_markdown(frame))
    ready = frame.get("mode_replay_status", pd.Series(dtype=object)).eq("READY_FOR_RESEARCH_REPLAY") if not frame.empty else pd.Series(dtype=bool)
    return CommandResult(
        paths={"wizard_mode_comparison": output, "wizard_mode_comparison_summary": summary},
        summary={
            "rows": int(len(frame)),
            "comparison_groups": int(frame.get("comparison_group_id", pd.Series(dtype=object)).nunique()) if not frame.empty else 0,
            "research_ready_rows": int(ready.sum()),
            "acceptance_eligible_rows": 0,
        },
    )


def _valid_captures(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    required = {"pair", "asset_x", "asset_y", "interval", "exact_mode"}
    if not required.issubset(frame.columns):
        return pd.DataFrame(columns=frame.columns)
    complete = frame.get("backtest_settings_complete", pd.Series(False, index=frame.index)).map(_as_bool)
    confirmed = frame.get("capture_confirmed", pd.Series(False, index=frame.index)).map(_as_bool)
    modes = frame["exact_mode"].map(normalize_exact_mode)
    return frame.loc[complete & confirmed & modes.ne("")].copy()


def _capture_groups(frame: pd.DataFrame) -> Iterable[tuple[tuple[str, ...], pd.DataFrame]]:
    if frame.empty:
        return []
    working = frame.copy()
    for column in ("pair", "asset_x", "asset_y", "exchange", "interval"):
        working[column] = working.get(column, pd.Series("", index=working.index)).map(_text)
    working["period"] = pd.to_numeric(working.get("period", pd.Series(index=working.index)), errors="coerce").fillna(0).astype(int).astype(str)
    group_columns = ["pair", "asset_x", "asset_y", "exchange", "interval", "period"]
    return working.groupby(group_columns, dropna=False, sort=True)


def _comparison_rows_for_group(
    group: pd.DataFrame,
    *,
    root: Path,
    captures_path: Path,
) -> list[dict[str, object]]:
    first = group.iloc[0]
    group_id = _group_id(first)
    captured_modes = sorted({normalize_exact_mode(value) for value in group["exact_mode"] if normalize_exact_mode(value)})
    history_path = _matching_history_path(root, first)
    if history_path is None:
        return [
            _blocked_row(
                row,
                group_id=group_id,
                captured_modes=captured_modes,
                captures_path=captures_path,
                blocker="matching_local_two_leg_history_missing",
                next_step="build_matching_point_in_time_history_before_mode_comparison",
                root=root,
            )
            for _, row in group.iterrows()
        ]
    try:
        history = _history_frame(history_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [
            _blocked_row(
                row,
                group_id=group_id,
                captured_modes=captured_modes,
                captures_path=captures_path,
                history_path=history_path,
                blocker=f"local_history_unusable:{type(exc).__name__}:{exc}",
                next_step="repair_matching_point_in_time_history_before_mode_comparison",
                root=root,
            )
            for _, row in group.iterrows()
        ]

    source_history_rows = int(len(history))
    rows: list[dict[str, object]] = []
    for _, capture in group.iterrows():
        try:
            comparison_history = _comparison_history(history, capture)
        except ValueError as exc:
            rows.append(
                _blocked_row(
                    capture,
                    group_id=group_id,
                    captured_modes=captured_modes,
                    captures_path=captures_path,
                    history_path=history_path,
                    history=history,
                    source_history_rows=source_history_rows,
                    blocker=str(exc),
                    next_step="capture_a_valid_period_and_collect_at_least_that_many_observations",
                    root=root,
                )
            )
            continue
        result = build_local_mode_signal(
            comparison_history,
            capture.to_dict(),
            exact_mode=capture.get("exact_mode", ""),
        )
        if result.mode_replay_status != "READY_FOR_RESEARCH_REPLAY":
            rows.append(
                _blocked_row(
                    capture,
                    group_id=group_id,
                    captured_modes=captured_modes,
                    captures_path=captures_path,
                    history_path=history_path,
                    history=comparison_history,
                    source_history_rows=source_history_rows,
                    blocker=";".join(result.missing_inputs),
                    next_step="repair_mode_specific_settings_or_history_and_reimport",
                    mode_result=result,
                    root=root,
                )
            )
            continue
        try:
            pnl_history = _history_with_captured_weights(comparison_history, capture)
        except ValueError as exc:
            rows.append(
                _blocked_row(
                    capture,
                    group_id=group_id,
                    captured_modes=captured_modes,
                    captures_path=captures_path,
                    history_path=history_path,
                    history=comparison_history,
                    source_history_rows=source_history_rows,
                    blocker=str(exc),
                    next_step="capture_and_validate_explicit_wizard_leg_weights",
                    mode_result=result,
                    root=root,
                )
            )
            continue
        for cost_case, model in _research_cost_cases(first.get("interval", "")).items():
            cost_history = _history_for_cost_case(pnl_history, cost_case)
            backtest = backtest_two_leg_spread(cost_history, result.signal, model)
            forced_close = backtest_two_leg_spread(
                cost_history,
                _force_close_signal(result.signal),
                model,
            )
            rows.append(
                _result_row(
                    capture,
                    group_id=group_id,
                    captured_modes=captured_modes,
                    captures_path=captures_path,
                    history_path=history_path,
                    history=pnl_history,
                    source_history_rows=source_history_rows,
                    root=root,
                    mode_result=result,
                    cost_case=cost_case,
                    backtest=backtest,
                    forced_close=forced_close,
                )
            )
    return rows


def _result_row(
    capture: pd.Series,
    *,
    group_id: str,
    captured_modes: list[str],
    captures_path: Path,
    history_path: Path,
    history: pd.DataFrame,
    source_history_rows: int,
    root: Path,
    mode_result: object,
    cost_case: str,
    backtest: BacktestResult,
    forced_close: BacktestResult,
) -> dict[str, object]:
    history_exchange = _history_exchange(history_path)
    history_match_status = _history_match_status(capture.get("exchange", ""), history_exchange)
    validity_blockers = [
        "local_formula_is_not_vendor_strategy_parity",
        "cost_assumptions_not_venue_validated",
    ]
    if "VENUE_MISMATCH" in history_match_status:
        validity_blockers.append("wizard_source_venue_differs_from_local_venue")
    funding_coverage = _funding_coverage_pct(history)
    if funding_coverage < 95.0:
        validity_blockers.append("realized_funding_coverage_below_95pct")
    backtest_values = asdict(backtest)
    # The report interval describes the captured Wizard configuration. The
    # backtest's inferred interval is diagnostic data and must not overwrite it.
    backtest_values.pop("interval", None)
    return {
        **_row_identity(capture, group_id=group_id, captured_modes=captured_modes, captures_path=captures_path),
        "history_path": _relative(history_path, root=root),
        "history_exchange": history_exchange,
        "history_match_status": history_match_status,
        "source_history_rows": source_history_rows,
        "history_rows": int(len(history)),
        "history_last_timestamp": history.index[-1].isoformat(),
        "comparison_window": "captured_wizard_period_observations",
        "sample_parity_status": _sample_parity_status(capture, history),
        "mode_replay_status": mode_result.mode_replay_status,
        "mode_fidelity_status": mode_result.mode_fidelity_status,
        "mode_fidelity_reason": mode_result.mode_fidelity_reason,
        "mode_metric_name": mode_result.metric_name,
        "mode_missing_inputs": "",
        "mode_computation_notes": ";".join(mode_result.computation_notes),
        "cost_case": cost_case,
        "cost_assumption_authority": "provisional_research_only_not_venue_validated",
        "pnl_weighting_source": history.attrs.get("pnl_weighting_source", ""),
        "intended_weight_x": history.attrs.get("intended_weight_x", ""),
        "intended_weight_y": history.attrs.get("intended_weight_y", ""),
        "realized_weight_x": history.attrs.get("realized_weight_x", ""),
        "realized_weight_y": history.attrs.get("realized_weight_y", ""),
        "funding_coverage_pct": funding_coverage,
        **backtest_values,
        "closed_trades": int(backtest.trades),
        "open_trade_policy": "report_mark_to_market_and_forced_close_use_conservative_result",
        "forced_close_total_return": forced_close.total_return,
        "forced_close_max_drawdown": forced_close.max_drawdown,
        "conservative_total_return": min(backtest.total_return, forced_close.total_return),
        "conservative_max_drawdown": max(backtest.max_drawdown, forced_close.max_drawdown),
        "acceptance_eligible": False,
        "paper_or_execution_eligible": False,
        "promotion_allowed": False,
        "comparison_validity": "RESEARCH_DIAGNOSTIC_ONLY",
        "validity_blocker": ";".join(validity_blockers),
        "training_eligible": False,
        "next_step": _result_next_step(history_match_status),
        "evidence_path": _evidence_path(capture, captures_path=captures_path, history_path=history_path),
    }


def _blocked_row(
    capture: pd.Series,
    *,
    group_id: str,
    captured_modes: list[str],
    captures_path: Path,
    blocker: str,
    next_step: str,
    history_path: Path | None = None,
    history: pd.DataFrame | None = None,
    source_history_rows: int | None = None,
    mode_result: object | None = None,
    root: Path | None = None,
) -> dict[str, object]:
    return {
        **_row_identity(capture, group_id=group_id, captured_modes=captured_modes, captures_path=captures_path),
        "history_path": _relative(history_path, root=root) if history_path is not None and root is not None else str(history_path or ""),
        "history_exchange": _history_exchange(history_path) if history_path is not None else "",
        "history_match_status": "MISSING_OR_UNUSABLE_LOCAL_HISTORY" if history_path is None else "LOCAL_HISTORY_UNUSABLE",
        "source_history_rows": source_history_rows if source_history_rows is not None else int(len(history)) if history is not None else 0,
        "history_rows": int(len(history)) if history is not None else 0,
        "history_last_timestamp": history.index[-1].isoformat() if history is not None and not history.empty else "",
        "comparison_window": "captured_wizard_period_observations",
        "sample_parity_status": "BLOCKED",
        "mode_replay_status": mode_result.mode_replay_status if mode_result is not None else "BLOCKED_HISTORY",
        "mode_fidelity_status": mode_result.mode_fidelity_status if mode_result is not None else "local_formula_approximation",
        "mode_fidelity_reason": mode_result.mode_fidelity_reason if mode_result is not None else "matching_local_history_required_for_local_formula_replay",
        "mode_metric_name": mode_result.metric_name if mode_result is not None else "",
        "mode_missing_inputs": ";".join(mode_result.missing_inputs) if mode_result is not None else blocker,
        "mode_computation_notes": ";".join(mode_result.computation_notes) if mode_result is not None else "",
        "cost_case": "",
        "cost_assumption_authority": "not_run",
        "pnl_weighting_source": "",
        "intended_weight_x": "",
        "intended_weight_y": "",
        "realized_weight_x": "",
        "realized_weight_y": "",
        "funding_coverage_pct": "",
        **_empty_backtest_values(),
        "closed_trades": 0,
        "open_trade_policy": "not_run",
        "forced_close_total_return": "",
        "forced_close_max_drawdown": "",
        "conservative_total_return": "",
        "conservative_max_drawdown": "",
        "acceptance_eligible": False,
        "paper_or_execution_eligible": False,
        "promotion_allowed": False,
        "comparison_validity": "INVALID",
        "validity_blocker": blocker,
        "training_eligible": False,
        "next_step": next_step,
        "evidence_path": _evidence_path(capture, captures_path=captures_path, history_path=history_path),
    }


def _row_identity(
    capture: pd.Series,
    *,
    group_id: str,
    captured_modes: list[str],
    captures_path: Path,
) -> dict[str, object]:
    return {
        "comparison_group_id": group_id,
        "candidate_set_id": _text(capture.get("_bound_candidate_set_id", "")),
        "pair": capture.get("pair", ""),
        "asset_x": capture.get("asset_x", ""),
        "asset_y": capture.get("asset_y", ""),
        "exchange": capture.get("exchange", ""),
        "interval": capture.get("interval", ""),
        "period": capture.get("period", ""),
        "exact_mode": normalize_exact_mode(capture.get("exact_mode", "")),
        "captured_modes_in_group": ";".join(captured_modes),
        "settings_capture_evidence_path": _text(capture.get("capture_evidence_path", "")),
    }


def _empty_backtest_values() -> dict[str, object]:
    values = {field: "" for field in asdict(BacktestResult(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))}
    values.pop("interval", None)
    return values


def _research_cost_cases(interval: object) -> dict[str, CostModel]:
    bars = _bars_per_day(interval)
    return {
        "zero_cost_upper_bound": CostModel(
            taker_fee_bps=0.0,
            slippage_bps=0.0,
            execution_risk_bps=0.0,
            funding_bps_per_day=0.0,
            bars_per_day=bars,
            partial_fill_probability=0.0,
            partial_fill_penalty_bps=0.0,
        ),
        "provisional_base": CostModel(bars_per_day=bars),
        "provisional_stress": CostModel(
            taker_fee_bps=7.5,
            slippage_bps=8.0,
            execution_risk_bps=4.0,
            funding_bps_per_day=3.0,
            bars_per_day=bars,
            partial_fill_probability=0.20,
            partial_fill_penalty_bps=4.0,
        ),
    }


def _comparison_history(history: pd.DataFrame, capture: pd.Series) -> pd.DataFrame:
    period = pd.to_numeric(pd.Series([capture.get("period", "")]), errors="coerce").iloc[0]
    if pd.isna(period) or int(period) < 2:
        raise ValueError("captured_wizard_period_missing_or_invalid")
    observations = int(period)
    if len(history) < observations:
        raise ValueError("local_history_shorter_than_captured_wizard_period")
    return history.tail(observations).copy()


def _history_with_captured_weights(history: pd.DataFrame, capture: pd.Series) -> pd.DataFrame:
    x = pd.to_numeric(pd.Series([capture.get("x_weighting", "")]), errors="coerce").iloc[0]
    y = pd.to_numeric(pd.Series([capture.get("y_weighting", "")]), errors="coerce").iloc[0]
    if pd.isna(x) or not 0.0 < float(x) < 1.0:
        raise ValueError("captured_x_weighting_missing_or_invalid")
    if pd.isna(y):
        y = 1.0 - float(x)
    if not 0.0 < float(y) < 1.0:
        raise ValueError("captured_y_weighting_missing_or_invalid")
    total = float(x) + float(y)
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("captured_weighting_total_invalid")
    normalized_x = float(x) / total
    normalized_y = float(y) / total
    ratio = normalized_x / normalized_y
    weighted = history.copy()
    weighted["hedge_ratio"] = ratio
    realized_x = ratio / (1.0 + ratio)
    realized_y = 1.0 / (1.0 + ratio)
    weighted.attrs.update(
        {
            "pnl_weighting_source": "captured_wizard_x_y_weighting",
            "intended_weight_x": normalized_x,
            "intended_weight_y": normalized_y,
            "realized_weight_x": realized_x,
            "realized_weight_y": realized_y,
        }
    )
    if max(abs(realized_x - normalized_x), abs(realized_y - normalized_y)) > 1e-12:
        raise ValueError("captured_weighting_reconciliation_failed")
    return weighted


def _funding_coverage_pct(history: pd.DataFrame) -> float:
    required = ("funding_x_bps", "funding_y_bps")
    if history.empty or not all(column in history.columns for column in required):
        return 0.0
    complete = pd.DataFrame(
        {column: pd.to_numeric(history[column], errors="coerce") for column in required}
    ).notna().all(axis=1)
    return float(complete.mean() * 100.0)


def _history_for_cost_case(history: pd.DataFrame, cost_case: str) -> pd.DataFrame:
    if cost_case != "zero_cost_upper_bound":
        return history
    zero_cost = history.copy()
    for column in ("funding_x_bps", "funding_y_bps"):
        if column in zero_cost.columns:
            zero_cost[column] = 0.0
    zero_cost.attrs.update(history.attrs)
    return zero_cost


def _force_close_signal(signal: pd.Series) -> pd.Series:
    forced = signal.copy()
    if not forced.empty:
        forced.iloc[-1] = 0.0
    return forced


def _sample_parity_status(capture: pd.Series, history: pd.DataFrame) -> str:
    period = pd.to_numeric(pd.Series([capture.get("period", "")]), errors="coerce").iloc[0]
    return "MATCHED_OBSERVATION_COUNT" if not pd.isna(period) and len(history) == int(period) else "MISMATCH"


def _bind_candidate_set_ids(captures: pd.DataFrame, queue: pd.DataFrame) -> pd.DataFrame:
    if captures.empty:
        return captures
    bound = captures.copy()
    fallback = wizard_hyperliquid_candidate_set_id(bound)
    by_config: dict[str, str] = {}
    if not queue.empty and {"candidate_config_hash", "candidate_set_id"}.issubset(queue.columns):
        by_config = {
            _text(row.get("candidate_config_hash", "")): _text(row.get("candidate_set_id", ""))
            for _, row in queue.iterrows()
            if _text(row.get("candidate_config_hash", "")) and _text(row.get("candidate_set_id", ""))
        }
    bound["_bound_candidate_set_id"] = bound.get(
        "candidate_config_hash", pd.Series("", index=bound.index)
    ).map(lambda value: by_config.get(_text(value), fallback))
    return bound


def _matching_history_path(root: Path, capture: pd.Series) -> Path | None:
    candidates: list[tuple[int, pd.Timestamp, Path]] = []
    expected_x = _asset_key(capture.get("asset_x", ""))
    expected_y = _asset_key(capture.get("asset_y", ""))
    expected_interval = _interval_key(capture.get("interval", ""))
    for path in sorted((root / "data" / "raw" / "pair_details").glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            history = payload.get("history", [])
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(history, list):
            continue
        if _asset_key(payload.get("asset_x", "")) != expected_x or _asset_key(payload.get("asset_y", "")) != expected_y:
            continue
        if expected_interval and _interval_key(payload.get("interval", "")) != expected_interval:
            continue
        frame = pd.DataFrame(history)
        if not {"timestamp", "price_x", "price_y"}.issubset(frame.columns):
            continue
        timestamp = _last_timestamp(frame.get("timestamp", pd.Series(dtype=object)))
        candidates.append((len(frame), timestamp, path))
    return sorted(candidates, key=lambda value: (value[0], value[1]), reverse=True)[0][2] if candidates else None


def _history_exchange(path: Path) -> str:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    return _text(payload.get("exchange", payload.get("venue", payload.get("source_exchange", "")))).lower()


def _history_match_status(capture_exchange: object, history_exchange: object) -> str:
    expected = _text(capture_exchange).lower()
    observed = _text(history_exchange).lower()
    if expected and observed and expected == observed:
        return "LOCAL_HISTORY_MATCHED_BY_LEGS_INTERVAL_AND_EXCHANGE_RESEARCH_ONLY"
    if expected and observed and expected != observed:
        return "LOCAL_HISTORY_LEGS_INTERVAL_VENUE_MISMATCH_RESEARCH_ONLY"
    if expected and not observed:
        return "LOCAL_HISTORY_LEGS_INTERVAL_VENUE_UNKNOWN_RESEARCH_ONLY"
    return "LOCAL_HISTORY_MATCHED_BY_LEGS_AND_INTERVAL_RESEARCH_ONLY"


def _result_next_step(history_match_status: str) -> str:
    if history_match_status == "LOCAL_HISTORY_LEGS_INTERVAL_VENUE_MISMATCH_RESEARCH_ONLY":
        return "compare_as_cross_venue_research_only_then_collect_same_venue_history_before_acceptance"
    if history_match_status == "LOCAL_HISTORY_LEGS_INTERVAL_VENUE_UNKNOWN_RESEARCH_ONLY":
        return "verify_history_venue_then_collect_same_venue_history_before_acceptance"
    return "compare_modes_for_research_then_require_vendor_parity_and_validated_venue_economics"


def _history_frame(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    history = pd.DataFrame(payload.get("history", []))
    if history.empty or not {"timestamp", "price_x", "price_y"}.issubset(history.columns):
        raise ValueError("history_requires_timestamp_price_x_price_y")
    history["timestamp"] = _timestamps(history["timestamp"])
    history = history.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp", drop=False)
    if history.empty:
        raise ValueError("history_has_no_parseable_timestamps")
    for column in history.columns:
        if column == "timestamp":
            continue
        numeric = pd.to_numeric(history[column], errors="coerce")
        # Keep textual metadata intact, but normalize columns that carry any
        # numeric observations used by mode replay or the two-leg backtest.
        if numeric.notna().any():
            history[column] = numeric
    return history


def _timestamps(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().mean() > 0.8:
        return pd.to_datetime(numeric, unit="s", utc=True, errors="coerce")
    return pd.to_datetime(values, utc=True, errors="coerce")


def _last_timestamp(values: pd.Series) -> pd.Timestamp:
    parsed = _timestamps(values)
    latest = parsed.max()
    return latest if pd.notna(latest) else pd.Timestamp.min.tz_localize("UTC")


def _bars_per_day(interval: object) -> int:
    key = _interval_key(interval)
    if key == "daily":
        return 1
    if key in {"hourly", "1hour", "1h"}:
        return 24
    if key in {"15mins", "15min", "15m"}:
        return 96
    if key in {"5mins", "5min", "5m"}:
        return 288
    return 24


def _group_id(capture: pd.Series) -> str:
    return "|".join(
        [
            _text(capture.get("pair", "")),
            _text(capture.get("exchange", "")),
            _interval_key(capture.get("interval", "")),
            _text(capture.get("period", "")),
        ]
    )


def _asset_key(value: object) -> str:
    text = _text(value).upper().replace("/", "-")
    for suffix in ("-USDT", "-USDC", "-USD", "-PERP"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
            break
    if "-" not in text:
        for suffix in ("USDT", "USDC", "PERP", "USD"):
            if text.endswith(suffix) and len(text) > len(suffix):
                text = text[: -len(suffix)]
                break
    return "".join(character for character in text if character.isalnum())


def _interval_key(value: object) -> str:
    text = _text(value).lower().replace(" ", "")
    aliases = {
        "daily": "daily",
        "1day": "daily",
        "day": "daily",
        "1d": "daily",
        "hourly": "hourly",
        "1hour": "hourly",
        "1h": "hourly",
        "15mins": "15min",
        "15min": "15min",
        "15m": "15min",
        "5mins": "5min",
        "5min": "5min",
        "5m": "5min",
    }
    return aliases.get(text, text)


def _evidence_path(capture: pd.Series, *, captures_path: Path, history_path: Path | None) -> str:
    values = [str(captures_path), _text(capture.get("capture_evidence_path", "")), str(history_path) if history_path else ""]
    return ";".join(dict.fromkeys(value for value in values if value))


def _comparison_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "# Wizard Mode Comparison\n\nNo valid, confirmed mode settings captures are available yet.\n"
    base = frame.loc[frame["cost_case"].eq("provisional_base")].copy()
    columns = [
        "pair",
        "exact_mode",
        "captured_modes_in_group",
        "mode_replay_status",
        "cost_case",
        "trades",
        "profit_factor",
        "sharpe",
        "max_drawdown",
        "total_return",
        "next_step",
    ]
    display = base[columns] if not base.empty else frame[columns]
    return "\n".join(
        [
            "# Wizard Mode Comparison",
            "",
            "This compares only modes with visible, confirmed settings against the same local two-leg history. All results use provisional research cost cases and cannot establish vendor parity, acceptance, paper readiness, or execution approval.",
            "",
            "## Provisional Base Comparison",
            "",
            display.to_markdown(index=False),
            "",
            "## Rules",
            "",
            "- Do not compare a local formula approximation to Wizard performance as if it were exact vendor parity.",
            "- A bounded vendor custom-series proof and validated venue economics remain required before acceptance.",
            "",
        ]
    )


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, dtype=object).fillna("") if path.exists() else pd.DataFrame()
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(frame, path, index=False)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, text, encoding="utf-8")


def _relative(path: Path, *, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _text(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return _text(value).lower() in {"1", "true", "yes", "y"}
