"""Causal regime attribution for exhaustive Hyperliquid walk-forward research."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import pandas as pd

from quant_platform.active_pipeline import CommandResult
from quant_platform.backtest import max_drawdown
from quant_platform.orchestration.corrective_redaction import safe_exception_code
from quant_platform.orchestration.corrective_runtime import (
    atomic_write_csv,
    atomic_write_text,
    immutable_snapshot_copy,
)
from quant_platform.orchestration.exhaustive_wizard_hyperliquid_canonical_replay import (
    _load_history,
)
from quant_platform.performance_math import calculate_annualized_sharpe

ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "exhaustive_wizard_hyperliquid_regime_attribution.v1"
VOLATILITY_WINDOW = 20
CORRELATION_WINDOW = 60
REGIME_REFERENCE_MIN_ROWS = 120
MIN_REGIME_TRADES = 3
MAX_REGIME_PROFIT_CONCENTRATION = 0.80
RESEARCH_ONLY_REASON = (
    "regime_attribution_is_research_only;statistical_selection_not_passed;"
    "current_l2_depth_is_point_in_time_not_historical_execution_evidence;"
    "mode_fidelity_parity_not_proven"
)

REGIME_DETAIL_COLUMNS = [
    "schema_version",
    "regime_attribution_id",
    "walkforward_id",
    "experiment_id",
    "pair_group_id",
    "pair",
    "wizard_exchange",
    "wizard_timeframe",
    "hyperliquid_interval",
    "exact_mode",
    "orientation",
    "regime",
    "bar_rows",
    "active_bar_rows",
    "closed_trades",
    "profit_factor",
    "expectancy",
    "win_rate",
    "conditional_bar_sharpe",
    "conditional_sharpe_status",
    "conditional_max_drawdown",
    "conditional_total_return",
    "total_trade_profit_after_cost",
    "total_fees",
    "total_slippage",
    "total_funding",
    "total_execution_risk",
    "total_partial_fill_cost",
    "acceptance_status",
    "acceptance_reason",
    "live_trading_authorized",
]


def build_exhaustive_wizard_hyperliquid_regime_attribution(
    *,
    root: Path = ROOT,
    now: datetime | None = None,
) -> CommandResult:
    """Attribute practical walk-forward survivors to causal market regimes."""

    as_of = _as_utc(now or datetime.now(timezone.utc))
    active = root / "reports" / "active"
    walkforward_manifest_path = (
        active / "exhaustive_wizard_hyperliquid_walkforward_manifest.json"
    )
    if not walkforward_manifest_path.exists():
        raise FileNotFoundError("Walk-forward manifest is required")
    walkforward_manifest = json.loads(
        walkforward_manifest_path.read_text(encoding="utf-8")
    )
    artifacts = walkforward_manifest.get("artifacts", {})
    snapshots = walkforward_manifest.get("input_snapshots", {})
    input_paths = {
        "walkforward_status": root / _text(artifacts.get("snapshot_status")),
        "walkforward_candidates": root / _text(artifacts.get("snapshot_candidates")),
        "walkforward_bars": root / _text(artifacts.get("snapshot_bars")),
        "walkforward_trades": root / _text(artifacts.get("snapshot_trades")),
        "pair_cost_evidence": root / _text(snapshots.get("pair_cost_evidence")),
        "walkforward_manifest": walkforward_manifest_path,
    }
    missing = [str(path) for path in input_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Regime-attribution inputs missing: {missing}")

    statuses = _read_csv(input_paths["walkforward_status"])
    candidates = _read_csv(input_paths["walkforward_candidates"])
    bars = _read_csv(input_paths["walkforward_bars"])
    trades = _read_csv(input_paths["walkforward_trades"])
    pair_costs = _read_csv(input_paths["pair_cost_evidence"])
    _require_unique(statuses, "experiment_id")
    _require_unique(pair_costs, "pair_group_id")

    run_id = _text(walkforward_manifest.get("run_id"))
    walkforward_id = _text(walkforward_manifest.get("walkforward_id"))
    observed_replay_id = _text(walkforward_manifest.get("observed_cost_replay_id"))
    policy = {
        "candidate_policy": "pass_research_walk_forward_only",
        "volatility_window": VOLATILITY_WINDOW,
        "correlation_window": CORRELATION_WINDOW,
        "reference_minimum_rows": REGIME_REFERENCE_MIN_ROWS,
        "volatility_threshold": "prior_only_expanding_75th_percentile_shifted_one_bar",
        "correlation_threshold": "prior_only_expanding_25th_percentile_shifted_one_bar",
        "regimes": ["crisis", "high_vol", "decorrelated", "calm"],
        "minimum_regime_trades": MIN_REGIME_TRADES,
        "maximum_regime_profit_concentration": MAX_REGIME_PROFIT_CONCENTRATION,
    }
    material = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "walkforward_id": walkforward_id,
        "as_of": as_of.isoformat(),
        "policy": policy,
        "input_hashes": {name: _file_hash(path) for name, path in input_paths.items()},
    }
    material_hash = sha256(_canonical_json(material).encode("utf-8")).hexdigest()
    attribution_id = f"hlregime_{as_of.strftime('%Y%m%dT%H%M%S%fZ')}_{material_hash[:8]}"
    walkforward_snapshot_manifest = root / _text(artifacts.get("snapshot_manifest"))
    snapshot_dir = (
        walkforward_snapshot_manifest.parent / "regime_attributions" / attribution_id
    )
    snapshot_input_dir = snapshot_dir / "inputs"
    snapshot_input_dir.mkdir(parents=True, exist_ok=True)
    snapshot_inputs: dict[str, Path] = {}
    for name, source in input_paths.items():
        target = immutable_snapshot_copy(
            source,
            snapshot_input_dir,
            artifact_name=name,
        )
        snapshot_inputs[name] = target

    paths = {
        "status": active / "exhaustive_wizard_hyperliquid_regime_status.csv",
        "candidates": active / "exhaustive_wizard_hyperliquid_regime_candidates.csv",
        "detail": active / "exhaustive_wizard_hyperliquid_regime_detail.csv",
        "bars": active / "exhaustive_wizard_hyperliquid_regime_bars.csv",
        "trades": active / "exhaustive_wizard_hyperliquid_regime_trades.csv",
        "manifest": active / "exhaustive_wizard_hyperliquid_regime_manifest.json",
        "summary_md": active / "exhaustive_wizard_hyperliquid_regime_summary.md",
        "snapshot_status": snapshot_dir / "regime_status.csv",
        "snapshot_candidates": snapshot_dir / "regime_candidates.csv",
        "snapshot_detail": snapshot_dir / "regime_detail.csv",
        "snapshot_bars": snapshot_dir / "regime_bars.csv",
        "snapshot_trades": snapshot_dir / "regime_trades.csv",
        "snapshot_manifest": snapshot_dir / "manifest.json",
        "snapshot_summary_md": snapshot_dir / "summary.md",
    }

    pair_lookup = _row_lookup(pair_costs, "pair_group_id")
    selected = candidates.loc[
        candidates["walkforward_status"].eq("PASS_RESEARCH_WALK_FORWARD")
    ].copy()
    selected_ids = set(selected["experiment_id"].astype(str))
    status_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    regime_bar_frames: list[pd.DataFrame] = []
    regime_trade_frames: list[pd.DataFrame] = []
    feature_cache: dict[str, pd.DataFrame] = {}

    for row in statuses.itertuples():
        base = _status_base(
            row,
            attribution_id=attribution_id,
            evidence_paths=snapshot_inputs.values(),
            root=root,
        )
        if _text(row.experiment_id) not in selected_ids:
            status_rows.append(
                {
                    **base,
                    "regime_status": "NOT_SELECTED_PRIOR_WALK_FORWARD_GATE",
                    "regime_blocker": (
                        _text(getattr(row, "walkforward_blocker", ""))
                        or _text(getattr(row, "walkforward_status", ""))
                    ),
                }
            )
            continue
        candidate = selected.loc[
            selected["experiment_id"].astype(str).eq(_text(row.experiment_id))
        ].iloc[0]
        pair_cost = pair_lookup.get(_text(candidate["pair_group_id"]))
        if pair_cost is None:
            status_rows.append(
                {
                    **base,
                    "regime_status": "BLOCKED_REGIME_INPUTS",
                    "regime_blocker": "pair_cost_evidence_missing",
                }
            )
            continue

        try:
            history_path = root / _text(pair_cost.enriched_history_path)
            cache_key = str(history_path)
            if cache_key not in feature_cache:
                feature_cache[cache_key] = build_causal_pair_regime_features(
                    _load_history(history_path)
                )
            features = feature_cache[cache_key]
            candidate_bars = bars.loc[
                bars["experiment_id"].astype(str).eq(_text(row.experiment_id))
            ].copy()
            candidate_trades = trades.loc[
                trades["experiment_id"].astype(str).eq(_text(row.experiment_id))
            ].copy()
            if candidate_bars.empty:
                raise ValueError("walkforward_bar_ledger_missing")
            enriched_bars = _join_regime(candidate_bars, features, timestamp_column="timestamp")
            enriched_trades = _join_regime(
                candidate_trades,
                features,
                timestamp_column="entry_timestamp",
            )
            details = _regime_details(
                candidate,
                enriched_bars,
                enriched_trades,
                attribution_id=attribution_id,
                walkforward_id=walkforward_id,
            )
            candidate_summary = _candidate_regime_summary(candidate, details)
        except Exception as exc:
            status_rows.append(
                {
                    **base,
                    "regime_status": "BLOCKED_REGIME_INPUTS",
                    "regime_blocker": f"{safe_exception_code(exc)}",
                }
            )
            continue

        result = {
            **base,
            **candidate_summary,
            "regime_status": "REGIME_ATTRIBUTION_COMPLETE",
            "regime_blocker": "",
            "acceptance_status": "BLOCKED",
            "acceptance_reason": _downstream_acceptance_reason(candidate),
            "acceptance_eligible": False,
            "live_trading_authorized": False,
        }
        status_rows.append(result)
        candidate_rows.append(result)
        detail_rows.extend(details)
        enriched_bars["schema_version"] = SCHEMA_VERSION
        enriched_bars["regime_attribution_id"] = attribution_id
        enriched_bars["walkforward_id"] = walkforward_id
        enriched_bars["live_trading_authorized"] = False
        regime_bar_frames.append(enriched_bars)
        enriched_trades["schema_version"] = SCHEMA_VERSION
        enriched_trades["regime_attribution_id"] = attribution_id
        enriched_trades["walkforward_id"] = walkforward_id
        enriched_trades["live_trading_authorized"] = False
        regime_trade_frames.append(enriched_trades)

    status = pd.DataFrame(status_rows)
    if len(status) != len(statuses) or status["experiment_id"].nunique() != len(statuses):
        raise ValueError("Regime attribution failed complete experiment accounting")
    regime_status_defaults = {
        "regimes_observed": 0,
        "regimes_with_minimum_trades": 0,
        "positive_expectancy_regimes": 0,
        "regime_profit_concentration": "",
        "worst_regime_expectancy": "",
        "crisis_trades": 0,
        "crisis_expectancy": "",
        "regime_stability_status": "NOT_EVALUATED",
        "regime_stability_blocker": "prior_walk_forward_gate_not_passed",
    }
    for column, default in regime_status_defaults.items():
        status[column] = status.get(
            column,
            pd.Series(index=status.index, dtype=object),
        ).fillna(default)
    candidate_frame = (
        pd.DataFrame(candidate_rows)
        if candidate_rows
        else status.iloc[0:0].copy()
    )
    detail = pd.DataFrame(detail_rows, columns=REGIME_DETAIL_COLUMNS)
    regime_bars = (
        pd.concat(regime_bar_frames, ignore_index=True)
        if regime_bar_frames
        else pd.DataFrame(columns=["experiment_id", "timestamp", "regime"])
    )
    regime_trades = (
        pd.concat(regime_trade_frames, ignore_index=True)
        if regime_trade_frames
        else pd.DataFrame(columns=["experiment_id", "entry_timestamp", "regime"])
    )
    for frame, active_path, snapshot_path in (
        (status, paths["status"], paths["snapshot_status"]),
        (candidate_frame, paths["candidates"], paths["snapshot_candidates"]),
        (detail, paths["detail"], paths["snapshot_detail"]),
        (regime_bars, paths["bars"], paths["snapshot_bars"]),
        (regime_trades, paths["trades"], paths["snapshot_trades"]),
    ):
        atomic_write_csv(frame, active_path, index=False)
        atomic_write_csv(frame, snapshot_path, index=False)

    status_counts = _status_counts(status, "regime_status")
    summary: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "observed_cost_replay_id": observed_replay_id,
        "walkforward_id": walkforward_id,
        "regime_attribution_id": attribution_id,
        "created_at": as_of.isoformat(),
        "experiments": int(len(status)),
        "unique_experiment_ids": int(status["experiment_id"].nunique()),
        "status_counts": status_counts,
        "experiment_status_accounted": bool(sum(status_counts.values()) == len(status)),
        "walkforward_research_passes_selected": int(len(selected)),
        "regime_attributions_complete": int(len(candidate_frame)),
        "regime_stability_passes": int(
            candidate_frame.get("regime_stability_status", pd.Series(dtype=str))
            .eq("PASS_RESEARCH_REGIME_STABILITY")
            .sum()
        ),
        "regime_detail_rows": int(len(detail)),
        "enriched_bar_rows": int(len(regime_bars)),
        "enriched_trade_rows": int(len(regime_trades)),
        "policy": policy,
        "point_in_time_regime_thresholds": True,
        "statistical_selection_passes_required_for_promotion": True,
        "acceptance_eligible_replays": 0,
        "live_trading_authorized": False,
        "input_hashes": material["input_hashes"],
        "artifacts": {name: _relative(path, root) for name, path in paths.items()},
        "input_snapshots": {
            name: _relative(path, root) for name, path in snapshot_inputs.items()
        },
    }
    manifest_text = json.dumps(summary, indent=2, sort_keys=True)
    summary_text = _summary_markdown(summary)
    atomic_write_text(paths["manifest"], manifest_text, encoding="utf-8")
    atomic_write_text(paths["snapshot_manifest"], manifest_text, encoding="utf-8")
    atomic_write_text(paths["summary_md"], summary_text, encoding="utf-8")
    atomic_write_text(paths["snapshot_summary_md"], summary_text, encoding="utf-8")
    return CommandResult(paths=paths, summary=summary)


def build_causal_pair_regime_features(history: pd.DataFrame) -> pd.DataFrame:
    """Build prior-only regime features whose prefixes are future-invariant."""

    required = {"timestamp", "price_x", "price_y"}
    missing = sorted(required.difference(history.columns))
    if missing:
        raise ValueError(f"regime_history_columns_missing:{','.join(missing)}")
    working = history.sort_index().copy()
    prices = working[["price_x", "price_y"]].apply(pd.to_numeric, errors="coerce")
    prices = prices.where((prices["price_x"] > 0.0) & (prices["price_y"] > 0.0))
    returns_x = prices["price_x"].pct_change()
    returns_y = prices["price_y"].pct_change()
    pair_variance = (returns_x.pow(2) + returns_y.pow(2)) / 2.0
    pair_volatility = pair_variance.rolling(
        VOLATILITY_WINDOW,
        min_periods=VOLATILITY_WINDOW,
    ).mean().pow(0.5)
    rolling_correlation = returns_x.rolling(
        CORRELATION_WINDOW,
        min_periods=CORRELATION_WINDOW,
    ).corr(returns_y)
    prior_volatility_threshold = (
        pair_volatility.expanding(min_periods=REGIME_REFERENCE_MIN_ROWS)
        .quantile(0.75)
        .shift(1)
    )
    prior_correlation_threshold = (
        rolling_correlation.expanding(min_periods=REGIME_REFERENCE_MIN_ROWS)
        .quantile(0.25)
        .shift(1)
    )
    regime = pd.Series("insufficient_regime_warmup", index=working.index, dtype=object)
    ready = pair_volatility.notna() & rolling_correlation.notna()
    ready &= prior_volatility_threshold.notna() & prior_correlation_threshold.notna()
    high_vol = ready & pair_volatility.gt(prior_volatility_threshold)
    decorrelated = ready & rolling_correlation.lt(prior_correlation_threshold)
    regime.loc[ready] = "calm"
    regime.loc[decorrelated] = "decorrelated"
    regime.loc[high_vol] = "high_vol"
    regime.loc[high_vol & decorrelated] = "crisis"
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(working["timestamp"], utc=True, errors="coerce"),
            "pair_volatility_20": pair_volatility,
            "rolling_correlation_60": rolling_correlation,
            "prior_volatility_q75": prior_volatility_threshold,
            "prior_correlation_q25": prior_correlation_threshold,
            "regime": regime,
            "regime_uses_future_data": False,
        },
        index=working.index,
    ).reset_index(drop=True)


def _join_regime(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    *,
    timestamp_column: str,
) -> pd.DataFrame:
    if frame.empty:
        return frame.assign(regime=pd.Series(dtype=str))
    working = frame.copy()
    working["_regime_timestamp"] = pd.to_datetime(
        working[timestamp_column],
        utc=True,
        errors="coerce",
    )
    feature_copy = features.copy()
    feature_copy["_regime_timestamp"] = pd.to_datetime(
        feature_copy["timestamp"], utc=True, errors="coerce"
    )
    feature_copy = feature_copy.drop(columns="timestamp")
    joined = working.merge(feature_copy, on="_regime_timestamp", how="left", validate="many_to_one")
    joined["regime"] = joined["regime"].fillna("regime_timestamp_unmatched")
    joined["regime_uses_future_data"] = joined["regime_uses_future_data"].fillna(False)
    return joined.drop(columns="_regime_timestamp")


def _regime_details(
    candidate: pd.Series,
    bars: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    attribution_id: str,
    walkforward_id: str,
) -> list[dict[str, object]]:
    details: list[dict[str, object]] = []
    regimes = sorted(set(bars["regime"].astype(str)) | set(trades.get("regime", pd.Series(dtype=str)).astype(str)))
    for regime in regimes:
        regime_bars = bars.loc[bars["regime"].astype(str).eq(regime)].copy()
        regime_trades = trades.loc[
            trades.get("regime", pd.Series(index=trades.index, dtype=str)).astype(str).eq(regime)
        ].copy()
        net = pd.to_numeric(regime_bars.get("net_return"), errors="coerce").fillna(0.0)
        equity = (1.0 + net).cumprod()
        closed = pd.to_numeric(
            regime_trades.get("profit_after_cost"), errors="coerce"
        ).dropna()
        wins = closed.loc[closed > 0.0]
        losses = closed.loc[closed < 0.0]
        if not losses.empty:
            profit_factor = float(wins.sum() / abs(losses.sum()))
        elif not wins.empty:
            profit_factor = float("inf")
        else:
            profit_factor = 0.0
        sharpe = calculate_annualized_sharpe(
            net,
            interval=_text(candidate.get("hyperliquid_interval")),
        )
        details.append(
            {
                "schema_version": SCHEMA_VERSION,
                "regime_attribution_id": attribution_id,
                "walkforward_id": walkforward_id,
                "experiment_id": _text(candidate.get("experiment_id")),
                "pair_group_id": _text(candidate.get("pair_group_id")),
                "pair": _text(candidate.get("pair")),
                "wizard_exchange": _text(candidate.get("wizard_exchange")),
                "wizard_timeframe": _text(candidate.get("wizard_timeframe")),
                "hyperliquid_interval": _text(candidate.get("hyperliquid_interval")),
                "exact_mode": _text(candidate.get("exact_mode")),
                "orientation": _text(candidate.get("orientation")),
                "regime": regime,
                "bar_rows": int(len(regime_bars)),
                "active_bar_rows": int(
                    pd.to_numeric(regime_bars.get("target_position"), errors="coerce")
                    .fillna(0.0)
                    .abs()
                    .gt(0.0)
                    .sum()
                ),
                "closed_trades": int(len(closed)),
                "profit_factor": profit_factor,
                "expectancy": float(closed.mean()) if not closed.empty else 0.0,
                "win_rate": float((closed > 0.0).mean()) if not closed.empty else 0.0,
                "conditional_bar_sharpe": sharpe.value,
                "conditional_sharpe_status": (
                    sharpe.status if sharpe.status == "valid" else f"blocked:{sharpe.reason}"
                ),
                "conditional_max_drawdown": max_drawdown(equity),
                "conditional_total_return": (
                    float(equity.iloc[-1] - 1.0) if len(equity) else 0.0
                ),
                "total_trade_profit_after_cost": float(closed.sum()) if not closed.empty else 0.0,
                "total_fees": _sum_column(regime_bars, "fees"),
                "total_slippage": _sum_column(regime_bars, "slippage"),
                "total_funding": _sum_column(regime_bars, "funding"),
                "total_execution_risk": _sum_column(regime_bars, "execution_risk"),
                "total_partial_fill_cost": _sum_column(regime_bars, "partial_fill"),
                "acceptance_status": "BLOCKED",
                "acceptance_reason": RESEARCH_ONLY_REASON,
                "live_trading_authorized": False,
            }
        )
    return details


def _candidate_regime_summary(
    candidate: pd.Series,
    details: list[dict[str, object]],
) -> dict[str, object]:
    meaningful = [
        row for row in details if int(_number(row.get("closed_trades")) or 0) >= MIN_REGIME_TRADES
    ]
    positive_profit = [
        float(_number(row.get("total_trade_profit_after_cost")) or 0.0)
        for row in meaningful
        if float(_number(row.get("total_trade_profit_after_cost")) or 0.0) > 0.0
    ]
    concentration = max(positive_profit) / sum(positive_profit) if positive_profit else 1.0
    positive_expectancy = sum(
        float(_number(row.get("expectancy")) or 0.0) > 0.0 for row in meaningful
    )
    blockers: list[str] = []
    if len(meaningful) < 2:
        blockers.append("fewer_than_two_regimes_with_minimum_trades")
    if meaningful and positive_expectancy < len(meaningful):
        blockers.append("negative_or_zero_expectancy_regime")
    if concentration > MAX_REGIME_PROFIT_CONCENTRATION:
        blockers.append(
            f"regime_profit_concentration>{MAX_REGIME_PROFIT_CONCENTRATION:g}"
        )
    if _text(candidate.get("statistical_selection_status")) != "PASS":
        blockers.append("statistical_selection_not_passed")
    crisis = next((row for row in details if _text(row.get("regime")) == "crisis"), None)
    return {
        "regimes_observed": len(details),
        "regimes_with_minimum_trades": len(meaningful),
        "positive_expectancy_regimes": positive_expectancy,
        "regime_profit_concentration": concentration,
        "worst_regime_expectancy": (
            min(float(_number(row.get("expectancy")) or 0.0) for row in meaningful)
            if meaningful
            else 0.0
        ),
        "crisis_trades": int(_number((crisis or {}).get("closed_trades")) or 0),
        "crisis_expectancy": _number((crisis or {}).get("expectancy")),
        "regime_stability_status": (
            "PASS_RESEARCH_REGIME_STABILITY"
            if not [item for item in blockers if item != "statistical_selection_not_passed"]
            else "FAIL_RESEARCH_REGIME_STABILITY"
        ),
        "regime_stability_blocker": ";".join(blockers),
        "statistical_selection_status": _text(
            candidate.get("statistical_selection_status")
        ),
        "bh_qvalue": _number(candidate.get("bh_qvalue")),
    }


def _status_base(
    row: object,
    *,
    attribution_id: str,
    evidence_paths: object,
    root: Path,
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "regime_attribution_id": attribution_id,
        "walkforward_id": _text(getattr(row, "walkforward_id", "")),
        "experiment_id": _text(row.experiment_id),
        "pair_group_id": _text(row.pair_group_id),
        "pair": _text(row.pair),
        "wizard_exchange": _text(getattr(row, "wizard_exchange", "")),
        "wizard_timeframe": _text(getattr(row, "wizard_timeframe", "")),
        "hyperliquid_interval": _text(getattr(row, "hyperliquid_interval", "")),
        "exact_mode": _text(row.exact_mode),
        "orientation": _text(row.orientation),
        "history_validation_lane": _text(
            getattr(row, "history_validation_lane", "")
        ),
        "walkforward_rank_eligible": _truthy(
            getattr(row, "walkforward_rank_eligible", False)
        ),
        "statistical_selection_status": _text(
            getattr(row, "statistical_selection_status", "")
        ),
        "statistical_selection_blocker": _text(
            getattr(row, "statistical_selection_blocker", "")
        ),
        "prior_walkforward_status": _text(getattr(row, "walkforward_status", "")),
        "regime_status": "",
        "regime_blocker": "",
        "acceptance_status": "BLOCKED",
        "acceptance_reason": _downstream_acceptance_reason(row),
        "acceptance_eligible": False,
        "evidence_path": ";".join(_relative(Path(path), root) for path in evidence_paths),
        "live_trading_authorized": False,
    }


def _downstream_acceptance_reason(row: object) -> str:
    reasons: list[str] = []
    if _text(getattr(row, "history_validation_lane", "")) == (
        "SHORT_HISTORY_RESEARCH_ONLY"
    ):
        reasons.append("short_history_research_only")
    reasons.append(RESEARCH_ONLY_REASON)
    selection_status = _text(getattr(row, "statistical_selection_status", ""))
    selection_blocker = _text(getattr(row, "statistical_selection_blocker", ""))
    if selection_status and selection_status != "PASS":
        reasons.append(f"statistical_selection_{selection_status.lower()}")
    if selection_blocker:
        reasons.append(selection_blocker)
    return ";".join(dict.fromkeys(reason for reason in reasons if reason))


def _row_lookup(frame: pd.DataFrame, key: str) -> dict[str, object]:
    return {_text(getattr(row, key)): row for row in frame.itertuples()}


def _require_unique(frame: pd.DataFrame, key: str) -> None:
    if frame.empty or key not in frame.columns or frame[key].astype(str).duplicated().any():
        raise ValueError(f"Expected non-empty unique {key} rows")


def _status_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    return {
        _text(status) or "MISSING_STATUS": int(count)
        for status, count in frame[column].value_counts(dropna=False).items()
    }


def _summary_markdown(summary: dict[str, object]) -> str:
    return "\n".join(
        [
            "# Exhaustive Hyperliquid Causal Regime Attribution",
            "",
            f"- Run: `{summary['run_id']}`",
            f"- Walk-forward: `{summary['walkforward_id']}`",
            f"- Regime attribution: `{summary['regime_attribution_id']}`",
            f"- Experiments accounted: {summary['unique_experiment_ids']} / {summary['experiments']}",
            f"- Walk-forward research passes selected: {summary['walkforward_research_passes_selected']}",
            f"- Regime attributions complete: {summary['regime_attributions_complete']}",
            f"- Research regime-stability passes: {summary['regime_stability_passes']}",
            f"- Detail rows: {summary['regime_detail_rows']}",
            f"- Acceptance-eligible replays: {summary['acceptance_eligible_replays']}",
            f"- Live trading authorized: `{str(summary['live_trading_authorized']).lower()}`",
            "",
            "Regimes use trailing two-leg returns and prior-only expanding thresholds shifted by one bar. Future rows cannot alter earlier labels. Results are conditional research diagnostics and do not override the failed family-wide statistical-selection gate.",
            "",
        ]
    )


def _sum_column(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return 0.0
    return float(pd.to_numeric(frame[column], errors="coerce").fillna(0.0).sum())


def _number(value: object) -> float | None:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(parsed) if pd.notna(parsed) else None


def _text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return _text(value).lower() in {"1", "true", "yes", "y"}


def _read_csv(path: Path) -> pd.DataFrame:
    if path.exists() and path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path, keep_default_na=False)


def _file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
