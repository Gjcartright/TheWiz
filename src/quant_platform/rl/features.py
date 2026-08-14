from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "zscore",
    "rolling_zscore",
    "spread",
    "spread_slope",
    "realized_volatility_percentile",
    "correlation",
    "hedge_ratio",
    "hedge_ratio_stability",
    "beta",
    "beta_stability",
    "copula_calibration_score",
    # Point-in-time Crypto Wizards Copula observations. These are dashboard
    # context, never dashboard return/Sharpe labels or reward inputs.
    "wizard_copula_available",
    "wizard_copula_stale_snapshot",
    "wizard_copula_x_given_y",
    "wizard_copula_y_given_x",
    "wizard_copula_probability_gap",
    "wizard_copula_correlation",
    "wizard_copula_pearson",
    "wizard_copula_spearman",
    "wizard_copula_kendall",
    "wizard_copula_hurst",
    "wizard_copula_half_life",
    "wizard_copula_hedge_ratio",
    "wizard_copula_capital_weight",
    "wizard_copula_volume_x_usd",
    "wizard_copula_volume_y_usd",
    "wizard_copula_var_99",
    "wizard_copula_cvar_99",
    "wizard_copula_zscore_norm",
    "wizard_copula_zscore_roll",
    "wizard_copula_signal_strength",
    "wizard_copula_snapshot_completeness",
    "wizard_copula_johansen_confirmed",
    "wizard_copula_engle_granger_confirmed",
    "wizard_copula_engle_granger_trending",
    "wizard_copula_ecm_x_available",
    "wizard_copula_ecm_y_available",
    "wizard_copula_ecm_strength_available",
    "wizard_copula_entry_lower",
    "wizard_copula_entry_upper",
    "wizard_copula_exit_lower",
    "wizard_copula_exit_upper",
    "wizard_copula_direction_short_x_long_y",
    "wizard_copula_direction_long_x_short_y",
    "wizard_copula_research_only",
    "wizard_copula_execution_blocked",
    "wizard_copula_model_studentt",
    "wizard_copula_model_gaussian",
    "wizard_copula_model_clayton",
    "wizard_copula_model_gumbel",
    "wizard_copula_model_frank",
    "wizard_copula_model_joe",
    "wizard_copula_lower_tail_family",
    "wizard_copula_upper_tail_family",
    "wizard_copula_symmetric_family",
    "wizard_copula_model_other",
    "ecm_x",
    "ecm_y",
    "ecm_strength",
    "funding_bps_per_day",
    "crisis_probability",
    "liquidity_score",
    "trade_quality_score",
]

FUTURE_ONLY_COLUMNS = {
    "good_trade",
    "profit_after_cost",
    "max_adverse_excursion",
    "max_favorable_excursion",
    "label_timestamp",
    "exit_timestamp",
    "exit_reason",
    "future_return",
    "realized_return",
    "hold_bars",
    "trade_bars",
    "entry_bar_index",
    "exit_bar_index",
}

RL_POLICY_SCHEMA_VERSION = "rl_policy_v2"
COPULA_FEATURE_PREFIX = "wizard_copula_"

# These dashboard fields are intentionally excluded. They are useful for
# discovery and human review, but can be full-sample/hindsight statistics.
WIZARD_DASHBOARD_NON_FEATURE_COLUMNS = {
    "return_total",
    "return_total_top",
    "returns_total",
    "returns_total_pct",
    "net_return_detail",
    "annualized_return_detail",
    "mean_period_return_detail",
    "sharpe",
    "sharpe_top",
    "sharpe_detail",
    "sortino_detail",
    "win_rate",
    "closed_trades",
}


def _pair_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(value or "").lower()).strip("-")


def _timeframe_key(value: object) -> str:
    text = re.sub(r"[^a-z0-9]+", "", str(value or "").lower())
    aliases = {"1d": "daily", "day": "daily", "24h": "daily", "4h": "4hour", "1h": "1hour", "5m": "5min"}
    return aliases.get(text, text)


def _mode_key(value: object) -> str:
    text = re.sub(r"[^a-z0-9]+", "", _text(value).lower())
    if "copula" in text:
        return "copula"
    suffix = "zscorer" if "zscore" in text else "spread"
    if text.startswith("ou") or "ornstein" in text:
        return f"ou_{suffix}"
    if "dyn" in text or "kalman" in text:
        return f"dynamic_{suffix}"
    if "static" in text:
        return f"static_{suffix}"
    return text


def _timestamp_column(frame: pd.DataFrame, names: tuple[str, ...]) -> pd.Series:
    result = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
    for name in names:
        if name in frame.columns:
            parsed = pd.to_datetime(frame[name], utc=True, errors="coerce", format="mixed")
            result = result.fillna(parsed)
    return result


def _numeric(row: pd.Series, *names: str, scale_percent: bool = False) -> float:
    for name in names:
        if name not in row.index:
            continue
        value = pd.to_numeric(pd.Series([row.get(name)]), errors="coerce").iloc[0]
        if pd.notna(value):
            numeric = float(value)
            return numeric / 100.0 if scale_percent and abs(numeric) > 1.0 else numeric
    return 0.0


def _text(value: object) -> str:
    try:
        if value is None or pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _probability_or_none(row: pd.Series, *names: str) -> float | None:
    for name in names:
        if name not in row.index:
            continue
        value = pd.to_numeric(pd.Series([row.get(name)]), errors="coerce").iloc[0]
        if pd.notna(value):
            numeric = float(value)
            probability = numeric / 100.0 if name.endswith("_pct") or abs(numeric) > 1.0 else numeric
            return probability if 0.0 <= probability <= 1.0 else None
    return None


def _state_is(row: pd.Series, column: str, *states: str) -> float:
    value = _text(row.get(column, "")).lower().replace("-", "_").replace(" ", "_")
    return float(value in states)


def _has_text(value: object) -> float:
    text = _text(value).lower()
    return float(bool(text and text not in {"nan", "none", "null", "false", "0"}))


def _boolish_float(value: object) -> float:
    text = _text(value).lower()
    if text in {"true", "1", "yes", "y", "available", "present"}:
        return 1.0
    if text in {"false", "0", "no", "n", "missing", "unavailable", "none", "nan", ""}:
        return 0.0
    return float(bool(value))


def _source_evidence(snapshot: pd.Series) -> str:
    for column in ("evidence_path", "page_detail_screenshot_path", "source_path", "scanner_snapshot_path"):
        value = _text(snapshot.get(column))
        if value:
            return value
    return ""


def _snapshot_completeness(snapshot: pd.Series) -> float:
    snapshot_mode = next(
        (value for value in (_text(snapshot.get("strategy_label")), _text(snapshot.get("exact_mode"))) if value),
        "",
    )
    values = [
        _source_evidence(snapshot), snapshot_mode, _text(snapshot.get("copula_best_fit")),
        _probability_or_none(snapshot, "copula_x_given_y_pct", "copula_x_given_y", "dependency_x_over_y"),
        _probability_or_none(snapshot, "copula_y_given_x_pct", "copula_y_given_x", "dependency_y_over_x"),
        snapshot.get("copula_correlation_rho"), snapshot.get("pearson_rho"), snapshot.get("spearman_rho"),
        snapshot.get("kendall_tau"), snapshot.get("johansen_badge_state"), snapshot.get("engle_granger_badge_state"),
        snapshot.get("ecm_x_available"), snapshot.get("ecm_y_available"), snapshot.get("ecm_strength_available"),
        snapshot.get("copula_entry_lower"), snapshot.get("copula_entry_upper"),
        snapshot.get("copula_exit_lower"), snapshot.get("copula_exit_upper"),
    ]
    populated = sum(
        value is not None
        and not (isinstance(value, float) and np.isnan(value))
        and not (isinstance(value, str) and not value.strip())
        for value in values
    )
    return round(populated / len(values), 6)


def _snapshot_is_complete(snapshot: pd.Series) -> bool:
    return bool(
        _source_evidence(snapshot)
        and _text(snapshot.get("copula_best_fit"))
        and _probability_or_none(snapshot, "copula_x_given_y_pct", "copula_x_given_y", "dependency_x_over_y") is not None
        and _probability_or_none(snapshot, "copula_y_given_x_pct", "copula_y_given_x", "dependency_y_over_x") is not None
    )


def _copula_feature_values(snapshot: pd.Series) -> dict[str, float]:
    best_fit = _text(snapshot.get("copula_best_fit", "")).lower().replace("-", "")
    signal = _text(snapshot.get("copula_signal_status", "")).lower()
    journal_status = _text(snapshot.get("copula_journal_status", "")).lower()
    x_given_y = _probability_or_none(snapshot, "copula_x_given_y_pct", "copula_x_given_y", "dependency_x_over_y")
    y_given_x = _probability_or_none(snapshot, "copula_y_given_x_pct", "copula_y_given_x", "dependency_y_over_x")
    explicit_gap = _probability_or_none(snapshot, "copula_probability_gap_pct", "copula_probability_gap")
    probability_gap = explicit_gap if explicit_gap is not None else abs(float(x_given_y) - float(y_given_x))
    direction = _text(snapshot.get("copula_trade_direction")).lower()
    student_t = best_fit in {"studentt", "student"}
    gaussian = best_fit in {"gaussian", "normal"}
    clayton = "clayton" in best_fit
    gumbel = "gumbel" in best_fit
    frank = "frank" in best_fit
    joe = best_fit == "joe" or best_fit.startswith("joe")
    return {
        "wizard_copula_available": 1.0,
        "wizard_copula_stale_snapshot": 0.0,
        "wizard_copula_x_given_y": float(x_given_y),
        "wizard_copula_y_given_x": float(y_given_x),
        "wizard_copula_probability_gap": probability_gap,
        "wizard_copula_correlation": _numeric(snapshot, "copula_correlation_rho", "correlation_value", "correlation_top", scale_percent=True),
        "wizard_copula_pearson": _numeric(snapshot, "pearson_rho", "pearson", scale_percent=True),
        "wizard_copula_spearman": _numeric(snapshot, "spearman_rho", "spearman", scale_percent=True),
        "wizard_copula_kendall": _numeric(snapshot, "kendall_tau", "kendall", scale_percent=True),
        "wizard_copula_hurst": _numeric(snapshot, "hurst_top", "hurst_value"),
        "wizard_copula_half_life": _numeric(snapshot, "half_life_top", "half_life_value"),
        "wizard_copula_hedge_ratio": _numeric(snapshot, "hedge_ratio_top"),
        "wizard_copula_capital_weight": _numeric(snapshot, "capital_weighting_slider_value"),
        "wizard_copula_volume_x_usd": _numeric(snapshot, "volume_x_top", "volume_x"),
        "wizard_copula_volume_y_usd": _numeric(snapshot, "volume_y_top", "volume_y"),
        "wizard_copula_var_99": _numeric(snapshot, "var_99_detail", "var_99"),
        "wizard_copula_cvar_99": _numeric(snapshot, "cvar_99_detail", "cvar_99"),
        "wizard_copula_zscore_norm": _numeric(snapshot, "zscore_norm_value"),
        "wizard_copula_zscore_roll": _numeric(snapshot, "zscore_roll_value"),
        "wizard_copula_signal_strength": 1.0 if signal == "strong_asymmetric_dislocation" else 0.5 if signal == "asymmetric_dislocation" else 0.0,
        "wizard_copula_snapshot_completeness": _snapshot_completeness(snapshot),
        "wizard_copula_johansen_confirmed": _state_is(snapshot, "johansen_badge_state", "confirmed", "green"),
        "wizard_copula_engle_granger_confirmed": _state_is(snapshot, "engle_granger_badge_state", "confirmed", "green"),
        "wizard_copula_engle_granger_trending": _state_is(snapshot, "engle_granger_badge_state", "trending", "orange"),
        "wizard_copula_ecm_x_available": _boolish_float(snapshot.get("ecm_x_available")),
        "wizard_copula_ecm_y_available": _boolish_float(snapshot.get("ecm_y_available")),
        "wizard_copula_ecm_strength_available": _boolish_float(snapshot.get("ecm_strength_available")),
        "wizard_copula_entry_lower": _numeric(snapshot, "copula_entry_lower"),
        "wizard_copula_entry_upper": _numeric(snapshot, "copula_entry_upper"),
        "wizard_copula_exit_lower": _numeric(snapshot, "copula_exit_lower"),
        "wizard_copula_exit_upper": _numeric(snapshot, "copula_exit_upper"),
        "wizard_copula_direction_short_x_long_y": float(direction == "short_x_long_y"),
        "wizard_copula_direction_long_x_short_y": float(direction == "long_x_short_y"),
        "wizard_copula_research_only": float(journal_status == "research_only"),
        "wizard_copula_execution_blocked": _has_text(snapshot.get("copula_execution_blockers")),
        "wizard_copula_model_studentt": float(student_t),
        "wizard_copula_model_gaussian": float(gaussian),
        "wizard_copula_model_clayton": float(clayton),
        "wizard_copula_model_gumbel": float(gumbel),
        "wizard_copula_model_frank": float(frank),
        "wizard_copula_model_joe": float(joe),
        "wizard_copula_lower_tail_family": float(clayton or "lower" in best_fit),
        "wizard_copula_upper_tail_family": float(gumbel or joe or "upper" in best_fit),
        "wizard_copula_symmetric_family": float(student_t or gaussian or frank),
        "wizard_copula_model_other": float(bool(best_fit and not any((student_t, gaussian, clayton, gumbel, frank, joe)))),
    }


def _empty_copula_feature_values(*, stale: bool = False) -> dict[str, float]:
    values = {column: 0.0 for column in FEATURE_COLUMNS if column.startswith(COPULA_FEATURE_PREFIX)}
    values["wizard_copula_stale_snapshot"] = float(stale)
    return values


def attach_copula_dashboard_features(
    candidates: pd.DataFrame,
    journal: pd.DataFrame,
    *,
    max_snapshot_age_hours: float = 2.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach the latest eligible Wizard Copula snapshot to each RL candidate.

    A snapshot is eligible only when its capture time is at or before the
    candidate's feature time. This makes the same helper safe for replay,
    paper, and live candidate construction. A dashboard field never supplies
    a label, reward, or realized outcome.
    """
    enriched = candidates.copy()
    audit_columns = [
        "candidate_index", "pair", "timeframe", "feature_timestamp", "snapshot_timestamp",
        "snapshot_age_hours", "candidate_mode", "snapshot_mode", "mode_match",
        "snapshot_completeness", "source_evidence_present", "join_status",
        "uses_future_data", "evidence_path",
    ]
    if enriched.empty:
        return enriched, pd.DataFrame(columns=audit_columns)

    for column, value in _empty_copula_feature_values().items():
        enriched[column] = value
    enriched["wizard_copula_snapshot_timestamp"] = ""
    enriched["wizard_copula_evidence_path"] = ""
    enriched["wizard_copula_join_status"] = "missing_journal_snapshot"

    candidate_times = _timestamp_column(enriched, ("feature_timestamp", "entry_timestamp", "detection_timestamp"))
    if journal.empty or "pair" not in journal.columns:
        audit = pd.DataFrame(
            [{"candidate_index": index, "pair": row.get("pair", ""), "timeframe": row.get("timeframe", ""), "feature_timestamp": candidate_times.loc[index], "snapshot_timestamp": "", "snapshot_age_hours": "", "join_status": "missing_journal", "uses_future_data": False, "evidence_path": ""} for index, row in enriched.iterrows()],
            columns=audit_columns,
        )
        enriched["wizard_copula_join_status"] = "missing_journal"
        return enriched, audit

    snapshots = journal.copy()
    snapshots["_pair_key"] = snapshots["pair"].map(_pair_key)
    snapshots["_timeframe_key"] = snapshots.get("timeframe", pd.Series("", index=snapshots.index)).map(_timeframe_key)
    snapshots["_mode_key"] = snapshots.get(
        "strategy_label", snapshots.get("exact_mode", pd.Series("", index=snapshots.index))
    ).map(_mode_key)
    snapshots["_journal_layer"] = snapshots.get("journal_layer", pd.Series("", index=snapshots.index)).astype(str).str.strip().str.lower()
    snapshots["_capture_status"] = snapshots.get("capture_status", pd.Series("", index=snapshots.index)).astype(str).str.strip().str.lower()
    snapshots["_snapshot_timestamp"] = _timestamp_column(snapshots, ("detail_capture_timestamp_utc", "capture_timestamp_utc", "scanner_refresh_timestamp_utc"))
    snapshots = snapshots.dropna(subset=["_snapshot_timestamp"]).sort_values("_snapshot_timestamp")

    audit_rows: list[dict[str, object]] = []
    max_age = pd.Timedelta(hours=max(0.0, float(max_snapshot_age_hours)))
    for index, row in enriched.iterrows():
        feature_time = candidate_times.loc[index]
        pair_key = _pair_key(row.get("pair", ""))
        timeframe_key = _timeframe_key(row.get("timeframe", ""))
        candidate_mode = ""
        for mode_column in ("exact_mode", "wizard_exact_mode", "strategy_label"):
            candidate_mode = _mode_key(row.get(mode_column))
            if candidate_mode:
                break
        audit_row = {
            "candidate_index": index,
            "pair": row.get("pair", ""),
            "timeframe": row.get("timeframe", ""),
            "feature_timestamp": feature_time,
            "snapshot_timestamp": "",
            "snapshot_age_hours": "",
            "candidate_mode": candidate_mode,
            "snapshot_mode": "",
            "mode_match": False,
            "snapshot_completeness": 0.0,
            "source_evidence_present": False,
            "join_status": "",
            "uses_future_data": False,
            "evidence_path": "",
        }
        if pd.isna(feature_time):
            status = "missing_feature_timestamp"
            enriched.at[index, "wizard_copula_join_status"] = status
            audit_row["join_status"] = status
            audit_rows.append(audit_row)
            continue
        if not timeframe_key:
            status = "missing_candidate_timeframe"
            enriched.at[index, "wizard_copula_join_status"] = status
            audit_row["join_status"] = status
            audit_rows.append(audit_row)
            continue
        if not candidate_mode:
            status = "missing_candidate_exact_mode"
            enriched.at[index, "wizard_copula_join_status"] = status
            audit_row["join_status"] = status
            audit_rows.append(audit_row)
            continue
        eligible = snapshots[
            snapshots["_pair_key"].eq(pair_key)
            & snapshots["_timeframe_key"].eq(timeframe_key)
            & snapshots["_mode_key"].eq(candidate_mode)
            & snapshots["_journal_layer"].eq("pair_detail_capture")
            & snapshots["_capture_status"].eq("captured")
            & snapshots["_snapshot_timestamp"].le(feature_time)
        ].copy()
        if eligible.empty:
            status = "no_point_in_time_snapshot"
            enriched.at[index, "wizard_copula_join_status"] = status
            audit_row["join_status"] = status
            audit_rows.append(audit_row)
            continue
        eligible["_evidence_path"] = eligible.apply(_source_evidence, axis=1)
        eligible["_snapshot_complete"] = eligible.apply(_snapshot_is_complete, axis=1)
        eligible["_snapshot_completeness"] = eligible.apply(_snapshot_completeness, axis=1)
        complete = eligible[eligible["_snapshot_complete"]].copy()
        if complete.empty:
            snapshot = eligible.iloc[-1]
            snapshot_time = snapshot["_snapshot_timestamp"]
            age_hours = float((feature_time - snapshot_time).total_seconds() / 3600.0)
            evidence_path = _text(snapshot.get("_evidence_path"))
            status = "incomplete_copula_snapshot" if evidence_path else "source_evidence_missing"
            enriched.at[index, "wizard_copula_snapshot_timestamp"] = snapshot_time.isoformat()
            enriched.at[index, "wizard_copula_evidence_path"] = evidence_path
            enriched.at[index, "wizard_copula_join_status"] = status
            audit_row.update({
                "snapshot_timestamp": snapshot_time, "snapshot_age_hours": age_hours,
                "snapshot_mode": snapshot.get("_mode_key", ""),
                "mode_match": snapshot.get("_mode_key", "") == candidate_mode,
                "snapshot_completeness": snapshot.get("_snapshot_completeness", 0.0),
                "source_evidence_present": bool(evidence_path), "join_status": status,
                "evidence_path": evidence_path,
            })
            audit_rows.append(audit_row)
            continue
        snapshot = complete.iloc[-1]
        snapshot_time = snapshot["_snapshot_timestamp"]
        age = feature_time - snapshot_time
        age_hours = float(age.total_seconds() / 3600.0)
        evidence_path = _text(snapshot.get("_evidence_path"))
        audit_row.update({
            "snapshot_timestamp": snapshot_time, "snapshot_age_hours": age_hours,
            "snapshot_mode": snapshot.get("_mode_key", ""),
            "mode_match": snapshot.get("_mode_key", "") == candidate_mode,
            "snapshot_completeness": snapshot.get("_snapshot_completeness", 0.0),
            "source_evidence_present": bool(evidence_path), "evidence_path": evidence_path,
        })
        if age > max_age:
            status = "stale_snapshot"
            values = _empty_copula_feature_values(stale=True)
            enriched.loc[index, list(values)] = list(values.values())
        else:
            status = "attached"
            values = _copula_feature_values(snapshot)
            enriched.loc[index, list(values)] = list(values.values())
        enriched.at[index, "wizard_copula_snapshot_timestamp"] = snapshot_time.isoformat()
        enriched.at[index, "wizard_copula_evidence_path"] = evidence_path
        enriched.at[index, "wizard_copula_join_status"] = status
        audit_row["join_status"] = status
        audit_rows.append(audit_row)
    return enriched, pd.DataFrame(audit_rows, columns=audit_columns)


def leakage_columns(columns: list[str] | pd.Index) -> list[str]:
    lowered = {str(column).lower(): str(column) for column in columns}
    return [original for key, original in lowered.items() if key in FUTURE_ONLY_COLUMNS]


def build_rl_feature_frame(frame: pd.DataFrame, *, allow_future_columns: bool = False) -> pd.DataFrame:
    leaked = leakage_columns(frame.columns)
    if leaked and not allow_future_columns:
        raise ValueError(f"rl_feature_leakage_columns:{','.join(sorted(leaked))}")
    features = pd.DataFrame(index=frame.index)
    for column in FEATURE_COLUMNS:
        if column in frame.columns:
            features[column] = pd.to_numeric(frame[column], errors="coerce")
        else:
            features[column] = 0.0
    return features.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float64")


def write_feature_schema(path: Path, columns: list[str] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": RL_POLICY_SCHEMA_VERSION,
        "features": columns or FEATURE_COLUMNS,
        "label_source": "backtest_trained",
        "live_enabled": False,
        "copula_dashboard_policy": {
            "source": "point_in_time_wizard_journal",
            "requires_snapshot_at_or_before_feature_timestamp": True,
            "requires_pair_detail_capture": True,
            "requires_exact_pair_timeframe_mode_match": True,
            "requires_complete_directional_probabilities_and_family": True,
            "requires_source_evidence_path": True,
            "excludes_dashboard_return_and_sharpe": True,
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path
