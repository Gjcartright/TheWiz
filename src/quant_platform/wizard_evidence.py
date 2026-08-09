from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache
import json
from pathlib import Path
from typing import Any

import pandas as pd

from quant_platform.active_pipeline import CommandResult
from quant_platform.pair_detail_ingestion import extract_history_rows, snapshot_from_payload
from quant_platform.strategies import STRATEGIES
from quant_platform.wizard_mode_replay import mode_requirements, normalize_exact_mode
from quant_platform.wizard_run_config import WIZARD_CONFIG_SCHEMA_VERSION, WizardRunConfiguration
from quant_platform.wizard_policy import DEFAULT_WIZARD_DISCOVERY_POLICY


ROOT = Path(__file__).resolve().parents[2]
ACTIVE = ROOT / "reports" / "active"
DATA_PROCESSED = ROOT / "data" / "processed"
COLLECTED = ROOT / "data" / "collected"

DISCOVERY_MIN_SHARPE = DEFAULT_WIZARD_DISCOVERY_POLICY.min_sharpe
DISCOVERY_MIN_RETURNS_TOTAL = DEFAULT_WIZARD_DISCOVERY_POLICY.min_returns_total_pct / 100.0
WIZARD_EVIDENCE_MAX_AGE_HOURS = 24.0
ENTRY_ZSCORE_THRESHOLD = 2.0

WIZARD_MODE_MAP: dict[tuple[int, int], str] = {
    (3, 1): "Static (Spread)",
    (3, 2): "Static (ZScoreR)",
    (1, 1): "Dyn (Spread)",
    (1, 2): "Dyn (ZScoreR)",
    (2, 1): "OU (Spread)",
    (2, 2): "OU (ZScoreR)",
    (1, 3): "Copula",
}

MODE_TO_IDS = {mode: ids for ids, mode in WIZARD_MODE_MAP.items()}

WIZARD_EVIDENCE_COLUMNS = [
    "pair",
    "asset_x",
    "asset_y",
    "exchange",
    "interval",
    "period",
    "wizard_pair_id",
    "wizard_setup_id",
    "backtest_timestamp",
    "backtest_timestamp_utc",
    "setup_identity",
    "setup_rank_within_pair",
    "setup_role",
    "primary_wizard_setup",
    "dashboard_recommended_strategy",
    "dashboard_pair_rank",
    "dashboard_selector_score",
    "exact_mode",
    "spread_id",
    "strategy_id",
    "local_strategy_id",
    "local_strategy_name",
    "local_strategy_family",
    "strategy_mapping_status",
    "mode_valid",
    "mode_source",
    "mode_blocker",
    "sharpe",
    "returns_total_raw",
    "returns_total_unit",
    "returns_total",
    "returns_total_pct",
    "source_annualized_return_raw",
    "discovery_min_sharpe",
    "discovery_min_returns_total",
    "passes_sharpe_gate",
    "passes_sharpe_gt_2",
    "passes_returns_total_gt_20pct",
    "hurst",
    "half_life",
    "zscore_last",
    "zscore_roll_last",
    "zscore_window",
    "mini_zscore",
    "stddev_cross",
    "zero_cross",
    "engle_granger_cointegrated",
    "engle_granger_trend",
    "engle_granger_pvalue",
    "johansen_cointegrated",
    "stationarity_status",
    "pearson",
    "spearman",
    "kendall",
    "copula",
    "corr_copula",
    "u1_given_u2",
    "u2_given_u1",
    "ecm_x_available",
    "ecm_y_available",
    "ecm_strength_available",
    "hedge_ratio",
    "volume_x",
    "volume_y",
    "volume_min",
    "volatility_x_lt",
    "volatility_y_lt",
    "x_weighting",
    "y_weighting",
    "profile_match",
    "ou_optimal",
    "closed_trades",
    "backtest_closed",
    "win_rate",
    "drawdown",
    "var",
    "cvar",
    "ml_confidence",
    "source_system",
    "source_authority",
    "source_timestamp",
    "source_age_hours",
    "source_fresh",
    "source_freshness_reason",
    "source_health",
    "backtest_settings_complete",
    "backtest_settings_missing",
    "source_path",
    "evidence_path",
]

HYPOTHESIS_COLUMNS = [
    "pair",
    "asset_x",
    "asset_y",
    "exchange",
    "interval",
    "setup_identity",
    "exact_mode",
    "spread_id",
    "strategy_id",
    "sharpe",
    "returns_total_raw",
    "returns_total",
    "returns_total_pct",
    "source_system",
    "source_authority",
    "source_timestamp",
    "source_age_hours",
    "source_fresh",
    "source_freshness_reason",
    "source_health",
    "stationarity_status",
    "engle_granger_cointegrated",
    "engle_granger_trend",
    "johansen_cointegrated",
    "zscore_last",
    "zscore_roll_last",
    "volume_min",
    "backtest_settings_complete",
    "backtest_settings_missing",
    "hypothesis_status",
    "hypothesis_reason",
    "next_step",
    "local_data_available",
    "evidence_path",
]

DIAGNOSTIC_COLUMNS = [
    "pair",
    "exchange",
    "interval",
    "setup_identity",
    "exact_mode",
    "correlation_status",
    "ecm_status",
    "copula_status",
    "mean_reversion_status",
    "stationarity_status",
    "wizard_diagnostic_score",
    "diagnostic_blocker",
    "source_authority",
    "source_timestamp",
    "source_fresh",
    "source_health",
    "evidence_path",
]

DISCOVERY_TRIAGE_COLUMNS = [
    "pair",
    "asset_x",
    "asset_y",
    "exchange",
    "interval",
    "period",
    "setup_identity",
    "exact_mode",
    "spread_id",
    "strategy_id",
    "source_authority",
    "source_timestamp",
    "source_age_hours",
    "source_fresh",
    "source_health",
    "evidence_path",
    "sharpe",
    "returns_total",
    "returns_total_pct",
    "passes_sharpe_gate",
    "passes_returns_total_gt_20pct",
    "discovery_screen_status",
    "stationarity_status",
    "engle_granger_cointegrated",
    "engle_granger_trend",
    "engle_granger_pvalue",
    "johansen_cointegrated",
    "hurst",
    "half_life",
    "pearson",
    "spearman",
    "kendall",
    "ecm_x_available",
    "ecm_y_available",
    "ecm_strength_available",
    "hedge_ratio",
    "signal_basis",
    "entry_signal_value",
    "entry_signal_abs",
    "entry_signal_status",
    "copula",
    "corr_copula",
    "u1_given_u2",
    "u2_given_u1",
    "copula_data_status",
    "volume_x",
    "volume_y",
    "volume_min",
    "volatility_x_lt",
    "volatility_y_lt",
    "x_weighting",
    "y_weighting",
    "closed_trades",
    "win_rate",
    "drawdown",
    "var",
    "cvar",
    "zero_cross",
    "profile_match",
    "ou_optimal",
    "backtest_timestamp_utc",
    "backtest_settings_complete",
    "backtest_settings_missing",
    "research_quality_status",
    "research_blockers",
    "execution_status",
    "next_step",
]

PAIR_DETAIL_CAPTURE_QUEUE_COLUMNS = [
    "priority",
    "pair",
    "asset_x",
    "asset_y",
    "exchange",
    "interval",
    "period",
    "setup_identity",
    "exact_mode",
    "spread_id",
    "strategy_id",
    "sharpe",
    "returns_total_pct",
    "research_quality_status",
    "research_blockers",
    "source_timestamp",
    "source_fresh",
    "source_authority",
    "required_settings",
    "capture_method",
    "api_retrieval_status",
    "capture_status",
    "next_step",
    "evidence_path",
]

WIZARD_MODE_MATRIX_CAPTURE_QUEUE_COLUMNS = [
    "priority",
    "setup_identity",
    "source_setup_identity",
    "pair",
    "asset_x",
    "asset_y",
    "exchange",
    "interval",
    "period",
    "exact_mode",
    "spread_id",
    "strategy_id",
    "source_exact_mode",
    "mode_switch_required",
    "source_sharpe",
    "source_returns_total_pct",
    "research_quality_status",
    "source_timestamp",
    "source_fresh",
    "capture_status",
    "required_settings",
    "required_mode_inputs",
    "mode_fidelity_target",
    "acceptance_eligible",
    "paper_or_execution_eligible",
    "next_step",
    "evidence_path",
]

WIZARD_REPLAY_HANDOFF_COLUMNS = [
    "priority",
    "setup_identity",
    "pair",
    "asset_x",
    "asset_y",
    "exchange",
    "interval",
    "period",
    "exact_mode",
    "spread_id",
    "strategy_id",
    "source_timestamp",
    "source_fresh",
    "source_authority",
    "sharpe",
    "returns_total_pct",
    "research_quality_status",
    "research_blockers",
    "entry_signal_status",
    "settings_status",
    "settings_missing",
    "required_settings",
    "venue_history_status",
    "venue_cost_status",
    "venue_slippage_status",
    "venue_funding_or_borrow_status",
    "venue_research_execution_status",
    "exploratory_replay_status",
    "exploratory_replay_blockers",
    "exploratory_next_step",
    "exploratory_cost_policy",
    "replay_status",
    "replay_blockers",
    "next_step",
    "acceptance_authority",
    "evidence_path",
]

WIZARD_MODE_REPLAY_CAPABILITY_COLUMNS = [
    "priority",
    "setup_identity",
    "pair",
    "asset_x",
    "asset_y",
    "exchange",
    "interval",
    "period",
    "exact_mode",
    "research_quality_status",
    "venue_history_status",
    "mode_capability_status",
    "mode_fidelity_status",
    "mode_fidelity_rule",
    "settings_capture_status",
    "required_mode_inputs",
    "captured_mode_inputs",
    "missing_mode_inputs",
    "research_replay_eligible",
    "acceptance_eligible",
    "paper_or_execution_eligible",
    "next_step",
    "settings_capture_evidence_path",
    "evidence_path",
]

EXPLORATORY_COST_SENSITIVITY_COLUMNS = [
    "setup_identity",
    "pair",
    "exchange",
    "interval",
    "exact_mode",
    "research_quality_status",
    "settings_status",
    "venue_history_status",
    "exploratory_replay_status",
    "cost_profile_id",
    "cost_profile_version",
    "cost_case",
    "cost_case_rank",
    "scenario_status",
    "taker_fee_bps",
    "slippage_bps",
    "execution_risk_bps",
    "funding_or_borrow_bps_per_day",
    "bars_per_day",
    "partial_fill_probability",
    "partial_fill_fraction",
    "partial_fill_penalty_bps",
    "expected_partial_fill_bps",
    "one_way_leg_cost_bps",
    "two_leg_round_trip_execution_cost_bps",
    "symmetric_two_leg_daily_carry_bps",
    "cost_assumption_authority",
    "acceptance_eligible",
    "paper_or_execution_eligible",
    "acceptance_requirements",
    "next_step",
    "evidence_path",
]

EXPLORATORY_COST_PROFILE_FILENAME = "exploratory_cost_sensitivity_profiles.json"

WIZARD_PAIR_SETTINGS_CAPTURE_TEMPLATE_COLUMNS = [
    "config_schema_version",
    "discovery_config_hash",
    "candidate_config_hash",
    "settings_config_hash",
    "setup_identity",
    "pair",
    "asset_x",
    "asset_y",
    "exchange",
    "interval",
    "period",
    "exact_mode",
    "spread_id",
    "strategy_id",
    "required_settings",
    "capture_timestamp_utc",
    "capture_evidence_path",
    "capture_confirmed",
    "pair_page_url",
    "entry_long_operator",
    "entry_long_value",
    "entry_long_position",
    "entry_short_operator",
    "entry_short_value",
    "entry_short_position",
    "exit_long_operator",
    "exit_long_value",
    "exit_short_operator",
    "exit_short_value",
    "capital_weighting_slider_value",
    "capital_weighting_asset",
    "capital_weighting",
    "metric_mode",
    "x_weighting",
    "y_weighting",
    "hedge_ratio",
    "dynamic_hedge_ratio_method",
    "dynamic_hedge_ratio_window",
    "ou_mu",
    "ou_sigma",
    "commission_rate",
    "slippage_rate",
    "stop_loss_rate",
    "exit_n_periods",
    "ecm_deviation_min",
    "correlation_strength_min",
    "simulation_runs",
    "live_update",
    "spread_type",
    "zscore_window",
    "copula_family",
    "copula_signal_type",
    "copula_direction_view",
    "copula_entry_lower",
    "copula_entry_upper",
    "copula_exit_lower",
    "copula_exit_upper",
]

WIZARD_PAIR_SETTINGS_CAPTURE_VALIDATION_COLUMNS = [
    "input_row_number",
    "setup_identity",
    "pair",
    "exchange",
    "interval",
    "exact_mode",
    "discovery_config_hash",
    "candidate_config_hash",
    "settings_config_hash",
    "capture_timestamp_utc",
    "capture_evidence_path",
    "valid",
    "validation_blockers",
    "accepted_into_active_capture",
    "next_step",
    "evidence_path",
]

PARITY_COLUMNS = [
    "pair",
    "interval",
    "setup_identity",
    "exact_mode",
    "wizard_source_path",
    "local_source_path",
    "wizard_zscore_last",
    "local_zscore_last",
    "wizard_rolling_zscore_last",
    "local_rolling_zscore_last",
    "zscore_delta",
    "rolling_zscore_delta",
    "parity_status",
    "parity_reason",
    "evidence_path",
]

STRATEGY_ALIGNMENT_COLUMNS = [
    "pair",
    "interval",
    "dashboard_recommended_strategy",
    "exact_mode",
    "spread_id",
    "strategy_id",
    "local_strategy_id",
    "local_strategy_name",
    "local_strategy_family",
    "strategy_mapping_status",
    "source_first_action",
    "evidence_path",
]

EXACT_MODE_CAPTURE_QUEUE_COLUMNS = [
    "priority_rank",
    "pair",
    "asset_x",
    "asset_y",
    "interval",
    "sharpe",
    "returns_total",
    "returns_total_pct",
    "mode_blocker",
    "pair_page_url",
    "required_capture_fields",
    "operator_action",
    "source_path",
    "evidence_path",
]


@dataclass(frozen=True)
class ModeResolution:
    exact_mode: str
    spread_id: int | None
    strategy_id: int | None
    mode_valid: bool
    mode_source: str
    mode_blocker: str


_LOCAL_STRATEGY_BY_ID = {spec.id: spec for spec in STRATEGIES}

_EXACT_MODE_LOCAL_STRATEGY = {
    "Static (Spread)": 1,
    "Static (ZScoreR)": 1,
    "Dyn (Spread)": 1,
    "Dyn (ZScoreR)": 20,
    "OU (Spread)": 14,
    "OU (ZScoreR)": 14,
    "Copula": 5,
}

_RECOMMENDED_STRATEGY_HINTS: list[tuple[str, str, int]] = [
    ("kalman", "Dyn (Spread)", 1),
    ("dynamic", "Dyn (Spread)", 1),
    ("rolling zscore", "Dyn (ZScoreR)", 20),
    ("zscorer", "Static (ZScoreR)", 1),
    ("z-score", "Static (ZScoreR)", 1),
    ("zscore", "Static (ZScoreR)", 1),
    ("ornstein", "OU (Spread)", 14),
    ("ou ", "OU (Spread)", 14),
    ("copula", "Copula", 5),
    ("ecm", "Static (ZScoreR)", 2),
    ("cointegration", "Static (Spread)", 8),
    ("static", "Static (Spread)", 1),
]


def mode_from_ids(spread_id: int | None, strategy_id: int | None) -> str:
    if spread_id is None or strategy_id is None:
        return ""
    return WIZARD_MODE_MAP.get((int(spread_id), int(strategy_id)), "")


def ids_from_exact_mode(exact_mode: str | None) -> tuple[int | None, int | None]:
    if not exact_mode:
        return None, None
    normalized = _normalize_mode_label(exact_mode)
    for label, ids in MODE_TO_IDS.items():
        if _normalize_mode_label(label) == normalized:
            return ids
    return None, None


def build_wizard_evidence(root: Path = ROOT, now: datetime | None = None) -> CommandResult:
    as_of = _as_utc(now or datetime.now(timezone.utc))
    rows = _wizard_evidence_rows(root, now=as_of)
    frame = pd.DataFrame(rows, columns=WIZARD_EVIDENCE_COLUMNS)
    if not frame.empty:
        pair_present = frame.get("pair", pd.Series(dtype=object)).fillna("").astype(str).str.strip().ne("")
        if not pair_present.all():
            frame = frame.loc[pair_present].copy()
        frame["_source_fresh"] = frame.get("source_fresh", pd.Series(False, index=frame.index)).map(_as_bool).astype(int)
        frame["_source_healthy"] = frame.get("source_health", pd.Series("", index=frame.index)).astype(str).eq("healthy").astype(int)
        frame["_source_priority"] = frame.get("source_system", pd.Series(dtype=object)).map(
            {
                "crypto_wizards_pair_page_capture": 0,
                "crypto_wizards_hourly_pair_panel": 1,
                "crypto_wizards_live_scanner_capture": 2,
                "crypto_wizards_active_report": 3,
                "crypto_wizards_pair_detail": 4,
            }
        ).fillna(9)
        frame = frame.sort_values(
            ["_source_fresh", "_source_healthy", "_source_priority", "pair", "interval", "exact_mode", "passes_sharpe_gate", "returns_total", "sharpe"],
            ascending=[False, False, True, True, True, True, False, False, False],
        )
        frame = _merge_duplicate_evidence_rows(frame)
        frame = _backfill_local_derived_history_fields(frame, root=root)
        frame = frame.drop_duplicates(subset=["pair", "interval", "exact_mode"], keep="first")
        pair_interval_keys = frame.apply(
            lambda row: _evidence_pair_interval_key(row),
            axis=1,
        )
        valid_pair_intervals = set(pair_interval_keys[frame.get("mode_valid", pd.Series(False, index=frame.index)).fillna(False).astype(bool)])
        if valid_pair_intervals:
            frame = frame.loc[
                frame.get("mode_valid", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
                | ~pair_interval_keys.isin(valid_pair_intervals)
            ].copy()
        frame = _assign_setup_identity(frame)
        frame = frame.sort_values(["source_fresh", "passes_sharpe_gate", "returns_total", "sharpe"], ascending=[False, False, False, False])
        frame = frame.drop(columns=[column for column in ["_source_fresh", "_source_healthy", "_source_priority"] if column in frame.columns])
    output = root / "data" / "processed" / "wizard_evidence.csv"
    summary = root / "reports" / "active" / "wizard_evidence_summary.md"
    _write_csv(frame, output)
    _write_text(summary, _wizard_evidence_markdown(frame))
    return CommandResult(
        paths={"wizard_evidence": output, "summary_md": summary},
        summary={"rows": len(frame), "mode_valid": int(frame["mode_valid"].astype(bool).sum()) if not frame.empty else 0},
    )


def build_wizard_hypotheses(root: Path = ROOT, now: datetime | None = None) -> CommandResult:
    evidence = _ensure_wizard_evidence(root)
    local_index = _local_history_index(root)
    as_of = _as_utc(now or datetime.now(timezone.utc))
    rows = []
    for _, row in evidence.iterrows():
        local_available = _local_data_available(local_index, row)
        freshness = _source_freshness(row, as_of)
        status, reason, next_step = _hypothesis_status(row, local_available, freshness=freshness)
        rows.append(
            {
                "pair": row.get("pair", ""),
                "asset_x": row.get("asset_x", ""),
                "asset_y": row.get("asset_y", ""),
                "exchange": row.get("exchange", ""),
                "interval": row.get("interval", ""),
                "setup_identity": row.get("setup_identity", ""),
                "exact_mode": row.get("exact_mode", ""),
                "spread_id": row.get("spread_id", ""),
                "strategy_id": row.get("strategy_id", ""),
                "sharpe": row.get("sharpe", ""),
                "returns_total_raw": row.get("returns_total_raw", ""),
                "returns_total": row.get("returns_total", ""),
                "returns_total_pct": row.get("returns_total_pct", ""),
                "source_system": row.get("source_system", ""),
                "source_authority": row.get("source_authority", ""),
                "source_timestamp": freshness["timestamp"].isoformat() if freshness["timestamp"] else "",
                "source_age_hours": freshness["age_hours"] if freshness["age_hours"] is not None else "",
                "source_fresh": freshness["fresh"],
                "source_freshness_reason": freshness["reason"],
                "source_health": row.get("source_health", ""),
                "stationarity_status": row.get("stationarity_status", ""),
                "engle_granger_cointegrated": row.get("engle_granger_cointegrated", False),
                "engle_granger_trend": row.get("engle_granger_trend", False),
                "johansen_cointegrated": row.get("johansen_cointegrated", False),
                "zscore_last": row.get("zscore_last", ""),
                "zscore_roll_last": row.get("zscore_roll_last", ""),
                "volume_min": row.get("volume_min", ""),
                "backtest_settings_complete": row.get("backtest_settings_complete", False),
                "backtest_settings_missing": row.get("backtest_settings_missing", ""),
                "hypothesis_status": status,
                "hypothesis_reason": reason,
                "next_step": next_step,
                "local_data_available": local_available,
                "evidence_path": row.get("evidence_path", ""),
            }
        )
    frame = pd.DataFrame(rows, columns=HYPOTHESIS_COLUMNS)
    output = root / "reports" / "active" / "wizard_hypotheses.csv"
    summary = root / "reports" / "active" / "wizard_hypotheses.md"
    _write_csv(frame, output)
    _write_text(summary, _simple_markdown("Wizard Hypotheses", frame))
    return CommandResult(paths={"hypotheses": output, "summary_md": summary}, summary={"rows": len(frame)})


def build_wizard_diagnostic_confirmation(root: Path = ROOT) -> CommandResult:
    evidence = _ensure_wizard_evidence(root)
    rows = []
    for _, row in evidence.iterrows():
        corr_status, corr_score = _correlation_status(row)
        ecm_status, ecm_score = _ecm_status(row)
        copula_status, copula_score = _copula_status(row)
        mr_status, mr_score = _mean_reversion_status(row)
        blockers = [
            status
            for status in [corr_status, ecm_status, copula_status, mr_status]
            if status.startswith("missing") or status.startswith("weak")
        ]
        rows.append(
            {
                "pair": row.get("pair", ""),
                "exchange": row.get("exchange", ""),
                "interval": row.get("interval", ""),
                "setup_identity": row.get("setup_identity", ""),
                "exact_mode": row.get("exact_mode", ""),
                "correlation_status": corr_status,
                "ecm_status": ecm_status,
                "copula_status": copula_status,
                "mean_reversion_status": mr_status,
                "stationarity_status": row.get("stationarity_status", ""),
                "wizard_diagnostic_score": round(corr_score + ecm_score + copula_score + mr_score, 3),
                "diagnostic_blocker": ";".join(blockers),
                "source_authority": row.get("source_authority", ""),
                "source_timestamp": row.get("source_timestamp", ""),
                "source_fresh": row.get("source_fresh", False),
                "source_health": row.get("source_health", ""),
                "evidence_path": row.get("evidence_path", ""),
            }
        )
    frame = pd.DataFrame(rows, columns=DIAGNOSTIC_COLUMNS)
    output = root / "reports" / "active" / "wizard_diagnostic_confirmation.csv"
    _write_csv(frame, output)
    return CommandResult(paths={"diagnostics": output}, summary={"rows": len(frame)})


def build_wizard_discovery_triage(root: Path = ROOT) -> CommandResult:
    evidence = _ensure_wizard_evidence(root)
    rows = [_discovery_triage_row(row) for _, row in evidence.iterrows()]
    frame = pd.DataFrame(rows, columns=DISCOVERY_TRIAGE_COLUMNS)
    if not frame.empty:
        frame["_fresh_rank"] = frame["source_fresh"].map(_as_bool).astype(int)
        frame["_screen_rank"] = frame["discovery_screen_status"].eq("SCREEN_PASS").astype(int)
        frame["_research_rank"] = frame["research_quality_status"].eq("RESEARCH_READY").astype(int)
        frame = frame.sort_values(
            ["_fresh_rank", "_screen_rank", "_research_rank", "sharpe", "returns_total"],
            ascending=[False, False, False, False, False],
        ).drop(columns=["_fresh_rank", "_screen_rank", "_research_rank"])

    current = frame[frame["source_fresh"].map(_as_bool)].copy() if not frame.empty else frame.copy()
    shortlist = current[current["discovery_screen_status"].eq("SCREEN_PASS")].copy() if not current.empty else current.copy()
    copula = current[current["exact_mode"].eq("Copula")].copy() if not current.empty else current.copy()
    output = root / "reports" / "active" / "wizard_discovery_triage.csv"
    current_output = root / "reports" / "active" / "wizard_discovery_current.csv"
    shortlist_output = root / "reports" / "active" / "wizard_discovery_shortlist.csv"
    copula_output = root / "reports" / "active" / "wizard_copula_discovery_triage.csv"
    summary = root / "reports" / "active" / "wizard_discovery_triage.md"
    _write_csv(frame, output)
    _write_csv(current, current_output)
    _write_csv(shortlist, shortlist_output)
    _write_csv(copula, copula_output)
    _write_text(summary, _discovery_triage_markdown(frame, current, shortlist, copula))
    return CommandResult(
        paths={
            "wizard_discovery_triage": output,
            "wizard_discovery_current": current_output,
            "wizard_discovery_shortlist": shortlist_output,
            "wizard_copula_discovery_triage": copula_output,
            "wizard_discovery_triage_summary": summary,
        },
        summary={
            "rows": len(frame),
            "current_rows": len(current),
            "shortlist_rows": len(shortlist),
            "current_copula_rows": len(copula),
        },
    )


def _discovery_triage_row(row: pd.Series) -> dict[str, object]:
    source_fresh = _as_bool(row.get("source_fresh"))
    source_health = _text_value(row.get("source_health")).lower()
    mode_valid = _as_bool(row.get("mode_valid"))
    sharpe_pass = _as_bool(row.get("passes_sharpe_gate"))
    returns_pass = _as_bool(row.get("passes_returns_total_gt_20pct"))
    if not source_fresh:
        screen_status = "STALE_SOURCE"
    elif source_health != "healthy":
        screen_status = "CAPTURE_UNHEALTHY"
    elif not mode_valid:
        screen_status = "EXACT_MODE_INVALID"
    elif sharpe_pass and returns_pass:
        screen_status = "SCREEN_PASS"
    else:
        screen_status = "SCREEN_THRESHOLD_NOT_MET"

    exact_mode = _text_value(row.get("exact_mode"))
    signal_basis, signal_value, signal_abs, signal_status = _entry_signal_state(row, exact_mode)
    research_blockers = _discovery_research_blockers(row, screen_status)
    research_status = "RESEARCH_READY" if screen_status == "SCREEN_PASS" and not research_blockers else "RESEARCH_REVIEW"
    exchange = _text_value(row.get("exchange")).lower() or "unknown_venue"
    copula_data_status = _copula_data_status(row, exact_mode)
    if screen_status != "SCREEN_PASS":
        next_step = "refresh_or_repair_current_discovery_capture"
    elif research_status == "RESEARCH_READY":
        detail = "copula_thresholds" if exact_mode == "Copula" else "exact_mode_settings"
        next_step = f"capture_pair_detail_{detail}_then_build_{exchange}_history_cost_funding_and_slippage"
    else:
        next_step = "review_research_blockers_before_committing_venue_data_collection"
    return {
        "pair": row.get("pair", ""),
        "asset_x": row.get("asset_x", ""),
        "asset_y": row.get("asset_y", ""),
        "exchange": row.get("exchange", ""),
        "interval": row.get("interval", ""),
        "period": row.get("period", ""),
        "setup_identity": row.get("setup_identity", ""),
        "exact_mode": exact_mode,
        "spread_id": row.get("spread_id", ""),
        "strategy_id": row.get("strategy_id", ""),
        "source_authority": row.get("source_authority", ""),
        "source_timestamp": row.get("source_timestamp", ""),
        "source_age_hours": row.get("source_age_hours", ""),
        "source_fresh": source_fresh,
        "source_health": row.get("source_health", ""),
        "evidence_path": row.get("evidence_path", ""),
        "sharpe": row.get("sharpe", ""),
        "returns_total": row.get("returns_total", ""),
        "returns_total_pct": row.get("returns_total_pct", ""),
        "passes_sharpe_gate": sharpe_pass,
        "passes_returns_total_gt_20pct": returns_pass,
        "discovery_screen_status": screen_status,
        "stationarity_status": row.get("stationarity_status", ""),
        "engle_granger_cointegrated": row.get("engle_granger_cointegrated", False),
        "engle_granger_trend": row.get("engle_granger_trend", False),
        "engle_granger_pvalue": row.get("engle_granger_pvalue", ""),
        "johansen_cointegrated": row.get("johansen_cointegrated", False),
        "hurst": row.get("hurst", ""),
        "half_life": row.get("half_life", ""),
        "pearson": row.get("pearson", ""),
        "spearman": row.get("spearman", ""),
        "kendall": row.get("kendall", ""),
        "ecm_x_available": row.get("ecm_x_available", False),
        "ecm_y_available": row.get("ecm_y_available", False),
        "ecm_strength_available": row.get("ecm_strength_available", False),
        "hedge_ratio": row.get("hedge_ratio", ""),
        "signal_basis": signal_basis,
        "entry_signal_value": signal_value if signal_value is not None else "",
        "entry_signal_abs": signal_abs if signal_abs is not None else "",
        "entry_signal_status": signal_status,
        "copula": row.get("copula", ""),
        "corr_copula": row.get("corr_copula", ""),
        "u1_given_u2": row.get("u1_given_u2", ""),
        "u2_given_u1": row.get("u2_given_u1", ""),
        "copula_data_status": copula_data_status,
        "volume_x": row.get("volume_x", ""),
        "volume_y": row.get("volume_y", ""),
        "volume_min": row.get("volume_min", ""),
        "volatility_x_lt": row.get("volatility_x_lt", ""),
        "volatility_y_lt": row.get("volatility_y_lt", ""),
        "x_weighting": row.get("x_weighting", ""),
        "y_weighting": row.get("y_weighting", ""),
        "closed_trades": row.get("closed_trades", ""),
        "win_rate": row.get("win_rate", ""),
        "drawdown": row.get("drawdown", ""),
        "var": row.get("var", ""),
        "cvar": row.get("cvar", ""),
        "zero_cross": row.get("zero_cross", ""),
        "profile_match": row.get("profile_match", False),
        "ou_optimal": row.get("ou_optimal", False),
        "backtest_timestamp_utc": row.get("backtest_timestamp_utc", ""),
        "backtest_settings_complete": row.get("backtest_settings_complete", False),
        "backtest_settings_missing": row.get("backtest_settings_missing", ""),
        "research_quality_status": research_status,
        "research_blockers": ";".join(research_blockers),
        "execution_status": f"{exchange}_history_cost_funding_and_slippage_required",
        "next_step": next_step,
    }


def _entry_signal_state(row: pd.Series, exact_mode: str) -> tuple[str, float | None, float | None, str]:
    if exact_mode == "Copula":
        return "copula_conditional_probabilities", None, None, "COPULA_THRESHOLD_CAPTURE_REQUIRED"
    uses_rolling = exact_mode.endswith("(ZScoreR)")
    field = "zscore_roll_last" if uses_rolling else "zscore_last"
    value = _as_float(row.get(field))
    if value is None:
        return field, None, None, "ZSCORE_MISSING"
    absolute = abs(value)
    status = "AT_OR_BEYOND_Z2" if absolute >= ENTRY_ZSCORE_THRESHOLD else "BELOW_Z2"
    return field, value, absolute, status


def _copula_data_status(row: pd.Series, exact_mode: str) -> str:
    if exact_mode != "Copula":
        return "NOT_COPULA"
    if _as_float(row.get("u1_given_u2")) is None or _as_float(row.get("u2_given_u1")) is None:
        return "CONDITIONAL_VALUES_MISSING"
    return "CONDITIONAL_VALUES_CAPTURED"


def _discovery_research_blockers(row: pd.Series, screen_status: str) -> list[str]:
    blockers: list[str] = []
    if screen_status != "SCREEN_PASS":
        return blockers
    stationarity = _text_value(row.get("stationarity_status"))
    if stationarity == "engle_granger_with_trend":
        blockers.append("engle_granger_trend_flag")
    elif stationarity == "unconfirmed":
        blockers.append("no_engle_granger_or_johansen_confirmation")
    hurst = _as_float(row.get("hurst"))
    if hurst is None:
        blockers.append("hurst_missing")
    elif hurst >= 0.6:
        blockers.append("hurst_not_mean_reverting")
    half_life = _as_float(row.get("half_life"))
    if half_life is None or half_life <= 0:
        blockers.append("half_life_missing_or_invalid")
    volume_min = _as_float(row.get("volume_min"))
    if volume_min is None or volume_min <= 0:
        blockers.append("leg_volume_missing")
    return blockers


def _discovery_triage_markdown(
    all_rows: pd.DataFrame,
    current: pd.DataFrame,
    shortlist: pd.DataFrame,
    copula: pd.DataFrame,
) -> str:
    counts = pd.DataFrame(
        [
            {"view": "all_evidence", "rows": len(all_rows)},
            {"view": "current_fresh", "rows": len(current)},
            {"view": "screen_pass", "rows": len(shortlist)},
            {"view": "current_copula", "rows": len(copula)},
            {
                "view": "research_ready",
                "rows": int(shortlist.get("research_quality_status", pd.Series(dtype=object)).eq("RESEARCH_READY").sum()),
            },
        ]
    )
    view_columns = [
        "pair",
        "exchange",
        "exact_mode",
        "sharpe",
        "returns_total_pct",
        "stationarity_status",
        "hurst",
        "entry_signal_status",
        "research_quality_status",
        "research_blockers",
        "next_step",
    ]
    top_shortlist = shortlist[view_columns].head(25) if not shortlist.empty else shortlist
    top_copula = copula[view_columns].head(25) if not copula.empty else copula
    return "\n".join(
        [
            "# Wizard Discovery Triage",
            "",
            "Scanner data is discovery-only. This report separates a fresh threshold screen from research-quality review and never authorizes an order.",
            "",
            "## Counts",
            "",
            counts.to_markdown(index=False),
            "",
            "## Current Sharpe And Return Screen",
            "",
            top_shortlist.to_markdown(index=False) if not top_shortlist.empty else "No fresh rows passed the current screen.",
            "",
            "## Current Copula Rows",
            "",
            top_copula.to_markdown(index=False) if not top_copula.empty else "No fresh Copula rows were captured.",
            "",
            "## Interpretation Rules",
            "",
            "- `SCREEN_PASS` means only fresh, healthy, exact-mode-valid scanner evidence passed the configured Sharpe and total-return thresholds.",
            "- `RESEARCH_READY` means the available stationarity, Hurst, half-life, and leg-volume facts did not raise a research-quality blocker. It still requires pair-detail settings and venue-specific local replay.",
            "- `AT_OR_BEYOND_Z2` is an observed z-score condition, not an execution instruction. Copula rows retain conditional probabilities without inventing a vendor threshold.",
            "- Any stale or unhealthy row is retained for lineage but is not current discovery evidence.",
            "",
        ]
    )


def build_wizard_pair_detail_capture_queue(root: Path = ROOT) -> CommandResult:
    triage_path = root / "reports" / "active" / "wizard_discovery_triage.csv"
    triage = _read_csv(triage_path)
    if triage.empty:
        build_wizard_discovery_triage(root)
        triage = _read_csv(triage_path)
    if triage.empty:
        frame = pd.DataFrame(columns=PAIR_DETAIL_CAPTURE_QUEUE_COLUMNS)
    else:
        eligible = triage[
            triage.get("source_fresh", pd.Series(False, index=triage.index)).map(_as_bool)
            & triage.get("discovery_screen_status", pd.Series("", index=triage.index)).astype(str).eq("SCREEN_PASS")
            & ~triage.get("backtest_settings_complete", pd.Series(False, index=triage.index)).map(_as_bool)
        ].copy()
        rows = [_pair_detail_capture_queue_row(row) for _, row in eligible.iterrows()]
        frame = pd.DataFrame(rows, columns=PAIR_DETAIL_CAPTURE_QUEUE_COLUMNS)
        if not frame.empty:
            priority_rank = frame["priority"].map({"P1": 1, "P2": 2, "P3": 3}).fillna(9)
            frame = frame.assign(_priority_rank=priority_rank).sort_values(
                ["_priority_rank", "sharpe", "returns_total_pct"],
                ascending=[True, False, False],
            ).drop(columns=["_priority_rank"])
    output = root / "reports" / "active" / "wizard_pair_detail_capture_queue.csv"
    summary = root / "reports" / "active" / "wizard_pair_detail_capture_queue.md"
    _write_csv(frame, output)
    _write_text(summary, _pair_detail_capture_queue_markdown(frame))
    return CommandResult(
        paths={"wizard_pair_detail_capture_queue": output, "wizard_pair_detail_capture_queue_summary": summary},
        summary={
            "rows": len(frame),
            "p1_rows": int(frame.get("priority", pd.Series(dtype=object)).eq("P1").sum()) if not frame.empty else 0,
        },
    )


def build_wizard_mode_matrix_capture_queue(root: Path = ROOT) -> CommandResult:
    """Queue every canonical Wizard mode for each fresh shortlisted pair.

    Scanner-selected modes describe discovery evidence. This queue makes the
    cross-mode comparison work explicit without assuming the other six modes
    use the same settings or return profile.
    """

    active = root / "reports" / "active"
    shortlist_path = active / "wizard_discovery_shortlist.csv"
    shortlist = _read_csv(shortlist_path)
    if shortlist.empty and not shortlist_path.exists():
        build_wizard_discovery_triage(root)
        shortlist = _read_csv(shortlist_path)
    captures = _read_csv(active / "crypto_wizards_pair_page_capture_settings.csv")
    if shortlist.empty:
        frame = pd.DataFrame(columns=WIZARD_MODE_MATRIX_CAPTURE_QUEUE_COLUMNS)
    else:
        working = shortlist.copy()
        source_fresh = working.get("source_fresh", pd.Series(True, index=working.index)).map(_as_bool)
        if source_fresh.any():
            working = working.loc[source_fresh].copy()
        working["_research_rank"] = working.get(
            "research_quality_status", pd.Series("", index=working.index)
        ).astype(str).eq("RESEARCH_READY").astype(int)
        working["_sharpe"] = _numeric_series(working.get("sharpe")).fillna(-999)
        working["_returns"] = _numeric_series(working.get("returns_total")).fillna(-999)
        dedupe_columns = [column for column in ["pair", "exchange", "interval", "period"] if column in working.columns]
        if dedupe_columns:
            working = working.sort_values(["_research_rank", "_sharpe", "_returns"], ascending=[False, False, False]).drop_duplicates(
                subset=dedupe_columns,
                keep="first",
            )
        rows = []
        for _, source in working.iterrows():
            for mode in WIZARD_MODE_MAP.values():
                rows.append(_wizard_mode_matrix_capture_row(source, mode=mode, captures=captures, shortlist_path=shortlist_path))
        frame = pd.DataFrame(rows, columns=WIZARD_MODE_MATRIX_CAPTURE_QUEUE_COLUMNS)
        if not frame.empty:
            priority_rank = frame["priority"].map({"P1": 1, "P2": 2, "P3": 3}).fillna(9)
            capture_rank = frame["capture_status"].eq("VALIDATED_MODE_SETTINGS_CAPTURED").astype(int)
            frame = frame.assign(_priority_rank=priority_rank, _capture_rank=capture_rank).sort_values(
                ["_priority_rank", "pair", "interval", "_capture_rank", "exact_mode"],
                ascending=[True, True, True, True, True],
            ).drop(columns=["_priority_rank", "_capture_rank"])
    output = active / "wizard_mode_matrix_capture_queue.csv"
    summary = active / "wizard_mode_matrix_capture_queue.md"
    _write_csv(frame, output)
    _write_text(summary, _wizard_mode_matrix_capture_queue_markdown(frame))
    return CommandResult(
        paths={"wizard_mode_matrix_capture_queue": output, "wizard_mode_matrix_capture_queue_summary": summary},
        summary={
            "rows": int(len(frame)),
            "pairs": int(frame[["pair", "exchange", "interval", "period"]].drop_duplicates().shape[0]) if not frame.empty else 0,
            "capture_required": int(frame.get("capture_status", pd.Series(dtype=object)).eq("CAPTURE_REQUIRED").sum()) if not frame.empty else 0,
        },
    )


def _wizard_mode_matrix_capture_row(
    source: pd.Series,
    *,
    mode: str,
    captures: pd.DataFrame,
    shortlist_path: Path,
) -> dict[str, object]:
    pair = _text_value(source.get("pair"))
    interval = _text_value(source.get("interval"))
    period = _as_int(source.get("period"))
    setup_identity = _wizard_setup_identity(pair, interval, period, mode)
    spread_id, strategy_id = ids_from_exact_mode(mode)
    candidate = pd.Series(
        {
            "setup_identity": setup_identity,
            "pair": pair,
            "asset_x": source.get("asset_x", ""),
            "asset_y": source.get("asset_y", ""),
            "interval": interval,
            "period": period,
            "exact_mode": mode,
        }
    )
    capture = _matching_handoff_row(captures, candidate)
    captured = capture is not None and _as_bool(capture.get("capture_confirmed")) and _as_bool(capture.get("backtest_settings_complete"))
    source_mode = _text_value(source.get("exact_mode"))
    required_settings = _settings_required_for_mode(mode, "")
    return {
        "priority": "P1" if _text_value(source.get("research_quality_status")) == "RESEARCH_READY" else "P2",
        "setup_identity": setup_identity,
        "source_setup_identity": source.get("setup_identity", ""),
        "pair": pair,
        "asset_x": source.get("asset_x", ""),
        "asset_y": source.get("asset_y", ""),
        "exchange": source.get("exchange", ""),
        "interval": interval,
        "period": period or "",
        "exact_mode": mode,
        "spread_id": spread_id or "",
        "strategy_id": strategy_id or "",
        "source_exact_mode": source_mode,
        "mode_switch_required": mode != source_mode,
        "source_sharpe": source.get("sharpe", ""),
        "source_returns_total_pct": source.get("returns_total_pct", ""),
        "research_quality_status": source.get("research_quality_status", ""),
        "source_timestamp": source.get("source_timestamp", ""),
        "source_fresh": source.get("source_fresh", False),
        "capture_status": "VALIDATED_MODE_SETTINGS_CAPTURED" if captured else "CAPTURE_REQUIRED",
        "required_settings": ";".join(required_settings),
        "required_mode_inputs": ";".join(mode_requirements(mode)),
        "mode_fidelity_target": "local_formula_approximation_then_bounded_vendor_custom_series_proof",
        "acceptance_eligible": False,
        "paper_or_execution_eligible": False,
        "next_step": "retain_validated_capture_for_mode_comparison" if captured else f"open_pair_detail_switch_to_{_wizard_mode_slug(mode)}_and_capture_visible_settings",
        "evidence_path": ";".join(
            dict.fromkeys(
                path
                for path in [str(shortlist_path), _text_value(source.get("evidence_path")), _text_value(capture.get("capture_evidence_path")) if capture is not None else ""]
                if path
            )
        ),
    }


def _wizard_mode_matrix_capture_queue_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "# Wizard Mode-Matrix Capture Queue\n\nNo fresh Wizard shortlist rows are available.\n"
    counts = frame["capture_status"].value_counts().rename_axis("capture_status").reset_index(name="rows")
    columns = [
        "priority",
        "pair",
        "exchange",
        "interval",
        "exact_mode",
        "mode_switch_required",
        "capture_status",
        "next_step",
    ]
    return "\n".join(
        [
            "# Wizard Mode-Matrix Capture Queue",
            "",
            "Every fresh shortlisted pair/timeframe is expanded into the seven canonical Wizard modes. The source scanner mode remains discovery evidence; all other modes must be visibly selected and captured before a local comparison can run.",
            "",
            "## Capture Counts",
            "",
            counts.to_markdown(index=False),
            "",
            "## Queue",
            "",
            frame[columns].to_markdown(index=False),
            "",
            "## Rule",
            "",
            "- Capture visible settings after switching the dashboard to the target mode; never reuse thresholds from a different mode.",
            "- Matrix rows are research work items only. They do not create acceptance, paper, or execution authority.",
            "",
        ]
    )


def _wizard_mode_slug(mode: str) -> str:
    return "".join(character.lower() if character.isalnum() else "_" for character in mode).strip("_")


def _pair_detail_capture_queue_row(row: pd.Series) -> dict[str, object]:
    exact_mode = _text_value(row.get("exact_mode"))
    missing = _settings_required_for_mode(exact_mode, row.get("backtest_settings_missing"))
    research_ready = _text_value(row.get("research_quality_status")) == "RESEARCH_READY"
    priority = "P1" if research_ready else "P2"
    exchange = _text_value(row.get("exchange")) or "dashboard_venue"
    detail = "Copula chart and thresholds" if exact_mode == "Copula" else "strategy settings and backtest controls"
    return {
        "priority": priority,
        "pair": row.get("pair", ""),
        "asset_x": row.get("asset_x", ""),
        "asset_y": row.get("asset_y", ""),
        "exchange": exchange,
        "interval": row.get("interval", ""),
        "period": row.get("period", ""),
        "setup_identity": row.get("setup_identity", ""),
        "exact_mode": exact_mode,
        "spread_id": row.get("spread_id", ""),
        "strategy_id": row.get("strategy_id", ""),
        "sharpe": row.get("sharpe", ""),
        "returns_total_pct": row.get("returns_total_pct", ""),
        "research_quality_status": row.get("research_quality_status", ""),
        "research_blockers": row.get("research_blockers", ""),
        "source_timestamp": row.get("source_timestamp", ""),
        "source_fresh": row.get("source_fresh", False),
        "source_authority": row.get("source_authority", ""),
        "required_settings": ";".join(missing),
        "capture_method": f"Crypto Wizards pair detail -> Backtest: record visible {detail} and export or capture the request payload if available",
        "api_retrieval_status": "scanner_api_is_discovery_only;parameterized_backtest_endpoint_cannot_retrieve_existing_dashboard_settings",
        "capture_status": "NEEDS_DASHBOARD_PAIR_DETAIL_CAPTURE",
        "next_step": f"capture_settings_then_replay_{exact_mode.lower().replace(' ', '_').replace('(', '').replace(')', '')}_on_{exchange}",
        "evidence_path": row.get("evidence_path", ""),
    }


def _settings_required_for_mode(exact_mode: str, source_missing: object) -> list[str]:
    missing = [part.strip() for part in _text_value(source_missing).split(";") if part.strip()]
    required = [
        "entry_thresholds",
        "entry_direction_mapping",
        "exit_thresholds",
        "wizard_cost_assumptions",
        "x_y_weighting_or_hedge_ratio",
    ]
    if exact_mode in {"Static (Spread)", "Static (ZScoreR)", "OU (Spread)", "OU (ZScoreR)"}:
        required.append("static_hedge_ratio")
    if exact_mode in {"Dyn (Spread)", "Dyn (ZScoreR)"}:
        required.extend(["dynamic_hedge_ratio_method", "dynamic_hedge_ratio_window"])
    if exact_mode in {"OU (Spread)", "OU (ZScoreR)"}:
        required.extend(["ou_mu", "ou_sigma"])
    if exact_mode.endswith("(ZScoreR)"):
        required.append("zscore_roll_window")
    if exact_mode == "Copula":
        required.extend(
            [
                "copula_family",
                "copula_signal_type",
                "copula_thresholds",
                "copula_directional_entry_rule",
                "copula_exit_rule",
            ]
        )
    return list(dict.fromkeys([*missing, *required]))


def _pair_detail_capture_queue_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "# Wizard Pair Detail Capture Queue\n\nNo fresh screen-pass rows currently need a settings capture.\n"
    columns = [
        "priority",
        "pair",
        "exchange",
        "exact_mode",
        "sharpe",
        "returns_total_pct",
        "research_quality_status",
        "required_settings",
        "capture_status",
        "next_step",
    ]
    return "\n".join(
        [
            "# Wizard Pair Detail Capture Queue",
            "",
            "This queue exists because the scanner provides discovery metrics but not the already-configured dashboard backtest settings.",
            "",
            frame[columns].to_markdown(index=False),
            "",
            "## Rule",
            "",
            "Do not call the parameterized Wizard backtest endpoint with guessed values and label it as the dashboard setup. Capture the displayed settings or an observed request first, then replay those exact settings locally with venue-specific economics.",
            "",
        ]
    )


def build_wizard_pair_settings_capture_template(root: Path = ROOT) -> CommandResult:
    """Create a mode-specific template for recording visible Wizard settings.

    The template lives outside active evidence so blank rows cannot affect the
    evidence builder. The importer below is the only route from a completed
    template into the active pair-page capture surface.
    """

    active = root / "reports" / "active"
    control_summary_path = active / "wizard_control_plane_summary.json"
    control_queue_path = active / "wizard_sweep_settings_capture_queue.csv"
    if control_summary_path.exists():
        queue_path = control_queue_path
        queue = _read_csv(queue_path)
        if not queue.empty:
            actionable = queue.get("actionable", pd.Series(False, index=queue.index)).map(_as_bool)
            queue = queue[actionable].copy()
            queue = queue.rename(
                columns={
                    "wizard_exchange": "exchange",
                    "timeframe": "interval",
                    "asset_x": "asset_x",
                    "asset_y": "asset_y",
                }
            )
    else:
        queue_path = active / "wizard_mode_matrix_capture_queue.csv"
        queue = _read_csv(queue_path)
        if queue.empty and not queue_path.exists():
            detail_queue_path = active / "wizard_pair_detail_capture_queue.csv"
            if not detail_queue_path.exists():
                build_wizard_pair_detail_capture_queue(root)
            build_wizard_mode_matrix_capture_queue(root)
            queue = _read_csv(queue_path)
        if queue.empty:
            queue_path = active / "wizard_pair_detail_capture_queue.csv"
            queue = _read_csv(queue_path)

    rows: list[dict[str, object]] = []
    capture_status = queue.get("capture_status", pd.Series("CAPTURE_REQUIRED", index=queue.index)).astype(str)
    for _, row in queue.loc[capture_status.ne("VALIDATED_MODE_SETTINGS_CAPTURED")].iterrows():
        template_row = {column: "" for column in WIZARD_PAIR_SETTINGS_CAPTURE_TEMPLATE_COLUMNS}
        template_row["config_schema_version"] = WIZARD_CONFIG_SCHEMA_VERSION
        for field in [
            "discovery_config_hash",
            "candidate_config_hash",
            "setup_identity",
            "pair",
            "asset_x",
            "asset_y",
            "exchange",
            "interval",
            "period",
            "exact_mode",
            "spread_type",
            "spread_id",
            "strategy_id",
            "required_settings",
        ]:
            template_row[field] = row.get(field, "")
        if not template_row["setup_identity"]:
            template_row["setup_identity"] = str(row.get("candidate_config_hash", ""))[:16]
        rows.append(template_row)

    frame = pd.DataFrame(rows, columns=WIZARD_PAIR_SETTINGS_CAPTURE_TEMPLATE_COLUMNS)
    template_dir = root / "reports" / "templates"
    output = template_dir / "wizard_pair_settings_capture_template.csv"
    summary = template_dir / "wizard_pair_settings_capture_template.md"
    validation_output = active / "wizard_pair_settings_capture_validation.csv"
    _write_csv(frame, output)
    _write_text(summary, _wizard_pair_settings_capture_template_markdown(frame))
    if not validation_output.exists():
        _write_csv(pd.DataFrame(columns=WIZARD_PAIR_SETTINGS_CAPTURE_VALIDATION_COLUMNS), validation_output)
    return CommandResult(
        paths={
            "wizard_pair_settings_capture_template": output,
            "wizard_pair_settings_capture_template_summary": summary,
            "wizard_pair_settings_capture_validation": validation_output,
        },
        summary={"rows": int(len(frame))},
    )


def import_wizard_pair_settings_capture(input_path: Path, root: Path = ROOT) -> CommandResult:
    """Validate a user-recorded settings capture and publish only valid rows.

    The input is a record of settings visibly observed on the current Wizard
    pair page. It cannot be a guessed parameter set. Invalid rows remain in the
    validation report but are excluded from active evidence.
    """

    if not input_path.exists() or not input_path.is_file():
        raise ValueError(f"Wizard settings capture CSV not found: {input_path}")
    if input_path.suffix.lower() != ".csv":
        raise ValueError("Wizard settings capture import requires a CSV file")
    try:
        source = pd.read_csv(input_path, dtype=object).fillna("")
    except pd.errors.EmptyDataError:
        source = pd.DataFrame(columns=WIZARD_PAIR_SETTINGS_CAPTURE_TEMPLATE_COLUMNS)
    missing_columns = [column for column in WIZARD_PAIR_SETTINGS_CAPTURE_TEMPLATE_COLUMNS if column not in source.columns]
    if missing_columns:
        raise ValueError(f"Wizard settings capture CSV is missing columns: {','.join(missing_columns)}")

    validated_rows: list[dict[str, object]] = []
    accepted_rows: list[dict[str, object]] = []
    for row_number, (_, source_row) in enumerate(source.iterrows(), start=2):
        canonical, blockers = _canonical_wizard_pair_settings_capture(source_row)
        valid = not blockers
        validated_rows.append(
            {
                "input_row_number": row_number,
                "setup_identity": canonical.get("setup_identity", ""),
                "pair": canonical.get("pair", ""),
                "exchange": canonical.get("exchange", ""),
                "interval": canonical.get("interval", ""),
                "exact_mode": canonical.get("exact_mode", ""),
                "discovery_config_hash": canonical.get("discovery_config_hash", ""),
                "candidate_config_hash": canonical.get("candidate_config_hash", ""),
                "settings_config_hash": canonical.get("settings_config_hash", ""),
                "capture_timestamp_utc": canonical.get("capture_timestamp_utc", ""),
                "capture_evidence_path": canonical.get("capture_evidence_path", ""),
                "valid": valid,
                "validation_blockers": ";".join(blockers),
                "accepted_into_active_capture": valid,
                "next_step": "rebuild_wizard_research_pack" if valid else "repair_capture_and_reimport",
                "evidence_path": str(input_path),
            }
        )
        if valid:
            accepted_rows.append(canonical)

    active = root / "reports" / "active"
    active_output = active / "crypto_wizards_pair_page_capture_settings.csv"
    validation_output = active / "wizard_pair_settings_capture_validation.csv"
    validation_summary = active / "wizard_pair_settings_capture_validation.md"
    existing = _read_csv(active_output)
    accepted = pd.DataFrame(accepted_rows)
    if not accepted.empty:
        accepted["source_path"] = accepted["capture_evidence_path"]
        accepted["evidence_path"] = accepted["capture_evidence_path"]
        accepted["source_capture_kind"] = "visible_dashboard_settings_capture"
        accepted["capture_status"] = "CURRENT_SETTINGS_CAPTURED_VALIDATED"
        accepted["backtest_settings_complete"] = True
        accepted["backtest_settings_missing"] = ""
        accepted_keys = set(accepted["setup_identity"].map(_text_value))
        if not existing.empty and "setup_identity" in existing.columns:
            existing = existing[~existing["setup_identity"].map(_text_value).isin(accepted_keys)].copy()
        merged = pd.concat([existing, accepted], ignore_index=True, sort=False)
    else:
        merged = existing

    validation = pd.DataFrame(validated_rows, columns=WIZARD_PAIR_SETTINGS_CAPTURE_VALIDATION_COLUMNS)
    _write_csv(merged, active_output)
    _write_csv(validation, validation_output)
    _write_text(validation_summary, _wizard_pair_settings_capture_validation_markdown(validation, input_path, active_output))
    return CommandResult(
        paths={
            "wizard_pair_page_settings_capture": active_output,
            "wizard_pair_settings_capture_validation": validation_output,
            "wizard_pair_settings_capture_validation_summary": validation_summary,
        },
        summary={
            "input_rows": int(len(source)),
            "accepted_rows": int(len(accepted_rows)),
            "blocked_rows": int(len(source) - len(accepted_rows)),
        },
    )


def _canonical_wizard_pair_settings_capture(row: pd.Series) -> tuple[dict[str, object], list[str]]:
    canonical = {column: row.get(column, "") for column in WIZARD_PAIR_SETTINGS_CAPTURE_TEMPLATE_COLUMNS}
    exact_mode = normalize_exact_mode(canonical.get("exact_mode"))
    expected_spread_id, expected_strategy_id = ids_from_exact_mode(exact_mode)
    canonical["exact_mode"] = exact_mode
    canonical["spread_id"] = _as_int(canonical.get("spread_id")) or expected_spread_id or ""
    canonical["strategy_id"] = _as_int(canonical.get("strategy_id")) or expected_strategy_id or ""
    canonical["period"] = _as_int(canonical.get("period")) or ""

    pair = _text_value(canonical.get("pair"))
    asset_x = _text_value(canonical.get("asset_x"))
    asset_y = _text_value(canonical.get("asset_y"))
    canonical["pair"] = pair or (f"{asset_x}/{asset_y}" if asset_x and asset_y else "")
    canonical["asset_x"] = asset_x
    canonical["asset_y"] = asset_y
    canonical["exchange"] = _text_value(canonical.get("exchange"))
    canonical["interval"] = _canonical_interval(_text_value(canonical.get("interval")))
    canonical["capture_evidence_path"] = _text_value(canonical.get("capture_evidence_path"))
    canonical["pair_page_url"] = _text_value(canonical.get("pair_page_url"))
    canonical["setup_identity"] = _text_value(canonical.get("setup_identity")) or _wizard_setup_identity(
        canonical["pair"],
        canonical["interval"],
        _as_int(canonical.get("period")),
        exact_mode,
    )

    blockers: list[str] = []
    for field in ["pair", "asset_x", "asset_y", "exchange", "interval", "period", "exact_mode", "capture_evidence_path"]:
        if not _text_value(canonical.get(field)):
            blockers.append(f"missing_{field}")
    capture_timestamp = _parse_timestamp(canonical.get("capture_timestamp_utc"))
    if capture_timestamp is None:
        blockers.append("invalid_capture_timestamp_utc")
    else:
        canonical["capture_timestamp_utc"] = capture_timestamp.isoformat()
    if not _as_bool(canonical.get("capture_confirmed")):
        blockers.append("capture_not_explicitly_confirmed")
    if expected_spread_id is None or expected_strategy_id is None:
        blockers.append("invalid_exact_mode")
    elif canonical["spread_id"] != expected_spread_id or canonical["strategy_id"] != expected_strategy_id:
        blockers.append("mode_ids_do_not_match_exact_mode")

    settings_complete, settings_missing = _backtest_settings_status(canonical, exact_mode, strict=True)
    if not settings_complete:
        blockers.extend(f"missing_{field}" for field in settings_missing)
    canonical["config_schema_version"] = WIZARD_CONFIG_SCHEMA_VERSION
    canonical["settings_config_hash"] = _wizard_settings_config_hash(canonical)
    return canonical, list(dict.fromkeys(blockers))


def _wizard_settings_config_hash(row: dict[str, object]) -> str:
    strategy_id = _as_int(row.get("strategy_id"))
    spread_id = _as_int(row.get("spread_id"))
    configuration = WizardRunConfiguration(
        source="visible_dashboard_settings_capture",
        symbol_1=row.get("asset_x"),
        symbol_2=row.get("asset_y"),
        wizard_exchange=row.get("exchange"),
        interval=row.get("interval"),
        period=_as_int(row.get("period")),
        strategy={1: "Spread", 2: "ZScoreRoll", 3: "Copula"}.get(strategy_id),
        spread_type=_text_value(row.get("spread_type")) or {1: "Dynamic", 2: "OU", 3: "Static"}.get(spread_id),
        exact_mode=row.get("exact_mode"),
        roll_window=_as_int(row.get("zscore_window")),
        entry_long_operator=_text_value(row.get("entry_long_operator")) or None,
        entry_long_level=_as_float(row.get("entry_long_value")),
        entry_short_operator=_text_value(row.get("entry_short_operator")) or None,
        entry_short_level=_as_float(row.get("entry_short_value")),
        exit_long_operator=_text_value(row.get("exit_long_operator")) or None,
        exit_long_level=_as_float(row.get("exit_long_value")),
        exit_short_operator=_text_value(row.get("exit_short_operator")) or None,
        exit_short_level=_as_float(row.get("exit_short_value")),
        copula_entry_lower=_as_float(row.get("copula_entry_lower")),
        copula_entry_upper=_as_float(row.get("copula_entry_upper")),
        copula_exit_lower=_as_float(row.get("copula_exit_lower")),
        copula_exit_upper=_as_float(row.get("copula_exit_upper")),
        forced_close_mode="n_periods" if (_as_int(row.get("exit_n_periods")) or 0) > 0 else "ignore",
        exit_n_periods=_as_int(row.get("exit_n_periods")),
        stop_loss_rate=_as_float(row.get("stop_loss_rate")),
        ecm_deviation_min=_as_float(row.get("ecm_deviation_min")),
        correlation_strength_min=_as_float(row.get("correlation_strength_min")),
        commission_rate=_as_float(row.get("commission_rate")),
        slippage_rate=_as_float(row.get("slippage_rate")),
        x_weighting=_as_float(row.get("x_weighting")),
        y_weighting=_as_float(row.get("y_weighting")),
        capital_weighting=_text_value(row.get("capital_weighting") or row.get("capital_weighting_asset")) or None,
        metric_mode=_text_value(row.get("metric_mode")) or None,
        simulation_runs=_as_int(row.get("simulation_runs")),
        live_update=_text_value(row.get("live_update")) or None,
        input_data_hash=_text_value(row.get("candidate_config_hash")) or None,
    )
    return configuration.config_hash


def _wizard_pair_settings_capture_template_markdown(frame: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# Wizard Pair Settings Capture Template",
            "",
            f"- rows to capture: {len(frame)}",
            "- one row is one exact Wizard pair, timeframe, period, and mode",
            "- record only settings visibly shown on the current pair page or observed request",
            "- include a timestamped screenshot, export, or request record in `capture_evidence_path`",
            "- use `capture_confirmed=true` only after the row was checked against the live dashboard",
            "- do not invent settings; incomplete rows are rejected by the importer",
            "",
        ]
    )


def _wizard_pair_settings_capture_validation_markdown(
    frame: pd.DataFrame,
    input_path: Path,
    active_output: Path,
) -> str:
    accepted = int(frame.get("accepted_into_active_capture", pd.Series(dtype=bool)).map(_as_bool).sum()) if not frame.empty else 0
    return "\n".join(
        [
            "# Wizard Pair Settings Capture Validation",
            "",
            f"- input: `{input_path}`",
            f"- active capture: `{active_output}`",
            f"- input rows: {len(frame)}",
            f"- accepted rows: {accepted}",
            f"- blocked rows: {len(frame) - accepted}",
            "",
            "Only accepted rows are written to active Wizard pair-page evidence. Importing settings never grants acceptance, paper, or execution authority; rebuild the research pack to recompute the exact-mode replay handoff.",
            "",
        ]
    )


def build_wizard_replay_handoff(root: Path = ROOT) -> CommandResult:
    active = root / "reports" / "active"
    shortlist_path = active / "wizard_discovery_shortlist.csv"
    shortlist = _read_csv(shortlist_path)
    if shortlist.empty and not shortlist_path.exists():
        build_wizard_discovery_triage(root)
        shortlist = _read_csv(shortlist_path)

    capture_path = active / "wizard_pair_detail_capture_queue.csv"
    capture_queue = _read_csv(capture_path)
    if capture_queue.empty and not capture_path.exists():
        build_wizard_pair_detail_capture_queue(root)
        capture_queue = _read_csv(capture_path)

    multi_venue_path = active / "multi_venue_history_readiness.csv"
    if not multi_venue_path.exists():
        multi_venue_path = active / "multi_venue_history_readiness_2026-06-25.csv"
    multi_venue = _read_csv(multi_venue_path)
    binance_path = active / "binance_spot_pair_readiness.csv"
    binance = _read_csv(binance_path)

    rows = [
        _wizard_replay_handoff_row(
            candidate,
            capture_queue=capture_queue,
            multi_venue=multi_venue,
            binance=binance,
            shortlist_path=shortlist_path,
            multi_venue_path=multi_venue_path,
            binance_path=binance_path,
        )
        for _, candidate in shortlist.iterrows()
    ]
    frame = pd.DataFrame(rows, columns=WIZARD_REPLAY_HANDOFF_COLUMNS)
    if not frame.empty:
        priority_rank = frame["priority"].map({"P1": 1, "P2": 2, "P3": 3}).fillna(9)
        ready_rank = frame["replay_status"].eq("READY_FOR_COSTED_LOCAL_REPLAY").astype(int)
        frame = frame.assign(_priority_rank=priority_rank, _ready_rank=ready_rank).sort_values(
            ["_ready_rank", "_priority_rank", "sharpe", "returns_total_pct"],
            ascending=[False, True, False, False],
        ).drop(columns=["_priority_rank", "_ready_rank"])
    output = active / "wizard_replay_handoff.csv"
    summary = active / "wizard_replay_handoff.md"
    _write_csv(frame, output)
    _write_text(summary, _wizard_replay_handoff_markdown(frame))
    return CommandResult(
        paths={"wizard_replay_handoff": output, "wizard_replay_handoff_summary": summary},
        summary={
            "rows": len(frame),
            "ready_for_exploratory_local_replay": int(frame.get("exploratory_replay_status", pd.Series(dtype=object)).eq("READY_FOR_EXPLORATORY_LOCAL_REPLAY").sum()) if not frame.empty else 0,
            "ready_for_costed_local_replay": int(frame.get("replay_status", pd.Series(dtype=object)).eq("READY_FOR_COSTED_LOCAL_REPLAY").sum()) if not frame.empty else 0,
            "settings_blocked": int(frame.get("replay_status", pd.Series(dtype=object)).eq("BLOCKED_SETTINGS_CAPTURE").sum()) if not frame.empty else 0,
            "research_review": int(frame.get("replay_status", pd.Series(dtype=object)).eq("RESEARCH_REVIEW_REQUIRED").sum()) if not frame.empty else 0,
        },
    )


def _wizard_replay_handoff_row(
    candidate: pd.Series,
    *,
    capture_queue: pd.DataFrame,
    multi_venue: pd.DataFrame,
    binance: pd.DataFrame,
    shortlist_path: Path,
    multi_venue_path: Path,
    binance_path: Path,
) -> dict[str, object]:
    capture = _matching_handoff_row(capture_queue, candidate)
    venue = _matching_handoff_row(multi_venue, candidate)
    binance_row = _matching_handoff_row(binance, candidate)
    exact_mode = _text_value(candidate.get("exact_mode"))
    research_ready = _text_value(candidate.get("research_quality_status")) == "RESEARCH_READY"
    settings_complete = _as_bool(candidate.get("backtest_settings_complete"))
    source_missing = _text_value(candidate.get("backtest_settings_missing"))
    required_settings = _settings_required_for_mode(exact_mode, source_missing)
    if settings_complete:
        settings_status = "CURRENT_PAIR_DETAIL_SETTINGS_CAPTURED"
        settings_missing = ""
    elif capture is not None:
        settings_status = _text_value(capture.get("capture_status")) or "NEEDS_DASHBOARD_PAIR_DETAIL_CAPTURE"
        settings_missing = _text_value(capture.get("required_settings")) or ";".join(required_settings)
    else:
        settings_status = "MISSING_PAIR_DETAIL_SETTINGS"
        settings_missing = ";".join(required_settings)

    generic_history_status = _text_value(venue.get("readiness_status")) if venue is not None else ""
    generic_cost_status = _text_value(venue.get("cost_model_status")) if venue is not None else ""
    generic_slippage_status = _text_value(venue.get("slippage_model_status")) if venue is not None else ""
    generic_funding_status = _text_value(venue.get("funding_or_borrow_status")) if venue is not None else ""
    binance_history_status = _text_value(binance_row.get("history_status")) if binance_row is not None else ""
    venue_history_status = binance_history_status or generic_history_status or "MISSING_VENUE_HISTORY_PLAN"
    venue_cost_status = _text_value(binance_row.get("cost_model_status")) if binance_row is not None else generic_cost_status
    venue_slippage_status = _text_value(binance_row.get("slippage_status")) if binance_row is not None else generic_slippage_status
    venue_funding_status = _text_value(binance_row.get("funding_borrow_status")) if binance_row is not None else generic_funding_status
    venue_research_execution_status = _text_value(binance_row.get("research_execution_status")) if binance_row is not None else "research_collection_only"

    history_ready = venue_history_status in {"history_ready_needs_settings_and_cost_model", "ready_for_replay"}
    cost_ready = _replay_cost_status_ready(venue_cost_status)
    slippage_ready = _replay_slippage_status_ready(venue_slippage_status)
    funding_ready = _replay_funding_status_ready(venue_funding_status)
    exploratory_blockers: list[str] = []
    if not research_ready:
        exploratory_blockers.append("research_quality_review_required")
    if not settings_complete:
        exploratory_blockers.append("exact_wizard_settings_missing")
    if not history_ready:
        exploratory_blockers.append("matching_venue_history_missing")

    if not research_ready:
        exploratory_replay_status = "RESEARCH_REVIEW_REQUIRED"
        exploratory_next_step = "review_research_blockers_before_spending_more_on_venue_data"
    elif not settings_complete:
        exploratory_replay_status = "BLOCKED_SETTINGS_CAPTURE"
        exploratory_next_step = _text_value(capture.get("next_step")) if capture is not None else "capture_exact_wizard_settings_before_local_replay"
    elif not history_ready:
        exploratory_replay_status = "BLOCKED_HISTORY_FETCH"
        exploratory_next_step = _text_value(venue.get("next_step")) if venue is not None else "build_matching_venue_history_for_exact_mode_replay"
    else:
        exploratory_replay_status = "READY_FOR_EXPLORATORY_LOCAL_REPLAY"
        exploratory_next_step = "run_provisional_exact_mode_local_replay_with_cost_sensitivity"

    blockers = list(exploratory_blockers)
    if not cost_ready:
        blockers.append("venue_cost_model_not_validated")
    if not slippage_ready:
        blockers.append("venue_slippage_not_calibrated")
    if not funding_ready:
        blockers.append("venue_funding_or_borrow_not_validated")

    if not research_ready:
        replay_status = "RESEARCH_REVIEW_REQUIRED"
        next_step = exploratory_next_step
    elif not settings_complete:
        replay_status = "BLOCKED_SETTINGS_CAPTURE"
        next_step = exploratory_next_step
    elif not history_ready:
        replay_status = "BLOCKED_HISTORY_FETCH"
        next_step = exploratory_next_step
    elif not cost_ready or not slippage_ready or not funding_ready:
        replay_status = "BLOCKED_VENUE_ECONOMICS"
        next_step = "calibrate_venue_cost_slippage_and_funding_or_borrow_before_costed_local_replay"
    else:
        replay_status = "READY_FOR_COSTED_LOCAL_REPLAY"
        next_step = "run_point_in_time_costed_exact_mode_local_replay"

    paths = [
        str(shortlist_path),
        _text_value(candidate.get("evidence_path")),
        _text_value(capture.get("evidence_path")) if capture is not None else "",
        str(multi_venue_path) if venue is not None else "",
        _text_value(venue.get("evidence_path")) if venue is not None else "",
        str(binance_path) if binance_row is not None else "",
        _text_value(binance_row.get("evidence_path")) if binance_row is not None else "",
    ]
    return {
        "priority": "P1" if research_ready else "P2",
        "setup_identity": candidate.get("setup_identity", ""),
        "pair": candidate.get("pair", ""),
        "asset_x": candidate.get("asset_x", ""),
        "asset_y": candidate.get("asset_y", ""),
        "exchange": candidate.get("exchange", ""),
        "interval": candidate.get("interval", ""),
        "period": candidate.get("period", ""),
        "exact_mode": exact_mode,
        "spread_id": candidate.get("spread_id", ""),
        "strategy_id": candidate.get("strategy_id", ""),
        "source_timestamp": candidate.get("source_timestamp", ""),
        "source_fresh": candidate.get("source_fresh", False),
        "source_authority": candidate.get("source_authority", ""),
        "sharpe": candidate.get("sharpe", ""),
        "returns_total_pct": candidate.get("returns_total_pct", ""),
        "research_quality_status": candidate.get("research_quality_status", ""),
        "research_blockers": candidate.get("research_blockers", ""),
        "entry_signal_status": candidate.get("entry_signal_status", ""),
        "settings_status": settings_status,
        "settings_missing": settings_missing,
        "required_settings": ";".join(required_settings),
        "venue_history_status": venue_history_status,
        "venue_cost_status": venue_cost_status,
        "venue_slippage_status": venue_slippage_status,
        "venue_funding_or_borrow_status": venue_funding_status,
        "venue_research_execution_status": venue_research_execution_status,
        "exploratory_replay_status": exploratory_replay_status,
        "exploratory_replay_blockers": ";".join(dict.fromkeys(exploratory_blockers)),
        "exploratory_next_step": exploratory_next_step,
        "exploratory_cost_policy": "validated_venue_economics" if cost_ready and slippage_ready and funding_ready else "provisional_cost_sensitivity_required",
        "replay_status": replay_status,
        "replay_blockers": ";".join(dict.fromkeys(blockers)),
        "next_step": next_step,
        "acceptance_authority": "local_point_in_time_costed_replay_only",
        "evidence_path": ";".join(dict.fromkeys(path for path in paths if path)),
    }


def _matching_handoff_row(frame: pd.DataFrame, candidate: pd.Series) -> pd.Series | None:
    if frame.empty:
        return None
    setup_identity = _text_value(candidate.get("setup_identity"))
    if setup_identity and "setup_identity" in frame.columns:
        matches = frame[frame["setup_identity"].map(_text_value).eq(setup_identity)]
        if not matches.empty:
            return matches.iloc[0]
    pair_key = _text_value(candidate.get("pair")).upper()
    mode_key = _text_value(candidate.get("exact_mode")).lower()
    if not pair_key or "pair" not in frame.columns:
        return None
    matches = frame[frame["pair"].map(_text_value).str.upper().eq(pair_key)]
    if "exact_mode" in matches.columns:
        matches = matches[matches["exact_mode"].map(_text_value).str.lower().eq(mode_key)]
    return matches.iloc[0] if not matches.empty else None


def _replay_cost_status_ready(status: str) -> bool:
    normalized = status.lower()
    return normalized.startswith("available_") or normalized in {"validated", "costed_local_replay"}


def _replay_slippage_status_ready(status: str) -> bool:
    normalized = status.lower()
    return normalized.startswith("available_") or normalized in {"validated", "costed_local_replay"}


def _replay_funding_status_ready(status: str) -> bool:
    normalized = status.lower()
    return normalized.startswith("available_") or normalized.startswith("current_") or normalized == "validated"


def _wizard_replay_handoff_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "# Wizard Replay Handoff\n\nNo current Wizard screen-pass setups are available.\n"
    exploratory_counts = frame["exploratory_replay_status"].value_counts().rename_axis("exploratory_replay_status").reset_index(name="rows")
    acceptance_counts = frame["replay_status"].value_counts().rename_axis("acceptance_replay_status").reset_index(name="rows")
    columns = [
        "priority",
        "pair",
        "exchange",
        "exact_mode",
        "research_quality_status",
        "settings_status",
        "venue_history_status",
        "exploratory_replay_status",
        "exploratory_cost_policy",
        "replay_status",
        "replay_blockers",
        "next_step",
    ]
    return "\n".join(
        [
            "# Wizard Replay Handoff",
            "",
            "One row represents one fresh, exact Wizard setup. It joins discovery quality, captured settings, matching venue history, and venue economics before a local replay may begin.",
            "",
            "## Status Counts",
            "",
            "### Exploratory Replay",
            "",
            exploratory_counts.to_markdown(index=False),
            "",
            "### Acceptance Replay",
            "",
            acceptance_counts.to_markdown(index=False),
            "",
            "## Current Setups",
            "",
            frame[columns].to_markdown(index=False),
            "",
            "## Rules",
            "",
            "- `READY_FOR_EXPLORATORY_LOCAL_REPLAY` allows a provisional research replay with explicit cost sensitivity. It is never acceptance, paper, or execution approval.",
            "- `READY_FOR_COSTED_LOCAL_REPLAY` is the stricter acceptance-replay state. It still requires model and strategy gates before any paper route can be considered.",
            "- A mode-specific settings capture is required. Scanner metrics and Copula conditional probabilities are never substituted for missing entry, exit, weighting, or cost settings.",
            "- Venue cost, slippage, and funding or borrow evidence must be validated before a replay can contribute to acceptance evidence.",
            "",
        ]
    )


def build_wizard_mode_replay_capability(root: Path = ROOT) -> CommandResult:
    """Expose whether each exact Wizard mode has enough inputs for local research.

    The result is intentionally a capability board, not a performance board.
    A ready row only permits a local formula approximation in the research lane;
    vendor parity and acceptance remain separate gates.
    """

    active = root / "reports" / "active"
    handoff_path = active / "wizard_replay_handoff.csv"
    handoff = _read_csv(handoff_path)
    if handoff.empty and not handoff_path.exists():
        build_wizard_replay_handoff(root)
        handoff = _read_csv(handoff_path)
    captures_path = active / "crypto_wizards_pair_page_capture_settings.csv"
    captures = _read_csv(captures_path)
    rows = [_wizard_mode_replay_capability_row(candidate, captures=captures, handoff_path=handoff_path) for _, candidate in handoff.iterrows()]
    frame = pd.DataFrame(rows, columns=WIZARD_MODE_REPLAY_CAPABILITY_COLUMNS)
    if not frame.empty:
        priority_rank = frame["priority"].map({"P1": 1, "P2": 2, "P3": 3}).fillna(9)
        ready_rank = frame["research_replay_eligible"].map(_as_bool).astype(int)
        frame = frame.assign(_priority_rank=priority_rank, _ready_rank=ready_rank).sort_values(
            ["_ready_rank", "_priority_rank", "pair", "exact_mode"],
            ascending=[False, True, True, True],
        ).drop(columns=["_priority_rank", "_ready_rank"])
    output = active / "wizard_mode_replay_capability.csv"
    summary = active / "wizard_mode_replay_capability.md"
    _write_csv(frame, output)
    _write_text(summary, _wizard_mode_replay_capability_markdown(frame))
    return CommandResult(
        paths={"wizard_mode_replay_capability": output, "wizard_mode_replay_capability_summary": summary},
        summary={
            "rows": int(len(frame)),
            "research_replay_eligible": int(frame.get("research_replay_eligible", pd.Series(dtype=bool)).map(_as_bool).sum()) if not frame.empty else 0,
            "settings_blocked": int(frame.get("mode_capability_status", pd.Series(dtype=object)).eq("BLOCKED_SETTINGS_CAPTURE").sum()) if not frame.empty else 0,
            "mode_inputs_blocked": int(frame.get("mode_capability_status", pd.Series(dtype=object)).eq("BLOCKED_MODE_INPUTS").sum()) if not frame.empty else 0,
        },
    )


def _wizard_mode_replay_capability_row(
    candidate: pd.Series,
    *,
    captures: pd.DataFrame,
    handoff_path: Path,
) -> dict[str, object]:
    capture = _matching_handoff_row(captures, candidate)
    captured_mode = normalize_exact_mode(candidate.get("exact_mode", ""))
    requirements = list(mode_requirements(captured_mode or candidate.get("exact_mode", "")))
    if not captured_mode:
        requirements = ["supported_exact_mode"]
    captured_inputs: list[str] = []
    missing_inputs: list[str] = []
    for field in requirements:
        value = capture.get(field) if capture is not None else None
        present = _as_bool(value) if field == "capture_confirmed" else _first_present({field: value}, [field]) is not None
        if present:
            captured_inputs.append(field)
        else:
            missing_inputs.append(field)

    research_ready = _text_value(candidate.get("research_quality_status")) == "RESEARCH_READY"
    history_ready = _text_value(candidate.get("venue_history_status")) in {"history_ready_needs_settings_and_cost_model", "ready_for_replay"}
    capture_confirmed = capture is not None and _as_bool(capture.get("capture_confirmed"))
    capture_complete = capture is not None and _as_bool(capture.get("backtest_settings_complete"))
    if not captured_mode:
        status = "BLOCKED_UNSUPPORTED_MODE"
        next_step = "capture_a_supported_exact_wizard_mode"
    elif not research_ready:
        status = "RESEARCH_REVIEW_REQUIRED"
        next_step = "resolve_research_quality_blockers_before_mode_replay"
    elif capture is None or not capture_confirmed or not capture_complete:
        status = "BLOCKED_SETTINGS_CAPTURE"
        next_step = "capture_and_import_visible_exact_mode_settings"
    elif missing_inputs:
        status = "BLOCKED_MODE_INPUTS"
        next_step = "repair_mode_specific_settings_capture_and_reimport"
    elif not history_ready:
        status = "BLOCKED_HISTORY_FETCH"
        next_step = "build_matching_point_in_time_two_leg_history"
    else:
        status = "READY_FOR_RESEARCH_MODE_REPLAY"
        next_step = "run_local_formula_approximation_with_explicit_cost_sensitivity"

    capture_path = _text_value(capture.get("capture_evidence_path")) if capture is not None else ""
    paths = [
        str(handoff_path),
        _text_value(candidate.get("evidence_path")),
        capture_path,
    ]
    return {
        "priority": candidate.get("priority", ""),
        "setup_identity": candidate.get("setup_identity", ""),
        "pair": candidate.get("pair", ""),
        "asset_x": candidate.get("asset_x", ""),
        "asset_y": candidate.get("asset_y", ""),
        "exchange": candidate.get("exchange", ""),
        "interval": candidate.get("interval", ""),
        "period": candidate.get("period", ""),
        "exact_mode": candidate.get("exact_mode", ""),
        "research_quality_status": candidate.get("research_quality_status", ""),
        "venue_history_status": candidate.get("venue_history_status", ""),
        "mode_capability_status": status,
        "mode_fidelity_status": "local_formula_approximation",
        "mode_fidelity_rule": "vendor_custom_series_proof_required_for_vendor_exact_or_promotion",
        "settings_capture_status": "VALIDATED_CURRENT_CAPTURE" if capture_confirmed and capture_complete else "MISSING_OR_INCOMPLETE_CAPTURE",
        "required_mode_inputs": ";".join(requirements),
        "captured_mode_inputs": ";".join(captured_inputs),
        "missing_mode_inputs": ";".join(missing_inputs),
        "research_replay_eligible": status == "READY_FOR_RESEARCH_MODE_REPLAY",
        "acceptance_eligible": False,
        "paper_or_execution_eligible": False,
        "next_step": next_step,
        "settings_capture_evidence_path": capture_path,
        "evidence_path": ";".join(dict.fromkeys(path for path in paths if path)),
    }


def _wizard_mode_replay_capability_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "# Wizard Mode Replay Capability\n\nNo current Wizard replay-handoff rows are available.\n"
    counts = frame["mode_capability_status"].value_counts().rename_axis("mode_capability_status").reset_index(name="rows")
    columns = [
        "priority",
        "pair",
        "exchange",
        "exact_mode",
        "mode_capability_status",
        "missing_mode_inputs",
        "next_step",
    ]
    return "\n".join(
        [
            "# Wizard Mode Replay Capability",
            "",
            "This board verifies the inputs needed to run the matching local research formula for each Wizard mode. A ready row remains research-only; it does not establish vendor parity, acceptance, paper readiness, or execution approval.",
            "",
            "## Status Counts",
            "",
            counts.to_markdown(index=False),
            "",
            "## Current Setups",
            "",
            frame[columns].to_markdown(index=False),
            "",
            "## Rule",
            "",
            "- Local formulas must use captured settings and point-in-time data only.",
            "- A bounded Wizard custom-series proof remains required before anything can be labeled vendor-exact.",
            "",
        ]
    )


def build_wizard_exploratory_cost_sensitivity(
    root: Path = ROOT,
    *,
    profile_path: Path | None = None,
) -> CommandResult:
    """Write reproducible research-only cost scenarios for exact-mode handoffs.

    This is intentionally an exploratory layer. The profile provides sensitivity
    bounds while exact account fees, slippage, funding, and borrow costs are
    still unvalidated. None of these rows can contribute acceptance, paper, or
    execution authority.
    """

    active = root / "reports" / "active"
    handoff_path = active / "wizard_replay_handoff.csv"
    handoff = _read_csv(handoff_path)
    if handoff.empty and not handoff_path.exists():
        build_wizard_replay_handoff(root)
        handoff = _read_csv(handoff_path)

    selected_profile_path = profile_path or root / "config" / EXPLORATORY_COST_PROFILE_FILENAME
    if profile_path is None and not selected_profile_path.exists():
        selected_profile_path = ROOT / "config" / EXPLORATORY_COST_PROFILE_FILENAME
    profile = _load_exploratory_cost_profile(selected_profile_path)
    rows: list[dict[str, object]] = []
    for _, handoff_row in handoff.iterrows():
        rows.extend(
            _exploratory_cost_sensitivity_rows(
                handoff_row,
                profile=profile,
                handoff_path=handoff_path,
                profile_path=selected_profile_path,
            )
        )

    frame = pd.DataFrame(rows, columns=EXPLORATORY_COST_SENSITIVITY_COLUMNS)
    if not frame.empty:
        state_rank = frame["scenario_status"].eq("READY_FOR_PROVISIONAL_COST_SENSITIVITY").astype(int)
        case_rank = pd.to_numeric(frame["cost_case_rank"], errors="coerce").fillna(99)
        frame = frame.assign(_state_rank=state_rank, _case_rank=case_rank).sort_values(
            ["_state_rank", "pair", "exact_mode", "_case_rank"],
            ascending=[False, True, True, True],
        ).drop(columns=["_state_rank", "_case_rank"])

    output = active / "wizard_exploratory_cost_sensitivity.csv"
    summary = active / "wizard_exploratory_cost_sensitivity.md"
    _write_csv(frame, output)
    _write_text(summary, _wizard_exploratory_cost_sensitivity_markdown(frame, selected_profile_path))
    ready = (
        int(frame["scenario_status"].eq("READY_FOR_PROVISIONAL_COST_SENSITIVITY").sum())
        if not frame.empty
        else 0
    )
    return CommandResult(
        paths={"wizard_exploratory_cost_sensitivity": output, "wizard_exploratory_cost_sensitivity_summary": summary},
        summary={
            "rows": int(len(frame)),
            "ready_for_provisional_sensitivity": ready,
            "acceptance_eligible_rows": 0,
            "profile_path": str(selected_profile_path),
        },
    )


def _load_exploratory_cost_profile(path: Path) -> dict[str, object]:
    if not path.exists():
        raise ValueError(f"exploratory cost sensitivity profile is missing: {path}")
    try:
        profile = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"exploratory cost sensitivity profile is not valid JSON: {path}") from exc
    if not isinstance(profile, dict):
        raise ValueError("exploratory cost sensitivity profile must be an object")
    if _as_bool(profile.get("acceptance_eligible")):
        raise ValueError("exploratory cost sensitivity profile may not be acceptance eligible")
    if _as_bool(profile.get("paper_or_execution_eligible")):
        raise ValueError("exploratory cost sensitivity profile may not be paper or execution eligible")
    profile_id = _text_value(profile.get("profile_id"))
    profile_version = _text_value(profile.get("profile_version"))
    scenarios = profile.get("scenarios")
    if not profile_id or not profile_version or not isinstance(scenarios, list) or not scenarios:
        raise ValueError("exploratory cost sensitivity profile requires profile_id, profile_version, and scenarios")

    required_numeric = (
        "taker_fee_bps",
        "slippage_bps",
        "execution_risk_bps",
        "funding_or_borrow_bps_per_day",
        "partial_fill_probability",
        "partial_fill_fraction",
        "partial_fill_penalty_bps",
    )
    seen_cases: set[str] = set()
    for scenario in scenarios:
        if not isinstance(scenario, dict):
            raise ValueError("exploratory cost sensitivity scenarios must be objects")
        case = _text_value(scenario.get("cost_case"))
        if not case or case in seen_cases:
            raise ValueError("exploratory cost sensitivity scenarios require unique cost_case values")
        seen_cases.add(case)
        for field in required_numeric:
            try:
                value = float(scenario[field])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"exploratory cost sensitivity scenario {case} is missing numeric {field}") from exc
            if value < 0:
                raise ValueError(f"exploratory cost sensitivity scenario {case} has negative {field}")
        if float(scenario["partial_fill_probability"]) > 1 or float(scenario["partial_fill_fraction"]) > 1:
            raise ValueError(f"exploratory cost sensitivity scenario {case} has invalid partial-fill values")
        if _as_bool(scenario.get("acceptance_eligible")) or _as_bool(scenario.get("paper_or_execution_eligible")):
            raise ValueError(f"exploratory cost sensitivity scenario {case} cannot grant acceptance or execution authority")
    return profile


def _exploratory_cost_sensitivity_rows(
    handoff: pd.Series,
    *,
    profile: dict[str, object],
    handoff_path: Path,
    profile_path: Path,
) -> list[dict[str, object]]:
    replay_status = _text_value(handoff.get("exploratory_replay_status"))
    ready = replay_status == "READY_FOR_EXPLORATORY_LOCAL_REPLAY"
    if ready:
        scenario_status = "READY_FOR_PROVISIONAL_COST_SENSITIVITY"
        next_step = "run_provisional_exact_mode_local_replay_with_cost_sensitivity"
    else:
        scenario_status = "BLOCKED_BY_EXPLORATORY_REPLAY_HANDOFF"
        next_step = _text_value(handoff.get("exploratory_next_step")) or "resolve_exact_mode_replay_blockers"

    bars_per_day = _bars_per_day(_text_value(handoff.get("interval")))
    profile_id = _text_value(profile.get("profile_id"))
    profile_version = _text_value(profile.get("profile_version"))
    authority = _text_value(profile.get("cost_assumption_authority")) or "provisional_research_sensitivity_only"
    rows: list[dict[str, object]] = []
    scenarios = profile.get("scenarios", [])
    for rank, scenario in enumerate(scenarios, start=1):
        assert isinstance(scenario, dict)  # Validated by _load_exploratory_cost_profile.
        fee = float(scenario["taker_fee_bps"])
        slippage = float(scenario["slippage_bps"])
        execution_risk = float(scenario["execution_risk_bps"])
        daily_carry = float(scenario["funding_or_borrow_bps_per_day"])
        partial_probability = float(scenario["partial_fill_probability"])
        partial_fraction = float(scenario["partial_fill_fraction"])
        partial_penalty = float(scenario["partial_fill_penalty_bps"])
        expected_partial_fill = partial_probability * (1.0 - partial_fraction) * partial_penalty
        one_way_leg = fee + slippage + execution_risk + expected_partial_fill
        evidence_paths = [
            str(handoff_path),
            str(profile_path),
            _text_value(handoff.get("evidence_path")),
        ]
        rows.append(
            {
                "setup_identity": handoff.get("setup_identity", ""),
                "pair": handoff.get("pair", ""),
                "exchange": handoff.get("exchange", ""),
                "interval": handoff.get("interval", ""),
                "exact_mode": handoff.get("exact_mode", ""),
                "research_quality_status": handoff.get("research_quality_status", ""),
                "settings_status": handoff.get("settings_status", ""),
                "venue_history_status": handoff.get("venue_history_status", ""),
                "exploratory_replay_status": replay_status,
                "cost_profile_id": profile_id,
                "cost_profile_version": profile_version,
                "cost_case": scenario["cost_case"],
                "cost_case_rank": rank,
                "scenario_status": scenario_status,
                "taker_fee_bps": fee,
                "slippage_bps": slippage,
                "execution_risk_bps": execution_risk,
                "funding_or_borrow_bps_per_day": daily_carry,
                "bars_per_day": bars_per_day,
                "partial_fill_probability": partial_probability,
                "partial_fill_fraction": partial_fraction,
                "partial_fill_penalty_bps": partial_penalty,
                "expected_partial_fill_bps": expected_partial_fill,
                "one_way_leg_cost_bps": one_way_leg,
                "two_leg_round_trip_execution_cost_bps": 4.0 * one_way_leg,
                "symmetric_two_leg_daily_carry_bps": 2.0 * daily_carry,
                "cost_assumption_authority": authority,
                "acceptance_eligible": False,
                "paper_or_execution_eligible": False,
                "acceptance_requirements": "validate_account_fee_tier;calibrate_venue_slippage;validate_funding_or_borrow",
                "next_step": next_step,
                "evidence_path": ";".join(dict.fromkeys(path for path in evidence_paths if path)),
            }
        )
    return rows


def _bars_per_day(interval: str) -> int:
    normalized = interval.strip().lower().replace(" ", "")
    mapping = {
        "daily": 1,
        "1d": 1,
        "1day": 1,
        "4h": 6,
        "4hour": 6,
        "4hours": 6,
        "1h": 24,
        "1hour": 24,
        "1hours": 24,
        "15m": 96,
        "15min": 96,
        "15mins": 96,
        "5m": 288,
        "5min": 288,
        "5mins": 288,
    }
    return mapping.get(normalized, 1)


def _wizard_exploratory_cost_sensitivity_markdown(frame: pd.DataFrame, profile_path: Path) -> str:
    if frame.empty:
        return "# Wizard Exploratory Cost Sensitivity\n\nNo current exact-mode handoff rows are available.\n"
    status_counts = frame["scenario_status"].value_counts().rename_axis("scenario_status").reset_index(name="rows")
    columns = [
        "pair",
        "exchange",
        "exact_mode",
        "cost_case",
        "scenario_status",
        "two_leg_round_trip_execution_cost_bps",
        "symmetric_two_leg_daily_carry_bps",
        "acceptance_eligible",
        "next_step",
    ]
    return "\n".join(
        [
            "# Wizard Exploratory Cost Sensitivity",
            "",
            "This is a reproducible sensitivity plan for an exact-mode local research replay. It is intentionally not a venue fee schedule or an acceptance model.",
            "",
            f"- profile: `{profile_path}`",
            "- authority: provisional research sensitivity only",
            "- execution risk remains a cost component; it is not an exploratory research gate",
            "",
            "## Status Counts",
            "",
            status_counts.to_markdown(index=False),
            "",
            "## Planned Scenarios",
            "",
            frame[columns].to_markdown(index=False),
            "",
            "## Rules",
            "",
            "- A scenario can run only after the exact Wizard settings and matching venue history are present.",
            "- Every row is `acceptance_eligible=false` and `paper_or_execution_eligible=false`.",
            "- Acceptance still requires account-specific fee evidence, venue-calibrated slippage, and current funding or borrow evidence.",
            "- The zero-cost case is a gross-return upper bound, never an economic result.",
            "",
        ]
    )


def build_wizard_local_parity(root: Path = ROOT) -> CommandResult:
    evidence = _ensure_wizard_evidence(root)
    local_index = _local_history_index(root)
    rows = []
    for _, row in evidence.iterrows():
        local_path = _matching_local_path(local_index, row)
        wizard_last = _history_last_values(Path(str(row.get("source_path", ""))))
        local_last = _history_last_values(local_path) if local_path else {}
        z_delta = _delta(wizard_last.get("zscore"), local_last.get("zscore"))
        rz_delta = _delta(wizard_last.get("rolling_zscore"), local_last.get("rolling_zscore"))
        status, reason = _parity_status(local_path, z_delta, rz_delta)
        rows.append(
            {
                "pair": row.get("pair", ""),
                "interval": row.get("interval", ""),
                "setup_identity": row.get("setup_identity", ""),
                "exact_mode": row.get("exact_mode", ""),
                "wizard_source_path": row.get("source_path", ""),
                "local_source_path": str(local_path or ""),
                "wizard_zscore_last": wizard_last.get("zscore", ""),
                "local_zscore_last": local_last.get("zscore", ""),
                "wizard_rolling_zscore_last": wizard_last.get("rolling_zscore", ""),
                "local_rolling_zscore_last": local_last.get("rolling_zscore", ""),
                "zscore_delta": z_delta if z_delta is not None else "",
                "rolling_zscore_delta": rz_delta if rz_delta is not None else "",
                "parity_status": status,
                "parity_reason": reason,
                "evidence_path": ";".join([p for p in [str(row.get("source_path", "")), str(local_path or "")] if p]),
            }
        )
    frame = pd.DataFrame(rows, columns=PARITY_COLUMNS)
    output = root / "reports" / "active" / "wizard_vs_local_parity_report.csv"
    summary = root / "reports" / "active" / "wizard_vs_local_parity_report.md"
    _write_csv(frame, output)
    _write_text(summary, _simple_markdown("Wizard vs Local Parity", frame))
    return CommandResult(paths={"parity": output, "summary_md": summary}, summary={"rows": len(frame)})


def build_wizard_strategy_alignment_report(root: Path = ROOT) -> CommandResult:
    evidence = _ensure_wizard_evidence(root)
    rows = []
    for _, row in evidence.iterrows():
        rows.append(
            {
                "pair": row.get("pair", ""),
                "interval": row.get("interval", ""),
                "dashboard_recommended_strategy": row.get("dashboard_recommended_strategy", ""),
                "exact_mode": row.get("exact_mode", ""),
                "spread_id": row.get("spread_id", ""),
                "strategy_id": row.get("strategy_id", ""),
                "local_strategy_id": row.get("local_strategy_id", ""),
                "local_strategy_name": row.get("local_strategy_name", ""),
                "local_strategy_family": row.get("local_strategy_family", ""),
                "strategy_mapping_status": row.get("strategy_mapping_status", ""),
                "source_first_action": _source_first_action(row),
                "evidence_path": row.get("evidence_path", ""),
            }
        )
    frame = pd.DataFrame(rows, columns=STRATEGY_ALIGNMENT_COLUMNS)
    output = root / "reports" / "active" / "wizard_strategy_alignment_report.csv"
    summary = root / "reports" / "active" / "wizard_strategy_alignment_report.md"
    _write_csv(frame, output)
    _write_text(summary, _simple_markdown("Wizard Strategy Alignment", frame))
    mapped = int(frame["local_strategy_id"].astype(str).str.strip().ne("").sum()) if not frame.empty else 0
    return CommandResult(paths={"alignment": output, "summary_md": summary}, summary={"rows": len(frame), "mapped": mapped})


def build_wizard_exact_mode_capture_queue(root: Path = ROOT) -> CommandResult:
    evidence = _ensure_wizard_evidence(root)
    if evidence.empty:
        frame = pd.DataFrame(columns=EXACT_MODE_CAPTURE_QUEUE_COLUMNS)
    else:
        valid_pairs = set(
            _evidence_pair_interval_key(row)
            for _, row in evidence[evidence.get("mode_valid", pd.Series(False, index=evidence.index)).astype(bool)].iterrows()
        )
        blocked = evidence[
            (~evidence.get("mode_valid", pd.Series(False, index=evidence.index)).astype(bool))
            & _passes_sharpe_gate(evidence).astype(bool)
            & evidence.get("passes_returns_total_gt_20pct", pd.Series(False, index=evidence.index)).astype(bool)
        ].copy()
        if not blocked.empty and valid_pairs:
            blocked["_pair_interval"] = [_evidence_pair_interval_key(row) for _, row in blocked.iterrows()]
            blocked = blocked[~blocked["_pair_interval"].isin(valid_pairs)]
        if not blocked.empty:
            blocked["_rank_sharpe"] = _numeric_series(blocked.get("sharpe")).fillna(-999)
            blocked["_rank_return"] = _numeric_series(blocked.get("returns_total")).fillna(-999)
            blocked = blocked.sort_values(["_rank_sharpe", "_rank_return"], ascending=[False, False])
            blocked = blocked.drop_duplicates(subset=["pair", "interval"], keep="first")
        rows = []
        for rank, (_, row) in enumerate(blocked.iterrows(), start=1):
            source_path = str(row.get("source_path", "") or "")
            pair_page_url = _pair_page_url_from_source(root, source_path)
            rows.append(
                {
                    "priority_rank": rank,
                    "pair": row.get("pair", ""),
                    "asset_x": row.get("asset_x", ""),
                    "asset_y": row.get("asset_y", ""),
                    "interval": row.get("interval", ""),
                    "sharpe": row.get("sharpe", ""),
                    "returns_total": row.get("returns_total", ""),
                    "returns_total_pct": row.get("returns_total_pct", ""),
                    "mode_blocker": row.get("mode_blocker", "missing_exact_mode"),
                    "pair_page_url": pair_page_url,
                    "required_capture_fields": "selected_strategy_value;spread_id;strategy_id;exact_mode;period;interval;backtest_settings",
                    "operator_action": _capture_action(pair_page_url),
                    "source_path": source_path,
                    "evidence_path": row.get("evidence_path", ""),
                }
            )
        frame = pd.DataFrame(rows, columns=EXACT_MODE_CAPTURE_QUEUE_COLUMNS)
    output = root / "reports" / "active" / "wizard_exact_mode_capture_queue.csv"
    summary = root / "reports" / "active" / "wizard_exact_mode_capture_queue.md"
    _write_csv(frame, output)
    _write_text(summary, _simple_markdown("Wizard Exact Mode Capture Queue", frame))
    return CommandResult(paths={"exact_mode_capture_queue": output, "summary_md": summary}, summary={"rows": len(frame)})


def build_wizard_research_pack(root: Path = ROOT) -> CommandResult:
    evidence = build_wizard_evidence(root)
    hypotheses = build_wizard_hypotheses(root)
    diagnostics = build_wizard_diagnostic_confirmation(root)
    triage = build_wizard_discovery_triage(root)
    pair_detail_queue = build_wizard_pair_detail_capture_queue(root)
    mode_matrix_queue = build_wizard_mode_matrix_capture_queue(root)
    settings_template = build_wizard_pair_settings_capture_template(root)
    replay_handoff = build_wizard_replay_handoff(root)
    mode_capability = build_wizard_mode_replay_capability(root)
    exploratory_cost_sensitivity = build_wizard_exploratory_cost_sensitivity(root)
    parity = build_wizard_local_parity(root)
    alignment = build_wizard_strategy_alignment_report(root)
    queue = build_wizard_exact_mode_capture_queue(root)
    return CommandResult(
        paths={
            "wizard_evidence": evidence.paths["wizard_evidence"],
            "wizard_evidence_summary": evidence.paths["summary_md"],
            "wizard_hypotheses": hypotheses.paths["hypotheses"],
            "wizard_hypotheses_summary": hypotheses.paths["summary_md"],
            "wizard_diagnostics": diagnostics.paths["diagnostics"],
            "wizard_discovery_triage": triage.paths["wizard_discovery_triage"],
            "wizard_discovery_current": triage.paths["wizard_discovery_current"],
            "wizard_discovery_shortlist": triage.paths["wizard_discovery_shortlist"],
            "wizard_copula_discovery_triage": triage.paths["wizard_copula_discovery_triage"],
            "wizard_discovery_triage_summary": triage.paths["wizard_discovery_triage_summary"],
            "wizard_pair_detail_capture_queue": pair_detail_queue.paths["wizard_pair_detail_capture_queue"],
            "wizard_pair_detail_capture_queue_summary": pair_detail_queue.paths["wizard_pair_detail_capture_queue_summary"],
            "wizard_mode_matrix_capture_queue": mode_matrix_queue.paths["wizard_mode_matrix_capture_queue"],
            "wizard_mode_matrix_capture_queue_summary": mode_matrix_queue.paths["wizard_mode_matrix_capture_queue_summary"],
            "wizard_pair_settings_capture_template": settings_template.paths["wizard_pair_settings_capture_template"],
            "wizard_pair_settings_capture_template_summary": settings_template.paths["wizard_pair_settings_capture_template_summary"],
            "wizard_replay_handoff": replay_handoff.paths["wizard_replay_handoff"],
            "wizard_replay_handoff_summary": replay_handoff.paths["wizard_replay_handoff_summary"],
            "wizard_mode_replay_capability": mode_capability.paths["wizard_mode_replay_capability"],
            "wizard_mode_replay_capability_summary": mode_capability.paths["wizard_mode_replay_capability_summary"],
            "wizard_exploratory_cost_sensitivity": exploratory_cost_sensitivity.paths["wizard_exploratory_cost_sensitivity"],
            "wizard_exploratory_cost_sensitivity_summary": exploratory_cost_sensitivity.paths["wizard_exploratory_cost_sensitivity_summary"],
            "wizard_parity": parity.paths["parity"],
            "wizard_parity_summary": parity.paths["summary_md"],
            "wizard_strategy_alignment": alignment.paths["alignment"],
            "wizard_strategy_alignment_summary": alignment.paths["summary_md"],
            "wizard_exact_mode_capture_queue": queue.paths["exact_mode_capture_queue"],
            "wizard_exact_mode_capture_queue_summary": queue.paths["summary_md"],
        },
        summary={
            "evidence_rows": evidence.summary["rows"],
            "hypotheses_rows": hypotheses.summary["rows"],
            "diagnostic_rows": diagnostics.summary["rows"],
            "discovery_triage_rows": triage.summary["rows"],
            "discovery_shortlist_rows": triage.summary["shortlist_rows"],
            "current_copula_rows": triage.summary["current_copula_rows"],
            "pair_detail_capture_queue_rows": pair_detail_queue.summary["rows"],
            "mode_matrix_capture_queue_rows": mode_matrix_queue.summary["rows"],
            "pair_settings_capture_template_rows": settings_template.summary["rows"],
            "replay_handoff_rows": replay_handoff.summary["rows"],
            "replay_handoff_exploratory_ready_rows": replay_handoff.summary["ready_for_exploratory_local_replay"],
            "replay_handoff_ready_rows": replay_handoff.summary["ready_for_costed_local_replay"],
            "mode_replay_capability_rows": mode_capability.summary["rows"],
            "mode_replay_research_ready_rows": mode_capability.summary["research_replay_eligible"],
            "exploratory_cost_sensitivity_rows": exploratory_cost_sensitivity.summary["rows"],
            "exploratory_cost_sensitivity_ready_rows": exploratory_cost_sensitivity.summary["ready_for_provisional_sensitivity"],
            "parity_rows": parity.summary["rows"],
            "strategy_alignment_rows": alignment.summary["rows"],
            "exact_mode_capture_queue_rows": queue.summary["rows"],
        },
    )


def _wizard_evidence_rows(root: Path, *, now: datetime) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen_paths: set[str] = set()
    hourly_pair_queue = _read_csv(root / "reports" / "active" / "wizard_hourly_database" / "wizard_hourly_pair_queue.csv")
    for active_path in sorted((root / "reports" / "active").glob("crypto_wizards*.csv")):
        frame = _read_csv(active_path)
        for _, active_row in frame.iterrows():
            source_path = str(active_row.get("source_path", active_row.get("raw_path", "")) or "")
            if source_path:
                seen_paths.add(source_path)
            active_values = {k: v for k, v in active_row.to_dict().items() if pd.notna(v)}
            payload_row = _row_from_source_path(root, source_path, active_row=active_values) if source_path else {}
            merged = {**payload_row, **active_values}
            rows.append(
                _normalize_evidence_row(
                    merged,
                    active_path,
                    source_path or str(active_path),
                    _active_wizard_source_system(active_path),
                    root=root,
                    now=now,
                )
            )
    hourly_alias = root / "reports" / "active" / "wizard_hourly_database" / "wizard_hourly_current_alias.csv"
    if hourly_alias.exists():
        frame = _read_csv(hourly_alias)
        for _, hourly_row in frame.iterrows():
            source_system = _hourly_wizard_source_system(hourly_row)
            if not source_system:
                continue
            enriched = {k: v for k, v in hourly_row.to_dict().items() if pd.notna(v)}
            pair_queue_row = _match_hourly_pair_queue_row(hourly_pair_queue, str(hourly_row.get("pair", "") or ""))
            if pair_queue_row is not None:
                enriched.setdefault("dashboard_selector_score", pair_queue_row.get("top_annualized_return_pct", ""))
                strategy = _normalize_mode_label(str(hourly_row.get("strategy", "") or ""))
                top_strategy = _normalize_mode_label(str(pair_queue_row.get("top_strategy", "") or ""))
                enriched.setdefault("dashboard_pair_rank", 1 if strategy and strategy == top_strategy else 2)
            rows.append(
                _normalize_evidence_row(
                    enriched,
                    hourly_alias,
                    str(hourly_alias),
                    source_system,
                    root=root,
                    now=now,
                )
            )
    rows.extend(_collected_live_scanner_evidence_rows(root, now=now))
    pair_detail_dir = root / "data" / "raw" / "pair_details"
    for path in sorted(pair_detail_dir.glob("*.json")):
        rel = path.relative_to(root).as_posix() if _is_relative_to(path, root) else str(path)
        if rel in seen_paths or str(path) in seen_paths:
            continue
        try:
            row = _row_from_payload_path(path)
        except Exception:
            continue
        if _looks_like_wizard_source(row, path):
            rows.append(_normalize_evidence_row(row, path, rel, "crypto_wizards_pair_detail", root=root, now=now))
    return rows


def _collected_live_scanner_evidence_rows(root: Path, *, now: datetime) -> list[dict[str, object]]:
    path = root / "data" / "collected" / "wizard_scanner_static_spread_visible_live" / "wizard_scanner_static_spread_visible_live_latest.json"
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(payload, dict):
        return []
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return []
    evidence_rows: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        normalized = _normalize_collected_scanner_row(row, path, captured_at=payload.get("captured_at"))
        if normalized:
            evidence_rows.append(
                _normalize_evidence_row(
                    normalized,
                    path,
                    str(path.relative_to(root) if _is_relative_to(path, root) else path),
                    "crypto_wizards_live_scanner_capture",
                    root=root,
                    now=now,
                )
            )
    return evidence_rows


def _normalize_collected_scanner_row(
    row: dict[str, object],
    source_path: Path,
    *,
    captured_at: object = None,
) -> dict[str, object]:
    pair = _scanner_text(row.get("pair"))
    pair_x = _scanner_text(row.get("pair_x"))
    pair_y = _scanner_text(row.get("pair_y"))
    if not pair and pair_x and pair_y:
        pair = f"{pair_x}/{pair_y}"
    stationarity = _scanner_text(row.get("stationarity")) or ""
    reward = _scanner_text(row.get("reward")) or ""
    corr_pct = _first_number_before_token(stationarity, "corr")
    hurst = _first_number_before_token(stationarity, "hurst")
    half_life = _first_number_before_token(stationarity, "half life")
    returns_total_pct = _first_number_before_token(reward, "return")
    sharpe = _first_number_before_token(reward, "sharpe")
    dashboard_strategy = _humanize_collected_strategy(_scanner_text(row.get("strategy")) or "")
    return {
        "pair": pair or "",
        "asset_x": pair_x or "",
        "asset_y": pair_y or "",
        "exchange": "dydx",
        "interval": "",
        "period": None,
        "dashboard_recommended_strategy": dashboard_strategy,
        "dashboard_pair_rank": _as_int(row.get("row_index")),
        "dashboard_selector_score": sharpe,
        "sharpe": sharpe,
        "returns_total_pct": returns_total_pct,
        "returns_total": (returns_total_pct / 100.0) if returns_total_pct is not None else None,
        "pearson": corr_pct / 100.0 if corr_pct is not None and corr_pct > 1 else corr_pct,
        "hurst": hurst,
        "half_life": half_life,
        "strategy_mode": dashboard_strategy,
        "strategy": dashboard_strategy,
        "updated": _scanner_text(row.get("updated")) or "",
        "capture_timestamp_utc": captured_at or "",
    }


def _humanize_collected_strategy(value: str) -> str:
    normalized = value.strip().lower()
    mapping = {
        "staticspread": "Static (Spread)",
        "staticzscorer": "Static (ZScoreR)",
        "dynspread": "Dyn (Spread)",
        "dynzscorer": "Dyn (ZScoreR)",
        "ouspread": "OU (Spread)",
        "ouzscorer": "OU (ZScoreR)",
        "copula": "Copula",
    }
    return mapping.get(normalized, value.strip())


def _first_number_before_token(text: str, token: str) -> float | None:
    if not text:
        return None
    import re

    pattern = rf"([-+]?\d+(?:\.\d+)?)\s*%?\s*{re.escape(token)}"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _scanner_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _active_wizard_source_system(path: Path) -> str:
    name = path.name.lower()
    if "pair_page_capture" in name:
        return "crypto_wizards_pair_page_capture"
    if "live_scanner_capture" in name:
        return "crypto_wizards_live_scanner_capture"
    return "crypto_wizards_active_report"


def _hourly_wizard_source_system(row: pd.Series) -> str:
    source_name = str(row.get("source_name", "") or "").strip().lower()
    if source_name in {"wizard_scanner_live_dom", "wizard_pair_page"}:
        return "crypto_wizards_hourly_pair_panel"
    return ""


def _match_hourly_pair_queue_row(frame: pd.DataFrame, pair: str) -> pd.Series | None:
    if frame.empty:
        return None
    pair_norm = _canonical_pair(pair)
    pairs = frame.get("pair", pd.Series(dtype=object)).map(_canonical_pair)
    matched = frame.loc[pairs == pair_norm]
    if matched.empty:
        return None
    return matched.iloc[0]


def _row_from_source_path(
    root: Path,
    source_path: str,
    *,
    active_row: dict[str, object] | None = None,
) -> dict[str, object]:
    path = Path(source_path)
    if not path.is_absolute():
        path = root / path
    if not path.exists() or path.suffix.lower() != ".json":
        return {}
    return _row_from_payload_path(path, active_row=active_row)


def _row_from_payload_path(path: Path, *, active_row: dict[str, object] | None = None) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list) and active_row is not None:
        matched = _match_merged_payload_record(payload, active_row)
        if matched is None:
            return {}
        payload = matched
    snapshot = snapshot_from_payload(payload)
    row = snapshot.to_row()
    normalized_payload = {str(k).lower(): v for k, v in payload.items()} if isinstance(payload, dict) else {}
    for field in [
        "spread_id",
        "strategy_id",
        "exact_mode",
        "strategy_mode",
        "dashboard_recommended_strategy",
        "pearson",
        "spearman",
        "kendall",
        "corr_copula",
        "copula",
        "u1_given_u2",
        "u2_given_u1",
        "ecm_x_available",
        "ecm_y_available",
        "ecm_strength_available",
        "pair_id",
        "id",
        "backtest_ts",
        "coint_eg",
        "coint_eg_inc_trend",
        "coint_eg_p",
        "johansen_coint",
        "zscore_last",
        "zscore_roll_last",
        "zscore_window",
        "mini_zscore",
        "stddev_cross",
        "zero_cross",
        "sym_1_volume",
        "sym_2_volume",
        "sym_1_volatility_lt",
        "sym_2_volatility_lt",
        "x_weighting",
        "y_weighting",
        "profile_match",
        "ou_optimal",
        "closed",
        "win_rate",
        "mdd",
        "var",
        "cvar",
        "ml_confidence",
    ]:
        if field in normalized_payload:
            row[field] = normalized_payload[field]
    derived = _derived_history_summary(payload)
    for field, value in derived.items():
        existing = row.get(field)
        if field.startswith("ecm_"):
            row[field] = _as_bool(existing) or _as_bool(value)
            continue
        if existing is None or (isinstance(existing, str) and not existing.strip()):
            row[field] = value
    row["source_path"] = str(path)
    row["history_rows"] = len(extract_history_rows(payload)) if isinstance(payload, dict) else 0
    return row


def _match_merged_payload_record(records: list[object], active_row: dict[str, object]) -> dict[str, object] | None:
    normalized_records = [record for record in records if isinstance(record, dict)]
    api_rank = _as_int(active_row.get("api_rank"))
    if api_rank is not None:
        rank_matches = [record for record in normalized_records if _as_int(record.get("_row_index")) == api_rank - 1]
        if len(rank_matches) == 1:
            return rank_matches[0]

    api_row_index = _as_int(active_row.get("api_row_index"))
    if api_row_index is not None:
        index_matches = [record for record in normalized_records if _as_int(record.get("_row_index")) == api_row_index]
        if len(index_matches) == 1:
            return index_matches[0]

    asset_x, asset_y = _asset_values(active_row, str(active_row.get("pair", "") or ""))
    expected_exchange = _text_value(active_row.get("exchange")).lower()
    expected_interval = _canonical_interval(_text_value(active_row.get("interval") or active_row.get("timeframe")))
    expected_strategy = _text_value(active_row.get("api_strategy") or active_row.get("scanner_strategy_filter")).lower()
    matches: list[dict[str, object]] = []
    for record in normalized_records:
        if asset_x and _text_value(record.get("symbol_1")).upper() != asset_x.upper():
            continue
        if asset_y and _text_value(record.get("symbol_2")).upper() != asset_y.upper():
            continue
        if expected_exchange and _text_value(record.get("exchange")).lower() != expected_exchange:
            continue
        if expected_interval and _canonical_interval(_text_value(record.get("interval"))) != expected_interval:
            continue
        if expected_strategy and _text_value(record.get("_strategy_request")).lower() != expected_strategy:
            continue
        matches.append(record)
    return matches[0] if len(matches) == 1 else None


def _normalize_evidence_row(
    row: dict[str, object],
    evidence_path: Path,
    source_path: str,
    source_system: str,
    *,
    root: Path,
    now: datetime,
) -> dict[str, object]:
    spread_id = _as_int(_first_present(row, ["spread_id"]))
    strategy_id = _as_int(_first_present(row, ["strategy_id"]))
    exact_mode = str(_first_present(row, ["exact_mode", "mode", "wizard_exact_mode"]) or "")
    dashboard_recommended_strategy = str(
        _first_present(
            row,
            [
                "dashboard_recommended_strategy",
                "recommended_strategy",
                "selected_strategy_value",
                "strategy_name",
                "strategy_mode",
                "strategy",
            ],
        )
        or ""
    )
    strategy_hint = str(_first_present(row, ["strategy_family_note", "strategy_mode", "spread_type"]) or "")
    z_or_spread_hint = str(_first_present(row, ["z_or_spread_note", "cw_strategy"]) or "")
    mode = _resolve_mode(spread_id, strategy_id, exact_mode, dashboard_recommended_strategy, strategy_hint, z_or_spread_hint)
    local_strategy_id, local_strategy_name, local_strategy_family, strategy_mapping_status = _local_strategy_mapping(
        mode.exact_mode,
        dashboard_recommended_strategy,
        mode.strategy_id,
    )
    returns_total, returns_total_raw, returns_total_unit = _normalized_returns_total(row, source_system)
    source_annualized_return_raw = _first_present(
        row,
        ["annualized_return", "annual_return", "returns_total_pct"],
    )
    sharpe = _as_float(_first_present(row, ["sharpe", "sharpe_ratio"]))
    pair = _pair_value(row)
    asset_x, asset_y = _asset_values(row, pair)
    pair = f"{asset_x}/{asset_y}" if asset_x and asset_y else _canonical_pair(pair)
    interval = _canonical_interval(str(_first_present(row, ["interval", "timeframe"]) or ""))
    ecm_x = _as_bool(_first_present(row, ["ecm_x_available"]))
    ecm_y = _as_bool(_first_present(row, ["ecm_y_available"]))
    ecm_strength = _as_bool(_first_present(row, ["ecm_strength_available"]))
    engle_granger_cointegrated = _as_bool(_first_present(row, ["coint_eg", "engle_granger_cointegrated"]))
    engle_granger_trend = _as_bool(_first_present(row, ["coint_eg_inc_trend", "engle_granger_trend"]))
    johansen_cointegrated = _as_bool(_first_present(row, ["johansen_coint", "johansen_cointegrated"]))
    volume_x = _as_float(_first_present(row, ["sym_1_volume", "volume_x"]))
    volume_y = _as_float(_first_present(row, ["sym_2_volume", "volume_y"]))
    backtest_timestamp = _first_present(row, ["backtest_ts", "backtest_timestamp"])
    freshness = _source_freshness(row, now)
    source_health = _source_health(row, root=root, source_path=source_path, source_system=source_system)
    source_authority = _source_authority(source_system)
    settings_complete, settings_missing = _backtest_settings_status(row, mode.exact_mode)
    return {
        "pair": pair,
        "asset_x": asset_x,
        "asset_y": asset_y,
        "exchange": str(_first_present(row, ["exchange"]) or ""),
        "interval": interval,
        "period": _as_int(_first_present(row, ["period"])),
        "wizard_pair_id": _first_present(row, ["pair_id"]),
        "wizard_setup_id": _first_present(row, ["id", "setup_id"]),
        "backtest_timestamp": _text_value(backtest_timestamp),
        "backtest_timestamp_utc": _epoch_timestamp_utc(backtest_timestamp),
        "setup_identity": "",
        "setup_rank_within_pair": None,
        "setup_role": "",
        "primary_wizard_setup": False,
        "dashboard_recommended_strategy": dashboard_recommended_strategy,
        "dashboard_pair_rank": _as_int(_first_present(row, ["dashboard_pair_rank", "pair_rank", "rank"])),
        "dashboard_selector_score": _as_float(_first_present(row, ["dashboard_selector_score", "selector_score", "score"])),
        "exact_mode": mode.exact_mode,
        "spread_id": mode.spread_id,
        "strategy_id": mode.strategy_id,
        "local_strategy_id": local_strategy_id,
        "local_strategy_name": local_strategy_name,
        "local_strategy_family": local_strategy_family,
        "strategy_mapping_status": strategy_mapping_status,
        "mode_valid": mode.mode_valid,
        "mode_source": mode.mode_source,
        "mode_blocker": mode.mode_blocker,
        "sharpe": sharpe,
        "returns_total_raw": returns_total_raw,
        "returns_total_unit": returns_total_unit,
        "returns_total": returns_total,
        "returns_total_pct": returns_total * 100 if returns_total is not None else None,
        "source_annualized_return_raw": source_annualized_return_raw if source_annualized_return_raw is not None else "",
        "discovery_min_sharpe": DISCOVERY_MIN_SHARPE,
        "discovery_min_returns_total": DISCOVERY_MIN_RETURNS_TOTAL,
        "passes_sharpe_gate": bool(sharpe is not None and sharpe >= DISCOVERY_MIN_SHARPE),
        "passes_sharpe_gt_2": bool(sharpe is not None and sharpe >= DISCOVERY_MIN_SHARPE),
        "passes_returns_total_gt_20pct": bool(returns_total is not None and returns_total > DISCOVERY_MIN_RETURNS_TOTAL),
        "hurst": _as_float(_first_present(row, ["hurst", "top_hurst"])),
        "half_life": _as_float(_first_present(row, ["half_life", "halflife", "top_half_life"])),
        "zscore_last": _as_float(_first_present(row, ["zscore_last", "zscore"])),
        "zscore_roll_last": _as_float(_first_present(row, ["zscore_roll_last", "rolling_zscore"])),
        "zscore_window": _as_int(_first_present(row, ["zscore_window", "roll_w", "rolling_window"])),
        "mini_zscore": _json_text(_first_present(row, ["mini_zscore"])),
        "stddev_cross": _as_int(_first_present(row, ["stddev_cross"])),
        "zero_cross": _as_int(_first_present(row, ["zero_cross"])),
        "engle_granger_cointegrated": engle_granger_cointegrated,
        "engle_granger_trend": engle_granger_trend,
        "engle_granger_pvalue": _as_float(_first_present(row, ["coint_eg_p", "engle_granger_pvalue"])),
        "johansen_cointegrated": johansen_cointegrated,
        "stationarity_status": _stationarity_status(
            engle_granger_cointegrated,
            engle_granger_trend,
            johansen_cointegrated,
        ),
        "pearson": _correlation_proxy(_first_present(row, ["pearson", "top_corr"])),
        "spearman": _as_float(_first_present(row, ["spearman"])),
        "kendall": _as_float(_first_present(row, ["kendall"])),
        "copula": str(_first_present(row, ["copula"]) or ""),
        "corr_copula": _as_float(_first_present(row, ["corr_copula", "copula_correlation"])),
        "u1_given_u2": _as_float(_first_present(row, ["u1_given_u2"])),
        "u2_given_u1": _as_float(_first_present(row, ["u2_given_u1"])),
        "ecm_x_available": ecm_x,
        "ecm_y_available": ecm_y,
        "ecm_strength_available": ecm_strength,
        "hedge_ratio": _as_float(_first_present(row, ["hedge_ratio"])),
        "volume_x": volume_x,
        "volume_y": volume_y,
        "volume_min": min(volume_x, volume_y) if volume_x is not None and volume_y is not None else None,
        "volatility_x_lt": _as_float(_first_present(row, ["sym_1_volatility_lt", "volatility_x_lt"])),
        "volatility_y_lt": _as_float(_first_present(row, ["sym_2_volatility_lt", "volatility_y_lt"])),
        "x_weighting": _as_float(_first_present(row, ["x_weighting"])),
        "y_weighting": _as_float(_first_present(row, ["y_weighting"])),
        "profile_match": _as_bool(_first_present(row, ["profile_match"])),
        "ou_optimal": _as_bool(_first_present(row, ["ou_optimal"])),
        "closed_trades": _as_int(_first_present(row, ["closed_trades"])),
        "backtest_closed": _as_bool(_first_present(row, ["closed", "backtest_closed"])),
        "win_rate": _as_float(_first_present(row, ["win_rate"])),
        "drawdown": _as_float(_first_present(row, ["drawdown", "max_drawdown", "mdd"])),
        "var": _as_float(_first_present(row, ["var"])),
        "cvar": _as_float(_first_present(row, ["cvar"])),
        "ml_confidence": _as_float(_first_present(row, ["ml_confidence"])),
        "source_system": source_system,
        "source_authority": source_authority,
        "source_timestamp": freshness["timestamp"].isoformat() if freshness["timestamp"] else "",
        "source_age_hours": freshness["age_hours"] if freshness["age_hours"] is not None else None,
        "source_fresh": freshness["fresh"],
        "source_freshness_reason": freshness["reason"],
        "source_health": source_health,
        "backtest_settings_complete": settings_complete,
        "backtest_settings_missing": ";".join(settings_missing),
        "source_path": source_path,
        "evidence_path": str(evidence_path),
    }


def _normalized_returns_total(row: dict[str, object], source_system: str) -> tuple[float | None, str, str]:
    raw = _first_present(row, ["returns_total", "returns_total_decimal"])
    if raw is None:
        fallback = _first_present(row, ["returns_total_pct"])
        fallback_value = _as_float(fallback)
        return (
            (fallback_value / 100.0) if fallback_value is not None else None,
            _text_value(fallback),
            "percent_fallback" if fallback_value is not None else "missing",
        )

    raw_text = _text_value(raw)
    value = _as_float(raw)
    if value is None:
        return None, raw_text, "invalid"

    source_percent_convention = source_system in {
        "crypto_wizards_pair_page_capture",
        "crypto_wizards_hourly_pair_panel",
    }
    has_display_percent = "%" in raw_text
    has_annualized_companion = _first_present(row, ["annualized_return", "annual_return", "returns_total_pct"]) is not None
    if has_display_percent:
        return value / 100.0, raw_text, "percent_text"
    if source_percent_convention and (abs(value) > 1.0 or has_annualized_companion):
        return value / 100.0, raw_text, "percent_source_convention"
    return value, raw_text, "decimal"


def _text_value(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _source_freshness(row: dict[str, object] | pd.Series, now: datetime) -> dict[str, object]:
    timestamp_value = _first_present(
        dict(row),
        ["source_timestamp", "capture_timestamp_utc", "captured_at", "scan_timestamp", "updated_at_utc"],
    )
    timestamp = _parse_timestamp(timestamp_value)
    if timestamp is None:
        return {"timestamp": None, "age_hours": None, "fresh": False, "reason": "source_timestamp_missing"}
    reference = _as_utc(now)
    if timestamp > reference + timedelta(minutes=5):
        return {"timestamp": timestamp, "age_hours": None, "fresh": False, "reason": "source_timestamp_in_future"}
    age_hours = max((reference - timestamp).total_seconds() / 3600.0, 0.0)
    if age_hours > WIZARD_EVIDENCE_MAX_AGE_HOURS:
        return {
            "timestamp": timestamp,
            "age_hours": round(age_hours, 6),
            "fresh": False,
            "reason": f"source_stale_over_{WIZARD_EVIDENCE_MAX_AGE_HOURS:g}h",
        }
    return {"timestamp": timestamp, "age_hours": round(age_hours, 6), "fresh": True, "reason": "source_current"}


def _parse_timestamp(value: object) -> datetime | None:
    text = _text_value(value)
    if not text:
        return None
    try:
        parsed = pd.Timestamp(text)
    except (TypeError, ValueError):
        return None
    if pd.isna(parsed):
        return None
    return _as_utc(parsed.to_pydatetime())


def _epoch_timestamp_utc(value: object) -> str:
    numeric = _as_float(value)
    if numeric is None:
        parsed = _parse_timestamp(value)
        return parsed.isoformat() if parsed else ""
    try:
        unit = "ms" if abs(numeric) >= 100_000_000_000 else "s"
        parsed = pd.to_datetime(numeric, unit=unit, utc=True)
    except (TypeError, ValueError, OverflowError):
        return ""
    if pd.isna(parsed):
        return ""
    return parsed.to_pydatetime().isoformat()


def _json_text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, (list, dict)):
        return json.dumps(value, separators=(",", ":"), sort_keys=True)
    return _text_value(value)


def _stationarity_status(
    engle_granger_cointegrated: bool,
    engle_granger_trend: bool,
    johansen_cointegrated: bool,
) -> str:
    if engle_granger_trend:
        return "engle_granger_with_trend"
    if engle_granger_cointegrated and johansen_cointegrated:
        return "engle_granger_and_johansen"
    if engle_granger_cointegrated:
        return "engle_granger"
    if johansen_cointegrated:
        return "johansen"
    return "unconfirmed"


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _source_authority(source_system: str) -> str:
    return {
        "crypto_wizards_live_scanner_capture": "discovery_only",
        "crypto_wizards_hourly_pair_panel": "discovery_only",
        "crypto_wizards_pair_page_capture": "pair_detail_capture",
        "crypto_wizards_pair_detail": "historical_pair_detail",
        "crypto_wizards_active_report": "report_only",
    }.get(source_system, "unknown")


def _source_health(row: dict[str, object], *, root: Path, source_path: str, source_system: str) -> str:
    status = _text_value(_first_present(row, ["capture_status"])).lower()
    blocker = _text_value(_first_present(row, ["capture_blocker"])).lower()
    if blocker or status in {"failed", "error", "blocked"}:
        return "capture_failed"
    if _source_has_retrieval_failure(str(root), source_path):
        return "data_retrieval_failed"
    if source_system in {
        "crypto_wizards_live_scanner_capture",
        "crypto_wizards_hourly_pair_panel",
        "crypto_wizards_pair_page_capture",
    }:
        return "healthy"
    return "not_assessed"


@lru_cache(maxsize=1024)
def _source_has_retrieval_failure(root_text: str, source_path: str) -> bool:
    path = Path(source_path)
    root = Path(root_text)
    if not path.is_absolute():
        path = root / path
    try:
        resolved = path.resolve()
        resolved.relative_to(root.resolve())
    except (OSError, ValueError):
        return False
    if not resolved.exists() or resolved.suffix.lower() != ".json":
        return False
    try:
        text = resolved.read_text(encoding="utf-8").lower()
    except OSError:
        return False
    return "data retrieval failed" in text or "data_retrieval_failed" in text


def _backtest_settings_status(
    row: dict[str, object], exact_mode: str, *, strict: bool = False
) -> tuple[bool, list[str]]:
    missing: list[str] = []
    if not _has_all_values(row, ["entry_long_operator", "entry_long_value", "entry_short_operator", "entry_short_value"]):
        missing.append("entry_thresholds")
    elif not all(_valid_capture_operator(_first_present(row, [field])) for field in ["entry_long_operator", "entry_short_operator"]):
        missing.append("invalid_entry_operators")
    elif not all(_as_float(_first_present(row, [field])) is not None for field in ["entry_long_value", "entry_short_value"]):
        missing.append("invalid_entry_threshold_values")
    if not _has_all_values(row, ["entry_long_position", "entry_short_position"]):
        missing.append("entry_direction_mapping")
    elif not all(_valid_capture_position(_first_present(row, [field])) for field in ["entry_long_position", "entry_short_position"]):
        missing.append("invalid_entry_direction_mapping")
    if not _has_all_values(row, ["exit_long_operator", "exit_long_value", "exit_short_operator", "exit_short_value"]):
        missing.append("exit_thresholds")
    elif not all(_valid_capture_operator(_first_present(row, [field])) for field in ["exit_long_operator", "exit_short_operator"]):
        missing.append("invalid_exit_operators")
    elif not all(_as_float(_first_present(row, [field])) is not None for field in ["exit_long_value", "exit_short_value"]):
        missing.append("invalid_exit_threshold_values")
    if _first_present(row, ["capital_weighting_slider_value", "x_weighting", "weight_x"]) is None:
        missing.append("capital_weighting")
    if strict:
        x_weight = _as_float(_first_present(row, ["x_weighting", "weight_x"]))
        y_weight = _as_float(_first_present(row, ["y_weighting", "weight_y"]))
        if x_weight is None or y_weight is None:
            missing.append("capital_weighting_legs")
        elif x_weight <= 0 or y_weight <= 0 or abs((x_weight + y_weight) - 1.0) > 0.02:
            missing.append("invalid_capital_weighting_legs")
        if _first_present(row, ["capital_weighting"]) is None:
            missing.append("capital_weighting_mode")
        if _first_present(row, ["metric_mode"]) is None:
            missing.append("metric_mode")
        for field in ["commission_rate", "slippage_rate"]:
            value = _as_float(_first_present(row, [field]))
            if value is None:
                missing.append(field)
            elif value < 0:
                missing.append(f"invalid_{field}")
        stop_loss = _as_float(_first_present(row, ["stop_loss_rate"]))
        if stop_loss is None:
            missing.append("stop_loss_rate")
        elif stop_loss < 0:
            missing.append("invalid_stop_loss_rate")
        close_periods = _as_float(_first_present(row, ["exit_n_periods"]))
        if close_periods is None:
            missing.append("exit_n_periods")
        elif close_periods < 0 or not close_periods.is_integer():
            missing.append("invalid_exit_n_periods")
        captured_spread_type = _text_value(row.get("spread_type")).lower().replace("_", " ")
        expected_spread_type = (
            "static"
            if exact_mode.startswith("Static")
            else "ou"
            if exact_mode.startswith("OU")
            else "dynamic"
            if exact_mode.startswith("Dyn")
            else ""
        )
        if not captured_spread_type:
            missing.append("spread_type")
        elif expected_spread_type and captured_spread_type != expected_spread_type:
            missing.append("mode_spread_type_mismatch")
    else:
        if _first_present(row, ["slippage_rate", "commission_rate"]) is None:
            missing.append("wizard_cost_assumptions")
        elif not any(_as_float(_first_present(row, [field])) is not None for field in ["slippage_rate", "commission_rate"]):
            missing.append("invalid_wizard_cost_assumptions")
    if exact_mode in {"Static (Spread)", "Static (ZScoreR)", "OU (Spread)", "OU (ZScoreR)"}:
        hedge_ratio = _as_float(_first_present(row, ["hedge_ratio", "static_hedge_ratio"]))
        if hedge_ratio is None:
            missing.append("static_hedge_ratio")
        elif hedge_ratio == 0:
            missing.append("invalid_static_hedge_ratio")
    if exact_mode in {"Dyn (Spread)", "Dyn (ZScoreR)"}:
        dynamic_method = _first_present(row, ["dynamic_hedge_ratio_method"])
        if dynamic_method is None:
            missing.append("dynamic_hedge_ratio_method")
        elif not _valid_dynamic_hedge_ratio_method(dynamic_method):
            missing.append("invalid_dynamic_hedge_ratio_method")
        if _first_present(row, ["dynamic_hedge_ratio_window"]) is None:
            missing.append("dynamic_hedge_ratio_window")
        elif not _positive_integer(_first_present(row, ["dynamic_hedge_ratio_window"])):
            missing.append("invalid_dynamic_hedge_ratio_window")
    if exact_mode in {"OU (Spread)", "OU (ZScoreR)"}:
        if _as_float(_first_present(row, ["ou_mu"])) is None:
            missing.append("ou_mu")
        ou_sigma = _as_float(_first_present(row, ["ou_sigma"]))
        if ou_sigma is None:
            missing.append("ou_sigma")
        elif ou_sigma <= 0:
            missing.append("invalid_ou_sigma")
    if exact_mode.endswith("(ZScoreR)") and _first_present(row, ["roll_w", "rolling_window", "zscore_window"]) is None:
        missing.append("rolling_window")
    elif exact_mode.endswith("(ZScoreR)") and not _positive_integer(_first_present(row, ["roll_w", "rolling_window", "zscore_window"])):
        missing.append("invalid_rolling_window")
    if exact_mode == "Copula":
        entry_lower = _as_float(_first_present(row, ["copula_entry_lower"]))
        entry_upper = _as_float(_first_present(row, ["copula_entry_upper"]))
        if entry_lower is None or entry_upper is None:
            missing.append("copula_thresholds")
        elif not entry_lower < entry_upper:
            missing.append("invalid_copula_thresholds")
        exit_lower = _as_float(_first_present(row, ["copula_exit_lower"]))
        exit_upper = _as_float(_first_present(row, ["copula_exit_upper"]))
        if exit_lower is None or exit_upper is None:
            missing.append("copula_exit_rule")
        elif not exit_lower <= exit_upper:
            missing.append("invalid_copula_exit_rule")
        direction_view = _first_present(row, ["copula_direction_view", "copula_directional_entry_rule"])
        if direction_view is None:
            missing.append("copula_directional_entry_rule")
        elif not _valid_copula_direction_view(direction_view):
            missing.append("invalid_copula_directional_entry_rule")
        if _first_present(row, ["copula_family"]) is None:
            missing.append("copula_family")
        if _first_present(row, ["copula_signal_type"]) is None:
            missing.append("copula_signal_type")
    return not missing, missing


def _valid_capture_operator(value: object) -> bool:
    normalized = _text_value(value).lower().replace(" ", "_")
    return normalized in {
        ">",
        ">=",
        "<",
        "<=",
        "greater_than",
        "greater_than_or_equal",
        "greater_or_equal",
        "less_than",
        "less_than_or_equal",
        "less_or_equal",
    }


def _valid_capture_position(value: object) -> bool:
    normalized = _text_value(value).lower().replace("-", "_").replace(" ", "_")
    return normalized in {"long_x_short_y", "short_x_long_y"}


def _valid_dynamic_hedge_ratio_method(value: object) -> bool:
    normalized = _text_value(value).lower().replace("-", "_").replace(" ", "_")
    return normalized in {
        "history_captured_hedge_ratio",
        "history",
        "captured_history",
        "captured_history_hedge_ratio",
        "rolling_ols_log_prices",
        "rolling_ols",
        "rolling_ols_log_price",
    }


def _positive_integer(value: object) -> bool:
    number = _as_float(value)
    return number is not None and number > 1 and number.is_integer()


def _valid_copula_direction_view(value: object) -> bool:
    normalized = "".join(character for character in _text_value(value).lower() if character.isalnum())
    return normalized in {
        "u1givenu2",
        "xgiveny",
        "conditionalu1u2",
        "u2givenu1",
        "ygivenx",
        "conditionalu2u1",
    }


def _has_all_values(row: dict[str, object], fields: list[str]) -> bool:
    return all(_first_present(row, [field]) is not None for field in fields)


def _correlation_proxy(value: object) -> float | None:
    raw = _as_float(value)
    if raw is None:
        return None
    if raw > 1:
        raw = raw / 100.0
    return raw


def _assign_setup_identity(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    working["pair"] = working.get("pair", pd.Series("", index=working.index)).astype(str)
    working["interval"] = working.get("interval", pd.Series("", index=working.index)).astype(str)
    working["exact_mode"] = working.get("exact_mode", pd.Series("", index=working.index)).astype(str)
    working["period"] = pd.to_numeric(working.get("period", pd.Series(index=working.index)), errors="coerce")
    working["_mode_valid"] = working.get("mode_valid", pd.Series(False, index=working.index)).fillna(False).astype(bool)
    working["_source_priority"] = pd.to_numeric(working.get("_source_priority", pd.Series(index=working.index)), errors="coerce").fillna(9)
    working["_dashboard_rank"] = pd.to_numeric(working.get("dashboard_pair_rank", pd.Series(index=working.index)), errors="coerce")
    working["_selector_score"] = pd.to_numeric(working.get("dashboard_selector_score", pd.Series(index=working.index)), errors="coerce")
    working["_returns_total"] = pd.to_numeric(working.get("returns_total", pd.Series(index=working.index)), errors="coerce")
    working["_sharpe"] = pd.to_numeric(working.get("sharpe", pd.Series(index=working.index)), errors="coerce")

    working["setup_identity"] = working.apply(
        lambda row: _wizard_setup_identity(
            str(row.get("pair", "")),
            str(row.get("interval", "")),
            _as_int(row.get("period")),
            str(row.get("exact_mode", "")),
        ),
        axis=1,
    )

    sort_frame = working.sort_values(
        ["pair", "interval", "_mode_valid", "_source_priority", "_dashboard_rank", "_selector_score", "_returns_total", "_sharpe"],
        ascending=[True, True, False, True, True, False, False, False],
        na_position="last",
        kind="mergesort",
    )
    rank_map: dict[int, int] = {}
    role_map: dict[int, str] = {}
    primary_map: dict[int, bool] = {}
    for _, group in sort_frame.groupby(["pair", "interval"], dropna=False, sort=False):
        for rank, idx in enumerate(group.index.tolist(), start=1):
            rank_map[int(idx)] = rank
            role_map[int(idx)] = "primary" if rank == 1 else "alternate"
            primary_map[int(idx)] = rank == 1
    working["setup_rank_within_pair"] = [rank_map.get(int(idx), 0) for idx in working.index]
    working["setup_role"] = [role_map.get(int(idx), "") for idx in working.index]
    working["primary_wizard_setup"] = [primary_map.get(int(idx), False) for idx in working.index]
    return working.drop(columns=["_mode_valid", "_source_priority", "_dashboard_rank", "_selector_score", "_returns_total", "_sharpe"])


def _merge_duplicate_evidence_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    merge_columns = [
        "dashboard_recommended_strategy",
        "pearson",
        "spearman",
        "kendall",
        "copula",
        "corr_copula",
        "u1_given_u2",
        "u2_given_u1",
        "ecm_x_available",
        "ecm_y_available",
        "ecm_strength_available",
        "hedge_ratio",
        "closed_trades",
        "drawdown",
        "hurst",
        "half_life",
        "local_strategy_id",
        "local_strategy_name",
        "local_strategy_family",
        "strategy_mapping_status",
        "mode_source",
    ]
    rows: list[dict[str, object]] = []
    for _, group in frame.groupby(["pair", "interval", "exact_mode"], dropna=False, sort=False):
        merged = group.iloc[0].to_dict()
        for _, donor in group.iloc[1:].iterrows():
            if not _same_source_capture(merged, donor):
                continue
            for column in merge_columns:
                if column not in donor.index:
                    continue
                donor_value = donor.get(column)
                if column.startswith("ecm_"):
                    merged[column] = bool(merged.get(column, False)) or bool(donor_value)
                elif _missing_scalar(merged.get(column)) and not _missing_scalar(donor_value):
                    merged[column] = donor_value
        rows.append(merged)
    return pd.DataFrame(rows, columns=frame.columns)


def _same_source_capture(merged: dict[str, object], donor: pd.Series) -> bool:
    source_system = _text_value(merged.get("source_system"))
    source_path = _text_value(merged.get("source_path"))
    source_timestamp = _text_value(merged.get("source_timestamp"))
    return bool(
        source_system
        and source_path
        and source_timestamp
        and source_system == _text_value(donor.get("source_system"))
        and source_path == _text_value(donor.get("source_path"))
        and source_timestamp == _text_value(donor.get("source_timestamp"))
    )


def _backfill_local_derived_history_fields(frame: pd.DataFrame, *, root: Path) -> pd.DataFrame:
    if frame.empty:
        return frame
    local_index = _local_history_index(root)
    working = frame.copy()
    fill_fields = [
        "pearson",
        "spearman",
        "kendall",
        "ecm_x_available",
        "ecm_y_available",
        "ecm_strength_available",
        "hurst",
        "half_life",
        "hedge_ratio",
    ]
    for idx, row in working.iterrows():
        local_path = _matching_local_path(local_index, row)
        if local_path is None:
            continue
        local_row = _row_from_payload_path(local_path)
        for field in fill_fields:
            existing = row.get(field)
            donor = local_row.get(field)
            if field.startswith("ecm_"):
                working.at[idx, field] = _as_bool(existing) or _as_bool(donor)
            elif _missing_scalar(existing) and not _missing_scalar(donor):
                working.at[idx, field] = donor
    return working


def _resolve_mode(
    spread_id: int | None,
    strategy_id: int | None,
    exact_mode: str,
    dashboard_recommended_strategy: str,
    strategy_hint: str,
    z_or_spread_hint: str,
) -> ModeResolution:
    if spread_id is not None and strategy_id is not None:
        label = mode_from_ids(spread_id, strategy_id)
        if label:
            return ModeResolution(label, spread_id, strategy_id, True, "spread_id_strategy_id", "")
        return ModeResolution("", spread_id, strategy_id, False, "spread_id_strategy_id", "unknown_spread_strategy_id")
    ids = ids_from_exact_mode(exact_mode)
    if ids != (None, None):
        return ModeResolution(WIZARD_MODE_MAP[ids], ids[0], ids[1], True, "exact_mode", "")
    ids = _ids_from_recommended_strategy(dashboard_recommended_strategy)
    if ids != (None, None):
        return ModeResolution(WIZARD_MODE_MAP[ids], ids[0], ids[1], True, "dashboard_recommended_strategy", "")
    inferred = _infer_exact_mode(strategy_hint, z_or_spread_hint)
    if inferred:
        ids = ids_from_exact_mode(inferred)
        return ModeResolution(inferred, ids[0], ids[1], True, "inferred_from_strategy_and_signal", "")
    return ModeResolution("", spread_id, strategy_id, False, "missing", "missing_exact_mode")


def _ids_from_recommended_strategy(value: str | None) -> tuple[int | None, int | None]:
    text = str(value or "").strip().lower()
    if not text:
        return None, None
    for needle, exact_mode, _ in _RECOMMENDED_STRATEGY_HINTS:
        if needle in text:
            return ids_from_exact_mode(exact_mode)
    return None, None


def _local_strategy_mapping(
    exact_mode: str,
    dashboard_recommended_strategy: str,
    strategy_id: int | None,
) -> tuple[int | None, str, str, str]:
    resolved_id: int | None = None
    if exact_mode in _EXACT_MODE_LOCAL_STRATEGY:
        resolved_id = _EXACT_MODE_LOCAL_STRATEGY[exact_mode]
    elif strategy_id is not None:
        resolved_id = _exact_mode_strategy_fallback(strategy_id)
    if resolved_id is None:
        return None, "", "", "unmapped"
    spec = _LOCAL_STRATEGY_BY_ID.get(resolved_id)
    if spec is None:
        return resolved_id, "", "", "unknown_local_strategy"
    status = "mapped_from_exact_mode" if exact_mode else "mapped_from_dashboard_hint"
    if dashboard_recommended_strategy and not exact_mode:
        status = "mapped_from_dashboard_hint"
    return resolved_id, spec.name, spec.family, status


def _exact_mode_strategy_fallback(strategy_id: int) -> int | None:
    if strategy_id == 3:
        return 5
    if strategy_id == 2:
        return 14
    if strategy_id == 1:
        return 1
    return None


def _source_first_action(row: pd.Series) -> str:
    if bool(row.get("primary_wizard_setup", False)):
        local_value = row.get("local_strategy_id", "")
        local_strategy_id = "" if pd.isna(local_value) else str(local_value).strip()
        if local_strategy_id:
            return f"run_wizard_primary_setup_first:{local_strategy_id}"
        return "capture_or_refine_primary_wizard_setup"
    local_value = row.get("local_strategy_id", "")
    local_strategy_id = "" if pd.isna(local_value) else str(local_value).strip()
    if local_strategy_id:
        return f"run_dashboard_strategy_first:{local_strategy_id}"
    recommended = row.get("dashboard_recommended_strategy", "")
    recommended_text = "" if pd.isna(recommended) else str(recommended).strip()
    if recommended_text:
        return "capture_or_refine_dashboard_strategy_mapping"
    return "capture_dashboard_strategy_first"


def _infer_exact_mode(strategy_hint: str, z_or_spread_hint: str) -> str:
    engine = strategy_hint.strip().lower()
    signal = z_or_spread_hint.strip().lower()
    if "copula" in engine or "copula" in signal:
        return "Copula"
    spread_label = ""
    if "static" in engine:
        spread_label = "Static"
    elif engine in {"dyn", "dynamic"} or "dynamic" in engine:
        spread_label = "Dyn"
    elif "ou" in engine:
        spread_label = "OU"
    signal_label = ""
    if "zscorer" in signal or "zscore r" in signal or "roll" in signal:
        signal_label = "ZScoreR"
    elif "spread" in signal:
        signal_label = "Spread"
    if spread_label and signal_label:
        return f"{spread_label} ({signal_label})"
    return ""


def _derived_history_summary(payload: dict[str, object]) -> dict[str, object]:
    rows = extract_history_rows(payload) if isinstance(payload, dict) else []
    if not rows:
        return {}
    frame = pd.DataFrame(rows)
    derived: dict[str, object] = {}
    if {"price_x", "price_y"}.issubset(frame.columns):
        prices = frame.loc[:, ["price_x", "price_y"]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(prices) >= 3:
            for method, column in [("pearson", "pearson"), ("spearman", "spearman"), ("kendall", "kendall")]:
                value = prices["price_x"].corr(prices["price_y"], method=method)
                if pd.notna(value):
                    derived[column] = float(value)
    for field in ["ecm_x", "ecm_y", "ecm_strength"]:
        if field in frame.columns:
            values = pd.to_numeric(frame[field], errors="coerce")
            derived[f"{field}_available"] = bool(values.notna().any())
    for field in ["hurst", "half_life", "hedge_ratio"]:
        if field in frame.columns:
            values = pd.to_numeric(frame[field], errors="coerce").dropna()
            if not values.empty:
                derived[field] = float(values.iloc[-1])
    return derived


def _ensure_wizard_evidence(root: Path) -> pd.DataFrame:
    path = root / "data" / "processed" / "wizard_evidence.csv"
    if not path.exists():
        build_wizard_evidence(root)
    frame = _read_csv(path)
    if frame.empty:
        return frame
    needs_setup_identity = any(column not in frame.columns for column in ["setup_identity", "setup_rank_within_pair", "setup_role", "primary_wizard_setup"])
    if not needs_setup_identity and "setup_identity" in frame.columns:
        needs_setup_identity = frame["setup_identity"].astype(str).str.strip().eq("").any()
    if needs_setup_identity:
        for column in WIZARD_EVIDENCE_COLUMNS:
            if column not in frame.columns:
                if column in {"primary_wizard_setup", "mode_valid"}:
                    frame[column] = False
                elif column in {"passes_sharpe_gate", "passes_sharpe_gt_2", "passes_returns_total_gt_20pct"}:
                    frame[column] = pd.NA
                else:
                    frame[column] = ""
        frame = _assign_setup_identity(frame)
    return frame


def _passes_sharpe_gate(frame: pd.DataFrame) -> pd.Series:
    if "passes_sharpe_gate" in frame:
        parsed = frame["passes_sharpe_gate"].map(_as_bool)
        raw = frame["passes_sharpe_gate"]
        blank = raw.isna() | raw.astype(str).str.strip().eq("")
        if not blank.all():
            fallback = pd.Series(False, index=frame.index)
            if "sharpe" in frame:
                sharpe = _numeric_series(frame["sharpe"])
                fallback = sharpe.ge(DISCOVERY_MIN_SHARPE).fillna(False)
            elif "passes_sharpe_gt_2" in frame:
                fallback = frame["passes_sharpe_gt_2"].map(_as_bool)
            return parsed.where(~blank, fallback)
    if "sharpe" in frame:
        sharpe = _numeric_series(frame["sharpe"])
        return sharpe.ge(DISCOVERY_MIN_SHARPE).fillna(False)
    if "passes_sharpe_gt_2" in frame:
        return frame["passes_sharpe_gt_2"].map(_as_bool)
    return pd.Series(False, index=frame.index)


def _row_passes_sharpe_gate(row: pd.Series) -> bool:
    if "passes_sharpe_gate" in row.index:
        value = row.get("passes_sharpe_gate", "")
        if str(value).strip() != "":
            return _as_bool(value)
    sharpe = _as_float(row.get("sharpe"))
    if sharpe is not None:
        return sharpe >= DISCOVERY_MIN_SHARPE
    return _as_bool(row.get("passes_sharpe_gt_2", False))


def _hypothesis_status(
    row: pd.Series,
    local_available: bool,
    *,
    freshness: dict[str, object],
) -> tuple[str, str, str]:
    if not bool(freshness["fresh"]):
        return "STALE_EVIDENCE", str(freshness["reason"]), "refresh_current_wizard_capture"
    source_health = _text_value(row.get("source_health"))
    if source_health and source_health != "healthy":
        return "CAPTURE_UNHEALTHY", source_health, "refresh_current_wizard_capture"
    if _text_value(row.get("source_authority")) == "discovery_only":
        return "DISCOVERY_ONLY", "fresh_scanner_evidence_requires_pair_detail_and_settings", "capture_current_wizard_pair_detail"
    if not bool(row.get("mode_valid", False)):
        return "RESEARCH_BLOCKED", "missing_exact_mode", "capture_pair_page_exact_mode"
    if not _row_passes_sharpe_gate(row):
        return "REJECT", f"sharpe_below_{DISCOVERY_MIN_SHARPE:g}", "return_to_wizard_scanner"
    if not bool(row.get("passes_returns_total_gt_20pct", False)):
        return "REJECT", f"returns_total_not_above_{int(DISCOVERY_MIN_RETURNS_TOTAL * 100)}pct", "return_to_wizard_scanner"
    if not _as_bool(row.get("backtest_settings_complete", False)):
        missing = _text_value(row.get("backtest_settings_missing")) or "backtest_settings_missing"
        return "NEEDS_SETTINGS_CAPTURE", missing, "capture_exact_wizard_entry_exit_weighting_and_cost_settings"
    if not local_available:
        return "NEEDS_LOCAL_DATA", "wizard_only_evidence_cannot_promote", "fetch_local_dydx_history"
    return "HYPOTHESIS_READY", "wizard_candidate_ready_for_local_after_cost_test", "run_local_mode_separated_backtest"


def _correlation_status(row: pd.Series) -> tuple[str, float]:
    values = [_as_float(row.get(name)) for name in ["pearson", "spearman", "kendall"]]
    values = [abs(v) for v in values if v is not None]
    if not values:
        return "missing_correlation", 0.0
    if max(values) >= 0.70 and sum(v >= 0.50 for v in values) >= 2:
        return "confirmed_correlation", 15.0
    if max(values) >= 0.50:
        return "partial_correlation", 8.0
    return "weak_correlation", -5.0


def _ecm_status(row: pd.Series) -> tuple[str, float]:
    flags = [bool(row.get(name, False)) for name in ["ecm_x_available", "ecm_y_available", "ecm_strength_available"]]
    if all(flags):
        return "confirmed_ecm", 15.0
    if any(flags):
        return "partial_ecm", 5.0
    return "missing_ecm", 0.0


def _copula_status(row: pd.Series) -> tuple[str, float]:
    exact = str(row.get("exact_mode", ""))
    has_copula = bool(str(row.get("copula", "")).strip()) or _as_float(row.get("corr_copula")) is not None
    has_conditionals = _as_float(row.get("u1_given_u2")) is not None and _as_float(row.get("u2_given_u1")) is not None
    if has_copula and has_conditionals:
        return "confirmed_copula", 15.0
    if has_copula:
        return "partial_copula", 8.0
    if exact == "Copula":
        return "missing_copula_for_copula_mode", -10.0
    return "missing_copula", 0.0


def _mean_reversion_status(row: pd.Series) -> tuple[str, float]:
    exact = str(row.get("exact_mode", "")).strip()
    hurst = _as_float(row.get("hurst"))
    half_life = _as_float(row.get("half_life"))
    if exact == "Copula" and hurst is None and half_life is None:
        return "copula_mode_no_mean_reversion_required", 5.0
    if hurst is None and half_life is None:
        return "missing_mean_reversion", 0.0
    if hurst is not None and hurst >= 0.60:
        return "weak_hurst_mean_reversion", -10.0
    if half_life is not None and half_life <= 0:
        return "weak_half_life", -5.0
    return "confirmed_mean_reversion", 15.0


def _local_history_index(root: Path) -> dict[tuple[str, str], Path]:
    index: dict[tuple[str, str], Path] = {}
    for path in sorted((root / "data" / "raw" / "pair_details").glob("*dydx*derived_history.json")):
        try:
            row = _row_from_payload_path(path)
        except Exception:
            continue
        pair = _canonical_pair_key(str(row.get("asset_x", "") or ""), str(row.get("asset_y", "") or ""), str(row.get("pair", "")))
        interval = str(row.get("interval", "") or "")
        if pair and (pair, interval) not in index:
            index[(pair, interval)] = path
    return index


def _local_data_available(index: dict[tuple[str, str], Path], row: pd.Series) -> bool:
    return _matching_local_path(index, row) is not None


def _matching_local_path(index: dict[tuple[str, str], Path], row: pd.Series) -> Path | None:
    pair = _canonical_pair_key(str(row.get("asset_x", "") or ""), str(row.get("asset_y", "") or ""), str(row.get("pair", "")))
    interval = str(row.get("interval", "") or "")
    direct = index.get((pair, interval)) or index.get((pair, ""))
    if direct is not None:
        return direct
    candidates = [path for (candidate_pair, _candidate_interval), path in index.items() if candidate_pair == pair]
    return candidates[0] if candidates else None


def _history_last_values(path: Path | None) -> dict[str, float]:
    if not path or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = extract_history_rows(payload)
    except Exception:
        return {}
    if not rows:
        return {}
    last = rows[-1]
    return {
        "zscore": _as_float(last.get("zscore")),
        "rolling_zscore": _as_float(last.get("rolling_zscore", last.get("zscore_roll"))),
    }


def _parity_status(local_path: Path | None, z_delta: float | None, rz_delta: float | None) -> tuple[str, str]:
    if local_path is None:
        return "MISSING_LOCAL_DATA", "no_matching_local_dydx_history"
    deltas = [abs(v) for v in [z_delta, rz_delta] if v is not None]
    if not deltas:
        return "NOT_REPLICATED", "no_comparable_zscore_fields"
    if max(deltas) <= 0.05:
        return "MATCH", "wizard_and_local_zscore_close"
    if max(deltas) <= 0.25:
        return "CLOSE", "small_zscore_difference"
    return "MISMATCH", "mode_or_formula_difference_needs_review"


def _pair_page_url_from_source(root: Path, source_path: str) -> str:
    if not source_path:
        return ""
    path = Path(source_path)
    if not path.is_absolute():
        path = root / path
    if not path.exists() or path.suffix.lower() != ".json":
        return ""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    if not isinstance(payload, dict):
        return ""
    return str(payload.get("source_url") or payload.get("url") or "")


def _capture_action(pair_page_url: str) -> str:
    if pair_page_url:
        return "open_pair_page_run_capture_helper_click_recalculate_download_json_import_pair_details"
    return "find_pair_page_from_scanner_then_capture_exact_strategy_dropdown_and_backtest_settings"


def _delta(left: Any, right: Any) -> float | None:
    left_f = _as_float(left)
    right_f = _as_float(right)
    if left_f is None or right_f is None:
        return None
    return round(left_f - right_f, 8)


def _looks_like_wizard_source(row: dict[str, object], path: Path) -> bool:
    text = path.as_posix().lower()
    return "cw" in text or "wizard" in text or _as_float(row.get("sharpe")) is not None or _as_float(row.get("returns_total")) is not None


def _missing_scalar(value: object) -> bool:
    if value is None:
        return True
    if pd.isna(value):
        return True
    if isinstance(value, str):
        return not value.strip()
    return False


def _pair_value(row: dict[str, object]) -> str:
    pair = str(_first_present(row, ["pair"]) or "")
    if pair and pair.upper() != "UNKNOWN":
        return pair
    asset_x, asset_y = _asset_values(row, "")
    return f"{asset_x}/{asset_y}" if asset_x and asset_y else ""


def _asset_values(row: dict[str, object], pair: str) -> tuple[str, str]:
    asset_x = str(_first_present(row, ["asset_x"]) or "")
    asset_y = str(_first_present(row, ["asset_y"]) or "")
    if asset_x and asset_y:
        return asset_x, asset_y
    if "/" in pair:
        left, right = pair.split("/", 1)
        return left, right
    return asset_x, asset_y


def _canonical_pair(pair: str) -> str:
    return pair.upper().replace("_", "-").replace(" / ", "/").strip()


def _canonical_interval(interval: str) -> str:
    text = str(interval or "").strip().lower().replace("_", "").replace(" ", "")
    if text in {"1d", "1day", "daily", "day"}:
        return "daily"
    if text in {"1h", "1hour", "hourly", "hour"}:
        return "hourly"
    if text in {"5m", "5min", "5mins", "5minute", "5minutes"}:
        return "5min"
    if text in {"4h", "4hour", "4hours"}:
        return "4hour"
    return str(interval or "").strip()


def _canonical_pair_key(asset_x: str, asset_y: str, pair: str = "") -> str:
    left = str(asset_x or "").upper().replace("_", "-").strip()
    right = str(asset_y or "").upper().replace("_", "-").strip()
    if left and right:
        return f"{left}|{right}"
    text = _canonical_pair(pair)
    if "/" in text:
        pieces = [piece for piece in text.split("/") if piece]
        if len(pieces) == 2:
            return f"{pieces[0]}|{pieces[1]}"
    pieces = [piece for piece in text.split("-") if piece]
    usd_positions = [i for i, piece in enumerate(pieces) if piece == "USD"]
    if len(usd_positions) >= 2:
        return f"{pieces[0]}-USD|{pieces[usd_positions[0] + 1]}-USD"
    return text


def _wizard_setup_identity(pair: str, interval: str, period: int | None, exact_mode: str) -> str:
    pair_key = _canonical_pair(pair).replace("/", "|")
    interval_key = _canonical_interval(str(interval or "")).strip().lower().replace(" ", "_")
    period_key = str(period if period is not None else "")
    mode_key = str(exact_mode or "").strip().lower().replace(" ", "_").replace("/", "_")
    return f"{pair_key}|{interval_key}|{period_key}|{mode_key}".strip("|")


def _evidence_pair_interval_key(row: pd.Series) -> str:
    return f"{_canonical_pair_key(str(row.get('asset_x', '') or ''), str(row.get('asset_y', '') or ''), str(row.get('pair', '') or ''))}|{str(row.get('interval', '') or '')}"


def _normalize_mode_label(value: str | None) -> str:
    normalized = str(value or "").lower().replace(" ", "").replace("-", "").replace("_", "")
    normalized = normalized.replace("zscoreroll", "zscorer")
    if normalized.endswith("copula"):
        return "copula"
    return normalized


def _first_present(row: dict[str, object], names: list[str]) -> object | None:
    lower = {str(k).lower(): v for k, v in row.items()}
    for name in names:
        value = lower.get(name.lower())
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except (TypeError, ValueError):
            pass
        if str(value).strip() != "":
            return value
    return None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    text = str(value).strip().replace("%", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _as_int(value: Any) -> int | None:
    number = _as_float(value)
    return int(number) if number is not None else None


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "available"}


def _numeric_series(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(series, errors="coerce")


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _wizard_evidence_markdown(frame: pd.DataFrame) -> str:
    lines = ["# Wizard Evidence Summary", "", f"Generated: {datetime.now(timezone.utc).isoformat()}", ""]
    lines.append(f"- rows: {len(frame)}")
    if frame.empty:
        return "\n".join(lines) + "\n"
    lines.append(f"- exact mode valid: {int(frame['mode_valid'].astype(bool).sum())}")
    lines.append(f"- Sharpe >= {DISCOVERY_MIN_SHARPE:g}: {int(_passes_sharpe_gate(frame).astype(bool).sum())}")
    lines.append(f"- returns_total > {DISCOVERY_MIN_RETURNS_TOTAL:.0%}: {int(frame['passes_returns_total_gt_20pct'].astype(bool).sum())}")
    lines.extend(["", "## Top Rows", ""])
    preview_cols = ["pair", "interval", "exact_mode", "sharpe", "returns_total", "mode_blocker", "source_path"]
    lines.append(frame[preview_cols].head(20).to_markdown(index=False))
    return "\n".join(lines) + "\n"


def _simple_markdown(title: str, frame: pd.DataFrame) -> str:
    lines = [f"# {title}", "", f"Generated: {datetime.now(timezone.utc).isoformat()}", "", f"- rows: {len(frame)}", ""]
    if not frame.empty:
        lines.append(frame.head(30).to_markdown(index=False))
    return "\n".join(lines) + "\n"


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False
