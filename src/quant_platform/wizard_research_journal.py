from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re

import pandas as pd

from quant_platform.active_pipeline import CommandResult
from quant_platform.wizard_evidence import ids_from_exact_mode
from quant_platform.wizard_mode_replay import normalize_exact_mode
from quant_platform.wizard_run_config import WIZARD_CONFIG_SCHEMA_VERSION


ROOT = Path(__file__).resolve().parents[2]
ACTIVE = ROOT / "reports" / "active"
REPORTS = ROOT / "reports"
CANONICAL_TIMEFRAMES = ("Daily", "4 Hour", "1 Hour", "5 Min")
LIVE_DASHBOARD_MAX_AGE = pd.Timedelta(hours=2, minutes=15)
PAIR_PAGE_CAPTURE_COLUMNS = [
    "pair",
    "asset_x",
    "asset_y",
    "interval",
    "period",
    "dashboard_recommended_strategy",
    "exact_mode",
    "spread_id",
    "strategy_id",
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
    "closed_trades",
    "sharpe",
    "returns_total",
    "returns_total_pct",
    "source_path",
    "evidence_path",
    "capture_context",
    "capture_timestamp_utc",
    "pair_page_url",
    "price_x",
    "price_y",
    "return_x_pct",
    "return_y_pct",
    "volume_x_top",
    "volume_y_top",
    "lt_vol_x",
    "lt_vol_y",
    "dependency_x_over_y_top",
    "dependency_y_over_x_top",
    "coint_johansen_top",
    "coint_engle_granger_top",
    "johansen_badge_state",
    "engle_granger_badge_state",
    "hurst_top",
    "half_life_top",
    "correlation_top",
    "hedge_ratio_top",
    "lt_beta",
    "max_drawdown_top",
    "return_total_top",
    "sharpe_top",
    "chart_left_mode",
    "chart_left_latest_x_value",
    "chart_left_latest_y_value",
    "chart_right_mode",
    "chart_right_latest_value",
    "ou_mu",
    "ou_alpha",
    "ou_beta",
    "ou_B",
    "ou_sigma",
    "conditional_chart_value",
    "copula_chart_mode",
    "copula_direction_view",
    "copula_entry_lower",
    "copula_entry_upper",
    "copula_exit_lower",
    "copula_exit_upper",
    "secondary_chart_mode",
    "entry_long_operator",
    "entry_long_value",
    "entry_long_unit",
    "entry_short_operator",
    "entry_short_value",
    "entry_short_unit",
    "exit_long_operator",
    "exit_long_value",
    "exit_long_unit",
    "exit_short_operator",
    "exit_short_value",
    "exit_short_unit",
    "close_n_periods_mode",
    "stop_loss_pct",
    "ecm_deviation_min_pct",
    "corr_strength_min_pct",
    "capital_weighting_slider_value",
    "capital_weighting_asset",
    "sortino_detail",
    "mean_period_return_detail",
    "win_rate",
    "max_drawdown_detail",
    "var_99_detail",
    "cvar_99_detail",
    "var_sim",
    "cvar_sim",
    "bottom_chart_mode",
    "bottom_chart_latest_net_value",
    "bottom_chart_latest_x_value",
    "bottom_chart_latest_y_value",
]


def _read_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _write_csv_atomic(frame: pd.DataFrame, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(output)
    return output


def _write_json_atomic(payload: object, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=False), encoding="utf-8")
    tmp.replace(output)
    return output


def _write_text_atomic(text: str, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(output)
    return output


def _matrix_source_paths(root: Path) -> list[Path]:
    paths: list[Path] = []
    for pattern in ("reports/active/pair1_*matrix.json", "archive/**/pair1_*matrix.json"):
        paths.extend(root.glob(pattern))
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in sorted(paths):
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        deduped.append(path)
    return deduped


def _schema_fields(root: Path) -> list[str]:
    schema = pd.read_csv(root / "reports" / "wizard_research_journal_schema.csv")
    return schema["field_name"].dropna().astype(str).tolist()


def _load_json_list(path: Path) -> list[dict[str, object]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    return payload if isinstance(payload, list) else []


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def _clean_num(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and not pd.isna(value):
        return float(value)
    text = _clean_text(value).replace("%", "").replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _clean_bool(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = _clean_text(value).lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def _badge_state(*values: object) -> str:
    """Normalize the dashboard's visual stationarity legend without inferring color."""
    for value in values:
        text = _clean_text(value).lower().replace("-", "_").replace(" ", "_")
        if text in {"green", "confirmed", "pass", "passed", "cointegrated"}:
            return "confirmed"
        if text in {"orange", "trending", "trend"}:
            return "trending"
        if text in {"grey", "gray", "none", "not_confirmed", "failed", "fail"}:
            return "not_confirmed"
    return "unknown"


def _stationarity_summary(johansen: object, engle_granger: object) -> str:
    jn = _clean_text(johansen) or "unknown"
    eg = _clean_text(engle_granger) or "unknown"
    if jn == "confirmed" and eg == "confirmed":
        return "both_confirmed"
    if eg == "trending":
        return "engle_granger_trending"
    if jn == "confirmed":
        return "johansen_confirmed"
    if eg == "confirmed":
        return "engle_granger_confirmed"
    if jn == "not_confirmed" and eg == "not_confirmed":
        return "none_confirmed"
    return "badge_state_unknown"


def _probability_percent(value: object) -> float | None:
    numeric = _clean_num(value)
    if numeric is None:
        return None
    return numeric * 100.0 if -1.0 <= numeric <= 1.0 else numeric


def _apply_stationarity_and_copula_interpretation(record: dict[str, object]) -> None:
    """Write an explicit research hypothesis; this never clears a live trade."""
    johansen = _clean_text(record.get("johansen_badge_state")) or "unknown"
    engle_granger = _clean_text(record.get("engle_granger_badge_state")) or "unknown"
    record["stationarity_summary"] = _stationarity_summary(johansen, engle_granger)

    x_probability = _probability_percent(
        record.get("copula_x_given_y")
        if record.get("copula_x_given_y") is not None
        else record.get("dependency_x_over_y")
    )
    y_probability = _probability_percent(
        record.get("copula_y_given_x")
        if record.get("copula_y_given_x") is not None
        else record.get("dependency_y_over_x")
    )
    record["copula_x_given_y_pct"] = x_probability
    record["copula_y_given_x_pct"] = y_probability
    if x_probability is None or y_probability is None:
        record.update(
            {
                "copula_signal_status": "missing_conditional_probabilities",
                "copula_probability_gap_pct": None,
                "copula_rich_asset": "",
                "copula_cheap_asset": "",
                "copula_trade_direction": "",
                "copula_entry_confirmation": "",
                "copula_exit_condition": "",
                "copula_journal_status": "incomplete_capture",
                "copula_execution_blockers": "missing_conditional_probabilities",
            }
        )
        return

    asset_x = _clean_text(record.get("asset_x"))
    asset_y = _clean_text(record.get("asset_y"))
    gap = abs(x_probability - y_probability)
    record["copula_probability_gap_pct"] = gap
    is_extreme = max(x_probability, y_probability) >= 95.0 and min(x_probability, y_probability) <= 5.0
    is_candidate = max(x_probability, y_probability) >= 90.0 and min(x_probability, y_probability) <= 10.0
    if not is_candidate:
        record.update(
            {
                "copula_signal_status": "no_asymmetric_dislocation",
                "copula_rich_asset": "",
                "copula_cheap_asset": "",
                "copula_trade_direction": "",
                "copula_entry_confirmation": "",
                "copula_exit_condition": "",
                "copula_journal_status": "watch_only",
                "copula_execution_blockers": "conditional_probabilities_not_extreme",
            }
        )
        return

    x_is_rich = x_probability > y_probability
    record.update(
        {
            "copula_signal_status": "strong_asymmetric_dislocation" if is_extreme else "asymmetric_dislocation",
            "copula_rich_asset": asset_x if x_is_rich else asset_y,
            "copula_cheap_asset": asset_y if x_is_rich else asset_x,
            "copula_trade_direction": "short_x_long_y" if x_is_rich else "long_x_short_y",
            "copula_entry_confirmation": "completed_bar_and_current_liquidity_check",
            "copula_exit_condition": "both_conditionals_return_to_45_to_55_pct",
        }
    )
    blockers: list[str] = []
    if engle_granger == "trending":
        blockers.append("engle_granger_trending")
    elif record["stationarity_summary"] == "none_confirmed":
        blockers.append("no_stationarity_confirmation")
    elif record["stationarity_summary"] == "badge_state_unknown":
        blockers.append("stationarity_badge_state_not_captured")
    observed_volumes = [
        value
        for value in [
            _clean_num(record.get("volume_x")),
            _clean_num(record.get("volume_y")),
            _clean_num(record.get("volume_x_top")),
            _clean_num(record.get("volume_y_top")),
        ]
        if value is not None
    ]
    if any(value <= 0 for value in observed_volumes):
        blockers.append("zero_displayed_liquidity")
    record["copula_execution_blockers"] = ";".join(blockers)
    record["copula_journal_status"] = "paper_candidate_pending_execution_checks" if not blockers else "research_only"


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def _slug(value: object) -> str:
    text = _clean_text(value).lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def _timeframe_sort_key(value: object) -> tuple[int, str]:
    text = _normalize_timeframe(value)
    order = {"Daily": 0, "4 Hour": 1, "1 Hour": 2, "5 Min": 3, "Live": 4}
    return order.get(text, 99), text


def _normalize_pair(value: object) -> str:
    text = re.sub(r"\s+", "", _clean_text(value).upper()).replace("/", "-")
    parts = [part for part in text.split("-") if part]
    if len(parts) == 4 and parts[1] == "USD" and parts[3] == "USD":
        return f"{parts[0]}-USD/{parts[2]}-USD"
    if len(parts) == 2:
        return f"{parts[0]}/{parts[1]}"
    return text.replace("--", "-")


def _pair_assets(pair: object, asset_x: object = "", asset_y: object = "") -> tuple[str, str]:
    x = _clean_text(asset_x).upper()
    y = _clean_text(asset_y).upper()
    if x and y:
        return x, y
    text = _normalize_pair(pair)
    if "/" in text:
        left, right = text.split("/", 1)
        return left, right
    return x, y


def _normalize_timeframe(value: object) -> str:
    text = _clean_text(value).lower()
    mapping = {
        "daily": "Daily",
        "day": "Daily",
        "1d": "Daily",
        "4 hour": "4 Hour",
        "4hour": "4 Hour",
        "4h": "4 Hour",
        "1 hour": "1 Hour",
        "1hour": "1 Hour",
        "1h": "1 Hour",
        "5 min": "5 Min",
        "5min": "5 Min",
        "5 mins": "5 Min",
        "5mins": "5 Min",
        "live": "Live",
        "scanner": "Live",
    }
    return mapping.get(text, _clean_text(value))


def _normalize_strategy(value: object) -> str:
    text = _clean_text(value)
    aliases = {
        "Static (ZScoreR)": "Static (ZScoreR)",
        "Static (ZScore)": "Static (ZScore)",
        "Dyn (ZScoreR)": "Dyn (ZScoreR)",
        "Dyn (ZScore)": "Dyn (ZScore)",
        "OU (ZScoreR)": "OU (ZScoreR)",
        "OU (ZScore)": "OU (ZScore)",
    }
    return aliases.get(text, text)


def _strategy_family(label: str) -> str:
    text = label.lower()
    if text.startswith("static"):
        return "Static"
    if text.startswith("dyn"):
        return "Dynamic"
    if text.startswith("ou"):
        return "OU"
    if "copula" in text:
        return "Copula"
    return ""


def _strategy_variant(label: str) -> str:
    text = label.lower()
    if "zscore" in text:
        return "ZScore"
    if "spread" in text:
        return "Spread"
    if "copula" in text:
        return "Copula"
    return ""


def _pair_id(pair: str) -> str:
    return _slug(pair.replace("/", "-"))


def _pair_markets(pair: str) -> list[str]:
    left, right = _pair_assets(pair)
    return [left, right] if left and right else []


def _green_source(norm_value: object, roll_value: object) -> str:
    norm = _clean_num(norm_value)
    roll = _clean_num(roll_value)
    norm_green = norm is not None and abs(norm) >= 1.5
    roll_green = roll is not None and abs(roll) >= 1.5
    if norm_green and roll_green:
        return "both"
    if norm_green:
        return "normal"
    if roll_green:
        return "rolling"
    if norm is not None and roll is not None:
        return "present_not_green"
    return ""


def _trigger_state(norm_value: object, roll_value: object) -> str:
    candidates = [abs(v) for v in (_clean_num(norm_value), _clean_num(roll_value)) if v is not None]
    if not candidates:
        return "unknown"
    value = max(candidates)
    if value >= 2.0:
        return "at-trigger"
    if value >= 1.5:
        return "near-trigger"
    return "pre-trigger"


def _paper_candidate_status(pair: str, timeframe: str, strategy: str, decisions: pd.DataFrame) -> tuple[str, dict[str, object]]:
    if decisions.empty:
        return "", {}
    pair_norm = _normalize_pair(pair)
    decision_rows = decisions.copy()
    decision_rows["pair_norm"] = decision_rows.get("pair", pd.Series(dtype=object)).map(_normalize_pair)
    match = decision_rows[decision_rows["pair_norm"] == pair_norm]
    if timeframe:
        tf = _slug(timeframe)
        if "candidate_id" in match.columns:
            tf_match = match["candidate_id"].astype(str).str.contains(tf, case=False, na=False)
            if tf_match.any():
                match = match[tf_match]
    if strategy:
        strategy_slug = _slug(strategy)
        if "candidate_id" in match.columns:
            strat_match = match["candidate_id"].astype(str).str.contains(strategy_slug, case=False, na=False)
            if strat_match.any():
                match = match[strat_match]
    if match.empty:
        return "", {}
    row = match.sort_values("decision_rank", na_position="last").iloc[0].to_dict()
    recommendation = _clean_text(row.get("recommendation"))
    gate_ready = _clean_bool(row.get("paper_gate_ready"))
    if gate_ready is True:
        return "paper_gate_ready", row
    if recommendation:
        return recommendation, row
    return _clean_text(row.get("current_submit_state")), row


def _readiness_label(pair: str, timeframe: str, strategy: str, decisions: pd.DataFrame, compatible: bool, account_blocked: bool) -> str:
    status, row = _paper_candidate_status(pair, timeframe, strategy, decisions)
    if account_blocked:
        return "blocked_account_state"
    if not compatible:
        return "blocked_execution_route"
    if status == "paper_gate_ready":
        return "paper_gate_ready"
    if status:
        return f"decision_{_slug(status)}"
    reason = _clean_text(row.get("reason"))
    return f"decision_{_slug(reason)}" if reason else "research_only"


def _compatibility_lookup(frame: pd.DataFrame) -> dict[str, bool]:
    if frame.empty or "market" not in frame.columns:
        return {}
    lookup: dict[str, bool] = {}
    for row in frame.to_dict(orient="records"):
        market = _clean_text(row.get("market")).upper()
        if not market:
            continue
        lookup[market] = bool(_clean_bool(row.get("compatible_for_paper_submit")))
    return lookup


def _pair_execution_compatible(pair: str, compatibility_lookup: dict[str, bool]) -> bool:
    markets = _pair_markets(pair)
    return bool(markets) and all(bool(compatibility_lookup.get(market, False)) for market in markets)


def _watch_lookup(frame: pd.DataFrame) -> dict[str, dict[str, object]]:
    if frame.empty or "pair" not in frame.columns:
        return {}
    rows: dict[str, dict[str, object]] = {}
    for row in frame.to_dict(orient="records"):
        pair = _normalize_pair(row.get("pair"))
        if pair:
            rows[pair] = row
    return rows


def _parity_lookup(frame: pd.DataFrame) -> dict[tuple[str, str, str], dict[str, object]]:
    lookup: dict[tuple[str, str, str], dict[str, object]] = {}
    if frame.empty:
        return lookup
    for row in frame.to_dict(orient="records"):
        key = (
            _normalize_pair(row.get("pair")),
            _normalize_timeframe(row.get("interval")),
            _normalize_strategy(row.get("exact_mode")),
        )
        lookup[key] = row
    return lookup


def _shared_outcome_lookup(root: Path) -> dict[tuple[str, str, str], dict[str, object]]:
    frame = _read_csv_or_empty(root / "reports" / "brain" / "shared_outcome_memory.csv")
    lookup: dict[tuple[str, str, str], dict[str, object]] = {}
    if frame.empty:
        return lookup
    for row in frame.to_dict(orient="records"):
        key = (
            _normalize_pair(row.get("pair")),
            _normalize_timeframe(row.get("timeframe")),
            _normalize_strategy(row.get("strategy_mode")),
        )
        lookup[key] = row
    return lookup


def _paper_journal_lookup(root: Path) -> dict[str, dict[str, object]]:
    frame = _read_csv_or_empty(root / "reports" / "paper_trading_journal.csv")
    if frame.empty or "pair" not in frame.columns:
        return {}
    lookup: dict[str, dict[str, object]] = {}
    for row in frame.to_dict(orient="records"):
        pair = _normalize_pair(row.get("pair"))
        if pair:
            lookup[pair] = row
    return lookup


def _matrix_rows_to_pair_page_rows(path: Path) -> list[dict[str, object]]:
    rows = _load_json_list(path)
    converted: list[dict[str, object]] = []
    capture_ts = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
    try:
        relative_path = str(path.relative_to(ROOT))
    except ValueError:
        # Some archive sources are already stored as repo-relative paths.
        relative_path = str(path)
    for row in rows:
        pair = _normalize_pair(row.get("pair"))
        strategy = _normalize_strategy(row.get("strategy"))
        timeframe = _normalize_timeframe(row.get("timeframe"))
        asset_x, asset_y = _pair_assets(pair)
        spread_id, strategy_id = ids_from_exact_mode(strategy)
        converted.append(
            {
                "pair": pair.replace("/", "-"),
                "asset_x": asset_x,
                "asset_y": asset_y,
                "interval": timeframe,
                "period": _clean_text(row.get("periods")),
                "dashboard_recommended_strategy": strategy,
                "exact_mode": strategy,
                "spread_id": spread_id,
                "strategy_id": strategy_id,
                "pearson": None,
                "spearman": None,
                "kendall": None,
                "copula": _clean_text(row.get("best_fit")),
                "corr_copula": _clean_num(row.get("conditional_chart")),
                "u1_given_u2": None,
                "u2_given_u1": None,
                "ecm_x_available": None,
                "ecm_y_available": None,
                "ecm_strength_available": None,
                "closed_trades": _clean_num(row.get("closed_trades_metric")),
                "sharpe": _clean_num(row.get("sharpe_metric")),
                "returns_total": _clean_num(row.get("net_return_metric")),
                "returns_total_pct": _clean_num(row.get("annualized_return_metric")),
                "source_path": relative_path,
                "evidence_path": relative_path,
                "capture_context": "matrix_bridge_capture",
                "capture_timestamp_utc": capture_ts,
                "pair_page_url": "",
            }
        )
    return converted


def refresh_wizard_pair_page_capture_from_matrices(root: Path = ROOT) -> CommandResult:
    output = root / "reports" / "active" / "crypto_wizards_pair_page_capture.csv"
    current = _read_csv_or_empty(output)
    current_rows = current.to_dict(orient="records") if not current.empty else []
    bridged_rows: list[dict[str, object]] = []
    for path in _matrix_source_paths(root):
        bridged_rows.extend(_matrix_rows_to_pair_page_rows(path))

    combined_rows = current_rows + bridged_rows
    if not combined_rows:
        frame = pd.DataFrame(columns=PAIR_PAGE_CAPTURE_COLUMNS)
        _write_csv_atomic(frame, output)
        return CommandResult(paths={"pair_page_capture": output}, summary={"rows": 0, "bridged_rows": 0})

    frame = pd.DataFrame(combined_rows)
    for column in PAIR_PAGE_CAPTURE_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    frame["interval"] = frame.get("interval", pd.Series(dtype=object)).map(_normalize_timeframe)
    frame["exact_mode"] = frame.get("exact_mode", pd.Series(dtype=object)).map(_normalize_strategy)
    frame["pair"] = frame.get("pair", pd.Series(dtype=object)).map(_normalize_pair).str.replace("/", "-", regex=False)
    frame = frame[
        frame["pair"].astype(str).str.strip().ne("")
        & frame["pair"].astype(str).str.strip().ne("-")
        & frame["exact_mode"].astype(str).str.strip().ne("")
    ].copy()
    frame["_pair_key"] = frame["pair"].astype(str)
    frame["_strategy_key"] = frame["exact_mode"].astype(str)
    frame["_interval_key"] = frame["interval"].astype(str)
    frame["_priority"] = frame.get("capture_context", pd.Series(dtype=object)).astype(str).map(
        lambda value: 0
        if value == "browser_page_two_capture"
        else 1
        if value not in {"matrix_bridge_capture"}
        else 2
    )
    frame["_capture_ts"] = pd.to_datetime(frame.get("capture_timestamp_utc", pd.Series(dtype=object)), utc=True, errors="coerce")
    frame = frame.sort_values(
        by=["_pair_key", "_interval_key", "_strategy_key", "_priority", "_capture_ts"],
        ascending=[True, True, True, False, False],
        na_position="last",
    )
    frame = frame.drop_duplicates(subset=["_pair_key", "_interval_key", "_strategy_key"], keep="first")
    frame = frame.sort_values(
        by=["_pair_key", "_interval_key", "_strategy_key"],
        key=lambda col: col.map(_timeframe_sort_key) if col.name == "_interval_key" else col,
        na_position="last",
    )
    frame = frame[PAIR_PAGE_CAPTURE_COLUMNS].reset_index(drop=True)
    _write_csv_atomic(frame, output)
    return CommandResult(
        paths={"pair_page_capture": output},
        summary={
            "rows": int(len(frame)),
            "bridged_rows": int(len(bridged_rows)),
            "four_hour_rows": int(frame.get("interval", pd.Series(dtype=object)).astype(str).eq("4 Hour").sum()),
        },
    )


def _why_selected_from_row(row: dict[str, object]) -> str:
    parts: list[str] = []
    green = _clean_text(row.get("zscore_green_source"))
    if green:
        parts.append(f"green={green}")
    norm = _clean_num(row.get("zscore_norm_value"))
    roll = _clean_num(row.get("zscore_roll_value"))
    if norm is not None:
        parts.append(f"norm_z={norm:.2f}")
    if roll is not None:
        parts.append(f"roll_z={roll:.2f}")
    sharpe = _clean_num(row.get("sharpe"))
    if sharpe is not None:
        parts.append(f"sharpe={sharpe:.2f}")
    ret = _clean_num(row.get("return_total"))
    if ret is not None:
        parts.append(f"return={ret:.2f}")
    corr = _clean_num(row.get("correlation_value"))
    if corr is not None:
        parts.append(f"corr={corr:.2f}")
    return "; ".join(parts)


def _blank_record(schema_fields: list[str]) -> dict[str, object]:
    return {field: None for field in schema_fields}


def _build_scanner_records(
    schema_fields: list[str],
    root: Path,
    decisions: pd.DataFrame,
    watch_lookup: dict[str, dict[str, object]],
    compatibility_lookup: dict[str, bool],
) -> pd.DataFrame:
    live_capture = _read_csv_or_empty(root / "reports" / "active" / "crypto_wizards_live_scanner_capture.csv")
    alias = _read_csv_or_empty(root / "reports" / "active" / "wizard_hourly_database" / "wizard_hourly_current_alias.csv")
    source_rows: list[dict[str, object]] = []
    if not live_capture.empty:
        source_rows.extend(live_capture.to_dict(orient="records"))
    if not alias.empty:
        source_rows.extend(alias.to_dict(orient="records"))

    if not source_rows:
        return pd.DataFrame(columns=["journal_layer", *schema_fields])

    records: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    refresh_ts = _now_iso()
    for row in source_rows:
        pair = _normalize_pair(row.get("pair"))
        strategy = _normalize_strategy(
            row.get("dashboard_recommended_strategy") or row.get("strategy") or row.get("exact_mode")
        )
        if not pair:
            continue
        dedupe_key = (pair, strategy)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        asset_x, asset_y = _pair_assets(pair, row.get("asset_x"), row.get("asset_y"))
        record = _blank_record(schema_fields)
        record.update(
            {
                "capture_id": f"scanner::{_pair_id(pair)}::{_slug(strategy or 'unknown')}",
                "capture_timestamp_utc": _clean_text(row.get("capture_timestamp_utc") or row.get("scan_timestamp") or refresh_ts),
                "scanner_refresh_timestamp_utc": refresh_ts,
                "pair": pair,
                "asset_x": asset_x,
                "asset_y": asset_y,
                "venue": _clean_text(row.get("exchange") or "dydx").lower(),
                "exchange_lane": _clean_text(row.get("exchange") or "dydx").lower(),
                "page_route": "scanner",
                "source_cycle_id": "wizard_research_refresh",
                "source_run_type": "automation",
                "scanner_sort_mode": _clean_text(row.get("scanner_sort_mode") or row.get("scanner_priority") or "dashboard_priority"),
                "scanner_cointegration_filter": _clean_text(row.get("scanner_cointegration_filter") or row.get("scanner_count_filter") or "all cases"),
                "scanner_correlation_filter": _clean_text(row.get("scanner_correlation_filter") or "all cases"),
                "scanner_hurst_filter": _clean_text(row.get("scanner_hurst_filter") or "all cases"),
                "scanner_half_life_filter": _clean_text(row.get("scanner_half_life_filter") or "all cases"),
                "scanner_copula_filter": _clean_text(row.get("scanner_copula_filter") or "all cases"),
                "scanner_strategy_filter": _clean_text(row.get("scanner_strategy_filter") or strategy or "all cases"),
                "scanner_exchange_filter": _clean_text(row.get("scanner_exchange_filter") or row.get("exchange") or "dydx"),
                "scanner_symbol_filter_text": "",
                "row_index": _clean_num(row.get("dashboard_pair_rank") or row.get("row_index")),
                "visible_row_count": None,
                "asset_x_raw": asset_x,
                "asset_y_raw": asset_y,
                "normalized_pair": pair,
                "pair_id": _pair_id(pair),
                "timeframe": _normalize_timeframe(row.get("timeframe") or row.get("interval")),
                "strategy_label": strategy,
                "volume_x": _clean_num(row.get("volume_x")),
                "volume_y": _clean_num(row.get("volume_y")),
                "updated_at_utc": _clean_text(row.get("updated_at_utc") or row.get("scan_timestamp")),
                "strategy_family_from_row": _strategy_family(strategy),
                "strategy_variant_from_row": _strategy_variant(strategy),
                "zscore_norm_value": _clean_num(row.get("zscore_norm")),
                "zscore_roll_value": _clean_num(row.get("zscore_roll")),
                "dependency_profile": _clean_text(row.get("dependency_profile")),
                "dependency_x_over_y": _clean_num(row.get("dependency_x_over_y")),
                "dependency_y_over_x": _clean_num(row.get("dependency_y_over_x")),
                "correlation_value": _clean_num(row.get("correlation") or row.get("top_corr")),
                "coint_johansen_flag": _clean_bool(row.get("jn_flag")),
                "coint_engle_granger_flag": _clean_bool(row.get("eg_flag")),
                "johansen_badge_state": _badge_state(
                    row.get("johansen_badge_state"),
                    row.get("johansen_status"),
                    row.get("jn_badge_state"),
                    row.get("jn_status"),
                ),
                "engle_granger_badge_state": _badge_state(
                    row.get("engle_granger_badge_state"),
                    row.get("engle_granger_status"),
                    row.get("eg_badge_state"),
                    row.get("eg_status"),
                ),
                "hurst_value": _clean_num(row.get("hurst") or row.get("top_hurst")),
                "half_life_value": _clean_num(row.get("half_life") or row.get("top_half_life")),
                "sigma_0_count": _clean_num(row.get("sigma_0_count")),
                "sigma_1_count": _clean_num(row.get("sigma_1_count")),
                "sigma_2_count": _clean_num(row.get("sigma_2_count")),
                "var_99": _clean_num(row.get("var")),
                "cvar_99": _clean_num(row.get("cvar")),
                "max_drawdown": _clean_num(row.get("mdd") or row.get("top_mdd") or row.get("max_drawdown_pct")),
                "return_total": _clean_num(
                    row.get("return_total")
                    or row.get("returns_total")
                    or row.get("top_returns_pct")
                    or row.get("annualized_return_pct")
                ),
                "sharpe": _clean_num(row.get("sharpe") or row.get("top_sharpe")),
                "scanner_row_text_raw": json.dumps(row, sort_keys=True, default=str),
                "scanner_row_screenshot_path": "",
                "scanner_snapshot_path": _clean_text(row.get("source_path") or row.get("evidence_path")),
                "capture_status": "captured" if not live_capture.empty else "derived_from_alias",
                "no_data_flag": False,
            }
        )
        record["zscore_green_source"] = _green_source(record.get("zscore_norm_value"), record.get("zscore_roll_value"))
        record["zscore_signal_type"] = record["zscore_green_source"]
        _apply_stationarity_and_copula_interpretation(record)
        record["paper_candidate_status"], decision_row = _paper_candidate_status(pair, "", strategy, decisions)
        compatible = _pair_execution_compatible(pair, compatibility_lookup)
        watch = watch_lookup.get(pair, {})
        orphan = _clean_text(watch.get("lifecycle_status")).lower() == "orphan_leg"
        record["execution_compatible"] = compatible
        record["account_state_blocked"] = orphan
        record["orphan_leg_blocker"] = orphan
        record["readiness_label"] = _readiness_label(pair, "", strategy, decisions, compatible, orphan)
        record["why_selected"] = _why_selected_from_row(record)
        if decision_row:
            record["capture_blocker"] = _clean_text(decision_row.get("reason"))
        records.append({"journal_layer": "scanner_capture", **record})
    return pd.DataFrame(records)


def _build_detail_records(
    schema_fields: list[str],
    root: Path,
    decisions: pd.DataFrame,
    watch_lookup: dict[str, dict[str, object]],
    compatibility_lookup: dict[str, bool],
    parity_lookup: dict[tuple[str, str, str], dict[str, object]],
    shared_outcomes: dict[tuple[str, str, str], dict[str, object]],
    paper_journal: dict[str, dict[str, object]],
) -> pd.DataFrame:
    frame = _read_csv_or_empty(root / "reports" / "active" / "crypto_wizards_pair_page_capture.csv")
    if frame.empty:
        return pd.DataFrame(columns=["journal_layer", *schema_fields])

    refresh_ts = _now_iso()
    rows: list[dict[str, object]] = []
    actual_keys: set[tuple[str, str, str]] = set()

    for source_row in frame.to_dict(orient="records"):
        pair = _normalize_pair(source_row.get("pair"))
        strategy = _normalize_strategy(source_row.get("exact_mode") or source_row.get("dashboard_recommended_strategy"))
        timeframe = _normalize_timeframe(source_row.get("interval"))
        asset_x, asset_y = _pair_assets(pair, source_row.get("asset_x"), source_row.get("asset_y"))
        key = (pair, timeframe, strategy)
        actual_keys.add(key)

        record = _blank_record(schema_fields)
        record.update(
            {
                "capture_id": f"detail::{_pair_id(pair)}::{_slug(timeframe)}::{_slug(strategy)}",
                "capture_timestamp_utc": _clean_text(source_row.get("capture_timestamp_utc") or refresh_ts),
                "scanner_refresh_timestamp_utc": _clean_text(source_row.get("capture_timestamp_utc") or refresh_ts),
                "pair": pair,
                "asset_x": asset_x,
                "asset_y": asset_y,
                "venue": "dydx",
                "exchange_lane": "dydx",
                "page_route": _clean_text(source_row.get("pair_page_url") or "pair/detail"),
                "source_cycle_id": "wizard_research_refresh",
                "source_run_type": "automation",
                "normalized_pair": pair,
                "pair_id": _pair_id(pair),
                "timeframe": timeframe,
                "strategy_label": strategy,
                "periods_input": _clean_text(source_row.get("period")),
                "periods_analyzed": _clean_text(source_row.get("period")),
                "detail_capture_timestamp_utc": _clean_text(source_row.get("capture_timestamp_utc") or refresh_ts),
                "page_detail_screenshot_path": "",
                "page_detail_full_screenshot_path": "",
                "changed_vs_last_capture": False,
                "change_reason": "",
                "price_x": _clean_num(source_row.get("price_x")),
                "price_y": _clean_num(source_row.get("price_y")),
                "return_x_pct": _clean_num(source_row.get("return_x_pct")),
                "return_y_pct": _clean_num(source_row.get("return_y_pct")),
                "volume_x_top": _clean_num(source_row.get("volume_x_top")),
                "volume_y_top": _clean_num(source_row.get("volume_y_top")),
                "lt_vol_x": _clean_num(source_row.get("lt_vol_x")),
                "lt_vol_y": _clean_num(source_row.get("lt_vol_y")),
                "dependency_x_over_y_top": _clean_num(source_row.get("dependency_x_over_y_top")),
                "dependency_y_over_x_top": _clean_num(source_row.get("dependency_y_over_x_top")),
                "coint_johansen_top": _clean_bool(source_row.get("coint_johansen_top")),
                "coint_engle_granger_top": _clean_bool(source_row.get("coint_engle_granger_top")),
                "johansen_badge_state": _badge_state(
                    source_row.get("johansen_badge_state"),
                    source_row.get("coint_johansen_badge_state"),
                    source_row.get("johansen_status"),
                ),
                "engle_granger_badge_state": _badge_state(
                    source_row.get("engle_granger_badge_state"),
                    source_row.get("coint_engle_granger_badge_state"),
                    source_row.get("engle_granger_status"),
                ),
                "hurst_top": _clean_num(source_row.get("hurst_top")),
                "half_life_top": _clean_num(source_row.get("half_life_top")),
                "correlation_top": _clean_num(source_row.get("correlation_top")),
                "hedge_ratio_top": _clean_num(source_row.get("hedge_ratio_top")),
                "lt_beta": _clean_num(source_row.get("lt_beta")),
                "max_drawdown_top": _clean_num(source_row.get("max_drawdown_top")),
                "return_total_top": _clean_num(source_row.get("return_total_top")),
                "sharpe_top": _clean_num(source_row.get("sharpe_top")),
                "chart_left_mode": _clean_text(source_row.get("chart_left_mode")),
                "chart_left_latest_x_value": _clean_num(source_row.get("chart_left_latest_x_value")),
                "chart_left_latest_y_value": _clean_num(source_row.get("chart_left_latest_y_value")),
                "chart_right_mode": _clean_text(source_row.get("chart_right_mode")),
                "chart_right_latest_value": _clean_num(source_row.get("chart_right_latest_value")),
                "ou_mu": _clean_num(source_row.get("ou_mu")),
                "ou_alpha": _clean_num(source_row.get("ou_alpha")),
                "ou_beta": _clean_num(source_row.get("ou_beta")),
                "ou_B": _clean_num(source_row.get("ou_B")),
                "ou_sigma": _clean_num(source_row.get("ou_sigma")),
                "pearson_rho": _clean_num(source_row.get("pearson")),
                "spearman_rho": _clean_num(source_row.get("spearman")),
                "kendall_tau": _clean_num(source_row.get("kendall")),
                "conditional_chart_value": _clean_num(source_row.get("conditional_chart_value")),
                "copula_best_fit": _clean_text(source_row.get("copula")),
                "copula_correlation_rho": _clean_num(source_row.get("corr_copula")),
                "copula_x_given_y": _clean_num(source_row.get("u1_given_u2")),
                "copula_y_given_x": _clean_num(source_row.get("u2_given_u1")),
                "copula_chart_mode": _clean_text(source_row.get("copula_chart_mode")),
                "copula_direction_view": _clean_text(source_row.get("copula_direction_view")),
                "secondary_chart_mode": _clean_text(source_row.get("secondary_chart_mode")),
                "entry_long_operator": _clean_text(source_row.get("entry_long_operator")),
                "entry_long_value": _clean_num(source_row.get("entry_long_value")),
                "entry_long_unit": _clean_text(source_row.get("entry_long_unit")),
                "entry_short_operator": _clean_text(source_row.get("entry_short_operator")),
                "entry_short_value": _clean_num(source_row.get("entry_short_value")),
                "entry_short_unit": _clean_text(source_row.get("entry_short_unit")),
                "exit_long_operator": _clean_text(source_row.get("exit_long_operator")),
                "exit_long_value": _clean_num(source_row.get("exit_long_value")),
                "exit_long_unit": _clean_text(source_row.get("exit_long_unit")),
                "exit_short_operator": _clean_text(source_row.get("exit_short_operator")),
                "exit_short_value": _clean_num(source_row.get("exit_short_value")),
                "exit_short_unit": _clean_text(source_row.get("exit_short_unit")),
                "close_n_periods_mode": _clean_text(source_row.get("close_n_periods_mode")),
                "stop_loss_pct": _clean_num(source_row.get("stop_loss_pct")),
                "ecm_deviation_min_pct": _clean_num(source_row.get("ecm_deviation_min_pct")),
                "corr_strength_min_pct": _clean_num(source_row.get("corr_strength_min_pct")),
                "capital_weighting_slider_value": _clean_num(source_row.get("capital_weighting_slider_value")),
                "capital_weighting_asset": _clean_text(source_row.get("capital_weighting_asset")),
                "sharpe_detail": _clean_num(source_row.get("sharpe")),
                "sortino_detail": _clean_num(source_row.get("sortino_detail")),
                "sharpe_top": _clean_num(source_row.get("sharpe")),
                "net_return_detail": _clean_num(source_row.get("returns_total")),
                "annualized_return_detail": _clean_num(source_row.get("returns_total_pct")),
                "mean_period_return_detail": _clean_num(source_row.get("mean_period_return_detail")),
                "win_rate": _clean_num(source_row.get("win_rate")),
                "return_total_top": _clean_num(source_row.get("returns_total_pct")),
                "sharpe": _clean_num(source_row.get("sharpe")),
                "return_total": _clean_num(source_row.get("returns_total_pct")),
                "correlation_value": _clean_num(source_row.get("corr_copula") or source_row.get("pearson")),
                "correlation_top": _clean_num(source_row.get("corr_copula") or source_row.get("pearson")),
                "closed_trades": _clean_num(source_row.get("closed_trades")),
                "max_drawdown_detail": _clean_num(source_row.get("max_drawdown_detail")),
                "var_99_detail": _clean_num(source_row.get("var_99_detail")),
                "cvar_99_detail": _clean_num(source_row.get("cvar_99_detail")),
                "var_sim": _clean_num(source_row.get("var_sim")),
                "cvar_sim": _clean_num(source_row.get("cvar_sim")),
                "bottom_chart_mode": _clean_text(source_row.get("bottom_chart_mode")),
                "bottom_chart_latest_net_value": _clean_num(source_row.get("bottom_chart_latest_net_value")),
                "bottom_chart_latest_x_value": _clean_num(source_row.get("bottom_chart_latest_x_value")),
                "bottom_chart_latest_y_value": _clean_num(source_row.get("bottom_chart_latest_y_value")),
                "capture_status": "captured",
                "capture_blocker": "",
                "timeframe_checked_flag": True,
                "timeframe_available_flag": True,
                "strategy_available_flag": True,
                "timeframe_unavailable_flag": False,
                "strategy_unavailable_flag": False,
                "pair_unavailable_flag": False,
                "page_not_recognized_flag": False,
                "no_data_flag": False,
                "unchanged_alias_skipped_flag": False,
            }
        )
        record["strategy_family_from_row"] = _strategy_family(strategy)
        record["strategy_variant_from_row"] = _strategy_variant(strategy)
        record["chart_right_mode"] = "zscore"
        parity = parity_lookup.get(key, {})
        record["chart_right_latest_value"] = _clean_num(
            parity.get("wizard_rolling_zscore_last")
            if "zscorer" in strategy.lower()
            else parity.get("wizard_zscore_last")
        )
        record["zscore_norm_value"] = _clean_num(parity.get("wizard_zscore_last"))
        record["zscore_roll_value"] = _clean_num(parity.get("wizard_rolling_zscore_last"))
        record["zscore_green_source"] = _green_source(record.get("zscore_norm_value"), record.get("zscore_roll_value"))
        record["zscore_signal_type"] = record["zscore_green_source"]
        _apply_stationarity_and_copula_interpretation(record)
        record["top_hurst"] = _clean_num(source_row.get("hurst_top"))
        record["top_half_life"] = _clean_num(source_row.get("half_life_top"))
        outcome = shared_outcomes.get(key, {})
        journal_row = paper_journal.get(pair, {})
        record["paper_trade_id"] = _clean_text(outcome.get("candidate_id"))
        record["paper_entry_timestamp_utc"] = _clean_text(outcome.get("submission_timestamp") or journal_row.get("opened_timestamp_utc"))
        record["paper_exit_timestamp_utc"] = _clean_text(journal_row.get("closed_timestamp_utc"))
        record["paper_realized_return"] = _clean_num(outcome.get("realized_return") or journal_row.get("realized_return"))
        record["paper_outcome_label"] = _clean_text(outcome.get("outcome_label") or journal_row.get("outcome_label"))
        record["paper_exit_reason"] = _clean_text(journal_row.get("plan_reason"))
        record["paper_candidate_status"], decision_row = _paper_candidate_status(pair, timeframe, strategy, decisions)
        compatible = _pair_execution_compatible(pair, compatibility_lookup)
        watch = watch_lookup.get(pair, {})
        orphan = _clean_text(watch.get("lifecycle_status")).lower() == "orphan_leg"
        record["execution_compatible"] = compatible
        record["account_state_blocked"] = orphan
        record["orphan_leg_blocker"] = orphan
        record["readiness_label"] = _readiness_label(pair, timeframe, strategy, decisions, compatible, orphan)
        record["risk_gate_pass"] = True if _clean_num(record.get("max_drawdown_detail")) is None else _clean_num(record.get("max_drawdown_detail")) <= 25.0
        annual_return = _clean_num(record.get("annualized_return_detail"))
        sharpe = _clean_num(record.get("sharpe_detail"))
        record["return_gate_pass"] = annual_return is not None and annual_return > 0
        record["sharpe_gate_pass"] = sharpe is not None and sharpe >= 1.5
        record["drawdown_gate_pass"] = record["risk_gate_pass"]
        record["profit_factor_gate_pass"] = None
        record["why_selected"] = _why_selected_from_row(record)
        if decision_row:
            record["capture_blocker"] = _clean_text(decision_row.get("reason"))
        rows.append({"journal_layer": "pair_detail_capture", **record})

    # Add explicit placeholders for the canonical timeframes we want to track.
    base_pairs = sorted({(_normalize_pair(row.get("pair")), _normalize_strategy(row.get("exact_mode") or row.get("dashboard_recommended_strategy"))) for row in frame.to_dict(orient="records")})
    for pair, strategy in base_pairs:
        asset_x, asset_y = _pair_assets(pair)
        for timeframe in CANONICAL_TIMEFRAMES:
            key = (pair, timeframe, strategy)
            if key in actual_keys:
                continue
            record = _blank_record(schema_fields)
            record.update(
                {
                    "capture_id": f"detail::{_pair_id(pair)}::{_slug(timeframe)}::{_slug(strategy)}",
                    "capture_timestamp_utc": refresh_ts,
                    "scanner_refresh_timestamp_utc": refresh_ts,
                    "pair": pair,
                    "asset_x": asset_x,
                    "asset_y": asset_y,
                    "venue": "dydx",
                    "exchange_lane": "dydx",
                    "page_route": "pair/detail",
                    "source_cycle_id": "wizard_research_refresh",
                    "source_run_type": "automation",
                    "normalized_pair": pair,
                    "pair_id": _pair_id(pair),
                    "timeframe": timeframe,
                    "strategy_label": strategy,
                    "detail_capture_timestamp_utc": refresh_ts,
                    "capture_status": "missing_timeframe_capture",
                    "capture_blocker": "timeframe_not_captured_in_current_active_surface",
                    "timeframe_checked_flag": False,
                    "timeframe_available_flag": False,
                    "strategy_available_flag": True,
                    "pair_unavailable_flag": False,
                    "timeframe_unavailable_flag": True,
                    "strategy_unavailable_flag": False,
                    "page_not_recognized_flag": False,
                    "no_data_flag": True,
                    "unchanged_alias_skipped_flag": False,
                    "paper_candidate_status": "",
                    "execution_compatible": _pair_execution_compatible(pair, compatibility_lookup),
                    "account_state_blocked": False,
                    "orphan_leg_blocker": False,
                    "readiness_label": "missing_timeframe_capture",
                }
            )
            rows.append({"journal_layer": "pair_detail_capture", **record})

    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail

    # Fill deltas for repeated pair/timeframe/strategy rows.
    detail = detail.sort_values(["pair", "timeframe", "strategy_label", "capture_timestamp_utc"]).reset_index(drop=True)
    delta_pairs = [
        ("zscore_norm_value", "delta_zscore_norm"),
        ("zscore_roll_value", "delta_zscore_roll"),
        ("correlation_top", "delta_correlation"),
        ("hurst_top", "delta_hurst"),
        ("half_life_top", "delta_half_life"),
        ("hedge_ratio_top", "delta_hedge_ratio"),
        ("return_total_top", "delta_return_total"),
        ("sharpe_top", "delta_sharpe"),
        ("max_drawdown_top", "delta_max_drawdown"),
        ("capital_weighting_slider_value", "delta_capital_weighting"),
    ]
    group_keys = ["pair", "timeframe", "strategy_label"]
    for source_col, delta_col in delta_pairs:
        if source_col not in detail.columns:
            continue
        detail[delta_col] = detail.groupby(group_keys)[source_col].transform(lambda s: pd.to_numeric(s, errors="coerce").diff())
    return detail


def _build_markdown(scanner: pd.DataFrame, detail: pd.DataFrame, current_json_path: Path, snapshot_dir: Path) -> str:
    lines = [
        "# Wizard Research Journal",
        "",
        f"- Generated: {_now_iso()}",
        f"- Scanner rows: {len(scanner)}",
        f"- Pair detail rows: {len(detail)}",
        f"- Missing timeframe placeholders: {int(detail.get('timeframe_unavailable_flag', pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not detail.empty else 0}",
        f"- Current JSON: `{current_json_path}`",
        f"- Snapshots directory: `{snapshot_dir}`",
        "",
    ]
    if not detail.empty:
        preview = detail[
            ["pair", "timeframe", "strategy_label", "readiness_label", "paper_candidate_status", "capture_status"]
        ].head(20)
        lines.extend(["## Pair Detail Preview", "", "```text", preview.to_string(index=False), "```", ""])
    if not scanner.empty:
        preview = scanner[
            ["pair", "strategy_family_from_row", "zscore_green_source", "sharpe", "return_total", "capture_status"]
        ].head(20)
        lines.extend(["## Scanner Preview", "", "```text", preview.to_string(index=False), "```", ""])
    return "\n".join(lines)


TWO_HOUR_COPULA_COLUMNS = [
    "report_generated_at_utc",
    "pair",
    "venue",
    "scanner_capture_timestamp_utc",
    "detail_capture_timestamp_utc",
    "scanner_strategy",
    "scanner_volume_x",
    "scanner_volume_y",
    "scanner_updated_at_utc",
    "scanner_zscore_norm",
    "scanner_zscore_roll",
    "scanner_dependency_x_given_y_pct",
    "scanner_dependency_y_given_x_pct",
    "scanner_correlation_pct",
    "stationarity_johansen_state",
    "stationarity_engle_granger_state",
    "stationarity_summary",
    "scanner_hurst",
    "scanner_half_life",
    "scanner_sigma_0_count",
    "scanner_sigma_2_count",
    "scanner_var_99",
    "scanner_cvar_99",
    "scanner_max_drawdown",
    "scanner_return_total",
    "scanner_sharpe",
    "detail_timeframe",
    "detail_periods",
    "detail_strategy",
    "copula_best_fit",
    "copula_correlation_pct",
    "copula_x_given_y_pct",
    "copula_y_given_x_pct",
    "copula_probability_gap_pct",
    "copula_signal_status",
    "copula_rich_asset",
    "copula_cheap_asset",
    "copula_trade_direction",
    "capital_weighting",
    "copula_entry_confirmation",
    "copula_exit_condition",
    "copula_journal_status",
    "copula_execution_blockers",
    "copula_detail_capture_status",
    "copula_detail_missing_fields",
    "scanner_evidence_path",
    "detail_evidence_path",
]

COPULA_DETAIL_CAPTURE_COLUMNS = [
    "pair",
    "timeframe",
    "venue",
    "scanner_capture_timestamp_utc",
    "capture_status",
    "required_fields",
    "missing_fields",
    "request_reason",
    "scanner_evidence_path",
    "detail_evidence_path",
]

COPULA_DETAIL_REQUIRED_FIELDS = {
    "copula_best_fit": ("copula_best_fit",),
    "copula_correlation": ("copula_correlation_rho",),
    "conditional_x_given_y": ("copula_x_given_y_pct",),
    "conditional_y_given_x": ("copula_y_given_x_pct",),
    "johansen_badge": ("johansen_badge_state",),
    "engle_granger_badge": ("engle_granger_badge_state",),
    "hurst": ("hurst_top", "hurst_value"),
    "half_life": ("half_life_top", "half_life_value"),
    "correlation": ("correlation_top", "correlation_value"),
    "hedge_ratio": ("hedge_ratio_top",),
    "volume_x": ("volume_x_top", "volume_x"),
    "volume_y": ("volume_y_top", "volume_y"),
    "var_99": ("var_99_detail", "var_99"),
    "cvar_99": ("cvar_99_detail", "cvar_99"),
    "detail_evidence": ("page_route", "evidence_path", "pair_page_url"),
}

REQUIRED_STRATEGY_MODES = (
    "Static (Spread)",
    "Static (ZScoreR)",
    "Dyn (Spread)",
    "Dyn (ZScoreR)",
    "OU (Spread)",
    "OU (ZScoreR)",
    "Copula",
)

MODE_DETAIL_CAPTURE_COLUMNS = [
    "pair",
    "timeframe",
    "venue",
    "required_mode",
    "scanner_capture_timestamp_utc",
    "capture_status",
    "required_fields",
    "missing_fields",
    "request_reason",
    "scanner_evidence_path",
    "detail_evidence_path",
]

MODE_DETAIL_REQUIRED_FIELDS = {
    "exact_mode": ("strategy_label",),
    "spread_chart_mode": ("chart_left_mode",),
    "spread_chart_x": ("chart_left_latest_x_value",),
    "spread_chart_y": ("chart_left_latest_y_value",),
    "zscore_chart_mode": ("chart_right_mode",),
    "zscore_chart_value": ("chart_right_latest_value",),
    "hedge_ratio": ("hedge_ratio_top",),
    "correlation": ("correlation_top", "correlation_value"),
    "hurst": ("hurst_top", "hurst_value"),
    "half_life": ("half_life_top", "half_life_value"),
    "ecm_x": ("ecm_x_available",),
    "ecm_y": ("ecm_y_available",),
    "ecm_strength": ("ecm_strength_available",),
    "entry_long": ("entry_long_value",),
    "entry_short": ("entry_short_value",),
    "exit_long": ("exit_long_value",),
    "exit_short": ("exit_short_value",),
    "max_drawdown": ("max_drawdown_detail",),
    "var_99": ("var_99_detail",),
    "cvar_99": ("cvar_99_detail",),
    "volume_x": ("volume_x_top",),
    "volume_y": ("volume_y_top",),
    "detail_evidence": ("page_route", "evidence_path", "pair_page_url"),
}


def _latest_records_by_pair(frame: pd.DataFrame, layer: str) -> dict[str, dict[str, object]]:
    if frame.empty or "journal_layer" not in frame.columns:
        return {}
    rows = frame.loc[frame["journal_layer"].astype(str).eq(layer)].copy()
    if rows.empty:
        return {}
    rows["_captured_at"] = pd.to_datetime(rows.get("capture_timestamp_utc"), utc=True, errors="coerce")
    rows = rows.sort_values(["pair", "_captured_at"], na_position="last").drop_duplicates(subset=["pair"], keep="last")
    return {_clean_text(row.get("pair")): row.to_dict() for _, row in rows.iterrows() if _clean_text(row.get("pair"))}


def _field_available(record: dict[str, object], fields: tuple[str, ...]) -> bool:
    return any(bool(_clean_text(record.get(field))) for field in fields)


def _copula_detail_missing_fields(detail: dict[str, object] | None) -> list[str]:
    if not detail:
        return list(COPULA_DETAIL_REQUIRED_FIELDS)
    return [name for name, fields in COPULA_DETAIL_REQUIRED_FIELDS.items() if not _field_available(detail, fields)]


def _best_detail_records_by_pair_timeframe(frame: pd.DataFrame) -> dict[tuple[str, str], dict[str, object]]:
    if frame.empty or "journal_layer" not in frame.columns:
        return {}
    details = frame.loc[frame["journal_layer"].astype(str).eq("pair_detail_capture")].copy()
    if details.empty:
        return {}
    records: dict[tuple[str, str], dict[str, object]] = {}
    for _, row in details.iterrows():
        record = row.to_dict()
        key = (_clean_text(record.get("pair")), _normalize_timeframe(record.get("timeframe")))
        if not key[0]:
            continue
        existing = records.get(key)
        if existing is None or len(_copula_detail_missing_fields(record)) < len(_copula_detail_missing_fields(existing)):
            records[key] = record
    return records


def _best_detail_records_by_pair(frame: pd.DataFrame) -> dict[str, dict[str, object]]:
    """Return the strongest available detail capture for each pair.

    This is only used when a scanner capture omitted its timeframe. A scanner
    that names a timeframe must always use the exact pair/timeframe record.
    """
    records: dict[str, dict[str, object]] = {}
    for (pair, _), record in _best_detail_records_by_pair_timeframe(frame).items():
        existing = records.get(pair)
        if existing is None:
            records[pair] = record
            continue
        record_missing = len(_copula_detail_missing_fields(record))
        existing_missing = len(_copula_detail_missing_fields(existing))
        if record_missing < existing_missing:
            records[pair] = record
            continue
        if record_missing == existing_missing:
            record_captured_at = _captured_at(record)
            existing_captured_at = _captured_at(existing)
            if record_captured_at is not None and (
                existing_captured_at is None or record_captured_at > existing_captured_at
            ):
                records[pair] = record
    return records


def _best_detail_records_by_pair_timeframe_mode(frame: pd.DataFrame) -> dict[tuple[str, str, str], dict[str, object]]:
    if frame.empty or "journal_layer" not in frame.columns:
        return {}
    details = frame.loc[frame["journal_layer"].astype(str).eq("pair_detail_capture")].copy()
    if details.empty:
        return {}
    details["_captured_at"] = pd.to_datetime(details.get("detail_capture_timestamp_utc", details.get("capture_timestamp_utc")), utc=True, errors="coerce")
    details = details.sort_values("_captured_at", na_position="last")
    records: dict[tuple[str, str, str], dict[str, object]] = {}
    for _, row in details.iterrows():
        record = row.to_dict()
        key = (
            _clean_text(record.get("pair")),
            _normalize_timeframe(record.get("timeframe")),
            _normalize_strategy(record.get("strategy_label")),
        )
        if all(key):
            records[key] = record
    return records


def _mode_detail_missing_fields(detail: dict[str, object] | None, mode: str) -> list[str]:
    if not detail:
        missing = list(MODE_DETAIL_REQUIRED_FIELDS)
    else:
        missing = [name for name, fields in MODE_DETAIL_REQUIRED_FIELDS.items() if not _field_available(detail, fields)]
    if mode == "Copula":
        missing.extend(f"copula_{name}" for name in _copula_detail_missing_fields(detail))
    return missing


def _captured_at(record: dict[str, object] | None) -> pd.Timestamp | None:
    if not record:
        return None
    for field in ("detail_capture_timestamp_utc", "capture_timestamp_utc"):
        timestamp = pd.to_datetime(record.get(field), utc=True, errors="coerce")
        if pd.notna(timestamp):
            return timestamp
    return None


def _live_scanner_capture(scanner: dict[str, object], as_of: pd.Timestamp) -> bool:
    captured_at = _captured_at(scanner)
    return bool(captured_at is not None and pd.Timedelta(0) <= as_of - captured_at <= LIVE_DASHBOARD_MAX_AGE)


def _build_strategy_mode_capture_queue(
    combined: pd.DataFrame,
    root: Path,
    *,
    as_of: pd.Timestamp | None = None,
) -> tuple[Path, int, int]:
    """Require a current pair-page capture for every exact mode of every scanner candidate."""
    detail_by_key = _best_detail_records_by_pair_timeframe_mode(combined)
    scanners = combined.loc[combined.get("journal_layer", pd.Series(dtype=str)).astype(str).eq("scanner_capture")].copy()
    run_at = as_of or pd.Timestamp.now(tz="UTC")
    rows: list[dict[str, object]] = []
    for _, scanner_row in scanners.iterrows():
        scanner = scanner_row.to_dict()
        pair = _clean_text(scanner.get("pair"))
        timeframe = _normalize_timeframe(scanner.get("timeframe"))
        scanner_timestamp = _captured_at(scanner)
        scanner_is_live = _live_scanner_capture(scanner, run_at)
        if not pair or not timeframe:
            continue
        for mode in REQUIRED_STRATEGY_MODES:
            detail = detail_by_key.get((pair, timeframe, mode))
            missing = _mode_detail_missing_fields(detail, mode)
            detail_timestamp = _captured_at(detail)
            if not scanner_is_live:
                status = "dashboard_capture_required"
                missing = ["fresh_live_dashboard_scanner_capture", *missing]
            elif detail and bool(detail.get("strategy_unavailable_flag")):
                status = "mode_unavailable"
                missing = ["mode_unavailable_on_dashboard"]
            elif detail_timestamp is None or scanner_timestamp is None or detail_timestamp < scanner_timestamp:
                status = "capture_required"
                missing = ["fresh_live_pair_page_capture", *missing]
            else:
                status = "full_detail_captured" if not missing else "capture_required"
            rows.append(
                {
                    "pair": pair,
                    "timeframe": timeframe,
                    "venue": _clean_text(scanner.get("venue")),
                    "required_mode": mode,
                    "scanner_capture_timestamp_utc": _clean_text(scanner.get("capture_timestamp_utc")),
                    "capture_status": status,
                    "required_fields": ";".join(MODE_DETAIL_REQUIRED_FIELDS),
                    "missing_fields": ";".join(dict.fromkeys(missing)),
                    "request_reason": "live_scanner_candidate_requires_all_spread_modes_and_copula",
                    "scanner_evidence_path": _clean_text(scanner.get("scanner_snapshot_path")),
                    "detail_evidence_path": _clean_text((detail or {}).get("evidence_path") or (detail or {}).get("pair_page_url") or (detail or {}).get("page_route")),
                }
            )
    queue = pd.DataFrame(rows, columns=MODE_DETAIL_CAPTURE_COLUMNS)
    output = root / "reports" / "active" / "wizard_strategy_mode_capture_queue.csv"
    _write_csv_atomic(queue, output)
    pending = int((~queue.get("capture_status", pd.Series(dtype=str)).eq("full_detail_captured")).sum())
    return output, int(len(queue)), pending


def _build_copula_detail_capture_queue(
    combined: pd.DataFrame,
    root: Path,
    *,
    as_of: pd.Timestamp | None = None,
) -> tuple[Path, int, int]:
    """Create mandatory page-two capture tasks for every Copula-arbitrage scanner row."""
    detail_by_pair_timeframe = _best_detail_records_by_pair_timeframe(combined)
    scanner_rows = combined.loc[combined.get("journal_layer", pd.Series(dtype=str)).astype(str).eq("scanner_capture")].copy()
    run_at = as_of or pd.Timestamp.now(tz="UTC")
    rows: list[dict[str, object]] = []
    for _, scanner_row in scanner_rows.iterrows():
        scanner = scanner_row.to_dict()
        if "arbitrage" not in _clean_text(scanner.get("scanner_copula_filter")).lower():
            continue
        pair = _clean_text(scanner.get("pair"))
        timeframe = _normalize_timeframe(scanner.get("timeframe"))
        detail = detail_by_pair_timeframe.get((pair, timeframe))
        missing = _copula_detail_missing_fields(detail)
        scanner_is_live = _live_scanner_capture(scanner, run_at)
        status = "full_detail_captured" if not missing else "capture_required"
        if not scanner_is_live:
            status = "dashboard_capture_required"
            missing = ["fresh_live_dashboard_scanner_capture", *missing]
        rows.append(
            {
                "pair": pair,
                "timeframe": timeframe,
                "venue": _clean_text(scanner.get("venue")),
                "scanner_capture_timestamp_utc": _clean_text(scanner.get("capture_timestamp_utc")),
                "capture_status": status,
                "required_fields": ";".join(COPULA_DETAIL_REQUIRED_FIELDS),
                "missing_fields": ";".join(missing),
                "request_reason": "copula_arbitrage_scanner_discovery",
                "scanner_evidence_path": _clean_text(scanner.get("scanner_snapshot_path")),
                "detail_evidence_path": _clean_text((detail or {}).get("evidence_path") or (detail or {}).get("pair_page_url") or (detail or {}).get("page_route")),
            }
        )
    queue = pd.DataFrame(rows, columns=COPULA_DETAIL_CAPTURE_COLUMNS)
    output = root / "reports" / "active" / "wizard_copula_detail_capture_queue.csv"
    _write_csv_atomic(queue, output)
    pending = int((~queue.get("capture_status", pd.Series(dtype=str)).eq("full_detail_captured")).sum())
    return output, int(len(queue)), pending


def _first_record_value(field: str, *rows: dict[str, object] | None) -> object:
    for row in rows:
        if not row:
            continue
        value = row.get(field)
        if _clean_text(value):
            return value
    return None


def _build_two_hour_copula_report(
    combined: pd.DataFrame,
    root: Path,
    *,
    as_of: pd.Timestamp | None = None,
) -> tuple[Path, Path, int]:
    scanner_by_pair = _latest_records_by_pair(combined, "scanner_capture")
    detail_by_pair_timeframe = _best_detail_records_by_pair_timeframe(combined)
    detail_by_pair = _best_detail_records_by_pair(combined)
    report_rows: list[dict[str, object]] = []
    run_at = as_of or pd.Timestamp.now(tz="UTC")
    generated_at = run_at.isoformat()

    for pair in sorted(scanner_by_pair):
        scanner = scanner_by_pair.get(pair)
        scanner_filter = _clean_text((scanner or {}).get("scanner_copula_filter")).lower()
        if "arbitrage" not in scanner_filter:
            continue
        scanner_timeframe = _normalize_timeframe((scanner or {}).get("timeframe"))
        detail = detail_by_pair_timeframe.get((pair, scanner_timeframe))
        if detail is None and not scanner_timeframe:
            detail = detail_by_pair.get(pair)
        missing_fields = _copula_detail_missing_fields(detail)
        detail_status = "full_detail_captured" if not missing_fields else "capture_required"
        if not _live_scanner_capture(scanner or {}, run_at):
            detail_status = "dashboard_capture_required"
            missing_fields = ["fresh_live_dashboard_scanner_capture", *missing_fields]
        signal_status = _clean_text(_first_record_value("copula_signal_status", detail))
        report_rows.append(
            {
                "report_generated_at_utc": generated_at,
                "pair": pair,
                "venue": _first_record_value("venue", detail, scanner),
                "scanner_capture_timestamp_utc": _first_record_value("capture_timestamp_utc", scanner),
                "detail_capture_timestamp_utc": _first_record_value("detail_capture_timestamp_utc", detail),
                "scanner_strategy": _first_record_value("strategy_label", scanner),
                "scanner_volume_x": _first_record_value("volume_x", scanner),
                "scanner_volume_y": _first_record_value("volume_y", scanner),
                "scanner_updated_at_utc": _first_record_value("updated_at_utc", scanner),
                "scanner_zscore_norm": _first_record_value("zscore_norm_value", scanner),
                "scanner_zscore_roll": _first_record_value("zscore_roll_value", scanner),
                "scanner_dependency_x_given_y_pct": _first_record_value("copula_x_given_y_pct", scanner),
                "scanner_dependency_y_given_x_pct": _first_record_value("copula_y_given_x_pct", scanner),
                "scanner_correlation_pct": _probability_percent(_first_record_value("correlation_value", scanner)),
                "stationarity_johansen_state": _first_record_value("johansen_badge_state", detail, scanner),
                "stationarity_engle_granger_state": _first_record_value("engle_granger_badge_state", detail, scanner),
                "stationarity_summary": _first_record_value("stationarity_summary", detail, scanner),
                "scanner_hurst": _first_record_value("hurst_value", scanner),
                "scanner_half_life": _first_record_value("half_life_value", scanner),
                "scanner_sigma_0_count": _first_record_value("sigma_0_count", scanner),
                "scanner_sigma_2_count": _first_record_value("sigma_2_count", scanner),
                "scanner_var_99": _first_record_value("var_99", scanner),
                "scanner_cvar_99": _first_record_value("cvar_99", scanner),
                "scanner_max_drawdown": _first_record_value("max_drawdown", scanner),
                "scanner_return_total": _first_record_value("return_total", scanner),
                "scanner_sharpe": _first_record_value("sharpe", scanner),
                "detail_timeframe": _first_record_value("timeframe", detail),
                "detail_periods": _first_record_value("periods_analyzed", detail),
                "detail_strategy": _first_record_value("strategy_label", detail),
                "copula_best_fit": _first_record_value("copula_best_fit", detail),
                "copula_correlation_pct": _probability_percent(_first_record_value("copula_correlation_rho", detail)),
                "copula_x_given_y_pct": _first_record_value("copula_x_given_y_pct", detail),
                "copula_y_given_x_pct": _first_record_value("copula_y_given_x_pct", detail),
                "copula_probability_gap_pct": _first_record_value("copula_probability_gap_pct", detail),
                "copula_signal_status": signal_status,
                "copula_rich_asset": _first_record_value("copula_rich_asset", detail),
                "copula_cheap_asset": _first_record_value("copula_cheap_asset", detail),
                "copula_trade_direction": _first_record_value("copula_trade_direction", detail),
                "capital_weighting": _first_record_value("capital_weighting_slider_value", detail),
                "copula_entry_confirmation": _first_record_value("copula_entry_confirmation", detail),
                "copula_exit_condition": _first_record_value("copula_exit_condition", detail),
                "copula_journal_status": _first_record_value("copula_journal_status", detail),
                "copula_execution_blockers": _first_record_value("copula_execution_blockers", detail),
                "copula_detail_capture_status": detail_status,
                "copula_detail_missing_fields": ";".join(missing_fields),
                "scanner_evidence_path": _first_record_value("scanner_snapshot_path", scanner),
                "detail_evidence_path": _first_record_value("page_route", detail),
            }
        )

    report = pd.DataFrame(report_rows, columns=TWO_HOUR_COPULA_COLUMNS)
    csv_path = root / "reports" / "active" / "wizard_two_hour_copula_report.csv"
    md_path = root / "reports" / "active" / "wizard_two_hour_copula_report.md"
    _write_csv_atomic(report, csv_path)
    lines = [
        "# Two-Hour Copula Report",
        "",
        f"- Generated: {generated_at}",
        f"- Candidate rows: {len(report)}",
        "- A scanner discovery is not a completed Copula check until every required pair-detail field is captured.",
        "- Orange Engle-Granger is recorded as engle_granger_trending, never as a green confirmation.",
        "- Every direction is research/paper-only until execution checks pass.",
        "",
    ]
    if not report.empty:
        preview = report[
            [
                "pair",
                "stationarity_summary",
                "copula_rich_asset",
                "copula_cheap_asset",
                "copula_trade_direction",
                "copula_detail_capture_status",
                "copula_detail_missing_fields",
                "copula_journal_status",
                "copula_execution_blockers",
            ]
        ]
        lines.extend(["", preview.to_string(index=False), ""])
    _write_text_atomic("\n".join(lines), md_path)
    return csv_path, md_path, int(len(report))


def build_wizard_research_journal(root: Path = ROOT) -> CommandResult:
    schema_fields = _schema_fields(root)
    refresh_wizard_pair_page_capture_from_matrices(root)
    decisions = _read_csv_or_empty(root / "reports" / "active" / "paper_trade_decision_report.csv")
    watch_lookup = _watch_lookup(_read_csv_or_empty(root / "reports" / "active" / "current_paper_watch_positions.csv"))
    compatibility_lookup = _compatibility_lookup(
        _read_csv_or_empty(root / "reports" / "active" / "dydx_execution_market_compatibility.csv")
    )
    parity_lookup = _parity_lookup(_read_csv_or_empty(root / "reports" / "active" / "wizard_vs_local_parity_report.csv"))
    shared_outcomes = _shared_outcome_lookup(root)
    paper_journal = _paper_journal_lookup(root)

    scanner = _build_scanner_records(schema_fields, root, decisions, watch_lookup, compatibility_lookup)
    detail = _build_detail_records(
        schema_fields,
        root,
        decisions,
        watch_lookup,
        compatibility_lookup,
        parity_lookup,
        shared_outcomes,
        paper_journal,
    )
    scanner = _attach_wizard_configuration_lineage(scanner, root)
    detail = _attach_wizard_configuration_lineage(detail, root)

    scanner_path = root / "reports" / "active" / "wizard_research_scanner_capture.csv"
    detail_path = root / "reports" / "active" / "wizard_research_pair_detail_capture.csv"
    journal_path = root / "reports" / "active" / "wizard_research_journal.csv"
    md_path = root / "reports" / "active" / "wizard_research_journal.md"
    current_json_path = root / "reports" / "active" / "wizard_research_journal_current.json"
    snapshot_dir = root / "reports" / "active" / "wizard_research_snapshots"

    _write_csv_atomic(scanner, scanner_path)
    _write_csv_atomic(detail, detail_path)
    combined = pd.concat([scanner, detail], ignore_index=True, sort=False)
    _write_csv_atomic(combined, journal_path)
    two_hour_copula_path, two_hour_copula_md_path, two_hour_copula_rows = _build_two_hour_copula_report(combined, root)
    copula_detail_queue_path, copula_detail_queue_rows, copula_detail_pending = _build_copula_detail_capture_queue(combined, root)
    strategy_mode_queue_path, strategy_mode_queue_rows, strategy_mode_pending = _build_strategy_mode_capture_queue(combined, root)

    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_count = 0
    for row in detail.to_dict(orient="records"):
        pair = _clean_text(row.get("pair"))
        timeframe = _clean_text(row.get("timeframe"))
        strategy = _clean_text(row.get("strategy_label"))
        if not pair or not timeframe or not strategy:
            continue
        payload = {field: row.get(field) for field in row.keys() if field != "journal_layer"}
        output = snapshot_dir / f"{_pair_id(pair)}__{_slug(timeframe)}__{_slug(strategy)}.json"
        _write_json_atomic(payload, output)
        snapshot_count += 1

    current_payload = {
        "generated_at_utc": _now_iso(),
        "summary": {
            "scanner_rows": int(len(scanner)),
            "pair_detail_rows": int(len(detail)),
            "missing_timeframe_rows": int(
                detail.get("timeframe_unavailable_flag", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()
            )
            if not detail.empty
            else 0,
            "snapshots": int(snapshot_count),
        },
        "scanner_capture": scanner.to_dict(orient="records"),
        "pair_detail_capture": detail.to_dict(orient="records"),
    }
    _write_json_atomic(current_payload, current_json_path)
    _write_text_atomic(_build_markdown(scanner, detail, current_json_path, snapshot_dir), md_path)

    return CommandResult(
        paths={
            "wizard_research_scanner_capture": scanner_path,
            "wizard_research_pair_detail_capture": detail_path,
            "wizard_research_journal": journal_path,
            "wizard_research_journal_md": md_path,
            "wizard_research_journal_json": current_json_path,
            "wizard_research_snapshots": snapshot_dir,
            "wizard_two_hour_copula_report": two_hour_copula_path,
            "wizard_two_hour_copula_report_md": two_hour_copula_md_path,
            "wizard_copula_detail_capture_queue": copula_detail_queue_path,
            "wizard_strategy_mode_capture_queue": strategy_mode_queue_path,
        },
        summary={
            "scanner_rows": int(len(scanner)),
            "pair_detail_rows": int(len(detail)),
            "missing_timeframes": int(
                detail.get("timeframe_unavailable_flag", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()
            )
            if not detail.empty
            else 0,
            "snapshots": int(snapshot_count),
            "two_hour_copula_rows": two_hour_copula_rows,
            "copula_detail_queue_rows": copula_detail_queue_rows,
            "copula_detail_capture_pending": copula_detail_pending,
            "strategy_mode_queue_rows": strategy_mode_queue_rows,
            "strategy_mode_capture_pending": strategy_mode_pending,
        },
    )


def _attach_wizard_configuration_lineage(frame: pd.DataFrame, root: Path) -> pd.DataFrame:
    output = frame.copy()
    lineage_columns = [
        "config_schema_version",
        "discovery_config_hash",
        "candidate_config_hash",
        "settings_config_hash",
    ]
    for column in lineage_columns:
        if column not in output:
            output[column] = ""
    if output.empty:
        return output

    queue = _read_csv_or_empty(root / "reports" / "active" / "wizard_sweep_settings_capture_queue.csv")
    settings = _read_csv_or_empty(
        root / "reports" / "active" / "crypto_wizards_pair_page_capture_settings.csv"
    )
    lookup: dict[tuple[str, str, str], dict[str, str]] = {}
    for source in (queue, settings):
        for _, row in source.iterrows():
            pair = _normalize_pair(row.get("pair", ""))
            timeframe = _normalize_timeframe(row.get("timeframe", row.get("interval", "")))
            mode = normalize_exact_mode(row.get("exact_mode", ""))
            if not pair or not timeframe or not mode:
                continue
            lookup[(pair, timeframe, mode)] = {
                "config_schema_version": _clean_text(row.get("config_schema_version"))
                or WIZARD_CONFIG_SCHEMA_VERSION,
                "discovery_config_hash": _clean_text(row.get("discovery_config_hash")),
                "candidate_config_hash": _clean_text(row.get("candidate_config_hash")),
                "settings_config_hash": _clean_text(row.get("settings_config_hash")),
            }

    if not lookup:
        return output
    for index, row in output.iterrows():
        pair = _normalize_pair(row.get("pair", ""))
        timeframe = _normalize_timeframe(row.get("timeframe", row.get("interval", "")))
        mode = normalize_exact_mode(row.get("exact_mode", row.get("strategy_label", "")))
        lineage = lookup.get((pair, timeframe, mode))
        if not lineage:
            continue
        for column, value in lineage.items():
            output.at[index, column] = value
    return output
