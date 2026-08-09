from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

import pandas as pd

from quant_platform.active_pipeline import CommandResult
from quant_platform.wizard_candidate_set import wizard_hyperliquid_candidate_set_id
from quant_platform.wizard_evidence import ids_from_exact_mode
from quant_platform.wizard_mode_replay import normalize_exact_mode
from quant_platform.wizard_policy import load_wizard_discovery_policy


ROOT = Path(__file__).resolve().parents[2]

HYPOTHESIS_COLUMNS = [
    "wizard_control_plane_ready",
    "discovery_policy_schema_version",
    "discovery_policy_hash",
    "discovery_config_hash",
    "candidate_config_hash",
    "settings_config_hash",
    "candidate_set_id",
    "pair",
    "asset_x",
    "asset_y",
    "venue",
    "local_interval",
    "local_history_path",
    "local_history_rows",
    "local_history_latest_candle_at",
    "local_history_ready",
    "wizard_pair",
    "wizard_interval",
    "wizard_period",
    "exact_mode",
    "spread_id",
    "strategy_id",
    "vendor_request_strategy",
    "vendor_request_spread_type",
    "wizard_sharpe",
    "wizard_returns_total_pct",
    "wizard_closed_trades",
    "passes_wizard_discovery_gate",
    "min_closed_trades_for_proof",
    "passes_research_spend_gate",
    "wizard_source_timestamp",
    "wizard_source_age_hours",
    "wizard_source_fresh",
    "wizard_capture_health",
    "wizard_settings_present",
    "wizard_settings_missing",
    "entry_level",
    "exit_level",
    "x_weighting",
    "slippage_rate",
    "commission_rate",
    "roll_w",
    "stop_loss_rate",
    "exit_n_periods",
    "mode_fidelity_requirement",
    "proof_observations",
    "vendor_custom_series_eligible",
    "promotion_allowed",
    "hypothesis_status",
    "blocker",
    "next_step",
    "evidence_path",
]

EXACT_MODE_VENDOR_PARAMS: dict[str, dict[str, str]] = {
    "Static (Spread)": {"strategy": "Spread", "spread_type": "Static"},
    "Static (ZScoreR)": {"strategy": "ZScoreRoll", "spread_type": "Static"},
    "Dyn (Spread)": {"strategy": "Spread", "spread_type": "Dynamic"},
    "Dyn (ZScoreR)": {"strategy": "ZScoreRoll", "spread_type": "Dynamic"},
    "OU (Spread)": {"strategy": "Spread", "spread_type": "OU"},
    "OU (ZScoreR)": {"strategy": "ZScoreRoll", "spread_type": "OU"},
    # The documented API leaves spread_type optional for Copula. A capture must still
    # record its effective setting before the call can be considered comparable.
    "Copula": {"strategy": "Copula", "spread_type": ""},
}

_REQUIRED_BACKTEST_SETTINGS = (
    "entry_level",
    "exit_level",
    "x_weighting",
    "slippage_rate",
    "commission_rate",
    "roll_w",
)


def build_hyperliquid_wizard_hypothesis_queue(
    *,
    root: Path = ROOT,
    max_wizard_age_hours: float = 24.0,
    max_local_history_age_hours: float = 24.0,
    min_sharpe: float | None = None,
    min_returns_total_pct: float | None = None,
    min_closed_trades_for_proof: int | None = None,
    as_of: datetime | None = None,
) -> CommandResult:
    """Join fresh Hyperliquid histories to capture-complete Wizard hypotheses.

    This is deliberately a readiness report. It never calls the Wizard API, changes
    a pair score, or authorizes paper/live trading. A row becomes eligible only when
    the exact Wizard configuration can be replayed against the local venue series.
    """

    now = _as_utc(as_of or datetime.now(timezone.utc))
    policy = load_wizard_discovery_policy(root)
    min_sharpe = policy.min_sharpe if min_sharpe is None else float(min_sharpe)
    min_returns_total_pct = (
        policy.min_returns_total_pct if min_returns_total_pct is None else float(min_returns_total_pct)
    )
    min_closed_trades_for_proof = (
        policy.min_closed_trades_for_proof
        if min_closed_trades_for_proof is None
        else int(min_closed_trades_for_proof)
    )
    active = root / "reports" / "active"
    dashboard = root / "reports" / "dashboard"
    bundle = _read_csv(active / "hyperliquid_research_bundle.csv")
    evidence = _wizard_evidence_with_live_dashboard(root)
    control_summary = _read_json(active / "wizard_control_plane_summary.json")
    control_ready = bool(control_summary.get("ready", False))

    rows: list[dict[str, object]] = []
    for _, local_row in bundle.iterrows():
        rows.extend(
            _hypothesis_rows_for_local_history(
                local_row,
                evidence=evidence,
                root=root,
                now=now,
                max_wizard_age_hours=max_wizard_age_hours,
                max_local_history_age_hours=max_local_history_age_hours,
                min_sharpe=min_sharpe,
                min_returns_total_pct=min_returns_total_pct,
                min_closed_trades_for_proof=min_closed_trades_for_proof,
                policy_schema_version=policy.schema_version,
                policy_hash=policy.policy_hash,
            )
        )

    frame = pd.DataFrame(rows, columns=HYPOTHESIS_COLUMNS)
    if not frame.empty:
        frame["candidate_set_id"] = wizard_hyperliquid_candidate_set_id(frame)
        frame["wizard_control_plane_ready"] = control_ready
        if not control_ready:
            frame["vendor_custom_series_eligible"] = False
            frame["promotion_allowed"] = False
            frame["hypothesis_status"] = "BLOCKED"
            frame["blocker"] = frame["blocker"].map(
                lambda value: _append_blocker(value, "wizard_control_plane_not_ready")
            )
            frame["next_step"] = "execute a complete fresh Wizard sweep and rebuild the control plane"
        frame = frame.sort_values(
            ["vendor_custom_series_eligible", "passes_wizard_discovery_gate", "wizard_sharpe", "pair", "local_interval"],
            ascending=[False, False, False, True, True],
            na_position="last",
        ).reset_index(drop=True)

    active.mkdir(parents=True, exist_ok=True)
    dashboard.mkdir(parents=True, exist_ok=True)
    queue_path = active / "hyperliquid_wizard_hypothesis_queue.csv"
    summary_path = active / "hyperliquid_wizard_hypothesis_queue.md"
    dashboard_path = dashboard / "hyperliquid_wizard_hypothesis_dashboard.csv"
    _write_csv(frame, queue_path)
    _write_csv(frame, dashboard_path)
    _write_text(summary_path, _markdown(frame, now=now))

    eligible = int(frame["vendor_custom_series_eligible"].astype(bool).sum()) if not frame.empty else 0
    return CommandResult(
        paths={
            "hypothesis_queue": queue_path,
            "summary_md": summary_path,
            "dashboard": dashboard_path,
        },
        summary={
            "rows": int(len(frame)),
            "vendor_custom_series_eligible": eligible,
            "blocked": int(len(frame) - eligible),
            "as_of": now.isoformat(),
            "discovery_policy_schema_version": policy.schema_version,
            "discovery_policy_hash": policy.policy_hash,
            "min_sharpe": min_sharpe,
            "min_returns_total_pct": min_returns_total_pct,
            "min_closed_trades_for_proof": min_closed_trades_for_proof,
        },
    )


def _hypothesis_rows_for_local_history(
    local_row: pd.Series,
    *,
    evidence: pd.DataFrame,
    root: Path,
    now: datetime,
    max_wizard_age_hours: float,
    max_local_history_age_hours: float,
    min_sharpe: float,
    min_returns_total_pct: float,
    min_closed_trades_for_proof: int,
    policy_schema_version: str,
    policy_hash: str,
) -> list[dict[str, object]]:
    asset_x = _text(local_row.get("asset_x", ""))
    asset_y = _text(local_row.get("asset_y", ""))
    local_interval = _text(local_row.get("interval", ""))
    local_timestamp = _parse_timestamp(local_row.get("history_latest_candle_at", ""))
    local_age_hours = _age_hours(local_timestamp, now)
    local_fresh = local_timestamp is not None and local_age_hours is not None and local_age_hours <= max_local_history_age_hours

    pair_matches = _pair_matches(evidence, asset_x=asset_x, asset_y=asset_y)
    if pair_matches.empty:
        return [
            _blocked_row(
                local_row,
                blocker="wizard_pair_not_in_evidence",
                next_step="capture_current_wizard_pair_page_and_exact_mode",
                local_fresh=local_fresh,
                local_age_hours=local_age_hours,
            )
        ]

    matching_interval = _interval_key(local_interval)
    interval_matches = pair_matches[pair_matches["interval"].map(_interval_key) == matching_interval].copy()
    if interval_matches.empty:
        return [
            _blocked_row(
                local_row,
                blocker="wizard_interval_not_comparable",
                next_step="capture_current_wizard_mode_for_matching_local_interval",
                local_fresh=local_fresh,
                local_age_hours=local_age_hours,
                wizard_pair=_text(pair_matches.iloc[0].get("pair", "")),
            )
        ]

    interval_matches["_returns_total_pct"] = interval_matches.apply(_normalized_return_pct, axis=1)
    interval_matches["_sharpe"] = interval_matches.get("sharpe", pd.Series(index=interval_matches.index)).map(_number)
    interval_matches = interval_matches.sort_values(["_sharpe", "_returns_total_pct"], ascending=[False, False], na_position="last")
    interval_matches = interval_matches.drop_duplicates(subset=["exact_mode"], keep="first")
    return [
        _matched_row(
            local_row,
            wizard_row,
            root=root,
            now=now,
            local_fresh=local_fresh,
            local_age_hours=local_age_hours,
            max_wizard_age_hours=max_wizard_age_hours,
                min_sharpe=min_sharpe,
                min_returns_total_pct=min_returns_total_pct,
                min_closed_trades_for_proof=min_closed_trades_for_proof,
                policy_schema_version=policy_schema_version,
                policy_hash=policy_hash,
        )
        for _, wizard_row in interval_matches.iterrows()
    ]


def _matched_row(
    local_row: pd.Series,
    wizard_row: pd.Series,
    *,
    root: Path,
    now: datetime,
    local_fresh: bool,
    local_age_hours: float | None,
    max_wizard_age_hours: float,
    min_sharpe: float,
    min_returns_total_pct: float,
    min_closed_trades_for_proof: int,
    policy_schema_version: str,
    policy_hash: str,
) -> dict[str, object]:
    exact_mode = normalize_exact_mode(wizard_row.get("exact_mode", ""))
    vendor_params = EXACT_MODE_VENDOR_PARAMS.get(exact_mode, {})
    capture = _capture_metadata(wizard_row, root=root)
    wizard_timestamp = capture["timestamp"]
    wizard_age_hours = _age_hours(wizard_timestamp, now)
    wizard_fresh = wizard_timestamp is not None and wizard_age_hours is not None and wizard_age_hours <= max_wizard_age_hours
    sharpe = _number(wizard_row.get("sharpe", ""))
    returns_total_pct = _normalized_return_pct(wizard_row)
    passes_discovery = sharpe >= min_sharpe and returns_total_pct >= min_returns_total_pct
    closed_trades = _number(wizard_row.get("closed_trades", ""))
    passes_research_spend = bool(
        not pd.isna(closed_trades) and closed_trades >= min_closed_trades_for_proof
    )
    ids = ids_from_exact_mode(exact_mode)
    mode_valid = bool(vendor_params) and ids != (None, None)

    blockers: list[str] = []
    if not _truthy(local_row.get("history_ready", False)):
        blockers.append("local_history_not_ready")
    if not local_fresh:
        blockers.append("local_history_stale_or_timestamp_missing")
    if not mode_valid:
        blockers.append("wizard_exact_mode_invalid_or_unmapped")
    if not passes_discovery:
        blockers.append("wizard_discovery_gate_not_met")
    if not passes_research_spend:
        blockers.append("insufficient_closed_trades_for_paid_proof")
    if not wizard_fresh:
        blockers.append("wizard_source_stale_or_timestamp_missing")
    if capture["health"] != "captured_healthy":
        blockers.append(capture["health"])
    if capture["settings_missing"]:
        blockers.append("wizard_backtest_settings_incomplete")
    if exact_mode == "Copula" and not capture["spread_type_captured"]:
        blockers.append("copula_spread_type_not_captured")

    eligible = not blockers
    evidence_paths = [
        _text(local_row.get("evidence_path", "")),
        _text(wizard_row.get("evidence_path", "")),
        _text(capture["path"]),
    ]
    status = "READY_FOR_VENDOR_MODE_PROOF" if eligible else "BLOCKED"
    return {
        "wizard_control_plane_ready": False,
        "discovery_policy_schema_version": policy_schema_version,
        "discovery_policy_hash": policy_hash,
        "discovery_config_hash": _text(wizard_row.get("discovery_config_hash", wizard_row.get("sweep_config_hash", ""))),
        "candidate_config_hash": _text(wizard_row.get("candidate_config_hash", "")),
        "settings_config_hash": _text(wizard_row.get("settings_config_hash", "")),
        "candidate_set_id": _text(wizard_row.get("candidate_config_hash", "")),
        "pair": _text(local_row.get("pair", "")),
        "asset_x": _text(local_row.get("asset_x", "")),
        "asset_y": _text(local_row.get("asset_y", "")),
        "venue": _text(local_row.get("venue", "hyperliquid")) or "hyperliquid",
        "local_interval": _text(local_row.get("interval", "")),
        "local_history_path": _text(local_row.get("history_path", "")),
        "local_history_rows": _number(local_row.get("history_rows", "")),
        "local_history_latest_candle_at": _text(local_row.get("history_latest_candle_at", "")),
        "local_history_ready": _truthy(local_row.get("history_ready", False)),
        "wizard_pair": _text(wizard_row.get("pair", "")),
        "wizard_interval": _text(wizard_row.get("interval", "")),
        "wizard_period": _number(wizard_row.get("period", "")),
        "exact_mode": exact_mode,
        "spread_id": ids[0] if ids != (None, None) else _text(wizard_row.get("spread_id", "")),
        "strategy_id": ids[1] if ids != (None, None) else _text(wizard_row.get("strategy_id", "")),
        "vendor_request_strategy": vendor_params.get("strategy", ""),
        "vendor_request_spread_type": _text(capture.get("spread_type", "")) or vendor_params.get("spread_type", ""),
        "wizard_sharpe": sharpe,
        "wizard_returns_total_pct": returns_total_pct,
        "wizard_closed_trades": closed_trades,
        "passes_wizard_discovery_gate": passes_discovery,
        "min_closed_trades_for_proof": min_closed_trades_for_proof,
        "passes_research_spend_gate": passes_research_spend,
        "wizard_source_timestamp": wizard_timestamp.isoformat() if wizard_timestamp else "",
        "wizard_source_age_hours": wizard_age_hours if wizard_age_hours is not None else "",
        "wizard_source_fresh": wizard_fresh,
        "wizard_capture_health": capture["health"],
        "wizard_settings_present": ";".join(capture["settings_present"]),
        "wizard_settings_missing": ";".join(capture["settings_missing"]),
        "entry_level": capture["settings"].get("entry_level", ""),
        "exit_level": capture["settings"].get("exit_level", ""),
        "x_weighting": capture["settings"].get("x_weighting", ""),
        "slippage_rate": capture["settings"].get("slippage_rate", ""),
        "commission_rate": capture["settings"].get("commission_rate", ""),
        "roll_w": capture["settings"].get("roll_w", ""),
        "stop_loss_rate": capture["settings"].get("stop_loss_rate", ""),
        "exit_n_periods": capture["settings"].get("exit_n_periods", ""),
        "mode_fidelity_requirement": "vendor_custom_series_backtest_with_captured_settings",
        "proof_observations": _number(wizard_row.get("period", "")),
        "vendor_custom_series_eligible": eligible,
        "promotion_allowed": False,
        "hypothesis_status": status,
        "blocker": ";".join(blockers),
        "next_step": (
            "run bounded Wizard custom-series mode proof, then apply Hyperliquid after-cost replay"
            if eligible
            else _next_step(blockers)
        ),
        "evidence_path": ";".join(path for path in evidence_paths if path),
    }


def _blocked_row(
    local_row: pd.Series,
    *,
    blocker: str,
    next_step: str,
    local_fresh: bool,
    local_age_hours: float | None,
    wizard_pair: str = "",
) -> dict[str, object]:
    return {
        "wizard_control_plane_ready": False,
        "discovery_policy_schema_version": "",
        "discovery_policy_hash": "",
        "discovery_config_hash": "",
        "candidate_config_hash": "",
        "settings_config_hash": "",
        "candidate_set_id": "",
        "pair": _text(local_row.get("pair", "")),
        "asset_x": _text(local_row.get("asset_x", "")),
        "asset_y": _text(local_row.get("asset_y", "")),
        "venue": _text(local_row.get("venue", "hyperliquid")) or "hyperliquid",
        "local_interval": _text(local_row.get("interval", "")),
        "local_history_path": _text(local_row.get("history_path", "")),
        "local_history_rows": _number(local_row.get("history_rows", "")),
        "local_history_latest_candle_at": _text(local_row.get("history_latest_candle_at", "")),
        "local_history_ready": _truthy(local_row.get("history_ready", False)),
        "wizard_pair": wizard_pair,
        "wizard_interval": "",
        "wizard_period": "",
        "exact_mode": "",
        "spread_id": "",
        "strategy_id": "",
        "vendor_request_strategy": "",
        "vendor_request_spread_type": "",
        "wizard_sharpe": "",
        "wizard_returns_total_pct": "",
        "wizard_closed_trades": "",
        "passes_wizard_discovery_gate": False,
        "min_closed_trades_for_proof": "",
        "passes_research_spend_gate": False,
        "wizard_source_timestamp": "",
        "wizard_source_age_hours": "",
        "wizard_source_fresh": False,
        "wizard_capture_health": "missing",
        "wizard_settings_present": "",
        "wizard_settings_missing": ";".join(_REQUIRED_BACKTEST_SETTINGS),
        "entry_level": "",
        "exit_level": "",
        "x_weighting": "",
        "slippage_rate": "",
        "commission_rate": "",
        "roll_w": "",
        "stop_loss_rate": "",
        "exit_n_periods": "",
        "mode_fidelity_requirement": "vendor_custom_series_backtest_with_captured_settings",
        "proof_observations": "",
        "vendor_custom_series_eligible": False,
        "promotion_allowed": False,
        "hypothesis_status": "BLOCKED",
        "blocker": blocker + (";local_history_stale_or_timestamp_missing" if not local_fresh else ""),
        "next_step": next_step,
        "evidence_path": _text(local_row.get("evidence_path", "")),
    }


def _pair_matches(evidence: pd.DataFrame, *, asset_x: str, asset_y: str) -> pd.DataFrame:
    if evidence.empty or not {"asset_x", "asset_y"}.issubset(evidence.columns):
        return pd.DataFrame(columns=evidence.columns)
    left = _asset_key(asset_x)
    right = _asset_key(asset_y)
    return evidence[
        evidence["asset_x"].map(_asset_key).eq(left) & evidence["asset_y"].map(_asset_key).eq(right)
    ].copy()


def _capture_metadata(wizard_row: pd.Series, *, root: Path) -> dict[str, object]:
    sources = [_text(wizard_row.get("source_path", "")), _text(wizard_row.get("evidence_path", ""))]
    for source in sources:
        path = _resolve_path(source, root=root)
        if path is None or not path.exists():
            continue
        record = _capture_record(path, wizard_row)
        timestamp = _parse_timestamp(
            record.get("capture_timestamp_utc", "")
            or record.get("source_timestamp", "")
            or record.get("updated_at_utc", "")
        )
        if timestamp is None:
            timestamp = _timestamp_from_path(path)
        raw_text = _text(record.get("raw_text", "")).lower()
        health = "captured_healthy"
        if "data retrieval failed" in raw_text or "no available metrics" in raw_text:
            health = "wizard_capture_data_retrieval_failed"
        present, missing, settings = _backtest_setting_completeness(record)
        spread_type = _text(record.get("spread_type", ""))
        return {
            "path": _relative(path, root=root),
            "timestamp": timestamp,
            "health": health,
            "settings_present": present,
            "settings_missing": missing,
            "settings": settings,
            "spread_type": spread_type,
            "spread_type_captured": bool(spread_type),
        }
    return {
        "path": _text(wizard_row.get("source_path", "")) or _text(wizard_row.get("evidence_path", "")),
        "timestamp": None,
        "health": "wizard_source_capture_missing",
        "settings_present": [],
        "settings_missing": list(_REQUIRED_BACKTEST_SETTINGS),
        "settings": {},
        "spread_type": "",
        "spread_type_captured": False,
    }


def _capture_record(path: Path, wizard_row: pd.Series) -> dict[str, object]:
    try:
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            records = payload.get("rows", []) if isinstance(payload, dict) else []
            if isinstance(records, list):
                selected = _select_capture_record(records, wizard_row)
                if selected is not None:
                    return selected
            return payload if isinstance(payload, dict) else {}
        if path.suffix.lower() == ".csv":
            frame = pd.read_csv(path)
            selected = _select_capture_record(frame.to_dict(orient="records"), wizard_row)
            if selected is not None:
                return selected
    except (OSError, ValueError, pd.errors.ParserError, json.JSONDecodeError):
        return {}
    return {}


def _select_capture_record(records: list[dict[str, Any]], wizard_row: pd.Series) -> dict[str, object] | None:
    want_x = _asset_key(wizard_row.get("asset_x", ""))
    want_y = _asset_key(wizard_row.get("asset_y", ""))
    want_interval = _interval_key(wizard_row.get("interval", ""))
    want_mode = normalize_exact_mode(wizard_row.get("exact_mode", ""))
    matches = [
        record
        for record in records
        if _asset_key(record.get("asset_x", "")) == want_x
        and _asset_key(record.get("asset_y", "")) == want_y
        and _interval_key(record.get("interval", record.get("timeframe", ""))) == want_interval
        and normalize_exact_mode(record.get("exact_mode", record.get("strategy_mode", ""))) == want_mode
    ]
    if not matches:
        return None
    return dict(matches[-1])


def _backtest_setting_completeness(record: dict[str, object]) -> tuple[list[str], list[str], dict[str, object]]:
    flattened: dict[str, object] = {str(key): value for key, value in record.items()}
    nested = record.get("backtest_settings")
    if isinstance(nested, dict):
        flattened.update({str(key): value for key, value in nested.items()})
    aliases = {
        "entry_level": ("entry_level", "entry_long_value", "entry_short_value", "entry_lower", "copula_entry_lower"),
        "exit_level": ("exit_level", "exit_long_value", "exit_short_value", "exit_lower", "copula_exit_lower"),
        "x_weighting": ("x_weighting", "capital_weighting"),
        "slippage_rate": ("slippage_rate", "slippage"),
        "commission_rate": ("commission_rate", "commission"),
        "roll_w": ("roll_w", "zscore_window", "rolling_window"),
        "stop_loss_rate": ("stop_loss_rate", "stop_loss_rate_opt", "stop_loss"),
        "exit_n_periods": ("exit_n_periods", "close_n_periods"),
    }
    settings = {
        setting: next((_number_or_text(flattened[name]) for name in names if _text(flattened.get(name, ""))), "")
        for setting, names in aliases.items()
    }
    present = [setting for setting, value in settings.items() if value != ""]
    missing = [setting for setting in _REQUIRED_BACKTEST_SETTINGS if setting not in present]
    return present, missing, settings


def _normalized_return_pct(row: pd.Series) -> float:
    value = _number(row.get("returns_total", ""))
    if pd.isna(value):
        return float("nan")
    magnitude = abs(value)
    if magnitude <= 1.0:
        return value * 100.0
    if magnitude <= 100.0:
        return value
    return value / 100.0


def _next_step(blockers: list[str]) -> str:
    if any("source_stale" in blocker or "capture_data" in blocker for blocker in blockers):
        return "refresh_current_wizard_pair_capture_with_full_backtest_settings"
    if any("settings" in blocker or "spread_type" in blocker for blocker in blockers):
        return "capture_selected_mode_and_all_backtest_settings"
    if any("local_history" in blocker for blocker in blockers):
        return "refresh_local_hyperliquid_history"
    if any("discovery_gate" in blocker for blocker in blockers):
        return "keep_as_discovery_only_and_rescan_wizard"
    if "insufficient_closed_trades_for_paid_proof" in blockers:
        return "keep_on_raw_leaderboard_and_wait_for_more_independent_closed_trades"
    return "inspect_evidence_and_resolve_blockers"


def _asset_key(value: object) -> str:
    text = _text(value).upper().replace("/", "-")
    for suffix in ("-USDT", "-USD", "-PERP"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    if "-" not in text:
        for suffix in ("USDT", "USDC", "PERP", "USD"):
            if text.endswith(suffix) and len(text) > len(suffix):
                text = text[: -len(suffix)]
                break
    return text


def _interval_key(value: object) -> str:
    text = _text(value).lower().replace(" ", "")
    aliases = {
        "1d": "daily",
        "day": "daily",
        "daily": "daily",
        "1day": "daily",
        "5m": "5min",
        "5min": "5min",
        "5mins": "5min",
        "hourly": "hourly",
        "1hour": "hourly",
        "1h": "hourly",
        "4h": "4hour",
        "4hour": "4hour",
    }
    return aliases.get(text, text)


def _resolve_path(value: str, *, root: Path) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


def _timestamp_from_path(path: Path) -> datetime | None:
    match = re.search(r"(20\d{2}-\d{2}-\d{2})", path.name)
    return _parse_timestamp(match.group(1)) if match else None


def _parse_timestamp(value: object) -> datetime | None:
    text = _text(value)
    if not text:
        return None
    parsed = pd.to_datetime(text, utc=True, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.to_pydatetime()


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _age_hours(timestamp: datetime | None, now: datetime) -> float | None:
    if timestamp is None:
        return None
    return round(max(0.0, (now - timestamp).total_seconds() / 3600.0), 3)


def _number(value: object) -> float:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(parsed) if not pd.isna(parsed) else float("nan")


def _number_or_text(value: object) -> object:
    parsed = _number(value)
    return parsed if not pd.isna(parsed) else _text(value)


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return _text(value).lower() in {"1", "true", "yes", "y"}


def _text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _relative(path: Path, *, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _append_blocker(value: object, blocker: str) -> str:
    blockers = [item for item in str(value).split(";") if item]
    if blocker not in blockers:
        blockers.append(blocker)
    return ";".join(blockers)


def _wizard_evidence_with_live_dashboard(root: Path) -> pd.DataFrame:
    evidence = _read_csv(root / "data" / "processed" / "wizard_evidence.csv")
    validated_settings = _validated_settings_evidence(root)
    if not validated_settings.empty:
        evidence = pd.concat([evidence, validated_settings], ignore_index=True, sort=False)
    api_path = root / "reports" / "active" / "crypto_wizards_live_scanner_capture.csv"
    api = _read_csv(api_path)
    if not api.empty:
        api = api.copy()
        if "exact_mode" in api:
            api["exact_mode"] = api["exact_mode"].map(normalize_exact_mode)
        api["source_path"] = _relative(api_path, root=root)
        if "evidence_path" not in api:
            api["evidence_path"] = api["source_path"]
        evidence = pd.concat([evidence, api], ignore_index=True, sort=False)

    live_path = root / "reports" / "active" / "wizard_dydx_daily_live_workflow.csv"
    live = _read_csv(live_path)
    required = {"pair", "asset_x", "asset_y", "timeframe", "exact_mode", "returns_total_pct", "sharpe"}
    if live.empty or not required.issubset(live.columns):
        return _prefer_fresh_wizard_rows(evidence)
    normalized = pd.DataFrame(
        {
            "pair": live["pair"],
            "asset_x": live["asset_x"],
            "asset_y": live["asset_y"],
            "exchange": live.get("wizard_exchange", "DYDX"),
            "interval": live["timeframe"],
            "period": "",
            "exact_mode": live["exact_mode"].map(normalize_exact_mode),
            "sharpe": live["sharpe"],
            "returns_total": live["returns_total_pct"],
            "returns_total_pct": live["returns_total_pct"],
            "closed_trades": "",
            "source_timestamp": live.get("updated_at_utc", ""),
            "source_path": _relative(live_path, root=root),
            "evidence_path": _relative(live_path, root=root),
        }
    )
    return _prefer_fresh_wizard_rows(pd.concat([evidence, normalized], ignore_index=True, sort=False))


def _validated_settings_evidence(root: Path) -> pd.DataFrame:
    active = root / "reports" / "active"
    settings_path = active / "crypto_wizards_pair_page_capture_settings.csv"
    settings = _read_csv(settings_path)
    queue = _read_csv(active / "wizard_sweep_settings_capture_queue.csv")
    if settings.empty or queue.empty or "candidate_config_hash" not in settings.columns:
        return pd.DataFrame()
    queue_index = {
        _text(row.get("candidate_config_hash", "")): row.to_dict()
        for _, row in queue.iterrows()
        if _text(row.get("candidate_config_hash", ""))
    }
    rows: list[dict[str, object]] = []
    for _, capture_row in settings.iterrows():
        capture = capture_row.to_dict()
        candidate_hash = _text(capture.get("candidate_config_hash", ""))
        discovery = queue_index.get(candidate_hash)
        if discovery is None or not _truthy(capture.get("capture_confirmed", False)):
            continue
        if "backtest_settings_complete" in capture and not _truthy(capture.get("backtest_settings_complete", False)):
            continue
        record = dict(discovery)
        record.update(capture)
        record["interval"] = _text(capture.get("interval", "")) or _text(discovery.get("timeframe", ""))
        record["sharpe"] = discovery.get("sharpe", "")
        record["returns_total"] = discovery.get("returns_total", "")
        record["closed_trades"] = discovery.get("closed_trades", "")
        record["source_timestamp"] = discovery.get("source_timestamp", "")
        record["source_path"] = _relative(settings_path, root=root)
        record["evidence_path"] = _text(capture.get("capture_evidence_path", "")) or record["source_path"]
        record["source_authority"] = "visible_dashboard_settings_capture"
        rows.append(record)
    return pd.DataFrame(rows)


def _prefer_fresh_wizard_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Prefer the newest evidence for the same pair, interval, and exact mode."""

    if frame.empty:
        return frame
    ranked = frame.copy()
    ranked["exact_mode"] = ranked.get("exact_mode", pd.Series("", index=ranked.index)).map(normalize_exact_mode)
    timestamps = pd.Series(pd.NaT, index=ranked.index, dtype="datetime64[ns, UTC]")
    for column in ("capture_timestamp_utc", "source_timestamp", "updated_at_utc", "backtest_timestamp_utc"):
        if column in ranked:
            candidate = pd.to_datetime(ranked[column], utc=True, errors="coerce")
            timestamps = timestamps.fillna(candidate)
    ranked["_evidence_timestamp"] = timestamps
    ranked["_evidence_priority"] = ranked.get(
        "source_path", pd.Series("", index=ranked.index)
    ).astype(str).map(_wizard_evidence_priority)
    ranked = ranked.sort_values(
        ["_evidence_timestamp", "_evidence_priority"],
        ascending=[False, False],
        na_position="last",
    )
    identity = [column for column in ("asset_x", "asset_y", "interval", "exact_mode") if column in ranked]
    if identity:
        ranked = ranked.drop_duplicates(subset=identity, keep="first")
    return ranked.drop(columns=["_evidence_timestamp", "_evidence_priority"]).reset_index(drop=True)


def _wizard_evidence_priority(source_path: str) -> int:
    if source_path.endswith("crypto_wizards_pair_page_capture_settings.csv"):
        return 3
    if source_path.endswith("crypto_wizards_live_scanner_capture.csv"):
        return 2
    return 1


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _markdown(frame: pd.DataFrame, *, now: datetime) -> str:
    eligible = int(frame["vendor_custom_series_eligible"].astype(bool).sum()) if not frame.empty else 0
    lines = [
        "# Hyperliquid to Crypto Wizards Hypothesis Queue",
        "",
        f"- As of: `{now.isoformat()}`",
        f"- Rows: `{len(frame)}`",
        f"- Eligible for a bounded vendor custom-series proof: `{eligible}`",
        "- Promotion authority: `none`; this queue is research intake only.",
        "- Exact-mode rule: a generic local z-score cannot stand in for a Wizard mode.",
        "",
    ]
    if frame.empty:
        return "\n".join(lines + ["No fresh Hyperliquid histories were available.", ""])
    preview = frame[
        [
            "pair",
            "local_interval",
            "exact_mode",
            "wizard_sharpe",
            "wizard_returns_total_pct",
            "hypothesis_status",
            "blocker",
            "next_step",
        ]
    ]
    return "\n".join(lines + ["## Queue", "", preview.to_markdown(index=False), ""])
