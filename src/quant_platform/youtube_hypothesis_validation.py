"""Fail-closed validation intake for Wizard-discovered Hyperliquid hypotheses."""

from __future__ import annotations

from quant_platform.orchestration.corrective_runtime import atomic_write_text

from quant_platform.orchestration.corrective_runtime import atomic_write_csv

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import pandas as pd

from quant_platform.active_pipeline import CommandResult, ROOT
from quant_platform.wizard_evidence import WIZARD_PAIR_SETTINGS_CAPTURE_TEMPLATE_COLUMNS, ids_from_exact_mode
from quant_platform.wizard_mode_replay import build_local_mode_signal, mode_requirements, normalize_exact_mode


REGIMES = ("all", "range_only", "exclude_crisis", "calm_vol_only", "range_low_tail", "stable_hedge")
REGIME_BEHAVIORS = ("entry_only", "hard_exit")
MIN_FORWARD_BARS = 30

VALIDATION_COLUMNS = [
    "hypothesis_id",
    "pair",
    "asset_x",
    "asset_y",
    "timeframe",
    "exact_mode",
    "wizard_exchange",
    "wizard_pair_id",
    "wizard_sharpe",
    "wizard_returns_total_pct",
    "wizard_discovery_pass",
    "wizard_capture_timestamp",
    "wizard_capture_fresh",
    "observed_entry_level",
    "observed_exit_level",
    "observed_x_weighting",
    "observed_y_weighting",
    "observed_hedge_ratio",
    "observed_zscore_window",
    "observed_ou_mu",
    "observed_ou_sigma",
    "settings_capture_confirmed",
    "exact_mode_required_settings",
    "exact_mode_missing_inputs",
    "local_mode_replay_ready",
    "mode_fidelity_status",
    "history_path",
    "history_rows",
    "history_latest_candle_at",
    "history_ready",
    "funding_ready",
    "funding_coverage_pct",
    "cost_model_ready",
    "slippage_model_ready",
    "slippage_samples_x",
    "slippage_samples_y",
    "required_slippage_samples",
    "estimated_pair_round_trip_cost_bps",
    "hyperliquid_mainnet_asset_x_tradable",
    "hyperliquid_mainnet_asset_y_tradable",
    "hyperliquid_mainnet_pair_tradable",
    "hyperliquid_testnet_asset_x_tradable",
    "hyperliquid_testnet_asset_y_tradable",
    "hyperliquid_testnet_pair_tradable",
    "execution_lane",
    "settings_capture_forward_bars",
    "point_in_time_settings_ready",
    "costed_walk_forward_ready",
    "regime_matrix_ready",
    "acceptance_eligible",
    "trade_authorized",
    "validation_status",
    "acceptance_reason",
    "blocker",
    "next_step",
    "evidence_path",
    "updated_at_utc",
]

REGIME_COLUMNS = [
    "pair",
    "asset_x",
    "asset_y",
    "timeframe",
    "exact_mode",
    "regime",
    "regime_behavior",
    "trades",
    "profit_factor",
    "sharpe",
    "max_drawdown",
    "total_return",
    "acceptance_eligible",
    "validation_status",
    "acceptance_reason",
    "blocker",
    "evidence_path",
]


def run_youtube_hypothesis_validation(
    *,
    root: Path = ROOT,
    candidate_path: Path | None = None,
    min_history_rows: int = 120,
    as_of: datetime | None = None,
) -> CommandResult:
    """Build exact-mode and venue-readiness reports without proxy substitution.

    The command never spends Wizard credits and never authorizes a trade. It
    runs no performance calculation until exact settings, point-in-time
    forward bars, venue costs, funding, and slippage calibration are complete.
    """

    now = _utc(as_of or datetime.now(timezone.utc))
    candidates_path = candidate_path or root / "reports" / "agents" / "youtube_brain_hypotheses.csv"
    candidates = _read_csv(candidates_path)
    if not candidates.empty:
        discovery = candidates.get("wizard_discovery_pass", pd.Series(False, index=candidates.index)).map(_boolish)
        candidates = candidates.loc[discovery].copy()
        candidates["exact_mode"] = candidates.get("exact_mode", pd.Series("", index=candidates.index)).map(normalize_exact_mode)
        candidates = candidates.loc[candidates["exact_mode"].ne("")]
        candidates = candidates.drop_duplicates(subset=["pair", "timeframe", "exact_mode"], keep="first")

    scanner_path = root / "reports" / "active" / "crypto_wizards_live_scanner_capture.csv"
    scanner = _read_csv(scanner_path)
    captures_path = root / "reports" / "active" / "crypto_wizards_pair_page_capture_settings.csv"
    captures = _read_csv(captures_path)
    funding_path = root / "reports" / "active" / "hyperliquid_funding_coverage.csv"
    funding = _read_csv(funding_path)
    costs_path = root / "reports" / "active" / "hyperliquid_pair_cost_model.csv"
    costs = _read_csv(costs_path)
    mainnet_path = root / "data" / "processed" / "hyperliquid_market_context.csv"
    if not mainnet_path.exists():
        mainnet_path = root / "reports" / "active" / "hyperliquid_market_context.csv"
    mainnet = _read_csv(mainnet_path)
    testnet_path = root / "reports" / "active" / "hyperliquid_testnet_market_inventory.csv"
    testnet = _read_csv(testnet_path)

    rows: list[dict[str, object]] = []
    capture_rows: list[dict[str, object]] = []
    for _, candidate in candidates.iterrows():
        row, capture_row = _validation_row(
            candidate,
            root=root,
            now=now,
            min_history_rows=min_history_rows,
            scanner=scanner,
            captures=captures,
            funding=funding,
            costs=costs,
            mainnet=mainnet,
            testnet=testnet,
            candidates_path=candidates_path,
            scanner_path=scanner_path,
            captures_path=captures_path,
            funding_path=funding_path,
            costs_path=costs_path,
            mainnet_path=mainnet_path,
            testnet_path=testnet_path,
        )
        rows.append(row)
        capture_rows.append(capture_row)

    frame = pd.DataFrame(rows, columns=VALIDATION_COLUMNS)
    if not frame.empty:
        frame = frame.sort_values(
            ["costed_walk_forward_ready", "wizard_sharpe", "pair", "exact_mode"],
            ascending=[False, False, True, True],
            na_position="last",
        ).reset_index(drop=True)
    regime = _regime_readiness_matrix(frame)
    capture_queue = pd.DataFrame(capture_rows, columns=WIZARD_PAIR_SETTINGS_CAPTURE_TEMPLATE_COLUMNS)

    active = root / "reports" / "active"
    dashboard = root / "reports" / "dashboard"
    templates = root / "reports" / "templates"
    validation_path = active / "youtube_hypothesis_validation_queue.csv"
    regime_path = active / "youtube_hypothesis_regime_matrix.csv"
    capture_path = templates / "youtube_hypothesis_wizard_settings_capture.csv"
    summary_path = active / "youtube_hypothesis_validation.md"
    dashboard_path = dashboard / "youtube_hypothesis_validation.csv"
    _write_csv(frame, validation_path)
    _write_csv(regime, regime_path)
    _write_csv(capture_queue, capture_path)
    _write_csv(frame, dashboard_path)
    _write_text(summary_path, _markdown(frame, regime, now=now))

    return CommandResult(
        paths={
            "validation_queue": validation_path,
            "regime_matrix": regime_path,
            "settings_capture_queue": capture_path,
            "summary_md": summary_path,
            "dashboard": dashboard_path,
        },
        summary={
            "candidates": int(len(frame)),
            "local_mode_replay_ready": int(frame.get("local_mode_replay_ready", pd.Series(dtype=bool)).map(_boolish).sum()),
            "costed_walk_forward_ready": int(frame.get("costed_walk_forward_ready", pd.Series(dtype=bool)).map(_boolish).sum()),
            "testnet_pair_tradable": int(frame.get("hyperliquid_testnet_pair_tradable", pd.Series(dtype=bool)).map(_boolish).sum()),
            "blocked": int((~frame.get("costed_walk_forward_ready", pd.Series(False, index=frame.index)).map(_boolish)).sum()),
            "as_of": now.isoformat(),
        },
    )


def _validation_row(
    candidate: pd.Series,
    *,
    root: Path,
    now: datetime,
    min_history_rows: int,
    scanner: pd.DataFrame,
    captures: pd.DataFrame,
    funding: pd.DataFrame,
    costs: pd.DataFrame,
    mainnet: pd.DataFrame,
    testnet: pd.DataFrame,
    candidates_path: Path,
    scanner_path: Path,
    captures_path: Path,
    funding_path: Path,
    costs_path: Path,
    mainnet_path: Path,
    testnet_path: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    pair = _text(candidate.get("pair"))
    asset_x = _asset(candidate.get("asset_x"))
    asset_y = _asset(candidate.get("asset_y"))
    timeframe = _text(candidate.get("timeframe"))
    exact_mode = normalize_exact_mode(candidate.get("exact_mode"))
    scanner_row = _matching_row(scanner, pair=pair, timeframe=timeframe, exact_mode=exact_mode)
    raw_row = _raw_scanner_row(scanner_row, exact_mode=exact_mode, root=root)
    capture = _matching_row(captures, pair=pair, timeframe=timeframe, exact_mode=exact_mode)
    capture_confirmed = _boolish(capture.get("capture_confirmed", False))

    settings = _observed_settings(scanner_row, raw_row, capture)
    history_path = _matching_history_path(root, asset_x=asset_x, asset_y=asset_y, timeframe=timeframe)
    history = _history_frame(history_path) if history_path else pd.DataFrame()
    history_rows = len(history)
    history_ready = history_rows >= min_history_rows
    latest_candle = history.index.max().isoformat() if history_ready else ""
    mode_result = build_local_mode_signal(history, settings, exact_mode=exact_mode)
    local_mode_ready = history_ready and mode_result.mode_replay_status == "READY_FOR_RESEARCH_REPLAY"

    funding_row = _matching_pair_row(funding, pair)
    cost_row = _matching_pair_row(costs, pair)
    funding_ready = _boolish(funding_row.get("funding_ready", False))
    cost_ready = _boolish(cost_row.get("cost_model_ready", False))
    slippage_ready = _boolish(cost_row.get("slippage_model_ready", False))
    mainnet_x = _asset_tradable(mainnet, asset_x, "tradable")
    mainnet_y = _asset_tradable(mainnet, asset_y, "tradable")
    testnet_x = _asset_tradable(testnet, asset_x, "tradable_perp")
    testnet_y = _asset_tradable(testnet, asset_y, "tradable_perp")
    mainnet_pair = mainnet_x and mainnet_y
    testnet_pair = testnet_x and testnet_y

    capture_timestamp = _timestamp(settings.get("capture_timestamp_utc"))
    forward_bars = int((history.index > pd.Timestamp(capture_timestamp)).sum()) if capture_timestamp and not history.empty else 0
    point_in_time_ready = capture_confirmed and forward_bars >= MIN_FORWARD_BARS
    walk_ready = bool(
        local_mode_ready
        and funding_ready
        and cost_ready
        and slippage_ready
        and mainnet_pair
        and point_in_time_ready
    )

    blockers: list[str] = []
    if not history_ready:
        blockers.append("hyperliquid_history_missing_or_short")
    if not mainnet_pair:
        blockers.append("hyperliquid_mainnet_pair_not_tradable")
    if not capture_confirmed:
        blockers.append("wizard_exact_settings_not_live_confirmed")
    blockers.extend(f"missing_exact_input:{value}" for value in mode_result.missing_inputs)
    if not funding_ready:
        blockers.append("hyperliquid_funding_not_ready")
    if not cost_ready:
        blockers.append("hyperliquid_cost_model_not_ready")
    if not slippage_ready:
        blockers.append("hyperliquid_l2_slippage_not_calibrated")
    if capture_confirmed and not point_in_time_ready:
        blockers.append(f"point_in_time_forward_bars_below_{MIN_FORWARD_BARS}")
    blockers = list(dict.fromkeys(blockers))

    if walk_ready:
        status = "READY_FOR_COSTED_WALK_FORWARD"
        reason = "all exact-mode, point-in-time, funding, cost, slippage, and mainnet venue prerequisites passed"
        next_step = "run_exact_mode_walk_forward_then_regime_matrix"
    elif not history_ready:
        status = "BLOCKED_HISTORY"
        reason = "matching Hyperliquid history is missing or shorter than the required validation window"
        next_step = "refresh_matching_hyperliquid_pair_history"
    elif not local_mode_ready:
        status = "BLOCKED_MISSING_EXACT_SETTINGS"
        reason = "generic z-score substitution is prohibited; capture the selected Wizard mode settings"
        next_step = "complete_and_import_youtube_hypothesis_wizard_settings_capture"
    elif not point_in_time_ready:
        status = "BLOCKED_POINT_IN_TIME_EVIDENCE"
        reason = "a current settings snapshot cannot validate the preceding historical period without hindsight"
        next_step = f"collect_at_least_{MIN_FORWARD_BARS}_daily_forward_bars_or_verify_foldwise_parameter_formula"
    else:
        status = "BLOCKED_VENUE_ECONOMICS"
        reason = "exact mode is available but Hyperliquid cost, funding, or slippage evidence is incomplete"
        next_step = "continue_hyperliquid_l2_and_funding_evidence_cadence"

    execution_lane = "testnet_candidate" if testnet_pair else ("mainnet_research_only" if mainnet_pair else "unavailable")
    evidence_paths = [
        candidates_path,
        scanner_path if not scanner_row.empty else None,
        Path(_text(scanner_row.get("raw_source_path"))) if _text(scanner_row.get("raw_source_path")) else None,
        captures_path if not capture.empty else None,
        history_path,
        funding_path,
        costs_path,
        mainnet_path,
        testnet_path,
    ]
    evidence = ";".join(_relative(path, root) for path in evidence_paths if path is not None)

    row = {
        "hypothesis_id": _text(candidate.get("hypothesis_id")),
        "pair": pair,
        "asset_x": asset_x,
        "asset_y": asset_y,
        "timeframe": timeframe,
        "exact_mode": exact_mode,
        "wizard_exchange": _text(candidate.get("wizard_exchange")),
        "wizard_pair_id": _value(scanner_row, raw_row, "pair_id"),
        "wizard_sharpe": _number(candidate.get("wizard_sharpe")),
        "wizard_returns_total_pct": _number(candidate.get("wizard_returns_total_pct")),
        "wizard_discovery_pass": _boolish(candidate.get("wizard_discovery_pass")),
        "wizard_capture_timestamp": _text(scanner_row.get("capture_timestamp_utc")),
        "wizard_capture_fresh": _fresh(scanner_row.get("capture_timestamp_utc"), now=now, hours=24),
        "observed_entry_level": settings.get("observed_entry_level", ""),
        "observed_exit_level": settings.get("observed_exit_level", ""),
        "observed_x_weighting": settings.get("x_weighting", ""),
        "observed_y_weighting": settings.get("y_weighting", ""),
        "observed_hedge_ratio": settings.get("hedge_ratio", ""),
        "observed_zscore_window": settings.get("zscore_window", ""),
        "observed_ou_mu": settings.get("ou_mu", ""),
        "observed_ou_sigma": settings.get("ou_sigma", ""),
        "settings_capture_confirmed": capture_confirmed,
        "exact_mode_required_settings": ";".join(mode_requirements(exact_mode)),
        "exact_mode_missing_inputs": ";".join(mode_result.missing_inputs),
        "local_mode_replay_ready": local_mode_ready,
        "mode_fidelity_status": mode_result.mode_fidelity_status,
        "history_path": _relative(history_path, root) if history_path else "",
        "history_rows": history_rows,
        "history_latest_candle_at": latest_candle,
        "history_ready": history_ready,
        "funding_ready": funding_ready,
        "funding_coverage_pct": _number(funding_row.get("funding_coverage_pct")),
        "cost_model_ready": cost_ready,
        "slippage_model_ready": slippage_ready,
        "slippage_samples_x": _number(cost_row.get("slippage_samples_x")),
        "slippage_samples_y": _number(cost_row.get("slippage_samples_y")),
        "required_slippage_samples": _number(cost_row.get("required_slippage_samples")),
        "estimated_pair_round_trip_cost_bps": _number(cost_row.get("estimated_pair_round_trip_cost_bps")),
        "hyperliquid_mainnet_asset_x_tradable": mainnet_x,
        "hyperliquid_mainnet_asset_y_tradable": mainnet_y,
        "hyperliquid_mainnet_pair_tradable": mainnet_pair,
        "hyperliquid_testnet_asset_x_tradable": testnet_x,
        "hyperliquid_testnet_asset_y_tradable": testnet_y,
        "hyperliquid_testnet_pair_tradable": testnet_pair,
        "execution_lane": execution_lane,
        "settings_capture_forward_bars": forward_bars,
        "point_in_time_settings_ready": point_in_time_ready,
        "costed_walk_forward_ready": walk_ready,
        "regime_matrix_ready": walk_ready,
        "acceptance_eligible": False,
        "trade_authorized": False,
        "validation_status": status,
        "acceptance_reason": reason,
        "blocker": ";".join(blockers),
        "next_step": next_step,
        "evidence_path": evidence,
        "updated_at_utc": now.isoformat(),
    }
    return row, _capture_queue_row(candidate, scanner_row, raw_row, capture, settings, exact_mode=exact_mode)


def _capture_queue_row(
    candidate: pd.Series,
    scanner_row: pd.Series,
    raw_row: dict[str, object],
    capture: pd.Series,
    settings: dict[str, object],
    *,
    exact_mode: str,
) -> dict[str, object]:
    row = {column: "" for column in WIZARD_PAIR_SETTINGS_CAPTURE_TEMPLATE_COLUMNS}
    pair = _text(candidate.get("pair"))
    asset_x = _text(candidate.get("asset_x"))
    asset_y = _text(candidate.get("asset_y"))
    spread_id, strategy_id = ids_from_exact_mode(exact_mode)
    row.update(
        {
            "setup_identity": f"{pair.replace('/', '|')}|{_text(candidate.get('timeframe')).lower()}|{_value(scanner_row, raw_row, 'period')}|{exact_mode.lower().replace(' ', '_')}",
            "pair": pair,
            "asset_x": asset_x,
            "asset_y": asset_y,
            "exchange": _text(candidate.get("wizard_exchange")),
            "interval": _text(candidate.get("timeframe")).lower(),
            "period": _value(scanner_row, raw_row, "period"),
            "exact_mode": exact_mode,
            "spread_id": spread_id or "",
            "strategy_id": strategy_id or "",
            "required_settings": ";".join(mode_requirements(exact_mode)),
            "capture_timestamp_utc": _text(capture.get("capture_timestamp_utc")),
            "capture_evidence_path": _text(capture.get("capture_evidence_path")),
            "capture_confirmed": _boolish(capture.get("capture_confirmed", False)),
            "pair_page_url": _text(capture.get("pair_page_url")) or f"https://cryptowizards.net/wizards/zscore/pair/{_value(scanner_row, raw_row, 'pair_id')}?origin=scanner",
        }
    )
    for field in WIZARD_PAIR_SETTINGS_CAPTURE_TEMPLATE_COLUMNS:
        if field in capture and _text(capture.get(field)):
            row[field] = capture.get(field)
        elif field in settings and _text(settings.get(field)):
            row[field] = settings.get(field)
    return row


def _observed_settings(scanner: pd.Series, raw: dict[str, object], capture: pd.Series) -> dict[str, object]:
    settings: dict[str, object] = {
        "capture_confirmed": False,
        "observed_entry_level": _value(scanner, raw, "entry_level"),
        "observed_exit_level": _value(scanner, raw, "exit_level"),
        "x_weighting": _value(scanner, raw, "x_weighting"),
        "y_weighting": _value(scanner, raw, "y_weighting"),
        "hedge_ratio": _value(scanner, raw, "hedge_ratio"),
        "zscore_window": _value(scanner, raw, "zscore_window") or _value(scanner, raw, "roll_w"),
        "ou_mu": _value(scanner, raw, "ou_mu"),
        "ou_sigma": _value(scanner, raw, "ou_sigma"),
    }
    if not capture.empty:
        for field in WIZARD_PAIR_SETTINGS_CAPTURE_TEMPLATE_COLUMNS:
            if _text(capture.get(field)):
                settings[field] = capture.get(field)
        settings["capture_confirmed"] = _boolish(capture.get("capture_confirmed", False))
        settings["capture_timestamp_utc"] = capture.get("capture_timestamp_utc", "")
    return settings


def _raw_scanner_row(scanner: pd.Series, *, exact_mode: str, root: Path) -> dict[str, object]:
    raw_path = _text(scanner.get("raw_source_path") or scanner.get("evidence_path"))
    path = Path(raw_path)
    if raw_path and not path.is_absolute():
        path = root / path
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    records = payload.get("rows", []) if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        return {}
    spread_id, strategy_id = ids_from_exact_mode(exact_mode)
    pair_id = _text(scanner.get("pair_id"))
    for record in records:
        if not isinstance(record, dict):
            continue
        if _text(record.get("pair_id")) == pair_id and record.get("spread_id") == spread_id and record.get("strategy_id") == strategy_id:
            return record
    return {}


def _matching_row(frame: pd.DataFrame, *, pair: str, timeframe: str, exact_mode: str) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=object)
    pair_keys = frame.get("pair", pd.Series("", index=frame.index)).map(_pair_key)
    intervals = frame.get("timeframe", frame.get("interval", pd.Series("", index=frame.index))).map(_interval_key)
    modes = frame.get("exact_mode", pd.Series("", index=frame.index)).map(normalize_exact_mode)
    matches = frame.loc[pair_keys.eq(_pair_key(pair)) & intervals.eq(_interval_key(timeframe)) & modes.eq(exact_mode)]
    return matches.iloc[0] if not matches.empty else pd.Series(dtype=object)


def _matching_pair_row(frame: pd.DataFrame, pair: str) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=object)
    matches = frame.loc[frame.get("pair", pd.Series("", index=frame.index)).map(_pair_key).eq(_pair_key(pair))]
    return matches.iloc[0] if not matches.empty else pd.Series(dtype=object)


def _matching_history_path(root: Path, *, asset_x: str, asset_y: str, timeframe: str) -> Path | None:
    candidates: list[tuple[int, pd.Timestamp, Path]] = []
    directory = root / "data" / "raw" / "pair_details"
    for path in directory.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            history = payload.get("history", [])
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(history, list):
            continue
        if _asset(payload.get("asset_x")) != asset_x or _asset(payload.get("asset_y")) != asset_y:
            continue
        if _interval_key(payload.get("interval")) != _interval_key(timeframe):
            continue
        latest = pd.to_datetime(pd.DataFrame(history).get("timestamp", pd.Series(dtype=object)), utc=True, errors="coerce").max()
        candidates.append((len(history), latest if pd.notna(latest) else pd.Timestamp.min.tz_localize("UTC"), path))
    return max(candidates, default=(0, pd.Timestamp.min.tz_localize("UTC"), None), key=lambda value: (value[0], value[1]))[2]


def _history_frame(path: Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return pd.DataFrame()
    frame = pd.DataFrame(payload.get("history", []))
    if frame.empty or not {"timestamp", "price_x", "price_y"}.issubset(frame.columns):
        return pd.DataFrame()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp", drop=False)
    for column in frame.columns:
        if column != "timestamp":
            numeric = pd.to_numeric(frame[column], errors="coerce")
            if numeric.notna().any():
                frame[column] = numeric
    return frame


def _asset_tradable(frame: pd.DataFrame, asset: str, column: str) -> bool:
    if frame.empty or "asset" not in frame or column not in frame:
        return False
    matches = frame.loc[frame["asset"].map(_asset).eq(asset)]
    return bool(matches[column].map(_boolish).any()) if not matches.empty else False


def _regime_readiness_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, candidate in frame.iterrows():
        for regime in REGIMES:
            for behavior in REGIME_BEHAVIORS:
                ready = _boolish(candidate.get("costed_walk_forward_ready"))
                rows.append(
                    {
                        "pair": candidate.get("pair", ""),
                        "asset_x": candidate.get("asset_x", ""),
                        "asset_y": candidate.get("asset_y", ""),
                        "timeframe": candidate.get("timeframe", ""),
                        "exact_mode": candidate.get("exact_mode", ""),
                        "regime": regime,
                        "regime_behavior": behavior,
                        "trades": "",
                        "profit_factor": "",
                        "sharpe": "",
                        "max_drawdown": "",
                        "total_return": "",
                        "acceptance_eligible": False,
                        "validation_status": "READY_TO_RUN" if ready else "BLOCKED_PREREQUISITES",
                        "acceptance_reason": candidate.get("acceptance_reason", ""),
                        "blocker": candidate.get("blocker", ""),
                        "evidence_path": candidate.get("evidence_path", ""),
                    }
                )
    return pd.DataFrame(rows, columns=REGIME_COLUMNS)


def _value(primary: pd.Series, secondary: dict[str, object], field: str) -> object:
    value = primary.get(field, "") if not primary.empty else ""
    return value if _text(value) else secondary.get(field, "")


def _pair_key(value: object) -> str:
    return "/".join(_asset(part) for part in _text(value).replace("-USD-", "-USD/").split("/") if _asset(part))


def _asset(value: object) -> str:
    text = _text(value).upper()
    for suffix in ("-USDT", "-USD", "-PERP"):
        if text.endswith(suffix):
            return text[: -len(suffix)]
    return text


def _interval_key(value: object) -> str:
    text = _text(value).lower().replace(" ", "")
    return {"daily": "1d", "day": "1d", "1day": "1d", "5min": "5m", "5mins": "5m"}.get(text, text)


def _timestamp(value: object) -> datetime | None:
    parsed = pd.to_datetime(_text(value), utc=True, errors="coerce")
    return parsed.to_pydatetime() if pd.notna(parsed) else None


def _fresh(value: object, *, now: datetime, hours: float) -> bool:
    parsed = _timestamp(value)
    return bool(parsed and 0 <= (now - parsed).total_seconds() <= hours * 3600)


def _number(value: object) -> object:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return "" if pd.isna(parsed) else float(parsed)


def _boolish(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return _text(value).lower() in {"1", "true", "yes", "y"}


def _text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(frame, path, index=False)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, content, encoding="utf-8")


def _markdown(frame: pd.DataFrame, regime: pd.DataFrame, *, now: datetime) -> str:
    ready = int(frame.get("costed_walk_forward_ready", pd.Series(dtype=bool)).map(_boolish).sum())
    testnet = int(frame.get("hyperliquid_testnet_pair_tradable", pd.Series(dtype=bool)).map(_boolish).sum())
    lines = [
        "# YouTube/Wizard to Hyperliquid Validation",
        "",
        f"- As of: `{now.isoformat()}`",
        f"- Discovery-qualified candidates: `{len(frame)}`",
        f"- Costed walk-forward ready: `{ready}`",
        f"- Hyperliquid Testnet-compatible pairs: `{testnet}`",
        f"- Regime/behavior rows staged: `{len(regime)}`",
        "- Promotion authority: `none`; all rows remain research-only until local acceptance and forward evidence pass.",
        "- Proxy rule: no generic z-score may substitute for Static, Dynamic, OU, or Copula exact modes.",
        "",
    ]
    if frame.empty:
        return "\n".join(lines + ["No discovery-qualified hypotheses were available.", ""])
    preview = frame[
        [
            "pair",
            "exact_mode",
            "wizard_sharpe",
            "wizard_returns_total_pct",
            "history_rows",
            "funding_ready",
            "slippage_model_ready",
            "hyperliquid_testnet_pair_tradable",
            "validation_status",
            "next_step",
        ]
    ]
    return "\n".join(lines + [preview.to_markdown(index=False), "", "## Current Decision", "", "Exact-mode performance tests did not run because their prerequisites are not complete. Blank metrics are intentional, not zero returns.", ""])
