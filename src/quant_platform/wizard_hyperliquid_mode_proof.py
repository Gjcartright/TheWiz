from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_platform.active_pipeline import CommandResult
from quant_platform.crypto_wizards_history import (
    CryptoWizardsCustomSeriesBacktestRequest,
    fetch_custom_series_backtest,
)


ROOT = Path(__file__).resolve().parents[2]
REQUEST_CAP = 3
CUSTOM_SERIES_MAX_ROWS = 1100

PROOF_COLUMNS = [
    "candidate_set_id",
    "discovery_policy_schema_version",
    "discovery_policy_hash",
    "pair",
    "asset_x",
    "asset_y",
    "venue",
    "local_interval",
    "exact_mode",
    "vendor_request_strategy",
    "vendor_request_spread_type",
    "wizard_period",
    "proof_observations",
    "proof_window_kind",
    "history_rows",
    "mode_proof_status",
    "execution_enabled",
    "credits_estimated",
    "vendor_sharpe",
    "vendor_sortino",
    "vendor_total_return",
    "vendor_annual_return",
    "vendor_max_drawdown",
    "vendor_win_rate",
    "vendor_var",
    "vendor_cvar",
    "vendor_half_life",
    "vendor_hurst",
    "vendor_hedge_ratio",
    "vendor_last_zscore",
    "vendor_last_zscore_roll",
    "vendor_sigma0_crossings",
    "vendor_sigma2_crossings",
    "vendor_engle_granger_cointegrated",
    "vendor_engle_granger_pvalue",
    "vendor_formula_parity_status",
    "vendor_spread_formula",
    "vendor_spread_max_abs_error",
    "vendor_zscore_formula",
    "vendor_zscore_max_abs_error",
    "vendor_zscore_roll_formula",
    "vendor_zscore_roll_max_abs_error",
    "vendor_formula_pit_status",
    "vendor_history_available",
    "mode_computation_source",
    "promotion_allowed",
    "proof_validity",
    "validity_blocker",
    "training_eligible",
    "blocker",
    "next_step",
    "request_path",
    "response_path",
    "evidence_path",
]


def run_hyperliquid_wizard_mode_proofs(
    *,
    root: Path = ROOT,
    max_pairs: int = REQUEST_CAP,
    execute: bool = False,
    api_key: str | None = None,
    now: datetime | None = None,
) -> CommandResult:
    """Preflight or run bounded Wizard custom-series proofs for queue-ready pairs.

    The default is a no-credit preflight. Execution needs both ``execute=True`` and
    ``QPA_ENABLE_WIZARD_CUSTOM_SERIES_PROOF=true``. Results remain research-only.
    """

    timestamp = _as_utc(now or datetime.now(timezone.utc))
    active = root / "reports" / "active"
    dashboard = root / "reports" / "dashboard"
    queue_path = active / "hyperliquid_wizard_hypothesis_queue.csv"
    proof_path = active / "hyperliquid_wizard_vendor_mode_proofs.csv"
    queue = _read_csv(queue_path)
    eligible = queue.loc[queue.get("vendor_custom_series_eligible", pd.Series(False, index=queue.index)).map(_truthy)].copy()
    eligible_total = int(len(eligible))
    existing_completed = pd.DataFrame()
    existing = _read_csv(proof_path)
    if not existing.empty and "mode_proof_status" in existing.columns:
        existing_completed = existing.loc[existing["mode_proof_status"].eq("completed")].copy()
        existing_completed = _refresh_existing_metrics(existing_completed, root=root)
        existing_completed = _enrich_existing_completed(existing_completed, queue=queue)
        completed_keys = {_proof_identity(row) for _, row in existing_completed.iterrows()}
        eligible = eligible.loc[
            ~eligible.apply(lambda row: _proof_identity(row) in completed_keys, axis=1)
        ].copy()
    selected = eligible.head(max(0, min(int(max_pairs), REQUEST_CAP))).reset_index(drop=True)

    execution_enabled = bool(execute and _truthy(os.getenv("QPA_ENABLE_WIZARD_CUSTOM_SERIES_PROOF", "")))
    resolved_api_key = api_key or os.getenv("CRYPTO_WIZARDS_API_KEY", "").strip()
    rows: list[dict[str, object]] = existing_completed.to_dict("records") if not existing_completed.empty else []
    for _, candidate in selected.iterrows():
        rows.append(
            _proof_row(
                candidate,
                root=root,
                timestamp=timestamp,
                execute=execute,
                execution_enabled=execution_enabled,
                api_key=resolved_api_key,
                queue_path=queue_path,
            )
        )

    frame = pd.DataFrame(rows, columns=PROOF_COLUMNS)
    active.mkdir(parents=True, exist_ok=True)
    dashboard.mkdir(parents=True, exist_ok=True)
    summary_path = active / "hyperliquid_wizard_vendor_mode_proofs.md"
    dashboard_path = dashboard / "hyperliquid_wizard_vendor_mode_proofs_dashboard.csv"
    _write_csv(frame, proof_path)
    _write_csv(frame, dashboard_path)
    _write_text(summary_path, _markdown(frame, selected=len(selected), eligible=eligible_total, execute=execute))

    completed = int(frame.get("mode_proof_status", pd.Series(dtype=str)).eq("completed").sum()) if not frame.empty else 0
    return CommandResult(
        paths={"proofs": proof_path, "summary_md": summary_path, "dashboard": dashboard_path},
        summary={
            "queue_eligible": eligible_total,
            "selected": int(len(selected)),
            "completed": completed,
            "preflight_only": bool(not execute),
            "execution_enabled": execution_enabled,
            "request_cap": REQUEST_CAP,
        },
    )


def _proof_row(
    candidate: pd.Series,
    *,
    root: Path,
    timestamp: datetime,
    execute: bool,
    execution_enabled: bool,
    api_key: str,
    queue_path: Path,
) -> dict[str, object]:
    base = _base_row(candidate, queue_path=queue_path)
    try:
        request = _request_from_candidate(candidate, root=root)
    except ValueError as exc:
        return {**base, "mode_proof_status": "input_blocked", "blocker": str(exc), "next_step": "repair_hypothesis_queue_inputs"}

    if not execute:
        return {
            **base,
            "history_rows": len(request.series_1_closes),
            "proof_observations": len(request.series_1_closes),
            "proof_window_kind": _proof_window_kind(base.get("wizard_period", ""), len(request.series_1_closes)),
            "mode_proof_status": "preflight_ready",
            "execution_enabled": False,
            "credits_estimated": 0,
            "next_step": "set QPA_ENABLE_WIZARD_CUSTOM_SERIES_PROOF=true and rerun with --execute-wizard-proof after review",
        }
    if not execution_enabled:
        return {
            **base,
            "history_rows": len(request.series_1_closes),
            "proof_observations": len(request.series_1_closes),
            "proof_window_kind": _proof_window_kind(base.get("wizard_period", ""), len(request.series_1_closes)),
            "mode_proof_status": "execution_disabled",
            "execution_enabled": False,
            "credits_estimated": 0,
            "blocker": "QPA_ENABLE_WIZARD_CUSTOM_SERIES_PROOF_not_true",
            "next_step": "enable the bounded custom-series proof gate only after reviewing the selected rows",
        }
    if not api_key:
        return {
            **base,
            "history_rows": len(request.series_1_closes),
            "proof_observations": len(request.series_1_closes),
            "proof_window_kind": _proof_window_kind(base.get("wizard_period", ""), len(request.series_1_closes)),
            "mode_proof_status": "execution_blocked",
            "execution_enabled": True,
            "credits_estimated": 0,
            "blocker": "CRYPTO_WIZARDS_API_KEY_missing",
            "next_step": "configure the key in .env.local",
        }

    raw_dir = root / "data" / "raw" / "crypto_wizards_custom_series_proofs" / timestamp.strftime("%Y-%m-%d_%H%M%S")
    stem = _safe_filename(f"{candidate.get('pair', '')}_{candidate.get('local_interval', '')}_{candidate.get('exact_mode', '')}")
    request_path = raw_dir / f"{stem}_request.json"
    response_path = raw_dir / f"{stem}_response.json"
    request_payload = request.payload()
    _write_json(request_path, request_payload)
    try:
        response = fetch_custom_series_backtest(request, api_key=api_key)
    except Exception as exc:  # pragma: no cover - exercised through the public report contract
        return {
            **base,
            "history_rows": len(request.series_1_closes),
            "proof_observations": len(request.series_1_closes),
            "proof_window_kind": _proof_window_kind(base.get("wizard_period", ""), len(request.series_1_closes)),
            "mode_proof_status": "request_failed",
            "execution_enabled": True,
            "credits_estimated": 2,
            "blocker": f"{type(exc).__name__}:{exc}",
            "next_step": "inspect the saved request and API response settings before retrying",
            "request_path": _relative(request_path, root=root),
        }
    _write_json(response_path, response)
    metrics = {**_response_metrics(response), **_formula_parity_metrics(request_payload, response)}
    completed_row = {
        **base,
        "history_rows": len(request.series_1_closes),
        "proof_observations": len(request.series_1_closes),
        "proof_window_kind": _proof_window_kind(base.get("wizard_period", ""), len(request.series_1_closes)),
        "mode_proof_status": "completed",
        "execution_enabled": True,
        "credits_estimated": 2,
        **metrics,
        "mode_computation_source": "crypto_wizards_custom_series_backtest",
        "next_step": "inspect vendor history and signal availability, then run a separate local after-cost replay",
        "request_path": _relative(request_path, root=root),
        "response_path": _relative(response_path, root=root),
        "evidence_path": f"{base['evidence_path']};{_relative(request_path, root=root)};{_relative(response_path, root=root)}",
    }
    return {**completed_row, **_proof_validity_fields(pd.Series(completed_row))}


def _base_row(candidate: pd.Series, *, queue_path: Path) -> dict[str, object]:
    return {
        "candidate_set_id": _text(candidate.get("candidate_set_id", "")),
        "discovery_policy_schema_version": _text(candidate.get("discovery_policy_schema_version", "")),
        "discovery_policy_hash": _text(candidate.get("discovery_policy_hash", "")),
        "pair": _text(candidate.get("pair", "")),
        "asset_x": _text(candidate.get("asset_x", "")),
        "asset_y": _text(candidate.get("asset_y", "")),
        "venue": _text(candidate.get("venue", "")),
        "local_interval": _text(candidate.get("local_interval", "")),
        "exact_mode": _text(candidate.get("exact_mode", "")),
        "vendor_request_strategy": _text(candidate.get("vendor_request_strategy", "")),
        "vendor_request_spread_type": _text(candidate.get("vendor_request_spread_type", "")),
        "wizard_period": _int_or_none(candidate.get("wizard_period", "")) or "",
        "proof_observations": _int_or_none(candidate.get("proof_observations", "")) or "",
        "proof_window_kind": "",
        "history_rows": "",
        "mode_proof_status": "",
        "execution_enabled": False,
        "credits_estimated": 0,
        "vendor_sharpe": "",
        "vendor_sortino": "",
        "vendor_total_return": "",
        "vendor_annual_return": "",
        "vendor_max_drawdown": "",
        "vendor_win_rate": "",
        "vendor_var": "",
        "vendor_cvar": "",
        "vendor_half_life": "",
        "vendor_hurst": "",
        "vendor_hedge_ratio": "",
        "vendor_last_zscore": "",
        "vendor_last_zscore_roll": "",
        "vendor_sigma0_crossings": "",
        "vendor_sigma2_crossings": "",
        "vendor_engle_granger_cointegrated": "",
        "vendor_engle_granger_pvalue": "",
        "vendor_formula_parity_status": "",
        "vendor_spread_formula": "",
        "vendor_spread_max_abs_error": "",
        "vendor_zscore_formula": "",
        "vendor_zscore_max_abs_error": "",
        "vendor_zscore_roll_formula": "",
        "vendor_zscore_roll_max_abs_error": "",
        "vendor_formula_pit_status": "",
        "vendor_history_available": False,
        "mode_computation_source": "",
        "promotion_allowed": False,
        "proof_validity": "NOT_COMPLETED",
        "validity_blocker": "vendor_proof_not_completed",
        "training_eligible": False,
        "blocker": "",
        "next_step": "",
        "request_path": "",
        "response_path": "",
        "evidence_path": f"{_relative(queue_path, root=queue_path.parents[2])};{_text(candidate.get('evidence_path', ''))}",
    }


def _request_from_candidate(candidate: pd.Series, *, root: Path) -> CryptoWizardsCustomSeriesBacktestRequest:
    history_path = _resolve_path(_text(candidate.get("local_history_path", "")), root=root)
    if history_path is None or not history_path.exists():
        raise ValueError("local_history_path_missing")
    payload = json.loads(history_path.read_text(encoding="utf-8"))
    history = pd.DataFrame(payload.get("history", []))
    if history.empty or not {"price_x", "price_y"}.issubset(history.columns):
        raise ValueError("local_history_missing_two_leg_prices")
    required_history_columns = {"open_x", "price_x", "open_y", "price_y"}
    if not required_history_columns.issubset(history.columns):
        raise ValueError("local_history_missing_two_leg_open_close_prices")
    prices = history[["open_x", "price_x", "open_y", "price_y"]].apply(pd.to_numeric, errors="coerce").dropna()
    observations = _requested_observations(candidate)
    prices = prices[(prices > 0).all(axis=1)].tail(observations)
    if len(prices) < 50:
        raise ValueError("local_history_has_insufficient_valid_prices")

    required = ("entry_level", "exit_level", "x_weighting", "slippage_rate", "commission_rate", "roll_w")
    settings: dict[str, float] = {}
    for key in required:
        value = _float(candidate.get(key, ""))
        if value is None:
            raise ValueError(f"captured_setting_missing:{key}")
        settings[key] = value
    stop_loss = _float(candidate.get("stop_loss_rate", ""))
    exit_n_periods = _int_or_none(candidate.get("exit_n_periods", ""))
    stop_loss = stop_loss if stop_loss is not None and stop_loss > 0.0 else None
    exit_n_periods = exit_n_periods if exit_n_periods is not None and exit_n_periods > 0 else None
    strategy = _text(candidate.get("vendor_request_strategy", ""))
    if not strategy:
        raise ValueError("vendor_request_strategy_missing")
    spread_type = _text(candidate.get("vendor_request_spread_type", "")) or None
    return CryptoWizardsCustomSeriesBacktestRequest(
        series_1_opens=tuple(float(value) for value in prices["open_x"].tolist()),
        series_1_closes=tuple(float(value) for value in prices["price_x"].tolist()),
        series_2_opens=tuple(float(value) for value in prices["open_y"].tolist()),
        series_2_closes=tuple(float(value) for value in prices["price_y"].tolist()),
        strategy=strategy,
        spread_type=spread_type,
        roll_w=int(settings["roll_w"]),
        entry_level=settings["entry_level"],
        exit_level=settings["exit_level"],
        x_weighting=settings["x_weighting"],
        slippage_rate=settings["slippage_rate"],
        commission_rate=settings["commission_rate"],
        stop_loss_rate=stop_loss,
        exit_n_periods=exit_n_periods,
        with_history=True,
    )


def _response_metrics(response: dict[str, Any]) -> dict[str, object]:
    data = response.get("data") if isinstance(response.get("data"), dict) else {}
    returns = data.get("strat_returns") if isinstance(data.get("strat_returns"), dict) else {}
    history = response.get("history")
    history_available = bool(history) and isinstance(history, (dict, list))
    history_dict = history if isinstance(history, dict) else {}
    spread_stats = history_dict.get("spread_stats") if isinstance(history_dict.get("spread_stats"), dict) else {}
    coint = history_dict.get("coint_eg") if isinstance(history_dict.get("coint_eg"), dict) else {}
    last_zscore = spread_stats.get("last_zscore") if isinstance(spread_stats.get("last_zscore"), dict) else {}
    return {
        "vendor_sharpe": data.get("sharpe_ratio", ""),
        "vendor_sortino": data.get("sortino_ratio", ""),
        "vendor_total_return": returns.get("total_return", ""),
        "vendor_annual_return": returns.get("annual_return", ""),
        "vendor_max_drawdown": data.get("max_drawdown", ""),
        "vendor_win_rate": data.get("win_rate", ""),
        "vendor_var": data.get("var", ""),
        "vendor_cvar": data.get("cvar", ""),
        "vendor_half_life": spread_stats.get("half_life", ""),
        "vendor_hurst": spread_stats.get("hurst", ""),
        "vendor_hedge_ratio": spread_stats.get("hedge_ratio", ""),
        "vendor_last_zscore": last_zscore.get("zscore", ""),
        "vendor_last_zscore_roll": last_zscore.get("zscore_roll", ""),
        "vendor_sigma0_crossings": spread_stats.get("sigma0crossings", ""),
        "vendor_sigma2_crossings": spread_stats.get("sigma2crossings", ""),
        "vendor_engle_granger_cointegrated": coint.get("is_coint", ""),
        "vendor_engle_granger_pvalue": coint.get("p_value", ""),
        "vendor_history_available": history_available,
    }


def _formula_parity_metrics(request: dict[str, Any], response: dict[str, Any]) -> dict[str, object]:
    unavailable = {
        "vendor_formula_parity_status": "not_available",
        "vendor_spread_formula": "",
        "vendor_spread_max_abs_error": "",
        "vendor_zscore_formula": "",
        "vendor_zscore_max_abs_error": "",
        "vendor_zscore_roll_formula": "",
        "vendor_zscore_roll_max_abs_error": "",
        "vendor_formula_pit_status": "historical_vendor_response_only",
    }
    params = request.get("params") if isinstance(request.get("params"), dict) else {}
    history = response.get("history") if isinstance(response.get("history"), dict) else {}
    stats = history.get("spread_stats") if isinstance(history.get("spread_stats"), dict) else {}
    if params.get("strategy") != "Spread" or params.get("spread_type") != "Static":
        return unavailable
    try:
        x = np.asarray(params["series_1_closes"], dtype=float)
        y = np.asarray(params["series_2_closes"], dtype=float)
        vendor_spread = np.asarray(stats["spread"], dtype=float)
        vendor_zscore = np.asarray(stats["zscore"], dtype=float)
        vendor_zscore_roll = np.asarray(stats["zscore_roll"], dtype=float)
        window = int(params["roll_w"])
    except (KeyError, TypeError, ValueError):
        return unavailable
    if len(x) < 2 or len({len(x), len(y), len(vendor_spread), len(vendor_zscore), len(vendor_zscore_roll)}) != 1:
        return unavailable
    log_used = bool(stats.get("log_used", False))
    transformed_x = np.log(x) if log_used else x
    transformed_y = np.log(y) if log_used else y
    beta, alpha = np.polyfit(transformed_x, transformed_y, 1)
    spread = transformed_y - (alpha + beta * transformed_x)
    spread_std = spread.std(ddof=1)
    if not np.isfinite(spread_std) or spread_std <= 0.0 or window < 2:
        return unavailable
    zscore = (spread - spread.mean()) / spread_std
    spread_series = pd.Series(spread)
    rolling_mean = spread_series.rolling(window, min_periods=window).mean()
    rolling_std = spread_series.rolling(window, min_periods=window).std(ddof=1)
    zscore_roll = spread_series.sub(rolling_mean).div(rolling_std.where(rolling_std.abs() > 1e-12)).fillna(0.0)
    spread_error = float(np.max(np.abs(spread - vendor_spread)))
    zscore_error = float(np.max(np.abs(zscore - vendor_zscore)))
    zscore_roll_error = float(np.max(np.abs(zscore_roll.to_numpy() - vendor_zscore_roll)))
    tolerance = 1e-9
    parity = "exact_reconstruction" if max(spread_error, zscore_error, zscore_roll_error) <= tolerance else "mismatch"
    return {
        "vendor_formula_parity_status": parity,
        "vendor_spread_formula": (
            "ols_log_y_on_log_x_with_intercept: log(y)-(alpha+beta*log(x))"
            if log_used
            else "ols_y_on_x_with_intercept: y-(alpha+beta*x)"
        ),
        "vendor_spread_max_abs_error": spread_error,
        "vendor_zscore_formula": "full_sample_sample_std_ddof1",
        "vendor_zscore_max_abs_error": zscore_error,
        "vendor_zscore_roll_formula": f"rolling_{window}_sample_std_ddof1_warmup_zero",
        "vendor_zscore_roll_max_abs_error": zscore_roll_error,
        "vendor_formula_pit_status": "hindsight_full_sample_fit_not_live_signal_safe",
    }


def _proof_identity(row: pd.Series) -> tuple[str, str, str, int]:
    return (
        _text(row.get("pair", "")).upper(),
        _text(row.get("local_interval", "")).lower(),
        _text(row.get("exact_mode", "")).lower(),
        _identity_observations(row),
    )


def _refresh_existing_metrics(frame: pd.DataFrame, *, root: Path) -> pd.DataFrame:
    refreshed = frame.copy().astype(object)
    for index, row in refreshed.iterrows():
        response_path = _resolve_path(_text(row.get("response_path", "")), root=root)
        if response_path is None or not response_path.exists():
            continue
        try:
            response = json.loads(response_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        request_path = _resolve_path(_text(row.get("request_path", "")), root=root)
        request: dict[str, Any] = {}
        if request_path is not None and request_path.exists():
            try:
                request = json.loads(request_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                request = {}
        metrics = {**_response_metrics(response), **_formula_parity_metrics(request, response)}
        for field, value in metrics.items():
            refreshed.at[index, field] = value
        validity = _proof_validity_fields(refreshed.loc[index])
        for field, value in validity.items():
            refreshed.at[index, field] = value
    return refreshed


def _enrich_existing_completed(frame: pd.DataFrame, *, queue: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    enriched = frame.copy().astype(object)
    metadata_fields = (
        "candidate_set_id",
        "discovery_policy_schema_version",
        "discovery_policy_hash",
        "wizard_period",
    )
    for index, row in enriched.iterrows():
        matches = queue.loc[
            queue.get("pair", pd.Series("", index=queue.index)).map(_text).str.upper().eq(_text(row.get("pair", "")).upper())
            & queue.get("local_interval", pd.Series("", index=queue.index)).map(_text).str.lower().eq(_text(row.get("local_interval", "")).lower())
            & queue.get("exact_mode", pd.Series("", index=queue.index)).map(_text).str.lower().eq(_text(row.get("exact_mode", "")).lower())
        ]
        if not matches.empty:
            candidate = matches.iloc[0]
            for field in metadata_fields:
                value = candidate.get(field, "")
                if _text(value):
                    enriched.at[index, field] = value
        observations = _identity_observations(enriched.loc[index])
        enriched.at[index, "proof_observations"] = observations
        enriched.at[index, "proof_window_kind"] = _proof_window_kind(
            enriched.at[index, "wizard_period"] if "wizard_period" in enriched.columns else "",
            observations,
        )
        validity = _proof_validity_fields(enriched.loc[index])
        for field, value in validity.items():
            enriched.at[index, field] = value
    return enriched


def _requested_observations(candidate: pd.Series) -> int:
    requested = _int_or_none(candidate.get("proof_observations", ""))
    if requested is None:
        requested = _int_or_none(candidate.get("wizard_period", ""))
    if requested is None:
        raise ValueError("proof_observations_missing")
    if requested < 50:
        raise ValueError("proof_observations_below_vendor_minimum_50")
    return min(requested, CUSTOM_SERIES_MAX_ROWS)


def _identity_observations(row: pd.Series) -> int:
    for field in ("proof_observations", "history_rows", "wizard_period"):
        value = _int_or_none(row.get(field, ""))
        if value is not None and value > 0:
            return min(value, CUSTOM_SERIES_MAX_ROWS)
    return 0


def _proof_window_kind(wizard_period: object, proof_observations: object) -> str:
    period = _int_or_none(wizard_period)
    observations = _int_or_none(proof_observations)
    if period is None or observations is None:
        return "unknown_window"
    if observations == min(period, CUSTOM_SERIES_MAX_ROWS):
        return "scanner_horizon_parity"
    return "extended_history" if observations > period else "truncated_history"


def _proof_validity_fields(row: pd.Series) -> dict[str, object]:
    if _text(row.get("mode_proof_status", "")) != "completed":
        return {
            "proof_validity": "NOT_COMPLETED",
            "validity_blocker": "vendor_proof_not_completed",
            "training_eligible": False,
        }
    blockers = [
        "vendor_performance_accounting_not_reconstructed",
        "vendor_full_sample_statistics_are_not_live_signal_safe",
        "vendor_cost_and_funding_parity_not_proven",
    ]
    window_kind = _text(row.get("proof_window_kind", ""))
    if window_kind != "scanner_horizon_parity":
        blockers.append("proof_window_does_not_match_scanner_horizon")
    return {
        "proof_validity": "RESEARCH_DIAGNOSTIC_WEAK",
        "validity_blocker": ";".join(blockers),
        "training_eligible": False,
    }


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def _resolve_path(value: str, *, root: Path) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


def _float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _int_or_none(value: object) -> int | None:
    number = _float(value)
    return int(number) if number is not None else None


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return _text(value).lower() in {"1", "true", "yes", "y"}


def _text(value: object) -> str:
    if value is None or (not isinstance(value, (dict, list)) and pd.isna(value)):
        return ""
    return str(value).strip()


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _safe_filename(value: object) -> str:
    cleaned = "".join(character.lower() if character.isalnum() else "_" for character in str(value))
    return "_".join(part for part in cleaned.split("_") if part)[:160] or "mode_proof"


def _relative(path: Path, *, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _markdown(frame: pd.DataFrame, *, selected: int, eligible: int, execute: bool) -> str:
    lines = [
        "# Hyperliquid Wizard Vendor Mode Proofs",
        "",
        f"- Queue-eligible rows: `{eligible}`",
        f"- Selected under the `{REQUEST_CAP}` request cap: `{selected}`",
        f"- Execution requested: `{execute}`",
        "- Promotion authority: `none`; a completed proof is still research evidence.",
        "",
    ]
    if frame.empty:
        return "\n".join(lines + ["No queue-eligible hypotheses were available.", ""])
    preview = frame[
        [
            "pair",
            "local_interval",
            "exact_mode",
            "mode_proof_status",
            "credits_estimated",
            "blocker",
            "next_step",
        ]
    ]
    return "\n".join(lines + ["## Proof Queue", "", preview.to_markdown(index=False), ""])
