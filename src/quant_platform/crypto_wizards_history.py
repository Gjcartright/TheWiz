from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import os
from urllib.parse import urlencode

import requests

from quant_platform.api_extraction import CryptoWizardsFetchError
from quant_platform.crypto_wizards_catalog import BASE_URL


PRESCANNED_ROW_FEATURES = (
    "profile_match",
    "ml_confidence",
    "ou_optimal",
    "sharpe",
    "sortino",
    "returns_total",
    "win_rate",
    "closed",
    "mdd",
    "var",
    "cvar",
    "johansen_coint",
    "coint_eg",
    "coint_eg_p",
    "hurst",
    "half_life",
    "hedge_ratio",
    "x_weighting",
    "y_weighting",
    "copula",
    "corr_copula",
    "u1_given_u2",
    "u2_given_u1",
    "beta_lt",
    "zscore_window",
)


@dataclass(frozen=True)
class CryptoWizardsHistoryRequest:
    symbol_1: str
    symbol_2: str
    exchange: str = "Dydx"
    interval: str = "Min5"
    period: int = 320
    spread_type: str = "Static"
    roll_w: int = 42
    with_history: bool = True

    @property
    def pair_id(self) -> str:
        return f"{self.symbol_1}_{self.symbol_2}_{self.exchange}_{self.interval}_{self.period}"

    @property
    def pair(self) -> str:
        return f"{self.symbol_1}-{self.symbol_2}"

    def zscores_params(self) -> dict[str, object]:
        return {
            "symbol_1": self.symbol_1,
            "symbol_2": self.symbol_2,
            "exchange": self.exchange,
            "interval": self.interval,
            "period": self.period,
            "spread_type": self.spread_type,
            "roll_w": self.roll_w,
            "with_history": str(self.with_history).lower(),
        }

    def backtest_params(
        self,
        *,
        strategy: str = "Spread",
        entry_level: float = 2.0,
        exit_level: float = 0.0,
        x_weighting: float = 0.5,
        slippage_rate: float = 0.0005,
        commission_rate: float = 0.0005,
        stop_loss_rate: float = 0.10,
        exit_n_periods: int | None = None,
    ) -> dict[str, object]:
        params: dict[str, object] = {
            **self.zscores_params(),
            "strategy": strategy,
            "entry_level": entry_level,
            "exit_level": exit_level,
            "x_weighting": x_weighting,
            "slippage_rate": slippage_rate,
            "commission_rate": commission_rate,
            "stop_loss_rate": stop_loss_rate,
        }
        if exit_n_periods is not None:
            params["exit_n_periods"] = exit_n_periods
        return params


def fetch_prescanned_pairs(
    *,
    api_key: str | None = None,
    base_url: str = BASE_URL,
    priority: str = "Sharpe",
    strategy: str = "Spread",
    exchange: str = "Dydx",
    interval: str = "Min5",
    asset: str | None = None,
    timeout: float = 30.0,
) -> list[dict[str, Any]]:
    params: dict[str, object] = {
        "priority": priority,
        "strategy": strategy,
        "exchange": exchange,
        "interval": interval,
    }
    if asset:
        params["asset"] = asset
    payload = _get_json(
        f"{base_url.rstrip('/')}/v1beta/prescanned",
        params=params,
        api_key=api_key,
        timeout=timeout,
    )
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("items", "data", "results", "pairs"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    raise CryptoWizardsFetchError("Crypto Wizards prescanned response did not contain a pair list")


def fetch_zscores_history(
    request: CryptoWizardsHistoryRequest,
    *,
    api_key: str | None = None,
    base_url: str = BASE_URL,
    timeout: float = 30.0,
) -> dict[str, Any]:
    return _get_json(
        f"{base_url.rstrip('/')}/v1beta/zscores",
        params=request.zscores_params(),
        api_key=api_key,
        timeout=timeout,
    )


def fetch_backtest_history(
    request: CryptoWizardsHistoryRequest,
    *,
    api_key: str | None = None,
    base_url: str = BASE_URL,
    timeout: float = 30.0,
    strategy: str = "Spread",
    entry_level: float = 2.0,
    exit_level: float = 0.0,
    x_weighting: float = 0.5,
    slippage_rate: float = 0.0005,
    commission_rate: float = 0.0005,
    stop_loss_rate: float = 0.10,
    exit_n_periods: int | None = None,
) -> dict[str, Any]:
    return _get_json(
        f"{base_url.rstrip('/')}/v1beta/backtest",
        params=request.backtest_params(
            strategy=strategy,
            entry_level=entry_level,
            exit_level=exit_level,
            x_weighting=x_weighting,
            slippage_rate=slippage_rate,
            commission_rate=commission_rate,
            stop_loss_rate=stop_loss_rate,
            exit_n_periods=exit_n_periods,
        ),
        api_key=api_key,
        timeout=timeout,
    )


def payload_from_zscores_history(
    request: CryptoWizardsHistoryRequest,
    response: dict[str, Any],
    *,
    prescanned_row: dict[str, Any] | None = None,
) -> dict[str, Any]:
    history_payload = response.get("history") if isinstance(response, dict) else None
    if not isinstance(history_payload, dict):
        history_payload = {}
    return _payload_from_history_payload(
        request,
        response,
        history_payload=history_payload,
        prescanned_row=prescanned_row,
        source_path="/v1beta/zscores",
        source_note=(
            "Official Crypto Wizards /v1beta/zscores history. Contains Crypto Wizards spread/zscore "
            "research history, not raw two-leg candles, funding, or native dashboard ECM fields."
        ),
    )


def payload_from_backtest_history(
    request: CryptoWizardsHistoryRequest,
    response: dict[str, Any],
    *,
    prescanned_row: dict[str, Any] | None = None,
) -> dict[str, Any]:
    history_payload = response.get("history") if isinstance(response, dict) else None
    if not isinstance(history_payload, dict):
        history_payload = {}
    spread_stats = history_payload.get("spread_stats")
    if not isinstance(spread_stats, dict):
        spread_stats = {}
    payload = _payload_from_history_payload(
        request,
        response,
        history_payload=spread_stats,
        prescanned_row=prescanned_row,
        source_path="/v1beta/backtest",
        source_note=(
            "Official Crypto Wizards /v1beta/backtest history. Contains CW backtest metrics and spread/zscore "
            "research history when with_history=true, not raw two-leg candles or funding."
        ),
    )
    data = response.get("data") if isinstance(response.get("data"), dict) else {}
    returns = data.get("strat_returns") if isinstance(data.get("strat_returns"), dict) else {}
    payload.update(
        {
            "annual_return": returns.get("annual_return"),
            "mean_period_return": returns.get("mean_period_return"),
            "returns_total": returns.get("total_return"),
            "drawdown": data.get("max_drawdown"),
            "sharpe": data.get("sharpe_ratio"),
            "sortino": data.get("sortino_ratio"),
            "cvar": data.get("cvar"),
            "var": data.get("var"),
            "win_rate": data.get("win_rate"),
        }
    )
    bt_returns = history_payload.get("bt_returns")
    if isinstance(bt_returns, list):
        for row, bt_return in zip(payload["history"], bt_returns):
            row["bt_return"] = bt_return
    for row in payload["history"]:
        row.update(
            {
                "annual_return": returns.get("annual_return"),
                "mean_period_return": returns.get("mean_period_return"),
                "returns_total": returns.get("total_return"),
                "drawdown": data.get("max_drawdown"),
                "sharpe": data.get("sharpe_ratio"),
                "sortino": data.get("sortino_ratio"),
                "cvar": data.get("cvar"),
                "var": data.get("var"),
                "win_rate": data.get("win_rate"),
            }
        )
    return payload


def _payload_from_history_payload(
    request: CryptoWizardsHistoryRequest,
    response: dict[str, Any],
    *,
    history_payload: dict[str, Any],
    prescanned_row: dict[str, Any] | None,
    source_path: str,
    source_note: str,
) -> dict[str, Any]:
    history = _history_rows(history_payload)
    snapshot = dict(prescanned_row or {})
    latest = response.get("data") if isinstance(response.get("data"), dict) else {}
    row_features = _row_features(snapshot, history_payload)
    history = [{**row, **row_features} for row in history]
    return {
        "pair_id": str(snapshot.get("pair_id") or request.pair_id),
        "pair": request.pair,
        "asset_x": request.symbol_1,
        "asset_y": request.symbol_2,
        "exchange": request.exchange.lower(),
        "interval": request.interval.lower(),
        "period": request.period,
        "strategy_mode": request.spread_type.lower(),
        "hedge_ratio": _first_present(history_payload.get("hedge_ratio"), snapshot.get("hedge_ratio")),
        "hurst": _first_present(history_payload.get("hurst"), snapshot.get("hurst")),
        "half_life": _first_present(history_payload.get("half_life"), snapshot.get("half_life")),
        "sharpe": snapshot.get("sharpe"),
        "sortino": snapshot.get("sortino"),
        "returns_total": snapshot.get("returns_total"),
        "win_rate": snapshot.get("win_rate"),
        "closed_trades": snapshot.get("closed"),
        "drawdown": snapshot.get("mdd"),
        "var": snapshot.get("var"),
        "cvar": snapshot.get("cvar"),
        "copula": snapshot.get("copula"),
        "corr_copula": snapshot.get("corr_copula"),
        "u1_given_u2": snapshot.get("u1_given_u2"),
        "u2_given_u1": snapshot.get("u2_given_u1"),
        "ml_confidence": snapshot.get("ml_confidence"),
        "profile_match": snapshot.get("profile_match"),
        "ou_optimal": snapshot.get("ou_optimal"),
        "x_weighting": snapshot.get("x_weighting"),
        "y_weighting": snapshot.get("y_weighting"),
        "zscore_latest": latest.get("zscore"),
        "zscore_roll_latest": latest.get("zscore_roll"),
        "source_url": f"{BASE_URL}{source_path}",
        "source_note": source_note,
        "prescanned": snapshot,
        "history": history,
    }


def write_zscores_pair_payload(
    request: CryptoWizardsHistoryRequest,
    response: dict[str, Any],
    output_dir: str | Path,
    *,
    prescanned_row: dict[str, Any] | None = None,
) -> Path:
    payload = payload_from_zscores_history(request, response, prescanned_row=prescanned_row)
    output = Path(output_dir) / f"pair_{_safe_filename(payload['pair_id'])}_{request.interval.lower()}_cw_zscores_history.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output


def write_backtest_pair_payload(
    request: CryptoWizardsHistoryRequest,
    response: dict[str, Any],
    output_dir: str | Path,
    *,
    prescanned_row: dict[str, Any] | None = None,
) -> Path:
    payload = payload_from_backtest_history(request, response, prescanned_row=prescanned_row)
    output = Path(output_dir) / f"pair_{_safe_filename(payload['pair_id'])}_{request.interval.lower()}_cw_backtest_history.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output


def crawl_prescanned_zscores_histories(
    *,
    api_key: str | None = None,
    base_url: str = BASE_URL,
    output_dir: str | Path,
    max_pairs: int = 10,
    priority: str = "Sharpe",
    strategy: str = "Spread",
    exchange: str = "Dydx",
    interval: str = "Min5",
    period: int = 320,
    spread_type: str = "Static",
    roll_w: int = 42,
    asset: str | None = None,
) -> list[Path]:
    if not api_key:
        api_key = os.getenv("CRYPTO_WIZARDS_API_KEY")
    if not api_key:
        raise CryptoWizardsFetchError("CRYPTO_WIZARDS_API_KEY is required for official Crypto Wizards API calls")
    pairs = fetch_prescanned_pairs(
        api_key=api_key,
        base_url=base_url,
        priority=priority,
        strategy=strategy,
        exchange=exchange,
        interval=interval,
        asset=asset,
    )
    written: list[Path] = []
    for row in pairs[:max_pairs]:
        symbol_1 = str(row.get("symbol_1") or row.get("asset_x") or "")
        symbol_2 = str(row.get("symbol_2") or row.get("asset_y") or "")
        if not symbol_1 or not symbol_2:
            continue
        request = CryptoWizardsHistoryRequest(
            symbol_1=symbol_1,
            symbol_2=symbol_2,
            exchange=exchange,
            interval=interval,
            period=int(row.get("period") or period),
            spread_type=spread_type,
            roll_w=int(row.get("zscore_window") or roll_w),
        )
        response = fetch_zscores_history(request, api_key=api_key, base_url=base_url)
        written.append(write_zscores_pair_payload(request, response, output_dir, prescanned_row=row))
    return written


def crawl_prescanned_backtest_histories(
    *,
    api_key: str | None = None,
    base_url: str = BASE_URL,
    output_dir: str | Path,
    max_pairs: int = 10,
    priority: str = "Sharpe",
    strategy: str = "Spread",
    exchange: str = "Dydx",
    interval: str = "Min5",
    period: int = 320,
    spread_type: str = "Static",
    roll_w: int = 42,
    asset: str | None = None,
    entry_level: float = 2.0,
    exit_level: float = 0.0,
    x_weighting: float = 0.5,
    slippage_rate: float = 0.0005,
    commission_rate: float = 0.0005,
    stop_loss_rate: float = 0.10,
) -> list[Path]:
    if not api_key:
        api_key = os.getenv("CRYPTO_WIZARDS_API_KEY")
    if not api_key:
        raise CryptoWizardsFetchError("CRYPTO_WIZARDS_API_KEY is required for official Crypto Wizards API calls")
    pairs = fetch_prescanned_pairs(
        api_key=api_key,
        base_url=base_url,
        priority=priority,
        strategy=strategy,
        exchange=exchange,
        interval=interval,
        asset=asset,
    )
    written: list[Path] = []
    for row in pairs[:max_pairs]:
        symbol_1 = str(row.get("symbol_1") or row.get("asset_x") or "")
        symbol_2 = str(row.get("symbol_2") or row.get("asset_y") or "")
        if not symbol_1 or not symbol_2:
            continue
        request = CryptoWizardsHistoryRequest(
            symbol_1=symbol_1,
            symbol_2=symbol_2,
            exchange=exchange,
            interval=interval,
            period=int(row.get("period") or period),
            spread_type=spread_type,
            roll_w=int(row.get("zscore_window") or roll_w),
        )
        response = fetch_backtest_history(
            request,
            api_key=api_key,
            base_url=base_url,
            strategy=strategy,
            entry_level=entry_level,
            exit_level=exit_level,
            x_weighting=float(row.get("x_weighting") or x_weighting),
            slippage_rate=slippage_rate,
            commission_rate=commission_rate,
            stop_loss_rate=stop_loss_rate,
        )
        written.append(write_backtest_pair_payload(request, response, output_dir, prescanned_row=row))
    return written


def official_min5_request_rows(
    *,
    symbol_1: str | None = None,
    symbol_2: str | None = None,
    base_url: str = BASE_URL,
    priority: str = "Sharpe",
    strategy: str = "Spread",
    exchange: str = "Dydx",
    interval: str = "Min5",
    period: int = 320,
    spread_type: str = "Static",
    roll_w: int = 42,
    asset: str | None = None,
    output_dir: str | Path = "data/raw/crypto_wizards_manual",
    entry_level: float = 2.0,
    exit_level: float = 0.0,
    x_weighting: float = 0.5,
    slippage_rate: float = 0.0005,
    commission_rate: float = 0.0005,
    stop_loss_rate: float = 0.10,
) -> list[dict[str, str]]:
    output_base = Path(output_dir)
    prescanned_params: dict[str, object] = {
        "priority": priority,
        "strategy": strategy,
        "exchange": exchange,
        "interval": interval,
    }
    if asset:
        prescanned_params["asset"] = asset
    rows = [
        _request_template_row(
            name="prescanned_min5_pairs",
            url=_request_url(base_url, "/v1beta/prescanned", prescanned_params),
            save_as=output_base / "prescanned_min5_pairs.json",
            notes="Discovers 5-minute Crypto Wizards candidate pairs. Use symbol_1 and symbol_2 from this response for pair requests.",
        )
    ]
    if not symbol_1 or not symbol_2:
        rows.append(
            {
                "request_name": "pair_history_requests",
                "method": "GET",
                "url": "",
                "curl": "",
                "save_as": "",
                "import_command": "",
                "notes": "Add --asset-x and --asset-y to generate exact /v1beta/zscores and /v1beta/backtest URLs for a pair.",
            }
        )
        return rows

    request = CryptoWizardsHistoryRequest(
        symbol_1=symbol_1,
        symbol_2=symbol_2,
        exchange=exchange,
        interval=interval,
        period=period,
        spread_type=spread_type,
        roll_w=roll_w,
    )
    pair_slug = _safe_filename(request.pair.lower())
    zscores_path = output_base / f"{pair_slug}_min5_zscores.json"
    backtest_path = output_base / f"{pair_slug}_min5_backtest.json"
    rows.append(
        _request_template_row(
            name="pair_min5_zscores_history",
            url=_request_url(base_url, "/v1beta/zscores", request.zscores_params()),
            save_as=zscores_path,
            import_command=(
                "PYTHONPATH=src python3 -m quant_platform.cli import-crypto-wizards-zscores "
                f"--json-path {zscores_path} --asset-x {symbol_1} --asset-y {symbol_2} "
                f"--exchange {exchange} --interval {interval} --period {period} "
                f"--spread-type {spread_type} --roll-w {roll_w} --run-research"
            ),
            notes="Imports official Crypto Wizards spread/z-score history for research rejection tests.",
        )
    )
    rows.append(
        _request_template_row(
            name="pair_min5_backtest_history",
            url=_request_url(
                base_url,
                "/v1beta/backtest",
                request.backtest_params(
                    strategy=strategy,
                    entry_level=entry_level,
                    exit_level=exit_level,
                    x_weighting=x_weighting,
                    slippage_rate=slippage_rate,
                    commission_rate=commission_rate,
                    stop_loss_rate=stop_loss_rate,
                ),
            ),
            save_as=backtest_path,
            import_command=(
                "PYTHONPATH=src python3 -m quant_platform.cli import-crypto-wizards-backtest "
                f"--json-path {backtest_path} --asset-x {symbol_1} --asset-y {symbol_2} "
                f"--exchange {exchange} --interval {interval} --period {period} "
                f"--spread-type {spread_type} --roll-w {roll_w} --run-research"
            ),
            notes="Preferred research artifact when credits allow because it carries CW backtest metrics plus history.",
        )
    )
    return rows


def _request_url(base_url: str, path: str, params: dict[str, object]) -> str:
    return f"{base_url.rstrip('/')}{path}?{urlencode(params)}"


def _request_template_row(
    *,
    name: str,
    url: str,
    save_as: Path,
    notes: str,
    import_command: str = "",
) -> dict[str, str]:
    return {
        "request_name": name,
        "method": "GET",
        "url": url,
        "curl": (
            f"curl -L '{url}' "
            "-H 'X-api-key: ${CRYPTO_WIZARDS_API_KEY}' "
            "-H 'Content-Type: application/json' "
            f"-o '{save_as}'"
        ),
        "save_as": str(save_as),
        "import_command": import_command,
        "notes": notes,
    }


def _get_json(url: str, *, params: dict[str, object], api_key: str | None, timeout: float) -> dict[str, Any] | list[Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-api-key"] = api_key
    try:
        response = requests.get(url, params=params, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as exc:
        raise CryptoWizardsFetchError(f"Crypto Wizards API request failed at {url}: {exc}") from exc
    except ValueError as exc:
        raise CryptoWizardsFetchError(f"Crypto Wizards API response was not JSON at {url}: {exc}") from exc


def _history_rows(history: dict[str, Any]) -> list[dict[str, Any]]:
    series = {
        "spread": history.get("spread"),
        "zscore": history.get("zscore"),
        "rolling_zscore": history.get("zscore_roll"),
    }
    lengths = [len(value) for value in series.values() if isinstance(value, list)]
    if not lengths:
        return []
    row_count = min(lengths)
    rows: list[dict[str, Any]] = []
    for idx in range(row_count):
        row: dict[str, Any] = {"timestamp": idx}
        for column, values in series.items():
            if isinstance(values, list) and idx < len(values):
                row[column] = values[idx]
        rows.append(row)
    return rows


def _row_features(snapshot: dict[str, Any], history: dict[str, Any]) -> dict[str, Any]:
    features: dict[str, Any] = {}
    for key in PRESCANNED_ROW_FEATURES:
        value = _first_present(history.get(key), snapshot.get(key))
        if value is not None:
            features[_canonical_feature_name(key)] = value
    if "u1_given_u2" in features and "u2_given_u1" in features:
        try:
            features["conditional_probability_distortion"] = float(features["u1_given_u2"]) - float(features["u2_given_u1"])
        except (TypeError, ValueError):
            pass
    if "cvar" in features:
        features["tail_dependence"] = abs(_safe_float(features["cvar"], 0.0))
    if "sharpe" in features or "cvar" in features or "drawdown" in features:
        features["copula_calibration_score"] = _safe_float(snapshot.get("ml_confidence"), 0.5)
    return features


def _canonical_feature_name(key: str) -> str:
    return {
        "closed": "completed_trades",
        "mdd": "drawdown",
        "beta_lt": "beta",
    }.get(key, key)


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_filename(value: object) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(value))
