from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from quant_platform.orchestration.corrective_wizard_browser_auth import (
    validate_wizard_browser_auth_readiness,
)


ROOT = Path(__file__).resolve().parents[1]
ACTIVE = ROOT / "reports" / "active"
RAW = ROOT / "data" / "raw" / "crypto_wizards_scanner"


def _load_env() -> None:
    env_path = ROOT / ".env.local"
    if not env_path.exists():
        return
    for line in env_path.read_text(errors="ignore").splitlines():
        if not line.strip() or line.strip().startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key] = value.strip().strip("\"'")


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _candidate_key(row: dict) -> str:
    def key_part(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value).strip().lower()

    return (
        f"{key_part(row.get('_strategy_request', ''))}::{key_part(row.get('symbol_1', ''))}/"
        f"{key_part(row.get('symbol_2', ''))}::{key_part(row.get('exchange', ''))}::"
        f"{key_part(row.get('interval', ''))}::{key_part(row.get('period', ''))}::"
        f"{key_part(row.get('spread_type', ''))}::{key_part(row.get('strategy', ''))}"
    )


def _norm(value: object) -> object:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return round(value, 10)
    return value


def _row_template(columns: list[str], **kwargs: object) -> dict:
    row = {column: "" for column in columns}
    row.update(kwargs)
    return row


def _fetch_scanner(timestamp_slug: str) -> tuple[list[dict], list[dict], Path]:
    RAW.mkdir(parents=True, exist_ok=True)
    base = os.environ.get("CRYPTO_WIZARDS_BASE_URL", "https://api.cryptowizards.net").rstrip("/")
    api_key = os.environ.get("CRYPTO_WIZARDS_API_KEY", "")
    rows: list[dict] = []
    statuses: list[dict] = []
    for strategy in ["Spread", "Copula", "ZScoreRoll"]:
        query = urllib.parse.urlencode(
            {
                "priority": "Sharpe",
                "strategy": strategy,
                "exchange": "Dydx",
                "interval": "Daily",
            }
        )
        url = f"{base}/v1beta/prescanned?{query}"
        request = urllib.request.Request(
            url,
            headers={
                "X-api-key": api_key,
                "Accept": "application/json",
                "User-Agent": "Codex-live-journal/1.0",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                body = response.read()
                payload = json.loads(body)
                items = payload.get("results") or payload.get("data") if isinstance(payload, dict) else payload
                items = items or []
                for item in items:
                    if isinstance(item, dict):
                        record = dict(item)
                        record["_strategy_request"] = strategy
                        record["_row_index"] = len(rows)
                        rows.append(record)
                statuses.append(
                    {
                        "strategy": strategy,
                        "url": url,
                        "status": "ok",
                        "rows_returned": len(items),
                        "status_code": response.status,
                        "sample_chars": body[:120].decode("utf-8", "replace"),
                    }
                )
        except urllib.error.HTTPError as exc:
            statuses.append(
                {
                    "strategy": strategy,
                    "url": url,
                    "status": "error",
                    "rows_returned": 0,
                    "status_code": exc.code,
                    "error": repr(exc),
                    "sample_chars": exc.read(200).decode("utf-8", "replace"),
                }
            )
        except Exception as exc:  # noqa: BLE001 - journal the blocker instead of failing silently.
            statuses.append(
                {
                    "strategy": strategy,
                    "url": url,
                    "status": "error",
                    "rows_returned": 0,
                    "status_code": None,
                    "error": repr(exc),
                    "sample_chars": "",
                }
            )
    capture_path = RAW / f"cw_api_daily_dydx_{timestamp_slug}_merged.json"
    capture_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    return rows, statuses, capture_path


def _probe_routes(timestamp_utc: str, api_rows: list[dict]) -> list[dict]:
    route_ids = [str(api_rows[0].get("pair_id"))] if api_rows else []
    for fallback in ["9020", "9002"]:
        if fallback not in route_ids:
            route_ids.append(fallback)
    routes = ["https://cryptowizards.net/wizards/zscore/scanner"]
    routes += [f"https://cryptowizards.net/wizards/zscore/pair/{pair_id}?origin=scanner" for pair_id in route_ids[:3]]
    probes = []
    for route in routes:
        request = urllib.request.Request(route, headers={"User-Agent": "Codex-live-journal/1.0"})
        try:
            redirect_chain = []
            for _ in range(5):
                try:
                    response = urllib.request.urlopen(request, timeout=20)
                    break
                except urllib.error.HTTPError as exc:
                    if exc.code != 308:
                        raise
                    location = exc.headers.get("location")
                    if not location:
                        raise
                    next_url = urllib.parse.urljoin(request.full_url, location)
                    redirect_chain.append({"status_code": exc.code, "location": next_url})
                    request = urllib.request.Request(
                        next_url,
                        headers={"User-Agent": "Codex-live-journal/1.0"},
                    )
            else:
                raise RuntimeError("too_many_redirects")
            with response:
                body = response.read(2500).decode("utf-8", "replace")
                lower = body.lower()
                title = ""
                if "<title" in lower and "</title>" in lower:
                    title = body[lower.find("<title") : lower.find("</title>")].split(">", 1)[-1].strip()[:120]
                signin = "signin" in response.geturl().lower() or "authentication/signin" in lower or "signin" in title.lower()
                content_heuristic = "sharpe" in lower or "zscore" in lower or "cointegration" in lower
                probes.append(
                    {
                        "route": route,
                        "timestamp_utc": timestamp_utc,
                        "status_code": response.status,
                        "redirect_chain": redirect_chain,
                        "content_type": response.headers.get("content-type", ""),
                        "final_url": response.geturl(),
                        "title": title,
                        "visible_auth_signal": (
                            "signin_or_public_homepage"
                            if signin
                            else "uncredentialed_http_probe_no_auth_authority"
                        ),
                        "protected_content_heuristic_only": content_heuristic,
                        "authenticated_rows_exposed": False,
                        "authenticated_pair_fields_exposed": False,
                        "authentication_authority": False,
                        "body_probe_chars": body[:2200],
                    }
                )
        except Exception as exc:  # noqa: BLE001
            probes.append(
                {
                    "route": route,
                    "timestamp_utc": timestamp_utc,
                    "status_code": None,
                    "content_type": "",
                    "final_url": route,
                    "title": "",
                    "visible_auth_signal": "probe_error",
                    "authenticated_rows_exposed": False,
                    "authenticated_pair_fields_exposed": False,
                    "error": repr(exc),
                    "body_probe_chars": "",
                }
            )
    return probes


def _candidate_snapshot(row: dict) -> dict:
    return {
        "api_row_index": row.get("_row_index"),
        "api_strategy_request": row.get("_strategy_request"),
        "pair_id": row.get("pair_id"),
        "spread_id": row.get("spread_id"),
        "strategy_id": row.get("strategy_id"),
        "exchange": row.get("exchange"),
        "interval": row.get("interval"),
        "period": row.get("period"),
        "symbol_1": row.get("symbol_1"),
        "symbol_2": row.get("symbol_2"),
        "volume_x": row.get("sym_1_volume"),
        "volume_y": row.get("sym_2_volume"),
        "return_total": row.get("returns_total"),
        "sharpe": row.get("sharpe"),
        "mdd": row.get("mdd"),
        "var": row.get("var"),
        "cvar": row.get("cvar"),
        "zscore_last": row.get("zscore_last"),
        "zscore_roll_last": row.get("zscore_roll_last"),
        "hurst": row.get("hurst"),
        "half_life": row.get("half_life"),
        "corr_copula": row.get("corr_copula"),
        "johansen_coint": row.get("johansen_coint"),
        "coint_eg": row.get("coint_eg"),
        "u1_given_u2": row.get("u1_given_u2"),
        "u2_given_u1": row.get("u2_given_u1"),
    }


def _candidate_status(row: dict) -> tuple[str, str]:
    strategy = str(row.get("strategy") or row.get("_strategy_request") or "").lower()
    if row.get("_strategy_request") == "Copula" or "copula" in strategy:
        family = "copula"
    elif row.get("_strategy_request") == "ZScoreRoll" or "zscore" in strategy:
        family = "zscore"
    else:
        family = "spread"
    mode = row.get("_strategy_request") or row.get("strategy") or ""
    if row.get("spread_type"):
        mode = f"{row.get('spread_type')} {mode}".strip()
    return family, mode


def _timeframe(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip().lower()
    if text == "hourly":
        return "Hourly"
    if text == "daily":
        return "Daily"
    return str(value or "Daily")


def _strategy_label(row: dict) -> str:
    spread_type = str(row.get("spread_type") or "").strip().lower()
    strategy = str(row.get("_strategy_request") or row.get("strategy") or "").strip()
    if strategy.lower() == "copula":
        return "Copula"
    strategy_label = "ZScoreR" if strategy.lower() == "zscoreroll" else strategy
    prefix = {"static": "Static", "dynamic": "Dyn", "ou": "OU"}.get(spread_type, spread_type.title() if spread_type else "")
    if prefix:
        return f"{prefix} ({strategy_label})"
    return strategy_label


def _documented_entry_exit_defaults(strategy: object) -> tuple[float, float]:
    """Defaults published by the Wizard GET /v1beta/backtest documentation."""

    normalized = str(strategy or "").strip().lower()
    if normalized == "copula":
        return 0.05, 0.50
    if normalized == "zscoreroll":
        return 1.50, 0.0
    return 2.0, 0.0


def _return_pct(value: object) -> object:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return ""
    return parsed * 100.0 if abs(parsed) <= 1.0 else parsed


def _badge(value: object) -> str:
    if isinstance(value, bool):
        return "confirmed" if value else "not_confirmed"
    text = "" if pd.isna(value) else str(value).strip().lower()
    if text in {"true", "1", "yes", "confirmed", "pass"}:
        return "confirmed"
    if text in {"trend", "trending"}:
        return "trending"
    if text in {"false", "0", "no", "not_confirmed", "fail"}:
        return "not_confirmed"
    return "unknown" if not text else text


def _write_active_scanner_reports(api_rows: list[dict], timestamp_utc: str, raw_api_capture: Path) -> dict[str, object]:
    live_path = ACTIVE / "crypto_wizards_live_scanner_capture.csv"
    top_path = ACTIVE / "wizard_strategy_top_five_live.csv"
    top_json_path = ACTIVE / "wizard_strategy_top_five_live.json"
    prior_live = pd.read_csv(live_path) if live_path.exists() else pd.DataFrame()
    prior_by_pair_strategy = {}
    if not prior_live.empty:
        for row in prior_live.fillna("").to_dict("records"):
            key = (str(row.get("pair", "")), str(row.get("api_strategy") or row.get("scanner_strategy_filter") or ""))
            prior_by_pair_strategy[key] = row

    rows: list[dict[str, object]] = []
    for rank, row in enumerate(api_rows, start=1):
        pair = f"{row.get('symbol_1', '')}/{row.get('symbol_2', '')}"
        strategy = str(row.get("_strategy_request") or row.get("strategy") or "")
        exact_mode = _strategy_label(row)
        entry_level, exit_level = _documented_entry_exit_defaults(strategy)
        key = (pair, strategy)
        prior = prior_by_pair_strategy.get(key, {})
        changed_vs_previous = "new" if not prior else "unchanged"
        if prior:
            for field, source in {
                "sharpe": "sharpe",
                "returns_total_pct": "returns_total",
                "zscore_norm": "zscore_last",
                "zscore_roll": "zscore_roll_last",
            }.items():
                if _norm(prior.get(field, "")) != _norm(row.get(source, "")):
                    changed_vs_previous = "changed"
                    break
        rows.append(
            {
                "pair": pair,
                "asset_x": row.get("symbol_1", ""),
                "asset_y": row.get("symbol_2", ""),
                "pair_id": row.get("pair_id", ""),
                "spread_id": row.get("spread_id", ""),
                "strategy_id": row.get("strategy_id", ""),
                "exchange": str(row.get("exchange") or "").lower(),
                "interval": _timeframe(row.get("interval")),
                "timeframe": _timeframe(row.get("interval")),
                "period": row.get("period", ""),
                "dashboard_recommended_strategy": exact_mode,
                "exact_mode": exact_mode,
                "dashboard_pair_rank": rank,
                "returns_total": row.get("returns_total"),
                "returns_total_pct": _return_pct(row.get("returns_total")),
                "sharpe": row.get("sharpe"),
                "source_path": "reports/active/crypto_wizards_live_scanner_capture.csv",
                "raw_source_path": str(raw_api_capture.relative_to(ROOT)),
                "evidence_path": str(raw_api_capture.relative_to(ROOT)),
                "capture_context": "official_prescanned_api_plus_documented_backtest_defaults",
                "capture_timestamp_utc": timestamp_utc,
                "source_timestamp": timestamp_utc,
                "scan_timestamp": timestamp_utc,
                "backtest_ts": row.get("backtest_ts", ""),
                "spread_type": row.get("spread_type", ""),
                "x_weighting": row.get("x_weighting", ""),
                "y_weighting": row.get("y_weighting", ""),
                "zscore_window": row.get("zscore_window", ""),
                "roll_w": row.get("zscore_window", ""),
                "entry_level": entry_level,
                "exit_level": exit_level,
                "slippage_rate": "",
                "commission_rate": "",
                "entry_exit_settings_source": "documented_api_defaults",
                "weighting_window_source": "current_prescanned_row",
                "cost_settings_source": "not_documented_by_wizard",
                "api_docs_url": "https://api.cryptowizards.net/docsv1beta/backtest-get.mdx/",
                "volume_x": row.get("sym_1_volume"),
                "volume_y": row.get("sym_2_volume"),
                "updated_at_utc": timestamp_utc,
                "zscore_norm": row.get("zscore_last"),
                "zscore_roll": row.get("zscore_roll_last"),
                "dependency_profile": row.get("profile_match"),
                "dependency_x_over_y": row.get("u1_given_u2"),
                "dependency_y_over_x": row.get("u2_given_u1"),
                "correlation": row.get("corr_copula"),
                "jn_flag": row.get("johansen_coint"),
                "eg_flag": row.get("coint_eg"),
                "johansen_badge_state": _badge(row.get("johansen_coint")),
                "engle_granger_badge_state": _badge(row.get("coint_eg")),
                "hurst": row.get("hurst"),
                "half_life": row.get("half_life"),
                "sigma_0_count": "",
                "sigma_1_count": "",
                "sigma_2_count": "",
                "var": row.get("var"),
                "cvar": row.get("cvar"),
                "mdd": row.get("mdd"),
                "scanner_sort_mode": "Sharpe (highest)",
                "scanner_cointegration_filter": "Coint (all cases)",
                "scanner_correlation_filter": "Correl (all cases)",
                "scanner_hurst_filter": "Hurst (all cases)",
                "scanner_half_life_filter": "Halflife (all cases)",
                "scanner_copula_filter": "all cases",
                "scanner_strategy_filter": strategy,
                "scanner_exchange_filter": str(row.get("exchange") or "").upper(),
                "visible_row_count": len(api_rows),
                "api_strategy": strategy,
                "api_rank": rank,
                "changed_vs_previous_journal": changed_vs_previous,
                "previous_sharpe": prior.get("sharpe", ""),
                "previous_return_total_pct": prior.get("returns_total_pct", ""),
            }
        )

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame.to_csv(live_path, index=False)
        top = frame.sort_values("sharpe", ascending=False, na_position="last").head(5)
        top.to_csv(top_path, index=False)
        top_json_path.write_text(
            json.dumps(
                {
                    "timestamp_utc": timestamp_utc,
                    "source": str(raw_api_capture),
                    "rows": top.to_dict("records"),
                },
                indent=2,
                sort_keys=True,
                default=str,
            ),
            encoding="utf-8",
        )
    return {
        "live_scanner_capture": str(live_path),
        "top_five": str(top_path),
        "top_five_json": str(top_json_path),
        "active_scanner_rows_written": int(len(frame)),
        "top_five_rows_written": int(min(len(frame), 5)),
    }


def main() -> None:
    _load_env()
    ACTIVE.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).replace(microsecond=0)
    timestamp_utc = now.isoformat()
    timestamp_slug = now.strftime("%Y%m%dT%H%M%S+0000")

    api_rows, fetch_status, raw_api_capture = _fetch_scanner(timestamp_slug)
    route_probes = _probe_routes(timestamp_utc, api_rows)
    browser_auth_readiness = validate_wizard_browser_auth_readiness(root=ROOT, now=now)
    browser_auth_gated = browser_auth_readiness.get("status") != "PASS"
    proven_routes = set(browser_auth_readiness.get("route_kinds", []))
    pair_fields_exposed = not browser_auth_gated and "pair_detail" in proven_routes
    scanner_rows_exposed = not browser_auth_gated and "scanner" in proven_routes
    pair_detail_blocked = not pair_fields_exposed

    latest_capture = ACTIVE / "wizard_live_browser_capture_latest.json"
    timed_capture = ACTIVE / f"wizard_live_browser_capture_{timestamp_utc.replace(':', '-').replace('+00:00', '+00-00')}.json"
    capture_payload = {
        "timestamp_utc": timestamp_utc,
        "mode": "live_journaling_only",
        "dydx_account_state_execution": "not_used_unreliable",
        "api_rows": len(api_rows),
        "fetch_status": fetch_status,
        "raw_api_capture": str(raw_api_capture),
        "route_probes": route_probes,
        "browser_auth_readiness": browser_auth_readiness,
        "browser_auth_gated": browser_auth_gated,
        "authenticated_scanner_rows_exposed": scanner_rows_exposed,
        "authenticated_pair_detail_fields_exposed": pair_fields_exposed,
    }
    capture_text = json.dumps(capture_payload, indent=2, sort_keys=True)
    latest_capture.write_text(capture_text, encoding="utf-8")
    timed_capture.write_text(capture_text, encoding="utf-8")

    watch_path = ACTIVE / "wizard_live_api_watch_candidates.csv"
    prior_watch = pd.read_csv(watch_path) if watch_path.exists() else pd.DataFrame()
    current_watch = pd.DataFrame(api_rows)
    if not current_watch.empty:
        columns = list(prior_watch.columns)
        for column in current_watch.columns:
            if column not in columns:
                columns.append(column)
        current_watch = current_watch.reindex(columns=columns)
        current_watch.to_csv(watch_path, index=False)
    active_scanner_report = _write_active_scanner_reports(api_rows, timestamp_utc, raw_api_capture) if api_rows else {}

    prior_by_key = {_candidate_key(row): row for row in prior_watch.fillna("").to_dict("records")} if not prior_watch.empty else {}
    current_by_key = {_candidate_key(row): row for row in current_watch.fillna("").to_dict("records")} if not current_watch.empty else {}
    new_keys = set(current_by_key) - set(prior_by_key)
    demoted_keys = set(prior_by_key) - set(current_by_key)
    common_keys = set(current_by_key) & set(prior_by_key)
    compare_fields = [
        "sharpe",
        "returns_total",
        "mdd",
        "var",
        "cvar",
        "half_life",
        "hurst",
        "corr_copula",
        "zscore_last",
        "zscore_roll_last",
        "u1_given_u2",
        "u2_given_u1",
        "symbol_1",
        "symbol_2",
        "exchange",
        "interval",
        "period",
        "strategy",
        "spread_type",
        "johansen_coint",
        "coint_eg",
        "closed",
        "win_rate",
    ]
    changed = {}
    unchanged = []
    for key in common_keys:
        fields = [field for field in compare_fields if _norm(prior_by_key[key].get(field, "")) != _norm(current_by_key[key].get(field, ""))]
        if fields:
            changed[key] = fields
        else:
            unchanged.append(key)

    events_path = ACTIVE / "wizard_live_journal_events.csv"
    events = pd.read_csv(events_path)
    columns = list(events.columns)
    source_files = ";".join(
        str(path)
        for path in [
            ACTIVE / "live_paper_trade_monitor.csv",
            ACTIVE / "wizard_research_scanner_capture.csv",
            ACTIVE / "wizard_research_pair_detail_capture.csv",
            latest_capture,
            raw_api_capture,
        ]
    )
    new_event_rows = []
    prior_timestamp = events["timestamp_utc"].iloc[-1]
    prior_events = events[events["timestamp_utc"].eq(prior_timestamp)].set_index("trade_id", drop=False)

    monitor = pd.read_csv(ACTIVE / "live_paper_trade_monitor.csv")
    for record in monitor.fillna("").to_dict("records"):
        status = "exit-candidate" if str(record.get("recommendation", "")).lower() == "close" else str(record.get("recommendation") or record.get("monitor_status") or "hold")
        trade_id = record.get("trade_id")
        old_status = prior_events.loc[trade_id, "journal_status"] if trade_id in prior_events.index else ""
        changed_fields = [] if old_status == status else ["journal_status"]
        try:
            page_one = json.loads(record.get("entry_snapshot_json") or "{}")
        except json.JSONDecodeError:
            page_one = {"entry_snapshot_json_parse_error": True}
        page_one.update(
            {
                "monitor_status": record.get("monitor_status"),
                "recommendation": record.get("recommendation"),
                "latest_dashboard_capture_timestamp_utc": record.get("latest_dashboard_capture_timestamp_utc"),
            }
        )
        new_event_rows.append(
            _row_template(
                columns,
                timestamp_utc=timestamp_utc,
                event_type="journal_event" if changed_fields else "checked_no_material_change",
                journal_status=status,
                pair=record.get("pair"),
                trade_id=trade_id,
                strategy_family="OU" if "OU" in str(page_one.get("dashboard_strategy", "")) else str(page_one.get("dashboard_strategy") or record.get("strategy_id")),
                strategy_mode=str(page_one.get("dashboard_strategy") or record.get("latest_dashboard_strategy") or ""),
                setup_identity=f"{record.get('pair')}::{page_one.get('dashboard_strategy') or record.get('latest_dashboard_strategy')}::{record.get('strategy_id')}",
                timeframe=str(record.get("latest_dashboard_timeframe") or "Live"),
                zscore_mode=record.get("current_zscore_source"),
                current_zscore_value=record.get("current_zscore_roll") if record.get("current_zscore_source") == "rolling" else record.get("current_zscore_norm"),
                spread_state=record.get("monitor_status"),
                dependency_metrics=_json({"correlation": record.get("current_correlation"), "half_life": record.get("current_half_life"), "hurst": record.get("current_hurst")}),
                stationarity_correlation_test_type=_json({"correlation": record.get("current_correlation"), "johansen": record.get("coint_johansen_top"), "engle_granger": record.get("coint_engle_granger_top")}),
                volume=_json({"x": record.get("volume_x_top"), "y": record.get("volume_y_top")}),
                risk=_json({"max_drawdown": record.get("current_max_drawdown")}),
                reward=_json({"return_total": record.get("current_return_total")}),
                sharpe=record.get("current_sharpe"),
                entry_price_x=record.get("venue_entry_price_x") or record.get("wizard_entry_price_x"),
                entry_price_y=record.get("venue_entry_price_y") or record.get("wizard_entry_price_y"),
                current_price_x=record.get("price_x") or record.get("venue_entry_price_x"),
                current_price_y=record.get("price_y") or record.get("venue_entry_price_y"),
                page_one_snapshot=_json(page_one),
                page_two_snapshot=_json(
                    {
                    "authenticated_pair_detail_fields_exposed": pair_fields_exposed,
                    "fresh_pair_detail_refresh": "not_available_auth_gated" if pair_detail_blocked else "route_probe_only",
                        "latest_saved_capture_timestamp_utc": record.get("latest_dashboard_capture_timestamp_utc"),
                    }
                ),
                route_execution_notes="live_journaling_only; dYdX API/account-state execution unreliable and not used; no exchange action submitted; current Wizard browser routes probed.",
                reason_for_status_change="position status changed versus prior journal pass" if changed_fields else "no material change versus prior journal pass; fresh pair-detail fields unavailable due browser auth gate",
                exit_conditions="close recommendation marks exit-candidate; otherwise hold until authenticated scanner/pair-detail refresh changes metrics or status",
                prior_value=_json({"journal_status": old_status} if changed_fields else {}),
                new_value=_json({"journal_status": status} if changed_fields else {}),
                changed_fields=";".join(changed_fields) if changed_fields else "none",
                summary_note=f"{record.get('pair')} remains {status}; saved monitor baseline unchanged; page-two refresh {'blocked' if pair_detail_blocked else 'probed'}.",
                source_files=source_files,
            )
        )

    def add_candidate(row: dict, event_type: str, status: str, reason: str, prior: dict | None, fields: list[str]) -> None:
        family, mode = _candidate_status(row)
        pair = f"{row.get('symbol_1', '')}/{row.get('symbol_2', '')}"
        snapshot = _candidate_snapshot(row)
        new_event_rows.append(
            _row_template(
                columns,
                timestamp_utc=timestamp_utc,
                event_type=event_type,
                journal_status=status,
                pair=pair,
                trade_id=_candidate_key(row),
                strategy_family=family,
                strategy_mode=mode,
                setup_identity=_candidate_key(row),
                timeframe=str(row.get("interval") or "daily"),
                zscore_mode="rolling" if row.get("_strategy_request") == "ZScoreRoll" else "normal",
                current_zscore_value=row.get("zscore_roll_last") if row.get("_strategy_request") == "ZScoreRoll" else row.get("zscore_last"),
                spread_state="api_scanner_active",
                dependency_metrics=_json({"x_given_y": row.get("u1_given_u2"), "y_given_x": row.get("u2_given_u1"), "correlation": row.get("corr_copula"), "half_life": row.get("half_life"), "hurst": row.get("hurst")}),
                stationarity_correlation_test_type=_json({"johansen": row.get("johansen_coint"), "engle_granger": row.get("coint_eg"), "copula": row.get("copula"), "correlation": row.get("corr_copula")}),
                volume=_json({"x": row.get("sym_1_volume"), "y": row.get("sym_2_volume")}),
                risk=_json({"max_drawdown": row.get("mdd"), "var": row.get("var"), "cvar": row.get("cvar")}),
                reward=_json({"return_total": row.get("returns_total"), "win_rate": row.get("win_rate"), "closed": row.get("closed")}),
                sharpe=row.get("sharpe"),
                page_one_snapshot=_json(snapshot),
                page_two_snapshot=_json({"authenticated_pair_detail_fields_exposed": pair_fields_exposed, "fresh_pair_detail_refresh": "not_available_auth_gated" if pair_detail_blocked else "route_probe_only", "route_probe_count": len(route_probes)}),
                route_execution_notes="live_journaling_only; dYdX API/account-state execution unreliable and not used; no exchange action submitted; scanner API refreshed.",
                reason_for_status_change=reason,
                exit_conditions="watch only; requires authenticated pair-detail confirmation and execution-compatible venue/account state before action",
                prior_value=_json(prior or {}),
                new_value=_json(snapshot),
                changed_fields=";".join(fields),
                summary_note=f"{pair} {status}; {reason}",
                source_files=source_files,
            )
        )

    for key in sorted(new_keys):
        add_candidate(current_by_key[key], "journal_event", "watch", "new API watch candidate appeared in refreshed scanner", {}, ["candidate_presence"])
    for key in sorted(changed):
        fields = changed[key]
        add_candidate(current_by_key[key], "journal_event", "watch", "API watch candidate fields changed versus prior baseline", {field: prior_by_key[key].get(field) for field in fields}, fields)
    for key in sorted(unchanged):
        add_candidate(current_by_key[key], "checked_no_material_change", "watch", "API watch candidate unchanged versus prior baseline", {}, ["none"])
    for key in sorted(demoted_keys):
        add_candidate(prior_by_key[key], "journal_event", "closed", "prior API watch candidate absent from refreshed scanner; demoted from active watch baseline", _candidate_snapshot(prior_by_key[key]), ["candidate_presence"])

    summary = {
        "active_api_scanner_rows": len(api_rows),
        "raw_api_rows": len(api_rows),
        "new_api_watch_candidates": len(new_keys),
        "changed_api_watch_candidates": len(changed),
        "unchanged_api_watch_candidates": len(unchanged),
        "demoted_api_watch_candidates": len(demoted_keys),
        "api_fetch_status": fetch_status,
    }
    new_event_rows.append(
        _row_template(
            columns,
            timestamp_utc=timestamp_utc,
            event_type="journal_event" if api_rows else "checked_no_material_change",
            journal_status="scanner_api_recovered" if api_rows else "dashboard_capture_failed",
            pair="SCANNER_API",
            trade_id=f"scanner_api::{timestamp_slug}",
            strategy_family="scanner_api",
            strategy_mode="api_probe",
            setup_identity=f"scanner_api::{timestamp_slug}",
            timeframe="Daily",
            spread_state="api_refreshed" if api_rows else "auth_blocked",
            dependency_metrics=_json({"api_fetch_status": fetch_status}),
            stationarity_correlation_test_type="prescanned API status check",
            volume=_json({}),
            risk=_json({}),
            reward=_json({}),
            page_one_snapshot=_json(summary),
            page_two_snapshot=_json({"authenticated_pair_detail_fields_exposed": pair_fields_exposed, "browser_auth_gated": browser_auth_gated, "route_probe_count": len(route_probes)}),
            route_execution_notes="live_journaling_only; dYdX API/account-state execution unreliable and not used; no exchange action submitted.",
            reason_for_status_change="scanner API recovered with authenticated key; watch baseline refreshed" if api_rows else "scanner API refresh returned zero active rows",
            exit_conditions="do not produce exchange actions or new actionable entries without pair-detail confirmation and account-state reliability",
            prior_value=_json({"last_api_candidate_count": len(prior_watch)}),
            new_value=_json({"current_api_candidate_count": len(api_rows)}),
            changed_fields="scanner_api_recovered;candidate_baseline" if api_rows else "scanner_api_auth",
            summary_note=f"Scanner refresh returned {len(api_rows)} rows; {len(new_keys)} new, {len(changed)} changed, {len(unchanged)} unchanged, {len(demoted_keys)} demoted versus prior baseline.",
            source_files=source_files,
        )
    )
    new_event_rows.append(
        _row_template(
            columns,
            timestamp_utc=timestamp_utc,
            event_type="checked_no_material_change" if pair_detail_blocked else "journal_event",
            journal_status="blocked_pair_detail" if pair_detail_blocked else "pair_detail_probe_ok",
            pair="PAIR_DETAIL_ROUTES",
            trade_id=f"pair_detail_blocker::{timestamp_slug}",
            strategy_family="pair_detail",
            strategy_mode="browser_probe",
            setup_identity="representative_pair_detail_routes",
            timeframe="Daily",
            spread_state="auth_or_field_blocked" if pair_detail_blocked else "route_probe_available",
            dependency_metrics=_json({"route_probes": route_probes}),
            stationarity_correlation_test_type="not_available_auth_gated" if pair_detail_blocked else "route_probe",
            volume=_json({}),
            risk=_json({"blocker": "authenticated_pair_detail_fields_unavailable"} if pair_detail_blocked else {}),
            reward=_json({}),
            page_one_snapshot=_json({"routes_checked": len(route_probes), "scanner_auth_gated": browser_auth_gated}),
            page_two_snapshot=_json({"authenticated_pair_detail_fields_exposed": pair_fields_exposed, "browser_auth_gated": browser_auth_gated, "route_probe_count": len(route_probes)}),
            route_execution_notes="live_journaling_only; dYdX API/account-state execution unreliable and not used; no exchange action submitted; browser dashboard routes probed.",
            reason_for_status_change="persistent blocker: authenticated page-two fields unavailable" if pair_detail_blocked else "browser routes exposed authenticated pair-detail fields",
            exit_conditions="refresh browser authentication before treating page-two route fields as current" if pair_detail_blocked else "confirm pair detail before execution",
            prior_value=_json({"blocker_state": "auth_gated"}),
            new_value=_json({"blocker_state": "pair_detail_fields_unavailable" if pair_detail_blocked else "route_probe_available"}),
            changed_fields="none" if pair_detail_blocked else "pair_detail_route_access",
            summary_note="Representative current and monitored pair-detail routes did not expose authenticated page-two fields; page-two fields retained from saved/API snapshots." if pair_detail_blocked else "Representative pair-detail routes exposed authenticated fields.",
            source_files=source_files,
        )
    )

    new_events = pd.DataFrame(new_event_rows).reindex(columns=columns)
    new_events.to_csv(events_path, mode="a", header=False, index=False)

    status_counts = pd.concat([events, new_events])["journal_status"].value_counts(dropna=False).to_dict()
    hold_count = int((monitor["recommendation"].astype(str).str.lower() == "hold").sum())
    exit_count = int((monitor["recommendation"].astype(str).str.lower() == "close").sum())
    (ACTIVE / "wizard_live_journal_status.md").write_text(
        "\n".join(
            [
                "# Wizard Live Journal Status",
                "",
                f"- timestamp_utc: {timestamp_utc}",
                "- mode: live_journaling_only",
                "- dYdX API/account-state execution: not_used_dydx_api_account_state_unreliable",
                f"- scanner_refresh: {'api_refreshed' if api_rows else 'dashboard_capture_failed'}; active_api_scanner_rows={len(api_rows)}; raw_api_rows={len(api_rows)}; new_api_watch_candidates={len(new_keys)}; changed_api_watch_candidates={len(changed)}; unchanged_api_watch_candidates={len(unchanged)}; demoted_api_watch_candidates={len(demoted_keys)}",
                f"- pair_detail_refresh: auth_gated_pair_detail_routes={sum(1 for probe in route_probes if probe.get('visible_auth_signal') == 'signin_or_public_homepage')}; browser_auth_gated_public_signin={browser_auth_gated}; authenticated_pair_detail_fields_exposed={pair_fields_exposed}; pair_detail_blocked={pair_detail_blocked}",
                f"- tracked_positions_checked: {len(monitor)}; hold={hold_count}; exit_candidate={exit_count}",
                f"- tracked_watch_candidates_checked: {len(api_rows)}",
                f"- tracked_status_counts: {json.dumps(status_counts, sort_keys=True)}",
                f"- event_rows_appended: {len(new_event_rows)}",
                f"- capture: {latest_capture}",
                f"- raw_api_capture: {raw_api_capture}",
                f"- active_scanner_capture: {active_scanner_report.get('live_scanner_capture', '')}",
                f"- top_five_live: {active_scanner_report.get('top_five', '')}",
                f"- state_change_notice: candidate_promoted={len(new_keys)}; candidate_changed={len(changed)}; candidate_demoted={len(demoted_keys)}; position_state_changed=0; metric_field_changed={len(changed)}; browser_auth_blocker_persisted={pair_detail_blocked}; scanner_api_auth_blocker={not bool(api_rows)}",
                f"- blocker: {'Crypto Wizards pair-detail routes did not expose authenticated page-two fields; API scanner data was refreshed successfully.' if api_rows else 'Crypto Wizards scanner API and browser routes blocked; current page-one/page-two fields unavailable until authentication is refreshed.'}",
                "",
                "# Current Scanner Recovery",
                "",
                f"Crypto Wizards API refresh returned {len(api_rows)} rows across Spread, Copula, and ZScoreRoll. Watch baseline changes versus the prior file: {len(new_keys)} new, {len(changed)} changed, {len(unchanged)} unchanged, {len(demoted_keys)} demoted. Browser scanner/pair-detail routes remain auth-gated, so detailed page-two fields are not current.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (ACTIVE / "wizard_live_scanner_refresh_summary.json").write_text(
        json.dumps(
            {
                "timestamp_utc": timestamp_utc,
                "scanner_refresh": "api_refreshed" if api_rows else "dashboard_capture_failed",
                "api_active_scanner_rows": len(api_rows),
                "api_fetch_status": fetch_status,
                "browser_auth_gated": browser_auth_gated,
                "authenticated_pair_detail_fields_exposed": pair_fields_exposed,
                "pair_detail_blocked": pair_detail_blocked,
                "latest_browser_capture": str(latest_capture),
                "raw_api_capture": str(raw_api_capture),
                "active_scanner_report": active_scanner_report,
                "new_api_watch_candidates": len(new_keys),
                "changed_api_watch_candidates": len(changed),
                "unchanged_api_watch_candidates": len(unchanged),
                "demoted_api_watch_candidates": len(demoted_keys),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    notice_path = ACTIVE / "wizard_live_state_change_notice.csv"
    notice_row = pd.DataFrame(
        [
            {
                "timestamp_utc": timestamp_utc,
                "api_watch_promoted": len(new_keys),
                "api_watch_demoted": len(demoted_keys),
                "api_watch_metric_changed": len(changed),
                "positions_hold": hold_count,
                "positions_exit_candidate": exit_count,
                "pair_detail_blocked": pair_detail_blocked,
                "status_counts": str(
                    {
                        "watch": len(api_rows),
                        "closed": len(demoted_keys),
                        "hold": hold_count,
                        "exit-candidate": exit_count,
                        "scanner_api_recovered": int(bool(api_rows)),
                        "blocked_pair_detail": int(pair_detail_blocked),
                    }
                ),
            }
        ]
    )
    notice_row.to_csv(
        notice_path,
        mode="a",
        header=not notice_path.exists() or notice_path.stat().st_size == 0,
        index=False,
    )
    print(
        json.dumps(
            {
                "timestamp_utc": timestamp_utc,
                "events_appended": len(new_event_rows),
                "api_rows": len(api_rows),
                "new": len(new_keys),
                "changed": len(changed),
                "unchanged": len(unchanged),
                "demoted": len(demoted_keys),
                "browser_auth_gated": browser_auth_gated,
                "auth_pair_exposed": pair_fields_exposed,
                "raw_api_capture": str(raw_api_capture),
                "active_scanner_rows_written": active_scanner_report.get("active_scanner_rows_written", 0),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
