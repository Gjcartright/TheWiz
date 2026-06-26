from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DOCS = ROOT / "docs"


def yes(value: bool) -> str:
    return "yes" if value else "no"


def row(
    *,
    area: str,
    url: str,
    fields: str,
    controls: str,
    point_in_time: str,
    export: str,
    pair_selection: str,
    strategy_matching: str,
    exits: str,
    regimes: str,
    risk: str,
    paper: str,
    ingested: str,
    missing: str,
    priority: str,
    evidence: str,
    feature_rank: str | None = None,
) -> dict[str, str]:
    return {
        "page_or_section_name": area,
        "url_if_available": url,
        "fields_or_metrics_exposed": fields,
        "filters_or_controls": controls,
        "point_in_time_or_hindsight": point_in_time,
        "export_or_capture": export,
        "helps_pair_selection": pair_selection,
        "helps_strategy_matching": strategy_matching,
        "helps_exits": exits,
        "helps_regimes": regimes,
        "helps_risk_control": risk,
        "helps_paper_trading": paper,
        "already_ingested": ingested,
        "missing_from_current_repo": missing,
        "feature_rank": feature_rank or priority,
        "recommended_integration_priority": priority,
        "evidence_source": evidence,
    }


def build_inventory() -> list[dict[str, str]]:
    pair_url = "https://cryptowizards.net/wizards/zscore/pair/1?origin=scanner"
    scanner_url = "https://cryptowizards.net/wizards/zscore/scanner"
    api_url = "https://api.cryptowizards.net"
    rows = [
        row(
            area="Scanner page / prescanned pair table",
            url=scanner_url,
            fields="pair_id, spread_id, symbols, exchange, interval, period, volume, strategy, spread_type, zscore_last, zscore_roll_last, Sharpe, returns, win_rate, closed trades, MDD, VaR, CVaR, cointegration flags, Hurst, half_life, hedge_ratio, copula, conditional probabilities, ML confidence, mini_zscore",
            controls="priority/sort: returns, Sharpe, spread high/low/high-Sharpe, zscore high/low/high-Sharpe, most recent; cointegration: Johansen, EG, EG excluding trend, both/either/all; correlation abs 75+, abs 50+, all; Hurst <0.50/<0.65/<0.75/<0.85/<1.00/all; half-life <10/<25/<50/all; copula arbitrage/any; strategy static/dynamic/OU/copula/all; symbol search; exchange: Binance, Binance US, ByBit, Coinbase, DYDX, Forex, Stocks/Other",
            point_in_time="mixed: scanner records are current as of backtest_ts but ranked by prior backtest outcomes; safe for discovery, risky for live signal generation without timestamped snapshots",
            export="API /v1beta/prescanned, browser helper scripts/capture_crypto_wizards_scanner_rows.js, and live browser row capture",
            pair_selection="Primary broad pair discovery and ranking surface.",
            strategy_matching="Initial clues for z-score, OU, copula, stationarity, and profile-match suitability.",
            exits="Closed trade count, win rate, MDD, VaR/CVaR flag fragile exit behavior but do not expose trade path.",
            regimes="Can be stratified by interval, period, copula, Hurst, correlation, and volatility fields.",
            risk="MDD, VaR, CVaR, volume, volatility, closed trades.",
            paper="Useful preflight source only after revalidation on local point-in-time data.",
            ingested="partially: data/raw/prescanned.json and scanner capture scripts exist; local evidence pipeline uses separate dYdX histories",
            missing="Systematic timestamped scanner snapshots by filter set; direct link map from scanner rows to pair-detail captures; live export audit for all filters.",
            priority="must_integrate",
            evidence="live browser observation 2026-06-22; data/raw/prescanned.json; docs/crypto_wizards_api_reference.md; scripts/capture_crypto_wizards_scanner_rows.js",
        ),
        row(
            area="Scanner selected pair inline detail panel",
            url=scanner_url,
            fields="selected asset X/Y, cumulative returns, SPREAD, ZSCORE (Rolling), spread/backtest view selector, strategy comparison summaries for DynSpread, DynZScoreR, OUSpread, OUZScoreR, StaticSpread, StaticZScoreR, Copula with return and Sharpe",
            controls="click scanner row card; asset X/Y selector buttons; SPREAD and ZSCORE (Rolling) toggles; spread/backtest select; unlabeled icon buttons",
            point_in_time="mixed: selected row and chart context are current page state, but strategy return/Sharpe summaries are historical backtest diagnostics and must be treated as hindsight until replayed locally",
            export="live browser capture only so far; no visible export in inline panel; raw chart/state payload still needs authenticated network capture",
            pair_selection="Fast sanity check for whether one pair has several plausible strategy families or only one fragile result.",
            strategy_matching="Directly compares static, dynamic, OU, z-score, and copula summaries on the selected pair.",
            exits="Limited: inline view did not expose threshold, trade path, entries, underwater, or hard-exit controls in the live session.",
            regimes="Limited: row dependency/stationarity context can seed regime follow-up, but panel did not expose regime controls.",
            risk="Shows return/Sharpe summaries and row-level MDD context, but not enough for risk approval.",
            paper="Useful preflight triage only; every result must be replayed with local point-in-time data.",
            ingested="no: selected inline panel state and strategy comparison table are not systematically ingested",
            missing="Raw selected-detail state, strategy comparison schema, chart arrays, and route/action that opens any deeper pair detail surface.",
            priority="must_integrate",
            evidence="live browser observation 2026-06-22: selecting EUR-USD/FIL-USD expanded inline detail on scanner",
            feature_rank="risky_or_hindsight",
        ),
        row(
            area="Pair detail top summary header",
            url=pair_url,
            fields="pair id, asset X/Y, exchange, period count, cointegration badges, Hurst, correlation, hedge ratio, zero-cross and sigma-cross counts, returns, Sharpe",
            controls="pair selector button, New button, symbol inputs, period input, timeframe select, strategy select, refresh/recalculate icon buttons",
            point_in_time="mostly point-in-time if captured after calculation with timestamp; returns/Sharpe/MDD are historical backtest summaries and must not drive live entries directly; the old direct route redirected to the marketing root in the 2026-06-22 live session",
            export="Browser read-only capture and pair-detail helper from older route; raw arrays and current route/action still need full authenticated capture.",
            pair_selection="Confirms whether scanner candidate has enough stationarity/dependence to research.",
            strategy_matching="Maps pair to mean-reversion, OU, copula, or ECM hypotheses.",
            exits="Half-life and crossing counts inform timeout and normalization exits.",
            regimes="Correlation and Hurst can seed range/stability filters.",
            risk="Header return and drawdown stats expose fragility quickly.",
            paper="Good preflight summary, but must be reproduced locally before paper trading.",
            ingested="partially: static pair/1 snapshot and many derived pair detail histories exist",
            missing="Rediscover current live pair-detail route or UI action; authenticated network payload capture for every selected research pair and every required timeframe.",
            priority="must_integrate",
            evidence="data/raw/pair_details/pair_1_dashboard_snapshot.json; data/raw/pair_details/crypto_wizards_pair_1_codex_read_only_capture.json; live browser route check 2026-06-22 redirected /wizards/zscore/pair/1?origin=scanner to /",
        ),
        row(
            area="Pair detail timeframe and period controls",
            url=pair_url,
            fields="Daily, 4 Hour, 1 Hour, 5 Min; period count input",
            controls="timeframe select; numeric period/lookback input",
            point_in_time="point-in-time when recalculated using only historical bars available at capture time; needs timestamped capture discipline",
            export="Capture helper can record selected controls; API endpoints accept interval and period.",
            pair_selection="Separates pairs that only look good at one timeframe from robust pairs.",
            strategy_matching="Aligns half-life and strategy family with holding horizon.",
            exits="Maps timeout and close-N-period settings to timeframe.",
            regimes="Allows regime features to be recomputed on comparable windows.",
            risk="Controls sample size and drawdown horizon.",
            paper="Required for paper-trade reproducibility.",
            ingested="partially: local evidence pipeline has 5m/15m/1h/4h/1d for many pairs, but dashboard uses daily/4hour/hourly/5min labels",
            missing="Full dashboard capture for each timeframe/period combination and mapping to local timeframe names.",
            priority="must_integrate",
            evidence="pair detail controls 18-20 in read-only capture; docs endpoint catalog",
        ),
        row(
            area="Strategy dropdown",
            url=pair_url,
            fields="Static Spread, Static ZScoreR, Dynamic Spread, Dynamic ZScoreR, OU Spread, OU ZScoreR, Copula",
            controls="strategy select with values 3-1, 3-2, 1-1, 1-2, 2-1, 2-2, 1-3",
            point_in_time="point-in-time for inputs; backtest result summaries are historical and require walk-forward revalidation",
            export="Control value captured by pair detail helper; API backtest endpoint accepts strategy.",
            pair_selection="Shows which model family the dashboard can run on a pair.",
            strategy_matching="Directly maps to our z-score, OU, beta/dislocation, and copula families.",
            exits="Each strategy interacts with entry/exit threshold controls.",
            regimes="Strategy performance can be compared across stable/unstable regimes after capture.",
            risk="Strategy selection changes trade count and drawdown.",
            paper="Needed to reproduce dashboard settings for local tests.",
            ingested="partially: local strategy registry covers z-score, OU, copula, beta but not exact dashboard settings",
            missing="Exact dashboard strategy parameter semantics and raw backtest trades for each option.",
            priority="must_integrate",
            evidence="read-only capture control 20; docs/crypto_wizards_live_field_dictionary.csv",
        ),
        row(
            area="Spread and rolling z-score charts",
            url=pair_url,
            fields="SPREAD, ZSCORE (Rolling), cumulative returns, static strategy line",
            controls="chart toggle buttons SPREAD and ZSCORE (Rolling); strategy selection; period/timeframe controls",
            point_in_time="chart history can be point-in-time if raw arrays are timestamped; visual-only capture is insufficient for live signals",
            export="Documented /v1beta/spread and /v1beta/zscores endpoints; pair-detail helper targets spread/zscore arrays.",
            pair_selection="Identifies stable spreads and mean-reverting deviations.",
            strategy_matching="Core input for z-score entries and exits.",
            exits="Normalization, giveback, and stall exits need the path of spread/zscore after entry.",
            regimes="Spread volatility and z-score rank are regime inputs.",
            risk="Spread gaps and outliers reveal tail-risk periods.",
            paper="Must be ingested as timestamped arrays for paper-trade signal replay.",
            ingested="partially: local dYdX histories have spread/zscore; dashboard raw arrays not systematically captured",
            missing="Raw dashboard spread/zscore arrays from authenticated pair pages and API with_history responses.",
            priority="must_integrate",
            evidence="pair detail snapshot text; docs/crypto_wizards_pair_detail_extraction.md; docs/crypto_wizards_api_reference.md",
        ),
        row(
            area="Correlation / dependency view",
            url=pair_url,
            fields="Pearson, Spearman, Kendall; dependency chart options betas, correlation, volatilities",
            controls="dependency chart select: betas, correlation, volatilities; chart toggles",
            point_in_time="point-in-time if rolling dependency arrays are captured; single summary can hide instability",
            export="Pair-detail helper can capture controls; raw arrays still need network/worker capture.",
            pair_selection="Filters weak or unstable dependencies.",
            strategy_matching="Determines whether z-score/beta/hedge strategies are credible.",
            exits="Correlation breakdown can trigger hard exits.",
            regimes="Primary source for stable_correlation and stable_hedge overlays.",
            risk="Low or unstable correlation raises hedge failure risk.",
            paper="Preflight should require rolling dependency stability, not just a summary.",
            ingested="partially: local pipeline computes rolling correlation; dashboard dependency arrays not systematically ingested",
            missing="Dashboard rolling beta/correlation/volatility arrays and point-in-time timestamps.",
            priority="must_integrate",
            evidence="read-only capture control 46; pair detail snapshot text",
        ),
        row(
            area="ECM / error-correction views",
            url=pair_url,
            fields="ecm_y, ecm_x, ecm_strength; ECM Deviation minimum override",
            controls="dependency chart select options ecm (y), ecm (x), ecm strength; ECM Deviation (min)% input",
            point_in_time="safe only if ECM is recomputed using data available at each bar; dashboard summary can be hindsight-calibrated",
            export="Pair-detail helper specifically targets ecm_x, ecm_y, ecm_strength; local importer recognizes these arrays.",
            pair_selection="Validates leader/follower and error-correction behavior beyond simple correlation.",
            strategy_matching="Enables ECM, ECM+zscore, and ECM+copula strategy families.",
            exits="ECM strength or deviation can be used for stall/invalidated-trade exits.",
            regimes="ECM strength can define stable error-correction regimes.",
            risk="Weak ECM means spread can drift rather than revert.",
            paper="Could become a paper-trade preflight criterion after point-in-time validation.",
            ingested="partially: many raw pair detail histories include ECM fields, but evidence pipeline does not use them yet",
            missing="Point-in-time ECM feature pipeline and walk-forward tests using dashboard ECM arrays.",
            priority="must_integrate",
            evidence="docs/crypto_wizards_pair_detail_extraction.md; reports/pair_detail_capture_audit.csv",
        ),
        row(
            area="Copula statistics and conditional probability view",
            url=pair_url,
            fields="best-fit copula family, copula correlation, x given y, y given x, conditional chart, prices/returns thresholds at 1%, 5%, 10%",
            controls="conditional threshold select; x|y and y|x buttons",
            point_in_time="potentially point-in-time if calibrated on rolling window; static best-fit/conditional stats can be hindsight if calibrated over full sample",
            export="/v1beta/copula endpoint and pair-detail capture; raw calibration windows not yet systematic",
            pair_selection="Finds non-linear dependency and tail co-movement missed by correlation.",
            strategy_matching="Supports copula-dislocation and tail-event strategies.",
            exits="Conditional probability normalization can exit dislocation trades.",
            regimes="Tail-dependence shifts can mark crisis/high-risk regimes.",
            risk="Tail co-movement and conditional distortion feed sizing/risk filters.",
            paper="Useful if recalculated walk-forward and stress-tested.",
            ingested="partially: prescanned and raw histories include copula fields; current evidence pipeline uses simplified copula proxy",
            missing="Full copula history/calibration window, tail-dependence arrays, and no-hindsight validation.",
            priority="must_integrate",
            evidence="pair detail snapshot; docs endpoint catalog; data/raw/prescanned.json",
        ),
        row(
            area="Backtest entry and exit threshold panel",
            url=pair_url,
            fields="Entry Long (x), Entry Short (x), Exit Long (x), Exit Short (x), operators, sigma thresholds",
            controls="operator selects >= <= > < = and numeric threshold inputs",
            point_in_time="settings are point-in-time; displayed backtest result is historical and must be walk-forwarded",
            export="Captured controls; /v1beta/backtest endpoint accepts strategy/settings via params or bt_inputs",
            pair_selection="Shows how sensitive a pair is to conventional threshold rules.",
            strategy_matching="Direct mapping to our entry/exit style matrix.",
            exits="Primary source for regular threshold exits.",
            regimes="Threshold robustness can be compared by regime after local replay.",
            risk="Over-optimized thresholds are high hindsight risk.",
            paper="Must be locked before paper-trade replay.",
            ingested="partially: local scripts test z-score thresholds; dashboard exact settings not synchronized",
            missing="Exact btSettingsCA export, per-trade returns, and locked parameter manifest.",
            priority="must_integrate",
            evidence="read-only capture controls 47-54; docs/crypto_wizards_pair_detail_extraction.md",
        ),
        row(
            area="Backtest override panel",
            url=pair_url,
            fields="Close N Periods, Halflife, N=1..N=20, Stop Loss %, ECM Deviation min %, Corr Strength min %, capital weighting slider",
            controls="override select, stop-loss input, ECM deviation input, correlation strength input, range slider for asset X capital weighting",
            point_in_time="high hindsight risk if tuned on full page result; safe only as pre-declared parameter set in walk-forward",
            export="Captured controls; local importer captures hedge/capital weighting where available",
            pair_selection="Reveals whether apparent edge depends on fragile overrides.",
            strategy_matching="Maps to timeout exits, correlation gates, ECM gates, and hedge sizing.",
            exits="Close-N, stop-loss, half-life, ECM deviation, and correlation-strength exits.",
            regimes="Corr strength and ECM deviation are regime filters.",
            risk="Stop-loss and weighting controls directly affect drawdown.",
            paper="Paper preflight must include these exact settings.",
            ingested="partially: local tests include max_hold, stops, regimes; not exact dashboard overrides",
            missing="Parameter-export schema and no-hindsight guardrails for override tuning.",
            priority="must_integrate",
            evidence="read-only capture controls 55-59; pair detail snapshot text",
            feature_rank="risky_or_hindsight",
        ),
        row(
            area="Backtest result metrics",
            url=pair_url,
            fields="Sharpe, Sortino, net return, annualized return, mean period return, win rate, closed trades, max drawdown, VaR/CVaR at 99%, simulated VaR/CVaR",
            controls="strategy/timeframe/period/threshold controls drive these metrics",
            point_in_time="hindsight backtest summary; never live-entry input unless recreated point-in-time",
            export="/v1beta/backtest; visible page capture; local reports can ingest some metrics",
            pair_selection="Rejects fragile low-trade or high-drawdown pairs.",
            strategy_matching="Compares model family outcomes.",
            exits="Closed trades, drawdown, and return path hint at exit failures.",
            regimes="Compare metrics across dashboard periods/timeframes and local regimes.",
            risk="Primary dashboard risk summary.",
            paper="Preflight comparison only; must pass local walk-forward and cost stress.",
            ingested="partially: local pipeline computes its own metrics; dashboard metrics not trusted for promotion",
            missing="Per-trade backtest table/equity arrays and cost-assumption metadata.",
            priority="useful",
            evidence="pair detail snapshot text; docs endpoint catalog",
            feature_rank="risky_or_hindsight",
        ),
        row(
            area="Backtest chart tabs",
            url=pair_url,
            fields="NET, asset X, asset Y, entries, returns, underwater",
            controls="NET/X/Y buttons and chart select entries/returns/underwater",
            point_in_time="visual hindsight result unless exported as timestamped trade/equity arrays",
            export="Visible capture only so far; pair-detail helper should capture network payloads after toggling charts",
            pair_selection="Underwater path quickly shows unacceptable drawdown shape.",
            strategy_matching="Shows whether profits come from spread or one-leg drift.",
            exits="Entries and underwater chart are direct exit diagnostics sources.",
            regimes="Equity/drawdown by time can be aligned to local regime labels.",
            risk="Underwater chart is a drawdown-control diagnostic.",
            paper="Needed for paper preflight and post-trade review.",
            ingested="no: current repo does not systematically ingest dashboard trade/equity chart arrays",
            missing="Timestamped entries, returns, underwater, leg-level PnL arrays.",
            priority="must_integrate",
            evidence="read-only capture controls 64-67; pair detail snapshot text",
            feature_rank="risky_or_hindsight",
        ),
        row(
            area="API / docs / code / downloads / data nav",
            url="https://cryptowizards.net/",
            fields="API docs, guide links, code/examples, downloads, data links",
            controls="top navigation buttons api, guide, code, courses, downloads, data, zscore",
            point_in_time="documentation/navigation, not signal data",
            export="Manual capture; endpoint catalog already built",
            pair_selection="API docs define scalable pair discovery routes.",
            strategy_matching="Docs explain endpoint semantics and strategies.",
            exits="Backtest endpoint docs can expose supported exit settings.",
            regimes="May document regime/cost settings if available.",
            risk="May document risk assumptions and credits.",
            paper="Needed for repeatable automated capture pipeline.",
            ingested="partially: docs/crypto_wizards_api_reference.md and endpoint catalog exist",
            missing="Full guide/downloads/data page inventory from live authenticated session.",
            priority="useful",
            evidence="read-only capture controls 1-7; docs/crypto_wizards_api_reference.md",
        ),
        row(
            area="Dashboard members landing",
            url="https://cryptowizards.net/wizards/dashboard",
            fields="members area welcome, ZScore main application description, prescanned pair discovery description, custom analysis description, Telegram alerts description, simulated real-time trades description, quick platform intro video, guide link",
            controls="LAUNCH APP, ZScore button, guide link, YouTube intro link, Discord link, top nav api/guide/code/courses/downloads/data/zscore, account link",
            point_in_time="navigation and product description only, not signal data",
            export="browser capture; no tabular export observed",
            pair_selection="Confirms prescanned pair discovery and custom analysis are first-class dashboard workflows.",
            strategy_matching="Guide and launch path can lead to strategy documentation and app controls.",
            exits="Indirect only through app and guide navigation.",
            regimes="Indirect only through app and guide navigation.",
            risk="Indirect; may point to documentation for assumptions and alerts.",
            paper="Confirms simulated real-time trades are a dashboard capability to inventory and integrate.",
            ingested="no: landing navigation metadata is not tracked",
            missing="Live capture of all guide/docs/data/download links behind the authenticated dashboard landing.",
            priority="useful",
            evidence="live browser observation 2026-06-22 at /wizards/dashboard",
        ),
        row(
            area="Trades / simulated positions page",
            url="https://cryptowizards.net/wizards/zscore/trades",
            fields="Open Positions, Simulated open positions, Closed Positions, Simulated closed positions",
            controls="scanner/trades/alerts navigation; open and closed position sections; no position rows visible in current account",
            point_in_time="operational point-in-time paper-trading state when positions exist; closed-position analytics are historical records",
            export="browser capture only so far; no visible export observed with empty account",
            pair_selection="Post-trade outcomes can feed pair acceptance or rejection after enough simulated trades.",
            strategy_matching="Open/closed positions should reveal which strategy tags survive paper trading once data exists.",
            exits="Critical source for realized exit timing, hold time, and close reason if the page exposes them with positions.",
            regimes="Can align simulated trades to local regime labels after export.",
            risk="Potential source for realized drawdown, loss streaks, and open exposure.",
            paper="Core dashboard surface for paper-trading validation, live monitoring, and post-trade review.",
            ingested="no",
            missing="Open/closed position schema, strategy tags, timestamps, entry/exit prices, PnL, drawdown, close reason, and any export/API route.",
            priority="must_integrate",
            evidence="live browser observation 2026-06-22 at /wizards/zscore/trades",
        ),
        row(
            area="Alerts / Telegram credentials page",
            url="https://cryptowizards.net/wizards/zscore/alerts",
            fields="Telegram bot token input, Telegram chat ID input, Send Test button",
            controls="bot token field, chat ID field, Send Test",
            point_in_time="operational alert configuration, not signal data; sensitive credentials must never be captured or stored",
            export="do not export secrets; capture page metadata only",
            pair_selection="No pair-selection value directly.",
            strategy_matching="No strategy-matching value directly.",
            exits="Useful only for delivery of exit or risk alerts after a separate signal engine exists.",
            regimes="Useful only for delivery of regime alerts after a separate signal engine exists.",
            risk="Sensitive external-action surface; can support monitoring but must be isolated from research data capture.",
            paper="Useful for paper-trade alerts and monitoring if credentials are user-managed outside repo.",
            ingested="no",
            missing="Safe alert integration design that stores no Telegram token/chat ID in repo and requires explicit user approval before any test send.",
            priority="useful",
            evidence="live browser observation 2026-06-22 at /wizards/zscore/alerts",
            feature_rank="risky_or_hindsight",
        ),
        row(
            area="Account / subscription page",
            url="https://cryptowizards.net/wizards/account",
            fields="account management, email/access status, manage billing, PayPal subscription note, Back to Dashboard, Sign out, contact us",
            controls="Back to Dashboard, Sign out, Manage Billing, contact us",
            point_in_time="account and billing state only; sensitive operational page, not signal data",
            export="do not export account details; capture only page existence and non-sensitive metadata",
            pair_selection="No direct value.",
            strategy_matching="No direct value.",
            exits="No direct value.",
            regimes="No direct value.",
            risk="Can explain access/API limits, but should be excluded from quantitative datasets.",
            paper="No direct value except confirming access state.",
            ingested="no",
            missing="Nothing required for strategy research; only document as out of scope and avoid capturing private account details.",
            priority="optional",
            evidence="live browser observation 2026-06-22 at /wizards/account",
            feature_rank="optional",
        ),
        row(
            area="Hidden network / worker / route chunks",
            url=pair_url,
            fields="route chunk _id_, zscore_library, progress states completed_prices/econometrics/garch/ecm/backtest, viewItem output",
            controls="refresh/recalculate triggers network/worker computation",
            point_in_time="depends on payload; worker output can be point-in-time if timestamped and generated from current settings",
            export="capture_crypto_wizards_pair_detail.js; HAR fallback; pair_detail_capture_audit.csv",
            pair_selection="Could expose full computed feature set not visible in UI.",
            strategy_matching="Likely source of raw strategy/backtest/ECM arrays.",
            exits="Can expose per-trade and path data needed for real exit research.",
            regimes="May include GARCH/regime/dependency arrays.",
            risk="May include raw VaR/CVaR/drawdown paths.",
            paper="Most important source for reproducible paper-trade validation.",
            ingested="partially: helper and importer exist; current endpoint capture failed to collect useful fetches",
            missing="Successful authenticated capture with fetch/XHR/worker messages after refresh/recalculate.",
            priority="must_integrate",
            evidence="docs/crypto_wizards_pair_detail_extraction.md; data/raw/pair_details/crypto_wizards_pair_1_endpoint_capture.json",
        ),
        row(
            area="Profile/account/advanced settings and hidden popovers",
            url="unknown authenticated dashboard area",
            fields="unknown: profile, API key/credits, account, advanced settings, export options may exist",
            controls="unknown: requires live guided click-through of menus/buttons/popovers",
            point_in_time="unknown",
            export="not captured yet",
            pair_selection="Unknown until live inventory.",
            strategy_matching="Unknown until live inventory.",
            exits="Unknown until live inventory.",
            regimes="Unknown until live inventory.",
            risk="Could expose cost, credit, export, or account limits.",
            paper="Could expose API key/credits or paper/live account settings.",
            ingested="no",
            missing="Full guided live dashboard inventory beyond scanner and pair detail.",
            priority="useful",
            evidence="absence in current captures; user requirement",
        ),
    ]
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def md_table(rows: list[dict[str, str]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for item in rows:
        lines.append("| " + " | ".join(str(item.get(c, "")).replace("|", "\\|") for c in columns) + " |")
    return "\n".join(lines)


def build_missing_integrations(inventory: list[dict[str, str]]) -> list[dict[str, str]]:
    rows = []
    for item in inventory:
        if item["recommended_integration_priority"] in {"must_integrate", "useful"} and item["missing_from_current_repo"]:
            rows.append(
                {
                    "page_or_section_name": item["page_or_section_name"],
                    "missing_integration": item["missing_from_current_repo"],
                    "why_it_matters": "; ".join(
                        [
                            item["helps_pair_selection"],
                            item["helps_strategy_matching"],
                            item["helps_exits"],
                            item["helps_regimes"],
                            item["helps_risk_control"],
                        ]
                    ),
                    "hindsight_risk": item["point_in_time_or_hindsight"],
                    "priority": item["recommended_integration_priority"],
                    "recommended_next_step": next_step(item),
                }
            )
    return rows


def next_step(item: dict[str, str]) -> str:
    name = item["page_or_section_name"].lower()
    if "network" in name or "worker" in name:
        return "Use authenticated browser helper, click refresh/recalculate, then import capture and inspect candidate JSON paths."
    if "scanner selected" in name:
        return "Click representative rows, capture selected inline panel state and network payloads, then map strategy comparison fields without using them as live signals."
    if "scanner" in name:
        return "Run scanner capture across priority/filter combinations and archive timestamped snapshots."
    if "trades" in name or "simulated positions" in name:
        return "Create a non-live/paper position only with explicit approval, then capture open/closed position schema and export/API behavior."
    if "alerts" in name:
        return "Document alert metadata only; never capture credentials and require explicit approval before any external test send."
    if "dashboard members" in name:
        return "Inventory every guide/docs/download/data link from the members landing and classify each as data, docs, code, or operational."
    if "ecm" in name:
        return "Toggle ECM dependency views, refresh, capture raw ecm_x/ecm_y/ecm_strength arrays, then add point-in-time ECM features."
    if "backtest chart" in name:
        return "Toggle entries/returns/underwater and capture timestamped chart arrays or HAR payloads."
    if "profile" in name or "account" in name:
        return "Keep account/billing pages out of research datasets; capture only non-sensitive metadata when needed."
    return "Capture authenticated payload, map fields to local schema, and add no-hindsight validation before signal use."


def build_capture_opportunities(inventory: list[dict[str, str]]) -> list[dict[str, str]]:
    opportunities = []
    for item in inventory:
        if item["recommended_integration_priority"] in {"must_integrate", "useful"}:
            opportunities.append(
                {
                    "capture_target": item["page_or_section_name"],
                    "url_or_route": item["url_if_available"],
                    "capture_method": item["export_or_capture"],
                    "controls_to_exercise": item["filters_or_controls"],
                    "payload_or_fields_to_verify": item["fields_or_metrics_exposed"],
                    "success_criteria": "Timestamped raw data or settings captured; point-in-time status documented; importer maps fields; no live signal use if hindsight risk remains.",
                    "integration_priority": item["recommended_integration_priority"],
                }
            )
    return opportunities


def build_field_dictionary() -> str:
    field_rows = [
        ("pair_id/spread_id", "scanner, pair detail", "Join scanner rows to pair-detail pages and captures.", "yes, scanner", "must_integrate"),
        ("symbol_1/symbol_2", "scanner, pair detail controls", "Defines two-leg universe and local history fetch targets.", "yes", "must_integrate"),
        ("exchange", "scanner, pair detail/API params", "Filters dYdX vs other exchanges.", "partial", "must_integrate"),
        ("interval/period", "scanner, pair detail controls/API params", "Controls timeframe and sample window.", "partial", "must_integrate"),
        ("spread/zscore/zscore_roll", "spread/zscore charts/API", "Core entry, exit, and regime path.", "partial local, dashboard raw arrays missing", "must_integrate"),
        ("hedge_ratio/x_weighting/y_weighting", "scanner, pair header, weighting slider", "Sizing and spread construction.", "partial", "must_integrate"),
        ("Pearson/Spearman/Kendall", "correlation/dependency view", "Dependency validation and stable-correlation filters.", "local computed, dashboard summary partial", "must_integrate"),
        ("beta/betas", "dependency view", "Hedge stability and beta-anchor strategy design.", "partial local", "must_integrate"),
        ("ecm_x/ecm_y/ecm_strength", "ECM dependency views", "Error-correction strategy and exits.", "raw pair detail partial, not evidence pipeline", "must_integrate"),
        ("copula/u1_given_u2/u2_given_u1/tail thresholds", "copula view/API", "Nonlinear dependency, tail dislocation, and risk filters.", "partial", "must_integrate"),
        ("Hurst/half_life/ou_optimal", "scanner/header/API", "Mean-reversion validation and timeout design.", "partial", "must_integrate"),
        ("Sharpe/Sortino/returns/win_rate/closed trades", "backtest metrics", "Historical diagnostics only; not live signal features.", "local recomputed separately", "useful"),
        ("MDD/drawdown/VaR/CVaR/underwater", "risk metrics and backtest chart", "Risk gates, drawdown controls, paper preflight.", "partial local", "must_integrate"),
        ("entry/exit thresholds/operators", "backtest panel", "Reproducible dashboard strategy settings.", "not systematic", "must_integrate"),
        ("Close N/Stop Loss/ECM min/Corr min", "override panel", "Exit/risk/regime controls.", "not systematic", "must_integrate"),
        ("funding/slippage/cost assumptions", "not visible in current dashboard capture", "Paper-trade realism and stress tests.", "local placeholder only", "must_integrate"),
        ("scanner filter set", "live scanner page", "Reproducible discovery snapshots across sort, cointegration, correlation, Hurst, half-life, copula, strategy, symbol, and exchange filters.", "not systematic", "must_integrate"),
        ("inline strategy comparison returns/Sharpe", "live scanner selected-row detail", "Strategy-family triage only; must be replayed locally before acceptance.", "not ingested", "must_integrate"),
        ("open/closed simulated positions", "live trades page", "Paper-trading validation, live monitoring, and post-trade review.", "not ingested", "must_integrate"),
        ("Telegram alert configuration", "live alerts page", "Operational alert delivery only; credentials must stay outside repo.", "not ingested by design", "useful"),
    ]
    lines = [
        "# Dashboard Field Dictionary",
        "",
        "This dictionary maps Crypto Wizards dashboard/API fields to the local dYdX pair-research pipeline. Fields marked historical/backtest-derived must not be used for live signal generation unless they are captured and recomputed point-in-time.",
        "",
        "| Field | Dashboard Source | Local Use | Current Ingestion | Priority |",
        "| --- | --- | --- | --- | --- |",
    ]
    for field, source, use, ingestion, priority in field_rows:
        lines.append(f"| {field} | {source} | {use} | {ingestion} | {priority} |")
    lines.extend(
        [
            "",
            "## No-Hindsight Rule",
            "",
            "- Scanner/backtest rankings are discovery inputs, not deployable signals.",
            "- Raw spread, z-score, dependency, ECM, copula, entry, return, and underwater arrays must be timestamped and replayed locally before strategy use.",
            "- Any dashboard feature that is calibrated on the full visible sample is `risky_or_hindsight` until it can be reconstructed bar-by-bar.",
            "- Account and alert credential fields are operational metadata only; never store private account data, bot tokens, or chat IDs in research artifacts.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    inventory = build_inventory()
    missing = build_missing_integrations(inventory)
    opportunities = build_capture_opportunities(inventory)

    write_csv(REPORTS / "dashboard_full_inventory.csv", inventory)
    write_csv(REPORTS / "dashboard_missing_integrations.csv", missing)
    write_csv(REPORTS / "dashboard_capture_opportunities.csv", opportunities)

    md_columns = [
        "page_or_section_name",
        "url_if_available",
        "point_in_time_or_hindsight",
        "already_ingested",
        "missing_from_current_repo",
        "feature_rank",
        "recommended_integration_priority",
    ]
    (REPORTS / "dashboard_full_inventory.md").write_text(
        "# Dashboard Full Inventory\n\n"
        "Built from local authenticated pair-detail snapshots, read-only browser-state capture, API docs, endpoint catalog, existing capture audits, and live guided dashboard exploration on 2026-06-22. The live session confirmed scanner, inline selected-pair detail, members dashboard, trades, alerts, and account pages; the older direct pair route redirected to the marketing root and needs route rediscovery.\n\n"
        + md_table(inventory, md_columns)
        + "\n\n## Feature Rank Counts\n\n"
        + "\n".join(
            f"- {rank}: {sum(1 for row in inventory if row['feature_rank'] == rank)}"
            for rank in ["must_integrate", "useful", "optional", "not_useful", "risky_or_hindsight"]
        )
        + "\n\n## Integration Priority Counts\n\n"
        + "\n".join(
            f"- {priority}: {sum(1 for row in inventory if row['recommended_integration_priority'] == priority)}"
            for priority in ["must_integrate", "useful", "optional", "not_useful", "risky_or_hindsight"]
        )
        + "\n\n## Access Note\n\n"
        "Live access was available for the scanner and top-level dashboard pages in this run. Full historical pair-detail captures still come from older saved evidence; the current live route/action for the deeper pair detail page remains unresolved.\n",
    )

    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "dashboard_field_dictionary.md").write_text(build_field_dictionary())

    summary = [
        "# Dashboard Integration Summary",
        "",
        "## Current State",
        "",
        "- Scanner/prescanned data is partially ingested and useful for discovery.",
        "- Live access on 2026-06-22 confirmed the scanner page, inline selected-pair detail, members dashboard, simulated trades page, Telegram alerts page, and account page.",
        "- Scanner/prescanned data is partially ingested and useful for discovery; live scanner filters include sort, cointegration, correlation, Hurst, half-life, copula, strategy, symbol, and exchange controls.",
        "- Pair-detail dashboard evidence exists from older pair/1 captures, including z-score, spread, correlation, copula, ECM controls, backtest metrics, and risk metrics. In the live session, the old direct /wizards/zscore/pair/1?origin=scanner route redirected to /, so the current full pair-detail route or action must be rediscovered.",
        "- Selecting a scanner row now exposes an inline detail panel with spread/z-score chart toggles and strategy-family return/Sharpe comparisons; those summaries are hindsight diagnostics until locally replayed.",
        "- Trades page exists for simulated open/closed positions, but the current account showed no position rows to map.",
        "- Alerts page contains Telegram token/chat ID fields and a Send Test action; capture only metadata and never store credentials or trigger external sends without explicit approval.",
        "- Existing local evidence pipeline does not yet systematically ingest dashboard ECM, dependency, copula calibration, entries/returns/underwater chart arrays, or exact dashboard backtest settings.",
        "- Funding/slippage/cost assumptions remain incomplete for dashboard-derived paper-trade validation.",
        "",
        "## Highest Priority Integrations",
        "",
    ]
    for item in missing[:10]:
        summary.append(f"- {item['page_or_section_name']}: {item['recommended_next_step']}")
    summary.extend(
        [
            "",
            "## Gate",
            "",
            "Dashboard features can improve research, but any feature marked mixed, hindsight, or unknown must stay out of live signal generation until captured as timestamped point-in-time data and validated through walk-forward plus cost stress.",
        ]
    )
    (REPORTS / "dashboard_integration_summary.md").write_text("\n".join(summary) + "\n")

    print(
        json.dumps(
            {
                "inventory_rows": len(inventory),
                "missing_integrations": len(missing),
                "capture_opportunities": len(opportunities),
                "outputs": [
                    str(REPORTS / "dashboard_full_inventory.csv"),
                    str(REPORTS / "dashboard_full_inventory.md"),
                    str(REPORTS / "dashboard_missing_integrations.csv"),
                    str(REPORTS / "dashboard_capture_opportunities.csv"),
                    str(DOCS / "dashboard_field_dictionary.md"),
                    str(REPORTS / "dashboard_integration_summary.md"),
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
