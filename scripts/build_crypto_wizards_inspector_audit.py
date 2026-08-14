from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
AUDIT_DATE = "2026-08-12"


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _api_row(
    name: str,
    method: str,
    path: str,
    credits: int,
    input_source: str,
    required: str,
    optional: str,
    response: str,
    verification: str,
    evidence: str,
    integration: str,
    missing: str,
    issue: str = "",
) -> dict[str, object]:
    return {
        "endpoint_name": name,
        "method": method,
        "path": path,
        "credits": credits,
        "input_source": input_source,
        "required_parameters": required,
        "optional_parameters": optional,
        "response_contract_summary": response,
        "documentation_status": "documented_public_v1beta",
        "live_verification_status": verification,
        "live_evidence_path": evidence,
        "point_in_time_policy": "point_in_time_only_when_input_end_timestamp_and_full_configuration_are_frozen",
        "dashboard_parity_policy": "diagnostic_only_until_same_inputs_and_settings_match",
        "repo_integration_status": integration,
        "missing_integration": missing,
        "known_contract_issue": issue,
        "recommended_priority": "must_integrate",
        "promotion_authority": False,
        "live_trading_authorized": False,
    }


def api_rows() -> list[dict[str, object]]:
    get_evidence = "reports/active/wizard_pair_detail_api_pilot_manifest.csv"
    return [
        _api_row(
            "backtest_get", "GET", "/v1beta/backtest", 6, "Wizard-supplied venue data",
            "symbol_1;symbol_2;exchange;interval;period;strategy",
            "spread_type;roll_w;with_history;entry_level;exit_level;exit_n_periods;x_weighting;slippage_rate;commission_rate;stop_loss_rate",
            "Sharpe;Sortino;total/annual/mean return;MDD;VaR;CVaR;win rate;optional spread/zscore/bt-return history",
            "live_verified_2026-08-08", get_evidence,
            "integrated_pair_detail_pilot_and_history_import",
            "closed-trade accounting and exact dashboard chart parity",
            "Official TypeScript GET example uses 0.0005 commission while Python and current dashboard use 0.001.",
        ),
        _api_row(
            "cointegration_get", "GET", "/v1beta/cointegration", 5, "Wizard-supplied venue data",
            "symbol_1;symbol_2;exchange;interval;period", "spread_type;roll_w;with_history",
            "Engle-Granger is_coint;p_value;t_stat;critical values;optional history",
            "live_verified_2026-08-08", get_evidence,
            "schema_pilot_only", "Johansen detail;deterministic-term specification;rolling PIT reconstruction",
        ),
        _api_row(
            "copula_get", "GET", "/v1beta/copula", 5, "Wizard-supplied venue data",
            "symbol_1;symbol_2;exchange;interval;period", "with_history",
            "copula family;copula correlation;u1_given_u2;u2_given_u1",
            "live_verified_2026-08-08", get_evidence,
            "schema_pilot_and_discovery_fields", "calibration history;1/5/10 percent price/return contours",
        ),
        _api_row(
            "correlations_get", "GET", "/v1beta/correlations", 5, "Wizard-supplied venue data",
            "symbol_1;symbol_2;exchange;interval;period", "with_history",
            "Pearson;Spearman;Kendall returns correlations",
            "live_verified_2026-08-08", get_evidence,
            "schema_pilot_only", "conditional correlation history and dashboard chart parity",
        ),
        _api_row(
            "credits_used_get", "GET", "/v1beta/credits-used", 0, "Account credit ledger",
            "API key", "none", "credits used/remaining for UTC day",
            "live_verified_2026-08-12", "reports/active/wizard_credit_ledger_status.json",
            "integrated_credit_ledger", "vendor docs response example is incorrect",
            "Documentation page displays a correlations path and Copula-shaped example.",
        ),
        _api_row(
            "prescanned_get", "GET", "/v1beta/prescanned", 10, "Wizard prescanned universe",
            "priority;strategy", "exchange;interval;asset",
            "Top 100 rows with pair/mode/performance/risk/stationarity/copula/zscore/profile fields",
            "live_verified_full_30_cell_sweep_2026-08-12", "reports/active/wizard_sweep_manifest.csv",
            "integrated_exhaustive_credit_aware_sweep", "API lacks browser-only priorities/filters and returns at most 100 rows per cell",
        ),
        _api_row(
            "spread_get", "GET", "/v1beta/spread", 5, "Wizard-supplied venue data",
            "symbol_1;symbol_2;exchange;interval;period", "spread_type;roll_w;with_history",
            "spread;zscore;rolling zscore;hedge ratio;Hurst;half-life;crossings;optional history",
            "live_verified_2026-08-08", get_evidence,
            "schema_pilot_and_history_import", "exact static/dynamic/OU formula parity remains incomplete",
            "Documentation heading shows /cointegration while the example request uses /spread.",
        ),
        _api_row(
            "zscores_get", "GET", "/v1beta/zscores", 5, "Wizard-supplied venue data",
            "symbol_1;symbol_2;exchange;interval;period", "spread_type;roll_w;with_history",
            "latest zscore;latest rolling zscore;optional spread/zscore history",
            "live_verified_2026-08-08", get_evidence,
            "integrated_history_import", "full causal formula parity and 4-hour support",
            "Documentation heading shows /cointegration while the example request uses /zscores.",
        ),
        _api_row(
            "backtest_post", "POST", "/v1beta/backtest", 2, "Caller-supplied equal-length OHLC series",
            "params.series_1_opens/closes;params.series_2_opens/closes;params.strategy;bt_inputs",
            "spread_type;roll_w;with_history;entry/exit;x_weighting;costs;exit_n;stop",
            "Same backtest metrics as GET on caller-supplied data",
            "live_verified_multi_mode_2026-08-12", "reports/active/hyperliquid_wizard_vendor_mode_proofs.csv",
            "integrated_exact_mode_proof_lane", "vendor performance accounting and formula parity still block acceptance",
            "Official POST examples use stop_loss_rate_opt, while the repo's successful proof lane uses stop_loss_rate; preserve response-tested contract evidence.",
        ),
        _api_row(
            "cointegration_post", "POST", "/v1beta/cointegration", 1, "Caller-supplied equal-length close series",
            "series_1_closes;series_2_closes", "spread_type;roll_w;with_history",
            "Engle-Granger statistics on supplied data",
            "live_verified_2026-08-12", "reports/active/crypto_wizards_post_contract_probe.csv",
            "integrated_typed_contract_probe", "PIT formula parity",
        ),
        _api_row(
            "copula_post", "POST", "/v1beta/copula", 1, "Caller-supplied equal-length close series",
            "series_1_closes;series_2_closes", "none documented",
            "copula family and directional conditional probabilities",
            "live_verified_behavioral_proof", "reports/active/wizard_copula_behavioral_v2_status.json",
            "integrated_typed_client_and_behavioral_proof", "formula identity and calibration-window parity",
        ),
        _api_row(
            "correlations_post", "POST", "/v1beta/correlations", 1, "Caller-supplied equal-length close series",
            "series_1_closes;series_2_closes", "none documented",
            "Pearson;Spearman;Kendall correlations",
            "live_verified_2026-08-12", "reports/active/crypto_wizards_post_contract_probe.csv",
            "integrated_typed_contract_probe", "PIT formula parity",
        ),
        _api_row(
            "spread_post", "POST", "/v1beta/spread", 1, "Caller-supplied equal-length close series",
            "series_1_closes;series_2_closes", "spread_type;roll_w;with_history",
            "spread/zscore/hedge/Hurst/half-life/crossing fields",
            "live_verified_2026-08-12", "reports/active/crypto_wizards_post_contract_probe.csv",
            "integrated_typed_contract_probe", "exact formula parity",
        ),
        _api_row(
            "zscores_post", "POST", "/v1beta/zscores", 1, "Caller-supplied equal-length close series",
            "series_1_closes;series_2_closes", "spread_type;roll_w;with_history",
            "latest and optional historical spread/zscore values",
            "live_verified_2026-08-12", "reports/active/crypto_wizards_post_contract_probe.csv",
            "integrated_typed_contract_probe", "exact formula parity",
        ),
    ]


def parity_rows() -> list[dict[str, object]]:
    values = [
        ("venues", "Binance;Binance US;ByBit;Coinbase;DYDX;Forex;Stocks", "GET supports documented market-data venues; POST is venue-neutral", "API discovery covers five crypto venues; dashboard also exposes Forex/Stocks", "research venue must be explicit; Hyperliquid acceptance remains local"),
        ("timeframes", "Daily;4 Hour;1 Hour;5 Min", "Daily;Hourly;Min5 documented", "4 Hour is a dashboard/API parity gap", "never map 4 Hour to Hourly"),
        ("exact modes", "Static/Dynamic/OU x Spread/ZScoreR plus Copula", "strategy Spread/ZScoreRoll/Copula plus spread_type Static/Dynamic/Ou", "representable in both", "configuration hash must preserve exact mode"),
        ("stationarity", "Johansen badge;EG badge;EG trend color;p-value tooltip", "prescanned has Johansen/EG/trend flags; cointegration endpoint exposes EG statistics", "Johansen specification absent from analytics API", "structured booleans/colors only; text labels are not flags"),
        ("ECM", "ECM Y;ECM X;ECM Strength charts and minimum override", "not exposed", "dashboard-only diagnostic", "capture/recompute causally; never infer from API absence"),
        ("volatility", "long-term/GARCH persistence/shock/asymmetry/joint behavior", "not exposed as dedicated endpoint", "dashboard-only and observed missing/implausible states", "missing/crazy/zero values create quality blockers"),
        ("copula", "family;correlation;directionals;price/return contours at 1/5/10%", "family;correlation/directionals", "API lacks contour surfaces", "API can screen; dashboard diagnoses; local replay accepts"),
        ("backtest controls", "separate long/short operators;close-N/half-life;ECM/corr overrides;weighting", "single entry/exit levels;exit_n;stop;weighting;costs", "dashboard has richer asymmetric controls", "capture full UI settings before claiming parity"),
        ("costs", "Preferences observed commission 0.10%;slippage 0.05%", "explicit commission/slippage request parameters", "parity is possible when explicitly set", "no implicit stop; record both rates in config hash"),
        ("performance accounting", "headline metrics plus closed trades and charts", "summary metrics and optional bt_returns", "nonzero metrics can coexist with zero closed trades", "ambiguous accounting is discovery-only and cannot train ML/RL"),
        ("raw market data", "Data page OHLCV and derived-field Excel export", "no documented raw OHLCV endpoint", "dashboard export only", "prefer local Hyperliquid raw bars for acceptance"),
        ("paper/alerts", "simulated trades and Telegram alerts", "no documented endpoint", "dashboard-only operational surfaces", "do not capture secrets; reconcile paper outcomes separately"),
    ]
    return [
        {
            "surface": name,
            "dashboard_contract": dashboard,
            "api_contract": api,
            "parity_finding": finding,
            "system_rule": rule,
            "evidence_date": AUDIT_DATE,
        }
        for name, dashboard, api, finding, rule in values
    ]


def optimization_rows() -> list[dict[str, object]]:
    values = [
        ("scanner interval", "Daily;Hourly", "all other scanner filters", "candidate-set stability", "different interval is a different experiment"),
        ("scanner exchange", "Binance;Binance US;ByBit;Coinbase;DYDX", "interval and filters", "venue-specific discovery", "never merge symbols/costs across venues"),
        ("scanner priority", "Sharpe;Returns;spread/zscore extremes;recent", "candidate universe", "ranking sensitivity", "sort changes ranking, not source eligibility"),
        ("cointegration overlay", "all;Johansen;EG;EG excluding trend;both;either", "base candidate set", "stationarity agreement", "orange EG is trend-qualified, not green confirmation"),
        ("dependency overlays", "correlation;Hurst;half-life;copula arbitrage", "one overlay per comparison", "diagnostic sensitivity", "zero-result filters are evidence, not permission to silently relax"),
        ("exact mode", "seven modes", "pair;orientation;venue;timeframe;lookback;costs", "after-cost metrics and trade count", "OU Optimal is an overlay, not an eighth mode"),
        ("orientation", "X/Y;Y/X", "same data window and settings", "directional asymmetry", "must recalculate, not relabel"),
        ("lookback", "100 through venue limit;DYDX max 360", "data end timestamp", "plateau stability", "default DYDX 365 UI state is invalid against its 360 limit"),
        ("entry", "mode-specific threshold neighborhood", "exit and risk controls", "trade count/PF/Sharpe/MDD", "prefer plateaus over a single peak"),
        ("exit", "mean/rolling normalization;copula 0.50;close-N/half-life", "entry", "expectancy and hold-time stability", "include hard-exit and entry-only regime variants locally"),
        ("rolling window", "auto(0) and bounded nearby windows", "spread family and mode", "walk-forward stability", "auto resolved value must be captured"),
        ("risk overrides", "stop;ECM min;correlation min", "signal parameters", "drawdown reduction without trade collapse", "full-sample tuning is hindsight"),
        ("capital weighting", "Risk Reduction;Even;slider", "signal and costs", "risk-adjusted return", "weighting changes exposure and cost; hash it"),
        ("metrics mode", "Standard;Aggressive", "all settings", "reporting sensitivity", "Aggressive removes idle periods and cannot be compared as ordinary Sharpe"),
        ("cost stress", "verified Wizard base;Hyperliquid observed;2x stress", "gross signal", "after-cost survival", "Wizard headline metrics never replace local execution costs"),
        ("copula contour", "prices/returns x 1/5/10%", "pair/window/orientation", "tail-direction consistency", "calibration instability blocks live feature use"),
        ("dependency chart", "betas;correlation;volatilities;ECM Y/X/strength", "same pair state", "diagnostic coherence", "chart summaries are not acceptance authority"),
    ]
    return [
        {
            "experiment_dimension": name,
            "values_to_compare": choices,
            "hold_constant": constants,
            "primary_readout": readout,
            "invalidation_rule": invalidation,
            "optimization_policy": "one_factor_then_small_predeclared_grid_then_purged_walk_forward",
            "promotion_authority": False,
        }
        for name, choices, constants, readout, invalidation in values
    ]


def inspector_markdown() -> str:
    return f"""# Crypto Wizards Inspector Second Pass

Audit date: {AUDIT_DATE}

## Scope

This pass rechecked the authenticated product as a stateful research system, not a single scanner page. It covered the members dashboard, scanner, expanded scanner detail, pair workspace, all seven exact modes, six dependency views, six copula contour views, the backtest machine and Preferences, custom analysis, Data export, simulated trades, alerts, API service, downloads, courses, code listing, account navigation, public v1beta docs, official Python/TypeScript examples, live API evidence, and repo ingestion contracts. No account secret, API key, Telegram credential, order, alert, purchase, or paper position was captured or submitted.

## High-Confidence Findings

- The scanner is a discovery ranking surface. It has Daily and Hourly runs, five crypto venues, broad sort/filter controls, exact-mode filters, and colored stationarity semantics.
- The pair page is a separate diagnostic and hypothesis surface. It adds 4 Hour and 5 Min, full settings, ECM, GARCH/volatility, copula contours, asymmetric entry/exit controls, and richer backtest charts.
- The seven modes are Static Spread, Static ZScoreR, Dyn Spread, Dyn ZScoreR, OU Spread, OU ZScoreR, and Copula. `ou_optimal` is only a scanner overlay.
- Green Johansen means the Johansen boolean is true. Green EG means EG is true without trend qualification. Orange EG occurs when both EG and its include-trend flag are true. Gray means the corresponding boolean is false. Plain `Jn`/`EG` text has no boolean meaning.
- A pair can show nonzero return/Sharpe with zero closed trades. That is ambiguous open or mark-to-market accounting and is blocked from acceptance, ML labels, and RL rewards.
- The live HBAR/XLM example exposed timeframe data-quality failures: implausible `crazy%` volatility at 4 Hour and zero/missing GARCH behavior at shorter intervals. Timeframe availability does not prove usable data.
- Preferences were observed at 0.10% commission, 0.05% slippage, Risk Reduction, Standard metrics, 1000 simulations, and Manual update. Aggressive metrics exclude idle periods and inflate comparisons.
- The custom-analysis modal allows Daily, 4 Hour, 1 Hour, and 5 Min, but its visible DYDX default of 365 conflicts with the client-enforced DYDX maximum of 360.
- `Restore` clears all local storage for the app origin. It is not a narrow preference reset.

## API Scrub

The public v1beta surface has 14 documented method/path contracts: eight GET and six POST. GET endpoints use Wizard-supplied venue data; POST endpoints use caller-supplied series and are the lower-cost parity lane. The repo now has live, archived evidence for every documented contract. The four previously missing POST analytics contracts were probed on the same frozen 360-bar Hyperliquid series; all four completed and the vendor credit counter reconciled exactly from 318 to 322.

The API does not replace the dashboard. It does not document ECM Y/X/Strength, Johansen test details, GARCH panels, raw OHLCV, 4 Hour analytics, paper trades, alerts, or all dashboard-specific asymmetric backtest controls. Conversely, the API is the scalable and reproducible surface for exhaustive discovery and identical-input parity tests.

Known documentation defects are recorded in `reports/crypto_wizards_api_full_inventory.csv`: incorrect credits-used example content, spread/zscores page heading paths, a POST backtest page showing a GET example, TypeScript/Python commission disagreement, and `stop_loss_rate_opt` versus `stop_loss_rate` ambiguity.

## Operating Model

1. Run the complete API prescanned sweep without prefiltering rows.
2. Use scanner/browser views to confirm visible state, colors, hidden filters, and candidates missing because of the API top-100 limit.
3. Open pair pages for retained cohorts and capture both orientations, all seven modes, all settings, ECM/GARCH/copula diagnostics, and chart/accounting quality.
4. Use the Wizard backtest only to generate hypotheses and robust parameter neighborhoods.
5. Replay on frozen Hyperliquid bars with explicit costs, funding, slippage, walk-forward splits, regimes, and trade accounting.
6. Keep all Wizard evidence discovery-only until local acceptance passes; no dashboard metric directly authorizes an order.

## Remaining Gaps

- No automatic authenticated browser crawler currently completes every scanner row and every pair page while proving infinite-scroll termination and both orientations.
- Live schema coverage is complete, but response-semantic and exact-formula parity remain incomplete.
- Dashboard chart arrays need network/DOM capture with timestamps; visible SVG presence is not the same as numerical history.
- Exact static/dynamic/OU formula identity remains incomplete. Full-sample fits are not live-signal-safe.
- Wizard performance needs realized/open PnL reconciliation when closed trades are zero.
- Pair-page data-quality blockers must flow into every downstream journal, teacher/student/RL feature packet, and acceptance report.

## Decision

Crypto Wizards is now understood well enough to operate as a structured discovery and diagnostic laboratory. It is not yet a standalone acceptance or execution engine. The next implementation priority is to automate the authenticated capture contract and finish identical-input semantic/formula parity, while preserving Hyperliquid local replay as the only acceptance authority.
"""


def main() -> None:
    _write_csv(REPORTS / "crypto_wizards_api_full_inventory.csv", api_rows())
    _write_csv(REPORTS / "crypto_wizards_browser_api_parity.csv", parity_rows())
    _write_csv(REPORTS / "crypto_wizards_dashboard_optimization_matrix.csv", optimization_rows())
    (REPORTS / "crypto_wizards_inspector_second_pass.md").write_text(
        inspector_markdown(), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
