"""Reconcile the restricted Udemy math evidence with active project methods.

The courses contain several valid but incompatible conventions.  This module
keeps those conventions named and auditable instead of silently treating every
course example as the production formula.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path

from quant_platform.active_pipeline import ROOT, CommandResult
from quant_platform.orchestration.corrective_runtime import atomic_write_text


@dataclass(frozen=True)
class UdemyMathContract:
    topic: str
    course: str
    lecture_id: str
    lecture_title: str
    source_convention: str
    current_convention: str
    match_status: str
    production_authority: str
    implementation_path: str
    historical_impact: str
    required_action: str
    test_path: str = "tests/test_udemy_math_reconciliation.py"


CONTRACTS: tuple[UdemyMathContract, ...] = (
    UdemyMathContract(
        "rolling_zscore",
        "DYDX Pairs Trading Bot",
        "udl_3f68b050e7fb9996",
        "Construct Cointegration Functions",
        "z_t=(s_t-rolling_mean_w(s))/rolling_sample_std_w(s); pandas ddof=1; course window=21",
        "causal rolling z-score with captured/configured window and sample standard deviation ddof=1",
        "MATCH",
        "local_validated_estimator",
        "src/quant_platform/wizard_mode_replay.py",
        "prior ddof=0 performance paths were inconsistent but exact-mode rolling z-score already used ddof=1",
        "retain both ddof variants only for vendor parity research; use ddof=1 for active ZScoreR",
        "tests/test_udemy_math_reconciliation.py;tests/test_wizard_mode_replay.py",
    ),
    UdemyMathContract(
        "spread_direction",
        "DYDX Pairs Trading Bot",
        "udl_6e269f86e9f73e84",
        "Trading the Spread Based on Z-Score",
        "spread=series_1-beta*series_2; negative spread signal means long series_1 and short series_2",
        "spread=log(price_y)-beta*log(price_x); negative metric maps to long Y and short X",
        "MATCH_DECLARED_CONVENTION",
        "local_validated_estimator",
        "src/quant_platform/wizard_mode_replay.py;src/quant_platform/backtest.py",
        "teacher and Hyperliquid research orientation was reversed before this audit",
        "keep explicit Y-on-X orientation in every setup identity and regression output",
    ),
    UdemyMathContract(
        "static_spread_estimation",
        "DYDX Pairs Trading Bot",
        "udl_3f68b050e7fb9996",
        "Construct Cointegration Functions",
        "older bot course: raw prices, OLS series_1 on series_2 without an intercept, spread=series_1-beta*series_2",
        "Math V2: log prices, OLS with intercept, residual=log(Y)-alpha-beta*log(X)",
        "INTENTIONAL_DIVERGENCE",
        "local_validated_estimator",
        "src/quant_platform/statistics/math_v2.py",
        "old course examples and Math V2 results are not numerically interchangeable",
        "never compare the two without method_id; preserve the old formula as educational reference only",
    ),
    UdemyMathContract(
        "engle_granger",
        "Financial Econometrics",
        "udl_3350ce663c33d71f",
        "Cointegration - Testing ONEUSDT vs MANAUSDT",
        "log-price regression with intercept; ADF on the fitted residual",
        "log-price Engle-Granger regression with intercept plus coint and residual ADF diagnostics",
        "MATCH",
        "local_validated_estimator",
        "src/quant_platform/statistics/math_v2.py",
        "older raw-price cointegration reports remain a different method",
        "require integration-order and structural-stability checks before acceptance",
    ),
    UdemyMathContract(
        "half_life",
        "DYDX Pairs Trading Bot",
        "udl_3f68b050e7fb9996",
        "Construct Cointegration Functions",
        "delta_s_t=a+lambda*s_(t-1)+e_t; half_life=-ln(2)/lambda",
        "s_t=c+phi*s_(t-1)+e_t; theta=-ln(phi)/delta_t; half_life=ln(2)/theta",
        "INTENTIONAL_DIVERGENCE",
        "local_validated_estimator",
        "src/quant_platform/statistics/math_v2.py",
        "the course approximation and exact discrete AR(1)-to-OU conversion differ materially when reversion is fast",
        "retain Math V2 exact conversion; report the course approximation only as a named comparison field",
    ),
    UdemyMathContract(
        "ecm",
        "Financial Econometrics",
        "udl_9e888d06bff70d44",
        "Cointegration - ECM Applied to ONEUSDT and MANAUSDT",
        "log differences regressed on lagged residual and lagged changes for both legs; gamma is adjustment speed",
        "two one-lag log-difference ECM equations with HC1 errors and explicit gamma signs/p-values",
        "MATCH_DECLARED_CONVENTION",
        "local_validated_estimator",
        "src/quant_platform/statistics/math_v2.py",
        "historical arbitrary ECM strength proxies are not comparable",
        "keep ecm_strength documented as the share of significant expected-sign adjustment coefficients",
    ),
    UdemyMathContract(
        "gaussian_copula_marginals",
        "Financial Econometrics",
        "udl_42f43d33143bc940",
        "Copulas - Plan of Attack",
        "log returns transformed to empirical marginal CDF values u1 and u2",
        "returns transformed to rank/(n+1) empirical pseudo-observations",
        "MATCH",
        "local_validated_estimator",
        "src/quant_platform/statistics/math_v2.py",
        "legacy tanh z-score copula proxies are invalid for this comparison",
        "keep pseudo-observations point-in-time and block legacy proxy fields from copula authority",
    ),
    UdemyMathContract(
        "gaussian_copula_rho",
        "Financial Econometrics",
        "udl_42f43d33143bc940",
        "Copulas - Plan of Attack",
        "course spreadsheet uses Pearson correlation of the modeled return series as Gaussian rho",
        "Math V2 uses Pearson correlation of inverse-normal empirical pseudo-observations",
        "INTENTIONAL_DIVERGENCE",
        "local_validated_estimator",
        "src/quant_platform/statistics/math_v2.py",
        "conditional probabilities can differ even when the same marginal ranks are used",
        "record rho_method and compare both variants out of sample before changing the active estimator",
    ),
    UdemyMathContract(
        "gaussian_copula_conditionals",
        "Financial Econometrics",
        "udl_50c241e9203a9747",
        "Copulas - Gaussian Conditional Probability",
        "h1=Phi((Phi^-1(u1)-rho*Phi^-1(u2))/sqrt(1-rho^2)); swap indices for h2",
        "same Gaussian h-functions for u1_given_u2 and u2_given_u1",
        "MATCH",
        "local_validated_estimator",
        "src/quant_platform/statistics/math_v2.py",
        "directional output was previously vulnerable to X/Y orientation ambiguity",
        "retain explicit conditional view and leg orientation in every copula setup",
    ),
    UdemyMathContract(
        "signal_timing",
        "Machine Learning Applied to Trading",
        "udl_e03daae45a2c9f7d",
        "Backtesting 101 - Calculations and Strategy Returns",
        "a close-derived signal must be shifted before earning the next bar return",
        "target weights are shifted one bar before multiplication by leg returns",
        "MATCH",
        "local_validated_estimator",
        "src/quant_platform/backtest.py",
        "same-bar fills would have introduced look-ahead bias",
        "keep the one-bar execution lag under regression test",
    ),
    UdemyMathContract(
        "portfolio_returns",
        "Machine Learning Applied to Trading",
        "udl_e03daae45a2c9f7d",
        "Backtesting 101 - Calculations and Strategy Returns",
        "single-asset log returns aggregate with exp(cumsum(log_return))-1",
        "weighted two-leg simple returns aggregate with product(1+bar_net_return)-1",
        "MATCH_DECLARED_CONVENTION",
        "local_validated_estimator",
        "src/quant_platform/backtest.py;src/quant_platform/trade_ledger.py",
        "mixing additive simple returns with compounded equity would overstate or understate results",
        "retain exact bar-to-trade-to-equity reconciliation",
    ),
    UdemyMathContract(
        "sharpe",
        "Machine Learning Applied to Trading",
        "udl_dd780157ebe1afb7",
        "Backtesting 101 - Metrics and Equity Curve",
        "annualized mean(log_return)/sample_std(log_return), with optional risk-free rate and 255 equity trading days",
        "annualized mean(net simple return)/sample_std(net simple return), zero risk-free rate, interval-aware crypto periods",
        "MATCH_DECLARED_CONVENTION",
        "local_validated_estimator",
        "src/quant_platform/performance_math.py",
        "population-standard-deviation paths previously inflated small-sample Sharpe",
        "use sample ddof=1 everywhere and always store return_type, risk_free_rate, and periods_per_year",
    ),
    UdemyMathContract(
        "max_drawdown",
        "Statistical Arbitrage Bot",
        "udl_4fb37e2134cc0dbd",
        "UPDATE - Automated Backtesting with PDF Output",
        "peak-to-trough loss on the equity curve",
        "max((running_peak-equity)/running_peak) with initial equity=1.0 included",
        "FIXED_MATCH",
        "local_validated_estimator",
        "src/quant_platform/backtest.py",
        "historical reports could understate a loss beginning on the first bar",
        "invalidate and rerun historical drawdown-dependent acceptance outputs",
    ),
    UdemyMathContract(
        "two_leg_sizing",
        "DYDX Pairs Trading Bot",
        "udl_628ccddcfd3fd33f",
        "About the Hedge Ratio",
        "course execution example allocates equal USD to each leg while hedge ratio defines the analytical spread",
        "active backtest uses hedge-ratio-normalized weights: w_y=signal/(1+abs(h)); w_x=-signal*h/(1+abs(h))",
        "INTENTIONAL_DIVERGENCE",
        "local_validated_estimator",
        "src/quant_platform/backtest.py",
        "returns from the course execution example cannot be compared directly with beta-neutral Math V2 returns",
        "make exposure_model mandatory in every backtest and compare equal-dollar as a separate ablation",
    ),
    UdemyMathContract(
        "costs",
        "DYDX Pairs Trading Bot",
        "udl_6e269f86e9f73e84",
        "Trading the Spread Based on Z-Score",
        "illustrative dashboard backtest explicitly excludes commissions",
        "active replay subtracts fees, slippage, funding and separately identified execution penalties",
        "INTENTIONAL_DIVERGENCE",
        "local_acceptance_required",
        "src/quant_platform/backtest.py",
        "gross course/dashboard return must never be compared with local net return as if cost assumptions matched",
        "store gross and every cost component separately; block acceptance when observed costs are unavailable",
    ),
    UdemyMathContract(
        "dynamic_spread",
        "Financial Econometrics",
        "udl_19fbdbdcb190a868",
        "Cointegration - Error Correction Model (ECM)",
        "course shows parameters and ECM behavior changing through time but does not define Crypto Wizards Dynamic internals",
        "local dynamic mode uses captured hedge series or rolling log-price OLS; experimental Kalman candidates remain parity research",
        "VENDOR_PARITY_BLOCKED",
        "research_only",
        "src/quant_platform/wizard_mode_replay.py;src/quant_platform/wizard_hyperliquid_mode_proof.py",
        "local Dynamic results cannot be called Crypto Wizards-equivalent",
        "require bounded custom-series parity before vendor_exact authority",
    ),
    UdemyMathContract(
        "ou_exact_mode",
        "DYDX Pairs Trading Bot",
        "udl_0c165c3c9c147410",
        "About Half-Life",
        "course explains half-life screening but does not disclose Crypto Wizards OU spread construction or optimal thresholds",
        "local OU is an AR(1)-validated estimator; vendor OU candidates remain proof-gated",
        "VENDOR_PARITY_BLOCKED",
        "research_only",
        "src/quant_platform/statistics/math_v2.py;src/quant_platform/wizard_hyperliquid_mode_proof.py",
        "local OU results cannot be called exact Wizard OU results",
        "keep OU Optimal as a vendor annotation and complete independent holdout parity evidence",
    ),
    UdemyMathContract(
        "copula_family_selection",
        "Financial Econometrics",
        "udl_42f43d33143bc940",
        "Copulas - Plan of Attack",
        "Gaussian, Frank, Clayton and Gumbel have different dependence and tail behavior; family should fit the data",
        "active local estimator implements Gaussian only",
        "NOT_IMPLEMENTED",
        "blocked_for_family_claims",
        "src/quant_platform/statistics/math_v2.py",
        "Gaussian-only research can miss asymmetric tail dependence",
        "add point-in-time family fitting and out-of-sample selection before claiming general copula coverage",
    ),
    UdemyMathContract(
        "hurst",
        "Machine Learning Applied to Trading",
        "udl_f000cd5e5d6e93b9",
        "Statistics - Testing for Market Efficiency Code Walkthrough",
        "course transcript gives H<0.5 mean reversion, H=0.5 randomness and H>0.5 trending, but delegates the estimator to an uncaptured Dynamic Hurst notebook",
        "Math V2 implements first-order detrended fluctuation analysis with minimum sample and scale gates",
        "SOURCE_FORMULA_NOT_CAPTURED",
        "local_validated_estimator_not_course_parity",
        "src/quant_platform/statistics/math_v2.py",
        "course interpretation matches, but numerical estimator parity cannot be asserted",
        "capture and hash the course notebook or retain DFA under its explicit method_id without Udemy parity claims",
    ),
    UdemyMathContract(
        "johansen",
        "Financial Econometrics",
        "udl_50240d5ed76ff3af",
        "Cointegration - Introduction",
        "course identifies Johansen as a second dashboard cointegration test but teaches Engle-Granger as the implemented test",
        "current active layer ingests the Wizard Johansen badge but has no local Math V2 Johansen estimator",
        "NOT_IMPLEMENTED",
        "vendor_observed_discovery_only",
        "src/quant_platform/wizard_evidence.py",
        "a green dashboard badge is not local Johansen verification",
        "implement trace and maximum-eigenvalue tests with lag/trend policy before local acceptance authority",
    ),
    UdemyMathContract(
        "garch",
        "Financial Econometrics",
        "udl_be1c5d5ea7e91299",
        "GARCH Asset Pairs Comparison Using CW (Optional)",
        "course explicitly warns that Crypto Wizards uses different and multiple GARCH models from the classroom symmetric/asymmetric examples",
        "current project captures Wizard GARCH fields but does not fit a local symmetric or asymmetric GARCH model",
        "VENDOR_PARITY_BLOCKED",
        "vendor_observed_discovery_only",
        "src/quant_platform/wizard_research_journal.py",
        "captured GARCH values cannot be recreated or treated as local volatility forecasts",
        "capture exact vendor model metadata or implement separately named local GARCH variants and validate out of sample",
    ),
    UdemyMathContract(
        "var_cvar",
        "Financial Econometrics",
        "udl_797ec683412c334f",
        "Copulas - Introduction",
        "course mentions copula simulation for VaR/CVaR but does not derive the dashboard implementation in the transcript",
        "current project ingests Wizard VaR/CVaR fields; no matching local simulation engine has production authority",
        "SOURCE_FORMULA_NOT_CAPTURED",
        "vendor_observed_discovery_only",
        "src/quant_platform/wizard_evidence.py;src/quant_platform/wizard_research_journal.py",
        "dashboard tail-risk values cannot be assumed to share confidence level, horizon, sample, or simulation method with local tests",
        "capture horizon/confidence/simulation settings and build a separately versioned local tail-risk estimator",
    ),
    UdemyMathContract(
        "spearman_kendall",
        "Financial Econometrics",
        "udl_7a8c8105ade12232",
        "Covariance vs Pearson's Correlation Coefficient",
        "the reviewed course derives Pearson; it does not provide a captured implementation contract for dashboard Spearman and Kendall fields",
        "current project preserves Wizard Spearman and Kendall values as vendor observations and documents their standard definitions",
        "OUTSIDE_CAPTURED_COURSE_FORMULA",
        "vendor_observed_discovery_only",
        "src/quant_platform/formula_registry.py;src/quant_platform/wizard_evidence.py",
        "field presence is not proof of matching window, return transform, tie handling, or missing-data policy",
        "capture dashboard calculation metadata or compute separately named local rank correlations with explicit conventions",
    ),
)


OUTPUT_COLUMNS = (
    "topic",
    "course",
    "lecture_id",
    "lecture_title",
    "transcript_status",
    "transcript_path",
    "transcript_sha256",
    "source_convention",
    "current_convention",
    "match_status",
    "production_authority",
    "implementation_path",
    "test_path",
    "historical_impact",
    "required_action",
)


def build_udemy_math_reconciliation(
    *,
    root: Path = ROOT,
    output_root: Path | None = None,
) -> CommandResult:
    """Write a source-to-code reconciliation without copying transcript text."""

    manifest_path = root / "reports" / "research" / "udemy_transcript_vault_manifest.csv"
    manifest = _manifest_by_lecture(manifest_path)
    rows: list[dict[str, object]] = []
    for contract in CONTRACTS:
        source = manifest.get(contract.lecture_id, {})
        row = asdict(contract)
        row.update(
            {
                "transcript_status": source.get("vault_status", "manifest_record_missing"),
                "transcript_path": source.get("transcript_path", ""),
                "transcript_sha256": source.get("transcript_sha256", ""),
            }
        )
        rows.append({column: row.get(column, "") for column in OUTPUT_COLUMNS})

    output_dir = (output_root or root) / "reports" / "audits"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "2026-08-15_udemy_math_to_code_reconciliation.csv"
    md_path = output_dir / "2026-08-15_udemy_math_to_code_reconciliation.md"
    _write_csv(csv_path, rows)
    status_counts = Counter(str(row["match_status"]) for row in rows)
    missing_sources = [
        str(row["lecture_id"])
        for row in rows
        if row["transcript_status"] != "captured"
    ]
    summary = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "contracts": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "source_records_missing_or_unavailable": missing_sources,
        "all_referenced_transcripts_captured": not missing_sources,
        "exact_wizard_parity": "blocked_for_dynamic_ou_and_copula_vendor_internals",
        "production_authority": "local_math_only; course evidence remains research_reference",
    }
    atomic_write_text(md_path, _markdown(rows, summary), encoding="utf-8")
    return CommandResult(
        paths={"reconciliation_csv": csv_path, "reconciliation_md": md_path},
        summary=summary,
    )


def _manifest_by_lecture(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            str(row.get("lecture_id", "")): row
            for row in csv.DictReader(handle)
            if row.get("lecture_id")
        }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=OUTPUT_COLUMNS)
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue(), encoding="utf-8")


def _markdown(rows: list[dict[str, object]], summary: dict[str, object]) -> str:
    status_counts = summary["status_counts"]
    lines = [
        "# Udemy Math-To-Code Reconciliation",
        "",
        (
            "This audit treats the Udemy material as educational source evidence, not trade authority. "
            "The courses contain multiple conventions, so a deliberate divergence is not mislabeled as a formula match."
        ),
        "",
        f"- contracts reviewed: {summary['contracts']}",
        f"- all referenced transcripts captured: `{str(summary['all_referenced_transcripts_captured']).lower()}`",
        f"- status counts: `{json.dumps(status_counts, sort_keys=True)}`",
        f"- exact Wizard parity: `{summary['exact_wizard_parity']}`",
        "",
        "## Reconciliation",
        "",
        "| Topic | Status | Current authority | Required action |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {topic} | {match_status} | {production_authority} | {required_action} |".format(
                **{key: str(value).replace("|", "\\|") for key, value in row.items()}
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            (
                "The active math is not a byte-for-byte copy of every course example, and it should not be. "
                "Exact matches are regression-tested; stronger production conventions are named; unresolved "
                "Crypto Wizards internals and unimplemented copula-family selection remain blocked."
            ),
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    print(json.dumps(build_udemy_math_reconciliation().summary, indent=2, sort_keys=True))
