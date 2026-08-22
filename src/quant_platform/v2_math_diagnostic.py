"""Executable Math V2.1 audit, incident ledger, and authority receipt."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import statsmodels

from quant_platform.economic_contract import ECONOMIC_CONTRACT_VERSION
from quant_platform.formula_registry import FORMULAS
from quant_platform.math_v2_acceptance import _run_checks
from quant_platform.orchestration.corrective_runtime import atomic_write_csv, atomic_write_text
from quant_platform.performance_math import MATH_VERSION
from quant_platform.runtime_types import CommandResult

ROOT = Path(__file__).resolve().parents[2]
AUDIT_SCHEMA_VERSION = "thewiz.v2_math_diagnostic.v2"
AUDIT_STEM = "2026-08-20_v2_math_diagnostic"
HISTORICAL_REEVALUATION_STEM = "2026-08-20_y_on_x_historical_reevaluation"
VENDOR_STATIC_PROOF_PATH = Path(
    "reports/active/hyperliquid_wizard_vendor_mode_proofs.csv"
)


def build_v2_math_diagnostic(
    *,
    root: Path = ROOT,
    source_root: Path | None = None,
    output_stem: str = AUDIT_STEM,
    vendor_static_proof_path: Path | None = None,
) -> CommandResult:
    """Write a complete, fail-closed mathematical audit without trading authority."""

    source_root = (source_root or root).resolve()
    vendor_static_proof_path = (
        vendor_static_proof_path or source_root / VENDOR_STATIC_PROOF_PATH
    ).resolve()
    output_dir = root / "reports" / "audits"
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "formula_inventory": output_dir / f"{output_stem}_formula_inventory.csv",
        "math_comparison": output_dir / f"{output_stem}_comparison.csv",
        "error_ledger": output_dir / f"{output_stem}_error_ledger.csv",
        "diagnostic_checks": output_dir / f"{output_stem}_checks.csv",
        "historical_impact": output_dir / f"{output_stem}_historical_impact.csv",
        "report": output_dir / f"{output_stem}.md",
        "authority": output_dir / f"{output_stem}_authority.json",
    }

    reevaluation = _historical_reevaluation_summary(root)
    inventory = _formula_inventory()
    comparison = _math_comparison()
    errors = _error_ledger()
    checks = _diagnostic_checks(
        source_root,
        vendor_static_proof_path=vendor_static_proof_path,
    )
    impact = _historical_impact(reevaluation)

    atomic_write_csv(inventory, paths["formula_inventory"], index=False)
    atomic_write_csv(comparison, paths["math_comparison"], index=False)
    atomic_write_csv(errors, paths["error_ledger"], index=False)
    atomic_write_csv(checks, paths["diagnostic_checks"], index=False)
    atomic_write_csv(impact, paths["historical_impact"], index=False)

    local_checks = checks.loc[
        checks["authority_domain"].isin({"local_core", "runtime", "source_contract"})
    ]
    local_pass = bool(not local_checks.empty and local_checks["status"].eq("PASS").all())
    vendor_static_pass = bool(
        checks.loc[checks["check_id"].eq("WIZARD_STATIC_EXACT_RECONSTRUCTION"), "status"]
        .eq("PASS")
        .all()
    )
    open_or_blocked = int(
        errors["status"].isin({"OPEN", "BLOCKED_VENDOR_UNKNOWN", "RERUN_REQUIRED"}).sum()
    )
    corrected_incidents = int(
        errors["status"].isin({"FIXED_THIS_AUDIT", "FIXED_PRIOR_AUDIT"}).sum()
    )
    core_check_count = len(local_checks)
    core_pass_count = int(local_checks["status"].eq("PASS").sum())
    summary: dict[str, object] = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "math_version": MATH_VERSION,
        "economic_contract_version": ECONOMIC_CONTRACT_VERSION,
        "local_core_status": "PASS" if local_pass else "BLOCKED",
        "local_core_checks_passed": core_pass_count,
        "local_core_checks_total": core_check_count,
        "wizard_static_reconstruction": "PASS_DIAGNOSTIC_ONLY" if vendor_static_pass else "BLOCKED",
        "wizard_dynamic_parity": "BLOCKED",
        "wizard_ou_parity": "BLOCKED",
        "wizard_copula_formula_parity": "BLOCKED",
        "wizard_performance_accounting_parity": "BLOCKED",
        "historical_canonical_replay": (
            "REEVALUATED_AND_SUPERSEDED" if reevaluation else "RERUN_REQUIRED"
        ),
        "historical_derived_outputs": "PARTIAL_RERUN_DESCENDANTS_STILL_BLOCKED",
        "historical_reevaluation": reevaluation or {},
        "model_authority": "RESEARCH_ONLY_BLOCKED_PENDING_REBUILD",
        "paper_or_live_authority": False,
        "corrected_incidents": corrected_incidents,
        "open_or_blocked_incidents": open_or_blocked,
        "formula_rows": len(inventory),
        "comparison_rows": len(comparison),
        "incident_rows": len(errors),
        "historical_impact_rows": len(impact),
        "runtime": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
        },
        "release_decision": "BLOCKED_PENDING_CONTROLLED_REGENERATION",
        "reason": (
            "Corrected local math passes its executable contract and the frozen canonical replay was "
            "re-evaluated, but all descendants still require controlled rebuilding and exact Wizard "
            "Dynamic, OU, Copula, and performance-accounting parity remain unproven."
        ),
    }
    atomic_write_text(paths["report"], _markdown_report(summary, inventory, comparison, errors, checks, impact), encoding="utf-8")
    authority = {
        **summary,
        "source_root": str(source_root),
        "input_hashes": _input_hashes(
            source_root,
            vendor_static_proof_path=vendor_static_proof_path,
        ),
        "artifacts": {
            key: {
                "path": _relative(path, root),
                "sha256": _file_hash(path),
            }
            for key, path in paths.items()
            if key != "authority"
        },
        "supersedes": [
            "reports/active/math_v2_acceptance.json when math_version=math-v2",
            "reports/audits/2026-08-15_full_math_revalidation.md local-core 19/19 conclusion",
        ],
        "promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    atomic_write_text(paths["authority"], json.dumps(authority, indent=2, sort_keys=True), encoding="utf-8")
    return CommandResult(paths=paths, summary=summary)


def _formula_inventory() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, contract in sorted(FORMULAS.items()):
        rows.append(
            _formula_row(
                formula_id=f"LOCAL_REGISTRY_{key.upper()}",
                topic=key,
                layer="local_formula_registry",
                formula=contract["formula"],
                authority="local_contract",
                point_in_time="depends_on_caller; acceptance requires causal training-only fit",
                orientation="see formula; Y-on-X where pair orientation applies",
                units="method_specific",
                wizard_comparison="not automatically equal to Crypto Wizards",
                reference_comparison="named local convention",
                known_limit=contract["failure_mode"],
                evidence_path="src/quant_platform/formula_registry.py",
                prevention_check="registry_coverage_check",
            )
        )

    rows.extend(
        [
            _formula_row(
                "LOCAL_EG_Y_ON_X",
                "engle_granger_orientation",
                "local_core",
                "log(Y_t)=alpha+beta_y_on_x*log(X_t)+e_t",
                "local_acceptance_estimator",
                "fit only on training rows",
                "X is regressor; Y is dependent variable",
                "log_price",
                "matches observed Wizard Static series_2-on-series_1 orientation",
                "matches active econometrics convention; differs from older raw X-on-Y course examples",
                "OLS direction is not symmetric; swapping assets creates a different setup",
                "src/quant_platform/statistics/math_v2.py",
                "engle_granger_recovers_y_on_x_contract",
            ),
            _formula_row(
                "LOCAL_EG_RESIDUAL",
                "equilibrium_residual",
                "local_core",
                "e_t=log(Y_t)-alpha-beta_y_on_x*log(X_t)",
                "local_acceptance_estimator",
                "alpha and beta frozen from training rows",
                "Y minus fitted Y from X",
                "log_spread",
                "matches reconstructed Wizard Static residual when log_used=true",
                "matches Financial Econometrics reference contract",
                "do not call the uncentered execution spread a fitted residual",
                "src/quant_platform/statistics/math_v2.py;src/quant_platform/economic_contract.py",
                "residual_identity_max_abs_error<1e-12",
            ),
            _formula_row(
                "LOCAL_EXECUTION_SPREAD",
                "execution_spread",
                "local_core",
                "s_t=log(Y_t)-beta_y_on_x*log(X_t)",
                "local_replay_series",
                "beta frozen or causally rolling; no future rows",
                "Y-on-X; omits constant alpha",
                "log_spread",
                "differs from Wizard raw Static residual by constant alpha",
                "z-scores are translation invariant; raw OU levels require explicit centering",
                "must remain named uncentered spread, not residual",
                "src/quant_platform/economic_contract.py",
                "execution_spread_and_residual_are_distinct",
            ),
            _formula_row(
                "LOCAL_INGESTED_SPREAD",
                "ingested_pair_spread",
                "local_ingestion_contract",
                "s_t=log(Y_t)-beta_y_on_x*log(X_t), only when an explicit finite positive hedge ratio is present",
                "research_input_only_until_fit_scope_is_proven",
                "operator fit scope or full-sample derivation is recorded explicitly",
                "Y-on-X",
                "log_spread",
                "provider spread remains provider evidence and is never silently reinterpreted",
                "matches the canonical uncentered local execution spread",
                "a missing hedge ratio does not default to 1.0 and cannot create a two-leg replay",
                "src/quant_platform/fixture_ingestion.py;src/quant_platform/dydx_candles.py",
                "LEGACY_INGESTION_SPREAD_REMOVED",
            ),
            _formula_row(
                "LOCAL_DERIVED_HEDGE_RATIO",
                "ingested_hedge_ratio",
                "local_ingestion_contract",
                "beta_y_on_x=cov(log(X),log(Y))/var(log(X))",
                "research_input_only_when_fit_on_full_sample",
                "full-sample imports are marked hindsight; acceptance refits inside each training fold",
                "X regressor; Y dependent variable",
                "elasticity",
                "not evidence of Wizard Dynamic, OU, or Copula parity",
                "same slope as OLS log(Y) on log(X) with an intercept",
                "nonpositive, nonfinite, or unidentified slopes fail instead of falling back to 1.0",
                "src/quant_platform/dydx_candles.py;src/quant_platform/economic_contract.py",
                "DYDX_INGESTED_HEDGE_FIT_Y_ON_X",
            ),
            _formula_row(
                "LEGACY_LEVEL_SPREAD_SUPERSEDED",
                "legacy_ingested_spread",
                "superseded_formula",
                "price_X-beta_x_on_y*price_Y, sometimes with beta=1.0",
                "none",
                "often full-sample or unspecified",
                "X-on-Y; opposite of the V2 pair identity",
                "price_units",
                "not comparable to the active Wizard Static log reconstruction",
                "a valid alternate spread only under a separately named level/X-on-Y contract",
                "unit-dependent, orientation-reversed, and previously allowed invented exposure",
                "historical fixture and dYdX-derived histories",
                "forbidden_source_pattern_scan",
            ),
            _formula_row(
                "MODE_STATIC_SPREAD",
                "Static (Spread)",
                "local_seven_mode_replay",
                "s_t=log(Y_t)-beta_y_on_x*log(X_t); metric_t=causal expanding zscore(s_t,ddof=1)",
                "local_research_replay_only",
                "captured beta; expanding statistics use only rows available by t",
                "Y-on-X; threshold action is validated against captured operator and leg labels",
                "dimensionless_sigma_metric",
                "not the same as Wizard full-sample Static custom-series output",
                "uses a causal normalization because captured Spread thresholds are sigma-scaled",
                "exact vendor live-safe fit scope is not proven",
                "src/quant_platform/wizard_mode_replay.py",
                "SEVEN_MODE_SHARED_ECONOMIC_CONTRACT",
            ),
            _formula_row(
                "MODE_STATIC_ZSCORER",
                "Static (ZScoreR)",
                "local_seven_mode_replay",
                "s_t=log(Y_t)-beta_y_on_x*log(X_t); z_t=(s_t-rolling_mean_w)/rolling_std_w(ddof=1)",
                "local_research_replay_only",
                "captured beta and window; trailing rows only",
                "Y-on-X; threshold action is validated against captured operator and leg labels",
                "dimensionless",
                "Static spread reconstruction is known diagnostically; live-safe upstream fit parity is not",
                "standard trailing sample z-score",
                "upstream captured beta fit scope may be unknown",
                "src/quant_platform/wizard_mode_replay.py",
                "SEVEN_MODE_SHARED_ECONOMIC_CONTRACT",
            ),
            _formula_row(
                "MODE_DYNAMIC_SPREAD",
                "Dyn (Spread)",
                "local_seven_mode_replay",
                "beta_t=causal rolling OLS log(Y) on log(X); s_t=log(Y_t)-beta_t*log(X_t); metric_t=causal expanding zscore(s_t)",
                "local_research_replay_only",
                "rolling beta is seeded from trailing training history and never from future rows",
                "Y-on-X with the same beta_t used for metric and sizing",
                "dimensionless_sigma_metric",
                "Wizard says Dynamic is Kalman-based; rolling OLS is not vendor parity",
                "valid local adaptive comparator",
                "nonpositive or missing beta_t blocks the opposing-leg contract",
                "src/quant_platform/wizard_mode_replay.py;src/quant_platform/orchestration/current_wizard_hyperliquid_replay.py",
                "SEVEN_MODE_SHARED_ECONOMIC_CONTRACT",
            ),
            _formula_row(
                "MODE_DYNAMIC_ZSCORER",
                "Dyn (ZScoreR)",
                "local_seven_mode_replay",
                "beta_t=causal rolling OLS log(Y) on log(X); s_t=log(Y_t)-beta_t*log(X_t); z_t=rolling_zscore_w(s_t,ddof=1)",
                "local_research_replay_only",
                "training-seeded rolling beta and trailing z-score window",
                "Y-on-X with the same beta_t used for signal and sizing",
                "dimensionless",
                "not exact Wizard Kalman/ZScoreR parity",
                "valid local adaptive comparator",
                "warm-up, state initialization, and vendor Kalman details remain non-equivalent",
                "src/quant_platform/wizard_mode_replay.py;src/quant_platform/orchestration/current_wizard_hyperliquid_replay.py",
                "SEVEN_MODE_SHARED_ECONOMIC_CONTRACT",
            ),
            _formula_row(
                "MODE_OU_SPREAD",
                "OU (Spread)",
                "local_seven_mode_replay",
                "s_t=log(Y_t)-beta_y_on_x*log(X_t); metric_t=s_t-captured_mu; captured_sigma scales thresholds",
                "local_research_replay_only",
                "captured beta, mu, and sigma; no test-window refit",
                "Y-on-X",
                "centered_log_spread",
                "Wizard says OU is MLE-based; terminal local holdout passed only 4/8 cells",
                "locally coherent conditional on captured parameters",
                "exact vendor transform, likelihood, and trend branch remain unknown",
                "src/quant_platform/wizard_mode_replay.py",
                "SEVEN_MODE_SHARED_ECONOMIC_CONTRACT",
            ),
            _formula_row(
                "MODE_OU_ZSCORER",
                "OU (ZScoreR)",
                "local_seven_mode_replay",
                "q_t=log(Y_t)-beta_y_on_x*log(X_t)-captured_mu; z_t=rolling_zscore_w(q_t,ddof=1)",
                "local_research_replay_only",
                "captured beta/mu/sigma and trailing z-score window",
                "Y-on-X",
                "dimensionless",
                "not exact Wizard OU/ZScoreR parity",
                "locally coherent conditional on captured parameters",
                "rolling z-score cannot repair unknown vendor OU estimation",
                "src/quant_platform/wizard_mode_replay.py",
                "SEVEN_MODE_SHARED_ECONOMIC_CONTRACT",
            ),
            _formula_row(
                "MODE_COPULA",
                "Copula",
                "local_seven_mode_replay",
                "metric_t is the explicitly selected u1_given_u2 or u2_given_u1 conditional; absent captured conditionals, a trailing Gaussian h-function is a local approximation",
                "local_research_replay_or_captured_behavior_only",
                "captured conditional series or trailing calibration only",
                "conditioned asset and lower/upper tail actions are explicit",
                "conditional_probability_0_to_1",
                "family selection and calibration are not certified as Wizard exact",
                "Gaussian h-function is valid only for the Gaussian family",
                "using the wrong directional view or family can reverse the trade",
                "src/quant_platform/wizard_mode_replay.py;src/quant_platform/statistics/math_v2.py",
                "SEVEN_MODE_SHARED_ECONOMIC_CONTRACT",
            ),
            _formula_row(
                "WIZARD_STATIC_OBSERVED",
                "wizard_static_spread",
                "vendor_reconstructed",
                "series_2-(alpha+beta*series_1), in levels or logs selected by log_used",
                "diagnostic_parity_only",
                "vendor response fits the entire submitted sample; hindsight",
                "series_1 is X; series_2 is Y",
                "level_or_log_spread",
                "10 captured Static mode rows reconstruct to <=1e-9",
                "not live authority because fit scope is full sample",
                "transform selector and future-safe fit policy are separate questions",
                "reports/active/hyperliquid_wizard_vendor_mode_proofs.csv",
                "WIZARD_STATIC_EXACT_RECONSTRUCTION",
            ),
            _formula_row(
                "WIZARD_FULL_ZSCORE_OBSERVED",
                "wizard_zscore",
                "vendor_reconstructed",
                "z_t=(spread_t-full_sample_mean)/full_sample_std(ddof=1)",
                "diagnostic_parity_only",
                "hindsight full-sample normalization",
                "inherits vendor spread orientation",
                "dimensionless",
                "reconstructed exactly on captured Static proofs",
                "must be replaced by train-frozen or causal normalization for local replay",
                "cannot be a live feature as captured",
                "src/quant_platform/wizard_hyperliquid_mode_proof.py",
                "full_sample_hindsight_never_live",
            ),
            _formula_row(
                "WIZARD_ROLLING_ZSCORE_OBSERVED",
                "wizard_zscore_roll",
                "vendor_reconstructed",
                "z_t=(spread_t-rolling_mean_w)/rolling_std_w(ddof=1); warmup=0",
                "diagnostic_parity_only",
                "rolling calculation; upstream spread fit may still be full sample",
                "inherits vendor spread orientation",
                "dimensionless",
                "captured roll_w values reconstruct exactly on Static proofs",
                "sample standard deviation matches local active convention",
                "rolling z-score alone does not remove spread-fit hindsight",
                "src/quant_platform/wizard_hyperliquid_mode_proof.py",
                "rolling_zscore_causality_and_upstream_fit_audit",
            ),
            _formula_row(
                "WIZARD_DYNAMIC_OFFICIAL",
                "wizard_dynamic_spread",
                "vendor_official_description",
                "time-varying hedge ratio estimated with a Kalman filter",
                "discovery_and_parity_research_only",
                "exact state initialization, covariance, update timing, and transform are undisclosed",
                "official docs do not expose enough detail for exact implementation",
                "level_or_log_spread_unknown",
                "local Kalman comparators have mixed or failed holdout evidence",
                "conceptually related, not certified numerically equal",
                "never label local rolling OLS or a guessed Kalman filter as vendor exact",
                "https://api.cryptowizards.net/docsv1beta/spread-post.mdx/",
                "WIZARD_DYNAMIC_REMAINS_BLOCKED",
            ),
            _formula_row(
                "WIZARD_OU_OFFICIAL",
                "wizard_ou_spread",
                "vendor_official_description",
                "OU parameters estimated by maximum likelihood",
                "discovery_and_parity_research_only",
                "exact transform, trend branch, likelihood, and threshold contract are undisclosed",
                "vendor-specific",
                "vendor_specific",
                "OU V6 terminal holdout passed only 4 of 8 cells",
                "local AR(1)-OU mapping is valid local math, not Wizard parity",
                "passing cells cannot be generalized after failed holdout",
                "https://api.cryptowizards.net/docsv1beta/spread-post.mdx/",
                "WIZARD_OU_REMAINS_BLOCKED",
            ),
            _formula_row(
                "WIZARD_COPULA_OFFICIAL",
                "wizard_copula",
                "vendor_official_description",
                "u1_given_u2=P(U1<=u1 | U2=u2), with a selected copula family",
                "behavioral_evidence_only",
                "vendor family selection and calibration history are not fully exposed",
                "conditioned asset must be explicit",
                "conditional_probability",
                "local Gaussian h-function is not universal vendor parity",
                "conditional CDF equation matches when family is Gaussian",
                "Clayton, Frank, Gumbel, and other families can reverse tail behavior",
                "https://api.cryptowizards.net/docsv1beta/copula-post.mdx/",
                "WIZARD_COPULA_FORMULA_REMAINS_BLOCKED",
            ),
            _formula_row(
                "LOCAL_TWO_LEG_WEIGHTS",
                "position_sizing",
                "local_core",
                "w_X=-signal*beta/(1+abs(beta)); w_Y=signal/(1+abs(beta))",
                "local_execution_and_backtest_contract",
                "signal shifted one bar before earning returns",
                "+1 means short X/long Y; -1 means long X/short Y",
                "gross_weight",
                "Wizard x_weighting is a separate vendor backtest control",
                "beta-normalized gross-one sizing is a declared local choice",
                "requires finite beta>0 or action labels no longer describe opposite legs",
                "src/quant_platform/economic_contract.py;src/quant_platform/backtest.py",
                "gross_one_opposite_sign_and_positive_beta_checks",
            ),
            _formula_row(
                "LOCAL_BAR_PNL",
                "gross_return",
                "local_core",
                "r_gross,t=w_X,t-1*r_X,t+w_Y,t-1*r_Y,t",
                "local_backtest_contract",
                "one-bar lag prevents same-close execution leakage",
                "weights follow canonical action contract",
                "simple_return",
                "vendor bar-level accounting is not published",
                "matches portfolio arithmetic, not a single-spread difference shortcut",
                "bad or missing prices and hedge ratios now fail closed",
                "src/quant_platform/backtest.py",
                "spread_convergence_has_correct_position_direction",
            ),
            _formula_row(
                "LOCAL_NET_RETURN",
                "net_return",
                "local_core",
                "r_net=r_gross-fees-slippage-funding-execution_risk-partial_fill",
                "local_backtest_contract",
                "costs accrue by turnover or held exposure as declared",
                "signed funding is position-aware; conservative funding uses absolute drag",
                "simple_return",
                "Wizard exact cost timing and funding treatment are not published",
                "course examples often omit costs and are not acceptance authority",
                "execution_risk remains reported separately and is not vendor slippage",
                "src/quant_platform/backtest.py;src/quant_platform/trade_ledger.py",
                "trade_factors_reconcile_to_equity",
            ),
            _formula_row(
                "LOCAL_EQUITY",
                "compounded_return",
                "local_core",
                "equity_t=product_{i<=t}(1+r_net,i); total_return=equity_T-1",
                "local_backtest_contract",
                "chronological compounding",
                "not orientation-specific",
                "equity_multiple",
                "Wizard total_return exact compounding contract is not exposed",
                "correct for simple returns; differs from adding returns",
                "any bar return <=-100% blocks ledger reconciliation",
                "src/quant_platform/trade_ledger.py;src/quant_platform/backtest.py",
                "trade_factors_reconcile_to_equity",
            ),
            _formula_row(
                "LOCAL_COST_TURNOVER",
                "fees_and_slippage",
                "local_core",
                "turnover=abs(delta w_X)+abs(delta w_Y); cost=turnover*bps/10000",
                "local_backtest_contract",
                "charged on each weight change, including entry, exit, and reversal",
                "per leg",
                "simple_return_drag",
                "Wizard commission/slippage input exists, but exact application timing is unknown",
                "declared local cost model",
                "must use current venue and notional-specific observed slippage for acceptance",
                "src/quant_platform/backtest.py",
                "cost_component_ledger_and_observed_cost_gate",
            ),
            _formula_row(
                "LOCAL_SHARPE",
                "sharpe",
                "local_core",
                "sqrt(periods_per_year)*mean(net simple returns)/std(net simple returns,ddof=1)",
                "local_ranking_metric",
                "interval must be known and agree with timestamp grid",
                "not orientation-specific",
                "dimensionless_annualized",
                "Wizard says exchange affects annual days, but its exact formula is not exposed",
                "zero risk-free rate and 365-day crypto calendar are declared local choices",
                "serial correlation, selection bias, and tails require separate inference",
                "src/quant_platform/performance_math.py",
                "sample_std_zero_variance_and_interval_consistency_checks",
            ),
            _formula_row(
                "LOCAL_MAX_DRAWDOWN",
                "drawdown",
                "local_core",
                "max_t((running_peak_t-equity_t)/running_peak_t), initial equity 1 included",
                "local_risk_metric",
                "full evaluated path",
                "not orientation-specific",
                "fraction_of_capital",
                "Wizard max_drawdown is negative in some API outputs; sign convention differs",
                "magnitude is the active local convention",
                "cannot infer future crash depth from historical drawdown",
                "src/quant_platform/backtest.py",
                "drawdown_includes_initial_capital",
            ),
            _formula_row(
                "LOCAL_MULTIPLE_TESTING",
                "multiple_testing",
                "local_statistical_validation",
                "Benjamini-Hochberg controls expected false-discovery proportion at q",
                "required_acceptance_control",
                "applied to the declared experiment family after OOS folds",
                "not orientation-specific",
                "adjusted_pvalue",
                "not a Wizard dashboard formula",
                "matches standard FDR procedure",
                "family definition must be frozen before testing",
                "src/quant_platform/statistical_validation.py",
                "benjamini_hochberg_unit_tests",
            ),
            _formula_row(
                "WIZARD_BACKTEST_ACCOUNTING_UNKNOWN",
                "wizard_backtest_performance",
                "vendor_unknown",
                "API exposes total_return, annual_return, Sharpe, Sortino, VaR, CVaR, win rate, and max drawdown",
                "discovery_and_hypothesis_only",
                "vendor implementation and trade ledger are not published",
                "mode and x_weighting dependent",
                "mixed",
                "cannot assert parity from matching summary values alone",
                "requires same prices, signals, sizing, costs, and trade lifecycle",
                "commission and slippage defaults vary across examples and captures",
                "https://api.cryptowizards.net/docsv1beta/backtest-post.mdx/",
                "WIZARD_PERFORMANCE_ACCOUNTING_REMAINS_BLOCKED",
            ),
            _formula_row(
                "DASHBOARD_COLOR_SEMANTICS",
                "stationarity_color_state",
                "observed_ui_semantics",
                "green/orange/uncolored states encode dashboard diagnostics observed by the operator",
                "discovery_and_interpretation_only",
                "snapshot-time UI state",
                "Engle-Granger and Johansen fields must remain separate",
                "categorical",
                "not a substitute for captured p-values, critical values, or trend model metadata",
                "operator interpretation, not an independently derived equation",
                "color alone cannot enter live features without a point-in-time field contract",
                "docs/crypto_wizards_operator_playbook.md",
                "capture_raw_fields_and_color_state_separately",
            ),
        ]
    )
    frame = pd.DataFrame(rows)
    return frame.sort_values(["topic", "layer", "formula_id"]).reset_index(drop=True)


def _math_comparison() -> pd.DataFrame:
    rows = [
        (
            "pair_orientation_and_hedge_ratio",
            "log(X) regressed on log(Y), with compensating caller swaps",
            "log(Y)=alpha+beta_y_on_x*log(X)+e; X is regressor and Y is dependent",
            "captured Static custom-series rows reconstruct as series_2 on series_1",
            "STATIC_DIAGNOSTIC_MATCH; OTHER_MODES_BLOCKED",
            "fit beta inside training folds; captured full-sample beta is hindsight",
            "old descendants superseded",
            "local_core_only",
        ),
        (
            "static_spread",
            "mixed X-beta*Y level spreads and a reversed local residual",
            "log(Y)-beta_y_on_x*log(X), with expanding causal sigma metric for local Spread replay",
            "vendor Static residual is Y-(alpha+beta*X), in level/log space selected by log_used",
            "PASS_DIAGNOSTIC_ONLY",
            "vendor captured fit is full-sample; local acceptance must be causal",
            "old fixture, dYdX, and replay derivatives superseded",
            "research_only_until_walk_forward",
        ),
        (
            "static_zscorer",
            "z-score could inherit reversed/unit-inconsistent spread; ddof varied in legacy paths",
            "trailing rolling mean/sample std(ddof=1) of canonical Y-on-X log spread",
            "captured zscore_roll reconstructs with roll_w, ddof=1, and zero warm-up on vendor spread",
            "UPSTREAM_LIVE_PARITY_BLOCKED",
            "trailing window is causal; captured upstream spread fit may not be",
            "rebuild every stale Static ZScoreR result",
            "local_research_only",
        ),
        (
            "dynamic_spread",
            "rolling beta began inside the test slice and missing warm-up exposure could become 1.0",
            "training-seeded causal rolling beta_y_on_x; same beta series drives metric and sizing",
            "official Dynamic description uses a Kalman filter; exact state/update contract is undisclosed",
            "BLOCKED_VENDOR_UNKNOWN",
            "training seed plus trailing updates only",
            "old Dynamic descendants superseded",
            "local_comparator_only",
        ),
        (
            "dynamic_zscorer",
            "rolling z-score was layered over an unseeded or defaulted dynamic spread",
            "rolling sample z-score(ddof=1) over training-seeded causal dynamic Y-on-X spread",
            "Kalman/ZScoreR implementation details remain undisclosed",
            "BLOCKED_VENDOR_UNKNOWN",
            "both beta and z-score windows are causal",
            "rebuild every stale Dynamic ZScoreR result",
            "local_comparator_only",
        ),
        (
            "ou_spread",
            "local AR(1)/OU approximation was sometimes treated as if it were Wizard exact",
            "canonical log spread centered by captured mu; captured sigma scales thresholds",
            "official OU description says MLE; exact transform, likelihood, and trend branch are unknown",
            "BLOCKED; TERMINAL_HOLDOUT_4_OF_8",
            "captured parameters only; no test-window refit",
            "retain failed holdout as negative evidence; rerun local descendants",
            "local_comparator_only",
        ),
        (
            "ou_zscorer",
            "rolling normalization could obscure a non-equivalent underlying OU fit",
            "rolling sample z-score(ddof=1) of canonical spread minus captured mu",
            "vendor OU/ZScoreR internals are not certified",
            "BLOCKED_VENDOR_UNKNOWN",
            "trailing normalization; captured parameter scope remains explicit",
            "rebuild every stale OU ZScoreR result",
            "local_comparator_only",
        ),
        (
            "copula",
            "Gaussian local conditionals could be overclaimed as general Wizard Copula parity",
            "use captured directional conditional when present; otherwise name trailing Gaussian h-function as local",
            "conditional views are documented, but selected family and calibration history are incomplete",
            "BLOCKED_VENDOR_UNKNOWN",
            "captured point-in-time conditional or trailing local calibration only",
            "retain behavioral captures; rebuild local Copula descendants",
            "behavioral_or_local_research_only",
        ),
        (
            "ecm",
            "expected gamma_x<0 and gamma_y>0 under a residual documented in the opposite direction",
            "for e=log(Y)-alpha-beta*log(X), expected correction is gamma_x>0 and gamma_y<0",
            "dashboard exposes ECMX, ECMY, and strength; exact estimator/significance contract is not proven",
            "LOCAL_VALID; VENDOR_PARITY_UNKNOWN",
            "fit only from rows available in the training fold",
            "regenerate ECM features, rankings, and models",
            "local_diagnostic_and_feature_only",
        ),
        (
            "two_leg_sizing_and_direction",
            "missing hedge ratios could become 1.0; beta<=0 could contradict long/short labels",
            "w_X=-signal*beta/(1+abs(beta)); w_Y=signal/(1+abs(beta)); finite beta>0 required",
            "Wizard x_weighting is a separate control; exact portfolio accounting is undisclosed",
            "LOCAL_CONTRACT_ONLY",
            "signal is shifted one bar before returns",
            "rerun all old two-leg results and labels",
            "local_core_only",
        ),
        (
            "returns_equity_and_drawdown",
            "some legacy paths added returns and omitted initial equity from drawdown",
            "equity=product(1+r_net); return=equity_T-1; drawdown includes initial equity 1",
            "summary fields exist, but exact vendor ledger and drawdown sign/timing are undisclosed",
            "LOCAL_VALID; VENDOR_PARITY_BLOCKED",
            "chronological bar ledger with one-bar-lagged weights",
            "old performance reports superseded",
            "local_core_only",
        ),
        (
            "sharpe",
            "population std, zero-variance 0.0, or mismatched timeframe labels occurred in legacy paths",
            "sqrt(periods/year)*mean(net returns)/sample std(ddof=1); degenerate or mismatched grids block",
            "Wizard Sharpe is exposed, but annualization, return stream, and cost timing are not fully published",
            "LOCAL_VALID; VENDOR_PARITY_BLOCKED",
            "declared interval must match observed regular timestamps",
            "rerank all stale reports",
            "local_core_only",
        ),
        (
            "costs_and_funding",
            "old comparisons could mix missing, default, or differently timed costs",
            "turnover-based fees/slippage/execution risk plus declared signed or conservative funding and partial fills",
            "commission/slippage controls exist, but exact application and funding timing are not certified",
            "VENDOR_PARITY_BLOCKED",
            "current venue evidence and bar timing required",
            "reattach observed Hyperliquid costs before acceptance",
            "local_costed_acceptance_only",
        ),
        (
            "ml_rl_student_labels",
            "duplicated PnL math and stale orientation could enter labels and rewards",
            "all two-leg labels reuse the canonical trade ledger and immutable math lineage",
            "Wizard outputs remain discovery inputs, not realized labels",
            "REBUILD_REQUIRED",
            "features at entry; labels strictly after entry; purged folds",
            "rebuild datasets, then retrain every learner",
            "blocked_pending_regeneration",
        ),
        (
            "ingestion_and_dispatch",
            "generic X-beta*Y spread, implicit beta=1.0, and price-only two-leg dispatch",
            "canonical Y-on-X log spread; explicit valid ratio required for derivation and two-leg accounting",
            "provider fields retain source semantics and are not silently converted into local execution authority",
            "FIXED_LOCAL_BOUNDARY",
            "full-sample derived ratio is labeled hindsight; incomplete rows are spread-only or blocked",
            "legacy normalized histories and labels superseded",
            "new_inputs_research_only_until_refit",
        ),
    ]
    return pd.DataFrame(
        rows,
        columns=(
            "component",
            "old_or_incorrect_local_math",
            "corrected_local_math",
            "crypto_wizards_math_or_evidence",
            "parity_status",
            "point_in_time_rule",
            "historical_disposition",
            "current_authority",
        ),
    )


def _formula_row(
    formula_id: str,
    topic: str,
    layer: str,
    formula: str,
    authority: str,
    point_in_time: str,
    orientation: str,
    units: str,
    wizard_comparison: str,
    reference_comparison: str,
    known_limit: str,
    evidence_path: str,
    prevention_check: str,
) -> dict[str, object]:
    return {
        "formula_id": formula_id,
        "topic": topic,
        "layer": layer,
        "formula": formula,
        "authority": authority,
        "point_in_time_contract": point_in_time,
        "orientation": orientation,
        "units": units,
        "wizard_comparison": wizard_comparison,
        "reference_comparison": reference_comparison,
        "known_limit": known_limit,
        "evidence_path": evidence_path,
        "prevention_check": prevention_check,
    }


def _error_ledger() -> pd.DataFrame:
    rows = [
        _incident(
            "MATH-001",
            "P0",
            "orientation",
            "math-v2 through 2026-08-20",
            "fit_engle_granger regressed log(X) on log(Y) while the registry and economic contract declared log(Y) on log(X).",
            "Hedge ratios, residuals, OU/Hurst/ECM inputs, and dependency features could describe the reciprocal setup.",
            "Coefficient-recovery and residual-identity diagnostics, plus static Wizard reconstruction evidence.",
            "FIXED_THIS_AUDIT",
            MATH_VERSION,
            "Regress Y on X, emit explicit orientation metadata, and use coint(Y,X).",
            "Rerun every derived artifact produced by the old estimator.",
            "replays;teacher rows;walk-forward;datasets;ML;RL;dashboards",
            "Known alpha/beta recovery and exact residual identity tests.",
            "src/quant_platform/statistics/math_v2.py",
        ),
        _incident(
            "MATH-002",
            "P0",
            "orientation",
            "compensating consumers before math-v2.1",
            "Replay, teacher, and Hyperliquid validation callers swapped price_y and price_x to compensate for the reversed estimator.",
            "Fixing only the estimator would have reversed these consumers a second time.",
            "Repository-wide fit_engle_granger call-site scan.",
            "FIXED_THIS_AUDIT",
            MATH_VERSION,
            "All consumers now call fit_engle_granger(price_x, price_y); forbidden swap scan added.",
            "Keep source-contract scan in every audit.",
            "current replay;teacher materializer;walk-forward validation",
            "WIZARD_CONSUMER_ARGUMENT_ORDER source check.",
            "src/quant_platform/orchestration",
        ),
        _incident(
            "MATH-003",
            "P0",
            "ecm",
            "math-v2 through 2026-08-20",
            "ECM support expected gamma_x<0 and gamma_y>0, signs consistent with the old X-on-Y residual but opposite the declared Y-on-X error term.",
            "ECM strength and leader/follower interpretation could be inverted.",
            "Economic sign derivation and controlled two-leg ECM process.",
            "FIXED_THIS_AUDIT",
            MATH_VERSION,
            "For e=Y-alpha-beta*X, require gamma_x>0 and gamma_y<0 when significant.",
            "Regenerate ECM fields and any features/rankings using them.",
            "ECM diagnostics;teacher features;pair ranking;model inputs",
            "Controlled ECM sign test with explicit residual formula metadata.",
            "src/quant_platform/statistics/math_v2.py",
        ),
        _incident(
            "MATH-004",
            "P0",
            "testing",
            "old 19-check Math V2 marker",
            "The cointegration test asserted only p<0.05 and never asserted alpha, beta, residual identity, orientation metadata, or ECM signs.",
            "A reversed estimator received a 19/19 PASS and the audit claimed the orientation was fixed.",
            "Adversarial review of what each test actually proved.",
            "FIXED_THIS_AUDIT",
            MATH_VERSION,
            "Acceptance now checks coefficient recovery, residual identity, orientation, ECM signs, direction, invalid beta, zero variance, and timestamp consistency.",
            "Treat prior 19/19 marker as superseded, not current evidence.",
            "reports/active/math_v2_acceptance.json;2026-08-15 audit conclusion",
            "Tests must prove semantic identities, not merely plausible significance.",
            "src/quant_platform/math_v2_acceptance.py",
        ),
        _incident(
            "MATH-005",
            "P0",
            "versioning",
            "math-v2",
            "Material formula changes reused the same math-v2 label.",
            "Stale and corrected artifacts were indistinguishable by version string.",
            "Lineage audit across performance, teacher, and dataset contracts.",
            "FIXED_THIS_AUDIT",
            MATH_VERSION,
            "Bumped math and economic-contract versions and all dependent settings/formula versions.",
            "Reject old-version markers in current adapters and rebuild all descendants.",
            "all math-v2 derived artifacts",
            "Exact version checks in teacher and acceptance adapters.",
            "src/quant_platform/performance_math.py;src/quant_platform/economic_contract.py",
        ),
        _incident(
            "MATH-006",
            "P0",
            "data_validation",
            "two-leg backtest before math-v2.1",
            "Missing or NaN hedge ratios silently became 1.0; missing prices were forward-filled.",
            "A test could complete with invented exposure or stale marks and look economically valid.",
            "Backtest source trace and teacher test-frame failure after removing the default.",
            "FIXED_THIS_AUDIT",
            MATH_VERSION,
            "Two-leg backtests require complete positive prices and a complete finite positive hedge_ratio column.",
            "Rerun old completed backtests; retain old outputs only as historical evidence.",
            "two-leg replay and labels",
            "Missing hedge ratio and bad-price fail-closed tests.",
            "src/quant_platform/backtest.py",
        ),
        _incident(
            "MATH-007",
            "P1",
            "economic_direction",
            "economic contract v1",
            "Negative or zero beta produced same-signed or undefined leg weights while action labels still said long/short pair.",
            "Reported direction and actual portfolio could disagree.",
            "Algebraic weight-sign audit.",
            "FIXED_THIS_AUDIT",
            ECONOMIC_CONTRACT_VERSION,
            "Canonical opposing-leg contract rejects beta<=0 or nonfinite beta.",
            "Research negative-beta relations only under a separately named contract.",
            "position weights;execution plans",
            "Positive-beta and gross-one weight invariants.",
            "src/quant_platform/economic_contract.py",
        ),
        _incident(
            "MATH-008",
            "P1",
            "performance",
            "math-v2",
            "Constant returns produced a valid Sharpe of 0.0 instead of an undefined/blocked ratio.",
            "Degenerate data could appear harmless and pass downstream finite-value checks.",
            "Zero-variance adversarial test.",
            "FIXED_THIS_AUDIT",
            MATH_VERSION,
            "Zero or nonfinite sample variance now blocks Sharpe and returns NaN.",
            "Regenerate metrics for degenerate paths.",
            "Sharpe fields and rankings",
            "zero_variance_sharpe_is_blocked.",
            "src/quant_platform/performance_math.py",
        ),
        _incident(
            "MATH-009",
            "P1",
            "timeframe",
            "math-v2",
            "A declared interval overrode timestamps without checking that the observed grid matched.",
            "Daily labels on hourly data could annualize Sharpe by 365 instead of 8760, or vice versa.",
            "Declared-daily/observed-hourly adversarial test.",
            "FIXED_THIS_AUDIT",
            MATH_VERSION,
            "When a regular timestamp grid is available, declared and inferred intervals must agree.",
            "Audit old rows where both interval and timestamps existed.",
            "Sharpe and all annualized metrics",
            "declared_interval_must_match_timestamps.",
            "src/quant_platform/performance_math.py",
        ),
        _incident(
            "MATH-010",
            "P0",
            "runtime",
            "Python 3.12 lock before 2026-08-20",
            "The lock selected SciPy 1.18 with statsmodels 0.14.6; statsmodels imported a SciPy helper removed in 1.18. The pytest script also used a stale Anaconda shebang.",
            "The estimator test suite could not import, and running bare pytest selected a different runtime than uv run python.",
            "Import smoke test and interpreter-path inspection.",
            "FIXED_THIS_AUDIT",
            "uv.lock current",
            "Pin SciPy <1.18 and invoke tests as uv run python -m pytest.",
            "Keep numerical import/version smoke check before diagnostics.",
            "all statistical diagnostics",
            "RUNTIME_NUMERICAL_IMPORTS and canonical command documentation.",
            "pyproject.toml;uv.lock",
        ),
        _incident(
            "MATH-011",
            "P0",
            "performance",
            "legacy and early V1 paths",
            "Some Sharpe paths used population standard deviation; active policy requires sample standard deviation.",
            "Sharpe was inflated, especially for small samples.",
            "2026-08-15 anti-pattern scan and deterministic sample-std comparison.",
            "FIXED_PRIOR_AUDIT",
            "math-v2",
            "Active Sharpe uses ddof=1; ddof=0 remains only in explicitly named parity/feature diagnostics.",
            "Rerun all legacy-derived rankings.",
            "legacy reports;rankings;models",
            "sharpe_uses_sample_standard_deviation.",
            "src/quant_platform/performance_math.py",
        ),
        _incident(
            "MATH-012",
            "P0",
            "drawdown",
            "legacy and early V1 paths",
            "Drawdown paths omitted initial equity 1.0 and could hide a first-bar loss.",
            "Maximum drawdown was understated.",
            "Immediate-loss fixture.",
            "FIXED_PRIOR_AUDIT",
            "math-v2",
            "Prepend initial equity before running peak calculation.",
            "Rerun all old drawdown and acceptance outputs.",
            "risk reports;acceptance;models",
            "drawdown_includes_initial_capital.",
            "src/quant_platform/backtest.py",
        ),
        _incident(
            "MATH-013",
            "P0",
            "returns",
            "legacy and early V1 paths",
            "Some paths added simple returns rather than compounding them.",
            "Total return, drawdown, labels, and RL rewards could disagree with realizable equity.",
            "Ledger reconciliation against product(1+r).",
            "FIXED_PRIOR_AUDIT",
            "math-v2",
            "Canonical ledger compounds simple returns and blocks bar loss <=-100%.",
            "Regenerate all descendants from canonical ledger.",
            "backtests;labels;ML;RL;scoreboards",
            "trade_factors_reconcile_to_equity.",
            "src/quant_platform/trade_ledger.py",
        ),
        _incident(
            "MATH-014",
            "P0",
            "labels",
            "V1 ML builder",
            "ML labels duplicated hedge/beta PnL math instead of using the canonical ledger.",
            "Training labels could disagree with the backtest being modeled.",
            "2026-08-15 code-path comparison.",
            "FIXED_PRIOR_AUDIT",
            "math-v2",
            "ML label generation reuses canonical two-leg trade ledger.",
            "Rebuild and rehash all training datasets and models.",
            "trade datasets;trade gate;student;RL",
            "label-to-ledger reconciliation and lineage hashes.",
            "src/quant_platform/ml_filter.py;src/quant_platform/trade_ledger.py",
        ),
        _incident(
            "MATH-015",
            "P1",
            "ou",
            "early Math V2",
            "OU half-life accepted invalid or nonpositive delta_t.",
            "A plausible half-life could be produced with meaningless time units.",
            "Nonpositive-delta adversarial test.",
            "FIXED_PRIOR_AUDIT",
            "math-v2",
            "OU fit blocks nonfinite or nonpositive delta_t.",
            "Every OU report must include delta_t units.",
            "OU half-life fields",
            "ou_rejects_nonpositive_time_step.",
            "src/quant_platform/statistics/math_v2.py",
        ),
        _incident(
            "MATH-016",
            "P0",
            "vendor_parity",
            "all versions",
            "Local Dynamic implementations do not have certified exact Crypto Wizards parameter/update parity.",
            "A local signal could be mislabeled as the dashboard's exact mode.",
            "Captured holdouts and mismatch rows.",
            "BLOCKED_VENDOR_UNKNOWN",
            "none",
            "Keep local Dynamic labeled local_validated_estimator or research comparator only.",
            "Obtain fresh preregistered holdout evidence before any exact claim.",
            "Dynamic mode parity",
            "vendor_formula_parity_status must be exact on untouched holdouts.",
            "reports/active/hyperliquid_wizard_vendor_mode_proofs.csv",
        ),
        _incident(
            "MATH-017",
            "P0",
            "vendor_parity",
            "all versions",
            "Local OU AR(1) math is valid locally but does not reproduce Wizard OU generally; terminal V6 failed 4 of 8 cells.",
            "Using local OU as vendor exact would be an unsupported semantic substitution.",
            "Registered OU holdouts and terminal closure.",
            "BLOCKED_VENDOR_UNKNOWN",
            "none",
            "Preserve local/Wizard namespaces and terminal rejection.",
            "A new hypothesis requires a new preregistered generation and untouched holdout.",
            "OU mode parity",
            "WIZARD_OU_REMAINS_BLOCKED.",
            "reports/active/wizard_ou_v6_terminal_closure.json",
        ),
        _incident(
            "MATH-018",
            "P0",
            "vendor_parity",
            "all versions",
            "Wizard Copula family selection/calibration is not reconstructed; local code fits only Gaussian copulas unless captured conditionals are supplied.",
            "Tail probabilities and trade directions can differ by family and directional view.",
            "Official docs plus behavioral captures with no formula proof.",
            "BLOCKED_VENDOR_UNKNOWN",
            "none",
            "Require explicit family, fitted parameters, conditional view, and point-in-time calibration metadata.",
            "Implement and cross-validate non-Gaussian families as local methods, not guessed vendor parity.",
            "Copula modes and features",
            "WIZARD_COPULA_FORMULA_REMAINS_BLOCKED.",
            "data/raw/crypto_wizards_copula_behavioral_proofs",
        ),
        _incident(
            "MATH-019",
            "P0",
            "hindsight",
            "captured Wizard custom-series proofs",
            "Static spread and full z-score reconstruction use the entire supplied sample.",
            "Exact historical parity can leak future data if copied into live or walk-forward features.",
            "Raw request/response reconstruction and formula scope review.",
            "OPEN",
            "none",
            "Use captured full-sample math only as parity evidence; refit alpha/beta and normalization inside training folds.",
            "Keep point-in-time flags and block dashboard hindsight from live features.",
            "Wizard static captures;features;labels",
            "full_sample_hindsight_never_live.",
            "data/raw/crypto_wizards_custom_series_proofs",
        ),
        _incident(
            "MATH-020",
            "P0",
            "vendor_performance",
            "all versions",
            "Wizard exposes performance outputs but not complete bar accounting, trade lifecycle, annualization, funding, or cost timing formulas.",
            "Matching Sharpe or return cannot prove strategy parity and may compare different economics.",
            "Official backtest schema and local ledger comparison.",
            "BLOCKED_VENDOR_UNKNOWN",
            "none",
            "Use Wizard backtest as hypothesis evidence; local costed point-in-time replay remains acceptance authority.",
            "Capture every setting and compare trade paths if the vendor exposes them later.",
            "Wizard/local backtest comparisons",
            "WIZARD_PERFORMANCE_ACCOUNTING_REMAINS_BLOCKED.",
            "https://api.cryptowizards.net/docsv1beta/backtest-post.mdx/",
        ),
        _incident(
            "MATH-021",
            "P0",
            "historical_lineage",
            "all pre-math-v2.1 derived outputs",
            "Historical outputs were generated under one or more corrected math, orientation, defaulting, or versioning defects.",
            "They cannot support current ranking, training, paper, or live decisions.",
            "Dependency graph and artifact-index review.",
            "RERUN_REQUIRED",
            MATH_VERSION,
            "Retain raw evidence; supersede every derived output with immutable math-v2.1 receipts.",
            "Run controlled regeneration in dependency order and compare against frozen prior outputs.",
            "replays;rankings;teachers;datasets;ML;RL;dashboards;paper preflight",
            "math_version and input hashes required in every descendant.",
            "reports/audits/2026-08-20_v2_math_diagnostic_historical_impact.csv",
        ),
        _incident(
            "MATH-022",
            "P0",
            "dynamic_hedge_ratio",
            "current replay settings through v3",
            "Dynamic modes estimated rolling beta from the held-out test slice alone, leaving the initial window undefined; older backtests then defaulted those missing exposures to beta=1.",
            "Dynamic signals and sizing could disagree, and early held-out bars could use invented exposure.",
            "Fail-closed hedge-ratio validation exposed the warm-up values in canonical and observed-cost replay tests.",
            "FIXED_THIS_AUDIT",
            "current_local_standardized_math_v2.v4_y_on_x",
            "Seed causal rolling Y-on-X beta with only the trailing training history, attach the resulting point-in-time series, and use the same series for signals and sizing.",
            "Regenerate every Dynamic descendant and retain old output only for failure attribution.",
            "Dynamic replay;observed-cost replay;walk-forward descendants;ML/RL labels",
            "Dynamic tests require finite point-in-time beta from the first evaluable bar and reject nonpositive beta regimes explicitly.",
            "src/quant_platform/orchestration/current_wizard_hyperliquid_replay.py",
        ),
        _incident(
            "MATH-023",
            "P0",
            "ingestion_orientation",
            "fixture and dYdX-derived history builders before this audit",
            "Legacy ingestion reconstructed generic spread as price_X-beta*price_Y and could substitute beta=1.0, while the active contract is log(Y)-beta_y_on_x*log(X).",
            "The same pair could enter research with different units, orientation, and direction depending on its source path.",
            "Repository-wide spread-construction scan after the full-suite boundary failure.",
            "FIXED_THIS_AUDIT",
            MATH_VERSION,
            "New derived histories use the canonical Y-on-X log spread; fixture normalization requires an explicit valid hedge ratio and never invents 1.0.",
            "Regenerate all legacy normalized fixtures and dYdX-derived histories before comparing them with V2 outputs.",
            "normalized enrichment fixtures;dYdX-derived histories;their strategy reports and descendants",
            "Source scan forbids the old formulas and deterministic ingestion tests assert orientation and provenance.",
            "src/quant_platform/fixture_ingestion.py;src/quant_platform/dydx_candles.py",
        ),
        _incident(
            "MATH-024",
            "P0",
            "backtest_dispatch",
            "experiment, threshold-sweep, and ML-label dispatch before this audit",
            "Several callers selected two-leg accounting when price_x and price_y existed even if hedge_ratio was absent.",
            "After fail-closed sizing was repaired, those paths crashed; before that repair they could silently use the old 1.0 default.",
            "Complete test-suite execution after enabling strict two-leg input validation.",
            "FIXED_THIS_AUDIT",
            MATH_VERSION,
            "Two-leg dispatch now requires price_x, price_y, and hedge_ratio; otherwise an explicit spread-only research path is used or the row is blocked.",
            "Rebuild affected fixture experiments and ML candidate labels.",
            "fixture experiments;z-score threshold sweeps;ML trade datasets",
            "TWO_LEG_DISPATCH_REQUIRES_HEDGE_RATIO source check and boundary tests.",
            "src/quant_platform/experiments.py;src/quant_platform/ml_filter.py;src/quant_platform/cli.py",
        ),
    ]
    return pd.DataFrame(rows)


def _incident(
    incident_id: str,
    severity: str,
    category: str,
    affected_versions: str,
    wrong_or_inconsistent_math: str,
    impact: str,
    detection: str,
    status: str,
    fixed_in: str,
    fix: str,
    remaining_action: str,
    invalidated_artifacts: str,
    prevention_control: str,
    evidence_path: str,
) -> dict[str, object]:
    return {
        "incident_id": incident_id,
        "severity": severity,
        "category": category,
        "affected_versions": affected_versions,
        "wrong_or_inconsistent_math": wrong_or_inconsistent_math,
        "impact": impact,
        "detection": detection,
        "status": status,
        "fixed_in": fixed_in,
        "fix": fix,
        "remaining_action": remaining_action,
        "invalidated_artifacts": invalidated_artifacts,
        "prevention_control": prevention_control,
        "evidence_path": evidence_path,
    }


def _diagnostic_checks(
    source_root: Path,
    *,
    vendor_static_proof_path: Path | None = None,
) -> pd.DataFrame:
    reconciliation, statistical = _run_checks()
    rows: list[dict[str, object]] = []
    for domain, frame in (("local_core", reconciliation), ("local_core", statistical)):
        for record in frame.to_dict("records"):
            rows.append(
                _check_row(
                    f"CORE_{str(record['check']).upper()}",
                    domain,
                    str(record["status"]),
                    record["observed"],
                    record["required"],
                    "local core validity",
                    "src/quant_platform/math_v2_acceptance.py",
                )
            )

    runtime_ok = _version_tuple(scipy.__version__) < (1, 18)
    rows.append(
        _check_row(
            "RUNTIME_NUMERICAL_IMPORTS",
            "runtime",
            "PASS" if runtime_ok else "BLOCKED",
            f"numpy={np.__version__};pandas={pd.__version__};scipy={scipy.__version__};statsmodels={statsmodels.__version__}",
            "all imports succeed; scipy<1.18 while statsmodels 0.14.6 is locked",
            "no diagnostic is valid if the numerical stack cannot import",
            "pyproject.toml;uv.lock",
        )
    )

    forbidden_calls = (
        'fit_engle_granger(train["price_y"], train["price_x"])',
        'fit_engle_granger(frame["price_y"], frame["price_x"])',
    )
    consumer_files = (
        "src/quant_platform/orchestration/current_wizard_hyperliquid_replay.py",
        "src/quant_platform/orchestration/teacher_evidence_materializer.py",
        "src/quant_platform/orchestration/hyperliquid_research_validation.py",
    )
    observed_forbidden: list[str] = []
    for relative in consumer_files:
        text = _read_text(source_root / relative)
        observed_forbidden.extend(
            f"{relative}:{pattern}" for pattern in forbidden_calls if pattern in text
        )
    rows.append(
        _check_row(
            "WIZARD_CONSUMER_ARGUMENT_ORDER",
            "source_contract",
            "PASS" if not observed_forbidden else "BLOCKED",
            ";".join(observed_forbidden) or "no compensating Y/X argument swaps",
            "all canonical consumers call fit_engle_granger(price_x, price_y)",
            "prevents a second reversal after estimator fixes",
            ";".join(consumer_files),
        )
    )

    backtest_text = _read_text(source_root / "src/quant_platform/backtest.py")
    fail_closed = (
        'required = {"price_x", "price_y", "hedge_ratio"}' in backtest_text
        and ".ffill()" not in backtest_text
        and "two-leg prices must be complete, finite, and positive" in backtest_text
    )
    rows.append(
        _check_row(
            "TWO_LEG_INPUTS_FAIL_CLOSED",
            "source_contract",
            "PASS" if fail_closed else "BLOCKED",
            "explicit hedge ratio; no forward-fill; positive-price guard"
            if fail_closed
            else "guard missing",
            "no invented 1.0 hedge ratio and no forward-filled prices",
            "prevents plausible metrics from fabricated exposures or stale marks",
            "src/quant_platform/backtest.py",
        )
    )

    fixture_text = _read_text(source_root / "src/quant_platform/fixture_ingestion.py")
    dydx_text = _read_text(source_root / "src/quant_platform/dydx_candles.py")
    legacy_spread_patterns = (
        'normalized["price_x"] - hedge_coeff * normalized["price_y"]',
        "spread = price_x - final_hedge_ratio * price_y",
    )
    legacy_spreads = [
        pattern
        for pattern in legacy_spread_patterns
        if pattern in fixture_text or pattern in dydx_text
    ]
    ingestion_contract_ok = (
        not legacy_spreads
        and "y_on_x_log_spread" in fixture_text
        and "y_on_x_log_spread" in dydx_text
        and "not_derived_missing_valid_hedge_ratio" in fixture_text
    )
    rows.append(
        _check_row(
            "LEGACY_INGESTION_SPREAD_REMOVED",
            "source_contract",
            "PASS" if ingestion_contract_ok else "BLOCKED",
            ";".join(legacy_spreads) or "canonical Y-on-X log constructors only",
            "no generic X-beta*Y spread and no missing-ratio fallback",
            "prevents source-dependent orientation, units, and invented exposures",
            "src/quant_platform/fixture_ingestion.py;src/quant_platform/dydx_candles.py",
        )
    )

    experiments_text = _read_text(source_root / "src/quant_platform/experiments.py")
    ml_text = _read_text(source_root / "src/quant_platform/ml_filter.py")
    cli_text = _read_text(source_root / "src/quant_platform/cli.py")
    dispatch_ok = (
        'two_leg_inputs = {"price_x", "price_y", "hedge_ratio"}' in experiments_text
        and 'if {"price_x", "price_y", "hedge_ratio"}.issubset(data.columns)' in ml_text
        and 'if not {"price_x", "price_y", "hedge_ratio", "zscore"}.issubset(frame.columns)'
        in cli_text
    )
    rows.append(
        _check_row(
            "TWO_LEG_DISPATCH_REQUIRES_HEDGE_RATIO",
            "source_contract",
            "PASS" if dispatch_ok else "BLOCKED",
            "all audited dispatchers require price_x, price_y, and hedge_ratio"
            if dispatch_ok
            else "one or more dispatchers still select incomplete two-leg inputs",
            "no two-leg backtest or ML label dispatch without an explicit hedge ratio",
            "keeps incomplete records research-only and prevents old default behavior",
            "src/quant_platform/experiments.py;src/quant_platform/ml_filter.py;src/quant_platform/cli.py",
        )
    )

    dydx_fit_ok = (
        "hedge_ratio = y_on_x_beta(" in dydx_text
        and '"derived_log_y_on_log_x_ols"' in dydx_text
        and '"full_sample_hindsight_research_only"' in dydx_text
    )
    rows.append(
        _check_row(
            "DYDX_INGESTED_HEDGE_FIT_Y_ON_X",
            "source_contract",
            "PASS" if dydx_fit_ok else "BLOCKED",
            "log(Y)-on-log(X) OLS with explicit hindsight label"
            if dydx_fit_ok
            else "contract missing",
            "derived history hedge ratio uses beta_y_on_x and records fit scope",
            "prevents reciprocal coefficients and hidden full-sample authority",
            "src/quant_platform/dydx_candles.py",
        )
    )

    mode_replay_text = _read_text(source_root / "src/quant_platform/wizard_mode_replay.py")
    economic_text = _read_text(source_root / "src/quant_platform/economic_contract.py")
    mode_names = (
        "Static (Spread)",
        "Static (ZScoreR)",
        "Dyn (Spread)",
        "Dyn (ZScoreR)",
        "OU (Spread)",
        "OU (ZScoreR)",
        "Copula",
    )
    mode_contract_ok = (
        all(name in mode_replay_text for name in mode_names)
        and "y_on_x_log_spread" in mode_replay_text
        and "rolling_y_on_x_beta" in mode_replay_text
        and "normalized_two_leg_weights" in economic_text
        and "copula_direction_view" in mode_replay_text
    )
    rows.append(
        _check_row(
            "SEVEN_MODE_SHARED_ECONOMIC_CONTRACT",
            "source_contract",
            "PASS" if mode_contract_ok else "BLOCKED",
            "seven modes share explicit orientation, direction, and sizing helpers"
            if mode_contract_ok
            else "one or more modes bypass the shared contract",
            "all seven modes preserve pair identity and route actions through one economic contract",
            "prevents mode-specific sign, asset-order, or sizing drift",
            "src/quant_platform/wizard_mode_replay.py;src/quant_platform/economic_contract.py",
        )
    )

    proof_path = (
        vendor_static_proof_path or source_root / VENDOR_STATIC_PROOF_PATH
    ).resolve()
    if proof_path.exists():
        proofs = pd.read_csv(proof_path, low_memory=False)
        exact = proofs.loc[
            proofs.get("exact_mode", pd.Series("", index=proofs.index))
            .astype(str)
            .str.startswith("Static")
            & proofs.get("vendor_formula_parity_status", pd.Series("", index=proofs.index))
            .astype(str)
            .eq("exact_reconstruction")
        ]
        error_columns = [
            "vendor_spread_max_abs_error",
            "vendor_zscore_max_abs_error",
            "vendor_zscore_roll_max_abs_error",
        ]
        max_error = max(
            (
                float(pd.to_numeric(exact[column], errors="coerce").max())
                for column in error_columns
                if column in exact and not exact.empty
            ),
            default=float("inf"),
        )
        orientation_ok = bool(
            not exact.empty
            and exact["vendor_spread_formula"]
            .astype(str)
            .str.contains("y_on_.*x|y-on-.*x|log_y_on_log_x|ols_y_on_x", regex=True)
            .all()
        )
        static_pass = len(exact) >= 10 and max_error <= 1e-9 and orientation_ok
        observed = f"exact_rows={len(exact)};max_abs_error={max_error};y_on_x={orientation_ok}"
    else:
        static_pass = False
        observed = "proof file missing"
    rows.append(
        _check_row(
            "WIZARD_STATIC_EXACT_RECONSTRUCTION",
            "vendor_parity",
            "PASS" if static_pass else "BLOCKED",
            observed,
            ">=10 untouched Static proof rows; all spread/z-score errors <=1e-9; series_2-on-series_1",
            "diagnostic parity only; no live authority",
            "reports/active/hyperliquid_wizard_vendor_mode_proofs.csv",
        )
    )
    for check_id, observed, evidence in (
        (
            "WIZARD_DYNAMIC_REMAINS_BLOCKED",
            "exact Kalman state/update/transform contract not certified",
            "reports/active/hyperliquid_wizard_vendor_mode_proofs.csv",
        ),
        (
            "WIZARD_OU_REMAINS_BLOCKED",
            "terminal OU V6 passed 4/8 and failed 4/8 holdout cells",
            "reports/active/wizard_ou_v6_terminal_closure.json",
        ),
        (
            "WIZARD_COPULA_FORMULA_REMAINS_BLOCKED",
            "behavioral responses captured; formula/family calibration proof unavailable",
            "data/raw/crypto_wizards_copula_behavioral_proofs",
        ),
        (
            "WIZARD_PERFORMANCE_ACCOUNTING_REMAINS_BLOCKED",
            "summary fields exposed; complete trade ledger and cost timing undisclosed",
            "https://api.cryptowizards.net/docsv1beta/backtest-post.mdx/",
        ),
    ):
        rows.append(
            _check_row(
                check_id,
                "vendor_parity",
                "EXPECTED_BLOCK",
                observed,
                "must remain blocked until preregistered untouched evidence passes",
                "prevents local approximation from being promoted as vendor exact",
                evidence,
            )
        )
    return pd.DataFrame(rows)


def _check_row(
    check_id: str,
    authority_domain: str,
    status: str,
    observed: object,
    required: object,
    authority_effect: str,
    evidence_path: str,
) -> dict[str, object]:
    return {
        "math_version": MATH_VERSION,
        "economic_contract_version": ECONOMIC_CONTRACT_VERSION,
        "check_id": check_id,
        "authority_domain": authority_domain,
        "status": status,
        "observed": observed,
        "required": required,
        "authority_effect": authority_effect,
        "evidence_path": evidence_path,
    }


def _historical_impact(reevaluation: dict[str, object] | None) -> pd.DataFrame:
    replay_status = "REEVALUATED_AND_SUPERSEDED" if reevaluation else "RERUN_REQUIRED"
    replay_reason = (
        "frozen-input comparison confirmed "
        f"{reevaluation.get('replay_status_changes', 0)} status changes, "
        f"{reevaluation.get('rows_with_metric_changes', 0)} metric changes, and "
        f"{reevaluation.get('eligibility_lost', 0)} lost eligibility cells"
        if reevaluation
        else "orientation, defaults, Sharpe, drawdown, or ledger changes can alter results"
    )
    rows = [
        (
            "raw Crypto Wizards API/dashboard captures",
            "RETAIN",
            "vendor observations are not changed by local formula repair",
            "preserve hashes, timestamps, settings, and hindsight flags",
            "discovery_and_parity_only",
        ),
        (
            "raw Hyperliquid candles/funding/L2",
            "RETAIN_WITH_QUALITY_CHECK",
            "source observations are not changed by downstream formulas",
            "preserve hashes; recheck alignment, freshness, and completeness",
            "eligible_input_only",
        ),
        (
            "research papers, YouTube, and Udemy source vaults",
            "RETAIN_REFERENCE",
            "reference evidence is not a calculated trading result",
            "keep provenance; never use prose/transcripts as outcome labels",
            "research_reference_only",
        ),
        (
            "old math-v2 acceptance marker",
            "SUPERSEDED",
            "19/19 did not test orientation coefficients/residual identity/ECM signs",
            "replace active marker with math-v2.1 output",
            "no_current_authority",
        ),
        (
            "pre-math-v2.1 canonical replay metrics",
            replay_status,
            replay_reason,
            "use corrected receipt for failure attribution; rebuild all descendants",
            "failure_attribution_only",
        ),
        (
            "legacy normalized fixtures and dYdX-derived histories",
            "SUPERSEDE_AND_REBUILD",
            "generic spread could use X-beta*Y, level units, or an implicit beta of 1.0",
            "re-normalize from retained raw observations under the Y-on-X log contract and preserve fit-scope metadata",
            "old_rows_failure_attribution_only",
        ),
        (
            "strategy/regime/ablation/ranking reports",
            "RERUN_REQUIRED",
            "derived from stale replay metrics",
            "rebuild only from math-v2.1 receipts",
            "none_until_rerun",
        ),
        (
            "teacher council and walk-forward rows",
            "RERUN_REQUIRED",
            "dependency orientation and ECM semantics changed",
            "rematerialize all seven modes and critics",
            "none_until_rerun",
        ),
        (
            "trade training datasets",
            "SUPERSEDE_AND_REBUILD",
            "features and labels can inherit stale orientation and PnL",
            "new immutable dataset ID, leakage audit, math version, and source hashes",
            "old_dataset_research_only",
        ),
        (
            "ML, RL, shadow, and student-teacher models",
            "RETRAIN_REQUIRED",
            "trained or evaluated on stale descendants",
            "retrain after dataset rebuild; repeat purged OOS acceptance",
            "research_only_blocked",
        ),
        (
            "dashboard rankings and paper/live decision rows",
            "RECOMPUTE_BEFORE_USE",
            "displayed scores/actions may descend from stale artifacts",
            "rebuild after all upstream gates; expose blockers",
            "no_order_authority",
        ),
        (
            "Wizard Static exact reconstruction captures",
            "RETAIN_DIAGNOSTIC",
            "formula reconstruction remains valid historical evidence",
            "keep full-sample hindsight restriction",
            "diagnostic_only",
        ),
        (
            "Wizard Dynamic/OU/Copula negative or partial proofs",
            "RETAIN_NEGATIVE_EVIDENCE",
            "failed and unavailable proofs prevent repeated overclaims",
            "do not relabel passing subsets as general parity",
            "blocked_vendor_parity",
        ),
    ]
    return pd.DataFrame(
        rows,
        columns=(
            "artifact_class",
            "status",
            "reason",
            "required_action",
            "authority_after_audit",
        ),
    )


def _historical_reevaluation_summary(root: Path) -> dict[str, object] | None:
    path = root / "reports" / "audits" / f"{HISTORICAL_REEVALUATION_STEM}_manifest.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("input_hashes_equal") is not True:
        return None
    keys = (
        "old_canonical_replay_id",
        "new_canonical_replay_id",
        "experiments_compared",
        "replay_status_changes",
        "rows_with_metric_changes",
        "eligibility_gained",
        "eligibility_lost",
        "old_trade_ledger_rows",
        "new_trade_ledger_rows",
    )
    return {key: payload.get(key) for key in keys}


def _markdown_report(
    summary: dict[str, object],
    inventory: pd.DataFrame,
    comparison: pd.DataFrame,
    errors: pd.DataFrame,
    checks: pd.DataFrame,
    impact: pd.DataFrame,
) -> str:
    incident_table = errors.loc[
        :,
        [
            "incident_id",
            "severity",
            "status",
            "category",
            "wrong_or_inconsistent_math",
            "fix",
            "remaining_action",
        ],
    ]
    check_summary = (
        checks.groupby(["authority_domain", "status"], dropna=False)
        .size()
        .rename("checks")
        .reset_index()
    )
    vendor = inventory.loc[
        inventory["layer"].astype(str).str.startswith(("vendor", "observed_ui")),
        ["formula_id", "authority", "point_in_time_contract", "wizard_comparison", "known_limit"],
    ]
    seven_modes = inventory.loc[
        inventory["layer"].eq("local_seven_mode_replay"),
        [
            "topic",
            "formula",
            "point_in_time_contract",
            "wizard_comparison",
            "known_limit",
        ],
    ]
    reevaluation = summary.get("historical_reevaluation", {})
    reevaluation_lines = (
        [
            "## Measured Historical Re-Evaluation",
            "",
            f"- Frozen experiments compared: {reevaluation['experiments_compared']}",
            f"- Replay-status changes: {reevaluation['replay_status_changes']}",
            f"- Rows with metric changes: {reevaluation['rows_with_metric_changes']}",
            f"- Eligibility changes: {reevaluation['eligibility_gained']} gained, {reevaluation['eligibility_lost']} lost",
            f"- Trade-ledger rows: {reevaluation['old_trade_ledger_rows']} -> {reevaluation['new_trade_ledger_rows']}",
            "- Inputs: hash-identical frozen experiment matrix and pair histories",
            "- Authority: failure attribution only; no promotion or trading authority",
            "",
        ]
        if reevaluation
        else []
    )
    return "\n".join(
        [
            "# V2 Mathematical Diagnostic And Incident Audit",
            "",
            f"Generated: `{summary['generated_at_utc']}`",
            f"Math version: `{summary['math_version']}`",
            f"Economic contract: `{summary['economic_contract_version']}`",
            "",
            "## Executive Verdict",
            "",
            f"- Corrected local core: **{summary['local_core_status']}** ({summary['local_core_checks_passed']}/{summary['local_core_checks_total']} checks).",
            f"- Wizard Static reconstruction: **{summary['wizard_static_reconstruction']}**.",
            "- Wizard Dynamic exact parity: **BLOCKED**.",
            "- Wizard OU exact parity: **BLOCKED**.",
            "- Wizard Copula formula parity: **BLOCKED**.",
            "- Wizard performance-accounting parity: **BLOCKED**.",
            f"- Historical canonical replay: **{summary['historical_canonical_replay']}**.",
            "- Historical descendants: **RERUN REQUIRED**.",
            "- Paper/live authority: **false**.",
            "",
            "The most serious new finding is that the old Math V2 acceptance was a false-pass. The implementation regressed X on Y while documentation and downstream economics declared Y on X. Several consumers then swapped arguments to compensate. The old tests checked cointegration significance but not the recovered coefficient, residual algebra, or ECM direction, so the defect received a 19/19 PASS.",
            "",
            *reevaluation_lines,
            "## Canonical V2.1 Contract",
            "",
            "```text",
            "regression: log(Y_t) = alpha + beta_y_on_x * log(X_t) + e_t",
            "residual:   e_t = log(Y_t) - alpha - beta_y_on_x * log(X_t)",
            "signal +1: short X / long Y",
            "signal -1: long X / short Y",
            "weights:    w_X = -signal*beta/(1+abs(beta)); w_Y = signal/(1+abs(beta))",
            "gross PnL:  w_X,t-1*r_X,t + w_Y,t-1*r_Y,t",
            "net PnL:    gross - fees - slippage - funding - execution_risk - partial_fill",
            "equity:     cumulative product of (1 + net simple return)",
            "Sharpe:     sqrt(periods/year)*mean(net return)/sample std(ddof=1)",
            "```",
            "",
            "A finite positive hedge ratio is required by this particular opposite-leg action contract. A negative-beta relation is not silently discarded as research, but it must use a separately named economic contract because the current long/short action labels no longer describe its weights.",
            "",
            "## Old vs Corrected vs Crypto Wizards",
            "",
            comparison.to_markdown(index=False),
            "",
            "## Complete Incident Ledger",
            "",
            incident_table.to_markdown(index=False),
            "",
            "## Diagnostic Results",
            "",
            check_summary.to_markdown(index=False),
            "",
            "## Crypto Wizards Comparison",
            "",
            vendor.to_markdown(index=False),
            "",
            "Static custom-series evidence reconstructs the vendor series as series 2 on series 1 with an intercept, using levels or logs according to `log_used`, and sample standard deviation (`ddof=1`). That is historical diagnostic evidence only: the fitted sample includes future rows relative to earlier timestamps. Official documentation describes Dynamic as Kalman-based and OU as MLE-based, but does not expose enough parameters and update details for exact parity. Copula conditionals are described, but family selection and full calibration history remain incomplete.",
            "",
            "## Seven-Mode Local Comparison",
            "",
            seven_modes.to_markdown(index=False),
            "",
            "Every row in this table is a local research contract. It does not convert an unproven Dynamic, OU, or Copula implementation into Crypto Wizards exact parity.",
            "",
            "## Historical Re-Evaluation Boundary",
            "",
            impact.to_markdown(index=False),
            "",
            "## What Was Fixed Now",
            "",
            "1. Corrected Engle-Granger to Y-on-X and added orientation/formula metadata.",
            "2. Corrected ECM expected signs for the Y-on-X residual.",
            "3. Removed compensating X/Y argument swaps from replay, teacher, and walk-forward consumers.",
            "4. Bumped math, economic-contract, replay-settings, teacher-formula, and validation versions.",
            "5. Made prices and hedge ratios fail closed; removed silent hedge-ratio 1.0 and price forward-fill behavior.",
            "6. Required positive beta for the named opposite-leg action contract.",
            "7. Blocked zero-variance Sharpe and timeframe/timestamp mismatches.",
            "8. Validated cost-model parameter ranges.",
            "9. Repaired the SciPy/statsmodels lock and standardized tests on `uv run python -m pytest`.",
            "10. Expanded acceptance from plausibility checks to coefficient, identity, sign, causality, direction, and failure-path checks.",
            "11. Seeded Dynamic rolling Y-on-X beta from trailing training history and made nonpositive point-in-time beta an explicit economic-contract blocker.",
            "12. Replaced legacy X-on-Y level-spread ingestion with the canonical Y-on-X log contract and explicit hindsight provenance.",
            "13. Required an explicit hedge ratio at every audited two-leg dispatch boundary; incomplete records remain spread-only research or block.",
            "",
            "## What Is Still Blocked",
            "",
            "- Exact live-safe Wizard Dynamic, OU, and Copula parity.",
            "- Exact Wizard trade-accounting and cost-timing parity.",
            "- Every pre-math-v2.1 derived replay, ranking, dataset, model, and dashboard action.",
            "- Paper/testnet/live authority until controlled regeneration and all existing downstream gates pass.",
            "",
            "## Required Regeneration Order",
            "",
            "1. Freeze and hash raw Wizard and Hyperliquid evidence.",
            "2. Generate the new Math V2.1 acceptance marker.",
            "3. Rerun canonical seven-mode replays on identical frozen inputs.",
            "4. Compare old/new metrics and eligibility by experiment ID.",
            "5. Reattach observed funding, L2 slippage, and cost evidence.",
            "6. Rerun purged walk-forward, regime, robustness, concentration, and multiple-testing controls.",
            "7. Rematerialize teacher/critic evidence.",
            "8. Rebuild datasets with leakage audit and immutable lineage.",
            "9. Retrain ML, RL, shadow, and student-teacher components.",
            "10. Rebuild dashboard and paper preflight; keep order authority false until every gate passes.",
            "",
            "## Prevention Rules",
            "",
            "- A p-value-only test can never validate an orientation-sensitive estimator.",
            "- Every pair setup stores asset order, dependent variable, regressor, beta orientation, residual formula, units, and transform.",
            "- Every formula-changing edit bumps a version and invalidates descendants.",
            "- Every active result carries source hashes and the exact math/economic-contract versions.",
            "- Vendor descriptions, reconstructions, local estimators, and local approximations remain separate namespaces.",
            "- Full-sample dashboard math is never a live feature without a point-in-time reconstruction.",
            "- Diagnostics run only in the locked runtime after numerical import/version smoke checks.",
            "- Historical artifacts are retained for comparison, never silently overwritten or promoted.",
            "",
            "The CSV companions contain the complete formula and incident rows. This report is the readable control document; the authority JSON is the machine gate.",
            "",
        ]
    )


def _input_hashes(
    source_root: Path,
    *,
    vendor_static_proof_path: Path | None = None,
) -> dict[str, str]:
    paths = (
        "pyproject.toml",
        "uv.lock",
        "src/quant_platform/economic_contract.py",
        "src/quant_platform/performance_math.py",
        "src/quant_platform/backtest.py",
        "src/quant_platform/trade_ledger.py",
        "src/quant_platform/experiments.py",
        "src/quant_platform/fixture_ingestion.py",
        "src/quant_platform/dydx_candles.py",
        "src/quant_platform/ml_filter.py",
        "src/quant_platform/statistics/math_v2.py",
        "src/quant_platform/math_v2_acceptance.py",
        "src/quant_platform/formula_registry.py",
        "src/quant_platform/wizard_mode_replay.py",
        "src/quant_platform/orchestration/current_wizard_hyperliquid_replay.py",
        "src/quant_platform/orchestration/current_wizard_hyperliquid_walkforward.py",
        "src/quant_platform/orchestration/hyperliquid_research_validation.py",
        "src/quant_platform/orchestration/teacher_evidence_materializer.py",
        "src/quant_platform/cli.py",
    )
    hashes = {
        relative: _file_hash(path)
        for relative in paths
        if (path := source_root / relative).exists() and path.is_file()
    }
    proof_path = (
        vendor_static_proof_path or source_root / VENDOR_STATIC_PROOF_PATH
    ).resolve()
    if proof_path.is_file():
        hashes[_relative(proof_path, source_root)] = _file_hash(proof_path)
    return hashes


def _version_tuple(value: str) -> tuple[int, ...]:
    numbers: list[int] = []
    for part in str(value).split("."):
        digits = "".join(character for character in part if character.isdigit())
        if not digits:
            break
        numbers.append(int(digits))
    return tuple(numbers)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())
