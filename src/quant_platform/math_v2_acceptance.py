"""Controlled acceptance checks and artifacts for the Math V2 library."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from quant_platform.backtest import CostModel, FundingPolicy, backtest_two_leg_spread
from quant_platform.performance_math import MATH_VERSION, calculate_annualized_sharpe
from quant_platform.statistics.math_v2 import (
    estimate_hurst_dfa,
    fit_engle_granger,
    fit_gaussian_copula,
    fit_ou,
    rolling_zscore_variants,
)
from quant_platform.trade_ledger import build_trade_ledger


ROOT = Path(__file__).resolve().parents[2]


def build_math_v2_acceptance(*, root: Path = ROOT) -> dict[str, object]:
    """Run deterministic checks and write a non-hand-authored acceptance marker."""

    output_dir = root / "reports" / "active"
    output_dir.mkdir(parents=True, exist_ok=True)
    reconciliation_path = output_dir / "math_v2_reconciliation.csv"
    statistical_path = output_dir / "statistical_validity_audit.csv"
    marker_path = output_dir / "math_v2_acceptance.json"

    reconciliation, statistical = _run_checks()
    reconciliation.to_csv(reconciliation_path, index=False)
    statistical.to_csv(statistical_path, index=False)
    all_checks = pd.concat([reconciliation, statistical], ignore_index=True)
    passed = bool(not all_checks.empty and all_checks["status"].eq("PASS").all())
    marker = {
        "math_version": MATH_VERSION,
        "status": "passed" if passed else "blocked",
        "acceptance_scope": "core_math_library",
        "all_checks_passed": passed,
        "passed_checks": int(all_checks["status"].eq("PASS").sum()),
        "total_checks": int(len(all_checks)),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_by": "quant_platform.math_v2_acceptance",
        "reconciliation_report": str(reconciliation_path.relative_to(root)),
        "statistical_validity_report": str(statistical_path.relative_to(root)),
        "wizard_exact_mode_parity": "BLOCKED",
        "walk_forward_signal_authority": "BLOCKED",
        "student_training_authority": "BLOCKED",
        "execution_authority": "BLOCKED",
        "remaining_components": [
            "wizard_exact_mode_parity",
            "point_in_time_walk_forward_materialization",
            "integration_order_and_johansen",
            "structural_break_and_parameter_stability",
            "multiple_testing_correction",
            "non_gaussian_copula_family_selection",
        ],
        "reason": (
            "core Math V2 checks passed; downstream authority remains separately gated"
            if passed
            else "one or more core Math V2 checks failed"
        ),
    }
    marker_path.write_text(json.dumps(marker, indent=2, sort_keys=True), encoding="utf-8")
    return {
        **marker,
        "marker": marker_path,
        "reconciliation": reconciliation_path,
        "statistical_validity": statistical_path,
    }


def _run_checks() -> tuple[pd.DataFrame, pd.DataFrame]:
    reconciliation_checks: list[tuple[str, Callable[[], tuple[bool, object, object]]]] = [
        ("daily_sharpe_uses_365", _daily_sharpe_check),
        ("hourly_sharpe_uses_8760", _hourly_sharpe_check),
        ("unknown_interval_blocks_sharpe", _unknown_interval_check),
        ("two_round_trips_equal_two_closed_trades", _two_round_trip_check),
        ("open_trade_excluded_from_closed_metrics", _open_trade_check),
        ("reversal_closes_then_opens", _reversal_check),
        ("trade_factors_reconcile_to_equity", _ledger_reconciliation_check),
        ("beta_is_diagnostic_only", _beta_diagnostic_check),
        ("funding_policies_are_distinct", _funding_policy_check),
    ]
    statistical_checks: list[tuple[str, Callable[[], tuple[bool, object, object]]]] = [
        ("zscore_retains_ddof0_and_ddof1", _zscore_variant_check),
        ("zscore_is_causal_with_respect_to_future_rows", _zscore_causality_check),
        ("engle_granger_detects_known_cointegration", _cointegration_check),
        ("ou_fit_recovers_valid_phi_and_half_life", _ou_check),
        ("invalid_ou_phi_is_not_abs_or_clipped", _invalid_ou_check),
        ("hurst_dfa_declares_estimator_and_sample_gate", _hurst_check),
        ("gaussian_copula_uses_conditional_cdf", _copula_check),
    ]
    return _check_frame(reconciliation_checks), _check_frame(statistical_checks)


def _check_frame(checks: list[tuple[str, Callable[[], tuple[bool, object, object]]]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name, check in checks:
        try:
            passed, observed, required = check()
            error = ""
        except Exception as exc:  # pragma: no cover - surfaced in the artifact
            passed, observed, required = False, f"{type(exc).__name__}:{exc}", "check completes without error"
            error = str(exc)
        rows.append(
            {
                "math_version": MATH_VERSION,
                "check": name,
                "status": "PASS" if passed else "BLOCKED",
                "observed": observed,
                "required": required,
                "error": error,
            }
        )
    return pd.DataFrame(rows)


def _daily_sharpe_check() -> tuple[bool, object, object]:
    result = calculate_annualized_sharpe(pd.Series([0.01, -0.005, 0.007]), interval="1d")
    return result.periods_per_year == 365.0, result.periods_per_year, 365


def _hourly_sharpe_check() -> tuple[bool, object, object]:
    result = calculate_annualized_sharpe(pd.Series([0.01, -0.005, 0.007]), interval="1h")
    return result.periods_per_year == 8760.0, result.periods_per_year, 8760


def _unknown_interval_check() -> tuple[bool, object, object]:
    result = calculate_annualized_sharpe(pd.Series([0.01, -0.005, 0.007]))
    return result.status == "blocked" and np.isnan(result.value), result.status, "blocked with NaN"


def _trade_fixture() -> tuple[pd.Series, pd.Series, dict[str, pd.Series]]:
    target = pd.Series([0.0, 1.0, 1.0, 0.0, 0.0, -1.0, -1.0, 0.0])
    gross = pd.Series([0.0, 0.0, 0.01, 0.02, 0.0, 0.0, 0.015, 0.01])
    zero = pd.Series(0.0, index=target.index)
    costs = {name: zero.copy() for name in ("fees", "slippage", "funding", "execution_risk", "partial_fill")}
    return target, gross, costs


def _two_round_trip_check() -> tuple[bool, object, object]:
    target, gross, costs = _trade_fixture()
    ledger = build_trade_ledger(target, gross, costs)
    return len(ledger.closed_trades) == 2, len(ledger.closed_trades), 2


def _open_trade_check() -> tuple[bool, object, object]:
    target = pd.Series([0.0, 1.0, 1.0])
    gross = pd.Series([0.0, 0.0, 0.01])
    ledger = build_trade_ledger(target, gross, {})
    observed = f"closed={len(ledger.closed_trades)};open={len(ledger.open_trades)}"
    return len(ledger.closed_trades) == 0 and len(ledger.open_trades) == 1, observed, "closed=0;open=1"


def _reversal_check() -> tuple[bool, object, object]:
    target = pd.Series([0.0, 1.0, 1.0, -1.0, -1.0, 0.0])
    gross = pd.Series([0.0, 0.0, 0.01, 0.005, 0.01, 0.01])
    ledger = build_trade_ledger(target, gross, {})
    reasons = ledger.closed_trades["exit_reason"].tolist()
    return reasons == ["signal_reversal", "signal_exit"], ";".join(reasons), "signal_reversal;signal_exit"


def _ledger_reconciliation_check() -> tuple[bool, object, object]:
    target, gross, costs = _trade_fixture()
    costs["fees"] = pd.Series([0.0, 0.001, 0.0, 0.001, 0.0, 0.001, 0.0, 0.001])
    ledger = build_trade_ledger(target, gross, costs)
    return ledger.reconciliation_error < 1e-12, ledger.reconciliation_error, "<1e-12"


def _two_leg_frame(*, beta: float = 1.0) -> tuple[pd.DataFrame, pd.Series]:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=8, freq="1h", tz="UTC"),
            "price_x": [100.0, 101.0, 102.0, 101.0, 100.0, 99.0, 100.0, 101.0],
            "price_y": [50.0, 49.5, 49.0, 49.5, 50.0, 50.5, 50.0, 49.5],
            "hedge_ratio": [1.2] * 8,
            "beta": [beta] * 8,
            "funding_x_bps": [2.0] * 8,
            "funding_y_bps": [2.0] * 8,
        }
    )
    return frame, pd.Series([0.0, 1.0, 1.0, 0.0, -1.0, -1.0, 0.0, 0.0])


def _beta_diagnostic_check() -> tuple[bool, object, object]:
    low, signal = _two_leg_frame(beta=0.1)
    high, _ = _two_leg_frame(beta=10.0)
    model = CostModel(funding_bps_per_day=0.0)
    low_result = backtest_two_leg_spread(low, signal, model)
    high_result = backtest_two_leg_spread(high, signal, model)
    difference = abs(low_result.total_return - high_result.total_return)
    return difference < 1e-15, difference, "<1e-15"


def _funding_policy_check() -> tuple[bool, object, object]:
    frame, signal = _two_leg_frame()
    signed = backtest_two_leg_spread(
        frame,
        signal,
        CostModel(
            taker_fee_bps=0.0,
            slippage_bps=0.0,
            execution_risk_bps=0.0,
            partial_fill_probability=0.0,
            funding_policy=FundingPolicy.SIGNED_REALIZED.value,
        ),
    )
    conservative = backtest_two_leg_spread(
        frame,
        signal,
        CostModel(
            taker_fee_bps=0.0,
            slippage_bps=0.0,
            execution_risk_bps=0.0,
            partial_fill_probability=0.0,
            funding_policy=FundingPolicy.CONSERVATIVE_ABSOLUTE_DRAG.value,
        ),
    )
    observed = f"signed={signed.total_funding:.12g};conservative={conservative.total_funding:.12g}"
    return signed.total_funding != conservative.total_funding and conservative.total_funding > 0.0, observed, "distinct; conservative>0"


def _zscore_variant_check() -> tuple[bool, object, object]:
    frame = rolling_zscore_variants(pd.Series(range(20), dtype=float), window=5)
    observed = ";".join(frame.columns)
    return set(frame.columns) == {"zscore_ddof0", "zscore_ddof1"}, observed, "both variants"


def _zscore_causality_check() -> tuple[bool, object, object]:
    original = pd.Series(np.arange(20), dtype=float)
    changed = original.copy()
    changed.iloc[-1] = 10_000.0
    first = rolling_zscore_variants(original, window=5)
    second = rolling_zscore_variants(changed, window=5)
    equal = first.iloc[:-1].equals(second.iloc[:-1])
    return equal, equal, True


def _synthetic_cointegrated(seed: int = 7, rows: int = 600) -> tuple[pd.Series, pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    log_y = 4.0 + np.cumsum(rng.normal(0.0, 0.01, rows))
    residual = np.zeros(rows)
    for index in range(1, rows):
        residual[index] = 0.82 * residual[index - 1] + rng.normal(0.0, 0.008)
    log_x = 0.4 + 1.15 * log_y + residual
    return pd.Series(np.exp(log_x)), pd.Series(np.exp(log_y)), pd.Series(residual)


def _cointegration_check() -> tuple[bool, object, object]:
    price_x, price_y, _ = _synthetic_cointegrated()
    result = fit_engle_granger(price_x, price_y)
    pvalue = result.values.get("cointegration_pvalue", 1.0)
    return result.validity_status == "valid" and pvalue < 0.05, pvalue, "<0.05"


def _ou_check() -> tuple[bool, object, object]:
    _, _, residual = _synthetic_cointegrated()
    result = fit_ou(residual)
    phi = result.values.get("phi", float("nan"))
    half_life = result.values.get("half_life", float("nan"))
    passed = result.validity_status == "valid" and 0.75 < phi < 0.90 and half_life > 0.0
    return passed, f"phi={phi};half_life={half_life}", "0.75<phi<0.90;half_life>0"


def _invalid_ou_check() -> tuple[bool, object, object]:
    rng = np.random.default_rng(11)
    random_walk = pd.Series(np.cumsum(rng.normal(size=600)))
    result = fit_ou(random_walk)
    phi = result.values.get("phi", float("nan"))
    # A near-unit estimate may still lie barely below one, so use a deterministic
    # explosive process to prove the invalid branch is never abs-valued or clipped.
    explosive = pd.Series([1.01**index for index in range(200)], dtype=float)
    explosive_result = fit_ou(explosive)
    passed = explosive_result.validity_status == "invalid" and explosive_result.values.get("phi", 0.0) >= 1.0
    return passed, f"random_walk_phi={phi};explosive={asdict(explosive_result)}", "explosive phi retained >=1 and invalid"


def _hurst_check() -> tuple[bool, object, object]:
    short = estimate_hurst_dfa(pd.Series(range(40), dtype=float))
    rng = np.random.default_rng(17)
    white_noise = estimate_hurst_dfa(pd.Series(rng.normal(size=1024)))
    hurst = white_noise.values.get("hurst", float("nan"))
    passed = short.validity_status == "invalid" and white_noise.validity_status == "valid" and 0.3 < hurst < 0.7
    return passed, f"short={short.validity_reason};white_noise_h={hurst}", "short invalid;0.3<H<0.7"


def _copula_check() -> tuple[bool, object, object]:
    rng = np.random.default_rng(23)
    covariance = np.array([[1.0, 0.65], [0.65, 1.0]])
    samples = rng.multivariate_normal([0.0, 0.0], covariance, size=800)
    result = fit_gaussian_copula(pd.Series(samples[:, 0]), pd.Series(samples[:, 1]))
    rho = result.values.get("rho", float("nan"))
    conditional = result.values.get("u1_given_u2", pd.Series(dtype=float))
    bounded = isinstance(conditional, pd.Series) and bool(conditional.dropna().between(0.0, 1.0).all())
    passed = result.validity_status == "valid" and 0.55 < rho < 0.75 and bounded
    return passed, f"rho={rho};bounded={bounded};method={result.method_id}", "0.55<rho<0.75;conditional CDF bounded"
