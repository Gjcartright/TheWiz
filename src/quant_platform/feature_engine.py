from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite
from typing import Mapping

import pandas as pd


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    if not isfinite(value):
        return 0.0
    return max(low, min(high, value))


def logistic_score(value: float, midpoint: float, slope: float, invert: bool = False) -> float:
    score = 100.0 / (1.0 + exp(-slope * (value - midpoint)))
    return clamp(100.0 - score if invert else score)


def bounded_score(value: float, low: float, high: float, invert: bool = False) -> float:
    score = 100.0 * (value - low) / (high - low)
    score = clamp(score)
    return 100.0 - score if invert else score


@dataclass(frozen=True)
class ScoreResult:
    name: str
    score: float
    explanation: str


class FeatureEngine:
    """Explainable factor scoring from raw Crypto Wizards fields."""

    def score(self, row: Mapping[str, float]) -> dict[str, ScoreResult]:
        return {
            "cointegration_score": self.cointegration_score(row),
            "mean_reversion_score": self.mean_reversion_score(row),
            "copula_dislocation_score": self.copula_dislocation_score(row),
            "tail_risk_score": self.tail_risk_score(row),
            "ecm_score": self.ecm_score(row),
            "backtest_quality_score": self.backtest_quality_score(row),
            "proprietary_signal_score": self.proprietary_signal_score(row),
            "regime_score": self.regime_score(row),
            "execution_quality_score": self.execution_quality_score(row),
        }

    def score_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Append explainable 0-100 scores used by composite strategy families."""
        enriched = frame.copy()
        score_rows: list[dict[str, float]] = []
        for _, row in enriched.iterrows():
            scores = self.score(row.to_dict())
            score_rows.append({name: result.score for name, result in scores.items()})
        if score_rows:
            score_frame = pd.DataFrame(score_rows, index=enriched.index)
            for column in score_frame.columns:
                if column not in enriched.columns:
                    enriched[column] = score_frame[column]
            if "composite_score" not in enriched.columns:
                enriched["composite_score"] = score_frame.mean(axis=1)
        return enriched

    def cointegration_score(self, row: Mapping[str, float]) -> ScoreResult:
        pvalue = float(row.get("cointegration_pvalue", row.get("cointegration", 1.0)))
        stability = float(row.get("hedge_ratio_stability", 0.5))
        score = 0.7 * bounded_score(pvalue, 0.10, 0.0) + 0.3 * bounded_score(stability, 0.0, 1.0)
        return ScoreResult("Cointegration Score", clamp(score), "Rewards low cointegration p-value and stable hedge ratio.")

    def mean_reversion_score(self, row: Mapping[str, float]) -> ScoreResult:
        hurst = float(row.get("hurst", 0.5))
        half_life = float(row.get("half_life", 999.0))
        z = abs(float(row.get("zscore", row.get("rolling_zscore", 0.0))))
        hurst_score = bounded_score(hurst, 0.5, 0.0)
        hl_score = 100.0 if 2 <= half_life <= 48 else clamp(100.0 - abs(half_life - 24.0) * 2.0)
        z_score = bounded_score(z, 0.5, 3.0)
        score = 0.35 * hurst_score + 0.35 * hl_score + 0.30 * z_score
        return ScoreResult("Mean Reversion Score", clamp(score), "Combines anti-persistence, tradable decay horizon, and current dislocation.")

    def copula_dislocation_score(self, row: Mapping[str, float]) -> ScoreResult:
        distortion = abs(float(row.get("conditional_probability_distortion", row.get("conditional_probabilities", 0.0))))
        tail_dep = float(row.get("tail_dependence", 0.0))
        calibration = float(row.get("copula_calibration_score", 0.5))
        score = 0.50 * bounded_score(distortion, 0.0, 0.35) + 0.25 * bounded_score(tail_dep, 0.0, 0.8) + 0.25 * bounded_score(calibration, 0.0, 1.0)
        return ScoreResult("Copula Dislocation Score", clamp(score), "Rewards calibrated conditional probability distortion and relevant tail dependence.")

    def tail_risk_score(self, row: Mapping[str, float]) -> ScoreResult:
        cvar = abs(float(row.get("cvar", 0.10)))
        var = abs(float(row.get("var", 0.06)))
        drawdown = abs(float(row.get("drawdown", 0.20)))
        score = 0.4 * bounded_score(cvar, 0.20, 0.01) + 0.25 * bounded_score(var, 0.12, 0.005) + 0.35 * bounded_score(drawdown, 0.20, 0.01)
        return ScoreResult("Tail Risk Score", clamp(score), "Higher score means lower expected tail loss and drawdown pressure.")

    def ecm_score(self, row: Mapping[str, float]) -> ScoreResult:
        strength = abs(float(row.get("ecm_strength", 0.0)))
        x = abs(float(row.get("ecm_x", 0.0)))
        y = abs(float(row.get("ecm_y", 0.0)))
        asymmetry = abs(x - y)
        score = 0.65 * bounded_score(strength, 0.0, 1.0) + 0.35 * bounded_score(asymmetry, 0.0, 0.5)
        return ScoreResult("ECM Score", clamp(score), "Rewards strong correction and measurable leader/follower asymmetry.")

    def backtest_quality_score(self, row: Mapping[str, float]) -> ScoreResult:
        pf = float(row.get("profit_factor", 1.0))
        sharpe = float(row.get("sharpe", 0.0))
        trades = float(row.get("completed_trades", 0.0))
        dd = abs(float(row.get("drawdown", 0.20)))
        score = (
            0.35 * bounded_score(pf, 1.0, 2.2)
            + 0.30 * bounded_score(sharpe, 0.0, 2.5)
            + 0.20 * bounded_score(trades, 30.0, 250.0)
            + 0.15 * bounded_score(dd, 0.20, 0.02)
        )
        return ScoreResult("Backtest Quality Score", clamp(score), "Rewards cost-adjusted PF, Sharpe, sample size, and controlled drawdown.")

    def proprietary_signal_score(self, row: Mapping[str, float]) -> ScoreResult:
        ml = float(row.get("ml_confidence", 0.5))
        profile = float(row.get("profile_match", 0.5))
        ou = float(row.get("ou_optimal", 0.5))
        score = 0.40 * bounded_score(ml, 0.5, 0.8) + 0.30 * bounded_score(profile, 0.0, 1.0) + 0.30 * bounded_score(ou, 0.0, 1.0)
        return ScoreResult("Proprietary Signal Score", clamp(score), "Combines calibrated model confidence, historical profile match, and OU threshold quality.")

    def regime_score(self, row: Mapping[str, float]) -> ScoreResult:
        match = float(row.get("regime_strategy_match", 0.5))
        vol = float(row.get("realized_volatility_percentile", 0.5))
        crisis = float(row.get("crisis_probability", 0.0))
        score = 0.55 * bounded_score(match, 0.0, 1.0) + 0.25 * bounded_score(vol, 1.0, 0.0) + 0.20 * bounded_score(crisis, 0.5, 0.0)
        return ScoreResult("Regime Score", clamp(score), "Rewards regimes where the strategy historically works and penalizes crisis/volatility stress.")

    def execution_quality_score(self, row: Mapping[str, float]) -> ScoreResult:
        spread_bps = float(row.get("bid_ask_spread_bps", 20.0))
        slippage_bps = float(row.get("slippage_bps", 15.0))
        funding_bps = abs(float(row.get("funding_bps_per_day", 5.0)))
        liquidity = float(row.get("liquidity_score", 0.5))
        score = (
            0.30 * bounded_score(spread_bps, 30.0, 1.0)
            + 0.30 * bounded_score(slippage_bps, 25.0, 1.0)
            + 0.20 * bounded_score(funding_bps, 10.0, 0.0)
            + 0.20 * bounded_score(liquidity, 0.0, 1.0)
        )
        return ScoreResult("Execution Quality Score", clamp(score), "Rewards tight spreads, low slippage/funding, and sufficient liquidity.")
