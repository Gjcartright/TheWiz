from quant_platform.feature_engine import FeatureEngine
import pandas as pd


def test_feature_engine_outputs_required_scores():
    scores = FeatureEngine().score(
        {
            "cointegration_pvalue": 0.01,
            "hedge_ratio_stability": 0.9,
            "hurst": 0.35,
            "half_life": 12,
            "zscore": 2.4,
            "conditional_probability_distortion": 0.25,
            "tail_dependence": 0.4,
            "copula_calibration_score": 0.8,
            "cvar": 0.04,
            "var": 0.02,
            "drawdown": 0.05,
            "ecm_strength": 0.7,
            "ecm_x": -0.3,
            "ecm_y": -0.05,
            "profit_factor": 1.9,
            "sharpe": 1.8,
            "completed_trades": 300,
            "ml_confidence": 0.7,
            "profile_match": 0.8,
            "ou_optimal": 0.75,
            "regime_strategy_match": 0.8,
            "realized_volatility_percentile": 0.4,
            "crisis_probability": 0.1,
            "bid_ask_spread_bps": 3,
            "slippage_bps": 4,
            "funding_bps_per_day": 1,
            "liquidity_score": 0.9,
        }
    )
    assert set(scores) == {
        "cointegration_score",
        "mean_reversion_score",
        "copula_dislocation_score",
        "tail_risk_score",
        "ecm_score",
        "backtest_quality_score",
        "proprietary_signal_score",
        "regime_score",
        "execution_quality_score",
    }
    assert all(0 <= result.score <= 100 for result in scores.values())


def test_feature_engine_enriches_frames_with_scores_and_composite():
    frame = pd.DataFrame(
        [
            {
                "spread": 0.1,
                "zscore": 2.1,
                "cointegration_pvalue": 0.02,
                "ecm_strength": 0.7,
                "conditional_probability_distortion": 0.25,
                "tail_dependence": 0.4,
            }
        ]
    )

    enriched = FeatureEngine().score_frame(frame)

    assert "cointegration_score" in enriched
    assert "ecm_score" in enriched
    assert "copula_dislocation_score" in enriched
    assert "composite_score" in enriched
    assert 0 <= enriched["composite_score"].iloc[0] <= 100
