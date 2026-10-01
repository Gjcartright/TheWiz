from quant_platform.feature_engine import FeatureEngine
from quant_platform.strategies import composite_quant_score_signal
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


def test_feature_engine_tail_and_backtest_scores_fail_closed_for_proxy_metrics():
    engine = FeatureEngine()
    proxy_only = {
        "research_proxy_cvar": 0.01,
        "research_proxy_var": 0.005,
        "research_proxy_drawdown": 0.01,
        "research_proxy_sharpe": 4.0,
        "profit_factor": 3.0,
        "completed_trades": 500,
    }

    tail = engine.tail_risk_score(proxy_only)
    backtest = engine.backtest_quality_score(proxy_only)

    assert tail.score == 0.0
    assert "point-in-time" in tail.explanation
    assert backtest.score == 0.0
    assert "point-in-time" in backtest.explanation


def test_feature_engine_rejects_canonical_values_with_proxy_provenance():
    engine = FeatureEngine()
    row = {
        "cvar": 0.01,
        "var": 0.005,
        "drawdown": 0.01,
        "sharpe": 4.0,
        "cvar_feature_source": "research_proxy_cvar",
        "var_feature_source": "research_proxy_var",
        "drawdown_feature_source": "full_sample_hindsight",
        "sharpe_feature_source": "research_proxy_sharpe",
        "profit_factor": 3.0,
        "completed_trades": 500,
    }

    assert engine.tail_risk_score(row).score == 0.0
    assert engine.backtest_quality_score(row).score == 0.0


def test_feature_engine_rejects_unlabeled_copies_of_research_proxy_metrics():
    engine = FeatureEngine()
    row = {
        "cvar": 0.01,
        "var": 0.005,
        "drawdown": 0.01,
        "sharpe": 4.0,
        "research_proxy_cvar": 0.01,
        "research_proxy_var": 0.005,
        "research_proxy_drawdown": 0.01,
        "research_proxy_sharpe": 4.0,
        "profit_factor": 3.0,
        "completed_trades": 500,
    }

    assert engine.tail_risk_score(row).score == 0.0
    assert engine.backtest_quality_score(row).score == 0.0


def test_feature_engine_preserves_scores_for_complete_canonical_metrics():
    engine = FeatureEngine()
    row = {
        "cvar": 0.04,
        "var": 0.02,
        "drawdown": 0.05,
        "sharpe": 1.8,
        "profit_factor": 1.9,
        "completed_trades": 300,
    }

    assert engine.tail_risk_score(row).score > 0.0
    assert engine.backtest_quality_score(row).score > 0.0


def test_feature_engine_rescores_supplied_claims_before_strategy_use():
    supplied = pd.DataFrame(
        [
            {
                "spread": 0.1,
                "zscore": 2.4,
                "cointegration_score": 99.0,
                "ecm_score": 99.0,
                "copula_dislocation_score": 99.0,
                "tail_risk_score": 99.0,
                "backtest_quality_score": 99.0,
                "composite_score": 99.0,
            }
        ]
    )

    scored = FeatureEngine().score_frame(supplied)

    assert composite_quant_score_signal(supplied).iloc[0] == -1.0
    assert composite_quant_score_signal(scored).iloc[0] == 0.0
    assert scored["tail_risk_score"].iloc[0] == 0.0
    assert scored["backtest_quality_score"].iloc[0] == 0.0
    assert scored["composite_score"].iloc[0] < 70.0
    for name in (
        "cointegration_score",
        "ecm_score",
        "copula_dislocation_score",
        "tail_risk_score",
        "backtest_quality_score",
        "composite_score",
    ):
        assert scored[f"reported_{name}"].iloc[0] == 99.0

    rescored = FeatureEngine().score_frame(scored)
    assert rescored["reported_composite_score"].iloc[0] == 99.0
    assert rescored["composite_score"].iloc[0] == scored["composite_score"].iloc[0]
