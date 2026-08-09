from __future__ import annotations

import json

import numpy as np
import pandas as pd

import quant_platform.wizard_hyperliquid_mode_proof as mode_proof


def _write_ready_queue(root) -> None:
    active = root / "reports" / "active"
    history_path = root / "data" / "raw" / "hyperliquid_pair.json"
    active.mkdir(parents=True)
    history_path.parent.mkdir(parents=True)
    history_path.write_text(
        json.dumps(
            {
                "history": [
                    {
                        "open_x": 99.5 + index,
                        "price_x": 100.0 + index,
                        "open_y": 49.75 + index * 0.5,
                        "price_y": 50.0 + index * 0.5,
                    }
                    for index in range(60)
                ]
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "pair": "BNB-USD-WLD-USD",
                "asset_x": "BNB-USD",
                "asset_y": "WLD-USD",
                "venue": "hyperliquid",
                "local_interval": "1d",
                "local_history_path": "data/raw/hyperliquid_pair.json",
                "exact_mode": "OU (ZScoreR)",
                "vendor_request_strategy": "ZScoreRoll",
                "vendor_request_spread_type": "OU",
                "wizard_period": 60,
                "proof_observations": 60,
                "candidate_set_id": "b" * 64,
                "discovery_policy_schema_version": "wizard_discovery_policy.v1",
                "discovery_policy_hash": "a" * 64,
                "entry_level": 2.0,
                "exit_level": 0.0,
                "x_weighting": 0.5,
                "slippage_rate": 0.0005,
                "commission_rate": 0.0005,
                "roll_w": 20,
                "vendor_custom_series_eligible": True,
                "evidence_path": "reports/active/hyperliquid_wizard_hypothesis_queue.csv",
            }
        ]
    ).to_csv(active / "hyperliquid_wizard_hypothesis_queue.csv", index=False)


def test_preflight_never_calls_vendor_or_spends_credits(tmp_path, monkeypatch):
    _write_ready_queue(tmp_path)
    calls = []
    monkeypatch.setattr(mode_proof, "fetch_custom_series_backtest", lambda *args, **kwargs: calls.append((args, kwargs)))

    result = mode_proof.run_hyperliquid_wizard_mode_proofs(root=tmp_path, execute=False)
    row = pd.read_csv(result.paths["proofs"]).iloc[0]

    assert calls == []
    assert result.summary["preflight_only"] is True
    assert result.summary["completed"] == 0
    assert row["mode_proof_status"] == "preflight_ready"
    assert row["credits_estimated"] == 0
    assert row["proof_observations"] == 60
    assert row["proof_window_kind"] == "scanner_horizon_parity"
    assert row["proof_validity"] == "NOT_COMPLETED"
    assert not row["training_eligible"]
    assert not row["promotion_allowed"]


def test_execute_request_is_blocked_without_explicit_environment_guard(tmp_path, monkeypatch):
    _write_ready_queue(tmp_path)
    calls = []
    monkeypatch.delenv("QPA_ENABLE_WIZARD_CUSTOM_SERIES_PROOF", raising=False)
    monkeypatch.setattr(mode_proof, "fetch_custom_series_backtest", lambda *args, **kwargs: calls.append((args, kwargs)))

    result = mode_proof.run_hyperliquid_wizard_mode_proofs(root=tmp_path, execute=True, api_key="test-key")
    row = pd.read_csv(result.paths["proofs"]).iloc[0]

    assert calls == []
    assert result.summary["execution_enabled"] is False
    assert row["mode_proof_status"] == "execution_disabled"
    assert row["blocker"] == "QPA_ENABLE_WIZARD_CUSTOM_SERIES_PROOF_not_true"


def test_guarded_execution_records_vendor_evidence_but_never_promotes(tmp_path, monkeypatch):
    _write_ready_queue(tmp_path)
    captured = {}
    monkeypatch.setenv("QPA_ENABLE_WIZARD_CUSTOM_SERIES_PROOF", "true")

    def fake_fetch(request, *, api_key):
        captured["payload"] = request.payload()
        captured["api_key"] = api_key
        return {
            "data": {
                "sharpe_ratio": 2.4,
                "strat_returns": {"total_return": 0.31},
                "max_drawdown": -0.08,
            },
            "history": {"spread": [0.1, 0.2]},
        }

    monkeypatch.setattr(mode_proof, "fetch_custom_series_backtest", fake_fetch)
    result = mode_proof.run_hyperliquid_wizard_mode_proofs(root=tmp_path, execute=True, api_key="test-key")
    row = pd.read_csv(result.paths["proofs"]).iloc[0]

    assert result.summary["completed"] == 1
    assert captured["api_key"] == "test-key"
    assert captured["payload"]["params"]["strategy"] == "ZScoreRoll"
    assert captured["payload"]["params"]["spread_type"] == "OU"
    assert len(captured["payload"]["params"]["series_1_opens"]) == 60
    assert len(captured["payload"]["params"]["series_2_opens"]) == 60
    assert row["mode_proof_status"] == "completed"
    assert row["mode_computation_source"] == "crypto_wizards_custom_series_backtest"
    assert row["proof_window_kind"] == "scanner_horizon_parity"
    assert row["proof_validity"] == "RESEARCH_DIAGNOSTIC_WEAK"
    assert "vendor_performance_accounting_not_reconstructed" in row["validity_blocker"]
    assert not row["training_eligible"]
    assert not row["promotion_allowed"]
    assert (tmp_path / row["request_path"]).exists()
    assert (tmp_path / row["response_path"]).exists()

    second = mode_proof.run_hyperliquid_wizard_mode_proofs(root=tmp_path, execute=True, api_key="test-key")
    assert second.summary["selected"] == 0
    assert len(captured["payload"]["params"]["series_1_closes"]) == 60
    assert len(pd.read_csv(second.paths["proofs"])) == 1

    preflight = mode_proof.run_hyperliquid_wizard_mode_proofs(root=tmp_path, execute=False)
    preserved = pd.read_csv(preflight.paths["proofs"])
    assert preflight.summary["selected"] == 0
    assert len(preserved) == 1
    assert preserved.loc[0, "mode_proof_status"] == "completed"


def test_static_vendor_formula_audit_reconstructs_spread_and_both_zscores():
    x = np.linspace(100.0, 160.0, 80)
    y = 4.0 + 0.25 * x + np.sin(np.arange(80) / 4.0)
    beta, alpha = np.polyfit(x, y, 1)
    spread = y - (alpha + beta * x)
    zscore = (spread - spread.mean()) / spread.std(ddof=1)
    spread_series = pd.Series(spread)
    zscore_roll = (
        spread_series.sub(spread_series.rolling(20, min_periods=20).mean())
        .div(spread_series.rolling(20, min_periods=20).std(ddof=1))
        .fillna(0.0)
    )
    request = {
        "params": {
            "strategy": "Spread",
            "spread_type": "Static",
            "roll_w": 20,
            "series_1_closes": x.tolist(),
            "series_2_closes": y.tolist(),
        }
    }
    response = {
        "history": {
            "spread_stats": {
                "spread": spread.tolist(),
                "zscore": zscore.tolist(),
                "zscore_roll": zscore_roll.tolist(),
            }
        }
    }

    metrics = mode_proof._formula_parity_metrics(request, response)

    assert metrics["vendor_formula_parity_status"] == "exact_reconstruction"
    assert metrics["vendor_spread_max_abs_error"] < 1e-12
    assert metrics["vendor_zscore_max_abs_error"] < 1e-12
    assert metrics["vendor_zscore_roll_max_abs_error"] < 1e-12
    assert metrics["vendor_formula_pit_status"] == "hindsight_full_sample_fit_not_live_signal_safe"


def test_static_vendor_formula_audit_honors_vendor_log_transform():
    x = np.linspace(0.10, 0.30, 80)
    y = np.exp(2.0 + 1.7 * np.log(x) + 0.03 * np.sin(np.arange(80)))
    transformed_x = np.log(x)
    transformed_y = np.log(y)
    beta, alpha = np.polyfit(transformed_x, transformed_y, 1)
    spread = transformed_y - (alpha + beta * transformed_x)
    zscore = (spread - spread.mean()) / spread.std(ddof=1)
    spread_series = pd.Series(spread)
    zscore_roll = (
        spread_series.sub(spread_series.rolling(20, min_periods=20).mean())
        .div(spread_series.rolling(20, min_periods=20).std(ddof=1))
        .fillna(0.0)
    )
    request = {
        "params": {
            "strategy": "Spread",
            "spread_type": "Static",
            "roll_w": 20,
            "series_1_closes": x.tolist(),
            "series_2_closes": y.tolist(),
        }
    }
    response = {
        "history": {
            "spread_stats": {
                "log_used": True,
                "spread": spread.tolist(),
                "zscore": zscore.tolist(),
                "zscore_roll": zscore_roll.tolist(),
            }
        }
    }

    metrics = mode_proof._formula_parity_metrics(request, response)

    assert metrics["vendor_formula_parity_status"] == "exact_reconstruction"
    assert metrics["vendor_spread_formula"].startswith("ols_log_y_on_log_x")
