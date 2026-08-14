from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

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


def _write_exhaustive_inputs(root) -> None:
    active = root / "reports" / "active"
    history_path = root / "data" / "raw" / "pair_history.json"
    active.mkdir(parents=True)
    history_path.parent.mkdir(parents=True)
    history_path.write_text(
        json.dumps(
            {
                "asset_x": "ADA",
                "asset_y": "ALGO",
                "history": [
                    {
                        "open_x": 1.0 + index,
                        "price_x": 1.1 + index,
                        "open_y": 101.0 + index,
                        "price_y": 101.1 + index,
                    }
                    for index in range(60)
                ],
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "pair_group_id": "pair-1",
                "pair": "ADA-USD-ALGO-USD",
                "hyperliquid_interval": "1d",
                "history_status": "READY_FOR_CANONICAL_REPLAY",
                "history_rows": 60,
                "latest_candle_at": "2026-08-09T00:00:00Z",
                "history_path": "data/raw/pair_history.json",
            }
        ]
    ).to_csv(active / "exhaustive_wizard_hyperliquid_pair_history_results.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair_group_id": "pair-1",
                "pair": "ADA-ALGO",
                "asset_x": "ADA",
                "asset_y": "ALGO",
                "capture_asset_x": "ADA" if orientation == "original" else "ALGO",
                "capture_asset_y": "ALGO" if orientation == "original" else "ADA",
                "exact_mode": "Static (Spread)",
                "orientation": orientation,
                "capture_status": "CAPTURED",
                "orientation_verified": True,
                "capture_timestamp": "2026-08-09T12:00:00Z",
                "periods_analyzed": 60,
                "entry_long": 2.0,
                "entry_short": -2.0,
                "exit_long": 0.0,
                "exit_short": 0.0,
                "rolling_window": None,
                "x_weighting": 0.35 if orientation == "original" else 0.63,
                "wizard_commission_pct": 0.1,
                "wizard_slippage_pct": 0.05,
                "stop_loss_pct": 0.0,
                "close_n_periods": 0,
                "evidence_path": f"raw/{orientation}.json",
            }
            for orientation in ("original", "reverse")
        ]
    ).to_csv(active / "exhaustive_wizard_pair_detail_mode_ledger.csv", index=False)


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

    result = mode_proof.run_hyperliquid_wizard_mode_proofs(
        root=tmp_path,
        execute=True,
        api_key="test-key",
        credits_fetcher=lambda **_: {"credits_used": 0, "credit_limit": 1000},
    )
    row = pd.read_csv(result.paths["proofs"]).iloc[0]

    assert calls == []
    assert result.summary["execution_enabled"] is False
    assert row["mode_proof_status"] == "execution_disabled"
    assert row["blocker"] == "QPA_ENABLE_WIZARD_CUSTOM_SERIES_PROOF_not_true"


def test_execute_request_preserves_credit_reserve(tmp_path, monkeypatch):
    _write_ready_queue(tmp_path)
    calls = []
    monkeypatch.setenv("QPA_ENABLE_WIZARD_CUSTOM_SERIES_PROOF", "true")
    monkeypatch.setattr(mode_proof, "fetch_custom_series_backtest", lambda *args, **kwargs: calls.append((args, kwargs)))

    result = mode_proof.run_hyperliquid_wizard_mode_proofs(
        root=tmp_path,
        execute=True,
        api_key="test-key",
        credits_fetcher=lambda **_: {"credits_used": 900, "credit_limit": 1000},
    )
    row = pd.read_csv(result.paths["proofs"]).iloc[0]

    assert calls == []
    assert result.summary["credit_preflight_status"] == "BLOCKED"
    assert result.summary["external_proof_requests"] == 0
    assert row["mode_proof_status"] == "credit_blocked"
    assert row["blocker"] == "insufficient_credits_after_reserve"
    assert row["credits_estimated"] == 0


def test_request_failure_is_not_retried_until_next_utc_day(tmp_path, monkeypatch):
    _write_ready_queue(tmp_path)
    calls = []
    monkeypatch.setenv("QPA_ENABLE_WIZARD_CUSTOM_SERIES_PROOF", "true")

    def fail_request(*args, **kwargs):
        calls.append((args, kwargs))
        raise mode_proof.CryptoWizardsFetchError("vendor rejected request")

    monkeypatch.setattr(mode_proof, "fetch_custom_series_backtest", fail_request)
    now = datetime(2026, 8, 10, 0, 5, tzinfo=UTC)
    kwargs = {
        "root": tmp_path,
        "execute": True,
        "api_key": "test-key",
        "credits_fetcher": lambda **_: {"credits_used": 0, "credit_limit": 1000},
    }

    first = mode_proof.run_hyperliquid_wizard_mode_proofs(now=now, **kwargs)
    same_day = mode_proof.run_hyperliquid_wizard_mode_proofs(
        now=now + timedelta(hours=2), **kwargs
    )
    next_day = mode_proof.run_hyperliquid_wizard_mode_proofs(
        now=now + timedelta(days=1), **kwargs
    )

    assert len(calls) == 2
    assert first.summary["external_proof_requests"] == 1
    assert first.summary["selected_request_failed"] == 1
    assert first.summary["selected_failures_quarantined"] is True
    assert same_day.summary["selected"] == 0
    assert next_day.summary["external_proof_requests"] == 1


def test_exhaustive_queue_uses_exact_orientation_and_cost_units(tmp_path):
    _write_exhaustive_inputs(tmp_path)

    result = mode_proof.build_exhaustive_wizard_mode_proof_queue(root=tmp_path)
    queue = pd.read_csv(result.paths["queue"])
    reverse = queue.loc[queue["orientation"].eq("reverse")].iloc[0]
    request = mode_proof._request_from_candidate(reverse, root=tmp_path)
    payload = request.payload()

    assert len(queue) == 2
    assert queue["vendor_custom_series_eligible"].all()
    assert reverse["asset_x"] == "ALGO"
    assert reverse["asset_y"] == "ADA"
    assert reverse["commission_rate"] == 0.001
    assert reverse["slippage_rate"] == 0.0005
    assert payload["params"]["series_1_closes"][0] == 101.1
    assert payload["params"]["series_2_closes"][0] == 1.1


def test_exhaustive_input_audit_validates_every_eligible_cell_without_api(tmp_path):
    _write_exhaustive_inputs(tmp_path)
    queue = mode_proof.build_exhaustive_wizard_mode_proof_queue(root=tmp_path)

    audit = mode_proof.audit_exhaustive_wizard_mode_proof_inputs(
        root=tmp_path,
        queue_path=queue.paths["queue"],
    )
    frame = pd.read_csv(audit.paths["input_audit"])

    assert audit.summary["status"] == "PASS"
    assert audit.summary["eligible_rows"] == 2
    assert audit.summary["ready_rows"] == 2
    assert audit.summary["api_calls_performed"] == 0
    assert audit.summary["credits_consumed"] == 0
    assert frame["input_audit_status"].eq("READY").all()
    assert frame["request_fingerprint"].str.len().eq(64).all()


def test_copula_queue_uses_one_shared_window_for_both_orientations(tmp_path):
    _write_exhaustive_inputs(tmp_path)
    ledger_path = (
        tmp_path
        / "reports"
        / "active"
        / "exhaustive_wizard_pair_detail_mode_ledger.csv"
    )
    ledger = pd.read_csv(ledger_path)
    ledger["exact_mode"] = "Copula"
    ledger.loc[ledger["orientation"].eq("original"), "periods_analyzed"] = 55
    ledger.loc[ledger["orientation"].eq("reverse"), "periods_analyzed"] = 60
    ledger.to_csv(ledger_path, index=False)

    queue_result = mode_proof.build_exhaustive_wizard_mode_proof_queue(root=tmp_path)
    queue = pd.read_csv(queue_result.paths["queue"])
    original = queue.loc[queue["orientation"].eq("original")].iloc[0]
    reverse = queue.loc[queue["orientation"].eq("reverse")].iloc[0]
    original_request = mode_proof._request_from_candidate(original, root=tmp_path)
    reverse_request = mode_proof._request_from_candidate(reverse, root=tmp_path)

    assert queue["proof_observations"].eq(55).all()
    assert original_request.series_1_closes == reverse_request.series_2_closes
    assert original_request.series_2_closes == reverse_request.series_1_closes
    audit = mode_proof.audit_exhaustive_wizard_mode_proof_inputs(
        root=tmp_path,
        queue_path=queue_result.paths["queue"],
    )
    assert audit.summary["status"] == "PASS"


def test_copula_input_audit_blocks_nonidentical_orientation_windows(tmp_path):
    _write_exhaustive_inputs(tmp_path)
    ledger_path = (
        tmp_path
        / "reports"
        / "active"
        / "exhaustive_wizard_pair_detail_mode_ledger.csv"
    )
    ledger = pd.read_csv(ledger_path)
    ledger["exact_mode"] = "Copula"
    ledger.to_csv(ledger_path, index=False)
    queue_result = mode_proof.build_exhaustive_wizard_mode_proof_queue(root=tmp_path)
    queue = pd.read_csv(queue_result.paths["queue"])
    queue.loc[queue["orientation"].eq("reverse"), "proof_observations"] = 50
    queue.to_csv(queue_result.paths["queue"], index=False)

    audit = mode_proof.audit_exhaustive_wizard_mode_proof_inputs(
        root=tmp_path,
        queue_path=queue_result.paths["queue"],
    )
    frame = pd.read_csv(audit.paths["input_audit"])

    assert audit.summary["status"] == "BLOCKED"
    assert audit.summary["blocked_rows"] == 2
    assert frame["retry_safety_status"].eq(
        "BLOCKED_COPULA_ORIENTATION_CONTRACT"
    ).all()
    assert frame["blocker"].eq(
        "copula_orientation_requests_not_exact_swaps"
    ).all()


def test_exhaustive_input_audit_fails_closed_on_missing_history(tmp_path):
    _write_exhaustive_inputs(tmp_path)
    queue = mode_proof.build_exhaustive_wizard_mode_proof_queue(root=tmp_path)
    frame = pd.read_csv(queue.paths["queue"])
    frame["local_history_path"] = "data/raw/missing.json"
    frame.to_csv(queue.paths["queue"], index=False)

    audit = mode_proof.audit_exhaustive_wizard_mode_proof_inputs(
        root=tmp_path,
        queue_path=queue.paths["queue"],
    )

    assert audit.summary["status"] == "BLOCKED"
    assert audit.summary["blocked_rows"] == 2
    assert audit.summary["api_calls_performed"] == 0


def test_input_audit_blocks_unchanged_payload_after_vendor_4xx(tmp_path):
    _write_ready_queue(tmp_path)
    queue_path = tmp_path / "reports" / "active" / "hyperliquid_wizard_hypothesis_queue.csv"
    candidate = pd.read_csv(queue_path).iloc[0]
    request_path = tmp_path / "data" / "raw" / "prior_ou_400_request.json"
    request_path.write_text(
        json.dumps(mode_proof._request_from_candidate(candidate, root=tmp_path).payload()),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "pair": candidate["pair"],
                "local_interval": candidate["local_interval"],
                "exact_mode": candidate["exact_mode"],
                "orientation": "",
                "proof_observations": 60,
                "mode_proof_status": "request_failed",
                "blocker": "400 Client Error: Bad Request",
                "request_path": "data/raw/prior_ou_400_request.json",
            }
        ]
    ).to_csv(
        tmp_path / "reports" / "active" / "hyperliquid_wizard_vendor_mode_proofs.csv",
        index=False,
    )

    audit = mode_proof.audit_exhaustive_wizard_mode_proof_inputs(
        root=tmp_path,
        queue_path=queue_path,
    )
    row = pd.read_csv(audit.paths["input_audit"]).iloc[0]

    assert audit.summary["status"] == "BLOCKED"
    assert audit.summary["unchanged_vendor_4xx_rows"] == 1
    assert row["wire_spread_type"] == "Ou"
    assert row["retry_safety_status"] == "BLOCKED_UNCHANGED_VENDOR_4XX_PAYLOAD"
    assert row["blocker"] == "unchanged_payload_previously_rejected_by_vendor_4xx"


def test_input_audit_allows_changed_ou_wire_payload_after_vendor_4xx(tmp_path):
    _write_ready_queue(tmp_path)
    queue_path = tmp_path / "reports" / "active" / "hyperliquid_wizard_hypothesis_queue.csv"
    candidate = pd.read_csv(queue_path).iloc[0]
    legacy_payload = mode_proof._request_from_candidate(candidate, root=tmp_path).payload()
    legacy_payload["params"]["spread_type"] = "OU"
    request_path = tmp_path / "data" / "raw" / "prior_ou_400_request.json"
    request_path.write_text(json.dumps(legacy_payload), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "pair": candidate["pair"],
                "local_interval": candidate["local_interval"],
                "exact_mode": candidate["exact_mode"],
                "orientation": "",
                "proof_observations": 60,
                "mode_proof_status": "request_failed",
                "blocker": "400 Client Error: Bad Request",
                "request_path": "data/raw/prior_ou_400_request.json",
            }
        ]
    ).to_csv(
        tmp_path / "reports" / "active" / "hyperliquid_wizard_vendor_mode_proofs.csv",
        index=False,
    )

    audit = mode_proof.audit_exhaustive_wizard_mode_proof_inputs(
        root=tmp_path,
        queue_path=queue_path,
    )
    row = pd.read_csv(audit.paths["input_audit"]).iloc[0]

    assert audit.summary["status"] == "PASS"
    assert audit.summary["changed_after_vendor_4xx_rows"] == 1
    assert row["wire_spread_type"] == "Ou"
    assert row["prior_vendor_4xx_count"] == 1
    assert row["retry_safety_status"] == "READY_PAYLOAD_CHANGED_AFTER_VENDOR_4XX"


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
    result = mode_proof.run_hyperliquid_wizard_mode_proofs(
        root=tmp_path,
        execute=True,
        api_key="test-key",
        credits_fetcher=lambda **_: {"credits_used": 0, "credit_limit": 1000},
    )
    row = pd.read_csv(result.paths["proofs"]).iloc[0]

    assert result.summary["completed"] == 1
    assert result.summary["queue_completed"] == 0
    assert result.summary["queue_responses_captured"] == 1
    assert captured["api_key"] == "test-key"
    assert captured["payload"]["params"]["strategy"] == "ZScoreRoll"
    assert captured["payload"]["params"]["spread_type"] == "Ou"
    assert len(captured["payload"]["params"]["series_1_opens"]) == 60
    assert len(captured["payload"]["params"]["series_2_opens"]) == 60
    assert row["mode_proof_status"] == "completed"
    assert row["vendor_response_captured"]
    assert not row["formula_proof_complete"]
    assert row["mode_computation_source"] == "crypto_wizards_custom_series_backtest"
    assert row["proof_window_kind"] == "scanner_horizon_parity"
    assert row["proof_validity"] == "RESEARCH_DIAGNOSTIC_WEAK"
    assert "vendor_performance_accounting_not_reconstructed" in row["validity_blocker"]
    assert not row["training_eligible"]
    assert not row["promotion_allowed"]
    assert (tmp_path / row["request_path"]).exists()
    assert (tmp_path / row["response_path"]).exists()

    second = mode_proof.run_hyperliquid_wizard_mode_proofs(
        root=tmp_path,
        execute=True,
        api_key="test-key",
        credits_fetcher=lambda **_: {"credits_used": 0, "credit_limit": 1000},
    )
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


def test_static_zscorer_uses_the_same_exact_static_spread_reconstruction():
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
            "strategy": "ZScoreRoll",
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


def _vendor_response_from_spread(spread: np.ndarray, *, window: int):
    series = pd.Series(spread)
    zscore = (spread - spread.mean()) / spread.std(ddof=1)
    zscore_roll = (
        series.sub(series.rolling(window, min_periods=window).mean())
        .div(series.rolling(window, min_periods=window).std(ddof=1))
        .fillna(0.0)
    )
    return {
        "history": {
            "spread_stats": {
                "spread": spread.tolist(),
                "zscore": zscore.tolist(),
                "zscore_roll": zscore_roll.tolist(),
                "log_used": False,
            }
        }
    }


def test_preregistered_dynamic_kalman_candidate_requires_exact_series_match():
    x = np.linspace(100.0, 140.0, 80)
    y = 5.0 + 0.4 * x + np.sin(np.arange(80) / 5.0)
    spread = mode_proof._kalman_dynamic_spread_candidate(x, y)
    request = {
        "params": {
            "strategy": "Spread",
            "spread_type": "Dynamic",
            "roll_w": 20,
            "series_1_closes": x.tolist(),
            "series_2_closes": y.tolist(),
        }
    }

    exact = mode_proof._formula_parity_metrics(
        request,
        _vendor_response_from_spread(spread, window=20),
    )
    mutated = mode_proof._formula_parity_metrics(
        request,
        _vendor_response_from_spread(spread + 0.01, window=20),
    )

    assert exact["vendor_formula_parity_status"] == "exact_reconstruction"
    assert exact["vendor_spread_formula"].startswith("candidate_v1:kalman")
    assert mutated["vendor_formula_parity_status"] == "mismatch"


def test_dynamic_v2_holdout_candidate_uses_post_update_residual_and_warmup():
    x = np.log(np.linspace(0.8, 1.4, 80))
    y = np.log(np.linspace(0.2, 0.5, 80))

    spread = mode_proof._kalman_dynamic_spread_candidate_v2_holdout(x, y)

    assert np.array_equal(spread[:30], np.zeros(30))
    assert np.isfinite(spread).all()
    assert np.any(np.abs(spread[30:]) > 0.0)


def test_preregistered_ou_mle_candidate_requires_exact_series_match():
    x = np.linspace(50.0, 90.0, 90)
    residual = np.zeros(90)
    shocks = 0.03 * np.sin(np.arange(90) / 3.0)
    for index in range(1, len(residual)):
        residual[index] = 0.92 * residual[index - 1] + shocks[index]
    y = 3.0 + 1.25 * x + residual
    spread = mode_proof._ou_profile_mle_spread_candidate(x, y)
    request = {
        "params": {
            "strategy": "ZScoreRoll",
            "spread_type": "OU",
            "roll_w": 20,
            "series_1_closes": x.tolist(),
            "series_2_closes": y.tolist(),
        }
    }

    metrics = mode_proof._formula_parity_metrics(
        request,
        _vendor_response_from_spread(spread, window=20),
    )

    assert metrics["vendor_formula_parity_status"] == "exact_reconstruction"
    assert metrics["vendor_spread_formula"].startswith("candidate_v1:ou_profile")


def test_copula_response_observables_are_visible_but_never_prove_formula_parity():
    request = {
        "params": {
            "strategy": "Copula",
            "roll_w": 20,
            "series_1_closes": [float(value) for value in range(1, 81)],
            "series_2_closes": [float(value) for value in range(81, 161)],
        }
    }
    response = {
        "data": {"copula_name": "clayton"},
        "history": [
            {"u1_given_u2": 0.35, "u2_given_u1": 0.62},
            {"u1_given_u2": 0.91, "u2_given_u1": 0.08},
        ],
    }

    response_metrics = mode_proof._response_metrics(response)
    formula_metrics = mode_proof._formula_parity_metrics(request, response)

    assert response_metrics["vendor_copula_name"] == "clayton"
    assert response_metrics["vendor_u1_given_u2"] == pytest.approx(0.91)
    assert response_metrics["vendor_u2_given_u1"] == pytest.approx(0.08)
    assert response_metrics["vendor_copula_history_points"] == 2
    assert formula_metrics["vendor_formula_parity_status"] == "not_available"
