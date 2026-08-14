from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json

import pandas as pd
import requests

from quant_platform.orchestration.wizard_pair_detail_api_pilot import (
    _endpoint_params,
    run_wizard_pair_detail_api_pilot,
)


PAIR_GROUP = "binance|daily|ETH|WIF"


def test_backtest_probe_uses_verified_cost_defaults_without_implicit_stop():
    params = _endpoint_params(
        "backtest",
        {
            "symbol_1": "BTCUSDT",
            "symbol_2": "ETHUSDT",
            "exchange": "Binance",
            "interval": "Daily",
            "period": 365,
            "spread_type": "Static",
            "roll_w": 42,
            "strategy": "Spread",
        },
    )

    assert params["commission_rate"] == 0.001
    assert params["slippage_rate"] == 0.0005
    assert "stop_loss_rate" not in params


def _write_inputs(root: Path) -> None:
    active = root / "reports" / "active"
    active.mkdir(parents=True)
    pd.DataFrame([{"pair_group_key": PAIR_GROUP}]).to_csv(
        active / "exhaustive_wizard_api_refresh_pair_detail_queue.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "pair_group_key": PAIR_GROUP,
                "api_source_row_id": "wapirow_copula",
                "pair_id": "2979683",
                "api_symbol_1": "ETHUSDC",
                "api_symbol_2": "WIFUSDC",
                "api_exchange": "Binance",
                "api_interval": "Daily",
                "api_period": "365",
                "api_roll_window": "10",
                "api_exact_mode": "Copula",
                "spread_type": "dynamic",
            },
            {
                "pair_group_key": PAIR_GROUP,
                "api_source_row_id": "wapirow_zscore",
                "pair_id": "2979683",
                "api_symbol_1": "ETHUSDC",
                "api_symbol_2": "WIFUSDC",
                "api_exchange": "Binance",
                "api_interval": "Daily",
                "api_period": "365",
                "api_roll_window": "10",
                "api_exact_mode": "Dyn (ZScoreR)",
                "spread_type": "dynamic",
            },
        ]
    ).to_csv(
        active / "exhaustive_wizard_api_refresh_source_accounting.csv",
        index=False,
    )
    (active / "exhaustive_wizard_api_refresh_manifest.json").write_text(
        json.dumps({"refresh_id": "ewapi_test"}),
        encoding="utf-8",
    )


def test_pair_detail_pilot_defaults_to_zero_credit_plan(tmp_path: Path) -> None:
    _write_inputs(tmp_path)

    result = run_wizard_pair_detail_api_pilot(root=tmp_path)

    assert result.summary["execute"] is False
    assert result.summary["planned_credits"] == 31
    assert result.summary["attempted_credits"] == 0
    assert result.summary["authority"] == "PREFLIGHT_ONLY"
    assert result.summary["promotion_authority"] is False
    manifest = pd.read_csv(result.paths["manifest_csv"], keep_default_na=False)
    assert len(manifest) == 6
    assert manifest["status"].eq("PLANNED").all()


def test_pair_detail_pilot_archives_fields_and_observed_credit_delta(tmp_path: Path) -> None:
    _write_inputs(tmp_path)
    credit_values = iter([300, 331])

    def credits_fetcher(**_: object) -> int:
        return next(credit_values)

    def endpoint_fetcher(*, endpoint_path: str, **_: object) -> dict[str, object]:
        return {
            "/v1beta/spread": {"spread": [0.1, 0.2], "zscore": [1.0, 2.0]},
            "/v1beta/zscores": {
                "history": {"zscore": [1.0, 2.0], "zscore_roll": [0.8, 1.5]}
            },
            "/v1beta/backtest": {
                "total_return": 0.2,
                "sharpe_ratio": 1.4,
                "max_drawdown": -0.1,
            },
            "/v1beta/cointegration": {
                "is_coint": True,
                "p_value": 0.01,
                "t_stat": -3.5,
            },
            "/v1beta/copula": {
                "copula_name": "studentt",
                "u1_given_u2": 0.95,
                "u2_given_u1": 0.05,
            },
            "/v1beta/correlations": {
                "pearson": 0.8,
                "spearman": 0.7,
                "kendall": 0.6,
            },
        }[endpoint_path]

    result = run_wizard_pair_detail_api_pilot(
        root=tmp_path,
        execute=True,
        api_key="secret-test-key",
        credits_fetcher=credits_fetcher,
        endpoint_fetcher=endpoint_fetcher,
    )

    assert result.summary["completed_endpoints"] == 6
    assert result.summary["planned_credits"] == 31
    assert result.summary["completed_credits"] == 31
    assert result.summary["observed_credit_delta"] == 31
    assert result.summary["pilot_complete"] is True
    assert result.summary["authority"] == "COMPLETE_SCHEMA_PROBE_ONLY"
    assert result.summary["ecm_fields_found"] is False
    assert result.summary["dashboard_pair_detail_complete"] is False
    assert result.summary["request_config"]["strategy"] == "Copula"
    assert result.summary["request_config"]["symbol_1"] == "ETHUSDC"
    coverage = pd.read_csv(result.paths["coverage"], keep_default_na=False)
    assert coverage.loc[coverage["field_group"].eq("pearson"), "status"].iloc[0] == "FOUND"
    assert coverage.loc[coverage["field_group"].eq("ecm_x"), "status"].iloc[0] == "MISSING"
    raw_files = list(
        (tmp_path / result.summary["raw_snapshot_directory"]).glob("*.json")
    )
    assert len(raw_files) == 6
    assert all("secret-test-key" not in path.read_text(encoding="utf-8") for path in raw_files)


def test_pair_detail_pilot_blocks_before_paid_calls_when_reserve_would_break(
    tmp_path: Path,
) -> None:
    _write_inputs(tmp_path)
    calls: list[str] = []

    def endpoint_fetcher(*, endpoint_path: str, **_: object) -> dict[str, object]:
        calls.append(endpoint_path)
        return {}

    result = run_wizard_pair_detail_api_pilot(
        root=tmp_path,
        execute=True,
        api_key="secret-test-key",
        credits_fetcher=lambda **_: 890,
        endpoint_fetcher=endpoint_fetcher,
    )

    assert calls == []
    assert result.summary["completed_endpoints"] == 0
    assert result.summary["attempted_credits"] == 0
    assert result.summary["authority"] == "BLOCKED_SCHEMA_PROBE"
    assert result.summary["blocker"] == "insufficient_credits_after_reserve"


def test_copula_backtest_uses_documented_levels_and_ignores_spread() -> None:
    params = _endpoint_params(
        "backtest",
        {
            "symbol_1": "ETHUSDC",
            "symbol_2": "WIFUSDC",
            "exchange": "Binance",
            "interval": "Daily",
            "period": 365,
            "strategy": "Copula",
            "spread_type": "Dynamic",
            "roll_w": 10,
        },
    )

    assert params["entry_level"] == 0.05
    assert params["exit_level"] == 0.50
    assert "spread_type" not in params
    assert "roll_w" not in params


def test_failed_endpoint_archives_sanitized_response_and_credit_charge(
    tmp_path: Path,
) -> None:
    _write_inputs(tmp_path)
    credit_values = iter([300, 306])

    def endpoint_fetcher(**_: object) -> dict[str, object]:
        response = requests.Response()
        response.status_code = 400
        response.headers["Content-Type"] = "application/json"
        response._content = b'{"detail":"invalid copula entry level"}'
        raise requests.HTTPError("bad request secret-test-key", response=response)

    result = run_wizard_pair_detail_api_pilot(
        root=tmp_path,
        execute=True,
        api_key="secret-test-key",
        endpoint_names=("backtest",),
        credits_fetcher=lambda **_: next(credit_values),
        endpoint_fetcher=endpoint_fetcher,
    )

    manifest = pd.read_csv(result.paths["manifest_csv"], keep_default_na=False)
    assert manifest.loc[0, "status"] == "FAILED"
    assert manifest.loc[0, "credit_charge_status"] == "CHARGED_FAILED_CONFIRMED"
    assert result.summary["charged_failed_credits"] == 6
    failure_path = tmp_path / manifest.loc[0, "evidence_path"]
    failure_text = failure_path.read_text(encoding="utf-8")
    assert "invalid copula entry level" in failure_text
    assert "secret-test-key" not in failure_text
    assert "[REDACTED]" in failure_text


def test_single_endpoint_retry_merges_prior_successful_bundle(tmp_path: Path) -> None:
    _write_inputs(tmp_path)

    def endpoint_fetcher(*, endpoint_path: str, **_: object) -> dict[str, object]:
        return {
            "endpoint": endpoint_path,
            "total_return": 0.2 if endpoint_path.endswith("backtest") else None,
        }

    first_credits = iter([300, 331])
    run_wizard_pair_detail_api_pilot(
        root=tmp_path,
        execute=True,
        api_key="secret-test-key",
        now=datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc),
        credits_fetcher=lambda **_: next(first_credits),
        endpoint_fetcher=endpoint_fetcher,
    )
    retry_credits = iter([331, 337])
    result = run_wizard_pair_detail_api_pilot(
        root=tmp_path,
        execute=True,
        api_key="secret-test-key",
        now=datetime(2026, 8, 8, 12, 1, tzinfo=timezone.utc),
        endpoint_names=("backtest",),
        credits_fetcher=lambda **_: next(retry_credits),
        endpoint_fetcher=endpoint_fetcher,
    )

    manifest = pd.read_csv(result.paths["manifest_csv"], keep_default_na=False)
    history = pd.read_csv(result.paths["attempt_history"], keep_default_na=False)
    assert len(manifest) == 6
    assert manifest["status"].eq("COMPLETED").all()
    assert len(history) == 7
    assert result.summary["requested_endpoints"] == ["backtest"]
    assert result.summary["planned_credits"] == 6
    assert result.summary["completed_endpoints"] == 6
    assert result.summary["pilot_complete"] is True


def test_plan_only_run_does_not_overwrite_completed_active_bundle(
    tmp_path: Path,
) -> None:
    _write_inputs(tmp_path)
    credit_values = iter([300, 331])
    run_wizard_pair_detail_api_pilot(
        root=tmp_path,
        execute=True,
        api_key="secret-test-key",
        credits_fetcher=lambda **_: next(credit_values),
        endpoint_fetcher=lambda **_: {"field": 1},
    )
    canonical_path = (
        tmp_path / "reports" / "active" / "wizard_pair_detail_api_pilot_manifest.csv"
    )
    before = canonical_path.read_text(encoding="utf-8")

    plan = run_wizard_pair_detail_api_pilot(root=tmp_path)

    assert plan.paths["manifest_csv"].name == (
        "wizard_pair_detail_api_pilot_plan_manifest.csv"
    )
    assert canonical_path.read_text(encoding="utf-8") == before
    active = pd.read_csv(canonical_path, keep_default_na=False)
    assert active["status"].eq("COMPLETED").all()
