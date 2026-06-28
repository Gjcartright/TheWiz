import json
import os
import subprocess
import pandas as pd
import pytest
from pathlib import Path

import quant_platform.cli as cli
from quant_platform.cli import (
    append_learning_outcome,
    build_dydx_long_history_pair,
    _resolve_research_funding_path,
    build_paper_plan_from_cli,
    crypto_wizards_live_coverage_report,
    crypto_wizards_min5_request_template_report,
    crawl_crypto_wizards_min5,
    dydx_two_leg_request_template_report,
    dydx_pair_expansion_plan_report,
    dydx_live_market_selector_report,
    dydx_long_history_plan_report,
    dydx_execution_checklist_report,
    dydx_order_adapter_contract_report,
    dydx_long_history_coverage_report,
    export_dydx_funding_payload,
    fetch_dydx_two_leg_data,
    fetch_dydx_funding,
    fetch_dydx_long_history_windows,
    funding_coverage_report,
    funding_requirements_report,
    funding_template_check_report,
    funding_template_report,
    funded_research_spine,
    import_crypto_wizards_backtest_history,
    import_crypto_wizards_payload,
    import_crypto_wizards_zscores_history,
    import_latest_pair_detail_download,
    import_funding_template,
    import_learning_outcomes_from_template,
    import_pair_detail_capture,
    inspect_pair_detail_capture,
    pair_detail_capture_preflight,
    learning_outcome_template_check_report,
    learning_outcome_template_report,
    materialize_p2_rerun_subset,
    print_funding_coverage,
    print_funding_requirements,
    print_funding_template,
    print_funding_template_check,
    print_learning_outcome_template,
    priority_action_plan,
    paper_execution_preflight_report,
    priority_gap_test_report,
    priority_runbook,
    priority_readiness_report,
    priority_spine_dashboard_report,
    research_spine,
    research_unblock_plan_report,
    rerun_p2_acceptance_evidence,
    run_dydx_local_pair_universe,
    run_dydx_pair_expansion,
    run_dydx_long_history,
    run_pair_detail_experiments,
    run_paper_plan,
    paper_venue_preflight_report,
    strategy_acceptance_checklist_report,
    strategy_failure_attribution_report,
    zscore_threshold_sweep_report,
    verify_crypto_wizards_live_artifacts,
)


def test_resolve_research_funding_path_prefers_research_path(tmp_path):
    primary = tmp_path / "primary.csv"
    legacy = tmp_path / "legacy.csv"
    assert _resolve_research_funding_path(primary, legacy) == primary
    assert _resolve_research_funding_path(None, legacy) == legacy
    assert _resolve_research_funding_path(None, None) is None


def test_cli_paper_plan_blocks_research_rejected_strategy(tmp_path):
    acceptance_path = tmp_path / "acceptance_report.csv"
    pd.DataFrame(
        [{"strategy_id": 1, "production_eligible": False, "acceptance_reason": "passing_pairs<2"}]
    ).to_csv(acceptance_path, index=False)

    plan, intents = build_paper_plan_from_cli(
        pair="ETH-BTC",
        strategy_id=1,
        signal=1.0,
        hedge_ratio=1.0,
        beta=1.0,
        notional_usd=1000.0,
        acceptance_path=acceptance_path,
    )

    assert plan.status == "blocked"
    assert plan.reason == "research_rejected:passing_pairs<2"
    assert intents == []


def test_write_csv_atomic_replaces_target_without_leaving_temp_file(tmp_path):
    output = tmp_path / "report.csv"
    output.write_text("old\n", encoding="utf-8")

    path = cli._write_csv_atomic(pd.DataFrame([{"step": "ready", "value": 1}]), output)

    assert path == output
    written = pd.read_csv(output)
    assert list(written["step"]) == ["ready"]
    assert not list(tmp_path.glob(".report.csv.*.tmp"))


def test_cli_paper_plan_builds_two_leg_intents_for_accepted_strategy(tmp_path):
    acceptance_path = tmp_path / "acceptance_report.csv"
    pd.DataFrame([{"strategy_id": 1, "production_eligible": True, "acceptance_reason": "passed"}]).to_csv(
        acceptance_path, index=False
    )

    plan, intents = build_paper_plan_from_cli(
        pair="SOL-ETH",
        strategy_id=1,
        signal=-1.0,
        hedge_ratio=1.5,
        beta=1.0,
        notional_usd=1000.0,
        acceptance_path=acceptance_path,
    )

    assert plan.status == "paper_ready"
    assert [intent["market"] for intent in intents] == ["SOL-USD", "ETH-USD"]
    assert [intent["side"] for intent in intents] == ["BUY", "SELL"]
    assert sum(intent["size"] for intent in intents) == 1000.0


def test_cli_paper_plan_writes_journal_for_rejected_strategy(tmp_path):
    acceptance_path = tmp_path / "acceptance_report.csv"
    journal_path = tmp_path / "paper_trading_journal.csv"
    pd.DataFrame(
        [{"strategy_id": 1, "production_eligible": False, "acceptance_reason": "passing_pairs<2"}]
    ).to_csv(acceptance_path, index=False)

    run_paper_plan(
        pair="ETH-BTC",
        strategy_id=1,
        signal=1.0,
        hedge_ratio=1.0,
        beta=1.0,
        notional_usd=1000.0,
        acceptance_path=acceptance_path,
        journal_path=journal_path,
    )

    journal = pd.read_csv(journal_path)
    assert list(journal["pair"]) == ["ETH-BTC"]
    assert list(journal["plan_status"]) == ["blocked"]
    assert list(journal["plan_reason"]) == ["research_rejected:passing_pairs<2"]


def test_cli_paper_plan_journals_dydx_config_blockers_before_submission(tmp_path, monkeypatch):
    acceptance_path = tmp_path / "acceptance_report.csv"
    journal_path = tmp_path / "paper_trading_journal.csv"
    pd.DataFrame([{"strategy_id": 1, "production_eligible": True, "acceptance_reason": "passed"}]).to_csv(
        acceptance_path, index=False
    )
    monkeypatch.delenv("DYDX_TESTNET_WALLET_ADDRESS", raising=False)
    monkeypatch.delenv("DYDX_TESTNET_PRIVATE_KEY", raising=False)
    monkeypatch.delenv("DYDX_TESTNET_SUBMIT_ORDERS", raising=False)
    monkeypatch.setattr("quant_platform.execution.dydx_v4_client_installed", lambda: False)

    run_paper_plan(
        pair="ETH-BTC",
        strategy_id=1,
        signal=1.0,
        hedge_ratio=1.0,
        beta=1.0,
        notional_usd=1000.0,
        acceptance_path=acceptance_path,
        journal_path=journal_path,
    )

    journal = pd.read_csv(journal_path)
    assert list(journal["plan_status"]) == ["blocked"]
    assert journal["plan_reason"].iloc[0].startswith("dydx_not_ready:")
    assert "submit_orders_false" in journal["blockers"].iloc[0]
    assert "missing_private_key" in journal["blockers"].iloc[0]
    assert journal["fills_json"].iloc[0] == "[]"


def test_cli_paper_plan_blocks_record_only_adapter_before_submission(tmp_path, monkeypatch):
    acceptance_path = tmp_path / "acceptance_report.csv"
    journal_path = tmp_path / "paper_trading_journal.csv"
    pd.DataFrame([{"strategy_id": 1, "production_eligible": True, "acceptance_reason": "passed"}]).to_csv(
        acceptance_path, index=False
    )
    monkeypatch.setattr("quant_platform.execution.dydx_v4_client_installed", lambda: True)
    monkeypatch.setenv("DYDX_TESTNET_WALLET_ADDRESS", "wallet")
    monkeypatch.setenv("DYDX_TESTNET_PRIVATE_KEY", "private")
    monkeypatch.setenv("DYDX_TESTNET_SUBMIT_ORDERS", "true")
    monkeypatch.setenv(
        "DYDX_TESTNET_ORDER_CLIENT_ADAPTER",
        "quant_platform.dydx_record_only_adapter:RecordOnlyDydxOrderAdapter",
    )

    run_paper_plan(
        pair="ETH-BTC",
        strategy_id=1,
        signal=1.0,
        hedge_ratio=1.0,
        beta=1.0,
        notional_usd=1000.0,
        acceptance_path=acceptance_path,
        journal_path=journal_path,
    )

    journal = pd.read_csv(journal_path)
    assert list(journal["plan_status"]) == ["blocked"]
    assert "record_only_dydx_order_client_adapter" in journal["plan_reason"].iloc[0]
    assert journal["fills_json"].iloc[0] == "[]"


def test_resolve_paper_venue_prefers_executable_dydx_when_multiple_venues_available(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    market_context = pd.DataFrame(
        [
            {
                "asset": "ETH",
                "venue": "dydx",
                "tradable": True,
                "execution_authority": True,
                "blocker": "",
                "open_interest": 0,
                "open_interest_usd": 0,
                "volume_24h": 0,
                "venue_lane": "dydx_execution_candidate",
            },
            {
                "asset": "BTC",
                "venue": "dydx",
                "tradable": True,
                "execution_authority": True,
                "blocker": "",
                "open_interest": 0,
                "open_interest_usd": 0,
                "volume_24h": 0,
                "venue_lane": "dydx_execution_candidate",
            },
            {
                "asset": "ETH",
                "venue": "hyperliquid",
                "tradable": True,
                "execution_authority": True,
                "blocker": "",
                "open_interest": 0,
                "open_interest_usd": 0,
                "volume_24h": 0,
                "venue_lane": "hyperliquid_research_candidate",
            },
            {
                "asset": "BTC",
                "venue": "hyperliquid",
                "tradable": True,
                "execution_authority": True,
                "blocker": "",
                "open_interest": 0,
                "open_interest_usd": 0,
                "volume_24h": 0,
                "venue_lane": "hyperliquid_research_candidate",
            },
        ]
    )
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    market_context.to_csv(processed / "market_venue_context.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair": "ETH-BTC",
                "best_execution_venue": "hyperliquid",
                "available_venues": "dydx;hyperliquid",
                "asset_x": "ETH",
                "asset_y": "BTC",
                "exchange": "hyperliquid",
            }
        ]
    ).to_csv(processed / "pair_universe.csv", index=False)

    assert cli._resolve_paper_venue("ETH-BTC", "auto") == "dydx"


def test_run_paper_plan_with_explicit_non_dydx_venue_blocks_for_execution_readiness(tmp_path):
    acceptance_path = tmp_path / "acceptance_report.csv"
    journal_path = tmp_path / "paper_trading_journal.csv"
    pd.DataFrame([{"strategy_id": 1, "production_eligible": True, "acceptance_reason": "passed"}]).to_csv(
        acceptance_path,
        index=False,
    )

    cli.run_paper_plan(
        pair="ETH-BTC",
        strategy_id=1,
        signal=1.0,
        hedge_ratio=1.0,
        beta=1.0,
        notional_usd=1000.0,
        acceptance_path=acceptance_path,
        journal_path=journal_path,
        venue="hyperliquid",
    )

    journal = pd.read_csv(journal_path)
    assert list(journal["plan_status"]) == ["blocked"]
    assert journal["plan_reason"].iloc[0].startswith("hyperliquid_not_ready")
    assert "missing_hyperliquid_order_client_adapter" in journal["blockers"].iloc[0]


def test_run_paper_plan_with_explicit_non_dydx_venue_runs_with_adapter(tmp_path, monkeypatch):
    acceptance_path = tmp_path / "acceptance_report.csv"
    journal_path = tmp_path / "paper_trading_journal.csv"
    pd.DataFrame([{"strategy_id": 1, "production_eligible": True, "acceptance_reason": "passed"}]).to_csv(
        acceptance_path,
        index=False,
    )

    adapter_module = tmp_path / "fake_hyperliquid_adapter.py"
    adapter_module.write_text(
        """
from quant_platform.execution import FillReport


class FakeHyperliquidAdapter:
    def __init__(self):
        self.calls = 0

    def place_order(self, intent, config):
        self.calls += 1
        return FillReport(
            order_id="hyperliquid-test",
            market=intent.market,
            side=intent.side,
            size=intent.size,
            avg_price=float(intent.limit_price or 0.0),
            fee=0.0,
            slippage_bps=0.0,
            status="paper_submitted",
        )
""",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("HYPERLIQUID_PAPER_ORDER_ADAPTER", "fake_hyperliquid_adapter:FakeHyperliquidAdapter")

    cli.run_paper_plan(
        pair="ETH-BTC",
        strategy_id=1,
        signal=1.0,
        hedge_ratio=1.0,
        beta=1.0,
        notional_usd=1000.0,
        acceptance_path=acceptance_path,
        journal_path=journal_path,
        venue="hyperliquid",
    )

    journal = pd.read_csv(journal_path)
    assert list(journal["plan_status"]) == ["paper_ready"]
    fills = json.loads(journal["fills_json"].iloc[0])
    assert fills and fills[0]["status"] == "paper_submitted"


def test_paper_venue_preflight_reports_missing_venue_blockers(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "build_dydx_indexer_adapter", lambda config: object())
    monkeypatch.setenv("DYDX_TESTNET_WALLET_ADDRESS", "wallet")
    monkeypatch.setenv("DYDX_TESTNET_PRIVATE_KEY", "private")
    monkeypatch.setenv("DYDX_TESTNET_SUBMIT_ORDERS", "true")

    adapter_module = tmp_path / "dydx_order_adapter.py"
    adapter_module.write_text(
        """
from quant_platform.execution import FillReport


class ReadyOrderAdapter:
    def place_order(self, intent, config):
        return FillReport(
            order_id="ready",
            market=intent.market,
            side=intent.side,
            size=intent.size,
            avg_price=0.0,
            fee=0.0,
            slippage_bps=0.0,
            status="paper_submitted",
        )
""",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("DYDX_TESTNET_ORDER_CLIENT_ADAPTER", "dydx_order_adapter:ReadyOrderAdapter")

    data_dir = tmp_path / "data" / "processed"
    reports = tmp_path / "reports"
    data_dir.mkdir(parents=True, exist_ok=True)
    reports.mkdir()
    market_context = pd.DataFrame(
        [
            {
                "asset": "ETH",
                "venue": "dydx",
                "tradable": True,
                "execution_authority": True,
                "blocker": "",
                "open_interest": 0,
                "open_interest_usd": 0,
                "volume_24h": 0,
                "venue_lane": "dydx_execution_candidate",
            },
            {
                "asset": "BTC",
                "venue": "dydx",
                "tradable": True,
                "execution_authority": True,
                "blocker": "",
                "open_interest": 0,
                "open_interest_usd": 0,
                "volume_24h": 0,
                "venue_lane": "dydx_execution_candidate",
            },
            {
                "asset": "ETH",
                "venue": "hyperliquid",
                "tradable": True,
                "execution_authority": True,
                "blocker": "",
                "open_interest": 0,
                "open_interest_usd": 0,
                "volume_24h": 0,
                "venue_lane": "hyperliquid_candidate",
            },
            {
                "asset": "BTC",
                "venue": "hyperliquid",
                "tradable": True,
                "execution_authority": True,
                "blocker": "",
                "open_interest": 0,
                "open_interest_usd": 0,
                "volume_24h": 0,
                "venue_lane": "hyperliquid_candidate",
            },
        ]
    )
    market_context.to_csv(data_dir / "market_venue_context.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair": "ETH-BTC",
                "best_execution_venue": "dydx",
                "available_venues": "dydx;hyperliquid",
                "asset_x": "ETH",
                "asset_y": "BTC",
                "exchange": "dydx",
                "combined_score": 10,
            }
        ]
    ).to_csv(data_dir / "pair_universe.csv", index=False)

    frame = paper_venue_preflight_report(pair="ETH-BTC")
    rows = frame.set_index("venue")

    assert bool(rows.loc["dydx", "ready_for_submission"]) is True
    assert bool(rows.loc["dydx", "execution_ready"]) is True
    assert bool(rows.loc["hyperliquid", "ready_for_submission"]) is False
    assert "missing_hyperliquid_order_client_adapter" in rows.loc["hyperliquid", "blockers"]


def test_paper_venue_preflight_marks_ready_when_non_dydx_adapter_ready(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    adapter_module = tmp_path / "hyperliquid_order_adapter.py"
    adapter_module.write_text(
        """
from quant_platform.execution import FillReport


class ReadyOrderAdapter:
    def place_order(self, intent, config):
        return FillReport(
            order_id="ready",
            market=intent.market,
            side=intent.side,
            size=intent.size,
            avg_price=0.0,
            fee=0.0,
            slippage_bps=0.0,
            status="paper_submitted",
        )
""",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("HYPERLIQUID_PAPER_ORDER_ADAPTER", "hyperliquid_order_adapter:ReadyOrderAdapter")

    data_dir = tmp_path / "data" / "processed"
    data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "asset": "ETH",
                "venue": "hyperliquid",
                "tradable": True,
                "execution_authority": True,
                "blocker": "",
                "open_interest": 0,
                "open_interest_usd": 0,
                "volume_24h": 0,
                "venue_lane": "hyperliquid_candidate",
            },
            {
                "asset": "BTC",
                "venue": "hyperliquid",
                "tradable": True,
                "execution_authority": True,
                "blocker": "",
                "open_interest": 0,
                "open_interest_usd": 0,
                "volume_24h": 0,
                "venue_lane": "hyperliquid_candidate",
            },
        ]
    ).to_csv(data_dir / "market_venue_context.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair": "ETH-BTC",
                "best_execution_venue": "hyperliquid",
                "available_venues": "hyperliquid",
                "asset_x": "ETH",
                "asset_y": "BTC",
                "exchange": "hyperliquid",
                "combined_score": 10,
            }
        ]
    ).to_csv(data_dir / "pair_universe.csv", index=False)

    frame = paper_venue_preflight_report(pair="ETH-BTC")
    rows = frame.set_index("venue")

    assert bool(rows.loc["hyperliquid", "ready_for_submission"]) is True
    assert bool(rows.loc["hyperliquid", "execution_ready"]) is True
    assert bool(rows.loc["hyperliquid", "adapter_ready"]) is True


def test_split_pair_assets_handles_dydx_four_segment_pairs():
    assert cli._split_pair_assets("DOGE-USD-LTC-USD") == ["DOGE", "LTC"]


def test_normalize_dydx_pair_keeps_four_segment_structure():
    assert cli._normalize_dydx_pair("DOGE-USD-LTC-USD") == "DOGE-USD-LTC-USD"


def test_verify_crypto_wizards_live_artifacts_reports_missing_payloads(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "docs").mkdir()

    with pytest.raises(SystemExit, match="missing live Crypto Wizards raw payloads"):
        verify_crypto_wizards_live_artifacts()


def test_verify_crypto_wizards_live_artifacts_accepts_payload_and_dictionary(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    raw = tmp_path / "data" / "raw"
    docs = tmp_path / "docs"
    raw.mkdir(parents=True)
    docs.mkdir()
    (raw / "prescanned.json").write_text('{"items": [{"symbol_1": "ETH", "symbol_2": "BTC"}]}', encoding="utf-8")
    pd.DataFrame(
        [{"field": "items[].symbol_1", "type": "str", "example": "ETH", "endpoint": "prescanned"}]
    ).to_csv(docs / "crypto_wizards_live_field_dictionary.csv", index=False)

    verify_crypto_wizards_live_artifacts()

    output = capsys.readouterr().out
    assert "live_payload_count: 1" in output
    assert "live_field_dictionary_exists: True" in output
    assert "live_field_dictionary_rows: 1" in output
    assert "live_ecm_fields_present: False" in output


def test_import_crypto_wizards_payload_copies_json_and_writes_dictionary(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    source = tmp_path / "downloaded.json"
    source.write_text('{"items": [{"symbol_1": "ETH", "symbol_2": "BTC"}]}', encoding="utf-8")

    import_crypto_wizards_payload(source, endpoint_name="prescanned")

    assert (tmp_path / "data" / "raw" / "prescanned.json").exists()
    dictionary = pd.read_csv(tmp_path / "docs" / "crypto_wizards_live_field_dictionary.csv")
    assert "items[].symbol_1" in set(dictionary["field"])
    assert set(dictionary["endpoint"]) == {"prescanned"}


def test_import_crypto_wizards_payload_rejects_invalid_json(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    source = tmp_path / "bad.json"
    source.write_text("not json", encoding="utf-8")

    with pytest.raises(SystemExit, match="input is not valid JSON"):
        import_crypto_wizards_payload(source)


def test_crawl_crypto_wizards_min5_writes_quality_report(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setenv("CRYPTO_WIZARDS_API_KEY", "secret")

    def fake_crawl(**kwargs):
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "pair_1_min5_cw_zscores_history.json"
        path.write_text(
            json.dumps(
                {
                    "pair_id": "1",
                    "pair": "BNB-USD-STX-USD",
                    "asset_x": "BNB-USD",
                    "asset_y": "STX-USD",
                    "exchange": "dydx",
                    "interval": "min5",
                    "history": [{"spread": idx / 100, "zscore": idx / 10} for idx in range(90)],
                }
            ),
            encoding="utf-8",
        )
        return [path]

    monkeypatch.setattr(cli, "crawl_prescanned_zscores_histories", fake_crawl)

    paths = crawl_crypto_wizards_min5(
        max_pairs=1,
        priority="Sharpe",
        cw_strategy="Spread",
        exchange="Dydx",
        period=320,
        spread_type="Static",
        roll_w=42,
        asset=None,
    )

    output = capsys.readouterr().out
    assert len(paths) == 1
    assert "crypto_wizards_min5_histories_written: 1" in output
    quality = pd.read_csv(tmp_path / "reports" / "pair_detail_quality_report.csv")
    assert bool(quality.loc[0, "research_usable"])
    assert not bool(quality.loc[0, "execution_usable"])


def test_import_crypto_wizards_zscores_history_writes_pair_detail_payload(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    source = tmp_path / "bnb_stx_zscores.json"
    source.write_text(
        json.dumps(
            {
                "data": {"zscore": -1.2, "zscore_roll": -0.8},
                "history": {
                    "spread": [idx / 100 for idx in range(90)],
                    "zscore": [idx / 10 for idx in range(90)],
                    "zscore_roll": [idx / 20 for idx in range(90)],
                    "hedge_ratio": 1.36,
                    "half_life": 9.5,
                    "hurst": 0.71,
                },
            }
        ),
        encoding="utf-8",
    )

    path = import_crypto_wizards_zscores_history(
        source,
        asset_x="BNB-USD",
        asset_y="STX-USD",
        exchange="Dydx",
        interval="Min5",
        period=320,
        spread_type="Static",
        roll_w=42,
    )

    output = capsys.readouterr().out
    assert path.exists()
    assert "imported_crypto_wizards_zscores_history:" in output
    quality = pd.read_csv(tmp_path / "reports" / "pair_detail_quality_report.csv")
    assert bool(quality.loc[0, "research_usable"])
    assert not bool(quality.loc[0, "execution_usable"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["pair"] == "BNB-USD-STX-USD"
    assert payload["history"][0]["rolling_zscore"] == 0.0


def test_import_crypto_wizards_backtest_history_writes_pair_detail_payload(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    source = tmp_path / "bnb_stx_backtest.json"
    source.write_text(
        json.dumps(
            {
                "data": {
                    "strat_returns": {"annual_return": 0.3, "mean_period_return": 0.001, "total_return": 0.12},
                    "max_drawdown": -0.04,
                    "sharpe_ratio": 2.1,
                    "sortino_ratio": 3.2,
                    "cvar": -0.03,
                    "var": -0.02,
                    "win_rate": 0.61,
                },
                "history": {
                    "spread_stats": {
                        "spread": [idx / 100 for idx in range(90)],
                        "zscore": [idx / 10 for idx in range(90)],
                        "zscore_roll": [idx / 20 for idx in range(90)],
                        "hedge_ratio": 1.36,
                        "half_life": 9.5,
                        "hurst": 0.71,
                    },
                    "bt_returns": [1.0 + idx / 1000 for idx in range(90)],
                },
            }
        ),
        encoding="utf-8",
    )

    path = import_crypto_wizards_backtest_history(
        source,
        asset_x="BNB-USD",
        asset_y="STX-USD",
        exchange="Dydx",
        interval="Min5",
        period=320,
        spread_type="Static",
        roll_w=42,
    )

    output = capsys.readouterr().out
    assert path.exists()
    assert "imported_crypto_wizards_backtest_histories: 1" in output
    quality = pd.read_csv(tmp_path / "reports" / "pair_detail_quality_report.csv")
    assert bool(quality.loc[0, "research_usable"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["sharpe"] == 2.1
    assert payload["history"][0]["bt_return"] == 1.0


def test_crypto_wizards_min5_request_template_report_writes_urls(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)

    frame = crypto_wizards_min5_request_template_report(
        asset_x="BNB-USD",
        asset_y="STX-USD",
        output_path=tmp_path / "reports" / "requests.csv",
    )

    assert len(frame) == 3
    assert "prescanned_min5_pairs" in set(frame["request_name"])
    assert "pair_min5_zscores_history" in set(frame["request_name"])
    assert "pair_min5_backtest_history" in set(frame["request_name"])
    assert frame["curl"].str.contains(r"\${CRYPTO_WIZARDS_API_KEY}", regex=True).all()
    assert not frame["curl"].str.lower().str.contains("secret").any()
    written = pd.read_csv(tmp_path / "reports" / "requests.csv")
    assert "/v1beta/zscores" in written.loc[written["request_name"] == "pair_min5_zscores_history", "url"].iloc[0]


def test_dydx_two_leg_request_template_report_resolves_pair_and_writes_requests(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)

    frame = dydx_two_leg_request_template_report(
        pair="BNB-USD-STX-USD",
        pair_id="1",
        hedge_ratio=1.36,
        limit=123,
        output_path=tmp_path / "reports" / "dydx_requests.csv",
    )

    assert len(frame) == 6
    assert "resolution=5MINS&limit=123" in frame.loc[0, "url"]
    assert "/v4/historicalFunding/STX-USD?limit=123" in frame.loc[3, "url"]
    assert "build-dydx-pair-history" in frame.loc[4, "import_command"]
    assert "funded-research-spine" in frame.loc[5, "import_command"]
    written = pd.read_csv(tmp_path / "reports" / "dydx_requests.csv")
    assert "asset_x_candles_5mins" in set(written["request_name"])


def test_dydx_two_leg_request_template_report_respects_forced_indexer_scheme(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)

    frame = dydx_two_leg_request_template_report(
        pair="BNB-USD-STX-USD",
        pair_id="1",
        indexer_base="https://indexer.dydx.trade",
        indexer_scheme="http",
        output_path=tmp_path / "reports" / "dydx_http_requests.csv",
    )

    get_rows = frame[frame["method"] == "GET"]
    assert not get_rows.empty
    assert all(url.startswith("http://indexer.dydx.trade") for url in get_rows["url"])


def test_fetch_url_scheme_variants_prefers_forced_scheme(tmp_path):
    candidates_http = cli._fetch_url_scheme_variants(
        "https://indexer.dydx.trade/v4/candles/perpetualMarkets/SOL-USD",
        forced_scheme="http",
    )
    candidates_https = cli._fetch_url_scheme_variants(
        "http://indexer.dydx.trade/v4/candles/perpetualMarkets/SOL-USD",
        forced_scheme="https",
    )
    assert candidates_http == ["http://indexer.dydx.trade/v4/candles/perpetualMarkets/SOL-USD"]
    assert candidates_https == ["https://indexer.dydx.trade/v4/candles/perpetualMarkets/SOL-USD"]


def test_indexer_base_normalization_adds_https_for_host_only_value():
    assert cli._normalize_indexer_base("indexer.dydx.trade") == "https://indexer.dydx.trade"
    assert cli._normalize_indexer_base("   indexer.dydx.trade  ") == "https://indexer.dydx.trade"


def test_indexer_base_normalization_preserves_https_urls():
    assert cli._normalize_indexer_base("https://indexer.dydx.trade") == "https://indexer.dydx.trade"
    assert cli._normalize_indexer_base("http://indexer.dydx.trade/") == "http://indexer.dydx.trade"


def test_indexer_base_with_scheme_normalizes_host_only_base(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)

    calls: list[str] = []

    def fake_fetch(url, output_path, timeout=30.0, max_retries=3, allow_stale_fetch=False, **_kwargs):
        calls.append(url)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if "historicalFunding/" in url:
            output_path.write_text(json.dumps({"historicalFunding": [{"effectiveAt": "2026-06-18T00:00:00Z", "rate": "0.0001"}]}), encoding="utf-8")
        else:
            output_path.write_text(json.dumps({"candles": []}), encoding="utf-8")
        return output_path

    monkeypatch.setattr(cli, "_fetch_public_json", fake_fetch)

    fetch_dydx_two_leg_data(
        pair="BNB-USD-STX-USD",
        pair_id="1",
        indexer_base="indexer.dydx.trade",
        indexer_scheme="https",
        output_dir=tmp_path / "manual",
    )

    assert all("https://indexer.dydx.trade" in url for url in calls)


def test_dydx_long_history_plan_report_respects_forced_indexer_scheme(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)

    frame = cli.dydx_long_history_plan_report(
        pair="SOL-USD-LINK-USD",
        windows=1,
        indexer_base="https://indexer.dydx.trade",
        indexer_scheme="http",
        output_path=tmp_path / "reports" / "dydx_long_history_plan.csv",
    )

    get_rows = frame[(frame["method"] == "GET") & frame["request_name"].str.contains("candles", case=False)]
    assert not get_rows.empty
    assert all(url.startswith("http://indexer.dydx.trade") for url in get_rows["url"])
    import_command = frame.loc[frame["request_name"] == "long_history_next_step", "import_command"].iloc[0]
    assert "--indexer-scheme http" in import_command


def test_fetch_dydx_two_leg_data_downloads_builds_and_normalizes_funding(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload
            self.text = json.dumps(payload)

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    def candles(market, base):
        return {
            "candles": [
                {
                    "startedAt": f"2026-06-18T00:{idx * 5:02d}:00.000Z",
                    "ticker": market,
                    "resolution": "5MINS",
                    "close": str(base + idx),
                    "usdVolume": "1000",
                }
                for idx in range(5)
            ]
        }

    def fake_get(url, headers=None, timeout=None):
        if "candles/perpetualMarkets/BNB-USD" in url:
            return FakeResponse(candles("BNB-USD", 600))
        if "candles/perpetualMarkets/STX-USD" in url:
            return FakeResponse(candles("STX-USD", 1))
        if "historicalFunding/BNB-USD" in url:
            return FakeResponse({"historicalFunding": [{"effectiveAt": "2026-06-18T00:00:00Z", "rate": "0.0001"}]})
        if "historicalFunding/STX-USD" in url:
            return FakeResponse({"historicalFunding": [{"effectiveAt": "2026-06-18T00:00:00Z", "rate": "0.0002"}]})
        raise AssertionError(url)

    monkeypatch.setattr(cli.requests, "get", fake_get)

    paths = fetch_dydx_two_leg_data(
        pair="BNB-USD-STX-USD",
        pair_id="1",
        hedge_ratio=1.36,
        beta=1.36,
        zscore_window=3,
        output_dir=tmp_path / "manual",
        derive_hedge_ratio=True,
    )

    assert paths["pair_history"].exists()
    pair_payload = json.loads(paths["pair_history"].read_text(encoding="utf-8"))
    assert {"price_x", "price_y", "spread", "zscore"}.issubset(pair_payload["history"][0])
    assert {"funding_x_bps", "funding_y_bps"}.issubset(pair_payload["history"][0])
    assert pair_payload["hedge_ratio_source"] == "derived_price_ols"
    assert pair_payload["beta_source"] == "derived_return_covariance"
    funding = pd.read_csv(paths["funding_csv"])
    assert set(funding["market"]) == {"BNB-USD", "STX-USD"}
    coverage = pd.read_csv(paths["funding_coverage"])
    assert bool(coverage.loc[0, "ready"])


def test_fetch_dydx_two_leg_data_can_rerun_p2_acceptance_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload
            self.text = json.dumps(payload)

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    def candles(market, base):
        return {
            "candles": [
                {
                    "startedAt": f"2026-06-18T00:{idx * 5:02d}:00.000Z",
                    "ticker": market,
                    "resolution": "5MINS",
                    "close": str(base + idx),
                    "usdVolume": "1000",
                }
                for idx in range(5)
            ]
        }

    def fake_get(url, headers=None, timeout=None):
        if "candles/perpetualMarkets/SOL-USD" in url:
            return FakeResponse(candles("SOL-USD", 100))
        if "candles/perpetualMarkets/LINK-USD" in url:
            return FakeResponse(candles("LINK-USD", 20))
        if "historicalFunding/SOL-USD" in url:
            return FakeResponse({"historicalFunding": [{"effectiveAt": "2026-06-18T00:00:00Z", "rate": "0.0001"}]})
        if "historicalFunding/LINK-USD" in url:
            return FakeResponse({"historicalFunding": [{"effectiveAt": "2026-06-18T00:00:00Z", "rate": "0.0002"}]})
        raise AssertionError(url)

    calls = {}

    def fake_rerun(*, input_dir, funding_path):
        calls["input_dir"] = input_dir
        calls["funding_path"] = funding_path
        return {"priority_readiness": tmp_path / "reports" / "priority_readiness.csv"}

    monkeypatch.setattr(cli.requests, "get", fake_get)
    monkeypatch.setattr(cli, "rerun_p2_acceptance_evidence", fake_rerun)

    paths = fetch_dydx_two_leg_data(
        asset_x="SOL-USD",
        asset_y="LINK-USD",
        pair_id="sol_link",
        zscore_window=3,
        output_dir=tmp_path / "manual",
        derive_hedge_ratio=True,
        run_research=True,
    )

    assert paths["pair_history"].exists()
    assert calls["input_dir"] == tmp_path / "data" / "raw" / "pair_details"
    assert calls["funding_path"] == paths["funding_csv"]


def test_fetch_dydx_two_leg_data_skips_network_when_payloads_preloaded(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    manual_dir = tmp_path / "manual"
    manual_dir.mkdir(parents=True)
    (manual_dir / "BNB-USD_5MINS_candles.json").write_text(
        json.dumps({"candles": [{"startedAt": "2026-06-18T00:00:00.000Z", "ticker": "BNB-USD", "close": "600", "usdVolume": "1000"}]}),
        encoding="utf-8",
    )
    (manual_dir / "STX-USD_5MINS_candles.json").write_text(
        json.dumps({"candles": [{"startedAt": "2026-06-18T00:00:00.000Z", "ticker": "STX-USD", "close": "1", "usdVolume": "1000"}]}),
        encoding="utf-8",
    )
    (manual_dir / "BNB-USD_funding.json").write_text(
        json.dumps({"historicalFunding": [{"effectiveAt": "2026-06-18T00:00:00Z", "rate": "0.0001"}]}),
        encoding="utf-8",
    )
    (manual_dir / "STX-USD_funding.json").write_text(
        json.dumps({"historicalFunding": [{"effectiveAt": "2026-06-18T00:00:00Z", "rate": "0.0002"}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "_fetch_public_json", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("fetch should not run")))
    funding_csv = tmp_path / "data" / "processed" / "dydx_funding.csv"

    paths = fetch_dydx_two_leg_data(
        pair="BNB-USD-STX-USD",
        pair_id="1",
        output_dir=manual_dir,
        skip_fetch=True,
        funding_path=funding_csv,
        derive_hedge_ratio=True,
    )

    assert paths["pair_history"].exists()
    pair_payload = json.loads(paths["pair_history"].read_text(encoding="utf-8"))
    assert {"funding_x_bps", "funding_y_bps"}.issubset(pair_payload["history"][0])
    assert funding_csv.exists()
    funding = pd.read_csv(paths["funding_csv"])
    assert set(funding["market"]) == {"BNB-USD", "STX-USD"}
    assert paths["funding_csv"] == funding_csv


def test_fetch_dydx_two_leg_data_respects_forced_indexer_scheme(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    calls: list[str] = []

    def fake_fetch(url, output_path, timeout=30.0, max_retries=3, allow_stale_fetch=False, **_kwargs):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        calls.append(url)
        if "candles/perpetualMarkets/BNB-USD" in url:
            payload = {
                "candles": [
                    {
                        "startedAt": "2026-06-18T00:00:00.000Z",
                        "ticker": "BNB-USD",
                        "resolution": "5MINS",
                        "close": "1",
                        "usdVolume": "1000",
                    }
                ]
            }
        elif "candles/perpetualMarkets/STX-USD" in url:
            payload = {
                "candles": [
                    {
                        "startedAt": "2026-06-18T00:00:00.000Z",
                        "ticker": "STX-USD",
                        "resolution": "5MINS",
                        "close": "2",
                        "usdVolume": "1000",
                    }
                ]
            }
        elif "historicalFunding/BNB-USD" in url or "historicalFunding/STX-USD" in url:
            payload = {"historicalFunding": [{"effectiveAt": "2026-06-18T00:00:00Z", "rate": "0.0001"}]}
        else:
            raise AssertionError(url)

        output_path.write_text(json.dumps(payload), encoding="utf-8")
        return output_path

    monkeypatch.setattr(cli, "_fetch_public_json", fake_fetch)

    paths = fetch_dydx_two_leg_data(
        pair="BNB-USD-STX-USD",
        pair_id="1",
        hedge_ratio=1.36,
        beta=1.36,
        zscore_window=3,
        indexer_base="https://indexer.dydx.trade",
        indexer_scheme="http",
        output_dir=tmp_path / "manual",
    )

    assert all(url.startswith("http://indexer.dydx.trade") for url in calls)
    request_rows = pd.read_csv(tmp_path / "reports" / "dydx_two_leg_data_requests.csv")
    get_rows = request_rows[request_rows["method"] == "GET"]
    assert not get_rows.empty
    assert all(url.startswith("http://indexer.dydx.trade") for url in get_rows["url"])
    assert paths["pair_history"].exists()


def test_import_pair_detail_capture_archives_and_reports_readiness(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    source = Path(__file__).parent / "fixtures" / "pair_detail_view_item_sample.json"

    import_pair_detail_capture(source, output_name="pair_1_capture")

    output = capsys.readouterr().out
    assert "history_rows_detected: 4" in output
    assert "experiment_ready: True" in output
    assert "ecm_history_ready: True" in output
    assert "two_leg_execution_ready: False" in output
    assert "missing_for_two_leg_backtest: price_x;price_y" in output
    assert "two_leg_ready_paths: none" in output
    assert "missing_required_fields: price_x;price_y" in output
    assert "missing_ecm_fields: none" in output
    assert "missing_two_leg_fields: price_x;price_y" in output
    assert "capture_completeness_score:" in output
    assert "capture_payload_sources:" in output
    assert "capture_fetches:" in output
    assert "capture_operator_hint: capture_leg_price_history:price_x;price_y" in output
    assert "next_capture_focus: capture_two_leg_prices:price_x;price_y" in output
    assert (tmp_path / "data" / "raw" / "pair_details" / "pair_1_capture.json").exists()
    coverage = pd.read_csv(tmp_path / "reports" / "pair_detail_history_coverage.csv")
    assert list(coverage["experiment_ready"]) == [True]
    assert (tmp_path / "reports" / "pair_detail_capture_checklist.csv").exists()


def test_import_latest_pair_detail_download_uses_newest_matching_capture(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    downloads = tmp_path / "Downloads"
    downloads.mkdir()
    old_capture = downloads / "crypto_wizards_pair_1_capture.json"
    new_capture = downloads / "crypto_wizards_pair_1_capture (1).har"
    sample = Path(__file__).parent / "fixtures" / "pair_detail_view_item_sample.json"
    old_capture.write_text('{"pair": "old"}', encoding="utf-8")
    new_capture.write_text(sample.read_text(encoding="utf-8"), encoding="utf-8")
    old_time = 1_700_000_000
    new_time = old_time + 60
    os.utime(old_capture, (old_time, old_time))
    os.utime(new_capture, (new_time, new_time))

    selected = import_latest_pair_detail_download(downloads)

    output = capsys.readouterr().out
    assert selected == new_capture
    assert f"latest_pair_detail_download: {new_capture}" in output
    assert "history_rows_detected: 4" in output
    assert (tmp_path / "data" / "raw" / "pair_details" / "pair_1_capture.json").exists()


def test_inspect_pair_detail_capture_reports_readiness_without_archiving(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    source = Path(__file__).parent / "fixtures" / "pair_detail_view_item_sample.json"

    inspect_pair_detail_capture(source)

    output = capsys.readouterr().out
    assert f"inspected_pair_detail_capture: {source}" in output
    assert "history_rows_detected: 4" in output
    assert "experiment_ready: True" in output
    assert "ecm_history_ready: True" in output
    assert "two_leg_execution_ready: False" in output
    assert "missing_for_two_leg_backtest: price_x;price_y" in output
    assert "two_leg_ready_paths: none" in output
    assert "missing_required_fields: price_x;price_y" in output
    assert "missing_ecm_fields: none" in output
    assert "missing_two_leg_fields: price_x;price_y" in output
    assert "capture_completeness_score:" in output
    assert "capture_payload_sources:" in output
    assert "capture_fetches:" in output
    assert "capture_operator_hint: capture_leg_price_history:price_x;price_y" in output
    assert "next_capture_focus: capture_two_leg_prices:price_x;price_y" in output
    assert not (tmp_path / "data" / "raw" / "pair_details").exists()
    assert not (tmp_path / "reports").exists()


def test_pair_detail_capture_preflight_writes_report_without_archiving(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    source = Path(__file__).parent / "fixtures" / "pair_detail_view_item_sample.json"

    frame = pair_detail_capture_preflight(source)
    row = frame.iloc[0]

    assert row["history_rows"] == 4
    assert bool(row["baseline_ready"]) is True
    assert bool(row["ecm_ready"]) is True
    assert bool(row["two_leg_ready"]) is False
    assert row["missing_required_fields"] == "price_x;price_y"
    assert row["next_capture_focus"] == "capture_two_leg_prices:price_x;price_y"
    assert row["capture_operator_hint"] == "capture_leg_price_history:price_x;price_y"
    assert "capture_fetches" in frame.columns
    assert "capture_payload_sources" in frame.columns
    assert "capture_operator_hint" in frame.columns
    assert (tmp_path / "reports" / "pair_detail_capture_preflight.csv").exists()
    assert not (tmp_path / "data" / "raw" / "pair_details").exists()


def test_import_pair_detail_capture_rejects_invalid_json(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    source = tmp_path / "bad_capture.json"
    source.write_text("not json", encoding="utf-8")

    with pytest.raises(SystemExit, match="input is not valid JSON"):
        import_pair_detail_capture(source)


def test_live_coverage_maps_crypto_wizards_aliases_and_flags_missing_ecm(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    pd.DataFrame(
        [
            {"field": "[].coint_eg", "type": "bool", "example": "true", "endpoint": "prescanned"},
            {"field": "[].zscore_last", "type": "float", "example": "2.1", "endpoint": "prescanned"},
            {"field": "[].zscore_roll_last", "type": "float", "example": "1.7", "endpoint": "prescanned"},
            {"field": "[].mdd", "type": "float", "example": "-0.03", "endpoint": "prescanned"},
            {"field": "[].u1_given_u2", "type": "float", "example": "0.9", "endpoint": "prescanned"},
            {"field": "[].u2_given_u1", "type": "float", "example": "0.1", "endpoint": "prescanned"},
        ]
    ).to_csv(docs / "crypto_wizards_live_field_dictionary.csv", index=False)

    report = crypto_wizards_live_coverage_report()
    fields = report[report["type"] == "field"].set_index("name")

    assert fields.loc["cointegration", "present_in_live"]
    assert fields.loc["zscore", "present_in_live"]
    assert fields.loc["rolling_zscore", "present_in_live"]
    assert fields.loc["drawdown", "present_in_live"]
    assert fields.loc["conditional_probabilities", "present_in_live"]
    assert not fields.loc["ecm_x", "present_in_live"]
    assert not fields.loc["ecm_y", "present_in_live"]
    assert not fields.loc["ecm_strength", "present_in_live"]

    pure_ecm = report.loc[report["name"] == "8: Pure ECM"].iloc[0]
    assert not pure_ecm["present_in_live"]
    assert "ecm_strength" in pure_ecm["missing_fields"]


def test_live_coverage_uses_pair_detail_snapshots_for_ecm(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    docs = tmp_path / "docs"
    pair_details = tmp_path / "data" / "raw" / "pair_details"
    docs.mkdir(parents=True)
    pair_details.mkdir(parents=True)
    pd.DataFrame(
        [
            {"field": "[].zscore_last", "type": "float", "example": "2.1", "endpoint": "prescanned"},
            {"field": "[].u1_given_u2", "type": "float", "example": "0.9", "endpoint": "prescanned"},
            {"field": "[].u2_given_u1", "type": "float", "example": "0.1", "endpoint": "prescanned"},
        ]
    ).to_csv(docs / "crypto_wizards_live_field_dictionary.csv", index=False)
    (pair_details / "pair_1.json").write_text(
        '{"pair_id":"1","pair":"BNB-USD-STX-USD","asset_x":"BNB-USD","asset_y":"STX-USD",'
        '"exchange":"dydx","ecm_x_available":true,"ecm_y_available":true,"ecm_strength_available":true,'
        '"history":[{"spread":0.1,"zscore":1.0}]}',
        encoding="utf-8",
    )

    report = crypto_wizards_live_coverage_report()
    fields = report[report["type"] == "field"].set_index("name")

    assert fields.loc["ecm_x", "present_in_live"]
    assert fields.loc["ecm_y", "present_in_live"]
    assert fields.loc["ecm_strength", "present_in_live"]
    assert fields.loc["ecm_x", "pair_detail_present"]


def test_run_pair_detail_experiments_accepts_history_payload(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    input_dir = tmp_path / "data" / "raw" / "pair_details"
    input_dir.mkdir(parents=True)
    sample = Path(__file__).parent / "fixtures" / "pair_detail_view_item_sample.json"
    (input_dir / "pair_1.json").write_text(sample.read_text(encoding="utf-8"), encoding="utf-8")

    run_pair_detail_experiments(input_dir)

    output = capsys.readouterr().out
    assert "loaded 1 pair-detail history dataset(s)" in output
    assert (tmp_path / "reports" / "experiment_results.csv").exists()


def test_priority_readiness_report_summarizes_current_gates(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "build_dydx_indexer_adapter", lambda config: object())
    monkeypatch.setenv("DYDX_TESTNET_WALLET_ADDRESS", "wallet")
    monkeypatch.setenv("DYDX_TESTNET_PRIVATE_KEY", "private")
    monkeypatch.setenv("DYDX_TESTNET_SUBMIT_ORDERS", "true")
    monkeypatch.setattr("quant_platform.execution.dydx_v4_client_installed", lambda: True)

    raw = tmp_path / "data" / "raw"
    docs = tmp_path / "docs"
    reports = tmp_path / "reports"
    pair_details = raw / "pair_details"
    raw.mkdir(parents=True)
    docs.mkdir()
    reports.mkdir()
    pair_details.mkdir(parents=True)
    (raw / "prescanned.json").write_text('{"items": [{"zscore_last": 2.1}]}', encoding="utf-8")
    pd.DataFrame([{"field": "[].zscore_last", "type": "float", "example": "2.1", "endpoint": "prescanned"}]).to_csv(
        docs / "crypto_wizards_live_field_dictionary.csv", index=False
    )
    (pair_details / "pair_1.json").write_text(
        """
        {
          "pair_id": "1",
          "pair": "BNB-USD-STX-USD",
          "asset_x": "BNB-USD",
          "asset_y": "STX-USD",
          "exchange": "dydx",
          "hedge_ratio": 1.36,
          "history": [
            {
              "spread": -0.2,
              "zscore": -2.0,
              "price_x": 600.0,
              "price_y": 1.8,
              "ecm_x": -0.3,
              "ecm_y": -0.1,
              "ecm_strength": 0.8
            },
            {
              "spread": 0.0,
              "zscore": 0.0,
              "price_x": 602.0,
              "price_y": 1.85,
              "ecm_x": -0.1,
              "ecm_y": -0.05,
              "ecm_strength": 0.7
            }
          ]
        }
        """,
        encoding="utf-8",
    )
    pd.DataFrame([{"strategy_id": 1, "production_eligible": True, "preferred_eligible": False}]).to_csv(
        reports / "acceptance_report.csv", index=False
    )

    frame = priority_readiness_report()
    gates = frame.set_index("gate")

    assert bool(gates.loc["crypto_wizards_live_artifacts", "ready"]) is True
    assert bool(gates.loc["pair_detail_history", "ready"]) is True
    assert bool(gates.loc["pair_detail_two_leg_execution_history", "ready"]) is True
    assert bool(gates.loc["pair_detail_capture_audit", "ready"]) is True
    assert bool(gates.loc["strategy_acceptance", "ready"]) is True
    assert bool(gates.loc["dydx_testnet_readiness", "ready"]) is False
    assert "missing_dydx_order_client_adapter" in gates.loc["dydx_testnet_readiness", "blocker"]
    assert (reports / "strategy_acceptance_checklist.csv").exists()
    assert bool(gates.loc["learning_event_store", "ready"]) is False
    assert gates.loc["learning_event_store", "blocker"] == "missing_learning_events"
    assert (reports / "learning_event_summary.csv").exists()
    assert (reports / "dydx_execution_checklist.csv").exists()
    assert (tmp_path / "reports" / "priority_readiness.csv").exists()
    assert (tmp_path / "reports" / "priority_action_plan.csv").exists()
    assert (tmp_path / "reports" / "priority_spine_dashboard.csv").exists()


def test_priority_readiness_report_blocks_without_pair_history(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "build_dydx_indexer_adapter", lambda config: None)
    monkeypatch.delenv("DYDX_TESTNET_WALLET_ADDRESS", raising=False)
    monkeypatch.delenv("DYDX_TESTNET_PRIVATE_KEY", raising=False)
    monkeypatch.delenv("DYDX_TESTNET_SUBMIT_ORDERS", raising=False)
    (tmp_path / "data" / "raw" / "pair_details").mkdir(parents=True)
    (tmp_path / "reports").mkdir()

    frame = priority_readiness_report()
    gates = frame.set_index("gate")

    assert bool(gates.loc["crypto_wizards_live_artifacts", "ready"]) is False
    assert gates.loc["pair_detail_history", "blocker"] == "missing_spread_zscore_or_ecm_history"
    assert gates.loc["pair_detail_two_leg_execution_history", "blocker"] == "missing_price_x_or_price_y_history"
    assert gates.loc["pair_detail_capture_audit", "blocker"] == "no_nested_execution_ready_history_candidate_detected"
    assert gates.loc["strategy_acceptance", "blocker"] == "no_strategy_passes_production_gates"
    assert gates.loc["learning_event_store", "blocker"] == "missing_learning_events"


def test_dydx_execution_checklist_blocks_without_credentials_or_research(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "build_dydx_indexer_adapter", lambda config: None)
    monkeypatch.setattr("quant_platform.execution.dydx_v4_client_installed", lambda: False)
    monkeypatch.delenv("DYDX_TESTNET_WALLET_ADDRESS", raising=False)
    monkeypatch.delenv("DYDX_TESTNET_PRIVATE_KEY", raising=False)
    monkeypatch.delenv("DYDX_TESTNET_SUBMIT_ORDERS", raising=False)

    frame = dydx_execution_checklist_report()
    rows = frame.set_index("step")

    assert rows.loc["indexer_market_data", "blocker"] == "missing_dydx_indexer_adapter"
    assert rows.loc["testnet_credentials", "blocker"] == "missing_wallet_address;missing_private_key"
    assert rows.loc["dydx_sdk", "blocker"] == "missing_dydx_v4_client"
    assert rows.loc["submit_flag", "blocker"] == "submit_orders_false"
    assert rows.loc["order_client_adapter", "blocker"] == "missing_dydx_order_client_adapter"
    assert rows.loc["research_acceptance", "blocker"] == "no_research_accepted_two_leg_strategy"
    assert rows.loc["paper_submission_gate", "status"] == "blocked"


def test_strategy_acceptance_checklist_blocks_without_experiments(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)

    frame = strategy_acceptance_checklist_report()
    rows = frame.set_index("step")

    assert rows.loc["experiment_results", "blocker"] == "missing_evaluated_experiments"
    assert rows.loc["two_leg_coverage", "blocker"] == "missing_two_leg_backtests"
    assert rows.loc["production_eligibility", "blocker"] == "no_production_eligible_strategy"


def test_strategy_failure_attribution_report_summarizes_trade_and_feature_failures(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    reports = tmp_path / "reports"
    reports.mkdir()
    pd.DataFrame(
        [
            {
                "strategy_id": 1,
                "strategy_name": "Classic ZScore Mean Reversion",
                "family": "zscore",
                "pair": "ETH-BTC",
                "status": "evaluated",
                "eligible": False,
                "reason": "trades<100;profit_factor<1.8",
                "trades": 3,
                "profit_factor": 1.2,
                "sharpe": 0.5,
                "expectancy": 0.001,
                "max_drawdown": 0.03,
                "gross_return": 0.02,
                "total_return": 0.01,
            },
            {
                "strategy_id": 5,
                "strategy_name": "Pure Copula",
                "family": "copula",
                "pair": "ETH-BTC",
                "status": "skipped",
                "eligible": False,
                "reason": "missing_columns:conditional_probability_distortion",
                "trades": 0,
            },
        ]
    ).to_csv(reports / "experiment_results.csv", index=False)
    pd.DataFrame(
        [
            {
                "strategy_id": 1,
                "acceptance_reason": "passing_pairs<2",
                "preferred_reason": "not_production_eligible",
            },
            {
                "strategy_id": 5,
                "acceptance_reason": "no_evaluated_runs;passing_pairs<2",
                "preferred_reason": "not_production_eligible",
            },
        ]
    ).to_csv(reports / "acceptance_report.csv", index=False)

    frame = strategy_failure_attribution_report()
    rows = frame.set_index("strategy_id")

    assert rows.loc[1, "diagnosis"] == "too_few_trades_and_no_passing_pairs"
    assert rows.loc[1, "median_cost_drag"] == 0.01
    assert rows.loc[5, "diagnosis"] == "missing_required_feature_columns"
    assert rows.loc[5, "missing_columns"] == "conditional_probability_distortion"
    assert (reports / "strategy_failure_attribution.csv").exists()


def test_research_unblock_plan_prioritizes_history_and_missing_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    reports = tmp_path / "reports"
    reports.mkdir()
    pd.DataFrame(
        [
            {
                "strategy_id": 1,
                "strategy_name": "Classic ZScore Mean Reversion",
                "family": "zscore",
                "pair": "ETH-BTC",
                "status": "evaluated",
                "eligible": False,
                "reason": "trades<100",
                "trades": 5,
                "profit_factor": 1.1,
                "sharpe": 0.2,
                "expectancy": 0.001,
                "max_drawdown": 0.02,
                "observations": 1000,
                "gross_return": 0.02,
                "total_return": 0.01,
            },
            {
                "strategy_id": 5,
                "strategy_name": "Pure Copula",
                "family": "copula",
                "pair": "ETH-BTC",
                "status": "skipped",
                "eligible": False,
                "reason": "missing_columns:conditional_probability_distortion",
                "trades": 0,
            },
        ]
    ).to_csv(reports / "experiment_results.csv", index=False)
    pd.DataFrame(
        [
            {
                "strategy_id": 1,
                "acceptance_reason": "passing_pairs<2",
                "preferred_reason": "not_production_eligible",
                "production_eligible": False,
            },
            {
                "strategy_id": 5,
                "acceptance_reason": "no_evaluated_runs;passing_pairs<2",
                "preferred_reason": "not_production_eligible",
                "production_eligible": False,
            },
        ]
    ).to_csv(reports / "acceptance_report.csv", index=False)
    pd.DataFrame(
        [
            {
                "research_usable": True,
                "execution_usable": False,
                "quality_blockers": "missing_execution_assumptions:funding_x_bps;funding_y_bps",
            }
        ]
    ).to_csv(reports / "pair_detail_quality_report.csv", index=False)
    pd.DataFrame(
        [
            {
                "threshold": 1.0,
                "cost_bucket": "base",
                "max_trades": 17,
                "passing_pairs": 0,
                "diagnosis": "threshold_still_trade_sparse",
            }
        ]
    ).to_csv(reports / "zscore_threshold_sweep_summary.csv", index=False)

    frame = research_unblock_plan_report()
    rows = frame.set_index("area")

    assert rows.loc["trade_sample_size", "minimum_history_multiplier_estimate"] == "20x_current_history"
    assert "max_trades_per_run=5" in rows.loc["trade_sample_size", "evidence"]
    assert rows.loc["threshold_sensitivity", "blocker"] == "threshold_sweep_has_no_passing_pairs"
    assert "max_trades=17" in rows.loc["threshold_sensitivity", "evidence"]
    assert "conditional_probability_distortion" in set(
        frame[frame["area"] == "missing_feature_coverage"]["blocker"].str.replace("missing_", "")
    )
    assert rows.loc["paper_trading_gate", "blocker"] == "research_rejected_all_strategies"
    assert (reports / "research_unblock_plan.csv").exists()


def test_dydx_pair_expansion_plan_ranks_untested_pairs(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    reports = tmp_path / "reports"
    reports.mkdir()
    pd.DataFrame(
        [
            {
                "pair": "ETH-USD-BTC-USD",
                "market_x": "ETH-USD",
                "market_y": "BTC-USD",
                "ready": True,
            }
        ]
    ).to_csv(reports / "funding_coverage.csv", index=False)
    pd.DataFrame(
        [
            {
                "priority": 1,
                "area": "trade_sample_size",
                "evidence": "max_trades_per_run=5",
                "minimum_history_multiplier_estimate": "20x_current_history",
                "preferred_history_multiplier_estimate": "50x_current_history",
            }
        ]
    ).to_csv(reports / "research_unblock_plan.csv", index=False)

    frame = dydx_pair_expansion_plan_report(max_pairs=10, limit=1000)

    rows = frame.set_index("pair_id")
    assert bool(rows.loc["btc_eth", "already_tested"]) is True
    assert rows.loc["btc_eth", "rank"] == ""
    assert rows.loc["btc_sol", "rank"] == 1
    assert rows.loc["eth_sol", "rank"] == 2
    assert "fetch-dydx-two-leg-data" in rows.loc["btc_sol", "fetch_command"]
    assert "--derive-hedge-ratio --run-research" in rows.loc["btc_sol", "fetch_command"]
    assert "SOL-USD" in rows.loc["btc_sol", "missing_funding_markets"]
    assert (reports / "dydx_pair_expansion_plan.csv").exists()


def test_dydx_live_market_selector_ranks_unfetched_active_markets(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    reports = tmp_path / "reports"
    reports.mkdir()
    pd.DataFrame(
        [
            {
                "priority": 1,
                "area": "trade_sample_size",
                "evidence": "max_trades_per_run=12",
                "minimum_history_multiplier_estimate": "8x_current_history",
                "preferred_history_multiplier_estimate": "20x_current_history",
            }
        ]
    ).to_csv(reports / "research_unblock_plan.csv", index=False)

    monkeypatch.setattr(cli, "_tested_market_pairs", lambda: {frozenset({"ETH-USD", "TAO-USD"})})
    monkeypatch.setattr(cli, "_fetched_market_pair_info", lambda: {})
    monkeypatch.setattr(cli, "_stale_market_risk_info", lambda: {"DOT-USD": "DOT-USD:stale_price_x"})
    monkeypatch.setattr(cli, "_local_cached_dydx_markets", lambda: {"BTC-USD", "ETH-USD", "SOL-USD", "ETC-USD"})

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "markets": {
                    "TAO-USD": {"status": "ACTIVE", "marketType": "CROSS", "trades24H": 500, "volume24H": "100000", "oraclePrice": "250"},
                    "NEAR-USD": {"status": "ACTIVE", "marketType": "CROSS", "trades24H": 700, "volume24H": "20000", "oraclePrice": "7"},
                    "DOT-USD": {"status": "ACTIVE", "marketType": "CROSS", "trades24H": 650, "volume24H": "15000", "oraclePrice": "6"},
                    "PAXG-USD": {"status": "ACTIVE", "marketType": "CROSS", "trades24H": 900, "volume24H": "50000", "oraclePrice": "2300"},
                    "BAD,MARKET-USD": {"status": "ACTIVE", "marketType": "CROSS", "trades24H": 900, "volume24H": "50000", "oraclePrice": "1"},
                    "INACTIVE-USD": {"status": "OFFLINE", "marketType": "CROSS", "trades24H": 900, "volume24H": "50000", "oraclePrice": "1"},
                }
            }

    monkeypatch.setattr(cli.requests, "get", lambda *args, **kwargs: FakeResponse())

    frame = dydx_live_market_selector_report(max_pairs=4)

    assert not frame.empty
    assert "NEAR-USD" in set(frame["asset_y"])
    assert "DOT-USD" not in set(frame["asset_y"])
    assert "PAXG-USD" not in set(frame["asset_y"])
    assert not ((frame["asset_x"] == "ETH-USD") & (frame["asset_y"] == "TAO-USD")).any()
    assert frame.iloc[0]["selector_score"] >= frame.iloc[-1]["selector_score"]
    assert "fetch-dydx-two-leg-data" in frame.iloc[0]["fetch_command"]
    assert (reports / "dydx_live_market_selector.csv").exists()


def test_dydx_live_market_counts_report_summarizes_live_and_local_coverage(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(
        cli,
        "_fetch_live_dydx_market_catalog",
        lambda indexer_base="": {
            "BTC-USD": {"status": "ACTIVE", "marketType": "CROSS", "trades24H": 10, "volume24H": 1000, "oraclePrice": 100000},
            "ETH-USD": {"status": "ACTIVE", "marketType": "CROSS", "trades24H": 10, "volume24H": 1000, "oraclePrice": 2000},
            "SOL-USD": {"status": "ACTIVE", "marketType": "CROSS", "trades24H": 10, "volume24H": 1000, "oraclePrice": 100},
            "TAO-USD": {"status": "ACTIVE", "marketType": "CROSS", "trades24H": 10, "volume24H": 1000, "oraclePrice": 400},
            "WLD-USD": {"status": "ACTIVE", "marketType": "CROSS", "trades24H": 10, "volume24H": 1000, "oraclePrice": 2},
            "PAXG-USD": {"status": "ACTIVE", "marketType": "CROSS", "trades24H": 10, "volume24H": 1000, "oraclePrice": 2000},
            "BAD-USD": {"status": "PAUSED", "marketType": "CROSS", "trades24H": 10, "volume24H": 1000, "oraclePrice": 1},
        },
    )
    monkeypatch.setattr(cli, "_local_cached_dydx_markets", lambda: {"WLD-USD"})
    monkeypatch.setattr(cli, "_stale_market_risk_info", lambda: {"TAO-USD": "TAO-USD:stale_price_x"})
    monkeypatch.setattr(cli, "_tested_market_pairs", lambda: {frozenset({"BTC-USD", "SOL-USD"})})
    monkeypatch.setattr(cli, "_fetched_market_pair_info", lambda: {frozenset({"ETH-USD", "SOL-USD"}): {"pair_id": "eth_sol"}})

    frame = cli.dydx_live_market_counts_report()
    row = frame.iloc[0]

    assert row["total_markets"] == 7
    assert row["active_markets"] == 6
    assert row["active_cross_markets"] == 6
    assert row["excluded_markets"] == 1
    assert row["cached_markets"] == 1
    assert row["risky_markets"] == 1
    assert row["untested_candidate_markets"] == 3
    assert row["untested_candidate_pairs"] == 2
    assert (tmp_path / "reports" / "dydx_live_market_counts.csv").exists()


def test_zscore_threshold_sweep_writes_detail_and_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    reports = tmp_path / "reports"
    reports.mkdir()
    frame = pd.DataFrame(
        {
            "price_x": [100 + idx + (6 if idx in {12, 24, 36, 48} else 0) for idx in range(60)],
            "price_y": [50 + idx * 0.4 for idx in range(60)],
            "spread": [0.0 for _ in range(60)],
            "zscore": [2.2 if idx in {12, 24, 36, 48} else (-2.2 if idx in {18, 30, 42, 54} else 0.0) for idx in range(60)],
            "hedge_ratio": [1.0 for _ in range(60)],
            "beta": [1.0 for _ in range(60)],
            "funding_x_bps": [0.01 for _ in range(60)],
            "funding_y_bps": [0.01 for _ in range(60)],
        }
    )
    monkeypatch.setattr(
        cli,
        "datasets_from_pair_detail_snapshots",
        lambda input_dir, require_research_usable=False: [cli.PairDataset("AAA-USD-BBB-USD", frame)],
    )

    result = zscore_threshold_sweep_report(thresholds=(1.0, 2.0))

    assert set(result["threshold"]) == {1.0, 2.0}
    assert set(result["cost_bucket"]) == {"base", "stress"}
    assert (reports / "zscore_threshold_sweep.csv").exists()
    summary = pd.read_csv(reports / "zscore_threshold_sweep_summary.csv")
    assert {"threshold", "median_trades", "diagnosis"}.issubset(summary.columns)


def test_dydx_long_history_plan_writes_windowed_candle_requests(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)

    frame = dydx_long_history_plan_report(
        asset_x="SOL-USD",
        asset_y="LINK-USD",
        pair_id="sol_link",
        windows=2,
        limit=1000,
        to_iso="2026-06-18T12:00:00Z",
    )

    candle_rows = frame[frame["method"] == "GET"]
    assert len(candle_rows) == 4
    assert set(candle_rows["window"]) == {1, 2}
    assert "fromISO=2026-06-15T00%3A40%3A00Z" in candle_rows.iloc[0]["url"]
    assert "toISO=2026-06-18T12%3A00%3A00Z" in candle_rows.iloc[0]["url"]
    assert "SOL-USD_5MINS_candles.json" in candle_rows.iloc[0]["save_as"]
    assert "window_001" in candle_rows.iloc[0]["save_as"]
    assert frame.iloc[-1]["request_name"] == "long_history_next_step"
    assert (tmp_path / "reports" / "dydx_long_history_plan.csv").exists()


def test_build_dydx_long_history_pair_builds_from_windowed_files(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    source = tmp_path / "data" / "raw" / "dydx_long_history" / "sol_link"
    for window, minutes in (("window_002", [0, 5, 10]), ("window_001", [10, 15, 20])):
        window_dir = source / window
        window_dir.mkdir(parents=True)
        for market, base in (("SOL-USD", 100), ("LINK-USD", 20)):
            candles = [
                {
                    "startedAt": f"2026-06-18T00:{minute:02d}:00.000Z",
                    "ticker": market,
                    "resolution": "5MINS",
                    "close": str(base + minute / 5),
                    "usdVolume": "1000",
                }
                for minute in minutes
            ]
            (window_dir / f"{market}_5MINS_candles.json").write_text(json.dumps({"candles": candles}), encoding="utf-8")

    paths = build_dydx_long_history_pair(
        input_dir=None,
        asset_x="SOL-USD",
        asset_y="LINK-USD",
        pair_id="sol_link",
        hedge_ratio=1.0,
        beta=1.0,
        interval="5mins",
        zscore_window=3,
        derive_hedge_ratio=True,
    )

    assert paths["left_candles"] == tmp_path / "data" / "raw" / "dydx_candles" / "SOL-USD_5MINS_candles.json"
    pair = json.loads(paths["pair_history"].read_text(encoding="utf-8"))
    assert len(pair["history"]) == 5
    assert pair["pair"] == "SOL-USD-LINK-USD"
    assert pair["hedge_ratio_source"] == "derived_price_ols"


def test_build_dydx_long_history_pair_can_rerun_research(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    source = tmp_path / "data" / "raw" / "dydx_long_history" / "sol_link"
    for window, minutes in (("window_001", [0, 5, 10]),):
        window_dir = source / window
        window_dir.mkdir(parents=True)
        for market, base in (("SOL-USD", 100), ("LINK-USD", 20)):
            candles = [
                {
                    "startedAt": f"2026-06-18T00:{minute:02d}:00.000Z",
                    "ticker": market,
                    "resolution": "5MINS",
                    "close": str(base + minute / 5),
                    "usdVolume": "1000",
                }
                for minute in minutes
            ]
            (window_dir / f"{market}_5MINS_candles.json").write_text(json.dumps({"candles": candles}), encoding="utf-8")
    funding = tmp_path / "data" / "processed" / "dydx_funding.csv"
    funding.parent.mkdir(parents=True)
    funding.write_text("market,timestamp,funding_bps\nSOL-USD,2026-06-18T00:00:00Z,0.1\nLINK-USD,2026-06-18T00:00:00Z,0.1\n", encoding="utf-8")

    calls = {}

    def fake_rerun(*, input_dir, funding_path):
        calls["input_dir"] = input_dir
        calls["funding_path"] = funding_path
        return {"priority_readiness": tmp_path / "reports" / "priority_readiness.csv"}

    monkeypatch.setattr(cli, "rerun_p2_acceptance_evidence", fake_rerun)

    paths = build_dydx_long_history_pair(
        input_dir=None,
        asset_x="SOL-USD",
        asset_y="LINK-USD",
        pair_id="sol_link",
        hedge_ratio=1.0,
        beta=1.0,
        interval="5mins",
        zscore_window=3,
        derive_hedge_ratio=True,
        run_research=True,
        funding_path=funding,
    )

    assert "input_dir" in calls
    assert calls["funding_path"] == funding
    assert paths["pair_history"].exists()


def test_rerun_p2_acceptance_evidence_refreshes_acceptance_reports(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    calls = []

    def fake_run_pair_detail_experiments(input_dir=None, funding_path=None, require_research_usable=False):
        calls.append(("run_pair_detail_experiments", input_dir, funding_path, require_research_usable))
        reports = tmp_path / "reports"
        reports.mkdir(exist_ok=True)
        pd.DataFrame(
            [
                {
                    "strategy_id": 1,
                    "strategy_name": "Test Strategy",
                    "family": "mean_reversion",
                    "pair": "SOL-USD-LINK-USD",
                    "status": "evaluated",
                    "backtest_mode": "two_leg",
                    "cost_bucket": "base",
                    "eligible": False,
                    "trades": 10,
                    "observations": 100,
                    "profit_factor": 0.9,
                    "sharpe": 0.1,
                    "expectancy": -0.01,
                    "max_drawdown": 0.2,
                    "reason": "passing_pairs<2",
                }
            ]
        ).to_csv(reports / "experiment_results.csv", index=False)
        pd.DataFrame(
            [
                {
                    "strategy_id": 1,
                    "production_eligible": False,
                    "preferred_eligible": False,
                    "two_leg_pairs_tested": 1,
                    "two_leg_execution_input_pairs": 1,
                    "two_leg_passing_pairs": 0,
                    "total_trades": 10,
                    "required_cost_buckets": "base;stress",
                    "required_two_leg_inputs": "price_x;price_y;hedge_ratio;beta;funding_x;funding_y",
                    "acceptance_reason": "passing_pairs<2",
                    "preferred_reason": "not_production_eligible",
                }
            ]
        ).to_csv(reports / "acceptance_report.csv", index=False)

    monkeypatch.setattr(cli, "run_pair_detail_experiments", fake_run_pair_detail_experiments)

    paths = rerun_p2_acceptance_evidence(
        input_dir=tmp_path / "data" / "raw" / "pair_details",
        funding_path=tmp_path / "data" / "processed" / "dydx_funding.csv",
    )

    assert calls == [("run_pair_detail_experiments", tmp_path / "data" / "raw" / "pair_details", tmp_path / "data" / "processed" / "dydx_funding.csv", False)]
    assert paths["strategy_acceptance_checklist"].exists()
    assert paths["priority_readiness"].exists()
    assert paths["priority_gap_test"].exists()


def test_fetch_dydx_long_history_windows_fetches_candle_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    reports = tmp_path / "reports"
    reports.mkdir(parents=True)
    plan = pd.DataFrame(
        [
            {
                "window": 1,
                "pair_id": "sol_link",
                "asset_x": "SOL-USD",
                "asset_y": "LINK-USD",
                "request_name": "asset_x_candles_5mins",
                "method": "GET",
                "url": "https://example.test/sol",
                "save_as": str(tmp_path / "data" / "raw" / "dydx_long_history" / "sol_link" / "window_001" / "SOL-USD_5MINS_candles.json"),
            },
            {
                "window": 1,
                "pair_id": "sol_link",
                "asset_x": "SOL-USD",
                "asset_y": "LINK-USD",
                "request_name": "asset_y_candles_5mins",
                "method": "GET",
                "url": "https://example.test/link",
                "save_as": str(tmp_path / "data" / "raw" / "dydx_long_history" / "sol_link" / "window_001" / "LINK-USD_5MINS_candles.json"),
            },
        ]
    )
    plan.to_csv(reports / "dydx_long_history_plan.csv", index=False)
    fetched = []

    def fake_fetch(url, output_path, timeout=30.0, **kwargs):
        fetched.append((url, output_path))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text('{"candles": []}', encoding="utf-8")
        return output_path

    monkeypatch.setattr(cli, "_fetch_public_json", fake_fetch)

    frame = fetch_dydx_long_history_windows(plan_path=reports / "dydx_long_history_plan.csv", max_windows=1)

    assert len(fetched) == 2
    assert set(frame["status"]) == {"fetched"}
    assert (reports / "dydx_long_history_fetch.csv").exists()


def test_fetch_dydx_long_history_windows_skips_existing_files(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    reports = tmp_path / "reports"
    reports.mkdir(parents=True)
    save_as = tmp_path / "data" / "raw" / "dydx_long_history" / "sol_link" / "window_001" / "SOL-USD_5MINS_candles.json"
    save_as.parent.mkdir(parents=True, exist_ok=True)
    save_as.write_text('{"candles": [{"startedAt": "2026-06-18T00:00:00Z"}]}', encoding="utf-8")
    plan = pd.DataFrame(
        [
            {
                "window": 1,
                "pair_id": "sol_link",
                "asset_x": "SOL-USD",
                "asset_y": "LINK-USD",
                "request_name": "asset_x_candles_5mins",
                "method": "GET",
                "url": "https://example.test/sol",
                "save_as": str(save_as),
            }
        ]
    )
    plan.to_csv(reports / "dydx_long_history_plan.csv", index=False)
    called = []
    monkeypatch.setattr(cli, "_fetch_public_json", lambda *args, **kwargs: called.append(args))

    frame = fetch_dydx_long_history_windows(plan_path=reports / "dydx_long_history_plan.csv", max_windows=1)

    assert called == []
    assert list(frame["status"]) == ["existing"]


def test_fetch_dydx_long_history_windows_passes_indexer_scheme(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    reports = tmp_path / "reports"
    reports.mkdir(parents=True)
    plan = pd.DataFrame(
        [
            {
                "window": 1,
                "pair_id": "sol_link",
                "asset_x": "SOL-USD",
                "asset_y": "LINK-USD",
                "request_name": "asset_x_candles_5mins",
                "method": "GET",
                "url": "https://example.test/sol",
                "save_as": str(tmp_path / "data" / "raw" / "dydx_long_history" / "sol_link" / "window_001" / "SOL-USD_5MINS_candles.json"),
            }
        ]
    )
    plan.to_csv(reports / "dydx_long_history_plan.csv", index=False)
    calls = []

    def fake_fetch(url, output_path, **kwargs):
        calls.append({"url": url, "fetch_scheme": kwargs.get("fetch_scheme")})
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text('{"candles": []}', encoding="utf-8")
        return output_path

    monkeypatch.setattr(cli, "_fetch_public_json", fake_fetch)

    frame = fetch_dydx_long_history_windows(
        plan_path=reports / "dydx_long_history_plan.csv",
        max_windows=1,
        indexer_scheme="http",
    )

    assert list(frame["status"]) == ["fetched"]
    assert calls == [{"url": "https://example.test/sol", "fetch_scheme": "http"}]


def test_fetch_dydx_long_history_windows_rejects_wrong_target(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    reports = tmp_path / "reports"
    reports.mkdir(parents=True)
    plan = pd.DataFrame(
        [
            {
                "window": 1,
                "pair_id": "sol_link",
                "asset_x": "SOL-USD",
                "asset_y": "LINK-USD",
                "request_name": "asset_x_candles_5mins",
                "method": "GET",
                "url": "https://example.test/sol",
                "save_as": str(tmp_path / "data" / "raw" / "dydx_long_history" / "sol_link" / "window_001" / "SOL-USD_5MINS_candles.json"),
            }
        ]
    )
    plan.to_csv(reports / "dydx_long_history_plan.csv", index=False)

    with pytest.raises(SystemExit):
        fetch_dydx_long_history_windows(
            plan_path=reports / "dydx_long_history_plan.csv",
            required_pair_id="eth_ena",
        )


def test_dydx_long_history_coverage_report_marks_missing_and_ready_windows(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    frame = dydx_long_history_plan_report(
        asset_x="SOL-USD",
        asset_y="LINK-USD",
        pair_id="sol_link",
        windows=2,
        limit=3,
        output_path=tmp_path / "reports" / "dydx_long_history_plan.csv",
    )
    for _, row in frame[frame["method"] == "GET"].iterrows():
        path = Path(row["save_as"])
        if "window_001" in str(path):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}", encoding="utf-8")
        elif "window_002" in str(path) and "SOL-USD" in str(path):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}", encoding="utf-8")

    coverage = dydx_long_history_coverage_report(
        asset_x="SOL-USD",
        asset_y="LINK-USD",
        pair_id="sol_link",
        windows=2,
        limit=3,
    )
    rows = coverage.set_index("window")

    assert bool(rows.loc[1, "ready"]) is True
    assert int(rows.loc[1, "missing_files"]) == 0
    assert bool(rows.loc[2, "ready"]) is False
    assert int(rows.loc[2, "missing_files"]) == 1
    assert "LINK-USD_5MINS_candles.json" in rows.loc[2, "missing_paths"]


def test_fetch_public_json_falls_back_to_curl_when_requests_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(
        cli.requests,
        "get",
        lambda *args, **kwargs: (_ for _ in ()).throw(cli.requests.exceptions.RequestException("dns fail")),
    )
    written = {}

    def fake_run(cmd, check):
        output_index = cmd.index("--output") + 1
        output_path = Path(cmd[output_index])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text('{"candles": [{"startedAt": "2026-06-18T00:00:00Z"}]}', encoding="utf-8")
        written["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    output = cli._fetch_public_json("https://example.test/candles", tmp_path / "payload.json")

    assert output.exists()
    assert "curl" in written["cmd"][0]


def test_fetch_public_json_falls_back_to_http_scheme(tmp_path, monkeypatch):
    calls = []

    class DummyResponse:
        def __init__(self, payload: str = "{}"):
            self.text = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            raise ValueError("not json")

    def fake_get(url, headers=None, timeout=None, params=None, **kwargs):
        calls.append(url)
        if url.startswith("https://"):
            raise cli.requests.exceptions.RequestException("tls failure")
        return DummyResponse('{"candles": []}')

    monkeypatch.setattr(cli.requests, "get", fake_get)
    output = cli._fetch_public_json("https://example.test/candles", tmp_path / "payload.json", max_retries=1)

    assert output.exists()
    assert calls == ["https://example.test/candles", "http://example.test/candles"]
    assert json.loads(output.read_text(encoding="utf-8")) == {"candles": []}


def test_fetch_public_json_respects_disable_scheme_fallback(tmp_path, monkeypatch):
    calls = []

    class DummyResponse:
        def __init__(self, payload: str = "{}"):
            self.text = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            raise ValueError("not json")

    def fake_get(url, headers=None, timeout=None, params=None, **kwargs):
        calls.append(url)
        if url != "https://example.test/candles":
            raise cli.requests.exceptions.RequestException("scheme fallback not expected")
        return DummyResponse('{"candles": []}')

    monkeypatch.setattr(cli.requests, "get", fake_get)
    monkeypatch.setenv("QPA_DISABLE_SCHEME_FALLBACK", "1")
    output = cli._fetch_public_json("https://example.test/candles", tmp_path / "payload.json", max_retries=1)

    assert output.exists()
    assert calls == ["https://example.test/candles"]
    assert json.loads(output.read_text(encoding="utf-8")) == {"candles": []}


def test_fetch_public_json_respects_forced_indexer_scheme(tmp_path, monkeypatch):
    calls = []

    class DummyResponse:
        def __init__(self, payload: str = "{}"):
            self.text = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"candles": []}

    def fake_get(url, headers=None, timeout=None, params=None, **kwargs):
        calls.append(url)
        if url == "http://example.test/candles":
            return DummyResponse('{"candles": []}')
        raise cli.requests.exceptions.RequestException("unexpected url scheme")

    monkeypatch.setattr(cli.requests, "get", fake_get)
    monkeypatch.setenv("QPA_INDEXER_SCHEME", "http")
    output = cli._fetch_public_json("https://example.test/candles", tmp_path / "payload.json", max_retries=1)

    assert output.exists()
    assert calls == ["http://example.test/candles"]


def test_fetch_public_json_uses_ip_alias_when_dns_hints_exist(tmp_path, monkeypatch):
    requests = cli.requests
    calls: list[tuple[str, str | None, bool]] = []

    def fake_get(url, headers=None, timeout=None, verify=True, **kwargs):
        headers_map = headers or {}
        calls.append((url, headers_map.get("Host"), verify))
        if "dydx-indexer-alias" in url:
            raise requests.exceptions.RequestException("alias not expected")
        if "198.51.100.77" in url:
            if verify:
                raise requests.exceptions.RequestException("ssl mismatch")

            class Response:
                def __init__(self, payload):
                    self.payload = payload
                    self.text = json.dumps(payload)

                def raise_for_status(self):
                    return None

                def json(self):
                    return self.payload

            return Response({"candles": [{"startedAt": "2026-06-18T00:00:00.000Z", "ticker": "SOL-USD"}]})
        raise requests.exceptions.RequestException(f"unexpected {url}")

    monkeypatch.setattr(cli, "_dns_fallback_ip_candidates", lambda host: ["198.51.100.77"])
    monkeypatch.setattr(cli.requests, "get", fake_get)

    output = cli._fetch_public_json(
        "https://indexer.dydx.trade/v4/candles/perpetualMarkets/SOL-USD",
        tmp_path / "payload.json",
        max_retries=1,
    )

    assert output.exists()
    assert any("198.51.100.77" in url for url, _, _ in calls)
    assert any(host == "indexer.dydx.trade" and not verify for _, host, verify in calls)
    assert any(verify is False for _, _, verify in calls)


def test_fetch_public_json_curl_uses_resolve_without_ip_in_url(tmp_path, monkeypatch):
    monkeypatch.setattr(cli.requests, "get", lambda *args, **kwargs: (_ for _ in ()).throw(cli.requests.exceptions.RequestException("dns fail")))
    recorded: list[list[str]] = []

    output_path = tmp_path / "payload.json"

    def fake_run(cmd, check):
        recorded.append(cmd)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text('{"candles": []}', encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    monkeypatch.setattr(cli, "_dns_fallback_ip_candidates", lambda host: ["198.51.100.77"])
    output = cli._fetch_public_json(
        "https://indexer.dydx.trade/v4/candles/perpetualMarkets/SOL-USD?resolution=5MINS&limit=1",
        output_path,
        max_retries=1,
    )

    assert output.exists()
    assert len(recorded) == 1
    command = recorded[0]
    assert any(part == "--resolve" for part in command)
    resolve_index = command.index("--resolve") + 1
    assert command[resolve_index] == "indexer.dydx.trade:443:198.51.100.77"
    assert "https://indexer.dydx.trade/v4/candles/perpetualMarkets/SOL-USD?resolution=5MINS&limit=1" in command
    assert "198.51.100.77/v4/candles/perpetualMarkets/SOL-USD" not in command
    assert json.loads(output.read_text(encoding="utf-8")) == {"candles": []}


def test_fetch_dydx_two_leg_data_falls_back_to_configured_indexer_base(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload
            self.text = json.dumps(payload)

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    def fake_get(url, headers=None, timeout=None, verify=True, **kwargs):
        if "indexer.dydx.trade" in url:
            raise cli.requests.exceptions.RequestException("dns fail")
        if "indexer.v4testnet.dydx.exchange" in url:
            if "candles" in url:
                return FakeResponse(
                    {
                        "candles": [
                            {
                                "startedAt": "2026-06-18T00:00:00.000Z",
                                "ticker": "ABC-USD",
                                "resolution": "5MINS",
                                "close": "1",
                                "usdVolume": "1000",
                            }
                        ]
                    }
                )
        if "historicalFunding" in url:
            return FakeResponse({"historicalFunding": [{"effectiveAt": "2026-06-18T00:00:00Z", "rate": "0.0001"}]})
        raise AssertionError(f"unexpected url: {url}")

    calls: list[str] = []
    original_fetch_public_json = cli._fetch_public_json

    def fake_fetch_public_json(url, output_path, timeout=30.0, max_retries=3, **kwargs):
        calls.append(url)
        return original_fetch_public_json(url, output_path, timeout=timeout, max_retries=max_retries)

    monkeypatch.setattr(cli.requests, "get", fake_get)
    monkeypatch.setattr(cli, "_fetch_public_json", fake_fetch_public_json)
    monkeypatch.setenv("QPA_INDEXER_BASES", "https://indexer.dydx.trade,https://indexer.v4testnet.dydx.exchange")

    paths = cli.fetch_dydx_two_leg_data(
        pair="ABC-USD-DEF-USD",
        pair_id="abc_def",
        output_dir=tmp_path / "manual",
        limit=10,
        run_research=False,
    )

    assert paths["pair_history"].exists()
    assert any("indexer.v4testnet.dydx.exchange" in url for url in calls)
    assert any("indexer.dydx.trade" in url for url in calls)


def test_run_dydx_long_history_orchestrates_plan_fetch_and_build(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    calls = {}

    def fake_plan(**kwargs):
        calls["plan"] = kwargs
        reports = tmp_path / "reports"
        reports.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "window": 1,
                    "pair_id": "sol_link",
                    "asset_x": "SOL-USD",
                    "asset_y": "LINK-USD",
                    "request_name": "asset_x_candles_5mins",
                    "method": "GET",
                    "url": "https://example.test/sol",
                    "save_as": str(tmp_path / "data" / "raw" / "dydx_long_history" / "sol_link" / "window_001" / "SOL-USD_5MINS_candles.json"),
                },
                {
                    "window": 1,
                    "pair_id": "sol_link",
                    "asset_x": "SOL-USD",
                    "asset_y": "LINK-USD",
                    "request_name": "asset_y_candles_5mins",
                    "method": "GET",
                    "url": "https://example.test/link",
                    "save_as": str(tmp_path / "data" / "raw" / "dydx_long_history" / "sol_link" / "window_001" / "LINK-USD_5MINS_candles.json"),
                },
            ]
        ).to_csv(reports / "dydx_long_history_plan.csv", index=False)
        return pd.DataFrame([{"pair_id": "sol_link", "asset_x": "SOL-USD", "asset_y": "LINK-USD"}])

    def fake_fetch(*, plan_path=None, max_windows=None, **kwargs):
        calls["fetch"] = {"plan_path": plan_path, "max_windows": max_windows}
        return pd.DataFrame([{"status": "fetched"}])

    def fake_build(**kwargs):
        calls["build"] = kwargs
        return {"pair_history": tmp_path / "pair.json", "left_candles": tmp_path / "left.json", "right_candles": tmp_path / "right.json"}

    monkeypatch.setattr(cli, "dydx_long_history_plan_report", fake_plan)
    monkeypatch.setattr(cli, "fetch_dydx_long_history_windows", fake_fetch)
    monkeypatch.setattr(cli, "build_dydx_long_history_pair", fake_build)

    paths = run_dydx_long_history(
        asset_x="SOL-USD",
        asset_y="LINK-USD",
        pair_id="sol_link",
        windows=1,
        limit=1000,
        run_research=True,
        funding_path=tmp_path / "data" / "processed" / "dydx_funding.csv",
    )

    assert "plan" in calls
    assert calls["fetch"]["max_windows"] == 1
    assert calls["build"]["pair_id"] == "sol_link"
    assert paths["plan"].name == "dydx_long_history_plan.csv"
    assert paths["fetch"].name == "dydx_long_history_fetch.csv"


def test_dydx_pair_expansion_plan_marks_quality_blocked_fetched_pair(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    reports = tmp_path / "reports"
    reports.mkdir()
    pd.DataFrame(
        [
            {
                "pair": "ETH-USD-AVAX-USD",
                "research_usable": "False",
                "quality_blockers": "price_y_stale_above_90pct",
            }
        ]
    ).to_csv(reports / "pair_detail_quality_report.csv", index=False)

    frame = dydx_pair_expansion_plan_report(max_pairs=10, limit=1000)
    rows = frame.set_index("pair_id")

    assert bool(rows.loc["eth_avax", "already_fetched"]) is True
    assert rows.loc["eth_avax", "quality_status"] == "quality_blocked"
    assert rows.loc["eth_avax", "quality_blockers"] == "price_y_stale_above_90pct"
    assert rows.loc["eth_avax", "market_risk_status"] == "stale_market_risk"
    assert "AVAX-USD:stale_price_y" in rows.loc["eth_avax", "market_risk_reasons"]
    assert rows.loc["eth_avax", "rank"] == ""

    tested_pairs = cli._tested_market_pairs()
    fetched_pairs = cli._fetched_market_pair_info()
    risky_markets = cli._stale_market_risk_info()
    fresh_candidates = []
    for order, (asset_x, asset_y) in enumerate(cli.DEFAULT_DYDX_EXPANSION_PAIRS):
        left = cli._normalize_dydx_market(asset_x)
        right = cli._normalize_dydx_market(asset_y)
        pair_key = frozenset({left, right})
        if pair_key in tested_pairs or pair_key in fetched_pairs:
            continue
        risk_score = 1 if (left in risky_markets or right in risky_markets) else 0
        fresh_candidates.append((risk_score, order, cli._pair_id_from_markets(left, right)))
    expected_ranks = {
        pair_id: rank for rank, (_, __, pair_id) in enumerate(sorted(fresh_candidates, key=lambda row: (row[0], row[1])), start=1)
    }

    assert rows.loc["eth_link", "rank"] == expected_ranks["eth_link"]
    for pair_id in ("sol_avax", "sol_link"):
        if expected_ranks[pair_id] <= 10:
            assert rows.loc[pair_id, "rank"] == expected_ranks[pair_id]
        else:
            assert pair_id not in rows.index


def test_run_dydx_pair_expansion_runs_first_fresh_pair(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    reports = tmp_path / "reports"
    reports.mkdir()
    pd.DataFrame(
        [
            {
                "pair": "ETH-USD-BTC-USD",
                "market_x": "ETH-USD",
                "market_y": "BTC-USD",
                "ready": True,
            }
        ]
    ).to_csv(reports / "funding_coverage.csv", index=False)

    calls = []

    def fake_fetch(**kwargs):
        calls.append(kwargs)
        pair_id = kwargs["pair_id"]
        pair_history = tmp_path / f"{pair_id}.json"
        pair_history.write_text("{}", encoding="utf-8")
        return {
            "pair_history": pair_history,
            "funding_csv": tmp_path / "funding.csv",
            "funding_coverage": reports / "funding_coverage.csv",
        }

    monkeypatch.setattr(cli, "fetch_dydx_two_leg_data", fake_fetch)

    frame = run_dydx_pair_expansion(max_pairs=1, limit=1000, run_research=False)

    assert len(calls) == 1
    assert calls[0]["asset_x"] == "BTC-USD"
    assert calls[0]["asset_y"] in {"ETH-USD", "SOL-USD"}
    assert calls[0]["derive_hedge_ratio"] is True
    row = frame.iloc[0]
    assert row["status"] == "completed"
    assert row["pair_id"] in {"btc_eth", "btc_sol"}
    assert row["pair_history"].endswith(".json")
    assert (reports / "dydx_pair_expansion_run.csv").exists()


def test_run_dydx_pair_expansion_supports_skip_fetch(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    reports = tmp_path / "reports"
    reports.mkdir()

    calls: list[dict[str, object]] = []

    def fake_fetch(**kwargs):
        calls.append(kwargs)
        pair_id = kwargs["pair_id"]
        pair_path = tmp_path / f"{pair_id}_pair.json"
        pair_path.write_text("{}", encoding="utf-8")
        return {
            "pair_history": pair_path,
            "funding_csv": tmp_path / "funding.csv",
            "funding_coverage": reports / "funding_coverage.csv",
        }

    # Provide local files so skip_fetch path succeeds and does not require live HTTP.
    (tmp_path / "funding_coverage.csv").write_text("market,ready\nBTC-USD,true\nETH-USD,true\n", encoding="utf-8")
    monkeypatch.setattr(cli, "fetch_dydx_two_leg_data", fake_fetch)
    (tmp_path / "funding.csv").write_text("market,rate\nBTC-USD,0\n", encoding="utf-8")

    frame = run_dydx_pair_expansion(
        max_pairs=1,
        limit=1000,
        run_research=False,
        skip_fetch=True,
        allow_stale_fetch=True,
    )

    assert len(calls) == 1
    assert calls[0]["skip_fetch"] is True
    assert calls[0]["allow_stale_fetch"] is True
    row = frame.iloc[0]
    assert row["status"] == "completed"
    assert row["pair_id"] in {"btc_eth", "btc_sol"}
def test_run_dydx_local_pair_universe_builds_pair_histories_from_manual_candles(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    manual = tmp_path / "data" / "raw" / "dydx_manual"
    manual.mkdir(parents=True)
    for market, base in (("BTC-USD", 100.0), ("ETH-USD", 50.0)):
        candles = [
            {
                "startedAt": f"2026-06-18T00:{minute:02d}:00.000Z",
                "ticker": market,
                "resolution": "5MINS",
                "close": str(base + minute / 10.0),
                "usdVolume": "1000",
            }
            for minute in range(0, 30, 5)
        ]
        (manual / f"{market}_5MINS_candles.json").write_text(json.dumps({"candles": candles}), encoding="utf-8")
        funding_rows = [
            {
                "effectiveAt": f"2026-06-18T00:{minute:02d}:00.000Z",
                "ticker": market,
                "rate": "0.00001",
            }
            for minute in range(0, 30, 5)
        ]
        (manual / f"{market}_funding.json").write_text(json.dumps({"historicalFunding": funding_rows}), encoding="utf-8")

    frame = run_dydx_local_pair_universe(input_dir=manual, run_research=False)

    rows = frame.set_index("pair_id")
    assert rows.loc["btc_eth", "status"] == "built"
    assert (tmp_path / "data" / "processed" / "dydx_funding.csv").exists()
    pair_path = tmp_path / "data" / "raw" / "pair_details" / "pair_btc_eth_5mins_dydx_candles_derived_history.json"
    assert pair_path.exists()
    pair = json.loads(pair_path.read_text(encoding="utf-8"))
    assert pair["history"][0]["funding_x_bps"] != ""
    assert pair["history"][0]["funding_y_bps"] != ""


def test_strategy_acceptance_checklist_explains_spread_only_acceptance_blocker(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    reports = tmp_path / "reports"
    reports.mkdir()
    pd.DataFrame(
        [
            {"status": "evaluated", "backtest_mode": "spread", "cost_bucket": "base"},
            {"status": "evaluated", "backtest_mode": "spread", "cost_bucket": "stress"},
        ]
    ).to_csv(reports / "experiment_results.csv", index=False)
    pd.DataFrame(
        [
            {
                "strategy_id": 1,
                "strategy_name": "Classic ZScore Mean Reversion",
                "family": "zscore",
                "production_eligible": False,
                "preferred_eligible": False,
                "acceptance_reason": "passing_pairs<2;two_leg_pairs<2",
                "preferred_reason": "not_production_eligible",
                "two_leg_pairs_tested": 0,
                "two_leg_passing_pairs": 0,
                "required_cost_buckets": "base;stress",
                "total_trades": 150,
            }
        ]
    ).to_csv(reports / "acceptance_report.csv", index=False)

    frame = strategy_acceptance_checklist_report()
    rows = frame.set_index("step")

    assert bool(rows.loc["experiment_results", "ready"]) is True
    assert rows.loc["two_leg_coverage", "blocker"] == "missing_two_leg_backtests"
    assert bool(rows.loc["cost_bucket_coverage", "ready"]) is True
    assert rows.loc["production_eligibility", "blocker"] == "no_production_eligible_strategy"
    assert "two_leg_pairs<2:1" in rows.loc["production_eligibility", "evidence"]


def test_strategy_acceptance_checklist_narrows_execution_input_blocker_to_funding(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    reports = tmp_path / "reports"
    reports.mkdir()
    pd.DataFrame(
        [
            {"pair": "ETH-BTC", "status": "evaluated", "backtest_mode": "two_leg", "cost_bucket": "base"},
            {"pair": "SOL-ETH", "status": "evaluated", "backtest_mode": "two_leg", "cost_bucket": "stress"},
        ]
    ).to_csv(reports / "experiment_results.csv", index=False)
    pd.DataFrame(
        [
            {
                "strategy_id": 1,
                "strategy_name": "Classic ZScore Mean Reversion",
                "family": "zscore",
                "production_eligible": False,
                "preferred_eligible": False,
                "acceptance_reason": (
                    "passing_pairs<2;two_leg_execution_input_pairs<2;"
                    "two_leg_missing_inputs:ETH-BTC[funding_x+funding_y],SOL-ETH[funding_x+funding_y]"
                ),
                "preferred_reason": "not_production_eligible",
                "two_leg_pairs_tested": 2,
                "two_leg_execution_input_pairs": 0,
                "two_leg_passing_pairs": 0,
                "required_cost_buckets": "base;stress",
                "required_two_leg_inputs": "price_x;price_y;hedge_ratio;beta;funding_x;funding_y",
                "total_trades": 150,
            }
        ]
    ).to_csv(reports / "acceptance_report.csv", index=False)

    frame = strategy_acceptance_checklist_report()
    rows = frame.set_index("step")

    assert bool(rows.loc["two_leg_coverage", "ready"]) is True
    assert rows.loc["two_leg_execution_assumptions", "blocker"] == "missing_funding_inputs"
    assert "funding_x;funding_y" in rows.loc["two_leg_execution_assumptions", "evidence"]
    assert (
        rows.loc["two_leg_execution_assumptions", "next_action"]
        == "fetch/export dYdX funding for required markets, then run funding-coverage"
    )
    assert rows.loc["funding_preflight", "blocker"] == "missing_funding_coverage_report"
    assert "required_markets=BTC-USD;ETH-USD;SOL-USD" in rows.loc["funding_preflight", "evidence"]
    assert "fetch/export dYdX funding for BTC-USD,ETH-USD,SOL-USD" in rows.loc["funding_preflight", "next_action"]
    assert (reports / "funding_requirements.csv").exists()


def test_strategy_acceptance_checklist_surfaces_ready_funding_preflight(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    reports = tmp_path / "reports"
    reports.mkdir()
    pd.DataFrame(
        [
            {"pair": "ETH-BTC", "status": "evaluated", "backtest_mode": "two_leg", "cost_bucket": "base"},
            {"pair": "SOL-ETH", "status": "evaluated", "backtest_mode": "two_leg", "cost_bucket": "stress"},
        ]
    ).to_csv(reports / "experiment_results.csv", index=False)
    pd.DataFrame(
        [
            {
                "strategy_id": 1,
                "strategy_name": "Classic ZScore Mean Reversion",
                "family": "zscore",
                "production_eligible": False,
                "preferred_eligible": False,
                "acceptance_reason": (
                    "passing_pairs<2;two_leg_execution_input_pairs<2;"
                    "two_leg_missing_inputs:ETH-BTC[funding_x+funding_y],SOL-ETH[funding_x+funding_y]"
                ),
                "preferred_reason": "not_production_eligible",
                "two_leg_pairs_tested": 2,
                "two_leg_execution_input_pairs": 0,
                "two_leg_passing_pairs": 0,
                "required_cost_buckets": "base;stress",
                "required_two_leg_inputs": "price_x;price_y;hedge_ratio;beta;funding_x;funding_y",
                "total_trades": 150,
            }
        ]
    ).to_csv(reports / "acceptance_report.csv", index=False)
    pd.DataFrame(
        [
            {"pair": "ETH-BTC", "ready": True, "missing": ""},
            {"pair": "SOL-ETH", "ready": True, "missing": ""},
        ]
    ).to_csv(reports / "funding_coverage.csv", index=False)

    frame = strategy_acceptance_checklist_report()
    rows = frame.set_index("step")

    assert bool(rows.loc["funding_preflight", "ready"]) is True
    assert rows.loc["funding_preflight", "blocker"] == ""
    assert "ready_pairs=2" in rows.loc["funding_preflight", "evidence"]
    assert rows.loc["funding_preflight", "next_action"] == "rerun experiments with --funding-path"


def test_strategy_acceptance_checklist_surfaces_partial_funding_missing_markets(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    reports = tmp_path / "reports"
    reports.mkdir()
    pd.DataFrame(
        [
            {"pair": "ETH-BTC", "status": "evaluated", "backtest_mode": "two_leg", "cost_bucket": "base"},
            {"pair": "SOL-ETH", "status": "evaluated", "backtest_mode": "two_leg", "cost_bucket": "stress"},
        ]
    ).to_csv(reports / "experiment_results.csv", index=False)
    pd.DataFrame(
        [
            {
                "strategy_id": 1,
                "strategy_name": "Classic ZScore Mean Reversion",
                "family": "zscore",
                "production_eligible": False,
                "preferred_eligible": False,
                "acceptance_reason": (
                    "passing_pairs<2;two_leg_execution_input_pairs<2;"
                    "two_leg_missing_inputs:ETH-BTC[funding_x+funding_y],SOL-ETH[funding_x+funding_y]"
                ),
                "preferred_reason": "not_production_eligible",
                "two_leg_pairs_tested": 2,
                "two_leg_execution_input_pairs": 0,
                "two_leg_passing_pairs": 0,
                "required_cost_buckets": "base;stress",
                "required_two_leg_inputs": "price_x;price_y;hedge_ratio;beta;funding_x;funding_y",
                "total_trades": 150,
            }
        ]
    ).to_csv(reports / "acceptance_report.csv", index=False)
    pd.DataFrame(
        [
            {"pair": "ETH-BTC", "ready": True, "missing": "", "missing_markets": ""},
            {"pair": "SOL-ETH", "ready": False, "missing": "funding_y", "missing_markets": "ETH-USD"},
        ]
    ).to_csv(reports / "funding_coverage.csv", index=False)

    frame = strategy_acceptance_checklist_report()
    rows = frame.set_index("step")

    assert rows.loc["funding_preflight", "blocker"] == "incomplete_funding_coverage"
    assert "blocked_pairs=SOL-ETH" in rows.loc["funding_preflight", "evidence"]
    assert "missing_markets=ETH-USD" in rows.loc["funding_preflight", "evidence"]
    assert (
        rows.loc["funding_preflight", "next_action"]
        == "fetch/export dYdX funding for ETH-USD, rerun funding-coverage, then rerun experiments"
    )


def test_dydx_execution_checklist_keeps_order_adapter_as_final_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "build_dydx_indexer_adapter", lambda config: object())
    monkeypatch.setattr("quant_platform.execution.dydx_v4_client_installed", lambda: True)
    monkeypatch.setenv("DYDX_TESTNET_WALLET_ADDRESS", "wallet")
    monkeypatch.setenv("DYDX_TESTNET_PRIVATE_KEY", "private")
    monkeypatch.setenv("DYDX_TESTNET_SUBMIT_ORDERS", "true")
    reports = tmp_path / "reports"
    reports.mkdir()
    pd.DataFrame(
        [
            {
                "strategy_id": 1,
                "production_eligible": True,
                "preferred_eligible": False,
                "two_leg_passing_pairs": 2,
            }
        ]
    ).to_csv(reports / "acceptance_report.csv", index=False)

    frame = dydx_execution_checklist_report()
    rows = frame.set_index("step")

    assert bool(rows.loc["indexer_market_data", "ready"]) is True
    assert bool(rows.loc["testnet_credentials", "ready"]) is True
    assert bool(rows.loc["dydx_sdk", "ready"]) is True
    assert bool(rows.loc["submit_flag", "ready"]) is True
    assert bool(rows.loc["research_acceptance", "ready"]) is True
    assert rows.loc["order_client_adapter", "blocker"] == "missing_dydx_order_client_adapter"
    assert "missing_dydx_order_client_adapter" in rows.loc["paper_submission_gate", "blocker"]


def test_dydx_execution_checklist_accepts_configured_order_adapter(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "build_dydx_indexer_adapter", lambda config: object())
    monkeypatch.setattr("quant_platform.execution.dydx_v4_client_installed", lambda: True)
    monkeypatch.setenv("DYDX_TESTNET_WALLET_ADDRESS", "wallet")
    monkeypatch.setenv("DYDX_TESTNET_PRIVATE_KEY", "private")
    monkeypatch.setenv("DYDX_TESTNET_SUBMIT_ORDERS", "true")
    adapter_module = tmp_path / "fake_cli_order_adapter.py"
    adapter_module.write_text(
        """
from quant_platform.execution import FillReport

class FakeCliOrderAdapter:
    def place_order(self, intent, config):
        return FillReport(
            order_id="fake",
            market=intent.market,
            side=intent.side,
            size=intent.size,
            avg_price=0.0,
            fee=0.0,
            slippage_bps=0.0,
            status="paper_submitted",
        )
""",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("DYDX_TESTNET_ORDER_CLIENT_ADAPTER", "fake_cli_order_adapter:FakeCliOrderAdapter")
    reports = tmp_path / "reports"
    reports.mkdir()
    pd.DataFrame(
        [
            {
                "strategy_id": 1,
                "production_eligible": True,
                "preferred_eligible": False,
                "two_leg_passing_pairs": 2,
            }
        ]
    ).to_csv(reports / "acceptance_report.csv", index=False)

    frame = dydx_execution_checklist_report()
    rows = frame.set_index("step")

    assert bool(rows.loc["order_client_adapter", "ready"]) is True
    assert rows.loc["order_client_adapter", "blocker"] == ""
    assert "order_adapter=True" in rows.loc["order_client_adapter", "evidence"]
    assert bool(rows.loc["paper_submission_gate", "ready"]) is True


def test_dydx_execution_checklist_rejects_record_only_adapter_for_submission(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "build_dydx_indexer_adapter", lambda config: object())
    monkeypatch.setattr("quant_platform.execution.dydx_v4_client_installed", lambda: True)
    monkeypatch.setenv("DYDX_TESTNET_WALLET_ADDRESS", "wallet")
    monkeypatch.setenv("DYDX_TESTNET_PRIVATE_KEY", "private")
    monkeypatch.setenv("DYDX_TESTNET_SUBMIT_ORDERS", "true")
    monkeypatch.setenv(
        "DYDX_TESTNET_ORDER_CLIENT_ADAPTER",
        "quant_platform.dydx_record_only_adapter:RecordOnlyDydxOrderAdapter",
    )
    reports = tmp_path / "reports"
    reports.mkdir()
    pd.DataFrame(
        [
            {
                "strategy_id": 1,
                "production_eligible": True,
                "preferred_eligible": False,
                "two_leg_passing_pairs": 2,
            }
        ]
    ).to_csv(reports / "acceptance_report.csv", index=False)

    frame = dydx_execution_checklist_report()
    rows = frame.set_index("step")

    assert bool(rows.loc["order_client_adapter", "ready"]) is False
    assert rows.loc["order_client_adapter", "blocker"] == "record_only_dydx_order_client_adapter"
    assert "contract_valid=True" in rows.loc["order_client_adapter", "evidence"]
    assert "exchange_submission_capable=False" in rows.loc["order_client_adapter", "evidence"]
    assert "record_only=True" in rows.loc["order_client_adapter", "evidence"]
    assert bool(rows.loc["paper_submission_gate", "ready"]) is False
    assert "record_only_dydx_order_client_adapter" in rows.loc["paper_submission_gate", "blocker"]


def test_dydx_order_adapter_contract_report_surfaces_bad_signature(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    adapter_module = tmp_path / "bad_cli_order_adapter.py"
    adapter_module.write_text(
        """
class BadCliOrderAdapter:
    def place_order(self, intent):
        raise RuntimeError("should not be called")
""",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("DYDX_TESTNET_ORDER_CLIENT_ADAPTER", "bad_cli_order_adapter:BadCliOrderAdapter")

    frame = dydx_order_adapter_contract_report()
    row = frame.iloc[0]

    assert bool(row["configured"]) is True
    assert bool(row["importable"]) is True
    assert bool(row["has_place_order"]) is True
    assert bool(row["signature_accepts_intent_config"]) is False
    assert bool(row["valid"]) is False
    assert row["error"] == "place_order must accept intent and config arguments"
    assert (tmp_path / "reports" / "dydx_order_adapter_contract.csv").exists()


def test_dydx_execution_checklist_rejects_order_adapter_with_bad_signature(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "build_dydx_indexer_adapter", lambda config: object())
    monkeypatch.setattr("quant_platform.execution.dydx_v4_client_installed", lambda: True)
    monkeypatch.setenv("DYDX_TESTNET_WALLET_ADDRESS", "wallet")
    monkeypatch.setenv("DYDX_TESTNET_PRIVATE_KEY", "private")
    monkeypatch.setenv("DYDX_TESTNET_SUBMIT_ORDERS", "true")
    adapter_module = tmp_path / "bad_signature_order_adapter.py"
    adapter_module.write_text(
        """
class BadSignatureOrderAdapter:
    def place_order(self, intent):
        raise RuntimeError("should not be called")
""",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("DYDX_TESTNET_ORDER_CLIENT_ADAPTER", "bad_signature_order_adapter:BadSignatureOrderAdapter")
    reports = tmp_path / "reports"
    reports.mkdir()
    pd.DataFrame(
        [
            {
                "strategy_id": 1,
                "production_eligible": True,
                "preferred_eligible": False,
                "two_leg_passing_pairs": 2,
            }
        ]
    ).to_csv(reports / "acceptance_report.csv", index=False)

    frame = dydx_execution_checklist_report()
    rows = frame.set_index("step")

    assert bool(rows.loc["order_client_adapter", "ready"]) is False
    assert rows.loc["order_client_adapter", "blocker"] == "invalid_dydx_order_client_adapter"
    assert "signature_accepts_intent_config=False" in rows.loc["order_client_adapter", "evidence"]
    assert bool(rows.loc["paper_submission_gate", "ready"]) is False


def test_export_dydx_funding_payload_writes_normalized_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    payload_path = tmp_path / "funding_payload.json"
    output_path = tmp_path / "funding.csv"
    payload_path.write_text(
        json.dumps(
            {
                "historicalFunding": [
                    {"effectiveAt": "2026-01-01T00:00:00Z", "rate": "0.0001"},
                    {"effectiveAt": "2026-01-01T01:00:00Z", "rate": "0.0002"},
                ]
            }
        ),
        encoding="utf-8",
    )

    path = export_dydx_funding_payload(payload_path, output_path, market="ETH-USD")

    written = pd.read_csv(path)
    assert path == output_path
    assert list(written["market"]) == ["ETH-USD", "ETH-USD"]
    assert list(written["funding_bps"]) == [1.0, 2.0]


def test_export_dydx_funding_payload_combines_directory_payloads(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    payload_dir = tmp_path / "funding_payloads"
    payload_dir.mkdir()
    output_path = tmp_path / "funding.csv"
    (payload_dir / "ETH-USD_funding.json").write_text(
        json.dumps({"historicalFunding": [{"effectiveAt": "2026-01-01T00:00:00Z", "rate": "0.0001"}]}),
        encoding="utf-8",
    )
    (payload_dir / "BTC-USD_funding.json").write_text(
        json.dumps({"historicalFunding": [{"effectiveAt": "2026-01-01T00:00:00Z", "rate": "0.0002"}]}),
        encoding="utf-8",
    )

    path = export_dydx_funding_payload(payload_dir, output_path)

    written = pd.read_csv(path)
    assert path == output_path
    assert list(written["market"]) == ["BTC-USD", "ETH-USD"]
    assert list(written["funding_bps"]) == [2.0, 1.0]


def test_fetch_dydx_funding_writes_normalized_indexer_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)

    class FakeFundingIndexer:
        def funding(self, market):
            return {
                "market": market,
                "payload": {
                    "historicalFunding": [
                        {"effectiveAt": "2026-01-01T00:00:00Z", "rate": "0.0001"},
                    ]
                },
            }

    monkeypatch.setattr(cli, "build_dydx_indexer_adapter", lambda config: FakeFundingIndexer())
    output_path = tmp_path / "dydx_funding.csv"

    path = fetch_dydx_funding(["ETH-USD", "BTC-USD"], output_path)

    written = pd.read_csv(path)
    assert path == output_path
    assert list(written["market"]) == ["BTC-USD", "ETH-USD"]
    assert list(written["funding_bps"]) == [1.0, 1.0]


def test_fetch_dydx_funding_blocks_when_indexer_adapter_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "build_dydx_indexer_adapter", lambda config: None)

    with pytest.raises(SystemExit, match="dYdX indexer adapter is not available"):
        fetch_dydx_funding(["ETH-USD"])


def test_funding_coverage_report_infers_pairs_from_experiment_results(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    reports = tmp_path / "reports"
    reports.mkdir()
    pd.DataFrame(
        [
            {"pair": "ETH-BTC", "status": "evaluated"},
            {"pair": "SOL-ETH", "status": "evaluated"},
        ]
    ).to_csv(reports / "experiment_results.csv", index=False)
    funding_path = tmp_path / "funding.csv"
    pd.DataFrame(
        [
            {"market": "ETH-USD", "funding_bps": 2.0},
            {"market": "BTC-USD", "funding_bps": 3.0},
            {"market": "SOL-USD", "funding_bps": 4.0},
        ]
    ).to_csv(funding_path, index=False)

    coverage = funding_coverage_report(funding_path)
    rows = coverage.set_index("pair")

    assert bool(rows.loc["ETH-BTC", "ready"]) is True
    assert bool(rows.loc["SOL-ETH", "ready"]) is True
    assert (reports / "funding_coverage.csv").exists()


def test_funding_requirements_report_infers_markets_from_experiment_results(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    reports = tmp_path / "reports"
    reports.mkdir()
    pd.DataFrame(
        [
            {"pair": "ETH-BTC", "status": "evaluated"},
            {"pair": "SOL-ETH", "status": "evaluated"},
        ]
    ).to_csv(reports / "experiment_results.csv", index=False)

    requirements = funding_requirements_report()
    rows = requirements.set_index("pair")

    assert rows.loc["ETH-BTC", "required_markets"] == "ETH-USD;BTC-USD"
    assert rows.loc["SOL-ETH", "required_markets"] == "SOL-USD;ETH-USD"
    assert (reports / "funding_requirements.csv").exists()


def test_print_funding_requirements_outputs_fetch_market_argument(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "ROOT", tmp_path)

    print_funding_requirements(pair="ETH-BTC")

    output = capsys.readouterr().out
    assert "funding_required_markets: BTC-USD;ETH-USD" in output
    assert "fetch_dydx_funding_market_arg: BTC-USD,ETH-USD" in output
    assert "funding_invalid_pairs: none" in output


def test_funding_template_report_writes_required_market_schema(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    reports = tmp_path / "reports"
    reports.mkdir()
    pd.DataFrame(
        [
            {"pair": "ETH-BTC", "status": "evaluated"},
            {"pair": "SOL-ETH", "status": "evaluated"},
        ]
    ).to_csv(reports / "experiment_results.csv", index=False)
    output = tmp_path / "data" / "processed" / "dydx_funding_template.csv"

    frame = funding_template_report(output_path=output)

    assert list(frame.columns) == ["market", "timestamp", "funding_bps"]
    assert list(frame["market"]) == ["BTC-USD", "ETH-USD", "SOL-USD"]
    written = pd.read_csv(output, dtype=str).fillna("")
    assert list(written.columns) == ["market", "timestamp", "funding_bps"]
    assert list(written["market"]) == ["BTC-USD", "ETH-USD", "SOL-USD"]
    assert list(written["timestamp"]) == ["", "", ""]
    assert list(written["funding_bps"]) == ["", "", ""]


def test_print_funding_template_outputs_path_and_rows(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "ROOT", tmp_path)

    print_funding_template(pair="ETH-BTC")

    output = capsys.readouterr().out
    assert "BTC-USD" in output
    assert "ETH-USD" in output
    assert "funding_template_rows: 2" in output
    assert "data/processed/dydx_funding_template.csv" in output


def test_funding_template_check_blocks_blank_required_values(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    reports = tmp_path / "reports"
    reports.mkdir()
    pd.DataFrame([{"pair": "ETH-BTC", "status": "evaluated"}]).to_csv(reports / "experiment_results.csv", index=False)
    template = tmp_path / "dydx_funding_template.csv"
    pd.DataFrame([{"market": "ETH-USD", "timestamp": "", "funding_bps": ""}]).to_csv(template, index=False)

    frame = funding_template_check_report(template)
    row = frame.iloc[0]

    assert row["rows"] == 1
    assert row["ready_rows"] == 0
    assert row["blocked_rows"] == 1
    assert row["required_markets"] == "BTC-USD;ETH-USD"
    assert row["ready_markets"] == ""
    assert row["missing_markets"] == "BTC-USD;ETH-USD"
    assert "row_2" in row["invalid_rows"]
    assert "funding_bps" in row["invalid_rows"]
    assert bool(row["ready_to_import"]) is False


def test_funding_template_check_reports_ready_to_import_when_all_markets_present(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    reports = tmp_path / "reports"
    reports.mkdir()
    pd.DataFrame([{"pair": "ETH-BTC", "status": "evaluated"}]).to_csv(reports / "experiment_results.csv", index=False)
    template = tmp_path / "dydx_funding_template.csv"
    pd.DataFrame(
        [
            {"market": "ETH-USD", "timestamp": "2026-01-01T00:00:00Z", "funding_bps": "2.0"},
            {"market": "BTC-USD", "timestamp": "2026-01-01T00:00:00Z", "funding_bps": "3.0"},
        ]
    ).to_csv(template, index=False)

    frame = funding_template_check_report(template)
    row = frame.iloc[0]

    assert row["ready_rows"] == 2
    assert row["blocked_rows"] == 0
    assert row["ready_markets"] == "BTC-USD;ETH-USD"
    assert row["missing_markets"] == ""
    assert bool(row["ready_to_import"]) is True
    assert (reports / "funding_template_check.csv").exists()


def test_import_funding_template_writes_normalized_dydx_funding_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    reports = tmp_path / "reports"
    reports.mkdir()
    pd.DataFrame([{"pair": "ETH-BTC", "status": "evaluated"}]).to_csv(reports / "experiment_results.csv", index=False)
    template = tmp_path / "dydx_funding_template.csv"
    output = tmp_path / "data" / "processed" / "dydx_funding.csv"
    pd.DataFrame(
        [
            {"market": "ETH-USD", "timestamp": "2026-01-01T00:00:00Z", "funding_bps": "2.0"},
            {"market": "BTC-USD", "timestamp": "2026-01-01T00:00:00Z", "funding_bps": "3.0"},
        ]
    ).to_csv(template, index=False)

    frame = import_funding_template(template, output)
    row = frame.iloc[0]
    written = pd.read_csv(output)

    assert row["status"] == "imported"
    assert row["imported_rows"] == 2
    assert list(written["market"]) == ["BTC-USD", "ETH-USD"]
    assert list(written["funding_bps"]) == [3.0, 2.0]
    assert (reports / "funding_template_import_report.csv").exists()


def test_import_funding_template_blocks_incomplete_template(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    reports = tmp_path / "reports"
    reports.mkdir()
    pd.DataFrame([{"pair": "ETH-BTC", "status": "evaluated"}]).to_csv(reports / "experiment_results.csv", index=False)
    template = tmp_path / "dydx_funding_template.csv"
    output = tmp_path / "data" / "processed" / "dydx_funding.csv"
    pd.DataFrame([{"market": "ETH-USD", "timestamp": "", "funding_bps": "2.0"}]).to_csv(template, index=False)

    frame = import_funding_template(template, output)
    row = frame.iloc[0]

    assert row["status"] == "blocked"
    assert row["imported_rows"] == 0
    assert "BTC-USD" in row["blocker"]
    assert not output.exists()


def test_print_funding_template_check_outputs_report_path(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    reports = tmp_path / "reports"
    reports.mkdir()
    pd.DataFrame([{"pair": "ETH-BTC", "status": "evaluated"}]).to_csv(reports / "experiment_results.csv", index=False)
    template = tmp_path / "dydx_funding_template.csv"
    pd.DataFrame([{"market": "ETH-USD", "timestamp": "", "funding_bps": ""}]).to_csv(template, index=False)

    print_funding_template_check(template)

    output = capsys.readouterr().out
    assert "funding_template_check:" in output
    assert "ready_to_import" in output


def test_funding_coverage_report_shows_missing_leg_when_file_is_incomplete(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    funding_path = tmp_path / "funding.csv"
    pd.DataFrame([{"market": "ETH-USD", "funding_bps": 2.0}]).to_csv(funding_path, index=False)

    coverage = funding_coverage_report(funding_path, pairs=["ETH-BTC"])
    row = coverage.iloc[0]

    assert bool(row["ready"]) is False
    assert row["missing"] == "funding_y"
    assert row["missing_markets"] == "BTC-USD"
    assert row["required_markets"] == "ETH-USD;BTC-USD"


def test_print_funding_coverage_summarizes_required_and_missing_markets(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    funding_path = tmp_path / "funding.csv"
    pd.DataFrame([{"market": "ETH-USD", "funding_bps": 2.0}]).to_csv(funding_path, index=False)

    print_funding_coverage(funding_path, pair="ETH-BTC")

    output = capsys.readouterr().out
    assert "funding_pairs_ready: 0/1" in output
    assert "funding_required_markets: BTC-USD;ETH-USD" in output
    assert "funding_missing_markets: BTC-USD" in output


def test_funded_research_spine_blocks_when_funding_path_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)

    frame = funded_research_spine(None)
    rows = frame.set_index("step")

    assert rows.loc["funding_coverage", "status"] == "blocked"
    assert rows.loc["funding_coverage", "detail"] == "missing_funding_path"
    assert rows.loc["research_spine", "status"] == "skipped"
    assert (tmp_path / "reports" / "funded_research_spine.csv").exists()


def test_funded_research_spine_blocks_incomplete_funding_coverage(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    reports = tmp_path / "reports"
    reports.mkdir()
    pd.DataFrame([{"pair": "ETH-BTC", "status": "evaluated"}]).to_csv(reports / "experiment_results.csv", index=False)
    funding_path = tmp_path / "funding.csv"
    pd.DataFrame([{"market": "ETH-USD", "funding_bps": 2.0}]).to_csv(funding_path, index=False)

    frame = funded_research_spine(funding_path)
    rows = frame.set_index("step")

    assert rows.loc["funding_coverage", "status"] == "blocked"
    assert "ready_pairs=0/1" in rows.loc["funding_coverage", "detail"]
    assert "missing_markets=BTC-USD" in rows.loc["funding_coverage", "detail"]
    assert rows.loc["research_spine", "status"] == "skipped"
    assert (reports / "funding_coverage.csv").exists()


def test_funded_research_spine_runs_research_after_complete_funding_coverage(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    reports = tmp_path / "reports"
    reports.mkdir()
    pd.DataFrame([{"pair": "ETH-BTC", "status": "evaluated"}]).to_csv(reports / "experiment_results.csv", index=False)
    funding_path = tmp_path / "funding.csv"
    pd.DataFrame(
        [
            {"market": "ETH-USD", "funding_bps": 2.0},
            {"market": "BTC-USD", "funding_bps": 3.0},
        ]
    ).to_csv(funding_path, index=False)
    calls = []

    def fake_research_spine(input_dir=None, require_two_leg=True, funding_path=None):
        calls.append((input_dir, require_two_leg, funding_path))
        return pd.DataFrame([{"step": "run_pair_detail_experiments", "status": "completed", "detail": "ok"}])

    def fake_acceptance(output_path=None):
        pd.DataFrame([{"step": "production_eligibility", "ready": False}]).to_csv(output_path, index=False)
        return pd.DataFrame([{"step": "production_eligibility", "ready": False}])

    monkeypatch.setattr(cli, "research_spine", fake_research_spine)
    monkeypatch.setattr(cli, "strategy_acceptance_checklist_report", fake_acceptance)

    frame = funded_research_spine(funding_path)
    rows = frame.set_index("step")

    assert rows.loc["funding_coverage", "status"] == "completed"
    assert rows.loc["research_spine", "status"] == "completed"
    assert "post_research_funding_ready_pairs=1/1" in rows.loc["research_spine", "detail"]
    assert rows.loc["strategy_acceptance_checklist", "status"] == "completed"
    assert calls == [(None, True, funding_path)]


def test_priority_readiness_report_keeps_learning_store_blocked_for_audit_only_paper_journal(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "build_dydx_indexer_adapter", lambda config: None)
    reports = tmp_path / "reports"
    reports.mkdir()
    pd.DataFrame(
        [
            {
                "timestamp_utc": "2026-01-01T00:00:00+00:00",
                "pair": "ETH-BTC",
                "strategy_id": 1,
                "plan_status": "blocked",
                "plan_reason": "research_rejected:two_leg_pairs<2",
                "blockers": "",
                "intents_json": "[]",
                "fills_json": "[]",
            }
        ]
    ).to_csv(reports / "paper_trading_journal.csv", index=False)

    frame = priority_readiness_report()
    gate = frame.set_index("gate").loc["learning_event_store"]

    assert bool(gate["ready"]) is False
    assert "paper_journal_rows=1" in gate["evidence"]
    assert "outcomes=0" in gate["evidence"]
    assert "audit_only=1" in gate["evidence"]
    assert "outcomes_remaining=100" in gate["evidence"]
    assert "learning_event_summary.csv" in gate["evidence"]
    assert gate["blocker"] == "missing_model_ready_outcomes"


def test_priority_readiness_report_uses_cached_capture_checklist(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "build_dydx_indexer_adapter", lambda config: None)
    reports = tmp_path / "reports"
    reports.mkdir()
    pd.DataFrame(
        [
            {
                "source_path": "pair.json",
                "experiment_ready": True,
                "ecm_history_ready": True,
                "two_leg_execution_ready": True,
            }
        ]
    ).to_csv(reports / "pair_detail_capture_checklist.csv", index=False)
    pd.DataFrame(
        [
            {
                "source_path": "pair.json",
                "research_usable": True,
                "execution_usable": True,
            }
        ]
    ).to_csv(reports / "pair_detail_quality_report.csv", index=False)
    pd.DataFrame(
        [
            {
                "strategy_id": 1,
                "production_eligible": False,
                "preferred_eligible": False,
                "two_leg_pairs_tested": 2,
                "two_leg_execution_input_pairs": 2,
                "two_leg_passing_pairs": 0,
                "total_trades": 10,
                "required_cost_buckets": "base;stress",
                "required_two_leg_inputs": "price_x;price_y;hedge_ratio;beta;funding_x;funding_y",
                "acceptance_reason": "passing_pairs<2",
                "preferred_reason": "not_production_eligible",
            }
        ]
    ).to_csv(reports / "acceptance_report.csv", index=False)
    pd.DataFrame(
        [
            {
                "strategy_id": 1,
                "strategy_name": "Test Strategy",
                "family": "mean_reversion",
                "pair": "SOL-USD-LINK-USD",
                "status": "evaluated",
                "backtest_mode": "two_leg",
                "cost_bucket": "base",
                "eligible": False,
                "trades": 10,
                "observations": 100,
                "profit_factor": 0.9,
                "sharpe": 0.1,
                "expectancy": -0.01,
                "max_drawdown": 0.2,
                "reason": "passing_pairs<2",
            }
        ]
    ).to_csv(reports / "experiment_results.csv", index=False)

    def fail_capture_audit(_):
        raise AssertionError("pair_detail_capture_audit should not run when cached report exists")

    monkeypatch.setattr(cli, "pair_detail_capture_audit", fail_capture_audit)

    frame = priority_readiness_report()
    gate = frame.set_index("gate").loc["pair_detail_capture_audit"]

    assert bool(gate["ready"]) is True
    assert "candidate_paths=1" in gate["evidence"]


def test_priority_readiness_report_marks_learning_store_ready_from_model_ready_outcomes(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "build_dydx_indexer_adapter", lambda config: None)
    reports = tmp_path / "reports"
    reports.mkdir()
    trade_store = tmp_path / "data" / "meta_learning" / "trades.jsonl"
    trade_store.parent.mkdir(parents=True)
    record = {
        "trade_id": "trade-1",
        "timestamp": "2026-01-01T00:00:00",
        "pair": "ETH-BTC",
        "strategy": "Classic ZScore Mean Reversion",
        "regime": "range",
        "features": {"zscore": -2.1},
        "signal": {"side": "LONG_SPREAD"},
        "execution": {"venue": "dydx_testnet"},
        "outcome": {"realized_return": 0.01},
    }
    with trade_store.open("w", encoding="utf-8") as handle:
        for index in range(100):
            row = dict(record)
            row["trade_id"] = f"trade-{index}"
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    frame = priority_readiness_report()
    gate = frame.set_index("gate").loc["learning_event_store"]

    assert bool(gate["ready"]) is True
    assert "trade_store_rows=100" in gate["evidence"]
    assert "outcomes=100" in gate["evidence"]
    assert "outcomes_remaining=0" in gate["evidence"]
    assert "ready_for_modeling=True" in gate["evidence"]
    assert gate["blocker"] == ""


def test_priority_readiness_blocks_record_only_dydx_adapter(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "build_dydx_indexer_adapter", lambda config: object())
    monkeypatch.setattr("quant_platform.execution.dydx_v4_client_installed", lambda: True)
    monkeypatch.setenv("DYDX_TESTNET_WALLET_ADDRESS", "wallet")
    monkeypatch.setenv("DYDX_TESTNET_PRIVATE_KEY", "private")
    monkeypatch.setenv("DYDX_TESTNET_SUBMIT_ORDERS", "true")
    monkeypatch.setenv(
        "DYDX_TESTNET_ORDER_CLIENT_ADAPTER",
        "quant_platform.dydx_record_only_adapter:RecordOnlyDydxOrderAdapter",
    )
    reports = tmp_path / "reports"
    reports.mkdir()
    pd.DataFrame(
        [
            {
                "strategy_id": 1,
                "production_eligible": True,
                "preferred_eligible": False,
                "two_leg_passing_pairs": 2,
            }
        ]
    ).to_csv(reports / "acceptance_report.csv", index=False)

    frame = priority_readiness_report()
    gate = frame.set_index("gate").loc["dydx_testnet_readiness"]
    paper_gate = frame.set_index("gate").loc["paper_execution_gate"]

    assert bool(gate["ready"]) is False
    assert "record_only_dydx_order_client_adapter" in gate["blocker"]
    assert "adapter_contract_valid=True" in gate["evidence"]
    assert "exchange_submission_capable=False" in gate["evidence"]
    assert "record_only=True" in gate["evidence"]
    assert bool(paper_gate["ready"]) is False


def test_append_learning_outcome_writes_trade_store_record(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    trade_store = tmp_path / "data" / "meta_learning" / "trades.jsonl"

    path = append_learning_outcome(
        pair="ETH-BTC",
        strategy_id=1,
        realized_return=0.012,
        signal=1.0,
        hedge_ratio=1.2,
        beta=0.9,
        notional_usd=1000.0,
        regime="range",
        trade_id="trade-1",
        trade_store_path=trade_store,
    )

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert path == trade_store
    assert rows[0]["trade_id"] == "trade-1"
    assert rows[0]["pair"] == "ETH-BTC"
    assert rows[0]["strategy"] == "Classic ZScore Mean Reversion"
    assert rows[0]["regime"] == "range"
    assert rows[0]["features"] == {"beta": 0.9, "hedge_ratio": 1.2}
    assert rows[0]["signal"] == {"value": 1.0}
    assert rows[0]["execution"] == {"notional_usd": 1000.0, "venue": "dydx_testnet"}
    assert rows[0]["outcome"] == {"realized_return": 0.012}


def test_append_learning_outcome_skips_duplicate_explicit_trade_id(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    trade_store = tmp_path / "data" / "meta_learning" / "trades.jsonl"

    for _ in range(2):
        append_learning_outcome(
            pair="ETH-BTC",
            strategy_id=1,
            realized_return=0.012,
            trade_id="trade-1",
            trade_store_path=trade_store,
        )

    rows = [json.loads(line) for line in trade_store.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["trade_id"] == "trade-1"


def test_append_learning_outcome_feeds_learning_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    reports = tmp_path / "reports"
    reports.mkdir()
    trade_store = tmp_path / "data" / "meta_learning" / "trades.jsonl"
    append_learning_outcome(
        pair="SOL-ETH",
        strategy_id=2,
        realized_return=-0.004,
        trade_id="trade-2",
        trade_store_path=trade_store,
    )

    cli.write_learning_report()

    summary = pd.read_csv(reports / "learning_event_summary.csv").set_index("source")
    assert int(summary.loc["trade_store", "events"]) == 1
    assert int(summary.loc["trade_store", "outcome_events"]) == 1
    assert int(summary.loc["trade_store", "profitable_outcomes"]) == 0
    assert int(summary.loc["combined", "outcome_events"]) == 1


def test_learning_outcome_template_report_writes_required_schema(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    output = tmp_path / "data" / "meta_learning" / "learning_outcome_template.csv"

    frame = learning_outcome_template_report(output)

    assert list(frame.columns) == cli.LEARNING_OUTCOME_TEMPLATE_COLUMNS
    assert list(frame["regime"]) == ["unknown"]
    written = pd.read_csv(output, dtype=str).fillna("")
    assert list(written.columns) == cli.LEARNING_OUTCOME_TEMPLATE_COLUMNS
    assert list(written["pair"]) == [""]
    assert list(written["strategy_id"]) == [""]
    assert list(written["realized_return"]) == [""]


def test_learning_outcome_template_check_reports_ready_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    template = tmp_path / "learning_outcomes.csv"
    pd.DataFrame(
        [
            {
                "trade_id": "paper-1",
                "pair": "ETH-BTC",
                "strategy_id": "1",
                "realized_return": "0.012",
                "signal": "1",
                "hedge_ratio": "1.2",
                "beta": "0.9",
                "notional_usd": "1000",
                "regime": "range",
            }
        ]
    ).to_csv(template, index=False)

    frame = learning_outcome_template_check_report(template)
    row = frame.iloc[0]

    assert row["rows"] == 1
    assert row["ready_rows"] == 1
    assert row["blocked_rows"] == 0
    assert row["missing_columns"] == ""
    assert row["invalid_rows"] == ""
    assert bool(row["ready_to_append"]) is True
    assert (tmp_path / "reports" / "learning_outcome_template_check.csv").exists()


def test_learning_outcome_template_check_blocks_missing_required_values(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    template = tmp_path / "learning_outcomes.csv"
    pd.DataFrame([{"pair": "ETH-BTC", "strategy_id": "", "realized_return": "bad"}]).to_csv(template, index=False)

    frame = learning_outcome_template_check_report(template)
    row = frame.iloc[0]

    assert row["ready_rows"] == 0
    assert row["blocked_rows"] == 1
    assert "row_2" in row["invalid_rows"]
    assert "strategy_id" in row["invalid_rows"]
    assert "realized_return" in row["invalid_rows"]
    assert bool(row["ready_to_append"]) is False


def test_print_learning_outcome_template_outputs_path_and_rows(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "ROOT", tmp_path)

    print_learning_outcome_template()

    output = capsys.readouterr().out
    assert "learning_outcome_template_rows: 1" in output
    assert "data/meta_learning/learning_outcome_template.csv" in output


def test_import_learning_outcomes_from_template_appends_ready_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    template = tmp_path / "learning_outcomes.csv"
    trade_store = tmp_path / "data" / "meta_learning" / "trades.jsonl"
    pd.DataFrame(
        [
            {
                "trade_id": "paper-1",
                "pair": "ETH-BTC",
                "strategy_id": "1",
                "realized_return": "0.012",
                "signal": "1",
                "hedge_ratio": "1.2",
                "beta": "0.9",
                "notional_usd": "1000",
                "regime": "range",
            }
        ]
    ).to_csv(template, index=False)

    frame = import_learning_outcomes_from_template(template, trade_store_path=trade_store)
    row = frame.iloc[0]
    records = [json.loads(line) for line in trade_store.read_text(encoding="utf-8").splitlines()]

    assert row["imported_rows"] == 1
    assert row["blocked_rows"] == 0
    assert row["status"] == "imported"
    assert records[0]["trade_id"] == "paper-1"
    assert records[0]["pair"] == "ETH-BTC"
    assert records[0]["strategy"] == "Classic ZScore Mean Reversion"
    assert records[0]["outcome"] == {"realized_return": 0.012}
    assert (tmp_path / "reports" / "learning_outcome_import_report.csv").exists()


def test_import_learning_outcomes_from_template_skips_duplicate_trade_ids(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    template = tmp_path / "learning_outcomes.csv"
    trade_store = tmp_path / "data" / "meta_learning" / "trades.jsonl"
    append_learning_outcome(
        pair="ETH-BTC",
        strategy_id=1,
        realized_return=0.012,
        trade_id="paper-1",
        trade_store_path=trade_store,
    )
    pd.DataFrame(
        [
            {
                "trade_id": "paper-1",
                "pair": "ETH-BTC",
                "strategy_id": "1",
                "realized_return": "0.012",
                "signal": "",
                "hedge_ratio": "",
                "beta": "",
                "notional_usd": "",
                "regime": "range",
            }
        ]
    ).to_csv(template, index=False)

    frame = import_learning_outcomes_from_template(template, trade_store_path=trade_store)
    row = frame.iloc[0]
    records = [json.loads(line) for line in trade_store.read_text(encoding="utf-8").splitlines()]

    assert row["imported_rows"] == 0
    assert row["blocked_rows"] == 0
    assert row["duplicate_rows"] == 1
    assert row["status"] == "skipped_duplicates"
    assert len(records) == 1


def test_import_learning_outcomes_from_template_blocks_invalid_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    template = tmp_path / "learning_outcomes.csv"
    trade_store = tmp_path / "data" / "meta_learning" / "trades.jsonl"
    pd.DataFrame([{"pair": "ETH-BTC", "strategy_id": "", "realized_return": "bad"}]).to_csv(template, index=False)

    frame = import_learning_outcomes_from_template(template, trade_store_path=trade_store)
    row = frame.iloc[0]

    assert row["imported_rows"] == 0
    assert row["blocked_rows"] == 1
    assert row["status"] == "blocked"
    assert "row_2" in row["invalid_rows"]
    assert not trade_store.exists()


def test_priority_action_plan_ranks_blocked_gates_by_priority():
    readiness = pd.DataFrame(
        [
            {
                "priority": "P3",
                "gate": "dydx_testnet_readiness",
                "ready": False,
                "evidence": "order_adapter=False",
                "blocker": "missing_dydx_order_client_adapter",
                "next_action": "wire order adapter",
            },
            {
                "priority": "P1",
                "gate": "pair_detail_two_leg_execution_history",
                "ready": False,
                "evidence": "two_leg_ready=0",
                "blocker": "missing_price_x_or_price_y_history",
                "next_action": "capture leg prices",
            },
            {
                "priority": "P1",
                "gate": "pair_detail_capture_audit",
                "ready": False,
                "evidence": "candidate_paths=0",
                "blocker": "no_nested_execution_ready_history_candidate_detected",
                "next_action": "run capture helper",
            },
            {
                "priority": "P2",
                "gate": "strategy_acceptance",
                "ready": True,
                "evidence": "production_eligible=1",
                "blocker": "",
                "next_action": "allow research-gated paper plans",
            },
        ]
    )

    actions = priority_action_plan(readiness)

    assert list(actions["rank"]) == [1, 2, 3]
    assert list(actions["gate"]) == [
        "pair_detail_capture_audit",
        "pair_detail_two_leg_execution_history",
        "dydx_testnet_readiness",
    ]
    assert actions.loc[0, "depends_on"] == "crypto_wizards_live_artifacts"
    assert actions.loc[1, "depends_on"] == "pair_detail_history"
    assert actions.loc[2, "depends_on"] == "strategy_acceptance"


def test_priority_spine_dashboard_summarizes_checklist_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    reports = tmp_path / "reports"
    reports.mkdir()
    readiness = pd.DataFrame(
        [
            {
                "priority": "P1",
                "gate": "crypto_wizards_live_artifacts",
                "ready": True,
                "status": "ready",
                "evidence": "payloads=1",
                "blocker": "",
                "next_action": "continue",
            },
            {
                "priority": "P1",
                "gate": "pair_detail_history",
                "ready": False,
                "status": "blocked",
                "evidence": "experiment_ready=0",
                "blocker": "missing_spread_zscore_or_ecm_history",
                "next_action": "capture history",
            },
            {
                "priority": "P1",
                "gate": "pair_detail_two_leg_execution_history",
                "ready": False,
                "status": "blocked",
                "evidence": "two_leg_ready=0",
                "blocker": "missing_price_x_or_price_y_history",
                "next_action": "capture leg prices",
            },
            {
                "priority": "P1",
                "gate": "pair_detail_capture_audit",
                "ready": False,
                "status": "blocked",
                "evidence": "candidate_paths=0",
                "blocker": "no_nested_execution_ready_history_candidate_detected",
                "next_action": "run capture helper",
            },
            {
                "priority": "P2",
                "gate": "strategy_acceptance",
                "ready": False,
                "status": "blocked",
                "evidence": "production_eligible=0",
                "blocker": "no_strategy_passes_production_gates",
                "next_action": "run experiments",
            },
            {
                "priority": "P3",
                "gate": "dydx_testnet_readiness",
                "ready": False,
                "status": "blocked",
                "evidence": "submit_orders=False",
                "blocker": "submit_orders_false",
                "next_action": "keep blocked",
            },
            {
                "priority": "P4",
                "gate": "paper_execution_gate",
                "ready": False,
                "status": "blocked",
                "evidence": "strategy_ready=False;dydx_ready=False",
                "blocker": "strategy_or_dydx_gate_not_ready",
                "next_action": "do not submit",
            },
            {
                "priority": "P5",
                "gate": "learning_event_store",
                "ready": False,
                "status": "blocked",
                "evidence": "paper_journal_rows=0",
                "blocker": "missing_learning_events",
                "next_action": "append records",
            },
        ]
    )
    pd.DataFrame(
        [
            {
                "research_spine_ready": False,
                "next_capture_focus": "capture_baseline_history:spread;zscore",
                "missing_required_fields": "spread;zscore",
            }
        ]
    ).to_csv(reports / "pair_detail_capture_checklist.csv", index=False)
    pd.DataFrame(
        [
            {"ready": True, "next_action": "continue"},
            {
                "ready": False,
                "blocker": "missing_two_leg_backtests",
                "next_action": "capture price_x/price_y and rerun two-leg experiments",
            },
        ]
    ).to_csv(reports / "strategy_acceptance_checklist.csv", index=False)
    pd.DataFrame([{"ready": True}, {"ready": False, "blocker": "submit_orders_false"}]).to_csv(
        reports / "dydx_execution_checklist.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "source": "combined",
                "events": 0,
                "outcome_events": 0,
                "outcome_events_remaining": 100,
                "ready_for_modeling": False,
            }
        ]
    ).to_csv(reports / "learning_event_summary.csv", index=False)

    dashboard = priority_spine_dashboard_report(readiness)
    rows = dashboard.set_index("area")

    assert list(dashboard["priority"]) == ["P1", "P2", "P3", "P4", "P5"]
    assert rows.loc["crypto_wizards_capture", "blocker"] == "no_nested_execution_ready_history_candidate_detected"
    assert "next_focus=capture_baseline_history:spread;zscore" in rows.loc["crypto_wizards_capture", "key_metric"]
    assert rows.loc["strategy_acceptance", "key_metric"] == "steps_ready=1/2;first_blocker=missing_two_leg_backtests"
    assert rows.loc["strategy_acceptance", "next_action"] == "capture price_x/price_y and rerun two-leg experiments"
    assert (
        rows.loc["learning_event_store", "key_metric"]
        == "events=0;outcomes=0;outcomes_remaining=100;ready_for_modeling=False"
    )


def test_print_priority_dashboard_refreshes_stale_paper_preflight(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    reports = tmp_path / "reports"
    reports.mkdir()
    readiness = pd.DataFrame(
        [
            {
                "priority": "P2",
                "gate": "strategy_acceptance",
                "ready": False,
                "status": "blocked",
                "evidence": "production_eligible=0",
                "blocker": "no_strategy_passes_production_gates",
                "next_action": "generic stale strategy action",
            },
            {
                "priority": "P3",
                "gate": "dydx_testnet_readiness",
                "ready": False,
                "status": "blocked",
                "evidence": "submit_orders=False",
                "blocker": "submit_orders_false",
                "next_action": "keep blocked",
            },
            {
                "priority": "P4",
                "gate": "paper_execution_gate",
                "ready": False,
                "status": "blocked",
                "evidence": "strategy_ready=False;dydx_ready=False",
                "blocker": "strategy_or_dydx_gate_not_ready",
                "next_action": "stale paper action",
            },
            {
                "priority": "P5",
                "gate": "learning_event_store",
                "ready": False,
                "status": "blocked",
                "evidence": "events=0",
                "blocker": "missing_learning_events",
                "next_action": "append records",
            },
        ]
    )
    monkeypatch.setattr(cli, "priority_readiness_report", lambda: readiness)
    pd.DataFrame(
        [
            {"ready": True, "next_action": "continue"},
            {
                "ready": False,
                "blocker": "missing_funding_inputs",
                "next_action": "fetch/export dYdX funding for required markets, then run funding-coverage",
            },
        ]
    ).to_csv(reports / "strategy_acceptance_checklist.csv", index=False)
    pd.DataFrame([{"ready": False, "blocker": "submit_orders_false", "next_action": "keep blocked"}]).to_csv(
        reports / "dydx_execution_checklist.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "step": "strategy_acceptance_dependency",
                "ready": False,
                "blocker": "old_blocker",
                "next_action": "old stale action",
            }
        ]
    ).to_csv(reports / "paper_execution_preflight.csv", index=False)

    cli.print_priority_dashboard()

    capsys.readouterr()
    dashboard = pd.read_csv(reports / "priority_spine_dashboard.csv").set_index("area")
    preflight = pd.read_csv(reports / "paper_execution_preflight.csv").set_index("step")
    expected = "fetch/export dYdX funding for required markets, then run funding-coverage"
    assert preflight.loc["strategy_acceptance_dependency", "next_action"] == expected
    assert dashboard.loc["paper_execution_gate", "next_action"] == expected


def test_paper_execution_preflight_reports_dependency_blockers(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "build_dydx_indexer_adapter", lambda config: None)
    monkeypatch.setattr("quant_platform.execution.dydx_v4_client_installed", lambda: False)
    reports = tmp_path / "reports"
    reports.mkdir()
    pd.DataFrame(
        [
            {
                "strategy_id": 1,
                "production_eligible": False,
                "preferred_eligible": False,
                "acceptance_reason": "passing_pairs<2",
                "two_leg_pairs_tested": 0,
                "two_leg_passing_pairs": 0,
            }
        ]
    ).to_csv(reports / "acceptance_report.csv", index=False)
    pd.DataFrame(
        [
            {"pair": "ETH-BTC", "required_markets": "ETH-USD;BTC-USD", "valid": True},
            {"pair": "SOL-ETH", "required_markets": "SOL-USD;ETH-USD", "valid": True},
        ]
    ).to_csv(reports / "funding_requirements.csv", index=False)

    frame = paper_execution_preflight_report()
    rows = frame.set_index("step")

    assert rows.loc["strategy_acceptance_dependency", "blocker"] == "no_strategy_passes_production_gates"
    assert "submit_orders_false" in rows.loc["dydx_testnet_dependency", "blocker"]
    assert rows.loc["paper_submission_gate", "blocker"] == "strategy_or_dydx_gate_not_ready"
    assert rows.loc["paper_journal", "blocker"] == "missing_paper_trading_journal"
    assert (reports / "paper_execution_preflight.csv").exists()


def test_paper_execution_preflight_marks_submission_ready_when_dependencies_ready(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "build_dydx_indexer_adapter", lambda config: object())
    monkeypatch.setattr("quant_platform.execution.dydx_v4_client_installed", lambda: True)
    monkeypatch.setenv("DYDX_TESTNET_WALLET_ADDRESS", "wallet")
    monkeypatch.setenv("DYDX_TESTNET_PRIVATE_KEY", "private")
    monkeypatch.setenv("DYDX_TESTNET_SUBMIT_ORDERS", "true")
    adapter_module = tmp_path / "ready_order_adapter.py"
    adapter_module.write_text(
        """
from quant_platform.execution import FillReport

class ReadyOrderAdapter:
    def place_order(self, intent, config):
        return FillReport(
            order_id="ready",
            market=intent.market,
            side=intent.side,
            size=intent.size,
            avg_price=0.0,
            fee=0.0,
            slippage_bps=0.0,
            status="paper_submitted",
        )
""",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("DYDX_TESTNET_ORDER_CLIENT_ADAPTER", "ready_order_adapter:ReadyOrderAdapter")
    reports = tmp_path / "reports"
    reports.mkdir()
    pd.DataFrame(
        [
            {
                "strategy_id": 1,
                "production_eligible": True,
                "preferred_eligible": False,
                "two_leg_pairs_tested": 2,
                "two_leg_passing_pairs": 2,
            }
        ]
    ).to_csv(reports / "acceptance_report.csv", index=False)
    pd.DataFrame([{"plan_status": "paper_ready"}]).to_csv(reports / "paper_trading_journal.csv", index=False)

    frame = paper_execution_preflight_report()
    rows = frame.set_index("step")

    assert bool(rows.loc["strategy_acceptance_dependency", "ready"]) is True
    assert bool(rows.loc["dydx_testnet_dependency", "ready"]) is True
    assert bool(rows.loc["paper_submission_gate", "ready"]) is True
    assert bool(rows.loc["paper_journal", "ready"]) is True


def test_priority_runbook_writes_operator_markdown(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "build_dydx_indexer_adapter", lambda config: None)
    monkeypatch.setattr("quant_platform.execution.dydx_v4_client_installed", lambda: False)
    reports = tmp_path / "reports"
    pair_details = tmp_path / "data" / "raw" / "pair_details"
    reports.mkdir(parents=True)
    pair_details.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "strategy_id": 1,
                "production_eligible": False,
                "preferred_eligible": False,
                "acceptance_reason": "passing_pairs<2",
                "two_leg_pairs_tested": 0,
                "two_leg_passing_pairs": 0,
            }
        ]
    ).to_csv(reports / "acceptance_report.csv", index=False)
    pd.DataFrame(
        [
            {"pair": "ETH-BTC", "required_markets": "ETH-USD;BTC-USD", "valid": True},
            {"pair": "SOL-ETH", "required_markets": "SOL-USD;ETH-USD", "valid": True},
        ]
    ).to_csv(reports / "funding_requirements.csv", index=False)

    output = priority_runbook()
    text = output.read_text(encoding="utf-8")

    assert output == reports / "priority_runbook.md"
    assert "# Priority Spine Runbook" in text
    assert "## Current Dashboard" in text
    assert "## Gap Proof Required" in text
    assert "## Ranked Work Queue" in text
    assert "## Operator Commands" in text
    assert "await __CW_CAPTURE_STATUS__()" in text
    assert "P2 funding coverage" in text
    assert "funding-template --output-path data/processed/dydx_funding_template.csv" in text
    assert "funding-template-check --input-dir data/processed/dydx_funding_template.csv" in text
    assert "import-funding-template --input-dir data/processed/dydx_funding_template.csv" in text
    assert "fetch-dydx-funding --market BTC-USD,ETH-USD,SOL-USD" in text
    assert "funded-research-spine --funding-path data/processed/dydx_funding.csv" in text
    assert "learning-outcome-template --output-path data/meta_learning/learning_outcome_template.csv" in text
    assert "learning-outcome-template-check --input-dir data/meta_learning/learning_outcome_template.csv" in text
    assert "import-learning-outcomes --input-dir data/meta_learning/learning_outcome_template.csv" in text
    assert "reports/strategy_acceptance_checklist.csv" in text


def test_priority_gap_test_report_classifies_open_gaps(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    reports = tmp_path / "reports"
    reports.mkdir()
    readiness = pd.DataFrame(
        [
            {
                "priority": "P1",
                "gate": "crypto_wizards_live_artifacts",
                "ready": True,
                "status": "ready",
                "evidence": "payloads=1",
                "blocker": "",
                "next_action": "continue",
            },
            {
                "priority": "P1",
                "gate": "pair_detail_history",
                "ready": False,
                "status": "blocked",
                "evidence": "experiment_ready=0",
                "blocker": "missing_spread_zscore_or_ecm_history",
                "next_action": "capture history",
            },
            {
                "priority": "P1",
                "gate": "pair_detail_two_leg_execution_history",
                "ready": False,
                "status": "blocked",
                "evidence": "two_leg_ready=0",
                "blocker": "missing_price_x_or_price_y_history",
                "next_action": "capture legs",
            },
            {
                "priority": "P1",
                "gate": "pair_detail_capture_audit",
                "ready": False,
                "status": "blocked",
                "evidence": "candidate_paths=0",
                "blocker": "no_nested_execution_ready_history_candidate_detected",
                "next_action": "run capture helper",
            },
            {
                "priority": "P2",
                "gate": "strategy_acceptance",
                "ready": False,
                "status": "blocked",
                "evidence": "production_eligible=0",
                "blocker": "no_strategy_passes_production_gates",
                "next_action": "run experiments",
            },
            {
                "priority": "P3",
                "gate": "dydx_testnet_readiness",
                "ready": False,
                "status": "blocked",
                "evidence": "submit_orders=False",
                "blocker": "submit_orders_false",
                "next_action": "keep blocked",
            },
            {
                "priority": "P4",
                "gate": "paper_execution_gate",
                "ready": False,
                "status": "blocked",
                "evidence": "strategy_ready=False;dydx_ready=False",
                "blocker": "strategy_or_dydx_gate_not_ready",
                "next_action": "do not submit",
            },
            {
                "priority": "P5",
                "gate": "learning_event_store",
                "ready": False,
                "status": "blocked",
                "evidence": "paper_journal_rows=0",
                "blocker": "missing_learning_events",
                "next_action": "append records",
            },
        ]
    )
    pd.DataFrame([{"next_capture_focus": "capture_baseline_history:spread;zscore"}]).to_csv(
        reports / "pair_detail_capture_checklist.csv", index=False
    )
    pd.DataFrame([{"ready": False, "blocker": "missing_two_leg_backtests"}]).to_csv(
        reports / "strategy_acceptance_checklist.csv", index=False
    )
    pd.DataFrame([{"ready": False, "blocker": "submit_orders_false"}]).to_csv(
        reports / "dydx_execution_checklist.csv", index=False
    )
    pd.DataFrame([{"source": "combined", "events": 0, "outcome_events": 0, "ready_for_modeling": False}]).to_csv(
        reports / "learning_event_summary.csv", index=False
    )

    report = priority_gap_test_report(readiness)
    rows = report.set_index("area")

    assert rows.loc["crypto_wizards_capture", "severity"] == "critical"
    assert rows.loc["strategy_acceptance", "severity"] == "critical"
    assert rows.loc["dydx_testnet_readiness", "severity"] == "high"
    assert rows.loc["paper_execution_gate", "severity"] == "high"
    assert rows.loc["learning_event_store", "severity"] == "medium"
    assert "spread,zscore,ecm_x" in rows.loc["crypto_wizards_capture", "required_proof"]


def test_gap_analysis_checklist_builds_checkpoint_files(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    fake_gap_report = pd.DataFrame(
        [
            {
                "priority": "P1",
                "area": "crypto_wizards_capture",
                "status": "gap",
                "severity": "critical",
                "gap": "missing_nested_payload",
                "current_evidence": "captures=0",
                "required_proof": "run capture helper",
                "source_report": "reports/pair_detail_capture_checklist.csv",
                "next_action": "capture history",
            },
            {
                "priority": "P2",
                "area": "strategy_acceptance",
                "status": "gap",
                "severity": "high",
                "gap": "missing_strategy_results",
                "current_evidence": "results=0",
                "required_proof": "run strategy experiments",
                "source_report": "reports/acceptance_report.csv",
                "next_action": "run experiments",
            },
        ]
    )
    monkeypatch.setattr(cli, "priority_gap_test_report", lambda readiness=None: fake_gap_report)

    gap_csv, gap_md = cli.print_gap_analysis_checklist()

    assert gap_csv.exists()
    assert gap_md.exists()
    csv_rows = pd.read_csv(gap_csv)
    assert len(csv_rows) == 2
    assert set(csv_rows["status"]) == {"gap"}
    assert set(csv_rows["severity"]) == {"critical", "high"}
    assert (tmp_path / "reports" / "gap_analysis_index.csv").exists()
    assert "Gap Analysis Checkpoint" in gap_md.read_text(encoding="utf-8")


def test_pre_mortem_checklist_builds_checkpoint_files(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    fake_gap_report = pd.DataFrame(
        [
            {
                "priority": "P1",
                "area": "crypto_wizards_capture",
                "status": "gap",
                "severity": "critical",
                "gap": "missing_nested_payload",
                "current_evidence": "captures=0",
                "required_proof": "run capture helper",
                "source_report": "reports/pair_detail_capture_checklist.csv",
                "next_action": "capture history",
            },
            {
                "priority": "P2",
                "area": "strategy_acceptance",
                "status": "gap",
                "severity": "high",
                "gap": "missing_strategy_results",
                "current_evidence": "results=0",
                "required_proof": "run strategy experiments",
                "source_report": "reports/acceptance_report.csv",
                "next_action": "run experiments",
            },
        ]
    )
    monkeypatch.setattr(cli, "priority_gap_test_report", lambda readiness=None: fake_gap_report)

    pm_csv, pm_md = cli.print_pre_mortem_checklist()

    assert pm_csv.exists()
    assert pm_md.exists()
    rows = pd.read_csv(pm_csv)
    assert len(rows) == 2
    assert set(rows["status"]) == {"gap"}
    assert set(rows["failure_mode"])  # non-empty
    assert set(rows["prevention"])  # non-empty
    assert "pre_mortem_question" in rows.columns
    assert "Pre-Mortem Checkpoint" in pm_md.read_text(encoding="utf-8")
    assert (tmp_path / "reports" / "pre_mortem_index.csv").exists()


def test_post_mortem_checklist_builds_checkpoint_files(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    fake_gap_report = pd.DataFrame(
        [
            {
                "priority": "P1",
                "area": "crypto_wizards_capture",
                "status": "gap",
                "severity": "critical",
                "gap": "missing_nested_payload",
                "current_evidence": "captures=0",
                "required_proof": "run capture helper",
                "source_report": "reports/pair_detail_capture_checklist.csv",
                "next_action": "capture history",
            },
            {
                "priority": "P2",
                "area": "strategy_acceptance",
                "status": "pass",
                "severity": "none",
                "gap": "",
                "current_evidence": "results=100",
                "required_proof": "run strategy experiments",
                "source_report": "reports/acceptance_report.csv",
                "next_action": "run experiments",
            },
        ]
    )
    monkeypatch.setattr(cli, "priority_gap_test_report", lambda readiness=None: fake_gap_report)

    pm_path = tmp_path / "reports" / "post_mortem" / "latest_post_mortem.csv"
    pm_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "run_id": "post_mortem_previous",
                "timestamp_utc": "2026-01-01_000000Z",
                "priority": "P1",
                "area": "crypto_wizards_capture",
                "status": "pass",
            },
            {
                "run_id": "post_mortem_previous",
                "timestamp_utc": "2026-01-01_000000Z",
                "priority": "P2",
                "area": "strategy_acceptance",
                "status": "gap",
            },
        ]
    ).to_csv(pm_path, index=False)

    post_csv, post_md = cli.print_post_mortem_checklist()

    assert post_csv.exists()
    assert post_md.exists()
    rows = pd.read_csv(post_csv)
    assert len(rows) == 2
    assert set(rows["trajectory"]) == {"regressed", "resolved"}
    assert set(rows["incident_observed"])  # non-empty
    assert "trajectory" in rows.columns
    assert "post_mortem_insight" in rows.columns
    assert "Post-Mortem Checkpoint" in post_md.read_text(encoding="utf-8")
    assert (tmp_path / "reports" / "post_mortem_index.csv").exists()


def test_red_team_checklist_builds_checkpoint_files(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    fake_gap_report = pd.DataFrame(
        [
            {
                "priority": "P1",
                "area": "crypto_wizards_capture",
                "status": "gap",
                "severity": "critical",
                "gap": "missing_nested_payload",
                "current_evidence": "captures=0",
                "required_proof": "run capture helper",
                "source_report": "reports/pair_detail_capture_checklist.csv",
                "next_action": "capture history",
            },
            {
                "priority": "P2",
                "area": "strategy_acceptance",
                "status": "pass",
                "severity": "low",
                "gap": "",
                "current_evidence": "results=100",
                "required_proof": "run strategy experiments",
                "source_report": "reports/acceptance_report.csv",
                "next_action": "run experiments",
            },
        ]
    )
    monkeypatch.setattr(cli, "priority_gap_test_report", lambda readiness=None: fake_gap_report)

    red_csv, red_md = cli.print_red_team_checklist()

    assert red_csv.exists()
    assert red_md.exists()
    rows = pd.read_csv(red_csv)
    assert len(rows) == 2
    assert set(rows["status"]) == {"gap", "pass"}
    assert "red_team_hypothesis" in rows.columns
    assert "attack_vector" in rows.columns
    assert "adversarial_question" in rows.columns
    assert "control_test" in rows.columns
    assert (rows.loc[rows["status"] == "pass", "done"] == True).all()
    assert "Red Team Checkpoint" in red_md.read_text(encoding="utf-8")
    assert (tmp_path / "reports" / "red_team_index.csv").exists()


def test_supreme_team_checkpoint_builds_next_action_plan(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)

    reports = tmp_path / "reports"
    gap_dir = reports / "gap_analysis"
    pre_dir = reports / "pre_mortem"
    post_dir = reports / "post_mortem"
    red_dir = reports / "red_team"
    gap_dir.mkdir(parents=True, exist_ok=True)
    pre_dir.mkdir(parents=True, exist_ok=True)
    post_dir.mkdir(parents=True, exist_ok=True)
    red_dir.mkdir(parents=True, exist_ok=True)

    fake_gap = gap_dir / "gap_fake.csv"
    fake_pre = pre_dir / "pre_fake.csv"
    fake_post = post_dir / "post_fake.csv"
    fake_red = red_dir / "red_fake.csv"

    pd.DataFrame(
        [
            {
                "run_id": "run_1",
                "timestamp_utc": "2026-01-01_000000Z",
                "priority": "P1",
                "area": "crypto_wizards_capture",
                "status": "gap",
                "severity": "critical",
                "gap": "missing_nested_payload",
                "current_evidence": "missing payload",
                "required_proof": "capture quality feed",
                "source_report": "reports/pair_detail_capture_checklist.csv",
                "next_action": "capture history",
                "done": False,
            }
        ]
    ).to_csv(fake_gap, index=False)
    pd.DataFrame(
        [
            {
                "run_id": "run_1",
                "timestamp_utc": "2026-01-01_000000Z",
                "priority": "P1",
                "area": "crypto_wizards_capture",
                "status": "gap",
                "severity": "critical",
                "gap": "missing_nested_payload",
                "current_evidence": "none",
                "required_proof": "capture quality feed",
                "source_report": "reports/pair_detail_capture_checklist.csv",
                "pre_mortem_question": "q",
                "failure_mode": "f",
                "prevention": "collect missing feeds",
                "done": False,
            }
        ]
    ).to_csv(fake_pre, index=False)
    pd.DataFrame(
        [
            {
                "run_id": "run_1",
                "timestamp_utc": "2026-01-01_000000Z",
                "priority": "P2",
                "area": "strategy_acceptance",
                "status": "pass",
                "severity": "high",
                "gap": "",
                "current_evidence": "",
                "required_proof": "strategy evidence",
                "source_report": "reports/strategy_acceptance_checklist.csv",
                "incident_observed": "",
                "trajectory": "resolved",
                "post_mortem_insight": "resolved already",
                "prevention_from_pre_mortem": "",
                "done": True,
            }
        ]
    ).to_csv(fake_post, index=False)
    pd.DataFrame(
        [
            {
                "run_id": "run_1",
                "timestamp_utc": "2026-01-01_000000Z",
                "priority": "P3",
                "area": "dydx_testnet_readiness",
                "status": "pass",
                "severity": "high",
                "gap": "",
                "current_evidence": "",
                "required_proof": "execute checklist",
                "source_report": "reports/dydx_execution_checklist.csv",
                "red_team_hypothesis": "",
                "attack_vector": "",
                "adversarial_question": "",
                "control_test": "run adapter contract check",
                "done": True,
            }
        ]
    ).to_csv(fake_red, index=False)

    monkeypatch.setattr(cli, "print_gap_analysis_checklist", lambda run_dir=None: (fake_gap, fake_gap.with_suffix(".md")))
    monkeypatch.setattr(cli, "print_pre_mortem_checklist", lambda run_dir=None: (fake_pre, fake_pre.with_suffix(".md")))
    monkeypatch.setattr(cli, "print_post_mortem_checklist", lambda run_dir=None: (fake_post, fake_post.with_suffix(".md")))
    monkeypatch.setattr(cli, "print_red_team_checklist", lambda run_dir=None: (fake_red, fake_red.with_suffix(".md")))

    (fake_gap.with_suffix(".md")).write_text("# gap", encoding="utf-8")
    (fake_pre.with_suffix(".md")).write_text("# pre", encoding="utf-8")
    (fake_post.with_suffix(".md")).write_text("# post", encoding="utf-8")
    (fake_red.with_suffix(".md")).write_text("# red", encoding="utf-8")

    plan_csv, plan_md = cli.print_supreme_team_checkpoint()

    assert plan_csv.exists()
    assert plan_md.exists()
    rows = pd.read_csv(plan_csv)
    assert len(rows) == 2
    assert set(rows["source_checkpoint"]) == {"gap_analysis", "pre_mortem"}
    assert list(rows["status"]) == ["gap", "gap"]
    assert "Supreme Team Checkpoint" in plan_md.read_text(encoding="utf-8")
    assert (tmp_path / "reports" / "supreme_team_index.csv").exists()
    assert plan_csv.exists()


def test_research_spine_skips_experiments_when_two_leg_history_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "build_dydx_indexer_adapter", lambda config: None)
    pair_details = tmp_path / "data" / "raw" / "pair_details"
    pair_details.mkdir(parents=True)
    sample = Path(__file__).parent / "fixtures" / "pair_detail_view_item_sample.json"
    (pair_details / "pair_1.json").write_text(sample.read_text(encoding="utf-8"), encoding="utf-8")

    frame = research_spine(input_dir=pair_details)
    rows = frame.set_index("step")

    assert rows.loc["ingest_pair_details", "status"] == "completed"
    assert rows.loc["priority_readiness", "status"] == "completed"
    assert rows.loc["run_pair_detail_experiments", "status"] == "skipped"
    assert rows.loc["run_pair_detail_experiments", "detail"] == "missing_price_x_or_price_y_history"
    assert (tmp_path / "reports" / "research_spine.csv").exists()


def test_research_spine_can_run_spread_only_when_explicitly_allowed(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "build_dydx_indexer_adapter", lambda config: None)
    pair_details = tmp_path / "data" / "raw" / "pair_details"
    pair_details.mkdir(parents=True)
    sample = Path(__file__).parent / "fixtures" / "pair_detail_view_item_sample.json"
    (pair_details / "pair_1.json").write_text(sample.read_text(encoding="utf-8"), encoding="utf-8")

    frame = research_spine(input_dir=pair_details, require_two_leg=False)
    rows = frame.set_index("step")

    assert rows.loc["run_pair_detail_experiments", "status"] == "completed"
    assert "priority_readiness_after_experiments" in rows.index
    assert (tmp_path / "reports" / "experiment_results.csv").exists()


def test_run_fixture_experiments_enriches_funding_from_file(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    normalized = tmp_path / "data" / "normalized" / "fixture_test"
    normalized.mkdir(parents=True)
    reports = tmp_path / "reports"
    reports.mkdir()
    fixture_rows = []
    for idx in range(6):
        fixture_rows.append(
            {
                "timestamp": f"2026-01-01T0{idx}:00:00Z",
                "pair": "ETH-BTC",
                "spread": [-2.0, -1.0, 0.0, 1.0, 2.0, 0.0][idx],
                "zscore": [-2.2, -1.0, 0.0, 1.0, 2.2, 0.0][idx],
                "price_x": 100.0 + idx,
                "price_y": 50.0 + idx * 0.5,
                "hedge_ratio": 1.2,
            }
        )
    pd.DataFrame(fixture_rows).to_csv(normalized / "pairs.csv", index=False)
    funding_path = tmp_path / "funding.csv"
    pd.DataFrame(
        [
            {"market": "ETH-USD", "funding_bps": 2.0},
            {"market": "BTC-USD", "funding_bps": 3.0},
        ]
    ).to_csv(funding_path, index=False)

    cli.run_fixture_experiments(normalized, funding_path)

    results = pd.read_csv(reports / "experiment_results.csv")
    evaluated = results[results["status"] == "evaluated"]
    assert not evaluated.empty
    assert evaluated["has_funding_x"].all()
    assert evaluated["has_funding_y"].all()


def test_run_fixture_experiments_blocks_raw_enrichment_feed(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    raw_feed = tmp_path / "data" / "raw" / "enrichment" / "hyperliquid"
    raw_feed.mkdir(parents=True)
    (raw_feed / "hyperliquid_pairs.json").write_text(
        json.dumps(
            [
                {"timestamp": "2026-01-01T00:00:00Z", "pair": "ETH-BTC", "spread": -1.0, "zscore": -1.0, "price_x": 100, "price_y": 50},
                {"timestamp": "2026-01-01T00:05:00Z", "pair": "ETH-BTC", "spread": 0.0, "zscore": 0.0, "price_x": 101, "price_y": 50.5},
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="Run normalize-enrichment-fixtures first"):
        cli.run_fixture_experiments(raw_feed)


def test_normalize_enrichment_fixtures_writes_to_separate_normalized_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    raw_feed = tmp_path / "data" / "raw" / "enrichment" / "hyperliquid"
    raw_feed.mkdir(parents=True)
    (raw_feed / "hyperliquid_pairs.json").write_text(
        json.dumps(
            [
                {"timestamp": "2026-01-01T00:00:00Z", "pair": "ETH-BTC", "spread": -1.0, "zscore": -1.0, "price_x": 100, "price_y": 50},
                {"timestamp": "2026-01-01T00:05:00Z", "pair": "ETH-BTC", "spread": 0.0, "zscore": 0.0, "price_x": 101, "price_y": 50.5},
                {"timestamp": "2026-01-01T00:10:00Z", "pair": "ETH-BTC", "spread": 1.0, "zscore": 1.0, "price_x": 102, "price_y": 51},
            ]
        ),
        encoding="utf-8",
    )

    output = cli.normalize_enrichment_fixtures("hyperliquid", raw_feed)

    assert output == tmp_path / "data" / "normalized" / "hyperliquid" / "hyperliquid_normalized_pairs.csv"
    frame = pd.read_csv(output)
    assert "source" in frame.columns
    assert set(frame["source"]) == {"hyperliquid"}
    assert not str(output).startswith(str(tmp_path / "data" / "raw" / "dydx_manual"))
    assert (tmp_path / "reports" / "hyperliquid_normalization_report.csv").exists()


def test_normalize_enrichment_fixtures_rejects_dydx_manual_target(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    raw_feed = tmp_path / "data" / "raw" / "enrichment" / "gmx"
    raw_feed.mkdir(parents=True)
    (raw_feed / "gmx_pairs.json").write_text(
        json.dumps(
            [
                {"timestamp": "2026-01-01T00:00:00Z", "pair": "ETH-BTC", "spread": -1.0, "zscore": -1.0, "price_x": 100, "price_y": 50},
                {"timestamp": "2026-01-01T00:05:00Z", "pair": "ETH-BTC", "spread": 0.0, "zscore": 0.0, "price_x": 101, "price_y": 50.5},
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="only dYdX actors may write"):
        cli.normalize_enrichment_fixtures("gmx", raw_feed, tmp_path / "data" / "raw" / "dydx_manual")


def test_run_fixture_experiments_accepts_normalized_enrichment_feed(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    raw_feed = tmp_path / "data" / "raw" / "enrichment" / "dexscreener"
    raw_feed.mkdir(parents=True)
    records = []
    for idx, spread in enumerate([-2.0, -1.0, 0.0, 1.0, 2.0, 0.0]):
        records.append(
            {
                "timestamp": f"2026-01-01T0{idx}:00:00Z",
                "pair": "ETH-BTC",
                "spread": spread,
                "zscore": spread,
                "price_x": 100.0 + idx,
                "price_y": 50.0 + idx * 0.5,
            }
        )
    (raw_feed / "dex_pairs.json").write_text(json.dumps(records), encoding="utf-8")

    output = cli.normalize_enrichment_fixtures("dexscreener", raw_feed)
    cli.run_fixture_experiments(output.parent)

    results = pd.read_csv(tmp_path / "reports" / "experiment_results.csv")
    assert not results.empty


def test_materialize_p2_rerun_subset_prefers_long_history_replacement(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    pair_dir = tmp_path / "data" / "raw" / "pair_details"
    pair_dir.mkdir(parents=True)
    reports = tmp_path / "reports"
    reports.mkdir()
    candles_path = pair_dir / "pair_sol_link_5mins_dydx_candles_derived_history.json"
    long_path = pair_dir / "pair_sol_link_5mins_dydx_long_history_derived_history.json"
    other_path = pair_dir / "pair_btc_eth_5mins_dydx_candles_derived_history.json"
    candles_path.write_text('{"pair":"SOL-USD-LINK-USD","history":[{"tag":"short"}]}', encoding="utf-8")
    long_path.write_text('{"pair":"SOL-USD-LINK-USD","history":[{"tag":"long"}]}', encoding="utf-8")
    other_path.write_text('{"pair":"BTC-USD-ETH-USD","history":[{"tag":"base"}]}', encoding="utf-8")
    pd.DataFrame(
        [
            {"path": str(candles_path), "pair": "SOL-USD-LINK-USD", "history_rows": 100, "research_usable": True, "execution_usable": True},
            {"path": str(other_path), "pair": "BTC-USD-ETH-USD", "history_rows": 120, "research_usable": True, "execution_usable": True},
        ]
    ).to_csv(reports / "pair_detail_quality_report.csv", index=False)

    frame = materialize_p2_rerun_subset(output_dir=tmp_path / "work" / "subset")
    rows = frame.set_index("pair")

    assert rows.loc["SOL-USD-LINK-USD", "detail"] == "long_history_replacement"
    copied = json.loads((tmp_path / "work" / "subset" / long_path.name).read_text(encoding="utf-8"))
    assert copied["history"][0]["tag"] == "long"
    assert (reports / "p2_rerun_subset_manifest.csv").exists()


def test_materialize_p2_rerun_subset_uses_quality_report_source_when_no_long_history_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    pair_dir = tmp_path / "data" / "raw" / "pair_details"
    pair_dir.mkdir(parents=True)
    reports = tmp_path / "reports"
    reports.mkdir()
    source = pair_dir / "pair_btc_link_5mins_dydx_candles_derived_history.json"
    source.write_text('{"pair":"BTC-USD-LINK-USD","history":[{"tag":"base"}]}', encoding="utf-8")
    pd.DataFrame(
        [
            {"path": str(source), "pair": "BTC-USD-LINK-USD", "history_rows": 100, "research_usable": True, "execution_usable": True},
        ]
    ).to_csv(reports / "pair_detail_quality_report.csv", index=False)

    frame = materialize_p2_rerun_subset(output_dir=tmp_path / "work" / "subset")
    row = frame.iloc[0]

    assert row["detail"] == "quality_report_source"
    copied = json.loads((tmp_path / "work" / "subset" / source.name).read_text(encoding="utf-8"))
    assert copied["history"][0]["tag"] == "base"


def test_research_spine_runs_strict_two_leg_experiments_when_capture_is_ready(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "build_dydx_indexer_adapter", lambda config: None)
    pair_details = tmp_path / "data" / "raw" / "pair_details"
    pair_details.mkdir(parents=True)
    history = []
    for idx in range(80):
        spread = ((idx % 16) - 8) / 4.0
        history.append(
            {
                "spread": spread,
                "zscore": spread,
                "price_x": 600.0 + idx + spread,
                "price_y": 2.0 + idx * 0.01 - spread * 0.01,
                "hedge_ratio": 1.3,
                "beta": 0.9,
                "funding_x_bps": 2.0,
                "funding_y_bps": 3.0,
                "ecm_x": -0.2,
                "ecm_y": -0.1,
                "ecm_strength": 0.8,
            }
        )
    (pair_details / "pair_1_two_leg.json").write_text(
        json.dumps(
            {
                "pair_id": "1",
                "pair": "BNB-USD-STX-USD",
                "asset_x": "BNB-USD",
                "asset_y": "STX-USD",
                "exchange": "dydx",
                "history": history,
            }
        ),
        encoding="utf-8",
    )

    frame = research_spine(input_dir=pair_details)
    rows = frame.set_index("step")

    assert rows.loc["run_pair_detail_experiments", "status"] == "completed"
    results = pd.read_csv(tmp_path / "reports" / "experiment_results.csv")
    evaluated = results[results["status"] == "evaluated"]
    assert not evaluated.empty
    assert set(evaluated["backtest_mode"]) == {"two_leg"}
    assert evaluated["has_price_x"].all()
    assert evaluated["has_price_y"].all()


def test_research_spine_enriches_pair_detail_experiments_with_funding_path(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "build_dydx_indexer_adapter", lambda config: None)
    pair_details = tmp_path / "data" / "raw" / "pair_details"
    pair_details.mkdir(parents=True)
    history = []
    for idx in range(80):
        spread = ((idx % 16) - 8) / 4.0
        history.append(
            {
                "spread": spread,
                "zscore": spread,
                "price_x": 100.0 + idx,
                "price_y": 50.0 + idx * 0.5,
                "hedge_ratio": 1.2,
                "beta": 0.9,
                "ecm_x": -0.2,
                "ecm_y": -0.1,
                "ecm_strength": 0.8,
            }
        )
    (pair_details / "pair_1_two_leg.json").write_text(
        json.dumps(
            {
                "pair_id": "1",
                "pair": "ETH-BTC",
                "asset_x": "ETH",
                "asset_y": "BTC",
                "exchange": "dydx",
                "history": history,
            }
        ),
        encoding="utf-8",
    )
    funding_path = tmp_path / "funding.csv"
    pd.DataFrame(
        [
            {"market": "ETH-USD", "funding_bps": 2.0},
            {"market": "BTC-USD", "funding_bps": 3.0},
        ]
    ).to_csv(funding_path, index=False)

    frame = research_spine(input_dir=pair_details, funding_path=funding_path)
    rows = frame.set_index("step")

    assert rows.loc["run_pair_detail_experiments", "status"] == "completed"
    assert f"funding_path={funding_path}" in rows.loc["run_pair_detail_experiments", "detail"]
    results = pd.read_csv(tmp_path / "reports" / "experiment_results.csv")
    evaluated = results[results["status"] == "evaluated"]
    assert not evaluated.empty
    assert evaluated["has_funding_x"].all()
    assert evaluated["has_funding_y"].all()
