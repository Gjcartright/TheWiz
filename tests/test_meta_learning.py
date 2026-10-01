import json
from datetime import datetime

import pandas as pd
import pytest

from quant_platform.meta_learning import (
    JsonlTradeStore,
    LEARNING_EVENT_SUMMARY_COLUMNS,
    TradeRecord,
    learning_event_summary,
    write_learning_event_summary_report,
)


def test_learning_event_summary_combines_paper_journal_and_trade_store(tmp_path):
    paper_journal = tmp_path / "paper_trading_journal.csv"
    trade_store = tmp_path / "trades.jsonl"
    pd.DataFrame(
        [
            {
                "timestamp_utc": "2026-01-01T00:00:00+00:00",
                "pair": "ETH-BTC",
                "strategy_id": 1,
                "plan_status": "blocked",
                "plan_reason": "research_rejected",
                "blockers": "",
                "intents_json": "[]",
                "fills_json": "[]",
            },
            {
                "timestamp_utc": "2026-01-01T01:00:00+00:00",
                "pair": "SOL-ETH",
                "strategy_id": 1,
                "plan_status": "paper_ready",
                "plan_reason": "accepted",
                "blockers": "",
                "intents_json": "[]",
                "fills_json": json.dumps(
                    [
                        {"status": "paper_submitted"},
                        {"status": "paper_blocked_missing_client"},
                    ]
                ),
            }
        ]
    ).to_csv(paper_journal, index=False)
    JsonlTradeStore(trade_store).append(
        TradeRecord(
            trade_id="trade-1",
            timestamp=datetime(2026, 1, 1),
            pair="ETH-BTC",
            strategy="Classic ZScore Mean Reversion",
            regime="range",
            features={"zscore": -2.1},
            signal={"side": "LONG_SPREAD"},
            execution={"venue": "dydx_testnet"},
            outcome={"realized_return": 0.012},
        )
    )

    rows = learning_event_summary(paper_journal, trade_store, min_modeling_events=2)
    summary = {row["source"]: row for row in rows}

    assert summary["paper_journal"]["events"] == 2
    assert summary["paper_journal"]["blocked_events"] == 1
    assert summary["paper_journal"]["research_rejected_events"] == 1
    assert summary["paper_journal"]["paper_ready_events"] == 1
    assert summary["paper_journal"]["submitted_fill_events"] == 1
    assert summary["paper_journal"]["blocked_fill_events"] == 1
    assert summary["trade_store"]["events"] == 1
    assert summary["trade_store"]["outcome_events"] == 0
    assert summary["trade_store"]["profitable_outcomes"] == 0
    assert summary["combined"]["events"] == 3
    assert summary["combined"]["outcome_events"] == 0
    assert summary["combined"]["audit_only_events"] == 3
    assert summary["combined"]["outcome_events_remaining"] == 2
    assert summary["combined"]["ready_for_modeling"] is False
    assert "reported_normalized_outcomes=1" in summary["combined"]["notes"]
    assert "verified_outcome_consumer_not_bound" in summary["combined"]["notes"]


def test_learning_event_summary_requires_realized_outcomes_not_audit_only_events(tmp_path):
    paper_journal = tmp_path / "paper_trading_journal.csv"
    trade_store = tmp_path / "trades.jsonl"
    pd.DataFrame(
        [
            {
                "timestamp_utc": f"2026-01-01T{i:02d}:00:00+00:00",
                "pair": "ETH-BTC",
                "strategy_id": 1,
                "plan_status": "blocked",
                "plan_reason": "research_rejected",
                "blockers": "",
                "intents_json": "[]",
                "fills_json": "[]",
            }
            for i in range(5)
        ]
    ).to_csv(paper_journal, index=False)
    JsonlTradeStore(trade_store).append(
        TradeRecord(
            trade_id="trade-1",
            timestamp=datetime(2026, 1, 1),
            pair="ETH-BTC",
            strategy="Classic ZScore Mean Reversion",
            regime="range",
            features={"zscore": -2.1},
            signal={"side": "LONG_SPREAD"},
            execution={"venue": "dydx_testnet"},
            outcome={"realized_return": 0.012},
        )
    )

    rows = learning_event_summary(paper_journal, trade_store, min_modeling_events=5)
    combined = {row["source"]: row for row in rows}["combined"]

    assert combined["events"] == 6
    assert combined["outcome_events"] == 0
    assert combined["audit_only_events"] == 6
    assert combined["outcome_events_remaining"] == 5
    assert combined["ready_for_modeling"] is False


def test_write_learning_event_summary_report_handles_missing_inputs(tmp_path):
    output = tmp_path / "learning_event_summary.csv"

    path = write_learning_event_summary_report(
        tmp_path / "missing_paper.csv",
        tmp_path / "missing_trades.jsonl",
        output,
    )

    report = pd.read_csv(path)
    assert list(report.columns) == LEARNING_EVENT_SUMMARY_COLUMNS
    assert list(report["source"]) == ["paper_journal", "trade_store", "combined"]
    assert int(report.loc[report["source"] == "combined", "events"].iloc[0]) == 0
    assert int(report.loc[report["source"] == "combined", "outcome_events_remaining"].iloc[0]) == 100
    assert path == output


def test_learning_event_summary_does_not_promote_single_leg_exit_claim(tmp_path):
    paper_journal = tmp_path / "paper_trading_journal.csv"
    trade_store = tmp_path / "trades.jsonl"
    pd.DataFrame(
        [
            {
                "timestamp_utc": "2026-01-01T00:00:00+00:00",
                "pair": "ETH-BTC",
                "strategy_id": 1,
                "plan_status": "paper_submitted",
                "plan_reason": "accepted",
                "blockers": "",
                "intents_json": "[]",
                "fills_json": json.dumps([{"status": "paper_submitted"}]),
                "trade_id": "t-1",
                "venue": "dydx",
                "lifecycle_status": "open",
                "opened_timestamp_utc": "2026-01-01T00:00:00+00:00",
                "closed_timestamp_utc": "",
                "entry_snapshot_json": "{}",
                "exit_snapshot_json": "{}",
                "realized_return": "",
                "outcome_label": "",
            },
            {
                "timestamp_utc": "2026-01-01T01:00:00+00:00",
                "pair": "ETH-BTC",
                "strategy_id": 1,
                "plan_status": "paper_completed",
                "plan_reason": "realized_outcome_recorded",
                "blockers": "",
                "intents_json": "[]",
                "fills_json": json.dumps([{"status": "paper_submitted"}]),
                "trade_id": "t-1",
                "venue": "dydx",
                "lifecycle_status": "closed",
                "opened_timestamp_utc": "2026-01-01T00:00:00+00:00",
                "closed_timestamp_utc": "2026-01-01T01:00:00+00:00",
                "entry_snapshot_json": "{}",
                "exit_snapshot_json": json.dumps({"venue_exit_price_x": 101.0}),
                "realized_return": "0.02",
                "outcome_label": "win",
            },
        ]
    ).to_csv(paper_journal, index=False)

    rows = learning_event_summary(paper_journal, trade_store, min_modeling_events=1)
    summary = {row["source"]: row for row in rows}

    assert summary["paper_journal"]["outcome_events"] == 0
    assert summary["paper_journal"]["profitable_outcomes"] == 0
    assert summary["paper_journal"]["ready_for_modeling"] is False
    assert "verified_outcome_consumer_not_bound" in summary["paper_journal"]["notes"]


def test_learning_event_summary_does_not_count_unverified_closed_journal_rows(tmp_path):
    paper_journal = tmp_path / "paper_trading_journal.csv"
    trade_store = tmp_path / "trades.jsonl"
    pd.DataFrame(
        [
            {
                "timestamp_utc": "2026-01-01T01:00:00+00:00",
                "pair": "ETH-BTC",
                "strategy_id": 1,
                "plan_status": "paper_completed",
                "plan_reason": "realized_outcome_recorded",
                "blockers": "",
                "intents_json": "[]",
                "fills_json": json.dumps([{"status": "paper_submitted"}]),
                "trade_id": "t-1",
                "venue": "dydx",
                "lifecycle_status": "closed",
                "opened_timestamp_utc": "2026-01-01T00:00:00+00:00",
                "closed_timestamp_utc": "2026-01-01T01:00:00+00:00",
                "entry_snapshot_json": "{}",
                "exit_snapshot_json": "{}",
                "realized_return": "0.02",
                "outcome_label": "win",
            },
        ]
    ).to_csv(paper_journal, index=False)

    rows = learning_event_summary(paper_journal, trade_store, min_modeling_events=1)
    summary = {row["source"]: row for row in rows}

    assert summary["paper_journal"]["outcome_events"] == 0
    assert summary["paper_journal"]["profitable_outcomes"] == 0
    assert summary["paper_journal"]["ready_for_modeling"] is False
    assert "verified_outcome_consumer_not_bound" in summary["paper_journal"]["notes"]


def test_learning_summary_keeps_unverified_claims_out_of_modeling_gate(tmp_path):
    paper = tmp_path / "paper.csv"
    trades = tmp_path / "trades.jsonl"
    pd.DataFrame([{
        "plan_status": "paper_completed",
        "lifecycle_status": "closed",
        "trade_id": "same-trade",
        "pair": "ETH-BTC",
        "venue": "dydx",
        "realized_return": 0.02,
        "exit_snapshot_json": json.dumps({"venue_exit_price_x": 101.0, "venue_exit_price_y": 99.0}),
    }]).to_csv(paper, index=False)
    rows = [
        {"trade_id": "same-trade", "pair": "ETH-BTC", "execution": {"venue": "dydx"}, "outcome": {"realized_return": 0.02}},
        {"trade_id": "same-trade", "pair": "ETH-BTC", "execution": {"venue": "dydx"}, "outcome": {"realized_return": 0.02}},
        {"trade_id": "conflict", "pair": "ETH-BTC", "execution": {"venue": "dydx"}, "outcome": {"realized_return": 0.03}},
        {"trade_id": "conflict", "pair": "ETH-BTC", "execution": {"venue": "dydx"}, "outcome": {"realized_return": -0.03}},
        {"trade_id": "currency", "outcome": {"pnl": 1000}},
        {"trade_id": "ambiguous", "outcome": {"pnl_pct": 12}},
        {"trade_id": "boolean", "outcome": {"realized_return": True}},
        {"trade_id": "infinite", "outcome": {"realized_return": "inf"}},
        {"trade_id": "", "outcome": {"realized_return": 0.5}, "pair": "ETH-BTC"},
        {"trade_id": "dual", "outcome": {"realized_return": 0.1, "return": 0.2}},
    ]
    trades.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    summary = {row["source"]: row for row in learning_event_summary(paper, trades, min_modeling_events=1)}

    assert summary["combined"]["events"] == 11
    assert summary["combined"]["outcome_events"] == 0
    assert summary["combined"]["ready_for_modeling"] is False
    assert summary["combined"]["outcome_events_remaining"] == 1
    assert summary["combined"]["audit_only_events"] == 11
    assert "reported_normalized_outcomes=1" in summary["combined"]["notes"]
    assert summary["combined"]["profitable_outcomes"] == 0


@pytest.mark.parametrize("threshold", [0, -1, True, 1.5])
def test_learning_summary_requires_positive_integer_threshold(tmp_path, threshold):
    with pytest.raises(ValueError, match="min_modeling_events"):
        learning_event_summary(tmp_path / "paper.csv", tmp_path / "trades.jsonl", threshold)
