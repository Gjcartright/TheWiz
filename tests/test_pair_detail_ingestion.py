import json

import pytest

from quant_platform.pair_detail_ingestion import (
    datasets_from_pair_detail_snapshots,
    extract_history_rows,
    pair_detail_capture_audit,
    pair_detail_capture_checklist,
    pair_detail_field_rows,
    pair_detail_history_coverage,
    pair_detail_quality_report,
    pair_detail_payload_capture_checklist,
    parse_pair_detail_text,
    write_pair_detail_reports,
)


PAIR_DETAIL_TEXT = """Crypto Wizards
BNB-USD STX-USD dyd
pair/1
1.05
hurst
51.3%
corr
1.36
hedge r
9 0σ
1 2σ
54.5%
returns
3.02
sharpe
periods
Daily
4 Hour
1 Hour
5 Min
Static (Spread)
dydx
320 periods analyzed
BNB-USD (asset X)
STX-USD (asset Y)
CORRELATION (Returns)
Pearsons
ρ
78.2%
Spearmans
ρ
66.4%
Kendalls
τ
48.4%
COPULA STATISTICS (Prices)
Best fit
clayton
Correlation
(ρ)
51.3%
BNB-USD given STX-USD
67.4%
STX-USD given BNB-USD
11.9%
betas
correlation
volatilities
ecm (y)
ecm (x)
ecm strength
Stop Loss
%
ECM Deviation (min)
%
0.69
(BNB-USD capital weighting)
sharpe:3.02
sortino:7.06
net return:54.5%
annualized return:64.3%
mean period return:0.17%
win rate:100.0%
closed trades:1
max drawdown:-4.5%
VaR (at 99%):-1.7%
CVaR (at 99%):-3.1%
VaR (sim): -5.29%
CVaR (sim): -8.85%
"""


def test_parse_pair_detail_text_extracts_dashboard_research_fields():
    snapshot = parse_pair_detail_text(PAIR_DETAIL_TEXT, source_url="https://cryptowizards.net/wizards/zscore/pair/1")

    assert snapshot.pair_id == "1"
    assert snapshot.pair == "BNB-USD-STX-USD"
    assert snapshot.asset_x == "BNB-USD"
    assert snapshot.asset_y == "STX-USD"
    assert snapshot.exchange == "dyd"
    assert snapshot.period == 320
    assert snapshot.hurst == pytest.approx(1.05)
    assert snapshot.hedge_ratio == pytest.approx(1.36)
    assert snapshot.copula == "clayton"
    assert snapshot.u1_given_u2 == pytest.approx(0.674)
    assert snapshot.u2_given_u1 == pytest.approx(0.119)
    assert snapshot.ecm_x_available
    assert snapshot.ecm_y_available
    assert snapshot.ecm_strength_available
    assert snapshot.ecm_deviation_override_available
    assert snapshot.sharpe == pytest.approx(3.02)
    assert snapshot.sortino == pytest.approx(7.06)
    assert snapshot.drawdown == pytest.approx(-0.045)


def test_pair_detail_field_rows_include_ecm_availability():
    snapshot = parse_pair_detail_text(PAIR_DETAIL_TEXT)

    rows = pair_detail_field_rows([snapshot])
    fields = {row["field"] for row in rows}

    assert "ecm_x_available" in fields
    assert "ecm_y_available" in fields
    assert "ecm_strength_available" in fields


def test_pair_detail_history_payload_becomes_pair_dataset(tmp_path):
    payload = {
        "pair_id": "1",
        "pair": "BNB-USD-STX-USD",
        "asset_x": "BNB-USD",
        "asset_y": "STX-USD",
        "exchange": "dydx",
        "hedge_ratio": 1.36,
        "ecm_x_available": True,
        "ecm_y_available": True,
        "ecm_strength_available": True,
        "history": [
            {"spread": -0.2, "zscore": -2.2, "ecm_x": -0.4, "ecm_y": -0.1, "ecm_strength": 0.7},
            {"spread": -0.1, "zscore": -1.1, "ecm_x": -0.2, "ecm_y": -0.05, "ecm_strength": 0.7},
            {"spread": 0.0, "zscore": 0.0, "ecm_x": -0.01, "ecm_y": -0.01, "ecm_strength": 0.7},
        ],
    }
    (tmp_path / "pair_1.json").write_text(json.dumps(payload), encoding="utf-8")

    datasets = datasets_from_pair_detail_snapshots(tmp_path)

    assert len(datasets) == 1
    assert datasets[0].pair == "BNB-USD-STX-USD"
    assert {"spread", "zscore", "ecm_x", "ecm_y", "ecm_strength", "hedge_ratio", "regime"}.issubset(
        datasets[0].frame.columns
    )


def test_extract_history_rows_accepts_nested_view_item_records():
    payload = {
        "viewItem": {
            "history": [
                {"spread": -0.2, "zscore": -2.0},
                {"spread": 0.1, "zscore": 1.0},
            ]
        }
    }

    rows = extract_history_rows(payload)

    assert rows == [{"spread": -0.2, "zscore": -2.0}, {"spread": 0.1, "zscore": 1.0}]


def test_extract_history_rows_accepts_parallel_series_payload():
    payload = {
        "spread": [-0.2, -0.1, 0.0],
        "zscore": [-2.0, -1.0, 0.0],
        "ecm_x": [-0.3, -0.2, -0.1],
        "ecm_y": [-0.1, -0.05, -0.01],
        "ecm_strength": [0.8, 0.8, 0.7],
    }

    rows = extract_history_rows(payload)

    assert rows[0] == {"spread": -0.2, "zscore": -2.0, "ecm_x": -0.3, "ecm_y": -0.1, "ecm_strength": 0.8}
    assert len(rows) == 3


def test_extract_history_rows_accepts_har_response_content_text():
    payload = {
        "log": {
            "entries": [
                {
                    "request": {"url": "https://cryptowizards.net/api/pair/1"},
                    "response": {
                        "content": {
                            "mimeType": "application/json",
                            "text": json.dumps(
                                {
                                    "history": [
                                        {
                                            "spread": -0.2,
                                            "zscore": -2.0,
                                            "ecm_x": -0.3,
                                            "ecm_y": -0.1,
                                            "ecm_strength": 0.8,
                                        },
                                        {
                                            "spread": 0.0,
                                            "zscore": 0.0,
                                            "ecm_x": -0.1,
                                            "ecm_y": -0.05,
                                            "ecm_strength": 0.7,
                                        },
                                    ]
                                }
                            ),
                        }
                    },
                }
            ]
        }
    }

    rows = extract_history_rows(payload)

    assert len(rows) == 2
    assert rows[0]["spread"] == -0.2
    assert rows[0]["ecm_strength"] == 0.8


def test_capture_checklist_flags_har_without_response_bodies():
    payload = {
        "log": {
            "entries": [
                {
                    "request": {
                        "url": "https://indexer.dydx.trade/v4/candles/perpetualMarkets/BNB-USD?resolution=1DAY"
                    },
                    "response": {"status": 200, "content": {"mimeType": "application/json", "size": 30000}},
                }
            ]
        }
    }

    row = pair_detail_payload_capture_checklist(payload, "crypto_wizards_network.har")

    assert row["capture_har_entries"] == 1
    assert row["capture_har_response_texts"] == 0
    assert row["capture_har_dydx_candle_requests"] == 1
    assert row["capture_payload_sources"] == "har"
    assert row["capture_operator_hint"] == "har_has_requests_but_no_response_bodies:copy_har_with_content_or_copy_response"


def test_extract_history_rows_accepts_alternate_leg_price_and_funding_aliases():
    payload = {
        "spread": [-0.2, 0.0],
        "zscore": [-2.0, 0.0],
        "asset_x_prices": [600.0, 602.0],
        "asset_y_prices": [1.8, 1.85],
        "pair_betas": [0.9, 0.91],
        "asset_x_funding_bps": [2.0, 2.1],
        "asset_y_funding_bps": [3.0, 3.1],
        "ecm_x": [-0.3, -0.1],
        "ecm_y": [-0.1, -0.05],
        "ecm_strength": [0.8, 0.7],
    }

    rows = extract_history_rows(payload)

    assert rows == [
        {
            "spread": -0.2,
            "zscore": -2.0,
            "price_x": 600.0,
            "price_y": 1.8,
            "beta": 0.9,
            "funding_x_bps": 2.0,
            "funding_y_bps": 3.0,
            "ecm_x": -0.3,
            "ecm_y": -0.1,
            "ecm_strength": 0.8,
        },
        {
            "spread": 0.0,
            "zscore": 0.0,
            "price_x": 602.0,
            "price_y": 1.85,
            "beta": 0.91,
            "funding_x_bps": 2.1,
            "funding_y_bps": 3.1,
            "ecm_x": -0.1,
            "ecm_y": -0.05,
            "ecm_strength": 0.7,
        },
    ]


def test_extract_history_rows_canonicalizes_list_record_aliases():
    payload = {
        "history": [
            {
                "spread": -0.2,
                "zscore_last": -2.0,
                "asset_x_price": 600.0,
                "asset_y_price": 1.8,
                "pair_beta": 0.9,
                "asset_x_funding_bps": 2.0,
                "asset_y_funding_bps": 3.0,
                "ecm_x": -0.3,
                "ecm_y": -0.1,
                "ecm_strength": 0.8,
            },
            {
                "spread": 0.0,
                "zscore_last": 0.0,
                "asset_x_price": 602.0,
                "asset_y_price": 1.85,
                "pair_beta": 0.91,
                "asset_x_funding_bps": 2.1,
                "asset_y_funding_bps": 3.1,
                "ecm_x": -0.1,
                "ecm_y": -0.05,
                "ecm_strength": 0.7,
            },
        ]
    }

    rows = extract_history_rows(payload)

    assert rows[0] == {
        "spread": -0.2,
        "zscore": -2.0,
        "price_x": 600.0,
        "price_y": 1.8,
        "beta": 0.9,
        "funding_x_bps": 2.0,
        "funding_y_bps": 3.0,
        "ecm_x": -0.3,
        "ecm_y": -0.1,
        "ecm_strength": 0.8,
    }
    assert rows[1]["price_x"] == 602.0


def test_extract_history_rows_accepts_browser_capture_worker_message():
    payload = {
        "worker_messages": [
            {"direction": "to_worker", "message": {"pair": {"symbol_1": "BNB-USD"}}},
            {
                "direction": "from_worker",
                "message": {
                    "result": {
                        "spread": [-0.2, 0.0],
                        "zscore": [-2.0, 0.0],
                    }
                },
            },
        ]
    }

    rows = extract_history_rows(payload)

    assert rows == [{"spread": -0.2, "zscore": -2.0}, {"spread": 0.0, "zscore": 0.0}]


def test_extract_history_rows_accepts_browser_capture_xhr_json():
    payload = {
        "xhrs": [
            {
                "url": "/internal/pair",
                "json": {
                    "data": {
                        "spread": [-0.2, 0.0],
                        "zscore": [-2.0, 0.0],
                        "ecm_x": [-0.3, -0.1],
                    }
                },
            }
        ]
    }

    rows = extract_history_rows(payload)

    assert rows == [
        {"spread": -0.2, "zscore": -2.0, "ecm_x": -0.3},
        {"spread": 0.0, "zscore": 0.0, "ecm_x": -0.1},
    ]


def test_extract_history_rows_accepts_browser_storage_json():
    payload = {
        "storage": [
            {
                "area": "sessionStorage",
                "key": "wizard_pair_view",
                "json": {
                    "history": [
                        {"spread": -0.2, "zscore": -2.0},
                        {"spread": 0.0, "zscore": 0.0},
                    ]
                },
            }
        ]
    }

    rows = extract_history_rows(payload)

    assert rows == [{"spread": -0.2, "zscore": -2.0}, {"spread": 0.0, "zscore": 0.0}]


def test_extract_history_rows_accepts_browser_script_json():
    payload = {
        "scripts": [
            {
                "type": "application/json",
                "json": {
                    "viewItem": {
                        "spread": [-0.2, 0.0],
                        "zscore": [-2.0, 0.0],
                        "ecm_x": [-0.3, -0.1],
                        "ecm_y": [-0.1, -0.05],
                        "ecm_strength": [0.8, 0.7],
                    }
                },
            }
        ]
    }

    rows = extract_history_rows(payload)

    assert rows == [
        {"spread": -0.2, "zscore": -2.0, "ecm_x": -0.3, "ecm_y": -0.1, "ecm_strength": 0.8},
        {"spread": 0.0, "zscore": 0.0, "ecm_x": -0.1, "ecm_y": -0.05, "ecm_strength": 0.7},
    ]


def test_extract_history_rows_accepts_indexeddb_capture_rows():
    payload = {
        "indexeddb": [
            {
                "name": "crypto-wizards",
                "stores": [
                    {
                        "name": "pair-cache",
                        "rows": [
                            {
                                "key": "pair-1",
                                "value": {
                                    "data": {
                                        "spread": [-0.2, 0.0],
                                        "zscore": [-2.0, 0.0],
                                        "price_x": [600.0, 602.0],
                                        "price_y": [1.8, 1.85],
                                        "ecm_x": [-0.3, -0.1],
                                        "ecm_y": [-0.1, -0.05],
                                        "ecm_strength": [0.8, 0.7],
                                    }
                                },
                            }
                        ],
                    }
                ],
            }
        ]
    }

    rows = extract_history_rows(payload)

    assert rows[0]["spread"] == -0.2
    assert rows[0]["price_x"] == 600.0
    assert rows[1]["ecm_strength"] == 0.7


def test_extract_history_rows_accepts_json_string_list_capture_value():
    payload = {
        "indexeddb": [
            {
                "stores": [
                    {
                        "rows": [
                            {
                                "value": json.dumps(
                                    [
                                        {
                                            "data": {
                                                "spread": [-0.2, 0.0],
                                                "zscore": [-2.0, 0.0],
                                                "ecm_x": [-0.3, -0.1],
                                                "ecm_y": [-0.1, -0.05],
                                                "ecm_strength": [0.8, 0.7],
                                            }
                                        }
                                    ]
                                )
                            }
                        ]
                    }
                ]
            }
        ]
    }

    rows = extract_history_rows(payload)

    assert rows == [
        {"spread": -0.2, "zscore": -2.0, "ecm_x": -0.3, "ecm_y": -0.1, "ecm_strength": 0.8},
        {"spread": 0.0, "zscore": 0.0, "ecm_x": -0.1, "ecm_y": -0.05, "ecm_strength": 0.7},
    ]


def test_extract_history_rows_accepts_har_response_content_text():
    payload = {
        "log": {
            "entries": [
                {
                    "request": {"url": "https://cryptowizards.net/irrelevant"},
                    "response": {"content": {"text": '{"status":"ok"}'}},
                },
                {
                    "request": {"url": "https://cryptowizards.net/internal/pair"},
                    "response": {
                        "content": {
                            "text": json.dumps(
                                {
                                    "result": {
                                        "spread": [-0.2, 0.0],
                                        "zscore": [-2.0, 0.0],
                                        "ecm_x": [-0.3, -0.1],
                                        "ecm_y": [-0.1, -0.05],
                                        "ecm_strength": [0.8, 0.7],
                                    }
                                }
                            )
                        }
                    },
                },
            ]
        }
    }

    rows = extract_history_rows(payload)

    assert rows == [
        {"spread": -0.2, "zscore": -2.0, "ecm_x": -0.3, "ecm_y": -0.1, "ecm_strength": 0.8},
        {"spread": 0.0, "zscore": 0.0, "ecm_x": -0.1, "ecm_y": -0.05, "ecm_strength": 0.7},
    ]


def test_pair_detail_capture_audit_finds_nested_ecm_history_candidate(tmp_path):
    payload = {
        "pair_id": "1",
        "pair": "BNB-USD-STX-USD",
        "asset_x": "BNB-USD",
        "asset_y": "STX-USD",
        "exchange": "dydx",
        "worker_messages": [
            {"direction": "to_worker", "message": {"pair": {"symbol_1": "BNB-USD"}}},
            {
                "direction": "from_worker",
                "message": {
                    "result": {
                        "data": {
                            "spread": [-0.2, -0.1, 0.0],
                            "zscore": [-2.1, -1.0, 0.0],
                            "ecm_x": [-0.3, -0.2, -0.1],
                            "ecm_y": [-0.1, -0.05, -0.01],
                            "ecm_strength": [0.8, 0.8, 0.7],
                        }
                    }
                },
            },
        ],
    }
    (tmp_path / "pair_1_capture.json").write_text(json.dumps(payload), encoding="utf-8")

    rows = pair_detail_capture_audit(tmp_path)
    ready = [row for row in rows if row["experiment_ready"] and row["ecm_history_ready"]]

    assert ready
    assert ready[0]["json_path"] == "$.worker_messages[1].message.result.data"
    assert ready[0]["candidate_type"] == "parallel_series"
    assert ready[0]["row_count"] == 3
    assert "ecm_strength" in ready[0]["columns"]


def test_pair_detail_capture_audit_reports_parsed_har_json_path(tmp_path):
    payload = {
        "pair_id": "1",
        "pair": "BNB-USD-STX-USD",
        "asset_x": "BNB-USD",
        "asset_y": "STX-USD",
        "exchange": "dydx",
        "log": {
            "entries": [
                {
                    "response": {
                        "content": {
                            "text": json.dumps(
                                {
                                    "data": {
                                        "spread": [-0.2, 0.0],
                                        "zscore": [-2.0, 0.0],
                                        "ecm_x": [-0.3, -0.1],
                                        "ecm_y": [-0.1, -0.05],
                                        "ecm_strength": [0.8, 0.7],
                                    }
                                }
                            )
                        }
                    }
                }
            ]
        },
    }
    (tmp_path / "pair_1_har.json").write_text(json.dumps(payload), encoding="utf-8")

    rows = pair_detail_capture_audit(tmp_path)
    ready = [row for row in rows if row["experiment_ready"] and row["ecm_history_ready"]]

    assert ready
    assert ready[0]["json_path"] == "$.log.entries[0].response.content.text#json.data"
    assert ready[0]["candidate_type"] == "parallel_series"


def test_pair_detail_capture_audit_marks_alias_payload_two_leg_ready(tmp_path):
    payload = {
        "pair_id": "1",
        "pair": "BNB-USD-STX-USD",
        "asset_x": "BNB-USD",
        "asset_y": "STX-USD",
        "exchange": "dydx",
        "data": {
            "spread": [-0.2, 0.0],
            "zscore": [-2.0, 0.0],
            "symbol1_prices": [600.0, 602.0],
            "symbol2_prices": [1.8, 1.85],
            "pair_beta": [0.9, 0.91],
            "symbol1_funding_bps": [2.0, 2.1],
            "symbol2_funding_bps": [3.0, 3.1],
            "ecm_x": [-0.3, -0.1],
            "ecm_y": [-0.1, -0.05],
            "ecm_strength": [0.8, 0.7],
        },
    }
    (tmp_path / "pair_1_alias_capture.json").write_text(json.dumps(payload), encoding="utf-8")

    rows = pair_detail_capture_audit(tmp_path)
    ready = [row for row in rows if row["two_leg_execution_ready"]]

    assert ready
    assert ready[0]["json_path"] == "$.data"
    assert ready[0]["missing_for_two_leg_backtest"] == ""
    assert ready[0]["beta_available"]
    assert ready[0]["funding_columns_available"]
    assert ready[0]["execution_assumption_notes"] == "hedge_ratio_default_1.0"


def test_pair_detail_capture_checklist_reports_next_missing_capture_focus(tmp_path):
    payload = {
        "pair_id": "1",
        "pair": "BNB-USD-STX-USD",
        "asset_x": "BNB-USD",
        "asset_y": "STX-USD",
        "exchange": "dydx",
        "capture_summary": {"fetches": 2, "worker_messages": 1, "wasm_extracts": 1, "storage": 1, "scripts": 1},
        "xhrs": [{"url": "/api"}],
        "resources": [{"name": "zscore_library.js"}],
        "viewItem": {
            "spread": [-0.2, 0.0],
            "zscore": [-2.0, 0.0],
            "ecm_x": [-0.3, -0.1],
            "ecm_y": [-0.1, -0.05],
            "ecm_strength": [0.8, 0.7],
        },
    }

    row = pair_detail_payload_capture_checklist(payload, tmp_path / "pair_1_capture.json")

    assert row["baseline_ready"]
    assert row["ecm_ready"]
    assert not row["two_leg_ready"]
    assert row["import_ready"]
    assert not row["research_spine_ready"]
    assert row["found_required_fields"] == "ecm_strength;ecm_x;ecm_y;spread;zscore"
    assert row["missing_required_fields"] == "price_x;price_y"
    assert row["missing_baseline_fields"] == ""
    assert row["missing_ecm_fields"] == ""
    assert row["missing_two_leg_fields"] == "price_x;price_y"
    assert row["missing_execution_assumption_fields"] == "beta;funding_x_bps;funding_y_bps;hedge_ratio"
    assert row["capture_completeness_score"] == 45.45
    assert row["capture_fetches"] == 2
    assert row["capture_xhrs"] == 1
    assert row["capture_worker_messages"] == 1
    assert row["capture_wasm_extracts"] == 1
    assert row["capture_storage_items"] == 1
    assert row["capture_scripts"] == 1
    assert row["capture_resources"] == 1
    assert row["capture_payload_sources"] == "network;worker;wasm;storage;scripts;resources"
    assert row["best_candidate_path"] == "$.viewItem"
    assert "spread=$.viewItem" in row["required_field_locations"]
    assert "ecm_strength=$.viewItem" in row["required_field_locations"]
    assert row["next_capture_focus"] == "capture_two_leg_prices:price_x;price_y"
    assert row["capture_operator_hint"] == "capture_leg_price_history:price_x;price_y"


def test_capture_summary_field_quality_does_not_count_as_research_history(tmp_path):
    payload = {
        "pair_id": "1",
        "pair": "BNB-USD-STX-USD",
        "asset_x": "BNB-USD",
        "asset_y": "STX-USD",
        "exchange": "dydx",
        "capture_summary": {
            "fetches": 1,
            "field_quality": {
                "required_field_hits": {
                    "spread": "$.fetches[0].json.spread",
                    "zscore": "$.fetches[0].json.zscore",
                    "ecm_x": "$.fetches[0].json.ecm_x",
                    "ecm_y": "$.fetches[0].json.ecm_y",
                    "ecm_strength": "$.fetches[0].json.ecm_strength",
                    "price_x": "$.fetches[0].json.price_x",
                    "price_y": "$.fetches[0].json.price_y",
                },
                "missing_baseline_fields": [],
                "missing_ecm_fields": [],
                "missing_two_leg_fields": [],
            },
        },
        "fetches": [],
    }

    row = pair_detail_payload_capture_checklist(payload, tmp_path / "pair_1_metadata_only_capture.json")

    assert row["capture_fetches"] == 1
    assert row["capture_candidate_paths"] == 0
    assert row["found_required_fields"] == ""
    assert row["missing_required_fields"] == "ecm_strength;ecm_x;ecm_y;price_x;price_y;spread;zscore"
    assert not row["baseline_ready"]
    assert not row["research_spine_ready"]
    assert row["next_capture_focus"] == "capture_baseline_history:spread;zscore"
    assert row["capture_operator_hint"] == (
        "payloads_captured_but_no_history_candidate:repeat_after_pair_refresh_or_export_network_har"
    )


def test_pair_detail_capture_checklist_marks_snapshot_without_payloads_as_not_browser_capture(tmp_path):
    payload = {
        "pair_id": "1",
        "pair": "BNB-USD-STX-USD",
        "asset_x": "BNB-USD",
        "asset_y": "STX-USD",
        "exchange": "dydx",
        "hedge_ratio": 1.36,
    }

    row = pair_detail_payload_capture_checklist(payload, tmp_path / "pair_1_dashboard_snapshot.json")

    assert row["capture_payload_sources"] == "none"
    assert row["capture_candidate_paths"] == 0
    assert row["next_capture_focus"] == "capture_baseline_history:spread;zscore"
    assert row["capture_operator_hint"] == (
        "not_a_browser_capture:paste_capture_helper_on_authenticated_pair_page,"
        "click_refresh_or_recalculate,run_await___CW_CAPTURE_STATUS__,"
        "then_await___CW_DOWNLOAD_CAPTURE__"
    )


def test_pair_detail_capture_checklist_marks_two_leg_ready_with_aliases(tmp_path):
    payload = {
        "pair_id": "1",
        "pair": "BNB-USD-STX-USD",
        "asset_x": "BNB-USD",
        "asset_y": "STX-USD",
        "exchange": "dydx",
        "data": {
            "spread": [-0.2, 0.0],
            "zscore": [-2.0, 0.0],
            "symbol1_prices": [600.0, 602.0],
            "symbol2_prices": [1.8, 1.85],
            "pair_beta": [0.9, 0.91],
            "symbol1_funding_bps": [2.0, 2.1],
            "symbol2_funding_bps": [3.0, 3.1],
            "hedge_ratio": [1.36, 1.36],
            "ecm_x": [-0.3, -0.1],
            "ecm_y": [-0.1, -0.05],
            "ecm_strength": [0.8, 0.7],
        },
    }
    (tmp_path / "pair_1_capture.json").write_text(json.dumps(payload), encoding="utf-8")

    rows = pair_detail_capture_checklist(tmp_path)

    assert rows[0]["research_spine_ready"]
    assert rows[0]["execution_assumptions_ready"]
    assert rows[0]["missing_required_fields"] == ""
    assert rows[0]["missing_execution_assumption_fields"] == ""
    assert rows[0]["capture_completeness_score"] == 100.0
    assert "price_x=$.data" in rows[0]["required_field_locations"]
    assert "price_y=$.data" in rows[0]["required_field_locations"]
    assert "hedge_ratio=$.data" in rows[0]["execution_assumption_locations"]
    assert "beta=$.data" in rows[0]["execution_assumption_locations"]
    assert rows[0]["next_capture_focus"] == "ready_for_research_spine"


def test_pair_detail_history_coverage_reports_ecm_readiness(tmp_path):
    payload = {
        "pair_id": "1",
        "pair": "BNB-USD-STX-USD",
        "asset_x": "BNB-USD",
        "asset_y": "STX-USD",
        "exchange": "dydx",
        "history": [
            {"spread": -0.2, "zscore": -2.0, "ecm_x": -0.3, "ecm_y": -0.1, "ecm_strength": 0.8},
        ],
    }
    (tmp_path / "pair_1.json").write_text(json.dumps(payload), encoding="utf-8")

    rows = pair_detail_history_coverage(tmp_path)

    assert rows[0]["experiment_ready"]
    assert rows[0]["ecm_history_ready"]
    assert not rows[0]["two_leg_execution_ready"]
    assert rows[0]["missing_for_two_leg_backtest"] == "price_x;price_y"
    assert rows[0]["execution_assumption_notes"] == "hedge_ratio_default_1.0;beta_default_1.0;funding_cost_model_default"
    assert rows[0]["missing_for_ecm_backtest"] == ""


def test_pair_detail_history_coverage_reports_two_leg_execution_readiness(tmp_path):
    payload = {
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
                "ecm_strength": 0.8,
                "funding_x_bps": 2.0,
                "funding_y_bps": 3.0,
            },
        ],
    }
    (tmp_path / "pair_1.json").write_text(json.dumps(payload), encoding="utf-8")

    rows = pair_detail_history_coverage(tmp_path)

    assert rows[0]["experiment_ready"]
    assert rows[0]["ecm_history_ready"]
    assert rows[0]["two_leg_execution_ready"]
    assert rows[0]["missing_for_two_leg_backtest"] == ""
    assert rows[0]["hedge_ratio_available"]
    assert not rows[0]["beta_available"]
    assert rows[0]["funding_columns_available"]
    assert rows[0]["execution_assumption_notes"] == "beta_default_1.0"


def test_pair_detail_dataset_derives_beta_from_leg_prices_when_missing(tmp_path):
    history = []
    for idx, (price_x, price_y, spread) in enumerate(
        [
            (100.0, 50.0, -0.4),
            (102.0, 51.0, -0.2),
            (101.0, 50.0, 0.1),
            (104.0, 52.0, 0.3),
            (103.0, 51.5, 0.0),
        ]
    ):
        history.append(
            {
                "timestamp": idx,
                "spread": spread,
                "zscore": spread * 2,
                "price_x": price_x,
                "price_y": price_y,
                "ecm_x": -0.1,
                "ecm_y": -0.05,
                "ecm_strength": 0.7,
            }
        )
    payload = {
        "pair_id": "1",
        "pair": "ETH-BTC",
        "asset_x": "ETH",
        "asset_y": "BTC",
        "exchange": "dydx",
        "history": history,
    }
    (tmp_path / "pair_1.json").write_text(json.dumps(payload), encoding="utf-8")

    datasets = datasets_from_pair_detail_snapshots(tmp_path)

    assert len(datasets) == 1
    assert "beta" in datasets[0].frame.columns
    assert set(datasets[0].frame["beta_source"]) == {"derived_from_price_returns"}


def test_datasets_from_pair_detail_snapshots_can_filter_quality_blocked_histories(tmp_path):
    good_history = [
        {
            "timestamp": idx,
            "spread": idx / 100,
            "zscore": idx / 10,
            "price_x": 100 + idx,
            "price_y": 50 + idx,
            "hedge_ratio": 1.0,
            "beta": 1.0,
        }
        for idx in range(90)
    ]
    stale_history = [
        {
            "timestamp": idx,
            "spread": idx / 100,
            "zscore": idx / 10,
            "price_x": 100 + idx,
            "price_y": 50.0,
            "hedge_ratio": 1.0,
            "beta": 1.0,
        }
        for idx in range(90)
    ]
    (tmp_path / "good.json").write_text(
        json.dumps({"pair": "AAA-USD-BBB-USD", "asset_x": "AAA-USD", "asset_y": "BBB-USD", "history": good_history}),
        encoding="utf-8",
    )
    (tmp_path / "stale.json").write_text(
        json.dumps({"pair": "CCC-USD-DDD-USD", "asset_x": "CCC-USD", "asset_y": "DDD-USD", "history": stale_history}),
        encoding="utf-8",
    )

    all_datasets = datasets_from_pair_detail_snapshots(tmp_path)
    filtered = datasets_from_pair_detail_snapshots(tmp_path, require_research_usable=True)

    assert {dataset.pair for dataset in all_datasets} == {"AAA-USD-BBB-USD", "CCC-USD-DDD-USD"}
    assert [dataset.pair for dataset in filtered] == ["AAA-USD-BBB-USD"]


def test_pair_detail_quality_report_blocks_short_or_placeholder_execution_history(tmp_path):
    history = []
    for idx in range(90):
        history.append(
            {
                "timestamp": idx,
                "spread": idx / 100,
                "zscore": idx / 10,
                "price_x": 600.0 + idx,
                "price_y": 2.0 + idx / 100,
                "hedge_ratio": 1.36,
                "beta": 1.36,
                "funding_x_bps": 0.0,
                "funding_y_bps": 0.0,
                "volume_x_usd": 1000.0,
                "volume_y_usd": 0.0,
            }
        )
    payload = {
        "pair_id": "1",
        "pair": "BNB-USD-STX-USD",
        "asset_x": "BNB-USD",
        "asset_y": "STX-USD",
        "exchange": "dydx",
        "interval": "5mins",
        "source_note": "Funding fields are zero placeholders.",
        "history": history,
    }
    (tmp_path / "pair_1.json").write_text(json.dumps(payload), encoding="utf-8")

    rows = pair_detail_quality_report(tmp_path)

    assert rows[0]["research_usable"]
    assert not rows[0]["execution_usable"]
    assert rows[0]["history_rows"] == 90
    assert rows[0]["zero_volume_y_rate"] == 1.0
    assert rows[0]["quality_blockers"] == "placeholder_execution_assumptions"


def test_pair_detail_quality_report_blocks_bad_research_history(tmp_path):
    payload = {
        "pair_id": "1",
        "pair": "BNB-USD-STX-USD",
        "asset_x": "BNB-USD",
        "asset_y": "STX-USD",
        "exchange": "dydx",
        "history": [{"spread": -0.2, "price_x": 600.0, "price_y": None}],
    }
    (tmp_path / "pair_1.json").write_text(json.dumps(payload), encoding="utf-8")

    rows = pair_detail_quality_report(tmp_path)

    assert not rows[0]["research_usable"]
    assert "history_rows_below_80" in rows[0]["quality_blockers"]
    assert "missing_required:zscore" in rows[0]["quality_blockers"]
    assert "price_y_missing_above_5pct" in rows[0]["quality_blockers"]


def test_pair_detail_quality_report_allows_spread_only_research_history(tmp_path):
    history = [{"spread": idx / 100, "zscore": idx / 10} for idx in range(90)]
    payload = {
        "pair_id": "1",
        "pair": "BNB-USD-STX-USD",
        "asset_x": "BNB-USD",
        "asset_y": "STX-USD",
        "exchange": "dydx",
        "interval": "min5",
        "source_note": "Official Crypto Wizards /v1beta/zscores history.",
        "history": history,
    }
    (tmp_path / "pair_1_cw_zscores.json").write_text(json.dumps(payload), encoding="utf-8")

    rows = pair_detail_quality_report(tmp_path)

    assert rows[0]["research_usable"]
    assert not rows[0]["execution_usable"]
    assert rows[0]["quality_blockers"] == (
        "missing_execution_assumptions:beta;funding_x_bps;funding_y_bps;hedge_ratio;"
        "missing_execution_history:price_x;price_y"
    )


def test_write_pair_detail_reports(tmp_path):
    (tmp_path / "pair_1.json").write_text(json.dumps({"text": PAIR_DETAIL_TEXT, "url": "dashboard"}), encoding="utf-8")

    paths = write_pair_detail_reports(tmp_path, tmp_path / "reports")

    assert paths["snapshots"].exists()
    assert paths["fields"].exists()
    assert paths["history_coverage"].exists()
    assert paths["quality"].exists()
    assert paths["capture_audit"].exists()
