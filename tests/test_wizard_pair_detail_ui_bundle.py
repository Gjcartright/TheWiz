import json
from copy import deepcopy
from pathlib import Path

import pandas as pd

from quant_platform.wizard_pair_detail_ui_bundle import (
    PAIR_PAGE_MODES,
    ingest_wizard_pair_detail_ui_bundles,
)


def _auth_observation(route_kind: str = "pair_detail") -> dict[str, object]:
    pair = route_kind == "pair_detail"
    url = (
        "https://cryptowizards.net/wizards/zscore/pair/2?origin=scanner"
        if pair
        else "https://cryptowizards.net/wizards/zscore/scanner"
    )
    return {
        "schema_version": "wizard_browser_auth_observation.v1",
        "captured_at": "2026-08-07T19:00:00Z",
        "requested_url": url,
        "requested_url_source": "capture_argument",
        "final_url": url,
        "route_kind": route_kind,
        "member_navigation_targets": [
            "https://cryptowizards.net/wizards/account",
            "https://cryptowizards.net/wizards/zscore/scanner",
            "https://cryptowizards.net/wizards/zscore/trades",
        ],
        "protected_content_markers": (
            [
                "pair_mode_selector",
                "timeframe_selector",
                "ordered_asset_inputs",
                "rendered_asset_labels",
            ]
            if pair
            else [
                "scanner_filter_controls",
                "scanner_strategy_control",
                "scanner_exchange_control",
                "scanner_results_surface",
            ]
        ),
        "sign_in_form_present": False,
        "verification_form_present": False,
        "public_marketing_shell_present": False,
        "browser_storage_accessed": False,
        "no_credentials_or_browser_storage_captured": True,
    }


def _mode_capture(mode: str, value: str) -> dict[str, object]:
    zscore = "ZScoreR" in mode
    inputs = [
        {"index": 0, "value": "DOT-USD"},
        {"index": 1, "value": "ZRO-USD"},
        {"index": 2, "value": "365"},
    ]
    if zscore:
        inputs.extend(
            [
                {"index": 3, "value": "72"},
                {"index": 4, "value": "72"},
                {"index": 5, "value": "1.5"},
                {"index": 6, "value": "-1.5"},
                {"index": 7, "value": "0"},
                {"index": 8, "value": "0"},
                {"index": 9, "value": "0"},
                {"index": 10, "value": "0"},
                {"index": 11, "value": "0"},
                {"index": 12, "value": "0.34"},
            ]
        )
    else:
        inputs.extend(
            [
                {"index": 3, "value": "0.05" if mode == "Copula" else "2"},
                {"index": 4, "value": "0.95" if mode == "Copula" else "-2"},
                {"index": 5, "value": "0.5" if mode == "Copula" else "0"},
                {"index": 6, "value": "0.5" if mode == "Copula" else "0"},
                {"index": 7, "value": "0"},
                {"index": 8, "value": "0"},
                {"index": 9, "value": "0"},
                {"index": 10, "value": "0.34"},
            ]
        )
    return {
        "capturedAt": "2026-08-07T19:00:00Z",
        "url": "https://cryptowizards.net/wizards/zscore/pair/2?origin=scanner",
        "exact_mode": mode,
        "mode_value": value,
        "bodyText": """DOT-USD ZRO-USD coi
pair/2
coinbase
364 periods analyzed
DOT-USD (asset X)
ZRO-USD (asset Y)
CORRELATION (Returns)
Pearsons
22.9%
Spearmans
20.6%
Kendalls
13.8%
Conditional (chart)
25.1%
COPULA STATISTICS (Prices)
Best fit
clayton
Correlation
9.0%
DOT-USD given ZRO-USD
52.3%
ZRO-USD given DOT-USD
69.4%
sharpe:2.44
sortino:3.95
net return:14.9%
annualized return:14.9%
mean period return:0.04%
win rate:100.0%
closed trades:1
max drawdown:-3.1%
VaR (at 99%):-0.8%
CVaR (at 99%):-0.9%
""",
        "inputs": inputs,
        "selects": [
            {"index": 0, "value": "hourly", "options": []},
            {"index": 1, "value": value, "options": []},
            {"index": 4, "value": "Gte", "options": []},
            {"index": 5, "value": "Lte", "options": []},
            {"index": 6, "value": "Lte", "options": []},
            {"index": 7, "value": "Gte", "options": []},
            {
                "index": 8,
                "value": "0",
                "options": [{"value": "0"}, {"value": "9999"}],
            },
        ],
        "stationarity": [
            {"label": "coint Jn", "svgClass": "fill-success-500"},
            {"label": "coint EG", "svgClass": "fill-natural-500"},
        ],
        "svgs": [{"paths": ["M0 0"]}],
    }


def _bundle() -> dict[str, object]:
    values = ["3-1", "3-2", "1-1", "1-2", "2-1", "2-2", "1-3"]
    return {
        "schema_version": "wizard_pair_detail_ui_bundle.v2",
        "capture_run_id": "pilot",
        "capture_method": "authenticated_browser_ui",
        "no_credentials_or_browser_storage_captured": True,
        "browser_auth_observation": _auth_observation(),
        "page_route": "https://cryptowizards.net/wizards/zscore/pair/2?origin=scanner",
        "scanner_context": {"exchange": "coinbase", "interval": "hourly"},
        "modes_not_available_on_pair_page": ["OU (Optimal)"],
        "orientation_blocker": "reverse_orientation_requires_explicit_asset_swap_and_recalculation",
        "mode_captures": [_mode_capture(mode, value) for mode, value in zip(PAIR_PAGE_MODES, values)],
        "chart_coverage": {
            "conditional_expected": 6,
            "conditional_captured": 6,
            "dependency_expected": 6,
            "dependency_captured": 6,
            "backtest_expected": 21,
            "backtest_captured": 21,
        },
    }


def _reverse_capture(capture: dict[str, object], *, update_body: bool = True) -> dict[str, object]:
    reverse = deepcopy(capture)
    reverse["orientation"] = "reverse"
    reverse["inputs"][0]["value"], reverse["inputs"][1]["value"] = (
        reverse["inputs"][1]["value"],
        reverse["inputs"][0]["value"],
    )
    if update_body:
        reverse["bodyText"] = (
            reverse["bodyText"]
            .replace("DOT-USD", "__ASSET_X__")
            .replace("ZRO-USD", "DOT-USD")
            .replace("__ASSET_X__", "ZRO-USD")
        )
    return reverse


def _write_queue(root: Path) -> None:
    active = root / "reports" / "active"
    active.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "exhaustive_run_id": "ewhl-test",
                "pair_group_key": "coinbase|hourly|DOT|ZRO",
                "pair_group_id": "pair-1",
                "pair_detail_queue_id": "queue-1",
                "capture_status": "NOT_CAPTURED",
                "capture_blocker": "route_missing",
                "pair_detail_route_status": "MISSING",
                "live_trading_authorized": False,
            },
            {
                "exhaustive_run_id": "ewhl-test",
                "pair_group_key": "coinbase|hourly|BTC|ETH",
                "pair_group_id": "pair-2",
                "pair_detail_queue_id": "queue-2",
                "capture_status": "NOT_CAPTURED",
                "capture_blocker": "route_missing",
                "pair_detail_route_status": "MISSING",
                "live_trading_authorized": False,
            },
        ]
    ).to_csv(active / "exhaustive_wizard_pair_detail_capture_queue.csv", index=False)


def test_ingest_accounts_for_all_modes_and_orientations(tmp_path):
    _write_queue(tmp_path)
    raw = tmp_path / "data" / "raw" / "crypto_wizards" / "dashboard_pair_details" / "2026-08-07"
    raw.mkdir(parents=True)
    (raw / "pilot.json").write_text(json.dumps(_bundle()), encoding="utf-8")

    result = ingest_wizard_pair_detail_ui_bundles(root=tmp_path)
    ledger = pd.read_csv(result.paths["mode_ledger"])
    progress = pd.read_csv(result.paths["capture_progress"])

    assert len(ledger) == 16
    assert ledger["capture_status"].eq("CAPTURED").sum() == 7
    assert ledger["capture_status"].eq("NOT_AVAILABLE_ON_PAIR_PAGE").sum() == 2
    assert ledger["capture_status"].eq("PENDING_REVERSE_RECALCULATION").sum() == 7
    assert set(ledger["orientation"]) == {"original", "reverse"}
    captured = ledger[ledger["capture_status"].eq("CAPTURED")]
    assert captured["pair_group_key"].eq("coinbase|hourly|DOT|ZRO").all()
    assert captured["johansen_state"].eq("CORRELATED_SIGNAL").all()
    assert captured["johansen_cointegrated"].all()
    assert captured["engle_granger_state"].eq("NO_COLOR_SIGNAL").all()
    assert not captured["engle_granger_cointegrated"].any()
    assert not captured["engle_granger_includes_trend"].any()
    assert captured["metric_accounting_state"].eq("CLOSED_TRADES_PRESENT").all()
    assert captured["u1_given_u2"].eq(0.523).all()
    assert captured["wizard_cost_semantics_status"].eq("UNVERIFIED_FOR_LOCAL_PARITY").all()
    assert not captured["live_trading_authorized"].any()
    assert progress.loc[progress["pair_group_key"].eq("coinbase|hourly|DOT|ZRO"), "capture_status"].item() == "PARTIAL"
    assert progress.loc[progress["pair_group_key"].eq("coinbase|hourly|BTC|ETH"), "capture_status"].item() == "NOT_CAPTURED"
    assert result.paths["snapshot_mode_ledger"].exists()
    assert len(pd.read_csv(result.paths["snapshot_mode_ledger"])) == 16


def test_legacy_self_asserted_auth_capture_is_retained_but_not_counted(tmp_path):
    _write_queue(tmp_path)
    raw = tmp_path / "data" / "raw" / "crypto_wizards" / "dashboard_pair_details"
    raw.mkdir(parents=True)
    bundle = _bundle()
    bundle["schema_version"] = "wizard_pair_detail_ui_bundle.v1"
    bundle.pop("browser_auth_observation")
    (raw / "legacy.json").write_text(json.dumps(bundle), encoding="utf-8")

    result = ingest_wizard_pair_detail_ui_bundles(root=tmp_path)
    ledger = pd.read_csv(result.paths["mode_ledger"])
    progress = pd.read_csv(result.paths["capture_progress"])

    assert ledger["capture_status"].eq("AUTH_UNPROVEN_CAPTURE").sum() == 7
    assert not ledger["authenticated_capture_authority"].fillna(False).any()
    assert progress.loc[
        progress["pair_group_key"].eq("coinbase|hourly|DOT|ZRO"), "capture_status"
    ].item() == "AUTHENTICATION_UNPROVEN"
    assert result.summary["captured_cells"] == 0


def test_route_unavailable_claim_without_auth_proof_is_not_terminal(tmp_path):
    _write_queue(tmp_path)
    raw = tmp_path / "captures"
    raw.mkdir()
    payload = {
        "schema_version": "wizard_pair_detail_route_unavailable.v1",
        "capture_run_id": "unproven-route",
        "capture_method": "authenticated_browser_ui",
        "no_credentials_or_browser_storage_captured": True,
        "observed_at": "2026-08-08T01:45:45Z",
        "route_attempt_url": "https://cryptowizards.net/wizards/zscore/scanner",
        "route_status": "PAIR_ROUTE_NOT_AVAILABLE",
        "route_blocker": "wizard_custom_analysis_no_data",
        "ui_message": "No data found",
        "scanner_context": {
            "exchange": "coinbase",
            "interval": "hourly",
            "asset_x_raw": "DOT-USD",
            "asset_y_raw": "ZRO-USD",
        },
    }
    (raw / "unproven.json").write_text(json.dumps(payload), encoding="utf-8")

    result = ingest_wizard_pair_detail_ui_bundles(root=tmp_path, input_dir=raw)
    ledger = pd.read_csv(result.paths["mode_ledger"])
    progress = pd.read_csv(result.paths["capture_progress"])

    assert ledger["capture_status"].eq("AUTH_UNPROVEN_ROUTE_UNAVAILABLE_CLAIM").all()
    assert progress.loc[
        progress["pair_group_key"].eq("coinbase|hourly|DOT|ZRO"), "capture_status"
    ].item() == "AUTHENTICATION_UNPROVEN"
    assert result.summary["pair_groups_route_unavailable"] == 0


def test_pair_bundle_blocks_ambiguous_performance_and_visible_garch_failure(tmp_path):
    _write_queue(tmp_path)
    raw = tmp_path / "captures"
    raw.mkdir()
    bundle = _bundle()
    for capture in bundle["mode_captures"]:
        capture["bodyText"] = (
            capture["bodyText"]
            .replace("closed trades:1", "closed trades:0")
            + "\nMissing GARCH Data\n"
        )
    (raw / "quality_blocked.json").write_text(json.dumps(bundle), encoding="utf-8")

    result = ingest_wizard_pair_detail_ui_bundles(root=tmp_path, input_dir=raw)
    ledger = pd.read_csv(result.paths["mode_ledger"], keep_default_na=False)
    captured = ledger[ledger["capture_status"].eq("CAPTURED")]

    assert captured["metric_accounting_state"].eq(
        "AMBIGUOUS_OPEN_OR_MARK_TO_MARKET"
    ).all()
    assert captured["metric_accounting_blocker"].eq(
        "nonzero_performance_with_zero_closed_trades"
    ).all()
    assert captured["pair_page_data_quality_state"].eq("MISSING_GARCH_DATA").all()
    assert captured["pair_page_data_quality_blocker"].eq("missing_garch_data").all()


def test_missing_original_mode_is_reported_not_dropped(tmp_path):
    _write_queue(tmp_path)
    raw = tmp_path / "captures"
    raw.mkdir()
    bundle = _bundle()
    bundle["mode_captures"] = bundle["mode_captures"][:-1]
    (raw / "missing.json").write_text(json.dumps(bundle), encoding="utf-8")

    result = ingest_wizard_pair_detail_ui_bundles(root=tmp_path, input_dir=raw)
    ledger = pd.read_csv(result.paths["mode_ledger"])
    validation = pd.read_csv(result.paths["coverage_validation"])

    copula_original = ledger[
        ledger["exact_mode"].eq("Copula") & ledger["orientation"].eq("original")
    ].iloc[0]
    assert copula_original["capture_status"] == "MISSING_CAPTURE"
    assert validation.loc[validation["check"].eq("all_original_pair_page_modes"), "status"].item() == "FAIL"


def test_no_bundle_still_preserves_every_queue_row(tmp_path):
    _write_queue(tmp_path)
    result = ingest_wizard_pair_detail_ui_bundles(root=tmp_path, input_dir=tmp_path / "missing")

    ledger = pd.read_csv(result.paths["mode_ledger"])
    progress = pd.read_csv(result.paths["capture_progress"])

    assert ledger.empty
    assert len(progress) == 2
    assert progress["capture_status"].eq("NOT_CAPTURED").all()


def test_explicit_wizard_route_unavailable_accounts_for_every_planned_cell(tmp_path):
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "exhaustive_run_id": "ewhl-test",
                "pair_group_key": "bybit|hourly|DRAM|ETHUSDT",
                "pair_group_id": "pair-dated",
                "pair_detail_queue_id": "queue-dated",
                "capture_status": "NOT_CAPTURED",
                "capture_blocker": "route_missing",
                "pair_detail_route_status": "MISSING",
                "live_trading_authorized": False,
            }
        ]
    ).to_csv(active / "exhaustive_wizard_pair_detail_capture_queue.csv", index=False)
    raw = tmp_path / "captures"
    raw.mkdir()
    unavailable = {
        "schema_version": "wizard_pair_detail_route_unavailable.v1",
        "capture_run_id": "route-unavailable-test",
        "capture_method": "authenticated_browser_ui",
        "no_credentials_or_browser_storage_captured": True,
        "browser_auth_observation": _auth_observation("scanner"),
        "observed_at": "2026-08-08T01:45:45Z",
        "route_attempt_url": "https://cryptowizards.net/wizards/zscore/scanner",
        "route_status": "PAIR_ROUTE_NOT_AVAILABLE",
        "route_blocker": "wizard_custom_analysis_no_data",
        "ui_message": "No data found for 'ETHUSDT-07AUG26'",
        "scanner_context": {
            "exchange": "bybit",
            "interval": "hourly",
            "asset_x_raw": "DRAMUSDT",
            "asset_y_raw": "ETHUSDT-07AUG26",
        },
    }
    evidence = raw / "dated_route_unavailable.json"
    evidence.write_text(json.dumps(unavailable), encoding="utf-8")

    result = ingest_wizard_pair_detail_ui_bundles(root=tmp_path, input_dir=raw)
    ledger = pd.read_csv(result.paths["mode_ledger"])
    progress = pd.read_csv(result.paths["capture_progress"])
    validation = pd.read_csv(result.paths["coverage_validation"])

    assert len(ledger) == 16
    assert ledger["capture_status"].eq("PAIR_ROUTE_NOT_AVAILABLE").all()
    assert ledger["pair_group_key"].eq("bybit|hourly|DRAM|ETHUSDT").all()
    assert ledger["capture_blocker"].str.contains("No data found").all()
    assert ledger["evidence_path"].eq("captures/dated_route_unavailable.json").all()
    assert not ledger["live_trading_authorized"].any()
    assert progress["capture_status"].item() == "ROUTE_UNAVAILABLE"
    assert progress["pair_detail_route_status"].item() == "NOT_AVAILABLE_ON_WIZARD"
    assert progress["accounted_planned_cells"].item() == 16
    assert progress["unavailable_planned_cells"].item() == 16
    assert validation["status"].eq("PASS").all()
    assert result.summary["pair_groups_captured"] == 0
    assert result.summary["pair_groups_route_unavailable"] == 1
    assert result.summary["route_unavailable_cells"] == 16
    assert result.summary["queue_pairs_complete"] == 0
    assert result.summary["queue_pairs_terminally_accounted"] == 1
    assert result.summary["coverage_failures"] == 0


def test_scanner_timeframe_mismatch_fails_validation(tmp_path):
    _write_queue(tmp_path)
    raw = tmp_path / "captures"
    raw.mkdir()
    bundle = _bundle()
    bundle["scanner_context"] = {"exchange": "coinbase", "interval": "daily"}
    (raw / "mismatch.json").write_text(json.dumps(bundle), encoding="utf-8")

    result = ingest_wizard_pair_detail_ui_bundles(root=tmp_path, input_dir=raw)
    validation = pd.read_csv(result.paths["coverage_validation"])

    check = validation[validation["check"].eq("scanner_pair_page_timeframe_match")].iloc[0]
    assert check["status"] == "FAIL"
    assert check["detail"] == "scanner=daily;pair_page=hourly"


def test_unavailable_conditional_chart_is_accounted_not_failed(tmp_path):
    _write_queue(tmp_path)
    raw = tmp_path / "captures"
    raw.mkdir()
    bundle = _bundle()
    bundle["conditional_chart_status"] = "NOT_AVAILABLE_ON_PAIR_PAGE"
    bundle["chart_coverage"]["conditional_expected"] = 0
    bundle["chart_coverage"]["conditional_captured"] = 0
    (raw / "no_conditional.json").write_text(json.dumps(bundle), encoding="utf-8")

    result = ingest_wizard_pair_detail_ui_bundles(root=tmp_path, input_dir=raw)
    validation = pd.read_csv(result.paths["coverage_validation"])

    check = validation[validation["check"].eq("conditional_chart_views")].iloc[0]
    assert check["status"] == "PASS"
    assert "status=NOT_AVAILABLE_ON_PAIR_PAGE" in check["detail"]


def test_usdt_pair_inputs_normalize_to_queue_base_assets(tmp_path):
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "pair_group_key": "binance|hourly|BTC|MOVE",
                "pair_group_id": "pair-1",
                "pair_detail_queue_id": "queue-1",
                "capture_status": "NOT_CAPTURED",
                "capture_blocker": "route_missing",
                "pair_detail_route_status": "MISSING",
                "live_trading_authorized": False,
            }
        ]
    ).to_csv(active / "exhaustive_wizard_pair_detail_capture_queue.csv", index=False)
    bundle = _bundle()
    bundle["scanner_context"] = {"exchange": "binance", "interval": "hourly"}
    for capture in bundle["mode_captures"]:
        capture["inputs"][0]["value"] = "BTCUSDT"
        capture["inputs"][1]["value"] = "MOVEUSDT"
        capture["bodyText"] = capture["bodyText"].replace("DOT-USD", "BTCUSDT")
        capture["bodyText"] = capture["bodyText"].replace("ZRO-USD", "MOVEUSDT")
        capture["bodyText"] = capture["bodyText"].replace("coinbase", "binance")
    raw = tmp_path / "captures"
    raw.mkdir()
    (raw / "usdt.json").write_text(json.dumps(bundle), encoding="utf-8")

    result = ingest_wizard_pair_detail_ui_bundles(root=tmp_path, input_dir=raw)
    ledger = pd.read_csv(result.paths["mode_ledger"])

    assert ledger["pair_group_key"].eq("binance|hourly|BTC|MOVE").all()
    assert ledger["asset_x"].eq("BTC").all()
    assert ledger["asset_y"].eq("MOVE").all()
    assert ledger["asset_x_raw"].eq("BTCUSDT").all()
    assert ledger["asset_y_raw"].eq("MOVEUSDT").all()


def test_verified_reverse_orientation_completes_pair_capture(tmp_path):
    _write_queue(tmp_path)
    raw = tmp_path / "captures"
    raw.mkdir()
    bundle = _bundle()
    bundle["mode_captures"].extend(
        _reverse_capture(capture) for capture in list(bundle["mode_captures"])
    )
    (raw / "with_reverse.json").write_text(json.dumps(bundle), encoding="utf-8")

    result = ingest_wizard_pair_detail_ui_bundles(root=tmp_path, input_dir=raw)
    ledger = pd.read_csv(result.paths["mode_ledger"])
    progress = pd.read_csv(result.paths["capture_progress"])
    validation = pd.read_csv(result.paths["coverage_validation"])

    assert ledger["capture_status"].eq("CAPTURED").sum() == 14
    assert ledger["capture_status"].eq("NOT_AVAILABLE_ON_PAIR_PAGE").sum() == 2
    reverse = ledger[
        ledger["orientation"].eq("reverse") & ledger["capture_status"].eq("CAPTURED")
    ]
    assert reverse["orientation_verified"].all()
    assert reverse["orientation_verification_status"].eq("VERIFIED_REVERSE").all()
    assert reverse["capture_asset_x"].eq("ZRO").all()
    assert reverse["capture_asset_y"].eq("DOT").all()
    assert reverse["rendered_asset_x"].eq("ZRO").all()
    assert reverse["rendered_asset_y"].eq("DOT").all()
    assert reverse["u1_given_u2"].eq(0.523).all()
    assert progress.loc[
        progress["pair_group_key"].eq("coinbase|hourly|DOT|ZRO"), "capture_status"
    ].item() == "COMPLETE"
    assert validation.loc[
        validation["check"].eq("all_captured_orientations_verified"), "status"
    ].item() == "PASS"


def test_dated_contract_orientation_validation_uses_raw_symbols(tmp_path):
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "exhaustive_run_id": "ewhl-test",
                "pair_group_key": "bybit|hourly|BTCUSDT|GUN",
                "pair_group_id": "pair-dated",
                "pair_detail_queue_id": "queue-dated",
                "capture_status": "NOT_CAPTURED",
                "capture_blocker": "route_missing",
                "pair_detail_route_status": "MISSING",
                "live_trading_authorized": False,
            }
        ]
    ).to_csv(active / "exhaustive_wizard_pair_detail_capture_queue.csv", index=False)
    original = _bundle()
    original["scanner_context"] = {
        "exchange": "bybit",
        "interval": "hourly",
        "asset_x_raw": "BTCUSDT-25SEP26",
        "asset_y_raw": "GUNUSDT",
    }
    original["capture_run_id"] = "dated-original"
    for capture in original["mode_captures"]:
        capture["inputs"][0]["value"] = "BTCUSDT-25SEP26"
        capture["inputs"][1]["value"] = "GUNUSDT"
        capture["bodyText"] = (
            capture["bodyText"]
            .replace("DOT-USD", "BTCUSDT-25SEP26")
            .replace("ZRO-USD", "GUNUSDT")
            .replace("coinbase", "bybit")
        )
    reverse = deepcopy(original)
    reverse["capture_run_id"] = "dated-reverse"
    reverse["page_route"] = "https://cryptowizards.net/wizards/zscore/pair/275"
    reverse["orientations_captured"] = ["reverse"]
    for capture in reverse["mode_captures"]:
        capture["orientation"] = "reverse"
        capture["inputs"][0]["value"] = "GUNUSDT"
        capture["inputs"][1]["value"] = "BTCUSDT-25SEP26"
        capture["bodyText"] = (
            capture["bodyText"]
            .replace("BTCUSDT-25SEP26", "__DATED_BTC__")
            .replace("GUNUSDT", "BTCUSDT-25SEP26")
            .replace("__DATED_BTC__", "GUNUSDT")
        )
    raw = tmp_path / "captures"
    raw.mkdir()
    (raw / "original.json").write_text(json.dumps(original), encoding="utf-8")
    (raw / "reverse.json").write_text(json.dumps(reverse), encoding="utf-8")

    result = ingest_wizard_pair_detail_ui_bundles(root=tmp_path, input_dir=raw)
    ledger = pd.read_csv(result.paths["mode_ledger"])
    validation = pd.read_csv(result.paths["coverage_validation"])

    assert ledger["capture_status"].eq("CAPTURED").sum() == 14
    assert ledger["pair_group_key"].eq("bybit|hourly|BTCUSDT|GUN").all()
    orientation_checks = validation[
        validation["check"].eq("all_captured_orientations_verified")
    ]
    assert len(orientation_checks) == 2
    assert orientation_checks["status"].eq("PASS").all()
    assert result.summary["coverage_failures"] == 0


def test_input_only_swap_is_retained_as_invalid_orientation_capture(tmp_path):
    _write_queue(tmp_path)
    raw = tmp_path / "captures"
    raw.mkdir()
    bundle = _bundle()
    bundle["mode_captures"].extend(
        _reverse_capture(capture, update_body=False)
        for capture in list(bundle["mode_captures"])
    )
    (raw / "input_only_swap.json").write_text(json.dumps(bundle), encoding="utf-8")

    result = ingest_wizard_pair_detail_ui_bundles(root=tmp_path, input_dir=raw)
    ledger = pd.read_csv(result.paths["mode_ledger"])
    validation = pd.read_csv(result.paths["coverage_validation"])
    reverse = ledger[ledger["orientation"].eq("reverse")]

    assert reverse["capture_status"].eq("INVALID_ORIENTATION_CAPTURE").sum() == 7
    assert reverse["capture_blocker"].eq("INPUT_RENDERED_ASSET_MISMATCH").sum() == 7
    assert not reverse["orientation_verified"].fillna(False).any()
    check = validation[validation["check"].eq("all_captured_orientations_verified")].iloc[0]
    assert check["status"] == "FAIL"
    assert "INPUT_RENDERED_ASSET_MISMATCH" in check["detail"]
    assert result.summary["invalid_orientation_capture_cells"] == 7


def test_separate_original_and_reverse_route_bundles_consolidate_without_drops(tmp_path):
    _write_queue(tmp_path)
    raw = tmp_path / "captures"
    raw.mkdir()
    original = _bundle()
    reverse = _bundle()
    reverse["capture_run_id"] = "reverse-route"
    reverse["page_route"] = "https://cryptowizards.net/wizards/zscore/pair/4"
    reverse["scanner_context"].update(
        {"asset_x_raw": "DOT-USD", "asset_y_raw": "ZRO-USD"}
    )
    reverse["orientations_captured"] = ["reverse"]
    reverse["mode_captures"] = [
        _reverse_capture(capture) for capture in reverse["mode_captures"]
    ]
    (raw / "original.json").write_text(json.dumps(original), encoding="utf-8")
    (raw / "reverse.json").write_text(json.dumps(reverse), encoding="utf-8")

    result = ingest_wizard_pair_detail_ui_bundles(root=tmp_path, input_dir=raw)
    ledger = pd.read_csv(result.paths["mode_ledger"])
    progress = pd.read_csv(result.paths["capture_progress"])
    validation = pd.read_csv(result.paths["coverage_validation"])

    assert len(ledger) == 16
    assert ledger["capture_candidate_count"].eq(2).all()
    assert ledger["capture_status"].eq("CAPTURED").sum() == 14
    assert ledger["capture_status"].eq("NOT_AVAILABLE_ON_PAIR_PAGE").sum() == 2
    assert ledger.loc[
        ledger["orientation"].eq("original")
        & ledger["capture_status"].eq("CAPTURED"),
        "capture_run_id",
    ].eq("pilot").all()
    assert ledger.loc[
        ledger["orientation"].eq("reverse")
        & ledger["capture_status"].eq("CAPTURED"),
        "capture_run_id",
    ].eq("reverse-route").all()
    assert ledger["superseded_evidence_paths"].astype(str).str.len().gt(0).all()
    assert progress.loc[
        progress["pair_group_key"].eq("coinbase|hourly|DOT|ZRO"), "capture_status"
    ].item() == "COMPLETE"
    assert validation.loc[
        validation["check"].eq("all_original_pair_page_modes"), "status"
    ].eq("PASS").all()
    assert validation.loc[
        validation["check"].eq("all_reverse_pair_page_modes"), "status"
    ].eq("PASS").all()
    assert result.summary["candidate_cells_before_consolidation"] == 32


def test_bundle_missing_from_current_queue_is_historical_warning_not_failure(tmp_path):
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "exhaustive_run_id": "current-run",
                "pair_group_key": "coinbase|hourly|BTC|ETH",
                "pair_group_id": "pair-2",
                "pair_detail_queue_id": "queue-2",
                "capture_status": "NOT_CAPTURED",
                "capture_blocker": "route_missing",
                "pair_detail_route_status": "MISSING",
                "live_trading_authorized": False,
            }
        ]
    ).to_csv(active / "exhaustive_wizard_pair_detail_capture_queue.csv", index=False)
    raw = tmp_path / "captures"
    raw.mkdir()
    (raw / "historical.json").write_text(json.dumps(_bundle()), encoding="utf-8")

    result = ingest_wizard_pair_detail_ui_bundles(root=tmp_path, input_dir=raw)
    validation = pd.read_csv(result.paths["coverage_validation"])
    snapshot_ledger = pd.read_csv(result.paths["snapshot_mode_ledger"])
    queue_check = validation[validation["check"].eq("queue_identity_unique")].iloc[0]

    assert queue_check["status"] == "WARN"
    assert "historical_evidence_retained" in queue_check["detail"]
    assert result.summary["coverage_failures"] == 0
    assert result.summary["active_queue_pair_groups_captured"] == 0
    assert result.summary["historical_pair_groups_retained"] == 1
    assert snapshot_ledger.empty
