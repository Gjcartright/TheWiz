"""Ingest authenticated Crypto Wizards pair-detail UI capture bundles.

The dashboard route number is session-local. Durable identity therefore comes
from the visible venue, timeframe, and asset inputs captured with each mode.
UI metrics remain discovery/diagnostic evidence; they are never local replay or
live-trading authority.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from quant_platform.active_pipeline import CommandResult
from quant_platform.pair_detail_ingestion import parse_pair_detail_text
from quant_platform.wizard_symbols import normalize_wizard_exchange, normalize_wizard_symbol

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "wizard_pair_detail_ui_ledger.v1"
BUNDLE_SCHEMA_VERSION = "wizard_pair_detail_ui_bundle.v1"
ROUTE_UNAVAILABLE_SCHEMA_VERSION = "wizard_pair_detail_route_unavailable.v1"
PAIR_PAGE_MODES = (
    "Static (Spread)",
    "Static (ZScoreR)",
    "Dyn (Spread)",
    "Dyn (ZScoreR)",
    "OU (Spread)",
    "OU (ZScoreR)",
    "Copula",
)
PLANNED_MODES = (*PAIR_PAGE_MODES[:-1], "OU (Optimal)", PAIR_PAGE_MODES[-1])
ORIENTATIONS = ("original", "reverse")

# These are verified current client defaults, not confirmed editable controls on
# the captured pair page. Their charging semantics must be proven before parity.
CLIENT_DEFAULT_COMMISSION_PCT = 0.10
CLIENT_DEFAULT_SLIPPAGE_PCT = 0.05


def ingest_wizard_pair_detail_ui_bundles(
    *,
    root: Path = ROOT,
    input_dir: Path | None = None,
    queue_path: Path | None = None,
) -> CommandResult:
    """Flatten pair-detail bundles and account for every planned mode/orientation."""

    input_dir = input_dir or root / "data" / "raw" / "crypto_wizards" / "dashboard_pair_details"
    active = root / "reports" / "active"
    active.mkdir(parents=True, exist_ok=True)
    queue_path = queue_path or active / "exhaustive_wizard_pair_detail_capture_queue.csv"
    queue = _read_csv(queue_path)

    bundle_paths = _bundle_paths(input_dir)
    ledger_rows: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []
    for path in bundle_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") == ROUTE_UNAVAILABLE_SCHEMA_VERSION:
            rows, checks = _route_unavailable_rows(
                payload, path=path, root=root, queue=queue
            )
        else:
            rows, checks = _bundle_rows(payload, path=path, root=root, queue=queue)
        ledger_rows.extend(rows)
        validation_rows.extend(checks)

    ledger_candidates = pd.DataFrame(ledger_rows, columns=_ledger_columns())
    ledger = _consolidate_ledger_candidates(ledger_candidates)
    validation = pd.DataFrame(validation_rows, columns=_validation_columns())
    progress = _capture_progress(queue, ledger)
    queue_run_id = _single_queue_run_id(queue)
    queue_pair_keys = {
        _text(value)
        for value in queue.get("pair_group_key", pd.Series(dtype=str)).tolist()
        if _text(value)
    }
    snapshot_ledger = ledger.loc[
        ledger["pair_group_key"].astype(str).isin(queue_pair_keys)
    ].copy() if not ledger.empty else ledger.copy()
    snapshot_validation = validation.loc[
        validation["pair_group_key"].astype(str).isin(queue_pair_keys)
    ].copy() if not validation.empty else validation.copy()

    paths = {
        "mode_ledger": active / "exhaustive_wizard_pair_detail_mode_ledger.csv",
        "coverage_validation": active / "exhaustive_wizard_pair_detail_coverage_validation.csv",
        "capture_progress": active / "exhaustive_wizard_pair_detail_capture_progress.csv",
        "summary": active / "exhaustive_wizard_pair_detail_capture_summary.md",
    }
    ledger.to_csv(paths["mode_ledger"], index=False)
    validation.to_csv(paths["coverage_validation"], index=False)
    progress.to_csv(paths["capture_progress"], index=False)

    captured = int(ledger["capture_status"].eq("CAPTURED").sum()) if not ledger.empty else 0
    pair_page_unavailable_cells = (
        int(ledger["capture_status"].eq("NOT_AVAILABLE_ON_PAIR_PAGE").sum())
        if not ledger.empty
        else 0
    )
    route_unavailable_cells = (
        int(ledger["capture_status"].eq("PAIR_ROUTE_NOT_AVAILABLE").sum())
        if not ledger.empty
        else 0
    )
    unavailable = pair_page_unavailable_cells + route_unavailable_cells
    reverse_pending = (
        int(ledger["capture_status"].eq("PENDING_REVERSE_RECALCULATION").sum())
        if not ledger.empty
        else 0
    )
    invalid_orientation = (
        int(ledger["capture_status"].eq("INVALID_ORIENTATION_CAPTURE").sum())
        if not ledger.empty
        else 0
    )
    failed_checks = int(validation["status"].eq("FAIL").sum()) if not validation.empty else 0
    accounted_pair_keys = (
        set(ledger["pair_group_key"].dropna().astype(str)) if not ledger.empty else set()
    )
    captured_pair_keys = (
        set(
            ledger.loc[ledger["capture_status"].eq("CAPTURED"), "pair_group_key"]
            .dropna()
            .astype(str)
        )
        if not ledger.empty
        else set()
    )
    route_unavailable_pair_keys = (
        set(
            ledger.loc[
                ledger["capture_status"].eq("PAIR_ROUTE_NOT_AVAILABLE"),
                "pair_group_key",
            ]
            .dropna()
            .astype(str)
        )
        if not ledger.empty
        else set()
    )
    pair_groups = len(accounted_pair_keys)
    complete_queue_pairs = (
        int(progress["capture_status"].eq("COMPLETE").sum()) if not progress.empty else 0
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "authority": "PAIR_DETAIL_UI_EVIDENCE_ONLY",
        "bundles": len(bundle_paths),
        "route_unavailable_artifacts": sum(
            json.loads(path.read_text(encoding="utf-8")).get("schema_version")
            == ROUTE_UNAVAILABLE_SCHEMA_VERSION
            for path in bundle_paths
        ),
        "candidate_cells_before_consolidation": len(ledger_candidates),
        "pair_groups_accounted": pair_groups,
        "pair_groups_captured": len(captured_pair_keys),
        "pair_groups_route_unavailable": len(route_unavailable_pair_keys),
        "active_queue_pair_groups_captured": (
            len(captured_pair_keys & queue_pair_keys)
        ),
        "active_queue_pair_groups_accounted": len(accounted_pair_keys & queue_pair_keys),
        "historical_pair_groups_retained": (
            len(accounted_pair_keys - queue_pair_keys)
        ),
        "planned_cells_accounted": len(ledger),
        "captured_cells": captured,
        "not_available_on_pair_page_cells": pair_page_unavailable_cells,
        "route_unavailable_cells": route_unavailable_cells,
        "explicitly_unavailable_cells": unavailable,
        "reverse_recalculation_pending_cells": reverse_pending,
        "invalid_orientation_capture_cells": invalid_orientation,
        "coverage_failures": failed_checks,
        "queue_pairs": len(progress),
        "queue_pairs_complete": complete_queue_pairs,
        "queue_pairs_terminally_accounted": (
            int(progress["capture_status"].isin(["COMPLETE", "ROUTE_UNAVAILABLE"]).sum())
            if not progress.empty
            else 0
        ),
        "queue_pairs_started": (
            int(
                progress["capture_status"]
                .isin(["PARTIAL", "COMPLETE", "ROUTE_UNAVAILABLE"])
                .sum()
            )
            if not progress.empty
            else 0
        ),
        "queue_run_id": queue_run_id,
        "live_trading_authorized": False,
    }
    paths["summary"].write_text(_summary_markdown(summary), encoding="utf-8")
    if queue_run_id:
        snapshot_dir = (
            root
            / "reports"
            / "snapshots"
            / "exhaustive_wizard_hyperliquid"
            / queue_run_id
        )
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        snapshot_paths = {
            "snapshot_mode_ledger": snapshot_dir / "pair_detail_mode_ledger.csv",
            "snapshot_coverage_validation": (
                snapshot_dir / "pair_detail_coverage_validation.csv"
            ),
            "snapshot_capture_progress": snapshot_dir / "pair_detail_capture_progress.csv",
            "snapshot_summary": snapshot_dir / "pair_detail_capture_summary.md",
        }
        snapshot_ledger.to_csv(snapshot_paths["snapshot_mode_ledger"], index=False)
        snapshot_validation.to_csv(
            snapshot_paths["snapshot_coverage_validation"], index=False
        )
        progress.to_csv(snapshot_paths["snapshot_capture_progress"], index=False)
        snapshot_paths["snapshot_summary"].write_text(
            _summary_markdown(summary), encoding="utf-8"
        )
        paths.update(snapshot_paths)
    return CommandResult(paths=paths, summary=summary)


def _bundle_rows(
    payload: dict[str, Any],
    *,
    path: Path,
    root: Path,
    queue: pd.DataFrame,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    captures = [item for item in payload.get("mode_captures", []) if isinstance(item, dict)]
    baseline = _baseline_capture(captures)
    inputs = _indexed(baseline.get("inputs"))
    selects = _indexed(baseline.get("selects"))
    scanner_context = payload.get("scanner_context", {})
    asset_x_raw = _text(
        scanner_context.get("asset_x_raw") or inputs.get(0, {}).get("value")
    ).upper()
    asset_y_raw = _text(
        scanner_context.get("asset_y_raw") or inputs.get(1, {}).get("value")
    ).upper()
    timeframe = _text(selects.get(0, {}).get("value")).lower()
    exchange = _extract_exchange(_text(baseline.get("bodyText")))
    normalized_x = normalize_wizard_symbol(asset_x_raw, exchange)
    normalized_y = normalize_wizard_symbol(asset_y_raw, exchange)
    asset_x = normalized_x.base_asset or _base(asset_x_raw)
    asset_y = normalized_y.base_asset or _base(asset_y_raw)
    pair_group_key = _pair_group_key(exchange, timeframe, asset_x, asset_y)
    queue_match = queue.loc[queue.get("pair_group_key", pd.Series(dtype=str)).astype(str).eq(pair_group_key)]
    pair_group_id = _first_value(queue_match, "pair_group_id")
    pair_detail_queue_id = _first_value(queue_match, "pair_detail_queue_id")
    evidence_path = _relative(path, root)
    capture_run_id = _text(payload.get("capture_run_id"))
    route = _text(payload.get("page_route") or baseline.get("url"))
    route_id = _match(r"/pair/([^?/#]+)", route)

    capture_map: dict[tuple[str, str], dict[str, Any]] = {}
    duplicates: set[tuple[str, str]] = set()
    for capture in captures:
        key = (
            _text(capture.get("exact_mode")),
            _text(capture.get("orientation") or "original").lower(),
        )
        if key in capture_map:
            duplicates.add(key)
        capture_map[key] = capture

    unavailable = {_text(value) for value in payload.get("modes_not_available_on_pair_page", [])}
    rows: list[dict[str, object]] = []
    for exact_mode in PLANNED_MODES:
        for orientation in ORIENTATIONS:
            capture = capture_map.get((exact_mode, orientation))
            orientation_evidence = _capture_orientation_evidence(
                capture,
                orientation=orientation,
                baseline_asset_x_raw=asset_x_raw,
                baseline_asset_y_raw=asset_y_raw,
                exchange=exchange,
            )
            status, blocker = _cell_status(
                capture=capture,
                exact_mode=exact_mode,
                orientation=orientation,
                unavailable=unavailable,
                orientation_blocker=_text(payload.get("orientation_blocker")),
                orientation_evidence=orientation_evidence,
            )
            row = _empty_ledger_row()
            row.update(
                {
                    "schema_version": SCHEMA_VERSION,
                    "capture_run_id": capture_run_id,
                    "pair_group_id": pair_group_id,
                    "pair_detail_queue_id": pair_detail_queue_id,
                    "pair_group_key": pair_group_key,
                    "pair_detail_session_route": route,
                    "pair_detail_session_route_id": route_id,
                    "route_identity_authority": "SESSION_EVIDENCE_ONLY",
                    "wizard_exchange": exchange,
                    "timeframe": timeframe,
                    "asset_x": asset_x,
                    "asset_y": asset_y,
                    "asset_x_raw": asset_x_raw,
                    "asset_y_raw": asset_y_raw,
                    "pair": f"{asset_x}-{asset_y}" if asset_x and asset_y else "",
                    "exact_mode": exact_mode,
                    "orientation": orientation,
                    **orientation_evidence,
                    "capture_status": status,
                    "capture_blocker": blocker,
                    "evidence_path": evidence_path,
                    "wizard_metrics_authority": "DISCOVERY_DIAGNOSTIC_ONLY",
                    "local_replay_completed": False,
                    "live_trading_authorized": False,
                }
            )
            if capture is not None:
                row.update(
                    _captured_fields(
                        capture,
                        asset_x=_text(
                            orientation_evidence.get("rendered_asset_x_raw")
                            or orientation_evidence.get("capture_asset_x_raw")
                        ),
                        asset_y=_text(
                            orientation_evidence.get("rendered_asset_y_raw")
                            or orientation_evidence.get("capture_asset_y_raw")
                        ),
                    )
                )
            rows.append(row)

    checks = _bundle_validation_rows(
        payload,
        evidence_path=evidence_path,
        capture_run_id=capture_run_id,
        pair_group_key=pair_group_key,
        capture_map=capture_map,
        duplicates=duplicates,
        queue_match_count=len(queue_match),
        asset_x_raw=asset_x_raw,
        asset_y_raw=asset_y_raw,
        exchange=exchange,
        timeframe=timeframe,
    )
    return rows, checks


def _route_unavailable_rows(
    payload: dict[str, Any],
    *,
    path: Path,
    root: Path,
    queue: pd.DataFrame,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    scanner_context = payload.get("scanner_context", {})
    exchange = normalize_wizard_exchange(
        _text(scanner_context.get("exchange")), default=""
    ) or ""
    timeframe = _text(scanner_context.get("interval")).lower()
    asset_x_raw = _text(scanner_context.get("asset_x_raw")).upper()
    asset_y_raw = _text(scanner_context.get("asset_y_raw")).upper()
    asset_x = _normalized_asset(asset_x_raw, exchange)
    asset_y = _normalized_asset(asset_y_raw, exchange)
    pair_group_key = _pair_group_key(exchange, timeframe, asset_x, asset_y)
    queue_match = queue.loc[
        queue.get("pair_group_key", pd.Series(dtype=str))
        .astype(str)
        .eq(pair_group_key)
    ]
    pair_group_id = _first_value(queue_match, "pair_group_id")
    pair_detail_queue_id = _first_value(queue_match, "pair_detail_queue_id")
    evidence_path = _relative(path, root)
    capture_run_id = _text(payload.get("capture_run_id"))
    route = _text(payload.get("route_attempt_url"))
    observed_at = _text(payload.get("observed_at"))
    ui_message = _text(payload.get("ui_message"))
    blocker = _text(payload.get("route_blocker")) or "wizard_pair_route_not_available"

    rows: list[dict[str, object]] = []
    for exact_mode in PLANNED_MODES:
        for orientation in ORIENTATIONS:
            expected_raw = (
                (asset_y_raw, asset_x_raw)
                if orientation == "reverse"
                else (asset_x_raw, asset_y_raw)
            )
            row = _empty_ledger_row()
            row.update(
                {
                    "schema_version": SCHEMA_VERSION,
                    "capture_run_id": capture_run_id,
                    "pair_group_id": pair_group_id,
                    "pair_detail_queue_id": pair_detail_queue_id,
                    "pair_group_key": pair_group_key,
                    "pair_detail_session_route": route,
                    "pair_detail_session_route_id": "",
                    "route_identity_authority": "SCANNER_CONTEXT_PLUS_UI_REJECTION",
                    "wizard_exchange": exchange,
                    "timeframe": timeframe,
                    "asset_x": asset_x,
                    "asset_y": asset_y,
                    "asset_x_raw": asset_x_raw,
                    "asset_y_raw": asset_y_raw,
                    "pair": f"{asset_x}-{asset_y}" if asset_x and asset_y else "",
                    "exact_mode": exact_mode,
                    "orientation": orientation,
                    "orientation_expected_asset_x": _normalized_asset(
                        expected_raw[0], exchange
                    ),
                    "orientation_expected_asset_y": _normalized_asset(
                        expected_raw[1], exchange
                    ),
                    "orientation_verified": False,
                    "orientation_verification_status": "PAIR_ROUTE_NOT_AVAILABLE",
                    "capture_status": "PAIR_ROUTE_NOT_AVAILABLE",
                    "capture_blocker": f"{blocker}:{ui_message}" if ui_message else blocker,
                    "capture_timestamp": observed_at,
                    "evidence_path": evidence_path,
                    "wizard_metrics_authority": "DISCOVERY_DIAGNOSTIC_ONLY",
                    "local_replay_completed": False,
                    "live_trading_authorized": False,
                }
            )
            rows.append(row)

    stable_identity = all(
        (exchange, timeframe, asset_x_raw, asset_y_raw, asset_x, asset_y, pair_group_key)
    )
    checks = [
        (
            "route_unavailable_schema",
            payload.get("schema_version") == ROUTE_UNAVAILABLE_SCHEMA_VERSION,
            payload.get("schema_version"),
        ),
        (
            "credentials_excluded",
            payload.get("no_credentials_or_browser_storage_captured") is True,
            payload.get("no_credentials_or_browser_storage_captured"),
        ),
        ("stable_identity_fields", stable_identity, pair_group_key),
        ("ui_route_rejection_present", bool(ui_message), ui_message),
        (
            "planned_cells_explicitly_unavailable",
            len(rows) == len(PLANNED_MODES) * len(ORIENTATIONS),
            f"accounted={len(rows)}",
        ),
    ]
    validation_rows = [
        {
            "schema_version": SCHEMA_VERSION,
            "capture_run_id": capture_run_id,
            "pair_group_key": pair_group_key,
            "check": check,
            "status": "PASS" if passed else "FAIL",
            "detail": _text(detail),
            "evidence_path": evidence_path,
        }
        for check, passed, detail in checks
    ]
    if len(queue_match) == 1:
        queue_status = "PASS"
        queue_detail = "queue_matches=1"
    elif queue_match.empty:
        queue_status = "WARN"
        queue_detail = "not_in_current_active_queue;historical_evidence_retained"
    else:
        queue_status = "FAIL"
        queue_detail = f"queue_matches={len(queue_match)};active_queue_identity_ambiguous"
    validation_rows.append(
        {
            "schema_version": SCHEMA_VERSION,
            "capture_run_id": capture_run_id,
            "pair_group_key": pair_group_key,
            "check": "queue_identity_unique",
            "status": queue_status,
            "detail": queue_detail,
            "evidence_path": evidence_path,
        }
    )
    return rows, validation_rows


def _captured_fields(capture: dict[str, Any], *, asset_x: str, asset_y: str) -> dict[str, object]:
    body = _text(capture.get("bodyText"))
    parsed = parse_pair_detail_text(body, source_url=_text(capture.get("url")))
    inputs = _indexed(capture.get("inputs"))
    selects = _indexed(capture.get("selects"))
    exact_mode = _text(capture.get("exact_mode"))
    settings = _mode_settings(exact_mode, inputs=inputs, selects=selects)
    stationarity = {
        _text(item.get("label")): _text(item.get("svgClass"))
        for item in capture.get("stationarity", [])
        if isinstance(item, dict)
    }
    values = asdict(parsed)
    return {
        "capture_timestamp": _text(capture.get("capturedAt")),
        "mode_value": _text(capture.get("mode_value")),
        "lookback_period_requested": _integer(inputs.get(2, {}).get("value")),
        "periods_analyzed": values.get("period"),
        "entry_long": settings["entry_long"],
        "entry_short": settings["entry_short"],
        "exit_long": settings["exit_long"],
        "exit_short": settings["exit_short"],
        "entry_long_operator": settings["entry_long_operator"],
        "entry_short_operator": settings["entry_short_operator"],
        "exit_long_operator": settings["exit_long_operator"],
        "exit_short_operator": settings["exit_short_operator"],
        "rolling_window_display": settings["rolling_window_display"],
        "rolling_window": settings["rolling_window"],
        "close_n_periods": settings["close_n_periods"],
        "stop_loss_pct": settings["stop_loss_pct"],
        "ecm_deviation_min_pct": settings["ecm_deviation_min_pct"],
        "corr_strength_min_pct": settings["corr_strength_min_pct"],
        "x_weighting": settings["x_weighting"],
        "wizard_commission_pct": CLIENT_DEFAULT_COMMISSION_PCT,
        "wizard_slippage_pct": CLIENT_DEFAULT_SLIPPAGE_PCT,
        "wizard_cost_source": "verified_current_client_default_not_pair_page_control",
        "wizard_cost_semantics_status": "UNVERIFIED_FOR_LOCAL_PARITY",
        "wizard_cost_point_in_time_ui_confirmed": False,
        "johansen_badge_class": stationarity.get("coint Jn", ""),
        "johansen_state": _badge_state("coint Jn", stationarity.get("coint Jn", "")),
        "engle_granger_badge_class": stationarity.get("coint EG", ""),
        "engle_granger_state": _badge_state("coint EG", stationarity.get("coint EG", "")),
        "hedge_ratio": values.get("hedge_ratio"),
        "hurst": values.get("hurst"),
        "half_life": values.get("half_life"),
        "pearson_returns": values.get("pearson"),
        "spearman_returns": values.get("spearman"),
        "kendall_returns": values.get("kendall"),
        "conditional_chart_value": _percent_after(body, "Conditional (chart)"),
        "copula_family": values.get("copula"),
        "copula_correlation": values.get("corr_copula"),
        "u1_given_u2": _conditional_probability(body, asset_x, asset_y),
        "u2_given_u1": _conditional_probability(body, asset_y, asset_x),
        "ou_mu": _ou_value(body, "mu"),
        "ou_alpha": _ou_value(body, "alpha"),
        "ou_beta": _ou_value(body, "beta"),
        "ou_b": _ou_value(body, "b"),
        "ou_sigma": _ou_value(body, "sigma"),
        "sharpe": values.get("sharpe"),
        "sortino": values.get("sortino"),
        "returns_total": values.get("returns_total"),
        "annual_return": values.get("annual_return"),
        "mean_period_return": values.get("mean_period_return"),
        "win_rate": values.get("win_rate"),
        "closed_trades": values.get("closed_trades"),
        "max_drawdown": values.get("drawdown"),
        "var_99": values.get("var"),
        "cvar_99": values.get("cvar"),
        "var_sim": values.get("var_sim"),
        "cvar_sim": values.get("cvar_sim"),
        "dependency_views_available": "betas;correlation;volatilities;ecm_y;ecm_x;ecm_strength",
        "conditional_views_available": "prices_1;prices_5;prices_10;returns_1;returns_5;returns_10",
        "raw_chart_data_preserved": bool(capture.get("svgs")),
    }


def _mode_settings(
    exact_mode: str,
    *,
    inputs: dict[int, dict[str, Any]],
    selects: dict[int, dict[str, Any]],
) -> dict[str, object]:
    zscore_roll = "ZScoreR" in exact_mode
    threshold_start = 5 if zscore_roll else 3
    thresholds = [_number(inputs.get(index, {}).get("value")) for index in range(threshold_start, threshold_start + 4)]
    operator_selects = [
        item
        for _, item in sorted(selects.items())
        if _text(item.get("value")) in {"Gte", "Lte", "Gt", "Lt", "Eq"}
    ]
    operators = [_text(item.get("value")) for item in operator_selects]
    while len(operators) < 4:
        operators.append("")
    after_thresholds = threshold_start + 4
    close_select = next(
        (
            item
            for item in selects.values()
            if any(_text(option.get("value")) == "9999" for option in item.get("options", []))
        ),
        {},
    )
    return {
        "entry_long": thresholds[0],
        "entry_short": thresholds[1],
        "exit_long": thresholds[2],
        "exit_short": thresholds[3],
        "entry_long_operator": operators[0],
        "entry_short_operator": operators[1],
        "exit_long_operator": operators[2],
        "exit_short_operator": operators[3],
        "rolling_window_display": _number(inputs.get(3, {}).get("value")) if zscore_roll else None,
        "rolling_window": _integer(inputs.get(4, {}).get("value")) if zscore_roll else None,
        "close_n_periods": _text(close_select.get("value")),
        "stop_loss_pct": _number(inputs.get(after_thresholds, {}).get("value")),
        "ecm_deviation_min_pct": _number(inputs.get(after_thresholds + 1, {}).get("value")),
        "corr_strength_min_pct": _number(inputs.get(after_thresholds + 2, {}).get("value")),
        "x_weighting": _number(inputs.get(after_thresholds + 3, {}).get("value")),
    }


def _cell_status(
    *,
    capture: dict[str, Any] | None,
    exact_mode: str,
    orientation: str,
    unavailable: set[str],
    orientation_blocker: str,
    orientation_evidence: dict[str, object],
) -> tuple[str, str]:
    if exact_mode in unavailable:
        return "NOT_AVAILABLE_ON_PAIR_PAGE", "mode_not_offered_by_current_pair_page_selector"
    if capture is not None:
        if not bool(orientation_evidence.get("orientation_verified")):
            return (
                "INVALID_ORIENTATION_CAPTURE",
                _text(orientation_evidence.get("orientation_verification_status"))
                or "orientation_not_verified",
            )
        return "CAPTURED", ""
    if orientation == "reverse":
        return (
            "PENDING_REVERSE_RECALCULATION",
            orientation_blocker or "reverse_orientation_requires_asset_swap_and_recalculation",
        )
    return "MISSING_CAPTURE", "expected_pair_page_mode_was_not_captured"


def _bundle_validation_rows(
    payload: dict[str, Any],
    *,
    evidence_path: str,
    capture_run_id: str,
    pair_group_key: str,
    capture_map: dict[tuple[str, str], dict[str, Any]],
    duplicates: set[tuple[str, str]],
    queue_match_count: int,
    asset_x_raw: str,
    asset_y_raw: str,
    exchange: str,
    timeframe: str,
) -> list[dict[str, object]]:
    chart = payload.get("chart_coverage", {})
    declared_orientations = [
        _text(value).lower()
        for value in payload.get("orientations_captured", ["original"])
        if _text(value).lower() in ORIENTATIONS
    ] or ["original"]
    orientation_failures = []
    for (mode, orientation), capture in capture_map.items():
        evidence = _capture_orientation_evidence(
            capture,
            orientation=orientation,
            baseline_asset_x_raw=asset_x_raw,
            baseline_asset_y_raw=asset_y_raw,
            exchange=exchange,
        )
        if not bool(evidence.get("orientation_verified")):
            orientation_failures.append(
                f"{mode}:{orientation}:{evidence.get('orientation_verification_status', '')}"
            )
    checks = [
        ("bundle_schema", payload.get("schema_version") == BUNDLE_SCHEMA_VERSION, payload.get("schema_version")),
        ("credentials_excluded", payload.get("no_credentials_or_browser_storage_captured") is True, payload.get("no_credentials_or_browser_storage_captured")),
        (
            "stable_identity_fields",
            all((asset_x_raw, asset_y_raw, exchange, timeframe)),
            pair_group_key,
        ),
        ("no_duplicate_mode_orientation", not duplicates, ";".join(f"{mode}:{orientation}" for mode, orientation in sorted(duplicates))),
        (
            "all_captured_orientations_verified",
            not orientation_failures,
            ";".join(orientation_failures) or f"verified={len(capture_map)}",
        ),
        (
            "conditional_chart_views",
            _chart_coverage_valid(payload, chart, "conditional"),
            _chart_coverage_detail(payload, chart, "conditional"),
        ),
        (
            "dependency_chart_views",
            _chart_coverage_valid(payload, chart, "dependency"),
            _chart_coverage_detail(payload, chart, "dependency"),
        ),
        (
            "backtest_chart_views",
            _chart_coverage_valid(payload, chart, "backtest"),
            _chart_coverage_detail(payload, chart, "backtest"),
        ),
        ("planned_cells_accounted", True, f"accounted={len(PLANNED_MODES) * len(ORIENTATIONS)}"),
    ]
    for orientation in declared_orientations:
        captured_count = sum((mode, orientation) in capture_map for mode in PAIR_PAGE_MODES)
        checks.append(
            (
                f"all_{orientation}_pair_page_modes",
                captured_count == len(PAIR_PAGE_MODES),
                f"captured={captured_count}/{len(PAIR_PAGE_MODES)}",
            )
        )
    rows = [
        {
            "schema_version": SCHEMA_VERSION,
            "capture_run_id": capture_run_id,
            "pair_group_key": pair_group_key,
            "check": check,
            "status": "PASS" if passed else "FAIL",
            "detail": _text(detail),
            "evidence_path": evidence_path,
        }
        for check, passed, detail in checks
    ]
    if queue_match_count == 1:
        queue_status = "PASS"
        queue_detail = "queue_matches=1"
    elif queue_match_count == 0:
        queue_status = "WARN"
        queue_detail = "not_in_current_active_queue;historical_evidence_retained"
    else:
        queue_status = "FAIL"
        queue_detail = f"queue_matches={queue_match_count};active_queue_identity_ambiguous"
    rows.append(
        {
            "schema_version": SCHEMA_VERSION,
            "capture_run_id": capture_run_id,
            "pair_group_key": pair_group_key,
            "check": "queue_identity_unique",
            "status": queue_status,
            "detail": queue_detail,
            "evidence_path": evidence_path,
        }
    )
    scanner_context = payload.get("scanner_context", {})
    scanner_interval = _text(scanner_context.get("interval")).lower()
    scanner_exchange = normalize_wizard_exchange(
        _text(scanner_context.get("exchange")), default=""
    ) or ""
    if scanner_interval:
        timeframe_status = "PASS" if scanner_interval == timeframe else "FAIL"
        timeframe_detail = f"scanner={scanner_interval};pair_page={timeframe}"
    else:
        timeframe_status = "WARN"
        timeframe_detail = "scanner_interval_not_embedded_in_legacy_bundle"
    if scanner_exchange:
        exchange_status = "PASS" if scanner_exchange == exchange else "FAIL"
        exchange_detail = f"scanner={scanner_exchange};pair_page={exchange}"
    else:
        exchange_status = "WARN"
        exchange_detail = "scanner_exchange_not_embedded_in_legacy_bundle"
    rows.extend(
        [
            {
                "schema_version": SCHEMA_VERSION,
                "capture_run_id": capture_run_id,
                "pair_group_key": pair_group_key,
                "check": "scanner_pair_page_timeframe_match",
                "status": timeframe_status,
                "detail": timeframe_detail,
                "evidence_path": evidence_path,
            },
            {
                "schema_version": SCHEMA_VERSION,
                "capture_run_id": capture_run_id,
                "pair_group_key": pair_group_key,
                "check": "scanner_pair_page_exchange_match",
                "status": exchange_status,
                "detail": exchange_detail,
                "evidence_path": evidence_path,
            },
        ]
    )
    return rows


def _consolidate_ledger_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    priorities = {
        "CAPTURED": 5,
        "NOT_AVAILABLE_ON_PAIR_PAGE": 4,
        "PAIR_ROUTE_NOT_AVAILABLE": 4,
        "INVALID_ORIENTATION_CAPTURE": 3,
        "PENDING_REVERSE_RECALCULATION": 2,
        "MISSING_CAPTURE": 1,
    }
    working = candidates.copy()
    working["_selection_rank"] = working["capture_status"].map(priorities).fillna(0)
    working["_capture_time"] = pd.to_datetime(
        working["capture_timestamp"], errors="coerce", utc=True
    )
    rows: list[dict[str, object]] = []
    group_columns = ["pair_group_key", "exact_mode", "orientation"]
    for _, group in working.groupby(group_columns, dropna=False, sort=True):
        ranked = group.sort_values(
            ["_selection_rank", "_capture_time", "evidence_path"],
            ascending=[False, False, False],
            kind="stable",
            na_position="last",
        )
        selected = ranked.iloc[0]
        record = {column: selected.get(column) for column in _ledger_columns()}
        selected_path = _text(selected.get("evidence_path"))
        all_paths = sorted(
            {
                _text(value)
                for value in group["evidence_path"].tolist()
                if _text(value)
            }
        )
        record.update(
            {
                "capture_candidate_count": len(group),
                "superseded_evidence_paths": ";".join(
                    path for path in all_paths if path != selected_path
                ),
                "capture_selection_reason": (
                    "single_capture_candidate"
                    if len(group) == 1
                    else "highest_status_rank_then_latest_capture_timestamp"
                ),
            }
        )
        rows.append(record)
    return pd.DataFrame(rows, columns=_ledger_columns())


def _capture_progress(queue: pd.DataFrame, ledger: pd.DataFrame) -> pd.DataFrame:
    columns = list(queue.columns) + [
        "captured_planned_cells",
        "accounted_planned_cells",
        "unavailable_planned_cells",
        "pending_reverse_cells",
        "pair_detail_evidence_paths",
        "capture_progress_reason",
    ]
    if queue.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    grouped = {str(key): group for key, group in ledger.groupby("pair_group_key")} if not ledger.empty else {}
    for queue_row in queue.to_dict("records"):
        group = grouped.get(_text(queue_row.get("pair_group_key")), pd.DataFrame())
        row = dict(queue_row)
        captured = int(group["capture_status"].eq("CAPTURED").sum()) if not group.empty else 0
        accounted = len(group)
        unavailable = (
            int(
                group["capture_status"]
                .isin(["NOT_AVAILABLE_ON_PAIR_PAGE", "PAIR_ROUTE_NOT_AVAILABLE"])
                .sum()
            )
            if not group.empty
            else 0
        )
        route_unavailable = (
            int(group["capture_status"].eq("PAIR_ROUTE_NOT_AVAILABLE").sum())
            if not group.empty
            else 0
        )
        reverse_pending = int(group["capture_status"].eq("PENDING_REVERSE_RECALCULATION").sum()) if not group.empty else 0
        invalid_orientation = int(group["capture_status"].eq("INVALID_ORIENTATION_CAPTURE").sum()) if not group.empty else 0
        if (
            accounted == len(PLANNED_MODES) * len(ORIENTATIONS)
            and route_unavailable == accounted
        ):
            status = "ROUTE_UNAVAILABLE"
            blocker = _first_group_text(group, "capture_blocker")
            reason = "wizard_pair_route_explicitly_unavailable;all_planned_cells_accounted"
        elif accounted == len(PLANNED_MODES) * len(ORIENTATIONS) and captured == len(PAIR_PAGE_MODES) * len(ORIENTATIONS):
            status = "COMPLETE"
            blocker = ""
            reason = "all_supported_modes_and_orientations_captured;unsupported_modes_accounted"
        elif captured:
            status = "PARTIAL"
            blocker = "pair_detail_reverse_orientation_or_mode_capture_incomplete"
            reason = (
                f"captured={captured};accounted={accounted};"
                f"reverse_pending={reverse_pending};invalid_orientation={invalid_orientation}"
            )
        else:
            status = "NOT_CAPTURED"
            blocker = _text(row.get("capture_blocker")) or "pair_detail_not_captured"
            reason = "no_pair_detail_ui_bundle_matched_queue_identity"
        row.update(
            {
                "capture_status": status,
                "capture_blocker": blocker,
                "pair_detail_route_status": (
                    "CAPTURED_SESSION_ROUTE"
                    if captured
                    else (
                        "NOT_AVAILABLE_ON_WIZARD"
                        if route_unavailable
                        else row.get("pair_detail_route_status", "MISSING")
                    )
                ),
                "captured_planned_cells": captured,
                "accounted_planned_cells": accounted,
                "unavailable_planned_cells": unavailable,
                "pending_reverse_cells": reverse_pending,
                "pair_detail_evidence_paths": ";".join(sorted(group["evidence_path"].dropna().astype(str).unique())) if not group.empty else "",
                "capture_progress_reason": reason,
                "live_trading_authorized": False,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def _ledger_columns() -> list[str]:
    return [
        "schema_version", "capture_run_id", "pair_group_id", "pair_detail_queue_id",
        "pair_group_key", "pair_detail_session_route", "pair_detail_session_route_id",
        "route_identity_authority", "wizard_exchange", "timeframe", "asset_x", "asset_y",
        "asset_x_raw", "asset_y_raw",
        "pair", "exact_mode", "mode_value", "orientation", "orientation_expected_asset_x",
        "orientation_expected_asset_y", "capture_asset_x", "capture_asset_y",
        "capture_asset_x_raw", "capture_asset_y_raw", "rendered_asset_x",
        "rendered_asset_y", "rendered_asset_x_raw", "rendered_asset_y_raw",
        "orientation_verified", "orientation_verification_status", "capture_status",
        "capture_blocker",
        "capture_timestamp", "lookback_period_requested", "periods_analyzed", "entry_long",
        "entry_short", "exit_long", "exit_short", "entry_long_operator", "entry_short_operator",
        "exit_long_operator", "exit_short_operator", "rolling_window_display", "rolling_window",
        "close_n_periods", "stop_loss_pct", "ecm_deviation_min_pct", "corr_strength_min_pct",
        "x_weighting", "wizard_commission_pct", "wizard_slippage_pct", "wizard_cost_source",
        "wizard_cost_semantics_status", "wizard_cost_point_in_time_ui_confirmed",
        "johansen_badge_class", "johansen_state", "engle_granger_badge_class",
        "engle_granger_state", "hedge_ratio", "hurst", "half_life", "pearson_returns",
        "spearman_returns", "kendall_returns", "conditional_chart_value", "copula_family",
        "copula_correlation", "u1_given_u2", "u2_given_u1", "ou_mu", "ou_alpha", "ou_beta",
        "ou_b", "ou_sigma", "sharpe", "sortino", "returns_total", "annual_return",
        "mean_period_return", "win_rate", "closed_trades", "max_drawdown", "var_99",
        "cvar_99", "var_sim", "cvar_sim", "dependency_views_available",
        "conditional_views_available", "raw_chart_data_preserved", "evidence_path",
        "capture_candidate_count", "superseded_evidence_paths", "capture_selection_reason",
        "wizard_metrics_authority", "local_replay_completed", "live_trading_authorized",
    ]


def _empty_ledger_row() -> dict[str, object]:
    return {column: None for column in _ledger_columns()}


def _validation_columns() -> list[str]:
    return ["schema_version", "capture_run_id", "pair_group_key", "check", "status", "detail", "evidence_path"]


def _badge_state(label: str, class_name: str) -> str:
    token = class_name.lower()
    if any(value in token for value in ("success", "green", "emerald", "positive")):
        return "CORRELATED_SIGNAL"
    if any(value in token for value in ("warning", "orange", "amber", "caution")):
        return "ENGLE_GRANGER_TRENDING" if label == "coint EG" else "TRENDING_WARNING"
    if any(value in token for value in ("natural", "gray", "grey")):
        return "NO_COLOR_SIGNAL"
    return "UNKNOWN"


def _chart_coverage_valid(
    payload: dict[str, Any],
    chart: dict[str, Any],
    family: str,
) -> bool:
    expected = _integer(chart.get(f"{family}_expected"))
    captured = _integer(chart.get(f"{family}_captured"))
    status = _text(payload.get(f"{family}_chart_status"))
    if expected == 0:
        return captured == 0 and status == "NOT_AVAILABLE_ON_PAIR_PAGE"
    return expected is not None and captured == expected


def _chart_coverage_detail(
    payload: dict[str, Any],
    chart: dict[str, Any],
    family: str,
) -> str:
    expected = _integer(chart.get(f"{family}_expected"))
    captured = _integer(chart.get(f"{family}_captured"))
    status = _text(payload.get(f"{family}_chart_status")) or "AVAILABLE"
    return f"captured={captured};expected={expected};status={status}"


def _baseline_capture(captures: list[dict[str, Any]]) -> dict[str, Any]:
    return next(
        (
            capture
            for capture in captures
            if _text(capture.get("orientation") or "original").lower() == "original"
        ),
        captures[0] if captures else {},
    )


def _capture_orientation_evidence(
    capture: dict[str, Any] | None,
    *,
    orientation: str,
    baseline_asset_x_raw: str,
    baseline_asset_y_raw: str,
    exchange: str,
) -> dict[str, object]:
    expected_raw = (
        (baseline_asset_y_raw, baseline_asset_x_raw)
        if orientation == "reverse"
        else (baseline_asset_x_raw, baseline_asset_y_raw)
    )
    expected = tuple(_normalized_asset(value, exchange) for value in expected_raw)
    result: dict[str, object] = {
        "orientation_expected_asset_x": expected[0],
        "orientation_expected_asset_y": expected[1],
        "capture_asset_x": "",
        "capture_asset_y": "",
        "capture_asset_x_raw": "",
        "capture_asset_y_raw": "",
        "rendered_asset_x": "",
        "rendered_asset_y": "",
        "rendered_asset_x_raw": "",
        "rendered_asset_y_raw": "",
        "orientation_verified": False,
        "orientation_verification_status": "CAPTURE_NOT_PRESENT",
    }
    if capture is None:
        return result

    inputs = _indexed(capture.get("inputs"))
    input_raw = (
        _text(inputs.get(0, {}).get("value")).upper(),
        _text(inputs.get(1, {}).get("value")).upper(),
    )
    rendered_raw = _rendered_asset_order(_text(capture.get("bodyText")))
    captured = tuple(_normalized_asset(value, exchange) for value in input_raw)
    rendered = tuple(_normalized_asset(value, exchange) for value in rendered_raw)
    result.update(
        {
            "capture_asset_x": captured[0],
            "capture_asset_y": captured[1],
            "capture_asset_x_raw": input_raw[0],
            "capture_asset_y_raw": input_raw[1],
            "rendered_asset_x": rendered[0],
            "rendered_asset_y": rendered[1],
            "rendered_asset_x_raw": rendered_raw[0],
            "rendered_asset_y_raw": rendered_raw[1],
        }
    )

    if not all(expected):
        status = "BASELINE_ASSET_IDENTITY_MISSING"
    elif not all(captured):
        status = "CAPTURE_ASSET_INPUTS_MISSING"
    elif not all(rendered):
        status = "RENDERED_ASSET_LABELS_MISSING"
    elif captured != rendered:
        status = "INPUT_RENDERED_ASSET_MISMATCH"
    elif captured != expected:
        status = "ORIENTATION_ASSET_ORDER_MISMATCH"
    else:
        status = f"VERIFIED_{orientation.upper()}"
        result["orientation_verified"] = True
    result["orientation_verification_status"] = status
    return result


def _rendered_asset_order(body: str) -> tuple[str, str]:
    assets: dict[str, str] = {}
    for line in body.splitlines():
        match = re.fullmatch(r"(.+?)\s+\(asset\s+([XY])\)", line.strip(), flags=re.IGNORECASE)
        if match:
            assets[match.group(2).upper()] = match.group(1).strip().upper()
    return assets.get("X", ""), assets.get("Y", "")


def _normalized_asset(symbol: str, exchange: str) -> str:
    normalized = normalize_wizard_symbol(symbol, exchange)
    return normalized.base_asset or _base(symbol)


def _pair_group_key(exchange: str, timeframe: str, asset_x: str, asset_y: str) -> str:
    assets = sorted(value for value in (asset_x.upper(), asset_y.upper()) if value)
    if len(assets) != 2:
        return ""
    return f"{exchange}|{timeframe}|{assets[0]}|{assets[1]}"


def _extract_exchange(body: str) -> str:
    lines = [line.strip().lower() for line in body.splitlines() if line.strip()]
    for index, line in enumerate(lines):
        if "periods analyzed" not in line or index == 0:
            continue
        candidate = normalize_wizard_exchange(lines[index - 1], default="") or ""
        if candidate:
            return candidate
    return ""


def _conditional_probability(body: str, given_asset: str, condition_asset: str) -> float | None:
    pattern = rf"{re.escape(given_asset)}\s+given\s+{re.escape(condition_asset)}\s+(-?[0-9.]+)%"
    match = re.search(pattern, body, flags=re.IGNORECASE | re.DOTALL)
    return _number(match.group(1)) / 100.0 if match else None


def _percent_after(body: str, label: str) -> float | None:
    match = re.search(rf"{re.escape(label)}\s+(-?[0-9.]+)%", body, flags=re.IGNORECASE)
    return _number(match.group(1)) / 100.0 if match else None


def _ou_value(body: str, label: str) -> float | None:
    label_pattern = {
        "mu": "(?:mu|μ)",
        "alpha": "(?:alpha|α)",
        "beta": "(?:beta|β)",
        "b": "B",
        "sigma": "(?:sigma|σ)",
    }[label]
    match = re.search(rf"(?:^|\n){label_pattern}:\s*\n?\s*(-?[0-9.]+)", body, flags=re.IGNORECASE)
    return _number(match.group(1)) if match else None


def _bundle_paths(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        return []
    paths: list[Path] = []
    for path in sorted(input_dir.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and payload.get("schema_version") in {
            BUNDLE_SCHEMA_VERSION,
            ROUTE_UNAVAILABLE_SCHEMA_VERSION,
        }:
            paths.append(path)
    return paths


def _indexed(value: Any) -> dict[int, dict[str, Any]]:
    return {
        int(item["index"]): item
        for item in value or []
        if isinstance(item, dict) and _integer(item.get("index")) is not None
    }


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _single_queue_run_id(queue: pd.DataFrame) -> str:
    if queue.empty or "exhaustive_run_id" not in queue.columns:
        return ""
    run_ids = sorted({_text(value) for value in queue["exhaustive_run_id"] if _text(value)})
    return run_ids[0] if len(run_ids) == 1 else ""


def _first_value(frame: pd.DataFrame, column: str) -> str:
    if frame.empty or column not in frame.columns:
        return ""
    return _text(frame.iloc[0][column])


def _first_group_text(frame: pd.DataFrame, column: str) -> str:
    if frame.empty or column not in frame.columns:
        return ""
    return next((_text(value) for value in frame[column] if _text(value)), "")


def _base(symbol: str) -> str:
    return symbol.upper().removesuffix("-USD")


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _match(pattern: str, value: str) -> str:
    match = re.search(pattern, value)
    return match.group(1) if match else ""


def _text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if pd.notna(number) else None


def _integer(value: Any) -> int | None:
    number = _number(value)
    return int(number) if number is not None else None


def _summary_markdown(summary: dict[str, object]) -> str:
    return "\n".join(
        [
            "# Exhaustive Wizard Pair-Detail Capture",
            "",
            f"- Authority: `{summary['authority']}`",
            f"- Bundles: {summary['bundles']}",
            f"- Route-unavailable artifacts: {summary['route_unavailable_artifacts']}",
            f"- Candidate cells before consolidation: {summary['candidate_cells_before_consolidation']}",
            f"- Pair groups accounted: {summary['pair_groups_accounted']}",
            f"- Pair groups captured: {summary['pair_groups_captured']}",
            f"- Pair groups with unavailable Wizard routes: {summary['pair_groups_route_unavailable']}",
            f"- Active queue pair groups captured: {summary['active_queue_pair_groups_captured']}",
            f"- Active queue pair groups accounted: {summary['active_queue_pair_groups_accounted']}",
            f"- Historical pair groups retained: {summary['historical_pair_groups_retained']}",
            f"- Queue run: `{summary['queue_run_id'] or 'not_bound'}`",
            f"- Planned cells accounted: {summary['planned_cells_accounted']}",
            f"- Captured cells: {summary['captured_cells']}",
            f"- Pair-page unavailable cells: {summary['not_available_on_pair_page_cells']}",
            f"- Route-unavailable cells: {summary['route_unavailable_cells']}",
            f"- Explicitly unavailable cells: {summary['explicitly_unavailable_cells']}",
            f"- Reverse recalculation pending cells: {summary['reverse_recalculation_pending_cells']}",
            f"- Invalid orientation captures: {summary['invalid_orientation_capture_cells']}",
            f"- Coverage failures: {summary['coverage_failures']}",
            f"- Queue progress: {summary['queue_pairs_started']}/{summary['queue_pairs']} started; {summary['queue_pairs_complete']} captured complete; {summary['queue_pairs_terminally_accounted']} terminally accounted",
            "- Live trading authorized: `false`",
            "",
            "Wizard metrics are discovery and diagnostic evidence. Hyperliquid replay, cost parity, leverage validation, and Testnet lifecycle proof remain separate gates.",
            "",
        ]
    )
