from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable
import json
import re

import pandas as pd

from quant_platform.derived_features import add_derived_beta_from_prices
from quant_platform.experiments import PairDataset
from quant_platform.fixture_ingestion import snake_case


ECM_FIELD_SOURCE = {
    "ecm_x": "pair_detail:dependency_chart_option",
    "ecm_y": "pair_detail:dependency_chart_option",
    "ecm_strength": "pair_detail:dependency_chart_option",
}

HISTORY_ALIASES = {
    "spread": ("spread", "spreads"),
    "zscore": ("zscore", "zscores", "zscore_last", "zscore_roll"),
    "rolling_zscore": ("rolling_zscore", "zscore_roll", "zscore_rolls"),
    "price_x": (
        "price_x",
        "prices_x",
        "x_price",
        "x_prices",
        "asset_x_price",
        "asset_x_prices",
        "symbol_1_price",
        "symbol_1_prices",
        "symbol1_price",
        "symbol1_prices",
        "symbol_1_closes",
        "symbol1_closes",
        "series_1_closes",
        "series1_closes",
        "close_x",
        "closes_x",
    ),
    "price_y": (
        "price_y",
        "prices_y",
        "y_price",
        "y_prices",
        "asset_y_price",
        "asset_y_prices",
        "symbol_2_price",
        "symbol_2_prices",
        "symbol2_price",
        "symbol2_prices",
        "symbol_2_closes",
        "symbol2_closes",
        "series_2_closes",
        "series2_closes",
        "close_y",
        "closes_y",
    ),
    "hedge_ratio": ("hedge_ratio", "hedge_ratios"),
    "beta": ("beta", "betas", "pair_beta", "pair_betas", "beta_pair"),
    "funding_x_bps": (
        "funding_x_bps",
        "funding_x",
        "funding_rate_x",
        "asset_x_funding_bps",
        "symbol_1_funding_bps",
        "symbol1_funding_bps",
        "symbol_1_funding_rate",
        "symbol1_funding_rate",
    ),
    "funding_y_bps": (
        "funding_y_bps",
        "funding_y",
        "funding_rate_y",
        "asset_y_funding_bps",
        "symbol_2_funding_bps",
        "symbol2_funding_bps",
        "symbol_2_funding_rate",
        "symbol2_funding_rate",
    ),
    "ecm_x": ("ecm_x", "ecm_xs"),
    "ecm_y": ("ecm_y", "ecm_ys"),
    "ecm_strength": ("ecm_strength", "ecm_strengths"),
    "u1_given_u2": ("u1_given_u2",),
    "u2_given_u1": ("u2_given_u1",),
}

HISTORY_ALIAS_TO_CANONICAL = {
    snake_case(alias): canonical for canonical, aliases in HISTORY_ALIASES.items() for alias in aliases
}

AUDIT_RESEARCH_FIELDS = {
    "spread",
    "spreads",
    "zscore",
    "zscores",
    "zscore_last",
    "zscore_roll",
    "zscore_rolls",
    "rolling_zscore",
    "price_x",
    "prices_x",
    "price_y",
    "prices_y",
    "symbol_1_prices",
    "symbol_2_prices",
    "series_1_closes",
    "series_2_closes",
    "ecm_x",
    "ecm_xs",
    "ecm_y",
    "ecm_ys",
    "ecm_strength",
    "ecm_strengths",
    "u1_given_u2",
    "u2_given_u1",
    "funding_x_bps",
    "funding_y_bps",
    "hedge_ratio",
    "beta",
}

PAIR_DETAIL_CAPTURE_AUDIT_COLUMNS = [
    "path",
    "pair",
    "json_path",
    "candidate_type",
    "row_count",
    "columns",
    "experiment_ready",
    "missing_for_baseline_backtest",
    "ecm_history_ready",
    "missing_for_ecm_backtest",
    "two_leg_execution_ready",
    "missing_for_two_leg_backtest",
    "hedge_ratio_available",
    "beta_available",
    "funding_columns_available",
    "execution_assumption_notes",
]

PAIR_DETAIL_CAPTURE_CHECKLIST_COLUMNS = [
    "path",
    "pair",
    "history_rows",
    "capture_candidate_paths",
    "best_candidate_path",
    "best_candidate_type",
    "best_candidate_rows",
    "found_required_fields",
    "required_field_locations",
    "execution_assumption_locations",
    "missing_required_fields",
    "missing_baseline_fields",
    "missing_ecm_fields",
    "missing_two_leg_fields",
    "missing_execution_assumption_fields",
    "capture_completeness_score",
    "capture_fetches",
    "capture_xhrs",
    "capture_worker_messages",
    "capture_wasm_extracts",
    "capture_har_entries",
    "capture_har_response_texts",
    "capture_har_dydx_candle_requests",
    "capture_storage_items",
    "capture_indexeddb_databases",
    "capture_scripts",
    "capture_resources",
    "capture_payload_sources",
    "baseline_ready",
    "ecm_ready",
    "two_leg_ready",
    "execution_assumptions_ready",
    "import_ready",
    "research_spine_ready",
    "next_capture_focus",
    "capture_operator_hint",
]

PAIR_DETAIL_QUALITY_COLUMNS = [
    "path",
    "pair",
    "interval",
    "history_rows",
    "price_rows",
    "missing_required_fields",
    "missing_price_x_rate",
    "missing_price_y_rate",
    "stale_price_x_rate",
    "stale_price_y_rate",
    "zero_volume_x_rate",
    "zero_volume_y_rate",
    "nonfinite_spread_rate",
    "nonfinite_zscore_rate",
    "research_usable",
    "execution_usable",
    "quality_blockers",
    "source_note",
]

BASELINE_REQUIRED_FIELDS = {"spread", "zscore"}
ECM_REQUIRED_FIELDS = {"ecm_x", "ecm_y", "ecm_strength"}
TWO_LEG_REQUIRED_FIELDS = {"price_x", "price_y"}
EXECUTION_ASSUMPTION_FIELDS = {"hedge_ratio", "beta", "funding_x_bps", "funding_y_bps"}


@dataclass(frozen=True)
class PairDetailSnapshot:
    pair_id: str
    pair: str
    asset_x: str
    asset_y: str
    exchange: str
    interval: str | None = None
    period: int | None = None
    strategy_mode: str | None = None
    hedge_ratio: float | None = None
    hurst: float | None = None
    half_life: float | None = None
    pearson: float | None = None
    spearman: float | None = None
    kendall: float | None = None
    copula: str | None = None
    corr_copula: float | None = None
    u1_given_u2: float | None = None
    u2_given_u1: float | None = None
    sharpe: float | None = None
    sortino: float | None = None
    returns_total: float | None = None
    annual_return: float | None = None
    mean_period_return: float | None = None
    win_rate: float | None = None
    closed_trades: int | None = None
    drawdown: float | None = None
    var: float | None = None
    cvar: float | None = None
    var_sim: float | None = None
    cvar_sim: float | None = None
    x_weighting: float | None = None
    ecm_x_available: bool = False
    ecm_y_available: bool = False
    ecm_strength_available: bool = False
    ecm_deviation_override_available: bool = False
    source_url: str | None = None

    def to_row(self) -> dict[str, object]:
        row = asdict(self)
        if self.u1_given_u2 is not None and self.u2_given_u1 is not None:
            row["conditional_probability_distortion"] = self.u1_given_u2 - self.u2_given_u1
            row["conditional_probabilities"] = True
        return row


def parse_pair_detail_text(text: str, source_url: str | None = None) -> PairDetailSnapshot:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    compact = "\n".join(lines)
    pair_id = _match(r"\bpair/([A-Za-z0-9_-]+)\b", compact, default="unknown")
    asset_x, asset_y, exchange = _extract_pair_header(lines)
    pair = f"{asset_x}-{asset_y}" if asset_x and asset_y else "UNKNOWN"
    strategy_mode = _extract_strategy_mode(lines)
    period = _extract_int_before_phrase(compact, "periods analyzed")
    interval = _extract_selected_label(lines, {"Daily", "4 Hour", "1 Hour", "5 Min"})

    return PairDetailSnapshot(
        pair_id=pair_id,
        pair=pair,
        asset_x=asset_x,
        asset_y=asset_y,
        exchange=exchange,
        interval=interval,
        period=period,
        strategy_mode=strategy_mode,
        hedge_ratio=_value_before_label(lines, "hedge r"),
        hurst=_value_before_label(lines, "hurst"),
        half_life=_value_before_label(lines, "half life"),
        pearson=_value_after_label(lines, "Pearsons"),
        spearman=_value_after_label(lines, "Spearmans"),
        kendall=_value_after_label(lines, "Kendalls"),
        copula=_text_after_label(lines, "Best fit"),
        corr_copula=_value_after_label(lines, "Correlation"),
        u1_given_u2=_conditional_probability(lines, asset_x, asset_y),
        u2_given_u1=_conditional_probability(lines, asset_y, asset_x),
        sharpe=_metric_after_colon(compact, "sharpe") or _value_before_label(lines, "sharpe"),
        sortino=_metric_after_colon(compact, "sortino"),
        returns_total=_metric_after_colon(compact, "net return"),
        annual_return=_metric_after_colon(compact, "annualized return"),
        mean_period_return=_metric_after_colon(compact, "mean period return"),
        win_rate=_metric_after_colon(compact, "win rate"),
        closed_trades=_int_metric_after_colon(compact, "closed trades"),
        drawdown=_metric_after_colon(compact, "max drawdown"),
        var=_metric_after_colon(compact, "VaR \\(at 99%\\)"),
        cvar=_metric_after_colon(compact, "CVaR \\(at 99%\\)"),
        var_sim=_metric_after_colon(compact, "VaR \\(sim\\)"),
        cvar_sim=_metric_after_colon(compact, "CVaR \\(sim\\)"),
        x_weighting=_weighting(compact, asset_x),
        ecm_x_available=_contains_line(lines, "ecm (x)"),
        ecm_y_available=_contains_line(lines, "ecm (y)"),
        ecm_strength_available=_contains_line(lines, "ecm strength"),
        ecm_deviation_override_available="ECM Deviation (min)" in compact,
        source_url=source_url,
    )


def load_pair_detail_payload(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _capture_paths(input_dir: str | Path) -> list[Path]:
    root = Path(input_dir)
    paths = [path for pattern in ("*.json", "*.har") for path in root.glob(pattern)]
    return sorted(set(paths))


def snapshot_from_payload(payload: dict[str, Any]) -> PairDetailSnapshot:
    if "text" in payload:
        return parse_pair_detail_text(str(payload["text"]), source_url=payload.get("url"))
    normalized = {snake_case(key): value for key, value in payload.items()}
    return PairDetailSnapshot(
        pair_id=str(normalized.get("pair_id", normalized.get("id", "unknown"))),
        pair=str(normalized.get("pair", "UNKNOWN")),
        asset_x=str(normalized.get("asset_x", "")),
        asset_y=str(normalized.get("asset_y", "")),
        exchange=str(normalized.get("exchange", "")),
        interval=normalized.get("interval"),
        period=_safe_int(normalized.get("period")),
        strategy_mode=normalized.get("strategy_mode"),
        hedge_ratio=_safe_float(normalized.get("hedge_ratio")),
        hurst=_safe_float(normalized.get("hurst")),
        half_life=_safe_float(normalized.get("half_life")),
        pearson=_safe_float(normalized.get("pearson")),
        spearman=_safe_float(normalized.get("spearman")),
        kendall=_safe_float(normalized.get("kendall")),
        copula=normalized.get("copula"),
        corr_copula=_safe_float(normalized.get("corr_copula")),
        u1_given_u2=_safe_float(normalized.get("u1_given_u2")),
        u2_given_u1=_safe_float(normalized.get("u2_given_u1")),
        sharpe=_safe_float(normalized.get("sharpe")),
        sortino=_safe_float(normalized.get("sortino")),
        returns_total=_safe_float(normalized.get("returns_total")),
        annual_return=_safe_float(normalized.get("annual_return")),
        mean_period_return=_safe_float(normalized.get("mean_period_return")),
        win_rate=_safe_float(normalized.get("win_rate")),
        closed_trades=_safe_int(normalized.get("closed_trades")),
        drawdown=_safe_float(normalized.get("drawdown")),
        var=_safe_float(normalized.get("var")),
        cvar=_safe_float(normalized.get("cvar")),
        var_sim=_safe_float(normalized.get("var_sim")),
        cvar_sim=_safe_float(normalized.get("cvar_sim")),
        x_weighting=_safe_float(normalized.get("x_weighting")),
        ecm_x_available=bool(normalized.get("ecm_x_available", normalized.get("ecm_x") is not None)),
        ecm_y_available=bool(normalized.get("ecm_y_available", normalized.get("ecm_y") is not None)),
        ecm_strength_available=bool(normalized.get("ecm_strength_available", normalized.get("ecm_strength") is not None)),
        ecm_deviation_override_available=bool(normalized.get("ecm_deviation_override_available", False)),
        source_url=normalized.get("source_url"),
    )


def load_pair_detail_snapshots(input_dir: str | Path) -> list[PairDetailSnapshot]:
    snapshots: list[PairDetailSnapshot] = []
    for path in _capture_paths(input_dir):
        snapshots.append(snapshot_from_payload(load_pair_detail_payload(path)))
    return snapshots


def pair_detail_field_rows(snapshots: Iterable[PairDetailSnapshot]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for snapshot in snapshots:
        row = snapshot.to_row()
        for field, value in row.items():
            if value is None:
                continue
            rows.append(
                {
                    "field": field,
                    "type": type(value).__name__,
                    "example": str(value)[:160],
                    "source": "pair_detail",
                    "pair": snapshot.pair,
                }
            )
    if not rows:
        return []
    frame = pd.DataFrame(rows).drop_duplicates(subset=["field", "source", "pair"])
    return frame.sort_values(["source", "field"]).to_dict("records")


def write_pair_detail_reports(input_dir: str | Path, output_dir: str | Path) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    snapshots = load_pair_detail_snapshots(input_dir)
    snapshot_path = output / "pair_detail_research_snapshots.csv"
    fields_path = output / "pair_detail_field_dictionary.csv"
    history_path = output / "pair_detail_history_coverage.csv"
    quality_path = output / "pair_detail_quality_report.csv"
    audit_path = output / "pair_detail_capture_audit.csv"
    checklist_path = output / "pair_detail_capture_checklist.csv"
    pd.DataFrame([snapshot.to_row() for snapshot in snapshots]).to_csv(snapshot_path, index=False)
    pd.DataFrame(pair_detail_field_rows(snapshots)).to_csv(fields_path, index=False)
    pd.DataFrame(pair_detail_history_coverage(input_dir)).to_csv(history_path, index=False)
    pd.DataFrame(pair_detail_quality_report(input_dir), columns=PAIR_DETAIL_QUALITY_COLUMNS).to_csv(
        quality_path, index=False
    )
    pd.DataFrame(pair_detail_capture_audit(input_dir), columns=PAIR_DETAIL_CAPTURE_AUDIT_COLUMNS).to_csv(
        audit_path, index=False
    )
    pd.DataFrame(pair_detail_capture_checklist(input_dir), columns=PAIR_DETAIL_CAPTURE_CHECKLIST_COLUMNS).to_csv(
        checklist_path, index=False
    )
    return {
        "snapshots": snapshot_path,
        "fields": fields_path,
        "history_coverage": history_path,
        "quality": quality_path,
        "capture_audit": audit_path,
        "capture_checklist": checklist_path,
    }


def datasets_from_pair_detail_snapshots(input_dir: str | Path, *, require_research_usable: bool = False) -> list[PairDataset]:
    datasets: list[PairDataset] = []
    root = Path(input_dir)
    usable_paths: set[Path] | None = None
    if require_research_usable:
        usable_paths = {
            Path(str(row["path"]))
            for row in pair_detail_quality_report(root)
            if bool(row.get("research_usable"))
            and "placeholder_execution_assumptions" not in str(row.get("quality_blockers") or "")
        }
    for path in sorted(root.glob("*.json")):
        if usable_paths is not None and path not in usable_paths:
            continue
        payload = load_pair_detail_payload(path)
        snapshot = snapshot_from_payload(payload)
        history = extract_history_rows(payload)
        if not history:
            continue
        frame = pd.DataFrame(history)
        for column, value in snapshot.to_row().items():
            if column not in frame.columns and value is not None:
                frame[column] = value
        frame = add_derived_beta_from_prices(frame)
        if "regime" not in frame.columns:
            frame["regime"] = "unknown"
        datasets.append(PairDataset(pair=snapshot.pair, frame=frame))
    return datasets


def extract_history_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize known pair-detail history payload shapes into row dictionaries."""
    normalized_payload = _parsed_json_value(payload)
    if isinstance(normalized_payload, dict) and normalized_payload is not payload:
        return extract_history_rows(normalized_payload)
    if isinstance(normalized_payload, list):
        normalized = _history_records(normalized_payload)
        if normalized and _has_research_columns(normalized):
            return normalized
        for item in normalized_payload:
            if isinstance(item, dict):
                normalized = extract_history_rows(item)
                if normalized:
                    return normalized

    for key in ("history", "series"):
        rows = payload.get(key)
        normalized = _history_records(rows)
        if normalized and _has_research_columns(normalized):
            return normalized

    for key in (
        "viewItem",
        "view_item",
        "result",
        "data",
        "response",
        "request",
        "postData",
        "content",
        "text",
        "value",
        "stores",
        "rows",
    ):
        nested = payload.get(key)
        if isinstance(nested, dict):
            normalized = extract_history_rows(nested)
            if normalized:
                return normalized
        elif isinstance(nested, list):
            for nested_item in nested:
                if isinstance(nested_item, dict):
                    normalized = extract_history_rows(nested_item)
                    if normalized:
                        return normalized
        elif isinstance(nested, str):
            parsed = _parsed_json_value(nested)
            if isinstance(parsed, dict):
                normalized = extract_history_rows(parsed)
                if normalized:
                    return normalized
            elif isinstance(parsed, list):
                normalized = _history_records(parsed)
                if normalized and _has_research_columns(normalized):
                    return normalized
                for parsed_item in parsed:
                    if isinstance(parsed_item, dict):
                        normalized = extract_history_rows(parsed_item)
                        if normalized:
                            return normalized

    for key in ("wasm_extracts", "worker_messages", "fetches", "xhrs", "storage", "indexeddb", "scripts"):
        nested_items = payload.get(key)
        if isinstance(nested_items, list):
            for item in nested_items:
                if not isinstance(item, dict):
                    continue
                for nested_key in (
                    "message",
                    "json",
                    "result",
                    "data",
                    "value",
                    "text",
                    "response",
                    "content",
                    "stores",
                    "rows",
                    "candidates",
                ):
                    nested = item.get(nested_key)
                    if isinstance(nested, dict):
                        normalized = extract_history_rows(nested)
                        if normalized:
                            return normalized
                    elif isinstance(nested, list):
                        for nested_item in nested:
                            if isinstance(nested_item, dict):
                                normalized = extract_history_rows(nested_item)
                                if normalized:
                                    return normalized
                    elif isinstance(nested, str):
                        parsed = _parsed_json_value(nested)
                        if isinstance(parsed, dict):
                            normalized = extract_history_rows(parsed)
                            if normalized:
                                return normalized
                        elif isinstance(parsed, list):
                            normalized = _history_records(parsed)
                            if normalized and _has_research_columns(normalized):
                                return normalized
                            for parsed_item in parsed:
                                if isinstance(parsed_item, dict):
                                    normalized = extract_history_rows(parsed_item)
                                    if normalized:
                                        return normalized

    har_entries = _har_entries(payload)
    for entry in har_entries:
        normalized = extract_history_rows(entry)
        if normalized:
            return normalized

    rows = _parallel_series_records(payload)
    if rows:
        return rows
    return []


def pair_detail_history_coverage(input_dir: str | Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in _capture_paths(input_dir):
        payload = load_pair_detail_payload(path)
        rows.append(pair_detail_payload_history_coverage(payload, path))
    return rows


def pair_detail_quality_report(input_dir: str | Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in _capture_paths(input_dir):
        payload = load_pair_detail_payload(path)
        rows.append(pair_detail_payload_quality(payload, path))
    return rows


def pair_detail_payload_quality(payload: dict[str, Any], path: str | Path) -> dict[str, object]:
    snapshot = snapshot_from_payload(payload)
    history = extract_history_rows(payload)
    frame = pd.DataFrame(history)
    history_rows = len(frame)
    missing_research_required = sorted(BASELINE_REQUIRED_FIELDS.difference(frame.columns))
    missing_execution_required = sorted(TWO_LEG_REQUIRED_FIELDS.difference(frame.columns))
    missing_required = sorted(set(missing_research_required) | set(missing_execution_required))
    price_rows = int(frame[["price_x", "price_y"]].dropna().shape[0]) if {"price_x", "price_y"}.issubset(frame.columns) else 0

    metrics = {
        "missing_price_x_rate": _missing_rate(frame, "price_x"),
        "missing_price_y_rate": _missing_rate(frame, "price_y"),
        "stale_price_x_rate": _stale_rate(frame, "price_x"),
        "stale_price_y_rate": _stale_rate(frame, "price_y"),
        "zero_volume_x_rate": _zero_rate(frame, "volume_x_usd"),
        "zero_volume_y_rate": _zero_rate(frame, "volume_y_usd"),
        "nonfinite_spread_rate": _nonfinite_rate(frame, "spread"),
        "nonfinite_zscore_rate": _nonfinite_rate(frame, "zscore"),
    }
    blockers = _pair_detail_quality_blockers(history_rows, missing_research_required, metrics)
    snapshot_fields = {column for column, value in snapshot.to_row().items() if value is not None}
    execution_missing = sorted(EXECUTION_ASSUMPTION_FIELDS.difference(set(frame.columns) | snapshot_fields))
    placeholder_execution = _has_placeholder_execution_assumptions(payload)
    execution_usable = not blockers and not missing_execution_required and not execution_missing and not placeholder_execution
    if execution_missing:
        blockers_for_output = blockers + [f"missing_execution_assumptions:{';'.join(execution_missing)}"]
    else:
        blockers_for_output = list(blockers)
    if missing_execution_required:
        blockers_for_output.append(f"missing_execution_history:{';'.join(missing_execution_required)}")
    if metrics["missing_price_x_rate"] > 0.05 and "price_x" not in missing_execution_required:
        blockers_for_output.append("price_x_missing_above_5pct")
        execution_usable = False
    if metrics["missing_price_y_rate"] > 0.05 and "price_y" not in missing_execution_required:
        blockers_for_output.append("price_y_missing_above_5pct")
        execution_usable = False
    if placeholder_execution:
        blockers_for_output.append("placeholder_execution_assumptions")

    return {
        "path": str(path),
        "pair": snapshot.pair,
        "interval": snapshot.interval or payload.get("interval") or payload.get("resolution") or "",
        "history_rows": history_rows,
        "price_rows": price_rows,
        "missing_required_fields": ";".join(missing_required),
        **metrics,
        "research_usable": not blockers,
        "execution_usable": execution_usable,
        "quality_blockers": ";".join(blockers_for_output),
        "source_note": payload.get("source_note", ""),
    }


def _pair_detail_quality_blockers(
    history_rows: int,
    missing_required: list[str],
    metrics: dict[str, float],
) -> list[str]:
    blockers: list[str] = []
    if history_rows < 80:
        blockers.append("history_rows_below_80")
    if missing_required:
        blockers.append(f"missing_required:{';'.join(missing_required)}")
    if metrics["nonfinite_spread_rate"] > 0.01:
        blockers.append("spread_nonfinite_above_1pct")
    if metrics["nonfinite_zscore_rate"] > 0.01:
        blockers.append("zscore_nonfinite_above_1pct")
    if metrics["missing_price_x_rate"] <= 0.05 and metrics["stale_price_x_rate"] > 0.90:
        blockers.append("price_x_stale_above_90pct")
    if metrics["missing_price_y_rate"] <= 0.05 and metrics["stale_price_y_rate"] > 0.90:
        blockers.append("price_y_stale_above_90pct")
    return blockers


def _has_placeholder_execution_assumptions(payload: dict[str, Any]) -> bool:
    source_note = str(payload.get("source_note", "") or "").lower()
    placeholder_phrases = (
        "funding fields are zero placeholders",
        "placeholder funding",
        "execution assumptions are placeholders",
    )
    return any(phrase in source_note for phrase in placeholder_phrases)


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns or frame.empty:
        return pd.Series(dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _missing_rate(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return 1.0
    return round(float(_numeric_series(frame, column).isna().mean()), 6)


def _nonfinite_rate(frame: pd.DataFrame, column: str) -> float:
    series = _numeric_series(frame, column)
    if series.empty:
        return 1.0
    return round(float(series.isna().mean()), 6)


def _zero_rate(frame: pd.DataFrame, column: str) -> float:
    series = _numeric_series(frame, column)
    if series.empty:
        return 1.0
    valid = series.dropna()
    if valid.empty:
        return 1.0
    return round(float((valid == 0).mean()), 6)


def _stale_rate(frame: pd.DataFrame, column: str) -> float:
    series = _numeric_series(frame, column).dropna()
    if len(series) < 2:
        return 1.0
    return round(float(series.diff().iloc[1:].eq(0).mean()), 6)


def pair_detail_payload_history_coverage(payload: dict[str, Any], path: str | Path) -> dict[str, object]:
    snapshot = snapshot_from_payload(payload)
    history = extract_history_rows(payload)
    history_rows = len(history)
    history_columns = sorted({column for item in history for column in item}) if history_rows else []
    snapshot_columns = sorted(column for column, value in snapshot.to_row().items() if value is not None)
    effective_columns = sorted(set(history_columns) | set(snapshot_columns))
    missing = sorted({"spread", "zscore"}.difference(history_columns))
    missing_ecm = sorted({"ecm_x", "ecm_y", "ecm_strength"}.difference(history_columns))
    missing_two_leg = sorted({"price_x", "price_y"}.difference(history_columns))
    hedge_ratio_available = "hedge_ratio" in effective_columns
    beta_available = "beta" in effective_columns
    funding_columns_available = {"funding_x_bps", "funding_y_bps"}.issubset(history_columns)
    assumption_notes = _execution_assumption_notes(
        hedge_ratio_available=hedge_ratio_available,
        beta_available=beta_available,
        funding_columns_available=funding_columns_available,
    )
    return {
        "path": str(path),
        "pair": snapshot.pair,
        "has_history": history_rows > 0,
        "history_rows": history_rows,
        "history_columns": ";".join(history_columns),
        "effective_columns": ";".join(effective_columns),
        "experiment_ready": history_rows > 0 and not missing,
        "missing_for_baseline_backtest": ";".join(missing),
        "ecm_history_ready": history_rows > 0 and not missing_ecm,
        "missing_for_ecm_backtest": ";".join(missing_ecm),
        "two_leg_execution_ready": history_rows > 0 and not missing and not missing_two_leg,
        "missing_for_two_leg_backtest": ";".join(missing_two_leg),
        "hedge_ratio_available": hedge_ratio_available,
        "beta_available": beta_available,
        "funding_columns_available": funding_columns_available,
        "execution_assumption_notes": ";".join(assumption_notes),
    }


def pair_detail_capture_audit(input_dir: str | Path) -> list[dict[str, object]]:
    """Find nested time-series candidates inside captured dashboard/network JSON.

    The audit reports structure only. It deliberately avoids raw values because
    captures may include account or research payload details that should not be
    copied into reports.
    """
    rows: list[dict[str, object]] = []
    for file_path in _capture_paths(input_dir):
        payload = load_pair_detail_payload(file_path)
        rows.extend(pair_detail_payload_capture_audit(payload, file_path))
    return rows


def pair_detail_payload_capture_audit(payload: dict[str, Any], path: str | Path) -> list[dict[str, object]]:
    snapshot = snapshot_from_payload(payload)
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, str, int, tuple[str, ...]]] = set()
    for json_path, value in _walk_json_with_parsed_strings(payload):
        for candidate_type, records in _candidate_history_records(value):
            if not records:
                continue
            columns = sorted({column for item in records for column in item})
            fingerprint = (json_path, candidate_type, len(records), tuple(columns))
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            missing = sorted({"spread", "zscore"}.difference(columns))
            missing_ecm = sorted({"ecm_x", "ecm_y", "ecm_strength"}.difference(columns))
            missing_two_leg = sorted({"price_x", "price_y"}.difference(columns))
            hedge_ratio_available = "hedge_ratio" in columns
            beta_available = "beta" in columns
            funding_columns_available = {"funding_x_bps", "funding_y_bps"}.issubset(columns)
            rows.append(
                {
                    "path": str(path),
                    "pair": snapshot.pair,
                    "json_path": json_path,
                    "candidate_type": candidate_type,
                    "row_count": len(records),
                    "columns": ";".join(columns),
                    "experiment_ready": len(records) > 0 and not missing,
                    "missing_for_baseline_backtest": ";".join(missing),
                    "ecm_history_ready": len(records) > 0 and not missing_ecm,
                    "missing_for_ecm_backtest": ";".join(missing_ecm),
                    "two_leg_execution_ready": len(records) > 0 and not missing and not missing_two_leg,
                    "missing_for_two_leg_backtest": ";".join(missing_two_leg),
                    "hedge_ratio_available": hedge_ratio_available,
                    "beta_available": beta_available,
                    "funding_columns_available": funding_columns_available,
                    "execution_assumption_notes": ";".join(
                        _execution_assumption_notes(
                            hedge_ratio_available=hedge_ratio_available,
                            beta_available=beta_available,
                            funding_columns_available=funding_columns_available,
                        )
                    ),
                }
            )
    return rows


def pair_detail_capture_checklist(input_dir: str | Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for file_path in _capture_paths(input_dir):
        payload = load_pair_detail_payload(file_path)
        rows.append(pair_detail_payload_capture_checklist(payload, file_path))
    return rows


def pair_detail_payload_capture_checklist(payload: dict[str, Any], path: str | Path) -> dict[str, object]:
    coverage = pair_detail_payload_history_coverage(payload, path)
    audit = pair_detail_payload_capture_audit(payload, path)
    best = _best_capture_candidate(audit)
    history_columns = _split_columns(str(coverage["history_columns"]))
    effective_columns = _split_columns(str(coverage["effective_columns"]))
    best_columns = _split_columns(str(best.get("columns", ""))) if best else set()
    found = set(history_columns) | set(best_columns)
    assumption_found = set(effective_columns) | set(best_columns)
    required = BASELINE_REQUIRED_FIELDS | ECM_REQUIRED_FIELDS | TWO_LEG_REQUIRED_FIELDS
    required_locations = _field_locations(
        fields=required,
        history_columns=history_columns,
        snapshot_columns=effective_columns.difference(history_columns),
        audit=audit,
    )
    assumption_locations = _field_locations(
        fields=EXECUTION_ASSUMPTION_FIELDS,
        history_columns=history_columns,
        snapshot_columns=effective_columns.difference(history_columns),
        audit=audit,
    )
    missing_required = sorted(required.difference(found))
    missing_baseline = sorted(BASELINE_REQUIRED_FIELDS.difference(found))
    missing_ecm = sorted(ECM_REQUIRED_FIELDS.difference(found))
    missing_two_leg = sorted(TWO_LEG_REQUIRED_FIELDS.difference(found))
    missing_assumptions = sorted(EXECUTION_ASSUMPTION_FIELDS.difference(assumption_found))
    baseline_ready = BASELINE_REQUIRED_FIELDS.issubset(found)
    ecm_ready = ECM_REQUIRED_FIELDS.issubset(found)
    two_leg_ready = baseline_ready and TWO_LEG_REQUIRED_FIELDS.issubset(found)
    assumptions_ready = EXECUTION_ASSUMPTION_FIELDS.issubset(assumption_found)
    completeness_fields = required | EXECUTION_ASSUMPTION_FIELDS
    completeness_found = found.union(assumption_found).intersection(completeness_fields)
    completeness_score = round(100.0 * len(completeness_found) / len(completeness_fields), 2)
    capture_counts = _capture_payload_counts(payload)
    next_focus = _next_capture_focus(
        baseline_ready=baseline_ready,
        ecm_ready=ecm_ready,
        two_leg_ready=two_leg_ready,
        assumptions_ready=assumptions_ready,
        missing_required=missing_required,
    )
    return {
        "path": str(path),
        "pair": coverage["pair"],
        "history_rows": coverage["history_rows"],
        "capture_candidate_paths": len(audit),
        "best_candidate_path": best.get("json_path", "") if best else "",
        "best_candidate_type": best.get("candidate_type", "") if best else "",
        "best_candidate_rows": best.get("row_count", 0) if best else 0,
        "found_required_fields": ";".join(sorted(found.intersection(required))),
        "required_field_locations": ";".join(f"{field}={required_locations[field]}" for field in sorted(required_locations)),
        "execution_assumption_locations": ";".join(
            f"{field}={assumption_locations[field]}" for field in sorted(assumption_locations)
        ),
        "missing_required_fields": ";".join(missing_required),
        "missing_baseline_fields": ";".join(missing_baseline),
        "missing_ecm_fields": ";".join(missing_ecm),
        "missing_two_leg_fields": ";".join(missing_two_leg),
        "missing_execution_assumption_fields": ";".join(missing_assumptions),
        "capture_completeness_score": completeness_score,
        "capture_fetches": capture_counts["fetches"],
        "capture_xhrs": capture_counts["xhrs"],
        "capture_worker_messages": capture_counts["worker_messages"],
        "capture_wasm_extracts": capture_counts["wasm_extracts"],
        "capture_har_entries": capture_counts["har_entries"],
        "capture_har_response_texts": capture_counts["har_response_texts"],
        "capture_har_dydx_candle_requests": capture_counts["har_dydx_candle_requests"],
        "capture_storage_items": capture_counts["storage"],
        "capture_indexeddb_databases": capture_counts["indexeddb"],
        "capture_scripts": capture_counts["scripts"],
        "capture_resources": capture_counts["resources"],
        "capture_payload_sources": capture_counts["payload_sources"],
        "baseline_ready": baseline_ready,
        "ecm_ready": ecm_ready,
        "two_leg_ready": two_leg_ready,
        "execution_assumptions_ready": assumptions_ready,
        "import_ready": baseline_ready and ecm_ready,
        "research_spine_ready": baseline_ready and ecm_ready and two_leg_ready,
        "next_capture_focus": next_focus,
        "capture_operator_hint": _capture_operator_hint(
            next_focus=next_focus,
            capture_counts=capture_counts,
            capture_candidate_paths=len(audit),
            baseline_ready=baseline_ready,
            ecm_ready=ecm_ready,
            two_leg_ready=two_leg_ready,
            assumptions_ready=assumptions_ready,
        ),
    }


def _best_capture_candidate(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        return {}

    def score(row: dict[str, object]) -> tuple[int, int, int, int]:
        return (
            int(bool(row.get("two_leg_execution_ready"))),
            int(bool(row.get("ecm_history_ready"))),
            int(bool(row.get("experiment_ready"))),
            int(row.get("row_count", 0) or 0),
        )

    return max(rows, key=score)


def _capture_payload_counts(payload: dict[str, Any]) -> dict[str, object]:
    summary = payload.get("capture_summary")
    har_entries = _har_entries(payload)
    counts = {
        "fetches": _safe_count(payload.get("fetches")),
        "xhrs": _safe_count(payload.get("xhrs")),
        "worker_messages": _safe_count(payload.get("worker_messages")),
        "wasm_extracts": _safe_count(payload.get("wasm_extracts")),
        "har_entries": len(har_entries),
        "har_response_texts": sum(1 for entry in har_entries if _har_response_text(entry)),
        "har_dydx_candle_requests": sum(1 for entry in har_entries if _is_har_dydx_candle_request(entry)),
        "storage": _safe_count(payload.get("storage")),
        "indexeddb": _safe_count(payload.get("indexeddb")),
        "scripts": _safe_count(payload.get("scripts")),
        "resources": _safe_count(payload.get("resources")),
    }
    if isinstance(summary, dict):
        for key in counts:
            counts[key] = _safe_int(summary.get(key)) or counts[key]
    sources = []
    if counts["fetches"] or counts["xhrs"]:
        sources.append("network")
    if counts["worker_messages"]:
        sources.append("worker")
    if counts["wasm_extracts"]:
        sources.append("wasm")
    if counts["har_entries"]:
        sources.append("har")
    if counts["storage"] or counts["indexeddb"]:
        sources.append("storage")
    if counts["scripts"]:
        sources.append("scripts")
    if counts["resources"]:
        sources.append("resources")
    counts["payload_sources"] = ";".join(sources) if sources else "none"
    return counts


def _capture_operator_hint(
    *,
    next_focus: str,
    capture_counts: dict[str, object],
    capture_candidate_paths: int,
    baseline_ready: bool,
    ecm_ready: bool,
    two_leg_ready: bool,
    assumptions_ready: bool,
) -> str:
    payload_sources = str(capture_counts.get("payload_sources", "none") or "none")
    if payload_sources == "none" and capture_candidate_paths == 0:
        return (
            "not_a_browser_capture:paste_capture_helper_on_authenticated_pair_page,"
            "click_refresh_or_recalculate,run_await___CW_CAPTURE_STATUS__,"
            "then_await___CW_DOWNLOAD_CAPTURE__"
        )
    if int(capture_counts.get("har_entries", 0) or 0) and not int(capture_counts.get("har_response_texts", 0) or 0):
        return "har_has_requests_but_no_response_bodies:copy_har_with_content_or_copy_response"
    if int(capture_counts.get("har_dydx_candle_requests", 0) or 0) and capture_candidate_paths == 0:
        return "har_has_dydx_candle_requests_without_parseable_history:copy_response_body_for_200_candle_requests"
    if capture_candidate_paths == 0:
        return (
            "payloads_captured_but_no_history_candidate:"
            "repeat_after_pair_refresh_or_export_network_har"
        )
    if not baseline_ready:
        return f"capture_required_baseline_history:{next_focus.split(':', 1)[-1]}"
    if not ecm_ready:
        return f"capture_required_ecm_history:{next_focus.split(':', 1)[-1]}"
    if not two_leg_ready:
        return "capture_leg_price_history:price_x;price_y"
    if not assumptions_ready:
        return "optional_before_production:hedge_ratio;beta;funding_x_bps;funding_y_bps"
    return "ready:import_capture_and_run_research_spine"


def _safe_count(value: Any) -> int:
    return len(value) if isinstance(value, list) else 0


def _split_columns(value: str) -> set[str]:
    if not value:
        return set()
    return {item for item in value.split(";") if item}


def _field_locations(
    *,
    fields: set[str],
    history_columns: set[str],
    snapshot_columns: set[str],
    audit: list[dict[str, object]],
) -> dict[str, str]:
    locations: dict[str, str] = {}
    for field in sorted(fields):
        candidate_paths = [
            str(row.get("json_path", ""))
            for row in audit
            if field in _split_columns(str(row.get("columns", ""))) and row.get("json_path")
        ]
        if candidate_paths:
            locations[field] = candidate_paths[0]
            continue
        if field in history_columns:
            locations[field] = "history"
            continue
        if field in snapshot_columns:
            locations[field] = "snapshot"
    return locations


def _next_capture_focus(
    *,
    baseline_ready: bool,
    ecm_ready: bool,
    two_leg_ready: bool,
    assumptions_ready: bool,
    missing_required: list[str],
) -> str:
    if not baseline_ready:
        missing = sorted(BASELINE_REQUIRED_FIELDS.intersection(missing_required))
        return f"capture_baseline_history:{';'.join(missing)}"
    if not ecm_ready:
        missing = sorted(ECM_REQUIRED_FIELDS.intersection(missing_required))
        return f"capture_ecm_history:{';'.join(missing)}"
    if not two_leg_ready:
        missing = sorted(TWO_LEG_REQUIRED_FIELDS.intersection(missing_required))
        return f"capture_two_leg_prices:{';'.join(missing)}"
    if not assumptions_ready:
        return "capture_optional_execution_assumptions:hedge_ratio;beta;funding_x_bps;funding_y_bps"
    return "ready_for_research_spine"


def _history_records(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        return []
    if all(isinstance(item, dict) for item in value):
        return [_canonical_history_row(item) for item in value]
    return []


def _parallel_series_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    series: dict[str, list[Any]] = {}
    for canonical, candidates in HISTORY_ALIASES.items():
        found = _first_list(payload, candidates)
        if found:
            series[canonical] = found
    if not series:
        return []
    row_count = max(len(values) for values in series.values())
    rows: list[dict[str, Any]] = []
    for idx in range(row_count):
        row: dict[str, Any] = {}
        for column, values in series.items():
            if idx < len(values):
                row[column] = values[idx]
        rows.append(row)
    return rows


def _canonical_history_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in row.items():
        canonical = HISTORY_ALIAS_TO_CANONICAL.get(snake_case(str(key)), snake_case(str(key)))
        normalized[canonical] = value
    return normalized


def _first_list(payload: dict[str, Any], keys: tuple[str, ...]) -> list[Any]:
    normalized = {snake_case(str(key)): value for key, value in payload.items()}
    for key in keys:
        value = normalized.get(snake_case(key))
        if isinstance(value, list) and value:
            return value
    return []


def _candidate_history_records(value: Any) -> list[tuple[str, list[dict[str, Any]]]]:
    candidates: list[tuple[str, list[dict[str, Any]]]] = []
    records = _history_records(value)
    if records and _has_research_columns(records):
        candidates.append(("list_of_records", records))
    if isinstance(value, dict):
        parallel = _parallel_series_records(value)
        if parallel:
            candidates.append(("parallel_series", parallel))
    return candidates


def _execution_assumption_notes(
    hedge_ratio_available: bool,
    beta_available: bool,
    funding_columns_available: bool,
) -> list[str]:
    notes: list[str] = []
    if not hedge_ratio_available:
        notes.append("hedge_ratio_default_1.0")
    if not beta_available:
        notes.append("beta_default_1.0")
    if not funding_columns_available:
        notes.append("funding_cost_model_default")
    return notes


def _har_entries(payload: dict[str, Any]) -> list[dict[str, Any]]:
    entries = payload.get("entries")
    if isinstance(entries, list):
        return [entry for entry in entries if isinstance(entry, dict)]
    log = payload.get("log")
    if isinstance(log, dict) and isinstance(log.get("entries"), list):
        return [entry for entry in log["entries"] if isinstance(entry, dict)]
    return []


def _har_response_text(entry: dict[str, Any]) -> str:
    response = entry.get("response")
    if not isinstance(response, dict):
        return ""
    content = response.get("content")
    if not isinstance(content, dict):
        return ""
    text = content.get("text")
    return text if isinstance(text, str) else ""


def _har_request_url(entry: dict[str, Any]) -> str:
    request = entry.get("request")
    if not isinstance(request, dict):
        return ""
    url = request.get("url")
    return url if isinstance(url, str) else ""


def _is_har_dydx_candle_request(entry: dict[str, Any]) -> bool:
    url = _har_request_url(entry)
    return "/v4/candles/perpetualMarkets/" in url


def _parsed_json_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if len(text) < 2 or text[0] not in "[{":
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def _has_research_columns(records: list[dict[str, Any]]) -> bool:
    if not records:
        return False
    required = AUDIT_RESEARCH_FIELDS
    for item in records:
        if not isinstance(item, dict):
            continue
        for column in item:
            if snake_case(str(column)) in required:
                return True
    return False


def _walk_json_with_parsed_strings(value: Any, path: str = "$") -> Iterable[tuple[str, Any]]:
    if _is_capture_metadata_path(path):
        return
    parsed = _parsed_json_value(value)
    if parsed is not value:
        yield from _walk_json_with_parsed_strings(parsed, f"{path}#json")
        return
    yield path, value
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _walk_json_with_parsed_strings(child, f"{path}.{_json_path_key(str(key))}")
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            yield from _walk_json_with_parsed_strings(child, f"{path}[{idx}]")


def _is_capture_metadata_path(path: str) -> bool:
    return path == "$.capture_summary" or path.startswith("$.capture_summary.")


def _walk_json(value: Any, path: str = "$") -> Iterable[tuple[str, Any]]:
    yield path, value
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _walk_json(child, f"{path}.{_json_path_key(str(key))}")
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            yield from _walk_json(child, f"{path}[{idx}]")


def _json_path_key(key: str) -> str:
    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
        return key
    return json.dumps(key)


def _match(pattern: str, text: str, default: str = "") -> str:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return match.group(1) if match else default


def _extract_pair_header(lines: list[str]) -> tuple[str, str, str]:
    for line in lines:
        parts = line.split()
        if len(parts) >= 3 and _looks_like_market(parts[0]) and _looks_like_market(parts[1]):
            return parts[0], parts[1], parts[2]
    return "", "", ""


def _looks_like_market(value: str) -> bool:
    return bool(re.match(r"^[A-Z0-9]+-[A-Z0-9]+$", value))


def _extract_strategy_mode(lines: list[str]) -> str | None:
    modes = {"static", "dynamic", "ou", "copula"}
    exact_hits: list[str] = []
    for line in lines:
        lower = line.lower()
        if lower in modes:
            exact_hits.append(lower)
    return exact_hits[-1] if exact_hits else None


def _extract_selected_label(lines: list[str], options: set[str]) -> str | None:
    for line in lines:
        if line in options:
            return line.lower().replace(" ", "_")
    return None


def _extract_int_before_phrase(text: str, phrase: str) -> int | None:
    match = re.search(rf"(\d+)\s+{re.escape(phrase)}", text, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _contains_line(lines: list[str], value: str) -> bool:
    return any(line.lower() == value.lower() for line in lines)


def _value_before_label(lines: list[str], label: str) -> float | None:
    lower = [line.lower() for line in lines]
    try:
        idx = lower.index(label.lower())
    except ValueError:
        return None
    for prior in reversed(lines[max(0, idx - 3) : idx]):
        value = _safe_float(prior)
        if value is not None:
            return value
    return None


def _value_after_label(lines: list[str], label: str) -> float | None:
    lower = [line.lower() for line in lines]
    try:
        idx = lower.index(label.lower())
    except ValueError:
        return None
    for later in lines[idx + 1 : idx + 5]:
        value = _safe_float(later)
        if value is not None:
            return value
    return None


def _text_after_label(lines: list[str], label: str) -> str | None:
    lower = [line.lower() for line in lines]
    try:
        idx = lower.index(label.lower())
    except ValueError:
        return None
    return lines[idx + 1] if idx + 1 < len(lines) else None


def _conditional_probability(lines: list[str], given: str, condition: str) -> float | None:
    label = f"{given} given {condition}".lower()
    lower = [line.lower() for line in lines]
    try:
        idx = lower.index(label)
    except ValueError:
        return None
    return _safe_float(lines[idx + 1]) if idx + 1 < len(lines) else None


def _metric_after_colon(text: str, label: str) -> float | None:
    match = re.search(rf"{label}\s*:\s*([-+]?\d+(?:\.\d+)?%?)", text, flags=re.IGNORECASE)
    return _safe_float(match.group(1)) if match else None


def _int_metric_after_colon(text: str, label: str) -> int | None:
    match = re.search(rf"{label}\s*:\s*(\d+)", text, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _weighting(text: str, asset_x: str) -> float | None:
    match = re.search(rf"([-+]?\d+(?:\.\d+)?)\s*\n?\({re.escape(asset_x)} capital weighting\)", text, flags=re.IGNORECASE)
    return _safe_float(match.group(1)) if match else None


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    scale = 0.01 if text.endswith("%") else 1.0
    text = text.rstrip("%")
    try:
        return float(text) * scale
    except ValueError:
        return None


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
