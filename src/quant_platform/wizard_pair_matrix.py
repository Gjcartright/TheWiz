from __future__ import annotations

import csv
import json
import re
from collections import Counter
from io import StringIO
from pathlib import Path
from typing import Iterable

from quant_platform.orchestration.corrective_runtime import atomic_write_text


def _text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _metric_float(value: object) -> float | None:
    text = _text(value)
    if not text:
        return None
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if match is None:
        return None
    return float(match.group(0))


def _top_strip_text(top_strip: object, key: str) -> str:
    if not isinstance(top_strip, dict):
        return ""
    return _text(top_strip.get(key, ""))


def normalize_matrix_rows(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    for row in rows:
        normalized.append(
            {
                "pair": _text(row.get("pair", "")),
                "periods": _text(row.get("periods", "")),
                "timeframe": _text(row.get("timeframe", "")),
                "strategy": _text(row.get("strategy", "")),
                "best_fit": _text(row.get("best_fit", "")),
                "conditional_chart": _metric_float(row.get("conditional_chart", "")),
                "pearson": _metric_float(row.get("pearson", "")),
                "spearman": _metric_float(row.get("spearman", "")),
                "kendall": _metric_float(row.get("kendall", "")),
                "sharpe": _metric_float(row.get("sharpe_metric", "")),
                "sortino": _metric_float(row.get("sortino_metric", "")),
                "net_return_pct": _metric_float(row.get("net_return_metric", "")),
                "annualized_return_pct": _metric_float(row.get("annualized_return_metric", "")),
                "win_rate_pct": _metric_float(row.get("win_rate_metric", "")),
                "closed_trades": _metric_float(row.get("closed_trades_metric", "")),
                "max_drawdown_pct": _metric_float(row.get("max_drawdown_metric", "")),
                "top_hurst": _metric_float(_top_strip_text(row.get("top_strip", {}), "hurst")),
                "top_half_life": _metric_float(_top_strip_text(row.get("top_strip", {}), "half_life")),
                "top_corr": _metric_float(_top_strip_text(row.get("top_strip", {}), "corr")),
                "top_hedger": _metric_float(_top_strip_text(row.get("top_strip", {}), "hedger")),
                "top_beta": _metric_float(_top_strip_text(row.get("top_strip", {}), "beta")),
                "top_mdd": _metric_float(_top_strip_text(row.get("top_strip", {}), "mdd")),
                "top_returns_pct": _metric_float(_top_strip_text(row.get("top_strip", {}), "returns")),
                "top_sharpe": _metric_float(_top_strip_text(row.get("top_strip", {}), "sharpe")),
            }
        )
    return normalized


def variant_fingerprint(row: dict[str, object]) -> tuple[object, ...]:
    return (
        _text(row.get("pair", "")),
        _text(row.get("periods", "")),
        _text(row.get("timeframe", "")),
        _text(row.get("strategy", "")),
        _text(row.get("best_fit", "")),
        _metric_float(row.get("sharpe_metric", "")),
        _metric_float(row.get("net_return_metric", "")),
        _metric_float(row.get("annualized_return_metric", "")),
        _metric_float(row.get("win_rate_metric", "")),
        _metric_float(row.get("closed_trades_metric", "")),
        _metric_float(row.get("max_drawdown_metric", "")),
        _metric_float(row.get("conditional_chart", "")),
    )


def detect_matrix_anomalies(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    items = list(rows)
    normalized = normalize_matrix_rows(items)
    by_timeframe: dict[str, list[dict[str, object]]] = {}
    for row in normalized:
        by_timeframe.setdefault(_text(row.get("timeframe", "")), []).append(row)

    anomalies: list[dict[str, object]] = []
    for row in normalized:
        pair = _text(row.get("pair", ""))
        timeframe = _text(row.get("timeframe", ""))
        strategy = _text(row.get("strategy", ""))
        if not pair or not timeframe or not strategy:
            anomalies.append(
                {
                    "pair": pair,
                    "timeframe": timeframe,
                    "strategy": strategy,
                    "severity": "high",
                    "anomaly": "blank_variant_fields",
                    "detail": "pair, timeframe, or strategy was blank in the captured variant",
                }
            )
            continue
        if timeframe == "5 Min" and not _text(row.get("best_fit", "")):
            anomalies.append(
                {
                    "pair": pair,
                    "timeframe": timeframe,
                    "strategy": strategy,
                    "severity": "medium",
                    "anomaly": "missing_best_fit",
                    "detail": "5-minute variant omitted copula best-fit output",
                }
            )
        if timeframe == "5 Min" and row.get("conditional_chart") is None:
            anomalies.append(
                {
                    "pair": pair,
                    "timeframe": timeframe,
                    "strategy": strategy,
                    "severity": "medium",
                    "anomaly": "missing_conditional_chart",
                    "detail": "5-minute variant omitted conditional chart output",
                }
            )

    for timeframe, timeframe_rows in by_timeframe.items():
        signature_counts = Counter(
            (
                row.get("sharpe"),
                row.get("net_return_pct"),
                row.get("annualized_return_pct"),
                row.get("closed_trades"),
                row.get("max_drawdown_pct"),
            )
            for row in timeframe_rows
        )
        duplicate_signatures = {signature for signature, count in signature_counts.items() if count > 1}
        for row in timeframe_rows:
            signature = (
                row.get("sharpe"),
                row.get("net_return_pct"),
                row.get("annualized_return_pct"),
                row.get("closed_trades"),
                row.get("max_drawdown_pct"),
            )
            if signature in duplicate_signatures:
                anomalies.append(
                    {
                        "pair": _text(row.get("pair", "")),
                        "timeframe": timeframe,
                        "strategy": _text(row.get("strategy", "")),
                        "severity": "medium" if timeframe == "5 Min" else "low",
                        "anomaly": "duplicate_outcome_signature",
                        "detail": "multiple strategy variants shared the same outcome signature",
                    }
                )
    return anomalies


def compare_hourly_matrix(current_rows: Iterable[dict[str, object]], prior_rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    prior_by_key = {
        (
            _text(row.get("pair", "")),
            _text(row.get("periods", "")),
            _text(row.get("timeframe", "")),
            _text(row.get("strategy", "")),
        ): variant_fingerprint(row)
        for row in prior_rows
    }
    review_rows: list[dict[str, object]] = []
    for row in current_rows:
        key = (
            _text(row.get("pair", "")),
            _text(row.get("periods", "")),
            _text(row.get("timeframe", "")),
            _text(row.get("strategy", "")),
        )
        fingerprint = variant_fingerprint(row)
        prior = prior_by_key.get(key)
        status = "new" if prior is None else ("unchanged" if prior == fingerprint else "changed")
        review_rows.append(
            {
                "pair": key[0],
                "periods": key[1],
                "timeframe": key[2],
                "strategy": key[3],
                "review_status": status,
                "should_deep_capture": status in {"new", "changed"},
            }
        )
    return review_rows


def write_hourly_review_artifacts(
    *,
    current_rows: Iterable[dict[str, object]],
    prior_rows: Iterable[dict[str, object]],
    output_dir: str | Path,
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    normalized = normalize_matrix_rows(current_rows)
    review = compare_hourly_matrix(list(current_rows), list(prior_rows))
    anomalies = detect_matrix_anomalies(current_rows)

    normalized_path = output / "wizard_pair_matrix_normalized.csv"
    review_path = output / "wizard_pair_matrix_hourly_review.csv"
    anomalies_path = output / "wizard_pair_matrix_anomalies.csv"

    _write_csv(normalized_path, normalized)
    _write_csv(review_path, review)
    _write_csv(anomalies_path, anomalies)
    return {
        "normalized": normalized_path,
        "review": review_path,
        "anomalies": anomalies_path,
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        atomic_write_text(path, "", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue(), encoding="utf-8")


def load_matrix_rows(path: str | Path) -> list[dict[str, object]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
