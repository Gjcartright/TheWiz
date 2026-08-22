from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

from quant_platform.orchestration.corrective_runtime import atomic_write_text
from quant_platform.wizard_pair_matrix import (
    compare_hourly_matrix,
    detect_matrix_anomalies,
    normalize_matrix_rows,
)


def build_hourly_database(
    *,
    current_rows: list[dict[str, object]],
    output_dir: str | Path,
    scan_timestamp: str | None = None,
    source_name: str = "wizard_pair_page",
    hourly_pair_limit: int = 2,
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    timestamp = scan_timestamp or datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    variant_db_path = output / "wizard_hourly_variant_database.csv"
    current_alias_path = output / "wizard_hourly_current_alias.csv"
    review_path = output / "wizard_hourly_review_queue.csv"
    anomaly_path = output / "wizard_hourly_anomalies.csv"
    candidate_path = output / "wizard_hourly_candidate_queue.csv"
    pair_queue_path = output / "wizard_hourly_pair_queue.csv"
    strategy_queue_path = output / "wizard_hourly_strategy_queue.csv"

    prior_rows = _read_csv_rows(current_alias_path)
    review_rows = compare_hourly_matrix(current_rows, prior_rows)
    anomaly_rows = detect_matrix_anomalies(current_rows)
    anomaly_index = _anomaly_index(anomaly_rows)
    normalized_rows = normalize_matrix_rows(current_rows)

    scan_rows: list[dict[str, object]] = []
    for raw_row, normalized_row in zip(current_rows, normalized_rows):
        key = _variant_key(raw_row)
        anomaly_key = _variant_anomaly_key(raw_row)
        review_row = _match_review_row(review_rows, key)
        scan_rows.append(
            {
                "scan_timestamp": timestamp,
                "source_name": source_name,
                "pair": normalized_row.get("pair", ""),
                "periods": normalized_row.get("periods", ""),
                "timeframe": normalized_row.get("timeframe", ""),
                "strategy": normalized_row.get("strategy", ""),
                "best_fit": normalized_row.get("best_fit", ""),
                "conditional_chart": normalized_row.get("conditional_chart", ""),
                "pearson": normalized_row.get("pearson", ""),
                "spearman": normalized_row.get("spearman", ""),
                "kendall": normalized_row.get("kendall", ""),
                "sharpe": normalized_row.get("sharpe", ""),
                "sortino": normalized_row.get("sortino", ""),
                "net_return_pct": normalized_row.get("net_return_pct", ""),
                "annualized_return_pct": normalized_row.get("annualized_return_pct", ""),
                "win_rate_pct": normalized_row.get("win_rate_pct", ""),
                "closed_trades": normalized_row.get("closed_trades", ""),
                "max_drawdown_pct": normalized_row.get("max_drawdown_pct", ""),
                "top_hurst": normalized_row.get("top_hurst", ""),
                "top_half_life": normalized_row.get("top_half_life", ""),
                "top_corr": normalized_row.get("top_corr", ""),
                "top_hedger": normalized_row.get("top_hedger", ""),
                "top_beta": normalized_row.get("top_beta", ""),
                "top_mdd": normalized_row.get("top_mdd", ""),
                "top_returns_pct": normalized_row.get("top_returns_pct", ""),
                "top_sharpe": normalized_row.get("top_sharpe", ""),
                "review_status": review_row.get("review_status", "") if review_row else "",
                "should_deep_capture": review_row.get("should_deep_capture", False) if review_row else False,
                "anomaly_count": len(anomaly_index.get(anomaly_key, [])),
                "highest_anomaly_severity": _highest_severity(anomaly_index.get(anomaly_key, [])),
            }
        )

    all_rows = _dedupe_history_rows(_read_csv_rows(variant_db_path) + scan_rows)
    _write_csv(variant_db_path, all_rows)
    _write_csv(review_path, review_rows)
    _write_csv(anomaly_path, anomaly_rows)

    current_alias = _latest_variant_rows(all_rows)
    _write_csv(current_alias_path, current_alias)
    candidate_rows = _candidate_queue(scan_rows, anomaly_index)
    pair_queue_rows = _pair_queue(candidate_rows, hourly_pair_limit=hourly_pair_limit)
    strategy_queue_rows = _strategy_queue(candidate_rows)
    _write_csv(candidate_path, candidate_rows)
    _write_csv(pair_queue_path, pair_queue_rows)
    _write_csv(strategy_queue_path, strategy_queue_rows)
    return {
        "variant_database": variant_db_path,
        "current_alias": current_alias_path,
        "review_queue": review_path,
        "anomalies": anomaly_path,
        "candidate_queue": candidate_path,
        "pair_queue": pair_queue_path,
        "strategy_queue": strategy_queue_path,
    }


def build_hourly_database_from_live_scanner(
    *,
    root: str | Path,
    output_dir: str | Path | None = None,
    hourly_pair_limit: int = 2,
) -> dict[str, Path]:
    root_path = Path(root)
    live_paths = _discover_live_scanner_paths(root_path)
    if not live_paths:
        return {}

    current_rows: list[dict[str, object]] = []
    timestamps: list[str] = []
    for live_path in live_paths:
        try:
            payload = json.loads(live_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        strategy_rows = _live_scanner_current_rows(payload)
        if not strategy_rows:
            continue
        current_rows.extend(strategy_rows)
        if isinstance(payload, dict):
            timestamps.append(str(payload.get("captured_at", "")).strip())
    if not current_rows:
        return {}
    target_dir = output_dir or root_path / "reports" / "active" / "wizard_hourly_database"
    return build_hourly_database(
        current_rows=current_rows,
        output_dir=target_dir,
        scan_timestamp=max((timestamp for timestamp in timestamps if timestamp), default="") or None,
        source_name="wizard_scanner_live_dom_multi_strategy",
        hourly_pair_limit=hourly_pair_limit,
    )


def _discover_live_scanner_paths(root: Path) -> list[Path]:
    collected_root = root / "data" / "collected"
    matches = sorted(collected_root.glob("wizard_scanner_*_visible_live/*_latest.json"))
    if matches:
        return matches
    legacy_green = sorted(collected_root.glob("wizard_scanner_*_green/*_latest.json"))
    if legacy_green:
        return legacy_green
    legacy_path = (
        collected_root
        / "wizard_scanner_static_spread_visible_live"
        / "wizard_scanner_static_spread_visible_live_latest.json"
    )
    if legacy_path.exists():
        return [legacy_path]
    legacy_green_path = (
        collected_root
        / "wizard_scanner_static_spread_green"
        / "wizard_scanner_static_spread_green_latest.json"
    )
    return [legacy_green_path] if legacy_green_path.exists() else []


def _variant_key(row: dict[str, object]) -> tuple[str, str, str, str]:
    return (
        str(row.get("pair", "")).strip(),
        str(row.get("periods", "")).strip(),
        str(row.get("timeframe", "")).strip(),
        str(row.get("strategy", "")).strip(),
    )


def _match_review_row(rows: list[dict[str, object]], key: tuple[str, str, str, str]) -> dict[str, object] | None:
    for row in rows:
        if _variant_key(row) == key:
            return row
    return None


def _anomaly_index(rows: list[dict[str, object]]) -> dict[tuple[str, str, str, str], list[dict[str, object]]]:
    index: dict[tuple[str, str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = _variant_anomaly_key(row)
        index.setdefault(key, []).append(row)
    return index


def _variant_anomaly_key(row: dict[str, object]) -> tuple[str, str, str, str]:
    return (
        str(row.get("pair", "")).strip(),
        "",
        str(row.get("timeframe", "")).strip(),
        str(row.get("strategy", "")).strip(),
    )


def _highest_severity(rows: list[dict[str, object]]) -> str:
    if not rows:
        return ""
    order = {"high": 3, "medium": 2, "low": 1, "": 0}
    return max((str(row.get("severity", "")).strip() for row in rows), key=lambda value: order.get(value, 0))


def _latest_variant_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    latest: dict[tuple[str, str, str, str], dict[str, object]] = {}
    for row in rows:
        key = _variant_key(row)
        current = latest.get(key)
        if current is None or str(row.get("scan_timestamp", "")) >= str(current.get("scan_timestamp", "")):
            latest[key] = row
    return list(latest.values())


def _candidate_queue(rows: list[dict[str, object]], anomaly_index: dict[tuple[str, str, str, str], list[dict[str, object]]]) -> list[dict[str, object]]:
    queue: list[dict[str, object]] = []
    for row in rows:
        key = _variant_anomaly_key(row)
        anomalies = anomaly_index.get(key, [])
        severity = _highest_severity(anomalies)
        if str(row.get("review_status", "")).strip() not in {"new", "changed"}:
            continue
        queue.append(
            {
                "scan_timestamp": row.get("scan_timestamp", ""),
                "pair": row.get("pair", ""),
                "periods": row.get("periods", ""),
                "timeframe": row.get("timeframe", ""),
                "strategy": row.get("strategy", ""),
                "annualized_return_pct": row.get("annualized_return_pct", ""),
                "sharpe": row.get("sharpe", ""),
                "closed_trades": row.get("closed_trades", ""),
                "review_status": row.get("review_status", ""),
                "highest_anomaly_severity": severity,
                "candidate_status": "blocked_by_anomaly" if severity == "high" else "review_now",
            }
        )
    queue.sort(
        key=lambda row: (
            str(row.get("candidate_status", "")) != "review_now",
            -_float_or_zero(row.get("annualized_return_pct", "")),
            -_float_or_zero(row.get("sharpe", "")),
            -_float_or_zero(row.get("closed_trades", "")),
        )
    )
    return queue


def _pair_queue(candidate_rows: list[dict[str, object]], *, hourly_pair_limit: int) -> list[dict[str, object]]:
    pair_index: dict[str, list[dict[str, object]]] = {}
    for row in candidate_rows:
        pair = str(row.get("pair", "")).strip()
        if not pair:
            continue
        pair_index.setdefault(pair, []).append(row)

    pair_rows: list[dict[str, object]] = []
    for pair, rows in pair_index.items():
        review_now = [row for row in rows if str(row.get("candidate_status", "")).strip() == "review_now"]
        blocked = [row for row in rows if str(row.get("candidate_status", "")).strip() == "blocked_by_anomaly"]
        top_review = max(
            review_now or rows,
            key=lambda row: (
                _float_or_zero(row.get("annualized_return_pct", "")),
                _float_or_zero(row.get("sharpe", "")),
                _float_or_zero(row.get("closed_trades", "")),
            ),
        )
        pair_rows.append(
            {
                "pair": pair,
                "actionable_variant_count": len(review_now),
                "blocked_variant_count": len(blocked),
                "top_strategy": top_review.get("strategy", ""),
                "top_timeframe": top_review.get("timeframe", ""),
                "top_annualized_return_pct": top_review.get("annualized_return_pct", ""),
                "top_sharpe": top_review.get("sharpe", ""),
                "top_closed_trades": top_review.get("closed_trades", ""),
                "pair_status": "review_now" if review_now else "blocked_or_empty",
            }
        )

    pair_rows.sort(
        key=lambda row: (
            str(row.get("pair_status", "")) != "review_now",
            -_float_or_zero(row.get("actionable_variant_count", "")),
            -_float_or_zero(row.get("top_annualized_return_pct", "")),
            -_float_or_zero(row.get("top_sharpe", "")),
        )
    )
    for index, row in enumerate(pair_rows, start=1):
        row["hourly_rank"] = index
        row["recommended_this_hour"] = index <= max(hourly_pair_limit, 0) and str(row.get("pair_status", "")) == "review_now"
    return pair_rows


def _strategy_queue(candidate_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    strategy_index: dict[str, list[dict[str, object]]] = {}
    for row in candidate_rows:
        strategy = str(row.get("strategy", "")).strip() or "Unknown"
        strategy_index.setdefault(strategy, []).append(row)

    rows: list[dict[str, object]] = []
    for strategy, variants in strategy_index.items():
        actionable = [row for row in variants if str(row.get("candidate_status", "")).strip() == "review_now"]
        blocked = [row for row in variants if str(row.get("candidate_status", "")).strip() == "blocked_by_anomaly"]
        top_variant = max(
            actionable or variants,
            key=lambda row: (
                _float_or_zero(row.get("annualized_return_pct", "")),
                _float_or_zero(row.get("sharpe", "")),
                _float_or_zero(row.get("closed_trades", "")),
            ),
        )
        rows.append(
            {
                "strategy": strategy,
                "variant_count": len(variants),
                "actionable_variant_count": len(actionable),
                "blocked_variant_count": len(blocked),
                "top_pair": top_variant.get("pair", ""),
                "top_timeframe": top_variant.get("timeframe", ""),
                "top_annualized_return_pct": top_variant.get("annualized_return_pct", ""),
                "top_sharpe": top_variant.get("sharpe", ""),
                "top_closed_trades": top_variant.get("closed_trades", ""),
                "strategy_status": "review_now" if actionable else "blocked_or_empty",
            }
        )

    rows.sort(
        key=lambda row: (
            str(row.get("strategy_status", "")) != "review_now",
            -_float_or_zero(row.get("actionable_variant_count", "")),
            -_float_or_zero(row.get("top_annualized_return_pct", "")),
            -_float_or_zero(row.get("top_sharpe", "")),
            str(row.get("strategy", "")),
        )
    )
    for index, row in enumerate(rows, start=1):
        row["strategy_rank"] = index
    return rows


def _dedupe_history_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    deduped: dict[tuple[str, str, str, str, str], dict[str, object]] = {}
    for row in rows:
        key = (
            str(row.get("scan_timestamp", "")).strip(),
            str(row.get("pair", "")).strip(),
            str(row.get("periods", "")).strip(),
            str(row.get("timeframe", "")).strip(),
            str(row.get("strategy", "")).strip(),
        )
        deduped[key] = row
    return list(deduped.values())


def _float_or_zero(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _live_scanner_current_rows(payload: object) -> list[dict[str, object]]:
    strategy = "Static (Spread)"
    if isinstance(payload, dict):
        rows = payload.get("rows")
        if not isinstance(rows, list):
            return []
        filters = payload.get("scanner_filters")
        if isinstance(filters, dict):
            strategy = str(filters.get("strategy", "") or "").strip() or strategy
    elif isinstance(payload, list):
        rows = payload
    else:
        return []
    current_rows: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        pair_x = str(row.get("pair_x", "") or "").strip()
        pair_y = str(row.get("pair_y", "") or "").strip()
        pair = str(row.get("pair", "") or "").strip()
        if not pair and pair_x and pair_y:
            pair = f"{pair_x}/{pair_y}"
        pair = pair.replace(" / ", "/")
        if not pair:
            continue
        row_strategy = str(
            row.get("strategy")
            or row.get("strategy_mode")
            or row.get("strategy_family")
            or strategy
        ).strip() or strategy
        sharpe = str(
            row.get("reward_sharpe")
            or row.get("reward_sharpe_value")
            or row.get("sharpe")
            or ""
        ).strip()
        returns = str(
            row.get("reward_return")
            or row.get("reward_return_value")
            or row.get("annualized_return_pct")
            or ""
        ).strip()
        max_drawdown = str(
            row.get("max_drawdown_pct")
            or row.get("mdd_value")
            or row.get("mdd")
            or ""
        ).strip()
        correlation = str(
            row.get("correlation_value")
            or row.get("top_corr")
            or ""
        ).strip()
        hurst = str(
            row.get("hurst_value")
            or row.get("top_hurst")
            or ""
        ).strip()
        half_life = str(
            row.get("half_life_value")
            or row.get("top_half_life")
            or ""
        ).strip()
        current_rows.append(
            {
                "pair": pair,
                "periods": "scanner",
                "timeframe": "Live",
                "strategy": row_strategy,
                "best_fit": "",
                "conditional_chart": "",
                "pearson": "",
                "spearman": "",
                "kendall": "",
                "sharpe_metric": sharpe,
                "sortino_metric": "",
                "net_return_metric": returns,
                "annualized_return_metric": returns,
                "win_rate_metric": "",
                "closed_trades_metric": "",
                "max_drawdown_metric": max_drawdown,
                "top_strip": {
                    "corr": correlation,
                    "hurst": hurst,
                    "half_life": half_life,
                    "returns": returns,
                    "sharpe": sharpe,
                },
            }
        )
    return current_rows


def _read_csv_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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
