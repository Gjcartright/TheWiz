from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_platform.pair_detail_ingestion import extract_history_rows, load_pair_detail_payload, snapshot_from_payload


TRADE_TIMING_TEMPLATE_COLUMNS = [
    "trade_id",
    "pair",
    "side",
    "entry_timestamp_current",
    "exit_timestamp_current",
]


def write_trade_timing_template(output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=TRADE_TIMING_TEMPLATE_COLUMNS).to_csv(output, index=False)
    return output


def load_trade_timing_history(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    if source.suffix.lower() == ".json":
        payload = load_pair_detail_payload(source)
        snapshot = snapshot_from_payload(payload)
        frame = pd.DataFrame(extract_history_rows(payload))
        if frame.empty:
            return frame
        if "pair" not in frame.columns:
            frame["pair"] = snapshot.pair
    else:
        frame = pd.read_csv(source)
    if frame.empty:
        return frame
    timestamp_column = _history_timestamp_column(frame)
    if timestamp_column != "timestamp":
        frame = frame.rename(columns={timestamp_column: "timestamp"})
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame["zscore"] = pd.to_numeric(frame.get("zscore"), errors="coerce")
    if "pair" in frame.columns:
        frame["pair"] = frame["pair"].fillna("").astype(str).str.strip()
    else:
        frame["pair"] = ""
    if "spread" in frame.columns:
        frame["spread"] = pd.to_numeric(frame["spread"], errors="coerce")
    return frame.dropna(subset=["timestamp", "zscore"]).sort_values(["pair", "timestamp"]).reset_index(drop=True)


def trade_timing_comparison_report_frame(
    trades: pd.DataFrame,
    history: pd.DataFrame,
    *,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.0,
) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    required = {"pair", "entry_timestamp_current", "exit_timestamp_current"}
    missing = sorted(required.difference(trades.columns))
    if missing:
        raise ValueError(f"missing trade columns: {missing}")

    frame = trades.copy()
    frame["trade_id"] = frame.get("trade_id", pd.Series([""] * len(frame))).fillna("").astype(str)
    frame["pair"] = frame["pair"].fillna("").astype(str).str.strip()
    frame["side"] = frame.get("side", pd.Series([""] * len(frame))).fillna("").astype(str)
    frame["entry_timestamp_current"] = pd.to_datetime(frame["entry_timestamp_current"], utc=True, errors="coerce")
    frame["exit_timestamp_current"] = pd.to_datetime(frame["exit_timestamp_current"], utc=True, errors="coerce")

    rows: list[dict[str, object]] = []
    history = history.copy()
    history["pair"] = history.get("pair", pd.Series([""] * len(history))).fillna("").astype(str).str.strip()

    for index, trade in frame.iterrows():
        pair = str(trade.get("pair", "")).strip()
        pair_history = history if not pair else history[history["pair"].isin({"", pair})].copy()
        pair_history = pair_history.sort_values("timestamp").reset_index(drop=True)
        notes: list[str] = []
        if pair_history.empty:
            notes.append("missing_pair_history")
            rows.append(_base_trade_row(index, trade, pair, notes))
            continue

        current_entry_row = _last_row_at_or_before(pair_history, trade["entry_timestamp_current"])
        current_exit_row = _last_row_at_or_before(pair_history, trade["exit_timestamp_current"])
        side = _normalize_side(str(trade.get("side", "")).strip())
        if side == "unknown":
            side = _infer_side(current_entry_row)

        entry_row = _signal_entry_row(pair_history, trade["entry_timestamp_current"], side, entry_threshold)
        if entry_row is None:
            notes.append("entry_signal_not_found")
        exit_anchor = entry_row["timestamp"] if entry_row is not None else trade["entry_timestamp_current"]
        exit_row = _signal_exit_row(pair_history, exit_anchor, side, exit_threshold)
        if exit_row is None:
            notes.append("exit_signal_not_found")

        rows.append(
            {
                "trade_id": str(trade.get("trade_id", "")).strip() or f"trade_{index + 1}",
                "pair": pair,
                "side": side,
                "entry_timestamp_current": _iso_or_empty(trade["entry_timestamp_current"]),
                "entry_timestamp_signal_z2": _row_iso(entry_row),
                "exit_timestamp_current": _iso_or_empty(trade["exit_timestamp_current"]),
                "exit_timestamp_signal_mean": _row_iso(exit_row),
                "entry_delay_minutes_current_minus_signal": _timestamp_delta_minutes(
                    trade["entry_timestamp_current"], None if entry_row is None else entry_row["timestamp"]
                ),
                "exit_delay_minutes_current_minus_signal": _timestamp_delta_minutes(
                    trade["exit_timestamp_current"], None if exit_row is None else exit_row["timestamp"]
                ),
                "entry_zscore_current": _row_value(current_entry_row, "zscore"),
                "entry_zscore_signal_z2": _row_value(entry_row, "zscore"),
                "exit_zscore_current": _row_value(current_exit_row, "zscore"),
                "exit_zscore_signal_mean": _row_value(exit_row, "zscore"),
                "entry_spread_current": _row_value(current_entry_row, "spread"),
                "entry_spread_signal_z2": _row_value(entry_row, "spread"),
                "exit_spread_current": _row_value(current_exit_row, "spread"),
                "exit_spread_signal_mean": _row_value(exit_row, "spread"),
                "entry_spread_delta_current_minus_signal": _numeric_delta(
                    _row_value(current_entry_row, "spread"), _row_value(entry_row, "spread")
                ),
                "exit_spread_delta_current_minus_signal": _numeric_delta(
                    _row_value(current_exit_row, "spread"), _row_value(exit_row, "spread")
                ),
                "notes": ";".join(notes),
            }
        )
    return pd.DataFrame(rows)


def trade_timing_comparison_summary(report: pd.DataFrame) -> pd.DataFrame:
    if report.empty:
        return pd.DataFrame(
            [
                {
                    "trades": 0,
                    "entry_signals_found": 0,
                    "exit_signals_found": 0,
                    "avg_entry_delay_minutes": "",
                    "median_entry_delay_minutes": "",
                    "avg_exit_delay_minutes": "",
                    "median_exit_delay_minutes": "",
                    "late_entries": 0,
                    "late_exits": 0,
                }
            ]
        )
    entry_delay = pd.to_numeric(report["entry_delay_minutes_current_minus_signal"], errors="coerce")
    exit_delay = pd.to_numeric(report["exit_delay_minutes_current_minus_signal"], errors="coerce")
    return pd.DataFrame(
        [
            {
                "trades": int(len(report)),
                "entry_signals_found": int(report["entry_timestamp_signal_z2"].astype(str).str.len().gt(0).sum()),
                "exit_signals_found": int(report["exit_timestamp_signal_mean"].astype(str).str.len().gt(0).sum()),
                "avg_entry_delay_minutes": float(entry_delay.dropna().mean()) if entry_delay.notna().any() else "",
                "median_entry_delay_minutes": float(entry_delay.dropna().median()) if entry_delay.notna().any() else "",
                "avg_exit_delay_minutes": float(exit_delay.dropna().mean()) if exit_delay.notna().any() else "",
                "median_exit_delay_minutes": float(exit_delay.dropna().median()) if exit_delay.notna().any() else "",
                "late_entries": int(entry_delay.fillna(0).gt(0).sum()),
                "late_exits": int(exit_delay.fillna(0).gt(0).sum()),
            }
        ]
    )


def _history_timestamp_column(frame: pd.DataFrame) -> str:
    for candidate in ("timestamp", "timestamp_utc", "datetime", "time"):
        if candidate in frame.columns:
            return candidate
    raise ValueError("history file must include a timestamp column")


def _normalize_side(value: str) -> str:
    lowered = value.strip().lower()
    if lowered in {"long", "buy", "long_spread"}:
        return "long_spread"
    if lowered in {"short", "sell", "short_spread"}:
        return "short_spread"
    return "unknown"


def _infer_side(row: pd.Series | None) -> str:
    if row is None:
        return "unknown"
    zscore = pd.to_numeric(pd.Series([row.get("zscore")]), errors="coerce").iloc[0]
    if pd.isna(zscore):
        return "unknown"
    return "long_spread" if float(zscore) < 0 else "short_spread"


def _last_row_at_or_before(frame: pd.DataFrame, timestamp: object) -> pd.Series | None:
    if pd.isna(timestamp):
        return None
    eligible = frame[frame["timestamp"] <= timestamp]
    if eligible.empty:
        return None
    return eligible.iloc[-1]


def _signal_entry_row(frame: pd.DataFrame, timestamp: object, side: str, threshold: float) -> pd.Series | None:
    if pd.isna(timestamp):
        return None
    eligible = frame[frame["timestamp"] <= timestamp].copy()
    if eligible.empty:
        return None
    mask = _entry_mask(eligible["zscore"], side, threshold)
    crosses = mask & ~mask.shift(1, fill_value=False)
    crossed = eligible.loc[crosses]
    if not crossed.empty:
        return crossed.iloc[-1]
    active = eligible.loc[mask]
    if active.empty:
        return None
    return active.iloc[-1]


def _signal_exit_row(frame: pd.DataFrame, timestamp: object, side: str, threshold: float) -> pd.Series | None:
    if pd.isna(timestamp):
        return None
    eligible = frame[frame["timestamp"] >= timestamp].copy()
    if eligible.empty:
        return None
    mask = _exit_mask(eligible["zscore"], side, threshold)
    exiting = eligible.loc[mask]
    if exiting.empty:
        return None
    return exiting.iloc[0]


def _entry_mask(zscore: pd.Series, side: str, threshold: float) -> pd.Series:
    values = pd.to_numeric(zscore, errors="coerce")
    if side == "long_spread":
        return values <= -abs(threshold)
    if side == "short_spread":
        return values >= abs(threshold)
    return values.abs() >= abs(threshold)


def _exit_mask(zscore: pd.Series, side: str, threshold: float) -> pd.Series:
    values = pd.to_numeric(zscore, errors="coerce")
    boundary = abs(threshold)
    if side == "long_spread":
        return values >= -boundary
    if side == "short_spread":
        return values <= boundary
    return values.abs() <= boundary


def _iso_or_empty(value: object) -> str:
    if pd.isna(value):
        return ""
    return pd.Timestamp(value).isoformat()


def _row_iso(row: pd.Series | None) -> str:
    if row is None:
        return ""
    return _iso_or_empty(row.get("timestamp"))


def _row_value(row: pd.Series | None, column: str) -> float | str:
    if row is None or column not in row.index:
        return ""
    value = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
    return "" if pd.isna(value) else float(value)


def _timestamp_delta_minutes(current: object, signal: object) -> float | str:
    if pd.isna(current) or pd.isna(signal):
        return ""
    delta = pd.Timestamp(current) - pd.Timestamp(signal)
    return float(delta.total_seconds() / 60.0)


def _numeric_delta(current: float | str, signal: float | str) -> float | str:
    try:
        return float(current) - float(signal)
    except (TypeError, ValueError):
        return ""


def _base_trade_row(index: int, trade: pd.Series, pair: str, notes: list[str]) -> dict[str, object]:
    return {
        "trade_id": str(trade.get("trade_id", "")).strip() or f"trade_{index + 1}",
        "pair": pair,
        "side": _normalize_side(str(trade.get("side", "")).strip()),
        "entry_timestamp_current": _iso_or_empty(trade.get("entry_timestamp_current")),
        "entry_timestamp_signal_z2": "",
        "exit_timestamp_current": _iso_or_empty(trade.get("exit_timestamp_current")),
        "exit_timestamp_signal_mean": "",
        "entry_delay_minutes_current_minus_signal": "",
        "exit_delay_minutes_current_minus_signal": "",
        "entry_zscore_current": "",
        "entry_zscore_signal_z2": "",
        "exit_zscore_current": "",
        "exit_zscore_signal_mean": "",
        "entry_spread_current": "",
        "entry_spread_signal_z2": "",
        "exit_spread_current": "",
        "exit_spread_signal_mean": "",
        "entry_spread_delta_current_minus_signal": "",
        "exit_spread_delta_current_minus_signal": "",
        "notes": ";".join(notes),
    }
