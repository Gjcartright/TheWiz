from __future__ import annotations

from dataclasses import replace
from typing import Any

import pandas as pd

from quant_platform.experiments import PairDataset


FUNDING_MARKET_COLUMNS = (
    "market",
    "ticker",
    "symbol",
    "market_id",
    "marketId",
    "product_id",
    "productId",
)
FUNDING_TIME_COLUMNS = (
    "timestamp",
    "time",
    "effective_at",
    "effectiveAt",
    "created_at",
    "createdAt",
    "scrapedAt",
    "updatedAt",
    "updated_at",
    "nextFundingTime",
    "next_funding_time",
    "time_ms",
    "timeMs",
)
FUNDING_VALUE_COLUMNS = (
    "funding_bps",
    "funding_rate_bps",
    "rate_bps",
    "rate",
    "funding_rate",
    "nextFundingRate",
    "next_funding",
    "next_funding_rate",
)
FUNDING_RECORD_CONTAINERS = (
    "historicalFunding",
    "funding",
    "payload",
    "data",
    "rows",
    "items",
    "results",
    "result",
    "markets",
    "perpetualMarkets",
)


def funding_market_requirements(pairs: list[str] | tuple[str, ...] | set[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pair in sorted({str(pair) for pair in pairs if str(pair)}):
        try:
            left_asset, right_asset = _split_pair(pair)
            market_x = _dydx_market(left_asset)
            market_y = _dydx_market(right_asset)
            rows.append(
                {
                    "pair": pair,
                    "market_x": market_x,
                    "market_y": market_y,
                    "required_markets": f"{market_x};{market_y}",
                    "valid": True,
                    "error": "",
                }
            )
        except ValueError as exc:
            rows.append(
                {
                    "pair": pair,
                    "market_x": "",
                    "market_y": "",
                    "required_markets": "",
                    "valid": False,
                    "error": str(exc),
                }
            )
    return pd.DataFrame(rows)


def funding_coverage_for_pairs(
    pairs: list[str] | tuple[str, ...] | set[str],
    funding_rows: pd.DataFrame | list[dict[str, Any]],
) -> pd.DataFrame:
    funding = normalize_funding_rows(funding_rows)
    market_counts = funding["market"].value_counts().to_dict() if not funding.empty else {}
    timestamped_counts = (
        funding.dropna(subset=["timestamp"])["market"].value_counts().to_dict()
        if not funding.empty and "timestamp" in funding.columns
        else {}
    )
    rows: list[dict[str, Any]] = []
    for pair in sorted({str(pair) for pair in pairs if str(pair)}):
        try:
            left_asset, right_asset = _split_pair(pair)
            market_x = _dydx_market(left_asset)
            market_y = _dydx_market(right_asset)
            count_x = int(market_counts.get(market_x, 0))
            count_y = int(market_counts.get(market_y, 0))
            timestamped_x = int(timestamped_counts.get(market_x, 0))
            timestamped_y = int(timestamped_counts.get(market_y, 0))
            missing = []
            if count_x == 0:
                missing.append("funding_x")
            if count_y == 0:
                missing.append("funding_y")
            missing_markets = []
            if count_x == 0:
                missing_markets.append(market_x)
            if count_y == 0:
                missing_markets.append(market_y)
            rows.append(
                {
                    "pair": pair,
                    "market_x": market_x,
                    "market_y": market_y,
                    "funding_x_rows": count_x,
                    "funding_y_rows": count_y,
                    "funding_x_timestamped_rows": timestamped_x,
                    "funding_y_timestamped_rows": timestamped_y,
                    "funding_x_available": count_x > 0,
                    "funding_y_available": count_y > 0,
                    "ready": count_x > 0 and count_y > 0,
                    "missing": ";".join(missing),
                    "missing_markets": ";".join(missing_markets),
                    "required_markets": f"{market_x};{market_y}",
                }
            )
        except ValueError as exc:
            rows.append(
                {
                    "pair": pair,
                    "market_x": "",
                    "market_y": "",
                    "funding_x_rows": 0,
                    "funding_y_rows": 0,
                    "funding_x_timestamped_rows": 0,
                    "funding_y_timestamped_rows": 0,
                    "funding_x_available": False,
                    "funding_y_available": False,
                    "ready": False,
                    "missing": str(exc),
                    "missing_markets": "",
                    "required_markets": "",
                }
            )
    return pd.DataFrame(rows)


def enrich_pair_dataset_with_funding(dataset: PairDataset, funding_rows: pd.DataFrame | list[dict[str, Any]]) -> PairDataset:
    funding = normalize_funding_rows(funding_rows)
    if funding.empty:
        return dataset
    left_asset, right_asset = _split_pair(dataset.pair)
    frame = dataset.frame.copy()
    frame = _attach_leg_funding(frame, funding, _dydx_market(left_asset), "funding_x_bps")
    frame = _attach_leg_funding(frame, funding, _dydx_market(right_asset), "funding_y_bps")
    return replace(dataset, frame=frame)


def normalize_funding_rows(rows: pd.DataFrame | list[dict[str, Any]]) -> pd.DataFrame:
    raw = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if raw.empty:
        return pd.DataFrame(columns=["market", "timestamp", "funding_bps"])
    market_column = _first_present(raw, FUNDING_MARKET_COLUMNS)
    value_column = _first_present(raw, FUNDING_VALUE_COLUMNS)
    if market_column is None or value_column is None:
        return pd.DataFrame(columns=["market", "timestamp", "funding_bps"])
    normalized = pd.DataFrame()
    normalized["market"] = raw[market_column].astype(str).map(_dydx_market)
    normalized["funding_bps"] = pd.to_numeric(raw[value_column], errors="coerce")
    if value_column not in {"funding_bps", "funding_rate_bps", "rate_bps"}:
        normalized["funding_bps"] = normalized["funding_bps"] * 10_000.0
    time_column = _first_present(raw, FUNDING_TIME_COLUMNS)
    normalized["timestamp"] = (
        _normalize_funding_timestamps(raw[time_column]) if time_column else pd.NaT
    )
    return normalized.dropna(subset=["market", "funding_bps"]).sort_values(["market", "timestamp"], na_position="last")


def funding_rows_from_dydx_payload(payload: Any, market: str | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    payload_market = _first_value(payload, FUNDING_MARKET_COLUMNS) if isinstance(payload, dict) else None
    for item in _funding_record_candidates(payload):
        record = dict(item)
        inferred_market = _first_value(record, FUNDING_MARKET_COLUMNS) or market or payload_market
        if inferred_market and not _first_value(record, FUNDING_MARKET_COLUMNS):
            record["market"] = inferred_market
        rows.append(record)
    return rows


def _attach_leg_funding(frame: pd.DataFrame, funding: pd.DataFrame, market: str, output_column: str) -> pd.DataFrame:
    if output_column in frame.columns:
        return frame
    market_funding = funding[funding["market"] == market].copy()
    if market_funding.empty:
        return frame
    output = frame.copy()
    if "timestamp" not in output.columns or market_funding["timestamp"].isna().all():
        output[output_column] = float(market_funding["funding_bps"].dropna().iloc[-1])
        return output
    left = output.reset_index().rename(columns={"index": "_row_index"})
    left["_timestamp"] = pd.to_datetime(left["timestamp"], errors="coerce", utc=True).astype("datetime64[ns, UTC]")
    right = market_funding.dropna(subset=["timestamp"]).sort_values("timestamp")
    right["_timestamp"] = pd.to_datetime(right["timestamp"], errors="coerce", utc=True).astype("datetime64[ns, UTC]")
    if right.empty:
        output[output_column] = float(market_funding["funding_bps"].dropna().iloc[-1])
        return output
    merged = pd.merge_asof(
        left.sort_values("_timestamp"),
        right[["_timestamp", "funding_bps"]].sort_values("_timestamp"),
        on="_timestamp",
        direction="backward",
    ).sort_values("_row_index")
    output[output_column] = merged["funding_bps"].ffill().bfill().to_numpy()
    return output


def _first_present(frame: pd.DataFrame, columns: tuple[str, ...]) -> str | None:
    for column in columns:
        if column in frame.columns:
            return column
    return None


def _funding_record_candidates(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        records: list[dict[str, Any]] = []
        for item in payload:
            records.extend(_funding_record_candidates(item))
        return records
    if not isinstance(payload, dict):
        return []
    records: list[dict[str, Any]] = []
    for key in FUNDING_RECORD_CONTAINERS:
        value = payload.get(key)
        if isinstance(value, list):
            records.extend(item for item in value if isinstance(item, dict))
        elif isinstance(value, dict):
            records.extend(_funding_record_candidates(value))
    if _first_value(payload, FUNDING_VALUE_COLUMNS) is not None:
        records.append(payload)
    return records


def _parse_funding_timestamp(value: Any) -> pd.Timestamp | pd.NaT:
    if value is None or value == "":
        return pd.NaT
    if isinstance(value, (int, float)) and not pd.isna(value):
        if value > 1e15:
            return pd.to_datetime(value, unit="ns", utc=True, errors="coerce")
        if value > 1e12:
            return pd.to_datetime(value, unit="ms", utc=True, errors="coerce")
        if value > 1e9:
            return pd.to_datetime(value, unit="s", utc=True, errors="coerce")
        return pd.NaT
    return pd.to_datetime(value, errors="coerce", utc=True)


def _normalize_funding_timestamps(values: pd.Series) -> pd.Series:
    if values.empty:
        return pd.Series(dtype="datetime64[ns, UTC]")
    if pd.api.types.is_datetime64_any_dtype(values.dtype):
        return pd.to_datetime(values, utc=True)
    numeric = pd.to_numeric(values, errors="coerce")
    timestamps = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns, UTC]")
    numeric_mask = numeric.notna()
    if numeric_mask.any():
        ns_mask = numeric_mask & (numeric > 1e15)
        ms_mask = numeric_mask & (numeric <= 1e15) & (numeric > 1e12)
        s_mask = numeric_mask & (numeric <= 1e12) & (numeric > 1e9)
        if ns_mask.any():
            timestamps.loc[ns_mask] = pd.to_datetime(numeric.loc[ns_mask], unit="ns", utc=True, errors="coerce")
        if ms_mask.any():
            timestamps.loc[ms_mask] = pd.to_datetime(numeric.loc[ms_mask], unit="ms", utc=True, errors="coerce")
        if s_mask.any():
            timestamps.loc[s_mask] = pd.to_datetime(numeric.loc[s_mask], unit="s", utc=True, errors="coerce")
    non_numeric_mask = ~numeric_mask
    if non_numeric_mask.any():
        timestamps.loc[non_numeric_mask] = pd.to_datetime(
            values.loc[non_numeric_mask],
            errors="coerce",
            utc=True,
            format="mixed",
        )
    return timestamps


def _first_value(record: dict[str, Any], columns: tuple[str, ...]) -> Any:
    for column in columns:
        value = record.get(column)
        if value not in (None, ""):
            return value
    return None


def _split_pair(pair: str) -> tuple[str, str]:
    parts = pair.replace("/", "-").split("-")
    if len(parts) == 4 and parts[1].upper() == "USD" and parts[3].upper() == "USD":
        return parts[0].upper(), parts[2].upper()
    if len(parts) == 2:
        return parts[0].upper(), parts[1].upper()
    raise ValueError(f"pair must be ASSET-ASSET or ASSET-USD-ASSET-USD: {pair}")


def _dydx_market(asset: str) -> str:
    normalized = str(asset).upper()
    if normalized.endswith("USDT"):
        normalized = normalized[:-4]
    if normalized.endswith("-USD"):
        return normalized
    return f"{normalized}-USD"
