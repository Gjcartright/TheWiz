"""Bounded, read-only multi-asset Hyperliquid observation collector.

This route is separate from the BTC/ETH pilot.  It retains a frozen liquid
universe, raw market metadata, and raw hourly candles without credentials or
trading authority.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

from quant_platform.orchestration.corrective_hyperliquid_network import (
    run_authorized_hyperliquid_info_call,
)
from quant_platform.orchestration.corrective_runtime import (
    create_exclusive_bytes,
    promote_staged_directory,
)
from quant_platform.orchestration.corrective_scheduler_lock import (
    current_publication_lease,
)
from quant_platform.orchestration.effect_authority import require_file_publication_target

INFO_URL = "https://api.hyperliquid.xyz/info"
SCHEMA_VERSION = "thewiz.multi_asset_capture.v1"
LOCAL_TIMEZONE = ZoneInfo("America/New_York")
MINIMUM_FREE_BYTES = 5 * 1024**3
HOUR_MS = 60 * 60 * 1000
CAPTURE_ID_RE = re.compile(r"^universe-\d{8}T\d{6}Z$")
Fetch = Callable[[dict[str, object]], bytes]


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _hash(value: bytes) -> str:
    return sha256(value).hexdigest()


def _raw_hyperliquid_info_call(payload: dict[str, object]) -> bytes:
    request = Request(
        INFO_URL,
        data=_canonical(payload),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=20) as response:
        if response.status != 200:
            raise ValueError(f"hyperliquid_public_http_status_{response.status}")
        return response.read()


def _authorized_fetch(payload: dict[str, object], fetch: Fetch) -> bytes:
    """Require a one-use public-network permit for each exact Hyperliquid read."""
    response = run_authorized_hyperliquid_info_call(
        target=INFO_URL,
        payload=payload,
        operation_prefix="HYPERLIQUID_MAINNET",
        transport=lambda: fetch(payload),
        result_recorder=_hash,
    )
    if not isinstance(response, bytes):
        raise TypeError("multi_asset_response_must_be_bytes")
    return response


def _parse_metadata(
    raw: bytes, *, minimum_notional: float, minimum_open_interest_notional: float, max_assets: int
) -> tuple[str, ...]:
    try:
        metadata, contexts = json.loads(raw)
        universe = metadata["universe"]
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError("multi_asset_metadata_invalid") from exc
    if (
        not isinstance(universe, list)
        or not isinstance(contexts, list)
        or len(universe) != len(contexts)
    ):
        raise ValueError("multi_asset_metadata_shape_invalid")
    ranked: list[tuple[float, str]] = []
    seen_names: set[str] = set()
    for item, context in zip(universe, contexts):
        if not isinstance(item, dict) or not isinstance(context, dict):
            raise TypeError("multi_asset_metadata_value_invalid")
        try:
            name_value = item["name"]
            numeric_values = (
                context["dayNtlVlm"],
                context["markPx"],
                context["openInterest"],
            )
            if (
                not isinstance(name_value, str)
                or not name_value
                or name_value != name_value.strip()
                or any(
                    isinstance(value, bool) or not isinstance(value, (str, int, float))
                    for value in numeric_values
                )
            ):
                raise ValueError("invalid_metadata_fields")
            name = name_value
            volume, mark, open_interest = (float(value) for value in numeric_values)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError("multi_asset_metadata_value_invalid") from exc
        if name in seen_names:
            raise ValueError("multi_asset_metadata_duplicate_asset")
        seen_names.add(name)
        open_interest_notional = open_interest * mark
        if (
            name
            and math.isfinite(volume)
            and math.isfinite(mark)
            and math.isfinite(open_interest)
            and math.isfinite(open_interest_notional)
            and volume >= minimum_notional
            and volume >= 0
            and mark > 0
            and open_interest >= 0
            and open_interest_notional >= minimum_open_interest_notional
        ):
            ranked.append((volume, name))
    assets = tuple(name for _, name in sorted(ranked, reverse=True)[:max_assets])
    if len(assets) < 2:
        raise ValueError("multi_asset_universe_insufficient")
    return assets


def _asset_raw_label(asset: str) -> str:
    """Encode API-provided asset names before using them in evidence paths."""
    return f"candles_{quote(asset, safe='')}"


def _read_receipt(root: Path, capture_id: str) -> dict[str, object]:
    if not isinstance(capture_id, str) or CAPTURE_ID_RE.fullmatch(capture_id) is None:
        raise ValueError("multi_asset_predecessor_missing_or_invalid")
    path = root / "data" / "research" / "multi_asset_captures" / capture_id / "receipt.json"
    try:
        receipt = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError("multi_asset_predecessor_missing_or_invalid") from exc
    if not isinstance(receipt, dict) or receipt.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("multi_asset_predecessor_missing_or_invalid")
    raw = receipt.get("raw")
    if not isinstance(raw, dict) or not raw:
        raise ValueError("multi_asset_predecessor_missing_or_invalid")
    parent = path.parent
    for label, entry in raw.items():
        if (
            not isinstance(label, str)
            or not label
            or Path(label).name != label
            or label in {".", ".."}
            or not isinstance(entry, dict)
        ):
            raise ValueError("multi_asset_predecessor_missing_or_invalid")
        expected_path = f"raw/{label}.json"
        if entry.get("path") != expected_path or not isinstance(entry.get("sha256"), str):
            raise ValueError("multi_asset_predecessor_missing_or_invalid")
        try:
            contents = (parent / expected_path).read_bytes()
        except OSError as exc:
            raise ValueError("multi_asset_predecessor_missing_or_invalid") from exc
        if _hash(contents) != entry["sha256"]:
            raise ValueError("multi_asset_predecessor_raw_hash_mismatch")
    return receipt


def _decimal_field(row: dict[str, object], key: str) -> Decimal:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise TypeError("multi_asset_candle_value_invalid")
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("multi_asset_candle_value_invalid") from exc
    if not number.is_finite():
        raise ValueError("multi_asset_candle_value_invalid")
    return number


def _validate_candle_payload(
    raw: bytes, *, asset: str, observed: datetime, interval: str = "1h"
) -> None:
    try:
        rows = json.loads(raw)
    except ValueError as exc:
        raise ValueError("multi_asset_candle_not_json") from exc
    if not isinstance(rows, list) or not rows:
        raise ValueError("multi_asset_candle_invalid")
    previous_start = -1
    closed = 0
    cutoff = int(observed.timestamp() * 1000)
    for row in rows:
        if not isinstance(row, dict) or row.get("s") != asset or row.get("i") != interval:
            raise ValueError("multi_asset_candle_identity_invalid")
        start_value, end_value, trade_value = row.get("t"), row.get("T"), row.get("n")
        if (
            isinstance(start_value, bool)
            or not isinstance(start_value, int)
            or isinstance(end_value, bool)
            or not isinstance(end_value, int)
            or isinstance(trade_value, bool)
            or not isinstance(trade_value, int)
        ):
            raise TypeError("multi_asset_candle_value_invalid")
        start, end = start_value, end_value
        try:
            open_px, high, low, close = (_decimal_field(row, key) for key in ("o", "h", "l", "c"))
            volume = _decimal_field(row, "v")
        except (TypeError, ValueError) as exc:
            raise ValueError("multi_asset_candle_value_invalid") from exc
        if (
            start <= previous_start
            or start % HOUR_MS != 0
            or end != start + HOUR_MS - 1
            or close <= 0
            or open_px <= 0
            or high <= 0
            or low <= 0
            or volume < 0
            or trade_value < 0
            or high < max(open_px, close, low)
            or low > min(open_px, close, high)
        ):
            raise ValueError("multi_asset_candle_value_invalid")
        previous_start = start
        if end <= cutoff:
            closed += 1
    if closed < 2:
        raise ValueError("multi_asset_closed_candles_insufficient")


def _closed_candle_map(
    raw: bytes, *, asset: str, observed: datetime
) -> dict[int, tuple[Decimal, Decimal, Decimal, Decimal, Decimal, int]]:
    """Return every material candle field for rows closed at the observation time."""
    _validate_candle_payload(raw, asset=asset, observed=observed)
    cutoff = int(observed.timestamp() * 1000)
    rows = json.loads(raw)
    return {
        int(row["t"]): tuple(_decimal_field(row, key) for key in ("o", "h", "l", "c", "v"))
        + (int(row["n"]),)
        for row in rows
        if int(row["T"]) <= cutoff
    }


def _prior_raw(root: Path, capture_id: str, label: str) -> bytes | None:
    receipt = _read_receipt(root, capture_id)
    index = receipt["raw"]
    entry = index.get(label) if isinstance(index, dict) else None
    if not isinstance(entry, dict):
        return None
    path = root / "data" / "research" / "multi_asset_captures" / capture_id / str(entry["path"])
    return path.read_bytes()


def _validate_full_predecessor_chain(root: Path, terminal_id: str) -> tuple[str, ...]:
    current_id: str | None = terminal_id
    seen: set[str] = set()
    chain: list[str] = []
    newest_time: datetime | None = None
    while current_id:
        if current_id in seen:
            raise ValueError("multi_asset_predecessor_cycle")
        seen.add(current_id)
        chain.append(current_id)
        receipt = _read_receipt(root, current_id)
        if receipt.get("capture_id") != current_id:
            raise ValueError("multi_asset_predecessor_missing_or_invalid")
        try:
            parsed_time = datetime.fromisoformat(str(receipt["observed_at_utc"]))
            if parsed_time.tzinfo is None or parsed_time.utcoffset() is None:
                raise ValueError("naive_timestamp")
            current_time = parsed_time.astimezone(UTC)
        except (ValueError, TypeError) as exc:
            raise ValueError("multi_asset_predecessor_missing_or_invalid") from exc
        if newest_time is not None and current_time >= newest_time:
            raise ValueError("multi_asset_predecessor_time_invalid")
        newest_time = current_time
        predecessor = receipt.get("prior_capture_id")
        if predecessor is None:
            return tuple(chain)
        if not isinstance(predecessor, str) or not predecessor:
            raise ValueError("multi_asset_predecessor_missing_or_invalid")
        expected = receipt.get("prior_receipt_sha256")
        prior = _read_receipt(root, predecessor)
        if expected != _hash(_canonical(prior)):
            raise ValueError("multi_asset_predecessor_link_invalid")
        current_id = predecessor


def capture_multi_asset_universe(
    *,
    root: Path,
    prior_capture_id: str | None = None,
    now: datetime | None = None,
    fetch: Fetch | None = None,
    max_assets: int = 25,
    minimum_notional: float = 10_000_000.0,
    minimum_open_interest_notional: float = 5_000_000.0,
    interval: str = "1h",
) -> dict[str, object]:
    """Capture one frozen liquid universe and its raw hourly candles."""

    if (
        isinstance(max_assets, bool)
        or not isinstance(max_assets, int)
        or not 2 <= max_assets <= 30
        or isinstance(minimum_notional, bool)
        or not isinstance(minimum_notional, (int, float))
        or isinstance(minimum_open_interest_notional, bool)
        or not isinstance(minimum_open_interest_notional, (int, float))
        or minimum_notional <= 0
        or minimum_open_interest_notional <= 0
        or not math.isfinite(minimum_notional)
        or not math.isfinite(minimum_open_interest_notional)
        or interval != "1h"
    ):
        raise ValueError("multi_asset_capture_configuration_invalid")
    raw_observed = now or datetime.now(UTC)
    if raw_observed.tzinfo is None or raw_observed.utcoffset() is None:
        raise ValueError("multi_asset_capture_time_must_be_timezone_aware")
    observed = raw_observed.astimezone(UTC)
    if shutil.disk_usage(root).free < MINIMUM_FREE_BYTES:
        raise ValueError("multi_asset_insufficient_free_storage")
    prior: dict[str, object] | None = None
    prior_time: datetime | None = None
    if prior_capture_id:
        prior = _read_receipt(root, prior_capture_id)
        try:
            parsed_time = datetime.fromisoformat(str(prior["observed_at_utc"]))
            if parsed_time.tzinfo is None or parsed_time.utcoffset() is None:
                raise ValueError("naive_timestamp")
            prior_time = parsed_time.astimezone(UTC)
        except ValueError as exc:
            raise ValueError("multi_asset_predecessor_missing_or_invalid") from exc
        if prior_time >= observed:
            raise ValueError("multi_asset_nonforward_observation")
        _validate_full_predecessor_chain(root, prior_capture_id)
        prior_hash = _hash(_canonical(prior))
    else:
        prior_hash = None
    capture_id = f"universe-{observed.strftime('%Y%m%dT%H%M%SZ')}"
    base = root / "data" / "research" / "multi_asset_captures"
    target = base / capture_id
    if target.exists() or target.is_symlink():
        raise FileExistsError("multi_asset_capture_target_exists")
    if base.is_symlink():
        raise ValueError("multi_asset_capture_store_must_not_be_symlink")
    existing_capture_ids: list[str] = []
    if base.exists():
        for entry in base.glob("universe-*"):
            if entry.is_symlink() or not entry.is_dir():
                raise ValueError("multi_asset_capture_entry_invalid")
            existing_capture_ids.append(entry.name)
    if existing_capture_ids:
        current_terminal = max(existing_capture_ids)
        if prior_capture_id is None:
            raise ValueError("multi_asset_predecessor_required")
        if prior_capture_id != current_terminal:
            raise ValueError("multi_asset_predecessor_not_terminal")
    lease = current_publication_lease()
    publication_scope = lease.scope if lease is not None else "research"
    require_file_publication_target(
        root=root,
        target=target,
        publication_scope=publication_scope,
    )
    selected_fetch = fetch or _raw_hyperliquid_info_call
    metadata_raw = _authorized_fetch({"type": "metaAndAssetCtxs"}, selected_fetch)
    selected = _parse_metadata(
        metadata_raw,
        minimum_notional=minimum_notional,
        minimum_open_interest_notional=minimum_open_interest_notional,
        max_assets=max_assets,
    )
    if (
        prior_capture_id
        and prior_time is not None
        and prior_time.astimezone(LOCAL_TIMEZONE).date()
        == observed.astimezone(LOCAL_TIMEZONE).date()
    ):
        prior_assets = prior.get("assets")
        if not isinstance(prior_assets, list | tuple) or not all(
            isinstance(asset, str) for asset in prior_assets
        ):
            raise ValueError("multi_asset_predecessor_missing_or_invalid")
        assets = tuple(prior_assets)
    else:
        assets = selected
    end_ms = int(observed.timestamp() * 1000)
    start_ms = int((observed - timedelta(days=35)).timestamp() * 1000)
    requests = {
        asset: {
            "type": "candleSnapshot",
            "req": {"coin": asset, "interval": interval, "startTime": start_ms, "endTime": end_ms},
        }
        for asset in assets
    }
    raw: dict[str, bytes] = {"metadata": metadata_raw}
    for asset, request in requests.items():
        value = _authorized_fetch(request, selected_fetch)
        _validate_candle_payload(value, asset=asset, observed=observed, interval=interval)
        if prior_capture_id:
            prior_value = _prior_raw(root, prior_capture_id, _asset_raw_label(asset))
            if prior_value is not None:
                # Only candles closed when the predecessor was observed are
                # immutable comparison points. Open candles may legitimately
                # change before closing, but all OHLCV/trade-count fields of a
                # prior closed candle must remain unchanged.
                previous = _closed_candle_map(prior_value, asset=asset, observed=prior_time)
                current = _closed_candle_map(value, asset=asset, observed=observed)
                overlapping_previous = {
                    stamp: fields for stamp, fields in previous.items() if stamp >= start_ms
                }
                if any(
                    stamp not in current or current[stamp] != fields
                    for stamp, fields in overlapping_previous.items()
                ):
                    raise ValueError("multi_asset_closed_candle_revision_unexplained")
        raw[_asset_raw_label(asset)] = value

    base.mkdir(mode=0o700, parents=True, exist_ok=True)
    if base.is_symlink():
        raise ValueError("multi_asset_capture_store_must_not_be_symlink")
    stage = base / f".{capture_id}.staging-{uuid.uuid4().hex}"
    stage.mkdir(mode=0o700)
    raw_dir = stage / "raw"
    raw_dir.mkdir(mode=0o700)
    raw_index: dict[str, dict[str, str]] = {}
    for label, contents in sorted(raw.items()):
        path = raw_dir / f"{label}.json"
        create_exclusive_bytes(
            path,
            contents,
            mode=0o600,
            publication_scope=publication_scope,
        )
        os.chmod(path, 0o400)
        raw_index[label] = {"path": f"raw/{label}.json", "sha256": _hash(contents)}
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "capture_id": capture_id,
        "observed_at_utc": observed.isoformat(),
        "prior_capture_id": prior_capture_id,
        "prior_receipt_sha256": prior_hash,
        "assets": assets,
        "maximum_assets": max_assets,
        "minimum_24h_notional": minimum_notional,
        "minimum_open_interest_notional": minimum_open_interest_notional,
        "interval": interval,
        "candidate_pair_count": len(assets) * (len(assets) - 1) // 2,
        "raw": raw_index,
        "authority": {
            "credentials_used": False,
            "orders": 0,
            "paper_trading": False,
            "execution": False,
            "wizard_credits": 0,
        },
    }
    receipt_path = stage / "receipt.json"
    create_exclusive_bytes(
        receipt_path,
        _canonical(receipt) + b"\n",
        mode=0o600,
        publication_scope=publication_scope,
    )
    os.chmod(receipt_path, 0o400)
    os.chmod(raw_dir, 0o500)
    _fsync_directory(raw_dir)
    _fsync_directory(stage)
    if target.exists() or target.is_symlink():
        raise FileExistsError("multi_asset_capture_target_exists")
    promote_staged_directory(stage, target, publication_scope=publication_scope)
    os.chmod(target, 0o500)
    _fsync_directory(target)
    _fsync_directory(base)
    return {
        "capture_id": capture_id,
        "assets": assets,
        "candidate_pair_count": receipt["candidate_pair_count"],
        "network_calls": len(raw),
        "wizard_credits": 0,
        "credentials_used": False,
        "execution": False,
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture a bounded liquid Hyperliquid universe")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prior-capture-id")
    args = parser.parse_args()
    print(
        json.dumps(
            capture_multi_asset_universe(
                root=args.root.resolve(), prior_capture_id=args.prior_capture_id
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
