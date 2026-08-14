"""Prospective, zero-authority Testnet readiness for the Stage 4 cohort."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from decimal import ROUND_CEILING, Decimal
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd

from quant_platform.active_pipeline import CommandResult

ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "thewiz.testnet_prospective_cohort_readiness.v1"
INVENTORY_MAX_AGE = pd.Timedelta(hours=24)
COST_MODEL_MAX_AGE = pd.Timedelta(hours=2)
TARGET_NOTIONAL_PER_LEG_USD = Decimal("10.5")
MIN_NOTIONAL_PER_LEG_USD = 10.0
MAX_NOTIONAL_PER_LEG_USD = 12.5
MAX_PAIR_NOTIONAL_USD = 25.0

READINESS_COLUMNS = (
    "schema_version",
    "evaluated_at_utc",
    "pair_group_key",
    "pair",
    "asset_x",
    "asset_y",
    "registered_contract_candidate",
    "stage_two_candidate",
    "cost_model_id",
    "cost_model_as_of_utc",
    "strict_cost_model_current",
    "funding_history_x_complete",
    "funding_history_y_complete",
    "funding_evidence_scope",
    "inventory_checked_at_utc",
    "inventory_current",
    "asset_x_tradable_perp",
    "asset_y_tradable_perp",
    "asset_x_index",
    "asset_y_index",
    "asset_x_size_decimals",
    "asset_y_size_decimals",
    "asset_x_max_leverage",
    "asset_y_max_leverage",
    "pair_max_leverage",
    "asset_x_reference_price",
    "asset_y_reference_price",
    "asset_x_indicative_size",
    "asset_y_indicative_size",
    "asset_x_indicative_notional_usd",
    "asset_y_indicative_notional_usd",
    "indicative_pair_notional_usd",
    "paired_size_precision_feasible",
    "prospective_readiness_status",
    "blocker",
    "must_revalidate_after_candidate_selection",
    "candidate_selection_performed",
    "collateral_transfer_attempted",
    "order_submission_performed",
    "candidate_promotion_authority",
    "testnet_order_authority",
    "live_trading_authorized",
    "evidence_path",
)


def build_testnet_prospective_cohort_readiness(
    *, root: Path = ROOT, now: datetime | None = None
) -> CommandResult:
    """Prove cohort-level Testnet feasibility without selecting or trading a pair."""

    evaluated_at = _as_utc(now)
    active = root / "reports" / "active"
    candidate_path = active / "corrective_l2_capture_candidates.csv"
    inventory_path = active / "hyperliquid_testnet_market_inventory.csv"
    cost_path = root / "data" / "processed" / "hyperliquid_pair_cost_models.csv"
    candidates = _read_csv(candidate_path)
    inventory = _read_csv(inventory_path)
    costs = _read_csv(cost_path)
    source_paths = (candidate_path, inventory_path, cost_path)
    source_hashes = {
        _relative(path, root): _sha256_file(path)
        for path in source_paths
        if path.is_file()
    }

    rows: list[dict[str, Any]] = []
    if not candidates.empty:
        cohort = candidates.drop_duplicates(subset=["pair_group_key"], keep="last")
        cohort = cohort.sort_values(["pair_group_key", "pair"], kind="stable")
        for candidate in cohort.to_dict(orient="records"):
            rows.append(
                _build_pair_row(
                    candidate=candidate,
                    inventory=inventory,
                    costs=costs,
                    evaluated_at=evaluated_at,
                    evidence_path=";".join(source_hashes),
                )
            )

    frame = pd.DataFrame(rows, columns=READINESS_COLUMNS)
    ready_pairs = int(
        frame.get("prospective_readiness_status", pd.Series(dtype=str))
        .astype(str)
        .eq("PASS_PROSPECTIVE_NO_AUTHORITY")
        .sum()
    )
    blockers: list[str] = []
    for path in source_paths:
        if not path.is_file():
            blockers.append(f"required_source_missing:{_relative(path, root)}")
    if frame.empty:
        blockers.append("stage4_research_cohort_missing")
    if ready_pairs != len(frame):
        blockers.append("one_or_more_cohort_pairs_not_prospectively_ready")
    status = (
        "PASS_PROSPECTIVE_NO_AUTHORITY"
        if not blockers and ready_pairs == len(frame) and ready_pairs > 0
        else "BLOCKED"
    )

    receipt_core = {
        "schema_version": SCHEMA_VERSION,
        "evaluated_at_utc": evaluated_at.isoformat(),
        "status": status,
        "cohort_pairs": len(frame),
        "prospectively_ready_pairs": ready_pairs,
        "required_revalidation": (
            "candidate_identity,current_testnet_inventory,current_testnet_funding,"
            "current_prices,paired_order_sizing,margin,and_explicit_one_run_approval"
        ),
        "source_artifact_hashes": source_hashes,
        "rows": rows,
        "blockers": list(dict.fromkeys(blockers)),
        "research_only": True,
        "candidate_selection_performed": False,
        "collateral_transfer_attempted": False,
        "order_submission_performed": False,
        "candidate_promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    receipt_id = "testnetcohortreadiness_" + _payload_hash(receipt_core)[:20]
    receipt = {**receipt_core, "receipt_id": receipt_id}
    receipt["receipt_sha256"] = _payload_hash(receipt)
    immutable_path = (
        root
        / "data"
        / "testnet"
        / "prospective_cohort_readiness"
        / f"{receipt_id}.json"
    )
    _write_immutable_json(receipt, immutable_path)

    csv_path = active / "testnet_prospective_cohort_readiness.csv"
    _atomic_csv(frame, csv_path)
    pointer = {
        "schema_version": SCHEMA_VERSION,
        "evaluated_at_utc": evaluated_at.isoformat(),
        "status": status,
        "receipt_id": receipt_id,
        "receipt_path": _relative(immutable_path, root),
        "receipt_file_sha256": _sha256_file(immutable_path),
        "receipt_sha256": receipt["receipt_sha256"],
        "active_csv_path": _relative(csv_path, root),
        "active_csv_sha256": _sha256_file(csv_path),
        "cohort_pairs": len(frame),
        "prospectively_ready_pairs": ready_pairs,
        "blockers": list(dict.fromkeys(blockers)),
        "research_only": True,
        "candidate_selection_performed": False,
        "collateral_transfer_attempted": False,
        "order_submission_performed": False,
        "candidate_promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    pointer_path = active / "testnet_prospective_cohort_readiness.json"
    _atomic_json(pointer, pointer_path)
    return CommandResult(
        paths={
            "prospective_cohort_readiness": csv_path,
            "prospective_cohort_readiness_receipt": immutable_path,
            "prospective_cohort_readiness_pointer": pointer_path,
        },
        summary=pointer,
    )


def _build_pair_row(
    *,
    candidate: dict[str, Any],
    inventory: pd.DataFrame,
    costs: pd.DataFrame,
    evaluated_at: datetime,
    evidence_path: str,
) -> dict[str, Any]:
    pair_group_key = str(candidate.get("pair_group_key", "")).strip()
    asset_x = str(candidate.get("asset_x", "")).strip().upper()
    asset_y = str(candidate.get("asset_y", "")).strip().upper()
    blockers: list[str] = []
    leg_x = _market_leg(inventory=inventory, asset=asset_x, evaluated_at=evaluated_at)
    leg_y = _market_leg(inventory=inventory, asset=asset_y, evaluated_at=evaluated_at)
    blockers.extend(leg_x["blockers"])
    blockers.extend(leg_y["blockers"])
    inventory_current = bool(leg_x["current"] and leg_y["current"])

    cost = _latest_cost(costs=costs, pair_group_key=pair_group_key)
    cost_time = pd.to_datetime(cost.get("model_as_of_utc"), utc=True, errors="coerce")
    strict_cost_current = bool(
        cost
        and str(cost.get("cost_model_status", "")) == "STRICT_OBSERVED"
        and _truthy(cost.get("strict_observed_cost_ready"))
        and pd.notna(cost_time)
        and cost_time <= pd.Timestamp(evaluated_at)
        and pd.Timestamp(evaluated_at) - cost_time <= COST_MODEL_MAX_AGE
    )
    if not cost:
        blockers.append(f"strict_pair_cost_model_missing:{pair_group_key}")
    elif not strict_cost_current:
        blockers.append(f"strict_pair_cost_model_stale_or_unready:{pair_group_key}")
    funding_x = _truthy(cost.get("funding_x_complete"))
    funding_y = _truthy(cost.get("funding_y_complete"))
    if not funding_x:
        blockers.append(f"funding_history_incomplete:{asset_x}")
    if not funding_y:
        blockers.append(f"funding_history_incomplete:{asset_y}")

    size_x, notional_x = _indicative_size(
        price=leg_x["price"], size_decimals=leg_x["size_decimals"]
    )
    size_y, notional_y = _indicative_size(
        price=leg_y["price"], size_decimals=leg_y["size_decimals"]
    )
    total_notional = notional_x + notional_y
    sizing_feasible = bool(
        _finite(size_x)
        and _finite(size_y)
        and MIN_NOTIONAL_PER_LEG_USD <= notional_x <= MAX_NOTIONAL_PER_LEG_USD
        and MIN_NOTIONAL_PER_LEG_USD <= notional_y <= MAX_NOTIONAL_PER_LEG_USD
        and total_notional <= MAX_PAIR_NOTIONAL_USD
    )
    if not sizing_feasible:
        blockers.append("indicative_two_leg_size_precision_not_feasible")
    pair_max_leverage = min(
        _finite(leg_x["max_leverage"]), _finite(leg_y["max_leverage"])
    )
    if not math.isfinite(pair_max_leverage) or pair_max_leverage < 1.0:
        blockers.append("one_x_pair_leverage_not_supported")

    status = (
        "PASS_PROSPECTIVE_NO_AUTHORITY"
        if not blockers
        else "BLOCKED"
    )
    inventory_times = [leg_x["checked_at"], leg_y["checked_at"]]
    inventory_checked = max(
        (value for value in inventory_times if value), default=""
    )
    return _json_safe({
        "schema_version": SCHEMA_VERSION,
        "evaluated_at_utc": evaluated_at.isoformat(),
        "pair_group_key": pair_group_key,
        "pair": str(candidate.get("pair", "")).strip(),
        "asset_x": asset_x,
        "asset_y": asset_y,
        "registered_contract_candidate": _truthy(
            candidate.get("registered_contract_candidate")
        ),
        "stage_two_candidate": _truthy(candidate.get("stage_two_candidate")),
        "cost_model_id": str(cost.get("cost_model_id", "")).strip(),
        "cost_model_as_of_utc": str(cost.get("model_as_of_utc", "")).strip(),
        "strict_cost_model_current": strict_cost_current,
        "funding_history_x_complete": funding_x,
        "funding_history_y_complete": funding_y,
        "funding_evidence_scope": (
            "research_history_only_testnet_funding_must_refresh_after_selection"
        ),
        "inventory_checked_at_utc": inventory_checked,
        "inventory_current": inventory_current,
        "asset_x_tradable_perp": leg_x["tradable"],
        "asset_y_tradable_perp": leg_y["tradable"],
        "asset_x_index": leg_x["asset_index"],
        "asset_y_index": leg_y["asset_index"],
        "asset_x_size_decimals": leg_x["size_decimals"],
        "asset_y_size_decimals": leg_y["size_decimals"],
        "asset_x_max_leverage": leg_x["max_leverage"],
        "asset_y_max_leverage": leg_y["max_leverage"],
        "pair_max_leverage": pair_max_leverage,
        "asset_x_reference_price": leg_x["price"],
        "asset_y_reference_price": leg_y["price"],
        "asset_x_indicative_size": size_x,
        "asset_y_indicative_size": size_y,
        "asset_x_indicative_notional_usd": notional_x,
        "asset_y_indicative_notional_usd": notional_y,
        "indicative_pair_notional_usd": total_notional,
        "paired_size_precision_feasible": sizing_feasible,
        "prospective_readiness_status": status,
        "blocker": ";".join(dict.fromkeys(blockers)),
        "must_revalidate_after_candidate_selection": True,
        "candidate_selection_performed": False,
        "collateral_transfer_attempted": False,
        "order_submission_performed": False,
        "candidate_promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
        "evidence_path": evidence_path,
    })


def _market_leg(
    *, inventory: pd.DataFrame, asset: str, evaluated_at: datetime
) -> dict[str, Any]:
    blockers: list[str] = []
    rows = inventory.loc[
        inventory.get("asset", pd.Series(dtype=str)).astype(str).str.upper().eq(asset)
    ]
    if len(rows) != 1:
        blockers.append(f"testnet_market_identity_missing_or_ambiguous:{asset}")
        return {
            "current": False,
            "tradable": False,
            "asset_index": math.nan,
            "size_decimals": math.nan,
            "max_leverage": math.nan,
            "price": math.nan,
            "checked_at": "",
            "blockers": blockers,
        }
    row = rows.iloc[0].to_dict()
    checked = pd.to_datetime(row.get("checked_at_utc"), utc=True, errors="coerce")
    tradable = bool(
        str(row.get("market_type", "")).lower() == "perp"
        and _truthy(row.get("tradable_perp"))
        and not _truthy(row.get("is_delisted"))
        and not _has_text(row.get("fetch_blocker"))
    )
    current = bool(
        tradable
        and pd.notna(checked)
        and checked <= pd.Timestamp(evaluated_at)
        and pd.Timestamp(evaluated_at) - checked <= INVENTORY_MAX_AGE
    )
    if not tradable:
        blockers.append(f"testnet_perp_not_tradable:{asset}")
    if not current:
        blockers.append(f"testnet_market_inventory_stale_or_future:{asset}")
    size_decimals = _nonnegative_int(row.get("sz_decimals"))
    if size_decimals is None:
        blockers.append(f"testnet_size_precision_missing:{asset}")
    price = _finite(row.get("mid_price"))
    if not math.isfinite(price) or price <= 0:
        blockers.append(f"testnet_reference_price_missing:{asset}")
    max_leverage = _finite(row.get("max_leverage"))
    return {
        "current": current,
        "tradable": tradable,
        "asset_index": row.get("asset_index", math.nan),
        "size_decimals": size_decimals if size_decimals is not None else math.nan,
        "max_leverage": max_leverage,
        "price": price,
        "checked_at": checked.isoformat() if pd.notna(checked) else "",
        "blockers": blockers,
    }


def _latest_cost(*, costs: pd.DataFrame, pair_group_key: str) -> dict[str, Any]:
    rows = costs.loc[
        costs.get("pair_group_key", pd.Series(dtype=str))
        .astype(str)
        .eq(pair_group_key)
    ].copy()
    if rows.empty:
        return {}
    rows["_as_of"] = pd.to_datetime(
        rows.get("model_as_of_utc"), utc=True, errors="coerce"
    )
    return rows.sort_values("_as_of", kind="stable").iloc[-1].to_dict()


def _indicative_size(*, price: float, size_decimals: Any) -> tuple[float, float]:
    decimals = _nonnegative_int(size_decimals)
    if not math.isfinite(price) or price <= 0 or decimals is None:
        return math.nan, math.nan
    quantum = Decimal(1).scaleb(-decimals)
    raw_size = TARGET_NOTIONAL_PER_LEG_USD / Decimal(str(price))
    size = (raw_size / quantum).to_integral_value(rounding=ROUND_CEILING) * quantum
    size_value = float(size)
    return size_value, size_value * price


def _as_utc(value: datetime | None) -> datetime:
    timestamp = value or datetime.now(UTC)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    return timestamp.astimezone(UTC)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "pass", "ready"}


def _has_text(value: Any) -> bool:
    return bool(pd.notna(value) and str(value).strip())


def _finite(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return math.nan
    return parsed if math.isfinite(parsed) else math.nan


def _nonnegative_int(value: Any) -> int | None:
    parsed = _finite(value)
    if not math.isfinite(parsed) or parsed < 0 or not parsed.is_integer():
        return None
    return int(parsed)


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame()


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _payload_hash(payload: Any) -> str:
    return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_immutable_json(payload: dict[str, Any], path: Path) -> None:
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise ValueError("prospective Testnet cohort readiness receipt collision")
        return
    _atomic_json(payload, path)
