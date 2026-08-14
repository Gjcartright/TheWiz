"""Shared daily credit contract for scheduled Crypto Wizards research lanes."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from quant_platform.active_pipeline import CommandResult
from quant_platform.crypto_wizards_sweep import (
    PRESCANNED_CREDIT_COST,
    WIZARD_CRYPTO_EXCHANGES,
    WIZARD_DISCOVERY_INTERVALS,
    WIZARD_DISCOVERY_PRIORITIES,
    WIZARD_DISCOVERY_STRATEGIES,
    build_wizard_sweep_cells,
)
from quant_platform.wizard_hyperliquid_mode_proof import (
    CUSTOM_SERIES_CREDIT_COST,
    DEFAULT_DAILY_CREDIT_LIMIT,
    DEFAULT_RESERVED_CREDITS,
    REQUEST_CAP,
)

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "thewiz.wizard_credit_budget.v1"
DEFAULT_PROOF_MAX_BATCHES = 10
COPULA_BEHAVIORAL_MAX_PROOFS = 4
COPULA_BEHAVIORAL_REPEATS = 2
COPULA_POST_CREDIT_COST = 1
OU_V3_PROSPECTIVE_MAX_PROOFS = 4
OU_V3_CUSTOM_SERIES_CREDIT_COST = CUSTOM_SERIES_CREDIT_COST
OU_V4_PROSPECTIVE_MAX_PROOFS = 8
OU_V4_CUSTOM_SERIES_CREDIT_COST = CUSTOM_SERIES_CREDIT_COST
OU_V5_PROSPECTIVE_MAX_PROOFS = 8
OU_V5_CUSTOM_SERIES_CREDIT_COST = CUSTOM_SERIES_CREDIT_COST
OU_V6_PROSPECTIVE_MAX_PROOFS = 8
OU_V6_CUSTOM_SERIES_CREDIT_COST = CUSTOM_SERIES_CREDIT_COST
WIZARD_DAILY_CREDIT_RESET_UTC = "00:00"


def build_wizard_credit_budget_contract(
    *,
    root: Path = ROOT,
    now: datetime | None = None,
    proof_max_batches: int = DEFAULT_PROOF_MAX_BATCHES,
    proof_batch_size: int = REQUEST_CAP,
    daily_credit_limit: int = DEFAULT_DAILY_CREDIT_LIMIT,
    reserved_credits: int = DEFAULT_RESERVED_CREDITS,
) -> CommandResult:
    """Prove all automatically scheduled Wizard lanes fit below the daily reserve."""

    if proof_max_batches <= 0:
        raise ValueError("proof_max_batches must be positive")
    if not 1 <= proof_batch_size <= REQUEST_CAP:
        raise ValueError(f"proof_batch_size must be between 1 and {REQUEST_CAP}")
    if daily_credit_limit <= 0:
        raise ValueError("daily_credit_limit must be positive")
    if not 0 <= reserved_credits < daily_credit_limit:
        raise ValueError("reserved_credits must be within the daily limit")

    generated_at = _as_utc(now)
    sweep_cells = build_wizard_sweep_cells(
        sweep_id="credit-budget-contract",
        exchanges=WIZARD_CRYPTO_EXCHANGES,
        intervals=WIZARD_DISCOVERY_INTERVALS,
        strategies=WIZARD_DISCOVERY_STRATEGIES,
        priorities=WIZARD_DISCOVERY_PRIORITIES,
    )
    discovery_credits = sum(cell.credit_cost for cell in sweep_cells)
    proof_request_slots = proof_max_batches * proof_batch_size
    proof_credits = proof_request_slots * CUSTOM_SERIES_CREDIT_COST
    copula_behavioral_requests = COPULA_BEHAVIORAL_MAX_PROOFS * COPULA_BEHAVIORAL_REPEATS
    copula_behavioral_credits = copula_behavioral_requests * COPULA_POST_CREDIT_COST
    ou_v3_prospective_credits = OU_V3_PROSPECTIVE_MAX_PROOFS * OU_V3_CUSTOM_SERIES_CREDIT_COST
    ou_v4_prospective_credits = OU_V4_PROSPECTIVE_MAX_PROOFS * OU_V4_CUSTOM_SERIES_CREDIT_COST
    ou_v5_prospective_credits = OU_V5_PROSPECTIVE_MAX_PROOFS * OU_V5_CUSTOM_SERIES_CREDIT_COST
    ou_v6_prospective_credits = OU_V6_PROSPECTIVE_MAX_PROOFS * OU_V6_CUSTOM_SERIES_CREDIT_COST
    available_after_reserve = daily_credit_limit - reserved_credits
    scheduled_ceiling = (
        discovery_credits
        + proof_credits
        + copula_behavioral_credits
        + ou_v3_prospective_credits
        + ou_v4_prospective_credits
        + ou_v5_prospective_credits
        + ou_v6_prospective_credits
    )
    headroom = available_after_reserve - scheduled_ceiling
    contract_passes = headroom >= 0
    rows = [
        {
            "lane": "exhaustive_discovery_sweep",
            "scheduled": True,
            "maximum_requests": len(sweep_cells),
            "credits_per_request": PRESCANNED_CREDIT_COST,
            "maximum_daily_credits": discovery_credits,
            "runtime_credit_preflight_required": True,
            "order_submission_capability": False,
            "notes": "five_exchanges_two_intervals_three_strategies_one_priority",
        },
        {
            "lane": "exact_mode_custom_series_proofs",
            "scheduled": True,
            "maximum_requests": proof_request_slots,
            "credits_per_request": CUSTOM_SERIES_CREDIT_COST,
            "maximum_daily_credits": proof_credits,
            "runtime_credit_preflight_required": True,
            "order_submission_capability": False,
            "notes": f"{proof_max_batches}_batches_of_at_most_{proof_batch_size}",
        },
        {
            "lane": "copula_behavioral_parity_proofs",
            "scheduled": True,
            "maximum_requests": copula_behavioral_requests,
            "credits_per_request": COPULA_POST_CREDIT_COST,
            "maximum_daily_credits": copula_behavioral_credits,
            "runtime_credit_preflight_required": True,
            "order_submission_capability": False,
            "notes": "four_mode_orientation_proofs_with_one_repeatability_call_each",
        },
        {
            "lane": "ou_v3_prospective_holdout",
            "scheduled": True,
            "maximum_requests": OU_V3_PROSPECTIVE_MAX_PROOFS,
            "credits_per_request": OU_V3_CUSTOM_SERIES_CREDIT_COST,
            "maximum_daily_credits": ou_v3_prospective_credits,
            "runtime_credit_preflight_required": True,
            "order_submission_capability": False,
            "notes": "four_preregistered_btc_eth_mode_orientation_holdout_cells",
        },
        {
            "lane": "ou_v4_prospective_holdout",
            "scheduled": True,
            "maximum_requests": OU_V4_PROSPECTIVE_MAX_PROOFS,
            "credits_per_request": OU_V4_CUSTOM_SERIES_CREDIT_COST,
            "maximum_daily_credits": ou_v4_prospective_credits,
            "runtime_credit_preflight_required": True,
            "order_submission_capability": False,
            "notes": "eight_preregistered_pair_asset_disjoint_selector_matrix_cells",
        },
        {
            "lane": "ou_v5_prospective_holdout",
            "scheduled": True,
            "maximum_requests": OU_V5_PROSPECTIVE_MAX_PROOFS,
            "credits_per_request": OU_V5_CUSTOM_SERIES_CREDIT_COST,
            "maximum_daily_credits": ou_v5_prospective_credits,
            "runtime_credit_preflight_required": True,
            "order_submission_capability": False,
            "notes": "eight_preregistered_disjoint_transform_trend_profile_holdout_cells",
        },
        {
            "lane": "ou_v6_prospective_holdout",
            "scheduled": True,
            "maximum_requests": OU_V6_PROSPECTIVE_MAX_PROOFS,
            "credits_per_request": OU_V6_CUSTOM_SERIES_CREDIT_COST,
            "maximum_daily_credits": ou_v6_prospective_credits,
            "runtime_credit_preflight_required": True,
            "order_submission_capability": False,
            "notes": "eight_preregistered_final_disjoint_ou_successor_holdout_cells",
        },
        {
            "lane": "ad_hoc_pair_detail_or_backtest_calls",
            "scheduled": False,
            "maximum_requests": 0,
            "credits_per_request": 0,
            "maximum_daily_credits": 0,
            "runtime_credit_preflight_required": True,
            "order_submission_capability": False,
            "notes": "no_automatic_allocation_requires_explicit_bounded_preflight",
        },
    ]
    frame = pd.DataFrame(rows)
    frame["daily_credit_limit"] = daily_credit_limit
    frame["reserved_credits"] = reserved_credits
    frame["available_after_reserve"] = available_after_reserve
    frame["scheduled_credit_ceiling"] = scheduled_ceiling
    frame["headroom_after_reserve"] = headroom
    frame["contract_status"] = "PASS" if contract_passes else "BLOCKED"
    frame["live_trading_authorized"] = False
    active = root / "reports" / "active"
    csv_path = active / "wizard_credit_budget_contract.csv"
    json_path = active / "wizard_credit_budget_contract.json"
    _atomic_csv(frame, csv_path)
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": generated_at.isoformat(),
        "wizard_daily_credit_reset_utc": WIZARD_DAILY_CREDIT_RESET_UTC,
        "status": "PASS" if contract_passes else "BLOCKED",
        "daily_credit_limit": daily_credit_limit,
        "reserved_credits": reserved_credits,
        "available_after_reserve": available_after_reserve,
        "discovery_sweep_credit_ceiling": discovery_credits,
        "exact_mode_proof_credit_ceiling": proof_credits,
        "copula_behavioral_credit_ceiling": copula_behavioral_credits,
        "ou_v3_prospective_credit_ceiling": ou_v3_prospective_credits,
        "ou_v4_prospective_credit_ceiling": ou_v4_prospective_credits,
        "ou_v5_prospective_credit_ceiling": ou_v5_prospective_credits,
        "ou_v6_prospective_credit_ceiling": ou_v6_prospective_credits,
        "scheduled_credit_ceiling": scheduled_ceiling,
        "headroom_after_reserve": headroom,
        "scheduled_lanes": int(frame["scheduled"].sum()),
        "blocker": "" if contract_passes else "scheduled_wizard_lanes_exceed_daily_budget",
        "runtime_credit_preflight_still_required": True,
        "order_submission_included": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
        "evidence_path": _relative(csv_path, root),
    }
    _atomic_json(summary, json_path)
    return CommandResult(
        paths={"budget_contract": csv_path, "budget_summary": json_path},
        summary=summary,
    )


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


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _as_utc(value: datetime | None) -> datetime:
    value = value or datetime.now(UTC)
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
