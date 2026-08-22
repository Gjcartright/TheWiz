"""Prospective OU-v6 selector holdout registration without vendor calls."""

from __future__ import annotations

import inspect
import json
import os
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_platform.active_pipeline import CommandResult
from quant_platform.api_extraction import CryptoWizardsFetchError
from quant_platform.crypto_wizards_history import (
    CryptoWizardsCustomSeriesBacktestRequest,
    fetch_credits_used,
    fetch_custom_series_backtest,
)
from quant_platform.crypto_wizards_sweep import parse_wizard_credit_usage
from quant_platform.orchestration.corrective_redaction import safe_exception_code
from quant_platform.orchestration.corrective_wizard_api_credit_receipt import (
    publish_wizard_api_credit_receipt,
)
from quant_platform.orchestration.corrective_wizard_ou_holdout import (
    _aligned_candles,
    _as_utc,
    _atomic_csv,
    _atomic_json,
    _file_hash,
    _read_json,
    _relative,
    _request_from_payload,
    _text,
    _truthy,
    _write_or_validate_immutable_csv,
    _write_or_validate_immutable_json,
)
from quant_platform.orchestration.corrective_wizard_ou_v4_holdout import (
    DEFAULT_TOLERANCES,
    _ou_local_transform_trend_selector_v2,
    _ou_stationary_constrained_profile_fit_v4,
)
from quant_platform.wizard_hyperliquid_mode_proof import (
    _ou_trend_aware_profile_beta_v3_candidate,
)

ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "thewiz.wizard_ou_v6_prospective_holdout.v1"
RECEIPT_SCHEMA_VERSION = "thewiz.wizard_ou_v6_prospective_holdout_receipt.v1"
EVALUATION_SCHEMA_VERSION = "thewiz.wizard_ou_v6_holdout_evaluation.v2"
CALL_INTENT_SCHEMA_VERSION = "thewiz.wizard_ou_v6_call_attempt_intent.v1"
CALL_COMPLETION_SCHEMA_VERSION = "thewiz.wizard_ou_v6_call_completion.v1"
ATTRIBUTION_STATUS = Path("reports/active/wizard_ou_v5_failure_attribution.json")
V5_DERIVATION = Path("reports/active/wizard_ou_v5_derivation.csv")
V5_EVALUATION = Path("reports/active/wizard_ou_v5_holdout_evaluation.csv")
CONTRACT_PATH = Path("config/wizard_ou_comparator_v6_holdout.json")
RECEIPT_PATH = Path("reports/active/wizard_ou_v6_holdout_receipt.json")
DERIVATION_PATH = Path("reports/active/wizard_ou_v6_derivation.csv")
PREDICTIONS_PATH = Path("reports/active/wizard_ou_v6_predictions.csv")
OU_MODES = ("OU (Spread)", "OU (ZScoreR)")
ORIENTATIONS = ("original", "reverse")
PRIOR_OU_ASSETS = {
    "ADA",
    "ALGO",
    "HBAR",
    "XLM",
    "BTC",
    "ETH",
    "MORPHO",
    "AAVE",
    "AVAX",
    "NEAR",
    "SOL",
    "BNB",
    "DOGE",
    "FET",
}
TRANSFORM_MEDIAN_THRESHOLD = 1.0
DEFAULT_HISTORY_BASE = (
    "reports/snapshots/current_wizard_hyperliquid/"
    "ewapi_69a86274da721a941e36/cwhandoff_61dc9e76532abd99c502/"
    "history_runs/cwhistoryrun_20260812T115658834898Z_1d7a24ca/assets"
)
DEFAULT_PAIR_SPECS: tuple[dict[str, str], ...] = (
    {
        "pair": "APT-ATOM",
        "asset_x": "APT",
        "asset_y": "ATOM",
        "interval": "1d",
        "asset_x_path": f"{DEFAULT_HISTORY_BASE}/APT_1d_candles.json",
        "asset_y_path": f"{DEFAULT_HISTORY_BASE}/ATOM_1d_candles.json",
    },
    {
        "pair": "ARB-OP",
        "asset_x": "ARB",
        "asset_y": "OP",
        "interval": "1h",
        "asset_x_path": f"{DEFAULT_HISTORY_BASE}/ARB_1h_candles.json",
        "asset_y_path": f"{DEFAULT_HISTORY_BASE}/OP_1h_candles.json",
    },
)


def _ou_v6_transform_selector(x: np.ndarray, y: np.ndarray) -> dict[str, object]:
    """Apply the retained, scale-sensitive transform hypothesis."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 50 or len(x) != len(y) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("OU v6 transform selector requires equal finite price series")
    if np.any(x <= 0.0) or np.any(y <= 0.0):
        raise ValueError("OU v6 transform selector requires positive prices")
    x_median = float(np.median(x))
    y_median = float(np.median(y))
    log_used = min(x_median, y_median) >= TRANSFORM_MEDIAN_THRESHOLD
    return {
        "predicted_log_used": log_used,
        "transform": "log" if log_used else "level",
        "series_1_median": x_median,
        "series_2_median": y_median,
        "minimum_series_median": min(x_median, y_median),
        "transform_threshold": TRANSFORM_MEDIAN_THRESHOLD,
        "transform_rule": "log_if_both_series_medians_are_at_least_one",
        "scale_sensitivity_warning": True,
    }


def _ou_v6_profile_branch_features(
    x: np.ndarray, y: np.ndarray, *, log_used: bool
) -> dict[str, float]:
    """Calculate request-only features for the two validated OU kernels."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if log_used:
        x, y = np.log(x), np.log(y)
    zero_fit = _ou_stationary_constrained_profile_fit_v4(x, y)
    intercept_beta = _ou_trend_aware_profile_beta_v3_candidate(x, y, inc_trend=False)
    normalized_x = x / float(x[0])
    normalized_y = y / float(y[0])
    spread = normalized_y - intercept_beta * normalized_x
    lagged = spread[:-1]
    current = spread[1:]
    design = np.column_stack([np.ones(len(lagged)), lagged])
    intercept, phi = np.linalg.lstsq(design, current, rcond=None)[0]
    return {
        "zero_mean_beta": float(zero_fit["beta"]),
        "zero_mean_profile_phi": float(zero_fit["profile_phi"]),
        "zero_mean_unconstrained_phi": float(zero_fit["unconstrained_profile_phi"]),
        "intercept_beta": float(intercept_beta),
        "intercept_ar1_intercept": float(intercept),
        "intercept_ar1_phi": float(phi),
    }


def _ou_v6_predictor(
    x: np.ndarray,
    y: np.ndarray,
) -> dict[str, object]:
    """Predict transform, EG trend, and OU profile branch separately."""

    transform = _ou_v6_transform_selector(x, y)
    branch = _ou_v6_profile_branch_features(x, y, log_used=bool(transform["predicted_log_used"]))
    trend = _ou_local_transform_trend_selector_v2(x, y)
    predicted_branch = (
        "intercept"
        if (
            float(transform["series_1_median"]) < float(transform["series_2_median"])
            and branch["intercept_ar1_intercept"] > 0.0
        )
        else "zero_mean"
    )
    return {
        **transform,
        **branch,
        "predicted_inc_trend": bool(trend["local_predicted_inc_trend"]),
        "trend_selector_cell": _text(trend["selected_selector_cell"]),
        "trend_selector_aic_margin": float(trend["aic_margin"]),
        "predicted_profile_branch": predicted_branch,
        "profile_branch_rule": (
            "intercept_if_series_1_median_below_series_2_median_"
            "and_ar1_intercept_positive_else_zero_mean"
        ),
        "profile_branch_scale_sensitive": True,
        "predicted_hedge_ratio": float(
            branch["intercept_beta"]
            if predicted_branch == "intercept"
            else branch["zero_mean_beta"]
        ),
    }


def register_ou_v6_prospective_holdout(
    *,
    root: Path = ROOT,
    now: datetime | None = None,
    pair_specs: Sequence[Mapping[str, str]] | None = None,
    observations: int = 360,
) -> CommandResult:
    """Freeze the final disjoint eight-cell OU successor and make no API request."""

    if observations < 50 or observations > 1100:
        raise ValueError("OU v6 prospective holdout requires 50 to 1100 observations")
    contract_path = root / CONTRACT_PATH
    receipt_path = root / RECEIPT_PATH
    if contract_path.exists() or receipt_path.exists():
        if not contract_path.is_file() or not receipt_path.is_file():
            raise ValueError("OU v6 registration is incomplete")
        contract = _read_json(contract_path)
        receipt = _read_json(receipt_path)
        _validate_registration(root, contract_path, contract, receipt)
        return CommandResult(
            paths={
                "contract": contract_path,
                "receipt": receipt_path,
                "derivation": root / DERIVATION_PATH,
                "predictions": root / PREDICTIONS_PATH,
            },
            summary=receipt,
        )

    attribution = _validated_v5_attribution(root)
    derivation = _build_derivation(root)
    if len(derivation) != 14:
        raise ValueError("OU v6 requires fourteen consumed derivation orientations")
    if not derivation["transform_rule_passed"].map(_truthy).all():
        raise ValueError("OU v6 transform hypothesis does not fit all derivation rows")
    if not derivation["profile_branch_rule_passed"].map(_truthy).all():
        raise ValueError("OU v6 profile branch rule does not fit every derivation row")
    derivation_path = root / DERIVATION_PATH
    _write_or_validate_immutable_csv(derivation, derivation_path)

    specs = [dict(item) for item in (pair_specs or DEFAULT_PAIR_SPECS)]
    if len(specs) != 2:
        raise ValueError("OU v6 requires exactly two disjoint holdout pair groups")
    assets = [_text(spec.get(field)).upper() for spec in specs for field in ("asset_x", "asset_y")]
    if any(not asset for asset in assets) or len(set(assets)) != 4:
        raise ValueError("OU v6 holdout pairs must contain four distinct assets")
    if set(assets) & PRIOR_OU_ASSETS:
        raise ValueError("OU v6 holdout assets overlap prior OU evidence")

    bindings: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    source_candles: list[dict[str, object]] = []
    request_dir = root / "data/research/wizard_ou_v6_holdout/requests"
    response_dir = root / "data/raw/wizard_ou_v6_holdout/responses"
    for spec in specs:
        pair = _text(spec.get("pair")).upper()
        asset_x = _text(spec.get("asset_x")).upper()
        asset_y = _text(spec.get("asset_y")).upper()
        interval = _text(spec.get("interval")).lower()
        if pair != f"{asset_x}-{asset_y}" or not interval:
            raise ValueError("OU v6 pair specification identity is invalid")
        x_path = _resolve(root, spec.get("asset_x_path"))
        y_path = _resolve(root, spec.get("asset_y_path"))
        if x_path is None or y_path is None:
            raise ValueError("OU v6 source candle path is missing")
        aligned = _aligned_candles(x_path, y_path, observations=observations)
        source_candles.extend(
            [
                {
                    "asset": asset_x,
                    "interval": interval,
                    "path": _relative(x_path, root),
                    "sha256": _file_hash(x_path),
                },
                {
                    "asset": asset_y,
                    "interval": interval,
                    "path": _relative(y_path, root),
                    "sha256": _file_hash(y_path),
                },
            ]
        )
        for exact_mode, strategy in (
            ("OU (Spread)", "Spread"),
            ("OU (ZScoreR)", "ZScoreRoll"),
        ):
            for orientation in ORIENTATIONS:
                if orientation == "original":
                    first, second = aligned["x"], aligned["y"]
                    oriented_pair = pair
                    first_asset, second_asset = asset_x, asset_y
                else:
                    first, second = aligned["y"], aligned["x"]
                    oriented_pair = f"{asset_y}-{asset_x}"
                    first_asset, second_asset = asset_y, asset_x
                request = CryptoWizardsCustomSeriesBacktestRequest(
                    series_1_opens=tuple(first["opens"]),
                    series_1_closes=tuple(first["closes"]),
                    series_2_opens=tuple(second["opens"]),
                    series_2_closes=tuple(second["closes"]),
                    strategy=strategy,
                    spread_type="Ou",
                    roll_w=42,
                    entry_level=2.0,
                    exit_level=0.0,
                    x_weighting=0.5,
                    slippage_rate=0.0005,
                    commission_rate=0.001,
                    with_history=True,
                )
                slug = f"{oriented_pair.lower().replace('-', '_')}_{interval}_{strategy.lower()}"
                request_path = request_dir / f"{slug}_request.json"
                response_path = response_dir / f"{slug}_response.json"
                if response_path.exists():
                    raise ValueError("OU v6 vendor response existed before registration")
                _write_or_validate_immutable_json(request.payload(), request_path)
                prediction = _ou_v6_predictor(
                    np.asarray(first["closes"], dtype=float),
                    np.asarray(second["closes"], dtype=float),
                )
                binding = {
                    "pair_group": pair,
                    "pair": oriented_pair,
                    "asset_x": first_asset,
                    "asset_y": second_asset,
                    "interval": interval,
                    "exact_mode": exact_mode,
                    "orientation": orientation,
                    "strategy": strategy,
                    "spread_type": "Ou",
                    "proof_observations": observations,
                    "first_timestamp": aligned["timestamps"][0],
                    "last_timestamp": aligned["timestamps"][-1],
                    "request_path": _relative(request_path, root),
                    "request_sha256": _file_hash(request_path),
                    "response_path": _relative(response_path, root),
                    "response_expected": True,
                    "response_captured_at_registration": False,
                }
                bindings.append(binding)
                predictions.append(
                    {
                        **{
                            key: binding[key]
                            for key in (
                                "pair_group",
                                "pair",
                                "interval",
                                "exact_mode",
                                "orientation",
                                "proof_observations",
                                "request_path",
                                "request_sha256",
                                "response_path",
                            )
                        },
                        **prediction,
                        "vendor_response_present_at_registration": False,
                    }
                )
    prediction_frame = pd.DataFrame(predictions)
    independent = prediction_frame.loc[prediction_frame["exact_mode"].eq("OU (Spread)")]
    transforms = set(independent["transform"].map(_text))
    branches = set(independent["predicted_profile_branch"].map(_text))
    trend_states = set(independent["predicted_inc_trend"].map(_truthy))
    if len(bindings) != 8 or transforms != {"level", "log"}:
        raise ValueError("OU v6 holdout does not cover both transform predictions")
    if branches != {"zero_mean", "intercept"}:
        raise ValueError("OU v6 holdout does not cover both profile branch predictions")
    if trend_states != {False, True}:
        raise ValueError("OU v6 holdout does not cover both trend predictions")
    predictions_path = root / PREDICTIONS_PATH
    _write_or_validate_immutable_csv(prediction_frame, predictions_path)

    source_hash = _implementation_source_hash()
    contract = {
        "schema_version": SCHEMA_VERSION,
        "status": "PREREGISTERED_WAITING_VENDOR_RESPONSES",
        "hypothesis": {
            "transform_rule": "log_if_both_series_medians_are_at_least_one",
            "transform_threshold": TRANSFORM_MEDIAN_THRESHOLD,
            "transform_scale_sensitive": True,
            "trend_rule": "minimum_aic_across_level_log_x_constant_trend",
            "trend_is_diagnostic_not_formula_switch": True,
            "profile_branch_rule": (
                "intercept_if_series_1_median_below_series_2_median_"
                "and_ar1_intercept_positive_else_zero_mean"
            ),
            "profile_branch_scale_sensitive": True,
            "profile_branch_rule_is_fragile": True,
            "holdout_pair_groups": sorted({_text(row["pair_group"]) for row in bindings}),
            "holdout_is_asset_and_pair_disjoint": True,
            "spread_and_zscorer_are_duplicate_formula_views": True,
            "independent_orientation_count": 4,
            "predicted_transform_states": sorted(transforms),
            "predicted_profile_branches": sorted(branches),
            "predicted_trend_states": sorted(trend_states),
        },
        "implementation": {
            "generation": 6,
            "source_sha256": source_hash,
            "point_in_time_inputs": "registered request close series only",
            "vendor_response_used_for_holdout_selection": False,
            "automatic_activation": False,
        },
        "tolerances": DEFAULT_TOLERANCES,
        "derivation_summary": {
            "independent_orientations": len(derivation),
            "transform_rows_passed": int(derivation["transform_rule_passed"].map(_truthy).sum()),
            "profile_branch_rows_passed": int(
                derivation["profile_branch_rule_passed"].map(_truthy).sum()
            ),
            "v5_rows_are_consumed_derivation_only": True,
            "v5_holdout_reuse_allowed": False,
        },
        "derivation_evidence": [
            {"path": _relative(derivation_path, root), "sha256": _file_hash(derivation_path)},
            {
                "path": _text(attribution.get("immutable_attribution_path")),
                "sha256": _text(attribution.get("immutable_attribution_sha256")),
            },
        ],
        "source_candles": source_candles,
        "holdout_predictions": {
            "path": _relative(predictions_path, root),
            "sha256": _file_hash(predictions_path),
            "cells": len(prediction_frame),
            "independent_orientations": len(independent),
            "vendor_responses_at_registration": 0,
        },
        "holdout_bindings": bindings,
        "vendor_responses_at_registration": 0,
        "hypothesis_changes_after_vendor_response_allowed": False,
        "final_successor_iteration": True,
        "successor_after_v6_failure_allowed": False,
        "automatic_activation": False,
        "research_only": True,
        "candidate_promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    _write_or_validate_immutable_json(contract, contract_path)
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "registered_at_utc": _as_utc(now).isoformat(),
        "status": "PREREGISTERED_WAITING_VENDOR_RESPONSES",
        "contract_path": _relative(contract_path, root),
        "contract_sha256": _file_hash(contract_path),
        "implementation_source_sha256": source_hash,
        "holdout_pair_groups": contract["hypothesis"]["holdout_pair_groups"],
        "required_cells": 8,
        "independent_orientations": 4,
        "derivation_orientations": len(derivation),
        "vendor_responses_at_registration": 0,
        "prospectively_registered": True,
        "v5_holdout_reuse_allowed": False,
        "final_successor_iteration": True,
        "successor_after_v6_failure_allowed": False,
        "research_only": True,
        "candidate_promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    _write_or_validate_immutable_json(receipt, receipt_path)
    return CommandResult(
        paths={
            "contract": contract_path,
            "receipt": receipt_path,
            "derivation": derivation_path,
            "predictions": predictions_path,
        },
        summary=receipt,
    )


def run_ou_v6_prospective_holdout(
    *,
    root: Path = ROOT,
    now: datetime | None = None,
    execute: bool = False,
    api_key: str | None = None,
    fetcher: Callable[..., dict[str, Any]] = fetch_custom_series_backtest,
    credits_fetcher: Callable[..., dict[str, Any] | list[Any]] = fetch_credits_used,
    daily_credit_limit: int = 1000,
    reserved_credits: int = 100,
) -> CommandResult:
    """Capture the preregistered v6 cohort with durable per-call intent."""

    timestamp = _as_utc(now)
    contract_path = root / CONTRACT_PATH
    receipt_path = root / RECEIPT_PATH
    contract = _read_json(contract_path)
    receipt = _read_json(receipt_path)
    _validate_registration(root, contract_path, contract, receipt, allow_responses=True)
    bindings = contract["holdout_bindings"]
    missing = [
        binding
        for binding in bindings
        if not (root / _text(binding.get("response_path"))).is_file()
    ]
    key = api_key or (
        os.getenv("CRYPTO_WIZARDS_API_KEY", "").strip() if execute else ""
    )
    credits_used: int | None = None
    credit_receipt_id = ""
    credit_receipt_path: Path | None = None
    blocker = ""
    ambiguous = (
        _ambiguous_call_intent_blockers(
            root=root,
            timestamp=timestamp,
            contract_path=contract_path,
            missing=missing,
        )
        if execute and missing
        else []
    )
    if ambiguous:
        blocker = ambiguous[0]
    if execute and missing and not blocker:
        if not key:
            blocker = "crypto_wizards_api_key_missing"
        else:
            try:
                raw_usage = credits_fetcher(api_key=key)
                usage = parse_wizard_credit_usage(
                    raw_usage,
                    configured_limit=daily_credit_limit,
                )
                credit_receipt = publish_wizard_api_credit_receipt(
                    root=root,
                    now=timestamp,
                    response=raw_usage,
                    api_key_present=True,
                    api_key_source="argument" if api_key else "environment",
                    daily_limit=daily_credit_limit,
                    protected_reserve=reserved_credits,
                    required_credits=len(missing) * 2,
                )
                credit_receipt_id = str(credit_receipt.summary.get("receipt_id", ""))
                credit_receipt_path = credit_receipt.paths["immutable_receipt"]
                if not usage.known or usage.used is None:
                    blocker = "credit_usage_unknown"
                else:
                    credits_used = usage.used
            except (CryptoWizardsFetchError, OSError, TypeError, ValueError) as exc:
                blocker = f"credit_preflight_failed:{safe_exception_code(exc)}"
            if (
                credits_used is not None
                and credits_used + len(missing) * 2 > daily_credit_limit - reserved_credits
            ):
                blocker = "insufficient_reserved_crypto_wizards_credits"

    attempt_path = (
        root
        / "data/research/wizard_ou_v6_holdout/attempts"
        / f"{timestamp.date().isoformat()}.json"
    )
    attempt_already_exists = execute and bool(missing) and attempt_path.exists()
    if attempt_already_exists:
        blocker = "ou_v6_daily_attempt_already_registered"
    calls_made = 0
    responses_captured = 0
    errors: list[str] = []
    intent_paths: list[str] = []
    completion_paths: list[str] = []
    if execute and missing and not blocker:
        for binding in missing:
            request_path = root / _text(binding.get("request_path"))
            if _file_hash(request_path) != _text(binding.get("request_sha256")):
                errors.append("ou_v6_request_hash_mismatch")
                break
            call_id = _call_id(binding)
            intent_path = _call_intent_path(root=root, timestamp=timestamp, call_id=call_id)
            intent = _call_intent_payload(
                root=root,
                timestamp=timestamp,
                contract_path=contract_path,
                binding=binding,
                call_id=call_id,
            )
            try:
                _write_or_validate_immutable_json(intent, intent_path)
            except (OSError, TypeError, ValueError) as exc:
                errors.append(f"call_intent_persistence_failed:{safe_exception_code(exc)}")
                break
            intent_paths.append(_relative(intent_path, root))
            calls_made += 1
            try:
                response = fetcher(_request_from_payload(_read_json(request_path)), api_key=key)
                response_path = root / _text(binding.get("response_path"))
                _write_or_validate_immutable_json(response, response_path)
                completion_path = _call_completion_path(
                    root=root,
                    call_id=call_id,
                )
                completion = _call_completion_payload(
                    root=root,
                    timestamp=timestamp,
                    contract_path=contract_path,
                    binding=binding,
                    call_id=call_id,
                    intent_path=intent_path,
                    response_path=response_path,
                )
                _write_or_validate_immutable_json(completion, completion_path)
                completion_paths.append(_relative(completion_path, root))
                responses_captured += 1
            except (CryptoWizardsFetchError, OSError, TypeError, ValueError) as exc:
                errors.append(f"vendor_capture_failed:{safe_exception_code(exc)}")
                break
    attempt = {
        "schema_version": "thewiz.wizard_ou_v6_capture_attempt.v1",
        "attempted_at_utc": timestamp.isoformat(),
        "execute_requested": execute,
        "status": (
            "BLOCKED"
            if blocker
            else "FAILED"
            if errors
            else "COMPLETE"
            if not missing or responses_captured == len(missing)
            else "PLANNED"
        ),
        "blocker": blocker,
        "errors": errors,
        "missing_cells_before": len(missing),
        "calls_made": calls_made,
        "responses_captured": responses_captured,
        "credits_used_before": credits_used,
        "credit_preflight_receipt_id": credit_receipt_id,
        "credit_preflight_receipt_path": (
            _relative(credit_receipt_path, root) if credit_receipt_path is not None else ""
        ),
        "credits_attempted": calls_made * 2,
        "credits_completed": responses_captured * 2,
        "call_attempt_intent_paths": intent_paths,
        "call_completion_receipt_paths": completion_paths,
        "ambiguous_call_intent_blockers": ambiguous,
        "contract_sha256": _file_hash(contract_path),
        "research_only": True,
        "candidate_promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    if execute and missing and not attempt_already_exists:
        _write_or_validate_immutable_json(attempt, attempt_path)
    responses_now = sum(
        (root / _text(binding.get("response_path"))).is_file() for binding in bindings
    )
    paths: dict[str, Path] = {}
    capture_readiness = audit_ou_v6_capture_readiness(root=root, now=timestamp)
    if responses_now == len(bindings) and capture_readiness["status"] == "PASS":
        evaluation = evaluate_ou_v6_prospective_holdout(root=root, now=timestamp)
        evaluation_status = _text(evaluation.summary.get("status"))
        paths.update(evaluation.paths)
    elif responses_now == len(bindings):
        evaluation_status = "BLOCKED_CAPTURE_LINEAGE"
    else:
        evaluation_status = "WAITING_VENDOR_RESPONSES"
    status = {
        **attempt,
        "schema_version": "thewiz.wizard_ou_v6_capture_status.v1",
        "responses_available": responses_now,
        "required_responses": len(bindings),
        "evaluation_status": evaluation_status,
        "capture_lineage_status": capture_readiness["status"],
        "capture_lineage_blockers": capture_readiness["blockers"],
        "completion_receipts_valid": capture_readiness["completion_receipts_valid"],
        "next_step": (
            "evaluate_registered_holdout"
            if responses_now == len(bindings)
            else "wait_for_next_eligible_credit_window"
        ),
    }
    status_path = root / "reports/active/wizard_ou_v6_capture_status.json"
    _atomic_json(status, status_path)
    paths["capture_status"] = status_path
    if attempt_path.is_file():
        paths["attempt"] = attempt_path
    if credit_receipt_path is not None:
        paths["credit_preflight_receipt"] = credit_receipt_path
    return CommandResult(paths=paths, summary=status)


def audit_ou_v6_capture_readiness(
    *,
    root: Path = ROOT,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Validate the frozen OU-v6 capture lineage without external requests."""

    timestamp = _as_utc(now)
    contract_path = root / CONTRACT_PATH
    receipt_path = root / RECEIPT_PATH
    blockers: list[str] = []
    contract: dict[str, Any] = {}
    receipt: dict[str, Any] = {}
    try:
        contract = _read_json(contract_path)
        receipt = _read_json(receipt_path)
        _validate_registration(
            root,
            contract_path,
            contract,
            receipt,
            allow_responses=True,
        )
    except (OSError, TypeError, ValueError) as exc:
        blockers.append(f"ou_v6_registration_invalid:{safe_exception_code(exc)}")

    bindings = contract.get("holdout_bindings", [])
    if not isinstance(bindings, list):
        bindings = []
    call_ids = [_call_id(binding) for binding in bindings if isinstance(binding, Mapping)]
    request_paths = [_text(binding.get("request_path")) for binding in bindings]
    response_paths = [_text(binding.get("response_path")) for binding in bindings]
    if (
        len(bindings) != 8
        or len(call_ids) != 8
        or len(set(call_ids)) != 8
        or len(set(request_paths)) != 8
        or len(set(response_paths)) != 8
    ):
        blockers.append("ou_v6_capture_binding_identity_invalid")

    for raw_path in request_paths + response_paths:
        candidate = root / raw_path
        if not raw_path or not _path_is_within_root(root, candidate):
            blockers.append(f"ou_v6_capture_path_unsafe:{raw_path or 'missing'}")

    missing = [
        binding
        for binding in bindings
        if isinstance(binding, Mapping)
        and not (root / _text(binding.get("response_path"))).is_file()
    ]
    blockers.extend(
        _ambiguous_call_intent_blockers(
            root=root,
            timestamp=timestamp,
            contract_path=contract_path,
            missing=missing,
        )
    )

    for binding in bindings:
        if not isinstance(binding, Mapping):
            continue
        response_path = root / _text(binding.get("response_path"))
        call_id = _call_id(binding)
        if response_path.is_file():
            if not _all_call_intent_paths(root=root, call_id=call_id):
                blockers.append("ou_v6_response_without_registered_call_intent:" + call_id)
            _, completion_blockers = _validated_call_completion(
                root=root,
                contract_path=contract_path,
                binding=binding,
            )
            blockers.extend(completion_blockers)
        elif _call_completion_path(root=root, call_id=call_id).is_file():
            blockers.append("ou_v6_call_completion_without_response:" + call_id)

    blockers = list(dict.fromkeys(blockers))
    completion_receipts_valid = sum(
        bool(
            (root / _text(binding.get("response_path"))).is_file()
            and not _validated_call_completion(
                root=root,
                contract_path=contract_path,
                binding=binding,
            )[1]
        )
        for binding in bindings
        if isinstance(binding, Mapping)
    )
    return {
        "status": "PASS" if not blockers else "BLOCKED",
        "blockers": blockers,
        "required_cells": len(bindings),
        "responses_available": len(bindings) - len(missing),
        "missing_cells": len(missing),
        "completion_receipts_valid": completion_receipts_valid,
        "unresolved_call_intents": sum(
            blocker.startswith("ou_v6_call_attempt_") for blocker in blockers
        ),
        "contract_path": _relative(contract_path, root),
        "contract_sha256": _file_hash(contract_path) if contract_path.is_file() else "",
        "receipt_path": _relative(receipt_path, root),
        "research_only": True,
        "candidate_promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }


def evaluate_ou_v6_prospective_holdout(
    *, root: Path = ROOT, now: datetime | None = None
) -> CommandResult:
    """Evaluate frozen v6 predictions without changing its hypothesis."""

    capture_readiness = audit_ou_v6_capture_readiness(root=root, now=now)
    if capture_readiness["status"] != "PASS":
        raise ValueError(
            "OU v6 capture lineage is invalid: " + ";".join(capture_readiness["blockers"])
        )
    contract_path = root / CONTRACT_PATH
    contract = _read_json(contract_path)
    receipt = _read_json(root / RECEIPT_PATH)
    _validate_registration(root, contract_path, contract, receipt, allow_responses=True)
    predictions_path = root / _text(contract["holdout_predictions"]["path"])
    if _file_hash(predictions_path) != _text(contract["holdout_predictions"]["sha256"]):
        raise ValueError("OU v6 frozen predictions binding mismatch")
    predictions = pd.read_csv(predictions_path)
    rows: list[dict[str, object]] = []
    for binding in contract["holdout_bindings"]:
        request_path = root / _text(binding.get("request_path"))
        response_path = root / _text(binding.get("response_path"))
        if not response_path.is_file():
            raise ValueError("OU v6 evaluation requires all vendor responses")
        prediction = _prediction_for_binding(predictions, binding)
        request = _read_json(request_path)
        response = _read_json(response_path)
        completion, completion_blockers = _validated_call_completion(
            root=root,
            contract_path=contract_path,
            binding=binding,
        )
        if completion_blockers:
            raise ValueError(
                "OU v6 call completion binding is invalid: " + ";".join(completion_blockers)
            )
        call_id = _call_id(binding)
        completion_path = _call_completion_path(root=root, call_id=call_id)
        intent_path = root / _text(completion.get("intent_path"))
        vendor_log, vendor_trend = _vendor_selector(response)
        predicted_log = _truthy(prediction.get("predicted_log_used"))
        predicted_trend = _truthy(prediction.get("predicted_inc_trend"))
        predicted_branch = _text(prediction.get("predicted_profile_branch"))
        if predicted_branch not in {"zero_mean", "intercept"}:
            raise ValueError("OU v6 frozen profile branch prediction is invalid")
        metrics = _evaluate_formula(
            request=request,
            response=response,
            log_used=predicted_log,
            inc_trend=predicted_branch == "zero_mean",
            tolerances=contract["tolerances"],
        )
        zero_oracle = _evaluate_formula(
            request=request,
            response=response,
            log_used=vendor_log,
            inc_trend=True,
            tolerances=contract["tolerances"],
        )
        intercept_oracle = _evaluate_formula(
            request=request,
            response=response,
            log_used=vendor_log,
            inc_trend=False,
            tolerances=contract["tolerances"],
        )
        oracle_passes = {
            "zero_mean": _truthy(zero_oracle["formula_parity_passed"]),
            "intercept": _truthy(intercept_oracle["formula_parity_passed"]),
        }
        passing_branches = [name for name, passed in oracle_passes.items() if passed]
        inferred_branch = passing_branches[0] if len(passing_branches) == 1 else "ambiguous"
        transform_passed = predicted_log == vendor_log
        trend_passed = predicted_trend == vendor_trend
        branch_passed = predicted_branch == inferred_branch
        formula_passed = _truthy(metrics["formula_parity_passed"])
        passed = transform_passed and trend_passed and branch_passed and formula_passed
        rows.append(
            {
                "pair_group": _text(binding.get("pair_group")),
                "pair": _text(binding.get("pair")),
                "interval": _text(binding.get("interval")),
                "exact_mode": _text(binding.get("exact_mode")),
                "orientation": _text(binding.get("orientation")),
                "cell_status": "PASS" if passed else "FAIL",
                "predicted_log_used": predicted_log,
                "vendor_log_used": vendor_log,
                "transform_selector_parity_passed": transform_passed,
                "predicted_inc_trend": predicted_trend,
                "vendor_inc_trend": vendor_trend,
                "trend_selector_parity_passed": trend_passed,
                "predicted_profile_branch": predicted_branch,
                "inferred_vendor_profile_branch": inferred_branch,
                "profile_branch_selector_parity_passed": branch_passed,
                **metrics,
                "zero_mean_oracle_formula_passed": oracle_passes["zero_mean"],
                "intercept_oracle_formula_passed": oracle_passes["intercept"],
                "request_path": _relative(request_path, root),
                "request_sha256": _file_hash(request_path),
                "response_path": _relative(response_path, root),
                "response_sha256": _file_hash(response_path),
                "call_id": call_id,
                "call_intent_path": _relative(intent_path, root),
                "call_intent_sha256": _file_hash(intent_path),
                "call_completion_path": _relative(completion_path, root),
                "call_completion_sha256": _file_hash(completion_path),
                "blocker": ""
                if passed
                else _v6_blocker(
                    transform_passed,
                    trend_passed,
                    branch_passed,
                    formula_passed,
                ),
                "candidate_promotion_authority": False,
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            }
        )
    frame = pd.DataFrame(rows)
    required = len(frame)
    passed_cells = int(frame["cell_status"].eq("PASS").sum())
    status_value = "PASS" if passed_cells == required else "FAIL"
    detail_path = root / "reports/active/wizard_ou_v6_holdout_evaluation.csv"
    status_path = root / "reports/active/wizard_ou_v6_holdout_status.json"
    _atomic_csv(frame, detail_path)
    payload = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "evaluated_at_utc": _as_utc(now).isoformat(),
        "status": status_value,
        "contract_path": _relative(contract_path, root),
        "contract_sha256": _file_hash(contract_path),
        "required_cells": required,
        "passed_cells": passed_cells,
        "failed_cells": required - passed_cells,
        "transform_selector_parity_passed_cells": int(
            frame["transform_selector_parity_passed"].map(_truthy).sum()
        ),
        "trend_selector_parity_passed_cells": int(
            frame["trend_selector_parity_passed"].map(_truthy).sum()
        ),
        "profile_branch_selector_parity_passed_cells": int(
            frame["profile_branch_selector_parity_passed"].map(_truthy).sum()
        ),
        "formula_parity_passed_cells": int(frame["formula_parity_passed"].map(_truthy).sum()),
        "implementation_source_sha256": _implementation_source_hash(),
        "detail_sha256": _file_hash(detail_path),
        "request_sha256s": sorted(frame["request_sha256"].map(_text)),
        "response_sha256s": sorted(frame["response_sha256"].map(_text)),
        "call_completion_sha256s": sorted(frame["call_completion_sha256"].map(_text)),
        "v5_holdout_reuse_allowed": False,
        "final_successor_iteration": True,
        "successor_after_v6_failure_allowed": False,
        "comparator_supersession_automatic": False,
        "research_only": True,
        "candidate_promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
        "evidence_path": _relative(detail_path, root),
    }
    immutable = {
        key: value
        for key, value in payload.items()
        if key not in {"evaluated_at_utc", "evidence_path"}
    }
    result_id = (
        "ouv6holdout_"
        + sha256(
            json.dumps(immutable, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:24]
    )
    immutable["result_id"] = result_id
    immutable_path = root / "data/research/wizard_ou_v6_holdout_evaluations" / f"{result_id}.json"
    _write_or_validate_immutable_json(immutable, immutable_path)
    payload.update(
        {
            "result_id": result_id,
            "immutable_result_path": _relative(immutable_path, root),
            "immutable_result_sha256": _file_hash(immutable_path),
        }
    )
    _atomic_json(payload, status_path)
    return CommandResult(
        paths={"detail": detail_path, "status": status_path, "immutable_result": immutable_path},
        summary=payload,
    )


def validate_ou_v6_evaluation_binding(
    *,
    root: Path = ROOT,
    status: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Recompute and validate the complete OU-v6 evaluation lineage."""

    status_path = root / "reports/active/wizard_ou_v6_holdout_status.json"
    detail_path = root / "reports/active/wizard_ou_v6_holdout_evaluation.csv"
    contract_path = root / CONTRACT_PATH
    receipt_path = root / RECEIPT_PATH
    payload = status if status is not None else _read_json(status_path)
    contract = _read_json(contract_path)
    receipt = _read_json(receipt_path)
    blockers: list[str] = []
    capture_readiness = audit_ou_v6_capture_readiness(root=root)
    if capture_readiness["status"] != "PASS":
        blockers.extend(capture_readiness["blockers"])
    try:
        _validate_registration(
            root,
            contract_path,
            contract,
            receipt,
            allow_responses=True,
        )
    except (OSError, TypeError, ValueError) as exc:
        blockers.append(f"ou_v6_evaluation_registration_invalid:{type(exc).__name__}")

    try:
        detail = pd.read_csv(detail_path)
    except (FileNotFoundError, OSError, pd.errors.ParserError):
        detail = pd.DataFrame()
        blockers.append("ou_v6_evaluation_detail_missing_or_invalid")

    immutable_relative = _text(payload.get("immutable_result_path"))
    immutable_path = root / immutable_relative if immutable_relative else None
    immutable_root = (root / "data/research/wizard_ou_v6_holdout_evaluations").resolve()
    immutable: dict[str, Any] = {}
    if immutable_path is None:
        blockers.append("ou_v6_evaluation_immutable_result_missing")
    else:
        try:
            immutable_path.resolve().relative_to(immutable_root)
        except (OSError, ValueError):
            blockers.append("ou_v6_evaluation_immutable_path_unsafe")
        if not immutable_path.is_file():
            blockers.append("ou_v6_evaluation_immutable_result_missing")
        elif _file_hash(immutable_path) != _text(payload.get("immutable_result_sha256")):
            blockers.append("ou_v6_evaluation_immutable_hash_mismatch")
        else:
            immutable = _read_json(immutable_path)

    expected_immutable = {
        key: value
        for key, value in payload.items()
        if key
        not in {
            "evaluated_at_utc",
            "evidence_path",
            "immutable_result_path",
            "immutable_result_sha256",
        }
    }
    if immutable and immutable != expected_immutable:
        blockers.append("ou_v6_evaluation_status_immutable_mismatch")
    result_id = _text(payload.get("result_id"))
    if immutable:
        identity_material = {key: value for key, value in immutable.items() if key != "result_id"}
        expected_result_id = (
            "ouv6holdout_"
            + sha256(
                json.dumps(
                    identity_material,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()[:24]
        )
        if (
            result_id != expected_result_id
            or _text(immutable.get("result_id")) != expected_result_id
            or immutable_path is None
            or immutable_path.stem != expected_result_id
        ):
            blockers.append("ou_v6_evaluation_result_identity_mismatch")

    if payload.get("schema_version") != EVALUATION_SCHEMA_VERSION:
        blockers.append("ou_v6_evaluation_schema_invalid")
    if _text(payload.get("contract_sha256")) != (
        _file_hash(contract_path) if contract_path.is_file() else ""
    ):
        blockers.append("ou_v6_evaluation_contract_hash_mismatch")
    if _text(payload.get("implementation_source_sha256")) != _implementation_source_hash():
        blockers.append("ou_v6_evaluation_implementation_hash_mismatch")
    if not detail_path.is_file() or _text(payload.get("detail_sha256")) != (
        _file_hash(detail_path) if detail_path.is_file() else ""
    ):
        blockers.append("ou_v6_evaluation_detail_hash_mismatch")

    expected_identities = sorted(
        (
            _text(binding.get("pair_group")),
            _text(binding.get("pair")),
            _text(binding.get("exact_mode")),
            _text(binding.get("orientation")),
            _text(binding.get("request_path")),
            _text(binding.get("request_sha256")),
            _text(binding.get("response_path")),
            _call_id(binding),
            _text(
                _read_json(_call_completion_path(root=root, call_id=_call_id(binding))).get(
                    "intent_path"
                )
            ),
            _text(
                _read_json(_call_completion_path(root=root, call_id=_call_id(binding))).get(
                    "intent_sha256"
                )
            ),
            _relative(
                _call_completion_path(root=root, call_id=_call_id(binding)),
                root,
            ),
            (
                _file_hash(_call_completion_path(root=root, call_id=_call_id(binding)))
                if _call_completion_path(root=root, call_id=_call_id(binding)).is_file()
                else ""
            ),
        )
        for binding in contract.get("holdout_bindings", [])
        if isinstance(binding, Mapping)
    )
    observed_identities = sorted(
        (
            _text(row.get("pair_group")),
            _text(row.get("pair")),
            _text(row.get("exact_mode")),
            _text(row.get("orientation")),
            _text(row.get("request_path")),
            _text(row.get("request_sha256")),
            _text(row.get("response_path")),
            _text(row.get("call_id")),
            _text(row.get("call_intent_path")),
            _text(row.get("call_intent_sha256")),
            _text(row.get("call_completion_path")),
            _text(row.get("call_completion_sha256")),
        )
        for row in detail.to_dict("records")
    )
    if len(expected_identities) != 8 or observed_identities != expected_identities:
        blockers.append("ou_v6_evaluation_cell_identity_mismatch")

    for row in detail.to_dict("records"):
        for path_field, hash_field in (
            ("request_path", "request_sha256"),
            ("response_path", "response_sha256"),
            ("call_intent_path", "call_intent_sha256"),
            ("call_completion_path", "call_completion_sha256"),
        ):
            evidence_path = root / _text(row.get(path_field))
            if (
                not _path_is_within_root(root, evidence_path)
                or not evidence_path.is_file()
                or _file_hash(evidence_path) != _text(row.get(hash_field))
            ):
                blockers.append(f"ou_v6_evaluation_raw_binding_mismatch:{path_field}")

    count_fields = {
        "formula_parity_passed_cells": "formula_parity_passed",
        "transform_selector_parity_passed_cells": "transform_selector_parity_passed",
        "trend_selector_parity_passed_cells": "trend_selector_parity_passed",
        "profile_branch_selector_parity_passed_cells": ("profile_branch_selector_parity_passed"),
    }
    passed_cells = (
        int(detail.get("cell_status", pd.Series(dtype=str)).astype(str).eq("PASS").sum())
        if not detail.empty
        else 0
    )
    required_cells = len(detail)
    expected_status = "PASS" if required_cells == 8 and passed_cells == 8 else "FAIL"
    if (
        int(payload.get("required_cells", 0) or 0) != required_cells
        or int(payload.get("passed_cells", 0) or 0) != passed_cells
        or int(payload.get("failed_cells", 0) or 0) != required_cells - passed_cells
        or payload.get("status") != expected_status
    ):
        blockers.append("ou_v6_evaluation_status_count_mismatch")
    for summary_field, detail_field in count_fields.items():
        actual = int(
            detail.get(detail_field, pd.Series(False, index=detail.index)).map(_truthy).sum()
        )
        if int(payload.get(summary_field, 0) or 0) != actual:
            blockers.append(f"ou_v6_evaluation_summary_count_mismatch:{summary_field}")

    request_hashes = sorted(detail.get("request_sha256", pd.Series(dtype=str)).map(_text))
    response_hashes = sorted(detail.get("response_sha256", pd.Series(dtype=str)).map(_text))
    if payload.get("request_sha256s") != request_hashes:
        blockers.append("ou_v6_evaluation_request_hash_set_mismatch")
    if payload.get("response_sha256s") != response_hashes:
        blockers.append("ou_v6_evaluation_response_hash_set_mismatch")
    completion_hashes = sorted(
        detail.get("call_completion_sha256", pd.Series(dtype=str)).map(_text)
    )
    if payload.get("call_completion_sha256s") != completion_hashes:
        blockers.append("ou_v6_evaluation_completion_hash_set_mismatch")
    authority_fields = (
        "candidate_promotion_authority",
        "testnet_order_authority",
        "live_trading_authorized",
    )
    if any(_truthy(payload.get(field)) for field in authority_fields) or any(
        detail.get(field, pd.Series(False, index=detail.index)).map(_truthy).any()
        for field in authority_fields
    ):
        blockers.append("ou_v6_evaluation_authority_violation")

    blockers = list(dict.fromkeys(blockers))
    return {
        "status": "PASS" if not blockers else "BLOCKED",
        "blockers": blockers,
        "result_id": result_id,
        "required_cells": required_cells,
        "passed_cells": passed_cells,
        "detail_path": _relative(detail_path, root),
        "detail_sha256": _file_hash(detail_path) if detail_path.is_file() else "",
        "immutable_result_path": immutable_relative,
        "immutable_result_sha256": (
            _file_hash(immutable_path)
            if immutable_path is not None and immutable_path.is_file()
            else ""
        ),
        "research_only": True,
        "candidate_promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }


def validate_ou_v6_registration_binding(*, root: Path = ROOT) -> dict[str, Any]:
    """Validate the current prospective registration without requiring responses."""

    contract_path = root / CONTRACT_PATH
    receipt_path = root / RECEIPT_PATH
    blockers: list[str] = []
    if not contract_path.is_file() or not receipt_path.is_file():
        blockers.append("ou_v6_registration_pair_incomplete")
    else:
        try:
            _validate_registration(
                root,
                contract_path,
                _read_json(contract_path),
                _read_json(receipt_path),
                allow_responses=True,
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            blockers.append(f"ou_v6_registration_binding_invalid:{type(exc).__name__}")
    return {
        "status": "PASS" if not blockers else "BLOCKED",
        "blockers": blockers,
        "contract_path": _relative(contract_path, root),
        "contract_sha256": _file_hash(contract_path) if contract_path.is_file() else "",
        "receipt_path": _relative(receipt_path, root),
        "receipt_sha256": _file_hash(receipt_path) if receipt_path.is_file() else "",
        "research_only": True,
        "candidate_promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }


def validate_ou_v6_stage3_evidence(
    *,
    root: Path = ROOT,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Require the current prospective OU-v6 evaluation at the Stage 3 boundary."""

    contract_path = root / CONTRACT_PATH
    receipt_path = root / RECEIPT_PATH
    configured = contract_path.exists() or receipt_path.exists()
    claimed = evidence.get("ou_v6_prospectively_registered") is True
    blockers: list[str] = []

    if not configured:
        if claimed:
            blockers.append("ou_v6_registration_claim_without_current_contract")
        return {
            "status": "PASS" if not blockers else "BLOCKED",
            "required": False,
            "blockers": blockers,
            "evaluation_status": "NOT_REQUIRED",
            "research_only": True,
            "candidate_promotion_authority": False,
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        }

    registration = validate_ou_v6_registration_binding(root=root)
    if registration.get("status") != "PASS":
        blockers.extend(str(value) for value in registration.get("blockers", []))
    if not claimed:
        blockers.append("ou_v6_current_registration_omitted_from_scheduler_evidence")

    try:
        required = int(evidence.get("ou_v6_required_responses", 0) or 0)
        available = int(evidence.get("ou_v6_responses_available", 0) or 0)
        generation = int(evidence.get("ou_v6_comparator_generation", 0) or 0)
        refreshed = int(evidence.get("ou_v6_proofs_refreshed", 0) or 0)
    except (TypeError, ValueError):
        required = available = generation = refreshed = -1
        blockers.append("ou_v6_scheduler_counts_invalid")

    scheduler_complete = bool(
        required == 8
        and available == required
        and evidence.get("ou_v6_holdout_status") == "COMPLETE"
        and evidence.get("ou_v6_evaluation_status") == "PASS"
        and evidence.get("ou_v6_response_accounting_valid") is True
        and evidence.get("ou_v6_activation_status") == "APPLIED_RESEARCH_COMPARATOR_ONLY"
        and generation == 6
        and evidence.get("ou_v6_proof_refresh_status") == "PASS"
        and refreshed >= 8
        and evidence.get("ou_v6_activation_automatic") is False
        and evidence.get("ou_v6_research_only") is True
    )
    if not scheduler_complete:
        blockers.append("ou_v6_scheduler_stage3_evidence_incomplete")

    evaluation: dict[str, Any] = {}
    if scheduler_complete and registration.get("status") == "PASS":
        try:
            evaluation = validate_ou_v6_evaluation_binding(root=root)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            blockers.append(f"ou_v6_evaluation_validation_failed:{type(exc).__name__}")
        if evaluation.get("status") != "PASS":
            blockers.extend(
                str(value)
                for value in evaluation.get("blockers", ["ou_v6_evaluation_not_valid"])
                if str(value)
            )
    if any(
        _truthy(evidence.get(field))
        for field in (
            "candidate_promotion_authority",
            "order_submission_included",
            "testnet_order_authority",
            "live_trading_authorized",
        )
    ):
        blockers.append("ou_v6_scheduler_authority_violation")

    blockers = list(dict.fromkeys(blockers))
    return {
        "status": "PASS" if not blockers else "BLOCKED",
        "required": True,
        "blockers": blockers,
        "registration_status": registration.get("status", "BLOCKED"),
        "evaluation_status": evaluation.get(
            "status",
            "NOT_EVALUATED" if not scheduler_complete else "BLOCKED",
        ),
        "evaluation_result_id": evaluation.get("result_id", ""),
        "evaluation_immutable_path": evaluation.get("immutable_result_path", ""),
        "evaluation_immutable_sha256": evaluation.get("immutable_result_sha256", ""),
        "research_only": True,
        "candidate_promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }


def _call_id(binding: Mapping[str, Any]) -> str:
    material = "|".join(
        (
            "ou_v6_holdout",
            _text(binding.get("pair")),
            _text(binding.get("exact_mode")),
            _text(binding.get("orientation")),
            str(int(binding.get("proof_observations", 0) or 0)),
            "1",
        )
    )
    return "wizcall_" + sha256(material.encode("utf-8")).hexdigest()[:20]


def _call_intent_path(*, root: Path, timestamp: datetime, call_id: str) -> Path:
    return (
        root
        / "data/research/wizard_ou_v6_holdout/call_attempts"
        / timestamp.date().isoformat()
        / f"{call_id}.json"
    )


def _all_call_intent_paths(*, root: Path, call_id: str) -> list[Path]:
    attempts_root = root / "data/research/wizard_ou_v6_holdout/call_attempts"
    if not attempts_root.is_dir():
        return []
    return sorted(
        path
        for path in attempts_root.glob(f"*/{call_id}.json")
        if path.is_file() and _path_is_within_root(root, path)
    )


def _call_completion_path(*, root: Path, call_id: str) -> Path:
    return root / "data/research/wizard_ou_v6_holdout/call_completions" / f"{call_id}.json"


def _call_intent_payload(
    *,
    root: Path,
    timestamp: datetime,
    contract_path: Path,
    binding: Mapping[str, Any],
    call_id: str,
) -> dict[str, Any]:
    return {
        "schema_version": CALL_INTENT_SCHEMA_VERSION,
        "registered_at_utc": timestamp.isoformat(),
        "attempt_date_utc": timestamp.date().isoformat(),
        "call_id": call_id,
        "pair": _text(binding.get("pair")),
        "exact_mode": _text(binding.get("exact_mode")),
        "orientation": _text(binding.get("orientation")),
        "observations": int(binding.get("proof_observations", 0) or 0),
        "request_path": _text(binding.get("request_path")),
        "request_sha256": _text(binding.get("request_sha256")),
        "response_path": _text(binding.get("response_path")),
        "credit_cost": 2,
        "contract_path": _relative(contract_path, root),
        "contract_sha256": _file_hash(contract_path),
        "research_only": True,
        "candidate_promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }


def _call_completion_payload(
    *,
    root: Path,
    timestamp: datetime,
    contract_path: Path,
    binding: Mapping[str, Any],
    call_id: str,
    intent_path: Path,
    response_path: Path,
) -> dict[str, Any]:
    material: dict[str, Any] = {
        "schema_version": CALL_COMPLETION_SCHEMA_VERSION,
        "status": "COMPLETE",
        "completed_at_utc": timestamp.isoformat(),
        "attempt_date_utc": timestamp.date().isoformat(),
        "call_id": call_id,
        "pair": _text(binding.get("pair")),
        "exact_mode": _text(binding.get("exact_mode")),
        "orientation": _text(binding.get("orientation")),
        "observations": int(binding.get("proof_observations", 0) or 0),
        "intent_path": _relative(intent_path, root),
        "intent_sha256": _file_hash(intent_path),
        "request_path": _text(binding.get("request_path")),
        "request_sha256": _text(binding.get("request_sha256")),
        "response_path": _relative(response_path, root),
        "response_sha256": _file_hash(response_path),
        "contract_path": _relative(contract_path, root),
        "contract_sha256": _file_hash(contract_path),
        "credit_cost": 2,
        "research_only": True,
        "candidate_promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    material["completion_id"] = (
        "wizcompletion_"
        + sha256(
            json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:20]
    )
    return material


def _validated_call_completion(
    *,
    root: Path,
    contract_path: Path,
    binding: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    call_id = _call_id(binding)
    completion_path = _call_completion_path(root=root, call_id=call_id)
    completion = _read_json(completion_path)
    blocker = f"ou_v6_call_completion_invalid:{call_id}"
    if not completion_path.is_file() or not completion:
        return {}, [f"ou_v6_call_completion_missing:{call_id}"]
    material = {key: value for key, value in completion.items() if key != "completion_id"}
    expected_completion_id = (
        "wizcompletion_"
        + sha256(
            json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:20]
    )
    intent_relative = _text(completion.get("intent_path"))
    intent_path = root / intent_relative
    intent = _read_json(intent_path)
    response_path = root / _text(binding.get("response_path"))
    registered_at = pd.to_datetime(intent.get("registered_at_utc"), utc=True, errors="coerce")
    completed_at = pd.to_datetime(completion.get("completed_at_utc"), utc=True, errors="coerce")
    authority_fields = (
        "candidate_promotion_authority",
        "testnet_order_authority",
        "live_trading_authorized",
    )
    valid = bool(
        completion.get("schema_version") == CALL_COMPLETION_SCHEMA_VERSION
        and completion.get("status") == "COMPLETE"
        and _text(completion.get("completion_id")) == expected_completion_id
        and _text(completion.get("call_id")) == call_id
        and _text(completion.get("pair")) == _text(binding.get("pair"))
        and _text(completion.get("exact_mode")) == _text(binding.get("exact_mode"))
        and _text(completion.get("orientation")) == _text(binding.get("orientation"))
        and int(completion.get("observations", 0) or 0)
        == int(binding.get("proof_observations", 0) or 0)
        and intent_relative
        and _path_is_within_root(root, intent_path)
        and intent_path.is_file()
        and _file_hash(intent_path) == _text(completion.get("intent_sha256"))
        and intent.get("schema_version") == CALL_INTENT_SCHEMA_VERSION
        and _text(intent.get("call_id")) == call_id
        and _text(intent.get("request_path")) == _text(binding.get("request_path"))
        and _text(intent.get("request_sha256")) == _text(binding.get("request_sha256"))
        and _text(intent.get("response_path")) == _text(binding.get("response_path"))
        and _text(completion.get("attempt_date_utc")) == _text(intent.get("attempt_date_utc"))
        and pd.notna(registered_at)
        and pd.notna(completed_at)
        and completed_at >= registered_at
        and _text(completion.get("request_path")) == _text(binding.get("request_path"))
        and _text(completion.get("request_sha256")) == _text(binding.get("request_sha256"))
        and _text(completion.get("response_path")) == _text(binding.get("response_path"))
        and response_path.is_file()
        and _file_hash(response_path) == _text(completion.get("response_sha256"))
        and _text(completion.get("contract_path")) == _relative(contract_path, root)
        and _text(completion.get("contract_sha256")) == _file_hash(contract_path)
        and int(completion.get("credit_cost", 0) or 0) == 2
        and _truthy(completion.get("research_only"))
        and not any(_truthy(completion.get(field)) for field in authority_fields)
        and not any(_truthy(intent.get(field)) for field in authority_fields)
    )
    return (completion, []) if valid else ({}, [blocker])


def validate_ou_v6_call_completion(
    *, root: Path, contract_path: Path, binding: Mapping[str, Any]
) -> dict[str, Any]:
    """Public cross-component validator for one immutable OU-v6 completion."""

    call_id = _call_id(binding)
    completion_path = _call_completion_path(root=root, call_id=call_id)
    _, blockers = _validated_call_completion(
        root=root,
        contract_path=contract_path,
        binding=binding,
    )
    return {
        "status": "PASS" if not blockers else "BLOCKED",
        "blockers": blockers,
        "call_id": call_id,
        "completion_path": _relative(completion_path, root),
        "completion_sha256": (_file_hash(completion_path) if completion_path.is_file() else ""),
        "research_only": True,
        "candidate_promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }


def _ambiguous_call_intent_blockers(
    *,
    root: Path,
    timestamp: datetime,
    contract_path: Path,
    missing: Sequence[Mapping[str, Any]],
) -> list[str]:
    blockers: list[str] = []
    for binding in missing:
        call_id = _call_id(binding)
        for path in _all_call_intent_paths(root=root, call_id=call_id):
            intent = _read_json(path)
            attempt_date = _text(intent.get("attempt_date_utc"))
            valid = bool(
                intent.get("schema_version") == CALL_INTENT_SCHEMA_VERSION
                and attempt_date == path.parent.name
                and attempt_date <= timestamp.date().isoformat()
                and _text(intent.get("call_id")) == call_id
                and _text(intent.get("request_path")) == _text(binding.get("request_path"))
                and _text(intent.get("request_sha256")) == _text(binding.get("request_sha256"))
                and _text(intent.get("response_path")) == _text(binding.get("response_path"))
                and int(intent.get("credit_cost", 0) or 0) == 2
                and _text(intent.get("contract_path")) == _relative(contract_path, root)
                and _text(intent.get("contract_sha256")) == _file_hash(contract_path)
                and _truthy(intent.get("research_only"))
                and not any(
                    _truthy(intent.get(field))
                    for field in (
                        "candidate_promotion_authority",
                        "testnet_order_authority",
                        "live_trading_authorized",
                    )
                )
            )
            blockers.append(
                (
                    "ou_v6_call_attempt_already_registered_without_response:"
                    if valid
                    else "ou_v6_call_attempt_intent_invalid:"
                )
                + call_id
                + ":"
                + (attempt_date or path.parent.name)
            )
    return blockers


def _path_is_within_root(root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return False
    return True


def _vendor_selector(response: dict[str, Any]) -> tuple[bool, bool]:
    history = response.get("history") if isinstance(response.get("history"), dict) else {}
    stats = history.get("spread_stats") if isinstance(history.get("spread_stats"), dict) else {}
    coint = history.get("coint_eg") if isinstance(history.get("coint_eg"), dict) else {}
    if "log_used" not in stats or "inc_trend" not in coint:
        raise ValueError("OU v6 response lacks transform or trend provenance")
    return _truthy(stats.get("log_used")), _truthy(coint.get("inc_trend"))


def _evaluate_formula(**kwargs: Any) -> dict[str, object]:
    from quant_platform.orchestration.corrective_wizard_ou_v4_holdout import (
        _evaluate_formula as evaluator,
    )

    return evaluator(**kwargs)


def _prediction_for_binding(predictions: pd.DataFrame, binding: Mapping[str, Any]) -> pd.Series:
    matches = predictions.loc[
        predictions["exact_mode"].map(_text).eq(_text(binding.get("exact_mode")))
        & predictions["orientation"].map(_text).eq(_text(binding.get("orientation")))
        & predictions["request_sha256"].map(_text).eq(_text(binding.get("request_sha256")))
    ]
    if len(matches) != 1:
        raise ValueError("OU v6 frozen prediction binding is not unique")
    return matches.iloc[0]


def _v6_blocker(
    transform_passed: bool,
    trend_passed: bool,
    branch_passed: bool,
    formula_passed: bool,
) -> str:
    blockers = []
    if not transform_passed:
        blockers.append("transform_selector_mismatch")
    if not trend_passed:
        blockers.append("trend_selector_mismatch")
    if not branch_passed:
        blockers.append("profile_branch_selector_mismatch")
    if not formula_passed:
        blockers.append("formula_parity_mismatch")
    return ";".join(blockers)


def _build_derivation(root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    prior = pd.read_csv(root / V5_DERIVATION)
    for _, source in prior.iterrows():
        rows.append(
            _derivation_row(
                root=root,
                source=source,
                cohort=_text(source.get("cohort")),
                vendor_branch=_text(source.get("vendor_profile_branch")),
            )
        )
    evaluation = pd.read_csv(root / V5_EVALUATION)
    evaluation = evaluation.loc[evaluation["exact_mode"].eq("OU (Spread)")]
    for _, source in evaluation.iterrows():
        rows.append(
            _derivation_row(
                root=root,
                source=source,
                cohort="v5_consumed_holdout_derivation_only",
                vendor_branch=_text(source.get("inferred_vendor_profile_branch")),
            )
        )
    frame = pd.DataFrame(rows)
    frame["predicted_profile_branch"] = np.where(
        frame["series_1_median"].lt(frame["series_2_median"]) & frame["ar1_intercept"].gt(0.0),
        "intercept",
        "zero_mean",
    )
    frame["profile_branch_rule_passed"] = (
        frame["predicted_profile_branch"] == frame["vendor_profile_branch"]
    )
    return frame


def _derivation_row(
    *, root: Path, source: pd.Series, cohort: str, vendor_branch: str
) -> dict[str, Any]:
    request_path = root / _text(source.get("request_path"))
    response_path = root / _text(source.get("response_path"))
    if (
        not request_path.is_file()
        or _file_hash(request_path) != _text(source.get("request_sha256"))
        or not response_path.is_file()
        or _file_hash(response_path) != _text(source.get("response_sha256"))
    ):
        raise ValueError("OU v6 derivation raw binding is invalid")
    request = _read_json(request_path)
    params = request.get("params") if isinstance(request.get("params"), dict) else {}
    x = np.asarray(params.get("series_1_closes"), dtype=float)
    y = np.asarray(params.get("series_2_closes"), dtype=float)
    transform = _ou_v6_transform_selector(x, y)
    vendor_log = _truthy(source.get("vendor_log_used"))
    branch = _ou_v6_profile_branch_features(x, y, log_used=vendor_log)
    return {
        "cohort": cohort,
        "orientation": _text(source.get("orientation")),
        "request_path": _relative(request_path, root),
        "request_sha256": _file_hash(request_path),
        "response_path": _relative(response_path, root),
        "response_sha256": _file_hash(response_path),
        "vendor_log_used": vendor_log,
        "predicted_log_used": transform["predicted_log_used"],
        "series_1_median": transform["series_1_median"],
        "series_2_median": transform["series_2_median"],
        "minimum_series_median": transform["minimum_series_median"],
        "transform_rule_passed": transform["predicted_log_used"] == vendor_log,
        "vendor_inc_trend": _truthy(source.get("vendor_inc_trend")),
        "vendor_profile_branch": vendor_branch,
        "ar1_intercept": branch["intercept_ar1_intercept"],
        "zero_mean_beta": branch["zero_mean_beta"],
        "intercept_beta": branch["intercept_beta"],
        "v5_evidence_role": (
            "consumed_negative_holdout_derivation_only"
            if cohort == "v5_consumed_holdout_derivation_only"
            else "prior_consumed_derivation"
        ),
        "holdout_reuse_allowed": False,
    }


def _validated_v5_attribution(root: Path) -> dict[str, Any]:
    payload = _read_json(root / ATTRIBUTION_STATUS)
    if payload.get("status") != "PASS_FAILURE_ATTRIBUTION_COMPLETE":
        raise ValueError("OU v5 failure attribution is not complete")
    if payload.get("v5_holdout_reuse_allowed") is not False:
        raise ValueError("OU v5 attribution does not prohibit holdout reuse")
    immutable = _resolve(root, payload.get("immutable_attribution_path"))
    if (
        immutable is None
        or not immutable.is_file()
        or _file_hash(immutable) != _text(payload.get("immutable_attribution_sha256"))
    ):
        raise ValueError("OU v5 immutable attribution binding is invalid")
    return payload


def _implementation_source_hash() -> str:
    source = "\n".join(
        inspect.getsource(function)
        for function in (
            _ou_v6_transform_selector,
            _ou_v6_profile_branch_features,
            _ou_v6_predictor,
            _ou_stationary_constrained_profile_fit_v4,
            _ou_trend_aware_profile_beta_v3_candidate,
            _ou_local_transform_trend_selector_v2,
        )
    )
    return sha256(source.encode("utf-8")).hexdigest()


def _validate_registration(
    root: Path,
    contract_path: Path,
    contract: dict[str, Any],
    receipt: dict[str, Any],
    *,
    allow_responses: bool = False,
) -> None:
    if (
        contract.get("schema_version") != SCHEMA_VERSION
        or receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION
        or contract.get("status") != "PREREGISTERED_WAITING_VENDOR_RESPONSES"
        or receipt.get("status") != "PREREGISTERED_WAITING_VENDOR_RESPONSES"
        or receipt.get("contract_path") != _relative(contract_path, root)
        or receipt.get("contract_sha256") != _file_hash(contract_path)
        or receipt.get("implementation_source_sha256") != _implementation_source_hash()
        or int(contract.get("vendor_responses_at_registration", -1)) != 0
        or int(receipt.get("vendor_responses_at_registration", -1)) != 0
        or len(contract.get("holdout_bindings", [])) != 8
        or receipt.get("prospectively_registered") is not True
        or receipt.get("v5_holdout_reuse_allowed") is not False
        or contract.get("hypothesis_changes_after_vendor_response_allowed") is not False
        or receipt.get("final_successor_iteration") is not True
        or contract.get("final_successor_iteration") is not True
        or receipt.get("successor_after_v6_failure_allowed") is not False
        or contract.get("successor_after_v6_failure_allowed") is not False
        or contract.get("research_only") is not True
        or receipt.get("research_only") is not True
        or any(
            _truthy(contract.get(field)) or _truthy(receipt.get(field))
            for field in (
                "candidate_promotion_authority",
                "testnet_order_authority",
                "live_trading_authorized",
            )
        )
    ):
        raise ValueError("OU v6 existing registration validation failed")
    for binding in contract["holdout_bindings"]:
        request_path = root / _text(binding.get("request_path"))
        response_path = root / _text(binding.get("response_path"))
        if (
            not request_path.is_file()
            or _file_hash(request_path) != _text(binding.get("request_sha256"))
            or (response_path.exists() and not allow_responses)
        ):
            raise ValueError("OU v6 existing registration raw binding failed")


def _resolve(root: Path, value: object) -> Path | None:
    raw = _text(value)
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_absolute() else root / path
