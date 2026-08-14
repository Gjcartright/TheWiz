from __future__ import annotations

import inspect
import json
import os
from collections.abc import Callable
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from quant_platform.active_pipeline import CommandResult
from quant_platform.api_extraction import CryptoWizardsFetchError
from quant_platform.crypto_wizards_history import (
    CryptoWizardsCustomSeriesBacktestRequest,
    fetch_credits_used,
    fetch_custom_series_backtest,
)
from quant_platform.crypto_wizards_sweep import parse_wizard_credit_usage
from quant_platform.wizard_hyperliquid_mode_proof import (
    _ou_trend_aware_profile_beta_v3_candidate,
    _ou_trend_aware_profile_spread_v3_candidate,
    _ou_zero_mean_profile_beta_v2_holdout,
    _ou_zero_mean_profile_spread_candidate_v2_holdout,
)
from quant_platform.wizard_ou_comparator_activation import (
    build_ou_v3_review_packet as _build_ou_v3_review_packet,
)
from quant_platform.wizard_ou_comparator_activation import (
    build_ou_v3_supersession_gate as _build_ou_v3_supersession_gate,
)
from quant_platform.wizard_ou_comparator_activation import (
    build_reviewed_ou_v3_activation as _build_reviewed_ou_v3_activation,
)

ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "thewiz.wizard_ou_v2_blind_holdout.v1"
OU_MODES = ("OU (Spread)", "OU (ZScoreR)")
ORIENTATIONS = ("original", "reverse")
DEFAULT_DERIVATION_GROUP = "wpair_a857c606d53e9206f2d1"
DEFAULT_HOLDOUT_GROUP = "wpair_4feb406df129644896e3"
DEFAULT_V3_ASSET_X_PATH = (
    "reports/snapshots/current_wizard_hyperliquid/"
    "ewapi_42f1d0835347a819409d/cwhandoff_6ed1842867bdb5342374/"
    "history_runs/cwhistoryrun_20260809T112207668978Z_aa2faebe/"
    "assets/BTC_1d_candles.json"
)
DEFAULT_V3_ASSET_Y_PATH = (
    "reports/snapshots/current_wizard_hyperliquid/"
    "ewapi_42f1d0835347a819409d/cwhandoff_6ed1842867bdb5342374/"
    "history_runs/cwhistoryrun_20260809T112207668978Z_aa2faebe/"
    "assets/ETH_1d_candles.json"
)
OU_TREND_SELECTOR_SCHEMA_VERSION = "thewiz.wizard_ou_trend_selector.v1"
OU_TREND_SELECTOR_CONTRACT_PATH = Path(
    "config/wizard_ou_trend_selector_v1_holdout.json"
)
OU_TREND_SELECTOR_RECEIPT_PATH = Path(
    "reports/active/wizard_ou_trend_selector_v1_receipt.json"
)
OU_TREND_SELECTOR_DERIVATION_PATH = Path(
    "reports/active/wizard_ou_trend_selector_v1_derivation.csv"
)
OU_TREND_SELECTOR_PREDICTIONS_PATH = Path(
    "reports/active/wizard_ou_trend_selector_v1_predictions.csv"
)


def build_ou_v3_supersession_gate(*, root: Path = ROOT) -> CommandResult:
    """Publish the fail-closed gate before any OU-v3 human review."""

    return _build_ou_v3_supersession_gate(
        root=root,
        implementation_source_sha256=_v3_implementation_source_hash(),
        selector_source_sha256=_ou_trend_selector_source_hash(),
    )


def build_ou_v3_review_handoff(
    *, root: Path = ROOT, now: datetime | None = None
) -> CommandResult:
    """Publish the immutable OU-v3 packet a reviewer must approve."""

    return _build_ou_v3_review_packet(
        root=root,
        implementation_source_sha256=_v3_implementation_source_hash(),
        selector_source_sha256=_ou_trend_selector_source_hash(),
        now=now,
    )


def build_ou_v3_reviewed_activation(
    *,
    root: Path = ROOT,
    apply: bool = False,
    reviewer: str = "",
    review_note: str = "",
    review_packet_id: str = "",
    now: datetime | None = None,
) -> CommandResult:
    """Plan or explicitly apply the reviewed OU-v3 research comparator."""

    return _build_reviewed_ou_v3_activation(
        root=root,
        implementation_source_sha256=_v3_implementation_source_hash(),
        selector_source_sha256=_ou_trend_selector_source_hash(),
        apply=apply,
        reviewer=reviewer,
        review_note=review_note,
        review_packet_id=review_packet_id,
        now=now,
    )


def register_ou_trend_selector_v1_holdout(
    *,
    root: Path = ROOT,
    now: datetime | None = None,
    v2_contract_path: Path | None = None,
    v3_contract_path: Path | None = None,
) -> CommandResult:
    """Preregister the local OU trend selector before v3 vendor capture."""

    selector_contract_path = root / OU_TREND_SELECTOR_CONTRACT_PATH
    selector_receipt_path = root / OU_TREND_SELECTOR_RECEIPT_PATH
    if selector_contract_path.exists() or selector_receipt_path.exists():
        if not selector_contract_path.is_file() or not selector_receipt_path.is_file():
            raise ValueError("OU trend selector registration is incomplete")
        contract = _read_json(selector_contract_path)
        receipt = _read_json(selector_receipt_path)
        _validate_ou_trend_selector_registration(
            root, selector_contract_path, contract, receipt
        )
        return CommandResult(
            paths={
                "contract": selector_contract_path,
                "receipt": selector_receipt_path,
                "derivation": root / OU_TREND_SELECTOR_DERIVATION_PATH,
                "predictions": root / OU_TREND_SELECTOR_PREDICTIONS_PATH,
            },
            summary=receipt,
        )

    v2_contract_path = v2_contract_path or (
        root / "config" / "wizard_ou_comparator_v2_holdout.json"
    )
    v3_contract_path = v3_contract_path or (
        root / "config" / "wizard_ou_comparator_v3_holdout.json"
    )
    v2_contract = _read_json(v2_contract_path)
    v3_contract = _read_json(v3_contract_path)
    if _text(v2_contract.get("schema_version")) != SCHEMA_VERSION:
        raise ValueError("OU trend selector requires the registered OU v2 contract")
    if (
        _text(v3_contract.get("schema_version"))
        != "thewiz.wizard_ou_v3_prospective_holdout.v1"
    ):
        raise ValueError("OU trend selector requires the registered OU v3 contract")

    derivation_rows: list[dict[str, object]] = []
    source_groups = (
        ("v2_derivation", v2_contract.get("derivation_bindings", [])),
        ("v2_blind_holdout", v2_contract.get("sealed_holdout_bindings", [])),
    )
    for cohort, bindings in source_groups:
        if not isinstance(bindings, list) or len(bindings) != 4:
            raise ValueError(f"OU trend selector requires four {cohort} cells")
        for binding in bindings:
            request_path = root / _text(binding.get("request_path"))
            response_path = root / _text(binding.get("response_path"))
            if _file_hash(request_path) != _text(binding.get("request_sha256")):
                raise ValueError("OU trend selector derivation request hash mismatch")
            if _file_hash(response_path) != _text(binding.get("response_sha256")):
                raise ValueError("OU trend selector derivation response hash mismatch")
            request = _read_json(request_path)
            response = _read_json(response_path)
            vendor_inc_trend = _vendor_inc_trend(response)
            selector = _ou_local_trend_selector_v1_from_request(request)
            local_inc_trend = bool(selector["local_predicted_inc_trend"])
            derivation_rows.append(
                {
                    "cohort": cohort,
                    "exact_mode": _text(binding.get("exact_mode")),
                    "orientation": _text(binding.get("orientation")),
                    "proof_observations": int(binding.get("proof_observations", 0)),
                    **selector,
                    "vendor_inc_trend": vendor_inc_trend,
                    "selector_match": local_inc_trend == vendor_inc_trend,
                    "request_path": _relative(request_path, root),
                    "request_sha256": _file_hash(request_path),
                    "response_path": _relative(response_path, root),
                    "response_sha256": _file_hash(response_path),
                }
            )
    derivation = pd.DataFrame(derivation_rows)
    if len(derivation) != 8 or not derivation["selector_match"].all():
        raise ValueError("OU trend selector does not reproduce all eight known cells")

    prediction_rows: list[dict[str, object]] = []
    holdout_bindings = v3_contract.get("holdout_bindings", [])
    if not isinstance(holdout_bindings, list) or len(holdout_bindings) != 4:
        raise ValueError("OU trend selector requires four OU v3 holdout cells")
    for binding in holdout_bindings:
        request_path = root / _text(binding.get("request_path"))
        response_path = root / _text(binding.get("response_path"))
        if response_path.exists():
            raise ValueError(
                "OU trend selector must be registered before v3 vendor responses"
            )
        if _file_hash(request_path) != _text(binding.get("request_sha256")):
            raise ValueError("OU trend selector holdout request hash mismatch")
        selector = _ou_local_trend_selector_v1_from_request(
            _read_json(request_path)
        )
        prediction_rows.append(
            {
                "pair": _text(binding.get("pair")),
                "exact_mode": _text(binding.get("exact_mode")),
                "orientation": _text(binding.get("orientation")),
                "proof_observations": int(binding.get("proof_observations", 0)),
                **selector,
                "request_path": _relative(request_path, root),
                "request_sha256": _file_hash(request_path),
                "response_path": _relative(response_path, root),
                "vendor_response_present_at_registration": False,
            }
        )
    predictions = pd.DataFrame(prediction_rows)
    derivation_path = root / OU_TREND_SELECTOR_DERIVATION_PATH
    predictions_path = root / OU_TREND_SELECTOR_PREDICTIONS_PATH
    _write_or_validate_immutable_csv(derivation, derivation_path)
    _write_or_validate_immutable_csv(predictions, predictions_path)

    source_hash = _ou_trend_selector_source_hash()
    contract = {
        "schema_version": OU_TREND_SELECTOR_SCHEMA_VERSION,
        "status": "PREREGISTERED_WAITING_VENDOR_RESPONSES",
        "selector": {
            "generation": 1,
            "source_sha256": source_hash,
            "candidate_branches": {
                "constant": "intercept_ar1_profile_spread_then_adf_c",
                "trend": "zero_mean_ar1_profile_spread_then_adf_ct",
            },
            "adf_maxlag": 2,
            "adf_autolag": None,
            "criterion": "aic",
            "rule": "inc_trend=true only when trend_aic<constant_aic",
            "tie_breaker": "constant_branch",
            "uses_vendor_response": False,
            "point_in_time_inputs": "registered_request_series_only",
        },
        "derivation": {
            "cells": 8,
            "independent_series_orientations": 4,
            "matched_cells": int(derivation["selector_match"].sum()),
            "path": _relative(derivation_path, root),
            "sha256": _file_hash(derivation_path),
        },
        "holdout_predictions": {
            "cells": 4,
            "independent_series_orientations": 2,
            "vendor_responses_at_registration": 0,
            "path": _relative(predictions_path, root),
            "sha256": _file_hash(predictions_path),
        },
        "source_contracts": [
            {
                "path": _relative(v2_contract_path, root),
                "sha256": _file_hash(v2_contract_path),
            },
            {
                "path": _relative(v3_contract_path, root),
                "sha256": _file_hash(v3_contract_path),
            },
        ],
        "automatic_activation": False,
        "research_only": True,
        "candidate_promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    _write_or_validate_immutable_json(contract, selector_contract_path)
    receipt = {
        "schema_version": "thewiz.wizard_ou_trend_selector_receipt.v1",
        "registered_at_utc": _as_utc(now).isoformat(),
        "status": "REGISTERED_WAITING_VENDOR_RESPONSES",
        "contract_path": _relative(selector_contract_path, root),
        "contract_sha256": _file_hash(selector_contract_path),
        "selector_source_sha256": source_hash,
        "derivation_cells": 8,
        "derivation_matched_cells": 8,
        "holdout_prediction_cells": 4,
        "vendor_responses_at_registration": 0,
        "prospectively_registered": True,
        "research_only": True,
        "candidate_promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    _write_or_validate_immutable_json(receipt, selector_receipt_path)
    return CommandResult(
        paths={
            "contract": selector_contract_path,
            "receipt": selector_receipt_path,
            "derivation": derivation_path,
            "predictions": predictions_path,
        },
        summary=receipt,
    )


def run_ou_v3_prospective_holdout(
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
    """Capture the preregistered OU-v3 holdout once, then evaluate it locally."""

    timestamp = _as_utc(now)
    contract_path = root / "config" / "wizard_ou_comparator_v3_holdout.json"
    receipt_path = root / "reports" / "active" / "wizard_ou_v3_holdout_receipt.json"
    contract = _read_json(contract_path)
    receipt = _read_json(receipt_path)
    _validate_v3_registration(root, contract_path, contract, receipt)
    selector_contract_path = root / OU_TREND_SELECTOR_CONTRACT_PATH
    selector_receipt_path = root / OU_TREND_SELECTOR_RECEIPT_PATH
    _validate_ou_trend_selector_registration(
        root,
        selector_contract_path,
        _read_json(selector_contract_path),
        _read_json(selector_receipt_path),
    )
    bindings = contract.get("holdout_bindings", [])
    if not isinstance(bindings, list) or len(bindings) != 4:
        raise ValueError("OU v3 contract must bind exactly four holdout cells")

    missing = [
        binding
        for binding in bindings
        if not (root / _text(binding.get("response_path"))).is_file()
    ]
    key = api_key or os.getenv("CRYPTO_WIZARDS_API_KEY", "").strip()
    credits_used: int | None = None
    blocker = ""
    if execute and missing:
        if not key:
            blocker = "crypto_wizards_api_key_missing"
        else:
            try:
                usage = parse_wizard_credit_usage(
                    credits_fetcher(api_key=key),
                    configured_limit=daily_credit_limit,
                )
                if not usage.known or usage.used is None:
                    blocker = "credit_usage_unknown"
                else:
                    credits_used = usage.used
            except (CryptoWizardsFetchError, OSError, TypeError, ValueError) as exc:
                blocker = f"credit_preflight_failed:{type(exc).__name__}:{exc}"
            required_credits = len(missing) * 2
            if (
                credits_used is not None
                and credits_used + required_credits
                > daily_credit_limit - reserved_credits
            ):
                blocker = "insufficient_reserved_crypto_wizards_credits"

    attempt_path = (
        root
        / "data"
        / "research"
        / "wizard_ou_v3_holdout"
        / "attempts"
        / f"{timestamp.date().isoformat()}.json"
    )
    attempt_already_exists = execute and bool(missing) and attempt_path.exists()
    if attempt_already_exists:
        blocker = "ou_v3_daily_attempt_already_registered"

    calls_made = 0
    responses_captured = 0
    errors: list[str] = []
    if execute and missing and not blocker:
        for binding in missing:
            request_path = root / _text(binding.get("request_path"))
            if _file_hash(request_path) != _text(binding.get("request_sha256")):
                errors.append("ou_v3_request_hash_mismatch")
                break
            request = _request_from_payload(_read_json(request_path))
            calls_made += 1
            try:
                response = fetcher(request, api_key=key)
                response_path = root / _text(binding.get("response_path"))
                _write_or_validate_immutable_json(response, response_path)
                responses_captured += 1
            except (CryptoWizardsFetchError, OSError, TypeError, ValueError) as exc:
                errors.append(f"vendor_capture_failed:{type(exc).__name__}:{exc}")
                break

    attempt = {
        "schema_version": "thewiz.wizard_ou_v3_capture_attempt.v1",
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
        "credits_attempted": calls_made * 2,
        "credits_completed": responses_captured * 2,
        "contract_sha256": _file_hash(contract_path),
        "research_only": True,
        "candidate_promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    if execute and missing and not attempt_already_exists:
        _write_or_validate_immutable_json(attempt, attempt_path)

    responses_now = sum(
        (root / _text(binding.get("response_path"))).is_file()
        for binding in bindings
    )
    if responses_now == 4:
        evaluation = evaluate_ou_v3_prospective_holdout(
            root=root, now=timestamp, contract_path=contract_path
        )
        evaluation_status = _text(evaluation.summary.get("status"))
        paths = dict(evaluation.paths)
    else:
        evaluation_status = "WAITING_VENDOR_RESPONSES"
        paths = {}
    status_payload = {
        **attempt,
        "schema_version": "thewiz.wizard_ou_v3_capture_status.v1",
        "responses_available": responses_now,
        "required_responses": 4,
        "evaluation_status": evaluation_status,
        "next_step": (
            "evaluate_registered_holdout"
            if responses_now == 4
            else "wait_for_next_eligible_credit_window"
        ),
    }
    status_path = root / "reports" / "active" / "wizard_ou_v3_capture_status.json"
    _atomic_json(status_payload, status_path)
    paths["capture_status"] = status_path
    if attempt_path.is_file():
        paths["attempt"] = attempt_path
    return CommandResult(paths=paths, summary=status_payload)


def evaluate_ou_v3_prospective_holdout(
    *,
    root: Path = ROOT,
    now: datetime | None = None,
    contract_path: Path | None = None,
) -> CommandResult:
    """Evaluate all four prospectively bound OU-v3 responses."""

    contract_path = contract_path or (
        root / "config" / "wizard_ou_comparator_v3_holdout.json"
    )
    receipt_path = root / "reports" / "active" / "wizard_ou_v3_holdout_receipt.json"
    contract = _read_json(contract_path)
    receipt = _read_json(receipt_path)
    _validate_v3_registration(root, contract_path, contract, receipt)
    selector_contract_path = root / OU_TREND_SELECTOR_CONTRACT_PATH
    selector_receipt_path = root / OU_TREND_SELECTOR_RECEIPT_PATH
    selector_contract = _read_json(selector_contract_path)
    selector_receipt = _read_json(selector_receipt_path)
    _validate_ou_trend_selector_registration(
        root, selector_contract_path, selector_contract, selector_receipt
    )
    selector_predictions = _read_csv(root / OU_TREND_SELECTOR_PREDICTIONS_PATH)
    tolerances = contract.get("tolerances", {})
    rows: list[dict[str, object]] = []
    for binding in contract.get("holdout_bindings", []):
        request_path = root / _text(binding.get("request_path"))
        response_path = root / _text(binding.get("response_path"))
        if not response_path.is_file():
            raise ValueError("OU v3 evaluation requires all four vendor responses")
        if _file_hash(request_path) != _text(binding.get("request_sha256")):
            raise ValueError("OU v3 request hash mismatch")
        prediction = _selector_prediction_for_binding(
            selector_predictions, binding=binding
        )
        request = _read_json(request_path)
        current_selector = _ou_local_trend_selector_v1_from_request(request)
        registered_inc_trend = _truthy(
            prediction.get("local_predicted_inc_trend")
        )
        if (
            bool(current_selector["local_predicted_inc_trend"])
            != registered_inc_trend
        ):
            raise ValueError("OU v3 registered selector prediction mismatch")
        metrics = _evaluate_response_v3(
            request=request,
            response=_read_json(response_path),
            tolerances=tolerances,
            local_inc_trend=registered_inc_trend,
        )
        passed = bool(metrics.pop("passed"))
        rows.append(
            {
                "pair": _text(binding.get("pair")),
                "exact_mode": _text(binding.get("exact_mode")),
                "orientation": _text(binding.get("orientation")),
                "cell_status": "PASS" if passed else "FAIL",
                "selector_constant_aic": float(
                    current_selector["constant_aic"]
                ),
                "selector_trend_aic": float(current_selector["trend_aic"]),
                "selector_aic_delta_trend_minus_constant": float(
                    current_selector["aic_delta_trend_minus_constant"]
                ),
                **metrics,
                "request_path": _relative(request_path, root),
                "request_sha256": _file_hash(request_path),
                "response_path": _relative(response_path, root),
                "response_sha256": _file_hash(response_path),
                "blocker": "" if passed else "ou_v3_prospective_holdout_parity_failed",
                "candidate_promotion_authority": False,
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            }
        )
    frame = pd.DataFrame(rows)
    passed_cells = int(frame["cell_status"].eq("PASS").sum())
    formula_passed_cells = int(frame["formula_parity_passed"].map(_truthy).sum())
    selector_passed_cells = int(frame["trend_selector_parity_passed"].map(_truthy).sum())
    status = "PASS" if passed_cells == 4 else "FAIL"
    detail_path = root / "reports" / "active" / "wizard_ou_v3_holdout_evaluation.csv"
    status_path = root / "reports" / "active" / "wizard_ou_v3_holdout_status.json"
    _atomic_csv(frame, detail_path)
    payload = {
        "schema_version": "thewiz.wizard_ou_v3_holdout_evaluation.v1",
        "evaluated_at_utc": _as_utc(now).isoformat(),
        "status": status,
        "contract_path": _relative(contract_path, root),
        "contract_sha256": _file_hash(contract_path),
        "holdout_pair": "BTC-ETH",
        "required_cells": 4,
        "passed_cells": passed_cells,
        "failed_cells": 4 - passed_cells,
        "formula_parity_passed_cells": formula_passed_cells,
        "trend_selector_parity_passed_cells": selector_passed_cells,
        "comparator_supersession_eligible": status == "PASS",
        "comparator_supersession_automatic": False,
        "local_point_in_time_trend_selector_proven": selector_passed_cells == 4,
        "selector_contract_path": _relative(selector_contract_path, root),
        "selector_contract_sha256": _file_hash(selector_contract_path),
        "selector_source_sha256": _ou_trend_selector_source_hash(),
        "response_sha256s": sorted(frame["response_sha256"].map(_text).tolist()),
        "research_only": True,
        "candidate_promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
        "evidence_path": _relative(detail_path, root),
    }
    immutable_payload = {
        key: value
        for key, value in payload.items()
        if key not in {"evaluated_at_utc", "evidence_path"}
    }
    result_id = "ouv3holdout_" + sha256(
        json.dumps(
            immutable_payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()[:24]
    immutable_payload["result_id"] = result_id
    immutable_path = (
        root
        / "data"
        / "research"
        / "wizard_ou_v3_holdout_evaluations"
        / f"{result_id}.json"
    )
    _write_or_validate_immutable_json(immutable_payload, immutable_path)
    payload.update(
        {
            "result_id": result_id,
            "immutable_result_path": _relative(immutable_path, root),
            "immutable_result_sha256": _file_hash(immutable_path),
        }
    )
    _atomic_json(payload, status_path)
    return CommandResult(
        paths={
            "detail": detail_path,
            "status": status_path,
            "immutable_result": immutable_path,
        },
        summary=payload,
    )


def register_ou_v3_prospective_holdout(
    *,
    root: Path = ROOT,
    now: datetime | None = None,
    asset_x_path: Path | None = None,
    asset_y_path: Path | None = None,
    observations: int = 360,
) -> CommandResult:
    """Freeze a fresh four-cell BTC/ETH holdout without calling the vendor."""

    if observations < 50:
        raise ValueError("OU v3 prospective holdout requires at least 50 observations")
    asset_x_path = asset_x_path or root / DEFAULT_V3_ASSET_X_PATH
    asset_y_path = asset_y_path or root / DEFAULT_V3_ASSET_Y_PATH
    aligned = _aligned_candles(asset_x_path, asset_y_path, observations=observations)
    request_dir = root / "data" / "research" / "wizard_ou_v3_holdout" / "requests"
    response_dir = root / "data" / "raw" / "wizard_ou_v3_holdout" / "responses"
    bindings: list[dict[str, object]] = []
    mode_params = {
        "OU (Spread)": "Spread",
        "OU (ZScoreR)": "ZScoreRoll",
    }
    for exact_mode, strategy in mode_params.items():
        for orientation in ORIENTATIONS:
            if orientation == "original":
                first, second = aligned["x"], aligned["y"]
                pair = "BTC-ETH"
            else:
                first, second = aligned["y"], aligned["x"]
                pair = "ETH-BTC"
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
            slug = f"{pair.lower().replace('-', '_')}_{strategy.lower()}"
            request_path = request_dir / f"{slug}_request.json"
            response_path = response_dir / f"{slug}_response.json"
            _write_or_validate_immutable_json(request.payload(), request_path)
            bindings.append(
                {
                    "pair": pair,
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
                    "response_captured_at_registration": response_path.is_file(),
                }
            )
    if any(item["response_captured_at_registration"] for item in bindings):
        raise ValueError("OU v3 vendor response existed before prospective registration")

    source_hash = _v3_implementation_source_hash()
    v2_status_path = root / "reports" / "active" / "wizard_ou_v2_holdout_status.json"
    v2_detail_path = root / "reports" / "active" / "wizard_ou_v2_holdout_evaluation.csv"
    contract = {
        "schema_version": "thewiz.wizard_ou_v3_prospective_holdout.v1",
        "status": "PREREGISTERED_WAITING_VENDOR_RESPONSES",
        "hypothesis": {
            "derivation_cohorts": [DEFAULT_DERIVATION_GROUP, DEFAULT_HOLDOUT_GROUP],
            "observed_branch_rule": (
                "coint_eg.inc_trend=true uses zero_mean_ar1_profile; "
                "coint_eg.inc_trend=false uses intercept_ar1_profile"
            ),
            "holdout_pair": "BTC-ETH",
            "holdout_is_disjoint": True,
        },
        "implementation": {
            "generation": 3,
            "source_sha256": source_hash,
            "input_transform": "divide_each_leg_by_its_first_observation",
            "spread": "normalized_y-beta*normalized_x_without_centering",
            "trend_selector_source": "vendor_coint_eg.inc_trend_for_formula_provenance",
            "point_in_time_note": (
                "local walk-forward use requires an independently matched local trend selector"
            ),
        },
        "tolerances": {
            "hedge_ratio_abs": 1e-6,
            "spread_max_abs": 1e-6,
            "zscore_max_abs": 1e-6,
            "zscore_roll_max_abs": 1e-6,
            "half_life_abs": 1e-3,
        },
        "source_candles": [
            {
                "asset": "BTC",
                "path": _relative(asset_x_path, root),
                "sha256": _file_hash(asset_x_path),
            },
            {
                "asset": "ETH",
                "path": _relative(asset_y_path, root),
                "sha256": _file_hash(asset_y_path),
            },
        ],
        "derivation_evidence": [
            {
                "path": _relative(v2_status_path, root),
                "sha256": _file_hash(v2_status_path),
            },
            {
                "path": _relative(v2_detail_path, root),
                "sha256": _file_hash(v2_detail_path),
            },
        ],
        "holdout_bindings": bindings,
        "vendor_responses_at_registration": 0,
        "threshold_changes_after_vendor_response_allowed": False,
        "automatic_activation": False,
        "research_only": True,
        "candidate_promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    contract_path = root / "config" / "wizard_ou_comparator_v3_holdout.json"
    _write_or_validate_immutable_json(contract, contract_path)
    receipt = {
        "schema_version": "thewiz.wizard_ou_v3_prospective_holdout_receipt.v1",
        "registered_at_utc": _as_utc(now).isoformat(),
        "status": "REGISTERED_WAITING_VENDOR_RESPONSES",
        "contract_path": _relative(contract_path, root),
        "contract_sha256": _file_hash(contract_path),
        "implementation_source_sha256": source_hash,
        "holdout_pair": "BTC-ETH",
        "required_cells": 4,
        "vendor_responses_at_registration": 0,
        "prospectively_registered": True,
        "research_only": True,
        "candidate_promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    receipt_path = root / "reports" / "active" / "wizard_ou_v3_holdout_receipt.json"
    _atomic_json(receipt, receipt_path)
    return CommandResult(
        paths={"contract": contract_path, "receipt": receipt_path}, summary=receipt
    )


def register_ou_v2_blind_holdout(
    *,
    root: Path = ROOT,
    now: datetime | None = None,
    proof_path: Path | None = None,
    derivation_group: str = DEFAULT_DERIVATION_GROUP,
    holdout_group: str = DEFAULT_HOLDOUT_GROUP,
) -> CommandResult:
    """Seal OU-v2 source, tolerances, and response identities before inspection."""

    if not derivation_group or derivation_group == holdout_group:
        raise ValueError("OU v2 holdout must be disjoint from its derivation cohort")
    proof_path = proof_path or (
        root / "reports" / "active" / "hyperliquid_wizard_vendor_mode_proofs.csv"
    )
    proofs = _read_csv(proof_path)
    expected = {(mode, orientation) for mode in OU_MODES for orientation in ORIENTATIONS}
    derivation = _select_cells(proofs, derivation_group)
    holdout = _select_cells(proofs, holdout_group)
    _require_exact_cells(derivation, expected, label="derivation")
    _require_exact_cells(holdout, expected, label="holdout")

    derivation_bindings = _cell_bindings(root, derivation, hash_files=True)
    holdout_bindings = _cell_bindings(root, holdout, hash_files=True)
    source_hash = _implementation_source_hash()
    contract = {
        "schema_version": SCHEMA_VERSION,
        "status": "PREREGISTERED_BLINDED",
        "derivation_cohort": {
            "pair_group_id": derivation_group,
            "required_exact_modes": list(OU_MODES),
            "required_orientations": list(ORIENTATIONS),
            "required_mode_orientation_cells": 4,
        },
        "holdout_cohort": {
            "pair_group_id": holdout_group,
            "required_exact_modes": list(OU_MODES),
            "required_orientations": list(ORIENTATIONS),
            "required_mode_orientation_cells": 4,
        },
        "implementation": {
            "generation": 2,
            "source_sha256": source_hash,
            "input_transform": "divide_each_leg_by_its_first_observation",
            "hedge_ratio_objective": (
                "minimize_zero_mean_ar1_one_step_residual_mean_square"
            ),
            "spread": "normalized_y-beta*normalized_x_without_centering",
            "displayed_half_life": (
                "-ln(2)/ln(phi), phi_from_ar1_with_intercept_on_spread"
            ),
            "optimizer": "deterministic_513_point_bracket_then_bounded_scalar_refinement",
        },
        "tolerances": {
            "hedge_ratio_abs": 1e-6,
            "spread_max_abs": 1e-6,
            "zscore_max_abs": 1e-6,
            "zscore_roll_max_abs": 1e-6,
            "half_life_abs": 1e-3,
        },
        "derivation_bindings": derivation_bindings,
        "sealed_holdout_bindings": holdout_bindings,
        "proof_ledger_path": _relative(proof_path, root),
        "registered_after_vendor_capture_but_before_holdout_raw_series_inspection": True,
        "research_only": True,
        "automatic_activation": False,
        "candidate_promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    contract_path = root / "config" / "wizard_ou_comparator_v2_holdout.json"
    _write_or_validate_immutable_json(contract, contract_path)
    receipt = {
        "schema_version": "thewiz.wizard_ou_v2_blind_holdout_receipt.v1",
        "registered_at_utc": _as_utc(now).isoformat(),
        "status": "REGISTERED_BLINDED",
        "contract_path": _relative(contract_path, root),
        "contract_sha256": _file_hash(contract_path),
        "implementation_source_sha256": source_hash,
        "derivation_pair_group_id": derivation_group,
        "holdout_pair_group_id": holdout_group,
        "sealed_holdout_cells": len(holdout_bindings),
        "registered_before_holdout_raw_series_inspection": True,
        "research_only": True,
        "candidate_promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    receipt_path = (
        root / "reports" / "active" / "wizard_ou_comparator_v2_holdout_receipt.json"
    )
    _atomic_json(receipt, receipt_path)
    return CommandResult(
        paths={"contract": contract_path, "receipt": receipt_path}, summary=receipt
    )


def evaluate_ou_v2_blind_holdout(
    *,
    root: Path = ROOT,
    now: datetime | None = None,
    contract_path: Path | None = None,
    proof_path: Path | None = None,
) -> CommandResult:
    """Open and evaluate only the sealed OU-v2 holdout identities."""

    contract_path = contract_path or (
        root / "config" / "wizard_ou_comparator_v2_holdout.json"
    )
    receipt_path = (
        root / "reports" / "active" / "wizard_ou_comparator_v2_holdout_receipt.json"
    )
    contract = _read_json(contract_path)
    receipt = _read_json(receipt_path)
    _validate_registration(root, contract_path, contract, receipt)
    proof_path = proof_path or root / _text(contract.get("proof_ledger_path"))
    proofs = _read_csv(proof_path)
    holdout_group = _text(contract["holdout_cohort"].get("pair_group_id"))
    selected = _select_cells(proofs, holdout_group)
    expected = {(mode, orientation) for mode in OU_MODES for orientation in ORIENTATIONS}
    _require_exact_cells(selected, expected, label="holdout")
    sealed = {
        (_text(item.get("exact_mode")), _text(item.get("orientation"))): item
        for item in contract.get("sealed_holdout_bindings", [])
    }
    tolerances = contract.get("tolerances", {})

    rows: list[dict[str, object]] = []
    for _, proof in selected.sort_values(["exact_mode", "orientation"]).iterrows():
        identity = (_text(proof.get("exact_mode")), _text(proof.get("orientation")))
        binding = sealed.get(identity)
        if binding is None:
            raise ValueError(f"unregistered OU v2 holdout identity: {identity}")
        request_path = _resolve(root, proof.get("request_path"))
        response_path = _resolve(root, proof.get("response_path"))
        if request_path is None or response_path is None:
            raise ValueError(f"OU v2 holdout paths missing: {identity}")
        if _file_hash(request_path) != _text(binding.get("request_sha256")):
            raise ValueError(f"OU v2 sealed request hash mismatch: {identity}")
        if _file_hash(response_path) != _text(binding.get("response_sha256")):
            raise ValueError(f"OU v2 sealed response hash mismatch: {identity}")
        metrics = _evaluate_response(
            request=_read_json(request_path),
            response=_read_json(response_path),
            tolerances=tolerances,
        )
        passed = bool(metrics.pop("passed"))
        rows.append(
            {
                "pair_group_id": holdout_group,
                "exact_mode": identity[0],
                "orientation": identity[1],
                "cell_status": "PASS" if passed else "FAIL",
                **metrics,
                "request_sha256": _file_hash(request_path),
                "response_sha256": _file_hash(response_path),
                "request_path": _relative(request_path, root),
                "response_path": _relative(response_path, root),
                "blocker": "" if passed else "ou_v2_blind_holdout_numerical_parity_failed",
                "candidate_promotion_authority": False,
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            }
        )

    frame = pd.DataFrame(rows)
    passed_cells = int(frame["cell_status"].eq("PASS").sum())
    status = "PASS" if passed_cells == 4 else "FAIL"
    active = root / "reports" / "active"
    detail_path = active / "wizard_ou_v2_holdout_evaluation.csv"
    status_path = active / "wizard_ou_v2_holdout_status.json"
    _atomic_csv(frame, detail_path)
    payload = {
        "schema_version": "thewiz.wizard_ou_v2_holdout_evaluation.v1",
        "evaluated_at_utc": _as_utc(now).isoformat(),
        "status": status,
        "contract_path": _relative(contract_path, root),
        "contract_sha256": _file_hash(contract_path),
        "derivation_pair_group_id": _text(
            contract["derivation_cohort"].get("pair_group_id")
        ),
        "holdout_pair_group_id": holdout_group,
        "required_cells": 4,
        "passed_cells": passed_cells,
        "failed_cells": 4 - passed_cells,
        "comparator_supersession_eligible": status == "PASS",
        "comparator_supersession_automatic": False,
        "research_only": True,
        "candidate_promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
        "evidence_path": _relative(detail_path, root),
    }
    immutable_payload = {
        key: value for key, value in payload.items() if key != "evaluated_at_utc"
    }
    result_id = "ouholdout_" + sha256(
        json.dumps(immutable_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:24]
    immutable_payload["result_id"] = result_id
    immutable_path = (
        root
        / "data"
        / "research"
        / "wizard_ou_holdout_evaluations"
        / f"{result_id}.json"
    )
    _write_or_validate_immutable_json(immutable_payload, immutable_path)
    payload.update(
        {
            "result_id": result_id,
            "immutable_result_path": _relative(immutable_path, root),
            "immutable_result_sha256": _file_hash(immutable_path),
        }
    )
    _atomic_json(payload, status_path)
    return CommandResult(
        paths={
            "detail": detail_path,
            "status": status_path,
            "immutable_result": immutable_path,
        },
        summary=payload,
    )


def _evaluate_response(
    *,
    request: dict[str, Any],
    response: dict[str, Any],
    tolerances: dict[str, Any],
) -> dict[str, object]:
    params = request.get("params") if isinstance(request.get("params"), dict) else {}
    history = response.get("history") if isinstance(response.get("history"), dict) else {}
    stats = history.get("spread_stats") if isinstance(history.get("spread_stats"), dict) else {}
    try:
        x = np.asarray(params["series_1_closes"], dtype=float)
        y = np.asarray(params["series_2_closes"], dtype=float)
        window = int(params["roll_w"])
        vendor_spread = np.asarray(stats["spread"], dtype=float)
        vendor_zscore = np.asarray(stats["zscore"], dtype=float)
        vendor_zscore_roll = np.asarray(stats["zscore_roll"], dtype=float)
        vendor_beta = float(stats["hedge_ratio"])
        vendor_half_life = float(stats["half_life"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("OU v2 response lacks required numeric evidence") from exc
    if len(x) < 50 or len({len(x), len(y), len(vendor_spread)}) != 1:
        raise ValueError("OU v2 holdout series are invalid")
    if bool(stats.get("log_used", False)):
        if np.any(x <= 0.0) or np.any(y <= 0.0):
            raise ValueError("OU v2 log transform requires positive prices")
        x, y = np.log(x), np.log(y)

    beta = _ou_zero_mean_profile_beta_v2_holdout(x, y)
    spread = _ou_zero_mean_profile_spread_candidate_v2_holdout(x, y)
    spread_std = float(spread.std(ddof=1))
    zscore = (spread - float(spread.mean())) / spread_std
    series = pd.Series(spread)
    rolling_mean = series.rolling(window, min_periods=window).mean()
    rolling_std = series.rolling(window, min_periods=window).std(ddof=1)
    zscore_roll = (
        series.sub(rolling_mean)
        .div(rolling_std.where(rolling_std.abs() > 1e-12))
        .fillna(0.0)
        .to_numpy()
    )
    lagged = spread[:-1]
    current = spread[1:]
    intercept, phi = np.linalg.lstsq(
        np.column_stack([np.ones(len(lagged)), lagged]), current, rcond=None
    )[0]
    del intercept
    half_life = float(-np.log(2.0) / np.log(phi)) if 0.0 < phi < 1.0 else np.nan
    errors = {
        "hedge_ratio_abs_error": abs(beta - vendor_beta),
        "spread_max_abs_error": float(np.max(np.abs(spread - vendor_spread))),
        "zscore_max_abs_error": float(np.max(np.abs(zscore - vendor_zscore))),
        "zscore_roll_max_abs_error": float(
            np.max(np.abs(zscore_roll - vendor_zscore_roll))
        ),
        "half_life_abs_error": abs(half_life - vendor_half_life),
    }
    limits = {
        "hedge_ratio_abs_error": float(tolerances["hedge_ratio_abs"]),
        "spread_max_abs_error": float(tolerances["spread_max_abs"]),
        "zscore_max_abs_error": float(tolerances["zscore_max_abs"]),
        "zscore_roll_max_abs_error": float(tolerances["zscore_roll_max_abs"]),
        "half_life_abs_error": float(tolerances["half_life_abs"]),
    }
    passed = all(
        np.isfinite(errors[key]) and errors[key] <= limits[key] for key in limits
    )
    return {
        "passed": passed,
        "local_hedge_ratio": beta,
        "vendor_hedge_ratio": vendor_beta,
        "local_half_life": half_life,
        "vendor_half_life": vendor_half_life,
        **errors,
        "formula_parity_status": (
            "validated_numerical_reconstruction" if passed else "mismatch"
        ),
        "formula_pit_status": "full_submitted_window_fit_not_live_signal_safe",
    }


# Frozen selector implementations are source-hash evidence.
# fmt: off
def _ou_local_trend_selector_v1(
    x: np.ndarray,
    y: np.ndarray,
) -> dict[str, object]:
    """Select the OU deterministic branch using branch-specific ADF AIC."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 50 or len(x) != len(y):
        raise ValueError("OU trend selector requires equal series of at least 50 rows")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("OU trend selector requires finite series")
    constant_spread = _ou_trend_aware_profile_spread_v3_candidate(
        x, y, inc_trend=False
    )
    trend_spread = _ou_trend_aware_profile_spread_v3_candidate(
        x, y, inc_trend=True
    )
    constant_result = adfuller(
        constant_spread,
        maxlag=2,
        regression="c",
        autolag=None,
        store=True,
        regresults=True,
    )
    trend_result = adfuller(
        trend_spread,
        maxlag=2,
        regression="ct",
        autolag=None,
        store=True,
        regresults=True,
    )
    constant_aic = float(constant_result[-1].resols.aic)
    trend_aic = float(trend_result[-1].resols.aic)
    if not np.isfinite(constant_aic) or not np.isfinite(trend_aic):
        raise ValueError("OU trend selector produced a non-finite AIC")
    return {
        "local_predicted_inc_trend": trend_aic < constant_aic,
        "constant_aic": constant_aic,
        "trend_aic": trend_aic,
        "aic_delta_trend_minus_constant": trend_aic - constant_aic,
        "constant_adf_t": float(constant_result[0]),
        "constant_adf_p": float(constant_result[1]),
        "trend_adf_t": float(trend_result[0]),
        "trend_adf_p": float(trend_result[1]),
        "adf_maxlag": 2,
        "adf_autolag": "none",
        "selector_rule": "trend_aic<constant_aic_else_constant",
    }


def _ou_local_trend_selector_v1_from_request(
    request: dict[str, Any],
) -> dict[str, object]:
    params = request.get("params") if isinstance(request.get("params"), dict) else {}
    try:
        x = np.asarray(params["series_1_closes"], dtype=float)
        y = np.asarray(params["series_2_closes"], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("OU trend selector request lacks numeric close series") from exc
    return _ou_local_trend_selector_v1(x, y)
# fmt: on


def _vendor_inc_trend(response: dict[str, Any]) -> bool:
    history = response.get("history") if isinstance(response.get("history"), dict) else {}
    coint = history.get("coint_eg") if isinstance(history.get("coint_eg"), dict) else {}
    if "inc_trend" not in coint:
        raise ValueError("OU response lacks coint_eg.inc_trend provenance")
    return _truthy(coint.get("inc_trend"))


def _selector_prediction_for_binding(
    predictions: pd.DataFrame,
    *,
    binding: dict[str, Any],
) -> dict[str, object]:
    if predictions.empty:
        raise ValueError("OU trend selector predictions are missing")
    mask = (
        predictions["exact_mode"].map(_text).eq(_text(binding.get("exact_mode")))
        & predictions["orientation"].map(_text).eq(
            _text(binding.get("orientation"))
        )
        & predictions["request_sha256"].map(_text).eq(
            _text(binding.get("request_sha256"))
        )
    )
    matches = predictions.loc[mask]
    if len(matches) != 1:
        raise ValueError("OU trend selector prediction identity mismatch")
    return matches.iloc[0].to_dict()


def _evaluate_response_v3(
    *,
    request: dict[str, Any],
    response: dict[str, Any],
    tolerances: dict[str, Any],
    local_inc_trend: bool,
) -> dict[str, object]:
    params = request.get("params") if isinstance(request.get("params"), dict) else {}
    history = response.get("history") if isinstance(response.get("history"), dict) else {}
    stats = history.get("spread_stats") if isinstance(history.get("spread_stats"), dict) else {}
    coint = history.get("coint_eg") if isinstance(history.get("coint_eg"), dict) else {}
    if "inc_trend" not in coint:
        raise ValueError("OU v3 response lacks coint_eg.inc_trend provenance")
    vendor_inc_trend = _truthy(coint.get("inc_trend"))
    try:
        x = np.asarray(params["series_1_closes"], dtype=float)
        y = np.asarray(params["series_2_closes"], dtype=float)
        window = int(params["roll_w"])
        vendor_spread = np.asarray(stats["spread"], dtype=float)
        vendor_zscore = np.asarray(stats["zscore"], dtype=float)
        vendor_zscore_roll = np.asarray(stats["zscore_roll"], dtype=float)
        vendor_beta = float(stats["hedge_ratio"])
        vendor_half_life = float(stats["half_life"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("OU v3 response lacks required numeric evidence") from exc
    if len(x) < 50 or len({len(x), len(y), len(vendor_spread)}) != 1:
        raise ValueError("OU v3 holdout series are invalid")
    if bool(stats.get("log_used", False)):
        if np.any(x <= 0.0) or np.any(y <= 0.0):
            raise ValueError("OU v3 log transform requires positive prices")
        x, y = np.log(x), np.log(y)

    beta = _ou_trend_aware_profile_beta_v3_candidate(
        x, y, inc_trend=local_inc_trend
    )
    spread = _ou_trend_aware_profile_spread_v3_candidate(
        x, y, inc_trend=local_inc_trend
    )
    spread_std = float(spread.std(ddof=1))
    zscore = (spread - float(spread.mean())) / spread_std
    series = pd.Series(spread)
    rolling_mean = series.rolling(window, min_periods=window).mean()
    rolling_std = series.rolling(window, min_periods=window).std(ddof=1)
    zscore_roll = (
        series.sub(rolling_mean)
        .div(rolling_std.where(rolling_std.abs() > 1e-12))
        .fillna(0.0)
        .to_numpy()
    )
    lagged = spread[:-1]
    current = spread[1:]
    _, phi = np.linalg.lstsq(
        np.column_stack([np.ones(len(lagged)), lagged]), current, rcond=None
    )[0]
    half_life = float(-np.log(2.0) / np.log(phi)) if 0.0 < phi < 1.0 else np.nan
    errors = {
        "hedge_ratio_abs_error": abs(beta - vendor_beta),
        "spread_max_abs_error": float(np.max(np.abs(spread - vendor_spread))),
        "zscore_max_abs_error": float(np.max(np.abs(zscore - vendor_zscore))),
        "zscore_roll_max_abs_error": float(
            np.max(np.abs(zscore_roll - vendor_zscore_roll))
        ),
        "half_life_abs_error": abs(half_life - vendor_half_life),
    }
    limits = {
        "hedge_ratio_abs_error": float(tolerances["hedge_ratio_abs"]),
        "spread_max_abs_error": float(tolerances["spread_max_abs"]),
        "zscore_max_abs_error": float(tolerances["zscore_max_abs"]),
        "zscore_roll_max_abs_error": float(tolerances["zscore_roll_max_abs"]),
        "half_life_abs_error": float(tolerances["half_life_abs"]),
    }
    formula_passed = all(
        np.isfinite(errors[key]) and errors[key] <= limits[key] for key in limits
    )
    selector_passed = local_inc_trend == vendor_inc_trend
    passed = formula_passed and selector_passed
    return {
        "passed": passed,
        "local_predicted_inc_trend": local_inc_trend,
        "vendor_inc_trend": vendor_inc_trend,
        "trend_selector_parity_passed": selector_passed,
        "trend_selector_parity_status": "matched" if selector_passed else "mismatch",
        "formula_parity_passed": formula_passed,
        "local_hedge_ratio": beta,
        "vendor_hedge_ratio": vendor_beta,
        "local_half_life": half_life,
        "vendor_half_life": vendor_half_life,
        **errors,
        "formula_parity_status": (
            "validated_numerical_reconstruction" if formula_passed else "mismatch"
        ),
        "formula_pit_status": "registered_local_aic_branch_full_window_research_only",
    }


def _request_from_payload(
    payload: dict[str, Any],
) -> CryptoWizardsCustomSeriesBacktestRequest:
    params = payload.get("params") if isinstance(payload.get("params"), dict) else {}
    inputs = payload.get("bt_inputs") if isinstance(payload.get("bt_inputs"), dict) else {}
    try:
        return CryptoWizardsCustomSeriesBacktestRequest(
            series_1_opens=tuple(float(value) for value in params["series_1_opens"]),
            series_1_closes=tuple(float(value) for value in params["series_1_closes"]),
            series_2_opens=tuple(float(value) for value in params["series_2_opens"]),
            series_2_closes=tuple(float(value) for value in params["series_2_closes"]),
            strategy=_text(params["strategy"]),
            spread_type=_text(params.get("spread_type")) or None,
            roll_w=int(params["roll_w"]),
            entry_level=float(inputs["entry_level"]),
            exit_level=float(inputs["exit_level"]),
            x_weighting=float(inputs["x_weighting"]),
            slippage_rate=float(inputs["slippage_rate"]),
            commission_rate=float(inputs["commission_rate"]),
            stop_loss_rate=(
                float(inputs["stop_loss_rate"])
                if "stop_loss_rate" in inputs
                else None
            ),
            exit_n_periods=(
                int(inputs["exit_n_periods"])
                if "exit_n_periods" in inputs
                else None
            ),
            with_history=bool(params.get("with_history", True)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("OU v3 registered request payload is invalid") from exc


def _select_cells(proofs: pd.DataFrame, pair_group_id: str) -> pd.DataFrame:
    if proofs.empty:
        return proofs.copy()
    return proofs.loc[
        proofs.get("pair_group_id", pd.Series("", index=proofs.index))
        .map(_text)
        .eq(pair_group_id)
        & proofs.get("exact_mode", pd.Series("", index=proofs.index)).isin(OU_MODES)
        & proofs.get("orientation", pd.Series("", index=proofs.index)).isin(ORIENTATIONS)
    ].copy()


def _require_exact_cells(
    frame: pd.DataFrame, expected: set[tuple[str, str]], *, label: str
) -> None:
    identities = [
        (_text(row.get("exact_mode")), _text(row.get("orientation")))
        for _, row in frame.iterrows()
    ]
    if len(identities) != len(set(identities)):
        raise ValueError(f"duplicate OU v2 {label} identities")
    if set(identities) != expected:
        raise ValueError(f"OU v2 {label} must contain the canonical four cells")
    if not frame.get("vendor_response_captured", pd.Series(False, index=frame.index)).map(_truthy).all():
        raise ValueError(f"OU v2 {label} responses must already be captured")


def _cell_bindings(
    root: Path, frame: pd.DataFrame, *, hash_files: bool
) -> list[dict[str, object]]:
    bindings: list[dict[str, object]] = []
    for _, row in frame.sort_values(["exact_mode", "orientation"]).iterrows():
        request = _resolve(root, row.get("request_path"))
        response = _resolve(root, row.get("response_path"))
        if request is None or response is None:
            raise ValueError("OU v2 binding paths are missing")
        bindings.append(
            {
                "exact_mode": _text(row.get("exact_mode")),
                "orientation": _text(row.get("orientation")),
                "proof_observations": int(float(row.get("proof_observations", 0))),
                "request_path": _relative(request, root),
                "response_path": _relative(response, root),
                "request_sha256": _file_hash(request) if hash_files else "",
                "response_sha256": _file_hash(response) if hash_files else "",
            }
        )
    return bindings


def _validate_registration(
    root: Path,
    contract_path: Path,
    contract: dict[str, Any],
    receipt: dict[str, Any],
) -> None:
    if _text(contract.get("schema_version")) != SCHEMA_VERSION:
        raise ValueError("OU v2 contract schema mismatch")
    if _text(receipt.get("contract_path")) != _relative(contract_path, root):
        raise ValueError("OU v2 receipt contract path mismatch")
    if _text(receipt.get("contract_sha256")) != _file_hash(contract_path):
        raise ValueError("OU v2 contract hash mismatch")
    if _text(contract.get("implementation", {}).get("source_sha256")) != _implementation_source_hash():
        raise ValueError("OU v2 implementation source hash mismatch")
    if not _truthy(receipt.get("registered_before_holdout_raw_series_inspection")):
        raise ValueError("OU v2 holdout was not registered while blinded")
    forbidden = (
        "candidate_promotion_authority",
        "testnet_order_authority",
        "live_trading_authorized",
    )
    if any(_truthy(contract.get(key)) or _truthy(receipt.get(key)) for key in forbidden):
        raise ValueError("OU v2 registration cannot grant trading authority")


def _validate_v3_registration(
    root: Path,
    contract_path: Path,
    contract: dict[str, Any],
    receipt: dict[str, Any],
) -> None:
    if _text(contract.get("schema_version")) != "thewiz.wizard_ou_v3_prospective_holdout.v1":
        raise ValueError("OU v3 contract schema mismatch")
    if _text(receipt.get("contract_path")) != _relative(contract_path, root):
        raise ValueError("OU v3 receipt contract path mismatch")
    if _text(receipt.get("contract_sha256")) != _file_hash(contract_path):
        raise ValueError("OU v3 contract hash mismatch")
    if _text(contract.get("implementation", {}).get("source_sha256")) != _v3_implementation_source_hash():
        raise ValueError("OU v3 implementation source hash mismatch")
    if int(contract.get("vendor_responses_at_registration", -1)) != 0:
        raise ValueError("OU v3 was not prospectively registered")
    for evidence in contract.get("source_candles", []):
        path = root / _text(evidence.get("path"))
        if not path.is_file() or _file_hash(path) != _text(evidence.get("sha256")):
            raise ValueError("OU v3 source candle binding mismatch")
    for evidence in contract.get("derivation_evidence", []):
        path = root / _text(evidence.get("path"))
        if not path.is_file() or _file_hash(path) != _text(evidence.get("sha256")):
            raise ValueError("OU v3 derivation evidence binding mismatch")
    for binding in contract.get("holdout_bindings", []):
        request_path = root / _text(binding.get("request_path"))
        if not request_path.is_file() or _file_hash(request_path) != _text(
            binding.get("request_sha256")
        ):
            raise ValueError("OU v3 holdout request binding mismatch")
    forbidden = (
        "candidate_promotion_authority",
        "testnet_order_authority",
        "live_trading_authorized",
    )
    if any(_truthy(contract.get(key)) or _truthy(receipt.get(key)) for key in forbidden):
        raise ValueError("OU v3 registration cannot grant trading authority")


def _validate_ou_trend_selector_registration(
    root: Path,
    contract_path: Path,
    contract: dict[str, Any],
    receipt: dict[str, Any],
) -> None:
    if _text(contract.get("schema_version")) != OU_TREND_SELECTOR_SCHEMA_VERSION:
        raise ValueError("OU trend selector contract schema mismatch")
    if (
        _text(receipt.get("schema_version"))
        != "thewiz.wizard_ou_trend_selector_receipt.v1"
    ):
        raise ValueError("OU trend selector receipt schema mismatch")
    if _text(receipt.get("contract_path")) != _relative(contract_path, root):
        raise ValueError("OU trend selector receipt contract path mismatch")
    if _text(receipt.get("contract_sha256")) != _file_hash(contract_path):
        raise ValueError("OU trend selector contract hash mismatch")
    source_hash = _ou_trend_selector_source_hash()
    if _text(contract.get("selector", {}).get("source_sha256")) != source_hash:
        raise ValueError("OU trend selector implementation hash mismatch")
    if _text(receipt.get("selector_source_sha256")) != source_hash:
        raise ValueError("OU trend selector receipt source hash mismatch")
    for source in contract.get("source_contracts", []):
        path = root / _text(source.get("path"))
        if not path.is_file() or _file_hash(path) != _text(source.get("sha256")):
            raise ValueError("OU trend selector source contract hash mismatch")
    for key in ("derivation", "holdout_predictions"):
        evidence = contract.get(key, {})
        path = root / _text(evidence.get("path"))
        if not path.is_file() or _file_hash(path) != _text(evidence.get("sha256")):
            raise ValueError(f"OU trend selector {key} evidence hash mismatch")
    predictions = _read_csv(root / OU_TREND_SELECTOR_PREDICTIONS_PATH)
    identities = predictions[["exact_mode", "orientation", "request_sha256"]]
    if len(predictions) != 4 or len(identities.drop_duplicates()) != 4:
        raise ValueError("OU trend selector must bind four unique holdout predictions")
    if predictions["vendor_response_present_at_registration"].map(_truthy).any():
        raise ValueError("OU trend selector predictions were not prospectively frozen")
    if int(receipt.get("vendor_responses_at_registration", -1)) != 0:
        raise ValueError("OU trend selector receipt is not prospective")
    forbidden = (
        "candidate_promotion_authority",
        "testnet_order_authority",
        "live_trading_authorized",
    )
    if any(_truthy(contract.get(key)) or _truthy(receipt.get(key)) for key in forbidden):
        raise ValueError("OU trend selector cannot grant trading authority")


def _implementation_source_hash() -> str:
    source = "\n\n".join(
        (
            inspect.getsource(_ou_zero_mean_profile_beta_v2_holdout),
            inspect.getsource(_ou_zero_mean_profile_spread_candidate_v2_holdout),
        )
    )
    return sha256(source.encode("utf-8")).hexdigest()


def _v3_implementation_source_hash() -> str:
    source = "\n\n".join(
        (
            inspect.getsource(_ou_zero_mean_profile_beta_v2_holdout),
            inspect.getsource(_ou_trend_aware_profile_beta_v3_candidate),
            inspect.getsource(_ou_trend_aware_profile_spread_v3_candidate),
        )
    )
    return sha256(source.encode("utf-8")).hexdigest()


def _ou_trend_selector_source_hash() -> str:
    source = "\n\n".join(
        (
            inspect.getsource(_ou_zero_mean_profile_beta_v2_holdout),
            inspect.getsource(_ou_trend_aware_profile_beta_v3_candidate),
            inspect.getsource(_ou_trend_aware_profile_spread_v3_candidate),
            inspect.getsource(_ou_local_trend_selector_v1),
            inspect.getsource(_ou_local_trend_selector_v1_from_request),
        )
    )
    return sha256(source.encode("utf-8")).hexdigest()


def _aligned_candles(
    asset_x_path: Path,
    asset_y_path: Path,
    *,
    observations: int,
) -> dict[str, Any]:
    def indexed(path: Path) -> dict[str, dict[str, Any]]:
        payload = _read_json(path)
        candles = payload.get("candles")
        if not isinstance(candles, list):
            raise TypeError(f"OU v3 candle source is invalid: {path}")
        result: dict[str, dict[str, Any]] = {}
        for candle in candles:
            if not isinstance(candle, dict):
                continue
            timestamp = _text(candle.get("startedAt"))
            try:
                opened = float(candle["open"])
                closed = float(candle["close"])
            except (KeyError, TypeError, ValueError):
                continue
            if timestamp and np.isfinite(opened) and np.isfinite(closed):
                result[timestamp] = {"open": opened, "close": closed}
        return result

    x = indexed(asset_x_path)
    y = indexed(asset_y_path)
    timestamps = sorted(set(x).intersection(y))
    if len(timestamps) < observations:
        raise ValueError(
            f"OU v3 aligned history has {len(timestamps)} rows, requires {observations}"
        )
    timestamps = timestamps[-observations:]
    return {
        "timestamps": timestamps,
        "x": {
            "opens": [x[timestamp]["open"] for timestamp in timestamps],
            "closes": [x[timestamp]["close"] for timestamp in timestamps],
        },
        "y": {
            "opens": [y[timestamp]["open"] for timestamp in timestamps],
            "closes": [y[timestamp]["close"] for timestamp in timestamps],
        },
    }


def _as_utc(now: datetime | None) -> datetime:
    value = now or datetime.now(UTC)
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _write_or_validate_immutable_json(payload: dict[str, Any], path: Path) -> None:
    canonical = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != canonical:
            raise ValueError(f"immutable OU v2 artifact conflict: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(canonical)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_text(encoding="utf-8") != canonical:
                raise ValueError(f"immutable OU v2 artifact conflict: {path}")
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _write_or_validate_immutable_csv(frame: pd.DataFrame, path: Path) -> None:
    canonical = frame.to_csv(index=False, lineterminator="\n")
    if path.exists():
        if path.read_text(encoding="utf-8") != canonical:
            raise ValueError(f"immutable OU selector artifact conflict: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical, encoding="utf-8")


def _file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _resolve(root: Path, value: object) -> Path | None:
    text = _text(value)
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = root / path
    return path if path.is_file() else None


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _text(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return _text(value).lower() in {"1", "true", "yes", "y"}
