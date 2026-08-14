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
from scipy.optimize import minimize_scalar
from statsmodels.tsa.stattools import adfuller

from quant_platform.active_pipeline import CommandResult
from quant_platform.api_extraction import CryptoWizardsFetchError
from quant_platform.crypto_wizards_history import (
    CryptoWizardsCustomSeriesBacktestRequest,
    fetch_credits_used,
    fetch_custom_series_backtest,
)
from quant_platform.crypto_wizards_sweep import parse_wizard_credit_usage
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
from quant_platform.wizard_hyperliquid_mode_proof import (
    _ou_trend_aware_profile_beta_v3_candidate,
    _ou_trend_aware_profile_spread_v3_candidate,
)

ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "thewiz.wizard_ou_v4_prospective_holdout.v1"
RECEIPT_SCHEMA_VERSION = "thewiz.wizard_ou_v4_prospective_holdout_receipt.v1"
OU_MODES = ("OU (Spread)", "OU (ZScoreR)")
ORIENTATIONS = ("original", "reverse")
EXPECTED_SELECTOR_CELLS = {
    "level_constant",
    "level_trend",
    "log_constant",
    "log_trend",
}
DERIVATION_ASSETS = {"ADA", "ALGO", "HBAR", "XLM", "BTC", "ETH"}
DEFAULT_HISTORY_BASE = (
    "reports/snapshots/current_wizard_hyperliquid/"
    "ewapi_69a86274da721a941e36/cwhandoff_61dc9e76532abd99c502/"
    "history_runs/cwhistoryrun_20260812T115658834898Z_1d7a24ca/assets"
)
DEFAULT_PAIR_SPECS: tuple[dict[str, str], ...] = (
    {
        "pair": "MORPHO-AAVE",
        "asset_x": "MORPHO",
        "asset_y": "AAVE",
        "interval": "1d",
        "asset_x_path": f"{DEFAULT_HISTORY_BASE}/MORPHO_1d_candles.json",
        "asset_y_path": f"{DEFAULT_HISTORY_BASE}/AAVE_1d_candles.json",
    },
    {
        "pair": "AVAX-NEAR",
        "asset_x": "AVAX",
        "asset_y": "NEAR",
        "interval": "1h",
        "asset_x_path": f"{DEFAULT_HISTORY_BASE}/AVAX_1h_candles.json",
        "asset_y_path": f"{DEFAULT_HISTORY_BASE}/NEAR_1h_candles.json",
    },
)
DEFAULT_TOLERANCES = {
    "hedge_ratio_abs": 1e-6,
    "spread_max_abs": 1e-6,
    "zscore_max_abs": 1e-6,
    "zscore_roll_max_abs": 1e-6,
    "half_life_abs": 1e-3,
}


# Frozen OU v4 implementations are source-hash evidence.
# fmt: off
def _ou_stationary_constrained_profile_fit_v4(
    x: np.ndarray,
    y: np.ndarray,
) -> dict[str, float | bool]:
    """Fit the zero-mean OU branch with persistence constrained to [0, 1]."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or len(x) != len(y):
        raise ValueError("OU v4 requires equal series with at least three rows")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("OU v4 requires finite series")
    if abs(float(x[0])) <= 1e-18 or abs(float(y[0])) <= 1e-18:
        raise ValueError("OU v4 cannot normalize a zero initial value")

    normalized_x = x / float(x[0])
    normalized_y = y / float(y[0])
    static_beta, _ = np.polyfit(normalized_x, normalized_y, 1)
    width = max(abs(float(static_beta)) * 4.0, 2.0)
    center = float(static_beta)

    def objective(beta: float) -> float:
        spread = normalized_y - beta * normalized_x
        lagged = spread[:-1]
        current = spread[1:]
        denominator = float(np.dot(lagged, lagged))
        if not np.isfinite(denominator) or denominator <= 1e-18:
            return float("inf")
        unconstrained_phi = float(np.dot(lagged, current) / denominator)
        phi = float(np.clip(unconstrained_phi, 0.0, 1.0))
        residual = current - phi * lagged
        value = float(np.mean(np.square(residual)))
        return value if np.isfinite(value) else float("inf")

    for _ in range(6):
        grid = np.linspace(center - width, center + width, 513)
        values = np.asarray([objective(float(beta)) for beta in grid])
        best = int(np.argmin(values))
        if 0 < best < len(grid) - 1:
            break
        center = float(grid[best])
        width *= 2.0
    else:
        raise ValueError("OU v4 hedge-ratio search found no interior beta")

    result = minimize_scalar(
        objective,
        bounds=(float(grid[best - 1]), float(grid[best + 1])),
        method="bounded",
        options={"xatol": 1e-14, "maxiter": 1000},
    )
    if not result.success or not np.isfinite(result.fun):
        raise ValueError("OU v4 hedge-ratio refinement failed")
    beta = float(result.x)
    spread = normalized_y - beta * normalized_x
    denominator = float(np.dot(spread[:-1], spread[:-1]))
    unconstrained_phi = float(np.dot(spread[:-1], spread[1:]) / denominator)
    phi = float(np.clip(unconstrained_phi, 0.0, 1.0))
    return {
        "beta": beta,
        "profile_phi": phi,
        "unconstrained_profile_phi": unconstrained_phi,
        "unit_root_boundary_active": unconstrained_phi >= 1.0,
        "objective": float(result.fun),
    }


def _ou_stationary_constrained_profile_beta_v4(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    return float(_ou_stationary_constrained_profile_fit_v4(x, y)["beta"])


def _ou_spread_candidate_v4(
    x: np.ndarray,
    y: np.ndarray,
    *,
    inc_trend: bool,
) -> tuple[float, np.ndarray, dict[str, object]]:
    """Return the v4 spread while preserving the validated v3 constant branch."""

    if inc_trend:
        fit = _ou_stationary_constrained_profile_fit_v4(x, y)
        beta = float(fit["beta"])
        spread = y / float(y[0]) - beta * x / float(x[0])
        return beta, spread, fit
    beta = _ou_trend_aware_profile_beta_v3_candidate(x, y, inc_trend=False)
    spread = _ou_trend_aware_profile_spread_v3_candidate(
        x, y, inc_trend=False
    )
    return beta, spread, {
        "beta": beta,
        "profile_phi": None,
        "unconstrained_profile_phi": None,
        "unit_root_boundary_active": False,
        "objective": None,
    }


def _ou_local_transform_trend_selector_v2(
    x: np.ndarray,
    y: np.ndarray,
) -> dict[str, object]:
    """Select level/log and constant/trend jointly from request-only data."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 50 or len(x) != len(y):
        raise ValueError("OU v4 selector requires equal series of at least 50 rows")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("OU v4 selector requires finite series")
    if np.any(x <= 0.0) or np.any(y <= 0.0):
        raise ValueError("OU v4 selector requires positive prices for log candidate")

    candidates: list[dict[str, object]] = []
    priority = 0
    for transform_name, transformed_x, transformed_y in (
        ("level", x, y),
        ("log", np.log(x), np.log(y)),
    ):
        for inc_trend, regression in ((False, "c"), (True, "ct")):
            beta, spread, fit = _ou_spread_candidate_v4(
                transformed_x,
                transformed_y,
                inc_trend=inc_trend,
            )
            result = adfuller(
                spread,
                maxlag=2,
                regression=regression,
                autolag=None,
                store=True,
                regresults=True,
            )
            aic = float(result[-1].resols.aic)
            if not np.isfinite(aic):
                raise ValueError("OU v4 selector produced a non-finite AIC")
            branch_name = "trend" if inc_trend else "constant"
            candidates.append(
                {
                    "selector_cell": f"{transform_name}_{branch_name}",
                    "transform": transform_name,
                    "log_used": transform_name == "log",
                    "inc_trend": inc_trend,
                    "regression": regression,
                    "aic": aic,
                    "adf_t": float(result[0]),
                    "adf_p": float(result[1]),
                    "beta": beta,
                    "profile_phi": fit["profile_phi"],
                    "unconstrained_profile_phi": fit[
                        "unconstrained_profile_phi"
                    ],
                    "unit_root_boundary_active": fit[
                        "unit_root_boundary_active"
                    ],
                    "priority": priority,
                }
            )
            priority += 1
    ranked = sorted(candidates, key=lambda item: (float(item["aic"]), int(item["priority"])))
    selected = ranked[0]
    output: dict[str, object] = {
        "local_predicted_log_used": bool(selected["log_used"]),
        "local_predicted_inc_trend": bool(selected["inc_trend"]),
        "selected_transform": _text(selected["transform"]),
        "selected_branch": "trend" if selected["inc_trend"] else "constant",
        "selected_selector_cell": _text(selected["selector_cell"]),
        "selected_aic": float(selected["aic"]),
        "runner_up_aic": float(ranked[1]["aic"]),
        "aic_margin": float(ranked[1]["aic"]) - float(selected["aic"]),
        "selected_beta": float(selected["beta"]),
        "selected_profile_phi": selected["profile_phi"],
        "selected_unconstrained_profile_phi": selected[
            "unconstrained_profile_phi"
        ],
        "selected_unit_root_boundary_active": bool(
            selected["unit_root_boundary_active"]
        ),
        "adf_maxlag": 2,
        "adf_autolag": "none",
        "selector_rule": "minimum_aic_across_level_log_x_constant_trend",
    }
    for candidate in candidates:
        prefix = _text(candidate["selector_cell"])
        output[f"{prefix}_aic"] = float(candidate["aic"])
        output[f"{prefix}_adf_t"] = float(candidate["adf_t"])
        output[f"{prefix}_adf_p"] = float(candidate["adf_p"])
    return output
# fmt: on


def register_ou_v4_prospective_holdout(
    *,
    root: Path = ROOT,
    now: datetime | None = None,
    pair_specs: Sequence[Mapping[str, str]] | None = None,
    observations: int = 360,
    require_full_selector_matrix: bool = True,
) -> CommandResult:
    """Freeze a disjoint eight-cell OU-v4 holdout without vendor calls."""

    if observations < 50 or observations > 1100:
        raise ValueError("OU v4 prospective holdout requires 50 to 1100 observations")
    contract_path = root / "config" / "wizard_ou_comparator_v4_holdout.json"
    receipt_path = root / "reports" / "active" / "wizard_ou_v4_holdout_receipt.json"
    if contract_path.exists() or receipt_path.exists():
        if not contract_path.is_file() or not receipt_path.is_file():
            raise ValueError("OU v4 registration is incomplete")
        contract = _read_json(contract_path)
        receipt = _read_json(receipt_path)
        _validate_registration(root, contract_path, contract, receipt)
        return CommandResult(
            paths={
                "contract": contract_path,
                "receipt": receipt_path,
                "derivation": root / _text(contract["derivation_evidence"][0]["path"]),
                "predictions": root / _text(contract["holdout_predictions"]["path"]),
            },
            summary=receipt,
        )

    specs = [dict(item) for item in (pair_specs or DEFAULT_PAIR_SPECS)]
    if len(specs) != 2:
        raise ValueError("OU v4 requires exactly two disjoint holdout pairs")
    assets = [
        _text(spec.get(key)).upper()
        for spec in specs
        for key in ("asset_x", "asset_y")
    ]
    if any(not asset for asset in assets) or len(set(assets)) != 4:
        raise ValueError("OU v4 holdout pairs must contain four distinct assets")
    if set(assets) & DERIVATION_ASSETS:
        raise ValueError("OU v4 holdout assets overlap the OU derivation assets")

    derivation = _build_derivation_frame(root)
    if len(derivation) != 12 or not derivation["cell_passed"].map(_truthy).all():
        raise ValueError("OU v4 must reconstruct all twelve open derivation cells")
    derivation_path = root / "reports" / "active" / "wizard_ou_v4_derivation.csv"
    _write_or_validate_immutable_csv(derivation, derivation_path)

    bindings: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    source_candles: list[dict[str, object]] = []
    request_dir = root / "data" / "research" / "wizard_ou_v4_holdout" / "requests"
    response_dir = root / "data" / "raw" / "wizard_ou_v4_holdout" / "responses"
    for spec in specs:
        pair = _text(spec.get("pair")).upper()
        asset_x = _text(spec.get("asset_x")).upper()
        asset_y = _text(spec.get("asset_y")).upper()
        interval = _text(spec.get("interval")).lower()
        if pair != f"{asset_x}-{asset_y}" or not interval:
            raise ValueError("OU v4 pair specification identity is invalid")
        x_path = _resolve_source_path(root, spec.get("asset_x_path"))
        y_path = _resolve_source_path(root, spec.get("asset_y_path"))
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
                else:
                    first, second = aligned["y"], aligned["x"]
                    oriented_pair = f"{asset_y}-{asset_x}"
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
                slug = (
                    f"{oriented_pair.lower().replace('-', '_')}_{interval}_"
                    f"{strategy.lower()}"
                )
                request_path = request_dir / f"{slug}_request.json"
                response_path = response_dir / f"{slug}_response.json"
                if response_path.exists():
                    raise ValueError("OU v4 vendor response existed before registration")
                _write_or_validate_immutable_json(request.payload(), request_path)
                selector = _selector_from_request(_read_json(request_path))
                binding = {
                    "pair_group": pair,
                    "pair": oriented_pair,
                    "asset_x": asset_x if orientation == "original" else asset_y,
                    "asset_y": asset_y if orientation == "original" else asset_x,
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
                prediction_rows.append(
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
                        **selector,
                        "vendor_response_present_at_registration": False,
                    }
                )
    if len(bindings) != 8:
        raise ValueError("OU v4 registration must bind eight mode-orientation cells")
    predictions = pd.DataFrame(prediction_rows)
    orientation_predictions = predictions.loc[
        predictions["exact_mode"].eq("OU (Spread)")
    ]
    selector_cells = set(
        orientation_predictions["selected_selector_cell"].map(_text)
    )
    if require_full_selector_matrix and selector_cells != EXPECTED_SELECTOR_CELLS:
        raise ValueError(
            "OU v4 default holdout must cover all level/log x constant/trend cells"
        )
    predictions_path = root / "reports" / "active" / "wizard_ou_v4_predictions.csv"
    _write_or_validate_immutable_csv(predictions, predictions_path)

    source_hash = _implementation_source_hash()
    v3_status = _read_json(
        root / "reports" / "active" / "wizard_ou_v3_holdout_status.json"
    )
    v3_immutable_result_path = _resolve_source_path(
        root, v3_status.get("immutable_result_path")
    )
    source_contract_paths = (
        root / "config" / "wizard_ou_comparator_v2_holdout.json",
        root / "config" / "wizard_ou_comparator_v3_holdout.json",
        v3_immutable_result_path,
        root / "reports" / "active" / "wizard_ou_v3_holdout_evaluation.csv",
    )
    contract = {
        "schema_version": SCHEMA_VERSION,
        "status": "PREREGISTERED_WAITING_VENDOR_RESPONSES",
        "hypothesis": {
            "transform_rule": "minimum ADF regression AIC across level and log inputs",
            "branch_rule": "minimum ADF regression AIC across constant and trend",
            "trend_formula": (
                "zero-mean AR1 residual profile with persistence constrained to [0,1]"
            ),
            "constant_formula": "validated OU-v3 intercept AR1 residual profile",
            "unit_root_boundary_interpretation": (
                "phi=1 reduces the trend-branch objective to no-intercept "
                "first-difference regression"
            ),
            "holdout_pair_groups": sorted({_text(item["pair_group"]) for item in bindings}),
            "holdout_is_pair_and_asset_disjoint": True,
            "selector_matrix_required": require_full_selector_matrix,
            "selector_matrix_predicted": sorted(selector_cells),
        },
        "implementation": {
            "generation": 4,
            "source_sha256": source_hash,
            "point_in_time_inputs": "registered request close series only",
            "vendor_response_used_for_selection": False,
            "automatic_activation": False,
        },
        "tolerances": DEFAULT_TOLERANCES,
        "derivation_summary": {
            "cells": len(derivation),
            "independent_series_orientations": 6,
            "passed_cells": int(derivation["cell_passed"].map(_truthy).sum()),
        },
        "derivation_evidence": [
            {
                "path": _relative(derivation_path, root),
                "sha256": _file_hash(derivation_path),
            }
        ],
        "source_contracts": [
            {"path": _relative(path, root), "sha256": _file_hash(path)}
            for path in source_contract_paths
        ],
        "source_candles": source_candles,
        "holdout_predictions": {
            "path": _relative(predictions_path, root),
            "sha256": _file_hash(predictions_path),
            "cells": len(predictions),
            "independent_series_orientations": len(orientation_predictions),
            "vendor_responses_at_registration": 0,
        },
        "holdout_bindings": bindings,
        "vendor_responses_at_registration": 0,
        "threshold_changes_after_vendor_response_allowed": False,
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
        "status": "REGISTERED_WAITING_VENDOR_RESPONSES",
        "contract_path": _relative(contract_path, root),
        "contract_sha256": _file_hash(contract_path),
        "implementation_source_sha256": source_hash,
        "holdout_pair_groups": contract["hypothesis"]["holdout_pair_groups"],
        "required_cells": 8,
        "derivation_cells_passed": 12,
        "selector_matrix_predicted": sorted(selector_cells),
        "vendor_responses_at_registration": 0,
        "prospectively_registered": True,
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


def run_ou_v4_prospective_holdout(
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
    """Capture the preregistered OU-v4 holdout under bounded credit controls."""

    timestamp = _as_utc(now)
    contract_path = root / "config" / "wizard_ou_comparator_v4_holdout.json"
    receipt_path = root / "reports" / "active" / "wizard_ou_v4_holdout_receipt.json"
    contract = _read_json(contract_path)
    receipt = _read_json(receipt_path)
    _validate_registration(root, contract_path, contract, receipt)
    bindings = contract["holdout_bindings"]
    missing = [
        item
        for item in bindings
        if not (root / _text(item.get("response_path"))).is_file()
    ]
    key = api_key or os.getenv("CRYPTO_WIZARDS_API_KEY", "").strip()
    credits_used: int | None = None
    blocker = ""
    ambiguous_intent_blockers = (
        _ambiguous_call_intent_blockers(
            root=root,
            timestamp=timestamp,
            contract_path=contract_path,
            missing=missing,
        )
        if execute and missing
        else []
    )
    if ambiguous_intent_blockers:
        blocker = ambiguous_intent_blockers[0]
    if execute and missing and not blocker:
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
        / "wizard_ou_v4_holdout"
        / "attempts"
        / f"{timestamp.date().isoformat()}.json"
    )
    attempt_already_exists = execute and bool(missing) and attempt_path.exists()
    if attempt_already_exists:
        blocker = "ou_v4_daily_attempt_already_registered"
    calls_made = 0
    responses_captured = 0
    errors: list[str] = []
    call_attempt_intent_paths: list[str] = []
    if execute and missing and not blocker:
        for binding in missing:
            request_path = root / _text(binding.get("request_path"))
            if _file_hash(request_path) != _text(binding.get("request_sha256")):
                errors.append("ou_v4_request_hash_mismatch")
                break
            call_id = _ou_v4_call_id(binding)
            intent_path = _call_intent_path(
                root=root,
                timestamp=timestamp,
                call_id=call_id,
            )
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
                errors.append(f"call_intent_persistence_failed:{type(exc).__name__}:{exc}")
                break
            call_attempt_intent_paths.append(_relative(intent_path, root))
            calls_made += 1
            try:
                response = fetcher(
                    _request_from_payload(_read_json(request_path)), api_key=key
                )
                response_path = root / _text(binding.get("response_path"))
                _write_or_validate_immutable_json(response, response_path)
                responses_captured += 1
            except (CryptoWizardsFetchError, OSError, TypeError, ValueError) as exc:
                errors.append(f"vendor_capture_failed:{type(exc).__name__}:{exc}")
                break
    attempt = {
        "schema_version": "thewiz.wizard_ou_v4_capture_attempt.v1",
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
        "call_attempt_intent_paths": call_attempt_intent_paths,
        "ambiguous_call_intent_blockers": ambiguous_intent_blockers,
        "contract_sha256": _file_hash(contract_path),
        "research_only": True,
        "candidate_promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    if execute and missing and not attempt_already_exists:
        _write_or_validate_immutable_json(attempt, attempt_path)
    responses_now = sum(
        (root / _text(item.get("response_path"))).is_file() for item in bindings
    )
    paths: dict[str, Path] = {}
    if responses_now == len(bindings):
        evaluation = evaluate_ou_v4_prospective_holdout(root=root, now=timestamp)
        evaluation_status = _text(evaluation.summary.get("status"))
        paths.update(evaluation.paths)
    else:
        evaluation_status = "WAITING_VENDOR_RESPONSES"
    status = {
        **attempt,
        "schema_version": "thewiz.wizard_ou_v4_capture_status.v1",
        "responses_available": responses_now,
        "required_responses": len(bindings),
        "evaluation_status": evaluation_status,
        "next_step": (
            "evaluate_registered_holdout"
            if responses_now == len(bindings)
            else "wait_for_next_eligible_credit_window"
        ),
    }
    status_path = root / "reports" / "active" / "wizard_ou_v4_capture_status.json"
    _atomic_json(status, status_path)
    paths["capture_status"] = status_path
    if attempt_path.is_file():
        paths["attempt"] = attempt_path
    return CommandResult(paths=paths, summary=status)


def _ou_v4_call_id(binding: Mapping[str, Any]) -> str:
    material = "|".join(
        (
            "ou_v4_holdout",
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
        / "data"
        / "research"
        / "wizard_ou_v4_holdout"
        / "call_attempts"
        / timestamp.date().isoformat()
        / f"{call_id}.json"
    )


def _call_intent_payload(
    *,
    root: Path,
    timestamp: datetime,
    contract_path: Path,
    binding: Mapping[str, Any],
    call_id: str,
) -> dict[str, Any]:
    return {
        "schema_version": "thewiz.wizard_ou_v4_call_attempt_intent.v1",
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


def _ambiguous_call_intent_blockers(
    *,
    root: Path,
    timestamp: datetime,
    contract_path: Path,
    missing: Sequence[Mapping[str, Any]],
) -> list[str]:
    blockers: list[str] = []
    for binding in missing:
        call_id = _ou_v4_call_id(binding)
        intent_path = _call_intent_path(
            root=root,
            timestamp=timestamp,
            call_id=call_id,
        )
        if not intent_path.is_file():
            continue
        intent = _read_json(intent_path)
        valid = bool(
            intent.get("schema_version") == "thewiz.wizard_ou_v4_call_attempt_intent.v1"
            and _text(intent.get("attempt_date_utc")) == timestamp.date().isoformat()
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
                "ou_v4_call_attempt_already_registered_without_response:"
                if valid
                else "ou_v4_call_attempt_intent_invalid:"
            )
            + call_id
        )
    return blockers


def evaluate_ou_v4_prospective_holdout(
    *,
    root: Path = ROOT,
    now: datetime | None = None,
) -> CommandResult:
    """Evaluate all frozen OU-v4 cells with independent failure attribution."""

    contract_path = root / "config" / "wizard_ou_comparator_v4_holdout.json"
    receipt_path = root / "reports" / "active" / "wizard_ou_v4_holdout_receipt.json"
    contract = _read_json(contract_path)
    receipt = _read_json(receipt_path)
    _validate_registration(root, contract_path, contract, receipt)
    predictions_path = root / _text(contract["holdout_predictions"]["path"])
    predictions = pd.read_csv(predictions_path)
    rows: list[dict[str, object]] = []
    for binding in contract["holdout_bindings"]:
        request_path = root / _text(binding.get("request_path"))
        response_path = root / _text(binding.get("response_path"))
        if not response_path.is_file():
            raise ValueError("OU v4 evaluation requires all vendor responses")
        prediction = _prediction_for_binding(predictions, binding)
        request = _read_json(request_path)
        selector = _selector_from_request(request)
        if _text(selector["selected_selector_cell"]) != _text(
            prediction.get("selected_selector_cell")
        ):
            raise ValueError("OU v4 frozen selector prediction mismatch")
        response = _read_json(response_path)
        vendor_log, vendor_trend = _vendor_selector(response)
        local_log = _truthy(prediction.get("local_predicted_log_used"))
        local_trend = _truthy(prediction.get("local_predicted_inc_trend"))
        metrics = _evaluate_formula(
            request=request,
            response=response,
            log_used=local_log,
            inc_trend=local_trend,
            tolerances=contract["tolerances"],
        )
        oracle_metrics = _evaluate_formula(
            request=request,
            response=response,
            log_used=vendor_log,
            inc_trend=vendor_trend,
            tolerances=contract["tolerances"],
        )
        transform_passed = local_log == vendor_log
        trend_passed = local_trend == vendor_trend
        formula_passed = _truthy(metrics["formula_parity_passed"])
        passed = transform_passed and trend_passed and formula_passed
        rows.append(
            {
                "pair_group": _text(binding.get("pair_group")),
                "pair": _text(binding.get("pair")),
                "interval": _text(binding.get("interval")),
                "exact_mode": _text(binding.get("exact_mode")),
                "orientation": _text(binding.get("orientation")),
                "cell_status": "PASS" if passed else "FAIL",
                "selected_selector_cell": _text(
                    prediction.get("selected_selector_cell")
                ),
                "selector_aic_margin": float(prediction.get("aic_margin", 0.0)),
                "local_predicted_log_used": local_log,
                "vendor_log_used": vendor_log,
                "transform_selector_parity_passed": transform_passed,
                "local_predicted_inc_trend": local_trend,
                "vendor_inc_trend": vendor_trend,
                "trend_selector_parity_passed": trend_passed,
                **metrics,
                "oracle_formula_parity_passed": oracle_metrics[
                    "formula_parity_passed"
                ],
                "oracle_hedge_ratio_abs_error": oracle_metrics[
                    "hedge_ratio_abs_error"
                ],
                "oracle_spread_max_abs_error": oracle_metrics[
                    "spread_max_abs_error"
                ],
                "request_path": _relative(request_path, root),
                "request_sha256": _file_hash(request_path),
                "response_path": _relative(response_path, root),
                "response_sha256": _file_hash(response_path),
                "blocker": "" if passed else _v4_blocker(
                    transform_passed, trend_passed, formula_passed
                ),
                "candidate_promotion_authority": False,
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            }
        )
    frame = pd.DataFrame(rows)
    required = len(contract["holdout_bindings"])
    passed_cells = int(frame["cell_status"].eq("PASS").sum())
    status_value = "PASS" if passed_cells == required else "FAIL"
    detail_path = root / "reports" / "active" / "wizard_ou_v4_holdout_evaluation.csv"
    status_path = root / "reports" / "active" / "wizard_ou_v4_holdout_status.json"
    _atomic_csv(frame, detail_path)
    payload = {
        "schema_version": "thewiz.wizard_ou_v4_holdout_evaluation.v1",
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
        "formula_parity_passed_cells": int(
            frame["formula_parity_passed"].map(_truthy).sum()
        ),
        "oracle_formula_parity_passed_cells": int(
            frame["oracle_formula_parity_passed"].map(_truthy).sum()
        ),
        "comparator_supersession_eligible": status_value == "PASS",
        "comparator_supersession_automatic": False,
        "implementation_source_sha256": _implementation_source_hash(),
        "response_sha256s": sorted(frame["response_sha256"].map(_text)),
        "evidence_path": _relative(detail_path, root),
        "research_only": True,
        "candidate_promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    immutable = {
        key: value
        for key, value in payload.items()
        if key not in {"evaluated_at_utc", "evidence_path"}
    }
    result_id = "ouv4holdout_" + sha256(
        json.dumps(immutable, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:24]
    immutable["result_id"] = result_id
    immutable_path = (
        root
        / "data"
        / "research"
        / "wizard_ou_v4_holdout_evaluations"
        / f"{result_id}.json"
    )
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
        paths={
            "detail": detail_path,
            "status": status_path,
            "immutable_result": immutable_path,
        },
        summary=payload,
    )


def _build_derivation_frame(root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    sources = (
        (
            "v2_derivation",
            root / "config" / "wizard_ou_comparator_v2_holdout.json",
            "derivation_bindings",
        ),
        (
            "v2_blind_holdout",
            root / "config" / "wizard_ou_comparator_v2_holdout.json",
            "sealed_holdout_bindings",
        ),
        (
            "v3_failed_holdout",
            root / "config" / "wizard_ou_comparator_v3_holdout.json",
            "holdout_bindings",
        ),
    )
    v3_status = _read_json(
        root / "reports" / "active" / "wizard_ou_v3_holdout_status.json"
    )
    v3_response_hashes = set(v3_status.get("response_sha256s", []))
    for cohort, contract_path, key in sources:
        contract = _read_json(contract_path)
        bindings = contract.get(key, [])
        if not isinstance(bindings, list) or len(bindings) != 4:
            raise ValueError(f"OU v4 derivation requires four {cohort} cells")
        for binding in bindings:
            request_path = root / _text(binding.get("request_path"))
            response_path = root / _text(binding.get("response_path"))
            if _file_hash(request_path) != _text(binding.get("request_sha256")):
                raise ValueError("OU v4 derivation request binding mismatch")
            response_hash = _file_hash(response_path)
            registered_response_hash = _text(binding.get("response_sha256"))
            if registered_response_hash and response_hash != registered_response_hash:
                raise ValueError("OU v4 derivation response binding mismatch")
            if cohort == "v3_failed_holdout" and response_hash not in v3_response_hashes:
                raise ValueError("OU v4 derivation lacks frozen v3 response provenance")
            request = _read_json(request_path)
            response = _read_json(response_path)
            selector = _selector_from_request(request)
            vendor_log, vendor_trend = _vendor_selector(response)
            metrics = _evaluate_formula(
                request=request,
                response=response,
                log_used=bool(selector["local_predicted_log_used"]),
                inc_trend=bool(selector["local_predicted_inc_trend"]),
                tolerances=DEFAULT_TOLERANCES,
            )
            transform_match = bool(selector["local_predicted_log_used"]) == vendor_log
            trend_match = bool(selector["local_predicted_inc_trend"]) == vendor_trend
            formula_match = _truthy(metrics["formula_parity_passed"])
            rows.append(
                {
                    "cohort": cohort,
                    "pair": _text(binding.get("pair")),
                    "exact_mode": _text(binding.get("exact_mode")),
                    "orientation": _text(binding.get("orientation")),
                    "proof_observations": int(binding.get("proof_observations", 0)),
                    "selected_selector_cell": selector["selected_selector_cell"],
                    "selector_aic_margin": selector["aic_margin"],
                    "local_predicted_log_used": selector[
                        "local_predicted_log_used"
                    ],
                    "vendor_log_used": vendor_log,
                    "transform_match": transform_match,
                    "local_predicted_inc_trend": selector[
                        "local_predicted_inc_trend"
                    ],
                    "vendor_inc_trend": vendor_trend,
                    "trend_match": trend_match,
                    **metrics,
                    "cell_passed": transform_match and trend_match and formula_match,
                    "request_path": _relative(request_path, root),
                    "request_sha256": _file_hash(request_path),
                    "response_path": _relative(response_path, root),
                    "response_sha256": response_hash,
                }
            )
    return pd.DataFrame(rows)


def _selector_from_request(request: dict[str, Any]) -> dict[str, object]:
    params = request.get("params") if isinstance(request.get("params"), dict) else {}
    try:
        x = np.asarray(params["series_1_closes"], dtype=float)
        y = np.asarray(params["series_2_closes"], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("OU v4 request lacks numeric close series") from exc
    return _ou_local_transform_trend_selector_v2(x, y)


def _vendor_selector(response: dict[str, Any]) -> tuple[bool, bool]:
    history = response.get("history") if isinstance(response.get("history"), dict) else {}
    stats = history.get("spread_stats") if isinstance(history.get("spread_stats"), dict) else {}
    coint = history.get("coint_eg") if isinstance(history.get("coint_eg"), dict) else {}
    if "log_used" not in stats or "inc_trend" not in coint:
        raise ValueError("OU v4 response lacks transform or trend provenance")
    return _truthy(stats.get("log_used")), _truthy(coint.get("inc_trend"))


def _evaluate_formula(
    *,
    request: dict[str, Any],
    response: dict[str, Any],
    log_used: bool,
    inc_trend: bool,
    tolerances: Mapping[str, Any],
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
        raise ValueError("OU v4 response lacks required numeric evidence") from exc
    if len(x) < 50 or len({len(x), len(y), len(vendor_spread)}) != 1:
        raise ValueError("OU v4 response series are invalid")
    if log_used:
        if np.any(x <= 0.0) or np.any(y <= 0.0):
            raise ValueError("OU v4 log transform requires positive prices")
        x, y = np.log(x), np.log(y)
    beta, spread, fit = _ou_spread_candidate_v4(x, y, inc_trend=inc_trend)
    zscore = (spread - float(spread.mean())) / float(spread.std(ddof=1))
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
    passed = all(
        np.isfinite(errors[key]) and errors[key] <= limits[key] for key in limits
    )
    return {
        "formula_parity_passed": passed,
        "formula_parity_status": (
            "validated_numerical_reconstruction" if passed else "mismatch"
        ),
        "local_hedge_ratio": beta,
        "vendor_hedge_ratio": vendor_beta,
        "local_half_life": half_life,
        "vendor_half_life": vendor_half_life,
        "profile_phi": fit["profile_phi"],
        "unconstrained_profile_phi": fit["unconstrained_profile_phi"],
        "unit_root_boundary_active": fit["unit_root_boundary_active"],
        **errors,
        "formula_pit_status": "registered_request_only_full_window_research_only",
    }


def _prediction_for_binding(
    predictions: pd.DataFrame,
    binding: Mapping[str, Any],
) -> dict[str, object]:
    mask = (
        predictions["exact_mode"].map(_text).eq(_text(binding.get("exact_mode")))
        & predictions["orientation"].map(_text).eq(_text(binding.get("orientation")))
        & predictions["request_sha256"].map(_text).eq(
            _text(binding.get("request_sha256"))
        )
    )
    matches = predictions.loc[mask]
    if len(matches) != 1:
        raise ValueError("OU v4 selector prediction identity mismatch")
    return matches.iloc[0].to_dict()


def _validate_registration(
    root: Path,
    contract_path: Path,
    contract: dict[str, Any],
    receipt: dict[str, Any],
) -> None:
    if _text(contract.get("schema_version")) != SCHEMA_VERSION:
        raise ValueError("OU v4 contract schema mismatch")
    if _text(receipt.get("schema_version")) != RECEIPT_SCHEMA_VERSION:
        raise ValueError("OU v4 receipt schema mismatch")
    if _text(receipt.get("contract_path")) != _relative(contract_path, root):
        raise ValueError("OU v4 receipt contract path mismatch")
    if _text(receipt.get("contract_sha256")) != _file_hash(contract_path):
        raise ValueError("OU v4 contract hash mismatch")
    source_hash = _implementation_source_hash()
    if _text(contract.get("implementation", {}).get("source_sha256")) != source_hash:
        raise ValueError("OU v4 implementation source hash mismatch")
    if _text(receipt.get("implementation_source_sha256")) != source_hash:
        raise ValueError("OU v4 receipt implementation hash mismatch")
    for key in ("source_contracts", "source_candles", "derivation_evidence"):
        for evidence in contract.get(key, []):
            path = _resolve_source_path(root, evidence.get("path"))
            if _file_hash(path) != _text(evidence.get("sha256")):
                raise ValueError(f"OU v4 {key} binding mismatch")
    predictions = contract.get("holdout_predictions", {})
    prediction_path = _resolve_source_path(root, predictions.get("path"))
    if _file_hash(prediction_path) != _text(predictions.get("sha256")):
        raise ValueError("OU v4 frozen predictions binding mismatch")
    bindings = contract.get("holdout_bindings", [])
    if not isinstance(bindings, list) or len(bindings) != 8:
        raise ValueError("OU v4 contract must bind eight holdout cells")
    identities: set[tuple[str, str, str]] = set()
    for binding in bindings:
        request_path = _resolve_source_path(root, binding.get("request_path"))
        if _file_hash(request_path) != _text(binding.get("request_sha256")):
            raise ValueError("OU v4 request binding mismatch")
        identities.add(
            (
                _text(binding.get("pair_group")),
                _text(binding.get("exact_mode")),
                _text(binding.get("orientation")),
            )
        )
    if len(identities) != 8:
        raise ValueError("OU v4 holdout identities are not unique")
    if int(contract.get("vendor_responses_at_registration", -1)) != 0:
        raise ValueError("OU v4 contract is not prospective")
    if int(receipt.get("vendor_responses_at_registration", -1)) != 0:
        raise ValueError("OU v4 receipt is not prospective")
    forbidden = (
        "candidate_promotion_authority",
        "testnet_order_authority",
        "live_trading_authorized",
    )
    if any(_truthy(contract.get(key)) or _truthy(receipt.get(key)) for key in forbidden):
        raise ValueError("OU v4 registration cannot grant trading authority")


def _implementation_source_hash() -> str:
    source = "\n\n".join(
        inspect.getsource(function)
        for function in (
            _ou_trend_aware_profile_beta_v3_candidate,
            _ou_trend_aware_profile_spread_v3_candidate,
            _ou_stationary_constrained_profile_fit_v4,
            _ou_stationary_constrained_profile_beta_v4,
            _ou_spread_candidate_v4,
            _ou_local_transform_trend_selector_v2,
        )
    )
    return sha256(source.encode("utf-8")).hexdigest()


def _resolve_source_path(root: Path, value: object) -> Path:
    path = Path(_text(value))
    path = path if path.is_absolute() else root / path
    if not path.is_file():
        raise ValueError(f"OU v4 evidence path is missing: {path}")
    return path


def _v4_blocker(
    transform_passed: bool,
    trend_passed: bool,
    formula_passed: bool,
) -> str:
    blockers: list[str] = []
    if not transform_passed:
        blockers.append("transform_selector_mismatch")
    if not trend_passed:
        blockers.append("trend_selector_mismatch")
    if not formula_passed:
        blockers.append("formula_parity_mismatch")
    return ";".join(blockers)


if __name__ == "__main__":
    result = run_ou_v4_prospective_holdout()
    print(json.dumps({"summary": result.summary}, indent=2))
