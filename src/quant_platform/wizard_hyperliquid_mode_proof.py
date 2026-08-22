from __future__ import annotations

from quant_platform.orchestration.corrective_runtime import promote_staged_file

from quant_platform.orchestration.corrective_runtime import atomic_write_text

from quant_platform.orchestration.corrective_runtime import atomic_write_csv

import inspect
import json
import math
import os
import platform
from datetime import UTC, datetime
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from quant_platform.active_pipeline import CommandResult
from quant_platform.api_extraction import CryptoWizardsFetchError
from quant_platform.crypto_wizards_history import (
    CryptoWizardsCustomSeriesBacktestRequest,
    fetch_credits_used,
    fetch_custom_series_backtest,
)
from quant_platform.crypto_wizards_sweep import parse_wizard_credit_usage
from quant_platform.wizard_hyperliquid_bridge import EXACT_MODE_VENDOR_PARAMS

ROOT = Path(__file__).resolve().parents[2]
REQUEST_CAP = 3
CUSTOM_SERIES_MAX_ROWS = 1100
CUSTOM_SERIES_CREDIT_COST = 2
DEFAULT_DAILY_CREDIT_LIMIT = 1000
DEFAULT_RESERVED_CREDITS = 100

PROOF_COLUMNS = [
    "candidate_set_id",
    "discovery_policy_schema_version",
    "discovery_policy_hash",
    "pair",
    "pair_group_id",
    "experiment_id",
    "asset_x",
    "asset_y",
    "venue",
    "local_interval",
    "exact_mode",
    "orientation",
    "vendor_request_strategy",
    "vendor_request_spread_type",
    "wizard_period",
    "proof_observations",
    "proof_window_kind",
    "history_rows",
    "mode_proof_status",
    "vendor_response_captured",
    "formula_proof_complete",
    "formula_comparator_generation",
    "formula_comparator_activation_id",
    "formula_comparator_activation_path",
    "formula_comparator_activation_sha256",
    "execution_enabled",
    "credits_estimated",
    "credits_used_before",
    "credits_remaining_before",
    "reserved_credits",
    "vendor_sharpe",
    "vendor_sortino",
    "vendor_total_return",
    "vendor_annual_return",
    "vendor_max_drawdown",
    "vendor_win_rate",
    "vendor_var",
    "vendor_cvar",
    "vendor_half_life",
    "vendor_hurst",
    "vendor_hedge_ratio",
    "vendor_last_zscore",
    "vendor_last_zscore_roll",
    "vendor_sigma0_crossings",
    "vendor_sigma2_crossings",
    "vendor_engle_granger_cointegrated",
    "vendor_engle_granger_pvalue",
    "vendor_copula_name",
    "vendor_u1_given_u2",
    "vendor_u2_given_u1",
    "vendor_copula_history_points",
    "vendor_formula_parity_status",
    "vendor_spread_formula",
    "vendor_spread_max_abs_error",
    "vendor_zscore_formula",
    "vendor_zscore_max_abs_error",
    "vendor_zscore_roll_formula",
    "vendor_zscore_roll_max_abs_error",
    "vendor_formula_pit_status",
    "vendor_history_available",
    "mode_computation_source",
    "promotion_allowed",
    "proof_validity",
    "validity_blocker",
    "training_eligible",
    "blocker",
    "next_step",
    "request_path",
    "response_path",
    "wizard_settings_evidence_path",
    "local_history_path",
    "evidence_path",
]

EXHAUSTIVE_QUEUE_COLUMNS = [
    "pair_group_id",
    "experiment_id",
    "pair",
    "asset_x",
    "asset_y",
    "venue",
    "local_interval",
    "local_history_path",
    "exact_mode",
    "orientation",
    "vendor_request_strategy",
    "vendor_request_spread_type",
    "wizard_period",
    "proof_observations",
    "entry_level",
    "exit_level",
    "x_weighting",
    "slippage_rate",
    "commission_rate",
    "roll_w",
    "roll_w_source",
    "stop_loss_rate",
    "exit_n_periods",
    "wizard_capture_status",
    "wizard_orientation_verified",
    "vendor_custom_series_eligible",
    "blocker",
    "wizard_settings_evidence_path",
    "evidence_path",
]


def build_exhaustive_wizard_mode_proof_queue(
    *,
    root: Path = ROOT,
    pair_group_id: str | None = None,
) -> CommandResult:
    """Materialize proof-ready exact-mode cells without discovery-score filtering.

    Formula parity is a mode-provenance question, so Sharpe and return thresholds do
    not belong in this queue. Rows still require an orientation-verified Wizard
    capture and a complete frozen Hyperliquid pair history.
    """

    active = root / "reports" / "active"
    ledger_path = active / "exhaustive_wizard_pair_detail_mode_ledger.csv"
    history_path = active / "exhaustive_wizard_hyperliquid_pair_history_results.csv"
    ledger = _read_csv(ledger_path)
    histories = _read_csv(history_path)
    if pair_group_id:
        ledger = ledger.loc[
            ledger.get("pair_group_id", pd.Series("", index=ledger.index))
            .map(_text)
            .eq(pair_group_id)
        ].copy()
        histories = histories.loc[
            histories.get("pair_group_id", pd.Series("", index=histories.index))
            .map(_text)
            .eq(pair_group_id)
        ].copy()

    ready_histories: dict[str, pd.Series] = {}
    if not histories.empty:
        ready = histories.loc[
            histories.get("history_status", pd.Series("", index=histories.index))
            .map(_text)
            .eq("READY_FOR_CANONICAL_REPLAY")
        ].copy()
        if not ready.empty:
            ready = ready.sort_values(
                [
                    column
                    for column in ("latest_candle_at", "history_rows")
                    if column in ready.columns
                ],
                ascending=False,
            )
            for _, row in ready.iterrows():
                ready_histories.setdefault(_text(row.get("pair_group_id", "")), row)

    rows: list[dict[str, object]] = []
    captured = ledger.loc[
        ledger.get("exact_mode", pd.Series("", index=ledger.index)).isin(EXACT_MODE_VENDOR_PARAMS)
    ].copy()
    mode_order = {mode: index for index, mode in enumerate(EXACT_MODE_VENDOR_PARAMS)}
    orientation_order = {"original": 0, "reverse": 1}
    for _, capture in captured.iterrows():
        group_id = _text(capture.get("pair_group_id", ""))
        history = ready_histories.get(group_id)
        blockers: list[str] = []
        capture_status = _text(capture.get("capture_status", ""))
        orientation_verified = _truthy(capture.get("orientation_verified", False))
        if capture_status != "CAPTURED":
            blockers.append("wizard_exact_mode_not_captured")
        if not orientation_verified:
            blockers.append("wizard_orientation_not_verified")
        if history is None:
            blockers.append("frozen_hyperliquid_pair_history_not_ready")

        exact_mode = _text(capture.get("exact_mode", ""))
        vendor = EXACT_MODE_VENDOR_PARAMS.get(exact_mode, {})
        entry_level = _entry_level(capture, exact_mode=exact_mode)
        exit_level = _exit_level(capture)
        x_weighting = _float(capture.get("x_weighting", ""))
        commission_pct = _float(capture.get("wizard_commission_pct", ""))
        slippage_pct = _float(capture.get("wizard_slippage_pct", ""))
        period = _int_or_none(capture.get("periods_analyzed", ""))
        rolling_window = _int_or_none(capture.get("rolling_window", ""))
        roll_w = rolling_window or 42
        for name, value in (
            ("entry_level", entry_level),
            ("exit_level", exit_level),
            ("x_weighting", x_weighting),
            ("wizard_commission_pct", commission_pct),
            ("wizard_slippage_pct", slippage_pct),
        ):
            if value is None:
                blockers.append(f"captured_setting_missing:{name}")
        if period is None or period < 50:
            blockers.append("wizard_period_missing_or_below_vendor_minimum")

        orientation = _text(capture.get("orientation", ""))
        desired_x = _capture_asset(capture, "x")
        desired_y = _capture_asset(capture, "y")
        local_history_path = _text(history.get("history_path", "")) if history is not None else ""
        evidence = ";".join(
            value
            for value in (
                _relative(ledger_path, root=root),
                _relative(history_path, root=root),
                _text(capture.get("evidence_path", "")),
                local_history_path,
            )
            if value
        )
        rows.append(
            {
                "pair_group_id": group_id,
                "experiment_id": _text(capture.get("experiment_id", "")),
                "pair": f"{desired_x}-{desired_y}"
                if desired_x and desired_y
                else _text(capture.get("pair", "")),
                "asset_x": desired_x,
                "asset_y": desired_y,
                "venue": "hyperliquid",
                "local_interval": _text(history.get("hyperliquid_interval", ""))
                if history is not None
                else "",
                "local_history_path": local_history_path,
                "exact_mode": exact_mode,
                "orientation": orientation,
                "vendor_request_strategy": vendor.get("strategy", ""),
                "vendor_request_spread_type": vendor.get("spread_type", ""),
                "wizard_period": period or "",
                "proof_observations": period or "",
                "entry_level": entry_level if entry_level is not None else "",
                "exit_level": exit_level if exit_level is not None else "",
                "x_weighting": x_weighting if x_weighting is not None else "",
                "slippage_rate": slippage_pct / 100.0 if slippage_pct is not None else "",
                "commission_rate": commission_pct / 100.0 if commission_pct is not None else "",
                "roll_w": roll_w,
                "roll_w_source": "captured_mode_setting"
                if rolling_window
                else "api_required_inactive_default_42",
                "stop_loss_rate": _positive_rate_from_pct(capture.get("stop_loss_pct", "")),
                "exit_n_periods": _positive_int(capture.get("close_n_periods", "")),
                "wizard_capture_status": capture_status,
                "wizard_orientation_verified": orientation_verified,
                "vendor_custom_series_eligible": not blockers,
                "blocker": ";".join(blockers),
                "wizard_settings_evidence_path": _text(capture.get("evidence_path", "")),
                "evidence_path": evidence,
                "_mode_order": mode_order.get(exact_mode, 999),
                "_orientation_order": orientation_order.get(orientation, 999),
                "_capture_timestamp": _text(capture.get("capture_timestamp", "")),
            }
        )

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = _normalize_copula_proof_windows(frame)
        group_priority = (
            frame.groupby("pair_group_id", dropna=False)
            .agg(
                eligible_cells=("vendor_custom_series_eligible", "sum"),
                latest_capture=("_capture_timestamp", "max"),
            )
            .sort_values(["eligible_cells", "latest_capture"], ascending=[False, False])
        )
        priority = {group_id: index for index, group_id in enumerate(group_priority.index)}
        frame["_group_priority"] = frame["pair_group_id"].map(priority).fillna(999)
        frame = frame.sort_values(
            ["_group_priority", "_mode_order", "_orientation_order"],
            kind="stable",
        )
        frame = frame[EXHAUSTIVE_QUEUE_COLUMNS].reset_index(drop=True)
    else:
        frame = pd.DataFrame(columns=EXHAUSTIVE_QUEUE_COLUMNS)

    output = active / "exhaustive_wizard_exact_mode_proof_queue.csv"
    summary_path = active / "exhaustive_wizard_exact_mode_proof_queue.md"
    _write_csv(frame, output)
    _write_text(
        summary_path,
        _exhaustive_queue_markdown(frame, ledger_path=ledger_path, history_path=history_path),
    )
    return CommandResult(
        paths={"queue": output, "summary_md": summary_path},
        summary={
            "rows": len(frame),
            "eligible_rows": int(
                frame.get("vendor_custom_series_eligible", pd.Series(dtype=bool)).map(_truthy).sum()
            ),
            "pair_groups": int(frame.get("pair_group_id", pd.Series(dtype=str)).nunique()),
            "promotion_allowed": False,
            "live_trading_authorized": False,
        },
    )


def _normalize_copula_proof_windows(frame: pd.DataFrame) -> pd.DataFrame:
    """Bind each Copula orientation pair to one deterministic source window."""

    output = frame.copy()
    copula = output.loc[output["exact_mode"].eq("Copula")]
    for group_id, group in copula.groupby("pair_group_id", dropna=False):
        indexes = list(group.index)
        orientations = group["orientation"].map(_text).tolist()
        blockers: list[str] = []
        if len(indexes) != 2 or sorted(orientations) != ["original", "reverse"]:
            blockers.append("copula_orientation_pair_incomplete_or_duplicate")
        if group["local_history_path"].map(_text).nunique(dropna=False) != 1:
            blockers.append("copula_orientation_history_source_mismatch")
        if group["local_interval"].map(_text).nunique(dropna=False) != 1:
            blockers.append("copula_orientation_interval_mismatch")
        observations = [_int_or_none(value) for value in group["proof_observations"].tolist()]
        if any(value is None or value < 50 for value in observations):
            blockers.append("copula_orientation_proof_window_invalid")
        if blockers:
            for index in indexes:
                output.at[index, "vendor_custom_series_eligible"] = False
                output.at[index, "blocker"] = _merge_blockers(
                    output.at[index, "blocker"], *blockers
                )
            continue
        shared_observations = min(int(value) for value in observations if value is not None)
        for index in indexes:
            output.at[index, "proof_observations"] = shared_observations
    return output


def _apply_copula_input_swap_audit(
    frame: pd.DataFrame,
    *,
    eligible: pd.DataFrame,
    requests: dict[tuple[str, str], list[CryptoWizardsCustomSeriesBacktestRequest]],
) -> pd.DataFrame:
    """Fail before the vendor boundary unless Copula inputs are exact swaps."""

    if frame.empty:
        return frame
    output = frame.copy().astype(object)
    groups = (
        eligible.loc[eligible.get("exact_mode", pd.Series("", index=eligible.index)).eq("Copula")]
        .get("pair_group_id", pd.Series(dtype=str))
        .map(_text)
        .unique()
    )
    for group_id in groups:
        original = requests.get((group_id, "original"), [])
        reverse = requests.get((group_id, "reverse"), [])
        exact_swap = bool(
            len(original) == 1
            and len(reverse) == 1
            and original[0].series_1_opens == reverse[0].series_2_opens
            and original[0].series_1_closes == reverse[0].series_2_closes
            and original[0].series_2_opens == reverse[0].series_1_opens
            and original[0].series_2_closes == reverse[0].series_1_closes
        )
        if exact_swap:
            continue
        mask = output.get("pair_group_id", pd.Series("", index=output.index)).map(_text).eq(
            group_id
        ) & output.get("exact_mode", pd.Series("", index=output.index)).eq("Copula")
        output.loc[mask, "input_audit_status"] = "BLOCKED"
        output.loc[mask, "retry_safety_status"] = "BLOCKED_COPULA_ORIENTATION_CONTRACT"
        output.loc[mask, "blocker"] = "copula_orientation_requests_not_exact_swaps"
    return output


def _merge_blockers(value: object, *blockers: str) -> str:
    merged = [part for part in _text(value).split(";") if part]
    for blocker in blockers:
        if blocker and blocker not in merged:
            merged.append(blocker)
    return ";".join(merged)


def audit_exhaustive_wizard_mode_proof_inputs(
    *, root: Path = ROOT, queue_path: Path | None = None
) -> CommandResult:
    """Validate every eligible custom-series request locally without API access."""

    active = root / "reports" / "active"
    queue_path = queue_path or active / "exhaustive_wizard_exact_mode_proof_queue.csv"
    queue = _read_csv(queue_path)
    prior_proofs = _read_csv(active / "hyperliquid_wizard_vendor_mode_proofs.csv")
    eligible = queue.loc[
        queue.get("vendor_custom_series_eligible", pd.Series(False, index=queue.index)).map(_truthy)
    ].copy()
    rows: list[dict[str, object]] = []
    copula_requests: dict[tuple[str, str], list[CryptoWizardsCustomSeriesBacktestRequest]] = {}
    for _, candidate in eligible.iterrows():
        base = {
            "pair_group_id": _text(candidate.get("pair_group_id", "")),
            "experiment_id": _text(candidate.get("experiment_id", "")),
            "pair": _text(candidate.get("pair", "")),
            "asset_x": _text(candidate.get("asset_x", "")),
            "asset_y": _text(candidate.get("asset_y", "")),
            "local_interval": _text(candidate.get("local_interval", "")),
            "exact_mode": _text(candidate.get("exact_mode", "")),
            "orientation": _text(candidate.get("orientation", "")),
            "proof_observations_requested": _identity_observations(candidate),
            "local_history_path": _text(candidate.get("local_history_path", "")),
            "queue_evidence_path": _relative(queue_path, root=root),
            "api_call_performed": False,
            "credits_consumed": 0,
            "promotion_allowed": False,
            "live_trading_authorized": False,
        }
        try:
            request = _request_from_candidate(candidate, root=root)
            if _text(candidate.get("exact_mode", "")) == "Copula":
                key = (
                    _text(candidate.get("pair_group_id", "")),
                    _text(candidate.get("orientation", "")),
                )
                copula_requests.setdefault(key, []).append(request)
            payload = request.payload()
            observations = len(request.series_1_closes)
            request_fingerprint = sha256(
                json.dumps(
                    payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()
            rejection_evidence = _prior_client_rejection_payloads(
                candidate=candidate,
                prior_proofs=prior_proofs,
                root=root,
            )
            prior_hashes = rejection_evidence["payload_hashes"]
            unverifiable_rejections = int(rejection_evidence["unverifiable_count"])
            if unverifiable_rejections:
                audit_status = "BLOCKED"
                retry_safety_status = "BLOCKED_PRIOR_VENDOR_4XX_PAYLOAD_UNVERIFIABLE"
                blocker = "prior_vendor_4xx_request_payload_unverifiable"
            elif request_fingerprint in prior_hashes:
                audit_status = "BLOCKED"
                retry_safety_status = "BLOCKED_UNCHANGED_VENDOR_4XX_PAYLOAD"
                blocker = "unchanged_payload_previously_rejected_by_vendor_4xx"
            elif prior_hashes:
                audit_status = "READY"
                retry_safety_status = "READY_PAYLOAD_CHANGED_AFTER_VENDOR_4XX"
                blocker = ""
            else:
                audit_status = "READY"
                retry_safety_status = "NOT_APPLICABLE"
                blocker = ""
            rows.append(
                {
                    **base,
                    "input_audit_status": audit_status,
                    "request_observations": observations,
                    "proof_window_kind": _proof_window_kind(
                        candidate.get("wizard_period", ""), observations
                    ),
                    "request_strategy": request.strategy,
                    "request_spread_type": request.spread_type or "",
                    "wire_spread_type": _text(payload.get("params", {}).get("spread_type", "")),
                    "request_fingerprint": request_fingerprint,
                    "prior_vendor_4xx_count": int(rejection_evidence["rejection_count"]),
                    "prior_vendor_4xx_payload_hashes": ";".join(sorted(prior_hashes)),
                    "retry_safety_status": retry_safety_status,
                    "blocker": blocker,
                }
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            rows.append(
                {
                    **base,
                    "input_audit_status": "BLOCKED",
                    "request_observations": 0,
                    "proof_window_kind": "unknown_window",
                    "request_strategy": _text(candidate.get("vendor_request_strategy", "")),
                    "request_spread_type": _text(candidate.get("vendor_request_spread_type", "")),
                    "wire_spread_type": "",
                    "request_fingerprint": "",
                    "prior_vendor_4xx_count": 0,
                    "prior_vendor_4xx_payload_hashes": "",
                    "retry_safety_status": "BLOCKED_INPUT_CONSTRUCTION",
                    "blocker": f"{type(exc).__name__}:{exc}",
                }
            )
    frame = pd.DataFrame(rows)
    frame = _apply_copula_input_swap_audit(
        frame,
        eligible=eligible,
        requests=copula_requests,
    )
    audit_path = active / "exhaustive_wizard_mode_proof_input_audit.csv"
    summary_path = active / "exhaustive_wizard_mode_proof_input_audit.json"
    _write_csv(frame, audit_path)
    ready = int(frame.get("input_audit_status", pd.Series(dtype=str)).eq("READY").sum())
    blocked = len(frame) - ready
    status = "PASS" if len(frame) > 0 and blocked == 0 else "BLOCKED"
    summary = {
        "status": status,
        "eligible_rows": len(eligible),
        "audited_rows": len(frame),
        "ready_rows": ready,
        "blocked_rows": blocked,
        "retry_safe_rows": int(
            frame.get("retry_safety_status", pd.Series(dtype=str))
            .isin({"NOT_APPLICABLE", "READY_PAYLOAD_CHANGED_AFTER_VENDOR_4XX"})
            .sum()
        ),
        "changed_after_vendor_4xx_rows": int(
            frame.get("retry_safety_status", pd.Series(dtype=str))
            .eq("READY_PAYLOAD_CHANGED_AFTER_VENDOR_4XX")
            .sum()
        ),
        "unchanged_vendor_4xx_rows": int(
            frame.get("retry_safety_status", pd.Series(dtype=str))
            .eq("BLOCKED_UNCHANGED_VENDOR_4XX_PAYLOAD")
            .sum()
        ),
        "all_eligible_rows_accounted": len(frame) == len(eligible),
        "api_calls_performed": 0,
        "credits_consumed": 0,
        "blocker": ("" if status == "PASS" else "one_or_more_exact_mode_inputs_failed_local_audit"),
        "promotion_allowed": False,
        "live_trading_authorized": False,
        "evidence_path": _relative(audit_path, root=root),
    }
    _write_json(summary_path, summary)
    return CommandResult(
        paths={"input_audit": audit_path, "input_audit_summary": summary_path},
        summary=summary,
    )


def _prior_client_rejection_payloads(
    *, candidate: pd.Series, prior_proofs: pd.DataFrame, root: Path
) -> dict[str, object]:
    """Return immutable payload hashes for prior permanent client rejections."""

    if prior_proofs.empty:
        return {"rejection_count": 0, "payload_hashes": set(), "unverifiable_count": 0}
    matching = prior_proofs.loc[
        prior_proofs.apply(lambda row: _proof_identity(row) == _proof_identity(candidate), axis=1)
        & prior_proofs.get("mode_proof_status", pd.Series("", index=prior_proofs.index)).eq(
            "request_failed"
        )
        & prior_proofs.get("blocker", pd.Series("", index=prior_proofs.index))
        .map(_text)
        .str.contains(r"\b4\d\d Client Error\b", regex=True)
    ]
    hashes: set[str] = set()
    unverifiable = 0
    for _, row in matching.iterrows():
        request_path = _resolve_path(_text(row.get("request_path", "")), root=root)
        if request_path is None or not request_path.is_file():
            unverifiable += 1
            continue
        try:
            payload = json.loads(request_path.read_text(encoding="utf-8"))
            hashes.add(
                sha256(
                    json.dumps(
                        payload,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode("utf-8")
                ).hexdigest()
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            unverifiable += 1
    return {
        "rejection_count": len(matching),
        "payload_hashes": hashes,
        "unverifiable_count": unverifiable,
    }


def run_hyperliquid_wizard_mode_proofs(
    *,
    root: Path = ROOT,
    max_pairs: int = REQUEST_CAP,
    execute: bool = False,
    api_key: str | None = None,
    now: datetime | None = None,
    queue_path: Path | None = None,
    daily_credit_limit: int = DEFAULT_DAILY_CREDIT_LIMIT,
    reserved_credits: int = DEFAULT_RESERVED_CREDITS,
    credits_fetcher: Any | None = None,
) -> CommandResult:
    """Preflight or run bounded Wizard custom-series proofs for queue-ready pairs.

    The default is a no-credit preflight. Execution needs both ``execute=True`` and
    ``QPA_ENABLE_WIZARD_CUSTOM_SERIES_PROOF=true``. Results remain research-only.
    """

    timestamp = _as_utc(now or datetime.now(UTC))
    active = root / "reports" / "active"
    dashboard = root / "reports" / "dashboard"
    queue_path = queue_path or active / "hyperliquid_wizard_hypothesis_queue.csv"
    proof_path = active / "hyperliquid_wizard_vendor_mode_proofs.csv"
    queue = _read_csv(queue_path)
    eligible = queue.loc[
        queue.get("vendor_custom_series_eligible", pd.Series(False, index=queue.index)).map(_truthy)
    ].copy()
    eligible_total = len(eligible)
    existing_captured = pd.DataFrame()
    failed_today = pd.DataFrame()
    existing = _read_csv(proof_path)
    if not existing.empty and "mode_proof_status" in existing.columns:
        captured_mask = existing.get(
            "vendor_response_captured", pd.Series(False, index=existing.index)
        ).map(_truthy) | existing["mode_proof_status"].eq("completed")
        existing_captured = existing.loc[captured_mask].copy()
        existing_captured = _refresh_existing_metrics(existing_captured, root=root)
        existing_captured = _enrich_existing_completed(existing_captured, queue=queue)
        failed_today = existing.loc[
            existing["mode_proof_status"].eq("request_failed")
            & existing.get("request_path", pd.Series("", index=existing.index))
            .map(_text)
            .str.contains(f"/{timestamp.date().isoformat()}_", regex=False)
        ].copy()
        attempted_keys = {
            _proof_identity(row)
            for frame in (existing_captured, failed_today)
            for _, row in frame.iterrows()
        }
        eligible = eligible.loc[
            ~eligible.apply(lambda row: _proof_identity(row) in attempted_keys, axis=1)
        ].copy()
    selected = eligible.head(max(0, min(int(max_pairs), REQUEST_CAP))).reset_index(drop=True)

    execution_enabled = bool(
        execute and _truthy(os.getenv("QPA_ENABLE_WIZARD_CUSTOM_SERIES_PROOF", ""))
    )
    resolved_api_key = api_key or (
        os.getenv("CRYPTO_WIZARDS_API_KEY", "").strip() if execute else ""
    )
    credit_state = _credit_preflight(
        execute=execute,
        execution_enabled=execution_enabled,
        api_key=resolved_api_key,
        selected=len(selected),
        daily_credit_limit=daily_credit_limit,
        reserved_credits=reserved_credits,
        credits_fetcher=credits_fetcher,
    )
    retained = (
        pd.concat(
            [frame for frame in (existing_captured, failed_today) if not frame.empty],
            ignore_index=True,
        )
        if not existing_captured.empty or not failed_today.empty
        else pd.DataFrame()
    )
    rows: list[dict[str, object]] = retained.to_dict("records") if not retained.empty else []
    for _, candidate in selected.iterrows():
        rows.append(
            _proof_row(
                candidate,
                root=root,
                timestamp=timestamp,
                execute=execute,
                execution_enabled=execution_enabled,
                api_key=resolved_api_key,
                queue_path=queue_path,
                credit_state=credit_state,
            )
        )

    frame = pd.DataFrame(rows, columns=PROOF_COLUMNS)
    active.mkdir(parents=True, exist_ok=True)
    dashboard.mkdir(parents=True, exist_ok=True)
    summary_path = active / "hyperliquid_wizard_vendor_mode_proofs.md"
    dashboard_path = dashboard / "hyperliquid_wizard_vendor_mode_proofs_dashboard.csv"
    _write_csv(frame, proof_path)
    _write_csv(frame, dashboard_path)
    _write_text(
        summary_path,
        _markdown(frame, selected=len(selected), eligible=eligible_total, execute=execute),
    )

    completed = (
        int(frame.get("mode_proof_status", pd.Series(dtype=str)).eq("completed").sum())
        if not frame.empty
        else 0
    )
    queue_completed = count_completed_exact_mode_proofs(
        queue_path=queue_path,
        proof_path=proof_path,
    )
    queue_responses_captured = count_captured_exact_mode_responses(
        queue_path=queue_path,
        proof_path=proof_path,
    )
    selected_identities = {_proof_identity(row) for _, row in selected.iterrows()}
    selected_results = (
        frame.loc[frame.apply(lambda row: _proof_identity(row) in selected_identities, axis=1)]
        if not frame.empty and selected_identities
        else pd.DataFrame(columns=frame.columns)
    )
    selected_failures = selected_results.loc[
        selected_results.get("mode_proof_status", pd.Series("", index=selected_results.index)).eq(
            "request_failed"
        )
    ].copy()
    selected_failure_identities = {_proof_identity(row) for _, row in selected_failures.iterrows()}
    selected_failures_quarantined = bool(
        not selected_failures.empty
        and len(selected_failure_identities) == len(selected_failures)
        and selected_failures.get("request_path", pd.Series("", index=selected_failures.index))
        .map(_text)
        .str.contains(f"/{timestamp.date().isoformat()}_", regex=False)
        .all()
    )
    return CommandResult(
        paths={"proofs": proof_path, "summary_md": summary_path, "dashboard": dashboard_path},
        summary={
            "queue_eligible": eligible_total,
            "selected": len(selected),
            "completed": completed,
            "queue_completed": queue_completed,
            "queue_responses_captured": queue_responses_captured,
            "queue_remaining": max(eligible_total - queue_completed, 0),
            "selected_completed": int(
                selected_results.get(
                    "formula_proof_complete",
                    pd.Series(False, index=selected_results.index),
                )
                .map(_truthy)
                .sum()
            ),
            "selected_responses_captured": int(
                selected_results.get(
                    "vendor_response_captured",
                    pd.Series(False, index=selected_results.index),
                )
                .map(_truthy)
                .sum()
            ),
            "selected_credit_blocked": int(
                selected_results.get("mode_proof_status", pd.Series(dtype=str))
                .eq("credit_blocked")
                .sum()
            ),
            "selected_request_failed": len(selected_failures),
            "selected_failures_quarantined": selected_failures_quarantined,
            "selected_preflight_ready": int(
                selected_results.get("mode_proof_status", pd.Series(dtype=str))
                .eq("preflight_ready")
                .sum()
            ),
            "external_proof_requests": int(
                selected_results.get("mode_proof_status", pd.Series(dtype=str))
                .isin({"completed", "request_failed"})
                .sum()
            ),
            "preflight_only": bool(not execute),
            "execution_enabled": execution_enabled,
            "request_cap": REQUEST_CAP,
            "credit_preflight_status": credit_state["status"],
            "credit_preflight_blocker": credit_state["blocker"],
            "credits_used_before": credit_state["used"],
            "credits_remaining_before": credit_state["remaining"],
            "reserved_credits": reserved_credits,
        },
    )


def count_completed_exact_mode_proofs(*, queue_path: Path, proof_path: Path) -> int:
    """Count non-Copula exact formula proofs that belong to the eligible queue."""

    queue = _read_csv(queue_path)
    proofs = _read_csv(proof_path)
    if queue.empty or proofs.empty:
        return 0
    eligible = queue.loc[
        queue.get("vendor_custom_series_eligible", pd.Series(False, index=queue.index)).map(_truthy)
        & queue.get("exact_mode", pd.Series("", index=queue.index)).ne("Copula")
    ]
    completed = proofs.loc[_formula_proof_mask(proofs)]
    queue_identities = {_proof_identity(row) for _, row in eligible.iterrows()}
    proof_identities = {_proof_identity(row) for _, row in completed.iterrows()}
    return len(queue_identities & proof_identities)


def count_captured_exact_mode_responses(*, queue_path: Path, proof_path: Path) -> int:
    """Count vendor responses without treating them as exact formula proofs."""

    queue = _read_csv(queue_path)
    proofs = _read_csv(proof_path)
    if queue.empty or proofs.empty:
        return 0
    eligible = queue.loc[
        queue.get("vendor_custom_series_eligible", pd.Series(False, index=queue.index)).map(_truthy)
    ]
    captured = proofs.loc[
        proofs.get("vendor_response_captured", pd.Series(False, index=proofs.index)).map(_truthy)
        | proofs.get("mode_proof_status", pd.Series("", index=proofs.index)).eq("completed")
    ]
    queue_identities = {_proof_identity(row) for _, row in eligible.iterrows()}
    proof_identities = {_proof_identity(row) for _, row in captured.iterrows()}
    return len(queue_identities & proof_identities)


def _proof_row(
    candidate: pd.Series,
    *,
    root: Path,
    timestamp: datetime,
    execute: bool,
    execution_enabled: bool,
    api_key: str,
    queue_path: Path,
    credit_state: dict[str, object],
) -> dict[str, object]:
    base = _base_row(candidate, queue_path=queue_path)
    try:
        request = _request_from_candidate(candidate, root=root)
    except ValueError as exc:
        return {
            **base,
            "mode_proof_status": "input_blocked",
            "blocker": str(exc),
            "next_step": "repair_hypothesis_queue_inputs",
        }

    if not execute:
        return {
            **base,
            "history_rows": len(request.series_1_closes),
            "proof_observations": len(request.series_1_closes),
            "proof_window_kind": _proof_window_kind(
                base.get("wizard_period", ""), len(request.series_1_closes)
            ),
            "mode_proof_status": "preflight_ready",
            "execution_enabled": False,
            "credits_estimated": 0,
            "next_step": "set QPA_ENABLE_WIZARD_CUSTOM_SERIES_PROOF=true and rerun with --execute-wizard-proof after review",
        }
    if not execution_enabled:
        return {
            **base,
            "history_rows": len(request.series_1_closes),
            "proof_observations": len(request.series_1_closes),
            "proof_window_kind": _proof_window_kind(
                base.get("wizard_period", ""), len(request.series_1_closes)
            ),
            "mode_proof_status": "execution_disabled",
            "execution_enabled": False,
            "credits_estimated": 0,
            "blocker": "QPA_ENABLE_WIZARD_CUSTOM_SERIES_PROOF_not_true",
            "next_step": "enable the bounded custom-series proof gate only after reviewing the selected rows",
        }
    if not api_key:
        return {
            **base,
            "history_rows": len(request.series_1_closes),
            "proof_observations": len(request.series_1_closes),
            "proof_window_kind": _proof_window_kind(
                base.get("wizard_period", ""), len(request.series_1_closes)
            ),
            "mode_proof_status": "execution_blocked",
            "execution_enabled": True,
            "credits_estimated": 0,
            "blocker": "CRYPTO_WIZARDS_API_KEY_missing",
            "next_step": "configure the key in .env.local",
        }
    if credit_state["status"] != "PASS":
        return {
            **base,
            "history_rows": len(request.series_1_closes),
            "proof_observations": len(request.series_1_closes),
            "proof_window_kind": _proof_window_kind(
                base.get("wizard_period", ""), len(request.series_1_closes)
            ),
            "mode_proof_status": "credit_blocked",
            "execution_enabled": True,
            "credits_estimated": 0,
            "credits_used_before": credit_state["used"],
            "credits_remaining_before": credit_state["remaining"],
            "reserved_credits": credit_state["reserved"],
            "blocker": credit_state["blocker"],
            "next_step": "wait for the Wizard daily credit reset or reduce only the reviewed request set without breaching reserve",
        }

    raw_dir = (
        root
        / "data"
        / "raw"
        / "crypto_wizards_custom_series_proofs"
        / timestamp.strftime("%Y-%m-%d_%H%M%S")
    )
    stem = _safe_filename(
        f"{candidate.get('pair', '')}_{candidate.get('local_interval', '')}_"
        f"{candidate.get('exact_mode', '')}_{candidate.get('orientation', '')}"
    )
    request_path = raw_dir / f"{stem}_request.json"
    response_path = raw_dir / f"{stem}_response.json"
    request_payload = request.payload()
    _write_json(request_path, request_payload)
    try:
        response = fetch_custom_series_backtest(request, api_key=api_key)
    except (CryptoWizardsFetchError, OSError, TypeError, ValueError) as exc:  # pragma: no cover
        return {
            **base,
            "history_rows": len(request.series_1_closes),
            "proof_observations": len(request.series_1_closes),
            "proof_window_kind": _proof_window_kind(
                base.get("wizard_period", ""), len(request.series_1_closes)
            ),
            "mode_proof_status": "request_failed",
            "execution_enabled": True,
            "credits_estimated": 2,
            "blocker": f"{type(exc).__name__}:{exc}",
            "next_step": "inspect the saved request and API response settings before retrying",
            "request_path": _relative(request_path, root=root),
        }
    _write_json(response_path, response)
    comparator = _active_dynamic_comparator_context(root)
    generation = (
        int(comparator["generation"]) if _text(base.get("exact_mode")).startswith("Dyn") else 1
    )
    metrics = {
        **_response_metrics(response),
        **_formula_parity_metrics_for_generation(
            request_payload,
            response,
            dynamic_generation=generation,
        ),
    }
    formula_proven = metrics["vendor_formula_parity_status"] == "exact_reconstruction"
    completed_row = {
        **base,
        "history_rows": len(request.series_1_closes),
        "proof_observations": len(request.series_1_closes),
        "proof_window_kind": _proof_window_kind(
            base.get("wizard_period", ""), len(request.series_1_closes)
        ),
        "mode_proof_status": "completed",
        "vendor_response_captured": True,
        "formula_proof_complete": formula_proven,
        "formula_comparator_generation": generation,
        "formula_comparator_activation_id": (
            comparator["activation_id"] if generation == 2 else ""
        ),
        "formula_comparator_activation_path": (
            comparator["activation_path"] if generation == 2 else ""
        ),
        "formula_comparator_activation_sha256": (
            comparator["activation_sha256"] if generation == 2 else ""
        ),
        "execution_enabled": True,
        "credits_estimated": CUSTOM_SERIES_CREDIT_COST,
        "credits_used_before": credit_state["used"],
        "credits_remaining_before": credit_state["remaining"],
        "reserved_credits": credit_state["reserved"],
        **metrics,
        "mode_computation_source": "crypto_wizards_custom_series_backtest",
        "next_step": "inspect vendor history and signal availability, then run a separate local after-cost replay",
        "request_path": _relative(request_path, root=root),
        "response_path": _relative(response_path, root=root),
        "evidence_path": f"{base['evidence_path']};{_relative(request_path, root=root)};{_relative(response_path, root=root)}",
    }
    return {**completed_row, **_proof_validity_fields(pd.Series(completed_row))}


def _base_row(candidate: pd.Series, *, queue_path: Path) -> dict[str, object]:
    return {
        "candidate_set_id": _text(candidate.get("candidate_set_id", "")),
        "discovery_policy_schema_version": _text(
            candidate.get("discovery_policy_schema_version", "")
        ),
        "discovery_policy_hash": _text(candidate.get("discovery_policy_hash", "")),
        "pair": _text(candidate.get("pair", "")),
        "pair_group_id": _text(candidate.get("pair_group_id", "")),
        "experiment_id": _text(candidate.get("experiment_id", "")),
        "asset_x": _text(candidate.get("asset_x", "")),
        "asset_y": _text(candidate.get("asset_y", "")),
        "venue": _text(candidate.get("venue", "")),
        "local_interval": _text(candidate.get("local_interval", "")),
        "exact_mode": _text(candidate.get("exact_mode", "")),
        "orientation": _text(candidate.get("orientation", "")),
        "vendor_request_strategy": _text(candidate.get("vendor_request_strategy", "")),
        "vendor_request_spread_type": _text(candidate.get("vendor_request_spread_type", "")),
        "wizard_period": _int_or_none(candidate.get("wizard_period", "")) or "",
        "proof_observations": _int_or_none(candidate.get("proof_observations", "")) or "",
        "proof_window_kind": "",
        "history_rows": "",
        "mode_proof_status": "",
        "vendor_response_captured": False,
        "formula_proof_complete": False,
        "formula_comparator_generation": 1,
        "formula_comparator_activation_id": "",
        "formula_comparator_activation_path": "",
        "formula_comparator_activation_sha256": "",
        "execution_enabled": False,
        "credits_estimated": 0,
        "credits_used_before": "",
        "credits_remaining_before": "",
        "reserved_credits": "",
        "vendor_sharpe": "",
        "vendor_sortino": "",
        "vendor_total_return": "",
        "vendor_annual_return": "",
        "vendor_max_drawdown": "",
        "vendor_win_rate": "",
        "vendor_var": "",
        "vendor_cvar": "",
        "vendor_half_life": "",
        "vendor_hurst": "",
        "vendor_hedge_ratio": "",
        "vendor_last_zscore": "",
        "vendor_last_zscore_roll": "",
        "vendor_sigma0_crossings": "",
        "vendor_sigma2_crossings": "",
        "vendor_engle_granger_cointegrated": "",
        "vendor_engle_granger_pvalue": "",
        "vendor_copula_name": "",
        "vendor_u1_given_u2": "",
        "vendor_u2_given_u1": "",
        "vendor_copula_history_points": 0,
        "vendor_formula_parity_status": "",
        "vendor_spread_formula": "",
        "vendor_spread_max_abs_error": "",
        "vendor_zscore_formula": "",
        "vendor_zscore_max_abs_error": "",
        "vendor_zscore_roll_formula": "",
        "vendor_zscore_roll_max_abs_error": "",
        "vendor_formula_pit_status": "",
        "vendor_history_available": False,
        "mode_computation_source": "",
        "promotion_allowed": False,
        "proof_validity": "NOT_COMPLETED",
        "validity_blocker": "vendor_proof_not_completed",
        "training_eligible": False,
        "blocker": "",
        "next_step": "",
        "request_path": "",
        "response_path": "",
        "wizard_settings_evidence_path": _text(candidate.get("wizard_settings_evidence_path", "")),
        "local_history_path": _text(candidate.get("local_history_path", "")),
        "evidence_path": f"{_relative(queue_path, root=queue_path.parents[2])};{_text(candidate.get('evidence_path', ''))}",
    }


def _request_from_candidate(
    candidate: pd.Series, *, root: Path
) -> CryptoWizardsCustomSeriesBacktestRequest:
    history_path = _resolve_path(_text(candidate.get("local_history_path", "")), root=root)
    if history_path is None or not history_path.exists():
        raise ValueError("local_history_path_missing")
    payload = json.loads(history_path.read_text(encoding="utf-8"))
    history = pd.DataFrame(payload.get("history", []))
    if history.empty or not {"price_x", "price_y"}.issubset(history.columns):
        raise ValueError("local_history_missing_two_leg_prices")
    required_history_columns = {"open_x", "price_x", "open_y", "price_y"}
    if not required_history_columns.issubset(history.columns):
        raise ValueError("local_history_missing_two_leg_open_close_prices")
    prices = (
        history[["open_x", "price_x", "open_y", "price_y"]]
        .apply(pd.to_numeric, errors="coerce")
        .dropna()
    )
    prices = _orient_history_prices(prices, payload=payload, candidate=candidate)
    observations = _requested_observations(candidate)
    prices = prices[(prices > 0).all(axis=1)].tail(observations)
    if len(prices) < 50:
        raise ValueError("local_history_has_insufficient_valid_prices")

    required = (
        "entry_level",
        "exit_level",
        "x_weighting",
        "slippage_rate",
        "commission_rate",
        "roll_w",
    )
    settings: dict[str, float] = {}
    for key in required:
        value = _float(candidate.get(key, ""))
        if value is None:
            raise ValueError(f"captured_setting_missing:{key}")
        settings[key] = value
    stop_loss = _float(candidate.get("stop_loss_rate", ""))
    exit_n_periods = _int_or_none(candidate.get("exit_n_periods", ""))
    stop_loss = stop_loss if stop_loss is not None and stop_loss > 0.0 else None
    exit_n_periods = exit_n_periods if exit_n_periods is not None and exit_n_periods > 0 else None
    strategy = _text(candidate.get("vendor_request_strategy", ""))
    if not strategy:
        raise ValueError("vendor_request_strategy_missing")
    spread_type = _text(candidate.get("vendor_request_spread_type", "")) or None
    return CryptoWizardsCustomSeriesBacktestRequest(
        series_1_opens=tuple(float(value) for value in prices["open_x"].tolist()),
        series_1_closes=tuple(float(value) for value in prices["price_x"].tolist()),
        series_2_opens=tuple(float(value) for value in prices["open_y"].tolist()),
        series_2_closes=tuple(float(value) for value in prices["price_y"].tolist()),
        strategy=strategy,
        spread_type=spread_type,
        roll_w=int(settings["roll_w"]),
        entry_level=settings["entry_level"],
        exit_level=settings["exit_level"],
        x_weighting=settings["x_weighting"],
        slippage_rate=settings["slippage_rate"],
        commission_rate=settings["commission_rate"],
        stop_loss_rate=stop_loss,
        exit_n_periods=exit_n_periods,
        with_history=True,
    )


def _response_metrics(response: dict[str, Any]) -> dict[str, object]:
    data = response.get("data") if isinstance(response.get("data"), dict) else {}
    returns = data.get("strat_returns") if isinstance(data.get("strat_returns"), dict) else {}
    history = response.get("history")
    history_available = bool(history) and isinstance(history, (dict, list))
    history_dict = history if isinstance(history, dict) else {}
    spread_stats = (
        history_dict.get("spread_stats")
        if isinstance(history_dict.get("spread_stats"), dict)
        else {}
    )
    coint = history_dict.get("coint_eg") if isinstance(history_dict.get("coint_eg"), dict) else {}
    last_zscore = (
        spread_stats.get("last_zscore") if isinstance(spread_stats.get("last_zscore"), dict) else {}
    )
    copula_names = _response_values(response, "copula_name")
    u1_values = _numeric_response_values(response, "u1_given_u2")
    u2_values = _numeric_response_values(response, "u2_given_u1")
    return {
        "vendor_sharpe": data.get("sharpe_ratio", ""),
        "vendor_sortino": data.get("sortino_ratio", ""),
        "vendor_total_return": returns.get("total_return", ""),
        "vendor_annual_return": returns.get("annual_return", ""),
        "vendor_max_drawdown": data.get("max_drawdown", ""),
        "vendor_win_rate": data.get("win_rate", ""),
        "vendor_var": data.get("var", ""),
        "vendor_cvar": data.get("cvar", ""),
        "vendor_half_life": spread_stats.get("half_life", ""),
        "vendor_hurst": spread_stats.get("hurst", ""),
        "vendor_hedge_ratio": spread_stats.get("hedge_ratio", ""),
        "vendor_last_zscore": last_zscore.get("zscore", ""),
        "vendor_last_zscore_roll": last_zscore.get("zscore_roll", ""),
        "vendor_sigma0_crossings": spread_stats.get("sigma0crossings", ""),
        "vendor_sigma2_crossings": spread_stats.get("sigma2crossings", ""),
        "vendor_engle_granger_cointegrated": coint.get("is_coint", ""),
        "vendor_engle_granger_pvalue": coint.get("p_value", ""),
        "vendor_copula_name": _text(copula_names[-1]) if copula_names else "",
        "vendor_u1_given_u2": u1_values[-1] if u1_values else "",
        "vendor_u2_given_u1": u2_values[-1] if u2_values else "",
        "vendor_copula_history_points": max(len(u1_values), len(u2_values)),
        "vendor_history_available": history_available,
    }


def _response_values(payload: object, key: str) -> list[object]:
    """Collect response observables without assuming a vendor nesting shape."""

    values: list[object] = []
    if isinstance(payload, dict):
        for candidate_key, value in payload.items():
            if candidate_key == key:
                if isinstance(value, list):
                    values.extend(value)
                else:
                    values.append(value)
            values.extend(_response_values(value, key))
    elif isinstance(payload, list):
        for value in payload:
            values.extend(_response_values(value, key))
    return values


def _numeric_response_values(payload: object, key: str) -> list[float]:
    values: list[float] = []
    for value in _response_values(payload, key):
        number = _float(value)
        if number is not None:
            values.append(number)
    return values


# Frozen comparator implementations are source-hash evidence. Keep formatter
# directives outside the function boundaries consumed by inspect.getsource.
# fmt: off
def _formula_parity_metrics(request: dict[str, Any], response: dict[str, Any]) -> dict[str, object]:
    unavailable = {
        "vendor_formula_parity_status": "not_available",
        "vendor_spread_formula": "",
        "vendor_spread_max_abs_error": "",
        "vendor_zscore_formula": "",
        "vendor_zscore_max_abs_error": "",
        "vendor_zscore_roll_formula": "",
        "vendor_zscore_roll_max_abs_error": "",
        "vendor_formula_pit_status": "historical_vendor_response_only",
    }
    params = request.get("params") if isinstance(request.get("params"), dict) else {}
    history = response.get("history") if isinstance(response.get("history"), dict) else {}
    stats = history.get("spread_stats") if isinstance(history.get("spread_stats"), dict) else {}
    strategy = params.get("strategy")
    spread_type = params.get("spread_type")
    if strategy not in {"Spread", "ZScoreRoll"}:
        return unavailable
    try:
        x = np.asarray(params["series_1_closes"], dtype=float)
        y = np.asarray(params["series_2_closes"], dtype=float)
        window = int(params["roll_w"])
    except (KeyError, TypeError, ValueError):
        return unavailable
    if len(x) < 2 or len(x) != len(y):
        return unavailable
    log_used = bool(stats.get("log_used", False))
    transformed_x = np.log(x) if log_used else x
    transformed_y = np.log(y) if log_used else y
    if spread_type == "Static":
        spread = _static_spread_candidate(transformed_x, transformed_y)
        spread_formula = (
            "ols_log_y_on_log_x_with_intercept: log(y)-(alpha+beta*log(x))"
            if log_used
            else "ols_y_on_x_with_intercept: y-(alpha+beta*x)"
        )
    elif spread_type == "Dynamic":
        spread = _kalman_dynamic_spread_candidate(transformed_x, transformed_y)
        spread_formula = (
            "candidate_v1:kalman_random_walk_beta_alpha_delta_1e-5_"
            "observation_variance_1_preupdate_innovation"
        )
    elif str(spread_type).lower() == "ou":
        spread = _ou_profile_mle_spread_candidate(transformed_x, transformed_y)
        spread_formula = (
            "candidate_v1:ou_profile_mle_beta_from_gaussian_ar1_"
            "centered_at_implied_long_run_mean"
        )
    else:
        return unavailable
    return _compare_spread_candidate(
        spread=spread,
        stats=stats,
        window=window,
        spread_formula=spread_formula,
    )


def _formula_parity_metrics_for_generation(
    request: dict[str, Any],
    response: dict[str, Any],
    *,
    dynamic_generation: int,
) -> dict[str, object]:
    """Apply the immutable v1 comparator or the reviewed v2 overlay."""

    if dynamic_generation == 1:
        return _formula_parity_metrics(request, response)
    if dynamic_generation != 2:
        raise ValueError(f"unsupported dynamic comparator generation: {dynamic_generation}")
    params = request.get("params") if isinstance(request.get("params"), dict) else {}
    if params.get("spread_type") != "Dynamic":
        return _formula_parity_metrics(request, response)
    if params.get("strategy") not in {"Spread", "ZScoreRoll"}:
        return _unavailable_formula_metrics()
    history = response.get("history") if isinstance(response.get("history"), dict) else {}
    stats = history.get("spread_stats") if isinstance(history.get("spread_stats"), dict) else {}
    try:
        x = np.asarray(params["series_1_closes"], dtype=float)
        y = np.asarray(params["series_2_closes"], dtype=float)
        window = int(params["roll_w"])
    except (KeyError, TypeError, ValueError):
        return _unavailable_formula_metrics()
    if len(x) < 2 or len(x) != len(y):
        return _unavailable_formula_metrics()
    log_used = bool(stats.get("log_used", False))
    if log_used and (np.any(x <= 0.0) or np.any(y <= 0.0)):
        return _unavailable_formula_metrics()
    transformed_x = np.log(x) if log_used else x
    transformed_y = np.log(y) if log_used else y
    spread = _kalman_dynamic_spread_candidate_v2_holdout(transformed_x, transformed_y)
    return _compare_spread_candidate(
        spread=spread,
        stats=stats,
        window=window,
        spread_formula=(
            "candidate_v2:kalman_random_walk_beta_alpha_transition_1e-5_"
            "observation_variance_1_postupdate_residual_warmup30_zero"
        ),
    )


def _formula_parity_metrics_for_mode_generation(
    request: dict[str, Any],
    response: dict[str, Any],
    *,
    dynamic_generation: int,
    ou_generation: int,
    ou_profile_intercept_threshold: float | None = None,
) -> dict[str, object]:
    """Apply independently reviewed Dynamic and OU comparator overlays."""

    params = request.get("params") if isinstance(request.get("params"), dict) else {}
    spread_type = str(params.get("spread_type", "")).lower()
    if spread_type == "dynamic":
        return _formula_parity_metrics_for_generation(
            request,
            response,
            dynamic_generation=dynamic_generation,
        )
    if spread_type != "ou" or ou_generation == 1:
        return _formula_parity_metrics(request, response)
    if ou_generation not in {3, 4, 5, 6}:
        raise ValueError(f"unsupported OU comparator generation: {ou_generation}")
    if params.get("strategy") not in {"Spread", "ZScoreRoll"}:
        return _unavailable_formula_metrics()
    history = response.get("history") if isinstance(response.get("history"), dict) else {}
    stats = history.get("spread_stats") if isinstance(history.get("spread_stats"), dict) else {}
    coint = history.get("coint_eg") if isinstance(history.get("coint_eg"), dict) else {}
    if "inc_trend" not in coint:
        return _unavailable_formula_metrics()
    try:
        x = np.asarray(params["series_1_closes"], dtype=float)
        y = np.asarray(params["series_2_closes"], dtype=float)
        window = int(params["roll_w"])
    except (KeyError, TypeError, ValueError):
        return _unavailable_formula_metrics()
    if len(x) < 50 or len(x) != len(y):
        return _unavailable_formula_metrics()

    vendor_inc_trend = _truthy(coint.get("inc_trend"))
    vendor_log_used = bool(stats.get("log_used", False))
    if vendor_log_used and (np.any(x <= 0.0) or np.any(y <= 0.0)):
        return _unavailable_formula_metrics()
    if ou_generation == 6:
        from quant_platform.orchestration.corrective_wizard_ou_v4_holdout import (
            _ou_spread_candidate_v4,
        )
        from quant_platform.orchestration.corrective_wizard_ou_v6_holdout import (
            _ou_v6_predictor,
        )

        selector = _ou_v6_predictor(x, y)
        local_log_used = bool(selector["predicted_log_used"])
        local_inc_trend = bool(selector["predicted_inc_trend"])
        local_profile_branch = _text(selector["predicted_profile_branch"])
        transformed_x = np.log(x) if local_log_used else x
        transformed_y = np.log(y) if local_log_used else y
        _, spread, _ = _ou_spread_candidate_v4(
            transformed_x,
            transformed_y,
            inc_trend=local_profile_branch == "zero_mean",
        )
        spread_formula = (
            "candidate_v6:normalized_ou_profile_branch_selected_by_frozen_"
            "request_only_transform_trend_diagnostic_and_profile_rules"
        )
    elif ou_generation == 5:
        from quant_platform.orchestration.corrective_wizard_ou_v4_holdout import (
            _ou_spread_candidate_v4,
        )
        from quant_platform.orchestration.corrective_wizard_ou_v5_holdout import (
            _ou_v5_predictor,
        )

        if ou_profile_intercept_threshold is None:
            raise ValueError("OU v5 profile intercept threshold is unavailable")
        selector = _ou_v5_predictor(
            x,
            y,
            profile_intercept_threshold=ou_profile_intercept_threshold,
        )
        local_log_used = bool(selector["predicted_log_used"])
        local_inc_trend = bool(selector["predicted_inc_trend"])
        local_profile_branch = _text(selector["predicted_profile_branch"])
        transformed_x = np.log(x) if local_log_used else x
        transformed_y = np.log(y) if local_log_used else y
        _, spread, _ = _ou_spread_candidate_v4(
            transformed_x,
            transformed_y,
            inc_trend=local_profile_branch == "zero_mean",
        )
        spread_formula = (
            "candidate_v5:normalized_ou_profile_branch_selected_by_frozen_"
            "request_only_scale_and_ar1_intercept_rules"
        )
    elif ou_generation == 4:
        from quant_platform.orchestration.corrective_wizard_ou_v4_holdout import (
            _ou_local_transform_trend_selector_v2,
            _ou_spread_candidate_v4,
        )

        selector = _ou_local_transform_trend_selector_v2(x, y)
        local_log_used = bool(selector["local_predicted_log_used"])
        local_inc_trend = bool(selector["local_predicted_inc_trend"])
        transformed_x = np.log(x) if local_log_used else x
        transformed_y = np.log(y) if local_log_used else y
        _, spread, _ = _ou_spread_candidate_v4(
            transformed_x,
            transformed_y,
            inc_trend=local_inc_trend,
        )
        spread_formula = (
            "candidate_v4:normalized_ou_profile_branch_selected_by_"
            "point_in_time_level_log_constant_trend_adf_aic"
        )
    else:
        from quant_platform.orchestration.corrective_wizard_ou_holdout import (
            _ou_local_trend_selector_v1,
        )

        local_log_used = vendor_log_used
        local_inc_trend = bool(_ou_local_trend_selector_v1(x, y)["local_predicted_inc_trend"])
        transformed_x = np.log(x) if local_log_used else x
        transformed_y = np.log(y) if local_log_used else y
        spread = _ou_trend_aware_profile_spread_v3_candidate(
            transformed_x,
            transformed_y,
            inc_trend=local_inc_trend,
        )
        spread_formula = (
            "candidate_v3:normalized_ou_profile_branch_selected_by_point_in_time_adf_aic"
        )
    metrics = _compare_spread_candidate(
        spread=spread,
        stats=stats,
        window=window,
        spread_formula=spread_formula,
    )
    if local_log_used != vendor_log_used:
        metrics["vendor_formula_parity_status"] = "transform_selector_mismatch"
    elif local_inc_trend != vendor_inc_trend:
        metrics["vendor_formula_parity_status"] = "trend_selector_mismatch"
    metrics["vendor_formula_pit_status"] = (
        "registered_terminal_request_only_transform_trend_diagnostic_and_"
        "profile_branch_research_only"
        if ou_generation == 6
        else "registered_local_scale_transform_aic_diagnostic_and_profile_branch_research_only"
        if ou_generation == 5
        else "registered_local_level_log_and_aic_branch_research_only"
        if ou_generation == 4
        else "registered_local_aic_branch_research_only"
    )
    if ou_generation in {5, 6}:
        metrics["vendor_formula_profile_branch"] = local_profile_branch
    return metrics


def build_comparator_formula_registration(exact_mode: str) -> dict[str, object]:
    """Return the machine-readable and source-bound comparator preregistration."""

    common = {
        "input_transform": "vendor_history_spread_stats.log_used_selects_log_or_level",
        "orientation": "series_1_is_x_series_2_is_y_without_implicit_reordering",
        "zscore": "full_sample_mean_and_sample_std_ddof_1",
        "rolling_zscore": "rolling_window_mean_and_sample_std_ddof_1_warmup_zero",
        "parity_tolerance_max_abs": 1e-9,
        "missing_value_policy": "fail_closed",
    }
    if exact_mode.startswith("Static"):
        family = "static"
        formula = {
            "estimator": "numpy_polyfit_degree_1_with_intercept",
            "spread": "y-(alpha+beta*x)",
            "fit_scope": "entire_submitted_vendor_history",
        }
        functions = (
            _formula_parity_metrics,
            _static_spread_candidate,
            _compare_spread_candidate,
        )
    elif exact_mode.startswith("Dyn"):
        family = "dynamic"
        formula = {
            "estimator": "two_state_random_walk_kalman_beta_alpha",
            "initial_state": [0.0, 0.0],
            "initial_covariance": "identity_2",
            "delta": 1e-5,
            "transition_covariance": "delta/(1-delta)*identity_2",
            "observation_variance": 1.0,
            "spread": "pre_update_innovation",
        }
        functions = (
            _formula_parity_metrics,
            _kalman_dynamic_spread_candidate,
            _compare_spread_candidate,
        )
    elif exact_mode.startswith("OU"):
        family = "ou"
        formula = {
            "estimator": "bounded_profile_gaussian_ar1_mle_beta",
            "initial_beta": "numpy_polyfit_degree_1_slope",
            "search_width": "max(abs(initial_beta)*4,1)",
            "bounds": "initial_beta_plus_or_minus_search_width",
            "optimizer": "scipy_minimize_scalar_bounded",
            "optimizer_xatol": 1e-12,
            "optimizer_maxiter": 1000,
            "valid_phi": "0<phi<1",
            "spread": "y-beta*x-centered_at_implied_long_run_mean",
        }
        functions = (
            _formula_parity_metrics,
            _ou_profile_mle_spread_candidate,
            _compare_spread_candidate,
        )
    else:
        family = "copula"
        formula = {
            "estimator": "unimplemented",
            "reason": "vendor_historical_copula_calibration_and_signal_series_not_published",
        }
        functions = (_formula_parity_metrics, _unavailable_formula_metrics)
    specification = {
        "schema_version": "thewiz.wizard_comparator_formula_registration.v1",
        "exact_mode": exact_mode,
        "family": family,
        "common": common,
        "formula": formula,
    }
    source = "\n\n".join(inspect.getsource(function) for function in functions)
    runtime = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": version("scipy"),
    }
    specification_json = json.dumps(specification, sort_keys=True, separators=(",", ":"))
    source_sha256 = sha256(source.encode("utf-8")).hexdigest()
    implementation_payload = {
        "specification": specification,
        "runtime": runtime,
        "source_sha256": source_sha256,
    }
    implementation_sha256 = sha256(
        json.dumps(implementation_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        **implementation_payload,
        "specification_json": specification_json,
        "specification_sha256": sha256(specification_json.encode("utf-8")).hexdigest(),
        "source": source,
        "implementation_sha256": implementation_sha256,
    }


def _static_spread_candidate(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    beta, alpha = np.polyfit(x, y, 1)
    return y - (alpha + beta * x)


def _compare_spread_candidate(
    *,
    spread: np.ndarray,
    stats: dict[str, Any],
    window: int,
    spread_formula: str,
) -> dict[str, object]:
    try:
        vendor_spread = np.asarray(stats["spread"], dtype=float)
        vendor_zscore = np.asarray(stats["zscore"], dtype=float)
        vendor_zscore_roll = np.asarray(stats["zscore_roll"], dtype=float)
    except (KeyError, TypeError, ValueError):
        return _unavailable_formula_metrics()
    if len(spread) < 2 or len(
        {len(spread), len(vendor_spread), len(vendor_zscore), len(vendor_zscore_roll)}
    ) != 1:
        return _unavailable_formula_metrics()
    spread_std = spread.std(ddof=1)
    if not np.isfinite(spread_std) or spread_std <= 0.0 or window < 2:
        return _unavailable_formula_metrics()
    zscore = (spread - spread.mean()) / spread_std
    spread_series = pd.Series(spread)
    rolling_mean = spread_series.rolling(window, min_periods=window).mean()
    rolling_std = spread_series.rolling(window, min_periods=window).std(ddof=1)
    zscore_roll = spread_series.sub(rolling_mean).div(rolling_std.where(rolling_std.abs() > 1e-12)).fillna(0.0)
    spread_error = float(np.max(np.abs(spread - vendor_spread)))
    zscore_error = float(np.max(np.abs(zscore - vendor_zscore)))
    zscore_roll_error = float(np.max(np.abs(zscore_roll.to_numpy() - vendor_zscore_roll)))
    tolerance = 1e-9
    parity = "exact_reconstruction" if max(spread_error, zscore_error, zscore_roll_error) <= tolerance else "mismatch"
    return {
        "vendor_formula_parity_status": parity,
        "vendor_spread_formula": spread_formula,
        "vendor_spread_max_abs_error": spread_error,
        "vendor_zscore_formula": "full_sample_sample_std_ddof1",
        "vendor_zscore_max_abs_error": zscore_error,
        "vendor_zscore_roll_formula": f"rolling_{window}_sample_std_ddof1_warmup_zero",
        "vendor_zscore_roll_max_abs_error": zscore_roll_error,
        "vendor_formula_pit_status": "hindsight_full_sample_fit_not_live_signal_safe",
    }


def _unavailable_formula_metrics() -> dict[str, object]:
    return {
        "vendor_formula_parity_status": "not_available",
        "vendor_spread_formula": "",
        "vendor_spread_max_abs_error": "",
        "vendor_zscore_formula": "",
        "vendor_zscore_max_abs_error": "",
        "vendor_zscore_roll_formula": "",
        "vendor_zscore_roll_max_abs_error": "",
        "vendor_formula_pit_status": "historical_vendor_response_only",
    }


def _kalman_dynamic_spread_candidate(
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Prospectively frozen Kalman candidate; not vendor authority until parity."""

    state = np.zeros(2, dtype=float)
    covariance = np.eye(2, dtype=float)
    delta = 1e-5
    transition_covariance = delta / (1.0 - delta) * np.eye(2)
    observation_variance = 1.0
    innovations = np.empty(len(x), dtype=float)
    identity = np.eye(2)
    for index, (x_value, y_value) in enumerate(zip(x, y, strict=True)):
        predicted_covariance = covariance + transition_covariance
        observation = np.asarray([x_value, 1.0], dtype=float)
        innovation = float(y_value - observation @ state)
        innovation_variance = float(
            observation @ predicted_covariance @ observation
            + observation_variance
        )
        gain = predicted_covariance @ observation / innovation_variance
        state = state + gain * innovation
        covariance = (identity - np.outer(gain, observation)) @ predicted_covariance
        innovations[index] = innovation
    return innovations


def _kalman_dynamic_spread_candidate_v2_holdout(
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Observed-data candidate reserved for validation on a disjoint pair."""

    state = np.zeros(2, dtype=float)
    covariance = np.eye(2, dtype=float)
    transition_covariance = 1e-5 * np.eye(2)
    observation_variance = 1.0
    residuals = np.empty(len(x), dtype=float)
    identity = np.eye(2)
    for index, (x_value, y_value) in enumerate(zip(x, y, strict=True)):
        predicted_covariance = covariance + transition_covariance
        observation = np.asarray([x_value, 1.0], dtype=float)
        innovation = float(y_value - observation @ state)
        innovation_variance = float(
            observation @ predicted_covariance @ observation
            + observation_variance
        )
        gain = predicted_covariance @ observation / innovation_variance
        state = state + gain * innovation
        covariance = (
            identity - np.outer(gain, observation)
        ) @ predicted_covariance
        residuals[index] = float(y_value - observation @ state)
    residuals[: min(30, len(residuals))] = 0.0
    return residuals


def _ou_profile_mle_spread_candidate(
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Prospectively frozen OU profile-likelihood candidate."""

    static_beta, _ = np.polyfit(x, y, 1)
    width = max(abs(float(static_beta)) * 4.0, 1.0)

    def objective(beta: float) -> float:
        spread = y - beta * x
        lagged = spread[:-1]
        current = spread[1:]
        design = np.column_stack([np.ones(len(lagged)), lagged])
        intercept, phi = np.linalg.lstsq(design, current, rcond=None)[0]
        residual = current - (intercept + phi * lagged)
        variance = float(np.mean(np.square(residual)))
        if not np.isfinite(variance) or variance <= 1e-18 or not 0.0 < phi < 1.0:
            return 1e100
        return 0.5 * len(residual) * math.log(variance)

    result = minimize_scalar(
        objective,
        bounds=(float(static_beta) - width, float(static_beta) + width),
        method="bounded",
        options={"xatol": 1e-12, "maxiter": 1000},
    )
    beta = float(result.x) if result.success and np.isfinite(result.fun) else float(static_beta)
    spread = y - beta * x
    lagged = spread[:-1]
    current = spread[1:]
    design = np.column_stack([np.ones(len(lagged)), lagged])
    intercept, phi = np.linalg.lstsq(design, current, rcond=None)[0]
    long_run_mean = float(intercept / (1.0 - phi)) if 0.0 < phi < 1.0 else float(spread.mean())
    return spread - long_run_mean


def _ou_zero_mean_profile_beta_v2_holdout(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    """Fit the blinded OU-v2 hedge ratio on first-price-normalized legs.

    Crypto Wizards selects the hedge ratio by minimizing the one-step error of
    a zero-mean AR(1) spread, then estimates the displayed OU mean-reversion
    statistics separately with an intercept.  The coarse scan makes the local
    scalar refinement deterministic and protects it from a poor initial slope.
    """

    if len(x) < 3 or len(x) != len(y):
        raise ValueError("OU v2 requires equal price series with at least three rows")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("OU v2 requires finite price series")
    if abs(float(x[0])) <= 1e-18 or abs(float(y[0])) <= 1e-18:
        raise ValueError("OU v2 cannot normalize a zero initial price")

    normalized_x = np.asarray(x, dtype=float) / float(x[0])
    normalized_y = np.asarray(y, dtype=float) / float(y[0])
    static_beta, _ = np.polyfit(normalized_x, normalized_y, 1)
    width = max(abs(float(static_beta)) * 4.0, 2.0)
    center = float(static_beta)

    def objective(beta: float) -> float:
        spread = normalized_y - beta * normalized_x
        lagged = spread[:-1]
        current = spread[1:]
        denominator = float(np.dot(lagged, lagged))
        if not np.isfinite(denominator) or denominator <= 1e-18:
            return math.inf
        phi = float(np.dot(lagged, current) / denominator)
        residual = current - phi * lagged
        value = float(np.mean(np.square(residual)))
        return value if np.isfinite(value) else math.inf

    for _ in range(6):
        grid = np.linspace(center - width, center + width, 513)
        values = np.asarray([objective(float(beta)) for beta in grid])
        best = int(np.argmin(values))
        if 0 < best < len(grid) - 1:
            break
        center = float(grid[best])
        width *= 2.0
    else:
        raise ValueError("OU v2 hedge-ratio search did not find an interior minimum")

    result = minimize_scalar(
        objective,
        bounds=(float(grid[best - 1]), float(grid[best + 1])),
        method="bounded",
        options={"xatol": 1e-14, "maxiter": 1000},
    )
    if not result.success or not np.isfinite(result.fun):
        raise ValueError("OU v2 hedge-ratio refinement failed")
    return float(result.x)


def _ou_zero_mean_profile_spread_candidate_v2_holdout(
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Return the blinded OU-v2 normalized spread without hindsight centering."""

    beta = _ou_zero_mean_profile_beta_v2_holdout(x, y)
    normalized_x = np.asarray(x, dtype=float) / float(x[0])
    normalized_y = np.asarray(y, dtype=float) / float(y[0])
    return normalized_y - beta * normalized_x


def _ou_trend_aware_profile_beta_v3_candidate(
    x: np.ndarray,
    y: np.ndarray,
    *,
    inc_trend: bool,
) -> float:
    """Fit the OU-v3 hedge ratio using the vendor cointegration trend branch."""

    if inc_trend:
        return _ou_zero_mean_profile_beta_v2_holdout(x, y)
    if len(x) < 3 or len(x) != len(y):
        raise ValueError("OU v3 requires equal price series with at least three rows")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("OU v3 requires finite price series")
    if abs(float(x[0])) <= 1e-18 or abs(float(y[0])) <= 1e-18:
        raise ValueError("OU v3 cannot normalize a zero initial price")

    normalized_x = np.asarray(x, dtype=float) / float(x[0])
    normalized_y = np.asarray(y, dtype=float) / float(y[0])
    static_beta, _ = np.polyfit(normalized_x, normalized_y, 1)
    width = max(abs(float(static_beta)) * 4.0, 2.0)
    center = float(static_beta)

    def objective(beta: float) -> float:
        spread = normalized_y - beta * normalized_x
        lagged = spread[:-1]
        current = spread[1:]
        design = np.column_stack([np.ones(len(lagged)), lagged])
        intercept, phi = np.linalg.lstsq(design, current, rcond=None)[0]
        residual = current - (intercept + phi * lagged)
        value = float(np.mean(np.square(residual)))
        return value if np.isfinite(value) else math.inf

    for _ in range(6):
        grid = np.linspace(center - width, center + width, 513)
        values = np.asarray([objective(float(beta)) for beta in grid])
        best = int(np.argmin(values))
        if 0 < best < len(grid) - 1:
            break
        center = float(grid[best])
        width *= 2.0
    else:
        raise ValueError("OU v3 hedge-ratio search did not find an interior minimum")

    result = minimize_scalar(
        objective,
        bounds=(float(grid[best - 1]), float(grid[best + 1])),
        method="bounded",
        options={"xatol": 1e-14, "maxiter": 1000},
    )
    if not result.success or not np.isfinite(result.fun):
        raise ValueError("OU v3 hedge-ratio refinement failed")
    return float(result.x)


def _ou_trend_aware_profile_spread_v3_candidate(
    x: np.ndarray,
    y: np.ndarray,
    *,
    inc_trend: bool,
) -> np.ndarray:
    """Return the OU-v3 normalized spread selected by trend provenance."""

    beta = _ou_trend_aware_profile_beta_v3_candidate(
        x, y, inc_trend=inc_trend
    )
    normalized_x = np.asarray(x, dtype=float) / float(x[0])
    normalized_y = np.asarray(y, dtype=float) / float(y[0])
    return normalized_y - beta * normalized_x
# fmt: on


def _proof_identity(row: pd.Series) -> tuple[str, str, str, int]:
    return (
        _text(row.get("pair_group_id", "")).lower() or _text(row.get("pair", "")).upper(),
        _text(row.get("local_interval", "")).lower(),
        f"{_text(row.get('exact_mode', '')).lower()}::{_text(row.get('orientation', '')).lower()}",
        _identity_observations(row),
    )


def _formula_proof_mask(frame: pd.DataFrame) -> pd.Series:
    explicit = (
        frame["formula_proof_complete"].map(_truthy)
        if "formula_proof_complete" in frame.columns
        else pd.Series(True, index=frame.index)
    )
    reconstructed = (
        frame.get("mode_proof_status", pd.Series("", index=frame.index)).eq("completed")
        & frame.get("vendor_formula_parity_status", pd.Series("", index=frame.index)).eq(
            "exact_reconstruction"
        )
        & frame.get("vendor_history_available", pd.Series(False, index=frame.index)).map(_truthy)
        & frame.get("proof_window_kind", pd.Series("", index=frame.index)).eq(
            "scanner_horizon_parity"
        )
    )
    return explicit & reconstructed


def _dynamic_v2_source_hash() -> str:
    return sha256(
        inspect.getsource(_kalman_dynamic_spread_candidate_v2_holdout).encode("utf-8")
    ).hexdigest()


def _active_dynamic_comparator_context(root: Path) -> dict[str, object]:
    """Resolve the reviewed comparator generation without silent fallback."""

    from quant_platform.wizard_dynamic_comparator_activation import (
        load_validated_dynamic_v2_activation,
    )

    activation = load_validated_dynamic_v2_activation(
        root=root,
        implementation_source_sha256=_dynamic_v2_source_hash(),
    )
    if activation is None:
        return {
            "generation": 1,
            "activation_id": "",
            "activation_path": "",
            "activation_sha256": "",
            "source_proof_sha256": "",
        }
    return {
        "generation": 2,
        "activation_id": _text(activation.get("activation_id")),
        "activation_path": _text(activation.get("immutable_activation_path")),
        "activation_sha256": _text(activation.get("immutable_activation_sha256")),
        "source_proof_sha256": _text(activation.get("source_proof_sha256")),
    }


def _ou_v3_source_hashes() -> tuple[str, str]:
    from quant_platform.orchestration.corrective_wizard_ou_holdout import (
        _ou_trend_selector_source_hash,
        _v3_implementation_source_hash,
    )

    return _v3_implementation_source_hash(), _ou_trend_selector_source_hash()


def _active_ou_comparator_context(root: Path) -> dict[str, object]:
    """Resolve the reviewed OU comparator generation without silent fallback."""

    from quant_platform.wizard_ou_comparator_activation import (
        load_validated_ou_v3_activation,
    )
    from quant_platform.wizard_ou_v4_comparator_activation import (
        load_validated_ou_v4_activation,
    )
    from quant_platform.wizard_ou_v5_comparator_activation import (
        load_validated_ou_v5_activation,
    )
    from quant_platform.wizard_ou_v6_comparator_activation import (
        load_validated_ou_v6_activation,
    )

    v6_activation = load_validated_ou_v6_activation(root=root)
    if v6_activation is not None:
        return {
            "generation": 6,
            "activation_id": _text(v6_activation.get("activation_id")),
            "activation_path": _text(v6_activation.get("immutable_activation_path")),
            "activation_sha256": _text(v6_activation.get("immutable_activation_sha256")),
            "source_proof_sha256": _text(v6_activation.get("source_proof_sha256")),
            "profile_intercept_threshold": None,
        }

    v5_activation = load_validated_ou_v5_activation(root=root)
    if v5_activation is not None:
        return {
            "generation": 5,
            "activation_id": _text(v5_activation.get("activation_id")),
            "activation_path": _text(v5_activation.get("immutable_activation_path")),
            "activation_sha256": _text(v5_activation.get("immutable_activation_sha256")),
            "source_proof_sha256": _text(v5_activation.get("source_proof_sha256")),
            "profile_intercept_threshold": float(v5_activation["profile_intercept_threshold"]),
        }

    v4_activation = load_validated_ou_v4_activation(root=root)
    if v4_activation is not None:
        return {
            "generation": 4,
            "activation_id": _text(v4_activation.get("activation_id")),
            "activation_path": _text(v4_activation.get("immutable_activation_path")),
            "activation_sha256": _text(v4_activation.get("immutable_activation_sha256")),
            "source_proof_sha256": _text(v4_activation.get("source_proof_sha256")),
            "profile_intercept_threshold": None,
        }
    implementation_hash, selector_hash = _ou_v3_source_hashes()
    activation = load_validated_ou_v3_activation(
        root=root,
        implementation_source_sha256=implementation_hash,
        selector_source_sha256=selector_hash,
    )
    if activation is None:
        return {
            "generation": 1,
            "activation_id": "",
            "activation_path": "",
            "activation_sha256": "",
            "source_proof_sha256": "",
            "profile_intercept_threshold": None,
        }
    return {
        "generation": 3,
        "activation_id": _text(activation.get("activation_id")),
        "activation_path": _text(activation.get("immutable_activation_path")),
        "activation_sha256": _text(activation.get("immutable_activation_sha256")),
        "source_proof_sha256": _text(activation.get("source_proof_sha256")),
        "profile_intercept_threshold": None,
    }


def refresh_activated_dynamic_v2_proofs(
    *,
    root: Path = ROOT,
    proof_path: Path | None = None,
    now: datetime | None = None,
) -> CommandResult:
    """Re-evaluate captured Dynamic proofs after explicit reviewed activation."""

    timestamp = _as_utc(now or datetime.now(UTC))
    proof_path = proof_path or (
        root / "reports" / "active" / "hyperliquid_wizard_vendor_mode_proofs.csv"
    )
    status_path = root / "reports" / "active" / "wizard_dynamic_v2_proof_refresh_status.json"
    comparator = _active_dynamic_comparator_context(root)
    base: dict[str, object] = {
        "schema_version": "thewiz.wizard_dynamic_v2_proof_refresh.v1",
        "evaluated_at_utc": timestamp.isoformat(),
        "comparator_generation": int(comparator["generation"]),
        "activation_id": comparator["activation_id"],
        "activation_path": comparator["activation_path"],
        "activation_sha256": comparator["activation_sha256"],
        "proof_path": _relative(proof_path, root=root),
        "research_only": True,
        "candidate_promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
        "raw_vendor_evidence_mutated": False,
        "original_comparator_mutated": False,
    }
    if int(comparator["generation"]) != 2:
        payload = {
            **base,
            "status": "NOT_ACTIVE",
            "dynamic_rows": 0,
            "captured_dynamic_rows": 0,
            "exact_dynamic_rows": 0,
            "refreshed_dynamic_rows": 0,
            "blocker": "reviewed_dynamic_v2_activation_not_applied",
        }
        _write_json(status_path, payload)
        return CommandResult(paths={"refresh_status": status_path}, summary=payload)
    if not proof_path.is_file():
        raise ValueError("dynamic v2 source proof file missing")

    frame = _read_csv(proof_path)
    if frame.empty:
        raise ValueError("dynamic v2 source proof file is empty")
    before_sha256 = sha256(proof_path.read_bytes()).hexdigest()
    source_sha256 = _text(comparator.get("source_proof_sha256"))
    dynamic_mask = _dynamic_proof_mask(frame)
    captured_mask = dynamic_mask & _captured_proof_mask(frame)
    if before_sha256 != source_sha256 and captured_mask.any():
        bound = (
            frame.loc[captured_mask]
            .get(
                "formula_comparator_generation",
                pd.Series(0, index=frame.loc[captured_mask].index),
            )
            .map(_int_or_none)
            .eq(2)
            & frame.loc[captured_mask]
            .get(
                "formula_comparator_activation_id",
                pd.Series("", index=frame.loc[captured_mask].index),
            )
            .map(_text)
            .eq(_text(comparator["activation_id"]))
            & frame.loc[captured_mask]
            .get(
                "formula_comparator_activation_sha256",
                pd.Series("", index=frame.loc[captured_mask].index),
            )
            .map(_text)
            .eq(_text(comparator["activation_sha256"]))
        )
        if not bound.all():
            raise ValueError("dynamic v2 active proof hash changed without generation-2 lineage")

    raw_before = _dynamic_raw_evidence_hashes(frame.loc[captured_mask], root=root)
    refreshed = _refresh_existing_metrics(frame, root=root)
    raw_after = _dynamic_raw_evidence_hashes(refreshed.loc[captured_mask], root=root)
    if raw_before != raw_after:
        raise ValueError("dynamic v2 refresh changed raw vendor evidence")
    for column in PROOF_COLUMNS:
        if column not in refreshed.columns:
            refreshed[column] = ""
    refreshed = refreshed.reindex(columns=PROOF_COLUMNS)
    _atomic_write_csv(refreshed, proof_path)
    after_sha256 = sha256(proof_path.read_bytes()).hexdigest()
    dynamic_after = _dynamic_proof_mask(refreshed)
    captured_after = dynamic_after & _captured_proof_mask(refreshed)
    exact_after = captured_after & _formula_proof_mask(refreshed)
    payload = {
        **base,
        "status": "PASS",
        "source_proof_sha256": source_sha256,
        "proof_sha256_before": before_sha256,
        "proof_sha256_after": after_sha256,
        "dynamic_rows": int(dynamic_after.sum()),
        "captured_dynamic_rows": int(captured_after.sum()),
        "exact_dynamic_rows": int(exact_after.sum()),
        "refreshed_dynamic_rows": int(captured_after.sum()),
        "blocker": "",
    }
    _write_json(status_path, payload)
    return CommandResult(
        paths={"refresh_status": status_path, "proofs": proof_path},
        summary=payload,
    )


def refresh_activated_ou_v3_proofs(
    *,
    root: Path = ROOT,
    proof_path: Path | None = None,
    now: datetime | None = None,
) -> CommandResult:
    """Re-evaluate captured OU proofs after explicit reviewed activation."""

    timestamp = _as_utc(now or datetime.now(UTC))
    proof_path = proof_path or (
        root / "reports" / "active" / "hyperliquid_wizard_vendor_mode_proofs.csv"
    )
    status_path = root / "reports" / "active" / "wizard_ou_v3_proof_refresh_status.json"
    comparator = _active_ou_comparator_context(root)
    base: dict[str, object] = {
        "schema_version": "thewiz.wizard_ou_v3_proof_refresh.v1",
        "evaluated_at_utc": timestamp.isoformat(),
        "comparator_generation": int(comparator["generation"]),
        "activation_id": comparator["activation_id"],
        "activation_path": comparator["activation_path"],
        "activation_sha256": comparator["activation_sha256"],
        "proof_path": _relative(proof_path, root=root),
        "research_only": True,
        "candidate_promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
        "raw_vendor_evidence_mutated": False,
    }
    if int(comparator["generation"]) != 3:
        payload = {
            **base,
            "status": "NOT_ACTIVE",
            "ou_rows": 0,
            "captured_ou_rows": 0,
            "exact_ou_rows": 0,
            "refreshed_ou_rows": 0,
            "blocker": "reviewed_ou_v3_activation_not_applied",
        }
        _write_json(status_path, payload)
        return CommandResult(paths={"refresh_status": status_path}, summary=payload)
    if not proof_path.is_file():
        raise ValueError("OU v3 source proof file missing")
    frame = _read_csv(proof_path)
    if frame.empty:
        raise ValueError("OU v3 source proof file is empty")
    before_sha256 = sha256(proof_path.read_bytes()).hexdigest()
    source_sha256 = _text(comparator.get("source_proof_sha256"))
    ou_mask = _ou_proof_mask(frame)
    captured_mask = ou_mask & _captured_proof_mask(frame)
    if before_sha256 != source_sha256 and captured_mask.any():
        bound = (
            frame.loc[captured_mask]
            .get(
                "formula_comparator_generation",
                pd.Series(0, index=frame.loc[captured_mask].index),
            )
            .map(_int_or_none)
            .eq(3)
            & frame.loc[captured_mask]
            .get(
                "formula_comparator_activation_id",
                pd.Series("", index=frame.loc[captured_mask].index),
            )
            .map(_text)
            .eq(_text(comparator["activation_id"]))
            & frame.loc[captured_mask]
            .get(
                "formula_comparator_activation_sha256",
                pd.Series("", index=frame.loc[captured_mask].index),
            )
            .map(_text)
            .eq(_text(comparator["activation_sha256"]))
        )
        if not bound.all():
            raise ValueError("OU v3 active proof hash changed without generation-3 lineage")

    raw_before = _mode_raw_evidence_hashes(frame.loc[captured_mask], root=root)
    refreshed = _refresh_existing_metrics(frame, root=root)
    raw_after = _mode_raw_evidence_hashes(refreshed.loc[captured_mask], root=root)
    if raw_before != raw_after:
        raise ValueError("OU v3 refresh changed raw vendor evidence")
    for column in PROOF_COLUMNS:
        if column not in refreshed.columns:
            refreshed[column] = ""
    refreshed = refreshed.reindex(columns=PROOF_COLUMNS)
    _atomic_write_csv(refreshed, proof_path)
    after_sha256 = sha256(proof_path.read_bytes()).hexdigest()
    ou_after = _ou_proof_mask(refreshed)
    captured_after = ou_after & _captured_proof_mask(refreshed)
    exact_after = captured_after & _formula_proof_mask(refreshed)
    payload = {
        **base,
        "status": "PASS",
        "source_proof_sha256": source_sha256,
        "proof_sha256_before": before_sha256,
        "proof_sha256_after": after_sha256,
        "ou_rows": int(ou_after.sum()),
        "captured_ou_rows": int(captured_after.sum()),
        "exact_ou_rows": int(exact_after.sum()),
        "refreshed_ou_rows": int(captured_after.sum()),
        "blocker": "",
    }
    _write_json(status_path, payload)
    return CommandResult(
        paths={"refresh_status": status_path, "proofs": proof_path},
        summary=payload,
    )


def refresh_activated_ou_v4_proofs(
    *,
    root: Path = ROOT,
    proof_path: Path | None = None,
    now: datetime | None = None,
) -> CommandResult:
    """Re-evaluate captured OU proofs after explicit reviewed v4 activation."""

    timestamp = _as_utc(now or datetime.now(UTC))
    proof_path = proof_path or (
        root / "reports" / "active" / "hyperliquid_wizard_vendor_mode_proofs.csv"
    )
    status_path = root / "reports" / "active" / "wizard_ou_v4_proof_refresh_status.json"
    comparator = _active_ou_comparator_context(root)
    base: dict[str, object] = {
        "schema_version": "thewiz.wizard_ou_v4_proof_refresh.v1",
        "evaluated_at_utc": timestamp.isoformat(),
        "comparator_generation": int(comparator["generation"]),
        "activation_id": comparator["activation_id"],
        "activation_path": comparator["activation_path"],
        "activation_sha256": comparator["activation_sha256"],
        "proof_path": _relative(proof_path, root=root),
        "research_only": True,
        "candidate_promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
        "raw_vendor_evidence_mutated": False,
    }
    if int(comparator["generation"]) != 4:
        payload = {
            **base,
            "status": "NOT_ACTIVE",
            "ou_rows": 0,
            "captured_ou_rows": 0,
            "exact_ou_rows": 0,
            "refreshed_ou_rows": 0,
            "blocker": "reviewed_ou_v4_activation_not_applied",
        }
        _write_json(status_path, payload)
        return CommandResult(paths={"refresh_status": status_path}, summary=payload)
    if not proof_path.is_file():
        raise ValueError("OU v4 source proof file missing")
    frame = _read_csv(proof_path)
    if frame.empty:
        raise ValueError("OU v4 source proof file is empty")
    before_sha256 = sha256(proof_path.read_bytes()).hexdigest()
    source_sha256 = _text(comparator.get("source_proof_sha256"))
    ou_mask = _ou_proof_mask(frame)
    captured_mask = ou_mask & _captured_proof_mask(frame)
    if before_sha256 != source_sha256 and captured_mask.any():
        bound = (
            frame.loc[captured_mask]
            .get(
                "formula_comparator_generation",
                pd.Series(0, index=frame.loc[captured_mask].index),
            )
            .map(_int_or_none)
            .eq(4)
            & frame.loc[captured_mask]
            .get(
                "formula_comparator_activation_id",
                pd.Series("", index=frame.loc[captured_mask].index),
            )
            .map(_text)
            .eq(_text(comparator["activation_id"]))
            & frame.loc[captured_mask]
            .get(
                "formula_comparator_activation_sha256",
                pd.Series("", index=frame.loc[captured_mask].index),
            )
            .map(_text)
            .eq(_text(comparator["activation_sha256"]))
        )
        if not bound.all():
            raise ValueError("OU v4 active proof hash changed without generation-4 lineage")

    raw_before = _mode_raw_evidence_hashes(frame.loc[captured_mask], root=root)
    refreshed = _refresh_existing_metrics(frame, root=root)
    raw_after = _mode_raw_evidence_hashes(refreshed.loc[captured_mask], root=root)
    if raw_before != raw_after:
        raise ValueError("OU v4 refresh changed raw vendor evidence")
    for column in PROOF_COLUMNS:
        if column not in refreshed.columns:
            refreshed[column] = ""
    refreshed = refreshed.reindex(columns=PROOF_COLUMNS)
    _atomic_write_csv(refreshed, proof_path)
    after_sha256 = sha256(proof_path.read_bytes()).hexdigest()
    ou_after = _ou_proof_mask(refreshed)
    captured_after = ou_after & _captured_proof_mask(refreshed)
    exact_after = captured_after & _formula_proof_mask(refreshed)
    payload = {
        **base,
        "status": "PASS",
        "source_proof_sha256": source_sha256,
        "proof_sha256_before": before_sha256,
        "proof_sha256_after": after_sha256,
        "ou_rows": int(ou_after.sum()),
        "captured_ou_rows": int(captured_after.sum()),
        "exact_ou_rows": int(exact_after.sum()),
        "refreshed_ou_rows": int(captured_after.sum()),
        "blocker": "",
    }
    _write_json(status_path, payload)
    return CommandResult(
        paths={"refresh_status": status_path, "proofs": proof_path},
        summary=payload,
    )


def refresh_activated_ou_v5_proofs(
    *,
    root: Path = ROOT,
    proof_path: Path | None = None,
    now: datetime | None = None,
) -> CommandResult:
    """Re-evaluate captured OU proofs after explicit reviewed v5 activation."""

    timestamp = _as_utc(now or datetime.now(UTC))
    proof_path = proof_path or (
        root / "reports" / "active" / "hyperliquid_wizard_vendor_mode_proofs.csv"
    )
    status_path = root / "reports" / "active" / "wizard_ou_v5_proof_refresh_status.json"
    comparator = _active_ou_comparator_context(root)
    base: dict[str, object] = {
        "schema_version": "thewiz.wizard_ou_v5_proof_refresh.v1",
        "evaluated_at_utc": timestamp.isoformat(),
        "comparator_generation": int(comparator["generation"]),
        "activation_id": comparator["activation_id"],
        "activation_path": comparator["activation_path"],
        "activation_sha256": comparator["activation_sha256"],
        "profile_intercept_threshold": comparator.get("profile_intercept_threshold"),
        "proof_path": _relative(proof_path, root=root),
        "research_only": True,
        "candidate_promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
        "raw_vendor_evidence_mutated": False,
    }
    if int(comparator["generation"]) != 5:
        payload = {
            **base,
            "status": "NOT_ACTIVE",
            "ou_rows": 0,
            "captured_ou_rows": 0,
            "exact_ou_rows": 0,
            "refreshed_ou_rows": 0,
            "blocker": "reviewed_ou_v5_activation_not_applied",
        }
        _write_json(status_path, payload)
        return CommandResult(paths={"refresh_status": status_path}, summary=payload)
    if not proof_path.is_file():
        raise ValueError("OU v5 source proof file missing")
    frame = _read_csv(proof_path)
    if frame.empty:
        raise ValueError("OU v5 source proof file is empty")
    before_sha256 = sha256(proof_path.read_bytes()).hexdigest()
    source_sha256 = _text(comparator.get("source_proof_sha256"))
    ou_mask = _ou_proof_mask(frame)
    captured_mask = ou_mask & _captured_proof_mask(frame)
    if before_sha256 != source_sha256 and captured_mask.any():
        bound = (
            frame.loc[captured_mask]
            .get(
                "formula_comparator_generation",
                pd.Series(0, index=frame.loc[captured_mask].index),
            )
            .map(_int_or_none)
            .eq(5)
            & frame.loc[captured_mask]
            .get(
                "formula_comparator_activation_id",
                pd.Series("", index=frame.loc[captured_mask].index),
            )
            .map(_text)
            .eq(_text(comparator["activation_id"]))
            & frame.loc[captured_mask]
            .get(
                "formula_comparator_activation_sha256",
                pd.Series("", index=frame.loc[captured_mask].index),
            )
            .map(_text)
            .eq(_text(comparator["activation_sha256"]))
        )
        if not bound.all():
            raise ValueError("OU v5 active proof hash changed without generation-5 lineage")

    raw_before = _mode_raw_evidence_hashes(frame.loc[captured_mask], root=root)
    refreshed = _refresh_existing_metrics(frame, root=root)
    raw_after = _mode_raw_evidence_hashes(refreshed.loc[captured_mask], root=root)
    if raw_before != raw_after:
        raise ValueError("OU v5 refresh changed raw vendor evidence")
    for column in PROOF_COLUMNS:
        if column not in refreshed.columns:
            refreshed[column] = ""
    refreshed = refreshed.reindex(columns=PROOF_COLUMNS)
    _atomic_write_csv(refreshed, proof_path)
    after_sha256 = sha256(proof_path.read_bytes()).hexdigest()
    ou_after = _ou_proof_mask(refreshed)
    captured_after = ou_after & _captured_proof_mask(refreshed)
    exact_after = captured_after & _formula_proof_mask(refreshed)
    payload = {
        **base,
        "status": "PASS",
        "source_proof_sha256": source_sha256,
        "proof_sha256_before": before_sha256,
        "proof_sha256_after": after_sha256,
        "ou_rows": int(ou_after.sum()),
        "captured_ou_rows": int(captured_after.sum()),
        "exact_ou_rows": int(exact_after.sum()),
        "refreshed_ou_rows": int(captured_after.sum()),
        "blocker": "",
    }
    _write_json(status_path, payload)
    return CommandResult(
        paths={"refresh_status": status_path, "proofs": proof_path},
        summary=payload,
    )


def refresh_activated_ou_v6_proofs(
    *,
    root: Path = ROOT,
    proof_path: Path | None = None,
    now: datetime | None = None,
) -> CommandResult:
    """Re-evaluate captured OU proofs after explicit terminal v6 activation."""

    timestamp = _as_utc(now or datetime.now(UTC))
    proof_path = proof_path or (
        root / "reports" / "active" / "hyperliquid_wizard_vendor_mode_proofs.csv"
    )
    status_path = root / "reports" / "active" / "wizard_ou_v6_proof_refresh_status.json"
    comparator = _active_ou_comparator_context(root)
    base: dict[str, object] = {
        "schema_version": "thewiz.wizard_ou_v6_proof_refresh.v1",
        "evaluated_at_utc": timestamp.isoformat(),
        "comparator_generation": int(comparator["generation"]),
        "activation_id": comparator["activation_id"],
        "activation_path": comparator["activation_path"],
        "activation_sha256": comparator["activation_sha256"],
        "proof_path": _relative(proof_path, root=root),
        "terminal_successor_generation": True,
        "successor_after_failure_allowed": False,
        "research_only": True,
        "candidate_promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
        "raw_vendor_evidence_mutated": False,
    }
    if int(comparator["generation"]) != 6:
        payload = {
            **base,
            "status": "NOT_ACTIVE",
            "ou_rows": 0,
            "captured_ou_rows": 0,
            "exact_ou_rows": 0,
            "refreshed_ou_rows": 0,
            "blocker": "reviewed_ou_v6_activation_not_applied",
        }
        _write_json(status_path, payload)
        return CommandResult(paths={"refresh_status": status_path}, summary=payload)
    if not proof_path.is_file():
        raise ValueError("OU v6 source proof file missing")
    frame = _read_csv(proof_path)
    if frame.empty:
        raise ValueError("OU v6 source proof file is empty")
    before_sha256 = sha256(proof_path.read_bytes()).hexdigest()
    source_sha256 = _text(comparator.get("source_proof_sha256"))
    ou_mask = _ou_proof_mask(frame)
    captured_mask = ou_mask & _captured_proof_mask(frame)
    if before_sha256 != source_sha256 and captured_mask.any():
        bound = (
            frame.loc[captured_mask]
            .get(
                "formula_comparator_generation",
                pd.Series(0, index=frame.loc[captured_mask].index),
            )
            .map(_int_or_none)
            .eq(6)
            & frame.loc[captured_mask]
            .get(
                "formula_comparator_activation_id",
                pd.Series("", index=frame.loc[captured_mask].index),
            )
            .map(_text)
            .eq(_text(comparator["activation_id"]))
            & frame.loc[captured_mask]
            .get(
                "formula_comparator_activation_sha256",
                pd.Series("", index=frame.loc[captured_mask].index),
            )
            .map(_text)
            .eq(_text(comparator["activation_sha256"]))
        )
        if not bound.all():
            raise ValueError("OU v6 active proof hash changed without generation-6 lineage")

    raw_before = _mode_raw_evidence_hashes(frame.loc[captured_mask], root=root)
    refreshed = _refresh_existing_metrics(frame, root=root)
    raw_after = _mode_raw_evidence_hashes(refreshed.loc[captured_mask], root=root)
    if raw_before != raw_after:
        raise ValueError("OU v6 refresh changed raw vendor evidence")
    for column in PROOF_COLUMNS:
        if column not in refreshed.columns:
            refreshed[column] = ""
    refreshed = refreshed.reindex(columns=PROOF_COLUMNS)
    _atomic_write_csv(refreshed, proof_path)
    after_sha256 = sha256(proof_path.read_bytes()).hexdigest()
    ou_after = _ou_proof_mask(refreshed)
    captured_after = ou_after & _captured_proof_mask(refreshed)
    exact_after = captured_after & _formula_proof_mask(refreshed)
    payload = {
        **base,
        "status": "PASS",
        "source_proof_sha256": source_sha256,
        "proof_sha256_before": before_sha256,
        "proof_sha256_after": after_sha256,
        "ou_rows": int(ou_after.sum()),
        "captured_ou_rows": int(captured_after.sum()),
        "exact_ou_rows": int(exact_after.sum()),
        "refreshed_ou_rows": int(captured_after.sum()),
        "blocker": "",
    }
    _write_json(status_path, payload)
    return CommandResult(
        paths={"refresh_status": status_path, "proofs": proof_path},
        summary=payload,
    )


def _refresh_existing_metrics(frame: pd.DataFrame, *, root: Path) -> pd.DataFrame:
    refreshed = frame.copy().astype(object)
    dynamic_comparator = _active_dynamic_comparator_context(root)
    ou_comparator = _active_ou_comparator_context(root)
    for index, row in refreshed.iterrows():
        response_path = _resolve_path(_text(row.get("response_path", "")), root=root)
        if response_path is None or not response_path.exists():
            continue
        try:
            response = json.loads(response_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        request_path = _resolve_path(_text(row.get("request_path", "")), root=root)
        request: dict[str, Any] = {}
        if request_path is not None and request_path.exists():
            try:
                request = json.loads(request_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                request = {}
        exact_mode = _text(row.get("exact_mode"))
        if exact_mode.startswith("Dyn"):
            comparator = dynamic_comparator
        elif exact_mode.startswith("OU"):
            comparator = ou_comparator
        else:
            comparator = {
                "generation": 1,
                "activation_id": "",
                "activation_path": "",
                "activation_sha256": "",
            }
        generation = int(comparator["generation"])
        metrics = {
            **_response_metrics(response),
            **_formula_parity_metrics_for_mode_generation(
                request,
                response,
                dynamic_generation=(generation if exact_mode.startswith("Dyn") else 1),
                ou_generation=(generation if exact_mode.startswith("OU") else 1),
                ou_profile_intercept_threshold=(
                    float(comparator["profile_intercept_threshold"])
                    if exact_mode.startswith("OU")
                    and comparator.get("profile_intercept_threshold") is not None
                    else None
                ),
            ),
        }
        for field, value in metrics.items():
            refreshed.at[index, field] = value
        refreshed.at[index, "formula_comparator_generation"] = generation
        reviewed_generation = generation in {2, 3, 4, 5, 6}
        refreshed.at[index, "formula_comparator_activation_id"] = (
            comparator["activation_id"] if reviewed_generation else ""
        )
        refreshed.at[index, "formula_comparator_activation_path"] = (
            comparator["activation_path"] if reviewed_generation else ""
        )
        refreshed.at[index, "formula_comparator_activation_sha256"] = (
            comparator["activation_sha256"] if reviewed_generation else ""
        )
        refreshed.at[index, "vendor_response_captured"] = True
        refreshed.at[index, "formula_proof_complete"] = (
            metrics["vendor_formula_parity_status"] == "exact_reconstruction"
        )
        validity = _proof_validity_fields(refreshed.loc[index])
        for field, value in validity.items():
            refreshed.at[index, field] = value
    return refreshed


def _dynamic_proof_mask(frame: pd.DataFrame) -> pd.Series:
    return (
        frame.get("exact_mode", pd.Series("", index=frame.index)).map(_text).str.startswith("Dyn")
    )


def _ou_proof_mask(frame: pd.DataFrame) -> pd.Series:
    return frame.get("exact_mode", pd.Series("", index=frame.index)).map(_text).str.startswith("OU")


def _captured_proof_mask(frame: pd.DataFrame) -> pd.Series:
    return frame.get("vendor_response_captured", pd.Series(False, index=frame.index)).map(
        _truthy
    ) | frame.get("mode_proof_status", pd.Series("", index=frame.index)).eq("completed")


def _dynamic_raw_evidence_hashes(frame: pd.DataFrame, *, root: Path) -> dict[str, str]:
    return _mode_raw_evidence_hashes(frame, root=root)


def _mode_raw_evidence_hashes(frame: pd.DataFrame, *, root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for _, row in frame.iterrows():
        for field in ("request_path", "response_path"):
            value = _text(row.get(field, ""))
            path = _resolve_path(value, root=root)
            if path is None or not path.is_file():
                raise ValueError(f"captured mode proof is missing {field}")
            hashes[value] = sha256(path.read_bytes()).hexdigest()
    return hashes


def _enrich_existing_completed(frame: pd.DataFrame, *, queue: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    enriched = frame.copy().astype(object)
    metadata_fields = (
        "candidate_set_id",
        "discovery_policy_schema_version",
        "discovery_policy_hash",
        "wizard_period",
    )
    for index, row in enriched.iterrows():
        matches = queue.loc[
            queue.get("pair", pd.Series("", index=queue.index))
            .map(_text)
            .str.upper()
            .eq(_text(row.get("pair", "")).upper())
            & queue.get("local_interval", pd.Series("", index=queue.index))
            .map(_text)
            .str.lower()
            .eq(_text(row.get("local_interval", "")).lower())
            & queue.get("exact_mode", pd.Series("", index=queue.index))
            .map(_text)
            .str.lower()
            .eq(_text(row.get("exact_mode", "")).lower())
        ]
        if not matches.empty:
            candidate = matches.iloc[0]
            for field in metadata_fields:
                value = candidate.get(field, "")
                if _text(value):
                    enriched.at[index, field] = value
        observations = _identity_observations(enriched.loc[index])
        enriched.at[index, "proof_observations"] = observations
        enriched.at[index, "proof_window_kind"] = _proof_window_kind(
            enriched.at[index, "wizard_period"] if "wizard_period" in enriched.columns else "",
            observations,
        )
        validity = _proof_validity_fields(enriched.loc[index])
        for field, value in validity.items():
            enriched.at[index, field] = value
    return enriched


def _requested_observations(candidate: pd.Series) -> int:
    requested = _int_or_none(candidate.get("proof_observations", ""))
    if requested is None:
        requested = _int_or_none(candidate.get("wizard_period", ""))
    if requested is None:
        raise ValueError("proof_observations_missing")
    if requested < 50:
        raise ValueError("proof_observations_below_vendor_minimum_50")
    return min(requested, CUSTOM_SERIES_MAX_ROWS)


def _identity_observations(row: pd.Series) -> int:
    for field in ("proof_observations", "history_rows", "wizard_period"):
        value = _int_or_none(row.get(field, ""))
        if value is not None and value > 0:
            return min(value, CUSTOM_SERIES_MAX_ROWS)
    return 0


def _proof_window_kind(wizard_period: object, proof_observations: object) -> str:
    period = _int_or_none(wizard_period)
    observations = _int_or_none(proof_observations)
    if period is None or observations is None:
        return "unknown_window"
    if observations == min(period, CUSTOM_SERIES_MAX_ROWS):
        return "scanner_horizon_parity"
    return "extended_history" if observations > period else "truncated_history"


def _proof_validity_fields(row: pd.Series) -> dict[str, object]:
    if _text(row.get("mode_proof_status", "")) != "completed":
        return {
            "proof_validity": "NOT_COMPLETED",
            "validity_blocker": "vendor_proof_not_completed",
            "training_eligible": False,
        }
    blockers = [
        "vendor_performance_accounting_not_reconstructed",
        "vendor_full_sample_statistics_are_not_live_signal_safe",
        "vendor_cost_and_funding_parity_not_proven",
    ]
    window_kind = _text(row.get("proof_window_kind", ""))
    if window_kind != "scanner_horizon_parity":
        blockers.append("proof_window_does_not_match_scanner_horizon")
    return {
        "proof_validity": "RESEARCH_DIAGNOSTIC_WEAK",
        "validity_blocker": ";".join(blockers),
        "training_eligible": False,
    }


def _credit_preflight(
    *,
    execute: bool,
    execution_enabled: bool,
    api_key: str,
    selected: int,
    daily_credit_limit: int,
    reserved_credits: int,
    credits_fetcher: Any | None,
) -> dict[str, object]:
    if daily_credit_limit <= 0:
        raise ValueError("daily_credit_limit must be positive")
    if reserved_credits < 0:
        raise ValueError("reserved_credits must be non-negative")
    base = {"used": "", "remaining": "", "reserved": reserved_credits}
    if not execute:
        return {**base, "status": "NOT_REQUESTED", "blocker": "preflight_only"}
    if not execution_enabled:
        return {
            **base,
            "status": "BLOCKED",
            "blocker": "QPA_ENABLE_WIZARD_CUSTOM_SERIES_PROOF_not_true",
        }
    if not api_key:
        return {**base, "status": "BLOCKED", "blocker": "CRYPTO_WIZARDS_API_KEY_missing"}
    if selected <= 0:
        return {**base, "status": "NOT_REQUIRED", "blocker": "no_selected_requests"}
    try:
        payload = (credits_fetcher or fetch_credits_used)(api_key=api_key)
        usage = parse_wizard_credit_usage(payload, configured_limit=daily_credit_limit)
    except (CryptoWizardsFetchError, OSError, TypeError, ValueError) as exc:  # pragma: no cover
        return {
            **base,
            "status": "BLOCKED",
            "blocker": f"credit_preflight_failed:{type(exc).__name__}:{exc}",
        }
    if not usage.known:
        return {
            **base,
            "status": "BLOCKED",
            "blocker": "credit_usage_unknown",
        }
    planned = int(selected) * CUSTOM_SERIES_CREDIT_COST
    available_after_reserve = max(int(usage.remaining or 0) - reserved_credits, 0)
    if planned > available_after_reserve:
        return {
            "used": usage.used if usage.used is not None else "",
            "remaining": usage.remaining if usage.remaining is not None else "",
            "reserved": reserved_credits,
            "status": "BLOCKED",
            "blocker": "insufficient_credits_after_reserve",
        }
    return {
        "used": usage.used if usage.used is not None else "",
        "remaining": usage.remaining if usage.remaining is not None else "",
        "reserved": reserved_credits,
        "status": "PASS",
        "blocker": "",
    }


def _orient_history_prices(
    prices: pd.DataFrame,
    *,
    payload: dict[str, Any],
    candidate: pd.Series,
) -> pd.DataFrame:
    base_x = _asset_token(payload.get("asset_x", ""))
    base_y = _asset_token(payload.get("asset_y", ""))
    desired_x = _asset_token(candidate.get("asset_x", ""))
    desired_y = _asset_token(candidate.get("asset_y", ""))
    if not all((base_x, base_y, desired_x, desired_y)):
        return prices
    if (desired_x, desired_y) == (base_x, base_y):
        return prices
    if (desired_x, desired_y) == (base_y, base_x):
        return prices.rename(
            columns={
                "open_x": "open_y",
                "price_x": "price_y",
                "open_y": "open_x",
                "price_y": "price_x",
            }
        )[["open_x", "price_x", "open_y", "price_y"]]
    raise ValueError("candidate_orientation_does_not_match_local_history_assets")


def _entry_level(capture: pd.Series, *, exact_mode: str) -> float | None:
    long_level = _float(capture.get("entry_long", ""))
    short_level = _float(capture.get("entry_short", ""))
    if exact_mode == "Copula":
        return long_level
    levels = [abs(value) for value in (long_level, short_level) if value is not None]
    return max(levels) if levels else None


def _exit_level(capture: pd.Series) -> float | None:
    long_level = _float(capture.get("exit_long", ""))
    short_level = _float(capture.get("exit_short", ""))
    if long_level is None:
        return short_level
    if short_level is None:
        return long_level
    return long_level if math.isclose(long_level, short_level, rel_tol=0.0, abs_tol=1e-12) else None


def _capture_asset(capture: pd.Series, leg: str) -> str:
    for field in (
        f"capture_asset_{leg}",
        f"orientation_expected_asset_{leg}",
        f"asset_{leg}",
    ):
        value = _asset_token(capture.get(field, ""))
        if value:
            return value
    return ""


def _asset_token(value: object) -> str:
    token = _text(value).upper()
    for suffix in ("-USD", "USDT", "USD"):
        if token.endswith(suffix):
            token = token[: -len(suffix)]
            break
    return token.strip("-_/ ")


def _positive_rate_from_pct(value: object) -> float | str:
    number = _float(value)
    return number / 100.0 if number is not None and number > 0.0 else ""


def _positive_int(value: object) -> int | str:
    number = _int_or_none(value)
    return number if number is not None and number > 0 else ""


def _exhaustive_queue_markdown(
    frame: pd.DataFrame, *, ledger_path: Path, history_path: Path
) -> str:
    lines = [
        "# Exhaustive Wizard Exact-Mode Proof Queue",
        "",
        f"- Queue rows: `{len(frame)}`",
        f"- Eligible mode/orientation cells: `{int(frame.get('vendor_custom_series_eligible', pd.Series(dtype=bool)).map(_truthy).sum())}`",
        f"- Pair groups: `{int(frame.get('pair_group_id', pd.Series(dtype=str)).nunique())}`",
        "- Discovery-score filtering: `none`",
        "- Promotion authority: `none`",
        "- Live trading authorized: `false`",
        "",
        f"Wizard settings source: `{ledger_path}`",
        f"Frozen Hyperliquid history source: `{history_path}`",
        "",
    ]
    if frame.empty:
        return "\n".join(lines + ["No exact-mode cells had matching active evidence.", ""])
    preview = frame[
        [
            "pair_group_id",
            "pair",
            "exact_mode",
            "orientation",
            "wizard_period",
            "vendor_custom_series_eligible",
            "blocker",
        ]
    ]
    return "\n".join(lines + ["## Cells", "", preview.to_markdown(index=False), ""])


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def _resolve_path(value: str, *, root: Path) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


def _float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _int_or_none(value: object) -> int | None:
    number = _float(value)
    return int(number) if number is not None else None


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return _text(value).lower() in {"1", "true", "yes", "y"}


def _text(value: object) -> str:
    if value is None or (not isinstance(value, (dict, list)) and pd.isna(value)):
        return ""
    return str(value).strip()


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _safe_filename(value: object) -> str:
    cleaned = "".join(character.lower() if character.isalnum() else "_" for character in str(value))
    return "_".join(part for part in cleaned.split("_") if part)[:160] or "mode_proof"


def _relative(path: Path, *, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(frame, path, index=False)


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    promote_staged_file(temporary, path)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, content, encoding="utf-8")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _markdown(frame: pd.DataFrame, *, selected: int, eligible: int, execute: bool) -> str:
    lines = [
        "# Hyperliquid Wizard Vendor Mode Proofs",
        "",
        f"- Queue-eligible rows: `{eligible}`",
        f"- Selected under the `{REQUEST_CAP}` request cap: `{selected}`",
        f"- Execution requested: `{execute}`",
        "- Promotion authority: `none`; a completed proof is still research evidence.",
        "",
    ]
    if frame.empty:
        return "\n".join(lines + ["No queue-eligible hypotheses were available.", ""])
    preview = frame[
        [
            "pair",
            "local_interval",
            "exact_mode",
            "orientation",
            "mode_proof_status",
            "credits_estimated",
            "blocker",
            "next_step",
        ]
    ]
    return "\n".join(lines + ["## Proof Queue", "", preview.to_markdown(index=False), ""])
