from __future__ import annotations

import json
import math
import os
from collections.abc import Callable
from copy import deepcopy
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_platform.active_pipeline import CommandResult
from quant_platform.crypto_wizards_history import (
    CryptoWizardsCustomSeriesCopulaRequest,
    fetch_credits_used,
    fetch_custom_series_copula,
)
from quant_platform.crypto_wizards_sweep import parse_wizard_credit_usage
from quant_platform.orchestration.corrective_redaction import safe_exception_code
from quant_platform.orchestration.corrective_runtime import (
    create_exclusive_bytes,
    promote_staged_file,
    write_immutable_bytes,
)
from quant_platform.wizard_credit_budget import (
    COPULA_BEHAVIORAL_REPEATS,
    COPULA_POST_CREDIT_COST,
)
from quant_platform.wizard_hyperliquid_mode_proof import (
    DEFAULT_DAILY_CREDIT_LIMIT,
    DEFAULT_RESERVED_CREDITS,
)

ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "thewiz.wizard_copula_behavioral_evaluation.v3"
CAPTURE_SCHEMA_VERSION = "thewiz.wizard_copula_behavioral_capture.v1"
COHORT_SCHEMA_VERSION = "thewiz.wizard_copula_behavioral_cohort.v1"
ATTEMPT_SCHEMA_VERSION = "thewiz.wizard_copula_behavioral_attempt.v1"
TOLERANCE = 1e-12


def register_copula_behavioral_v2(
    *,
    root: Path = ROOT,
    now: datetime | None = None,
) -> CommandResult:
    """Freeze corrected exact-swap Copula inputs without calling the vendor."""

    paths = _artifact_paths(root, generation=2)
    if paths["contract"].is_file() or paths["contract_receipt"].is_file():
        contract = _read_json_strict(paths["contract"])
        receipt = _read_json_strict(paths["contract_receipt"])
        _validate_contract(
            root=root,
            contract=contract,
            contract_path=paths["contract"],
            receipt=receipt,
        )
        return CommandResult(
            paths={
                "contract": paths["contract"],
                "contract_receipt": paths["contract_receipt"],
                "proof_queue": paths["queue"],
                "source_ledger": paths["proof"],
            },
            summary={
                **receipt,
                "status": "REUSED_PREREGISTRATION",
                "artifact_generation": 2,
            },
        )

    active = root / "reports" / "active"
    source_queue_path = active / "exhaustive_wizard_exact_mode_proof_queue.csv"
    source_proof_path = active / "hyperliquid_wizard_vendor_mode_proofs.csv"
    source_queue = _read_csv(source_queue_path)
    source_proofs = _read_csv(source_proof_path)
    eligible = source_queue.loc[
        source_queue.get("exact_mode", pd.Series("", index=source_queue.index)).eq(
            "Copula"
        )
        & source_queue.get(
            "vendor_custom_series_eligible", pd.Series(False, index=source_queue.index)
        ).map(_truthy)
    ].copy()
    identities = {
        (_text(row.get("pair_group_id")), _text(row.get("orientation")))
        for _, row in eligible.iterrows()
    }
    if len(eligible) != 4 or len(identities) != 4:
        raise ValueError("Copula v2 registration requires exactly four eligible identities")

    queue_rows: list[dict[str, object]] = []
    source_rows: list[dict[str, object]] = []
    bindings: list[dict[str, object]] = []
    request_dir = root / "data" / "research" / "wizard_copula_v2" / "source_requests"
    for source_group_id, group in eligible.groupby("pair_group_id", sort=True):
        orientations = {
            _text(row.get("orientation")): row for _, row in group.iterrows()
        }
        if set(orientations) != {"original", "reverse"}:
            raise ValueError("Copula v2 source group must have original and reverse rows")
        original_queue = orientations["original"]
        reverse_queue = orientations["reverse"]
        original_proof = _single_source_proof(
            source_proofs, group_id=_text(source_group_id), orientation="original"
        )
        reverse_proof = _single_source_proof(
            source_proofs, group_id=_text(source_group_id), orientation="reverse"
        )
        source_request_path = _resolve(root, original_proof.get("request_path"))
        prior_reverse_path = _resolve(root, reverse_proof.get("request_path"))
        if source_request_path is None or prior_reverse_path is None:
            raise ValueError("Copula v2 source request evidence is missing")
        original_payload = _read_json_strict(source_request_path)
        observations = min(
            int(float(original_queue.get("proof_observations"))),
            int(float(reverse_queue.get("proof_observations"))),
        )
        normalized_original = _normalize_backtest_request(
            original_payload, observations=observations
        )
        normalized_reverse = _swap_backtest_request(normalized_original)
        v2_group_id = f"{_text(source_group_id)}:copula_v2"
        for orientation, queue_row, payload in (
            ("original", original_queue, normalized_original),
            ("reverse", reverse_queue, normalized_reverse),
        ):
            request_path = request_dir / f"{v2_group_id.replace(':', '_')}_{orientation}.json"
            _write_or_validate_immutable_json(payload, request_path)
            queue_record = {key: _json_scalar(value) for key, value in queue_row.items()}
            queue_record.update(
                {
                    "pair_group_id": v2_group_id,
                    "proof_observations": observations,
                    "vendor_custom_series_eligible": True,
                    "copula_source_generation": 2,
                    "copula_source_group_id": _text(source_group_id),
                }
            )
            queue_rows.append(queue_record)
            source_rows.append(
                {
                    "pair_group_id": v2_group_id,
                    "source_pair_group_id": _text(source_group_id),
                    "pair": _text(queue_row.get("pair")),
                    "exact_mode": "Copula",
                    "orientation": orientation,
                    "proof_observations": observations,
                    "source_request_frozen": True,
                    "vendor_response_captured": False,
                    "request_path": _relative(request_path, root),
                    "request_sha256": _file_hash(request_path),
                    "source_original_request_path": _relative(source_request_path, root),
                    "source_original_request_sha256": _file_hash(source_request_path),
                    "prior_reverse_request_path": _relative(prior_reverse_path, root),
                    "prior_reverse_request_sha256": _file_hash(prior_reverse_path),
                    "normalization_rule": (
                        "canonical_original_last_n_then_reverse_exact_swap"
                    ),
                    "research_only": True,
                    "candidate_promotion_authority": False,
                    "testnet_order_authority": False,
                    "live_trading_authorized": False,
                }
            )
            bindings.append(
                {
                    "pair_group_id": v2_group_id,
                    "source_pair_group_id": _text(source_group_id),
                    "pair": _text(queue_row.get("pair")),
                    "orientation": orientation,
                    "proof_observations": observations,
                    "request_path": _relative(request_path, root),
                    "request_sha256": _file_hash(request_path),
                }
            )

    queue_frame = pd.DataFrame(queue_rows).sort_values(
        ["pair_group_id", "orientation"], kind="stable"
    )
    source_frame = pd.DataFrame(source_rows).sort_values(
        ["pair_group_id", "orientation"], kind="stable"
    )
    _write_or_validate_immutable_csv(queue_frame, paths["queue"])
    _write_or_validate_immutable_csv(source_frame, paths["proof"])
    _validate_v2_source_swaps(root=root, source_frame=source_frame)

    comparator_contract = active / "wizard_mode_comparator_contract.csv"
    comparator_receipt = active / "wizard_mode_comparator_contract_receipt.json"
    for required in (
        source_queue_path,
        source_proof_path,
        comparator_contract,
        comparator_receipt,
    ):
        if not required.is_file():
            raise ValueError(f"Copula v2 registration evidence missing: {required}")
    contract = {
        "schema_version": "thewiz.wizard_copula_behavioral_parity.v2",
        "artifact_generation": 2,
        "status": "PREREGISTERED_WAITING_VENDOR_RESPONSES",
        "frozen_checks": [
            "two_repeat_responses_per_cell",
            "exact_original_reverse_input_swap",
            "copula_family_orientation_symmetry",
            "conditional_probability_orientation_symmetry",
            "immutable_request_response_and_cohort_hashes",
        ],
        "holdout_bindings": bindings,
        "normalization_rule": "canonical_original_last_n_then_reverse_exact_swap",
        "vendor_responses_at_registration": 0,
        "threshold_changes_after_vendor_response_allowed": False,
        "automatic_activation": False,
        "research_only": True,
        "candidate_promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    _write_or_validate_immutable_json(contract, paths["contract"])
    receipt_core = {
        "schema_version": "thewiz.wizard_copula_behavioral_contract_receipt.v2",
        "registered_at_utc": _as_utc(now).isoformat(),
        "contract_path": _relative(paths["contract"], root),
        "contract_sha256": _file_hash(paths["contract"]),
        "proof_queue_path": _relative(paths["queue"], root),
        "proof_queue_sha256": _file_hash(paths["queue"]),
        "formula_comparator_contract_path": _relative(comparator_contract, root),
        "formula_comparator_contract_sha256": _file_hash(comparator_contract),
        "formula_comparator_receipt_path": _relative(comparator_receipt, root),
        "formula_comparator_receipt_sha256": _file_hash(comparator_receipt),
        "source_queue_path": _relative(source_queue_path, root),
        "source_queue_sha256": _file_hash(source_queue_path),
        "source_proof_path": _relative(source_proof_path, root),
        "source_proof_sha256": _file_hash(source_proof_path),
        "registered_before_current_queue_vendor_responses": True,
        "vendor_responses_at_registration": 0,
        "threshold_changes_after_vendor_response_allowed": False,
        "research_only": True,
        "candidate_promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    receipt = {
        **receipt_core,
        "receipt_id": "copulav2contract_" + _json_hash(receipt_core)[:20],
    }
    _write_or_validate_immutable_json(receipt, paths["contract_receipt"])
    return CommandResult(
        paths={
            "contract": paths["contract"],
            "contract_receipt": paths["contract_receipt"],
            "proof_queue": paths["queue"],
            "source_ledger": paths["proof"],
        },
        summary={
            **receipt,
            "status": "REGISTERED_WAITING_VENDOR_RESPONSES",
            "artifact_generation": 2,
            "registered_cells": len(bindings),
        },
    )


def run_copula_behavioral_proofs(
    *,
    root: Path = ROOT,
    now: datetime | None = None,
    execute: bool = False,
    api_key: str | None = None,
    fetcher: Callable[..., dict[str, Any]] = fetch_custom_series_copula,
    credits_fetcher: Callable[..., dict[str, Any] | list[Any]] = fetch_credits_used,
    daily_credit_limit: int = DEFAULT_DAILY_CREDIT_LIMIT,
    reserved_credits: int = DEFAULT_RESERVED_CREDITS,
    artifact_generation: int = 1,
) -> CommandResult:
    """Capture and evaluate preregistered Copula black-box behavior."""

    timestamp = _as_utc(now)
    paths = _artifact_paths(root, generation=artifact_generation)
    contract_path = paths["contract"]
    contract_receipt_path = paths["contract_receipt"]
    contract = _read_json(contract_path)
    contract_receipt = _read_json(contract_receipt_path)
    _validate_contract(
        root=root,
        contract=contract,
        contract_path=contract_path,
        receipt=contract_receipt,
    )

    queue_path = paths["queue"]
    proof_path = paths["proof"]
    queue = _read_csv(queue_path)
    expected = queue.loc[
        queue.get("exact_mode", pd.Series("", index=queue.index)).eq("Copula")
        & queue.get(
            "vendor_custom_series_eligible", pd.Series(False, index=queue.index)
        ).map(_truthy)
    ].copy()
    proofs = _read_csv(proof_path)
    source_ready_field = (
        "vendor_response_captured" if artifact_generation == 1 else "source_request_frozen"
    )
    captured = proofs.loc[
        proofs.get("exact_mode", pd.Series("", index=proofs.index)).eq("Copula")
        & proofs.get(
            source_ready_field, pd.Series(False, index=proofs.index)
        ).map(_truthy)
        & proofs.get("pair_group_id", pd.Series("", index=proofs.index))
        .map(_text)
        .ne("")
    ].copy()
    if not captured.empty:
        identities = captured.apply(_identity, axis=1)
        if identities.duplicated().any():
            raise ValueError("duplicate captured Copula proof identities")

    expected_identities = {_identity(row) for _, row in expected.iterrows()}
    captured = captured.loc[
        captured.apply(lambda row: _identity(row) in expected_identities, axis=1)
    ].copy()
    if artifact_generation == 2:
        _validate_v2_runtime_bindings(
            root=root,
            contract=contract,
            expected=expected,
            source_frame=captured,
        )
    plans = [
        _build_capture_plan(
            root=root,
            proof=proof,
            contract_path=contract_path,
            contract_receipt_path=contract_receipt_path,
            queue_path=queue_path,
        )
        for _, proof in captured.iterrows()
    ]
    plans = _apply_source_orientation_contract(plans)
    resolved_key = api_key or (
        os.getenv("CRYPTO_WIZARDS_API_KEY", "").strip() if execute else ""
    )
    calls_required = sum(
        COPULA_BEHAVIORAL_REPEATS
        for plan in plans
        if plan["capture_state"] == "MISSING"
    )
    credit_state = _credit_preflight(
        execute=execute,
        api_key=resolved_key,
        calls_required=calls_required,
        credits_fetcher=credits_fetcher,
        daily_credit_limit=daily_credit_limit,
        reserved_credits=reserved_credits,
    )
    attempt = _register_daily_attempt(
        root=root,
        timestamp=timestamp,
        execute=execute,
        plans=plans,
        calls_required=calls_required,
        credit_state=credit_state,
        contract_path=contract_path,
        contract_receipt_path=contract_receipt_path,
    )
    if attempt["status"] == "BLOCKED":
        credit_state = {
            "status": "BLOCKED",
            "blocker": attempt["blocker"],
        }
    rows: list[dict[str, object]] = []
    calls_made = 0
    responses_captured_this_cycle = 0
    newly_captured = 0
    reused_captures = 0
    for plan in plans:
        row, row_calls, was_new, was_reused = _capture_identity(
            root=root,
            plan=plan,
            execute=execute,
            api_key=resolved_key,
            fetcher=fetcher,
            credit_state=credit_state,
        )
        rows.append(row)
        calls_made += row_calls
        responses_captured_this_cycle += min(
            max(int(row.get("endpoint_responses_captured", 0) or 0), 0),
            row_calls,
        )
        newly_captured += int(was_new)
        reused_captures += int(was_reused)
        if (
            plan["capture_state"] == "MISSING"
            and row_calls > 0
            and not was_new
        ):
            credit_state = {
                "status": "BLOCKED",
                "blocker": "copula_capture_failed_remaining_calls_aborted",
            }

    frame = pd.DataFrame(rows)
    frame = _apply_orientation_checks(frame)
    expected_count = len(expected_identities)
    endpoint_captured = int(
        frame.get("endpoint_responses_captured", pd.Series(dtype=int)).eq(
            COPULA_BEHAVIORAL_REPEATS
        ).sum()
    )
    failed = int(frame.get("behavioral_status", pd.Series(dtype=str)).eq("FAIL").sum())
    passed = int(frame.get("behavioral_status", pd.Series(dtype=str)).eq("PASS").sum())
    provenance_complete = int(
        frame.get("provenance_status", pd.Series(dtype=str)).eq("PASS").sum()
    )
    incomplete_evidence = int(
        frame.get("behavioral_status", pd.Series(dtype=str))
        .isin({"MISSING", "BLOCKED"})
        .sum()
    )
    if expected_count == 0:
        status = "BLOCKED_NO_REGISTERED_COPULA_CELLS"
    elif captured.empty:
        status = "WAITING_FOR_COPULA_BACKTEST_RESPONSES"
    elif failed > 0:
        status = "FAIL"
    elif (
        endpoint_captured == expected_count
        and provenance_complete == expected_count
        and passed == expected_count
    ):
        status = "PASS"
    elif incomplete_evidence > 0:
        status = "INCOMPLETE"
    elif not execute:
        status = "PLANNED"
    elif credit_state["status"] != "PASS":
        status = "BLOCKED_CREDIT_PREFLIGHT"
    elif endpoint_captured < expected_count:
        status = "INCOMPLETE"
    elif provenance_complete < expected_count:
        status = "FAIL_PROVENANCE"
    elif passed == expected_count:
        status = "PASS"
    else:
        status = "INCOMPLETE"

    detail_path = paths["detail"]
    status_path = paths["status"]
    _atomic_csv(frame, detail_path)
    cohort = _publish_cohort_receipt(
        root=root,
        frame=frame,
        expected_count=expected_count,
        contract_path=contract_path,
        contract_receipt_path=contract_receipt_path,
        queue_path=queue_path,
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "artifact_generation": artifact_generation,
        "evaluated_at_utc": timestamp.isoformat(),
        "status": status,
        "expected_cells": expected_count,
        "backtest_responses_available": len(captured),
        "source_requests_available": len(captured),
        "endpoint_cells_captured": endpoint_captured,
        "behavioral_cells_passed": passed,
        "behavioral_cells_failed": failed,
        "provenance_cells_complete": provenance_complete,
        "endpoint_calls_made": calls_made,
        "endpoint_responses_captured_this_cycle": (
            responses_captured_this_cycle
        ),
        "endpoint_calls_required": calls_required,
        "credits_estimated": calls_required * COPULA_POST_CREDIT_COST,
        "endpoint_calls_total_contract": (
            expected_count * COPULA_BEHAVIORAL_REPEATS
        ),
        "endpoint_calls_avoided_by_reuse": (
            reused_captures * COPULA_BEHAVIORAL_REPEATS
        ),
        "newly_captured_cells": newly_captured,
        "reused_immutable_capture_cells": reused_captures,
        "credit_preflight_status": credit_state["status"],
        "credit_preflight_blocker": credit_state["blocker"],
        "daily_attempt_status": attempt["status"],
        "daily_attempt_id": attempt["attempt_id"],
        "daily_attempt_path": attempt["attempt_path"],
        "daily_attempt_sha256": attempt["attempt_sha256"],
        "cohort_receipt_id": cohort["cohort_id"],
        "cohort_receipt_path": cohort["cohort_path"],
        "cohort_receipt_sha256": cohort["cohort_sha256"],
        "formula_parity_proven": False,
        "behavioral_parity_proven": status == "PASS",
        "research_only": True,
        "candidate_promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
        "contract_path": _relative(contract_path, root),
        "contract_sha256": _file_hash(contract_path),
        "contract_receipt_path": _relative(contract_receipt_path, root),
        "contract_receipt_sha256": _file_hash(contract_receipt_path),
        "proof_queue_path": _relative(queue_path, root),
        "proof_queue_sha256": _file_hash(queue_path),
        "vendor_proof_path": _relative(proof_path, root),
        "vendor_proof_sha256": _file_hash(proof_path),
        "evidence_path": _relative(detail_path, root),
        "evidence_sha256": _file_hash(detail_path),
    }
    _atomic_json(summary, status_path)
    paths = {"detail": detail_path, "status": status_path}
    if cohort["cohort_path"]:
        paths["immutable_cohort_receipt"] = root / cohort["cohort_path"]
    if attempt["attempt_path"]:
        paths["immutable_daily_attempt"] = root / attempt["attempt_path"]
    return CommandResult(paths=paths, summary=summary)


def run_current_copula_behavioral_proofs(
    *,
    root: Path = ROOT,
    **kwargs: Any,
) -> CommandResult:
    """Use the newest preregistered Copula generation, falling back to v1."""

    generation = 2 if _artifact_paths(root, generation=2)["contract"].is_file() else 1
    return run_copula_behavioral_proofs(
        root=root,
        artifact_generation=generation,
        **kwargs,
    )


def _artifact_paths(root: Path, *, generation: int) -> dict[str, Path]:
    active = root / "reports" / "active"
    if generation == 1:
        return {
            "contract": root / "config" / "wizard_copula_behavioral_parity.json",
            "contract_receipt": active / "wizard_copula_behavioral_contract_receipt.json",
            "queue": active / "exhaustive_wizard_exact_mode_proof_queue.csv",
            "proof": active / "hyperliquid_wizard_vendor_mode_proofs.csv",
            "detail": active / "wizard_copula_behavioral_evaluation.csv",
            "status": active / "wizard_copula_behavioral_status.json",
        }
    if generation == 2:
        return {
            "contract": root / "config" / "wizard_copula_behavioral_parity_v2.json",
            "contract_receipt": active / "wizard_copula_behavioral_v2_contract_receipt.json",
            "queue": active / "exhaustive_wizard_copula_proof_queue_v2.csv",
            "proof": active / "wizard_copula_source_ledger_v2.csv",
            "detail": active / "wizard_copula_behavioral_v2_evaluation.csv",
            "status": active / "wizard_copula_behavioral_v2_status.json",
        }
    raise ValueError(f"unsupported Copula artifact generation: {generation}")


def _single_source_proof(
    proofs: pd.DataFrame, *, group_id: str, orientation: str
) -> pd.Series:
    selected = proofs.loc[
        proofs.get("pair_group_id", pd.Series("", index=proofs.index)).map(_text).eq(
            group_id
        )
        & proofs.get("exact_mode", pd.Series("", index=proofs.index)).eq("Copula")
        & proofs.get("orientation", pd.Series("", index=proofs.index)).eq(orientation)
        & proofs.get(
            "vendor_response_captured", pd.Series(False, index=proofs.index)
        ).map(_truthy)
    ]
    if len(selected) != 1:
        raise ValueError(
            f"Copula v2 requires one captured source proof: {group_id}:{orientation}"
        )
    return selected.iloc[0]


def _normalize_backtest_request(
    payload: dict[str, Any], *, observations: int
) -> dict[str, Any]:
    if observations < 50:
        raise ValueError("Copula v2 requires at least 50 shared observations")
    normalized = deepcopy(payload)
    params = normalized.get("params")
    if not isinstance(params, dict):
        raise TypeError("Copula v2 source request params are missing")
    fields = (
        "series_1_opens",
        "series_1_closes",
        "series_2_opens",
        "series_2_closes",
    )
    for field in fields:
        values = params.get(field)
        if not isinstance(values, list) or len(values) < observations:
            raise ValueError(f"Copula v2 source request field is too short: {field}")
        params[field] = values[-observations:]
    return normalized


def _swap_backtest_request(payload: dict[str, Any]) -> dict[str, Any]:
    swapped = deepcopy(payload)
    params = swapped["params"]
    for suffix in ("opens", "closes"):
        first = f"series_1_{suffix}"
        second = f"series_2_{suffix}"
        params[first], params[second] = params[second], params[first]
    return swapped


def _validate_v2_source_swaps(*, root: Path, source_frame: pd.DataFrame) -> None:
    for _, group in source_frame.groupby("pair_group_id", sort=True):
        if len(group) != 2:
            raise ValueError("Copula v2 source group must contain exactly two rows")
        by_orientation = {
            _text(row.get("orientation")): row for _, row in group.iterrows()
        }
        if set(by_orientation) != {"original", "reverse"}:
            raise ValueError("Copula v2 source orientations are incomplete")
        original = _read_json_strict(root / _text(by_orientation["original"]["request_path"]))
        reverse = _read_json_strict(root / _text(by_orientation["reverse"]["request_path"]))
        original_params = original.get("params", {})
        reverse_params = reverse.get("params", {})
        for suffix in ("opens", "closes"):
            if (
                original_params.get(f"series_1_{suffix}")
                != reverse_params.get(f"series_2_{suffix}")
                or original_params.get(f"series_2_{suffix}")
                != reverse_params.get(f"series_1_{suffix}")
            ):
                raise ValueError("Copula v2 source requests are not exact swaps")


def _validate_v2_runtime_bindings(
    *,
    root: Path,
    contract: dict[str, Any],
    expected: pd.DataFrame,
    source_frame: pd.DataFrame,
) -> None:
    bindings = {
        (_text(item.get("pair_group_id")), _text(item.get("orientation"))): item
        for item in contract.get("holdout_bindings", [])
        if isinstance(item, dict)
    }
    expected_identities = {
        (_text(row.get("pair_group_id")), _text(row.get("orientation")))
        for _, row in expected.iterrows()
    }
    source_identities = {
        (_text(row.get("pair_group_id")), _text(row.get("orientation")))
        for _, row in source_frame.iterrows()
    }
    if len(bindings) != 4 or set(bindings) != expected_identities:
        raise ValueError("Copula v2 contract identity binding mismatch")
    if source_identities != expected_identities:
        raise ValueError("Copula v2 source ledger identity coverage mismatch")
    for _, row in source_frame.iterrows():
        identity = (_text(row.get("pair_group_id")), _text(row.get("orientation")))
        binding = bindings[identity]
        request_path = _resolve(root, row.get("request_path"))
        if request_path is None:
            raise ValueError("Copula v2 bound request is missing")
        if (
            _text(row.get("request_path")) != _text(binding.get("request_path"))
            or _file_hash(request_path) != _text(binding.get("request_sha256"))
            or _text(row.get("request_sha256")) != _text(binding.get("request_sha256"))
        ):
            raise ValueError("Copula v2 request binding hash mismatch")
        if not _truthy(row.get("source_request_frozen")):
            raise ValueError("Copula v2 source request is not frozen")
        if any(
            _truthy(row.get(field))
            for field in (
                "candidate_promotion_authority",
                "testnet_order_authority",
                "live_trading_authorized",
            )
        ):
            raise ValueError("Copula v2 source ledger has forbidden authority")
    _validate_v2_source_swaps(root=root, source_frame=source_frame)


def _write_or_validate_immutable_csv(frame: pd.DataFrame, path: Path) -> None:
    content = frame.to_csv(index=False).encode("utf-8")
    try:
        write_immutable_bytes(path, content)
    except ValueError as exc:
        raise ValueError(f"immutable CSV content mismatch: {path}") from exc


def _apply_source_orientation_contract(
    plans: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Block new endpoint calls unless each source request is an exact swap."""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for plan in plans:
        grouped.setdefault(_text(plan.get("base", {}).get("pair_group_id")), []).append(
            plan
        )
    for group in grouped.values():
        by_orientation = {
            _text(plan.get("base", {}).get("orientation")): plan for plan in group
        }
        original = by_orientation.get("original")
        reverse = by_orientation.get("reverse")
        exact_swap = bool(
            len(group) == 2
            and original is not None
            and reverse is not None
            and original.get("request") is not None
            and reverse.get("request") is not None
            and original["request"].series_1_closes
            == reverse["request"].series_2_closes
            and original["request"].series_2_closes
            == reverse["request"].series_1_closes
        )
        if exact_swap:
            continue
        for plan in group:
            if plan.get("capture_state") == "MISSING":
                plan["capture_state"] = "BLOCKED"
                plan["blocker"] = (
                    "copula_source_orientation_requests_not_exact_swaps"
                )
    return plans


def _build_capture_plan(
    *,
    root: Path,
    proof: pd.Series,
    contract_path: Path,
    contract_receipt_path: Path,
    queue_path: Path,
) -> dict[str, Any]:
    base = _empty_evaluation_row(proof)
    backtest_request_path = _resolve(root, proof.get("request_path"))
    if backtest_request_path is None:
        return {
            "base": base,
            "capture_state": "BLOCKED",
            "blocker": "copula_backtest_request_path_missing",
        }
    base["source_backtest_request_path"] = _relative(
        backtest_request_path, root
    )
    base["source_backtest_request_sha256"] = _file_hash(
        backtest_request_path
    )
    backtest_request = _read_json(backtest_request_path)
    params = (
        backtest_request.get("params")
        if isinstance(backtest_request.get("params"), dict)
        else {}
    )
    try:
        request = CryptoWizardsCustomSeriesCopulaRequest(
            series_1_closes=tuple(
                float(value) for value in params["series_1_closes"]
            ),
            series_2_closes=tuple(
                float(value) for value in params["series_2_closes"]
            ),
        )
        payload = request.payload()
    except (KeyError, TypeError, ValueError) as exc:
        return {
            "base": base,
            "capture_state": "BLOCKED",
            "blocker": (
                f"invalid_copula_endpoint_input:{safe_exception_code(exc)}"
            ),
        }
    series_1_sha256 = _json_hash(
        {"values": list(request.series_1_closes)}
    )
    series_2_sha256 = _json_hash(
        {"values": list(request.series_2_closes)}
    )
    payload_sha256 = _json_hash(payload)
    material = {
        "schema_version": CAPTURE_SCHEMA_VERSION,
        "contract_receipt_sha256": _file_hash(contract_receipt_path),
        "pair_group_id": base["pair_group_id"],
        "pair": base["pair"],
        "exact_mode": "Copula",
        "orientation": base["orientation"],
        "source_backtest_request_path": base[
            "source_backtest_request_path"
        ],
        "source_backtest_request_sha256": base[
            "source_backtest_request_sha256"
        ],
        "endpoint_payload_sha256": payload_sha256,
    }
    capture_id = "copulacapture_" + _json_hash(material)[:20]
    capture_dir = (
        root
        / "data"
        / "raw"
        / "crypto_wizards_copula_behavioral_proofs"
        / capture_id
    )
    request_path = capture_dir / "request.json"
    response_paths = [
        capture_dir / f"response_{index}.json" for index in (1, 2)
    ]
    receipt_path = capture_dir / "receipt.json"
    base.update(
        {
            "capture_id": capture_id,
            "series_1_sha256": series_1_sha256,
            "series_2_sha256": series_2_sha256,
            "series_length": len(request.series_1_closes),
            "request_path": _relative(request_path, root),
            "response_1_path": _relative(response_paths[0], root),
            "response_2_path": _relative(response_paths[1], root),
            "capture_receipt_path": _relative(receipt_path, root),
        }
    )
    plan = {
        "base": base,
        "capture_state": "MISSING",
        "blocker": "",
        "capture_id": capture_id,
        "capture_dir": capture_dir,
        "request": request,
        "payload": payload,
        "payload_sha256": payload_sha256,
        "request_path": request_path,
        "response_paths": response_paths,
        "receipt_path": receipt_path,
        "contract_path": contract_path,
        "contract_receipt_path": contract_receipt_path,
        "queue_path": queue_path,
    }
    _validate_capture_identity_uniqueness(root=root, plan=plan)
    if receipt_path.is_file():
        plan["receipt"] = _load_and_validate_capture_receipt(
            root=root, plan=plan
        )
        plan["capture_state"] = "COMPLETE"
        return plan
    if capture_dir.exists() and any(capture_dir.iterdir()):
        plan["capture_state"] = "BLOCKED"
        plan["blocker"] = "copula_capture_partial_or_orphaned_artifacts_present"
    return plan


def _empty_evaluation_row(proof: pd.Series) -> dict[str, object]:
    return {
        "pair_group_id": _text(proof.get("pair_group_id")),
        "pair": _text(proof.get("pair")),
        "exact_mode": "Copula",
        "orientation": _text(proof.get("orientation")),
        "capture_id": "",
        "endpoint_responses_captured": 0,
        "copula_name": "",
        "u1_given_u2": "",
        "u2_given_u1": "",
        "repeat_copula_name": "",
        "repeat_u1_given_u2": "",
        "repeat_u2_given_u1": "",
        "repeatability_max_abs_delta": "",
        "repeatability_status": "NOT_EVALUATED",
        "input_orientation_status": "NOT_EVALUATED",
        "orientation_status": "NOT_EVALUATED",
        "behavioral_status": "MISSING",
        "provenance_status": "NOT_EVALUATED",
        "series_1_sha256": "",
        "series_2_sha256": "",
        "series_length": "",
        "source_backtest_request_path": "",
        "source_backtest_request_sha256": "",
        "request_path": "",
        "request_sha256": "",
        "response_1_path": "",
        "response_1_sha256": "",
        "response_1_captured_at_utc": "",
        "response_2_path": "",
        "response_2_sha256": "",
        "response_2_captured_at_utc": "",
        "capture_receipt_path": "",
        "capture_receipt_sha256": "",
        "blocker": "",
        "candidate_promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }


def _build_capture_receipt(
    *,
    root: Path,
    plan: dict[str, Any],
    response_times: list[str],
) -> dict[str, Any]:
    if len(response_times) != COPULA_BEHAVIORAL_REPEATS:
        raise ValueError("complete Copula capture requires both response timestamps")
    base = plan["base"]
    core = {
        "schema_version": CAPTURE_SCHEMA_VERSION,
        "capture_id": plan["capture_id"],
        "pair_group_id": base["pair_group_id"],
        "pair": base["pair"],
        "exact_mode": "Copula",
        "orientation": base["orientation"],
        "series_1_sha256": base["series_1_sha256"],
        "series_2_sha256": base["series_2_sha256"],
        "series_length": base["series_length"],
        "source_backtest_request_path": base[
            "source_backtest_request_path"
        ],
        "source_backtest_request_sha256": base[
            "source_backtest_request_sha256"
        ],
        "contract_path": _relative(plan["contract_path"], root),
        "contract_sha256": _file_hash(plan["contract_path"]),
        "contract_receipt_path": _relative(
            plan["contract_receipt_path"], root
        ),
        "contract_receipt_sha256": _file_hash(
            plan["contract_receipt_path"]
        ),
        "proof_queue_path": _relative(plan["queue_path"], root),
        "proof_queue_sha256": _file_hash(plan["queue_path"]),
        "request_path": _relative(plan["request_path"], root),
        "request_sha256": _file_hash(plan["request_path"]),
        "endpoint_payload_sha256": plan["payload_sha256"],
        "response_1_path": _relative(plan["response_paths"][0], root),
        "response_1_sha256": _file_hash(plan["response_paths"][0]),
        "response_1_captured_at_utc": response_times[0],
        "response_2_path": _relative(plan["response_paths"][1], root),
        "response_2_sha256": _file_hash(plan["response_paths"][1]),
        "response_2_captured_at_utc": response_times[1],
        "endpoint_responses_captured": COPULA_BEHAVIORAL_REPEATS,
        "research_only": True,
        "candidate_promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    return {
        **core,
        "receipt_id": "copulareceipt_" + _json_hash(core)[:20],
    }


def _load_and_validate_capture_receipt(
    *, root: Path, plan: dict[str, Any]
) -> dict[str, Any]:
    receipt = _read_json_strict(plan["receipt_path"])
    _validate_content_id(
        receipt,
        id_field="receipt_id",
        prefix="copulareceipt_",
    )
    base = plan["base"]
    expected_values = {
        "schema_version": CAPTURE_SCHEMA_VERSION,
        "capture_id": plan["capture_id"],
        "pair_group_id": base["pair_group_id"],
        "pair": base["pair"],
        "exact_mode": "Copula",
        "orientation": base["orientation"],
        "series_1_sha256": base["series_1_sha256"],
        "series_2_sha256": base["series_2_sha256"],
        "series_length": base["series_length"],
        "source_backtest_request_path": base[
            "source_backtest_request_path"
        ],
        "source_backtest_request_sha256": base[
            "source_backtest_request_sha256"
        ],
        "contract_path": _relative(plan["contract_path"], root),
        "contract_sha256": _file_hash(plan["contract_path"]),
        "contract_receipt_path": _relative(
            plan["contract_receipt_path"], root
        ),
        "contract_receipt_sha256": _file_hash(
            plan["contract_receipt_path"]
        ),
        "proof_queue_path": _relative(plan["queue_path"], root),
        "proof_queue_sha256": _file_hash(plan["queue_path"]),
        "request_path": _relative(plan["request_path"], root),
        "endpoint_payload_sha256": plan["payload_sha256"],
        "response_1_path": _relative(plan["response_paths"][0], root),
        "response_2_path": _relative(plan["response_paths"][1], root),
        "endpoint_responses_captured": COPULA_BEHAVIORAL_REPEATS,
        "research_only": True,
        "candidate_promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    for field, expected in expected_values.items():
        if receipt.get(field) != expected:
            raise ValueError(f"immutable Copula capture mismatch: {field}")
    if _canonical_json_bytes(plan["payload"]) != plan["request_path"].read_bytes():
        raise ValueError("immutable Copula request payload changed")
    for field, path in (
        ("request_sha256", plan["request_path"]),
        ("response_1_sha256", plan["response_paths"][0]),
        ("response_2_sha256", plan["response_paths"][1]),
    ):
        if not path.is_file() or _text(receipt.get(field)) != _file_hash(path):
            raise ValueError(f"immutable Copula raw artifact changed: {field}")
    for field in (
        "response_1_captured_at_utc",
        "response_2_captured_at_utc",
    ):
        if not _valid_aware_timestamp(receipt.get(field)):
            raise ValueError(f"immutable Copula response timestamp invalid: {field}")
    _validate_existing_cohort_anchor(
        root=root,
        capture_id=plan["capture_id"],
        capture_receipt_sha256=_file_hash(plan["receipt_path"]),
    )
    return receipt


def _validate_capture_identity_uniqueness(
    *, root: Path, plan: dict[str, Any]
) -> None:
    capture_root = (
        root
        / "data"
        / "raw"
        / "crypto_wizards_copula_behavioral_proofs"
    )
    if not capture_root.is_dir():
        return
    base = plan["base"]
    for receipt_path in capture_root.glob("*/receipt.json"):
        receipt = _read_json_strict(receipt_path)
        _validate_content_id(
            receipt,
            id_field="receipt_id",
            prefix="copulareceipt_",
        )
        same_identity = all(
            _text(receipt.get(field)) == _text(base.get(field))
            for field in ("pair_group_id", "orientation")
        )
        if same_identity and _text(receipt.get("capture_id")) != plan["capture_id"]:
            raise ValueError(
                "Copula source request changed after immutable identity capture"
            )


def _register_daily_attempt(
    *,
    root: Path,
    timestamp: datetime,
    execute: bool,
    plans: list[dict[str, Any]],
    calls_required: int,
    credit_state: dict[str, object],
    contract_path: Path,
    contract_receipt_path: Path,
) -> dict[str, str]:
    empty = {
        "status": "NOT_REQUIRED",
        "blocker": "",
        "attempt_id": "",
        "attempt_path": "",
        "attempt_sha256": "",
    }
    if not execute or calls_required <= 0:
        return empty
    if credit_state["status"] != "PASS":
        return {**empty, "status": "NOT_REGISTERED_CREDIT_BLOCKED"}
    path = (
        root
        / "data"
        / "research"
        / "wizard_copula_behavioral_attempts"
        / f"{timestamp.date().isoformat()}.json"
    )
    if path.exists():
        prior = _read_json_strict(path)
        _validate_content_id(
            prior,
            id_field="attempt_id",
            prefix="copulaattempt_",
        )
        return {
            "status": "BLOCKED",
            "blocker": "copula_external_attempt_already_registered_for_utc_day",
            "attempt_id": _text(prior.get("attempt_id")),
            "attempt_path": _relative(path, root),
            "attempt_sha256": _file_hash(path),
        }
    core = {
        "schema_version": ATTEMPT_SCHEMA_VERSION,
        "utc_date": timestamp.date().isoformat(),
        "registered_at_utc": timestamp.isoformat(),
        "capture_ids": sorted(
            plan["capture_id"]
            for plan in plans
            if plan["capture_state"] == "MISSING"
        ),
        "maximum_endpoint_calls": calls_required,
        "maximum_credits": calls_required * COPULA_POST_CREDIT_COST,
        "contract_path": _relative(contract_path, root),
        "contract_sha256": _file_hash(contract_path),
        "contract_receipt_path": _relative(contract_receipt_path, root),
        "contract_receipt_sha256": _file_hash(contract_receipt_path),
        "research_only": True,
        "candidate_promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    payload = {
        **core,
        "attempt_id": "copulaattempt_" + _json_hash(core)[:20],
    }
    try:
        attempt_sha256 = _write_exclusive_json(payload, path)
    except FileExistsError:
        return {
            **empty,
            "status": "BLOCKED",
            "blocker": "copula_external_attempt_concurrently_registered",
        }
    return {
        "status": "REGISTERED",
        "blocker": "",
        "attempt_id": payload["attempt_id"],
        "attempt_path": _relative(path, root),
        "attempt_sha256": attempt_sha256,
    }


def _publish_cohort_receipt(
    *,
    root: Path,
    frame: pd.DataFrame,
    expected_count: int,
    contract_path: Path,
    contract_receipt_path: Path,
    queue_path: Path,
) -> dict[str, str]:
    empty = {"cohort_id": "", "cohort_path": "", "cohort_sha256": ""}
    if (
        expected_count <= 0
        or len(frame) != expected_count
        or not frame.get(
            "endpoint_responses_captured", pd.Series(dtype=int)
        ).eq(COPULA_BEHAVIORAL_REPEATS).all()
        or not frame.get(
            "capture_receipt_sha256", pd.Series(dtype=str)
        ).map(_text).ne("").all()
    ):
        return empty
    fields = (
        "capture_id",
        "pair_group_id",
        "pair",
        "orientation",
        "series_1_sha256",
        "series_2_sha256",
        "source_backtest_request_path",
        "source_backtest_request_sha256",
        "request_path",
        "request_sha256",
        "response_1_path",
        "response_1_sha256",
        "response_1_captured_at_utc",
        "response_2_path",
        "response_2_sha256",
        "response_2_captured_at_utc",
        "capture_receipt_path",
        "capture_receipt_sha256",
        "repeatability_status",
        "input_orientation_status",
        "orientation_status",
        "behavioral_status",
        "provenance_status",
    )
    cells = [
        {field: _json_scalar(row.get(field)) for field in fields}
        for row in frame.sort_values(
            ["pair_group_id", "orientation"], kind="stable"
        ).to_dict("records")
    ]
    _validate_existing_cohort_receipts(root=root, cells=cells)
    core = {
        "schema_version": COHORT_SCHEMA_VERSION,
        "evaluation_schema_version": SCHEMA_VERSION,
        "contract_path": _relative(contract_path, root),
        "contract_sha256": _file_hash(contract_path),
        "contract_receipt_path": _relative(contract_receipt_path, root),
        "contract_receipt_sha256": _file_hash(contract_receipt_path),
        "proof_queue_path": _relative(queue_path, root),
        "proof_queue_sha256": _file_hash(queue_path),
        "cells": cells,
        "research_only": True,
        "candidate_promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    cohort_id = "copulacohort_" + _json_hash(core)[:20]
    payload = {**core, "cohort_id": cohort_id}
    path = (
        root
        / "data"
        / "research"
        / "wizard_copula_behavioral_cohorts"
        / f"{cohort_id}.json"
    )
    cohort_sha256 = _write_or_validate_immutable_json(payload, path)
    return {
        "cohort_id": cohort_id,
        "cohort_path": _relative(path, root),
        "cohort_sha256": cohort_sha256,
    }


def _validate_existing_cohort_receipts(
    *, root: Path, cells: list[dict[str, Any]]
) -> None:
    expected_hashes = {
        _text(cell.get("capture_id")): _text(
            cell.get("capture_receipt_sha256")
        )
        for cell in cells
    }
    cohort_root = (
        root / "data" / "research" / "wizard_copula_behavioral_cohorts"
    )
    if not cohort_root.is_dir():
        return
    for path in cohort_root.glob("copulacohort_*.json"):
        payload = _read_json_strict(path)
        _validate_content_id(
            payload,
            id_field="cohort_id",
            prefix="copulacohort_",
        )
        if path.stem != _text(payload.get("cohort_id")):
            raise ValueError("immutable Copula cohort filename mismatch")
        for prior in payload.get("cells", []):
            capture_id = _text(prior.get("capture_id"))
            if (
                capture_id in expected_hashes
                and _text(prior.get("capture_receipt_sha256"))
                != expected_hashes[capture_id]
            ):
                raise ValueError("immutable Copula capture receipt anchor changed")


def _validate_existing_cohort_anchor(
    *, root: Path, capture_id: str, capture_receipt_sha256: str
) -> None:
    _validate_existing_cohort_receipts(
        root=root,
        cells=[
            {
                "capture_id": capture_id,
                "capture_receipt_sha256": capture_receipt_sha256,
            }
        ],
    )


def _capture_identity(
    *,
    root: Path,
    plan: dict[str, Any],
    execute: bool,
    api_key: str,
    fetcher: Callable[..., dict[str, Any]],
    credit_state: dict[str, object],
) -> tuple[dict[str, object], int, bool, bool]:
    base = dict(plan["base"])
    if plan["capture_state"] == "COMPLETE":
        return _evaluate_completed_capture(root=root, plan=plan), 0, False, True
    if plan["capture_state"] == "BLOCKED":
        return {
            **base,
            "behavioral_status": "BLOCKED",
            "blocker": plan["blocker"],
        }, 0, False, False
    if not execute:
        return {
            **base,
            "behavioral_status": "PLANNED",
            "blocker": "execution_not_requested",
        }, 0, False, False
    if credit_state["status"] != "PASS":
        return {
            **base,
            "behavioral_status": "BLOCKED",
            "blocker": _text(credit_state["blocker"]),
        }, 0, False, False
    request = plan["request"]
    payload = plan["payload"]
    request_path = plan["request_path"]
    response_paths = plan["response_paths"]
    receipt_path = plan["receipt_path"]
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    if any(path.exists() for path in (request_path, *response_paths, receipt_path)):
        return {
            **base,
            "behavioral_status": "BLOCKED",
            "blocker": "copula_capture_partial_or_concurrent_artifacts_present",
        }, 0, False, False
    _write_or_validate_immutable_json(payload, request_path)
    base["request_sha256"] = _file_hash(request_path)
    responses: list[dict[str, Any]] = []
    response_times: list[str] = []
    calls = 0
    for response_number, response_path in enumerate(response_paths, start=1):
        try:
            calls += 1
            response = fetcher(request, api_key=api_key)
            _write_or_validate_immutable_json(response, response_path)
            base[f"response_{response_number}_sha256"] = _file_hash(response_path)
            captured_at = datetime.now(UTC).isoformat()
            base[f"response_{response_number}_captured_at_utc"] = captured_at
            response_times.append(captured_at)
            responses.append(response)
        except Exception as exc:  # noqa: BLE001 - preserve bounded research failure
            return {
                **base,
                "endpoint_responses_captured": len(responses),
                "behavioral_status": "FAIL",
                "blocker": f"copula_endpoint_request_failed:{safe_exception_code(exc)}",
            }, calls, False, False
    receipt = _build_capture_receipt(
        root=root,
        plan=plan,
        response_times=response_times,
    )
    receipt_sha256 = _write_or_validate_immutable_json(receipt, receipt_path)
    plan["receipt"] = receipt
    plan["receipt_sha256"] = receipt_sha256
    plan["responses"] = responses
    base["capture_receipt_sha256"] = receipt_sha256
    return _evaluate_completed_capture(root=root, plan=plan), calls, True, False


def _evaluate_completed_capture(
    *, root: Path, plan: dict[str, Any]
) -> dict[str, object]:
    base = dict(plan["base"])
    receipt = plan.get("receipt") or _load_and_validate_capture_receipt(
        root=root, plan=plan
    )
    responses = plan.get("responses") or [
        _read_json_strict(path) for path in plan["response_paths"]
    ]
    base.update(
        {
            "capture_receipt_sha256": _file_hash(plan["receipt_path"]),
            "request_sha256": _text(receipt.get("request_sha256")),
            "response_1_sha256": _text(receipt.get("response_1_sha256")),
            "response_1_captured_at_utc": _text(
                receipt.get("response_1_captured_at_utc")
            ),
            "response_2_sha256": _text(receipt.get("response_2_sha256")),
            "response_2_captured_at_utc": _text(
                receipt.get("response_2_captured_at_utc")
            ),
        }
    )
    parsed = [_parse_response(response) for response in responses]
    if any(item is None for item in parsed):
        return {
            **base,
            "endpoint_responses_captured": len(responses),
            "behavioral_status": "FAIL",
            "provenance_status": "PASS",
            "blocker": "copula_endpoint_response_schema_invalid",
        }
    first, repeat = parsed
    assert first is not None and repeat is not None
    repeat_delta = max(
        abs(first["u1_given_u2"] - repeat["u1_given_u2"]),
        abs(first["u2_given_u1"] - repeat["u2_given_u1"]),
    )
    repeat_ok = (
        first["copula_name"] == repeat["copula_name"]
        and repeat_delta <= TOLERANCE
    )
    return {
        **base,
        "endpoint_responses_captured": len(responses),
        "copula_name": first["copula_name"],
        "u1_given_u2": first["u1_given_u2"],
        "u2_given_u1": first["u2_given_u1"],
        "repeat_copula_name": repeat["copula_name"],
        "repeat_u1_given_u2": repeat["u1_given_u2"],
        "repeat_u2_given_u1": repeat["u2_given_u1"],
        "repeatability_max_abs_delta": repeat_delta,
        "repeatability_status": "PASS" if repeat_ok else "FAIL",
        "provenance_status": "PASS",
        "behavioral_status": "PENDING_ORIENTATION" if repeat_ok else "FAIL",
        "blocker": "" if repeat_ok else "copula_endpoint_repeatability_failed",
    }


def _apply_orientation_checks(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    output = frame.copy()
    for group_id, group in output.groupby("pair_group_id", dropna=False):
        original = group.loc[group["orientation"].eq("original")]
        reverse = group.loc[group["orientation"].eq("reverse")]
        if len(original) != 1 or len(reverse) != 1:
            for index in group.index:
                if output.at[index, "behavioral_status"] == "PENDING_ORIENTATION":
                    output.at[index, "behavioral_status"] = "FAIL"
                    output.at[index, "input_orientation_status"] = "FAIL"
                    output.at[index, "orientation_status"] = "FAIL"
                    output.at[index, "blocker"] = (
                        "copula_orientation_pair_incomplete_or_duplicate"
                    )
            continue
        original_index = original.index[0]
        reverse_index = reverse.index[0]
        if not all(
            output.at[index, "repeatability_status"] == "PASS"
            for index in (original_index, reverse_index)
        ):
            continue
        exact_input_swap = (
            output.at[original_index, "series_1_sha256"]
            == output.at[reverse_index, "series_2_sha256"]
            and output.at[original_index, "series_2_sha256"]
            == output.at[reverse_index, "series_1_sha256"]
            and int(output.at[original_index, "series_length"])
            == int(output.at[reverse_index, "series_length"])
        )
        for index in (original_index, reverse_index):
            output.at[index, "input_orientation_status"] = (
                "PASS" if exact_input_swap else "FAIL"
            )
        if not exact_input_swap:
            for index in (original_index, reverse_index):
                output.at[index, "orientation_status"] = "FAIL"
                output.at[index, "behavioral_status"] = "FAIL"
                output.at[index, "blocker"] = (
                    "copula_input_orientation_not_exact_swap"
                )
            continue
        family_match = (
            output.at[original_index, "copula_name"]
            == output.at[reverse_index, "copula_name"]
        )
        u1_delta = abs(
            float(output.at[original_index, "u1_given_u2"])
            - float(output.at[reverse_index, "u2_given_u1"])
        )
        u2_delta = abs(
            float(output.at[original_index, "u2_given_u1"])
            - float(output.at[reverse_index, "u1_given_u2"])
        )
        orientation_ok = family_match and max(u1_delta, u2_delta) <= TOLERANCE
        for index in (original_index, reverse_index):
            output.at[index, "orientation_status"] = (
                "PASS" if orientation_ok else "FAIL"
            )
            output.at[index, "behavioral_status"] = (
                "PASS" if orientation_ok else "FAIL"
            )
            if not orientation_ok:
                output.at[index, "blocker"] = "copula_endpoint_orientation_symmetry_failed"
    return output


def _parse_response(response: dict[str, Any]) -> dict[str, object] | None:
    family = _text(response.get("copula_name"))
    u1 = _float(response.get("u1_given_u2"))
    u2 = _float(response.get("u2_given_u1"))
    if not family or u1 is None or u2 is None:
        return None
    if not 0.0 <= u1 <= 1.0 or not 0.0 <= u2 <= 1.0:
        return None
    return {"copula_name": family, "u1_given_u2": u1, "u2_given_u1": u2}


def _credit_preflight(
    *,
    execute: bool,
    api_key: str,
    calls_required: int,
    credits_fetcher: Callable[..., dict[str, Any] | list[Any]],
    daily_credit_limit: int,
    reserved_credits: int,
) -> dict[str, object]:
    if not execute:
        return {"status": "NOT_REQUESTED", "blocker": ""}
    if calls_required <= 0:
        return {"status": "PASS", "blocker": ""}
    if not api_key:
        return {"status": "BLOCKED", "blocker": "CRYPTO_WIZARDS_API_KEY_missing"}
    try:
        usage = parse_wizard_credit_usage(
            credits_fetcher(api_key=api_key), configured_limit=daily_credit_limit
        )
    except Exception as exc:  # noqa: BLE001 - credit uncertainty blocks calls
        return {"status": "BLOCKED", "blocker": f"credit_preflight_failed:{safe_exception_code(exc)}"}
    if not usage.known:
        return {"status": "BLOCKED", "blocker": "credit_usage_unknown"}
    required = calls_required * COPULA_POST_CREDIT_COST
    available = max(int(usage.remaining or 0) - reserved_credits, 0)
    if required > available:
        return {"status": "BLOCKED", "blocker": "insufficient_credits_after_reserve"}
    return {"status": "PASS", "blocker": ""}


def _validate_contract(
    *, root: Path, contract: dict[str, Any], contract_path: Path, receipt: dict[str, Any]
) -> None:
    if not contract or not receipt:
        raise ValueError("Copula behavioral contract and receipt are required")
    if _text(receipt.get("contract_path")) != _relative(contract_path, root):
        raise ValueError("Copula behavioral contract path mismatch")
    if _text(receipt.get("contract_sha256")) != _file_hash(contract_path):
        raise ValueError("Copula behavioral contract hash mismatch")
    for path_field, hash_field in (
        ("proof_queue_path", "proof_queue_sha256"),
        ("formula_comparator_contract_path", "formula_comparator_contract_sha256"),
        ("formula_comparator_receipt_path", "formula_comparator_receipt_sha256"),
    ):
        bound_path = _resolve(root, receipt.get(path_field))
        if bound_path is None:
            raise ValueError(f"Copula behavioral binding missing: {path_field}")
        if _text(receipt.get(hash_field)) != _file_hash(bound_path):
            raise ValueError(f"Copula behavioral binding hash mismatch: {path_field}")
    if not _truthy(receipt.get("registered_before_current_queue_vendor_responses")):
        raise ValueError("Copula behavioral contract was not preregistered")
    if _truthy(contract.get("threshold_changes_after_vendor_response_allowed")):
        raise ValueError("Copula behavioral thresholds cannot change after response")
    forbidden = (
        "candidate_promotion_authority",
        "testnet_order_authority",
        "live_trading_authorized",
    )
    if any(_truthy(contract.get(key)) or _truthy(receipt.get(key)) for key in forbidden):
        raise ValueError("Copula behavioral contract cannot grant authority")


def _identity(row: pd.Series) -> tuple[str, str, str]:
    return (
        _text(row.get("pair_group_id")),
        _text(row.get("exact_mode")),
        _text(row.get("orientation")),
    )


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _read_json_strict(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid immutable JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"immutable JSON artifact must be an object: {path}")
    return payload


def _resolve(root: Path, value: object) -> Path | None:
    text = _text(value)
    if not text:
        return None
    path = Path(text)
    path = path if path.is_absolute() else root / path
    return path if path.is_file() else None


def _file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(f"{path.suffix}.tmp")
    frame.to_csv(temp, index=False)
    promote_staged_file(temp, path)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(f"{path.suffix}.tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    promote_staged_file(temp, path)


def _canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _json_hash(payload: dict[str, Any]) -> str:
    return sha256(_canonical_json_bytes(payload)).hexdigest()


def _write_or_validate_immutable_json(
    payload: dict[str, Any], path: Path
) -> str:
    expected = _canonical_json_bytes(payload)
    expected_hash = sha256(expected).hexdigest()
    if path.is_file():
        if _file_hash(path) != expected_hash:
            raise ValueError(f"immutable JSON artifact changed: {path}")
        return expected_hash
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(expected)
    promote_staged_file(temporary, path)
    return expected_hash


def _write_exclusive_json(payload: dict[str, Any], path: Path) -> str:
    expected = _canonical_json_bytes(payload)
    expected_hash = sha256(expected).hexdigest()
    create_exclusive_bytes(path, expected)
    return expected_hash


def _validate_content_id(
    payload: dict[str, Any], *, id_field: str, prefix: str
) -> None:
    material = {key: value for key, value in payload.items() if key != id_field}
    expected = prefix + _json_hash(material)[:20]
    if _text(payload.get(id_field)) != expected:
        raise ValueError(f"immutable content identity mismatch: {id_field}")


def _valid_aware_timestamp(value: object) -> bool:
    text = _text(value)
    if not text:
        return False
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return False
    return parsed.tzinfo is not None


def _json_scalar(value: Any) -> Any:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _safe_name(value: object) -> str:
    text = "".join(char if char.isalnum() else "_" for char in _text(value).lower())
    return text.strip("_") or "unknown"


def _text(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


def _float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return _text(value).lower() in {"1", "true", "yes", "y"}


def _as_utc(value: datetime | None) -> datetime:
    value = value or datetime.now(UTC)
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


if __name__ == "__main__":
    result = run_copula_behavioral_proofs()
    print(json.dumps({"summary": result.summary}, indent=2))
