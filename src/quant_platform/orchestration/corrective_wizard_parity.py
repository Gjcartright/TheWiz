"""Crypto Wizards exact-mode capture accounting and fail-closed parity authority."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_platform.active_pipeline import CommandResult
from quant_platform.wizard_symbols import (
    normalize_wizard_exchange,
    normalize_wizard_symbol,
)

ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "thewiz.wizard_exact_mode_parity.v1"
MODE_EVIDENCE_SCHEMA_VERSION = "thewiz.wizard_mode_evidence_completion.v1"
PAIR_PAGE_EXACT_MODES = (
    "Copula",
    "Dyn (Spread)",
    "Dyn (ZScoreR)",
    "OU (Spread)",
    "OU (ZScoreR)",
    "Static (Spread)",
    "Static (ZScoreR)",
)
EXACT_MODES = PAIR_PAGE_EXACT_MODES
ORIENTATIONS = ("original", "reverse")
WIZARD_PRESCANNED_CONTRACT_URL = "https://api.cryptowizards.net/docsv1beta/prescanned-get.mdx/"
WIZARD_BACKTEST_CONTRACT_URL = "https://api.cryptowizards.net/docsv1beta/backtest-post.mdx/"
WIZARD_SPREAD_CONTRACT_URL = "https://api.cryptowizards.net/docsv1beta/spread-post.mdx/"
WIZARD_COPULA_CONTRACT_URL = "https://api.cryptowizards.net/docsv1beta/copula-post.mdx/"
WIZARD_PRESCANNED_DIRECTION_FIELDS = (
    "symbol_1",
    "symbol_2",
    "spread_type",
    "strategy",
    "ou_optimal",
)
FIXTURE_FIELDS = (
    "pair_group_key",
    "pair",
    "wizard_exchange",
    "timeframe",
    "asset_x",
    "asset_y",
    "exact_mode",
    "orientation",
    "capture_timestamp",
    "periods_analyzed",
    "entry_long",
    "entry_short",
    "exit_long",
    "exit_short",
    "rolling_window",
    "close_n_periods",
    "stop_loss_pct",
    "x_weighting",
    "wizard_commission_pct",
    "wizard_slippage_pct",
    "hedge_ratio",
    "hurst",
    "half_life",
    "pearson_returns",
    "spearman_returns",
    "kendall_returns",
    "conditional_chart_value",
    "copula_family",
    "copula_correlation",
    "u1_given_u2",
    "u2_given_u1",
    "ou_mu",
    "ou_alpha",
    "ou_beta",
    "ou_b",
    "ou_sigma",
    "sharpe",
    "sortino",
    "returns_total",
    "closed_trades",
    "max_drawdown",
    "var_99",
    "cvar_99",
    "evidence_path",
)


def compare_series(
    expected: np.ndarray, observed: np.ndarray, *, tolerance: float = 1e-9
) -> dict[str, Any]:
    expected = np.asarray(expected, dtype=float)
    observed = np.asarray(observed, dtype=float)
    if expected.shape != observed.shape or expected.size == 0:
        return {
            "status": "BLOCKED",
            "max_abs_delta": math.inf,
            "blocker": "shape_or_empty_mismatch",
        }
    delta = np.abs(expected - observed)
    if not np.isfinite(delta).all():
        return {"status": "BLOCKED", "max_abs_delta": math.inf, "blocker": "nonfinite_series"}
    maximum = float(delta.max())
    return {
        "status": "PASS" if maximum <= tolerance else "FAIL",
        "max_abs_delta": maximum,
        "blocker": "" if maximum <= tolerance else "parity_tolerance_exceeded",
    }


def build_wizard_mode_evidence_completion(*, root: Path = ROOT) -> dict[str, Any]:
    """Combine formula and preregistered Copula evidence without conflating them."""

    from quant_platform.wizard_hyperliquid_mode_proof import (
        count_completed_exact_mode_proofs,
    )

    active = root / "reports" / "active"
    queue_path = active / "exhaustive_wizard_exact_mode_proof_queue.csv"
    proof_path = active / "hyperliquid_wizard_vendor_mode_proofs.csv"
    queue = _read_csv(queue_path)
    eligible = queue.loc[
        queue.get("vendor_custom_series_eligible", pd.Series(False, index=queue.index)).map(_truthy)
    ].copy()
    formula_expected = int(
        eligible.get("exact_mode", pd.Series("", index=eligible.index)).ne("Copula").sum()
    )
    formula_passed = count_completed_exact_mode_proofs(
        queue_path=queue_path,
        proof_path=proof_path,
    )
    copula_expected_frame = eligible.loc[
        eligible.get("exact_mode", pd.Series("", index=eligible.index)).eq("Copula")
    ].copy()
    copula_queue_path = queue_path
    copula_proof_path = proof_path
    v2_queue_path = active / "exhaustive_wizard_copula_proof_queue_v2.csv"
    v2_proof_path = active / "wizard_copula_source_ledger_v2.csv"
    if (
        (root / "config" / "wizard_copula_behavioral_parity_v2.json").is_file()
        and v2_queue_path.is_file()
        and v2_proof_path.is_file()
    ):
        copula_queue_path = v2_queue_path
        copula_proof_path = v2_proof_path
        v2_queue = _read_csv(v2_queue_path)
        copula_expected_frame = v2_queue.loc[
            v2_queue.get("exact_mode", pd.Series("", index=v2_queue.index)).eq("Copula")
            & v2_queue.get(
                "vendor_custom_series_eligible",
                pd.Series(False, index=v2_queue.index),
            ).map(_truthy)
        ].copy()
    copula = _copula_behavioral_evidence_gate(
        root=root,
        expected=copula_expected_frame,
        queue_path=copula_queue_path,
        proof_path=copula_proof_path,
    )
    ou_expected_cells = int(
        eligible.get("exact_mode", pd.Series("", index=eligible.index))
        .isin(("OU (Spread)", "OU (ZScoreR)"))
        .sum()
    )
    ou = _active_ou_evidence_gate(root=root, expected_cells=ou_expected_cells)
    ou_evidence_generation = int(ou.get("evidence_generation", ou["comparator_generation"]))
    formula_complete = formula_expected > 0 and formula_passed == formula_expected
    complete = (
        formula_complete and copula["status"] == "PASS" and ou["status"] in {"PASS", "NOT_REQUIRED"}
    )
    blockers: list[str] = []
    if not formula_complete:
        blockers.append(
            "vendor_formula_parity_unproven_for_"
            f"{max(formula_expected - formula_passed, 0)}_of_"
            f"{formula_expected}_non_copula_queue_cells"
        )
    if copula["status"] != "PASS":
        blockers.append(str(copula["blocker"]))
    if ou["status"] not in {"PASS", "NOT_REQUIRED"}:
        blockers.append(str(ou["blocker"]))
    return {
        "mode_evidence_schema_version": MODE_EVIDENCE_SCHEMA_VERSION,
        "status": "PASS" if complete else "BLOCKED",
        "formula_queue_cells_expected": formula_expected,
        "formula_queue_cells_passed": formula_passed,
        "copula_queue_cells_expected": int(copula["expected_cells"]),
        "copula_behavioral_cells_passed": int(copula["passed_cells"]),
        "copula_provenance_cells_complete": int(copula["provenance_cells"]),
        "copula_behavioral_status": str(copula["status"]),
        "copula_formula_parity_proven": False,
        "ou_status": str(ou["status"]),
        "ou_generation": ou_evidence_generation,
        "ou_active_comparator_generation": int(ou["comparator_generation"]),
        "ou_formula_cells_passed": int(ou["formula_cells_passed"]),
        "ou_selector_cells_passed": int(ou["selector_cells_passed"]),
        "ou_transform_selector_cells_passed": int(ou.get("transform_selector_cells_passed", 0)),
        "ou_trend_selector_cells_passed": int(
            ou.get("trend_selector_cells_passed", ou["selector_cells_passed"])
        ),
        "ou_local_selector_proven": bool(ou["local_selector_proven"]),
        "ou_activation_status": str(ou["activation_status"]),
        "ou_proof_refresh_status": str(ou["proof_refresh_status"]),
        "ou_exact_rows": int(ou["exact_rows"]),
        "ou_v3_status": (
            str(ou["status"])
            if ou_evidence_generation == 3
            else f"SUPERSEDED_BY_OU_V{ou_evidence_generation}"
        ),
        "ou_v3_formula_cells_passed": (
            int(ou["formula_cells_passed"]) if ou_evidence_generation == 3 else 0
        ),
        "ou_v3_selector_cells_passed": (
            int(ou["selector_cells_passed"]) if ou_evidence_generation == 3 else 0
        ),
        "ou_v3_local_selector_proven": bool(
            ou["local_selector_proven"] and ou_evidence_generation == 3
        ),
        "ou_v3_activation_status": (
            str(ou["activation_status"])
            if ou_evidence_generation == 3
            else f"SUPERSEDED_BY_OU_V{ou_evidence_generation}"
        ),
        "ou_v3_comparator_generation": (
            int(ou["comparator_generation"]) if ou_evidence_generation == 3 else 3
        ),
        "ou_v3_proof_refresh_status": (
            str(ou["proof_refresh_status"])
            if ou_evidence_generation == 3
            else f"SUPERSEDED_BY_OU_V{ou_evidence_generation}"
        ),
        "ou_v3_exact_rows": (int(ou["exact_rows"]) if ou_evidence_generation == 3 else 0),
        "ou_v4_status": (
            str(ou["status"])
            if ou_evidence_generation == 4
            else f"SUPERSEDED_BY_OU_V{ou_evidence_generation}"
            if ou_evidence_generation > 4
            else "NOT_REGISTERED"
        ),
        "ou_v4_formula_cells_passed": (
            int(ou["formula_cells_passed"]) if ou_evidence_generation == 4 else 0
        ),
        "ou_v4_transform_selector_cells_passed": (
            int(ou.get("transform_selector_cells_passed", 0)) if ou_evidence_generation == 4 else 0
        ),
        "ou_v4_trend_selector_cells_passed": (
            int(ou.get("trend_selector_cells_passed", 0)) if ou_evidence_generation == 4 else 0
        ),
        "ou_v4_activation_status": (
            str(ou["activation_status"])
            if ou_evidence_generation == 4
            else f"SUPERSEDED_BY_OU_V{ou_evidence_generation}"
            if ou_evidence_generation > 4
            else "NOT_REGISTERED"
        ),
        "ou_v4_proof_refresh_status": (
            str(ou["proof_refresh_status"])
            if ou_evidence_generation == 4
            else f"SUPERSEDED_BY_OU_V{ou_evidence_generation}"
            if ou_evidence_generation > 4
            else "NOT_REGISTERED"
        ),
        "ou_v4_exact_rows": (int(ou["exact_rows"]) if int(ou["comparator_generation"]) == 4 else 0),
        "ou_v5_status": (
            str(ou["status"])
            if ou_evidence_generation == 5
            else "SUPERSEDED_BY_OU_V6"
            if ou_evidence_generation > 5
            else "NOT_REGISTERED"
        ),
        "ou_v5_formula_cells_passed": (
            int(ou["formula_cells_passed"]) if ou_evidence_generation == 5 else 0
        ),
        "ou_v5_transform_selector_cells_passed": (
            int(ou.get("transform_selector_cells_passed", 0)) if ou_evidence_generation == 5 else 0
        ),
        "ou_v5_trend_selector_cells_passed": (
            int(ou.get("trend_selector_cells_passed", 0)) if ou_evidence_generation == 5 else 0
        ),
        "ou_v5_profile_branch_selector_cells_passed": (
            int(ou.get("profile_branch_selector_cells_passed", 0))
            if ou_evidence_generation == 5
            else 0
        ),
        "ou_v5_activation_status": (
            str(ou["activation_status"])
            if ou_evidence_generation == 5
            else "SUPERSEDED_BY_OU_V6"
            if ou_evidence_generation > 5
            else "NOT_REGISTERED"
        ),
        "ou_v5_proof_refresh_status": (
            str(ou["proof_refresh_status"])
            if ou_evidence_generation == 5
            else "SUPERSEDED_BY_OU_V6"
            if ou_evidence_generation > 5
            else "NOT_REGISTERED"
        ),
        "ou_v5_exact_rows": (int(ou["exact_rows"]) if ou_evidence_generation == 5 else 0),
        "ou_v6_status": (str(ou["status"]) if ou_evidence_generation == 6 else "NOT_REGISTERED"),
        "ou_v6_formula_cells_passed": (
            int(ou["formula_cells_passed"]) if ou_evidence_generation == 6 else 0
        ),
        "ou_v6_transform_selector_cells_passed": (
            int(ou.get("transform_selector_cells_passed", 0)) if ou_evidence_generation == 6 else 0
        ),
        "ou_v6_trend_selector_cells_passed": (
            int(ou.get("trend_selector_cells_passed", 0)) if ou_evidence_generation == 6 else 0
        ),
        "ou_v6_profile_branch_selector_cells_passed": (
            int(ou.get("profile_branch_selector_cells_passed", 0))
            if ou_evidence_generation == 6
            else 0
        ),
        "ou_v6_activation_status": (
            str(ou["activation_status"]) if ou_evidence_generation == 6 else "NOT_REGISTERED"
        ),
        "ou_v6_proof_refresh_status": (
            str(ou["proof_refresh_status"]) if ou_evidence_generation == 6 else "NOT_REGISTERED"
        ),
        "ou_v6_exact_rows": (int(ou["exact_rows"]) if ou_evidence_generation == 6 else 0),
        "ou_v6_terminal_failure": bool(
            ou.get("terminal_failure", False) and ou_evidence_generation == 6
        ),
        "ou_v6_successor_after_failure_allowed": bool(
            ou.get("successor_after_failure_allowed", False)
            if ou_evidence_generation == 6
            else False
        ),
        "accepted_queue_cells": formula_passed + int(copula["passed_cells"]),
        "required_queue_cells": formula_expected + int(copula["expected_cells"]),
        "copula_orientations_passed": list(copula["orientations_passed"]),
        "blocker": ";".join(blockers),
        "promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }


def _active_ou_evidence_gate(*, root: Path, expected_cells: int) -> dict[str, Any]:
    active = root / "reports" / "active"
    v6_registered = bool(
        (root / "config" / "wizard_ou_comparator_v6_holdout.json").is_file()
        or (active / "wizard_ou_v6_holdout_receipt.json").is_file()
    )
    if v6_registered:
        return _ou_v6_evidence_gate(root=root, expected_cells=expected_cells)
    v5_registered = bool(
        (root / "config" / "wizard_ou_comparator_v5_holdout.json").is_file()
        or (active / "wizard_ou_v5_holdout_receipt.json").is_file()
    )
    if v5_registered:
        return _ou_v5_evidence_gate(root=root, expected_cells=expected_cells)
    v4_registered = bool(
        (root / "config" / "wizard_ou_comparator_v4_holdout.json").is_file()
        or (active / "wizard_ou_v4_holdout_receipt.json").is_file()
    )
    if v4_registered:
        return _ou_v4_evidence_gate(root=root, expected_cells=expected_cells)
    return _ou_v3_evidence_gate(root=root, expected_cells=expected_cells)


def _ou_v6_evidence_gate(*, root: Path, expected_cells: int) -> dict[str, Any]:
    """Use only manifest-bound scheduler evidence for the final OU-v6 generation."""

    if expected_cells == 0:
        return {
            "status": "NOT_REQUIRED",
            "formula_cells_passed": 0,
            "selector_cells_passed": 0,
            "transform_selector_cells_passed": 0,
            "trend_selector_cells_passed": 0,
            "profile_branch_selector_cells_passed": 0,
            "local_selector_proven": False,
            "activation_status": "NOT_REQUIRED",
            "evidence_generation": 6,
            "comparator_generation": 1,
            "proof_refresh_status": "NOT_REQUIRED",
            "exact_rows": 0,
            "terminal_failure": False,
            "successor_after_failure_allowed": False,
            "blocker": "",
        }

    from quant_platform.orchestration.corrective_wizard_ou_v6_holdout import (
        validate_ou_v6_stage3_evidence,
    )

    active = root / "reports" / "active"
    scheduler = _read_json(active / "corrective_wizard_proof_scheduler_status.json")
    status = _read_json(active / "wizard_ou_v6_holdout_status.json")
    validation = validate_ou_v6_stage3_evidence(root=root, evidence=scheduler)
    blockers = [
        _text(value) for value in validation.get("blockers", []) if _text(value)
    ]

    formula_cells = int(status.get("formula_parity_passed_cells", 0) or 0)
    transform_cells = int(status.get("transform_selector_parity_passed_cells", 0) or 0)
    trend_cells = int(status.get("trend_selector_parity_passed_cells", 0) or 0)
    profile_cells = int(status.get("profile_branch_selector_parity_passed_cells", 0) or 0)
    activation_status = _text(
        scheduler.get(
            "ou_v6_activation_status",
            "NOT_AUTHORIZED_PENDING_PASS_AND_EXPLICIT_REVIEW",
        )
    )
    comparator_generation = int(scheduler.get("ou_v6_comparator_generation", 1) or 1)
    proof_refresh_status = _text(scheduler.get("ou_v6_proof_refresh_status"))
    exact_rows = int(scheduler.get("ou_v6_proofs_refreshed", 0) or 0)
    terminal_failure = bool(
        scheduler.get("ou_v6_terminal_failure", False) or status.get("status") == "FAIL"
    )
    successor_allowed = bool(scheduler.get("ou_v6_successor_after_failure_allowed", False))
    if terminal_failure:
        blockers.extend(
            [
                "ou_v6_terminal_failure_exact_local_ou_parity_rejected",
                "ou_v6_successor_prohibited_after_terminal_failure",
            ]
        )
    if successor_allowed:
        blockers.append("ou_v6_terminal_policy_violation_successor_allowed")
    if expected_cells != 8:
        blockers.append("ou_v6_current_queue_cell_count_mismatch")
    if any(
        _truthy(payload.get(key))
        for payload in (scheduler, status)
        for key in (
            "candidate_promotion_authority",
            "order_submission_included",
            "testnet_order_authority",
            "live_trading_authorized",
        )
    ):
        blockers.append("ou_v6_evidence_has_forbidden_authority")

    blockers = list(dict.fromkeys(blockers))
    local_selector_proven = bool(
        transform_cells == 8 and trend_cells == 8 and profile_cells == 8
    )
    return {
        "status": "PASS" if not blockers else "BLOCKED",
        "formula_cells_passed": formula_cells,
        "selector_cells_passed": min(transform_cells, trend_cells, profile_cells),
        "transform_selector_cells_passed": transform_cells,
        "trend_selector_cells_passed": trend_cells,
        "profile_branch_selector_cells_passed": profile_cells,
        "local_selector_proven": local_selector_proven,
        "activation_status": activation_status,
        "evidence_generation": 6,
        "comparator_generation": comparator_generation,
        "proof_refresh_status": proof_refresh_status,
        "exact_rows": exact_rows,
        "terminal_failure": terminal_failure,
        "successor_after_failure_allowed": successor_allowed,
        "blocker": ";".join(blockers),
    }


def _ou_v5_evidence_gate(*, root: Path, expected_cells: int) -> dict[str, Any]:
    if expected_cells == 0:
        return {
            "status": "NOT_REQUIRED",
            "formula_cells_passed": 0,
            "selector_cells_passed": 0,
            "transform_selector_cells_passed": 0,
            "trend_selector_cells_passed": 0,
            "profile_branch_selector_cells_passed": 0,
            "local_selector_proven": False,
            "activation_status": "NOT_REQUIRED",
            "evidence_generation": 5,
            "comparator_generation": 1,
            "proof_refresh_status": "NOT_REQUIRED",
            "exact_rows": 0,
            "blocker": "",
        }

    from quant_platform.orchestration.corrective_wizard_ou_v5_holdout import (
        _implementation_source_hash,
        validate_ou_v5_evaluation_binding,
    )
    from quant_platform.wizard_ou_v5_comparator_activation import (
        load_validated_ou_v5_activation,
    )

    active = root / "reports" / "active"
    receipt_path = active / "wizard_ou_v5_holdout_receipt.json"
    contract_path = root / "config" / "wizard_ou_comparator_v5_holdout.json"
    status_path = active / "wizard_ou_v5_holdout_status.json"
    detail_path = active / "wizard_ou_v5_holdout_evaluation.csv"
    activation_path = active / "wizard_ou_v5_activation_status.json"
    refresh_path = active / "wizard_ou_v5_proof_refresh_status.json"
    receipt = _read_json(receipt_path)
    contract = _read_json(contract_path)
    status = _read_json(status_path)
    detail = _read_csv(detail_path)
    activation = _read_json(activation_path)
    refresh = _read_json(refresh_path)
    blockers: list[str] = []
    implementation_hash = _implementation_source_hash()

    if receipt.get("status") != "PREREGISTERED_WAITING_VENDOR_RESPONSES":
        blockers.append("ou_v5_receipt_missing_or_invalid")
    if not _truthy(receipt.get("prospectively_registered")):
        blockers.append("ou_v5_not_prospectively_registered")
    if int(receipt.get("vendor_responses_at_registration", -1)) != 0:
        blockers.append("ou_v5_vendor_responses_existed_at_registration")
    if int(receipt.get("required_cells", 0) or 0) != 8:
        blockers.append("ou_v5_registered_cell_count_mismatch")
    if not contract_path.is_file() or _text(receipt.get("contract_sha256")) != _file_hash(
        contract_path
    ):
        blockers.append("ou_v5_contract_hash_mismatch")
    if _text(receipt.get("implementation_source_sha256")) != implementation_hash:
        blockers.append("ou_v5_receipt_source_hash_mismatch")
    if _text(contract.get("implementation", {}).get("source_sha256")) != implementation_hash:
        blockers.append("ou_v5_contract_source_hash_mismatch")

    formula_cells = int(status.get("formula_parity_passed_cells", 0) or 0)
    transform_cells = int(status.get("transform_selector_parity_passed_cells", 0) or 0)
    trend_cells = int(status.get("trend_selector_parity_passed_cells", 0) or 0)
    profile_cells = int(status.get("profile_branch_selector_parity_passed_cells", 0) or 0)
    local_selector_proven = bool(transform_cells == 8 and trend_cells == 8 and profile_cells == 8)
    if status.get("status") != "PASS":
        blockers.append("ou_v5_prospective_holdout_not_passed")
    if formula_cells != 8:
        blockers.append("ou_v5_formula_cells_incomplete")
    if transform_cells != 8:
        blockers.append("ou_v5_transform_selector_cells_incomplete")
    if trend_cells != 8:
        blockers.append("ou_v5_trend_selector_cells_incomplete")
    if profile_cells != 8:
        blockers.append("ou_v5_profile_branch_selector_cells_incomplete")
    if len(detail) != 8:
        blockers.append("ou_v5_evaluation_cell_count_mismatch")
    elif not all(
        detail.get(column, pd.Series(False, index=detail.index)).map(_truthy).all()
        for column in (
            "formula_parity_passed",
            "transform_selector_parity_passed",
            "trend_selector_parity_passed",
            "profile_branch_selector_parity_passed",
        )
    ):
        blockers.append("ou_v5_evaluation_cells_not_all_passed")

    immutable_relative = _text(status.get("immutable_result_path"))
    immutable_path = root / immutable_relative if immutable_relative else None
    immutable_root = (root / "data/research/wizard_ou_v5_holdout_evaluations").resolve()
    immutable_binding_valid = False
    if immutable_path is not None:
        try:
            immutable_path.resolve().relative_to(immutable_root)
            immutable_binding_valid = bool(
                immutable_path.is_file()
                and immutable_path.stem == _text(status.get("result_id"))
                and _file_hash(immutable_path) == _text(status.get("immutable_result_sha256"))
            )
        except ValueError:
            immutable_binding_valid = False
    if not immutable_binding_valid:
        blockers.append("ou_v5_immutable_evaluation_binding_invalid")
    evaluation_binding = validate_ou_v5_evaluation_binding(root=root, status=status)
    if evaluation_binding.get("status") != "PASS":
        blockers.append("ou_v5_evaluation_content_binding_invalid")
        blockers.extend(
            _text(blocker) for blocker in evaluation_binding.get("blockers", []) if _text(blocker)
        )

    activation_status = _text(activation.get("status"))
    comparator_generation = (
        int(activation.get("comparator_generation", 0) or 0)
        if activation_status == "APPLIED_RESEARCH_COMPARATOR_ONLY"
        else 1
    )
    if activation_status != "APPLIED_RESEARCH_COMPARATOR_ONLY":
        blockers.append("ou_v5_reviewed_activation_not_applied")
    elif comparator_generation != 5:
        blockers.append("ou_v5_active_comparator_generation_not_five")
    else:
        try:
            if load_validated_ou_v5_activation(root=root) is None:
                blockers.append("ou_v5_active_activation_binding_invalid")
        except ValueError:
            blockers.append("ou_v5_active_activation_binding_invalid")

    proof_refresh_status = _text(refresh.get("status"))
    exact_rows = int(refresh.get("exact_ou_rows", 0) or 0)
    if proof_refresh_status != "PASS":
        blockers.append("ou_v5_activation_bound_proof_refresh_not_passed")
    if exact_rows != expected_cells:
        blockers.append("ou_v5_refreshed_queue_cell_count_mismatch")
    if _text(refresh.get("activation_id")) != _text(activation.get("activation_id")):
        blockers.append("ou_v5_refresh_activation_lineage_mismatch")
    if int(refresh.get("comparator_generation", 0) or 0) != 5:
        blockers.append("ou_v5_refresh_generation_mismatch")

    forbidden = (
        "candidate_promotion_authority",
        "testnet_order_authority",
        "live_trading_authorized",
    )
    if any(
        _truthy(payload.get(key))
        for payload in (receipt, contract, status, activation, refresh)
        for key in forbidden
    ):
        blockers.append("ou_v5_evidence_has_forbidden_authority")
    if not detail.empty and any(
        detail.get(key, pd.Series(False, index=detail.index)).map(_truthy).any()
        for key in forbidden
    ):
        blockers.append("ou_v5_evaluation_has_forbidden_authority")
    return {
        "status": "PASS" if not blockers else "BLOCKED",
        "formula_cells_passed": formula_cells,
        "selector_cells_passed": min(transform_cells, trend_cells, profile_cells),
        "transform_selector_cells_passed": transform_cells,
        "trend_selector_cells_passed": trend_cells,
        "profile_branch_selector_cells_passed": profile_cells,
        "local_selector_proven": local_selector_proven,
        "activation_status": activation_status,
        "evidence_generation": 5,
        "comparator_generation": comparator_generation,
        "proof_refresh_status": proof_refresh_status,
        "exact_rows": exact_rows,
        "blocker": ";".join(blockers),
    }


def _ou_v4_evidence_gate(*, root: Path, expected_cells: int) -> dict[str, Any]:
    if expected_cells == 0:
        return {
            "status": "NOT_REQUIRED",
            "formula_cells_passed": 0,
            "selector_cells_passed": 0,
            "transform_selector_cells_passed": 0,
            "trend_selector_cells_passed": 0,
            "local_selector_proven": False,
            "activation_status": "NOT_REQUIRED",
            "evidence_generation": 4,
            "comparator_generation": 4,
            "proof_refresh_status": "NOT_REQUIRED",
            "exact_rows": 0,
            "blocker": "",
        }

    from quant_platform.orchestration.corrective_wizard_ou_v4_holdout import (
        _implementation_source_hash,
    )
    from quant_platform.wizard_ou_v4_comparator_activation import (
        load_validated_ou_v4_activation,
    )

    active = root / "reports" / "active"
    receipt_path = active / "wizard_ou_v4_holdout_receipt.json"
    contract_path = root / "config" / "wizard_ou_comparator_v4_holdout.json"
    status_path = active / "wizard_ou_v4_holdout_status.json"
    detail_path = active / "wizard_ou_v4_holdout_evaluation.csv"
    activation_path = active / "wizard_ou_v4_activation_status.json"
    refresh_path = active / "wizard_ou_v4_proof_refresh_status.json"
    receipt = _read_json(receipt_path)
    contract = _read_json(contract_path)
    status = _read_json(status_path)
    detail = _read_csv(detail_path)
    activation = _read_json(activation_path)
    refresh = _read_json(refresh_path)
    blockers: list[str] = []
    implementation_hash = _implementation_source_hash()

    if receipt.get("status") != "REGISTERED_WAITING_VENDOR_RESPONSES":
        blockers.append("ou_v4_receipt_missing_or_invalid")
    if not _truthy(receipt.get("prospectively_registered")):
        blockers.append("ou_v4_not_prospectively_registered")
    if int(receipt.get("vendor_responses_at_registration", -1)) != 0:
        blockers.append("ou_v4_vendor_responses_existed_at_registration")
    if int(receipt.get("required_cells", 0) or 0) != 8:
        blockers.append("ou_v4_registered_cell_count_mismatch")
    if not contract_path.is_file() or _text(receipt.get("contract_sha256")) != _file_hash(
        contract_path
    ):
        blockers.append("ou_v4_contract_hash_mismatch")
    if _text(receipt.get("implementation_source_sha256")) != implementation_hash:
        blockers.append("ou_v4_receipt_source_hash_mismatch")
    if _text(contract.get("implementation", {}).get("source_sha256")) != (implementation_hash):
        blockers.append("ou_v4_contract_source_hash_mismatch")

    formula_cells = int(status.get("formula_parity_passed_cells", 0) or 0)
    transform_cells = int(status.get("transform_selector_parity_passed_cells", 0) or 0)
    trend_cells = int(status.get("trend_selector_parity_passed_cells", 0) or 0)
    local_selector_proven = bool(transform_cells == 8 and trend_cells == 8)
    if status.get("status") != "PASS":
        blockers.append("ou_v4_prospective_holdout_not_passed")
    if formula_cells != 8:
        blockers.append("ou_v4_formula_cells_incomplete")
    if transform_cells != 8:
        blockers.append("ou_v4_transform_selector_cells_incomplete")
    if trend_cells != 8:
        blockers.append("ou_v4_trend_selector_cells_incomplete")
    if len(detail) != 8:
        blockers.append("ou_v4_evaluation_cell_count_mismatch")
    elif not all(
        detail.get(column, pd.Series(False, index=detail.index)).map(_truthy).all()
        for column in (
            "formula_parity_passed",
            "transform_selector_parity_passed",
            "trend_selector_parity_passed",
        )
    ):
        blockers.append("ou_v4_evaluation_cells_not_all_passed")

    activation_status = _text(activation.get("status"))
    comparator_generation = (
        int(activation.get("comparator_generation", 0) or 0)
        if activation_status == "APPLIED_RESEARCH_COMPARATOR_ONLY"
        else 1
    )
    if activation_status != "APPLIED_RESEARCH_COMPARATOR_ONLY":
        blockers.append("ou_v4_reviewed_activation_not_applied")
    elif comparator_generation != 4:
        blockers.append("ou_v4_active_comparator_generation_not_four")
    else:
        try:
            if load_validated_ou_v4_activation(root=root) is None:
                blockers.append("ou_v4_active_activation_binding_invalid")
        except ValueError:
            blockers.append("ou_v4_active_activation_binding_invalid")

    proof_refresh_status = _text(refresh.get("status"))
    exact_rows = int(refresh.get("exact_ou_rows", 0) or 0)
    if proof_refresh_status != "PASS":
        blockers.append("ou_v4_activation_bound_proof_refresh_not_passed")
    if exact_rows != expected_cells:
        blockers.append("ou_v4_refreshed_queue_cell_count_mismatch")
    if _text(refresh.get("activation_id")) != _text(activation.get("activation_id")):
        blockers.append("ou_v4_refresh_activation_lineage_mismatch")
    if int(refresh.get("comparator_generation", 0) or 0) != 4:
        blockers.append("ou_v4_refresh_generation_mismatch")

    forbidden = (
        "candidate_promotion_authority",
        "testnet_order_authority",
        "live_trading_authorized",
    )
    if any(
        _truthy(payload.get(key))
        for payload in (receipt, contract, status, activation, refresh)
        for key in forbidden
    ):
        blockers.append("ou_v4_evidence_has_forbidden_authority")
    if not detail.empty and any(
        detail.get(key, pd.Series(False, index=detail.index)).map(_truthy).any()
        for key in forbidden
    ):
        blockers.append("ou_v4_evaluation_has_forbidden_authority")
    return {
        "status": "PASS" if not blockers else "BLOCKED",
        "formula_cells_passed": formula_cells,
        "selector_cells_passed": min(transform_cells, trend_cells),
        "transform_selector_cells_passed": transform_cells,
        "trend_selector_cells_passed": trend_cells,
        "local_selector_proven": local_selector_proven,
        "activation_status": activation_status,
        "evidence_generation": 4,
        "comparator_generation": comparator_generation,
        "proof_refresh_status": proof_refresh_status,
        "exact_rows": exact_rows,
        "blocker": ";".join(blockers),
    }


def _ou_v3_evidence_gate(*, root: Path, expected_cells: int) -> dict[str, Any]:
    if expected_cells == 0:
        return {
            "status": "NOT_REQUIRED",
            "formula_cells_passed": 0,
            "selector_cells_passed": 0,
            "local_selector_proven": False,
            "activation_status": "NOT_REQUIRED",
            "evidence_generation": 3,
            "comparator_generation": 1,
            "proof_refresh_status": "NOT_REQUIRED",
            "exact_rows": 0,
            "blocker": "",
        }

    active = root / "reports" / "active"
    receipt_path = active / "wizard_ou_trend_selector_v1_receipt.json"
    selector_contract_path = root / "config" / "wizard_ou_trend_selector_v1_holdout.json"
    status_path = active / "wizard_ou_v3_holdout_status.json"
    detail_path = active / "wizard_ou_v3_holdout_evaluation.csv"
    activation_path = active / "wizard_ou_v3_activation_status.json"
    refresh_path = active / "wizard_ou_v3_proof_refresh_status.json"
    receipt = _read_json(receipt_path)
    selector_contract = _read_json(selector_contract_path)
    status = _read_json(status_path)
    detail = _read_csv(detail_path)
    activation = _read_json(activation_path)
    refresh = _read_json(refresh_path)
    blockers: list[str] = []
    if receipt.get("status") != "REGISTERED_WAITING_VENDOR_RESPONSES":
        blockers.append("ou_v3_selector_receipt_missing_or_invalid")
    if not _truthy(receipt.get("prospectively_registered")):
        blockers.append("ou_v3_selector_not_prospectively_registered")
    if int(receipt.get("vendor_responses_at_registration", -1)) != 0:
        blockers.append("ou_v3_vendor_responses_existed_at_selector_registration")
    if int(receipt.get("derivation_matched_cells", 0) or 0) != 8:
        blockers.append("ou_v3_selector_derivation_incomplete")
    if int(receipt.get("holdout_prediction_cells", 0) or 0) != 4:
        blockers.append("ou_v3_selector_predictions_incomplete")
    if not selector_contract_path.is_file() or _text(receipt.get("contract_sha256")) != _file_hash(
        selector_contract_path
    ):
        blockers.append("ou_v3_selector_contract_hash_mismatch")

    formula_cells = int(status.get("formula_parity_passed_cells", 0) or 0)
    selector_cells = int(status.get("trend_selector_parity_passed_cells", 0) or 0)
    local_selector_proven = _truthy(status.get("local_point_in_time_trend_selector_proven"))
    if status.get("status") != "PASS":
        blockers.append("ou_v3_prospective_holdout_not_passed")
    if formula_cells != 4:
        blockers.append("ou_v3_formula_cells_incomplete")
    if selector_cells != 4 or not local_selector_proven:
        blockers.append("ou_v3_selector_cells_incomplete")
    if len(detail) != 4:
        blockers.append("ou_v3_evaluation_cell_count_mismatch")
    elif not (
        detail.get("formula_parity_passed", pd.Series(False, index=detail.index)).map(_truthy).all()
        and detail.get("trend_selector_parity_passed", pd.Series(False, index=detail.index))
        .map(_truthy)
        .all()
    ):
        blockers.append("ou_v3_evaluation_cells_not_all_passed")

    activation_status = _text(activation.get("status"))
    comparator_generation = (
        int(activation.get("comparator_generation", 0) or 0)
        if activation_status == "APPLIED_RESEARCH_COMPARATOR_ONLY"
        else 1
    )
    proof_refresh_status = _text(refresh.get("status"))
    exact_rows = int(refresh.get("exact_ou_rows", 0) or 0)
    if activation_status != "APPLIED_RESEARCH_COMPARATOR_ONLY":
        blockers.append("ou_v3_reviewed_activation_not_applied")
    if comparator_generation != 3:
        blockers.append("ou_v3_active_comparator_generation_not_three")
    if proof_refresh_status != "PASS":
        blockers.append("ou_v3_activation_bound_proof_refresh_not_passed")
    if exact_rows != expected_cells:
        blockers.append("ou_v3_refreshed_queue_cell_count_mismatch")
    if _text(refresh.get("activation_id")) != _text(activation.get("activation_id")):
        blockers.append("ou_v3_refresh_activation_lineage_mismatch")

    forbidden = (
        "candidate_promotion_authority",
        "testnet_order_authority",
        "live_trading_authorized",
    )
    if any(
        _truthy(payload.get(key))
        for payload in (receipt, selector_contract, status, activation, refresh)
        for key in forbidden
    ):
        blockers.append("ou_v3_evidence_has_forbidden_authority")
    if not detail.empty and any(
        detail.get(key, pd.Series(False, index=detail.index)).map(_truthy).any()
        for key in forbidden
    ):
        blockers.append("ou_v3_evaluation_has_forbidden_authority")
    return {
        "status": "PASS" if not blockers else "BLOCKED",
        "formula_cells_passed": formula_cells,
        "selector_cells_passed": selector_cells,
        "local_selector_proven": local_selector_proven,
        "activation_status": activation_status,
        "evidence_generation": 3,
        "comparator_generation": comparator_generation,
        "proof_refresh_status": proof_refresh_status,
        "exact_rows": exact_rows,
        "blocker": ";".join(blockers),
    }


def _copula_behavioral_evidence_gate(
    *,
    root: Path,
    expected: pd.DataFrame,
    queue_path: Path,
    proof_path: Path,
) -> dict[str, Any]:
    active = root / "reports" / "active"
    v2 = queue_path.name == "exhaustive_wizard_copula_proof_queue_v2.csv"
    status_path = active / (
        "wizard_copula_behavioral_v2_status.json" if v2 else "wizard_copula_behavioral_status.json"
    )
    detail_path = active / (
        "wizard_copula_behavioral_v2_evaluation.csv"
        if v2
        else "wizard_copula_behavioral_evaluation.csv"
    )
    contract_path = (
        root
        / "config"
        / (
            "wizard_copula_behavioral_parity_v2.json"
            if v2
            else "wizard_copula_behavioral_parity.json"
        )
    )
    receipt_path = active / (
        "wizard_copula_behavioral_v2_contract_receipt.json"
        if v2
        else "wizard_copula_behavioral_contract_receipt.json"
    )
    status = _read_json(status_path)
    detail = _read_csv(detail_path)
    expected_identities = {
        (_text(row.get("pair_group_id")), _text(row.get("orientation")))
        for _, row in expected.iterrows()
    }
    expected_count = len(expected_identities)
    blockers: list[str] = []
    if expected_count == 0:
        blockers.append("registered_copula_queue_cells_missing")
    if _text(status.get("status")) != "PASS":
        blockers.append(
            "copula_behavioral_status_" + (_text(status.get("status")) or "missing").lower()
        )
    if int(status.get("expected_cells", 0) or 0) != expected_count:
        blockers.append("copula_behavioral_expected_cell_count_mismatch")
    if int(status.get("behavioral_cells_passed", 0) or 0) != expected_count:
        blockers.append("copula_behavioral_cells_incomplete")
    if int(status.get("provenance_cells_complete", 0) or 0) != expected_count:
        blockers.append("copula_behavioral_provenance_incomplete")
    if _truthy(status.get("formula_parity_proven")):
        blockers.append("copula_behavioral_evidence_mislabeled_as_formula_parity")
    if any(
        _truthy(status.get(key))
        for key in (
            "candidate_promotion_authority",
            "testnet_order_authority",
            "live_trading_authorized",
        )
    ):
        blockers.append("copula_behavioral_evidence_has_forbidden_authority")
    bindings = (
        ("contract_path", "contract_sha256", contract_path),
        ("contract_receipt_path", "contract_receipt_sha256", receipt_path),
        ("proof_queue_path", "proof_queue_sha256", queue_path),
        ("vendor_proof_path", "vendor_proof_sha256", proof_path),
        ("evidence_path", "evidence_sha256", detail_path),
    )
    for path_field, hash_field, expected_path in bindings:
        if _text(status.get(path_field)) != _relative(expected_path, root):
            blockers.append(f"copula_behavioral_binding_path_mismatch:{path_field}")
        elif not expected_path.is_file() or _text(status.get(hash_field)) != _file_hash(
            expected_path
        ):
            blockers.append(f"copula_behavioral_binding_hash_mismatch:{path_field}")
    blockers.extend(
        _copula_immutable_cohort_blockers(
            root=root,
            status=status,
            detail=detail,
            expected_identities=expected_identities,
            contract_path=contract_path,
            receipt_path=receipt_path,
            queue_path=queue_path,
        )
    )
    observed_identities = {
        (_text(row.get("pair_group_id")), _text(row.get("orientation")))
        for _, row in detail.iterrows()
        if _text(row.get("behavioral_status")) == "PASS"
        and _text(row.get("provenance_status")) == "PASS"
        and int(row.get("endpoint_responses_captured", 0) or 0) == 2
    }
    if observed_identities != expected_identities:
        blockers.append("copula_behavioral_identity_coverage_mismatch")
    passed = not blockers
    return {
        "status": "PASS" if passed else "BLOCKED",
        "expected_cells": expected_count,
        "passed_cells": expected_count if passed else 0,
        "provenance_cells": expected_count if passed else 0,
        "orientations_passed": (
            sorted({orientation for _, orientation in expected_identities}) if passed else []
        ),
        "blocker": ";".join(dict.fromkeys(blockers)),
    }


def _copula_immutable_cohort_blockers(
    *,
    root: Path,
    status: dict[str, Any],
    detail: pd.DataFrame,
    expected_identities: set[tuple[str, str]],
    contract_path: Path,
    receipt_path: Path,
    queue_path: Path,
) -> list[str]:
    blockers: list[str] = []
    cohort_id = _text(status.get("cohort_receipt_id"))
    cohort_relative = _text(status.get("cohort_receipt_path"))
    cohort_hash = _text(status.get("cohort_receipt_sha256"))
    if not cohort_id or not cohort_relative or not cohort_hash:
        return ["copula_immutable_cohort_receipt_missing"]
    cohort_path = root / cohort_relative
    expected_root = root / "data" / "research" / "wizard_copula_behavioral_cohorts"
    try:
        cohort_path.resolve().relative_to(expected_root.resolve())
    except ValueError:
        return ["copula_immutable_cohort_path_outside_evidence_root"]
    if (
        not cohort_path.is_file()
        or cohort_path.stem != cohort_id
        or _file_hash(cohort_path) != cohort_hash
    ):
        return ["copula_immutable_cohort_binding_mismatch"]
    cohort = _read_json(cohort_path)
    material = {key: value for key, value in cohort.items() if key != "cohort_id"}
    expected_cohort_id = "copulacohort_" + sha256(_canonical_json_bytes(material)).hexdigest()[:20]
    if _text(cohort.get("cohort_id")) != expected_cohort_id:
        blockers.append("copula_immutable_cohort_content_identity_mismatch")
    if _text(cohort.get("schema_version")) != ("thewiz.wizard_copula_behavioral_cohort.v1"):
        blockers.append("copula_immutable_cohort_schema_mismatch")
    expected_bindings = {
        "contract_path": _relative(contract_path, root),
        "contract_sha256": _file_hash(contract_path),
        "contract_receipt_path": _relative(receipt_path, root),
        "contract_receipt_sha256": _file_hash(receipt_path),
        "proof_queue_path": _relative(queue_path, root),
        "proof_queue_sha256": _file_hash(queue_path),
    }
    for field, expected in expected_bindings.items():
        if _text(cohort.get(field)) != expected:
            blockers.append(f"copula_immutable_cohort_binding_mismatch:{field}")
    if any(
        _truthy(cohort.get(key))
        for key in (
            "candidate_promotion_authority",
            "testnet_order_authority",
            "live_trading_authorized",
        )
    ):
        blockers.append("copula_immutable_cohort_has_forbidden_authority")
    cells = cohort.get("cells")
    if not isinstance(cells, list):
        return [*blockers, "copula_immutable_cohort_cells_missing"]
    cohort_identities = {
        (_text(cell.get("pair_group_id")), _text(cell.get("orientation")))
        for cell in cells
        if isinstance(cell, dict)
    }
    if cohort_identities != expected_identities:
        blockers.append("copula_immutable_cohort_identity_coverage_mismatch")
    detail_by_identity = {
        (_text(row.get("pair_group_id")), _text(row.get("orientation"))): row
        for row in detail.to_dict("records")
    }
    for cell in cells:
        if not isinstance(cell, dict):
            blockers.append("copula_immutable_cohort_cell_invalid")
            continue
        identity = (
            _text(cell.get("pair_group_id")),
            _text(cell.get("orientation")),
        )
        detail_row = detail_by_identity.get(identity, {})
        for field in (
            "capture_id",
            "capture_receipt_path",
            "capture_receipt_sha256",
            "request_sha256",
            "response_1_sha256",
            "response_2_sha256",
        ):
            if _text(cell.get(field)) != _text(detail_row.get(field)):
                blockers.append(
                    f"copula_immutable_cohort_detail_mismatch:{identity[0]}:{identity[1]}:{field}"
                )
        capture_relative = _text(cell.get("capture_receipt_path"))
        capture_path = root / capture_relative
        if (
            not capture_relative
            or not capture_path.is_file()
            or _text(cell.get("capture_receipt_sha256")) != _file_hash(capture_path)
        ):
            blockers.append(
                f"copula_immutable_capture_receipt_mismatch:{identity[0]}:{identity[1]}"
            )
    return blockers


def build_wizard_parity_capture_status(*, root: Path = ROOT) -> dict[str, Any]:
    source_path = root / "reports" / "active" / "exhaustive_wizard_pair_detail_mode_ledger.csv"
    source = _read_csv(source_path)
    rows = []
    for mode in EXACT_MODES:
        for orientation in ORIENTATIONS:
            group = source.loc[
                source.get("exact_mode", pd.Series(dtype=str)).eq(mode)
                & source.get("orientation", pd.Series(dtype=str)).eq(orientation)
            ]
            captured = group.loc[group.get("capture_status", pd.Series(dtype=str)).eq("CAPTURED")]
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "exact_mode": mode,
                    "orientation": orientation,
                    "expected_cells": len(group),
                    "captured_cells": len(captured),
                    "orientation_verified_cells": int(
                        captured.get("orientation_verified", pd.Series(False, index=captured.index))
                        .map(_truthy)
                        .sum()
                    ),
                    "raw_chart_preserved_cells": int(
                        captured.get(
                            "raw_chart_data_preserved", pd.Series(False, index=captured.index)
                        )
                        .map(_truthy)
                        .sum()
                    ),
                    "point_in_time_cost_confirmed_cells": int(
                        captured.get(
                            "wizard_cost_point_in_time_ui_confirmed",
                            pd.Series(False, index=captured.index),
                        )
                        .map(_truthy)
                        .sum()
                    ),
                    "capture_status": "CAPTURED" if len(captured) else "UNAVAILABLE_OR_MISSING",
                    "parity_claim_allowed": False,
                    "blocker": "" if len(captured) else "vendor_mode_not_available_or_not_captured",
                    "evidence_path": _relative(source_path, root),
                    "live_trading_authorized": False,
                }
            )
    frame = pd.DataFrame(rows)
    status_path = root / "reports" / "active" / "wizard_parity_capture_status.csv"
    _atomic_csv(frame, status_path)
    raw_dir = root / "data" / "raw" / "crypto_wizards_parity"
    raw_dir.mkdir(parents=True, exist_ok=True)
    capture_manifest = {
        "schema_version": SCHEMA_VERSION,
        "source_path": _relative(source_path, root),
        "source_sha256": _file_hash(source_path),
        "expected_modes": list(EXACT_MODES),
        "orientations": list(ORIENTATIONS),
        "cells_accounted": int(frame["expected_cells"].sum()),
        "cells_captured": int(frame["captured_cells"].sum()),
        "point_in_time_cost_semantics_proven": False,
        "vendor_formula_parity_proven": False,
        "live_trading_authorized": False,
    }
    (raw_dir / "capture_manifest.json").write_text(
        json.dumps(capture_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "frame": frame,
        "status": status_path,
        "capture_manifest": raw_dir / "capture_manifest.json",
        "source": source,
        "summary": capture_manifest,
    }


def build_ou_optimal_overlay_provenance(*, root: Path = ROOT) -> dict[str, Any]:
    """Account for OU Optimal as a scanner annotation, not a pair-page mode."""

    source_path = root / "reports" / "active" / "current_wizard_ou_optimal_overlay_ledger.csv"
    source = _read_csv(source_path)
    ordered_audit = build_ou_optimal_ordered_orientation_audit(root=root)
    ordered_frame = ordered_audit["frame"]
    required = {
        "experiment_orientation",
        "ou_optimal",
        "ou_optimal_semantics",
        "independent_pair_page_mode",
        "source_exact_mode",
        "evidence_path",
    }
    rows = []
    for orientation in ORIENTATIONS:
        group = (
            source.loc[source.get("experiment_orientation", pd.Series(dtype=str)).eq(orientation)]
            if required.issubset(source.columns)
            else pd.DataFrame()
        )
        true_count = int(group.get("ou_optimal", pd.Series(dtype=bool)).map(_truthy).sum())
        false_count = int(len(group) - true_count)
        historical_group = ordered_frame.loc[
            ordered_frame.get("orientation", pd.Series(dtype=str)).eq(orientation)
        ]
        historical_rows = int(
            pd.to_numeric(
                historical_group.get("source_rows", pd.Series(dtype=float)),
                errors="coerce",
            )
            .fillna(0)
            .sum()
        )
        historical_true = int(
            pd.to_numeric(
                historical_group.get("true_rows", pd.Series(dtype=float)),
                errors="coerce",
            )
            .fillna(0)
            .sum()
        )
        historical_false = int(
            pd.to_numeric(
                historical_group.get("false_rows", pd.Series(dtype=float)),
                errors="coerce",
            )
            .fillna(0)
            .sum()
        )
        historical_reverse_provenance = bool(
            orientation == "reverse"
            and group.empty
            and ordered_audit["status"] == "PASS"
            and historical_rows > 0
        )
        if historical_reverse_provenance:
            true_count = historical_true
            false_count = historical_false
        semantics_valid = (
            bool(
                not group.empty
                and group["ou_optimal_semantics"].astype(str).eq("scanner_boolean_annotation").all()
            )
            or historical_reverse_provenance
        )
        not_pair_mode = (
            bool(not group.empty and not group["independent_pair_page_mode"].map(_truthy).any())
            or historical_reverse_provenance
        )
        status = "PASS" if semantics_valid and not_pair_mode else "BLOCKED"
        blockers = []
        if group.empty and not historical_reverse_provenance:
            blockers.append("overlay_orientation_not_captured")
        if not semantics_valid:
            blockers.append("ou_optimal_scanner_semantics_unproven")
        if not not_pair_mode:
            blockers.append("ou_optimal_pair_page_mode_classification_conflict")
        rows.append(
            {
                "overlay": "OU Optimal",
                "orientation": orientation,
                "source_rows": len(group) or historical_rows,
                "current_source_rows": len(group),
                "historical_ordered_source_rows": historical_rows,
                "true_rows": true_count,
                "false_rows": false_count,
                "row_accounting_complete": (
                    (len(group) or historical_rows) == true_count + false_count
                ),
                "semantics": "scanner_boolean_annotation" if semantics_valid else "unproven",
                "independent_pair_page_mode": False if not_pair_mode else pd.NA,
                "vendor_contract_url": WIZARD_PRESCANNED_CONTRACT_URL,
                "vendor_contract_fields": ";".join(WIZARD_PRESCANNED_DIRECTION_FIELDS),
                "orientation_semantics": ("symbol_1_is_asset_x_symbol_2_is_asset_y"),
                "vendor_orientation_value_observed": bool(
                    not group.empty or historical_reverse_provenance
                ),
                "provenance_scope": (
                    "current_scanner_overlay_ledger"
                    if not group.empty
                    else "historical_point_in_time_ordered_pair_evidence"
                    if historical_reverse_provenance
                    else "missing"
                ),
                "cross_snapshot_hindsight_aggregate": historical_reverse_provenance,
                "live_signal_eligible": False,
                "overlay_provenance_status": status,
                "formula_parity_status": "UNPROVEN",
                "blocker": ";".join(blockers),
                "evidence_path": (
                    _relative(source_path, root)
                    if not group.empty
                    else _relative(Path(ordered_audit["path"]), root)
                ),
                "promotion_authority": False,
                "live_trading_authorized": False,
            }
        )
    frame = pd.DataFrame(rows)
    path = root / "reports" / "active" / "wizard_ou_optimal_overlay_provenance.csv"
    _atomic_csv(frame, path)
    return {
        "path": path,
        "ordered_orientation_audit": ordered_audit["path"],
        "frame": frame,
        "orientations_accounted": int(frame["overlay_provenance_status"].eq("PASS").sum()),
        "expected_orientations": len(ORIENTATIONS),
        "status": "PASS" if frame["overlay_provenance_status"].eq("PASS").all() else "BLOCKED",
        "vendor_formula_parity_proven": False,
        "live_trading_authorized": False,
    }


def build_ou_optimal_ordered_orientation_audit(*, root: Path = ROOT) -> dict[str, Any]:
    """Prove both vendor symbol orders from immutable point-in-time API captures."""

    raw_base = root / "data" / "raw" / "crypto_wizards" / "prescanned"
    observations: list[dict[str, Any]] = []
    files_scanned = 0
    for path in sorted(raw_base.glob("**/*.json")):
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(envelope, dict):
            continue
        response = envelope.get("response")
        request = envelope.get("request")
        metadata = envelope.get("capture_metadata")
        if not isinstance(response, list) or not isinstance(request, dict):
            continue
        files_scanned += 1
        exchange = normalize_wizard_exchange(request.get("exchange"))
        interval = _text(request.get("interval")).lower()
        captured_at = _text(metadata.get("captured_at")) if isinstance(metadata, dict) else ""
        for row in response:
            if not isinstance(row, dict) or "ou_optimal" not in row:
                continue
            left = normalize_wizard_symbol(row.get("symbol_1"), exchange).base_asset
            right = normalize_wizard_symbol(row.get("symbol_2"), exchange).base_asset
            if not left or not right or left == right:
                continue
            canonical = tuple(sorted((left, right)))
            orientation = "original" if (left, right) == canonical else "reverse"
            value = row.get("ou_optimal")
            observations.append(
                {
                    "exchange": exchange,
                    "interval": interval,
                    "canonical_pair": "/".join(canonical),
                    "orientation": orientation,
                    "ordered_pair": f"{left}/{right}",
                    "ou_optimal": value,
                    "explicit_boolean": isinstance(value, bool),
                    "captured_at": captured_at,
                    "evidence_path": _relative(path, root),
                }
            )
    frame = pd.DataFrame(observations)
    rows: list[dict[str, Any]] = []
    if not frame.empty:
        keys = ["exchange", "interval", "canonical_pair"]
        orientation_counts = frame.groupby(keys)["orientation"].nunique()
        bidirectional_keys = set(orientation_counts.loc[orientation_counts.ge(2)].index)
        bidirectional = frame.loc[
            frame.apply(
                lambda row: (
                    (row["exchange"], row["interval"], row["canonical_pair"]) in bidirectional_keys
                ),
                axis=1,
            )
        ].copy()
        for group_key, group in bidirectional.groupby(keys + ["orientation"]):
            paths = sorted(set(group["evidence_path"].astype(str)))
            rows.append(
                {
                    "exchange": group_key[0],
                    "interval": group_key[1],
                    "canonical_pair": group_key[2],
                    "orientation": group_key[3],
                    "orientation_reference": "lexicographically_sorted_base_assets",
                    "ordered_pairs_observed": ";".join(
                        sorted(set(group["ordered_pair"].astype(str)))
                    ),
                    "source_rows": len(group),
                    "true_rows": int(group["ou_optimal"].map(_truthy).sum()),
                    "false_rows": int(len(group) - group["ou_optimal"].map(_truthy).sum()),
                    "explicit_boolean_rows": int(group["explicit_boolean"].sum()),
                    "first_captured_at": min(group["captured_at"].astype(str)),
                    "last_captured_at": max(group["captured_at"].astype(str)),
                    "point_in_time_rows": True,
                    "cross_snapshot_hindsight_aggregate": True,
                    "provenance_only": True,
                    "live_signal_eligible": False,
                    "promotion_authority": False,
                    "live_trading_authorized": False,
                    "evidence_path": ";".join(paths),
                }
            )
    audit = pd.DataFrame(rows)
    audit_path = root / "reports" / "active" / "wizard_ou_optimal_ordered_orientation_audit.csv"
    summary_path = audit_path.with_suffix(".json")
    _atomic_csv(audit, audit_path)
    orientations = set(audit.get("orientation", pd.Series(dtype=str)))
    all_explicit = bool(
        not audit.empty
        and pd.to_numeric(audit["explicit_boolean_rows"], errors="coerce")
        .eq(pd.to_numeric(audit["source_rows"], errors="coerce"))
        .all()
    )
    status = "PASS" if set(ORIENTATIONS).issubset(orientations) and all_explicit else "BLOCKED"
    summary = {
        "status": status,
        "official_response_files_scanned": files_scanned,
        "ordered_rows_audited": len(frame),
        "bidirectional_pair_groups": int(
            audit.get("canonical_pair", pd.Series(dtype=str)).nunique()
        ),
        "orientations_observed": sorted(orientations),
        "all_values_explicit_boolean": all_explicit,
        "cross_snapshot_hindsight_aggregate": True,
        "provenance_only": True,
        "live_signal_eligible": False,
        "promotion_authority": False,
        "live_trading_authorized": False,
        "blocker": "" if status == "PASS" else "both_vendor_symbol_orders_not_observed",
        "evidence_path": _relative(audit_path, root),
    }
    _atomic_json(summary, summary_path)
    return {"path": audit_path, "summary": summary_path, "frame": audit, **summary}


def build_wizard_golden_fixtures(*, root: Path = ROOT) -> dict[str, Any]:
    capture = build_wizard_parity_capture_status(root=root)
    source = capture["source"]
    fixture_dir = root / "data" / "fixtures" / "wizard_exact_modes"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    audit_rows = []
    for mode in EXACT_MODES:
        for orientation in ORIENTATIONS:
            eligible = source.loc[
                source.get("exact_mode", pd.Series(dtype=str)).eq(mode)
                & source.get("orientation", pd.Series(dtype=str)).eq(orientation)
                & source.get("capture_status", pd.Series(dtype=str)).eq("CAPTURED")
                & source.get("orientation_verified", pd.Series(False, index=source.index)).map(
                    _truthy
                )
                & source.get("raw_chart_data_preserved", pd.Series(False, index=source.index)).map(
                    _truthy
                )
            ].copy()
            fixture_status = "GOLDEN_CAPTURE" if not eligible.empty else "MISSING_VENDOR_MODE"
            fixture_path = ""
            evidence_path = ""
            source_hash = ""
            if not eligible.empty:
                eligible["_completeness"] = (
                    eligible[list(set(FIXTURE_FIELDS) & set(eligible.columns))].notna().sum(axis=1)
                )
                row = eligible.sort_values(
                    ["_completeness", "capture_timestamp"], ascending=[False, False]
                ).iloc[0]
                payload = {field: _json_scalar(row.get(field)) for field in FIXTURE_FIELDS}
                payload.update(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "fixture_status": fixture_status,
                        "vendor_formula_parity_proven": False,
                        "live_trading_authorized": False,
                    }
                )
                name = f"{_slug(mode)}__{orientation}.json"
                path = fixture_dir / name
                path.write_text(
                    json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8",
                )
                fixture_path = _relative(path, root)
                source_hash = _file_hash(path)
                evidence_path = _text(row.get("evidence_path"))
            blocker = (
                ""
                if fixture_status == "GOLDEN_CAPTURE"
                else "mode_unavailable_on_captured_pair_pages"
            )
            entry = {
                "exact_mode": mode,
                "orientation": orientation,
                "fixture_status": fixture_status,
                "fixture_path": fixture_path,
                "fixture_sha256": source_hash,
                "source_evidence_path": evidence_path,
                "blocker": blocker,
                "vendor_formula_parity_proven": False,
                "live_trading_authorized": False,
            }
            entries.append(entry)
            audit_rows.append(entry)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "required_modes": list(EXACT_MODES),
        "required_orientations": list(ORIENTATIONS),
        "required_cells": len(EXACT_MODES) * len(ORIENTATIONS),
        "cells_accounted": len(entries),
        "golden_captures": sum(row["fixture_status"] == "GOLDEN_CAPTURE" for row in entries),
        "missing_vendor_modes": sum(row["fixture_status"] != "GOLDEN_CAPTURE" for row in entries),
        "entries": entries,
        "live_trading_authorized": False,
    }
    manifest_path = fixture_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    audit = pd.DataFrame(audit_rows)
    audit_path = root / "reports" / "active" / "wizard_golden_fixture_audit.csv"
    _atomic_csv(audit, audit_path)
    return {
        "manifest": manifest_path,
        "audit": audit_path,
        "summary": manifest,
        "audit_frame": audit,
    }


def run_wizard_mode_mutation_tests(*, root: Path = ROOT) -> pd.DataFrame:
    baseline = np.linspace(-3.0, 3.0, 121)
    cases = {
        "matching_formula": baseline.copy(),
        "window_mutation": np.roll(baseline, 1),
        "spread_sign_mutation": -baseline,
        "zscore_scale_mutation": baseline * 0.5,
        "copula_tail_mutation": np.clip(baseline, -1.0, 1.0),
        "orientation_mutation": baseline[::-1],
        "exit_mutation": np.where(np.abs(baseline) < 0.25, 1.0, baseline),
    }
    rows = []
    for case, observed in cases.items():
        comparison = compare_series(baseline, observed)
        expected = "PASS" if case == "matching_formula" else "FAIL"
        rows.append(
            {
                "case": case,
                "expected_status": expected,
                "actual_status": comparison["status"],
                "max_abs_delta": comparison["max_abs_delta"],
                "blocker": comparison["blocker"],
                "mislabeled_mode_can_receive_parity": case != "matching_formula"
                and comparison["status"] == "PASS",
                "live_trading_authorized": False,
                "status": "PASS" if expected == comparison["status"] else "FAIL",
            }
        )
    frame = pd.DataFrame(rows)
    path = root / "reports" / "red_team" / "wizard_mode_mutation_results.csv"
    _atomic_csv(frame, path)
    if not frame["status"].eq("PASS").all():
        raise ValueError("Wizard mode mutation escaped parity comparison")
    return frame


def build_wizard_mode_comparator_contract(*, root: Path = ROOT) -> dict[str, Any]:
    """Freeze mode-specific exact-comparator readiness before vendor responses."""

    from quant_platform.wizard_hyperliquid_mode_proof import (
        build_comparator_formula_registration,
    )

    registrations = {
        exact_mode: build_comparator_formula_registration(exact_mode) for exact_mode in EXACT_MODES
    }
    rows = []
    for exact_mode in EXACT_MODES:
        registration = registrations[exact_mode]
        if exact_mode.startswith("Static"):
            comparator_version = "static_ols_full_history_v1"
            implementation_status = "READY"
            semantics = "Static spread uses OLS; ZScoreR is rolling normalization"
            blocker = ""
            contract_url = WIZARD_SPREAD_CONTRACT_URL
        elif exact_mode.startswith("Dyn"):
            comparator_version = "kalman_random_walk_beta_alpha_delta_1e-5_obsvar_1_v1"
            implementation_status = "CANDIDATE_READY_FOR_VENDOR_VALIDATION"
            semantics = "Dynamic spread uses a time-varying Kalman-filter hedge ratio"
            blocker = "preregistered_kalman_candidate_not_vendor_validated"
            contract_url = WIZARD_SPREAD_CONTRACT_URL
        elif exact_mode.startswith("OU"):
            comparator_version = "ou_profile_gaussian_ar1_mle_beta_v1"
            implementation_status = "CANDIDATE_READY_FOR_VENDOR_VALIDATION"
            semantics = "OU spread is estimated by maximum likelihood"
            blocker = "preregistered_ou_mle_candidate_not_vendor_validated"
            contract_url = WIZARD_SPREAD_CONTRACT_URL
        else:
            comparator_version = "unimplemented_copula_calibration_contract"
            implementation_status = "RESEARCH_BACKLOG"
            semantics = "Copula strategy ignores spread and uses dependency conditionals"
            blocker = "vendor_copula_family_selection_and_historical_signal_series_not_published"
            contract_url = WIZARD_COPULA_CONTRACT_URL
        for orientation in ORIENTATIONS:
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "exact_mode": exact_mode,
                    "orientation": orientation,
                    "comparator_version": comparator_version,
                    "comparator_specification_sha256": registration["specification_sha256"],
                    "comparator_source_sha256": registration["source_sha256"],
                    "comparator_implementation_sha256": registration["implementation_sha256"],
                    "comparator_runtime_json": json.dumps(
                        registration["runtime"], sort_keys=True, separators=(",", ":")
                    ),
                    "comparator_specification_json": registration["specification_json"],
                    "implementation_status": implementation_status,
                    "vendor_semantics": semantics,
                    "official_contract_url": contract_url,
                    "backtest_contract_url": WIZARD_BACKTEST_CONTRACT_URL,
                    "exact_reconstruction_required": True,
                    "vendor_response_capture_sufficient_for_parity": False,
                    "blocker": blocker,
                    "threshold_changes_after_vendor_response_allowed": False,
                    "promotion_authority": False,
                    "testnet_order_authority": False,
                    "live_trading_authorized": False,
                }
            )
    frame = pd.DataFrame(rows)
    path = root / "reports" / "active" / "wizard_mode_comparator_contract.csv"
    contract_sha256 = sha256(frame.to_csv(index=False).encode("utf-8")).hexdigest()
    contract_id = f"wizardcomparator_{contract_sha256[:20]}"
    immutable_path = (
        root / "data" / "research" / "wizard_mode_comparator_contracts" / f"{contract_id}.csv"
    )
    implementation_path = immutable_path.with_suffix(".implementation.json")
    implementation_bundle = {
        "schema_version": "thewiz.wizard_comparator_implementation_bundle.v1",
        "contract_id": contract_id,
        "registrations": {
            exact_mode: {
                key: value
                for key, value in registrations[exact_mode].items()
                if key != "specification_json"
            }
            for exact_mode in EXACT_MODES
        },
    }
    implementation_sha256 = sha256(_canonical_json_bytes(implementation_bundle)).hexdigest()
    proof_path = root / "reports" / "active" / "hyperliquid_wizard_vendor_mode_proofs.csv"
    proofs = _read_csv(proof_path)
    current_responses = int(
        (
            proofs.get("vendor_response_captured", pd.Series(False, index=proofs.index)).map(
                _truthy
            )
            & proofs.get("pair_group_id", pd.Series("", index=proofs.index)).map(_text).ne("")
        ).sum()
    )
    receipt_path = root / "reports" / "active" / "wizard_mode_comparator_contract_receipt.json"
    receipt = _read_json(receipt_path)
    previous_contract_id = ""
    if receipt:
        old_contract_id = _text(receipt.get("contract_id"))
        old_contract_path = root / _text(receipt.get("immutable_contract_path"))
        old_contract_sha256 = _text(receipt.get("contract_sha256"))
        if (
            not old_contract_id
            or not old_contract_path.is_file()
            or _file_hash(old_contract_path) != old_contract_sha256
        ):
            raise ValueError("immutable Wizard comparator contract hash mismatch")
        _write_or_validate_immutable_json(
            receipt,
            old_contract_path.parent / f"{old_contract_id}.receipt.json",
        )
        if old_contract_id != contract_id:
            if current_responses > 0:
                raise ValueError(
                    "registered Wizard comparator contract cannot change after vendor responses"
                )
            previous_contract_id = old_contract_id
            receipt = {}
        elif (
            receipt.get("contract_sha256") != contract_sha256
            or receipt.get("immutable_contract_path") != _relative(immutable_path, root)
            or receipt.get("implementation_bundle_path") != _relative(implementation_path, root)
            or receipt.get("implementation_bundle_sha256") != implementation_sha256
        ):
            raise ValueError("registered Wizard comparator contract changed")
    else:
        previous_contract_id = ""
    if not receipt:
        if not immutable_path.is_file():
            _atomic_csv(frame, immutable_path)
        elif _file_hash(immutable_path) != contract_sha256:
            raise ValueError("immutable Wizard comparator contract hash mismatch")
        _write_or_validate_immutable_json(implementation_bundle, implementation_path)
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "contract_id": contract_id,
            "contract_sha256": contract_sha256,
            "registered_at_utc": datetime.now(UTC).isoformat(),
            "current_queue_vendor_responses_at_registration": current_responses,
            "registered_before_current_queue_vendor_responses": (current_responses == 0),
            "immutable_contract_path": _relative(immutable_path, root),
            "implementation_bundle_path": _relative(implementation_path, root),
            "implementation_bundle_sha256": implementation_sha256,
            "previous_contract_id": previous_contract_id,
            "supersession_before_current_queue_vendor_responses": bool(
                previous_contract_id and current_responses == 0
            ),
            "threshold_changes_after_vendor_response_allowed": False,
            "promotion_authority": False,
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        }
        _write_or_validate_immutable_json(
            receipt, immutable_path.parent / f"{contract_id}.receipt.json"
        )
        _atomic_json(receipt, receipt_path)
    elif (
        _file_hash(immutable_path) != contract_sha256
        or _file_hash(implementation_path) != implementation_sha256
    ):
        raise ValueError("immutable Wizard comparator implementation hash mismatch")
    _atomic_csv(frame, path)
    if _file_hash(path) != contract_sha256:
        raise ValueError("active Wizard comparator contract hash mismatch")
    return {
        "path": path,
        "receipt_path": receipt_path,
        "immutable_path": immutable_path,
        "implementation_path": implementation_path,
        "implementation_sha256": implementation_sha256,
        "contract_id": contract_id,
        "registered_before_current_queue_vendor_responses": bool(
            receipt["registered_before_current_queue_vendor_responses"]
        ),
        "frame": frame,
        "ready_cells": int(frame["implementation_status"].eq("READY").sum()),
        "backlog_cells": int(frame["implementation_status"].eq("RESEARCH_BACKLOG").sum()),
        "candidate_validation_cells": int(
            frame["implementation_status"].eq("CANDIDATE_READY_FOR_VENDOR_VALIDATION").sum()
        ),
    }


def build_wizard_mode_authority(*, root: Path = ROOT) -> dict[str, Any]:
    fixtures = build_wizard_golden_fixtures(root=root)
    audit = fixtures["audit_frame"]
    comparators = build_wizard_mode_comparator_contract(root=root)
    comparator_lookup = {
        (row["exact_mode"], row["orientation"]): row
        for row in comparators["frame"].to_dict("records")
    }
    proof_path = root / "reports" / "active" / "hyperliquid_wizard_vendor_mode_proofs.csv"
    proofs = _read_csv(proof_path)
    mixed_evidence = build_wizard_mode_evidence_completion(root=root)
    copula_orientations = set(mixed_evidence["copula_orientations_passed"])
    rows = []
    for row in audit.to_dict("records"):
        comparator = comparator_lookup[(row["exact_mode"], row["orientation"])]
        golden = row["fixture_status"] == "GOLDEN_CAPTURE"
        proof = (
            None
            if row["exact_mode"] == "Copula"
            else _matching_formula_proof(
                proofs,
                exact_mode=row["exact_mode"],
                orientation=row["orientation"],
            )
        )
        response = (
            None
            if row["exact_mode"] == "Copula"
            else _matching_vendor_response(
                proofs,
                exact_mode=row["exact_mode"],
                orientation=row["orientation"],
            )
        )
        formula_proven = proof is not None
        behavioral_proven = bool(
            row["exact_mode"] == "Copula" and row["orientation"] in copula_orientations
        )
        if formula_proven:
            blocker = (
                "vendor_formula_parity_proven_but_performance_cost_and_live_signal_parity_unproven"
            )
        elif behavioral_proven:
            blocker = (
                "vendor_copula_behavioral_parity_proven_but_formula_performance_"
                "cost_and_live_signal_parity_unproven"
            )
        elif response is not None:
            observed_status = (
                _text(response.get("vendor_formula_parity_status", "unproven")) or "unproven"
            )
            blocker = f"vendor_response_captured_but_formula_parity_not_proven:{observed_status}"
        else:
            blocker = (
                "comparable_vendor_output_series_and_formula_parity_missing"
                if golden
                else row["blocker"]
            )
        evidence_paths = [row["fixture_path"] or row["source_evidence_path"]]
        if proof is not None:
            evidence_paths.extend(
                [
                    _text(proof.get("request_path")),
                    _text(proof.get("response_path")),
                    _text(proof.get("evidence_path")),
                ]
            )
        elif response is not None:
            evidence_paths.extend(
                [
                    _text(response.get("request_path")),
                    _text(response.get("response_path")),
                    _text(response.get("evidence_path")),
                ]
            )
        rows.append(
            {
                "exact_mode": row["exact_mode"],
                "orientation": row["orientation"],
                "capture_authority": "point_in_time_vendor_observation" if golden else "unproven",
                "formula_authority": (
                    "vendor_custom_series_exact_reconstruction"
                    if formula_proven
                    else "not_available_vendor_behavioral_contract"
                    if behavioral_proven
                    else ("local_approximation" if golden else "unproven")
                ),
                "behavioral_authority": (
                    "vendor_custom_series_repeatable_orientation_symmetric"
                    if behavioral_proven
                    else "unproven"
                ),
                "vendor_parity_status": (
                    "FORMULA_PARITY_PROVEN"
                    if formula_proven
                    else "BEHAVIORAL_PARITY_PROVEN_FORMULA_NOT_AVAILABLE"
                    if behavioral_proven
                    else "UNPROVEN"
                ),
                "vendor_exact_mode_parity_proven": formula_proven,
                "vendor_behavioral_parity_proven": behavioral_proven,
                "vendor_mode_evidence_gate_passed": (formula_proven or behavioral_proven),
                "vendor_response_captured": response is not None,
                "vendor_observed_formula_parity_status": (
                    _text(response.get("vendor_formula_parity_status"))
                    if response is not None
                    else ""
                ),
                "vendor_observed_formula_max_abs_error": (
                    _proof_max_error(response) if response is not None else ""
                ),
                "formula_parity_scope": "single_pair_mode_orientation"
                if formula_proven
                else "none",
                "proof_pair": _text(proof.get("pair")) if proof is not None else "",
                "proof_pair_group_id": _text(proof.get("pair_group_id"))
                if proof is not None
                else "",
                "proof_observations": proof.get("proof_observations", "")
                if proof is not None
                else "",
                "proof_window_kind": _text(proof.get("proof_window_kind"))
                if proof is not None
                else "",
                "proof_max_abs_error": _proof_max_error(proof) if proof is not None else "",
                "formula_comparator_version": comparator["comparator_version"],
                "formula_comparator_status": comparator["implementation_status"],
                "formula_comparator_blocker": comparator["blocker"],
                "performance_accounting_parity_proven": False,
                "cost_parity_proven": False,
                "live_signal_point_in_time_parity_proven": False,
                "research_use": (
                    "vendor_formula_confirmed_research_only"
                    if formula_proven
                    else "vendor_behavior_observed_research_only"
                    if behavioral_proven
                    else (
                        "local_approximation_research" if golden else "blocked_until_vendor_capture"
                    )
                ),
                "wizard_formula_confirmed_claim_allowed": formula_proven,
                "wizard_behavioral_confirmed_claim_allowed": behavioral_proven,
                "wizard_confirmed_claim_allowed": False,
                "acceptance_eligible": False,
                "blocker": blocker,
                "evidence_path": ";".join(dict.fromkeys(path for path in evidence_paths if path)),
                "promotion_authority": False,
                "live_trading_authorized": False,
            }
        )
    authority = pd.DataFrame(rows)
    path = root / "reports" / "active" / "wizard_mode_authority.csv"
    _atomic_csv(authority, path)
    parity_path = root / "reports" / "active" / "wizard_mode_parity.csv"
    _atomic_csv(authority, parity_path)
    overlay = _read_csv(root / "reports" / "active" / "wizard_ou_optimal_overlay_provenance.csv")
    overlay_accounted = int(
        overlay.get("overlay_provenance_status", pd.Series(dtype=str)).eq("PASS").sum()
    )
    docs = root / "docs" / "wizard_hyperliquid_mode_fidelity.md"
    docs.parent.mkdir(parents=True, exist_ok=True)
    docs.write_text(
        _fidelity_markdown(
            authority,
            fixtures["summary"],
            overlay_accounted=overlay_accounted,
            overlay_expected=len(ORIENTATIONS),
        ),
        encoding="utf-8",
    )
    return {
        "authority": path,
        "parity": parity_path,
        "docs": docs,
        "comparator_contract": comparators["path"],
        "comparator_contract_receipt": comparators["receipt_path"],
        "comparator_immutable_contract": comparators["immutable_path"],
        "comparator_implementation_bundle": comparators["implementation_path"],
        "comparator_implementation_sha256": comparators["implementation_sha256"],
        "comparator_contract_id": comparators["contract_id"],
        "comparator_registered_before_vendor_responses": comparators[
            "registered_before_current_queue_vendor_responses"
        ],
        "comparator_ready_cells": comparators["ready_cells"],
        "comparator_backlog_cells": comparators["backlog_cells"],
        "comparator_candidate_validation_cells": comparators["candidate_validation_cells"],
        "frame": authority,
    }


def build_corrective_wizard_parity(*, root: Path = ROOT) -> CommandResult:
    capture = build_wizard_parity_capture_status(root=root)
    overlay = build_ou_optimal_overlay_provenance(root=root)
    fixtures = build_wizard_golden_fixtures(root=root)
    mutations = run_wizard_mode_mutation_tests(root=root)
    authority = build_wizard_mode_authority(root=root)
    frame = authority["frame"]
    parity_cells = int(frame["vendor_exact_mode_parity_proven"].sum())
    behavioral_authority_cells = int(frame["vendor_behavioral_parity_proven"].sum())
    accepted_authority_cells = int(frame["vendor_mode_evidence_gate_passed"].sum())
    mixed_evidence = build_wizard_mode_evidence_completion(root=root)
    complete = mixed_evidence["status"] == "PASS" and overlay["status"] == "PASS"
    blockers = []
    if mixed_evidence["status"] != "PASS":
        blockers.append(str(mixed_evidence["blocker"]))
    if overlay["status"] != "PASS":
        blockers.append("ou_optimal_overlay_orientation_provenance_incomplete")
    paths = {
        "capture_status": Path(capture["status"]),
        "capture_manifest": Path(capture["capture_manifest"]),
        "ou_optimal_overlay_provenance": Path(overlay["path"]),
        "ou_optimal_ordered_orientation_audit": Path(overlay["ordered_orientation_audit"]),
        "fixture_manifest": Path(fixtures["manifest"]),
        "fixture_audit": Path(fixtures["audit"]),
        "mutation_results": root / "reports" / "red_team" / "wizard_mode_mutation_results.csv",
        "mode_parity": Path(authority["parity"]),
        "mode_authority": Path(authority["authority"]),
        "mode_comparator_contract": Path(authority["comparator_contract"]),
        "mode_comparator_contract_receipt": Path(authority["comparator_contract_receipt"]),
        "mode_comparator_immutable_contract": Path(authority["comparator_immutable_contract"]),
        "mode_comparator_implementation_bundle": Path(
            authority["comparator_implementation_bundle"]
        ),
        "mode_fidelity_docs": Path(authority["docs"]),
    }
    return CommandResult(
        paths=paths,
        summary={
            "capture_cells_accounted": capture["summary"]["cells_accounted"],
            "capture_cells_captured": capture["summary"]["cells_captured"],
            "fixture_cells_accounted": fixtures["summary"]["cells_accounted"],
            "golden_capture_cells": fixtures["summary"]["golden_captures"],
            "missing_vendor_mode_cells": fixtures["summary"]["missing_vendor_modes"],
            "ou_optimal_overlay_status": overlay["status"],
            "ou_optimal_orientations_accounted": overlay["orientations_accounted"],
            "ou_optimal_expected_orientations": overlay["expected_orientations"],
            "vendor_parity_cells": parity_cells,
            "vendor_formula_authority_cells_expected": ((len(EXACT_MODES) - 1) * len(ORIENTATIONS)),
            "vendor_behavioral_authority_cells": behavioral_authority_cells,
            "vendor_behavioral_authority_cells_expected": len(ORIENTATIONS),
            "accepted_mode_authority_cells": accepted_authority_cells,
            "accepted_mode_authority_cells_expected": (len(EXACT_MODES) * len(ORIENTATIONS)),
            **mixed_evidence,
            "status": "PASS" if complete else "BLOCKED",
            "comparator_ready_cells": authority["comparator_ready_cells"],
            "comparator_backlog_cells": authority["comparator_backlog_cells"],
            "comparator_candidate_validation_cells": authority[
                "comparator_candidate_validation_cells"
            ],
            "comparator_contract_id": authority["comparator_contract_id"],
            "comparator_registered_before_vendor_responses": authority[
                "comparator_registered_before_vendor_responses"
            ],
            "mutation_cases_passed": int(mutations["status"].eq("PASS").sum()),
            "blocker": ";".join(blockers),
            "live_trading_authorized": False,
        },
    )


def _fidelity_markdown(
    authority: pd.DataFrame,
    summary: dict[str, Any],
    *,
    overlay_accounted: int,
    overlay_expected: int,
) -> str:
    lines = [
        "# Wizard and Hyperliquid Mode Fidelity",
        "",
        f"- Required mode/orientation cells: `{summary['required_cells']}`",
        f"- Golden dashboard captures: `{summary['golden_captures']}`",
        f"- Missing vendor-mode captures: `{summary['missing_vendor_modes']}`",
        f"- Vendor formula parity proven: `{int(authority['vendor_exact_mode_parity_proven'].sum())}`",
        f"- Vendor Copula behavioral parity proven: `{int(authority['vendor_behavioral_parity_proven'].sum())}`",
        "- Live trading authorized: `false`",
        "",
        "A dashboard label and a local formula with the same name are separate claims. Captured Wizard fields support diagnosis and research hypotheses. Until comparable vendor output series reproduce within declared tolerances, local computations remain `local_approximation` and cannot be described as Wizard-confirmed.",
        "Copula is evaluated under a separately preregistered behavioral contract because the vendor does not publish its family-selection or historical-signal formula. A Copula behavioral pass requires repeatability, orientation symmetry, probability bounds, complete raw provenance, and every registered queue cell. It never becomes a formula-parity claim.",
        "OU parity separates two claims: local formula reconstruction and local prediction of the vendor's `coint_eg.inc_trend` branch. The immutable `wizard_ou_trend_selector_v1_holdout.json` contract compares branch-specific two-lag ADF AIC values and freezes holdout predictions before vendor capture. OU cannot pass unless all four prospective cells match both the registered selector and the reconstructed numeric series.",
        "",
        f"`OU (Optimal)` is a scanner boolean annotation layered onto a source exact mode, not an eighth Wizard pair-page mode. The official prescanned contract exposes ordered `symbol_1` / `symbol_2`, `spread_type`, `strategy`, and `ou_optimal`; it does not expose a separate pair-page mode. Current orientation provenance is `{overlay_accounted}/{overlay_expected}` in `wizard_ou_optimal_overlay_provenance.csv`. Any orientation without an observed vendor row remains blocked. Its formula semantics are also unproven, so it cannot receive a Wizard-exact claim.",
        f"Vendor contract: {WIZARD_PRESCANNED_CONTRACT_URL}",
        "",
    ]
    return "\n".join(lines)


def _matching_formula_proof(
    proofs: pd.DataFrame,
    *,
    exact_mode: str,
    orientation: str,
    tolerance: float = 1e-9,
) -> pd.Series | None:
    required = {
        "exact_mode",
        "orientation",
        "mode_proof_status",
        "proof_window_kind",
        "vendor_formula_parity_status",
        "vendor_history_available",
    }
    if proofs.empty or not required.issubset(proofs.columns):
        return None
    matches = proofs.loc[
        proofs["exact_mode"].map(_text).eq(exact_mode)
        & proofs["orientation"].map(_text).eq(orientation)
        & proofs["mode_proof_status"].map(_text).eq("completed")
        & proofs["proof_window_kind"].map(_text).eq("scanner_horizon_parity")
        & proofs["vendor_formula_parity_status"].map(_text).eq("exact_reconstruction")
        & proofs["vendor_history_available"].map(_truthy)
    ].copy()
    if matches.empty:
        return None
    error_columns = (
        "vendor_spread_max_abs_error",
        "vendor_zscore_max_abs_error",
        "vendor_zscore_roll_max_abs_error",
    )
    valid_indices = []
    for index, row in matches.iterrows():
        errors = [_finite_number(row.get(column)) for column in error_columns]
        if all(error is not None and error <= tolerance for error in errors):
            valid_indices.append(index)
    if not valid_indices:
        return None
    valid = matches.loc[valid_indices].copy()
    valid["_max_error"] = valid.apply(_proof_max_error, axis=1)
    return valid.sort_values(["_max_error", "proof_observations"], ascending=[True, False]).iloc[0]


def _matching_vendor_response(
    proofs: pd.DataFrame,
    *,
    exact_mode: str,
    orientation: str,
) -> pd.Series | None:
    required = {
        "exact_mode",
        "orientation",
        "mode_proof_status",
        "proof_window_kind",
    }
    if proofs.empty or not required.issubset(proofs.columns):
        return None
    captured = proofs["mode_proof_status"].map(_text).eq("completed")
    if "vendor_response_captured" in proofs.columns:
        captured &= proofs["vendor_response_captured"].map(_truthy)
    if "vendor_history_available" in proofs.columns:
        captured &= proofs["vendor_history_available"].map(_truthy)
    matches = proofs.loc[
        proofs["exact_mode"].map(_text).eq(exact_mode)
        & proofs["orientation"].map(_text).eq(orientation)
        & proofs["proof_window_kind"].map(_text).eq("scanner_horizon_parity")
        & captured
    ].copy()
    if matches.empty:
        return None
    matches["_response_observations"] = pd.to_numeric(
        matches.get("proof_observations", 0), errors="coerce"
    ).fillna(0)
    matches["_response_path"] = matches.get(
        "response_path", pd.Series("", index=matches.index, dtype=str)
    ).map(_text)
    return matches.sort_values(
        ["_response_observations", "_response_path"],
        ascending=[False, False],
    ).iloc[0]


def _proof_max_error(row: pd.Series) -> float | str:
    errors = [
        _finite_number(row.get(column))
        for column in (
            "vendor_spread_max_abs_error",
            "vendor_zscore_max_abs_error",
            "vendor_zscore_roll_max_abs_error",
        )
    ]
    finite = [error for error in errors if error is not None]
    return max(finite) if len(finite) == 3 else ""


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


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


def _canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_or_validate_immutable_json(payload: dict[str, Any], path: Path) -> str:
    expected = _canonical_json_bytes(payload)
    expected_hash = sha256(expected).hexdigest()
    if path.is_file():
        if _file_hash(path) != expected_hash:
            raise ValueError(f"immutable JSON artifact changed: {_text(path)}")
        return expected_hash
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(expected)
    temporary.replace(path)
    return expected_hash


def _file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest() if path.is_file() else ""


def _json_scalar(value: Any) -> Any:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _truthy(value: Any) -> bool:
    return value is True or str(value).strip().lower() in {"1", "true", "yes", "pass", "captured"}


def _text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def _slug(value: str) -> str:
    return "_".join(value.lower().replace("(", " ").replace(")", " ").split())


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    result = build_corrective_wizard_parity()
    print(
        json.dumps(
            {
                "summary": result.summary,
                "paths": {key: str(value) for key, value in result.paths.items()},
            },
            indent=2,
        )
    )
