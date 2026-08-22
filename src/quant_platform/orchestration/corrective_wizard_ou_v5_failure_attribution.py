"""Immutable failure attribution for a completed OU-v5 holdout."""

from __future__ import annotations

import json
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd

from quant_platform.active_pipeline import CommandResult
from quant_platform.orchestration.corrective_runtime import promote_staged_file
from quant_platform.orchestration.corrective_wizard_ou_holdout import (
    _as_utc,
    _atomic_csv,
    _atomic_json,
    _file_hash,
    _read_json,
    _relative,
    _text,
    _truthy,
    _write_or_validate_immutable_json,
)

ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "thewiz.wizard_ou_v5_failure_attribution.v1"
EVALUATION_PATH = Path("reports/active/wizard_ou_v5_holdout_evaluation.csv")
STATUS_PATH = Path("reports/active/wizard_ou_v5_holdout_status.json")
OUTPUT_CSV = Path("reports/active/wizard_ou_v5_failure_attribution.csv")
OUTPUT_JSON = Path("reports/active/wizard_ou_v5_failure_attribution.json")
OUTPUT_MD = Path("reports/supreme_team/wizard_ou_v5_failure_checkpoint.md")
IMMUTABLE_DIR = Path("data/research/wizard_ou_v5_failure_attributions")
EXPECTED_MODES = {"OU (Spread)", "OU (ZScoreR)"}


def build_ou_v5_failure_attribution(
    *, root: Path = ROOT, now: datetime | None = None
) -> CommandResult:
    """Explain a failed v5 cohort without designing or registering its successor."""

    timestamp = _as_utc(now)
    evaluation_path = root / EVALUATION_PATH
    status_path = root / STATUS_PATH
    if not evaluation_path.is_file() or not status_path.is_file():
        raise FileNotFoundError("OU-v5 evaluation evidence is missing")
    evaluation = pd.read_csv(evaluation_path)
    status = _read_json(status_path)
    blockers: list[str] = []
    if status.get("status") != "FAIL":
        blockers.append("ou_v5_status_is_not_a_completed_failure")
    immutable_result = _resolve(root, status.get("immutable_result_path"))
    if immutable_result is None or not immutable_result.is_file():
        blockers.append("ou_v5_immutable_result_missing")
    elif _file_hash(immutable_result) != _text(status.get("immutable_result_sha256")):
        blockers.append("ou_v5_immutable_result_hash_mismatch")
    else:
        immutable_payload = _read_json(immutable_result)
        blockers.extend(_source_binding_blockers(evaluation, status, immutable_payload))
    if len(evaluation) != 8:
        blockers.append(f"ou_v5_evaluation_requires_eight_cells:{len(evaluation)}")

    identity_columns = ["pair_group", "pair", "interval", "orientation"]
    required_columns = {
        *identity_columns,
        "exact_mode",
        "cell_status",
        "transform_selector_parity_passed",
        "trend_selector_parity_passed",
        "profile_branch_selector_parity_passed",
        "formula_parity_passed",
        "request_path",
        "request_sha256",
        "response_path",
        "response_sha256",
    }
    if not required_columns.issubset(evaluation.columns):
        blockers.append("ou_v5_evaluation_columns_missing")
        grouped: Any = []
    else:
        grouped = evaluation.groupby(identity_columns, sort=True, dropna=False)

    rows: list[dict[str, Any]] = []
    for identity, cells in grouped:
        pair_group, pair, interval, orientation = identity
        modes = {_text(value) for value in cells["exact_mode"]}
        if len(cells) != 2 or modes != EXPECTED_MODES:
            blockers.append(f"ou_v5_orientation_mode_accounting_invalid:{pair}:{orientation}")
            continue
        for field in (
            "predicted_log_used",
            "vendor_log_used",
            "transform_selector_parity_passed",
            "predicted_inc_trend",
            "vendor_inc_trend",
            "trend_selector_parity_passed",
            "predicted_profile_branch",
            "inferred_vendor_profile_branch",
            "profile_branch_selector_parity_passed",
        ):
            if field in cells and cells[field].nunique(dropna=False) != 1:
                blockers.append(
                    f"ou_v5_duplicate_mode_evidence_disagrees:{pair}:{orientation}:{field}"
                )
        raw_bindings_valid = True
        for _, cell in cells.iterrows():
            request_path = _resolve(root, cell.get("request_path"))
            response_path = _resolve(root, cell.get("response_path"))
            if (
                request_path is None
                or not request_path.is_file()
                or _file_hash(request_path) != _text(cell.get("request_sha256"))
                or response_path is None
                or not response_path.is_file()
                or _file_hash(response_path) != _text(cell.get("response_sha256"))
            ):
                raw_bindings_valid = False
        if not raw_bindings_valid:
            blockers.append(f"ou_v5_raw_binding_invalid:{pair}:{orientation}")

        transform_passed = cells["transform_selector_parity_passed"].map(_truthy).all()
        trend_passed = cells["trend_selector_parity_passed"].map(_truthy).all()
        branch_passed = cells["profile_branch_selector_parity_passed"].map(_truthy).all()
        formula_passed = cells["formula_parity_passed"].map(_truthy).all()
        causes = [
            name
            for name, passed in (
                ("transform_selector", transform_passed),
                ("trend_selector", trend_passed),
                ("profile_branch_selector", branch_passed),
                ("formula_kernel", formula_passed),
            )
            if not passed
        ]
        rows.append(
            {
                "pair_group": pair_group,
                "pair": pair,
                "interval": interval,
                "orientation": orientation,
                "exact_modes_accounted": ";".join(sorted(modes)),
                "duplicate_mode_cells": len(cells),
                "transform_selector_match": bool(transform_passed),
                "trend_selector_match": bool(trend_passed),
                "profile_branch_selector_match": bool(branch_passed),
                "formula_kernel_match": bool(formula_passed),
                "orientation_status": ("PASS" if not causes else "FAIL"),
                "root_cause": ";".join(causes) if causes else "none",
                "raw_bindings_valid": raw_bindings_valid,
                "evidence_role": "consumed_negative_holdout_derivation_only",
                "holdout_reuse_allowed": False,
                "candidate_promotion_authority": False,
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            }
        )

    detail = pd.DataFrame(rows)
    orientations = len(detail)
    if orientations != 4:
        blockers.append(f"ou_v5_requires_four_independent_orientations:{orientations}")
    counts = {
        "transform_selector_orientations_passed": _passed(detail, "transform_selector_match"),
        "trend_selector_orientations_passed": _passed(detail, "trend_selector_match"),
        "profile_branch_selector_orientations_passed": _passed(
            detail, "profile_branch_selector_match"
        ),
        "formula_kernel_orientations_passed": _passed(detail, "formula_kernel_match"),
    }
    blockers = list(dict.fromkeys(blockers))
    failed_orientations = int(
        detail.get("orientation_status", pd.Series(dtype=str)).eq("FAIL").sum()
    )
    findings = [
        {
            "lens": "gap_analysis",
            "finding": (
                f"OU-v5 failed in {failed_orientations} of {orientations} independent "
                "orientations; selector and formula causes are preserved separately."
            ),
            "corrective_order": (
                "Use only the attributed failure classes to form a new hypothesis; do not "
                "reuse v5 observations as successor validation."
            ),
        },
        {
            "lens": "premortem",
            "finding": (
                "A successor tuned against these revealed rows could appear to improve while "
                "only laundering consumed holdout evidence."
            ),
            "corrective_order": (
                "Require a new implementation hash, frozen predictions, disjoint cohort, and "
                "zero vendor responses at successor registration."
            ),
        },
        {
            "lens": "postmortem",
            "finding": (
                "The observed failure is retained as negative research evidence with raw "
                "request and response bindings for each orientation."
            ),
            "corrective_order": (
                "Review this immutable receipt before deciding whether the comparator family "
                "merits another prospective test."
            ),
        },
        {
            "lens": "red_team",
            "finding": (
                "Automatic successor design after a failed holdout would create an adaptive "
                "path around the preregistration boundary."
            ),
            "corrective_order": (
                "Keep activation, successor registration, testnet orders, and live trading "
                "disabled until a separately preregistered cohort passes."
            ),
        },
    ]
    stable = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS_FAILURE_ATTRIBUTION_COMPLETE" if not blockers else "BLOCKED",
        "blockers": blockers,
        "source_holdout_result_id": _text(status.get("result_id")),
        "source_holdout_status": _text(status.get("status")),
        "source_evaluation_path": _relative(evaluation_path, root),
        "source_evaluation_sha256": _file_hash(evaluation_path),
        "source_immutable_result_path": (
            _relative(immutable_result, root) if immutable_result is not None else ""
        ),
        "source_immutable_result_sha256": _text(status.get("immutable_result_sha256")),
        "mode_cells": len(evaluation),
        "independent_orientations": orientations,
        **counts,
        "failed_orientations": failed_orientations,
        "findings": findings,
        "v5_evidence_role": "consumed_negative_holdout_derivation_only",
        "v5_holdout_reuse_allowed": False,
        "successor_design_automatic": False,
        "successor_registration_automatic": False,
        "successor_registration_status": "NOT_REGISTERED_REQUIRES_NEW_HYPOTHESIS",
        "next_decision": (
            "supreme_team_review_failure_attribution_then_preregister_new_disjoint_hypothesis"
        ),
        "successor_hard_gates": [
            "new_hypothesis_from_v5_failure_attribution",
            "new_disjoint_assets_pair_groups_and_time_window",
            "new_implementation_hash_and_frozen_predictions",
            "zero_successor_vendor_responses_at_registration",
            "new_credit_bounded_immutable_capture_manifest",
            "no_v5_holdout_rows_reused_as_successor_validation",
        ],
        "research_only": True,
        "candidate_promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    attribution_id = (
        "ouv5failure_" + sha256(_canonical_json(stable).encode("utf-8")).hexdigest()[:20]
    )
    immutable_payload = {
        **stable,
        "attribution_id": attribution_id,
        "rows": detail.fillna("").to_dict("records"),
    }
    immutable_path = root / IMMUTABLE_DIR / f"{attribution_id}.json"
    _write_or_validate_immutable_json(immutable_payload, immutable_path)
    payload = {
        **stable,
        "attribution_id": attribution_id,
        "generated_at_utc": timestamp.isoformat(),
        "immutable_attribution_path": _relative(immutable_path, root),
        "immutable_attribution_sha256": _file_hash(immutable_path),
        "evidence_path": str(OUTPUT_CSV),
    }
    output_csv = root / OUTPUT_CSV
    output_json = root / OUTPUT_JSON
    output_md = root / OUTPUT_MD
    _atomic_csv(detail, output_csv)
    _atomic_json(payload, output_json)
    _atomic_text(_markdown(payload), output_md)
    return CommandResult(
        paths={
            "attribution": output_csv,
            "status": output_json,
            "supreme_team": output_md,
            "immutable_attribution": immutable_path,
        },
        summary=payload,
    )


def _passed(frame: pd.DataFrame, column: str) -> int:
    return int(frame.get(column, pd.Series(dtype=bool)).map(_truthy).sum())


def _source_binding_blockers(
    evaluation: pd.DataFrame,
    status: dict[str, Any],
    immutable: dict[str, Any],
) -> list[str]:
    blockers: list[str] = []
    if _text(immutable.get("result_id")) != _text(status.get("result_id")):
        blockers.append("ou_v5_source_result_id_mismatch")
    if immutable.get("status") != "FAIL" or status.get("status") != "FAIL":
        blockers.append("ou_v5_source_failure_status_mismatch")
    required_columns = {
        "cell_status",
        "transform_selector_parity_passed",
        "trend_selector_parity_passed",
        "profile_branch_selector_parity_passed",
        "formula_parity_passed",
        "response_sha256",
    }
    if not required_columns.issubset(evaluation.columns):
        return [*blockers, "ou_v5_source_binding_columns_missing"]
    expected = {
        "required_cells": len(evaluation),
        "passed_cells": int(evaluation["cell_status"].map(_text).eq("PASS").sum()),
        "failed_cells": int(evaluation["cell_status"].map(_text).ne("PASS").sum()),
        "transform_selector_parity_passed_cells": _passed(
            evaluation, "transform_selector_parity_passed"
        ),
        "trend_selector_parity_passed_cells": _passed(evaluation, "trend_selector_parity_passed"),
        "profile_branch_selector_parity_passed_cells": _passed(
            evaluation, "profile_branch_selector_parity_passed"
        ),
        "formula_parity_passed_cells": _passed(evaluation, "formula_parity_passed"),
        "response_sha256s": sorted(evaluation["response_sha256"].map(_text)),
    }
    for field, value in expected.items():
        if immutable.get(field) != value:
            blockers.append(f"ou_v5_source_immutable_summary_mismatch:{field}")
        if status.get(field) != value:
            blockers.append(f"ou_v5_source_active_summary_mismatch:{field}")
    return blockers


def _resolve(root: Path, value: object) -> Path | None:
    raw = _text(value)
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_absolute() else root / path


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _atomic_text(value: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    promote_staged_file(temporary, path)


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Wizard OU-v5 Failure Checkpoint",
        "",
        f"- Status: `{payload['status']}`",
        f"- Attribution ID: `{payload['attribution_id']}`",
        f"- Independent orientations: `{payload['independent_orientations']}`",
        f"- Failed orientations: `{payload['failed_orientations']}`",
        "- OU-v5 reuse as successor validation: `forbidden`",
        "- Automatic successor design/registration: `false`",
        "- Trading authority: `false`",
        "",
        "## Supreme Team",
        "",
    ]
    for finding in payload["findings"]:
        lines.extend(
            [
                f"### {str(finding['lens']).replace('_', ' ').title()}",
                "",
                str(finding["finding"]),
                "",
                f"Corrective order: {finding['corrective_order']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Successor Hard Gates",
            "",
        ]
    )
    lines.extend(f"- `{gate}`" for gate in payload["successor_hard_gates"])
    return "\n".join(lines) + "\n"
