"""Reproducible failure attribution for the completed OU-v4 holdout."""

from __future__ import annotations

import json
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd

from quant_platform.active_pipeline import CommandResult
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
from quant_platform.orchestration.corrective_wizard_ou_v4_holdout import (
    DEFAULT_TOLERANCES,
    _evaluate_formula,
)

ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "thewiz.wizard_ou_v4_failure_attribution.v1"
EVALUATION_PATH = Path("reports/active/wizard_ou_v4_holdout_evaluation.csv")
STATUS_PATH = Path("reports/active/wizard_ou_v4_holdout_status.json")
OUTPUT_CSV = Path("reports/active/wizard_ou_v4_failure_attribution.csv")
OUTPUT_JSON = Path("reports/active/wizard_ou_v4_failure_attribution.json")
OUTPUT_MD = Path("reports/supreme_team/wizard_ou_v4_failure_checkpoint.md")
IMMUTABLE_DIR = Path("data/research/wizard_ou_v4_failure_attributions")
EXPECTED_MODES = {"OU (Spread)", "OU (ZScoreR)"}


def build_ou_v4_failure_attribution(
    *, root: Path = ROOT, now: datetime | None = None
) -> CommandResult:
    """Explain the v4 failure without converting failed evidence into a holdout."""

    timestamp = _as_utc(now)
    evaluation_path = root / EVALUATION_PATH
    status_path = root / STATUS_PATH
    blockers: list[str] = []
    if not evaluation_path.is_file() or not status_path.is_file():
        raise FileNotFoundError("OU-v4 evaluation evidence is missing")
    evaluation = pd.read_csv(evaluation_path)
    status = _read_json(status_path)
    if status.get("status") != "FAIL":
        blockers.append("ou_v4_status_is_not_a_completed_failure")
    immutable_result = _resolve(root, status.get("immutable_result_path"))
    if immutable_result is None or not immutable_result.is_file():
        blockers.append("ou_v4_immutable_result_missing")
    elif _file_hash(immutable_result) != _text(status.get("immutable_result_sha256")):
        blockers.append("ou_v4_immutable_result_hash_mismatch")
    if len(evaluation) != 8:
        blockers.append(f"ou_v4_evaluation_requires_eight_cells:{len(evaluation)}")

    rows: list[dict[str, Any]] = []
    group_columns = ["pair_group", "pair", "interval", "orientation"]
    if all(column in evaluation.columns for column in group_columns):
        grouped = evaluation.groupby(group_columns, sort=True, dropna=False)
    else:
        grouped = []
        blockers.append("ou_v4_evaluation_identity_columns_missing")
    for identity, cells in grouped:
        pair_group, pair, interval, orientation = identity
        modes = {_text(value) for value in cells.get("exact_mode", [])}
        if len(cells) != 2 or modes != EXPECTED_MODES:
            blockers.append(f"ou_v4_orientation_mode_accounting_invalid:{pair}:{orientation}")
            continue
        first = cells.iloc[0]
        for field in (
            "vendor_log_used",
            "vendor_inc_trend",
            "vendor_hedge_ratio",
            "local_predicted_log_used",
            "local_predicted_inc_trend",
        ):
            if cells[field].nunique(dropna=False) != 1:
                blockers.append(
                    f"ou_v4_duplicate_mode_evidence_disagrees:{pair}:{orientation}:{field}"
                )
        branch_metrics: dict[str, list[dict[str, object]]] = {
            "zero_mean": [],
            "intercept": [],
        }
        for _, cell in cells.iterrows():
            request_path = root / _text(cell.get("request_path"))
            response_path = root / _text(cell.get("response_path"))
            if (
                not request_path.is_file()
                or _file_hash(request_path) != _text(cell.get("request_sha256"))
                or not response_path.is_file()
                or _file_hash(response_path) != _text(cell.get("response_sha256"))
            ):
                blockers.append(
                    f"ou_v4_raw_binding_invalid:{pair}:{orientation}:"
                    f"{_text(cell.get('exact_mode'))}"
                )
                continue
            request = _read_json(request_path)
            response = _read_json(response_path)
            vendor_log = _truthy(first.get("vendor_log_used"))
            branch_metrics["zero_mean"].append(
                _evaluate_formula(
                    request=request,
                    response=response,
                    log_used=vendor_log,
                    inc_trend=True,
                    tolerances=DEFAULT_TOLERANCES,
                )
            )
            branch_metrics["intercept"].append(
                _evaluate_formula(
                    request=request,
                    response=response,
                    log_used=vendor_log,
                    inc_trend=False,
                    tolerances=DEFAULT_TOLERANCES,
                )
            )
        if any(len(values) != 2 for values in branch_metrics.values()):
            continue
        branch_pass = {
            name: all(_truthy(metric.get("formula_parity_passed")) for metric in values)
            for name, values in branch_metrics.items()
        }
        passing = [name for name, passed in branch_pass.items() if passed]
        inferred_branch = passing[0] if len(passing) == 1 else "ambiguous"
        if inferred_branch == "ambiguous":
            blockers.append(f"ou_v4_profile_branch_not_uniquely_attributed:{pair}:{orientation}")
        expected_from_trend = (
            "zero_mean" if _truthy(first.get("vendor_inc_trend")) else "intercept"
        )
        transform_match = _truthy(first.get("transform_selector_parity_passed"))
        branch_match = inferred_branch == expected_from_trend
        root_causes = []
        if not transform_match:
            root_causes.append("transform_selector")
        if not branch_match:
            root_causes.append("profile_branch_selector")
        if not root_causes:
            root_causes.append("none_for_independent_orientation")
        chosen = branch_metrics.get(inferred_branch, [{}])[0]
        alternate_name = "intercept" if inferred_branch == "zero_mean" else "zero_mean"
        alternate = branch_metrics.get(alternate_name, [{}])[0]
        rows.append(
            {
                "pair_group": pair_group,
                "pair": pair,
                "interval": interval,
                "orientation": orientation,
                "exact_modes_accounted": ";".join(sorted(modes)),
                "duplicate_mode_cells": len(cells),
                "vendor_log_used": _truthy(first.get("vendor_log_used")),
                "local_predicted_log_used": _truthy(
                    first.get("local_predicted_log_used")
                ),
                "transform_selector_match": transform_match,
                "vendor_inc_trend": _truthy(first.get("vendor_inc_trend")),
                "local_predicted_inc_trend": _truthy(
                    first.get("local_predicted_inc_trend")
                ),
                "trend_selector_match": _truthy(
                    first.get("trend_selector_parity_passed")
                ),
                "profile_branch_assumed_from_inc_trend": expected_from_trend,
                "inferred_vendor_profile_branch": inferred_branch,
                "profile_branch_selector_match": branch_match,
                "vendor_hedge_ratio": float(first.get("vendor_hedge_ratio")),
                "zero_mean_hedge_ratio": float(
                    branch_metrics["zero_mean"][0]["local_hedge_ratio"]
                ),
                "zero_mean_hedge_ratio_abs_error": float(
                    branch_metrics["zero_mean"][0]["hedge_ratio_abs_error"]
                ),
                "zero_mean_formula_passed": branch_pass["zero_mean"],
                "intercept_hedge_ratio": float(
                    branch_metrics["intercept"][0]["local_hedge_ratio"]
                ),
                "intercept_hedge_ratio_abs_error": float(
                    branch_metrics["intercept"][0]["hedge_ratio_abs_error"]
                ),
                "intercept_formula_passed": branch_pass["intercept"],
                "inferred_branch_spread_max_abs_error": float(
                    chosen.get("spread_max_abs_error", float("nan"))
                ),
                "alternate_branch_spread_max_abs_error": float(
                    alternate.get("spread_max_abs_error", float("nan"))
                ),
                "root_cause": ";".join(root_causes),
                "evidence_role": "derivation_only",
                "holdout_reuse_allowed": False,
                "candidate_promotion_authority": False,
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            }
        )

    detail = pd.DataFrame(rows)
    unique_orientations = len(detail)
    best_branch_passed = int(
        detail.get("inferred_vendor_profile_branch", pd.Series(dtype=str))
        .ne("ambiguous")
        .sum()
    )
    transform_passed = int(
        detail.get("transform_selector_match", pd.Series(dtype=bool)).map(_truthy).sum()
    )
    trend_passed = int(
        detail.get("trend_selector_match", pd.Series(dtype=bool)).map(_truthy).sum()
    )
    branch_selector_passed = int(
        detail.get("profile_branch_selector_match", pd.Series(dtype=bool)).map(_truthy).sum()
    )
    if unique_orientations != 4:
        blockers.append(f"ou_v4_requires_four_independent_orientations:{unique_orientations}")
    if best_branch_passed != 4:
        blockers.append(f"ou_v4_formula_kernel_not_attributed:{best_branch_passed}_of_4")
    blockers = list(dict.fromkeys(blockers))
    findings = [
        {
            "lens": "gap_analysis",
            "finding": (
                f"Independent orientation parity was transform {transform_passed}/4, "
                f"trend {trend_passed}/4, and profile-branch selector "
                f"{branch_selector_passed}/4."
            ),
            "corrective_order": (
                "Separate transform, Engle-Granger trend, and OU profile-branch "
                "selection; inc_trend is not the profile-branch switch."
            ),
        },
        {
            "lens": "pre_mortem",
            "finding": (
                "A v5 model could appear perfect by tuning on these same four "
                "orientations or by counting Spread and ZScoreR duplicates as "
                "independent evidence."
            ),
            "corrective_order": (
                "Collapse duplicate modes, freeze a disjoint asset cohort, and "
                "record zero vendor responses at v5 registration."
            ),
        },
        {
            "lens": "post_mortem",
            "finding": (
                f"The numerical formula kernel reconstructs {best_branch_passed}/4 "
                "orientations when the vendor transform and uniquely inferred "
                "profile branch are supplied."
            ),
            "corrective_order": (
                "Keep the kernels; replace only the unproven transform and "
                "profile-branch selectors."
            ),
        },
        {
            "lens": "red_team",
            "finding": (
                "Relabeling OU-v4 as derivation and then reporting it as v5 "
                "validation would be holdout laundering."
            ),
            "corrective_order": (
                "Set holdout_reuse_allowed=false and require new immutable v5 "
                "requests, implementation hash, cohort, and capture manifest."
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
        "source_implementation_sha256": _text(status.get("implementation_source_sha256")),
        "mode_cells": len(evaluation),
        "independent_orientations": unique_orientations,
        "transform_selector_orientations_passed": transform_passed,
        "trend_selector_orientations_passed": trend_passed,
        "profile_branch_selector_orientations_passed": branch_selector_passed,
        "formula_kernel_orientations_attributed": best_branch_passed,
        "v4_evidence_role": "derivation_only",
        "v4_holdout_reuse_allowed": False,
        "v5_requirements": [
            "new_disjoint_assets_and_pair_groups",
            "new_frozen_request_hashes",
            "new_implementation_source_hash",
            "zero_vendor_responses_at_registration",
            "transform_and_profile_branch_predictions_frozen_before_capture",
            "spread_and_zscorer_counted_as_duplicate_formula_views",
            "all_orientation_predictions_must_pass_without_threshold_changes",
        ],
        "findings": findings,
        "research_only": True,
        "candidate_promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    attribution_id = "ouv4failure_" + sha256(
        _canonical_json(stable).encode("utf-8")
    ).hexdigest()[:20]
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
    temporary.replace(path)


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Wizard OU-v4 Failure Checkpoint",
        "",
        f"- Status: `{payload['status']}`",
        f"- Attribution ID: `{payload['attribution_id']}`",
        f"- Source holdout: `{payload['source_holdout_result_id']}` (`FAIL`)",
        f"- Independent orientations: `{payload['independent_orientations']}`",
        (
            "- Selector results: "
            f"transform `{payload['transform_selector_orientations_passed']}/4`, "
            f"trend `{payload['trend_selector_orientations_passed']}/4`, "
            "profile branch "
            f"`{payload['profile_branch_selector_orientations_passed']}/4`"
        ),
        (
            "- Formula kernel attribution: "
            f"`{payload['formula_kernel_orientations_attributed']}/4`"
        ),
        "- Evidence role: `derivation_only`",
        "- OU-v4 reuse as validation: `forbidden`",
        "- Trading authority: `false`",
        "",
    ]
    for finding in payload["findings"]:
        title = _text(finding["lens"]).replace("_", " ").title()
        lines.extend(
            [
                f"## {title}",
                "",
                _text(finding["finding"]),
                "",
                f"Corrective order: {_text(finding['corrective_order'])}",
                "",
            ]
        )
    lines.extend(["## V5 Hard Gates", ""])
    lines.extend(f"- `{requirement}`" for requirement in payload["v5_requirements"])
    lines.append("")
    return "\n".join(lines)
