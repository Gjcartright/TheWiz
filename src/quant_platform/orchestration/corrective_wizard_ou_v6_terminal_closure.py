"""Terminal, evidence-bound closure for the failed OU-v6 comparator holdout."""

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
SCHEMA_VERSION = "thewiz.wizard_ou_v6_terminal_closure.v1"
EVALUATION_PATH = Path("reports/active/wizard_ou_v6_holdout_evaluation.csv")
STATUS_PATH = Path("reports/active/wizard_ou_v6_holdout_status.json")
CONTRACT_PATH = Path("config/wizard_ou_comparator_v6_holdout.json")
CAPTURE_STATUS_PATH = Path("reports/active/wizard_ou_v6_capture_status.json")
FAILURE_CSV = Path("reports/active/wizard_ou_v6_failure_attribution.csv")
ORIENTATION_CSV = Path("reports/active/wizard_ou_v6_orientation_policy.csv")
JOURNAL_CSV = Path("reports/active/wizard_ou_v6_terminal_journal_rows.csv")
CLOSURE_JSON = Path("reports/active/wizard_ou_v6_terminal_closure.json")
CLOSURE_CSV = Path("reports/active/wizard_ou_v6_terminal_outcome.csv")
CLOSURE_MD = Path("reports/supreme_team/wizard_ou_v6_terminal_closure.md")
V7_POLICY_JSON = Path("reports/active/wizard_ou_v7_preregistration_policy.json")
IMMUTABLE_DIR = Path("data/research/wizard_ou_v6_terminal_closures")
EXPECTED_MODES = {"OU (Spread)", "OU (ZScoreR)"}


def build_ou_v6_terminal_closure(
    *, root: Path = ROOT, now: datetime | None = None
) -> CommandResult:
    """Close the failed v6 experiment without activating or adapting the comparator."""

    timestamp = _as_utc(now)
    evaluation_path = root / EVALUATION_PATH
    status_path = root / STATUS_PATH
    contract_path = root / CONTRACT_PATH
    for required in (evaluation_path, status_path, contract_path):
        if not required.is_file():
            raise FileNotFoundError(f"required OU-v6 closure evidence missing: {required}")

    evaluation = pd.read_csv(evaluation_path)
    status = _read_json(status_path)
    capture_status = _read_json(root / CAPTURE_STATUS_PATH)
    blockers: list[str] = []
    if status.get("status") != "FAIL":
        blockers.append("ou_v6_status_is_not_a_completed_failure")
    if status.get("final_successor_iteration") is not True:
        blockers.append("ou_v6_is_not_terminal")
    if status.get("successor_after_v6_failure_allowed") is not False:
        blockers.append("ou_v6_successor_policy_is_not_closed")
    if len(evaluation) != 8:
        blockers.append(f"ou_v6_closure_requires_eight_cells:{len(evaluation)}")
    if _file_hash(contract_path) != _text(status.get("contract_sha256")):
        blockers.append("ou_v6_contract_hash_mismatch")

    immutable_result = _resolve(root, status.get("immutable_result_path"))
    if immutable_result is None or not immutable_result.is_file():
        blockers.append("ou_v6_immutable_result_missing")
    elif _file_hash(immutable_result) != _text(status.get("immutable_result_sha256")):
        blockers.append("ou_v6_immutable_result_hash_mismatch")
    else:
        blockers.extend(_summary_binding_blockers(evaluation, status, _read_json(immutable_result)))

    required_columns = {
        "pair_group",
        "pair",
        "interval",
        "exact_mode",
        "orientation",
        "cell_status",
        "predicted_inc_trend",
        "vendor_inc_trend",
        "predicted_profile_branch",
        "inferred_vendor_profile_branch",
        "transform_selector_parity_passed",
        "trend_selector_parity_passed",
        "profile_branch_selector_parity_passed",
        "formula_parity_passed",
        "request_path",
        "request_sha256",
        "response_path",
        "response_sha256",
        "call_id",
        "call_intent_path",
        "call_intent_sha256",
        "call_completion_path",
        "call_completion_sha256",
        "blocker",
    }
    if not required_columns.issubset(evaluation.columns):
        blockers.append("ou_v6_closure_evaluation_columns_missing")
    else:
        blockers.extend(_raw_binding_blockers(root, evaluation))

    orientation = _orientation_policy(evaluation)
    failures = _failure_attribution(evaluation)
    if len(orientation) != 4:
        blockers.append(f"ou_v6_closure_requires_four_orientations:{len(orientation)}")
    if failures.empty:
        blockers.append("ou_v6_closure_requires_at_least_one_failed_cell")

    passed_orientations = (
        orientation.loc[
            orientation.get("orientation_status", pd.Series(dtype=str)).eq("PASS"), "pair"
        ]
        .astype(str)
        .tolist()
    )
    failed_orientations = (
        orientation.loc[
            orientation.get("orientation_status", pd.Series(dtype=str)).eq("FAIL"), "pair"
        ]
        .astype(str)
        .tolist()
    )
    consumed_pair_groups, consumed_assets = _consumed_ou_universe(root)
    decision = "REJECT_GENERAL_OU_V6_KEEP_PASSING_CELLS_DIAGNOSTIC_ONLY"
    v7_policy = {
        "schema_version": "thewiz.wizard_ou_v7_preregistration_policy.v1",
        "status": "BLOCKED_BY_TERMINAL_OU_V6_FAILURE",
        "source_result_id": _text(status.get("result_id")),
        "registration_authorized": False,
        "automatic_successor_authorized": False,
        "governance_reopen_required": True,
        "consumed_pair_groups": consumed_pair_groups,
        "consumed_assets": consumed_assets,
        "required_if_governance_is_reopened": [
            "new_named_hypothesis_not_selected_from_v6_outcomes",
            "new_implementation_hash_frozen_before_vendor_calls",
            "new_predictions_frozen_before_vendor_calls",
            "pair_groups_and_assets_disjoint_from_all_consumed_ou_holdouts",
            "zero_vendor_responses_present_at_registration",
            "point_in_time_inputs_only",
            "credit_bounded_immutable_capture_manifest",
            "explicit_supreme_team_governance_reopen_receipt",
        ],
        "v6_rows_reusable_as_holdout": False,
        "research_only": True,
        "candidate_promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }

    blockers = list(dict.fromkeys(blockers))
    evidence_hashes = {
        "contract_sha256": _file_hash(contract_path),
        "evaluation_sha256": _file_hash(evaluation_path),
        "immutable_result_sha256": (
            _file_hash(immutable_result) if immutable_result is not None else ""
        ),
        "request_sha256s": sorted(
            evaluation.get("request_sha256", pd.Series(dtype=str)).map(_text)
        ),
        "response_sha256s": sorted(
            evaluation.get("response_sha256", pd.Series(dtype=str)).map(_text)
        ),
        "call_intent_sha256s": sorted(
            evaluation.get("call_intent_sha256", pd.Series(dtype=str)).map(_text)
        ),
        "call_completion_sha256s": sorted(
            evaluation.get("call_completion_sha256", pd.Series(dtype=str)).map(_text)
        ),
    }
    stable = {
        "schema_version": SCHEMA_VERSION,
        "status": "CLOSED_TERMINAL_FAILURE" if not blockers else "BLOCKED_EVIDENCE_INVALID",
        "blockers": blockers,
        "decision": decision,
        "source_result_id": _text(status.get("result_id")),
        "source_status": _text(status.get("status")),
        "source_evaluation_path": _relative(evaluation_path, root),
        "source_immutable_result_path": (
            _relative(immutable_result, root) if immutable_result is not None else ""
        ),
        "required_cells": 8,
        "passed_cells": int(evaluation.get("cell_status", pd.Series(dtype=str)).eq("PASS").sum()),
        "failed_cells": int(evaluation.get("cell_status", pd.Series(dtype=str)).ne("PASS").sum()),
        "passed_orientations": passed_orientations,
        "failed_orientations": failed_orientations,
        "passing_cell_use": "diagnostic_reconstruction_reference_only",
        "failed_cell_use": "negative_research_evidence_only",
        "general_ou_v6_activation": False,
        "v7_registration_authorized": False,
        "v7_policy_status": v7_policy["status"],
        "evidence_locked": not blockers,
        "capture_status": _text(capture_status.get("status")),
        "calls_made": int(capture_status.get("calls_made", 0) or 0),
        "responses_captured": int(capture_status.get("responses_captured", 0) or 0),
        "credits_completed": int(capture_status.get("credits_completed", 0) or 0),
        "evidence_hashes": evidence_hashes,
        "next_action": ("continue_static_dynamic_and_copula_research;keep_ou_v6_blocked"),
        "research_only": True,
        "candidate_promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    closure_id = "ouv6closure_" + sha256(_canonical_json(stable).encode("utf-8")).hexdigest()[:20]
    immutable_payload = {
        **stable,
        "closure_id": closure_id,
        "failure_rows": failures.fillna("").to_dict("records"),
        "orientation_policy_rows": orientation.fillna("").to_dict("records"),
        "v7_policy": v7_policy,
    }
    immutable_path = root / IMMUTABLE_DIR / f"{closure_id}.json"
    _write_or_validate_immutable_json(immutable_payload, immutable_path)

    payload = {
        **stable,
        "closure_id": closure_id,
        "closed_at_utc": timestamp.isoformat(),
        "immutable_closure_path": _relative(immutable_path, root),
        "immutable_closure_sha256": _file_hash(immutable_path),
        "failure_attribution_path": str(FAILURE_CSV),
        "orientation_policy_path": str(ORIENTATION_CSV),
        "journal_rows_path": str(JOURNAL_CSV),
        "v7_policy_path": str(V7_POLICY_JSON),
    }
    outcome = pd.DataFrame(
        [
            {
                "status": payload["status"],
                "decision": decision,
                "required_cells": 8,
                "passed_cells": payload["passed_cells"],
                "failed_cells": payload["failed_cells"],
                "passed_orientations": ";".join(passed_orientations),
                "failed_orientations": ";".join(failed_orientations),
                "evidence_locked": payload["evidence_locked"],
                "general_ou_v6_activation": False,
                "v7_registration_authorized": False,
                "blocker": ";".join(blockers) or "ou_v6_terminal_holdout_failed",
                "next_action": payload["next_action"],
                "evidence_path": str(CLOSURE_JSON),
                "candidate_promotion_authority": False,
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            }
        ]
    )
    journal = _journal_rows(evaluation, status)
    _atomic_csv(failures, root / FAILURE_CSV)
    _atomic_csv(orientation, root / ORIENTATION_CSV)
    _atomic_csv(journal, root / JOURNAL_CSV)
    _atomic_csv(outcome, root / CLOSURE_CSV)
    _atomic_json(v7_policy, root / V7_POLICY_JSON)
    _atomic_json(payload, root / CLOSURE_JSON)
    _atomic_text(_markdown(payload, failures, orientation), root / CLOSURE_MD)
    return CommandResult(
        paths={
            "closure": root / CLOSURE_JSON,
            "outcome": root / CLOSURE_CSV,
            "failure_attribution": root / FAILURE_CSV,
            "orientation_policy": root / ORIENTATION_CSV,
            "journal_rows": root / JOURNAL_CSV,
            "v7_policy": root / V7_POLICY_JSON,
            "supreme_team": root / CLOSURE_MD,
            "immutable_closure": immutable_path,
        },
        summary=payload,
    )


def _raw_binding_blockers(root: Path, evaluation: pd.DataFrame) -> list[str]:
    blockers: list[str] = []
    path_hash_fields = (
        ("request_path", "request_sha256"),
        ("response_path", "response_sha256"),
        ("call_intent_path", "call_intent_sha256"),
        ("call_completion_path", "call_completion_sha256"),
    )
    for index, row in evaluation.iterrows():
        for path_field, hash_field in path_hash_fields:
            path = _resolve(root, row.get(path_field))
            expected_hash = _text(row.get(hash_field)).lower()
            if path is None or not path.is_file():
                blockers.append(f"ou_v6_closure_missing_binding:{index}:{path_field}")
            elif len(expected_hash) != 64 or _file_hash(path) != expected_hash:
                blockers.append(f"ou_v6_closure_hash_mismatch:{index}:{path_field}")
    if evaluation.get("call_id", pd.Series(dtype=str)).map(_text).nunique() != 8:
        blockers.append("ou_v6_closure_call_ids_not_unique")
    return blockers


def _summary_binding_blockers(
    evaluation: pd.DataFrame, status: dict[str, Any], immutable: dict[str, Any]
) -> list[str]:
    blockers: list[str] = []
    if _text(immutable.get("result_id")) != _text(status.get("result_id")):
        blockers.append("ou_v6_closure_result_id_mismatch")
    expected = {
        "required_cells": len(evaluation),
        "passed_cells": int(evaluation.get("cell_status", pd.Series(dtype=str)).eq("PASS").sum()),
        "failed_cells": int(evaluation.get("cell_status", pd.Series(dtype=str)).ne("PASS").sum()),
        "response_sha256s": sorted(
            evaluation.get("response_sha256", pd.Series(dtype=str)).map(_text)
        ),
    }
    for field, value in expected.items():
        if status.get(field) != value or immutable.get(field) != value:
            blockers.append(f"ou_v6_closure_summary_mismatch:{field}")
    return blockers


def _orientation_policy(evaluation: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    required = {"pair_group", "pair", "interval", "orientation", "exact_mode"}
    if not required.issubset(evaluation.columns):
        return pd.DataFrame()
    for identity, cells in evaluation.groupby(
        ["pair_group", "pair", "interval", "orientation"], sort=True, dropna=False
    ):
        pair_group, pair, interval, orientation = identity
        modes = {_text(value) for value in cells["exact_mode"]}
        selectors = {
            "transform_selector": cells.get(
                "transform_selector_parity_passed", pd.Series(False, index=cells.index)
            )
            .map(_truthy)
            .all(),
            "trend_selector": cells.get(
                "trend_selector_parity_passed", pd.Series(False, index=cells.index)
            )
            .map(_truthy)
            .all(),
            "profile_branch_selector": cells.get(
                "profile_branch_selector_parity_passed", pd.Series(False, index=cells.index)
            )
            .map(_truthy)
            .all(),
            "formula_kernel": cells.get(
                "formula_parity_passed", pd.Series(False, index=cells.index)
            )
            .map(_truthy)
            .all(),
        }
        causes = [name for name, passed in selectors.items() if not passed]
        valid_modes = len(cells) == 2 and modes == EXPECTED_MODES
        passed = valid_modes and not causes and cells["cell_status"].map(_text).eq("PASS").all()
        rows.append(
            {
                "pair_group": pair_group,
                "pair": pair,
                "interval": interval,
                "orientation": orientation,
                "exact_modes_accounted": ";".join(sorted(modes)),
                "orientation_status": "PASS" if passed else "FAIL",
                "root_cause": ";".join(causes) if causes else "none",
                "permitted_use": (
                    "diagnostic_reconstruction_reference_only"
                    if passed
                    else "negative_research_evidence_only"
                ),
                "generalization_allowed": False,
                "strategy_acceptance_authority": False,
                "candidate_promotion_authority": False,
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            }
        )
    return pd.DataFrame(rows)


def _failure_attribution(evaluation: pd.DataFrame) -> pd.DataFrame:
    if "cell_status" not in evaluation:
        return pd.DataFrame()
    failed = evaluation.loc[~evaluation["cell_status"].map(_text).eq("PASS")].copy()
    fields = [
        "pair_group",
        "pair",
        "interval",
        "exact_mode",
        "orientation",
        "predicted_inc_trend",
        "vendor_inc_trend",
        "predicted_profile_branch",
        "inferred_vendor_profile_branch",
        "transform_selector_parity_passed",
        "trend_selector_parity_passed",
        "profile_branch_selector_parity_passed",
        "formula_parity_passed",
        "hedge_ratio_abs_error",
        "spread_max_abs_error",
        "zscore_max_abs_error",
        "zscore_roll_max_abs_error",
        "half_life_abs_error",
        "blocker",
        "call_id",
        "response_path",
        "response_sha256",
    ]
    for field in fields:
        if field not in failed:
            failed[field] = ""
    detail = failed[fields].copy()
    detail["root_cause_class"] = detail["blocker"].map(
        lambda value: _text(value) or "unclassified_holdout_failure"
    )
    detail["corrective_order"] = detail["root_cause_class"].map(_corrective_order)
    detail["holdout_reuse_allowed"] = False
    detail["candidate_promotion_authority"] = False
    detail["testnet_order_authority"] = False
    detail["live_trading_authorized"] = False
    return detail


def _corrective_order(cause: object) -> str:
    text = _text(cause)
    if text == "trend_selector_mismatch":
        return "diagnose_vendor_trend_flag_semantics_without_tuning_or_reusing_this_holdout"
    if "profile_branch_selector_mismatch" in text:
        return "treat_profile_branch_rule_as_non_generalizing_and_keep_ou_v6_blocked"
    return "preserve_as_negative_evidence_and_do_not_adapt_against_this_cell"


def _journal_rows(evaluation: pd.DataFrame, status: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    evaluated_at = _text(status.get("evaluated_at_utc"))
    for _, cell in evaluation.iterrows():
        pair = _text(cell.get("pair"))
        assets = pair.split("-", 1)
        passed = _text(cell.get("cell_status")) == "PASS"
        blocker = _text(cell.get("blocker")) or "ou_v6_terminal_family_rejected"
        rows.append(
            {
                "journal_layer": "ou_v6_terminal_evaluation",
                "capture_id": _text(cell.get("call_id")),
                "capture_timestamp_utc": evaluated_at,
                "scanner_refresh_timestamp_utc": evaluated_at,
                "pair": pair,
                "asset_x": assets[0] if assets else "",
                "asset_y": assets[1] if len(assets) > 1 else "",
                "venue": "crypto_wizards_api",
                "exchange_lane": "wizard_research_only",
                "page_route": "api/custom-series",
                "source_path": str(EVALUATION_PATH),
                "evidence_path": _text(cell.get("response_path")),
                "capture_context": "prospective_ou_v6_holdout",
                "source_cycle_id": _text(status.get("result_id")),
                "source_run_type": "scheduled_automation",
                "timeframe": _text(cell.get("interval")),
                "strategy_label": _text(cell.get("exact_mode")),
                "readiness_label": "research_only_terminal_failed_family",
                "risk_gate_pass": False,
                "return_gate_pass": False,
                "sharpe_gate_pass": False,
                "drawdown_gate_pass": False,
                "profit_factor_gate_pass": False,
                "paper_candidate_status": "blocked",
                "execution_compatible": False,
                "account_state_blocked": False,
                "orphan_leg_blocker": False,
                "capture_status": "captured_research_only" if passed else "failed",
                "capture_blocker": blocker,
                "pair_unavailable_flag": False,
                "timeframe_unavailable_flag": False,
                "strategy_unavailable_flag": False,
                "page_not_recognized_flag": False,
                "no_data_flag": False,
                "unchanged_alias_skipped_flag": False,
            }
        )
    return pd.DataFrame(rows)


def _consumed_ou_universe(root: Path) -> tuple[list[str], list[str]]:
    pair_groups: set[str] = set()
    assets: set[str] = set()
    for path in sorted((root / "config").glob("wizard_ou*_holdout.json")):
        payload = _read_json(path)
        for binding in payload.get("holdout_bindings", []):
            if not isinstance(binding, dict):
                continue
            pair_group = _text(binding.get("pair_group") or binding.get("pair"))
            if pair_group:
                pair_groups.add(pair_group)
            for field in ("asset_x", "asset_y"):
                asset = _text(binding.get(field))
                if asset:
                    assets.add(asset)
    return sorted(pair_groups), sorted(assets)


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


def _markdown(payload: dict[str, Any], failures: pd.DataFrame, orientation: pd.DataFrame) -> str:
    lines = [
        "# Wizard OU-v6 Terminal Closure",
        "",
        f"- Status: `{payload['status']}`",
        f"- Decision: `{payload['decision']}`",
        f"- Holdout: `{payload['passed_cells']}/8` cells passed",
        f"- Evidence locked: `{str(payload['evidence_locked']).lower()}`",
        "- OU-v6 activation: `false`",
        "- OU-v7 registration: `false`",
        "- Candidate, testnet, and live authority: `false`",
        "",
        "Passing cells are numerical reconstruction references only. They are not pair,",
        "strategy, testnet, or live-trading approvals. The revealed holdout cannot be reused.",
        "",
        "## Orientation Decision",
        "",
    ]
    for row in orientation.to_dict("records"):
        lines.append(
            f"- `{row.get('pair')}` `{row.get('interval')}`: "
            f"`{row.get('orientation_status')}`; {row.get('permitted_use')}"
        )
    lines.extend(["", "## Failed Cells", ""])
    for row in failures.to_dict("records"):
        lines.append(
            f"- `{row.get('pair')}` `{row.get('exact_mode')}`: `{row.get('root_cause_class')}`"
        )
    lines.extend(["", "## Next Action", "", f"`{payload['next_action']}`", ""])
    if payload.get("blockers"):
        lines.extend(["## Evidence Blockers", ""])
        lines.extend(f"- `{item}`" for item in payload["blockers"])
        lines.append("")
    return "\n".join(lines)
