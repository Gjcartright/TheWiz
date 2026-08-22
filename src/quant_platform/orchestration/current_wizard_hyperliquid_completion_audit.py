"""Objective-level completion audit for the current Wizard-to-Hyperliquid pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd

from quant_platform.active_pipeline import (
    CommandResult,
    _read_csv,
    _read_json,
    _write_csv,
    _write_json,
    _write_text,
)
from quant_platform.economic_contract import CANONICAL_WIZARD_MODES


ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "current_wizard_hyperliquid_completion_audit.v2"
EXACT_MODES = CANONICAL_WIZARD_MODES
ORIENTATIONS = ("original", "reverse")
SATISFIED_STATUSES = {"PROVEN", "CONDITIONALLY_PROVEN"}


def build_current_wizard_hyperliquid_completion_audit(
    *, root: Path = ROOT, now: datetime | None = None
) -> CommandResult:
    """Prove or block every explicit objective requirement from current evidence."""

    as_of = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    active = root / "reports" / "active"
    inputs = _input_paths(root)
    hashes = {
        str(path.relative_to(root)): _sha256_file(path)
        for path in inputs
        if path.is_file()
    }
    audit_id = "cwcompletion_" + sha256(
        json.dumps(hashes, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]

    refresh = _read_json(active / "exhaustive_wizard_api_refresh_manifest.json")
    canonical = _read_json(
        active / "current_wizard_hyperliquid_canonical_replay_manifest.json"
    )
    leverage = _read_json(active / "current_wizard_hyperliquid_leverage_manifest.json")
    learning = _read_json(active / "current_wizard_hyperliquid_learning_manifest.json")
    chain = _read_json(
        active / "current_wizard_hyperliquid_chain_validation_manifest.json"
    )
    protocol = _read_json(
        active / "current_wizard_hyperliquid_testnet_protocol_manifest.json"
    )
    daily = _read_json(
        active / "current_wizard_hyperliquid_daily_run_manifest.json"
    )
    matrix = _read_csv(active / "current_wizard_hyperliquid_experiment_matrix.csv")
    handoff_validation = _read_csv(
        active / "current_wizard_hyperliquid_handoff_validation.csv"
    )
    leverage_validation = _read_csv(
        active / "current_wizard_hyperliquid_leverage_validation.csv"
    )
    learning_validation = _read_csv(
        active / "current_wizard_hyperliquid_learning_validation.csv"
    )
    chain_validation = _read_csv(
        active / "current_wizard_hyperliquid_chain_validation.csv"
    )
    live_lock = _read_csv(active / "current_wizard_hyperliquid_live_lock.csv")

    pair_groups = _as_int(refresh.get("api_pair_groups"))
    source_rows = _as_int(refresh.get("api_source_rows"))
    source_rows_accounted = _as_int(refresh.get("api_source_rows_accounted"))
    expected_cells = pair_groups * len(EXACT_MODES) * len(ORIENTATIONS)
    observed_groups = (
        int(matrix["pair_group_key"].nunique())
        if "pair_group_key" in matrix.columns
        else 0
    )
    observed_cells = _unique_cell_count(matrix)
    no_matrix_prefilters = _column_all_false(matrix, "discovery_prefilter_applied")
    blocked_rows_explicit = _blocked_rows_have_reasons(matrix)
    handoff_pass = _validation_all_pass(handoff_validation)

    prefilters = refresh.get("discovery_prefilters", {})
    no_prefilters = isinstance(prefilters, dict) and all(
        value in (None, "", False) for value in prefilters.values()
    )
    refresh_proven = bool(
        refresh.get("sweep_complete")
        and source_rows > 0
        and source_rows == source_rows_accounted
        and no_prefilters
        and refresh.get("no_silent_drops") is True
    )

    rows: list[dict[str, object]] = []
    rows.append(
        _row(
            audit_id,
            "exhaustive_discovery_no_prefilter",
            "Capture every current Wizard source row without Sharpe or other discovery filters",
            "discovery",
            "PROVEN" if refresh_proven else "UNPROVEN",
            "sweep_complete=true; source_rows=accounted; all discovery prefilters null",
            f"sweep_complete={refresh.get('sweep_complete')}; source_rows={source_rows}; accounted={source_rows_accounted}; prefilters={json.dumps(prefilters, sort_keys=True)}",
            "" if refresh_proven else "exhaustive_refresh_or_prefilter_contract_not_proven",
            "reports/active/exhaustive_wizard_api_refresh_manifest.json",
            "refresh the exhaustive Wizard source and rebuild its accounting manifest",
        )
    )

    pair_accounting_proven = bool(
        pair_groups > 0
        and observed_groups == pair_groups
        and handoff_pass
        and blocked_rows_explicit
    )
    rows.append(
        _row(
            audit_id,
            "every_pair_accounted",
            "Represent every Wizard pair in the Hyperliquid handoff with explicit status",
            "coverage",
            "PROVEN" if pair_accounting_proven else "UNPROVEN",
            "matrix_pair_groups=refresh_pair_groups; handoff validation PASS; every blocked row has a reason",
            f"refresh_pair_groups={pair_groups}; matrix_pair_groups={observed_groups}; handoff_validation_pass={handoff_pass}; blocked_rows_explicit={blocked_rows_explicit}",
            "" if pair_accounting_proven else "pair_accounting_or_blocker_reason_incomplete",
            "reports/active/current_wizard_hyperliquid_experiment_matrix.csv;reports/active/current_wizard_hyperliquid_handoff_validation.csv",
            "rebuild the handoff and resolve missing pair or blocker rows",
        )
    )

    required_modes = tuple(refresh.get("exact_modes_required", ()))
    required_orientations = tuple(refresh.get("orientations_required", ()))
    matrix_contract_proven = bool(
        set(required_modes) == set(EXACT_MODES)
        and set(required_orientations) == set(ORIENTATIONS)
        and expected_cells > 0
        and len(matrix) == expected_cells
        and observed_cells == expected_cells
        and no_matrix_prefilters
    )
    rows.append(
        _row(
            audit_id,
            "exact_modes_and_asymmetric_orientations",
            "Test every applicable exact mode in original and reverse orientation",
            "experiment_design",
            "PROVEN" if matrix_contract_proven else "UNPROVEN",
            f"{pair_groups} pairs x {len(EXACT_MODES)} exact modes x 2 orientations = {expected_cells} unique cells",
            f"matrix_rows={len(matrix)}; unique_cells={observed_cells}; modes={json.dumps(sorted(set(required_modes)))}; orientations={json.dumps(sorted(set(required_orientations)))}; prefilters_absent={no_matrix_prefilters}",
            "" if matrix_contract_proven else "mode_orientation_matrix_incomplete_or_prefiltered",
            "reports/active/current_wizard_hyperliquid_experiment_matrix.csv",
            "rebuild the complete mode-orientation matrix from the frozen refresh",
        )
    )

    mapping_proven = bool(
        pair_accounting_proven
        and _as_int(canonical.get("mapping_blocked_experiments")) >= 0
        and _as_int(canonical.get("experiments_accounted")) == expected_cells
    )
    rows.append(
        _row(
            audit_id,
            "hyperliquid_mapping_with_explicit_blockers",
            "Map every pair to Hyperliquid perpetuals or retain an explicit blocker",
            "venue_mapping",
            "PROVEN" if mapping_proven else "UNPROVEN",
            "all experiment cells accounted; every unmapped cell has an explicit blocker",
            f"experiments_accounted={canonical.get('experiments_accounted')}; mapping_blocked={canonical.get('mapping_blocked_experiments')}; blocked_rows_explicit={blocked_rows_explicit}",
            "" if mapping_proven else "hyperliquid_mapping_accounting_incomplete",
            "reports/active/current_wizard_hyperliquid_experiment_matrix.csv;reports/active/current_wizard_hyperliquid_canonical_replay_manifest.json",
            "refresh Hyperliquid inventory and rebuild mapping accounting",
        )
    )

    canonical_proven = bool(
        _as_int(canonical.get("experiments_accounted")) == expected_cells
        and _as_int(canonical.get("unique_experiment_ids")) == expected_cells
        and float(canonical.get("canonical_replay_leverage", 0.0) or 0.0) == 1.0
        and canonical.get("train_only_parameter_fit") is True
        and canonical.get("test_only_performance_measurement") is True
        and _as_int(canonical.get("replay_status_count_total")) == expected_cells
    )
    rows.append(
        _row(
            audit_id,
            "canonical_one_x_point_in_time_replay",
            "Run canonical 1x, point-in-time, train-only-fit replays with all outcomes retained",
            "replay",
            "PROVEN" if canonical_proven else "UNPROVEN",
            f"{expected_cells} unique cells accounted at 1x with train-only fit and test-only measurement",
            f"accounted={canonical.get('experiments_accounted')}; complete={canonical.get('research_replays_complete')}; rank_eligible={canonical.get('research_rank_eligible_replays')}; point_in_time_blocked={canonical.get('blocked_point_in_time_history')}; mapping_blocked={canonical.get('mapping_blocked_experiments')}",
            "" if canonical_proven else "canonical_one_x_replay_contract_not_proven",
            "reports/active/current_wizard_hyperliquid_canonical_replay_manifest.json",
            "rebuild the canonical replay and reconcile every status count",
        )
    )

    one_x_survivors = _as_int(leverage.get("one_x_research_survivors_selected"))
    leverage_counts_match = bool(
        _as_int(leverage.get("experiments_accounted")) == expected_cells
        and _as_int(leverage.get("scenario_rows"))
        == _as_int(leverage.get("expected_scenario_rows"))
        and _validation_all_pass(leverage_validation)
    )
    leverage_status = "PROVEN"
    leverage_blocker = ""
    if one_x_survivors == 0 and leverage_counts_match:
        leverage_status = "CONDITIONALLY_PROVEN"
        leverage_blocker = "no_one_x_research_survivor_so_no_leverage_scenario_is_permitted"
    elif not leverage_counts_match:
        leverage_status = "UNPROVEN"
        leverage_blocker = "leverage_or_margin_scenario_accounting_incomplete"
    rows.append(
        _row(
            audit_id,
            "separate_leverage_and_margin_scenarios",
            "Keep leverage and margin analysis separate and conditional on a canonical 1x survivor",
            "leverage_risk",
            leverage_status,
            "scenario_rows=expected_scenario_rows; leverage cannot create eligibility before a 1x survivor",
            f"one_x_survivors={one_x_survivors}; leverage_candidates={leverage.get('leverage_candidates_complete')}; scenario_rows={leverage.get('scenario_rows')}; expected={leverage.get('expected_scenario_rows')}; validation_pass={_validation_all_pass(leverage_validation)}",
            leverage_blocker,
            "reports/active/current_wizard_hyperliquid_leverage_manifest.json;reports/active/current_wizard_hyperliquid_leverage_validation.csv",
            "run leverage and margin surfaces only after the refreshed 1x gate selects a survivor",
        )
    )

    protocol_proven = bool(
        protocol.get("protocol_status") == "PASS"
        and protocol.get("required_scenarios_complete") is True
        and _as_int(protocol.get("scenarios"))
        == _as_int(protocol.get("scenarios_passed"))
        and protocol.get("simulation_is_testnet_proof") is False
    )
    testnet_candidates = _as_int(protocol.get("actual_testnet_candidates"))
    testnet_proven = _as_int(protocol.get("actual_testnet_lifecycle_proven"))
    if testnet_candidates == 0 and one_x_survivors == 0 and protocol_proven:
        testnet_status = "CONDITIONALLY_PROVEN"
        testnet_blocker = "no_eligible_one_x_configuration_exists_for_actual_testnet_validation"
    elif testnet_candidates > 0 and testnet_proven == testnet_candidates and protocol_proven:
        testnet_status = "PROVEN"
        testnet_blocker = ""
    else:
        testnet_status = "UNPROVEN"
        testnet_blocker = "eligible_testnet_lifecycle_evidence_incomplete"
    rows.append(
        _row(
            audit_id,
            "eligible_hyperliquid_testnet_validation",
            "Validate every eligible configuration through the bounded Hyperliquid Testnet lifecycle",
            "testnet",
            testnet_status,
            "protocol scenarios pass; each eligible candidate has an actual structured lifecycle receipt",
            f"protocol={protocol.get('protocol_status')}; scenarios={protocol.get('scenarios_passed')}/{protocol.get('scenarios')}; candidates={testnet_candidates}; actual_lifecycles={testnet_proven}; one_x_survivors={one_x_survivors}",
            testnet_blocker,
            "reports/active/current_wizard_hyperliquid_testnet_protocol_manifest.json;reports/active/hyperliquid_testnet_lifecycle_gate.csv;reports/active/hyperliquid_testnet_lifecycle_evidence_capture.csv",
            "capture a signed, exchange-reconciled lifecycle receipt only when a refreshed candidate becomes eligible",
        )
    )

    learning_proven = bool(
        _as_int(learning.get("records")) == expected_cells
        and _as_int(learning.get("unique_experiment_ids")) == expected_cells
        and learning.get("wizard_is_label_authority") is False
        and _validation_all_pass(learning_validation)
    )
    rows.append(
        _row(
            audit_id,
            "dated_learning_without_hindsight",
            "Preserve dated learning outcomes without Wizard or future-label authority",
            "learning",
            "PROVEN" if learning_proven else "UNPROVEN",
            f"one dated research outcome per {expected_cells} cells; learning validation all PASS; Wizard is not label authority",
            f"records={learning.get('records')}; training_eligible={learning.get('training_eligible_records')}; paper_labels={learning.get('paper_label_records')}; live_labels={learning.get('live_label_records')}; validation_pass={_validation_all_pass(learning_validation)}",
            "" if learning_proven else "learning_lineage_or_hindsight_controls_not_proven",
            "reports/active/current_wizard_hyperliquid_learning_manifest.json;reports/active/current_wizard_hyperliquid_learning_validation.csv",
            "rebuild the dated ledger and resolve every failed learning validation",
        )
    )

    required_reports = _required_report_paths(root)
    missing_reports = [str(path.relative_to(root)) for path in required_reports if not path.is_file()]
    reports_proven = bool(
        not missing_reports
        and chain.get("chain_status") == "PASS"
        and _as_int(chain.get("checks")) == _as_int(chain.get("checks_passed"))
        and _validation_all_pass(chain_validation)
    )
    rows.append(
        _row(
            audit_id,
            "coverage_ranking_risk_and_reproducibility_reports",
            "Publish complete coverage, ranking, risk, failure, and reproducibility evidence",
            "reporting",
            "PROVEN" if reports_proven else "UNPROVEN",
            "all required reports exist and the frozen-chain validation passes every check",
            f"required_reports={len(required_reports)}; missing={json.dumps(missing_reports)}; chain_checks={chain.get('checks_passed')}/{chain.get('checks')}; chain_status={chain.get('chain_status')}",
            "" if reports_proven else "required_report_or_chain_validation_missing",
            "reports/active/current_wizard_hyperliquid_chain_validation_manifest.json;reports/dashboard/command_center.md",
            "rebuild missing reports and rerun frozen-chain validation",
        )
    )

    daily_complete = bool(
        daily.get("execution_requested") is True
        and daily.get("run_status") == "PASS"
        and _as_int(daily.get("stages")) > 0
        and _as_int(daily.get("stages_passed")) == _as_int(daily.get("stages"))
    )
    rows.append(
        _row(
            audit_id,
            "repeatable_fresh_daily_cycle",
            "Execute the complete fail-closed daily research cadence from fresh discovery through monitoring",
            "orchestration",
            "PROVEN" if daily_complete else "BLOCKED",
            "execution_requested=true; run_status=PASS; stages_passed=stages",
            f"run_id={daily.get('daily_run_id')}; execution_requested={daily.get('execution_requested')}; status={daily.get('run_status')}; stages_passed={daily.get('stages_passed')}/{daily.get('stages')}; free_bytes={daily.get('final_free_disk_bytes')}; required_bytes={daily.get('minimum_free_disk_bytes')}",
            "" if daily_complete else "fresh_daily_cycle_blocked_by_storage_or_stage_failure",
            "reports/active/current_wizard_hyperliquid_daily_run_manifest.json;reports/active/current_wizard_hyperliquid_daily_run_status.csv",
            "restore the storage floor, then run the explicit executable daily pipeline",
        )
    )

    no_silent_drops = bool(
        refresh.get("no_silent_drops") is True
        and source_rows == source_rows_accounted
        and observed_cells == expected_cells
        and _as_int(chain.get("experiment_authority_count")) == expected_cells
    )
    rows.append(
        _row(
            audit_id,
            "no_silent_drops",
            "Account for every source row, pair, and experiment cell through final validation",
            "lineage",
            "PROVEN" if no_silent_drops else "UNPROVEN",
            "source rows, matrix cells, and final authority count reconcile exactly",
            f"source_rows={source_rows_accounted}/{source_rows}; matrix_cells={observed_cells}/{expected_cells}; final_authority={chain.get('experiment_authority_count')}/{expected_cells}",
            "" if no_silent_drops else "source_pair_or_experiment_lineage_gap",
            "reports/active/exhaustive_wizard_api_refresh_manifest.json;reports/active/current_wizard_hyperliquid_chain_validation_manifest.json",
            "trace the first mismatched identifier and rebuild from that stage",
        )
    )

    no_live_authority = _no_live_authority(
        refresh, canonical, leverage, learning, chain, protocol, daily, live_lock
    ) and not (
        active / "hyperliquid_testnet_pair_execution_state.json"
    ).exists() and not (active / "hyperliquid_testnet_smoke_receipt.json").exists()
    rows.append(
        _row(
            audit_id,
            "no_live_trading_authorization",
            "Keep Testnet order authority conditional and live trading permanently unauthorized",
            "authority",
            "PROVEN" if no_live_authority else "UNPROVEN",
            "all manifests deny live authority; live lock is permanent; no executor state or receipt exists",
            f"live_lock={_first_value(live_lock, 'lock_state')}; execution_state_exists={(active / 'hyperliquid_testnet_pair_execution_state.json').exists()}; receipt_exists={(active / 'hyperliquid_testnet_smoke_receipt.json').exists()}",
            "" if no_live_authority else "unexpected_execution_or_live_authority_evidence",
            "reports/active/current_wizard_hyperliquid_live_lock.csv;reports/active/current_wizard_hyperliquid_chain_validation_manifest.json",
            "halt the pipeline and reconcile every authority-bearing artifact",
        )
    )

    frame = pd.DataFrame(rows)
    counts = frame["status"].value_counts().to_dict()
    completion_claim_allowed = bool(frame["status"].isin(SATISFIED_STATUSES).all())
    summary = {
        "schema_version": SCHEMA_VERSION,
        "audit_id": audit_id,
        "as_of": as_of.isoformat(),
        "completion_status": "PASS" if completion_claim_allowed else "BLOCKED",
        "completion_claim_allowed": completion_claim_allowed,
        "requirements": int(len(frame)),
        "proven": int(counts.get("PROVEN", 0)),
        "conditionally_proven": int(counts.get("CONDITIONALLY_PROVEN", 0)),
        "blocked": int(counts.get("BLOCKED", 0)),
        "unproven": int(counts.get("UNPROVEN", 0)),
        "input_hashes": hashes,
        "order_submission_performed": False,
        "live_trading_authorized": False,
    }

    run_dir = root / "reports" / "runs" / "current_wizard_hyperliquid_completion" / audit_id
    outputs = {
        "completion_audit": active / "current_wizard_hyperliquid_completion_audit.csv",
        "completion_manifest": active
        / "current_wizard_hyperliquid_completion_audit_manifest.json",
        "completion_summary": active
        / "current_wizard_hyperliquid_completion_audit_summary.md",
        "dated_completion_audit": run_dir / "completion_audit.csv",
        "dated_completion_manifest": run_dir / "manifest.json",
    }
    _write_csv(frame, outputs["completion_audit"])
    _write_json(outputs["completion_manifest"], summary)
    _write_text(outputs["completion_summary"], _markdown(summary, frame))
    _write_csv(frame, outputs["dated_completion_audit"])
    _write_json(outputs["dated_completion_manifest"], summary)
    return CommandResult(paths=outputs, summary=summary)


def _input_paths(root: Path) -> tuple[Path, ...]:
    active = root / "reports" / "active"
    return (
        active / "exhaustive_wizard_api_refresh_manifest.json",
        active / "current_wizard_hyperliquid_experiment_matrix.csv",
        active / "current_wizard_hyperliquid_handoff_validation.csv",
        active / "current_wizard_hyperliquid_canonical_replay_manifest.json",
        active / "current_wizard_hyperliquid_leverage_manifest.json",
        active / "current_wizard_hyperliquid_leverage_validation.csv",
        active / "current_wizard_hyperliquid_learning_manifest.json",
        active / "current_wizard_hyperliquid_learning_validation.csv",
        active / "current_wizard_hyperliquid_chain_validation_manifest.json",
        active / "current_wizard_hyperliquid_chain_validation.csv",
        active / "current_wizard_hyperliquid_testnet_protocol_manifest.json",
        active / "current_wizard_hyperliquid_daily_run_manifest.json",
        active / "current_wizard_hyperliquid_live_lock.csv",
    )


def _required_report_paths(root: Path) -> tuple[Path, ...]:
    active = root / "reports" / "active"
    return (
        active / "current_wizard_hyperliquid_experiment_matrix.csv",
        active / "current_wizard_hyperliquid_canonical_replay_ranked.csv",
        active / "current_wizard_hyperliquid_walkforward_ranked.csv",
        active / "current_wizard_hyperliquid_concentration_status.csv",
        active / "current_wizard_hyperliquid_failure_attribution.csv",
        active / "current_wizard_hyperliquid_leverage_status.csv",
        active / "current_wizard_hyperliquid_learning_ledger.csv.gz",
        active / "current_wizard_hyperliquid_chain_validation.csv",
        active / "current_wizard_hyperliquid_daily_run_status.csv",
        root / "reports" / "dashboard" / "command_center.md",
    )


def _row(
    audit_id: str,
    requirement_id: str,
    requirement: str,
    scope: str,
    status: str,
    required: str,
    observed: str,
    blocker: str,
    evidence_path: str,
    next_action: str,
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "audit_id": audit_id,
        "requirement_id": requirement_id,
        "requirement": requirement,
        "scope": scope,
        "status": status,
        "required": required,
        "observed": observed,
        "blocker": blocker,
        "evidence_path": evidence_path,
        "next_action": next_action,
        "completion_authority": status in SATISFIED_STATUSES,
        "order_submission_performed": False,
        "live_trading_authorized": False,
    }


def _validation_all_pass(frame: pd.DataFrame) -> bool:
    return bool(
        not frame.empty
        and "status" in frame.columns
        and frame["status"].astype(str).eq("PASS").all()
    )


def _unique_cell_count(frame: pd.DataFrame) -> int:
    columns = ["pair_group_key", "exact_mode", "orientation"]
    if frame.empty or not set(columns).issubset(frame.columns):
        return 0
    return int(frame[columns].drop_duplicates().shape[0])


def _column_all_false(frame: pd.DataFrame, column: str) -> bool:
    if frame.empty or column not in frame.columns:
        return False
    values = frame[column].astype(str).str.strip().str.lower()
    return bool(values.isin({"", "false", "0", "none", "nan"}).all())


def _blocked_rows_have_reasons(frame: pd.DataFrame) -> bool:
    required = {"experiment_status", "experiment_blocker"}
    if frame.empty or not required.issubset(frame.columns):
        return False
    blocked = frame["experiment_status"].astype(str).str.startswith("BLOCKED")
    if not blocked.any():
        return True
    reasons = frame.loc[blocked, "experiment_blocker"].fillna("").astype(str).str.strip()
    return bool(reasons.ne("").all())


def _no_live_authority(*values: object) -> bool:
    manifests = values[:-1]
    live_lock = values[-1]
    manifests_false = all(
        isinstance(value, dict)
        and value.get("live_trading_authorized") is False
        and value.get("order_submission_performed", False) is False
        for value in manifests
    )
    return bool(
        manifests_false
        and isinstance(live_lock, pd.DataFrame)
        and not live_lock.empty
        and _first_value(live_lock, "lock_state") == "PERMANENT_RESEARCH_ONLY"
        and not _truthy(_first_value(live_lock, "live_order_authority"))
        and not _truthy(_first_value(live_lock, "order_submission_performed"))
    )


def _first_value(frame: pd.DataFrame, column: str) -> object:
    if frame.empty or column not in frame.columns:
        return ""
    return frame.iloc[0][column]


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _as_int(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _markdown(summary: dict[str, object], frame: pd.DataFrame) -> str:
    lines = [
        "# Current Wizard Hyperliquid Completion Audit",
        "",
        f"- Audit: `{summary['audit_id']}`",
        f"- Status: `{summary['completion_status']}`",
        f"- Completion claim allowed: `{summary['completion_claim_allowed']}`",
        f"- Requirements: `{summary['requirements']}`",
        f"- Proven: `{summary['proven']}`",
        f"- Conditionally proven: `{summary['conditionally_proven']}`",
        f"- Blocked: `{summary['blocked']}`",
        f"- Unproven: `{summary['unproven']}`",
        "- Order submission performed: `False`",
        "- Live trading authorized: `False`",
        "",
        "| Requirement | Status | Blocker | Next action |",
        "|---|---|---|---|",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            f"| {row.requirement_id} | {row.status} | {row.blocker or ''} | {row.next_action} |"
        )
    return "\n".join(lines) + "\n"
