"""Fail-closed validation for the complete current Wizard research chain."""

from __future__ import annotations

import gzip
import json
import math
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import pandas as pd

from quant_platform.active_pipeline import CommandResult
from quant_platform.orchestration.corrective_runtime import (
    atomic_write_csv,
    atomic_write_text,
    immutable_snapshot_copy,
)

ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "current_wizard_hyperliquid_chain_validation.v1"

STAGES = {
    "refresh": (
        "exhaustive_wizard_api_refresh_manifest.json",
        "exhaustive_wizard_api_refresh_validation.csv",
        "refresh_id",
    ),
    "handoff": (
        "current_wizard_hyperliquid_handoff_manifest.json",
        "current_wizard_hyperliquid_handoff_validation.csv",
        "handoff_id",
    ),
    "history": (
        "current_wizard_hyperliquid_history_manifest.json",
        "current_wizard_hyperliquid_history_validation.csv",
        "history_run_id",
    ),
    "canonical": (
        "current_wizard_hyperliquid_canonical_replay_manifest.json",
        "current_wizard_hyperliquid_replay_validation.csv",
        "canonical_replay_id",
    ),
    "cost": (
        "current_wizard_hyperliquid_cost_manifest.json",
        "current_wizard_hyperliquid_cost_validation.csv",
        "cost_evidence_id",
    ),
    "observed": (
        "current_wizard_hyperliquid_observed_cost_replay_manifest.json",
        "current_wizard_hyperliquid_observed_cost_replay_validation.csv",
        "observed_cost_replay_id",
    ),
    "walkforward": (
        "current_wizard_hyperliquid_walkforward_manifest.json",
        "current_wizard_hyperliquid_walkforward_validation.csv",
        "walkforward_id",
    ),
    "ou_optimal_overlay": (
        "current_wizard_ou_optimal_overlay_manifest.json",
        "current_wizard_ou_optimal_overlay_validation.csv",
        "overlay_id",
    ),
    "regime": (
        "current_wizard_hyperliquid_regime_manifest.json",
        "current_wizard_hyperliquid_regime_validation.csv",
        "regime_attribution_id",
    ),
    "robustness": (
        "current_wizard_hyperliquid_robustness_manifest.json",
        "current_wizard_hyperliquid_robustness_validation.csv",
        "robustness_id",
    ),
    "concentration": (
        "current_wizard_hyperliquid_concentration_manifest.json",
        "current_wizard_hyperliquid_concentration_validation.csv",
        "concentration_id",
    ),
    "failure_attribution": (
        "current_wizard_hyperliquid_failure_attribution_manifest.json",
        "current_wizard_hyperliquid_failure_attribution_validation.csv",
        "failure_attribution_id",
    ),
    "leverage": (
        "current_wizard_hyperliquid_leverage_manifest.json",
        "current_wizard_hyperliquid_leverage_validation.csv",
        "leverage_surface_id",
    ),
    "learning": (
        "current_wizard_hyperliquid_learning_manifest.json",
        "current_wizard_hyperliquid_learning_validation.csv",
        "learning_ledger_id",
    ),
}

TESTNET_FILES = {
    "testnet_preflight": "hyperliquid_testnet_preflight.csv",
    "testnet_margin": "hyperliquid_testnet_margin_snapshot.csv",
    "testnet_lifecycle": "hyperliquid_testnet_lifecycle_gate.csv",
}


def validate_current_wizard_hyperliquid_chain(
    *,
    root: Path = ROOT,
    now: datetime | None = None,
) -> CommandResult:
    """Validate frozen evidence and safety state without network or order actions."""

    as_of = _as_utc(now or datetime.now(timezone.utc))
    active = root / "reports" / "active"
    stage_paths: dict[str, tuple[Path, Path, str]] = {
        name: (active / manifest, active / validation, identity)
        for name, (manifest, validation, identity) in STAGES.items()
    }
    stage_paths["refresh"] = _frozen_refresh_stage_paths(
        root=root,
        handoff_manifest_path=stage_paths["handoff"][0],
        fallback=stage_paths["refresh"],
    )
    testnet_paths = {
        name: active / filename for name, filename in TESTNET_FILES.items()
    }
    all_paths = {
        **{
            f"{name}_manifest": manifest
            for name, (manifest, _, _) in stage_paths.items()
        },
        **{
            f"{name}_validation": validation
            for name, (_, validation, _) in stage_paths.items()
        },
        **testnet_paths,
    }
    missing = [str(path) for path in all_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Current chain-validation inputs missing: " + ",".join(missing)
        )

    manifests = {
        name: _read_json(manifest)
        for name, (manifest, _, _) in stage_paths.items()
    }
    validations = {
        name: _read_csv(validation)
        for name, (_, validation, _) in stage_paths.items()
    }
    preflight = _read_csv(testnet_paths["testnet_preflight"])
    margin = _read_csv(testnet_paths["testnet_margin"])
    lifecycle = _read_csv(testnet_paths["testnet_lifecycle"])

    material = {
        "schema_version": SCHEMA_VERSION,
        "as_of": as_of.isoformat(),
        "input_hashes": {name: _file_hash(path) for name, path in all_paths.items()},
    }
    validation_id = "cwvalidation_" + sha256(
        _canonical_json(material).encode("utf-8")
    ).hexdigest()[:20]
    learning_snapshot = root / _text(
        manifests["learning"].get("artifacts", {}).get("snapshot_manifest")
    )
    if not learning_snapshot.exists():
        raise FileNotFoundError("Current chain learning snapshot manifest is missing")
    snapshot_dir = learning_snapshot.parent / "chain_validation" / validation_id
    snapshot_inputs = snapshot_dir / "inputs"
    snapshot_inputs.mkdir(parents=True, exist_ok=True)
    copied_inputs: dict[str, Path] = {}
    for name, source in all_paths.items():
        target = immutable_snapshot_copy(
            source,
            snapshot_inputs,
            artifact_name=name,
        )
        copied_inputs[name] = target

    checks = _checks(
        root=root,
        manifests=manifests,
        validations=validations,
        preflight=preflight,
        margin=margin,
        lifecycle=lifecycle,
        all_paths=all_paths,
    )
    validation = pd.DataFrame(checks)
    chain_valid = validation["status"].eq("PASS").all()
    if not chain_valid:
        failed = validation.loc[validation["status"].ne("PASS"), "check"].tolist()
        raise ValueError("Current chain validation failed: " + ",".join(failed))

    failure = manifests["failure_attribution"]
    leverage = manifests["leverage"]
    learning = manifests["learning"]
    survivors = int(failure.get("one_x_research_survivors", 0) or 0)
    testnet_ready = int(leverage.get("testnet_1x_lifecycle_ready", 0) or 0)
    margin_status = _first_text(margin, "status")
    margin_blockers = _first_text(margin, "blockers")
    lifecycle_pass = bool(
        not lifecycle.empty
        and "status" in lifecycle.columns
        and lifecycle["status"].eq("PASS").all()
    )
    trade_eligibility_status = (
        "BLOCKED_NO_ONE_X_RESEARCH_SURVIVOR"
        if survivors == 0
        else (
            "BLOCKED_NO_TESTNET_1X_LIFECYCLE_CANDIDATE"
            if testnet_ready == 0
            else (
                "BLOCKED_TESTNET_ACCOUNT_NOT_READY"
                if margin_status != "READY"
                else (
                    "READY_FOR_EXPLICIT_TESTNET_LIFECYCLE_APPROVAL"
                    if not lifecycle_pass
                    else "TESTNET_LIFECYCLE_EVIDENCE_COMPLETE"
                )
            )
        )
    )
    paths = _paths(active, snapshot_dir)
    atomic_write_csv(validation, paths["validation"], index=False)
    atomic_write_csv(validation, paths["snapshot_validation"], index=False)
    summary: dict[str, object] = {
        **material,
        "validation_id": validation_id,
        "chain_status": "PASS",
        "stages_expected": len(STAGES),
        "stages_complete": len(STAGES),
        "checks": int(len(validation)),
        "checks_passed": int(validation["status"].eq("PASS").sum()),
        "experiment_authority_count": int(
            manifests["handoff"].get("planned_experiments", 0) or 0
        ),
        "exhaustive_source_rows_accounted": int(
            manifests["refresh"].get("api_source_rows_accounted", 0) or 0
        ),
        "ou_optimal_overlay_id": _text(
            manifests["ou_optimal_overlay"].get("overlay_id")
        ),
        "ou_optimal_true_rows": int(
            manifests["ou_optimal_overlay"].get("ou_optimal_true_rows", 0) or 0
        ),
        "ou_optimal_false_rows": int(
            manifests["ou_optimal_overlay"].get("ou_optimal_false_rows", 0) or 0
        ),
        "one_x_research_survivors": survivors,
        "leverage_research_candidates": int(
            leverage.get("leverage_candidates_complete", 0) or 0
        ),
        "testnet_1x_lifecycle_ready": testnet_ready,
        "trade_eligibility_status": trade_eligibility_status,
        "testnet_no_order_preflight_ready": _first_truthy(
            preflight, "ready_for_no_order_preflight"
        ),
        "testnet_submit_orders_enabled": _first_truthy(
            preflight, "submit_orders_enabled"
        ),
        "testnet_margin_status": margin_status,
        "testnet_margin_blockers": margin_blockers,
        "testnet_lifecycle_receipt_complete": lifecycle_pass,
        "learning_records": int(learning.get("records", 0) or 0),
        "training_eligible_records": int(
            learning.get("training_eligible_records", 0) or 0
        ),
        "order_submission_performed": False,
        "testnet_order_authority": False,
        "promotion_authority": False,
        "live_trading_authorized": False,
        "stage_identities": {
            name: _text(manifests[name].get(identity))
            for name, (_, _, identity) in stage_paths.items()
        },
        "artifacts": {name: _relative(path, root) for name, path in paths.items()},
        "input_snapshots": {
            name: _relative(path, root) for name, path in copied_inputs.items()
        },
    }
    manifest_text = json.dumps(summary, indent=2, sort_keys=True)
    summary_text = _summary_markdown(summary)
    for key in ("manifest", "snapshot_manifest"):
        atomic_write_text(paths[key], manifest_text, encoding="utf-8")
    for key in ("summary_md", "snapshot_summary_md"):
        atomic_write_text(paths[key], summary_text, encoding="utf-8")
    return CommandResult(paths=paths, summary=summary)


def _frozen_refresh_stage_paths(
    *,
    root: Path,
    handoff_manifest_path: Path,
    fallback: tuple[Path, Path, str],
) -> tuple[Path, Path, str]:
    if not handoff_manifest_path.exists():
        return fallback
    handoff = _read_json(handoff_manifest_path)
    input_snapshots = handoff.get("input_snapshots", {})
    if not isinstance(input_snapshots, dict):
        raise ValueError("Current handoff input_snapshots must be an object")
    refresh_manifest_path = root / _text(input_snapshots.get("refresh_manifest"))
    if not refresh_manifest_path.is_file():
        raise FileNotFoundError("Frozen handoff refresh manifest is missing")
    refresh = _read_json(refresh_manifest_path)
    artifacts = refresh.get("artifacts", {})
    if not isinstance(artifacts, dict):
        raise ValueError("Frozen refresh artifacts must be an object")
    snapshot_validation = _text(artifacts.get("snapshot_validation"))
    refresh_validation_path = (
        root / snapshot_validation if snapshot_validation else fallback[1]
    )
    if not refresh_validation_path.is_file():
        raise FileNotFoundError("Frozen handoff refresh validation is missing")
    return refresh_manifest_path, refresh_validation_path, fallback[2]


def _checks(
    *,
    root: Path,
    manifests: dict[str, dict[str, object]],
    validations: dict[str, pd.DataFrame],
    preflight: pd.DataFrame,
    margin: pd.DataFrame,
    lifecycle: pd.DataFrame,
    all_paths: dict[str, Path],
) -> list[dict[str, object]]:
    checks: list[tuple[str, str, bool, str]] = []
    handoff = manifests["handoff"]
    authority_count = int(handoff.get("planned_experiments", 0) or 0)
    authority_handoff = _text(handoff.get("handoff_id"))
    authority_refresh = _text(handoff.get("refresh_id"))
    refresh = manifests["refresh"]
    overlay = manifests["ou_optimal_overlay"]
    refresh_rows = int(refresh.get("api_source_rows_accounted", 0) or 0)
    overlay_rows = int(overlay.get("source_rows_accounted", 0) or 0)
    overlay_true = int(overlay.get("ou_optimal_true_rows", 0) or 0)
    overlay_false = int(overlay.get("ou_optimal_false_rows", 0) or 0)
    checks.append(
        (
            "authority_count_positive",
            "accounting",
            authority_count > 0,
            str(authority_count),
        )
    )
    checks.append(
        (
            "handoff_refresh_manifest_lineage",
            "lineage",
            authority_refresh == _text(refresh.get("refresh_id")),
            authority_refresh,
        )
    )
    checks.append(
        (
            "history_handoff_lineage",
            "lineage",
            _text(manifests["history"].get("handoff_id")) == authority_handoff,
            authority_handoff,
        )
    )
    checks.append(
        (
            "history_refresh_lineage",
            "lineage",
            _text(manifests["history"].get("refresh_id")) == authority_refresh,
            authority_refresh,
        )
    )
    checks.append(
        (
            "ou_optimal_overlay_refresh_lineage",
            "lineage",
            _text(overlay.get("refresh_id")) == authority_refresh,
            authority_refresh,
        )
    )
    checks.append(
        (
            "ou_optimal_overlay_handoff_lineage",
            "lineage",
            _text(overlay.get("handoff_id")) == authority_handoff,
            authority_handoff,
        )
    )
    checks.append(
        (
            "refresh_source_rows_accounted",
            "accounting",
            refresh_rows > 0
            and refresh_rows == int(refresh.get("api_source_rows", 0) or 0),
            f"accounted={refresh_rows};source={refresh.get('api_source_rows', 0)}",
        )
    )
    checks.append(
        (
            "ou_optimal_overlay_source_rows_accounted",
            "accounting",
            overlay_rows == refresh_rows,
            f"overlay={overlay_rows};refresh={refresh_rows}",
        )
    )
    checks.append(
        (
            "ou_optimal_overlay_true_false_reconcile",
            "accounting",
            overlay_true + overlay_false == overlay_rows,
            f"true={overlay_true};false={overlay_false};rows={overlay_rows}",
        )
    )
    checks.append(
        (
            "ou_optimal_overlay_semantics",
            "fidelity",
            _text(overlay.get("scanner_overlay")) == "ou_optimal"
            and len(overlay.get("pair_page_exact_modes", [])) == 7
            and int(overlay.get("pair_page_ou_optimal_captured_rows", 0) or 0)
            == 0,
            "seven_pair_page_modes;zero_independent_ou_optimal_captures",
        )
    )
    checks.append(
        (
            "ou_optimal_overlay_has_no_authority",
            "safety",
            not _truthy(overlay.get("acceptance_authority"))
            and not _truthy(overlay.get("promotion_authority"))
            and not _truthy(overlay.get("order_submission_performed"))
            and not _truthy(overlay.get("live_trading_authorized")),
            "all_authority=false",
        )
    )
    checks.append(
        (
            "canonical_history_lineage",
            "lineage",
            _text(manifests["canonical"].get("history_run_id"))
            == _text(manifests["history"].get("history_run_id")),
            _text(manifests["history"].get("history_run_id")),
        )
    )
    lineage_specs = (
        ("cost", "canonical_replay_id", "canonical", "canonical_replay_id"),
        ("observed", "cost_evidence_id", "cost", "cost_evidence_id"),
        (
            "observed",
            "canonical_replay_id",
            "canonical",
            "canonical_replay_id",
        ),
        (
            "walkforward",
            "observed_cost_replay_id",
            "observed",
            "observed_cost_replay_id",
        ),
        ("regime", "walkforward_id", "walkforward", "walkforward_id"),
        ("robustness", "walkforward_id", "walkforward", "walkforward_id"),
        (
            "concentration",
            "robustness_id",
            "robustness",
            "robustness_id",
        ),
        (
            "concentration",
            "regime_attribution_id",
            "regime",
            "regime_attribution_id",
        ),
        (
            "failure_attribution",
            "concentration_id",
            "concentration",
            "concentration_id",
        ),
        (
            "leverage",
            "failure_attribution_id",
            "failure_attribution",
            "failure_attribution_id",
        ),
        ("leverage", "walkforward_id", "walkforward", "walkforward_id"),
        (
            "learning",
            "concentration_id",
            "concentration",
            "concentration_id",
        ),
        (
            "learning",
            "failure_attribution_id",
            "failure_attribution",
            "failure_attribution_id",
        ),
        (
            "learning",
            "leverage_surface_id",
            "leverage",
            "leverage_surface_id",
        ),
    )
    for stage, field, upstream_stage, upstream_field in lineage_specs:
        expected = _text(manifests[upstream_stage].get(upstream_field))
        actual = _text(manifests[stage].get(field))
        checks.append(
            (
                f"{stage}_{field}_lineage",
                "lineage",
                bool(expected and actual == expected),
                f"actual={actual};expected={expected}",
            )
        )

    all_cell_stages = (
        ("canonical", "experiments_accounted"),
        ("cost", "experiments_accounted"),
        ("observed", "experiments_accounted"),
        ("walkforward", "experiments_accounted"),
        ("regime", "experiments_accounted"),
        ("robustness", "experiments_accounted"),
        ("concentration", "experiments_accounted"),
        ("failure_attribution", "experiments_accounted"),
        ("leverage", "experiments_accounted"),
        ("learning", "records"),
    )
    for stage, field in all_cell_stages:
        value = int(manifests[stage].get(field, 0) or 0)
        checks.append(
            (
                f"{stage}_all_cell_accounting",
                "accounting",
                value == authority_count,
                f"actual={value};expected={authority_count}",
            )
        )

    for stage, frame in validations.items():
        passed = bool(
            not frame.empty
            and "status" in frame.columns
            and frame["status"].eq("PASS").all()
        )
        checks.append(
            (
                f"{stage}_validation_passes",
                "validation",
                passed,
                _relative(all_paths[f"{stage}_validation"], root),
            )
        )
    for stage, manifest in manifests.items():
        artifacts = manifest.get("artifacts", {})
        if not isinstance(artifacts, dict):
            checks.append(
                (
                    f"{stage}_artifact_map_present",
                    "reproducibility",
                    False,
                    "manifest artifacts is not an object",
                )
            )
            continue
        for key, active_value in artifacts.items():
            if key.startswith("snapshot_"):
                continue
            snapshot_key = f"snapshot_{key}"
            if snapshot_key not in artifacts:
                continue
            snapshot_path = root / _text(artifacts[snapshot_key])
            active_path = (
                snapshot_path if stage == "refresh" else root / _text(active_value)
            )
            paths_exist = active_path.is_file() and snapshot_path.is_file()
            hashes_match = bool(
                paths_exist
                and _artifact_content_hash(active_path)
                == _artifact_content_hash(snapshot_path)
            )
            checks.append(
                (
                    f"{stage}_{key}_active_snapshot_parity",
                    "reproducibility",
                    hashes_match,
                    (
                        f"active={_relative(active_path, root)};"
                        f"snapshot={_relative(snapshot_path, root)}"
                    ),
                )
            )
    for stage, manifest in manifests.items():
        checks.append(
            (
                f"{stage}_live_authority_false",
                "safety",
                not _truthy(manifest.get("live_trading_authorized")),
                "live_trading_authorized=false",
            )
        )
    checks.extend(
        [
            (
                "testnet_no_order_preflight_ready",
                "testnet",
                _first_truthy(preflight, "ready_for_no_order_preflight"),
                _first_text(preflight, "blockers"),
            ),
            (
                "testnet_submission_switch_disabled",
                "safety",
                not _first_truthy(preflight, "submit_orders_enabled"),
                "submit_orders_enabled=false",
            ),
            (
                "testnet_margin_state_explicit",
                "testnet",
                _first_text(margin, "status") in {"READY", "BLOCKED"},
                _first_text(margin, "blockers"),
            ),
            (
                "testnet_lifecycle_execution_disabled",
                "safety",
                bool(
                    not lifecycle.empty
                    and "execution_allowed" in lifecycle.columns
                    and not lifecycle["execution_allowed"].map(_truthy).any()
                ),
                "execution_allowed=false",
            ),
            (
                "learning_has_no_order_submission",
                "safety",
                not _truthy(manifests["learning"].get("order_submission_performed")),
                "order_submission_performed=false",
            ),
            (
                "learning_has_no_paper_or_live_labels",
                "learning",
                int(manifests["learning"].get("paper_label_records", 0) or 0) == 0
                and int(manifests["learning"].get("live_label_records", 0) or 0)
                == 0,
                "paper_labels=0;live_labels=0",
            ),
        ]
    )
    return [
        {
            "schema_version": SCHEMA_VERSION,
            "check": check,
            "category": category,
            "status": "PASS" if passed else "FAIL",
            "evidence": evidence,
            "order_submission_performed": False,
            "live_trading_authorized": False,
        }
        for check, category, passed, evidence in checks
    ]


def _paths(active: Path, snapshot: Path) -> dict[str, Path]:
    stem = "current_wizard_hyperliquid_chain_validation"
    return {
        "validation": active / f"{stem}.csv",
        "manifest": active / f"{stem}_manifest.json",
        "summary_md": active / f"{stem}_summary.md",
        "snapshot_validation": snapshot / "validation.csv",
        "snapshot_manifest": snapshot / "manifest.json",
        "snapshot_summary_md": snapshot / "summary.md",
    }


def _summary_markdown(summary: dict[str, object]) -> str:
    return "\n".join(
        [
            "# Current Wizard To Hyperliquid Chain Validation",
            "",
            "This validates frozen lineage, all-cell accounting, learning separation, and safety state. It performs no network call, transfer, approval, or order submission.",
            "",
            f"- validation: {summary['validation_id']}",
            f"- chain status: {summary['chain_status']}",
            f"- stages complete: {summary['stages_complete']} of {summary['stages_expected']}",
            f"- checks passed: {summary['checks_passed']} of {summary['checks']}",
            f"- experiments accounted: {summary['experiment_authority_count']}",
            f"- exhaustive source rows accounted: {summary['exhaustive_source_rows_accounted']}",
            f"- OU Optimal overlay: {summary['ou_optimal_overlay_id']}",
            "- OU Optimal true / false rows: "
            f"{summary['ou_optimal_true_rows']} / {summary['ou_optimal_false_rows']}",
            f"- 1x research survivors: {summary['one_x_research_survivors']}",
            f"- trade eligibility: {summary['trade_eligibility_status']}",
            f"- Testnet no-order preflight ready: {summary['testnet_no_order_preflight_ready']}",
            f"- Testnet margin status: {summary['testnet_margin_status']}",
            f"- Testnet margin blockers: {summary['testnet_margin_blockers']}",
            f"- Testnet lifecycle receipt complete: {summary['testnet_lifecycle_receipt_complete']}",
            f"- learning records: {summary['learning_records']}",
            "- order submission performed: false",
            "- live trading authorized: false",
            "",
        ]
    )


def _first_text(frame: pd.DataFrame, column: str) -> str:
    if frame.empty or column not in frame.columns:
        return ""
    return _text(frame.iloc[0][column])


def _first_truthy(frame: pd.DataFrame, column: str) -> bool:
    return _truthy(_first_text(frame, column))


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False, low_memory=False)


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _artifact_content_hash(path: Path) -> str:
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as handle:
            return sha256(handle.read()).hexdigest()
    return _file_hash(path)


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _as_utc(value: datetime) -> datetime:
    return (
        value.astimezone(timezone.utc)
        if value.tzinfo
        else value.replace(tzinfo=timezone.utc)
    )


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return _text(value).lower() in {"1", "true", "yes", "y", "pass", "ready"}


def _text(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()
