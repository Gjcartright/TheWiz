"""Operating cadence and permanent live lock for the current Wizard pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shutil

import pandas as pd

from quant_platform.active_pipeline import CommandResult


ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "current_wizard_hyperliquid_operating_cadence.v1"
MINIMUM_FREE_BYTES = 3 * 1024**3

STAGES = (
    (1, "storage_preflight", "before_each_run", "system-check", False, False),
    (
        2,
        "wizard_exhaustive_discovery",
        "daily",
        "crypto-wizards-full-sweep --execute-wizard-sweep",
        True,
        True,
    ),
    (
        3,
        "wizard_refresh_accounting",
        "after_discovery",
        "build-exhaustive-wizard-api-refresh-delta",
        False,
        False,
    ),
    (
        4,
        "hyperliquid_market_inventory",
        "daily",
        "hyperliquid-testnet-market-inventory",
        True,
        False,
    ),
    (
        5,
        "pair_mode_orientation_handoff",
        "after_refresh",
        "build-current-wizard-hyperliquid-handoff",
        False,
        False,
    ),
    (
        6,
        "point_in_time_history",
        "after_handoff",
        "materialize-current-wizard-hyperliquid-history --current-pair-group-keys <all mapped pair_group_keys>",
        True,
        False,
    ),
    (
        7,
        "canonical_one_x_replay",
        "after_history",
        "run-current-wizard-hyperliquid-canonical-replay",
        False,
        False,
    ),
    (
        8,
        "funding_liquidity_cost_evidence",
        "after_canonical_replay",
        "materialize-current-wizard-hyperliquid-cost-evidence --current-pair-group-keys <all replayed pair_group_keys>",
        True,
        False,
    ),
    (
        9,
        "observed_cost_replay",
        "after_cost_evidence",
        "run-current-wizard-hyperliquid-observed-cost-replay",
        False,
        False,
    ),
    (
        10,
        "purged_walkforward",
        "after_observed_cost_replay",
        "run-current-wizard-hyperliquid-walkforward",
        False,
        False,
    ),
    (
        11,
        "ou_optimal_outcome_stratification",
        "after_walkforward",
        "build-current-wizard-ou-optimal-overlay",
        False,
        False,
    ),
    (
        12,
        "causal_regime_attribution",
        "after_walkforward",
        "build-current-wizard-hyperliquid-regime-attribution",
        False,
        False,
    ),
    (
        13,
        "robustness_stress",
        "after_regime_attribution",
        "run-current-wizard-hyperliquid-robustness",
        False,
        False,
    ),
    (
        14,
        "cross_cell_concentration",
        "after_robustness",
        "build-current-wizard-hyperliquid-concentration",
        False,
        False,
    ),
    (
        15,
        "failure_attribution",
        "after_concentration",
        "build-current-wizard-hyperliquid-failure-attribution",
        False,
        False,
    ),
    (
        16,
        "conditional_leverage_surface",
        "only_after_one_x_survivor",
        "build-current-wizard-hyperliquid-leverage-surface",
        False,
        False,
    ),
    (
        17,
        "dated_learning_ledger",
        "after_leverage_gate",
        "build-current-wizard-hyperliquid-learning-ledger",
        False,
        False,
    ),
    (
        18,
        "frozen_chain_validation",
        "after_learning_ledger",
        "validate-current-wizard-hyperliquid-chain",
        False,
        False,
    ),
    (
        19,
        "monitor_dashboard",
        "after_chain_validation",
        "build-command-dashboard --dashboard-refresh-profile monitor",
        False,
        False,
    ),
)

STORAGE_STAGE_MANIFESTS = {
    "concentration": (
        "current_wizard_hyperliquid_concentration_manifest.json",
        "concentration_id",
    ),
    "failure_attribution": (
        "current_wizard_hyperliquid_failure_attribution_manifest.json",
        "failure_attribution_id",
    ),
    "leverage": (
        "current_wizard_hyperliquid_leverage_manifest.json",
        "leverage_surface_id",
    ),
    "learning": (
        "current_wizard_hyperliquid_learning_manifest.json",
        "learning_ledger_id",
    ),
}


def build_current_wizard_hyperliquid_operating_cadence(
    *,
    root: Path = ROOT,
    now: datetime | None = None,
    available_disk_bytes: int | None = None,
    wizard_api_key_present: bool | None = None,
) -> CommandResult:
    """Publish the repeatable research schedule without running it or placing orders."""

    as_of = _as_utc(now or datetime.now(timezone.utc))
    active = root / "reports" / "active"
    chain_path = active / "current_wizard_hyperliquid_chain_validation_manifest.json"
    refresh_path = active / "exhaustive_wizard_api_refresh_manifest.json"
    overlay_path = active / "current_wizard_ou_optimal_overlay_manifest.json"
    storage_manifest_paths = {
        stage: active / filename
        for stage, (filename, _) in STORAGE_STAGE_MANIFESTS.items()
    }
    for path in (
        chain_path,
        refresh_path,
        overlay_path,
        *storage_manifest_paths.values(),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Operating-cadence input missing: {path}")
    chain = _read_json(chain_path)
    refresh = _read_json(refresh_path)
    overlay = _read_json(overlay_path)
    storage_manifests = {
        stage: _read_json(path) for stage, path in storage_manifest_paths.items()
    }
    free_bytes = int(
        available_disk_bytes
        if available_disk_bytes is not None
        else shutil.disk_usage(root).free
    )
    storage_ready = free_bytes >= MINIMUM_FREE_BYTES
    key_ready = bool(
        wizard_api_key_present
        if wizard_api_key_present is not None
        else _key_configured(root, "CRYPTO_WIZARDS_API_KEY")
    )
    chain_ready = _text(chain.get("chain_status")) == "PASS"
    experiment_count = int(chain.get("experiment_authority_count", 0) or 0)
    survivors = int(chain.get("one_x_research_survivors", 0) or 0)
    rows = [
        _stage_row(
            sequence=sequence,
            stage=stage,
            cadence=cadence,
            command=command,
            requires_network=requires_network,
            requires_wizard_auth=requires_wizard_auth,
            storage_ready=storage_ready,
            key_ready=key_ready,
            chain_ready=chain_ready,
            survivors=survivors,
            chain_path=chain_path,
            root=root,
        )
        for (
            sequence,
            stage,
            cadence,
            command,
            requires_network,
            requires_wizard_auth,
        ) in STAGES
    ]
    cadence = pd.DataFrame(rows)
    live_lock = pd.DataFrame(
        [
            {
                "schema_version": SCHEMA_VERSION,
                "lock_state": "PERMANENT_RESEARCH_ONLY",
                "unlock_authority_in_this_pipeline": False,
                "scheduled_testnet_order_authority": False,
                "live_order_authority": False,
                "order_submission_performed": False,
                "reason": (
                    "testnet_lifecycle_requires_eligible_one_x_configuration_and_"
                    "separate_explicit_runtime_approval;live_release_is_out_of_scope"
                ),
                "evidence_path": _relative(chain_path, root),
            }
        ]
    )
    storage_efficiency = pd.DataFrame(
        [
            {
                "schema_version": SCHEMA_VERSION,
                "stage": stage,
                "stage_id": _text(manifest.get(STORAGE_STAGE_MANIFESTS[stage][1])),
                "snapshot_storage_mode": "verified_reference_first",
                "referenced_upstream_bytes": int(
                    manifest.get("referenced_upstream_bytes", 0) or 0
                ),
                "locally_copied_input_bytes": int(
                    manifest.get("locally_copied_input_bytes", 0) or 0
                ),
                "recursive_copy_bytes_avoided": int(
                    manifest.get("referenced_upstream_bytes", 0) or 0
                ),
                "manifest_path": _relative(storage_manifest_paths[stage], root),
                "order_submission_performed": False,
                "live_trading_authorized": False,
            }
            for stage, manifest in storage_manifests.items()
        ]
    )
    validation = _validation(
        cadence=cadence,
        live_lock=live_lock,
        storage_efficiency=storage_efficiency,
        chain=chain,
        refresh=refresh,
        overlay=overlay,
    )
    if not validation["status"].eq("PASS").all():
        failed = validation.loc[validation["status"].ne("PASS"), "check"].tolist()
        raise ValueError("Operating-cadence validation failed: " + ",".join(failed))

    material = {
        "schema_version": SCHEMA_VERSION,
        "as_of": as_of.isoformat(),
        "chain_validation_id": _text(chain.get("validation_id")),
        "refresh_id": _text(refresh.get("refresh_id")),
        "ou_optimal_overlay_id": _text(overlay.get("overlay_id")),
        "input_hashes": {
            "chain_validation_manifest": _file_hash(chain_path),
            "wizard_refresh_manifest": _file_hash(refresh_path),
            "ou_optimal_overlay_manifest": _file_hash(overlay_path),
            **{
                f"{stage}_manifest": _file_hash(path)
                for stage, path in storage_manifest_paths.items()
            },
        },
    }
    cadence_id = "cwcadence_" + sha256(
        _canonical_json(material).encode("utf-8")
    ).hexdigest()[:20]
    cadence.insert(1, "cadence_id", cadence_id)
    live_lock.insert(1, "cadence_id", cadence_id)
    storage_efficiency.insert(1, "cadence_id", cadence_id)
    validation.insert(1, "cadence_id", cadence_id)
    chain_snapshot = root / _text(
        chain.get("artifacts", {}).get("snapshot_manifest")
    )
    if not chain_snapshot.is_file():
        raise FileNotFoundError("Chain validation snapshot manifest is missing")
    snapshot = chain_snapshot.parent / "operating_cadence" / cadence_id
    snapshot.mkdir(parents=True, exist_ok=True)
    paths = _paths(active, snapshot)
    for frame, active_key, snapshot_key in (
        (cadence, "cadence", "snapshot_cadence"),
        (live_lock, "live_lock", "snapshot_live_lock"),
        (
            storage_efficiency,
            "storage_efficiency",
            "snapshot_storage_efficiency",
        ),
        (validation, "validation", "snapshot_validation"),
    ):
        frame.to_csv(paths[active_key], index=False)
        frame.to_csv(paths[snapshot_key], index=False)

    daily_ready = bool(storage_ready and key_ready and chain_ready)
    summary: dict[str, object] = {
        **material,
        "cadence_id": cadence_id,
        "stages": int(len(cadence)),
        "experiment_authority_count": experiment_count,
        "one_x_research_survivors": survivors,
        "storage_ready": storage_ready,
        "free_disk_bytes": free_bytes,
        "minimum_free_disk_bytes": MINIMUM_FREE_BYTES,
        "wizard_api_key_present": key_ready,
        "current_chain_valid": chain_ready,
        "ou_optimal_source_rows_accounted": int(
            overlay.get("source_rows_accounted", 0) or 0
        ),
        "ou_optimal_true_rows": int(overlay.get("ou_optimal_true_rows", 0) or 0),
        "ou_optimal_false_rows": int(
            overlay.get("ou_optimal_false_rows", 0) or 0
        ),
        "referenced_upstream_bytes": int(
            storage_efficiency["referenced_upstream_bytes"].sum()
        ),
        "locally_copied_input_bytes": int(
            storage_efficiency["locally_copied_input_bytes"].sum()
        ),
        "recursive_copy_bytes_avoided": int(
            storage_efficiency["recursive_copy_bytes_avoided"].sum()
        ),
        "daily_research_run_ready": daily_ready,
        "daily_research_run_blocker": ";".join(
            blocker
            for blocker, blocked in (
                ("insufficient_free_space", not storage_ready),
                ("crypto_wizards_api_key_missing", not key_ready),
                ("current_chain_not_valid", not chain_ready),
            )
            if blocked
        ),
        "trade_eligibility_status": _text(chain.get("trade_eligibility_status")),
        "scheduled_testnet_order_authority": False,
        "order_submission_performed": False,
        "promotion_authority": False,
        "live_trading_authorized": False,
        "artifacts": {name: _relative(path, root) for name, path in paths.items()},
    }
    manifest_text = json.dumps(summary, indent=2, sort_keys=True)
    summary_text = _summary_markdown(summary)
    for key in ("manifest", "snapshot_manifest"):
        paths[key].write_text(manifest_text, encoding="utf-8")
    for key in ("summary_md", "snapshot_summary_md"):
        paths[key].write_text(summary_text, encoding="utf-8")
    return CommandResult(paths=paths, summary=summary)


def _stage_row(
    *,
    sequence: int,
    stage: str,
    cadence: str,
    command: str,
    requires_network: bool,
    requires_wizard_auth: bool,
    storage_ready: bool,
    key_ready: bool,
    chain_ready: bool,
    survivors: int,
    chain_path: Path,
    root: Path,
) -> dict[str, object]:
    blockers: list[str] = []
    if stage == "storage_preflight" and not storage_ready:
        blockers.append("insufficient_free_space")
    if stage == "wizard_exhaustive_discovery" and not key_ready:
        blockers.append("crypto_wizards_api_key_missing")
    if sequence <= 17 and stage != "storage_preflight" and not storage_ready:
        blockers.append("storage_preflight_blocked")
    if stage == "conditional_leverage_surface" and survivors == 0:
        blockers.append("no_one_x_research_survivor")
    if stage in {"dated_learning_ledger", "frozen_chain_validation", "monitor_dashboard"}:
        if not chain_ready:
            blockers.append("current_chain_not_valid")
    return {
        "schema_version": SCHEMA_VERSION,
        "sequence": sequence,
        "stage": stage,
        "cadence": cadence,
        "command": f"PYTHONPATH=src python -m quant_platform.cli {command}",
        "requires_network": requires_network,
        "requires_authenticated_wizard": requires_wizard_auth,
        "discovery_prefilters": "none" if stage == "wizard_exhaustive_discovery" else "",
        "run_status": "BLOCKED" if blockers else "READY",
        "blocker": ";".join(dict.fromkeys(blockers)),
        "scheduled_order_submission": False,
        "testnet_order_authority": False,
        "promotion_authority": False,
        "live_trading_authorized": False,
        "evidence_path": _relative(chain_path, root),
    }


def _validation(
    *,
    cadence: pd.DataFrame,
    live_lock: pd.DataFrame,
    storage_efficiency: pd.DataFrame,
    chain: dict[str, object],
    refresh: dict[str, object],
    overlay: dict[str, object],
) -> pd.DataFrame:
    sequence = cadence["sequence"].astype(int).tolist()
    discovery = cadence.loc[cadence["stage"].eq("wizard_exhaustive_discovery")]
    overlay_stage = cadence.loc[
        cadence["stage"].eq("ou_optimal_outcome_stratification")
    ]
    overlay_rows = int(overlay.get("source_rows_accounted", 0) or 0)
    overlay_true = int(overlay.get("ou_optimal_true_rows", 0) or 0)
    overlay_false = int(overlay.get("ou_optimal_false_rows", 0) or 0)
    checks = (
        ("stage_count", len(cadence) == len(STAGES), str(len(cadence))),
        ("stage_identity_unique", cadence["stage"].is_unique, "stage"),
        ("sequence_contiguous", sequence == list(range(1, len(STAGES) + 1)), str(sequence)),
        (
            "discovery_has_no_prefilters",
            len(discovery) == 1 and discovery.iloc[0]["discovery_prefilters"] == "none",
            "discovery_prefilters=none",
        ),
        (
            "current_chain_passes",
            _text(chain.get("chain_status")) == "PASS",
            _text(chain.get("validation_id")),
        ),
        (
            "current_experiment_count_positive",
            int(chain.get("experiment_authority_count", 0) or 0) > 0,
            str(chain.get("experiment_authority_count", 0)),
        ),
        (
            "refresh_policy_exhaustive",
            _text(refresh.get("discovery_policy")) == "exhaustive_no_prefilter",
            _text(refresh.get("discovery_policy")),
        ),
        (
            "ou_optimal_overlay_stage_present",
            len(overlay_stage) == 1,
            "ou_optimal_outcome_stratification",
        ),
        (
            "ou_optimal_overlay_accounts_refresh_rows",
            overlay_rows > 0
            and overlay_rows
            == int(refresh.get("api_source_rows_accounted", 0) or 0),
            f"overlay={overlay_rows};refresh={refresh.get('api_source_rows_accounted', 0)}",
        ),
        (
            "ou_optimal_true_false_reconcile",
            overlay_true + overlay_false == overlay_rows,
            f"true={overlay_true};false={overlay_false};rows={overlay_rows}",
        ),
        (
            "ou_optimal_is_scanner_overlay_not_mode",
            _text(overlay.get("scanner_overlay")) == "ou_optimal"
            and len(overlay.get("pair_page_exact_modes", [])) == 7
            and int(overlay.get("pair_page_ou_optimal_captured_rows", 0) or 0)
            == 0,
            "seven_pair_page_modes;zero_independent_ou_optimal_captures",
        ),
        (
            "ou_optimal_overlay_has_no_authority",
            not bool(overlay.get("acceptance_authority", False))
            and not bool(overlay.get("promotion_authority", False))
            and not bool(overlay.get("order_submission_performed", False))
            and not bool(overlay.get("live_trading_authorized", False)),
            "all_authority=false",
        ),
        (
            "scheduled_order_submission_false",
            not cadence["scheduled_order_submission"].astype(bool).any(),
            "scheduled_order_submission=false",
        ),
        (
            "scheduled_testnet_authority_false",
            not cadence["testnet_order_authority"].astype(bool).any(),
            "testnet_order_authority=false",
        ),
        (
            "scheduled_live_authority_false",
            not cadence["live_trading_authorized"].astype(bool).any(),
            "live_trading_authorized=false",
        ),
        (
            "permanent_live_lock",
            len(live_lock) == 1
            and live_lock.iloc[0]["lock_state"] == "PERMANENT_RESEARCH_ONLY"
            and not bool(live_lock.iloc[0]["unlock_authority_in_this_pipeline"]),
            "PERMANENT_RESEARCH_ONLY",
        ),
        (
            "storage_stage_count",
            len(storage_efficiency) == len(STORAGE_STAGE_MANIFESTS),
            str(len(storage_efficiency)),
        ),
        (
            "storage_stage_ids_present",
            storage_efficiency["stage_id"].astype(str).str.strip().ne("").all(),
            "stage_id",
        ),
        (
            "storage_references_present",
            storage_efficiency["referenced_upstream_bytes"].astype(int).gt(0).all(),
            "referenced_upstream_bytes>0",
        ),
        (
            "storage_live_authority_false",
            not storage_efficiency["live_trading_authorized"].astype(bool).any(),
            "live_trading_authorized=false",
        ),
    )
    return pd.DataFrame(
        [
            {
                "schema_version": SCHEMA_VERSION,
                "check": check,
                "status": "PASS" if passed else "FAIL",
                "evidence": evidence,
                "order_submission_performed": False,
                "live_trading_authorized": False,
            }
            for check, passed, evidence in checks
        ]
    )


def _paths(active: Path, snapshot: Path) -> dict[str, Path]:
    stem = "current_wizard_hyperliquid_operating_cadence"
    return {
        "cadence": active / f"{stem}.csv",
        "live_lock": active / "current_wizard_hyperliquid_live_lock.csv",
        "storage_efficiency": active
        / "current_wizard_hyperliquid_storage_efficiency.csv",
        "validation": active / f"{stem}_validation.csv",
        "manifest": active / f"{stem}_manifest.json",
        "summary_md": active / f"{stem}_summary.md",
        "snapshot_cadence": snapshot / "operating_cadence.csv",
        "snapshot_live_lock": snapshot / "live_lock.csv",
        "snapshot_storage_efficiency": snapshot / "storage_efficiency.csv",
        "snapshot_validation": snapshot / "validation.csv",
        "snapshot_manifest": snapshot / "manifest.json",
        "snapshot_summary_md": snapshot / "summary.md",
    }


def _summary_markdown(summary: dict[str, object]) -> str:
    return "\n".join(
        [
            "# Current Wizard To Hyperliquid Operating Cadence",
            "",
            "This is a research schedule and live lock. It does not execute the listed commands, transfer funds, approve agents, or submit orders.",
            "",
            f"- cadence: {summary['cadence_id']}",
            f"- chain validation: {summary['chain_validation_id']}",
            f"- OU Optimal overlay: {summary['ou_optimal_overlay_id']}",
            f"- stages: {summary['stages']}",
            f"- experiments accounted: {summary['experiment_authority_count']}",
            f"- daily research run ready: {summary['daily_research_run_ready']}",
            f"- daily research blocker: {summary['daily_research_run_blocker']}",
            "- OU Optimal source rows true / false / accounted: "
            f"{summary['ou_optimal_true_rows']} / "
            f"{summary['ou_optimal_false_rows']} / "
            f"{summary['ou_optimal_source_rows_accounted']}",
            f"- immutable upstream bytes referenced: {summary['referenced_upstream_bytes']}",
            f"- input bytes physically copied: {summary['locally_copied_input_bytes']}",
            f"- recursive copy bytes avoided: {summary['recursive_copy_bytes_avoided']}",
            f"- 1x research survivors: {summary['one_x_research_survivors']}",
            f"- trade eligibility: {summary['trade_eligibility_status']}",
            "- scheduled Testnet order authority: false",
            "- order submission performed: false",
            "- live trading authorized: false",
            "",
        ]
    )


def _key_configured(root: Path, key: str) -> bool:
    if _text(os.getenv(key)):
        return True
    for path in (root / ".env", root / ".env.local"):
        if not path.is_file():
            continue
        for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            if name.strip() == key and value.strip().strip('"\''):
                return True
    return False


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


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


def _text(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()
