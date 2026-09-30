"""Immutable, run-scoped control plane for The Wizard V2 pipeline."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd

from quant_platform.active_pipeline import CommandResult
from quant_platform.orchestration.corrective_runtime import (
    atomic_copy_file,
    promote_staged_directory,
    promote_staged_file,
)
from quant_platform.wizard_policy import load_wizard_discovery_policy
from quant_platform.runtime_types import strict_bool

ROOT = Path(__file__).resolve().parents[2]
V2_RUN_SCHEMA_VERSION = "the_wizard_v2_run.v1"
V2_SEAL_SCHEMA_VERSION = "the_wizard_v2_seal.v1"


@dataclass(frozen=True)
class V2ArtifactSpec:
    stage: str
    name: str
    source_path: str
    identity_required: bool = False
    policy_required: bool = False
    pair_coverage_required: bool = False
    required: bool = True


V2_ARTIFACT_SPECS = (
    V2ArtifactSpec("discovery", "wizard_sweep_candidates", "reports/active/wizard_sweep_candidates.csv"),
    V2ArtifactSpec(
        "wizard_capture",
        "wizard_pair_settings",
        "reports/active/crypto_wizards_pair_page_capture_settings.csv",
        pair_coverage_required=True,
    ),
    V2ArtifactSpec(
        "wizard_capture",
        "wizard_hyperliquid_queue",
        "reports/active/hyperliquid_wizard_hypothesis_queue.csv",
        identity_required=True,
        policy_required=True,
        pair_coverage_required=True,
    ),
    V2ArtifactSpec(
        "vendor_proof",
        "wizard_vendor_mode_proofs",
        "reports/active/hyperliquid_wizard_vendor_mode_proofs.csv",
        identity_required=True,
        policy_required=True,
        pair_coverage_required=True,
    ),
    V2ArtifactSpec(
        "hyperliquid_data",
        "hyperliquid_research_bundle",
        "reports/active/hyperliquid_research_bundle.csv",
        pair_coverage_required=True,
    ),
    V2ArtifactSpec(
        "hyperliquid_data",
        "hyperliquid_funding_coverage",
        "reports/active/hyperliquid_funding_coverage.csv",
        pair_coverage_required=True,
    ),
    V2ArtifactSpec(
        "costs",
        "hyperliquid_pair_cost_model",
        "reports/active/hyperliquid_pair_cost_model.csv",
        pair_coverage_required=True,
    ),
    V2ArtifactSpec(
        "local_replay",
        "wizard_mode_comparison",
        "reports/active/wizard_mode_comparison.csv",
        identity_required=True,
        pair_coverage_required=True,
    ),
    V2ArtifactSpec(
        "walkforward",
        "walkforward_mode_summary",
        "reports/orchestration/teacher_council/walkforward_mode_summary.csv",
        identity_required=True,
        pair_coverage_required=True,
    ),
    V2ArtifactSpec(
        "council",
        "council_decisions",
        "reports/orchestration/teacher_council/council_decisions.csv",
        identity_required=True,
        pair_coverage_required=True,
    ),
    V2ArtifactSpec(
        "council",
        "portfolio_critic",
        "reports/orchestration/teacher_council/portfolio_critic.csv",
        identity_required=True,
    ),
)


ARTIFACT_REGISTRY_COLUMNS = [
    "run_id",
    "candidate_set_id",
    "policy_hash",
    "stage",
    "artifact_name",
    "required",
    "source_path",
    "snapshot_path",
    "source_exists",
    "size_bytes",
    "sha256",
    "identity_required",
    "observed_candidate_set_ids",
    "identity_binding",
    "candidate_set_match",
    "policy_required",
    "observed_policy_hashes",
    "policy_hash_match",
    "pair_coverage_required",
    "observed_pairs",
    "pair_coverage_match",
    "semantic_status",
    "status",
    "blocker",
]


STAGE_ORDER = (
    "discovery",
    "wizard_capture",
    "vendor_proof",
    "hyperliquid_data",
    "costs",
    "local_replay",
    "walkforward",
    "council",
    "authority",
)


def build_v2_preflight_run(
    *,
    root: Path = ROOT,
    now: datetime | None = None,
    publish: bool = True,
) -> CommandResult:
    """Snapshot current evidence into one sealed run and evaluate it fail-closed."""

    captured_at = _as_utc(now or datetime.now(timezone.utc))
    policy = load_wizard_discovery_policy(root)
    candidate_set_id, candidate_pairs, identity_blockers = _source_candidate_set(root)
    source_hashes = {
        spec.name: _file_hash(root / spec.source_path)
        for spec in V2_ARTIFACT_SPECS
    }
    run_material = json.dumps(
        {
            "captured_at": captured_at.isoformat(),
            "candidate_set_id": candidate_set_id,
            "policy_hash": policy.policy_hash,
            "source_hashes": source_hashes,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    suffix = sha256(run_material.encode("utf-8")).hexdigest()[:12]
    run_id = f"v2run_{captured_at.strftime('%Y%m%dT%H%M%S%fZ')}_{suffix}"
    runs_dir = root / "runs"
    temporary = runs_dir / f".{run_id}.building"
    final = runs_dir / run_id
    if temporary.exists() or final.exists():
        raise FileExistsError(f"V2 run already exists: {run_id}")
    temporary.mkdir(parents=True, exist_ok=False)

    try:
        records = [
            _snapshot_artifact(
                root=root,
                run_dir=temporary,
                run_id=run_id,
                candidate_set_id=candidate_set_id,
                candidate_pairs=candidate_pairs,
                policy_hash=policy.policy_hash,
                spec=spec,
            )
            for spec in V2_ARTIFACT_SPECS
        ]
        registry = pd.DataFrame(records, columns=ARTIFACT_REGISTRY_COLUMNS)
        stage_status = _build_stage_status(
            registry,
            run_id=run_id,
            candidate_set_id=candidate_set_id,
            identity_blockers=identity_blockers,
        )
        blockers = list(identity_blockers)
        blockers.extend(
            stage_status.loc[stage_status["status"].eq("BLOCKED"), "blocker"]
            .dropna()
            .astype(str)
            .tolist()
        )
        blockers = _unique_blockers(blockers)
        research_ready = not blockers
        authority = {
            "schema_version": V2_RUN_SCHEMA_VERSION,
            "run_id": run_id,
            "candidate_set_id": candidate_set_id,
            "candidate_pairs": sorted(candidate_pairs),
            "policy_hash": policy.policy_hash,
            "status": "READY_FOR_RESEARCH" if research_ready else "BLOCKED",
            "research_ready": research_ready,
            "paper_ready": False,
            "live_ready": False,
            "execution_allowed": False,
            "blockers": blockers,
            "generated_at": captured_at.isoformat(),
        }
        manifest = {
            "schema_version": V2_RUN_SCHEMA_VERSION,
            "run_id": run_id,
            "candidate_set_id": candidate_set_id,
            "policy_schema_version": policy.schema_version,
            "policy_hash": policy.policy_hash,
            "created_at": captured_at.isoformat(),
            "status": authority["status"],
            "research_ready": research_ready,
            "paper_ready": False,
            "live_ready": False,
            "execution_allowed": False,
            "blockers": blockers,
            "artifact_count": len(registry),
            "stage_count": len(stage_status),
            "source_hashes": source_hashes,
            "artifact_registry_path": "artifact_registry.csv",
            "stage_status_path": "stage_status.csv",
            "authority_path": "authority/authority.json",
        }
        _atomic_csv(registry, temporary / "artifact_registry.csv")
        _atomic_csv(stage_status, temporary / "stage_status.csv")
        _atomic_json(authority, temporary / "authority" / "authority.json")
        _atomic_json(manifest, temporary / "manifest.json")
        _atomic_text(
            _run_markdown(manifest, stage_status, registry),
            temporary / "run_summary.md",
        )
        seal = _build_seal(temporary, run_id=run_id)
        _atomic_json(seal, temporary / "seal.json")
        runs_dir.mkdir(parents=True, exist_ok=True)
        promote_staged_directory(temporary, final)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    if publish:
        publish_v2_run_status(root=root, run_id=run_id)
    return CommandResult(
        paths={
            "run_dir": final,
            "manifest": final / "manifest.json",
            "artifact_registry": final / "artifact_registry.csv",
            "stage_status": final / "stage_status.csv",
            "authority": final / "authority" / "authority.json",
            "seal": final / "seal.json",
        },
        summary={
            "run_id": run_id,
            "candidate_set_id": candidate_set_id,
            "status": authority["status"],
            "research_ready": research_ready,
            "paper_ready": False,
            "live_ready": False,
            "execution_allowed": False,
            "blockers": blockers,
            "artifacts": len(registry),
            "stages": len(stage_status),
        },
    )


def validate_v2_run(*, root: Path = ROOT, run_id: str) -> dict[str, object]:
    """Verify a sealed run without modifying it or reading global reports."""

    run_dir = _run_directory(root, run_id)
    seal = _read_json(run_dir / "seal.json")
    manifest = _read_json(run_dir / "manifest.json")
    blockers: list[str] = []
    if seal.get("schema_version") != V2_SEAL_SCHEMA_VERSION:
        blockers.append("v2_seal_schema_invalid")
    if manifest.get("schema_version") != V2_RUN_SCHEMA_VERSION:
        blockers.append("v2_manifest_schema_invalid")
    if manifest.get("run_id") != run_id or seal.get("run_id") != run_id:
        blockers.append("v2_run_identity_mismatch")
    expected = seal.get("files") if isinstance(seal.get("files"), dict) else {}
    actual_files = {
        str(path.relative_to(run_dir))
        for path in run_dir.rglob("*")
        if path.is_file() and path.name != "seal.json"
    }
    if actual_files != set(expected):
        blockers.append("v2_sealed_file_set_changed")
    for relative_path, expected_hash in expected.items():
        path = run_dir / relative_path
        if not path.is_file() or _file_hash(path) != expected_hash:
            blockers.append(f"v2_sealed_file_hash_mismatch:{relative_path}")
    blockers.extend(_control_plane_blockers(run_dir, manifest, run_id=run_id))
    blockers = _unique_blockers(blockers)
    return {
        "run_id": run_id,
        "candidate_set_id": _text(manifest.get("candidate_set_id", "")),
        "sealed": not blockers,
        "status": "VALID" if not blockers else "INVALID",
        "blockers": blockers,
        "manifest_status": _text(manifest.get("status", "")),
        "execution_allowed": False,
        "run_dir": run_dir,
    }


def _control_plane_blockers(
    run_dir: Path,
    manifest: dict[str, object],
    *,
    run_id: str,
) -> list[str]:
    authority = _read_json(run_dir / "authority" / "authority.json")
    registry = _read_csv(run_dir / "artifact_registry.csv")
    stages = _read_csv(run_dir / "stage_status.csv")
    blockers: list[str] = []
    candidate_set_id = _text(manifest.get("candidate_set_id", ""))
    policy_hash = _text(manifest.get("policy_hash", ""))

    if not authority:
        blockers.append("v2_authority_record_missing")
    else:
        if authority.get("run_id") != run_id:
            blockers.append("v2_authority_run_identity_mismatch")
        if _text(authority.get("candidate_set_id", "")) != candidate_set_id:
            blockers.append("v2_authority_candidate_identity_mismatch")
        if _text(authority.get("policy_hash", "")) != policy_hash:
            blockers.append("v2_authority_policy_identity_mismatch")
        if authority.get("status") != manifest.get("status"):
            blockers.append("v2_authority_status_mismatch")
        if authority.get("research_ready") is not manifest.get("research_ready"):
            blockers.append("v2_authority_research_readiness_mismatch")

    ready_status = manifest.get("status") == "READY_FOR_RESEARCH"
    if strict_bool(manifest.get("research_ready")) != ready_status:
        blockers.append("v2_manifest_research_status_inconsistent")
    for field in ("paper_ready", "live_ready", "execution_allowed"):
        if _truthy(manifest.get(field, False)) or _truthy(authority.get(field, False)):
            blockers.append(f"v2_preflight_must_not_enable:{field}")

    if registry.empty:
        blockers.append("v2_artifact_registry_missing_or_empty")
    else:
        registry_identity_columns = {"run_id", "candidate_set_id", "policy_hash"}
        if not registry_identity_columns.issubset(registry.columns):
            blockers.append("v2_registry_identity_columns_missing")
        elif not registry["run_id"].astype(str).eq(run_id).all():
            blockers.append("v2_registry_run_identity_mismatch")
        elif not registry["candidate_set_id"].astype(str).eq(
            candidate_set_id
        ).all():
            blockers.append("v2_registry_candidate_identity_mismatch")
        elif not registry["policy_hash"].astype(str).eq(policy_hash).all():
            blockers.append("v2_registry_policy_identity_mismatch")
        if _safe_int(manifest.get("artifact_count"), default=-1) != len(registry):
            blockers.append("v2_manifest_artifact_count_mismatch")

    if stages.empty:
        blockers.append("v2_stage_status_missing_or_empty")
    else:
        stage_identity_columns = {"run_id", "candidate_set_id", "stage", "status"}
        if not stage_identity_columns.issubset(stages.columns):
            blockers.append("v2_stage_identity_columns_missing")
        elif not stages["run_id"].astype(str).eq(run_id).all():
            blockers.append("v2_stage_run_identity_mismatch")
        elif not stages["candidate_set_id"].astype(str).eq(
            candidate_set_id
        ).all():
            blockers.append("v2_stage_candidate_identity_mismatch")
        if _safe_int(manifest.get("stage_count"), default=-1) != len(stages):
            blockers.append("v2_manifest_stage_count_mismatch")
        authority_rows = (
            stages.loc[stages["stage"].astype(str).eq("authority")]
            if "stage" in stages
            else pd.DataFrame()
        )
        expected_authority_status = "PASS" if ready_status else "BLOCKED"
        if (
            len(authority_rows) != 1
            or _text(authority_rows.iloc[0].get("status", "")) != expected_authority_status
        ):
            blockers.append("v2_stage_authority_status_mismatch")
    return _unique_blockers(blockers)


def publish_v2_run_status(*, root: Path = ROOT, run_id: str) -> CommandResult:
    """Publish status only; actionable pointers require a valid ready run."""

    validation = validate_v2_run(root=root, run_id=run_id)
    run_dir = Path(validation["run_dir"])
    manifest = _read_json(run_dir / "manifest.json")
    authority = _read_json(run_dir / "authority" / "authority.json")
    sealed = bool(validation["sealed"])
    completed = bool(
        sealed
        and manifest.get("status") == "READY_FOR_RESEARCH"
        and authority.get("status") == "READY_FOR_RESEARCH"
        and authority.get("research_ready") is True
    )
    published_at = datetime.now(timezone.utc).isoformat()
    pointer = {
        "schema_version": V2_RUN_SCHEMA_VERSION,
        "run_id": run_id,
        "candidate_set_id": _text(manifest.get("candidate_set_id", "")),
        "status": _text(manifest.get("status", "INVALID")) if sealed else "INVALID",
        "sealed": sealed,
        "research_ready": bool(completed and strict_bool(authority.get("research_ready", False))),
        "paper_ready": False,
        "live_ready": False,
        "execution_allowed": False,
        "blockers": validation["blockers"] or manifest.get("blockers", []),
        "run_path": _relative(run_dir, root),
        "manifest_path": _relative(run_dir / "manifest.json", root),
        "published_at": published_at,
    }
    active = root / "reports" / "active"
    dashboard = root / "reports" / "dashboard"
    attempt_path = active / "v2_latest_attempt.json"
    status_path = dashboard / "v2_run_status.csv"
    _atomic_json(pointer, attempt_path)
    _atomic_csv(pd.DataFrame([_pointer_row(pointer)]), status_path)
    paths = {"latest_attempt": attempt_path, "dashboard_status": status_path}
    if completed:
        completed_path = active / "v2_latest_completed_run.json"
        dashboard_completed = dashboard / "v2_latest_completed_run.json"
        _atomic_json(pointer, completed_path)
        _atomic_json(pointer, dashboard_completed)
        paths.update(
            {"latest_completed": completed_path, "dashboard_completed": dashboard_completed}
        )
    return CommandResult(
        paths=paths,
        summary={
            "run_id": run_id,
            "status": pointer["status"],
            "sealed": sealed,
            "completed_published": completed,
            "execution_allowed": False,
            "blockers": pointer["blockers"],
        },
    )


def _snapshot_artifact(
    *,
    root: Path,
    run_dir: Path,
    run_id: str,
    candidate_set_id: str,
    candidate_pairs: set[str],
    policy_hash: str,
    spec: V2ArtifactSpec,
) -> dict[str, object]:
    source = root / spec.source_path
    snapshot = run_dir / spec.stage / source.name
    base = {
        "run_id": run_id,
        "candidate_set_id": candidate_set_id,
        "policy_hash": policy_hash,
        "stage": spec.stage,
        "artifact_name": spec.name,
        "required": spec.required,
        "source_path": spec.source_path,
        "snapshot_path": str(snapshot.relative_to(run_dir)),
        "source_exists": source.is_file(),
        "size_bytes": source.stat().st_size if source.is_file() else 0,
        "sha256": "",
        "identity_required": spec.identity_required,
        "observed_candidate_set_ids": "",
        "identity_binding": "embedded" if spec.identity_required else "manifest_envelope",
        "candidate_set_match": not spec.identity_required,
        "policy_required": spec.policy_required,
        "observed_policy_hashes": "",
        "policy_hash_match": not spec.policy_required,
        "pair_coverage_required": spec.pair_coverage_required,
        "observed_pairs": "",
        "pair_coverage_match": not spec.pair_coverage_required,
        "semantic_status": "NOT_CHECKED",
        "status": "",
        "blocker": "",
    }
    if not source.is_file():
        blocker = f"required_v2_artifact_missing:{spec.name}" if spec.required else ""
        return {
            **base,
            "status": "BLOCKED" if spec.required else "MISSING_OPTIONAL",
            "blocker": blocker,
        }
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    atomic_copy_file(source, snapshot, immutable=True)
    frame = _read_csv(snapshot) if snapshot.suffix.lower() == ".csv" else pd.DataFrame()
    observed_ids = _column_values(frame, "candidate_set_id")
    observed_policies = _column_values(frame, "discovery_policy_hash")
    observed_pairs = _pair_values(frame)
    candidate_match = (
        bool(candidate_set_id)
        and observed_ids == {candidate_set_id}
        if spec.identity_required
        else True
    )
    policy_match = observed_policies == {policy_hash} if spec.policy_required else True
    pair_coverage_match = observed_pairs == candidate_pairs if spec.pair_coverage_required else True
    semantic_blockers = _semantic_blockers(spec.name, frame)
    blockers: list[str] = []
    if spec.identity_required and not candidate_match:
        blockers.append(f"candidate_set_identity_mismatch:{spec.name}")
    if spec.policy_required and not policy_match:
        blockers.append(f"policy_hash_mismatch:{spec.name}")
    if spec.pair_coverage_required and not pair_coverage_match:
        blockers.append(f"candidate_pair_coverage_mismatch:{spec.name}")
    blockers.extend(semantic_blockers)
    blockers = _unique_blockers(blockers)
    return {
        **base,
        "sha256": _file_hash(snapshot),
        "observed_candidate_set_ids": ";".join(sorted(observed_ids)),
        "candidate_set_match": candidate_match,
        "observed_policy_hashes": ";".join(sorted(observed_policies)),
        "policy_hash_match": policy_match,
        "observed_pairs": ";".join(sorted(observed_pairs)),
        "pair_coverage_match": pair_coverage_match,
        "semantic_status": "PASS" if not semantic_blockers else "BLOCKED",
        "status": "PASS" if not blockers else "BLOCKED",
        "blocker": ";".join(blockers),
    }


def _semantic_blockers(name: str, frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return [f"v2_artifact_has_no_rows:{name}"]
    blockers: list[str] = []
    if name == "wizard_pair_settings":
        _require_all_truthy(frame, "capture_confirmed", blockers, "wizard_capture_not_confirmed")
        _require_all_truthy(frame, "backtest_settings_complete", blockers, "wizard_settings_incomplete")
    elif name == "wizard_hyperliquid_queue":
        _require_all_truthy(frame, "passes_wizard_discovery_gate", blockers, "wizard_discovery_gate_failed")
        _require_all_truthy(frame, "passes_research_spend_gate", blockers, "wizard_paid_proof_gate_failed")
        _require_all_truthy(frame, "wizard_source_fresh", blockers, "wizard_source_not_fresh")
    elif name == "wizard_vendor_mode_proofs":
        _require_all_equal(frame, "mode_proof_status", "completed", blockers, "vendor_proof_incomplete")
        _require_all_equal(
            frame,
            "proof_window_kind",
            "scanner_horizon_parity",
            blockers,
            "vendor_proof_horizon_mismatch",
        )
    elif name == "hyperliquid_research_bundle":
        _require_all_truthy(frame, "history_ready", blockers, "hyperliquid_history_not_ready")
    elif name == "hyperliquid_funding_coverage":
        _require_all_truthy(frame, "funding_ready", blockers, "funding_not_ready")
        coverage = pd.to_numeric(
            frame.get(
                "funding_coverage_pct",
                pd.Series(index=frame.index, dtype=float),
            ),
            errors="coerce",
        )
        if coverage.isna().any() or (coverage < 95.0).any():
            blockers.append("funding_coverage_below_95pct")
    elif name == "hyperliquid_pair_cost_model":
        _require_all_truthy(frame, "cost_model_ready", blockers, "cost_model_not_ready")
        _require_all_truthy(frame, "slippage_model_ready", blockers, "slippage_model_not_ready")
    elif name == "wizard_mode_comparison":
        _require_all_equal(
            frame,
            "sample_parity_status",
            "MATCHED_OBSERVATION_COUNT",
            blockers,
            "local_sample_horizon_mismatch",
        )
        if "comparison_validity" not in frame or not frame["comparison_validity"].astype(str).isin(
            {"VALID", "ACCEPTANCE_VALID"}
        ).all():
            blockers.append("local_comparison_not_acceptance_valid")
    elif name == "walkforward_mode_summary":
        if "selection_status" not in frame or not frame["selection_status"].astype(str).eq("PASS").all():
            blockers.append("walkforward_selection_not_passed")
    elif name == "council_decisions":
        _require_all_equal(frame, "status", "SHADOW_TEST", blockers, "teacher_council_not_clear")
        if "action" not in frame or frame["action"].astype(str).eq("abstain").any():
            blockers.append("teacher_council_abstained")
    elif name == "portfolio_critic":
        _require_all_equal(frame, "verdict", "pass", blockers, "portfolio_critic_not_clear")
    return _unique_blockers(blockers)


def _build_stage_status(
    registry: pd.DataFrame,
    *,
    run_id: str,
    candidate_set_id: str,
    identity_blockers: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for stage in STAGE_ORDER[:-1]:
        artifacts = registry.loc[registry["stage"].eq(stage)]
        blockers = artifacts.loc[artifacts["status"].eq("BLOCKED"), "blocker"].dropna().astype(str).tolist()
        if stage == "wizard_capture":
            blockers.extend(identity_blockers)
        blockers = _unique_blockers(blockers)
        rows.append(
            {
                "run_id": run_id,
                "candidate_set_id": candidate_set_id,
                "stage": stage,
                "artifacts": len(artifacts),
                "passed_artifacts": int(artifacts["status"].eq("PASS").sum()),
                "status": "PASS" if not blockers else "BLOCKED",
                "blocker": ";".join(blockers),
            }
        )
    upstream_blockers = [row["blocker"] for row in rows if row["status"] == "BLOCKED"]
    rows.append(
        {
            "run_id": run_id,
            "candidate_set_id": candidate_set_id,
            "stage": "authority",
            "artifacts": 0,
            "passed_artifacts": 0,
            "status": "PASS" if not upstream_blockers else "BLOCKED",
            "blocker": ";".join(_unique_blockers(upstream_blockers)),
        }
    )
    return pd.DataFrame(rows)


def _source_candidate_set(root: Path) -> tuple[str, set[str], list[str]]:
    queue = _read_csv(root / "reports" / "active" / "hyperliquid_wizard_hypothesis_queue.csv")
    values = _column_values(queue, "candidate_set_id")
    pairs = _pair_values(queue)
    blockers: list[str] = []
    if not pairs:
        blockers.append("v2_candidate_pairs_missing")
    if len(values) == 1:
        return next(iter(values)), pairs, blockers
    if not values:
        return "", pairs, blockers + ["v2_candidate_set_id_missing"]
    return "", pairs, blockers + ["v2_candidate_set_id_not_unique"]


def _build_seal(run_dir: Path, *, run_id: str) -> dict[str, object]:
    files = {
        str(path.relative_to(run_dir)): _file_hash(path)
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path.name != "seal.json"
    }
    return {
        "schema_version": V2_SEAL_SCHEMA_VERSION,
        "run_id": run_id,
        "sealed_at": datetime.now(timezone.utc).isoformat(),
        "files": files,
    }


def _run_markdown(manifest: dict[str, object], stages: pd.DataFrame, registry: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# The Wizard V2 Run",
            "",
            f"- Run ID: `{manifest['run_id']}`",
            f"- Candidate set: `{manifest['candidate_set_id']}`",
            f"- Policy hash: `{manifest['policy_hash']}`",
            f"- Status: `{manifest['status']}`",
            "- Execution allowed: `false`",
            "",
            "## Stage Status",
            "",
            stages.to_markdown(index=False),
            "",
            "## Artifact Registry",
            "",
            registry.to_markdown(index=False),
            "",
        ]
    )


def _pointer_row(pointer: dict[str, object]) -> dict[str, object]:
    blockers = pointer.get("blockers", [])
    return {
        "run_id": pointer.get("run_id", ""),
        "candidate_set_id": pointer.get("candidate_set_id", ""),
        "status": pointer.get("status", ""),
        "sealed": pointer.get("sealed", False),
        "research_ready": pointer.get("research_ready", False),
        "paper_ready": False,
        "live_ready": False,
        "execution_allowed": False,
        "blocker": ";".join(blockers) if isinstance(blockers, list) else str(blockers),
        "run_path": pointer.get("run_path", ""),
        "manifest_path": pointer.get("manifest_path", ""),
        "published_at": pointer.get("published_at", ""),
    }


def _require_all_truthy(frame: pd.DataFrame, column: str, blockers: list[str], blocker: str) -> None:
    if column not in frame or not frame[column].map(_truthy).all():
        blockers.append(blocker)


def _require_all_equal(
    frame: pd.DataFrame,
    column: str,
    expected: str,
    blockers: list[str],
    blocker: str,
) -> None:
    if column not in frame or not frame[column].astype(str).eq(expected).all():
        blockers.append(blocker)


def _column_values(frame: pd.DataFrame, column: str) -> set[str]:
    if frame.empty or column not in frame:
        return set()
    return {value for value in frame[column].dropna().map(_text) if value}


def _pair_values(frame: pd.DataFrame) -> set[str]:
    if frame.empty or "pair" not in frame:
        return set()
    return {
        normalized
        for normalized in frame["pair"].dropna().map(_normalize_pair_identity)
        if normalized
    }


def _normalize_pair_identity(value: object) -> str:
    text = _text(value).upper()
    return "".join(character for character in text if character.isalnum() or character == "/")


def _unique_blockers(values: list[str]) -> list[str]:
    output: list[str] = []
    for value in values:
        for blocker in str(value).split(";"):
            clean = blocker.strip()
            if clean and clean not in output:
                output.append(clean)
    return output


def _run_directory(root: Path, run_id: str) -> Path:
    if not run_id or Path(run_id).name != run_id or not run_id.startswith("v2run_"):
        raise ValueError("invalid V2 run ID")
    path = (root / "runs" / run_id).resolve()
    runs = (root / "runs").resolve()
    if runs not in path.parents or not path.is_dir():
        raise FileNotFoundError(f"V2 run not found: {run_id}")
    return path


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError):
        return pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _file_hash(path: Path) -> str:
    if not path.is_file():
        return ""
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    promote_staged_file(temporary, path)


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    promote_staged_file(temporary, path)


def _atomic_text(content: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    promote_staged_file(temporary, path)


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return _text(value).lower() in {"1", "true", "yes", "y"}


def _safe_int(value: object, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _text(value: object) -> str:
    if value is None or (not isinstance(value, (dict, list)) and pd.isna(value)):
        return ""
    return str(value).strip()


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)
