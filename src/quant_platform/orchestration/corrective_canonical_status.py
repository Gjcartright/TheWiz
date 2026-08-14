"""Canonical, freshness-validated status for the corrective program."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd

from quant_platform.active_pipeline import CommandResult

ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "thewiz.corrective_canonical_status.v1"
COMPLETION_SCHEMA_VERSION = "thewiz.corrective_program_completion.v1"
SCHEDULER_SCHEMA_VERSION = "thewiz.corrective_wizard_proof_scheduler.v1"
SCHEDULER_LEGACY_POINTER = "corrective_wizard_proof_scheduler_status.json"
SCHEDULER_EXECUTION_POINTER = "corrective_wizard_proof_scheduler_execution_status.json"
SCHEDULER_OBSERVATION_POINTER = "corrective_wizard_proof_scheduler_observation_status.json"
STAGE4_RECEIPT_PATTERN = re.compile(r"(?:^|;)stage4_handoff_receipt_id=([^;]+)")
STAGE4_CLOSURE_PATTERN = re.compile(
    r"(?:^|;)stage4_handoff_source_closure_sha256=([0-9a-f]{64})(?:;|$)"
)
STAGE4_REBIND_SCHEMA_VERSION = "thewiz.canonical_stage4_rebinding.v1"


def rebind_checkpoint_stage4_evidence(
    *,
    root: Path = ROOT,
    now: datetime | None = None,
    stage4_validator: Callable[..., dict[str, Any]] | None = None,
) -> CommandResult:
    """Atomically bind the checkpoint to the current validated Stage 4 receipt."""

    generated_at = _as_utc(now)
    active = root / "reports" / "active"
    active.mkdir(parents=True, exist_ok=True)
    checkpoint_path = active / "seven_stage_goal_checkpoint.csv"
    stage4_path = active / "stage4_handoff_readiness.json"
    stage4 = _read_json(stage4_path)
    checkpoint = _read_csv(checkpoint_path)
    before_sha256 = _file_hash(checkpoint_path)
    blockers: list[str] = []

    if stage4_validator is None:
        from quant_platform.orchestration.corrective_stage4_handoff_readiness import (
            validate_stage4_handoff_readiness_receipt,
        )

        stage4_validator = validate_stage4_handoff_readiness_receipt
    try:
        validation = stage4_validator(root=root, receipt=stage4)
    except Exception as exc:  # noqa: BLE001 - rebinding must fail closed
        validation = {
            "status": "BLOCKED",
            "blockers": [f"stage4_validator_failed:{type(exc).__name__}:{exc}"],
        }
    if validation.get("status") != "PASS":
        blockers.append("canonical_stage4_source_receipt_invalid")

    authority_fields = (
        "candidate_promotion_authority",
        "order_submission_included",
        "testnet_order_authority",
        "live_trading_authorized",
    )
    if any(stage4.get(field) is not False for field in authority_fields):
        blockers.append("canonical_stage4_source_authority_not_strictly_zero")

    required_checkpoint_columns = {"stage", "evidence_progress"}
    stage4_rows = (
        checkpoint.index[checkpoint["stage"].astype(str).eq("4")].tolist()
        if not checkpoint.empty and required_checkpoint_columns.issubset(checkpoint.columns)
        else []
    )
    if len(stage4_rows) != 1:
        blockers.append("canonical_stage4_checkpoint_row_invalid")

    current_receipt_id = str(stage4.get("receipt_id", "")).strip()
    current_closure = str(stage4.get("source_closure_sha256", "")).strip().lower()
    current_checked_at = str(stage4.get("checked_at_utc", "")).strip()
    if not current_receipt_id:
        blockers.append("canonical_stage4_source_receipt_id_missing")
    if not re.fullmatch(r"[0-9a-f]{64}", current_closure):
        blockers.append("canonical_stage4_source_closure_invalid")
    if _parse_optional_utc(current_checked_at) is None:
        blockers.append("canonical_stage4_source_checked_at_invalid")

    previous_evidence = ""
    rebound_evidence = ""
    semantic_preservation_valid = False
    if not blockers:
        row_index = stage4_rows[0]
        previous = checkpoint.copy(deep=True)
        previous_evidence = str(checkpoint.at[row_index, "evidence_progress"])
        rebound_evidence = previous_evidence
        for key, value in (
            ("stage4_handoff_receipt_id", current_receipt_id),
            ("stage4_handoff_source_closure_sha256", current_closure),
            ("stage4_handoff_checked_at", current_checked_at),
        ):
            rebound_evidence = _set_evidence_token(rebound_evidence, key=key, value=value)
        checkpoint.at[row_index, "evidence_progress"] = rebound_evidence

        comparison = checkpoint.copy(deep=True)
        comparison.at[row_index, "evidence_progress"] = previous_evidence
        semantic_preservation_valid = comparison.equals(previous)
        if not semantic_preservation_valid:
            blockers.append("canonical_stage4_rebinding_changed_unrelated_checkpoint_fields")
        else:
            _atomic_text(checkpoint_path, checkpoint.to_csv(index=False))

    after_sha256 = _file_hash(checkpoint_path)
    payload: dict[str, Any] = {
        "schema_version": STAGE4_REBIND_SCHEMA_VERSION,
        "generated_at_utc": generated_at.isoformat(),
        "status": "PASS_STAGE4_CHECKPOINT_REBOUND"
        if not blockers
        else "BLOCKED_STAGE4_CHECKPOINT_REBIND",
        "blockers": list(dict.fromkeys(blockers)),
        "checkpoint_path": _relative(checkpoint_path, root),
        "checkpoint_sha256_before": before_sha256,
        "checkpoint_sha256_after": after_sha256,
        "stage4_path": _relative(stage4_path, root),
        "stage4_sha256": _file_hash(stage4_path),
        "stage4_receipt_id": current_receipt_id,
        "stage4_source_closure_sha256": current_closure,
        "stage4_checked_at_utc": current_checked_at,
        "stage4_validation_status": str(validation.get("status", "BLOCKED")),
        "stage4_validation_blockers": list(validation.get("blockers", [])),
        "previous_evidence_progress": previous_evidence,
        "rebound_evidence_progress": rebound_evidence,
        "only_stage4_evidence_progress_changed": semantic_preservation_valid,
        "research_only": True,
        "candidate_promotion_authority": False,
        "order_submission_included": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    payload["receipt_id"] = (
        "stage4rebind_" + sha256(_canonical_json(payload).encode("utf-8")).hexdigest()[:20]
    )
    immutable_path = (
        root / "data" / "research" / "canonical_stage4_rebindings" / f"{payload['receipt_id']}.json"
    )
    status_path = active / "canonical_stage4_rebinding.json"
    _write_or_validate_immutable_json(payload, immutable_path)
    active_payload = {
        **payload,
        "immutable_receipt_path": _relative(immutable_path, root),
        "immutable_receipt_sha256": _file_hash(immutable_path),
    }
    _atomic_json(active_payload, status_path)
    return CommandResult(
        paths={
            "status": status_path,
            "checkpoint": checkpoint_path,
            "immutable_receipt": immutable_path,
        },
        summary=active_payload,
    )


def build_canonical_program_status(
    *, root: Path = ROOT, now: datetime | None = None
) -> CommandResult:
    """Publish one current snapshot without inheriting execution authority."""

    generated_at = _as_utc(now)
    active = root / "reports" / "active"
    active.mkdir(parents=True, exist_ok=True)
    role_paths = materialize_scheduler_status_roles(root=root)

    completion_path = active / "corrective_plan_completion.json"
    checkpoint_path = active / "seven_stage_goal_checkpoint.csv"
    stage4_path = active / "stage4_handoff_readiness.json"
    launcher_path = active / "corrective_wizard_proof_launcher_status.json"
    completion = _read_json(completion_path)
    checkpoint = _read_csv(checkpoint_path)
    stage4 = _read_json(stage4_path)
    launcher = _read_json(launcher_path)
    execution = _read_json(role_paths["execution"])
    observation = _read_json(role_paths["observation"])

    blockers: list[str] = []
    if completion.get("schema_version") != COMPLETION_SCHEMA_VERSION:
        blockers.append("canonical_completion_schema_invalid")
    expected_stages = {str(value) for value in range(1, 8)}
    if (
        checkpoint.empty
        or not {"stage", "status", "evidence_progress"}.issubset(checkpoint.columns)
        or set(checkpoint["stage"].astype(str)) != expected_stages
        or checkpoint["stage"].astype(str).duplicated().any()
    ):
        blockers.append("canonical_seven_stage_checkpoint_invalid")

    stage4_rows = (
        checkpoint.loc[checkpoint["stage"].astype(str).eq("4")]
        if not checkpoint.empty and "stage" in checkpoint
        else pd.DataFrame()
    )
    checkpoint_stage4_receipt_id = ""
    checkpoint_stage4_closure = ""
    if len(stage4_rows) == 1:
        evidence = str(stage4_rows.iloc[0].get("evidence_progress", ""))
        match = STAGE4_RECEIPT_PATTERN.search(evidence)
        checkpoint_stage4_receipt_id = match.group(1).strip() if match else ""
        closure_match = STAGE4_CLOSURE_PATTERN.search(evidence)
        checkpoint_stage4_closure = closure_match.group(1) if closure_match else ""
    current_stage4_receipt_id = str(stage4.get("receipt_id", "")).strip()
    current_stage4_closure = str(stage4.get("source_closure_sha256", "")).strip().lower()
    stage4_receipt_binding_match = bool(
        checkpoint_stage4_receipt_id
        and current_stage4_receipt_id
        and checkpoint_stage4_receipt_id == current_stage4_receipt_id
    )
    stage4_closure_binding_match = bool(
        checkpoint_stage4_closure
        and current_stage4_closure
        and checkpoint_stage4_closure == current_stage4_closure
    )
    stage4_binding_match = stage4_receipt_binding_match or stage4_closure_binding_match
    if not stage4_binding_match:
        blockers.append("canonical_checkpoint_stage4_receipt_stale_or_missing")

    completion_at = _parse_optional_utc(completion.get("generated_at_utc"))
    stage4_at = _parse_optional_utc(stage4.get("checked_at_utc"))
    completion_precedes_stage4 = bool(
        completion_at is not None and stage4_at is not None and completion_at < stage4_at
    )
    if completion_precedes_stage4 and not stage4_binding_match:
        blockers.append("canonical_completion_precedes_current_stage4_handoff")

    execution_role_valid = bool(
        execution.get("schema_version") == SCHEDULER_SCHEMA_VERSION
        and execution.get("execution_requested") is True
        and str(execution.get("receipt_id", "")).strip()
    )
    observation_role_valid = bool(
        observation.get("schema_version") == SCHEDULER_SCHEMA_VERSION
        and observation.get("execution_requested") is False
        and str(observation.get("receipt_id", "")).strip()
    )
    if not execution_role_valid:
        blockers.append("canonical_scheduler_execution_pointer_invalid")
    if not observation_role_valid:
        blockers.append("canonical_scheduler_observation_pointer_invalid")

    authority_fields = ("testnet_order_authority", "live_trading_authorized")
    authority_sources = {
        "completion": completion,
        "stage4_handoff": stage4,
        "wizard_launcher": launcher,
        "scheduler_execution": execution,
        "scheduler_observation": observation,
    }
    authority_violations = [
        f"{name}:{field}"
        for name, payload in authority_sources.items()
        for field in authority_fields
        if payload.get(field) is not False
    ]
    if authority_violations:
        blockers.extend(f"canonical_authority_violation:{value}" for value in authority_violations)

    existing_count, planned_count, missing_required_count = _evidence_path_counts(
        checkpoint=checkpoint,
        root=root,
    )
    if missing_required_count:
        blockers.append(f"canonical_required_evidence_paths_missing:{missing_required_count}")

    stage_statuses = (
        {
            str(row["stage"]): str(row["status"])
            for _, row in checkpoint.sort_values(
                "stage", key=lambda values: pd.to_numeric(values, errors="coerce")
            ).iterrows()
        }
        if not checkpoint.empty and {"stage", "status"}.issubset(checkpoint.columns)
        else {}
    )
    source_paths = {
        "completion": completion_path,
        "seven_stage_checkpoint": checkpoint_path,
        "stage4_handoff": stage4_path,
        "wizard_launcher": launcher_path,
        "scheduler_execution": role_paths["execution"],
        "scheduler_observation": role_paths["observation"],
    }
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": generated_at.isoformat(),
        "status": "PASS_CANONICAL_CURRENT" if not blockers else "BLOCKED_CANONICAL_STALE",
        "blockers": list(dict.fromkeys(blockers)),
        "decision": "CONTINUE_RESEARCH_ONLY",
        "stage_statuses": stage_statuses,
        "completion_generated_at_utc": _iso_or_empty(completion_at),
        "stage4_checked_at_utc": _iso_or_empty(stage4_at),
        "completion_precedes_stage4_handoff": completion_precedes_stage4,
        "checkpoint_stage4_receipt_id": checkpoint_stage4_receipt_id,
        "current_stage4_receipt_id": current_stage4_receipt_id,
        "checkpoint_stage4_source_closure_sha256": checkpoint_stage4_closure,
        "current_stage4_source_closure_sha256": current_stage4_closure,
        "stage4_exact_receipt_binding_match": stage4_receipt_binding_match,
        "stage4_closure_binding_match": stage4_closure_binding_match,
        "stage4_receipt_binding_match": stage4_binding_match,
        "scheduler_execution_receipt_id": str(execution.get("receipt_id", "")),
        "scheduler_observation_receipt_id": str(observation.get("receipt_id", "")),
        "scheduler_execution_pointer_valid": execution_role_valid,
        "scheduler_observation_pointer_valid": observation_role_valid,
        "existing_evidence_paths": existing_count,
        "planned_output_paths": planned_count,
        "missing_required_paths": missing_required_count,
        "source_artifacts": {
            name: {"path": _relative(path, root), "sha256": _file_hash(path)}
            for name, path in source_paths.items()
        },
        "research_only": True,
        "candidate_promotion_authority": False,
        "order_submission_included": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    payload["source_closure_sha256"] = sha256(
        _canonical_json(payload["source_artifacts"]).encode("utf-8")
    ).hexdigest()
    payload["receipt_id"] = (
        "canonicalstatus_" + sha256(_canonical_json(payload).encode("utf-8")).hexdigest()[:20]
    )

    status_path = active / "canonical_program_status.json"
    markdown_path = active / "canonical_program_status.md"
    immutable_path = (
        root / "data" / "research" / "canonical_program_status" / f"{payload['receipt_id']}.json"
    )
    _write_or_validate_immutable_json(payload, immutable_path)
    active_payload = {
        **payload,
        "immutable_receipt_path": _relative(immutable_path, root),
        "immutable_receipt_sha256": _file_hash(immutable_path),
    }
    _atomic_json(active_payload, status_path)
    _atomic_text(markdown_path, _markdown(active_payload))
    return CommandResult(
        paths={
            "status": status_path,
            "summary": markdown_path,
            "immutable_receipt": immutable_path,
            "scheduler_execution_pointer": role_paths["execution"],
            "scheduler_observation_pointer": role_paths["observation"],
        },
        summary=active_payload,
    )


def materialize_scheduler_status_roles(*, root: Path = ROOT) -> dict[str, Path]:
    """Publish role-specific pointers from durable scheduler receipts."""

    active = root / "reports" / "active"
    receipts = active / "wizard_proof_scheduler_receipts"
    execution_path = active / SCHEDULER_EXECUTION_POINTER
    observation_path = active / SCHEDULER_OBSERVATION_POINTER
    candidates: dict[bool, list[tuple[datetime, Path, dict[str, Any]]]] = {
        True: [],
        False: [],
    }
    if receipts.is_dir():
        for path in receipts.glob("*.json"):
            payload = _read_json(path)
            role = payload.get("execution_requested")
            started_at = _parse_optional_utc(payload.get("started_at_utc"))
            if (
                role not in candidates
                or started_at is None
                or payload.get("schema_version") != SCHEDULER_SCHEMA_VERSION
                or not str(payload.get("receipt_id", "")).strip()
            ):
                continue
            candidates[bool(role)].append((started_at, path, payload))

    legacy = _read_json(active / SCHEDULER_LEGACY_POINTER)
    legacy_role = legacy.get("execution_requested")
    legacy_started = _parse_optional_utc(legacy.get("started_at_utc"))
    if (
        legacy_role in candidates
        and legacy_started is not None
        and legacy.get("schema_version") == SCHEDULER_SCHEMA_VERSION
        and str(legacy.get("receipt_id", "")).strip()
    ):
        candidates[bool(legacy_role)].append(
            (legacy_started, active / SCHEDULER_LEGACY_POINTER, legacy)
        )

    for role, pointer in ((True, execution_path), (False, observation_path)):
        if not candidates[role]:
            continue
        _, source_path, payload = max(candidates[role], key=lambda value: value[0])
        pointer_payload = {
            **payload,
            "pointer_role": "execution" if role else "observation",
            "pointer_source_path": _relative(source_path, root),
        }
        _atomic_json(pointer_payload, pointer)
    return {"execution": execution_path, "observation": observation_path}


def publish_scheduler_status_pointers(
    *, root: Path, payload: dict[str, Any], receipt_path: Path
) -> dict[str, Path]:
    """Publish latest-any plus the one pointer matching this invocation role."""

    active = root / "reports" / "active"
    legacy_path = active / SCHEDULER_LEGACY_POINTER
    execution_path = active / SCHEDULER_EXECUTION_POINTER
    observation_path = active / SCHEDULER_OBSERVATION_POINTER
    pointer_payload = {**payload, "receipt_path": _relative(receipt_path, root)}
    _atomic_json(pointer_payload, legacy_path)
    role_is_execution = payload.get("execution_requested") is True
    role_path = execution_path if role_is_execution else observation_path
    _atomic_json(
        {
            **pointer_payload,
            "pointer_role": "execution" if role_is_execution else "observation",
            "pointer_source_path": _relative(receipt_path, root),
        },
        role_path,
    )
    return {
        "latest": legacy_path,
        "execution": execution_path,
        "observation": observation_path,
    }


def _evidence_path_counts(*, checkpoint: pd.DataFrame, root: Path) -> tuple[int, int, int]:
    if checkpoint.empty:
        return 0, 0, 0
    if {
        "existing_evidence_paths",
        "planned_output_paths",
        "missing_required_paths",
    }.issubset(checkpoint.columns):
        return (
            _path_cell_count(checkpoint["existing_evidence_paths"]),
            _path_cell_count(checkpoint["planned_output_paths"]),
            _path_cell_count(checkpoint["missing_required_paths"]),
        )
    existing = 0
    planned = 0
    missing_required = 0
    for _, row in checkpoint.iterrows():
        passed = str(row.get("status", "")).upper() == "PASS"
        for value in str(row.get("evidence_path", "")).split(";"):
            relative = value.strip()
            if not relative:
                continue
            if (root / relative).exists():
                existing += 1
            elif passed:
                missing_required += 1
            else:
                planned += 1
    return existing, planned, missing_required


def _path_cell_count(series: pd.Series) -> int:
    return sum(
        1 for cell in series.fillna("").astype(str) for value in cell.split(";") if value.strip()
    )


def _markdown(payload: dict[str, Any]) -> str:
    stages = " | ".join(
        f"{stage}:{status}" for stage, status in payload.get("stage_statuses", {}).items()
    )
    blockers = payload.get("blockers", [])
    blocker_lines = "\n".join(f"- `{value}`" for value in blockers) or "- none"
    return (
        "# Canonical Program Status\n\n"
        f"- Status: `{payload['status']}`\n"
        f"- Generated: `{payload['generated_at_utc']}`\n"
        f"- Decision: `{payload['decision']}`\n"
        f"- Stages: `{stages}`\n"
        f"- Stage 4 binding current: `{payload['stage4_receipt_binding_match']}`\n"
        f"- Existing evidence paths: `{payload['existing_evidence_paths']}`\n"
        f"- Planned output paths: `{payload['planned_output_paths']}`\n"
        f"- Missing required paths: `{payload['missing_required_paths']}`\n\n"
        "## Blockers\n\n"
        f"{blocker_lines}\n\n"
        "This snapshot grants no Testnet or live order authority.\n"
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, keep_default_na=False)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError):
        return pd.DataFrame()


def _set_evidence_token(evidence: str, *, key: str, value: str) -> str:
    token = f"{key}={value}"
    pattern = re.compile(rf"(?:(?<=;)|^){re.escape(key)}=[^;]*(?=;|$)")
    if pattern.search(evidence):
        return pattern.sub(token, evidence, count=1)
    separator = "" if not evidence or evidence.endswith(";") else ";"
    return f"{evidence}{separator}{token};"


def _as_utc(value: datetime | None) -> datetime:
    timestamp = value or datetime.now(UTC)
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=UTC)
    return timestamp.astimezone(UTC)


def _parse_optional_utc(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return _as_utc(parsed)


def _iso_or_empty(value: datetime | None) -> str:
    return value.isoformat() if value is not None else ""


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _file_hash(path: Path) -> str:
    try:
        return sha256(path.read_bytes()).hexdigest() if path.is_file() else ""
    except OSError:
        return ""


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except (OSError, ValueError):
        return str(path)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_or_validate_immutable_json(payload: dict[str, Any], path: Path) -> None:
    expected = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.is_file():
        if path.read_text(encoding="utf-8") != expected:
            raise ValueError(f"immutable canonical status mismatch: {path}")
        return
    _atomic_text(path, expected)
