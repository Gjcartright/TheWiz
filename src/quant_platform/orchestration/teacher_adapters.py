"""Fail-closed adapters from Math V2 replay evidence to council events."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import ValidationError

from quant_platform.orchestration.contracts import CandidateIdentity
from quant_platform.orchestration.teacher_contracts import (
    CouncilContext,
    CriticAssessment,
    CriticType,
    CriticVerdict,
    EvidenceAuthority,
    EXACT_MODES,
    MATH_V2,
    REQUIRED_CRITICS,
    TeacherAction,
    TeacherProposal,
    normalize_exact_mode,
)


ROOT = Path(__file__).resolve().parents[3]

TEACHER_INPUT_COLUMNS = (
    "run_id",
    "candidate_set_id",
    "setup_identity",
    "pair",
    "venue",
    "timeframe",
    "lookback",
    "source_snapshot_id",
    "source_timestamp",
    "exact_mode",
    "proposed_action",
    "confidence",
    "uncertainty",
    "expected_net_return",
    "lower_bound_net_return",
    "expected_holding_bars",
    "entry_style",
    "exit_style",
    "invalidation_condition",
    "required_regime",
    "source_system",
    "formula_version",
    "math_version",
    "mode_fidelity_status",
    "point_in_time_status",
    "history_hash",
    "settings_version",
    "cost_model_version",
    "train_start",
    "train_end",
    "test_start",
    "test_end",
    "blockers",
    "evidence_path",
)

CRITIC_INPUT_COLUMNS = (
    "run_id",
    "candidate_set_id",
    "setup_identity",
    "pair",
    "venue",
    "timeframe",
    "lookback",
    "source_snapshot_id",
    "source_timestamp",
    "critic_type",
    "verdict",
    "score",
    "reason",
    "authority",
    "point_in_time_status",
    "blocker_codes",
    "evidence_path",
)


def build_teacher_evidence_adapters(*, root: Path = ROOT) -> dict[str, object]:
    """Validate complete contexts and emit council JSONL streams."""

    active_dir = root / "reports" / "active"
    output_dir = root / "reports" / "orchestration" / "teacher_council"
    output_dir.mkdir(parents=True, exist_ok=True)
    teacher_input_path = active_dir / "math_v2_teacher_inputs.csv"
    critic_input_path = active_dir / "math_v2_critic_inputs.csv"
    marker_path = active_dir / "math_v2_acceptance.json"
    proposal_path = output_dir / "teacher_proposals.jsonl"
    critic_path = output_dir / "critic_assessments.jsonl"
    readiness_path = output_dir / "teacher_adapter_readiness.csv"
    markdown_path = output_dir / "teacher_adapter_readiness.md"
    teacher_schema_path = output_dir / "teacher_input_schema.csv"
    critic_schema_path = output_dir / "critic_input_schema.csv"

    _write_schema(teacher_schema_path, TEACHER_INPUT_COLUMNS)
    _write_schema(critic_schema_path, CRITIC_INPUT_COLUMNS)
    teacher_frame = _read_csv(teacher_input_path)
    critic_frame = _read_csv(critic_input_path)
    readiness: list[dict[str, object]] = []

    marker_pass = _math_marker_passes(marker_path)
    _check(
        readiness,
        "core_math_v2_accepted",
        marker_pass,
        marker_path,
        "machine-generated Math V2 marker passes",
        "math_v2_marker_missing_or_invalid",
    )
    teacher_missing = sorted(set(TEACHER_INPUT_COLUMNS) - set(teacher_frame.columns))
    critic_missing = sorted(set(CRITIC_INPUT_COLUMNS) - set(critic_frame.columns))
    _check(
        readiness,
        "teacher_input_schema",
        bool(not teacher_frame.empty and not teacher_missing),
        f"rows={len(teacher_frame)};missing={';'.join(teacher_missing)}",
        "nonempty canonical teacher inputs",
        "teacher_inputs_missing_or_invalid",
    )
    _check(
        readiness,
        "critic_input_schema",
        bool(not critic_frame.empty and not critic_missing),
        f"rows={len(critic_frame)};missing={';'.join(critic_missing)}",
        "nonempty canonical critic inputs",
        "critic_inputs_missing_or_invalid",
    )

    proposals: list[TeacherProposal] = []
    assessments: list[CriticAssessment] = []
    validation_errors: list[str] = []
    if marker_pass and not teacher_missing and not critic_missing and not teacher_frame.empty and not critic_frame.empty:
        proposals, assessments, validation_errors = _complete_context_events(teacher_frame, critic_frame)
    _check(
        readiness,
        "complete_event_contexts",
        bool(proposals and assessments and not validation_errors),
        f"proposals={len(proposals)};critics={len(assessments)};errors={len(validation_errors)}",
        "7 teachers and 6 critics for every emitted context",
        "no_complete_valid_math_v2_context",
    )
    _check(
        readiness,
        "adapter_validation_errors",
        not validation_errors,
        ";".join(validation_errors),
        "no validation errors",
        "adapter_contract_validation_failed",
    )

    # Always replace streams. Keeping an old stream after current evidence fails
    # validation would let stale events survive a supposedly fail-closed run.
    _write_jsonl(proposal_path, proposals)
    _write_jsonl(critic_path, assessments)
    readiness_frame = pd.DataFrame(readiness)
    readiness_frame.to_csv(readiness_path, index=False)
    status = "READY_FOR_COUNCIL" if readiness_frame["status"].eq("PASS").all() else "BLOCKED"
    markdown_path.write_text(
        "\n".join(
            [
                "# Teacher Evidence Adapters",
                "",
                f"- Status: **{status}**",
                "- Authority: local Hyperliquid, point-in-time, Math V2 only",
                "- Incomplete contexts emit no council events.",
                "- Wizard discovery and legacy reports are never converted into local votes.",
                "",
                readiness_frame.to_markdown(index=False),
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "status": status,
        "proposal_count": len(proposals),
        "assessment_count": len(assessments),
        "validation_errors": len(validation_errors),
        "proposals": proposal_path,
        "critics": critic_path,
        "readiness": readiness_path,
        "markdown": markdown_path,
        "teacher_schema": teacher_schema_path,
        "critic_schema": critic_schema_path,
    }


def _complete_context_events(
    teacher_frame: pd.DataFrame,
    critic_frame: pd.DataFrame,
) -> tuple[list[TeacherProposal], list[CriticAssessment], list[str]]:
    proposals: list[TeacherProposal] = []
    assessments: list[CriticAssessment] = []
    errors: list[str] = []
    context_columns = [
        "run_id",
        "candidate_set_id",
        "setup_identity",
        "pair",
        "venue",
        "timeframe",
        "lookback",
        "source_snapshot_id",
    ]
    teacher_groups = {_group_key(key): group for key, group in teacher_frame.groupby(context_columns, dropna=False)}
    critic_groups = {_group_key(key): group for key, group in critic_frame.groupby(context_columns, dropna=False)}
    for key in sorted(set(teacher_groups) | set(critic_groups)):
        teacher_rows = teacher_groups.get(key, pd.DataFrame())
        critic_rows = critic_groups.get(key, pd.DataFrame())
        observed_modes = {_safe_mode(value) for value in teacher_rows.get("exact_mode", pd.Series(dtype=object))}
        observed_critics = {
            str(value).strip().lower() for value in critic_rows.get("critic_type", pd.Series(dtype=object))
        }
        required_modes = {mode.value for mode in EXACT_MODES}
        required_critics = {critic.value for critic in REQUIRED_CRITICS}
        if len(teacher_rows) != 7 or observed_modes != required_modes:
            errors.append(f"{key}:incomplete_or_duplicate_teacher_modes")
            continue
        if len(critic_rows) != 6 or observed_critics != required_critics:
            errors.append(f"{key}:incomplete_or_duplicate_critics")
            continue
        try:
            context = _context(teacher_rows.iloc[0])
            context_proposals = [_proposal(row, context) for _, row in teacher_rows.iterrows()]
            context_assessments = [_assessment(row, context) for _, row in critic_rows.iterrows()]
        except (ValueError, ValidationError) as exc:
            errors.append(f"{key}:{str(exc).replace(chr(10), ' | ')}")
            continue
        proposals.extend(context_proposals)
        assessments.extend(context_assessments)
    return proposals, assessments, errors


def _context(row: pd.Series) -> CouncilContext:
    return CouncilContext(
        pair=str(row["pair"]),
        venue=str(row["venue"]),
        timeframe=str(row["timeframe"]),
        lookback=int(row["lookback"]),
        source_snapshot_id=str(row["source_snapshot_id"]),
        source_timestamp=_timestamp(row["source_timestamp"]),
        run_id=str(row["run_id"]),
        candidate_set_id=str(row["candidate_set_id"]),
        setup_identity=str(row["setup_identity"]),
    )


def _proposal(row: pd.Series, context: CouncilContext) -> TeacherProposal:
    mode = normalize_exact_mode(str(row["exact_mode"]))
    if str(row["venue"]).strip().lower() != "hyperliquid":
        raise ValueError("teacher acceptance venue must be hyperliquid")
    if str(row["math_version"]).strip() != MATH_V2:
        raise ValueError("teacher math_version must be math-v2")
    if str(row["point_in_time_status"]).strip().lower() != "confirmed":
        raise ValueError("teacher input must be point-in-time confirmed")
    fidelity = str(row["mode_fidelity_status"]).strip().lower()
    if fidelity not in {"local_validated_estimator", "vendor_exact"}:
        raise ValueError("teacher mode fidelity must be local_validated_estimator or vendor_exact")
    for field in (
        "history_hash",
        "settings_version",
        "cost_model_version",
        "train_start",
        "train_end",
        "test_start",
        "test_end",
    ):
        if not str(row[field]).strip():
            raise ValueError(f"teacher lineage field cannot be blank:{field}")
    if "wizard" in str(row["source_system"]).lower():
        raise ValueError("Wizard evidence cannot be adapted into a local teacher vote")
    candidate = CandidateIdentity(
        pair=context.pair,
        venue=context.venue,
        strategy_family=mode.value,
        timeframe=context.timeframe,
        lookback=context.lookback,
        formula_version=str(row["formula_version"]),
    )
    token = _token(context.context_id, mode.value, str(row["evidence_path"]))
    return TeacherProposal(
        proposal_id=f"teacher_proposal_{token}",
        context=context,
        candidate=candidate,
        teacher_id=f"{_slug(mode.value)}_teacher",
        exact_mode=mode,
        proposed_action=TeacherAction(str(row["proposed_action"]).strip().lower()),
        confidence=float(row["confidence"]),
        uncertainty=float(row["uncertainty"]),
        expected_net_return=_optional_float(row["expected_net_return"]),
        lower_bound_net_return=_optional_float(row["lower_bound_net_return"]),
        expected_holding_bars=_optional_float(row["expected_holding_bars"]),
        entry_style=str(row["entry_style"]),
        exit_style=str(row["exit_style"]),
        invalidation_condition=str(row["invalidation_condition"]),
        required_regime=str(row["required_regime"]),
        source_system=str(row["source_system"]),
        authority=EvidenceAuthority.LOCAL_POINT_IN_TIME,
        point_in_time_status="confirmed",
        formula_version=str(row["formula_version"]),
        math_version=MATH_V2,
        source_timestamp=_timestamp(row["source_timestamp"]),
        blockers=_split(row["blockers"]),
        evidence_paths=_evidence_paths(row["evidence_path"]),
    )


def _assessment(row: pd.Series, context: CouncilContext) -> CriticAssessment:
    authority = EvidenceAuthority(str(row["authority"]).strip().lower())
    if authority not in {EvidenceAuthority.LOCAL_POINT_IN_TIME, EvidenceAuthority.REALIZED_OUTCOME}:
        raise ValueError("critic authority must be local point-in-time or realized outcome")
    if str(row["point_in_time_status"]).strip().lower() != "confirmed":
        raise ValueError("critic input must be point-in-time confirmed")
    critic_type = CriticType(str(row["critic_type"]).strip().lower())
    token = _token(context.context_id, critic_type.value, str(row["evidence_path"]))
    return CriticAssessment(
        assessment_id=f"critic_assessment_{token}",
        context=context,
        critic_id=f"{critic_type.value}_critic",
        critic_type=critic_type,
        verdict=CriticVerdict(str(row["verdict"]).strip().lower()),
        score=_optional_float(row["score"]),
        reason=str(row["reason"]),
        authority=authority,
        point_in_time_status="confirmed",
        source_timestamp=_timestamp(row["source_timestamp"]),
        blocker_codes=_split(row["blocker_codes"]),
        evidence_paths=_evidence_paths(row["evidence_path"]),
    )


def _math_marker_passes(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("status") == "passed"
        and payload.get("math_version") == MATH_V2
        and payload.get("acceptance_scope") == "core_math_library"
        and payload.get("all_checks_passed") is True
        and payload.get("generated_by") == "quant_platform.math_v2_acceptance"
    )


def _write_schema(path: Path, columns: tuple[str, ...]) -> None:
    pd.DataFrame(
        [{"column": column, "required": True, "authority": "local_point_in_time"} for column in columns]
    ).to_csv(path, index=False)


def _write_jsonl(path: Path, records: list[Any]) -> None:
    path.write_text(
        "".join(json.dumps(record.model_dump(mode="json"), sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _check(
    rows: list[dict[str, object]],
    check: str,
    passed: bool,
    observed: object,
    required: object,
    blocker: str,
) -> None:
    rows.append(
        {
            "check": check,
            "status": "PASS" if passed else "BLOCKED",
            "observed": observed,
            "required": required,
            "blocker": "" if passed else blocker,
        }
    )


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame()


def _timestamp(value: object) -> datetime:
    timestamp = pd.to_datetime(value, utc=True, errors="raise")
    return timestamp.to_pydatetime()


def _optional_float(value: object) -> float | None:
    if pd.isna(value) or str(value).strip() == "":
        return None
    return float(value)


def _split(value: object) -> tuple[str, ...]:
    if pd.isna(value):
        return ()
    return tuple(part.strip() for part in str(value).split(";") if part.strip())


def _evidence_paths(value: object) -> tuple[str, ...]:
    paths = _split(value)
    if not paths:
        raise ValueError("evidence_path cannot be blank")
    return paths


def _safe_mode(value: object) -> str:
    try:
        return normalize_exact_mode(str(value)).value
    except ValueError:
        return ""


def _group_key(value: object) -> tuple[str, ...]:
    values = value if isinstance(value, tuple) else (value,)
    return tuple(str(item) for item in values)


def _token(*values: str) -> str:
    return sha256("|".join(values).encode("utf-8")).hexdigest()[:20]


def _slug(value: str) -> str:
    return "_".join(str(value).lower().replace("zscorer", "zscore_r").split())
