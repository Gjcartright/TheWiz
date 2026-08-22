"""Fail-closed executor for the validated 19-stage daily research cadence."""

from __future__ import annotations

import io
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext, redirect_stderr, redirect_stdout
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import pandas as pd

from quant_platform.active_pipeline import CommandResult, _write_csv, _write_json, _write_text
from quant_platform.crypto_wizards_sweep import (
    WizardSweepResult,
    build_wizard_sweep_cells,
    run_authorized_wizard_discovery_sweep,
)
from quant_platform.orchestration.corrective_external_effects import (
    current_external_effect_issuer,
    external_effect_provider_session,
)
from quant_platform.orchestration.corrective_runtime import promote_staged_file
from quant_platform.orchestration.current_wizard_hyperliquid_cadence import (
    MINIMUM_FREE_BYTES,
    STAGES,
)
from quant_platform.wizard_credit_ledger import (
    DISCOVERY_LANE,
    validate_wizard_credit_lane_evidence,
)

ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "current_wizard_hyperliquid_daily_runner.v1"
WIZARD_API_KEY_ENV = "CRYPTO_WIZARDS_API_KEY"
SAFE_DAILY_ENVIRONMENT_KEYS = frozenset(
    {
        "HOME",
        "LANG",
        "LC_ALL",
        "LOGNAME",
        "PATH",
        "SHELL",
        "TMPDIR",
        "TZ",
        "USER",
    }
)
SNAPSHOT_VALIDATED_STAGES = frozenset(
    {
        "wizard_refresh_accounting",
        "pair_mode_orientation_handoff",
        "point_in_time_history",
        "canonical_one_x_replay",
        "funding_liquidity_cost_evidence",
        "observed_cost_replay",
        "purged_walkforward",
        "ou_optimal_outcome_stratification",
        "causal_regime_attribution",
        "robustness_stress",
        "cross_cell_concentration",
        "failure_attribution",
        "conditional_leverage_surface",
        "dated_learning_ledger",
        "frozen_chain_validation",
    }
)
STAGE3_PROOF_COMMANDS = frozenset(
    {
        "build-corrective-wizard-parity",
        "build-exhaustive-wizard-mode-proof-queue",
        "run-hyperliquid-wizard-mode-proofs",
        "review-wizard-dynamic-v2",
        "build-wizard-dynamic-v2-supreme-review",
        "evaluate-wizard-ou-v2-holdout",
        "register-wizard-ou-trend-selector-v1",
        "build-wizard-credit-budget",
        "build-wizard-next-capture-manifest",
        "build-wizard-reset-readiness",
        "build-wizard-comparator-review-control",
        "reconcile-wizard-capture-manifest",
        "run-wizard-ou-v3-holdout",
        "review-wizard-ou-v3",
        "register-wizard-ou-v4-holdout",
        "run-wizard-ou-v4-holdout",
        "build-wizard-ou-v4-failure-attribution",
        "build-wizard-ou-v4-supreme-review",
        "review-wizard-ou-v4",
        "register-wizard-ou-v5-holdout",
        "run-wizard-ou-v5-holdout",
        "build-wizard-ou-v5-failure-attribution",
        "build-wizard-ou-v5-supreme-review",
        "review-wizard-ou-v5",
        "register-wizard-copula-v2",
        "run-wizard-copula-proof",
    }
)


def run_current_wizard_hyperliquid_daily_pipeline(
    *,
    root: Path = ROOT,
    execute: bool = False,
    now: datetime | None = None,
    minimum_free_bytes: int = MINIMUM_FREE_BYTES,
    available_disk_bytes: int | None = None,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    wizard_stage_runner: Callable[..., WizardSweepResult] | None = None,
    semantic_evidence_builder: Callable[..., dict[str, str]] | None = None,
    stage_timeout_seconds: float | None = None,
    run_deadline_monotonic: float | None = None,
) -> CommandResult:
    """Plan or execute the research cadence; never submit Testnet or live orders."""

    as_of = (now or datetime.now(UTC)).astimezone(UTC)
    run_id = "cwdaily_" + sha256(
        f"{as_of.isoformat()}|{execute}|{minimum_free_bytes}".encode()
    ).hexdigest()[:20]
    stage3_isolation = _validate_stage3_command_isolation(root)
    if stage3_isolation["status"] != "PASS":
        forbidden = ",".join(stage3_isolation["forbidden_commands"])
        raise RuntimeError(f"daily_stage3_command_isolation_failed:{forbidden}")
    active = root / "reports" / "active"
    run_dir = root / "reports" / "runs" / "current_wizard_hyperliquid_daily" / run_id
    active.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=False)
    runner = command_runner
    child_env, wizard_credential = _secure_daily_child_environment(root)
    wizard_credential_reader = _wizard_credential_reader(
        root,
        wizard_credential,
    )
    external_credits_reconciled = 0
    rows: list[dict[str, object]] = []
    halted = False
    for sequence, stage, cadence, _, _, _ in STAGES:
        free_bytes = _free_bytes(root, available_disk_bytes)
        storage_ready = free_bytes >= minimum_free_bytes
        command, command_blocker = _stage_command(root, stage)
        status = "PLANNED"
        blocker = ""
        return_code: int | None = None
        output_hash = ""
        started_at = ""
        completed_at = ""
        semantic_status = "NOT_REQUIRED"
        semantic_blocker = ""
        semantic_evidence_path = ""
        semantic_evidence_sha256 = ""
        execution_mode = "PLANNED" if not execute else "NOT_EXECUTED"
        if halted:
            status = "NOT_STARTED_DEPENDENCY"
            blocker = "prior_stage_not_successful"
        elif not storage_ready:
            status = "BLOCKED_STORAGE" if sequence == 1 else "NOT_STARTED_STORAGE"
            blocker = (
                f"insufficient_free_space:free_bytes={free_bytes};"
                f"required_bytes={minimum_free_bytes}"
            )
            halted = True
        elif command_blocker:
            status = "BLOCKED_INPUT"
            blocker = command_blocker
            halted = True
        elif (
            execute
            and stage == "wizard_exhaustive_discovery"
            and _same_day_completed_wizard_sweep_payload(root=root, as_of=as_of) is None
            and wizard_credential["status"]
            not in {"PASS", "PASS_UNVERIFIED_UNTIL_AUTHORIZED_READ"}
        ):
            timestamp = datetime.now(UTC).isoformat()
            started_at = timestamp
            completed_at = timestamp
            execution_mode = "BLOCKED_CREDENTIAL"
            status = "BLOCKED_INPUT"
            blocker = str(wizard_credential["blocker"])
            halted = True
        elif execute:
            started = datetime.now(UTC)
            started_at = started.isoformat()
            env = child_env.copy()
            source_path = str(root / "src")
            env["PYTHONPATH"] = (
                source_path
                if not env.get("PYTHONPATH")
                else source_path + os.pathsep + env["PYTHONPATH"]
            )
            try:
                reused = (
                    _same_day_completed_wizard_sweep_payload(root=root, as_of=as_of)
                    if stage == "wizard_exhaustive_discovery"
                    else None
                )
                if reused is not None:
                    execution_mode = "REUSED_SAME_DAY"
                    result = subprocess.CompletedProcess(
                        command,
                        0,
                        stdout=json.dumps(reused, sort_keys=True),
                        stderr="",
                    )
                elif stage == "wizard_exhaustive_discovery":
                    execution_mode = "EXECUTED_IN_PROCESS_AUTHORITY_BOUND"
                    provider_context = (
                        external_effect_provider_session("crypto_wizards")
                        if current_external_effect_issuer("crypto_wizards")
                        is not None
                        or wizard_stage_runner is None
                        else nullcontext()
                    )
                    with provider_context:
                        sweep = (
                            wizard_stage_runner
                            or run_authorized_wizard_discovery_sweep
                        )(
                            root=root,
                            api_key=None,
                            credential_reader=wizard_credential_reader,
                            now=as_of,
                        )
                    if str(
                        sweep.summary.get("credit_reconciliation_status", "")
                    ) in {"PASS_RECONCILED", "REUSED_RECONCILIATION"}:
                        external_credits_reconciled = int(
                            sweep.summary.get("attempted_credits", 0) or 0
                        )
                    result = subprocess.CompletedProcess(
                        command,
                        0,
                        stdout=json.dumps(
                            {
                                "summary": sweep.summary,
                                "paths": {
                                    key: str(value)
                                    for key, value in sweep.paths.items()
                                },
                            },
                            sort_keys=True,
                        ),
                        stderr="",
                    )
                else:
                    timeout = _remaining_stage_timeout(
                        command=command,
                        stage_timeout_seconds=stage_timeout_seconds,
                        run_deadline_monotonic=run_deadline_monotonic,
                    )
                    if runner is None:
                        execution_mode = "EXECUTED_IN_PROCESS_AUTHORITY_BOUND"
                        provider_context = (
                            external_effect_provider_session(
                                "hyperliquid_public"
                            )
                            if stage == "hyperliquid_market_inventory"
                            else nullcontext()
                        )
                        with provider_context:
                            result = _run_cli_stage_in_process(
                                command,
                                root=root,
                                stage=stage,
                                timeout=timeout,
                            )
                    else:
                        execution_mode = "EXECUTED_INJECTED_RUNNER"
                        runner_kwargs: dict[str, object] = {
                            "cwd": root,
                            "env": env,
                            "capture_output": True,
                            "text": True,
                            "check": False,
                        }
                        if timeout is not None:
                            runner_kwargs["timeout"] = timeout
                        result = runner(command, **runner_kwargs)
                return_code = int(result.returncode)
                output = (result.stdout or "") + (result.stderr or "")
                output_hash = sha256(output.encode("utf-8")).hexdigest()
                completed = datetime.now(UTC)
                if return_code == 0:
                    semantic = (
                        semantic_evidence_builder or _build_stage_semantic_evidence
                    )(
                        root=root,
                        run_dir=run_dir,
                        run_id=run_id,
                        stage=stage,
                        stdout=result.stdout or "",
                        output_sha256=output_hash,
                        command_started_at=started,
                        command_completed_at=completed,
                    )
                    semantic_status = str(semantic["status"])
                    semantic_blocker = str(semantic["blocker"])
                    semantic_evidence_path = str(semantic["evidence_path"])
                    semantic_evidence_sha256 = str(semantic["evidence_sha256"])
                    if semantic_status == "PASS":
                        status = "PASS"
                    else:
                        status = "FAILED_SEMANTIC"
                        blocker = f"stage_semantic_validation:{semantic_blocker}"
                        halted = True
                else:
                    status = "FAILED"
                    blocker = f"stage_command_exit_code:{return_code}"
                    halted = True
            except Exception as exc:  # noqa: BLE001 - stage failures must halt and persist
                status = "FAILED"
                blocker = f"stage_command_error:{type(exc).__name__}"
                halted = True
            completed_at = datetime.now(UTC).isoformat()
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "daily_run_id": run_id,
                "sequence": sequence,
                "stage": stage,
                "cadence": cadence,
                "status": status,
                "blocker": blocker,
                "free_disk_bytes_before_stage": free_bytes,
                "minimum_free_disk_bytes": minimum_free_bytes,
                "resolved_command": shlex.join(command),
                "return_code": return_code,
                "output_sha256": output_hash,
                "semantic_validation_status": semantic_status,
                "semantic_validation_blocker": semantic_blocker,
                "semantic_evidence_path": semantic_evidence_path,
                "semantic_evidence_sha256": semantic_evidence_sha256,
                "started_at_utc": started_at,
                "completed_at_utc": completed_at,
                "execution_requested": execute,
                "execution_mode": execution_mode,
                "order_submission_authority": False,
                "live_trading_authorized": False,
            }
        )

    frame = pd.DataFrame(rows)
    if frame["status"].eq("PASS").all():
        run_status = "PASS"
    elif frame["status"].eq("PLANNED").all():
        run_status = "PLANNED"
    elif frame["status"].isin({"BLOCKED_STORAGE", "NOT_STARTED_STORAGE"}).any():
        run_status = "BLOCKED_STORAGE"
    elif frame["status"].isin({"FAILED", "FAILED_SEMANTIC"}).any():
        run_status = "FAILED"
    else:
        run_status = "BLOCKED"
    effect_accounting = _current_effect_accounting()
    summary = {
        "schema_version": SCHEMA_VERSION,
        "daily_run_id": run_id,
        "as_of": as_of.isoformat(),
        "run_status": run_status,
        "execution_requested": execute,
        "stages": len(frame),
        "stages_passed": int(frame["status"].eq("PASS").sum()),
        "stages_planned": int(frame["status"].eq("PLANNED").sum()),
        "stages_blocked_or_not_started": int(
            frame["status"].isin(
                {
                    "BLOCKED_STORAGE",
                    "NOT_STARTED_STORAGE",
                    "BLOCKED_INPUT",
                    "NOT_STARTED_DEPENDENCY",
                    "FAILED",
                    "FAILED_SEMANTIC",
                }
            ).sum()
        ),
        "minimum_free_disk_bytes": minimum_free_bytes,
        "final_free_disk_bytes": _free_bytes(root, available_disk_bytes),
        "stage3_command_isolation_status": stage3_isolation["status"],
        "stage3_forbidden_commands": stage3_isolation["forbidden_commands"],
        "stage3_external_execution_included": False,
        "wizard_api_credential_status": wizard_credential["status"],
        "wizard_api_credential_source": wizard_credential["source"],
        "wizard_api_credential_insecure_files": wizard_credential["insecure_files"],
        "external_calls": effect_accounting["external_calls"],
        "external_credits_reserved": effect_accounting[
            "external_credits_reserved"
        ],
        "external_credits_consumed": effect_accounting[
            "external_credits_consumed"
        ],
        "external_credits_reconciled": external_credits_reconciled,
        "testnet_execution_included": False,
        "order_submission_authority": False,
        "live_trading_authorized": False,
    }
    active_stem = (
        "current_wizard_hyperliquid_daily_run"
        if execute
        else "current_wizard_hyperliquid_daily_plan"
    )
    active_csv = active / f"{active_stem}_status.csv"
    active_manifest = active / f"{active_stem}_manifest.json"
    active_md = active / f"{active_stem}_summary.md"
    run_csv = run_dir / "daily_run_status.csv"
    run_manifest = run_dir / "manifest.json"
    _atomic_csv(frame, active_csv)
    _atomic_csv(frame, run_csv)
    _atomic_json(summary, active_manifest)
    _atomic_json(summary, run_manifest)
    _atomic_text(_markdown(summary, frame), active_md)
    return CommandResult(
        paths={
            "daily_run_status": active_csv,
            "daily_run_manifest": active_manifest,
            "daily_run_summary": active_md,
            "dated_daily_run_status": run_csv,
            "dated_daily_run_manifest": run_manifest,
        },
        summary=summary,
    )


def _run_cli_stage_in_process(
    command: list[str],
    *,
    root: Path,
    stage: str,
    timeout: float | None,
) -> subprocess.CompletedProcess[str]:
    """Run one exact daily CLI stage without losing supervisor context."""

    expected, blocker = _stage_command(root, stage)
    if blocker or command != expected:
        raise RuntimeError("daily_in_process_command_identity_mismatch")
    if (
        len(command) < 4
        or Path(command[0]).resolve() != Path(sys.executable).resolve()
        or command[1:3] != ["-m", "quant_platform.cli"]
    ):
        raise RuntimeError("daily_in_process_command_prefix_invalid")

    from quant_platform import cli

    if root.resolve() != cli.ROOT.resolve():
        raise RuntimeError("daily_in_process_cli_root_mismatch")
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    previous_cwd = Path.cwd()
    return_code = 0
    try:
        os.chdir(root)
        with (
            _in_process_stage_timeout(command, timeout),
            redirect_stdout(stdout_buffer),
            redirect_stderr(stderr_buffer),
        ):
            try:
                cli.main(command[3:], load_environment=False)
            except SystemExit as exc:
                if exc.code in {None, 0}:
                    return_code = 0
                elif isinstance(exc.code, int):
                    return_code = exc.code
                else:
                    return_code = 2
                    print(str(exc.code), file=sys.stderr)
    finally:
        os.chdir(previous_cwd)
    return subprocess.CompletedProcess(
        command,
        return_code,
        stdout=stdout_buffer.getvalue(),
        stderr=stderr_buffer.getvalue(),
    )


@contextmanager
def _in_process_stage_timeout(
    command: list[str],
    timeout: float | None,
) -> Iterator[None]:
    if timeout is None:
        yield
        return
    if timeout <= 0:
        raise subprocess.TimeoutExpired(command, timeout)
    if not hasattr(signal, "SIGALRM") or not hasattr(signal, "setitimer"):
        raise RuntimeError("daily_in_process_timeout_unsupported")
    prior_timer = signal.getitimer(signal.ITIMER_REAL)
    if prior_timer != (0.0, 0.0):
        raise RuntimeError("daily_in_process_timeout_already_active")
    prior_handler = signal.getsignal(signal.SIGALRM)

    def expire(_signum: int, _frame: object) -> None:
        raise subprocess.TimeoutExpired(command, timeout)

    signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, prior_handler)


def _remaining_stage_timeout(
    *,
    command: list[str],
    stage_timeout_seconds: float | None,
    run_deadline_monotonic: float | None,
) -> float | None:
    limits = [
        value
        for value in (
            stage_timeout_seconds,
            None
            if run_deadline_monotonic is None
            else run_deadline_monotonic - time.monotonic(),
        )
        if value is not None
    ]
    if not limits:
        return None
    timeout = min(limits)
    if timeout <= 0:
        raise subprocess.TimeoutExpired(command, max(timeout, 0))
    return timeout


def _current_effect_accounting() -> dict[str, int]:
    issuer = current_external_effect_issuer(
        "crypto_wizards"
    ) or current_external_effect_issuer("hyperliquid_public")
    if issuer is None:
        return {
            "external_calls": 0,
            "external_credits_reserved": 0,
            "external_credits_consumed": 0,
        }
    accounting = issuer.authority.run_accounting(
        run_id=issuer.run_id,
        intended_slot_id=issuer.intended_slot_id,
    )
    return {
        field: int(accounting.get(field, 0) or 0)
        for field in (
            "external_calls",
            "external_credits_reserved",
            "external_credits_consumed",
        )
    }


def _secure_daily_child_environment(root: Path) -> tuple[dict[str, str], dict[str, object]]:
    """Describe credential availability without reading or propagating secrets."""

    env = {
        key: os.environ[key]
        for key in SAFE_DAILY_ENVIRONMENT_KEYS
        if key in os.environ
    }
    candidates = (root / ".env.local", root / ".env")
    existing = [path for path in candidates if path.exists()]
    insecure = [
        _relative(path, root)
        for path in existing
        if not _secret_file_is_owner_only(path)
    ]
    if insecure:
        return env, {
            "status": "BLOCKED",
            "source": "",
            "blocker": "insecure_wizard_api_key_file:" + ",".join(insecure),
            "insecure_files": insecure,
        }

    if _environment_key_present(WIZARD_API_KEY_ENV):
        return env, {
            "status": "PASS",
            "source": "process_environment",
            "blocker": "",
            "insecure_files": [],
        }

    if existing:
        return env, {
            "status": "PASS_UNVERIFIED_UNTIL_AUTHORIZED_READ",
            "source": _relative(existing[0], root),
            "blocker": "",
            "insecure_files": [],
        }

    return env, {
        "status": "MISSING",
        "source": "",
        "blocker": f"{WIZARD_API_KEY_ENV}_missing",
        "insecure_files": [],
    }


def _environment_key_present(key: str) -> bool:
    """Observe a key name without crossing the guarded credential-value boundary."""

    return any(candidate == key for candidate in os.environ)


def _wizard_credential_reader(
    root: Path,
    credential: dict[str, object],
) -> Callable[[str], str | None]:
    source = str(credential.get("source", ""))

    def read(key: str) -> str | None:
        if key != WIZARD_API_KEY_ENV:
            return None
        if source == "process_environment":
            return os.getenv(key)
        if not source:
            return None
        path = root / source
        if not _secret_file_is_owner_only(path):
            return None
        return _env_file_value(path, key)

    return read


def _secret_file_is_owner_only(path: Path) -> bool:
    try:
        stat = path.stat()
    except OSError:
        return False
    owner_matches = not hasattr(os, "getuid") or stat.st_uid == os.getuid()
    return not path.is_symlink() and owner_matches and not bool(stat.st_mode & 0o077)


def _env_file_value(path: Path, key: str, *, preserve_empty: bool = False) -> str | None:
    if not path.is_file():
        return None
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return None
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        candidate, value = line.split("=", 1)
        if candidate.strip() != key:
            continue
        resolved = value.strip()
        if len(resolved) >= 2 and resolved[0] == resolved[-1] and resolved[0] in {"'", '"'}:
            resolved = resolved[1:-1]
        return resolved if resolved or preserve_empty else None
    return None


def _build_stage_semantic_evidence(
    *,
    root: Path,
    run_dir: Path,
    run_id: str,
    stage: str,
    stdout: str,
    output_sha256: str,
    command_started_at: datetime,
    command_completed_at: datetime,
) -> dict[str, str]:
    if stage == "wizard_exhaustive_discovery":
        return _build_wizard_stage_semantic_evidence(
            root=root,
            run_dir=run_dir,
            run_id=run_id,
            stage=stage,
            stdout=stdout,
            output_sha256=output_sha256,
            command_started_at=command_started_at,
            command_completed_at=command_completed_at,
        )
    if stage in SNAPSHOT_VALIDATED_STAGES:
        return _build_snapshot_stage_semantic_evidence(
            root=root,
            run_dir=run_dir,
            run_id=run_id,
            stage=stage,
            stdout=stdout,
            output_sha256=output_sha256,
            command_started_at=command_started_at,
            command_completed_at=command_completed_at,
        )
    return _build_artifact_stage_semantic_evidence(
        root=root,
        run_dir=run_dir,
        run_id=run_id,
        stage=stage,
        stdout=stdout,
        output_sha256=output_sha256,
        command_started_at=command_started_at,
        command_completed_at=command_completed_at,
    )


def _build_wizard_stage_semantic_evidence(
    *,
    root: Path,
    run_dir: Path,
    run_id: str,
    stage: str,
    stdout: str,
    output_sha256: str,
    command_started_at: datetime,
    command_completed_at: datetime,
) -> dict[str, str]:

    blockers: list[str] = []
    semantic_dir = run_dir / "semantic_evidence"
    semantic_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = semantic_dir / "wizard_exhaustive_discovery.json"
    manifest_snapshot = semantic_dir / "wizard_sweep_manifest.csv"
    summary_snapshot = semantic_dir / "wizard_sweep_summary.json"
    payload: dict[str, object] = {}
    summary: dict[str, object] = {}
    paths: dict[str, object] = {}
    try:
        parsed = json.loads(stdout)
        if not isinstance(parsed, dict):
            raise TypeError("Wizard CLI output must be an object")
        payload = parsed
        summary_value = payload.get("summary")
        paths_value = payload.get("paths")
        if not isinstance(summary_value, dict) or not isinstance(paths_value, dict):
            raise TypeError("Wizard CLI output is missing summary or paths")
        summary = summary_value
        paths = paths_value
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        blockers.append(f"wizard_cli_output_invalid:{type(exc).__name__}")

    sweep_id = str(summary.get("sweep_id", ""))
    expected_cells = build_wizard_sweep_cells(sweep_id=sweep_id or "missing")
    expected_request_ids = {cell.request_id for cell in expected_cells}
    expected_credits = sum(cell.credit_cost for cell in expected_cells)
    manifest_path = _safe_artifact_path(root, str(paths.get("manifest", "")))
    summary_path = _safe_artifact_path(root, str(paths.get("summary", "")))
    manifest = _read_csv(manifest_path)
    artifact_summary = _read_json(summary_path)

    if not sweep_id:
        blockers.append("wizard_sweep_id_missing")
    if summary.get("sweep_complete") is not True:
        blockers.append("wizard_sweep_not_complete")
    if str(summary.get("discovery_authority", "")) != "complete_discovery":
        blockers.append("wizard_discovery_authority_not_complete")
    if str(summary.get("blocker", "")):
        blockers.append("wizard_sweep_reports_blocker")
    if summary.get("credit_usage_known") is not True:
        blockers.append("wizard_credit_usage_unknown")
    for field in ("planned_cells", "completed_cells"):
        if int(summary.get(field, 0) or 0) != len(expected_cells):
            blockers.append(f"wizard_{field}_mismatch")
    for field in ("planned_credits", "attempted_credits", "completed_credits"):
        if int(summary.get(field, 0) or 0) != expected_credits:
            blockers.append(f"wizard_{field}_mismatch")
    if str(summary.get("credit_reservation_status", "")) not in {"PASS", "REUSED"}:
        blockers.append("wizard_credit_reservation_not_proven")
    if str(summary.get("credit_reconciliation_status", "")) not in {
        "PASS_RECONCILED",
        "REUSED_RECONCILIATION",
    }:
        blockers.append("wizard_credit_reconciliation_not_proven")

    same_day_reuse = payload.get("daily_same_day_reuse") is True
    reuse_date = str(payload.get("reuse_date_utc", ""))
    sweep_started = pd.to_datetime(summary.get("started_at"), utc=True, errors="coerce")
    freshness_start = pd.Timestamp(command_started_at).tz_convert("UTC") - pd.Timedelta(seconds=5)
    freshness_end = pd.Timestamp(command_completed_at).tz_convert("UTC") + pd.Timedelta(seconds=5)
    if same_day_reuse:
        if (
            pd.isna(sweep_started)
            or not reuse_date
            or sweep_started.date().isoformat() != reuse_date
            or sweep_started > freshness_end
        ):
            blockers.append("wizard_reused_sweep_not_same_day_or_preexisting")
    elif pd.isna(sweep_started) or not freshness_start <= sweep_started <= freshness_end:
        blockers.append("wizard_sweep_output_not_fresh_for_stage")
    if artifact_summary != summary:
        blockers.append("wizard_summary_artifact_output_mismatch")

    required_manifest_columns = {
        "sweep_id",
        "request_id",
        "status",
        "credit_cost",
        "response_hash",
        "evidence_path",
    }
    if manifest.empty or not required_manifest_columns.issubset(manifest.columns):
        blockers.append("wizard_manifest_missing_or_invalid")
    else:
        if len(manifest) != len(expected_cells):
            blockers.append("wizard_manifest_cell_count_mismatch")
        if set(manifest["request_id"].astype(str)) != expected_request_ids:
            blockers.append("wizard_manifest_request_identity_mismatch")
        if not manifest["sweep_id"].astype(str).eq(sweep_id).all():
            blockers.append("wizard_manifest_sweep_identity_mismatch")
        if not manifest["status"].astype(str).eq("completed").all():
            blockers.append("wizard_manifest_has_noncompleted_cells")
        if int(pd.to_numeric(manifest["credit_cost"], errors="coerce").fillna(0).sum()) != (
            expected_credits
        ):
            blockers.append("wizard_manifest_credit_total_mismatch")

    raw_bindings: list[dict[str, str]] = []
    if not manifest.empty and required_manifest_columns.issubset(manifest.columns):
        raw_root = (root / "data" / "raw" / "crypto_wizards" / "prescanned").resolve()
        for row in manifest.to_dict("records"):
            raw_path = _safe_artifact_path(root, str(row.get("evidence_path", "")))
            try:
                raw_path.resolve().relative_to(raw_root)
                envelope = _read_json(raw_path)
                metadata = envelope.get("capture_metadata")
                response = envelope.get("response")
                if not isinstance(metadata, dict):
                    raise TypeError("raw capture metadata missing")
                response_hash = sha256(_canonical_json(response).encode("utf-8")).hexdigest()
                if (
                    metadata.get("sweep_id") != sweep_id
                    or metadata.get("request_id") != str(row.get("request_id", ""))
                    or metadata.get("response_hash") != str(row.get("response_hash", ""))
                    or response_hash != str(row.get("response_hash", ""))
                ):
                    raise ValueError("raw capture identity mismatch")
                raw_bindings.append(
                    {
                        "path": str(raw_path.relative_to(root.resolve())),
                        "file_sha256": sha256(raw_path.read_bytes()).hexdigest(),
                        "response_hash": response_hash,
                    }
                )
            except (OSError, TypeError, ValueError) as exc:
                blockers.append(
                    f"wizard_raw_capture_invalid:{row.get('request_id', '')}:{type(exc).__name__}"
                )

    credit_usage_bindings: list[dict[str, str]] = []
    credit_usage_root = (
        root / "data" / "raw" / "crypto_wizards" / "credit_usage"
    ).resolve()
    for phase in ("before", "after"):
        relative_path = str(
            summary.get(f"credit_usage_{phase}_evidence_path", "")
        )
        expected_file_hash = str(
            summary.get(f"credit_usage_{phase}_evidence_sha256", "")
        )
        evidence_path = _safe_artifact_path(root, relative_path)
        try:
            evidence_path.resolve().relative_to(credit_usage_root)
            envelope = _read_json(evidence_path)
            metadata = envelope.get("capture_metadata")
            response = envelope.get("response")
            if not isinstance(metadata, dict):
                raise TypeError("credit usage capture metadata missing")
            response_hash = sha256(
                _canonical_json(response).encode("utf-8")
            ).hexdigest()
            file_hash = _file_sha256(evidence_path)
            if (
                metadata.get("sweep_id") != sweep_id
                or metadata.get("capture_type") != "credit_usage"
                or metadata.get("phase") != phase
                or metadata.get("response_hash") != response_hash
                or file_hash != expected_file_hash
            ):
                raise ValueError("credit usage evidence identity mismatch")
            credit_usage_bindings.append(
                {
                    "phase": phase,
                    "path": _relative(evidence_path, root),
                    "file_sha256": file_hash,
                    "response_hash": response_hash,
                }
            )
        except (OSError, TypeError, ValueError) as exc:
            blockers.append(
                f"wizard_credit_usage_evidence_invalid:{phase}:{type(exc).__name__}"
            )

    credit = validate_wizard_credit_lane_evidence(
        root=root,
        lane=DISCOVERY_LANE,
        credit_date_utc=(
            reuse_date
            if same_day_reuse and reuse_date
            else command_started_at.astimezone(UTC).date().isoformat()
        ),
        reservation_id=str(summary.get("credit_reservation_id", "")),
        reconciliation_id=str(summary.get("credit_reconciliation_id", "")),
        reservation_path=str(summary.get("credit_reservation_path", "")),
        reconciliation_path=str(summary.get("credit_reconciliation_path", "")),
    )
    if credit.get("status") != "PASS":
        blockers.append(f"wizard_credit_evidence:{credit.get('blocker', 'invalid')}")
    elif (
        int(credit.get("planned_credits", 0)) != expected_credits
        or int(credit.get("attempted_credits", 0)) != expected_credits
        or int(credit.get("completed_credits", 0)) != expected_credits
        or int(credit.get("external_requests", 0)) != len(expected_cells)
    ):
        blockers.append("wizard_credit_evidence_totals_mismatch")

    if manifest_path.is_file():
        _atomic_bytes(manifest_path.read_bytes(), manifest_snapshot)
    if summary_path.is_file():
        _atomic_bytes(summary_path.read_bytes(), summary_snapshot)
    semantic_body: dict[str, object] = {
        "schema_version": "thewiz.daily_wizard_semantic_evidence.v1",
        "daily_run_id": run_id,
        "stage": stage,
        "validated_at_utc": datetime.now(UTC).isoformat(),
        "status": "PASS" if not blockers else "BLOCKED",
        "blockers": sorted(set(blockers)),
        "command_output_sha256": output_sha256,
        "same_day_sweep_reused": same_day_reuse,
        "sweep_id": sweep_id,
        "expected_cells": len(expected_cells),
        "expected_credits": expected_credits,
        "raw_snapshot_count": len(raw_bindings),
        "raw_bindings": raw_bindings,
        "raw_response_set_sha256": sha256(
            _canonical_json(raw_bindings).encode("utf-8")
        ).hexdigest(),
        "credit_usage_snapshot_count": len(credit_usage_bindings),
        "credit_usage_bindings": credit_usage_bindings,
        "credit_usage_binding_set_sha256": sha256(
            _canonical_json(credit_usage_bindings).encode("utf-8")
        ).hexdigest(),
        "manifest_snapshot_path": _relative(manifest_snapshot, root)
        if manifest_snapshot.is_file()
        else "",
        "manifest_snapshot_sha256": _file_sha256(manifest_snapshot),
        "summary_snapshot_path": _relative(summary_snapshot, root)
        if summary_snapshot.is_file()
        else "",
        "summary_snapshot_sha256": _file_sha256(summary_snapshot),
        "credit_evidence": credit,
        "research_only": True,
        "candidate_promotion_authority": False,
        "order_submission_included": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    semantic_body["semantic_receipt_sha256"] = sha256(
        _canonical_json(semantic_body).encode("utf-8")
    ).hexdigest()
    _atomic_json(semantic_body, receipt_path)
    return {
        "status": str(semantic_body["status"]),
        "blocker": ";".join(semantic_body["blockers"]),
        "evidence_path": _relative(receipt_path, root),
        "evidence_sha256": _file_sha256(receipt_path),
    }


def _same_day_completed_wizard_sweep_payload(
    *, root: Path, as_of: datetime
) -> dict[str, object] | None:
    """Reuse one complete UTC-day sweep instead of spending the lane twice."""

    active = root / "reports" / "active"
    manifest_path = active / "wizard_sweep_manifest.csv"
    summary_path = active / "wizard_sweep_summary.json"
    summary = _read_json(summary_path)
    manifest = _read_csv(manifest_path)
    sweep_id = str(summary.get("sweep_id", ""))
    started = pd.to_datetime(summary.get("started_at"), utc=True, errors="coerce")
    expected = build_wizard_sweep_cells(sweep_id=sweep_id or "missing")
    expected_request_ids = {cell.request_id for cell in expected}
    if (
        not sweep_id
        or pd.isna(started)
        or started.date() != as_of.astimezone(UTC).date()
        or summary.get("sweep_complete") is not True
        or str(summary.get("discovery_authority", "")) != "complete_discovery"
        or str(summary.get("blocker", ""))
        or summary.get("credit_usage_known") is not True
        or len(manifest) != len(expected)
        or not {"sweep_id", "request_id", "status"}.issubset(manifest.columns)
        or not manifest["sweep_id"].astype(str).eq(sweep_id).all()
        or set(manifest["request_id"].astype(str)) != expected_request_ids
        or not manifest["status"].astype(str).eq("completed").all()
    ):
        return None
    return {
        "summary": summary,
        "paths": {
            "manifest": str(manifest_path),
            "summary": str(summary_path),
        },
        "daily_same_day_reuse": True,
        "reuse_date_utc": as_of.astimezone(UTC).date().isoformat(),
        "reuse_policy": "one_complete_exhaustive_sweep_per_utc_day",
        "reused_at_utc": as_of.astimezone(UTC).isoformat(),
    }


def _build_snapshot_stage_semantic_evidence(
    *,
    root: Path,
    run_dir: Path,
    run_id: str,
    stage: str,
    stdout: str,
    output_sha256: str,
    command_started_at: datetime,
    command_completed_at: datetime,
) -> dict[str, str]:
    summary, paths, blockers = _parse_json_cli_output(stdout)
    if _contains_execution_authority(summary):
        blockers.append("stage_summary_contains_execution_authority")
    freshness_blocker = _summary_freshness_blocker(
        summary,
        command_started_at=command_started_at,
        command_completed_at=command_completed_at,
    )
    if freshness_blocker:
        blockers.append(freshness_blocker)

    validation_path = _safe_artifact_path(root, str(paths.get("snapshot_validation", "")))
    manifest_path = _safe_artifact_path(root, str(paths.get("snapshot_manifest", "")))
    validation = _read_csv(validation_path)
    manifest = _read_json(manifest_path)
    if not _artifacts_fresh_for_command(
        (validation_path, manifest_path),
        command_started_at=command_started_at,
        command_completed_at=command_completed_at,
    ):
        blockers.append("snapshot_contract_artifacts_not_fresh")
    if validation.empty or "status" not in validation.columns:
        blockers.append("snapshot_validation_missing_or_invalid")
    elif not validation["status"].astype(str).str.upper().eq("PASS").all():
        blockers.append("snapshot_validation_has_nonpassing_rows")
    if not manifest:
        blockers.append("snapshot_manifest_missing_or_invalid")
    elif _contains_execution_authority(manifest):
        blockers.append("snapshot_manifest_contains_execution_authority")

    snapshot_paths = {
        str(key): _safe_artifact_path(root, str(value))
        for key, value in paths.items()
        if str(key).startswith("snapshot_")
    }
    if "snapshot_validation" not in snapshot_paths or "snapshot_manifest" not in snapshot_paths:
        blockers.append("required_snapshot_paths_missing")
    bindings, binding_blockers = _copy_semantic_artifacts(
        root=root,
        run_dir=run_dir,
        stage=stage,
        artifacts=snapshot_paths,
    )
    blockers.extend(binding_blockers)
    return _write_generic_semantic_receipt(
        root=root,
        run_dir=run_dir,
        run_id=run_id,
        stage=stage,
        output_sha256=output_sha256,
        summary=summary,
        artifact_bindings=bindings,
        validation_rows=len(validation),
        blockers=blockers,
    )


def _build_artifact_stage_semantic_evidence(
    *,
    root: Path,
    run_dir: Path,
    run_id: str,
    stage: str,
    stdout: str,
    output_sha256: str,
    command_started_at: datetime,
    command_completed_at: datetime,
) -> dict[str, str]:
    summary, paths, blockers = _parse_json_cli_output(stdout)
    if stage not in {"storage_preflight", "hyperliquid_market_inventory", "monitor_dashboard"}:
        blockers.append("stage_semantic_contract_not_registered")
    if _contains_execution_authority(summary):
        blockers.append("stage_summary_contains_execution_authority")
    freshness_blocker = _summary_freshness_blocker(
        summary,
        command_started_at=command_started_at,
        command_completed_at=command_completed_at,
        required=stage == "hyperliquid_market_inventory",
    )
    if freshness_blocker:
        blockers.append(freshness_blocker)

    selected_paths: dict[str, Path] = {}
    validation_rows = 0
    if stage == "storage_preflight":
        checks = int(summary.get("checks", 0) or 0)
        ready = int(summary.get("ready", 0) or 0)
        blocked = int(summary.get("blocked", 0) or 0)
        check_path = _safe_artifact_path(root, str(paths.get("system_check", "")))
        check_frame = _read_csv(check_path)
        validation_rows = len(check_frame)
        if checks <= 0 or checks != ready + blocked or len(check_frame) != checks:
            blockers.append("system_check_accounting_mismatch")
        selected_paths = {
            key: _safe_artifact_path(root, str(value)) for key, value in paths.items()
        }
    elif stage == "hyperliquid_market_inventory":
        inventory_path = _safe_artifact_path(root, str(paths.get("inventory", "")))
        evidence_path = _safe_artifact_path(
            root,
            str(paths.get("inventory_evidence", "")),
        )
        inventory = _read_csv(inventory_path)
        evidence = _read_json(evidence_path)
        validation_rows = len(inventory)
        if (
            inventory.empty
            or int(summary.get("rows", 0) or 0) != len(inventory)
            or int(summary.get("tradable_perps", 0) or 0) <= 0
            or int(summary.get("fetch_blocked_rows", 0) or 0) != 0
        ):
            blockers.append("hyperliquid_inventory_incomplete_or_blocked")
        if not _valid_hyperliquid_inventory_evidence(
            root=root,
            evidence=evidence,
        ):
            blockers.append("hyperliquid_inventory_evidence_invalid")
        selected_paths = {
            "inventory": inventory_path,
            "inventory_evidence": evidence_path,
        }
    elif stage == "monitor_dashboard":
        if int(summary.get("dashboard_files", 0) or 0) != len(paths):
            blockers.append("dashboard_file_count_mismatch")
        for key in ("command_center", "data_health", "candidate_ranking"):
            if key in paths:
                selected_paths[key] = _safe_artifact_path(root, str(paths[key]))
        if "command_center" not in selected_paths:
            blockers.append("dashboard_command_center_missing")
        validation_rows = len(paths)

    if selected_paths and not _artifacts_fresh_for_command(
        tuple(selected_paths.values()),
        command_started_at=command_started_at,
        command_completed_at=command_completed_at,
    ):
        blockers.append("stage_artifacts_not_fresh")
    bindings, binding_blockers = _copy_semantic_artifacts(
        root=root,
        run_dir=run_dir,
        stage=stage,
        artifacts=selected_paths,
    )
    blockers.extend(binding_blockers)
    return _write_generic_semantic_receipt(
        root=root,
        run_dir=run_dir,
        run_id=run_id,
        stage=stage,
        output_sha256=output_sha256,
        summary=summary,
        artifact_bindings=bindings,
        validation_rows=validation_rows,
        blockers=blockers,
    )


def _valid_hyperliquid_inventory_evidence(
    *,
    root: Path,
    evidence: dict[str, object],
) -> bool:
    if (
        evidence.get("schema_version")
        != "thewiz.hyperliquid_public_inventory_evidence.v1"
        or int(evidence.get("external_requests", 0) or 0) != 2
        or int(evidence.get("external_credits", -1) or 0) != 0
        or evidence.get("blockers") != []
        or evidence.get("order_submission_included") is not False
        or evidence.get("live_trading_authorized") is not False
    ):
        return False
    reservation_path = _safe_artifact_path(
        root,
        str(evidence.get("reservation_path", "")),
    )
    try:
        reservation_path.resolve().relative_to(
            (root / "data" / "research" / "hyperliquid_public_ledger").resolve()
        )
    except (OSError, ValueError):
        return False
    if _file_sha256(reservation_path) != str(
        evidence.get("reservation_sha256", "")
    ):
        return False
    refresh_id = str(evidence.get("refresh_id", ""))
    reservation = _read_json(reservation_path)
    if (
        not refresh_id
        or reservation.get("schema_version")
        != "thewiz.hyperliquid_public_reservation.v1"
        or reservation.get("reservation_id") != refresh_id
        or reservation.get("provider_id") != "hyperliquid_public"
        or int(reservation.get("max_total_requests", 0) or 0) != 2
        or int(reservation.get("max_total_credits", -1) or 0) != 0
        or reservation.get("order_submission_included") is not False
        or reservation.get("live_trading_authorized") is not False
    ):
        return False
    bindings = evidence.get("response_bindings")
    if not isinstance(bindings, list) or len(bindings) != 2:
        return False
    normalized: list[dict[str, str]] = []
    request_types: set[str] = set()
    raw_root = (
        root / "data" / "raw" / "hyperliquid" / "testnet_market_inventory"
    ).resolve()
    for item in bindings:
        if not isinstance(item, dict):
            return False
        request_type = str(item.get("request_type", ""))
        response_path = _safe_artifact_path(root, str(item.get("path", "")))
        try:
            response_path.resolve().relative_to(raw_root)
        except (OSError, ValueError):
            return False
        file_hash = _file_sha256(response_path)
        if file_hash != str(item.get("sha256", "")):
            return False
        envelope = _read_json(response_path)
        metadata = envelope.get("capture_metadata")
        response = envelope.get("response")
        if not isinstance(metadata, dict):
            return False
        response_hash = sha256(
            _canonical_json(response).encode("utf-8")
        ).hexdigest()
        if (
            metadata.get("schema_version")
            != "thewiz.hyperliquid_public_response.v1"
            or metadata.get("refresh_id") != refresh_id
            or metadata.get("request_type") != request_type
            or metadata.get("response_sha256") != response_hash
            or not isinstance(envelope.get("request"), dict)
            or envelope["request"].get("type") != request_type
        ):
            return False
        request_types.add(request_type)
        normalized.append(
            {
                "request_type": request_type,
                "path": _relative(response_path, root),
                "sha256": file_hash,
            }
        )
    return request_types == {"meta", "allMids"} and sha256(
        _canonical_json(normalized).encode("utf-8")
    ).hexdigest() == str(evidence.get("response_binding_set_sha256", ""))


def _parse_json_cli_output(
    stdout: str,
) -> tuple[dict[str, object], dict[str, object], list[str]]:
    blockers: list[str] = []
    try:
        payload = json.loads(stdout)
        if not isinstance(payload, dict):
            raise TypeError("CLI output must be an object")
        summary = payload.get("summary")
        paths = payload.get("paths")
        if not isinstance(summary, dict) or not isinstance(paths, dict):
            raise TypeError("CLI output is missing summary or paths")
        return summary, paths, blockers
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        return {}, {}, [f"stage_cli_output_invalid:{type(exc).__name__}"]


def _copy_semantic_artifacts(
    *, root: Path, run_dir: Path, stage: str, artifacts: dict[str, Path]
) -> tuple[list[dict[str, object]], list[str]]:
    bindings: list[dict[str, object]] = []
    blockers: list[str] = []
    target_dir = run_dir / "semantic_evidence" / stage / "artifacts"
    for key, source in sorted(artifacts.items()):
        if not source.is_file():
            blockers.append(f"semantic_artifact_missing:{key}")
            continue
        safe_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", key)
        target = target_dir / f"{safe_key}{source.suffix}"
        try:
            _atomic_copy(source, target)
            bindings.append(
                {
                    "key": key,
                    "source_path": _relative(source, root),
                    "source_sha256": _file_sha256(source),
                    "snapshot_path": _relative(target, root),
                    "snapshot_sha256": _file_sha256(target),
                    "size_bytes": target.stat().st_size,
                }
            )
        except OSError as exc:
            blockers.append(f"semantic_artifact_copy_failed:{key}:{type(exc).__name__}")
    return bindings, blockers


def _write_generic_semantic_receipt(
    *,
    root: Path,
    run_dir: Path,
    run_id: str,
    stage: str,
    output_sha256: str,
    summary: dict[str, object],
    artifact_bindings: list[dict[str, object]],
    validation_rows: int,
    blockers: list[str],
) -> dict[str, str]:
    receipt_path = run_dir / "semantic_evidence" / stage / "receipt.json"
    body: dict[str, object] = {
        "schema_version": "thewiz.daily_stage_semantic_evidence.v1",
        "daily_run_id": run_id,
        "stage": stage,
        "validated_at_utc": datetime.now(UTC).isoformat(),
        "status": "PASS" if not blockers else "BLOCKED",
        "blockers": sorted(set(blockers)),
        "command_output_sha256": output_sha256,
        "stage_summary": summary,
        "validation_rows": validation_rows,
        "artifact_bindings": artifact_bindings,
        "artifact_binding_set_sha256": sha256(
            _canonical_json(artifact_bindings).encode("utf-8")
        ).hexdigest(),
        "research_only": True,
        "candidate_promotion_authority": False,
        "order_submission_included": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    body["semantic_receipt_sha256"] = sha256(
        _canonical_json(body).encode("utf-8")
    ).hexdigest()
    _atomic_json(body, receipt_path)
    return {
        "status": str(body["status"]),
        "blocker": ";".join(body["blockers"]),
        "evidence_path": _relative(receipt_path, root),
        "evidence_sha256": _file_sha256(receipt_path),
    }


def _contains_execution_authority(payload: object) -> bool:
    authority_keys = {
        "candidate_promotion_authority",
        "promotion_authority",
        "order_submission_authority",
        "order_submission_included",
        "testnet_order_authority",
        "live_trading_authorized",
    }
    if isinstance(payload, dict):
        return any(
            (str(key) in authority_keys and _truthy(value))
            or _contains_execution_authority(value)
            for key, value in payload.items()
        )
    if isinstance(payload, list):
        return any(_contains_execution_authority(value) for value in payload)
    return False


def _summary_freshness_blocker(
    summary: dict[str, object],
    *,
    command_started_at: datetime,
    command_completed_at: datetime,
    required: bool = False,
) -> str:
    candidates = [
        summary.get(key)
        for key in (
            "created_at",
            "created_at_utc",
            "started_at",
            "started_at_utc",
            "as_of",
            "as_of_utc",
            "generated_at",
            "generated_at_utc",
            "checked_at_max_utc",
        )
        if summary.get(key) not in {None, ""}
    ]
    if not candidates:
        return "stage_summary_freshness_timestamp_missing" if required else ""
    start = pd.Timestamp(command_started_at).tz_convert("UTC") - pd.Timedelta(seconds=5)
    end = pd.Timestamp(command_completed_at).tz_convert("UTC") + pd.Timedelta(seconds=5)
    timestamps = pd.to_datetime(pd.Series(candidates), utc=True, errors="coerce").dropna()
    if timestamps.empty or not timestamps.between(start, end).any():
        return "stage_summary_not_fresh_for_command"
    return ""


def _artifacts_fresh_for_command(
    paths: tuple[Path, ...],
    *,
    command_started_at: datetime,
    command_completed_at: datetime,
) -> bool:
    if not paths:
        return False
    start = pd.Timestamp(command_started_at).tz_convert("UTC") - pd.Timedelta(seconds=5)
    end = pd.Timestamp(command_completed_at).tz_convert("UTC") + pd.Timedelta(seconds=5)
    for path in paths:
        try:
            modified = pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC")
        except OSError:
            return False
        if not start <= modified <= end:
            return False
    return True


def _stage_command(root: Path, stage: str) -> tuple[list[str], str]:
    prefix = [sys.executable, "-m", "quant_platform.cli"]
    commands: dict[str, list[str]] = {
        "storage_preflight": ["system-check"],
        "wizard_exhaustive_discovery": [
            "crypto-wizards-full-sweep",
            "--execute-wizard-sweep",
        ],
        "wizard_refresh_accounting": ["build-exhaustive-wizard-api-refresh-delta"],
        "hyperliquid_market_inventory": ["hyperliquid-testnet-market-inventory"],
        "pair_mode_orientation_handoff": ["build-current-wizard-hyperliquid-handoff"],
        "canonical_one_x_replay": ["run-current-wizard-hyperliquid-canonical-replay"],
        "observed_cost_replay": ["run-current-wizard-hyperliquid-observed-cost-replay"],
        "purged_walkforward": ["run-current-wizard-hyperliquid-walkforward"],
        "ou_optimal_outcome_stratification": ["build-current-wizard-ou-optimal-overlay"],
        "causal_regime_attribution": ["build-current-wizard-hyperliquid-regime-attribution"],
        "robustness_stress": ["run-current-wizard-hyperliquid-robustness"],
        "cross_cell_concentration": ["build-current-wizard-hyperliquid-concentration"],
        "failure_attribution": ["build-current-wizard-hyperliquid-failure-attribution"],
        "conditional_leverage_surface": ["build-current-wizard-hyperliquid-leverage-surface"],
        "dated_learning_ledger": ["build-current-wizard-hyperliquid-learning-ledger"],
        "frozen_chain_validation": ["validate-current-wizard-hyperliquid-chain"],
        "monitor_dashboard": [
            "build-command-dashboard",
            "--dashboard-refresh-profile",
            "monitor",
        ],
    }
    if stage in {"point_in_time_history", "funding_liquidity_cost_evidence"}:
        pair_keys = _ready_pair_group_keys(root)
        if not pair_keys:
            return prefix, "ready_pair_group_keys_missing"
        command = (
            "materialize-current-wizard-hyperliquid-history"
            if stage == "point_in_time_history"
            else "materialize-current-wizard-hyperliquid-cost-evidence"
        )
        args = [command, "--current-pair-group-keys", ",".join(pair_keys)]
        if stage == "point_in_time_history":
            args.extend(["--minimum-free-disk-mib", str(MINIMUM_FREE_BYTES // 1024**2)])
        return [*prefix, *args], ""
    args = commands.get(stage)
    return ([*prefix, *args], "") if args is not None else (prefix, "unknown_daily_stage")


def _validate_stage3_command_isolation(
    root: Path,
    *,
    stage_command_resolver: Callable[[Path, str], tuple[list[str], str]] | None = None,
) -> dict[str, object]:
    resolver = stage_command_resolver or _stage_command
    resolved_tokens: set[str] = set()
    for _, stage, _, _, _, _ in STAGES:
        command, _ = resolver(root, stage)
        resolved_tokens.update(command)
    forbidden_commands = stage3_forbidden_command_tokens(resolved_tokens)
    return {
        "status": "BLOCKED" if forbidden_commands else "PASS",
        "forbidden_commands": forbidden_commands,
        "stage3_external_execution_included": bool(forbidden_commands),
    }


def stage3_forbidden_command_tokens(tokens: set[str]) -> list[str]:
    """Identify Stage 3 CLI, module, and direct-script execution tokens."""

    forbidden: set[str] = set(tokens & STAGE3_PROOF_COMMANDS)
    module_prefix = "quant_platform.orchestration.corrective_wizard_"
    for token in tokens:
        normalized = str(token).replace("\\", "/")
        if normalized.startswith(module_prefix):
            forbidden.add(str(token))
            continue
        candidate = Path(normalized)
        if candidate.suffix == ".py" and candidate.stem.startswith("corrective_wizard_"):
            forbidden.add(str(token))
    return sorted(forbidden)


def _ready_pair_group_keys(root: Path) -> tuple[str, ...]:
    path = root / "reports" / "active" / "current_wizard_hyperliquid_pair_history_queue.csv"
    try:
        frame = pd.read_csv(path, keep_default_na=False)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return ()
    if not {"pair_group_key", "history_request_status"}.issubset(frame.columns):
        return ()
    ready = frame.loc[
        frame["history_request_status"].astype(str).eq("READY_TO_FETCH"),
        "pair_group_key",
    ]
    return tuple(sorted(set(ready.astype(str).str.strip()) - {""}))


def _free_bytes(root: Path, available_disk_bytes: int | None) -> int:
    return int(
        available_disk_bytes
        if available_disk_bytes is not None
        else shutil.disk_usage(root).free
    )


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, keep_default_na=False)
    except (OSError, UnicodeDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame()


def _safe_artifact_path(root: Path, value: str) -> Path:
    if not value:
        return root / ".missing-semantic-artifact"
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        resolved = candidate.resolve()
        resolved.relative_to(root.resolve())
        return resolved
    except (OSError, ValueError):
        return root / ".invalid-semantic-artifact"


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _file_sha256(path: Path) -> str:
    try:
        return sha256(path.read_bytes()).hexdigest() if path.is_file() else ""
    except OSError:
        return ""


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _truthy(value: object) -> bool:
    return value is True or str(value).strip().lower() in {"1", "true", "yes"}


def _atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    promote_staged_file(temporary, target)


def _atomic_bytes(payload: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    promote_staged_file(temporary, path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    _write_csv(frame, path)


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    _write_json(path, payload)


def _atomic_text(text: str, path: Path) -> None:
    _write_text(path, text)


def _markdown(summary: dict[str, object], frame: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# Current Wizard Hyperliquid Daily Run",
            "",
            f"- Run: `{summary['daily_run_id']}`",
            f"- Status: `{summary['run_status']}`",
            f"- Execution requested: `{summary['execution_requested']}`",
            "- Testnet execution included: `False`",
            "- Live trading authorized: `False`",
            "",
            frame.to_markdown(index=False),
            "",
        ]
    )
