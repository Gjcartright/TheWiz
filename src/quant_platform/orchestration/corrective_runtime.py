"""Canonical runtime contract for corrective background processes."""

from __future__ import annotations

import errno
import json
import os
import plistlib
import re
import stat
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from quant_platform.active_pipeline import CommandResult

RUNTIME_CONTRACT_SCHEMA_VERSION = "thewiz.scheduler_runtime_contract.v2"
CAPABILITY_PROFILE_SCHEMA_VERSION = "thewiz.scheduler_capability_profile.v1"
RUNTIME_TEMP_DIRNAME = ".runtime_tmp"
RUNTIME_TEMP_ENV_NAMES = ("TMPDIR", "TMP", "TEMP")
RUNTIME_AGENT_DIRNAME = ".runtime_agents"
RUNTIME_TIMEZONE = "America/New_York"
SCHEDULER_BOOTSTRAP_MODULE = (
    "quant_platform.orchestration.corrective_scheduler_bootstrap"
)
SCHEDULER_LOG_DIRECTORY = "reports/active/schedule_logs"
SCHEDULER_LOG_MAX_BYTES = 10 * 1024**2
SCHEDULER_LOG_RETENTION_DAYS = 14
SCHEDULER_PROBE_TIMEOUT_SECONDS = 15.0
_GIT_OBJECT_ID_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_MAX_GIT_CONTROL_BYTES = 16 * 1024**2


@dataclass(frozen=True)
class SchedulerCapabilityProfile:
    """Deny-by-default capabilities available to one scheduler process."""

    name: str
    external_network_allowed: bool
    public_hyperliquid_allowed: bool
    wizard_api_allowed: bool
    wizard_credit_spend_allowed: bool
    keychain_allowed: bool
    order_adapter_allowed: bool
    order_submission_allowed: bool


SCHEDULER_CAPABILITY_PROFILES = {
    "NO_EXTERNAL_NO_ORDER": SchedulerCapabilityProfile(
        name="NO_EXTERNAL_NO_ORDER",
        external_network_allowed=False,
        public_hyperliquid_allowed=False,
        wizard_api_allowed=False,
        wizard_credit_spend_allowed=False,
        keychain_allowed=False,
        order_adapter_allowed=False,
        order_submission_allowed=False,
    ),
    "PUBLIC_L2_ONLY": SchedulerCapabilityProfile(
        name="PUBLIC_L2_ONLY",
        external_network_allowed=True,
        public_hyperliquid_allowed=True,
        wizard_api_allowed=False,
        wizard_credit_spend_allowed=False,
        keychain_allowed=False,
        order_adapter_allowed=False,
        order_submission_allowed=False,
    ),
    "WIZARD_EXTERNAL_RESEARCH": SchedulerCapabilityProfile(
        name="WIZARD_EXTERNAL_RESEARCH",
        external_network_allowed=True,
        public_hyperliquid_allowed=True,
        wizard_api_allowed=True,
        wizard_credit_spend_allowed=True,
        keychain_allowed=False,
        order_adapter_allowed=False,
        order_submission_allowed=False,
    ),
}


@dataclass(frozen=True)
class SchedulerContract:
    """Static identity and schedule for one governed LaunchAgent."""

    key: str
    label: str
    module: str
    action: str
    capability_profile: str
    stdout_name: str
    stderr_name: str
    interval_seconds: int | None = None
    calendar_hour: int | None = None
    calendar_minute: int | None = None
    process_type: str | None = None
    low_priority_io: bool = False


SCHEDULER_CONTRACTS = (
    SchedulerContract(
        key="daily_research",
        label="com.thewiz.corrective-research-daily",
        module="quant_platform.orchestration.corrective_daily_scheduler",
        action="--execute",
        capability_profile="WIZARD_EXTERNAL_RESEARCH",
        stdout_name="daily.stdout.log",
        stderr_name="daily.stderr.log",
        calendar_hour=6,
        calendar_minute=15,
    ),
    SchedulerContract(
        key="hyperliquid_l2",
        label="com.thewiz.corrective-l2-cadence",
        module="quant_platform.orchestration.corrective_l2_scheduler",
        action="--capture",
        capability_profile="PUBLIC_L2_ONLY",
        stdout_name="l2.stdout.log",
        stderr_name="l2.stderr.log",
        interval_seconds=300,
    ),
    SchedulerContract(
        key="wizard_proof",
        label="com.thewiz.corrective-wizard-proof",
        module="quant_platform.orchestration.corrective_wizard_proof_launcher",
        action="--execute",
        capability_profile="WIZARD_EXTERNAL_RESEARCH",
        stdout_name="wizard_proof.stdout.log",
        stderr_name="wizard_proof.stderr.log",
        interval_seconds=600,
        process_type="Background",
        low_priority_io=True,
    ),
)


def scheduler_contract(key: str) -> SchedulerContract:
    for contract in SCHEDULER_CONTRACTS:
        if contract.key == key:
            return contract
    raise KeyError(f"unknown scheduler contract: {key}")


def scheduler_capability_profile(name: str) -> SchedulerCapabilityProfile:
    try:
        return SCHEDULER_CAPABILITY_PROFILES[name]
    except KeyError as exc:
        raise KeyError(f"unknown scheduler capability profile: {name}") from exc


def runtime_temp_directory(root: Path) -> Path:
    """Return the repository-local temp directory without following unsafe links."""

    resolved_root = root.resolve()
    candidate = root / RUNTIME_TEMP_DIRNAME
    if candidate.is_symlink():
        raise ValueError("runtime temp directory must not be a symbolic link")
    resolved_candidate = candidate.resolve(strict=False)
    if resolved_candidate.parent != resolved_root:
        raise ValueError("runtime temp directory must remain inside the repository root")
    return candidate


def ensure_runtime_temp_directory(root: Path) -> Path:
    """Create the private workspace temp directory used by LaunchAgents."""

    path = runtime_temp_directory(root)
    path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise NotADirectoryError(f"runtime temp path is not a directory: {path}")
    path.chmod(0o700)
    return path


def scheduler_python_path(root: Path) -> Path:
    """Return the interpreter owned by the uv-locked project environment."""

    return root / ".venv" / "bin" / "python3"


def scheduler_log_directory(root: Path) -> Path:
    return root / SCHEDULER_LOG_DIRECTORY


def ensure_scheduler_log_directory(root: Path) -> Path:
    path = scheduler_log_directory(root)
    if path.is_symlink():
        raise ValueError("scheduler log directory must not be a symbolic link")
    path.mkdir(parents=True, exist_ok=True)
    path.chmod(0o700)
    return path


def scheduler_runtime_fingerprints(root: Path) -> dict[str, str]:
    """Hash source, configuration, dependency, and interpreter identities."""

    source_paths = sorted(
        path
        for source_root in (root / "src", root / "scripts")
        if source_root.is_dir()
        for path in source_root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    )
    config_paths = (
        sorted(path for path in (root / "config").rglob("*") if path.is_file())
        if (root / "config").is_dir()
        else []
    )
    dependency_paths = [root / "pyproject.toml", root / "uv.lock"]
    python = scheduler_python_path(root)
    runtime_paths = [
        python,
        root / ".venv" / "pyvenv.cfg",
        *_installed_distribution_identity_paths(root),
    ]
    source_tree_sha256 = _path_set_fingerprint(root, source_paths)
    repository_head = _repository_head(root)
    return {
        "repository_head": repository_head,
        "source_tree_sha256": source_tree_sha256,
        "source_fingerprint_sha256": sha256(
            _canonical_json(
                {
                    "repository_head": repository_head,
                    "source_tree_sha256": source_tree_sha256,
                }
            ).encode("utf-8")
        ).hexdigest(),
        "configuration_fingerprint_sha256": _path_set_fingerprint(root, config_paths),
        "dependency_fingerprint_sha256": _path_set_fingerprint(root, dependency_paths),
        "interpreter_fingerprint_sha256": _path_set_fingerprint(root, runtime_paths),
    }


def scheduler_runtime_contract(
    root: Path,
    *,
    contract: SchedulerContract,
    interval_seconds: int | None = None,
    calendar_hour: int | None = None,
    calendar_minute: int | None = None,
) -> dict[str, Any]:
    """Build the single reviewed contract consumed by installers and readiness."""

    schedule = _resolved_schedule(
        contract,
        interval_seconds=interval_seconds,
        calendar_hour=calendar_hour,
        calendar_minute=calendar_minute,
    )
    fingerprints = scheduler_runtime_fingerprints(root)
    profile = scheduler_capability_profile(contract.capability_profile)
    profile_material = {
        "schema_version": CAPABILITY_PROFILE_SCHEMA_VERSION,
        **asdict(profile),
    }
    profile_sha256 = sha256(
        _canonical_json(profile_material).encode("utf-8")
    ).hexdigest()
    material: dict[str, Any] = {
        "schema_version": RUNTIME_CONTRACT_SCHEMA_VERSION,
        "root": str(root),
        "scheduler": asdict(contract),
        "capability_profile": profile_material,
        "capability_profile_sha256": profile_sha256,
        "schedule": schedule,
        "python": str(scheduler_python_path(root)),
        "resolved_python": str(scheduler_python_path(root).resolve(strict=False)),
        "python_executable_sha256": (
            _file_sha256(scheduler_python_path(root).resolve())
            if scheduler_python_path(root).resolve().is_file()
            else ""
        ),
        "working_directory": str(root),
        "runtime_temp": str(runtime_temp_directory(root)),
        "log_directory": str(scheduler_log_directory(root)),
        "timezone": RUNTIME_TIMEZONE,
        "log_policy": {
            "max_bytes": SCHEDULER_LOG_MAX_BYTES,
            "retention_days": SCHEDULER_LOG_RETENTION_DAYS,
            "directory_mode": "0700",
            "file_mode": "0600",
            "secret_content_allowed": False,
        },
        **fingerprints,
    }
    material["schedule_fingerprint_sha256"] = sha256(
        _canonical_json({"scheduler": asdict(contract), "schedule": schedule}).encode(
            "utf-8"
        )
    ).hexdigest()
    material["runtime_contract_sha256"] = sha256(
        _canonical_json(material).encode("utf-8")
    ).hexdigest()
    return material


def launch_agent_runtime_environment(
    root: Path,
    *,
    contract: SchedulerContract | None = None,
    interval_seconds: int | None = None,
    calendar_hour: int | None = None,
    calendar_minute: int | None = None,
) -> dict[str, str]:
    """Build the exact environment required by corrective LaunchAgents."""

    runtime_temp = str(runtime_temp_directory(root))
    environment = {
        "PYTHONPATH": str(root / "src"),
        **{name: runtime_temp for name in RUNTIME_TEMP_ENV_NAMES},
        "TZ": RUNTIME_TIMEZONE,
    }
    if contract is None:
        return environment
    runtime = scheduler_runtime_contract(
        root,
        contract=contract,
        interval_seconds=interval_seconds,
        calendar_hour=calendar_hour,
        calendar_minute=calendar_minute,
    )
    environment.update(
        {
            "THEWIZ_SCHEDULER_RUNTIME": "launchd",
            "THEWIZ_SCHEDULER_PROFILE": contract.capability_profile,
            "THEWIZ_SCHEDULER_PROFILE_SHA256": runtime[
                "capability_profile_sha256"
            ],
            "THEWIZ_RUNTIME_CONTRACT_SCHEMA": RUNTIME_CONTRACT_SCHEMA_VERSION,
            "THEWIZ_RUNTIME_CONTRACT_SHA256": runtime["runtime_contract_sha256"],
            "THEWIZ_SOURCE_FINGERPRINT_SHA256": runtime[
                "source_fingerprint_sha256"
            ],
            "THEWIZ_CONFIGURATION_FINGERPRINT_SHA256": runtime[
                "configuration_fingerprint_sha256"
            ],
            "THEWIZ_DEPENDENCY_FINGERPRINT_SHA256": runtime[
                "dependency_fingerprint_sha256"
            ],
            "THEWIZ_INTERPRETER_FINGERPRINT_SHA256": runtime[
                "interpreter_fingerprint_sha256"
            ],
            "THEWIZ_SCHEDULE_FINGERPRINT_SHA256": runtime[
                "schedule_fingerprint_sha256"
            ],
        }
    )
    return environment


def scheduler_launch_agent_payload(
    root: Path,
    *,
    contract: SchedulerContract,
    interval_seconds: int | None = None,
    calendar_hour: int | None = None,
    calendar_minute: int | None = None,
) -> dict[str, Any]:
    runtime = scheduler_runtime_contract(
        root,
        contract=contract,
        interval_seconds=interval_seconds,
        calendar_hour=calendar_hour,
        calendar_minute=calendar_minute,
    )
    environment = launch_agent_runtime_environment(
        root,
        contract=contract,
        interval_seconds=interval_seconds,
        calendar_hour=calendar_hour,
        calendar_minute=calendar_minute,
    )
    logs = scheduler_log_directory(root)
    payload: dict[str, Any] = {
        "Label": contract.label,
        "ProgramArguments": [
            runtime["python"],
            "-m",
            SCHEDULER_BOOTSTRAP_MODULE,
            str(root.resolve()),
            contract.key,
        ],
        "WorkingDirectory": runtime["working_directory"],
        "EnvironmentVariables": environment,
        "RunAtLoad": False,
        "StandardOutPath": str(logs / contract.stdout_name),
        "StandardErrorPath": str(logs / contract.stderr_name),
    }
    schedule = runtime["schedule"]
    if schedule["kind"] == "interval":
        payload["StartInterval"] = schedule["interval_seconds"]
    else:
        payload["StartCalendarInterval"] = {
            "Hour": schedule["hour"],
            "Minute": schedule["minute"],
        }
    if contract.process_type:
        payload["ProcessType"] = contract.process_type
    if contract.low_priority_io:
        payload["LowPriorityIO"] = True
    return payload


def scheduler_launch_agent_plist(
    root: Path,
    *,
    contract: SchedulerContract,
    interval_seconds: int | None = None,
    calendar_hour: int | None = None,
    calendar_minute: int | None = None,
) -> str:
    payload = scheduler_launch_agent_payload(
        root,
        contract=contract,
        interval_seconds=interval_seconds,
        calendar_hour=calendar_hour,
        calendar_minute=calendar_minute,
    )
    return plistlib.dumps(payload, fmt=plistlib.FMT_XML, sort_keys=True).decode("utf-8")


def scheduler_run_identity(
    root: Path,
    *,
    contract: SchedulerContract,
    interval_seconds: int | None = None,
    calendar_hour: int | None = None,
    calendar_minute: int | None = None,
    environment: dict[str, str] | None = None,
    require_launchd: bool = False,
) -> dict[str, Any]:
    """Bind a run receipt to current source, dependency, config, and schedule state."""

    runtime = scheduler_runtime_contract(
        root,
        contract=contract,
        interval_seconds=interval_seconds,
        calendar_hour=calendar_hour,
        calendar_minute=calendar_minute,
    )
    environment = dict(os.environ if environment is None else environment)
    launchd = environment.get("THEWIZ_SCHEDULER_RUNTIME") == "launchd"
    expected = launch_agent_runtime_environment(
        root,
        contract=contract,
        interval_seconds=interval_seconds,
        calendar_hour=calendar_hour,
        calendar_minute=calendar_minute,
    )
    mismatches = [
        key
        for key, wanted in expected.items()
        if launchd and environment.get(key) != wanted
    ]
    if require_launchd and not launchd:
        mismatches.insert(0, "THEWIZ_SCHEDULER_RUNTIME")
    profile = scheduler_capability_profile(contract.capability_profile)
    return {
        "scheduler_key": contract.key,
        "trigger_provenance": "launchd" if launchd else "manual_or_test",
        "launchd_provenance_required": require_launchd,
        "capability_profile": profile.name,
        "capability_profile_sha256": runtime["capability_profile_sha256"],
        "external_network_allowed": profile.external_network_allowed,
        "public_hyperliquid_allowed": profile.public_hyperliquid_allowed,
        "wizard_api_allowed": profile.wizard_api_allowed,
        "wizard_credit_spend_allowed": profile.wizard_credit_spend_allowed,
        "keychain_allowed": profile.keychain_allowed,
        "order_adapter_allowed": profile.order_adapter_allowed,
        "order_submission_allowed": profile.order_submission_allowed,
        "runtime_contract_schema_version": RUNTIME_CONTRACT_SCHEMA_VERSION,
        "runtime_contract_sha256": runtime["runtime_contract_sha256"],
        "source_fingerprint_sha256": runtime["source_fingerprint_sha256"],
        "configuration_fingerprint_sha256": runtime[
            "configuration_fingerprint_sha256"
        ],
        "dependency_fingerprint_sha256": runtime["dependency_fingerprint_sha256"],
        "interpreter_fingerprint_sha256": runtime[
            "interpreter_fingerprint_sha256"
        ],
        "schedule_fingerprint_sha256": runtime["schedule_fingerprint_sha256"],
        "runtime_environment_valid": not mismatches,
        "runtime_environment_blockers": [
            (
                "scheduler_launchd_provenance_missing"
                if key == "THEWIZ_SCHEDULER_RUNTIME" and not launchd
                else f"scheduler_runtime_environment_mismatch:{key}"
            )
            for key in mismatches
        ],
    }


def build_canonical_scheduler_runtime_contract(
    *, root: Path, now_iso: str
) -> CommandResult:
    """Publish the reviewed static contract without installing or loading jobs."""

    from quant_platform.active_pipeline import CommandResult

    services = [
        scheduler_runtime_contract(root, contract=contract)
        for contract in SCHEDULER_CONTRACTS
    ]
    payload: dict[str, Any] = {
        "schema_version": RUNTIME_CONTRACT_SCHEMA_VERSION,
        "generated_at_utc": now_iso,
        "status": "PASS_CONTRACT_RENDERED",
        "services": services,
        "research_only": True,
        "order_submission_included": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    contract_material = dict(payload)
    contract_material.pop("generated_at_utc", None)
    payload["contract_id"] = "runtimecontract_" + sha256(
        _canonical_json(contract_material).encode("utf-8")
    ).hexdigest()[:20]
    receipt_id = "runtimecontractreceipt_" + sha256(
        _canonical_json(payload).encode("utf-8")
    ).hexdigest()[:20]
    payload["receipt_id"] = receipt_id
    active = root / "reports" / "active" / "scheduler_runtime_contract.json"
    immutable = (
        root / "data" / "research" / "scheduler_runtime_contracts" / f"{receipt_id}.json"
    )
    write_immutable_json(immutable, payload)
    atomic_write_text(active, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return CommandResult(
        paths={"active_contract": active, "immutable_contract": immutable},
        summary=payload,
    )


def workspace_launch_agent_path(root: Path, label: str) -> Path:
    if not label or "/" in label:
        raise ValueError("invalid LaunchAgent label")
    return root / RUNTIME_AGENT_DIRNAME / f"{label}.plist"


def write_launch_agent_plist(
    *,
    root: Path,
    label: str,
    payload: str,
    system_path: Path | None = None,
) -> dict[str, Any]:
    """Publish a workspace plist and mirror it to LaunchAgents when possible."""

    from quant_platform.orchestration.effect_authority import (
        require_file_publication_target,
    )

    workspace_path = workspace_launch_agent_path(root, label)
    system_path = system_path or (
        Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"
    )
    require_file_publication_target(
        root=root,
        target=workspace_path,
        publication_scope="scheduler_config",
    )
    require_file_publication_target(
        root=root,
        target=system_path,
        publication_scope="scheduler_config",
    )
    workspace_path.parent.mkdir(parents=True, exist_ok=True)
    workspace_path.parent.chmod(0o700)
    atomic_write_text(
        workspace_path,
        payload,
        publication_scope="scheduler_config",
    )

    system_installed = False
    system_blocker = ""
    system_install_mode = "blocked"
    try:
        system_path.parent.mkdir(parents=True, exist_ok=True)
        if system_path.is_file() and system_path.read_text(encoding="utf-8") == payload:
            system_install_mode = "already_current"
        else:
            atomic_write_text(
                system_path,
                payload,
                publication_scope="scheduler_config",
            )
            system_install_mode = "atomic_replace"
        system_installed = True
    except OSError as exc:
        system_blocker = (
            f"system_launch_agent_install_failed:{type(exc).__name__}:{exc.errno}"
        )
    installation_status = (
        "PERSISTED_NOT_LOADED"
        if system_installed
        else "WORKSPACE_RENDERED_SYSTEM_PERSISTENCE_BLOCKED"
    )
    return {
        "status": installation_status,
        "workspace_plist": workspace_path,
        "workspace_generation_status": "PASS",
        "system_plist": system_path,
        "system_persistence_installed": system_installed,
        "system_persistence_status": "PASS" if system_installed else "BLOCKED",
        "system_install_mode": system_install_mode,
        "system_persistence_blocker": system_blocker,
        "launchd_load_status": "NOT_ATTEMPTED",
        "first_execution_status": "NOT_OBSERVED",
    }


def atomic_write_text(
    path: Path,
    payload: str,
    *,
    encoding: str = "utf-8",
    errors: str = "strict",
    mode: int = 0o600,
    publication_scope: str | None = None,
    publication_operation: str | None = None,
) -> None:
    """Durably replace one file using a unique same-directory temporary."""

    atomic_write_bytes(
        path,
        payload.encode(encoding, errors),
        mode=mode,
        publication_scope=publication_scope,
        publication_operation=publication_operation,
    )


def atomic_write_bytes(
    path: Path,
    payload: bytes,
    *,
    mode: int = 0o600,
    publication_scope: str | None = None,
    publication_operation: str | None = None,
) -> None:
    """Durably replace one binary artifact through the governed store boundary."""

    from quant_platform.orchestration.corrective_scheduler_lock import (
        governed_root_for_path,
        publication_lease_for_path,
        validate_current_publication_lease,
    )

    publication_scope = _effective_publication_scope(publication_scope)
    with publication_lease_for_path(path, scope=publication_scope) as lease:
        governed_root = governed_root_for_path(path)
        if lease is not None and governed_root is not None:
            validate_current_publication_lease(
                governed_root,
                target=path,
                required_scope=publication_scope,
            )
            _atomic_write_governed_bytes(
                root=governed_root,
                path=path,
                payload=payload,
                mode=mode,
                publication_scope=publication_scope,
                publication_operation=publication_operation,
            )
            return
        _atomic_write_unmanaged_bytes(
            path=path,
            payload=payload,
            mode=mode,
            publication_scope=publication_scope,
            publication_operation=publication_operation,
        )


def atomic_copy_file(
    source: Path,
    target: Path,
    *,
    mode: int | None = None,
    publication_scope: str | None = None,
    immutable: bool = False,
) -> None:
    """Copy one stable regular file through the governed publication boundary."""

    source = Path(os.path.abspath(source))
    target = Path(os.path.abspath(target))
    if source == target:
        raise ValueError("atomic copy requires distinct source and target")
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(source, flags)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise ValueError("atomic copy source must not be a symbolic link") from exc
        raise
    descriptor_open = True
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("atomic copy source must be a regular file")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor_open = False
            payload = handle.read()
            after = os.fstat(handle.fileno())
    finally:
        if descriptor_open:
            os.close(descriptor)
    if _file_identity_changed(before, after):
        raise ValueError("atomic copy source changed while being read")
    try:
        path_identity = source.lstat()
    except FileNotFoundError as exc:
        raise ValueError("atomic copy source disappeared after being read") from exc
    if (
        stat.S_ISLNK(path_identity.st_mode)
        or path_identity.st_dev != before.st_dev
        or path_identity.st_ino != before.st_ino
    ):
        raise ValueError("atomic copy source identity changed while being read")
    target_mode = stat.S_IMODE(before.st_mode) if mode is None else mode
    if immutable:
        target_mode = (target_mode & ~0o222) | stat.S_IRUSR
        write_immutable_bytes(
            target,
            payload,
            mode=target_mode,
            publication_scope=publication_scope,
        )
        return
    atomic_write_bytes(
        target,
        payload,
        mode=target_mode,
        publication_scope=publication_scope,
    )


def immutable_snapshot_copy(
    source: Path,
    directory: Path,
    *,
    artifact_name: str,
    publication_scope: str | None = None,
) -> Path:
    """Copy evidence under a semantic name so generic basenames cannot collide."""

    name = artifact_name.strip()
    if (
        not name
        or name in {".", ".."}
        or Path(name).name != name
        or any(token in name for token in ("/", "\\", "\x00"))
    ):
        raise ValueError("snapshot artifact name must be one safe path component")
    source = Path(source)
    suffix = "".join(source.suffixes) or ".dat"
    target = Path(directory) / f"{name}{suffix}"
    atomic_copy_file(
        source,
        target,
        publication_scope=publication_scope,
        immutable=True,
    )
    return target


def write_immutable_json(
    path: Path,
    payload: dict[str, Any],
    *,
    publication_scope: str | None = None,
) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    write_immutable_bytes(
        path,
        encoded,
        publication_scope=publication_scope,
    )


def write_immutable_bytes(
    path: Path,
    payload: bytes,
    *,
    mode: int = 0o400,
    publication_scope: str | None = None,
) -> None:
    """Create or validate one immutable binary artifact through the same fence."""

    from quant_platform.orchestration.corrective_scheduler_lock import (
        governed_root_for_path,
        publication_lease_for_path,
        validate_current_publication_lease,
    )

    publication_scope = _effective_publication_scope(publication_scope)
    with publication_lease_for_path(path, scope=publication_scope) as lease:
        governed_root = governed_root_for_path(path)
        if lease is not None and governed_root is not None:
            validate_current_publication_lease(
                governed_root,
                target=path,
                required_scope=publication_scope,
            )
            existing = _read_governed_bytes(governed_root, path)
            if existing is not None:
                if existing != payload:
                    raise ValueError(f"immutable artifact collision: {path}")
                if path.stat().st_mode & 0o222:
                    raise ValueError(f"immutable artifact is writable: {path}")
                return
        elif path.exists():
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"immutable artifact target is not a regular file: {path}")
            if path.read_bytes() != payload:
                raise ValueError(f"immutable artifact collision: {path}")
            if path.stat().st_mode & 0o222:
                raise ValueError(f"immutable artifact is writable: {path}")
            return
        atomic_write_bytes(
            path,
            payload,
            mode=mode,
            publication_scope=publication_scope,
            publication_operation="immutable_create",
        )


def create_exclusive_bytes(
    path: Path,
    payload: bytes,
    *,
    mode: int = 0o600,
    publication_scope: str | None = None,
) -> None:
    """Create one new artifact and fail if any target already exists."""

    atomic_write_bytes(
        path,
        payload,
        mode=mode,
        publication_scope=publication_scope,
        publication_operation="immutable_create",
    )


def create_exclusive_text(
    path: Path,
    payload: str,
    *,
    encoding: str = "utf-8",
    errors: str = "strict",
    mode: int = 0o600,
    publication_scope: str | None = None,
) -> None:
    create_exclusive_bytes(
        path,
        payload.encode(encoding, errors),
        mode=mode,
        publication_scope=publication_scope,
    )


def create_exclusive_json(
    path: Path,
    payload: dict[str, Any],
    *,
    mode: int = 0o600,
    publication_scope: str | None = None,
) -> None:
    create_exclusive_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        mode=mode,
        publication_scope=publication_scope,
    )


def atomic_append_text(
    path: Path,
    payload: str,
    *,
    encoding: str = "utf-8",
    errors: str = "strict",
    mode: int = 0o600,
    publication_scope: str | None = None,
) -> None:
    """Append by atomically replacing the complete prior-plus-new byte stream."""

    from quant_platform.orchestration.corrective_scheduler_lock import (
        governed_root_for_path,
        publication_lease_for_path,
    )

    target = Path(os.path.abspath(path))
    scope = _effective_publication_scope(publication_scope)
    with publication_lease_for_path(target, scope=scope):
        root = governed_root_for_path(target)
        if root is not None:
            existing = _read_governed_bytes(root, target) or b""
        elif target.exists():
            identity = target.lstat()
            if not stat.S_ISREG(identity.st_mode) or identity.st_nlink != 1:
                raise ValueError("append target must be one regular file")
            existing = target.read_bytes()
        else:
            existing = b""
        atomic_write_bytes(
            target,
            existing + payload.encode(encoding, errors),
            mode=mode,
            publication_scope=scope,
        )


def promote_staged_file(
    staged: Path,
    target: Path,
    *,
    mode: int | None = None,
    publication_scope: str | None = None,
    immutable: bool = False,
) -> None:
    """Promote a same-directory staging file through the governed atomic writer."""

    staged = Path(os.path.abspath(staged))
    target = Path(os.path.abspath(target))
    if staged == target:
        raise ValueError("staged publication requires distinct source and target")
    identity = staged.lstat()
    if not stat.S_ISREG(identity.st_mode) or identity.st_nlink != 1:
        raise ValueError("staged publication source must be one regular file")
    payload = staged.read_bytes()
    if immutable:
        write_immutable_bytes(
            target,
            payload,
            publication_scope=publication_scope,
        )
    else:
        atomic_write_bytes(
            target,
            payload,
            mode=stat.S_IMODE(identity.st_mode) if mode is None else mode,
            publication_scope=publication_scope,
        )
    staged.unlink()


def promote_staged_directory(
    staged: Path,
    target: Path,
    *,
    publication_scope: str | None = None,
) -> None:
    """Atomically publish one complete directory under a manifest-bound lease."""

    from quant_platform.orchestration.corrective_scheduler_lock import (
        _open_or_create_directory,
        governed_root_for_path,
        publication_lease_for_path,
        validate_current_publication_lease,
    )
    from quant_platform.orchestration.effect_authority import (
        authorize_file_publication,
    )

    staged = Path(os.path.abspath(staged))
    target = Path(os.path.abspath(target))
    if staged == target or staged.parent != target.parent:
        raise ValueError("directory promotion requires distinct same-parent paths")
    scope = _effective_publication_scope(publication_scope)
    manifest = _directory_manifest_payload(staged)
    with publication_lease_for_path(target, scope=scope):
        root = governed_root_for_path(target)
        if root is None:
            if target.exists() or target.is_symlink():
                raise FileExistsError(f"directory publication target exists: {target}")
            os.replace(staged, target)
            _fsync_directory(target.parent)
            return
        root = root.resolve()
        relative_target = target.relative_to(root)
        relative_staged = staged.relative_to(root)
        if relative_target.parent != relative_staged.parent:
            raise ValueError("directory publication crosses governed parents")
        parent_descriptor = _open_or_create_directory(root, relative_target.parent)
        try:
            _assert_directory_binding(
                descriptor=parent_descriptor,
                path=root / relative_target.parent,
            )
            source_identity = os.stat(
                relative_staged.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if not stat.S_ISDIR(source_identity.st_mode):
                raise ValueError("directory publication source is not a directory")
            try:
                os.stat(
                    relative_target.name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                pass
            else:
                raise FileExistsError(
                    f"directory publication target exists: {target}"
                )
            validate_current_publication_lease(
                root,
                target=target,
                required_scope=scope,
            )
            authorize_file_publication(
                root=root,
                target=target,
                publication_scope=scope,
                operation="create",
                payload=manifest,
            )
            os.replace(
                relative_staged.name,
                relative_target.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
            )
            os.fsync(parent_descriptor)
            _assert_directory_binding(
                descriptor=parent_descriptor,
                path=root / relative_target.parent,
            )
        finally:
            os.close(parent_descriptor)


def _directory_manifest_payload(path: Path) -> bytes:
    identity = path.lstat()
    if not stat.S_ISDIR(identity.st_mode):
        raise ValueError("directory publication source is not a directory")
    rows: list[dict[str, object]] = []
    for candidate in sorted(path.rglob("*"), key=lambda item: item.as_posix()):
        candidate_identity = candidate.lstat()
        relative = candidate.relative_to(path).as_posix()
        if stat.S_ISLNK(candidate_identity.st_mode):
            raise ValueError(f"directory publication symlink denied: {relative}")
        if stat.S_ISDIR(candidate_identity.st_mode):
            rows.append({"path": relative, "type": "directory"})
            continue
        if not stat.S_ISREG(candidate_identity.st_mode):
            raise ValueError(f"directory publication special file denied: {relative}")
        rows.append(
            {
                "path": relative,
                "type": "file",
                "size": candidate_identity.st_size,
                "sha256": sha256(candidate.read_bytes()).hexdigest(),
            }
        )
    return json.dumps(rows, separators=(",", ":"), sort_keys=True).encode()


def atomic_write_csv(
    frame: Any,
    path: Path,
    *args: Any,
    publication_scope: str | None = None,
    **kwargs: Any,
) -> None:
    """Serialize a pandas-compatible object and publish it through one lease."""

    from quant_platform.orchestration.corrective_scheduler_lock import (
        publication_lease_for_path,
    )

    target = Path(os.path.abspath(path))
    scope = _effective_publication_scope(publication_scope)
    with publication_lease_for_path(target, scope=scope):
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.parent / f".{uuid4().hex}.{target.name}"
        mode = str(kwargs.get("mode", "w"))
        try:
            if mode.startswith("a") and target.exists():
                identity = target.lstat()
                if not stat.S_ISREG(identity.st_mode) or identity.st_nlink != 1:
                    raise ValueError("CSV append target must be one regular file")
                temporary.write_bytes(target.read_bytes())
            frame.to_csv(temporary, *args, **kwargs)
            promote_staged_file(
                temporary,
                target,
                publication_scope=scope,
            )
        finally:
            temporary.unlink(missing_ok=True)


def atomic_write_parquet(
    frame: Any,
    path: Path,
    *args: Any,
    publication_scope: str | None = None,
    **kwargs: Any,
) -> None:
    """Serialize a pandas-compatible object to Parquet and publish atomically."""

    from quant_platform.orchestration.corrective_scheduler_lock import (
        publication_lease_for_path,
    )

    target = Path(os.path.abspath(path))
    scope = _effective_publication_scope(publication_scope)
    with publication_lease_for_path(target, scope=scope):
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.parent / f".{uuid4().hex}.{target.name}"
        try:
            frame.to_parquet(temporary, *args, **kwargs)
            promote_staged_file(
                temporary,
                target,
                publication_scope=scope,
            )
        finally:
            temporary.unlink(missing_ok=True)


def _effective_publication_scope(publication_scope: str | None) -> str:
    if publication_scope is not None:
        scope = publication_scope.strip()
        if not scope:
            raise ValueError("publication scope is required")
        return scope
    from quant_platform.orchestration.corrective_scheduler_lock import (
        current_publication_lease,
    )

    lease = current_publication_lease()
    return lease.scope if lease is not None else "research"


def _file_identity_changed(before: os.stat_result, after: os.stat_result) -> bool:
    return any(
        getattr(before, field) != getattr(after, field)
        for field in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    )


def _atomic_write_unmanaged_bytes(
    *,
    path: Path,
    payload: bytes,
    mode: int,
    publication_scope: str,
    publication_operation: str | None,
) -> None:
    from quant_platform.orchestration.effect_authority import (
        EffectAuthorityError,
        authorize_file_publication,
        current_publication_authority,
        require_file_publication_target,
    )

    authority = current_publication_authority()
    if authority is None:
        raise EffectAuthorityError("unmanaged_publication_authority_session_missing")
    path = Path(os.path.abspath(path))
    if path.is_symlink():
        raise ValueError("unmanaged publication target symlink denied")
    require_file_publication_target(
        root=authority.root,
        target=path,
        publication_scope=publication_scope,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid4().hex}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    descriptor = os.open(temporary, flags, mode)
    descriptor_open = True
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor_open = False
            _write_all(handle, payload)
            handle.flush()
            os.fsync(handle.fileno())
        target_exists = path.exists()
        operation = publication_operation or (
            "replace" if target_exists else "create"
        )
        if operation not in {"create", "replace", "immutable_create"}:
            raise ValueError("invalid publication operation")
        if operation == "immutable_create" and target_exists:
            raise FileExistsError(f"immutable publication target exists: {path}")
        authorize_file_publication(
            root=authority.root,
            target=path,
            publication_scope=publication_scope,
            operation=operation,
            payload=payload,
        )
        if operation == "immutable_create":
            os.link(temporary, path, follow_symlinks=False)
            temporary.unlink()
        else:
            os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        if descriptor_open:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise


def _atomic_write_governed_bytes(
    *,
    root: Path,
    path: Path,
    payload: bytes,
    mode: int,
    publication_scope: str,
    publication_operation: str | None,
) -> None:
    from quant_platform.orchestration.corrective_scheduler_lock import (
        _open_or_create_directory,
        validate_current_publication_lease,
    )
    from quant_platform.orchestration.effect_authority import (
        authorize_file_publication,
    )

    root = root.resolve()
    lexical_target = Path(os.path.abspath(path))
    try:
        relative = lexical_target.relative_to(root)
    except ValueError as exc:
        raise ValueError("governed publication target is outside the repository") from exc
    if not relative.name or ".." in relative.parts:
        raise ValueError("invalid governed publication target")
    parent_descriptor = _open_or_create_directory(root, relative.parent)
    temporary_name = f".{relative.name}.{uuid4().hex}.tmp"
    temporary_created = False
    try:
        _assert_directory_binding(
            descriptor=parent_descriptor,
            path=root / relative.parent,
        )
        _deny_symlink_at(parent_descriptor, relative.name)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(
            temporary_name,
            flags,
            mode,
            dir_fd=parent_descriptor,
        )
        temporary_created = True
        descriptor_open = True
        try:
            os.fchmod(descriptor, mode)
            with os.fdopen(descriptor, "wb") as handle:
                descriptor_open = False
                _write_all(handle, payload)
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            if descriptor_open:
                os.close(descriptor)
            raise
        validate_current_publication_lease(
            root,
            target=lexical_target,
            required_scope=publication_scope,
        )
        _assert_directory_binding(
            descriptor=parent_descriptor,
            path=root / relative.parent,
        )
        _deny_symlink_at(parent_descriptor, relative.name)
        try:
            os.stat(relative.name, dir_fd=parent_descriptor, follow_symlinks=False)
            target_exists = True
        except FileNotFoundError:
            target_exists = False
        operation = publication_operation or (
            "replace" if target_exists else "create"
        )
        if operation not in {"create", "replace", "immutable_create"}:
            raise ValueError("invalid publication operation")
        if operation == "immutable_create" and target_exists:
            raise FileExistsError(f"immutable publication target exists: {path}")
        authorize_file_publication(
            root=root,
            target=lexical_target,
            publication_scope=publication_scope,
            operation=operation,
            payload=payload,
        )
        if operation == "immutable_create":
            os.link(
                temporary_name,
                relative.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            os.unlink(temporary_name, dir_fd=parent_descriptor)
        else:
            os.replace(
                temporary_name,
                relative.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
            )
        temporary_created = False
        os.fsync(parent_descriptor)
        _assert_directory_binding(
            descriptor=parent_descriptor,
            path=root / relative.parent,
        )
    finally:
        if temporary_created:
            try:
                os.unlink(temporary_name, dir_fd=parent_descriptor)
            except FileNotFoundError:
                pass
        os.close(parent_descriptor)


def _write_all(handle: Any, payload: bytes) -> None:
    """Write every byte or fail before the temporary can be promoted."""

    view = memoryview(payload)
    offset = 0
    while offset < len(view):
        written = handle.write(view[offset:])
        remaining = len(view) - offset
        if not isinstance(written, int) or written <= 0 or written > remaining:
            raise OSError(errno.EIO, "atomic publication made invalid write progress")
        offset += written


def _read_governed_bytes(root: Path, path: Path) -> bytes | None:
    from quant_platform.orchestration.corrective_scheduler_lock import (
        _open_or_create_directory,
    )

    lexical_target = Path(os.path.abspath(path))
    try:
        relative = lexical_target.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("governed publication target is outside the repository") from exc
    parent_descriptor = _open_or_create_directory(root.resolve(), relative.parent)
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        _assert_directory_binding(
            descriptor=parent_descriptor,
            path=root.resolve() / relative.parent,
        )
        try:
            descriptor = os.open(relative.name, flags, dir_fd=parent_descriptor)
        except FileNotFoundError:
            return None
        with os.fdopen(descriptor, "rb") as handle:
            identity = os.fstat(handle.fileno())
            if not stat.S_ISREG(identity.st_mode) or identity.st_nlink != 1:
                raise ValueError("immutable artifact target is not a regular file")
            return handle.read()
    finally:
        os.close(parent_descriptor)


def _deny_symlink_at(directory_descriptor: int, name: str) -> None:
    try:
        identity = os.stat(
            name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return
    if stat.S_ISLNK(identity.st_mode):
        raise ValueError("governed publication target symlink denied")


def _assert_directory_binding(*, descriptor: int, path: Path) -> None:
    descriptor_identity = os.fstat(descriptor)
    try:
        path_identity = path.lstat()
    except FileNotFoundError as exc:
        raise ValueError("governed publication directory disappeared") from exc
    if (
        not stat.S_ISDIR(path_identity.st_mode)
        or path_identity.st_dev != descriptor_identity.st_dev
        or path_identity.st_ino != descriptor_identity.st_ino
    ):
        raise ValueError("governed publication directory identity changed")


def _resolved_schedule(
    contract: SchedulerContract,
    *,
    interval_seconds: int | None,
    calendar_hour: int | None,
    calendar_minute: int | None,
) -> dict[str, Any]:
    if contract.interval_seconds is not None:
        interval = interval_seconds or contract.interval_seconds
        if interval < 60:
            raise ValueError("scheduler interval must be at least 60 seconds")
        return {"kind": "interval", "interval_seconds": interval}
    hour = contract.calendar_hour if calendar_hour is None else calendar_hour
    minute = contract.calendar_minute if calendar_minute is None else calendar_minute
    if hour is None or minute is None or not 0 <= hour <= 23 or not 0 <= minute <= 59:
        raise ValueError("invalid scheduler calendar time")
    return {"kind": "calendar", "hour": hour, "minute": minute}


def _installed_distribution_identity_paths(root: Path) -> list[Path]:
    """Return bounded metadata paths that identify the locked environment."""

    site_packages_roots = sorted(
        path
        for path in (root / ".venv" / "lib").glob("python*/site-packages")
        if path.is_dir() and not path.is_symlink()
    )
    paths: set[Path] = set()
    for site_packages in site_packages_roots:
        paths.update(path for path in site_packages.glob("*.pth") if path.is_file())
        for pattern in (
            "*.dist-info/METADATA",
            "*.dist-info/INSTALLER",
            "*.dist-info/direct_url.json",
            "*.egg-info/PKG-INFO",
        ):
            paths.update(path for path in site_packages.glob(pattern) if path.is_file())
    return sorted(paths)


def _repository_head(root: Path) -> str:
    """Resolve Git HEAD without invoking Git or trusting an unbounded control file."""

    marker = root / ".git"
    try:
        if marker.is_dir() and not marker.is_symlink():
            git_directory = marker.resolve()
        elif marker.is_file() and not marker.is_symlink():
            marker_text = _read_bounded_text(marker, max_bytes=4_096)
            prefix = "gitdir: "
            if not marker_text.startswith(prefix):
                return "UNAVAILABLE"
            candidate = Path(marker_text[len(prefix) :].strip())
            if not candidate.is_absolute():
                candidate = marker.parent / candidate
            git_directory = candidate.resolve()
            if not git_directory.is_dir():
                return "UNAVAILABLE"
        else:
            return "UNVERSIONED"

        head_text = _read_bounded_text(git_directory / "HEAD", max_bytes=4_096).strip()
        if _GIT_OBJECT_ID_RE.fullmatch(head_text):
            return head_text
        prefix = "ref: "
        if not head_text.startswith(prefix):
            return "UNAVAILABLE"
        ref_name = head_text[len(prefix) :].strip()
        ref_parts = Path(ref_name).parts
        if (
            not ref_name.startswith("refs/")
            or not ref_parts
            or any(part in {"", ".", ".."} for part in ref_parts)
        ):
            return "UNAVAILABLE"
        ref_path = git_directory.joinpath(*ref_parts)
        if ref_path.is_file() and not ref_path.is_symlink():
            value = _read_bounded_text(ref_path, max_bytes=4_096).strip()
            return value if _GIT_OBJECT_ID_RE.fullmatch(value) else "UNAVAILABLE"
        packed_refs = git_directory / "packed-refs"
        if not packed_refs.is_file() or packed_refs.is_symlink():
            return "UNAVAILABLE"
        for line in _read_bounded_text(
            packed_refs,
            max_bytes=_MAX_GIT_CONTROL_BYTES,
        ).splitlines():
            if not line or line.startswith(("#", "^")):
                continue
            object_id, separator, packed_ref_name = line.partition(" ")
            if (
                separator
                and packed_ref_name == ref_name
                and _GIT_OBJECT_ID_RE.fullmatch(object_id)
            ):
                return object_id
    except (OSError, UnicodeDecodeError, ValueError):
        return "UNAVAILABLE"
    return "UNAVAILABLE"


def _read_bounded_text(path: Path, *, max_bytes: int) -> str:
    identity = path.lstat()
    if not stat.S_ISREG(identity.st_mode) or identity.st_size > max_bytes:
        raise ValueError("runtime identity control file is invalid")
    return path.read_text(encoding="utf-8")


def _path_set_fingerprint(root: Path, paths: list[Path]) -> str:
    rows: list[dict[str, Any]] = []
    for path in sorted(set(paths), key=str):
        relative = _relative(path, root)
        if path.is_symlink():
            resolved = path.resolve(strict=False)
            rows.append(
                {
                    "path": relative,
                    "kind": "symlink",
                    "link_target": os.readlink(path),
                    "link_target_sha256": sha256(
                        os.readlink(path).encode("utf-8")
                    ).hexdigest(),
                    "resolved_target": str(resolved),
                    "resolved_target_sha256": (
                        _file_sha256(resolved) if resolved.is_file() else ""
                    ),
                }
            )
        elif path.is_file():
            rows.append(
                {"path": relative, "kind": "file", "sha256": _file_sha256(path)}
            )
        else:
            rows.append({"path": relative, "kind": "missing", "sha256": ""})
    return sha256(_canonical_json(rows).encode("utf-8")).hexdigest()


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve(strict=False))


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
