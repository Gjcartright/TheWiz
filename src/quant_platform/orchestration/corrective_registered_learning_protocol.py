"""Prospective, immutable protocol registration for Stage 5 research."""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from hashlib import sha256
from importlib import metadata
from pathlib import Path
from typing import Any

import pandas as pd

from quant_platform.active_pipeline import CommandResult
from quant_platform.ml_filter import available_model_specs

ROOT = Path(__file__).resolve().parents[3]
CONFIG_SCHEMA = "thewiz.registered_stage5_protocol_config.v1"
RECEIPT_SCHEMA = "thewiz.registered_stage5_protocol.v1"
POINTER_SCHEMA = "thewiz.registered_stage5_protocol_pointer.v1"
CONFIG_PATH = Path("config/registered_stage5_protocol.json")
SUPPORT_POLICY_PATH = Path("config/registered_survivor_oos_support_policy.json")
REQUIRED_SOURCE_ROLES = {
    "agent_learning_governance",
    "registered_learning_orchestrator",
    "registered_stage4_contract",
    "registered_stage4_execution_validator",
    "registered_stage5_protocol_registry",
    "dataset_and_model_gate_pipeline",
    "exact_mode_schema",
    "model_training_and_selection",
    "rl_feature_construction",
    "rl_research_and_execution_simulator",
    "rl_acceptance",
    "rl_chronological_partitioning",
}
AUTHORITY_FIELDS = (
    "promotion_authority",
    "testnet_candidate_authority",
    "testnet_order_authority",
    "live_trading_authorized",
)


def build_registered_stage5_protocol(
    *, root: Path = ROOT, now: datetime | None = None
) -> CommandResult:
    """Freeze Stage 5 choices before a future Stage 4 cohort is observed."""

    registered_at = _as_utc(now)
    config_path = root / CONFIG_PATH
    support_path = root / SUPPORT_POLICY_PATH
    config = _load_config(config_path)
    if not support_path.is_file():
        raise FileNotFoundError(support_path)
    source_bindings = _source_bindings(root=root, config=config)
    material = {
        "schema_version": RECEIPT_SCHEMA,
        "protocol_version": str(config["protocol_version"]),
        "config_path": str(CONFIG_PATH),
        "config_sha256": _file_hash(config_path),
        "support_policy_path": str(SUPPORT_POLICY_PATH),
        "support_policy_sha256": _file_hash(support_path),
        "source_bindings": source_bindings,
        "runtime_bindings": _runtime_bindings(),
        "protocol_contract": config["protocol_contract"],
        "thresholds_changed_after_results": False,
        "research_only": True,
        "promotion_authority": False,
        "testnet_candidate_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    protocol_id = "stage5protocol_" + _payload_hash(material)[:20]
    receipt_path = (
        root / "data" / "research" / "registered_stage5_protocols" / f"{protocol_id}.json"
    )
    if receipt_path.is_file():
        receipt = _read_json(receipt_path)
        _validate_receipt_material(receipt=receipt, material=material)
    else:
        receipt = {
            **material,
            "protocol_id": protocol_id,
            "registered_at_utc": registered_at.isoformat(),
        }
        receipt["receipt_sha256"] = _payload_hash(receipt)
        _write_immutable_json(receipt, receipt_path)

    pointer = {
        "schema_version": POINTER_SCHEMA,
        "status": "PASS_PROSPECTIVE_PROTOCOL_REGISTERED",
        "protocol_id": protocol_id,
        "registered_at_utc": receipt["registered_at_utc"],
        "protocol_receipt_path": _relative(receipt_path, root),
        "protocol_receipt_sha256": _file_hash(receipt_path),
        "config_sha256": receipt["config_sha256"],
        "support_policy_sha256": receipt["support_policy_sha256"],
        "thresholds_changed_after_results": False,
        "research_only": True,
        "promotion_authority": False,
        "testnet_candidate_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    pointer_path = root / "reports" / "active" / "registered_stage5_protocol.json"
    _atomic_json(pointer, pointer_path)
    return CommandResult(
        paths={"protocol_receipt": receipt_path, "active_pointer": pointer_path},
        summary={
            "status": pointer["status"],
            "protocol_id": protocol_id,
            "registered_at_utc": receipt["registered_at_utc"],
            "source_bindings": len(source_bindings),
            "research_only": True,
            "promotion_authority": False,
            "testnet_candidate_authority": False,
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        },
    )


def validate_registered_stage5_protocol(
    *, root: Path = ROOT, stage4_completed_at_utc: str = ""
) -> tuple[Path, dict[str, Any]]:
    """Validate identity, current implementation binding, and chronology."""

    pointer_path = root / "reports" / "active" / "registered_stage5_protocol.json"
    pointer = _read_json(pointer_path)
    if (
        pointer.get("schema_version") != POINTER_SCHEMA
        or pointer.get("status") != "PASS_PROSPECTIVE_PROTOCOL_REGISTERED"
    ):
        raise ValueError("registered Stage 5 protocol pointer is missing or blocked")
    _assert_no_authority(pointer, "registered Stage 5 protocol pointer")
    receipt_path = _safe_relative_path(
        root, str(pointer.get("protocol_receipt_path", ""))
    )
    expected_parent = (root / "data" / "research" / "registered_stage5_protocols").resolve()
    try:
        receipt_path.resolve().relative_to(expected_parent)
    except (OSError, ValueError) as exc:
        raise ValueError("registered Stage 5 protocol receipt path is unsafe") from exc
    if not receipt_path.is_file() or _file_hash(receipt_path) != str(
        pointer.get("protocol_receipt_sha256", "")
    ):
        raise ValueError("registered Stage 5 protocol pointer hash mismatch")

    receipt = _read_json(receipt_path)
    supplied_hash = str(receipt.pop("receipt_sha256", ""))
    if supplied_hash != _payload_hash(receipt):
        raise ValueError("registered Stage 5 protocol receipt hash mismatch")
    receipt["receipt_sha256"] = supplied_hash
    if (
        receipt.get("schema_version") != RECEIPT_SCHEMA
        or receipt.get("protocol_id") != pointer.get("protocol_id")
        or receipt.get("thresholds_changed_after_results") is not False
        or receipt.get("research_only") is not True
    ):
        raise ValueError("registered Stage 5 protocol receipt contract mismatch")
    _assert_no_authority(receipt, "registered Stage 5 protocol receipt")

    config = _load_config(root / CONFIG_PATH)
    observed_bindings = _source_bindings(root=root, config=config)
    current_checks = {
        "config_path": str(CONFIG_PATH),
        "config_sha256": _file_hash(root / CONFIG_PATH),
        "support_policy_path": str(SUPPORT_POLICY_PATH),
        "support_policy_sha256": _file_hash(root / SUPPORT_POLICY_PATH),
        "source_bindings": observed_bindings,
        "runtime_bindings": _runtime_bindings(),
        "protocol_contract": config["protocol_contract"],
    }
    for field, value in current_checks.items():
        if receipt.get(field) != value:
            raise ValueError(f"registered Stage 5 protocol current binding mismatch: {field}")
    if pointer.get("config_sha256") != receipt.get("config_sha256") or pointer.get(
        "support_policy_sha256"
    ) != receipt.get("support_policy_sha256"):
        raise ValueError("registered Stage 5 protocol pointer policy binding mismatch")

    registered_at = pd.to_datetime(receipt.get("registered_at_utc"), utc=True, errors="coerce")
    if pd.isna(registered_at):
        raise ValueError("registered Stage 5 protocol timestamp is invalid")
    if stage4_completed_at_utc:
        completed_at = pd.to_datetime(stage4_completed_at_utc, utc=True, errors="coerce")
        if pd.isna(completed_at):
            raise ValueError("registered Stage 4 completion timestamp is invalid")
        if registered_at > completed_at:
            raise ValueError("registered Stage 5 protocol was created after Stage 4 completion")
    return receipt_path, receipt


def _load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = _read_json(path)
    if (
        payload.get("schema_version") != CONFIG_SCHEMA
        or not str(payload.get("protocol_version", "")).strip()
        or payload.get("thresholds_changed_after_results") is not False
        or not isinstance(payload.get("protocol_contract"), dict)
    ):
        raise ValueError("registered Stage 5 protocol config is invalid")
    sources = payload.get("source_contracts")
    if not isinstance(sources, list):
        raise TypeError("registered Stage 5 protocol source contracts are missing")
    roles = {str(row.get("role", "")) for row in sources if isinstance(row, dict)}
    if roles != REQUIRED_SOURCE_ROLES or len(sources) != len(REQUIRED_SOURCE_ROLES):
        raise ValueError("registered Stage 5 protocol source roles are incomplete")
    _assert_no_authority(payload, "registered Stage 5 protocol config")
    return payload


def _source_bindings(*, root: Path, config: dict[str, Any]) -> list[dict[str, str]]:
    rows = []
    for source in config["source_contracts"]:
        role = str(source.get("role", ""))
        relative = str(source.get("path", ""))
        path = _safe_relative_path(root, relative)
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append({"role": role, "path": relative, "sha256": _file_hash(path)})
    return sorted(rows, key=lambda row: row["role"])


def _runtime_bindings() -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "pandas", "scikit-learn", "xgboost", "lightgbm"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = "NOT_INSTALLED"
    return {
        "python": ".".join(str(value) for value in sys.version_info[:3]),
        "packages": packages,
        "available_model_specs": [
            {
                "name": spec.name,
                "family": spec.family,
                "available": bool(spec.available),
                "unavailable_reason": str(spec.unavailable_reason),
            }
            for spec in available_model_specs()
        ],
    }


def _validate_receipt_material(
    *, receipt: dict[str, Any], material: dict[str, Any]
) -> None:
    supplied_hash = str(receipt.get("receipt_sha256", ""))
    core = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if supplied_hash != _payload_hash(core):
        raise ValueError("immutable registered Stage 5 protocol receipt hash mismatch")
    for key, value in material.items():
        if receipt.get(key) != value:
            raise ValueError(f"immutable registered Stage 5 protocol mismatch: {key}")
    expected_id = "stage5protocol_" + _payload_hash(material)[:20]
    if receipt.get("protocol_id") != expected_id:
        raise ValueError("immutable registered Stage 5 protocol identity mismatch")


def _assert_no_authority(payload: dict[str, Any], name: str) -> None:
    for field in AUTHORITY_FIELDS:
        if _truthy(payload.get(field)):
            raise ValueError(f"{name} unexpectedly grants {field}")


def _safe_relative_path(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if not relative or candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("registered Stage 5 protocol path is unsafe")
    return root / candidate


def _write_immutable_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        if _read_json(path) != payload:
            raise ValueError(f"immutable registered Stage 5 protocol conflict: {path}")
        return
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _payload_hash(payload: dict[str, Any]) -> str:
    return sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest() if path.is_file() else ""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _as_utc(value: datetime | None) -> datetime:
    current = value or datetime.now(UTC)
    if current.tzinfo is None:
        current = current.replace(tzinfo=UTC)
    return current.astimezone(UTC)
