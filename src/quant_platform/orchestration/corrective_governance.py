"""Immutable policy, hypothesis-lineage, and holdout governance receipts."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from copy import deepcopy
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd

from quant_platform.active_pipeline import CommandResult
from quant_platform.orchestration.corrective_runtime import promote_staged_file
from quant_platform.orchestration.evidence_integrity import sha256_file

ROOT = Path(__file__).resolve().parents[3]
ACCEPTANCE_SCHEMA = "thewiz.acceptance_policy.v1"
HOLDOUT_SCHEMA = "thewiz.research_holdout_policy.v1"
LEDGER_SCHEMA = "thewiz.semantic_hypothesis_ledger.v1"


class GovernanceError(ValueError):
    """Raised when policy or hypothesis lineage cannot be trusted."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def content_identity(value: Any, *, prefix: str = "") -> str:
    digest = sha256(canonical_json(value).encode("utf-8")).hexdigest()
    return f"{prefix}{digest}" if prefix else digest


def load_acceptance_policy(root: Path = ROOT) -> dict[str, Any]:
    path = root / "config" / "acceptance_policy_manifest.json"
    payload = _read_json(path)
    if payload.get("schema_version") != ACCEPTANCE_SCHEMA:
        raise GovernanceError("unsupported or missing acceptance policy schema")
    if bool(payload.get("safety_gates", {}).get("live_trading_authorized")):
        raise GovernanceError("acceptance policy must default live trading to false")
    return payload


def load_holdout_policy(root: Path = ROOT) -> dict[str, Any]:
    path = root / "config" / "research_holdout_policy.json"
    payload = _read_json(path)
    if payload.get("schema_version") != HOLDOUT_SCHEMA:
        raise GovernanceError("unsupported or missing holdout policy schema")
    assignment = payload.get("assignment", {})
    fractions = [
        float(assignment.get("development_fraction", math.nan)),
        float(assignment.get("validation_fraction", math.nan)),
        float(assignment.get("locked_final_holdout_fraction", math.nan)),
    ]
    if not all(math.isfinite(value) and 0 < value < 1 for value in fractions):
        raise GovernanceError("holdout fractions must be finite and strictly between zero and one")
    if not math.isclose(sum(fractions), 1.0, abs_tol=1e-12):
        raise GovernanceError("holdout fractions must sum to one")
    if bool(payload.get("live_trading_authorized")):
        raise GovernanceError("holdout policy cannot grant live trading authority")
    return payload


def build_acceptance_policy_receipt(
    *, root: Path = ROOT, now: datetime | None = None
) -> dict[str, Any]:
    policy = load_acceptance_policy(root)
    source_rows: list[dict[str, Any]] = []
    for contract in policy.get("source_contracts", []):
        relative = str(contract.get("path", ""))
        expected = str(contract.get("sha256", ""))
        path = _safe_project_file(root, relative)
        observed = sha256_file(path)
        source_rows.append(
            {
                "path": relative,
                "expected_sha256": expected,
                "observed_sha256": observed,
                "hash_match": observed == expected,
            }
        )
    policy_id = content_identity(policy, prefix="acceptance_")
    blockers = [f"source_contract_mismatch:{row['path']}" for row in source_rows if not row["hash_match"]]
    receipt = {
        "schema_version": "thewiz.acceptance_policy_receipt.v1",
        "generated_at_utc": _as_utc(now).isoformat(),
        "policy_id": policy_id,
        "policy_version": policy["policy_version"],
        "policy_path": "config/acceptance_policy_manifest.json",
        "policy_sha256": sha256_file(root / "config" / "acceptance_policy_manifest.json"),
        "source_contracts": source_rows,
        "source_contracts_match": not blockers,
        "status": "PASS" if not blockers else "BLOCKED",
        "blockers": blockers,
        # Matching policy source hashes proves file identity, not candidate performance.
        "promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    path = root / "reports" / "active" / "acceptance_policy_receipt.json"
    _atomic_json(receipt, path)
    return receipt


def run_gate_policy_tamper_tests(*, root: Path = ROOT) -> pd.DataFrame:
    policy = load_acceptance_policy(root)
    baseline_id = content_identity(policy, prefix="acceptance_")
    rows: list[dict[str, Any]] = []
    for field_path, original in _material_leaves(policy):
        mutated = deepcopy(policy)
        replacement = _mutated_value(original)
        _set_nested(mutated, field_path, replacement)
        mutated_id = content_identity(mutated, prefix="acceptance_")
        identity_changed = mutated_id != baseline_id
        rows.append(
            {
                "schema_version": "thewiz.gate_policy_tamper.v1",
                "field_path": ".".join(map(str, field_path)),
                "original_value": canonical_json(original),
                "mutated_value": canonical_json(replacement),
                "baseline_policy_id": baseline_id,
                "mutated_policy_id": mutated_id,
                "identity_changed": identity_changed,
                "comparability_invalidated": identity_changed,
                "promotion_authority_inherited": False,
                "status": "PASS" if identity_changed else "FAIL",
                "live_trading_authorized": False,
            }
        )
    frame = pd.DataFrame(rows)
    path = root / "reports" / "red_team" / "gate_policy_tamper_results.csv"
    _atomic_csv(frame, path)
    if frame.empty or not frame["status"].eq("PASS").all():
        raise GovernanceError("one or more material policy mutations preserved identity")
    return frame


def semantic_hypothesis_material(
    row: dict[str, Any], *, policy_id: str, holdout_policy_id: str
) -> dict[str, Any]:
    assets = sorted(
        value for value in {_asset(row.get("asset_x") or row.get("asset_a")), _asset(row.get("asset_y") or row.get("asset_b"))} if value
    )
    if len(assets) != 2:
        assets = sorted(_pair_assets(row.get("pair", "")))
    if len(assets) != 2:
        raise GovernanceError(f"cannot derive two assets from hypothesis: {row.get('pair', '')}")
    parameters = row.get("parameters")
    if not isinstance(parameters, dict):
        parameters = {
            "formula_authority": str(row.get("local_settings_authority", "local_formula_approximation")),
            "experiment_kind": str(row.get("experiment_kind", "exact_mode_replay")),
        }
    return {
        "assets": assets,
        "execution_venue": "hyperliquid",
        "timeframe": _normalize_token(row.get("wizard_timeframe") or row.get("timeframe") or "daily"),
        "exact_mode": _normalize_mode(row.get("exact_mode", "")),
        "orientation": _normalize_token(row.get("orientation", "forward")),
        "parameters": parameters,
        "acceptance_policy_id": policy_id,
        "holdout_policy_id": holdout_policy_id,
    }


def semantic_hypothesis_id(
    row: dict[str, Any], *, policy_id: str, holdout_policy_id: str
) -> str:
    return content_identity(
        semantic_hypothesis_material(row, policy_id=policy_id, holdout_policy_id=holdout_policy_id),
        prefix="hypothesis_",
    )


def build_hypothesis_ledger(
    *, root: Path = ROOT, source_path: Path | None = None, now: datetime | None = None
) -> dict[str, Any]:
    policy_receipt = build_acceptance_policy_receipt(root=root, now=now)
    if policy_receipt["status"] != "PASS":
        raise GovernanceError("acceptance policy receipt is blocked")
    holdout = load_holdout_policy(root)
    holdout_id = content_identity(holdout, prefix="holdout_")
    source_path = source_path or root / "reports" / "active" / "current_wizard_hyperliquid_failure_attribution.csv"
    source = pd.read_csv(source_path) if source_path.exists() else pd.DataFrame()
    ledger_path = root / "data" / "research" / "hypothesis_ledger.jsonl"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    existing = _read_ledger(ledger_path)
    chain_valid, chain_error = _validate_chain(existing)
    if not chain_valid:
        raise GovernanceError(f"hypothesis ledger chain invalid: {chain_error}")
    known_fingerprints = {str(row["evidence_fingerprint"]) for row in existing}
    family_attempts: dict[str, int] = {}
    aliases: dict[str, set[str]] = {}
    for record in existing:
        hypothesis_id = str(record["semantic_hypothesis_id"])
        family_attempts[hypothesis_id] = max(family_attempts.get(hypothesis_id, 0), int(record["attempt_number"]))
        aliases.setdefault(hypothesis_id, set()).add(str(record.get("source_experiment_id", "")))
    records = list(existing)
    previous_hash = str(records[-1]["record_hash"]) if records else "GENESIS"
    added = 0
    registered_at = _as_utc(now).isoformat()
    for row in source.to_dict("records"):
        hypothesis_id = semantic_hypothesis_id(
            row,
            policy_id=str(policy_receipt["policy_id"]),
            holdout_policy_id=holdout_id,
        )
        evidence_material = {
            "semantic_hypothesis_id": hypothesis_id,
            "failure_attribution_id": _clean(row.get("failure_attribution_id")),
            "evidence_path": _clean(row.get("evidence_path")),
            "observed_trades": _json_scalar(row.get("observed_trades")),
            "walkforward_trades": _json_scalar(row.get("walkforward_trades")),
            "first_blocker": _clean(row.get("first_blocker")),
            "consolidated_status": _clean(row.get("consolidated_status")),
        }
        fingerprint = content_identity(evidence_material, prefix="evidence_")
        aliases.setdefault(hypothesis_id, set()).add(_clean(row.get("experiment_id")))
        if fingerprint in known_fingerprints:
            continue
        attempt = family_attempts.get(hypothesis_id, 0) + 1
        record = {
            "schema_version": LEDGER_SCHEMA,
            "registered_at_utc": registered_at,
            "semantic_hypothesis_id": hypothesis_id,
            "semantic_material": semantic_hypothesis_material(
                row,
                policy_id=str(policy_receipt["policy_id"]),
                holdout_policy_id=holdout_id,
            ),
            "attempt_number": attempt,
            "source_experiment_id": _clean(row.get("experiment_id")),
            "source_run_id": _clean(row.get("failure_attribution_id")),
            "source_display_name": _clean(row.get("pair")),
            "evidence_fingerprint": fingerprint,
            "evidence_path": _clean(row.get("evidence_path")),
            "result_status": _clean(row.get("consolidated_status")),
            "promotion_authority": False,
            "testnet_order_authority": False,
            "live_trading_authorized": False,
            "previous_record_hash": previous_hash,
        }
        record["record_hash"] = content_identity(record, prefix="ledger_")
        previous_hash = str(record["record_hash"])
        records.append(record)
        known_fingerprints.add(fingerprint)
        family_attempts[hypothesis_id] = attempt
        added += 1
    _atomic_jsonl(records, ledger_path)
    chain_valid, chain_error = _validate_chain(records)
    audit_rows = [
        {
            "schema_version": LEDGER_SCHEMA,
            "semantic_hypothesis_id": hypothesis_id,
            "attempt_count": family_attempts[hypothesis_id],
            "source_alias_count": len({value for value in aliases.get(hypothesis_id, set()) if value}),
            "ledger_chain_valid": chain_valid,
            "chain_error": chain_error,
            "acceptance_policy_id": policy_receipt["policy_id"],
            "holdout_policy_id": holdout_id,
            "multiplicity_history_present": True,
            "promotion_authority": False,
            "live_trading_authorized": False,
        }
        for hypothesis_id in sorted(family_attempts)
    ]
    audit = pd.DataFrame(audit_rows)
    audit_path = root / "reports" / "active" / "hypothesis_ledger_audit.csv"
    _atomic_csv(audit, audit_path)
    return {
        "status": "PASS" if chain_valid else "BLOCKED",
        "ledger": ledger_path,
        "audit": audit_path,
        "source_rows": len(source),
        "ledger_records": len(records),
        "semantic_families": len(family_attempts),
        "records_added": added,
        "chain_valid": chain_valid,
        "chain_error": chain_error,
        "holdout_policy_id": holdout_id,
        "live_trading_authorized": False,
    }


def build_holdout_policy_receipt(
    *, root: Path = ROOT, now: datetime | None = None
) -> dict[str, Any]:
    policy = load_holdout_policy(root)
    policy_id = content_identity(policy, prefix="holdout_")
    receipt = {
        "schema_version": "thewiz.research_holdout_policy_receipt.v1",
        "registered_at_utc": _as_utc(now).isoformat(),
        "policy_id": policy_id,
        "policy_version": policy["policy_version"],
        "policy_path": "config/research_holdout_policy.json",
        "policy_sha256": sha256_file(root / "config" / "research_holdout_policy.json"),
        "prospective_roles_registered": True,
        "final_holdout_locked": True,
        "reuse_after_tuning_permitted": False,
        "status": "PASS",
        "promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    path = root / "reports" / "active" / "holdout_policy_receipt.json"
    _atomic_json(receipt, path)
    return receipt


def build_corrective_governance(
    *, root: Path = ROOT, now: datetime | None = None
) -> CommandResult:
    now = _as_utc(now)
    acceptance = build_acceptance_policy_receipt(root=root, now=now)
    tamper = run_gate_policy_tamper_tests(root=root)
    holdout = build_holdout_policy_receipt(root=root, now=now)
    ledger = build_hypothesis_ledger(root=root, now=now)
    paths = {
        "acceptance_policy_receipt": root / "reports" / "active" / "acceptance_policy_receipt.json",
        "gate_policy_tamper_results": root / "reports" / "red_team" / "gate_policy_tamper_results.csv",
        "hypothesis_ledger": Path(ledger["ledger"]),
        "hypothesis_ledger_audit": Path(ledger["audit"]),
        "holdout_policy_receipt": root / "reports" / "active" / "holdout_policy_receipt.json",
    }
    all_pass = acceptance["status"] == "PASS" and holdout["status"] == "PASS" and ledger["status"] == "PASS" and tamper["status"].eq("PASS").all()
    return CommandResult(
        paths=paths,
        summary={
            "status": "PASS" if all_pass else "BLOCKED",
            "acceptance_policy_id": acceptance["policy_id"],
            "holdout_policy_id": holdout["policy_id"],
            "tamper_cases": len(tamper),
            "semantic_hypothesis_families": ledger["semantic_families"],
            "ledger_records": ledger["ledger_records"],
            "ledger_records_added": ledger["records_added"],
            "live_trading_authorized": False,
        },
    )


def _material_leaves(value: Any, path: tuple[str | int, ...] = ()) -> Iterable[tuple[tuple[str | int, ...], Any]]:
    if isinstance(value, dict):
        for key in sorted(value):
            yield from _material_leaves(value[key], path + (key,))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _material_leaves(item, path + (index,))
    else:
        yield path, value


def _set_nested(payload: Any, path: tuple[str | int, ...], value: Any) -> None:
    current = payload
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = value


def _mutated_value(value: Any) -> Any:
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value + 0.125
    if value is None:
        return "tampered"
    return f"{value}__tampered"


def _read_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise GovernanceError(f"invalid ledger JSON at line {line_number}") from exc
    return records


def _validate_chain(records: list[dict[str, Any]]) -> tuple[bool, str]:
    previous = "GENESIS"
    for index, record in enumerate(records, start=1):
        if record.get("previous_record_hash") != previous:
            return False, f"previous_hash_mismatch:{index}"
        claimed = str(record.get("record_hash", ""))
        material = {key: value for key, value in record.items() if key != "record_hash"}
        expected = content_identity(material, prefix="ledger_")
        if claimed != expected:
            return False, f"record_hash_mismatch:{index}"
        previous = claimed
    return True, ""


def _atomic_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(canonical_json(row) + "\n" for row in records), encoding="utf-8")
    promote_staged_file(temporary, path)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    promote_staged_file(temporary, path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    promote_staged_file(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise GovernanceError(f"required governance file missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise GovernanceError(f"governance file must contain an object: {path}")
    return payload


def _safe_project_file(root: Path, relative: str) -> Path:
    if not relative or Path(relative).is_absolute():
        raise GovernanceError("source contract path must be relative")
    resolved_root = root.resolve()
    path = (resolved_root / relative).resolve()
    if resolved_root != path and resolved_root not in path.parents:
        raise GovernanceError("source contract path escapes project root")
    if not path.is_file():
        raise GovernanceError(f"source contract missing: {relative}")
    return path


def _normalize_mode(value: Any) -> str:
    token = _normalize_token(value).replace("zscorer", "zscore_r")
    aliases = {
        "static": "static_spread",
        "static_zscore_r": "static_zscore_r",
        "dyn": "dyn_spread",
        "dyn_zscore_r": "dyn_zscore_r",
        "ou": "ou_spread",
        "ou_zscore_r": "ou_zscore_r",
        "ou_optimal": "ou_optimal",
        "copula": "copula",
    }
    return aliases.get(token, token)


def _normalize_token(value: Any) -> str:
    return "_".join(str(value or "").strip().lower().replace("-", " ").replace("(", " ").replace(")", " ").split())


def _asset(value: Any) -> str:
    return str(value or "").upper().replace("-USD", "").replace("/USD", "").strip()


def _pair_assets(value: Any) -> list[str]:
    tokens = [token for token in str(value or "").upper().replace("/", "-").split("-") if token and token != "USD"]
    return list(dict.fromkeys(tokens))[:2]


def _clean(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def _json_scalar(value: Any) -> Any:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _as_utc(value: datetime | None) -> datetime:
    value = value or datetime.now(UTC)
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


if __name__ == "__main__":
    result = build_corrective_governance()
    print(json.dumps({"summary": result.summary, "paths": {key: str(value) for key, value in result.paths.items()}}, indent=2))
