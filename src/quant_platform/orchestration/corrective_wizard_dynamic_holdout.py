from __future__ import annotations

import inspect
import json
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_platform.active_pipeline import CommandResult
from quant_platform.wizard_dynamic_comparator_activation import (
    build_dynamic_v2_review_packet,
    build_reviewed_dynamic_v2_activation,
)
from quant_platform.wizard_hyperliquid_mode_proof import (
    _compare_spread_candidate,
    _kalman_dynamic_spread_candidate_v2_holdout,
)

ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "thewiz.wizard_dynamic_holdout_evaluation.v2"
DYNAMIC_MODES = ("Dyn (Spread)", "Dyn (ZScoreR)")
ORIENTATIONS = ("original", "reverse")


def evaluate_dynamic_v2_holdout(
    *,
    root: Path = ROOT,
    now: datetime | None = None,
    contract_path: Path | None = None,
    proof_path: Path | None = None,
) -> CommandResult:
    """Evaluate the frozen Dynamic-v2 candidate on its disjoint holdout only."""

    timestamp = _as_utc(now)
    contract_path = contract_path or root / "config" / "wizard_dynamic_comparator_v2_holdout.json"
    receipt_path = root / "reports" / "active" / "wizard_dynamic_comparator_v2_holdout_receipt.json"
    proof_path = proof_path or root / "reports" / "active" / "hyperliquid_wizard_vendor_mode_proofs.csv"
    contract = _read_json(contract_path)
    receipt = _read_json(receipt_path)
    _validate_registration(
        root=root,
        contract=contract,
        contract_path=contract_path,
        receipt=receipt,
    )

    holdout = contract["holdout_cohort"]
    derivation = contract["derivation_cohort"]
    holdout_group = _text(holdout.get("pair_group_id"))
    derivation_group = _text(derivation.get("pair_group_id"))
    if not holdout_group or holdout_group == derivation_group:
        raise ValueError("dynamic v2 holdout must be disjoint from derivation cohort")

    expected = {
        (mode, orientation)
        for mode in holdout.get("required_exact_modes", DYNAMIC_MODES)
        for orientation in holdout.get("required_orientations", ORIENTATIONS)
    }
    canonical_expected = {
        (mode, orientation) for mode in DYNAMIC_MODES for orientation in ORIENTATIONS
    }
    if expected != canonical_expected or int(holdout.get("required_mode_orientation_cells", 0)) != 4:
        raise ValueError("dynamic v2 holdout contract must require the canonical four cells")

    proofs = _read_csv(proof_path)
    if proofs.empty:
        selected = pd.DataFrame(
            columns=(
                "pair_group_id",
                "exact_mode",
                "orientation",
                "vendor_response_captured",
                "request_path",
                "response_path",
            )
        )
    else:
        selected = proofs.loc[
            proofs.get("pair_group_id", pd.Series("", index=proofs.index))
            .map(_text)
            .eq(holdout_group)
            & proofs.get("exact_mode", pd.Series("", index=proofs.index)).isin(DYNAMIC_MODES)
            & proofs.get("orientation", pd.Series("", index=proofs.index)).isin(ORIENTATIONS)
        ].copy()
    identities = list(zip(selected.get("exact_mode", []), selected.get("orientation", []), strict=False))
    duplicate_identities = sorted(
        {identity for identity in identities if identities.count(identity) > 1}
    )
    if duplicate_identities:
        raise ValueError(f"duplicate dynamic holdout proof identities: {duplicate_identities}")

    rows: list[dict[str, object]] = []
    for exact_mode, orientation in sorted(expected):
        match = selected.loc[
            selected["exact_mode"].eq(exact_mode)
            & selected["orientation"].eq(orientation)
        ]
        if match.empty:
            rows.append(
                _missing_row(
                    holdout_group=holdout_group,
                    exact_mode=exact_mode,
                    orientation=orientation,
                    blocker="holdout_vendor_response_missing",
                )
            )
            continue
        proof = match.iloc[0]
        if not _truthy(proof.get("vendor_response_captured")):
            rows.append(
                _missing_row(
                    holdout_group=holdout_group,
                    exact_mode=exact_mode,
                    orientation=orientation,
                    blocker="holdout_vendor_response_not_captured",
                )
            )
            continue
        request_path = _resolve(root, proof.get("request_path"))
        response_path = _resolve(root, proof.get("response_path"))
        if request_path is None or response_path is None:
            rows.append(
                _missing_row(
                    holdout_group=holdout_group,
                    exact_mode=exact_mode,
                    orientation=orientation,
                    blocker="holdout_request_or_response_path_missing",
                )
            )
            continue
        request = _read_json(request_path)
        response = _read_json(response_path)
        metrics = _evaluate_response(request=request, response=response)
        passed = metrics["vendor_formula_parity_status"] == "exact_reconstruction"
        rows.append(
            {
                "pair_group_id": holdout_group,
                "exact_mode": exact_mode,
                "orientation": orientation,
                "cell_status": "PASS" if passed else "FAIL",
                "vendor_response_captured": True,
                "spread_max_abs_error": metrics["vendor_spread_max_abs_error"],
                "zscore_max_abs_error": metrics["vendor_zscore_max_abs_error"],
                "zscore_roll_max_abs_error": metrics[
                    "vendor_zscore_roll_max_abs_error"
                ],
                "request_sha256": _file_hash(request_path),
                "response_sha256": _file_hash(response_path),
                "request_path": _relative(request_path, root),
                "response_path": _relative(response_path, root),
                "blocker": "" if passed else "dynamic_v2_holdout_exact_reconstruction_failed",
                "candidate_promotion_authority": False,
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            }
        )

    frame = pd.DataFrame(rows)
    captured = int(frame["vendor_response_captured"].map(_truthy).sum())
    passed = int(frame["cell_status"].eq("PASS").sum())
    failed = int(frame["cell_status"].eq("FAIL").sum())
    if captured == 0:
        status = "WAITING_FOR_HOLDOUT"
    elif failed > 0:
        status = "FAIL"
    elif captured < len(expected):
        status = "INCOMPLETE"
    elif passed == len(expected):
        status = "PASS"
    else:
        status = "INCOMPLETE"

    active = root / "reports" / "active"
    detail_path = active / "wizard_dynamic_v2_holdout_evaluation.csv"
    status_path = active / "wizard_dynamic_v2_holdout_status.json"
    _atomic_csv(frame, detail_path)
    response_hashes = sorted(
        value for value in frame["response_sha256"].map(_text).tolist() if value
    )
    result_payload = {
        "schema_version": SCHEMA_VERSION,
        "evaluated_at_utc": timestamp.isoformat(),
        "status": status,
        "contract_path": _relative(contract_path, root),
        "contract_sha256": _file_hash(contract_path),
        "holdout_pair_group_id": holdout_group,
        "derivation_pair_group_id": derivation_group,
        "required_cells": len(expected),
        "captured_cells": captured,
        "passed_cells": passed,
        "failed_cells": failed,
        "response_sha256s": response_hashes,
        "comparator_supersession_eligible": status == "PASS",
        "comparator_supersession_automatic": False,
        "research_only": True,
        "candidate_promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
        "evidence_path": _relative(detail_path, root),
    }
    _atomic_json(result_payload, status_path)
    immutable_path: Path | None = None
    if status in {"PASS", "FAIL"} and captured == len(expected):
        result_id = _result_id(result_payload)
        immutable_path = (
            root
            / "data"
            / "research"
            / "wizard_dynamic_holdout_evaluations"
            / f"{result_id}.json"
        )
        immutable_payload = {
            **{
                key: value
                for key, value in result_payload.items()
                if key not in {"evaluated_at_utc", "evidence_path"}
            },
            "result_id": result_id,
        }
        _write_or_validate_immutable_json(immutable_payload, immutable_path)
        result_payload["result_id"] = result_id
        result_payload["immutable_result_path"] = _relative(immutable_path, root)
        result_payload["immutable_result_sha256"] = _file_hash(immutable_path)
        _atomic_json(result_payload, status_path)

    paths = {"detail": detail_path, "status": status_path}
    if immutable_path is not None:
        paths["immutable_result"] = immutable_path
    return CommandResult(paths=paths, summary=result_payload)


def build_dynamic_v2_supersession_gate(*, root: Path = ROOT) -> CommandResult:
    """Require an immutable disjoint holdout PASS before reviewed supersession."""

    status_path = root / "reports" / "active" / "wizard_dynamic_v2_holdout_status.json"
    holdout = _read_json(status_path)
    blockers: list[str] = []
    if _text(holdout.get("status")) != "PASS":
        blockers.append("dynamic_v2_disjoint_holdout_not_passed")
    if int(holdout.get("required_cells", 0) or 0) != 4:
        blockers.append("dynamic_v2_required_cell_contract_not_four")
    if int(holdout.get("passed_cells", 0) or 0) != 4:
        blockers.append("dynamic_v2_four_holdout_cells_not_passed")
    if not _truthy(holdout.get("comparator_supersession_eligible")):
        blockers.append("dynamic_v2_evaluator_did_not_grant_supersession_eligibility")
    immutable_path = _resolve(root, holdout.get("immutable_result_path"))
    immutable_hash = _text(holdout.get("immutable_result_sha256"))
    if immutable_path is None or not immutable_hash:
        blockers.append("dynamic_v2_immutable_holdout_result_missing")
    elif _file_hash(immutable_path) != immutable_hash:
        blockers.append("dynamic_v2_immutable_holdout_result_hash_mismatch")
    if _truthy(holdout.get("comparator_supersession_automatic")):
        blockers.append("dynamic_v2_automatic_supersession_forbidden")
    forbidden = (
        "candidate_promotion_authority",
        "testnet_order_authority",
        "live_trading_authorized",
    )
    if any(_truthy(holdout.get(key)) for key in forbidden):
        blockers.append("dynamic_v2_holdout_result_exceeded_research_authority")

    gate_status = "READY_FOR_REVIEWED_SUPERSESSION" if not blockers else "BLOCKED"
    payload = {
        "schema_version": "thewiz.wizard_dynamic_comparator_supersession_gate.v1",
        "status": gate_status,
        "blockers": blockers,
        "source_holdout_status_path": _relative(status_path, root),
        "source_holdout_result_id": _text(holdout.get("result_id")),
        "source_holdout_result_path": (
            _relative(immutable_path, root) if immutable_path is not None else ""
        ),
        "source_holdout_result_sha256": immutable_hash,
        "supersession_requires_reviewed_apply": True,
        "supersession_applied": False,
        "research_only": True,
        "candidate_promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    output = root / "reports" / "active" / "wizard_dynamic_v2_supersession_gate.json"
    _atomic_json(payload, output)
    return CommandResult(paths={"supersession_gate": output}, summary=payload)


def build_dynamic_v2_reviewed_activation(
    *,
    root: Path = ROOT,
    apply: bool = False,
    reviewer: str = "",
    review_note: str = "",
    review_packet_id: str = "",
    now: datetime | None = None,
) -> CommandResult:
    """Plan or explicitly apply the versioned, research-only v2 comparator."""

    source_hash = sha256(
        inspect.getsource(_kalman_dynamic_spread_candidate_v2_holdout).encode(
            "utf-8"
        )
    ).hexdigest()
    return build_reviewed_dynamic_v2_activation(
        root=root,
        implementation_source_sha256=source_hash,
        apply=apply,
        reviewer=reviewer,
        review_note=review_note,
        review_packet_id=review_packet_id,
        now=now,
    )


def build_dynamic_v2_review_handoff(
    *, root: Path = ROOT, now: datetime | None = None
) -> CommandResult:
    """Publish the exact immutable packet a human reviewer must approve."""

    source_hash = sha256(
        inspect.getsource(_kalman_dynamic_spread_candidate_v2_holdout).encode(
            "utf-8"
        )
    ).hexdigest()
    return build_dynamic_v2_review_packet(
        root=root,
        implementation_source_sha256=source_hash,
        now=now,
    )


def _evaluate_response(
    *, request: dict[str, Any], response: dict[str, Any]
) -> dict[str, object]:
    params = request.get("params") if isinstance(request.get("params"), dict) else {}
    history = response.get("history") if isinstance(response.get("history"), dict) else {}
    stats = history.get("spread_stats") if isinstance(history.get("spread_stats"), dict) else {}
    try:
        x = np.asarray(params["series_1_closes"], dtype=float)
        y = np.asarray(params["series_2_closes"], dtype=float)
        window = int(params["roll_w"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("dynamic holdout request is missing required numeric inputs") from exc
    if len(x) < 50 or len(x) != len(y):
        raise ValueError("dynamic holdout request series are invalid")
    if bool(stats.get("log_used", False)):
        if np.any(x <= 0.0) or np.any(y <= 0.0):
            raise ValueError("dynamic holdout log transform requires positive prices")
        x, y = np.log(x), np.log(y)
    spread = _kalman_dynamic_spread_candidate_v2_holdout(x, y)
    return _compare_spread_candidate(
        spread=spread,
        stats=stats,
        window=window,
        spread_formula="dynamic_v2_disjoint_holdout",
    )


def _validate_registration(
    *,
    root: Path,
    contract: dict[str, Any],
    contract_path: Path,
    receipt: dict[str, Any],
) -> None:
    if not contract or not receipt:
        raise ValueError("dynamic v2 contract and receipt are required")
    if _text(receipt.get("contract_path")) != _relative(contract_path, root):
        raise ValueError("dynamic v2 receipt contract path mismatch")
    if _text(receipt.get("contract_sha256")) != _file_hash(contract_path):
        raise ValueError("dynamic v2 contract hash mismatch")
    source_hash = sha256(
        inspect.getsource(_kalman_dynamic_spread_candidate_v2_holdout).encode(
            "utf-8"
        )
    ).hexdigest()
    if source_hash != _text(contract.get("implementation", {}).get("source_sha256")):
        raise ValueError("dynamic v2 implementation source hash mismatch")
    for evidence in contract.get("derivation_evidence", []):
        path = root / _text(evidence.get("path"))
        if not path.is_file() or _file_hash(path) != _text(evidence.get("sha256")):
            raise ValueError("dynamic v2 derivation evidence hash mismatch")
    forbidden = (
        "candidate_promotion_authority",
        "testnet_order_authority",
        "live_trading_authorized",
    )
    if any(_truthy(contract.get(key)) or _truthy(receipt.get(key)) for key in forbidden):
        raise ValueError("dynamic v2 registration cannot grant authority")
    if not _truthy(receipt.get("registered_before_holdout_vendor_responses")):
        raise ValueError("dynamic v2 candidate was not registered before holdout")


def _missing_row(
    *, holdout_group: str, exact_mode: str, orientation: str, blocker: str
) -> dict[str, object]:
    return {
        "pair_group_id": holdout_group,
        "exact_mode": exact_mode,
        "orientation": orientation,
        "cell_status": "MISSING",
        "vendor_response_captured": False,
        "spread_max_abs_error": "",
        "zscore_max_abs_error": "",
        "zscore_roll_max_abs_error": "",
        "request_sha256": "",
        "response_sha256": "",
        "request_path": "",
        "response_path": "",
        "blocker": blocker,
        "candidate_promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }


def _result_id(payload: dict[str, Any]) -> str:
    stable = {
        key: value
        for key, value in payload.items()
        if key not in {"evaluated_at_utc", "evidence_path"}
    }
    digest = sha256(
        json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"dynamic_holdout_{digest[:20]}"


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _resolve(root: Path, value: object) -> Path | None:
    text = _text(value)
    if not text:
        return None
    path = Path(text)
    path = path if path.is_absolute() else root / path
    return path if path.is_file() else None


def _file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(f"{path.suffix}.tmp")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(f"{path.suffix}.tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def _write_or_validate_immutable_json(payload: dict[str, Any], path: Path) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.is_file():
        if path.read_text(encoding="utf-8") != encoded:
            raise ValueError(f"immutable dynamic holdout result changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded, encoding="utf-8")


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _text(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return _text(value).lower() in {"1", "true", "yes", "y"}


def _as_utc(value: datetime | None) -> datetime:
    value = value or datetime.now(UTC)
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


if __name__ == "__main__":
    result = evaluate_dynamic_v2_holdout()
    gate = build_dynamic_v2_supersession_gate()
    print(
        json.dumps(
            {"summary": result.summary, "supersession_gate": gate.summary}, indent=2
        )
    )
