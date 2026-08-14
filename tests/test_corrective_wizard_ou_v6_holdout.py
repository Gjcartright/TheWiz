from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_platform.orchestration.corrective_wizard_ou_v4_holdout import (
    _ou_spread_candidate_v4,
)
from quant_platform.orchestration.corrective_wizard_ou_v6_holdout import (
    ATTRIBUTION_STATUS,
    DEFAULT_PAIR_SPECS,
    PRIOR_OU_ASSETS,
    V5_DERIVATION,
    V5_EVALUATION,
    _build_derivation,
    _ou_v6_predictor,
    register_ou_v6_prospective_holdout,
    run_ou_v6_prospective_holdout,
    validate_ou_v6_evaluation_binding,
    validate_ou_v6_registration_binding,
    validate_ou_v6_stage3_evidence,
)
from quant_platform.orchestration.corrective_wizard_ou_v6_supreme_review import (
    build_ou_v6_supreme_review,
)
from quant_platform.wizard_hyperliquid_mode_proof import (
    refresh_activated_ou_v6_proofs,
)
from quant_platform.wizard_ou_v6_comparator_activation import (
    build_ou_v6_review_packet,
    build_ou_v6_supersession_gate,
    build_reviewed_ou_v6_activation,
    load_validated_ou_v6_activation,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _copy_file(source: Path, *, root: Path, relative: Path | None = None) -> Path:
    destination = root / (relative or source.relative_to(PROJECT_ROOT))
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def _copy_v6_registration_inputs(root: Path) -> None:
    attribution_source = PROJECT_ROOT / ATTRIBUTION_STATUS
    attribution = json.loads(attribution_source.read_text(encoding="utf-8"))
    _copy_file(attribution_source, root=root)
    immutable = PROJECT_ROOT / str(attribution["immutable_attribution_path"])
    _copy_file(immutable, root=root)

    for relative in (V5_DERIVATION, V5_EVALUATION):
        source = PROJECT_ROOT / relative
        _copy_file(source, root=root)
        frame = pd.read_csv(source)
        if "exact_mode" in frame:
            frame = frame.loc[frame["exact_mode"].eq("OU (Spread)")]
        for column in ("request_path", "response_path"):
            for raw in frame[column].dropna().astype(str).unique():
                _copy_file(PROJECT_ROOT / raw, root=root)

    for spec in DEFAULT_PAIR_SPECS:
        for field in ("asset_x_path", "asset_y_path"):
            _copy_file(PROJECT_ROOT / spec[field], root=root)


def test_v6_derivation_uses_consumed_v5_rows_without_treating_them_as_validation() -> None:
    frame = _build_derivation(PROJECT_ROOT)

    assert len(frame) == 14
    assert frame["transform_rule_passed"].all()
    assert frame["profile_branch_rule_passed"].all()
    assert set(frame["predicted_profile_branch"]) == {"intercept", "zero_mean"}
    assert frame["holdout_reuse_allowed"].eq(False).all()
    assert frame["cohort"].eq("v5_consumed_holdout_derivation_only").sum() == 4


def test_v6_registration_is_disjoint_prospective_terminal_and_idempotent(
    tmp_path: Path,
) -> None:
    _copy_v6_registration_inputs(tmp_path)

    first = register_ou_v6_prospective_holdout(root=tmp_path)
    repeated = register_ou_v6_prospective_holdout(root=tmp_path)
    contract = json.loads(first.paths["contract"].read_text(encoding="utf-8"))
    predictions = pd.read_csv(first.paths["predictions"])

    assert first.summary == repeated.summary
    assert validate_ou_v6_registration_binding(root=tmp_path)["status"] == "PASS"
    assert first.summary["vendor_responses_at_registration"] == 0
    assert first.summary["final_successor_iteration"] is True
    assert first.summary["successor_after_v6_failure_allowed"] is False
    assert contract["final_successor_iteration"] is True
    assert contract["successor_after_v6_failure_allowed"] is False
    assert contract["hypothesis_changes_after_vendor_response_allowed"] is False
    assert contract["automatic_activation"] is False
    assert contract["research_only"] is True
    assert contract["candidate_promotion_authority"] is False
    assert contract["testnet_order_authority"] is False
    assert contract["live_trading_authorized"] is False
    assert contract["vendor_responses_at_registration"] == 0
    assert len(contract["holdout_bindings"]) == 8
    assert len(predictions) == 8
    assert predictions["vendor_response_present_at_registration"].eq(False).all()
    assert set(predictions["transform"]) == {"level", "log"}
    assert set(predictions["predicted_profile_branch"]) == {
        "intercept",
        "zero_mean",
    }
    assert set(predictions["predicted_inc_trend"]) == {False, True}
    assert set(first.summary["holdout_pair_groups"]) == {"APT-ATOM", "ARB-OP"}

    assets = {
        str(binding[field])
        for binding in contract["holdout_bindings"]
        for field in ("asset_x", "asset_y")
    }
    assert assets == {"APT", "ATOM", "ARB", "OP"}
    assert assets.isdisjoint(PRIOR_OU_ASSETS)
    for binding in contract["holdout_bindings"]:
        assert not (tmp_path / str(binding["response_path"])).exists()


def test_v6_registration_rejects_prior_ou_asset_overlap(tmp_path: Path) -> None:
    _copy_v6_registration_inputs(tmp_path)
    overlapping = [dict(item) for item in DEFAULT_PAIR_SPECS]
    overlapping[0] = {
        **overlapping[0],
        "pair": "SOL-ATOM",
        "asset_x": "SOL",
        "asset_x_path": str(
            PROJECT_ROOT / "reports/snapshots/current_wizard_hyperliquid/"
            "ewapi_69a86274da721a941e36/"
            "cwhandoff_61dc9e76532abd99c502/history_runs/"
            "cwhistoryrun_20260812T115658834898Z_1d7a24ca/assets/"
            "SOL_1d_candles.json"
        ),
    }

    with pytest.raises(ValueError, match="overlap prior OU evidence"):
        register_ou_v6_prospective_holdout(root=tmp_path, pair_specs=overlapping)


def test_v6_review_activation_refresh_and_terminal_stage3_binding(tmp_path: Path) -> None:
    _copy_v6_registration_inputs(tmp_path)
    registration = register_ou_v6_prospective_holdout(root=tmp_path)

    result = run_ou_v6_prospective_holdout(
        root=tmp_path,
        now=datetime(2026, 8, 15, 0, 5, tzinfo=UTC),
        execute=True,
        api_key="test-secret",
        fetcher=lambda request, **_: _v6_response_for_request(request.payload()),
        credits_fetcher=lambda **_: 0,
    )
    assert result.summary["evaluation_status"] == "PASS"
    assert validate_ou_v6_evaluation_binding(root=tmp_path)["status"] == "PASS"

    _write_v6_activation_dependencies(tmp_path)
    _write_v6_refresh_proofs(tmp_path)
    gate = build_ou_v6_supersession_gate(root=tmp_path)
    packet = build_ou_v6_review_packet(root=tmp_path)
    planned = build_reviewed_ou_v6_activation(root=tmp_path)
    no_supreme = build_reviewed_ou_v6_activation(
        root=tmp_path,
        apply=True,
        reviewer="research-reviewer",
        review_note="Reviewed every immutable terminal OU v6 comparator cell independently.",
        review_packet_id=packet.summary["review_packet_id"],
    )
    supreme = build_ou_v6_supreme_review(root=tmp_path)
    applied = build_reviewed_ou_v6_activation(
        root=tmp_path,
        apply=True,
        reviewer="research-reviewer",
        review_note=(
            "Reviewed all terminal OU v6 formula, transform, trend, and profile cells."
        ),
        review_packet_id=packet.summary["review_packet_id"],
    )
    refreshed = refresh_activated_ou_v6_proofs(root=tmp_path)
    activation = load_validated_ou_v6_activation(root=tmp_path)
    stage3 = validate_ou_v6_stage3_evidence(
        root=tmp_path,
        evidence={
            "ou_v6_prospectively_registered": True,
            "ou_v6_required_responses": 8,
            "ou_v6_responses_available": 8,
            "ou_v6_holdout_status": "COMPLETE",
            "ou_v6_evaluation_status": "PASS",
            "ou_v6_response_accounting_valid": True,
            "ou_v6_activation_status": "APPLIED_RESEARCH_COMPARATOR_ONLY",
            "ou_v6_comparator_generation": 6,
            "ou_v6_proof_refresh_status": "PASS",
            "ou_v6_proofs_refreshed": 8,
            "ou_v6_activation_automatic": False,
            "ou_v6_research_only": True,
            "candidate_promotion_authority": False,
            "order_submission_included": False,
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        },
    )

    assert registration.summary["final_successor_iteration"] is True
    assert gate.summary["status"] == "READY_FOR_REVIEWED_SUPERSESSION"
    assert packet.summary["status"] == "READY_FOR_EXPLICIT_REVIEW"
    assert packet.summary["raw_bindings_valid"] is True
    assert planned.summary["status"] == "READY_REQUIRES_EXPLICIT_APPLY"
    assert no_supreme.summary["status"] == "BLOCKED"
    assert "ou_v6_supreme_review_required_for_apply" in no_supreme.summary["blockers"]
    assert supreme.summary["status"] == "PASS_ADVISORY_ONLY"
    assert supreme.summary["max_tolerance_ratio"] <= 1.0
    assert supreme.summary["final_successor_iteration"] is True
    assert supreme.summary["successor_after_v6_failure_allowed"] is False
    assert applied.summary["status"] == "APPLIED_RESEARCH_COMPARATOR_ONLY"
    assert applied.summary["comparator_generation"] == 6
    assert refreshed.summary["status"] == "PASS"
    assert refreshed.summary["exact_ou_rows"] == 8
    assert activation is not None
    assert stage3["status"] == "PASS"
    assert not applied.summary["candidate_promotion_authority"]
    assert not applied.summary["testnet_order_authority"]
    assert not applied.summary["live_trading_authorized"]

    detail = pd.read_csv(result.paths["detail"])
    detail.loc[0, "predicted_profile_branch"] = "tampered_after_review"
    detail.to_csv(result.paths["detail"], index=False)
    tampered = validate_ou_v6_evaluation_binding(root=tmp_path)
    assert tampered["status"] == "BLOCKED"
    assert "ou_v6_evaluation_detail_hash_mismatch" in tampered["blockers"]


def test_v6_failed_holdout_is_terminal_and_cannot_activate_or_register_v7(
    tmp_path: Path,
) -> None:
    _copy_v6_registration_inputs(tmp_path)
    register_ou_v6_prospective_holdout(root=tmp_path)
    calls = 0

    def mismatched_fetcher(request, **_):
        nonlocal calls
        response = _v6_response_for_request(request.payload())
        if calls == 0:
            response["history"]["coint_eg"]["inc_trend"] = not response["history"][
                "coint_eg"
            ]["inc_trend"]
        calls += 1
        return response

    result = run_ou_v6_prospective_holdout(
        root=tmp_path,
        now=datetime(2026, 8, 15, 0, 5, tzinfo=UTC),
        execute=True,
        api_key="test-secret",
        fetcher=mismatched_fetcher,
        credits_fetcher=lambda **_: 0,
    )
    _write_v6_activation_dependencies(tmp_path)
    _write_v6_refresh_proofs(tmp_path)
    gate = build_ou_v6_supersession_gate(root=tmp_path)
    packet = build_ou_v6_review_packet(root=tmp_path)
    supreme = build_ou_v6_supreme_review(root=tmp_path)
    activation = build_reviewed_ou_v6_activation(root=tmp_path)

    assert calls == 8
    assert result.summary["evaluation_status"] == "FAIL"
    assert gate.summary["status"] == "BLOCKED"
    assert packet.summary["status"] == "BLOCKED_HOLDOUT_EVIDENCE"
    assert supreme.summary["status"] == "BLOCKED"
    assert supreme.summary["recommendation"] == "DO_NOT_ACTIVATE"
    assert activation.summary["status"] == "BLOCKED"
    assert activation.summary["final_successor_iteration"] is True
    assert activation.summary["successor_after_v6_failure_allowed"] is False
    assert not list(
        (tmp_path / "data/research/wizard_ou_v6_comparator_activations").glob("*.json")
    )
    assert not (tmp_path / "config/wizard_ou_comparator_v7_holdout.json").exists()


def _v6_response_for_request(payload: dict[str, object]) -> dict[str, object]:
    params = payload["params"]
    assert isinstance(params, dict)
    x = np.asarray(params["series_1_closes"], dtype=float)
    y = np.asarray(params["series_2_closes"], dtype=float)
    prediction = _ou_v6_predictor(x, y)
    local_x = np.log(x) if prediction["predicted_log_used"] else x
    local_y = np.log(y) if prediction["predicted_log_used"] else y
    beta, spread, _ = _ou_spread_candidate_v4(
        local_x,
        local_y,
        inc_trend=prediction["predicted_profile_branch"] == "zero_mean",
    )
    zscore = (spread - spread.mean()) / spread.std(ddof=1)
    window = int(params["roll_w"])
    series = pd.Series(spread)
    zscore_roll = (
        series.sub(series.rolling(window, min_periods=window).mean())
        .div(series.rolling(window, min_periods=window).std(ddof=1))
        .fillna(0.0)
        .to_numpy()
    )
    lagged = spread[:-1]
    current = spread[1:]
    _, phi = np.linalg.lstsq(
        np.column_stack([np.ones(len(lagged)), lagged]),
        current,
        rcond=None,
    )[0]
    half_life = float(-np.log(2.0) / np.log(phi))
    return {
        "data": {},
        "history": {
            "coint_eg": {"inc_trend": prediction["predicted_inc_trend"]},
            "spread_stats": {
                "spread": spread.tolist(),
                "zscore": zscore.tolist(),
                "zscore_roll": zscore_roll.tolist(),
                "hedge_ratio": beta,
                "half_life": half_life,
                "log_used": prediction["predicted_log_used"],
            },
        },
    }


def _write_v6_activation_dependencies(root: Path) -> None:
    active = root / "reports/active"
    active.mkdir(parents=True, exist_ok=True)
    (active / "wizard_mode_comparator_contract.csv").write_text(
        "exact_mode,version\nOU (Spread),v1\n",
        encoding="utf-8",
    )
    (active / "wizard_mode_comparator_contract_receipt.json").write_text(
        json.dumps({"contract_id": "wizard-comparator-v1"}),
        encoding="utf-8",
    )


def _write_v6_refresh_proofs(root: Path) -> None:
    contract = json.loads(
        (root / "config/wizard_ou_comparator_v6_holdout.json").read_text(encoding="utf-8")
    )
    rows = [
        {
            "pair_group_id": binding["pair_group"],
            "pair": binding["pair"],
            "exact_mode": binding["exact_mode"],
            "orientation": binding["orientation"],
            "vendor_response_captured": True,
            "mode_proof_status": "completed",
            "vendor_history_available": True,
            "proof_window_kind": "scanner_horizon_parity",
            "formula_comparator_generation": 1,
            "request_path": binding["request_path"],
            "response_path": binding["response_path"],
        }
        for binding in contract["holdout_bindings"]
    ]
    pd.DataFrame(rows).to_csv(
        root / "reports/active/hyperliquid_wizard_vendor_mode_proofs.csv",
        index=False,
    )
