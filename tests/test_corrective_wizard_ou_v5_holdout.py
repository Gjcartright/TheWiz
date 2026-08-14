from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_platform.active_pipeline import CommandResult
from quant_platform.orchestration import corrective_wizard_ou_v5_holdout
from quant_platform.orchestration.corrective_wizard_ou_v4_holdout import (
    _ou_stationary_constrained_profile_fit_v4,
)
from quant_platform.orchestration.corrective_wizard_ou_v5_holdout import (
    _ou_v5_transform_selector,
    audit_ou_v5_capture_readiness,
    evaluate_ou_v5_prospective_holdout,
    register_ou_v5_prospective_holdout,
    run_ou_v5_prospective_holdout,
    validate_ou_v5_evaluation_binding,
    validate_ou_v5_stage3_evidence,
)
from quant_platform.orchestration.corrective_wizard_ou_v5_supreme_review import (
    build_ou_v5_supreme_review,
)
from quant_platform.orchestration.corrective_wizard_parity import (
    _ou_v5_evidence_gate,
)
from quant_platform.wizard_hyperliquid_mode_proof import (
    _ou_trend_aware_profile_beta_v3_candidate,
    refresh_activated_ou_v5_proofs,
)
from quant_platform.wizard_ou_v5_comparator_activation import (
    build_ou_v5_review_packet,
    build_ou_v5_supersession_gate,
    build_reviewed_ou_v5_activation,
    load_validated_ou_v5_activation,
)


def test_v5_transform_hypothesis_is_frozen_and_scale_sensitive() -> None:
    high = _series(100.0, 120)
    low = _series(0.1, 120)

    assert _ou_v5_transform_selector(high, high * 2)["predicted_log_used"] is True
    result = _ou_v5_transform_selector(high, low)
    assert result["predicted_log_used"] is False
    assert result["transform_threshold"] == 1.0
    assert result["scale_sensitivity_warning"] is True


def test_v5_supreme_review_reports_actual_failed_cell_count(tmp_path: Path) -> None:
    identities = [
        (pair_group, exact_mode, orientation)
        for pair_group in ("PAIR-A", "PAIR-B")
        for exact_mode in ("OU (Spread)", "OU (ZScoreR)")
        for orientation in ("original", "reverse")
    ]
    contract_path = tmp_path / "config/wizard_ou_comparator_v5_holdout.json"
    contract_path.parent.mkdir(parents=True)
    contract_path.write_text(
        json.dumps(
            {
                "holdout_bindings": [
                    {
                        "pair_group": pair_group,
                        "exact_mode": exact_mode,
                        "orientation": orientation,
                    }
                    for pair_group, exact_mode, orientation in identities
                ]
            }
        ),
        encoding="utf-8",
    )
    cells = []
    for index, (pair_group, exact_mode, orientation) in enumerate(identities):
        passed = index < 4
        cells.append(
            {
                "pair_group": pair_group,
                "exact_mode": exact_mode,
                "orientation": orientation,
                "cell_status": "PASS" if passed else "FAIL",
                "formula_parity_passed": passed,
                "transform_selector_parity_passed": True,
                "trend_selector_parity_passed": True,
                "profile_branch_selector_parity_passed": passed,
                "raw_binding_valid": True,
                "hedge_ratio_abs_error": 0.0 if passed else 0.4,
                "spread_max_abs_error": 0.0,
                "zscore_max_abs_error": 0.0,
                "zscore_roll_max_abs_error": 0.0,
            }
        )

    result = build_ou_v5_supreme_review(
        root=tmp_path,
        packet_builder=lambda **_: CommandResult(
            paths={},
            summary={
                "status": "BLOCKED",
                "cells": cells,
                "cohorts_disjoint": True,
                "all_cells_passed": False,
                "raw_bindings_valid": True,
                "transform_scale_sensitive": True,
                "profile_threshold_is_fragile": True,
                "profile_intercept_threshold": 0.1,
                "profile_training_separation": 0.01,
                "candidate_promotion_authority": False,
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            },
        ),
        activation_planner=lambda **_: CommandResult(
            paths={},
            summary={
                "status": "BLOCKED",
                "apply_requested": False,
                "candidate_promotion_authority": False,
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            },
        ),
    )

    post_mortem = next(row for row in result.summary["findings"] if row["lens"] == "post_mortem")
    assert result.summary["passed_cells"] == 4
    assert "passed 4/8 cells" in post_mortem["finding"]
    assert "do not activate v5" in post_mortem["required_action"]


def test_v5_cross_day_unresolved_intent_blocks_replay(tmp_path: Path) -> None:
    contract_path = tmp_path / "config/wizard_ou_comparator_v5_holdout.json"
    contract_path.parent.mkdir(parents=True)
    contract_path.write_text('{"frozen":true}\n', encoding="utf-8")
    request_path = tmp_path / "data/research/wizard_ou_v5_holdout/requests/request.json"
    request_path.parent.mkdir(parents=True)
    request_path.write_text('{"request":true}\n', encoding="utf-8")
    binding = {
        "pair": "SOL-BNB",
        "exact_mode": "OU (Spread)",
        "orientation": "original",
        "proof_observations": 120,
        "request_path": str(request_path.relative_to(tmp_path)),
        "request_sha256": _sha(request_path),
        "response_path": ("data/raw/wizard_ou_v5_holdout/responses/sol_bnb_response.json"),
    }
    first_day = datetime(2026, 8, 13, tzinfo=UTC)
    call_id = corrective_wizard_ou_v5_holdout._call_id(binding)
    intent_path = corrective_wizard_ou_v5_holdout._call_intent_path(
        root=tmp_path,
        timestamp=first_day,
        call_id=call_id,
    )
    intent_path.parent.mkdir(parents=True)
    intent = corrective_wizard_ou_v5_holdout._call_intent_payload(
        root=tmp_path,
        timestamp=first_day,
        contract_path=contract_path,
        binding=binding,
        call_id=call_id,
    )
    intent_path.write_text(json.dumps(intent), encoding="utf-8")

    blockers = corrective_wizard_ou_v5_holdout._ambiguous_call_intent_blockers(
        root=tmp_path,
        timestamp=datetime(2026, 8, 14, tzinfo=UTC),
        contract_path=contract_path,
        missing=[binding],
    )

    assert blockers == [
        (f"ou_v5_call_attempt_already_registered_without_response:{call_id}:2026-08-13")
    ]


def test_v5_runner_never_calls_vendor_after_cross_day_unresolved_intent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path = tmp_path / "data/research/wizard_ou_v5_holdout/requests/request.json"
    request_path.parent.mkdir(parents=True)
    request_path.write_text('{"request":true}\n', encoding="utf-8")
    binding = {
        "pair": "SOL-BNB",
        "exact_mode": "OU (Spread)",
        "orientation": "original",
        "proof_observations": 120,
        "request_path": str(request_path.relative_to(tmp_path)),
        "request_sha256": _sha(request_path),
        "response_path": ("data/raw/wizard_ou_v5_holdout/responses/sol_bnb_response.json"),
    }
    contract_path = tmp_path / corrective_wizard_ou_v5_holdout.CONTRACT_PATH
    receipt_path = tmp_path / corrective_wizard_ou_v5_holdout.RECEIPT_PATH
    contract_path.parent.mkdir(parents=True)
    receipt_path.parent.mkdir(parents=True)
    contract_path.write_text(json.dumps({"holdout_bindings": [binding]}), encoding="utf-8")
    receipt_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        corrective_wizard_ou_v5_holdout,
        "_validate_registration",
        lambda *args, **kwargs: None,
    )
    first_day = datetime(2026, 8, 13, tzinfo=UTC)
    call_id = corrective_wizard_ou_v5_holdout._call_id(binding)
    intent_path = corrective_wizard_ou_v5_holdout._call_intent_path(
        root=tmp_path,
        timestamp=first_day,
        call_id=call_id,
    )
    intent_path.parent.mkdir(parents=True)
    intent_path.write_text(
        json.dumps(
            corrective_wizard_ou_v5_holdout._call_intent_payload(
                root=tmp_path,
                timestamp=first_day,
                contract_path=contract_path,
                binding=binding,
                call_id=call_id,
            )
        ),
        encoding="utf-8",
    )
    vendor_calls = 0

    def forbidden_fetcher(*args, **kwargs):
        nonlocal vendor_calls
        vendor_calls += 1
        raise AssertionError("vendor fetch must remain blocked")

    result = run_ou_v5_prospective_holdout(
        root=tmp_path,
        now=datetime(2026, 8, 14, tzinfo=UTC),
        execute=True,
        api_key="secret",
        fetcher=forbidden_fetcher,
        credits_fetcher=lambda **_: (_ for _ in ()).throw(
            AssertionError("credit preflight must remain blocked")
        ),
    )

    assert vendor_calls == 0
    assert result.summary["calls_made"] == 0
    assert result.summary["responses_captured"] == 0
    assert result.summary["blocker"] == (
        f"ou_v5_call_attempt_already_registered_without_response:{call_id}:2026-08-13"
    )


def test_v5_registration_freezes_disjoint_eight_cells_without_vendor_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specs = _write_pair_specs(tmp_path)
    attribution = tmp_path / "data/research/attribution.json"
    attribution.parent.mkdir(parents=True)
    attribution.write_text('{"status":"complete"}\n', encoding="utf-8")
    monkeypatch.setattr(
        corrective_wizard_ou_v5_holdout,
        "_validated_v4_attribution",
        lambda _root: {
            "immutable_attribution_path": str(attribution.relative_to(tmp_path)),
            "immutable_attribution_sha256": _sha(attribution),
        },
    )
    monkeypatch.setattr(
        corrective_wizard_ou_v5_holdout,
        "_build_derivation",
        lambda _root: (
            pd.DataFrame(
                {
                    "transform_rule_passed": [True] * 10,
                    "profile_branch_rule_passed": [True] * 10,
                }
            ),
            0.0027,
            0.00002,
        ),
    )

    def predictor(x, y, *, profile_intercept_threshold):
        high = min(float(np.median(x)), float(np.median(y))) >= 1.0
        first_is_larger = float(np.median(x)) > float(np.median(y))
        branch = (
            "zero_mean"
            if (high and first_is_larger) or (not high and not first_is_larger)
            else "intercept"
        )
        return {
            "transform": "log" if high else "level",
            "predicted_log_used": high,
            "predicted_inc_trend": bool(not high and first_is_larger),
            "predicted_profile_branch": branch,
            "profile_intercept_threshold": profile_intercept_threshold,
        }

    monkeypatch.setattr(
        corrective_wizard_ou_v5_holdout,
        "_ou_v5_predictor",
        predictor,
    )
    first = register_ou_v5_prospective_holdout(
        root=tmp_path,
        now=datetime(2026, 8, 13, tzinfo=UTC),
        pair_specs=specs,
        observations=120,
    )
    repeated = register_ou_v5_prospective_holdout(
        root=tmp_path,
        pair_specs=specs,
        observations=120,
    )
    contract = json.loads(first.paths["contract"].read_text(encoding="utf-8"))
    predictions = pd.read_csv(first.paths["predictions"])

    assert first.summary["status"] == "PREREGISTERED_WAITING_VENDOR_RESPONSES"
    assert repeated.summary == first.summary
    assert len(contract["holdout_bindings"]) == 8
    assert len(predictions) == 8
    assert set(predictions["transform"]) == {"level", "log"}
    assert set(predictions["predicted_profile_branch"]) == {"zero_mean", "intercept"}
    assert set(predictions["predicted_inc_trend"].map(bool)) == {False, True}
    assert not predictions["vendor_response_present_at_registration"].map(bool).any()
    assert not any(
        (tmp_path / binding["response_path"]).exists() for binding in contract["holdout_bindings"]
    )
    assert contract["derivation_summary"]["v4_holdout_reuse_allowed"] is False
    assert first.summary["candidate_promotion_authority"] is False
    assert first.summary["testnet_order_authority"] is False
    assert first.summary["live_trading_authorized"] is False


def test_v5_registration_rejects_prior_ou_assets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specs = _write_pair_specs(tmp_path)
    specs[0]["pair"] = "BTC-NEW2"
    specs[0]["asset_x"] = "BTC"
    monkeypatch.setattr(
        corrective_wizard_ou_v5_holdout,
        "_validated_v4_attribution",
        lambda _root: {},
    )
    monkeypatch.setattr(
        corrective_wizard_ou_v5_holdout,
        "_build_derivation",
        lambda _root: (
            pd.DataFrame(
                {
                    "transform_rule_passed": [True] * 10,
                    "profile_branch_rule_passed": [True] * 10,
                }
            ),
            0.0027,
            0.00002,
        ),
    )

    with pytest.raises(ValueError, match="overlap prior OU evidence"):
        register_ou_v5_prospective_holdout(
            root=tmp_path,
            pair_specs=specs,
            observations=120,
        )


def test_v5_capture_writes_intents_and_evaluates_all_frozen_cells(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _register_v5_fixture(tmp_path, monkeypatch)
    calls = 0

    def fetcher(request, *, api_key):
        nonlocal calls
        assert api_key == "secret"
        calls += 1
        return _response_for_request(request.payload())

    result = run_ou_v5_prospective_holdout(
        root=tmp_path,
        now=datetime(2026, 8, 14, tzinfo=UTC),
        execute=True,
        api_key="secret",
        fetcher=fetcher,
        credits_fetcher=lambda **_: 0,
    )
    status = json.loads(result.paths["status"].read_text(encoding="utf-8"))
    raw_paths = sorted((tmp_path / "data/raw/wizard_ou_v5_holdout/responses").glob("*.json"))
    raw_hashes_before = {path: _sha(path) for path in raw_paths}

    assert calls == 8
    assert result.summary["calls_made"] == 8
    assert result.summary["responses_captured"] == 8
    assert result.summary["credits_attempted"] == 16
    assert result.summary["evaluation_status"] == "PASS"
    assert status["passed_cells"] == 8
    assert status["transform_selector_parity_passed_cells"] == 8
    assert status["trend_selector_parity_passed_cells"] == 8
    assert status["profile_branch_selector_parity_passed_cells"] == 8
    assert status["formula_parity_passed_cells"] == 8
    evaluation_binding = validate_ou_v5_evaluation_binding(root=tmp_path)
    assert evaluation_binding["status"] == "PASS"
    assert evaluation_binding["required_cells"] == 8
    assert evaluation_binding["passed_cells"] == 8
    stage3_evidence = {
        "ou_v5_prospectively_registered": True,
        "ou_v5_required_responses": 8,
        "ou_v5_responses_available": 8,
        "ou_v5_holdout_status": "COMPLETE",
        "ou_v5_evaluation_status": "PASS",
        "ou_v5_response_accounting_valid": True,
        "ou_v5_activation_status": "APPLIED_RESEARCH_COMPARATOR_ONLY",
        "ou_v5_comparator_generation": 5,
        "ou_v5_proof_refresh_status": "PASS",
        "ou_v5_proofs_refreshed": 8,
        "ou_v5_activation_automatic": False,
        "ou_v5_research_only": True,
        "candidate_promotion_authority": False,
        "order_submission_included": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    assert (
        validate_ou_v5_stage3_evidence(
            root=tmp_path,
            evidence=stage3_evidence,
        )["status"]
        == "PASS"
    )
    omitted_registration = dict(stage3_evidence)
    omitted_registration["ou_v5_prospectively_registered"] = False
    assert (
        validate_ou_v5_stage3_evidence(
            root=tmp_path,
            evidence=omitted_registration,
        )["status"]
        == "BLOCKED"
    )
    intents = list(
        (tmp_path / "data/research/wizard_ou_v5_holdout/call_attempts/2026-08-14").glob("*.json")
    )
    assert len(intents) == 8
    assert all(json.loads(path.read_text(encoding="utf-8"))["research_only"] for path in intents)
    completions = sorted(
        (tmp_path / "data/research/wizard_ou_v5_holdout/call_completions").glob("*.json")
    )
    assert len(completions) == 8
    assert result.summary["completion_receipts_valid"] == 8
    assert audit_ou_v5_capture_readiness(root=tmp_path)["status"] == "PASS"
    assert status["candidate_promotion_authority"] is False
    assert status["testnet_order_authority"] is False
    assert status["live_trading_authorized"] is False
    pre_activation_gate = _ou_v5_evidence_gate(root=tmp_path, expected_cells=8)
    assert pre_activation_gate["status"] == "BLOCKED"
    assert pre_activation_gate["evidence_generation"] == 5
    assert pre_activation_gate["formula_cells_passed"] == 8
    assert pre_activation_gate["selector_cells_passed"] == 8

    original_response = raw_paths[0].read_bytes()
    injected = json.loads(raw_paths[0].read_text(encoding="utf-8"))
    injected["injected_without_vendor_completion"] = True
    raw_paths[0].write_text(json.dumps(injected), encoding="utf-8")
    injected_binding = validate_ou_v5_evaluation_binding(root=tmp_path)
    assert injected_binding["status"] == "BLOCKED"
    assert any(
        blocker.startswith("ou_v5_call_completion_invalid:")
        for blocker in injected_binding["blockers"]
    )
    raw_paths[0].write_bytes(original_response)
    assert validate_ou_v5_evaluation_binding(root=tmp_path)["status"] == "PASS"

    _write_v5_activation_dependencies(tmp_path)
    _write_v5_refresh_proofs(tmp_path)
    supersession = build_ou_v5_supersession_gate(root=tmp_path)
    packet = build_ou_v5_review_packet(root=tmp_path)
    planned = build_reviewed_ou_v5_activation(root=tmp_path)
    no_supreme = build_reviewed_ou_v5_activation(
        root=tmp_path,
        apply=True,
        reviewer="research-reviewer",
        review_note="Reviewed every immutable OU v5 comparator cell independently.",
        review_packet_id=packet.summary["review_packet_id"],
    )
    supreme = build_ou_v5_supreme_review(root=tmp_path)
    missing_human = build_reviewed_ou_v5_activation(
        root=tmp_path,
        apply=True,
        review_packet_id=packet.summary["review_packet_id"],
    )
    applied = build_reviewed_ou_v5_activation(
        root=tmp_path,
        apply=True,
        reviewer="research-reviewer",
        review_note=(
            "Reviewed all OU v5 formula, transform, trend diagnostic, and profile branch cells."
        ),
        review_packet_id=packet.summary["review_packet_id"],
    )
    refreshed = refresh_activated_ou_v5_proofs(root=tmp_path)
    gate = _ou_v5_evidence_gate(root=tmp_path, expected_cells=8)

    assert supersession.summary["status"] == "READY_FOR_REVIEWED_SUPERSESSION"
    assert packet.summary["status"] == "READY_FOR_EXPLICIT_REVIEW"
    assert packet.summary["profile_branch_selector_cells_passed"] == 8
    assert planned.summary["status"] == "READY_REQUIRES_EXPLICIT_APPLY"
    assert no_supreme.summary["status"] == "BLOCKED"
    assert "ou_v5_supreme_review_required_for_apply" in no_supreme.summary["blockers"]
    assert supreme.summary["status"] == "PASS_ADVISORY_ONLY"
    assert supreme.summary["activation_applied_by_supreme_team"] is False
    post_mortem = next(row for row in supreme.summary["findings"] if row["lens"] == "post_mortem")
    assert "passed 8/8 cells" in post_mortem["finding"]
    assert missing_human.summary["status"] == "BLOCKED"
    assert "reviewer_required_for_ou_v5_apply" in missing_human.summary["blockers"]
    assert applied.summary["status"] == "APPLIED_RESEARCH_COMPARATOR_ONLY"
    assert applied.summary["comparator_generation"] == 5
    assert refreshed.summary["status"] == "PASS"
    assert refreshed.summary["exact_ou_rows"] == 8
    assert gate["status"] == "PASS"
    assert gate["evidence_generation"] == 5
    assert gate["comparator_generation"] == 5
    validated = load_validated_ou_v5_activation(root=tmp_path)
    assert validated is not None
    assert validated["activation_id"] == applied.summary["activation_id"]
    proofs = pd.read_csv(tmp_path / "reports/active/hyperliquid_wizard_vendor_mode_proofs.csv")
    assert proofs["formula_comparator_generation"].eq(5).all()
    assert proofs["formula_comparator_activation_id"].eq(applied.summary["activation_id"]).all()
    assert proofs["vendor_formula_parity_status"].eq("exact_reconstruction").all()
    assert {path: _sha(path) for path in raw_paths} == raw_hashes_before
    assert not applied.summary["candidate_promotion_authority"]
    assert not applied.summary["testnet_order_authority"]
    assert not applied.summary["live_trading_authorized"]

    detail_path = result.paths["detail"]
    tampered_detail = pd.read_csv(detail_path)
    tampered_detail.loc[0, "predicted_profile_branch"] = "locally_fabricated_change"
    tampered_detail.to_csv(detail_path, index=False)
    tampered_binding = validate_ou_v5_evaluation_binding(root=tmp_path)
    tampered_packet = build_ou_v5_review_packet(root=tmp_path)
    tampered_gate = _ou_v5_evidence_gate(root=tmp_path, expected_cells=8)
    assert tampered_binding["status"] == "BLOCKED"
    assert "ou_v5_evaluation_detail_hash_mismatch" in tampered_binding["blockers"]
    assert tampered_packet.summary["status"] == "BLOCKED_EVIDENCE_BINDING"
    assert "ou_v5_evaluation_binding_invalid" in tampered_packet.summary["blockers"]
    assert tampered_gate["status"] == "BLOCKED"
    assert "ou_v5_evaluation_content_binding_invalid" in tampered_gate["blocker"]


def test_v5_evaluator_rejects_injected_responses_without_call_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _register_v5_fixture(tmp_path, monkeypatch)
    for binding in contract["holdout_bindings"]:
        request = json.loads((tmp_path / binding["request_path"]).read_text(encoding="utf-8"))
        response_path = tmp_path / binding["response_path"]
        response_path.parent.mkdir(parents=True, exist_ok=True)
        response_path.write_text(
            json.dumps(_response_for_request(request)),
            encoding="utf-8",
        )

    readiness = audit_ou_v5_capture_readiness(root=tmp_path)

    assert readiness["status"] == "BLOCKED"
    assert readiness["responses_available"] == 8
    assert readiness["completion_receipts_valid"] == 0
    assert any(
        blocker.startswith("ou_v5_response_without_registered_call_intent:")
        for blocker in readiness["blockers"]
    )
    assert any(
        blocker.startswith("ou_v5_call_completion_missing:") for blocker in readiness["blockers"]
    )
    with pytest.raises(ValueError, match="OU v5 capture lineage is invalid"):
        evaluate_ou_v5_prospective_holdout(root=tmp_path)


def _write_pair_specs(root: Path) -> list[dict[str, str]]:
    specs = []
    for index, (asset_x, asset_y, scale) in enumerate(
        (("NEW1", "NEW2", 100.0), ("NEW3", "NEW4", 0.1))
    ):
        x_path = root / "candles" / f"{asset_x}.json"
        y_path = root / "candles" / f"{asset_y}.json"
        x, y = _pair_series(scale=scale, observations=140, seed=510 + index)
        _write_candles(x_path, x)
        _write_candles(y_path, y)
        specs.append(
            {
                "pair": f"{asset_x}-{asset_y}",
                "asset_x": asset_x,
                "asset_y": asset_y,
                "interval": "1h" if index else "1d",
                "asset_x_path": str(x_path.relative_to(root)),
                "asset_y_path": str(y_path.relative_to(root)),
            }
        )
    return specs


def _register_v5_fixture(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, object]:
    specs = _write_pair_specs(root)
    attribution = root / "data/research/attribution.json"
    attribution.parent.mkdir(parents=True, exist_ok=True)
    attribution.write_text('{"status":"complete"}\n', encoding="utf-8")
    monkeypatch.setattr(
        corrective_wizard_ou_v5_holdout,
        "_validated_v4_attribution",
        lambda _root: {
            "immutable_attribution_path": str(attribution.relative_to(root)),
            "immutable_attribution_sha256": _sha(attribution),
        },
    )
    monkeypatch.setattr(
        corrective_wizard_ou_v5_holdout,
        "_build_derivation",
        lambda _root: (
            pd.DataFrame(
                {
                    "transform_rule_passed": [True] * 10,
                    "profile_branch_rule_passed": [True] * 10,
                }
            ),
            0.0027,
            0.00002,
        ),
    )
    monkeypatch.setattr(
        corrective_wizard_ou_v5_holdout,
        "_ou_v5_predictor",
        _fixture_predictor,
    )
    result = register_ou_v5_prospective_holdout(
        root=root,
        pair_specs=specs,
        observations=120,
    )
    return json.loads(result.paths["contract"].read_text(encoding="utf-8"))


def _fixture_predictor(x, y, *, profile_intercept_threshold):
    high = min(float(np.median(x)), float(np.median(y))) >= 1.0
    first_is_larger = float(np.median(x)) > float(np.median(y))
    branch = (
        "zero_mean"
        if (high and first_is_larger) or (not high and not first_is_larger)
        else "intercept"
    )
    return {
        "transform": "log" if high else "level",
        "predicted_log_used": high,
        "predicted_inc_trend": bool(not high and first_is_larger),
        "predicted_profile_branch": branch,
        "profile_intercept_threshold": profile_intercept_threshold,
    }


def _response_for_request(payload: dict[str, object]) -> dict[str, object]:
    params = payload["params"]
    assert isinstance(params, dict)
    x = np.asarray(params["series_1_closes"], dtype=float)
    y = np.asarray(params["series_2_closes"], dtype=float)
    prediction = _fixture_predictor(x, y, profile_intercept_threshold=0.0027)
    if prediction["predicted_log_used"]:
        x, y = np.log(x), np.log(y)
    if prediction["predicted_profile_branch"] == "zero_mean":
        beta = float(_ou_stationary_constrained_profile_fit_v4(x, y)["beta"])
    else:
        beta = _ou_trend_aware_profile_beta_v3_candidate(x, y, inc_trend=False)
    spread = y / float(y[0]) - beta * x / float(x[0])
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
    _, phi = np.linalg.lstsq(np.column_stack([np.ones(len(lagged)), lagged]), current, rcond=None)[
        0
    ]
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


def _series(start: float, observations: int) -> np.ndarray:
    values = np.linspace(start, start * 1.2, observations)
    return values + np.sin(np.arange(observations) / 5.0) * start * 0.002


def _pair_series(*, scale: float, observations: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.empty(observations)
    residual = np.empty(observations)
    x[0] = scale
    residual[0] = scale * 0.15
    for index in range(1, observations):
        x[index] = x[index - 1] + scale * 0.001 + rng.normal(0.0, scale * 0.004)
        residual[index] = 0.94 * residual[index - 1] + rng.normal(0.0, scale * 0.003)
    y = 2.0 * x + residual
    return x, y


def _write_candles(path: Path, closes: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamps = pd.date_range("2026-01-01", periods=len(closes), freq="h", tz="UTC")
    path.write_text(
        json.dumps(
            {
                "candles": [
                    {
                        "startedAt": timestamp.isoformat(),
                        "open": float(closes[index] * 0.999),
                        "close": float(closes[index]),
                    }
                    for index, timestamp in enumerate(timestamps)
                ]
            }
        ),
        encoding="utf-8",
    )


def _write_v5_activation_dependencies(root: Path) -> None:
    active = root / "reports/active"
    active.mkdir(parents=True, exist_ok=True)
    (active / "wizard_mode_comparator_contract.csv").write_text(
        "exact_mode,version\nOU (Spread),v1\n", encoding="utf-8"
    )
    (active / "wizard_mode_comparator_contract_receipt.json").write_text(
        json.dumps({"contract_id": "wizard-comparator-v1"}),
        encoding="utf-8",
    )


def _write_v5_refresh_proofs(root: Path) -> None:
    contract = json.loads(
        (root / "config/wizard_ou_comparator_v5_holdout.json").read_text(encoding="utf-8")
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


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
