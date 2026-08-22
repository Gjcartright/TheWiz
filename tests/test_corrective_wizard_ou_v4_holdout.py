from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_platform.crypto_wizards_history import (
    CryptoWizardsCustomSeriesBacktestRequest,
)
from quant_platform.orchestration import corrective_wizard_ou_holdout
from quant_platform.orchestration.corrective_wizard_comparator_review_control import (
    build_corrective_wizard_comparator_review_control,
)
from quant_platform.orchestration.corrective_wizard_ou_v4_holdout import (
    _implementation_source_hash,
    _ou_local_transform_trend_selector_v2,
    _ou_spread_candidate_v4,
    _ou_stationary_constrained_profile_fit_v4,
    register_ou_v4_prospective_holdout,
    run_ou_v4_prospective_holdout,
)
from quant_platform.orchestration.corrective_wizard_ou_v4_supreme_review import (
    build_ou_v4_supreme_review,
)
from quant_platform.orchestration.corrective_wizard_parity import (
    _ou_v4_evidence_gate,
)
from quant_platform.wizard_hyperliquid_mode_proof import (
    refresh_activated_ou_v4_proofs,
)
from quant_platform.wizard_ou_v4_comparator_activation import (
    build_ou_v4_review_packet,
    build_ou_v4_supersession_gate,
    build_reviewed_ou_v4_activation,
    load_validated_ou_v4_activation,
)


def test_ou_v4_comparator_source_remains_frozen() -> None:
    assert _implementation_source_hash() == (
        "f7a069f0f7b1933ea6cfd7cf5639257a5e08b8f4466488f8c2ac48b988d93a5c"
    )


def test_immutable_wizard_response_publication_never_leaves_partial_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "responses" / "cell.json"

    def fail_publish(
        _source: Path,
        _destination: Path,
        *,
        follow_symlinks: bool,
    ) -> None:
        assert follow_symlinks is False
        raise OSError("simulated publication interruption")

    monkeypatch.setattr(corrective_wizard_ou_holdout.os, "link", fail_publish)

    with pytest.raises(OSError, match="simulated publication interruption"):
        corrective_wizard_ou_holdout._write_or_validate_immutable_json(
            {"history": {"spread_stats": {"hedge_ratio": 1.0}}},
            target,
        )

    assert not target.exists()
    assert list(target.parent.glob(".*.tmp")) == []


def test_ou_v4_stationary_constraint_uses_unit_root_boundary() -> None:
    rng = np.random.default_rng(811)
    normalized_x = np.empty(360, dtype=float)
    residual = np.empty(360, dtype=float)
    normalized_x[0] = 1.0
    residual[0] = -0.8
    for index in range(1, len(normalized_x)):
        normalized_x[index] = normalized_x[index - 1] + rng.normal(0.001, 0.01)
        residual[index] = residual[index - 1] + rng.normal(0.0, 0.006)
    normalized_y = 1.8 * normalized_x + residual

    fit = _ou_stationary_constrained_profile_fit_v4(
        normalized_x * 100.0,
        normalized_y * 40.0,
    )
    expected_beta = float(
        np.dot(np.diff(normalized_x), np.diff(normalized_y))
        / np.dot(np.diff(normalized_x), np.diff(normalized_x))
    )

    assert fit["unit_root_boundary_active"] is True
    assert fit["profile_phi"] == pytest.approx(1.0)
    assert fit["beta"] == pytest.approx(expected_beta, abs=1e-6)


def test_ou_v4_selector_is_request_only_and_rejects_nonpositive_prices() -> None:
    x, y = _pair_series(seed=812, observations=180)

    first = _ou_local_transform_trend_selector_v2(x, y)
    second = _ou_local_transform_trend_selector_v2(x.copy(), y.copy())

    assert first == second
    assert first["selected_selector_cell"] in {
        "level_constant",
        "level_trend",
        "log_constant",
        "log_trend",
    }
    assert float(first["aic_margin"]) >= 0.0
    with pytest.raises(ValueError, match="positive prices"):
        _ou_local_transform_trend_selector_v2(x, y - y.max())


def test_ou_v4_registration_is_prospective_immutable_and_zero_authority(
    tmp_path: Path,
) -> None:
    pair_specs = _write_v4_fixture(tmp_path)
    v3_paths = (
        tmp_path / "config/wizard_ou_comparator_v3_holdout.json",
        tmp_path / "reports/active/wizard_ou_v3_holdout_status.json",
        tmp_path / "reports/active/wizard_ou_v3_holdout_evaluation.csv",
    )
    before = {path: _sha(path) for path in v3_paths}

    result = register_ou_v4_prospective_holdout(
        root=tmp_path,
        now=datetime(2026, 8, 12, tzinfo=UTC),
        pair_specs=pair_specs,
        observations=120,
        require_full_selector_matrix=False,
    )
    repeated = register_ou_v4_prospective_holdout(
        root=tmp_path,
        pair_specs=pair_specs,
        observations=120,
        require_full_selector_matrix=False,
    )
    contract = json.loads(result.paths["contract"].read_text(encoding="utf-8"))
    predictions = pd.read_csv(result.paths["predictions"])

    assert result.summary["status"] == "REGISTERED_WAITING_VENDOR_RESPONSES"
    assert repeated.summary == result.summary
    assert len(contract["holdout_bindings"]) == 8
    assert len(predictions) == 8
    assert not predictions["vendor_response_present_at_registration"].map(bool).any()
    assert not any(
        (tmp_path / binding["response_path"]).exists() for binding in contract["holdout_bindings"]
    )
    assert result.summary["candidate_promotion_authority"] is False
    assert result.summary["testnet_order_authority"] is False
    assert result.summary["live_trading_authorized"] is False
    assert {path: _sha(path) for path in v3_paths} == before


def test_ou_v4_capture_accounts_for_eight_calls_and_passes(
    tmp_path: Path,
) -> None:
    pair_specs = _write_v4_fixture(tmp_path)
    register_ou_v4_prospective_holdout(
        root=tmp_path,
        pair_specs=pair_specs,
        observations=120,
        require_full_selector_matrix=False,
    )
    calls: list[str] = []

    def fetcher(request, *, api_key):
        calls.append(api_key)
        return _response_for_payload(request.payload())

    result = run_ou_v4_prospective_holdout(
        root=tmp_path,
        now=datetime(2026, 8, 13, tzinfo=UTC),
        execute=True,
        api_key="secret",
        fetcher=fetcher,
        credits_fetcher=lambda **_: 100,
    )
    status = json.loads(result.paths["status"].read_text(encoding="utf-8"))

    assert result.summary["status"] == "COMPLETE"
    assert result.summary["calls_made"] == 8
    assert result.summary["responses_captured"] == 8
    assert result.summary["credits_attempted"] == 16
    assert result.summary["credits_completed"] == 16
    assert result.summary["evaluation_status"] == "PASS"
    assert status["passed_cells"] == 8
    assert status["transform_selector_parity_passed_cells"] == 8
    assert status["trend_selector_parity_passed_cells"] == 8
    assert status["formula_parity_passed_cells"] == 8
    assert calls == ["secret"] * 8
    assert status["testnet_order_authority"] is False
    assert status["live_trading_authorized"] is False


def test_ou_v4_capture_resumes_only_missing_cells_after_partial_failure(
    tmp_path: Path,
) -> None:
    pair_specs = _write_v4_fixture(tmp_path)
    register_ou_v4_prospective_holdout(
        root=tmp_path,
        pair_specs=pair_specs,
        observations=120,
        require_full_selector_matrix=False,
    )
    first_cycle_calls = 0

    def fail_second_call(request, *, api_key):
        nonlocal first_cycle_calls
        first_cycle_calls += 1
        if first_cycle_calls == 2:
            raise OSError("simulated transient vendor failure")
        return _response_for_payload(request.payload())

    partial = run_ou_v4_prospective_holdout(
        root=tmp_path,
        now=datetime(2026, 8, 13, tzinfo=UTC),
        execute=True,
        api_key="secret",
        fetcher=fail_second_call,
        credits_fetcher=lambda **_: 100,
    )
    same_day_calls = 0

    def same_day_fetcher(*_args, **_kwargs):
        nonlocal same_day_calls
        same_day_calls += 1
        raise AssertionError("same-day retry must not make a vendor call")

    same_day = run_ou_v4_prospective_holdout(
        root=tmp_path,
        now=datetime(2026, 8, 13, 1, tzinfo=UTC),
        execute=True,
        api_key="secret",
        fetcher=same_day_fetcher,
        credits_fetcher=lambda **_: 102,
    )
    next_day_calls = 0

    def resume_fetcher(request, *, api_key):
        nonlocal next_day_calls
        next_day_calls += 1
        return _response_for_payload(request.payload())

    resumed = run_ou_v4_prospective_holdout(
        root=tmp_path,
        now=datetime(2026, 8, 14, tzinfo=UTC),
        execute=True,
        api_key="secret",
        fetcher=resume_fetcher,
        credits_fetcher=lambda **_: 0,
    )

    assert partial.summary["status"] == "FAILED"
    assert partial.summary["calls_made"] == 2
    assert partial.summary["responses_captured"] == 1
    assert partial.summary["responses_available"] == 1
    assert same_day.summary["status"] == "BLOCKED"
    assert same_day.summary["blocker"] == "ou_v4_daily_attempt_already_registered"
    assert same_day_calls == 0
    assert resumed.summary["status"] == "COMPLETE"
    assert resumed.summary["missing_cells_before"] == 7
    assert resumed.summary["calls_made"] == 7
    assert resumed.summary["responses_captured"] == 7
    assert resumed.summary["responses_available"] == 8
    assert resumed.summary["evaluation_status"] == "PASS"
    assert next_day_calls == 7


def test_ou_v4_uncertain_crash_blocks_same_day_duplicate_call(
    tmp_path: Path,
) -> None:
    pair_specs = _write_v4_fixture(tmp_path)
    register_ou_v4_prospective_holdout(
        root=tmp_path,
        pair_specs=pair_specs,
        observations=120,
        require_full_selector_matrix=False,
    )

    class SimulatedProcessDeath(BaseException):
        pass

    def crash_after_intent(*_args, **_kwargs):
        raise SimulatedProcessDeath("process died after request intent")

    with pytest.raises(SimulatedProcessDeath, match="process died after request intent"):
        run_ou_v4_prospective_holdout(
            root=tmp_path,
            now=datetime(2026, 8, 13, tzinfo=UTC),
            execute=True,
            api_key="secret",
            fetcher=crash_after_intent,
            credits_fetcher=lambda **_: 100,
        )

    same_day_calls = 0
    same_day_credit_checks = 0

    def forbidden_fetcher(*_args, **_kwargs):
        nonlocal same_day_calls
        same_day_calls += 1
        raise AssertionError("same-day ambiguous call must not be retried")

    def forbidden_credit_check(**_kwargs):
        nonlocal same_day_credit_checks
        same_day_credit_checks += 1
        raise AssertionError("ambiguous intent must block before credit preflight")

    blocked = run_ou_v4_prospective_holdout(
        root=tmp_path,
        now=datetime(2026, 8, 13, 1, tzinfo=UTC),
        execute=True,
        api_key="secret",
        fetcher=forbidden_fetcher,
        credits_fetcher=forbidden_credit_check,
    )
    next_day_calls = 0

    def next_day_fetcher(request, *, api_key):
        nonlocal next_day_calls
        next_day_calls += 1
        return _response_for_payload(request.payload())

    resumed = run_ou_v4_prospective_holdout(
        root=tmp_path,
        now=datetime(2026, 8, 14, tzinfo=UTC),
        execute=True,
        api_key="secret",
        fetcher=next_day_fetcher,
        credits_fetcher=lambda **_: 0,
    )

    assert blocked.summary["status"] == "BLOCKED"
    assert blocked.summary["blocker"].startswith(
        "ou_v4_call_attempt_already_registered_without_response:"
    )
    assert blocked.summary["calls_made"] == 0
    assert blocked.summary["credits_attempted"] == 0
    assert same_day_calls == 0
    assert same_day_credit_checks == 0
    assert resumed.summary["status"] == "COMPLETE"
    assert resumed.summary["responses_available"] == 8
    assert resumed.summary["evaluation_status"] == "PASS"
    assert next_day_calls == 8


def test_ou_v4_review_activation_and_refresh_are_explicit_and_immutable(
    tmp_path: Path,
) -> None:
    pair_specs = _write_v4_fixture(tmp_path)
    register_ou_v4_prospective_holdout(
        root=tmp_path,
        pair_specs=pair_specs,
        observations=120,
        require_full_selector_matrix=False,
    )
    run_ou_v4_prospective_holdout(
        root=tmp_path,
        now=datetime(2026, 8, 13, tzinfo=UTC),
        execute=True,
        api_key="secret",
        fetcher=lambda request, **_: _response_for_payload(request.payload()),
        credits_fetcher=lambda **_: 100,
    )
    _write_v4_activation_dependencies(tmp_path)
    _write_v4_refresh_proofs(tmp_path)
    raw_paths = sorted((tmp_path / "data/raw/wizard_ou_v4_holdout/responses").glob("*.json"))
    raw_hashes_before = {path: _sha(path) for path in raw_paths}

    gate = build_ou_v4_supersession_gate(root=tmp_path)
    packet = build_ou_v4_review_packet(root=tmp_path)
    planned = build_reviewed_ou_v4_activation(root=tmp_path)
    supreme = build_ou_v4_supreme_review(root=tmp_path)
    control = build_corrective_wizard_comparator_review_control(root=tmp_path)
    missing_review = build_reviewed_ou_v4_activation(
        root=tmp_path,
        apply=True,
        review_packet_id=packet.summary["review_packet_id"],
    )
    applied = build_reviewed_ou_v4_activation(
        root=tmp_path,
        apply=True,
        reviewer="research-reviewer",
        review_note=("Reviewed all OU v4 formula, transform, and trend selector cells."),
        review_packet_id=packet.summary["review_packet_id"],
    )
    refreshed = refresh_activated_ou_v4_proofs(root=tmp_path)
    parity = _ou_v4_evidence_gate(root=tmp_path, expected_cells=8)
    proofs = pd.read_csv(tmp_path / "reports/active/hyperliquid_wizard_vendor_mode_proofs.csv")

    assert gate.summary["status"] == "READY_FOR_REVIEWED_SUPERSESSION"
    assert packet.summary["status"] == "READY_FOR_EXPLICIT_REVIEW"
    assert packet.summary["formula_cells_passed"] == 8
    assert packet.summary["transform_selector_cells_passed"] == 8
    assert packet.summary["trend_selector_cells_passed"] == 8
    assert planned.summary["status"] == "READY_REQUIRES_EXPLICIT_APPLY"
    assert supreme.summary["status"] == "PASS_ADVISORY_ONLY"
    assert supreme.summary["recommendation"] == (
        "RECOMMEND_EXPLICIT_RESEARCH_COMPARATOR_ACTIVATION"
    )
    assert supreme.summary["activation_applied_by_supreme_team"] is False
    control_rows = pd.read_csv(control.paths["queue"])
    ou_control = control_rows.loc[control_rows["comparator"].eq("ou_v4")].iloc[0]
    assert bool(ou_control["review_ready"])
    assert bool(ou_control["apply_window_open"])
    assert not bool(ou_control["activation_applied"])
    assert missing_review.summary["status"] == "BLOCKED"
    assert "reviewer_required_for_ou_v4_apply" in missing_review.summary["blockers"]
    assert applied.summary["status"] == "APPLIED_RESEARCH_COMPARATOR_ONLY"
    assert refreshed.summary["status"] == "PASS"
    assert parity["status"] == "PASS"
    assert parity["evidence_generation"] == 4
    assert parity["comparator_generation"] == 4
    assert refreshed.summary["exact_ou_rows"] == 8
    assert proofs["formula_comparator_generation"].eq(4).all()
    assert proofs["formula_comparator_activation_id"].eq(applied.summary["activation_id"]).all()
    assert proofs["vendor_formula_parity_status"].eq("exact_reconstruction").all()
    assert {path: _sha(path) for path in raw_paths} == raw_hashes_before
    assert applied.summary["candidate_promotion_authority"] is False
    assert applied.summary["testnet_order_authority"] is False
    assert applied.summary["live_trading_authorized"] is False
    assert applied.summary["schema_version"] == "thewiz.wizard_ou_v4_activation.v2"
    assert applied.summary["supreme_review_id"] == supreme.summary["review_id"]
    assert applied.summary["supreme_review_path"] == supreme.summary["immutable_review_path"]
    assert applied.summary["supreme_review_sha256"] == supreme.summary["immutable_review_sha256"]
    validated = load_validated_ou_v4_activation(root=tmp_path)
    assert validated is not None
    assert validated["activation_id"] == applied.summary["activation_id"]


def test_ou_v4_activation_rejects_stale_packet_and_tampered_pointer(
    tmp_path: Path,
) -> None:
    pair_specs = _write_v4_fixture(tmp_path)
    register_ou_v4_prospective_holdout(
        root=tmp_path,
        pair_specs=pair_specs,
        observations=120,
        require_full_selector_matrix=False,
    )
    run_ou_v4_prospective_holdout(
        root=tmp_path,
        now=datetime(2026, 8, 13, tzinfo=UTC),
        execute=True,
        api_key="secret",
        fetcher=lambda request, **_: _response_for_payload(request.payload()),
        credits_fetcher=lambda **_: 100,
    )
    _write_v4_activation_dependencies(tmp_path)
    _write_v4_refresh_proofs(tmp_path)
    packet = build_ou_v4_review_packet(root=tmp_path)
    build_ou_v4_supreme_review(root=tmp_path)

    stale = build_reviewed_ou_v4_activation(
        root=tmp_path,
        apply=True,
        reviewer="research-reviewer",
        review_note="Reviewed every immutable OU v4 parity cell independently.",
        review_packet_id="stale-packet",
    )
    assert stale.summary["status"] == "BLOCKED"
    assert "review_packet_id_does_not_match_current_ou_v4_evidence" in stale.summary["blockers"]

    build_reviewed_ou_v4_activation(
        root=tmp_path,
        apply=True,
        reviewer="research-reviewer",
        review_note="Reviewed every immutable OU v4 parity cell independently.",
        review_packet_id=packet.summary["review_packet_id"],
    )
    status_path = tmp_path / "reports/active/wizard_ou_v4_activation_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["immutable_activation_sha256"] = "0" * 64
    status_path.write_text(json.dumps(status), encoding="utf-8")

    with pytest.raises(ValueError, match="immutable activation binding mismatch"):
        refresh_activated_ou_v4_proofs(root=tmp_path)


def test_ou_v4_direct_apply_requires_current_immutable_supreme_review(
    tmp_path: Path,
) -> None:
    pair_specs = _write_v4_fixture(tmp_path)
    register_ou_v4_prospective_holdout(
        root=tmp_path,
        pair_specs=pair_specs,
        observations=120,
        require_full_selector_matrix=False,
    )
    run_ou_v4_prospective_holdout(
        root=tmp_path,
        now=datetime(2026, 8, 13, tzinfo=UTC),
        execute=True,
        api_key="secret",
        fetcher=lambda request, **_: _response_for_payload(request.payload()),
        credits_fetcher=lambda **_: 100,
    )
    _write_v4_activation_dependencies(tmp_path)
    _write_v4_refresh_proofs(tmp_path)
    packet = build_ou_v4_review_packet(root=tmp_path)

    result = build_reviewed_ou_v4_activation(
        root=tmp_path,
        apply=True,
        reviewer="research-reviewer",
        review_note="Reviewed every immutable OU v4 parity cell independently.",
        review_packet_id=packet.summary["review_packet_id"],
    )

    assert result.summary["status"] == "BLOCKED"
    assert "ou_v4_supreme_review_required_for_apply" in result.summary["blockers"]
    assert "activation_id" not in result.summary
    assert result.summary["candidate_promotion_authority"] is False
    assert result.summary["testnet_order_authority"] is False
    assert result.summary["live_trading_authorized"] is False


def test_ou_v4_direct_apply_rejects_rehashed_tampered_supreme_review(
    tmp_path: Path,
) -> None:
    pair_specs = _write_v4_fixture(tmp_path)
    register_ou_v4_prospective_holdout(
        root=tmp_path,
        pair_specs=pair_specs,
        observations=120,
        require_full_selector_matrix=False,
    )
    run_ou_v4_prospective_holdout(
        root=tmp_path,
        now=datetime(2026, 8, 13, tzinfo=UTC),
        execute=True,
        api_key="secret",
        fetcher=lambda request, **_: _response_for_payload(request.payload()),
        credits_fetcher=lambda **_: 100,
    )
    _write_v4_activation_dependencies(tmp_path)
    _write_v4_refresh_proofs(tmp_path)
    packet = build_ou_v4_review_packet(root=tmp_path)
    supreme = build_ou_v4_supreme_review(root=tmp_path)
    immutable_path = supreme.paths["immutable_review"]
    immutable = json.loads(immutable_path.read_text(encoding="utf-8"))
    immutable["recommendation"] = "DO_NOT_ACTIVATE"
    immutable_path.write_text(
        json.dumps(immutable, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    status_path = supreme.paths["status"]
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["immutable_review_sha256"] = _sha(immutable_path)
    status_path.write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    result = build_reviewed_ou_v4_activation(
        root=tmp_path,
        apply=True,
        reviewer="research-reviewer",
        review_note="Reviewed every immutable OU v4 parity cell independently.",
        review_packet_id=packet.summary["review_packet_id"],
    )

    assert result.summary["status"] == "BLOCKED"
    assert "ou_v4_supreme_review_invalid_for_apply" in result.summary["blockers"]
    assert "activation_id" not in result.summary
    assert result.summary["testnet_order_authority"] is False
    assert result.summary["live_trading_authorized"] is False


def test_ou_v4_loader_rejects_rehashed_activation_content(
    tmp_path: Path,
) -> None:
    pair_specs = _write_v4_fixture(tmp_path)
    register_ou_v4_prospective_holdout(
        root=tmp_path,
        pair_specs=pair_specs,
        observations=120,
        require_full_selector_matrix=False,
    )
    run_ou_v4_prospective_holdout(
        root=tmp_path,
        now=datetime(2026, 8, 13, tzinfo=UTC),
        execute=True,
        api_key="secret",
        fetcher=lambda request, **_: _response_for_payload(request.payload()),
        credits_fetcher=lambda **_: 100,
    )
    _write_v4_activation_dependencies(tmp_path)
    _write_v4_refresh_proofs(tmp_path)
    packet = build_ou_v4_review_packet(root=tmp_path)
    build_ou_v4_supreme_review(root=tmp_path)
    applied = build_reviewed_ou_v4_activation(
        root=tmp_path,
        apply=True,
        reviewer="research-reviewer",
        review_note="Reviewed every immutable OU v4 parity cell independently.",
        review_packet_id=packet.summary["review_packet_id"],
    )
    immutable_path = applied.paths["immutable_activation"]
    immutable = json.loads(immutable_path.read_text(encoding="utf-8"))
    immutable["review_note"] = "A different but still substantive review note was inserted."
    immutable_path.write_text(
        json.dumps(immutable, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    status_path = applied.paths["activation_status"]
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["review_note"] = immutable["review_note"]
    status["immutable_activation_sha256"] = _sha(immutable_path)
    status_path.write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="immutable activation identity mismatch"):
        load_validated_ou_v4_activation(root=tmp_path)


def test_ou_v4_selector_mismatch_has_independent_failure_attribution(
    tmp_path: Path,
) -> None:
    pair_specs = _write_v4_fixture(tmp_path)
    register_ou_v4_prospective_holdout(
        root=tmp_path,
        pair_specs=pair_specs,
        observations=120,
        require_full_selector_matrix=False,
    )
    calls = 0

    def fetcher(request, *, api_key):
        nonlocal calls
        del api_key
        calls += 1
        response = _response_for_payload(request.payload())
        if calls == 1:
            current = bool(response["history"]["coint_eg"]["inc_trend"])
            response["history"]["coint_eg"]["inc_trend"] = not current
        return response

    result = run_ou_v4_prospective_holdout(
        root=tmp_path,
        now=datetime(2026, 8, 13, tzinfo=UTC),
        execute=True,
        api_key="secret",
        fetcher=fetcher,
        credits_fetcher=lambda **_: 100,
    )
    detail = pd.read_csv(result.paths["detail"])

    assert result.summary["evaluation_status"] == "FAIL"
    assert int(detail["formula_parity_passed"].map(bool).sum()) == 8
    assert int(detail["trend_selector_parity_passed"].map(bool).sum()) == 7
    assert detail["blocker"].fillna("").str.contains("trend_selector_mismatch").sum() == 1
    assert result.summary["testnet_order_authority"] is False
    assert result.summary["live_trading_authorized"] is False


def test_ou_v4_tampered_predictions_block_before_vendor_call(
    tmp_path: Path,
) -> None:
    pair_specs = _write_v4_fixture(tmp_path)
    registration = register_ou_v4_prospective_holdout(
        root=tmp_path,
        pair_specs=pair_specs,
        observations=120,
        require_full_selector_matrix=False,
    )
    registration.paths["predictions"].write_text(
        registration.paths["predictions"].read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    called = False

    def fetcher(*args, **kwargs):
        nonlocal called
        called = True
        return {}

    with pytest.raises(ValueError, match="frozen predictions binding mismatch"):
        run_ou_v4_prospective_holdout(
            root=tmp_path,
            execute=True,
            api_key="secret",
            fetcher=fetcher,
            credits_fetcher=lambda **_: 0,
        )
    assert called is False


def _write_v4_fixture(root: Path) -> list[dict[str, str]]:
    config = root / "config"
    active = root / "reports" / "active"
    config.mkdir(parents=True, exist_ok=True)
    active.mkdir(parents=True, exist_ok=True)
    cohorts: list[list[dict[str, object]]] = []
    response_hashes: list[str] = []
    for cohort_index in range(3):
        x, y = _pair_series(seed=900 + cohort_index, observations=140)
        bindings: list[dict[str, object]] = []
        for exact_mode, strategy in (
            ("OU (Spread)", "Spread"),
            ("OU (ZScoreR)", "ZScoreRoll"),
        ):
            for orientation in ("original", "reverse"):
                first, second = (x, y) if orientation == "original" else (y, x)
                request = _request(first, second, strategy=strategy)
                response = _response_for_payload(request)
                stem = f"c{cohort_index}_{strategy}_{orientation}".lower()
                request_path = root / "data" / f"{stem}_request.json"
                response_path = root / "data" / f"{stem}_response.json"
                request_path.parent.mkdir(parents=True, exist_ok=True)
                request_path.write_text(json.dumps(request), encoding="utf-8")
                response_path.write_text(json.dumps(response), encoding="utf-8")
                binding = {
                    "exact_mode": exact_mode,
                    "orientation": orientation,
                    "proof_observations": len(first),
                    "request_path": str(request_path.relative_to(root)),
                    "request_sha256": _sha(request_path),
                    "response_path": str(response_path.relative_to(root)),
                }
                if cohort_index < 2:
                    binding["response_sha256"] = _sha(response_path)
                else:
                    binding["pair"] = "OLD5-OLD6" if orientation == "original" else "OLD6-OLD5"
                    response_hashes.append(_sha(response_path))
                bindings.append(binding)
        cohorts.append(bindings)
    (config / "wizard_ou_comparator_v2_holdout.json").write_text(
        json.dumps(
            {
                "derivation_bindings": cohorts[0],
                "sealed_holdout_bindings": cohorts[1],
            }
        ),
        encoding="utf-8",
    )
    (config / "wizard_ou_comparator_v3_holdout.json").write_text(
        json.dumps({"holdout_bindings": cohorts[2]}),
        encoding="utf-8",
    )
    immutable_v3 = root / "data" / "research" / "v3_immutable.json"
    immutable_v3.parent.mkdir(parents=True, exist_ok=True)
    immutable_v3.write_text(json.dumps({"status": "FAIL"}), encoding="utf-8")
    (active / "wizard_ou_v3_holdout_status.json").write_text(
        json.dumps(
            {
                "response_sha256s": sorted(response_hashes),
                "immutable_result_path": str(immutable_v3.relative_to(root)),
                "immutable_result_sha256": _sha(immutable_v3),
            }
        ),
        encoding="utf-8",
    )
    (active / "wizard_ou_v3_holdout_evaluation.csv").write_text(
        "cell_status\nFAIL\n",
        encoding="utf-8",
    )

    specs: list[dict[str, str]] = []
    for index, (asset_x, asset_y) in enumerate((("NEW1", "NEW2"), ("NEW3", "NEW4"))):
        x, y = _pair_series(seed=950 + index, observations=160)
        x_path = root / "candles" / f"{asset_x}.json"
        y_path = root / "candles" / f"{asset_y}.json"
        _write_candles(x_path, x)
        _write_candles(y_path, y)
        specs.append(
            {
                "pair": f"{asset_x}-{asset_y}",
                "asset_x": asset_x,
                "asset_y": asset_y,
                "interval": "1h",
                "asset_x_path": str(x_path.relative_to(root)),
                "asset_y_path": str(y_path.relative_to(root)),
            }
        )
    return specs


def _pair_series(*, seed: int, observations: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.empty(observations, dtype=float)
    residual = np.empty(observations, dtype=float)
    x[0] = 100.0
    residual[0] = 20.0
    for index in range(1, observations):
        x[index] = x[index - 1] + 0.08 + rng.normal(0.0, 0.45)
        residual[index] = 0.92 * residual[index - 1] + rng.normal(0.0, 0.35)
    y = 0.8 * x + residual
    return x, y


def _request(x: np.ndarray, y: np.ndarray, *, strategy: str) -> dict[str, object]:
    return CryptoWizardsCustomSeriesBacktestRequest(
        series_1_opens=tuple(float(value * 0.999) for value in x),
        series_1_closes=tuple(float(value) for value in x),
        series_2_opens=tuple(float(value * 0.999) for value in y),
        series_2_closes=tuple(float(value) for value in y),
        strategy=strategy,
        spread_type="Ou",
        roll_w=42,
        entry_level=2.0,
        exit_level=0.0,
        x_weighting=0.5,
        slippage_rate=0.0005,
        commission_rate=0.001,
        with_history=True,
    ).payload()


def _response_for_payload(payload: dict[str, object]) -> dict[str, object]:
    params = payload["params"]
    assert isinstance(params, dict)
    x = np.asarray(params["series_1_closes"], dtype=float)
    y = np.asarray(params["series_2_closes"], dtype=float)
    selector = _ou_local_transform_trend_selector_v2(x, y)
    log_used = bool(selector["local_predicted_log_used"])
    inc_trend = bool(selector["local_predicted_inc_trend"])
    if log_used:
        x, y = np.log(x), np.log(y)
    beta, spread, _ = _ou_spread_candidate_v4(x, y, inc_trend=inc_trend)
    zscore = (spread - spread.mean()) / spread.std(ddof=1)
    series = pd.Series(spread)
    window = int(params["roll_w"])
    zscore_roll = (
        series.sub(series.rolling(window, min_periods=window).mean())
        .div(series.rolling(window, min_periods=window).std(ddof=1))
        .fillna(0.0)
        .to_numpy()
    )
    _, phi = np.linalg.lstsq(
        np.column_stack([np.ones(len(spread) - 1), spread[:-1]]),
        spread[1:],
        rcond=None,
    )[0]
    assert 0.0 < phi < 1.0
    return {
        "history": {
            "coint_eg": {"inc_trend": inc_trend},
            "spread_stats": {
                "spread": spread.tolist(),
                "zscore": zscore.tolist(),
                "zscore_roll": zscore_roll.tolist(),
                "hedge_ratio": beta,
                "half_life": float(-np.log(2.0) / np.log(phi)),
                "log_used": log_used,
            },
        }
    }


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


def _write_v4_activation_dependencies(root: Path) -> None:
    active = root / "reports" / "active"
    (active / "wizard_mode_comparator_contract.csv").write_text(
        "exact_mode,version\nOU (Spread),v1\n", encoding="utf-8"
    )
    (active / "wizard_mode_comparator_contract_receipt.json").write_text(
        json.dumps({"contract_id": "wizard-comparator-v1"}),
        encoding="utf-8",
    )


def _write_v4_refresh_proofs(root: Path) -> None:
    contract = json.loads(
        (root / "config/wizard_ou_comparator_v4_holdout.json").read_text(encoding="utf-8")
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
