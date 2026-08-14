from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_platform import wizard_ou_comparator_activation
from quant_platform.orchestration.corrective_wizard_ou_holdout import (
    _implementation_source_hash,
    _ou_local_trend_selector_v1,
    _ou_trend_selector_source_hash,
    _v3_implementation_source_hash,
    build_ou_v3_review_handoff,
    build_ou_v3_reviewed_activation,
    build_ou_v3_supersession_gate,
    evaluate_ou_v2_blind_holdout,
    register_ou_trend_selector_v1_holdout,
    register_ou_v2_blind_holdout,
    register_ou_v3_prospective_holdout,
    run_ou_v3_prospective_holdout,
)
from quant_platform.wizard_hyperliquid_mode_proof import (
    _ou_trend_aware_profile_beta_v3_candidate,
    _ou_trend_aware_profile_spread_v3_candidate,
    _ou_zero_mean_profile_beta_v2_holdout,
    _ou_zero_mean_profile_spread_candidate_v2_holdout,
    refresh_activated_ou_v3_proofs,
)
from quant_platform.wizard_ou_comparator_activation import (
    load_validated_ou_v3_activation,
)


def test_registered_ou_v2_v3_and_selector_sources_remain_frozen() -> None:
    assert _implementation_source_hash() == (
        "32c915b5293e20ea81c12d446a7aa53dab9034f48c2fdffece6050521f32b38f"
    )
    assert _v3_implementation_source_hash() == (
        "010d9d41699386e2da4b42f2c6b67aaa1bb41dd7f7d902450b68c0d8f41489a4"
    )
    assert _ou_trend_selector_source_hash() == (
        "fabdbc1e0999ebda21c49ccc0cd4f6ebfc9e1c8ebd0194a5ed23ea0fa9fd1da9"
    )


def _write_fixture(root: Path, *, break_holdout: bool = False) -> Path:
    rows: list[dict[str, object]] = []
    modes = ("OU (Spread)", "OU (ZScoreR)")
    orientations = ("original", "reverse")
    for group_index, group in enumerate(("dev", "holdout")):
        for mode in modes:
            for orientation in orientations:
                rng = np.random.default_rng(100 + group_index)
                x = np.empty(160, dtype=float)
                residual = np.empty(160, dtype=float)
                x[0] = 1.0
                residual[0] = 0.2
                for index in range(1, len(x)):
                    x[index] = x[index - 1] + 0.002 + rng.normal(0.0, 0.006)
                    residual[index] = 0.94 * residual[index - 1] + rng.normal(0.0, 0.008)
                y = 0.8 * x + residual
                if orientation == "reverse":
                    x, y = y, x
                beta = _ou_zero_mean_profile_beta_v2_holdout(x, y)
                spread = _ou_zero_mean_profile_spread_candidate_v2_holdout(x, y)
                zscore = (spread - spread.mean()) / spread.std(ddof=1)
                spread_series = pd.Series(spread)
                rolling = (
                    spread_series.sub(spread_series.rolling(20, min_periods=20).mean())
                    .div(spread_series.rolling(20, min_periods=20).std(ddof=1))
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
                if break_holdout and group == "holdout" and mode == modes[0] and orientation == "original":
                    spread = spread + 0.01
                stem = f"{group}_{mode}_{orientation}".lower().replace(" ", "_").replace("(", "").replace(")", "")
                request_path = root / "data" / f"{stem}_request.json"
                response_path = root / "data" / f"{stem}_response.json"
                request_path.parent.mkdir(parents=True, exist_ok=True)
                request_path.write_text(
                    json.dumps(
                        {
                            "params": {
                                "series_1_closes": x.tolist(),
                                "series_2_closes": y.tolist(),
                                "roll_w": 20,
                            }
                        }
                    ),
                    encoding="utf-8",
                )
                response_path.write_text(
                    json.dumps(
                        {
                            "history": {
                                "coint_eg": {
                                    "inc_trend": bool(
                                        _ou_local_trend_selector_v1(x, y)[
                                            "local_predicted_inc_trend"
                                        ]
                                    )
                                },
                                "spread_stats": {
                                    "spread": spread.tolist(),
                                    "zscore": zscore.tolist(),
                                    "zscore_roll": rolling.tolist(),
                                    "hedge_ratio": beta,
                                    "half_life": half_life,
                                    "log_used": False,
                                }
                            }
                        }
                    ),
                    encoding="utf-8",
                )
                rows.append(
                    {
                        "pair_group_id": group,
                        "exact_mode": mode,
                        "orientation": orientation,
                        "proof_observations": len(x),
                        "vendor_response_captured": True,
                        "request_path": request_path.relative_to(root),
                        "response_path": response_path.relative_to(root),
                    }
                )
    proof_path = root / "reports" / "active" / "hyperliquid_wizard_vendor_mode_proofs.csv"
    proof_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(proof_path, index=False)
    return proof_path


def test_ou_v2_blind_holdout_passes_all_four_registered_cells(tmp_path: Path) -> None:
    proof_path = _write_fixture(tmp_path)
    registration = register_ou_v2_blind_holdout(
        root=tmp_path,
        now=datetime(2026, 8, 11, tzinfo=UTC),
        proof_path=proof_path,
        derivation_group="dev",
        holdout_group="holdout",
    )

    result = evaluate_ou_v2_blind_holdout(
        root=tmp_path,
        now=datetime(2026, 8, 11, 1, tzinfo=UTC),
        proof_path=proof_path,
    )

    assert registration.summary["status"] == "REGISTERED_BLINDED"
    assert result.summary["status"] == "PASS"
    assert result.summary["passed_cells"] == 4
    assert result.summary["comparator_supersession_eligible"] is True
    assert result.summary["testnet_order_authority"] is False
    assert result.summary["live_trading_authorized"] is False


def test_ou_v2_rejects_response_changed_after_registration(tmp_path: Path) -> None:
    proof_path = _write_fixture(tmp_path)
    register_ou_v2_blind_holdout(
        root=tmp_path,
        proof_path=proof_path,
        derivation_group="dev",
        holdout_group="holdout",
    )
    proofs = pd.read_csv(proof_path)
    target = tmp_path / proofs.loc[proofs["pair_group_id"].eq("holdout"), "response_path"].iloc[0]
    target.write_text(target.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="sealed response hash mismatch"):
        evaluate_ou_v2_blind_holdout(root=tmp_path, proof_path=proof_path)


def test_ou_v2_holdout_failure_remains_research_only(tmp_path: Path) -> None:
    proof_path = _write_fixture(tmp_path, break_holdout=True)
    register_ou_v2_blind_holdout(
        root=tmp_path,
        proof_path=proof_path,
        derivation_group="dev",
        holdout_group="holdout",
    )

    result = evaluate_ou_v2_blind_holdout(root=tmp_path, proof_path=proof_path)

    assert result.summary["status"] == "FAIL"
    assert result.summary["failed_cells"] == 1
    assert result.summary["comparator_supersession_eligible"] is False
    assert result.summary["candidate_promotion_authority"] is False


def test_ou_v2_requires_disjoint_cohorts(tmp_path: Path) -> None:
    proof_path = _write_fixture(tmp_path)

    with pytest.raises(ValueError, match="disjoint"):
        register_ou_v2_blind_holdout(
            root=tmp_path,
            proof_path=proof_path,
            derivation_group="dev",
            holdout_group="dev",
        )


def test_ou_v3_trend_branch_changes_profile_objective() -> None:
    rng = np.random.default_rng(44)
    x = np.cumsum(rng.normal(0.01, 0.03, 240)) + 5.0
    y = 0.85 * x + np.cumsum(rng.normal(0.0, 0.02, 240)) + 1.0

    with_trend = _ou_trend_aware_profile_beta_v3_candidate(
        x, y, inc_trend=True
    )
    without_trend = _ou_trend_aware_profile_beta_v3_candidate(
        x, y, inc_trend=False
    )

    assert with_trend == pytest.approx(
        _ou_zero_mean_profile_beta_v2_holdout(x, y)
    )
    assert abs(with_trend - without_trend) > 1e-5


def test_ou_v3_registration_freezes_four_swapped_requests_without_responses(
    tmp_path: Path,
) -> None:
    result = _register_v3_fixture(tmp_path)
    contract = json.loads(result.paths["contract"].read_text(encoding="utf-8"))
    bindings = contract["holdout_bindings"]
    requests = {
        (binding["exact_mode"], binding["orientation"]): json.loads(
            (tmp_path / binding["request_path"]).read_text(encoding="utf-8")
        )
        for binding in bindings
    }

    assert result.summary["status"] == "REGISTERED_WAITING_VENDOR_RESPONSES"
    assert len(bindings) == 4
    assert contract["vendor_responses_at_registration"] == 0
    assert not any(binding["response_captured_at_registration"] for binding in bindings)
    for mode in ("OU (Spread)", "OU (ZScoreR)"):
        original = requests[(mode, "original")]["params"]
        reverse = requests[(mode, "reverse")]["params"]
        assert original["series_1_closes"] == reverse["series_2_closes"]
        assert original["series_2_closes"] == reverse["series_1_closes"]
    assert result.summary["testnet_order_authority"] is False
    assert result.summary["live_trading_authorized"] is False


def test_ou_v3_guarded_capture_accounts_for_four_calls_and_evaluates(
    tmp_path: Path,
) -> None:
    _register_v3_fixture(tmp_path)
    calls: list[str] = []

    def fetcher(request, *, api_key):
        calls.append(api_key)
        params = request.payload()["params"]
        x = np.asarray(params["series_1_closes"], dtype=float)
        y = np.asarray(params["series_2_closes"], dtype=float)
        inc_trend = bool(
            _ou_local_trend_selector_v1(x, y)["local_predicted_inc_trend"]
        )
        beta = _ou_trend_aware_profile_beta_v3_candidate(
            x, y, inc_trend=inc_trend
        )
        spread = _ou_trend_aware_profile_spread_v3_candidate(
            x, y, inc_trend=inc_trend
        )
        zscore = (spread - spread.mean()) / spread.std(ddof=1)
        series = pd.Series(spread)
        zscore_roll = (
            series.sub(series.rolling(42, min_periods=42).mean())
            .div(series.rolling(42, min_periods=42).std(ddof=1))
            .fillna(0.0)
            .to_numpy()
        )
        lagged = spread[:-1]
        current = spread[1:]
        _, phi = np.linalg.lstsq(
            np.column_stack([np.ones(len(lagged)), lagged]), current, rcond=None
        )[0]
        return {
            "history": {
                "coint_eg": {"inc_trend": inc_trend},
                "spread_stats": {
                    "spread": spread.tolist(),
                    "zscore": zscore.tolist(),
                    "zscore_roll": zscore_roll.tolist(),
                    "hedge_ratio": beta,
                    "half_life": float(-np.log(2.0) / np.log(phi)),
                    "log_used": False,
                },
            }
        }

    result = run_ou_v3_prospective_holdout(
        root=tmp_path,
        now=datetime(2026, 8, 12, tzinfo=UTC),
        execute=True,
        api_key="secret",
        fetcher=fetcher,
        credits_fetcher=lambda **_: 0,
    )

    assert result.summary["status"] == "COMPLETE"
    assert result.summary["calls_made"] == 4
    assert result.summary["responses_captured"] == 4
    assert result.summary["credits_attempted"] == 8
    assert result.summary["credits_completed"] == 8
    assert result.summary["evaluation_status"] == "PASS"
    assert calls == ["secret"] * 4
    assert result.summary["testnet_order_authority"] is False
    assert result.summary["live_trading_authorized"] is False


def test_ou_trend_selector_registration_freezes_predictions_before_responses(
    tmp_path: Path,
) -> None:
    _register_v3_fixture(tmp_path)

    result = register_ou_trend_selector_v1_holdout(root=tmp_path)
    predictions = pd.read_csv(result.paths["predictions"])
    derivation = pd.read_csv(result.paths["derivation"])

    assert result.summary["status"] == "REGISTERED_WAITING_VENDOR_RESPONSES"
    assert result.summary["derivation_matched_cells"] == 8
    assert result.summary["holdout_prediction_cells"] == 4
    assert result.summary["vendor_responses_at_registration"] == 0
    assert len(predictions) == 4
    assert not predictions["vendor_response_present_at_registration"].map(bool).any()
    assert len(derivation) == 8
    assert derivation["selector_match"].map(bool).all()
    assert result.summary["testnet_order_authority"] is False
    assert result.summary["live_trading_authorized"] is False


def test_ou_v3_selector_mismatch_fails_even_when_local_formula_matches(
    tmp_path: Path,
) -> None:
    _register_v3_fixture(tmp_path)
    calls = 0

    def fetcher(request, *, api_key):
        nonlocal calls
        del api_key
        calls += 1
        params = request.payload()["params"]
        x = np.asarray(params["series_1_closes"], dtype=float)
        y = np.asarray(params["series_2_closes"], dtype=float)
        local_inc_trend = bool(
            _ou_local_trend_selector_v1(x, y)["local_predicted_inc_trend"]
        )
        beta = _ou_trend_aware_profile_beta_v3_candidate(
            x, y, inc_trend=local_inc_trend
        )
        spread = _ou_trend_aware_profile_spread_v3_candidate(
            x, y, inc_trend=local_inc_trend
        )
        zscore = (spread - spread.mean()) / spread.std(ddof=1)
        series = pd.Series(spread)
        zscore_roll = (
            series.sub(series.rolling(42, min_periods=42).mean())
            .div(series.rolling(42, min_periods=42).std(ddof=1))
            .fillna(0.0)
            .to_numpy()
        )
        lagged = spread[:-1]
        current = spread[1:]
        _, phi = np.linalg.lstsq(
            np.column_stack([np.ones(len(lagged)), lagged]), current, rcond=None
        )[0]
        return {
            "history": {
                "coint_eg": {
                    "inc_trend": (
                        not local_inc_trend if calls == 1 else local_inc_trend
                    )
                },
                "spread_stats": {
                    "spread": spread.tolist(),
                    "zscore": zscore.tolist(),
                    "zscore_roll": zscore_roll.tolist(),
                    "hedge_ratio": beta,
                    "half_life": float(-np.log(2.0) / np.log(phi)),
                    "log_used": False,
                },
            }
        }

    result = run_ou_v3_prospective_holdout(
        root=tmp_path,
        now=datetime(2026, 8, 12, tzinfo=UTC),
        execute=True,
        api_key="secret",
        fetcher=fetcher,
        credits_fetcher=lambda **_: 0,
    )

    assert result.summary["evaluation_status"] == "FAIL"
    status = json.loads(
        (tmp_path / "reports/active/wizard_ou_v3_holdout_status.json").read_text(
            encoding="utf-8"
        )
    )
    detail = pd.read_csv(tmp_path / status["evidence_path"])
    assert status["formula_parity_passed_cells"] == 4
    assert status["trend_selector_parity_passed_cells"] == 3
    assert status["local_point_in_time_trend_selector_proven"] is False
    assert detail["formula_parity_passed"].map(bool).all()
    assert int(detail["trend_selector_parity_passed"].map(bool).sum()) == 3
    assert status["testnet_order_authority"] is False
    assert status["live_trading_authorized"] is False


def test_ou_v3_tampered_selector_prediction_blocks_before_capture(
    tmp_path: Path,
) -> None:
    _register_v3_fixture(tmp_path)
    predictions_path = (
        tmp_path / "reports/active/wizard_ou_trend_selector_v1_predictions.csv"
    )
    predictions_path.write_text(
        predictions_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    called = False

    def fetcher(*args, **kwargs):
        nonlocal called
        called = True
        return {}

    with pytest.raises(ValueError, match="holdout_predictions evidence hash mismatch"):
        run_ou_v3_prospective_holdout(
            root=tmp_path,
            execute=True,
            api_key="secret",
            fetcher=fetcher,
            credits_fetcher=lambda **_: 0,
        )
    assert called is False


def test_ou_v3_review_activation_and_refresh_are_explicit_and_immutable(
    tmp_path: Path,
) -> None:
    _register_v3_fixture(tmp_path)
    _complete_v3_holdout(tmp_path)
    _write_activation_dependencies(tmp_path)
    _write_v3_refresh_proofs(tmp_path)
    raw_paths = sorted(
        (tmp_path / "data" / "raw" / "wizard_ou_v3_holdout" / "responses").glob(
            "*.json"
        )
    )
    raw_hashes_before = {path: _sha(path) for path in raw_paths}

    gate = build_ou_v3_supersession_gate(root=tmp_path)
    packet = build_ou_v3_review_handoff(root=tmp_path)
    planned = build_ou_v3_reviewed_activation(root=tmp_path)
    missing_review = build_ou_v3_reviewed_activation(
        root=tmp_path,
        apply=True,
        review_packet_id=packet.summary["review_packet_id"],
    )
    applied = build_ou_v3_reviewed_activation(
        root=tmp_path,
        apply=True,
        reviewer="research-reviewer",
        review_note="Reviewed OU formula and point-in-time selector independently.",
        review_packet_id=packet.summary["review_packet_id"],
    )
    refreshed = refresh_activated_ou_v3_proofs(root=tmp_path)
    proofs = pd.read_csv(
        tmp_path / "reports/active/hyperliquid_wizard_vendor_mode_proofs.csv"
    )

    assert gate.summary["status"] == "READY_FOR_REVIEWED_SUPERSESSION"
    assert packet.summary["status"] == "READY_FOR_EXPLICIT_REVIEW"
    assert packet.summary["formula_cells_passed"] == 4
    assert packet.summary["selector_cells_passed"] == 4
    assert planned.summary["status"] == "READY_REQUIRES_EXPLICIT_APPLY"
    assert missing_review.summary["status"] == "BLOCKED"
    assert "reviewer_required_for_ou_v3_apply" in missing_review.summary["blockers"]
    assert applied.summary["status"] == "APPLIED_RESEARCH_COMPARATOR_ONLY"
    assert refreshed.summary["status"] == "PASS"
    assert refreshed.summary["exact_ou_rows"] == 4
    assert proofs["formula_comparator_generation"].eq(3).all()
    assert proofs["formula_comparator_activation_id"].eq(
        applied.summary["activation_id"]
    ).all()
    assert proofs["vendor_formula_parity_status"].eq(
        "exact_reconstruction"
    ).all()
    assert {path: _sha(path) for path in raw_paths} == raw_hashes_before
    assert applied.summary["candidate_promotion_authority"] is False
    assert applied.summary["testnet_order_authority"] is False
    assert applied.summary["live_trading_authorized"] is False
    validated = load_validated_ou_v3_activation(
        root=tmp_path,
        implementation_source_sha256=applied.summary[
            "implementation_source_sha256"
        ],
        selector_source_sha256=applied.summary["selector_source_sha256"],
    )
    assert validated is not None
    assert validated["activation_id"] == applied.summary["activation_id"]


def test_ou_v3_activation_waits_for_frozen_capture_reconciliation(
    tmp_path: Path, monkeypatch
) -> None:
    _register_v3_fixture(tmp_path)
    _complete_v3_holdout(tmp_path)
    _write_activation_dependencies(tmp_path)
    _write_v3_refresh_proofs(tmp_path)
    build_ou_v3_supersession_gate(root=tmp_path)
    packet = build_ou_v3_review_handoff(root=tmp_path)
    blocker = "frozen_capture_manifest_unresolved:wizardcapture_test"
    monkeypatch.setattr(
        wizard_ou_comparator_activation,
        "frozen_capture_manifest_mutation_blockers",
        lambda *, root: [blocker],
    )

    result = build_ou_v3_reviewed_activation(
        root=tmp_path,
        apply=True,
        reviewer="research-reviewer",
        review_note="Reviewed OU formula and point-in-time selector independently.",
        review_packet_id=packet.summary["review_packet_id"],
    )

    assert result.summary["status"] == "BLOCKED"
    assert blocker in result.summary["blockers"]
    assert "immutable_activation" not in result.paths


def test_ou_v3_activation_rejects_stale_packet_and_tampered_pointer(
    tmp_path: Path,
) -> None:
    _register_v3_fixture(tmp_path)
    _complete_v3_holdout(tmp_path)
    _write_activation_dependencies(tmp_path)
    _write_v3_refresh_proofs(tmp_path)
    build_ou_v3_supersession_gate(root=tmp_path)
    packet = build_ou_v3_review_handoff(root=tmp_path)

    stale = build_ou_v3_reviewed_activation(
        root=tmp_path,
        apply=True,
        reviewer="research-reviewer",
        review_note="Reviewed both independent OU parity claims before activation.",
        review_packet_id="stale-packet",
    )
    assert stale.summary["status"] == "BLOCKED"
    assert "review_packet_id_does_not_match_current_ou_v3_evidence" in stale.summary[
        "blockers"
    ]

    build_ou_v3_reviewed_activation(
        root=tmp_path,
        apply=True,
        reviewer="research-reviewer",
        review_note="Reviewed both independent OU parity claims before activation.",
        review_packet_id=packet.summary["review_packet_id"],
    )
    status_path = tmp_path / "reports/active/wizard_ou_v3_activation_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["immutable_activation_sha256"] = "0" * 64
    status_path.write_text(json.dumps(status), encoding="utf-8")

    with pytest.raises(ValueError, match="immutable activation binding mismatch"):
        refresh_activated_ou_v3_proofs(root=tmp_path)


def test_ou_v3_review_remains_blocked_without_vendor_holdout(
    tmp_path: Path,
) -> None:
    _register_v3_fixture(tmp_path)
    _write_activation_dependencies(tmp_path)

    gate = build_ou_v3_supersession_gate(root=tmp_path)
    packet = build_ou_v3_review_handoff(root=tmp_path)
    activation = build_ou_v3_reviewed_activation(root=tmp_path)

    assert gate.summary["status"] == "BLOCKED"
    assert packet.summary["status"] == "WAITING_FOR_HOLDOUT"
    assert activation.summary["status"] == "BLOCKED"
    assert activation.summary["testnet_order_authority"] is False
    assert activation.summary["live_trading_authorized"] is False


def _register_v3_fixture(tmp_path: Path):
    proof_path = _write_fixture(tmp_path)
    register_ou_v2_blind_holdout(
        root=tmp_path,
        proof_path=proof_path,
        derivation_group="dev",
        holdout_group="holdout",
    )
    timestamps = pd.date_range("2025-01-01", periods=80, freq="D", tz="UTC")
    x_path = tmp_path / "btc.json"
    y_path = tmp_path / "eth.json"
    rng = np.random.default_rng(2026)
    x_closes = np.empty(len(timestamps), dtype=float)
    residual = np.empty(len(timestamps), dtype=float)
    x_closes[0] = 100.0
    residual[0] = 20.0
    for index in range(1, len(timestamps)):
        x_closes[index] = x_closes[index - 1] + 0.2 + rng.normal(0.0, 0.5)
        residual[index] = 0.93 * residual[index - 1] + rng.normal(0.0, 0.4)
    y_closes = 0.8 * x_closes + residual
    x_path.write_text(
        json.dumps(
            {
                "candles": [
                    {
                        "startedAt": timestamp.isoformat(),
                        "open": float(x_closes[index] * 0.999),
                        "close": float(x_closes[index]),
                    }
                    for index, timestamp in enumerate(timestamps)
                ]
            }
        ),
        encoding="utf-8",
    )
    y_path.write_text(
        json.dumps(
            {
                "candles": [
                    {
                        "startedAt": timestamp.isoformat(),
                        "open": float(y_closes[index] * 0.999),
                        "close": float(y_closes[index]),
                    }
                    for index, timestamp in enumerate(timestamps)
                ]
            }
        ),
        encoding="utf-8",
    )
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True, exist_ok=True)
    (active / "wizard_ou_v2_holdout_status.json").write_text("{}\n", encoding="utf-8")
    (active / "wizard_ou_v2_holdout_evaluation.csv").write_text(
        "cell_status\nFAIL\n", encoding="utf-8"
    )

    result = register_ou_v3_prospective_holdout(
        root=tmp_path,
        asset_x_path=x_path,
        asset_y_path=y_path,
        observations=60,
    )
    register_ou_trend_selector_v1_holdout(root=tmp_path)
    return result


def _complete_v3_holdout(root: Path) -> None:
    def fetcher(request, *, api_key):
        del api_key
        params = request.payload()["params"]
        x = np.asarray(params["series_1_closes"], dtype=float)
        y = np.asarray(params["series_2_closes"], dtype=float)
        inc_trend = bool(
            _ou_local_trend_selector_v1(x, y)["local_predicted_inc_trend"]
        )
        beta = _ou_trend_aware_profile_beta_v3_candidate(
            x, y, inc_trend=inc_trend
        )
        spread = _ou_trend_aware_profile_spread_v3_candidate(
            x, y, inc_trend=inc_trend
        )
        zscore = (spread - spread.mean()) / spread.std(ddof=1)
        series = pd.Series(spread)
        zscore_roll = (
            series.sub(series.rolling(42, min_periods=42).mean())
            .div(series.rolling(42, min_periods=42).std(ddof=1))
            .fillna(0.0)
            .to_numpy()
        )
        _, phi = np.linalg.lstsq(
            np.column_stack([np.ones(len(spread) - 1), spread[:-1]]),
            spread[1:],
            rcond=None,
        )[0]
        return {
            "history": {
                "coint_eg": {"inc_trend": inc_trend},
                "spread_stats": {
                    "spread": spread.tolist(),
                    "zscore": zscore.tolist(),
                    "zscore_roll": zscore_roll.tolist(),
                    "hedge_ratio": beta,
                    "half_life": float(-np.log(2.0) / np.log(phi)),
                    "log_used": False,
                },
            }
        }

    result = run_ou_v3_prospective_holdout(
        root=root,
        now=datetime(2026, 8, 12, tzinfo=UTC),
        execute=True,
        api_key="secret",
        fetcher=fetcher,
        credits_fetcher=lambda **_: 0,
    )
    assert result.summary["evaluation_status"] == "PASS"


def _write_activation_dependencies(root: Path) -> None:
    active = root / "reports" / "active"
    (active / "wizard_mode_comparator_contract.csv").write_text(
        "exact_mode,version\nOU (Spread),v1\n", encoding="utf-8"
    )
    (active / "wizard_mode_comparator_contract_receipt.json").write_text(
        json.dumps({"contract_id": "wizard-comparator-v1"}), encoding="utf-8"
    )


def _write_v3_refresh_proofs(root: Path) -> None:
    contract = json.loads(
        (root / "config/wizard_ou_comparator_v3_holdout.json").read_text(
            encoding="utf-8"
        )
    )
    rows = []
    for binding in contract["holdout_bindings"]:
        rows.append(
            {
                "pair_group_id": "ou-v3-holdout",
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
        )
    pd.DataFrame(rows).to_csv(
        root / "reports/active/hyperliquid_wizard_vendor_mode_proofs.csv",
        index=False,
    )


def _sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()
