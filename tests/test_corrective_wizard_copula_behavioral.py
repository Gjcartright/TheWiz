from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from quant_platform.crypto_wizards_history import fetch_custom_series_copula
from quant_platform.orchestration.corrective_wizard_copula_behavioral import (
    register_copula_behavioral_v2,
    run_copula_behavioral_proofs,
    run_current_copula_behavioral_proofs,
)
from quant_platform.orchestration.corrective_wizard_parity import (
    build_wizard_mode_evidence_completion,
)


def _write_contract(root: Path) -> None:
    config = root / "config"
    active = root / "reports" / "active"
    config.mkdir(parents=True, exist_ok=True)
    active.mkdir(parents=True, exist_ok=True)
    contract = {
        "threshold_changes_after_vendor_response_allowed": False,
        "candidate_promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    path = config / "wizard_copula_behavioral_parity.json"
    path.write_text(json.dumps(contract), encoding="utf-8")
    comparator_contract = active / "wizard_mode_comparator_contract.csv"
    comparator_contract.write_text("exact_mode\nCopula\n", encoding="utf-8")
    comparator_receipt = active / "wizard_mode_comparator_contract_receipt.json"
    comparator_receipt.write_text("{}\n", encoding="utf-8")
    proof_queue = active / "exhaustive_wizard_exact_mode_proof_queue.csv"
    receipt = {
        "contract_path": "config/wizard_copula_behavioral_parity.json",
        "contract_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "proof_queue_path": str(proof_queue.relative_to(root)),
        "proof_queue_sha256": hashlib.sha256(proof_queue.read_bytes()).hexdigest(),
        "formula_comparator_contract_path": str(
            comparator_contract.relative_to(root)
        ),
        "formula_comparator_contract_sha256": hashlib.sha256(
            comparator_contract.read_bytes()
        ).hexdigest(),
        "formula_comparator_receipt_path": str(
            comparator_receipt.relative_to(root)
        ),
        "formula_comparator_receipt_sha256": hashlib.sha256(
            comparator_receipt.read_bytes()
        ).hexdigest(),
        "registered_before_current_queue_vendor_responses": True,
        "candidate_promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    (active / "wizard_copula_behavioral_contract_receipt.json").write_text(
        json.dumps(receipt), encoding="utf-8"
    )


def _write_inputs(root: Path, *, captured: bool = True) -> None:
    active = root / "reports" / "active"
    raw = root / "data" / "raw"
    active.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True)
    queue = []
    proofs = []
    for group_index in range(2):
        for orientation in ("original", "reverse"):
            group = f"group-{group_index}"
            x = [100.0 + index + group_index for index in range(50)]
            y = [50.0 + index + group_index for index in range(50)]
            if orientation == "reverse":
                x, y = y, x
            request_path = raw / f"{group}_{orientation}.json"
            request_path.write_text(
                json.dumps(
                    {
                        "params": {
                            "series_1_closes": x,
                            "series_2_closes": y,
                        }
                    }
                ),
                encoding="utf-8",
            )
            queue.append(
                {
                    "pair_group_id": group,
                    "exact_mode": "Copula",
                    "orientation": orientation,
                    "vendor_custom_series_eligible": True,
                }
            )
            if captured:
                proofs.append(
                    {
                        "pair_group_id": group,
                        "pair": f"PAIR-{group_index}",
                        "exact_mode": "Copula",
                        "orientation": orientation,
                        "vendor_response_captured": True,
                        "request_path": str(request_path.relative_to(root)),
                    }
                )
    pd.DataFrame(queue).to_csv(
        active / "exhaustive_wizard_exact_mode_proof_queue.csv", index=False
    )
    pd.DataFrame(proofs).to_csv(
        active / "hyperliquid_wizard_vendor_mode_proofs.csv", index=False
    )


def _write_v2_registration_inputs(root: Path) -> None:
    active = root / "reports" / "active"
    raw = root / "data" / "raw"
    active.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)
    queue = []
    proofs = []
    for group_index, observations in enumerate((50, 60)):
        group = f"group-{group_index}"
        x = [100.0 + index + group_index for index in range(observations + 5)]
        y = [50.0 + index + group_index for index in range(observations + 5)]
        for orientation in ("original", "reverse"):
            first, second = (x, y) if orientation == "original" else (y, x)
            request_path = raw / f"{group}_{orientation}_backtest.json"
            request_path.write_text(
                json.dumps(
                    {
                        "params": {
                            "series_1_opens": first,
                            "series_1_closes": first,
                            "series_2_opens": second,
                            "series_2_closes": second,
                        }
                    }
                ),
                encoding="utf-8",
            )
            queue.append(
                {
                    "pair_group_id": group,
                    "pair": f"PAIR-{group_index}-{orientation}",
                    "exact_mode": "Copula",
                    "orientation": orientation,
                    "proof_observations": observations,
                    "vendor_custom_series_eligible": True,
                }
            )
            proofs.append(
                {
                    "pair_group_id": group,
                    "pair": f"PAIR-{group_index}-{orientation}",
                    "exact_mode": "Copula",
                    "orientation": orientation,
                    "vendor_response_captured": True,
                    "request_path": str(request_path.relative_to(root)),
                }
            )
    pd.DataFrame(queue).to_csv(
        active / "exhaustive_wizard_exact_mode_proof_queue.csv", index=False
    )
    pd.DataFrame(proofs).to_csv(
        active / "hyperliquid_wizard_vendor_mode_proofs.csv", index=False
    )
    (active / "wizard_mode_comparator_contract.csv").write_text(
        "exact_mode\nCopula\n", encoding="utf-8"
    )
    (active / "wizard_mode_comparator_contract_receipt.json").write_text(
        "{}\n", encoding="utf-8"
    )


def test_v2_registration_freezes_exact_swaps_without_vendor_responses(tmp_path):
    _write_v2_registration_inputs(tmp_path)

    result = register_copula_behavioral_v2(
        root=tmp_path,
        now=datetime(2026, 8, 11, tzinfo=UTC),
    )
    source = pd.read_csv(result.paths["source_ledger"])

    assert result.summary["status"] == "REGISTERED_WAITING_VENDOR_RESPONSES"
    assert result.summary["registered_cells"] == 4
    assert source["source_request_frozen"].astype(bool).all()
    assert not source["vendor_response_captured"].astype(bool).any()
    assert not (tmp_path / "data" / "raw" / "crypto_wizards_copula_behavioral_proofs").exists()
    for _, group in source.groupby("pair_group_id"):
        original = json.loads(
            (tmp_path / group.loc[group["orientation"].eq("original"), "request_path"].iloc[0]).read_text()
        )["params"]
        reverse = json.loads(
            (tmp_path / group.loc[group["orientation"].eq("reverse"), "request_path"].iloc[0]).read_text()
        )["params"]
        assert original["series_1_closes"] == reverse["series_2_closes"]
        assert original["series_2_closes"] == reverse["series_1_closes"]
    assert result.summary["testnet_order_authority"] is False
    assert result.summary["live_trading_authorized"] is False


def test_current_runner_uses_registered_v2_and_passes_symmetric_responses(tmp_path):
    _write_v2_registration_inputs(tmp_path)
    register_copula_behavioral_v2(
        root=tmp_path,
        now=datetime(2026, 8, 11, tzinfo=UTC),
    )
    calls = []

    def fetcher(request, **kwargs):
        calls.append(request)
        original = request.series_1_closes[0] > request.series_2_closes[0]
        return {
            "copula_name": "clayton",
            "u1_given_u2": 0.91 if original else 0.08,
            "u2_given_u1": 0.08 if original else 0.91,
        }

    result = run_current_copula_behavioral_proofs(
        root=tmp_path,
        now=datetime(2026, 8, 12, tzinfo=UTC),
        execute=True,
        api_key="key",
        fetcher=fetcher,
        credits_fetcher=lambda **_: {"credits_used": 0, "credit_limit": 1000},
    )

    assert result.summary["artifact_generation"] == 2
    assert result.summary["status"] == "PASS"
    assert result.summary["endpoint_calls_made"] == 8
    assert result.summary["behavioral_cells_passed"] == 4
    assert len(calls) == 8
    assert result.paths["status"].name == "wizard_copula_behavioral_v2_status.json"
    assert result.summary["candidate_promotion_authority"] is False
    completion = build_wizard_mode_evidence_completion(root=tmp_path)
    assert completion["copula_behavioral_status"] == "PASS"
    assert completion["copula_queue_cells_expected"] == 4
    assert completion["copula_behavioral_cells_passed"] == 4
    assert completion["accepted_queue_cells"] == 4


def test_v2_bound_request_mutation_fails_before_vendor_calls(tmp_path):
    _write_v2_registration_inputs(tmp_path)
    registered = register_copula_behavioral_v2(
        root=tmp_path,
        now=datetime(2026, 8, 11, tzinfo=UTC),
    )
    source = pd.read_csv(registered.paths["source_ledger"])
    request_path = tmp_path / source.iloc[0]["request_path"]
    request_path.write_text(
        request_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    calls = []

    with pytest.raises(ValueError, match="request binding hash mismatch"):
        run_current_copula_behavioral_proofs(
            root=tmp_path,
            execute=True,
            api_key="key",
            fetcher=lambda *args, **kwargs: calls.append((args, kwargs)),
            credits_fetcher=lambda **_: {"credits_used": 0, "credit_limit": 1000},
        )

    assert calls == []


def test_waits_for_copula_backtest_responses_without_calls(tmp_path):
    _write_inputs(tmp_path, captured=False)
    _write_contract(tmp_path)
    calls = []

    result = run_copula_behavioral_proofs(
        root=tmp_path,
        execute=True,
        api_key="key",
        fetcher=lambda *args, **kwargs: calls.append((args, kwargs)),
        credits_fetcher=lambda **_: {"credits_used": 0, "credit_limit": 1000},
    )

    assert result.summary["status"] == "WAITING_FOR_COPULA_BACKTEST_RESPONSES"
    assert result.summary["endpoint_calls_made"] == 0
    assert calls == []
    assert not result.summary["testnet_order_authority"]


def test_preflight_plans_eight_calls_but_uses_no_credits(tmp_path):
    _write_inputs(tmp_path)
    _write_contract(tmp_path)
    calls = []

    result = run_copula_behavioral_proofs(
        root=tmp_path,
        execute=False,
        fetcher=lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    assert result.summary["status"] == "PLANNED"
    assert result.summary["endpoint_calls_required"] == 8
    assert result.summary["credits_estimated"] == 8
    assert result.summary["endpoint_calls_made"] == 0
    assert calls == []


def test_nonidentical_source_windows_block_before_endpoint_or_credit_calls(tmp_path):
    _write_inputs(tmp_path)
    proof_path = (
        tmp_path
        / "reports"
        / "active"
        / "hyperliquid_wizard_vendor_mode_proofs.csv"
    )
    proofs = pd.read_csv(proof_path)
    for value in proofs.loc[proofs["orientation"].eq("reverse"), "request_path"]:
        request_path = tmp_path / value
        payload = json.loads(request_path.read_text(encoding="utf-8"))
        payload["params"]["series_1_closes"].append(
            payload["params"]["series_1_closes"][-1]
        )
        payload["params"]["series_2_closes"].append(
            payload["params"]["series_2_closes"][-1]
        )
        request_path.write_text(json.dumps(payload), encoding="utf-8")
    _write_contract(tmp_path)
    calls = []

    result = run_copula_behavioral_proofs(
        root=tmp_path,
        execute=True,
        api_key="key",
        fetcher=lambda *args, **kwargs: calls.append((args, kwargs)),
        credits_fetcher=lambda **kwargs: pytest.fail(
            "credit endpoint called for invalid source orientation"
        ),
    )
    detail = pd.read_csv(result.paths["detail"])

    assert result.summary["endpoint_calls_made"] == 0
    assert result.summary["endpoint_calls_required"] == 0
    assert calls == []
    blocked = detail.loc[detail["pair_group_id"].eq("group-0")]
    assert blocked["behavioral_status"].eq("BLOCKED").all()
    assert blocked["blocker"].eq(
        "copula_source_orientation_requests_not_exact_swaps"
    ).all()


def test_complete_repeatable_orientation_symmetric_cohort_passes(tmp_path):
    _write_inputs(tmp_path)
    _write_contract(tmp_path)
    calls = []

    def fetcher(request, **kwargs):
        calls.append(request)
        original = request.series_1_closes[0] > request.series_2_closes[0]
        return {
            "copula_name": "clayton",
            "u1_given_u2": 0.91 if original else 0.08,
            "u2_given_u1": 0.08 if original else 0.91,
        }

    result = run_copula_behavioral_proofs(
        root=tmp_path,
        now=datetime(2026, 8, 11, tzinfo=UTC),
        execute=True,
        api_key="key",
        fetcher=fetcher,
        credits_fetcher=lambda **_: {"credits_used": 0, "credit_limit": 1000},
    )

    assert result.summary["status"] == "PASS"
    assert result.summary["endpoint_calls_made"] == 8
    assert result.summary["behavioral_cells_passed"] == 4
    assert result.summary["provenance_cells_complete"] == 4
    assert result.summary["behavioral_parity_proven"]
    assert not result.summary["formula_parity_proven"]
    assert not result.summary["candidate_promotion_authority"]
    detail = pd.read_csv(result.paths["detail"])
    for _, row in detail.iterrows():
        for index in (1, 2):
            response_path = tmp_path / row[f"response_{index}_path"]
            assert response_path.is_file()
            assert row[f"response_{index}_sha256"] == hashlib.sha256(
                response_path.read_bytes()
            ).hexdigest()
            datetime.fromisoformat(row[f"response_{index}_captured_at_utc"])
        request_path = tmp_path / row["request_path"]
        assert row["request_sha256"] == hashlib.sha256(
            request_path.read_bytes()
        ).hexdigest()
        assert row["provenance_status"] == "PASS"


def test_reversed_pair_labels_do_not_invalidate_an_exact_series_swap(tmp_path):
    _write_inputs(tmp_path)
    proof_path = (
        tmp_path / "reports" / "active" / "hyperliquid_wizard_vendor_mode_proofs.csv"
    )
    proofs = pd.read_csv(proof_path)
    reverse = proofs["orientation"].eq("reverse")
    proofs.loc[reverse, "pair"] = proofs.loc[reverse, "pair"].map(
        lambda value: f"REVERSED-{value}"
    )
    proofs.to_csv(proof_path, index=False)
    _write_contract(tmp_path)

    def fetcher(request, **kwargs):
        original = request.series_1_closes[0] > request.series_2_closes[0]
        return {
            "copula_name": "clayton",
            "u1_given_u2": 0.91 if original else 0.08,
            "u2_given_u1": 0.08 if original else 0.91,
        }

    result = run_copula_behavioral_proofs(
        root=tmp_path,
        now=datetime(2026, 8, 11, tzinfo=UTC),
        execute=True,
        api_key="key",
        fetcher=fetcher,
        credits_fetcher=lambda **_: {"credits_used": 0, "credit_limit": 1000},
    )

    assert result.summary["status"] == "PASS"
    assert result.summary["behavioral_cells_passed"] == 4
    assert pd.read_csv(result.paths["detail"])["input_orientation_status"].eq(
        "PASS"
    ).all()


def test_real_copula_client_signature_is_compatible_with_evaluator(
    tmp_path, monkeypatch
):
    _write_inputs(tmp_path)
    _write_contract(tmp_path)
    posts = []

    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def fake_post(url, json=None, headers=None, timeout=None):
        posts.append(
            {"url": url, "json": json, "headers": headers, "timeout": timeout}
        )
        original = json["series_1_closes"][0] > json["series_2_closes"][0]
        return FakeResponse(
            {
                "copula_name": "clayton",
                "u1_given_u2": 0.91 if original else 0.08,
                "u2_given_u1": 0.08 if original else 0.91,
            }
        )

    monkeypatch.setattr(
        "quant_platform.crypto_wizards_history.requests.post", fake_post
    )
    result = run_copula_behavioral_proofs(
        root=tmp_path,
        execute=True,
        api_key="key",
        fetcher=fetch_custom_series_copula,
        credits_fetcher=lambda **_: {"credits_used": 0, "credit_limit": 1000},
    )

    assert result.summary["status"] == "PASS"
    assert len(posts) == 8
    assert all(post["url"].endswith("/v1beta/copula") for post in posts)
    assert all(
        set(post["json"]) == {"series_1_closes", "series_2_closes"}
        for post in posts
    )
    assert all(post["headers"]["X-api-key"] == "key" for post in posts)


def test_repeatability_mutation_fails_closed(tmp_path):
    _write_inputs(tmp_path)
    _write_contract(tmp_path)
    counter = {"value": 0}

    def fetcher(request, **kwargs):
        counter["value"] += 1
        original = request.series_1_closes[0] > request.series_2_closes[0]
        drift = 0.01 if counter["value"] == 2 else 0.0
        return {
            "copula_name": "clayton",
            "u1_given_u2": (0.91 if original else 0.08) + drift,
            "u2_given_u1": 0.08 if original else 0.91,
        }

    result = run_copula_behavioral_proofs(
        root=tmp_path,
        execute=True,
        api_key="key",
        fetcher=fetcher,
        credits_fetcher=lambda **_: {"credits_used": 0, "credit_limit": 1000},
    )

    assert result.summary["status"] == "FAIL"
    assert result.summary["behavioral_cells_failed"] >= 1
    assert not result.summary["behavioral_parity_proven"]


def test_registered_queue_mutation_fails_before_endpoint_calls(tmp_path):
    _write_inputs(tmp_path)
    _write_contract(tmp_path)
    queue_path = (
        tmp_path
        / "reports"
        / "active"
        / "exhaustive_wizard_exact_mode_proof_queue.csv"
    )
    queue_path.write_text(
        queue_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    calls = []

    with pytest.raises(ValueError, match="binding hash mismatch: proof_queue_path"):
        run_copula_behavioral_proofs(
            root=tmp_path,
            execute=True,
            api_key="key",
            fetcher=lambda *args, **kwargs: calls.append((args, kwargs)),
            credits_fetcher=lambda **_: {"credits_used": 0, "credit_limit": 1000},
        )

    assert calls == []


def test_complete_capture_is_reused_without_credits_or_endpoint_calls(tmp_path):
    _write_inputs(tmp_path)
    _write_contract(tmp_path)
    calls = []

    def fetcher(request, **kwargs):
        calls.append(request)
        original = request.series_1_closes[0] > request.series_2_closes[0]
        return {
            "copula_name": "clayton",
            "u1_given_u2": 0.91 if original else 0.08,
            "u2_given_u1": 0.08 if original else 0.91,
        }

    first = run_copula_behavioral_proofs(
        root=tmp_path,
        now=datetime(2026, 8, 11, 1, tzinfo=UTC),
        execute=True,
        api_key="key",
        fetcher=fetcher,
        credits_fetcher=lambda **_: {
            "credits_used": 0,
            "credit_limit": 1000,
        },
    )
    first_detail = first.paths["detail"].read_bytes()
    first_cohort = first.summary["cohort_receipt_path"]
    first_cohort_hash = first.summary["cohort_receipt_sha256"]
    calls.clear()

    second = run_copula_behavioral_proofs(
        root=tmp_path,
        now=datetime(2026, 8, 12, 1, tzinfo=UTC),
        execute=True,
        fetcher=lambda *args, **kwargs: pytest.fail("endpoint called on reuse"),
        credits_fetcher=lambda **kwargs: pytest.fail("credit API called on reuse"),
    )
    third = run_copula_behavioral_proofs(
        root=tmp_path,
        now=datetime(2026, 8, 12, 2, tzinfo=UTC),
        execute=False,
        fetcher=lambda *args, **kwargs: pytest.fail("endpoint called in preflight"),
    )

    assert calls == []
    assert second.summary["status"] == "PASS"
    assert third.summary["status"] == "PASS"
    assert second.summary["endpoint_calls_required"] == 0
    assert second.summary["endpoint_calls_made"] == 0
    assert second.summary["reused_immutable_capture_cells"] == 4
    assert second.summary["endpoint_calls_avoided_by_reuse"] == 8
    assert second.summary["daily_attempt_status"] == "NOT_REQUIRED"
    assert second.summary["cohort_receipt_path"] == first_cohort
    assert second.summary["cohort_receipt_sha256"] == first_cohort_hash
    assert second.paths["detail"].read_bytes() == first_detail
    assert not second.summary["testnet_order_authority"]
    assert not second.summary["live_trading_authorized"]


def test_tampered_raw_response_fails_closed_without_new_calls(tmp_path):
    _write_inputs(tmp_path)
    _write_contract(tmp_path)

    def fetcher(request, **kwargs):
        original = request.series_1_closes[0] > request.series_2_closes[0]
        return {
            "copula_name": "clayton",
            "u1_given_u2": 0.91 if original else 0.08,
            "u2_given_u1": 0.08 if original else 0.91,
        }

    first = run_copula_behavioral_proofs(
        root=tmp_path,
        now=datetime(2026, 8, 11, tzinfo=UTC),
        execute=True,
        api_key="key",
        fetcher=fetcher,
        credits_fetcher=lambda **_: {
            "credits_used": 0,
            "credit_limit": 1000,
        },
    )
    detail = pd.read_csv(first.paths["detail"])
    response_path = tmp_path / detail.iloc[0]["response_1_path"]
    response_path.write_text('{"copula_name":"tampered"}\n', encoding="utf-8")
    calls = []

    with pytest.raises(ValueError, match="raw artifact changed"):
        run_copula_behavioral_proofs(
            root=tmp_path,
            now=datetime(2026, 8, 12, tzinfo=UTC),
            execute=True,
            api_key="key",
            fetcher=lambda *args, **kwargs: calls.append((args, kwargs)),
            credits_fetcher=lambda **_: {
                "credits_used": 0,
                "credit_limit": 1000,
            },
        )

    assert calls == []


def test_tampered_capture_timestamp_breaks_content_identity(tmp_path):
    _write_inputs(tmp_path)
    _write_contract(tmp_path)

    def fetcher(request, **kwargs):
        original = request.series_1_closes[0] > request.series_2_closes[0]
        return {
            "copula_name": "clayton",
            "u1_given_u2": 0.91 if original else 0.08,
            "u2_given_u1": 0.08 if original else 0.91,
        }

    first = run_copula_behavioral_proofs(
        root=tmp_path,
        now=datetime(2026, 8, 11, tzinfo=UTC),
        execute=True,
        api_key="key",
        fetcher=fetcher,
        credits_fetcher=lambda **_: {
            "credits_used": 0,
            "credit_limit": 1000,
        },
    )
    detail = pd.read_csv(first.paths["detail"])
    receipt_path = tmp_path / detail.iloc[0]["capture_receipt_path"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["response_1_captured_at_utc"] = "2026-01-01T00:00:00+00:00"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ValueError, match="content identity mismatch"):
        run_copula_behavioral_proofs(
            root=tmp_path,
            now=datetime(2026, 8, 12, tzinfo=UTC),
            execute=False,
        )


def test_orientation_label_requires_exactly_swapped_input_series(tmp_path):
    _write_inputs(tmp_path)
    reverse_path = tmp_path / "data" / "raw" / "group-0_reverse.json"
    reverse_request = json.loads(reverse_path.read_text(encoding="utf-8"))
    reverse_request["params"]["series_1_closes"][0] += 0.5
    reverse_path.write_text(json.dumps(reverse_request), encoding="utf-8")
    _write_contract(tmp_path)

    def fetcher(request, **kwargs):
        original = request.series_1_closes[0] > request.series_2_closes[0]
        return {
            "copula_name": "clayton",
            "u1_given_u2": 0.91 if original else 0.08,
            "u2_given_u1": 0.08 if original else 0.91,
        }

    result = run_copula_behavioral_proofs(
        root=tmp_path,
        now=datetime(2026, 8, 11, tzinfo=UTC),
        execute=True,
        api_key="key",
        fetcher=fetcher,
        credits_fetcher=lambda **_: {
            "credits_used": 0,
            "credit_limit": 1000,
        },
    )

    assert result.summary["status"] == "INCOMPLETE"
    assert result.summary["endpoint_calls_made"] == 4
    detail = pd.read_csv(result.paths["detail"])
    broken = detail.loc[detail["pair_group_id"].eq("group-0")]
    assert broken["behavioral_status"].eq("BLOCKED").all()
    assert broken["blocker"].eq(
        "copula_source_orientation_requests_not_exact_swaps"
    ).all()
    assert not result.summary["behavioral_parity_proven"]


def test_endpoint_failure_aborts_remaining_calls_and_same_day_retry(tmp_path):
    _write_inputs(tmp_path)
    _write_contract(tmp_path)
    calls = []

    def failing_fetcher(request, **kwargs):
        calls.append(request)
        raise RuntimeError("endpoint unavailable")

    first = run_copula_behavioral_proofs(
        root=tmp_path,
        now=datetime(2026, 8, 11, 1, tzinfo=UTC),
        execute=True,
        api_key="key",
        fetcher=failing_fetcher,
        credits_fetcher=lambda **_: {
            "credits_used": 0,
            "credit_limit": 1000,
        },
    )
    second = run_copula_behavioral_proofs(
        root=tmp_path,
        now=datetime(2026, 8, 11, 2, tzinfo=UTC),
        execute=True,
        api_key="key",
        fetcher=failing_fetcher,
        credits_fetcher=lambda **_: {
            "credits_used": 0,
            "credit_limit": 1000,
        },
    )

    assert first.summary["endpoint_calls_made"] == 1
    assert first.summary["endpoint_responses_captured_this_cycle"] == 0
    assert len(calls) == 1
    assert first.summary["status"] == "FAIL"
    assert second.summary["endpoint_calls_made"] == 0
    assert second.summary["endpoint_responses_captured_this_cycle"] == 0
    assert second.summary["daily_attempt_status"] == "BLOCKED"
    assert (
        second.summary["daily_attempt_status"] == "BLOCKED"
        and second.summary["credit_preflight_blocker"]
        == "copula_external_attempt_already_registered_for_utc_day"
    )
