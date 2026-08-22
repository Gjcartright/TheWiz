from __future__ import annotations

import json
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import pandas as pd
import pytest

from quant_platform.orchestration import corrective_wizard_capture_manifest
from quant_platform.orchestration.corrective_wizard_capture_manifest import (
    build_corrective_wizard_capture_manifest,
    validate_ou_v4_capture_manifest_contract,
    validate_ou_v5_capture_manifest_contract,
    validate_ou_v6_capture_manifest_contract,
)

NOW = datetime(2026, 8, 11, 3, 30, tzinfo=UTC)


def test_immutable_capture_artifact_publication_never_leaves_partial_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "immutable" / "manifest.json"

    def fail_publish(
        _source: Path,
        _destination: Path,
        *,
        follow_symlinks: bool,
    ) -> None:
        assert follow_symlinks is False
        raise OSError("simulated publication interruption")

    monkeypatch.setattr(corrective_wizard_capture_manifest.os, "link", fail_publish)

    with pytest.raises(OSError, match="simulated publication interruption"):
        corrective_wizard_capture_manifest._write_or_validate_immutable_json(
            {"manifest_id": "test"},
            target,
        )

    assert not target.exists()
    assert list(target.parent.glob(".*.tmp")) == []


def test_manifest_binds_all_thirteen_calls_and_eighteen_credits(tmp_path):
    paths = _write_inputs(tmp_path)
    result = build_corrective_wizard_capture_manifest(
        root=tmp_path,
        now=NOW,
        **paths,
        copula_plan_provider=lambda _: _copula_plans(),
    )

    assert result.summary["status"] == "PASS"
    assert result.summary["pending_calls"] == 13
    assert result.summary["planned_credits"] == 18
    assert result.summary["lane_totals"] == {
        "exact_mode_backtest": {"calls": 1, "credits": 2},
        "ou_v3_holdout": {"calls": 4, "credits": 8},
        "ou_v4_holdout": {"calls": 0, "credits": 0},
        "ou_v5_holdout": {"calls": 0, "credits": 0},
        "ou_v6_holdout": {"calls": 0, "credits": 0},
        "copula_behavioral": {"calls": 8, "credits": 8},
    }
    assert result.summary["intentional_cross_lane_overlap_calls"] == 2
    assert result.summary["capture_eligible_now"] is True
    assert result.summary["testnet_order_authority"] is False
    assert result.summary["live_trading_authorized"] is False
    assert Path(result.paths["immutable_manifest"]).is_file()
    assert Path(result.paths["immutable_source_receipt"]).is_file()
    assert result.summary["source_artifacts_current_match"] is True
    assert result.summary["source_artifacts_current_drift_classification"] == "EXACT_MATCH"
    assert result.summary["source_artifacts_current_metadata_only_paths"] == []
    assert result.summary["source_artifacts_current_scientific_or_structural_paths"] == []


def test_manifest_reuses_frozen_source_snapshots_when_mutable_inputs_change(tmp_path):
    paths = _write_inputs(tmp_path)
    first = build_corrective_wizard_capture_manifest(
        root=tmp_path,
        now=NOW,
        **paths,
        copula_plan_provider=lambda _: _copula_plans(),
    )
    first_fingerprint = first.summary["source_artifacts_sha256"]
    queue = pd.read_csv(paths["queue_path"])
    queue.to_csv(paths["queue_path"], index=False, lineterminator="\r\n")

    second = build_corrective_wizard_capture_manifest(
        root=tmp_path,
        now=NOW,
        **paths,
        copula_plan_provider=lambda _: _copula_plans(),
    )

    assert second.summary["manifest_id"] == first.summary["manifest_id"]
    assert second.summary["source_receipt_id"] == first.summary["source_receipt_id"]
    assert second.summary["source_artifacts_sha256"] == first_fingerprint
    assert second.summary["source_artifacts_current_match"] is False
    assert second.summary["status"] == "BLOCKED"
    assert (
        second.summary["source_artifacts_current_drift_classification"]
        == "SCIENTIFIC_OR_STRUCTURAL_DRIFT"
    )
    assert (
        str(paths["queue_path"].relative_to(tmp_path))
        in second.summary["source_artifacts_current_scientific_or_structural_paths"]
    )
    assert "current_capture_source_scientific_or_structural_drift" in second.summary["blockers"]


def test_manifest_allows_generated_at_only_source_drift(tmp_path):
    paths = _write_inputs(tmp_path)
    first = build_corrective_wizard_capture_manifest(
        root=tmp_path,
        now=NOW,
        **paths,
        copula_plan_provider=lambda _: _copula_plans(),
    )
    budget = json.loads(paths["credit_budget_path"].read_text(encoding="utf-8"))
    budget["generated_at_utc"] = "2026-08-11T04:30:00+00:00"
    paths["credit_budget_path"].write_text(
        json.dumps(budget),
        encoding="utf-8",
    )

    second = build_corrective_wizard_capture_manifest(
        root=tmp_path,
        now=NOW,
        **paths,
        copula_plan_provider=lambda _: _copula_plans(),
    )

    assert second.summary["manifest_id"] == first.summary["manifest_id"]
    assert second.summary["status"] == "PASS"
    assert second.summary["source_artifacts_current_match"] is False
    assert (
        second.summary["source_artifacts_current_drift_classification"] == "GENERATED_METADATA_ONLY"
    )
    assert second.summary["source_artifacts_current_metadata_only_paths"] == [
        str(paths["credit_budget_path"].relative_to(tmp_path))
    ]
    assert second.summary["source_artifacts_current_scientific_or_structural_paths"] == []


def test_manifest_rejects_tampered_frozen_source_snapshot(tmp_path):
    paths = _write_inputs(tmp_path)
    first = build_corrective_wizard_capture_manifest(
        root=tmp_path,
        now=NOW,
        **paths,
        copula_plan_provider=lambda _: _copula_plans(),
    )
    source_receipt = json.loads(
        Path(first.paths["immutable_source_receipt"]).read_text(encoding="utf-8")
    )
    snapshot = tmp_path / source_receipt["source_artifacts"][0]["snapshot_path"]
    snapshot.chmod(0o600)
    snapshot.write_text("tampered\n", encoding="utf-8")

    with pytest.raises(ValueError, match="source receipt invalid"):
        build_corrective_wizard_capture_manifest(
            root=tmp_path,
            now=NOW,
            **paths,
            copula_plan_provider=lambda _: _copula_plans(),
        )


def test_manifest_defers_after_a_verified_same_day_external_attempt(tmp_path):
    paths = _write_inputs(tmp_path)
    status = tmp_path / "reports" / "active" / "corrective_wizard_proof_scheduler_status.json"
    status.write_text(
        json.dumps({"attempt_date_utc": "2026-08-11", "external_attempt_made": True}),
        encoding="utf-8",
    )

    result = build_corrective_wizard_capture_manifest(
        root=tmp_path,
        now=NOW,
        **paths,
        copula_plan_provider=lambda _: _copula_plans(),
    )

    assert result.summary["status"] == "PASS"
    assert result.summary["capture_state"] == "DEFERRED_UNTIL_UTC_RESET"
    assert result.summary["capture_eligible_now"] is False
    assert result.summary["next_external_attempt_eligible_at"] == ("2026-08-12T00:00:00+00:00")


def test_manifest_blocks_a_mutated_ou_request(tmp_path):
    paths = _write_inputs(tmp_path)
    request = tmp_path / "data" / "research" / "ou" / "btc_eth_spread.json"
    request.write_text('{"mutated":true}\n', encoding="utf-8")

    result = build_corrective_wizard_capture_manifest(
        root=tmp_path,
        now=NOW,
        **paths,
        copula_plan_provider=lambda _: _copula_plans(),
    )

    assert result.summary["status"] == "BLOCKED"
    assert result.summary["capture_state"] == "BLOCKED_PREREQUISITES"
    assert "one_or_more_manifest_prerequisites_blocked" in result.summary["blockers"]
    assert any("ou_v3_request_hash_mismatch" in item for item in result.summary["blockers"])


def test_manifest_binds_all_eight_preregistered_ou_v4_calls(tmp_path):
    paths = _write_inputs(tmp_path)
    _write_v4_contract(tmp_path)

    result = build_corrective_wizard_capture_manifest(
        root=tmp_path,
        now=NOW,
        **paths,
        copula_plan_provider=lambda _: [],
    )

    assert result.summary["status"] == "PASS"
    assert result.summary["lane_totals"]["ou_v4_holdout"] == {
        "calls": 8,
        "credits": 16,
    }
    frame = pd.read_csv(result.paths["manifest"])
    v4 = frame.loc[frame["lane"].eq("ou_v4_holdout")]
    assert len(v4) == 8
    assert v4["prerequisite_status"].eq("PASS").all()
    assert not v4["candidate_promotion_authority"].astype(bool).any()
    assert not v4["testnet_order_authority"].astype(bool).any()
    assert not v4["live_trading_authorized"].astype(bool).any()
    immutable = json.loads(Path(result.paths["immutable_manifest"]).read_text(encoding="utf-8"))
    binding = validate_ou_v4_capture_manifest_contract(root=tmp_path, manifest=immutable)
    assert binding == {
        "status": "PASS",
        "blockers": [],
        "expected_calls": 8,
        "observed_calls": 8,
    }


def test_manifest_excludes_a_fully_captured_ou_v4_contract_from_pending_calls(tmp_path):
    paths = _write_inputs(tmp_path)
    _write_v4_contract(tmp_path)
    contract = json.loads(
        (tmp_path / "config" / "wizard_ou_comparator_v4_holdout.json").read_text(encoding="utf-8")
    )
    for binding in contract["holdout_bindings"]:
        response = tmp_path / binding["response_path"]
        response.parent.mkdir(parents=True, exist_ok=True)
        response.write_text('{"captured":true}\n', encoding="utf-8")

    result = build_corrective_wizard_capture_manifest(
        root=tmp_path,
        now=NOW,
        **paths,
        copula_plan_provider=lambda _: [],
    )

    assert result.summary["status"] == "PASS"
    assert result.summary["capture_state"] != "BLOCKED_PREREQUISITES"
    assert result.summary["lane_totals"]["ou_v4_holdout"] == {
        "calls": 0,
        "credits": 0,
    }
    immutable = json.loads(Path(result.paths["immutable_manifest"]).read_text(encoding="utf-8"))
    pending_binding = validate_ou_v4_capture_manifest_contract(
        root=tmp_path,
        manifest=immutable,
        pending_only=True,
    )
    assert pending_binding == {
        "status": "PASS",
        "blockers": [],
        "expected_calls": 0,
        "observed_calls": 0,
    }


def test_ou_v4_manifest_contract_binding_rejects_semantic_call_drift(tmp_path):
    paths = _write_inputs(tmp_path)
    _write_v4_contract(tmp_path)
    result = build_corrective_wizard_capture_manifest(
        root=tmp_path,
        now=NOW,
        **paths,
        copula_plan_provider=lambda _: [],
    )
    manifest = json.loads(Path(result.paths["immutable_manifest"]).read_text(encoding="utf-8"))
    first_v4 = next(row for row in manifest["calls"] if row["lane"] == "ou_v4_holdout")
    first_v4["expected_output_path"] = "data/raw/ou_v4/unregistered_response.json"

    binding = validate_ou_v4_capture_manifest_contract(root=tmp_path, manifest=manifest)

    assert binding["status"] == "BLOCKED"
    assert any(
        blocker.startswith("ou_v4_manifest_contract_call_mismatch:")
        for blocker in binding["blockers"]
    )


def test_manifest_blocks_a_mutated_ou_v4_request(tmp_path):
    paths = _write_inputs(tmp_path)
    request = _write_v4_contract(tmp_path)
    request.write_text('{"mutated":true}\n', encoding="utf-8")

    result = build_corrective_wizard_capture_manifest(
        root=tmp_path,
        now=NOW,
        **paths,
        copula_plan_provider=lambda _: [],
    )

    assert result.summary["status"] == "BLOCKED"
    assert any("ou_v4_request_hash_mismatch" in item for item in result.summary["blockers"])


def test_manifest_binds_all_eight_preregistered_ou_v5_calls(tmp_path):
    paths = _write_inputs(tmp_path)
    _write_v5_contract(tmp_path)

    result = build_corrective_wizard_capture_manifest(
        root=tmp_path,
        now=NOW,
        **paths,
        copula_plan_provider=lambda _: [],
    )

    assert result.summary["status"] == "PASS"
    assert result.summary["lane_totals"]["ou_v5_holdout"] == {
        "calls": 8,
        "credits": 16,
    }
    immutable = json.loads(Path(result.paths["immutable_manifest"]).read_text(encoding="utf-8"))
    binding = validate_ou_v5_capture_manifest_contract(root=tmp_path, manifest=immutable)
    assert binding == {
        "status": "PASS",
        "blockers": [],
        "expected_calls": 8,
        "observed_calls": 8,
    }


def test_manifest_binds_all_eight_final_preregistered_ou_v6_calls(tmp_path):
    paths = _write_inputs(tmp_path)
    _write_v6_contract(tmp_path)

    result = build_corrective_wizard_capture_manifest(
        root=tmp_path,
        now=NOW,
        **paths,
        copula_plan_provider=lambda _: [],
    )

    assert result.summary["status"] == "PASS"
    assert result.summary["lane_totals"]["ou_v6_holdout"] == {
        "calls": 8,
        "credits": 16,
    }
    immutable = json.loads(Path(result.paths["immutable_manifest"]).read_text(encoding="utf-8"))
    binding = validate_ou_v6_capture_manifest_contract(root=tmp_path, manifest=immutable)
    assert binding == {
        "status": "PASS",
        "blockers": [],
        "expected_calls": 8,
        "observed_calls": 8,
    }


def test_manifest_excludes_a_fully_captured_ou_v5_contract_from_pending_calls(tmp_path):
    paths = _write_inputs(tmp_path)
    _write_v5_contract(tmp_path)
    contract = json.loads(
        (tmp_path / "config" / "wizard_ou_comparator_v5_holdout.json").read_text(encoding="utf-8")
    )
    for binding in contract["holdout_bindings"]:
        response = tmp_path / binding["response_path"]
        response.parent.mkdir(parents=True, exist_ok=True)
        response.write_text('{"captured":true}\n', encoding="utf-8")

    result = build_corrective_wizard_capture_manifest(
        root=tmp_path,
        now=NOW,
        **paths,
        copula_plan_provider=lambda _: [],
    )

    assert result.summary["status"] == "PASS"
    assert result.summary["lane_totals"]["ou_v5_holdout"] == {
        "calls": 0,
        "credits": 0,
    }
    immutable = json.loads(Path(result.paths["immutable_manifest"]).read_text(encoding="utf-8"))
    pending_binding = validate_ou_v5_capture_manifest_contract(
        root=tmp_path,
        manifest=immutable,
        pending_only=True,
    )
    assert pending_binding == {
        "status": "PASS",
        "blockers": [],
        "expected_calls": 0,
        "observed_calls": 0,
    }


def _write_inputs(root: Path) -> dict[str, Path]:
    active = root / "reports" / "active"
    active.mkdir(parents=True)
    queue_path = active / "queue.csv"
    proof_path = active / "proofs.csv"
    audit_path = active / "audit.csv"
    budget_path = active / "budget.json"
    queue = pd.DataFrame(
        [
            {
                "pair": "XLM-HBAR",
                "pair_group_id": "group-1",
                "local_interval": "1d",
                "exact_mode": "Copula",
                "orientation": "reverse",
                "proof_observations": 320,
                "wizard_period": 360,
                "vendor_custom_series_eligible": True,
            }
        ]
    )
    queue.to_csv(queue_path, index=False)
    pd.DataFrame(
        columns=[
            "pair_group_id",
            "local_interval",
            "exact_mode",
            "orientation",
            "proof_observations",
            "vendor_response_captured",
            "mode_proof_status",
        ]
    ).to_csv(proof_path, index=False)
    pd.DataFrame(
        [
            {
                "pair": "XLM-HBAR",
                "pair_group_id": "group-1",
                "local_interval": "1d",
                "exact_mode": "Copula",
                "orientation": "reverse",
                "proof_observations_requested": 320,
                "input_audit_status": "READY",
                "request_fingerprint": _json_hash({"exact": "request"}),
            }
        ]
    ).to_csv(audit_path, index=False)
    budget_path.write_text(
        json.dumps(
            {
                "generated_at_utc": NOW.isoformat(),
                "status": "PASS",
                "exact_mode_proof_credit_ceiling": 60,
                "copula_behavioral_credit_ceiling": 8,
                "ou_v3_prospective_credit_ceiling": 8,
                "ou_v4_prospective_credit_ceiling": 16,
                "ou_v5_prospective_credit_ceiling": 16,
                "ou_v6_prospective_credit_ceiling": 16,
            }
        ),
        encoding="utf-8",
    )

    bindings = []
    for pair, mode, orientation, slug in (
        ("BTC-ETH", "OU (Spread)", "original", "btc_eth_spread"),
        ("ETH-BTC", "OU (Spread)", "reverse", "eth_btc_spread"),
        ("BTC-ETH", "OU (ZScoreR)", "original", "btc_eth_zscore"),
        ("ETH-BTC", "OU (ZScoreR)", "reverse", "eth_btc_zscore"),
    ):
        request = root / "data" / "research" / "ou" / f"{slug}.json"
        request.parent.mkdir(parents=True, exist_ok=True)
        request.write_text(json.dumps({"slug": slug}) + "\n", encoding="utf-8")
        bindings.append(
            {
                "pair": pair,
                "exact_mode": mode,
                "orientation": orientation,
                "proof_observations": 360,
                "request_path": str(request.relative_to(root)),
                "request_sha256": sha256(request.read_bytes()).hexdigest(),
                "response_path": f"data/raw/ou/{slug}.json",
            }
        )
    contract = root / "config" / "wizard_ou_comparator_v3_holdout.json"
    contract.parent.mkdir(parents=True)
    contract.write_text(json.dumps({"holdout_bindings": bindings}), encoding="utf-8")
    return {
        "queue_path": queue_path,
        "proof_path": proof_path,
        "input_audit_path": audit_path,
        "credit_budget_path": budget_path,
    }


def _copula_plans() -> list[dict[str, object]]:
    plans = []
    for pair, group, orientation in (
        ("HBAR-XLM", "group-1:copula_v2", "original"),
        ("XLM-HBAR", "group-1:copula_v2", "reverse"),
        ("ADA-ALGO", "group-2:copula_v2", "original"),
        ("ALGO-ADA", "group-2:copula_v2", "reverse"),
    ):
        capture = pair.lower().replace("-", "_")
        request_payload = {"pair": pair}
        plans.append(
            {
                "capture_state": "MISSING",
                "blocker": "",
                "payload_sha256": _pretty_json_hash(request_payload),
                "base": {
                    "pair": pair,
                    "pair_group_id": group,
                    "orientation": orientation,
                    "series_length": 320,
                    "request_path": f"data/raw/copula/{capture}/request.json",
                    "response_1_path": f"data/raw/copula/{capture}/response_1.json",
                    "response_2_path": f"data/raw/copula/{capture}/response_2.json",
                    "source_backtest_request_path": f"data/research/{capture}.json",
                },
            }
        )
    return plans


def _write_v4_contract(root: Path) -> Path:
    bindings = []
    first_request: Path | None = None
    for pair_group, interval in (("MORPHO-AAVE", "1d"), ("AVAX-NEAR", "1h")):
        asset_x, asset_y = pair_group.split("-")
        for mode, mode_slug in (("OU (Spread)", "spread"), ("OU (ZScoreR)", "zscore")):
            for pair, orientation in (
                (pair_group, "original"),
                (f"{asset_y}-{asset_x}", "reverse"),
            ):
                slug = f"{pair.lower().replace('-', '_')}_{interval}_{mode_slug}"
                request = root / "data" / "research" / "ou_v4" / f"{slug}.json"
                request.parent.mkdir(parents=True, exist_ok=True)
                request.write_text(json.dumps({"slug": slug}) + "\n", encoding="utf-8")
                first_request = first_request or request
                bindings.append(
                    {
                        "pair": pair,
                        "pair_group": pair_group,
                        "interval": interval,
                        "exact_mode": mode,
                        "orientation": orientation,
                        "proof_observations": 360,
                        "request_path": str(request.relative_to(root)),
                        "request_sha256": sha256(request.read_bytes()).hexdigest(),
                        "response_path": f"data/raw/ou_v4/{slug}.json",
                    }
                )
    contract = root / "config" / "wizard_ou_comparator_v4_holdout.json"
    contract.write_text(
        json.dumps(
            {
                "status": "PREREGISTERED_WAITING_VENDOR_RESPONSES",
                "vendor_responses_at_registration": 0,
                "holdout_bindings": bindings,
            }
        ),
        encoding="utf-8",
    )
    assert first_request is not None
    return first_request


def _write_v5_contract(root: Path) -> Path:
    bindings = []
    first_request: Path | None = None
    for pair_group, interval in (("SOL-BNB", "1d"), ("DOGE-FET", "1h")):
        asset_x, asset_y = pair_group.split("-")
        for mode, mode_slug in (("OU (Spread)", "spread"), ("OU (ZScoreR)", "zscore")):
            for pair, orientation in (
                (pair_group, "original"),
                (f"{asset_y}-{asset_x}", "reverse"),
            ):
                slug = f"{pair.lower().replace('-', '_')}_{interval}_{mode_slug}"
                request = root / "data" / "research" / "ou_v5" / f"{slug}.json"
                request.parent.mkdir(parents=True, exist_ok=True)
                request.write_text(json.dumps({"slug": slug}) + "\n", encoding="utf-8")
                first_request = first_request or request
                bindings.append(
                    {
                        "pair": pair,
                        "pair_group": pair_group,
                        "interval": interval,
                        "exact_mode": mode,
                        "orientation": orientation,
                        "proof_observations": 360,
                        "request_path": str(request.relative_to(root)),
                        "request_sha256": sha256(request.read_bytes()).hexdigest(),
                        "response_path": f"data/raw/ou_v5/{slug}.json",
                    }
                )
    contract = root / "config" / "wizard_ou_comparator_v5_holdout.json"
    contract.write_text(
        json.dumps(
            {
                "status": "PREREGISTERED_WAITING_VENDOR_RESPONSES",
                "vendor_responses_at_registration": 0,
                "holdout_bindings": bindings,
            }
        ),
        encoding="utf-8",
    )
    assert first_request is not None
    return first_request


def _write_v6_contract(root: Path) -> Path:
    bindings = []
    first_request: Path | None = None
    for pair_group, interval in (("APT-ATOM", "1d"), ("ARB-OP", "1h")):
        asset_x, asset_y = pair_group.split("-")
        for mode, mode_slug in (("OU (Spread)", "spread"), ("OU (ZScoreR)", "zscore")):
            for pair, orientation in (
                (pair_group, "original"),
                (f"{asset_y}-{asset_x}", "reverse"),
            ):
                slug = f"{pair.lower().replace('-', '_')}_{interval}_{mode_slug}"
                request = root / "data" / "research" / "ou_v6" / f"{slug}.json"
                request.parent.mkdir(parents=True, exist_ok=True)
                request.write_text(json.dumps({"slug": slug}) + "\n", encoding="utf-8")
                first_request = first_request or request
                bindings.append(
                    {
                        "pair": pair,
                        "pair_group": pair_group,
                        "interval": interval,
                        "exact_mode": mode,
                        "orientation": orientation,
                        "proof_observations": 360,
                        "request_path": str(request.relative_to(root)),
                        "request_sha256": sha256(request.read_bytes()).hexdigest(),
                        "response_path": f"data/raw/ou_v6/{slug}.json",
                    }
                )
    contract = root / "config" / "wizard_ou_comparator_v6_holdout.json"
    contract.write_text(
        json.dumps(
            {
                "status": "PREREGISTERED_WAITING_VENDOR_RESPONSES",
                "vendor_responses_at_registration": 0,
                "final_successor_iteration": True,
                "successor_after_v6_failure_allowed": False,
                "holdout_bindings": bindings,
            }
        ),
        encoding="utf-8",
    )
    assert first_request is not None
    return first_request


def _json_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return sha256(encoded).hexdigest()


def _pretty_json_hash(payload: object) -> str:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return sha256(encoded).hexdigest()
