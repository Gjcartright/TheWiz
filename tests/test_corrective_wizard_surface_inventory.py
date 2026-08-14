from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from quant_platform.orchestration.corrective_wizard_surface_inventory import (
    API_CONSUMERS,
    API_CONTRACTS,
    CAPTURE_OPPORTUNITIES,
    DASHBOARD_SURFACES,
    MISSING_INTEGRATIONS,
    OPTIMIZATION_DIMENSIONS,
    PARITY_SURFACES,
    build_wizard_surface_inventory_readiness,
    validate_wizard_surface_inventory_readiness,
)

NOW = datetime(2026, 8, 13, 5, 0, tzinfo=UTC)


def _write_fixture(root: Path) -> None:
    reports = root / "reports"
    docs = root / "docs"
    config = root / "config"
    reports.mkdir(parents=True)
    docs.mkdir(parents=True)
    config.mkdir(parents=True)
    (config / "wizard_surface_inventory_contract.json").write_text(
        json.dumps(
            {
                "schema_version": "thewiz.wizard_surface_inventory_readiness.v2",
                "enforced": True,
                "api_endpoint_count": 14,
                "dashboard_surface_count": 29,
                "missing_integration_count": 28,
                "capture_opportunity_count": 21,
                "browser_api_parity_surface_count": 12,
                "optimization_dimension_count": 17,
                "candidate_promotion_authority": False,
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    evidence = reports / "active" / "wizard-evidence.json"
    evidence.parent.mkdir(parents=True)
    evidence.write_text('{"status":"PASS"}\n', encoding="utf-8")

    for relative in set(API_CONSUMERS.values()):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture\n", encoding="utf-8")

    api_rows = []
    for name, (method, path, credits) in API_CONTRACTS.items():
        api_rows.append(
            {
                "endpoint_name": name,
                "method": method,
                "path": path,
                "credits": credits,
                "live_verification_status": "live_verified_fixture",
                "live_evidence_path": str(evidence.relative_to(root)),
                "point_in_time_policy": "diagnostic_until_causal_replay",
                "repo_integration_status": "integrated_or_explicit_partial",
                "missing_integration": "formula parity remains",
                "recommended_priority": "must_integrate",
                "promotion_authority": False,
                "live_trading_authorized": False,
            }
        )
    pd.DataFrame(api_rows).to_csv(reports / "crypto_wizards_api_full_inventory.csv", index=False)

    dashboard_rows = [
        {
            "page_or_section": name,
            "url": "https://cryptowizards.net/wizards",
            "fields_or_metrics": "fixture metrics",
            "filters_or_controls": "fixture controls",
            "point_in_time_status": "mixed_reconstructable",
            "already_ingested": "partial",
            "current_repo_gap": "point-in-time parity remains",
            "integration_priority": "must_integrate",
            "risk_notes": "discovery only",
        }
        for name in sorted(DASHBOARD_SURFACES)
    ]
    pd.DataFrame(dashboard_rows).to_csv(reports / "dashboard_full_inventory.csv", index=False)

    missing_rows = [
        {
            "feature": name,
            "source_page_or_endpoint": "fixture",
            "missing_or_partial": "partial",
            "why_it_matters": "research lineage",
            "recommended_integration": "capture causally",
            "priority": "must_integrate",
            "blocker_or_risk": "hindsight risk",
        }
        for name in sorted(MISSING_INTEGRATIONS)
    ]
    pd.DataFrame(missing_rows).to_csv(reports / "dashboard_missing_integrations.csv", index=False)

    capture_rows = [
        {
            "feature": name,
            "source": "fixture",
            "method": "read_only_capture",
            "frequency": "registered",
            "required_context": "timestamp and configuration",
            "pit_safety": "discovery_only",
            "storage_target": "data/research/fixture",
            "priority": "must_integrate",
            "notes": "no authority",
        }
        for name in sorted(CAPTURE_OPPORTUNITIES)
    ]
    pd.DataFrame(capture_rows).to_csv(reports / "dashboard_capture_opportunities.csv", index=False)

    parity_rows = [
        {
            "surface": name,
            "dashboard_contract": "fixture dashboard",
            "api_contract": "fixture api",
            "parity_finding": "diagnostic only",
            "system_rule": "local acceptance required",
            "evidence_date": NOW.date().isoformat(),
        }
        for name in sorted(PARITY_SURFACES)
    ]
    pd.DataFrame(parity_rows).to_csv(reports / "crypto_wizards_browser_api_parity.csv", index=False)

    optimization_rows = [
        {
            "experiment_dimension": name,
            "values_to_compare": "registered values",
            "hold_constant": "all other inputs",
            "primary_readout": "after-cost OOS evidence",
            "invalidation_rule": "fail closed on drift",
            "optimization_policy": "purged_walk_forward",
            "promotion_authority": False,
        }
        for name in sorted(OPTIMIZATION_DIMENSIONS)
    ]
    pd.DataFrame(optimization_rows).to_csv(
        reports / "crypto_wizards_dashboard_optimization_matrix.csv", index=False
    )

    (docs / "dashboard_field_dictionary.md").write_text(
        "# Dashboard Field Dictionary\n\nNo hindsight fields authorize acceptance.\n",
        encoding="utf-8",
    )
    (reports / "dashboard_integration_summary.md").write_text(
        "# Integration Summary\n\nResearch only.\n", encoding="utf-8"
    )
    (reports / "crypto_wizards_inspector_second_pass.md").write_text(
        "# Inspector Report\n\nAll surfaces inventoried; integration remains gated.\n",
        encoding="utf-8",
    )


def test_complete_inventory_builds_stable_immutable_non_authority_receipt(tmp_path):
    _write_fixture(tmp_path)

    first = build_wizard_surface_inventory_readiness(root=tmp_path, now=NOW)
    second = build_wizard_surface_inventory_readiness(
        root=tmp_path,
        now=NOW.replace(minute=1),
    )
    validation = validate_wizard_surface_inventory_readiness(root=tmp_path)

    assert first.summary["status"] == "PASS_INVENTORY_CONTRACT"
    assert first.summary["checks_passed"] == first.summary["checks_total"]
    assert first.summary["api_endpoint_count"] == 14
    assert first.summary["dashboard_surface_count"] == 29
    assert first.summary["integration_status"] == "IN_PROGRESS_RECORDED_GAPS"
    assert first.summary["receipt_id"] == second.summary["receipt_id"]
    assert first.summary["receipt_sha256"] == second.summary["receipt_sha256"]
    assert validation["status"] == "PASS"
    assert first.summary["candidate_promotion_authority"] is False
    assert first.summary["testnet_order_authority"] is False
    assert first.summary["live_trading_authorized"] is False


def test_missing_api_contract_blocks_inventory_completeness(tmp_path):
    _write_fixture(tmp_path)
    path = tmp_path / "reports/crypto_wizards_api_full_inventory.csv"
    frame = pd.read_csv(path)
    frame = frame[frame["endpoint_name"] != "copula_post"]
    frame.to_csv(path, index=False)

    result = build_wizard_surface_inventory_readiness(root=tmp_path, now=NOW)

    assert result.summary["status"] == "BLOCKED"
    assert result.summary["inventory_complete"] is False
    assert any("missing:copula_post" in blocker for blocker in result.summary["blockers"])
    assert result.summary["testnet_order_authority"] is False


def test_hindsight_inventory_cannot_claim_promotion_authority(tmp_path):
    _write_fixture(tmp_path)
    path = tmp_path / "reports/crypto_wizards_api_full_inventory.csv"
    frame = pd.read_csv(path)
    frame.loc[frame["endpoint_name"] == "backtest_get", "promotion_authority"] = True
    frame.to_csv(path, index=False)

    result = build_wizard_surface_inventory_readiness(root=tmp_path, now=NOW)

    assert result.summary["status"] == "BLOCKED"
    assert "backtest_get" in result.summary["blockers"]
    assert result.summary["candidate_promotion_authority"] is False


def test_symlinked_api_evidence_is_rejected(tmp_path):
    _write_fixture(tmp_path)
    target = tmp_path / "reports/active/wizard-evidence.json"
    link = tmp_path / "reports/active/wizard-evidence-link.json"
    link.symlink_to(target)
    path = tmp_path / "reports/crypto_wizards_api_full_inventory.csv"
    frame = pd.read_csv(path)
    frame["live_evidence_path"] = str(link.relative_to(tmp_path))
    frame.to_csv(path, index=False)

    result = build_wizard_surface_inventory_readiness(root=tmp_path, now=NOW)

    assert result.summary["status"] == "BLOCKED"
    assert any("backtest_get" in blocker for blocker in result.summary["blockers"])


def test_active_pointer_tamper_invalidates_inventory_receipt(tmp_path):
    _write_fixture(tmp_path)
    build_wizard_surface_inventory_readiness(root=tmp_path, now=NOW)
    path = tmp_path / "reports/active/wizard_surface_inventory_readiness.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["integration_status"] = "COMPLETE"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    validation = validate_wizard_surface_inventory_readiness(root=tmp_path)

    assert validation["status"] == "BLOCKED"
    assert validation["blocker"].startswith("invalid_inventory_readiness:")


def test_source_artifact_drift_invalidates_registered_inventory(tmp_path):
    _write_fixture(tmp_path)
    build_wizard_surface_inventory_readiness(root=tmp_path, now=NOW)
    source = tmp_path / "reports/dashboard_integration_summary.md"
    source.write_text("# Drifted integration summary\n", encoding="utf-8")

    validation = validate_wizard_surface_inventory_readiness(root=tmp_path)

    assert validation["status"] == "BLOCKED"
    assert validation["blocker"].startswith("invalid_inventory_readiness:")
