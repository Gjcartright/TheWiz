"""Versioned Crypto Wizards dashboard/API surface inventory readiness."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd

from quant_platform.active_pipeline import CommandResult
from quant_platform.orchestration.corrective_runtime import (
    promote_staged_file,
    write_immutable_json,
)

ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "thewiz.wizard_surface_inventory_readiness.v2"
MAX_PARITY_EVIDENCE_AGE_DAYS = 30

API_CONTRACTS = {
    "backtest_get": ("GET", "/v1beta/backtest", 6),
    "cointegration_get": ("GET", "/v1beta/cointegration", 5),
    "copula_get": ("GET", "/v1beta/copula", 5),
    "correlations_get": ("GET", "/v1beta/correlations", 5),
    "credits_used_get": ("GET", "/v1beta/credits-used", 0),
    "prescanned_get": ("GET", "/v1beta/prescanned", 10),
    "spread_get": ("GET", "/v1beta/spread", 5),
    "zscores_get": ("GET", "/v1beta/zscores", 5),
    "backtest_post": ("POST", "/v1beta/backtest", 2),
    "cointegration_post": ("POST", "/v1beta/cointegration", 1),
    "copula_post": ("POST", "/v1beta/copula", 1),
    "correlations_post": ("POST", "/v1beta/correlations", 1),
    "spread_post": ("POST", "/v1beta/spread", 1),
    "zscores_post": ("POST", "/v1beta/zscores", 1),
}

DASHBOARD_SURFACES = {
    "Members Area",
    "Authenticated Session State",
    "API Service",
    "Backtest Guide",
    "Correlation Guide",
    "ECM Guide",
    "Scanner",
    "Scanner Expanded Row",
    "Pair Header",
    "Betas",
    "Return Correlations",
    "Copula Statistics And Charts",
    "ECM Y And ECM X",
    "ECM Strength",
    "Volatilities And Conditional GARCH",
    "Backtest Machine",
    "Strategy Settings Drawer",
    "Paper Trade Form",
    "Trades Journal",
    "Alerts",
    "Custom Analysis",
    "Data Export",
    "Downloads",
    "Code Listing",
    "Courses",
    "Account",
    "Prescanned API",
    "Analytics APIs",
    "Own-Data POST APIs",
}

MISSING_INTEGRATIONS = {
    "Authenticated dashboard preflight",
    "Daily and Hourly browser capture completion",
    "Strategy versus supporting spread context",
    "Backtest metric accounting integrity",
    "Daily full venue and strategy sweep",
    "Complete run configuration provenance",
    "Seven exact-mode comparison for every shortlisted row",
    "Raw pair-page chart histories",
    "Johansen detail and history",
    "Dynamic ECM; ECM strength; and impulse histories",
    "Conditional beta and GARCH histories",
    "Copula contour points and family stability",
    "Wizard Data export ingestion",
    "Wizard paper journal reconciliation",
    "Alert manifest and health",
    "API contract tests",
    "API credit budget blocker",
    "Hyperliquid symbol and venue bridge",
    "Point-in-time feature policy enforcement",
    "Backtest parity and failure attribution",
    "Download knowledge evidence registry",
    "Stationarity badge provenance",
    "Pair-page metric accounting integrity",
    "Pair-timeframe volatility quality",
    "Authenticated inspector status",
    "POST analytics semantic parity",
    "4 Hour API contract",
    "Restore action safety",
}

CAPTURE_OPPORTUNITIES = {
    "Authenticated capture preflight",
    "Scanner full sweep",
    "Scanner browser parity",
    "Expanded exact-mode table",
    "Pair header snapshot",
    "Settings manifest",
    "Backtest result and paths",
    "Spread and Z histories",
    "Copula detail",
    "ECM detail",
    "Conditional beta and volatility",
    "OHLCV export",
    "Paper trades",
    "Alert definitions",
    "API credits",
    "API contract fixtures",
    "Guide and download references",
    "Stationarity computed-color proof",
    "Pair-page accounting and quality state",
    "Inspector route receipt",
    "POST analytics live contract fixture",
}

PARITY_SURFACES = {
    "venues",
    "timeframes",
    "exact modes",
    "stationarity",
    "ECM",
    "volatility",
    "copula",
    "backtest controls",
    "costs",
    "performance accounting",
    "raw market data",
    "paper/alerts",
}

OPTIMIZATION_DIMENSIONS = {
    "scanner interval",
    "scanner exchange",
    "scanner priority",
    "cointegration overlay",
    "dependency overlays",
    "exact mode",
    "orientation",
    "lookback",
    "entry",
    "exit",
    "rolling window",
    "risk overrides",
    "capital weighting",
    "metrics mode",
    "cost stress",
    "copula contour",
    "dependency chart",
}

ALLOWED_PRIORITIES = {
    "must_integrate",
    "useful",
    "optional",
    "not_useful",
    "risky_or_hindsight",
}

SOURCE_FILES = {
    "contract": Path("config/wizard_surface_inventory_contract.json"),
    "api_inventory": Path("reports/crypto_wizards_api_full_inventory.csv"),
    "dashboard_inventory": Path("reports/dashboard_full_inventory.csv"),
    "missing_integrations": Path("reports/dashboard_missing_integrations.csv"),
    "capture_opportunities": Path("reports/dashboard_capture_opportunities.csv"),
    "browser_api_parity": Path("reports/crypto_wizards_browser_api_parity.csv"),
    "optimization_matrix": Path("reports/crypto_wizards_dashboard_optimization_matrix.csv"),
    "field_dictionary": Path("docs/dashboard_field_dictionary.md"),
    "integration_summary": Path("reports/dashboard_integration_summary.md"),
    "inspector_report": Path("reports/crypto_wizards_inspector_second_pass.md"),
}

REQUIRED_COLUMNS = {
    "api_inventory": {
        "endpoint_name",
        "method",
        "path",
        "credits",
        "live_verification_status",
        "live_evidence_path",
        "point_in_time_policy",
        "repo_integration_status",
        "missing_integration",
        "recommended_priority",
        "promotion_authority",
        "live_trading_authorized",
    },
    "dashboard_inventory": {
        "page_or_section",
        "url",
        "fields_or_metrics",
        "filters_or_controls",
        "point_in_time_status",
        "already_ingested",
        "current_repo_gap",
        "integration_priority",
        "risk_notes",
    },
    "missing_integrations": {
        "feature",
        "source_page_or_endpoint",
        "missing_or_partial",
        "why_it_matters",
        "recommended_integration",
        "priority",
        "blocker_or_risk",
    },
    "capture_opportunities": {
        "feature",
        "source",
        "method",
        "frequency",
        "required_context",
        "pit_safety",
        "storage_target",
        "priority",
        "notes",
    },
    "browser_api_parity": {
        "surface",
        "dashboard_contract",
        "api_contract",
        "parity_finding",
        "system_rule",
        "evidence_date",
    },
    "optimization_matrix": {
        "experiment_dimension",
        "values_to_compare",
        "hold_constant",
        "primary_readout",
        "invalidation_rule",
        "optimization_policy",
        "promotion_authority",
    },
}

API_CONSUMERS = {
    "credits_used_get": "src/quant_platform/wizard_credit_ledger.py",
    "prescanned_get": "src/quant_platform/crypto_wizards_sweep.py",
    "backtest_get": "src/quant_platform/orchestration/wizard_pair_detail_api_pilot.py",
    "backtest_post": "src/quant_platform/wizard_hyperliquid_mode_proof.py",
    "cointegration_get": "src/quant_platform/orchestration/wizard_pair_detail_api_pilot.py",
    "copula_get": "src/quant_platform/orchestration/wizard_pair_detail_api_pilot.py",
    "correlations_get": "src/quant_platform/orchestration/wizard_pair_detail_api_pilot.py",
    "spread_get": "src/quant_platform/orchestration/wizard_pair_detail_api_pilot.py",
    "zscores_get": "src/quant_platform/crypto_wizards_history.py",
    "cointegration_post": "scripts/run_crypto_wizards_post_contract_probe.py",
    "copula_post": "src/quant_platform/orchestration/corrective_wizard_copula_behavioral.py",
    "correlations_post": "scripts/run_crypto_wizards_post_contract_probe.py",
    "spread_post": "scripts/run_crypto_wizards_post_contract_probe.py",
    "zscores_post": "scripts/run_crypto_wizards_post_contract_probe.py",
}


def build_wizard_surface_inventory_readiness(
    *, root: Path = ROOT, now: datetime | None = None
) -> CommandResult:
    """Validate exhaustive inventory identity, evidence, and non-authority."""

    checked_at = _as_utc(now)
    checks: list[dict[str, Any]] = []
    frames: dict[str, pd.DataFrame] = {}
    source_artifacts: list[dict[str, str]] = []

    for name, relative in SOURCE_FILES.items():
        path = root / relative
        valid = path.is_file() and not path.is_symlink() and path.stat().st_size > 0
        _check(
            checks,
            f"source_{name}",
            "source_evidence",
            valid,
            "" if valid else f"missing_or_invalid_source:{relative}",
            relative,
        )
        if valid:
            source_artifacts.append(
                {"path": str(relative), "sha256": sha256(path.read_bytes()).hexdigest()}
            )
        if name in REQUIRED_COLUMNS and valid:
            try:
                frame = pd.read_csv(path).fillna("")
            except (OSError, TypeError, ValueError, pd.errors.ParserError) as exc:
                _check(
                    checks,
                    f"schema_{name}",
                    "schema",
                    False,
                    f"unreadable_csv:{type(exc).__name__}",
                    relative,
                )
                continue
            frames[name] = frame
            missing_columns = sorted(REQUIRED_COLUMNS[name] - set(frame.columns))
            _check(
                checks,
                f"schema_{name}",
                "schema",
                not missing_columns,
                "" if not missing_columns else "missing_columns:" + ";".join(missing_columns),
                relative,
            )

    contract_path = root / SOURCE_FILES["contract"]
    contract_failures: list[str] = []
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        expected_counts = {
            "api_endpoint_count": len(API_CONTRACTS),
            "dashboard_surface_count": len(DASHBOARD_SURFACES),
            "missing_integration_count": len(MISSING_INTEGRATIONS),
            "capture_opportunity_count": len(CAPTURE_OPPORTUNITIES),
            "browser_api_parity_surface_count": len(PARITY_SURFACES),
            "optimization_dimension_count": len(OPTIMIZATION_DIMENSIONS),
        }
        if contract.get("schema_version") != SCHEMA_VERSION:
            contract_failures.append("schema_version")
        if contract.get("enforced") is not True:
            contract_failures.append("enforced")
        for field, expected in expected_counts.items():
            if _as_int(contract.get(field)) != expected:
                contract_failures.append(field)
        if contract.get("candidate_promotion_authority") is not False:
            contract_failures.append("candidate_promotion_authority")
        if contract.get("testnet_order_authority") is not False:
            contract_failures.append("testnet_order_authority")
        if contract.get("live_trading_authorized") is not False:
            contract_failures.append("live_trading_authorized")
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        contract_failures.append(f"unreadable_contract:{type(exc).__name__}")
    _check_list(checks, "versioned_inventory_contract", "governance", contract_failures)

    api = frames.get("api_inventory")
    if api is not None and REQUIRED_COLUMNS["api_inventory"].issubset(api.columns):
        identities = api["endpoint_name"].astype(str).tolist()
        _identity_check(checks, "api_endpoints", identities, set(API_CONTRACTS))
        contract_failures: list[str] = []
        evidence_failures: list[str] = []
        authority_failures: list[str] = []
        consumer_failures: list[str] = []
        for row in api.to_dict("records"):
            name = str(row["endpoint_name"])
            expected = API_CONTRACTS.get(name)
            observed = (
                str(row["method"]).upper(),
                str(row["path"]),
                _as_int(row["credits"]),
            )
            if expected is None or observed != expected:
                contract_failures.append(f"{name}:{observed}")
            evidence_path = _safe_existing_path(root, str(row["live_evidence_path"]))
            if (
                evidence_path is None
                or evidence_path.stat().st_size <= 0
                or not str(row["live_verification_status"]).startswith("live_verified")
            ):
                evidence_failures.append(name)
            if _as_bool(row["promotion_authority"]) or _as_bool(row["live_trading_authorized"]):
                authority_failures.append(name)
            consumer = root / API_CONSUMERS.get(name, ".missing-consumer")
            if not consumer.is_file():
                consumer_failures.append(name)
        _check_list(checks, "api_contract_values", "api_contract", contract_failures)
        _check_list(checks, "api_live_evidence", "api_contract", evidence_failures)
        _check_list(checks, "api_zero_authority", "authority", authority_failures)
        _check_list(checks, "api_consumer_modules", "integration", consumer_failures)

    dashboard = frames.get("dashboard_inventory")
    if dashboard is not None and REQUIRED_COLUMNS["dashboard_inventory"].issubset(
        dashboard.columns
    ):
        identities = dashboard["page_or_section"].astype(str).tolist()
        _identity_check(checks, "dashboard_surfaces", identities, DASHBOARD_SURFACES)
        priority_failures = [
            str(row["page_or_section"])
            for row in dashboard.to_dict("records")
            if str(row["integration_priority"]) not in ALLOWED_PRIORITIES
        ]
        pit_failures = [
            str(row["page_or_section"])
            for row in dashboard.to_dict("records")
            if not str(row["point_in_time_status"]).strip()
            or (
                str(row["integration_priority"]) in {"must_integrate", "risky_or_hindsight"}
                and not str(row["current_repo_gap"]).strip()
            )
        ]
        _check_list(checks, "dashboard_priorities", "dashboard_contract", priority_failures)
        _check_list(checks, "dashboard_pit_and_gap_policy", "dashboard_contract", pit_failures)

    missing = frames.get("missing_integrations")
    if missing is not None and REQUIRED_COLUMNS["missing_integrations"].issubset(missing.columns):
        _identity_check(
            checks,
            "missing_integration_register",
            missing["feature"].astype(str).tolist(),
            MISSING_INTEGRATIONS,
        )
        failures = [
            str(row["feature"])
            for row in missing.to_dict("records")
            if str(row["priority"]) not in ALLOWED_PRIORITIES
            or not str(row["blocker_or_risk"]).strip()
        ]
        _check_list(checks, "missing_integration_policy", "integration", failures)

    captures = frames.get("capture_opportunities")
    if captures is not None and REQUIRED_COLUMNS["capture_opportunities"].issubset(
        captures.columns
    ):
        _identity_check(
            checks,
            "capture_opportunity_register",
            captures["feature"].astype(str).tolist(),
            CAPTURE_OPPORTUNITIES,
        )
        failures = [
            str(row["feature"])
            for row in captures.to_dict("records")
            if str(row["priority"]) not in ALLOWED_PRIORITIES
            or not str(row["pit_safety"]).strip()
            or not str(row["storage_target"]).strip()
        ]
        _check_list(checks, "capture_pit_policy", "capture_contract", failures)

    parity = frames.get("browser_api_parity")
    if parity is not None and REQUIRED_COLUMNS["browser_api_parity"].issubset(parity.columns):
        _identity_check(
            checks,
            "browser_api_parity_surfaces",
            parity["surface"].astype(str).tolist(),
            PARITY_SURFACES,
        )
        date_failures: list[str] = []
        for row in parity.to_dict("records"):
            observed = pd.to_datetime(row["evidence_date"], utc=True, errors="coerce")
            if (
                pd.isna(observed)
                or observed > pd.Timestamp(checked_at)
                or pd.Timestamp(checked_at) - observed
                > pd.Timedelta(days=MAX_PARITY_EVIDENCE_AGE_DAYS)
            ):
                date_failures.append(str(row["surface"]))
        _check_list(checks, "browser_api_parity_freshness", "freshness", date_failures)

    optimization = frames.get("optimization_matrix")
    if optimization is not None and REQUIRED_COLUMNS["optimization_matrix"].issubset(
        optimization.columns
    ):
        _identity_check(
            checks,
            "optimization_dimensions",
            optimization["experiment_dimension"].astype(str).tolist(),
            OPTIMIZATION_DIMENSIONS,
        )
        authority_failures = [
            str(row["experiment_dimension"])
            for row in optimization.to_dict("records")
            if _as_bool(row["promotion_authority"])
            or not str(row["optimization_policy"]).strip()
            or not str(row["invalidation_rule"]).strip()
        ]
        _check_list(
            checks,
            "optimization_zero_authority",
            "authority",
            authority_failures,
        )

    matrix = _build_integration_matrix(frames)
    matrix_path = root / "reports" / "active" / "wizard_surface_integration_matrix.csv"
    checks_path = root / "reports" / "active" / "wizard_surface_inventory_checks.csv"
    status_path = root / "reports" / "active" / "wizard_surface_inventory_readiness.json"
    markdown_path = root / "reports" / "active" / "wizard_surface_inventory_readiness.md"
    _atomic_csv(pd.DataFrame(checks), checks_path)
    _atomic_csv(matrix, matrix_path)

    blockers = sorted(
        str(row["blocker"]) for row in checks if row["status"] != "PASS" and row["blocker"]
    )
    integration_gap_count = len(frames.get("missing_integrations", pd.DataFrame()))
    core: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "checked_at_utc": checked_at.isoformat(),
        "status": "PASS_INVENTORY_CONTRACT" if not blockers else "BLOCKED",
        "inventory_complete": not blockers,
        "integration_status": (
            "IN_PROGRESS_RECORDED_GAPS"
            if not blockers and integration_gap_count
            else "COMPLETE"
            if not blockers
            else "BLOCKED"
        ),
        "api_endpoint_count": len(api) if api is not None else 0,
        "dashboard_surface_count": len(dashboard) if dashboard is not None else 0,
        "missing_integration_count": integration_gap_count,
        "capture_opportunity_count": len(captures) if captures is not None else 0,
        "browser_api_parity_surface_count": len(parity) if parity is not None else 0,
        "optimization_dimension_count": len(optimization) if optimization is not None else 0,
        "checks_total": len(checks),
        "checks_passed": sum(row["status"] == "PASS" for row in checks),
        "blockers": blockers,
        "source_artifacts": sorted(source_artifacts, key=lambda item: item["path"]),
        "source_artifacts_sha256": sha256(
            _canonical_json(sorted(source_artifacts, key=lambda item: item["path"])).encode("utf-8")
        ).hexdigest(),
        "research_only": True,
        "candidate_promotion_authority": False,
        "order_submission_included": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    scientific_core = dict(core)
    scientific_core.pop("checked_at_utc")
    receipt_id = (
        "wizardsurfaceinventory_"
        + sha256(_canonical_json(scientific_core).encode("utf-8")).hexdigest()[:20]
    )
    receipt = {**scientific_core, "receipt_id": receipt_id}
    receipt_path = (
        root / "data" / "research" / "wizard_surface_inventory_readiness" / f"{receipt_id}.json"
    )
    _write_immutable_json(receipt, receipt_path)
    receipt_sha = sha256(receipt_path.read_bytes()).hexdigest()
    summary = {
        **receipt,
        "checked_at_utc": checked_at.isoformat(),
        "receipt_path": _relative(receipt_path, root),
        "receipt_sha256": receipt_sha,
        "evidence_paths": {
            "checks": _relative(checks_path, root),
            "integration_matrix": _relative(matrix_path, root),
            "status": _relative(status_path, root),
            "summary": _relative(markdown_path, root),
            "immutable_receipt": _relative(receipt_path, root),
        },
    }
    _atomic_json(summary, status_path)
    _atomic_text(_markdown(summary), markdown_path)
    return CommandResult(
        paths={
            "checks": checks_path,
            "integration_matrix": matrix_path,
            "status": status_path,
            "summary": markdown_path,
            "immutable_receipt": receipt_path,
        },
        summary=summary,
    )


def validate_wizard_surface_inventory_readiness(
    *, root: Path = ROOT, status_path: Path | None = None
) -> dict[str, Any]:
    """Verify the active pointer and immutable inventory receipt."""

    status_path = status_path or root / "reports/active/wizard_surface_inventory_readiness.json"
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if not isinstance(status, dict):
            raise TypeError("status must be an object")
        receipt_path = _safe_existing_path(root, str(status.get("receipt_path", "")))
        if receipt_path is None:
            raise ValueError("receipt path invalid")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        expected = dict(status)
        for key in (
            "checked_at_utc",
            "receipt_path",
            "receipt_sha256",
            "evidence_paths",
        ):
            expected.pop(key, None)
        if (
            receipt != expected
            or sha256(receipt_path.read_bytes()).hexdigest() != status.get("receipt_sha256")
            or status.get("schema_version") != SCHEMA_VERSION
            or status.get("status") != "PASS_INVENTORY_CONTRACT"
            or status.get("inventory_complete") is not True
            or status.get("candidate_promotion_authority") is not False
            or status.get("testnet_order_authority") is not False
            or status.get("live_trading_authorized") is not False
        ):
            raise ValueError("inventory readiness binding invalid")
        source_artifacts = receipt.get("source_artifacts")
        if not isinstance(source_artifacts, list) or not source_artifacts:
            raise ValueError("inventory source artifacts missing")
        observed_artifacts: list[dict[str, str]] = []
        for item in source_artifacts:
            if not isinstance(item, dict):
                raise TypeError("inventory source artifact must be an object")
            source_path = _safe_existing_path(root, str(item.get("path", "")))
            if source_path is None:
                raise ValueError("inventory source artifact path invalid")
            observed_sha = sha256(source_path.read_bytes()).hexdigest()
            if observed_sha != item.get("sha256"):
                raise ValueError("inventory source artifact hash mismatch")
            observed_artifacts.append({"path": str(item["path"]), "sha256": observed_sha})
        observed_artifacts.sort(key=lambda item: item["path"])
        if observed_artifacts != source_artifacts or sha256(
            _canonical_json(observed_artifacts).encode("utf-8")
        ).hexdigest() != receipt.get("source_artifacts_sha256"):
            raise ValueError("inventory source artifact set mismatch")
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return {"status": "BLOCKED", "blocker": f"invalid_inventory_readiness:{type(exc).__name__}"}
    return {
        "status": "PASS",
        "blocker": "",
        "receipt_id": str(status["receipt_id"]),
        "receipt_path": _relative(receipt_path, root),
        "receipt_sha256": str(status["receipt_sha256"]),
        "api_endpoint_count": int(status["api_endpoint_count"]),
        "dashboard_surface_count": int(status["dashboard_surface_count"]),
        "integration_status": str(status["integration_status"]),
        "candidate_promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }


def _build_integration_matrix(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    api = frames.get("api_inventory", pd.DataFrame())
    if REQUIRED_COLUMNS["api_inventory"].issubset(api.columns):
        for row in api.to_dict("records"):
            name = str(row["endpoint_name"])
            rows.append(
                {
                    "surface_type": "api_endpoint",
                    "surface_id": name,
                    "source_locator": f"{row['method']} {row['path']}",
                    "point_in_time_policy": str(row["point_in_time_policy"]),
                    "integration_status": str(row["repo_integration_status"]),
                    "integration_priority": str(row["recommended_priority"]),
                    "consumer_module": API_CONSUMERS.get(name, ""),
                    "captured_evidence_path": str(row["live_evidence_path"]),
                    "remaining_gap": str(row["missing_integration"]),
                    "acceptance_authority": False,
                    "testnet_order_authority": False,
                    "live_trading_authorized": False,
                }
            )
    dashboard = frames.get("dashboard_inventory", pd.DataFrame())
    if REQUIRED_COLUMNS["dashboard_inventory"].issubset(dashboard.columns):
        for row in dashboard.to_dict("records"):
            name = str(row["page_or_section"])
            rows.append(
                {
                    "surface_type": "dashboard_surface",
                    "surface_id": name,
                    "source_locator": str(row["url"]),
                    "point_in_time_policy": str(row["point_in_time_status"]),
                    "integration_status": str(row["already_ingested"]),
                    "integration_priority": str(row["integration_priority"]),
                    "consumer_module": _dashboard_consumer(name),
                    "captured_evidence_path": "reports/dashboard_full_inventory.csv",
                    "remaining_gap": str(row["current_repo_gap"]),
                    "acceptance_authority": False,
                    "testnet_order_authority": False,
                    "live_trading_authorized": False,
                }
            )
    return pd.DataFrame(rows)


def _dashboard_consumer(name: str) -> str:
    if name in {"Scanner", "Scanner Expanded Row", "Prescanned API"}:
        return "src/quant_platform/crypto_wizards_sweep.py"
    if name in {"API Service", "Authenticated Session State"}:
        return "src/quant_platform/wizard_credit_ledger.py"
    if name in {"Backtest Machine", "Strategy Settings Drawer"}:
        return "src/quant_platform/wizard_hyperliquid_mode_proof.py"
    if name in {"Paper Trade Form", "Trades Journal"}:
        return "src/quant_platform/wizard_research_journal.py"
    if name in {"Account", "Alerts", "Code Listing"}:
        return "no_research_consumer_by_design"
    if name in {
        "Members Area",
        "Backtest Guide",
        "Correlation Guide",
        "ECM Guide",
        "Downloads",
        "Courses",
    }:
        return "docs/crypto_wizards_dashboard_operator_playbook.md"
    return "src/quant_platform/pair_detail_ingestion.py"


def _identity_check(
    checks: list[dict[str, Any]], check_id: str, values: list[str], expected: set[str]
) -> None:
    observed = set(values)
    duplicates = sorted({value for value in values if values.count(value) > 1})
    failures = []
    if observed != expected:
        failures.extend(f"missing:{value}" for value in sorted(expected - observed))
        failures.extend(f"unexpected:{value}" for value in sorted(observed - expected))
    failures.extend(f"duplicate:{value}" for value in duplicates)
    _check_list(checks, check_id, "identity", failures)


def _check_list(
    checks: list[dict[str, Any]], check_id: str, component: str, failures: list[str]
) -> None:
    _check(
        checks,
        check_id,
        component,
        not failures,
        "" if not failures else ";".join(sorted(set(failures))),
        "",
    )


def _check(
    checks: list[dict[str, Any]],
    check_id: str,
    component: str,
    passed: bool,
    blocker: str,
    evidence_path: Path | str,
) -> None:
    checks.append(
        {
            "check_id": check_id,
            "component": component,
            "status": "PASS" if passed else "BLOCKED",
            "blocker": "" if passed else blocker,
            "evidence_path": str(evidence_path),
            "research_only": True,
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        }
    )


def _safe_existing_path(root: Path, value: str) -> Path | None:
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        return None
    local_path = root / candidate
    if local_path.is_symlink():
        return None
    try:
        resolved = local_path.resolve(strict=True)
        resolved.relative_to(root.resolve())
    except (OSError, ValueError):
        return None
    if not resolved.is_file() or resolved.is_symlink():
        return None
    return resolved


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _write_immutable_json(payload: dict[str, Any], path: Path) -> None:
    try:
        write_immutable_json(path, payload)
    except ValueError as exc:
        raise ValueError(f"immutable receipt collision: {path}") from exc


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    promote_staged_file(temporary, path)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    _atomic_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", path)


def _atomic_text(value: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    promote_staged_file(temporary, path)


def _markdown(summary: dict[str, Any]) -> str:
    return (
        "# Wizard Surface Inventory Readiness\n\n"
        f"- Status: `{summary['status']}`\n"
        f"- API endpoints: `{summary['api_endpoint_count']}`\n"
        f"- Dashboard surfaces: `{summary['dashboard_surface_count']}`\n"
        f"- Recorded integration gaps: `{summary['missing_integration_count']}`\n"
        f"- Capture opportunities: `{summary['capture_opportunity_count']}`\n"
        f"- Checks: `{summary['checks_passed']}/{summary['checks_total']}`\n"
        f"- Integration status: `{summary['integration_status']}`\n"
        "- Promotion authority: `false`\n"
        "- Testnet/live authority: `false`\n\n"
        "Inventory completeness proves that the product surfaces and known gaps are accounted for. "
        "It does not prove formula parity, point-in-time safety, strategy acceptance, "
        "or execution authority.\n"
    )


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _as_utc(value: datetime | None) -> datetime:
    value = value or datetime.now(UTC)
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def main() -> None:
    result = build_wizard_surface_inventory_readiness()
    print(
        json.dumps(
            {
                "summary": result.summary,
                "paths": {key: str(value) for key, value in result.paths.items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
