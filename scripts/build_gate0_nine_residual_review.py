#!/usr/bin/env python3
"""Reproduce the nine residual Gate 0 custody decisions from pinned evidence.

This is an evidence-only builder. It does not mutate the shared source queue, import
preserved desk packages, run schedulers, or call providers.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_COMMIT = "109de903c3631f057f8a9b8712907266e214f95d"
QUEUE = "audit/GATE0_UNION_SOURCE_QUEUE_2026-09-30.csv"
FREEZE = "audit/evidence_freeze_2026-09-29/source_reconciliation.csv"
EXTENDED = "audit/GATE0_EXTENDED_VARIANT_INPUTS_2026-09-30.csv"
STRATEGY_REPORT = "audit/GATE0_STRATEGY_DESKS_SOURCE_RECONCILIATION_2026-09-30.json"
CORRECTIVE_REPORT = "audit/GATE0_CORRECTIVE_TEST_COHORT_2026-09-30.json"
REPORT = ROOT / "audit/GATE0_NINE_RESIDUAL_REVIEW_2026-09-30.json"
PROPOSAL = ROOT / "audit/GATE0_NINE_RESIDUAL_QUEUE_PROPOSAL_2026-09-30.csv"

DESKS = (
    "src/quant_platform/copula_desk/signal_path.py",
    "src/quant_platform/dynamic_desk/signal_path.py",
    "src/quant_platform/static_desk/profile.py",
    "src/quant_platform/static_desk/profile_data.py",
    "src/quant_platform/static_desk/risk_link.py",
    "src/quant_platform/static_desk/runtime.py",
    "src/quant_platform/static_desk/signal_path.py",
)
TESTS = (
    "tests/test_corrective_data_evidence.py",
    "tests/test_phase00_scheduler_supervisor.py",
)
PATHS = DESKS + TESTS
DESK_STATUS = "REVIEWED_DEPENDENCY_BLOCKED_UNADOPTED_DESK_NO_PORT"
DATA_STATUS = "REVIEWED_SOURCE_DEPENDENT_HISTORICAL_TEST_DEFERRED_NO_PORT"
SUPERVISOR_STATUS = (
    "REVIEWED_SOURCE_DEPENDENT_CONFLICTING_HISTORICAL_TEST_DEFERRED_NO_PORT"
)

DESK_REASONS = {
    DESKS[0]: "Offline copula signal has no selected importer or direct component test; its static signal dependency is absent. Adopt only with the complete desk and math-contract cohort.",
    DESKS[1]: "Offline dynamic signal has no selected importer or direct component test; its static signal dependency and four zscore_utils symbols are absent. Adopt only with the complete desk and math-contract cohort.",
    DESKS[2]: "Static profile has no selected importer; institutional_risk and static runtime/accounting/controls dependencies are absent. Research profile custody grants no paper or order authority.",
    DESKS[3]: "Static profile data has no selected importer; its profile and accounting dependencies are unadopted. Research inputs grant no paper or order authority.",
    DESKS[4]: "Static risk link has no selected importer; institutional_risk types are absent. No selected risk binding adopts this package.",
    DESKS[5]: "Static runtime has no selected importer; institutional_risk and static accounting/controls/risk_link/signal_path dependencies are absent. Whole-cohort adoption requires risk and math review.",
    DESKS[6]: "Offline static signal has no selected importer or direct component test; four zscore_utils symbols are absent. Adopt only with the complete desk and math-contract cohort.",
}

TEST_REASONS = {
    TESTS[0]: "Strict L2 cadence coverage was selectively ported. Remaining runtime test drift changes the collection-only candidate model/validator contract and requires terminal cohort evidence plus atomic remediation publication; defer those tests with their producer changes.",
    TESTS[1]: "Missing/false/truthy proof and nonretryability guards were selectively ported. Remaining runtime tests need callback redaction, public-L2 issuer, run fencing, and campaign/request accounting source; two recovery expectations conflict with selected absence-proof handling.",
}

TEST_SEMANTICS = {
    TESTS[0]: {
        "selected_guards_retained": [
            "test_cost_collection_rejects_strict_l2_cadence_holes",
            "test_history_remediation_excludes_intentionally_deferred_and_structural_rows",
        ],
        "runtime_cadence_test_subsumed": "test_cost_collection_rejects_large_interior_gap_despite_sample_count_and_span",
        "source_dependent_gaps": [
            {
                "runtime_test": "test_pair_cost_models_cover_every_registered_candidate_lane",
                "selected_source": "src/quant_platform/orchestration/corrective_data_evidence.py",
                "requirement": "Collection-only candidate must remain in frozen candidate rows while model rows exclude it; selected producer includes all candidates in model_rows and selected validator requires all collection-eligible candidate keys in model rows.",
                "selected_locations": ["build_pair_cost_models:1187-1335", "validate_pair_cost_bundle_artifacts:1804-1841"],
            },
            {
                "runtime_test": "test_history_remediation_missing_cohort_evidence_preserves_active_outputs",
                "selected_source": "src/quant_platform/orchestration/corrective_data_evidence.py",
                "requirement": "Missing terminal cohort evidence must block with active coverage/queue/pointer unchanged; selected build_history_remediation reads history/failure directly and writes coverage/queue without that terminal evidence or pointer contract.",
                "selected_locations": ["build_history_remediation:307-383"],
            },
        ],
    },
    TESTS[1]: {
        "selected_guards_retained": [
            "test_abandoned_intent_without_slot_or_effects_requires_manual_reauthorization",
            "test_abandoned_reservation_without_absence_proof_remains_open",
        ],
        "source_dependent_gaps": [
            {
                "runtime_test": "test_effect_authority_crash_preserves_reason_without_leaking_cause",
                "selected_source": "src/quant_platform/orchestration/corrective_scheduler_supervisor.py",
                "requirement": "Callback blocker must use already imported safe_exception_code; selected callback path uses type(exc).__name__.",
                "selected_locations": ["supervise_scheduler_run:532"],
            },
            {
                "runtime_test": "test_public_l2_supervisor_installs_zero_credit_no_credential_issuer",
                "selected_source": "src/quant_platform/orchestration/corrective_scheduler_supervisor.py",
                "requirement": "Selected hyperliquid_l2 lane uses PHASE00_REPAIR_PROFILE and has no public-L2 zero-credit issuer/profile.",
                "selected_locations": ["supervise_scheduler_run:351"],
            },
            {
                "runtime_test": "test_successful_scheduler_run_is_fenced_before_terminal_accounting",
                "selected_source": "src/quant_platform/orchestration/effect_authority.py;src/quant_platform/orchestration/corrective_scheduler_supervisor.py",
                "requirement": "Run fences before terminal accounting are absent in selected effect authority and supervisor.",
                "selected_locations": [],
            },
            {
                "runtime_test": "test_supervised_daily_run_reconciles_both_provider_lanes_without_orders",
                "selected_source": "src/quant_platform/orchestration/corrective_scheduler_supervisor.py;src/quant_platform/orchestration/corrective_daily_scheduler.py",
                "requirement": "Runtime fixture/admission assertions need absent daily_workload_preflight, active-campaign admission, and request-accounting fields.",
                "selected_locations": [],
            },
        ],
        "conflicting_runtime_recovery_tests": [
            "test_abandoned_intent_without_slot_or_effects_is_recovered_retryable",
            "test_abandoned_reservation_without_provider_effect_is_nonretryable",
        ],
        "conflict": "Selected recovery requires positive provider-effect-absence proof and manual reauthorization; no such proof may be inferred from missing or false fields.",
    },
}


def git_bytes(path: str) -> bytes | None:
    result = subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{BASE_COMMIT}:{path}"],
        capture_output=True,
        check=False,
    )
    return result.stdout if result.returncode == 0 else None


def git_grep(pattern: str, *scopes: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(ROOT), "grep", "-n", "-E", pattern, BASE_COMMIT, "--", *(scopes or ("src", "scripts", "tests"))],
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr)
    return result.stdout


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def csv_rows(data: bytes) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(data.decode("utf-8"))))


def test_functions(data: bytes) -> dict[str, str]:
    source = data.decode("utf-8")
    tree = ast.parse(source)
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
            segment = ast.get_source_segment(source, node)
            if segment is None:
                raise AssertionError(node.name)
            found[node.name] = sha(segment.encode("utf-8"))
    return dict(sorted(found.items()))


def check_copy(path: Path, expected: str) -> None:
    if not path.is_file():
        raise AssertionError(f"missing preserved copy: {path}")
    actual = sha(path.read_bytes())
    if actual != expected:
        raise AssertionError(f"preserved copy hash mismatch: {path}: {actual} != {expected}")


def build() -> tuple[dict, list[dict[str, str]]]:
    pinned = {name: git_bytes(name) for name in (QUEUE, FREEZE, EXTENDED, STRATEGY_REPORT, CORRECTIVE_REPORT)}
    if any(value is None for value in pinned.values()):
        raise AssertionError("pinned evidence input missing from base commit")
    queue = {row["relative_path"]: row for row in csv_rows(pinned[QUEUE])}
    freeze = {row["relative_path"]: row for row in csv_rows(pinned[FREEZE])}
    extended = csv_rows(pinned[EXTENDED])
    prior_strategy = json.loads(pinned[STRATEGY_REPORT])
    prior_strategy_paths = {entry["path"] for entry in prior_strategy["paths"]}
    assert set(DESKS) <= prior_strategy_paths
    assert set(PATHS) <= queue.keys() & freeze.keys()

    # A filename such as test_static_desk_math_v2 does not constitute an import
    # of the unadopted desk package. Demand an empty selected direct-reference scan.
    desk_refs = git_grep(r"quant_platform\.(static_desk|dynamic_desk|copula_desk)|from[[:space:]]+(static_desk|dynamic_desk|copula_desk)")
    if desk_refs:
        raise AssertionError(f"selected direct desk reference appeared:\n{desk_refs}")
    if git_grep(r"static_desk|dynamic_desk|copula_desk", "src/quant_platform"):
        raise AssertionError("selected package source mentions an unadopted desk")
    if git_bytes("src/quant_platform/institutional_risk.py") is not None:
        raise AssertionError("institutional_risk unexpectedly selected")
    for path in DESKS:
        if git_bytes(path) is not None:
            raise AssertionError(f"desk file unexpectedly selected: {path}")

    zscore = git_bytes("src/quant_platform/zscore_utils.py")
    assert zscore is not None
    zscore_names = {node.name for node in ast.parse(zscore).body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    zscore_names |= {target.id for node in ast.parse(zscore).body if isinstance(node, ast.Assign) for target in node.targets if isinstance(target, ast.Name)}
    missing_zscore = [name for name in (
        "ROLLING_SAMPLE_IMPLEMENTATION_VERSION", "EXPANDING_SAMPLE_IMPLEMENTATION_VERSION",
        "rolling_sample_zscore", "expanding_sample_zscore",
    ) if name not in zscore_names]
    if len(missing_zscore) != 4:
        raise AssertionError(f"zscore import closure changed: {missing_zscore}")

    report_rows: list[dict] = []
    proposals: list[dict[str, str]] = []
    for path in PATHS:
        q = queue[path]
        f = freeze[path]
        assert q["working_sha256_at_freeze"] == f["working_sha256"]
        assert q["recovery_sha256_at_freeze"] == f["recovery_sha256"]
        assert q["runtime_sha256_at_freeze"] == f["runtime_sha256"]
        assert q["historical_variant_count"] == "0"
        assert f["recovery_sha256"] == f["runtime_sha256"]
        copies = [row for row in extended if row["relative_path"] == path]
        candidate = [row for row in copies if row["root_label"] == "source_candidate"]
        assert len(candidate) == 1 and candidate[0]["sha256"] == f["runtime_sha256"]
        for copy in copies:
            check_copy(Path(copy["absolute_path"]), copy["sha256"])
        check_copy(Path("/Volumes/Expansion/TheWiz-Workspace/project") / path, f["recovery_sha256"])
        check_copy(Path("/Users/gregc/TheWiz-LocalRuntime") / path, f["runtime_sha256"])

        selected = git_bytes(path)
        if path in DESKS:
            assert selected is None and not f["working_sha256"] and q["working_vs_runtime"] == "RUNTIME_ONLY"
            status = DESK_STATUS
            reason = DESK_REASONS[path]
            semantic = {
                "kind": "offline_signal" if path.endswith("signal_path.py") else "unadopted_static_desk_dependency",
                "active_direct_reference_paths": [],
                "active_package_import_closure_complete": False,
                "missing_selected_dependencies": (
                    ["quant_platform.static_desk.signal_path"] + (missing_zscore if path == DESKS[1] else [])
                    if path in DESKS[:2] else
                    ["quant_platform.institutional_risk", "quant_platform.static_desk.runtime", "quant_platform.static_desk.accounting", "quant_platform.static_desk.controls"] if path == DESKS[2] else
                    ["quant_platform.static_desk.profile", "quant_platform.static_desk.accounting"] if path == DESKS[3] else
                    ["quant_platform.institutional_risk"] if path == DESKS[4] else
                    ["quant_platform.institutional_risk", "quant_platform.static_desk.accounting", "quant_platform.static_desk.controls", "quant_platform.static_desk.risk_link", "quant_platform.static_desk.signal_path"] if path == DESKS[5] else
                    missing_zscore
                ),
                "future_adoption_requires": "coherent dependency, math-contract, direct-test, risk-authority review; no standalone file copy",
            }
        else:
            assert selected is not None and q["working_vs_runtime"] == "CHANGED"
            assert q["custody_status"] == "PORTED_V23_PARTIAL_HISTORICAL_REVIEW_REQUIRED"
            status = DATA_STATUS if path == TESTS[0] else SUPERVISOR_STATUS
            reason = TEST_REASONS[path]
            historical = Path(candidate[0]["absolute_path"]).read_bytes()
            current_tests = test_functions(selected)
            historical_tests = test_functions(historical)
            semantic = {
                "kind": "partial_test_port_with_source_dependent_runtime_drift",
                "selected_test_names": sorted(current_tests),
                "runtime_test_names": sorted(historical_tests),
                "selected_only_test_names": sorted(current_tests.keys() - historical_tests.keys()),
                "runtime_only_test_names": sorted(historical_tests.keys() - current_tests.keys()),
                "changed_shared_test_names": sorted(name for name in current_tests.keys() & historical_tests.keys() if current_tests[name] != historical_tests[name]),
                "selected_test_function_sha256": current_tests,
                "runtime_test_function_sha256": historical_tests,
                **TEST_SEMANTICS[path],
            }
        if "REVIEW_REQUIRED" not in q["custody_status"] or "REVIEW_REQUIRED" in status:
            raise AssertionError(f"unexpected queue transition: {path}")
        record = {
            "relative_path": path,
            "queue_status_at_base": q["custody_status"],
            "proposed_custody_status": status,
            "decision_rationale": reason,
            "working_vs_runtime": q["working_vs_runtime"],
            "historical_variant_count": int(q["historical_variant_count"]),
            "selected_sha256_at_base": sha(selected) if selected else None,
            "four_root_freeze_sha256": {
                "working": f["working_sha256"] or None,
                "recovery": f["recovery_sha256"] or None,
                "runtime": f["runtime_sha256"] or None,
            },
            "extended_copies": [
                {"root_label": copy["root_label"], "sha256": copy["sha256"], "size_bytes": int(copy["size_bytes"]), "in_initial_four_root_freeze": copy["in_initial_four_root_freeze"]}
                for copy in sorted(copies, key=lambda row: row["root_label"])
            ],
            "semantic_review": semantic,
        }
        report_rows.append(record)
        proposals.append({
            "relative_path": path,
            "selected_sha256_at_base": record["selected_sha256_at_base"] or "",
            "working_vs_runtime": q["working_vs_runtime"],
            "queue_current_status_at_base": q["custody_status"],
            "proposed_custody_status": status,
            "proposed_decision_rationale": reason,
            "proposed_decision_evidence": "; ".join(part for part in (
                q["decision_evidence"].strip("; "),
                "audit/GATE0_NINE_RESIDUAL_REVIEW_2026-09-30.json",
                "scripts/build_gate0_nine_residual_review.py",
            ) if part),
        })

    report = {
        "schema_version": "thewiz.gate0.nine_residual_review.v1",
        "base_commit": BASE_COMMIT,
        "scope": "Seven unadopted desk source files and two V23 partial-port historical test files; evidence and queue proposal only.",
        "authority": {"provider_calls": False, "orders_authorized": False, "paper_trading_authorized": False, "selected_source_ported": False, "shared_queue_modified": False},
        "pinned_inputs_sha256": {name: sha(value) for name, value in pinned.items()},
        "external_copy_check": "Every extended-manifest copy and both four-root recovery/runtime copies was SHA-256 checked against pinned metadata during generation and verification.",
        "selected_desk_direct_reference_scan": {
            "scope": ["src", "scripts", "tests"],
            "matches": [],
            "package_absent": True,
            "institutional_risk_module_absent": True,
            "missing_selected_zscore_symbols": missing_zscore,
            "named_test_qualification": "Selected test_static_desk_* files exercise active math/economic/RL helpers, not imports or execution of the absent desk package.",
        },
        "disposition": "Semantic review of all nine paths is complete for current selection; seven desk paths remain unadopted and two historical test contracts remain deferred. No release or trading authority is inferred.",
        "paths": report_rows,
    }
    return report, proposals


def encoded_outputs() -> tuple[bytes, bytes]:
    report, proposals = build()
    report_bytes = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    columns = list(proposals[0])
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(proposals)
    return report_bytes, buffer.getvalue().encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="verify versioned outputs and preserved-copy SHA-256 values")
    args = parser.parse_args()
    report_bytes, proposal_bytes = encoded_outputs()
    if args.verify:
        for path, expected in ((REPORT, report_bytes), (PROPOSAL, proposal_bytes)):
            if not path.is_file() or path.read_bytes() != expected:
                raise SystemExit(f"verification mismatch: {path}")
        print(f"verified {len(PATHS)} rows, preserved copies, and pinned base {BASE_COMMIT}")
    else:
        REPORT.write_bytes(report_bytes)
        PROPOSAL.write_bytes(proposal_bytes)
        print(f"wrote {REPORT.relative_to(ROOT)} and {PROPOSAL.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
