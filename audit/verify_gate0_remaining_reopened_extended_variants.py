#!/usr/bin/env python3
"""Reconcile the 14 remaining reopened Gate 0 extended variants.

This verifier only reads selected source and frozen local evidence. The decision
snapshot is pinned to 99df4d1; current queue statuses may advance independently.
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
BASE = "99df4d1f3b3f34a6c6c50c2d111a2344c05c6a61"
GAP = "audit/GATE0_EXTENDED_VARIANT_GAP_2026-09-30.json"
MANIFEST = "audit/GATE0_EXTENDED_VARIANT_INPUTS_2026-09-30.csv"
FOUR_ROOT = "audit/evidence_freeze_2026-09-29/source_reconciliation.csv"
QUEUE = "audit/GATE0_UNION_SOURCE_QUEUE_2026-09-30.csv"
REPORT = ROOT / "audit/GATE0_REMAINING_REOPENED_EXTENDED_VARIANTS_2026-09-30.json"
INPUT_SHA256 = {
    GAP: "843abe2a6f4c2234ee1acad4d78c53e13e4ebbc9ef375a402dc6ef1daf136383",
    MANIFEST: "ca055e5cfbd6c5cdcd1e42b712b394aac2701bcf01a173fef151248e9eb046b3",
    FOUR_ROOT: "cecdd2603471da26db24b826d0a6aafe1ab11b36823f014c1abfe02ec2406943",
    QUEUE: "6e0beb2f87635f5e7af344519a333ef06b9da8c5ace6175e911380b79791b9b5",
}
# new: newly unlisted SHA; prior: previously reviewed forensic SHA;
# baseline: initial four-root source_candidate or hash-only working copy.
FINDINGS = {
    "pyproject.toml": {
        "new": "The paired 157-package Math Desk/collection candidate adds optional math-research and collection-dev extras and changes pytest authority markers; that source set was not selected or qualified.",
        "prior": "The older manifest permits SciPy 1.18 on Python 3.12 and omits the active pinned ccxt/dYdX dev dependencies; forensic_b only adds a package-discovery exclude to the same older contract.",
        "baseline": "No four-root file hash was recorded; the source_candidate copy is the separately reviewed runtime manifest.",
        "prior_report": "audit/GATE0_DEPENDENCY_LOCK_DECISION_2026-09-30.json",
        "selected_anchor": "without_public_network_authority:",
        "new_anchor": "manual_external_effect_issuer:",
    },
    "src/quant_platform/dydx_candles.py": {
        "new": "The visual copy maps calendar 1M to 1m, allows provisional features to overwrite native values, backfills future funding into earlier candles, uses price-level pair economics, and writes evidence directly.",
        "prior": "The forensic_b copy retains price-level X-on-Y spread and lacks selected Y-on-X log economics, namespaced proxy backfill, and governed output; its leading funding gaps stay unknown but do not restore the selected contract.",
        "baseline": "The four-root recovery/runtime source_candidate has price-level hedge-ratio tests and predates selected Y-on-X log, monthly interval, native-feature, funding-chronology and atomic-output repairs; working is an earlier hash-only snapshot.",
        "prior_report": "audit/GATE0_DYDX_ML_SOURCE_RECONCILIATION_2026-09-30.json",
        "selected_anchor": '"hedge_ratio_orientation": "beta_y_on_x"',
        "new_anchor": "_attach_provisional_research_features(rows, overwrite=True)",
    },
    "src/quant_platform/experiments.py": {
        "new": "The visual copy accepts string truthiness, treats merely present nonfinite two-leg columns as complete, double-counts trades across repriced cost buckets, drops independent eligibility verification, and removes flat-end/reconciliation/uncertainty blockers.",
        "prior": "The forensic_b copy has strict boolean and independent cost-bucket checks, but predates selected V2.3 acceptance-row verification, completed numeric-input checks, atomic reporting and current acceptance lineage.",
        "baseline": "The four-root recovery/runtime source_candidate is the pre-V2.3 experiment implementation; selected source later adds costed causal acceptance and no-proxy-authority guards; working is an earlier hash-only snapshot.",
        "prior_report": "audit/GATE0_V23_SOURCE_TRANSITION_2026-09-30.json",
        "selected_anchor": "def _verified_acceptance_row(",
        "new_anchor": 'fillna(False).astype(bool).all(axis=1)',
    },
    "src/quant_platform/orchestration/student_readiness.py": {
        "new": "The visual/forensic copy omits duplicate and required-column, explicit timestamp, finite label, provenance, logged bandit-policy, current math-version and atomic-report checks; ambiguous rows can appear ready.",
        "prior": "The same SHA is both visual_recovery and forensic_b.",
        "baseline": "The four-root recovery/runtime source_candidate supplied strict student dataset checks but predated current math-version and governed report refinements; working is an earlier hash-only snapshot.",
        "prior_report": "audit/GATE0_STUDENT_TEACHER_SOURCE_RECONCILIATION_2026-09-30.json; audit/GATE0_TEACHER_NUMERIC_GUARDS_2026-09-30.json",
        "selected_anchor": "duplicate_training_event_id",
        "new_anchor": 'label_count = int(pd.to_numeric(working["good_trade"], errors="coerce").dropna().nunique())',
    },
    "src/quant_platform/orchestration/teacher_adapters.py": {
        "new": "The visual/forensic copy uses an inline weaker math marker check, hardcoded math-v2, raw exception text and direct publication; selected source checks bound/current math evidence and writes atomically.",
        "prior": "The same SHA is both visual_recovery and forensic_b.",
        "baseline": "The four-root recovery/runtime source_candidate is an older adapter before shared currentness, safe redaction and atomic output; working is an earlier hash-only snapshot.",
        "prior_report": "audit/GATE0_MATH_MARKER_CURRENTNESS_2026-09-30.json",
        "selected_anchor": "math_acceptance_marker_passes",
        "new_anchor": "readiness_frame.to_csv(readiness_path, index=False)",
    },
    "src/quant_platform/orchestration/teacher_contracts.py": {
        "new": "The visual/forensic copy hardcodes math-v2 and permits NaN/Infinity in teacher and student contract numbers, so invalid return evidence can enter council voting.",
        "prior": "The same SHA is both visual_recovery and forensic_b.",
        "baseline": "The four-root recovery/runtime source_candidate has the contract shapes but lacks selected finite-number validation and current shared math version; working is an earlier hash-only snapshot.",
        "prior_report": "audit/GATE0_TEACHER_NUMERIC_GUARDS_2026-09-30.json",
        "selected_anchor": "allow_inf_nan=False",
        "new_anchor": 'MATH_V2 = "math-v2"',
    },
    "src/quant_platform/orchestration/teacher_control_plane.py": {
        "new": "The visual copy uses inline weaker math marker checks, raw exception text, hardcoded math-v2 and direct CSV/JSONL publication in place of shared currentness, redaction and atomic writes.",
        "prior": "The forensic_b copy has the same publication/currentness gaps; its _boolish handling of persisted blocking_for_shadow is a possible separate narrow guard, while the selected readiness frame is already boolean.",
        "baseline": "The four-root recovery/runtime source_candidate predates shared bound marker currentness, safe exception codes and atomic control-plane publication; working is an earlier hash-only snapshot.",
        "prior_report": "audit/GATE0_MATH_MARKER_CURRENTNESS_2026-09-30.json",
        "selected_anchor": "math_acceptance_marker_passes",
        "new_anchor": 'readiness.to_csv(readiness_path, index=False)',
    },
    "src/quant_platform/strategies.py": {
        "new": "The visual copy drops proxy implementation/acceptance metadata, converts exit strings with astype(bool), uses current/future distribution statistics for regime decisions and inserts favorable defaults for missing signals.",
        "prior": "The forensic_b copy keeps strict exits, prior-only expanding quantiles and proxy metadata, but predates selected V2.3 canonical strategy economics and acceptance qualification.",
        "baseline": "The four-root recovery/runtime source_candidate is a pre-V2.3 strategy catalog; selected source adds causal thresholds and explicit proxy authority; working is an earlier hash-only snapshot.",
        "prior_report": "audit/GATE0_V23_SOURCE_TRANSITION_2026-09-30.json",
        "selected_anchor": "def _prior_expanding_quantile(",
        "new_anchor": "vol.rolling(50, min_periods=2).quantile(0.60).fillna(vol.median())",
    },
    "tests/test_dydx_candles.py": {
        "new": "The visual test copy omits six selected regressions for monthly identity, funding chronology and native-feature preservation, and asserts that provisional values occupy canonical feature names.",
        "prior": "The forensic_b test copy retains a pre-observation funding check but lacks selected monthly, unsorted-funding and native-feature regressions; obsolete price-level assertions were replaced by log Y-on-X tests.",
        "baseline": "The four-root recovery/runtime source_candidate retains ten price-level hedge-ratio tests that do not match selected log Y-on-X economics and lacks six later chronology/feature regressions; working is an earlier hash-only snapshot.",
        "prior_report": "audit/GATE0_DYDX_ML_SOURCE_RECONCILIATION_2026-09-30.json",
        "selected_anchor": "def test_funding_merge_does_not_leak_across_out_of_order_candles(",
        "new_anchor": 'assert "conditional_probability_distortion" in row',
    },
    "tests/test_experiments.py": {
        "new": "The visual copy lacks five selected regressions for incomplete two-leg inputs, nonfinite/open-trade gating, claimed eligibility, independent repriced trades and string boolean flags.",
        "prior": "The five forensic_b-only tests map to current experiment/historical-strategy-authority successors; they do not warrant restoring its older file.",
        "baseline": "The four-root source_candidate has 11 tests and lacks current acceptance guards; the initial working/recovery/runtime hashes are identical and prior migration is recorded.",
        "prior_report": "audit/GATE0_STUDENT_TEACHER_SOURCE_RECONCILIATION_2026-09-30.json",
        "selected_anchor": "def test_strategy_acceptance_rejects_string_boolean_execution_flags(",
        "new_anchor": "def test_harness_runs_executable_strategies_and_marks_missing_ones_skipped(",
    },
    "tests/test_student_readiness.py": {
        "new": "The visual/forensic copy omits five selected schema, identity, provenance, numeric timestamp and finite-label regressions and hardcodes math-v2 test data.",
        "prior": "The same SHA is both visual_recovery and forensic_b.",
        "baseline": "The four-root recovery/runtime test source has the original five readiness cases; selected tests add five fail-closed regressions; working is an earlier hash-only snapshot.",
        "prior_report": "audit/GATE0_STUDENT_TEACHER_SOURCE_RECONCILIATION_2026-09-30.json",
        "selected_anchor": "def test_duplicate_columns_are_rejected_before_row_audit(",
        "new_anchor": '"math_version": "math-v2"',
    },
    "tests/test_teacher_adapters.py": {
        "new": "The visual/forensic copy hardcodes math-v2 in its fixture, so it does not track current shared math lineage; selected fixtures derive MATH_VERSION from the active math module.",
        "prior": "The same SHA is both visual_recovery and forensic_b.",
        "baseline": "The four-root working/source_candidate test has the same two behavioral cases; the separate runtime hash predates the selected global scoped publication fixture and current-math fixture.",
        "prior_report": "audit/GATE0_STUDENT_TEACHER_SOURCE_RECONCILIATION_2026-09-30.json",
        "selected_anchor": "from quant_platform.statistics.math_v2 import MATH_VERSION",
        "new_anchor": '"math_version": "math-v2"',
    },
    "tests/test_teacher_council.py": {
        "new": "The visual/forensic copy omits six selected regressions for prior/stale math, nonfinite returns, invalid policy thresholds, duplicate teacher IDs and overflow-safe weighted averages.",
        "prior": "The same SHA is both visual_recovery and forensic_b.",
        "baseline": "The four-root recovery/runtime test has five cases including prior-math rejection but lacks the five later numeric/identity tests; working is an earlier hash-only snapshot.",
        "prior_report": "audit/GATE0_STUDENT_TEACHER_SOURCE_RECONCILIATION_2026-09-30.json; audit/GATE0_TEACHER_NUMERIC_GUARDS_2026-09-30.json",
        "selected_anchor": "def test_teacher_proposal_rejects_nonfinite_return_evidence(",
        "new_anchor": 'math_version: str = "math-v2"',
    },
    "uv.lock": {
        "new": "The 157-package source_candidate lock adds 17 unselected Math Desk/collection packages, including arch, cvxpy, pyvinecopulib and matplotlib, paired with an unselected pyproject manifest.",
        "prior": "The 129-package older forensic lock admits SciPy 1.18 under Python 3.12 and omits active ccxt, aiohttp and related dependencies.",
        "baseline": "No four-root file hash was recorded; the source_candidate copy is the separately reviewed runtime lock.",
        "prior_report": "audit/GATE0_DEPENDENCY_LOCK_DECISION_2026-09-30.json",
        "selected_anchor": "",
        "new_anchor": "",
    },
}
TEST_NEW_ONLY_EXPECTED = {
    "tests/test_dydx_candles.py": 6,
    "tests/test_experiments.py": 5,
    "tests/test_student_readiness.py": 5,
    "tests/test_teacher_adapters.py": 0,
    "tests/test_teacher_council.py": 6,
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def snapshot(relative: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{BASE}:{relative}"],
        cwd=ROOT, check=True, capture_output=True,
    ).stdout


def rows(data: bytes) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(data.decode("utf-8"))))


def test_names(data: bytes) -> set[str]:
    return {
        node.name for node in ast.parse(data.decode("utf-8")).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    }


def build() -> dict[str, object]:
    subprocess.run(["git", "merge-base", "--is-ancestor", BASE, "HEAD"], cwd=ROOT, check=True)
    inputs = {}
    for path, expected in INPUT_SHA256.items():
        data = snapshot(path)
        if sha(data) != expected:
            raise ValueError(f"base input hash drift: {path}")
        if path != QUEUE and (ROOT / path).read_bytes() != data:
            raise ValueError(f"frozen input changed: {path}")
        inputs[path] = data
    gap = {row["relative_path"]: row for row in json.loads(inputs[GAP])["rows"]}
    manifest = rows(inputs[MANIFEST])
    baseline = {row["relative_path"]: row for row in rows(inputs[FOUR_ROOT])}
    queue = {row["relative_path"]: row for row in rows(inputs[QUEUE])}
    current_queue = {row["relative_path"]: row for row in rows((ROOT / QUEUE).read_bytes())}
    reviews = []
    for path, finding in sorted(FINDINGS.items()):
        source = snapshot(path)
        if (ROOT / path).read_bytes() != source:
            raise ValueError(f"selected source changed: {path}")
        selected_sha = sha(source)
        g = gap[path]
        if len(g["unlisted_variants"]) != 1:
            raise ValueError(f"new variant cardinality changed: {path}")
        new_sha = g["unlisted_variants"][0]["sha256"]
        q = queue[path]
        active_q = current_queue[path]
        historical = set(filter(None, q["historical_variant_sha256"].split(";")))
        if (q["custody_status"] != "PRESERVED_ADDITIONAL_VARIANT_REVIEW_REQUIRED"
                or len(historical) != int(q["historical_variant_count"])
                or new_sha not in historical
                or g["prior_custody_status"] not in {
                    "REVIEWED_ACTIVE_LOCK_RETAIN_HISTORICAL_NO_PORT",
                    "REVIEWED_HISTORICAL_VARIANTS_RECONCILED",
                    "REVIEWED_PORTED_V23_SOURCE",
                    "REVIEWED_STUDENT_TEACHER_HISTORICAL_VARIANTS",
                    "REVIEWED_MATH_MARKER_CURRENTNESS",
                    "REVIEWED_TEACHER_NUMERIC_GUARDS",
                }):
            raise ValueError(f"queue metadata mismatch: {path}")
        if not historical.issubset(set(filter(None, active_q["historical_variant_sha256"].split(";")))):
            raise ValueError(f"current queue lost variant metadata: {path}")
        b = baseline[path] if path in baseline else {
            "working_sha256": "", "recovery_sha256": "", "runtime_sha256": "", "decision": "NOT_IN_FOUR_ROOT_TABLE",
        }
        for key in ("working_sha256", "recovery_sha256", "runtime_sha256"):
            if b[key] != g["four_root_baseline_sha256"][key]:
                raise ValueError(f"four-root baseline mismatch: {path} {key}")
        frozen_rows = [
            row for row in manifest
            if row["relative_path"] == path and row["status"] == "hashed"
        ]
        indexed: dict[str, list[dict[str, str]]] = {}
        frozen_data: dict[str, bytes] = {}
        for entry in frozen_rows:
            copy = Path(entry["absolute_path"])
            if not copy.is_file() or copy.is_symlink():
                raise ValueError(f"missing/symlink frozen copy: {copy}")
            data = copy.read_bytes()
            if sha(data) != entry["sha256"]:
                raise ValueError(f"frozen copy drift: {copy}")
            variant_sha = entry["sha256"]
            if variant_sha in frozen_data and frozen_data[variant_sha] != data:
                raise ValueError(f"SHA collision: {path}")
            frozen_data[variant_sha] = data
            indexed.setdefault(variant_sha, []).append({
                "root": entry["root_label"],
                "absolute_path": entry["absolute_path"],
                "sha256": variant_sha,
            })
        if new_sha not in indexed or not historical.issubset(indexed):
            raise ValueError(f"manifest lacks historical SHA: {path}")
        new_copies = {
            (copy["root"], copy["absolute_path"], copy["frozen_sha256"], copy["current_sha256"])
            for copy in g["unlisted_variants"][0]["copies"]
        }
        expected_new_copies = {
            (copy["root"], copy["absolute_path"], new_sha, new_sha)
            for copy in indexed[new_sha]
        }
        if new_copies != expected_new_copies:
            raise ValueError(f"gap/manifest copy mismatch: {path}")
        selected_text = source.decode("utf-8") if path != "uv.lock" else ""
        new_text = frozen_data[new_sha].decode("utf-8") if path != "uv.lock" else ""
        if finding["selected_anchor"]:
            if finding["selected_anchor"] not in selected_text:
                raise ValueError(f"selected semantic anchor absent: {path}")
            if path not in TEST_NEW_ONLY_EXPECTED and finding["selected_anchor"] in new_text:
                raise ValueError(f"selected guard exists in new variant: {path}")
        if finding["new_anchor"]:
            if finding["new_anchor"] not in new_text:
                raise ValueError(f"new semantic anchor absent: {path}")
            if path not in TEST_NEW_ONLY_EXPECTED and finding["new_anchor"] in selected_text:
                raise ValueError(f"new regression exists in selected: {path}")
        baseline_shas = {b[key] for key in ("working_sha256", "recovery_sha256", "runtime_sha256") if b[key]}
        all_variant_shas = (set(indexed) | baseline_shas) - {selected_sha}
        variant_records = []
        for variant_sha in sorted(all_variant_shas):
            copies = indexed.get(variant_sha, [])
            if variant_sha == new_sha:
                kind, assessment = "newly_discovered", finding["new"]
            elif variant_sha in historical:
                kind, assessment = "previously_reviewed_historical", finding["prior"]
            else:
                kind, assessment = "four_root_baseline", finding["baseline"]
            if variant_sha in historical and not {
                copy["root"] for copy in copies
            }.issubset(set(filter(None, active_q["historical_copy_roots"].split(";")))):
                raise ValueError(f"current queue lost root metadata: {path}")
            variant_records.append({
                "sha256": variant_sha,
                "classification": kind,
                "recorded_in_queue_historical_sha_field": variant_sha in historical,
                "four_root_roles": [
                    key.removesuffix("_sha256") for key in
                    ("working_sha256", "recovery_sha256", "runtime_sha256")
                    if b[key] == variant_sha
                ],
                "copies": copies,
                "byte_reverified": bool(copies),
                "semantic_difference_from_selected": assessment,
                "decision": "NO_WHOLE_FILE_PORT",
            })
        test_delta = None
        if path in TEST_NEW_ONLY_EXPECTED:
            selected_tests = test_names(source)
            new_tests = test_names(frozen_data[new_sha])
            missing = sorted(selected_tests - new_tests)
            if len(missing) != TEST_NEW_ONLY_EXPECTED[path]:
                raise ValueError(f"test delta changed: {path}")
            test_delta = {
                "selected_test_functions": len(selected_tests),
                "new_variant_test_functions": len(new_tests),
                "selected_tests_missing_from_new_variant": missing,
                "new_variant_tests_absent_from_selected": sorted(new_tests - selected_tests),
            }
        prior_reports = {
            evidence: sha(snapshot(evidence))
            for evidence in finding["prior_report"].split("; ")
        }
        for evidence, expected in prior_reports.items():
            if sha((ROOT / evidence).read_bytes()) != expected:
                raise ValueError(f"prior decision snapshot drift: {evidence}")
        rationale = (
            "Additional frozen variant reviewed: " + finding["new"] +
            " Preserve all SHA/root custody and retain selected source; no whole-file port."
        )
        if active_q["custody_status"] != g["prior_custody_status"]:
            raise ValueError(f"current queue decision not closed: {path}")
        if rationale not in active_q["decision_rationale"]:
            raise ValueError(f"current queue missing variant rationale: {path}")
        if REPORT.relative_to(ROOT).as_posix() not in active_q["decision_evidence"].split("; "):
            raise ValueError(f"current queue missing report citation: {path}")
        reviews.append({
            "relative_path": path,
            "selected_active_at_base_sha256": selected_sha,
            "current_selected_sha256": selected_sha,
            "frozen_four_root_sha256": {
                key: b[key] for key in ("working_sha256", "recovery_sha256", "runtime_sha256")
            },
            "frozen_four_root_decision": b["decision"],
            "prior_custody_status_snapshot": g["prior_custody_status"],
            "queue_status_at_base": q["custody_status"],
            "recommended_queue_status": g["prior_custody_status"],
            "recommended_decision_rationale_append": rationale,
            "recommended_decision_evidence_append": REPORT.relative_to(ROOT).as_posix(),
            "prior_decision_evidence_sha256": prior_reports,
            "queue_historical_variant_sha256_at_base": sorted(historical),
            "semantic_anchor_selected": finding["selected_anchor"],
            "semantic_anchor_new_variant": finding["new_anchor"],
            "source_decision": "RETAIN_SELECTED_ACTIVE_NO_PORT",
            "test_function_comparison": test_delta,
            "variants": variant_records,
        })
    return {
        "schema_version": "thewiz.gate0.remaining_reopened_extended_variants.v1",
        "base_commit": BASE,
        "inputs_sha256": INPUT_SHA256,
        "reviewed_paths": len(reviews),
        "newly_discovered_variant_shas": 14,
        "whole_file_ports": 0,
        "source_repair_commits": 0,
        "external_effects_or_orders_executed": False,
        "queue_edited": False,
        "baseline_hash_only_note": "A four-root SHA without a manifest copy is preserved as a hash-only baseline; the earlier decision report supplies its semantic review.",
        "rows": reviews,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    result = build()
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.write_report:
        REPORT.write_text(encoded, encoding="utf-8")
    elif REPORT.read_text(encoding="utf-8") != encoded:
        raise ValueError("committed report differs from frozen evidence")
    print(
        f"PASS remaining reopened: {result['reviewed_paths']} paths, "
        f"{sum(len(row['variants']) for row in result['rows'])} distinct variant records, "
        f"{result['newly_discovered_variant_shas']} newly discovered SHAs"
    )


if __name__ == "__main__":
    main()
