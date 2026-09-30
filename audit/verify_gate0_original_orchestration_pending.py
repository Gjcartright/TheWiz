#!/usr/bin/env python3
"""Verify the original-811 pending orchestration cohort against frozen copies.

The review is a historical decision at BASE. It does not execute provider code.
The separately proposed boolean repair may be cherry-picked after this report.
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
BASE = "c906d5769044e83951a6e09ca06b09303e610fa3"
ORIGINAL = "99df4d1f3b3f34a6c6c50c2d111a2344c05c6a61"
QUEUE = "audit/GATE0_UNION_SOURCE_QUEUE_2026-09-30.csv"
MANIFEST = "audit/GATE0_EXTENDED_VARIANT_INPUTS_2026-09-30.csv"
FOUR_ROOT = "audit/evidence_freeze_2026-09-29/source_reconciliation.csv"
MISSING_IMPORTS = "audit/evidence_freeze_2026-09-29/missing_import_edges.csv"
REPORT = ROOT / "audit/GATE0_ORIGINAL_ORCHESTRATION_PENDING_RECONCILIATION_2026-09-30.json"
INPUT_SHA256 = {
    QUEUE: "31a65a6dc7633d802e0e4bba1a7c484300fdadeb1062802b8f8f1aa51c3782bd",
    MANIFEST: "ca055e5cfbd6c5cdcd1e42b712b394aac2701bcf01a173fef151248e9eb046b3",
    FOUR_ROOT: "cecdd2603471da26db24b826d0a6aafe1ab11b36823f014c1abfe02ec2406943",
    MISSING_IMPORTS: "b0639a70bc87a15c219ffddb858139e2f54f962116d5a6b3c3a7fb8909755f82",
}
RECOVERY_ROOT = Path("/Volumes/Expansion/TheWiz-Workspace/project")
RUNTIME_ROOT = Path("/Users/gregc/TheWiz-LocalRuntime")
PREFIX = "src/quant_platform/orchestration/"
ACTIVE_STATUS = "REVIEWED_ORIGINAL_ORCHESTRATION_ACTIVE_RETAIN_NO_PORT"
RUNTIME_STATUS = "REVIEWED_ORIGINAL_ORCHESTRATION_RUNTIME_DEPENDENCY_NO_PORT"
BOOL_STATUS = "REVIEWED_SELECTIVE_PERSISTED_BOOL_PORT_REMAINDER_NO_PORT"
# Each entry is a path-specific semantic finding. Variant records below bind
# this finding to every distinct frozen SHA and physical copy.
FINDINGS = {
    "__init__.py": "The runtime initializer lazily exports institutional, operating-lane and personal-trader graph APIs absent from selected source. Selected imports remain coherent; adding exports before those modules would create unresolved API surfaces.",
    "canonical_wizard_hyperliquid_contract.py": "Selected canonical policy JSON, stage CSV and Markdown use atomic publication. Visual/outer copies directly write all three; forensic_b additionally adds an unqualified exact-mode regeneration queue and strict flag parsing but keeps direct writes.",
    "current_wizard_collection_display.py": "Runtime-only Stage 19 collection display prepares 21 exact outputs and imports the absent precollection campaign IO stack. No selected caller or collection owner/release path adopts this publisher.",
    "daily_snapshot_contracts.py": "Runtime-only daily snapshot validator composes the two absent input/outcome contract modules and describes producer checks rather than independent scientific acceptance. It is not an adopted daily release gate.",
    "daily_snapshot_contracts_inputs.py": "Runtime-only input-stage snapshot specifications depend on the absent daily contract caller and producer paths; no selected source imports this module.",
    "daily_snapshot_contracts_outcomes.py": "Runtime-only final-seven-stage output/chain checks depend on the absent daily contract caller and retained campaign artifacts; no selected release path consumes them.",
    "daily_workload_preflight.py": "Runtime-only request projection depends on absent funding allocation and precollection campaign IO. Its bounded history/funding plan does not authorize acquisition independently of the unselected collection stack.",
    "discovery_accounting.py": "Runtime-only pure request accounting has no selected caller. The recovery copy caps per-provider credits at 2100, while the later LocalRuntime copy raises that threshold to 7000; no budget expansion is selected.",
    "dynamic_stage_runner.py": "Selected hyperliquid_research_cycle stage interprets execution_allowed with bool(), so a string 'False' can mark the stage passed. The forensic_b strict_bool change is a narrow safety-positive candidate; preserve all other selected stage routing.",
    "effect_authority.py": "Selected v2 permit/journal is used by active effect callers. The runtime v6 candidate changes journal/key/seal paths, reservation signatures, root binding and directory publication contracts; existing callers are incompatible without a coordinated migration.",
    "elapsed_capture_job.py": "Runtime-only daily BTC/ETH capture performs two direct unauthenticated Hyperliquid HTTP POSTs via urllib and writes retained evidence. It has no selected effect permit or active caller, so importing it would introduce an ungoverned network path.",
    "elapsed_capture_store.py": "Runtime-only append-only raw/receipt store is coupled to elapsed_causal_capture and elapsed_capture_job, neither selected. Its local chain alone grants no collection authority.",
    "elapsed_causal_capture.py": "Runtime-only pure forward-time bar and receipt validator has no selected caller; it is a component of the unselected elapsed network capture lane.",
    "exhaustive_wizard_api_refresh.py": "Selected refresh uses atomic active/snapshot publication. Visual/outer copies write directly; forensic_b adds strict persisted-flag parsing but still writes directly. The runtime source_candidate adds a 21-output prepared Stage 3 and lease checks while importing absent daily workload/precollection modules; it needs the full collection stack.",
    "exhaustive_wizard_hyperliquid_canonical_replay.py": "Selected v2 canonical replay retains tail-action economics, safe exception codes and atomic reports. Older v1 copies directly publish; forensic_b parses research-rank flags more strictly but retains the old publication path.",
    "exhaustive_wizard_hyperliquid_cost_bridge.py": "Selected cost bridge atomically publishes cost evidence/manifest and immutable snapshots. Older copy uses direct CSV/text writes and shutil.copy2, allowing partial or mutable snapshot evidence.",
    "exhaustive_wizard_hyperliquid_cost_evidence.py": "Selected cost capture/redaction and atomic queue/checkpoint publication are absent from older copies. Forensic_b adds strict metadata flags and a paced funding interval, but retains direct writes and unsafe error handling.",
    "exhaustive_wizard_hyperliquid_leverage.py": "Selected leverage evidence uses atomic outputs and immutable snapshots; older copy directly writes CSV/Markdown and shutil.copy2 copies.",
    "exhaustive_wizard_hyperliquid_observed_cost_replay.py": "Selected observed-cost replay uses safe exception codes, atomic outputs and immutable snapshots; older copy directly writes and copies evidence and exposes raw exception text.",
    "exhaustive_wizard_hyperliquid_regimes.py": "Selected regime reports use safe redaction, atomic publication and immutable snapshots; older copy directly writes/copies artifacts and embeds raw exception details.",
    "exhaustive_wizard_hyperliquid_replay.py": "Selected pair-history replay uses safe exception codes and atomic bytes/CSV/text publication. Older copies directly write reports and include raw errors; forensic_b only tightens mapping_ready flag parsing.",
    "exhaustive_wizard_hyperliquid_robustness.py": "Selected robustness replay retains Y-on-X log spread, safe redaction and atomic/immutable reports. Older copy drops those guards and uses direct writes/copies.",
    "exhaustive_wizard_hyperliquid_run.py": "Selected v4 frozen run uses canonical mode contract and atomic outputs. Older v3 copies directly publish source/experiment artifacts; forensic_b only tightens persisted mapping and authority flags.",
    "exhaustive_wizard_hyperliquid_walkforward.py": "Selected walkforward uses Y-on-X economics, safe redaction and atomic/immutable reports. Forensic_b adds a research trial ledger and declared-family multiple-test count, a separate scientific idea, while retaining direct writes and older economics.",
    "funding_request_allocation.py": "Runtime-only pure hash-bound funding page allocation has no selected caller; the daily workload owner and collection route are absent.",
    "human_evidence_dashboard.py": "Runtime-only 2182-line dashboard combines Gate 1-3 and research views, imports absent cadence/history/view modules, and writes status artifacts. No selected caller or qualified dashboard contract adopts it.",
    "human_evidence_research.py": "Runtime-only retained-evidence observation is explicitly nongating, with no selected consumer except the absent dashboard; do not treat its review cells as acceptance.",
    "human_evidence_view.py": "Runtime-only pure HTML/presentation helpers have no selected consumer except the absent dashboard and grant no release status.",
    "hyperliquid_l2_calibration_intake.py": "Runtime-only intake promotes retained L2 calibration into source-qualification input; the selected source has no adopted L2 calibration lifecycle or caller for this promotion.",
    "hyperliquid_research_validation.py": "Selected validation v3 uses current Y-on-X log spread, positive hedge-ratio gate and staged/atomic publication. Older v2 copies hardcode math-v2 and direct-write; forensic_b additionally reverses Engle-Granger orientation.",
    "institutional_contracts.py": "Runtime-only institutional authority manifest and desk receipt contracts have no selected graph/caller. Their control evidence cannot be made operative by a standalone model import.",
    "institutional_fund_graph.py": "Runtime-only institutional LangGraph consumes the absent institutional contracts, evaluates desk submissions and writes reports. It is not connected to selected release or business authority.",
    "interactive_mixtape_graph.py": "Selected graph report writes Markdown and JSON atomically. Identical visual/forensic copies directly write both artifacts, risking partial operator evidence.",
    "multi_asset_capture_job.py": "Selected multi-asset Hyperliquid collector binds each public info effect to authorization, publication lease and exclusive immutable output. Runtime copy calls urllib directly and omits those source/receipt guards.",
    "multi_asset_capture_verify.py": "Selected verifier rejects symlink entries, incomplete predecessor sets, invalid timestamps/assets and inconsistent terminal receipts. Runtime copy omits those checks, so retained capture authority could be overstated.",
    "operating_lane.py": "Runtime-only router switches self-directed and institutional graphs, both absent from selected source, with no selected lane admission or caller.",
    "personal_trader_graph.py": "Runtime-only personal evidence graph and self-directed declaration are unadopted; no selected caller or approved declaration/authority envelope binds them.",
    "publication_target_inventory.py": "Runtime-only exact directory layout/quarantine contract supports the unselected v6 directory-publication stack and grants no authority by itself.",
    "scheduler_terminal_paths.py": "Runtime-only terminal-path templates have no selected writer/reader; importing them alone would not establish a release or scheduler contract.",
    "venue_gate.py": "Selected venue gate interprets persisted dYdX and Hyperliquid compatibility flags with astype(bool)/bool, so string 'False' can confirm a route. Forensic_b strictly parses both flags and is a narrow safety-positive candidate.",
    "versioned_learning_comparison.py": "Selected versioned-learning evidence uses atomic CSV/JSON/Markdown publication. Older copies directly write; forensic_b strictly parses persisted authority flags but keeps the publication regression.",
    "wizard_hyperliquid_frozen_adapter.py": "Runtime-only pure retained Hyperliquid /info adapter feeds the unselected frozen-source qualification lane; no selected consumer or source admission adopts it.",
    "wizard_hyperliquid_l2_calibration_runner.py": "Runtime-only bounded one-shot L2 calibration planner/runner has no selected collector, callback authorization or qualification consumer.",
    "wizard_keychain.py": "Runtime-only Mac Keychain adapter invokes permit-bound credential reader but has no selected caller or adopted credential route; activating it would add a credential access path.",
    "wizard_source_route_config.py": "Runtime-only nonexecuting Pydantic route schema fixes authority flags false and allowlists provider operations, but no selected route/probe reads it.",
}
BOOLEAN_GUARDS = {
    "dynamic_stage_runner.py": (
        'ready = bool(result.get("execution_allowed", False))',
        'ready = strict_bool(result.get("execution_allowed", False))',
    ),
    "venue_gate.py": (
        'frame.get("compatible_for_paper_submit", False).astype(bool)',
        ').map(strict_bool)',
    ),
}
EXTRA_OLDER_FINDINGS = {
    "canonical_wizard_hyperliquid_contract.py": "forensic_b adds a regeneration queue and strict parsing but still directly publishes it.",
    "exhaustive_wizard_api_refresh.py": "forensic_b uses _truthy on persisted flags but retains direct active/snapshot writes.",
    "exhaustive_wizard_hyperliquid_canonical_replay.py": "forensic_b uses _truthy on research-rank flags but retains v1/direct publication.",
    "exhaustive_wizard_hyperliquid_cost_evidence.py": "forensic_b strictly parses metadata and paces funding pages but retains direct writes.",
    "exhaustive_wizard_hyperliquid_replay.py": "forensic_b uses _truthy for mapping_ready but retains direct writes.",
    "exhaustive_wizard_hyperliquid_run.py": "forensic_b uses _truthy for mapping/authority flags but retains v3/direct publication.",
    "exhaustive_wizard_hyperliquid_walkforward.py": "forensic_b adds a trial ledger and family-test count but retains direct publication and older math.",
    "hyperliquid_research_validation.py": "forensic_b reverses the selected Y-on-X Engle-Granger orientation and retains v2/direct publication.",
    "versioned_learning_comparison.py": "forensic_b strictly parses authority fields but retains direct publication.",
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_show(commit: str, path: str) -> bytes | None:
    run = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=ROOT, capture_output=True)
    return run.stdout if run.returncode == 0 else None


def parse_csv(data: bytes) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(data.decode("utf-8"))))


def refs(path: str) -> list[str]:
    module = path.removeprefix("src/").removesuffix(".py").replace("/", ".")
    run = subprocess.run(
        ["git", "grep", "-l", "-F", module, BASE, "--", "src", "scripts", "tests"],
        cwd=ROOT, capture_output=True, text=True,
    )
    if run.returncode not in (0, 1):
        raise RuntimeError(run.stderr)
    return sorted(
        line.removeprefix(BASE + ":") for line in run.stdout.splitlines()
        if line.removeprefix(BASE + ":") != path
    )


def missing_internal_imports(data: bytes) -> list[str]:
    tree = ast.parse(data.decode("utf-8"))
    modules = {
        node.module for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
        and node.module.startswith("quant_platform.")
    }
    return sorted(
        module for module in modules
        if git_show(BASE, "src/" + module.replace(".", "/") + ".py") is None
        and git_show(BASE, "src/" + module.replace(".", "/") + "/__init__.py") is None
    )


def build() -> dict[str, object]:
    subprocess.run(["git", "merge-base", "--is-ancestor", BASE, "HEAD"], cwd=ROOT, check=True)
    frozen = {}
    for path, expected in INPUT_SHA256.items():
        data = git_show(BASE, path)
        if data is None or sha(data) != expected:
            raise ValueError(f"frozen input drift: {path}")
        if path != QUEUE and (ROOT / path).read_bytes() != data:
            raise ValueError(f"working frozen input drift: {path}")
        frozen[path] = data
    original = git_show(ORIGINAL, QUEUE)
    if original is None or len(parse_csv(original)) != 811:
        raise ValueError("original 811-path decision snapshot missing")
    original_paths = {row["relative_path"] for row in parse_csv(original)}
    queue = parse_csv(frozen[QUEUE])
    baseline = {row["relative_path"]: row for row in parse_csv(frozen[FOUR_ROOT])}
    manifest = parse_csv(frozen[MANIFEST])
    missing_edges = parse_csv(frozen[MISSING_IMPORTS])
    cohort = {
        row["relative_path"]: row for row in queue
        if row["relative_path"] in original_paths
        and row["relative_path"].startswith(PREFIX)
        and row["custody_status"] == "PRESERVED_REVIEW_REQUIRED_NO_PORT"
    }
    expected_paths = {PREFIX + basename for basename in FINDINGS}
    if len(cohort) != 45 or set(cohort) != expected_paths:
        raise ValueError("original pending orchestration cohort changed")
    report_rows = []
    for path in sorted(cohort):
        basename = path.removeprefix(PREFIX)
        q = cohort[path]
        selected = git_show(BASE, path)
        selected_sha = sha(selected) if selected is not None else None
        if selected is not None and basename in BOOLEAN_GUARDS:
            active_anchor, candidate_anchor = BOOLEAN_GUARDS[basename]
            if active_anchor not in selected.decode("utf-8"):
                raise ValueError(f"selected boolean guard anchor drift: {path}")
        b = baseline[path]
        for key, qkey in (
            ("working_sha256", "working_sha256_at_freeze"),
            ("recovery_sha256", "recovery_sha256_at_freeze"),
            ("runtime_sha256", "runtime_sha256_at_freeze"),
        ):
            if q[qkey] and b[key] != q[qkey]:
                raise ValueError(f"four-root/queue mismatch: {path} {key}")
            if not q[qkey] and b[key] and b[key] != selected_sha:
                raise ValueError(f"omitted identical four-root hash differs from selected: {path} {key}")
        indexed: dict[str, list[dict[str, str]]] = {}
        bytes_by_sha: dict[str, bytes] = {}
        for entry in manifest:
            if entry["relative_path"] != path or entry["status"] != "hashed":
                continue
            copy = Path(entry["absolute_path"])
            if not copy.is_file() or copy.is_symlink():
                raise ValueError(f"frozen copy unavailable: {copy}")
            data = copy.read_bytes()
            if sha(data) != entry["sha256"]:
                raise ValueError(f"frozen copy hash drift: {copy}")
            digest = entry["sha256"]
            if digest in bytes_by_sha and bytes_by_sha[digest] != data:
                raise ValueError(f"same digest different bytes: {path}")
            bytes_by_sha[digest] = data
            indexed.setdefault(digest, []).append({
                "root": entry["root_label"],
                "absolute_path": entry["absolute_path"],
                "sha256": digest,
            })
        for root_name, root_path, key in (
            ("recovery_project", RECOVERY_ROOT, "recovery_sha256"),
            ("local_runtime", RUNTIME_ROOT, "runtime_sha256"),
        ):
            digest = b[key]
            if not digest:
                continue
            copy = root_path / path
            if not copy.is_file() or copy.is_symlink():
                raise ValueError(f"four-root copy unavailable: {copy}")
            data = copy.read_bytes()
            if sha(data) != digest:
                raise ValueError(f"four-root copy hash drift: {copy}")
            if digest in bytes_by_sha and bytes_by_sha[digest] != data:
                raise ValueError(f"four-root digest collision: {path}")
            bytes_by_sha[digest] = data
            indexed.setdefault(digest, []).append({
                "root": root_name,
                "absolute_path": str(copy),
                "sha256": digest,
            })
        historical = set(filter(None, q["historical_variant_sha256"].split(";")))
        if len(historical) != int(q["historical_variant_count"]) or not historical.issubset(indexed):
            raise ValueError(f"historical SHA custody mismatch: {path}")
        if not set(q["historical_copy_roots"].split(";")) - {""} <= {
            copy["root"] for digest in historical for copy in indexed[digest]
        }:
            raise ValueError(f"historical root custody mismatch: {path}")
        if basename in BOOLEAN_GUARDS:
            _, candidate_anchor = BOOLEAN_GUARDS[basename]
            if not any(candidate_anchor in bytes_by_sha[digest].decode("utf-8") for digest in historical):
                raise ValueError(f"strict boolean candidate missing: {path}")
        all_shas = set(indexed) | {
            b[key] for key in ("working_sha256", "recovery_sha256", "runtime_sha256") if b[key]
        }
        if selected_sha:
            all_shas.discard(selected_sha)
        variants = []
        for digest in sorted(all_shas):
            copies = sorted(indexed.get(digest, []), key=lambda x: (x["root"], x["absolute_path"]))
            roots = {copy["root"] for copy in copies}
            if not copies:
                category = "four_root_hash_only"
                assessment = "Frozen working SHA only; its byte copy is unavailable. No behavior is attributed to this hash beyond the initial four-root ledger."
            elif digest in historical:
                category = "historical_variant"
                assessment = FINDINGS[basename]
                if "forensic_b" in roots and basename in EXTRA_OLDER_FINDINGS:
                    assessment += " " + EXTRA_OLDER_FINDINGS[basename]
            else:
                category = "four_root_or_source_candidate"
                assessment = FINDINGS[basename]
            variants.append({
                "sha256": digest,
                "category": category,
                "four_root_roles": [
                    key.removesuffix("_sha256") for key in
                    ("working_sha256", "recovery_sha256", "runtime_sha256")
                    if b[key] == digest
                ],
                "copies": copies,
                "byte_reverified": bool(copies),
                "semantic_disposition": assessment,
                "whole_file_port": False,
            })
        runtime_bytes = (RUNTIME_ROOT / path).read_bytes() if b["runtime_sha256"] else None
        missing = missing_internal_imports(runtime_bytes) if runtime_bytes is not None else []
        frozen_missing = sorted({
            edge["imports_missing_from_working"] for edge in missing_edges
            if edge["source_runtime_only"] == path
        })
        # The import catalog predates later selective ports; keep both views.
        if basename in BOOLEAN_GUARDS:
            status_now = "PRESERVED_REVIEW_REQUIRED_NO_PORT"
            status_after_guard = BOOL_STATUS
            source_decision = "NARROW_STRICT_BOOL_PORT_ONLY_AFTER_TESTED_CANDIDATE"
        elif basename == "effect_authority.py":
            status_now = status_after_guard = "REVIEWED_EFFECT_V6_DEPENDENCY_NO_PORT"
            source_decision = "RETAIN_SELECTED_V2_NO_V6_PORT"
        elif selected is None:
            status_now = status_after_guard = RUNTIME_STATUS
            source_decision = "DEFER_RUNTIME_ONLY_DEPENDENCY_COHORT"
        else:
            status_now = status_after_guard = ACTIVE_STATUS
            source_decision = "RETAIN_SELECTED_ACTIVE_NO_WHOLE_FILE_PORT"
        report_rows.append({
            "relative_path": path,
            "selected_active_at_base_sha256": selected_sha,
            "selected_present": selected is not None,
            "four_root_sha256": {key: b[key] for key in ("working_sha256", "recovery_sha256", "runtime_sha256")},
            "four_root_decision": b["decision"],
            "queue_historical_sha256_at_base": sorted(historical),
            "queue_historical_copy_roots_at_base": q["historical_copy_roots"],
            "selected_python_reference_paths_at_base": refs(path),
            "runtime_missing_selected_internal_imports": missing,
            "prior_missing_import_catalog_entries": frozen_missing,
            "semantic_assessment": FINDINGS[basename],
            "source_decision": source_decision,
            "recommended_queue_status_now": status_now,
            "recommended_queue_status_after_guard_candidate": status_after_guard,
            "recommended_decision_rationale": FINDINGS[basename] + " Preserve all frozen SHA/copy-root custody; do not whole-file port.",
            "recommended_decision_evidence_append": REPORT.relative_to(ROOT).as_posix(),
            "existing_queue_decision_evidence": q["decision_evidence"],
            "variants": variants,
        })
    return {
        "schema_version": "thewiz.gate0.original_orchestration_pending_reconciliation.v1",
        "base_commit": BASE,
        "original_811_decision_commit": ORIGINAL,
        "inputs_sha256": INPUT_SHA256,
        "cohort_paths": len(report_rows),
        "selected_active_paths": sum(row["selected_present"] for row in report_rows),
        "runtime_only_paths": sum(not row["selected_present"] for row in report_rows),
        "queue_historical_variant_shas": sum(len(row["queue_historical_sha256_at_base"]) for row in report_rows),
        "whole_file_ports": 0,
        "separate_narrow_guard_candidate_paths": [
            PREFIX + "dynamic_stage_runner.py", PREFIX + "venue_gate.py"
        ],
        "external_effects_or_orders_executed": False,
        "expansion_queue_edited": False,
        "rows": report_rows,
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
        raise ValueError("committed report does not match frozen evidence")
    print(
        f"PASS original orchestration: {result['cohort_paths']} paths; "
        f"{result['queue_historical_variant_shas']} queue historical SHAs; "
        f"{result['runtime_only_paths']} runtime-only"
    )


if __name__ == "__main__":
    main()
