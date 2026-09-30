#!/usr/bin/env python3
"""Verify the ten other-orchestration extended source variant decisions.

All reads are local. The selected-source and queue snapshots are pinned to the
99df4d1 decision commit; current queue metadata may advance after review.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "99df4d1f3b3f34a6c6c50c2d111a2344c05c6a61"
REPAIR = "2fc9b433af2b3b9f8effffb8f9d997525c2dc2e0"
PREFIX = "src/quant_platform/orchestration/"
GAP = "audit/GATE0_EXTENDED_VARIANT_GAP_2026-09-30.json"
MANIFEST = "audit/GATE0_EXTENDED_VARIANT_INPUTS_2026-09-30.csv"
FOUR_ROOT = "audit/evidence_freeze_2026-09-29/source_reconciliation.csv"
QUEUE = "audit/GATE0_UNION_SOURCE_QUEUE_2026-09-30.csv"
REPORT = ROOT / "audit/GATE0_OTHER_ORCHESTRATION_EXTENDED_VARIANT_RECONCILIATION_2026-09-30.json"
REPAIRED_SOURCE_SHA256 = "b7447a22b9e9287b8aec645e3bbe2f13d3dcf03122f45c6d3863e4e34f20c2d8"
REPAIR_TEST_SHA256 = "ff0f6278bf4eacd5dbffe9f709d52d8f6f2b52e2aa390697e29b131e91f3c641"
INPUT_SHA256 = {
    GAP: "843abe2a6f4c2234ee1acad4d78c53e13e4ebbc9ef375a402dc6ef1daf136383",
    MANIFEST: "ca055e5cfbd6c5cdcd1e42b712b394aac2701bcf01a173fef151248e9eb046b3",
    FOUR_ROOT: "cecdd2603471da26db24b826d0a6aafe1ab11b36823f014c1abfe02ec2406943",
    QUEUE: "6e0beb2f87635f5e7af344519a333ef06b9da8c5ace6175e911380b79791b9b5",
}
# (New variant finding, selected-only guard, new-variant-only regression,
# prior forensic finding, prior forensic-only anchor). Empty prior entries mean
# the newly discovered SHA also resides in forensic_b.
FINDINGS = {
    "current_wizard_ou_optimal_overlay.py": (
        "Visual copy directly writes overlay CSV and manifest evidence, bypassing governed atomic publication.",
        "atomic_write_csv(frame, paths[active_key], index=False)", "frame.to_csv(paths[active_key], index=False)",
        "Older forensic copy also parses ledger flags with _truthy. Ledger flags are constructed as booleans in this builder, while both frozen copies lose governed publication.",
        'ledger["ou_optimal"].map(_truthy)'),
    "exhaustive_wizard_hyperliquid_validation.py": (
        "Visual copy directly writes validation manifest and Markdown, allowing partial public evidence on interruption.",
        "atomic_write_text(path, manifest, encoding=\"utf-8\")", "path.write_text(manifest, encoding=\"utf-8\")",
        "Older forensic copy parses completion and live-authority fields with strict_bool, which can avoid false-string acceptance; it also directly writes output. Leave boolean parsing as a separate scoped follow-up.",
        'strict_bool(summary.get("experiment_status_accounted", False))'),
    "hyperliquid_research_cycle.py": (
        "Visual copy replaces governed CSV/JSON promotion and atomic Markdown with direct writes.",
        "promote_staged_file(temporary, path)", "temporary.replace(path)",
        "Older forensic copy additionally parses family, preflight and authority flags strictly, but retains direct publication regressions; these are lower-authority research summaries.",
        'strict_bool(authority.get("execution_allowed", False))'),
    "hyperliquid_run_manifest.py": (
        "Visual copy directly writes canonical manifest, candidate, validation and authority artifacts.",
        "promote_staged_file(temporary, path)", "temporary.replace(path)",
        "Older forensic copy also strictly parses persisted manifest gates. The isolated 2fc9b43 repair ports only those authority flags and retains governed publication; raw False strings otherwise reached research/paper/live readiness.",
        'strict_bool(manifest.get("lineage_ready", False))'),
    "hyperliquid_testnet_lifecycle_evidence.py": (
        "Visual/forensic copy bypasses authorized Hyperliquid testnet info effect recording and directly replaces lifecycle evidence files.",
        "run_authorized_hyperliquid_info_call(", "response = session.post(f\"{config.base_url.rstrip('/')}/info\"",
        "Same SHA is present in visual_recovery and forensic_b; no separate older variant exists.",
        ""),
    "langgraph_workflow.py": (
        "Visual copy writes agent lane, edge, state and Markdown files directly instead of using governed atomic publication.",
        "atomic_write_csv(lanes, lane_path, index=False)", "lanes.to_csv(lane_path, index=False)",
        "Older forensic copy additionally parses persisted graph state booleans strictly; useful only as a separate tested state-routing change, while its direct publication remains unsafe.",
        'strict_bool(state.get("blocked", False))'),
    "mini_agents.py": (
        "Visual copy directly writes the agent registry, next-action queue and Markdown.",
        "atomic_write_csv(registry, registry_path, index=False)", "registry.to_csv(registry_path, index=False)",
        "Older forensic copy additionally parses persisted RL acceptance with _boolish; direct publication still weakens agent evidence integrity.",
        'acceptance["accepted"].map(_boolish)'),
    "nodes.py": (
        "Visual copy logs raw exception text and adds YouTube research to the broad RL group; current YouTube config enables live refresh, so RL runs could trigger network collection unexpectedly.",
        "reason=safe_exception_code(exc)", "reason=str(exc)",
        "Older forensic copy also adds YouTube to RL and raw error text; its strict bool readiness parsing and report-only fetch override do not justify whole-file import.",
        'strict_bool(control.summary.get("ready", False))'),
    "orchestrator_assistant.py": (
        "Visual copy directly writes task CSV, JSONL memory and reasoning artifacts, losing governed atomic append and publication.",
        "atomic_append_text(path, json.dumps(event, sort_keys=True) + \"\\n\")", 'with path.open("a", encoding="utf-8") as handle:',
        "Older forensic copy additionally parses outcome and promotion flags strictly, but keeps direct task and memory publication.",
        'strict_bool(event.get("outcome_known", False))'),
    "specialist_scoreboard.py": (
        "Visual copy directly writes scoreboard CSV and Markdown, risking partial dashboard research evidence.",
        "atomic_write_csv(scoreboard, scoreboard_path, index=False)", "scoreboard.to_csv(scoreboard_path, index=False)",
        "Older forensic copy additionally parses Sharpe and RL acceptance flags strictly; retain this as a separate dashboard repair idea, not a whole-file port.",
        'rl_acceptance["accepted"].map(strict_bool)'),
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def snapshot(relative: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{BASE}:{relative}"], cwd=ROOT,
        check=True, capture_output=True,
    ).stdout


def rows(data: bytes) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(data.decode("utf-8"))))


def build() -> dict[str, object]:
    subprocess.run(["git", "merge-base", "--is-ancestor", BASE, "HEAD"], cwd=ROOT, check=True)
    inputs: dict[str, bytes] = {}
    for relative, expected in INPUT_SHA256.items():
        data = snapshot(relative)
        if sha(data) != expected:
            raise ValueError(f"base input hash changed: {relative}")
        if relative != QUEUE and (ROOT / relative).read_bytes() != data:
            raise ValueError(f"current frozen input changed: {relative}")
        inputs[relative] = data
    gap = json.loads(inputs[GAP])
    manifest = rows(inputs[MANIFEST])
    baseline = {row["relative_path"]: row for row in rows(inputs[FOUR_ROOT])}
    queue = {row["relative_path"]: row for row in rows(inputs[QUEUE])}
    current_queue = {row["relative_path"]: row for row in rows((ROOT / QUEUE).read_bytes())}
    manifest_index = {
        (row["relative_path"], row["sha256"], row["root_label"], row["absolute_path"])
        for row in manifest if row["status"] == "hashed"
    }
    gaps = {row["relative_path"]: row for row in gap["rows"]}
    expected_paths = {PREFIX + name for name in FINDINGS}
    if len(FINDINGS) != 10 or not expected_paths.issubset(gaps):
        raise ValueError("other-orchestration path set changed")
    reviews = []
    for path in sorted(expected_paths):
        new_finding, selected_anchor, variant_anchor, older_finding, older_anchor = FINDINGS[path.removeprefix(PREFIX)]
        selected = snapshot(path)
        selected_text = selected.decode("utf-8")
        current_sha = sha((ROOT / path).read_bytes())
        permitted_current = {sha(selected)}
        if path.endswith("hyperliquid_run_manifest.py"):
            permitted_current.add(REPAIRED_SOURCE_SHA256)
            if current_sha == REPAIRED_SOURCE_SHA256 and sha((ROOT / "tests/test_hyperliquid_run_manifest_gate_types.py").read_bytes()) != REPAIR_TEST_SHA256:
                raise ValueError("strict gate repair test drift")
        if current_sha not in permitted_current:
            raise ValueError(f"selected source changed outside reviewed repair: {path}")
        gap_row = gaps[path]
        if len(gap_row["unlisted_variants"]) != 1:
            raise ValueError(f"new variant cardinality changed: {path}")
        new_sha = gap_row["unlisted_variants"][0]["sha256"]
        q = queue[path]
        active_q = current_queue[path]
        historical = [value for value in q["historical_variant_sha256"].split(";") if value]
        if (q["custody_status"] not in {gap_row["prior_custody_status"], "PRESERVED_ADDITIONAL_VARIANT_REVIEW_REQUIRED"}
                or new_sha not in historical or len(historical) != int(q["historical_variant_count"])):
            raise ValueError(f"frozen queue metadata mismatch: {path}")
        if not set(historical).issubset(active_q["historical_variant_sha256"].split(";")):
            raise ValueError(f"current queue lost historical SHA metadata: {path}")
        for key in ("working_sha256", "recovery_sha256", "runtime_sha256"):
            if gap_row["four_root_baseline_sha256"][key] != baseline[path][key]:
                raise ValueError(f"gap/four-root baseline mismatch: {path}")
        variants = []
        for variant_sha in historical:
            copies = []
            variant_text = None
            for entry in manifest:
                if entry["relative_path"] != path or entry["sha256"] != variant_sha:
                    continue
                frozen = Path(entry["absolute_path"])
                if not frozen.is_file() or frozen.is_symlink() or sha(frozen.read_bytes()) != variant_sha:
                    raise ValueError(f"frozen source drift: {path} @ {entry['root_label']}")
                if (path, variant_sha, entry["root_label"], entry["absolute_path"]) not in manifest_index:
                    raise ValueError(f"manifest copy missing: {path}")
                candidate_text = frozen.read_text(encoding="utf-8")
                if variant_text is not None and variant_text != candidate_text:
                    raise ValueError(f"same SHA has different text: {path}")
                variant_text = candidate_text
                copies.append({"root": entry["root_label"], "absolute_path": entry["absolute_path"], "sha256": variant_sha})
            if not copies or variant_text is None:
                raise ValueError(f"preserved variant missing from manifest: {path} {variant_sha}")
            if not {copy["root"] for copy in copies}.issubset(active_q["historical_copy_roots"].split(";")):
                raise ValueError(f"current queue lost historical root metadata: {path}")
            if variant_sha == new_sha:
                gap_copies = gap_row["unlisted_variants"][0]["copies"]
                if any(
                    copy["frozen_sha256"] != new_sha or copy["current_sha256"] != new_sha
                    for copy in gap_copies
                ) or {
                    (copy["root"], copy["absolute_path"]) for copy in gap_copies
                } != {(copy["root"], copy["absolute_path"]) for copy in copies}:
                    raise ValueError(f"gap/manifest copy set mismatch: {path}")
                if selected_anchor not in selected_text or selected_anchor in variant_text:
                    raise ValueError(f"selected guard anchor mismatch: {path}")
                if variant_anchor not in variant_text or variant_anchor in selected_text:
                    raise ValueError(f"new variant regression anchor mismatch: {path}")
                finding = new_finding
            else:
                if not older_anchor or older_anchor not in variant_text or older_anchor in selected_text:
                    raise ValueError(f"older forensic semantic anchor mismatch: {path}")
                if variant_anchor not in variant_text:
                    raise ValueError(f"older forensic copy lost direct-publish regression: {path}")
                finding = older_finding
            variants.append({
                "sha256": variant_sha,
                "newly_discovered": variant_sha == new_sha,
                "copies": copies,
                "semantic_difference": finding,
                "review_decision": "NO_WHOLE_FILE_PORT",
            })
        prior = gap_row["prior_custody_status"]
        status = (
            prior if prior.startswith("REVIEWED_")
            else "REVIEWED_STRICT_GATE_REPAIR_REMAINDER_NO_PORT"
            if path.endswith("hyperliquid_run_manifest.py")
            else "REVIEWED_OTHER_ORCHESTRATION_EXTENDED_VARIANTS_NO_PORT"
        )
        rationale = "Reviewed all frozen source variants: " + new_finding + " " + older_finding + " Keep selected source and variant SHA/root custody; port only the isolated strict manifest-gate repair where applicable."
        if active_q["custody_status"] != status:
            raise ValueError(f"current queue decision not closed: {path}")
        if rationale not in active_q["decision_rationale"]:
            raise ValueError(f"current queue missing variant rationale: {path}")
        if REPORT.relative_to(ROOT).as_posix() not in active_q["decision_evidence"].split("; "):
            raise ValueError(f"current queue missing report citation: {path}")
        reviews.append({
            "relative_path": path,
            "selected_active_at_base_sha256": sha(selected),
            "current_selected_sha256": current_sha,
            "frozen_four_root_sha256": {key: baseline[path][key] for key in ("working_sha256", "recovery_sha256", "runtime_sha256")},
            "frozen_four_root_decision": baseline[path]["decision"],
            "prior_custody_status_snapshot": prior,
            "queue_status_at_base": q["custody_status"],
            "recommended_queue_status": status,
            "recommended_decision_rationale_append": rationale,
            "recommended_decision_evidence_append": REPORT.relative_to(ROOT).as_posix(),
            "source_decision": "RETAIN_SELECTED_ACTIVE_NO_WHOLE_FILE_PORT",
            "variants": variants,
        })
    return {
        "schema_version": "thewiz.gate0.other_orchestration_extended_variant_reconciliation.v1",
        "base_commit": BASE,
        "selected_source_repair_commit": REPAIR,
        "selected_source_repair_path": PREFIX + "hyperliquid_run_manifest.py",
        "selected_source_repair_sha256": REPAIRED_SOURCE_SHA256,
        "selected_source_repair_test_sha256": REPAIR_TEST_SHA256,
        "inputs_sha256": INPUT_SHA256,
        "reviewed_paths": len(reviews),
        "reviewed_historical_variant_shas": sum(len(row["variants"]) for row in reviews),
        "newly_discovered_variant_shas": sum(sum(v["newly_discovered"] for v in row["variants"]) for row in reviews),
        "whole_file_ports": 0,
        "isolated_safety_repair": "strictly_parse_five_persisted_hyperliquid_manifest_readiness_gates",
        "external_effects_or_orders_executed": False,
        "queue_edited": False,
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
        raise ValueError("committed other-orchestration report differs from frozen evidence")
    print(f"PASS other orchestration: {result['reviewed_paths']} paths, {result['reviewed_historical_variant_shas']} frozen SHAs, 1 narrow repair")


if __name__ == "__main__":
    main()
