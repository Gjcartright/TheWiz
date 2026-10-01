#!/usr/bin/env python3
"""Reconcile the 26 newly discovered corrective source variants at 99df4d1.

Read-only: hashes selected Git source and frozen copies, checks the frozen queue
and manifests, then verifies path-specific semantic regression anchors. Use
--write-report only to reproduce the committed evidence JSON.
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
GAP = "audit/GATE0_EXTENDED_VARIANT_GAP_2026-09-30.json"
MANIFEST = "audit/GATE0_EXTENDED_VARIANT_INPUTS_2026-09-30.csv"
FOUR_ROOT = "audit/evidence_freeze_2026-09-29/source_reconciliation.csv"
QUEUE = "audit/GATE0_UNION_SOURCE_QUEUE_2026-09-30.csv"
REPORT = ROOT / "audit" / "GATE0_CORRECTIVE_EXTENDED_VARIANT_RECONCILIATION_2026-09-30.json"
INPUT_SHA256 = {
    GAP: "843abe2a6f4c2234ee1acad4d78c53e13e4ebbc9ef375a402dc6ef1daf136383",
    MANIFEST: "ca055e5cfbd6c5cdcd1e42b712b394aac2701bcf01a173fef151248e9eb046b3",
    FOUR_ROOT: "cecdd2603471da26db24b826d0a6aafe1ab11b36823f014c1abfe02ec2406943",
    QUEUE: "6e0beb2f87635f5e7af344519a333ef06b9da8c5ace6175e911380b79791b9b5",
}
PREFIX = "src/quant_platform/orchestration/"
# Each pair is an exact source-only and variant-only anchor. Findings describe
# the safety consequence of importing the additional frozen bytes.
FINDINGS = {
    "corrective_agent_governance.py": (
        "publication_authority", "Variant bypasses the governed staged publisher for agent-governance CSV and JSON; UTC alias change is equivalent.",
        "promote_staged_file(temporary, path)", "temporary.replace(path)"),
    "corrective_canonical_status.py": (
        "publication_and_redaction", "Variant publishes status via os.replace and embeds raw Stage 4 validator exception text in blockers.",
        "stage4_validator_failed:{safe_exception_code(exc)}", "stage4_validator_failed:{type(exc).__name__}:{exc}"),
    "corrective_daily_scheduler.py": (
        "scheduler_authority", "Variant removes supervised LaunchAgent provenance, terminal receipt lineage, governed lock and blocked post-run handoff; handwritten schedule can credit unproven runs.",
        "load_validated_scheduler_terminal_receipt_by_run(", "def _timeout_runner("),
    "corrective_governance.py": (
        "promotion_authority", "Variant grants promotion_authority when policy source hashes match, although that proves file identity only; also bypasses governed promotion.",
        "# Matching policy source hashes proves file identity, not candidate performance.", '"promotion_authority": not blockers,'),
    "corrective_live_canary_execution.py": (
        "live_order_authority", "Variant removes exact consumed and claimed order authority from leverage, entry, exit, cancellation and recovery, plus authorized mainnet info access.",
        "require_order_authority(order_authority)", "def _default_info_client(base_url: str)"),
    "corrective_live_canary_executor.py": (
        "mainnet_credential_authority", "Variant reads the live Keychain secret and mainnet info without authorized credential or network effect wrappers.",
        "read_authorized_keychain_credential(", "def read_live_agent_key_from_keychain(service: str, account: str) -> str:"),
    "corrective_live_parity_capture.py": (
        "mainnet_info_authority", "Variant calls Hyperliquid info directly, loosens environment-to-endpoint matching, omits result recorder, and persists raw error strings.",
        "run_authorized_hyperliquid_info_call(", "if info_url not in {HYPERLIQUID_TESTNET_INFO_URL, HYPERLIQUID_LIVE_INFO_URL}:"),
    "corrective_program.py": (
        "publication_authority", "Variant writes checkpoint Markdown directly and replaces governed CSV/JSON promotion with Path.replace.",
        "atomic_write_text(md_path,", "md_path.write_text("),
    "corrective_registered_learning.py": (
        "research_acceptance", "Variant trusts runner acceptance without requiring recomputed RL acceptance and persists raw validation/error text.",
        "and _truthy(computed_rl_row.get(\"accepted\"))", '"runner_summary_accepted": _truthy(rl.summary.get("accepted")),'),
    "corrective_registered_rerun_executor.py": (
        "research_publication", "Variant removes coded research-authority violation, uses raw errors and copy2/Path.replace rather than governed atomic file and directory publication.",
        "class ResearchAuthorityViolation(ValueError):", "temporary.replace(workspace)"),
    "corrective_release_gates.py": (
        "release_publication", "Variant replaces governed archive and checkpoint publication with direct writes/rename and persists raw validation errors.",
        "promote_staged_directory(temporary, archive_path)", "temporary.rename(archive_path)"),
    "corrective_runtime.py": (
        "runtime_publication", "Variant lacks scheduler capability and runtime contract binding and the governed atomic writer's lease/target checks.",
        "class SchedulerContract:", "def _replace_existing_file_in_place("),
    "corrective_scheduler_runtime_readiness.py": (
        "scheduler_readiness", "Variant omits fresh terminal PASS identity, interpreter/dependency probe and log redaction checks from readiness.",
        "def _terminal_receipt_observation(", "def _log_stat(path: Path)"),
    "corrective_testnet_pair_execution.py": (
        "testnet_effect_authority", "Variant allows substitute pair executor and raw testnet info requests, and uses direct exclusive writes.",
        "def _require_gate00g_testnet_pair_executor(", 'with path.open("x", encoding="utf-8") as handle:'),
    "corrective_wizard_browser_auth.py": (
        "browser_auth_validation", "Variant drops bounded receipt age and exact independent route-kind validation and bypasses governed publication.",
        'if receipt.get("required_route_kinds") != list(REQUIRED_ROUTE_KINDS):', "os.replace(temporary, path)"),
    "corrective_wizard_comparator_review_control.py": (
        "review_publication", "Variant emits a stale .venv312 command and replaces governed review artifact publication with Path.replace.",
        "PYTHONPATH=src .venv/bin/python3 -m quant_platform.cli ", "PYTHONPATH=src .venv312/bin/python -m quant_platform.cli "),
    "corrective_wizard_dynamic_holdout.py": (
        "holdout_publication", "Variant replaces governed staged and immutable JSON publication with Path.replace/direct write.",
        "atomic_write_text(path, encoded, encoding=\"utf-8\")", "path.write_text(encoded, encoding=\"utf-8\")"),
    "corrective_wizard_ou_holdout.py": (
        "credential_and_holdout_publication", "Variant reads environment key during nonexecute paths, logs raw provider errors and writes selector evidence directly; its custom immutable link is redundant.",
        "safe_exception_code(exc)", 'key = api_key or os.getenv("CRYPTO_WIZARDS_API_KEY", "").strip()'),
    "corrective_wizard_ou_v4_holdout.py": (
        "credential_and_redaction", "Variant reads environment key during nonexecute paths and persists raw credit/capture errors.",
        "safe_exception_code(exc)", 'key = api_key or os.getenv("CRYPTO_WIZARDS_API_KEY", "").strip()'),
    "corrective_wizard_ou_v5_holdout.py": (
        "credential_and_redaction", "Variant reads environment key during nonexecute paths and persists raw credit/capture/registration errors.",
        "safe_exception_code(exc)", 'key = api_key or os.getenv("CRYPTO_WIZARDS_API_KEY", "").strip()'),
    "corrective_wizard_ou_v6_holdout.py": (
        "credential_and_redaction", "Variant reads environment key during nonexecute paths and persists raw credit/capture/registration errors.",
        "safe_exception_code(exc)", 'key = api_key or os.getenv("CRYPTO_WIZARDS_API_KEY", "").strip()'),
    "corrective_wizard_parity.py": (
        "canonical_mode_publication", "Variant duplicates canonical Wizard mode names locally and writes capture, fixture and fidelity evidence directly.",
        "PAIR_PAGE_EXACT_MODES = CANONICAL_WIZARD_MODES", "PAIR_PAGE_EXACT_MODES = ("),
    "corrective_wizard_proof_launcher.py": (
        "external_effect_supervision", "Variant launches proof outside scheduler supervision, omits authorized unattended preflight and in-process effect issuer, and writes heartbeat without publication lease.",
        "supervise_scheduler_run(", "result = run_wizard_proof_launcher("),
    "corrective_wizard_proof_scheduler.py": (
        "paid_external_effect_authority", "Variant removes reserved external-effect session, request ceiling, authorized credential and supervisor; treats a passed credit reservation as spend authority by default.",
        "reserved_external_effect_session(", "load_env_file(path, override=False)"),
    "corrective_wizard_reset_readiness.py": (
        "scheduler_identity_and_publication", "Variant hardcodes .venv312 interpreter and replaces governed readiness artifact promotion with Path.replace.",
        "expected_python = scheduler_python_path(root)", 'expected_python = root / ".venv312" / "bin" / "python"'),
    "corrective_wizard_surface_inventory.py": (
        "inventory_publication", "Variant replaces governed immutable and mutable publication with a local hard-link implementation and Path.replace; repeat-link collision handling differs.",
        "write_immutable_json(path, payload)", "os.link(temporary, path)"),
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def at_base(relative: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{BASE}:{relative}"], cwd=ROOT,
        check=True, capture_output=True,
    ).stdout


def csv_rows(data: bytes) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(data.decode("utf-8"))))


def build() -> dict[str, object]:
    subprocess.run(["git", "merge-base", "--is-ancestor", BASE, "HEAD"], cwd=ROOT, check=True)
    inputs: dict[str, bytes] = {}
    for relative, expected in INPUT_SHA256.items():
        data = at_base(relative)
        if sha(data) != expected:
            raise ValueError(f"frozen input drift: {relative}")
        # The queue is intentionally updated after this review; its exact
        # decision snapshot remains anchored to BASE.
        if relative != QUEUE and (ROOT / relative).read_bytes() != data:
            raise ValueError(f"current frozen input drift: {relative}")
        inputs[relative] = data
    gap = json.loads(inputs[GAP])
    manifest = csv_rows(inputs[MANIFEST])
    baseline = {row["relative_path"]: row for row in csv_rows(inputs[FOUR_ROOT])}
    queue = {row["relative_path"]: row for row in csv_rows(inputs[QUEUE])}
    current_queue = {
        row["relative_path"]: row for row in csv_rows((ROOT / QUEUE).read_bytes())
    }
    manifest_copies = {
        (row["root_label"], row["relative_path"], row["sha256"], row["absolute_path"])
        for row in manifest if row["status"] == "hashed"
    }
    gap_rows = {
        row["relative_path"]: row for row in gap["rows"]
        if row["relative_path"].startswith(PREFIX + "corrective_")
    }
    expected_paths = {PREFIX + name for name in FINDINGS}
    if len(FINDINGS) != 26 or set(gap_rows) != expected_paths:
        raise ValueError("corrective gap path set changed")
    rows = []
    for path in sorted(expected_paths):
        level, finding, active_anchor, variant_anchor = FINDINGS[path.removeprefix(PREFIX)]
        row = gap_rows[path]
        if len(row["unlisted_variants"]) != 1:
            raise ValueError(f"new variant cardinality changed: {path}")
        variant = row["unlisted_variants"][0]
        active_bytes = at_base(path)
        if (ROOT / path).read_bytes() != active_bytes:
            raise ValueError(f"selected active source drift: {path}")
        active_text = active_bytes.decode("utf-8")
        variant_text = None
        copies = []
        for copy in variant["copies"]:
            frozen = Path(copy["absolute_path"])
            if not frozen.is_file() or frozen.is_symlink():
                raise ValueError(f"frozen copy unavailable: {path} @ {copy['root']}")
            frozen_bytes = frozen.read_bytes()
            if sha(frozen_bytes) != variant["sha256"] or copy["frozen_sha256"] != variant["sha256"]:
                raise ValueError(f"frozen copy hash drift: {path} @ {copy['root']}")
            if copy["current_sha256"] != variant["sha256"]:
                raise ValueError(f"gap copy changed before review: {path} @ {copy['root']}")
            if (copy["root"], path, variant["sha256"], copy["absolute_path"]) not in manifest_copies:
                raise ValueError(f"copy missing from filtered manifest: {path} @ {copy['root']}")
            candidate_text = frozen_bytes.decode("utf-8")
            if variant_text is not None and candidate_text != variant_text:
                raise ValueError(f"same SHA, different text: {path}")
            variant_text = candidate_text
            copies.append({"root": copy["root"], "absolute_path": copy["absolute_path"], "sha256": variant["sha256"]})
        assert variant_text is not None
        if active_anchor not in active_text or active_anchor in variant_text:
            raise ValueError(f"selected guard anchor invalid: {path}")
        if variant_anchor not in variant_text or variant_anchor in active_text:
            raise ValueError(f"variant regression anchor invalid: {path}")
        if any(
            row["four_root_baseline_sha256"][key] != baseline[path][key]
            for key in ("working_sha256", "recovery_sha256", "runtime_sha256")
        ):
            raise ValueError(f"gap/four-root baseline mismatch: {path}")
        queued = queue[path]
        current = current_queue[path]
        if (
            variant["sha256"] not in current["historical_variant_sha256"].split(";")
            or not {copy["root"] for copy in copies}.issubset(
                current["historical_copy_roots"].split(";")
            )
        ):
            raise ValueError(f"current queue lost frozen variant metadata: {path}")
        recorded_shas = [value for value in queued["historical_variant_sha256"].split(";") if value]
        recorded_roots = [value for value in queued["historical_copy_roots"].split(";") if value]
        if (queued["custody_status"] != "PRESERVED_ADDITIONAL_VARIANT_REVIEW_REQUIRED"
                or variant["sha256"] not in recorded_shas
                or len(recorded_shas) != int(queued["historical_variant_count"])):
            raise ValueError(f"frozen queue metadata mismatch: {path}")
        if not {copy["root"] for copy in copies}.issubset(recorded_roots):
            raise ValueError(f"frozen queue root metadata mismatch: {path}")
        expected_rationale = (
            current["decision_rationale"].replace(
                "Additional frozen source variant requires semantic review; prior decision covers the previously recorded copies.",
                "",
            ).strip()
        )
        if current["custody_status"] != row["prior_custody_status"]:
            raise ValueError(f"current queue decision not closed: {path}")
        if "Additional frozen variant " + variant["sha256"] not in expected_rationale:
            raise ValueError(f"current queue missing variant rationale: {path}")
        if REPORT.relative_to(ROOT).as_posix() not in current["decision_evidence"].split("; "):
            raise ValueError(f"current queue missing report citation: {path}")
        rows.append({
            "relative_path": path,
            "selected_active_sha256": sha(active_bytes),
            "frozen_four_root_sha256": {key: baseline[path][key] for key in ("working_sha256", "recovery_sha256", "runtime_sha256")},
            "additional_variant_sha256": variant["sha256"],
            "additional_variant_copies": copies,
            "previously_recorded_variant_sha256": [value for value in recorded_shas if value != variant["sha256"]],
            "queue_status_at_base": queued["custody_status"],
            "recommended_queue_status": row["prior_custody_status"],
            "recommended_decision_rationale_append": "Additional frozen variant " + variant["sha256"] + ": " + finding + " Retain selected active source.",
            "recommended_decision_evidence_append": REPORT.relative_to(ROOT).as_posix(),
            "recommended_source_decision": "NO_PORT_RETAIN_SELECTED_ACTIVE",
            "safety_area": level,
            "semantic_difference": finding,
            "selected_guard_anchor": active_anchor,
            "variant_regression_anchor": variant_anchor,
        })
    return {
        "schema_version": "thewiz.gate0.corrective_extended_variant_reconciliation.v1",
        "base_commit": BASE,
        "decision": "RECONCILE_26_ADDITIONAL_CORRECTIVE_VARIANTS_NO_PORT",
        "inputs_sha256": INPUT_SHA256,
        "source_prefix": PREFIX + "corrective_",
        "additional_variant_paths": len(rows),
        "additional_variant_sha256_count": len({row["additional_variant_sha256"] for row in rows}),
        "selected_source_changes": 0,
        "external_effects_or_orders_executed": False,
        "queue_edited": False,
        "recommendation": "Restore each row's pre-gap reviewed status after appending this report to decision_evidence; retain selected active bytes and all historical SHA/root metadata.",
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    expected = build()
    encoded = json.dumps(expected, indent=2, sort_keys=True) + "\n"
    if args.write_report:
        REPORT.write_text(encoded, encoding="utf-8")
    elif REPORT.read_text(encoding="utf-8") != encoded:
        raise ValueError("committed corrective reconciliation report differs from verified evidence")
    print(f"PASS corrective extended gap: {expected['additional_variant_paths']} paths; 0 ports")


if __name__ == "__main__":
    main()
