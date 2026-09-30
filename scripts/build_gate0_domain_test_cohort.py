"""Build and verify the original-811 pending domain-test custody review.

This is an evidence-only audit. It reads frozen manifests and preserved copies;
it does not import project modules, run tests, or invoke providers.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import subprocess
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "audit/GATE0_UNION_SOURCE_QUEUE_2026-09-30.csv"
FREEZE = ROOT / "audit/evidence_freeze_2026-09-29/source_reconciliation.csv"
EXTENDED = Path("/Volumes/Expansion/Crypto Wizard/audit/SECOND_PASS_EXTENDED_MANIFEST_2026-09-29.csv")
RECOVERY = Path("/Volumes/Expansion/TheWiz-Workspace/project")
RUNTIME = Path("/Users/gregc/TheWiz-LocalRuntime")
REPORT = ROOT / "audit/GATE0_DOMAIN_TEST_COHORT_REVIEW_2026-09-30.json"
PROPOSAL = ROOT / "audit/GATE0_DOMAIN_TEST_QUEUE_PROPOSAL_2026-09-30.csv"
BASE_COMMIT = "9b7fabfd5133904451199219cce53111f14de7cb"
PRE_FREEZE_COMMIT = "b98d5b9447a109938b57b239a7d458f47606d7c9"

PREFIXES = (
    "tests/test_wizard_",
    "tests/test_current_wizard_",
    "tests/test_hyperliquid_",
    "tests/test_binance_",
    "tests/test_pair_",
    "tests/test_research_",
    "tests/test_v2_",
    "tests/test_youtube_",
    "tests/test_execution",
)

# Each decision is tied to the exact preserved test path, not a family-wide port.
# A candidate is still NO_PORT until its source and test repair is independently reviewed.
DECISIONS = {
    "tests/test_binance_testnet.py": (
        "REVIEWED_DOMAIN_AUTHORITY_DEPENDENCY_DEFERRED_NO_PORT",
        "Runtime adds canonical instance-method shadow denial. Historical forensic tests instead permit a testnet submission path and omit active gate00g denial. Review the runtime adapter guard and its test together; retain selected active tests now.",
        "Canonical method identity and pair submission authority need a source-bound guard review.",
        "src/quant_platform/binance_testnet.py",
    ),
    "tests/test_current_wizard_hyperliquid_daily_runner.py": (
        "REVIEWED_DOMAIN_AUTHORITY_DEPENDENCY_DEFERRED_NO_PORT",
        "Runtime adds accounting failure stop, Keychain-reference, and invalid-placeholder checks while changing the selected fresh-passing-artifact test. Historical tests omit selected in-process/effect-supervision checks. Integrate only with its scheduler source cohort.",
        "Daily child/effect accounting and artifact qualification remain source-dependent.",
        "src/quant_platform/orchestration/current_wizard_hyperliquid_daily_runner.py",
    ),
    "tests/test_current_wizard_hyperliquid_handoff.py": (
        "REVIEWED_DOMAIN_AUTHORITY_DEPENDENCY_DEFERRED_NO_PORT",
        "Runtime adds 24 path, byte, mutation, funding-cache, terminal-beta, and replay custody tests for a later handoff contract. Forensic also adds a causal Y-on-X exposure test. Keep selected test contract until corresponding source family is integrated.",
        "Historical custody and causal-orientation checks warrant a coherent handoff source/test review.",
        "src/quant_platform/orchestration/current_wizard_hyperliquid_handoff.py",
    ),
    "tests/test_current_wizard_hyperliquid_math_comparison.py": (
        "REVIEWED_ACTIVE_TEST_RETAIN_HISTORICAL_NO_PORT",
        "Runtime is byte-identical to selected; older forensic copy lacks current comparison coverage. No source or test change justified by this path.",
        "",
        "src/quant_platform/orchestration/current_wizard_hyperliquid_math_comparison.py",
    ),
    "tests/test_execution.py": (
        "REVIEWED_DOMAIN_AUTHORITY_DEPENDENCY_DEFERRED_NO_PORT",
        "Runtime adds browser-state coercion and freshness, dYdX market metadata, adapter identity, helper/method mutation, and wrapper revalidation tests. Historical forensic tests allow unfenced client paths. Review runtime execution/effect source and tests as one authority contract.",
        "Current selected execution source may lack runtime gate00g method-identity and metadata guards; do not infer a proved exploit from test drift alone.",
        "src/quant_platform/execution.py",
    ),
    "tests/test_hyperliquid_testnet.py": (
        "REVIEWED_DOMAIN_AUTHORITY_DEPENDENCY_DEFERRED_NO_PORT",
        "Runtime adds canonical executor method-shadow denial. Older forensic test delegates through a substituted executor and shifts approval timing. Keep selected denial/approval tests; review guard source with the runtime test.",
        "Canonical executor method identity should be checked before pair submission.",
        "src/quant_platform/hyperliquid_testnet.py",
    ),
    "tests/test_research_assertion_isolation.py": (
        "REVIEWED_RUNTIME_ONLY_DEPENDENCY_DEFERRED_NO_PORT",
        "Runtime-only AST scan asserts research_assertions is isolated from runtime imports. Selected checkout has no research_assertions.py, so it is not an applicable standalone test port.",
        "",
        "src/quant_platform/research_assertions.py",
    ),
    "tests/test_research_ingestion.py": (
        "REVIEWED_TARGETED_SOURCE_TEST_REPAIR_CANDIDATE_NO_PORT",
        "Runtime matches selected. Forensic adds AppleDouble sidecar exclusion, but removes selected CCXT evidence/failure tests. Select only the sidecar regression with a narrow registry-source fix; do not replace this file wholesale.",
        "Selected registry glob includes ._*.json sidecars and can create spurious quarantined source rows.",
        "src/quant_platform/research_ingestion.py",
    ),
    "tests/test_research_paper_reproduction.py": (
        "REVIEWED_ACTIVE_TEST_RETAIN_HISTORICAL_NO_PORT",
        "Selected changed after freeze to add canonical causal Y-on-X pair-frame test. Runtime and forensic are older. Retain selected stronger test; preserve frozen byte history.",
        "",
        "src/quant_platform/research_paper_reproduction.py",
    ),
    "tests/test_research_paper_reviews.py": (
        "REVIEWED_ACTIVE_TEST_RETAIN_HISTORICAL_NO_PORT",
        "Selected uses synthetic fixtures for all paper assessments; forensic differs by using a copied real inventory. Runtime matches selected. Keep deterministic selected test.",
        "",
        "src/quant_platform/research_paper_reviews.py",
    ),
    "tests/test_research_producer_publication_safety.py": (
        "REVIEWED_RUNTIME_ONLY_DEPENDENCY_DEFERRED_NO_PORT",
        "Runtime-only four-case publication test imports seven producer scripts absent in selected checkout. It cannot be collected here without the script/source family.",
        "Source publication path and symlink authority need review if those seven scripts are adopted.",
        "scripts/build_authoritative_lineage_closure.py",
    ),
    "tests/test_wizard_candidate_set.py": (
        "REVIEWED_TARGETED_SOURCE_TEST_REPAIR_CANDIDATE_NO_PORT",
        "Runtime matches selected. Forensic adds 1m minute versus 1M monthly candidate-set identity regression. Selected source lowercases interval before hashing, collapsing these distinct timeframes. Port narrowly with source fix.",
        "Lowercasing interval can collide minute/monthly candidate identities and contaminate downstream lineage.",
        "src/quant_platform/wizard_candidate_set.py",
    ),
    "tests/test_wizard_control_plane.py": (
        "REVIEWED_ACTIVE_TEST_RETAIN_HISTORICAL_NO_PORT",
        "Runtime changes only the shared effect-authority test secret fixture; forensic lacks selected authority tests. Keep selected fixture until shared authority setup is integrated.",
        "",
        "src/quant_platform/wizard_control_plane.py",
    ),
    "tests/test_wizard_credit_ledger.py": (
        "REVIEWED_DOMAIN_AUTHORITY_DEPENDENCY_DEFERRED_NO_PORT",
        "Runtime changes publication worker context and zero-attempt external-spend authority semantics; forensic drops exact supervisor-binding test. Runtime references source helper absent in selected. Review source and test together.",
        "Zero-attempt reconciliation and shared credit reservations are authority-sensitive; no test-only transplant.",
        "src/quant_platform/wizard_credit_ledger.py",
    ),
    "tests/test_wizard_evidence.py": (
        "REVIEWED_TARGETED_SOURCE_TEST_REPAIR_CANDIDATE_NO_PORT",
        "Runtime expects v2 config identity while selected expects v1. Forensic adds _bars_per_day minute/monthly regression. Selected source lowercases 1M and maps neither 1m nor 1M, defaulting both to one bar/day. Select only the timeframe repair and regression.",
        "Cost sensitivity understates minute bars and misstates monthly bars.",
        "src/quant_platform/wizard_evidence.py",
    ),
    "tests/test_wizard_mode_replay.py": (
        "REVIEWED_DOMAIN_AUTHORITY_DEPENDENCY_DEFERRED_NO_PORT",
        "Runtime adds public CANONICAL_WIZARD_MODES and strict dynamic warmup tests; selected source lacks that symbol. Forensic adds stored-ratio/conditional tests but removes selected OU unit and economic-contract guards. Review source/test as a bounded math contract.",
        "Preserve selected OU unit and directional economic guards during any replay integration.",
        "src/quant_platform/wizard_mode_replay.py",
    ),
    "tests/test_wizard_papers_reproduction_controls.py": (
        "REVIEWED_RUNTIME_ONLY_DEPENDENCY_DEFERRED_NO_PORT",
        "Runtime-only WP001 custody test binds 53 paper hashes, 57 physical PDFs, and 59 sidecars to an external Wizard Papers drive and runtime report package; neither package nor drive corpus is in selected checkout. Preserve as frozen external custody evidence.",
        "",
        "reports/research/wizard_papers_2026-08-24/research_receipt.json",
    ),
    "tests/test_wizard_research_journal.py": (
        "REVIEWED_ACTIVE_TEST_RETAIN_HISTORICAL_NO_PORT",
        "Selected changed after freeze to reject duplicate schema definitions and a schema-owned journal_layer. Runtime/forensic are older. Retain selected regression and preserve frozen bytes.",
        "",
        "src/quant_platform/wizard_research_journal.py",
    ),
    "tests/test_youtube_brain.py": (
        "REVIEWED_TARGETED_SOURCE_TEST_REPAIR_CANDIDATE_NO_PORT",
        "Runtime and visual recovery match selected. Forensic adds external-prior-only versus native-evidence readiness tests; selected brain status currently marks any nonempty claims ready and has no native_evidence_ready field. Requires narrow source/test contract review.",
        "External priors can currently be labeled ready in brain status without native video evidence.",
        "src/quant_platform/youtube_brain.py",
    ),
}


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _test_names(data: bytes) -> list[str]:
    tree = ast.parse(data.decode("utf-8"))
    return sorted({node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_")})


def _copy(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {"exists": False, "sha256": "", "test_names": []}
    data = path.read_bytes()
    return {"exists": True, "sha256": _hash(data), "test_names": _test_names(data) if path.suffix == ".py" and path.name.startswith("test_") else []}


def _git_bytes(revision: str, path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{revision}:{path}"], cwd=ROOT)


def build() -> dict[str, object]:
    queue = [(line, row) for line, row in enumerate(_read_csv(QUEUE), 2) if row["relative_path"].startswith(PREFIXES) and row["custody_status"] == "PRESERVED_REVIEW_REQUIRED_NO_PORT"]
    assert {row["relative_path"] for _, row in queue} == set(DECISIONS), "cohort membership changed"
    freeze = {row["relative_path"]: row for row in _read_csv(FREEZE)}
    extended: dict[str, list[dict[str, str]]] = {}
    for row in _read_csv(EXTENDED):
        if row["relative_path"] in DECISIONS and row["status"] == "hashed":
            extended.setdefault(row["relative_path"], []).append(row)
    files = []
    for line, q in queue:
        relative = q["relative_path"]
        status, rationale, risk, source = DECISIONS[relative]
        selected = _copy(ROOT / relative)
        if selected["exists"]:
            current_data = _git_bytes(BASE_COMMIT, relative)
            selected_data = (ROOT / relative).read_bytes()
            assert selected_data == current_data, f"selected test changed: {relative}"
        frozen = freeze[relative]
        if q["working_sha256_at_freeze"]:
            assert q["working_sha256_at_freeze"] == frozen["working_sha256"]
        if q["runtime_sha256_at_freeze"]:
            assert q["runtime_sha256_at_freeze"] == frozen["runtime_sha256"]
        frozen_working_sha = frozen["working_sha256"]
        provenance = "base_commit"
        if selected["sha256"] != frozen_working_sha:
            assert relative in {"tests/test_research_paper_reproduction.py", "tests/test_wizard_research_journal.py"}
            assert _hash(_git_bytes(PRE_FREEZE_COMMIT, relative)) == frozen_working_sha
            provenance = PRE_FREEZE_COMMIT
        recovery = _copy(RECOVERY / relative)
        runtime = _copy(RUNTIME / relative)
        assert recovery["sha256"] == frozen["recovery_sha256"]
        assert runtime["sha256"] == frozen["runtime_sha256"]
        variants = []
        for item in sorted(extended.get(relative, []), key=lambda row: (row["root_label"], row["absolute_path"])):
            path = Path(item["absolute_path"])
            actual = _copy(path)
            assert actual["sha256"] == item["sha256"], f"extended copy changed: {path}"
            variants.append({"root": item["root_label"], "path": str(path), "sha256": actual["sha256"], "test_names": actual["test_names"]})
        frozen_hashes = {value for value in (frozen_working_sha, frozen["recovery_sha256"], frozen["runtime_sha256"]) if value}
        distinct_historical = {item["sha256"] for item in variants} - frozen_hashes
        assert distinct_historical == set(q["historical_variant_sha256"].split(";")) - {""}
        assert len(distinct_historical) == int(q["historical_variant_count"] or 0)
        src = _copy(ROOT / source)
        entry = {
            "path": relative,
            "queue_line": line,
            "queue_status_at_review": q["custody_status"],
            "queue_working_vs_runtime": q["working_vs_runtime"],
            "queue_historical_variant_count": int(q["historical_variant_count"] or 0),
            "queue_historical_variant_sha256": q["historical_variant_sha256"].split(";") if q["historical_variant_sha256"] else [],
            "frozen_working_sha256": frozen_working_sha,
            "frozen_working_git_provenance": provenance,
            "frozen_recovery_sha256": frozen["recovery_sha256"],
            "frozen_runtime_sha256": frozen["runtime_sha256"],
            "selected_base_commit": BASE_COMMIT,
            "selected_sha256": selected["sha256"],
            "selected_changed_since_freeze": selected["sha256"] != frozen_working_sha,
            "recovery_sha256": recovery["sha256"],
            "runtime_sha256": runtime["sha256"],
            "selected_test_names": selected["test_names"],
            "runtime_added_test_names": sorted(set(runtime["test_names"]) - set(selected["test_names"])),
            "runtime_removed_test_names": sorted(set(selected["test_names"]) - set(runtime["test_names"])),
            "extended_copies": variants,
            "source_path": source,
            "selected_source_exists": src["exists"],
            "selected_source_sha256": src["sha256"],
            "recommended_queue_status": status,
            "rationale": rationale,
            "source_authority_risk": risk,
        }
        files.append(entry)
    counts = dict(sorted(Counter(row["recommended_queue_status"] for row in files).items()))
    return {
        "report_type": "gate0_original_811_domain_test_cohort",
        "base_commit": BASE_COMMIT,
        "queue_path": str(QUEUE),
        "freeze_path": str(FREEZE),
        "extended_manifest_path": str(EXTENDED),
        "pending_count": len(files),
        "status_counts": counts,
        "selection_rule": "original-811 pending PRESERVED_REVIEW_REQUIRED_NO_PORT paths matching listed test prefixes",
        "preserved_copies_verified": sum(2 + len(row["extended_copies"]) for row in files),
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true", help="compare saved report with fresh frozen/copy evidence")
    args = parser.parse_args()
    report = build()
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    table = io.StringIO()
    writer = csv.DictWriter(table, fieldnames=("relative_path", "custody_status", "decision_rationale", "decision_evidence"), lineterminator="\n")
    writer.writeheader()
    for item in report["files"]:
        writer.writerow({
            "relative_path": item["path"],
            "custody_status": item["recommended_queue_status"],
            "decision_rationale": item["rationale"],
            "decision_evidence": "audit/GATE0_DOMAIN_TEST_COHORT_REVIEW_2026-09-30.json; exact selected/frozen/runtime/recovery/extended hashes",
        })
    proposal = table.getvalue()
    if args.verify:
        assert REPORT.read_text(encoding="utf-8") == rendered, "saved report differs from evidence"
        assert PROPOSAL.read_text(encoding="utf-8") == proposal, "saved queue proposal differs from evidence"
        print(f"verified {report['pending_count']} paths and {report['preserved_copies_verified']} copy hashes")
    else:
        REPORT.write_text(rendered, encoding="utf-8")
        PROPOSAL.write_text(proposal, encoding="utf-8")
        print(f"wrote report and proposal: {report['pending_count']} paths")


if __name__ == "__main__":
    main()
