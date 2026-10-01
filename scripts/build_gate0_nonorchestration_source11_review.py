"""Verify the 11 original pending non-orchestration source variants.

The audit only reads pinned Git blobs, frozen manifests, and preserved copies.
It neither imports project modules nor contacts a provider.
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
BASE_COMMIT = "7776c58c4bca3216765ccb46be7e3a79afc21e3d"
FREEZE_GIT_COMMIT = "b98d5b9447a109938b57b239a7d458f47606d7c9"
QUEUE = ROOT / "audit/GATE0_UNION_SOURCE_QUEUE_2026-09-30.csv"
FREEZE = ROOT / "audit/evidence_freeze_2026-09-29/source_reconciliation.csv"
EXTENDED = ROOT / "audit/GATE0_EXTENDED_VARIANT_INPUTS_2026-09-30.csv"
RECOVERY = Path("/Volumes/Expansion/TheWiz-Workspace/project")
RUNTIME = Path("/Users/gregc/TheWiz-LocalRuntime")
REPORT = ROOT / "audit/GATE0_NONORCHESTRATION_SOURCE11_REVIEW_2026-09-30.json"
PROPOSAL = ROOT / "audit/GATE0_NONORCHESTRATION_SOURCE11_QUEUE_PROPOSAL_2026-09-30.csv"

# Source-only status proposals. All preserved variants remain in custody, and
# completed narrow ports are explicitly distinguished from a whole-file port.
DECISIONS = {
    "ablations.py": (
        "REVIEWED_NARROW_PORT_COMPLETE_HISTORICAL_NO_WHOLE_FILE_PORT",
        "0311956",
        "Selected source retains atomic CSV publication and now parses persisted tested/baseline eligibility with strict_bool. Forensic_b only changes atomic publication to direct to_csv; visual_recovery also uses bool('False') and would elevate false eligibility. No whole-file port.",
        "",
    ),
    "apify_sources.py": (
        "REVIEWED_EXACT_REQUEST_DEPENDENCY_DEFERRED_NO_WHOLE_FILE_PORT",
        "",
        "Recovery/runtime source switches real Apify requests to ExactHttpRequest with exact method, path, query, and payload binding while retaining the injected-client callback. The selected corrective external-effect module lacks that request type/API, so the source cannot be transplanted. Forensic/visual copy predates the governed call path. Keep selected callback authority until a complete exact-request dependency is integrated.",
        "Selected callback authorization does not attest the exact Apify URL/path/query/payload; this remains an external-effect authority gap.",
    ),
    "binance_testnet.py": (
        "REVIEWED_ADAPTER_AUTHORITY_REPAIR_CANDIDATE_NO_WHOLE_FILE_PORT",
        "",
        "Selected source enforces gate00g testnet adapter type/order authority; older forensic/visual copy removes those checks. Runtime source candidate adds canonical type registration and instance-method binding checks, but depends on a broader execution attestation API. Review a focused source/test guard; no wholesale import of the older or runtime file.",
        "Pair submission validates adapter type once, then directly invokes mutable place_order for both entry and rollback; method shadowing can evade the selected type-only check.",
    ),
    "env.py": (
        "REVIEWED_SECRET_PROVENANCE_REPAIR_CANDIDATE_NO_WHOLE_FILE_PORT",
        "",
        "Selected, recovery, and runtime bytes match. Forensic adds owner-only and non-symlink secret-file resolution but removes load_selected_env_keys, which active proof scheduler and API credit receipt import; visual also drops the loader. Keep selected API and review a narrow file-provenance guard.",
        "The API credit receipt caller uses load_selected_env_keys without adjacent ownership, mode, or symlink checks; selected loader does not enforce those secret-file provenance controls.",
    ),
    "execution.py": (
        "REVIEWED_NARROW_PORT_COMPLETE_AUTHORITY_GAP_DEFERRED_NO_WHOLE_FILE_PORT",
        "1089747",
        "Selected source now strictly parses persisted venue compatibility/preflight flags. Older forensic/visual source omits later gate00g controls; runtime candidate adds a large coherent canonical-adapter attestation and advisory-only browser account-state contract. Keep completed strict-bool repair; review high-risk authority guards separately, not as a whole-file port.",
        "A confirmed-flat browser override can synthesize checked empty dYdX account state when live state is unchecked or blocked; adapter identity/callable checks also remain weaker than runtime candidate.",
    ),
    "family_matrix.py": (
        "REVIEWED_NARROW_PORT_COMPLETE_HISTORICAL_NO_WHOLE_FILE_PORT",
        "0311956",
        "Selected source retains atomic family report publication and strict persisted production/preferred eligibility parsing. Forensic_b replaces atomic writers with direct writes; visual_recovery also drops registry authority columns and uses bool('False'). No whole-file port.",
        "",
    ),
    "meta_learning.py": (
        "REVIEWED_MODEL_READINESS_REPAIR_CANDIDATE_NO_WHOLE_FILE_PORT",
        "",
        "Recovery/runtime revision separates finite normalized returns and holds verified readiness false but changes report schema and CLI expectations. Selected CLI P5 directly consumes ready_for_modeling. Selected file must retain current report/CLI contract while a focused outcome-proof and readiness repair is reviewed; older forensic/visual variant also reverts atomic publication.",
        "Selected outcome count can include raw currency pnl, ambiguous pnl_pct, duplicate/nonfinite/bool returns, and any nonempty exit-price paper row; it can mark modeling ready without verified outcomes.",
    ),
    "pair_detail_ingestion.py": (
        "REVIEWED_NARROW_PORT_COMPLETE_HISTORICAL_NO_WHOLE_FILE_PORT",
        "0311956",
        "Selected source retains atomic/promoted capture publication and now strictly parses persisted ECM, research usability, and execution-readiness flags. Forensic_b replaces atomic/promoted writes; visual_recovery also uses bool('False') for those guards. No whole-file port.",
        "",
    ),
    "research_ingestion.py": (
        "REVIEWED_NARROW_PORT_COMPLETE_HISTORICAL_NO_WHOLE_FILE_PORT",
        "543d415",
        "Selected source keeps atomic manifest/audit publication and the narrow AppleDouble sidecar exclusion from forensic_b. Forensic_b uses direct writes; visual_recovery lacks sidecar filtering and UnicodeDecodeError quarantine. No whole-file port.",
        "",
    ),
    "research_knowledge_store.py": (
        "REVIEWED_RAG_DEPENDENCY_DEFERRED_NO_WHOLE_FILE_PORT",
        "",
        "Recovery/runtime only add query_research_evidence and answer_research_question wrappers around research_rag.client.configured_rag, but selected tree has no research_rag package. Existing CLI/YouTube/Udemy consumers use selected CSV builder/query paths. Forensic/visual revert atomic CSV/Markdown publication. Defer RAG wrappers until the package and contract are selected.",
        "",
    ),
    "v2_run.py": (
        "REVIEWED_NARROW_PORT_COMPLETE_HISTORICAL_NO_WHOLE_FILE_PORT",
        "0311956",
        "Selected V2 source retains atomic pointer/publication controls and now strictly parses persisted readiness and authority booleans. Forensic_b replaces atomic publication; visual_recovery also uses truthiness on serialized flags. A remaining _control_plane_blockers permissive false-like field parse is a schema issue, but the pointer still forces live authority false. No whole-file port.",
        "",
    ),
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_bytes(revision: str, relative: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{revision}:{relative}"], cwd=ROOT)


def _git_csv(relative: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(_git_bytes(BASE_COMMIT, relative).decode("utf-8"))))


def _physical(path: Path) -> dict[str, object]:
    assert path.is_file(), f"missing preserved copy: {path}"
    data = path.read_bytes()
    return {"sha256": _sha(data), "size_bytes": len(data)}


def _top_defs(data: bytes, filename: str) -> list[str]:
    tree = ast.parse(data.decode("utf-8"), filename=filename)
    return sorted({node.name for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))})


def _direct_importers() -> dict[str, dict[str, list[str]]]:
    names = {f"quant_platform.{name[:-3]}": name for name in DECISIONS}
    result = {name: {"source": [], "scripts": [], "tests": []} for name in DECISIONS}
    tracked = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", BASE_COMMIT, "--", "src", "scripts", "tests"],
        cwd=ROOT,
        text=True,
    ).splitlines()
    for relative in sorted(path for path in tracked if path.endswith(".py")):
        kind = "source" if relative.startswith("src/") else "scripts" if relative.startswith("scripts/") else "tests"
        try:
            tree = ast.parse(_git_bytes(BASE_COMMIT, relative).decode("utf-8"), filename=relative)
        except (UnicodeDecodeError, SyntaxError):
            continue
        targets: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                targets.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    targets.add(node.module)
                    targets.update(f"{node.module}.{alias.name}" for alias in node.names)
        for module, name in names.items():
            if module in targets and relative != f"src/quant_platform/{name}":
                result[name][kind].append(relative)
    return result


def build() -> dict[str, object]:
    queue = {row["relative_path"]: (line, row) for line, row in enumerate(_git_csv(QUEUE.relative_to(ROOT).as_posix()), 2)}
    freeze = {row["relative_path"]: row for row in _git_csv(FREEZE.relative_to(ROOT).as_posix())}
    ext_by_path: dict[str, list[dict[str, str]]] = {}
    for row in _git_csv(EXTENDED.relative_to(ROOT).as_posix()):
        ext_by_path.setdefault(row["relative_path"], []).append(row)
    importers = _direct_importers()
    files = []
    for name, (status, completed_commit, rationale, risk) in DECISIONS.items():
        relative = f"src/quant_platform/{name}"
        line, q = queue[relative]
        assert q["custody_status"] == "PRESERVED_REVIEW_REQUIRED_NO_PORT"
        f = freeze[relative]
        selected_bytes = _git_bytes(BASE_COMMIT, relative)
        selected = {"sha256": _sha(selected_bytes), "size_bytes": len(selected_bytes)}
        assert f["working_sha256"] == _sha(_git_bytes(FREEZE_GIT_COMMIT, relative))
        if q["working_sha256_at_freeze"]:
            assert q["working_sha256_at_freeze"] == f["working_sha256"]
        if q["runtime_sha256_at_freeze"]:
            assert q["runtime_sha256_at_freeze"] == f["runtime_sha256"]
        recovery = _physical(RECOVERY / relative)
        runtime = _physical(RUNTIME / relative)
        assert recovery["sha256"] == f["recovery_sha256"]
        assert runtime["sha256"] == f["runtime_sha256"]
        if completed_commit:
            subprocess.check_call(["git", "merge-base", "--is-ancestor", completed_commit, BASE_COMMIT], cwd=ROOT)
            changed_paths = subprocess.check_output(["git", "diff-tree", "--no-commit-id", "--name-only", "-r", completed_commit], cwd=ROOT, text=True).splitlines()
            assert relative in changed_paths
        variants = []
        for row in sorted(ext_by_path.get(relative, []), key=lambda item: (item["root_label"], item["absolute_path"])):
            assert row["status"] == "hashed"
            physical = _physical(Path(row["absolute_path"]))
            assert physical["sha256"] == row["sha256"]
            assert physical["size_bytes"] == int(row["size_bytes"])
            variants.append({
                "root_label": row["root_label"],
                "relative_path": relative,
                "sha256": physical["sha256"],
                "size_bytes": physical["size_bytes"],
                "top_level_definition_count": len(_top_defs(Path(row["absolute_path"]).read_bytes(), relative)),
            })
        frozen_hashes = {f[key] for key in ("working_sha256", "recovery_sha256", "runtime_sha256") if f[key]}
        historical = {variant["sha256"] for variant in variants} - frozen_hashes
        assert historical == set(q["historical_variant_sha256"].split(";")) - {""}
        assert len(historical) == int(q["historical_variant_count"] or 0)
        files.append({
            "path": relative,
            "queue_line": line,
            "queue_status_at_review": q["custody_status"],
            "queue_working_vs_runtime": q["working_vs_runtime"],
            "queue_historical_variant_count": int(q["historical_variant_count"] or 0),
            "queue_historical_variant_sha256": sorted(historical),
            "selected_commit": BASE_COMMIT,
            "selected_sha256": selected["sha256"],
            "selected_changed_since_freeze": selected["sha256"] != f["working_sha256"],
            "selected_top_level_definitions": _top_defs(selected_bytes, relative),
            "frozen_working_sha256": f["working_sha256"],
            "frozen_working_git_provenance": FREEZE_GIT_COMMIT,
            "frozen_recovery_sha256": f["recovery_sha256"],
            "frozen_runtime_sha256": f["runtime_sha256"],
            "recovery_copy": {"root_label": "recovery", "relative_path": relative, **recovery},
            "runtime_copy": {"root_label": "runtime", "relative_path": relative, **runtime},
            "extended_copies": variants,
            "direct_importers": importers[name],
            "completed_narrow_port_commit": completed_commit,
            "recommended_queue_status": status,
            "rationale": rationale,
            "high_impact_remaining_risk": risk,
        })
    return {
        "report_type": "gate0_original_811_nonorchestration_source11",
        "base_commit": BASE_COMMIT,
        "freeze_git_commit": FREEZE_GIT_COMMIT,
        "queue_path": QUEUE.relative_to(ROOT).as_posix(),
        "freeze_path": FREEZE.relative_to(ROOT).as_posix(),
        "extended_manifest_path": EXTENDED.relative_to(ROOT).as_posix(),
        "selection_rule": "11 named original-811 pending non-orchestration source paths",
        "pending_count": len(files),
        "physical_copy_hashes_verified": sum(2 + len(item["extended_copies"]) for item in files),
        "status_counts": dict(sorted(Counter(item["recommended_queue_status"] for item in files).items())),
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    report = build()
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=("relative_path", "custody_status", "decision_rationale", "decision_evidence"), lineterminator="\n")
    writer.writeheader()
    for item in report["files"]:
        writer.writerow({
            "relative_path": item["path"],
            "custody_status": item["recommended_queue_status"],
            "decision_rationale": item["rationale"],
            "decision_evidence": "audit/GATE0_NONORCHESTRATION_SOURCE11_REVIEW_2026-09-30.json; exact selected/frozen/runtime/recovery/extended hashes",
        })
    proposed = output.getvalue()
    if args.verify:
        assert REPORT.read_text(encoding="utf-8") == rendered, "report differs from preserved evidence"
        assert PROPOSAL.read_text(encoding="utf-8") == proposed, "queue proposal differs from preserved evidence"
        print(f"verified {report['pending_count']} paths and {report['physical_copy_hashes_verified']} copy hashes")
    else:
        REPORT.write_text(rendered, encoding="utf-8")
        PROPOSAL.write_text(proposed, encoding="utf-8")
        print(f"wrote {report['pending_count']}-path report and queue proposal")


if __name__ == "__main__":
    main()
