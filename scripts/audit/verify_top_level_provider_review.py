"""Verify the frozen Gate 0 top-level Wizard/provider source decision without imports or effects.

Use --phase pre before queue reconciliation, or the default integrated phase after
cherry-picking the two narrow source fixes and correcting the nine queue rows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path

BASE_HEAD = "932108584a4ec4b5c9c5260fd310adb499e6600b"
REPORT_HASHES = {
    "combined": "fae7db7c1929914c44a7bbb876f13cc89d023fba4a236108d4bf8f81d2e364ff",
    "provider": "f0bdb7ac47f3266d083320c8b9edd7677350c838cf034a4f558a89aa862dfadb",
    "supplement": "b1219119e972bce202f56f48b8fd2db080e687be1b549aeabb561df2a4b0ded0",
    "all_variants": "301c74bf0e7d96bdffb4fb045dfee36219b568a76f37d2dfcf4eb4f6ec86597c",
}
NARROW_TEST_SHA256 = {
    "tests/test_wizard_research_journal.py": "d70314df668d170955d1603ee46b9fa09ef6bfe0b14883df22831f1720155b74",
    "tests/test_hyperliquid.py": "79087dcfc3ab8a0da100158115b8eb09c0f077ada793393f73056f4b3faa29f9",
}
HISTORICAL_ROOTS = frozenset({"forensic_a", "forensic_b", "visual_recovery", "workspace_outer"})
BASELINE_ROOTS = frozenset({"working_git", "recovery_project", "local_runtime", "source_candidate"})
PROVIDER_PATHS = frozenset(
    "src/quant_platform/" + name + ".py"
    for name in (
        "dydx_public_endpoints", "dydx_record_only_adapter", "dydx_sdk_order_adapter",
        "hyperliquid", "hyperliquid_testnet",
    )
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def base_bytes(root: Path, relative_path: str) -> bytes | None:
    result = subprocess.run(
        ["git", "show", f"{BASE_HEAD}:{relative_path}"],
        cwd=root, capture_output=True, check=False,
    )
    return result.stdout if result.returncode == 0 else None


def decision_bytes(root: Path, path: Path, commit: str | None) -> bytes | None:
    resolved = path.resolve()
    if commit is None or not resolved.is_relative_to(root):
        return resolved.read_bytes() if resolved.is_file() else None
    relative_path = resolved.relative_to(root).as_posix()
    result = subprocess.run(
        ["git", "show", f"{commit}:{relative_path}"],
        cwd=root, capture_output=True, check=False,
    )
    return result.stdout if result.returncode == 0 else None


def safe_relative_path(value: str, *, prefix: str) -> bool:
    path = Path(value)
    return (
        not path.is_absolute() and ".." not in path.parts
        and value.startswith(prefix) and value.endswith(".py")
    )


def expected_historical_metadata(row: dict) -> dict[str, str]:
    baseline = {row["active_base_sha256"], row["runtime_current_sha256"]}
    baseline.update(
        copy["frozen_sha256"] for copy in row["frozen_copies"]
        if copy["root"] in BASELINE_ROOTS
    )
    historical = sorted({
        copy["frozen_sha256"] for copy in row["frozen_copies"]
        if copy["root"] in HISTORICAL_ROOTS and copy["frozen_sha256"] not in baseline
    })
    roots = sorted({
        copy["root"] for copy in row["frozen_copies"]
        if copy["frozen_sha256"] in historical
    })
    return {
        "historical_variant_count": str(len(historical)),
        "historical_variant_sha256": ";".join(historical),
        "historical_copy_roots": ";".join(roots),
    }


def verify(args: argparse.Namespace) -> tuple[dict, list[str]]:
    errors: list[str] = []
    verified_copies = 0
    verified_test_copies = 0
    verified_provider_copies = 0
    verified_provider_test_copies = 0
    metadata_pending: list[str] = []
    cache: dict[str, str] = {}

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    def hash_path(path: Path, *, decision: bool = False) -> str | None:
        key = ("decision:" if decision else "copy:") + str(path)
        if key not in cache:
            if decision:
                content = decision_bytes(args.root, path, args.decision_commit)
                cache[key] = sha256_bytes(content) if content is not None else ""
            else:
                cache[key] = sha256_file(path) if path.is_file() else ""
        return cache[key] or None

    report_paths = {
        "combined": args.combined or args.root / "audit/GATE0_TOP_LEVEL_PROVIDER_SOURCE_REVIEW_2026-09-30.json",
        "provider": args.provider or args.root / "audit/GATE0_PROVIDER_ADAPTER_SOURCE_REVIEW_2026-09-30.json",
        "supplement": args.supplement or args.root / "audit/GATE0_TOP_LEVEL_NINE_UNLISTED_VARIANTS_SUPPLEMENT_2026-09-30.json",
        "all_variants": args.all_variants or args.root / "audit/GATE0_TOP_LEVEL_ALL_EXTENDED_VARIANTS_RECONCILIATION_2026-09-30.json",
    }
    reports = {}
    for name, path in report_paths.items():
        # This reconciliation report is created after the frozen decision commit;
        # its pinned hash is checked from the supplied file.
        in_decision = name != "all_variants"
        if hash_path(path, decision=in_decision) != REPORT_HASHES[name]:
            errors.append(f"{name} report missing or SHA-256 differs: {path}")
            continue
        content = (
            decision_bytes(args.root, path, args.decision_commit)
            if in_decision else path.read_bytes()
        )
        reports[name] = json.loads(content.decode("utf-8"))
    if len(reports) != 4:
        return {"phase": args.phase, "report_paths": {k: str(v) for k, v in report_paths.items()}}, errors

    combined, provider, supplement, all_variants = (
        reports[name] for name in ("combined", "provider", "supplement", "all_variants")
    )
    require(combined.get("canonical_head") == BASE_HEAD, "combined base HEAD differs")
    require(provider.get("canonical_head_before_fix") == BASE_HEAD, "provider base HEAD differs")
    require(supplement.get("canonical_base_head") == BASE_HEAD, "supplement base HEAD differs")
    require(all_variants.get("canonical_head") == BASE_HEAD, "all-variants base HEAD differs")
    require(combined.get("provider_subreview", {}).get("report_sha256") == REPORT_HASHES["provider"], "provider link hash differs")
    require(supplement.get("combined_review_sha256") == REPORT_HASHES["combined"], "supplement combined link hash differs")
    require(all_variants.get("combined_review", {}).get("sha256") == REPORT_HASHES["combined"], "all-variants combined link hash differs")
    require(all_variants.get("nine_variant_supplement", {}).get("sha256") == REPORT_HASHES["supplement"], "all-variants supplement link hash differs")

    source_rows = combined.get("rows", [])
    source_by_path = {row["relative_path"]: row for row in source_rows}
    require(len(source_rows) == len(source_by_path) == 22, "combined report must contain 22 unique source paths")
    require(all(safe_relative_path(path, prefix="src/quant_platform/") for path in source_by_path), "invalid source path")
    provider_rows = provider.get("source_rows", [])
    provider_by_path = {row["relative_path"]: row for row in provider_rows}
    require(set(provider_by_path) == PROVIDER_PATHS and len(provider_rows) == 5, "provider subreview must cover exact five paths")
    require(PROVIDER_PATHS.issubset(source_by_path), "provider paths missing from combined report")
    supplement_rows = supplement.get("rows", [])
    supplement_by_path = {row["relative_path"]: row for row in supplement_rows}
    expected_omitted = {
        path for path, row in source_by_path.items()
        if row["frozen_variants_unlisted_in_queue_historical_field"]
    }
    require(set(supplement_by_path) == expected_omitted and len(supplement_rows) == 9, "supplement must cover exact nine omitted variants")
    all_rows = all_variants.get("rows", [])
    all_by_path = {row["relative_path"]: row for row in all_rows}
    require(set(all_by_path) == set(source_by_path) and len(all_rows) == 22, "all-variants report must cover exact 22 paths")

    queue_path = args.queue or args.root / "audit/GATE0_UNION_SOURCE_QUEUE_2026-09-30.csv"
    queue_data = decision_bytes(args.root, queue_path, args.decision_commit)
    if queue_data is None:
        errors.append(f"queue missing from decision commit or filesystem: {queue_path}")
        return {"phase": args.phase, "decision_commit": args.decision_commit}, errors
    queue = {row["relative_path"]: row for row in csv.DictReader(io.StringIO(queue_data.decode("utf-8")))}
    reconciliation_info = all_variants["source_reconciliation"]
    reconciliation_path = Path(reconciliation_info["path"])
    require(hash_path(reconciliation_path) == reconciliation_info["sha256"], "four-root source reconciliation SHA differs")
    with reconciliation_path.open(newline="", encoding="utf-8") as source:
        reconciled = {row["relative_path"]: row for row in csv.DictReader(source)}
    manifest_pairs: dict[str, list[tuple[str, str]]] = {path: [] for path in source_by_path}
    for label in ("initial_four_root_manifest", "extended_manifest"):
        info = all_variants[label]
        path = Path(info["path"])
        require(hash_path(path) == info["sha256"], f"{label} SHA differs")
        with path.open(newline="", encoding="utf-8") as source:
            for copy in csv.DictReader(source):
                if copy["relative_path"] in source_by_path and copy.get("sha256"):
                    manifest_pairs[copy["relative_path"]].append((copy["root_label"], copy["sha256"]))
    test_by_path: dict[str, dict] = {}
    for path, row in sorted(source_by_path.items()):
        if not safe_relative_path(path, prefix="src/quant_platform/"):
            continue
        base = base_bytes(args.root, path)
        reported_base = row["active_base_sha256"]
        require((sha256_bytes(base) if base is not None else None) == reported_base, f"base source hash differs: {path}")
        four_root = reconciled.get(path)
        if four_root is None:
            errors.append(f"four-root source reconciliation missing: {path}")
            continue
        require(four_root["working_sha256"] == (reported_base or ""), f"four-root working hash differs: {path}")
        require(four_root["runtime_sha256"] == row["runtime_current_sha256"], f"four-root runtime hash differs: {path}")
        require(
            {copy["frozen_sha256"] for copy in row["frozen_copies"] if copy["root"] == "recovery_project"}
            == {four_root["recovery_sha256"]},
            f"four-root recovery hash differs: {path}",
        )
        require(
            sorted(manifest_pairs[path])
            == sorted((copy["root"], copy["frozen_sha256"]) for copy in row["frozen_copies"]),
            f"all extended/four-root manifest copies not represented: {path}",
        )
        candidate = row.get("candidate_sha256_after_narrow_port")
        current = hash_path(args.root / path, decision=True)
        allowed = {candidate} if args.phase == "integrated" and candidate else {reported_base}
        if args.phase == "pre" and candidate:
            allowed.add(candidate)
        require(current in allowed, f"current source hash outside selected base/candidate: {path}")
        for copy in row["frozen_copies"]:
            expected = copy["frozen_sha256"]
            observed = (sha256_bytes(base) if base is not None else None) if copy["root"] == "working_git" else hash_path(Path(copy["absolute_path"]))
            require(observed == expected, f"frozen source copy differs: {path} {copy['root']}")
            verified_copies += 1
        for test in row["direct_pending_tests"]:
            test_path = test["path"]
            require(safe_relative_path(test_path, prefix="tests/"), f"invalid test path: {test_path}")
            previous = test_by_path.setdefault(test_path, test)
            require(previous["active_sha256"] == test["active_sha256"], f"conflicting test baseline: {test_path}")
        expected_meta = expected_historical_metadata(row)
        all_row = all_by_path[path]
        require(all_row["unclassified_sha256"] == [], f"all-variants report retains unclassified SHA: {path}")
        require(all_row["corrected_historical_sha256"] == sorted(filter(None, expected_meta["historical_variant_sha256"].split(";"))), f"all-variants corrected historical set differs: {path}")
        grouped: dict[str, list[str]] = {}
        for root_label, variant_sha in manifest_pairs[path]:
            grouped.setdefault(variant_sha, []).append(root_label)
        recorded_variants = {variant["sha256"]: variant for variant in all_row["distinct_frozen_variants"]}
        require(set(recorded_variants) == set(grouped), f"all-variants SHA census differs: {path}")
        baseline_roles: dict[str, list[str]] = {}
        for role, field in (("working", "working_sha256"), ("recovery", "recovery_sha256"), ("runtime", "runtime_sha256")):
            value = four_root[field]
            if value:
                baseline_roles.setdefault(value, []).append(role)
        old_historical = set(filter(None, all_row["current_queue_historical_sha256"]))
        corrected_historical = set(all_row["corrected_historical_sha256"])
        for variant_sha, roots in grouped.items():
            recorded = recorded_variants[variant_sha]
            if variant_sha in baseline_roles:
                kind = "selected_four_root_baseline"
            elif variant_sha in old_historical:
                kind = "historical_already_queued"
            elif variant_sha in corrected_historical:
                kind = "historical_metadata_correction"
            else:
                kind = "unclassified"
            require(kind != "unclassified", f"extended variant unclassified: {path} {variant_sha}")
            require(recorded["coverage_kind"] == kind, f"all-variants classification differs: {path} {variant_sha}")
            require(recorded["copy_roots"] == sorted(roots), f"all-variants copy roots differ: {path} {variant_sha}")
            require(recorded["baseline_roles"] == baseline_roles.get(variant_sha, []), f"all-variants baseline roles differ: {path} {variant_sha}")
        if path in supplement_by_path:
            supp = supplement_by_path[path]
            require(supp["queue_metadata_replacement"] == expected_meta, f"supplement metadata differs: {path}")
            require(supp["active_selected_sha256"] == reported_base and supp["runtime_selected_sha256"] == row["runtime_current_sha256"], f"supplement selected source hash differs: {path}")
            require(supp["omitted_variant_sha256"] in row["frozen_variants_unlisted_in_queue_historical_field"], f"supplement variant hash differs: {path}")
            require(supp.get("decision") == "NO_PORT_OMITTED_VARIANT", f"supplement no-port decision differs: {path}")
        queued = queue.get(path)
        if queued is None:
            errors.append(f"queue missing source row: {path}")
            continue
        got = {key: queued[key] for key in expected_meta}
        baseline_known_in_queue = set(filter(None, (
            queued["working_sha256_at_freeze"],
            queued["recovery_sha256_at_freeze"],
            queued["runtime_sha256_at_freeze"],
        ))) | old_historical
        expected_baseline_absent = sorted(set(baseline_roles) - baseline_known_in_queue)
        require(
            all_row["baseline_sha256_absent_from_queue_hash_fields_but_covered_by_source_reconciliation"]
            == expected_baseline_absent,
            f"all-variants baseline coverage differs: {path}",
        )
        if got != expected_meta:
            metadata_pending.append(path)
            if args.phase == "integrated" or path not in supplement_by_path or got != supplement_by_path[path]["queue_metadata_old"]:
                errors.append(f"queue historical metadata differs: {path}")
        if args.phase == "integrated":
            require(queued["custody_status"] != "PRESERVED_REVIEW_REQUIRED_NO_PORT", f"queue source remains pending: {path}")
            require(bool(queued["decision_rationale"].strip()) and bool(queued["decision_evidence"].strip()), f"queue decision/evidence missing: {path}")

    for path, test in sorted(test_by_path.items()):
        base = base_bytes(args.root, path)
        require(base is not None and sha256_bytes(base) == test["active_sha256"], f"base test hash differs: {path}")
        expected_current = NARROW_TEST_SHA256.get(path, test["active_sha256"])
        allowed = {expected_current} if args.phase == "integrated" else {test["active_sha256"], expected_current}
        require(hash_path(args.root / path, decision=True) in allowed, f"current direct test hash differs: {path}")
        for copy in test["frozen_copies"]:
            observed = sha256_bytes(base) if copy["root"] == "working_git" else hash_path(Path(copy["absolute_path"]))
            require(observed == copy["frozen_sha256"], f"frozen test copy differs: {path} {copy['root']}")
            verified_test_copies += 1

    for path, row in sorted(provider_by_path.items()):
        combined_row = source_by_path[path]
        require(row["all_frozen_copy_hashes_match_manifest"] is True, f"provider frozen manifest failed: {path}")
        for copy in row["copies"]:
            if copy["root"] == "working_git":
                base = base_bytes(args.root, path)
                observed = sha256_bytes(base) if base is not None else None
            else:
                observed = hash_path(Path(copy["path"]))
            require(observed == copy["sha256"], f"provider copy differs: {path} {copy['root']}")
            if copy.get("frozen_manifest_sha256"):
                require(observed == copy["frozen_manifest_sha256"], f"provider frozen manifest hash differs: {path} {copy['root']}")
            verified_provider_copies += 1
        require(
            {copy["sha256"] for copy in row["copies"] if copy["root"] == "local_runtime"} == {combined_row["runtime_current_sha256"]},
            f"provider runtime hash disagrees with combined: {path}",
        )

    provider_tests = provider.get("pending_test_rows", [])
    require(len(provider_tests) == len({row["relative_path"] for row in provider_tests}) == 8, "provider subreview must cover eight unique related tests")
    for row in provider_tests:
        path = row["relative_path"]
        require(safe_relative_path(path, prefix="tests/"), f"invalid provider test path: {path}")
        base = base_bytes(args.root, path)
        for copy in row["copies"]:
            if copy["root"] == "working_git":
                observed = sha256_bytes(base) if base is not None else None
            else:
                observed = hash_path(Path(copy["path"]))
            require(observed == copy["sha256"], f"provider test copy differs: {path} {copy['root']}")
            if copy.get("frozen_manifest_sha256"):
                require(observed == copy["frozen_manifest_sha256"], f"provider test frozen manifest differs: {path} {copy['root']}")
            verified_provider_test_copies += 1

    for path, row in sorted(supplement_by_path.items()):
        for copy in row["omitted_variant_copies"]:
            require(hash_path(Path(copy["absolute_path"])) == row["omitted_variant_sha256"], f"omitted variant copy differs: {path} {copy['root']}")
    all_summary = all_variants["summary"]
    require(all_summary["source_paths"] == 22, "all-variants source count differs")
    require(all_summary["frozen_copy_entries"] == sum(len(pairs) for pairs in manifest_pairs.values()), "all-variants copy count differs")
    require(all_summary["distinct_path_sha_variants"] == sum(len(row["distinct_frozen_variants"]) for row in all_rows), "all-variants distinct SHA count differs")
    require(all_summary["baseline_path_sha_absent_from_queue_hash_fields_but_covered_by_source_reconciliation"] == 9, "all-variants baseline-only queue omission count differs")
    require(all_summary["historical_metadata_correction_path_sha"] == 9, "all-variants historical correction count differs")
    require(all_summary["unclassified_path_sha"] == 0, "all-variants unclassified count differs")
    summary = {
        "phase": args.phase,
        "decision_commit": args.decision_commit,
        "base_head": BASE_HEAD,
        "source_rows": len(source_rows),
        "provider_rows": len(provider_rows),
        "omitted_variant_rows": len(supplement_rows),
        "unique_direct_pending_tests": len(test_by_path),
        "verified_combined_source_copies": verified_copies,
        "verified_direct_test_copies": verified_test_copies,
        "verified_provider_subreview_copies": verified_provider_copies,
        "verified_provider_subreview_test_copies": verified_provider_test_copies,
        "all_variants_distinct_path_sha": sum(len(row["distinct_frozen_variants"]) for row in all_rows),
        "queue_metadata_pending_paths": metadata_pending,
        "error_count": len(errors),
    }
    return summary, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Git repository and queue root")
    parser.add_argument("--phase", choices=("pre", "integrated"), default="integrated")
    parser.add_argument("--queue", type=Path)
    parser.add_argument("--combined", type=Path)
    parser.add_argument("--provider", type=Path)
    parser.add_argument("--supplement", type=Path)
    parser.add_argument("--all-variants", type=Path)
    parser.add_argument("--decision-commit", help="Read queue, reports, source, and tests from this exact Git commit")
    args = parser.parse_args()
    args.root = args.root.resolve()
    summary, errors = verify(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    for error in errors:
        print("FAIL: " + error, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
