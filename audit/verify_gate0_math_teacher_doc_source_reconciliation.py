#!/usr/bin/env python3
"""Verify selected math and teacher documentation against preserved variants."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELECTED_COMMIT = "785b8da316cae8541922b2b51a5738de35cbd179"
QUEUE_COMMIT = "edad2395346733e7269a21301708cfee603bd219"
QUEUE = ROOT / "audit/GATE0_UNION_SOURCE_QUEUE_2026-09-30.csv"
OUTPUT = ROOT / "audit/GATE0_MATH_TEACHER_DOC_SOURCE_RECONCILIATION_2026-09-30.json"
SAVEPOINT = Path("/Users/gregc/Backups/TheWiz/savepoints/2026-09-30/local_runtime")
VARIANTS = Path("/Users/gregc/Backups/TheWiz/source-delta-evidence/2026-09-30/variants")
SELECTED = {
    "docs/formula_dictionary.md": "3e98f9ce7f1d4639d79afc93e67cfe0836e00785757120d5ab38913b59c3da42",
    "docs/math_v2_teacher_adapter.md": "2f95f7363f0d2e62a4c328e578b390386970e430286d59b843c17ef4797efb03",
    "docs/quant_brain.md": "9db366df5a772c9eb814ca3a528084d31c591e63be81c8edaecdda82e6c131c0",
    "docs/teacher_council_student_stack.md": "5a6248fce2d3f81813ad13a8bd33c1bced3b4345881c595156077795d408ec88",
    "docs/v2_math_validation_contract.md": "758d25c12788a7a87fd6c3fc680b88f4a1b78d257bdb4ebb83dd322612191529",
}
OLD_RUNTIME = {
    "docs/formula_dictionary.md": "583d9329262ef0ef95455ff40c552e1902951b82e449a814f3df5f47f132c55b",
    "docs/math_v2_teacher_adapter.md": "a96d78cdcd8bbe87a1236a7a984b88f40477925b9a49e0ea3ea826902771bb1a",
    "docs/teacher_council_student_stack.md": "db336c9854ea7d705a86428178d043751e13e2e98d0347a92a1d325ff8741389",
}
OLD_VARIANTS = {
    "docs/formula_dictionary.md": "694491ac6a3635f27939c34b5c8577f3e75157b887ab4623ce8a057400abb12c",
    "docs/quant_brain.md": "d4232479fbef6a1faf233eca318689f398a797a2137b7d7a46da5bfb3531c041",
}
QUEUE_SHA256 = "4b2ec512bfffd5cca48f2ee5b694aaa18cbb894e1d8d16451bff140e9ce0a31b"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def selected_bytes(relative: str, commit: str = SELECTED_COMMIT) -> bytes:
    return subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    ).stdout


def main() -> None:
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", SELECTED_COMMIT, "HEAD"],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", QUEUE_COMMIT, "HEAD"],
        cwd=ROOT,
        check=True,
    )
    docs = {}
    for relative, expected in SELECTED.items():
        content = selected_bytes(relative)
        if sha256(content) != expected:
            raise ValueError(f"selected math/teacher document drift: {relative}")
        docs[relative] = content.decode("utf-8")
    for relative, expected in OLD_RUNTIME.items():
        path = SAVEPOINT / relative
        if not path.is_file() or sha256(path.read_bytes()) != expected:
            raise ValueError(f"historical runtime document drift: {relative}")
    for relative, expected in OLD_VARIANTS.items():
        path = VARIANTS / expected
        if not path.is_file() or sha256(path.read_bytes()) != expected:
            raise ValueError(f"historical document variant drift: {relative}")

    formula = docs["docs/formula_dictionary.md"]
    adapter = docs["docs/math_v2_teacher_adapter.md"]
    brain = docs["docs/quant_brain.md"]
    council = docs["docs/teacher_council_student_stack.md"]
    contract = docs["docs/v2_math_validation_contract.md"]
    if not all(term in formula for term in ("detrended fluctuation analysis", "delta_t", "does not measure coefficient magnitude")):
        raise ValueError("local formula definitions changed")
    if not all(term in adapter for term in (
        "math-v2.3-venue-clock-execution", "15", "8", "UV_PROJECT_ENVIRONMENT",
        "five complete contexts", "uv run --locked python",
    )) or ".venv312" in adapter:
        raise ValueError("teacher adapter source/marker/operator contract changed")
    if "does not measure coefficient magnitude" not in brain:
        raise ValueError("ECM strength documentation changed")
    if ("do not certify parity" not in council
            or "UV_PROJECT_ENVIRONMENT" not in council
            or "uv run --locked python" not in council
            or ".venv312" in council):
        raise ValueError("teacher council parity or operator contract changed")
    if ("same complete" not in contract
            or "BLOCKED_PENDING_CONTROLLED_REGENERATION" not in contract
            or "neither receipt" not in contract):
        raise ValueError("Math V2 historical/current authority boundary changed")

    queue_bytes = selected_bytes(QUEUE.relative_to(ROOT).as_posix(), QUEUE_COMMIT)
    if sha256(queue_bytes) != QUEUE_SHA256:
        raise ValueError("union source queue drift")
    rows = list(csv.DictReader(queue_bytes.decode("utf-8").splitlines()))
    if len(rows) != 811 or len({row["relative_path"] for row in rows}) != 811:
        raise ValueError("union source queue changed")
    queue = {row["relative_path"]: row for row in rows}
    for relative in SELECTED:
        row = queue[relative]
        if (row["custody_status"] != "REVIEWED_MATH_TEACHER_DOC_SELECTED"
                or OUTPUT.name not in row["decision_evidence"]):
            raise ValueError(f"math/teacher documentation decision missing: {relative}")
    pending = sum(row["custody_status"] in {
        "PRESERVED_REVIEW_REQUIRED_NO_PORT",
        "PORTED_V23_PARTIAL_HISTORICAL_REVIEW_REQUIRED",
    } for row in rows)
    reviewed = sum(row["custody_status"].startswith("REVIEWED_") for row in rows)
    if (pending, reviewed, len(rows) - pending - reviewed) != (626, 167, 18):
        raise ValueError("source queue accounting changed")

    report = {
        "schema_version": "thewiz.gate0.math_teacher_doc_source_reconciliation.v1",
        "decision": "FIVE_MATH_TEACHER_DOCS_SELECTED_WITH_HISTORICAL_ALTERNATIVES_NO_PORT",
        "selected_commit": SELECTED_COMMIT,
        "selected_sha256": SELECTED,
        "historical_runtime_sha256": OLD_RUNTIME,
        "historical_variant_sha256": OLD_VARIANTS,
        "operator_help_probes": ["quant_platform.orchestration.dynamic_cli", "quant_platform.cli"],
        "union_source_queue": {
            "sha256": QUEUE_SHA256,
            "distinct_paths": len(rows),
            "reviewed_paths": reviewed,
            "nonsource_paths": len(rows) - pending - reviewed,
            "remaining_semantic_review_paths": pending,
        },
        "research_acceptance": "BLOCKED",
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("PASS math_teacher_doc_source_reconciliation")


if __name__ == "__main__":
    main()
