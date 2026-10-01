"""Inventory changed Python symbols across the active and LocalRuntime copies.

This is a syntactic review aid. It cannot establish behavioral equivalence or
that any historical implementation is safe to port.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path


AUDIT = Path(__file__).resolve().parent
ACTIVE = AUDIT.parent
RUNTIME = Path("/Users/gregc/TheWiz-LocalRuntime")
SOURCE = AUDIT / "evidence_freeze_2026-09-29" / "source_reconciliation.csv"
OUTPUT = AUDIT / "GATE0_RUNTIME_SYMBOL_TRIAGE_2026-09-30.csv"
SUMMARY = AUDIT / "GATE0_RUNTIME_SYMBOL_TRIAGE_2026-09-30.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def symbols(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    result = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            result.add(node.name)
            if isinstance(node, ast.ClassDef):
                result.update(
                    f"{node.name}.{member.name}"
                    for member in node.body
                    if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
                )
    return result


def main() -> None:
    with SOURCE.open(newline="", encoding="utf-8") as stream:
        candidates = {
            row["relative_path"]: row
            for row in csv.DictReader(stream)
            if row["working_vs_runtime"] == "CHANGED"
            and row["relative_path"].endswith(".py")
        }
    rows = []
    for relative in sorted(candidates):
        active = ACTIVE / relative
        runtime = RUNTIME / relative
        active_symbols = symbols(active)
        runtime_symbols = symbols(runtime)
        active_hash = digest(active)
        runtime_hash = digest(runtime)
        rows.append(
            {
                "relative_path": relative,
                "active_current_sha256": active_hash,
                "runtime_current_sha256": runtime_hash,
                "active_differs_from_freeze": active_hash != candidates[relative]["working_sha256"],
                "runtime_differs_from_freeze": runtime_hash != candidates[relative]["runtime_sha256"],
                "active_symbol_count": len(active_symbols),
                "runtime_symbol_count": len(runtime_symbols),
                "active_only_symbols": ";".join(sorted(active_symbols - runtime_symbols)),
                "runtime_only_symbols": ";".join(sorted(runtime_symbols - active_symbols)),
                "decision": "SEMANTIC_REVIEW_REQUIRED_NO_PORT",
            }
        )
    fields = list(rows[0])
    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema_version": "thewiz.gate0.runtime_symbol_triage.v1",
        "source": str(SOURCE.relative_to(AUDIT)),
        "active_root": str(ACTIVE),
        "runtime_root": str(RUNTIME),
        "changed_python_paths": len(rows),
        "active_paths_changed_since_freeze": sum(r["active_differs_from_freeze"] for r in rows),
        "runtime_paths_changed_since_freeze": sum(r["runtime_differs_from_freeze"] for r in rows),
        "paths_with_runtime_only_symbols": sum(bool(r["runtime_only_symbols"]) for r in rows),
        "paths_with_active_only_symbols": sum(bool(r["active_only_symbols"]) for r in rows),
        "runtime_only_symbol_count": sum(
            len(r["runtime_only_symbols"].split(";"))
            for r in rows if r["runtime_only_symbols"]
        ),
        "active_only_symbol_count": sum(
            len(r["active_only_symbols"].split(";"))
            for r in rows if r["active_only_symbols"]
        ),
        "meaning": "AST surface comparison only; semantic review remains open.",
    }
    SUMMARY.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
