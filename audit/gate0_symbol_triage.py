"""List top-level Python symbols found only in historical source variants.

This is a priority aid, not a semantic equivalence proof or a port decision.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTER = ROOT / "audit/GATE0_SOURCE_DELTA_REGISTER_2026-09-30.csv"
ROOTS = ROOT / "audit/SECOND_PASS_EXTENDED_MANIFEST_SUMMARY_2026-09-29.json"
OUTPUT = ROOT / "audit/GATE0_SYMBOL_TRIAGE_2026-09-30.csv"
SUMMARY = ROOT / "audit/GATE0_SYMBOL_TRIAGE_SUMMARY_2026-09-30.json"


def symbols(path: Path) -> set[str] | None:
    try:
        tree = ast.parse(path.read_text())
    except (OSError, UnicodeError, SyntaxError):
        return None
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def main() -> None:
    roots = {
        key: Path(value)
        for key, value in json.loads(ROOTS.read_text())["roots"].items()
    }
    with REGISTER.open(newline="") as handle:
        variants = list(csv.DictReader(handle))
    rows = []
    for variant in variants:
        path = variant["relative_path"]
        if (
            variant["decision"] != "SEMANTIC_REVIEW_REQUIRED_NO_PORT"
            or not path.endswith(".py")
        ):
            continue
        current = symbols(ROOT / path)
        for name in variant["copy_roots"].split(";"):
            historical_path = roots[name] / path
            if not historical_path.is_file():
                continue
            contents = historical_path.read_bytes()
            if hashlib.sha256(contents).hexdigest() != variant["copy_sha256"]:
                continue
            historical = symbols(historical_path)
            if historical is None:
                continue
            older_only = sorted(historical - (current or set()))
            active_only = sorted((current or set()) - historical)
            rows.append({
                "relative_path": path,
                "historical_root": name,
                "copy_sha256": variant["copy_sha256"],
                "active_path_present": (ROOT / path).is_file(),
                "historical_only_count": len(older_only),
                "historical_only_symbols": ";".join(older_only),
                "active_only_count": len(active_only),
                "active_only_symbols": ";".join(active_only),
                "decision": "SEMANTIC_REVIEW_REQUIRED_NO_PORT",
            })
            break
    with OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema_version": "thewiz.gate0.symbol_triage.v1",
        "python_variants_parsed": len(rows),
        "variants_with_historical_only_top_level_symbols": sum(
            int(row["historical_only_count"] > 0) for row in rows
        ),
        "historical_only_by_root": dict(sorted(Counter(
            row["historical_root"]
            for row in rows
            if row["historical_only_count"] > 0
        ).items())),
        "limitation": "Top-level AST names do not compare bodies, imports, nested functions, behavior, or safety. Every flagged variant remains review-required.",
    }
    SUMMARY.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
