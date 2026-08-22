"""Migrate registered Phase 00 rename promotions to the fenced store boundary."""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path

from quant_platform.orchestration.corrective_publication_registry import (
    publication_surface_rows,
)

IMPORT_LINE = (
    "from quant_platform.orchestration.corrective_runtime "
    "import promote_staged_file\n"
)
BLOCKING_SCOPES = {"execution_blocking", "phase00_blocking"}


@dataclass(frozen=True)
class Replacement:
    start: int
    end: int
    text: str


def migrate(
    root: Path,
    *,
    apply: bool,
    all_scopes: bool = False,
    excluded_paths: frozenset[str] = frozenset(),
) -> dict[str, int]:
    root = root.resolve()
    rows = [
        row
        for row in publication_surface_rows(root)
        if (all_scopes or row["authority_scope"] in BLOCKING_SCOPES)
        and row["migration_state"] == "UNMIGRATED"
        and row["call_kind"] in {"rename", "replace"}
        and row["source_path"] not in excluded_paths
    ]
    by_path: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_path.setdefault(str(row["source_path"]), []).append(row)

    changed_files = 0
    changed_calls = 0
    for relative, expected in sorted(by_path.items()):
        path = root / relative
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        offsets = _line_offsets(source)
        expected_by_line = {int(row["line"]): row for row in expected}
        if len(expected_by_line) != len(expected):
            raise RuntimeError(f"ambiguous publication rows in {relative}")
        replacements: list[Replacement] = []
        matched_lines: set[int] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or node.lineno not in expected_by_line:
                continue
            replacement = _promotion_replacement(source, offsets, node)
            if replacement is None:
                continue
            replacements.append(replacement)
            matched_lines.add(node.lineno)
        missing = sorted(set(expected_by_line) - matched_lines)
        if missing:
            raise RuntimeError(f"unmatched publication rows in {relative}: {missing}")
        updated = source
        for replacement in sorted(replacements, key=lambda item: item.start, reverse=True):
            updated = (
                updated[: replacement.start]
                + replacement.text
                + updated[replacement.end :]
            )
        if "import promote_staged_file" not in updated:
            updated = _insert_import(updated)
        ast.parse(updated, filename=str(path))
        if apply:
            path.write_text(updated, encoding="utf-8")
        changed_files += 1
        changed_calls += len(replacements)
    return {
        "registered_calls": len(rows),
        "changed_calls": changed_calls,
        "changed_files": changed_files,
    }


def _promotion_replacement(
    source: str,
    offsets: list[int],
    node: ast.Call,
) -> Replacement | None:
    if not isinstance(node.func, ast.Attribute):
        return None
    if node.func.attr not in {"rename", "replace"}:
        return None
    call_name = _call_name(node.func)
    if call_name in {"os.rename", "os.replace"}:
        if len(node.args) != 2 or node.keywords:
            raise RuntimeError(f"unsupported os promotion at line {node.lineno}")
        source_expr = ast.get_source_segment(source, node.args[0])
        target_expr = ast.get_source_segment(source, node.args[1])
    else:
        if len(node.args) != 1 or node.keywords:
            raise RuntimeError(f"unsupported path promotion at line {node.lineno}")
        source_expr = ast.get_source_segment(source, node.func.value)
        target_expr = ast.get_source_segment(source, node.args[0])
    if source_expr is None or target_expr is None:
        raise RuntimeError(f"missing promotion source at line {node.lineno}")
    start = offsets[node.lineno - 1] + node.col_offset
    end = offsets[node.end_lineno - 1] + node.end_col_offset
    return Replacement(
        start=start,
        end=end,
        text=f"promote_staged_file({source_expr}, {target_expr})",
    )


def _insert_import(source: str) -> str:
    tree = ast.parse(source)
    future = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.ImportFrom) and node.module == "__future__"
        ),
        None,
    )
    if future is None or future.end_lineno is None:
        raise RuntimeError("module has no future import insertion point")
    lines = source.splitlines(keepends=True)
    lines.insert(future.end_lineno, "\n" + IMPORT_LINE)
    return "".join(lines)


def _line_offsets(source: str) -> list[int]:
    offsets = [0]
    for line in source.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


def _call_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--all-scopes", action="store_true")
    parser.add_argument("--exclude", action="append", default=[])
    args = parser.parse_args()
    result = migrate(
        args.root,
        apply=args.apply,
        all_scopes=bool(args.all_scopes),
        excluded_paths=frozenset(args.exclude),
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
