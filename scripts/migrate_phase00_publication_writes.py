"""Migrate registered direct Phase 00 writes to fenced store boundaries."""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path

from quant_platform.orchestration.corrective_publication_registry import (
    publication_surface_rows,
)

BLOCKING_SCOPES = {"execution_blocking", "phase00_blocking"}
SUPPORTED_KINDS = {"to_csv", "write_bytes", "write_text"}


@dataclass(frozen=True)
class Replacement:
    start: int
    end: int
    text: str
    import_name: str


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
        and row["call_kind"] in SUPPORTED_KINDS
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
            row = expected_by_line.get(getattr(node, "lineno", -1))
            if not isinstance(node, ast.Call) or row is None:
                continue
            if _call_name(node.func) != row["call"]:
                continue
            replacement = _write_replacement(
                source,
                offsets,
                node,
                call_kind=str(row["call_kind"]),
            )
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
        for import_name in sorted({item.import_name for item in replacements}):
            if f"import {import_name}" not in updated:
                updated = _insert_import(updated, import_name)
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


def _write_replacement(
    source: str,
    offsets: list[int],
    node: ast.Call,
    *,
    call_kind: str,
) -> Replacement:
    if not isinstance(node.func, ast.Attribute):
        raise TypeError(f"unsupported direct write at line {node.lineno}")
    receiver = ast.get_source_segment(source, node.func.value)
    if receiver is None:
        raise RuntimeError(f"missing write receiver at line {node.lineno}")
    if call_kind == "to_csv":
        replacement, import_name = _csv_replacement(source, node, receiver)
    elif call_kind == "write_text":
        replacement, import_name = _path_write_replacement(
            source,
            node,
            receiver,
            binary=False,
        )
    elif call_kind == "write_bytes":
        replacement, import_name = _path_write_replacement(
            source,
            node,
            receiver,
            binary=True,
        )
    else:
        raise RuntimeError(f"unsupported call kind {call_kind}")
    start = offsets[node.lineno - 1] + node.col_offset
    end = offsets[node.end_lineno - 1] + node.end_col_offset
    return Replacement(start, end, replacement, import_name)


def _csv_replacement(
    source: str,
    node: ast.Call,
    receiver: str,
) -> tuple[str, str]:
    args = list(node.args)
    keywords = list(node.keywords)
    if args:
        target = _segment(source, args.pop(0), node.lineno)
    else:
        target_keywords = [item for item in keywords if item.arg == "path_or_buf"]
        if len(target_keywords) != 1:
            raise RuntimeError(f"CSV target is ambiguous at line {node.lineno}")
        target = _segment(source, target_keywords[0].value, node.lineno)
        keywords.remove(target_keywords[0])
    parts = [receiver, target]
    parts.extend(_segment(source, arg, node.lineno) for arg in args)
    parts.extend(_keyword_segment(source, item, node.lineno) for item in keywords)
    return f"atomic_write_csv({', '.join(parts)})", "atomic_write_csv"


def _path_write_replacement(
    source: str,
    node: ast.Call,
    receiver: str,
    *,
    binary: bool,
) -> tuple[str, str]:
    if not node.args:
        raise RuntimeError(f"write payload is missing at line {node.lineno}")
    if binary and (len(node.args) != 1 or node.keywords):
        raise RuntimeError(f"binary write signature unsupported at line {node.lineno}")
    payload = _segment(source, node.args[0], node.lineno)
    parts = [receiver, payload]
    if not binary:
        if len(node.args) > 2:
            raise RuntimeError(f"text write signature unsupported at line {node.lineno}")
        if len(node.args) == 2:
            parts.append(f"encoding={_segment(source, node.args[1], node.lineno)}")
        for keyword in node.keywords:
            if keyword.arg not in {"encoding", "errors"}:
                raise RuntimeError(
                    f"text write keyword unsupported at line {node.lineno}"
                )
            parts.append(_keyword_segment(source, keyword, node.lineno))
    import_name = "atomic_write_bytes" if binary else "atomic_write_text"
    return f"{import_name}({', '.join(parts)})", import_name


def _keyword_segment(source: str, node: ast.keyword, line: int) -> str:
    if node.arg is None:
        return f"**{_segment(source, node.value, line)}"
    return f"{node.arg}={_segment(source, node.value, line)}"


def _segment(source: str, node: ast.AST, line: int) -> str:
    value = ast.get_source_segment(source, node)
    if value is None:
        raise RuntimeError(f"missing source segment at line {line}")
    return value


def _insert_import(source: str, import_name: str) -> str:
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
    lines.insert(
        future.end_lineno,
        "\nfrom quant_platform.orchestration.corrective_runtime "
        f"import {import_name}\n",
    )
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
    return "<dynamic>"


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
