"""Source-derived inventory of terminal publication surfaces."""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

RUNTIME_FENCED_FUNCTIONS = {
    "atomic_append_text",
    "atomic_copy_file",
    "atomic_write_bytes",
    "atomic_write_csv",
    "atomic_write_parquet",
    "atomic_write_text",
    "create_exclusive_bytes",
    "create_exclusive_json",
    "create_exclusive_text",
    "immutable_snapshot_copy",
    "promote_staged_directory",
    "promote_staged_file",
    "write_immutable_bytes",
    "write_immutable_json",
}
FENCED_STORE_METHODS = {"publish_text", "publish_json", "publish_immutable_json"}
APPROVED_FENCE_MODULE = "quant_platform.orchestration.corrective_runtime"
OUTPUT_METHODS = {
    "write_text",
    "write_bytes",
    "to_csv",
    "to_json",
    "to_parquet",
    "to_pickle",
    "to_excel",
}
OUTPUT_FUNCTIONS = {
    "json.dump",
    "joblib.dump",
    "pickle.dump",
    "numpy.save",
    "np.save",
    "torch.save",
    "shutil.copy",
    "shutil.copy2",
    "shutil.copyfile",
    "shutil.copytree",
    "shutil.move",
    "os.replace",
    "os.rename",
    "os.link",
    "os.symlink",
}
FILE_PATH_RECEIVER_NAMES = {
    "candidate",
    "destination",
    "immutable",
    "path",
    "partial",
    "source",
    "staged",
    "temp",
    "temporary",
    "tmp",
}
STORE_INTERNAL_FUNCTIONS = {
    "atomic_append_text",
    "atomic_copy_file",
    "atomic_write_bytes",
    "atomic_write_csv",
    "atomic_write_parquet",
    "atomic_write_text",
    "immutable_snapshot_copy",
    "promote_staged_directory",
    "write_immutable_bytes",
    "write_immutable_json",
    "_atomic_write_governed_bytes",
    "_atomic_write_unmanaged_bytes",
    "_replace_existing_file_in_place",
}
STAGING_TARGET_NAMES = {"partial", "staged", "staging", "temp", "temporary", "tmp"}
STAGING_PROMOTION_FUNCTIONS = {"promote_staged_directory", "promote_staged_file"}


@dataclass(frozen=True)
class PublicationSurface:
    surface_id: str
    source_path: str
    source_sha256: str
    line: int
    function: str
    call: str
    call_kind: str
    authority_scope: str
    classification: str
    migration_state: str
    blocker: str


class _PublicationVisitor(ast.NodeVisitor):
    def __init__(
        self,
        *,
        relative: str,
        source_sha256: str,
        tree: ast.Module,
    ) -> None:
        self.relative = relative
        self.source_sha256 = source_sha256
        self.function_stack: list[str] = []
        self.local_bindings_stack: list[set[str]] = []
        self.surfaces: list[PublicationSurface] = []
        (
            self.approved_direct_fences,
            self.unapproved_direct_fences,
            self.approved_module_aliases,
        ) = _fence_import_bindings(tree)
        self.module_bindings = _scope_local_bindings(tree.body)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.function_stack.append(node.name)
        self.local_bindings_stack.append(_function_local_bindings(node))
        self.generic_visit(node)
        self.local_bindings_stack.pop()
        self.function_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)

    def visit_Call(self, node: ast.Call) -> None:
        call = _call_name(node.func)
        short = call.rsplit(".", maxsplit=1)[-1]
        function = self.function_stack[-1] if self.function_stack else "<module>"
        call_kind = _terminal_call_kind(node, call)
        fence_state = self._fence_state(node, call, short)
        if fence_state == "approved":
            self._append(
                node=node,
                function=function,
                call=call,
                call_kind="fenced_sink",
                classification="fenced_store_boundary",
                migration_state="MIGRATED",
                blocker="",
            )
        elif fence_state == "unverified":
            self._append(
                node=node,
                function=function,
                call=call,
                call_kind="unverified_fenced_sink",
                classification="unverified_fence_provenance",
                migration_state="UNMIGRATED",
                blocker="publication_fence_provenance_unverified",
            )
        elif call_kind:
            if (
                self.relative.endswith("corrective_runtime.py")
                and function in STORE_INTERNAL_FUNCTIONS
            ):
                classification = "store_boundary_internal"
                migration_state = "MIGRATED_INTERNAL"
                blocker = ""
            elif _is_staging_write(node, call_kind):
                classification = "staging_write"
                migration_state = "EXCLUDED_STAGING"
                blocker = ""
            elif "lock" in function.lower() or _call_receiver_is_lock(node):
                classification = "coordination_lock"
                migration_state = "EXCLUDED_COORDINATION"
                blocker = ""
            else:
                classification = "direct_terminal_writer"
                migration_state = "UNMIGRATED"
                blocker = "publication_not_routed_through_fenced_store"
            self._append(
                node=node,
                function=function,
                call=call,
                call_kind=call_kind,
                classification=classification,
                migration_state=migration_state,
                blocker=blocker,
            )
        self.generic_visit(node)

    def _fence_state(self, node: ast.Call, call: str, short: str) -> str:
        if (
            self.relative.endswith("corrective_evidence_store.py")
            and call == "self.publish_text"
            and self.function_stack[-1:] == ["publish_json"]
        ):
            return "approved"
        if isinstance(node.func, ast.Name):
            local_name = node.func.id
            if (
                self.relative.endswith("corrective_runtime.py")
                and local_name in RUNTIME_FENCED_FUNCTIONS
                and not any(
                    local_name in bindings for bindings in self.local_bindings_stack
                )
            ):
                return "approved"
            if self._is_locally_shadowed(local_name):
                return (
                    "unverified"
                    if local_name in RUNTIME_FENCED_FUNCTIONS
                    or local_name in self.approved_direct_fences
                    or local_name in self.unapproved_direct_fences
                    else ""
                )
            if local_name in self.approved_direct_fences:
                return "approved"
            if local_name in self.unapproved_direct_fences:
                return "unverified"
            return "unverified" if local_name in RUNTIME_FENCED_FUNCTIONS else ""
        if isinstance(node.func, ast.Attribute):
            receiver = _call_name(node.func.value)
            if (
                receiver in self.approved_module_aliases
                and short in RUNTIME_FENCED_FUNCTIONS
                and not self._is_locally_shadowed(receiver)
            ):
                return "approved"
            if short in RUNTIME_FENCED_FUNCTIONS or short in FENCED_STORE_METHODS:
                return "unverified"
        return ""

    def _is_locally_shadowed(self, name: str) -> bool:
        if any(name in bindings for bindings in self.local_bindings_stack):
            return True
        return (
            name in self.module_bindings
            and name not in self.approved_direct_fences
            and name not in self.approved_module_aliases
        )

    def _append(
        self,
        *,
        node: ast.Call,
        function: str,
        call: str,
        call_kind: str,
        classification: str,
        migration_state: str,
        blocker: str,
    ) -> None:
        material = f"{self.relative}:{node.lineno}:{function}:{call}:{call_kind}"
        self.surfaces.append(
            PublicationSurface(
                surface_id="pubsurface_"
                + sha256(material.encode("utf-8")).hexdigest()[:20],
                source_path=self.relative,
                source_sha256=self.source_sha256,
                line=node.lineno,
                function=function,
                call=call,
                call_kind=call_kind,
                authority_scope=_authority_scope(self.relative),
                classification=classification,
                migration_state=migration_state,
                blocker=blocker,
            )
        )


def publication_surface_rows(root: Path) -> list[dict[str, object]]:
    """Return a deterministic source-derived publication inventory."""

    root = root.resolve()
    source_root = root / "src" / "quant_platform"
    if not source_root.is_dir():
        return []
    surfaces: list[PublicationSurface] = []
    for path in sorted(source_root.rglob("*.py")):
        # AppleDouble metadata files on mounted macOS volumes are not Python source.
        if path.name.startswith("._"):
            continue
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
        except (OSError, UnicodeDecodeError, SyntaxError):
            relative = path.relative_to(root).as_posix()
            digest = _file_sha256(path) if path.is_file() else ""
            material = f"{relative}:parse_error"
            surfaces.append(
                PublicationSurface(
                    surface_id="pubsurface_"
                    + sha256(material.encode("utf-8")).hexdigest()[:20],
                    source_path=relative,
                    source_sha256=digest,
                    line=0,
                    function="<module>",
                    call="<parse>",
                    call_kind="parse_error",
                    authority_scope=_authority_scope(relative),
                    classification="uninspectable_source",
                    migration_state="UNMIGRATED",
                    blocker="publication_source_ast_unreadable",
                )
            )
            continue
        relative = path.relative_to(root).as_posix()
        visitor = _PublicationVisitor(
            relative=relative,
            source_sha256=sha256(source.encode("utf-8")).hexdigest(),
            tree=tree,
        )
        visitor.visit(tree)
        surfaces.extend(visitor.surfaces)
    return [
        asdict(surface)
        for surface in sorted(
            surfaces,
            key=lambda item: (item.source_path, item.line, item.surface_id),
        )
    ]


def unmigrated_publication_surface_ids(root: Path) -> list[str]:
    """Return every unfenced publication surface across all authority scopes."""

    return [
        str(row["surface_id"])
        for row in publication_surface_rows(root)
        if row["migration_state"] == "UNMIGRATED"
    ]


def unpaired_staging_surface_ids(root: Path) -> list[str]:
    """Return staging writes without an explicit governed promotion."""

    rows = publication_surface_rows(root)
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["source_path"]), str(row["function"]))
        grouped.setdefault(key, []).append(row)
    unpaired: list[str] = []
    for function_rows in grouped.values():
        staging_rows = [
            row
            for row in function_rows
            if row["migration_state"] == "EXCLUDED_STAGING"
        ]
        if not staging_rows:
            continue
        has_promotion = any(
            str(row["call"]).rsplit(".", maxsplit=1)[-1]
            in STAGING_PROMOTION_FUNCTIONS
            for row in function_rows
        )
        if not has_promotion:
            unpaired.extend(str(row["surface_id"]) for row in staging_rows)
    return sorted(unpaired)


def _terminal_call_kind(node: ast.Call, call: str) -> str:
    short = call.rsplit(".", maxsplit=1)[-1]
    if short in OUTPUT_METHODS:
        if short.startswith("to_") and not _has_output_argument(node):
            return ""
        return short
    if call in OUTPUT_FUNCTIONS:
        return short
    if short in {"replace", "rename"} and isinstance(node.func, ast.Attribute):
        receiver = _receiver_name(node.func.value)
        if receiver in FILE_PATH_RECEIVER_NAMES and len(node.args) == 1 and not node.keywords:
            return short
    if short == "open":
        mode = _open_mode(node, method=call != "open")
        if mode and any(token in mode for token in ("w", "a", "x", "+")):
            return "open_write"
    if call == "os.open" and _os_open_is_writable(node):
        return "os_open_write"
    return ""


def _fence_import_bindings(
    tree: ast.Module,
) -> tuple[dict[str, str], dict[str, str], set[str]]:
    approved: dict[str, str] = {}
    unapproved: dict[str, str] = {}
    module_aliases: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            for imported in node.names:
                if imported.name not in RUNTIME_FENCED_FUNCTIONS:
                    continue
                local_name = imported.asname or imported.name
                target = approved if node.module == APPROVED_FENCE_MODULE else unapproved
                target[local_name] = imported.name
        elif isinstance(node, ast.Import):
            for imported in node.names:
                if imported.name == APPROVED_FENCE_MODULE:
                    module_aliases.add(imported.asname or imported.name)
    return approved, unapproved, module_aliases


def _function_local_bindings(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> set[str]:
    bindings = {
        argument.arg
        for argument in (
            list(node.args.posonlyargs)
            + list(node.args.args)
            + list(node.args.kwonlyargs)
        )
    }
    if node.args.vararg is not None:
        bindings.add(node.args.vararg.arg)
    if node.args.kwarg is not None:
        bindings.add(node.args.kwarg.arg)
    bindings.update(_scope_local_bindings(node.body))
    return bindings


def _scope_local_bindings(statements: list[ast.stmt]) -> set[str]:
    bindings: set[str] = set()

    class BindingVisitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            bindings.add(node.name)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            bindings.add(node.name)

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            bindings.add(node.name)

        def visit_Name(self, node: ast.Name) -> None:
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                bindings.add(node.id)

        def visit_Import(self, node: ast.Import) -> None:
            for imported in node.names:
                bindings.add(imported.asname or imported.name.split(".", maxsplit=1)[0])

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            for imported in node.names:
                bindings.add(imported.asname or imported.name)

    visitor = BindingVisitor()
    for statement in statements:
        visitor.visit(statement)
    return bindings


def _has_output_argument(node: ast.Call) -> bool:
    if node.args:
        return not (isinstance(node.args[0], ast.Constant) and node.args[0].value is None)
    return any(
        keyword.arg in {"path", "path_or_buf", "excel_writer"}
        and not (isinstance(keyword.value, ast.Constant) and keyword.value.value is None)
        for keyword in node.keywords
    )


def _is_staging_write(node: ast.Call, call_kind: str) -> bool:
    if call_kind in {"replace", "rename"}:
        return False
    target = _terminal_target_expression(node, call_kind)
    return _receiver_name(target) in STAGING_TARGET_NAMES if target is not None else False


def _terminal_target_expression(node: ast.Call, call_kind: str) -> ast.expr | None:
    if call_kind in {"write_text", "write_bytes", "open_write"}:
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == "open":
                return node.func.value
            if node.func.attr in {"write_text", "write_bytes"}:
                return node.func.value
        return node.args[0] if node.args else None
    if call_kind.startswith("to_"):
        if node.args:
            return node.args[0]
        for keyword in node.keywords:
            if keyword.arg in {"path", "path_or_buf", "excel_writer"}:
                return keyword.value
    if call_kind == "os_open_write":
        return node.args[0] if node.args else None
    if call_kind in {
        "copy",
        "copy2",
        "copyfile",
        "copytree",
        "link",
        "move",
        "symlink",
    }:
        return node.args[1] if len(node.args) > 1 else None
    return None


def _open_mode(node: ast.Call, *, method: bool) -> str:
    index = 0 if method else 1
    if len(node.args) > index and isinstance(node.args[index], ast.Constant):
        value = node.args[index].value
        return value if isinstance(value, str) else ""
    for keyword in node.keywords:
        if keyword.arg == "mode" and isinstance(keyword.value, ast.Constant):
            value = keyword.value.value
            return value if isinstance(value, str) else ""
    return ""


def _os_open_is_writable(node: ast.Call) -> bool:
    if len(node.args) < 2:
        return False
    flags = ast.unparse(node.args[1])
    return any(
        token in flags for token in ("O_CREAT", "O_WRONLY", "O_RDWR", "O_APPEND")
    )


def _call_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return "<dynamic>"


def _receiver_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id.lower()
    if isinstance(node, ast.Attribute):
        return node.attr.lower()
    if isinstance(node, ast.BinOp):
        return _receiver_name(node.left)
    return ""


def _call_receiver_is_lock(node: ast.Call) -> bool:
    return (
        isinstance(node.func, ast.Attribute)
        and "lock" in _receiver_name(node.func.value)
    )


def _authority_scope(relative: str) -> str:
    if "/orchestration/corrective_" in relative:
        return "phase00_blocking"
    if relative in {
        "src/quant_platform/active_pipeline.py",
        "src/quant_platform/cli.py",
        "src/quant_platform/execution.py",
        "src/quant_platform/hyperliquid.py",
    }:
        return "execution_blocking"
    return "later_inventory"


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
