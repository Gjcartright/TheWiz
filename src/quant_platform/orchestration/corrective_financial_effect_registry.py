"""Source-derived inventory of order and other financial effect surfaces."""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

FINANCIAL_EFFECT_METHODS = frozenset(
    {
        "approve_agent",
        "approve_builder_fee",
        "broadcast_transaction",
        "bulk_cancel",
        "bulk_orders",
        "cancel_all_orders",
        "cancel_order",
        "close_position",
        "create_limit_order",
        "create_market_order",
        "create_order",
        "create_orders",
        "market_close",
        "market_open",
        "place_order",
        "send_transaction",
        "set_leverage",
        "sign_and_broadcast",
        "submit_order",
        "submit_pair",
        "transfer",
        "update_leverage",
        "usd_class_transfer",
        "vault_transfer",
        "withdraw",
    }
)


@dataclass(frozen=True)
class ReviewedFinancialSurface:
    classification: str
    required_fence_evidence: str


REVIEWED_FINANCIAL_SURFACES: dict[
    tuple[str, str, str], ReviewedFinancialSurface
] = {
    (
        "src/quant_platform/binance_testnet.py",
        "execute_binance_testnet_pair",
        "place_order",
    ): ReviewedFinancialSurface(
        "canonical_adapter_delegation",
        "_require_gate00g_binance_pair_adapter",
    ),
    (
        "src/quant_platform/dydx_sdk_order_adapter.py",
        "_place_order",
        "place_order",
    ): ReviewedFinancialSurface("direct_order_sink", "claim_effect_dispatch"),
    (
        "src/quant_platform/dydx_sdk_order_adapter.py",
        "_close_position",
        "close_position",
    ): ReviewedFinancialSurface("direct_order_sink", "claim_effect_dispatch"),
    (
        "src/quant_platform/execution.py",
        "place_order",
        "place_order",
    ): ReviewedFinancialSurface(
        "canonical_adapter_delegation",
        "_gate00g_adapter_instance_allowed",
    ),
    (
        "src/quant_platform/execution.py",
        "submit_paper_plan",
        "place_order",
    ): ReviewedFinancialSurface(
        "canonical_venue_delegation",
        "_gate00g_execution_venue_allowed",
    ),
    (
        "src/quant_platform/hyperliquid_testnet.py",
        "place_order",
        "place_order",
    ): ReviewedFinancialSurface(
        "record_only_adapter_delegation",
        "HyperliquidTestnetOrderAdapter",
    ),
    (
        "src/quant_platform/hyperliquid_testnet.py",
        "submit_pair",
        "submit_pair",
    ): ReviewedFinancialSurface(
        "canonical_executor_delegation",
        "_require_hyperliquid_pair_executor",
    ),
    (
        "src/quant_platform/hyperliquid_testnet.py",
        "_submit_pair_locked",
        "update_leverage",
    ): ReviewedFinancialSurface("direct_leverage_sink", "_claim_pair_authorization"),
    (
        "src/quant_platform/hyperliquid_testnet.py",
        "_submit_pair_locked",
        "bulk_orders",
    ): ReviewedFinancialSurface(
        "direct_order_sink",
        "_claim_all_pair_order_authorizations",
    ),
    (
        "src/quant_platform/hyperliquid_testnet.py",
        "_recover_pair_to_flat",
        "bulk_cancel",
    ): ReviewedFinancialSurface("direct_cancel_sink", "claim_effect_dispatch_all"),
    (
        "src/quant_platform/hyperliquid_testnet.py",
        "_recover_pair_to_flat",
        "market_close",
    ): ReviewedFinancialSurface("direct_close_sink", "claim_effect_dispatch"),
    (
        "src/quant_platform/orchestration/corrective_live_canary_execution.py",
        "run_live_canary_executor",
        "update_leverage",
    ): ReviewedFinancialSurface(
        "direct_leverage_sink",
        "_claim_canary_authorization",
    ),
    (
        "src/quant_platform/orchestration/corrective_live_canary_execution.py",
        "run_live_canary_executor",
        "bulk_orders",
    ): ReviewedFinancialSurface(
        "direct_order_sink",
        "claim_effect_dispatch_all",
    ),
    (
        "src/quant_platform/orchestration/corrective_live_canary_execution.py",
        "_recover_to_flat",
        "bulk_cancel",
    ): ReviewedFinancialSurface("direct_cancel_sink", "claim_effect_dispatch_all"),
    (
        "src/quant_platform/orchestration/corrective_live_canary_execution.py",
        "_recover_to_flat",
        "market_close",
    ): ReviewedFinancialSurface("direct_close_sink", "claim_effect_dispatch"),
    (
        "src/quant_platform/orchestration/corrective_testnet_collateral_transfer.py",
        "run_testnet_collateral_transfer",
        "usd_class_transfer",
    ): ReviewedFinancialSurface("direct_transfer_sink", "claim_effect_dispatch"),
    (
        "src/quant_platform/orchestration/corrective_testnet_pair_execution.py",
        "run_testnet_pair_execution",
        "submit_pair",
    ): ReviewedFinancialSurface(
        "canonical_executor_delegation",
        "_require_gate00g_testnet_pair_executor",
    ),
}


@dataclass(frozen=True)
class FinancialEffectSurface:
    surface_id: str
    source_path: str
    source_sha256: str
    line: int
    function: str
    call: str
    method: str
    effect_kind: str
    classification: str
    migration_state: str
    required_fence_evidence: str
    blocker: str


class _FinancialEffectVisitor(ast.NodeVisitor):
    def __init__(self, *, relative: str, source_sha256: str) -> None:
        self.relative = relative
        self.source_sha256 = source_sha256
        self.function_stack: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
        self.surfaces: list[FinancialEffectSurface] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.function_stack.append(node)
        self.generic_visit(node)
        self.function_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.function_stack.append(node)
        self.generic_visit(node)
        self.function_stack.pop()

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute):
            method = node.func.attr
            if method in FINANCIAL_EFFECT_METHODS:
                self._append(node, method)
        self.generic_visit(node)

    def _append(self, node: ast.Call, method: str) -> None:
        function_node = self.function_stack[-1] if self.function_stack else None
        function = function_node.name if function_node is not None else "<module>"
        key = (self.relative, function, method)
        reviewed = REVIEWED_FINANCIAL_SURFACES.get(key)
        call = ast.unparse(node.func)
        if reviewed is None:
            classification = "unknown_financial_effect_surface"
            migration_state = "UNMIGRATED"
            required_fence = ""
            blocker = "financial_effect_surface_not_reviewed"
        elif function_node is None or not _fence_seen_before(
            function_node,
            node,
            reviewed.required_fence_evidence,
        ):
            classification = reviewed.classification
            migration_state = "UNMIGRATED"
            required_fence = reviewed.required_fence_evidence
            blocker = "financial_effect_fence_evidence_missing_or_late"
        else:
            classification = reviewed.classification
            migration_state = "MIGRATED"
            required_fence = reviewed.required_fence_evidence
            blocker = ""
        material = f"{self.relative}:{node.lineno}:{function}:{call}:{method}"
        self.surfaces.append(
            FinancialEffectSurface(
                surface_id="financialsurface_"
                + sha256(material.encode("utf-8")).hexdigest()[:20],
                source_path=self.relative,
                source_sha256=self.source_sha256,
                line=node.lineno,
                function=function,
                call=call,
                method=method,
                effect_kind=_effect_kind(method),
                classification=classification,
                migration_state=migration_state,
                required_fence_evidence=required_fence,
                blocker=blocker,
            )
        )


def financial_effect_surface_rows(root: Path) -> list[dict[str, object]]:
    """Return the deterministic financial-effect inventory for project source."""

    root = root.resolve()
    source_root = root / "src" / "quant_platform"
    if not source_root.is_dir():
        return []
    surfaces: list[FinancialEffectSurface] = []
    for path in sorted(source_root.rglob("*.py")):
        if path.name.startswith("._"):
            continue
        relative = path.relative_to(root).as_posix()
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
        except (OSError, UnicodeDecodeError, SyntaxError):
            material = f"{relative}:parse_error"
            surfaces.append(
                FinancialEffectSurface(
                    surface_id="financialsurface_"
                    + sha256(material.encode("utf-8")).hexdigest()[:20],
                    source_path=relative,
                    source_sha256=_file_sha256(path) if path.is_file() else "",
                    line=0,
                    function="<module>",
                    call="<parse>",
                    method="<parse>",
                    effect_kind="unknown",
                    classification="uninspectable_source",
                    migration_state="UNMIGRATED",
                    required_fence_evidence="",
                    blocker="financial_effect_source_ast_unreadable",
                )
            )
            continue
        digest = sha256(source.encode("utf-8")).hexdigest()
        visitor = _FinancialEffectVisitor(relative=relative, source_sha256=digest)
        visitor.visit(tree)
        surfaces.extend(visitor.surfaces)
    return [
        asdict(surface)
        for surface in sorted(
            surfaces,
            key=lambda item: (item.source_path, item.line, item.surface_id),
        )
    ]


def unfenced_financial_effect_surface_ids(root: Path) -> list[str]:
    return [
        str(row["surface_id"])
        for row in financial_effect_surface_rows(root)
        if row["migration_state"] != "MIGRATED"
    ]


def _fence_seen_before(
    function_node: ast.FunctionDef | ast.AsyncFunctionDef,
    effect_node: ast.Call,
    token: str,
) -> bool:
    return _fence_dominates_block(
        function_node.body,
        effect_node=effect_node,
        token=token,
        inherited=False,
    )


def _fence_dominates_block(
    statements: list[ast.stmt],
    *,
    effect_node: ast.Call,
    token: str,
    inherited: bool,
) -> bool:
    fenced = inherited
    for statement in statements:
        if _contains_node(statement, effect_node):
            if _effect_is_in_direct_statement_expression(statement, effect_node):
                return fenced or _effect_depends_on_fence_call(effect_node, token)
            child = _containing_statement_block(statement, effect_node)
            if child is None:
                return False
            return _fence_dominates_block(
                child,
                effect_node=effect_node,
                token=token,
                inherited=fenced,
            )
        if _unconditional_statement_calls_fence(statement, token):
            fenced = True
    return False


def _unconditional_statement_calls_fence(statement: ast.stmt, token: str) -> bool:
    if (
        isinstance(statement, ast.If)
        and not statement.orelse
        and isinstance(statement.test, ast.UnaryOp)
        and isinstance(statement.test.op, ast.Not)
        and _expression_calls_fence(statement.test.operand, token)
        and _block_always_terminates(statement.body)
    ):
        return True
    if not isinstance(
        statement,
        (ast.Expr, ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Return, ast.Assert),
    ):
        return False
    return _expression_calls_fence(statement, token)


def _expression_calls_fence(node: ast.AST, token: str) -> bool:
    class FenceCallVisitor(ast.NodeVisitor):
        found = False

        def visit_Call(self, candidate: ast.Call) -> None:
            name = _call_name(candidate.func)
            if name == token or name.endswith(f".{token}"):
                self.found = True
                return
            self.generic_visit(candidate)

        def visit_Lambda(self, candidate: ast.Lambda) -> None:
            del candidate

        def visit_FunctionDef(self, candidate: ast.FunctionDef) -> None:
            del candidate

        def visit_AsyncFunctionDef(self, candidate: ast.AsyncFunctionDef) -> None:
            del candidate

    visitor = FenceCallVisitor()
    visitor.visit(node)
    return visitor.found


def _effect_depends_on_fence_call(effect_node: ast.Call, token: str) -> bool:
    return any(
        candidate is not effect_node
        and isinstance(candidate, ast.Call)
        and (
            _call_name(candidate.func) == token
            or _call_name(candidate.func).endswith(f".{token}")
        )
        for candidate in ast.walk(effect_node)
    )


def _block_always_terminates(statements: list[ast.stmt]) -> bool:
    return bool(statements) and isinstance(statements[-1], (ast.Return, ast.Raise))


def _contains_node(parent: ast.AST, target: ast.AST) -> bool:
    return any(node is target for node in ast.walk(parent))


def _effect_is_in_direct_statement_expression(
    statement: ast.stmt,
    effect_node: ast.Call,
) -> bool:
    if isinstance(statement, (ast.Expr, ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Return)):
        return _contains_node(statement, effect_node)
    return False


def _containing_statement_block(
    statement: ast.stmt,
    effect_node: ast.Call,
) -> list[ast.stmt] | None:
    candidate_blocks: list[list[ast.stmt]] = []
    for field_name, value in ast.iter_fields(statement):
        if field_name in {"body", "orelse", "finalbody"} and isinstance(value, list):
            candidate_blocks.append(
                [item for item in value if isinstance(item, ast.stmt)]
            )
        elif field_name == "handlers" and isinstance(value, list):
            candidate_blocks.extend(
                list(handler.body)
                for handler in value
                if isinstance(handler, ast.ExceptHandler)
            )
        elif field_name == "cases" and isinstance(value, list):
            candidate_blocks.extend(
                list(case.body) for case in value if isinstance(case, ast.match_case)
            )
    matches = [
        block
        for block in candidate_blocks
        if any(_contains_node(item, effect_node) for item in block)
    ]
    return matches[0] if len(matches) == 1 else None


def _effect_kind(method: str) -> str:
    if "cancel" in method:
        return "cancel"
    if "leverage" in method:
        return "leverage"
    if "transfer" in method or method in {"withdraw", "approve_agent"}:
        return "transfer"
    if "close" in method:
        return "close"
    return "order"


def _call_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return "<dynamic>"


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
