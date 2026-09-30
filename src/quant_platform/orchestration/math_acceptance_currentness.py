"""Read-only source/artifact preflight for scoped Math V2 acceptance.

Old markers lacking bindings remain historical, not current acceptance. This
checks local integrity and exact check identities, not source authentication,
numerical truth, independent approval or downstream learning/trading authority.
It never executes the acceptance producer or writes a marker. The producer emits
this schema; current qualifying execution remains separate from schema adoption.
"""
from __future__ import annotations

import ast
import csv
import json
import re
from datetime import UTC, datetime
from hashlib import sha256
from io import StringIO
from pathlib import Path

from quant_platform.performance_math import MATH_VERSION

BINDING_SCHEMA = "thewiz.math_acceptance.current_source_artifact_binding.v1"
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
# The producer plus its local arithmetic dependency closure. The shared rolling
# helper is deliberately included as a conservative numerical binding even when
# the retained separate Math V2 rolling variant does not call that helper.
REQUIRED_MATH_SOURCES = (
    "src/quant_platform/math_v2_acceptance.py",
    "src/quant_platform/orchestration/math_acceptance_currentness.py",
    "src/quant_platform/backtest.py",
    "src/quant_platform/economic_contract.py",
    "src/quant_platform/performance_math.py",
    "src/quant_platform/risk_metrics.py",
    "src/quant_platform/trade_ledger.py",
    "src/quant_platform/statistics/math_v2.py",
    "src/quant_platform/zscore_utils.py",
)


def source_path(relative: str) -> Path:
    """Resolve a bound module from the loaded package in source or wheel form."""
    prefix = "src/quant_platform/"
    if relative not in REQUIRED_MATH_SOURCES or not relative.startswith(prefix):
        raise ValueError("unregistered math source")
    return PACKAGE_ROOT / relative.removeprefix(prefix)
REPORTS = {
    "reconciliation_report": ("reports/active/math_v2_reconciliation.csv", "reconciliation_checks"),
    "statistical_validity_report": ("reports/active/statistical_validity_audit.csv", "statistical_checks"),
}
_COLUMNS = ["math_version", "check", "status", "observed", "required", "error"]
_MAX_EVIDENCE_BYTES = 4 * 1024 * 1024


def _read(path: Path) -> bytes:
    if not path.is_file() or not 0 < path.stat().st_size <= _MAX_EVIDENCE_BYTES:
        raise ValueError("bounded nonempty evidence file required")
    # The observed size can change after stat. Bound the actual read too and
    # reject growth/truncation beyond the declared nonempty size envelope.
    with path.open("rb") as stream:
        raw = stream.read(_MAX_EVIDENCE_BYTES + 1)
    if not 0 < len(raw) <= _MAX_EVIDENCE_BYTES:
        raise ValueError("bounded nonempty evidence file required")
    return raw


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError("nonfinite JSON")


def _hash_map(value, expected):
    if type(value) is not dict or set(value) != set(expected):
        raise ValueError("exact binding population required")
    if any(type(item) is not str or re.fullmatch(r"[0-9a-f]{64}", item) is None for item in value.values()):
        raise ValueError("invalid SHA256")


def _check_names(source: bytes) -> dict[str, tuple[str, ...]]:
    """Read the producer's literal check declarations without importing it."""
    tree = ast.parse(source)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_run_checks"]
    if len(functions) != 1:
        raise ValueError("one check declaration function required")
    result = {}
    for node in functions[0].body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id in {x[1] for x in REPORTS.values()}:
            if not isinstance(node.value, ast.List):
                raise ValueError("literal check list required")
            names = []
            for item in node.value.elts:
                if (not isinstance(item, ast.Tuple) or len(item.elts) != 2
                        or not isinstance(item.elts[0], ast.Constant) or type(item.elts[0].value) is not str
                        or not item.elts[0].value or not isinstance(item.elts[1], ast.Name)):
                    raise ValueError("explicit check identity and callable required")
                names.append(item.elts[0].value)
            if not names or len(names) != len(set(names)) or node.target.id in result:
                raise ValueError("unique nonempty check population required")
            result[node.target.id] = tuple(names)
    if set(result) != {x[1] for x in REPORTS.values()}:
        raise ValueError("both producer check populations required")
    if len(set(sum(result.values(), ()))) != sum(map(len, result.values())):
        raise ValueError("cross-report duplicate check identities")
    return result


def math_acceptance_marker_passes(path: Path, *, expected_math_version: str) -> bool:
    """Match marker, exact current source and both current acceptance reports.

    The source root is the loaded implementation, not a caller-selected alternate
    source tree. Reports must stay alongside the selected marker. Historical
    unbound/version-incompatible markers fail closed without being modified.
    """
    try:
        path = Path(path).resolve(strict=True)
        if path.name != "math_v2_acceptance.json" or path.parent.name != "active" or path.parent.parent.name != "reports":
            return False
        payload = json.loads(_read(path), object_pairs_hook=_pairs, parse_constant=_invalid_constant)
        if type(payload) is not dict or type(expected_math_version) is not str or expected_math_version != MATH_VERSION:
            return False
        required = {"status": "passed", "math_version": expected_math_version, "acceptance_scope": "core_math_library",
            "all_checks_passed": True, "generated_by": "quant_platform.math_v2_acceptance",
            "currentness_binding_schema": BINDING_SCHEMA}
        if any(type(payload.get(key)) is not type(value) or payload[key] != value for key, value in required.items()):
            return False
        for key in ("student_training_authority", "execution_authority", "wizard_exact_mode_parity", "walk_forward_signal_authority"):
            if payload.get(key) != "BLOCKED":
                return False
        generated = payload.get("generated_at")
        if type(generated) is not str:
            return False
        generated_at = datetime.fromisoformat(generated)
        if generated_at.tzinfo is None or generated_at > datetime.now(UTC):
            return False
        source_hashes, artifact_hashes = payload.get("source_hashes"), payload.get("artifact_hashes")
        _hash_map(source_hashes, REQUIRED_MATH_SOURCES)
        _hash_map(artifact_hashes, [value[0] for value in REPORTS.values()])
        sources = {name: _read(source_path(name)) for name in REQUIRED_MATH_SOURCES}
        if any(sha256(raw).hexdigest() != source_hashes[name] for name, raw in sources.items()):
            return False
        populations = _check_names(sources[REQUIRED_MATH_SOURCES[0]])
        count = 0
        root = path.parents[2]
        for field, (relative, population) in REPORTS.items():
            if payload.get(field) != relative:
                return False
            report_path = (root / relative).resolve(strict=True)
            if report_path.parent != path.parent:
                return False
            raw = _read(report_path)
            if sha256(raw).hexdigest() != artifact_hashes[relative]:
                return False
            reader = csv.DictReader(StringIO(raw.decode("utf-8")))
            if reader.fieldnames != _COLUMNS:
                return False
            rows = list(reader)
            if tuple(row["check"] for row in rows) != populations[population]:
                return False
            if any(set(row) != set(_COLUMNS) or row["math_version"] != expected_math_version
                   or row["status"] != "PASS" or row["error"] != "" for row in rows):
                return False
            count += len(rows)
        return all(type(payload.get(field)) is int and payload[field] == count for field in ("passed_checks", "total_checks"))
    except (OSError, ValueError, TypeError, KeyError, UnicodeError, csv.Error, SyntaxError, RecursionError, OverflowError):
        return False
