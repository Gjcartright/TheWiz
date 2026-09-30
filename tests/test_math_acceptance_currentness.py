"""Math acceptance markers must bind current source and the published checks."""

from __future__ import annotations

import json
from contextvars import Context
from pathlib import Path

import pytest

from quant_platform.math_v2_acceptance import build_math_v2_acceptance
from quant_platform.orchestration.effect_authority import EffectAuthorityError
from quant_platform.orchestration.math_acceptance_currentness import (
    REQUIRED_MATH_SOURCES,
    math_acceptance_marker_passes,
    source_path,
)
from quant_platform.orchestration.teacher_adapters import _math_marker_passes
from quant_platform.orchestration.teacher_contracts import MATH_V2
from quant_platform.orchestration.teacher_control_plane import _math_v2_marker_passes


def test_self_reported_marker_without_bound_reports_is_rejected(tmp_path):
    marker = tmp_path / "reports" / "active" / "math_v2_acceptance.json"
    marker.parent.mkdir(parents=True)
    marker.write_text(json.dumps({
        "status": "passed",
        "math_version": MATH_V2,
        "acceptance_scope": "core_math_library",
        "all_checks_passed": True,
        "generated_by": "quant_platform.math_v2_acceptance",
    }))
    assert not math_acceptance_marker_passes(marker, expected_math_version=MATH_V2)
    assert not _math_marker_passes(marker)
    assert not _math_v2_marker_passes(marker)


def test_math_marker_publication_requires_scoped_authority(tmp_path):
    with pytest.raises(EffectAuthorityError, match="unmanaged_publication_authority_session_missing"):
        Context().run(build_math_v2_acceptance, root=tmp_path)
    assert not (tmp_path / "reports" / "active" / "math_v2_acceptance.json").exists()


def test_bound_marker_rejects_tampered_report_and_source_hash(tmp_path):
    result = build_math_v2_acceptance(root=tmp_path)
    marker = result["marker"]
    assert math_acceptance_marker_passes(marker, expected_math_version=MATH_V2)
    assert _math_marker_passes(marker)
    assert _math_v2_marker_passes(marker)

    reconciliation = tmp_path / "reports" / "active" / "math_v2_reconciliation.csv"
    original_report = reconciliation.read_bytes()
    reconciliation.write_bytes(original_report + b"\n")
    assert not math_acceptance_marker_passes(marker, expected_math_version=MATH_V2)
    reconciliation.write_bytes(original_report)

    payload = json.loads(marker.read_text())
    payload["source_hashes"][REQUIRED_MATH_SOURCES[0]] = "0" * 64
    marker.write_text(json.dumps(payload))
    assert not math_acceptance_marker_passes(marker, expected_math_version=MATH_V2)


def test_bound_source_paths_follow_loaded_package():
    package_root = Path(__import__("quant_platform").__file__).resolve().parent
    for relative in REQUIRED_MATH_SOURCES:
        assert source_path(relative) == package_root / relative.removeprefix("src/quant_platform/")
        assert source_path(relative).is_file()
