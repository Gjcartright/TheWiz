from __future__ import annotations

import pandas as pd

from quant_platform.math_v2_acceptance import build_math_v2_acceptance
from quant_platform.orchestration.dynamic_stage_runner import run_dynamic_stage
from quant_platform.orchestration.state import OrchestratorState
from quant_platform.orchestration.teacher_control_plane import build_teacher_council_control_plane


def test_control_plane_connects_wizard_as_discovery_and_blocks_missing_local_evidence(tmp_path):
    source = tmp_path / "data" / "processed" / "wizard_evidence.csv"
    source.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "pair": "SOL-USD/WLD-USD",
                "exchange": "dydx",
                "interval": "daily",
                "exact_mode": "OU (Spread)",
                "sharpe": 2.1,
                "returns_total": 0.20,
                "discovery_min_returns_total": 0.10,
                "passes_sharpe_gate": True,
                "source_timestamp": "2026-08-06T18:00:00+00:00",
                "source_fresh": True,
                "evidence_path": "reports/wizard.csv",
            }
        ]
    ).to_csv(source, index=False)

    result = build_teacher_council_control_plane(root=tmp_path)
    wizard = pd.read_csv(result["wizard_discovery"])
    readiness = pd.read_csv(result["readiness"])
    assert result["status"] == "BLOCKED"
    assert wizard.loc[0, "authority"] == "discovery_only"
    assert not bool(wizard.loc[0, "local_vote_eligible"])
    assert not bool(wizard.loc[0, "student_label_eligible"])
    assert "math_v2_not_accepted" in set(readiness["blocker"])
    assert "teacher_proposal_stream_empty" in set(readiness["blocker"])


def test_dynamic_teacher_council_stage_reports_blocked_readiness(tmp_path):
    result = run_dynamic_stage(
        "teacher_council_control_plane",
        OrchestratorState(report_only=True, root=tmp_path),
        tmp_path,
    )
    assert result.status.value == "blocked"
    assert result.blocker == "teacher_council_readiness_blocked"
    assert "teacher_council_readiness.csv" in result.evidence_path


def test_control_plane_accepts_only_the_machine_generated_math_v2_marker(tmp_path):
    build_math_v2_acceptance(root=tmp_path)
    result = build_teacher_council_control_plane(root=tmp_path)
    readiness = pd.read_csv(result["readiness"]).set_index("check")
    assert readiness.loc["math_v2_accepted", "status"] == "PASS"
    assert readiness.loc["teacher_proposals_present", "status"] == "BLOCKED"
