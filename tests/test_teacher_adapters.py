from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from quant_platform.math_v2_acceptance import build_math_v2_acceptance
from quant_platform.statistics.math_v2 import MATH_VERSION
from quant_platform.orchestration.teacher_adapters import build_teacher_evidence_adapters
from quant_platform.orchestration.teacher_contracts import EXACT_MODES, REQUIRED_CRITICS


def _write_complete_inputs(root):
    active = root / "reports" / "active"
    active.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    common = {
        "run_id": "hlrun-fixture-1",
        "candidate_set_id": "hlset-fixture-1",
        "setup_identity": "SOL-USD|WLD-USD|1d|320|all_modes",
        "pair": "SOL-USD/WLD-USD",
        "venue": "hyperliquid",
        "timeframe": "1d",
        "lookback": 320,
        "source_snapshot_id": "snapshot-1",
        "source_timestamp": timestamp,
    }
    teachers = []
    for mode in EXACT_MODES:
        teachers.append(
            {
                **common,
                "exact_mode": mode.value,
                "proposed_action": "abstain",
                "confidence": 0.6,
                "uncertainty": 0.3,
                "expected_net_return": 0.01,
                "lower_bound_net_return": -0.01,
                "expected_holding_bars": 5,
                "entry_style": "captured_threshold",
                "exit_style": "captured_exit",
                "invalidation_condition": "structural_break",
                "required_regime": "all",
                "source_system": "hyperliquid_local_replay",
                "formula_version": "formula-v2",
                "math_version": MATH_VERSION,
                "mode_fidelity_status": "local_validated_estimator",
                "point_in_time_status": "confirmed",
                "history_hash": "history-hash-1",
                "settings_version": "settings-v1",
                "cost_model_version": "cost-v1",
                "train_start": "2025-01-01T00:00:00+00:00",
                "train_end": "2025-10-31T00:00:00+00:00",
                "test_start": "2025-11-01T00:00:00+00:00",
                "test_end": timestamp,
                "blockers": "",
                "evidence_path": f"reports/replay/{mode.name.lower()}.csv",
            }
        )
    critics = []
    for critic in REQUIRED_CRITICS:
        critics.append(
            {
                **common,
                "critic_type": critic.value,
                "verdict": "pass",
                "score": 0.8,
                "reason": "controlled fixture passes",
                "authority": "local_point_in_time",
                "point_in_time_status": "confirmed",
                "blocker_codes": "",
                "evidence_path": f"reports/critics/{critic.value}.csv",
            }
        )
    pd.DataFrame(teachers).to_csv(active / "math_v2_teacher_inputs.csv", index=False)
    pd.DataFrame(critics).to_csv(active / "math_v2_critic_inputs.csv", index=False)


def test_adapter_emits_only_a_complete_seven_teacher_six_critic_context(tmp_path):
    build_math_v2_acceptance(root=tmp_path)
    _write_complete_inputs(tmp_path)
    result = build_teacher_evidence_adapters(root=tmp_path)
    assert result["status"] == "READY_FOR_COUNCIL"
    assert result["proposal_count"] == 7
    assert result["assessment_count"] == 6
    assert len(result["proposals"].read_text(encoding="utf-8").splitlines()) == 7
    assert len(result["critics"].read_text(encoding="utf-8").splitlines()) == 6


def test_adapter_clears_streams_when_context_is_incomplete(tmp_path):
    build_math_v2_acceptance(root=tmp_path)
    _write_complete_inputs(tmp_path)
    first = build_teacher_evidence_adapters(root=tmp_path)
    teacher_input = tmp_path / "reports" / "active" / "math_v2_teacher_inputs.csv"
    pd.read_csv(teacher_input).iloc[:-1].to_csv(teacher_input, index=False)
    second = build_teacher_evidence_adapters(root=tmp_path)
    assert first["proposal_count"] == 7
    assert second["status"] == "BLOCKED"
    assert second["proposal_count"] == 0
    assert second["proposals"].read_text(encoding="utf-8") == ""
