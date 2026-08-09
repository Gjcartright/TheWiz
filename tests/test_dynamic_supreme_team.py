from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from quant_platform.orchestration.dynamic_supreme_team import build_dynamic_supreme_team_checkpoint


def test_dynamic_supreme_team_checkpoint_records_blocked_state_and_repair_path(tmp_path):
    directory = tmp_path / "reports" / "orchestration" / "dynamic_agents"; directory.mkdir(parents=True)
    pd.DataFrame(
        [{"strategy_family": "Copula", "rollout_status": "COPULA_SHADOW_CONTINUE", "promotion_allowed": False}]
    ).to_csv(directory / "strategy_cell_rollout.csv", index=False)

    result = build_dynamic_supreme_team_checkpoint(root=tmp_path, now=datetime(2026, 7, 24, tzinfo=timezone.utc))
    frame = pd.read_csv(result["csv"])
    assert set(frame["lens"]) == {"gap_analysis", "pre_mortem", "post_mortem", "red_team"}
    assert "Collect distinct, fresh, complete Copula comparisons" in frame.loc[frame["lens"] == "gap_analysis", "required_action"].item()
    assert "BLOCKED" in result["markdown"].read_text(encoding="utf-8")
