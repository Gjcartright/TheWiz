from __future__ import annotations

import json

from quant_platform.orchestration.interactive_mixtape_graph import build_interactive_mixtape_solution


def test_interactive_mixtape_solution_writes_solution_artifacts(tmp_path):
    result = build_interactive_mixtape_solution(
        brief="Build The Ave as a virtual retail world with sneakers, apparel, lore, and hidden drops.",
        root=tmp_path,
    )

    markdown = result.paths["mixtape_solution_md"].read_text(encoding="utf-8")
    state = json.loads(result.paths["mixtape_solution_state"].read_text(encoding="utf-8"))

    assert "LangGraph Agent Workflow" in markdown
    assert "Sneaker District" in markdown
    assert "Merchitecture" in markdown
    assert "Hidden Tracks" in markdown
    assert "play_pressed" in markdown
    assert "Implementation Phases" in markdown
    assert "brand_strategist" in state["agents"]
    assert "technical_planner" in state["agents"]
    assert result.summary["agents"] == 7
    assert result.summary["phases"] == 6
    assert result.summary["zones"] == 5
