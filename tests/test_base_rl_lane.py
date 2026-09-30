import pandas as pd
import json
import pytest

import quant_platform.active_pipeline as active_pipeline
from quant_platform.active_pipeline import CommandResult
import quant_platform.rl.base_rl_lane as base_rl_lane
from quant_platform.rl.base_rl_lane import (
    _candidate_set_hash,
    _build_pair_blocker_report,
    _base_rl_pair_coverage,
    _build_comparison_and_promotion,
    base_rl_paper_handoff_report,
    _route_model_gate_status,
    run_augmented_rl,
    refresh_base_rl_feedback,
)


def test_base_rl_paper_handoff_blocks_when_shortlist_pair_lacks_support(tmp_path, monkeypatch):
    reports = tmp_path / "reports"
    (reports / "rl").mkdir(parents=True)
    (reports / "ml").mkdir(parents=True)
    (reports / "dashboard").mkdir(parents=True)
    (reports / "active").mkdir(parents=True)

    pd.DataFrame(
        [
            {"gate": "strategy_acceptance", "ready": True},
            {"gate": "paper_execution_gate", "ready": True},
        ]
    ).to_csv(reports / "priority_readiness.csv", index=False)
    pd.DataFrame([{"ready": True}, {"ready": True}]).to_csv(reports / "paper_execution_preflight.csv", index=False)
    pd.DataFrame([{"accepted": True}]).to_csv(reports / "ml" / "model_gated_acceptance.csv", index=False)
    pd.DataFrame(
        [
            {"pair": "BTC-USD-ETH-USD", "best_execution_venue": "dydx", "evidence_path": "reports/pair.csv"},
            {"pair": "SOL-USD-TRX-USD", "best_execution_venue": "dydx", "evidence_path": "reports/pair2.csv"},
        ]
    ).to_csv(reports / "dashboard" / "paper_candidate_shortlist.csv", index=False)
    pd.DataFrame(
        [
            {"pair": "BTC-USD-ETH-USD", "best_execution_venue": "dydx", "evidence_path": "reports/pair.csv"},
            {"pair": "SOL-USD-TRX-USD", "best_execution_venue": "dydx", "evidence_path": "reports/pair2.csv"},
        ]
    ).to_csv(reports / "rl" / "base_rl_route_candidates.csv", index=False)
    pd.DataFrame(
        [
            {
                "policy_name": "winner",
                "winner": 1,
                "profit_factor": 3.0,
                "top_pairs_entered": "BTC-USD-ETH-USD:8",
            }
        ]
    ).to_csv(reports / "rl" / "rl_learning_cycle_summary.csv", index=False)
    pd.DataFrame(
        [
            {"market": "BTC-USD", "compatible_for_paper_submit": True},
            {"market": "ETH-USD", "compatible_for_paper_submit": True},
            {"market": "SOL-USD", "compatible_for_paper_submit": True},
            {"market": "TRX-USD", "compatible_for_paper_submit": True},
        ]
    ).to_csv(reports / "active" / "dydx_execution_market_compatibility.csv", index=False)
    monkeypatch.setattr(
        base_rl_lane,
        "effective_dydx_account_state_snapshot",
        lambda config, root=tmp_path: {"checked": True, "open_markets": [], "positions": [], "blocker": "", "source": "dydx_indexer"},
    )

    frame = base_rl_paper_handoff_report(root=tmp_path)

    assert frame.iloc[0]["status"] == "research_only"
    assert frame.iloc[0]["blocker"] == "pair_specific_rl_support_missing"
    assert bool(frame.iloc[0]["paper_authorized"]) is False


@pytest.mark.parametrize(
    ("preflight_row", "expected_status"),
    [({"ready": True}, "paper_authorized"), ({"check": "present_without_ready"}, "research_only")],
)
def test_base_rl_paper_handoff_accepts_route_specific_model_gate(tmp_path, monkeypatch, preflight_row, expected_status):
    reports = tmp_path / "reports"
    (reports / "rl").mkdir(parents=True)
    (reports / "ml").mkdir(parents=True)
    (reports / "active").mkdir(parents=True)

    pd.DataFrame(
        [
            {"gate": "strategy_acceptance", "ready": True},
            {"gate": "paper_execution_gate", "ready": True},
        ]
    ).to_csv(reports / "priority_readiness.csv", index=False)
    pd.DataFrame([preflight_row, preflight_row]).to_csv(reports / "paper_execution_preflight.csv", index=False)
    pd.DataFrame([{"accepted": False}]).to_csv(reports / "ml" / "model_gated_acceptance.csv", index=False)
    pd.DataFrame(
        [
            {"pair": "BTC-USD-LDO-USD", "best_execution_venue": "dydx", "evidence_path": "reports/pair.csv"},
        ]
    ).to_csv(reports / "rl" / "base_rl_route_candidates.csv", index=False)
    pd.DataFrame(
        [
            {"pair": "BTC-USD-LDO-USD", "support_status": "strong_model_support", "recommended_repair_action": "preserve"},
        ]
    ).to_csv(reports / "ml" / "model_gate_pair_support_report.csv", index=False)
    pd.DataFrame(
        [
            {"policy_name": "winner", "winner": 1, "profit_factor": 3.0, "top_pairs_entered": "BTC-USD-LDO-USD:8"},
        ]
    ).to_csv(reports / "rl" / "rl_learning_cycle_summary.csv", index=False)
    pd.DataFrame(
        [
            {"market": "BTC-USD", "compatible_for_paper_submit": True},
            {"market": "LDO-USD", "compatible_for_paper_submit": True},
        ]
    ).to_csv(reports / "active" / "dydx_execution_market_compatibility.csv", index=False)
    monkeypatch.setattr(
        base_rl_lane,
        "effective_dydx_account_state_snapshot",
        lambda config, root=tmp_path: {"checked": True, "open_markets": [], "positions": [], "blocker": "", "source": "dydx_indexer"},
    )

    frame = base_rl_paper_handoff_report(root=tmp_path)

    assert bool(frame.iloc[0]["route_model_gate_accepted"]) is True
    assert bool(frame.iloc[0]["model_gate_accepted"]) is True
    assert frame.iloc[0]["status"] == expected_status
    assert bool(frame.iloc[0]["paper_authorized"]) is (expected_status == "paper_authorized")
    if expected_status == "research_only":
        assert bool(frame.iloc[0]["paper_execution_ready"]) is False
        assert frame.iloc[0]["blocker"] == "paper_execution_not_ready"


def test_base_rl_paper_handoff_blocks_when_route_specific_model_gate_is_weak(tmp_path, monkeypatch):
    reports = tmp_path / "reports"
    (reports / "rl").mkdir(parents=True)
    (reports / "ml").mkdir(parents=True)
    (reports / "active").mkdir(parents=True)

    pd.DataFrame(
        [
            {"gate": "strategy_acceptance", "ready": True},
            {"gate": "paper_execution_gate", "ready": True},
        ]
    ).to_csv(reports / "priority_readiness.csv", index=False)
    pd.DataFrame([{"ready": True}, {"ready": True}]).to_csv(reports / "paper_execution_preflight.csv", index=False)
    pd.DataFrame([{"accepted": False}]).to_csv(reports / "ml" / "model_gated_acceptance.csv", index=False)
    pd.DataFrame(
        [
            {"pair": "BTC-USD-LDO-USD", "best_execution_venue": "dydx", "evidence_path": "reports/pair.csv"},
        ]
    ).to_csv(reports / "rl" / "base_rl_route_candidates.csv", index=False)
    pd.DataFrame(
        [
            {"pair": "BTC-USD-LDO-USD", "support_status": "weak_model_support", "recommended_repair_action": "improve"},
        ]
    ).to_csv(reports / "ml" / "model_gate_pair_support_report.csv", index=False)
    pd.DataFrame(
        [
            {"policy_name": "winner", "winner": 1, "profit_factor": 3.0, "top_pairs_entered": "BTC-USD-LDO-USD:8"},
        ]
    ).to_csv(reports / "rl" / "rl_learning_cycle_summary.csv", index=False)
    pd.DataFrame(
        [
            {"market": "BTC-USD", "compatible_for_paper_submit": True},
            {"market": "LDO-USD", "compatible_for_paper_submit": True},
        ]
    ).to_csv(reports / "active" / "dydx_execution_market_compatibility.csv", index=False)
    monkeypatch.setattr(
        base_rl_lane,
        "effective_dydx_account_state_snapshot",
        lambda config, root=tmp_path: {"checked": True, "open_markets": [], "positions": [], "blocker": "", "source": "dydx_indexer"},
    )

    frame = base_rl_paper_handoff_report(root=tmp_path)

    assert bool(frame.iloc[0]["route_model_gate_accepted"]) is False
    assert frame.iloc[0]["route_model_gate_blocker"] == "route_model_gate_not_accepted"
    assert frame.iloc[0]["blocker"] == "route_model_gate_not_accepted"
    assert frame.iloc[0]["status"] == "research_only"


def test_route_model_gate_status_matches_normalized_pair_formats(tmp_path):
    reports = tmp_path / "reports"
    (reports / "ml").mkdir(parents=True)

    candidate_set = pd.DataFrame(
        [
            {"pair": "AAA-USD-BBB-USD"},
            {"pair": "CCC/DDD"},
        ]
    )
    pd.DataFrame(
        [
            {"pair": "AAA/BBB", "support_status": "strong_model_support"},
            {"pair": "CCC-USD-DDD-USD", "support_status": "strong_model_support"},
        ]
    ).to_csv(reports / "ml" / "model_gate_pair_support_report.csv", index=False)

    route_ok, blocker = _route_model_gate_status(root=tmp_path, candidate_set=candidate_set)

    assert route_ok is True
    assert blocker == ""


def test_route_model_gate_status_reports_empty_shortlist(tmp_path):
    route_ok, blocker = _route_model_gate_status(root=tmp_path, candidate_set=pd.DataFrame())

    assert route_ok is False
    assert blocker == "route_shortlist_empty"


def test_base_rl_handoff_reports_execution_blocker_when_empty_route_has_blocked_audit_candidates(tmp_path, monkeypatch):
    reports = tmp_path / "reports"
    (reports / "rl").mkdir(parents=True)
    (reports / "ml").mkdir(parents=True)
    (reports / "active").mkdir(parents=True)

    pd.DataFrame(
        [
            {"gate": "strategy_acceptance", "ready": True},
            {"gate": "paper_execution_gate", "ready": True},
        ]
    ).to_csv(reports / "priority_readiness.csv", index=False)
    pd.DataFrame([{"ready": True}, {"ready": True}]).to_csv(reports / "paper_execution_preflight.csv", index=False)
    pd.DataFrame([{"accepted": True}]).to_csv(reports / "ml" / "model_gated_acceptance.csv", index=False)
    pd.DataFrame(columns=["pair"]).to_csv(reports / "rl" / "base_rl_route_candidates.csv", index=False)
    pd.DataFrame(columns=["policy_name", "winner", "profit_factor", "top_pairs_entered"]).to_csv(
        reports / "rl" / "rl_learning_cycle_summary.csv", index=False
    )
    pd.DataFrame(
        [
            {"market": "BNB-USD", "compatible_for_paper_submit": False},
            {"market": "LDO-USD", "compatible_for_paper_submit": False},
        ]
    ).to_csv(reports / "active" / "dydx_execution_market_compatibility.csv", index=False)
    monkeypatch.setattr(
        base_rl_lane,
        "paper_candidate_shortlist_rows",
        lambda root, max_pairs=5, require_execution_compatible=True: pd.DataFrame(
            [
                {
                    "pair": "BNB-USD-LDO-USD",
                    "best_execution_venue": "dydx",
                    "execution_compatible": False,
                    "execution_blocker": "unsupported_execution_legs",
                }
            ]
        )
        if not require_execution_compatible
        else pd.DataFrame(columns=["pair"]),
    )
    monkeypatch.setattr(
        base_rl_lane,
        "effective_dydx_account_state_snapshot",
        lambda config, root=tmp_path: {"checked": True, "open_markets": [], "positions": [], "blocker": "", "source": "browser_override"},
    )

    frame = base_rl_paper_handoff_report(root=tmp_path)

    row = frame.iloc[0]
    assert row["status"] == "research_only"
    assert row["blocker"] == "route_market_unconfirmed_on_exchange"
    assert bool(row["journal_monitoring_authorized"]) is True
    assert bool(row["journal_learning_ready"]) is True
    assert row["compatibility_audit_pairs"] == 1
    assert row["compatibility_audit_execution_compatible_pairs"] == 0
    assert row["compatibility_audit_blocked_pairs"] == 1


def test_route_model_gate_status_rejects_missing_strong_support(tmp_path):
    reports = tmp_path / "reports"
    (reports / "ml").mkdir(parents=True)

    candidate_set = pd.DataFrame([{"pair": "AAA-USD-BBB-USD"}])
    pd.DataFrame(
        [{"pair": "AAA/BBB", "support_status": "weak_model_support"}]
    ).to_csv(reports / "ml" / "model_gate_pair_support_report.csv", index=False)

    route_ok, blocker = _route_model_gate_status(root=tmp_path, candidate_set=candidate_set)

    assert route_ok is False
    assert blocker == "route_model_gate_not_accepted"


def test_base_rl_paper_handoff_prefers_specific_execution_blocker(tmp_path, monkeypatch):
    reports = tmp_path / "reports"
    (reports / "rl").mkdir(parents=True)
    (reports / "ml").mkdir(parents=True)
    (reports / "active").mkdir(parents=True)

    pd.DataFrame(
        [
            {"gate": "strategy_acceptance", "ready": True},
            {"gate": "paper_execution_gate", "ready": False},
        ]
    ).to_csv(reports / "priority_readiness.csv", index=False)
    pd.DataFrame(
        [
            {"step": "paper_submission_gate", "ready": False, "blocker": "route_market_unconfirmed_on_exchange"},
            {"step": "execution_compatibility", "ready": False, "blocker": "route_market_unconfirmed_on_exchange"},
            {"step": "account_state_clean", "ready": True, "blocker": ""},
        ]
    ).to_csv(reports / "paper_execution_preflight.csv", index=False)
    pd.DataFrame([{"accepted": True}]).to_csv(reports / "ml" / "model_gated_acceptance.csv", index=False)
    pd.DataFrame([{"pair": "BTC-USD-LDO-USD"}]).to_csv(reports / "rl" / "base_rl_route_candidates.csv", index=False)
    pd.DataFrame([{"pair": "BTC-USD-LDO-USD", "support_status": "strong_model_support"}]).to_csv(
        reports / "ml" / "model_gate_pair_support_report.csv", index=False
    )
    pd.DataFrame(
        [
            {"policy_name": "winner", "winner": 1, "profit_factor": 3.0, "top_pairs_entered": "BTC-USD-LDO-USD:8"},
        ]
    ).to_csv(reports / "rl" / "rl_learning_cycle_summary.csv", index=False)
    pd.DataFrame(
        [
            {"market": "BTC-USD", "compatible_for_paper_submit": False},
            {"market": "LDO-USD", "compatible_for_paper_submit": False},
        ]
    ).to_csv(reports / "active" / "dydx_execution_market_compatibility.csv", index=False)
    monkeypatch.setattr(
        base_rl_lane,
        "effective_dydx_account_state_snapshot",
        lambda config, root=tmp_path: {"checked": True, "open_markets": [], "positions": [], "blocker": "", "source": "dydx_indexer"},
    )

    frame = base_rl_paper_handoff_report(root=tmp_path)

    assert frame.iloc[0]["status"] == "research_only"
    assert frame.iloc[0]["blocker"] == "route_market_unconfirmed_on_exchange"


def test_base_rl_paper_handoff_accepts_browser_account_override(tmp_path, monkeypatch):
    reports = tmp_path / "reports"
    (reports / "rl").mkdir(parents=True)
    (reports / "ml").mkdir(parents=True)
    (reports / "active").mkdir(parents=True)

    pd.DataFrame(
        [
            {"gate": "strategy_acceptance", "ready": True},
            {"gate": "paper_execution_gate", "ready": True},
        ]
    ).to_csv(reports / "priority_readiness.csv", index=False)
    pd.DataFrame(
        [
            {"step": "paper_submission_gate", "ready": True, "blocker": ""},
            {"step": "execution_compatibility", "ready": True, "blocker": ""},
            {"step": "account_state_clean", "ready": True, "blocker": ""},
        ]
    ).to_csv(reports / "paper_execution_preflight.csv", index=False)
    pd.DataFrame([{"accepted": True}]).to_csv(reports / "ml" / "model_gated_acceptance.csv", index=False)
    pd.DataFrame([{"pair": "BTC-USD-LDO-USD"}]).to_csv(reports / "rl" / "base_rl_route_candidates.csv", index=False)
    pd.DataFrame([{"pair": "BTC-USD-LDO-USD", "support_status": "strong_model_support"}]).to_csv(
        reports / "ml" / "model_gate_pair_support_report.csv", index=False
    )
    pd.DataFrame(
        [
            {"policy_name": "winner", "winner": 1, "profit_factor": 3.0, "top_pairs_entered": "BTC-USD-LDO-USD:8"},
        ]
    ).to_csv(reports / "rl" / "rl_learning_cycle_summary.csv", index=False)
    pd.DataFrame(
        [
            {"market": "BTC-USD", "compatible_for_paper_submit": True},
            {"market": "LDO-USD", "compatible_for_paper_submit": True},
        ]
    ).to_csv(reports / "active" / "dydx_execution_market_compatibility.csv", index=False)
    (reports / "active" / "browser_account_state_override.json").write_text(
        json.dumps(
            {
                "confirmed_flat": True,
                "source": "browser_user_confirmed",
                "note": "browser shows no open positions",
                "confirmed_at_utc": "2026-07-08T16:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        base_rl_lane,
        "effective_dydx_account_state_snapshot",
        lambda config, root=tmp_path: {
            "checked": True,
            "open_markets": [],
            "positions": [],
            "blocker": "",
            "source": "browser_override",
        },
    )

    frame = base_rl_paper_handoff_report(root=tmp_path)

    assert frame.iloc[0]["status"] == "paper_authorized"
    assert bool(frame.iloc[0]["paper_authorized"]) is True


def test_base_rl_paper_handoff_prioritizes_orphan_account_state_blocker(tmp_path, monkeypatch):
    reports = tmp_path / "reports"
    (reports / "rl").mkdir(parents=True)
    (reports / "ml").mkdir(parents=True)
    (reports / "active").mkdir(parents=True)

    pd.DataFrame(
        [
            {"gate": "strategy_acceptance", "ready": True},
            {"gate": "paper_execution_gate", "ready": True},
        ]
    ).to_csv(reports / "priority_readiness.csv", index=False)
    pd.DataFrame(
        [
            {"step": "paper_submission_gate", "ready": True, "blocker": ""},
            {"step": "execution_compatibility", "ready": True, "blocker": ""},
            {"step": "account_state_clean", "ready": False, "blocker": "orphan_leg_open"},
        ]
    ).to_csv(reports / "paper_execution_preflight.csv", index=False)
    pd.DataFrame([{"accepted": True}]).to_csv(reports / "ml" / "model_gated_acceptance.csv", index=False)
    pd.DataFrame([{"pair": "BTC-USD-LDO-USD"}]).to_csv(reports / "rl" / "base_rl_route_candidates.csv", index=False)
    pd.DataFrame([{"pair": "BTC-USD-LDO-USD", "support_status": "strong_model_support"}]).to_csv(
        reports / "ml" / "model_gate_pair_support_report.csv", index=False
    )
    pd.DataFrame(
        [
            {"policy_name": "winner", "winner": 1, "profit_factor": 3.0, "top_pairs_entered": "BTC-USD-LDO-USD:8"},
        ]
    ).to_csv(reports / "rl" / "rl_learning_cycle_summary.csv", index=False)
    pd.DataFrame(
        [
            {"market": "BTC-USD", "compatible_for_paper_submit": True},
            {"market": "LDO-USD", "compatible_for_paper_submit": True},
        ]
    ).to_csv(reports / "active" / "dydx_execution_market_compatibility.csv", index=False)
    monkeypatch.setattr(
        base_rl_lane,
        "effective_dydx_account_state_snapshot",
        lambda config, root=tmp_path: {
            "checked": True,
            "open_markets": ["LDO-USD"],
            "positions": [{"market": "LDO-USD", "size": 4.0}],
            "blocker": "orphan_leg_open",
            "source": "dydx_indexer",
        },
    )

    frame = base_rl_paper_handoff_report(root=tmp_path)

    assert frame.iloc[0]["status"] == "research_only"
    assert frame.iloc[0]["blocker"] == "orphan_leg_open"
    assert bool(frame.iloc[0]["paper_authorized"]) is False


def test_run_base_rl_reuses_existing_dataset_for_targeted_pair(tmp_path, monkeypatch):
    (tmp_path / "data" / "ml").mkdir(parents=True)
    (tmp_path / "reports" / "rl").mkdir(parents=True)
    (tmp_path / "models" / "rl" / "learning").mkdir(parents=True)
    (tmp_path / "reports" / "ml").mkdir(parents=True)
    (tmp_path / "reports").mkdir(exist_ok=True)

    pd.DataFrame([{"pair": "BTC-USD-LDO-USD"}]).to_csv(tmp_path / "data" / "ml" / "trade_training_dataset.csv", index=False)
    pd.DataFrame([{"gate": "strategy_acceptance", "ready": True}, {"gate": "paper_execution_gate", "ready": True}]).to_csv(
        tmp_path / "reports" / "priority_readiness.csv", index=False
    )
    pd.DataFrame([{"ready": True}, {"ready": True}]).to_csv(tmp_path / "reports" / "paper_execution_preflight.csv", index=False)
    pd.DataFrame([{"accepted": False}]).to_csv(tmp_path / "reports" / "ml" / "model_gated_acceptance.csv", index=False)

    monkeypatch.setattr(
        base_rl_lane,
        "build_trade_dataset",
        lambda root=tmp_path: (_ for _ in ()).throw(AssertionError("build_trade_dataset should not run")),
    )
    monkeypatch.setattr(
        base_rl_lane,
        "run_rl_research",
        lambda root=tmp_path, pair_id="": CommandResult(
            paths={
                "training_report": tmp_path / "reports" / "rl" / "research_training.csv",
                "evaluation_report": tmp_path / "reports" / "rl" / "research_eval.csv",
                "blocked_actions": tmp_path / "reports" / "rl" / "research_blocked.csv",
            },
            summary={},
        ),
    )
    monkeypatch.setattr(
        base_rl_lane,
        "run_rl_learning_cycle",
        lambda root=tmp_path, pair_id="", policy_candidates=12: CommandResult(
            paths={
                "summary": tmp_path / "reports" / "rl" / "rl_learning_cycle_summary.csv",
                "best_policy": tmp_path / "reports" / "rl" / "rl_learning_cycle_best_policy.json",
            },
            summary={},
        ),
    )
    monkeypatch.setattr(base_rl_lane, "_resolve_frozen_candidates", lambda root, pair_id="": pd.DataFrame([{"pair": pair_id, "best_execution_venue": "dydx"}]))
    monkeypatch.setattr(base_rl_lane, "_base_rl_pair_coverage", lambda root, learning_summary, candidate_set=None: pd.DataFrame([{"shortlist_rank": 1, "pair": "BTC-USD-LDO-USD", "best_execution_venue": "dydx", "rl_policy_name": "", "rl_focus_count": 0, "rl_supported": False, "evidence_path": ""}]))
    monkeypatch.setattr(base_rl_lane, "base_rl_paper_handoff_report", lambda root=tmp_path: pd.DataFrame([{"status": "research_only", "paper_authorized": False}]))
    monkeypatch.setattr(base_rl_lane, "_build_base_rl_manifest", lambda **kwargs: {"status": "research_only", "candidate_rows": 1})
    monkeypatch.setattr(base_rl_lane, "_write_paper_readiness_files", lambda root, coverage, pair_id: None)
    monkeypatch.setattr(base_rl_lane, "_build_blocker_delta_report", lambda root, current_manifest, previous_manifest: None)
    monkeypatch.setattr(base_rl_lane, "_load_json", lambda path: {})

    for rel in ["reports/rl/research_training.csv", "reports/rl/research_eval.csv", "reports/rl/research_blocked.csv", "reports/rl/rl_learning_cycle_summary.csv"]:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_csv(path, index=False)
    (tmp_path / "reports" / "rl" / "rl_learning_cycle_best_policy.json").write_text("{}", encoding="utf-8")

    result = base_rl_lane.run_base_rl(root=tmp_path, pair_id="BTC-USD-LDO-USD")

    assert result.summary["candidate_rows"] == 1
    assert result.summary["status"] == "research_only"


def test_run_base_rl_summary_uses_authoritative_handoff_status(tmp_path, monkeypatch):
    (tmp_path / "data" / "ml").mkdir(parents=True)
    (tmp_path / "reports" / "rl").mkdir(parents=True)
    (tmp_path / "models" / "rl" / "learning").mkdir(parents=True)
    (tmp_path / "reports" / "ml").mkdir(parents=True)
    (tmp_path / "reports").mkdir(exist_ok=True)

    pd.DataFrame([{"pair": "BTC-USD-LDO-USD"}]).to_csv(tmp_path / "data" / "ml" / "trade_training_dataset.csv", index=False)
    pd.DataFrame([{"gate": "strategy_acceptance", "ready": True}, {"gate": "paper_execution_gate", "ready": True}]).to_csv(
        tmp_path / "reports" / "priority_readiness.csv", index=False
    )
    pd.DataFrame([{"ready": True}, {"ready": True}]).to_csv(tmp_path / "reports" / "paper_execution_preflight.csv", index=False)
    pd.DataFrame([{"accepted": False}]).to_csv(tmp_path / "reports" / "ml" / "model_gated_acceptance.csv", index=False)

    monkeypatch.setattr(base_rl_lane, "build_trade_dataset", lambda root=tmp_path: None)
    monkeypatch.setattr(
        base_rl_lane,
        "run_rl_research",
        lambda root=tmp_path, pair_id="": CommandResult(
            paths={
                "training_report": tmp_path / "reports" / "rl" / "research_training.csv",
                "evaluation_report": tmp_path / "reports" / "rl" / "research_eval.csv",
                "blocked_actions": tmp_path / "reports" / "rl" / "research_blocked.csv",
            },
            summary={},
        ),
    )
    monkeypatch.setattr(
        base_rl_lane,
        "run_rl_learning_cycle",
        lambda root=tmp_path, pair_id="", policy_candidates=12: CommandResult(
            paths={
                "summary": tmp_path / "reports" / "rl" / "rl_learning_cycle_summary.csv",
                "best_policy": tmp_path / "reports" / "rl" / "rl_learning_cycle_best_policy.json",
            },
            summary={},
        ),
    )
    monkeypatch.setattr(base_rl_lane, "_resolve_frozen_candidates", lambda root, pair_id="": pd.DataFrame([{"pair": pair_id, "best_execution_venue": "dydx"}]))
    monkeypatch.setattr(base_rl_lane, "_base_rl_pair_coverage", lambda root, learning_summary, candidate_set=None: pd.DataFrame([{"shortlist_rank": 1, "pair": "BTC-USD-LDO-USD", "best_execution_venue": "dydx", "rl_policy_name": "", "rl_focus_count": 14, "rl_supported": True, "evidence_path": ""}]))
    monkeypatch.setattr(base_rl_lane, "base_rl_paper_handoff_report", lambda root=tmp_path: pd.DataFrame([{"status": "paper_authorized", "paper_authorized": True}]))
    monkeypatch.setattr(base_rl_lane, "_build_base_rl_manifest", lambda **kwargs: {"status": "paper_authorized", "candidate_rows": 1})
    monkeypatch.setattr(base_rl_lane, "_write_paper_readiness_files", lambda root, coverage, pair_id: None)
    monkeypatch.setattr(base_rl_lane, "_build_blocker_delta_report", lambda root, current_manifest, previous_manifest: None)
    monkeypatch.setattr(base_rl_lane, "_load_json", lambda path: {})

    for rel in ["reports/rl/research_training.csv", "reports/rl/research_eval.csv", "reports/rl/research_blocked.csv", "reports/rl/rl_learning_cycle_summary.csv"]:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_csv(path, index=False)
    (tmp_path / "reports" / "rl" / "rl_learning_cycle_best_policy.json").write_text("{}", encoding="utf-8")

    result = base_rl_lane.run_base_rl(root=tmp_path, pair_id="BTC-USD-LDO-USD")

    assert result.summary["status"] == "paper_authorized"
    assert result.summary["paper_authorized"] is True


def test_run_base_rl_reuses_existing_dataset_for_full_run(tmp_path, monkeypatch):
    (tmp_path / "data" / "ml").mkdir(parents=True)
    (tmp_path / "reports" / "rl").mkdir(parents=True)
    (tmp_path / "models" / "rl" / "learning").mkdir(parents=True)
    (tmp_path / "reports" / "ml").mkdir(parents=True)
    (tmp_path / "reports").mkdir(exist_ok=True)

    pd.DataFrame([{"pair": "BTC-USD-LDO-USD"}]).to_csv(tmp_path / "data" / "ml" / "trade_training_dataset.csv", index=False)
    pd.DataFrame([{"gate": "strategy_acceptance", "ready": True}, {"gate": "paper_execution_gate", "ready": True}]).to_csv(
        tmp_path / "reports" / "priority_readiness.csv", index=False
    )
    pd.DataFrame([{"ready": True}, {"ready": True}]).to_csv(tmp_path / "reports" / "paper_execution_preflight.csv", index=False)
    pd.DataFrame([{"accepted": False}]).to_csv(tmp_path / "reports" / "ml" / "model_gated_acceptance.csv", index=False)

    monkeypatch.setattr(
        base_rl_lane,
        "build_trade_dataset",
        lambda root=tmp_path: (_ for _ in ()).throw(AssertionError("build_trade_dataset should not run")),
    )
    monkeypatch.setattr(
        base_rl_lane,
        "run_rl_research",
        lambda root=tmp_path, pair_id="": CommandResult(
            paths={
                "training_report": tmp_path / "reports" / "rl" / "research_training.csv",
                "evaluation_report": tmp_path / "reports" / "rl" / "research_eval.csv",
                "blocked_actions": tmp_path / "reports" / "rl" / "research_blocked.csv",
            },
            summary={},
        ),
    )
    monkeypatch.setattr(
        base_rl_lane,
        "run_rl_learning_cycle",
        lambda root=tmp_path, pair_id="", policy_candidates=12: CommandResult(
            paths={
                "summary": tmp_path / "reports" / "rl" / "rl_learning_cycle_summary.csv",
                "best_policy": tmp_path / "reports" / "rl" / "rl_learning_cycle_best_policy.json",
            },
            summary={},
        ),
    )
    monkeypatch.setattr(base_rl_lane, "_resolve_frozen_candidates", lambda root, pair_id="": pd.DataFrame([{"pair": "BTC-USD-LDO-USD", "best_execution_venue": "dydx"}]))
    monkeypatch.setattr(base_rl_lane, "_base_rl_pair_coverage", lambda root, learning_summary, candidate_set=None: pd.DataFrame([{"shortlist_rank": 1, "pair": "BTC-USD-LDO-USD", "best_execution_venue": "dydx", "rl_policy_name": "", "rl_focus_count": 1, "rl_supported": True, "evidence_path": ""}]))
    monkeypatch.setattr(base_rl_lane, "base_rl_paper_handoff_report", lambda root=tmp_path: pd.DataFrame([{"status": "paper_authorized", "paper_authorized": True}]))
    monkeypatch.setattr(base_rl_lane, "_build_base_rl_manifest", lambda **kwargs: {"status": "paper_authorized", "candidate_rows": 1})
    monkeypatch.setattr(base_rl_lane, "_write_paper_readiness_files", lambda root, coverage, pair_id: None)
    monkeypatch.setattr(base_rl_lane, "_build_blocker_delta_report", lambda root, current_manifest, previous_manifest: None)
    monkeypatch.setattr(base_rl_lane, "_load_json", lambda path: {})

    for rel in ["reports/rl/research_training.csv", "reports/rl/research_eval.csv", "reports/rl/research_blocked.csv", "reports/rl/rl_learning_cycle_summary.csv"]:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_csv(path, index=False)
    (tmp_path / "reports" / "rl" / "rl_learning_cycle_best_policy.json").write_text("{}", encoding="utf-8")

    result = base_rl_lane.run_base_rl(root=tmp_path)

    assert result.summary["status"] == "paper_authorized"
    assert result.summary["paper_authorized"] is True


def test_base_rl_pair_coverage_uses_frozen_candidate_support_fields(tmp_path):
    learning_summary = pd.DataFrame(
        [
            {"policy_name": "winner", "winner": 1, "profit_factor": 3.0, "top_pairs_entered": "BTC-USD-LDO-USD:8"},
        ]
    )
    candidate_set = pd.DataFrame(
        [
            {
                "pair": "BTC-USD-LDO-USD",
                "best_execution_venue": "dydx",
                "evidence_path": "reports/pair.csv",
                "rl_pair_focus_count": 14,
                "pair_model_support_status": "strong_model_support",
                "pair_model_taken_trades": 21,
                "rl_policy_winner": True,
            },
            {
                "pair": "BTC-USD-DYDX-USD",
                "best_execution_venue": "dydx",
                "evidence_path": "reports/pair2.csv",
                "rl_pair_focus_count": 0,
                "pair_model_support_status": "strong_model_support",
                "pair_model_taken_trades": 15,
                "rl_policy_winner": True,
            },
        ]
    )

    frame = _base_rl_pair_coverage(tmp_path, learning_summary, candidate_set)

    assert set(frame["pair"]) == {"BTC-USD-LDO-USD", "BTC-USD-DYDX-USD"}
    assert bool(frame.loc[frame["pair"] == "BTC-USD-LDO-USD", "rl_supported"].iloc[0]) is True
    assert bool(frame.loc[frame["pair"] == "BTC-USD-DYDX-USD", "rl_supported"].iloc[0]) is True


def test_base_rl_feedback_normalizes_outcome_tiers(tmp_path):
    reports = tmp_path / "reports"
    (tmp_path / "data" / "meta_learning").mkdir(parents=True)
    reports.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "timestamp_utc": "2026-06-29T00:00:00Z",
                "pair": "BTC-USD-ETH-USD",
                "strategy_id": "copula_01",
                "intents_json": "{}",
                "plan_status": "paper_submitted",
            }
        ]
    ).to_csv(reports / "paper_trading_journal.csv", index=False)
    (tmp_path / "data" / "meta_learning" / "trades.jsonl").write_text(
        '{"trade_id":"t1","pair":"BTC-USD-ETH-USD","regime":"trend","strategy":"copula_01","signal":{"action":"enter"},"execution":{"submission_type":"paper-completed","quality_tier":"paper-completed"},"outcome":{"realized_return":0.12}}\n',
        encoding="utf-8",
    )

    result = refresh_base_rl_feedback(root=tmp_path)
    frame = pd.read_csv(result.paths["outcome_training_dataset"])

    assert set(frame["outcome_state"]) >= {"paper_submitted", "outcome_verified"}
    summary = pd.read_csv(result.paths["feedback_summary"])
    assert int(summary.iloc[0]["verified_outcomes"]) == 1


def test_base_rl_feedback_treats_unverified_closed_journal_as_audit_only(tmp_path):
    reports = tmp_path / "reports"
    (tmp_path / "data" / "meta_learning").mkdir(parents=True)
    reports.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "timestamp_utc": "2026-06-29T00:00:00Z",
                "pair": "BTC-USD-ETH-USD",
                "strategy_id": "copula_01",
                "intents_json": "{}",
                "plan_status": "paper_completed",
                "realized_return": "0.12",
                "exit_snapshot_json": "{}",
            }
        ]
    ).to_csv(reports / "paper_trading_journal.csv", index=False)
    (tmp_path / "data" / "meta_learning" / "trades.jsonl").write_text("", encoding="utf-8")

    result = refresh_base_rl_feedback(root=tmp_path)
    frame = pd.read_csv(result.paths["outcome_training_dataset"])

    row = frame.iloc[0]
    assert row["outcome_state"] == "audit_only"
    assert pd.isna(row["realized_return"])


def test_base_rl_feedback_keeps_verified_closed_journal_return(tmp_path):
    reports = tmp_path / "reports"
    (tmp_path / "data" / "meta_learning").mkdir(parents=True)
    reports.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "timestamp_utc": "2026-06-29T00:00:00Z",
                "pair": "BTC-USD-ETH-USD",
                "strategy_id": "copula_01",
                "intents_json": "{}",
                "plan_status": "paper_completed",
                "realized_return": "0.12",
                "exit_snapshot_json": json.dumps({"venue_exit_price_x": 100.0}),
            }
        ]
    ).to_csv(reports / "paper_trading_journal.csv", index=False)
    (tmp_path / "data" / "meta_learning" / "trades.jsonl").write_text("", encoding="utf-8")

    result = refresh_base_rl_feedback(root=tmp_path)
    frame = pd.read_csv(result.paths["outcome_training_dataset"])

    row = frame.iloc[0]
    assert row["outcome_state"] == "paper_completed"
    assert float(row["realized_return"]) == 0.12


def test_run_augmented_rl_builds_divergent_comparison_when_research_disables_pairs(tmp_path, monkeypatch):
    reports = tmp_path / "reports"
    (reports / "rl").mkdir(parents=True)

    (tmp_path / "data" / "processed" / "research_knowledge").mkdir(parents=True)
    pd.DataFrame(
        [
            {"pair": "BTC-USD-ETH-USD", "pair_filter": "BTC-USD-ETH-USD", "row_type": "strategy_hint"},
            {"pair": "SOL-USD-ETH-USD", "pair_filter": "SOL-USD-ETH-USD", "row_type": "strategy_hint"},
        ]
    ).to_csv(
        tmp_path / "data" / "processed" / "research_knowledge" / "research_strategy_hints.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {"pair_filter": "BTC-USD-ETH-USD", "confidence": 0.9, "risk_rule": "avoid_pair", "row_type": "risk_prior"}
        ]
    ).to_csv(tmp_path / "data" / "processed" / "research_knowledge" / "research_risk_priors.csv", index=False)
    pd.DataFrame(
        [{"pair_filter": "SOL-USD-ETH-USD", "row_type": "feature_idea", "feature_name": "ewm_speed"}]
    ).to_csv(tmp_path / "data" / "processed" / "research_knowledge" / "research_features.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair": "BTC-USD-ETH-USD",
                "rl_policy_name": "policy_a",
                "rl_supported": True,
            },
            {
                "pair": "SOL-USD-ETH-USD",
                "rl_policy_name": "policy_a",
                "rl_supported": True,
            },
        ]
    ).to_csv(reports / "rl" / "base_rl_pair_coverage.csv", index=False)

    monkeypatch.setattr(
        "quant_platform.rl.base_rl_lane.run_base_rl",
        lambda root=tmp_path, pair_id="": __import__("quant_platform.active_pipeline").active_pipeline.CommandResult(paths={}, summary={}),
    )

    result = run_augmented_rl(root=tmp_path)
    comparison = pd.read_csv(result.paths["comparison_report"])
    assert set(comparison["pair"]) == {"BTC-USD-ETH-USD", "SOL-USD-ETH-USD"}
    assert bool(comparison.loc[comparison["pair"] == "BTC-USD-ETH-USD", "agreement"].astype(bool).iloc[0]) is False
    assert bool(comparison.loc[comparison["pair"] == "BTC-USD-ETH-USD", "augmented_coverage"].iloc[0]) is False


def test_candidate_set_hash_changes_with_route_context(tmp_path):
    frame = pd.DataFrame({"pair": ["BTC-USD-ETH-USD"], "route_context": ["paper_route_default"], "route_blocker": [""]})
    frame2 = pd.DataFrame({"pair": ["BTC-USD-ETH-USD"], "route_context": ["paper_route_alt"], "route_blocker": [""]})
    assert _candidate_set_hash(frame) != _candidate_set_hash(frame2)


def test_dashboard_and_current_state_surface_base_rl_handoff(tmp_path, monkeypatch):
    monkeypatch.setattr(active_pipeline, "ROOT", tmp_path)
    monkeypatch.setattr(active_pipeline, "ACTIVE", tmp_path / "reports" / "active")
    monkeypatch.setattr(active_pipeline, "DASHBOARD", tmp_path / "reports" / "dashboard")
    monkeypatch.setattr(active_pipeline, "ML_REPORTS", tmp_path / "reports" / "ml")
    monkeypatch.setattr(active_pipeline, "DATA_ML", tmp_path / "data" / "ml")
    monkeypatch.setattr(active_pipeline, "MODELS", tmp_path / "models" / "trade_gate")

    (tmp_path / "reports" / "active").mkdir(parents=True)
    (tmp_path / "reports" / "rl").mkdir(parents=True)
    (tmp_path / "reports" / "ml").mkdir(parents=True)
    (tmp_path / "reports" / "brain").mkdir(parents=True)
    (tmp_path / "reports" / "dashboard").mkdir(parents=True)
    (tmp_path / "data" / "processed").mkdir(parents=True)
    (tmp_path / "data" / "ml").mkdir(parents=True)
    (tmp_path / "models" / "trade_gate").mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "pair": "BTC-USD-ETH-USD",
                "decision_bucket": "PROMOTE",
                "combined_score": 90,
                "best_execution_venue": "dydx",
                "acceptance_score": 80,
                "funding_drag_bps": 1.0,
                "available_timeframes": "1d",
                "decision_reason": "ready",
                "evidence_path": "reports/pair.csv",
            }
        ]
    ).to_csv(tmp_path / "data" / "processed" / "pair_universe.csv", index=False)
    pd.DataFrame([{"accepted": False}]).to_csv(tmp_path / "reports" / "ml" / "model_gated_acceptance.csv", index=False)
    pd.DataFrame([{"status": "research_only", "blocker": "pair_specific_rl_support_missing", "paper_authorized": False}]).to_csv(
        tmp_path / "reports" / "rl" / "base_rl_paper_handoff_status.csv", index=False
    )
    pd.DataFrame().to_csv(tmp_path / "reports" / "rl" / "base_rl_training_report.csv", index=False)
    pd.DataFrame().to_csv(tmp_path / "reports" / "rl" / "base_rl_evaluation_report.csv", index=False)
    pd.DataFrame().to_csv(tmp_path / "reports" / "rl" / "base_rl_pair_coverage.csv", index=False)
    pd.DataFrame().to_csv(tmp_path / "reports" / "rl" / "base_rl_blocked_actions.csv", index=False)
    pd.DataFrame().to_csv(tmp_path / "reports" / "brain" / "brain_readiness_report.csv", index=False)
    pd.DataFrame().to_csv(tmp_path / "reports" / "brain" / "brain_candidate_rollup.csv", index=False)
    pd.DataFrame().to_csv(tmp_path / "reports" / "brain" / "paper_readiness_trend.csv", index=False)
    pd.DataFrame([{"check": "artifact:pair_universe.csv", "ready": True, "blocker": "", "evidence_path": "x", "next_action": "x"}]).to_csv(
        tmp_path / "reports" / "active" / "system_check.csv", index=False
    )

    current = active_pipeline.current_state(root=tmp_path)
    dashboard = active_pipeline.build_command_dashboard(root=tmp_path)

    state = pd.read_csv(current.paths["current_state"])
    assert "base_rl_handoff" in set(state["area"])
    assert {"wizard_readiness", "native_readiness", "overall_readiness", "paper_status"}.issubset(set(state["area"]))
    assert {"candidate_id", "setup_identity", "setup_role", "setup_status", "setup_blocker"}.issubset(state.columns)
    command_center = dashboard.paths["command_center"].read_text(encoding="utf-8")
    assert "## Base RL" in command_center
    assert "## Current Setup Truth" in command_center
    assert dashboard.paths["base_rl_paper_handoff_status"].exists()
    assert dashboard.paths["wizard_readiness"].exists()
    assert dashboard.paths["three_brain_contract"].exists()


def test_build_pair_blocker_report_flags_route_support_and_gates(tmp_path):
    reports = tmp_path / "reports" / "rl"
    reports.mkdir(parents=True)

    coverage = pd.DataFrame(
        [
            {
                "pair": "BTC-USD-ETH-USD",
                "best_execution_venue": "dydx",
                "rl_supported": True,
                "execution_compatible": True,
                "rl_policy_name": "policy_alpha",
                "shortlist_reason": "recommended mean reversion",
            },
            {
                "pair": "SOL-USD-ETH-USD",
                "best_execution_venue": "binance",
                "rl_supported": True,
                "execution_compatible": True,
                "rl_policy_name": "policy_alpha",
                "shortlist_reason": "high-quality pair",
            },
            {
                "pair": "DOGE-USD-ETH-USD",
                "best_execution_venue": "dydx",
                "rl_supported": False,
                "execution_compatible": True,
                "rl_policy_name": "policy_alpha",
                "shortlist_reason": "insufficient",
            },
        ]
    )
    handoff = pd.DataFrame(
        [
            {
                "strategy_acceptance_ready": False,
                "paper_execution_ready": True,
                "model_gate_accepted": True,
            }
        ]
    )

    frame = _build_pair_blocker_report(root=tmp_path, coverage=coverage, handoff=handoff)

    assert len(frame) == 3
    row_btc = frame.loc[frame["pair"] == "BTC-USD-ETH-USD"].iloc[0]
    assert bool(row_btc["in_route"]) is True
    assert row_btc["paper_status"] == "global_gate_blocked"
    assert bool(row_btc["paper_eligible"]) is False

    row_sol = frame.loc[frame["pair"] == "SOL-USD-ETH-USD"].iloc[0]
    assert bool(row_sol["in_route"]) is False
    assert row_sol["paper_status"] == "out_of_route"

    row_doge = frame.loc[frame["pair"] == "DOGE-USD-ETH-USD"].iloc[0]
    assert row_doge["coverage_status"] == "unsupported"
    assert row_doge["paper_status"] == "no_rl_support"
    assert "no_rl_support" in str(row_doge["pair_blockers"])


def test_build_pair_blocker_report_prioritizes_execution_incompatible_over_global_gates(tmp_path):
    reports = tmp_path / "reports" / "rl"
    reports.mkdir(parents=True)

    coverage = pd.DataFrame(
        [
            {
                "pair": "BETA-USD-ALPHA-USD",
                "best_execution_venue": "dydx",
                "rl_supported": True,
                "execution_compatible": False,
                "execution_blocker": "unsupported_execution_legs",
            }
        ]
    )
    handoff = pd.DataFrame(
        [
            {
                "strategy_acceptance_ready": False,
                "paper_execution_ready": True,
                "model_gate_accepted": True,
            }
        ]
    )

    frame = _build_pair_blocker_report(root=tmp_path, coverage=coverage, handoff=handoff)
    row = frame.iloc[0]

    assert row["paper_status"] == "execution_incompatible"
    assert "unsupported_execution_legs" in str(row["pair_blockers"])


def test_promotion_decision_report_requires_base_handoff_ready(tmp_path):
    reports = tmp_path / "reports" / "rl"
    reports.mkdir(parents=True)

    coverage = pd.DataFrame(
        [
            {
                "pair": "SOL-USD-HYPE-USD",
                "best_execution_venue": "dydx",
                "rl_supported": True,
            }
        ]
    )
    handoff = pd.DataFrame(
        [
            {
                "status": "research_only",
                "blocker": "strategy_acceptance_not_ready",
            }
        ]
    )

    from quant_platform.rl.base_rl_lane import _write_promotion_decision_report

    _write_promotion_decision_report(root=tmp_path, coverage=coverage, handoff=handoff)
    frame = pd.read_csv(reports / "base_rl_promotion_decision.csv")
    row = frame.iloc[0]

    assert bool(row["base_handoff_ready"]) is False
    assert bool(row["paper_eligible"]) is False
    assert row["reason"] == "strategy_acceptance_not_ready"


def test_comparison_and_promotion_blocks_unsupported_execution_pairs(tmp_path):
    reports = tmp_path / "reports" / "rl"
    reports.mkdir(parents=True)

    coverage = pd.DataFrame(
        [
            {
                "pair": "BTC-USD-ETH-USD",
                "best_execution_venue": "dydx",
                "rl_supported": True,
                "execution_compatible": False,
                "execution_blocker": "route_market_unconfirmed_on_exchange",
            }
        ]
    )
    handoff = pd.DataFrame([{"status": "paper_authorized"}])
    comparison, promotion = _build_comparison_and_promotion(root=tmp_path, coverage=coverage, handoff=handoff)

    assert bool(comparison.iloc[0]["augmented_coverage"]) is False
    assert promotion.iloc[0]["coverage_status"] == "unsupported"
    assert bool(promotion.iloc[0]["paper_eligible"]) is False
    assert str(promotion.iloc[0]["reason"]) == "route_market_unconfirmed_on_exchange"


def test_paper_handoff_pair_support_coverage_requires_execution_compatibility(tmp_path, monkeypatch):
    reports = tmp_path / "reports"
    (reports / "rl").mkdir(parents=True)
    (reports / "ml").mkdir(parents=True)
    (reports / "dashboard").mkdir(parents=True)
    (reports / "active").mkdir(parents=True)

    pd.DataFrame(
        [
            {"gate": "strategy_acceptance", "ready": True},
            {"gate": "paper_execution_gate", "ready": True},
        ]
    ).to_csv(reports / "priority_readiness.csv", index=False)
    pd.DataFrame([{"ready": True}, {"ready": True}]).to_csv(reports / "paper_execution_preflight.csv", index=False)
    pd.DataFrame([{"accepted": True}]).to_csv(reports / "ml" / "model_gated_acceptance.csv", index=False)
    pd.DataFrame(
        [
            {"pair": "BTC-USD-ETH-USD", "best_execution_venue": "dydx", "evidence_path": "reports/pair.csv"},
            {"pair": "SOL-USD-TRX-USD", "best_execution_venue": "dydx", "evidence_path": "reports/pair2.csv"},
        ]
    ).to_csv(reports / "dashboard" / "paper_candidate_shortlist.csv", index=False)
    pd.DataFrame(
        [
            {"pair": "BTC-USD-ETH-USD", "best_execution_venue": "dydx", "evidence_path": "reports/pair.csv"},
            {"pair": "SOL-USD-TRX-USD", "best_execution_venue": "dydx", "evidence_path": "reports/pair2.csv"},
        ]
    ).to_csv(reports / "rl" / "base_rl_route_candidates.csv", index=False)
    pd.DataFrame(
        [
            {
                "policy_name": "winner",
                "winner": 1,
                "profit_factor": 3.0,
                "top_pairs_entered": "BTC-USD-ETH-USD:8",
            }
        ]
    ).to_csv(reports / "rl" / "rl_learning_cycle_summary.csv", index=False)
    pd.DataFrame(
        [
            {"market": "BTC-USD", "compatible_for_paper_submit": True},
            {"market": "ETH-USD", "compatible_for_paper_submit": False},
            {"market": "SOL-USD", "compatible_for_paper_submit": True},
            {"market": "TRX-USD", "compatible_for_paper_submit": True},
        ]
    ).to_csv(reports / "active" / "dydx_execution_market_compatibility.csv", index=False)
    pd.DataFrame(
        [
            {"pair": "BTC-USD-ETH-USD", "support_status": "strong_model_support"},
            {"pair": "SOL-USD-TRX-USD", "support_status": "strong_model_support"},
        ]
    ).to_csv(reports / "ml" / "model_gate_pair_support_report.csv", index=False)
    monkeypatch.setattr(
        base_rl_lane,
        "effective_dydx_account_state_snapshot",
        lambda config, root=tmp_path: {"checked": True, "open_markets": [], "positions": [], "blocker": "", "source": "dydx_indexer"},
    )

    frame = base_rl_paper_handoff_report(root=tmp_path)

    assert frame.iloc[0]["shortlist_pairs"] == 2
    assert frame.iloc[0]["pairs_with_rl_support"] == 1
    assert frame.iloc[0]["pair_support_coverage"] == 0.5
    assert frame.iloc[0]["status"] == "research_only"
    assert frame.iloc[0]["blocker"] == "pair_specific_rl_support_missing"


def test_resolve_frozen_candidates_requires_execution_compatibility(tmp_path, monkeypatch):
    monkeypatch.setattr(
        base_rl_lane,
        "paper_candidate_shortlist_rows",
        lambda root, require_execution_compatible=True: pd.DataFrame(
            [
                {
                    "pair": "BTC-USD-ETH-USD",
                    "execution_compatible": True,
                    "best_execution_venue": "dydx",
                },
                {
                    "pair": "SOL-USD-TRX-USD",
                    "execution_compatible": False,
                    "best_execution_venue": "dydx",
                },
            ]
        ),
    )

    frame = base_rl_lane._resolve_frozen_candidates(root=tmp_path)

    assert len(frame) == 1
    assert set(frame["pair"]) == {"BTC-USD-ETH-USD"}
