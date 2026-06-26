from __future__ import annotations

import json

import pandas as pd

from quant_platform.wizard_evidence import (
    DISCOVERY_MIN_RETURNS_TOTAL,
    DISCOVERY_MIN_SHARPE,
    build_wizard_diagnostic_confirmation,
    build_wizard_evidence,
    build_wizard_exact_mode_capture_queue,
    build_wizard_hypotheses,
    build_wizard_local_parity,
    ids_from_exact_mode,
    mode_from_ids,
)


def test_exact_mode_id_mapping_round_trips():
    assert mode_from_ids(3, 1) == "Static (Spread)"
    assert mode_from_ids(1, 3) == "Copula"
    assert ids_from_exact_mode("OU (ZScoreR)") == (2, 2)


def test_missing_exact_mode_is_blocked(tmp_path):
    pair_dir = tmp_path / "data" / "raw" / "pair_details"
    pair_dir.mkdir(parents=True)
    _write_json(
        pair_dir / "pair_SOL-USD_WLD-USD_wizard.json",
        {
            "pair": "SOL-USD/WLD-USD",
            "asset_x": "SOL-USD",
            "asset_y": "WLD-USD",
            "interval": "daily",
            "sharpe": 3.1,
            "returns_total": 0.42,
        },
    )

    result = build_wizard_evidence(root=tmp_path)
    frame = pd.read_csv(result.paths["wizard_evidence"])

    assert len(frame) == 1
    assert not bool(frame["mode_valid"].iloc[0])
    assert frame["mode_blocker"].iloc[0] == "missing_exact_mode"


def test_wizard_only_evidence_cannot_promote(tmp_path):
    pair_dir = tmp_path / "data" / "raw" / "pair_details"
    pair_dir.mkdir(parents=True)
    _write_json(
        pair_dir / "pair_SOL-USD_WLD-USD_wizard.json",
        {
            "pair": "SOL-USD/WLD-USD",
            "asset_x": "SOL-USD",
            "asset_y": "WLD-USD",
            "interval": "daily",
            "spread_id": 2,
            "strategy_id": 1,
            "sharpe": 3.1,
            "returns_total": 0.42,
        },
    )

    result = build_wizard_hypotheses(root=tmp_path)
    frame = pd.read_csv(result.paths["hypotheses"])

    assert frame["hypothesis_status"].iloc[0] == "NEEDS_LOCAL_DATA"
    assert "wizard_only_evidence_cannot_promote" in frame["hypothesis_reason"].iloc[0]


def test_discovery_sharpe_gate_is_one_point_seventy_five_or_above(tmp_path):
    pair_dir = tmp_path / "data" / "raw" / "pair_details"
    pair_dir.mkdir(parents=True)
    _write_json(
        pair_dir / "pair_BLUR-USD_ETHFI-USD_wizard.json",
        {
            "pair": "BLUR-USD/ETHFI-USD",
            "asset_x": "BLUR-USD",
            "asset_y": "ETHFI-USD",
            "interval": "daily",
            "spread_id": 3,
            "strategy_id": 1,
            "sharpe": DISCOVERY_MIN_SHARPE,
            "returns_total": 0.21,
        },
    )

    result = build_wizard_evidence(root=tmp_path)
    frame = pd.read_csv(result.paths["wizard_evidence"])

    assert bool(frame["passes_sharpe_gate"].iloc[0])
    assert bool(frame["passes_sharpe_gt_2"].iloc[0])
    assert float(frame["discovery_min_sharpe"].iloc[0]) == DISCOVERY_MIN_SHARPE


def test_discovery_returns_gate_is_above_ten_percent(tmp_path):
    pair_dir = tmp_path / "data" / "raw" / "pair_details"
    pair_dir.mkdir(parents=True)
    _write_json(
        pair_dir / "pair_ETH-USD_FIL-USD_wizard.json",
        {
            "pair": "ETH-USD/FIL-USD",
            "asset_x": "ETH-USD",
            "asset_y": "FIL-USD",
            "interval": "daily",
            "spread_id": 3,
            "strategy_id": 1,
            "sharpe": 2.0,
            "returns_total": DISCOVERY_MIN_RETURNS_TOTAL + 0.001,
        },
    )

    result = build_wizard_evidence(root=tmp_path)
    frame = pd.read_csv(result.paths["wizard_evidence"])

    assert bool(frame["passes_returns_total_gt_20pct"].iloc[0])
    assert float(frame["discovery_min_returns_total"].iloc[0]) == DISCOVERY_MIN_RETURNS_TOTAL


def test_missing_ecm_creates_diagnostic_blocker(tmp_path):
    evidence_dir = tmp_path / "data" / "processed"
    evidence_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "pair": "SOL-USD/WLD-USD",
                "interval": "daily",
                "exact_mode": "OU (Spread)",
                "pearson": 0.8,
                "spearman": 0.76,
                "kendall": 0.6,
                "hurst": 0.42,
                "half_life": 12,
                "ecm_x_available": False,
                "ecm_y_available": False,
                "ecm_strength_available": False,
                "evidence_path": "wizard",
            }
        ]
    ).to_csv(evidence_dir / "wizard_evidence.csv", index=False)

    result = build_wizard_diagnostic_confirmation(root=tmp_path)
    frame = pd.read_csv(result.paths["diagnostics"])

    assert frame["ecm_status"].iloc[0] == "missing_ecm"
    assert "missing_ecm" in frame["diagnostic_blocker"].iloc[0]


def test_local_parity_flags_zscore_mismatch(tmp_path):
    processed = tmp_path / "data" / "processed"
    raw = tmp_path / "data" / "raw" / "pair_details"
    processed.mkdir(parents=True)
    raw.mkdir(parents=True)
    wizard_path = raw / "pair_SOL-USD_WLD-USD_wizard.json"
    local_path = raw / "pair_SOL-USD_WLD-USD_daily_dydx_derived_history.json"
    _write_json(
        wizard_path,
        {
            "pair": "SOL-USD/WLD-USD",
            "asset_x": "SOL-USD",
            "asset_y": "WLD-USD",
            "interval": "daily",
            "history": [{"zscore": 2.4, "rolling_zscore": 2.2}],
        },
    )
    _write_json(
        local_path,
        {
            "pair": "SOL-USD/WLD-USD",
            "asset_x": "SOL-USD",
            "asset_y": "WLD-USD",
            "interval": "daily",
            "history": [{"zscore": 0.1, "rolling_zscore": 0.0}],
        },
    )
    pd.DataFrame(
        [
            {
                "pair": "SOL-USD/WLD-USD",
                "asset_x": "SOL-USD",
                "asset_y": "WLD-USD",
                "interval": "daily",
                "exact_mode": "OU (Spread)",
                "source_path": str(wizard_path),
            }
        ]
    ).to_csv(processed / "wizard_evidence.csv", index=False)

    result = build_wizard_local_parity(root=tmp_path)
    frame = pd.read_csv(result.paths["parity"])

    assert frame["parity_status"].iloc[0] == "MISMATCH"


def test_exact_mode_capture_queue_keeps_only_high_sharpe_return_blockers(tmp_path):
    processed = tmp_path / "data" / "processed"
    raw = tmp_path / "data" / "raw" / "pair_details"
    processed.mkdir(parents=True)
    raw.mkdir(parents=True)
    source = raw / "pair_one.json"
    _write_json(source, {"url": "https://cryptowizards.net/wizards/zscore/pair/1?origin=scanner"})
    pd.DataFrame(
        [
            {
                "pair": "BNB-USD/STX-USD",
                "asset_x": "BNB-USD",
                "asset_y": "STX-USD",
                "interval": "daily",
                "sharpe": 3.0,
                "returns_total": 0.54,
                "returns_total_pct": 54.0,
                "mode_valid": False,
                "passes_sharpe_gt_2": True,
                "passes_returns_total_gt_20pct": True,
                "mode_blocker": "missing_exact_mode",
                "source_path": str(source),
                "evidence_path": str(source),
            },
            {
                "pair": "LOW-USD/RET-USD",
                "asset_x": "LOW-USD",
                "asset_y": "RET-USD",
                "interval": "daily",
                "sharpe": 1.5,
                "returns_total": 0.54,
                "returns_total_pct": 54.0,
                "mode_valid": False,
                "passes_sharpe_gt_2": False,
                "passes_returns_total_gt_20pct": True,
                "mode_blocker": "missing_exact_mode",
                "source_path": "",
                "evidence_path": "wizard",
            },
        ]
    ).to_csv(processed / "wizard_evidence.csv", index=False)

    result = build_wizard_exact_mode_capture_queue(root=tmp_path)
    frame = pd.read_csv(result.paths["exact_mode_capture_queue"])

    assert frame["pair"].tolist() == ["BNB-USD/STX-USD"]
    assert frame["pair_page_url"].iloc[0].endswith("/pair/1?origin=scanner")
    assert "selected_strategy_value" in frame["required_capture_fields"].iloc[0]


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
