from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _load_phase9():
    path = ROOT / "scripts" / "run_evidence_pipeline_phase9_final_promotion.py"
    spec = importlib.util.spec_from_file_location("legacy_phase9", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_legacy_v1_pipeline_cannot_create_paper_trade_authority():
    module = _load_phase9()
    best = pd.Series(
        {
            "walk_forward_pass": True,
            "profit_factor": 2.0,
            "test_profit_factor": 1.5,
        }
    )

    label, reason, next_action = module.final_label("ETH-SOL", best, 1.5, True)

    assert module.LEGACY_PIPELINE_AUTHORITY == "HISTORICAL_RESEARCH_ONLY"
    assert label == "research_only"
    assert "no acceptance or paper-trade authority" in reason
    assert "canonical V2" in next_action


def test_curl_helper_does_not_serialize_wizard_key_into_process_environment():
    source = (ROOT / "scripts" / "crawl_crypto_wizards_with_curl.sh").read_text(
        encoding="utf-8"
    )

    assert '"api_key": config.api_key' not in source
    assert 'CONFIG_JSON="$CONFIG_JSON"' not in source
    assert 'API_KEY="$(' not in source
