"""Canonical contracts shared by current and exhaustive Wizard-Hyperliquid lanes."""

from __future__ import annotations

from quant_platform.orchestration.corrective_runtime import atomic_write_text

from quant_platform.orchestration.corrective_runtime import atomic_write_csv

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from quant_platform.wizard_mode_replay import CANONICAL_WIZARD_MODES

SCHEMA_VERSION = "thewiz.canonical_wizard_hyperliquid_contract.v1"
ORIENTATIONS = ("original", "reverse")
STRICT_L2_WINDOW_HOURS = 2.0
PROVISIONAL_L2_WINDOW_HOURS = 24.0
MINIMUM_STRICT_L2_SAMPLES = 12
MINIMUM_PROVISIONAL_L2_SAMPLES = 3
MINIMUM_FUNDING_COVERAGE = 0.95
MINIMUM_PROVISIONAL_FUNDED_ROWS = 250
MAXIMUM_FEE_EVIDENCE_AGE_DAYS = 30
REFERENCE_LEG_NOTIONAL_USD = 1_000.0
CAPACITY_NOTIONALS_USD = (100.0, 500.0, 1_000.0, 5_000.0, 10_000.0)
MINIMUM_SHARPE_PROBABILITY = 0.95
FALSE_DISCOVERY_RATE = 0.10

PIPELINE_STAGES = (
    ("wizard_discovery", "crypto_wizards", "discovery_only"),
    ("mode_contract", "quant_platform.wizard_mode_replay", "research_only"),
    ("point_in_time_history", "hyperliquid_public_info", "acceptance_input"),
    ("funding_alignment", "hyperliquid_public_funding", "acceptance_input"),
    ("capacity_cost", "quant_platform.hyperliquid", "acceptance_input"),
    (
        "walkforward",
        "quant_platform.orchestration.exhaustive_wizard_hyperliquid_walkforward",
        "acceptance_input",
    ),
    (
        "regime_attribution",
        "quant_platform.orchestration.exhaustive_wizard_hyperliquid_regimes",
        "acceptance_input",
    ),
    (
        "robustness",
        "quant_platform.orchestration.exhaustive_wizard_hyperliquid_robustness",
        "acceptance_input",
    ),
    ("statistical_selection", "quant_platform.statistical_validation", "acceptance_input"),
    ("testnet_lifecycle", "hyperliquid_testnet", "signed_bounded_authority_required"),
    ("live", "hyperliquid_mainnet", "disabled"),
)


def build_canonical_pipeline_contract(
    *, root: Path, now: datetime | None = None
) -> dict[str, object]:
    generated = _as_utc(now or datetime.now(timezone.utc))
    active = root / "reports" / "active"
    active.mkdir(parents=True, exist_ok=True)
    json_path = active / "canonical_wizard_hyperliquid_contract.json"
    csv_path = active / "canonical_wizard_hyperliquid_stages.csv"
    markdown_path = active / "canonical_wizard_hyperliquid_contract.md"
    policy = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": generated.isoformat(),
        "exact_modes": list(CANONICAL_WIZARD_MODES),
        "orientations": list(ORIENTATIONS),
        "cost_policy": {
            "strict_l2_window_hours": STRICT_L2_WINDOW_HOURS,
            "provisional_l2_window_hours": PROVISIONAL_L2_WINDOW_HOURS,
            "minimum_strict_l2_samples": MINIMUM_STRICT_L2_SAMPLES,
            "minimum_provisional_l2_samples": MINIMUM_PROVISIONAL_L2_SAMPLES,
            "minimum_funding_coverage": MINIMUM_FUNDING_COVERAGE,
            "minimum_provisional_funded_rows": MINIMUM_PROVISIONAL_FUNDED_ROWS,
            "maximum_fee_evidence_age_days": MAXIMUM_FEE_EVIDENCE_AGE_DAYS,
            "reference_leg_notional_usd": REFERENCE_LEG_NOTIONAL_USD,
            "capacity_notionals_usd": list(CAPACITY_NOTIONALS_USD),
            "volume_24h_is_executable_liquidity": False,
        },
        "statistical_policy": {
            "minimum_sharpe_probability": MINIMUM_SHARPE_PROBABILITY,
            "false_discovery_rate": FALSE_DISCOVERY_RATE,
            "fold_count_is_not_deflated_sharpe": True,
            "dependent_return_inference_required": True,
        },
        "authority": {
            "wizard_promotion_authority": False,
            "education_promotion_authority": False,
            "testnet_requires_signed_bounded_authority": True,
            "live_trading_authorized": False,
        },
        "stages": [
            {
                "sequence": index,
                "stage": stage,
                "implementation": implementation,
                "authority": authority,
            }
            for index, (stage, implementation, authority) in enumerate(PIPELINE_STAGES, 1)
        ],
    }
    atomic_write_text(json_path, json.dumps(policy, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    stages = pd.DataFrame(policy["stages"])
    atomic_write_csv(stages, csv_path, index=False)
    atomic_write_text(markdown_path, _markdown(policy, stages), encoding="utf-8")
    return {
        "contract": json_path,
        "stages": csv_path,
        "markdown": markdown_path,
        "stage_count": len(stages),
        "exact_mode_cells_per_pair": len(CANONICAL_WIZARD_MODES) * len(ORIENTATIONS),
        "live_trading_authorized": False,
    }


def _markdown(policy: dict[str, object], stages: pd.DataFrame) -> str:
    cost = policy["cost_policy"]
    statistics = policy["statistical_policy"]
    return "\n".join(
        [
            "# Canonical Wizard-Hyperliquid Contract",
            "",
            "Current-board and exhaustive orchestration are views over these shared contracts.",
            "A view may bound network work, but it may not change mode math, costs, "
            "statistics, or authority.",
            "",
            f"- Exact modes: `{len(policy['exact_modes'])}`",
            f"- Orientations: `{len(policy['orientations'])}`",
            f"- Strict L2 samples per leg/notional: `{cost['minimum_strict_l2_samples']}`",
            f"- Funding coverage: `{cost['minimum_funding_coverage']:.0%}`",
            f"- PSR/DSR probability gate: `{statistics['minimum_sharpe_probability']:.0%}`",
            "- Live authority: `false`",
            "",
            "## Stages",
            "",
            stages.to_markdown(index=False),
            "",
        ]
    )


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
