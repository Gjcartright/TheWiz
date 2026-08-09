"""Cross-cell concentration controls for the current Wizard board."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd

from quant_platform.active_pipeline import CommandResult
from quant_platform.orchestration.snapshot_lineage import (
    existing_snapshot_reference,
    unique_file_bytes,
    verified_snapshot_reference,
)
from quant_platform.orchestration.exhaustive_wizard_hyperliquid_concentration import (
    COHORTS,
    DIMENSION_POLICIES,
    _as_utc,
    _canonical_json,
    _deduplicate,
    _evaluate_cohort,
    _file_hash,
    _id_set,
    _read_csv,
    _read_json,
    _relative,
    _require_unique,
    _status_counts,
    _status_frame,
    _text,
    _truthy,
)


ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "current_wizard_hyperliquid_concentration.v1"
RESEARCH_ONLY_REASON = (
    "concentration_is_research_only;"
    "current_l2_depth_is_point_in_time_not_historical_execution_evidence;"
    "mode_fidelity_parity_not_proven;testnet_lifecycle_not_proven"
)


def build_current_wizard_hyperliquid_concentration(
    *,
    root: Path = ROOT,
    now: datetime | None = None,
) -> CommandResult:
    """Measure concentration without changing any canonical 1x result."""

    as_of = _as_utc(now or datetime.now(timezone.utc))
    active = root / "reports" / "active"
    manifest_paths = {
        "robustness_manifest": active
        / "current_wizard_hyperliquid_robustness_manifest.json",
        "walkforward_manifest": active
        / "current_wizard_hyperliquid_walkforward_manifest.json",
        "regime_manifest": active
        / "current_wizard_hyperliquid_regime_manifest.json",
    }
    missing_manifests = [str(path) for path in manifest_paths.values() if not path.exists()]
    if missing_manifests:
        raise FileNotFoundError(
            "Current concentration manifests missing: " + ",".join(missing_manifests)
        )

    robustness_manifest = _read_json(manifest_paths["robustness_manifest"])
    walkforward_manifest = _read_json(manifest_paths["walkforward_manifest"])
    regime_manifest = _read_json(manifest_paths["regime_manifest"])
    walkforward_id = _text(robustness_manifest.get("walkforward_id"))
    robustness_id = _text(robustness_manifest.get("robustness_id"))
    regime_attribution_id = _text(regime_manifest.get("regime_attribution_id"))
    if not walkforward_id or not robustness_id or not regime_attribution_id:
        raise ValueError("Current concentration requires non-empty upstream identities")
    if walkforward_id != _text(walkforward_manifest.get("walkforward_id")):
        raise ValueError("Current robustness and walk-forward identities do not match")
    if walkforward_id != _text(regime_manifest.get("walkforward_id")):
        raise ValueError("Current regime and walk-forward identities do not match")

    robustness_artifacts = robustness_manifest.get("artifacts", {})
    walkforward_artifacts = walkforward_manifest.get("artifacts", {})
    regime_artifacts = regime_manifest.get("artifacts", {})
    input_paths = {
        "robustness_status": root
        / _text(robustness_artifacts.get("snapshot_status")),
        "robustness_candidates": root
        / _text(robustness_artifacts.get("snapshot_candidates")),
        "walkforward_candidates": root
        / _text(walkforward_artifacts.get("snapshot_candidates")),
        "walkforward_trades": root
        / _text(walkforward_artifacts.get("snapshot_trades")),
        "regime_detail": root / _text(regime_artifacts.get("snapshot_detail")),
        **manifest_paths,
    }
    missing_inputs = [str(path) for path in input_paths.values() if not path.exists()]
    if missing_inputs:
        raise FileNotFoundError(
            "Current concentration inputs missing: " + ",".join(missing_inputs)
        )

    statuses = _read_csv(input_paths["robustness_status"])
    robustness = _read_csv(input_paths["robustness_candidates"])
    candidates = _read_csv(input_paths["walkforward_candidates"])
    trades = _read_csv(input_paths["walkforward_trades"])
    regimes = _read_csv(input_paths["regime_detail"])
    _require_unique(statuses, "experiment_id")
    _require_unique(robustness, "experiment_id")
    _require_unique(candidates, "experiment_id")
    if statuses.empty:
        raise ValueError("Current concentration requires non-empty all-cell status evidence")

    experiment_ids = set(statuses["experiment_id"].astype(str))
    candidate_ids = set(candidates.get("experiment_id", pd.Series(dtype=str)).astype(str))
    robustness_ids = set(
        robustness.get("experiment_id", pd.Series(dtype=str)).astype(str)
    )
    if not candidate_ids.issubset(experiment_ids) or not robustness_ids.issubset(
        experiment_ids
    ):
        raise ValueError("Current concentration candidate IDs escape all-cell authority")

    cohort_ids = {
        "practical_walkforward_pass": _id_set(
            candidates,
            candidates.get(
                "walkforward_status", pd.Series("", index=candidates.index)
            ).eq("PASS_RESEARCH_WALK_FORWARD"),
        ),
        "research_robustness_pass": _id_set(
            robustness,
            robustness.get(
                "research_robustness_status", pd.Series("", index=robustness.index)
            ).eq("PASS_RESEARCH_ROBUSTNESS"),
        ),
        "statistically_selected_robustness_pass": _id_set(
            robustness,
            robustness.get(
                "research_robustness_status", pd.Series("", index=robustness.index)
            ).eq("PASS_RESEARCH_ROBUSTNESS")
            & robustness.get(
                "statistical_selection_status", pd.Series("", index=robustness.index)
            ).eq("PASS"),
        ),
    }
    if not cohort_ids["research_robustness_pass"].issubset(
        cohort_ids["practical_walkforward_pass"]
    ):
        raise ValueError("Current robustness cohort is not a walk-forward subset")
    if not cohort_ids["statistically_selected_robustness_pass"].issubset(
        cohort_ids["research_robustness_pass"]
    ):
        raise ValueError("Current statistical cohort is not a robustness subset")

    policy = {
        "cohorts": list(COHORTS),
        "dimensions": DIMENSION_POLICIES,
        "promotion_cohort": "statistically_selected_robustness_pass",
        "promotion_requires_nonempty_cohort": True,
        "promotion_requires_all_dimensions": True,
        "canonical_1x_signal_unchanged": True,
        "acceptance_authority": False,
    }
    material = {
        "schema_version": SCHEMA_VERSION,
        "walkforward_id": walkforward_id,
        "regime_attribution_id": regime_attribution_id,
        "robustness_id": robustness_id,
        "as_of": as_of.isoformat(),
        "policy": policy,
        "input_hashes": {name: _file_hash(path) for name, path in input_paths.items()},
    }
    digest = sha256(_canonical_json(material).encode("utf-8")).hexdigest()
    concentration_id = "cwconcentration_" + digest[:20]
    robustness_snapshot_manifest = root / _text(
        robustness_artifacts.get("snapshot_manifest")
    )
    if not robustness_snapshot_manifest.exists():
        raise FileNotFoundError("Current robustness snapshot manifest is missing")
    snapshot_dir = (
        robustness_snapshot_manifest.parent / "concentration" / concentration_id
    )
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_inputs = {
        "robustness_status": existing_snapshot_reference(
            root=root, snapshot_path=input_paths["robustness_status"]
        ),
        "robustness_candidates": existing_snapshot_reference(
            root=root, snapshot_path=input_paths["robustness_candidates"]
        ),
        "walkforward_candidates": existing_snapshot_reference(
            root=root, snapshot_path=input_paths["walkforward_candidates"]
        ),
        "walkforward_trades": existing_snapshot_reference(
            root=root, snapshot_path=input_paths["walkforward_trades"]
        ),
        "regime_detail": existing_snapshot_reference(
            root=root, snapshot_path=input_paths["regime_detail"]
        ),
        "robustness_manifest": verified_snapshot_reference(
            root=root,
            active_path=manifest_paths["robustness_manifest"],
            upstream_manifest=robustness_manifest,
            artifact_key="manifest",
        ),
        "walkforward_manifest": verified_snapshot_reference(
            root=root,
            active_path=manifest_paths["walkforward_manifest"],
            upstream_manifest=walkforward_manifest,
            artifact_key="manifest",
        ),
        "regime_manifest": verified_snapshot_reference(
            root=root,
            active_path=manifest_paths["regime_manifest"],
            upstream_manifest=regime_manifest,
            artifact_key="manifest",
        ),
    }

    dimension_rows: list[dict[str, object]] = []
    contributor_rows: list[dict[str, object]] = []
    cohort_rows: list[dict[str, object]] = []
    for cohort in COHORTS:
        dimensions, contributors = _evaluate_cohort(
            cohort=cohort,
            experiment_ids=cohort_ids[cohort],
            candidates=candidates,
            trades=trades,
            regimes=regimes,
            concentration_id=concentration_id,
        )
        _normalize_reused_rows(dimensions)
        _normalize_reused_rows(contributors)
        dimension_rows.extend(dimensions)
        contributor_rows.extend(contributors)
        passed = bool(
            cohort_ids[cohort]
            and len(dimensions) == len(DIMENSION_POLICIES)
            and all(_truthy(row["dimension_gate_pass"]) for row in dimensions)
        )
        blockers = _deduplicate(
            [
                _text(row.get("dimension_blocker"))
                for row in dimensions
                if _text(row.get("dimension_blocker"))
            ]
        )
        if not cohort_ids[cohort]:
            blockers = ["empty_cohort"]
        cohort_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "concentration_id": concentration_id,
                "cohort": cohort,
                "candidate_count": len(cohort_ids[cohort]),
                "dimensions_required": len(DIMENSION_POLICIES),
                "dimensions_complete": len(dimensions),
                "dimensions_passed": sum(
                    _truthy(row["dimension_gate_pass"]) for row in dimensions
                ),
                "cohort_concentration_status": (
                    "PASS_RESEARCH_CONCENTRATION"
                    if passed
                    else (
                        "BLOCKED_EMPTY_COHORT"
                        if not cohort_ids[cohort]
                        else "FAIL_RESEARCH_CONCENTRATION"
                    )
                ),
                "cohort_concentration_blocker": ";".join(blockers),
                "promotion_cohort": cohort
                == "statistically_selected_robustness_pass",
                "ready_for_next_research_gate": bool(
                    passed and cohort == "statistically_selected_robustness_pass"
                ),
                "acceptance_status": "BLOCKED",
                "acceptance_reason": RESEARCH_ONLY_REASON,
                "live_trading_authorized": False,
            }
        )

    cohort_frame = pd.DataFrame(cohort_rows)
    dimension_frame = pd.DataFrame(dimension_rows)
    contributor_frame = pd.DataFrame(contributor_rows)
    promotion_row = cohort_frame.loc[
        cohort_frame["cohort"].eq("statistically_selected_robustness_pass")
    ].iloc[0]
    promotion_pass = _truthy(promotion_row["ready_for_next_research_gate"])

    helper_statuses = statuses.copy()
    helper_statuses["pair_group_id"] = helper_statuses.get(
        "pair_group_key", pd.Series("", index=helper_statuses.index)
    )
    status_frame = _status_frame(
        helper_statuses,
        robust_ids=cohort_ids["research_robustness_pass"],
        selected_ids=cohort_ids["statistically_selected_robustness_pass"],
        concentration_id=concentration_id,
        promotion_pass=promotion_pass,
        promotion_blocker=_text(promotion_row["cohort_concentration_blocker"]),
        evidence_paths=snapshot_inputs.values(),
        root=root,
    ).rename(columns={"pair_group_id": "pair_group_key"})
    _normalize_reused_frame(status_frame)
    validation = _validation(
        statuses=statuses,
        status_frame=status_frame,
        cohort_frame=cohort_frame,
        dimension_frame=dimension_frame,
        cohort_ids=cohort_ids,
        promotion_pass=promotion_pass,
    )
    if not validation["status"].eq("PASS").all():
        failed = validation.loc[validation["status"].ne("PASS"), "check"].tolist()
        raise ValueError("Current concentration validation failed: " + ",".join(failed))

    paths = _paths(active, snapshot_dir)
    for frame, active_key, snapshot_key in (
        (status_frame, "status", "snapshot_status"),
        (cohort_frame, "cohorts", "snapshot_cohorts"),
        (dimension_frame, "dimensions", "snapshot_dimensions"),
        (contributor_frame, "contributors", "snapshot_contributors"),
        (validation, "validation", "snapshot_validation"),
    ):
        frame.to_csv(paths[active_key], index=False)
        frame.to_csv(paths[snapshot_key], index=False)

    status_counts = _status_counts(status_frame, "concentration_status")
    summary: dict[str, object] = {
        **material,
        "concentration_id": concentration_id,
        "experiments_accounted": int(len(status_frame)),
        "unique_experiment_ids": int(status_frame["experiment_id"].nunique()),
        "status_counts": status_counts,
        "experiment_status_accounted": bool(
            sum(status_counts.values()) == len(status_frame) == len(statuses)
        ),
        "practical_walkforward_candidates": len(
            cohort_ids["practical_walkforward_pass"]
        ),
        "research_robustness_candidates": len(
            cohort_ids["research_robustness_pass"]
        ),
        "statistically_selected_robustness_candidates": len(
            cohort_ids["statistically_selected_robustness_pass"]
        ),
        "cohort_rows": int(len(cohort_frame)),
        "dimension_rows": int(len(dimension_frame)),
        "contributor_rows": int(len(contributor_frame)),
        "promotion_concentration_pass": promotion_pass,
        "ready_for_leverage_gate": promotion_pass,
        "acceptance_eligible_replays": 0,
        "testnet_order_authority": False,
        "promotion_authority": False,
        "live_trading_authorized": False,
        "artifacts": {name: _relative(path, root) for name, path in paths.items()},
        "input_snapshots": {
            name: _relative(path, root) for name, path in snapshot_inputs.items()
        },
        "input_snapshot_modes": {
            name: "verified_upstream_reference" for name in snapshot_inputs
        },
        "referenced_upstream_bytes": unique_file_bytes(
            list(snapshot_inputs.values())
        ),
        "locally_copied_input_bytes": 0,
    }
    manifest_text = json.dumps(summary, indent=2, sort_keys=True)
    summary_text = _summary_markdown(summary, cohort_frame, dimension_frame)
    for path in (paths["manifest"], paths["snapshot_manifest"]):
        path.write_text(manifest_text, encoding="utf-8")
    for path in (paths["summary_md"], paths["snapshot_summary_md"]):
        path.write_text(summary_text, encoding="utf-8")
    return CommandResult(paths=paths, summary=summary)


def _normalize_reused_rows(rows: list[dict[str, object]]) -> None:
    for row in rows:
        row["schema_version"] = SCHEMA_VERSION
        if "acceptance_reason" in row:
            row["acceptance_reason"] = RESEARCH_ONLY_REASON


def _normalize_reused_frame(frame: pd.DataFrame) -> None:
    frame["schema_version"] = SCHEMA_VERSION
    frame["acceptance_reason"] = RESEARCH_ONLY_REASON
    frame["acceptance_status"] = "BLOCKED"
    frame["acceptance_eligible"] = False
    frame["live_trading_authorized"] = False
    frame["evidence_path"] = (
        "reports/active/current_wizard_hyperliquid_concentration_manifest.json"
    )


def _validation(
    *,
    statuses: pd.DataFrame,
    status_frame: pd.DataFrame,
    cohort_frame: pd.DataFrame,
    dimension_frame: pd.DataFrame,
    cohort_ids: dict[str, set[str]],
    promotion_pass: bool,
) -> pd.DataFrame:
    authority_ids = set(statuses["experiment_id"].astype(str))
    output_ids = set(status_frame["experiment_id"].astype(str))
    selected_ids = cohort_ids["statistically_selected_robustness_pass"]
    checks = {
        "experiment_count_preserved": len(status_frame) == len(statuses),
        "experiment_ids_unique": status_frame["experiment_id"].nunique()
        == len(statuses),
        "experiment_id_set_preserved": output_ids == authority_ids,
        "cohorts_complete": set(cohort_frame["cohort"].astype(str)) == set(COHORTS),
        "dimensions_complete_per_cohort": len(dimension_frame)
        == len(COHORTS) * len(DIMENSION_POLICIES),
        "selected_subset_robust": selected_ids.issubset(
            cohort_ids["research_robustness_pass"]
        ),
        "empty_promotion_cohort_cannot_pass": bool(selected_ids)
        or not promotion_pass,
        "ready_rows_require_promotion_pass": (
            ~status_frame["ready_for_leverage_gate"].map(_truthy)
            | promotion_pass
        ).all(),
        "acceptance_disabled": status_frame["acceptance_status"].eq("BLOCKED").all()
        and not status_frame["acceptance_eligible"].map(_truthy).any(),
        "live_trading_disabled": not status_frame["live_trading_authorized"]
        .map(_truthy)
        .any(),
    }
    return pd.DataFrame(
        [
            {"check": check, "status": "PASS" if passed else "FAIL"}
            for check, passed in checks.items()
        ]
    )


def _paths(active: Path, snapshot: Path) -> dict[str, Path]:
    stem = "current_wizard_hyperliquid_concentration"
    return {
        "status": active / f"{stem}_status.csv",
        "cohorts": active / f"{stem}_cohorts.csv",
        "dimensions": active / f"{stem}_dimensions.csv",
        "contributors": active / f"{stem}_contributors.csv",
        "validation": active / f"{stem}_validation.csv",
        "manifest": active / f"{stem}_manifest.json",
        "summary_md": active / f"{stem}_summary.md",
        "snapshot_status": snapshot / "concentration_status.csv",
        "snapshot_cohorts": snapshot / "concentration_cohorts.csv",
        "snapshot_dimensions": snapshot / "concentration_dimensions.csv",
        "snapshot_contributors": snapshot / "concentration_contributors.csv",
        "snapshot_validation": snapshot / "validation.csv",
        "snapshot_manifest": snapshot / "manifest.json",
        "snapshot_summary_md": snapshot / "summary.md",
    }


def _summary_markdown(
    summary: dict[str, object],
    cohorts: pd.DataFrame,
    dimensions: pd.DataFrame,
) -> str:
    return "\n".join(
        [
            "# Current Wizard To Hyperliquid Cross-Cell Concentration",
            "",
            "This research gate tests whether apparent after-cost gains and trades are dominated by a narrow pair, timeframe, Wizard venue, exact mode, orientation, or causal regime.",
            "Only a non-empty statistically selected robustness cohort can pass. This artifact cannot authorize Testnet or live orders.",
            "",
            "## Summary",
            "",
            pd.DataFrame(
                [
                    {"metric": "concentration_id", "value": summary["concentration_id"]},
                    {"metric": "experiments", "value": summary["experiments_accounted"]},
                    {
                        "metric": "practical_walkforward_candidates",
                        "value": summary["practical_walkforward_candidates"],
                    },
                    {
                        "metric": "research_robustness_candidates",
                        "value": summary["research_robustness_candidates"],
                    },
                    {
                        "metric": "statistically_selected_robustness_candidates",
                        "value": summary[
                            "statistically_selected_robustness_candidates"
                        ],
                    },
                    {
                        "metric": "promotion_concentration_pass",
                        "value": summary["promotion_concentration_pass"],
                    },
                    {"metric": "live_trading_authorized", "value": False},
                ]
            ).to_markdown(index=False),
            "",
            "## Cohorts",
            "",
            cohorts.to_markdown(index=False),
            "",
            "## Dimensions",
            "",
            dimensions.to_markdown(index=False),
            "",
        ]
    )
