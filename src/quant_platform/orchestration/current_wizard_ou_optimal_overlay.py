"""Account for the Wizard ``ou_optimal`` scanner annotation without inventing a mode."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import pandas as pd

from quant_platform.active_pipeline import CommandResult
from quant_platform.orchestration.exhaustive_wizard_api_refresh import (
    PAIR_PAGE_EXACT_MODES,
)
from quant_platform.orchestration.snapshot_lineage import (
    artifact_content_hash,
    verified_snapshot_reference,
)


ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "current_wizard_ou_optimal_overlay.v1"


def build_current_wizard_ou_optimal_overlay(
    *,
    root: Path = ROOT,
    now: datetime | None = None,
) -> CommandResult:
    """Map every scanner boolean to its actual exact-mode/orientation experiment."""

    as_of = _as_utc(now or datetime.now(timezone.utc))
    active = root / "reports" / "active"
    manifests = {
        "refresh": _read_json(active / "exhaustive_wizard_api_refresh_manifest.json"),
        "handoff": _read_json(active / "current_wizard_hyperliquid_handoff_manifest.json"),
        "canonical": _read_json(
            active / "current_wizard_hyperliquid_canonical_replay_manifest.json"
        ),
        "observed": _read_json(
            active / "current_wizard_hyperliquid_observed_cost_replay_manifest.json"
        ),
        "walkforward": _read_json(
            active / "current_wizard_hyperliquid_walkforward_manifest.json"
        ),
    }
    active_inputs = {
        "source_accounting": active
        / "exhaustive_wizard_api_refresh_source_accounting.csv",
        "api_candidates": active / "wizard_sweep_candidates.csv",
        "pair_status": active / "current_wizard_pair_detail_status.csv",
        "experiments": active / "current_wizard_hyperliquid_experiment_matrix.csv",
        "canonical_results": active
        / "current_wizard_hyperliquid_canonical_replay.csv",
        "observed_results": active
        / "current_wizard_hyperliquid_observed_cost_replay.csv",
        "walkforward_status": active
        / "current_wizard_hyperliquid_walkforward_status.csv",
    }
    manifest_keys = {
        "source_accounting": ("refresh", "source_accounting"),
        "api_candidates": ("refresh", "api_candidates"),
        "pair_status": ("handoff", "pair_status"),
        "experiments": ("handoff", "experiments"),
        "canonical_results": ("canonical", "results"),
        "observed_results": ("observed", "results"),
        "walkforward_status": ("walkforward", "status"),
    }
    snapshots = {
        name: verified_snapshot_reference(
            root=root,
            active_path=active_path,
            upstream_manifest=manifests[manifest_name],
            artifact_key=artifact_key,
        )
        for name, active_path in active_inputs.items()
        for manifest_name, artifact_key in (manifest_keys[name],)
    }
    source = _read_csv(snapshots["source_accounting"])
    candidates = _read_csv(snapshots["api_candidates"])
    pair_status = _read_csv(snapshots["pair_status"])
    experiments = _read_csv(snapshots["experiments"])
    canonical = _read_csv(snapshots["canonical_results"])
    observed = _read_csv(snapshots["observed_results"])
    walkforward = _read_csv(snapshots["walkforward_status"])
    _validate_source_alignment(source, candidates)

    refresh_id = _text(manifests["refresh"].get("refresh_id"))
    handoff_id = _text(manifests["handoff"].get("handoff_id"))
    material = {
        "schema_version": SCHEMA_VERSION,
        "refresh_id": refresh_id,
        "handoff_id": handoff_id,
        "input_hashes": {
            name: artifact_content_hash(path) for name, path in snapshots.items()
        },
        "pair_page_exact_modes": PAIR_PAGE_EXACT_MODES,
        "scanner_overlay": "ou_optimal",
    }
    overlay_id = "cwouoverlay_" + sha256(
        _canonical_json(material).encode("utf-8")
    ).hexdigest()[:20]

    pair_lookup = {
        _text(row.pair_group_key): row._asdict()
        for row in pair_status.itertuples(index=False)
    }
    experiment_lookup = {
        (
            _text(row.pair_group_key),
            _text(row.exact_mode),
            _text(row.orientation),
        ): row._asdict()
        for row in experiments.itertuples(index=False)
        if _text(row.exact_mode) in PAIR_PAGE_EXACT_MODES
    }
    canonical_lookup = _row_lookup(canonical, "experiment_id")
    observed_lookup = _row_lookup(observed, "experiment_id")
    walk_lookup = _row_lookup(walkforward, "experiment_id")

    rows: list[dict[str, object]] = []
    ordered = source.sort_values("api_source_row_index", kind="mergesort")
    for source_row in ordered.itertuples(index=False):
        source_values = source_row._asdict()
        index = int(source_values["api_source_row_index"])
        candidate = candidates.iloc[index]
        flag = _truthy(candidate.get("ou_optimal"))
        if "api_ou_optimal" in source.columns:
            if flag != _truthy(source_values.get("api_ou_optimal")):
                raise ValueError(f"OU optimal source mismatch at row {index}")
        pair_key = _text(source_values.get("pair_group_key"))
        exact_mode = _text(source_values.get("api_exact_mode"))
        pair = pair_lookup.get(pair_key, {})
        orientation = _experiment_orientation(source_values, pair)
        orientation_resolution = _orientation_resolution(
            source_values, orientation
        )
        experiment = experiment_lookup.get((pair_key, exact_mode, orientation), {})
        experiment_id = _text(experiment.get("experiment_id"))
        canonical_row = canonical_lookup.get(experiment_id, {})
        observed_row = observed_lookup.get(experiment_id, {})
        walk_row = walk_lookup.get(experiment_id, {})
        canonical_status = _text(canonical_row.get("replay_status"))
        if not flag:
            overlay_status = "NOT_FLAGGED"
            overlay_blocker = ""
        elif not experiment_id:
            overlay_status = "BLOCKED_BASE_EXPERIMENT_MISSING"
            overlay_blocker = "source_exact_mode_orientation_experiment_missing"
        elif canonical_status == "RESEARCH_REPLAY_COMPLETE":
            overlay_status = "BASE_EXACT_MODE_REPLAY_COMPLETE"
            overlay_blocker = ""
        else:
            overlay_status = "BASE_EXACT_MODE_REPLAY_BLOCKED"
            overlay_blocker = (
                _text(canonical_row.get("replay_blocker"))
                or canonical_status
                or _text(experiment.get("experiment_blocker"))
                or _text(experiment.get("experiment_status"))
            )
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "overlay_id": overlay_id,
                "refresh_id": refresh_id,
                "handoff_id": handoff_id,
                "api_source_row_id": _text(source_values.get("api_source_row_id")),
                "api_source_row_index": index,
                "source_row_fingerprint": _text(
                    source_values.get("source_row_fingerprint")
                ),
                "pair_group_key": pair_key,
                "pair": _text(source_values.get("pair")),
                "wizard_exchange": _text(source_values.get("wizard_exchange")),
                "timeframe": _text(source_values.get("timeframe")),
                "source_orientation": _text(source_values.get("orientation")),
                "experiment_orientation": orientation,
                "orientation_resolution": orientation_resolution,
                "source_exact_mode": exact_mode,
                "ou_optimal": flag,
                "ou_optimal_semantics": "scanner_boolean_annotation",
                "independent_pair_page_mode": False,
                "overlay_on_source_exact_mode": True,
                "base_experiment_id": experiment_id,
                "hyperliquid_pair_ready": _truthy(
                    pair.get("hyperliquid_pair_ready")
                ),
                "base_experiment_status": _text(
                    experiment.get("experiment_status")
                ),
                "canonical_replay_status": canonical_status,
                "canonical_research_rank_eligible": _truthy(
                    canonical_row.get("research_rank_eligible")
                ),
                "observed_cost_replay_status": _text(
                    observed_row.get("replay_status")
                ),
                "walkforward_status": _text(
                    walk_row.get("walkforward_status")
                ),
                "statistical_selection_status": _text(
                    walk_row.get("statistical_selection_status")
                ),
                "overlay_status": overlay_status,
                "overlay_blocker": overlay_blocker,
                "sweep_captured_at": _text(
                    source_values.get("sweep_captured_at")
                ),
                "sweep_source_timestamp": _text(
                    source_values.get("sweep_source_timestamp")
                ),
                "evidence_path": _text(source_values.get("evidence_path")),
                "acceptance_authority": False,
                "promotion_authority": False,
                "order_submission_performed": False,
                "live_trading_authorized": False,
            }
        )
    ledger = pd.DataFrame(rows)
    coverage = _coverage(ledger)
    pair_page_mode_ledger = active / "exhaustive_wizard_pair_detail_mode_ledger.csv"
    pair_page_evidence = _pair_page_evidence(pair_page_mode_ledger)
    validation = _validation(
        ledger=ledger,
        source=source,
        candidates=candidates,
        pair_page_evidence=pair_page_evidence,
    )
    if not validation["status"].eq("PASS").all():
        failed = validation.loc[validation["status"].ne("PASS"), "check"].tolist()
        raise ValueError("OU optimal overlay validation failed: " + ",".join(failed))

    handoff_snapshot = root / _text(
        manifests["handoff"].get("artifacts", {}).get("snapshot_manifest")
    )
    snapshot_dir = handoff_snapshot.parent / "ou_optimal_overlays" / overlay_id
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    paths = _paths(active, snapshot_dir)
    for frame, active_key, snapshot_key in (
        (ledger, "ledger", "snapshot_ledger"),
        (coverage, "coverage", "snapshot_coverage"),
        (validation, "validation", "snapshot_validation"),
    ):
        frame.to_csv(paths[active_key], index=False)
        frame.to_csv(paths[snapshot_key], index=False)

    flagged = ledger.loc[ledger["ou_optimal"].astype(bool)]
    summary: dict[str, object] = {
        **material,
        "overlay_id": overlay_id,
        "created_at": as_of.isoformat(),
        "source_rows": int(len(ledger)),
        "source_rows_accounted": int(ledger["api_source_row_id"].nunique()),
        "ou_optimal_true_rows": int(len(flagged)),
        "ou_optimal_false_rows": int(len(ledger) - len(flagged)),
        "ou_optimal_true_pair_groups": int(flagged["pair_group_key"].nunique()),
        "self_pair_rows_accounted": int(
            ledger["orientation_resolution"]
            .eq("self_pair_orientations_equivalent_original_selected")
            .sum()
        ),
        "ou_optimal_true_base_replays_complete": int(
            flagged["canonical_replay_status"].eq("RESEARCH_REPLAY_COMPLETE").sum()
        ),
        "ou_optimal_true_walkforward_passes": int(
            flagged["walkforward_status"].eq("PASS_RESEARCH_WALK_FORWARD").sum()
        ),
        "ou_optimal_true_statistically_selected": int(
            flagged["statistical_selection_status"].eq("PASS").sum()
        ),
        "pair_page_ou_optimal_rows": pair_page_evidence["rows"],
        "pair_page_ou_optimal_captured_rows": pair_page_evidence["captured_rows"],
        "pair_page_semantics": "ou_optimal_not_offered_as_independent_mode",
        "input_snapshots": {
            name: _relative(path, root) for name, path in snapshots.items()
        },
        "locally_copied_input_bytes": 0,
        "acceptance_authority": False,
        "promotion_authority": False,
        "order_submission_performed": False,
        "live_trading_authorized": False,
        "artifacts": {name: _relative(path, root) for name, path in paths.items()},
    }
    manifest_text = json.dumps(summary, indent=2, sort_keys=True)
    summary_text = _summary_markdown(summary)
    paths["manifest"].write_text(manifest_text, encoding="utf-8")
    paths["snapshot_manifest"].write_text(manifest_text, encoding="utf-8")
    paths["summary_md"].write_text(summary_text, encoding="utf-8")
    paths["snapshot_summary_md"].write_text(summary_text, encoding="utf-8")
    return CommandResult(paths=paths, summary=summary)


def _validate_source_alignment(source: pd.DataFrame, candidates: pd.DataFrame) -> None:
    if len(source) != len(candidates):
        raise ValueError("API source accounting and candidate row counts differ")
    positions = pd.to_numeric(source["api_source_row_index"], errors="raise").astype(int)
    if set(positions) != set(range(len(candidates))):
        raise ValueError("API source accounting indexes are not complete")
    for row in source.itertuples(index=False):
        values = row._asdict()
        index = int(values["api_source_row_index"])
        candidate = candidates.iloc[index]
        fingerprint = sha256(
            _canonical_json(
                {str(key): _json_value(value) for key, value in candidate.items()}
            ).encode("utf-8")
        ).hexdigest()
        if fingerprint != _text(values.get("source_row_fingerprint")):
            raise ValueError(f"API source fingerprint mismatch at row {index}")


def _experiment_orientation(
    source: dict[str, object], pair: dict[str, object]
) -> str:
    source_x = _text(source.get("asset_x")).upper()
    source_y = _text(source.get("asset_y")).upper()
    asset_a = _text(pair.get("asset_a")).upper()
    asset_b = _text(pair.get("asset_b")).upper()
    if source_x and source_x == source_y:
        return "original"
    if source_x == asset_a and source_y == asset_b:
        return "original"
    if source_x == asset_b and source_y == asset_a:
        return "reverse"
    return "unresolved"


def _orientation_resolution(
    source: dict[str, object], orientation: str
) -> str:
    source_x = _text(source.get("asset_x")).upper()
    source_y = _text(source.get("asset_y")).upper()
    if source_x and source_x == source_y and orientation == "original":
        return "self_pair_orientations_equivalent_original_selected"
    if orientation in {"original", "reverse"}:
        return "direct_asset_order"
    return "unresolved"


def _row_lookup(frame: pd.DataFrame, key: str) -> dict[str, dict[str, object]]:
    return {
        _text(row[key]): row
        for row in frame.to_dict(orient="records")
        if _text(row.get(key))
    }


def _coverage(ledger: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mode, group in ledger.groupby("source_exact_mode", sort=True):
        flagged = group.loc[group["ou_optimal"].astype(bool)]
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "source_exact_mode": mode,
                "source_rows": int(len(group)),
                "ou_optimal_true_rows": int(len(flagged)),
                "ou_optimal_false_rows": int(len(group) - len(flagged)),
                "true_pair_groups": int(flagged["pair_group_key"].nunique()),
                "true_hyperliquid_ready_rows": int(
                    flagged["hyperliquid_pair_ready"].astype(bool).sum()
                ),
                "true_base_replays_complete": int(
                    flagged["canonical_replay_status"]
                    .eq("RESEARCH_REPLAY_COMPLETE")
                    .sum()
                ),
                "true_walkforward_passes": int(
                    flagged["walkforward_status"]
                    .eq("PASS_RESEARCH_WALK_FORWARD")
                    .sum()
                ),
                "true_statistically_selected": int(
                    flagged["statistical_selection_status"].eq("PASS").sum()
                ),
                "acceptance_authority": False,
                "live_trading_authorized": False,
            }
        )
    return pd.DataFrame(rows)


def _pair_page_evidence(path: Path) -> dict[str, int]:
    if not path.is_file():
        return {"rows": 0, "captured_rows": 0}
    frame = pd.read_csv(path, keep_default_na=False)
    rows = frame.loc[frame["exact_mode"].astype(str).eq("OU (Optimal)")]
    return {
        "rows": int(len(rows)),
        "captured_rows": int(rows["capture_status"].astype(str).eq("CAPTURED").sum()),
    }


def _validation(
    *,
    ledger: pd.DataFrame,
    source: pd.DataFrame,
    candidates: pd.DataFrame,
    pair_page_evidence: dict[str, int],
) -> pd.DataFrame:
    checks = (
        ("all_source_rows_accounted", len(ledger) == len(source) == len(candidates)),
        (
            "source_row_ids_unique",
            ledger["api_source_row_id"].nunique() == len(ledger),
        ),
        (
            "true_and_false_rows_reconcile",
            int(ledger["ou_optimal"].astype(bool).sum())
            + int((~ledger["ou_optimal"].astype(bool)).sum())
            == len(ledger),
        ),
        (
            "only_actual_pair_page_modes_are_base_experiments",
            ledger["source_exact_mode"].isin(PAIR_PAGE_EXACT_MODES).all(),
        ),
        (
            "every_source_row_maps_to_base_experiment",
            ledger["base_experiment_id"].astype(str).str.len().gt(0).all(),
        ),
        (
            "every_source_orientation_resolved",
            ledger["experiment_orientation"].astype(str).ne("unresolved").all(),
        ),
        (
            "ou_optimal_not_treated_as_independent_pair_page_mode",
            not ledger["independent_pair_page_mode"].astype(bool).any()
            and pair_page_evidence["captured_rows"] == 0,
        ),
        (
            "no_acceptance_authority",
            not ledger["acceptance_authority"].astype(bool).any(),
        ),
        (
            "no_promotion_authority",
            not ledger["promotion_authority"].astype(bool).any(),
        ),
        (
            "no_order_submission",
            not ledger["order_submission_performed"].astype(bool).any(),
        ),
        (
            "live_authority_false",
            not ledger["live_trading_authorized"].astype(bool).any(),
        ),
    )
    return pd.DataFrame(
        [
            {
                "schema_version": SCHEMA_VERSION,
                "check": check,
                "status": "PASS" if passed else "FAIL",
                "evidence": str(len(ledger)),
                "live_trading_authorized": False,
            }
            for check, passed in checks
        ]
    )


def _paths(active: Path, snapshot: Path) -> dict[str, Path]:
    stem = "current_wizard_ou_optimal_overlay"
    return {
        "ledger": active / f"{stem}_ledger.csv",
        "coverage": active / f"{stem}_coverage.csv",
        "validation": active / f"{stem}_validation.csv",
        "manifest": active / f"{stem}_manifest.json",
        "summary_md": active / f"{stem}_summary.md",
        "snapshot_ledger": snapshot / "ledger.csv",
        "snapshot_coverage": snapshot / "coverage.csv",
        "snapshot_validation": snapshot / "validation.csv",
        "snapshot_manifest": snapshot / "manifest.json",
        "snapshot_summary_md": snapshot / "summary.md",
    }


def _summary_markdown(summary: dict[str, object]) -> str:
    return "\n".join(
        [
            "# Current Wizard OU Optimal Overlay",
            "",
            "`ou_optimal` is a scanner annotation on an actual source mode, not an independent pair-page mode.",
            "",
            f"- overlay: `{summary['overlay_id']}`",
            f"- source rows accounted: {summary['source_rows_accounted']} of {summary['source_rows']}",
            f"- true / false rows: {summary['ou_optimal_true_rows']} / {summary['ou_optimal_false_rows']}",
            f"- true pair groups: {summary['ou_optimal_true_pair_groups']}",
            f"- blocked self-pair rows accounted: {summary['self_pair_rows_accounted']}",
            f"- true base replays complete: {summary['ou_optimal_true_base_replays_complete']}",
            f"- true walk-forward passes: {summary['ou_optimal_true_walkforward_passes']}",
            f"- true statistically selected: {summary['ou_optimal_true_statistically_selected']}",
            f"- independent OU Optimal pair-page captures: {summary['pair_page_ou_optimal_captured_rows']}",
            "- promotion authority: false",
            "- order submission performed: false",
            "- live trading authorized: false",
            "",
        ]
    )


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, keep_default_na=False)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "pass", "complete"}


def _json_value(value: object) -> object:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    return value


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _as_utc(value: datetime) -> datetime:
    return (
        value.astimezone(timezone.utc)
        if value.tzinfo
        else value.replace(tzinfo=timezone.utc)
    )
