"""Canonical run identity and authority for the Hyperliquid research lane."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
MANIFEST_VERSION = "hyperliquid-run-manifest-v1"
EXECUTION_TRUTH_MODE = "hyperliquid_testnet"


def build_hyperliquid_run_manifest(
    *,
    root: Path = ROOT,
    now: datetime | None = None,
) -> dict[str, object]:
    """Freeze one candidate set and validate every upstream evidence join."""

    now = _as_utc(now or datetime.now(timezone.utc))
    active = root / "reports" / "active"
    output = root / "reports" / "orchestration" / "hyperliquid_run"
    active.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)

    bundle_path = active / "hyperliquid_research_bundle.csv"
    bundle = _read_csv(bundle_path)
    sources = {
        "bundle": bundle_path,
        "wizard": active / "hyperliquid_wizard_hypothesis_queue.csv",
        "cost": _matching_cost_source(active, bundle),
        "inventory": active / "hyperliquid_testnet_market_inventory.csv",
    }
    wizard = _read_csv(sources["wizard"])
    costs = _read_csv(sources["cost"])
    inventory = _read_csv(sources["inventory"])

    source_hashes = {name: _file_hash(path) for name, path in sources.items()}
    candidate_material = []
    if not bundle.empty and "pair" in bundle.columns:
        for _, row in bundle.sort_values("pair").iterrows():
            pair = _text(row.get("pair"))
            queue_row = _pair_row(wizard, pair)
            candidate_material.append(
                "|".join(
                    [
                        pair,
                        _text(row.get("interval")) or "1d",
                        _text(row.get("history_rows")),
                        _text(queue_row.get("exact_mode")),
                    ]
                )
            )
    wizard_candidate_set_ids = (
        wizard.get("candidate_set_id", pd.Series(dtype=object)).dropna().astype(str).str.strip()
    )
    wizard_candidate_set_ids = sorted(value for value in wizard_candidate_set_ids.unique() if value)
    if len(wizard_candidate_set_ids) == 1 and _pair_set(bundle) == _pair_set(wizard):
        candidate_set_id = wizard_candidate_set_ids[0]
    else:
        candidate_set_id = "hlset_" + sha256("\n".join(candidate_material).encode("utf-8")).hexdigest()[:20]
    run_material = "|".join([candidate_set_id, MANIFEST_VERSION, *[source_hashes[name] for name in sorted(source_hashes)]])
    run_id = "hlrun_" + sha256(run_material.encode("utf-8")).hexdigest()[:20]

    candidate_rows: list[dict[str, object]] = []
    for _, candidate in bundle.iterrows():
        pair = _text(candidate.get("pair"))
        queue_row = _pair_row(wizard, pair)
        cost_row = _pair_row(costs, pair)
        asset_x = (_text(candidate.get("asset_x")) or _pair_assets(pair)[0]).upper()
        asset_y = (_text(candidate.get("asset_y")) or _pair_assets(pair)[1]).upper()
        timeframe = _text(candidate.get("interval")) or "1d"
        lookback = _integer(candidate.get("history_rows"))
        nominated_mode = _text(queue_row.get("exact_mode"))
        setup_identity = _setup_identity(pair, timeframe, lookback, nominated_mode)
        history_path = _resolve_path(_text(candidate.get("history_path")), root)
        history_exists = bool(history_path and history_path.exists())
        history_hash = _file_hash(history_path) if history_exists and history_path is not None else ""
        wizard_blocker = _text(queue_row.get("blocker"))
        cost_ready = _truthy(cost_row.get("cost_model_ready"))
        slippage_ready = _truthy(cost_row.get("slippage_model_ready"))
        tradable_assets = _tradable_assets(inventory)
        blockers: list[str] = []
        if queue_row.empty:
            blockers.append("wizard_candidate_missing")
        if cost_row.empty:
            blockers.append("pair_cost_evidence_missing")
        if not history_exists:
            blockers.append("pair_history_missing")
        if asset_x not in tradable_assets:
            blockers.append(f"hyperliquid_testnet_market_missing:{asset_x}")
        if asset_y not in tradable_assets:
            blockers.append(f"hyperliquid_testnet_market_missing:{asset_y}")
        if wizard_blocker:
            blockers.extend(part for part in wizard_blocker.split(";") if part)
        if not cost_ready:
            blockers.append("cost_model_not_ready")
        if not slippage_ready:
            blockers.append("slippage_model_not_ready")
        evidence = [str(path.relative_to(root)) for path in sources.values() if path.exists()]
        if history_path is not None and history_path.exists():
            evidence.append(_relative(history_path, root))
        candidate_rows.append(
            {
                "run_id": run_id,
                "candidate_set_id": candidate_set_id,
                "setup_identity": setup_identity,
                "pair": pair,
                "asset_x": asset_x,
                "asset_y": asset_y,
                "venue": "hyperliquid",
                "timeframe": timeframe,
                "lookback": lookback,
                "nominated_exact_mode": nominated_mode,
                "history_path": _relative(history_path, root) if history_path is not None else "",
                "history_hash": history_hash,
                "wizard_source_timestamp": _text(queue_row.get("source_timestamp")),
                "wizard_settings_complete": "wizard_backtest_settings_incomplete" not in wizard_blocker,
                "wizard_source_fresh": "wizard_source_stale_or_timestamp_missing" not in wizard_blocker,
                "cost_model_ready": cost_ready,
                "slippage_model_ready": slippage_ready,
                "testnet_leg_x_ready": asset_x in tradable_assets,
                "testnet_leg_y_ready": asset_y in tradable_assets,
                "lineage_ready": bool(not queue_row.empty and not cost_row.empty and history_exists),
                "blockers": ";".join(dict.fromkeys(blockers)),
                "evidence_path": ";".join(dict.fromkeys(evidence)),
            }
        )
    candidates = pd.DataFrame(candidate_rows, columns=_candidate_columns())

    validation = _manifest_validation(
        candidates,
        bundle=bundle,
        wizard=wizard,
        costs=costs,
        inventory=inventory,
    )
    lineage_ready = bool(not validation.empty and validation["status"].eq("PASS").all())
    evidence_ready = bool(
        lineage_ready
        and not candidates.empty
        and candidates["history_hash"].astype(str).str.len().eq(64).all()
        and candidates["testnet_leg_x_ready"].astype(bool).all()
        and candidates["testnet_leg_y_ready"].astype(bool).all()
    )
    cost_ready = bool(not candidates.empty and candidates["cost_model_ready"].astype(bool).all() and candidates["slippage_model_ready"].astype(bool).all())
    wizard_parity_ready = bool(not candidates.empty and candidates["wizard_settings_complete"].astype(bool).all() and candidates["wizard_source_fresh"].astype(bool).all())
    candidate_blockers_clear = bool(
        not candidates.empty and candidates["blockers"].fillna("").astype(str).str.strip().eq("").all()
    )
    research_gate_ready = bool(
        evidence_ready and cost_ready and wizard_parity_ready and candidate_blockers_clear
    )

    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "run_id": run_id,
        "candidate_set_id": candidate_set_id,
        "generated_at": now.isoformat(),
        "execution_truth_mode": EXECUTION_TRUTH_MODE,
        "wizard_authority": "discovery_only",
        "local_research_authority": "hyperliquid_point_in_time_math_v2",
        "paper_execution_authority": "none",
        "candidate_count": len(candidates),
        "lineage_ready": lineage_ready,
        "research_evidence_ready": evidence_ready,
        "cost_evidence_ready": cost_ready,
        "wizard_parity_ready": wizard_parity_ready,
        "candidate_blockers_clear": candidate_blockers_clear,
        "status": "READY_FOR_RESEARCH" if research_gate_ready else "BLOCKED",
        "source_hashes": source_hashes,
        "source_paths": {name: _relative(path, root) for name, path in sources.items()},
    }
    manifest_path = active / "hyperliquid_run_manifest.json"
    candidate_path = active / "hyperliquid_run_candidates.csv"
    validation_path = active / "hyperliquid_run_manifest_validation.csv"
    summary_path = output / "hyperliquid_run_manifest.md"
    _atomic_json(manifest, manifest_path)
    _atomic_csv(candidates, candidate_path)
    _atomic_csv(validation, validation_path)
    summary_path.write_text(_manifest_markdown(manifest, validation, candidates), encoding="utf-8")
    return {
        **manifest,
        "manifest": manifest_path,
        "candidates": candidate_path,
        "validation": validation_path,
        "summary": summary_path,
    }


def build_hyperliquid_authority_state(*, root: Path = ROOT, now: datetime | None = None) -> dict[str, object]:
    """Publish layered readiness without granting execution authority."""

    now = _as_utc(now or datetime.now(timezone.utc))
    active = root / "reports" / "active"
    manifest = _read_json(active / "hyperliquid_run_manifest.json")
    candidates = _read_csv(active / "hyperliquid_run_candidates.csv")
    decisions = _read_csv(root / "reports" / "orchestration" / "teacher_council" / "council_decisions.csv")
    portfolio = _read_csv(root / "reports" / "orchestration" / "teacher_council" / "portfolio_critic.csv")
    student = _read_csv(root / "reports" / "orchestration" / "teacher_council" / "student_training_readiness.csv")
    preflight = _read_csv(active / "hyperliquid_testnet_preflight.csv")
    lifecycle = _read_csv(active / "hyperliquid_testnet_lifecycle_gate.csv")

    identity_match = _identity_matches_manifest(decisions, manifest)
    decisions_present = not decisions.empty
    council_clear = bool(
        decisions_present
        and identity_match
        and decisions.get("status", pd.Series(dtype=str)).astype(str).eq("SHADOW_TEST").all()
        and decisions.get("action", pd.Series(dtype=str)).astype(str).ne("abstain").all()
    )
    portfolio_clear = bool(not portfolio.empty and portfolio.get("verdict", pd.Series(dtype=str)).astype(str).eq("pass").all())
    supervised_scope = student.get("scope", pd.Series("all_learning", index=student.index)).astype(str).isin(
        {"all_learning", "supervised_student"}
    )
    supervised_ready = bool(
        not student.empty
        and not (student["status"].astype(str).eq("BLOCKED") & supervised_scope).any()
    )
    bandit_scope = student.get("scope", pd.Series("", index=student.index)).astype(str).eq("contextual_bandit")
    bandit_ready = bool(
        not student.empty
        and bool(bandit_scope.any())
        and not (student["status"].astype(str).eq("BLOCKED") & bandit_scope).any()
    )
    no_order_ready = bool(not preflight.empty and preflight.get("ready_for_no_order_preflight", pd.Series([False])).map(_truthy).iloc[0])
    submit_enabled = bool(not preflight.empty and preflight.get("submit_orders_enabled", pd.Series([False])).map(_truthy).iloc[0])
    lifecycle_ready = bool(not lifecycle.empty and lifecycle.get("status", pd.Series(dtype=str)).astype(str).eq("PASS").all())
    lineage_ready = bool(manifest.get("lineage_ready", False))
    research_ready = bool(
        lineage_ready
        and manifest.get("research_evidence_ready", False)
        and manifest.get("cost_evidence_ready", False)
        and manifest.get("wizard_parity_ready", False)
        and manifest.get("candidate_blockers_clear", False)
        and decisions_present
        and identity_match
    )
    paper_ready = bool(research_ready and council_clear and portfolio_clear and no_order_ready and supervised_ready and lifecycle_ready)
    # Live authority deliberately requires a separate realized-paper validation artifact.
    realized = _read_csv(root / "reports" / "paper" / "hyperliquid_realized_validation.csv")
    live_validated = bool(not realized.empty and realized.get("accepted", pd.Series([False])).map(_truthy).all())
    live_ready = bool(paper_ready and submit_enabled and live_validated)

    blockers: list[str] = []
    if not lineage_ready:
        blockers.append("canonical_lineage_not_ready")
    if not bool(manifest.get("cost_evidence_ready", False)):
        blockers.append("cost_evidence_not_ready")
    if not bool(manifest.get("wizard_parity_ready", False)):
        blockers.append("wizard_parity_not_ready")
    if not bool(manifest.get("candidate_blockers_clear", False)):
        blockers.append("candidate_research_blockers_present")
    if not identity_match:
        blockers.append("council_manifest_identity_mismatch")
    if not council_clear:
        blockers.append("teacher_council_not_clear")
    if not portfolio_clear:
        blockers.append("portfolio_critic_not_clear")
    if not supervised_ready:
        blockers.append("student_evidence_not_ready")
    if not no_order_ready:
        blockers.append("hyperliquid_testnet_preflight_not_ready")
    if not lifecycle_ready:
        blockers.append("hyperliquid_testnet_lifecycle_not_proven")
    if not submit_enabled:
        blockers.append("hyperliquid_testnet_submission_disabled")
    if not live_validated:
        blockers.append("realized_paper_validation_missing")

    row = {
        "run_id": _text(manifest.get("run_id")),
        "candidate_set_id": _text(manifest.get("candidate_set_id")),
        "execution_truth_mode": EXECUTION_TRUTH_MODE,
        "wizard_authority": "discovery_only",
        "infrastructure_ready": no_order_ready,
        "data_ready": lineage_ready and bool(manifest.get("research_evidence_ready", False)),
        "research_ready": research_ready,
        "paper_ready": paper_ready,
        "live_ready": live_ready,
        "supervised_student_ready": supervised_ready,
        "contextual_bandit_ready": bandit_ready,
        "candidate_count": len(candidates),
        "council_decisions": len(decisions),
        "identity_match": identity_match,
        "submit_orders_enabled": submit_enabled,
        "execution_allowed": live_ready,
        "status": "LIVE_READY" if live_ready else "RESEARCH_ONLY" if research_ready else "BLOCKED",
        "blocker": ";".join(dict.fromkeys(blockers)),
        "generated_at": now.isoformat(),
        "evidence_path": ";".join(
            [
                "reports/active/hyperliquid_run_manifest.json",
                "reports/orchestration/teacher_council/council_decisions.csv",
                "reports/orchestration/teacher_council/portfolio_critic.csv",
                "reports/orchestration/teacher_council/student_training_readiness.csv",
                "reports/active/hyperliquid_testnet_preflight.csv",
                "reports/active/hyperliquid_testnet_lifecycle_gate.csv",
            ]
        ),
    }
    frame = pd.DataFrame([row])
    authority_path = active / "hyperliquid_authority_state.csv"
    layered_path = root / "reports" / "dashboard" / "layered_readiness.csv"
    markdown_path = root / "reports" / "dashboard" / "layered_readiness.md"
    _atomic_csv(frame, authority_path)
    _atomic_csv(frame, layered_path)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(_authority_markdown(frame), encoding="utf-8")
    return {**row, "authority": authority_path, "layered_readiness": layered_path, "summary": markdown_path}


def load_hyperliquid_run_manifest(root: Path = ROOT) -> dict[str, Any]:
    return _read_json(root / "reports" / "active" / "hyperliquid_run_manifest.json")


def _manifest_validation(
    candidates: pd.DataFrame,
    *,
    bundle: pd.DataFrame,
    wizard: pd.DataFrame,
    costs: pd.DataFrame,
    inventory: pd.DataFrame,
) -> pd.DataFrame:
    candidate_pairs = _pair_set(candidates)
    rows = [
        _check("candidate_set_nonempty", bool(candidate_pairs), len(candidate_pairs), "> 0", "candidate_set_empty"),
        _check("candidate_pairs_unique", len(candidates) == len(candidate_pairs), len(candidates), len(candidate_pairs), "duplicate_candidate_pair"),
        _check("wizard_pair_set_contains_candidates", candidate_pairs.issubset(_pair_set(wizard)), len(candidate_pairs & _pair_set(wizard)), len(candidate_pairs), "wizard_candidate_set_mismatch"),
        _check("cost_pair_set_contains_candidates", candidate_pairs.issubset(_pair_set(costs)), len(candidate_pairs & _pair_set(costs)), len(candidate_pairs), "cost_candidate_set_mismatch"),
        _check("bundle_pair_set_matches_candidates", candidate_pairs == _pair_set(bundle), len(candidate_pairs), len(_pair_set(bundle)), "bundle_candidate_set_mismatch"),
        _check("history_lineage_complete", bool(not candidates.empty and candidates["history_hash"].astype(str).str.len().eq(64).all()), int(candidates["history_hash"].astype(str).str.len().eq(64).sum()) if not candidates.empty else 0, len(candidates), "history_hash_missing"),
        _check("testnet_markets_complete", bool(not candidates.empty and candidates["testnet_leg_x_ready"].astype(bool).all() and candidates["testnet_leg_y_ready"].astype(bool).all()), int((candidates["testnet_leg_x_ready"].astype(bool) & candidates["testnet_leg_y_ready"].astype(bool)).sum()) if not candidates.empty else 0, len(candidates), "testnet_market_coverage_incomplete"),
        _check("inventory_nonempty", not inventory.empty, len(inventory), "> 0", "hyperliquid_inventory_missing"),
    ]
    return pd.DataFrame(rows)


def _matching_cost_source(active: Path, bundle: pd.DataFrame) -> Path:
    direct = active / "hyperliquid_pair_cost_model.csv"
    workflow = active / "workflow_hyperliquid_pair_cost_model.csv"
    required_pairs = _pair_set(bundle)
    for path in (direct, workflow):
        frame = _read_csv(path)
        if required_pairs and required_pairs.issubset(_pair_set(frame)):
            return path
    return direct if direct.exists() else workflow


def _check(name: str, passed: bool, observed: object, required: object, blocker: str) -> dict[str, object]:
    return {
        "check": name,
        "status": "PASS" if passed else "BLOCKED",
        "observed": observed,
        "required": required,
        "blocker": "" if passed else blocker,
    }


def _identity_matches_manifest(frame: pd.DataFrame, manifest: dict[str, Any]) -> bool:
    if frame.empty or not manifest:
        return False
    for column in ("run_id", "candidate_set_id"):
        if column not in frame.columns:
            return False
        values = set(frame[column].dropna().astype(str))
        if values != {_text(manifest.get(column))}:
            return False
    return True


def _candidate_columns() -> list[str]:
    return [
        "run_id", "candidate_set_id", "setup_identity", "pair", "asset_x", "asset_y", "venue",
        "timeframe", "lookback", "nominated_exact_mode", "history_path", "history_hash",
        "wizard_source_timestamp", "wizard_settings_complete", "wizard_source_fresh", "cost_model_ready",
        "slippage_model_ready", "testnet_leg_x_ready", "testnet_leg_y_ready", "lineage_ready", "blockers",
        "evidence_path",
    ]


def _setup_identity(pair: str, timeframe: str, lookback: int, mode: str) -> str:
    normalized_pair = pair.replace("/", "|")
    normalized_mode = "_".join(mode.lower().replace("(", "").replace(")", "").split()) or "all_modes"
    return f"{normalized_pair}|{timeframe.lower()}|{lookback}|{normalized_mode}"


def _pair_assets(pair: str) -> tuple[str, str]:
    parts = pair.replace("/", "-").split("-")
    assets = [part for part in parts if part and part != "USD"]
    return (assets[0] if assets else "", assets[1] if len(assets) > 1 else "")


def _tradable_assets(frame: pd.DataFrame) -> set[str]:
    if frame.empty or "asset" not in frame.columns:
        return set()
    ready = frame.get("tradable_perp", pd.Series(False, index=frame.index)).map(_truthy)
    return set(frame.loc[ready, "asset"].astype(str).str.upper())


def _pair_row(frame: pd.DataFrame, pair: str) -> pd.Series:
    if frame.empty or "pair" not in frame.columns:
        return pd.Series(dtype=object)
    matches = frame.loc[frame["pair"].astype(str) == pair]
    return matches.iloc[0] if not matches.empty else pd.Series(dtype=object)


def _pair_set(frame: pd.DataFrame) -> set[str]:
    if frame.empty or "pair" not in frame.columns:
        return set()
    return {value for value in frame["pair"].dropna().astype(str) if value}


def _manifest_markdown(manifest: dict[str, Any], validation: pd.DataFrame, candidates: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# Canonical Hyperliquid Run Manifest",
            "",
            f"- Run: `{manifest['run_id']}`",
            f"- Candidate set: `{manifest['candidate_set_id']}`",
            f"- Status: **{manifest['status']}**",
            f"- Execution truth: **{manifest['execution_truth_mode']}**",
            "- Crypto Wizards: **discovery only**",
            "- Execution authority: **none until council, portfolio, paper, and live gates pass**",
            "",
            "## Validation",
            "",
            validation.to_markdown(index=False),
            "",
            "## Candidates",
            "",
            candidates.to_markdown(index=False) if not candidates.empty else "No canonical candidates.",
            "",
        ]
    )


def _authority_markdown(frame: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# Layered Hyperliquid Readiness",
            "",
            "Infrastructure readiness is not trading readiness. The final execution flag is fail-closed.",
            "",
            frame.to_markdown(index=False),
            "",
        ]
    )


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _resolve_path(value: str, root: Path) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


def _file_hash(path: Path | None) -> str:
    if path is None or not path.exists() or not path.is_file():
        return ""
    if path.suffix.lower() == ".csv":
        frame = _read_csv(path)
        if not frame.empty:
            stable = frame.drop(columns=["run_id", "candidate_set_id"], errors="ignore")
            return sha256(stable.to_csv(index=False).encode("utf-8")).hexdigest()
    return sha256(path.read_bytes()).hexdigest()


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _integer(value: object) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)
