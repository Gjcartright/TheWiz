"""Crypto Wizards exact-mode capture accounting and fail-closed parity authority."""

from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_platform.active_pipeline import CommandResult


ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "thewiz.wizard_exact_mode_parity.v1"
PAIR_PAGE_EXACT_MODES = (
    "Copula",
    "Dyn (Spread)",
    "Dyn (ZScoreR)",
    "OU (Spread)",
    "OU (ZScoreR)",
    "Static (Spread)",
    "Static (ZScoreR)",
)
EXACT_MODES = PAIR_PAGE_EXACT_MODES
ORIENTATIONS = ("original", "reverse")
FIXTURE_FIELDS = (
    "pair_group_key", "pair", "wizard_exchange", "timeframe", "asset_x", "asset_y",
    "exact_mode", "orientation", "capture_timestamp", "periods_analyzed", "entry_long",
    "entry_short", "exit_long", "exit_short", "rolling_window", "close_n_periods",
    "stop_loss_pct", "x_weighting", "wizard_commission_pct", "wizard_slippage_pct",
    "hedge_ratio", "hurst", "half_life", "pearson_returns", "spearman_returns",
    "kendall_returns", "conditional_chart_value", "copula_family", "copula_correlation",
    "u1_given_u2", "u2_given_u1", "ou_mu", "ou_alpha", "ou_beta", "ou_b", "ou_sigma",
    "sharpe", "sortino", "returns_total", "closed_trades", "max_drawdown", "var_99",
    "cvar_99", "evidence_path",
)


def compare_series(expected: np.ndarray, observed: np.ndarray, *, tolerance: float = 1e-9) -> dict[str, Any]:
    expected = np.asarray(expected, dtype=float)
    observed = np.asarray(observed, dtype=float)
    if expected.shape != observed.shape or expected.size == 0:
        return {"status": "BLOCKED", "max_abs_delta": math.inf, "blocker": "shape_or_empty_mismatch"}
    delta = np.abs(expected - observed)
    if not np.isfinite(delta).all():
        return {"status": "BLOCKED", "max_abs_delta": math.inf, "blocker": "nonfinite_series"}
    maximum = float(delta.max())
    return {"status": "PASS" if maximum <= tolerance else "FAIL", "max_abs_delta": maximum, "blocker": "" if maximum <= tolerance else "parity_tolerance_exceeded"}


def build_wizard_parity_capture_status(*, root: Path = ROOT) -> dict[str, Any]:
    source_path = root / "reports" / "active" / "exhaustive_wizard_pair_detail_mode_ledger.csv"
    source = _read_csv(source_path)
    rows = []
    for mode in EXACT_MODES:
        for orientation in ORIENTATIONS:
            group = source.loc[source.get("exact_mode", pd.Series(dtype=str)).eq(mode) & source.get("orientation", pd.Series(dtype=str)).eq(orientation)]
            captured = group.loc[group.get("capture_status", pd.Series(dtype=str)).eq("CAPTURED")]
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "exact_mode": mode,
                    "orientation": orientation,
                    "expected_cells": len(group),
                    "captured_cells": len(captured),
                    "orientation_verified_cells": int(captured.get("orientation_verified", pd.Series(False, index=captured.index)).map(_truthy).sum()),
                    "raw_chart_preserved_cells": int(captured.get("raw_chart_data_preserved", pd.Series(False, index=captured.index)).map(_truthy).sum()),
                    "point_in_time_cost_confirmed_cells": int(captured.get("wizard_cost_point_in_time_ui_confirmed", pd.Series(False, index=captured.index)).map(_truthy).sum()),
                    "capture_status": "CAPTURED" if len(captured) else "UNAVAILABLE_OR_MISSING",
                    "parity_claim_allowed": False,
                    "blocker": "" if len(captured) else "vendor_mode_not_available_or_not_captured",
                    "evidence_path": _relative(source_path, root),
                    "live_trading_authorized": False,
                }
            )
    frame = pd.DataFrame(rows)
    status_path = root / "reports" / "active" / "wizard_parity_capture_status.csv"
    _atomic_csv(frame, status_path)
    raw_dir = root / "data" / "raw" / "crypto_wizards_parity"
    raw_dir.mkdir(parents=True, exist_ok=True)
    capture_manifest = {
        "schema_version": SCHEMA_VERSION,
        "source_path": _relative(source_path, root),
        "source_sha256": _file_hash(source_path),
        "expected_modes": list(EXACT_MODES),
        "orientations": list(ORIENTATIONS),
        "cells_accounted": int(frame["expected_cells"].sum()),
        "cells_captured": int(frame["captured_cells"].sum()),
        "point_in_time_cost_semantics_proven": False,
        "vendor_formula_parity_proven": False,
        "live_trading_authorized": False,
    }
    (raw_dir / "capture_manifest.json").write_text(json.dumps(capture_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"frame": frame, "status": status_path, "capture_manifest": raw_dir / "capture_manifest.json", "source": source, "summary": capture_manifest}


def build_ou_optimal_overlay_provenance(*, root: Path = ROOT) -> dict[str, Any]:
    """Account for OU Optimal as a scanner annotation, not a pair-page mode."""

    source_path = root / "reports" / "active" / "current_wizard_ou_optimal_overlay_ledger.csv"
    source = _read_csv(source_path)
    required = {
        "experiment_orientation",
        "ou_optimal",
        "ou_optimal_semantics",
        "independent_pair_page_mode",
        "source_exact_mode",
        "evidence_path",
    }
    rows = []
    for orientation in ORIENTATIONS:
        group = (
            source.loc[source.get("experiment_orientation", pd.Series(dtype=str)).eq(orientation)]
            if required.issubset(source.columns)
            else pd.DataFrame()
        )
        true_count = int(group.get("ou_optimal", pd.Series(dtype=bool)).map(_truthy).sum())
        false_count = int(len(group) - true_count)
        semantics_valid = bool(
            not group.empty
            and group["ou_optimal_semantics"].astype(str).eq("scanner_boolean_annotation").all()
        )
        not_pair_mode = bool(
            not group.empty
            and not group["independent_pair_page_mode"].map(_truthy).any()
        )
        status = "PASS" if semantics_valid and not_pair_mode else "BLOCKED"
        blockers = []
        if group.empty:
            blockers.append("overlay_orientation_not_captured")
        if not semantics_valid:
            blockers.append("ou_optimal_scanner_semantics_unproven")
        if not not_pair_mode:
            blockers.append("ou_optimal_pair_page_mode_classification_conflict")
        rows.append(
            {
                "overlay": "OU Optimal",
                "orientation": orientation,
                "source_rows": len(group),
                "true_rows": true_count,
                "false_rows": false_count,
                "row_accounting_complete": len(group) == true_count + false_count,
                "semantics": "scanner_boolean_annotation" if semantics_valid else "unproven",
                "independent_pair_page_mode": False if not_pair_mode else pd.NA,
                "overlay_provenance_status": status,
                "formula_parity_status": "UNPROVEN",
                "blocker": ";".join(blockers),
                "evidence_path": _relative(source_path, root),
                "promotion_authority": False,
                "live_trading_authorized": False,
            }
        )
    frame = pd.DataFrame(rows)
    path = root / "reports" / "active" / "wizard_ou_optimal_overlay_provenance.csv"
    _atomic_csv(frame, path)
    return {
        "path": path,
        "frame": frame,
        "orientations_accounted": int(frame["overlay_provenance_status"].eq("PASS").sum()),
        "expected_orientations": len(ORIENTATIONS),
        "status": "PASS" if frame["overlay_provenance_status"].eq("PASS").all() else "BLOCKED",
        "vendor_formula_parity_proven": False,
        "live_trading_authorized": False,
    }


def build_wizard_golden_fixtures(*, root: Path = ROOT) -> dict[str, Any]:
    capture = build_wizard_parity_capture_status(root=root)
    source = capture["source"]
    fixture_dir = root / "data" / "fixtures" / "wizard_exact_modes"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    audit_rows = []
    for mode in EXACT_MODES:
        for orientation in ORIENTATIONS:
            eligible = source.loc[
                source.get("exact_mode", pd.Series(dtype=str)).eq(mode)
                & source.get("orientation", pd.Series(dtype=str)).eq(orientation)
                & source.get("capture_status", pd.Series(dtype=str)).eq("CAPTURED")
                & source.get("orientation_verified", pd.Series(False, index=source.index)).map(_truthy)
                & source.get("raw_chart_data_preserved", pd.Series(False, index=source.index)).map(_truthy)
            ].copy()
            fixture_status = "GOLDEN_CAPTURE" if not eligible.empty else "MISSING_VENDOR_MODE"
            fixture_path = ""
            evidence_path = ""
            source_hash = ""
            if not eligible.empty:
                eligible["_completeness"] = eligible[list(set(FIXTURE_FIELDS) & set(eligible.columns))].notna().sum(axis=1)
                row = eligible.sort_values(["_completeness", "capture_timestamp"], ascending=[False, False]).iloc[0]
                payload = {field: _json_scalar(row.get(field)) for field in FIXTURE_FIELDS}
                payload.update({"schema_version": SCHEMA_VERSION, "fixture_status": fixture_status, "vendor_formula_parity_proven": False, "live_trading_authorized": False})
                name = f"{_slug(mode)}__{orientation}.json"
                path = fixture_dir / name
                path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
                fixture_path = _relative(path, root)
                source_hash = _file_hash(path)
                evidence_path = _text(row.get("evidence_path"))
            blocker = "" if fixture_status == "GOLDEN_CAPTURE" else "mode_unavailable_on_captured_pair_pages"
            entry = {
                "exact_mode": mode,
                "orientation": orientation,
                "fixture_status": fixture_status,
                "fixture_path": fixture_path,
                "fixture_sha256": source_hash,
                "source_evidence_path": evidence_path,
                "blocker": blocker,
                "vendor_formula_parity_proven": False,
                "live_trading_authorized": False,
            }
            entries.append(entry)
            audit_rows.append(entry)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "required_modes": list(EXACT_MODES),
        "required_orientations": list(ORIENTATIONS),
        "required_cells": len(EXACT_MODES) * len(ORIENTATIONS),
        "cells_accounted": len(entries),
        "golden_captures": sum(row["fixture_status"] == "GOLDEN_CAPTURE" for row in entries),
        "missing_vendor_modes": sum(row["fixture_status"] != "GOLDEN_CAPTURE" for row in entries),
        "entries": entries,
        "live_trading_authorized": False,
    }
    manifest_path = fixture_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit = pd.DataFrame(audit_rows)
    audit_path = root / "reports" / "active" / "wizard_golden_fixture_audit.csv"
    _atomic_csv(audit, audit_path)
    return {"manifest": manifest_path, "audit": audit_path, "summary": manifest, "audit_frame": audit}


def run_wizard_mode_mutation_tests(*, root: Path = ROOT) -> pd.DataFrame:
    baseline = np.linspace(-3.0, 3.0, 121)
    cases = {
        "matching_formula": baseline.copy(),
        "window_mutation": np.roll(baseline, 1),
        "spread_sign_mutation": -baseline,
        "zscore_scale_mutation": baseline * 0.5,
        "copula_tail_mutation": np.clip(baseline, -1.0, 1.0),
        "orientation_mutation": baseline[::-1],
        "exit_mutation": np.where(np.abs(baseline) < 0.25, 1.0, baseline),
    }
    rows = []
    for case, observed in cases.items():
        comparison = compare_series(baseline, observed)
        expected = "PASS" if case == "matching_formula" else "FAIL"
        rows.append(
            {
                "case": case,
                "expected_status": expected,
                "actual_status": comparison["status"],
                "max_abs_delta": comparison["max_abs_delta"],
                "blocker": comparison["blocker"],
                "mislabeled_mode_can_receive_parity": case != "matching_formula" and comparison["status"] == "PASS",
                "live_trading_authorized": False,
                "status": "PASS" if expected == comparison["status"] else "FAIL",
            }
        )
    frame = pd.DataFrame(rows)
    path = root / "reports" / "red_team" / "wizard_mode_mutation_results.csv"
    _atomic_csv(frame, path)
    if not frame["status"].eq("PASS").all():
        raise ValueError("Wizard mode mutation escaped parity comparison")
    return frame


def build_wizard_mode_authority(*, root: Path = ROOT) -> dict[str, Any]:
    fixtures = build_wizard_golden_fixtures(root=root)
    audit = fixtures["audit_frame"]
    rows = []
    for row in audit.to_dict("records"):
        golden = row["fixture_status"] == "GOLDEN_CAPTURE"
        rows.append(
            {
                "exact_mode": row["exact_mode"],
                "orientation": row["orientation"],
                "capture_authority": "point_in_time_vendor_observation" if golden else "unproven",
                "formula_authority": "local_approximation" if golden else "unproven",
                "vendor_parity_status": "UNPROVEN",
                "vendor_exact_mode_parity_proven": False,
                "research_use": "local_approximation_research" if golden else "blocked_until_vendor_capture",
                "wizard_confirmed_claim_allowed": False,
                "acceptance_eligible": False,
                "blocker": "comparable_vendor_output_series_and_formula_parity_missing" if golden else row["blocker"],
                "evidence_path": row["fixture_path"] or row["source_evidence_path"],
                "promotion_authority": False,
                "live_trading_authorized": False,
            }
        )
    authority = pd.DataFrame(rows)
    path = root / "reports" / "active" / "wizard_mode_authority.csv"
    _atomic_csv(authority, path)
    parity_path = root / "reports" / "active" / "wizard_mode_parity.csv"
    _atomic_csv(authority, parity_path)
    overlay = _read_csv(
        root / "reports" / "active" / "wizard_ou_optimal_overlay_provenance.csv"
    )
    overlay_accounted = int(
        overlay.get("overlay_provenance_status", pd.Series(dtype=str)).eq("PASS").sum()
    )
    docs = root / "docs" / "wizard_hyperliquid_mode_fidelity.md"
    docs.write_text(
        _fidelity_markdown(
            authority,
            fixtures["summary"],
            overlay_accounted=overlay_accounted,
            overlay_expected=len(ORIENTATIONS),
        ),
        encoding="utf-8",
    )
    return {"authority": path, "parity": parity_path, "docs": docs, "frame": authority}


def build_corrective_wizard_parity(*, root: Path = ROOT) -> CommandResult:
    capture = build_wizard_parity_capture_status(root=root)
    overlay = build_ou_optimal_overlay_provenance(root=root)
    fixtures = build_wizard_golden_fixtures(root=root)
    mutations = run_wizard_mode_mutation_tests(root=root)
    authority = build_wizard_mode_authority(root=root)
    frame = authority["frame"]
    paths = {
        "capture_status": Path(capture["status"]),
        "capture_manifest": Path(capture["capture_manifest"]),
        "ou_optimal_overlay_provenance": Path(overlay["path"]),
        "fixture_manifest": Path(fixtures["manifest"]),
        "fixture_audit": Path(fixtures["audit"]),
        "mutation_results": root / "reports" / "red_team" / "wizard_mode_mutation_results.csv",
        "mode_parity": Path(authority["parity"]),
        "mode_authority": Path(authority["authority"]),
        "mode_fidelity_docs": Path(authority["docs"]),
    }
    return CommandResult(
        paths=paths,
        summary={
            "status": "BLOCKED",
            "capture_cells_accounted": capture["summary"]["cells_accounted"],
            "capture_cells_captured": capture["summary"]["cells_captured"],
            "fixture_cells_accounted": fixtures["summary"]["cells_accounted"],
            "golden_capture_cells": fixtures["summary"]["golden_captures"],
            "missing_vendor_mode_cells": fixtures["summary"]["missing_vendor_modes"],
            "ou_optimal_overlay_status": overlay["status"],
            "ou_optimal_orientations_accounted": overlay["orientations_accounted"],
            "ou_optimal_expected_orientations": overlay["expected_orientations"],
            "vendor_parity_cells": int(frame["vendor_exact_mode_parity_proven"].sum()),
            "mutation_cases_passed": int(mutations["status"].eq("PASS").sum()),
            "blocker": "vendor_formula_parity_unproven_for_all_modes",
            "live_trading_authorized": False,
        },
    )


def _fidelity_markdown(
    authority: pd.DataFrame,
    summary: dict[str, Any],
    *,
    overlay_accounted: int,
    overlay_expected: int,
) -> str:
    lines = [
        "# Wizard and Hyperliquid Mode Fidelity",
        "",
        f"- Required mode/orientation cells: `{summary['required_cells']}`",
        f"- Golden dashboard captures: `{summary['golden_captures']}`",
        f"- Missing vendor-mode captures: `{summary['missing_vendor_modes']}`",
        f"- Vendor formula parity proven: `{int(authority['vendor_exact_mode_parity_proven'].sum())}`",
        "- Live trading authorized: `false`",
        "",
        "A dashboard label and a local formula with the same name are separate claims. Captured Wizard fields support diagnosis and research hypotheses. Until comparable vendor output series reproduce within declared tolerances, local computations remain `local_approximation` and cannot be described as Wizard-confirmed.",
        "",
        f"`OU (Optimal)` is a scanner boolean annotation layered onto a source exact mode, not an eighth Wizard pair-page mode. Current orientation provenance is `{overlay_accounted}/{overlay_expected}` in `wizard_ou_optimal_overlay_provenance.csv`. Any missing orientation remains blocked. Its formula semantics are also unproven, so it cannot receive a Wizard-exact claim.",
        "",
    ]
    return "\n".join(lines)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest() if path.is_file() else ""


def _json_scalar(value: Any) -> Any:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _truthy(value: Any) -> bool:
    return value is True or str(value).strip().lower() in {"1", "true", "yes", "pass", "captured"}


def _text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def _slug(value: str) -> str:
    return "_".join(value.lower().replace("(", " ").replace(")", " ").split())


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    result = build_corrective_wizard_parity()
    print(json.dumps({"summary": result.summary, "paths": {key: str(value) for key, value in result.paths.items()}}, indent=2))
