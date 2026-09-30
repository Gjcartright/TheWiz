"""Fail-closed readiness audit for student, bandit, and later offline-RL data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import pandas as pd

from quant_platform.orchestration.corrective_runtime import atomic_write_csv, atomic_write_text
from quant_platform.orchestration.teacher_contracts import (
    EXACT_MODES,
    MATH_V2,
    normalize_exact_mode,
)

STUDENT_TRAINING_COLUMNS: tuple[str, ...] = (
    "training_event_id",
    "context_id",
    "candidate_id",
    "pair",
    "timeframe",
    "exact_mode",
    "proposed_action",
    "feature_timestamp",
    "label_timestamp",
    "point_in_time_status",
    "math_version",
    "source_system",
    "label_source",
    "uses_wizard_as_label",
    "uses_dashboard_hindsight",
    "profit_after_cost",
    "good_trade",
    "max_adverse_excursion",
    "max_favorable_excursion",
    "hold_bars",
    "exit_reason",
    "action_propensity",
    "evidence_path",
)


@dataclass(frozen=True)
class StudentReadinessPolicy:
    min_rows: int = 250
    min_pairs: int = 5
    min_timeframes: int = 2
    min_label_classes: int = 2
    min_rows_per_mode: int = 25
    max_pair_share: float = 0.50
    max_mode_share: float = 0.50
    required_math_version: str = MATH_V2

    def __post_init__(self) -> None:
        if self.required_math_version != MATH_V2:
            raise ValueError("student readiness requires the current math version")


def audit_student_training_dataset(
    dataset: pd.DataFrame,
    *,
    policy: StudentReadinessPolicy | None = None,
) -> pd.DataFrame:
    """Return explicit pass/block rows for supervised and adaptive learning."""

    policy = policy or StudentReadinessPolicy()
    rows: list[dict[str, object]] = []
    if not isinstance(dataset, pd.DataFrame) or not dataset.columns.is_unique:
        _check(rows, "unique_column_schema", False, "invalid or duplicate columns",
               "DataFrame with unique columns", "invalid_student_column_schema")
        return pd.DataFrame(rows)
    missing_columns = sorted(set(STUDENT_TRAINING_COLUMNS) - set(dataset.columns))
    _check(
        rows,
        name="required_schema",
        passed=not missing_columns,
        observed="complete" if not missing_columns else ";".join(missing_columns),
        required="all canonical student-training columns",
        blocker="missing_required_columns",
    )
    if missing_columns:
        return pd.DataFrame(rows)

    working = dataset.copy()
    feature_time = pd.to_datetime(working["feature_timestamp"].map(_explicit_timestamp), utc=True)
    label_time = pd.to_datetime(working["label_timestamp"].map(_explicit_timestamp), utc=True)
    valid_time = feature_time.notna() & label_time.notna() & (feature_time < label_time)
    normalized_modes = working["exact_mode"].map(_safe_mode)
    identities_valid = all(bool(working[name].map(_identifier).all()) for name in
        ("training_event_id", "context_id", "candidate_id", "pair", "timeframe", "source_system", "label_source", "evidence_path"))
    _check(rows, "nonempty_identities", identities_valid, identities_valid,
           "explicit nonempty identifiers and lineage labels", "invalid_training_identity")
    _check(rows, "unique_training_events", bool(working["training_event_id"].map(_identifier).all()) and working["training_event_id"].is_unique,
           len(working), "one row per training_event_id", "duplicate_training_event_id")
    pair_count = int(working.loc[working["pair"].map(_identifier), "pair"].nunique())
    timeframe_count = int(working.loc[working["timeframe"].map(_identifier), "timeframe"].nunique())
    labels = _finite_numbers(working["good_trade"], allow_bool=True)
    labels_valid = labels.notna() & working["good_trade"].map(_binary_label)
    label_count = int(labels.loc[labels_valid].nunique())
    _check(rows, "binary_trade_labels", bool(labels_valid.all()), int(labels_valid.sum()),
           "finite binary good_trade labels on every row", "invalid_binary_trade_label")
    excursions = [_finite_numbers(working[name]) for name in ("max_adverse_excursion", "max_favorable_excursion")]
    # Producers retain signed MAE or adverse-magnitude conventions. Do not
    # invent a sign migration here; only reject unavailable/nonfinite labels.
    _check(rows, "finite_excursion_labels", all(bool(values.notna().all()) for values in excursions),
           len(working), "finite excursion labels; sign/unit qualification remains separate", "invalid_excursion_label")
    hold = _finite_numbers(working["hold_bars"])
    hold_valid = hold.notna() & hold.ge(0) & hold.mod(1).eq(0)
    _check(rows, "holding_bar_labels", bool(hold_valid.all()), int(hold_valid.sum()),
           "finite nonnegative integral hold bars", "invalid_holding_bar_label")
    pair_share = _largest_share(working["pair"])
    raw_mode_share = _largest_share(normalized_modes)
    mode_share = _weighted_largest_share(normalized_modes, working.get("sample_weight"))
    observed_modes = {mode for mode in normalized_modes.dropna().astype(str)}
    required_modes = {mode.value for mode in EXACT_MODES}
    rows_per_mode = normalized_modes.value_counts()
    minimum_mode_rows = int(rows_per_mode.min()) if not rows_per_mode.empty else 0

    _check(rows, "minimum_rows", len(working) >= policy.min_rows, len(working), policy.min_rows, "insufficient_training_rows")
    _check(rows, "minimum_pairs", pair_count >= policy.min_pairs, pair_count, policy.min_pairs, "insufficient_pair_coverage")
    _check(
        rows,
        "minimum_timeframes",
        timeframe_count >= policy.min_timeframes,
        timeframe_count,
        policy.min_timeframes,
        "insufficient_timeframe_coverage",
    )
    _check(
        rows,
        "label_diversity",
        label_count >= policy.min_label_classes,
        label_count,
        policy.min_label_classes,
        "one_class_or_missing_labels",
    )
    _check(
        rows,
        "all_exact_modes",
        observed_modes == required_modes,
        ";".join(sorted(observed_modes)),
        ";".join(sorted(required_modes)),
        "incomplete_exact_mode_coverage",
    )
    _check(
        rows,
        "minimum_rows_per_mode",
        minimum_mode_rows >= policy.min_rows_per_mode,
        minimum_mode_rows,
        policy.min_rows_per_mode,
        "insufficient_mode_specialist_training_rows",
        scope="supervised_student",
    )
    _check(
        rows,
        "timestamp_order",
        bool(valid_time.all()),
        f"valid={int(valid_time.sum())}/{len(working)}",
        "feature_timestamp < label_timestamp for every row",
        "future_leakage_or_invalid_timestamp",
    )
    point_in_time = working["point_in_time_status"].astype(str).str.lower().eq("confirmed")
    _check(
        rows,
        "point_in_time_features",
        bool(point_in_time.all()),
        f"confirmed={int(point_in_time.sum())}/{len(working)}",
        "all rows confirmed",
        "non_point_in_time_features",
    )
    math_v2 = working["math_version"].astype(str).eq(policy.required_math_version)
    _check(
        rows,
        "math_v2_lineage",
        bool(math_v2.all()),
        f"verified={int(math_v2.sum())}/{len(working)}",
        policy.required_math_version,
        "unverified_math_version",
    )
    wizard_labels = _bool_series(working["uses_wizard_as_label"])
    _check(rows, "known_provenance_flags", bool(wizard_labels.notna().all() and
           _bool_series(working["uses_dashboard_hindsight"]).notna().all()), len(working),
           "explicit true or false flags; unknown is not false", "unknown_provenance_flag")
    _check(
        rows,
        "wizard_not_label_authority",
        not bool(wizard_labels.any()),
        int(wizard_labels.sum()),
        0,
        "wizard_hindsight_used_as_label",
    )
    dashboard_hindsight = _bool_series(working["uses_dashboard_hindsight"])
    _check(
        rows,
        "no_dashboard_hindsight",
        not bool(dashboard_hindsight.any()),
        int(dashboard_hindsight.sum()),
        0,
        "dashboard_hindsight_feature_present",
    )
    after_cost = _finite_numbers(working["profit_after_cost"]).notna()
    _check(
        rows,
        "after_cost_labels",
        bool(after_cost.all()),
        f"present={int(after_cost.sum())}/{len(working)}",
        "all rows",
        "missing_after_cost_label",
    )
    evidence = working["evidence_path"].astype(str).str.strip().ne("") & working["evidence_path"].notna()
    _check(
        rows,
        "evidence_lineage",
        bool(evidence.all()),
        f"present={int(evidence.sum())}/{len(working)}",
        "all rows",
        "missing_evidence_path",
    )
    _check(
        rows,
        "pair_concentration",
        pair_share <= policy.max_pair_share,
        round(pair_share, 6),
        f"<= {policy.max_pair_share}",
        "single_pair_concentration",
    )
    _check(
        rows,
        "mode_concentration",
        mode_share <= policy.max_mode_share,
        f"effective={mode_share:.6f};raw={raw_mode_share:.6f}",
        f"<= {policy.max_mode_share}",
        "single_mode_concentration",
    )
    if "sample_weight" in working.columns:
        sample_weight = _finite_numbers(working["sample_weight"])
        valid_weight = sample_weight.notna() & sample_weight.gt(0.0)
        _check(
            rows,
            "sample_weight_validity",
            bool(valid_weight.all()),
            f"valid={int(valid_weight.sum())}/{len(working)}",
            "finite positive sample weight for every row",
            "invalid_supervised_sample_weight",
            scope="supervised_student",
        )
    if "record_granularity" in working.columns:
        trade_level = working["record_granularity"].astype(str).str.lower().eq("trade_entry")
        _check(
            rows,
            "trade_level_granularity",
            bool(trade_level.all()),
            f"trade_entry={int(trade_level.sum())}/{len(working)}",
            "every row represents one candidate trade entry",
            "mode_summary_rows_not_training_eligible",
            scope="supervised_student",
        )
    if "training_eligible" in working.columns:
        eligible = _bool_series(working["training_eligible"])
        _check(
            rows,
            "row_training_eligibility",
            bool(eligible.notna().all() and eligible.all()),
            f"eligible={int(eligible.sum())}/{len(working)}",
            "all rows explicitly training eligible",
            "rows_marked_research_only",
            scope="supervised_student",
        )
    propensity = _finite_numbers(working["action_propensity"])
    valid_propensity = propensity.notna() & propensity.gt(0.0) & propensity.le(1.0)
    _check(
        rows,
        "bandit_action_propensity",
        bool(valid_propensity.all()),
        f"valid={int(valid_propensity.sum())}/{len(working)}",
        "0 < action_propensity <= 1 for every row",
        "bandit_off_policy_evaluation_not_ready",
        scope="contextual_bandit",
    )
    if "propensity_source" in working.columns:
        logged = working["propensity_source"].astype(str).str.lower().eq("logged_behavior_policy")
        _check(
            rows,
            "bandit_propensity_lineage",
            bool(logged.all()),
            f"logged={int(logged.sum())}/{len(working)}",
            "logged_behavior_policy for every row",
            "synthetic_or_missing_propensity_lineage",
            scope="contextual_bandit",
        )
    else:
        _check(rows, "bandit_propensity_lineage", False, "missing", "logged_behavior_policy for every row",
               "synthetic_or_missing_propensity_lineage", scope="contextual_bandit")
    if "behavior_policy_exploratory" in working.columns:
        exploratory = _bool_series(working["behavior_policy_exploratory"])
        _check(
            rows,
            "bandit_exploratory_action_support",
            bool(exploratory.notna().all() and exploratory.all()),
            f"exploratory={int(exploratory.sum())}/{len(working)}",
            "logged nonzero support for alternative actions at every context",
            "deterministic_behavior_policy_has_no_counterfactual_support",
            scope="contextual_bandit",
        )
    else:
        _check(rows, "bandit_exploratory_action_support", False, "missing", "explicit logged alternative-action support",
               "deterministic_behavior_policy_has_no_counterfactual_support", scope="contextual_bandit")
    return pd.DataFrame(rows)


def write_student_training_readiness(
    *,
    root: Path,
    dataset_path: Path | None = None,
    policy: StudentReadinessPolicy | None = None,
) -> dict[str, Path | str | int]:
    dataset_path = dataset_path or root / "data" / "ml" / "student_teacher_training_dataset.csv"
    output_dir = root / "reports" / "orchestration" / "teacher_council"
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = output_dir / "student_training_readiness.csv"
    schema_path = output_dir / "student_training_schema.csv"
    markdown_path = output_dir / "student_training_readiness.md"

    dataset = _read_csv(dataset_path)
    audit = audit_student_training_dataset(dataset, policy=policy) if not dataset.empty else _missing_dataset_audit(dataset_path)
    atomic_write_csv(audit, audit_path, index=False)
    atomic_write_csv(pd.DataFrame(
        [{"column": column, "required": True, "purpose": _column_purpose(column)} for column in STUDENT_TRAINING_COLUMNS]
    ), schema_path, index=False)
    blocked = audit.loc[audit["status"] == "BLOCKED"]
    supervised_blocked = blocked.loc[blocked["scope"].isin(["all_learning", "supervised_student"])]
    bandit_blocked = blocked.loc[blocked["scope"].isin(["all_learning", "contextual_bandit"])]
    atomic_write_text(markdown_path, _readiness_markdown(
            audit,
            dataset_path=dataset_path,
            supervised_status="BLOCKED" if not supervised_blocked.empty else "READY_FOR_SHADOW_TRAINING",
            bandit_status="BLOCKED" if not bandit_blocked.empty else "READY_FOR_SHADOW_EVALUATION",
        ), encoding="utf-8")
    return {
        "audit": audit_path,
        "schema": schema_path,
        "markdown": markdown_path,
        "supervised_status": "BLOCKED" if not supervised_blocked.empty else "READY_FOR_SHADOW_TRAINING",
        "bandit_status": "BLOCKED" if not bandit_blocked.empty else "READY_FOR_SHADOW_EVALUATION",
        "rows": len(dataset),
    }


def _missing_dataset_audit(path: Path) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "scope": "all_learning",
                "check": "training_dataset_exists",
                "status": "BLOCKED",
                "observed": str(path),
                "required": "canonical student-teacher dataset",
                "reason": "training dataset is missing or empty",
                "blocker": "missing_student_training_dataset",
            }
        ]
    )


def _check(
    rows: list[dict[str, object]],
    name: str,
    passed: bool,
    observed: object,
    required: object,
    blocker: str,
    *,
    scope: str = "all_learning",
) -> None:
    rows.append(
        {
            "scope": scope,
            "check": name,
            "status": "PASS" if passed else "BLOCKED",
            "observed": observed,
            "required": required,
            "reason": "requirement satisfied" if passed else blocker.replace("_", " "),
            "blocker": "" if passed else blocker,
        }
    )


def _safe_mode(value: object) -> str | None:
    try:
        return normalize_exact_mode(str(value)).value
    except ValueError:
        return None


def _largest_share(series: pd.Series) -> float:
    values = series.dropna().astype(str)
    if values.empty:
        return 1.0
    return float(values.value_counts(normalize=True).iloc[0])


def _weighted_largest_share(series: pd.Series, weights: pd.Series | None) -> float:
    if weights is None:
        return _largest_share(series)
    frame = pd.DataFrame(
        {
            "value": series,
            "weight": _finite_numbers(weights),
        }
    ).dropna()
    frame = frame.loc[frame["weight"] > 0.0]
    if frame.empty:
        return 1.0
    # Scale before aggregation to avoid finite large weights overflowing.
    frame["weight"] = frame["weight"] / frame["weight"].max()
    totals = frame.groupby(frame["value"].astype(str))["weight"].sum()
    total = float(totals.sum())
    return float(totals.max() / total) if total > 0.0 else 1.0


def _bool_series(series: pd.Series) -> pd.Series:
    def parse(value):
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, Integral) and value in (0, 1):
            return bool(value)
        if isinstance(value, str):
            token = value.strip().lower()
            if token in {"1", "true", "yes", "y"}:
                return True
            if token in {"0", "false", "no", "n"}:
                return False
        return pd.NA
    return series.map(parse).astype("boolean")


def _identifier(value: object) -> bool:
    return isinstance(value, str) and bool(value) and value == value.strip() and value.lower() not in {"nan", "none", "null", "unknown"}


def _binary_label(value: object) -> bool:
    if isinstance(value, str):
        try:
            # Do not round an exact nonbinary decimal string into 0 or 1.
            exact = Decimal(value)
            return exact.is_finite() and exact in (Decimal(0), Decimal(1))
        except InvalidOperation:
            return False
    return isinstance(value, (Real, np.bool_)) and value in (0, 1)


def _finite_numbers(series: pd.Series, *, allow_bool: bool = False) -> pd.Series:
    def parse(value):
        if isinstance(value, (bool, np.bool_)) and not allow_bool:
            return float("nan")
        if not isinstance(value, (str, Real)):
            return float("nan")
        try:
            result = float(value)
        except (ValueError, TypeError, OverflowError):
            return float("nan")
        return result if np.isfinite(result) else float("nan")
    return series.map(parse).astype(float)


def _explicit_timestamp(value: object):
    if not isinstance(value, (str, datetime, pd.Timestamp)):
        return pd.NaT
    try:
        stamp = pd.Timestamp(value)
        if pd.isna(stamp) or stamp.tzinfo is None:
            return pd.NaT
        return stamp.tz_convert("UTC").as_unit("ns")
    except (ValueError, TypeError, OverflowError):
        return pd.NaT


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame()


def _column_purpose(column: str) -> str:
    purposes = {
        "feature_timestamp": "prove every feature existed before entry",
        "label_timestamp": "prove the outcome occurred after the proposed action",
        "math_version": "block Math V1 and proxy calculations from training",
        "uses_wizard_as_label": "prevent Crypto Wizards hindsight from becoming ground truth",
        "uses_dashboard_hindsight": "block full-sample dashboard fields from features",
        "profit_after_cost": "train on fees, slippage, funding, and execution-aware outcomes",
        "action_propensity": "support contextual-bandit off-policy evaluation",
        "evidence_path": "preserve row-level lineage",
    }
    return purposes.get(column, "canonical teacher/student training field")


def _readiness_markdown(
    audit: pd.DataFrame,
    *,
    dataset_path: Path,
    supervised_status: str,
    bandit_status: str,
) -> str:
    return "\n".join(
        [
            "# Student Stack Readiness",
            "",
            f"- Dataset: `{dataset_path}`",
            f"- Supervised router/outcome student: **{supervised_status}**",
            f"- Contextual bandit: **{bandit_status}**",
            "- Offline RL: **BLOCKED until supervised and bandit evidence gates pass**",
            "- Execution authority: **none**",
            "",
            audit.to_markdown(index=False),
            "",
        ]
    )
