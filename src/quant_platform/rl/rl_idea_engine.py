from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from quant_platform.active_pipeline import CommandResult, ROOT


RL_IDEAS_COLUMNS = [
    "idea_id",
    "pair",
    "timeframe",
    "strategy",
    "regime",
    "entry_style",
    "exit_style",
    "source_policy",
    "confidence_score",
    "expected_return",
    "risk_proxy",
    "expected_trade_count",
    "zscore",
    "spread",
    "spread_slope",
    "beta_stability",
    "copula_calibration_score",
    "liquidity_score",
    "source",
    "reasoning",
    "evidence_path",
    "generated_at",
    "status",
]

RL_SIMILARITY_COLUMNS = [
    "idea_seed",
    "pair",
    "timeframe",
    "similar_pair",
    "similar_timeframe",
    "similarity_rank",
    "distance",
    "shared_features",
    "generated_at",
]

RL_IDEA_SUMMARY_COLUMNS = [
    "generated_ideas",
    "generated_similar_pairs",
    "policy_type",
    "blocker",
    "evidence_source",
    "generated_at",
]


def run_rl_idea_scout(
    root: Path = ROOT,
    pair_filter: str | None = None,
    top_ideas: int = 25,
    similarity_k: int = 6,
) -> CommandResult:
    """Produce RL idea artifacts as hypothesis outputs only."""

    reports = root / "reports" / "rl"
    agents = root / "reports" / "agents"
    reports.mkdir(parents=True, exist_ok=True)
    agents.mkdir(parents=True, exist_ok=True)

    training_path = reports / "rl_training_report.csv"
    dataset_path = root / "data" / "ml" / "trade_training_dataset.csv"
    dataset = _read_csv(dataset_path)
    eval_report = _read_csv(reports / "rl_evaluation_report.csv")
    training = _read_csv(training_path)
    policy_type = _extract_policy(training)
    timestamp = pd.Timestamp.utcnow().isoformat()

    if pair_filter and "pair" in dataset.columns:
        dataset = _filter_pairs(dataset, pair_filter)

    ideas_path = agents / "rl_ideas.csv"
    sim_path = agents / "rl_pair_similarity.csv"
    summary_path = agents / "rl_idea_summary.csv"

    if dataset.empty:
        _write_empty_idea_artifacts(
            ideas_path,
            sim_path,
            summary_path,
            timestamp,
            policy_type,
            "missing_trade_dataset",
            dataset_path,
        )
        return CommandResult(
            paths={"rl_ideas": ideas_path, "rl_pair_similarity": sim_path, "rl_idea_summary": summary_path},
            summary={"ideas": 0, "similar_pairs": 0, "policy_type": policy_type, "blocker": "missing_trade_dataset"},
        )

    ideas_frame = _build_rl_idea_rows(dataset, policy_type, eval_report, training, timestamp, top_ideas=max(1, int(top_ideas)))
    similarity_frame = _build_similarity_frame(
        dataset,
        top_pairs=max(1, min(10, len(ideas_frame))),
        similarity_k=max(1, int(similarity_k)),
        timestamp=timestamp,
    )

    ideas_frame.to_csv(ideas_path, index=False)
    similarity_frame.to_csv(sim_path, index=False)
    summary = pd.DataFrame(
        [
            {
                "generated_ideas": int(len(ideas_frame)),
                "generated_similar_pairs": int(len(similarity_frame)),
                "policy_type": policy_type,
                "blocker": "",
                "evidence_source": str(dataset_path),
                "generated_at": timestamp,
            }
        ]
    )
    summary.to_csv(summary_path, index=False)

    return CommandResult(
        paths={"rl_ideas": ideas_path, "rl_pair_similarity": sim_path, "rl_idea_summary": summary_path},
        summary={
            "ideas": int(len(ideas_frame)),
            "similar_pairs": int(len(similarity_frame)),
            "policy_type": policy_type,
            "blocker": "",
        },
    )


def _build_rl_idea_rows(
    dataset: pd.DataFrame,
    policy_type: str,
    eval_report: pd.DataFrame,
    training: pd.DataFrame,
    timestamp: str,
    top_ideas: int,
) -> pd.DataFrame:
    if dataset.empty:
        return pd.DataFrame(columns=RL_IDEAS_COLUMNS)

    candidates = dataset.copy()
    candidates["expected_return"] = _to_numeric_series(_select_return_series(candidates))
    candidates["risk_proxy"] = _to_numeric_series(
        candidates.get("profit_after_cost", candidates.get("realized_return", pd.Series(0.0, index=candidates.index)))
    )

    for column in [
        "zscore",
        "spread",
        "spread_slope",
        "beta_stability",
        "copula_calibration_score",
        "liquidity_score",
    ]:
        if column in candidates.columns:
            candidates[column] = _to_numeric_series(candidates[column])

    return_abs = candidates["expected_return"].abs()
    max_abs = return_abs.replace([np.inf, -np.inf], 0.0).max()
    if pd.isna(max_abs) or max_abs <= 0:
        max_abs = 1.0
    candidates["confidence_score"] = (0.1 + 0.8 * (return_abs / max_abs)).clip(0.0, 1.0)
    candidates["expected_trade_count"] = _safe_int(candidates.get("closed_trades", 1), default=1)
    candidates["source"] = "rl_research_backtest"
    candidates["reasoning"] = candidates.apply(_reasoning_row, axis=1)
    candidates["source_policy"] = (
        _coalesce(candidates.iloc[0], ["policy", "source_policy", "policy_name", "policy_type"], default=policy_type)
        if not candidates.empty
        else policy_type
    )
    candidates["evidence_path"] = _report_or_dataset_path(training, eval_report)
    candidates["status"] = "candidate"
    candidates["generated_at"] = timestamp

    sort_keys: list[str] = ["expected_return", "confidence_score"]
    if "zscore" in candidates.columns:
        sort_keys.append("zscore")
    candidates = candidates.sort_values(sort_keys, ascending=[False] * len(sort_keys), na_position="last")

    ranked = candidates.head(max(1, min(top_ideas, len(candidates)))).copy().reset_index(drop=True)
    if ranked.empty:
        return pd.DataFrame(columns=RL_IDEAS_COLUMNS)

    ranked["idea_id"] = [f"rl_idea_{i+1:04d}" for i in range(len(ranked))]
    ranked["entry_style"] = _resolve_col(ranked, "entry_style", default="quantile_policy")
    ranked["exit_style"] = _resolve_col(ranked, "exit_style", default="quantile_boundary")
    ranked["pair"] = _resolve_col(ranked, "pair")
    ranked["timeframe"] = _resolve_col(ranked, "timeframe", default="")
    ranked["strategy"] = _resolve_col(ranked, "strategy", default=_resolve_col(ranked, "exact_mode", default="unspecified"))
    ranked["regime"] = _resolve_col(ranked, "regime", default="unknown")

    ranked["expected_return"] = _to_numeric_series(ranked["expected_return"]).replace([np.inf, -np.inf], 0.0)
    ranked["confidence_score"] = _to_numeric_series(ranked["confidence_score"]).clip(0.0, 1.0)
    ranked["risk_proxy"] = _to_numeric_series(ranked["risk_proxy"]).clip(lower=0.0)
    ranked["expected_trade_count"] = _safe_int(ranked["expected_trade_count"], default=1)
    ranked["zscore"] = _to_numeric_series(ranked.get("zscore", pd.Series(0.0, index=ranked.index)))
    ranked["spread"] = _to_numeric_series(ranked.get("spread", pd.Series(0.0, index=ranked.index)))
    ranked["spread_slope"] = _to_numeric_series(ranked.get("spread_slope", pd.Series(0.0, index=ranked.index)))
    ranked["beta_stability"] = _to_numeric_series(ranked.get("beta_stability", pd.Series(0.0, index=ranked.index)))
    ranked["copula_calibration_score"] = _to_numeric_series(
        ranked.get("copula_calibration_score", pd.Series(0.0, index=ranked.index))
    )
    ranked["liquidity_score"] = _to_numeric_series(ranked.get("liquidity_score", pd.Series(0.0, index=ranked.index)))

    result = ranked[[c for c in RL_IDEAS_COLUMNS if c in ranked.columns]].copy()
    for field in RL_IDEAS_COLUMNS:
        if field not in result.columns:
            result[field] = ""
    return result[RL_IDEAS_COLUMNS]


def _build_similarity_frame(
    dataset: pd.DataFrame,
    top_pairs: int,
    similarity_k: int,
    timestamp: str,
) -> pd.DataFrame:
    feature_frame = _pair_feature_aggregates(dataset)
    if feature_frame.empty or len(feature_frame) < 2:
        return pd.DataFrame(columns=RL_SIMILARITY_COLUMNS)

    feature_columns = _select_similarity_features(feature_frame)
    if not feature_columns:
        return pd.DataFrame(columns=RL_SIMILARITY_COLUMNS)

    features = feature_frame[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).replace([np.inf, -np.inf], 0.0)
    std = features.std().replace(0.0, 1.0)
    features = (features - features.mean()) / std
    values = features.to_numpy(dtype=float)
    if values.ndim != 2 or len(values) < 2:
        return pd.DataFrame(columns=RL_SIMILARITY_COLUMNS)

    rows: list[dict[str, object]] = []
    seeds = feature_frame.head(top_pairs).reset_index(drop=True)
    for seed_idx in range(len(seeds)):
        seed_row = seeds.iloc[seed_idx]
        seed_vec = values[seed_idx]
        distances = np.linalg.norm(values - seed_vec, axis=1)
        nearest = np.argsort(distances)
        rank_idx = 1
        for neighbor_idx in nearest[1:]:
            if len(rows) >= similarity_k * len(feature_frame):
                break
            if neighbor_idx == seed_idx:
                continue
            if rank_idx > similarity_k:
                break

            similar_row = feature_frame.iloc[int(neighbor_idx)]
            rows.append(
                {
                    "idea_seed": str(seeds.iloc[seed_idx].get("pair", "")),
                    "pair": str(seed_row.get("pair", "")),
                    "timeframe": str(seed_row.get("timeframe", "")),
                    "similar_pair": str(similar_row.get("pair", "")),
                    "similar_timeframe": str(similar_row.get("timeframe", "")),
                    "similarity_rank": int(rank_idx),
                    "distance": float(distances[neighbor_idx]),
                    "shared_features": ";".join(feature_columns),
                    "generated_at": timestamp,
                }
            )
            rank_idx += 1

    if not rows:
        return pd.DataFrame(columns=RL_SIMILARITY_COLUMNS)
    frame = pd.DataFrame(rows)
    frame["similarity_rank"] = frame["distance"].rank(method="dense").astype(int)
    return frame[RL_SIMILARITY_COLUMNS]


def _pair_feature_aggregates(dataset: pd.DataFrame) -> pd.DataFrame:
    if dataset.empty:
        return pd.DataFrame(columns=["pair", "timeframe"])

    frame = dataset.copy()
    for col in ["pair", "timeframe"]:
        if col not in frame.columns:
            frame[col] = ""
    frame["pair"] = _safe_pair_label(frame["pair"])
    frame["timeframe"] = frame.get("timeframe", "").astype(str).fillna("")

    candidate_columns = [
        "expected_return",
        "risk_proxy",
        "confidence_score",
        "zscore",
        "spread",
        "spread_slope",
        "beta_stability",
        "copula_calibration_score",
        "liquidity_score",
    ]
    available = [column for column in candidate_columns if column in frame.columns]
    if not available:
        available = [column for column in frame.columns if column not in {"pair", "timeframe"}]

    grouped = frame.groupby(["pair", "timeframe"], dropna=False)
    if not available:
        return pd.DataFrame(columns=["pair", "timeframe"])

    aggregated = grouped[available].mean(numeric_only=True).reset_index(drop=True)
    if aggregated.empty:
        return pd.DataFrame(columns=["pair", "timeframe"])

    keys = grouped[['pair', 'timeframe']].first().reset_index(drop=True)
    if keys.empty:
        return pd.DataFrame(columns=["pair", "timeframe"])

    return pd.concat([keys, aggregated.drop(columns=["pair", "timeframe"], errors="ignore")], axis=1)


def _select_similarity_features(feature_frame: pd.DataFrame) -> list[str]:
    requested = {
        "expected_return",
        "risk_proxy",
        "confidence_score",
        "zscore",
        "spread",
        "spread_slope",
        "beta_stability",
        "copula_calibration_score",
        "liquidity_score",
    }
    candidates = [column for column in requested if column in feature_frame.columns]
    normalized = feature_frame[candidates].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return [column for column in candidates if normalized[column].std() > 0 or normalized[column].ne(0.0).any()]


def _report_or_dataset_path(training: pd.DataFrame, eval_report: pd.DataFrame) -> str:
    if not training.empty:
        return "reports/rl/rl_training_report.csv"
    if not eval_report.empty:
        return "reports/rl/rl_evaluation_report.csv"
    return "data/ml/trade_training_dataset.csv"


def _select_return_series(frame: pd.DataFrame) -> pd.Series:
    for column in ["profit_after_cost", "trade_return", "return", "returns", "realized_return", "label"]:
        if column in frame.columns:
            return frame[column]
    return pd.Series(0.0, index=frame.index)


def _to_numeric_series(value) -> pd.Series:
    try:
        return pd.to_numeric(value, errors="coerce").fillna(0.0)
    except Exception:
        return pd.Series(0.0, index=getattr(value, "index", pd.RangeIndex(0)))


def _safe_int(series, default: int = 0) -> pd.Series:
    if not hasattr(series, "fillna"):
        converted = pd.to_numeric(pd.Series([series]), errors="coerce")
    else:
        converted = pd.to_numeric(series, errors="coerce")
    return converted.fillna(default).astype(int).clip(lower=default).replace([np.inf, -np.inf], default)


def _resolve_col(frame: pd.DataFrame, column: str, default: str = "") -> pd.Series:
    if column in frame.columns:
        return frame[column].astype(str).fillna(default)
    return pd.Series([default] * len(frame), index=frame.index)


def _safe_pair_label(values: pd.Series | list[str]) -> pd.Series:
    if isinstance(values, pd.Series):
        series = values
    else:
        series = pd.Series(values)
    return series.astype(str).str.replace("/", "-", regex=False).str.strip().replace({"": "unknown"})


def _reasoning_row(row: pd.Series) -> str:
    return (
        f"rl-policy={_coalesce(row, ['source_policy', 'strategy', 'exact_mode', 'mode', 'source_strategy'], 'unknown')}; "
        f"return={row.get('expected_return', 0.0)}; "
        f"confidence={row.get('confidence_score', 0.0)}"
    )


def _coalesce(row: pd.Series, keys: list[str], default: str = "") -> str:
    for key in keys:
        if key in row and pd.notna(row[key]) and str(row[key]).strip():
            return str(row[key])
    return default


def _extract_policy(training: pd.DataFrame) -> str:
    if training.empty:
        return "safe_quantile_baseline"
    for column in ["policy", "policy_name", "policy_type", "source_policy"]:
        if column in training.columns:
            value = _coalesce(training.iloc[0], [column], default="")
            if value:
                return value
    return "safe_quantile_baseline"


def _filter_pairs(dataset: pd.DataFrame, pair_filter: str) -> pd.DataFrame:
    normalized_filter = str(pair_filter).replace("/", "-").replace("_", "-").strip().lower()
    pairs = dataset["pair"].astype(str).str.replace("/", "-", regex=False).str.replace("_", "-", regex=False).str.lower()
    return dataset[pairs.str.contains(normalized_filter, case=False, regex=False)]


def _write_empty_idea_artifacts(
    ideas_path: Path,
    sim_path: Path,
    summary_path: Path,
    timestamp: str,
    policy_type: str,
    blocker: str,
    dataset_path: Path,
) -> None:
    empty_ideas = pd.DataFrame(columns=RL_IDEAS_COLUMNS)
    empty_sim = pd.DataFrame(columns=RL_SIMILARITY_COLUMNS)
    empty_summary = pd.DataFrame(
        [
            {
                "generated_ideas": 0,
                "generated_similar_pairs": 0,
                "policy_type": policy_type,
                "blocker": blocker,
                "evidence_source": str(dataset_path),
                "generated_at": timestamp,
            }
        ],
        columns=RL_IDEA_SUMMARY_COLUMNS,
    )
    empty_ideas.to_csv(ideas_path, index=False)
    empty_sim.to_csv(sim_path, index=False)
    empty_summary.to_csv(summary_path, index=False)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
