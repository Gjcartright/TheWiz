from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from functools import lru_cache
import json
import re

import pandas as pd

from quant_platform.derived_features import add_derived_beta_from_prices
from quant_platform.experiments import PairDataset


CANONICAL_ALIASES: dict[str, tuple[str, ...]] = {
    "timestamp": ("timestamp", "time", "datetime", "date"),
    "pair": ("pair", "symbol_pair", "market_pair", "asset_pair"),
    "asset_x": ("asset_x", "x_asset", "base_asset", "asset1", "asset_1"),
    "asset_y": ("asset_y", "y_asset", "quote_asset", "asset2", "asset_2"),
    "price_x": ("price_x", "x_price", "asset_x_price", "price_asset_x"),
    "price_y": ("price_y", "y_price", "asset_y_price", "price_asset_y"),
    "cointegration": ("cointegration", "is_cointegrated", "coint_eg", "johansen_coint"),
    "cointegration_pvalue": ("cointegration_pvalue", "cointegration_p_value", "coint_pvalue", "coint_eg_p", "pvalue"),
    "hedge_ratio": ("hedge_ratio", "hedge", "beta_hedge"),
    "beta": ("beta", "pair_beta"),
    "ecm_x": ("ecm_x", "ecm(x)", "ecm_asset_x", "ecmX"),
    "ecm_y": ("ecm_y", "ecm(y)", "ecm_asset_y", "ecmY"),
    "ecm_strength": ("ecm_strength", "ecm_score"),
    "half_life": ("half_life", "halflife"),
    "hurst": ("hurst", "hurst_exponent"),
    "zscore": ("zscore", "z_score", "z", "spread_zscore", "zscore_last"),
    "rolling_zscore": ("rolling_zscore", "zscore_roll", "zscore_rolling", "rolling_z_score", "zscore_roll_last"),
    "spread": ("spread", "pair_spread", "residual_spread"),
    "pearson": ("pearson", "pearson_corr", "pearson_correlation"),
    "spearman": ("spearman", "spearman_corr", "spearman_correlation"),
    "kendall": ("kendall", "kendall_tau"),
    "copula": ("copula", "copula_family"),
    "u1_given_u2": ("u1_given_u2", "p_u1_given_u2", "conditional_u1_given_u2"),
    "u2_given_u1": ("u2_given_u1", "p_u2_given_u1", "conditional_u2_given_u1"),
    "conditional_probability_distortion": (
        "conditional_probability_distortion",
        "conditional_probabilities",
        "copula_dislocation",
    ),
    "tail_dependence": ("tail_dependence", "copula_tail_dependence"),
    "sharpe": ("sharpe", "sharpe_ratio"),
    "sortino": ("sortino", "sortino_ratio"),
    "var": ("var", "value_at_risk"),
    "cvar": ("cvar", "expected_shortfall"),
    "drawdown": ("drawdown", "max_drawdown", "mdd"),
    "win_rate": ("win_rate", "hit_rate"),
    "ml_confidence": ("ml_confidence", "model_confidence"),
    "profile_match": ("profile_match", "profile_score"),
    "ou_optimal": ("ou_optimal", "ou_score"),
    "regime": ("regime", "market_regime", "state"),
}


IMPORTANT_FIELDS = {
    "pair",
    "timestamp",
    "spread",
    "zscore",
    "rolling_zscore",
    "conditional_probability_distortion",
    "u1_given_u2",
    "u2_given_u1",
    "ecm_x",
    "ecm_y",
    "ecm_strength",
    "half_life",
    "hurst",
    "regime",
}


@dataclass(frozen=True)
class FixturePayload:
    endpoint: str
    path: Path
    payload: Any


@lru_cache(maxsize=8192)
def snake_case(value: str) -> str:
    """Normalize arbitrary keys to snake_case with stable, cached transformation."""
    if not value:
        return ""
    value = value.strip().replace("-", "_").replace(" ", "_").replace("/", "_")
    value = re.sub(r"(?<!^)(?=[A-Z])", "_", value)
    value = re.sub(r"[^0-9a-zA-Z_()]+", "_", value)
    return re.sub(r"_+", "_", value).strip("_").lower()


def load_fixture_payloads(input_dir: str | Path) -> list[FixturePayload]:
    root = Path(input_dir)
    payloads: list[FixturePayload] = []
    for path in sorted(root.glob("**/*")):
        if path.suffix.lower() == ".json":
            payloads.append(FixturePayload(endpoint=path.stem, path=path, payload=json.loads(path.read_text(encoding="utf-8"))))
        elif path.suffix.lower() == ".csv":
            payloads.append(FixturePayload(endpoint=path.stem, path=path, payload=pd.read_csv(path).to_dict("records")))
    return payloads


def _flatten_dict(record: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in record.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(_flatten_dict(value, full_key))
        elif isinstance(value, list):
            flattened[full_key] = json.dumps(value, sort_keys=True)
        else:
            flattened[full_key] = value
    return flattened


def _record_score(record: dict[str, Any]) -> int:
    normalized = {snake_case(key.split(".")[-1]) for key in record}
    score = 0
    score += 3 if normalized.intersection({"pair", "symbol_pair", "market_pair"}) else 0
    score += 3 if normalized.intersection({"spread", "zscore", "z_score", "conditional_probability_distortion"}) else 0
    score += 2 if normalized.intersection({"timestamp", "time", "datetime"}) else 0
    return score


def _extract_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        records: list[dict[str, Any]] = []
        for item in payload:
            records.extend(_extract_records(item))
        return records
    if not isinstance(payload, dict):
        return []

    child_records: list[dict[str, Any]] = []
    for value in payload.values():
        if isinstance(value, list):
            for item in value:
                child_records.extend(_extract_records(item))
        elif isinstance(value, dict):
            child_records.extend(_extract_records(value))

    flattened = _flatten_dict(payload)
    if not child_records or _record_score(flattened) >= 4:
        return [flattened]
    return child_records


def _alias_lookup(columns: Iterable[str]) -> dict[str, list[str]]:
    leaf_pairs = [(snake_case(column.split(".")[-1]), column) for column in columns]
    full_pairs = [(snake_case(column), column) for column in columns]
    lookup: dict[str, list[str]] = {}
    for canonical, aliases in CANONICAL_ALIASES.items():
        sources: list[str] = []
        for alias in aliases:
            key = snake_case(alias)
            sources.extend(column for leaf, column in leaf_pairs if leaf == key)
            sources.extend(column for full, column in full_pairs if full == key)
        if sources:
            lookup[canonical] = list(dict.fromkeys(sources))
    return lookup


def normalize_crypto_wizards_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    raw = pd.DataFrame(records)
    lookup = _alias_lookup(raw.columns)
    normalized = pd.DataFrame(index=raw.index)
    for canonical, sources in lookup.items():
        normalized[canonical] = raw[sources].bfill(axis=1).iloc[:, 0]

    if "pair" not in normalized and {"asset_x", "asset_y"}.issubset(normalized.columns):
        normalized["pair"] = normalized["asset_x"].astype(str) + "-" + normalized["asset_y"].astype(str)
    if "pair" in normalized:
        normalized["pair"] = normalized["pair"].astype(str).str.replace("/", "-", regex=False).str.upper()

    numeric_columns = [column for column in normalized.columns if column not in {"timestamp", "pair", "asset_x", "asset_y", "copula", "regime"}]
    for column in numeric_columns:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    if "conditional_probability_distortion" not in normalized and {"u1_given_u2", "u2_given_u1"}.issubset(normalized.columns):
        normalized["conditional_probability_distortion"] = normalized["u1_given_u2"] - normalized["u2_given_u1"]
    if "zscore" not in normalized and "rolling_zscore" in normalized:
        normalized["zscore"] = normalized["rolling_zscore"]
    if "zscore" not in normalized and "spread" in normalized:
        rolling_mean = normalized["spread"].rolling(80, min_periods=20).mean()
        rolling_std = normalized["spread"].rolling(80, min_periods=20).std()
        normalized["zscore"] = (normalized["spread"] - rolling_mean) / rolling_std
    if "regime" not in normalized:
        normalized["regime"] = "unknown"
    if "timestamp" in normalized:
        normalized = normalized.sort_values("timestamp")
    normalized = add_derived_beta_from_prices(normalized)
    return normalized.dropna(subset=["pair", "spread"], how="any") if {"pair", "spread"}.issubset(normalized.columns) else normalized


def datasets_from_fixtures(input_dir: str | Path) -> list[PairDataset]:
    payloads = load_fixture_payloads(input_dir)
    all_records: list[dict[str, Any]] = []
    for fixture in payloads:
        for record in _extract_records(fixture.payload):
            record["_fixture_endpoint"] = fixture.endpoint
            record["_fixture_path"] = str(fixture.path)
            all_records.append(record)
    normalized = normalize_crypto_wizards_records(all_records)
    if normalized.empty or "pair" not in normalized.columns:
        return []
    datasets = []
    for pair, frame in normalized.groupby("pair", sort=True):
        clean = frame.drop(columns=[column for column in ("pair",) if column in frame.columns]).reset_index(drop=True)
        datasets.append(PairDataset(pair=str(pair), frame=clean))
    return datasets


def discovered_field_rows(input_dir: str | Path) -> list[dict[str, str | float]]:
    rows: list[dict[str, str | float]] = []
    for fixture in load_fixture_payloads(input_dir):
        for record in _extract_records(fixture.payload):
            for field, value in record.items():
                leaf = snake_case(field.split(".")[-1])
                canonical = next(
                    (name for name, aliases in CANONICAL_ALIASES.items() if leaf in {snake_case(alias) for alias in aliases}),
                    leaf,
                )
                rows.append(
                    {
                        "name": canonical,
                        "description": "",
                        "type": type(value).__name__,
                        "example_value": str(value)[:160],
                        "endpoint": fixture.endpoint,
                        "importance_score": 90.0 if canonical in IMPORTANT_FIELDS else 50.0,
                        "notes": f"fixture:{fixture.path.name}; raw_field:{field}",
                    }
                )
    if not rows:
        return []
    frame = pd.DataFrame(rows).drop_duplicates(subset=["name", "endpoint", "notes"])
    return frame.sort_values(["importance_score", "name"], ascending=[False, True]).to_dict("records")


def write_fixture_field_dictionary(input_dir: str | Path, output_path: str | Path) -> Path:
    rows = discovered_field_rows(input_dir)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["name", "description", "type", "example_value", "endpoint", "importance_score", "notes"]).to_csv(
        output, index=False
    )
    return output
from functools import lru_cache
