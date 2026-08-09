"""Research-only local approximations for Crypto Wizards exact signal modes.

The dashboard's mode labels are part of the hypothesis, not a licence to
recreate unknown vendor calculations.  This module therefore only computes a
local signal when the displayed settings and point-in-time inputs are present.
Every successful result remains a ``local_formula_approximation``; a bounded
vendor custom-series proof is the only path that can establish vendor parity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from quant_platform.statistics.math_v2 import rolling_gaussian_copula_conditionals


CANONICAL_WIZARD_MODES = (
    "Static (Spread)",
    "Static (ZScoreR)",
    "Dyn (Spread)",
    "Dyn (ZScoreR)",
    "OU (Spread)",
    "OU (ZScoreR)",
    "Copula",
)


MODE_REQUIRED_SETTINGS: dict[str, tuple[str, ...]] = {
    "Static (Spread)": ("hedge_ratio",),
    "Static (ZScoreR)": ("hedge_ratio", "zscore_window"),
    "Dyn (Spread)": ("dynamic_hedge_ratio_method", "dynamic_hedge_ratio_window"),
    "Dyn (ZScoreR)": ("dynamic_hedge_ratio_method", "dynamic_hedge_ratio_window", "zscore_window"),
    "OU (Spread)": ("hedge_ratio", "ou_mu", "ou_sigma"),
    "OU (ZScoreR)": ("hedge_ratio", "ou_mu", "ou_sigma", "zscore_window"),
    "Copula": (
        "copula_family",
        "copula_signal_type",
        "copula_direction_view",
        "copula_entry_lower",
        "copula_entry_upper",
        "copula_exit_lower",
        "copula_exit_upper",
    ),
}

COMMON_REQUIRED_SETTINGS = (
    "capture_confirmed",
    "entry_long_position",
    "entry_short_position",
)

THRESHOLD_REQUIRED_SETTINGS = (
    "entry_long_operator",
    "entry_long_value",
    "entry_short_operator",
    "entry_short_value",
    "exit_long_operator",
    "exit_long_value",
    "exit_short_operator",
    "exit_short_value",
)

SUPPORTED_DYNAMIC_METHODS = {
    "history_captured_hedge_ratio",
    "rolling_ols_log_prices",
}

POSITION_VALUES = {
    # The shared two-leg backtester maps a negative signal to long X / short Y.
    "long_x_short_y": -1.0,
    "short_x_long_y": 1.0,
}

_OPERATOR_ALIASES = {
    ">": ">",
    ">=": ">=",
    "gt": ">",
    "gte": ">=",
    "greater_than": ">",
    "greater_than_or_equal": ">=",
    "greater_or_equal": ">=",
    "<": "<",
    "<=": "<=",
    "lt": "<",
    "lte": "<=",
    "less_than": "<",
    "less_than_or_equal": "<=",
    "less_or_equal": "<=",
    "=": "==",
    "==": "==",
    "eq": "==",
    "equal": "==",
}


@dataclass(frozen=True)
class WizardModeReplayResult:
    """The result of one local, non-vendor-equivalent mode replay."""

    exact_mode: str
    signal: pd.Series
    trades: list[dict[str, object]]
    metric: pd.Series
    metric_name: str
    mode_replay_status: str
    mode_fidelity_status: str
    mode_fidelity_reason: str
    missing_inputs: tuple[str, ...]
    computation_notes: tuple[str, ...]
    acceptance_eligible: bool = False


def normalize_exact_mode(value: object) -> str:
    """Return the canonical Wizard mode label or an empty string."""

    normalized = "".join(character for character in str(value or "").lower() if character.isalnum())
    aliases = {
        "staticspread": "Static (Spread)",
        "static": "Static (Spread)",
        "staticzscorer": "Static (ZScoreR)",
        "staticzscore": "Static (ZScoreR)",
        "dynspread": "Dyn (Spread)",
        "dynamicspread": "Dyn (Spread)",
        "dynzscorer": "Dyn (ZScoreR)",
        "dynzscore": "Dyn (ZScoreR)",
        "dynamiczscorer": "Dyn (ZScoreR)",
        "dynamiczscore": "Dyn (ZScoreR)",
        "ouspread": "OU (Spread)",
        "ou": "OU (Spread)",
        "ouzscorer": "OU (ZScoreR)",
        "ouzscore": "OU (ZScoreR)",
        "copula": "Copula",
    }
    return aliases.get(normalized, "")


def mode_requirements(exact_mode: object) -> tuple[str, ...]:
    """Return the settings that a local approximation needs for this mode."""

    mode = normalize_exact_mode(exact_mode)
    if not mode:
        return COMMON_REQUIRED_SETTINGS
    threshold_requirements = THRESHOLD_REQUIRED_SETTINGS if mode != "Copula" else ()
    return (*COMMON_REQUIRED_SETTINGS, *threshold_requirements, *MODE_REQUIRED_SETTINGS[mode])


def build_local_mode_signal(
    history: pd.DataFrame,
    settings: Mapping[str, object],
    *,
    exact_mode: object | None = None,
) -> WizardModeReplayResult:
    """Build a local signal from captured settings and entry-time history.

    This intentionally does *not* attempt to mirror unknown Wizard internals.
    It makes the approximation and all missing prerequisites explicit so the
    research lane can compare each named mode fairly without claiming parity.
    """

    mode = normalize_exact_mode(exact_mode or settings.get("exact_mode", ""))
    index = history.index
    empty_metric = pd.Series(np.nan, index=index, dtype="float64")
    if not mode:
        return _blocked_result(
            exact_mode=str(exact_mode or settings.get("exact_mode", "")),
            index=index,
            metric=empty_metric,
            metric_name="",
            missing=("supported_exact_mode",),
            notes=("The captured Wizard mode is unsupported by the local research engine.",),
        )

    missing = _missing_setting_inputs(settings, mode)
    if missing:
        return _blocked_result(
            exact_mode=mode,
            index=index,
            metric=empty_metric,
            metric_name="",
            missing=missing,
            notes=("The local replay refuses to fill in unobserved dashboard settings.",),
        )

    try:
        metric, metric_name, notes = _metric_for_mode(history, settings, mode)
    except ValueError as exc:
        return _blocked_result(
            exact_mode=mode,
            index=index,
            metric=empty_metric,
            metric_name="",
            missing=(str(exc),),
            notes=("The required point-in-time history is not available for this mode.",),
        )

    if mode == "Copula":
        signal, trades = _copula_threshold_signal(metric, settings)
    else:
        signal, trades = _threshold_signal(
            metric, settings, exit_reason=f"{_mode_slug(mode)}_captured_exit"
        )

    return WizardModeReplayResult(
        exact_mode=mode,
        signal=signal,
        trades=trades,
        metric=metric,
        metric_name=metric_name,
        mode_replay_status="READY_FOR_RESEARCH_REPLAY",
        mode_fidelity_status="local_formula_approximation",
        mode_fidelity_reason="local_formula_is_not_a_vendor_custom_series_replay",
        missing_inputs=(),
        computation_notes=notes,
        acceptance_eligible=False,
    )


def _missing_setting_inputs(settings: Mapping[str, object], mode: str) -> tuple[str, ...]:
    missing: list[str] = []
    for field in mode_requirements(mode):
        if field == "capture_confirmed":
            if not _as_bool(settings.get(field)):
                missing.append(field)
        elif _is_blank(settings.get(field)):
            missing.append(field)

    if mode != "Copula":
        for field in (
            "entry_long_operator",
            "entry_short_operator",
            "exit_long_operator",
            "exit_short_operator",
        ):
            if field not in missing and _operator(settings.get(field)) is None:
                missing.append(f"invalid_{field}")
        for field in (
            "entry_long_value",
            "entry_short_value",
            "exit_long_value",
            "exit_short_value",
        ):
            if field not in missing and _number(settings.get(field)) is None:
                missing.append(f"invalid_{field}")
    for field in ("entry_long_position", "entry_short_position"):
        if field not in missing and _position(settings.get(field)) is None:
            missing.append(f"invalid_{field}")

    if mode.startswith("Dyn"):
        method = _dynamic_method(settings.get("dynamic_hedge_ratio_method"))
        if method not in SUPPORTED_DYNAMIC_METHODS:
            missing.append("unsupported_dynamic_hedge_ratio_method")
        if _positive_int(settings.get("dynamic_hedge_ratio_window")) is None:
            missing.append("invalid_dynamic_hedge_ratio_window")
    if mode.startswith("OU"):
        if _number(settings.get("ou_mu")) is None:
            missing.append("invalid_ou_mu")
        sigma = _number(settings.get("ou_sigma"))
        if sigma is None or sigma <= 0:
            missing.append("invalid_ou_sigma")
    if mode.endswith("ZScoreR") and _positive_int(settings.get("zscore_window")) is None:
        missing.append("invalid_zscore_window")
    if mode == "Copula":
        if _copula_view(settings.get("copula_direction_view")) is None:
            missing.append("invalid_copula_direction_view")
        lower = _number(settings.get("copula_entry_lower"))
        upper = _number(settings.get("copula_entry_upper"))
        if lower is None or upper is None or not lower < upper:
            missing.append("invalid_copula_entry_thresholds")
        exit_lower = _number(settings.get("copula_exit_lower"))
        exit_upper = _number(settings.get("copula_exit_upper"))
        if exit_lower is None or exit_upper is None or not exit_lower <= exit_upper:
            missing.append("invalid_copula_exit_thresholds")
    return tuple(dict.fromkeys(missing))


def _metric_for_mode(
    history: pd.DataFrame,
    settings: Mapping[str, object],
    mode: str,
) -> tuple[pd.Series, str, tuple[str, ...]]:
    if mode == "Copula":
        view = _copula_view(settings.get("copula_direction_view"))
        assert view is not None
        column = _first_history_column(history, _copula_column_aliases(view))
        if column is not None:
            metric = pd.to_numeric(history[column], errors="coerce")
            source_note = f"uses the captured point-in-time {view} conditional series directly"
        else:
            prices = _two_leg_prices(history)
            window = _positive_int(settings.get("copula_window")) or 120
            conditionals = rolling_gaussian_copula_conditionals(
                prices["price_x"].pct_change(),
                prices["price_y"].pct_change(),
                window=window,
                min_rows=min(60, window),
            )
            metric = conditionals[view]
            source_note = (
                f"reconstructs {view} with a causal trailing Gaussian copula window={window}; "
                "this is a local approximation and may differ from the captured family"
            )
        return (
            metric,
            view,
            (
                source_note,
                "no full-sample conditional series is used",
                f"captured copula family: {str(settings.get('copula_family')).strip()}",
                f"captured copula signal type: {str(settings.get('copula_signal_type')).strip()}",
            ),
        )

    if mode.startswith("Dyn"):
        spread, dynamic_note = _dynamic_spread(history, settings)
        if mode == "Dyn (Spread)":
            min_periods = _positive_int(settings.get("zscore_window")) or 20
            return (
                _expanding_zscore(spread, min_periods=min_periods),
                "dynamic_spread_expanding_zscore",
                (
                    dynamic_note,
                    f"Spread thresholds are sigma-scaled; the local causal approximation uses expanding sample statistics after {min_periods} observations",
                ),
            )
        window = _positive_int(settings.get("zscore_window"))
        assert window is not None
        return (
            _rolling_zscore(spread, window),
            "dynamic_spread_rolling_zscore",
            (
                dynamic_note,
                f"rolling ZScoreR uses captured window={window} and only data available at each candle close",
            ),
        )

    spread = _static_spread(history, settings)
    if mode == "Static (Spread)":
        min_periods = _positive_int(settings.get("zscore_window")) or 20
        return (
            _expanding_zscore(spread, min_periods=min_periods),
            "static_spread_expanding_zscore",
            (
                "captured Wizard hedge ratio is applied in y-on-x orientation to a local log spread",
                f"Spread thresholds are sigma-scaled; the local causal approximation uses expanding sample statistics after {min_periods} observations",
                "Wizard custom-series history uses a full-sample sample-standard-deviation zscore; that hindsight formula is not used for live-safe local signals",
            ),
        )
    if mode == "Static (ZScoreR)":
        window = _positive_int(settings.get("zscore_window"))
        assert window is not None
        return (
            _rolling_zscore(spread, window),
            "static_spread_rolling_zscore",
            (
                f"rolling ZScoreR uses captured window={window} and only data available at each candle close",
            ),
        )

    mu = _number(settings.get("ou_mu"))
    sigma = _number(settings.get("ou_sigma"))
    assert mu is not None and sigma is not None
    residual = spread - mu
    if mode == "OU (Spread)":
        return (
            residual / sigma,
            "ou_sigma_scaled_spread",
            (
                "OU residual is centered by captured mu and scaled by captured sigma; it is not a re-fit full-sample OU process",
                f"captured OU sigma={sigma:g} supplies the threshold unit",
            ),
        )
    window = _positive_int(settings.get("zscore_window"))
    assert window is not None
    return (
        _rolling_zscore(residual, window),
        "ou_residual_rolling_zscore",
        (
            "OU residual uses captured mu and sigma; no full-sample OU parameter fit is performed",
            f"rolling ZScoreR uses captured window={window} and only data available at each candle close",
        ),
    )


def _static_spread(history: pd.DataFrame, settings: Mapping[str, object]) -> pd.Series:
    prices = _two_leg_prices(history)
    hedge_ratio = _number(settings.get("hedge_ratio"))
    if hedge_ratio is None:
        raise ValueError("missing_static_hedge_ratio")
    return np.log(prices["price_y"]) - hedge_ratio * np.log(prices["price_x"])


def _dynamic_spread(history: pd.DataFrame, settings: Mapping[str, object]) -> tuple[pd.Series, str]:
    prices = _two_leg_prices(history)
    method = _dynamic_method(settings.get("dynamic_hedge_ratio_method"))
    window = _positive_int(settings.get("dynamic_hedge_ratio_window"))
    assert window is not None
    log_x = np.log(prices["price_x"])
    log_y = np.log(prices["price_y"])
    if method == "history_captured_hedge_ratio":
        column = _first_history_column(history, ("hedge_ratio", "dynamic_hedge_ratio", "beta"))
        if column is None:
            raise ValueError("missing_historical_dynamic_hedge_ratio")
        ratio = pd.to_numeric(history[column], errors="coerce")
        return (
            log_x - ratio * log_y,
            "uses the point-in-time hedge-ratio series captured in history",
        )
    if method == "rolling_ols_log_prices":
        var_y = log_y.rolling(window, min_periods=window).var(ddof=0)
        cov_yx = log_y.rolling(window, min_periods=window).cov(log_x, ddof=0)
        ratio = cov_yx.div(var_y.where(var_y.abs() > 1e-12))
        return (
            log_x - ratio * log_y,
            f"uses a point-in-time rolling OLS log-price hedge ratio with captured window={window}",
        )
    raise ValueError("unsupported_dynamic_hedge_ratio_method")


def _two_leg_prices(history: pd.DataFrame) -> pd.DataFrame:
    columns = {"price_x", "price_y"}
    missing = sorted(columns - set(history.columns))
    if missing:
        raise ValueError(f"missing_history_columns:{','.join(missing)}")
    prices = history[["price_x", "price_y"]].apply(pd.to_numeric, errors="coerce")
    if not ((prices["price_x"] > 0) & (prices["price_y"] > 0)).any():
        raise ValueError("history_has_no_positive_two_leg_prices")
    return prices.where((prices["price_x"] > 0) & (prices["price_y"] > 0))


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std(ddof=0)
    return series.sub(mean).div(std.where(std.abs() > 1e-12))


def _expanding_zscore(series: pd.Series, *, min_periods: int) -> pd.Series:
    mean = series.expanding(min_periods=min_periods).mean()
    std = series.expanding(min_periods=min_periods).std(ddof=1)
    return series.sub(mean).div(std.where(std.abs() > 1e-12))


def _threshold_signal(
    metric: pd.Series,
    settings: Mapping[str, object],
    *,
    exit_reason: str,
) -> tuple[pd.Series, list[dict[str, object]]]:
    entry_long_operator = _operator(settings.get("entry_long_operator"))
    entry_short_operator = _operator(settings.get("entry_short_operator"))
    exit_long_operator = _operator(settings.get("exit_long_operator"))
    exit_short_operator = _operator(settings.get("exit_short_operator"))
    assert all([entry_long_operator, entry_short_operator, exit_long_operator, exit_short_operator])
    long_position = _position(settings.get("entry_long_position"))
    short_position = _position(settings.get("entry_short_position"))
    assert long_position is not None and short_position is not None
    return _stateful_signal(
        metric,
        lower_entry=(
            entry_long_operator,
            _number(settings.get("entry_long_value")),
            long_position,
            "entry_long",
        ),
        upper_entry=(
            entry_short_operator,
            _number(settings.get("entry_short_value")),
            short_position,
            "entry_short",
        ),
        lower_exit=(exit_long_operator, _number(settings.get("exit_long_value"))),
        upper_exit=(exit_short_operator, _number(settings.get("exit_short_value"))),
        exit_reason=exit_reason,
    )


def _copula_threshold_signal(
    metric: pd.Series, settings: Mapping[str, object]
) -> tuple[pd.Series, list[dict[str, object]]]:
    long_position = _position(settings.get("entry_long_position"))
    short_position = _position(settings.get("entry_short_position"))
    assert long_position is not None and short_position is not None
    return _stateful_signal(
        metric,
        lower_entry=(
            "<=",
            _number(settings.get("copula_entry_lower")),
            long_position,
            "copula_lower_tail",
        ),
        upper_entry=(
            ">=",
            _number(settings.get("copula_entry_upper")),
            short_position,
            "copula_upper_tail",
        ),
        lower_exit=(">=", _number(settings.get("copula_exit_lower"))),
        upper_exit=("<=", _number(settings.get("copula_exit_upper"))),
        exit_reason="copula_captured_exit",
    )


def _stateful_signal(
    metric: pd.Series,
    *,
    lower_entry: tuple[str, float | None, float, str],
    upper_entry: tuple[str, float | None, float, str],
    lower_exit: tuple[str, float | None],
    upper_exit: tuple[str, float | None],
    exit_reason: str,
) -> tuple[pd.Series, list[dict[str, object]]]:
    """Apply captured entry and exit rules without looking beyond each bar."""

    signal: list[float] = []
    trades: list[dict[str, object]] = []
    state = 0.0
    current: dict[str, object] | None = None
    for index, (timestamp, raw_value) in enumerate(metric.items()):
        value = float(raw_value) if pd.notna(raw_value) else np.nan
        lower_match = pd.notna(value) and _compare(value, lower_entry[0], lower_entry[1])
        upper_match = pd.notna(value) and _compare(value, upper_entry[0], upper_entry[1])
        if state == 0.0 and lower_match and not upper_match:
            state = lower_entry[2]
            current = _open_trade(trades, timestamp, value, index, lower_entry[3], state)
        elif state == 0.0 and upper_match and not lower_match:
            state = upper_entry[2]
            current = _open_trade(trades, timestamp, value, index, upper_entry[3], state)
        elif state != 0.0 and current is not None:
            entry_rule = str(current["entry_rule"])
            exit_rule = (
                lower_exit if entry_rule in {"entry_long", "copula_lower_tail"} else upper_exit
            )
            if pd.notna(value) and _compare(value, exit_rule[0], exit_rule[1]):
                current.update(
                    {
                        "exit_timestamp": timestamp,
                        "exit_signal_value": value,
                        "end_i": index,
                        "bars_held": index - int(current["start_i"]) + 1,
                        "exit_reason": exit_reason,
                    }
                )
                trades.append(current)
                current = None
                state = 0.0
        signal.append(state)
    if current is not None:
        current.update(
            {
                "exit_timestamp": "",
                "exit_signal_value": "",
                "end_i": len(signal) - 1,
                "bars_held": len(signal) - int(current["start_i"]),
                "exit_reason": "open_at_end_of_history",
            }
        )
        trades.append(current)
    return pd.Series(signal, index=metric.index, dtype="float64"), trades


def _open_trade(
    trades: list[dict[str, object]],
    timestamp: object,
    value: float,
    index: int,
    entry_rule: str,
    position: float,
) -> dict[str, object]:
    return {
        "trade_id": len(trades) + 1,
        "entry_timestamp": timestamp,
        "entry_signal_value": value,
        "entry_rule": entry_rule,
        "direction": "long_x_short_y" if position < 0 else "short_x_long_y",
        "start_i": index,
    }


def _blocked_result(
    *,
    exact_mode: str,
    index: pd.Index,
    metric: pd.Series,
    metric_name: str,
    missing: tuple[str, ...],
    notes: tuple[str, ...],
) -> WizardModeReplayResult:
    return WizardModeReplayResult(
        exact_mode=exact_mode,
        signal=pd.Series(0.0, index=index, dtype="float64"),
        trades=[],
        metric=metric,
        metric_name=metric_name,
        mode_replay_status="BLOCKED_MODE_INPUTS",
        mode_fidelity_status="local_formula_approximation",
        mode_fidelity_reason="local_mode_inputs_are_incomplete_or_unsupported",
        missing_inputs=tuple(dict.fromkeys(missing)),
        computation_notes=notes,
        acceptance_eligible=False,
    )


def _compare(value: float, operator: str, threshold: float | None) -> bool:
    if threshold is None:
        return False
    if operator == ">":
        return value > threshold
    if operator == ">=":
        return value >= threshold
    if operator == "<":
        return value < threshold
    if operator == "<=":
        return value <= threshold
    if operator == "==":
        return value == threshold
    return False


def _operator(value: object) -> str | None:
    raw = str(value or "").strip().lower().replace(" ", "_")
    return _OPERATOR_ALIASES.get(raw)


def _position(value: object) -> float | None:
    raw = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return POSITION_VALUES.get(raw)


def _dynamic_method(value: object) -> str:
    raw = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "history": "history_captured_hedge_ratio",
        "captured_history": "history_captured_hedge_ratio",
        "captured_history_hedge_ratio": "history_captured_hedge_ratio",
        "rolling_ols": "rolling_ols_log_prices",
        "rolling_ols_log_price": "rolling_ols_log_prices",
    }
    return aliases.get(raw, raw)


def _copula_view(value: object) -> str | None:
    raw = "".join(character for character in str(value or "").lower() if character.isalnum())
    aliases = {
        "u1givenu2": "u1_given_u2",
        "xgiveny": "u1_given_u2",
        "conditionalu1u2": "u1_given_u2",
        "u2givenu1": "u2_given_u1",
        "ygivenx": "u2_given_u1",
        "conditionalu2u1": "u2_given_u1",
    }
    return aliases.get(raw)


def _copula_column_aliases(view: str) -> tuple[str, ...]:
    if view == "u1_given_u2":
        return ("u1_given_u2", "copula_u1_given_u2", "conditional_u1_given_u2", "x_given_y")
    return ("u2_given_u1", "copula_u2_given_u1", "conditional_u2_given_u1", "y_given_x")


def _first_history_column(history: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    normalized = {
        "".join(character for character in column.lower() if character.isalnum()): column
        for column in history.columns
    }
    for candidate in candidates:
        if candidate in history.columns:
            return candidate
        match = normalized.get(
            "".join(character for character in candidate.lower() if character.isalnum())
        )
        if match:
            return match
    return None


def _number(value: object) -> float | None:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "null"}:
            return None
        return float(text)
    except (TypeError, ValueError):
        return None


def _positive_int(value: object) -> int | None:
    number = _number(value)
    if number is None or number <= 1 or not number.is_integer():
        return None
    return int(number)


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _is_blank(value: object) -> bool:
    return (
        value is None
        or (isinstance(value, float) and np.isnan(value))
        or not str(value).strip()
        or str(value).strip().lower()
        in {
            "nan",
            "none",
            "null",
        }
    )


def _mode_slug(mode: str) -> str:
    return "".join(character.lower() if character.isalnum() else "_" for character in mode).strip(
        "_"
    )
