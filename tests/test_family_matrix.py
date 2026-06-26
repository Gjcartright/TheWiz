import numpy as np
import pandas as pd

from quant_platform.backtest import CostModel
from quant_platform.experiments import AcceptanceGate, ExperimentConfig, PairDataset
from quant_platform.family_matrix import run_family_matrix, strategy_family_registry
from quant_platform.strategies import StrategySpec


def _frame(n: int = 160) -> pd.DataFrame:
    x = np.linspace(0, 12 * np.pi, n)
    spread = np.sin(x)
    frame = pd.DataFrame({"spread": spread})
    frame["zscore"] = frame["spread"] * 2.5
    frame["conditional_probability_distortion"] = np.tanh(frame["zscore"] / 3.0)
    frame["regime"] = np.where(frame.index < n / 2, "bull", "range")
    frame["price_x"] = 100 + np.linspace(0, 5, n) + frame["spread"]
    frame["price_y"] = 50 + np.linspace(0, 2, n) - frame["spread"]
    frame["hedge_ratio"] = 1.2
    frame["beta"] = 0.8
    frame["funding_x_bps"] = 2.0
    frame["funding_y_bps"] = 3.0
    return frame


def _stateful_signal(frame: pd.DataFrame, threshold: float) -> pd.Series:
    z = frame["zscore"].astype(float)
    signal = pd.Series(0.0, index=frame.index)
    signal[z > threshold] = -1.0
    signal[z < -threshold] = 1.0
    signal[z.abs() < 0.25] = 0.0
    return signal.replace(0.0, np.nan).ffill().fillna(0.0)


def _strategies() -> tuple[StrategySpec, ...]:
    return (
        StrategySpec(9001, "Alpha One", "alpha", "alpha family first strategy", ("zscore",), (), lambda f: _stateful_signal(f, 1.5)),
        StrategySpec(9002, "Alpha Two", "alpha", "alpha family second strategy", ("zscore",), (), lambda f: _stateful_signal(f, 1.8)),
        StrategySpec(9101, "Beta One", "beta", "beta family first strategy", ("zscore",), (), lambda f: _stateful_signal(f, 1.4)),
        StrategySpec(9201, "Gamma One", "gamma", "gamma family first strategy", ("zscore",), (), lambda f: _stateful_signal(f, 1.6)),
        StrategySpec(9301, "Delta One", "delta", "delta family first strategy", ("zscore",), (), lambda f: _stateful_signal(f, 1.55)),
    )


def test_strategy_family_registry_groups_custom_strategies():
    frame = strategy_family_registry(_strategies())
    assert list(frame["family"].unique()) == ["alpha", "beta", "delta", "gamma"]
    assert len(frame) == 5


def test_run_family_matrix_writes_separate_and_combo_outputs(tmp_path):
    datasets = [PairDataset("BTC-USD-SOL-USD", _frame()), PairDataset("DOGE-USD-ETH-USD", _frame())]
    paths = run_family_matrix(
        datasets,
        output_dir=tmp_path,
        strategies=_strategies(),
        gate=AcceptanceGate(min_profit_factor=0.1, min_sharpe=-99, max_drawdown=1.0, min_trades=1, required_regime="ALL"),
    )

    assert set(paths) == {
        "registry",
        "separate_summary",
        "best_strategies",
        "combo_summary",
        "combo_pair_summary",
        "combo_detail",
        "runbook",
    }
    separate = pd.read_csv(paths["separate_summary"])
    combos = pd.read_csv(paths["combo_summary"])
    registry = pd.read_csv(paths["registry"])
    assert len(separate) == 4
    assert {"family", "best_strategy_name", "family_output_dir"}.issubset(separate.columns)
    assert len(registry) == 5
    assert not combos.empty
    assert {"combo_name", "combo_size", "families", "strategies"}.issubset(combos.columns)
    assert {2, 3, 4}.issubset(set(combos["combo_size"].astype(int)))
    assert (tmp_path / "families" / "alpha" / "experiment_results.csv").exists()
    assert (tmp_path / "families" / "beta" / "family_acceptance_report.csv").exists()
