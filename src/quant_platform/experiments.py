from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite, isinf
from pathlib import Path
from typing import Iterable

import pandas as pd

from quant_platform.ablations import write_ablation_report
from quant_platform.backtest import BacktestResult, CostModel, backtest_pair, backtest_two_leg_spread
from quant_platform.feature_engine import FeatureEngine
from quant_platform.regimes import regime_pair_strategy_report
from quant_platform.strategies import STRATEGIES, STRATEGY_REQUIRED_COLUMNS, StrategySpec


@dataclass(frozen=True)
class PairDataset:
    pair: str
    frame: pd.DataFrame


@dataclass(frozen=True)
class AcceptanceGate:
    min_profit_factor: float = 1.8
    preferred_profit_factor: float = 2.0
    min_sharpe: float = 1.2
    preferred_sharpe: float = 1.5
    max_drawdown: float = 0.15
    preferred_max_drawdown: float = 0.10
    min_trades: int = 100
    preferred_trades: int = 250
    min_pairs: int = 2
    required_cost_buckets: tuple[str, ...] = ("base", "stress")
    required_regime: str = "ALL"
    require_positive_expectancy: bool = True
    require_two_leg_backtests: bool = True
    require_two_leg_execution_inputs: bool = True

    def evaluate(self, result: BacktestResult) -> tuple[bool, str]:
        failures: list[str] = []
        if result.trades < self.min_trades:
            failures.append(f"trades<{self.min_trades}")
        if not isfinite(result.profit_factor) and not isinf(result.profit_factor):
            failures.append("profit_factor_invalid")
        elif result.profit_factor < self.min_profit_factor:
            failures.append(f"profit_factor<{self.min_profit_factor}")
        if result.sharpe < self.min_sharpe:
            failures.append(f"sharpe<{self.min_sharpe}")
        if result.max_drawdown > self.max_drawdown:
            failures.append(f"max_drawdown>{self.max_drawdown}")
        if self.require_positive_expectancy and result.expectancy <= 0:
            failures.append("expectancy<=0")
        if failures:
            return False, ";".join(failures)
        return True, "passed"

    def evaluate_strategy(self, rows: pd.DataFrame) -> dict[str, object]:
        evaluated = rows[rows["status"] == "evaluated"].copy()
        deployable_scope = evaluated[
            (evaluated["regime"] == self.required_regime)
            & (evaluated["cost_bucket"].isin(self.required_cost_buckets))
        ]
        if "backtest_mode" not in deployable_scope.columns:
            deployable_scope = deployable_scope.assign(backtest_mode="unknown")
        pairs_tested = int(deployable_scope["pair"].nunique()) if not deployable_scope.empty else 0
        passing_scope = deployable_scope[deployable_scope["eligible"]]
        if self.require_two_leg_backtests:
            passing_scope = passing_scope[passing_scope["backtest_mode"] == "two_leg"]
            if self.require_two_leg_execution_inputs:
                passing_scope = _complete_two_leg_execution_input_scope(passing_scope)
        passing_pairs = 0
        for _, pair_rows in passing_scope.groupby("pair"):
            if set(self.required_cost_buckets).issubset(set(pair_rows["cost_bucket"])):
                passing_pairs += 1

        failures: list[str] = []
        if evaluated.empty:
            failures.append("no_evaluated_runs")
        if pairs_tested < self.min_pairs:
            failures.append(f"pairs_tested<{self.min_pairs}")
        if passing_pairs < self.min_pairs:
            failures.append(f"passing_pairs<{self.min_pairs}")
        if self.require_two_leg_backtests:
            two_leg_scope = deployable_scope[deployable_scope["backtest_mode"] == "two_leg"]
            two_leg_pairs = int(two_leg_scope["pair"].nunique()) if not two_leg_scope.empty else 0
            complete_two_leg_scope = _complete_two_leg_execution_input_scope(two_leg_scope)
            two_leg_execution_input_pairs = int(complete_two_leg_scope["pair"].nunique()) if not complete_two_leg_scope.empty else 0
            two_leg_passing_scope = complete_two_leg_scope[complete_two_leg_scope["eligible"]]
            two_leg_passing_pairs = 0
            for _, pair_rows in two_leg_passing_scope.groupby("pair"):
                if set(self.required_cost_buckets).issubset(set(pair_rows["cost_bucket"])):
                    two_leg_passing_pairs += 1
            if two_leg_pairs < self.min_pairs:
                failures.append(f"two_leg_pairs<{self.min_pairs}")
            if self.require_two_leg_execution_inputs and two_leg_execution_input_pairs < self.min_pairs:
                failures.append(f"two_leg_execution_input_pairs<{self.min_pairs}")
                missing_inputs = _pairs_missing_two_leg_execution_inputs(two_leg_scope)
                if missing_inputs:
                    failures.append(f"two_leg_missing_inputs:{','.join(missing_inputs)}")
            missing_two_leg_cost_pairs = _pairs_missing_cost_buckets(two_leg_scope, self.required_cost_buckets)
            if missing_two_leg_cost_pairs:
                failures.append(f"two_leg_missing_cost_buckets:{','.join(missing_two_leg_cost_pairs)}")
        else:
            two_leg_scope = deployable_scope[deployable_scope["backtest_mode"] == "two_leg"]
            two_leg_pairs = int(two_leg_scope["pair"].nunique()) if not two_leg_scope.empty else 0
            two_leg_passing_scope = two_leg_scope[two_leg_scope["eligible"]]
            two_leg_passing_pairs = int(two_leg_passing_scope["pair"].nunique()) if not two_leg_passing_scope.empty else 0
            two_leg_execution_input_pairs = int(_complete_two_leg_execution_input_scope(two_leg_scope)["pair"].nunique()) if not two_leg_scope.empty else 0

        missing_cost_buckets = sorted(set(self.required_cost_buckets).difference(set(deployable_scope["cost_bucket"])))
        if missing_cost_buckets:
            failures.append(f"missing_cost_buckets:{','.join(missing_cost_buckets)}")

        production_eligible = not failures
        median_profit_factor = float(deployable_scope["profit_factor"].median()) if not deployable_scope.empty else 0.0
        median_sharpe = float(deployable_scope["sharpe"].median()) if not deployable_scope.empty else 0.0
        worst_drawdown = float(deployable_scope["max_drawdown"].max()) if not deployable_scope.empty else 0.0
        total_trades = int(deployable_scope["trades"].sum()) if not deployable_scope.empty else 0

        preferred_failures: list[str] = []
        if not production_eligible:
            preferred_failures.append("not_production_eligible")
        if median_profit_factor < self.preferred_profit_factor:
            preferred_failures.append(f"median_profit_factor<{self.preferred_profit_factor}")
        if median_sharpe < self.preferred_sharpe:
            preferred_failures.append(f"median_sharpe<{self.preferred_sharpe}")
        if worst_drawdown > self.preferred_max_drawdown:
            preferred_failures.append(f"worst_drawdown>{self.preferred_max_drawdown}")
        if total_trades < self.preferred_trades:
            preferred_failures.append(f"total_trades<{self.preferred_trades}")

        return {
            "production_eligible": production_eligible,
            "preferred_eligible": not preferred_failures,
            "acceptance_reason": "passed" if production_eligible else ";".join(failures),
            "preferred_reason": "passed" if not preferred_failures else ";".join(preferred_failures),
            "evaluated_runs": int(len(evaluated)),
            "passing_runs": int(evaluated["eligible"].sum()) if not evaluated.empty else 0,
            "pairs_tested": pairs_tested,
            "passing_pairs": passing_pairs,
            "two_leg_pairs_tested": two_leg_pairs,
            "two_leg_execution_input_pairs": two_leg_execution_input_pairs,
            "two_leg_passing_pairs": two_leg_passing_pairs,
            "required_cost_buckets": ";".join(self.required_cost_buckets),
            "required_backtest_mode": "two_leg" if self.require_two_leg_backtests else "any",
            "required_two_leg_inputs": ";".join(TWO_LEG_EXECUTION_INPUT_FIELDS)
            if self.require_two_leg_execution_inputs
            else "price_x;price_y",
            "total_trades": total_trades,
            "median_profit_factor": median_profit_factor,
            "median_sharpe": median_sharpe,
            "worst_drawdown": worst_drawdown,
        }


@dataclass(frozen=True)
class CostBucket:
    name: str
    cost_model: CostModel


@dataclass(frozen=True)
class ExperimentConfig:
    cost_buckets: tuple[CostBucket, ...] = (
        CostBucket("base", CostModel()),
        CostBucket(
            "stress",
            CostModel(taker_fee_bps=7.5, slippage_bps=8.0, execution_risk_bps=4.0, funding_bps_per_day=3.0),
        ),
    )
    gate: AcceptanceGate = AcceptanceGate()
    regime_column: str = "regime"
    min_rows: int = 20
    include_overall_regime: bool = True


@dataclass(frozen=True)
class ExperimentResult:
    pair: str
    strategy_id: int
    strategy_name: str
    family: str
    regime: str
    cost_bucket: str
    status: str
    eligible: bool
    reason: str
    trades: int = 0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    total_return: float = 0.0
    observations: int = 0
    gross_return: float = 0.0
    total_fees: float = 0.0
    total_slippage: float = 0.0
    total_funding: float = 0.0
    total_execution_risk: float = 0.0
    total_partial_fill_cost: float = 0.0
    avg_gross_exposure: float = 0.0
    backtest_mode: str = "not_run"
    has_price_x: bool = False
    has_price_y: bool = False
    has_hedge_ratio: bool = False
    has_beta: bool = False
    has_funding_x: bool = False
    has_funding_y: bool = False


TWO_LEG_EXECUTION_INPUT_FLAGS = (
    "has_price_x",
    "has_price_y",
    "has_hedge_ratio",
    "has_beta",
    "has_funding_x",
    "has_funding_y",
)

TWO_LEG_EXECUTION_INPUT_FIELDS = tuple(column.replace("has_", "") for column in TWO_LEG_EXECUTION_INPUT_FLAGS)


def _required_columns(strategy: StrategySpec) -> set[str]:
    return STRATEGY_REQUIRED_COLUMNS.get(strategy.id, {"spread"})


def _missing_columns(frame: pd.DataFrame, strategy: StrategySpec) -> list[str]:
    return sorted(_required_columns(strategy).difference(frame.columns))


def _input_coverage_flags(frame: pd.DataFrame) -> dict[str, bool]:
    return {
        "has_price_x": "price_x" in frame.columns,
        "has_price_y": "price_y" in frame.columns,
        "has_hedge_ratio": "hedge_ratio" in frame.columns,
        "has_beta": "beta" in frame.columns,
        "has_funding_x": "funding_x_bps" in frame.columns,
        "has_funding_y": "funding_y_bps" in frame.columns,
    }


def _pairs_missing_cost_buckets(scope: pd.DataFrame, required_cost_buckets: tuple[str, ...]) -> list[str]:
    missing: list[str] = []
    if scope.empty or "pair" not in scope.columns or "cost_bucket" not in scope.columns:
        return missing
    required = set(required_cost_buckets)
    for pair, pair_rows in scope.groupby("pair"):
        missing_costs = sorted(required.difference(set(pair_rows["cost_bucket"])))
        if missing_costs:
            missing.append(f"{pair}[{'+'.join(missing_costs)}]")
    return missing


def _complete_two_leg_execution_input_scope(scope: pd.DataFrame) -> pd.DataFrame:
    if scope.empty:
        return scope
    complete = scope.copy()
    for column in TWO_LEG_EXECUTION_INPUT_FLAGS:
        if column not in complete.columns:
            complete[column] = False
    mask = complete.loc[:, TWO_LEG_EXECUTION_INPUT_FLAGS].fillna(False).astype(bool).all(axis=1)
    return complete[mask]


def _pairs_missing_two_leg_execution_inputs(scope: pd.DataFrame) -> list[str]:
    missing: list[str] = []
    if scope.empty or "pair" not in scope.columns:
        return missing
    scoped = scope.copy()
    for column in TWO_LEG_EXECUTION_INPUT_FLAGS:
        if column not in scoped.columns:
            scoped[column] = False
    for pair, pair_rows in scoped.groupby("pair"):
        pair_missing = [
            column.replace("has_", "")
            for column in TWO_LEG_EXECUTION_INPUT_FLAGS
            if not bool(pair_rows[column].fillna(False).astype(bool).all())
        ]
        if pair_missing:
            missing.append(f"{pair}[{'+'.join(pair_missing)}]")
    return missing


def _regime_slices(frame: pd.DataFrame, config: ExperimentConfig) -> Iterable[tuple[str, pd.DataFrame]]:
    if config.include_overall_regime:
        yield "ALL", frame
    if config.regime_column not in frame.columns:
        return
    for regime, regime_frame in frame.groupby(config.regime_column, sort=True):
        yield str(regime), regime_frame


class ExperimentHarness:
    """Runs strategies through one cost-aware scoreboard."""

    def __init__(
        self,
        strategies: Iterable[StrategySpec] = STRATEGIES,
        config: ExperimentConfig | None = None,
        feature_engine: FeatureEngine | None = None,
    ) -> None:
        self.strategies = tuple(strategies)
        self.config = config or ExperimentConfig()
        self.feature_engine = feature_engine or FeatureEngine()

    def run(self, datasets: Iterable[PairDataset]) -> pd.DataFrame:
        rows: list[ExperimentResult] = []
        for dataset in datasets:
            rows.extend(self._run_dataset(dataset))
        frame = pd.DataFrame(asdict(row) for row in rows)
        if frame.empty:
            return frame
        return self.rank(frame)

    def _run_dataset(self, dataset: PairDataset) -> list[ExperimentResult]:
        results: list[ExperimentResult] = []
        frame = self.feature_engine.score_frame(dataset.frame)
        for strategy in self.strategies:
            for regime, regime_frame in _regime_slices(frame, self.config):
                for bucket in self.config.cost_buckets:
                    results.append(self._run_one(dataset.pair, regime_frame, regime, strategy, bucket))
        return results

    def _run_one(
        self,
        pair: str,
        frame: pd.DataFrame,
        regime: str,
        strategy: StrategySpec,
        bucket: CostBucket,
    ) -> ExperimentResult:
        observations = len(frame)
        input_flags = _input_coverage_flags(frame)
        base = {
            "pair": pair,
            "strategy_id": strategy.id,
            "strategy_name": strategy.name,
            "family": strategy.family,
            "regime": regime,
            "cost_bucket": bucket.name,
            "observations": observations,
            **input_flags,
        }
        if observations < self.config.min_rows:
            return ExperimentResult(**base, status="skipped", eligible=False, reason=f"rows<{self.config.min_rows}")
        if strategy.signal_function is None:
            return ExperimentResult(**base, status="skipped", eligible=False, reason="no_signal_function")
        missing = _missing_columns(frame, strategy)
        if missing:
            return ExperimentResult(**base, status="skipped", eligible=False, reason=f"missing_columns:{','.join(missing)}")

        signal = strategy.signal_function(frame)
        backtest_mode = "two_leg" if {"price_x", "price_y"}.issubset(frame.columns) else "spread"
        result = backtest_two_leg_spread(frame, signal, bucket.cost_model) if backtest_mode == "two_leg" else backtest_pair(
            frame, signal, bucket.cost_model
        )
        eligible, reason = self.config.gate.evaluate(result)
        return ExperimentResult(
            **base,
            status="evaluated",
            eligible=eligible,
            reason=reason,
            trades=result.trades,
            profit_factor=result.profit_factor,
            expectancy=result.expectancy,
            sharpe=result.sharpe,
            max_drawdown=result.max_drawdown,
            win_rate=result.win_rate,
            total_return=result.total_return,
            gross_return=result.gross_return,
            total_fees=result.total_fees,
            total_slippage=result.total_slippage,
            total_funding=result.total_funding,
            total_execution_risk=result.total_execution_risk,
            total_partial_fill_cost=result.total_partial_fill_cost,
            avg_gross_exposure=result.avg_gross_exposure,
            backtest_mode=backtest_mode,
        )

    @staticmethod
    def rank(frame: pd.DataFrame) -> pd.DataFrame:
        ranked = frame.copy()
        ranked["rank_score"] = (
            ranked["profit_factor"].clip(upper=5.0).fillna(0.0) * 25.0
            + ranked["sharpe"].clip(lower=-5.0, upper=5.0).fillna(0.0) * 10.0
            + ranked["expectancy"].fillna(0.0) * 100.0
            + ranked["win_rate"].fillna(0.0) * 10.0
            - ranked["max_drawdown"].fillna(1.0) * 50.0
        )
        ranked.loc[ranked["status"] != "evaluated", "rank_score"] = -1_000_000.0
        ranked = ranked.sort_values(
            ["eligible", "rank_score", "profit_factor", "sharpe"],
            ascending=[False, False, False, False],
        )
        ranked["rank"] = range(1, len(ranked) + 1)
        return ranked

    def write_reports(self, results: pd.DataFrame, output_dir: str | Path) -> dict[str, Path]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        results = results.copy()
        if "backtest_mode" not in results.columns:
            results["backtest_mode"] = "unknown"
        paths = {
            "ablation": output / "ablation_report.csv",
            "all_results": output / "experiment_results.csv",
            "acceptance": output / "acceptance_report.csv",
            "strategy_summary": output / "strategy_summary.csv",
            "regime_summary": output / "regime_summary.csv",
            "regime_pair_strategy": output / "regime_pair_strategy_report.csv",
            "coverage": output / "strategy_coverage.csv",
        }
        results.to_csv(paths["all_results"], index=False)
        evaluated = results[results["status"] == "evaluated"].copy()
        if evaluated.empty:
            pd.DataFrame().to_csv(paths["strategy_summary"], index=False)
            pd.DataFrame().to_csv(paths["regime_summary"], index=False)
        else:
            _ensure_cost_columns(evaluated)
            evaluated["cost_drag"] = evaluated["gross_return"].fillna(0.0) - evaluated["total_return"].fillna(0.0)
            evaluated.groupby(["strategy_id", "strategy_name", "family"], as_index=False).agg(
                runs=("status", "count"),
                eligible_runs=("eligible", "sum"),
                median_profit_factor=("profit_factor", "median"),
                median_sharpe=("sharpe", "median"),
                median_max_drawdown=("max_drawdown", "median"),
                median_gross_return=("gross_return", "median"),
                median_total_return=("total_return", "median"),
                median_cost_drag=("cost_drag", "median"),
                total_fees=("total_fees", "sum"),
                total_slippage=("total_slippage", "sum"),
                total_funding=("total_funding", "sum"),
                total_execution_risk=("total_execution_risk", "sum"),
                total_partial_fill_cost=("total_partial_fill_cost", "sum"),
                median_avg_gross_exposure=("avg_gross_exposure", "median"),
                total_trades=("trades", "sum"),
                two_leg_runs=("backtest_mode", lambda values: int((values == "two_leg").sum())),
                spread_runs=("backtest_mode", lambda values: int((values == "spread").sum())),
            ).sort_values(["eligible_runs", "median_profit_factor"], ascending=[False, False]).to_csv(
                paths["strategy_summary"], index=False
            )
            evaluated.groupby(["regime", "strategy_name"], as_index=False).agg(
                runs=("status", "count"),
                eligible_runs=("eligible", "sum"),
                median_profit_factor=("profit_factor", "median"),
                median_sharpe=("sharpe", "median"),
                median_max_drawdown=("max_drawdown", "median"),
                median_gross_return=("gross_return", "median"),
                median_total_return=("total_return", "median"),
                median_cost_drag=("cost_drag", "median"),
                total_fees=("total_fees", "sum"),
                total_slippage=("total_slippage", "sum"),
                total_funding=("total_funding", "sum"),
                total_execution_risk=("total_execution_risk", "sum"),
                total_partial_fill_cost=("total_partial_fill_cost", "sum"),
                median_avg_gross_exposure=("avg_gross_exposure", "median"),
                total_trades=("trades", "sum"),
                two_leg_runs=("backtest_mode", lambda values: int((values == "two_leg").sum())),
                spread_runs=("backtest_mode", lambda values: int((values == "spread").sum())),
            ).sort_values(["regime", "eligible_runs", "median_profit_factor"], ascending=[True, False, False]).to_csv(
                paths["regime_summary"], index=False
            )
        regime_pair_strategy_report(results).to_csv(paths["regime_pair_strategy"], index=False)
        results.groupby(["strategy_id", "strategy_name", "status", "reason"], as_index=False).agg(
            rows=("status", "count")
        ).to_csv(paths["coverage"], index=False)
        write_ablation_report(results, paths["ablation"])
        write_strategy_acceptance_report(results, paths["acceptance"], self.config.gate)
        return paths


def _ensure_cost_columns(frame: pd.DataFrame) -> None:
    for column in (
        "gross_return",
        "total_return",
        "total_fees",
        "total_slippage",
        "total_funding",
        "total_execution_risk",
        "total_partial_fill_cost",
        "avg_gross_exposure",
    ):
        if column not in frame.columns:
            frame[column] = 0.0


def strategy_acceptance_report(results: pd.DataFrame, gate: AcceptanceGate | None = None) -> pd.DataFrame:
    gate = gate or AcceptanceGate()
    if results.empty:
        return pd.DataFrame(
            columns=[
                "strategy_id",
                "strategy_name",
                "family",
                "production_eligible",
                "preferred_eligible",
                "acceptance_reason",
                "preferred_reason",
                "evaluated_runs",
                "passing_runs",
                "pairs_tested",
                "passing_pairs",
                "two_leg_pairs_tested",
                "two_leg_execution_input_pairs",
                "two_leg_passing_pairs",
                "required_cost_buckets",
                "required_backtest_mode",
                "required_two_leg_inputs",
                "total_trades",
                "median_profit_factor",
                "median_sharpe",
                "worst_drawdown",
            ]
        )

    rows: list[dict[str, object]] = []
    for (strategy_id, strategy_name, family), group in results.groupby(["strategy_id", "strategy_name", "family"], sort=True):
        decision = gate.evaluate_strategy(group)
        rows.append(
            {
                "strategy_id": strategy_id,
                "strategy_name": strategy_name,
                "family": family,
                **decision,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["production_eligible", "preferred_eligible", "passing_pairs", "median_profit_factor"],
        ascending=[False, False, False, False],
    )


def write_strategy_acceptance_report(
    results: pd.DataFrame,
    output_path: str | Path,
    gate: AcceptanceGate | None = None,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    strategy_acceptance_report(results, gate).to_csv(output, index=False)
    return output
