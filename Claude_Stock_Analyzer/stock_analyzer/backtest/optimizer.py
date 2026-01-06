"""
Strategy Hyperparameter Optimizer
==================================
Optimize trading strategy parameters using:
- Grid search
- Random search
- Bayesian optimization (optional)
- Walk-forward optimization
"""

import logging
from typing import Dict, List, Optional, Tuple, Callable, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools

import numpy as np
import pandas as pd

from backtest.engine import Backtester, BacktestResult, RiskManager, PositionSizingMethod

logger = logging.getLogger("stock_analyzer.optimizer")


@dataclass
class ParameterSpace:
    """Define a parameter search space."""
    name: str
    param_type: str  # 'float', 'int', 'choice'
    low: Optional[float] = None
    high: Optional[float] = None
    step: Optional[float] = None
    choices: Optional[List[Any]] = None

    def get_values(self) -> List[Any]:
        """Get all values in this parameter space."""
        if self.param_type == 'choice':
            return self.choices or []
        elif self.param_type == 'float':
            if self.step:
                return list(np.arange(self.low, self.high + self.step, self.step))
            else:
                return [self.low, self.high]
        elif self.param_type == 'int':
            step = int(self.step) if self.step else 1
            return list(range(int(self.low), int(self.high) + 1, step))
        return []

    def sample(self) -> Any:
        """Sample a random value from this space."""
        if self.param_type == 'choice':
            return np.random.choice(self.choices)
        elif self.param_type == 'float':
            return np.random.uniform(self.low, self.high)
        elif self.param_type == 'int':
            return np.random.randint(self.low, self.high + 1)
        return None


@dataclass
class OptimizationResult:
    """Result of parameter optimization."""
    best_params: Dict[str, Any]
    best_score: float
    best_metrics: Dict[str, float]
    all_results: List[Dict[str, Any]]
    optimization_time: float
    n_iterations: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'best_metrics': self.best_metrics,
            'n_iterations': self.n_iterations,
            'optimization_time': self.optimization_time
        }

    def get_top_n(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get top N parameter combinations by score."""
        sorted_results = sorted(self.all_results, key=lambda x: x['score'], reverse=True)
        return sorted_results[:n]


class StrategyOptimizer:
    """
    Optimize strategy parameters using various methods.

    Supports:
    - Grid search: Exhaustive search over parameter grid
    - Random search: Random sampling from parameter space
    - Walk-forward: Rolling window optimization
    """

    # Common parameter spaces for trading strategies
    COMMON_PARAMS = {
        'stop_loss': ParameterSpace('stop_loss', 'float', 0.02, 0.10, 0.01),
        'take_profit': ParameterSpace('take_profit', 'float', 0.05, 0.25, 0.02),
        'buy_threshold': ParameterSpace('buy_threshold', 'float', 0.50, 0.80, 0.05),
        'sell_threshold': ParameterSpace('sell_threshold', 'float', 0.20, 0.50, 0.05),
        'position_size': ParameterSpace('position_size', 'float', 0.05, 0.25, 0.05),
        'lookback_period': ParameterSpace('lookback_period', 'int', 10, 50, 5),
    }

    def __init__(
        self,
        objective: str = 'sharpe',
        n_jobs: int = 1,
        verbose: bool = True
    ):
        """
        Initialize optimizer.

        Args:
            objective: Metric to optimize ('sharpe', 'sortino', 'returns', 'calmar')
            n_jobs: Number of parallel jobs (1 = sequential)
            verbose: Print progress
        """
        self.objective = objective
        self.n_jobs = n_jobs
        self.verbose = verbose

    def _get_score(self, result: BacktestResult) -> float:
        """Extract optimization score from backtest result."""
        metrics = result.metrics

        if self.objective == 'sharpe':
            return metrics.sharpe_ratio
        elif self.objective == 'sortino':
            return metrics.sortino_ratio
        elif self.objective == 'returns':
            return metrics.annualized_return
        elif self.objective == 'calmar':
            return metrics.calmar_ratio
        elif self.objective == 'profit_factor':
            return metrics.profit_factor if metrics.profit_factor != float('inf') else 10.0
        else:
            return metrics.sharpe_ratio

    def _run_backtest_with_params(
        self,
        prices: pd.DataFrame,
        signals: pd.Series,
        params: Dict[str, Any],
        symbol: str = "STOCK"
    ) -> Tuple[Dict[str, Any], float, Dict[str, float]]:
        """Run a single backtest with given parameters."""
        try:
            # Create risk manager with parameters
            risk_manager = RiskManager(
                max_position_pct=params.get('position_size', 0.1),
                stop_loss_pct=params.get('stop_loss', 0.05),
                take_profit_pct=params.get('take_profit', 0.15),
                sizing_method=params.get('sizing_method', PositionSizingMethod.VOLATILITY_ADJUSTED)
            )

            # Create backtester
            bt = Backtester(
                initial_capital=params.get('initial_capital', 100000),
                commission_pct=params.get('commission', 0.001),
                slippage_pct=params.get('slippage', 0.001),
                risk_manager=risk_manager
            )

            # Adjust signals based on thresholds if predictions provided
            if 'predictions' in params:
                from backtest.engine import generate_signals_from_predictions
                signals = generate_signals_from_predictions(
                    params['predictions'],
                    buy_threshold=params.get('buy_threshold', 0.6),
                    sell_threshold=params.get('sell_threshold', 0.4)
                )

            # Run backtest
            result = bt.run(prices, signals, symbol=symbol)
            score = self._get_score(result)

            metrics = {
                'sharpe': result.metrics.sharpe_ratio,
                'sortino': result.metrics.sortino_ratio,
                'returns': result.metrics.annualized_return,
                'max_drawdown': result.metrics.max_drawdown,
                'win_rate': result.metrics.win_rate,
                'total_trades': result.metrics.total_trades
            }

            return params, score, metrics

        except Exception as e:
            logger.warning(f"Backtest failed with params {params}: {e}")
            return params, -np.inf, {}

    def grid_search(
        self,
        prices: pd.DataFrame,
        signals: pd.Series,
        param_spaces: Dict[str, ParameterSpace],
        symbol: str = "STOCK"
    ) -> OptimizationResult:
        """
        Exhaustive grid search over parameter combinations.

        Args:
            prices: Price DataFrame
            signals: Signal Series (or None if using predictions)
            param_spaces: Dict of parameter name -> ParameterSpace
            symbol: Stock symbol

        Returns:
            OptimizationResult
        """
        start_time = datetime.now()

        # Generate all parameter combinations
        param_names = list(param_spaces.keys())
        param_values = [param_spaces[name].get_values() for name in param_names]
        all_combinations = list(itertools.product(*param_values))

        if self.verbose:
            logger.info(f"Grid search: {len(all_combinations)} combinations")

        all_results = []
        best_score = -np.inf
        best_params = {}
        best_metrics = {}

        for i, values in enumerate(all_combinations):
            params = dict(zip(param_names, values))

            # Skip invalid combinations (buy_threshold must be > sell_threshold)
            if 'buy_threshold' in params and 'sell_threshold' in params:
                if params['buy_threshold'] <= params['sell_threshold']:
                    continue

            params_result, score, metrics = self._run_backtest_with_params(
                prices, signals, params, symbol
            )

            result_entry = {
                'params': params,
                'score': score,
                'metrics': metrics
            }
            all_results.append(result_entry)

            if score > best_score:
                best_score = score
                best_params = params
                best_metrics = metrics

            if self.verbose and (i + 1) % 10 == 0:
                logger.info(f"Progress: {i + 1}/{len(all_combinations)}, Best score: {best_score:.4f}")

        elapsed = (datetime.now() - start_time).total_seconds()

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            best_metrics=best_metrics,
            all_results=all_results,
            optimization_time=elapsed,
            n_iterations=len(all_results)
        )

    def random_search(
        self,
        prices: pd.DataFrame,
        signals: pd.Series,
        param_spaces: Dict[str, ParameterSpace],
        n_iterations: int = 100,
        symbol: str = "STOCK"
    ) -> OptimizationResult:
        """
        Random search over parameter space.

        More efficient than grid search for high-dimensional spaces.

        Args:
            prices: Price DataFrame
            signals: Signal Series
            param_spaces: Dict of parameter name -> ParameterSpace
            n_iterations: Number of random samples
            symbol: Stock symbol

        Returns:
            OptimizationResult
        """
        start_time = datetime.now()

        if self.verbose:
            logger.info(f"Random search: {n_iterations} iterations")

        all_results = []
        best_score = -np.inf
        best_params = {}
        best_metrics = {}

        for i in range(n_iterations):
            # Sample random parameters
            params = {name: space.sample() for name, space in param_spaces.items()}

            # Skip invalid combinations
            if 'buy_threshold' in params and 'sell_threshold' in params:
                if params['buy_threshold'] <= params['sell_threshold']:
                    continue

            params_result, score, metrics = self._run_backtest_with_params(
                prices, signals, params, symbol
            )

            result_entry = {
                'params': params,
                'score': score,
                'metrics': metrics
            }
            all_results.append(result_entry)

            if score > best_score:
                best_score = score
                best_params = params
                best_metrics = metrics

            if self.verbose and (i + 1) % 20 == 0:
                logger.info(f"Progress: {i + 1}/{n_iterations}, Best score: {best_score:.4f}")

        elapsed = (datetime.now() - start_time).total_seconds()

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            best_metrics=best_metrics,
            all_results=all_results,
            optimization_time=elapsed,
            n_iterations=len(all_results)
        )

    def walk_forward_optimize(
        self,
        prices: pd.DataFrame,
        signals: pd.Series,
        param_spaces: Dict[str, ParameterSpace],
        n_splits: int = 5,
        train_pct: float = 0.7,
        symbol: str = "STOCK"
    ) -> OptimizationResult:
        """
        Walk-forward optimization to avoid overfitting.

        Optimizes on training window, validates on test window,
        then rolls forward.

        Args:
            prices: Price DataFrame
            signals: Signal Series
            param_spaces: Dict of parameter name -> ParameterSpace
            n_splits: Number of train/test splits
            train_pct: Percentage of each window for training
            symbol: Stock symbol

        Returns:
            OptimizationResult with out-of-sample performance
        """
        start_time = datetime.now()

        n_samples = len(prices)
        window_size = n_samples // n_splits

        if self.verbose:
            logger.info(f"Walk-forward: {n_splits} splits, window size: {window_size}")

        all_window_results = []
        oos_scores = []  # Out-of-sample scores

        for split_idx in range(n_splits):
            window_start = split_idx * window_size
            window_end = min((split_idx + 1) * window_size, n_samples)

            train_end = window_start + int((window_end - window_start) * train_pct)

            # Training data
            train_prices = prices.iloc[window_start:train_end]
            train_signals = signals.iloc[window_start:train_end]

            # Test data
            test_prices = prices.iloc[train_end:window_end]
            test_signals = signals.iloc[train_end:window_end]

            if len(train_prices) < 50 or len(test_prices) < 20:
                continue

            # Optimize on training data (use random search for speed)
            train_result = self.random_search(
                train_prices, train_signals, param_spaces,
                n_iterations=50, symbol=symbol
            )

            # Validate on test data
            _, oos_score, oos_metrics = self._run_backtest_with_params(
                test_prices, test_signals, train_result.best_params, symbol
            )

            oos_scores.append(oos_score)
            all_window_results.append({
                'split': split_idx,
                'train_score': train_result.best_score,
                'test_score': oos_score,
                'params': train_result.best_params,
                'metrics': oos_metrics
            })

            if self.verbose:
                logger.info(
                    f"Split {split_idx + 1}: Train={train_result.best_score:.4f}, "
                    f"Test={oos_score:.4f}"
                )

        # Use parameters from best OOS window
        if all_window_results:
            best_window = max(all_window_results, key=lambda x: x['test_score'])
            best_params = best_window['params']
            best_score = np.mean(oos_scores) if oos_scores else 0
            best_metrics = best_window['metrics']
        else:
            best_params = {}
            best_score = 0
            best_metrics = {}

        elapsed = (datetime.now() - start_time).total_seconds()

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            best_metrics=best_metrics,
            all_results=all_window_results,
            optimization_time=elapsed,
            n_iterations=n_splits
        )


def optimize_strategy(
    prices: pd.DataFrame,
    predictions: pd.Series,
    method: str = 'random',
    n_iterations: int = 100,
    objective: str = 'sharpe'
) -> OptimizationResult:
    """
    Convenience function to optimize a strategy.

    Args:
        prices: Price DataFrame with OHLCV
        predictions: Model probability predictions
        method: 'grid', 'random', or 'walk_forward'
        n_iterations: Number of iterations for random search
        objective: Metric to optimize

    Returns:
        OptimizationResult
    """
    from backtest.engine import generate_signals_from_predictions

    # Default parameter spaces
    param_spaces = {
        'stop_loss': ParameterSpace('stop_loss', 'float', 0.03, 0.10, 0.01),
        'take_profit': ParameterSpace('take_profit', 'float', 0.08, 0.20, 0.02),
        'buy_threshold': ParameterSpace('buy_threshold', 'float', 0.55, 0.75, 0.05),
        'sell_threshold': ParameterSpace('sell_threshold', 'float', 0.25, 0.45, 0.05),
        'position_size': ParameterSpace('position_size', 'float', 0.05, 0.15, 0.025),
    }

    # Generate initial signals with default thresholds
    signals = generate_signals_from_predictions(predictions)

    optimizer = StrategyOptimizer(objective=objective, verbose=True)

    if method == 'grid':
        return optimizer.grid_search(prices, signals, param_spaces)
    elif method == 'random':
        return optimizer.random_search(prices, signals, param_spaces, n_iterations)
    elif method == 'walk_forward':
        return optimizer.walk_forward_optimize(prices, signals, param_spaces)
    else:
        raise ValueError(f"Unknown method: {method}")


@dataclass
class SensitivityResult:
    """Result of parameter sensitivity analysis."""
    parameter: str
    values: List[Any]
    scores: List[float]
    sensitivity: float  # Standard deviation of scores
    best_value: Any
    worst_value: Any


def analyze_parameter_sensitivity(
    prices: pd.DataFrame,
    signals: pd.Series,
    param_name: str,
    param_space: ParameterSpace,
    base_params: Optional[Dict[str, Any]] = None,
    objective: str = 'sharpe'
) -> SensitivityResult:
    """
    Analyze sensitivity of results to a single parameter.

    Useful for understanding which parameters matter most.

    Args:
        prices: Price DataFrame
        signals: Signal Series
        param_name: Name of parameter to analyze
        param_space: ParameterSpace for the parameter
        base_params: Base parameters (others held constant)
        objective: Metric to optimize

    Returns:
        SensitivityResult
    """
    base_params = base_params or {}
    optimizer = StrategyOptimizer(objective=objective, verbose=False)

    values = param_space.get_values()
    scores = []

    for value in values:
        params = base_params.copy()
        params[param_name] = value

        _, score, _ = optimizer._run_backtest_with_params(prices, signals, params)
        scores.append(score if score != -np.inf else 0)

    best_idx = np.argmax(scores)
    worst_idx = np.argmin(scores)

    return SensitivityResult(
        parameter=param_name,
        values=values,
        scores=scores,
        sensitivity=np.std(scores),
        best_value=values[best_idx],
        worst_value=values[worst_idx]
    )
