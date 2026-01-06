"""
Portfolio Management
=====================
Multi-stock portfolio analysis with:
- Correlation tracking
- Portfolio-level risk management
- Combined performance metrics
- Diversification analysis
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from core.config import config
from backtest.engine import (
    Backtester, BacktestResult, BacktestMetrics,
    RiskManager, PositionSizingMethod, Trade, PositionType
)

logger = logging.getLogger("stock_analyzer.portfolio")


@dataclass
class PortfolioPosition:
    """A position within the portfolio."""
    symbol: str
    shares: int
    entry_price: float
    entry_date: datetime
    current_price: float
    position_type: PositionType = PositionType.LONG

    @property
    def market_value(self) -> float:
        if self.position_type == PositionType.LONG:
            return self.shares * self.current_price
        else:
            # Short: value is the unrealized P&L
            return self.shares * (self.entry_price - self.current_price)

    @property
    def unrealized_pnl(self) -> float:
        if self.position_type == PositionType.LONG:
            return self.shares * (self.current_price - self.entry_price)
        else:
            return self.shares * (self.entry_price - self.current_price)

    @property
    def unrealized_pnl_pct(self) -> float:
        if self.entry_price == 0:
            return 0.0
        if self.position_type == PositionType.LONG:
            return (self.current_price / self.entry_price - 1) * 100
        else:
            return (self.entry_price / self.current_price - 1) * 100


@dataclass
class PortfolioMetrics:
    """Portfolio-level performance metrics."""
    total_value: float
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    beta: float
    alpha: float

    # Diversification metrics
    num_positions: int
    concentration_ratio: float  # Top position as % of portfolio
    correlation_avg: float  # Average pairwise correlation
    diversification_ratio: float

    # Individual stock metrics
    best_performer: str
    worst_performer: str
    stock_returns: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_value': self.total_value,
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'max_drawdown': self.max_drawdown,
            'beta': self.beta,
            'alpha': self.alpha,
            'num_positions': self.num_positions,
            'concentration_ratio': self.concentration_ratio,
            'correlation_avg': self.correlation_avg,
            'diversification_ratio': self.diversification_ratio,
            'best_performer': self.best_performer,
            'worst_performer': self.worst_performer,
            'stock_returns': self.stock_returns
        }


class CorrelationTracker:
    """Track and analyze correlations between portfolio assets."""

    def __init__(self, lookback: int = 60):
        self.lookback = lookback
        self.returns_history: Dict[str, pd.Series] = {}

    def update(self, symbol: str, returns: pd.Series) -> None:
        """Update returns history for a symbol."""
        self.returns_history[symbol] = returns.tail(self.lookback)

    def get_correlation_matrix(self) -> pd.DataFrame:
        """Calculate correlation matrix for all tracked symbols."""
        if len(self.returns_history) < 2:
            return pd.DataFrame()

        returns_df = pd.DataFrame(self.returns_history)
        return returns_df.corr()

    def get_average_correlation(self) -> float:
        """Get average pairwise correlation."""
        corr_matrix = self.get_correlation_matrix()
        if corr_matrix.empty:
            return 0.0

        # Get upper triangle (excluding diagonal)
        n = len(corr_matrix)
        if n < 2:
            return 0.0

        upper_tri = corr_matrix.values[np.triu_indices(n, k=1)]
        return float(np.mean(upper_tri)) if len(upper_tri) > 0 else 0.0

    def get_diversification_ratio(self) -> float:
        """
        Calculate diversification ratio.

        DR = weighted avg volatility / portfolio volatility
        Higher is better (more diversification benefit)
        """
        if len(self.returns_history) < 2:
            return 1.0

        returns_df = pd.DataFrame(self.returns_history)

        # Assume equal weights for simplicity
        weights = np.ones(len(returns_df.columns)) / len(returns_df.columns)

        # Individual volatilities
        individual_vols = returns_df.std()
        weighted_avg_vol = np.dot(weights, individual_vols)

        # Portfolio volatility
        cov_matrix = returns_df.cov()
        portfolio_var = np.dot(weights, np.dot(cov_matrix, weights))
        portfolio_vol = np.sqrt(portfolio_var)

        if portfolio_vol == 0:
            return 1.0

        return weighted_avg_vol / portfolio_vol


class PortfolioRiskManager:
    """
    Portfolio-level risk management.

    Enforces:
    - Maximum position size per stock
    - Maximum sector exposure
    - Portfolio-wide drawdown limits
    - Correlation-based position limits
    """

    def __init__(
        self,
        max_position_pct: float = 0.20,  # Max 20% in single stock
        max_portfolio_drawdown: float = 0.25,  # 25% portfolio drawdown limit
        max_correlation_exposure: float = 0.80,  # Don't add highly correlated positions
        target_num_positions: int = 10
    ):
        self.max_position_pct = max_position_pct
        self.max_portfolio_drawdown = max_portfolio_drawdown
        self.max_correlation_exposure = max_correlation_exposure
        self.target_num_positions = target_num_positions

        self.peak_value = 0.0
        self.is_killed = False

    def can_add_position(
        self,
        symbol: str,
        position_value: float,
        portfolio_value: float,
        correlation_tracker: CorrelationTracker
    ) -> Tuple[bool, str]:
        """
        Check if a new position can be added.

        Returns:
            (can_add, reason)
        """
        if self.is_killed:
            return False, "Portfolio risk limit triggered"

        # Check position size
        if portfolio_value > 0:
            position_pct = position_value / portfolio_value
            if position_pct > self.max_position_pct:
                return False, f"Position size {position_pct:.1%} exceeds limit {self.max_position_pct:.1%}"

        # Check correlation with existing positions
        corr_matrix = correlation_tracker.get_correlation_matrix()
        if symbol in corr_matrix.columns:
            max_corr = corr_matrix[symbol].drop(symbol, errors='ignore').max()
            if max_corr > self.max_correlation_exposure:
                return False, f"High correlation ({max_corr:.2f}) with existing position"

        return True, "OK"

    def calculate_position_weight(
        self,
        symbol: str,
        volatility: float,
        correlation_tracker: CorrelationTracker
    ) -> float:
        """
        Calculate optimal position weight based on volatility and correlations.

        Uses inverse volatility weighting adjusted for correlations.
        """
        base_weight = 1.0 / self.target_num_positions

        # Adjust for volatility (lower vol = higher weight)
        target_vol = 0.20  # 20% target volatility
        if volatility > 0:
            vol_adjustment = target_vol / volatility
            vol_adjustment = min(2.0, max(0.5, vol_adjustment))  # Clamp
        else:
            vol_adjustment = 1.0

        # Adjust for correlation (higher correlation = lower weight)
        corr_matrix = correlation_tracker.get_correlation_matrix()
        if symbol in corr_matrix.columns and len(corr_matrix) > 1:
            avg_corr = corr_matrix[symbol].drop(symbol, errors='ignore').mean()
            corr_adjustment = 1.0 - (avg_corr * 0.5)  # Reduce weight for high correlation
        else:
            corr_adjustment = 1.0

        final_weight = base_weight * vol_adjustment * corr_adjustment
        return min(self.max_position_pct, max(0.02, final_weight))

    def update_portfolio_value(self, current_value: float) -> None:
        """Update portfolio value and check drawdown limits."""
        self.peak_value = max(self.peak_value, current_value)

        if self.peak_value > 0:
            drawdown = (self.peak_value - current_value) / self.peak_value
            if drawdown >= self.max_portfolio_drawdown:
                self.is_killed = True
                logger.warning(f"Portfolio drawdown limit exceeded: {drawdown:.1%}")


@dataclass
class PortfolioBacktestResult:
    """Results from portfolio backtest."""
    metrics: PortfolioMetrics
    equity_curve: pd.Series
    drawdown_curve: pd.Series
    individual_results: Dict[str, BacktestResult]
    correlation_matrix: pd.DataFrame
    position_history: pd.DataFrame  # Daily positions


class PortfolioBacktester:
    """
    Backtest a portfolio of multiple stocks.

    Runs individual backtests and combines them with
    portfolio-level risk management.
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission_pct: float = 0.001,
        slippage_pct: float = 0.001,
        risk_manager: Optional[PortfolioRiskManager] = None
    ):
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.risk_manager = risk_manager or PortfolioRiskManager()
        self.correlation_tracker = CorrelationTracker()

    def run(
        self,
        stock_data: Dict[str, pd.DataFrame],
        signals: Dict[str, pd.Series],
        benchmark: Optional[pd.Series] = None
    ) -> PortfolioBacktestResult:
        """
        Run portfolio backtest.

        Args:
            stock_data: Dict of symbol -> price DataFrame
            signals: Dict of symbol -> signal Series
            benchmark: Optional benchmark for comparison

        Returns:
            PortfolioBacktestResult
        """
        n_stocks = len(stock_data)
        if n_stocks == 0:
            raise ValueError("No stocks provided for portfolio backtest")

        # Allocate capital equally initially
        capital_per_stock = self.initial_capital / n_stocks

        # Run individual backtests
        individual_results: Dict[str, BacktestResult] = {}
        stock_returns: Dict[str, pd.Series] = {}

        for symbol, prices in stock_data.items():
            if symbol not in signals:
                continue

            # Update correlation tracker
            if 'returns' in prices.columns:
                self.correlation_tracker.update(symbol, prices['returns'])
            else:
                returns = prices['close'].pct_change()
                self.correlation_tracker.update(symbol, returns)

            # Calculate position weight
            volatility = prices['close'].pct_change().std() * np.sqrt(252)
            weight = self.risk_manager.calculate_position_weight(
                symbol, volatility, self.correlation_tracker
            )
            allocated_capital = self.initial_capital * weight

            # Create individual backtester
            bt = Backtester(
                initial_capital=allocated_capital,
                commission_pct=self.commission_pct,
                slippage_pct=self.slippage_pct
            )

            try:
                result = bt.run(prices, signals[symbol], symbol=symbol, benchmark=benchmark)
                individual_results[symbol] = result
                stock_returns[symbol] = result.metrics.total_return
            except Exception as e:
                logger.warning(f"Backtest failed for {symbol}: {e}")
                continue

        if not individual_results:
            raise ValueError("All individual backtests failed")

        # Combine equity curves
        combined_equity = self._combine_equity_curves(individual_results)

        # Calculate portfolio metrics
        metrics = self._calculate_portfolio_metrics(
            combined_equity, individual_results, stock_returns, benchmark
        )

        # Calculate drawdown
        peak = combined_equity.expanding().max()
        drawdown = (combined_equity - peak) / peak

        # Get correlation matrix
        corr_matrix = self.correlation_tracker.get_correlation_matrix()

        return PortfolioBacktestResult(
            metrics=metrics,
            equity_curve=combined_equity,
            drawdown_curve=drawdown,
            individual_results=individual_results,
            correlation_matrix=corr_matrix,
            position_history=pd.DataFrame()  # TODO: Track daily positions
        )

    def _combine_equity_curves(
        self,
        results: Dict[str, BacktestResult]
    ) -> pd.Series:
        """Combine individual equity curves into portfolio equity."""
        equity_curves = {}
        for symbol, result in results.items():
            equity_curves[symbol] = result.equity_curve

        equity_df = pd.DataFrame(equity_curves)
        combined = equity_df.sum(axis=1)

        return combined

    def _calculate_portfolio_metrics(
        self,
        equity: pd.Series,
        individual_results: Dict[str, BacktestResult],
        stock_returns: Dict[str, float],
        benchmark: Optional[pd.Series]
    ) -> PortfolioMetrics:
        """Calculate portfolio-level metrics."""
        returns = equity.pct_change().dropna()

        # Basic metrics
        total_value = equity.iloc[-1] if len(equity) > 0 else self.initial_capital
        total_return = (total_value / self.initial_capital) - 1

        trading_days = len(equity)
        years = trading_days / 252
        annualized_return = (1 + total_return) ** (1 / max(years, 0.01)) - 1 if years > 0 else 0

        volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0

        # Risk-adjusted metrics
        sharpe_ratio = (returns.mean() * 252) / (volatility + 1e-10) if volatility > 0 else 0

        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 1 else 0
        sortino_ratio = (returns.mean() * 252) / (downside_std + 1e-10) if downside_std > 0 else 0

        # Drawdown
        peak = equity.expanding().max()
        drawdown = (equity - peak) / peak
        max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0

        # Benchmark comparison
        if benchmark is not None and len(benchmark) > 1:
            bench_returns = benchmark.pct_change().dropna()
            if len(bench_returns) > 1 and len(returns) > 1:
                min_len = min(len(returns), len(bench_returns))
                cov = np.cov(returns.iloc[:min_len], bench_returns.iloc[:min_len])[0, 1]
                bench_var = np.var(bench_returns.iloc[:min_len])
                beta = cov / bench_var if bench_var > 0 else 0
                alpha = annualized_return - beta * (bench_returns.mean() * 252)
            else:
                beta, alpha = 0, annualized_return
        else:
            beta, alpha = 0, annualized_return

        # Diversification metrics
        num_positions = len(individual_results)

        if stock_returns:
            max_weight = max(stock_returns.values()) if stock_returns else 0
            concentration_ratio = max_weight / (total_return + 1e-10) if total_return > 0 else 0
        else:
            concentration_ratio = 0

        correlation_avg = self.correlation_tracker.get_average_correlation()
        diversification_ratio = self.correlation_tracker.get_diversification_ratio()

        # Best/worst performers
        if stock_returns:
            sorted_returns = sorted(stock_returns.items(), key=lambda x: x[1], reverse=True)
            best_performer = sorted_returns[0][0] if sorted_returns else "N/A"
            worst_performer = sorted_returns[-1][0] if sorted_returns else "N/A"
        else:
            best_performer = worst_performer = "N/A"

        return PortfolioMetrics(
            total_value=total_value,
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            beta=beta,
            alpha=alpha,
            num_positions=num_positions,
            concentration_ratio=concentration_ratio,
            correlation_avg=correlation_avg,
            diversification_ratio=diversification_ratio,
            best_performer=best_performer,
            worst_performer=worst_performer,
            stock_returns=stock_returns
        )


def run_portfolio_backtest(
    symbols: List[str],
    data_fetcher,
    model,
    feature_engine_class,
    period: str = "2y",
    initial_capital: float = 100000.0
) -> PortfolioBacktestResult:
    """
    Convenience function to run a full portfolio backtest.

    Args:
        symbols: List of stock symbols
        data_fetcher: DataFetcher instance
        model: Trained ensemble model
        feature_engine_class: FeatureEngine class
        period: Time period for data
        initial_capital: Starting capital

    Returns:
        PortfolioBacktestResult
    """
    from models.ensemble import create_labels, prepare_training_data
    from backtest.engine import generate_signals_from_predictions

    stock_data = {}
    signals = {}

    for symbol in symbols:
        try:
            # Fetch data
            data = data_fetcher.get_stock_data(symbol, period)
            if data is None or data.prices is None:
                continue

            stock_data[symbol] = data.prices

            # Generate features and predictions
            engine = feature_engine_class(data.prices)
            ml_features = engine.get_ml_features()

            if len(ml_features) > 100:
                # Get predictions
                X, y = prepare_training_data(ml_features, data.prices)
                if len(X) > 50:
                    proba = model.predict_proba(X)
                    predictions = pd.Series(
                        proba[:, 1] if proba.shape[1] > 1 else proba[:, 0],
                        index=X.index
                    )
                    signals[symbol] = generate_signals_from_predictions(predictions)

        except Exception as e:
            logger.warning(f"Failed to process {symbol}: {e}")
            continue

    # Run portfolio backtest
    portfolio_bt = PortfolioBacktester(initial_capital=initial_capital)
    return portfolio_bt.run(stock_data, signals)
