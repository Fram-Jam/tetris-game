"""
Benchmarking Module
====================
Compare trading strategies and algorithms with:
- Multiple benchmark strategies
- Statistical significance testing
- Performance tracking over time
- Leaderboard management
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np
import pandas as pd

from core.config import DATA_DIR
from backtest.engine import Backtester, BacktestResult, BacktestMetrics

logger = logging.getLogger("stock_analyzer.benchmark")


@dataclass
class StrategyResult:
    """Result from a strategy benchmark."""
    name: str
    description: str
    metrics: BacktestMetrics
    equity_curve: pd.Series
    run_at: datetime = field(default_factory=datetime.now)
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'description': self.description,
            'metrics': self.metrics.to_dict(),
            'run_at': self.run_at.isoformat(),
            'parameters': self.parameters
        }


class BenchmarkStrategies:
    """Built-in benchmark strategies for comparison."""
    
    @staticmethod
    def buy_and_hold(prices: pd.DataFrame) -> pd.Series:
        """Simple buy and hold strategy."""
        signals = pd.Series(0, index=prices.index)
        signals.iloc[0] = 1  # Buy on first day
        return signals
    
    @staticmethod
    def sma_crossover(prices: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.Series:
        """SMA crossover strategy."""
        close = prices['close']
        sma_fast = close.rolling(fast).mean()
        sma_slow = close.rolling(slow).mean()
        
        signals = pd.Series(0, index=prices.index)
        signals[sma_fast > sma_slow] = 1
        signals[sma_fast < sma_slow] = -1
        
        # Convert to trade signals (only on crossover)
        trade_signals = signals.diff()
        trade_signals = trade_signals.fillna(0)
        trade_signals[trade_signals > 0] = 1
        trade_signals[trade_signals < 0] = -1
        
        return trade_signals
    
    @staticmethod
    def rsi_strategy(prices: pd.DataFrame, period: int = 14, oversold: int = 30, overbought: int = 70) -> pd.Series:
        """RSI mean reversion strategy."""
        close = prices['close']
        delta = close.diff()
        
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        signals = pd.Series(0, index=prices.index)
        signals[rsi < oversold] = 1  # Buy oversold
        signals[rsi > overbought] = -1  # Sell overbought
        
        return signals
    
    @staticmethod
    def momentum_strategy(prices: pd.DataFrame, lookback: int = 20) -> pd.Series:
        """Momentum strategy - buy winners, sell losers."""
        close = prices['close']
        returns = close.pct_change(lookback)
        
        signals = pd.Series(0, index=prices.index)
        signals[returns > 0.05] = 1  # Strong positive momentum
        signals[returns < -0.05] = -1  # Strong negative momentum
        
        return signals
    
    @staticmethod
    def mean_reversion(prices: pd.DataFrame, window: int = 20, threshold: float = 2.0) -> pd.Series:
        """Mean reversion using Bollinger Bands."""
        close = prices['close']
        sma = close.rolling(window).mean()
        std = close.rolling(window).std()
        
        upper = sma + threshold * std
        lower = sma - threshold * std
        
        signals = pd.Series(0, index=prices.index)
        signals[close < lower] = 1  # Buy below lower band
        signals[close > upper] = -1  # Sell above upper band
        
        return signals
    
    @staticmethod
    def random_strategy(prices: pd.DataFrame, seed: int = 42) -> pd.Series:
        """Random trading strategy (for baseline comparison)."""
        np.random.seed(seed)
        signals = pd.Series(
            np.random.choice([-1, 0, 1], size=len(prices), p=[0.1, 0.8, 0.1]),
            index=prices.index
        )
        return signals


@dataclass
class BenchmarkComparison:
    """Comparison of multiple strategies."""
    strategies: List[StrategyResult]
    best_strategy: str
    ranking: List[Tuple[str, float]]  # (name, sharpe)
    statistical_tests: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'strategies': [s.to_dict() for s in self.strategies],
            'best_strategy': self.best_strategy,
            'ranking': self.ranking,
            'statistical_tests': self.statistical_tests
        }
    
    def get_summary_df(self) -> pd.DataFrame:
        """Get summary DataFrame for display."""
        data = []
        for s in self.strategies:
            data.append({
                'Strategy': s.name,
                'Total Return': f"{s.metrics.total_return:.2%}",
                'Annual Return': f"{s.metrics.annualized_return:.2%}",
                'Sharpe': f"{s.metrics.sharpe_ratio:.2f}",
                'Sortino': f"{s.metrics.sortino_ratio:.2f}",
                'Max DD': f"{s.metrics.max_drawdown:.2%}",
                'Win Rate': f"{s.metrics.win_rate:.2%}",
                'Trades': s.metrics.total_trades
            })
        
        df = pd.DataFrame(data)
        return df.sort_values('Sharpe', ascending=False, key=lambda x: x.str.replace('%', '').astype(float))


class Benchmarker:
    """
    Main benchmarking engine.
    
    Compares your strategy against standard benchmarks
    and tracks performance over time.
    """
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.backtester = Backtester(initial_capital=initial_capital)
        self.results_history: List[BenchmarkComparison] = []
        self.leaderboard: Dict[str, List[float]] = {}  # strategy -> list of sharpe ratios
    
    def run_benchmark(
        self,
        prices: pd.DataFrame,
        custom_signals: Optional[pd.Series] = None,
        custom_name: str = "Custom Strategy",
        include_benchmarks: List[str] = None
    ) -> BenchmarkComparison:
        """
        Run benchmark comparison.
        
        Args:
            prices: Price data for backtesting
            custom_signals: Your strategy's signals
            custom_name: Name for your strategy
            include_benchmarks: List of benchmark names to include
        
        Returns:
            BenchmarkComparison with all results
        """
        if include_benchmarks is None:
            include_benchmarks = [
                'buy_and_hold', 'sma_crossover', 'rsi_strategy',
                'momentum', 'mean_reversion', 'random'
            ]
        
        results = []
        
        # Run custom strategy if provided
        if custom_signals is not None:
            result = self._run_strategy(
                prices, custom_signals, custom_name, "Your custom strategy"
            )
            results.append(result)
        
        # Run benchmark strategies
        benchmark_configs = {
            'buy_and_hold': (BenchmarkStrategies.buy_and_hold, {}, "Buy and hold baseline"),
            'sma_crossover': (BenchmarkStrategies.sma_crossover, {'fast': 20, 'slow': 50}, "20/50 SMA crossover"),
            'rsi_strategy': (BenchmarkStrategies.rsi_strategy, {'period': 14}, "RSI 30/70 strategy"),
            'momentum': (BenchmarkStrategies.momentum_strategy, {'lookback': 20}, "20-day momentum"),
            'mean_reversion': (BenchmarkStrategies.mean_reversion, {'window': 20}, "Bollinger mean reversion"),
            'random': (BenchmarkStrategies.random_strategy, {'seed': 42}, "Random baseline")
        }
        
        for name in include_benchmarks:
            if name in benchmark_configs:
                func, params, desc = benchmark_configs[name]
                signals = func(prices, **params)
                result = self._run_strategy(prices, signals, name, desc, params)
                results.append(result)
        
        # Create comparison
        comparison = self._create_comparison(results)
        
        # Update leaderboard
        for result in results:
            if result.name not in self.leaderboard:
                self.leaderboard[result.name] = []
            self.leaderboard[result.name].append(result.metrics.sharpe_ratio)
        
        self.results_history.append(comparison)
        
        return comparison
    
    def _run_strategy(
        self,
        prices: pd.DataFrame,
        signals: pd.Series,
        name: str,
        description: str,
        parameters: Dict[str, Any] = None
    ) -> StrategyResult:
        """Run a single strategy and return results."""
        # Align signals with prices
        signals = signals.reindex(prices.index).fillna(0)
        
        # Run backtest
        bt_result = self.backtester.run(prices, signals, symbol="BENCHMARK")
        
        return StrategyResult(
            name=name,
            description=description,
            metrics=bt_result.metrics,
            equity_curve=bt_result.equity_curve,
            parameters=parameters or {}
        )
    
    def _create_comparison(self, results: List[StrategyResult]) -> BenchmarkComparison:
        """Create comparison from results."""
        # Rank by Sharpe ratio
        ranking = sorted(
            [(r.name, r.metrics.sharpe_ratio) for r in results],
            key=lambda x: x[1],
            reverse=True
        )
        
        best_strategy = ranking[0][0] if ranking else ""
        
        # Statistical tests
        statistical_tests = self._run_statistical_tests(results)
        
        return BenchmarkComparison(
            strategies=results,
            best_strategy=best_strategy,
            ranking=ranking,
            statistical_tests=statistical_tests
        )
    
    def _run_statistical_tests(self, results: List[StrategyResult]) -> Dict[str, Any]:
        """Run statistical significance tests."""
        if len(results) < 2:
            return {}
        
        tests = {}
        
        # Compare each strategy to buy and hold
        buy_hold = next((r for r in results if r.name == 'buy_and_hold'), None)
        
        if buy_hold:
            for result in results:
                if result.name != 'buy_and_hold':
                    # Simple t-test approximation using Sharpe ratio
                    diff = result.metrics.sharpe_ratio - buy_hold.metrics.sharpe_ratio
                    n = result.metrics.trading_days
                    
                    # Approximate standard error
                    se = np.sqrt(2 / n) if n > 0 else 1
                    t_stat = diff / se if se > 0 else 0
                    
                    # Approximate p-value (two-tailed)
                    from scipy import stats
                    try:
                        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=max(n-1, 1)))
                    except:
                        p_value = 1.0
                    
                    tests[f"{result.name}_vs_buy_hold"] = {
                        'sharpe_diff': diff,
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'significant_5pct': p_value < 0.05
                    }
        
        return tests
    
    def get_leaderboard(self) -> pd.DataFrame:
        """Get leaderboard of all strategies."""
        data = []
        
        for name, sharpes in self.leaderboard.items():
            data.append({
                'Strategy': name,
                'Avg Sharpe': np.mean(sharpes),
                'Best Sharpe': max(sharpes),
                'Worst Sharpe': min(sharpes),
                'Std Sharpe': np.std(sharpes),
                'Runs': len(sharpes)
            })
        
        df = pd.DataFrame(data)
        return df.sort_values('Avg Sharpe', ascending=False)
    
    def save_results(self, path: Optional[Path] = None) -> Path:
        """Save benchmark results to disk."""
        if path is None:
            path = DATA_DIR / f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        data = {
            'leaderboard': {k: v for k, v in self.leaderboard.items()},
            'history': [c.to_dict() for c in self.results_history],
            'saved_at': datetime.now().isoformat()
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Benchmark results saved to {path}")
        return path
    
    def load_results(self, path: Path) -> None:
        """Load benchmark results from disk."""
        with open(path) as f:
            data = json.load(f)
        
        self.leaderboard = data.get('leaderboard', {})
        logger.info(f"Loaded benchmark results from {path}")


class PerformanceMonitor:
    """
    Monitor strategy performance over time.
    
    Tracks live performance and compares to backtest results.
    """
    
    def __init__(self):
        self.backtest_metrics: Optional[BacktestMetrics] = None
        self.live_trades: List[Dict[str, Any]] = []
        self.live_equity: List[Tuple[datetime, float]] = []
    
    def set_backtest_baseline(self, metrics: BacktestMetrics) -> None:
        """Set backtest metrics as baseline."""
        self.backtest_metrics = metrics
    
    def record_live_trade(
        self,
        symbol: str,
        entry_price: float,
        exit_price: float,
        shares: int,
        entry_time: datetime,
        exit_time: datetime
    ) -> None:
        """Record a live trade."""
        pnl = (exit_price - entry_price) * shares
        pnl_pct = (exit_price / entry_price - 1) * 100
        
        self.live_trades.append({
            'symbol': symbol,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'shares': shares,
            'entry_time': entry_time,
            'exit_time': exit_time,
            'pnl': pnl,
            'pnl_pct': pnl_pct
        })
    
    def record_equity(self, equity: float) -> None:
        """Record current equity."""
        self.live_equity.append((datetime.now(), equity))
    
    def get_live_metrics(self) -> Dict[str, float]:
        """Calculate live trading metrics."""
        if not self.live_trades:
            return {}
        
        pnls = [t['pnl'] for t in self.live_trades]
        
        total_pnl = sum(pnls)
        win_rate = sum(1 for p in pnls if p > 0) / len(pnls)
        avg_win = np.mean([p for p in pnls if p > 0]) if any(p > 0 for p in pnls) else 0
        avg_loss = np.mean([abs(p) for p in pnls if p < 0]) if any(p < 0 for p in pnls) else 0
        
        return {
            'total_pnl': total_pnl,
            'total_trades': len(self.live_trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': sum(p for p in pnls if p > 0) / abs(sum(p for p in pnls if p < 0)) if any(p < 0 for p in pnls) else float('inf')
        }
    
    def compare_to_backtest(self) -> Dict[str, Any]:
        """Compare live performance to backtest."""
        if not self.backtest_metrics:
            return {'error': 'No backtest baseline set'}
        
        live = self.get_live_metrics()
        if not live:
            return {'error': 'No live trades recorded'}
        
        return {
            'backtest_win_rate': self.backtest_metrics.win_rate,
            'live_win_rate': live['win_rate'],
            'win_rate_diff': live['win_rate'] - self.backtest_metrics.win_rate,
            'backtest_profit_factor': self.backtest_metrics.profit_factor,
            'live_profit_factor': live['profit_factor'],
            'is_outperforming': live['win_rate'] >= self.backtest_metrics.win_rate * 0.9
        }


# Global benchmarker instance
benchmarker = Benchmarker()
