"""
Backtesting Engine
===================
Production-grade backtesting with:
- Transaction cost modeling (commission + slippage)
- Position sizing (Kelly, fixed fractional)
- Risk management (stop-loss, take-profit)
- Comprehensive metrics (Sharpe, Sortino, Calmar, etc.)
- Benchmark comparison
- Kill switch for risk limits
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings

import numpy as np
import pandas as pd

from core.config import config
from core.errors import RiskLimitError

logger = logging.getLogger("stock_analyzer.backtest")


class PositionType(Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class PositionSizingMethod(Enum):
    """Position sizing methods."""
    FIXED_FRACTIONAL = "fixed_fractional"  # Fixed % of capital
    VOLATILITY_ADJUSTED = "volatility_adjusted"  # Adjust for volatility
    KELLY = "kelly"  # Kelly criterion
    RISK_PARITY = "risk_parity"  # Equal risk per trade


@dataclass
class Trade:
    """Record of a single trade."""
    symbol: str
    entry_date: datetime
    entry_price: float
    position_type: PositionType
    shares: int
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    exit_reason: Optional[str] = None
    
    @property
    def is_open(self) -> bool:
        return self.exit_date is None
    
    def close(self, exit_date: datetime, exit_price: float, reason: str = "signal") -> None:
        """Close the trade."""
        self.exit_date = exit_date
        self.exit_price = exit_price
        self.exit_reason = reason

        if self.position_type == PositionType.LONG:
            self.pnl = (exit_price - self.entry_price) * self.shares
            if self.entry_price > 0:
                self.pnl_pct = (exit_price / self.entry_price - 1) * 100
            else:
                self.pnl_pct = 0.0
        else:  # SHORT
            self.pnl = (self.entry_price - exit_price) * self.shares
            if exit_price > 0:
                self.pnl_pct = (self.entry_price / exit_price - 1) * 100
            else:
                # Stock went to zero - maximum profit for short
                self.pnl_pct = 100.0 if self.entry_price > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'entry_date': self.entry_date.isoformat() if self.entry_date else None,
            'entry_price': self.entry_price,
            'position_type': self.position_type.value,
            'shares': self.shares,
            'exit_date': self.exit_date.isoformat() if self.exit_date else None,
            'exit_price': self.exit_price,
            'pnl': self.pnl,
            'pnl_pct': self.pnl_pct,
            'exit_reason': self.exit_reason
        }


@dataclass
class BacktestMetrics:
    """Comprehensive backtest metrics."""
    # Returns
    total_return: float
    annualized_return: float
    
    # Risk metrics
    volatility: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Trade metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    avg_trade: float
    
    # Benchmark comparison
    benchmark_return: float
    alpha: float
    beta: float
    information_ratio: float
    
    # Time metrics
    start_date: datetime
    end_date: datetime
    trading_days: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'volatility': self.volatility,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'avg_trade': self.avg_trade,
            'benchmark_return': self.benchmark_return,
            'alpha': self.alpha,
            'beta': self.beta,
            'information_ratio': self.information_ratio,
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'trading_days': self.trading_days
        }


@dataclass
class BacktestResult:
    """Complete backtest results."""
    metrics: BacktestMetrics
    equity_curve: pd.Series
    drawdown_curve: pd.Series
    trades: List[Trade]
    daily_returns: pd.Series
    signals: pd.DataFrame
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'metrics': self.metrics.to_dict(),
            'equity_curve': self.equity_curve.to_dict(),
            'trades': [t.to_dict() for t in self.trades],
            'total_trades': len(self.trades)
        }


class RiskManager:
    """
    Risk management with position sizing and limits.

    Implements:
    - Multiple position sizing methods (fixed, volatility-adjusted, Kelly, risk parity)
    - Stop-loss / take-profit
    - Drawdown limits (kill switch)
    - Consecutive loss limits
    """

    def __init__(
        self,
        max_position_pct: float = 0.1,
        stop_loss_pct: float = 0.05,
        take_profit_pct: float = 0.15,
        max_drawdown: float = 0.2,
        max_consecutive_losses: int = 10,
        sizing_method: PositionSizingMethod = PositionSizingMethod.VOLATILITY_ADJUSTED,
        target_volatility: float = 0.02,  # 2% daily target volatility
        risk_per_trade: float = 0.02,  # 2% risk per trade for risk parity
    ):
        self.max_position_pct = max_position_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_drawdown = max_drawdown
        self.max_consecutive_losses = max_consecutive_losses
        self.sizing_method = sizing_method
        self.target_volatility = target_volatility
        self.risk_per_trade = risk_per_trade

        # Track historical performance for Kelly
        self.win_rate: float = 0.5
        self.avg_win: float = 0.0
        self.avg_loss: float = 0.0

        self.consecutive_losses = 0
        self.peak_equity = 0.0
        self.is_killed = False

    def reset(self) -> None:
        """Reset risk manager state."""
        self.consecutive_losses = 0
        self.peak_equity = 0.0
        self.is_killed = False

    def update_kelly_params(self, win_rate: float, avg_win: float, avg_loss: float) -> None:
        """Update parameters used for Kelly criterion calculation."""
        self.win_rate = max(0.01, min(0.99, win_rate))  # Clamp between 1-99%
        self.avg_win = max(0.001, avg_win)
        self.avg_loss = max(0.001, avg_loss)

    def _calculate_kelly_fraction(self) -> float:
        """
        Calculate Kelly criterion fraction.

        Kelly % = W - (1-W)/R
        Where W = win rate, R = win/loss ratio

        Returns a fraction between 0 and max_position_pct.
        """
        if self.avg_loss == 0:
            return self.max_position_pct

        win_loss_ratio = self.avg_win / self.avg_loss
        kelly = self.win_rate - (1 - self.win_rate) / win_loss_ratio

        # Half-Kelly is more conservative and commonly used
        half_kelly = kelly / 2

        # Clamp between 0 and max_position_pct
        return max(0.0, min(self.max_position_pct, half_kelly))

    def calculate_position_size(
        self,
        capital: float,
        price: float,
        volatility: Optional[float] = None,
        win_probability: Optional[float] = None
    ) -> int:
        """
        Calculate position size based on selected sizing method.

        Args:
            capital: Available capital
            price: Current stock price
            volatility: Stock's annualized volatility (required for volatility methods)
            win_probability: Model's predicted win probability (used for Kelly)

        Returns:
            Number of shares to trade
        """
        if self.is_killed:
            return 0

        # Validate inputs - can't calculate position with invalid price or capital
        if price <= 0 or capital <= 0:
            return 0

        if self.sizing_method == PositionSizingMethod.FIXED_FRACTIONAL:
            # Simple fixed percentage of capital
            position_fraction = self.max_position_pct

        elif self.sizing_method == PositionSizingMethod.VOLATILITY_ADJUSTED:
            # Adjust position size inversely to volatility
            # Higher volatility = smaller position
            if volatility is not None and volatility > 0:
                # Convert annual volatility to daily
                daily_vol = volatility / np.sqrt(252)
                # Scale position: if vol is 2x target, position is 0.5x
                vol_scalar = self.target_volatility / max(daily_vol, 0.001)
                position_fraction = self.max_position_pct * min(vol_scalar, 2.0)  # Cap at 2x
            else:
                position_fraction = self.max_position_pct

        elif self.sizing_method == PositionSizingMethod.KELLY:
            # Kelly criterion based on historical win rate
            position_fraction = self._calculate_kelly_fraction()

        elif self.sizing_method == PositionSizingMethod.RISK_PARITY:
            # Size based on fixed risk per trade
            # Position size = (Capital * Risk%) / (Price * Stop Loss%)
            if self.stop_loss_pct > 0:
                risk_amount = capital * self.risk_per_trade
                max_loss_per_share = price * self.stop_loss_pct
                shares = int(risk_amount / max_loss_per_share)
                return max(0, shares)
            else:
                position_fraction = self.max_position_pct
        else:
            position_fraction = self.max_position_pct

        # Calculate shares from position fraction
        max_value = capital * position_fraction
        shares = int(max_value / price)

        return max(0, shares)
    
    def check_stop_loss(self, trade: Trade, current_price: float) -> bool:
        """Check if stop-loss should be triggered."""
        if trade.entry_price <= 0:
            return False  # Can't calculate loss with invalid entry price

        if trade.position_type == PositionType.LONG:
            loss_pct = (trade.entry_price - current_price) / trade.entry_price
        else:
            loss_pct = (current_price - trade.entry_price) / trade.entry_price

        return loss_pct >= self.stop_loss_pct

    def check_take_profit(self, trade: Trade, current_price: float) -> bool:
        """Check if take-profit should be triggered."""
        if trade.entry_price <= 0:
            return False  # Can't calculate profit with invalid entry price

        if trade.position_type == PositionType.LONG:
            gain_pct = (current_price - trade.entry_price) / trade.entry_price
        else:
            gain_pct = (trade.entry_price - current_price) / trade.entry_price

        return gain_pct >= self.take_profit_pct
    
    def update_equity(self, current_equity: float) -> None:
        """Update equity tracking for drawdown calculation."""
        self.peak_equity = max(self.peak_equity, current_equity)
        
        # Check drawdown limit
        if self.peak_equity > 0:
            drawdown = (self.peak_equity - current_equity) / self.peak_equity
            
            if drawdown >= self.max_drawdown:
                self.is_killed = True
                logger.warning(f"KILL SWITCH: Max drawdown {drawdown:.1%} exceeded limit")
                raise RiskLimitError(
                    limit_type="max_drawdown",
                    current_value=drawdown,
                    limit_value=self.max_drawdown
                )
    
    def record_trade_result(self, pnl: float) -> None:
        """Record trade result for consecutive loss tracking."""
        if pnl < 0:
            self.consecutive_losses += 1
            
            if self.consecutive_losses >= self.max_consecutive_losses:
                self.is_killed = True
                logger.warning(f"KILL SWITCH: {self.consecutive_losses} consecutive losses")
                raise RiskLimitError(
                    limit_type="consecutive_losses",
                    current_value=self.consecutive_losses,
                    limit_value=self.max_consecutive_losses
                )
        else:
            self.consecutive_losses = 0


class Backtester:
    """
    Production backtesting engine.
    
    Features:
    - Realistic transaction cost modeling
    - Multiple position sizing methods
    - Risk management integration
    - Comprehensive metrics calculation
    - Benchmark comparison
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission_pct: float = 0.001,
        slippage_pct: float = 0.001,
        risk_manager: Optional[RiskManager] = None
    ):
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.risk_manager = risk_manager or RiskManager(
            max_position_pct=config.backtest.max_position_pct,
            stop_loss_pct=config.backtest.stop_loss_pct,
            take_profit_pct=config.backtest.take_profit_pct,
            max_drawdown=config.backtest.max_drawdown_limit
        )
        
        self._reset()
    
    def _reset(self) -> None:
        """Reset backtester state."""
        self.capital = self.initial_capital
        self.equity = self.initial_capital
        self.position: Optional[Trade] = None
        self.trades: List[Trade] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.daily_returns: List[float] = []
        self.risk_manager.reset()
    
    def _apply_slippage(self, price: float, is_buy: bool) -> float:
        """Apply slippage to execution price."""
        if is_buy:
            return price * (1 + self.slippage_pct)
        else:
            return price * (1 - self.slippage_pct)
    
    def _apply_commission(self, trade_value: float) -> float:
        """Calculate commission for a trade."""
        return trade_value * self.commission_pct
    
    def _open_position(
        self,
        date: datetime,
        price: float,
        symbol: str,
        position_type: PositionType = PositionType.LONG,
        volatility: Optional[float] = None
    ) -> Optional[Trade]:
        """Open a new position."""
        if self.position is not None:
            return None  # Already have a position

        # Apply slippage
        exec_price = self._apply_slippage(price, position_type == PositionType.LONG)

        # Calculate position size (with volatility for sizing methods that use it)
        shares = self.risk_manager.calculate_position_size(
            self.capital, exec_price, volatility=volatility
        )
        
        if shares <= 0:
            return None
        
        # Apply commission
        trade_value = shares * exec_price
        commission = self._apply_commission(trade_value)
        
        if trade_value + commission > self.capital:
            shares = int((self.capital - commission) / exec_price)
            if shares <= 0:
                return None
            trade_value = shares * exec_price
            commission = self._apply_commission(trade_value)
        
        # Create trade
        trade = Trade(
            symbol=symbol,
            entry_date=date,
            entry_price=exec_price,
            position_type=position_type,
            shares=shares
        )

        # Update capital based on position type
        if position_type == PositionType.LONG:
            # Long: pay for shares + commission
            self.capital -= (trade_value + commission)
        else:
            # Short: receive proceeds from short sale minus commission
            # Capital stays as collateral, we track the short position separately
            self.capital -= commission  # Only pay commission, proceeds tracked via equity

        self.position = trade

        logger.debug(f"Opened {position_type.value} position: {shares} shares @ ${exec_price:.2f}")

        return trade
    
    def _close_position(
        self,
        date: datetime,
        price: float,
        reason: str = "signal"
    ) -> Optional[Trade]:
        """Close current position."""
        if self.position is None:
            return None
        
        # Apply slippage
        is_buy = self.position.position_type == PositionType.SHORT
        exec_price = self._apply_slippage(price, is_buy)
        
        # Close trade
        self.position.close(date, exec_price, reason)

        # Calculate value and commission
        trade_value = self.position.shares * exec_price
        commission = self._apply_commission(trade_value)

        # Update capital based on position type
        if self.position.position_type == PositionType.LONG:
            # Long: receive sale proceeds minus commission
            self.capital += (trade_value - commission)
        else:
            # Short: pay to buy back shares, realize P&L
            # P&L already calculated in trade.close(), add it to capital
            entry_value = self.position.shares * self.position.entry_price
            self.capital += entry_value + (self.position.pnl or 0) - commission
        
        # Record in risk manager
        self.risk_manager.record_trade_result(self.position.pnl or 0)
        
        # Store trade
        closed_trade = self.position
        self.trades.append(closed_trade)
        self.position = None
        
        logger.debug(
            f"Closed position: PnL ${closed_trade.pnl:.2f} ({closed_trade.pnl_pct:.1f}%) - {reason}"
        )
        
        return closed_trade
    
    def run(
        self,
        prices: pd.DataFrame,
        signals: pd.Series,
        symbol: str = "STOCK",
        benchmark: Optional[pd.Series] = None
    ) -> BacktestResult:
        """
        Run backtest with given signals.
        
        Args:
            prices: DataFrame with 'close', 'high', 'low' columns
            signals: Series with 1 (buy), -1 (sell), 0 (hold)
            symbol: Stock symbol
            benchmark: Optional benchmark prices for comparison
        
        Returns:
            BacktestResult with full metrics and trade history
        """
        self._reset()

        if len(prices) != len(signals):
            raise ValueError("Prices and signals must have same length")

        # Ensure proper index
        if not isinstance(prices.index, pd.DatetimeIndex):
            prices.index = pd.to_datetime(prices.index)

        signals = signals.reindex(prices.index)

        # Pre-calculate rolling volatility for position sizing (20-day annualized)
        if 'returns' in prices.columns:
            returns = prices['returns']
        else:
            returns = prices['close'].pct_change()
        rolling_volatility = returns.rolling(window=20).std() * np.sqrt(252)

        prev_equity = self.initial_capital

        for i, (date, row) in enumerate(prices.iterrows()):
            current_price = row['close']
            signal = signals.iloc[i] if i < len(signals) else 0
            
            # Calculate current equity
            if self.position:
                if self.position.position_type == PositionType.LONG:
                    # Long: equity = cash + (shares * current_price)
                    position_value = self.position.shares * current_price
                    self.equity = self.capital + position_value
                else:
                    # Short: equity = cash + initial_proceeds - (shares * current_price)
                    # As price rises, short position loses value
                    initial_value = self.position.shares * self.position.entry_price
                    current_liability = self.position.shares * current_price
                    unrealized_pnl = initial_value - current_liability
                    self.equity = self.capital + initial_value + unrealized_pnl
            else:
                self.equity = self.capital
            
            # Record equity
            self.equity_curve.append((date, self.equity))
            
            # Calculate daily return
            if prev_equity > 0:
                daily_ret = (self.equity - prev_equity) / prev_equity
                self.daily_returns.append(daily_ret)
            
            prev_equity = self.equity
            
            # Update risk manager
            try:
                self.risk_manager.update_equity(self.equity)
            except RiskLimitError:
                # Kill switch triggered - close position and stop
                if self.position:
                    self._close_position(date, current_price, "kill_switch")
                break
            
            # Check stop-loss / take-profit
            if self.position:
                if self.risk_manager.check_stop_loss(self.position, current_price):
                    self._close_position(date, current_price, "stop_loss")
                elif self.risk_manager.check_take_profit(self.position, current_price):
                    self._close_position(date, current_price, "take_profit")
            
            # Get current volatility for position sizing
            current_volatility = rolling_volatility.iloc[i] if i < len(rolling_volatility) else None
            if pd.isna(current_volatility):
                current_volatility = None

            # Process signal
            if signal == 1:  # Buy signal
                if self.position is None:
                    # No position - open long
                    self._open_position(date, current_price, symbol, PositionType.LONG, current_volatility)
                elif self.position.position_type == PositionType.SHORT:
                    # In short position - close it (buy to cover)
                    self._close_position(date, current_price, "signal")
            elif signal == -1:  # Sell signal
                if self.position is None:
                    # No position - open short
                    self._open_position(date, current_price, symbol, PositionType.SHORT, current_volatility)
                elif self.position.position_type == PositionType.LONG:
                    # In long position - close it (sell)
                    self._close_position(date, current_price, "signal")
        
        # Close any remaining position
        if self.position and len(prices) > 0:
            last_date = prices.index[-1]
            last_price = prices['close'].iloc[-1]
            self._close_position(last_date, last_price, "end_of_backtest")
        
        # Calculate metrics
        metrics = self._calculate_metrics(prices, benchmark)
        
        # Create result
        equity_series = pd.Series(
            [e[1] for e in self.equity_curve],
            index=[e[0] for e in self.equity_curve]
        )
        
        drawdown_series = self._calculate_drawdown_series(equity_series)
        
        return BacktestResult(
            metrics=metrics,
            equity_curve=equity_series,
            drawdown_curve=drawdown_series,
            trades=self.trades,
            daily_returns=pd.Series(self.daily_returns),
            signals=pd.DataFrame({'signal': signals})
        )
    
    def _calculate_drawdown_series(self, equity: pd.Series) -> pd.Series:
        """Calculate drawdown series."""
        peak = equity.expanding().max()
        drawdown = (equity - peak) / peak
        return drawdown
    
    def _calculate_metrics(
        self,
        prices: pd.DataFrame,
        benchmark: Optional[pd.Series]
    ) -> BacktestMetrics:
        """Calculate comprehensive backtest metrics."""
        if not self.equity_curve:
            raise ValueError("No equity data to calculate metrics")
        
        equity = pd.Series(
            [e[1] for e in self.equity_curve],
            index=[e[0] for e in self.equity_curve]
        )
        
        returns = pd.Series(self.daily_returns)
        
        # Time metrics
        start_date = self.equity_curve[0][0]
        end_date = self.equity_curve[-1][0]
        trading_days = len(self.equity_curve)
        years = trading_days / 252
        
        # Return metrics
        total_return = (self.equity - self.initial_capital) / self.initial_capital
        annualized_return = (1 + total_return) ** (1 / max(years, 0.01)) - 1 if years > 0 else 0
        
        # Risk metrics
        volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
        
        peak = equity.expanding().max()
        drawdown = (equity - peak) / peak
        max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0
        
        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = (returns.mean() * 252) / (volatility + 1e-10) if volatility > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 1 else 0
        sortino_ratio = (returns.mean() * 252) / (downside_std + 1e-10) if downside_std > 0 else 0
        
        # Calmar ratio
        calmar_ratio = annualized_return / (max_drawdown + 1e-10) if max_drawdown > 0 else 0
        
        # Trade metrics
        total_trades = len(self.trades)
        winning_trades = sum(1 for t in self.trades if (t.pnl or 0) > 0)
        losing_trades = sum(1 for t in self.trades if (t.pnl or 0) < 0)
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        wins = [t.pnl for t in self.trades if (t.pnl or 0) > 0]
        losses = [abs(t.pnl) for t in self.trades if (t.pnl or 0) < 0]
        
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        avg_trade = np.mean([t.pnl or 0 for t in self.trades]) if self.trades else 0
        
        profit_factor = sum(wins) / sum(losses) if sum(losses) > 0 else float('inf')
        
        # Benchmark comparison
        if benchmark is not None and len(benchmark) > 0:
            benchmark_return = (benchmark.iloc[-1] / benchmark.iloc[0] - 1) if len(benchmark) > 1 else 0
            
            # Calculate alpha and beta
            bench_returns = benchmark.pct_change().dropna()
            
            if len(bench_returns) > 1 and len(returns) > 1:
                # Align returns
                min_len = min(len(returns), len(bench_returns))
                strategy_rets = returns.iloc[:min_len].values
                bench_rets = bench_returns.iloc[:min_len].values
                
                # Beta
                covariance = np.cov(strategy_rets, bench_rets)[0, 1]
                bench_variance = np.var(bench_rets)
                beta = covariance / bench_variance if bench_variance > 0 else 0
                
                # Alpha
                alpha = annualized_return - beta * (bench_returns.mean() * 252)
                
                # Information ratio
                tracking_error = (pd.Series(strategy_rets) - pd.Series(bench_rets)).std() * np.sqrt(252)
                information_ratio = (annualized_return - benchmark_return) / (tracking_error + 1e-10)
            else:
                beta = 0
                alpha = annualized_return
                information_ratio = 0
        else:
            benchmark_return = 0
            alpha = annualized_return
            beta = 0
            information_ratio = 0
        
        return BacktestMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_trade=avg_trade,
            benchmark_return=benchmark_return,
            alpha=alpha,
            beta=beta,
            information_ratio=information_ratio,
            start_date=start_date,
            end_date=end_date,
            trading_days=trading_days
        )


def generate_signals_from_predictions(
    predictions: pd.Series,
    buy_threshold: float = 0.6,
    sell_threshold: float = 0.4
) -> pd.Series:
    """
    Convert model predictions to trading signals.
    
    Args:
        predictions: Series of probabilities (0-1)
        buy_threshold: Probability above which to buy
        sell_threshold: Probability below which to sell
    
    Returns:
        Series with 1 (buy), -1 (sell), 0 (hold)
    """
    signals = pd.Series(0, index=predictions.index)
    
    signals[predictions >= buy_threshold] = 1
    signals[predictions <= sell_threshold] = -1
    
    return signals
