"""
Adaptive Threshold Optimizer
=============================
Optimizes buy/sell thresholds per stock based on:
- Historical prediction accuracy at different thresholds
- Risk-adjusted returns
- Market regime detection
"""

import logging
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass
import numpy as np
import pandas as pd

logger = logging.getLogger("stock_analyzer.models.threshold")


@dataclass
class ThresholdResult:
    """Result of threshold optimization."""
    buy_threshold: float
    sell_threshold: float
    expected_accuracy: float
    expected_sharpe: float
    confidence: float  # How confident we are in these thresholds
    regime: str  # Current market regime


class MarketRegimeDetector:
    """
    Detect current market regime for threshold adjustment.

    Regimes:
    - TRENDING_UP: Strong upward momentum
    - TRENDING_DOWN: Strong downward momentum
    - MEAN_REVERTING: Sideways/choppy market
    - HIGH_VOLATILITY: Volatile conditions
    """

    @staticmethod
    def detect(prices: pd.DataFrame, lookback: int = 60) -> str:
        """
        Detect the current market regime.

        Args:
            prices: DataFrame with 'close' column
            lookback: Days to look back for regime detection

        Returns:
            Regime string
        """
        if len(prices) < lookback:
            return "UNKNOWN"

        close = prices['close'].iloc[-lookback:]
        returns = close.pct_change().dropna()

        # Calculate metrics
        cumulative_return = (close.iloc[-1] / close.iloc[0]) - 1
        volatility = returns.std() * np.sqrt(252)
        trend_strength = abs(cumulative_return) / (volatility * np.sqrt(lookback / 252) + 0.001)

        # Mean reversion indicator (Hurst exponent approximation)
        # H < 0.5 = mean reverting, H > 0.5 = trending
        log_returns = np.log(close / close.shift(1)).dropna()
        if len(log_returns) > 10:
            lags = range(2, min(20, len(log_returns) // 2))
            tau = [np.sqrt(np.std(np.subtract(log_returns[lag:].values, log_returns[:-lag].values)))
                   for lag in lags]
            if len(tau) > 1 and all(t > 0 for t in tau):
                reg = np.polyfit(np.log(list(lags)), np.log(tau), 1)
                hurst = reg[0] * 2.0
            else:
                hurst = 0.5
        else:
            hurst = 0.5

        # Classify regime
        if volatility > 0.35:  # >35% annualized volatility
            return "HIGH_VOLATILITY"
        elif trend_strength > 1.5 and cumulative_return > 0:
            return "TRENDING_UP"
        elif trend_strength > 1.5 and cumulative_return < 0:
            return "TRENDING_DOWN"
        elif hurst < 0.45:
            return "MEAN_REVERTING"
        else:
            return "NEUTRAL"


class ThresholdOptimizer:
    """
    Optimize buy/sell thresholds based on historical data.

    Uses grid search over threshold combinations to find
    the pair that maximizes risk-adjusted returns.
    """

    # Default thresholds by regime
    REGIME_DEFAULTS = {
        "TRENDING_UP": (0.55, 0.45),  # More aggressive buying
        "TRENDING_DOWN": (0.65, 0.35),  # More conservative
        "MEAN_REVERTING": (0.60, 0.40),  # Standard
        "HIGH_VOLATILITY": (0.70, 0.30),  # Very conservative
        "NEUTRAL": (0.60, 0.40),
        "UNKNOWN": (0.60, 0.40),
    }

    def __init__(
        self,
        buy_range: Tuple[float, float] = (0.50, 0.80),
        sell_range: Tuple[float, float] = (0.20, 0.50),
        step: float = 0.05
    ):
        """
        Initialize optimizer.

        Args:
            buy_range: (min, max) for buy threshold search
            sell_range: (min, max) for sell threshold search
            step: Step size for grid search
        """
        self.buy_range = buy_range
        self.sell_range = sell_range
        self.step = step
        self.regime_detector = MarketRegimeDetector()

    def optimize(
        self,
        predictions: pd.Series,
        actual_returns: pd.Series,
        prices: Optional[pd.DataFrame] = None,
        min_trades: int = 10
    ) -> ThresholdResult:
        """
        Find optimal thresholds for given predictions and actual returns.

        Args:
            predictions: Model probability predictions (0-1)
            actual_returns: Actual forward returns
            prices: Price data for regime detection
            min_trades: Minimum trades required for valid result

        Returns:
            ThresholdResult with optimized thresholds
        """
        # Detect current regime
        regime = "NEUTRAL"
        if prices is not None:
            regime = self.regime_detector.detect(prices)

        # Align data
        common_idx = predictions.index.intersection(actual_returns.index)
        if len(common_idx) < min_trades:
            # Not enough data - return regime defaults
            buy_th, sell_th = self.REGIME_DEFAULTS[regime]
            return ThresholdResult(
                buy_threshold=buy_th,
                sell_threshold=sell_th,
                expected_accuracy=0.5,
                expected_sharpe=0.0,
                confidence=0.0,
                regime=regime
            )

        pred = predictions.loc[common_idx]
        ret = actual_returns.loc[common_idx]

        # Grid search
        best_sharpe = -np.inf
        best_buy = 0.6
        best_sell = 0.4
        best_accuracy = 0.5

        buy_thresholds = np.arange(self.buy_range[0], self.buy_range[1] + self.step, self.step)
        sell_thresholds = np.arange(self.sell_range[0], self.sell_range[1] + self.step, self.step)

        results: List[Dict] = []

        for buy_th in buy_thresholds:
            for sell_th in sell_thresholds:
                if buy_th <= sell_th:
                    continue  # Invalid: buy must be > sell

                # Generate signals
                signals = pd.Series(0, index=pred.index)
                signals[pred >= buy_th] = 1
                signals[pred <= sell_th] = -1

                # Calculate strategy returns
                strategy_returns = signals.shift(1) * ret  # Shift to avoid lookahead
                strategy_returns = strategy_returns.dropna()

                n_trades = (signals != 0).sum()
                if n_trades < min_trades:
                    continue

                # Calculate metrics
                if len(strategy_returns) > 0 and strategy_returns.std() > 0:
                    sharpe = (strategy_returns.mean() * 252) / (strategy_returns.std() * np.sqrt(252))
                else:
                    sharpe = 0

                # Calculate directional accuracy
                correct = ((signals.shift(1) == 1) & (ret > 0)) | ((signals.shift(1) == -1) & (ret < 0))
                traded = signals.shift(1) != 0
                if traded.sum() > 0:
                    accuracy = correct[traded].mean()
                else:
                    accuracy = 0.5

                results.append({
                    'buy_th': buy_th,
                    'sell_th': sell_th,
                    'sharpe': sharpe,
                    'accuracy': accuracy,
                    'n_trades': n_trades
                })

                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_buy = buy_th
                    best_sell = sell_th
                    best_accuracy = accuracy

        # Calculate confidence based on consistency of results
        if results:
            sharpes = [r['sharpe'] for r in results if r['sharpe'] > 0]
            if sharpes:
                confidence = min(1.0, best_sharpe / (np.std(sharpes) + 0.1))
            else:
                confidence = 0.3
        else:
            confidence = 0.0

        # If optimization failed, use regime defaults
        if best_sharpe <= 0:
            best_buy, best_sell = self.REGIME_DEFAULTS[regime]
            confidence = 0.2

        return ThresholdResult(
            buy_threshold=best_buy,
            sell_threshold=best_sell,
            expected_accuracy=best_accuracy,
            expected_sharpe=max(0, best_sharpe),
            confidence=confidence,
            regime=regime
        )

    def get_regime_adjusted_thresholds(
        self,
        prices: pd.DataFrame,
        base_buy: float = 0.6,
        base_sell: float = 0.4
    ) -> Tuple[float, float]:
        """
        Adjust thresholds based on current market regime.

        Args:
            prices: Price data
            base_buy: Base buy threshold
            base_sell: Base sell threshold

        Returns:
            (adjusted_buy, adjusted_sell) tuple
        """
        regime = self.regime_detector.detect(prices)

        # Regime adjustments
        adjustments = {
            "TRENDING_UP": (-0.05, 0.05),  # Lower buy, raise sell (more aggressive)
            "TRENDING_DOWN": (0.05, -0.05),  # Raise buy, lower sell (more conservative)
            "MEAN_REVERTING": (0.0, 0.0),  # Standard
            "HIGH_VOLATILITY": (0.10, -0.10),  # Much more conservative
            "NEUTRAL": (0.0, 0.0),
            "UNKNOWN": (0.0, 0.0),
        }

        buy_adj, sell_adj = adjustments.get(regime, (0, 0))

        adjusted_buy = min(0.90, max(0.50, base_buy + buy_adj))
        adjusted_sell = max(0.10, min(0.50, base_sell + sell_adj))

        return adjusted_buy, adjusted_sell


def optimize_thresholds_for_stock(
    model,
    X: pd.DataFrame,
    prices: pd.DataFrame,
    lookahead: int = 5,
    validation_split: float = 0.3
) -> ThresholdResult:
    """
    Convenience function to optimize thresholds for a specific stock.

    Uses proper train/validation split to avoid data leakage:
    - Trains threshold optimization on first (1-validation_split) of data
    - Validates on remaining data

    Args:
        model: Trained model with predict_proba method
        X: Feature matrix
        prices: Price data with 'close' column
        lookahead: Days ahead for return calculation
        validation_split: Fraction of data to hold out for validation

    Returns:
        ThresholdResult
    """
    # Get predictions
    proba = model.predict_proba(X)
    predictions = pd.Series(
        proba[:, 1] if proba.shape[1] > 1 else proba[:, 0],
        index=X.index
    )

    # Calculate forward returns
    forward_returns = prices['close'].pct_change(lookahead).shift(-lookahead)

    # Split data chronologically to avoid data leakage
    # Use first portion for threshold optimization, validate on rest
    split_idx = int(len(predictions) * (1 - validation_split))

    if split_idx < 50:  # Not enough training data
        # Fall back to regime-based defaults
        optimizer = ThresholdOptimizer()
        regime = optimizer.regime_detector.detect(prices)
        buy_th, sell_th = optimizer.REGIME_DEFAULTS.get(regime, (0.6, 0.4))
        return ThresholdResult(
            buy_threshold=buy_th,
            sell_threshold=sell_th,
            expected_accuracy=0.5,
            expected_sharpe=0.0,
            confidence=0.0,
            regime=regime
        )

    train_predictions = predictions.iloc[:split_idx]
    train_returns = forward_returns.iloc[:split_idx]
    train_prices = prices.iloc[:split_idx]

    # Optimize on training data only
    optimizer = ThresholdOptimizer()
    result = optimizer.optimize(train_predictions, train_returns, train_prices)

    return result
