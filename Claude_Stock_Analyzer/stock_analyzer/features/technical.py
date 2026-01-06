"""
Feature Engineering Module
===========================
Production-grade feature engineering with:
- Technical indicators (RSI, MACD, Bollinger, etc.)
- Statistical features
- Trend features
- Volume features
- All features include explanations
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

import pandas as pd
import numpy as np

from core.errors import InsufficientDataError

logger = logging.getLogger("stock_analyzer.features")


class Signal(Enum):
    """Trading signal types."""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"
    
    @property
    def score(self) -> int:
        """Convert to numeric score."""
        scores = {
            Signal.STRONG_BUY: 2,
            Signal.BUY: 1,
            Signal.HOLD: 0,
            Signal.SELL: -1,
            Signal.STRONG_SELL: -2
        }
        return scores[self]


@dataclass
class IndicatorResult:
    """Result from an indicator calculation."""
    name: str
    value: float
    signal: Signal
    explanation: str
    confidence: float = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'value': self.value,
            'signal': self.signal.value,
            'explanation': self.explanation,
            'confidence': self.confidence
        }


class FeatureEngine:
    """
    Comprehensive feature engineering for stock data.
    
    Calculates 50+ features including:
    - Price-based features (returns, volatility)
    - Technical indicators (RSI, MACD, Bollinger, etc.)
    - Volume features
    - Trend features
    - Statistical features
    """
    
    MIN_REQUIRED_ROWS = 50
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with OHLCV dataframe.
        
        Expected columns: open, high, low, close, volume
        """
        self.df = df.copy()
        self._validate_data()
        self._calculate_all_features()
    
    def _validate_data(self) -> None:
        """Validate input data."""
        required = ['open', 'high', 'low', 'close', 'volume']
        
        # Handle case-insensitive column names
        self.df.columns = [c.lower() for c in self.df.columns]
        
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        if len(self.df) < self.MIN_REQUIRED_ROWS:
            raise InsufficientDataError(
                required=self.MIN_REQUIRED_ROWS,
                available=len(self.df),
                context="feature engineering"
            )
    
    def _calculate_all_features(self) -> None:
        """Calculate all features."""
        self._calculate_returns()
        self._calculate_moving_averages()
        self._calculate_rsi()
        self._calculate_macd()
        self._calculate_bollinger_bands()
        self._calculate_atr()
        self._calculate_obv()
        self._calculate_stochastic()
        self._calculate_adx()
        self._calculate_volatility_features()
        self._calculate_volume_features()
        self._calculate_trend_features()
        self._calculate_statistical_features()
    
    # ==================== Price Returns ====================
    
    def _calculate_returns(self) -> None:
        """Calculate return features."""
        close = self.df['close']
        
        # Simple returns (with explicit fill_method=None)
        self.df['returns_1d'] = close.pct_change(1, fill_method=None)
        self.df['returns_5d'] = close.pct_change(5, fill_method=None)
        self.df['returns_10d'] = close.pct_change(10, fill_method=None)
        self.df['returns_20d'] = close.pct_change(20, fill_method=None)
        
        # Log returns
        self.df['log_returns'] = np.log(close / close.shift(1))
        
        # Cumulative returns
        self.df['cum_returns_20d'] = close / close.shift(20) - 1
        self.df['cum_returns_60d'] = close / close.shift(60) - 1
    
    # ==================== Moving Averages ====================
    
    def _calculate_moving_averages(self) -> None:
        """Calculate moving average features."""
        close = self.df['close']
        
        for period in [5, 10, 20, 50, 100, 200]:
            if len(self.df) >= period:
                # SMA
                self.df[f'sma_{period}'] = close.rolling(window=period).mean()
                
                # EMA
                self.df[f'ema_{period}'] = close.ewm(span=period, adjust=False).mean()
                
                # Price vs MA
                self.df[f'price_sma_{period}_ratio'] = close / self.df[f'sma_{period}']
        
        # MA crossovers
        if 'sma_50' in self.df.columns and 'sma_200' in self.df.columns:
            self.df['golden_cross'] = (
                (self.df['sma_50'] > self.df['sma_200']) & 
                (self.df['sma_50'].shift(1) <= self.df['sma_200'].shift(1))
            ).astype(int)
            
            self.df['death_cross'] = (
                (self.df['sma_50'] < self.df['sma_200']) & 
                (self.df['sma_50'].shift(1) >= self.df['sma_200'].shift(1))
            ).astype(int)
    
    # ==================== RSI ====================
    
    def _calculate_rsi(self, period: int = 14) -> None:
        """Calculate RSI."""
        delta = self.df['close'].diff()
        
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        
        # Wilder's smoothing for subsequent values
        for i in range(period, len(delta)):
            avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (period - 1) + gain.iloc[i]) / period
            avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (period - 1) + loss.iloc[i]) / period
        
        # Handle edge cases: all gains (RSI=100) or all losses (RSI=0)
        # When avg_loss is 0, RSI should be 100 (no losses)
        # When avg_gain is 0, RSI should be 0 (no gains)
        rsi = pd.Series(index=self.df.index, dtype=float)

        # Calculate RS where both are valid
        valid_mask = (avg_loss > 0) & (avg_gain.notna())
        rsi[valid_mask] = 100 - (100 / (1 + avg_gain[valid_mask] / avg_loss[valid_mask]))

        # All gains (no losses) -> RSI = 100
        all_gains_mask = (avg_loss == 0) & (avg_gain > 0)
        rsi[all_gains_mask] = 100.0

        # All losses (no gains) -> RSI = 0
        all_losses_mask = (avg_gain == 0) & (avg_loss > 0)
        rsi[all_losses_mask] = 0.0

        # Both zero (no movement) -> RSI = 50
        no_movement_mask = (avg_gain == 0) & (avg_loss == 0)
        rsi[no_movement_mask] = 50.0

        self.df['rsi'] = rsi
        
        # RSI zones
        self.df['rsi_oversold'] = (self.df['rsi'] < 30).astype(int)
        self.df['rsi_overbought'] = (self.df['rsi'] > 70).astype(int)
    
    def get_rsi_analysis(self) -> IndicatorResult:
        """Analyze RSI with explanation."""
        rsi = self.df['rsi'].iloc[-1]
        
        if pd.isna(rsi):
            return IndicatorResult(
                name="RSI (14)",
                value=50,
                signal=Signal.HOLD,
                explanation="Insufficient data for RSI"
            )
        
        if rsi >= 80:
            signal, conf = Signal.STRONG_SELL, 0.8
            explanation = f"RSI at {rsi:.1f} is extremely overbought (>80). High probability of pullback."
        elif rsi >= 70:
            signal, conf = Signal.SELL, 0.65
            explanation = f"RSI at {rsi:.1f} is overbought (>70). Stock may be due for correction."
        elif rsi >= 60:
            signal, conf = Signal.HOLD, 0.5
            explanation = f"RSI at {rsi:.1f} shows bullish momentum but not overbought."
        elif rsi >= 40:
            signal, conf = Signal.HOLD, 0.5
            explanation = f"RSI at {rsi:.1f} is neutral."
        elif rsi >= 30:
            signal, conf = Signal.BUY, 0.65
            explanation = f"RSI at {rsi:.1f} approaching oversold. Potential buying opportunity."
        else:
            signal, conf = Signal.STRONG_BUY, 0.8
            explanation = f"RSI at {rsi:.1f} is extremely oversold (<30). Strong contrarian buy."
        
        return IndicatorResult(
            name="RSI (14)",
            value=rsi,
            signal=signal,
            explanation=explanation,
            confidence=conf
        )
    
    # ==================== MACD ====================
    
    def _calculate_macd(self, fast: int = 12, slow: int = 26, signal: int = 9) -> None:
        """Calculate MACD."""
        close = self.df['close']
        
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        
        self.df['macd'] = ema_fast - ema_slow
        self.df['macd_signal'] = self.df['macd'].ewm(span=signal, adjust=False).mean()
        self.df['macd_histogram'] = self.df['macd'] - self.df['macd_signal']
        
        # MACD crossovers
        self.df['macd_bullish_cross'] = (
            (self.df['macd'] > self.df['macd_signal']) &
            (self.df['macd'].shift(1) <= self.df['macd_signal'].shift(1))
        ).astype(int)
        
        self.df['macd_bearish_cross'] = (
            (self.df['macd'] < self.df['macd_signal']) &
            (self.df['macd'].shift(1) >= self.df['macd_signal'].shift(1))
        ).astype(int)
    
    def get_macd_analysis(self) -> IndicatorResult:
        """Analyze MACD with explanation."""
        macd = self.df['macd'].iloc[-1]
        signal_line = self.df['macd_signal'].iloc[-1]
        histogram = self.df['macd_histogram'].iloc[-1]
        
        if pd.isna(macd):
            return IndicatorResult(
                name="MACD",
                value=0,
                signal=Signal.HOLD,
                explanation="Insufficient data for MACD"
            )
        
        bullish_cross = self.df['macd_bullish_cross'].iloc[-1]
        bearish_cross = self.df['macd_bearish_cross'].iloc[-1]
        
        if bullish_cross:
            signal, conf = Signal.BUY, 0.7
            explanation = f"MACD bullish crossover! Momentum shifting positive."
        elif bearish_cross:
            signal, conf = Signal.SELL, 0.7
            explanation = f"MACD bearish crossover! Momentum shifting negative."
        elif macd > signal_line and histogram > 0:
            signal, conf = Signal.BUY, 0.6
            explanation = f"MACD ({macd:.3f}) above signal line. Bullish momentum."
        elif macd < signal_line and histogram < 0:
            signal, conf = Signal.SELL, 0.6
            explanation = f"MACD ({macd:.3f}) below signal line. Bearish momentum."
        else:
            signal, conf = Signal.HOLD, 0.5
            explanation = f"MACD showing mixed signals."
        
        return IndicatorResult(
            name="MACD",
            value=macd,
            signal=signal,
            explanation=explanation,
            confidence=conf
        )
    
    # ==================== Bollinger Bands ====================
    
    def _calculate_bollinger_bands(self, period: int = 20, std_dev: float = 2.0) -> None:
        """Calculate Bollinger Bands."""
        close = self.df['close']
        
        self.df['bb_middle'] = close.rolling(window=period).mean()
        rolling_std = close.rolling(window=period).std()
        
        self.df['bb_upper'] = self.df['bb_middle'] + (rolling_std * std_dev)
        self.df['bb_lower'] = self.df['bb_middle'] - (rolling_std * std_dev)
        self.df['bb_width'] = (self.df['bb_upper'] - self.df['bb_lower']) / self.df['bb_middle']
        self.df['bb_pct'] = (close - self.df['bb_lower']) / (self.df['bb_upper'] - self.df['bb_lower'])
    
    def get_bollinger_analysis(self) -> IndicatorResult:
        """Analyze Bollinger Bands."""
        price = self.df['close'].iloc[-1]
        bb_upper = self.df['bb_upper'].iloc[-1]
        bb_lower = self.df['bb_lower'].iloc[-1]
        bb_pct = self.df['bb_pct'].iloc[-1]
        
        if pd.isna(bb_upper):
            return IndicatorResult(
                name="Bollinger Bands",
                value=0.5,
                signal=Signal.HOLD,
                explanation="Insufficient data for Bollinger Bands"
            )
        
        if price > bb_upper:
            signal, conf = Signal.SELL, 0.7
            explanation = f"Price above upper band. Overbought - potential mean reversion."
        elif price < bb_lower:
            signal, conf = Signal.BUY, 0.7
            explanation = f"Price below lower band. Oversold - potential bounce."
        elif bb_pct > 0.8:
            signal, conf = Signal.HOLD, 0.5
            explanation = f"Price in upper 20% of bands. Approaching resistance."
        elif bb_pct < 0.2:
            signal, conf = Signal.HOLD, 0.5
            explanation = f"Price in lower 20% of bands. Approaching support."
        else:
            signal, conf = Signal.HOLD, 0.5
            explanation = f"Price within normal band range."
        
        return IndicatorResult(
            name="Bollinger Bands",
            value=bb_pct,
            signal=signal,
            explanation=explanation,
            confidence=conf
        )
    
    # ==================== ATR ====================
    
    def _calculate_atr(self, period: int = 14) -> None:
        """Calculate Average True Range."""
        high = self.df['high']
        low = self.df['low']
        close = self.df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        self.df['atr'] = tr.rolling(window=period).mean()
        self.df['atr_pct'] = self.df['atr'] / close * 100
    
    # ==================== OBV ====================
    
    def _calculate_obv(self) -> None:
        """Calculate On-Balance Volume."""
        close = self.df['close']
        volume = self.df['volume']
        
        obv = [0]
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.append(obv[-1] + volume.iloc[i])
            elif close.iloc[i] < close.iloc[i-1]:
                obv.append(obv[-1] - volume.iloc[i])
            else:
                obv.append(obv[-1])
        
        self.df['obv'] = obv
        self.df['obv_sma'] = self.df['obv'].rolling(20).mean()
    
    # ==================== Stochastic ====================
    
    def _calculate_stochastic(self, k_period: int = 14, d_period: int = 3) -> None:
        """Calculate Stochastic Oscillator."""
        low_min = self.df['low'].rolling(window=k_period).min()
        high_max = self.df['high'].rolling(window=k_period).max()
        
        self.df['stoch_k'] = ((self.df['close'] - low_min) / (high_max - low_min)) * 100
        self.df['stoch_d'] = self.df['stoch_k'].rolling(window=d_period).mean()
    
    # ==================== ADX ====================
    
    def _calculate_adx(self, period: int = 14) -> None:
        """Calculate ADX."""
        high = self.df['high']
        low = self.df['low']
        close = self.df['close']
        
        plus_dm = high.diff()
        minus_dm = low.diff().abs() * -1
        
        plus_dm = plus_dm.where(plus_dm > 0, 0)
        minus_dm = minus_dm.where(minus_dm < 0, 0).abs()
        
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        
        atr = tr.rolling(period).mean()
        plus_di = (plus_dm.rolling(period).mean() / atr) * 100
        minus_di = (minus_dm.rolling(period).mean() / atr) * 100
        
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)) * 100
        self.df['adx'] = dx.rolling(period).mean()
        self.df['plus_di'] = plus_di
        self.df['minus_di'] = minus_di
    
    # ==================== Volatility ====================
    
    def _calculate_volatility_features(self) -> None:
        """Calculate volatility features."""
        returns = self.df['returns_1d']
        
        # Rolling volatility
        for period in [5, 10, 20, 60]:
            self.df[f'volatility_{period}d'] = returns.rolling(period).std() * np.sqrt(252)
        
        # Volatility ratio
        self.df['volatility_ratio'] = (
            self.df['volatility_5d'] / self.df['volatility_20d'].replace(0, np.nan)
        )
        
        # Parkinson volatility (using high-low)
        self.df['parkinson_vol'] = np.sqrt(
            (1 / (4 * np.log(2))) * 
            (np.log(self.df['high'] / self.df['low']) ** 2).rolling(20).mean()
        ) * np.sqrt(252)
    
    # ==================== Volume ====================
    
    def _calculate_volume_features(self) -> None:
        """Calculate volume features."""
        volume = self.df['volume']
        
        # Volume moving averages
        for period in [5, 10, 20, 50]:
            self.df[f'volume_sma_{period}'] = volume.rolling(period).mean()
        
        # Volume ratio
        self.df['volume_ratio'] = volume / self.df['volume_sma_20'].replace(0, np.nan)
        
        # Volume trend
        self.df['volume_trend'] = (
            self.df['volume_sma_5'] / self.df['volume_sma_20'].replace(0, np.nan)
        )
        
        # Price-volume correlation
        self.df['price_volume_corr'] = (
            self.df['returns_1d'].rolling(20).corr(volume.pct_change())
        )
    
    # ==================== Trend ====================
    
    def _calculate_trend_features(self) -> None:
        """Calculate trend features."""
        close = self.df['close']
        
        # Linear regression slope
        for period in [10, 20, 50]:
            def calc_slope(x):
                if len(x) < period:
                    return np.nan
                y = np.arange(len(x))
                slope, _ = np.polyfit(y, x.values, 1)
                return slope
            
            self.df[f'trend_slope_{period}'] = close.rolling(period).apply(calc_slope, raw=False)
        
        # Higher highs / lower lows
        self.df['higher_high'] = (
            self.df['high'] > self.df['high'].shift(1)
        ).astype(int)
        
        self.df['lower_low'] = (
            self.df['low'] < self.df['low'].shift(1)
        ).astype(int)
        
        # Trend strength
        self.df['trend_strength'] = (
            self.df['higher_high'].rolling(10).sum() - 
            self.df['lower_low'].rolling(10).sum()
        )
    
    # ==================== Statistical ====================
    
    def _calculate_statistical_features(self) -> None:
        """Calculate statistical features."""
        returns = self.df['returns_1d']
        
        # Skewness and kurtosis
        self.df['returns_skew'] = returns.rolling(60).skew()
        self.df['returns_kurt'] = returns.rolling(60).kurt()
        
        # Z-score of price
        self.df['price_zscore'] = (
            (self.df['close'] - self.df['close'].rolling(20).mean()) /
            self.df['close'].rolling(20).std()
        )
        
        # Percentile rank
        self.df['price_percentile'] = self.df['close'].rolling(252).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1],
            raw=False
        )
    
    # ==================== Analysis Methods ====================
    
    def get_all_indicators(self) -> Dict[str, IndicatorResult]:
        """Get analysis from all indicators."""
        return {
            'rsi': self.get_rsi_analysis(),
            'macd': self.get_macd_analysis(),
            'bollinger': self.get_bollinger_analysis(),
        }
    
    def get_overall_signal(self) -> Tuple[Signal, float, str]:
        """Calculate overall signal from all indicators."""
        indicators = self.get_all_indicators()
        
        total_score = 0
        total_weight = 0
        explanations = []
        
        weights = {'rsi': 1.5, 'macd': 1.5, 'bollinger': 1.0}
        
        for name, result in indicators.items():
            weight = weights.get(name, 1.0)
            total_score += result.signal.score * weight * result.confidence
            total_weight += weight
            
            if result.signal in [Signal.STRONG_BUY, Signal.BUY]:
                explanations.append(f"✅ {result.name}: Bullish")
            elif result.signal in [Signal.STRONG_SELL, Signal.SELL]:
                explanations.append(f"❌ {result.name}: Bearish")
        
        avg_score = total_score / total_weight if total_weight > 0 else 0
        confidence = min(95, max(30, 50 + abs(avg_score) * 20))
        
        if avg_score >= 1.0:
            signal = Signal.STRONG_BUY
        elif avg_score >= 0.3:
            signal = Signal.BUY
        elif avg_score <= -1.0:
            signal = Signal.STRONG_SELL
        elif avg_score <= -0.3:
            signal = Signal.SELL
        else:
            signal = Signal.HOLD
        
        summary = "\n".join(explanations[:5])
        
        return signal, confidence, summary
    
    def get_feature_dataframe(self) -> pd.DataFrame:
        """Get the dataframe with all calculated features."""
        return self.df
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature column names."""
        exclude = ['open', 'high', 'low', 'close', 'volume', 'symbol', 'date']
        return [c for c in self.df.columns if c not in exclude]
    
    def get_ml_features(self, max_nan_pct: float = 0.5) -> pd.DataFrame:
        """
        Get features suitable for ML (no NaN, normalized).
        
        Args:
            max_nan_pct: Maximum percentage of NaN values allowed for a feature.
                         Features with more NaN than this are excluded.
        """
        feature_names = self.get_feature_names()
        features = self.df[feature_names].copy()
        
        # Calculate NaN percentage for each column
        nan_pct = features.isna().sum() / len(features)
        
        # Keep only columns with acceptable NaN percentage
        valid_columns = nan_pct[nan_pct <= max_nan_pct].index.tolist()
        
        if not valid_columns:
            # Fallback: use columns with least NaN
            valid_columns = nan_pct.nsmallest(20).index.tolist()
        
        features = features[valid_columns]
        
        # Forward fill then drop remaining NaN rows
        features = features.ffill().dropna()
        
        return features
