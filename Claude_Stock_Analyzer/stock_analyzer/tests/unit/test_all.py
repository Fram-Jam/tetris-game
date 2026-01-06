"""
Unit Tests
===========
Comprehensive tests for all core components.
"""

import unittest
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil

import numpy as np
import pandas as pd

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.config import config, AppConfig, ModelConfig
from core.errors import (
    StockAnalyzerError, DataFetchError, RateLimitError,
    CircuitBreaker, retry_with_backoff, ErrorCode
)
from data.fetchers import (
    CacheManager, RateLimiter, MockDataSource, DataFetcher
)
from features.technical import FeatureEngine, Signal, IndicatorResult
from models.ensemble import (
    BaseModel, RandomForestModel, EnsembleModel,
    create_labels, prepare_training_data, ModelMetrics
)
from backtest.engine import (
    Backtester, RiskManager, Trade, PositionType,
    generate_signals_from_predictions
)
from adaptive.learning import (
    PerformanceTracker, DriftDetector, OnlineLearner,
    AdaptiveLearningManager, PredictionRecord, DriftType
)


class TestConfig(unittest.TestCase):
    """Tests for configuration module."""
    
    def test_config_loads(self):
        """Test that config loads successfully."""
        self.assertIsInstance(config, AppConfig)
        self.assertIsNotNone(config.model)
        self.assertIsNotNone(config.backtest)
    
    def test_model_config_defaults(self):
        """Test model config has sensible defaults."""
        mc = ModelConfig()
        self.assertEqual(mc.train_ratio + mc.validation_ratio + mc.test_ratio, 1.0)
        self.assertGreater(mc.buy_threshold, mc.sell_threshold)
    
    def test_config_validation(self):
        """Test config validation catches errors."""
        errors = config.validate()
        self.assertEqual(len(errors), 0)


class TestErrors(unittest.TestCase):
    """Tests for error handling module."""
    
    def test_stock_analyzer_error(self):
        """Test custom exception."""
        error = StockAnalyzerError(
            message="Test error",
            code=ErrorCode.DATA_FETCH_FAILED,
            details={"source": "test"}
        )
        
        self.assertEqual(error.code, ErrorCode.DATA_FETCH_FAILED)
        self.assertIn("source", error.details)
        
        error_dict = error.to_dict()
        self.assertIn("error", error_dict)
        self.assertIn("code", error_dict)
    
    def test_rate_limit_error(self):
        """Test rate limit error."""
        error = RateLimitError("test_source", retry_after=60)
        self.assertEqual(error.retry_after, 60)
    
    def test_circuit_breaker_closed(self):
        """Test circuit breaker allows requests when closed."""
        cb = CircuitBreaker(failure_threshold=3)
        
        result = cb.call("test", lambda: "success")
        self.assertEqual(result, "success")
    
    def test_circuit_breaker_opens(self):
        """Test circuit breaker opens after failures."""
        cb = CircuitBreaker(failure_threshold=3, expected_exceptions=(ValueError,))
        
        def failing_func():
            raise ValueError("fail")
        
        for _ in range(3):
            with self.assertRaises(ValueError):
                cb.call("test", failing_func)
        
        # Circuit should be open now
        with self.assertRaises(StockAnalyzerError):
            cb.call("test", lambda: "success")
    
    def test_retry_decorator(self):
        """Test retry with backoff."""
        attempts = [0]
        
        @retry_with_backoff(max_retries=2, base_delay=0.01)
        def flaky_func():
            attempts[0] += 1
            if attempts[0] < 3:
                raise ValueError("not yet")
            return "success"
        
        result = flaky_func()
        self.assertEqual(result, "success")
        self.assertEqual(attempts[0], 3)


class TestDataFetchers(unittest.TestCase):
    """Tests for data fetching module."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.cache = CacheManager(Path(self.temp_dir))
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_cache_set_get(self):
        """Test cache stores and retrieves data."""
        self.cache.set("test_key", {"data": [1, 2, 3]})
        
        result = self.cache.get("test_key", ttl_seconds=3600)
        self.assertEqual(result, {"data": [1, 2, 3]})
    
    def test_cache_expiry(self):
        """Test cache expires after TTL."""
        self.cache.set("test_key", "data")
        
        # Should not exist with 0 TTL
        result = self.cache.get("test_key", ttl_seconds=0)
        self.assertIsNone(result)
    
    def test_rate_limiter(self):
        """Test rate limiter controls request rate."""
        limiter = RateLimiter(requests_per_minute=60)
        
        # First request should succeed immediately
        start = datetime.now()
        self.assertTrue(limiter.acquire(timeout=0.1))
        elapsed = (datetime.now() - start).total_seconds()
        self.assertLess(elapsed, 0.1)
    
    def test_mock_data_source(self):
        """Test mock data source returns valid data."""
        source = MockDataSource()
        
        # Test price data
        df = source.fetch_prices("AAPL", "1mo")
        self.assertIsNotNone(df)
        self.assertGreater(len(df), 0)
        self.assertIn('close', df.columns)
        self.assertIn('volume', df.columns)
        
        # Test info
        info = source.fetch_info("AAPL")
        self.assertIsNotNone(info)
        self.assertEqual(info['symbol'], 'AAPL')
        self.assertIn('sector', info)
    
    def test_data_fetcher_with_mock(self):
        """Test data fetcher uses mock source."""
        fetcher = DataFetcher(use_mock=True)
        
        df = fetcher.get_prices("MSFT", "3mo")
        self.assertIsNotNone(df)
        self.assertGreater(len(df), 0)


class TestFeatureEngineering(unittest.TestCase):
    """Tests for feature engineering module."""
    
    def setUp(self):
        """Create sample price data."""
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=100, freq='B')
        
        prices = 100 + np.cumsum(np.random.randn(100) * 0.02)
        
        self.df = pd.DataFrame({
            'open': prices * (1 + np.random.randn(100) * 0.01),
            'high': prices * (1 + np.abs(np.random.randn(100) * 0.02)),
            'low': prices * (1 - np.abs(np.random.randn(100) * 0.02)),
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)
    
    def test_feature_engine_initialization(self):
        """Test feature engine initializes correctly."""
        engine = FeatureEngine(self.df)
        self.assertIsNotNone(engine.df)
        self.assertGreater(len(engine.get_feature_names()), 20)
    
    def test_rsi_calculation(self):
        """Test RSI is calculated correctly."""
        engine = FeatureEngine(self.df)
        
        self.assertIn('rsi', engine.df.columns)
        
        rsi = engine.df['rsi'].dropna()
        self.assertTrue((rsi >= 0).all())
        self.assertTrue((rsi <= 100).all())
    
    def test_rsi_analysis(self):
        """Test RSI analysis returns proper result."""
        engine = FeatureEngine(self.df)
        
        result = engine.get_rsi_analysis()
        self.assertIsInstance(result, IndicatorResult)
        self.assertIsInstance(result.signal, Signal)
        self.assertGreater(len(result.explanation), 0)
    
    def test_macd_calculation(self):
        """Test MACD is calculated."""
        engine = FeatureEngine(self.df)
        
        self.assertIn('macd', engine.df.columns)
        self.assertIn('macd_signal', engine.df.columns)
        self.assertIn('macd_histogram', engine.df.columns)
    
    def test_bollinger_bands(self):
        """Test Bollinger Bands are calculated."""
        engine = FeatureEngine(self.df)
        
        self.assertIn('bb_upper', engine.df.columns)
        self.assertIn('bb_lower', engine.df.columns)
        self.assertIn('bb_middle', engine.df.columns)
        
        # Upper should be > middle > lower
        valid_rows = engine.df.dropna(subset=['bb_upper', 'bb_middle', 'bb_lower'])
        self.assertTrue((valid_rows['bb_upper'] >= valid_rows['bb_middle']).all())
        self.assertTrue((valid_rows['bb_middle'] >= valid_rows['bb_lower']).all())
    
    def test_overall_signal(self):
        """Test overall signal calculation."""
        engine = FeatureEngine(self.df)
        
        signal, confidence, summary = engine.get_overall_signal()
        
        self.assertIsInstance(signal, Signal)
        self.assertGreater(confidence, 0)
        self.assertLess(confidence, 100)
    
    def test_ml_features(self):
        """Test ML features are clean."""
        engine = FeatureEngine(self.df)
        
        ml_features = engine.get_ml_features()
        
        # Should have no NaN
        self.assertFalse(ml_features.isna().any().any())
        
        # Should have multiple columns
        self.assertGreater(len(ml_features.columns), 10)


class TestModels(unittest.TestCase):
    """Tests for ML models module."""
    
    def setUp(self):
        """Create sample training data."""
        np.random.seed(42)
        n_samples = 500  # Increased for CV folds
        
        self.X = pd.DataFrame({
            f'feature_{i}': np.random.randn(n_samples) 
            for i in range(10)
        })
        
        # Create labels correlated with features
        score = self.X['feature_0'] + self.X['feature_1'] * 0.5
        self.y = (score > score.median()).astype(int)
    
    def test_random_forest_model(self):
        """Test Random Forest model trains and predicts."""
        model = RandomForestModel()
        
        # Split data
        X_train, X_test = self.X[:400], self.X[400:]
        y_train, y_test = self.y[:400], self.y[400:]
        
        # Train
        model.fit(X_train, y_train)
        self.assertTrue(model.is_fitted)
        
        # Predict
        predictions = model.predict(X_test)
        self.assertEqual(len(predictions), len(X_test))
        
        # Predict proba
        proba = model.predict_proba(X_test)
        self.assertEqual(proba.shape[0], len(X_test))
        self.assertTrue((proba >= 0).all())
        self.assertTrue((proba <= 1).all())
    
    def test_ensemble_model(self):
        """Test Ensemble model trains and predicts."""
        model = EnsembleModel()
        
        # Split data
        X_train, X_test = self.X[:400], self.X[400:]
        y_train, y_test = self.y[:400], self.y[400:]
        
        # Train
        model.fit(X_train, y_train)
        self.assertTrue(model.is_fitted)
        
        # Predict
        predictions = model.predict(X_test)
        self.assertEqual(len(predictions), len(X_test))
    
    def test_model_evaluation(self):
        """Test model evaluation returns metrics."""
        model = RandomForestModel()
        
        X_train, X_test = self.X[:400], self.X[400:]
        y_train, y_test = self.y[:400], self.y[400:]
        
        model.fit(X_train, y_train)
        metrics = model.evaluate(X_test, y_test)
        
        self.assertIsInstance(metrics, ModelMetrics)
        self.assertGreater(metrics.accuracy, 0)
        self.assertLessEqual(metrics.accuracy, 1)
    
    def test_feature_importance(self):
        """Test feature importance extraction."""
        model = RandomForestModel()
        model.fit(self.X[:400], self.y[:400])
        
        importance = model.get_feature_importance()
        
        self.assertGreater(len(importance), 0)
        self.assertTrue(all(v >= 0 for v in importance.values()))
    
    def test_create_labels(self):
        """Test label creation from prices."""
        dates = pd.date_range(end=datetime.now(), periods=100, freq='B')
        prices = pd.DataFrame({
            'close': 100 + np.cumsum(np.random.randn(100) * 0.02)
        }, index=dates)
        
        labels = create_labels(prices, lookahead=5, threshold=0.02)
        
        self.assertEqual(len(labels), len(prices))
        self.assertTrue(set(labels.dropna().unique()).issubset({0, 1}))


class TestBacktest(unittest.TestCase):
    """Tests for backtesting module."""
    
    def setUp(self):
        """Create sample price data and signals."""
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=100, freq='B')
        
        prices = 100 + np.cumsum(np.random.randn(100) * 0.02)
        
        self.prices = pd.DataFrame({
            'open': prices * 0.999,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)
        
        # Create signals (buy when price drops, sell when rises)
        returns = pd.Series(prices, index=dates).pct_change()
        self.signals = pd.Series(0, index=dates)
        self.signals.loc[returns < -0.01] = 1  # Buy
        self.signals.loc[returns > 0.01] = -1  # Sell
    
    def test_backtester_initialization(self):
        """Test backtester initializes correctly."""
        bt = Backtester(initial_capital=100000)
        self.assertEqual(bt.initial_capital, 100000)
        self.assertEqual(bt.capital, 100000)
    
    def test_backtester_run(self):
        """Test backtest runs successfully."""
        bt = Backtester(initial_capital=100000)
        
        result = bt.run(self.prices, self.signals, symbol="TEST")
        
        self.assertIsNotNone(result.metrics)
        self.assertIsNotNone(result.equity_curve)
        self.assertGreater(len(result.equity_curve), 0)
    
    def test_backtest_metrics(self):
        """Test backtest calculates metrics."""
        bt = Backtester(initial_capital=100000)
        result = bt.run(self.prices, self.signals)
        
        metrics = result.metrics
        
        # Check all metrics exist
        self.assertIsNotNone(metrics.total_return)
        self.assertIsNotNone(metrics.sharpe_ratio)
        self.assertIsNotNone(metrics.max_drawdown)
        self.assertGreaterEqual(metrics.max_drawdown, 0)
    
    def test_risk_manager(self):
        """Test risk manager controls."""
        rm = RiskManager(
            max_position_pct=0.1,
            stop_loss_pct=0.05
        )
        
        # Test position sizing
        shares = rm.calculate_position_size(100000, 100)
        self.assertGreater(shares, 0)
        self.assertLessEqual(shares * 100, 100000 * 0.1)
    
    def test_stop_loss(self):
        """Test stop-loss trigger."""
        rm = RiskManager(stop_loss_pct=0.05)
        
        trade = Trade(
            symbol="TEST",
            entry_date=datetime.now(),
            entry_price=100.0,
            position_type=PositionType.LONG,
            shares=100
        )
        
        # Price drops 6% - should trigger stop
        self.assertTrue(rm.check_stop_loss(trade, 94.0))
        
        # Price drops 3% - should not trigger
        self.assertFalse(rm.check_stop_loss(trade, 97.0))
    
    def test_signal_generation(self):
        """Test signal generation from predictions."""
        predictions = pd.Series([0.7, 0.3, 0.5, 0.8, 0.2])
        
        signals = generate_signals_from_predictions(
            predictions,
            buy_threshold=0.6,
            sell_threshold=0.4
        )
        
        self.assertEqual(signals.iloc[0], 1)  # 0.7 > 0.6 = buy
        self.assertEqual(signals.iloc[1], -1)  # 0.3 < 0.4 = sell
        self.assertEqual(signals.iloc[2], 0)  # 0.5 = hold


class TestAdaptiveLearning(unittest.TestCase):
    """Tests for adaptive learning module."""
    
    def test_performance_tracker(self):
        """Test performance tracking."""
        tracker = PerformanceTracker(window_size=50)
        
        # Record predictions
        for i in range(10):
            record = tracker.record_prediction(
                symbol="TEST",
                prediction=0.6 + i * 0.01,
                signal="buy",
                confidence=0.7,
                features={"rsi": 40 + i, "macd": 0.1}
            )
            
            # Record outcome
            tracker.record_outcome(record, actual_return=0.02, pnl=100)
        
        metrics = tracker.get_current_metrics()
        
        self.assertEqual(metrics['n_predictions'], 10)
        self.assertEqual(metrics['accuracy'], 1.0)  # All correct
    
    def test_drift_detector(self):
        """Test drift detection."""
        detector = DriftDetector(threshold=0.1, min_samples=10)
        
        # Add baseline (good performance)
        for _ in range(20):
            detector.update(was_correct=True)
        
        # Add degraded performance - track if any drift detected
        drift_detected = False
        for _ in range(15):
            report = detector.update(was_correct=False)
            if report is not None:
                drift_detected = True
                break
        
        # Should have detected at least one drift
        self.assertTrue(drift_detected)
    
    def test_online_learner(self):
        """Test online learning updates."""
        learner = OnlineLearner(learning_rate=0.1)
        
        # Train on some examples
        for _ in range(100):
            features = {"feature_1": np.random.randn(), "feature_2": np.random.randn()}
            target = 1 if features["feature_1"] > 0 else 0
            learner.partial_fit(features, target)
        
        # Should have learned feature_1 is important
        importance = learner.get_feature_importance()
        self.assertIn("feature_1", importance)
        
        # Make prediction
        proba = learner.predict_proba({"feature_1": 1.0, "feature_2": 0.0})
        self.assertGreater(proba, 0)
        self.assertLess(proba, 1)
    
    def test_adaptive_manager(self):
        """Test adaptive learning manager."""
        manager = AdaptiveLearningManager()
        
        # Record some predictions and outcomes
        for i in range(20):
            record = manager.record_prediction(
                symbol="TEST",
                prediction=0.6,
                signal="buy",
                confidence=0.7,
                features={"rsi": 40, "macd": 0.1}
            )
            
            manager.record_outcome(record, actual_return=0.01, pnl=50)
        
        # Get insights
        insights = manager.get_learning_insights()
        
        self.assertIn('current_metrics', insights)
        self.assertIn('performance_trend', insights)
        self.assertIn('should_retrain', insights)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline."""
    
    def test_full_pipeline(self):
        """Test complete analysis pipeline."""
        # 1. Fetch data
        fetcher = DataFetcher(use_mock=True)
        data = fetcher.get_stock_data("AAPL", period="1y")
        
        self.assertIsNotNone(data)
        self.assertIsNotNone(data.prices)
        self.assertGreater(len(data.prices), 0)
        
        # 2. Generate features
        # Reset index if it's a DatetimeIndex
        prices_df = data.prices.reset_index() if hasattr(data.prices.index, 'date') else data.prices
        engine = FeatureEngine(prices_df)
        features = engine.get_ml_features()
        
        self.assertGreater(len(features), 50)
        
        # 3. Get signal
        signal, confidence, summary = engine.get_overall_signal()
        
        self.assertIsInstance(signal, Signal)
    
    def test_training_and_backtest_pipeline(self):
        """Test training and backtesting pipeline."""
        # 1. Get data
        fetcher = DataFetcher(use_mock=True)
        prices = fetcher.get_prices("MSFT", period="2y")
        
        self.assertIsNotNone(prices)
        
        # 2. Generate features
        engine = FeatureEngine(prices)
        features = engine.get_ml_features()
        
        # 3. Create labels
        labels = create_labels(
            prices.reset_index() if 'date' in prices.columns else prices.assign(close=prices['close']),
            lookahead=5
        )
        
        # 4. Prepare data
        X, y = prepare_training_data(features, prices)
        
        if len(X) > 300:  # Only test if enough data
            # 5. Train model
            model = EnsembleModel()
            
            train_size = int(len(X) * 0.7)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            model.fit(X_train, y_train)
            
            # 6. Generate predictions
            proba = model.predict_proba(X_test)
            predictions = pd.Series(proba[:, 1], index=X_test.index)
            
            # 7. Generate signals
            signals = generate_signals_from_predictions(predictions)
            
            # 8. Run backtest
            bt = Backtester(initial_capital=100000)
            
            # Align prices with signals
            test_prices = prices.loc[signals.index]
            
            result = bt.run(test_prices, signals, symbol="MSFT")
            
            self.assertIsNotNone(result.metrics)
            self.assertGreater(result.metrics.trading_days, 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
