"""
Adaptive Learning Module
=========================
Production-grade adaptive learning with:
- Online learning (River integration)
- Drift detection
- Performance tracking
- Automatic model retraining triggers
- Learning from mistakes/successes
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Deque
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from core.config import config, DATA_DIR
from core.errors import handle_errors

logger = logging.getLogger("stock_analyzer.adaptive")


class DriftType(Enum):
    """Types of drift that can be detected."""
    NONE = "none"
    GRADUAL = "gradual"
    SUDDEN = "sudden"
    RECURRING = "recurring"


@dataclass
class PredictionRecord:
    """Record of a prediction and its outcome."""
    timestamp: datetime
    symbol: str
    prediction: float  # Probability
    signal: str  # buy, sell, hold
    confidence: float
    features: Dict[str, float]
    
    # Outcome (filled when known)
    actual_return: Optional[float] = None
    was_correct: Optional[bool] = None
    pnl: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'prediction': self.prediction,
            'signal': self.signal,
            'confidence': self.confidence,
            'actual_return': self.actual_return,
            'was_correct': self.was_correct,
            'pnl': self.pnl
        }


@dataclass
class DriftReport:
    """Report on detected drift."""
    drift_type: DriftType
    drift_score: float
    detected_at: datetime
    metrics_before: Dict[str, float]
    metrics_after: Dict[str, float]
    recommendation: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'drift_type': self.drift_type.value,
            'drift_score': self.drift_score,
            'detected_at': self.detected_at.isoformat(),
            'metrics_before': self.metrics_before,
            'metrics_after': self.metrics_after,
            'recommendation': self.recommendation
        }


class PerformanceTracker:
    """
    Track model performance over time.
    
    Maintains rolling windows of predictions and outcomes
    for performance monitoring.
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.predictions: Deque[PredictionRecord] = deque(maxlen=window_size)
        self.accuracy_history: Deque[float] = deque(maxlen=1000)
        self.pnl_history: Deque[float] = deque(maxlen=1000)
        
        # Metrics cache
        self._metrics_cache: Optional[Dict[str, float]] = None
        self._cache_valid = False
    
    def record_prediction(
        self,
        symbol: str,
        prediction: float,
        signal: str,
        confidence: float,
        features: Dict[str, float]
    ) -> PredictionRecord:
        """Record a new prediction."""
        record = PredictionRecord(
            timestamp=datetime.now(),
            symbol=symbol,
            prediction=prediction,
            signal=signal,
            confidence=confidence,
            features=features
        )
        
        self.predictions.append(record)
        self._cache_valid = False
        
        return record
    
    def record_outcome(
        self,
        record: PredictionRecord,
        actual_return: float,
        pnl: float
    ) -> None:
        """Record the outcome of a prediction."""
        record.actual_return = actual_return
        record.pnl = pnl
        
        # Determine if prediction was correct
        if record.signal == "buy":
            record.was_correct = actual_return > 0
        elif record.signal == "sell":
            record.was_correct = actual_return < 0
        else:  # hold
            record.was_correct = abs(actual_return) < 0.02  # Less than 2% move
        
        # Update histories
        self.accuracy_history.append(1.0 if record.was_correct else 0.0)
        self.pnl_history.append(pnl)
        
        self._cache_valid = False
        
        logger.debug(
            f"Outcome recorded: {record.symbol} {record.signal} -> "
            f"return={actual_return:.2%}, correct={record.was_correct}"
        )
    
    def get_current_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        if self._cache_valid and self._metrics_cache:
            return self._metrics_cache
        
        completed = [p for p in self.predictions if p.was_correct is not None]
        
        if not completed:
            return {
                'accuracy': 0.5,
                'win_rate': 0.5,
                'avg_pnl': 0.0,
                'sharpe': 0.0,
                'n_predictions': 0
            }
        
        correct = sum(1 for p in completed if p.was_correct)
        total = len(completed)
        
        pnls = [p.pnl for p in completed if p.pnl is not None]
        
        metrics = {
            'accuracy': correct / total,
            'win_rate': sum(1 for p in pnls if p > 0) / len(pnls) if pnls else 0.5,
            'avg_pnl': np.mean(pnls) if pnls else 0.0,
            'sharpe': np.mean(pnls) / (np.std(pnls) + 1e-10) * np.sqrt(252) if pnls else 0.0,
            'n_predictions': total
        }
        
        self._metrics_cache = metrics
        self._cache_valid = True
        
        return metrics
    
    def get_recent_accuracy(self, n: int = 20) -> float:
        """Get accuracy over last n predictions."""
        if len(self.accuracy_history) < n:
            n = len(self.accuracy_history)
        
        if n == 0:
            return 0.5
        
        recent = list(self.accuracy_history)[-n:]
        return np.mean(recent)
    
    def get_performance_trend(self) -> str:
        """Analyze performance trend."""
        if len(self.accuracy_history) < 20:
            return "insufficient_data"
        
        recent = self.get_recent_accuracy(20)
        historical = np.mean(list(self.accuracy_history)[:-20]) if len(self.accuracy_history) > 40 else recent
        
        diff = recent - historical
        
        if diff > 0.05:
            return "improving"
        elif diff < -0.05:
            return "degrading"
        else:
            return "stable"


class DriftDetector:
    """
    Detect concept drift in model performance.
    
    Implements ADWIN-style adaptive windowing for drift detection.
    """
    
    def __init__(
        self,
        threshold: float = 0.05,
        window_size: int = 100,
        min_samples: int = 30
    ):
        self.threshold = threshold
        self.window_size = window_size
        self.min_samples = min_samples
        
        self.accuracy_window: Deque[float] = deque(maxlen=window_size)
        self.baseline_accuracy: Optional[float] = None
        self.last_drift_check: Optional[datetime] = None
    
    def update(self, was_correct: bool) -> Optional[DriftReport]:
        """
        Update with new prediction result and check for drift.
        
        Returns DriftReport if drift is detected.
        """
        self.accuracy_window.append(1.0 if was_correct else 0.0)
        
        if len(self.accuracy_window) < self.min_samples:
            return None
        
        # Set baseline if not set
        if self.baseline_accuracy is None:
            self.baseline_accuracy = np.mean(list(self.accuracy_window)[:self.min_samples])
        
        # Check for drift
        current_accuracy = np.mean(list(self.accuracy_window)[-self.min_samples:])
        accuracy_change = abs(current_accuracy - self.baseline_accuracy)
        
        if accuracy_change > self.threshold:
            # Determine drift type
            if current_accuracy < self.baseline_accuracy:
                drift_type = DriftType.GRADUAL if accuracy_change < 0.15 else DriftType.SUDDEN
                recommendation = "Model retraining recommended - performance degradation detected"
            else:
                drift_type = DriftType.GRADUAL
                recommendation = "Model improvement detected - consider updating baseline"
            
            report = DriftReport(
                drift_type=drift_type,
                drift_score=accuracy_change,
                detected_at=datetime.now(),
                metrics_before={'accuracy': self.baseline_accuracy},
                metrics_after={'accuracy': current_accuracy},
                recommendation=recommendation
            )
            
            logger.warning(
                f"Drift detected: {drift_type.value}, "
                f"accuracy change: {accuracy_change:.2%}"
            )
            
            # Reset baseline after detection
            self.baseline_accuracy = current_accuracy
            self.last_drift_check = datetime.now()
            
            return report
        
        return None


class OnlineLearner:
    """
    Online learning wrapper for incremental model updates.
    
    Provides a simple interface for online learning without
    requiring River library.
    """
    
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        self.feature_weights: Dict[str, float] = {}
        self.bias = 0.0
        self.n_updates = 0
    
    def partial_fit(self, features: Dict[str, float], target: int) -> None:
        """
        Update model with a single example.
        
        Uses simple stochastic gradient descent for logistic regression.
        """
        # Initialize weights for new features
        for feature in features:
            if feature not in self.feature_weights:
                self.feature_weights[feature] = 0.0
        
        # Compute prediction
        z = self.bias
        for feature, value in features.items():
            z += self.feature_weights.get(feature, 0) * value
        
        prediction = 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # Sigmoid
        
        # Compute gradient
        error = target - prediction
        
        # Update weights
        for feature, value in features.items():
            self.feature_weights[feature] += self.learning_rate * error * value
        
        self.bias += self.learning_rate * error
        self.n_updates += 1
    
    def predict_proba(self, features: Dict[str, float]) -> float:
        """Get probability prediction."""
        z = self.bias
        for feature, value in features.items():
            z += self.feature_weights.get(feature, 0) * value
        
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance (absolute weights)."""
        return {k: abs(v) for k, v in sorted(
            self.feature_weights.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )}


class AdaptiveLearningManager:
    """
    Main adaptive learning manager.
    
    Coordinates:
    - Performance tracking
    - Drift detection
    - Online learning
    - Retraining triggers
    """
    
    def __init__(
        self,
        enable_online_learning: bool = True,
        enable_drift_detection: bool = True
    ):
        self.enable_online_learning = enable_online_learning
        self.enable_drift_detection = enable_drift_detection
        
        # Components
        self.performance_tracker = PerformanceTracker()
        self.drift_detector = DriftDetector(
            threshold=config.adaptive.drift_threshold,
            window_size=config.adaptive.drift_window_size
        )
        self.online_learner = OnlineLearner(
            learning_rate=config.adaptive.learning_rate
        )
        
        # State
        self.last_model_update: Optional[datetime] = None
        self.drift_reports: List[DriftReport] = []
        self.retraining_needed = False
    
    def record_prediction(
        self,
        symbol: str,
        prediction: float,
        signal: str,
        confidence: float,
        features: Dict[str, float]
    ) -> PredictionRecord:
        """Record a new prediction."""
        return self.performance_tracker.record_prediction(
            symbol=symbol,
            prediction=prediction,
            signal=signal,
            confidence=confidence,
            features=features
        )
    
    def record_outcome(
        self,
        record: PredictionRecord,
        actual_return: float,
        pnl: float
    ) -> Optional[DriftReport]:
        """
        Record prediction outcome and check for drift.
        
        Returns DriftReport if drift is detected.
        """
        # Update performance tracker
        self.performance_tracker.record_outcome(record, actual_return, pnl)
        
        # Check for drift
        drift_report = None
        if self.enable_drift_detection and record.was_correct is not None:
            drift_report = self.drift_detector.update(record.was_correct)
            
            if drift_report:
                self.drift_reports.append(drift_report)
                self.retraining_needed = True
        
        # Online learning update
        if self.enable_online_learning:
            target = 1 if record.was_correct else 0
            self.online_learner.partial_fit(record.features, target)
        
        return drift_report
    
    def should_retrain(self) -> Tuple[bool, str]:
        """
        Check if model should be retrained.
        
        Returns (should_retrain, reason)
        """
        reasons = []
        
        # Check if drift was detected
        if self.retraining_needed:
            reasons.append("drift_detected")
        
        # Check model age
        if self.last_model_update:
            age = datetime.now() - self.last_model_update
            if age.days > config.adaptive.max_model_age_days:
                reasons.append(f"model_age_{age.days}_days")
        
        # Check performance degradation
        metrics = self.performance_tracker.get_current_metrics()
        if metrics['accuracy'] < 0.45:  # Below random
            reasons.append(f"low_accuracy_{metrics['accuracy']:.2%}")
        
        # Check trend
        trend = self.performance_tracker.get_performance_trend()
        if trend == "degrading":
            reasons.append("degrading_performance")
        
        if reasons:
            return True, ", ".join(reasons)
        
        return False, ""
    
    def mark_retrained(self) -> None:
        """Mark that model has been retrained."""
        self.last_model_update = datetime.now()
        self.retraining_needed = False
        logger.info("Model marked as retrained")
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights from adaptive learning."""
        metrics = self.performance_tracker.get_current_metrics()
        
        return {
            'current_metrics': metrics,
            'performance_trend': self.performance_tracker.get_performance_trend(),
            'recent_accuracy_20': self.performance_tracker.get_recent_accuracy(20),
            'drift_reports': len(self.drift_reports),
            'last_drift': self.drift_reports[-1].to_dict() if self.drift_reports else None,
            'online_updates': self.online_learner.n_updates,
            'top_features': dict(list(self.online_learner.get_feature_importance().items())[:10]),
            'should_retrain': self.should_retrain()
        }
    
    def save_state(self, path: Optional[Path] = None) -> Path:
        """Save adaptive learning state."""
        if path is None:
            path = DATA_DIR / "adaptive_state.pkl"
        
        state = {
            'predictions': list(self.performance_tracker.predictions),
            'accuracy_history': list(self.performance_tracker.accuracy_history),
            'pnl_history': list(self.performance_tracker.pnl_history),
            'drift_reports': [r.to_dict() for r in self.drift_reports],
            'online_weights': self.online_learner.feature_weights,
            'online_bias': self.online_learner.bias,
            'last_model_update': self.last_model_update,
            'saved_at': datetime.now().isoformat()
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"Adaptive state saved to {path}")
        return path
    
    @classmethod
    def load_state(cls, path: Path) -> 'AdaptiveLearningManager':
        """Load adaptive learning state."""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        manager = cls()
        
        # Restore state
        manager.performance_tracker.predictions = deque(
            state['predictions'],
            maxlen=manager.performance_tracker.window_size
        )
        manager.performance_tracker.accuracy_history = deque(
            state['accuracy_history'],
            maxlen=1000
        )
        manager.performance_tracker.pnl_history = deque(
            state['pnl_history'],
            maxlen=1000
        )
        
        manager.online_learner.feature_weights = state['online_weights']
        manager.online_learner.bias = state['online_bias']
        manager.last_model_update = state['last_model_update']
        
        logger.info(f"Adaptive state loaded from {path}")
        return manager


# Global instance
adaptive_manager = AdaptiveLearningManager(
    enable_online_learning=config.adaptive.enable_online_learning,
    enable_drift_detection=config.adaptive.enable_drift_detection
)
