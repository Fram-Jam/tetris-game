"""
ML Models Module
=================
Production-grade ML models with:
- Ensemble methods (XGBoost + Random Forest + Gradient Boosting)
- Walk-forward validation to prevent overfitting
- Feature importance tracking
- Model persistence and versioning
"""

import logging
import pickle
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings('ignore')

from core.config import config, MODEL_DIR
from core.errors import ModelError, InsufficientDataError

logger = logging.getLogger("stock_analyzer.models")

# Try to import XGBoost
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logger.warning("XGBoost not available, using alternatives")


@dataclass
class ModelMetrics:
    """Metrics for model evaluation."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc_roc: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1': self.f1,
            'auc_roc': self.auc_roc,
            'sharpe_ratio': self.sharpe_ratio
        }


@dataclass
class Prediction:
    """Model prediction with confidence."""
    symbol: str
    signal: str  # buy, sell, hold
    probability: float
    confidence: float
    features_used: List[str]
    model_version: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'signal': self.signal,
            'probability': self.probability,
            'confidence': self.confidence,
            'model_version': self.model_version,
            'timestamp': self.timestamp.isoformat()
        }


class BaseModel(ABC):
    """Abstract base class for ML models."""
    
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.is_fitted = False
        self.version = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metrics: Optional[ModelMetrics] = None
    
    @abstractmethod
    def _create_model(self) -> Any:
        """Create the underlying model."""
        pass
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BaseModel':
        """Train the model."""
        if len(X) < config.model.min_train_samples:
            raise InsufficientDataError(
                required=config.model.min_train_samples,
                available=len(X),
                context=f"training {self.name}"
            )
        
        self.feature_names = list(X.columns)
        self.model = self._create_model()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        logger.info(f"Model {self.name} trained on {len(X)} samples")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ModelError(
                message="Model not fitted",
                model_name=self.name,
                operation="predict"
            )
        
        X_scaled = self.scaler.transform(X[self.feature_names])
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_fitted:
            raise ModelError(
                message="Model not fitted",
                model_name=self.name,
                operation="predict_proba"
            )
        
        X_scaled = self.scaler.transform(X[self.feature_names])
        return self.model.predict_proba(X_scaled)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> ModelMetrics:
        """Evaluate model performance."""
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1] if len(np.unique(y)) == 2 else None
        
        metrics = ModelMetrics(
            accuracy=accuracy_score(y, y_pred),
            precision=precision_score(y, y_pred, average='weighted', zero_division=0),
            recall=recall_score(y, y_pred, average='weighted', zero_division=0),
            f1=f1_score(y, y_pred, average='weighted', zero_division=0),
            auc_roc=roc_auc_score(y, y_proba) if y_proba is not None else None
        )
        
        self.metrics = metrics
        return metrics
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if not self.is_fitted:
            return {}
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_[0])
        else:
            return {}
        
        return dict(zip(self.feature_names, importances))
    
    def save(self, path: Optional[Path] = None) -> Path:
        """Save model to disk."""
        if path is None:
            path = MODEL_DIR / f"{self.name}_{self.version}.pkl"
        
        model_data = {
            'name': self.name,
            'version': self.version,
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'metrics': self.metrics.to_dict() if self.metrics else None,
            'saved_at': datetime.now().isoformat()
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {path}")
        return path
    
    @classmethod
    def load(cls, path: Path) -> 'BaseModel':
        """Load model from disk."""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        instance = cls.__new__(cls)
        instance.name = model_data['name']
        instance.version = model_data['version']
        instance.model = model_data['model']
        instance.scaler = model_data['scaler']
        instance.feature_names = model_data['feature_names']
        instance.is_fitted = True
        
        if model_data['metrics']:
            instance.metrics = ModelMetrics(**model_data['metrics'])
        
        logger.info(f"Model loaded from {path}")
        return instance


class XGBoostModel(BaseModel):
    """XGBoost classifier."""
    
    def __init__(self):
        super().__init__("xgboost")
    
    def _create_model(self) -> Any:
        if HAS_XGBOOST:
            return xgb.XGBClassifier(
                n_estimators=config.model.xgb_n_estimators,
                max_depth=config.model.xgb_max_depth,
                learning_rate=config.model.xgb_learning_rate,
                objective='binary:logistic',
                eval_metric='logloss',
                use_label_encoder=False,
                random_state=42,
                n_jobs=-1
            )
        else:
            # Fallback to GradientBoosting
            return GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )


class RandomForestModel(BaseModel):
    """Random Forest classifier."""
    
    def __init__(self):
        super().__init__("random_forest")
    
    def _create_model(self) -> Any:
        return RandomForestClassifier(
            n_estimators=config.model.rf_n_estimators,
            max_depth=config.model.rf_max_depth,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )


class GradientBoostingModel(BaseModel):
    """Gradient Boosting classifier."""
    
    def __init__(self):
        super().__init__("gradient_boosting")
    
    def _create_model(self) -> Any:
        return GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            min_samples_split=5,
            random_state=42
        )


class EnsembleModel:
    """
    Ensemble model combining multiple base models.
    
    Uses stacking with a meta-learner for final predictions.
    """
    
    def __init__(self):
        self.name = "ensemble"
        self.base_models: List[BaseModel] = []
        self.meta_learner = LogisticRegression(random_state=42, max_iter=1000)
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.version = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metrics: Optional[ModelMetrics] = None
        self.feature_names: List[str] = []
        
        # Initialize base models
        if config.model.use_xgboost:
            self.base_models.append(XGBoostModel())
        if config.model.use_random_forest:
            self.base_models.append(RandomForestModel())
        if config.model.use_gradient_boosting:
            self.base_models.append(GradientBoostingModel())
        
        logger.info(f"Ensemble initialized with {len(self.base_models)} models")
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'EnsembleModel':
        """
        Train ensemble with cross-validation for meta-features.
        
        Uses time series split to respect temporal order.
        """
        if len(X) < config.model.min_train_samples:
            raise InsufficientDataError(
                required=config.model.min_train_samples,
                available=len(X),
                context="training ensemble"
            )
        
        self.feature_names = list(X.columns)
        
        # Determine number of CV splits based on data size
        # Each fold needs at least min_train_samples
        max_splits = max(2, len(X) // config.model.min_train_samples - 1)
        n_splits = min(config.model.walk_forward_windows, max_splits)
        
        # Generate meta-features using time series CV
        tscv = TimeSeriesSplit(n_splits=n_splits)
        meta_features = np.zeros((len(X), len(self.base_models)))
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train = y.iloc[train_idx]
            
            # Skip folds that are too small
            if len(X_train) < config.model.min_train_samples:
                continue
            
            for i, model in enumerate(self.base_models):
                try:
                    # Train on fold
                    model.fit(X_train, y_train)
                    
                    # Generate meta-features for validation set
                    proba = model.predict_proba(X_val)
                    meta_features[val_idx, i] = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
                except InsufficientDataError:
                    # Skip this model for this fold
                    continue
        
        # Train base models on full data
        for model in self.base_models:
            try:
                model.fit(X, y)
            except InsufficientDataError:
                logger.warning(f"Skipping {model.name} due to insufficient data")
        
        # Train meta-learner on non-zero rows (those used in CV)
        valid_mask = meta_features.sum(axis=1) > 0
        if valid_mask.sum() > 0:
            meta_scaled = self.scaler.fit_transform(meta_features[valid_mask])
            self.meta_learner.fit(meta_scaled, y[valid_mask])
        else:
            # Fallback: train meta-learner on all data using base model predictions
            for i, model in enumerate(self.base_models):
                if model.is_fitted:
                    proba = model.predict_proba(X)
                    meta_features[:, i] = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
            
            meta_scaled = self.scaler.fit_transform(meta_features)
            self.meta_learner.fit(meta_scaled, y)
        
        self.is_fitted = True
        logger.info(f"Ensemble trained on {len(X)} samples")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using ensemble."""
        if not self.is_fitted:
            raise ModelError(
                message="Ensemble not fitted",
                model_name=self.name,
                operation="predict"
            )
        
        meta_features = self._get_meta_features(X)
        meta_scaled = self.scaler.transform(meta_features)
        return self.meta_learner.predict(meta_scaled)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_fitted:
            raise ModelError(
                message="Ensemble not fitted",
                model_name=self.name,
                operation="predict_proba"
            )
        
        meta_features = self._get_meta_features(X)
        meta_scaled = self.scaler.transform(meta_features)
        return self.meta_learner.predict_proba(meta_scaled)
    
    def _get_meta_features(self, X: pd.DataFrame) -> np.ndarray:
        """Generate meta-features from base model predictions."""
        meta_features = np.zeros((len(X), len(self.base_models)))
        
        for i, model in enumerate(self.base_models):
            proba = model.predict_proba(X)
            meta_features[:, i] = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
        
        return meta_features
    
    def predict_with_explanation(self, X: pd.DataFrame, symbol: str) -> Prediction:
        """Make prediction with full explanation."""
        proba = self.predict_proba(X)
        
        # Get latest prediction
        latest_proba = proba[-1, 1] if proba.shape[1] > 1 else proba[-1, 0]
        
        # Determine signal
        if latest_proba >= config.model.buy_threshold:
            signal = "buy"
        elif latest_proba <= config.model.sell_threshold:
            signal = "sell"
        else:
            signal = "hold"
        
        # Calculate confidence based on distance from 0.5
        confidence = abs(latest_proba - 0.5) * 2
        
        return Prediction(
            symbol=symbol,
            signal=signal,
            probability=float(latest_proba),
            confidence=float(confidence),
            features_used=self.feature_names,
            model_version=self.version
        )
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> ModelMetrics:
        """Evaluate ensemble performance."""
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1] if self.predict_proba(X).shape[1] > 1 else None
        
        metrics = ModelMetrics(
            accuracy=accuracy_score(y, y_pred),
            precision=precision_score(y, y_pred, average='weighted', zero_division=0),
            recall=recall_score(y, y_pred, average='weighted', zero_division=0),
            f1=f1_score(y, y_pred, average='weighted', zero_division=0),
            auc_roc=roc_auc_score(y, y_proba) if y_proba is not None else None
        )
        
        self.metrics = metrics
        return metrics
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get aggregated feature importance from all base models."""
        all_importances: Dict[str, List[float]] = {}
        
        for model in self.base_models:
            imp = model.get_feature_importance()
            for feature, importance in imp.items():
                if feature not in all_importances:
                    all_importances[feature] = []
                all_importances[feature].append(importance)
        
        # Average across models
        return {
            feature: np.mean(values)
            for feature, values in all_importances.items()
        }
    
    def save(self, path: Optional[Path] = None) -> Path:
        """Save ensemble to disk."""
        if path is None:
            path = MODEL_DIR / f"ensemble_{self.version}.pkl"
        
        model_data = {
            'name': self.name,
            'version': self.version,
            'base_models': self.base_models,
            'meta_learner': self.meta_learner,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'metrics': self.metrics.to_dict() if self.metrics else None,
            'saved_at': datetime.now().isoformat()
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Ensemble saved to {path}")
        return path
    
    @classmethod
    def load(cls, path: Path) -> 'EnsembleModel':
        """Load ensemble from disk."""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        instance = cls.__new__(cls)
        instance.name = model_data['name']
        instance.version = model_data['version']
        instance.base_models = model_data['base_models']
        instance.meta_learner = model_data['meta_learner']
        instance.scaler = model_data['scaler']
        instance.feature_names = model_data['feature_names']
        instance.is_fitted = True
        
        if model_data['metrics']:
            instance.metrics = ModelMetrics(**model_data['metrics'])
        
        logger.info(f"Ensemble loaded from {path}")
        return instance


class WalkForwardValidator:
    """
    Walk-forward validation for time series.
    
    Implements proper temporal validation to prevent lookahead bias.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        train_ratio: float = 0.7,
        embargo_pct: float = 0.01
    ):
        self.n_splits = n_splits
        self.train_ratio = train_ratio
        self.embargo_pct = embargo_pct
    
    def validate(
        self,
        model: Union[BaseModel, EnsembleModel],
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, Any]:
        """
        Perform walk-forward validation.
        
        Returns metrics for each fold and aggregated results.
        """
        n_samples = len(X)
        fold_size = n_samples // self.n_splits
        embargo_size = int(n_samples * self.embargo_pct)
        
        fold_metrics = []
        predictions_all = []
        actuals_all = []
        
        for fold in range(self.n_splits):
            # Calculate split indices
            test_start = fold * fold_size
            test_end = min((fold + 1) * fold_size, n_samples)
            
            # Training set is everything before test, minus embargo
            train_end = max(0, test_start - embargo_size)
            
            if train_end < config.model.min_train_samples:
                continue
            
            X_train = X.iloc[:train_end]
            y_train = y.iloc[:train_end]
            X_test = X.iloc[test_start:test_end]
            y_test = y.iloc[test_start:test_end]
            
            # Clone and train model
            if isinstance(model, EnsembleModel):
                fold_model = EnsembleModel()
            else:
                fold_model = model.__class__()
            
            fold_model.fit(X_train, y_train)
            
            # Evaluate
            metrics = fold_model.evaluate(X_test, y_test)
            fold_metrics.append(metrics.to_dict())
            
            # Store predictions
            predictions_all.extend(fold_model.predict(X_test).tolist())
            actuals_all.extend(y_test.tolist())
            
            logger.debug(f"Fold {fold + 1}/{self.n_splits}: accuracy={metrics.accuracy:.3f}")
        
        # Aggregate metrics
        if not fold_metrics:
            raise InsufficientDataError(
                required=config.model.min_train_samples * 2,
                available=n_samples,
                context="walk-forward validation"
            )
        
        aggregated = {
            'mean_accuracy': np.mean([m['accuracy'] for m in fold_metrics]),
            'std_accuracy': np.std([m['accuracy'] for m in fold_metrics]),
            'mean_f1': np.mean([m['f1'] for m in fold_metrics]),
            'mean_precision': np.mean([m['precision'] for m in fold_metrics]),
            'mean_recall': np.mean([m['recall'] for m in fold_metrics]),
            'n_folds': len(fold_metrics),
            'fold_metrics': fold_metrics
        }
        
        logger.info(
            f"Walk-forward validation: {aggregated['n_folds']} folds, "
            f"mean accuracy: {aggregated['mean_accuracy']:.3f} ± {aggregated['std_accuracy']:.3f}"
        )
        
        return aggregated


def create_labels(
    prices: pd.DataFrame,
    lookahead: int = 5,
    threshold: float = 0.02
) -> pd.Series:
    """
    Create classification labels from price data.
    
    Args:
        prices: DataFrame with 'close' column
        lookahead: Days to look ahead for return calculation
        threshold: Minimum return for buy/sell signal (default 2%)
    
    Returns:
        Series with labels: 1 (buy), 0 (hold/sell)
    """
    future_returns = prices['close'].pct_change(lookahead).shift(-lookahead)
    
    # Binary classification: 1 if return > threshold, else 0
    labels = (future_returns > threshold).astype(int)
    
    return labels


def prepare_training_data(
    features: pd.DataFrame,
    prices: pd.DataFrame,
    lookahead: int = 5
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and labels for training.
    
    Handles alignment and removes NaN values.
    """
    labels = create_labels(prices, lookahead=lookahead)
    
    # Align indices
    common_idx = features.index.intersection(labels.dropna().index)
    
    X = features.loc[common_idx]
    y = labels.loc[common_idx]
    
    # Drop any remaining NaN
    valid_mask = ~(X.isna().any(axis=1) | y.isna())
    
    return X[valid_mask], y[valid_mask]
