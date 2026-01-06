"""
Core Configuration Module
==========================
Production-grade configuration management with validation,
environment variable support, and type safety.
"""

import os
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
import json

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data_storage"
CACHE_DIR = BASE_DIR / ".cache"
LOG_DIR = BASE_DIR / "logs"
MODEL_DIR = BASE_DIR / "saved_models"

# Ensure directories exist
for dir_path in [DATA_DIR, CACHE_DIR, LOG_DIR, MODEL_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)


class Environment(Enum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass(frozen=True)
class DataSourceConfig:
    """Configuration for data sources."""
    # Primary sources (free)
    use_yfinance: bool = True
    use_fred: bool = True
    use_sec_edgar: bool = True
    
    # News sources
    newsapi_key: Optional[str] = field(
        default_factory=lambda: os.getenv("NEWSAPI_KEY")
    )
    
    # Rate limits (requests per minute)
    yfinance_rate_limit: int = 100
    fred_rate_limit: int = 120
    sec_rate_limit: int = 10
    
    # Cache TTL (seconds)
    price_cache_ttl: int = 300  # 5 minutes
    fundamental_cache_ttl: int = 3600  # 1 hour
    news_cache_ttl: int = 900  # 15 minutes


@dataclass(frozen=True)
class ModelConfig:
    """ML model configuration."""
    # Ensemble components
    use_xgboost: bool = True
    use_random_forest: bool = True
    use_gradient_boosting: bool = True
    
    # Training parameters
    train_ratio: float = 0.7
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Walk-forward settings
    walk_forward_windows: int = 5
    min_train_samples: int = 252  # 1 year of trading days for robust model training
    embargo_pct: float = 0.01  # 1% gap after test periods
    
    # Hyperparameters
    xgb_n_estimators: int = 100
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.1
    rf_n_estimators: int = 100
    rf_max_depth: int = 10
    
    # Feature settings
    lookback_periods: tuple = (5, 10, 20, 50, 100, 200)
    
    # Prediction thresholds
    buy_threshold: float = 0.6
    sell_threshold: float = 0.4
    min_confidence: float = 0.55


@dataclass(frozen=True)
class BacktestConfig:
    """Backtesting configuration."""
    # Transaction costs
    commission_pct: float = 0.001  # 0.1%
    slippage_pct: float = 0.001  # 0.1%
    
    # Position sizing
    max_position_pct: float = 0.1  # Max 10% per position
    initial_capital: float = 100000.0
    
    # Risk management
    stop_loss_pct: float = 0.05  # 5%
    take_profit_pct: float = 0.15  # 15%
    max_drawdown_limit: float = 0.2  # 20%
    
    # Kill switch thresholds
    daily_loss_limit_pct: float = 0.05  # 5%
    max_consecutive_losses: int = 10


@dataclass(frozen=True)
class AdaptiveConfig:
    """Adaptive learning configuration."""
    # Online learning
    enable_online_learning: bool = True
    learning_rate: float = 0.01
    
    # Drift detection
    enable_drift_detection: bool = True
    drift_threshold: float = 0.05
    drift_window_size: int = 100
    
    # Retraining triggers
    max_model_age_days: int = 30
    min_new_samples: int = 50
    performance_degradation_threshold: float = 0.1


@dataclass
class AppConfig:
    """Main application configuration."""
    # Environment
    env: Environment = field(
        default_factory=lambda: Environment(
            os.getenv("APP_ENV", "development")
        )
    )
    debug: bool = field(
        default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true"
    )
    
    # Logging
    log_level: LogLevel = field(
        default_factory=lambda: LogLevel(os.getenv("LOG_LEVEL", "INFO"))
    )
    log_file: Path = LOG_DIR / "app.log"
    
    # Sub-configurations
    data: DataSourceConfig = field(default_factory=DataSourceConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    adaptive: AdaptiveConfig = field(default_factory=AdaptiveConfig)
    
    # Paths
    base_dir: Path = BASE_DIR
    data_dir: Path = DATA_DIR
    cache_dir: Path = CACHE_DIR
    model_dir: Path = MODEL_DIR
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        # Validate ratios sum to 1
        ratio_sum = self.model.train_ratio + self.model.validation_ratio + self.model.test_ratio
        if abs(ratio_sum - 1.0) > 0.001:
            errors.append(f"Train/val/test ratios must sum to 1.0, got {ratio_sum}")
        
        # Validate thresholds
        if self.model.buy_threshold <= self.model.sell_threshold:
            errors.append("Buy threshold must be greater than sell threshold")
        
        # Validate percentages
        if not 0 < self.backtest.max_position_pct <= 1:
            errors.append("max_position_pct must be between 0 and 1")
        
        return errors


# Global config instance
config = AppConfig()

# Validate on import
_errors = config.validate()
if _errors:
    raise ValueError(f"Configuration errors: {_errors}")


# Configure logging
def setup_logging() -> logging.Logger:
    """Set up application logging."""
    logger = logging.getLogger("stock_analyzer")
    logger.setLevel(getattr(logging, config.log_level.value))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if config.debug else logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(config.log_file)
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    return logger


logger = setup_logging()


# Disclaimer
DISCLAIMER = """
⚠️ IMPORTANT DISCLAIMER ⚠️

This application is for EDUCATIONAL and RESEARCH purposes only.
It does NOT constitute financial advice.

• Past performance does not guarantee future results
• All investments carry risk of loss
• Consult a qualified financial advisor before trading
• The developers are not responsible for any financial losses

By using this application, you acknowledge these risks.
"""
