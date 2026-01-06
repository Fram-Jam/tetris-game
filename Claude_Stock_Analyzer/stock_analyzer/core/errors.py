"""
Error Handling Module
======================
Production-grade error handling with:
- Custom exception hierarchy
- Error codes for debugging
- Retry logic with exponential backoff
- Circuit breaker pattern
"""

import time
import logging
import functools
import threading
from enum import Enum
from typing import Callable, Optional, Type, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import traceback

logger = logging.getLogger("stock_analyzer.errors")


class ErrorCode(Enum):
    """Error codes for debugging and monitoring."""
    # Data errors (1xxx)
    DATA_FETCH_FAILED = 1001
    DATA_PARSE_ERROR = 1002
    DATA_VALIDATION_ERROR = 1003
    DATA_NOT_FOUND = 1004
    RATE_LIMIT_EXCEEDED = 1005
    
    # Model errors (2xxx)
    MODEL_TRAINING_FAILED = 2001
    MODEL_PREDICTION_FAILED = 2002
    MODEL_NOT_FOUND = 2003
    MODEL_VALIDATION_ERROR = 2004
    INSUFFICIENT_DATA = 2005
    
    # Backtest errors (3xxx)
    BACKTEST_FAILED = 3001
    INVALID_STRATEGY = 3002
    RISK_LIMIT_EXCEEDED = 3003
    
    # System errors (4xxx)
    CONFIGURATION_ERROR = 4001
    DATABASE_ERROR = 4002
    CACHE_ERROR = 4003
    
    # External service errors (5xxx)
    API_ERROR = 5001
    TIMEOUT_ERROR = 5002
    AUTHENTICATION_ERROR = 5003


class StockAnalyzerError(Exception):
    """Base exception for all application errors."""
    
    def __init__(
        self,
        message: str,
        code: ErrorCode,
        details: Optional[dict] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}
        self.cause = cause
        self.timestamp = datetime.now()
        
        # Log the error
        logger.error(
            f"[{code.name}] {message}",
            extra={
                "error_code": code.value,
                "details": details,
                "cause": str(cause) if cause else None
            }
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "error": self.code.name,
            "code": self.code.value,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat()
        }


class DataFetchError(StockAnalyzerError):
    """Error fetching data from external source."""
    
    def __init__(
        self,
        message: str,
        source: str,
        symbol: Optional[str] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(
            message=message,
            code=ErrorCode.DATA_FETCH_FAILED,
            details={"source": source, "symbol": symbol},
            cause=cause
        )


class RateLimitError(StockAnalyzerError):
    """Rate limit exceeded."""
    
    def __init__(self, source: str, retry_after: Optional[int] = None):
        super().__init__(
            message=f"Rate limit exceeded for {source}",
            code=ErrorCode.RATE_LIMIT_EXCEEDED,
            details={"source": source, "retry_after": retry_after}
        )
        self.retry_after = retry_after


class ModelError(StockAnalyzerError):
    """Error in ML model operations."""
    
    def __init__(
        self,
        message: str,
        model_name: str,
        operation: str,
        cause: Optional[Exception] = None
    ):
        super().__init__(
            message=message,
            code=ErrorCode.MODEL_PREDICTION_FAILED,
            details={"model": model_name, "operation": operation},
            cause=cause
        )


class InsufficientDataError(StockAnalyzerError):
    """Not enough data for operation."""
    
    def __init__(self, required: int, available: int, context: str):
        super().__init__(
            message=f"Insufficient data for {context}: need {required}, have {available}",
            code=ErrorCode.INSUFFICIENT_DATA,
            details={"required": required, "available": available, "context": context}
        )


class RiskLimitError(StockAnalyzerError):
    """Risk limit exceeded - triggers kill switch."""
    
    def __init__(self, limit_type: str, current_value: float, limit_value: float):
        super().__init__(
            message=f"Risk limit exceeded: {limit_type}",
            code=ErrorCode.RISK_LIMIT_EXCEEDED,
            details={
                "limit_type": limit_type,
                "current_value": current_value,
                "limit_value": limit_value
            }
        )


@dataclass
class CircuitBreakerState:
    """State for circuit breaker pattern."""
    failures: int = 0
    last_failure: Optional[datetime] = None
    state: str = "closed"  # closed, open, half-open
    
    def record_failure(self):
        self.failures += 1
        self.last_failure = datetime.now()
    
    def record_success(self):
        self.failures = 0
        self.state = "closed"
    
    def should_allow_request(self, failure_threshold: int, recovery_timeout: int) -> bool:
        if self.state == "closed":
            return True
        
        if self.state == "open":
            # Check if we should try again
            if self.last_failure and \
               datetime.now() - self.last_failure > timedelta(seconds=recovery_timeout):
                self.state = "half-open"
                return True
            return False
        
        # half-open: allow one request to test
        return True


class CircuitBreaker:
    """
    Circuit breaker pattern implementation.

    Prevents cascading failures by stopping requests to failing services.
    Thread-safe implementation using locks.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exceptions: Tuple[Type[Exception], ...] = (Exception,)
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exceptions = expected_exceptions
        self._states: dict[str, CircuitBreakerState] = {}
        self._lock = threading.Lock()

    def _get_state(self, key: str) -> CircuitBreakerState:
        # Lock is held by caller
        if key not in self._states:
            self._states[key] = CircuitBreakerState()
        return self._states[key]

    def call(self, key: str, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection (thread-safe)."""
        with self._lock:
            state = self._get_state(key)

            if not state.should_allow_request(self.failure_threshold, self.recovery_timeout):
                raise StockAnalyzerError(
                    message=f"Circuit breaker open for {key}",
                    code=ErrorCode.API_ERROR,
                    details={"failures": state.failures, "state": state.state}
                )

        # Execute function outside lock to avoid blocking other keys
        try:
            result = func(*args, **kwargs)
            with self._lock:
                state.record_success()
            return result
        except self.expected_exceptions as e:
            with self._lock:
                state.record_failure()
                if state.failures >= self.failure_threshold:
                    state.state = "open"
                    logger.warning(f"Circuit breaker opened for {key} after {state.failures} failures")
            raise


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
):
    """
    Decorator for retry logic with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        exponential_base: Base for exponential backoff calculation
        exceptions: Tuple of exception types to catch and retry
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        delay = min(
                            base_delay * (exponential_base ** attempt),
                            max_delay
                        )
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"All {max_retries + 1} attempts failed for {func.__name__}"
                        )
            
            raise last_exception
        
        return wrapper
    return decorator


def handle_errors(
    default_return: Any = None,
    log_level: str = "error",
    reraise: bool = False
):
    """
    Decorator for graceful error handling.
    
    Args:
        default_return: Value to return on error
        log_level: Logging level for errors
        reraise: Whether to reraise the exception
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except StockAnalyzerError:
                # Already logged, just handle
                if reraise:
                    raise
                return default_return
            except Exception as e:
                log_func = getattr(logger, log_level)
                log_func(
                    f"Unhandled error in {func.__name__}: {e}\n"
                    f"Traceback: {traceback.format_exc()}"
                )
                if reraise:
                    raise
                return default_return
        
        return wrapper
    return decorator


class ErrorTracker:
    """
    Track errors for monitoring and alerting.
    
    Useful for detecting patterns and triggering alerts.
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.errors: list[dict] = []
    
    def record(self, error: StockAnalyzerError):
        """Record an error."""
        self.errors.append({
            "code": error.code.value,
            "message": error.message,
            "timestamp": error.timestamp
        })
        
        # Keep only recent errors
        if len(self.errors) > self.window_size:
            self.errors = self.errors[-self.window_size:]
    
    def get_error_rate(self, window_minutes: int = 5) -> float:
        """Get error rate in the specified time window."""
        cutoff = datetime.now() - timedelta(minutes=window_minutes)
        recent = [e for e in self.errors if e["timestamp"] > cutoff]
        return len(recent) / window_minutes
    
    def get_error_counts_by_code(self) -> dict[str, int]:
        """Get counts of each error code."""
        counts: dict[str, int] = {}
        for error in self.errors:
            code = error["code"]
            counts[code] = counts.get(code, 0) + 1
        return counts


# Global instances
circuit_breaker = CircuitBreaker()
error_tracker = ErrorTracker()
