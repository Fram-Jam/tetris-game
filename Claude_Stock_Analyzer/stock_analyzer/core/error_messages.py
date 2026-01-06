"""
User-Friendly Error Messages
=============================
Provides clear, actionable error messages for the UI layer.
"""

from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum
import traceback
import logging

logger = logging.getLogger("stock_analyzer.errors")


class ErrorCategory(Enum):
    DATA_FETCH = "data_fetch"
    VALIDATION = "validation"
    MODEL = "model"
    BACKTEST = "backtest"
    NETWORK = "network"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"


@dataclass
class UserError:
    """User-friendly error with context and suggestions."""
    title: str
    message: str
    category: ErrorCategory
    suggestion: str
    technical_details: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'title': self.title,
            'message': self.message,
            'category': self.category.value,
            'suggestion': self.suggestion,
            'technical_details': self.technical_details
        }


def classify_error(exception: Exception, context: Optional[Dict[str, Any]] = None) -> UserError:
    """
    Convert a raw exception into a user-friendly error message.

    Args:
        exception: The caught exception
        context: Optional context dict with keys like 'symbol', 'operation'

    Returns:
        UserError with clear message and suggestions
    """
    context = context or {}
    symbol = context.get('symbol', 'the stock')
    operation = context.get('operation', 'operation')

    error_str = str(exception).lower()
    error_type = type(exception).__name__

    # Get technical details for logging
    technical = f"{error_type}: {exception}\n{traceback.format_exc()}"
    logger.error(f"Error during {operation}: {technical}")

    # Network/Connection errors
    if any(term in error_str for term in ['connection', 'timeout', 'network', 'refused', 'unreachable']):
        return UserError(
            title="Connection Error",
            message=f"Could not connect to the data provider while fetching {symbol}.",
            category=ErrorCategory.NETWORK,
            suggestion="Check your internet connection and try again. If the problem persists, the data provider may be temporarily unavailable.",
            technical_details=technical
        )

    # Rate limiting
    if any(term in error_str for term in ['rate limit', 'too many requests', '429', 'throttl']):
        return UserError(
            title="Rate Limit Exceeded",
            message="Too many requests to the data provider.",
            category=ErrorCategory.NETWORK,
            suggestion="Please wait a few minutes before trying again. Consider analyzing fewer stocks at once.",
            technical_details=technical
        )

    # Invalid symbol
    if any(term in error_str for term in ['no data', 'not found', 'invalid symbol', 'no price', 'empty']):
        return UserError(
            title="Invalid Stock Symbol",
            message=f"Could not find data for '{symbol}'.",
            category=ErrorCategory.DATA_FETCH,
            suggestion=f"Verify that '{symbol}' is a valid stock ticker. Try common symbols like AAPL, MSFT, or GOOGL.",
            technical_details=technical
        )

    # Insufficient data for model training
    if any(term in error_str for term in ['insufficient', 'not enough', 'min_train', 'too few', 'samples']):
        return UserError(
            title="Insufficient Data",
            message=f"Not enough historical data to train the model for {symbol}.",
            category=ErrorCategory.MODEL,
            suggestion="Try selecting a longer time period (1-2 years) or choose a stock with more trading history.",
            technical_details=technical
        )

    # Model training errors
    if any(term in error_str for term in ['fit', 'train', 'converge', 'singular', 'nan', 'inf']):
        return UserError(
            title="Model Training Failed",
            message="The ML model could not be trained on this data.",
            category=ErrorCategory.MODEL,
            suggestion="This may be due to unusual price patterns. Try a different stock or time period.",
            technical_details=technical
        )

    # Risk limit errors
    if any(term in error_str for term in ['drawdown', 'risk limit', 'consecutive loss', 'kill switch']):
        return UserError(
            title="Risk Limit Triggered",
            message="The backtest was stopped due to risk management rules.",
            category=ErrorCategory.BACKTEST,
            suggestion="The strategy hit a risk limit (e.g., max drawdown). This is expected behavior to protect capital. Consider adjusting risk parameters.",
            technical_details=technical
        )

    # Configuration errors
    if any(term in error_str for term in ['config', 'setting', 'parameter', 'invalid value']):
        return UserError(
            title="Configuration Error",
            message="There's an issue with the application settings.",
            category=ErrorCategory.CONFIGURATION,
            suggestion="Try resetting to default settings or check the configuration file.",
            technical_details=technical
        )

    # Import/dependency errors
    if 'import' in error_str or 'module' in error_str:
        return UserError(
            title="Missing Dependency",
            message="A required library is not installed.",
            category=ErrorCategory.CONFIGURATION,
            suggestion="Run 'pip install -r requirements.txt' to install all dependencies.",
            technical_details=technical
        )

    # Default fallback
    return UserError(
        title="Unexpected Error",
        message=f"An error occurred during {operation}.",
        category=ErrorCategory.UNKNOWN,
        suggestion="Try again or select a different stock. If the problem persists, check the logs for details.",
        technical_details=technical
    )


def format_error_for_ui(error: UserError, show_technical: bool = False) -> str:
    """
    Format error for display in Streamlit UI.

    Returns markdown-formatted string.
    """
    parts = [
        f"**{error.title}**",
        "",
        error.message,
        "",
        f"**Suggestion:** {error.suggestion}"
    ]

    if show_technical and error.technical_details:
        parts.extend([
            "",
            "<details><summary>Technical Details</summary>",
            "",
            f"```\n{error.technical_details[:500]}...\n```" if len(error.technical_details) > 500 else f"```\n{error.technical_details}\n```",
            "</details>"
        ])

    return "\n".join(parts)


def get_data_fetch_error_message(symbol: str, source: str = "data provider") -> Tuple[str, str]:
    """
    Get a user-friendly error message for data fetch failures.

    Returns:
        Tuple of (error_message, suggestion)
    """
    return (
        f"Could not fetch data for {symbol} from {source}.",
        "This could be due to:\n"
        "- Invalid stock symbol\n"
        "- Network connectivity issues\n"
        "- Data provider temporarily unavailable\n\n"
        "Try checking the symbol or waiting a moment before retrying."
    )


def get_model_training_error_message(reason: str = "unknown") -> Tuple[str, str]:
    """
    Get a user-friendly error message for model training failures.

    Returns:
        Tuple of (error_message, suggestion)
    """
    reasons = {
        "insufficient_data": (
            "Not enough data to train the model.",
            "Select a longer time period (at least 1 year recommended) or choose a stock with more trading history."
        ),
        "convergence": (
            "The model failed to converge during training.",
            "The price patterns may be too erratic. Try a different stock or time period."
        ),
        "nan_values": (
            "The data contains invalid values.",
            "This stock may have missing data. Try a different symbol or time period."
        ),
    }
    return reasons.get(reason, (
        "Model training failed.",
        "Try a different stock or adjust the time period."
    ))


def get_backtest_error_message(reason: str = "unknown") -> Tuple[str, str]:
    """
    Get a user-friendly error message for backtest failures.

    Returns:
        Tuple of (error_message, suggestion)
    """
    reasons = {
        "max_drawdown": (
            "Backtest stopped: Maximum drawdown limit exceeded.",
            "The strategy lost too much capital. This is a safety feature. Consider:\n"
            "- Adjusting the stop-loss settings\n"
            "- Using a less aggressive strategy\n"
            "- Testing on a different time period"
        ),
        "consecutive_losses": (
            "Backtest stopped: Too many consecutive losing trades.",
            "The strategy had a long losing streak. Consider:\n"
            "- Reviewing the signal generation logic\n"
            "- Adjusting buy/sell thresholds\n"
            "- Testing on different market conditions"
        ),
        "no_trades": (
            "No trades were executed during the backtest.",
            "The strategy signals may be too conservative. Try:\n"
            "- Lowering the buy threshold\n"
            "- Increasing the time period\n"
            "- Checking if the model is producing varied predictions"
        ),
    }
    return reasons.get(reason, (
        "Backtest encountered an issue.",
        "Try adjusting the parameters or selecting a different stock."
    ))
