"""
Utility Helper Functions
"""

from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Union
import pandas as pd


def format_date(d: Union[date, datetime, str], format: str = "%B %d, %Y") -> str:
    """Format a date for display."""
    if isinstance(d, str):
        d = datetime.fromisoformat(d)
    return d.strftime(format)


def days_ago(days: int) -> date:
    """Get the date N days ago."""
    return date.today() - timedelta(days=days)


def calculate_trend(data: List[float], periods: int = 7) -> str:
    """Calculate trend direction from a list of values."""
    if len(data) < periods * 2:
        return "stable"

    recent = sum(data[-periods:]) / periods
    previous = sum(data[-periods*2:-periods]) / periods

    pct_change = ((recent - previous) / previous) * 100 if previous != 0 else 0

    if pct_change > 5:
        return "up"
    elif pct_change < -5:
        return "down"
    return "stable"


def format_number(value: Union[int, float], decimals: int = 1) -> str:
    """Format a number for display with appropriate precision."""
    if isinstance(value, int) or value == int(value):
        return f"{int(value):,}"
    return f"{value:,.{decimals}f}"


def calculate_average(data: List[Dict], key: str) -> Optional[float]:
    """Calculate average of a key from a list of dicts."""
    values = [d.get(key) for d in data if d.get(key) is not None]
    if not values:
        return None
    return sum(values) / len(values)


def calculate_change(current: float, previous: float) -> Dict:
    """Calculate absolute and percentage change."""
    absolute = current - previous
    percentage = (absolute / previous) * 100 if previous != 0 else 0
    return {
        'absolute': absolute,
        'percentage': percentage,
        'direction': 'up' if absolute > 0 else 'down' if absolute < 0 else 'stable'
    }


def get_score_color(score: int, thresholds: Dict[str, int] = None) -> str:
    """Get color based on score value."""
    if thresholds is None:
        thresholds = {'good': 75, 'warning': 60}

    if score >= thresholds['good']:
        return '#10B981'  # Green
    elif score >= thresholds['warning']:
        return '#F59E0B'  # Yellow/Orange
    return '#EF4444'  # Red


def get_score_emoji(score: int, thresholds: Dict[str, int] = None) -> str:
    """Get emoji based on score value."""
    if thresholds is None:
        thresholds = {'good': 75, 'warning': 60}

    if score >= thresholds['good']:
        return "🟢"
    elif score >= thresholds['warning']:
        return "🟡"
    return "🔴"


def filter_data_by_date_range(
    data: List[Dict],
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    days: Optional[int] = None
) -> List[Dict]:
    """Filter data list by date range."""
    if days is not None:
        return data[-days:] if len(data) > days else data

    filtered = data
    if start_date:
        filtered = [d for d in filtered if d.get('date') >= start_date]
    if end_date:
        filtered = [d for d in filtered if d.get('date') <= end_date]

    return filtered


def data_to_dataframe(data: List[Dict]) -> pd.DataFrame:
    """Convert health data to pandas DataFrame."""
    df = pd.DataFrame(data)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    return df


def safe_get(data: Dict, *keys, default=None):
    """Safely get nested dictionary values."""
    result = data
    for key in keys:
        if isinstance(result, dict):
            result = result.get(key, default)
        else:
            return default
    return result if result is not None else default
