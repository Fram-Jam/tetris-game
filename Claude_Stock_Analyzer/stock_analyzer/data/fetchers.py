"""
Data Fetching Layer
====================
Production-grade data fetching with:
- Multiple data sources with fallbacks
- Intelligent caching (disk + memory)
- Rate limiting per source
- Circuit breaker pattern
- Comprehensive error handling
"""

import os
import re
import time
import hashlib
import pickle
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Any, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import json

import pandas as pd
import numpy as np

from core.config import config, DATA_DIR, CACHE_DIR
from core.errors import (
    DataFetchError, RateLimitError, InsufficientDataError,
    retry_with_backoff, handle_errors, circuit_breaker
)


# Stock symbol validation pattern
# Allows 1-10 uppercase letters, optionally followed by a dot and more letters (e.g., BRK.A)
VALID_SYMBOL_PATTERN = re.compile(r'^[A-Z]{1,10}(\.[A-Z]{1,5})?$')


def validate_symbol(symbol: str) -> str:
    """
    Validate and sanitize a stock symbol.

    Args:
        symbol: Stock symbol to validate

    Returns:
        Sanitized uppercase symbol

    Raises:
        ValueError: If symbol is invalid
    """
    if not symbol or not isinstance(symbol, str):
        raise ValueError("Symbol must be a non-empty string")

    # Convert to uppercase and strip whitespace
    clean_symbol = symbol.strip().upper()

    # Check against pattern
    if not VALID_SYMBOL_PATTERN.match(clean_symbol):
        raise ValueError(
            f"Invalid stock symbol '{symbol}'. "
            "Symbols must be 1-10 uppercase letters, optionally with a dot suffix (e.g., AAPL, BRK.A)"
        )

    return clean_symbol

logger = logging.getLogger("stock_analyzer.data")


class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self, requests_per_minute: int):
        # Ensure minimum rate of 1 request per minute to avoid division by zero
        requests_per_minute = max(1, requests_per_minute)
        self.rate = requests_per_minute / 60.0
        self.tokens = float(requests_per_minute)
        self.max_tokens = float(requests_per_minute)
        self.last_update = time.time()

    def acquire(self, timeout: float = 30.0) -> bool:
        """Wait until a request can be made. Returns False if timeout."""
        start = time.time()

        while True:
            now = time.time()
            elapsed = now - self.last_update
            self.tokens = min(self.max_tokens, self.tokens + elapsed * self.rate)
            self.last_update = now

            if self.tokens >= 1:
                self.tokens -= 1
                return True

            if now - start > timeout:
                return False

            # Rate is guaranteed to be > 0 due to constructor validation
            sleep_time = (1 - self.tokens) / self.rate
            time.sleep(min(sleep_time, 0.5))


class CacheManager:
    """Disk + memory cache with TTL."""
    
    def __init__(self, cache_dir: Path = CACHE_DIR, max_memory_items: int = 1000):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        self.max_memory_items = max_memory_items
    
    def _get_cache_path(self, key: str) -> Path:
        # Use SHA256 instead of MD5 for better security (collision resistance)
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.pkl"
    
    def get(self, key: str, ttl_seconds: int) -> Optional[Any]:
        """Get from cache if valid."""
        # Check memory first
        if key in self.memory_cache:
            cached = self.memory_cache[key]
            if datetime.now() - cached['timestamp'] < timedelta(seconds=ttl_seconds):
                logger.debug(f"Memory cache hit: {key[:50]}")
                return cached['data']
            else:
                del self.memory_cache[key]
        
        # Check disk
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    cached = pickle.load(f)
                
                if datetime.now() - cached['timestamp'] < timedelta(seconds=ttl_seconds):
                    # Promote to memory cache
                    self._add_to_memory(key, cached)
                    logger.debug(f"Disk cache hit: {key[:50]}")
                    return cached['data']
                else:
                    cache_path.unlink()
            except Exception as e:
                logger.warning(f"Cache read error: {e}")
        
        return None
    
    def set(self, key: str, data: Any) -> None:
        """Store in cache."""
        cached = {'timestamp': datetime.now(), 'data': data}
        
        # Memory cache
        self._add_to_memory(key, cached)
        
        # Disk cache
        try:
            cache_path = self._get_cache_path(key)
            with open(cache_path, 'wb') as f:
                pickle.dump(cached, f)
        except Exception as e:
            logger.warning(f"Cache write error: {e}")
    
    def _add_to_memory(self, key: str, cached: dict) -> None:
        """Add to memory cache with eviction."""
        if len(self.memory_cache) >= self.max_memory_items:
            # Evict oldest
            oldest_key = min(
                self.memory_cache.keys(),
                key=lambda k: self.memory_cache[k]['timestamp']
            )
            del self.memory_cache[oldest_key]
        
        self.memory_cache[key] = cached
    
    def clear(self) -> None:
        """Clear all caches."""
        self.memory_cache.clear()
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
        logger.info("Cache cleared")


@dataclass
class StockData:
    """Container for stock data."""
    symbol: str
    prices: pd.DataFrame
    info: Dict[str, Any]
    source: str
    fetched_at: datetime = field(default_factory=datetime.now)
    
    def is_valid(self) -> bool:
        """Check if data is valid."""
        return (
            self.prices is not None and 
            len(self.prices) > 0 and
            'close' in self.prices.columns
        )


class DataSource(ABC):
    """Abstract base class for data sources."""
    
    def __init__(self, name: str, rate_limit: int):
        self.name = name
        self.rate_limiter = RateLimiter(rate_limit)
        self.enabled = True
    
    @abstractmethod
    def fetch_prices(self, symbol: str, period: str) -> Optional[pd.DataFrame]:
        """Fetch price data."""
        pass
    
    @abstractmethod
    def fetch_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch stock info."""
        pass
    
    def _acquire_rate_limit(self) -> None:
        """Acquire rate limit or raise error."""
        if not self.rate_limiter.acquire(timeout=30):
            raise RateLimitError(self.name)


class YFinanceSource(DataSource):
    """Yahoo Finance data source."""
    
    def __init__(self):
        super().__init__("yfinance", config.data.yfinance_rate_limit)
    
    @retry_with_backoff(max_retries=3, base_delay=1.0)
    def fetch_prices(self, symbol: str, period: str = "2y") -> Optional[pd.DataFrame]:
        """Fetch historical prices from Yahoo Finance."""
        self._acquire_rate_limit()
        
        try:
            import yfinance as yf
            
            logger.info(f"Fetching {symbol} prices from yfinance")
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period)
            
            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return None
            
            # Standardize columns
            df = df.reset_index()
            df.columns = [c.lower().replace(' ', '_') for c in df.columns]
            
            # Add computed columns
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            df['symbol'] = symbol
            
            logger.info(f"Fetched {len(df)} rows for {symbol}")
            return df
            
        except Exception as e:
            raise DataFetchError(
                f"Failed to fetch prices for {symbol}",
                source=self.name,
                symbol=symbol,
                cause=e
            )
    
    @retry_with_backoff(max_retries=2, base_delay=0.5)
    def fetch_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch stock info from Yahoo Finance."""
        self._acquire_rate_limit()
        
        try:
            import yfinance as yf
            
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'symbol': symbol,
                'name': info.get('longName', symbol),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE'),
                'forward_pe': info.get('forwardPE'),
                'pb_ratio': info.get('priceToBook'),
                'ps_ratio': info.get('priceToSalesTrailing12Months'),
                'dividend_yield': info.get('dividendYield'),
                'beta': info.get('beta'),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh'),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow'),
                'avg_volume': info.get('averageVolume'),
                'revenue_growth': info.get('revenueGrowth'),
                'earnings_growth': info.get('earningsGrowth'),
                'profit_margin': info.get('profitMargins'),
                'debt_to_equity': info.get('debtToEquity'),
                'current_ratio': info.get('currentRatio'),
                'return_on_equity': info.get('returnOnEquity'),
                'description': info.get('longBusinessSummary', '')[:500],
            }
            
        except Exception as e:
            raise DataFetchError(
                f"Failed to fetch info for {symbol}",
                source=self.name,
                symbol=symbol,
                cause=e
            )


class FREDSource(DataSource):
    """Federal Reserve Economic Data source."""
    
    BASE_URL = "https://api.stlouisfed.org/fred"
    
    # Common economic indicators
    INDICATORS = {
        'GDP': 'GDP',
        'UNEMPLOYMENT': 'UNRATE',
        'INFLATION': 'CPIAUCSL',
        'FED_FUNDS': 'FEDFUNDS',
        'TREASURY_10Y': 'DGS10',
        'VIX': 'VIXCLS',
        'SP500': 'SP500',
    }
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__("fred", config.data.fred_rate_limit)
        self.api_key = api_key or os.getenv("FRED_API_KEY")
    
    def fetch_prices(self, symbol: str, period: str = "2y") -> Optional[pd.DataFrame]:
        """FRED doesn't provide stock prices, returns None."""
        return None
    
    def fetch_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """FRED doesn't provide stock info, returns None."""
        return None
    
    @retry_with_backoff(max_retries=2)
    def fetch_series(
        self,
        series_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """Fetch a FRED data series."""
        if not self.api_key:
            logger.warning("FRED API key not configured")
            return None
        
        self._acquire_rate_limit()
        
        try:
            import requests
            
            params = {
                'series_id': series_id,
                'api_key': self.api_key,
                'file_type': 'json'
            }
            
            if start_date:
                params['observation_start'] = start_date
            if end_date:
                params['observation_end'] = end_date
            
            response = requests.get(
                f"{self.BASE_URL}/series/observations",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            observations = data.get('observations', [])
            
            df = pd.DataFrame(observations)
            df['date'] = pd.to_datetime(df['date'])
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df = df.dropna(subset=['value'])
            
            return df[['date', 'value']].set_index('date')
            
        except Exception as e:
            logger.error(f"Failed to fetch FRED series {series_id}: {e}")
            return None
    
    def fetch_economic_indicators(self) -> Dict[str, pd.DataFrame]:
        """Fetch common economic indicators."""
        indicators = {}
        
        for name, series_id in self.INDICATORS.items():
            df = self.fetch_series(series_id)
            if df is not None:
                indicators[name] = df
        
        return indicators


class MockDataSource(DataSource):
    """Mock data source for testing."""
    
    MOCK_STOCKS = {
        'AAPL': {'name': 'Apple Inc.', 'sector': 'Technology', 'pe': 28.5, 'base': 185},
        'MSFT': {'name': 'Microsoft Corporation', 'sector': 'Technology', 'pe': 35.2, 'base': 378},
        'GOOGL': {'name': 'Alphabet Inc.', 'sector': 'Technology', 'pe': 24.8, 'base': 142},
        'AMZN': {'name': 'Amazon.com Inc.', 'sector': 'Consumer Cyclical', 'pe': 62.5, 'base': 155},
        'NVDA': {'name': 'NVIDIA Corporation', 'sector': 'Technology', 'pe': 65.3, 'base': 480},
        'META': {'name': 'Meta Platforms Inc.', 'sector': 'Technology', 'pe': 23.1, 'base': 355},
        'TSLA': {'name': 'Tesla Inc.', 'sector': 'Consumer Cyclical', 'pe': 72.4, 'base': 245},
    }
    
    def __init__(self):
        super().__init__("mock", 10000)
        np.random.seed(42)
    
    def fetch_prices(self, symbol: str, period: str = "2y") -> Optional[pd.DataFrame]:
        """Generate realistic mock price data."""
        period_days = {'1mo': 22, '3mo': 66, '6mo': 126, '1y': 252, '2y': 504}
        days = period_days.get(period, 252)
        
        stock = self.MOCK_STOCKS.get(symbol.upper(), {'base': 100})
        base_price = stock['base']
        
        dates = pd.date_range(end=datetime.now(), periods=days, freq='B')
        
        # Generate returns with realistic properties
        volatility = 0.02
        trend = 0.0003
        returns = np.random.normal(trend, volatility, days)
        
        # Add autocorrelation
        for i in range(1, len(returns)):
            returns[i] += 0.1 * returns[i-1]
        
        prices = base_price * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame({
            'date': dates,
            'open': prices * (1 + np.random.normal(0, 0.002, days)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.01, days))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.01, days))),
            'close': prices,
            'volume': np.random.randint(10_000_000, 100_000_000, days),
            'symbol': symbol.upper()
        })
        
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        return df.set_index('date')
    
    def fetch_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Generate mock stock info."""
        stock = self.MOCK_STOCKS.get(symbol.upper())
        
        if not stock:
            return {
                'symbol': symbol.upper(),
                'name': f'{symbol.upper()} Corp',
                'sector': 'Unknown',
                'pe_ratio': np.random.uniform(15, 40),
                'market_cap': np.random.uniform(1e9, 500e9),
            }
        
        return {
            'symbol': symbol.upper(),
            'name': stock['name'],
            'sector': stock['sector'],
            'industry': 'Technology',
            'pe_ratio': stock['pe'],
            'market_cap': np.random.uniform(100e9, 3e12),
            'revenue_growth': np.random.uniform(0.05, 0.3),
            'profit_margin': np.random.uniform(0.1, 0.4),
            'debt_to_equity': np.random.uniform(20, 150),
        }


class DataFetcher:
    """
    Main data fetcher with multiple sources and fallbacks.
    
    Provides a unified interface for fetching stock data from
    multiple sources with automatic fallback, caching, and
    error handling.
    """
    
    def __init__(self, use_mock: bool = False):
        self.cache = CacheManager()
        self.use_mock = use_mock
        
        # Initialize sources in priority order
        self.sources: List[DataSource] = []
        
        if use_mock:
            self.sources.append(MockDataSource())
        else:
            self.sources.append(YFinanceSource())
            # Add more sources here as fallbacks
        
        # Always have mock as last fallback
        if not use_mock:
            self.sources.append(MockDataSource())
    
    def get_prices(
        self,
        symbol: str,
        period: str = "2y",
        use_cache: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Fetch price data with caching and fallback.

        Tries each source in order until one succeeds.
        """
        # Validate and sanitize symbol to prevent injection
        symbol = validate_symbol(symbol)

        cache_key = f"prices_{symbol}_{period}"
        
        if use_cache:
            cached = self.cache.get(cache_key, config.data.price_cache_ttl)
            if cached is not None:
                return cached
        
        for source in self.sources:
            if not source.enabled:
                continue
            
            try:
                df = circuit_breaker.call(
                    source.name,
                    source.fetch_prices,
                    symbol,
                    period
                )
                
                if df is not None and len(df) > 0:
                    self.cache.set(cache_key, df)
                    return df
                    
            except Exception as e:
                logger.warning(f"Source {source.name} failed for {symbol}: {e}")
                continue
        
        logger.error(f"All sources failed for {symbol}")
        return None
    
    def get_info(self, symbol: str, use_cache: bool = True) -> Optional[Dict[str, Any]]:
        """Fetch stock info with caching and fallback."""
        # Validate and sanitize symbol to prevent injection
        symbol = validate_symbol(symbol)

        cache_key = f"info_{symbol}"
        
        if use_cache:
            cached = self.cache.get(cache_key, config.data.fundamental_cache_ttl)
            if cached is not None:
                return cached
        
        for source in self.sources:
            if not source.enabled:
                continue
            
            try:
                info = circuit_breaker.call(
                    source.name,
                    source.fetch_info,
                    symbol
                )
                
                if info is not None:
                    self.cache.set(cache_key, info)
                    return info
                    
            except Exception as e:
                logger.warning(f"Source {source.name} failed for {symbol} info: {e}")
                continue
        
        return None
    
    def get_multiple(
        self,
        symbols: List[str],
        period: str = "2y"
    ) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple symbols."""
        results = {}
        
        for symbol in symbols:
            df = self.get_prices(symbol, period)
            if df is not None:
                results[symbol] = df
            time.sleep(0.1)  # Small delay between requests
        
        return results
    
    def get_stock_data(self, symbol: str, period: str = "2y") -> Optional[StockData]:
        """Fetch complete stock data (prices + info)."""
        prices = self.get_prices(symbol, period)
        info = self.get_info(symbol)
        
        if prices is None:
            return None
        
        return StockData(
            symbol=symbol,
            prices=prices,
            info=info or {},
            source=self.sources[0].name if self.sources else "unknown"
        )


# Global instance - uses real data sources (YFinance) with mock as fallback
data_fetcher = DataFetcher(use_mock=False)
