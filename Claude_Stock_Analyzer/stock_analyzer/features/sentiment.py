"""
News Sentiment Analysis
========================
Sentiment analysis for stock-related news:
- Multiple news source support
- Basic sentiment scoring
- FinBERT integration (optional)
- Sentiment trend tracking
"""

import os
import re
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger("stock_analyzer.sentiment")


class SentimentLabel(Enum):
    """Sentiment classification labels."""
    VERY_BEARISH = -2
    BEARISH = -1
    NEUTRAL = 0
    BULLISH = 1
    VERY_BULLISH = 2


@dataclass
class NewsArticle:
    """Represents a news article."""
    title: str
    description: Optional[str]
    source: str
    published_at: datetime
    url: Optional[str] = None
    sentiment_score: Optional[float] = None
    sentiment_label: Optional[SentimentLabel] = None


@dataclass
class SentimentResult:
    """Result of sentiment analysis for a stock."""
    symbol: str
    timestamp: datetime
    overall_score: float  # -1 to 1
    overall_label: SentimentLabel
    num_articles: int
    bullish_count: int
    bearish_count: int
    neutral_count: int
    articles: List[NewsArticle] = field(default_factory=list)
    trend: str = "stable"  # 'improving', 'declining', 'stable'
    confidence: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'overall_score': self.overall_score,
            'overall_label': self.overall_label.name,
            'num_articles': self.num_articles,
            'bullish_count': self.bullish_count,
            'bearish_count': self.bearish_count,
            'neutral_count': self.neutral_count,
            'trend': self.trend,
            'confidence': self.confidence
        }


class BasicSentimentAnalyzer:
    """
    Simple rule-based sentiment analyzer.

    Uses keyword matching for sentiment scoring.
    Suitable when FinBERT is not available.
    """

    # Positive keywords (financial context)
    POSITIVE_WORDS = {
        'strong': 1, 'growth': 1, 'profit': 1, 'gains': 1, 'surge': 2,
        'rally': 2, 'bullish': 2, 'soar': 2, 'jump': 1, 'rise': 1,
        'beat': 1, 'exceed': 1, 'outperform': 2, 'upgrade': 2, 'buy': 1,
        'positive': 1, 'success': 1, 'record': 1, 'high': 1, 'boom': 2,
        'breakthrough': 2, 'innovation': 1, 'opportunity': 1, 'optimistic': 2,
        'recovery': 1, 'momentum': 1, 'dividend': 1, 'earnings': 1
    }

    # Negative keywords (financial context)
    NEGATIVE_WORDS = {
        'loss': -1, 'decline': -1, 'drop': -1, 'fall': -1, 'crash': -2,
        'bearish': -2, 'plunge': -2, 'tumble': -2, 'slump': -2, 'weak': -1,
        'miss': -1, 'downgrade': -2, 'sell': -1, 'negative': -1, 'risk': -1,
        'concern': -1, 'warning': -1, 'cut': -1, 'layoff': -2, 'bankruptcy': -2,
        'lawsuit': -1, 'investigation': -1, 'fraud': -2, 'scandal': -2,
        'recession': -2, 'inflation': -1, 'debt': -1, 'default': -2
    }

    def analyze_text(self, text: str) -> Tuple[float, SentimentLabel]:
        """
        Analyze sentiment of a text.

        Returns:
            (score, label) where score is -1 to 1
        """
        if not text:
            return 0.0, SentimentLabel.NEUTRAL

        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)

        score = 0
        word_count = 0

        for word in words:
            if word in self.POSITIVE_WORDS:
                score += self.POSITIVE_WORDS[word]
                word_count += 1
            elif word in self.NEGATIVE_WORDS:
                score += self.NEGATIVE_WORDS[word]
                word_count += 1

        # Normalize score
        if word_count > 0:
            normalized_score = score / (word_count * 2)  # Max possible per word is 2
            normalized_score = max(-1, min(1, normalized_score))
        else:
            normalized_score = 0.0

        # Determine label
        if normalized_score >= 0.5:
            label = SentimentLabel.VERY_BULLISH
        elif normalized_score >= 0.2:
            label = SentimentLabel.BULLISH
        elif normalized_score <= -0.5:
            label = SentimentLabel.VERY_BEARISH
        elif normalized_score <= -0.2:
            label = SentimentLabel.BEARISH
        else:
            label = SentimentLabel.NEUTRAL

        return normalized_score, label


class FinBERTAnalyzer:
    """
    FinBERT-based sentiment analyzer.

    Uses the ProsusAI/finbert model for financial sentiment.
    Falls back to basic analyzer if transformers not available.
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.available = False
        self._load_model()

    def _load_model(self):
        """Attempt to load FinBERT model."""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch

            model_name = "ProsusAI/finbert"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.eval()
            self.available = True
            logger.info("FinBERT model loaded successfully")

        except ImportError:
            logger.warning("transformers/torch not available, using basic sentiment analyzer")
            self.available = False
        except Exception as e:
            logger.warning(f"Failed to load FinBERT: {e}")
            self.available = False

    def analyze_text(self, text: str) -> Tuple[float, SentimentLabel]:
        """
        Analyze sentiment using FinBERT.

        Returns:
            (score, label) where score is -1 to 1
        """
        if not self.available or not text:
            return BasicSentimentAnalyzer().analyze_text(text)

        try:
            import torch

            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

            # FinBERT outputs: positive, negative, neutral
            positive = probs[0][0].item()
            negative = probs[0][1].item()
            neutral = probs[0][2].item()

            # Calculate score (-1 to 1)
            score = positive - negative

            # Determine label
            if score >= 0.5:
                label = SentimentLabel.VERY_BULLISH
            elif score >= 0.2:
                label = SentimentLabel.BULLISH
            elif score <= -0.5:
                label = SentimentLabel.VERY_BEARISH
            elif score <= -0.2:
                label = SentimentLabel.BEARISH
            else:
                label = SentimentLabel.NEUTRAL

            return score, label

        except Exception as e:
            logger.warning(f"FinBERT analysis failed: {e}")
            return BasicSentimentAnalyzer().analyze_text(text)


class NewsFetcher:
    """
    Fetch news from various sources.

    Supports:
    - NewsAPI (requires API key)
    - Mock news for testing
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("NEWSAPI_KEY")
        self.use_mock = not self.api_key

    def fetch_news(
        self,
        symbol: str,
        company_name: Optional[str] = None,
        days: int = 7,
        max_articles: int = 20
    ) -> List[NewsArticle]:
        """
        Fetch news articles for a stock.

        Args:
            symbol: Stock symbol
            company_name: Company name for better search
            days: Number of days to look back
            max_articles: Maximum articles to return

        Returns:
            List of NewsArticle objects
        """
        if self.use_mock:
            return self._generate_mock_news(symbol, max_articles)

        return self._fetch_from_newsapi(symbol, company_name, days, max_articles)

    def _fetch_from_newsapi(
        self,
        symbol: str,
        company_name: Optional[str],
        days: int,
        max_articles: int
    ) -> List[NewsArticle]:
        """Fetch from NewsAPI."""
        try:
            import requests

            query = company_name or symbol
            from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

            url = "https://newsapi.org/v2/everything"
            params = {
                'q': f"{query} stock",
                'from': from_date,
                'sortBy': 'relevancy',
                'language': 'en',
                'pageSize': max_articles,
                'apiKey': self.api_key
            }

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            articles = []
            for item in data.get('articles', []):
                articles.append(NewsArticle(
                    title=item.get('title', ''),
                    description=item.get('description'),
                    source=item.get('source', {}).get('name', 'Unknown'),
                    published_at=datetime.fromisoformat(
                        item.get('publishedAt', '').replace('Z', '+00:00')
                    ) if item.get('publishedAt') else datetime.now(),
                    url=item.get('url')
                ))

            logger.info(f"Fetched {len(articles)} articles for {symbol}")
            return articles

        except Exception as e:
            logger.warning(f"NewsAPI fetch failed: {e}")
            return self._generate_mock_news(symbol, max_articles)

    def _generate_mock_news(self, symbol: str, max_articles: int) -> List[NewsArticle]:
        """Generate mock news for testing."""
        templates = [
            ("{symbol} Reports Strong Quarterly Earnings", "bullish"),
            ("{symbol} Stock Surges on Positive Outlook", "bullish"),
            ("{symbol} Announces New Product Launch", "bullish"),
            ("{symbol} Faces Regulatory Challenges", "bearish"),
            ("{symbol} Stock Drops Amid Market Uncertainty", "bearish"),
            ("{symbol} Reports Mixed Results", "neutral"),
            ("{symbol} Maintains Steady Growth", "neutral"),
            ("Analysts Upgrade {symbol} to Buy", "bullish"),
            ("Concerns Grow Over {symbol} Debt Levels", "bearish"),
            ("{symbol} CEO Discusses Future Strategy", "neutral"),
        ]

        articles = []
        np.random.seed(hash(symbol) % 2**32)

        for i in range(min(max_articles, len(templates))):
            template, _ = templates[i]
            title = template.format(symbol=symbol)

            articles.append(NewsArticle(
                title=title,
                description=f"News article about {symbol}",
                source="Mock News",
                published_at=datetime.now() - timedelta(days=np.random.randint(0, 7)),
                url=None
            ))

        return articles


class SentimentAggregator:
    """
    Aggregate sentiment from multiple news articles.

    Features:
    - Time-weighted averaging
    - Confidence scoring
    - Trend detection
    """

    def __init__(self, use_finbert: bool = False):
        if use_finbert:
            self.analyzer = FinBERTAnalyzer()
            if not self.analyzer.available:
                self.analyzer = BasicSentimentAnalyzer()
        else:
            self.analyzer = BasicSentimentAnalyzer()

        self.news_fetcher = NewsFetcher()

    def analyze_stock(
        self,
        symbol: str,
        company_name: Optional[str] = None,
        days: int = 7
    ) -> SentimentResult:
        """
        Perform complete sentiment analysis for a stock.

        Args:
            symbol: Stock symbol
            company_name: Company name for better news search
            days: Days of news to analyze

        Returns:
            SentimentResult with aggregated sentiment
        """
        # Fetch news
        articles = self.news_fetcher.fetch_news(symbol, company_name, days)

        if not articles:
            return SentimentResult(
                symbol=symbol,
                timestamp=datetime.now(),
                overall_score=0.0,
                overall_label=SentimentLabel.NEUTRAL,
                num_articles=0,
                bullish_count=0,
                bearish_count=0,
                neutral_count=0,
                articles=[],
                trend="stable",
                confidence=0.0
            )

        # Analyze each article
        scores = []
        bullish = 0
        bearish = 0
        neutral = 0

        for article in articles:
            text = f"{article.title} {article.description or ''}"
            score, label = self.analyzer.analyze_text(text)
            article.sentiment_score = score
            article.sentiment_label = label

            # Time decay weight (newer articles weighted more)
            age_days = (datetime.now() - article.published_at).days
            weight = np.exp(-age_days / 7)  # Exponential decay
            scores.append((score, weight))

            if label in [SentimentLabel.BULLISH, SentimentLabel.VERY_BULLISH]:
                bullish += 1
            elif label in [SentimentLabel.BEARISH, SentimentLabel.VERY_BEARISH]:
                bearish += 1
            else:
                neutral += 1

        # Weighted average score
        if scores:
            total_weight = sum(w for _, w in scores)
            if total_weight > 0:
                overall_score = sum(s * w for s, w in scores) / total_weight
            else:
                overall_score = 0.0
        else:
            overall_score = 0.0

        # Determine overall label
        if overall_score >= 0.5:
            overall_label = SentimentLabel.VERY_BULLISH
        elif overall_score >= 0.2:
            overall_label = SentimentLabel.BULLISH
        elif overall_score <= -0.5:
            overall_label = SentimentLabel.VERY_BEARISH
        elif overall_score <= -0.2:
            overall_label = SentimentLabel.BEARISH
        else:
            overall_label = SentimentLabel.NEUTRAL

        # Detect trend (comparing recent vs older articles)
        if len(articles) >= 4:
            sorted_articles = sorted(articles, key=lambda a: a.published_at, reverse=True)
            mid = len(sorted_articles) // 2
            recent_avg = np.mean([a.sentiment_score or 0 for a in sorted_articles[:mid]])
            older_avg = np.mean([a.sentiment_score or 0 for a in sorted_articles[mid:]])

            if recent_avg > older_avg + 0.15:
                trend = "improving"
            elif recent_avg < older_avg - 0.15:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "stable"

        # Confidence based on article count and agreement
        max_count = max(bullish, bearish, neutral)
        total = bullish + bearish + neutral
        agreement = max_count / total if total > 0 else 0
        confidence = min(1.0, (len(articles) / 10) * agreement)

        return SentimentResult(
            symbol=symbol,
            timestamp=datetime.now(),
            overall_score=overall_score,
            overall_label=overall_label,
            num_articles=len(articles),
            bullish_count=bullish,
            bearish_count=bearish,
            neutral_count=neutral,
            articles=articles,
            trend=trend,
            confidence=confidence
        )


def get_sentiment_signal(result: SentimentResult) -> Tuple[int, float]:
    """
    Convert sentiment result to trading signal.

    Returns:
        (signal, confidence) where signal is 1 (buy), -1 (sell), 0 (hold)
    """
    if result.confidence < 0.3:
        return 0, result.confidence

    if result.overall_label in [SentimentLabel.VERY_BULLISH, SentimentLabel.BULLISH]:
        if result.trend == "improving":
            return 1, result.confidence * 1.2
        return 1, result.confidence

    elif result.overall_label in [SentimentLabel.VERY_BEARISH, SentimentLabel.BEARISH]:
        if result.trend == "declining":
            return -1, result.confidence * 1.2
        return -1, result.confidence

    return 0, result.confidence


def create_sentiment_feature(
    symbol: str,
    prices: pd.DataFrame,
    company_name: Optional[str] = None
) -> pd.Series:
    """
    Create a sentiment feature for use in ML models.

    Returns a Series with sentiment scores aligned to price dates.
    """
    aggregator = SentimentAggregator()
    result = aggregator.analyze_stock(symbol, company_name)

    # Create a constant sentiment feature for now
    # In production, you'd want historical sentiment data
    sentiment_series = pd.Series(
        result.overall_score,
        index=prices.index,
        name='news_sentiment'
    )

    return sentiment_series
