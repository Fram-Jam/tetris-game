"""
Tests for AI Insights Module
"""

import pytest
from datetime import date, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.insights.ai_coach import (
    prepare_health_summary,
    _get_rule_based_insights,
    get_correlation_insights
)
from src.data.synthetic.patient_generator import generate_demo_data


class TestHealthSummary:
    """Tests for health summary preparation."""

    def test_prepare_health_summary_returns_string(self):
        """Test that summary is a string."""
        _, data = generate_demo_data(days=14)
        summary = prepare_health_summary(data)

        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_summary_contains_key_sections(self):
        """Test that summary contains expected sections."""
        _, data = generate_demo_data(days=14)
        summary = prepare_health_summary(data)

        assert 'Sleep' in summary
        assert 'Heart Health' in summary
        assert 'Activity' in summary
        assert 'Recovery' in summary

    def test_summary_with_fewer_days(self):
        """Test summary with less data than requested."""
        _, data = generate_demo_data(days=5)
        summary = prepare_health_summary(data, days=7)

        assert isinstance(summary, str)
        assert len(summary) > 0


class TestRuleBasedInsights:
    """Tests for rule-based insights fallback."""

    def test_rule_based_insights_returns_string(self):
        """Test that insights are returned as string."""
        _, data = generate_demo_data(days=14)
        insights = _get_rule_based_insights(data)

        assert isinstance(insights, str)
        assert len(insights) > 0

    def test_insights_with_low_sleep(self):
        """Test that low sleep triggers appropriate insight."""
        _, data = generate_demo_data(days=14)
        # Artificially lower sleep
        for d in data:
            d['sleep_duration'] = 5.5

        insights = _get_rule_based_insights(data)

        # Should mention sleep being below recommended
        assert 'sleep' in insights.lower() or 'Sleep' in insights

    def test_insights_with_high_activity(self):
        """Test insights with high activity levels."""
        _, data = generate_demo_data(days=14)
        # Artificially increase steps
        for d in data:
            d['steps'] = 15000

        insights = _get_rule_based_insights(data)

        # Should have positive activity mention
        assert 'steps' in insights.lower() or 'activity' in insights.lower()


class TestCorrelationInsights:
    """Tests for correlation analysis."""

    def test_correlation_insights_returns_list(self):
        """Test that correlations are returned as list."""
        _, data = generate_demo_data(days=90)
        correlations = get_correlation_insights(data)

        assert isinstance(correlations, list)

    def test_correlation_structure(self):
        """Test correlation item structure."""
        _, data = generate_demo_data(days=90)
        correlations = get_correlation_insights(data)

        if correlations:
            corr = correlations[0]
            assert 'finding' in corr
            assert 'detail' in corr
            assert 'strength' in corr

    def test_correlations_sorted_by_strength(self):
        """Test that correlations are sorted by strength."""
        _, data = generate_demo_data(days=90)
        correlations = get_correlation_insights(data)

        if len(correlations) > 1:
            for i in range(1, len(correlations)):
                assert correlations[i-1]['strength'] >= correlations[i]['strength']

    def test_correlation_strength_range(self):
        """Test that correlation strengths are in valid range."""
        _, data = generate_demo_data(days=90)
        correlations = get_correlation_insights(data)

        for corr in correlations:
            assert 0 <= corr['strength'] <= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
