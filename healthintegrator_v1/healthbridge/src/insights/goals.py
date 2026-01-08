"""
Goal Setting and Tracking System

Allows users to set, track, and achieve health goals.
"""

from datetime import date, datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum
import json


class GoalCategory(Enum):
    SLEEP = "sleep"
    ACTIVITY = "activity"
    RECOVERY = "recovery"
    HEART = "heart"
    WEIGHT = "weight"
    CUSTOM = "custom"


class GoalFrequency(Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class GoalStatus(Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    PAUSED = "paused"
    FAILED = "failed"


@dataclass
class Goal:
    """A health goal with tracking."""
    id: str
    name: str
    category: GoalCategory
    metric: str
    target_value: float
    current_value: float = 0
    frequency: GoalFrequency = GoalFrequency.DAILY
    comparison: str = "gte"  # gte, lte, eq, between
    target_max: Optional[float] = None  # For "between" comparison
    start_date: date = field(default_factory=date.today)
    end_date: Optional[date] = None
    status: GoalStatus = GoalStatus.ACTIVE
    streak: int = 0
    best_streak: int = 0
    times_achieved: int = 0
    total_attempts: int = 0
    notes: str = ""

    def check_achievement(self, value: float) -> bool:
        """Check if value meets the goal."""
        if self.comparison == "gte":
            return value >= self.target_value
        elif self.comparison == "lte":
            return value <= self.target_value
        elif self.comparison == "eq":
            return abs(value - self.target_value) < 0.01
        elif self.comparison == "between":
            return self.target_value <= value <= (self.target_max or self.target_value)
        return False

    def update_progress(self, value: float, achieved: bool):
        """Update goal progress."""
        self.current_value = value
        self.total_attempts += 1

        if achieved:
            self.times_achieved += 1
            self.streak += 1
            self.best_streak = max(self.best_streak, self.streak)
        else:
            self.streak = 0

    @property
    def achievement_rate(self) -> float:
        """Calculate achievement rate."""
        if self.total_attempts == 0:
            return 0
        return (self.times_achieved / self.total_attempts) * 100

    @property
    def progress_pct(self) -> float:
        """Calculate progress percentage toward target."""
        if self.target_value == 0:
            return 0
        if self.comparison in ["gte", "eq"]:
            return min(100, (self.current_value / self.target_value) * 100)
        elif self.comparison == "lte":
            # For "less than" goals, being under is good
            if self.current_value <= self.target_value:
                return 100
            return max(0, 100 - ((self.current_value - self.target_value) / self.target_value * 100))
        return 0

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            'id': self.id,
            'name': self.name,
            'category': self.category.value,
            'metric': self.metric,
            'target_value': self.target_value,
            'current_value': self.current_value,
            'frequency': self.frequency.value,
            'comparison': self.comparison,
            'target_max': self.target_max,
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat() if self.end_date else None,
            'status': self.status.value,
            'streak': self.streak,
            'best_streak': self.best_streak,
            'times_achieved': self.times_achieved,
            'total_attempts': self.total_attempts,
            'notes': self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Goal':
        """Create Goal from dictionary."""
        return cls(
            id=data['id'],
            name=data['name'],
            category=GoalCategory(data['category']),
            metric=data['metric'],
            target_value=data['target_value'],
            current_value=data.get('current_value', 0),
            frequency=GoalFrequency(data.get('frequency', 'daily')),
            comparison=data.get('comparison', 'gte'),
            target_max=data.get('target_max'),
            start_date=date.fromisoformat(data['start_date']) if data.get('start_date') else date.today(),
            end_date=date.fromisoformat(data['end_date']) if data.get('end_date') else None,
            status=GoalStatus(data.get('status', 'active')),
            streak=data.get('streak', 0),
            best_streak=data.get('best_streak', 0),
            times_achieved=data.get('times_achieved', 0),
            total_attempts=data.get('total_attempts', 0),
            notes=data.get('notes', ''),
        )


# Preset goal templates
GOAL_TEMPLATES = {
    'sleep_7h': {
        'name': 'Get 7+ Hours of Sleep',
        'category': GoalCategory.SLEEP,
        'metric': 'sleep_duration',
        'target_value': 7.0,
        'comparison': 'gte',
        'frequency': GoalFrequency.DAILY,
    },
    'sleep_8h': {
        'name': 'Get 8+ Hours of Sleep',
        'category': GoalCategory.SLEEP,
        'metric': 'sleep_duration',
        'target_value': 8.0,
        'comparison': 'gte',
        'frequency': GoalFrequency.DAILY,
    },
    'steps_10k': {
        'name': 'Walk 10,000 Steps',
        'category': GoalCategory.ACTIVITY,
        'metric': 'steps',
        'target_value': 10000,
        'comparison': 'gte',
        'frequency': GoalFrequency.DAILY,
    },
    'steps_8k': {
        'name': 'Walk 8,000 Steps',
        'category': GoalCategory.ACTIVITY,
        'metric': 'steps',
        'target_value': 8000,
        'comparison': 'gte',
        'frequency': GoalFrequency.DAILY,
    },
    'active_30': {
        'name': '30 Minutes Active',
        'category': GoalCategory.ACTIVITY,
        'metric': 'active_minutes',
        'target_value': 30,
        'comparison': 'gte',
        'frequency': GoalFrequency.DAILY,
    },
    'rhr_under_60': {
        'name': 'Resting HR Under 60',
        'category': GoalCategory.HEART,
        'metric': 'resting_hr',
        'target_value': 60,
        'comparison': 'lte',
        'frequency': GoalFrequency.DAILY,
    },
    'hrv_above_50': {
        'name': 'HRV Above 50ms',
        'category': GoalCategory.HEART,
        'metric': 'hrv',
        'target_value': 50,
        'comparison': 'gte',
        'frequency': GoalFrequency.DAILY,
    },
    'readiness_80': {
        'name': 'Readiness Score 80+',
        'category': GoalCategory.RECOVERY,
        'metric': 'readiness_score',
        'target_value': 80,
        'comparison': 'gte',
        'frequency': GoalFrequency.DAILY,
    },
    'sleep_score_80': {
        'name': 'Sleep Score 80+',
        'category': GoalCategory.SLEEP,
        'metric': 'sleep_score',
        'target_value': 80,
        'comparison': 'gte',
        'frequency': GoalFrequency.DAILY,
    },
    'weekly_steps_70k': {
        'name': '70,000 Weekly Steps',
        'category': GoalCategory.ACTIVITY,
        'metric': 'steps',
        'target_value': 70000,
        'comparison': 'gte',
        'frequency': GoalFrequency.WEEKLY,
    },
}


def create_goal_from_template(template_id: str, goal_id: str = None) -> Goal:
    """Create a goal from a preset template."""
    if template_id not in GOAL_TEMPLATES:
        raise ValueError(f"Unknown template: {template_id}")

    template = GOAL_TEMPLATES[template_id]

    return Goal(
        id=goal_id or f"goal_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        name=template['name'],
        category=template['category'],
        metric=template['metric'],
        target_value=template['target_value'],
        comparison=template.get('comparison', 'gte'),
        frequency=template.get('frequency', GoalFrequency.DAILY),
    )


def evaluate_goals(goals: List[Goal], health_data: List[Dict]) -> List[Dict]:
    """
    Evaluate goals against recent health data.

    Returns list of goal results with achievement status.
    """
    results = []

    if not health_data:
        return results

    # Get most recent data
    latest = health_data[-1]

    # For weekly goals, aggregate last 7 days
    weekly_data = health_data[-7:] if len(health_data) >= 7 else health_data

    for goal in goals:
        if goal.status != GoalStatus.ACTIVE:
            continue

        result = {
            'goal': goal,
            'achieved': False,
            'value': None,
            'target': goal.target_value,
            'progress_pct': 0,
        }

        # Get the relevant value based on frequency
        if goal.frequency == GoalFrequency.DAILY:
            value = latest.get(goal.metric)
        elif goal.frequency == GoalFrequency.WEEKLY:
            # Sum or average based on metric type
            values = [d.get(goal.metric) for d in weekly_data if d.get(goal.metric) is not None]
            if values:
                if goal.metric in ['steps', 'calories_active', 'active_minutes']:
                    value = sum(values)
                else:
                    value = sum(values) / len(values)
            else:
                value = None
        else:
            value = latest.get(goal.metric)

        if value is not None:
            result['value'] = value
            result['achieved'] = goal.check_achievement(value)

            # Calculate progress
            if goal.comparison == 'gte':
                result['progress_pct'] = min(100, (value / goal.target_value) * 100) if goal.target_value > 0 else 0
            elif goal.comparison == 'lte':
                if value <= goal.target_value:
                    result['progress_pct'] = 100
                else:
                    result['progress_pct'] = max(0, 100 - ((value - goal.target_value) / goal.target_value * 100))

            # Update goal tracking
            goal.update_progress(value, result['achieved'])

        results.append(result)

    return results


def get_goal_insights(goals: List[Goal], results: List[Dict]) -> List[str]:
    """Generate insights about goal progress."""
    insights = []

    # Overall achievement rate
    achieved_count = sum(1 for r in results if r['achieved'])
    total = len(results)

    if total > 0:
        rate = (achieved_count / total) * 100
        if rate >= 80:
            insights.append(f"Excellent! You're hitting {rate:.0f}% of your goals today.")
        elif rate >= 50:
            insights.append(f"Good progress! You've achieved {achieved_count} of {total} goals today.")
        else:
            insights.append(f"Keep pushing! {total - achieved_count} goals remaining for today.")

    # Streak insights
    for goal in goals:
        if goal.streak >= 7:
            insights.append(f"🔥 {goal.streak}-day streak on '{goal.name}'!")
        elif goal.streak == 0 and goal.best_streak >= 5:
            insights.append(f"Your best streak for '{goal.name}' was {goal.best_streak} days. Start a new one!")

    # Achievement rate insights
    for goal in goals:
        if goal.total_attempts >= 7:
            rate = goal.achievement_rate
            if rate >= 90:
                insights.append(f"'{goal.name}' is almost automatic for you ({rate:.0f}% success rate).")
            elif rate < 50:
                insights.append(f"Consider adjusting '{goal.name}' - only {rate:.0f}% success rate.")

    return insights


def suggest_goals(health_data: List[Dict]) -> List[str]:
    """Suggest goals based on user's health data patterns."""
    suggestions = []

    if len(health_data) < 7:
        return ['Start tracking for at least 7 days to get personalized goal suggestions.']

    recent = health_data[-7:]

    # Analyze sleep
    sleep_vals = [d.get('sleep_duration') for d in recent if d.get('sleep_duration')]
    if sleep_vals:
        avg_sleep = sum(sleep_vals) / len(sleep_vals)
        if avg_sleep < 7:
            suggestions.append('sleep_7h')
        elif avg_sleep >= 7:
            suggestions.append('sleep_8h')

    # Analyze steps
    step_vals = [d.get('steps') for d in recent if d.get('steps')]
    if step_vals:
        avg_steps = sum(step_vals) / len(step_vals)
        if avg_steps < 8000:
            suggestions.append('steps_8k')
        elif avg_steps >= 8000:
            suggestions.append('steps_10k')

    # Analyze HRV
    hrv_vals = [d.get('hrv') for d in recent if d.get('hrv')]
    if hrv_vals:
        avg_hrv = sum(hrv_vals) / len(hrv_vals)
        if avg_hrv < 50:
            suggestions.append('hrv_above_50')

    # Analyze readiness
    readiness_vals = [d.get('readiness_score') for d in recent if d.get('readiness_score')]
    if readiness_vals:
        avg_readiness = sum(readiness_vals) / len(readiness_vals)
        if avg_readiness >= 70:
            suggestions.append('readiness_80')

    return suggestions
