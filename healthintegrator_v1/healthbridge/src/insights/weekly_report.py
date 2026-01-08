"""
Weekly Health Report Generator

Generates comprehensive weekly health reports with insights and trends.
"""

from datetime import date, datetime, timedelta
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np


def get_week_range(target_date: date = None) -> Tuple[date, date]:
    """Get start and end of week containing target_date."""
    if target_date is None:
        target_date = date.today()

    # Start of week (Monday)
    start = target_date - timedelta(days=target_date.weekday())
    end = start + timedelta(days=6)

    return start, end


def calculate_weekly_stats(data: List[Dict], start_date: date, end_date: date) -> Dict:
    """Calculate statistics for a week of data."""
    # Filter to week
    week_data = [d for d in data if start_date <= d['date'] <= end_date]

    if not week_data:
        return None

    df = pd.DataFrame(week_data)

    stats = {
        'date_range': {
            'start': start_date.isoformat(),
            'end': end_date.isoformat(),
        },
        'days_with_data': len(week_data),
    }

    # Sleep stats
    if 'sleep_duration' in df.columns:
        sleep_vals = df['sleep_duration'].dropna()
        if len(sleep_vals) > 0:
            stats['sleep'] = {
                'avg_duration': round(sleep_vals.mean(), 1),
                'total_hours': round(sleep_vals.sum(), 1),
                'best_night': round(sleep_vals.max(), 1),
                'worst_night': round(sleep_vals.min(), 1),
                'consistency': round(100 - (sleep_vals.std() / sleep_vals.mean() * 100), 1) if sleep_vals.mean() > 0 else 0,
            }

    if 'sleep_score' in df.columns:
        scores = df['sleep_score'].dropna()
        if len(scores) > 0:
            stats['sleep']['avg_score'] = round(scores.mean(), 1)
            stats['sleep']['best_score'] = int(scores.max())

    # HRV stats
    if 'hrv' in df.columns:
        hrv_vals = df['hrv'].dropna()
        if len(hrv_vals) > 0:
            stats['hrv'] = {
                'avg': round(hrv_vals.mean(), 1),
                'max': round(hrv_vals.max(), 1),
                'min': round(hrv_vals.min(), 1),
                'trend': 'stable',
            }
            # Calculate trend
            if len(hrv_vals) >= 3:
                first_half = hrv_vals.iloc[:len(hrv_vals)//2].mean()
                second_half = hrv_vals.iloc[len(hrv_vals)//2:].mean()
                pct_change = ((second_half - first_half) / first_half) * 100 if first_half > 0 else 0
                if pct_change > 5:
                    stats['hrv']['trend'] = 'improving'
                elif pct_change < -5:
                    stats['hrv']['trend'] = 'declining'

    # Resting HR stats
    if 'resting_hr' in df.columns:
        rhr_vals = df['resting_hr'].dropna()
        if len(rhr_vals) > 0:
            stats['resting_hr'] = {
                'avg': round(rhr_vals.mean(), 1),
                'min': int(rhr_vals.min()),
                'max': int(rhr_vals.max()),
            }

    # Activity stats
    if 'steps' in df.columns:
        steps_vals = df['steps'].dropna()
        if len(steps_vals) > 0:
            stats['activity'] = {
                'total_steps': int(steps_vals.sum()),
                'avg_daily_steps': int(steps_vals.mean()),
                'best_day': int(steps_vals.max()),
                'days_over_10k': int((steps_vals >= 10000).sum()),
            }

    if 'active_minutes' in df.columns:
        active = df['active_minutes'].dropna()
        if len(active) > 0:
            stats['activity']['total_active_minutes'] = int(active.sum())
            stats['activity']['avg_active_minutes'] = int(active.mean())

    if 'calories_active' in df.columns:
        cals = df['calories_active'].dropna()
        if len(cals) > 0:
            stats['activity']['total_calories_burned'] = int(cals.sum())

    # Readiness stats
    if 'readiness_score' in df.columns:
        readiness = df['readiness_score'].dropna()
        if len(readiness) > 0:
            stats['readiness'] = {
                'avg_score': round(readiness.mean(), 1),
                'peak_day': int(readiness.max()),
                'low_days': int((readiness < 60).sum()),
                'high_days': int((readiness >= 80).sum()),
            }

    # Glucose stats (if CGM)
    glucose_data = [d.get('glucose') for d in week_data if d.get('glucose')]
    if glucose_data:
        avg_glucose = np.mean([g['avg'] for g in glucose_data])
        avg_tir = np.mean([g['time_in_range'] for g in glucose_data])
        stats['glucose'] = {
            'avg': round(avg_glucose, 1),
            'avg_time_in_range': round(avg_tir, 1),
        }

    return stats


def compare_weeks(current_stats: Dict, previous_stats: Dict) -> Dict:
    """Compare current week to previous week."""
    if not current_stats or not previous_stats:
        return {}

    comparisons = {}

    # Sleep comparison
    if 'sleep' in current_stats and 'sleep' in previous_stats:
        curr_sleep = current_stats['sleep'].get('avg_duration', 0)
        prev_sleep = previous_stats['sleep'].get('avg_duration', 0)
        if prev_sleep > 0:
            change = ((curr_sleep - prev_sleep) / prev_sleep) * 100
            comparisons['sleep_duration'] = {
                'change': round(change, 1),
                'direction': 'up' if change > 0 else 'down' if change < 0 else 'same',
                'current': curr_sleep,
                'previous': prev_sleep,
            }

    # HRV comparison
    if 'hrv' in current_stats and 'hrv' in previous_stats:
        curr_hrv = current_stats['hrv'].get('avg', 0)
        prev_hrv = previous_stats['hrv'].get('avg', 0)
        if prev_hrv > 0:
            change = ((curr_hrv - prev_hrv) / prev_hrv) * 100
            comparisons['hrv'] = {
                'change': round(change, 1),
                'direction': 'up' if change > 0 else 'down' if change < 0 else 'same',
                'current': curr_hrv,
                'previous': prev_hrv,
            }

    # Steps comparison
    if 'activity' in current_stats and 'activity' in previous_stats:
        curr_steps = current_stats['activity'].get('avg_daily_steps', 0)
        prev_steps = previous_stats['activity'].get('avg_daily_steps', 0)
        if prev_steps > 0:
            change = ((curr_steps - prev_steps) / prev_steps) * 100
            comparisons['steps'] = {
                'change': round(change, 1),
                'direction': 'up' if change > 0 else 'down' if change < 0 else 'same',
                'current': curr_steps,
                'previous': prev_steps,
            }

    # Readiness comparison
    if 'readiness' in current_stats and 'readiness' in previous_stats:
        curr_ready = current_stats['readiness'].get('avg_score', 0)
        prev_ready = previous_stats['readiness'].get('avg_score', 0)
        if prev_ready > 0:
            change = ((curr_ready - prev_ready) / prev_ready) * 100
            comparisons['readiness'] = {
                'change': round(change, 1),
                'direction': 'up' if change > 0 else 'down' if change < 0 else 'same',
                'current': curr_ready,
                'previous': prev_ready,
            }

    return comparisons


def generate_weekly_insights(stats: Dict, comparisons: Dict) -> List[str]:
    """Generate text insights for the weekly report."""
    insights = []

    # Sleep insights
    if 'sleep' in stats:
        avg_sleep = stats['sleep'].get('avg_duration', 0)
        if avg_sleep < 7:
            insights.append(f"Your average sleep of {avg_sleep:.1f} hours is below the recommended 7-9 hours. Consider earlier bedtimes.")
        elif avg_sleep >= 7.5:
            insights.append(f"Great sleep week! Averaging {avg_sleep:.1f} hours puts you in the optimal range.")

        consistency = stats['sleep'].get('consistency', 0)
        if consistency > 80:
            insights.append(f"Excellent sleep consistency ({consistency:.0f}%). Maintaining regular sleep times supports better recovery.")
        elif consistency < 60:
            insights.append(f"Your sleep timing varied significantly this week ({consistency:.0f}% consistency). Try to keep a more regular schedule.")

    # HRV insights
    if 'hrv' in stats:
        trend = stats['hrv'].get('trend', 'stable')
        if trend == 'improving':
            insights.append("Your HRV trended upward this week, indicating improving recovery and stress resilience.")
        elif trend == 'declining':
            insights.append("Your HRV declined this week. Consider prioritizing rest, reducing alcohol, and managing stress.")

    # Activity insights
    if 'activity' in stats:
        avg_steps = stats['activity'].get('avg_daily_steps', 0)
        days_10k = stats['activity'].get('days_over_10k', 0)

        if days_10k >= 5:
            insights.append(f"Outstanding activity! You hit 10k+ steps on {days_10k} days this week.")
        elif avg_steps < 7000:
            insights.append(f"Activity was lower this week ({avg_steps:,} avg steps). Try adding a daily walk.")

    # Week-over-week comparisons
    if 'hrv' in comparisons:
        hrv_change = comparisons['hrv']['change']
        if hrv_change > 10:
            insights.append(f"Your HRV improved {hrv_change:.0f}% vs last week - great recovery progress!")
        elif hrv_change < -10:
            insights.append(f"Your HRV dropped {abs(hrv_change):.0f}% from last week. Consider a recovery-focused week ahead.")

    if 'steps' in comparisons:
        steps_change = comparisons['steps']['change']
        if steps_change > 20:
            insights.append(f"Activity up {steps_change:.0f}% from last week! Make sure to balance with adequate rest.")
        elif steps_change < -20:
            insights.append(f"Activity down {abs(steps_change):.0f}% from last week.")

    # Readiness insights
    if 'readiness' in stats:
        low_days = stats['readiness'].get('low_days', 0)
        high_days = stats['readiness'].get('high_days', 0)

        if high_days >= 5:
            insights.append(f"Excellent readiness this week with {high_days} high-readiness days. You're well-recovered!")
        elif low_days >= 3:
            insights.append(f"You had {low_days} low-readiness days this week. Consider scaling back intensity.")

    return insights


def generate_weekly_report(
    health_data: List[Dict],
    week_offset: int = 0
) -> Dict:
    """
    Generate a complete weekly report.

    Args:
        health_data: List of daily health summaries
        week_offset: 0 for current week, -1 for last week, etc.

    Returns:
        Complete weekly report dictionary
    """
    # Get week range
    target_date = date.today() + timedelta(weeks=week_offset)
    start_date, end_date = get_week_range(target_date)

    # Get previous week for comparison
    prev_start = start_date - timedelta(days=7)
    prev_end = end_date - timedelta(days=7)

    # Calculate stats
    current_stats = calculate_weekly_stats(health_data, start_date, end_date)
    previous_stats = calculate_weekly_stats(health_data, prev_start, prev_end)

    if not current_stats:
        return {
            'error': 'No data available for this week',
            'date_range': {'start': start_date.isoformat(), 'end': end_date.isoformat()},
        }

    # Compare weeks
    comparisons = compare_weeks(current_stats, previous_stats)

    # Generate insights
    insights = generate_weekly_insights(current_stats, comparisons)

    # Build report
    report = {
        'generated_at': datetime.now().isoformat(),
        'week_of': start_date.isoformat(),
        'date_range': {
            'start': start_date.isoformat(),
            'end': end_date.isoformat(),
        },
        'stats': current_stats,
        'comparisons': comparisons,
        'insights': insights,
        'grade': calculate_week_grade(current_stats),
    }

    return report


def calculate_week_grade(stats: Dict) -> Dict:
    """Calculate overall grade for the week."""
    scores = []
    breakdown = {}

    # Sleep grade (25%)
    if 'sleep' in stats:
        sleep_score = min(100, (stats['sleep'].get('avg_duration', 0) / 8) * 100)
        if 'avg_score' in stats['sleep']:
            sleep_score = (sleep_score + stats['sleep']['avg_score']) / 2
        scores.append(('sleep', sleep_score, 0.25))
        breakdown['sleep'] = round(sleep_score)

    # Activity grade (25%)
    if 'activity' in stats:
        step_score = min(100, (stats['activity'].get('avg_daily_steps', 0) / 10000) * 100)
        scores.append(('activity', step_score, 0.25))
        breakdown['activity'] = round(step_score)

    # Recovery grade (25%)
    if 'readiness' in stats:
        recovery_score = stats['readiness'].get('avg_score', 70)
        scores.append(('recovery', recovery_score, 0.25))
        breakdown['recovery'] = round(recovery_score)

    # Heart health grade (25%)
    if 'hrv' in stats:
        # Assume baseline of 50ms is average
        hrv_score = min(100, (stats['hrv'].get('avg', 40) / 50) * 100)
        scores.append(('heart', hrv_score, 0.25))
        breakdown['heart'] = round(hrv_score)

    if not scores:
        return {'grade': 'N/A', 'score': 0, 'breakdown': {}}

    # Calculate weighted average
    total_weight = sum(w for _, _, w in scores)
    weighted_sum = sum(s * w for _, s, w in scores)
    final_score = weighted_sum / total_weight if total_weight > 0 else 0

    # Convert to letter grade
    if final_score >= 90:
        grade = 'A'
    elif final_score >= 80:
        grade = 'B+'
    elif final_score >= 70:
        grade = 'B'
    elif final_score >= 60:
        grade = 'C'
    else:
        grade = 'D'

    return {
        'grade': grade,
        'score': round(final_score, 1),
        'breakdown': breakdown,
    }
