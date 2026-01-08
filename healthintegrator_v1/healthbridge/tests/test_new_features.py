"""
Tests for new features: Reports, Goals, Anomaly Detection, Workouts
"""

import pytest
from datetime import date, datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.synthetic.patient_generator import generate_synthetic_patient
from src.data.synthetic.workout_generator import (
    generate_workout, generate_workout_history, calculate_workout_stats,
    get_training_load, WorkoutType, Workout
)
from src.insights.weekly_report import (
    generate_weekly_report, calculate_weekly_stats, get_week_range
)
from src.insights.anomaly_detection import (
    detect_anomalies, get_metric_status, AlertSeverity
)
from src.insights.goals import (
    Goal, GoalCategory, GoalFrequency, GoalStatus,
    create_goal_from_template, evaluate_goals, suggest_goals
)


class TestWeeklyReport:
    """Tests for weekly report generation."""

    @pytest.fixture
    def health_data(self):
        patient = generate_synthetic_patient(days=30)
        return patient['daily_summaries']

    def test_get_week_range(self):
        start, end = get_week_range()
        assert isinstance(start, date)
        assert isinstance(end, date)
        assert (end - start).days == 6

    def test_generate_weekly_report_structure(self, health_data):
        report = generate_weekly_report(health_data, week_offset=0)
        assert 'date_range' in report
        assert 'stats' in report or 'error' in report

    def test_calculate_weekly_stats(self, health_data):
        start, end = get_week_range()
        stats = calculate_weekly_stats(health_data, start, end)
        if stats:
            assert 'days_with_data' in stats

    def test_report_with_offset(self, health_data):
        report = generate_weekly_report(health_data, week_offset=-1)
        assert 'date_range' in report


class TestAnomalyDetection:
    """Tests for anomaly detection."""

    @pytest.fixture
    def health_data(self):
        patient = generate_synthetic_patient(days=30)
        return patient['daily_summaries']

    def test_detect_anomalies_returns_list(self, health_data):
        alerts = detect_anomalies(health_data, days_to_analyze=7)
        assert isinstance(alerts, list)

    def test_alert_structure(self, health_data):
        alerts = detect_anomalies(health_data, days_to_analyze=7)
        for alert in alerts:
            assert hasattr(alert, 'metric')
            assert hasattr(alert, 'severity')
            assert hasattr(alert, 'title')
            assert hasattr(alert, 'description')
            assert isinstance(alert.severity, AlertSeverity)

    def test_get_metric_status_valid_metric(self):
        status = get_metric_status(75, 'resting_hr')
        assert 'status' in status
        assert status['status'] in ['good', 'warning', 'alert']

    def test_get_metric_status_unknown_metric(self):
        status = get_metric_status(100, 'unknown_metric')
        assert status['status'] == 'unknown'


class TestGoals:
    """Tests for goal tracking system."""

    def test_create_goal_from_template(self):
        goal = create_goal_from_template('sleep_7h')
        assert goal.name == 'Get 7+ Hours of Sleep'
        assert goal.target_value == 7.0
        assert goal.category == GoalCategory.SLEEP

    def test_goal_check_achievement_gte(self):
        goal = Goal(
            id='test',
            name='Test Goal',
            category=GoalCategory.SLEEP,
            metric='sleep_duration',
            target_value=7.0,
            comparison='gte'
        )
        assert goal.check_achievement(7.5) == True
        assert goal.check_achievement(6.5) == False

    def test_goal_check_achievement_lte(self):
        goal = Goal(
            id='test',
            name='Test Goal',
            category=GoalCategory.HEART,
            metric='resting_hr',
            target_value=60,
            comparison='lte'
        )
        assert goal.check_achievement(55) == True
        assert goal.check_achievement(65) == False

    def test_goal_progress_tracking(self):
        goal = Goal(
            id='test',
            name='Test Goal',
            category=GoalCategory.ACTIVITY,
            metric='steps',
            target_value=10000,
            comparison='gte'
        )
        goal.update_progress(8000, False)
        assert goal.total_attempts == 1
        assert goal.times_achieved == 0
        assert goal.streak == 0

        goal.update_progress(12000, True)
        assert goal.total_attempts == 2
        assert goal.times_achieved == 1
        assert goal.streak == 1

    def test_evaluate_goals(self):
        health_data = [{'date': date.today(), 'sleep_duration': 7.5, 'steps': 8000}]
        goals = [create_goal_from_template('sleep_7h')]
        results = evaluate_goals(goals, health_data)
        assert len(results) == 1
        assert results[0]['achieved'] == True

    def test_suggest_goals(self):
        patient = generate_synthetic_patient(days=14)
        suggestions = suggest_goals(patient['daily_summaries'])
        assert isinstance(suggestions, list)


class TestWorkouts:
    """Tests for workout tracking."""

    def test_generate_workout(self):
        workout = generate_workout(date.today())
        assert isinstance(workout, Workout)
        assert workout.duration_minutes > 0
        assert workout.calories > 0

    def test_generate_workout_specific_type(self):
        workout = generate_workout(date.today(), workout_type=WorkoutType.RUN)
        assert workout.workout_type == WorkoutType.RUN
        assert workout.distance_km is not None  # Running has distance

    def test_generate_workout_history(self):
        workouts = generate_workout_history(days=30, workouts_per_week=3)
        assert len(workouts) > 0
        assert all(isinstance(w, Workout) for w in workouts)

    def test_workout_to_dict(self):
        workout = generate_workout(date.today())
        d = workout.to_dict()
        assert 'id' in d
        assert 'workout_type' in d
        assert 'duration_minutes' in d

    def test_workout_from_dict(self):
        workout = generate_workout(date.today())
        d = workout.to_dict()
        restored = Workout.from_dict(d)
        assert restored.id == workout.id
        assert restored.workout_type == workout.workout_type

    def test_calculate_workout_stats(self):
        workouts = generate_workout_history(days=14, workouts_per_week=4)
        stats = calculate_workout_stats(workouts, days=7)
        assert 'total_workouts' in stats
        assert 'total_duration' in stats
        assert 'total_calories' in stats

    def test_get_training_load(self):
        workouts = generate_workout_history(days=21, workouts_per_week=4)
        load = get_training_load(workouts, days=7)
        assert 'current_load' in load
        assert 'status' in load
        assert load['status'] in ['overreaching', 'productive', 'maintaining', 'detraining']


class TestCSVImporter:
    """Tests for CSV data importer."""

    def test_column_mapping(self):
        from src.data.connectors.csv_importer import find_column_match, COLUMN_MAPPINGS

        # Test exact match
        assert find_column_match(['date', 'sleep'], 'date') == 'date'

        # Test case-insensitive
        assert find_column_match(['Date', 'Sleep'], 'date') == 'Date'

    def test_date_column_detection(self):
        import pandas as pd
        from src.data.connectors.csv_importer import detect_date_column

        df = pd.DataFrame({
            'date': ['2024-01-01', '2024-01-02'],
            'value': [1, 2]
        })
        assert detect_date_column(df) == 'date'


class TestAppleHealthParser:
    """Tests for Apple Health parser."""

    def test_record_types_mapping(self):
        from src.data.connectors.apple_health_parser import RECORD_TYPES

        assert 'HKQuantityTypeIdentifierStepCount' in RECORD_TYPES
        assert RECORD_TYPES['HKQuantityTypeIdentifierStepCount'] == 'steps'

    def test_sleep_values_mapping(self):
        from src.data.connectors.apple_health_parser import SLEEP_VALUES

        assert 'HKCategoryValueSleepAnalysisAsleepDeep' in SLEEP_VALUES
        assert SLEEP_VALUES['HKCategoryValueSleepAnalysisAsleepDeep'] == 'deep'


class TestCalendarHeatmap:
    """Tests for calendar heatmap visualization."""

    def test_create_calendar_heatmap(self):
        import pandas as pd
        from src.visualizations.charts import create_calendar_heatmap

        # Create test data
        dates = [date.today() - timedelta(days=i) for i in range(30)]
        df = pd.DataFrame({
            'date': dates,
            'value': [i % 10 for i in range(30)]
        })

        fig = create_calendar_heatmap(df, 'date', 'value')
        assert fig is not None
