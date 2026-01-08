"""
Anomaly Detection for Health Metrics

Detects unusual patterns in health data that may warrant attention.
"""

from datetime import date, timedelta
from typing import List, Dict, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from enum import Enum


class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ALERT = "alert"


@dataclass
class HealthAlert:
    """A health alert/anomaly detection result."""
    metric: str
    severity: AlertSeverity
    title: str
    description: str
    value: float
    expected_range: Tuple[float, float]
    date: date
    recommendation: Optional[str] = None


# Thresholds for anomaly detection
METRIC_THRESHOLDS = {
    'resting_hr': {
        'normal_range': (45, 80),
        'warning_high': 85,
        'alert_high': 95,
        'warning_low': 40,
        'alert_low': 35,
    },
    'hrv': {
        'min_healthy': 20,
        'drop_threshold': 0.3,  # 30% drop from baseline
    },
    'sleep_duration': {
        'min_recommended': 6,
        'max_recommended': 10,
        'warning_low': 5,
        'alert_low': 4,
    },
    'sleep_score': {
        'warning_low': 60,
        'alert_low': 50,
    },
    'readiness_score': {
        'warning_low': 60,
        'alert_low': 50,
        'consecutive_low_threshold': 3,
    },
    'steps': {
        'sedentary_threshold': 3000,
        'very_low_threshold': 2000,
    },
}


def calculate_baseline(values: List[float], window: int = 14) -> Tuple[float, float]:
    """Calculate baseline mean and std from recent values."""
    if len(values) < 3:
        return np.mean(values), np.std(values) if len(values) > 1 else 0

    recent = values[-window:]
    return np.mean(recent), np.std(recent)


def detect_anomalies(
    health_data: List[Dict],
    days_to_analyze: int = 7
) -> List[HealthAlert]:
    """
    Detect anomalies in recent health data.

    Args:
        health_data: List of daily health summaries
        days_to_analyze: Number of recent days to check

    Returns:
        List of detected health alerts
    """
    if len(health_data) < days_to_analyze:
        return []

    alerts = []
    recent_data = health_data[-days_to_analyze:]
    historical_data = health_data[:-days_to_analyze] if len(health_data) > days_to_analyze else []

    # Check each day
    for day_data in recent_data:
        day = day_data['date']

        # Resting Heart Rate checks
        if day_data.get('resting_hr'):
            rhr = day_data['resting_hr']
            thresholds = METRIC_THRESHOLDS['resting_hr']

            if rhr >= thresholds['alert_high']:
                alerts.append(HealthAlert(
                    metric='resting_hr',
                    severity=AlertSeverity.ALERT,
                    title='Very High Resting Heart Rate',
                    description=f'Your resting heart rate of {rhr} bpm is significantly elevated.',
                    value=rhr,
                    expected_range=thresholds['normal_range'],
                    date=day,
                    recommendation='This could indicate illness, stress, dehydration, or overtraining. Consider rest and consult a doctor if persistent.'
                ))
            elif rhr >= thresholds['warning_high']:
                alerts.append(HealthAlert(
                    metric='resting_hr',
                    severity=AlertSeverity.WARNING,
                    title='Elevated Resting Heart Rate',
                    description=f'Your resting heart rate of {rhr} bpm is higher than normal.',
                    value=rhr,
                    expected_range=thresholds['normal_range'],
                    date=day,
                    recommendation='Monitor for signs of illness or stress. Ensure adequate hydration and rest.'
                ))
            elif rhr <= thresholds['alert_low']:
                alerts.append(HealthAlert(
                    metric='resting_hr',
                    severity=AlertSeverity.WARNING,
                    title='Very Low Resting Heart Rate',
                    description=f'Your resting heart rate of {rhr} bpm is quite low.',
                    value=rhr,
                    expected_range=thresholds['normal_range'],
                    date=day,
                    recommendation='This may be normal for athletes, but consult a doctor if you experience dizziness or fatigue.'
                ))

        # HRV checks
        if day_data.get('hrv') and historical_data:
            hrv = day_data['hrv']
            historical_hrv = [d.get('hrv') for d in historical_data if d.get('hrv')]

            if historical_hrv:
                baseline_mean, baseline_std = calculate_baseline(historical_hrv)
                drop_threshold = baseline_mean * (1 - METRIC_THRESHOLDS['hrv']['drop_threshold'])

                if hrv < METRIC_THRESHOLDS['hrv']['min_healthy']:
                    alerts.append(HealthAlert(
                        metric='hrv',
                        severity=AlertSeverity.ALERT,
                        title='Very Low HRV',
                        description=f'Your HRV of {hrv:.0f}ms is very low.',
                        value=hrv,
                        expected_range=(METRIC_THRESHOLDS['hrv']['min_healthy'], baseline_mean + baseline_std),
                        date=day,
                        recommendation='Low HRV indicates stress or poor recovery. Prioritize sleep, reduce intensity, and manage stress.'
                    ))
                elif hrv < drop_threshold:
                    alerts.append(HealthAlert(
                        metric='hrv',
                        severity=AlertSeverity.WARNING,
                        title='HRV Drop Detected',
                        description=f'Your HRV of {hrv:.0f}ms is {((baseline_mean - hrv) / baseline_mean * 100):.0f}% below your baseline.',
                        value=hrv,
                        expected_range=(drop_threshold, baseline_mean + baseline_std),
                        date=day,
                        recommendation='Consider a recovery day. Check sleep quality and stress levels.'
                    ))

        # Sleep checks
        if day_data.get('sleep_duration'):
            sleep = day_data['sleep_duration']
            thresholds = METRIC_THRESHOLDS['sleep_duration']

            if sleep < thresholds['alert_low']:
                alerts.append(HealthAlert(
                    metric='sleep_duration',
                    severity=AlertSeverity.ALERT,
                    title='Severely Low Sleep',
                    description=f'You only slept {sleep:.1f} hours.',
                    value=sleep,
                    expected_range=(thresholds['min_recommended'], thresholds['max_recommended']),
                    date=day,
                    recommendation='Chronic sleep deprivation has serious health consequences. Prioritize sleep tonight.'
                ))
            elif sleep < thresholds['warning_low']:
                alerts.append(HealthAlert(
                    metric='sleep_duration',
                    severity=AlertSeverity.WARNING,
                    title='Insufficient Sleep',
                    description=f'You slept {sleep:.1f} hours, below the recommended minimum.',
                    value=sleep,
                    expected_range=(thresholds['min_recommended'], thresholds['max_recommended']),
                    date=day,
                    recommendation='Try to get to bed earlier tonight to recover.'
                ))

        # Sleep score checks
        if day_data.get('sleep_score'):
            score = day_data['sleep_score']
            thresholds = METRIC_THRESHOLDS['sleep_score']

            if score < thresholds['alert_low']:
                alerts.append(HealthAlert(
                    metric='sleep_score',
                    severity=AlertSeverity.WARNING,
                    title='Poor Sleep Quality',
                    description=f'Your sleep score of {score} indicates very poor sleep quality.',
                    value=score,
                    expected_range=(70, 100),
                    date=day,
                    recommendation='Review sleep environment: temperature, light, noise. Avoid screens before bed.'
                ))

        # Activity checks
        if day_data.get('steps') is not None:
            steps = day_data['steps']
            thresholds = METRIC_THRESHOLDS['steps']

            if steps < thresholds['very_low_threshold']:
                alerts.append(HealthAlert(
                    metric='steps',
                    severity=AlertSeverity.INFO,
                    title='Very Low Activity Day',
                    description=f'You only logged {steps:,} steps.',
                    value=steps,
                    expected_range=(7000, 15000),
                    date=day,
                    recommendation='Try to incorporate more movement tomorrow. Even light walking helps.'
                ))

    # Check for patterns across multiple days
    alerts.extend(_detect_pattern_anomalies(recent_data, historical_data))

    # Sort by severity and date
    severity_order = {AlertSeverity.ALERT: 0, AlertSeverity.WARNING: 1, AlertSeverity.INFO: 2}
    alerts.sort(key=lambda a: (severity_order[a.severity], a.date), reverse=True)

    return alerts


def _detect_pattern_anomalies(
    recent_data: List[Dict],
    historical_data: List[Dict]
) -> List[HealthAlert]:
    """Detect patterns across multiple days."""
    alerts = []

    # Check for consecutive low readiness
    readiness_scores = [d.get('readiness_score') for d in recent_data if d.get('readiness_score')]
    threshold = METRIC_THRESHOLDS['readiness_score']

    low_count = sum(1 for s in readiness_scores if s < threshold['warning_low'])
    if low_count >= threshold['consecutive_low_threshold']:
        alerts.append(HealthAlert(
            metric='readiness_score',
            severity=AlertSeverity.WARNING,
            title='Sustained Low Readiness',
            description=f'You\'ve had {low_count} low-readiness days this week.',
            value=np.mean(readiness_scores) if readiness_scores else 0,
            expected_range=(70, 100),
            date=recent_data[-1]['date'],
            recommendation='Your body needs recovery. Consider reducing training load and prioritizing rest.'
        ))

    # Check for declining HRV trend
    hrv_values = [d.get('hrv') for d in recent_data if d.get('hrv')]
    if len(hrv_values) >= 5:
        first_half = np.mean(hrv_values[:len(hrv_values)//2])
        second_half = np.mean(hrv_values[len(hrv_values)//2:])

        if first_half > 0 and (second_half - first_half) / first_half < -0.15:
            alerts.append(HealthAlert(
                metric='hrv',
                severity=AlertSeverity.WARNING,
                title='Declining HRV Trend',
                description='Your HRV has been trending downward over the past week.',
                value=hrv_values[-1],
                expected_range=(first_half * 0.9, first_half * 1.1),
                date=recent_data[-1]['date'],
                recommendation='This could indicate accumulated stress or fatigue. Plan for recovery.'
            ))

    # Check for sleep debt
    sleep_values = [d.get('sleep_duration', 0) for d in recent_data]
    if sleep_values:
        avg_sleep = np.mean(sleep_values)
        sleep_debt = sum(max(0, 7 - s) for s in sleep_values)

        if sleep_debt > 7:  # More than 7 hours of sleep debt
            alerts.append(HealthAlert(
                metric='sleep_duration',
                severity=AlertSeverity.WARNING,
                title='Accumulated Sleep Debt',
                description=f'You\'ve accumulated about {sleep_debt:.0f} hours of sleep debt this week.',
                value=avg_sleep,
                expected_range=(7, 9),
                date=recent_data[-1]['date'],
                recommendation='Try to pay back sleep debt gradually over the next few days with extra sleep.'
            ))

    return alerts


def get_metric_status(
    current_value: float,
    metric_name: str,
    historical_values: List[float] = None
) -> Dict:
    """
    Get status assessment for a single metric value.

    Returns status dict with 'status' (good/warning/alert), 'message', etc.
    """
    if metric_name not in METRIC_THRESHOLDS:
        return {'status': 'unknown', 'message': 'No thresholds defined for this metric'}

    thresholds = METRIC_THRESHOLDS[metric_name]

    # Calculate baseline if historical data provided
    baseline_mean = None
    baseline_std = None
    if historical_values and len(historical_values) >= 3:
        baseline_mean, baseline_std = calculate_baseline(historical_values)

    result = {
        'value': current_value,
        'status': 'good',
        'message': 'Within normal range',
        'baseline': baseline_mean,
    }

    # Check against thresholds
    if 'normal_range' in thresholds:
        low, high = thresholds['normal_range']
        if current_value < low:
            result['status'] = 'warning'
            result['message'] = f'Below normal range ({low}-{high})'
        elif current_value > high:
            result['status'] = 'warning'
            result['message'] = f'Above normal range ({low}-{high})'

    if 'alert_high' in thresholds and current_value >= thresholds['alert_high']:
        result['status'] = 'alert'
        result['message'] = 'Significantly elevated'

    if 'alert_low' in thresholds and current_value <= thresholds['alert_low']:
        result['status'] = 'alert'
        result['message'] = 'Significantly low'

    # Check against personal baseline
    if baseline_mean and baseline_std:
        z_score = (current_value - baseline_mean) / baseline_std if baseline_std > 0 else 0
        if abs(z_score) > 2:
            result['deviation'] = 'significant'
            result['z_score'] = round(z_score, 2)
        elif abs(z_score) > 1:
            result['deviation'] = 'moderate'
            result['z_score'] = round(z_score, 2)

    return result
