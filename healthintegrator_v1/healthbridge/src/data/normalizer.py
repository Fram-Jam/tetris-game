"""
Unified Health Data Schema

All data sources (Oura, Apple Health, Whoop, CGM, Labs) must normalize
to these schemas before storage/visualization.
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional, List, Dict, Any
from enum import Enum
import pandas as pd


class DataSource(Enum):
    OURA = "oura"
    APPLE_HEALTH = "apple_health"
    WHOOP = "whoop"
    GARMIN = "garmin"
    FITBIT = "fitbit"
    DEXCOM = "dexcom"
    LIBRE = "libre"
    MANUAL = "manual"
    SYNTHETIC = "synthetic"
    LAB = "lab"
    EHR = "ehr"


class MetricType(Enum):
    # Sleep metrics
    SLEEP_DURATION = "sleep_duration"
    SLEEP_EFFICIENCY = "sleep_efficiency"
    DEEP_SLEEP = "deep_sleep"
    REM_SLEEP = "rem_sleep"
    LIGHT_SLEEP = "light_sleep"
    AWAKE_TIME = "awake_time"
    SLEEP_SCORE = "sleep_score"

    # Heart metrics
    RESTING_HR = "resting_hr"
    HRV = "hrv"
    HRV_RMSSD = "hrv_rmssd"
    MAX_HR = "max_hr"
    AVG_HR = "avg_hr"

    # Activity metrics
    STEPS = "steps"
    CALORIES_ACTIVE = "calories_active"
    CALORIES_TOTAL = "calories_total"
    DISTANCE = "distance"
    FLOORS = "floors"
    ACTIVE_MINUTES = "active_minutes"
    WORKOUT_DURATION = "workout_duration"

    # Recovery/Readiness
    READINESS_SCORE = "readiness_score"
    RECOVERY_SCORE = "recovery_score"
    STRAIN = "strain"

    # Body metrics
    WEIGHT = "weight"
    BODY_FAT = "body_fat"
    BMI = "bmi"
    BODY_TEMP = "body_temp"
    SPO2 = "spo2"
    RESPIRATORY_RATE = "respiratory_rate"

    # Glucose
    GLUCOSE = "glucose"
    GLUCOSE_AVG = "glucose_avg"
    GLUCOSE_VARIABILITY = "glucose_variability"
    TIME_IN_RANGE = "time_in_range"

    # Lab values
    CHOLESTEROL_TOTAL = "cholesterol_total"
    LDL = "ldl"
    HDL = "hdl"
    TRIGLYCERIDES = "triglycerides"
    HBA1C = "hba1c"
    VITAMIN_D = "vitamin_d"
    B12 = "b12"
    IRON = "iron"
    FERRITIN = "ferritin"
    TSH = "tsh"
    CORTISOL = "cortisol"
    TESTOSTERONE = "testosterone"
    CREATININE = "creatinine"
    ALT = "alt"
    AST = "ast"


@dataclass
class HealthMetric:
    """Single health measurement."""
    metric_type: MetricType
    value: float
    unit: str
    timestamp: datetime
    source: DataSource
    confidence: Optional[float] = None  # 0-1 confidence score
    raw_data: Optional[Dict[str, Any]] = None  # Original data for debugging


@dataclass
class DailySummary:
    """Aggregated daily health summary."""
    date: date
    user_id: str

    # Sleep
    sleep_duration_hours: Optional[float] = None
    sleep_efficiency_pct: Optional[float] = None
    deep_sleep_hours: Optional[float] = None
    rem_sleep_hours: Optional[float] = None
    sleep_score: Optional[int] = None

    # Heart
    resting_hr: Optional[int] = None
    hrv_ms: Optional[float] = None

    # Activity
    steps: Optional[int] = None
    calories_active: Optional[int] = None
    active_minutes: Optional[int] = None
    workouts: Optional[List[Dict]] = None

    # Recovery
    readiness_score: Optional[int] = None
    recovery_score: Optional[int] = None

    # Body
    weight_kg: Optional[float] = None
    body_temp_c: Optional[float] = None
    spo2_pct: Optional[float] = None
    respiratory_rate: Optional[float] = None

    # Glucose (if CGM connected)
    glucose_avg: Optional[float] = None
    glucose_min: Optional[float] = None
    glucose_max: Optional[float] = None
    time_in_range_pct: Optional[float] = None

    # Data sources that contributed
    sources: List[DataSource] = field(default_factory=list)

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        """Convert to dictionary for DataFrame creation."""
        return {
            'date': self.date,
            'sleep_duration': self.sleep_duration_hours,
            'sleep_efficiency': self.sleep_efficiency_pct,
            'deep_sleep': self.deep_sleep_hours,
            'rem_sleep': self.rem_sleep_hours,
            'sleep_score': self.sleep_score,
            'resting_hr': self.resting_hr,
            'hrv': self.hrv_ms,
            'steps': self.steps,
            'calories_active': self.calories_active,
            'active_minutes': self.active_minutes,
            'readiness_score': self.readiness_score,
            'recovery_score': self.recovery_score,
            'weight': self.weight_kg,
            'glucose_avg': self.glucose_avg,
            'time_in_range': self.time_in_range_pct,
            'sources': [s.value for s in self.sources]
        }


def normalize_to_daily_summary(
    metrics: List[HealthMetric],
    target_date: date,
    user_id: str = "demo_user"
) -> DailySummary:
    """
    Convert raw metrics into a unified daily summary.
    Handles conflicts by preferring higher-confidence sources.
    """
    summary = DailySummary(date=target_date, user_id=user_id)
    sources = set()

    # Group metrics by type
    by_type: Dict[MetricType, List[HealthMetric]] = {}
    for m in metrics:
        if m.metric_type not in by_type:
            by_type[m.metric_type] = []
        by_type[m.metric_type].append(m)
        sources.add(m.source)

    # For each metric type, pick best value (highest confidence or most recent)
    def best_value(metric_list: List[HealthMetric]) -> Optional[float]:
        if not metric_list:
            return None
        # Sort by confidence (desc), then timestamp (desc)
        sorted_metrics = sorted(
            metric_list,
            key=lambda x: (x.confidence or 0.5, x.timestamp),
            reverse=True
        )
        return sorted_metrics[0].value

    # Map metric types to summary fields
    if MetricType.SLEEP_DURATION in by_type:
        summary.sleep_duration_hours = best_value(by_type[MetricType.SLEEP_DURATION])
    if MetricType.SLEEP_EFFICIENCY in by_type:
        summary.sleep_efficiency_pct = best_value(by_type[MetricType.SLEEP_EFFICIENCY])
    if MetricType.DEEP_SLEEP in by_type:
        summary.deep_sleep_hours = best_value(by_type[MetricType.DEEP_SLEEP])
    if MetricType.REM_SLEEP in by_type:
        summary.rem_sleep_hours = best_value(by_type[MetricType.REM_SLEEP])
    if MetricType.SLEEP_SCORE in by_type:
        summary.sleep_score = int(best_value(by_type[MetricType.SLEEP_SCORE]) or 0)
    if MetricType.RESTING_HR in by_type:
        summary.resting_hr = int(best_value(by_type[MetricType.RESTING_HR]) or 0)
    if MetricType.HRV in by_type:
        summary.hrv_ms = best_value(by_type[MetricType.HRV])
    if MetricType.STEPS in by_type:
        summary.steps = int(best_value(by_type[MetricType.STEPS]) or 0)
    if MetricType.CALORIES_ACTIVE in by_type:
        summary.calories_active = int(best_value(by_type[MetricType.CALORIES_ACTIVE]) or 0)
    if MetricType.ACTIVE_MINUTES in by_type:
        summary.active_minutes = int(best_value(by_type[MetricType.ACTIVE_MINUTES]) or 0)
    if MetricType.READINESS_SCORE in by_type:
        summary.readiness_score = int(best_value(by_type[MetricType.READINESS_SCORE]) or 0)
    if MetricType.RECOVERY_SCORE in by_type:
        summary.recovery_score = int(best_value(by_type[MetricType.RECOVERY_SCORE]) or 0)
    if MetricType.GLUCOSE_AVG in by_type:
        summary.glucose_avg = best_value(by_type[MetricType.GLUCOSE_AVG])
    if MetricType.TIME_IN_RANGE in by_type:
        summary.time_in_range_pct = best_value(by_type[MetricType.TIME_IN_RANGE])

    summary.sources = list(sources)
    return summary


def summaries_to_dataframe(summaries: List[DailySummary]) -> pd.DataFrame:
    """Convert list of daily summaries to pandas DataFrame."""
    return pd.DataFrame([s.to_dict() for s in summaries])
