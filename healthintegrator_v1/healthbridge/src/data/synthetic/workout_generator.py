"""
Workout Data Generator

Generates synthetic workout data for demo purposes.
"""

from datetime import date, datetime, timedelta
from typing import List, Dict, Optional
import random
from dataclasses import dataclass
from enum import Enum


class WorkoutType(Enum):
    RUN = "run"
    WALK = "walk"
    CYCLE = "cycle"
    SWIM = "swim"
    STRENGTH = "strength"
    HIIT = "hiit"
    YOGA = "yoga"
    HIKING = "hiking"
    ROWING = "rowing"
    ELLIPTICAL = "elliptical"
    OTHER = "other"


# Workout type configurations
WORKOUT_CONFIGS = {
    WorkoutType.RUN: {
        'name': 'Running',
        'icon': '🏃',
        'duration_range': (20, 90),
        'calories_per_min': (8, 14),
        'hr_zone': (0.7, 0.9),  # % of max HR
        'has_distance': True,
        'pace_range': (6, 12),  # min/km
    },
    WorkoutType.WALK: {
        'name': 'Walking',
        'icon': '🚶',
        'duration_range': (15, 120),
        'calories_per_min': (3, 6),
        'hr_zone': (0.5, 0.65),
        'has_distance': True,
        'pace_range': (10, 18),
    },
    WorkoutType.CYCLE: {
        'name': 'Cycling',
        'icon': '🚴',
        'duration_range': (30, 180),
        'calories_per_min': (6, 12),
        'hr_zone': (0.65, 0.85),
        'has_distance': True,
        'pace_range': (2, 5),  # min/km (faster than running)
    },
    WorkoutType.SWIM: {
        'name': 'Swimming',
        'icon': '🏊',
        'duration_range': (20, 60),
        'calories_per_min': (8, 12),
        'hr_zone': (0.6, 0.8),
        'has_distance': True,
        'pace_range': (15, 30),  # min/km (in water)
    },
    WorkoutType.STRENGTH: {
        'name': 'Strength Training',
        'icon': '🏋️',
        'duration_range': (30, 90),
        'calories_per_min': (4, 8),
        'hr_zone': (0.5, 0.75),
        'has_distance': False,
    },
    WorkoutType.HIIT: {
        'name': 'HIIT',
        'icon': '⚡',
        'duration_range': (15, 45),
        'calories_per_min': (10, 16),
        'hr_zone': (0.8, 0.95),
        'has_distance': False,
    },
    WorkoutType.YOGA: {
        'name': 'Yoga',
        'icon': '🧘',
        'duration_range': (30, 90),
        'calories_per_min': (2, 4),
        'hr_zone': (0.4, 0.6),
        'has_distance': False,
    },
    WorkoutType.HIKING: {
        'name': 'Hiking',
        'icon': '⛰️',
        'duration_range': (60, 300),
        'calories_per_min': (5, 10),
        'hr_zone': (0.55, 0.75),
        'has_distance': True,
        'pace_range': (12, 25),
    },
    WorkoutType.ROWING: {
        'name': 'Rowing',
        'icon': '🚣',
        'duration_range': (20, 60),
        'calories_per_min': (7, 12),
        'hr_zone': (0.65, 0.85),
        'has_distance': True,
        'pace_range': (3, 6),
    },
    WorkoutType.ELLIPTICAL: {
        'name': 'Elliptical',
        'icon': '🔄',
        'duration_range': (20, 60),
        'calories_per_min': (6, 10),
        'hr_zone': (0.6, 0.8),
        'has_distance': False,
    },
}


@dataclass
class Workout:
    """A workout session."""
    id: str
    workout_type: WorkoutType
    date: date
    start_time: datetime
    duration_minutes: int
    calories: int
    avg_hr: Optional[int] = None
    max_hr: Optional[int] = None
    distance_km: Optional[float] = None
    pace_min_km: Optional[float] = None
    elevation_gain: Optional[int] = None
    notes: str = ""
    source: str = "manual"

    @property
    def name(self) -> str:
        return WORKOUT_CONFIGS[self.workout_type]['name']

    @property
    def icon(self) -> str:
        return WORKOUT_CONFIGS[self.workout_type]['icon']

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'workout_type': self.workout_type.value,
            'date': self.date.isoformat(),
            'start_time': self.start_time.isoformat(),
            'duration_minutes': self.duration_minutes,
            'calories': self.calories,
            'avg_hr': self.avg_hr,
            'max_hr': self.max_hr,
            'distance_km': self.distance_km,
            'pace_min_km': self.pace_min_km,
            'elevation_gain': self.elevation_gain,
            'notes': self.notes,
            'source': self.source,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Workout':
        return cls(
            id=data['id'],
            workout_type=WorkoutType(data['workout_type']),
            date=date.fromisoformat(data['date']),
            start_time=datetime.fromisoformat(data['start_time']),
            duration_minutes=data['duration_minutes'],
            calories=data['calories'],
            avg_hr=data.get('avg_hr'),
            max_hr=data.get('max_hr'),
            distance_km=data.get('distance_km'),
            pace_min_km=data.get('pace_min_km'),
            elevation_gain=data.get('elevation_gain'),
            notes=data.get('notes', ''),
            source=data.get('source', 'manual'),
        )


def generate_workout(
    workout_date: date,
    workout_type: WorkoutType = None,
    max_hr: int = 185
) -> Workout:
    """Generate a single synthetic workout."""

    if workout_type is None:
        # Weight towards common workout types
        weights = [0.25, 0.15, 0.15, 0.05, 0.2, 0.1, 0.05, 0.03, 0.01, 0.01]
        workout_type = random.choices(list(WorkoutType)[:10], weights=weights)[0]

    config = WORKOUT_CONFIGS[workout_type]

    # Generate workout details
    duration = random.randint(*config['duration_range'])
    cal_per_min = random.uniform(*config['calories_per_min'])
    calories = int(duration * cal_per_min)

    # Heart rate
    hr_low, hr_high = config['hr_zone']
    avg_hr = int(max_hr * random.uniform(hr_low, hr_high))
    max_hr_workout = int(avg_hr * random.uniform(1.05, 1.15))

    # Start time (morning bias for cardio, evening for strength)
    if workout_type in [WorkoutType.RUN, WorkoutType.CYCLE, WorkoutType.SWIM]:
        hour = random.choice([6, 7, 8, 17, 18, 19])
    elif workout_type == WorkoutType.YOGA:
        hour = random.choice([6, 7, 18, 19, 20])
    else:
        hour = random.choice([6, 7, 12, 17, 18, 19])

    start_time = datetime.combine(workout_date, datetime.min.time().replace(hour=hour, minute=random.randint(0, 59)))

    # Distance and pace for applicable workouts
    distance = None
    pace = None
    if config.get('has_distance'):
        pace_range = config.get('pace_range', (5, 10))
        pace = random.uniform(*pace_range)
        distance = duration / pace  # km

    # Elevation for outdoor activities
    elevation = None
    if workout_type in [WorkoutType.RUN, WorkoutType.CYCLE, WorkoutType.HIKING]:
        elevation = random.randint(10, 500)

    return Workout(
        id=f"workout_{workout_date.strftime('%Y%m%d')}_{random.randint(1000, 9999)}",
        workout_type=workout_type,
        date=workout_date,
        start_time=start_time,
        duration_minutes=duration,
        calories=calories,
        avg_hr=avg_hr,
        max_hr=max_hr_workout,
        distance_km=round(distance, 2) if distance else None,
        pace_min_km=round(pace, 2) if pace else None,
        elevation_gain=elevation,
        source='synthetic',
    )


def generate_workout_history(
    days: int = 90,
    workouts_per_week: float = 4.5,
    max_hr: int = 185
) -> List[Workout]:
    """Generate workout history for a user."""

    workouts = []
    today = date.today()

    # Calculate expected number of workouts
    total_workouts = int((days / 7) * workouts_per_week)

    # Generate random workout dates
    all_dates = [today - timedelta(days=i) for i in range(days)]
    workout_dates = random.sample(all_dates, min(total_workouts, len(all_dates)))

    for workout_date in sorted(workout_dates):
        # Sometimes do multiple workouts per day (rare)
        num_workouts = random.choices([1, 2], weights=[0.95, 0.05])[0]

        for _ in range(num_workouts):
            workout = generate_workout(workout_date, max_hr=max_hr)
            workouts.append(workout)

    return sorted(workouts, key=lambda w: w.start_time)


def calculate_workout_stats(workouts: List[Workout], days: int = 7) -> Dict:
    """Calculate workout statistics for a time period."""

    cutoff = date.today() - timedelta(days=days)
    recent = [w for w in workouts if w.date >= cutoff]

    if not recent:
        return {
            'total_workouts': 0,
            'total_duration': 0,
            'total_calories': 0,
            'total_distance': 0,
            'avg_duration': 0,
            'by_type': {},
        }

    total_duration = sum(w.duration_minutes for w in recent)
    total_calories = sum(w.calories for w in recent)
    total_distance = sum(w.distance_km or 0 for w in recent)

    # By type breakdown
    by_type = {}
    for w in recent:
        t = w.workout_type.value
        if t not in by_type:
            by_type[t] = {'count': 0, 'duration': 0, 'calories': 0}
        by_type[t]['count'] += 1
        by_type[t]['duration'] += w.duration_minutes
        by_type[t]['calories'] += w.calories

    return {
        'total_workouts': len(recent),
        'total_duration': total_duration,
        'total_calories': total_calories,
        'total_distance': round(total_distance, 1),
        'avg_duration': round(total_duration / len(recent), 1) if recent else 0,
        'avg_hr': round(sum(w.avg_hr or 0 for w in recent) / len(recent), 0) if recent else 0,
        'by_type': by_type,
    }


def get_training_load(workouts: List[Workout], days: int = 7) -> Dict:
    """Calculate training load metrics."""

    cutoff = date.today() - timedelta(days=days)
    recent = [w for w in workouts if w.date >= cutoff]

    # Simple training load based on duration * intensity
    load = 0
    for w in recent:
        # Estimate intensity from HR zone
        intensity = (w.avg_hr or 140) / 185  # Normalized to max HR
        load += w.duration_minutes * intensity

    # Compare to previous period
    prev_cutoff = cutoff - timedelta(days=days)
    previous = [w for w in workouts if prev_cutoff <= w.date < cutoff]

    prev_load = 0
    for w in previous:
        intensity = (w.avg_hr or 140) / 185
        prev_load += w.duration_minutes * intensity

    change = ((load - prev_load) / prev_load * 100) if prev_load > 0 else 0

    # Training status
    if change > 20:
        status = 'overreaching'
        color = '#EF4444'
    elif change > 10:
        status = 'productive'
        color = '#10B981'
    elif change < -20:
        status = 'detraining'
        color = '#F59E0B'
    else:
        status = 'maintaining'
        color = '#3B82F6'

    return {
        'current_load': round(load, 0),
        'previous_load': round(prev_load, 0),
        'change_pct': round(change, 1),
        'status': status,
        'status_color': color,
    }
