"""
Synthetic Health Data Generator

Generates realistic health data for demo purposes.
Based on published population norms and physiological relationships.
"""

import random
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Tuple
import numpy as np
from faker import Faker
from dataclasses import dataclass

fake = Faker()


@dataclass
class SyntheticPatient:
    """A synthetic patient profile."""
    id: str
    name: str
    age: int
    sex: str  # 'M' or 'F'
    height_cm: float
    weight_kg: float
    activity_level: str  # 'sedentary', 'light', 'moderate', 'active', 'very_active'
    health_conditions: List[str]

    # Derived baselines (these affect generated data)
    baseline_rhr: int = 65
    baseline_hrv: float = 45.0
    baseline_sleep_hours: float = 7.5
    baseline_steps: int = 8000
    baseline_glucose: float = 95.0

    def __post_init__(self):
        """Calculate physiologically appropriate baselines."""
        # RHR varies by age, sex, and fitness
        age_factor = max(0, (self.age - 30) * 0.3)  # Increases ~0.3 bpm per year after 30
        fitness_factor = {
            'sedentary': 10, 'light': 5, 'moderate': 0,
            'active': -5, 'very_active': -10
        }.get(self.activity_level, 0)
        sex_factor = 3 if self.sex == 'F' else 0  # Women slightly higher RHR

        self.baseline_rhr = int(65 + age_factor + fitness_factor + sex_factor + random.gauss(0, 3))
        self.baseline_rhr = max(45, min(85, self.baseline_rhr))  # Clamp to realistic range

        # HRV decreases with age, increases with fitness
        age_hrv_factor = max(10, 60 - (self.age - 25) * 0.8)
        fitness_hrv_factor = {
            'sedentary': -10, 'light': -5, 'moderate': 0,
            'active': 10, 'very_active': 20
        }.get(self.activity_level, 0)

        self.baseline_hrv = max(15, age_hrv_factor + fitness_hrv_factor + random.gauss(0, 5))

        # Steps based on activity level
        self.baseline_steps = {
            'sedentary': 3000, 'light': 5000, 'moderate': 8000,
            'active': 12000, 'very_active': 15000
        }.get(self.activity_level, 8000) + random.randint(-1000, 1000)

        # Glucose baseline (higher if diabetic/prediabetic)
        if 'type_2_diabetes' in self.health_conditions:
            self.baseline_glucose = random.uniform(140, 180)
        elif 'prediabetes' in self.health_conditions:
            self.baseline_glucose = random.uniform(100, 125)
        else:
            self.baseline_glucose = random.uniform(85, 99)


def generate_patient(
    age_range: Tuple[int, int] = (25, 65),
    conditions: Optional[List[str]] = None
) -> SyntheticPatient:
    """Generate a random synthetic patient."""

    sex = random.choice(['M', 'F'])
    age = random.randint(*age_range)

    # Height/weight based on sex (US population distributions)
    if sex == 'M':
        height_cm = random.gauss(175.3, 7.6)
        bmi = random.gauss(26.6, 4.5)
    else:
        height_cm = random.gauss(161.3, 7.1)
        bmi = random.gauss(26.5, 5.5)

    weight_kg = bmi * (height_cm / 100) ** 2

    activity_level = random.choices(
        ['sedentary', 'light', 'moderate', 'active', 'very_active'],
        weights=[0.25, 0.35, 0.25, 0.10, 0.05]  # US population distribution
    )[0]

    # Health conditions (if not specified, randomly assign based on prevalence)
    if conditions is None:
        conditions = []
        if random.random() < 0.11:  # ~11% diabetes prevalence
            conditions.append('type_2_diabetes')
        elif random.random() < 0.38:  # ~38% prediabetes
            conditions.append('prediabetes')
        if random.random() < 0.47:  # ~47% hypertension
            conditions.append('hypertension')
        if random.random() < 0.20:  # ~20% sleep issues
            conditions.append('sleep_disorder')

    return SyntheticPatient(
        id=fake.uuid4()[:8],
        name=fake.name(),
        age=age,
        sex=sex,
        height_cm=round(height_cm, 1),
        weight_kg=round(weight_kg, 1),
        activity_level=activity_level,
        health_conditions=conditions
    )


def generate_daily_data(
    patient: SyntheticPatient,
    target_date: date,
    include_glucose: bool = True
) -> Dict:
    """
    Generate realistic daily health metrics for a patient.

    Incorporates:
    - Day-of-week effects (weekends = more sleep, less activity)
    - Physiological correlations (poor sleep -> lower HRV next day)
    - Random variation within realistic bounds
    """

    is_weekend = target_date.weekday() >= 5

    # Sleep metrics
    sleep_base = patient.baseline_sleep_hours
    if is_weekend:
        sleep_base += random.uniform(0.5, 1.5)  # Sleep in on weekends
    if 'sleep_disorder' in patient.health_conditions:
        sleep_base -= random.uniform(0.5, 2.0)

    sleep_duration = max(4, min(10, sleep_base + random.gauss(0, 0.7)))
    sleep_efficiency = max(70, min(98, 85 + random.gauss(0, 5)))

    # Sleep stages (should sum to ~sleep_duration)
    deep_pct = random.uniform(0.13, 0.23)  # 13-23% is normal
    rem_pct = random.uniform(0.20, 0.25)  # 20-25% is normal
    light_pct = 1 - deep_pct - rem_pct - 0.05  # Rest is light + small awake

    deep_sleep = sleep_duration * deep_pct
    rem_sleep = sleep_duration * rem_pct
    light_sleep = sleep_duration * light_pct

    # Sleep score (composite)
    sleep_score = int(
        (sleep_efficiency * 0.4) +
        (min(sleep_duration / 8, 1) * 100 * 0.3) +
        (min(deep_sleep / 1.5, 1) * 100 * 0.15) +
        (min(rem_sleep / 2, 1) * 100 * 0.15)
    )
    sleep_score = max(40, min(100, sleep_score + random.randint(-5, 5)))

    # Heart metrics (affected by previous night's sleep)
    sleep_quality_factor = (sleep_score - 70) / 30  # -1 to +1

    resting_hr = patient.baseline_rhr - int(sleep_quality_factor * 3) + random.randint(-3, 3)
    resting_hr = max(40, min(100, resting_hr))

    hrv = patient.baseline_hrv * (1 + sleep_quality_factor * 0.15) + random.gauss(0, 3)
    hrv = max(10, min(120, hrv))

    # Activity metrics
    steps_base = patient.baseline_steps
    if is_weekend:
        steps_base *= random.uniform(0.7, 1.3)  # More variable on weekends

    steps = int(max(500, steps_base + random.gauss(0, steps_base * 0.2)))

    # Calories based on steps and baseline metabolic rate
    bmr = 10 * patient.weight_kg + 6.25 * patient.height_cm - 5 * patient.age
    if patient.sex == 'M':
        bmr += 5
    else:
        bmr -= 161

    calories_active = int(steps * 0.04 + random.randint(-50, 50))
    calories_total = int(bmr + calories_active)

    active_minutes = int(steps / 100 + random.randint(-10, 20))
    active_minutes = max(0, min(180, active_minutes))

    # Readiness/Recovery score (composite of sleep + HRV + RHR trends)
    readiness_score = int(
        sleep_score * 0.4 +
        min(hrv / patient.baseline_hrv, 1.2) * 100 * 0.35 +
        max(0, 1 - (resting_hr - patient.baseline_rhr) / 10) * 100 * 0.25
    )
    readiness_score = max(30, min(100, readiness_score + random.randint(-5, 5)))

    # Glucose data (if CGM connected)
    glucose_data = None
    if include_glucose:
        glucose_base = patient.baseline_glucose

        # Generate 24-hour glucose curve (simplified)
        # Meals cause spikes, overnight is stable
        glucose_readings = []
        for hour in range(24):
            if hour in [7, 8]:  # Breakfast spike
                glucose = glucose_base + random.uniform(20, 40)
            elif hour in [12, 13]:  # Lunch spike
                glucose = glucose_base + random.uniform(15, 35)
            elif hour in [18, 19]:  # Dinner spike
                glucose = glucose_base + random.uniform(25, 50)
            elif hour in [0, 1, 2, 3, 4, 5]:  # Overnight
                glucose = glucose_base + random.uniform(-10, 5)
            else:
                glucose = glucose_base + random.uniform(-5, 15)

            glucose_readings.append(glucose)

        glucose_avg = np.mean(glucose_readings)
        glucose_min = min(glucose_readings)
        glucose_max = max(glucose_readings)

        # Time in range (70-180 mg/dL)
        in_range = sum(1 for g in glucose_readings if 70 <= g <= 180)
        time_in_range = (in_range / len(glucose_readings)) * 100

        glucose_data = {
            'avg': round(glucose_avg, 1),
            'min': round(glucose_min, 1),
            'max': round(glucose_max, 1),
            'time_in_range': round(time_in_range, 1),
            'readings': glucose_readings
        }

    return {
        'date': target_date,
        'patient_id': patient.id,

        # Sleep
        'sleep_duration': round(sleep_duration, 2),
        'sleep_efficiency': round(sleep_efficiency, 1),
        'deep_sleep': round(deep_sleep, 2),
        'rem_sleep': round(rem_sleep, 2),
        'light_sleep': round(light_sleep, 2),
        'sleep_score': sleep_score,

        # Heart
        'resting_hr': resting_hr,
        'hrv': round(hrv, 1),

        # Activity
        'steps': steps,
        'calories_active': calories_active,
        'calories_total': calories_total,
        'active_minutes': active_minutes,

        # Recovery
        'readiness_score': readiness_score,

        # Glucose
        'glucose': glucose_data,

        # Meta
        'sources': ['synthetic']
    }


def generate_date_range(
    patient: SyntheticPatient,
    start_date: date,
    end_date: date,
    include_glucose: bool = True
) -> List[Dict]:
    """Generate health data for a date range."""
    data = []
    current = start_date
    while current <= end_date:
        daily = generate_daily_data(patient, current, include_glucose)
        data.append(daily)
        current += timedelta(days=1)
    return data


# Convenience function for demo
def generate_demo_data(days: int = 90) -> Tuple[SyntheticPatient, List[Dict]]:
    """Generate a demo patient with 90 days of data."""
    patient = generate_patient(age_range=(30, 45), conditions=[])
    end_date = date.today()
    start_date = end_date - timedelta(days=days)
    data = generate_date_range(patient, start_date, end_date)
    return patient, data


def generate_synthetic_patient(days: int = 90) -> Dict:
    """
    Generate a synthetic patient with health data.

    Returns a dictionary with patient info and daily_summaries.
    This is the preferred interface for UI components.
    """
    patient = generate_patient(age_range=(30, 50), conditions=[])
    end_date = date.today()
    start_date = end_date - timedelta(days=days)
    data = generate_date_range(patient, start_date, end_date)

    return {
        'patient': {
            'id': patient.id,
            'name': patient.name,
            'age': patient.age,
            'sex': patient.sex,
            'activity_level': patient.activity_level,
        },
        'daily_summaries': data,
    }
