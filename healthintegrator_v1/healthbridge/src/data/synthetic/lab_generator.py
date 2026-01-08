"""
Synthetic Lab Results Generator

Generates realistic clinical lab values based on patient profile.
"""

from datetime import date, timedelta
from typing import List, Dict, Optional
import random
from dataclasses import dataclass


@dataclass
class LabPanel:
    """A set of lab results from a single blood draw."""
    date: date
    patient_id: str
    results: Dict[str, Dict]  # biomarker -> {value, unit, reference_range, flag}


# Reference ranges (simplified, adult values)
REFERENCE_RANGES = {
    'glucose_fasting': {'unit': 'mg/dL', 'range': (70, 99), 'optimal': (70, 90)},
    'hba1c': {'unit': '%', 'range': (4.0, 5.6), 'optimal': (4.0, 5.4)},
    'cholesterol_total': {'unit': 'mg/dL', 'range': (0, 200), 'optimal': (0, 180)},
    'ldl': {'unit': 'mg/dL', 'range': (0, 100), 'optimal': (0, 70)},
    'hdl': {'unit': 'mg/dL', 'range': (40, 999), 'optimal': (60, 999)},  # Higher is better
    'triglycerides': {'unit': 'mg/dL', 'range': (0, 150), 'optimal': (0, 100)},
    'creatinine': {'unit': 'mg/dL', 'range': (0.7, 1.3), 'optimal': (0.7, 1.2)},
    'egfr': {'unit': 'mL/min', 'range': (90, 999), 'optimal': (90, 999)},
    'alt': {'unit': 'U/L', 'range': (7, 56), 'optimal': (7, 40)},
    'ast': {'unit': 'U/L', 'range': (10, 40), 'optimal': (10, 35)},
    'tsh': {'unit': 'mIU/L', 'range': (0.4, 4.0), 'optimal': (1.0, 2.5)},
    'vitamin_d': {'unit': 'ng/mL', 'range': (30, 100), 'optimal': (40, 60)},
    'b12': {'unit': 'pg/mL', 'range': (200, 900), 'optimal': (400, 800)},
    'iron': {'unit': 'mcg/dL', 'range': (60, 170), 'optimal': (80, 150)},
    'ferritin': {'unit': 'ng/mL', 'range': (20, 200), 'optimal': (50, 150)},
    'cortisol_am': {'unit': 'mcg/dL', 'range': (6, 23), 'optimal': (10, 18)},
    'testosterone_total': {'unit': 'ng/dL', 'range': (300, 1000), 'optimal': (500, 800)},  # Male
    'testosterone_total_f': {'unit': 'ng/dL', 'range': (15, 70), 'optimal': (20, 50)},  # Female
    'crp': {'unit': 'mg/L', 'range': (0, 3), 'optimal': (0, 1)},
    'homocysteine': {'unit': 'umol/L', 'range': (5, 15), 'optimal': (5, 10)},
}


def generate_lab_value(
    biomarker: str,
    health_conditions: List[str],
    age: int,
    sex: str
) -> Optional[Dict]:
    """Generate a single lab value with appropriate variation."""

    ref = REFERENCE_RANGES.get(biomarker)
    if not ref:
        return None

    # Base value in normal range
    low, high = ref['range']
    opt_low, opt_high = ref['optimal']

    # Start with optimal value
    base = random.uniform(opt_low, opt_high)

    # Adjust based on conditions
    if biomarker in ['glucose_fasting', 'hba1c']:
        if 'type_2_diabetes' in health_conditions:
            base = random.uniform(high * 1.2, high * 1.8)  # Above range
        elif 'prediabetes' in health_conditions:
            base = random.uniform(high * 0.95, high * 1.15)  # Borderline

    if biomarker in ['cholesterol_total', 'ldl', 'triglycerides']:
        if 'hyperlipidemia' in health_conditions:
            base = random.uniform(high * 1.1, high * 1.5)

    if biomarker == 'hdl':
        if 'hyperlipidemia' in health_conditions:
            base = random.uniform(low * 0.6, low * 0.9)  # Low HDL

    if biomarker == 'tsh':
        if 'hypothyroid' in health_conditions:
            base = random.uniform(high * 1.2, high * 3)
        elif 'hyperthyroid' in health_conditions:
            base = random.uniform(0.1, low * 0.8)

    if biomarker == 'vitamin_d':
        # Many people are deficient
        if random.random() < 0.4:
            base = random.uniform(15, 29)  # Insufficient

    # Add random variation
    value = base + random.gauss(0, (high - low) * 0.05)
    value = max(0, value)  # No negative values

    # Determine flag
    flag = None
    if value < low:
        flag = 'LOW'
    elif value > high:
        flag = 'HIGH'
    elif value < opt_low or value > opt_high:
        flag = 'BORDERLINE'

    return {
        'value': round(value, 2),
        'unit': ref['unit'],
        'reference_range': f"{low}-{high}",
        'optimal_range': f"{opt_low}-{opt_high}",
        'flag': flag
    }


def generate_comprehensive_panel(
    patient_id: str,
    panel_date: date,
    health_conditions: List[str],
    age: int,
    sex: str
) -> LabPanel:
    """Generate a comprehensive metabolic + wellness panel."""

    biomarkers = [
        'glucose_fasting', 'hba1c',
        'cholesterol_total', 'ldl', 'hdl', 'triglycerides',
        'creatinine', 'egfr', 'alt', 'ast',
        'tsh', 'vitamin_d', 'b12', 'iron', 'ferritin',
        'cortisol_am', 'crp', 'homocysteine'
    ]

    # Add sex-specific markers
    if sex == 'M':
        biomarkers.append('testosterone_total')
    else:
        biomarkers.append('testosterone_total_f')

    results = {}
    for marker in biomarkers:
        result = generate_lab_value(marker, health_conditions, age, sex)
        if result:
            # Normalize name for display
            display_name = marker.replace('_f', '').replace('_', ' ').title()
            results[display_name] = result

    return LabPanel(
        date=panel_date,
        patient_id=patient_id,
        results=results
    )


def generate_lab_history(
    patient_id: str,
    health_conditions: List[str],
    age: int,
    sex: str,
    num_panels: int = 4,
    months_between: int = 3
) -> List[LabPanel]:
    """Generate historical lab panels (e.g., quarterly for a year)."""

    panels = []
    current_date = date.today()

    for i in range(num_panels):
        panel_date = current_date - timedelta(days=i * months_between * 30)
        panel = generate_comprehensive_panel(
            patient_id, panel_date, health_conditions, age, sex
        )
        panels.append(panel)

    return list(reversed(panels))  # Chronological order
