"""
Tests for Synthetic Data Generation
"""

import pytest
from datetime import date, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.synthetic.patient_generator import (
    generate_patient,
    generate_daily_data,
    generate_date_range,
    generate_demo_data,
    SyntheticPatient
)
from src.data.synthetic.lab_generator import (
    generate_lab_value,
    generate_comprehensive_panel,
    generate_lab_history,
    REFERENCE_RANGES
)


class TestPatientGenerator:
    """Tests for patient generation."""

    def test_generate_patient_creates_valid_patient(self):
        """Test that generate_patient creates a valid patient."""
        patient = generate_patient()

        assert patient.id is not None
        assert patient.name is not None
        assert 18 <= patient.age <= 100
        assert patient.sex in ['M', 'F']
        assert patient.height_cm > 0
        assert patient.weight_kg > 0
        assert patient.activity_level in ['sedentary', 'light', 'moderate', 'active', 'very_active']

    def test_generate_patient_respects_age_range(self):
        """Test that age range is respected."""
        patient = generate_patient(age_range=(30, 40))
        assert 30 <= patient.age <= 40

    def test_generate_patient_with_conditions(self):
        """Test patient generation with specific conditions."""
        conditions = ['type_2_diabetes', 'hypertension']
        patient = generate_patient(conditions=conditions)
        assert patient.health_conditions == conditions

    def test_baseline_calculations(self):
        """Test that baselines are calculated appropriately."""
        patient = generate_patient()

        # RHR should be in realistic range
        assert 45 <= patient.baseline_rhr <= 85

        # HRV should be positive
        assert patient.baseline_hrv > 10

        # Steps should be positive
        assert patient.baseline_steps > 0

        # Glucose should be positive
        assert patient.baseline_glucose > 0


class TestDailyDataGenerator:
    """Tests for daily data generation."""

    def test_generate_daily_data_creates_valid_data(self):
        """Test that daily data is valid."""
        patient = generate_patient()
        data = generate_daily_data(patient, date.today())

        # Check required fields exist
        assert 'date' in data
        assert 'sleep_duration' in data
        assert 'sleep_score' in data
        assert 'hrv' in data
        assert 'resting_hr' in data
        assert 'steps' in data
        assert 'readiness_score' in data

    def test_sleep_duration_realistic(self):
        """Test that sleep duration is realistic."""
        patient = generate_patient()
        data = generate_daily_data(patient, date.today())

        assert 4 <= data['sleep_duration'] <= 10

    def test_sleep_score_in_range(self):
        """Test that sleep score is in valid range."""
        patient = generate_patient()
        data = generate_daily_data(patient, date.today())

        assert 40 <= data['sleep_score'] <= 100

    def test_hrv_positive(self):
        """Test that HRV is positive."""
        patient = generate_patient()
        data = generate_daily_data(patient, date.today())

        assert data['hrv'] > 0

    def test_glucose_data_structure(self):
        """Test glucose data structure when included."""
        patient = generate_patient()
        data = generate_daily_data(patient, date.today(), include_glucose=True)

        assert 'glucose' in data
        assert data['glucose'] is not None
        assert 'avg' in data['glucose']
        assert 'min' in data['glucose']
        assert 'max' in data['glucose']
        assert 'time_in_range' in data['glucose']

    def test_glucose_excluded_when_requested(self):
        """Test that glucose can be excluded."""
        patient = generate_patient()
        data = generate_daily_data(patient, date.today(), include_glucose=False)

        assert data['glucose'] is None


class TestDateRangeGenerator:
    """Tests for date range generation."""

    def test_generate_date_range_correct_length(self):
        """Test that correct number of days are generated."""
        patient = generate_patient()
        start = date.today() - timedelta(days=30)
        end = date.today()

        data = generate_date_range(patient, start, end)

        assert len(data) == 31  # Inclusive of both dates

    def test_dates_are_sequential(self):
        """Test that dates are sequential."""
        patient = generate_patient()
        start = date.today() - timedelta(days=10)
        end = date.today()

        data = generate_date_range(patient, start, end)

        for i in range(1, len(data)):
            assert data[i]['date'] == data[i-1]['date'] + timedelta(days=1)


class TestDemoDataGenerator:
    """Tests for demo data generation."""

    def test_generate_demo_data_returns_tuple(self):
        """Test that generate_demo_data returns patient and data."""
        patient, data = generate_demo_data(days=30)

        assert isinstance(patient, SyntheticPatient)
        assert isinstance(data, list)
        assert len(data) == 31  # 30 days + today

    def test_demo_data_default_days(self):
        """Test default 90 days of data."""
        patient, data = generate_demo_data()

        assert len(data) == 91  # 90 days + today


class TestLabGenerator:
    """Tests for lab data generation."""

    def test_generate_lab_value_valid(self):
        """Test that lab values are valid."""
        result = generate_lab_value('glucose_fasting', [], 35, 'M')

        assert result is not None
        assert 'value' in result
        assert 'unit' in result
        assert 'reference_range' in result

    def test_lab_value_unit_correct(self):
        """Test that units match reference."""
        result = generate_lab_value('glucose_fasting', [], 35, 'M')
        assert result['unit'] == 'mg/dL'

    def test_diabetic_glucose_higher(self):
        """Test that diabetic patients have higher glucose."""
        normal_values = [
            generate_lab_value('glucose_fasting', [], 35, 'M')['value']
            for _ in range(10)
        ]
        diabetic_values = [
            generate_lab_value('glucose_fasting', ['type_2_diabetes'], 35, 'M')['value']
            for _ in range(10)
        ]

        assert sum(diabetic_values) / len(diabetic_values) > sum(normal_values) / len(normal_values)

    def test_generate_comprehensive_panel(self):
        """Test comprehensive panel generation."""
        panel = generate_comprehensive_panel('test123', date.today(), [], 35, 'M')

        assert panel.patient_id == 'test123'
        assert panel.date == date.today()
        assert len(panel.results) > 0

    def test_panel_has_expected_markers(self):
        """Test that panel includes expected biomarkers."""
        panel = generate_comprehensive_panel('test123', date.today(), [], 35, 'M')

        # Check that panel has a reasonable number of biomarkers
        assert len(panel.results) >= 15
        # Check some key markers are present (names are title-cased from snake_case)
        assert 'Cholesterol Total' in panel.results
        assert 'Vitamin D' in panel.results
        assert 'Tsh' in panel.results

    def test_generate_lab_history_length(self):
        """Test lab history generates correct number of panels."""
        history = generate_lab_history('test123', [], 35, 'M', num_panels=4)

        assert len(history) == 4

    def test_lab_history_chronological(self):
        """Test that lab history is in chronological order."""
        history = generate_lab_history('test123', [], 35, 'M', num_panels=4)

        for i in range(1, len(history)):
            assert history[i].date > history[i-1].date


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
