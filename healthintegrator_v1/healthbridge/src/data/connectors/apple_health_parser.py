"""
Apple Health XML Export Parser

Parses Apple Health export.xml files to extract health metrics.
Apple Health exports can be generated from: Settings > Health > Export All Health Data
"""

import xml.etree.ElementTree as ET
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Tuple, Generator
from collections import defaultdict
import zipfile
import os
import tempfile


# Apple Health record type mappings
RECORD_TYPES = {
    # Sleep
    'HKCategoryTypeIdentifierSleepAnalysis': 'sleep',

    # Heart
    'HKQuantityTypeIdentifierHeartRate': 'heart_rate',
    'HKQuantityTypeIdentifierRestingHeartRate': 'resting_hr',
    'HKQuantityTypeIdentifierHeartRateVariabilitySDNN': 'hrv',

    # Activity
    'HKQuantityTypeIdentifierStepCount': 'steps',
    'HKQuantityTypeIdentifierDistanceWalkingRunning': 'distance',
    'HKQuantityTypeIdentifierActiveEnergyBurned': 'calories_active',
    'HKQuantityTypeIdentifierBasalEnergyBurned': 'calories_basal',
    'HKQuantityTypeIdentifierAppleExerciseTime': 'exercise_minutes',
    'HKQuantityTypeIdentifierFlightsClimbed': 'floors',

    # Body
    'HKQuantityTypeIdentifierBodyMass': 'weight',
    'HKQuantityTypeIdentifierBodyMassIndex': 'bmi',
    'HKQuantityTypeIdentifierBodyFatPercentage': 'body_fat',
    'HKQuantityTypeIdentifierHeight': 'height',

    # Vitals
    'HKQuantityTypeIdentifierOxygenSaturation': 'spo2',
    'HKQuantityTypeIdentifierRespiratoryRate': 'respiratory_rate',
    'HKQuantityTypeIdentifierBodyTemperature': 'body_temp',
    'HKQuantityTypeIdentifierBloodPressureSystolic': 'bp_systolic',
    'HKQuantityTypeIdentifierBloodPressureDiastolic': 'bp_diastolic',

    # Workouts
    'HKWorkoutTypeIdentifier': 'workout',
}

# Sleep value mappings (Apple uses numeric codes)
SLEEP_VALUES = {
    'HKCategoryValueSleepAnalysisInBed': 'in_bed',
    'HKCategoryValueSleepAnalysisAsleep': 'asleep',
    'HKCategoryValueSleepAnalysisAsleepCore': 'core',
    'HKCategoryValueSleepAnalysisAsleepDeep': 'deep',
    'HKCategoryValueSleepAnalysisAsleepREM': 'rem',
    'HKCategoryValueSleepAnalysisAwake': 'awake',
}


class AppleHealthParser:
    """Parser for Apple Health export.xml files."""

    def __init__(self, file_path: str):
        """
        Initialize parser with path to export.xml or export.zip.

        Args:
            file_path: Path to export.xml or export.zip file
        """
        self.file_path = file_path
        self.temp_dir = None
        self.xml_path = self._get_xml_path()

    def _get_xml_path(self) -> str:
        """Get path to XML file, extracting from zip if needed."""
        if self.file_path.endswith('.zip'):
            self.temp_dir = tempfile.mkdtemp()
            with zipfile.ZipFile(self.file_path, 'r') as zip_ref:
                # Find export.xml in the zip
                for name in zip_ref.namelist():
                    if name.endswith('export.xml'):
                        zip_ref.extract(name, self.temp_dir)
                        return os.path.join(self.temp_dir, name)
            raise ValueError("No export.xml found in zip file")
        return self.file_path

    def _parse_date(self, date_str: str) -> datetime:
        """Parse Apple Health date format."""
        # Format: 2024-01-15 08:30:00 -0800
        try:
            # Try with timezone
            return datetime.strptime(date_str[:19], '%Y-%m-%d %H:%M:%S')
        except ValueError:
            # Try ISO format
            return datetime.fromisoformat(date_str.replace('Z', '+00:00'))

    def _iter_records(self) -> Generator[ET.Element, None, None]:
        """Iterate over Record elements in the XML file efficiently."""
        # Use iterparse for memory efficiency with large files
        context = ET.iterparse(self.xml_path, events=('end',))

        for event, elem in context:
            if elem.tag == 'Record':
                yield elem
                elem.clear()  # Free memory
            elif elem.tag == 'Workout':
                yield elem
                elem.clear()

    def parse_records(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        record_types: Optional[List[str]] = None
    ) -> Dict[str, List[Dict]]:
        """
        Parse health records from the XML file.

        Args:
            start_date: Only include records after this date
            end_date: Only include records before this date
            record_types: List of record types to include (e.g., ['steps', 'heart_rate'])

        Returns:
            Dictionary mapping record type to list of records
        """
        records = defaultdict(list)

        for elem in self._iter_records():
            record_type_raw = elem.get('type', '')
            record_type = RECORD_TYPES.get(record_type_raw)

            if record_type is None:
                continue

            if record_types and record_type not in record_types:
                continue

            # Parse dates
            try:
                start_dt = self._parse_date(elem.get('startDate', ''))
                end_dt = self._parse_date(elem.get('endDate', ''))
            except (ValueError, TypeError):
                continue

            # Filter by date range
            if start_date and start_dt.date() < start_date:
                continue
            if end_date and start_dt.date() > end_date:
                continue

            # Parse value
            value = elem.get('value', '')
            unit = elem.get('unit', '')

            # Handle sleep categories
            if record_type == 'sleep':
                value = SLEEP_VALUES.get(value, value)
            else:
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    pass

            record = {
                'type': record_type,
                'value': value,
                'unit': unit,
                'start_date': start_dt,
                'end_date': end_dt,
                'source': elem.get('sourceName', 'Apple Health'),
                'device': elem.get('device', ''),
            }

            records[record_type].append(record)

        return dict(records)

    def get_daily_summary(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> List[Dict]:
        """
        Aggregate records into daily summaries.

        Returns list of daily summary dictionaries.
        """
        records = self.parse_records(start_date, end_date)

        # Group by date
        daily_data = defaultdict(lambda: {
            'steps': 0,
            'calories_active': 0,
            'distance': 0,
            'floors': 0,
            'exercise_minutes': 0,
            'heart_rates': [],
            'resting_hr': None,
            'hrv_values': [],
            'sleep_minutes': 0,
            'deep_sleep_minutes': 0,
            'rem_sleep_minutes': 0,
            'weight': None,
        })

        # Aggregate steps
        for record in records.get('steps', []):
            day = record['start_date'].date()
            daily_data[day]['steps'] += int(record['value'])

        # Aggregate calories
        for record in records.get('calories_active', []):
            day = record['start_date'].date()
            daily_data[day]['calories_active'] += int(record['value'])

        # Aggregate distance
        for record in records.get('distance', []):
            day = record['start_date'].date()
            daily_data[day]['distance'] += record['value']

        # Aggregate floors
        for record in records.get('floors', []):
            day = record['start_date'].date()
            daily_data[day]['floors'] += int(record['value'])

        # Aggregate exercise minutes
        for record in records.get('exercise_minutes', []):
            day = record['start_date'].date()
            daily_data[day]['exercise_minutes'] += int(record['value'])

        # Heart rate - collect all readings
        for record in records.get('heart_rate', []):
            day = record['start_date'].date()
            daily_data[day]['heart_rates'].append(record['value'])

        # Resting heart rate - take latest reading per day
        for record in records.get('resting_hr', []):
            day = record['start_date'].date()
            daily_data[day]['resting_hr'] = int(record['value'])

        # HRV - collect all readings
        for record in records.get('hrv', []):
            day = record['start_date'].date()
            daily_data[day]['hrv_values'].append(record['value'])

        # Sleep - calculate duration by stage
        for record in records.get('sleep', []):
            # Sleep is logged for the night before, so use end_date
            day = record['end_date'].date()
            duration_minutes = (record['end_date'] - record['start_date']).seconds / 60

            if record['value'] in ['asleep', 'core']:
                daily_data[day]['sleep_minutes'] += duration_minutes
            elif record['value'] == 'deep':
                daily_data[day]['sleep_minutes'] += duration_minutes
                daily_data[day]['deep_sleep_minutes'] += duration_minutes
            elif record['value'] == 'rem':
                daily_data[day]['sleep_minutes'] += duration_minutes
                daily_data[day]['rem_sleep_minutes'] += duration_minutes

        # Weight - take latest reading per day
        for record in records.get('weight', []):
            day = record['start_date'].date()
            # Convert to kg if needed
            value = record['value']
            if record['unit'] == 'lb':
                value = value * 0.453592
            daily_data[day]['weight'] = round(value, 1)

        # Convert to list of daily summaries
        summaries = []
        for day in sorted(daily_data.keys()):
            data = daily_data[day]

            # Calculate averages
            avg_hr = sum(data['heart_rates']) / len(data['heart_rates']) if data['heart_rates'] else None
            avg_hrv = sum(data['hrv_values']) / len(data['hrv_values']) if data['hrv_values'] else None

            # Calculate sleep score (simplified)
            sleep_hours = data['sleep_minutes'] / 60
            sleep_score = None
            if sleep_hours > 0:
                base_score = min(100, (sleep_hours / 8) * 100)
                deep_bonus = min(10, (data['deep_sleep_minutes'] / 60) * 5)
                rem_bonus = min(10, (data['rem_sleep_minutes'] / 60) * 5)
                sleep_score = int(min(100, base_score + deep_bonus + rem_bonus))

            summary = {
                'date': day,
                'steps': data['steps'],
                'calories_active': data['calories_active'],
                'distance': round(data['distance'], 2),
                'floors': data['floors'],
                'active_minutes': data['exercise_minutes'],
                'avg_hr': int(avg_hr) if avg_hr else None,
                'resting_hr': data['resting_hr'],
                'hrv': round(avg_hrv, 1) if avg_hrv else None,
                'sleep_duration': round(sleep_hours, 2),
                'deep_sleep': round(data['deep_sleep_minutes'] / 60, 2),
                'rem_sleep': round(data['rem_sleep_minutes'] / 60, 2),
                'light_sleep': round((data['sleep_minutes'] - data['deep_sleep_minutes'] - data['rem_sleep_minutes']) / 60, 2),
                'sleep_score': sleep_score,
                'sleep_efficiency': 85 if sleep_hours > 0 else None,  # Placeholder
                'weight': data['weight'],
                'readiness_score': None,  # Calculate if we have enough data
                'sources': ['apple_health'],
            }

            # Calculate readiness score if we have HRV and sleep
            if avg_hrv and sleep_score:
                hrv_component = min(40, (avg_hrv / 50) * 40)
                sleep_component = (sleep_score / 100) * 40
                rhr_component = 20  # Default without baseline
                if data['resting_hr']:
                    # Lower RHR is better (assuming baseline of 60)
                    rhr_component = max(0, min(20, 20 - (data['resting_hr'] - 55) * 0.5))
                summary['readiness_score'] = int(min(100, hrv_component + sleep_component + rhr_component))

            summaries.append(summary)

        return summaries

    def cleanup(self):
        """Clean up temporary files."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)


def parse_apple_health_export(
    file_path: str,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None
) -> Tuple[List[Dict], Dict]:
    """
    Convenience function to parse Apple Health export.

    Args:
        file_path: Path to export.xml or export.zip
        start_date: Optional start date filter
        end_date: Optional end date filter

    Returns:
        Tuple of (daily_summaries, metadata)
    """
    parser = AppleHealthParser(file_path)

    try:
        summaries = parser.get_daily_summary(start_date, end_date)

        metadata = {
            'source': 'apple_health',
            'file': os.path.basename(file_path),
            'days_imported': len(summaries),
            'date_range': {
                'start': summaries[0]['date'].isoformat() if summaries else None,
                'end': summaries[-1]['date'].isoformat() if summaries else None,
            }
        }

        return summaries, metadata
    finally:
        parser.cleanup()
