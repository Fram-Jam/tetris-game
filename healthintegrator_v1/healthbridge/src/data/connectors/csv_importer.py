"""
Generic CSV Health Data Importer

Imports health data from CSV files with flexible column mapping.
Supports data from various sources including Fitbit, Garmin, Whoop exports.
"""

import pandas as pd
from datetime import datetime, date
from typing import List, Dict, Optional, Tuple
import io


# Common column name mappings
COLUMN_MAPPINGS = {
    # Date columns
    'date': ['date', 'day', 'timestamp', 'datetime', 'time', 'recorded_at', 'created_at'],

    # Sleep columns
    'sleep_duration': ['sleep_duration', 'total_sleep', 'sleep_hours', 'sleep_time', 'duration_asleep', 'asleep_duration'],
    'sleep_score': ['sleep_score', 'sleep_quality', 'quality_score'],
    'deep_sleep': ['deep_sleep', 'deep', 'deep_duration', 'slow_wave_sleep'],
    'rem_sleep': ['rem_sleep', 'rem', 'rem_duration'],
    'light_sleep': ['light_sleep', 'light', 'light_duration'],
    'awake_time': ['awake', 'awake_time', 'time_awake', 'awake_duration'],
    'sleep_efficiency': ['sleep_efficiency', 'efficiency'],

    # Heart columns
    'resting_hr': ['resting_hr', 'resting_heart_rate', 'rhr', 'rest_hr', 'avg_resting_hr'],
    'hrv': ['hrv', 'heart_rate_variability', 'hrv_ms', 'rmssd', 'hrv_rmssd'],
    'avg_hr': ['avg_hr', 'average_heart_rate', 'mean_hr', 'heart_rate'],
    'max_hr': ['max_hr', 'max_heart_rate', 'peak_hr'],

    # Activity columns
    'steps': ['steps', 'step_count', 'total_steps', 'daily_steps'],
    'calories_active': ['calories_active', 'active_calories', 'calories_burned', 'activity_calories'],
    'calories_total': ['calories_total', 'total_calories', 'calories'],
    'distance': ['distance', 'distance_km', 'distance_miles', 'total_distance'],
    'active_minutes': ['active_minutes', 'exercise_minutes', 'activity_minutes', 'active_time'],
    'floors': ['floors', 'floors_climbed', 'elevation_gain'],

    # Recovery columns
    'readiness_score': ['readiness_score', 'readiness', 'recovery_score', 'recovery', 'strain_score'],

    # Body columns
    'weight': ['weight', 'body_weight', 'weight_kg', 'weight_lbs', 'mass'],
    'body_fat': ['body_fat', 'fat_percentage', 'body_fat_pct'],
    'bmi': ['bmi', 'body_mass_index'],

    # Vitals
    'spo2': ['spo2', 'oxygen_saturation', 'blood_oxygen'],
    'respiratory_rate': ['respiratory_rate', 'breathing_rate', 'resp_rate'],
    'body_temp': ['body_temp', 'temperature', 'temp', 'skin_temp'],
}


def find_column_match(df_columns: List[str], target: str) -> Optional[str]:
    """Find matching column name in DataFrame."""
    # Exact match first
    if target in df_columns:
        return target

    # Check known mappings
    possible_names = COLUMN_MAPPINGS.get(target, [])
    for name in possible_names:
        # Case-insensitive match
        for col in df_columns:
            if col.lower() == name.lower():
                return col
            # Partial match
            if name.lower() in col.lower() or col.lower() in name.lower():
                return col

    return None


def detect_date_column(df: pd.DataFrame) -> Optional[str]:
    """Detect the date column in a DataFrame."""
    date_candidates = COLUMN_MAPPINGS['date']

    for col in df.columns:
        if col.lower() in [c.lower() for c in date_candidates]:
            return col

    # Try to find any column that parses as dates
    for col in df.columns:
        try:
            pd.to_datetime(df[col].head(10))
            return col
        except:
            continue

    return None


def parse_csv_data(
    file_path_or_buffer,
    column_mapping: Optional[Dict[str, str]] = None,
    date_column: Optional[str] = None,
    date_format: Optional[str] = None,
) -> Tuple[List[Dict], Dict]:
    """
    Parse health data from CSV file.

    Args:
        file_path_or_buffer: Path to CSV file or file-like object
        column_mapping: Optional explicit column name mapping
        date_column: Optional explicit date column name
        date_format: Optional date format string

    Returns:
        Tuple of (daily_summaries, metadata)
    """
    # Read CSV
    df = pd.read_csv(file_path_or_buffer)

    # Detect or use provided date column
    date_col = date_column or detect_date_column(df)
    if not date_col:
        raise ValueError("Could not detect date column. Please specify date_column parameter.")

    # Parse dates
    if date_format:
        df['_date'] = pd.to_datetime(df[date_col], format=date_format)
    else:
        df['_date'] = pd.to_datetime(df[date_col])

    df['_date'] = df['_date'].dt.date

    # Build column mapping
    mapping = column_mapping or {}
    df_columns = list(df.columns)

    # Auto-detect columns not explicitly mapped
    for target_col in COLUMN_MAPPINGS.keys():
        if target_col not in mapping:
            matched = find_column_match(df_columns, target_col)
            if matched:
                mapping[target_col] = matched

    # Convert to list of daily summaries
    summaries = []
    detected_columns = []

    for _, row in df.iterrows():
        summary = {
            'date': row['_date'],
            'sources': ['csv_import'],
        }

        # Map each known field
        for target, source in mapping.items():
            if source in df.columns:
                value = row[source]
                if pd.notna(value):
                    # Convert to appropriate type
                    if target in ['steps', 'floors', 'resting_hr', 'avg_hr', 'max_hr',
                                  'calories_active', 'calories_total', 'active_minutes',
                                  'sleep_score', 'readiness_score']:
                        value = int(float(value))
                    elif target in ['sleep_duration', 'deep_sleep', 'rem_sleep', 'light_sleep',
                                    'hrv', 'distance', 'weight', 'body_fat', 'bmi',
                                    'spo2', 'respiratory_rate', 'body_temp', 'sleep_efficiency']:
                        value = float(value)

                    summary[target] = value
                    if source not in detected_columns:
                        detected_columns.append(source)

        # Calculate derived fields if missing
        if 'sleep_duration' not in summary:
            # Try to calculate from components
            deep = summary.get('deep_sleep', 0) or 0
            rem = summary.get('rem_sleep', 0) or 0
            light = summary.get('light_sleep', 0) or 0
            if deep + rem + light > 0:
                summary['sleep_duration'] = deep + rem + light

        if 'calories_total' not in summary and 'calories_active' in summary:
            # Estimate total with BMR
            summary['calories_total'] = summary['calories_active'] + 1800  # Rough estimate

        summaries.append(summary)

    # Sort by date
    summaries.sort(key=lambda x: x['date'])

    metadata = {
        'source': 'csv_import',
        'rows_imported': len(summaries),
        'columns_detected': detected_columns,
        'column_mapping': mapping,
        'date_range': {
            'start': summaries[0]['date'].isoformat() if summaries else None,
            'end': summaries[-1]['date'].isoformat() if summaries else None,
        }
    }

    return summaries, metadata


def merge_health_data(
    existing_data: List[Dict],
    new_data: List[Dict],
    prefer_new: bool = True
) -> List[Dict]:
    """
    Merge new health data with existing data.

    Args:
        existing_data: Existing daily summaries
        new_data: New daily summaries to merge
        prefer_new: If True, new data overwrites existing for same date

    Returns:
        Merged list of daily summaries
    """
    # Index by date
    by_date = {}

    for record in existing_data:
        by_date[record['date']] = record.copy()

    for record in new_data:
        d = record['date']
        if d in by_date:
            if prefer_new:
                # Merge, preferring new values
                merged = by_date[d].copy()
                for key, value in record.items():
                    if value is not None:
                        merged[key] = value
                # Combine sources
                sources = set(merged.get('sources', []) + record.get('sources', []))
                merged['sources'] = list(sources)
                by_date[d] = merged
            else:
                # Keep existing, fill in gaps
                for key, value in record.items():
                    if key not in by_date[d] or by_date[d][key] is None:
                        by_date[d][key] = value
        else:
            by_date[d] = record.copy()

    # Sort by date
    return sorted(by_date.values(), key=lambda x: x['date'])


def validate_csv_format(file_path_or_buffer) -> Dict:
    """
    Validate CSV file and return detected columns info.

    Returns dict with validation results and column detection.
    """
    try:
        df = pd.read_csv(file_path_or_buffer)

        # Reset buffer position if it's a file-like object
        if hasattr(file_path_or_buffer, 'seek'):
            file_path_or_buffer.seek(0)

        date_col = detect_date_column(df)
        detected_mappings = {}

        for target_col in COLUMN_MAPPINGS.keys():
            matched = find_column_match(list(df.columns), target_col)
            if matched:
                detected_mappings[target_col] = matched

        return {
            'valid': True,
            'rows': len(df),
            'columns': list(df.columns),
            'date_column': date_col,
            'detected_mappings': detected_mappings,
            'unmapped_columns': [c for c in df.columns if c not in detected_mappings.values()],
        }
    except Exception as e:
        return {
            'valid': False,
            'error': str(e),
        }
