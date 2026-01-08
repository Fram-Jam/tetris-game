"""
Session State Management for Health Data

Handles data persistence within Streamlit sessions.
"""

import streamlit as st
from typing import List, Dict, Optional, Any
from datetime import datetime


def init_storage():
    """Initialize all session state variables."""
    defaults = {
        'authenticated': True,
        'user_id': 'demo_user',
        'user_name': 'Demo User',
        'connected_devices': [],
        'health_data': None,
        'patient_profile': None,
        'lab_data': None,
        'demo_mode': True,
        'data_loaded': False,
        'settings': {
            'weight_unit': 'kg',
            'height_unit': 'cm',
            'temp_unit': 'celsius',
            'date_format': 'YYYY-MM-DD',
            'time_format': '12h',
            'default_time_range': 30,
            'show_targets': True,
            'show_averages': True,
        }
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_health_data() -> Optional[List[Dict]]:
    """Get health data from session state."""
    return st.session_state.get('health_data')


def set_health_data(data: List[Dict]):
    """Set health data in session state."""
    st.session_state.health_data = data


def get_patient_profile() -> Optional[Any]:
    """Get patient profile from session state."""
    return st.session_state.get('patient_profile')


def set_patient_profile(profile: Any):
    """Set patient profile in session state."""
    st.session_state.patient_profile = profile


def get_lab_data() -> Optional[List]:
    """Get lab data from session state."""
    return st.session_state.get('lab_data')


def set_lab_data(data: List):
    """Set lab data in session state."""
    st.session_state.lab_data = data


def get_connected_devices() -> List[Dict]:
    """Get list of connected devices."""
    return st.session_state.get('connected_devices', [])


def add_device(device: Dict):
    """Add a connected device."""
    devices = get_connected_devices()
    if not any(d['type'] == device['type'] for d in devices):
        devices.append(device)
        st.session_state.connected_devices = devices


def remove_device(device_type: str):
    """Remove a connected device by type."""
    devices = get_connected_devices()
    st.session_state.connected_devices = [
        d for d in devices if d['type'] != device_type
    ]


def is_device_connected(device_type: str) -> bool:
    """Check if a device type is connected."""
    return any(d['type'] == device_type for d in get_connected_devices())


def get_setting(key: str, default: Any = None) -> Any:
    """Get a setting value."""
    settings = st.session_state.get('settings', {})
    return settings.get(key, default)


def set_setting(key: str, value: Any):
    """Set a setting value."""
    if 'settings' not in st.session_state:
        st.session_state.settings = {}
    st.session_state.settings[key] = value


def clear_all_data():
    """Clear all health data from session."""
    st.session_state.health_data = None
    st.session_state.lab_data = None
    st.session_state.patient_profile = None
    st.session_state.data_loaded = False


def mark_data_loaded():
    """Mark data as loaded."""
    st.session_state.data_loaded = True


def is_data_loaded() -> bool:
    """Check if data has been loaded."""
    return st.session_state.get('data_loaded', False)
