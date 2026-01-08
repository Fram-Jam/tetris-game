"""
Device Connection Page

In demo mode, simulates device connections.
In production, would use Terra API or direct OAuth.
"""

import streamlit as st
from datetime import datetime
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(page_title="Connect Devices | HealthBridge", layout="wide", page_icon="🌉")

# Initialize session state
if 'connected_devices' not in st.session_state:
    st.session_state.connected_devices = []
if 'demo_mode' not in st.session_state:
    st.session_state.demo_mode = True

st.title("🔗 Connect Your Devices")
st.markdown("Link your wearables and health devices to get a complete picture of your health.")

# Device catalog
DEVICES = {
    'wearables': [
        {'id': 'oura', 'name': 'Oura Ring', 'icon': '💍', 'data': ['Sleep', 'HRV', 'Activity', 'Readiness', 'Temperature']},
        {'id': 'apple', 'name': 'Apple Watch', 'icon': '⌚', 'data': ['Activity', 'Heart Rate', 'Workouts', 'ECG', 'Sleep']},
        {'id': 'whoop', 'name': 'WHOOP', 'icon': '🔴', 'data': ['Strain', 'Recovery', 'Sleep', 'HRV']},
        {'id': 'garmin', 'name': 'Garmin', 'icon': '🏃', 'data': ['Activity', 'GPS', 'Heart Rate', 'Sleep', 'Training Load']},
        {'id': 'fitbit', 'name': 'Fitbit', 'icon': '📱', 'data': ['Steps', 'Sleep', 'Heart Rate', 'SpO2', 'Stress']},
    ],
    'cgm': [
        {'id': 'dexcom', 'name': 'Dexcom G6/G7', 'icon': '📊', 'data': ['Glucose', 'Trends', 'Alerts', 'Time in Range']},
        {'id': 'libre', 'name': 'FreeStyle Libre', 'icon': '🩸', 'data': ['Glucose', 'Trends', 'Glucose Patterns']},
    ],
    'other': [
        {'id': 'withings', 'name': 'Withings Scale', 'icon': '⚖️', 'data': ['Weight', 'Body Comp', 'Heart Health']},
        {'id': 'omron', 'name': 'Omron BP Monitor', 'icon': '💓', 'data': ['Blood Pressure', 'Pulse', 'AFib Detection']},
        {'id': 'eightsleep', 'name': 'Eight Sleep', 'icon': '🛏️', 'data': ['Sleep Tracking', 'Temperature', 'HRV']},
    ]
}


def is_connected(device_id: str) -> bool:
    """Check if a device is connected."""
    return any(d['type'] == device_id for d in st.session_state.connected_devices)


def connect_device(device_id: str, device_name: str):
    """Simulate connecting a device."""
    if not is_connected(device_id):
        st.session_state.connected_devices.append({
            'name': device_name,
            'type': device_id,
            'connected': True,
            'last_sync': datetime.now()
        })


def disconnect_device(device_id: str):
    """Disconnect a device."""
    st.session_state.connected_devices = [
        d for d in st.session_state.connected_devices if d['type'] != device_id
    ]


# Currently connected
st.markdown("### ✅ Connected Devices")
if st.session_state.connected_devices:
    cols = st.columns(min(len(st.session_state.connected_devices), 4))
    for i, device in enumerate(st.session_state.connected_devices):
        with cols[i % 4]:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #D1FAE5 0%, #A7F3D0 100%); padding: 1rem; border-radius: 12px; text-align: center; border: 1px solid #6EE7B7;">
                <h4 style="margin: 0; color: #065F46;">{device['name']}</h4>
                <small style="color: #047857;">Last sync: {device['last_sync'].strftime('%I:%M %p')}</small>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Disconnect", key=f"dc_{device['type']}", use_container_width=True):
                disconnect_device(device['type'])
                st.rerun()
else:
    st.info("No devices connected yet. Add your first device below!")

st.markdown("---")

# Available devices
st.markdown("### 📱 Available Devices")

# Wearables section
st.markdown("#### Wearables & Fitness Trackers")
cols = st.columns(5)
for i, device in enumerate(DEVICES['wearables']):
    with cols[i]:
        connected = is_connected(device['id'])

        # Card styling
        bg_color = "#D1FAE5" if connected else "#F8FAFC"
        border_color = "#6EE7B7" if connected else "#E2E8F0"

        st.markdown(f"""
        <div style="background: {bg_color}; padding: 1rem; border-radius: 12px; text-align: center; border: 1px solid {border_color}; min-height: 140px;">
            <div style="font-size: 2rem;">{device['icon']}</div>
            <h5 style="margin: 0.5rem 0;">{device['name']}</h5>
            <small style="color: #6B7280;">{', '.join(device['data'][:2])}</small>
        </div>
        """, unsafe_allow_html=True)

        if connected:
            st.button("✓ Connected", key=f"sync_{device['id']}", disabled=True, use_container_width=True)
        else:
            if st.button("Connect", key=f"connect_{device['id']}", use_container_width=True):
                connect_device(device['id'], device['name'])
                st.success(f"✅ {device['name']} connected!")
                st.rerun()

st.markdown("---")

# CGM section
st.markdown("#### Continuous Glucose Monitors")
cols = st.columns(4)
for i, device in enumerate(DEVICES['cgm']):
    with cols[i]:
        connected = is_connected(device['id'])

        bg_color = "#D1FAE5" if connected else "#F8FAFC"
        border_color = "#6EE7B7" if connected else "#E2E8F0"

        st.markdown(f"""
        <div style="background: {bg_color}; padding: 1rem; border-radius: 12px; text-align: center; border: 1px solid {border_color}; min-height: 140px;">
            <div style="font-size: 2rem;">{device['icon']}</div>
            <h5 style="margin: 0.5rem 0;">{device['name']}</h5>
            <small style="color: #6B7280;">{', '.join(device['data'][:2])}</small>
        </div>
        """, unsafe_allow_html=True)

        if connected:
            st.button("✓ Connected", key=f"sync_{device['id']}", disabled=True, use_container_width=True)
        else:
            if st.button("Connect", key=f"connect_{device['id']}", use_container_width=True):
                connect_device(device['id'], device['name'])
                st.success(f"✅ {device['name']} connected!")
                st.rerun()

st.markdown("---")

# Other devices
st.markdown("#### Other Health Devices")
cols = st.columns(4)
for i, device in enumerate(DEVICES['other']):
    with cols[i]:
        connected = is_connected(device['id'])

        bg_color = "#D1FAE5" if connected else "#F8FAFC"
        border_color = "#6EE7B7" if connected else "#E2E8F0"

        st.markdown(f"""
        <div style="background: {bg_color}; padding: 1rem; border-radius: 12px; text-align: center; border: 1px solid {border_color}; min-height: 140px;">
            <div style="font-size: 2rem;">{device['icon']}</div>
            <h5 style="margin: 0.5rem 0;">{device['name']}</h5>
            <small style="color: #6B7280;">{', '.join(device['data'][:2])}</small>
        </div>
        """, unsafe_allow_html=True)

        if connected:
            st.button("✓ Connected", key=f"sync_{device['id']}", disabled=True, use_container_width=True)
        else:
            if st.button("Connect", key=f"connect_{device['id']}", use_container_width=True):
                connect_device(device['id'], device['name'])
                st.success(f"✅ {device['name']} connected!")
                st.rerun()

st.markdown("---")

# Manual upload option
st.markdown("### 📤 Manual Data Upload")
st.markdown("Don't see your device? Upload data manually.")

col1, col2 = st.columns(2)

with col1:
    upload_type = st.selectbox(
        "Data Type",
        ["Apple Health Export (XML)", "Oura Export (JSON)", "Fitbit Export (JSON)", "Generic CSV"]
    )

with col2:
    file_types = {
        "Apple Health Export (XML)": ['xml', 'zip'],
        "Oura Export (JSON)": ['json'],
        "Fitbit Export (JSON)": ['json'],
        "Generic CSV": ['csv']
    }
    uploaded_file = st.file_uploader(
        "Choose file",
        type=file_types.get(upload_type, ['csv', 'json', 'xml'])
    )

if uploaded_file:
    st.success(f"File uploaded: {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Process Upload", use_container_width=True):
            with st.spinner("Processing your health data..."):
                import time
                time.sleep(2)  # Simulate processing
                st.success("✅ Data imported successfully! 90 days of data added.")
    with col2:
        if st.button("Cancel", use_container_width=True):
            st.rerun()

# Demo mode notice
if st.session_state.demo_mode:
    st.markdown("---")
    st.info("""
    🎭 **Demo Mode Active**

    Device connections are simulated. In the full version, clicking "Connect" would:
    1. Open OAuth authentication with the device provider
    2. Request permission to access your health data
    3. Sync historical data automatically (30-90 days)
    4. Set up real-time data syncing

    **Supported integrations** include Terra API for unified wearable access,
    direct APIs for Oura, Whoop, and Apple Health XML imports.
    """)

# Connection status summary
st.markdown("---")
st.markdown("### 📊 Data Coverage")

coverage_data = {
    'Sleep': is_connected('oura') or is_connected('apple') or is_connected('whoop') or is_connected('eightsleep'),
    'HRV': is_connected('oura') or is_connected('whoop') or is_connected('garmin'),
    'Activity': is_connected('apple') or is_connected('garmin') or is_connected('fitbit'),
    'Glucose': is_connected('dexcom') or is_connected('libre'),
    'Body Composition': is_connected('withings'),
    'Blood Pressure': is_connected('omron'),
}

cols = st.columns(6)
for i, (metric, covered) in enumerate(coverage_data.items()):
    with cols[i]:
        status = "✅" if covered else "❌"
        color = "#10B981" if covered else "#9CA3AF"
        st.markdown(f"""
        <div style="text-align: center;">
            <div style="font-size: 1.5rem;">{status}</div>
            <small style="color: {color};">{metric}</small>
        </div>
        """, unsafe_allow_html=True)
