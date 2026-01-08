"""
HealthBridge - Unified Health Data Platform
Main Streamlit Application
"""

import streamlit as st
from datetime import datetime
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Page config must be first Streamlit command
st.set_page_config(
    page_title="HealthBridge",
    page_icon="🌉",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'mailto:support@healthbridge.demo',
        'Report a bug': 'mailto:bugs@healthbridge.demo',
        'About': """
        ## HealthBridge
        **The unified platform for all your health data.**

        Connect your wearables, see your labs, get AI-powered insights.

        Demo Version 0.1.0
        """
    }
)

# Custom CSS for polish
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.5rem;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 1.5rem;
        color: white;
    }

    /* Status indicators */
    .status-good { color: #10B981; }
    .status-warning { color: #F59E0B; }
    .status-alert { color: #EF4444; }

    /* Hide Streamlit branding for cleaner demo */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Sidebar styling */
    .css-1d391kg {
        background-color: #F8FAFC;
    }

    /* Connection status badges */
    .device-connected {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        font-weight: 500;
    }

    .device-disconnected {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        font-weight: 500;
    }

    /* Card styling */
    .health-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border: 1px solid #E2E8F0;
        margin-bottom: 1rem;
    }

    /* Navigation buttons */
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    defaults = {
        'authenticated': True,  # Skip auth for demo
        'user_id': 'demo_user',
        'user_name': 'Demo User',
        'connected_devices': [],
        'health_data': None,
        'patient_profile': None,
        'lab_data': None,
        'demo_mode': True,
        'data_loaded': False,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def load_demo_data():
    """Load synthetic demo data."""
    if not st.session_state.data_loaded:
        from src.data.synthetic.patient_generator import generate_demo_data
        from src.data.synthetic.lab_generator import generate_lab_history

        with st.spinner("Loading your health data..."):
            # Generate patient and health data
            patient, health_data = generate_demo_data(days=90)
            st.session_state.patient_profile = patient
            st.session_state.health_data = health_data

            # Generate lab history
            labs = generate_lab_history(
                patient.id,
                patient.health_conditions,
                patient.age,
                patient.sex,
                num_panels=4
            )
            st.session_state.lab_data = labs

            # Set connected devices (simulated)
            st.session_state.connected_devices = [
                {'name': 'Oura Ring', 'type': 'oura', 'connected': True, 'last_sync': datetime.now()},
                {'name': 'Apple Watch', 'type': 'apple', 'connected': True, 'last_sync': datetime.now()},
                {'name': 'Dexcom G7', 'type': 'cgm', 'connected': True, 'last_sync': datetime.now()},
            ]

            st.session_state.data_loaded = True


def main():
    """Main application."""
    init_session_state()
    load_demo_data()

    # Sidebar
    with st.sidebar:
        st.markdown("# 🌉 HealthBridge")
        st.markdown("---")

        # User info
        if st.session_state.patient_profile:
            patient = st.session_state.patient_profile
            st.markdown(f"**👤 {patient.name}**")
            st.caption(f"{patient.age} years old • {patient.activity_level.replace('_', ' ').title()}")
        else:
            st.markdown(f"**👤 {st.session_state.user_name}**")

        # Connected devices summary
        st.markdown("### Connected Devices")
        for device in st.session_state.connected_devices:
            status = "🟢" if device['connected'] else "🔴"
            st.markdown(f"{status} {device['name']}")

        st.markdown("---")

        # Demo mode indicator
        if st.session_state.demo_mode:
            st.info("🎭 **Demo Mode**\n\nUsing synthetic data. Connect real devices in Settings.")

    # Main content - Landing page
    st.markdown('<p class="main-header">Welcome to HealthBridge</p>', unsafe_allow_html=True)
    st.markdown("Your unified health data platform. All your devices, one clear picture.")

    st.markdown("---")

    # Quick stats row
    if st.session_state.health_data:
        latest = st.session_state.health_data[-1]  # Most recent day

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            delta_sleep = None
            if len(st.session_state.health_data) > 1:
                delta_sleep = latest['sleep_score'] - st.session_state.health_data[-2]['sleep_score']
            st.metric(
                "Sleep Score",
                f"{latest['sleep_score']}",
                delta=f"{delta_sleep}" if delta_sleep else None
            )

        with col2:
            delta_hrv = None
            if len(st.session_state.health_data) > 1:
                delta_hrv = latest['hrv'] - st.session_state.health_data[-2]['hrv']
            st.metric(
                "HRV",
                f"{latest['hrv']:.0f} ms",
                delta=f"{delta_hrv:.0f}" if delta_hrv else None
            )

        with col3:
            delta_rhr = None
            if len(st.session_state.health_data) > 1:
                delta_rhr = latest['resting_hr'] - st.session_state.health_data[-2]['resting_hr']
            st.metric(
                "Resting HR",
                f"{latest['resting_hr']} bpm",
                delta=f"{delta_rhr}" if delta_rhr else None,
                delta_color="inverse"  # Lower is better for RHR
            )

        with col4:
            delta_steps = None
            if len(st.session_state.health_data) > 1:
                delta_steps = latest['steps'] - st.session_state.health_data[-2]['steps']
            st.metric(
                "Steps",
                f"{latest['steps']:,}",
                delta=f"{delta_steps:,}" if delta_steps else None
            )

        with col5:
            delta_readiness = None
            if len(st.session_state.health_data) > 1:
                delta_readiness = latest['readiness_score'] - st.session_state.health_data[-2]['readiness_score']
            st.metric(
                "Readiness",
                f"{latest['readiness_score']}",
                delta=f"{delta_readiness}" if delta_readiness else None
            )

    st.markdown("---")

    # Navigation cards
    st.markdown("### Explore Your Data")
    st.caption("Use the sidebar navigation to explore different sections")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%); padding: 1.5rem; border-radius: 12px; color: white; min-height: 160px;">
            <h4 style="margin: 0; color: white;">🏠 Dashboard</h4>
            <p style="font-size: 0.9rem; opacity: 0.9;">See all your health metrics at a glance. Trends, patterns, and daily summaries.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #10B981 0%, #059669 100%); padding: 1.5rem; border-radius: 12px; color: white; min-height: 160px;">
            <h4 style="margin: 0; color: white;">🤖 AI Insights</h4>
            <p style="font-size: 0.9rem; opacity: 0.9;">Get personalized recommendations powered by AI. Understand what your data means.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%); padding: 1.5rem; border-radius: 12px; color: white; min-height: 160px;">
            <h4 style="margin: 0; color: white;">🧬 Clinical Data</h4>
            <p style="font-size: 0.9rem; opacity: 0.9;">View lab results and clinical metrics. Track biomarkers over time.</p>
        </div>
        """, unsafe_allow_html=True)

    # Second row of cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%); padding: 1.5rem; border-radius: 12px; color: white; min-height: 160px;">
            <h4 style="margin: 0; color: white;">🔗 Connect Devices</h4>
            <p style="font-size: 0.9rem; opacity: 0.9;">Link your wearables and health devices. Oura, Apple Watch, Whoop, and more.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #EC4899 0%, #BE185D 100%); padding: 1.5rem; border-radius: 12px; color: white; min-height: 160px;">
            <h4 style="margin: 0; color: white;">📊 Deep Dive</h4>
            <p style="font-size: 0.9rem; opacity: 0.9;">Detailed analysis of specific metrics. Correlations and trends over time.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #6B7280 0%, #374151 100%); padding: 1.5rem; border-radius: 12px; color: white; min-height: 160px;">
            <h4 style="margin: 0; color: white;">⚙️ Settings</h4>
            <p style="font-size: 0.9rem; opacity: 0.9;">Configure your preferences. Manage data and privacy.</p>
        </div>
        """, unsafe_allow_html=True)

    # Footer with data freshness
    st.markdown("---")
    if st.session_state.connected_devices:
        last_sync = max(d['last_sync'] for d in st.session_state.connected_devices)
        st.caption(f"Last synced: {last_sync.strftime('%B %d, %Y at %I:%M %p')}")


if __name__ == "__main__":
    main()
