"""
Settings Page - User preferences and configuration
"""

import streamlit as st
from datetime import datetime
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(page_title="Settings | HealthBridge", layout="wide", page_icon="🌉")

# Initialize session state
if 'demo_mode' not in st.session_state:
    st.session_state.demo_mode = True
if 'user_name' not in st.session_state:
    st.session_state.user_name = 'Demo User'
if 'connected_devices' not in st.session_state:
    st.session_state.connected_devices = []

st.title("⚙️ Settings")
st.markdown("Configure your HealthBridge experience")

# Create tabs for different settings sections
tab1, tab2, tab3, tab4 = st.tabs(["👤 Profile", "🔑 API Keys", "🎨 Preferences", "📊 Data Management"])

with tab1:
    st.markdown("### Profile Settings")

    col1, col2 = st.columns(2)

    with col1:
        name = st.text_input("Display Name", value=st.session_state.user_name)
        if name != st.session_state.user_name:
            st.session_state.user_name = name

        email = st.text_input("Email", value="demo@healthbridge.app", disabled=True)
        st.caption("Email cannot be changed in demo mode")

    with col2:
        if st.session_state.get('patient_profile'):
            patient = st.session_state.patient_profile
            st.markdown(f"""
            **Generated Profile:**
            - Age: {patient.age} years
            - Sex: {'Male' if patient.sex == 'M' else 'Female'}
            - Height: {patient.height_cm:.0f} cm
            - Weight: {patient.weight_kg:.0f} kg
            - Activity Level: {patient.activity_level.replace('_', ' ').title()}
            """)
        else:
            st.info("Profile data will be shown after data is loaded")

    st.markdown("---")

    st.markdown("### Health Goals")

    col1, col2 = st.columns(2)
    with col1:
        sleep_goal = st.slider("Daily Sleep Goal (hours)", 6.0, 10.0, 7.5, 0.5)
        steps_goal = st.slider("Daily Steps Goal", 5000, 20000, 10000, 1000)

    with col2:
        hrv_goal = st.slider("Target HRV (ms)", 20, 100, 50, 5)
        weight_goal = st.number_input("Target Weight (kg)", 40.0, 150.0, 70.0, 0.5)

    if st.button("Save Goals", use_container_width=True):
        st.success("Goals saved successfully!")

with tab2:
    st.markdown("### API Configuration")

    st.info("""
    Add API keys to enable AI-powered insights and real device connections.
    Keys are stored securely and never shared.
    """)

    st.markdown("#### AI Provider")

    anthropic_key = st.text_input(
        "Anthropic API Key",
        type="password",
        placeholder="sk-ant-...",
        help="Get your key at console.anthropic.com"
    )

    openai_key = st.text_input(
        "OpenAI API Key (Alternative)",
        type="password",
        placeholder="sk-...",
        help="Get your key at platform.openai.com"
    )

    st.markdown("---")

    st.markdown("#### Device Integrations")

    terra_key = st.text_input(
        "Terra API Key",
        type="password",
        placeholder="terra_...",
        help="Terra provides unified access to 200+ wearables"
    )

    terra_dev_id = st.text_input(
        "Terra Developer ID",
        placeholder="your-dev-id",
        help="Found in your Terra dashboard"
    )

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Save API Keys", use_container_width=True):
            st.success("API keys would be saved to .streamlit/secrets.toml")
            st.code("""
# .streamlit/secrets.toml
ANTHROPIC_API_KEY = "your-key"
OPENAI_API_KEY = "your-key"
TERRA_API_KEY = "your-key"
TERRA_DEV_ID = "your-id"
            """)
    with col2:
        if st.button("Test Connections", use_container_width=True):
            with st.spinner("Testing API connections..."):
                import time
                time.sleep(1)
                st.warning("Demo mode: API testing simulated")

with tab3:
    st.markdown("### Display Preferences")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Units")
        weight_unit = st.radio("Weight", ["Kilograms (kg)", "Pounds (lb)"], horizontal=True)
        height_unit = st.radio("Height", ["Centimeters (cm)", "Feet/Inches"], horizontal=True)
        temp_unit = st.radio("Temperature", ["Celsius (°C)", "Fahrenheit (°F)"], horizontal=True)

    with col2:
        st.markdown("#### Date & Time")
        date_format = st.selectbox("Date Format", ["MM/DD/YYYY", "DD/MM/YYYY", "YYYY-MM-DD"])
        time_format = st.radio("Time Format", ["12-hour (AM/PM)", "24-hour"], horizontal=True)
        timezone = st.selectbox("Timezone", ["Auto-detect", "UTC", "US/Eastern", "US/Pacific", "Europe/London"])

    st.markdown("---")

    st.markdown("#### Dashboard Preferences")

    col1, col2 = st.columns(2)
    with col1:
        default_range = st.selectbox("Default Time Range", ["Last 7 days", "Last 14 days", "Last 30 days", "Last 90 days"], index=2)
        show_targets = st.checkbox("Show target lines on charts", value=True)
        show_averages = st.checkbox("Show moving averages", value=True)

    with col2:
        chart_theme = st.selectbox("Chart Color Theme", ["Default", "Vibrant", "Muted", "Monochrome"])
        compact_mode = st.checkbox("Compact dashboard mode", value=False)
        auto_refresh = st.checkbox("Auto-refresh data", value=True)

    if st.button("Save Preferences", use_container_width=True):
        st.success("Preferences saved!")

with tab4:
    st.markdown("### Data Management")

    # Current data summary
    st.markdown("#### Current Data")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.session_state.get('health_data'):
            st.metric("Health Records", len(st.session_state.health_data))
        else:
            st.metric("Health Records", 0)
    with col2:
        if st.session_state.get('lab_data'):
            st.metric("Lab Panels", len(st.session_state.lab_data))
        else:
            st.metric("Lab Panels", 0)
    with col3:
        st.metric("Connected Devices", len(st.session_state.connected_devices))

    st.markdown("---")

    st.markdown("#### Data Export")

    col1, col2 = st.columns(2)
    with col1:
        export_format = st.selectbox("Export Format", ["CSV", "JSON", "PDF Report"])

    with col2:
        if st.button("Export All Data", use_container_width=True):
            if st.session_state.get('health_data'):
                import pandas as pd
                df = pd.DataFrame(st.session_state.health_data)

                if export_format == "CSV":
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "Download CSV",
                        csv,
                        "healthbridge_export.csv",
                        "text/csv",
                        use_container_width=True
                    )
                elif export_format == "JSON":
                    import json
                    json_str = df.to_json(orient='records', date_format='iso')
                    st.download_button(
                        "Download JSON",
                        json_str,
                        "healthbridge_export.json",
                        "application/json",
                        use_container_width=True
                    )
                else:
                    st.info("PDF export coming soon!")
            else:
                st.warning("No data to export")

    st.markdown("---")

    st.markdown("#### Data Reset")

    st.warning("⚠️ These actions cannot be undone")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Reset Demo Data", use_container_width=True):
            st.session_state.data_loaded = False
            st.session_state.health_data = None
            st.session_state.lab_data = None
            st.success("Demo data will be regenerated on next page load")

    with col2:
        if st.button("Disconnect All Devices", use_container_width=True):
            st.session_state.connected_devices = []
            st.success("All devices disconnected")

    with col3:
        if st.button("Delete All Data", type="primary", use_container_width=True):
            st.error("This would delete all data in production. Demo mode: action simulated.")

    st.markdown("---")

    st.markdown("#### Privacy")

    st.markdown("""
    **Your data privacy matters to us:**
    - All health data is stored locally in this session
    - No data is sent to external servers (except AI APIs if configured)
    - Demo mode uses synthetic data only
    - You can export or delete your data at any time
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.checkbox("Allow analytics (anonymized)", value=False)
        st.checkbox("Share data with research (anonymized)", value=False)
    with col2:
        st.checkbox("Enable data backups", value=True)
        st.checkbox("Encrypt local storage", value=True)

# Footer
st.markdown("---")
st.markdown("### About HealthBridge")
st.markdown("""
**Version:** 0.1.0 (Demo)

**Built with:**
- Streamlit
- Plotly
- Python

**Contact:** support@healthbridge.demo

**License:** MIT
""")
