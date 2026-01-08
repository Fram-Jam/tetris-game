"""
Main Health Dashboard
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(page_title="Dashboard | HealthBridge", layout="wide", page_icon="🌉")


def create_sleep_chart(data: list) -> go.Figure:
    """Create sleep duration and quality chart."""
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Sleep Duration", "Sleep Stages"),
        row_heights=[0.4, 0.6]
    )

    # Sleep duration line
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df['sleep_duration'],
            mode='lines+markers',
            name='Total Sleep',
            line=dict(color='#6366F1', width=2),
            marker=dict(size=6)
        ),
        row=1, col=1
    )

    # Target sleep line
    fig.add_hline(y=7.5, line_dash="dash", line_color="gray",
                  annotation_text="Target: 7.5h", row=1, col=1)

    # Sleep stages stacked bar
    fig.add_trace(
        go.Bar(x=df['date'], y=df['deep_sleep'], name='Deep', marker_color='#1E3A8A'),
        row=2, col=1
    )
    fig.add_trace(
        go.Bar(x=df['date'], y=df['rem_sleep'], name='REM', marker_color='#3B82F6'),
        row=2, col=1
    )
    fig.add_trace(
        go.Bar(x=df['date'], y=df['light_sleep'], name='Light', marker_color='#93C5FD'),
        row=2, col=1
    )

    fig.update_layout(
        barmode='stack',
        height=500,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    fig.update_yaxes(title_text="Hours", row=1, col=1, gridcolor='#E2E8F0')
    fig.update_yaxes(title_text="Hours", row=2, col=1, gridcolor='#E2E8F0')
    fig.update_xaxes(gridcolor='#E2E8F0')

    return fig


def create_hrv_rhr_chart(data: list) -> go.Figure:
    """Create HRV and Resting HR chart."""
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # HRV
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df['hrv'],
            mode='lines+markers',
            name='HRV (ms)',
            line=dict(color='#10B981', width=2),
            marker=dict(size=5)
        ),
        secondary_y=False
    )

    # RHR
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df['resting_hr'],
            mode='lines+markers',
            name='Resting HR (bpm)',
            line=dict(color='#EF4444', width=2),
            marker=dict(size=5)
        ),
        secondary_y=True
    )

    fig.update_layout(
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    fig.update_yaxes(title_text="HRV (ms)", secondary_y=False, gridcolor='#E2E8F0')
    fig.update_yaxes(title_text="RHR (bpm)", secondary_y=True, gridcolor='#E2E8F0')
    fig.update_xaxes(gridcolor='#E2E8F0')

    return fig


def create_activity_chart(data: list) -> go.Figure:
    """Create steps and activity chart."""
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])

    fig = go.Figure()

    # Steps bar chart
    fig.add_trace(
        go.Bar(
            x=df['date'],
            y=df['steps'],
            name='Steps',
            marker_color='#8B5CF6'
        )
    )

    # 10k target line
    fig.add_hline(y=10000, line_dash="dash", line_color="gray",
                  annotation_text="Goal: 10,000")

    # 7-day moving average
    df['steps_ma'] = df['steps'].rolling(window=7, min_periods=1).mean()
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df['steps_ma'],
            mode='lines',
            name='7-day avg',
            line=dict(color='#C4B5FD', width=3)
        )
    )

    fig.update_layout(
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    fig.update_yaxes(gridcolor='#E2E8F0')
    fig.update_xaxes(gridcolor='#E2E8F0')

    return fig


def create_glucose_chart(data: list) -> go.Figure:
    """Create glucose chart if CGM data available."""
    # Filter to days with glucose data
    glucose_days = [d for d in data if d.get('glucose')]
    if not glucose_days:
        return None

    df = pd.DataFrame([
        {
            'date': d['date'],
            'avg': d['glucose']['avg'],
            'min': d['glucose']['min'],
            'max': d['glucose']['max'],
            'tir': d['glucose']['time_in_range']
        }
        for d in glucose_days
    ])
    df['date'] = pd.to_datetime(df['date'])

    fig = go.Figure()

    # Range band (min to max)
    fig.add_trace(
        go.Scatter(
            x=pd.concat([df['date'], df['date'][::-1]]),
            y=pd.concat([df['max'], df['min'][::-1]]),
            fill='toself',
            fillcolor='rgba(99, 102, 241, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Range',
            showlegend=True
        )
    )

    # Average line
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df['avg'],
            mode='lines+markers',
            name='Average',
            line=dict(color='#6366F1', width=2),
            marker=dict(size=6)
        )
    )

    # Target range
    fig.add_hrect(y0=70, y1=140, fillcolor="rgba(16, 185, 129, 0.1)",
                  line_width=0, annotation_text="Target Range")

    fig.update_layout(
        height=350,
        yaxis_title="Glucose (mg/dL)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    fig.update_yaxes(gridcolor='#E2E8F0')
    fig.update_xaxes(gridcolor='#E2E8F0')

    return fig


def create_readiness_chart(data: list) -> go.Figure:
    """Create readiness/recovery score chart."""
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])

    fig = go.Figure()

    # Color based on score
    colors = ['#EF4444' if s < 60 else '#F59E0B' if s < 75 else '#10B981'
              for s in df['readiness_score']]

    fig.add_trace(
        go.Bar(
            x=df['date'],
            y=df['readiness_score'],
            marker_color=colors,
            name='Readiness'
        )
    )

    fig.update_layout(
        height=300,
        yaxis_title="Score",
        yaxis_range=[0, 100],
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    fig.update_yaxes(gridcolor='#E2E8F0')
    fig.update_xaxes(gridcolor='#E2E8F0')

    return fig


# Initialize session state if not already done
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.health_data = None
    st.session_state.connected_devices = []

# Load data if needed
if not st.session_state.data_loaded:
    from src.data.synthetic.patient_generator import generate_demo_data
    from src.data.synthetic.lab_generator import generate_lab_history
    from datetime import datetime

    with st.spinner("Loading health data..."):
        patient, health_data = generate_demo_data(days=90)
        st.session_state.patient_profile = patient
        st.session_state.health_data = health_data
        labs = generate_lab_history(patient.id, patient.health_conditions, patient.age, patient.sex, num_panels=4)
        st.session_state.lab_data = labs
        st.session_state.connected_devices = [
            {'name': 'Oura Ring', 'type': 'oura', 'connected': True, 'last_sync': datetime.now()},
            {'name': 'Apple Watch', 'type': 'apple', 'connected': True, 'last_sync': datetime.now()},
            {'name': 'Dexcom G7', 'type': 'cgm', 'connected': True, 'last_sync': datetime.now()},
        ]
        st.session_state.data_loaded = True
        st.session_state.demo_mode = True

# Main dashboard
st.title("🏠 Health Dashboard")

# Date range selector
col1, col2, col3 = st.columns([2, 2, 6])
with col1:
    time_range = st.selectbox(
        "Time Range",
        ["Last 7 days", "Last 14 days", "Last 30 days", "Last 90 days"],
        index=2
    )
with col2:
    if st.button("🔄 Refresh Data"):
        st.session_state.data_loaded = False
        st.rerun()

# Filter data based on selection
days_map = {"Last 7 days": 7, "Last 14 days": 14, "Last 30 days": 30, "Last 90 days": 90}
days = days_map[time_range]

if st.session_state.health_data:
    data = st.session_state.health_data[-days:]

    # Today's summary
    st.markdown("### Today's Summary")
    today = data[-1]

    cols = st.columns(6)

    metrics = [
        ("😴 Sleep", f"{today['sleep_duration']:.1f}h", today['sleep_score']),
        ("❤️ HRV", f"{today['hrv']:.0f} ms", None),
        ("💓 RHR", f"{today['resting_hr']} bpm", None),
        ("🚶 Steps", f"{today['steps']:,}", None),
        ("🔥 Calories", f"{today['calories_active']}", None),
        ("⚡ Readiness", f"{today['readiness_score']}", None),
    ]

    for col, (label, value, score) in zip(cols, metrics):
        with col:
            st.metric(label, value)

    st.markdown("---")

    # Charts grid
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 😴 Sleep")
        st.plotly_chart(create_sleep_chart(data), use_container_width=True)

    with col2:
        st.markdown("### ❤️ Heart Health")
        st.plotly_chart(create_hrv_rhr_chart(data), use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🚶 Activity")
        st.plotly_chart(create_activity_chart(data), use_container_width=True)

    with col2:
        st.markdown("### ⚡ Readiness Score")
        st.plotly_chart(create_readiness_chart(data), use_container_width=True)

    # Glucose chart (if available)
    glucose_chart = create_glucose_chart(data)
    if glucose_chart:
        st.markdown("### 🩸 Glucose")
        st.plotly_chart(glucose_chart, use_container_width=True)

    # Data sources footer
    st.markdown("---")
    st.markdown("**Data Sources:** " + ", ".join([d['name'] for d in st.session_state.connected_devices]))
else:
    st.warning("No health data available. Please connect your devices in Settings.")
