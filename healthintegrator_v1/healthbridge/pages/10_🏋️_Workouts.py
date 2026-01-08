"""
Workouts Page

Track and analyze your exercise sessions.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, datetime, timedelta
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.synthetic.workout_generator import (
    Workout, WorkoutType, WORKOUT_CONFIGS,
    generate_workout_history, calculate_workout_stats, get_training_load
)
from src.visualizations.charts import create_calendar_heatmap, COLORS

st.set_page_config(page_title="Workouts", page_icon="🏋️", layout="wide")

st.title("🏋️ Workout Tracker")
st.markdown("*Track your exercise sessions and training progress*")

# Initialize session state
if 'workouts' not in st.session_state:
    st.session_state.workouts = generate_workout_history(days=90, workouts_per_week=4.5)


def format_duration(minutes: int) -> str:
    """Format duration in human-readable format."""
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours}h {mins}m" if mins > 0 else f"{hours}h"


def format_pace(pace: float) -> str:
    """Format pace as mm:ss/km."""
    if not pace:
        return "-"
    mins = int(pace)
    secs = int((pace - mins) * 60)
    return f"{mins}:{secs:02d}/km"


def render_workout_card(workout: Workout):
    """Render a workout summary card."""
    config = WORKOUT_CONFIGS[workout.workout_type]
    color = COLORS.get(workout.workout_type.value, COLORS['primary'])

    # Build details string
    details = []
    if workout.distance_km:
        details.append(f"📏 {workout.distance_km:.1f} km")
    if workout.pace_min_km:
        details.append(f"⏱️ {format_pace(workout.pace_min_km)}")
    if workout.avg_hr:
        details.append(f"❤️ {workout.avg_hr} bpm avg")
    if workout.elevation_gain:
        details.append(f"⛰️ {workout.elevation_gain}m gain")

    details_str = " • ".join(details) if details else ""

    st.markdown(f"""
    <div style="background: #1E293B; border-left: 4px solid {COLORS['primary']};
                border-radius: 0 8px 8px 0; padding: 1rem; margin-bottom: 0.5rem;">
        <div style="display: flex; justify-content: space-between; align-items: start;">
            <div>
                <h4 style="margin: 0;">{workout.icon} {workout.name}</h4>
                <p style="color: #9CA3AF; margin: 0.25rem 0; font-size: 0.85rem;">
                    {workout.date.strftime('%a, %b %d')} at {workout.start_time.strftime('%I:%M %p')}
                </p>
            </div>
            <div style="text-align: right;">
                <p style="margin: 0; font-size: 1.2rem; font-weight: bold;">{format_duration(workout.duration_minutes)}</p>
                <p style="color: #F59E0B; margin: 0; font-size: 0.9rem;">🔥 {workout.calories} cal</p>
            </div>
        </div>
        {f'<p style="color: #6B7280; margin: 0.5rem 0 0 0; font-size: 0.85rem;">{details_str}</p>' if details_str else ''}
    </div>
    """, unsafe_allow_html=True)


# Calculate stats
stats_7d = calculate_workout_stats(st.session_state.workouts, days=7)
stats_30d = calculate_workout_stats(st.session_state.workouts, days=30)
training_load = get_training_load(st.session_state.workouts, days=7)

# Overview metrics
st.markdown("### This Week")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Workouts", stats_7d['total_workouts'])

with col2:
    st.metric("Duration", format_duration(stats_7d['total_duration']))

with col3:
    st.metric("Calories", f"{stats_7d['total_calories']:,}")

with col4:
    if stats_7d['total_distance'] > 0:
        st.metric("Distance", f"{stats_7d['total_distance']:.1f} km")
    else:
        st.metric("Avg HR", f"{stats_7d['avg_hr']:.0f} bpm")

with col5:
    st.markdown(f"""
    <div style="background: {training_load['status_color']}22; border: 1px solid {training_load['status_color']};
                border-radius: 8px; padding: 0.5rem; text-align: center;">
        <p style="margin: 0; font-size: 0.8rem; color: #9CA3AF;">Training Status</p>
        <p style="margin: 0; color: {training_load['status_color']}; font-weight: bold;">
            {training_load['status'].title()}
        </p>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["📋 Recent", "📊 Analytics", "📅 Calendar", "➕ Log Workout"])

with tab1:
    st.markdown("### Recent Workouts")

    # Filter options
    col1, col2 = st.columns([2, 1])
    with col1:
        workout_filter = st.selectbox(
            "Filter by Type",
            options=['All'] + [wt.value.title() for wt in WorkoutType],
            index=0
        )

    with col2:
        time_filter = st.selectbox(
            "Time Period",
            options=['Last 7 Days', 'Last 14 Days', 'Last 30 Days', 'All Time'],
            index=2
        )

    # Apply filters
    filtered_workouts = st.session_state.workouts.copy()

    if workout_filter != 'All':
        filtered_workouts = [w for w in filtered_workouts
                           if w.workout_type.value == workout_filter.lower()]

    if time_filter == 'Last 7 Days':
        cutoff = date.today() - timedelta(days=7)
    elif time_filter == 'Last 14 Days':
        cutoff = date.today() - timedelta(days=14)
    elif time_filter == 'Last 30 Days':
        cutoff = date.today() - timedelta(days=30)
    else:
        cutoff = date.min

    filtered_workouts = [w for w in filtered_workouts if w.date >= cutoff]

    # Sort by date descending
    filtered_workouts = sorted(filtered_workouts, key=lambda w: w.start_time, reverse=True)

    if filtered_workouts:
        for workout in filtered_workouts[:15]:  # Show last 15
            render_workout_card(workout)

        if len(filtered_workouts) > 15:
            st.info(f"Showing 15 of {len(filtered_workouts)} workouts")
    else:
        st.info("No workouts found for the selected filters.")

with tab2:
    st.markdown("### Workout Analytics")

    # Time period selector
    analysis_period = st.selectbox(
        "Analysis Period",
        options=[30, 60, 90],
        format_func=lambda x: f"Last {x} Days",
        index=0,
        key="analysis_period"
    )

    cutoff = date.today() - timedelta(days=analysis_period)
    period_workouts = [w for w in st.session_state.workouts if w.date >= cutoff]

    if period_workouts:
        col1, col2 = st.columns(2)

        with col1:
            # Workouts by type
            st.markdown("#### By Workout Type")

            type_counts = {}
            for w in period_workouts:
                t = w.name
                if t not in type_counts:
                    type_counts[t] = {'count': 0, 'duration': 0, 'calories': 0}
                type_counts[t]['count'] += 1
                type_counts[t]['duration'] += w.duration_minutes
                type_counts[t]['calories'] += w.calories

            df_types = pd.DataFrame([
                {'Type': k, 'Count': v['count'], 'Duration (h)': round(v['duration']/60, 1),
                 'Calories': v['calories']}
                for k, v in type_counts.items()
            ])

            fig_types = px.pie(
                df_types,
                values='Count',
                names='Type',
                title='Workout Distribution',
                hole=0.4,
            )
            fig_types.update_layout(
                height=300,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig_types, use_container_width=True)

        with col2:
            # Weekly volume
            st.markdown("#### Weekly Volume")

            # Group by week
            df_workouts = pd.DataFrame([w.to_dict() for w in period_workouts])
            df_workouts['date'] = pd.to_datetime(df_workouts['date'])
            df_workouts['week'] = df_workouts['date'].dt.isocalendar().week

            weekly = df_workouts.groupby('week').agg({
                'duration_minutes': 'sum',
                'calories': 'sum',
                'id': 'count'
            }).rename(columns={'id': 'workouts'})

            fig_weekly = go.Figure()
            fig_weekly.add_trace(go.Bar(
                x=weekly.index,
                y=weekly['duration_minutes'] / 60,
                name='Duration (hours)',
                marker_color=COLORS['primary']
            ))

            fig_weekly.update_layout(
                title='Weekly Training Volume',
                xaxis_title='Week',
                yaxis_title='Hours',
                height=300,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig_weekly, use_container_width=True)

        # Heart rate zones (if HR data available)
        hr_workouts = [w for w in period_workouts if w.avg_hr]
        if hr_workouts:
            st.markdown("#### Heart Rate Analysis")

            col1, col2 = st.columns(2)

            with col1:
                # HR distribution
                hrs = [w.avg_hr for w in hr_workouts]
                fig_hr = go.Figure()
                fig_hr.add_trace(go.Histogram(
                    x=hrs,
                    nbinsx=15,
                    marker_color=COLORS['danger'],
                ))
                fig_hr.update_layout(
                    title='Average HR Distribution',
                    xaxis_title='BPM',
                    yaxis_title='Workouts',
                    height=250,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                )
                st.plotly_chart(fig_hr, use_container_width=True)

            with col2:
                # HR zones
                zones = {'Zone 1 (Recovery)': 0, 'Zone 2 (Aerobic)': 0,
                         'Zone 3 (Tempo)': 0, 'Zone 4 (Threshold)': 0, 'Zone 5 (Max)': 0}

                max_hr = 185
                for w in hr_workouts:
                    pct = w.avg_hr / max_hr
                    if pct < 0.6:
                        zones['Zone 1 (Recovery)'] += 1
                    elif pct < 0.7:
                        zones['Zone 2 (Aerobic)'] += 1
                    elif pct < 0.8:
                        zones['Zone 3 (Tempo)'] += 1
                    elif pct < 0.9:
                        zones['Zone 4 (Threshold)'] += 1
                    else:
                        zones['Zone 5 (Max)'] += 1

                zone_colors = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6']
                fig_zones = go.Figure()
                fig_zones.add_trace(go.Bar(
                    y=list(zones.keys()),
                    x=list(zones.values()),
                    orientation='h',
                    marker_color=zone_colors,
                ))
                fig_zones.update_layout(
                    title='Time in HR Zones',
                    xaxis_title='Workouts',
                    height=250,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                )
                st.plotly_chart(fig_zones, use_container_width=True)

        # Training load trend
        st.markdown("#### Training Load Trend")

        # Calculate weekly load
        df_load = df_workouts.copy()
        df_load['intensity'] = df_load['avg_hr'].fillna(140) / 185
        df_load['load'] = df_load['duration_minutes'] * df_load['intensity']

        weekly_load = df_load.groupby('week')['load'].sum()

        fig_load = go.Figure()
        fig_load.add_trace(go.Scatter(
            x=weekly_load.index,
            y=weekly_load.values,
            mode='lines+markers',
            name='Training Load',
            line=dict(color=COLORS['primary'], width=3),
            marker=dict(size=8)
        ))

        # Add trend line
        if len(weekly_load) >= 3:
            z = pd.np.polyfit(range(len(weekly_load)), weekly_load.values, 1) if hasattr(pd, 'np') else None
            if z is None:
                import numpy as np
                z = np.polyfit(range(len(weekly_load)), weekly_load.values, 1)
                p = np.poly1d(z)
                fig_load.add_trace(go.Scatter(
                    x=weekly_load.index,
                    y=p(range(len(weekly_load))),
                    mode='lines',
                    name='Trend',
                    line=dict(color='#E2E8F0', dash='dash', width=2)
                ))

        fig_load.update_layout(
            title='Weekly Training Load',
            xaxis_title='Week',
            yaxis_title='Load (arbitrary units)',
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig_load, use_container_width=True)

    else:
        st.info("No workout data for the selected period.")

with tab3:
    st.markdown("### Workout Calendar")

    # Create workout frequency data
    df_cal = pd.DataFrame([w.to_dict() for w in st.session_state.workouts])

    if not df_cal.empty:
        df_cal['date'] = pd.to_datetime(df_cal['date'])

        # Aggregate by day
        daily_workouts = df_cal.groupby('date').agg({
            'duration_minutes': 'sum',
            'calories': 'sum',
            'id': 'count'
        }).reset_index()
        daily_workouts.columns = ['date', 'duration', 'calories', 'count']

        # Calendar heatmap
        metric_choice = st.radio(
            "Show",
            options=['duration', 'calories', 'count'],
            format_func=lambda x: {'duration': 'Duration (min)', 'calories': 'Calories', 'count': 'Workout Count'}[x],
            horizontal=True
        )

        heatmap_fig = create_calendar_heatmap(
            daily_workouts,
            date_col='date',
            value_col=metric_choice,
            title=f'Workout {metric_choice.title()} Heatmap',
            colorscale='Greens',
            weeks_to_show=12
        )
        st.plotly_chart(heatmap_fig, use_container_width=True)

        # Day of week patterns
        st.markdown("#### Best Training Days")
        df_cal['dayofweek'] = df_cal['date'].dt.day_name()

        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_counts = df_cal['dayofweek'].value_counts().reindex(day_order).fillna(0)

        fig_dow = go.Figure()
        fig_dow.add_trace(go.Bar(
            x=day_order,
            y=dow_counts.values,
            marker_color=[COLORS['primary'] if d in ['Tuesday', 'Thursday', 'Saturday'] else COLORS['secondary']
                          for d in day_order]
        ))
        fig_dow.update_layout(
            title='Workouts by Day of Week',
            height=250,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig_dow, use_container_width=True)

with tab4:
    st.markdown("### Log a Workout")

    with st.form("log_workout"):
        col1, col2 = st.columns(2)

        with col1:
            workout_type = st.selectbox(
                "Workout Type",
                options=list(WorkoutType),
                format_func=lambda x: f"{WORKOUT_CONFIGS[x]['icon']} {WORKOUT_CONFIGS[x]['name']}"
            )

            workout_date = st.date_input("Date", value=date.today())

            workout_time = st.time_input("Start Time", value=datetime.now().time())

        with col2:
            duration = st.number_input("Duration (minutes)", min_value=1, max_value=600, value=45)

            calories = st.number_input("Calories", min_value=0, max_value=5000, value=300)

            avg_hr = st.number_input("Average Heart Rate", min_value=0, max_value=220, value=0)

        # Conditional fields based on workout type
        config = WORKOUT_CONFIGS[workout_type]

        col3, col4 = st.columns(2)

        with col3:
            if config.get('has_distance'):
                distance = st.number_input("Distance (km)", min_value=0.0, max_value=500.0, value=0.0)
            else:
                distance = None

        with col4:
            if workout_type in [WorkoutType.RUN, WorkoutType.CYCLE, WorkoutType.HIKING]:
                elevation = st.number_input("Elevation Gain (m)", min_value=0, max_value=5000, value=0)
            else:
                elevation = None

        notes = st.text_area("Notes (optional)", placeholder="How did it feel?")

        submitted = st.form_submit_button("Log Workout", type="primary")

        if submitted:
            # Calculate pace if distance provided
            pace = None
            if distance and distance > 0:
                pace = round(duration / distance, 2)

            new_workout = Workout(
                id=f"workout_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                workout_type=workout_type,
                date=workout_date,
                start_time=datetime.combine(workout_date, workout_time),
                duration_minutes=duration,
                calories=calories,
                avg_hr=avg_hr if avg_hr > 0 else None,
                max_hr=int(avg_hr * 1.1) if avg_hr > 0 else None,
                distance_km=distance if distance and distance > 0 else None,
                pace_min_km=pace,
                elevation_gain=elevation if elevation and elevation > 0 else None,
                notes=notes,
                source='manual',
            )

            st.session_state.workouts.append(new_workout)
            st.session_state.workouts.sort(key=lambda w: w.start_time)
            st.success(f"Logged {config['name']} workout!")
            st.rerun()

# Sidebar
with st.sidebar:
    st.markdown("### Quick Stats")

    st.markdown(f"""
    **Last 30 Days:**
    - Workouts: {stats_30d['total_workouts']}
    - Total Time: {format_duration(stats_30d['total_duration'])}
    - Calories: {stats_30d['total_calories']:,}
    - Distance: {stats_30d['total_distance']:.1f} km
    """)

    st.divider()

    st.markdown("### Training Status")
    st.markdown(f"""
    **Current Load:** {training_load['current_load']:.0f}

    **vs Last Week:** {training_load['change_pct']:+.1f}%

    **Status:** {training_load['status'].title()}
    """)

    # Status explanation
    status_tips = {
        'overreaching': "Consider a recovery day. Training load increased significantly.",
        'productive': "Good balance of training stimulus and recovery.",
        'maintaining': "Stable training load. Consider progressive overload.",
        'detraining': "Training load decreased. Increase volume if healthy.",
    }
    st.info(status_tips.get(training_load['status'], ""))
