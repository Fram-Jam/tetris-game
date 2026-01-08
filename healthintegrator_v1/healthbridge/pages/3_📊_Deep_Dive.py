"""
Deep Dive Analysis Page - Detailed metric analysis and correlations
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(page_title="Deep Dive | HealthBridge", layout="wide", page_icon="🌉")

# Initialize and load data
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

if not st.session_state.data_loaded:
    from src.data.synthetic.patient_generator import generate_demo_data
    from src.data.synthetic.lab_generator import generate_lab_history

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

st.title("📊 Deep Dive Analysis")
st.markdown("Explore correlations and patterns in your health data")

if st.session_state.health_data:
    df = pd.DataFrame(st.session_state.health_data)
    df['date'] = pd.to_datetime(df['date'])

    # Metric selector
    col1, col2 = st.columns([1, 3])
    with col1:
        metric_options = {
            'Sleep Duration': 'sleep_duration',
            'Sleep Score': 'sleep_score',
            'HRV': 'hrv',
            'Resting HR': 'resting_hr',
            'Steps': 'steps',
            'Active Minutes': 'active_minutes',
            'Readiness Score': 'readiness_score',
            'Calories': 'calories_active'
        }
        selected_metric = st.selectbox("Select Primary Metric", list(metric_options.keys()))
        metric_col = metric_options[selected_metric]

    st.markdown("---")

    # Three analysis tabs
    tab1, tab2, tab3 = st.tabs(["📈 Trend Analysis", "🔗 Correlations", "📅 Day-of-Week Patterns"])

    with tab1:
        st.markdown(f"### {selected_metric} Over Time")

        # Time range
        time_range = st.radio(
            "Time Range",
            ["Last 7 days", "Last 14 days", "Last 30 days", "Last 90 days"],
            horizontal=True,
            index=2
        )
        days_map = {"Last 7 days": 7, "Last 14 days": 14, "Last 30 days": 30, "Last 90 days": 90}
        days = days_map[time_range]
        filtered_df = df.tail(days)

        # Create trend chart
        fig = go.Figure()

        # Main metric line
        fig.add_trace(go.Scatter(
            x=filtered_df['date'],
            y=filtered_df[metric_col],
            mode='lines+markers',
            name=selected_metric,
            line=dict(color='#6366F1', width=2),
            marker=dict(size=6)
        ))

        # Add moving average
        ma_window = min(7, len(filtered_df))
        ma_values = filtered_df[metric_col].rolling(window=ma_window, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=filtered_df['date'],
            y=ma_values,
            mode='lines',
            name=f'{ma_window}-day avg',
            line=dict(color='#C4B5FD', width=3, dash='dash')
        ))

        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        fig.update_yaxes(gridcolor='#E2E8F0')
        fig.update_xaxes(gridcolor='#E2E8F0')

        st.plotly_chart(fig, use_container_width=True)

        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Average", f"{filtered_df[metric_col].mean():.1f}")
        with col2:
            st.metric("Min", f"{filtered_df[metric_col].min():.1f}")
        with col3:
            st.metric("Max", f"{filtered_df[metric_col].max():.1f}")
        with col4:
            std = filtered_df[metric_col].std()
            st.metric("Std Dev", f"{std:.1f}")

        # Trend analysis
        st.markdown("#### Trend Analysis")
        first_half = filtered_df[metric_col].head(len(filtered_df)//2).mean()
        second_half = filtered_df[metric_col].tail(len(filtered_df)//2).mean()
        trend_pct = ((second_half - first_half) / first_half) * 100 if first_half > 0 else 0

        if trend_pct > 5:
            st.success(f"📈 **Improving**: Your {selected_metric.lower()} has increased by {trend_pct:.1f}% over this period.")
        elif trend_pct < -5:
            st.warning(f"📉 **Declining**: Your {selected_metric.lower()} has decreased by {abs(trend_pct):.1f}% over this period.")
        else:
            st.info(f"➡️ **Stable**: Your {selected_metric.lower()} has remained relatively stable (±{abs(trend_pct):.1f}%).")

    with tab2:
        st.markdown("### Correlation Analysis")
        st.markdown("Discover how different health metrics relate to each other")

        # Correlation heatmap
        corr_cols = ['sleep_duration', 'sleep_score', 'hrv', 'resting_hr', 'steps', 'active_minutes', 'readiness_score']
        corr_df = df[corr_cols].corr()

        # Rename for display
        display_names = {
            'sleep_duration': 'Sleep Duration',
            'sleep_score': 'Sleep Score',
            'hrv': 'HRV',
            'resting_hr': 'Resting HR',
            'steps': 'Steps',
            'active_minutes': 'Active Minutes',
            'readiness_score': 'Readiness'
        }
        corr_df.index = [display_names.get(c, c) for c in corr_df.index]
        corr_df.columns = [display_names.get(c, c) for c in corr_df.columns]

        fig = px.imshow(
            corr_df,
            color_continuous_scale='RdBu_r',
            aspect='auto',
            zmin=-1, zmax=1
        )
        fig.update_layout(
            height=500,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Notable correlations
        st.markdown("#### Key Insights")

        # Find strongest correlations
        correlations = []
        for i, col1 in enumerate(corr_cols):
            for j, col2 in enumerate(corr_cols):
                if i < j:
                    corr = df[col1].corr(df[col2])
                    if abs(corr) > 0.3:
                        correlations.append({
                            'metric1': display_names[col1],
                            'metric2': display_names[col2],
                            'correlation': corr
                        })

        correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)

        for corr in correlations[:5]:
            direction = "positively" if corr['correlation'] > 0 else "negatively"
            strength = "strongly" if abs(corr['correlation']) > 0.5 else "moderately"

            icon = "🟢" if corr['correlation'] > 0 else "🔴"
            st.markdown(f"{icon} **{corr['metric1']}** and **{corr['metric2']}** are {strength} {direction} correlated (r={corr['correlation']:.2f})")

        # Scatter plot for selected metrics
        st.markdown("#### Explore Relationship")
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            x_metric = st.selectbox("X-axis", list(metric_options.keys()), index=0)
        with col2:
            y_metric = st.selectbox("Y-axis", list(metric_options.keys()), index=2)
        with col3:
            show_trendline = st.checkbox("Trendline", value=True)

        # Try to create scatter with trendline, fall back to no trendline
        try:
            if show_trendline:
                fig = px.scatter(
                    df,
                    x=metric_options[x_metric],
                    y=metric_options[y_metric],
                    trendline="ols",
                    labels={metric_options[x_metric]: x_metric, metric_options[y_metric]: y_metric}
                )
            else:
                fig = px.scatter(
                    df,
                    x=metric_options[x_metric],
                    y=metric_options[y_metric],
                    labels={metric_options[x_metric]: x_metric, metric_options[y_metric]: y_metric}
                )
        except Exception:
            # Fallback without trendline if statsmodels not available
            fig = px.scatter(
                df,
                x=metric_options[x_metric],
                y=metric_options[y_metric],
                labels={metric_options[x_metric]: x_metric, metric_options[y_metric]: y_metric}
            )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("### Day-of-Week Patterns")
        st.markdown("See how your metrics vary throughout the week")

        # Add day of week
        df['day_of_week'] = df['date'].dt.day_name()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

        # Aggregate by day
        daily_avg = df.groupby('day_of_week')[metric_col].mean().reindex(day_order)

        fig = go.Figure()
        colors = ['#6366F1' if d in ['Saturday', 'Sunday'] else '#A5B4FC' for d in day_order]

        fig.add_trace(go.Bar(
            x=day_order,
            y=daily_avg.values,
            marker_color=colors,
            text=[f"{v:.1f}" for v in daily_avg.values],
            textposition='outside'
        ))

        # Add average line
        overall_avg = df[metric_col].mean()
        fig.add_hline(y=overall_avg, line_dash="dash", line_color="#EF4444",
                      annotation_text=f"Avg: {overall_avg:.1f}")

        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        fig.update_yaxes(gridcolor='#E2E8F0')

        st.plotly_chart(fig, use_container_width=True)

        # Best and worst days
        best_day = daily_avg.idxmax()
        worst_day = daily_avg.idxmin()

        col1, col2 = st.columns(2)
        with col1:
            st.success(f"🏆 **Best Day**: {best_day} ({daily_avg[best_day]:.1f})")
        with col2:
            st.warning(f"📉 **Lowest Day**: {worst_day} ({daily_avg[worst_day]:.1f})")

        # Weekend vs weekday comparison
        st.markdown("#### Weekend vs Weekday")
        weekend_avg = df[df['date'].dt.dayofweek >= 5][metric_col].mean()
        weekday_avg = df[df['date'].dt.dayofweek < 5][metric_col].mean()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Weekday Average", f"{weekday_avg:.1f}")
        with col2:
            st.metric("Weekend Average", f"{weekend_avg:.1f}")
        with col3:
            diff = weekend_avg - weekday_avg
            diff_pct = (diff / weekday_avg) * 100 if weekday_avg > 0 else 0
            st.metric("Difference", f"{diff:+.1f}", f"{diff_pct:+.1f}%")

else:
    st.warning("No health data available. Please connect your devices to analyze your data.")
