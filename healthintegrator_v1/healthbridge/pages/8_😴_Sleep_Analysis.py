"""
Sleep Analysis Deep Dive Page

Comprehensive sleep analysis with trends, patterns, and recommendations.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import date, timedelta
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.synthetic.patient_generator import generate_synthetic_patient
from src.visualizations.charts import (
    create_calendar_heatmap, create_sleep_stages_chart,
    create_distribution_chart, create_radar_chart, COLORS
)

st.set_page_config(page_title="Sleep Analysis", page_icon="😴", layout="wide")

st.title("😴 Sleep Analysis")
st.markdown("*Deep dive into your sleep patterns and quality*")

# Initialize session state
if 'health_data' not in st.session_state:
    patient = generate_synthetic_patient(days=90)
    st.session_state.health_data = patient['daily_summaries']


def calculate_sleep_stats(df: pd.DataFrame) -> dict:
    """Calculate comprehensive sleep statistics."""
    stats = {}

    if 'sleep_duration' in df.columns:
        sleep_vals = df['sleep_duration'].dropna()
        if len(sleep_vals) > 0:
            stats['avg_duration'] = sleep_vals.mean()
            stats['total_hours'] = sleep_vals.sum()
            stats['best_night'] = sleep_vals.max()
            stats['worst_night'] = sleep_vals.min()
            stats['std_duration'] = sleep_vals.std()
            stats['nights_over_7h'] = (sleep_vals >= 7).sum()
            stats['nights_under_6h'] = (sleep_vals < 6).sum()

    if 'sleep_score' in df.columns:
        score_vals = df['sleep_score'].dropna()
        if len(score_vals) > 0:
            stats['avg_score'] = score_vals.mean()
            stats['best_score'] = score_vals.max()
            stats['worst_score'] = score_vals.min()

    if 'deep_sleep' in df.columns:
        deep = df['deep_sleep'].dropna()
        if len(deep) > 0:
            stats['avg_deep'] = deep.mean()
            stats['deep_pct'] = (deep.sum() / df['sleep_duration'].sum() * 100) if df['sleep_duration'].sum() > 0 else 0

    if 'rem_sleep' in df.columns:
        rem = df['rem_sleep'].dropna()
        if len(rem) > 0:
            stats['avg_rem'] = rem.mean()
            stats['rem_pct'] = (rem.sum() / df['sleep_duration'].sum() * 100) if df['sleep_duration'].sum() > 0 else 0

    if 'light_sleep' in df.columns:
        light = df['light_sleep'].dropna()
        if len(light) > 0:
            stats['avg_light'] = light.mean()
            stats['light_pct'] = (light.sum() / df['sleep_duration'].sum() * 100) if df['sleep_duration'].sum() > 0 else 0

    return stats


def get_sleep_grade(stats: dict) -> tuple:
    """Get letter grade and color for sleep performance."""
    score = 0
    max_score = 0

    # Duration (40 points)
    if 'avg_duration' in stats:
        max_score += 40
        avg = stats['avg_duration']
        if avg >= 7.5:
            score += 40
        elif avg >= 7:
            score += 35
        elif avg >= 6.5:
            score += 25
        elif avg >= 6:
            score += 15
        else:
            score += 5

    # Consistency (20 points)
    if 'std_duration' in stats:
        max_score += 20
        std = stats['std_duration']
        if std < 0.5:
            score += 20
        elif std < 1.0:
            score += 15
        elif std < 1.5:
            score += 10
        else:
            score += 5

    # Deep sleep (20 points)
    if 'deep_pct' in stats:
        max_score += 20
        deep_pct = stats['deep_pct']
        if deep_pct >= 20:
            score += 20
        elif deep_pct >= 15:
            score += 15
        elif deep_pct >= 10:
            score += 10
        else:
            score += 5

    # REM sleep (20 points)
    if 'rem_pct' in stats:
        max_score += 20
        rem_pct = stats['rem_pct']
        if rem_pct >= 25:
            score += 20
        elif rem_pct >= 20:
            score += 15
        elif rem_pct >= 15:
            score += 10
        else:
            score += 5

    # Calculate percentage
    if max_score > 0:
        pct = (score / max_score) * 100
    else:
        pct = 0

    if pct >= 90:
        return 'A', '#10B981', pct
    elif pct >= 80:
        return 'B+', '#34D399', pct
    elif pct >= 70:
        return 'B', '#60A5FA', pct
    elif pct >= 60:
        return 'C', '#FBBF24', pct
    else:
        return 'D', '#EF4444', pct


def generate_sleep_insights(stats: dict, df: pd.DataFrame) -> list:
    """Generate personalized sleep insights."""
    insights = []

    # Duration insights
    if 'avg_duration' in stats:
        avg = stats['avg_duration']
        if avg < 6:
            insights.append({
                'type': 'warning',
                'icon': '⚠️',
                'title': 'Sleep Deficit',
                'text': f'Your average of {avg:.1f}h is significantly below recommended. Aim for 7-9 hours.'
            })
        elif avg < 7:
            insights.append({
                'type': 'info',
                'icon': '💡',
                'title': 'Room for Improvement',
                'text': f'Averaging {avg:.1f}h. An extra 30-60 minutes could improve recovery.'
            })
        elif avg >= 7.5:
            insights.append({
                'type': 'success',
                'icon': '✅',
                'title': 'Optimal Duration',
                'text': f'Averaging {avg:.1f}h puts you in the optimal range for recovery.'
            })

    # Consistency insights
    if 'std_duration' in stats:
        std = stats['std_duration']
        if std > 1.5:
            insights.append({
                'type': 'warning',
                'icon': '⚠️',
                'title': 'Irregular Sleep Schedule',
                'text': 'Your sleep timing varies significantly. Consistent bedtimes improve sleep quality.'
            })
        elif std < 0.5:
            insights.append({
                'type': 'success',
                'icon': '✅',
                'title': 'Excellent Consistency',
                'text': 'Very consistent sleep schedule helps maintain healthy circadian rhythm.'
            })

    # Deep sleep insights
    if 'deep_pct' in stats:
        deep = stats['deep_pct']
        if deep < 10:
            insights.append({
                'type': 'warning',
                'icon': '⚠️',
                'title': 'Low Deep Sleep',
                'text': f'Only {deep:.0f}% deep sleep. Try avoiding alcohol and screens before bed.'
            })
        elif deep >= 20:
            insights.append({
                'type': 'success',
                'icon': '✅',
                'title': 'Strong Deep Sleep',
                'text': f'{deep:.0f}% deep sleep indicates good physical recovery.'
            })

    # Day-of-week pattern
    if 'date' in df.columns:
        df_copy = df.copy()
        df_copy['dayofweek'] = pd.to_datetime(df_copy['date']).dt.dayofweek
        if 'sleep_duration' in df_copy.columns:
            day_avg = df_copy.groupby('dayofweek')['sleep_duration'].mean()
            worst_day = day_avg.idxmin()
            best_day = day_avg.idxmax()
            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            if day_avg[worst_day] < day_avg[best_day] - 1:
                insights.append({
                    'type': 'info',
                    'icon': '📊',
                    'title': 'Weekly Pattern',
                    'text': f'You sleep least on {days[worst_day]}s ({day_avg[worst_day]:.1f}h) and most on {days[best_day]}s ({day_avg[best_day]:.1f}h).'
                })

    return insights


# Main content
df = pd.DataFrame(st.session_state.health_data)
df['date'] = pd.to_datetime(df['date'])

# Date range filter
col1, col2 = st.columns([3, 1])
with col1:
    date_range = st.selectbox(
        "Time Period",
        options=['Last 7 Days', 'Last 14 Days', 'Last 30 Days', 'Last 90 Days', 'All Time'],
        index=2
    )

# Filter data
if date_range == 'Last 7 Days':
    cutoff = df['date'].max() - timedelta(days=7)
elif date_range == 'Last 14 Days':
    cutoff = df['date'].max() - timedelta(days=14)
elif date_range == 'Last 30 Days':
    cutoff = df['date'].max() - timedelta(days=30)
elif date_range == 'Last 90 Days':
    cutoff = df['date'].max() - timedelta(days=90)
else:
    cutoff = df['date'].min()

filtered_df = df[df['date'] >= cutoff].copy()

# Calculate stats
stats = calculate_sleep_stats(filtered_df)
grade, grade_color, grade_score = get_sleep_grade(stats)

# Overview metrics
st.markdown("### Overview")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {grade_color}22 0%, {grade_color}11 100%);
                border: 2px solid {grade_color}; border-radius: 12px; padding: 1rem; text-align: center;">
        <h2 style="font-size: 2.5rem; margin: 0; color: {grade_color};">{grade}</h2>
        <p style="color: #9CA3AF; margin: 0;">Sleep Grade</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    avg_duration = stats.get('avg_duration', 0)
    delta = avg_duration - 7
    st.metric(
        "Avg Duration",
        f"{avg_duration:.1f}h",
        delta=f"{delta:+.1f}h vs 7h goal" if delta != 0 else "On target"
    )

with col3:
    avg_score = stats.get('avg_score', 0)
    st.metric("Avg Score", f"{avg_score:.0f}")

with col4:
    nights_over = stats.get('nights_over_7h', 0)
    total_nights = len(filtered_df)
    pct = (nights_over / total_nights * 100) if total_nights > 0 else 0
    st.metric("Nights 7h+", f"{nights_over}/{total_nights}", delta=f"{pct:.0f}%")

with col5:
    consistency = 100 - min(100, stats.get('std_duration', 0) * 30)
    st.metric("Consistency", f"{consistency:.0f}%")

st.divider()

# Tabs for different views
tab1, tab2, tab3, tab4 = st.tabs(["📈 Trends", "📊 Patterns", "🎯 Stages", "💡 Insights"])

with tab1:
    st.markdown("### Sleep Trends")

    # Main trend chart
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.6, 0.4],
        subplot_titles=('Sleep Duration', 'Sleep Score')
    )

    # Duration chart with target zone
    fig.add_hrect(
        y0=7, y1=9,
        fillcolor="rgba(16, 185, 129, 0.1)",
        line_width=0,
        row=1, col=1
    )

    fig.add_trace(go.Scatter(
        x=filtered_df['date'],
        y=filtered_df['sleep_duration'],
        mode='lines+markers',
        name='Duration',
        line=dict(color=COLORS['primary'], width=2),
        marker=dict(size=6)
    ), row=1, col=1)

    # Moving average
    ma_values = filtered_df['sleep_duration'].rolling(window=7, min_periods=1).mean()
    fig.add_trace(go.Scatter(
        x=filtered_df['date'],
        y=ma_values,
        mode='lines',
        name='7-day avg',
        line=dict(color='#E2E8F0', width=3, dash='dash')
    ), row=1, col=1)

    # Sleep score
    if 'sleep_score' in filtered_df.columns:
        fig.add_trace(go.Scatter(
            x=filtered_df['date'],
            y=filtered_df['sleep_score'],
            mode='lines+markers',
            name='Score',
            line=dict(color='#8B5CF6', width=2),
            marker=dict(size=6)
        ), row=2, col=1)

    fig.update_layout(
        height=500,
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    fig.update_yaxes(title_text='Hours', row=1, col=1)
    fig.update_yaxes(title_text='Score', row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # Calendar heatmap
    st.markdown("### Sleep Calendar")
    col1, col2 = st.columns(2)

    with col1:
        heatmap_metric = st.selectbox(
            "Heatmap Metric",
            options=['sleep_duration', 'sleep_score', 'deep_sleep'],
            format_func=lambda x: x.replace('_', ' ').title()
        )

    with col2:
        colorscale = st.selectbox(
            "Color Scale",
            options=['Viridis', 'Blues', 'Greens', 'Purples', 'RdYlGn'],
            index=0
        )

    if heatmap_metric in filtered_df.columns:
        heatmap_fig = create_calendar_heatmap(
            filtered_df,
            date_col='date',
            value_col=heatmap_metric,
            title=f'{heatmap_metric.replace("_", " ").title()} Heatmap',
            colorscale=colorscale,
            weeks_to_show=12
        )
        st.plotly_chart(heatmap_fig, use_container_width=True)

with tab2:
    st.markdown("### Sleep Patterns")

    col1, col2 = st.columns(2)

    with col1:
        # Day of week pattern
        st.markdown("#### By Day of Week")
        df_dow = filtered_df.copy()
        df_dow['dayofweek'] = df_dow['date'].dt.dayofweek
        df_dow['day_name'] = df_dow['date'].dt.strftime('%a')

        day_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        day_stats = df_dow.groupby('day_name').agg({
            'sleep_duration': 'mean',
            'sleep_score': 'mean'
        }).reindex(day_order)

        fig_dow = go.Figure()
        fig_dow.add_trace(go.Bar(
            x=day_order,
            y=day_stats['sleep_duration'],
            name='Avg Duration',
            marker_color=COLORS['primary']
        ))

        fig_dow.add_hline(y=7, line_dash='dash', line_color='#10B981',
                         annotation_text='7h Target')

        fig_dow.update_layout(
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis_title='Hours',
        )
        st.plotly_chart(fig_dow, use_container_width=True)

    with col2:
        # Distribution
        st.markdown("#### Duration Distribution")
        if 'sleep_duration' in filtered_df.columns:
            dist_fig = create_distribution_chart(
                filtered_df['sleep_duration'].dropna().tolist(),
                title='Sleep Duration Distribution',
                unit='h',
                color=COLORS['primary']
            )
            st.plotly_chart(dist_fig, use_container_width=True)

    # Correlation with other metrics
    st.markdown("#### Sleep vs Other Metrics")
    col1, col2 = st.columns(2)

    with col1:
        # Sleep vs HRV
        if 'hrv' in filtered_df.columns and 'sleep_duration' in filtered_df.columns:
            fig_hrv = px.scatter(
                filtered_df,
                x='sleep_duration',
                y='hrv',
                trendline='ols',
                title='Sleep Duration vs HRV',
                labels={'sleep_duration': 'Sleep (hours)', 'hrv': 'HRV (ms)'}
            )
            fig_hrv.update_layout(
                height=300,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
            )
            fig_hrv.update_traces(marker=dict(color=COLORS['primary']))
            st.plotly_chart(fig_hrv, use_container_width=True)

    with col2:
        # Sleep vs Readiness
        if 'readiness_score' in filtered_df.columns and 'sleep_duration' in filtered_df.columns:
            fig_ready = px.scatter(
                filtered_df,
                x='sleep_duration',
                y='readiness_score',
                trendline='ols',
                title='Sleep Duration vs Readiness',
                labels={'sleep_duration': 'Sleep (hours)', 'readiness_score': 'Readiness'}
            )
            fig_ready.update_layout(
                height=300,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
            )
            fig_ready.update_traces(marker=dict(color='#8B5CF6'))
            st.plotly_chart(fig_ready, use_container_width=True)

with tab3:
    st.markdown("### Sleep Stages")

    # Stages over time
    if all(col in filtered_df.columns for col in ['deep_sleep', 'rem_sleep', 'light_sleep']):
        stages_fig = create_sleep_stages_chart(filtered_df, date_col='date')
        st.plotly_chart(stages_fig, use_container_width=True)

        # Stage breakdown
        st.markdown("#### Stage Breakdown")
        col1, col2, col3 = st.columns(3)

        with col1:
            deep_pct = stats.get('deep_pct', 0)
            avg_deep = stats.get('avg_deep', 0)
            st.markdown(f"""
            <div style="background: {COLORS['deep_sleep']}22; border-left: 4px solid {COLORS['deep_sleep']};
                        padding: 1rem; border-radius: 0 8px 8px 0;">
                <h4 style="color: {COLORS['deep_sleep']}; margin: 0;">Deep Sleep</h4>
                <p style="font-size: 2rem; margin: 0.5rem 0;">{avg_deep:.1f}h</p>
                <p style="color: #9CA3AF; margin: 0;">{deep_pct:.0f}% of total sleep</p>
                <p style="font-size: 0.8rem; color: #6B7280; margin-top: 0.5rem;">
                    Target: 13-23% • Physical recovery & memory consolidation
                </p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            rem_pct = stats.get('rem_pct', 0)
            avg_rem = stats.get('avg_rem', 0)
            st.markdown(f"""
            <div style="background: {COLORS['rem_sleep']}22; border-left: 4px solid {COLORS['rem_sleep']};
                        padding: 1rem; border-radius: 0 8px 8px 0;">
                <h4 style="color: {COLORS['rem_sleep']}; margin: 0;">REM Sleep</h4>
                <p style="font-size: 2rem; margin: 0.5rem 0;">{avg_rem:.1f}h</p>
                <p style="color: #9CA3AF; margin: 0;">{rem_pct:.0f}% of total sleep</p>
                <p style="font-size: 0.8rem; color: #6B7280; margin-top: 0.5rem;">
                    Target: 20-25% • Cognitive function & emotional processing
                </p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            light_pct = stats.get('light_pct', 0)
            avg_light = stats.get('avg_light', 0)
            st.markdown(f"""
            <div style="background: {COLORS['light_sleep']}22; border-left: 4px solid {COLORS['light_sleep']};
                        padding: 1rem; border-radius: 0 8px 8px 0;">
                <h4 style="color: {COLORS['light_sleep']}; margin: 0;">Light Sleep</h4>
                <p style="font-size: 2rem; margin: 0.5rem 0;">{avg_light:.1f}h</p>
                <p style="color: #9CA3AF; margin: 0;">{light_pct:.0f}% of total sleep</p>
                <p style="font-size: 0.8rem; color: #6B7280; margin-top: 0.5rem;">
                    Target: 50-60% • Transition stage & light restoration
                </p>
            </div>
            """, unsafe_allow_html=True)

        # Radar chart for sleep profile
        st.markdown("#### Your Sleep Profile")
        col1, col2 = st.columns([1, 2])

        with col1:
            categories = ['Duration', 'Deep Sleep', 'REM Sleep', 'Consistency', 'Score']
            values = [
                min(100, (stats.get('avg_duration', 0) / 8) * 100),
                min(100, (stats.get('deep_pct', 0) / 20) * 100),
                min(100, (stats.get('rem_pct', 0) / 25) * 100),
                100 - min(100, stats.get('std_duration', 0) * 30),
                stats.get('avg_score', 0),
            ]

            radar_fig = create_radar_chart(categories, values, title='Sleep Profile')
            st.plotly_chart(radar_fig, use_container_width=True)

        with col2:
            st.markdown("""
            **Understanding Your Sleep Profile:**

            - **Duration**: Are you getting enough total sleep?
            - **Deep Sleep**: Physical recovery and immune function
            - **REM Sleep**: Memory, learning, and emotional health
            - **Consistency**: How regular is your sleep schedule?
            - **Score**: Overall sleep quality metric

            *A balanced profile with all metrics above 70% indicates optimal sleep.*
            """)
    else:
        st.info("Sleep stage data not available. Connect a device that tracks sleep stages.")

with tab4:
    st.markdown("### Sleep Insights")

    insights = generate_sleep_insights(stats, filtered_df)

    if insights:
        for insight in insights:
            bg_colors = {
                'success': '#10B98111',
                'warning': '#F59E0B11',
                'info': '#3B82F611',
            }
            border_colors = {
                'success': '#10B981',
                'warning': '#F59E0B',
                'info': '#3B82F6',
            }

            st.markdown(f"""
            <div style="background: {bg_colors.get(insight['type'], '#3B82F611')};
                        border-left: 4px solid {border_colors.get(insight['type'], '#3B82F6')};
                        padding: 1rem; border-radius: 0 8px 8px 0; margin-bottom: 1rem;">
                <h4 style="margin: 0;">{insight['icon']} {insight['title']}</h4>
                <p style="margin: 0.5rem 0 0 0;">{insight['text']}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Not enough data to generate insights yet.")

    # Sleep tips
    st.markdown("### Sleep Optimization Tips")

    tips = [
        ("🌙", "Consistent Schedule", "Go to bed and wake up at the same time every day, even on weekends."),
        ("📵", "Screen-Free Hour", "Avoid screens 1 hour before bed. Blue light suppresses melatonin."),
        ("🌡️", "Cool Room", "Keep bedroom temperature between 60-67°F (15-19°C) for optimal sleep."),
        ("☕", "Caffeine Cutoff", "Stop caffeine intake 8-10 hours before your target bedtime."),
        ("🍷", "Limit Alcohol", "Alcohol fragments sleep and reduces REM. Avoid 3+ hours before bed."),
        ("🏃", "Exercise Timing", "Regular exercise improves sleep, but finish workouts 3+ hours before bed."),
    ]

    cols = st.columns(3)
    for i, (icon, title, tip) in enumerate(tips):
        with cols[i % 3]:
            st.markdown(f"""
            <div style="background: #1E293B; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; min-height: 120px;">
                <h4 style="margin: 0;">{icon} {title}</h4>
                <p style="font-size: 0.85rem; color: #9CA3AF; margin: 0.5rem 0 0 0;">{tip}</p>
            </div>
            """, unsafe_allow_html=True)

# Sidebar stats
with st.sidebar:
    st.markdown("### Quick Stats")

    st.markdown(f"""
    **This Period:**
    - Total Nights: {len(filtered_df)}
    - Total Sleep: {stats.get('total_hours', 0):.0f}h
    - Best Night: {stats.get('best_night', 0):.1f}h
    - Worst Night: {stats.get('worst_night', 0):.1f}h
    """)

    st.divider()

    st.markdown("### Targets")
    st.markdown("""
    - Duration: 7-9 hours
    - Deep Sleep: 13-23%
    - REM Sleep: 20-25%
    - Consistency: < 30 min variance
    """)
