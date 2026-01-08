"""
Weekly Health Reports Page

Generates comprehensive weekly reports with insights, alerts, and trends.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.synthetic.patient_generator import generate_synthetic_patient
from src.insights.weekly_report import generate_weekly_report, get_week_range
from src.insights.anomaly_detection import detect_anomalies, AlertSeverity

st.set_page_config(page_title="Weekly Reports", page_icon="📋", layout="wide")

st.title("📋 Weekly Health Reports")
st.markdown("*Comprehensive weekly summaries with insights and alerts*")

# Initialize session state
if 'health_data' not in st.session_state:
    patient = generate_synthetic_patient(days=90)
    st.session_state.health_data = patient['daily_summaries']


def get_severity_color(severity: AlertSeverity) -> str:
    """Get color for alert severity."""
    colors = {
        AlertSeverity.ALERT: "#EF4444",
        AlertSeverity.WARNING: "#F59E0B",
        AlertSeverity.INFO: "#3B82F6",
    }
    return colors.get(severity, "#6B7280")


def get_severity_icon(severity: AlertSeverity) -> str:
    """Get icon for alert severity."""
    icons = {
        AlertSeverity.ALERT: "🚨",
        AlertSeverity.WARNING: "⚠️",
        AlertSeverity.INFO: "ℹ️",
    }
    return icons.get(severity, "•")


def render_grade_card(grade_info: dict):
    """Render the weekly grade card."""
    grade = grade_info.get('grade', 'N/A')
    score = grade_info.get('score', 0)
    breakdown = grade_info.get('breakdown', {})

    # Grade colors
    grade_colors = {
        'A': '#10B981',
        'B+': '#34D399',
        'B': '#60A5FA',
        'C': '#FBBF24',
        'D': '#EF4444',
        'N/A': '#6B7280',
    }
    color = grade_colors.get(grade, '#6B7280')

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {color}22 0%, {color}11 100%);
                border: 2px solid {color}; border-radius: 16px; padding: 2rem; text-align: center;">
        <h1 style="font-size: 4rem; margin: 0; color: {color};">{grade}</h1>
        <p style="font-size: 1.2rem; color: #9CA3AF; margin: 0.5rem 0;">Weekly Score: {score:.0f}/100</p>
    </div>
    """, unsafe_allow_html=True)

    # Breakdown
    if breakdown:
        st.markdown("#### Score Breakdown")
        cols = st.columns(len(breakdown))
        icons = {'sleep': '😴', 'activity': '🏃', 'recovery': '💪', 'heart': '❤️'}

        for i, (category, cat_score) in enumerate(breakdown.items()):
            with cols[i]:
                icon = icons.get(category, '📊')
                st.metric(
                    label=f"{icon} {category.title()}",
                    value=f"{cat_score}/100"
                )


def render_stats_section(stats: dict):
    """Render weekly statistics."""
    st.markdown("### 📊 Week at a Glance")

    col1, col2, col3, col4 = st.columns(4)

    # Sleep stats
    with col1:
        if 'sleep' in stats:
            sleep = stats['sleep']
            st.markdown("""
            <div style="background: #1E1B4B; padding: 1rem; border-radius: 12px; border-left: 4px solid #8B5CF6;">
                <h4 style="color: #8B5CF6; margin: 0;">😴 Sleep</h4>
            </div>
            """, unsafe_allow_html=True)
            st.metric("Avg Duration", f"{sleep.get('avg_duration', 0):.1f}h")
            st.metric("Best Night", f"{sleep.get('best_night', 0):.1f}h")
            if 'avg_score' in sleep:
                st.metric("Avg Score", f"{sleep['avg_score']:.0f}")

    # Activity stats
    with col2:
        if 'activity' in stats:
            activity = stats['activity']
            st.markdown("""
            <div style="background: #1E3A2F; padding: 1rem; border-radius: 12px; border-left: 4px solid #10B981;">
                <h4 style="color: #10B981; margin: 0;">🏃 Activity</h4>
            </div>
            """, unsafe_allow_html=True)
            st.metric("Total Steps", f"{activity.get('total_steps', 0):,}")
            st.metric("Avg Daily", f"{activity.get('avg_daily_steps', 0):,}")
            st.metric("Days 10k+", f"{activity.get('days_over_10k', 0)}")

    # Heart stats
    with col3:
        if 'hrv' in stats or 'resting_hr' in stats:
            st.markdown("""
            <div style="background: #3B1C32; padding: 1rem; border-radius: 12px; border-left: 4px solid #EC4899;">
                <h4 style="color: #EC4899; margin: 0;">❤️ Heart</h4>
            </div>
            """, unsafe_allow_html=True)
            if 'hrv' in stats:
                st.metric("Avg HRV", f"{stats['hrv'].get('avg', 0):.0f}ms")
                trend = stats['hrv'].get('trend', 'stable')
                trend_icon = '📈' if trend == 'improving' else '📉' if trend == 'declining' else '➡️'
                st.metric("Trend", f"{trend_icon} {trend.title()}")
            if 'resting_hr' in stats:
                st.metric("Avg RHR", f"{stats['resting_hr'].get('avg', 0):.0f} bpm")

    # Readiness stats
    with col4:
        if 'readiness' in stats:
            readiness = stats['readiness']
            st.markdown("""
            <div style="background: #1E3A5F; padding: 1rem; border-radius: 12px; border-left: 4px solid #3B82F6;">
                <h4 style="color: #3B82F6; margin: 0;">💪 Readiness</h4>
            </div>
            """, unsafe_allow_html=True)
            st.metric("Avg Score", f"{readiness.get('avg_score', 0):.0f}")
            st.metric("High Days", f"{readiness.get('high_days', 0)}")
            st.metric("Low Days", f"{readiness.get('low_days', 0)}")


def render_comparisons(comparisons: dict):
    """Render week-over-week comparisons."""
    if not comparisons:
        return

    st.markdown("### 📈 Week-over-Week Changes")

    cols = st.columns(len(comparisons))

    icons = {
        'sleep_duration': '😴',
        'hrv': '❤️',
        'steps': '🏃',
        'readiness': '💪',
    }

    for i, (metric, data) in enumerate(comparisons.items()):
        with cols[i]:
            change = data['change']
            direction = data['direction']

            # Determine if change is good or bad
            good_up = metric in ['hrv', 'steps', 'readiness', 'sleep_duration']
            is_good = (direction == 'up' and good_up) or (direction == 'down' and not good_up)

            color = '#10B981' if is_good else '#EF4444' if direction != 'same' else '#6B7280'
            arrow = '↑' if direction == 'up' else '↓' if direction == 'down' else '→'

            icon = icons.get(metric, '📊')
            label = metric.replace('_', ' ').title()

            st.markdown(f"""
            <div style="background: {color}11; border: 1px solid {color}33;
                        border-radius: 12px; padding: 1rem; text-align: center;">
                <p style="font-size: 1.5rem; margin: 0;">{icon}</p>
                <p style="color: #9CA3AF; margin: 0.5rem 0 0 0; font-size: 0.9rem;">{label}</p>
                <p style="color: {color}; font-size: 1.5rem; font-weight: bold; margin: 0.5rem 0;">
                    {arrow} {abs(change):.1f}%
                </p>
                <p style="color: #6B7280; font-size: 0.8rem; margin: 0;">
                    {data['previous']:.1f} → {data['current']:.1f}
                </p>
            </div>
            """, unsafe_allow_html=True)


def render_insights(insights: list):
    """Render weekly insights."""
    if not insights:
        return

    st.markdown("### 💡 Insights")

    for insight in insights:
        # Determine insight type by keywords
        if any(word in insight.lower() for word in ['great', 'excellent', 'outstanding', 'improved']):
            icon = "✅"
            bg_color = "#10B98111"
            border_color = "#10B981"
        elif any(word in insight.lower() for word in ['below', 'low', 'dropped', 'declined', 'down']):
            icon = "⚠️"
            bg_color = "#F59E0B11"
            border_color = "#F59E0B"
        else:
            icon = "💡"
            bg_color = "#3B82F611"
            border_color = "#3B82F6"

        st.markdown(f"""
        <div style="background: {bg_color}; border-left: 4px solid {border_color};
                    padding: 1rem; border-radius: 0 8px 8px 0; margin-bottom: 0.5rem;">
            <p style="margin: 0;">{icon} {insight}</p>
        </div>
        """, unsafe_allow_html=True)


def render_alerts(alerts: list):
    """Render health alerts."""
    if not alerts:
        st.info("No health alerts this week. Keep up the good work!")
        return

    for alert in alerts[:5]:  # Show top 5 alerts
        color = get_severity_color(alert.severity)
        icon = get_severity_icon(alert.severity)

        with st.expander(f"{icon} {alert.title}", expanded=alert.severity == AlertSeverity.ALERT):
            st.markdown(f"""
            <div style="border-left: 4px solid {color}; padding-left: 1rem;">
                <p><strong>Date:</strong> {alert.date}</p>
                <p>{alert.description}</p>
                <p><strong>Value:</strong> {alert.value:.1f} (Expected: {alert.expected_range[0]:.0f}-{alert.expected_range[1]:.0f})</p>
            </div>
            """, unsafe_allow_html=True)

            if alert.recommendation:
                st.info(f"💡 **Recommendation:** {alert.recommendation}")


# Main content
tab1, tab2, tab3 = st.tabs(["📊 Weekly Report", "🚨 Health Alerts", "📅 Report History"])

with tab1:
    # Week selector
    col1, col2 = st.columns([3, 1])
    with col1:
        week_offset = st.selectbox(
            "Select Week",
            options=[0, -1, -2, -3, -4],
            format_func=lambda x: "This Week" if x == 0 else f"{abs(x)} Week{'s' if abs(x) > 1 else ''} Ago",
            index=0
        )

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Refresh Report"):
            st.rerun()

    # Generate report
    report = generate_weekly_report(st.session_state.health_data, week_offset)

    if 'error' in report:
        st.warning(f"⚠️ {report['error']}")
        st.info("Try selecting a different week or ensure you have enough data.")
    else:
        # Header with date range
        start = report['date_range']['start']
        end = report['date_range']['end']
        st.markdown(f"**Week of {start} to {end}**")

        st.divider()

        # Grade card
        col1, col2 = st.columns([1, 2])
        with col1:
            render_grade_card(report.get('grade', {}))

        with col2:
            render_comparisons(report.get('comparisons', {}))

        st.divider()

        # Stats section
        render_stats_section(report.get('stats', {}))

        st.divider()

        # Insights
        render_insights(report.get('insights', []))

with tab2:
    st.markdown("### 🚨 Health Alerts & Anomalies")
    st.markdown("*Unusual patterns detected in your recent health data*")

    # Detect anomalies
    alerts = detect_anomalies(st.session_state.health_data, days_to_analyze=14)

    # Filter controls
    col1, col2 = st.columns(2)
    with col1:
        severity_filter = st.multiselect(
            "Filter by Severity",
            options=['Alert', 'Warning', 'Info'],
            default=['Alert', 'Warning']
        )

    with col2:
        metric_options = list(set(a.metric for a in alerts))
        metric_filter = st.multiselect(
            "Filter by Metric",
            options=metric_options,
            default=metric_options
        )

    # Filter alerts
    severity_map = {'Alert': AlertSeverity.ALERT, 'Warning': AlertSeverity.WARNING, 'Info': AlertSeverity.INFO}
    filtered_alerts = [
        a for a in alerts
        if a.severity in [severity_map[s] for s in severity_filter]
        and a.metric in metric_filter
    ]

    # Summary
    alert_count = sum(1 for a in alerts if a.severity == AlertSeverity.ALERT)
    warning_count = sum(1 for a in alerts if a.severity == AlertSeverity.WARNING)
    info_count = sum(1 for a in alerts if a.severity == AlertSeverity.INFO)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🚨 Alerts", alert_count)
    with col2:
        st.metric("⚠️ Warnings", warning_count)
    with col3:
        st.metric("ℹ️ Info", info_count)

    st.divider()

    # Render alerts
    render_alerts(filtered_alerts)

with tab3:
    st.markdown("### 📅 Report History")
    st.markdown("*View past weekly reports*")

    # Generate reports for last 4 weeks
    reports_data = []
    for offset in range(0, -5, -1):
        report = generate_weekly_report(st.session_state.health_data, offset)
        if 'error' not in report:
            grade_info = report.get('grade', {})
            stats = report.get('stats', {})

            reports_data.append({
                'Week': report['date_range']['start'],
                'Grade': grade_info.get('grade', 'N/A'),
                'Score': grade_info.get('score', 0),
                'Avg Sleep': stats.get('sleep', {}).get('avg_duration', 0),
                'Avg Steps': stats.get('activity', {}).get('avg_daily_steps', 0),
                'Avg HRV': stats.get('hrv', {}).get('avg', 0),
                'Avg Readiness': stats.get('readiness', {}).get('avg_score', 0),
            })

    if reports_data:
        df = pd.DataFrame(reports_data)

        # Style the dataframe
        def color_grade(val):
            colors = {'A': '#10B981', 'B+': '#34D399', 'B': '#60A5FA', 'C': '#FBBF24', 'D': '#EF4444'}
            return f'color: {colors.get(val, "#6B7280")}'

        try:
            styled_df = df.style.map(color_grade, subset=['Grade'])
        except AttributeError:
            styled_df = df.style.applymap(color_grade, subset=['Grade'])

        st.dataframe(styled_df, use_container_width=True, hide_index=True)

        # Trend chart
        st.markdown("#### Score Trend")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['Week'],
            y=df['Score'],
            mode='lines+markers',
            name='Weekly Score',
            line=dict(color='#8B5CF6', width=3),
            marker=dict(size=10)
        ))
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis_range=[0, 100],
            height=300,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough data to show report history.")

# Sidebar info
with st.sidebar:
    st.markdown("### About Reports")
    st.markdown("""
    Weekly reports provide:
    - **Grade**: Overall health score (A-D)
    - **Stats**: Detailed metrics breakdown
    - **Comparisons**: Week-over-week changes
    - **Insights**: AI-generated observations
    - **Alerts**: Anomalies that need attention
    """)

    st.divider()

    st.markdown("### Grade Components")
    st.markdown("""
    - 😴 **Sleep** (25%): Duration & quality
    - 🏃 **Activity** (25%): Steps & exercise
    - 💪 **Recovery** (25%): Readiness score
    - ❤️ **Heart** (25%): HRV baseline
    """)
