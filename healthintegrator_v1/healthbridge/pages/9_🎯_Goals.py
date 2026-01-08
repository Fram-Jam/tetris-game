"""
Goals Page

Set, track, and achieve your health goals.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import date, datetime, timedelta
import sys
import os
import uuid

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.synthetic.patient_generator import generate_synthetic_patient
from src.insights.goals import (
    Goal, GoalCategory, GoalFrequency, GoalStatus,
    GOAL_TEMPLATES, create_goal_from_template,
    evaluate_goals, get_goal_insights, suggest_goals
)

st.set_page_config(page_title="Goals", page_icon="🎯", layout="wide")

st.title("🎯 Health Goals")
st.markdown("*Set targets, build streaks, achieve results*")

# Initialize session state
if 'health_data' not in st.session_state:
    patient = generate_synthetic_patient(days=90)
    st.session_state.health_data = patient['daily_summaries']

if 'goals' not in st.session_state:
    # Create some default goals
    st.session_state.goals = [
        create_goal_from_template('sleep_7h', 'goal_sleep_7h'),
        create_goal_from_template('steps_10k', 'goal_steps_10k'),
        create_goal_from_template('readiness_80', 'goal_readiness_80'),
    ]
    # Simulate some history
    for goal in st.session_state.goals:
        goal.streak = 3
        goal.best_streak = 7
        goal.times_achieved = 12
        goal.total_attempts = 15


def get_category_icon(category: GoalCategory) -> str:
    """Get icon for goal category."""
    icons = {
        GoalCategory.SLEEP: "😴",
        GoalCategory.ACTIVITY: "🏃",
        GoalCategory.RECOVERY: "💪",
        GoalCategory.HEART: "❤️",
        GoalCategory.WEIGHT: "⚖️",
        GoalCategory.CUSTOM: "🎯",
    }
    return icons.get(category, "🎯")


def get_category_color(category: GoalCategory) -> str:
    """Get color for goal category."""
    colors = {
        GoalCategory.SLEEP: "#8B5CF6",
        GoalCategory.ACTIVITY: "#10B981",
        GoalCategory.RECOVERY: "#3B82F6",
        GoalCategory.HEART: "#EF4444",
        GoalCategory.WEIGHT: "#F59E0B",
        GoalCategory.CUSTOM: "#6366F1",
    }
    return colors.get(category, "#6366F1")


def render_goal_card(goal: Goal, result: dict):
    """Render a goal progress card."""
    icon = get_category_icon(goal.category)
    color = get_category_color(goal.category)
    achieved = result.get('achieved', False)
    progress = result.get('progress_pct', 0)
    value = result.get('value')

    # Status indicator
    if achieved:
        status_icon = "✅"
        status_text = "Achieved!"
        border_color = "#10B981"
    elif progress >= 75:
        status_icon = "🔥"
        status_text = "Almost there!"
        border_color = "#F59E0B"
    else:
        status_icon = "🎯"
        status_text = "In progress"
        border_color = color

    # Format value
    if value is not None:
        if goal.metric == 'steps':
            value_str = f"{int(value):,}"
        elif goal.metric in ['sleep_duration', 'deep_sleep', 'rem_sleep']:
            value_str = f"{value:.1f}h"
        elif goal.metric in ['hrv']:
            value_str = f"{value:.0f}ms"
        elif goal.metric in ['resting_hr']:
            value_str = f"{value:.0f}bpm"
        else:
            value_str = f"{value:.0f}"
    else:
        value_str = "No data"

    # Target string
    if goal.comparison == 'gte':
        target_str = f"≥ {goal.target_value:,.0f}" if goal.target_value >= 100 else f"≥ {goal.target_value}"
    elif goal.comparison == 'lte':
        target_str = f"≤ {goal.target_value:,.0f}" if goal.target_value >= 100 else f"≤ {goal.target_value}"
    else:
        target_str = f"{goal.target_value}"

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {color}11 0%, {color}05 100%);
                border: 1px solid {border_color}44; border-left: 4px solid {border_color};
                border-radius: 12px; padding: 1rem; margin-bottom: 1rem;">
        <div style="display: flex; justify-content: space-between; align-items: start;">
            <div>
                <h4 style="margin: 0; color: {color};">{icon} {goal.name}</h4>
                <p style="color: #9CA3AF; margin: 0.25rem 0; font-size: 0.85rem;">
                    Target: {target_str} • {goal.frequency.value.title()}
                </p>
            </div>
            <div style="text-align: right;">
                <span style="font-size: 1.5rem;">{status_icon}</span>
                <p style="margin: 0; font-size: 0.8rem; color: #9CA3AF;">{status_text}</p>
            </div>
        </div>
        <div style="margin-top: 1rem;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                <span style="font-size: 1.2rem; font-weight: bold;">{value_str}</span>
                <span style="color: #9CA3AF;">{progress:.0f}%</span>
            </div>
            <div style="background: #1E293B; border-radius: 4px; height: 8px; overflow: hidden;">
                <div style="background: {border_color}; height: 100%; width: {min(100, progress)}%;
                            transition: width 0.3s ease;"></div>
            </div>
        </div>
        <div style="display: flex; gap: 1rem; margin-top: 0.75rem; font-size: 0.8rem; color: #6B7280;">
            <span>🔥 {goal.streak} day streak</span>
            <span>🏆 Best: {goal.best_streak} days</span>
            <span>📊 {goal.achievement_rate:.0f}% success</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_streak_calendar(goal: Goal, health_data: list):
    """Render a streak calendar for a goal."""
    # Get last 30 days of data
    dates = []
    achieved_list = []

    for i in range(30):
        d = date.today() - timedelta(days=29-i)
        dates.append(d)

        # Find data for this date
        day_data = next((h for h in health_data if h['date'] == d), None)
        if day_data and goal.metric in day_data:
            achieved = goal.check_achievement(day_data[goal.metric])
            achieved_list.append(1 if achieved else 0)
        else:
            achieved_list.append(-1)  # No data

    # Create calendar grid (5 rows x 6 columns for 30 days)
    fig = go.Figure()

    # Create 5x6 grid
    for i, (d, achieved) in enumerate(zip(dates, achieved_list)):
        row = i // 6
        col = i % 6

        if achieved == 1:
            color = '#10B981'
        elif achieved == 0:
            color = '#EF4444'
        else:
            color = '#374151'

        fig.add_trace(go.Scatter(
            x=[col],
            y=[4-row],
            mode='markers',
            marker=dict(
                size=30,
                color=color,
                symbol='square',
            ),
            hovertemplate=f"{d.strftime('%b %d')}: {'✓' if achieved == 1 else '✗' if achieved == 0 else 'No data'}<extra></extra>",
            showlegend=False,
        ))

    fig.update_layout(
        height=200,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, showticklabels=False, range=[-0.5, 5.5]),
        yaxis=dict(showgrid=False, showticklabels=False, range=[-0.5, 4.5]),
    )

    return fig


# Evaluate current goals
results = evaluate_goals(st.session_state.goals, st.session_state.health_data)

# Main content tabs
tab1, tab2, tab3 = st.tabs(["📊 Today's Progress", "➕ Add Goals", "📈 History"])

with tab1:
    # Summary metrics
    achieved_count = sum(1 for r in results if r['achieved'])
    total_goals = len(results)
    total_streak = sum(g.streak for g in st.session_state.goals)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Goals Achieved", f"{achieved_count}/{total_goals}")

    with col2:
        pct = (achieved_count / total_goals * 100) if total_goals > 0 else 0
        st.metric("Achievement Rate", f"{pct:.0f}%")

    with col3:
        st.metric("Active Streaks", f"{total_streak} days")

    with col4:
        best_overall = max((g.best_streak for g in st.session_state.goals), default=0)
        st.metric("Best Streak", f"{best_overall} days")

    st.divider()

    # Insights
    insights = get_goal_insights(st.session_state.goals, results)
    if insights:
        for insight in insights[:3]:
            st.info(insight)

    st.divider()

    # Goal cards
    st.markdown("### Your Goals")

    for goal, result in zip(st.session_state.goals, results):
        col1, col2 = st.columns([3, 1])

        with col1:
            render_goal_card(goal, result)

        with col2:
            with st.expander("📅 30-day view"):
                fig = render_streak_calendar(goal, st.session_state.health_data)
                st.plotly_chart(fig, use_container_width=True)
                st.caption("🟢 Achieved  🔴 Missed  ⬛ No data")

with tab2:
    st.markdown("### Add New Goal")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Quick Add from Templates")

        # Get suggestions
        suggestions = suggest_goals(st.session_state.health_data)

        if suggestions:
            st.markdown("**Recommended for you:**")
            for template_id in suggestions[:3]:
                if template_id in GOAL_TEMPLATES:
                    template = GOAL_TEMPLATES[template_id]
                    icon = get_category_icon(template['category'])

                    # Check if already have this goal
                    existing = any(g.metric == template['metric'] and g.target_value == template['target_value']
                                   for g in st.session_state.goals)

                    if not existing:
                        col_a, col_b = st.columns([3, 1])
                        with col_a:
                            st.markdown(f"{icon} **{template['name']}**")
                        with col_b:
                            if st.button("Add", key=f"add_{template_id}"):
                                new_goal = create_goal_from_template(template_id, f"goal_{uuid.uuid4().hex[:8]}")
                                st.session_state.goals.append(new_goal)
                                st.success(f"Added goal: {template['name']}")
                                st.rerun()

        st.divider()

        st.markdown("**All Templates:**")

        # Group templates by category
        by_category = {}
        for tid, template in GOAL_TEMPLATES.items():
            cat = template['category']
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append((tid, template))

        for category, templates in by_category.items():
            icon = get_category_icon(category)
            with st.expander(f"{icon} {category.value.title()} Goals"):
                for tid, template in templates:
                    existing = any(g.metric == template['metric'] and g.target_value == template['target_value']
                                   for g in st.session_state.goals)

                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        st.markdown(f"**{template['name']}**")
                        st.caption(f"Target: {template['target_value']} • {template.get('frequency', GoalFrequency.DAILY).value}")
                    with col_b:
                        if existing:
                            st.markdown("✓ Active")
                        else:
                            if st.button("Add", key=f"template_{tid}"):
                                new_goal = create_goal_from_template(tid, f"goal_{uuid.uuid4().hex[:8]}")
                                st.session_state.goals.append(new_goal)
                                st.success(f"Added: {template['name']}")
                                st.rerun()

    with col2:
        st.markdown("#### Create Custom Goal")

        with st.form("custom_goal"):
            name = st.text_input("Goal Name", placeholder="e.g., Meditate 10 minutes")

            category = st.selectbox(
                "Category",
                options=list(GoalCategory),
                format_func=lambda x: f"{get_category_icon(x)} {x.value.title()}"
            )

            metric = st.selectbox(
                "Metric to Track",
                options=['sleep_duration', 'sleep_score', 'steps', 'active_minutes',
                         'hrv', 'resting_hr', 'readiness_score', 'calories_active'],
                format_func=lambda x: x.replace('_', ' ').title()
            )

            col_t1, col_t2 = st.columns(2)
            with col_t1:
                comparison = st.selectbox(
                    "Comparison",
                    options=['gte', 'lte', 'eq'],
                    format_func=lambda x: {'gte': '≥ At least', 'lte': '≤ At most', 'eq': '= Exactly'}[x]
                )
            with col_t2:
                target = st.number_input("Target Value", min_value=0.0, value=0.0)

            frequency = st.selectbox(
                "Frequency",
                options=list(GoalFrequency),
                format_func=lambda x: x.value.title()
            )

            submitted = st.form_submit_button("Create Goal")

            if submitted and name and target > 0:
                new_goal = Goal(
                    id=f"goal_{uuid.uuid4().hex[:8]}",
                    name=name,
                    category=category,
                    metric=metric,
                    target_value=target,
                    comparison=comparison,
                    frequency=frequency,
                )
                st.session_state.goals.append(new_goal)
                st.success(f"Created goal: {name}")
                st.rerun()

with tab3:
    st.markdown("### Goal History & Statistics")

    if st.session_state.goals:
        # Overall stats
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Achievement Rates by Goal")

            goal_stats = []
            for goal in st.session_state.goals:
                goal_stats.append({
                    'Goal': goal.name,
                    'Success Rate': f"{goal.achievement_rate:.0f}%",
                    'Current Streak': goal.streak,
                    'Best Streak': goal.best_streak,
                    'Total Attempts': goal.total_attempts,
                })

            df_stats = pd.DataFrame(goal_stats)
            st.dataframe(df_stats, use_container_width=True, hide_index=True)

        with col2:
            st.markdown("#### Success Rate Over Time")

            # Create bar chart of achievement rates
            fig = go.Figure()

            names = [g.name[:20] + '...' if len(g.name) > 20 else g.name for g in st.session_state.goals]
            rates = [g.achievement_rate for g in st.session_state.goals]
            colors = [get_category_color(g.category) for g in st.session_state.goals]

            fig.add_trace(go.Bar(
                x=names,
                y=rates,
                marker_color=colors,
            ))

            fig.add_hline(y=80, line_dash='dash', line_color='#10B981',
                         annotation_text='80% Target')

            fig.update_layout(
                height=300,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                yaxis_title='Success Rate (%)',
                yaxis_range=[0, 100],
                xaxis_tickangle=-45,
            )

            st.plotly_chart(fig, use_container_width=True)

        # Manage goals
        st.divider()
        st.markdown("#### Manage Goals")

        for i, goal in enumerate(st.session_state.goals):
            col1, col2, col3 = st.columns([3, 1, 1])

            with col1:
                icon = get_category_icon(goal.category)
                st.markdown(f"{icon} **{goal.name}**")

            with col2:
                status = "🟢 Active" if goal.status == GoalStatus.ACTIVE else "⏸️ Paused"
                st.markdown(status)

            with col3:
                if st.button("🗑️ Remove", key=f"remove_{goal.id}"):
                    st.session_state.goals.pop(i)
                    st.rerun()
    else:
        st.info("No goals set yet. Add some goals to start tracking!")

# Sidebar
with st.sidebar:
    st.markdown("### Goal Tips")
    st.markdown("""
    **Setting Effective Goals:**

    1. **Start small** - Build momentum with achievable targets
    2. **Be specific** - "7 hours sleep" vs "sleep more"
    3. **Track consistently** - Check progress daily
    4. **Celebrate streaks** - Momentum builds habits

    **Recommended Targets:**
    - Sleep: 7-8 hours
    - Steps: 8,000-10,000
    - HRV: Above your baseline
    - Readiness: 70+
    """)

    st.divider()

    st.markdown("### Quick Actions")
    if st.button("🔄 Refresh Progress"):
        st.rerun()

    if st.button("🗑️ Clear All Goals"):
        if st.session_state.goals:
            st.session_state.goals = []
            st.rerun()
