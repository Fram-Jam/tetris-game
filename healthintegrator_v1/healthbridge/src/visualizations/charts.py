"""
Reusable Plotly Chart Components
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import List, Dict, Optional


# Color palette
COLORS = {
    'primary': '#6366F1',
    'secondary': '#8B5CF6',
    'success': '#10B981',
    'warning': '#F59E0B',
    'danger': '#EF4444',
    'info': '#3B82F6',
    'light': '#E2E8F0',
    'dark': '#1E293B',

    # Sleep stages
    'deep_sleep': '#1E3A8A',
    'rem_sleep': '#3B82F6',
    'light_sleep': '#93C5FD',

    # Chart specific
    'hrv': '#10B981',
    'rhr': '#EF4444',
    'steps': '#8B5CF6',
    'glucose': '#6366F1',
}


def apply_default_layout(fig: go.Figure, height: int = 350) -> go.Figure:
    """Apply consistent styling to a figure."""
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        font=dict(family="Inter, sans-serif")
    )
    fig.update_xaxes(gridcolor=COLORS['light'], showgrid=True)
    fig.update_yaxes(gridcolor=COLORS['light'], showgrid=True)
    return fig


def create_line_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    name: str = None,
    color: str = None,
    show_markers: bool = True,
    show_ma: bool = False,
    ma_window: int = 7
) -> go.Figure:
    """Create a simple line chart."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df[x],
        y=df[y],
        mode='lines+markers' if show_markers else 'lines',
        name=name or y,
        line=dict(color=color or COLORS['primary'], width=2),
        marker=dict(size=6) if show_markers else None
    ))

    if show_ma and len(df) >= ma_window:
        df_ma = df.copy()
        df_ma[f'{y}_ma'] = df_ma[y].rolling(window=ma_window, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=df_ma[x],
            y=df_ma[f'{y}_ma'],
            mode='lines',
            name=f'{ma_window}-day avg',
            line=dict(color=COLORS['light'], width=3, dash='dash')
        ))

    return apply_default_layout(fig)


def create_bar_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    name: str = None,
    color: str = None,
    color_by_value: bool = False,
    thresholds: Dict = None
) -> go.Figure:
    """Create a bar chart."""
    fig = go.Figure()

    if color_by_value and thresholds:
        colors = [
            COLORS['danger'] if v < thresholds.get('low', 0) else
            COLORS['warning'] if v < thresholds.get('medium', 0) else
            COLORS['success']
            for v in df[y]
        ]
    else:
        colors = color or COLORS['primary']

    fig.add_trace(go.Bar(
        x=df[x],
        y=df[y],
        name=name or y,
        marker_color=colors
    ))

    return apply_default_layout(fig)


def create_dual_axis_chart(
    df: pd.DataFrame,
    x: str,
    y1: str,
    y2: str,
    name1: str = None,
    name2: str = None,
    color1: str = None,
    color2: str = None
) -> go.Figure:
    """Create a chart with two y-axes."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=df[x],
            y=df[y1],
            mode='lines+markers',
            name=name1 or y1,
            line=dict(color=color1 or COLORS['hrv'], width=2),
            marker=dict(size=5)
        ),
        secondary_y=False
    )

    fig.add_trace(
        go.Scatter(
            x=df[x],
            y=df[y2],
            mode='lines+markers',
            name=name2 or y2,
            line=dict(color=color2 or COLORS['rhr'], width=2),
            marker=dict(size=5)
        ),
        secondary_y=True
    )

    fig.update_yaxes(title_text=name1 or y1, secondary_y=False)
    fig.update_yaxes(title_text=name2 or y2, secondary_y=True)

    return apply_default_layout(fig)


def create_stacked_bar_chart(
    df: pd.DataFrame,
    x: str,
    y_columns: List[str],
    names: List[str] = None,
    colors: List[str] = None
) -> go.Figure:
    """Create a stacked bar chart."""
    fig = go.Figure()

    default_colors = [COLORS['deep_sleep'], COLORS['rem_sleep'], COLORS['light_sleep']]

    for i, y_col in enumerate(y_columns):
        name = names[i] if names and i < len(names) else y_col
        color = colors[i] if colors and i < len(colors) else default_colors[i % len(default_colors)]

        fig.add_trace(go.Bar(
            x=df[x],
            y=df[y_col],
            name=name,
            marker_color=color
        ))

    fig.update_layout(barmode='stack')
    return apply_default_layout(fig, height=400)


def create_area_chart_with_range(
    df: pd.DataFrame,
    x: str,
    y_avg: str,
    y_min: str,
    y_max: str,
    name: str = None,
    color: str = None
) -> go.Figure:
    """Create an area chart with min/max range band."""
    fig = go.Figure()
    c = color or COLORS['primary']

    # Range band
    fig.add_trace(go.Scatter(
        x=pd.concat([df[x], df[x][::-1]]),
        y=pd.concat([df[y_max], df[y_min][::-1]]),
        fill='toself',
        fillcolor=f'rgba({int(c[1:3], 16)}, {int(c[3:5], 16)}, {int(c[5:7], 16)}, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Range',
        showlegend=True
    ))

    # Average line
    fig.add_trace(go.Scatter(
        x=df[x],
        y=df[y_avg],
        mode='lines+markers',
        name=name or 'Average',
        line=dict(color=c, width=2),
        marker=dict(size=6)
    ))

    return apply_default_layout(fig)


def create_gauge_chart(
    value: float,
    title: str,
    max_value: float = 100,
    thresholds: Dict = None
) -> go.Figure:
    """Create a gauge/dial chart for scores."""
    if thresholds is None:
        thresholds = {'low': 60, 'medium': 75}

    # Determine color based on value
    if value < thresholds['low']:
        color = COLORS['danger']
    elif value < thresholds['medium']:
        color = COLORS['warning']
    else:
        color = COLORS['success']

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title},
        gauge={
            'axis': {'range': [0, max_value]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, thresholds['low']], 'color': "rgba(239, 68, 68, 0.3)"},
                {'range': [thresholds['low'], thresholds['medium']], 'color': "rgba(245, 158, 11, 0.3)"},
                {'range': [thresholds['medium'], max_value], 'color': "rgba(16, 185, 129, 0.3)"}
            ],
        }
    ))

    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def add_target_line(fig: go.Figure, value: float, text: str = "Target") -> go.Figure:
    """Add a horizontal target line to a figure."""
    fig.add_hline(
        y=value,
        line_dash="dash",
        line_color=COLORS['dark'],
        annotation_text=text,
        annotation_position="right"
    )
    return fig


def add_range_band(
    fig: go.Figure,
    y_min: float,
    y_max: float,
    text: str = "Normal Range"
) -> go.Figure:
    """Add a horizontal range band to a figure."""
    fig.add_hrect(
        y0=y_min,
        y1=y_max,
        fillcolor=f"rgba({int(COLORS['success'][1:3], 16)}, {int(COLORS['success'][3:5], 16)}, {int(COLORS['success'][5:7], 16)}, 0.1)",
        line_width=0,
        annotation_text=text,
        annotation_position="top left"
    )
    return fig


def create_calendar_heatmap(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    title: str = "",
    colorscale: str = "Viridis",
    weeks_to_show: int = 12
) -> go.Figure:
    """
    Create a GitHub-style calendar heatmap.

    Args:
        df: DataFrame with date and value columns
        date_col: Name of date column
        value_col: Name of value column
        title: Chart title
        colorscale: Plotly colorscale name
        weeks_to_show: Number of weeks to display
    """
    import numpy as np
    from datetime import timedelta

    # Prepare data
    df_cal = df.copy()
    df_cal[date_col] = pd.to_datetime(df_cal[date_col])
    df_cal = df_cal.sort_values(date_col)

    # Get date range
    end_date = df_cal[date_col].max()
    start_date = end_date - timedelta(weeks=weeks_to_show)
    df_cal = df_cal[df_cal[date_col] >= start_date]

    # Create week and day-of-week columns
    df_cal['week'] = df_cal[date_col].dt.isocalendar().week
    df_cal['year'] = df_cal[date_col].dt.year
    df_cal['weekday'] = df_cal[date_col].dt.weekday
    df_cal['day_name'] = df_cal[date_col].dt.strftime('%a')

    # Create combined week identifier for proper ordering
    df_cal['week_id'] = df_cal['year'] * 100 + df_cal['week']

    # Get unique weeks and create mapping
    unique_weeks = sorted(df_cal['week_id'].unique())
    week_mapping = {w: i for i, w in enumerate(unique_weeks)}
    df_cal['week_num'] = df_cal['week_id'].map(week_mapping)

    # Create heatmap matrix
    n_weeks = len(unique_weeks)
    matrix = np.full((7, n_weeks), np.nan)
    text_matrix = [['' for _ in range(n_weeks)] for _ in range(7)]

    for _, row in df_cal.iterrows():
        week_idx = week_mapping.get(row['week_id'])
        if week_idx is not None:
            day_idx = row['weekday']
            matrix[day_idx, week_idx] = row[value_col]
            date_str = row[date_col].strftime('%b %d')
            text_matrix[day_idx][week_idx] = f"{date_str}: {row[value_col]:.1f}"

    # Create figure
    day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    # Generate week labels (show month at start of each month)
    week_labels = []
    prev_month = None
    for week_id in unique_weeks:
        # Find a date in this week
        week_dates = df_cal[df_cal['week_id'] == week_id][date_col]
        if len(week_dates) > 0:
            sample_date = week_dates.iloc[0]
            month = sample_date.strftime('%b')
            if month != prev_month:
                week_labels.append(month)
                prev_month = month
            else:
                week_labels.append('')
        else:
            week_labels.append('')

    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=list(range(n_weeks)),
        y=day_labels,
        text=text_matrix,
        hovertemplate='%{text}<extra></extra>',
        colorscale=colorscale,
        showscale=True,
        xgap=3,
        ygap=3,
        colorbar=dict(
            title=value_col.replace('_', ' ').title(),
            thickness=15,
            len=0.7,
        )
    ))

    fig.update_layout(
        title=title,
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(n_weeks)),
            ticktext=week_labels,
            side='top',
            showgrid=False,
        ),
        yaxis=dict(
            showgrid=False,
            autorange='reversed',
        ),
        height=250,
        margin=dict(l=50, r=20, t=60, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )

    return fig


def create_sleep_stages_chart(
    df: pd.DataFrame,
    date_col: str = 'date'
) -> go.Figure:
    """Create a stacked area chart for sleep stages."""

    fig = go.Figure()

    # Add each sleep stage
    stages = [
        ('deep_sleep', 'Deep Sleep', COLORS['deep_sleep']),
        ('rem_sleep', 'REM Sleep', COLORS['rem_sleep']),
        ('light_sleep', 'Light Sleep', COLORS['light_sleep']),
    ]

    for col, name, color in stages:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df[date_col],
                y=df[col],
                mode='lines',
                name=name,
                stackgroup='one',
                fillcolor=color,
                line=dict(width=0.5, color=color),
            ))

    fig.update_layout(
        title='Sleep Stages Over Time',
        yaxis_title='Hours',
        hovermode='x unified',
    )

    return apply_default_layout(fig, height=350)


def create_correlation_matrix(
    df: pd.DataFrame,
    columns: List[str],
    labels: List[str] = None
) -> go.Figure:
    """Create a correlation matrix heatmap."""
    import numpy as np

    # Calculate correlations
    corr_df = df[columns].corr()

    # Use custom labels if provided
    display_labels = labels if labels else [c.replace('_', ' ').title() for c in columns]

    # Create annotations
    annotations = []
    for i, row in enumerate(corr_df.values):
        for j, val in enumerate(row):
            annotations.append(dict(
                x=j,
                y=i,
                text=f'{val:.2f}',
                showarrow=False,
                font=dict(color='white' if abs(val) > 0.5 else 'black')
            ))

    fig = go.Figure(data=go.Heatmap(
        z=corr_df.values,
        x=display_labels,
        y=display_labels,
        colorscale='RdBu',
        zmid=0,
        zmin=-1,
        zmax=1,
        showscale=True,
        colorbar=dict(title='Correlation', thickness=15),
    ))

    fig.update_layout(
        title='Metric Correlations',
        annotations=annotations,
        height=400,
        margin=dict(l=100, r=20, t=50, b=100),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(tickangle=-45),
    )

    return fig


def create_distribution_chart(
    values: List[float],
    title: str,
    unit: str = "",
    color: str = None,
    show_stats: bool = True
) -> go.Figure:
    """Create a distribution histogram with KDE overlay."""
    import numpy as np

    fig = go.Figure()

    c = color or COLORS['primary']

    # Histogram
    fig.add_trace(go.Histogram(
        x=values,
        nbinsx=20,
        name='Distribution',
        marker_color=c,
        opacity=0.7,
    ))

    # Add statistics as annotations
    if show_stats and values:
        mean_val = np.mean(values)
        std_val = np.std(values)
        median_val = np.median(values)

        fig.add_vline(
            x=mean_val,
            line_dash='dash',
            line_color='white',
            annotation_text=f'Mean: {mean_val:.1f}{unit}',
            annotation_position='top'
        )

    fig.update_layout(
        title=title,
        xaxis_title=f'Value ({unit})' if unit else 'Value',
        yaxis_title='Count',
        showlegend=False,
    )

    return apply_default_layout(fig, height=300)


def create_radar_chart(
    categories: List[str],
    values: List[float],
    max_values: List[float] = None,
    title: str = "Health Profile"
) -> go.Figure:
    """Create a radar/spider chart for multi-dimensional health profile."""

    # Normalize values to 0-100 scale if max_values provided
    if max_values:
        normalized = [(v / m * 100) if m > 0 else 0 for v, m in zip(values, max_values)]
    else:
        normalized = values

    # Close the polygon
    categories_closed = categories + [categories[0]]
    values_closed = normalized + [normalized[0]]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill='toself',
        fillcolor=f'rgba(99, 102, 241, 0.3)',
        line=dict(color=COLORS['primary'], width=2),
        name='Current'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                showticklabels=True,
            ),
            angularaxis=dict(
                tickfont=dict(size=12),
            ),
        ),
        showlegend=False,
        title=title,
        height=400,
        margin=dict(l=60, r=60, t=60, b=60),
        paper_bgcolor='rgba(0,0,0,0)',
    )

    return fig
