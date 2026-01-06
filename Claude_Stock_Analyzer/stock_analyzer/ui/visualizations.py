"""
Advanced Visualization Components
==================================
Rich visualizations for:
- Performance attribution
- Risk analytics
- Correlation heatmaps
- Trade analysis
- Monte Carlo simulations
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger("stock_analyzer.visualizations")

# Color palette
COLORS = {
    'primary': '#00D4AA',
    'secondary': '#00B894',
    'success': '#27ae60',
    'danger': '#e74c3c',
    'warning': '#f39c12',
    'info': '#3498db',
    'neutral': '#95a5a6',
    'background': 'rgba(0,0,0,0)',
    'grid': 'rgba(255,255,255,0.1)'
}


def create_performance_attribution_chart(
    returns: pd.Series,
    factor_returns: Dict[str, pd.Series],
    title: str = "Performance Attribution"
) -> go.Figure:
    """
    Create performance attribution waterfall chart.

    Shows contribution of each factor to total returns.
    """
    # Calculate factor contributions
    contributions = {}
    residual = returns.copy()

    for factor_name, factor_ret in factor_returns.items():
        aligned = factor_ret.reindex(returns.index).fillna(0)
        corr = returns.corr(aligned)
        contribution = corr * aligned.std() / returns.std() * returns.mean()
        contributions[factor_name] = contribution
        residual = residual - aligned * corr

    contributions['Residual (Alpha)'] = residual.mean()

    # Create waterfall chart
    names = list(contributions.keys())
    values = [contributions[n] * 252 * 100 for n in names]  # Annualize and convert to %

    # Calculate cumulative for waterfall
    cumulative = np.cumsum([0] + values[:-1])

    fig = go.Figure(go.Waterfall(
        name="Attribution",
        orientation="v",
        measure=["relative"] * (len(values) - 1) + ["total"],
        x=names,
        y=values,
        connector={"line": {"color": COLORS['neutral']}},
        decreasing={"marker": {"color": COLORS['danger']}},
        increasing={"marker": {"color": COLORS['success']}},
        totals={"marker": {"color": COLORS['primary']}}
    ))

    fig.update_layout(
        title=title,
        template='plotly_dark',
        paper_bgcolor=COLORS['background'],
        plot_bgcolor=COLORS['background'],
        yaxis_title="Contribution (%)",
        showlegend=False,
        height=400
    )

    return fig


def create_risk_dashboard(
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    title: str = "Risk Analytics Dashboard"
) -> go.Figure:
    """
    Create comprehensive risk analytics dashboard.

    Includes:
    - VaR analysis
    - Drawdown chart
    - Rolling volatility
    - Return distribution
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Value at Risk (VaR)',
            'Drawdown Analysis',
            'Rolling Volatility',
            'Return Distribution'
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    # 1. VaR Chart
    sorted_returns = returns.sort_values()
    var_95 = sorted_returns.quantile(0.05)
    var_99 = sorted_returns.quantile(0.01)

    fig.add_trace(
        go.Histogram(
            x=returns * 100,
            nbinsx=50,
            marker_color=COLORS['primary'],
            opacity=0.7,
            name='Returns'
        ),
        row=1, col=1
    )
    fig.add_vline(x=var_95 * 100, line_dash="dash", line_color=COLORS['warning'],
                  annotation_text=f"VaR 95%: {var_95*100:.2f}%", row=1, col=1)
    fig.add_vline(x=var_99 * 100, line_dash="dash", line_color=COLORS['danger'],
                  annotation_text=f"VaR 99%: {var_99*100:.2f}%", row=1, col=1)

    # 2. Drawdown Chart
    cumulative = (1 + returns).cumprod()
    peak = cumulative.expanding().max()
    drawdown = (cumulative - peak) / peak * 100

    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            fill='tozeroy',
            fillcolor='rgba(231, 76, 60, 0.3)',
            line=dict(color=COLORS['danger']),
            name='Drawdown'
        ),
        row=1, col=2
    )

    # 3. Rolling Volatility
    rolling_vol = returns.rolling(window=20).std() * np.sqrt(252) * 100

    fig.add_trace(
        go.Scatter(
            x=rolling_vol.index,
            y=rolling_vol.values,
            line=dict(color=COLORS['info']),
            name='20-day Vol'
        ),
        row=2, col=1
    )

    if benchmark_returns is not None:
        bench_vol = benchmark_returns.rolling(window=20).std() * np.sqrt(252) * 100
        fig.add_trace(
            go.Scatter(
                x=bench_vol.index,
                y=bench_vol.values,
                line=dict(color=COLORS['neutral'], dash='dash'),
                name='Benchmark Vol'
            ),
            row=2, col=1
        )

    # 4. Return Distribution with Normal overlay
    fig.add_trace(
        go.Histogram(
            x=returns * 100,
            nbinsx=50,
            marker_color=COLORS['secondary'],
            opacity=0.7,
            name='Actual'
        ),
        row=2, col=2
    )

    # Normal distribution overlay
    x_range = np.linspace(returns.min() * 100, returns.max() * 100, 100)
    normal_dist = (1 / (returns.std() * 100 * np.sqrt(2 * np.pi))) * \
                  np.exp(-0.5 * ((x_range - returns.mean() * 100) / (returns.std() * 100)) ** 2)
    normal_dist = normal_dist * len(returns) * (returns.max() - returns.min()) * 100 / 50

    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=normal_dist,
            line=dict(color=COLORS['warning'], width=2),
            name='Normal'
        ),
        row=2, col=2
    )

    fig.update_layout(
        title=title,
        template='plotly_dark',
        paper_bgcolor=COLORS['background'],
        plot_bgcolor=COLORS['background'],
        height=700,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )

    return fig


def create_correlation_heatmap(
    returns_dict: Dict[str, pd.Series],
    title: str = "Correlation Matrix"
) -> go.Figure:
    """
    Create correlation heatmap for multiple assets.
    """
    # Build returns DataFrame
    returns_df = pd.DataFrame(returns_dict)
    corr_matrix = returns_df.corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale=[
            [0, COLORS['danger']],
            [0.5, COLORS['neutral']],
            [1, COLORS['success']]
        ],
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverinfo="z"
    ))

    fig.update_layout(
        title=title,
        template='plotly_dark',
        paper_bgcolor=COLORS['background'],
        plot_bgcolor=COLORS['background'],
        height=500,
        xaxis_title="",
        yaxis_title=""
    )

    return fig


def create_trade_analysis_chart(
    trades: List[Dict[str, Any]],
    title: str = "Trade Analysis"
) -> go.Figure:
    """
    Create comprehensive trade analysis visualization.

    Shows:
    - Win/loss distribution
    - Trade duration analysis
    - PnL by entry time
    - Cumulative PnL
    """
    if not trades:
        fig = go.Figure()
        fig.add_annotation(text="No trades to analyze", showarrow=False)
        return fig

    # Convert to DataFrame
    df = pd.DataFrame(trades)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'PnL Distribution',
            'Cumulative PnL',
            'Win/Loss by Day of Week',
            'Trade Duration vs PnL'
        ),
        specs=[[{"type": "histogram"}, {"type": "scatter"}],
               [{"type": "bar"}, {"type": "scatter"}]]
    )

    # 1. PnL Distribution
    if 'pnl' in df.columns:
        colors = [COLORS['success'] if p > 0 else COLORS['danger'] for p in df['pnl']]
        fig.add_trace(
            go.Histogram(
                x=df['pnl'],
                marker_color=COLORS['primary'],
                nbinsx=30,
                name='PnL'
            ),
            row=1, col=1
        )

    # 2. Cumulative PnL
    if 'pnl' in df.columns:
        cumulative_pnl = df['pnl'].cumsum()
        fig.add_trace(
            go.Scatter(
                x=list(range(len(cumulative_pnl))),
                y=cumulative_pnl,
                fill='tozeroy',
                fillcolor='rgba(0, 212, 170, 0.3)',
                line=dict(color=COLORS['primary']),
                name='Cumulative'
            ),
            row=1, col=2
        )

    # 3. Win/Loss by Day of Week
    if 'entry_date' in df.columns and 'pnl' in df.columns:
        df['day'] = pd.to_datetime(df['entry_date']).dt.dayofweek
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
        day_stats = df.groupby('day')['pnl'].agg(['sum', 'count']).reindex(range(5))

        fig.add_trace(
            go.Bar(
                x=day_names,
                y=day_stats['sum'].fillna(0),
                marker_color=[
                    COLORS['success'] if v > 0 else COLORS['danger']
                    for v in day_stats['sum'].fillna(0)
                ],
                name='PnL by Day'
            ),
            row=2, col=1
        )

    # 4. Trade Duration vs PnL
    if all(col in df.columns for col in ['entry_date', 'exit_date', 'pnl']):
        df['duration'] = (
            pd.to_datetime(df['exit_date']) - pd.to_datetime(df['entry_date'])
        ).dt.days

        colors = [COLORS['success'] if p > 0 else COLORS['danger'] for p in df['pnl']]
        fig.add_trace(
            go.Scatter(
                x=df['duration'],
                y=df['pnl'],
                mode='markers',
                marker=dict(color=colors, size=8),
                name='Duration vs PnL'
            ),
            row=2, col=2
        )

    fig.update_layout(
        title=title,
        template='plotly_dark',
        paper_bgcolor=COLORS['background'],
        plot_bgcolor=COLORS['background'],
        height=600,
        showlegend=False
    )

    return fig


def create_monte_carlo_chart(
    returns: pd.Series,
    n_simulations: int = 1000,
    n_days: int = 252,
    initial_value: float = 100000,
    title: str = "Monte Carlo Simulation"
) -> go.Figure:
    """
    Create Monte Carlo simulation visualization.

    Simulates future portfolio paths based on historical returns.
    """
    mean_return = returns.mean()
    std_return = returns.std()

    # Run simulations
    simulations = np.zeros((n_simulations, n_days))
    simulations[:, 0] = initial_value

    for t in range(1, n_days):
        random_returns = np.random.normal(mean_return, std_return, n_simulations)
        simulations[:, t] = simulations[:, t-1] * (1 + random_returns)

    # Calculate percentiles
    percentiles = [5, 25, 50, 75, 95]
    percentile_values = np.percentile(simulations, percentiles, axis=0)

    fig = go.Figure()

    # Add percentile bands
    x = list(range(n_days))

    # 5-95 percentile band
    fig.add_trace(go.Scatter(
        x=x + x[::-1],
        y=list(percentile_values[4]) + list(percentile_values[0][::-1]),
        fill='toself',
        fillcolor='rgba(0, 212, 170, 0.1)',
        line=dict(color='rgba(0,0,0,0)'),
        name='5-95% Range'
    ))

    # 25-75 percentile band
    fig.add_trace(go.Scatter(
        x=x + x[::-1],
        y=list(percentile_values[3]) + list(percentile_values[1][::-1]),
        fill='toself',
        fillcolor='rgba(0, 212, 170, 0.3)',
        line=dict(color='rgba(0,0,0,0)'),
        name='25-75% Range'
    ))

    # Median
    fig.add_trace(go.Scatter(
        x=x,
        y=percentile_values[2],
        line=dict(color=COLORS['primary'], width=2),
        name='Median'
    ))

    # Sample paths (show 20 random paths)
    for i in np.random.choice(n_simulations, min(20, n_simulations), replace=False):
        fig.add_trace(go.Scatter(
            x=x,
            y=simulations[i],
            line=dict(color=COLORS['neutral'], width=0.5),
            opacity=0.3,
            showlegend=False
        ))

    # Add annotations
    final_median = percentile_values[2][-1]
    final_5th = percentile_values[0][-1]
    final_95th = percentile_values[4][-1]

    fig.add_annotation(
        x=n_days - 1,
        y=final_median,
        text=f"Median: ${final_median:,.0f}",
        showarrow=True,
        arrowhead=2
    )

    fig.update_layout(
        title=f"{title}<br><sub>{n_simulations} simulations, {n_days} days</sub>",
        template='plotly_dark',
        paper_bgcolor=COLORS['background'],
        plot_bgcolor=COLORS['background'],
        height=500,
        xaxis_title="Days",
        yaxis_title="Portfolio Value ($)",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )

    return fig


def create_feature_importance_chart(
    importance: Dict[str, float],
    top_n: int = 20,
    title: str = "Feature Importance"
) -> go.Figure:
    """
    Create horizontal bar chart for feature importance.
    """
    # Sort and get top N
    sorted_imp = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    names = [x[0] for x in sorted_imp]
    values = [x[1] for x in sorted_imp]

    # Color based on positive/negative
    colors = [COLORS['success'] if v > 0 else COLORS['danger'] for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=names,
        orientation='h',
        marker_color=colors
    ))

    fig.update_layout(
        title=title,
        template='plotly_dark',
        paper_bgcolor=COLORS['background'],
        plot_bgcolor=COLORS['background'],
        height=max(400, len(names) * 25),
        yaxis=dict(autorange="reversed"),
        xaxis_title="Importance",
        showlegend=False
    )

    return fig


def create_regime_chart(
    prices: pd.Series,
    regimes: pd.Series,
    title: str = "Market Regime Analysis"
) -> go.Figure:
    """
    Create chart showing price with regime overlay.
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3],
        subplot_titles=(title, 'Regime')
    )

    # Price chart
    fig.add_trace(
        go.Scatter(
            x=prices.index,
            y=prices.values,
            line=dict(color=COLORS['primary']),
            name='Price'
        ),
        row=1, col=1
    )

    # Regime background colors
    regime_colors = {
        'TRENDING_UP': COLORS['success'],
        'TRENDING_DOWN': COLORS['danger'],
        'MEAN_REVERTING': COLORS['warning'],
        'HIGH_VOLATILITY': COLORS['info'],
        'NEUTRAL': COLORS['neutral']
    }

    # Add regime indicator
    if hasattr(regimes, 'cat'):
        regime_numeric = regimes.cat.codes
    else:
        unique_regimes = regimes.unique()
        regime_map = {r: i for i, r in enumerate(unique_regimes)}
        regime_numeric = regimes.map(regime_map)

    fig.add_trace(
        go.Scatter(
            x=regimes.index,
            y=regime_numeric,
            mode='markers',
            marker=dict(
                color=[regime_colors.get(str(r), COLORS['neutral']) for r in regimes],
                size=8
            ),
            name='Regime'
        ),
        row=2, col=1
    )

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor=COLORS['background'],
        plot_bgcolor=COLORS['background'],
        height=600,
        showlegend=True
    )

    return fig


def create_rolling_metrics_chart(
    returns: pd.Series,
    window: int = 60,
    title: str = "Rolling Performance Metrics"
) -> go.Figure:
    """
    Create chart showing rolling performance metrics.
    """
    # Calculate rolling metrics
    rolling_return = returns.rolling(window).mean() * 252 * 100
    rolling_vol = returns.rolling(window).std() * np.sqrt(252) * 100
    rolling_sharpe = rolling_return / rolling_vol

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=('Rolling Return (%)', 'Rolling Volatility (%)', 'Rolling Sharpe')
    )

    fig.add_trace(
        go.Scatter(
            x=rolling_return.index,
            y=rolling_return.values,
            fill='tozeroy',
            line=dict(color=COLORS['primary']),
            name='Return'
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=rolling_vol.index,
            y=rolling_vol.values,
            line=dict(color=COLORS['warning']),
            name='Volatility'
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=rolling_sharpe.index,
            y=rolling_sharpe.values,
            fill='tozeroy',
            line=dict(color=COLORS['info']),
            name='Sharpe'
        ),
        row=3, col=1
    )

    # Add zero line for Sharpe
    fig.add_hline(y=0, line_dash="dash", line_color=COLORS['neutral'], row=3, col=1)
    fig.add_hline(y=1, line_dash="dash", line_color=COLORS['success'], row=3, col=1)

    fig.update_layout(
        title=f"{title} ({window}-day rolling window)",
        template='plotly_dark',
        paper_bgcolor=COLORS['background'],
        plot_bgcolor=COLORS['background'],
        height=600,
        showlegend=False
    )

    return fig
