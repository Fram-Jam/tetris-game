"""
StockAnalyzer Pro - Production UI
==================================
Professional stock analysis dashboard with:
- Real-time analysis
- ML predictions
- Backtesting
- Benchmarking
- Adaptive learning insights
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

# Add parent to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import config, DISCLAIMER
from core.error_messages import classify_error, format_error_for_ui, UserError, ErrorCategory
from data.fetchers import DataFetcher
from features.technical import FeatureEngine, Signal
from models.ensemble import EnsembleModel, create_labels, prepare_training_data
from backtest.engine import Backtester, generate_signals_from_predictions
from adaptive.learning import AdaptiveLearningManager

# Page config
st.set_page_config(
    page_title="StockAnalyzer Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    .main-header {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00D4AA 0%, #00B894 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 1.2rem;
        border: 1px solid rgba(0, 212, 170, 0.2);
    }
    
    .signal-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    .signal-strong-buy { background: linear-gradient(135deg, #00D4AA 0%, #00B894 100%); color: white; }
    .signal-buy { background: rgba(0, 212, 170, 0.3); color: #00D4AA; border: 1px solid #00D4AA; }
    .signal-hold { background: rgba(255, 193, 7, 0.3); color: #FFC107; border: 1px solid #FFC107; }
    .signal-sell { background: rgba(255, 107, 107, 0.3); color: #FF6B6B; border: 1px solid #FF6B6B; }
    .signal-strong-sell { background: linear-gradient(135deg, #FF6B6B 0%, #EE5A5A 100%); color: white; }
    
    .explanation-card {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 3px solid #00D4AA;
    }
</style>
""", unsafe_allow_html=True)


def get_signal_badge(signal: Signal) -> str:
    """Get HTML for signal badge."""
    signal_classes = {
        Signal.STRONG_BUY: "signal-strong-buy",
        Signal.BUY: "signal-buy",
        Signal.HOLD: "signal-hold",
        Signal.SELL: "signal-sell",
        Signal.STRONG_SELL: "signal-strong-sell"
    }
    css_class = signal_classes.get(signal, "signal-hold")
    return f'<span class="signal-badge {css_class}">{signal.value.upper().replace("_", " ")}</span>'


def create_price_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    """Create interactive candlestick chart with indicators."""
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.5, 0.15, 0.15, 0.2],
        subplot_titles=(f'{symbol} Price', 'Volume', 'RSI', 'MACD')
    )
    
    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price',
            increasing_line_color='#00D4AA',
            decreasing_line_color='#FF6B6B'
        ),
        row=1, col=1
    )
    
    # Moving averages
    if 'sma_20' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['sma_20'], name='SMA 20',
                      line=dict(color='#FFC107', width=1)),
            row=1, col=1
        )
    
    if 'sma_50' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['sma_50'], name='SMA 50',
                      line=dict(color='#00B894', width=1)),
            row=1, col=1
        )
    
    # Volume
    colors = ['#00D4AA' if c >= o else '#FF6B6B' 
              for c, o in zip(df['close'], df['open'])]
    fig.add_trace(
        go.Bar(x=df.index, y=df['volume'], name='Volume',
               marker_color=colors, opacity=0.7),
        row=2, col=1
    )
    
    # RSI
    if 'rsi' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['rsi'], name='RSI',
                      line=dict(color='#00D4AA', width=1.5)),
            row=3, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    # MACD
    if 'macd' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['macd'], name='MACD',
                      line=dict(color='#00D4AA', width=1.5)),
            row=4, col=1
        )
        if 'macd_signal' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['macd_signal'], name='Signal',
                          line=dict(color='#FF6B6B', width=1.5)),
                row=4, col=1
            )
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=800,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_rangeslider_visible=False,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig


def create_equity_chart(result) -> go.Figure:
    """Create equity curve chart."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3],
        subplot_titles=('Equity Curve', 'Drawdown')
    )
    
    fig.add_trace(
        go.Scatter(
            x=result.equity_curve.index,
            y=result.equity_curve.values,
            name='Portfolio Value',
            fill='tozeroy',
            line=dict(color='#00D4AA', width=2),
            fillcolor='rgba(0, 212, 170, 0.1)'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=result.drawdown_curve.index,
            y=result.drawdown_curve.values * 100,
            name='Drawdown %',
            fill='tozeroy',
            line=dict(color='#FF6B6B', width=1),
            fillcolor='rgba(255, 107, 107, 0.2)'
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=500,
        showlegend=True,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig


def main():
    """Main application."""
    
    # Initialize session state
    if 'data_fetcher' not in st.session_state:
        st.session_state.data_fetcher = DataFetcher(use_mock=False)  # Use real data sources
    if 'adaptive_manager' not in st.session_state:
        st.session_state.adaptive_manager = AdaptiveLearningManager()
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'backtest_result' not in st.session_state:
        st.session_state.backtest_result = None
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        
        symbol = st.text_input("Stock Symbol", value="AAPL", max_chars=10).upper()
        
        period = st.selectbox(
            "Time Period",
            options=['1mo', '3mo', '6mo', '1y', '2y'],
            index=4
        )
        
        st.markdown("**Quick Picks:**")
        cols = st.columns(4)
        quick_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'JPM']
        for i, sym in enumerate(quick_symbols):
            if cols[i % 4].button(sym, key=f"quick_{sym}"):
                symbol = sym
        
        st.divider()
        
        run_backtest = st.checkbox("Run Backtest", value=True)
        train_model = st.checkbox("Train ML Model", value=True)
        
        if run_backtest:
            initial_capital = st.number_input("Initial Capital ($)", 
                                            value=100000, min_value=1000, step=10000)
            commission = st.slider("Commission (%)", 0.0, 1.0, 0.1, 0.05) / 100
            stop_loss = st.slider("Stop Loss (%)", 1.0, 20.0, 5.0, 0.5) / 100
        else:
            initial_capital = 100000
            commission = 0.001
            stop_loss = 0.05
        
        st.divider()
        
        with st.expander("⚠️ Disclaimer"):
            st.markdown(DISCLAIMER)
        
        analyze_clicked = st.button("🔍 Analyze", type="primary", use_container_width=True)
    
    # Main content
    st.markdown('<h1 class="main-header">📈 StockAnalyzer Pro</h1>', unsafe_allow_html=True)
    st.markdown("*Production-grade stock analysis with ML predictions and backtesting*")
    
    if analyze_clicked:
        with st.spinner(f"Analyzing {symbol}..."):
            # Data fetching phase
            try:
                data = st.session_state.data_fetcher.get_stock_data(symbol, period)

                if data is None or data.prices is None or len(data.prices) == 0:
                    st.error(f"**Could not fetch data for {symbol}**")
                    st.info(
                        "This could be due to:\n"
                        "- Invalid stock symbol - verify the ticker is correct\n"
                        "- Network issues - check your internet connection\n"
                        "- Data provider temporarily unavailable\n\n"
                        "Try one of the quick pick symbols (AAPL, MSFT, etc.) to verify connectivity."
                    )
                    return

            except Exception as e:
                user_error = classify_error(e, {'symbol': symbol, 'operation': 'fetching stock data'})
                st.error(format_error_for_ui(user_error, show_technical=config.debug))
                return

            # Feature engineering phase
            try:
                engine = FeatureEngine(data.prices)

                st.session_state.current_data = data
                st.session_state.current_engine = engine

                signal, confidence, summary = engine.get_overall_signal()
                st.session_state.current_signal = (signal, confidence, summary)

            except Exception as e:
                user_error = classify_error(e, {'symbol': symbol, 'operation': 'calculating indicators'})
                st.error(format_error_for_ui(user_error, show_technical=config.debug))
                return

            # Model training phase
            if train_model:
                try:
                    ml_features = engine.get_ml_features()
                    min_samples = config.model.min_train_samples

                    if len(ml_features) < min_samples:
                        st.warning(
                            f"**Insufficient data for ML training**\n\n"
                            f"Available: {len(ml_features)} samples, Required: {min_samples}\n\n"
                            f"**Suggestion:** Select a longer time period (1-2 years) to get enough training data."
                        )
                    else:
                        labels = create_labels(data.prices)
                        X, y = prepare_training_data(ml_features, data.prices)

                        if len(X) > 100:
                            train_size = int(len(X) * 0.8)
                            X_train, X_test = X[:train_size], X[train_size:]
                            y_train, y_test = y[:train_size], y[train_size:]

                            model = EnsembleModel()
                            model.fit(X_train, y_train)
                            st.session_state.model = model
                            st.session_state.model_metrics = model.evaluate(X_test, y_test)
                            st.session_state.X_test = X_test
                            st.session_state.y_test = y_test
                        else:
                            st.warning(
                                f"**Not enough valid samples for training**\n\n"
                                f"After preprocessing, only {len(X)} samples remain.\n\n"
                                f"**Suggestion:** Try a longer time period or a more liquid stock."
                            )

                except Exception as e:
                    user_error = classify_error(e, {'symbol': symbol, 'operation': 'training ML model'})
                    st.warning(format_error_for_ui(user_error, show_technical=config.debug))

            # Backtesting phase
            if run_backtest and st.session_state.model is not None:
                try:
                    proba = st.session_state.model.predict_proba(st.session_state.X_test)
                    predictions = pd.Series(
                        proba[:, 1] if proba.shape[1] > 1 else proba[:, 0],
                        index=st.session_state.X_test.index
                    )
                    signals_series = generate_signals_from_predictions(predictions)
                    test_prices = data.prices.loc[signals_series.index]

                    bt = Backtester(
                        initial_capital=initial_capital,
                        commission_pct=commission
                    )
                    bt.risk_manager.stop_loss_pct = stop_loss

                    result = bt.run(test_prices, signals_series, symbol=symbol)
                    st.session_state.backtest_result = result

                    if result.metrics.total_trades == 0:
                        st.info(
                            "**No trades executed during backtest**\n\n"
                            "The model predictions didn't trigger any buy signals above the threshold.\n\n"
                            "**Suggestions:**\n"
                            "- Lower the buy threshold in model settings\n"
                            "- Try a longer time period for more opportunities\n"
                            "- Check if the model is producing varied predictions"
                        )

                except Exception as e:
                    error_str = str(e).lower()
                    if 'drawdown' in error_str:
                        st.warning(
                            "**Backtest stopped: Maximum drawdown exceeded**\n\n"
                            "The strategy lost too much capital, triggering the safety limit.\n\n"
                            "This is normal - it means the strategy underperformed during this period. "
                            "Consider adjusting stop-loss settings or testing on a different time period."
                        )
                    elif 'consecutive' in error_str:
                        st.warning(
                            "**Backtest stopped: Too many consecutive losses**\n\n"
                            "The strategy hit a losing streak, triggering the safety limit.\n\n"
                            "Consider adjusting the buy/sell thresholds or testing on different market conditions."
                        )
                    else:
                        user_error = classify_error(e, {'symbol': symbol, 'operation': 'running backtest'})
                        st.warning(format_error_for_ui(user_error, show_technical=config.debug))

            st.success(f"Analysis complete for {symbol}!")
    
    # Display results
    if hasattr(st.session_state, 'current_data') and st.session_state.current_data is not None:
        data = st.session_state.current_data
        engine = st.session_state.current_engine
        signal, confidence, summary = st.session_state.current_signal
        
        tabs = st.tabs(["📊 Overview", "📈 Charts", "🤖 ML Analysis", "📉 Backtest", "🧠 Adaptive"])
        
        with tabs[0]:
            info = data.info or {}
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.markdown(f"### {info.get('name', symbol)}")
                st.markdown(f"**Sector:** {info.get('sector', 'Unknown')}")
            
            with col2:
                current_price = data.prices['close'].iloc[-1]
                prev_price = data.prices['close'].iloc[-2] if len(data.prices) > 1 else current_price
                change = (current_price - prev_price) / prev_price * 100
                st.metric("Current Price", f"${current_price:.2f}", f"{change:+.2f}%")
            
            with col3:
                st.markdown(f"**Signal:** {get_signal_badge(signal)}", unsafe_allow_html=True)
                st.markdown(f"**Confidence:** {confidence:.1f}%")
            
            st.divider()
            
            st.markdown("### Key Metrics")
            metrics_cols = st.columns(5)
            
            indicators = engine.get_all_indicators()
            
            with metrics_cols[0]:
                rsi = indicators.get('rsi')
                if rsi:
                    st.metric("RSI (14)", f"{rsi.value:.1f}")
            
            with metrics_cols[1]:
                macd = indicators.get('macd')
                if macd:
                    st.metric("MACD", f"{macd.value:.4f}")
            
            with metrics_cols[2]:
                bb = indicators.get('bollinger')
                if bb:
                    st.metric("BB Position", f"{bb.value:.1%}")
            
            with metrics_cols[3]:
                if 'volatility_20d' in engine.df.columns:
                    vol = engine.df['volatility_20d'].iloc[-1]
                    st.metric("Volatility", f"{vol:.1%}" if not pd.isna(vol) else "N/A")
            
            with metrics_cols[4]:
                if 'volume_ratio' in engine.df.columns:
                    vr = engine.df['volume_ratio'].iloc[-1]
                    st.metric("Volume Ratio", f"{vr:.2f}" if not pd.isna(vr) else "N/A")
            
            st.markdown("### Analysis Summary")
            for name, result in indicators.items():
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.markdown(f"**{result.name}**")
                    st.markdown(get_signal_badge(result.signal), unsafe_allow_html=True)
                with col2:
                    st.info(result.explanation)
        
        with tabs[1]:
            st.markdown("### Price Chart")
            fig = create_price_chart(engine.df, symbol)
            st.plotly_chart(fig, use_container_width=True)
        
        with tabs[2]:
            st.markdown("### ML Analysis")
            
            if st.session_state.model is not None and hasattr(st.session_state, 'model_metrics'):
                metrics = st.session_state.model_metrics
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Accuracy", f"{metrics.accuracy:.1%}")
                with col2:
                    st.metric("Precision", f"{metrics.precision:.1%}")
                with col3:
                    st.metric("Recall", f"{metrics.recall:.1%}")
                with col4:
                    st.metric("F1 Score", f"{metrics.f1:.1%}")
                
                st.markdown("### Feature Importance")
                importance = st.session_state.model.get_feature_importance()
                if importance:
                    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15]
                    
                    fig = go.Figure(go.Bar(
                        x=[v for k, v in sorted_imp],
                        y=[k for k, v in sorted_imp],
                        orientation='h',
                        marker_color='#00D4AA'
                    ))
                    fig.update_layout(
                        template='plotly_dark',
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        height=400,
                        yaxis=dict(autorange="reversed")
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Train the ML model to see analysis.")
        
        with tabs[3]:
            st.markdown("### Backtest Results")
            
            if st.session_state.backtest_result is not None:
                result = st.session_state.backtest_result
                m = result.metrics
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Return", f"{m.total_return:.2%}")
                with col2:
                    st.metric("Sharpe Ratio", f"{m.sharpe_ratio:.2f}")
                with col3:
                    st.metric("Max Drawdown", f"{m.max_drawdown:.2%}")
                with col4:
                    st.metric("Win Rate", f"{m.win_rate:.1%}")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Trades", m.total_trades)
                with col2:
                    pf = f"{m.profit_factor:.2f}" if m.profit_factor != float('inf') else "∞"
                    st.metric("Profit Factor", pf)
                with col3:
                    st.metric("Sortino", f"{m.sortino_ratio:.2f}")
                with col4:
                    st.metric("Calmar", f"{m.calmar_ratio:.2f}")
                
                if len(result.equity_curve) > 0:
                    st.markdown("### Equity Curve")
                    fig = create_equity_chart(result)
                    st.plotly_chart(fig, use_container_width=True)
                
                if result.trades:
                    st.markdown("### Trade History")
                    trades_df = pd.DataFrame([t.to_dict() for t in result.trades])
                    st.dataframe(trades_df, use_container_width=True)
            else:
                st.info("Run backtest to see results.")
        
        with tabs[4]:
            st.markdown("### Adaptive Learning")
            
            insights = st.session_state.adaptive_manager.get_learning_insights()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Accuracy", f"{insights['current_metrics']['accuracy']:.1%}")
            with col2:
                st.metric("Trend", insights['performance_trend'].replace('_', ' ').title())
            with col3:
                should_retrain, reason = insights['should_retrain']
                st.metric("Retrain Needed", "Yes" if should_retrain else "No")
            
            if insights['last_drift']:
                st.warning(f"Drift: {insights['last_drift']['recommendation']}")
            
            if insights['top_features']:
                st.markdown("### Top Features")
                for feature, importance in list(insights['top_features'].items())[:10]:
                    st.progress(min(abs(importance) * 10, 1.0), text=f"{feature}: {importance:.4f}")
    
    else:
        st.markdown("""
        <div style="text-align: center; padding: 4rem 0;">
            <h2>Welcome to StockAnalyzer Pro</h2>
            <p>Enter a stock symbol and click <b>Analyze</b> to get started.</p>
            <p style="font-size: 0.9rem; margin-top: 2rem;">
                Features: Technical Analysis • ML Predictions • Backtesting • Adaptive Learning
            </p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
