# 📈 StockAnalyzer Pro

**Production-grade stock analysis system with ML predictions, backtesting, and adaptive learning.**

[![Tests](https://img.shields.io/badge/tests-37%20passed-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.10+-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

## ⚠️ Disclaimer

**This software is for EDUCATIONAL and RESEARCH purposes only. It does NOT constitute financial advice. Past performance does not guarantee future results. All investments carry risk of loss.**

## Features

### 🔬 Technical Analysis
- **15+ Indicators**: RSI, MACD, Bollinger Bands, ATR, OBV, Stochastic, ADX, and more
- **73 Features**: Comprehensive feature engineering for ML
- **Signal Generation**: Automated buy/sell/hold signals with confidence scores

### 🤖 Machine Learning
- **Ensemble Models**: XGBoost + Random Forest + Gradient Boosting
- **Walk-Forward Validation**: Prevents overfitting with proper temporal validation
- **Feature Importance**: Understand what drives predictions

### 📉 Backtesting
- **Realistic Costs**: Commission and slippage modeling
- **Risk Management**: Stop-loss, take-profit, max drawdown limits
- **Kill Switch**: Automatic trading halt on excessive losses
- **Comprehensive Metrics**: Sharpe, Sortino, Calmar, profit factor, win rate

### 🧠 Adaptive Learning
- **Online Learning**: Models update incrementally with new data
- **Drift Detection**: Automatic detection of performance degradation
- **Retraining Triggers**: Smart identification of when to retrain

### 🖥️ Production UI
- **Interactive Dashboard**: Built with Streamlit and Plotly
- **Real-time Analysis**: Instant technical analysis
- **Beautiful Charts**: Candlestick, volume, RSI, MACD panels

## Quick Start

```bash
# Clone and install
cd stock_analyzer
pip install -r requirements.txt

# Run the UI
streamlit run ui/app.py

# Or run tests
python -m pytest tests/ -v
```

## Project Structure

```
stock_analyzer/
├── core/
│   ├── config.py      # Configuration management
│   └── errors.py      # Error handling, circuit breaker
├── data/
│   └── fetchers.py    # Data sources with caching
├── features/
│   └── technical.py   # 73 technical features
├── models/
│   └── ensemble.py    # ML ensemble with validation
├── backtest/
│   └── engine.py      # Production backtester
├── adaptive/
│   └── learning.py    # Online learning & drift detection
├── ui/
│   └── app.py         # Streamlit dashboard
├── tests/
│   └── unit/
│       └── test_all.py  # 37 comprehensive tests
└── requirements.txt
```

## Architecture

### Data Flow
```
Data Sources → Cache → Feature Engine → ML Models → Signals → Backtest
                                          ↓
                              Adaptive Learning ← Outcomes
```

### Key Design Decisions

1. **Ensemble over single models**: Combines strengths of different algorithms
2. **Walk-forward validation**: Prevents lookahead bias
3. **Circuit breaker pattern**: Protects against cascading failures
4. **Kill switch**: Stops trading on excessive losses
5. **Online learning**: Adapts to changing market conditions

## Metrics Explained

| Metric | Good Value | Description |
|--------|------------|-------------|
| Sharpe Ratio | > 1.0 | Risk-adjusted return |
| Sortino Ratio | > 1.5 | Downside risk-adjusted return |
| Calmar Ratio | > 2.0 | Return vs max drawdown |
| Win Rate | > 50% | Percentage of profitable trades |
| Profit Factor | > 1.5 | Gross profit / gross loss |
| Max Drawdown | < 20% | Largest peak-to-trough decline |

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test class
python -m pytest tests/unit/test_all.py::TestModels -v

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=html
```

## Configuration

All settings are in `core/config.py`:

```python
# Model settings
min_train_samples = 50
buy_threshold = 0.6
sell_threshold = 0.4

# Backtest settings
commission_pct = 0.001
stop_loss_pct = 0.05
max_drawdown_limit = 0.2

# Adaptive learning
enable_online_learning = True
drift_threshold = 0.05
```

## Future Roadmap

- [ ] Real-time data integration
- [ ] News sentiment analysis (FinBERT)
- [ ] Portfolio optimization
- [ ] Email/SMS alerts
- [ ] Multi-asset support
- [ ] API server (FastAPI)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Write tests for new features
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

---

**Remember**: This is a research tool. Never risk money you can't afford to lose. Always do your own research.
