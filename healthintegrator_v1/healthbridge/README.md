# HealthBridge

**The unified platform for all your health data.**

HealthBridge connects your wearables, lab results, and clinical data into a single view with AI-powered insights. Think of it as "Plaid for Healthcare" - aggregating data from multiple sources into one coherent picture.

## Features

- **Multi-device support**: Oura Ring, Apple Watch, WHOOP, Garmin, Fitbit, CGMs (Dexcom, Libre)
- **Unified dashboard**: See all your health metrics at a glance
- **AI-powered insights**: Get personalized recommendations based on your data
- **Lab tracking**: Monitor biomarkers over time with trend analysis
- **Correlation analysis**: Discover patterns between sleep, activity, HRV, and more
- **Day-of-week patterns**: See how your metrics vary throughout the week

## Quick Start

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)

### Installation

```bash
# Clone the repo (or download the files)
cd healthbridge

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up secrets (optional, for AI features)
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit secrets.toml with your API keys

# Run the app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Demo Mode

The app runs in **demo mode** by default, using synthetic data that mimics real health patterns. This allows you to:

- Explore all features without connecting real devices
- Show the platform to investors or stakeholders
- Understand what the full product would look like

To use real data:
1. Add API keys in Settings or `.streamlit/secrets.toml`
2. Connect your devices via the "Connect Devices" page
3. Upload Apple Health exports or other data files

## Testing & Validation

Run the validation script to check for errors:

```bash
python tests/validate_app.py
```

Run the full test suite:

```bash
python -m pytest tests/ -v
```

## Project Structure

```
healthbridge/
├── app.py                    # Main entry point
├── pages/                    # Streamlit pages
│   ├── 1_🏠_Dashboard.py    # Main health dashboard
│   ├── 2_🔗_Connect_Devices.py  # Device management
│   ├── 3_📊_Deep_Dive.py    # Detailed analysis
│   ├── 4_🧬_Clinical_Data.py    # Lab results
│   ├── 5_🤖_AI_Insights.py  # AI recommendations
│   └── 6_⚙️_Settings.py     # User settings
├── src/
│   ├── data/                 # Data handling
│   │   ├── connectors/       # Device API connectors
│   │   ├── synthetic/        # Demo data generators
│   │   └── normalizer.py     # Unified data schema
│   └── insights/             # Analytics & AI
│       └── ai_coach.py       # AI-powered insights
├── data/                     # Data storage
│   ├── sample/               # Sample data files
│   └── user/                 # User uploaded data
├── .streamlit/               # Streamlit configuration
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Configuration

### API Keys

For AI-powered insights, add your API key to `.streamlit/secrets.toml`:

```toml
ANTHROPIC_API_KEY = "sk-ant-your-key-here"
# or
OPENAI_API_KEY = "sk-your-key-here"
```

### Device Connections

The demo simulates device connections. For real integrations, the platform supports:

- **Terra API**: Unified access to 200+ wearable devices
- **Direct APIs**: Oura, WHOOP (with API access)
- **File imports**: Apple Health XML exports, Fitbit/Garmin JSON exports

## Deployment

### Streamlit Community Cloud (Free)

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Add secrets in the Streamlit dashboard (Settings > Secrets)
5. Deploy!

### Other Platforms

The app can be deployed on any platform that supports Python:
- Heroku
- Railway
- Render
- AWS/GCP/Azure

## Technology Stack

- **Frontend**: Streamlit
- **Charts**: Plotly
- **Data**: Pandas, NumPy
- **AI**: Anthropic Claude / OpenAI GPT-4
- **Synthetic Data**: Faker

## Roadmap

- [ ] Real Terra API integration
- [ ] Apple Health direct sync (via HealthKit)
- [ ] More CGM integrations
- [ ] Meal logging with glucose correlation
- [ ] Workout recommendations based on readiness
- [ ] Mobile-optimized layout
- [ ] Multi-user support

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - feel free to use this for your own projects.

## Support

For questions or issues:
- Open a GitHub issue
- Email: support@healthbridge.demo

---

Built with love for the health-conscious community.
