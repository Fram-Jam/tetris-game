"""
AI-Powered Health Insights Page
"""

import streamlit as st
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.insights.ai_coach import get_ai_insights, get_correlation_insights

st.set_page_config(page_title="AI Insights | HealthBridge", layout="wide", page_icon="🌉")

# Initialize session state if not already done
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.health_data = None
    st.session_state.patient_profile = None
    st.session_state.lab_data = None

# Load data if needed
if not st.session_state.data_loaded:
    from src.data.synthetic.patient_generator import generate_demo_data
    from src.data.synthetic.lab_generator import generate_lab_history
    from datetime import datetime

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

st.title("🤖 AI Health Insights")
st.markdown("Personalized analysis and recommendations powered by AI")

# Check for API key
has_api_key = False
try:
    has_api_key = bool(st.secrets.get("ANTHROPIC_API_KEY") or st.secrets.get("OPENAI_API_KEY"))
except:
    pass

if not has_api_key:
    st.info("""
    💡 **Enhanced AI Available**

    To enable AI-powered insights, add your API key to `.streamlit/secrets.toml`:
    ```
    ANTHROPIC_API_KEY = "your-key-here"
    ```

    Currently using rule-based insights which still provide valuable analysis!
    """)

st.markdown("---")

# Main insights section
if st.session_state.health_data:

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 📊 Your Health Analysis")

        with st.spinner("Analyzing your health data..."):
            insights = get_ai_insights(
                st.session_state.health_data,
                st.session_state.patient_profile,
                st.session_state.lab_data
            )

        st.markdown(insights)

        st.markdown("---")

        # Ask a specific question
        st.markdown("### 💬 Ask a Question")
        question = st.text_input(
            "What would you like to know about your health data?",
            placeholder="e.g., Why is my HRV lower this week?"
        )

        if question:
            with st.spinner("Thinking..."):
                answer = get_ai_insights(
                    st.session_state.health_data,
                    st.session_state.patient_profile,
                    st.session_state.lab_data,
                    specific_question=question
                )
            st.markdown("#### Answer")
            st.markdown(answer)

    with col2:
        st.markdown("### 🔗 Correlations Found")

        correlations = get_correlation_insights(st.session_state.health_data)

        if correlations:
            for corr in correlations[:5]:
                strength_emoji = "🔴" if corr['strength'] > 0.5 else "🟡" if corr['strength'] > 0.3 else "🟢"
                with st.expander(f"{strength_emoji} {corr['finding']}"):
                    st.markdown(corr['detail'])
                    st.progress(min(corr['strength'], 1.0))
        else:
            st.info("Not enough data to identify correlations yet. Keep tracking!")

        st.markdown("---")

        st.markdown("### 💡 Quick Tips")
        st.markdown("""
        **Low HRV?**
        - Prioritize sleep
        - Reduce alcohol
        - Try meditation

        **Poor sleep?**
        - Limit caffeine after 2pm
        - Reduce screen time before bed
        - Keep bedroom cool (65-68°F)

        **Low readiness?**
        - Consider an active recovery day
        - Light walking or yoga
        - Extra hydration

        **High resting HR?**
        - Could indicate stress
        - Check hydration
        - Monitor for illness
        """)

        st.markdown("---")

        # Weekly summary stats
        st.markdown("### 📈 This Week vs Last")

        if len(st.session_state.health_data) >= 14:
            this_week = st.session_state.health_data[-7:]
            last_week = st.session_state.health_data[-14:-7]

            metrics = [
                ("Sleep", "sleep_duration", "h"),
                ("HRV", "hrv", "ms"),
                ("Steps", "steps", ""),
                ("Readiness", "readiness_score", ""),
            ]

            for name, key, unit in metrics:
                this_avg = sum(d[key] for d in this_week) / 7
                last_avg = sum(d[key] for d in last_week) / 7
                delta = this_avg - last_avg
                delta_pct = (delta / last_avg) * 100 if last_avg > 0 else 0

                if key == "steps":
                    st.metric(name, f"{this_avg:,.0f}{unit}", f"{delta_pct:+.1f}%")
                else:
                    st.metric(name, f"{this_avg:.1f}{unit}", f"{delta_pct:+.1f}%")

else:
    st.warning("No health data available. Connect your devices to get personalized insights.")

# Disclaimer
st.markdown("---")
st.caption("""
⚕️ **Disclaimer:** These insights are for informational purposes only and are not medical advice.
Always consult with healthcare professionals for medical decisions.
""")
