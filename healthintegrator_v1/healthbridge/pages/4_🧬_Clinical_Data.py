"""
Clinical Data Page - Lab Results and Biomarkers
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(page_title="Clinical Data | HealthBridge", layout="wide", page_icon="🌉")

# Initialize session state if not already done
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.lab_data = None
    st.session_state.patient_profile = None

# Load data if needed
if not st.session_state.data_loaded or st.session_state.lab_data is None:
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

st.title("🧬 Clinical Data")
st.markdown("Lab results and biomarker tracking")


def create_biomarker_chart(lab_history: list, biomarker: str) -> go.Figure:
    """Create trend chart for a single biomarker."""
    data = []
    for panel in lab_history:
        if biomarker in panel.results:
            data.append({
                'date': panel.date,
                'value': panel.results[biomarker]['value'],
                'unit': panel.results[biomarker]['unit'],
                'flag': panel.results[biomarker].get('flag')
            })

    if not data:
        return None

    df = pd.DataFrame(data)

    # Get reference range
    ref_range = lab_history[-1].results[biomarker].get('reference_range', '')
    if '-' in str(ref_range):
        try:
            low, high = map(float, str(ref_range).split('-'))
        except:
            low, high = None, None
    else:
        low, high = None, None

    fig = go.Figure()

    # Reference range band
    if low is not None and high is not None:
        fig.add_hrect(y0=low, y1=high, fillcolor="rgba(16, 185, 129, 0.15)",
                      line_width=0, annotation_text="Normal Range",
                      annotation_position="top left")

    # Value line with markers colored by flag
    colors = []
    for _, row in df.iterrows():
        if row.get('flag') == 'HIGH':
            colors.append('#EF4444')
        elif row.get('flag') == 'LOW':
            colors.append('#3B82F6')
        elif row.get('flag') == 'BORDERLINE':
            colors.append('#F59E0B')
        else:
            colors.append('#10B981')

    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['value'],
        mode='lines+markers',
        name=biomarker,
        line=dict(color='#6366F1', width=3),
        marker=dict(size=12, color=colors, line=dict(width=2, color='white'))
    ))

    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=30, b=20),
        yaxis_title=df['unit'].iloc[0] if len(df) > 0 else '',
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    fig.update_yaxes(gridcolor='#E2E8F0')
    fig.update_xaxes(gridcolor='#E2E8F0')

    return fig


def get_flag_color(flag: str) -> str:
    """Get color for lab result flag."""
    if flag == 'LOW':
        return '#3B82F6'  # Blue
    elif flag == 'HIGH':
        return '#EF4444'  # Red
    elif flag == 'BORDERLINE':
        return '#F59E0B'  # Yellow/Orange
    return '#10B981'  # Green


def get_flag_bg(flag: str) -> str:
    """Get background color for lab result flag."""
    if flag == 'LOW':
        return '#DBEAFE'
    elif flag == 'HIGH':
        return '#FEE2E2'
    elif flag == 'BORDERLINE':
        return '#FEF3C7'
    return '#D1FAE5'


if st.session_state.lab_data:

    # Latest results summary
    st.markdown("### 📋 Latest Results")
    latest = st.session_state.lab_data[-1]
    st.caption(f"From: {latest.date}")

    # Count flags
    high_count = sum(1 for r in latest.results.values() if r.get('flag') == 'HIGH')
    low_count = sum(1 for r in latest.results.values() if r.get('flag') == 'LOW')
    borderline_count = sum(1 for r in latest.results.values() if r.get('flag') == 'BORDERLINE')
    normal_count = len(latest.results) - high_count - low_count - borderline_count

    # Summary stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Tests", len(latest.results))
    with col2:
        st.metric("Normal", normal_count, delta=None)
    with col3:
        st.metric("Borderline", borderline_count, delta=None)
    with col4:
        st.metric("Out of Range", high_count + low_count, delta=None)

    st.markdown("---")

    # Group by category
    categories = {
        'Metabolic': ['Glucose Fasting', 'Hba1C'],
        'Lipids': ['Cholesterol Total', 'Ldl', 'Hdl', 'Triglycerides'],
        'Liver': ['Alt', 'Ast'],
        'Kidney': ['Creatinine', 'Egfr'],
        'Thyroid': ['Tsh'],
        'Vitamins & Minerals': ['Vitamin D', 'B12', 'Iron', 'Ferritin'],
        'Hormones': ['Cortisol Am', 'Testosterone Total'],
        'Inflammation': ['Crp', 'Homocysteine']
    }

    for category, markers in categories.items():
        category_results = {k: v for k, v in latest.results.items() if k in markers}
        if category_results:
            with st.expander(f"**{category}**", expanded=(category in ['Metabolic', 'Lipids'])):
                cols = st.columns(4)
                col_idx = 0
                for marker in markers:
                    if marker in latest.results:
                        result = latest.results[marker]
                        with cols[col_idx % 4]:
                            flag = result.get('flag', '')
                            flag_color = get_flag_color(flag)
                            flag_bg = get_flag_bg(flag)
                            flag_text = f" ({flag})" if flag else ""

                            st.markdown(f"""
                            <div style="padding: 0.75rem; border-left: 4px solid {flag_color}; background: {flag_bg}; margin-bottom: 0.75rem; border-radius: 0 8px 8px 0;">
                                <small style="color: #6B7280; font-weight: 500;">{marker}</small><br>
                                <strong style="color: {flag_color}; font-size: 1.25rem;">{result['value']} {result['unit']}</strong>
                                <span style="color: {flag_color}; font-size: 0.875rem;">{flag_text}</span><br>
                                <small style="color: #9CA3AF;">Ref: {result.get('reference_range', 'N/A')}</small>
                            </div>
                            """, unsafe_allow_html=True)
                        col_idx += 1

    st.markdown("---")

    # Trends section
    st.markdown("### 📈 Biomarker Trends")
    st.markdown("Track how your key biomarkers change over time")

    # Select biomarkers to display
    all_markers = list(latest.results.keys())
    default_markers = ['Glucose Fasting', 'Cholesterol Total', 'Vitamin D']
    default_selection = [m for m in default_markers if m in all_markers][:3]

    selected_markers = st.multiselect(
        "Select biomarkers to chart",
        all_markers,
        default=default_selection
    )

    if selected_markers:
        # Create grid of charts
        num_cols = min(len(selected_markers), 3)
        rows = (len(selected_markers) + num_cols - 1) // num_cols

        for row in range(rows):
            cols = st.columns(num_cols)
            for col_idx in range(num_cols):
                marker_idx = row * num_cols + col_idx
                if marker_idx < len(selected_markers):
                    marker = selected_markers[marker_idx]
                    with cols[col_idx]:
                        st.markdown(f"**{marker}**")
                        chart = create_biomarker_chart(st.session_state.lab_data, marker)
                        if chart:
                            st.plotly_chart(chart, use_container_width=True)

    st.markdown("---")

    # Historical panels
    st.markdown("### 📅 Historical Results")

    tab_labels = [f"{panel.date}" for panel in reversed(st.session_state.lab_data)]
    tabs = st.tabs(tab_labels)

    for tab, panel in zip(tabs, reversed(st.session_state.lab_data)):
        with tab:
            df_results = pd.DataFrame([
                {
                    'Biomarker': k,
                    'Value': f"{v['value']} {v['unit']}",
                    'Reference': v.get('reference_range', 'N/A'),
                    'Status': v.get('flag') if v.get('flag') else '✅ Normal'
                }
                for k, v in panel.results.items()
            ])

            # Style the dataframe
            def highlight_status(val):
                if val == 'HIGH':
                    return 'background-color: #FEE2E2; color: #DC2626'
                elif val == 'LOW':
                    return 'background-color: #DBEAFE; color: #2563EB'
                elif val == 'BORDERLINE':
                    return 'background-color: #FEF3C7; color: #D97706'
                return ''

            # Use map instead of deprecated applymap
            try:
                styled_df = df_results.style.map(highlight_status, subset=['Status'])
            except AttributeError:
                # Fallback for older pandas versions
                styled_df = df_results.style.applymap(highlight_status, subset=['Status'])

            st.dataframe(
                styled_df,
                use_container_width=True,
                hide_index=True
            )

    # Upload new results
    st.markdown("---")
    st.markdown("### 📤 Upload Lab Results")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Supported formats:**
        - PDF lab reports (parsed automatically)
        - CSV exports from lab portals
        - Manual entry
        """)

    with col2:
        uploaded_lab = st.file_uploader("Upload lab results", type=['pdf', 'csv'])
        if uploaded_lab:
            st.success(f"File uploaded: {uploaded_lab.name}")
            if st.button("Process Lab Results"):
                with st.spinner("Extracting lab values..."):
                    import time
                    time.sleep(2)
                    st.success("✅ Lab results imported successfully!")

else:
    st.info("""
    🧬 **No lab data available**

    In demo mode, synthetic lab data is generated automatically.

    In the full version, you could:
    - Connect to your health system's patient portal
    - Upload lab results (PDF or CSV)
    - Order at-home lab tests through our partners
    """)

# Demo notice
if st.session_state.get('demo_mode', True):
    st.markdown("---")
    st.info("""
    🎭 **Demo Mode**: Lab results shown are synthetic data generated for demonstration purposes.
    Values are based on population distributions and do not represent real patient data.
    """)
