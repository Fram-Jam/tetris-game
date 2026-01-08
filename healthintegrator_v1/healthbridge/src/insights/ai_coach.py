"""
AI-Powered Health Insights

Uses LLM to generate personalized health recommendations.
"""

import os
from typing import List, Dict, Optional
from datetime import date, timedelta
import json

# Try Anthropic first, fall back to OpenAI
AI_PROVIDER = None

try:
    from anthropic import Anthropic
    AI_PROVIDER = "anthropic"
except ImportError:
    pass

if AI_PROVIDER is None:
    try:
        from openai import OpenAI
        AI_PROVIDER = "openai"
    except ImportError:
        AI_PROVIDER = "none"

HEALTH_COACH_SYSTEM_PROMPT = """You are an expert health coach AI assistant integrated into HealthBridge, a unified health data platform. You analyze data from multiple wearables (Oura, Apple Watch, Whoop, Garmin) and clinical sources to provide personalized, actionable health insights.

Your role:
1. Analyze patterns in sleep, HRV, activity, recovery, and glucose data
2. Identify correlations between behaviors and outcomes
3. Provide specific, actionable recommendations
4. Flag concerning trends that may warrant medical attention
5. Celebrate wins and positive trends

Guidelines:
- Be encouraging but honest
- Ground recommendations in the data provided
- Use plain language, avoid medical jargon
- When suggesting behavior changes, be specific (e.g., "try finishing dinner by 7pm" not "eat earlier")
- Always note that you're not a replacement for medical advice
- Focus on the 1-3 most impactful insights, don't overwhelm

Response format:
- Start with a brief overall assessment (1-2 sentences)
- List 2-3 key insights with specific data points
- Provide 1-2 actionable recommendations
- End with encouragement or a question to engage the user"""


def prepare_health_summary(data: List[Dict], days: int = 7) -> str:
    """Prepare a text summary of recent health data for the AI."""
    recent = data[-days:] if len(data) >= days else data

    # Calculate averages and trends
    avg_sleep = sum(d['sleep_duration'] for d in recent) / len(recent)
    avg_hrv = sum(d['hrv'] for d in recent) / len(recent)
    avg_rhr = sum(d['resting_hr'] for d in recent) / len(recent)
    avg_steps = sum(d['steps'] for d in recent) / len(recent)
    avg_readiness = sum(d['readiness_score'] for d in recent) / len(recent)

    # Compare to previous period
    if len(data) >= days * 2:
        prev = data[-(days*2):-days]
        prev_sleep = sum(d['sleep_duration'] for d in prev) / len(prev)
        prev_hrv = sum(d['hrv'] for d in prev) / len(prev)
        prev_steps = sum(d['steps'] for d in prev) / len(prev)

        sleep_trend = "up" if avg_sleep > prev_sleep else "down"
        hrv_trend = "up" if avg_hrv > prev_hrv else "down"
        steps_trend = "up" if avg_steps > prev_steps else "down"
    else:
        sleep_trend = hrv_trend = steps_trend = "stable"

    # Find best and worst days
    best_sleep_day = max(recent, key=lambda d: d['sleep_score'])
    worst_sleep_day = min(recent, key=lambda d: d['sleep_score'])

    # Glucose summary if available
    glucose_days = [d for d in recent if d.get('glucose')]
    glucose_summary = ""
    if glucose_days:
        avg_glucose = sum(d['glucose']['avg'] for d in glucose_days) / len(glucose_days)
        avg_tir = sum(d['glucose']['time_in_range'] for d in glucose_days) / len(glucose_days)
        glucose_summary = f"""
Glucose (CGM):
- Average: {avg_glucose:.0f} mg/dL
- Time in range (70-180): {avg_tir:.0f}%
"""

    summary = f"""
HEALTH DATA SUMMARY (Last {days} days)

Sleep:
- Average duration: {avg_sleep:.1f} hours (trend: {sleep_trend})
- Average sleep score: {sum(d['sleep_score'] for d in recent) / len(recent):.0f}
- Best night: {best_sleep_day['date']} (score: {best_sleep_day['sleep_score']})
- Worst night: {worst_sleep_day['date']} (score: {worst_sleep_day['sleep_score']})

Heart Health:
- Average HRV: {avg_hrv:.0f} ms (trend: {hrv_trend})
- Average resting HR: {avg_rhr:.0f} bpm

Activity:
- Average daily steps: {avg_steps:,.0f} (trend: {steps_trend})
- Average active minutes: {sum(d['active_minutes'] for d in recent) / len(recent):.0f}

Recovery:
- Average readiness score: {avg_readiness:.0f}
{glucose_summary}
Latest readings ({recent[-1]['date']}):
- Sleep: {recent[-1]['sleep_duration']:.1f}h (score: {recent[-1]['sleep_score']})
- HRV: {recent[-1]['hrv']:.0f} ms
- Resting HR: {recent[-1]['resting_hr']} bpm
- Steps: {recent[-1]['steps']:,}
- Readiness: {recent[-1]['readiness_score']}
"""

    return summary


def get_ai_insights(
    health_data: List[Dict],
    patient_profile: Optional[object] = None,
    lab_data: Optional[List] = None,
    specific_question: Optional[str] = None
) -> str:
    """Get AI-generated health insights."""

    if AI_PROVIDER == "none":
        return _get_rule_based_insights(health_data)

    # Prepare context
    health_summary = prepare_health_summary(health_data)

    # Add patient context if available
    patient_context = ""
    if patient_profile:
        patient_context = f"""
User Profile:
- Age: {patient_profile.age}
- Activity Level: {patient_profile.activity_level}
- Health Goals: General wellness and optimization
"""

    # Add lab context if available
    lab_context = ""
    if lab_data and len(lab_data) > 0:
        latest_labs = lab_data[-1]
        lab_context = f"""
Recent Lab Results ({latest_labs.date}):
"""
        for marker, result in list(latest_labs.results.items())[:10]:
            flag = f" ({result['flag']})" if result.get('flag') else ""
            lab_context += f"- {marker}: {result['value']} {result['unit']}{flag}\n"

    # Build prompt
    user_message = f"""
{health_summary}
{patient_context}
{lab_context}

{"User Question: " + specific_question if specific_question else "Please provide your health insights and recommendations based on this data."}
"""

    # Call AI
    try:
        if AI_PROVIDER == "anthropic":
            import streamlit as st
            api_key = os.environ.get("ANTHROPIC_API_KEY") or st.secrets.get("ANTHROPIC_API_KEY")
            if not api_key:
                return _get_rule_based_insights(health_data)

            client = Anthropic(api_key=api_key)
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                system=HEALTH_COACH_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}]
            )
            return response.content[0].text

        elif AI_PROVIDER == "openai":
            import streamlit as st
            api_key = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
            if not api_key:
                return _get_rule_based_insights(health_data)

            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4-turbo-preview",
                max_tokens=1024,
                messages=[
                    {"role": "system", "content": HEALTH_COACH_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ]
            )
            return response.choices[0].message.content

    except Exception as e:
        return f"AI insights temporarily unavailable. Error: {str(e)}\n\n" + _get_rule_based_insights(health_data)

    return _get_rule_based_insights(health_data)


def _get_rule_based_insights(data: List[Dict]) -> str:
    """Fallback rule-based insights when AI is unavailable."""
    recent = data[-7:] if len(data) >= 7 else data

    insights = []

    # Sleep analysis
    avg_sleep = sum(d['sleep_duration'] for d in recent) / len(recent)
    if avg_sleep < 7:
        insights.append(f"⚠️ **Sleep Alert**: Your average sleep of {avg_sleep:.1f} hours is below the recommended 7-9 hours. Consider setting a consistent bedtime and limiting screen time before bed.")
    elif avg_sleep >= 7:
        insights.append(f"✅ **Great Sleep**: Averaging {avg_sleep:.1f} hours puts you in a healthy range. Keep up the good work!")

    # HRV analysis
    avg_hrv = sum(d['hrv'] for d in recent) / len(recent)
    hrv_trend = recent[-1]['hrv'] - recent[0]['hrv']
    if hrv_trend > 5:
        insights.append(f"📈 **HRV Improving**: Your HRV is trending up (+{hrv_trend:.0f} ms this week), suggesting improving recovery and stress resilience.")
    elif hrv_trend < -5:
        insights.append(f"📉 **HRV Declining**: Your HRV has dropped {abs(hrv_trend):.0f} ms this week. Consider prioritizing rest, reducing alcohol, and managing stress.")

    # Activity analysis
    avg_steps = sum(d['steps'] for d in recent) / len(recent)
    if avg_steps < 7000:
        insights.append(f"🚶 **Activity Opportunity**: You're averaging {avg_steps:,.0f} steps. Try to hit 8,000+ daily for optimal cardiovascular benefits.")
    elif avg_steps > 10000:
        insights.append(f"🏃 **Excellent Activity**: Outstanding! Averaging {avg_steps:,.0f} steps daily puts you in elite territory.")

    # Readiness analysis
    avg_readiness = sum(d['readiness_score'] for d in recent) / len(recent)
    if avg_readiness < 65:
        insights.append(f"⚡ **Recovery Needed**: Your readiness score has been averaging {avg_readiness:.0f}. Consider taking a rest day or reducing training intensity.")
    elif avg_readiness > 80:
        insights.append(f"💪 **Peak Condition**: With an average readiness of {avg_readiness:.0f}, you're well-recovered and ready to perform!")

    # Correlation insight
    best_sleep_day = max(recent, key=lambda d: d['sleep_score'])
    if best_sleep_day['steps'] > avg_steps:
        insights.append(f"💡 **Pattern Found**: Your best sleep ({best_sleep_day['date']}) came after a high-activity day ({best_sleep_day['steps']:,} steps). Exercise may improve your sleep quality.")

    # Glucose insight if available
    glucose_days = [d for d in recent if d.get('glucose')]
    if glucose_days:
        avg_tir = sum(d['glucose']['time_in_range'] for d in glucose_days) / len(glucose_days)
        if avg_tir > 85:
            insights.append(f"🩸 **Excellent Glucose Control**: {avg_tir:.0f}% time in range is outstanding! Your diet and activity are working well.")
        elif avg_tir < 70:
            insights.append(f"🩸 **Glucose Variability**: {avg_tir:.0f}% time in range suggests room for improvement. Consider timing carbs around activity.")

    result = "\n\n".join(insights) if insights else "Keep up the good work! Your health metrics look stable."

    result += "\n\n---\n*💡 Tip: Add an API key in Settings to enable AI-powered personalized insights.*"

    return result


def get_correlation_insights(data: List[Dict]) -> List[Dict]:
    """Find statistical correlations in the data."""
    import pandas as pd
    import numpy as np

    df = pd.DataFrame(data)

    correlations = []

    # Sleep vs next-day HRV
    if len(df) > 7:
        df['next_day_hrv'] = df['hrv'].shift(-1)
        corr = df['sleep_duration'].corr(df['next_day_hrv'])
        if abs(corr) > 0.3:
            direction = "better" if corr > 0 else "worse"
            correlations.append({
                'finding': f"Sleep duration correlates with next-day HRV",
                'detail': f"More sleep = {direction} HRV (r={corr:.2f})",
                'strength': abs(corr)
            })

    # Steps vs sleep quality
    corr = df['steps'].corr(df['sleep_score'])
    if abs(corr) > 0.25:
        correlations.append({
            'finding': "Activity level affects sleep quality",
            'detail': f"Higher step counts correlate with {'better' if corr > 0 else 'worse'} sleep (r={corr:.2f})",
            'strength': abs(corr)
        })

    # HRV vs Readiness
    corr = df['hrv'].corr(df['readiness_score'])
    if abs(corr) > 0.3:
        correlations.append({
            'finding': "HRV strongly predicts readiness",
            'detail': f"Your HRV is {'closely' if abs(corr) > 0.5 else 'moderately'} tied to readiness scores (r={corr:.2f})",
            'strength': abs(corr)
        })

    # Sleep score vs next-day readiness
    if len(df) > 1:
        df['next_day_readiness'] = df['readiness_score'].shift(-1)
        corr = df['sleep_score'].corr(df['next_day_readiness'])
        if abs(corr) > 0.3:
            correlations.append({
                'finding': "Sleep quality predicts next-day recovery",
                'detail': f"Better sleep leads to higher readiness the next day (r={corr:.2f})",
                'strength': abs(corr)
            })

    # Resting HR vs sleep quality (inverse relationship expected)
    corr = df['resting_hr'].corr(df['sleep_score'])
    if abs(corr) > 0.25:
        correlations.append({
            'finding': "Resting heart rate reflects sleep quality",
            'detail': f"Lower resting HR correlates with {'better' if corr < 0 else 'worse'} sleep (r={corr:.2f})",
            'strength': abs(corr)
        })

    return sorted(correlations, key=lambda x: x['strength'], reverse=True)
