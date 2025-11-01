import json
import logging
import os

from google import genai
from google.genai import types
from pydantic import BaseModel


# IMPORTANT: KEEP THIS COMMENT
# Follow these instructions when using this blueprint:
# - Note that the newest Gemini model series is "gemini-2.5-flash" or gemini-2.5-pro"
#   - do not change this unless explicitly requested by the user
# - Sometimes the google genai SDK has occasional type errors. You might need to run to validate, at time.  
# The SDK was recently renamed from google-generativeai to google-genai. This file reflects the new name and the new APIs.

# This API key is from Gemini Developer API Key, not vertex AI API Key
def get_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        return genai.Client(api_key=api_key)
    except Exception as e:
        print(f"Error initializing Gemini client: {e}")
        return None


def generate_water_conservation_tips(usage_data: dict) -> str:
    client = get_client()
    if not client:
        return "⚠️ Gemini API key not configured. Please add your GEMINI_API_KEY to use AI-powered features."
    
    prompt = f"""
    Based on the following water usage data, provide 3-5 personalized, actionable water conservation tips.
    
    Usage Data:
    - Average Daily Usage: {usage_data.get('avg_daily_usage', 'N/A')} liters
    - Peak Usage Time: {usage_data.get('peak_time', 'N/A')}
    - Detected Anomalies: {usage_data.get('anomalies_count', 0)}
    - Usage Trend: {usage_data.get('trend', 'N/A')}
    
    Provide practical, friendly advice that can help reduce water consumption.
    Format each tip as a bullet point with estimated water savings.
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text or "Unable to generate conservation tips at this time."


def answer_water_usage_question(question: str, context_data: dict) -> str:
    client = get_client()
    if not client:
        return "⚠️ Gemini API key not configured. Please add your GEMINI_API_KEY to use AI-powered features."
    
    prompt = f"""
    You are AquaMind, an AI assistant specialized in water management and conservation.
    
    User Question: {question}
    
    Context Data Available:
    - Total Records: {context_data.get('total_records', 'N/A')}
    - Date Range: {context_data.get('date_range', 'N/A')}
    - Average Usage: {context_data.get('avg_usage', 'N/A')} liters/day
    - Anomalies Detected: {context_data.get('anomalies_count', 0)}
    - Last Anomaly Date: {context_data.get('last_anomaly_date', 'None detected')}
    
    Additional Info: {context_data.get('additional_info', '')}
    
    Provide a clear, helpful answer based on the available data. If you don't have enough information,
    explain what data would be needed to answer the question accurately.
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text or "I'm unable to answer that question at this time."


def predict_water_usage_insights(usage_history: list, prediction: float) -> str:
    client = get_client()
    if not client:
        return "⚠️ Gemini API key not configured. Please add your GEMINI_API_KEY to use AI-powered features."
    
    prompt = f"""
    Analyze this water usage pattern and provide insights:
    
    Recent Usage History (last 7 days in liters): {usage_history}
    Predicted Next Day Usage: {prediction} liters
    
    Provide:
    1. Analysis of the usage trend
    2. Whether the prediction suggests normal or elevated consumption
    3. Potential reasons for the predicted usage
    4. Recommendations for optimization
    
    Keep the response concise and actionable.
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text or "Unable to generate insights at this time."


def analyze_anomaly(anomaly_data: dict) -> str:
    client = get_client()
    if not client:
        return "⚠️ Gemini API key not configured. Please add your GEMINI_API_KEY to use AI-powered features."
    
    prompt = f"""
    Analyze this water usage anomaly and provide diagnostic information:
    
    Anomaly Details:
    - Flow Rate: {anomaly_data.get('flow_rate', 'N/A')} L/min
    - Normal Range: {anomaly_data.get('normal_range', 'N/A')} L/min
    - Time Detected: {anomaly_data.get('timestamp', 'N/A')}
    - Severity Score: {anomaly_data.get('severity', 'N/A')}
    
    Provide:
    1. Likely cause of the anomaly
    2. Severity assessment
    3. Recommended actions
    4. Whether this could indicate a leak
    
    Be specific and actionable.
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text or "Unable to analyze anomaly at this time."
