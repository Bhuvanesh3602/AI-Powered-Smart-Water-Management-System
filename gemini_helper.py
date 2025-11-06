import os
import requests
import json

# 🔹 Set your Gemini API key (Replace with your actual key or set via environment variable)
os.environ["GEMINI_API_KEY"] = "YOUR_API_KEY_HERE"

# 🔹 Gemini REST API endpoint
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

# 🔹 Helper function to call Gemini API
def call_gemini_api(prompt: str) -> str:
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": os.environ.get("GEMINI_API_KEY")
    }

    data = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }

    response = requests.post(GEMINI_URL, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        try:
            # Extract text safely from response JSON
            return response.json()["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            return "⚠️ Could not parse Gemini API response."
    else:
        return f"❌ Error {response.status_code}: {response.text}"


# 🔹 Generate Water Conservation Tips
def generate_water_conservation_tips(usage_data: dict) -> str:
    prompt = f"""
    Based on the following water usage data, provide 3–5 personalized, actionable water conservation tips.
    
    Usage Data:
    - Average Daily Usage: {usage_data.get('avg_daily_usage', 'N/A')} liters
    - Peak Usage Time: {usage_data.get('peak_time', 'N/A')}
    - Detected Anomalies: {usage_data.get('anomalies_count', 0)}
    - Usage Trend: {usage_data.get('trend', 'N/A')}
    
    Provide practical, friendly advice that can help reduce water consumption.
    Format each tip as a bullet point with estimated water savings.
    """
    return call_gemini_api(prompt)


# 🔹 Answer Water Usage Question
def answer_water_usage_question(question: str, context_data: dict) -> str:
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
    return call_gemini_api(prompt)


# 🔹 Predict Water Usage Insights
def predict_water_usage_insights(usage_history: list, prediction: float) -> str:
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
    return call_gemini_api(prompt)


# 🔹 Analyze Anomaly
def analyze_anomaly(anomaly_data: dict) -> str:
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
    return call_gemini_api(prompt)


# 🔹 Example Test (Run directly)
if __name__ == "__main__":
    test_data = {
        "avg_daily_usage": 135,
        "peak_time": "7:00 AM",
        "anomalies_count": 2,
        "trend": "Slightly increasing"
    }

    print("💧 Generating Water Conservation Tips...\n")
    print(generate_water_conservation_tips(test_data))
