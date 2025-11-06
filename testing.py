import os
import requests

API_KEY = os.getenv("GEMINI_API_KEY")  # Make sure you set this first

url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={API_KEY}"

payload = {
    "contents": [
        {
            "parts": [
                {"text": "Hello! Can you confirm you're working?"}
            ]
        }
    ]
}

headers = {"Content-Type": "application/json"}

response = requests.post(url, headers=headers, json=payload)

print("Status:", response.status_code)
print("Response:")
print(response.json())
