import requests
import json
import os

API_KEY = "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"
MODEL = "anthropic/claude-4.5-sonnet"
PROMPT = "Why does the knee lock up after a menisectomy?"

def debug_call():
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/banditgpt/llm-jury",
        "X-Title": "llm_jury",
    }
    
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0.7,
        "max_tokens": 5000,
        "stream": False
    }
    
    print(f"Sending request to {MODEL}...")
    try:
        resp = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=60)
        print(f"Status Code: {resp.status_code}")
        print("Raw Response Headers:", resp.headers)
        print("Raw Response Body:")
        print(json.dumps(resp.json(), indent=2))
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    debug_call()
