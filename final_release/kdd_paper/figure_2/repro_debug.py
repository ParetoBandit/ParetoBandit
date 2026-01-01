import requests
import json
import time

API_KEY = "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"
BASE_URL = "https://openrouter.ai/api/v1"

def test_model(model_id):
    print(f"Testing {model_id}...")
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/banditgpt/llm-jury",
        "X-Title": "llm_jury_debug",
    }
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": "Say hello."}],
        "temperature": 0.7,
        "max_tokens": 100,
    }
    try:
        resp = requests.post(f"{BASE_URL}/chat/completions", headers=headers, json=payload, timeout=30)
        print(f"Status Code: {resp.status_code}")
        if resp.status_code != 200:
            print(f"Error Body: {resp.text}")
        else:
            print(f"Success! Content: {resp.json()['choices'][0]['message']['content'][:50]}...")
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    test_model("qwen/qwen3-8b")
    print("-" * 20)
    test_model("openai/gpt-oss-20b")
