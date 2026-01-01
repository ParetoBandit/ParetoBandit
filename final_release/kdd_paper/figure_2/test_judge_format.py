#!/usr/bin/env python3
"""Test markdown judge format on a real failed pair."""

import requests
import json
import re

API_KEY = "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"
BASE_URL = "https://openrouter.ai/api/v1"

# Load test data
with open('/tmp/test_data.json') as f:
    data = json.load(f)
    model_id = data['model']
    prompt = data['prompt']
    cluster = data['cluster']

print(f"=== TESTING MARKDOWN JUDGE ===")
print(f"Model: {model_id}")
print(f"Cluster: C{cluster}")
print(f"Prompt: {prompt[:80]}...")
print()

# Step 1: Get model response with max_tokens=8000
print("Step 1: Getting model response (max_tokens=8000)...")
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://github.com/banditgpt/llm-jury",
}

payload = {
    "model": model_id,
    "messages": [{"role": "user", "content": prompt}],
    "temperature": 0.7,
    "max_tokens": 8000,
    "stream": False
}

try:
    resp = requests.post(f"{BASE_URL}/chat/completions", 
                        headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    
    if "choices" not in data:
        print(f"ERROR: No choices in response: {data}")
        exit(1)
    
    response = data["choices"][0]["message"].get("content", "")
    print(f"✓ Got response: {len(response)} chars")
    print(f"  Preview: {response[:100]}...")
    print()
except Exception as e:
    print(f"✗ Model API failed: {e}")
    exit(1)

# Step 2: Test markdown judge (max_tokens=50)
print("Step 2: Testing markdown judge (max_tokens=50)...")
judge = "google/gemini-3-flash-preview" if 'gemini' not in model_id.lower() else "anthropic/claude-sonnet-4.5"
print(f"Judge model: {judge}")

judge_prompt = f"""Rate this response on a scale of 0.0 to 1.0.
Respond in markdown format with just the score.

PROMPT: {prompt}

RESPONSE: {response}

## Score
**Rating:**"""

payload = {
    "model": judge,
    "messages": [{"role": "user", "content": judge_prompt}],
    "temperature": 0.0,
    "max_tokens": 50,
    "stream": False
}

try:
    resp = requests.post(f"{BASE_URL}/chat/completions",
                        headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    judge_resp = resp.json()["choices"][0]["message"]["content"]
    
    print(f"✓ Judge response: '{judge_resp}'")
    print()
    
    # Extract score
    match = re.search(r'(\d+\.?\d*)', judge_resp)
    if match:
        score = float(match.group(1))
        if score > 1:
            score = score / 100
        score = max(0.0, min(1.0, score))
        print(f"✅ SUCCESS! Extracted score: {score}")
    else:
        print(f"⚠️  No numeric score found, trying fallback...")
        positive_words = ['excellent', 'good', 'great', 'comprehensive', 'accurate', 'clear']
        if any(word in judge_resp.lower() for word in positive_words):
            score = 0.7
        else:
            score = 0.3
        print(f"Fallback score: {score}")
        
except Exception as e:
    print(f"✗ Judge API failed: {e}")
    exit(1)
