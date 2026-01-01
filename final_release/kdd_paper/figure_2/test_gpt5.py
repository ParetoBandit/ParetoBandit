#!/usr/bin/env python3
"""Test GPT-5 API call with real missing prompt."""

import json
import requests
from pathlib import Path

API_KEY = "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"
DATA_DIR = Path('/Users/annette/repostitories/llm_jury/final_release/data')

# Find a missing prompt for gpt-5
print("Finding missing prompt for gpt-5...")
existing = set()
with open(DATA_DIR / 'train_rewards.jsonl') as f:
    for line in f:
        try:
            r = json.loads(line)
            if r.get('ok') and r['model_id'] == 'openai/gpt-5' and r['cluster_id'] in [36, 80]:
                existing.add(r['prompt'])
        except: pass

missing_prompt = None
with open(DATA_DIR / 'train_prompts.jsonl') as f:
    for line in f:
        p = json.loads(line)
        if p['cluster_id'] in [36, 80] and p['prompt'] not in existing:
            missing_prompt = p['prompt']
            cluster = p['cluster_id']
            break

if not missing_prompt:
    print("No missing prompts found!")
    exit(1)

print(f"\nPrompt (C{cluster}): {missing_prompt[:100]}...")
print(f"Length: {len(missing_prompt)} chars\n")

# Call GPT-5
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://github.com/banditgpt/llm-jury",
    "X-Title": "llm_jury",
}

payload = {
    "model": "openai/gpt-5",
    "messages": [{"role": "user", "content": missing_prompt}],
    "temperature": 0.7,
    "max_tokens": 8000,
    "stream": False
}

print("Calling GPT-5 API...")
try:
    resp = requests.post("https://openrouter.ai/api/v1/chat/completions", 
                        headers=headers, json=payload, timeout=180)
    print(f"Status: {resp.status_code}\n")
    
    data = resp.json()
    
    # Print structure
    print(f"Response keys: {list(data.keys())}")
    print(f"Has 'choices': {'choices' in data}")
    
    if 'choices' in data:
        msg = data['choices'][0]['message']
        print(f"\nMessage keys: {list(msg.keys())}")
        
        content = msg.get('content', '')
        reasoning = msg.get('reasoning', '')
        
        print(f"\nContent length: {len(content)}")
        print(f"Reasoning length: {len(reasoning)}")
        
        if content:
            print(f"\nContent preview: {content[:200]}...")
        else:
            print("\n⚠️  CONTENT IS EMPTY!")
            
        if reasoning:
            print(f"\nReasoning preview: {reasoning[:200]}...")
            
        print(f"\n✅ SUCCESS - Response received")
    else:
        print(f"\n❌ ERROR: No 'choices' in response")
        print(f"Full response: {json.dumps(data, indent=2)}")
        
except Exception as e:
    print(f"\n❌ ERROR: {e}")
