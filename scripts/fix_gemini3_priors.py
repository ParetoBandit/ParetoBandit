#!/usr/bin/env python3
"""
Fix Gemini 3 Pro Preview entries that were generated before the max_tokens fix.

This script:
1. Identifies the 32 clusters with bad Gemini 3 entries (before Dec 18 05:34)
2. Re-runs those clusters with max_tokens=4000
3. Grades using GPT-4o as teacher
4. Updates the log file
5. Regenerates expert priors

Usage:
    python scripts/fix_gemini3_priors.py
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from openai import OpenAI


# Configuration
MODEL = "google/gemini-3-pro-preview"
MAX_TOKENS = 4000
FIX_TIMESTAMP = datetime(2025, 12, 18, 5, 34, 10).timestamp()
TEACHER_MODEL = "openai/gpt-4o"


def call_openrouter(model_id: str, prompt: str, max_tokens: int = 4000) -> str:
    """Call a model via OpenRouter."""
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            timeout=120.0,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[ERROR] {e}"


def grade_with_gpt4o(prompt: str, response: str) -> float:
    """Grade a response using GPT-4o as teacher."""
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    
    grading_prompt = f"""Rate the quality of this AI response on a scale from 0 to 1.

PROMPT: {prompt[:500]}

RESPONSE: {response[:2000]}

Provide ONLY a number between 0 and 1 (e.g., 0.85). No explanation."""
    
    try:
        result = client.chat.completions.create(
            model=TEACHER_MODEL,
            messages=[{"role": "user", "content": grading_prompt}],
            max_tokens=10,
            temperature=0,
        )
        score_text = result.choices[0].message.content.strip()
        # Extract first number
        import re
        match = re.search(r'(\d*\.?\d+)', score_text)
        if match:
            score = float(match.group(1))
            return min(max(score, 0), 1)
        return 0.5
    except Exception as e:
        print(f"  Grading error: {e}")
        return 0.5


def main():
    # Check API key
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY not set!")
        return 1
    
    print("=" * 80)
    print("Fixing Gemini 3 Pro Preview Entries")
    print("=" * 80)
    
    # Load log and find bad entries
    log_path = PROJECT_ROOT / "banditgpt" / "data" / "priors" / "archetype_grid_dense_run.jsonl"
    prompts_path = PROJECT_ROOT / "banditgpt" / "data" / "priors" / "archetype_grid_prompts.jsonl"
    
    # Load prompts
    prompts = {}
    with open(prompts_path) as f:
        for line in f:
            data = json.loads(line)
            prompts[data['cluster_id']] = data['prompt']
    
    # Find bad Gemini 3 entries
    bad_clusters = set()
    all_entries = []
    
    with open(log_path) as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            all_entries.append(data)
            
            if data.get('model_id') == MODEL:
                ts = data.get('ts', 0)
                if ts < FIX_TIMESTAMP:
                    bad_clusters.add(data['cluster_id'])
    
    print(f"Found {len(bad_clusters)} clusters with bad Gemini 3 entries")
    print(f"Clusters: {sorted(bad_clusters)[:10]}...")
    
    if not bad_clusters:
        print("No bad entries to fix!")
        return 0
    
    # Re-run bad clusters
    new_entries = []
    
    for i, cluster_id in enumerate(sorted(bad_clusters)):
        prompt = prompts.get(cluster_id)
        if not prompt:
            print(f"[{i+1}/{len(bad_clusters)}] Cluster {cluster_id}: No prompt found, skipping")
            continue
        
        print(f"\n[{i+1}/{len(bad_clusters)}] Cluster {cluster_id}: {prompt[:50]}...")
        
        # Call Gemini 3 with proper max_tokens
        response = call_openrouter(MODEL, prompt, max_tokens=MAX_TOKENS)
        ok = response and not response.startswith("[ERROR")
        
        if not ok:
            print(f"  ERROR: {response[:50]}")
            new_entries.append({
                "cluster_id": cluster_id,
                "model_id": MODEL,
                "ok": False,
                "reward_logit": None,
                "ts": time.time(),
            })
            continue
        
        print(f"  Response length: {len(response)}")
        
        # Grade with GPT-4o
        score = grade_with_gpt4o(prompt, response)
        # Convert probability to logit (clamp to avoid log(0) or log(inf))
        import math
        score_clamped = max(0.01, min(0.99, score))
        reward_logit = math.log(score_clamped / (1 - score_clamped))
        
        print(f"  Score: {score:.2f}, Reward logit: {reward_logit:.2f}")
        
        new_entries.append({
            "cluster_id": cluster_id,
            "model_id": MODEL,
            "ok": True,
            "reward_logit": reward_logit,
            "teacher_used": True,
            "ts": time.time(),
        })
        
        time.sleep(1)  # Rate limit
    
    # Update log file - remove old bad entries and add new ones
    print("\n" + "=" * 80)
    print("Updating log file...")
    
    updated_entries = []
    removed_count = 0
    
    for entry in all_entries:
        # Keep entry unless it's a bad Gemini 3 entry
        if entry.get('model_id') == MODEL and entry.get('cluster_id') in bad_clusters:
            ts = entry.get('ts', 0)
            if ts < FIX_TIMESTAMP:
                removed_count += 1
                continue
        updated_entries.append(entry)
    
    # Add new entries
    updated_entries.extend(new_entries)
    
    print(f"Removed {removed_count} old bad entries")
    print(f"Added {len(new_entries)} new entries")
    
    # Write updated log
    backup_path = log_path.with_suffix('.jsonl.bak')
    import shutil
    shutil.copy(log_path, backup_path)
    print(f"Backed up log to {backup_path}")
    
    with open(log_path, 'w') as f:
        for entry in updated_entries:
            f.write(json.dumps(entry) + "\n")
    
    print(f"Updated {log_path}")
    
    # Summary of new Gemini 3 entries
    new_rewards = [e['reward_logit'] for e in new_entries if e.get('reward_logit') is not None]
    if new_rewards:
        print(f"\nNew entries mean reward: {sum(new_rewards)/len(new_rewards):.2f}")
    
    print("\n" + "=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print("1. Verify the log file looks correct")
    print("2. Regenerate expert priors:")
    print("   python experiments/generate_expert_priors.py")
    print("3. Test routing to Gemini 3:")
    print("   python -c \"from banditgpt import BanditRouter; ...\"")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
