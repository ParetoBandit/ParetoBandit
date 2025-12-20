#!/usr/bin/env python3
"""
Re-run archetype grid for Gemini 3 Pro Preview with higher max_tokens.
Then regenerate the expert priors.

Usage:
    python scripts/rerun_gemini3_priors.py
"""

import json
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from banditgpt.core.model_manager import call_openrouter
from banditgpt.core.quality_cost_predictor import QualityCostPredictor
from banditgpt.core.tiered_grader import TieredGrader
from banditgpt._resources import get_quality_predictor_path, get_priors_path


def main():
    # Configuration
    MODEL = "google/gemini-3-pro-preview"
    MAX_TOKENS = 4000  # Higher limit for Gemini 3
    WORKERS = 5
    
    # Check API key
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY not set!")
        print("Set it in .env or export OPENROUTER_API_KEY='...'")
        return 1
    
    print("=" * 80)
    print(f"Re-running Archetype Grid for {MODEL}")
    print(f"Max tokens: {MAX_TOKENS}")
    print("=" * 80)
    
    # Load prompts
    prompts_path = get_priors_path("archetype_grid_prompts.jsonl")
    prompts = []
    with open(prompts_path) as f:
        for line in f:
            data = json.loads(line)
            prompts.append((data['cluster_id'], data['prompt']))
    print(f"Loaded {len(prompts)} prompts")
    
    # Load grader (optional - just for quality scoring)
    grader_path = get_quality_predictor_path()
    grader = None
    if grader_path.exists():
        print("Loading grader...")
        soft = QualityCostPredictor.load(grader_path)
        soft.eval()
        grader = TieredGrader(soft_grader=soft, teacher_verifier=None)
    else:
        print("Grader not found - will call API without grading")
    
    # Load existing log to see old results
    log_path = get_priors_path("archetype_grid_dense_run.jsonl")
    old_entries = {}
    if log_path.exists():
        with open(log_path) as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                if data.get('model_id') == MODEL:
                    old_entries[data['cluster_id']] = data
    print(f"Found {len(old_entries)} existing entries for {MODEL}")
    
    # Output file for new results
    output_path = get_priors_path("gemini3_rerun.jsonl")
    print(f"Output: {output_path}")
    
    def process_prompt(cluster_id, prompt):
        """Process a single prompt."""
        resp = call_openrouter(MODEL, prompt, max_tokens=MAX_TOKENS, timeout_s=120.0)
        ok = resp and not str(resp).startswith("[ERROR")
        
        reward_logit = None
        if ok and grader:
            prod = grader.predict_production(prompt, resp, reward_normalizer=None)
            reward_logit = prod.get("reward_logit", 0.0)
        
        return {
            "cluster_id": cluster_id,
            "model_id": MODEL,
            "ok": ok,
            "reward_logit": reward_logit,
            "response_len": len(str(resp)) if ok else 0,
            "ts": time.time(),
        }
    
    # Process all prompts
    results = []
    t0 = time.time()
    
    with open(output_path, "w") as fout:
        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            futures = {
                executor.submit(process_prompt, cid, prompt): cid
                for cid, prompt in prompts
            }
            
            for i, future in enumerate(as_completed(futures)):
                cid = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Write to file
                    fout.write(json.dumps(result) + "\n")
                    fout.flush()
                    
                    # Progress
                    elapsed = time.time() - t0
                    rate = (i + 1) / elapsed
                    eta = (len(prompts) - i - 1) / rate if rate > 0 else 0
                    
                    old_reward = old_entries.get(cid, {}).get('reward_logit')
                    new_reward = result.get('reward_logit')
                    
                    status = "OK" if result['ok'] else "FAIL"
                    print(f"[{i+1}/{len(prompts)}] cluster={cid} {status} "
                          f"len={result['response_len']} "
                          f"old_reward={old_reward:.2f if old_reward else 'N/A':>6} "
                          f"new_reward={new_reward:.2f if new_reward else 'N/A':>6} "
                          f"ETA={eta/60:.1f}m")
                    
                except Exception as e:
                    print(f"[{i+1}/{len(prompts)}] cluster={cid} ERROR: {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    ok_results = [r for r in results if r['ok']]
    print(f"Success rate: {len(ok_results)}/{len(results)} ({100*len(ok_results)/len(results):.1f}%)")
    
    new_rewards = [r['reward_logit'] for r in results if r.get('reward_logit') is not None]
    old_rewards = [old_entries[r['cluster_id']].get('reward_logit') 
                   for r in results 
                   if r['cluster_id'] in old_entries and old_entries[r['cluster_id']].get('reward_logit') is not None]
    
    if new_rewards:
        print(f"New mean reward: {sum(new_rewards)/len(new_rewards):.3f}")
    if old_rewards:
        print(f"Old mean reward: {sum(old_rewards)/len(old_rewards):.3f}")
    if new_rewards and old_rewards:
        diff = (sum(new_rewards)/len(new_rewards)) - (sum(old_rewards)/len(old_rewards))
        print(f"Improvement: {diff:+.3f}")
    
    print(f"\nResults saved to: {output_path}")
    print("\nNext steps:")
    print("1. Replace Gemini 3 entries in archetype_grid_dense_run.jsonl")
    print("2. Re-run: python experiments/generate_expert_priors.py")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
