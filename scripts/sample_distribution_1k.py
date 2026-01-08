#!/usr/bin/env python3
"""
Sample two sets of 1000 prompts (Train & Test) from LMSYS 1M with a 20/30/50 distribution.
Reuses existing rewards from test_rewards_hle_models.jsonl.gz and train_rewards_hle_models.jsonl.gz.

Targets per set (1000 prompts each):
- Frontier (20%): 200 (All New)
- Contentious (30%): 300 (Reused + New)
- Commodity (50%): 500 (Reused)
"""

import json
import random
import heapq
import gzip
import numpy as np
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

def calculate_frontier_score(text):
    """Scoring function to identify 'Frontier' candidates."""
    score = 0
    text_lower = text.lower()
    
    # 1. Length Penalty
    length = len(text)
    if length > 2000: score += 20
    if length > 4000: score += 10
        
    # 2. Constraint Density
    constraints = ["do not", "don't", "without", "exclude", "limit", "only use", "must be"]
    constraint_count = sum(1 for c in constraints if c in text_lower)
    score += (constraint_count * 5)
    
    # 3. Reasoning Complexity
    reasoning_keys = ["step-by-step", "chain of thought", "proof", "disprove", 
                      "logical fallacy", "paradox", "derive", "critical analysis"]
    if any(k in text_lower for k in reasoning_keys): score += 15
    
    # 4. Advanced Coding/Math
    advanced_terms = ["recursion", "dynamic programming", "optimization", "time complexity", 
                      "differential equation", "integral", "theorem", "lemma"]
    if any(k in text_lower for k in advanced_terms): score += 20
    
    # 5. "Gotcha" Detection
    if "trick" in text_lower or "riddle" in text_lower: score += 10
    
    return score

def get_existing_rewards(path: Path):
    """Load and categorize existing rewards from a path."""
    contentious = []
    commodity = []
    seen = set()
    keywords = ["code", "math", "translate", "summarize"]
    
    if not path.exists():
        print(f"⚠️ Warning: {path} not found.")
        return [], []

    print(f"📂 Loading existing rewards from {path.name}...")
    with gzip.open(path, 'rt') as f:
        for line in f:
            try:
                data = json.loads(line)
                p = data["prompt"]
                if p in seen: continue
                seen.add(p)
                
                text_lower = p.lower()
                if any(kw in text_lower for kw in keywords):
                    contentious.append(data)
                else:
                    commodity.append(data)
            except: continue
    return contentious, commodity

def main():
    print("🚀 Starting DUAL LMSYS 1M Weighted Sampling ($N=2000$ total)...")
    
    # Data Paths
    data_dir = Path('src/bandit_gpt/data/offline_dataset')
    test_rewards_path = data_dir / 'test_rewards_hle_models.jsonl.gz'
    train_rewards_path = data_dir / 'train_rewards_hle_models.jsonl.gz'
    
    # 1. Load Existing for both sets
    test_cont, test_comm = get_existing_rewards(test_rewards_path)
    train_cont, train_comm = get_existing_rewards(train_rewards_path)
    
    # 2. Base Distribution for TEST
    final_test = []
    # 500 Commodity (existing)
    test_comm_sample = random.sample(test_comm, min(500, len(test_comm)))
    for r in test_comm_sample:
        r['target_category'] = 'Commodity (Easy)'
        r['source'] = 'existing_test'
        final_test.append(r)
    # Up to 300 Contentious (existing)
    test_cont_sample = test_cont[:300]
    for r in test_cont_sample:
        r['target_category'] = 'Contentious'
        r['source'] = 'existing_test'
        final_test.append(r)
    test_cont_gap = 300 - len(test_cont_sample)
    
    # 3. Base Distribution for TRAIN
    final_train = []
    # 500 Commodity (existing)
    train_comm_sample = random.sample(train_comm, min(500, len(train_comm)))
    for r in train_comm_sample:
        r['target_category'] = 'Commodity (Easy)'
        r['source'] = 'existing_train'
        final_train.append(r)
    # Up to 300 Contentious (existing)
    train_cont_sample = train_cont[:300]
    for r in train_cont_sample:
        r['target_category'] = 'Contentious'
        r['source'] = 'existing_train'
        final_train.append(r)
    train_cont_gap = 300 - len(train_cont_sample)
    
    print(f"📊 Reusing: {len(test_comm_sample)}+{len(test_cont_sample)} for Test, {len(train_comm_sample)}+{len(train_cont_sample)} for Train")
    print(f"🎯 Gaps: Test-Contentious: {test_cont_gap}, Train-Contentious: {train_cont_gap}, Frontier: 200 each")

    # 4. Exclusions (Global)
    global_seen = set()
    for r in final_test + final_train: global_seen.add(r['prompt'])
    
    # Add other eval sets to exclusions if they exist
    for p in ['src/bandit_gpt/data/train_prompts.jsonl', 'src/bandit_gpt/data/test_prompts.jsonl']:
        path = Path(p)
        if path.exists():
            with open(path, 'r') as f:
                for line in f:
                    try: global_seen.add(json.loads(line)['prompt'])
                    except: pass

    # 5. Stream LMSYS 1M for Gaps
    print("\n📦 Streaming LMSYS 1M to fill gaps...")
    dataset = load_dataset("lmsys/lmsys-chat-1m", split="train")
    
    frontier_candidates = [] # Top pool for Frontier
    counter = 0
    max_frontier_heap = 50000 
    
    # Use dicts for pools to ensure prompt-level uniqueness from the start
    new_contentious_pool = {}
    new_commodity_pool = {}
    contentious_keywords = ["code", "math", "translate", "summarize"]

    for record in tqdm(dataset, desc="Scoring 1M"):
        if record.get('turn') != 1: continue
        conv = record.get('conversation', [])
        if not conv or conv[0].get('role') != 'user': continue
        
        prompt = conv[0].get('content', '')
        if not prompt or prompt in global_seen: continue
        
        # IMMEDIATELY mark as seen to prevent duplicates within the stream
        global_seen.add(prompt)
        
        score = calculate_frontier_score(prompt)
        text_lower = prompt.lower()
        
        # Frontier Pool (Top scoring)
        record['prompt'] = prompt
        if len(frontier_candidates) < max_frontier_heap:
            heapq.heappush(frontier_candidates, (score, counter, record))
            counter += 1
        elif score > frontier_candidates[0][0]:
            heapq.heapreplace(frontier_candidates, (score, counter, record))
            counter += 1
            
        # Contentious Pool
        is_contentious = any(kw in text_lower for kw in contentious_keywords)
        if is_contentious:
            if len(new_contentious_pool) < 20000:
                new_contentious_pool[prompt] = record
        else:
            # Commodity Pool (Everything else)
            if len(new_commodity_pool) < 20000:
                new_commodity_pool[prompt] = record

    # 6. Fill final sets
    print("\n🎯 Finalizing distributions...")
    
    # Track everything currently in final sets
    all_set_prompts = {r['prompt'] if 'prompt' in r else r['conversation'][0]['content'] for r in final_test + final_train}

    # A. Sample Frontier (400 total)
    frontier_data = [rec for _, _, rec in frontier_candidates]
    # Filter for anything that might have snuck in (should be 0)
    frontier_data = [r for r in frontier_data if r['prompt'] not in all_set_prompts]
    
    frontier_selection = random.sample(frontier_data, min(400, len(frontier_data)))
    for i, rec in enumerate(frontier_selection):
        p = rec['prompt']
        all_set_prompts.add(p)
        rec['target_category'] = 'Frontier (Hard)'
        rec['source'] = 'lmsys_1m_new'
        rec['frontier_score'] = calculate_frontier_score(p)
        if i < 200:
            final_test.append(rec)
        else:
            final_train.append(rec)
            
    # B. Sample Contentious Gaps
    # Filter contentious pool for current set contents
    contentious_list = [r for p, r in new_contentious_pool.items() if p not in all_set_prompts]
    
    test_cont_count = len([r for r in final_test if r['target_category'] == 'Contentious'])
    train_cont_count = len([r for r in final_train if r['target_category'] == 'Contentious'])
    test_cont_needed = max(0, 300 - test_cont_count)
    train_cont_needed = max(0, 300 - train_cont_count)
    
    new_cont_samples = random.sample(contentious_list, min(test_cont_needed + train_cont_needed, len(contentious_list)))
    for i, rec in enumerate(new_cont_samples):
        p = rec['prompt']
        all_set_prompts.add(p)
        rec['target_category'] = 'Contentious'
        rec['source'] = 'lmsys_1m_new'
        if i < test_cont_needed:
            final_test.append(rec)
        else:
            final_train.append(rec)

    # C. Final Gap Fill (Commodity) up to exactly 1000 each
    # Filter commodity pool
    commodity_list = [r for p, r in new_commodity_pool.items() if p not in all_set_prompts]
    random.shuffle(commodity_list) # ensures random distribution
    for label, dset in [('test', final_test), ('train', final_train)]:
        # use a set to count unique prompts in current dset
        unique_in_dset = {r['prompt'] if 'prompt' in r else r['conversation'][0]['content'] for r in dset}
        needed = 1000 - len(unique_in_dset)
        if needed > 0:
            print(f"  Filling {needed} commodity gaps for {label}...")
            gap_samples = []
            while len(gap_samples) < needed and commodity_list:
                rec = commodity_list.pop()
                p = rec['prompt']
                if p not in all_set_prompts:
                    all_set_prompts.add(p)
                    rec['target_category'] = 'Commodity (Easy)'
                    rec['source'] = 'lmsys_1m_new'
                    gap_samples.append(rec)
            dset.extend(gap_samples)

    # 7. Save results
    random.shuffle(final_test)
    random.shuffle(final_train)
    
    Path('data').mkdir(exist_ok=True)
    needs_rewards = []
    
    for label, dset in [('test', final_test), ('train', final_train)]:
        path = Path(f'data/lmsys_{label}_distribution_1k.jsonl')
        with open(path, 'w') as f:
            for r in dset[:1000]: # Exact limit
                clean_rec = {
                    "prompt": r.get('prompt') if 'prompt' in r else r['conversation'][0]['content'],
                    "target_category": r.get('target_category'),
                    "source": r.get('source')
                }
                if 'frontier_score' in r:
                    clean_rec['frontier_score'] = r['frontier_score']
                
                f.write(json.dumps(clean_rec) + '\n')
                if r.get('source') == 'lmsys_1m_new':
                    needs_rewards.append(clean_rec)
        print(f"✅ Saved {len(dset[:1000])} unique prompts to {path}")

    combined_needs_path = Path('data/lmsys_needs_rewards_combined.jsonl')
    with open(combined_needs_path, 'w') as f:
        unique_needs = []
        seen_needs = set()
        for r in needs_rewards:
            if r['prompt'] not in seen_needs:
                unique_needs.append(r)
                seen_needs.add(r['prompt'])
                
        for r in unique_needs:
            f.write(json.dumps(r) + '\n')
            
    print(f"\n🎉 All done!")
    print(f"💰 Total new rewards needed: {len(unique_needs)}")
    print(f"📁 Needs rewards list saved to: {combined_needs_path}")

if __name__ == "__main__":
    main()
