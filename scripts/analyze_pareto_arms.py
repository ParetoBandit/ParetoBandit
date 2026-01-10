import gzip
import json
import numpy as np
from collections import defaultdict
import glob
from dataclasses import dataclass
from typing import List, Dict, Tuple

@dataclass
class ModelStats:
    id: str
    display_name: str
    cost: float
    success_rate: float
    sample_count: int

def get_pareto_frontier(models: List[ModelStats]) -> List[ModelStats]:
    """
    Finds models on the Pareto frontier (maximizing success_rate for minimize cost).
    Sorts by cost (ascending). Keeps model if it has higher success_rate than all cheaper models.
    Actually, strictly speaking for convex hull in efficient bandit routing, we want the upper convex hull.
    But for simple filtering, just finding non-dominated points is a great start.
    
    A model M is dominated if there exists M' such that Cost(M') <= Cost(M) AND Quality(M') >= Quality(M)
    (and at least one strict inequality).
    """
    # Sort by cost ascending, then by quality descending (to keep best quality for same cost)
    sorted_models = sorted(models, key=lambda x: (x.cost, -x.success_rate))
    
    frontier = []
    max_quality_so_far = -1.0
    
    for model in sorted_models:
        # If this model is more expensive but doesn't improve quality, it's dominated (or equal).
        # We strictly want improvement to justify cost.
        if model.success_rate > max_quality_so_far:
            frontier.append(model)
            max_quality_so_far = model.success_rate
            
    return frontier

def main():
    # 1. Load Models Config
    print("Loading model config...")
    with open('src/bandit_gpt/config/models.json', 'r') as f:
        config = json.load(f)
    
    model_costs = {}
    model_names = {}
    for m in config['models']:
        # Use blended price or input price? Blended is safer proxy
        # If blended is 0 (some free models?), use small epsilon
        cost = m.get('price_1m_blended', 0.0)
        model_costs[m['openrouter_id']] = cost
        model_names[m['openrouter_id']] = m['display_name']

    # 2. Load Training Data
    print("Loading training data...")
    data_path = 'src/bandit_gpt/data/offline_dataset/lmsys_train_final_rewards_1k_clean.jsonl.gz'
    
    # Store rewards: prompt -> model -> score
    prompt_rewards = defaultdict(dict)
    
    try:
        with gzip.open(data_path, 'rb') as f:
            for line in f:
                rec = json.loads(line)
                pid = rec['prompt']
                mid = rec['model_id']
                # Use raw_score as the ground truth reward signal
                score = float(rec.get('raw_score', 0.0))
                prompt_rewards[pid][mid] = score
    except FileNotFoundError:
        print(f"Error: Could not find {data_path}")
        return

    print(f"Loaded {len(prompt_rewards)} unique prompts.")

    # 3. Stratify Prompts
    # Calculate average difficulty per prompt (across ALL models)
    prompt_difficulty = {}
    difficulty_values = []
    
    for pid, rewards in prompt_rewards.items():
        if not rewards: continue
        avg_score = sum(rewards.values()) / len(rewards)
        prompt_difficulty[pid] = avg_score
        difficulty_values.append(avg_score)
        
    # Histogram
    print("\nPrompt Difficulty Distribution (Avg Success Rate):")
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    hist, _ = np.histogram(difficulty_values, bins=bins)
    for i, count in enumerate(hist):
        range_str = f"[{bins[i]:.1f}, {bins[i+1]:.1f})"
        bar = "#" * (count // 10)
        print(f"{range_str:<12} : {count:>4} {bar}")

    # Failure Analysis
    print("\n" + "="*80)
    print("MODEL FAILURE ANALYSIS (Failures / Total Samples)")
    print("="*80)
    
    # Recalculate per-model stats globally
    global_failures = defaultdict(int)
    global_counts = defaultdict(int)
    
    for pid, rewards in prompt_rewards.items():
        for mid, score in rewards.items():
            if score < 1.0:
                global_failures[mid] += 1
            global_counts[mid] += 1
            
    # Print top failures
    sorted_failures = sorted(global_failures.items(), key=lambda x: x[1], reverse=True)
    for mid, count in sorted_failures:
        total = global_counts[mid]
        rate = (count / total) * 100
        print(f"{mid:<40} : {count:>4}/{total:<4} ({rate:.1f}% Fail)")

    # Check gpt-oss-20b specifically
    target = 'openai/gpt-oss-20b'
    print(f"\nSpecific check for {target}:")
    print(f"Failures: {global_failures[target]}/{global_counts[target]}")
    
    # Buckets (Refined based on perfection)
    print("\nRefining Buckets...")
    buckets = {'Imperfect': [], 'Perfect': [], 'Overall': []}
    
    for pid, score in prompt_difficulty.items():
        buckets['Overall'].append(pid)
        if score < 1.0:
            buckets['Imperfect'].append(pid)
        else:
            buckets['Perfect'].append(pid)

    print(f"Stratification: Perfect={len(buckets['Perfect'])}, Imperfect={len(buckets['Imperfect'])}")
    
    # 4. Analyze Each Bucket
    all_pareto_models = set()
    
    print("\n" + "="*80)
    print(f"{'BUCKET':<10} | {'MODEL':<40} | {'COST ($/1M)':<12} | {'SUCCESS %':<10}")
    print("="*80)

    for bucket_name, pids in buckets.items():
        if not pids: continue
        
        # Calculate aggregate stats for this bucket
        stats_map = defaultdict(list)
        for pid in pids:
            msg_rewards = prompt_rewards[pid]
            for mid, score in msg_rewards.items():
                stats_map[mid].append(score)
        
        # Convert to ModelStats
        candidates = []
        for mid, scores in stats_map.items():
            if mid not in model_costs: continue # Skip if not in config
            avg_acc = sum(scores) / len(scores)
            candidates.append(ModelStats(
                id=mid,
                display_name=model_names.get(mid, mid),
                cost=model_costs[mid],
                success_rate=avg_acc,
                sample_count=len(scores)
            ))
            
        # Get Frontier
        frontier = get_pareto_frontier(candidates)
        
        print(f"--- {bucket_name} ({len(pids)} prompts) ---")
        for m in frontier:
            print(f"{'':<10} | {m.id:<40} | ${m.cost:<11.4f} | {m.success_rate*100:.1f}%")
            if bucket_name == 'Overall': # Only add overall winners to final set? Or union?
                # Using Union of "Imperfect" and "Perfect" is better for coverage
                pass
            all_pareto_models.add(m.id)
        print("-" * 80)

    # 5. Summary
    print("\n" + "="*80)
    print("SCIENTIFICALLY RECOMMENDED PORTFOLIO (Union of Frontiers)")
    print("="*80)
    
    # Sort final list by cost
    final_list = []
    for mid in all_pareto_models:
        final_list.append((mid, model_costs.get(mid, 0)))
    
    final_list.sort(key=lambda x: x[1])
    
    for mid, cost in final_list:
        name = model_names.get(mid, mid)
        print(f"- {mid:<40} (${cost:.4f})  [{name}]")
        
    print(f"\nTotal: {len(final_list)} models")

if __name__ == "__main__":
    main()
